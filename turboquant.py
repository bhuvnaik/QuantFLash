"""
TurboQuant Reference Implementation
Matches Algorithm 1 (TurboQuantmse) and Algorithm 2 (TurboQuantprod)
from: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
arXiv:2504.19874v1

All variable names map directly to paper notation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm as scipy_norm
from typing import Tuple


# ---------------------------------------------------------------------------
# Codebook construction — solves the 1D k-means problem in Eq. (4)
# ---------------------------------------------------------------------------

def solve_lloyd_max(d: int, b: int, n_iter: int = 2000) -> np.ndarray:
    """
    Solve the continuous 1D k-means problem (Eq. 4) via Lloyd-Max iteration.
    Returns 2^b centroids c_1 < c_2 < ... < c_{2^b}.

    For large d, coordinate distribution f_X -> N(0, 1/d).
    Uses Gaussian approximation which is accurate for d >> 1.
    """
    k   = 2 ** b
    std = 1.0 / np.sqrt(d)
    pdf = scipy_norm(0, std)

    # Initialize centroids at quantiles of N(0, 1/d)
    quantile_points = np.linspace(0.5 / k, 1 - 0.5 / k, k)
    centroids       = pdf.ppf(quantile_points)

    for _ in range(n_iter):
        # Voronoi boundaries: midpoints between consecutive centroids
        boundaries = np.concatenate([
            [-np.inf],
            (centroids[:-1] + centroids[1:]) / 2,
            [np.inf]
        ])

        new_centroids = np.zeros(k)
        for i in range(k):
            lo, hi       = boundaries[i], boundaries[i + 1]
            p_interval   = pdf.cdf(hi) - pdf.cdf(lo)
            if p_interval < 1e-12:
                new_centroids[i] = centroids[i]
                continue
            # Centroid condition: E[X | X in [lo, hi]]
            # For Gaussian: E[X * 1_{lo<=X<=hi}] = std^2 * (f(lo) - f(hi))
            ex               = (std ** 2) * (pdf.pdf(lo) - pdf.pdf(hi)) / p_interval
            new_centroids[i] = ex

        if np.max(np.abs(new_centroids - centroids)) < 1e-12:
            break
        centroids = new_centroids

    return centroids.astype(np.float32)


# ---------------------------------------------------------------------------
# Algorithm 1: TurboQuant_mse
# ---------------------------------------------------------------------------

class TurboQuantMSE:
    """
    MSE-optimal vector quantizer. Algorithm 1 from the paper.

    Theorem 1: For x in S^{d-1}, b bits per coordinate:
        D_mse <= sqrt(3*pi)/2 * 4^{-b}
    """

    def __init__(self, d: int, b: int, device: torch.device):
        self.d  = d
        self.b  = b
        self.k  = 2 ** b
        self.device = device

        # Line 2: random rotation Pi via QR of random Gaussian matrix
        G        = torch.randn(d, d, device=device)
        Q, _     = torch.linalg.qr(G)
        self.Pi  = Q                            # (d, d), orthogonal

        # Line 3: solve Eq.(4) — Lloyd-Max codebook for Gaussian marginal
        self.centroids = torch.tensor(
            solve_lloyd_max(d, b), dtype=torch.float32, device=device
        )                                       # (2^b,)

    def quant(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lines 4-7: Quantmse(x) -> idx in [2^b]^d
        x: (n, d)
        """
        y     = x @ self.Pi.T                   # (n, d)
        dists = (y.unsqueeze(-1) - self.centroids).abs()  # (n, d, 2^b)
        return dists.argmin(dim=-1)             # (n, d)

    def dequant(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Lines 8-11: DeQuantmse(idx) -> x_tilde in R^d
        idx: (n, d)
        """
        y_tilde = self.centroids[idx]           # (n, d)
        return y_tilde @ self.Pi                # (n, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequant(self.quant(x))


# ---------------------------------------------------------------------------
# QJL: 1-bit inner product quantizer (Definition 1)
# ---------------------------------------------------------------------------

class QJL:
    """
    Quantized Johnson-Lindenstrauss transform. Definition 1 from the paper.

    For x in S^{d-1}:
        Q_qjl(x)      = sign(S * x)               S in R^{d x d}, S_ij ~ N(0,1)
        Q^{-1}_qjl(z) = sqrt(pi/2)/d * S^T * z

    Lemma 4 guarantees:
        E[<y, Q^{-1}_qjl(Q_qjl(x))>] = <y, x>           (unbiased)
        Var <= pi/(2d) * ||y||^2
    """

    def __init__(self, d: int, device: torch.device):
        self.d = d
        self.S = torch.randn(d, d, device=device)   # (d, d)

    def quant(self, x: torch.Tensor) -> torch.Tensor:
        """x: (n, d) unit-norm -> z: (n, d) in {-1, +1}"""
        z = torch.sign(x @ self.S.T)
        return torch.where(z == 0, torch.ones_like(z), z)

    def dequant(self, z: torch.Tensor) -> torch.Tensor:
        """z: (n, d) -> (n, d) estimate of original unit-norm vector"""
        return (np.sqrt(np.pi / 2) / self.d) * (z @ self.S)


# ---------------------------------------------------------------------------
# Algorithm 2: TurboQuant_prod
# ---------------------------------------------------------------------------

class TurboQuantProd:
    """
    Inner-product optimal vector quantizer. Algorithm 2 from the paper.

    Two-stage pipeline:
      Stage 1: TurboQuantMSE with b-1 bits  -> minimizes ||residual||_2
      Stage 2: QJL on *unit-norm* residual   -> unbiased IP correction

    Critical fix vs naive implementation:
      QJL's Lemma 4 unbiasedness requires the input to be on S^{d-1}.
      So we normalize r before QJL and store gamma = ||r||_2 separately.
      Reconstruction: x_tilde = x_tilde_mse + gamma * Q^{-1}_qjl(sign(S * r/||r||))

    Theorem 2:
        E[<y, x_tilde>] = <y, x>                           (unbiased)
        D_prod <= sqrt(3pi)/2 * ||y||^2/d * 4^{-b}
    """

    def __init__(self, d: int, b: int, device: torch.device):
        self.d    = d
        self.b    = b
        self.qmse = TurboQuantMSE(d, max(1, b - 1), device)
        self.qjl  = QJL(d, device)

    def quant(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (n, d) unit-norm
        Returns:
            idx:   (n, d)  MSE codebook indices
            z:     (n, d)  QJL bits in {-1, +1}
            gamma: (n, 1)  ||residual||_2
        """
        idx    = self.qmse.quant(x)
        r      = x - self.qmse.dequant(idx)            # (n, d) residual
        gamma  = torch.norm(r, dim=-1, keepdim=True)   # (n, 1)
        r_unit = F.normalize(r, dim=-1)                # (n, d) unit norm
        z      = self.qjl.quant(r_unit)                # (n, d)
        return idx, z, gamma

    def dequant(
        self,
        idx:   torch.Tensor,
        z:     torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruction:
            x_tilde = x_tilde_mse + gamma * Q^{-1}_qjl(z)
        where Q^{-1}_qjl(z) estimates r/||r||, so gamma * Q^{-1}_qjl(z) estimates r.
        """
        x_tilde_mse = self.qmse.dequant(idx)           # (n, d)
        x_tilde_qjl = gamma * self.qjl.dequant(z)      # (n, d)
        return x_tilde_mse + x_tilde_qjl

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx, z, gamma = self.quant(x)
        return self.dequant(idx, z, gamma)


# ---------------------------------------------------------------------------
# Distortion metrics  (Eqs. 1 and 2)
# ---------------------------------------------------------------------------

def mse_distortion(x: torch.Tensor, x_tilde: torch.Tensor) -> float:
    """D_mse = E[||x - x_tilde||^2_2]"""
    return (x - x_tilde).pow(2).sum(dim=-1).mean().item()


def inner_product_distortion(
    x:     torch.Tensor,
    x_tilde: torch.Tensor,
    y:     torch.Tensor,
) -> Tuple[float, float]:
    """
    D_prod = E[(<y,x> - <y,x_tilde>)^2]
    bias   = E[<y,x_tilde> - <y,x>]
    """
    ip_true = (y * x).sum(dim=-1)
    ip_est  = (y * x_tilde).sum(dim=-1)
    error   = ip_true - ip_est
    return error.pow(2).mean().item(), (ip_est - ip_true).mean().item()


# ---------------------------------------------------------------------------
# Theoretical bounds
# ---------------------------------------------------------------------------

def mse_upper(b: int) -> float:
    return (np.sqrt(3 * np.pi) / 2) * (4 ** -b)

def mse_lower(b: int) -> float:
    return 4 ** -b

def prod_upper(b: int, d: int, y_norm_sq: float) -> float:
    return (np.sqrt(3 * np.pi) / 2) * (y_norm_sq / d) * (4 ** -b)

def prod_lower(b: int, d: int) -> float:
    return (1 / d) * (4 ** -b)


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda")

    d, n = 512, 2000
    print(f"TurboQuant Reference Implementation")
    print(f"d={d}, n={n}, device={device}")
    print("=" * 64)

    # Unit-norm vectors on S^{d-1}
    x = F.normalize(torch.randn(n, d, device=device), dim=-1)

    # Fixed query vector y (not unit norm)
    torch.manual_seed(7)
    y     = torch.randn(d, device=device)
    y_nsq = y.pow(2).sum().item()
    Y     = y.unsqueeze(0).expand(n, -1)

    for b in [1, 2, 3, 4]:
        print(f"\n--- b={b} bits ---")

        # TurboQuant_mse
        qmse       = TurboQuantMSE(d, b, device)
        x_hat_mse  = qmse.forward(x)
        d_mse      = mse_distortion(x, x_hat_mse)
        d_ip, bias = inner_product_distortion(x, x_hat_mse, Y)

        print(f"[MSE quantizer]")
        print(f"  D_mse:         {d_mse:.6f}"
              f"  (bounds [{mse_lower(b):.6f}, {mse_upper(b):.6f}])")
        print(f"  IP distortion: {d_ip:.6f}")
        print(f"  IP bias:       {bias:+.6f}  <- should be nonzero")

        # TurboQuant_prod
        qprod      = TurboQuantProd(d, b, device)
        x_hat_prod = qprod.forward(x)
        d_ip_p, bias_p = inner_product_distortion(x, x_hat_prod, Y)

        print(f"[Prod quantizer]")
        print(f"  IP distortion: {d_ip_p:.6f}"
              f"  (bounds [{prod_lower(b,d):.6f}, {prod_upper(b,d,y_nsq):.6f}])")
        print(f"  IP bias:       {bias_p:+.6f}  <- should be ~0")

    print("\n" + "=" * 64)
    print("Done.")