import numpy as np
import math
from data_generation import generate_data
from ckks_primitives import CKKSSimulator


def pca_cca(
    X: np.ndarray,
    Y: np.ndarray,
    sim: CKKSSimulator,
    lambda_reg: float = 1e-3,
    T_max: int = 50,
    n_newton: int = 3,
    n_goldschmidt: int = 3,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Encrypted CCA over full PCA components (Section 10, Case 2 of the paper).

    Each party independently whitens their data via full PCA. The
    cross-covariance in PCA space S_ZxZy is constructed jointly under
    simulated CKKS. Because the covariance of Zx and Zy is identity,
    the CCA problem simplifies to max c^T S_ZxZy d subject to
    ||c|| = 1, ||d|| = 1, with no S^{-1} action needed.

    All iteration counts are fixed plaintext hyperparameters.
    No runtime comparisons on encrypted data are performed.

    Since Zx and Zy are whitened, the singular values of S_ZxZy are
    bounded by 1 by construction. Therefore q = ||w||^2 is in [0, 1]
    at every iteration and k = 0 throughout, with no runtime comparison
    needed to determine k.

    Parameters
    ----------
    X : np.ndarray of shape (n, p)
        Alice's centered data matrix.
    Y : np.ndarray of shape (n, q)
        Bob's centered data matrix.
    sim : CKKSSimulator
        CKKS primitive simulator.
    lambda_reg : float
        Regularization parameter (used for numerical safety in PCA).
    T_max : int
        Number of alternating iterations. Fixed before encryption.
    n_newton : int
        Newton iterations for inverse square root normalization.
    n_goldschmidt : int
        Goldschmidt iterations for inverse square root normalization.

    Returns
    -------
    a1 : np.ndarray of shape (p,)
        Recovered first canonical direction for X.
    b1 : np.ndarray of shape (q,)
        Recovered first canonical direction for Y.
    rho1 : float
        Recovered first canonical correlation a1^T Sxy b1.
    """
    n, p = X.shape
    q    = Y.shape[1]

    # ------------------------------------------------------------------
    # Local PCA — each party computes independently in plaintext
    # (Section 10, Case 2)
    # ------------------------------------------------------------------

    # Alice's local PCA
    Sxx = X.T @ X / (n - 1)
    lam_x, Ux = np.linalg.eigh(Sxx)
    lam_x = lam_x[::-1]
    Ux    = Ux[:, ::-1]
    lam_x = np.maximum(lam_x, 1e-12)
    Lam_x_inv_sqrt = np.diag(lam_x ** (-0.5))
    Zx = X @ Ux @ Lam_x_inv_sqrt                             # shape (n, p)

    # Bob's local PCA
    Syy = Y.T @ Y / (n - 1)
    lam_y, Uy = np.linalg.eigh(Syy)
    lam_y = lam_y[::-1]
    Uy    = Uy[:, ::-1]
    lam_y = np.maximum(lam_y, 1e-12)
    Lam_y_inv_sqrt = np.diag(lam_y ** (-0.5))
    Zy = Y @ Uy @ Lam_y_inv_sqrt                             # shape (n, q)

    # ------------------------------------------------------------------
    # Original space cross-covariance (for final rho computation)
    # ------------------------------------------------------------------
    Sxy = X.T @ Y / (n - 1)

    # ------------------------------------------------------------------
    # Joint cross-covariance in PCA space
    # Constructed jointly then passed through encoding noise.
    # ------------------------------------------------------------------
    S_ZxZy = Zx.T @ Zy / (n - 1)                            # shape (p, q)
    S_ZxZy_enc = sim.simulate_encoding(S_ZxZy)
    S_ZyZx_enc = S_ZxZy_enc.T

    # ------------------------------------------------------------------
    # Plaintext preprocessing — k fixed before encryption begins.
    # Since Zx and Zy are whitened, singular values of S_ZxZy are <= 1
    # by construction, so q = ||w||^2 is in [0, 1] at every iteration.
    # Therefore k = 0 throughout with no runtime comparison needed.
    # ------------------------------------------------------------------
    k_c = 0
    k_steady = 0

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    rng   = np.random.default_rng(42)
    d_enc = rng.standard_normal(q)
    d_enc = d_enc / np.linalg.norm(d_enc)
    d_enc = sim.simulate_encoding(d_enc)

    c_enc = np.zeros(p)
    sim.reset_level()

    # ------------------------------------------------------------------
    # Simplified alternating iteration in PCA space
    # Fixed T_max iterations — no convergence check.
    # Since SZxZx = I_p and SZyZy = I_q, no inverse action needed.
    # Normalization is Euclidean: ||c|| = 1, ||d|| = 1.
    # ------------------------------------------------------------------
    for t in range(T_max):

        # C-side update: w_c = S_ZxZy d^(t)
        w_c = sim.encrypted_matmul(S_ZxZy_enc, d_enc)       # shape (p,)
        w_c = np.clip(w_c, -1e6, 1e6)

        # Normalize: c^(t+1) = w_c / ||w_c||
        # k = 0 since ||w_c||^2 in [0, 1] by whitening construction
        k_use = k_c if t == 0 else k_steady
        inv_c = sim.approx_inverse_sqrt_newton(
            float(w_c @ w_c),
            k=k_use,
            n_newton=n_newton,
            n_goldschmidt=n_goldschmidt,
        )
        c_enc = w_c * inv_c                                   # shape (p,)

        # D-side update: w_d = S_ZyZx c^(t+1)
        w_d = sim.encrypted_matmul(S_ZyZx_enc, c_enc)       # shape (q,)
        w_d = np.clip(w_d, -1e6, 1e6)

        # Normalize: d^(t+1) = w_d / ||w_d||
        inv_d = sim.approx_inverse_sqrt_newton(
            float(w_d @ w_d),
            k=k_use,
            n_newton=n_newton,
            n_goldschmidt=n_goldschmidt,
        )
        d_enc = w_d * inv_d                                   # shape (q,)

    print(f"[pca_cca] Completed {T_max} iterations.")

    # ------------------------------------------------------------------
    # Map back to original feature spaces (Section 10, reversion step)
    # a = Ux Lam_x^{-1/2} c,  b = Uy Lam_y^{-1/2} d
    # ------------------------------------------------------------------
    c1 = c_enc
    d1 = d_enc

    a1 = Ux @ Lam_x_inv_sqrt @ c1                            # shape (p,)
    b1 = Uy @ Lam_y_inv_sqrt @ d1                            # shape (q,)

    rho1 = float(a1 @ Sxy @ b1)

    return a1, b1, rho1


if __name__ == "__main__":
    X, Y = generate_data(n=200, p=10, q=8, seed=42)
    sim  = CKKSSimulator(scale_bits=40, max_levels=10000, seed=0)

    a1, b1, rho1 = pca_cca(X, Y, sim)

    Sxx = X.T @ X / (X.shape[0] - 1)
    Syy = Y.T @ Y / (Y.shape[0] - 1)

    print(f"a1 shape : {a1.shape}")
    print(f"b1 shape : {b1.shape}")
    print(f"Canonical correlation (rho1)     : {rho1:.6f}")
    print(f"a^T Sxx a (expect ~1.0)          : {(a1 @ Sxx @ a1):.6f}")
    print(f"b^T Syy b (expect ~1.0)          : {(b1 @ Syy @ b1):.6f}")
    print(f"Multiplicative levels consumed   : {sim.level}")