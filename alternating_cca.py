import numpy as np
import math
from data_generation import generate_data
from ckks_primitives import CKKSSimulator


def alternating_cca(
    X: np.ndarray,
    Y: np.ndarray,
    sim: CKKSSimulator,
    lambda_reg: float = 1e-3,
    T_max: int = 50,
    n_newton: int = 3,
    n_goldschmidt: int = 3,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Encrypted two-sided power iteration for CCA in the whitened space.

    Implements the alternating two-sided power iteration on the whitened
    cross-covariance operator:

        W = Sxx^{-1/2} Sxy Syy^{-1/2}

    The iteration alternates between:

        u_tilde = W v^(t)         (ApplyW)
        u^(t+1) = u_tilde / ||u_tilde||

        v_tilde = W^T u^(t+1)    (ApplyWT)
        v^(t+1) = v_tilde / ||v_tilde||

    u and v live in the whitened space. The canonical directions a and b
    in the original feature space are recovered only at the end via the
    back-transformation:

        a = Sxx^{-1/2} u,   b = Syy^{-1/2} v

    Sxx^{-1/2} is computed by Alice locally from her plaintext X before
    encryption. Syy^{-1/2} is computed by Bob locally from his plaintext Y.
    The whitened cross-covariance W is constructed jointly then encrypted.

    All iteration counts are fixed plaintext hyperparameters. No runtime
    comparisons on encrypted data are performed. Since u and v are
    Euclidean-normalized at every step, ||u||^2 = 1 and k = 0 throughout
    after the first iteration.

    Parameters
    ----------
    X : np.ndarray of shape (n, p)
        Alice's centered data matrix.
    Y : np.ndarray of shape (n, q)
        Bob's centered data matrix.
    sim : CKKSSimulator
        CKKS primitive simulator.
    lambda_reg : float
        Regularization parameter for numerical stability in Sxx^{-1/2}
        and Syy^{-1/2} computation.
    T_max : int
        Number of alternating iterations. Fixed before encryption.
    n_newton : int
        Newton iterations for inverse square root normalization.
    n_goldschmidt : int
        Goldschmidt iterations for inverse square root normalization.

    Returns
    -------
    a1 : np.ndarray of shape (p,)
        Recovered first canonical direction for X in original space.
    b1 : np.ndarray of shape (q,)
        Recovered first canonical direction for Y in original space.
    rho1 : float
        Recovered first canonical correlation a1^T Sxy b1.
    """
    n, p = X.shape
    q    = Y.shape[1]

    # ------------------------------------------------------------------
    # Plaintext preprocessing — computed locally by each party
    # before any encryption begins.
    # ------------------------------------------------------------------

    # Alice computes Sxx^{-1/2} locally from her plaintext X
    Sxx = X.T @ X / (n - 1)
    lam_x, Ux = np.linalg.eigh(Sxx + lambda_reg * np.eye(p))
    lam_x = np.maximum(lam_x, 1e-12)
    Sxx_inv_sqrt = Ux @ np.diag(lam_x ** (-0.5)) @ Ux.T    # shape (p, p)

    # Bob computes Syy^{-1/2} locally from his plaintext Y
    Syy = Y.T @ Y / (n - 1)
    lam_y, Uy = np.linalg.eigh(Syy + lambda_reg * np.eye(q))
    lam_y = np.maximum(lam_y, 1e-12)
    Syy_inv_sqrt = Uy @ np.diag(lam_y ** (-0.5)) @ Uy.T    # shape (q, q)

    # Cross-covariance in original space (for final rho computation)
    Sxy = X.T @ Y / (n - 1)

    # Whitened cross-covariance W = Sxx^{-1/2} Sxy Syy^{-1/2}
    # Constructed jointly then passed through encoding noise
    W = Sxx_inv_sqrt @ Sxy @ Syy_inv_sqrt                   # shape (p, q)

    # ------------------------------------------------------------------
    # Plaintext preprocessing — k fixed before encryption begins.
    # After first iteration Euclidean normalization enforces ||u||=1
    # so ||u_tilde||^2 is bounded by the largest singular value of W
    # squared. For the first iteration we use the spectral norm of W.
    # After that k=0 since q=||u_tilde||^2 stays near 1.
    # ------------------------------------------------------------------
    sv_max = float(np.linalg.svd(W, compute_uv=False).max())
    k_init = math.ceil(math.log(sv_max ** 2, 4)) if sv_max > 1.0 else 0
    k_steady = 0

    # ------------------------------------------------------------------
    # Simulate encryption of W (Section 5)
    # ------------------------------------------------------------------
    W_enc  = sim.simulate_encoding(W)
    WT_enc = W_enc.T

    # ------------------------------------------------------------------
    # Initialization
    # Choose a random unit vector v^(0) in R^q and encrypt it.
    # ------------------------------------------------------------------
    rng   = np.random.default_rng(42)
    v_enc = rng.standard_normal(q)
    v_enc = v_enc / np.linalg.norm(v_enc)
    v_enc = sim.simulate_encoding(v_enc)

    u_enc = np.zeros(p)
    sim.reset_level()

    # ------------------------------------------------------------------
    # Main alternating loop — fixed T_max iterations, no convergence check
    # Euclidean normalization: ||u|| = 1, ||v|| = 1
    # No Sxx or Syy needed inside the loop
    # ------------------------------------------------------------------
    for t in range(T_max):

        k_use = k_init if t == 0 else k_steady

        # ApplyW: u_tilde = W v^(t)
        u_tilde = sim.encrypted_matmul(W_enc, v_enc)         # shape (p,)
        u_tilde = np.clip(u_tilde, -1e6, 1e6)

        # Normalize: u^(t+1) = u_tilde / ||u_tilde||
        inv_u = sim.approx_inverse_sqrt_newton(
            float(u_tilde @ u_tilde),
            k=k_use,
            n_newton=n_newton,
            n_goldschmidt=n_goldschmidt,
        )
        u_enc = u_tilde * inv_u                               # shape (p,)

        # ApplyWT: v_tilde = W^T u^(t+1)
        v_tilde = sim.encrypted_matmul(WT_enc, u_enc)        # shape (q,)
        v_tilde = np.clip(v_tilde, -1e6, 1e6)

        # Normalize: v^(t+1) = v_tilde / ||v_tilde||
        inv_v = sim.approx_inverse_sqrt_newton(
            float(v_tilde @ v_tilde),
            k=k_use,
            n_newton=n_newton,
            n_goldschmidt=n_goldschmidt,
        )
        v_enc = v_tilde * inv_v                               # shape (q,)

    print(f"[alternating_cca] Completed {T_max} iterations.")

    # ------------------------------------------------------------------
    # Back-transformation to original feature space (plaintext step)
    # a = Sxx^{-1/2} u,   b = Syy^{-1/2} v
    # ------------------------------------------------------------------
    a1 = Sxx_inv_sqrt @ u_enc                                # shape (p,)
    b1 = Syy_inv_sqrt @ v_enc                                # shape (q,)

    # Canonical correlation in original space
    rho1 = float(a1 @ Sxy @ b1)

    return a1, b1, rho1


if __name__ == "__main__":
    X, Y = generate_data(n=200, p=10, q=8, seed=42)
    sim  = CKKSSimulator(scale_bits=40, max_levels=10000, seed=0)

    a1, b1, rho1 = alternating_cca(X, Y, sim)

    Sxx = X.T @ X / (X.shape[0] - 1)
    Syy = Y.T @ Y / (Y.shape[0] - 1)

    print(f"a1 shape : {a1.shape}")
    print(f"b1 shape : {b1.shape}")
    print(f"Canonical correlation (rho1)     : {rho1:.6f}")
    print(f"a^T Sxx a (expect ~1.0)          : {(a1 @ Sxx @ a1):.6f}")
    print(f"b^T Syy b (expect ~1.0)          : {(b1 @ Syy @ b1):.6f}")
    print(f"Multiplicative levels consumed   : {sim.level}")