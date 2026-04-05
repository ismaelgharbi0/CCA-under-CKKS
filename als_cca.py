import numpy as np
import math
from data_generation import generate_data
from ckks_primitives import CKKSSimulator


def als_cca(
    X: np.ndarray,
    Y: np.ndarray,
    sim: CKKSSimulator,
    lambda_reg: float = 1e-3,
    n_iters_inv: int = 20,
    T_max: int = 50,
    n_newton: int = 3,
    n_goldschmidt: int = 3,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Encrypted single-vector alternating CCA via direct covariance iteration.

    Directly iterates the canonical direction vectors a and b in the
    original feature space. At each step the inverse action of the
    regularized covariance matrix is approximated via Richardson iteration,
    and normalization enforces the CCA constraint a^T Sxx a = 1.

    Updates:
        w_x^(t)   = (1/n-1) X^T Y b^(t)
        u^(t+1)   = (X^TX/n-1 + lambda I)^{-1} w_x^(t)
        a^(t+1)   = u^(t+1) / sqrt(u^(t+1)^T Sxx u^(t+1))

        w_y^(t+1) = (1/n-1) Y^T X a^(t+1)
        v^(t+1)   = (Y^TY/n-1 + lambda I)^{-1} w_y^(t+1)
        b^(t+1)   = v^(t+1) / sqrt(v^(t+1)^T Syy v^(t+1))

    All iteration counts are fixed plaintext hyperparameters. No runtime
    comparisons on encrypted data are performed. k_x and k_y for the
    inverse square root normalization are computed by Alice and Bob
    respectively from their local plaintext covariance matrices before
    encryption begins.

    Parameters
    ----------
    X : np.ndarray of shape (n, p)
        Alice's centered data matrix.
    Y : np.ndarray of shape (n, q)
        Bob's centered data matrix.
    sim : CKKSSimulator
        CKKS primitive simulator.
    lambda_reg : float
        Regularization parameter for S_tilde = S + lambda * I.
    n_iters_inv : int
        Number of Richardson iterations for the inverse action.
    T_max : int
        Number of outer alternating iterations. Fixed before encryption.
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
    # Compute covariance and cross-covariance matrices
    # ------------------------------------------------------------------
    Sxx = X.T @ X / (n - 1)
    Syy = Y.T @ Y / (n - 1)
    Sxy = X.T @ Y / (n - 1)
    Syx = Sxy.T

    # ------------------------------------------------------------------
    # Plaintext preprocessing — computed locally by each party
    # before any encryption begins.
    # Alice computes k_x from her local Sxx.
    # Bob computes k_y from his local Syy.
    # k is the integer such that q / 4^k is in [0.25, 1] where
    # q = v^T Sxx v. After normalization q ~ 1 so k_steady = 0.
    # ------------------------------------------------------------------
    lambda_max_xx = float(np.linalg.eigvalsh(Sxx).max())
    lambda_max_yy = float(np.linalg.eigvalsh(Syy).max())
    k_x_init = math.ceil(math.log(lambda_max_xx, 4)) if lambda_max_xx > 1.0 else 0
    k_y_init = math.ceil(math.log(lambda_max_yy, 4)) if lambda_max_yy > 1.0 else 0
    k_steady = 0

    # ------------------------------------------------------------------
    # Simulate encryption of the covariance matrices (Section 5)
    # ------------------------------------------------------------------
    Sxx_enc = sim.simulate_encoding(Sxx)
    Syy_enc = sim.simulate_encoding(Syy)
    Sxy_enc = sim.simulate_encoding(Sxy)
    Syx_enc = Sxy_enc.T

    # ------------------------------------------------------------------
    # Initialization
    # Choose a random unit vector b^(0) in R^q and encrypt it.
    # ------------------------------------------------------------------
    rng   = np.random.default_rng(42)
    b_enc = rng.standard_normal(q)
    b_enc = b_enc / np.linalg.norm(b_enc)
    b_enc = sim.simulate_encoding(b_enc)

    a_enc = np.zeros(p)
    sim.reset_level()

    # ------------------------------------------------------------------
    # Main alternating loop — fixed T_max iterations, no convergence check
    # ------------------------------------------------------------------
    for t in range(T_max):

        k_x = k_x_init if t == 0 else k_steady
        k_y = k_y_init if t == 0 else k_steady

        # --------------------------------------------------------------
        # A-side update
        # w_x = (1/n-1) X^T Y b^(t) = Sxy b^(t)
        # u   = S_tilde_xx^{-1} w_x  (Richardson iteration)
        # --------------------------------------------------------------
        w_x = sim.encrypted_matmul(Sxy_enc, b_enc)
        w_x = np.clip(w_x, -1e6, 1e6)
        u   = sim.approx_inverse_action(
            Sxx_enc, w_x,
            lambda_reg=lambda_reg,
            n_iters=n_iters_inv,
        )
        u = np.clip(u, -1e6, 1e6)

        # Normalize: a^(t+1) = u / sqrt(u^T Sxx u)
        a_enc = sim.normalize_sxx(
            u, Sxx_enc,
            k=k_x,
            n_newton=n_newton,
            n_goldschmidt=n_goldschmidt,
        )

        # --------------------------------------------------------------
        # B-side update
        # w_y = (1/n-1) Y^T X a^(t+1) = Syx a^(t+1)
        # v   = S_tilde_yy^{-1} w_y  (Richardson iteration)
        # --------------------------------------------------------------
        w_y = sim.encrypted_matmul(Syx_enc, a_enc)
        w_y = np.clip(w_y, -1e6, 1e6)
        v   = sim.approx_inverse_action(
            Syy_enc, w_y,
            lambda_reg=lambda_reg,
            n_iters=n_iters_inv,
        )
        v = np.clip(v, -1e6, 1e6)

        # Normalize: b^(t+1) = v / sqrt(v^T Syy v)
        b_enc = sim.normalize_sxx(
            v, Syy_enc,
            k=k_y,
            n_newton=n_newton,
            n_goldschmidt=n_goldschmidt,
        )

    print(f"[als_cca] Completed {T_max} iterations.")

    # ------------------------------------------------------------------
    # Output — a and b are recovered directly, no back-transformation
    # ------------------------------------------------------------------
    a1   = a_enc
    b1   = b_enc
    rho1 = float(a1 @ Sxy @ b1)

    return a1, b1, rho1


if __name__ == "__main__":
    X, Y = generate_data(n=200, p=10, q=8, seed=42)
    sim  = CKKSSimulator(scale_bits=40, max_levels=10000, seed=0)

    a1, b1, rho1 = als_cca(X, Y, sim)

    Sxx = X.T @ X / (X.shape[0] - 1)
    Syy = Y.T @ Y / (Y.shape[0] - 1)

    print(f"a1 shape : {a1.shape}")
    print(f"b1 shape : {b1.shape}")
    print(f"Canonical correlation (rho1)     : {rho1:.6f}")
    print(f"a^T Sxx a (expect ~1.0)          : {(a1 @ Sxx @ a1):.6f}")
    print(f"b^T Syy b (expect ~1.0)          : {(b1 @ Syy @ b1):.6f}")
    print(f"Multiplicative levels consumed   : {sim.level}")