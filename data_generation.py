import numpy as np
from scipy.stats import wishart


def generate_data(
    n: int,
    p: int,
    q: int,
    rho1: float = 0.9,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate paired datasets X and Y with a known first canonical correlation.

    Construction:
    1. Sxx and Syy are drawn independently from the Wishart distribution.
    2. The first canonical directions u1 and v1 are fixed as random unit
       vectors in R^p and R^q respectively.
    3. The cross-covariance matrix is constructed as:
           Sxy = rho1 * Sxx^{1/2} u1 v1^T Syy^{1/2}
       which guarantees the first canonical correlation is exactly rho1.
    4. The joint covariance matrix S is assembled:
           S = [[Sxx, Sxy],
                [Syx, Syy]]
    5. n samples are drawn from N(0, S) and split into X and Y.

    Parameters
    ----------
    n : int
        Number of samples (rows of X and Y).
    p : int
        Number of features in X (columns).
    q : int
        Number of features in Y (columns).
    rho1 : float
        First canonical correlation to enforce. Must be in (0, 1).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray of shape (n, p)
        Centered data matrix for Alice.
    Y : np.ndarray of shape (n, q)
        Centered data matrix for Bob.
    """
    rng = np.random.default_rng(seed)

    if not 0 < rho1 < 1:
        raise ValueError(f"rho1 must be in (0, 1), got {rho1}")

    # ------------------------------------------------------------------
    # Step 1 — Draw Sxx and Syy from the Wishart distribution
    # Degrees of freedom = dimension + 1 ensures positive definiteness.
    # Scale matrix = I ensures the covariance is well conditioned.
    # ------------------------------------------------------------------
    Sxx = wishart.rvs(df=p + 1, scale=np.eye(p), random_state=rng)  # shape (p, p)
    Syy = wishart.rvs(df=q + 1, scale=np.eye(q), random_state=rng)  # shape (q, q)

    # Symmetrize for numerical safety
    Sxx = (Sxx + Sxx.T) / 2
    Syy = (Syy + Syy.T) / 2

    # ------------------------------------------------------------------
    # Step 2 — Compute Sxx^{1/2} and Syy^{1/2} via eigendecomposition
    # ------------------------------------------------------------------
    lam_x, Ux = np.linalg.eigh(Sxx)
    lam_x     = np.maximum(lam_x, 1e-12)
    Sxx_sqrt  = Ux @ np.diag(lam_x ** 0.5) @ Ux.T             # shape (p, p)
    Sxx_inv_sqrt = Ux @ np.diag(lam_x ** (-0.5)) @ Ux.T       # shape (p, p)

    lam_y, Uy = np.linalg.eigh(Syy)
    lam_y     = np.maximum(lam_y, 1e-12)
    Syy_sqrt  = Uy @ np.diag(lam_y ** 0.5) @ Uy.T             # shape (q, q)
    Syy_inv_sqrt = Uy @ np.diag(lam_y ** (-0.5)) @ Uy.T       # shape (q, q)

    # ------------------------------------------------------------------
    # Step 3 — Fix canonical directions u1 and v1
    # u1 and v1 are random unit vectors in the whitened spaces.
    # The true canonical directions in original space are:
    #     a_true = Sxx^{-1/2} u1,   b_true = Syy^{-1/2} v1
    # ------------------------------------------------------------------
    u1 = rng.standard_normal(p)
    u1 = u1 / np.linalg.norm(u1)                               # unit vector in R^p

    v1 = rng.standard_normal(q)
    v1 = v1 / np.linalg.norm(v1)                               # unit vector in R^q

    # ------------------------------------------------------------------
    # Step 4 — Construct cross-covariance matrix
    # Sxy = rho1 * Sxx^{1/2} u1 v1^T Syy^{1/2}
    # This guarantees the first canonical correlation is exactly rho1.
    # ------------------------------------------------------------------
    Sxy = rho1 * Sxx_sqrt @ np.outer(u1, v1) @ Syy_sqrt       # shape (p, q)
    Syx = Sxy.T                                                  # shape (q, p)

    # ------------------------------------------------------------------
    # Step 5 — Assemble joint covariance matrix S in R^{(p+q) x (p+q)}
    # S = [[Sxx, Sxy],
    #      [Syx, Syy]]
    # ------------------------------------------------------------------
    S = np.block([[Sxx, Sxy],
                  [Syx, Syy]])                                   # shape (p+q, p+q)

    # Symmetrize for numerical safety
    S = (S + S.T) / 2

    # Verify S is positive definite
    min_eig = float(np.linalg.eigvalsh(S).min())
    if min_eig <= 0:
        raise ValueError(
            f"Joint covariance matrix S is not positive definite "
            f"(min eigenvalue = {min_eig:.4e}). "
            f"Try a smaller rho1 or larger n."
        )

    # ------------------------------------------------------------------
    # Step 6 — Generate n samples from N(0, S) and split into X and Y
    # ------------------------------------------------------------------
    L   = np.linalg.cholesky(S)                                 # S = L L^T
    Z   = rng.standard_normal((n, p + q))                       # shape (n, p+q)
    A   = Z @ L.T                                               # shape (n, p+q)

    X   = A[:, :p]                                              # shape (n, p)
    Y   = A[:, p:]                                              # shape (n, q)

    # Center columns
    X   = X - X.mean(axis=0)
    Y   = Y - Y.mean(axis=0)


    return X, Y


if __name__ == "__main__":
    X, Y= generate_data(n=200, p=10, q=8, rho1=0.9, seed=42)

    Sxx_emp = X.T @ X / (X.shape[0] - 1)
    Syy_emp = Y.T @ Y / (Y.shape[0] - 1)
    Sxy_emp = X.T @ Y / (X.shape[0] - 1)

    print(f"X shape                          : {X.shape}")
    print(f"Y shape                          : {Y.shape}")

    print(f"X column means (expect ~0)       : {X.mean(axis=0).round(6)}")
    print(f"Y column means (expect ~0)       : {Y.mean(axis=0).round(6)}")
    print(f"Min eigenvalue of S (expect > 0) : {float(np.linalg.eigvalsh(np.block([[Sxx_emp, Sxy_emp],[Sxy_emp.T, Syy_emp]])).min()):.6f}")