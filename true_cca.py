import numpy as np
import scipy.linalg
from data_generation import generate_data


def true_cca(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute the first canonical directions and canonical correlation
    using classic CCA via direct generalized eigendecomposition.

    Solves M a = rho^2 a where M = Sxx^{-1} Sxy Syy^{-1} Syx.
    The eigenvector corresponding to the largest eigenvalue is a_true.
    b_true is then recovered from b ∝ Syy^{-1} Syx a.

    Parameters
    ----------
    X : np.ndarray of shape (n, p)
        Centered data matrix for Alice.
    Y : np.ndarray of shape (n, q)
        Centered data matrix for Bob.

    Returns
    -------
    a_true : np.ndarray of shape (p,)
        First canonical direction for X, normalized so a^T Sxx a = 1.
    b_true : np.ndarray of shape (q,)
        First canonical direction for Y, normalized so b^T Syy b = 1.
    rho_true : float
        First canonical correlation, i.e. a^T Sxy b.
    """
    n = X.shape[0]

    # Compute covariance and cross-covariance matrices
    Sxx = X.T @ X / (n - 1)   # shape (p, p)
    Syy = Y.T @ Y / (n - 1)   # shape (q, q)
    Sxy = X.T @ Y / (n - 1)   # shape (p, q)
    Syx = Sxy.T                # shape (q, p)

    # Form M = Sxx^{-1} Sxy Syy^{-1} Syx
    # Use scipy.linalg.solve instead of explicit inversion for numerical stability
    # Sxx^{-1} Sxy  →  solve Sxx @ Z = Sxy for Z
    SxxInv_Sxy = scipy.linalg.solve(Sxx, Sxy)          # shape (p, q)
    # Syy^{-1} Syx  →  solve Syy @ W = Syx for W
    SyyInv_Syx = scipy.linalg.solve(Syy, Syx)          # shape (q, p)

    M = SxxInv_Sxy @ SyyInv_Syx                         # shape (p, p)

    # Eigendecomposition of M — eigenvalues are rho^2
    eigenvalues, eigenvectors = scipy.linalg.eig(M)

    # Keep only real parts (imaginary parts are numerical noise)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real

    # First canonical direction corresponds to the largest eigenvalue
    idx = np.argmax(eigenvalues)
    a_true = eigenvectors[:, idx]                        # shape (p,)

    # Normalize so that a^T Sxx a = 1
    scale_a = np.sqrt(a_true @ Sxx @ a_true)
    a_true = a_true / scale_a

    # Recover b from b ∝ Syy^{-1} Syx a  (Section 6.1 of the paper)
    b_raw = SyyInv_Syx @ a_true                          # shape (q,)

    # Normalize so that b^T Syy b = 1
    scale_b = np.sqrt(b_raw @ Syy @ b_raw)
    b_true = b_raw / scale_b

    # Canonical correlation rho = a^T Sxy b
    rho_true = float(a_true @ Sxy @ b_true)

    return a_true, b_true, rho_true


if __name__ == "__main__":
    X, Y = generate_data(n=200, p=10, q=8, seed=42)

    a_true, b_true, rho_true = true_cca(X, Y)

    print(f"a_true shape : {a_true.shape}")
    print(f"b_true shape : {b_true.shape}")
    print(f"First canonical correlation (rho_true) : {rho_true:.6f}")
    print(f"a^T Sxx a (should be 1.0)             : {(a_true @ (X.T @ X / (X.shape[0]-1)) @ a_true):.10f}")
    print(f"b^T Syy b (should be 1.0)             : {(b_true @ (Y.T @ Y / (Y.shape[0]-1)) @ b_true):.10f}")
    print(f"a_true : {a_true.round(6)}")
    print(f"b_true : {b_true.round(6)}")