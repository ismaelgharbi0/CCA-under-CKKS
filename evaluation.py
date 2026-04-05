import numpy as np
from data_generation import generate_data
from true_cca import true_cca
from ckks_primitives import CKKSSimulator
from alternating_cca import alternating_cca
from als_cca import als_cca
from pca_cca import pca_cca


def relative_error(v_hat: np.ndarray, v_true: np.ndarray) -> float:
    """
    Compute the relative error between a recovered vector and the true vector,
    accounting for sign ambiguity.

    Canonical directions are unique only up to a global sign flip, so we take
    the minimum relative error over both signs:
        RE = min(||v_hat - v_true||, ||v_hat + v_true||) / ||v_true||

    Parameters
    ----------
    v_hat : np.ndarray of shape (d,)
        Recovered canonical direction.
    v_true : np.ndarray of shape (d,)
        True canonical direction.

    Returns
    -------
    float
        Relative error in [0, inf).
    """
    norm_true = np.linalg.norm(v_true)
    if norm_true < 1e-12:
        raise ValueError("v_true has near-zero norm — cannot compute relative error.")
    re_pos = np.linalg.norm(v_hat - v_true) / norm_true
    re_neg = np.linalg.norm(v_hat + v_true) / norm_true
    return float(min(re_pos, re_neg))


def evaluate(
    X: np.ndarray,
    Y: np.ndarray,
    a_true: np.ndarray,
    b_true: np.ndarray,
    rho_true: float,
    a_hat: np.ndarray,
    b_hat: np.ndarray,
    rho_hat: float,
    method_name: str,
) -> dict:
    """
    Compute relative errors for a single method against the true CCA output.

    Parameters
    ----------
    X : np.ndarray of shape (n, p)
        Alice's centered data matrix (used for nothing here, kept for consistency).
    Y : np.ndarray of shape (n, q)
        Bob's centered data matrix.
    a_true : np.ndarray of shape (p,)
        True first canonical direction for X.
    b_true : np.ndarray of shape (q,)
        True first canonical direction for Y.
    rho_true : float
        True first canonical correlation.
    a_hat : np.ndarray of shape (p,)
        Recovered first canonical direction for X.
    b_hat : np.ndarray of shape (q,)
        Recovered first canonical direction for Y.
    rho_hat : float
        Recovered first canonical correlation.
    method_name : str
        Label for this method (used in the printed table).

    Returns
    -------
    dict with keys: method, RE_a, RE_b, RE_rho
    """
    re_a   = relative_error(a_hat, a_true)
    re_b   = relative_error(b_hat, b_true)
    re_rho = abs(rho_hat - rho_true) / (abs(rho_true) + 1e-12)

    return {
        "method" : method_name,
        "RE_a"   : re_a,
        "RE_b"   : re_b,
        "RE_rho" : re_rho,
    }
def print_directions(
    a_true: np.ndarray,
    b_true: np.ndarray,
    a_hat: np.ndarray,
    b_hat: np.ndarray,
    method_name: str,
) -> None:
    """
    Print the true and recovered canonical directions side by side.
    """
    print(f"\n--- {method_name} ---")
    print(f"  a_true : {np.round(a_true, 6)}")
    print(f"  a_hat  : {np.round(a_hat,  6)}")
    print(f"  b_true : {np.round(b_true, 6)}")
    print(f"  b_hat  : {np.round(b_hat,  6)}")

def print_table(results: list[dict]) -> None:
    """
    Print a formatted relative error table for all methods.

    Parameters
    ----------
    results : list of dicts, each with keys: method, RE_a, RE_b, RE_rho
    """
    header = f"{'Method':<20} {'RE(a)':>12} {'RE(b)':>12} {'RE(rho)':>12}"
    sep    = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['method']:<20} "
            f"{r['RE_a']:>12.6f} "
            f"{r['RE_b']:>12.6f} "
            f"{r['RE_rho']:>12.6f}"
        )
    print(sep)


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    X, Y = generate_data(n=200, p=10, q=8, seed=42)
    a_true, b_true, rho_true = true_cca(X, Y)

    print(f"True canonical correlation : {rho_true:.6f}\n")

    results = []

    # ------------------------------------------------------------------
    # Section 6 — single-vector alternating CCA
    # ------------------------------------------------------------------
    sim6 = CKKSSimulator(scale_bits=40, max_levels=30, seed=0)
    a6, b6, rho6 = alternating_cca(X, Y, sim6)
    results.append(evaluate(X, Y, a_true, b_true, rho_true, a6, b6, rho6, "Section 6"))
    print(f"Section 6  levels consumed : {sim6.level}")

    # ------------------------------------------------------------------
    # Section 8 — block alternating CCA with AGD
    # ------------------------------------------------------------------
    sim8 = CKKSSimulator(scale_bits=40, max_levels=30, seed=0)
    a8, b8, rho8 = als_cca(X, Y, sim8, m=1)
    results.append(evaluate(X, Y, a_true, b_true, rho_true, a8, b8, rho8, "Section 8"))
    print(f"Section 8  levels consumed : {sim8.level}")

    # ------------------------------------------------------------------
    # Section 10 — CCA over full PCA components
    # ------------------------------------------------------------------
    sim10 = CKKSSimulator(scale_bits=40, max_levels=30, seed=0)
    a10, b10, rho10 = pca_cca(X, Y, sim10)
    results.append(evaluate(X, Y, a_true, b_true, rho_true, a10, b10, rho10, "Section 10"))
    print(f"Section 10 levels consumed : {sim10.level}\n")

    # ------------------------------------------------------------------
    # Print RE table
    # ------------------------------------------------------------------
    print_table(results)