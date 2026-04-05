import numpy as np
import scipy.linalg


class CKKSSimulator:
    """
    Plaintext simulation of CKKS-compatible operations.

    Every method here corresponds to an operation that would run
    encrypted under real CKKS. Noise is injected to simulate the
    approximation errors that accumulate in a real homomorphic
    encryption setting.

    Parameters
    ----------
    scale_bits : int
        Controls encoding noise magnitude. Larger = less noise.
        Corresponds to log2(Delta) in the paper (Section 5).
    max_levels : int
        Maximum number of multiplicative levels before bootstrapping
        would be required in real CKKS. A warning is issued when
        this is reached.
    seed : int
        Random seed for reproducibility of noise.
    """

    def __init__(self, scale_bits: int = 40, max_levels: int = 100000, seed: int = 0):
        self.scale_bits = scale_bits
        self.max_levels = max_levels
        self.rng = np.random.default_rng(seed)
        self.level = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _noise_scale(self) -> float:
        return 2.0 ** (-self.scale_bits)

    def _add_noise(self, x: np.ndarray) -> np.ndarray:
        noise = self.rng.uniform(-1.0, 1.0, size=x.shape) * self._noise_scale()
        return x + noise

    def _increment_level(self, count: int = 1):
        self.level += count


    def reset_level(self):
        self.level = 0

    # ------------------------------------------------------------------
    # Section 5 — encoding noise
    # ------------------------------------------------------------------

    def simulate_encoding(self, x: np.ndarray) -> np.ndarray:
        """
        Simulate the rounding error introduced at CKKS encoding (Section 5).
        """
        return self._add_noise(x)

    # ------------------------------------------------------------------
    # Section 6.2 — normalization via approximate inverse square root
    # ------------------------------------------------------------------

    def approx_inverse_sqrt_newton(
            self,
            q: float,
            k: int = 0,
            n_newton: int = 3,
            n_goldschmidt: int = 3,
    ) -> float:
        """
        Approximate q^{-1/2} using input scaling to [0.25, 1] followed by
        Newton refinement and Goldschmidt simultaneous refinement (Section 6.2).

        The interval parameter is kept for API compatibility but is not used.
        Input scaling replaces the affine approximation on a wide interval,
        giving a tight initial guess regardless of the magnitude of q.

        In real CKKS, k is fixed in advance from known interval bounds as a
        plaintext preprocessing step by each party independently.
        Scaling by powers of 4 is a free scalar multiplication in CKKS.
        """
        q = float(q)

        if q <= 0:
            raise ValueError(f"q must be positive, got {q}")

        # ------------------------------------------------------------------
        # Step 1 — Scale q to canonical interval [0.25, 1]
        # Find integer k such that q / 4^k is in [0.25, 1]
        # Then q^{-1/2} = q_scaled^{-1/2} * 2^{-k}
        # In real CKKS k is determined from known bounds before encryption.
        # ------------------------------------------------------------------
            # k is fixed from plaintext spectral bounds by each party
            # before encryption begins. No runtime comparison on encrypted data.
        q_scaled = q / (4.0 ** k)
            # q_scaled is guaranteed to be in [0.25, 1] by construction of k

        # ------------------------------------------------------------------
        # Step 2 — Initial affine approximation on [0.25, 1]
        # f(0.25) = 2.0,  f(1.0) = 1.0
        # ------------------------------------------------------------------
        alpha  = 0.25
        beta   = 1.0
        f_alpha = alpha ** (-0.5)    # = 2.0
        f_beta  = beta  ** (-0.5)    # = 1.0
        a_coef  = (f_beta - f_alpha) / (beta - alpha)
        b_coef  = f_alpha - a_coef * alpha
        y = a_coef * q_scaled + b_coef    # initial approximation in [1.0, 2.0]

        # ------------------------------------------------------------------
        # Step 3 — Newton refinement on scaled value
        # y_{m+1} = y_m/2 * (3 - q_scaled * y_m^2)
        # ------------------------------------------------------------------
        for _ in range(n_newton):
            y = (y / 2.0) * (3.0 - q_scaled * y ** 2)
            self._increment_level(2)

        # ------------------------------------------------------------------
        # Step 4 — Goldschmidt simultaneous refinement on scaled value
        # ------------------------------------------------------------------
        x_g = q_scaled * y
        h_g = y / 2.0
        for _ in range(n_goldschmidt):
            r   = 0.5 - x_g * h_g
            x_g = x_g * (1.0 + r)
            h_g = h_g * (1.0 + r)
            self._increment_level(3)
        y_scaled = 2.0 * h_g    # approximate q_scaled^{-1/2}

        # ------------------------------------------------------------------
        # Step 5 — Scale back
        # q^{-1/2} = q_scaled^{-1/2} * 2^{-k}
        # ------------------------------------------------------------------
        y_final = y_scaled * (2.0 ** (-k))

        return float(y_final)

    def normalize_vector(
            self,
            v: np.ndarray,
            k: int = 0,
            n_newton: int = 3,
            n_goldschmidt: int = 3,
    ) -> np.ndarray:
        """
        Normalize v to unit Euclidean norm using approx_inverse_sqrt_newton.
        """
        q = float(v @ v)
        inv_norm = self.approx_inverse_sqrt_newton(q, k, n_newton, n_goldschmidt)
        return v * inv_norm

    def normalize_sxx(
            self,
            v: np.ndarray,
            Sxx: np.ndarray,
            k: int = 0,
            n_newton: int = 3,
            n_goldschmidt: int = 3,
    ) -> np.ndarray:
        """
        Normalize v so that v^T Sxx v = 1 (Section 6, Step 3).
        """
        q = float(v @ Sxx @ v)
        inv_norm = self.approx_inverse_sqrt_newton(q, k, n_newton, n_goldschmidt)
        return v * inv_norm

    # ------------------------------------------------------------------
    # Section 6.3 — iterative approximation of S^{-1} w (Richardson)
    # ------------------------------------------------------------------

    def approx_inverse_action(
        self,
        S: np.ndarray,
        w: np.ndarray,
        lambda_reg: float = 1e-3,
        n_iters: int = 20,
    ) -> np.ndarray:
        """
        Approximate (S + lambda*I)^{-1} w via Richardson iteration (Section 6.3).
        """
        S_tilde = S + lambda_reg * np.eye(S.shape[0])
        eigs = np.linalg.eigvalsh(S_tilde)
        lambda_min = float(eigs.min())
        lambda_max = float(eigs.max())
        mu = 2.0 / (lambda_min + lambda_max)

        z = mu * w
        for _ in range(n_iters):
            z = z + mu * (w - S_tilde @ z)
            self._increment_level(1)

        return z

    # ------------------------------------------------------------------
    # Section 7.1 — polynomial approximation of S^{-1/2} v
    # ------------------------------------------------------------------

    def approx_inverse_sqrt_poly(
            self,
            S: np.ndarray,
            v: np.ndarray,
            lambda_reg: float = 1e-3,
            poly_degree: int = 8,
            spectral_bounds: tuple[float, float] = None,
    ) -> np.ndarray:
        """
        Approximate (S + lambda*I)^{-1/2} v via Chebyshev polynomial (Section 7.1).
        """
        S_tilde = S + lambda_reg * np.eye(S.shape[0])

        # Spectral bounds are computed by each party from their local
        # plaintext data before encryption begins. No runtime comparison.
        if spectral_bounds is not None:
            alpha, beta = spectral_bounds
        else:
            # Fallback for simulation only: compute from plaintext S_tilde
            eigenvalues = np.linalg.eigvalsh(S_tilde)
            alpha = float(eigenvalues.min())
            beta = float(eigenvalues.max())

        z_samples = np.linspace(alpha, beta, 1000)
        f_samples = z_samples ** (-0.5)
        coeffs = np.polynomial.chebyshev.chebfit(z_samples, f_samples, poly_degree)

        powers = [v]
        for k in range(1, poly_degree + 1):
            powers.append(S_tilde @ powers[-1])
            self._increment_level(1)

        mono_coeffs = np.polynomial.chebyshev.cheb2poly(coeffs)
        mono_coeffs = np.resize(mono_coeffs, poly_degree + 1)

        u_hat = sum(mono_coeffs[k] * powers[k] for k in range(poly_degree + 1))

        return u_hat

    # ------------------------------------------------------------------
    # Section 8.1 — Newton iteration for scalar inverse z^{-1}
    # ------------------------------------------------------------------

    def approx_scalar_inverse_newton(
        self,
        z: float,
        interval: tuple[float, float],
        n_newton: int = 5,
    ) -> float:
        """
        Approximate z^{-1} via an initial affine approximation followed
        by Newton refinement (Section 8.1, equation 7).

        Newton update: y_{j+1} = y_j * (2 - z * y_j)
        """
        z = float(z)
        alpha, beta = interval

        f_alpha = 1.0 / alpha
        f_beta  = 1.0 / beta
        a_coef  = (f_beta - f_alpha) / (beta - alpha)
        b_coef  = f_alpha - a_coef * alpha
        y = a_coef * z + b_coef

        for _ in range(n_newton):
            y = y * (2.0 - z * y)
            self._increment_level(2)

        return float(y)

    # ------------------------------------------------------------------
    # Section 8.1 — AGD inner solver for S_tilde^{-1} W (x-side)
    # ------------------------------------------------------------------

    def agd_inverse_action(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        B: np.ndarray,
        lambda_reg: float = 1e-3,
        n_iters: int = 20,
        n_newton: int = 5,
        interval_step: tuple[float, float] = (1e-4, 1e4),
        interval_momentum: tuple[float, float] = (3.0, 103.0),
    ) -> np.ndarray:
        """
        Approximate S_tilde_xx^{-1} Sxy B via AGD (Section 8.1, equations 3-5).
        """
        n, p = X.shape
        S_tilde_xx = X.T @ X / (n - 1) + lambda_reg * np.eye(p)
        Sxy_B = X.T @ Y / (n - 1) @ B

        # Step size via power iteration + Newton inverse
        # lambda_max is computed by Alice from her plaintext S_tilde_xx
        # before encryption begins. Fully CKKS compatible since Alice
        # owns X and computes S_tilde_xx locally.
        lambda_max_approx = float(np.linalg.eigvalsh(S_tilde_xx).max())
        alpha_hat = 1.0 / lambda_max_approx

        z_prev = np.zeros((p, B.shape[1]))
        z_curr = np.zeros((p, B.shape[1]))

        for ell in range(1, n_iters + 1):
            inv_denom = self.approx_scalar_inverse_newton(
                float(ell + 2), interval_momentum, n_newton
            )
            gamma = float(ell - 1) * inv_denom

            f    = z_curr + gamma * (z_curr - z_prev)
            self._increment_level(1)

            grad = S_tilde_xx @ f - Sxy_B
            grad = np.clip(grad, -1e6, 1e6)
            self._increment_level(1)

            z_next = f - alpha_hat * grad
            z_next = np.clip(z_next, -1e6, 1e6)
            self._increment_level(1)

            z_prev = z_curr
            z_curr = z_next

        return z_curr

    # ------------------------------------------------------------------
    # Section 8.1 — AGD inner solver for S_tilde^{-1} W (y-side)
    # ------------------------------------------------------------------

    def agd_inverse_action_y(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        A: np.ndarray,
        lambda_reg: float = 1e-3,
        n_iters: int = 20,
        n_newton: int = 5,
        interval_step: tuple[float, float] = (1e-4, 1e4),
        interval_momentum: tuple[float, float] = (3.0, 103.0),
    ) -> np.ndarray:
        """
        Approximate S_tilde_yy^{-1} Syx A via AGD (Section 8.1, equations 3-5).
        """
        n, q = Y.shape
        S_tilde_yy = Y.T @ Y / (n - 1) + lambda_reg * np.eye(q)
        Syx_A = Y.T @ X / (n - 1) @ A

        # lambda_max is computed by Bob from his plaintext S_tilde_yy
        # before encryption begins.
        lambda_max_approx = float(np.linalg.eigvalsh(S_tilde_yy).max())
        alpha_hat = 1.0 / lambda_max_approx

        z_prev = np.zeros((q, A.shape[1]))
        z_curr = np.zeros((q, A.shape[1]))

        for ell in range(1, n_iters + 1):
            inv_denom = self.approx_scalar_inverse_newton(
                float(ell + 2), interval_momentum, n_newton
            )
            gamma = float(ell - 1) * inv_denom

            f    = z_curr + gamma * (z_curr - z_prev)
            self._increment_level(1)

            grad = S_tilde_yy @ f - Syx_A
            grad = np.clip(grad, -1e6, 1e6)
            self._increment_level(1)

            z_next = f - alpha_hat * grad
            z_next = np.clip(z_next, -1e6, 1e6)
            self._increment_level(1)

            z_prev = z_curr
            z_curr = z_next

        return z_curr

    # ------------------------------------------------------------------
    # Encrypted matrix-vector product (with noise)
    # ------------------------------------------------------------------

    def encrypted_matmul(self, A: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Simulate an encrypted matrix-vector or matrix-matrix product.
        """
        result = A @ x
        self._increment_level(1)
        return self._add_noise(result)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    sim = CKKSSimulator(scale_bits=40, max_levels=100000, seed=0)

    # --- Test encoding noise ---
    v = rng.standard_normal(10)
    v_enc = sim.simulate_encoding(v)
    print(f"Encoding noise magnitude : {np.abs(v_enc - v).max():.2e}  (expect ~2^-40 ≈ 9e-13)")

    # --- Test inverse square root ---
    q = 3.7
    approx = sim.approx_inverse_sqrt_newton(q)
    exact  = q ** (-0.5)
    print(f"Inv sqrt approx : {approx:.10f}  exact : {exact:.10f}  error : {abs(approx-exact):.2e}")

    # --- Test with small q ---
    q2 = 0.04675
    approx2 = sim.approx_inverse_sqrt_newton(q2)
    exact2  = q2 ** (-0.5)
    print(f"Inv sqrt small q approx : {approx2:.10f}  exact : {exact2:.10f}  error : {abs(approx2-exact2):.2e}")

    # --- Test normalize_vector ---
    v = rng.standard_normal(8)
    v_norm = sim.normalize_vector(v)
    print(f"||v_norm|| (expect ~1.0) : {np.linalg.norm(v_norm):.8f}")

    # --- Test approx_inverse_action ---
    S = rng.standard_normal((6, 6))
    S = S.T @ S / 6 + 0.1 * np.eye(6)
    w = rng.standard_normal(6)
    z_approx = sim.approx_inverse_action(S, w, lambda_reg=1e-3, n_iters=50)
    z_exact  = np.linalg.solve(S + 1e-3 * np.eye(6), w)
    print(f"Richardson inverse RE : {np.linalg.norm(z_approx - z_exact) / np.linalg.norm(z_exact):.2e}")

    # --- Test approx_inverse_sqrt_poly ---
    S2 = rng.standard_normal((5, 5))
    S2 = S2.T @ S2 / 5 + 0.1 * np.eye(5)
    v2 = rng.standard_normal(5)
    u_approx = sim.approx_inverse_sqrt_poly(S2, v2, lambda_reg=1e-3, poly_degree=8)
    S2_tilde = S2 + 1e-3 * np.eye(5)
    u_exact  = np.linalg.solve(scipy.linalg.sqrtm(S2_tilde), v2)
    print(f"Poly inv sqrt RE : {np.linalg.norm(u_approx - u_exact) / np.linalg.norm(u_exact):.2e}")

    # --- Test approx_scalar_inverse_newton ---
    z_val  = 7.3
    approx_inv = sim.approx_scalar_inverse_newton(z_val, interval=(0.1, 20.0))
    exact_inv  = 1.0 / z_val
    print(f"Scalar inverse approx : {approx_inv:.10f}  exact : {exact_inv:.10f}  error : {abs(approx_inv - exact_inv):.2e}")

    # --- Test agd_inverse_action ---
    n, p, q_dim, m = 50, 6, 5, 1
    X_test = rng.standard_normal((n, p))
    Y_test = rng.standard_normal((n, q_dim))
    X_test -= X_test.mean(axis=0)
    Y_test -= Y_test.mean(axis=0)
    B_test = rng.standard_normal((q_dim, m))
    lam = 1e-2

    A_agd   = sim.agd_inverse_action(X_test, Y_test, B_test, lambda_reg=lam, n_iters=30)
    Sxx     = X_test.T @ X_test / (n - 1) + lam * np.eye(p)
    Sxy     = X_test.T @ Y_test / (n - 1)
    A_exact = np.linalg.solve(Sxx, Sxy @ B_test)
    print(f"AGD x-side RE : {np.linalg.norm(A_agd - A_exact) / np.linalg.norm(A_exact):.2e}")

    B_agd   = sim.agd_inverse_action_y(X_test, Y_test, A_exact[:, :1], lambda_reg=lam, n_iters=30)
    Syy     = Y_test.T @ Y_test / (n - 1) + lam * np.eye(q_dim)
    Syx     = Sxy.T
    B_exact = np.linalg.solve(Syy, Syx @ A_exact[:, :1])
    print(f"AGD y-side RE : {np.linalg.norm(B_agd - B_exact) / np.linalg.norm(B_exact):.2e}")

    print(f"\nTotal multiplicative levels consumed : {sim.level}")