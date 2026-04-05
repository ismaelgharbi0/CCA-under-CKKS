import numpy as np
import matplotlib.pyplot as plt
import time
from data_generation import generate_data
from ckks_primitives import CKKSSimulator
from alternating_cca import alternating_cca
from als_cca import als_cca
from pca_cca import pca_cca
from evaluation import evaluate, print_table
from true_cca import true_cca

def run_all_experiments(
    n: int = 1000,
    p_values: list[int] = None,
    n_runs: int = 10,
    rho1: float = 0.9,
    scale_bits: int = 40,
    max_levels: int = 1000,
    lambda_reg: float = 1e-3,
    n_iters_inv: int = 20,
    T_max: int = 50,
    n_newton: int = 5,
    n_goldschmidt: int = 5,
) -> None:
    """
    Run the full experiment for each value of p=q and each of n_runs
    random dataset pairs.

    For each p value and each run:
      - Generate X, Y with known first canonical correlation rho1=0.9
      - Run all three CKKS-based CCA methods
      - Record RE(a), RE(b), RE(rho), runtime, and multiplicative levels

    After all runs, produce:
      - Per-p summary tables (mean and std of all metrics)
      - Global summary table across all p values
      - Plot 1: mean RE(a) vs p for all three methods
      - Plot 2: mean runtime vs p for all three methods
      - Plot 3: mean RE(a) * mean runtime vs p for all three methods

    Error bars show 1.96 * std (95% confidence interval).

    Parameters
    ----------
    n : int
        Number of samples. Fixed at 1000.
    p_values : list of int
        Values of p=q to test. Default: [5, 10, 25, 50, 100, 250, 500].
    n_runs : int
        Number of independent (X, Y) pairs per p value.
    rho1 : float
        True first canonical correlation. Fixed at 0.9.
    scale_bits : int
        CKKS scale parameter log2(Delta).
    max_levels : int
        Maximum multiplicative levels before bootstrapping warning.
    lambda_reg : float
        Regularization parameter for all methods.
    n_iters_inv : int
        Richardson iterations for als_cca inverse action.
    T_max : int
        Fixed number of outer iterations for all methods.
        No convergence check — runs exactly T_max iterations.
    n_newton : int
        Newton iterations for inverse square root normalization.
    n_goldschmidt : int
        Goldschmidt iterations for inverse square root normalization.
    """
    if p_values is None:
        p_values = [5, 10, 25, 50, 100, 250, 500]

    methods = ["alternating_cca", "als_cca", "pca_cca"]

    # ------------------------------------------------------------------
    # Storage structure:
    # all_results[p][method] = {
    #     RE_a: [], RE_b: [], RE_rho: [], levels: [], time: []
    # }
    # ------------------------------------------------------------------
    all_results = {
        p: {
            m: {"RE_a": [], "RE_b": [], "RE_rho": [], "levels": [], "time": []}
            for m in methods
        }
        for p in p_values
    }

    # ------------------------------------------------------------------
    # Main loop over p values and seeds
    # ------------------------------------------------------------------
    for p in p_values:
        q = p
        print(f"\n{'='*70}")
        print(f"p = q = {p}   n = {n}   rho1 = {rho1}   runs = {n_runs}")
        print(f"{'='*70}")








        print(
            f"  p={p}  T_max={T_max}  n_newton={n_newton}  n_goldschmidt={n_goldschmidt}  ")


        for run in range(n_runs):
            seed = run
            print(f"\n  --- Run {run+1}/{n_runs}  (seed={seed}) ---")

            # Generate data with known canonical correlation
            X, Y = generate_data(
                n=n, p=p, q=q, rho1=rho1, seed=seed
            )

            # Ground truth from empirical covariance matrices
            # Both true_cca and all CKKS methods use the same
            # empirical Sxx, Syy, Sxy — RE measures only CKKS
            # approximation error not statistical estimation error
            a_true, b_true, rho_true = true_cca(X, Y)

            # ----------------------------------------------------------
            # alternating_cca
            # Two-sided power iteration in whitened space.
            # a and b recovered via back-transformation at the end.
            # ----------------------------------------------------------
            sim1 = CKKSSimulator(
                scale_bits=scale_bits, max_levels=max_levels, seed=0
            )
            t_start = time.perf_counter()
            a1, b1, rho1_hat = alternating_cca(
                X, Y, sim1,
                lambda_reg=lambda_reg,
                T_max=T_max,
                n_newton=n_newton,
                n_goldschmidt=n_goldschmidt,
            )
            t_end = time.perf_counter()
            r1 = evaluate(
                X, Y, a_true, b_true, rho_true,
                a1, b1, rho1_hat, "alternating_cca"
            )
            all_results[p]["alternating_cca"]["RE_a"].append(r1["RE_a"])
            all_results[p]["alternating_cca"]["RE_b"].append(r1["RE_b"])
            all_results[p]["alternating_cca"]["RE_rho"].append(r1["RE_rho"])
            all_results[p]["alternating_cca"]["levels"].append(sim1.level)
            all_results[p]["alternating_cca"]["time"].append(t_end - t_start)

            # ----------------------------------------------------------
            # als_cca
            # Direct iteration of a and b in original space.
            # Inverse action via Richardson iteration on Sxx and Syy.
            # ----------------------------------------------------------
            sim2 = CKKSSimulator(
                scale_bits=scale_bits, max_levels=max_levels, seed=0
            )
            t_start = time.perf_counter()
            a2, b2, rho2_hat = als_cca(
                X, Y, sim2,
                lambda_reg=lambda_reg,
                n_iters_inv=n_iters_inv,
                T_max=T_max,
                n_newton=n_newton,
                n_goldschmidt=n_goldschmidt,
            )
            t_end = time.perf_counter()
            r2 = evaluate(
                X, Y, a_true, b_true, rho_true,
                a2, b2, rho2_hat, "als_cca"
            )
            all_results[p]["als_cca"]["RE_a"].append(r2["RE_a"])
            all_results[p]["als_cca"]["RE_b"].append(r2["RE_b"])
            all_results[p]["als_cca"]["RE_rho"].append(r2["RE_rho"])
            all_results[p]["als_cca"]["levels"].append(sim2.level)
            all_results[p]["als_cca"]["time"].append(t_end - t_start)

            # ----------------------------------------------------------
            # pca_cca
            # CCA over full PCA components in whitened space.
            # Simplified iteration since covariance is identity.
            # ----------------------------------------------------------
            sim3 = CKKSSimulator(
                scale_bits=scale_bits, max_levels=max_levels, seed=0
            )
            t_start = time.perf_counter()
            a3, b3, rho3_hat = pca_cca(
                X, Y, sim3,
                lambda_reg=lambda_reg,
                T_max=T_max,
                n_newton=n_newton,
                n_goldschmidt=n_goldschmidt,
            )
            t_end = time.perf_counter()
            r3 = evaluate(
                X, Y, a_true, b_true, rho_true,
                a3, b3, rho3_hat, "pca_cca"
            )
            all_results[p]["pca_cca"]["RE_a"].append(r3["RE_a"])
            all_results[p]["pca_cca"]["RE_b"].append(r3["RE_b"])
            all_results[p]["pca_cca"]["RE_rho"].append(r3["RE_rho"])
            all_results[p]["pca_cca"]["levels"].append(sim3.level)
            all_results[p]["pca_cca"]["time"].append(t_end - t_start)

            # Per-run table
            print_table([r1, r2, r3])

        # --------------------------------------------------------------
        # Per-p summary
        # --------------------------------------------------------------
        print(f"\n  Summary for p=q={p} over {n_runs} runs:")
        _print_summary_table(all_results[p], methods)

    # ------------------------------------------------------------------
    # Global summary table across all p values
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("Global summary — mean RE(a) and runtime across all p values")
    print(f"{'='*70}")
    _print_global_table(all_results, p_values, methods)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    _plot_re_vs_p(all_results, p_values, methods, n_runs)
    _plot_runtime_vs_p(all_results, p_values, methods, n_runs)
    _plot_re_times_runtime_vs_p(all_results, p_values, methods, n_runs)


def _print_summary_table(
    results_p: dict,
    methods: list[str],
) -> None:
    """Print mean and std of RE, levels and runtime for a single p value."""
    header = (
        f"  {'Method':<18} "
        f"{'RE(a) mean':>12} {'RE(a) std':>10} "
        f"{'RE(b) mean':>12} {'RE(b) std':>10} "
        f"{'RE(rho) mean':>14} "
        f"{'levels mean':>12} "
        f"{'time mean':>12}"
    )
    sep = "  " + "-" * (len(header) - 2)
    print(sep)
    print(header)
    print(sep)
    for m in methods:
        re_a   = np.array(results_p[m]["RE_a"])
        re_b   = np.array(results_p[m]["RE_b"])
        re_rho = np.array(results_p[m]["RE_rho"])
        lvl    = np.array(results_p[m]["levels"])
        t      = np.array(results_p[m]["time"])
        print(
            f"  {m:<18} "
            f"{re_a.mean():>12.6f} {re_a.std():>10.6f} "
            f"{re_b.mean():>12.6f} {re_b.std():>10.6f} "
            f"{re_rho.mean():>14.6f} "
            f"{lvl.mean():>12.0f} "
            f"{t.mean():>11.3f}s"
        )
    print(sep)


def _print_global_table(
    all_results: dict,
    p_values: list[int],
    methods: list[str],
) -> None:
    """Print mean RE(a) and runtime for all methods across all p values."""
    col_width = 30
    header = f"  {'p=q':<8}" + "".join(f"{m:>{col_width}}" for m in methods)
    sep    = "  " + "-" * (len(header) - 2)
    print(sep)
    print(header)
    print(f"  {'':8}" + "".join(
        f"{'RE(a) / time(s)':>{col_width}}" for _ in methods
    ))
    print(sep)
    for p in p_values:
        row = f"  {p:<8}"
        for m in methods:
            mean_re   = np.mean(all_results[p][m]["RE_a"])
            mean_time = np.mean(all_results[p][m]["time"])
            row += f"{mean_re:.6f} / {mean_time:.3f}s".rjust(col_width)
        print(row)
    print(sep)


def _plot_re_vs_p(
    all_results: dict,
    p_values: list[int],
    methods: list[str],
    n_runs: int = 10,
) -> None:
    """
    Plot mean RE(a) vs p for all three methods on the same axes.
    Error bars show 1.96 * std (95% confidence interval).
    Axis tick labels show normal digits, not powers of 10.
    """
    colors = {
        "alternating_cca": "#4C72B0",
        "als_cca"        : "#DD8452",
        "pca_cca"        : "#55A868",
    }
    fig, ax = plt.subplots(figsize=(8, 5))

    for m in methods:
        means = [np.mean(all_results[p][m]["RE_a"]) for p in p_values]
        stds  = [1.96 * np.std(all_results[p][m]["RE_a"]) for p in p_values]
        ax.errorbar(
            p_values, means, yerr=stds,
            label=m, color=colors[m],
            marker="o", capsize=4, linewidth=1.5
        )

    ax.set_xlabel("p = q")
    ax.set_ylabel("Mean RE(a)")
    ax.set_title("Average RE(a) vs dimension\n(n=1000, rho1=0.9, 10 runs per p)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2e}"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("re_vs_p.png", dpi=150)
    print("\nSaved: re_vs_p.png")
    plt.show()


def _plot_runtime_vs_p(
    all_results: dict,
    p_values: list[int],
    methods: list[str],
    n_runs: int = 10,
) -> None:
    """
    Plot mean runtime vs p for all three methods on the same axes.
    Error bars show 1.96 * std (95% confidence interval).
    Axis tick labels show normal digits, not powers of 10.
    """
    colors = {
        "alternating_cca": "#4C72B0",
        "als_cca"        : "#DD8452",
        "pca_cca"        : "#55A868",
    }
    fig, ax = plt.subplots(figsize=(8, 5))

    for m in methods:
        means = [np.mean(all_results[p][m]["time"]) for p in p_values]
        stds  = [1.96 * np.std(all_results[p][m]["time"]) for p in p_values]
        ax.errorbar(
            p_values, means, yerr=stds,
            label=m, color=colors[m],
            marker="o", capsize=4, linewidth=1.5
        )

    ax.set_xlabel("p = q")
    ax.set_ylabel("Mean runtime (seconds)")
    ax.set_title("Average runtime vs dimension\n(n=1000, rho1=0.9, 10 runs per p)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.3f}"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("runtime_vs_p.png", dpi=150)
    print("Saved: runtime_vs_p.png")
    plt.show()


def _plot_re_times_runtime_vs_p(
    all_results: dict,
    p_values: list[int],
    methods: list[str],
    n_runs: int = 10,
) -> None:
    """
    Plot mean RE(a) * mean runtime vs p for all three methods.
    This combined efficiency metric penalizes both inaccuracy and slowness.
    Error bars use error propagation with 1.96 * std.
    Axis tick labels show normal digits, not powers of 10.
    """
    colors = {
        "alternating_cca": "#4C72B0",
        "als_cca"        : "#DD8452",
        "pca_cca"        : "#55A868",
    }
    fig, ax = plt.subplots(figsize=(8, 5))

    for m in methods:
        re_means = [np.mean(all_results[p][m]["RE_a"]) for p in p_values]
        t_means  = [np.mean(all_results[p][m]["time"])  for p in p_values]
        re_stds  = [np.std(all_results[p][m]["RE_a"])   for p in p_values]
        t_stds   = [np.std(all_results[p][m]["time"])   for p in p_values]

        products = [re_means[i] * t_means[i] for i in range(len(p_values))]

        # Error propagation for product of two independent quantities:
        # std(RE * t) = sqrt((std_RE / mean_RE)^2 + (std_t / mean_t)^2) * mean_RE * mean_t
        product_stds = [
            1.96 * np.sqrt(
                (re_stds[i] / (re_means[i] + 1e-12)) ** 2 +
                (t_stds[i]  / (t_means[i]  + 1e-12)) ** 2
            ) * products[i]
            for i in range(len(p_values))
        ]

        ax.errorbar(
            p_values, products, yerr=product_stds,
            label=m, color=colors[m],
            marker="o", capsize=4, linewidth=1.5
        )

    ax.set_xlabel("p = q")
    ax.set_ylabel("Mean RE(a) x Mean runtime (s)")
    ax.set_title("RE(a) x Runtime vs dimension\n(n=1000, rho1=0.9, 10 runs per p)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.6f}"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("re_times_runtime_vs_p.png", dpi=150)
    print("Saved: re_times_runtime_vs_p.png")
    plt.show()


if __name__ == "__main__":
    run_all_experiments(
        n=1000,
        p_values=[5, 10, 25, 50, 100, 250, 500],
        n_runs=1,
        rho1=0.9,
        scale_bits=40,
        max_levels=1000,
        lambda_reg=1e-3,
        n_iters_inv=20,
        T_max=50,
        n_newton=3,
        n_goldschmidt=3,
    )