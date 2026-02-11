#!/usr/bin/env python3
"""Run paper-quality controls for tissue invariance diagnostics.

This script adds three controls used for submission-gatekeeper assessment:
1) threshold sensitivity analysis for pass/fail criteria,
2) permutation-based null control for pass counts,
3) leave-one-environment-out stability for beta estimates.

It consumes `tissue_edge_environment_stats.csv` and writes artifacts to
`implementation/outputs/tissue_invariance_quality/`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def fit_invariance_model(var_x: np.ndarray, slopes: np.ndarray) -> Dict[str, float]:
    """Fit slope = beta + c / var_x and return key diagnostics.

    Parameters
    ----------
    var_x:
        Per-environment variance of TF proxy X.
    slopes:
        Per-environment observational slope Cov(X,Y)/Var(X).

    Returns
    -------
    Dict[str, float]
        beta_hat, c_hat, rmse, r2, varx_ratio, pairwise_beta_sd.
    """
    inv_varx = 1.0 / np.clip(var_x, 1e-12, None)
    design = np.column_stack([np.ones(inv_varx.size), inv_varx])

    coef, *_ = np.linalg.lstsq(design, slopes, rcond=None)
    beta_hat = float(coef[0])
    c_hat = float(coef[1])

    fitted = design @ coef
    residuals = slopes - fitted
    rmse = float(np.sqrt(np.mean(residuals**2)))

    denom = float(np.sum((slopes - np.mean(slopes)) ** 2))
    if denom <= 1e-12:
        r2 = 1.0 if rmse <= 1e-12 else 0.0
    else:
        r2 = 1.0 - float(np.sum(residuals**2) / denom)

    varx_ratio = float(np.max(var_x) / max(np.min(var_x), 1e-12))

    # Pairwise beta stability from all two-environment solves.
    pairwise_betas: List[float] = []
    for i in range(slopes.size):
        for j in range(i + 1, slopes.size):
            v1 = float(var_x[i])
            v2 = float(var_x[j])
            s1 = float(slopes[i])
            s2 = float(slopes[j])
            denom_pair = (1.0 / v1) - (1.0 / v2)
            if abs(denom_pair) <= 1e-12:
                continue
            c_ij = (s1 - s2) / denom_pair
            beta_ij = s1 - c_ij / v1
            pairwise_betas.append(float(beta_ij))

    if len(pairwise_betas) >= 2:
        pairwise_beta_sd = float(np.std(np.asarray(pairwise_betas), ddof=1))
    else:
        pairwise_beta_sd = float("nan")

    return {
        "beta_hat": beta_hat,
        "c_hat": c_hat,
        "rmse": rmse,
        "r2": r2,
        "varx_ratio": varx_ratio,
        "pairwise_beta_sd": pairwise_beta_sd,
    }


def evaluate_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Compute invariance-fit metrics for each edge in the input dataframe."""
    rows: List[Dict[str, float | str]] = []
    for (source, target), group in df.groupby(["source", "target"]):
        var_x = group["var_x"].to_numpy(dtype=float)
        slopes = group["slope"].to_numpy(dtype=float)
        fit = fit_invariance_model(var_x=var_x, slopes=slopes)

        rows.append(
            {
                "source": source,
                "target": target,
                "n_env": int(group.shape[0]),
                **fit,
            }
        )
    return pd.DataFrame(rows)


def threshold_sensitivity(edge_fit_df: pd.DataFrame) -> pd.DataFrame:
    """Sweep pass thresholds and report pass fractions."""
    r2_grid = [0.30, 0.40, 0.50, 0.60, 0.70]
    rmse_grid = [0.05, 0.10, 0.15]
    varx_ratio_grid = [1.10, 1.25, 1.50]
    pairwise_sd_grid = [0.15, 0.20, 0.30]

    rows: List[Dict[str, float]] = []
    n_edges = float(edge_fit_df.shape[0])

    for r2_min in r2_grid:
        for rmse_max in rmse_grid:
            for varx_ratio_min in varx_ratio_grid:
                for pairwise_sd_max in pairwise_sd_grid:
                    passed = (
                        (edge_fit_df["r2"] >= r2_min)
                        & (edge_fit_df["rmse"] <= rmse_max)
                        & (edge_fit_df["varx_ratio"] >= varx_ratio_min)
                        & (edge_fit_df["pairwise_beta_sd"] <= pairwise_sd_max)
                    )
                    pass_count = int(passed.sum())
                    rows.append(
                        {
                            "r2_min": r2_min,
                            "rmse_max": rmse_max,
                            "varx_ratio_min": varx_ratio_min,
                            "pairwise_sd_max": pairwise_sd_max,
                            "pass_count": pass_count,
                            "pass_fraction": pass_count / n_edges,
                        }
                    )

    return pd.DataFrame(rows)


def permutation_null(
    edge_env_df: pd.DataFrame,
    n_perm: int,
    threshold: Dict[str, float],
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Estimate null distribution by permuting slopes across edges per environment.

    This preserves each environment's slope distribution but destroys cross-environment
    edge correspondence, providing a falsification control for invariance pass counts.
    """
    rng = np.random.default_rng(seed)

    pivot_slopes = edge_env_df.pivot(index=["source", "target"], columns="environment", values="slope")
    pivot_varx = edge_env_df.pivot(index=["source", "target"], columns="environment", values="var_x")

    edges = pivot_slopes.index
    envs = list(pivot_slopes.columns)

    observed_fit = evaluate_edges(edge_env_df)
    observed_pass_mask = (
        (observed_fit["r2"] >= threshold["r2_min"])
        & (observed_fit["rmse"] <= threshold["rmse_max"])
        & (observed_fit["varx_ratio"] >= threshold["varx_ratio_min"])
        & (observed_fit["pairwise_beta_sd"] <= threshold["pairwise_sd_max"])
    )
    observed_pass_count = int(observed_pass_mask.sum())

    perm_rows: List[Dict[str, float]] = []
    null_pass_counts: List[int] = []

    slope_matrix = pivot_slopes.to_numpy(dtype=float)
    varx_matrix = pivot_varx.to_numpy(dtype=float)

    for perm_idx in range(n_perm):
        shuffled = np.empty_like(slope_matrix)
        for env_idx in range(slope_matrix.shape[1]):
            perm = rng.permutation(slope_matrix.shape[0])
            shuffled[:, env_idx] = slope_matrix[perm, env_idx]

        # Fit each pseudo-edge quickly in matrix form.
        pass_count = 0
        for edge_idx in range(shuffled.shape[0]):
            fit = fit_invariance_model(var_x=varx_matrix[edge_idx, :], slopes=shuffled[edge_idx, :])
            passed = (
                (fit["r2"] >= threshold["r2_min"])
                and (fit["rmse"] <= threshold["rmse_max"])
                and (fit["varx_ratio"] >= threshold["varx_ratio_min"])
                and (fit["pairwise_beta_sd"] <= threshold["pairwise_sd_max"])
            )
            if passed:
                pass_count += 1

        null_pass_counts.append(pass_count)
        perm_rows.append(
            {
                "perm_idx": perm_idx,
                "null_pass_count": pass_count,
                "null_pass_fraction": pass_count / float(shuffled.shape[0]),
            }
        )

    null_array = np.asarray(null_pass_counts, dtype=float)
    # One-sided empirical p-value with +1 correction.
    p_emp = float((1 + np.sum(null_array >= observed_pass_count)) / (n_perm + 1))

    summary = {
        "observed_pass_count": observed_pass_count,
        "observed_pass_fraction": observed_pass_count / float(len(edges)),
        "null_mean_pass_count": float(null_array.mean()),
        "null_sd_pass_count": float(null_array.std(ddof=1)),
        "null_q95_pass_count": float(np.quantile(null_array, 0.95)),
        "empirical_p_value": p_emp,
        "n_permutations": int(n_perm),
    }

    return pd.DataFrame(perm_rows), summary


def leave_one_environment_out(edge_env_df: pd.DataFrame) -> pd.DataFrame:
    """Compute beta stability under leave-one-environment-out fits."""
    rows: List[Dict[str, float | str]] = []

    for (source, target), group in edge_env_df.groupby(["source", "target"]):
        full_fit = fit_invariance_model(
            var_x=group["var_x"].to_numpy(dtype=float),
            slopes=group["slope"].to_numpy(dtype=float),
        )
        full_beta = full_fit["beta_hat"]

        loo_betas: List[float] = []
        for held_out_env in group["environment"].tolist():
            subset = group[group["environment"] != held_out_env]
            subset_fit = fit_invariance_model(
                var_x=subset["var_x"].to_numpy(dtype=float),
                slopes=subset["slope"].to_numpy(dtype=float),
            )
            loo_betas.append(subset_fit["beta_hat"])

        loo_array = np.asarray(loo_betas, dtype=float)
        rows.append(
            {
                "source": source,
                "target": target,
                "beta_hat_full": float(full_beta),
                "beta_hat_loo_mean": float(loo_array.mean()),
                "beta_hat_loo_sd": float(loo_array.std(ddof=1)),
                "beta_hat_loo_min": float(loo_array.min()),
                "beta_hat_loo_max": float(loo_array.max()),
                "beta_hat_loo_range": float(loo_array.max() - loo_array.min()),
            }
        )

    return pd.DataFrame(rows)


def plot_threshold_heatmap(
    threshold_df: pd.DataFrame,
    output_path: Path,
    rmse_max: float,
    varx_ratio_min: float,
    pairwise_sd_max: float,
) -> None:
    """Plot pass fraction heatmap for a fixed triplet of non-r2 thresholds."""
    subset = threshold_df[
        (threshold_df["rmse_max"] == rmse_max)
        & (threshold_df["varx_ratio_min"] == varx_ratio_min)
        & (threshold_df["pairwise_sd_max"] == pairwise_sd_max)
    ].copy()

    pivot = subset.pivot(index="r2_min", columns="rmse_max", values="pass_fraction")
    # Because rmse_max is fixed above, we still keep a compact one-column heatmap.
    fig, ax = plt.subplots(figsize=(4.0, 3.8))
    matrix = pivot.to_numpy(dtype=float)
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", origin="lower")

    ax.set_xticks(np.arange(pivot.shape[1]), labels=[f"{x:.2f}" for x in pivot.columns])
    ax.set_yticks(np.arange(pivot.shape[0]), labels=[f"{x:.2f}" for x in pivot.index])
    ax.set_xlabel("RMSE max")
    ax.set_ylabel("R2 min")
    ax.set_title("Pass fraction sensitivity")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_permutation_null(
    null_df: pd.DataFrame,
    observed_pass_count: int,
    output_path: Path,
) -> None:
    """Plot histogram of null pass counts and mark observed value."""
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    ax.hist(null_df["null_pass_count"], bins=20, color="#4f7cac", alpha=0.85)
    ax.axvline(observed_pass_count, color="black", linestyle="--", linewidth=1.2, label="Observed")
    ax.set_xlabel("Pass count under permutation null")
    ax.set_ylabel("Frequency")
    ax.set_title("Null control for edge pass count")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--edge-env-stats",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "outputs"
        / "tissue_invariance"
        / "tissue_edge_environment_stats.csv",
        help="Path to edge-environment statistics table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "tissue_invariance_quality",
        help="Directory for quality-check CSV/JSON outputs.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "reports" / "figures",
        help="Directory for generated figures.",
    )
    parser.add_argument("--n-permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)

    # Default threshold aligned with prior tissue diagnostic script.
    parser.add_argument("--r2-min", type=float, default=0.50)
    parser.add_argument("--rmse-max", type=float, default=0.10)
    parser.add_argument("--varx-ratio-min", type=float, default=1.25)
    parser.add_argument("--pairwise-sd-max", type=float, default=0.20)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    edge_env_stats_path: Path = args.edge_env_stats
    output_dir: Path = args.output_dir
    figure_dir: Path = args.figure_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    edge_env_df = pd.read_csv(edge_env_stats_path)

    edge_fit_df = evaluate_edges(edge_env_df)
    edge_fit_df.to_csv(output_dir / "edge_fit_metrics_recomputed.csv", index=False)

    threshold_df = threshold_sensitivity(edge_fit_df)
    threshold_df.to_csv(output_dir / "threshold_sensitivity.csv", index=False)

    threshold = {
        "r2_min": float(args.r2_min),
        "rmse_max": float(args.rmse_max),
        "varx_ratio_min": float(args.varx_ratio_min),
        "pairwise_sd_max": float(args.pairwise_sd_max),
    }

    null_df, null_summary = permutation_null(
        edge_env_df=edge_env_df,
        n_perm=args.n_permutations,
        threshold=threshold,
        seed=args.seed,
    )
    null_df.to_csv(output_dir / "permutation_null_pass_counts.csv", index=False)

    loo_df = leave_one_environment_out(edge_env_df)
    loo_df.to_csv(output_dir / "loo_beta_stability.csv", index=False)

    # Merge pass labels for convenient downstream reporting.
    pass_mask = (
        (edge_fit_df["r2"] >= threshold["r2_min"])
        & (edge_fit_df["rmse"] <= threshold["rmse_max"])
        & (edge_fit_df["varx_ratio"] >= threshold["varx_ratio_min"])
        & (edge_fit_df["pairwise_beta_sd"] <= threshold["pairwise_sd_max"])
    )
    merged = edge_fit_df.merge(loo_df, on=["source", "target"], how="left")
    merged["diagnostic_pass"] = pass_mask.astype(int)
    merged.to_csv(output_dir / "edge_fit_with_loo_and_pass.csv", index=False)

    # Aggregate summaries used directly in manuscript writing.
    pass_subset = merged[merged["diagnostic_pass"] == 1]
    fail_subset = merged[merged["diagnostic_pass"] == 0]

    summary = {
        "input_edge_count": int(merged.shape[0]),
        "threshold": threshold,
        "pass_count": int(pass_subset.shape[0]),
        "pass_fraction": float(pass_subset.shape[0] / merged.shape[0]),
        "pass_median_r2": float(pass_subset["r2"].median()) if not pass_subset.empty else float("nan"),
        "pass_median_rmse": float(pass_subset["rmse"].median()) if not pass_subset.empty else float("nan"),
        "pass_median_loo_range": float(pass_subset["beta_hat_loo_range"].median())
        if not pass_subset.empty
        else float("nan"),
        "fail_median_r2": float(fail_subset["r2"].median()) if not fail_subset.empty else float("nan"),
        "fail_median_loo_range": float(fail_subset["beta_hat_loo_range"].median())
        if not fail_subset.empty
        else float("nan"),
        "permutation_null": null_summary,
    }

    with (output_dir / "quality_check_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    # Figures for paper-quality audit.
    plot_threshold_heatmap(
        threshold_df=threshold_df,
        output_path=figure_dir / "fig09_threshold_sensitivity.png",
        rmse_max=float(args.rmse_max),
        varx_ratio_min=float(args.varx_ratio_min),
        pairwise_sd_max=float(args.pairwise_sd_max),
    )
    plot_permutation_null(
        null_df=null_df,
        observed_pass_count=int(summary["pass_count"]),
        output_path=figure_dir / "fig10_permutation_null_pass_count.png",
    )


if __name__ == "__main__":
    main()
