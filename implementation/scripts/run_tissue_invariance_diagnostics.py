#!/usr/bin/env python3
"""Run cross-tissue environment-invariance diagnostics on existing h5ad outputs.

This script is a lightweight real-data sanity component for proposal 4.
It does not claim causal truth of edges. It evaluates whether observed
cross-environment slope patterns are compatible with the invariance form:

    s_e = beta + c / Var(X | E=e)

where s_e is the observational slope Cov(X, Y)/Var(X) in environment e.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(frozen=True)
class EnvMatrix:
    """Holds expression matrix and indexing for one environment."""

    env: str
    n_cells: int
    genes: List[str]
    matrix: np.ndarray
    gene_to_col: Dict[str, int]


def to_dense(x: np.ndarray | sparse.spmatrix) -> np.ndarray:
    """Convert dense/sparse matrix-like object into a dense numpy array."""
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Return OLS slope for y ~ 1 + x."""
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    var_x = float(np.mean(x_centered * x_centered))
    if var_x <= 1e-12:
        return float("nan")
    cov_xy = float(np.mean(x_centered * y_centered))
    return cov_xy / var_x


def fit_invariance_line(inv_varx: np.ndarray, slopes: np.ndarray) -> Tuple[float, float, float, float]:
    """Fit slopes = beta + c * inv_varx and return fit diagnostics.

    Returns:
    - beta_hat
    - c_hat
    - rmse
    - r2
    """
    design = np.column_stack([np.ones(inv_varx.size), inv_varx])
    coef, *_ = np.linalg.lstsq(design, slopes, rcond=None)
    beta_hat = float(coef[0])
    c_hat = float(coef[1])

    preds = design @ coef
    residuals = slopes - preds
    rmse = float(np.sqrt(np.mean(residuals**2)))

    denom = float(np.sum((slopes - np.mean(slopes)) ** 2))
    if denom <= 1e-12:
        r2 = 1.0 if rmse <= 1e-12 else 0.0
    else:
        r2 = 1.0 - float(np.sum(residuals**2) / denom)
    return beta_hat, c_hat, rmse, r2


def pairwise_beta_estimates(slopes: np.ndarray, varx: np.ndarray) -> List[float]:
    """Compute two-environment beta estimates for all environment pairs."""
    estimates: List[float] = []
    for i, j in itertools.combinations(range(slopes.size), 2):
        s1 = float(slopes[i])
        s2 = float(slopes[j])
        v1 = float(varx[i])
        v2 = float(varx[j])

        denom = (1.0 / v1) - (1.0 / v2)
        if abs(denom) <= 1e-12:
            continue
        c = (s1 - s2) / denom
        beta = s1 - c / v1
        estimates.append(beta)
    return estimates


def load_env_matrix(path: Path, env: str, genes: Sequence[str]) -> EnvMatrix:
    """Load only requested genes from an h5ad file into a dense matrix."""
    adata = ad.read_h5ad(path)
    var_names = pd.Index(adata.var_names)
    present = [gene for gene in genes if gene in var_names]

    if not present:
        raise ValueError(f"No requested genes found in environment {env} ({path}).")

    # Slice once per environment for efficiency and stable memory use.
    env_matrix = to_dense(adata[:, present].X)
    gene_to_col = {gene: idx for idx, gene in enumerate(present)}

    return EnvMatrix(
        env=env,
        n_cells=int(adata.n_obs),
        genes=present,
        matrix=env_matrix,
        gene_to_col=gene_to_col,
    )


def write_json(path: Path, payload: Dict[str, object]) -> None:
    """Write traceable JSON metadata/summaries."""
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs-path",
        type=Path,
        default=Path(
            "single_cell_mechinterp/outputs/invariant_causal_edges/pairs/"
            "invariant_causal_pairs_with_refs.tsv"
        ),
        help="Path to candidate TF-target pairs TSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "tissue_invariance",
        help="Directory for diagnostic CSV/JSON outputs.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "reports" / "figures",
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--env-spec",
        nargs="+",
        default=[
            "kidney=single_cell_mechinterp/outputs/invariant_causal_edges/kidney/processed.h5ad",
            "lung=single_cell_mechinterp/outputs/invariant_causal_edges/lung/processed.h5ad",
            "immune=single_cell_mechinterp/outputs/invariant_causal_edges/immune/processed.h5ad",
            "external_lung=single_cell_mechinterp/outputs/invariant_causal_edges/external_lung/processed.h5ad",
        ],
        help="Environment specs as env_name=path_to_h5ad.",
    )
    parser.add_argument(
        "--min-r2",
        type=float,
        default=0.50,
        help="Minimum R^2 threshold for diagnostic pass.",
    )
    parser.add_argument(
        "--max-rmse",
        type=float,
        default=0.10,
        help="Maximum RMSE threshold for diagnostic pass.",
    )
    parser.add_argument(
        "--min-varx-ratio",
        type=float,
        default=1.25,
        help="Minimum max/min Var(X) ratio across environments.",
    )
    parser.add_argument(
        "--max-pairwise-beta-sd",
        type=float,
        default=0.20,
        help="Maximum SD of pairwise two-env beta estimates for pass.",
    )
    return parser


def parse_env_specs(specs: Sequence[str]) -> Dict[str, Path]:
    """Parse env_name=path specifications."""
    mapping: Dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid env-spec (missing '='): {spec}")
        env, path = spec.split("=", 1)
        env = env.strip()
        path_obj = Path(path.strip())
        if not env:
            raise ValueError(f"Empty environment name in spec: {spec}")
        mapping[env] = path_obj
    return mapping


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    figure_dir: Path = args.figure_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    env_paths = parse_env_specs(args.env_spec)

    pairs = pd.read_csv(args.pairs_path, sep="\t")
    pairs = pairs[["source", "target", "references", "n_references"]].drop_duplicates().copy()

    all_genes = sorted(set(pairs["source"]).union(set(pairs["target"])))

    env_data: Dict[str, EnvMatrix] = {}
    env_available: Dict[str, set[str]] = {}
    for env, path in env_paths.items():
        env_matrix = load_env_matrix(path=path, env=env, genes=all_genes)
        env_data[env] = env_matrix
        env_available[env] = set(env_matrix.genes)

    # Keep only edges observed in all environments.
    common_edges: List[Tuple[str, str]] = []
    for _, row in pairs.iterrows():
        source = str(row["source"])
        target = str(row["target"])
        if all(source in env_available[e] and target in env_available[e] for e in env_paths):
            common_edges.append((source, target))

    edge_env_rows: List[Dict[str, object]] = []
    edge_fit_rows: List[Dict[str, object]] = []

    pair_meta = {
        (str(row["source"]), str(row["target"])): {
            "references": str(row["references"]),
            "n_references": int(row["n_references"]),
        }
        for _, row in pairs.iterrows()
    }

    env_order = list(env_paths.keys())

    for source, target in common_edges:
        slopes: List[float] = []
        varx_values: List[float] = []

        for env in env_order:
            matrix = env_data[env]
            x = matrix.matrix[:, matrix.gene_to_col[source]]
            y = matrix.matrix[:, matrix.gene_to_col[target]]

            varx = float(np.var(x))
            slope = ols_slope(x, y)
            cov_xy = float(np.mean((x - x.mean()) * (y - y.mean())))
            corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-12 and np.std(y) > 1e-12 else float("nan")

            slopes.append(slope)
            varx_values.append(varx)

            edge_env_rows.append(
                {
                    "source": source,
                    "target": target,
                    "environment": env,
                    "n_cells": matrix.n_cells,
                    "var_x": varx,
                    "cov_xy": cov_xy,
                    "slope": slope,
                    "corr_xy": corr,
                    "references": pair_meta[(source, target)]["references"],
                    "n_references": pair_meta[(source, target)]["n_references"],
                }
            )

        slopes_arr = np.asarray(slopes, dtype=float)
        varx_arr = np.asarray(varx_values, dtype=float)
        inv_varx = 1.0 / np.clip(varx_arr, 1e-12, None)

        beta_hat, c_hat, rmse, r2 = fit_invariance_line(inv_varx, slopes_arr)
        preds = beta_hat + c_hat * inv_varx
        max_abs_resid = float(np.max(np.abs(slopes_arr - preds)))

        pairwise_betas = pairwise_beta_estimates(slopes_arr, varx_arr)
        pairwise_beta_sd = float(np.std(pairwise_betas, ddof=1)) if len(pairwise_betas) >= 2 else float("nan")
        pairwise_beta_mean = float(np.mean(pairwise_betas)) if pairwise_betas else float("nan")

        varx_ratio = float(np.max(varx_arr) / max(np.min(varx_arr), 1e-12))

        diagnostic_pass = (
            (r2 >= args.min_r2)
            and (rmse <= args.max_rmse)
            and (varx_ratio >= args.min_varx_ratio)
            and (np.isfinite(pairwise_beta_sd) and pairwise_beta_sd <= args.max_pairwise_beta_sd)
        )

        edge_fit_rows.append(
            {
                "source": source,
                "target": target,
                "references": pair_meta[(source, target)]["references"],
                "n_references": pair_meta[(source, target)]["n_references"],
                "n_env": len(env_order),
                "beta_hat": beta_hat,
                "c_hat": c_hat,
                "rmse": rmse,
                "r2": r2,
                "max_abs_residual": max_abs_resid,
                "varx_ratio": varx_ratio,
                "pairwise_beta_mean": pairwise_beta_mean,
                "pairwise_beta_sd": pairwise_beta_sd,
                "diagnostic_pass": int(diagnostic_pass),
            }
        )

    edge_env_df = pd.DataFrame(edge_env_rows)
    edge_fit_df = pd.DataFrame(edge_fit_rows).sort_values(["diagnostic_pass", "r2", "rmse"], ascending=[False, False, True])

    edge_env_df.to_csv(output_dir / "tissue_edge_environment_stats.csv", index=False)
    edge_fit_df.to_csv(output_dir / "tissue_edge_invariance_fits.csv", index=False)

    # Aggregate summary for manuscript traceability.
    summary = {
        "n_pairs_input": int(pairs.shape[0]),
        "n_pairs_common_across_env": int(len(common_edges)),
        "n_env": int(len(env_order)),
        "env_order": env_order,
        "pass_thresholds": {
            "min_r2": args.min_r2,
            "max_rmse": args.max_rmse,
            "min_varx_ratio": args.min_varx_ratio,
            "max_pairwise_beta_sd": args.max_pairwise_beta_sd,
        },
        "n_pass": int(edge_fit_df["diagnostic_pass"].sum()) if not edge_fit_df.empty else 0,
        "pass_fraction": float(edge_fit_df["diagnostic_pass"].mean()) if not edge_fit_df.empty else float("nan"),
        "median_r2": float(edge_fit_df["r2"].median()) if not edge_fit_df.empty else float("nan"),
        "median_rmse": float(edge_fit_df["rmse"].median()) if not edge_fit_df.empty else float("nan"),
        "median_varx_ratio": float(edge_fit_df["varx_ratio"].median()) if not edge_fit_df.empty else float("nan"),
    }
    write_json(output_dir / "tissue_invariance_summary.json", summary)

    # Figure 1: R^2 distribution across edges.
    fig, ax = plt.subplots(figsize=(5.5, 4))
    if not edge_fit_df.empty:
        ax.hist(edge_fit_df["r2"], bins=20, color="#3a6ea5", alpha=0.85)
        ax.axvline(args.min_r2, color="black", linestyle="--", linewidth=1.0, label="Pass threshold")
    ax.set_xlabel("R^2 for slope ~ (1 / VarX)")
    ax.set_ylabel("Edge count")
    ax.set_title("Cross-Tissue Invariance Fit Quality")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "fig06_tissue_invariance_r2_hist.png", dpi=220)
    plt.close(fig)

    # Figure 2: Observed vs fitted slopes pooled over edge-environment points.
    fit_lookup = {
        (row["source"], row["target"]): (float(row["beta_hat"]), float(row["c_hat"]))
        for _, row in edge_fit_df.iterrows()
    }
    pooled_obs: List[float] = []
    pooled_pred: List[float] = []
    for _, row in edge_env_df.iterrows():
        key = (str(row["source"]), str(row["target"]))
        beta_hat, c_hat = fit_lookup[key]
        pred = beta_hat + c_hat / float(row["var_x"])
        pooled_obs.append(float(row["slope"]))
        pooled_pred.append(pred)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    if pooled_obs:
        ax.scatter(pooled_obs, pooled_pred, s=12, alpha=0.6, color="#2f7f5f")
        min_v = min(min(pooled_obs), min(pooled_pred))
        max_v = max(max(pooled_obs), max(pooled_pred))
        ax.plot([min_v, max_v], [min_v, max_v], color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Observed slope")
    ax.set_ylabel("Invariance-model fitted slope")
    ax.set_title("Observed vs Fitted Slopes (All Edges x Environments)")
    fig.tight_layout()
    fig.savefig(figure_dir / "fig07_tissue_observed_vs_fitted_slopes.png", dpi=220)
    plt.close(fig)

    # Figure 3: Example edges (two best-fit and two worst-fit).
    example_edges: List[Tuple[str, str]] = []
    if not edge_fit_df.empty:
        best = edge_fit_df.sort_values(["r2", "rmse"], ascending=[False, True]).head(2)
        worst = edge_fit_df.sort_values(["r2", "rmse"], ascending=[True, False]).head(2)
        example_edges = list(zip(best["source"], best["target"])) + list(zip(worst["source"], worst["target"]))

    if example_edges:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes_flat = list(axes.ravel())
        for ax, (source, target) in zip(axes_flat, example_edges):
            edge_rows = edge_env_df[(edge_env_df["source"] == source) & (edge_env_df["target"] == target)].copy()
            fit_row = edge_fit_df[(edge_fit_df["source"] == source) & (edge_fit_df["target"] == target)].iloc[0]

            x_vals = 1.0 / edge_rows["var_x"].to_numpy(dtype=float)
            y_vals = edge_rows["slope"].to_numpy(dtype=float)
            beta_hat = float(fit_row["beta_hat"])
            c_hat = float(fit_row["c_hat"])

            order = np.argsort(x_vals)
            x_sorted = x_vals[order]
            y_fit = beta_hat + c_hat * x_sorted

            ax.scatter(x_vals, y_vals, color="#3a6ea5", s=35)
            for _, erow in edge_rows.iterrows():
                ax.text(1.0 / float(erow["var_x"]), float(erow["slope"]), str(erow["environment"]), fontsize=7)
            ax.plot(x_sorted, y_fit, color="#cc5500", linewidth=1.4)
            ax.set_title(
                f"{source}->{target} | R2={fit_row['r2']:.2f}, RMSE={fit_row['rmse']:.3f}",
                fontsize=9,
            )
            ax.set_xlabel("1 / Var(X)")
            ax.set_ylabel("Observed slope")

        fig.tight_layout()
        fig.savefig(figure_dir / "fig08_tissue_edge_examples.png", dpi=220)
        plt.close(fig)


if __name__ == "__main__":
    main()
