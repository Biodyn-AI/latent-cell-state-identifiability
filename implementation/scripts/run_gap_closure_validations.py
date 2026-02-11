#!/usr/bin/env python3
"""Run targeted gap-closure validations for submission readiness.

This script addresses three remaining paper-quality gaps:
1) Targeted perturbation-backed validation for pass/fail edges.
2) Pathway-level biological validation (Enrichr Hallmark + GO BP).
3) Cell-state-level validation via within-cellstate slope consistency.

Outputs are written under:
`implementation/outputs/gap_closure_validations/`
and figures under:
`reports/figures/`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy import sparse
from scipy.stats import mannwhitneyu, norm


def to_dense(x: np.ndarray | sparse.spmatrix) -> np.ndarray:
    """Convert sparse/dense array-like to dense ndarray."""
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Compute OLS slope for y ~ 1 + x."""
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    var_x = float(np.mean(x_centered * x_centered))
    if var_x <= 1e-12:
        return float("nan")
    cov_xy = float(np.mean(x_centered * y_centered))
    return cov_xy / var_x


def parse_perturb_source(dataset_name: str, obs: pd.DataFrame) -> pd.Series:
    """Return per-cell perturbation source gene with 'control' for controls.

    Parsing is dataset-specific based on observed metadata conventions.
    """
    if dataset_name == "adamson":
        cond = obs["condition"].astype(str)
        control = obs["control"].astype(str).isin(["1", "True", "true"])
        source = pd.Series(np.where(control, "control", cond.str.split("+", n=1).str[0]), index=obs.index)
        return source

    # Dixit / Dixit7 / Shifrut: use condition label and keep single perturbations.
    cond = obs["condition"].astype(str)
    source = cond.where(cond.notna(), "control")
    source = source.where(source != "nan", "control")

    if "nperts" in obs.columns:
        nperts = pd.to_numeric(obs["nperts"], errors="coerce").fillna(1.0)
        source = pd.Series(np.where((source != "control") & (nperts == 1.0), source, "control"), index=obs.index)
    return source


def parse_path_specs(specs: Sequence[str], workspace_root: Path) -> Dict[str, Path]:
    """Parse name=path specs into resolved Path mappings."""
    mapping: Dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid path spec (missing '='): {spec}")
        name, raw_path = spec.split("=", 1)
        key = name.strip()
        path = Path(raw_path.strip())
        if not key:
            raise ValueError(f"Invalid path spec (empty name): {spec}")
        mapping[key] = path if path.is_absolute() else workspace_root / path
    return mapping


def compute_targeted_perturbation_validation(
    edge_fit_path: Path,
    output_dir: Path,
    figure_dir: Path,
    perturb_paths: Dict[str, Path],
) -> Dict[str, object]:
    """Compute perturbation deltas for exact paper pass/fail edges.

    For each available perturbation dataset and overlapping source TFs, compute
    target-wise delta = mean(perturbed) - mean(control) for edges in the paper set.
    """
    edge_df = pd.read_csv(edge_fit_path)
    edge_map: Dict[Tuple[str, str], Dict[str, float]] = {}
    source_to_targets: Dict[str, set[str]] = {}

    for row in edge_df.itertuples():
        key = (str(row.source), str(row.target))
        edge_map[key] = {
            "beta_hat": float(row.beta_hat_full),
            "diagnostic_pass": int(row.diagnostic_pass),
            "r2": float(row.r2),
            "rmse": float(row.rmse),
        }
        source_to_targets.setdefault(str(row.source), set()).add(str(row.target))

    rows: List[Dict[str, object]] = []

    for dataset_name, h5ad_path in perturb_paths.items():
        if not h5ad_path.exists():
            continue

        adata = ad.read_h5ad(h5ad_path)
        obs = adata.obs.copy()
        source_series = parse_perturb_source(dataset_name, obs)

        control_mask = (source_series == "control").to_numpy()
        n_control = int(control_mask.sum())
        if n_control < 100:
            # Skip pathological control availability.
            continue

        available_sources = sorted(set(source_series[source_series != "control"]) & set(source_to_targets))
        if not available_sources:
            continue

        var_index = {gene: idx for idx, gene in enumerate(adata.var_names)}

        for source_gene in available_sources:
            pert_mask = (source_series == source_gene).to_numpy()
            n_pert = int(pert_mask.sum())
            if n_pert < 50:
                continue

            candidate_targets = sorted([t for t in source_to_targets[source_gene] if t in var_index])
            if not candidate_targets:
                continue

            target_idx = [var_index[t] for t in candidate_targets]
            expr = to_dense(adata[:, target_idx].X)
            if expr.ndim == 1:
                expr = expr.reshape(-1, 1)

            pert_values = expr[pert_mask, :]
            ctrl_values = expr[control_mask, :]

            pert_mean = pert_values.mean(axis=0)
            ctrl_mean = ctrl_values.mean(axis=0)
            delta = pert_mean - ctrl_mean

            pert_var = pert_values.var(axis=0)
            ctrl_var = ctrl_values.var(axis=0)
            se = np.sqrt((pert_var / max(n_pert, 1)) + (ctrl_var / max(n_control, 1)))
            z = np.divide(delta, np.where(se <= 1e-12, np.nan, se))
            p = 2.0 * (1.0 - norm.cdf(np.abs(z)))

            pooled_sd = np.sqrt((pert_var + ctrl_var) / 2.0)
            cohen_d = np.divide(delta, np.where(pooled_sd <= 1e-12, np.nan, pooled_sd))

            for idx, target_gene in enumerate(candidate_targets):
                key = (source_gene, target_gene)
                meta = edge_map[key]
                beta_hat = float(meta["beta_hat"])
                perturb_delta = float(delta[idx])

                # Sign concordance is defined on non-zero directional estimates.
                if abs(beta_hat) <= 1e-12 or abs(perturb_delta) <= 1e-12:
                    sign_concordant = np.nan
                else:
                    sign_concordant = float(np.sign(beta_hat) == np.sign(perturb_delta))

                rows.append(
                    {
                        "dataset": dataset_name,
                        "source": source_gene,
                        "target": target_gene,
                        "diagnostic_pass": int(meta["diagnostic_pass"]),
                        "beta_hat": beta_hat,
                        "r2": float(meta["r2"]),
                        "rmse": float(meta["rmse"]),
                        "n_pert": n_pert,
                        "n_control": n_control,
                        "perturb_delta": perturb_delta,
                        "abs_perturb_delta": abs(perturb_delta),
                        "cohen_d": float(cohen_d[idx]),
                        "z_score": float(z[idx]),
                        "p_value": float(p[idx]),
                        "sign_concordant": sign_concordant,
                    }
                )

    perturb_df = pd.DataFrame(rows)
    perturb_path = output_dir / "targeted_perturbation_edge_effects.csv"
    perturb_df.to_csv(perturb_path, index=False)

    if perturb_df.empty:
        summary = {
            "n_measurements": 0,
            "message": "No overlap between paper edges and perturbation sources/targets in available datasets.",
        }
        with (output_dir / "targeted_perturbation_summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2)
        return summary

    summary_rows: List[Dict[str, object]] = []
    for group_name, group_df in perturb_df.groupby("diagnostic_pass"):
        label = "pass" if int(group_name) == 1 else "fail"
        concordant = group_df["sign_concordant"].dropna()

        summary_rows.append(
            {
                "group": label,
                "n_measurements": int(group_df.shape[0]),
                "n_unique_edges": int(group_df[["source", "target"]].drop_duplicates().shape[0]),
                "median_abs_perturb_delta": float(group_df["abs_perturb_delta"].median()),
                "median_abs_cohen_d": float(group_df["cohen_d"].abs().median()),
                "sign_concordance_rate": float(concordant.mean()) if not concordant.empty else float("nan"),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "targeted_perturbation_summary_by_group.csv", index=False)

    # Statistical comparison: pass vs fail abs delta and sign concordance.
    pass_df = perturb_df[perturb_df["diagnostic_pass"] == 1]
    fail_df = perturb_df[perturb_df["diagnostic_pass"] == 0]

    if not pass_df.empty and not fail_df.empty:
        mw = mannwhitneyu(
            pass_df["abs_perturb_delta"].to_numpy(),
            fail_df["abs_perturb_delta"].to_numpy(),
            alternative="two-sided",
        )
        abs_delta_p = float(mw.pvalue)
    else:
        abs_delta_p = float("nan")

    concordant_all = perturb_df["sign_concordant"].dropna()
    if not concordant_all.empty:
        sign_concordance_overall = float(concordant_all.mean())
    else:
        sign_concordance_overall = float("nan")

    summary = {
        "n_measurements": int(perturb_df.shape[0]),
        "n_unique_edges_measured": int(perturb_df[["source", "target"]].drop_duplicates().shape[0]),
        "datasets_used": sorted(perturb_df["dataset"].unique().tolist()),
        "pass_measurements": int(pass_df.shape[0]),
        "fail_measurements": int(fail_df.shape[0]),
        "median_abs_delta_pass": float(pass_df["abs_perturb_delta"].median()) if not pass_df.empty else float("nan"),
        "median_abs_delta_fail": float(fail_df["abs_perturb_delta"].median()) if not fail_df.empty else float("nan"),
        "abs_delta_mannwhitney_p": abs_delta_p,
        "overall_sign_concordance": sign_concordance_overall,
    }

    with (output_dir / "targeted_perturbation_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    # Figure 11: perturb delta vs beta estimate.
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    for pass_value, color, label in [(1, "#1f77b4", "Pass"), (0, "#d62728", "Fail")]:
        group = perturb_df[perturb_df["diagnostic_pass"] == pass_value]
        if group.empty:
            continue
        ax.scatter(group["beta_hat"], group["perturb_delta"], s=28, alpha=0.8, color=color, label=label)

    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Invariance beta estimate")
    ax.set_ylabel("Perturbation delta (perturbed - control)")
    ax.set_title("Targeted Perturbation Validation for Paper Edges")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "fig11_targeted_perturbation_beta_vs_delta.png", dpi=220)
    plt.close(fig)

    return summary


def enrichr_query(genes: Sequence[str], gene_set_library: str) -> pd.DataFrame:
    """Query Enrichr and return a tidy enrichment DataFrame.

    Uses the public Enrichr API endpoints.
    """
    gene_list = [g for g in genes if isinstance(g, str) and g]
    if len(gene_list) < 3:
        return pd.DataFrame()

    add_url = "https://maayanlab.cloud/Enrichr/addList"
    payload = {
        "list": (None, "\n".join(gene_list)),
        "description": (None, "gap_closure_validation"),
    }
    add_resp = requests.post(add_url, files=payload, timeout=30)
    add_resp.raise_for_status()
    user_list_id = add_resp.json()["userListId"]

    enrich_url = "https://maayanlab.cloud/Enrichr/enrich"
    enrich_resp = requests.get(
        enrich_url,
        params={"userListId": user_list_id, "backgroundType": gene_set_library},
        timeout=30,
    )
    enrich_resp.raise_for_status()

    data = enrich_resp.json().get(gene_set_library, [])
    rows = []
    # Enrichr columns documented as:
    # [rank, term_name, pvalue, zscore, combined_score, overlapping_genes,
    #  adjusted_pvalue, old_pvalue, old_adjusted_pvalue]
    for rec in data:
        if len(rec) < 7:
            continue
        rows.append(
            {
                "rank": int(rec[0]),
                "term": str(rec[1]),
                "p_value": float(rec[2]),
                "z_score": float(rec[3]),
                "combined_score": float(rec[4]),
                "overlap_genes": str(rec[5]),
                "adjusted_p_value": float(rec[6]),
            }
        )
    return pd.DataFrame(rows)


def compute_pathway_validation(
    edge_fit_path: Path,
    output_dir: Path,
    figure_dir: Path,
) -> Dict[str, object]:
    """Run pathway enrichment for pass and fail target sets."""
    edge_df = pd.read_csv(edge_fit_path)
    pass_targets = sorted(set(edge_df.loc[edge_df["diagnostic_pass"] == 1, "target"].astype(str)))
    fail_targets = sorted(set(edge_df.loc[edge_df["diagnostic_pass"] == 0, "target"].astype(str)))

    libraries = ["MSigDB_Hallmark_2020", "GO_Biological_Process_2023"]

    summary: Dict[str, object] = {
        "n_pass_targets": len(pass_targets),
        "n_fail_targets": len(fail_targets),
        "libraries": libraries,
    }

    top_plot_df: List[pd.DataFrame] = []

    for label, genes in [("pass", pass_targets), ("fail", fail_targets)]:
        for lib in libraries:
            df = enrichr_query(genes=genes, gene_set_library=lib)
            out_path = output_dir / f"pathway_enrichment_{label}_{lib}.tsv"
            if df.empty:
                df.to_csv(out_path, sep="\t", index=False)
                summary[f"{label}_{lib}_n_significant_fdr_0_1"] = 0
                continue

            df = df.sort_values("adjusted_p_value", ascending=True).reset_index(drop=True)
            df.to_csv(out_path, sep="\t", index=False)

            sig = df[df["adjusted_p_value"] <= 0.10]
            summary[f"{label}_{lib}_n_significant_fdr_0_1"] = int(sig.shape[0])

            # Keep top terms for plotting Hallmark only.
            if lib == "MSigDB_Hallmark_2020":
                top = df.head(8).copy()
                top["group"] = label
                top["neg_log10_fdr"] = -np.log10(np.clip(top["adjusted_p_value"].to_numpy(), 1e-300, None))
                top_plot_df.append(top)

    # Figure 12: top hallmark terms for pass/fail sets.
    if top_plot_df:
        plot_df = pd.concat(top_plot_df, ignore_index=True)
        # Keep a concise combined panel.
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
        for ax, group in zip(axes, ["pass", "fail"]):
            sub = plot_df[plot_df["group"] == group].copy()
            sub = sub.sort_values("neg_log10_fdr", ascending=True)
            ax.barh(sub["term"], sub["neg_log10_fdr"], color="#2ca02c" if group == "pass" else "#ff7f0e")
            ax.set_title(f"{group.capitalize()} targets: Hallmark enrichment")
            ax.set_xlabel("-log10(FDR)")
        fig.tight_layout()
        fig.savefig(figure_dir / "fig12_hallmark_enrichment_pass_fail.png", dpi=220)
        plt.close(fig)

    return summary


def compute_cellstate_validation(
    edge_fit_path: Path,
    output_dir: Path,
    figure_dir: Path,
    min_cells_per_state: int,
    top_states_per_env: int,
    env_paths: Dict[str, Path],
) -> Dict[str, object]:
    """Compute cell-state-specific slope consistency for pass vs fail edges."""
    edge_df = pd.read_csv(edge_fit_path)
    edge_info = {
        (str(r.source), str(r.target)): {
            "diagnostic_pass": int(r.diagnostic_pass),
            "beta_hat_full": float(r.beta_hat_full),
        }
        for r in edge_df.itertuples()
    }

    edge_state_rows: List[Dict[str, object]] = []
    edge_env_summary_rows: List[Dict[str, object]] = []

    for env_name, h5ad_path in env_paths.items():
        if not h5ad_path.exists():
            continue
        adata = ad.read_h5ad(h5ad_path)

        # Prefer broad classes when available for robustness; fallback to cell_type.
        if "broad_cell_class" in adata.obs.columns and adata.obs["broad_cell_class"].notna().sum() > 0:
            state_col = "broad_cell_class"
        else:
            state_col = "cell_type"

        states = adata.obs[state_col].astype(str)
        counts = states.value_counts()
        chosen_states = counts[counts >= min_cells_per_state].head(top_states_per_env).index.tolist()
        if not chosen_states:
            continue

        state_masks = {state: (states == state).to_numpy() for state in chosen_states}
        var_index = {gene: idx for idx, gene in enumerate(adata.var_names)}

        for (source, target), meta in edge_info.items():
            if source not in var_index or target not in var_index:
                continue

            x = to_dense(adata[:, source].X).ravel()
            y = to_dense(adata[:, target].X).ravel()

            state_slopes: List[float] = []
            for state in chosen_states:
                mask = state_masks[state]
                if int(mask.sum()) < min_cells_per_state:
                    continue
                slope = ols_slope(x[mask], y[mask])
                if np.isnan(slope):
                    continue

                state_slopes.append(float(slope))
                edge_state_rows.append(
                    {
                        "environment": env_name,
                        "state_col": state_col,
                        "state": state,
                        "source": source,
                        "target": target,
                        "diagnostic_pass": int(meta["diagnostic_pass"]),
                        "beta_hat_full": float(meta["beta_hat_full"]),
                        "state_slope": float(slope),
                        "abs_state_slope": abs(float(slope)),
                        "n_cells_state": int(mask.sum()),
                    }
                )

            if len(state_slopes) >= 2:
                arr = np.asarray(state_slopes)
                pos_frac = float(np.mean(arr > 0.0))
                neg_frac = float(np.mean(arr < 0.0))
                sign_consistency = max(pos_frac, neg_frac)
                slope_cv = float(np.std(arr, ddof=1) / (abs(np.mean(arr)) + 1e-12))
                edge_env_summary_rows.append(
                    {
                        "environment": env_name,
                        "source": source,
                        "target": target,
                        "diagnostic_pass": int(meta["diagnostic_pass"]),
                        "state_count": int(len(arr)),
                        "state_sign_consistency": sign_consistency,
                        "state_slope_mean": float(np.mean(arr)),
                        "state_slope_sd": float(np.std(arr, ddof=1)),
                        "state_slope_cv": slope_cv,
                    }
                )

    edge_state_df = pd.DataFrame(edge_state_rows)
    edge_env_df = pd.DataFrame(edge_env_summary_rows)

    edge_state_df.to_csv(output_dir / "cellstate_edge_slopes.tsv", sep="\t", index=False)
    edge_env_df.to_csv(output_dir / "cellstate_edge_environment_summary.tsv", sep="\t", index=False)

    summary: Dict[str, object] = {
        "n_state_level_rows": int(edge_state_df.shape[0]),
        "n_edge_env_rows": int(edge_env_df.shape[0]),
    }

    if not edge_env_df.empty:
        pass_vals = edge_env_df.loc[edge_env_df["diagnostic_pass"] == 1, "state_sign_consistency"].to_numpy()
        fail_vals = edge_env_df.loc[edge_env_df["diagnostic_pass"] == 0, "state_sign_consistency"].to_numpy()

        if len(pass_vals) > 0 and len(fail_vals) > 0:
            mw = mannwhitneyu(pass_vals, fail_vals, alternative="two-sided")
            summary["consistency_mannwhitney_p"] = float(mw.pvalue)
        else:
            summary["consistency_mannwhitney_p"] = float("nan")

        summary.update(
            {
                "pass_median_state_sign_consistency": float(np.median(pass_vals)) if len(pass_vals) else float("nan"),
                "fail_median_state_sign_consistency": float(np.median(fail_vals)) if len(fail_vals) else float("nan"),
                "pass_median_state_slope_cv": float(
                    edge_env_df.loc[edge_env_df["diagnostic_pass"] == 1, "state_slope_cv"].median()
                )
                if len(pass_vals)
                else float("nan"),
                "fail_median_state_slope_cv": float(
                    edge_env_df.loc[edge_env_df["diagnostic_pass"] == 0, "state_slope_cv"].median()
                )
                if len(fail_vals)
                else float("nan"),
            }
        )

        fig, ax = plt.subplots(figsize=(5.0, 4.0))
        data = [pass_vals, fail_vals]
        bp = ax.boxplot(data, tick_labels=["Pass", "Fail"], patch_artist=True)
        for patch, color in zip(bp["boxes"], ["#1f77b4", "#d62728"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        ax.set_ylabel("Cell-state sign consistency")
        ax.set_title("Within-Environment Cell-State Consistency")
        fig.tight_layout()
        fig.savefig(figure_dir / "fig13_cellstate_sign_consistency_pass_fail.png", dpi=220)
        plt.close(fig)

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    workspace_root = Path(__file__).resolve().parents[3]
    parser.add_argument(
        "--edge-fit-path",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "outputs"
        / "tissue_invariance_quality"
        / "edge_fit_with_loo_and_pass.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "gap_closure_validations",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "reports" / "figures",
    )
    parser.add_argument("--min-cells-per-state", type=int, default=200)
    parser.add_argument("--top-states-per-env", type=int, default=8)
    parser.add_argument(
        "--perturb-spec",
        nargs="+",
        default=[
            "adamson=single_cell_mechinterp/data/perturb/adamson/perturb_processed_symbols.h5ad",
            "dixit=single_cell_mechinterp/data/perturb/dixit/perturb_processed_symbols.h5ad",
            "dixit7=single_cell_mechinterp/data/perturb/dixit_7_days/perturb_processed_symbols.h5ad",
            "shifrut=single_cell_mechinterp/data/perturb/shifrut/perturb_processed_symbols.h5ad",
        ],
        help=(
            "Perturbation dataset specs as name=path. Relative paths are "
            f"resolved against {workspace_root}."
        ),
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
        help=(
            "Environment dataset specs as name=path. Relative paths are "
            f"resolved against {workspace_root}."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    workspace_root = Path(__file__).resolve().parents[3]

    output_dir: Path = args.output_dir
    figure_dir: Path = args.figure_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    perturb_paths = parse_path_specs(args.perturb_spec, workspace_root=workspace_root)
    env_paths = parse_path_specs(args.env_spec, workspace_root=workspace_root)

    perturb_summary = compute_targeted_perturbation_validation(
        edge_fit_path=args.edge_fit_path,
        output_dir=output_dir,
        figure_dir=figure_dir,
        perturb_paths=perturb_paths,
    )

    pathway_summary = compute_pathway_validation(
        edge_fit_path=args.edge_fit_path,
        output_dir=output_dir,
        figure_dir=figure_dir,
    )

    cellstate_summary = compute_cellstate_validation(
        edge_fit_path=args.edge_fit_path,
        output_dir=output_dir,
        figure_dir=figure_dir,
        min_cells_per_state=args.min_cells_per_state,
        top_states_per_env=args.top_states_per_env,
        env_paths=env_paths,
    )

    summary = {
        "perturbation_validation": perturb_summary,
        "pathway_validation": pathway_summary,
        "cellstate_validation": cellstate_summary,
    }
    with (output_dir / "gap_closure_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
