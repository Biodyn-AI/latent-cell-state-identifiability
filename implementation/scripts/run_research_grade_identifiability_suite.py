#!/usr/bin/env python3
"""Run a research-grade adversarial identifiability benchmark suite.

This script expands Proposal 4 into a broader synthetic benchmark:
- constructive non-identifiability continuum,
- classifier indistinguishability under matched observational laws,
- adversarial stress scenarios for estimator robustness,
- anchor-quality phase diagrams,
- two-environment invariance stress tests.

Outputs are written to CSV/JSON and publication-oriented figures.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LinearSCM:
    """Linear-Gaussian SCM for non-identifiability constructions."""

    model_id: str
    beta: float
    alpha: float
    gamma: float
    sigma_x: float
    sigma_y: float
    lambda_r: float
    nu_r: float
    sigma_r: float


@dataclass(frozen=True)
class Scenario:
    """Config for adversarial observational/interventional simulation scenarios."""

    name: str
    description: str
    noise_kind: str = "gaussian"
    interaction_eta: float = 0.0
    measurement_error_x: float = 0.0
    selection_quantile: float | None = None


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Return OLS slope for y ~ 1 + x."""
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    var_x = float(np.mean(x_centered * x_centered))
    if var_x <= 1e-12:
        return float("nan")
    cov_xy = float(np.mean(x_centered * y_centered))
    return cov_xy / var_x


def ols_coef_x_with_anchor(x: np.ndarray, anchor: np.ndarray, y: np.ndarray) -> float:
    """Return coefficient on x for y ~ 1 + x + anchor."""
    design = np.column_stack([np.ones(x.size), x, anchor])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[1])


def standardized_t_noise(df: float, size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw Student-t noise rescaled to unit variance."""
    raw = rng.standard_t(df=df, size=size)
    scale = math.sqrt(df / (df - 2.0))
    return raw / scale


def sample_noise(kind: str, size: int, rng: np.random.Generator) -> np.ndarray:
    """Return zero-mean unit-variance noise for a named family."""
    if kind == "gaussian":
        return rng.normal(0.0, 1.0, size=size)
    if kind == "heavy_tail_t3":
        return standardized_t_noise(df=3.0, size=size, rng=rng)
    raise ValueError(f"Unsupported noise kind: {kind}")


def auc_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute ROC-AUC via rank-statistic (Mann-Whitney U) formula."""
    if scores.size != labels.size:
        raise ValueError("scores and labels must have the same length")
    positives = labels == 1
    n_pos = int(np.sum(positives))
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1.0, scores.size + 1.0)

    sum_ranks_pos = float(np.sum(ranks[positives]))
    u_stat = sum_ranks_pos - (n_pos * (n_pos + 1.0) / 2.0)
    return u_stat / (n_pos * n_neg)


def rmse(estimates: np.ndarray, truth: float) -> float:
    """Root-mean-square error around truth."""
    return float(np.sqrt(np.mean((estimates - truth) ** 2)))


def analytic_moments(model: LinearSCM) -> Dict[str, float]:
    """Closed-form second moments under linear-Gaussian SCM."""
    var_x = model.alpha**2 + model.sigma_x**2
    cov_xy = model.beta * var_x + model.gamma * model.alpha
    var_y = (
        model.beta**2 * var_x
        + model.gamma**2
        + 2.0 * model.beta * model.gamma * model.alpha
        + model.sigma_y**2
    )
    cov_xr = model.lambda_r * var_x + model.nu_r * cov_xy
    cov_yr = model.lambda_r * cov_xy + model.nu_r * var_y
    var_r = (
        model.lambda_r**2 * var_x
        + model.nu_r**2 * var_y
        + 2.0 * model.lambda_r * model.nu_r * cov_xy
        + model.sigma_r**2
    )
    return {
        "var_x": var_x,
        "cov_xy": cov_xy,
        "var_y": var_y,
        "cov_xr": cov_xr,
        "cov_yr": cov_yr,
        "var_r": var_r,
        "obs_slope": cov_xy / var_x,
    }


def analytic_covariance_xyzr(model: LinearSCM) -> np.ndarray:
    """Covariance matrix for [X, Y, R] under the model."""
    m = analytic_moments(model)
    return np.array(
        [
            [m["var_x"], m["cov_xy"], m["cov_xr"]],
            [m["cov_xy"], m["var_y"], m["cov_yr"]],
            [m["cov_xr"], m["cov_yr"], m["var_r"]],
        ]
    )


def matched_model(base: LinearSCM, beta_alt: float, model_id: str) -> LinearSCM | None:
    """Build an observationally-equivalent alternative model with changed beta.

    Returns None when positivity of sigma_y^2 cannot be satisfied.
    """
    if abs(base.alpha) < 1e-12:
        raise ValueError("alpha must be non-zero for matched-model construction")

    base_m = analytic_moments(base)
    var_x = base_m["var_x"]
    cov_xy_target = base_m["cov_xy"]
    var_y_target = base_m["var_y"]

    gamma_alt = (cov_xy_target - beta_alt * var_x) / base.alpha
    sigma_y_sq_alt = var_y_target - (
        beta_alt**2 * var_x + gamma_alt**2 + 2.0 * beta_alt * gamma_alt * base.alpha
    )
    if sigma_y_sq_alt <= 1e-12:
        return None

    return LinearSCM(
        model_id=model_id,
        beta=beta_alt,
        alpha=base.alpha,
        gamma=gamma_alt,
        sigma_x=base.sigma_x,
        sigma_y=math.sqrt(sigma_y_sq_alt),
        lambda_r=base.lambda_r,
        nu_r=base.nu_r,
        sigma_r=base.sigma_r,
    )


def simulate_xyzr(model: LinearSCM, n: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate matrix with columns [X, Y, R]."""
    z = rng.normal(0.0, 1.0, size=n)
    x = model.alpha * z + model.sigma_x * rng.normal(0.0, 1.0, size=n)
    y = model.beta * x + model.gamma * z + model.sigma_y * rng.normal(0.0, 1.0, size=n)
    r = model.lambda_r * x + model.nu_r * y + model.sigma_r * rng.normal(0.0, 1.0, size=n)
    return np.column_stack([x, y, r])


def empirical_covariance(matrix: np.ndarray) -> np.ndarray:
    """Empirical covariance with population normalization (divide by n)."""
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    return centered.T @ centered / matrix.shape[0]


def fit_lda(train_x: np.ndarray, train_y: np.ndarray) -> Tuple[np.ndarray, float]:
    """Fit a lightweight equal-covariance LDA classifier."""
    x0 = train_x[train_y == 0]
    x1 = train_x[train_y == 1]
    mu0 = x0.mean(axis=0)
    mu1 = x1.mean(axis=0)

    cov0 = empirical_covariance(x0)
    cov1 = empirical_covariance(x1)
    pooled = 0.5 * (cov0 + cov1) + 1e-6 * np.eye(train_x.shape[1])

    w = np.linalg.solve(pooled, mu1 - mu0)
    threshold = 0.5 * float(np.dot(w, mu1 + mu0))
    return w, threshold


def lda_scores(x: np.ndarray, w: np.ndarray, threshold: float) -> np.ndarray:
    """Compute signed distance from LDA decision boundary."""
    return x @ w - threshold


def simulate_observational(
    scenario: Scenario,
    n: int,
    beta: float,
    alpha: float,
    gamma: float,
    sigma_x: float,
    sigma_y: float,
    lambda_r: float,
    nu_r: float,
    sigma_r: float,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Simulate observational data with optional adversarial distortions."""
    z = rng.normal(0.0, 1.0, size=n)
    eps_x = sample_noise(scenario.noise_kind, size=n, rng=rng)
    eps_y = sample_noise(scenario.noise_kind, size=n, rng=rng)

    x_true = alpha * z + sigma_x * eps_x
    interaction_term = scenario.interaction_eta * x_true * z
    y = beta * x_true + gamma * z + interaction_term + sigma_y * eps_y

    x_obs = x_true + scenario.measurement_error_x * rng.normal(0.0, 1.0, size=n)

    # Selection is a deliberate adversarial violation: conditioning on readout
    # introduces collider-like bias and can break identification assumptions.
    if scenario.selection_quantile is not None:
        eps_r = sample_noise(scenario.noise_kind, size=n, rng=rng)
        r = lambda_r * x_true + nu_r * y + sigma_r * eps_r
        threshold = float(np.quantile(r, scenario.selection_quantile))
        mask = r >= threshold

        # Keep a valid subset even in pathological tails.
        if np.sum(mask) < 200:
            order = np.argsort(r)
            mask = np.zeros_like(r, dtype=bool)
            mask[order[-200:]] = True

        z = z[mask]
        x_true = x_true[mask]
        x_obs = x_obs[mask]
        y = y[mask]

    return {"z": z, "x_true": x_true, "x_obs": x_obs, "y": y}


def simulate_interventional(
    scenario: Scenario,
    n: int,
    beta: float,
    gamma: float,
    sigma_y: float,
    lambda_r: float,
    nu_r: float,
    sigma_r: float,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Simulate data under do(X) with optional measurement/selection distortions."""
    z = rng.normal(0.0, 1.0, size=n)

    # Randomized intervention input independent of z.
    x_true = rng.normal(0.0, 1.0, size=n)
    eps_y = sample_noise(scenario.noise_kind, size=n, rng=rng)
    interaction_term = scenario.interaction_eta * x_true * z
    y = beta * x_true + gamma * z + interaction_term + sigma_y * eps_y

    x_obs = x_true + scenario.measurement_error_x * rng.normal(0.0, 1.0, size=n)

    if scenario.selection_quantile is not None:
        eps_r = sample_noise(scenario.noise_kind, size=n, rng=rng)
        r = lambda_r * x_true + nu_r * y + sigma_r * eps_r
        threshold = float(np.quantile(r, scenario.selection_quantile))
        mask = r >= threshold
        if np.sum(mask) < 200:
            order = np.argsort(r)
            mask = np.zeros_like(r, dtype=bool)
            mask[order[-200:]] = True

        z = z[mask]
        x_true = x_true[mask]
        x_obs = x_obs[mask]
        y = y[mask]

    return {"z": z, "x_true": x_true, "x_obs": x_obs, "y": y}


def run_nonidentifiability_continuum(
    output_dir: Path,
    figure_dir: Path,
    n_obs: int,
    classifier_reps: int,
    n_classifier_per_class: int,
    rng: np.random.Generator,
) -> Dict[str, object]:
    """Run constructive non-identifiability continuum + classifier test."""
    base = LinearSCM(
        model_id="edge_present",
        beta=0.8,
        alpha=1.0,
        gamma=-0.3,
        sigma_x=1.0,
        sigma_y=1.0,
        lambda_r=0.55,
        nu_r=0.40,
        sigma_r=0.70,
    )

    beta_grid = np.linspace(-0.4, 1.2, 17)
    rows: List[Dict[str, float]] = []
    matched_models: List[LinearSCM] = []

    for beta_alt in beta_grid:
        model_alt = matched_model(base, beta_alt=float(beta_alt), model_id=f"alt_beta_{beta_alt:.2f}")
        if model_alt is None:
            rows.append(
                {
                    "beta_alt": float(beta_alt),
                    "feasible": 0.0,
                    "gamma_alt": float("nan"),
                    "sigma_y_alt": float("nan"),
                    "analytic_obs_slope_alt": float("nan"),
                    "analytic_obs_slope_base": analytic_moments(base)["obs_slope"],
                    "sample_obs_slope_alt": float("nan"),
                    "sample_obs_slope_base": float("nan"),
                    "analytic_max_abs_cov_diff": float("nan"),
                    "sample_max_abs_cov_diff": float("nan"),
                }
            )
            continue

        matched_models.append(model_alt)
        sim_base = simulate_xyzr(base, n_obs, rng)
        sim_alt = simulate_xyzr(model_alt, n_obs, rng)

        cov_base = empirical_covariance(sim_base)
        cov_alt = empirical_covariance(sim_alt)
        analytic_diff = np.abs(analytic_covariance_xyzr(base) - analytic_covariance_xyzr(model_alt))
        sample_diff = np.abs(cov_base - cov_alt)

        rows.append(
            {
                "beta_alt": float(beta_alt),
                "feasible": 1.0,
                "gamma_alt": model_alt.gamma,
                "sigma_y_alt": model_alt.sigma_y,
                "analytic_obs_slope_alt": analytic_moments(model_alt)["obs_slope"],
                "analytic_obs_slope_base": analytic_moments(base)["obs_slope"],
                "sample_obs_slope_alt": ols_slope(sim_alt[:, 0], sim_alt[:, 1]),
                "sample_obs_slope_base": ols_slope(sim_base[:, 0], sim_base[:, 1]),
                "analytic_max_abs_cov_diff": float(analytic_diff.max()),
                "sample_max_abs_cov_diff": float(sample_diff.max()),
            }
        )

    continuum_df = pd.DataFrame(rows)
    continuum_df.to_csv(output_dir / "synthetic_nonidentifiability_continuum.csv", index=False)

    # Choose the strongest contrast model for indistinguishability classification.
    # We prefer beta_alt closest to zero and feasible.
    feasible = [m for m in matched_models if abs(m.beta) <= 1e-8]
    alt_for_classifier = feasible[0] if feasible else min(matched_models, key=lambda m: abs(m.beta))

    cls_rows: List[Dict[str, float]] = []
    for rep in range(classifier_reps):
        samples_a = simulate_xyzr(base, n_classifier_per_class, rng)
        samples_b = simulate_xyzr(alt_for_classifier, n_classifier_per_class, rng)

        labels_a = np.zeros(n_classifier_per_class, dtype=int)
        labels_b = np.ones(n_classifier_per_class, dtype=int)

        x = np.vstack([samples_a, samples_b])
        y = np.concatenate([labels_a, labels_b])

        perm = rng.permutation(x.shape[0])
        x = x[perm]
        y = y[perm]

        split = x.shape[0] // 2
        train_x, test_x = x[:split], x[split:]
        train_y, test_y = y[:split], y[split:]

        w, threshold = fit_lda(train_x, train_y)
        scores = lda_scores(test_x, w, threshold)
        preds = (scores >= 0.0).astype(int)

        auc = auc_from_scores(scores, test_y)
        acc = float(np.mean(preds == test_y))

        cls_rows.append(
            {
                "rep": rep,
                "auc": auc,
                "accuracy": acc,
                "beta_base": base.beta,
                "beta_alt": alt_for_classifier.beta,
                "gamma_alt": alt_for_classifier.gamma,
            }
        )

    cls_df = pd.DataFrame(cls_rows)
    cls_df.to_csv(output_dir / "synthetic_nonidentifiability_classifier_replicates.csv", index=False)

    cls_summary = {
        "auc_mean": float(cls_df["auc"].mean()),
        "auc_sd": float(cls_df["auc"].std(ddof=1)),
        "accuracy_mean": float(cls_df["accuracy"].mean()),
        "accuracy_sd": float(cls_df["accuracy"].std(ddof=1)),
        "beta_base": base.beta,
        "beta_alt": alt_for_classifier.beta,
        "gamma_alt": alt_for_classifier.gamma,
        "classifier_reps": classifier_reps,
    }
    pd.DataFrame([cls_summary]).to_csv(
        output_dir / "synthetic_nonidentifiability_classifier_summary.csv", index=False
    )

    # Figure: continuum construction and observed slope invariance.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    feas_df = continuum_df[continuum_df["feasible"] == 1.0]
    axes[0].plot(feas_df["beta_alt"], feas_df["gamma_alt"], marker="o", linewidth=1.5)
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title("Matched Gamma vs Beta")
    axes[0].set_xlabel("Alternative true beta")
    axes[0].set_ylabel("Solved gamma")

    axes[1].plot(feas_df["beta_alt"], feas_df["sigma_y_alt"], marker="o", color="#cc5500", linewidth=1.5)
    axes[1].set_title("Noise Rescaling for Equivalence")
    axes[1].set_xlabel("Alternative true beta")
    axes[1].set_ylabel("Solved sigma_y")

    axes[2].plot(
        feas_df["beta_alt"],
        feas_df["sample_obs_slope_base"],
        marker="o",
        linewidth=1.2,
        label="Sample obs slope (base)",
    )
    axes[2].plot(
        feas_df["beta_alt"],
        feas_df["sample_obs_slope_alt"],
        marker="o",
        linewidth=1.2,
        label="Sample obs slope (alt)",
    )
    axes[2].axhline(
        analytic_moments(base)["obs_slope"],
        color="black",
        linestyle="--",
        linewidth=0.8,
        label="Analytic obs slope",
    )
    axes[2].set_title("Observed Slope Invariance")
    axes[2].set_xlabel("Alternative true beta")
    axes[2].set_ylabel("Observed OLS slope")
    axes[2].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(figure_dir / "fig01_nonidentifiability_continuum.png", dpi=220)
    plt.close(fig)

    # Figure: indistinguishability classifier performance.
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(cls_df["auc"], bins=20, color="#2f7f5f", alpha=0.8)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.0, label="Random guessing")
    ax.set_xlabel("AUC")
    ax.set_ylabel("Count")
    ax.set_title("Classifier AUC Under Observational Equivalence")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "fig02_nonidentifiability_classifier_auc.png", dpi=220)
    plt.close(fig)

    return {
        "continuum_rows": int(continuum_df.shape[0]),
        "continuum_feasible_rows": int(feas_df.shape[0]),
        "classifier_auc_mean": cls_summary["auc_mean"],
        "classifier_accuracy_mean": cls_summary["accuracy_mean"],
        "classifier_alt_beta": cls_summary["beta_alt"],
    }


def run_estimator_benchmark(
    output_dir: Path,
    figure_dir: Path,
    reps: int,
    n_obs: int,
    n_int: int,
    anchor_noises: Sequence[float],
    rng: np.random.Generator,
) -> Dict[str, object]:
    """Run adversarial estimator benchmark across multiple scenarios."""
    beta = 0.80
    alpha = 1.00
    gamma = -0.30
    sigma_x = 1.00
    sigma_y = 1.00
    lambda_r = 0.55
    nu_r = 0.40
    sigma_r = 0.70

    scenarios = [
        Scenario(
            name="linear_gaussian",
            description="Baseline linear SCM with Gaussian noise.",
            noise_kind="gaussian",
        ),
        Scenario(
            name="heavy_tail_noise",
            description="Linear SCM with heavy-tailed t3 noise.",
            noise_kind="heavy_tail_t3",
        ),
        Scenario(
            name="nonlinear_interaction",
            description="Adds x*z interaction in outcome equation.",
            interaction_eta=0.35,
            noise_kind="gaussian",
        ),
        Scenario(
            name="measurement_error_x",
            description="Observes x with additive measurement error.",
            measurement_error_x=0.70,
            noise_kind="gaussian",
        ),
        Scenario(
            name="selection_on_readout",
            description="Condition on high-R subset inducing selection bias.",
            selection_quantile=0.60,
            noise_kind="gaussian",
        ),
    ]

    rep_rows: List[Dict[str, float]] = []

    for scenario in scenarios:
        for rep in range(reps):
            obs = simulate_observational(
                scenario=scenario,
                n=n_obs,
                beta=beta,
                alpha=alpha,
                gamma=gamma,
                sigma_x=sigma_x,
                sigma_y=sigma_y,
                lambda_r=lambda_r,
                nu_r=nu_r,
                sigma_r=sigma_r,
                rng=rng,
            )

            obs_naive = ols_slope(obs["x_obs"], obs["y"])
            rep_rows.append(
                {
                    "scenario": scenario.name,
                    "estimator": "naive_observational",
                    "rep": rep,
                    "estimate": obs_naive,
                    "truth_beta": beta,
                    "n_used": int(obs["x_obs"].size),
                }
            )

            for noise in anchor_noises:
                anchor = obs["z"] + noise * rng.normal(0.0, 1.0, size=obs["z"].size)
                est = ols_coef_x_with_anchor(obs["x_obs"], anchor, obs["y"])
                rep_rows.append(
                    {
                        "scenario": scenario.name,
                        "estimator": f"anchor_noise_{noise:.2f}",
                        "rep": rep,
                        "estimate": est,
                        "truth_beta": beta,
                        "n_used": int(obs["x_obs"].size),
                    }
                )

            interventional = simulate_interventional(
                scenario=scenario,
                n=n_int,
                beta=beta,
                gamma=gamma,
                sigma_y=sigma_y,
                lambda_r=lambda_r,
                nu_r=nu_r,
                sigma_r=sigma_r,
                rng=rng,
            )
            dox_est = ols_slope(interventional["x_obs"], interventional["y"])
            rep_rows.append(
                {
                    "scenario": scenario.name,
                    "estimator": "intervention_doX",
                    "rep": rep,
                    "estimate": dox_est,
                    "truth_beta": beta,
                    "n_used": int(interventional["x_obs"].size),
                }
            )

    rep_df = pd.DataFrame(rep_rows)
    rep_df.to_csv(output_dir / "synthetic_estimator_replicates.csv", index=False)

    summary_rows: List[Dict[str, float]] = []
    for (scenario, estimator), group in rep_df.groupby(["scenario", "estimator"]):
        estimates = group["estimate"].to_numpy(dtype=float)
        truth = float(group["truth_beta"].iloc[0])
        summary_rows.append(
            {
                "scenario": scenario,
                "estimator": estimator,
                "truth_beta": truth,
                "mean_estimate": float(np.mean(estimates)),
                "sd_estimate": float(np.std(estimates, ddof=1)),
                "mean_bias": float(np.mean(estimates - truth)),
                "rmse": rmse(estimates, truth),
                "q05": float(np.quantile(estimates, 0.05)),
                "q95": float(np.quantile(estimates, 0.95)),
                "avg_n_used": float(np.mean(group["n_used"])),
                "n_reps": int(group.shape[0]),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["scenario", "estimator"]).reset_index(drop=True)
    summary_df.to_csv(output_dir / "synthetic_estimator_summary.csv", index=False)

    # RMSE heatmap: scenario x estimator for quick adverse-condition comparison.
    pivot = summary_df.pivot(index="scenario", columns="estimator", values="rmse")
    fig, ax = plt.subplots(figsize=(11, 3.8))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(pivot.shape[1]), labels=list(pivot.columns), rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(pivot.shape[0]), labels=list(pivot.index), fontsize=9)
    ax.set_title("Estimator RMSE Across Adversarial Scenarios")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.to_numpy()[i, j]
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=7, color="black")

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(figure_dir / "fig03_estimator_rmse_heatmap.png", dpi=220)
    plt.close(fig)

    return {
        "n_scenarios": len(scenarios),
        "n_estimators": int(summary_df["estimator"].nunique()),
        "best_rmse": float(summary_df["rmse"].min()),
        "worst_rmse": float(summary_df["rmse"].max()),
    }


def run_anchor_phase_diagram(
    output_dir: Path,
    figure_dir: Path,
    phase_reps: int,
    phase_n: int,
    rng: np.random.Generator,
) -> Dict[str, object]:
    """Run grid study for anchor quality and confounding intensity."""
    beta = 0.80
    alpha = 1.00
    sigma_x = 1.00
    sigma_y = 1.00

    gamma_grid = np.linspace(0.0, 1.5, 7)
    anchor_noise_grid = np.linspace(0.0, 1.5, 7)

    rows: List[Dict[str, float]] = []
    base_scenario = Scenario(name="phase_linear", description="phase-grid")

    for gamma in gamma_grid:
        for anchor_noise in anchor_noise_grid:
            naive_estimates: List[float] = []
            anchor_estimates: List[float] = []

            for _ in range(phase_reps):
                obs = simulate_observational(
                    scenario=base_scenario,
                    n=phase_n,
                    beta=beta,
                    alpha=alpha,
                    gamma=float(gamma),
                    sigma_x=sigma_x,
                    sigma_y=sigma_y,
                    lambda_r=0.55,
                    nu_r=0.40,
                    sigma_r=0.70,
                    rng=rng,
                )
                naive_estimates.append(ols_slope(obs["x_obs"], obs["y"]))

                anchor = obs["z"] + anchor_noise * rng.normal(0.0, 1.0, size=obs["z"].size)
                anchor_estimates.append(ols_coef_x_with_anchor(obs["x_obs"], anchor, obs["y"]))

            naive_abs_bias = float(np.mean(np.abs(np.asarray(naive_estimates) - beta)))
            anchor_abs_bias = float(np.mean(np.abs(np.asarray(anchor_estimates) - beta)))

            rows.append(
                {
                    "gamma": float(gamma),
                    "anchor_noise": float(anchor_noise),
                    "naive_abs_bias": naive_abs_bias,
                    "anchor_abs_bias": anchor_abs_bias,
                    "improvement": naive_abs_bias - anchor_abs_bias,
                    "anchor_better": 1 if anchor_abs_bias < naive_abs_bias else 0,
                    "phase_reps": phase_reps,
                }
            )

    phase_df = pd.DataFrame(rows)
    phase_df.to_csv(output_dir / "synthetic_anchor_phase_diagram.csv", index=False)

    # Improvement heatmap (positive means anchor wins).
    pivot = phase_df.pivot(index="gamma", columns="anchor_noise", values="improvement")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    im = ax.imshow(pivot.to_numpy(), origin="lower", aspect="auto", cmap="RdYlGn")
    ax.set_xticks(np.arange(pivot.shape[1]), labels=[f"{x:.2f}" for x in pivot.columns], fontsize=8)
    ax.set_yticks(np.arange(pivot.shape[0]), labels=[f"{x:.2f}" for x in pivot.index], fontsize=8)
    ax.set_xlabel("Anchor noise scale")
    ax.set_ylabel("Confounding strength gamma")
    ax.set_title("Anchor Improvement (naive abs-bias minus anchor abs-bias)")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.to_numpy()[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(figure_dir / "fig04_anchor_phase_diagram.png", dpi=220)
    plt.close(fig)

    return {
        "gamma_grid_size": int(len(gamma_grid)),
        "anchor_noise_grid_size": int(len(anchor_noise_grid)),
        "mean_improvement": float(phase_df["improvement"].mean()),
        "fraction_anchor_better": float(phase_df["anchor_better"].mean()),
    }


def two_env_invariance_beta(slopes: Sequence[float], var_xs: Sequence[float]) -> float:
    """Recover beta from two environments under invariance assumptions."""
    s1, s2 = float(slopes[0]), float(slopes[1])
    v1, v2 = float(var_xs[0]), float(var_xs[1])
    denom = (1.0 / v1) - (1.0 / v2)
    if abs(denom) <= 1e-12:
        return float("nan")
    c = (s1 - s2) / denom
    return s1 - c / v1


def simulate_two_env(
    beta: float,
    gamma: float,
    alpha1: float,
    alpha2: float,
    sigma_x1: float,
    sigma_x2: float,
    sigma_y: float,
    n_per_env: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Simulate two environments and return naive + invariance estimates."""
    z1 = rng.normal(0.0, 1.0, size=n_per_env)
    z2 = rng.normal(0.0, 1.0, size=n_per_env)

    x1 = alpha1 * z1 + sigma_x1 * rng.normal(0.0, 1.0, size=n_per_env)
    x2 = alpha2 * z2 + sigma_x2 * rng.normal(0.0, 1.0, size=n_per_env)

    y1 = beta * x1 + gamma * z1 + sigma_y * rng.normal(0.0, 1.0, size=n_per_env)
    y2 = beta * x2 + gamma * z2 + sigma_y * rng.normal(0.0, 1.0, size=n_per_env)

    slope1 = ols_slope(x1, y1)
    slope2 = ols_slope(x2, y2)
    var1 = float(np.var(x1))
    var2 = float(np.var(x2))

    pooled_x = np.concatenate([x1, x2])
    pooled_y = np.concatenate([y1, y2])

    pooled_naive = ols_slope(pooled_x, pooled_y)
    inv_est = two_env_invariance_beta([slope1, slope2], [var1, var2])

    return {
        "pooled_naive": pooled_naive,
        "invariance_est": inv_est,
        "slope1": slope1,
        "slope2": slope2,
        "var1": var1,
        "var2": var2,
    }


def run_environment_stress(
    output_dir: Path,
    figure_dir: Path,
    env_reps: int,
    env_n: int,
    rng: np.random.Generator,
) -> Dict[str, object]:
    """Run environment-invariance stress grid over mechanism drift and variance separation."""
    beta = 0.60
    gamma = 0.90
    sigma_y = 1.00

    alpha1 = 1.00
    sigma_x1 = 0.45

    alpha_shift_grid = [0.0, 0.2, 0.4, 0.6]
    variance_ratio_grid = [1.0, 1.8, 2.6, 3.4]

    rows: List[Dict[str, float]] = []
    for alpha_shift in alpha_shift_grid:
        for variance_ratio in variance_ratio_grid:
            alpha2 = alpha1 * (1.0 + alpha_shift)
            sigma_x2 = sigma_x1 * variance_ratio

            pooled_estimates: List[float] = []
            inv_estimates: List[float] = []

            for _ in range(env_reps):
                sim = simulate_two_env(
                    beta=beta,
                    gamma=gamma,
                    alpha1=alpha1,
                    alpha2=alpha2,
                    sigma_x1=sigma_x1,
                    sigma_x2=sigma_x2,
                    sigma_y=sigma_y,
                    n_per_env=env_n,
                    rng=rng,
                )
                pooled_estimates.append(sim["pooled_naive"])
                inv_estimates.append(sim["invariance_est"])

            pooled_arr = np.asarray(pooled_estimates)
            inv_arr = np.asarray(inv_estimates)

            rows.append(
                {
                    "alpha_shift": alpha_shift,
                    "variance_ratio": variance_ratio,
                    "pooled_mean_bias": float(np.mean(pooled_arr - beta)),
                    "invariance_mean_bias": float(np.mean(inv_arr - beta)),
                    "pooled_rmse": rmse(pooled_arr, beta),
                    "invariance_rmse": rmse(inv_arr, beta),
                    "rmse_gain": rmse(pooled_arr, beta) - rmse(inv_arr, beta),
                    "env_reps": env_reps,
                }
            )

    env_df = pd.DataFrame(rows)
    env_df.to_csv(output_dir / "synthetic_environment_stress.csv", index=False)

    # Heatmap of RMSE gain (positive: invariance better).
    pivot = env_df.pivot(index="alpha_shift", columns="variance_ratio", values="rmse_gain")
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    im = ax.imshow(pivot.to_numpy(), origin="lower", aspect="auto", cmap="RdYlGn")
    ax.set_xticks(np.arange(pivot.shape[1]), labels=[f"{x:.1f}" for x in pivot.columns], fontsize=8)
    ax.set_yticks(np.arange(pivot.shape[0]), labels=[f"{x:.1f}" for x in pivot.index], fontsize=8)
    ax.set_xlabel("Variance ratio sigma_x2 / sigma_x1")
    ax.set_ylabel("Alpha shift")
    ax.set_title("Two-env Invariance RMSE Gain Over Pooled Naive")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.to_numpy()[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(figure_dir / "fig05_environment_stress_rmse_gain.png", dpi=220)
    plt.close(fig)

    return {
        "alpha_shift_grid_size": len(alpha_shift_grid),
        "variance_ratio_grid_size": len(variance_ratio_grid),
        "best_rmse_gain": float(env_df["rmse_gain"].max()),
        "worst_rmse_gain": float(env_df["rmse_gain"].min()),
    }


def write_json(path: Path, payload: Dict[str, object]) -> None:
    """Write JSON with stable indentation for manuscript traceability."""
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "research_grade_synthetic",
        help="Output directory for synthetic suite CSV/JSON.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "reports" / "figures",
        help="Directory for generated figures.",
    )
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--nonid-n-obs", type=int, default=8000)
    parser.add_argument("--classifier-reps", type=int, default=200)
    parser.add_argument("--classifier-n-per-class", type=int, default=1800)

    parser.add_argument("--benchmark-reps", type=int, default=250)
    parser.add_argument("--benchmark-n-obs", type=int, default=5000)
    parser.add_argument("--benchmark-n-int", type=int, default=5000)
    parser.add_argument(
        "--anchor-noises",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 1.0],
    )

    parser.add_argument("--phase-reps", type=int, default=120)
    parser.add_argument("--phase-n", type=int, default=3200)

    parser.add_argument("--env-reps", type=int, default=220)
    parser.add_argument("--env-n", type=int, default=3000)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    figure_dir: Path = args.figure_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    summary: Dict[str, object] = {
        "seed": args.seed,
        "paths": {
            "output_dir": str(output_dir),
            "figure_dir": str(figure_dir),
        },
    }

    summary["nonidentifiability"] = run_nonidentifiability_continuum(
        output_dir=output_dir,
        figure_dir=figure_dir,
        n_obs=args.nonid_n_obs,
        classifier_reps=args.classifier_reps,
        n_classifier_per_class=args.classifier_n_per_class,
        rng=rng,
    )

    summary["estimator_benchmark"] = run_estimator_benchmark(
        output_dir=output_dir,
        figure_dir=figure_dir,
        reps=args.benchmark_reps,
        n_obs=args.benchmark_n_obs,
        n_int=args.benchmark_n_int,
        anchor_noises=args.anchor_noises,
        rng=rng,
    )

    summary["anchor_phase"] = run_anchor_phase_diagram(
        output_dir=output_dir,
        figure_dir=figure_dir,
        phase_reps=args.phase_reps,
        phase_n=args.phase_n,
        rng=rng,
    )

    summary["environment_stress"] = run_environment_stress(
        output_dir=output_dir,
        figure_dir=figure_dir,
        env_reps=args.env_reps,
        env_n=args.env_n,
        rng=rng,
    )

    summary["config"] = {
        "nonid_n_obs": args.nonid_n_obs,
        "classifier_reps": args.classifier_reps,
        "classifier_n_per_class": args.classifier_n_per_class,
        "benchmark_reps": args.benchmark_reps,
        "benchmark_n_obs": args.benchmark_n_obs,
        "benchmark_n_int": args.benchmark_n_int,
        "anchor_noises": args.anchor_noises,
        "phase_reps": args.phase_reps,
        "phase_n": args.phase_n,
        "env_reps": args.env_reps,
        "env_n": args.env_n,
    }

    write_json(output_dir / "synthetic_suite_metadata.json", summary)


if __name__ == "__main__":
    main()
