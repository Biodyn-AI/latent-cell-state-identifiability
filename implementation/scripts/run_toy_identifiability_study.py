#!/usr/bin/env python3
"""Run low-compute toy simulations for identifiability under latent confounding.

This script operationalizes proposal 4 from
`market_research/ambitious_paper_questions/theory_first_low_compute_proposals.md`.

Outputs:
- `nonidentifiability_model_summary.csv`
- `nonidentifiability_covariance_diff.csv`
- `restoration_estimator_summary.csv`
- `environment_invariance_summary.csv`
- `run_metadata.json`
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class LinearSCM:
    """Linear-Gaussian SCM used for observational and intervention simulations.

    Variables:
    - Z: latent cell-state confounder
    - X: TF activity proxy
    - Y: target expression proxy
    - R: model-internal readout proxy (attention/probe/causal trace aggregate)

    Structural equations:
    X = alpha * Z + sigma_x * eps_x
    Y = beta * X + gamma * Z + sigma_y * eps_y
    R = lambda_r * X + nu_r * Y + sigma_r * eps_r
    """

    model_id: str
    beta: float
    alpha: float
    gamma: float
    sigma_x: float
    sigma_y: float
    lambda_r: float
    nu_r: float
    sigma_r: float


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Return simple OLS slope for y ~ x with intercept."""
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    var_x = float(np.dot(x_centered, x_centered) / x_centered.size)
    cov_xy = float(np.dot(x_centered, y_centered) / x_centered.size)
    if var_x == 0.0:
        raise ValueError("Encountered zero variance in X for OLS slope.")
    return cov_xy / var_x


def ols_coef_x_given_anchor(x: np.ndarray, anchor: np.ndarray, y: np.ndarray) -> float:
    """Return coefficient on x for regression y ~ 1 + x + anchor."""
    design = np.column_stack([np.ones(x.shape[0]), x, anchor])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[1])


def analytic_moments(model: LinearSCM) -> Dict[str, float]:
    """Return closed-form moments under the linear-Gaussian SCM."""
    var_x = model.alpha**2 + model.sigma_x**2
    cov_xy = model.beta * var_x + model.gamma * model.alpha
    var_y = (
        (model.beta**2) * var_x
        + model.gamma**2
        + 2.0 * model.beta * model.gamma * model.alpha
        + model.sigma_y**2
    )
    cov_xr = model.lambda_r * var_x + model.nu_r * cov_xy
    cov_yr = model.lambda_r * cov_xy + model.nu_r * var_y
    var_r = (
        (model.lambda_r**2) * var_x
        + (model.nu_r**2) * var_y
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


def analytic_covariance_matrix(model: LinearSCM) -> np.ndarray:
    """Return covariance matrix for [X, Y, R] under the model."""
    mom = analytic_moments(model)
    return np.array(
        [
            [mom["var_x"], mom["cov_xy"], mom["cov_xr"]],
            [mom["cov_xy"], mom["var_y"], mom["cov_yr"]],
            [mom["cov_xr"], mom["cov_yr"], mom["var_r"]],
        ]
    )


def matched_observational_model(base_model: LinearSCM, beta_alt: float, model_id: str) -> LinearSCM:
    """Construct an alternative model with the same observational moments for [X, Y, R].

    We keep alpha/sigma_x/readout parameters fixed, change beta, then solve for gamma and
    sigma_y so that Cov(X,Y) and Var(Y) match the base model exactly.
    """
    if base_model.alpha == 0.0:
        raise ValueError("Cannot solve matched model when alpha is zero.")

    base_mom = analytic_moments(base_model)
    var_x = base_mom["var_x"]
    cov_xy_target = base_mom["cov_xy"]
    var_y_target = base_mom["var_y"]

    gamma_alt = (cov_xy_target - beta_alt * var_x) / base_model.alpha

    sigma_y_sq_alt = var_y_target - (
        (beta_alt**2) * var_x
        + gamma_alt**2
        + 2.0 * beta_alt * gamma_alt * base_model.alpha
    )
    if sigma_y_sq_alt <= 0.0:
        raise ValueError(
            "Matched-model construction failed: solved sigma_y^2 is non-positive. "
            f"sigma_y_sq_alt={sigma_y_sq_alt:.6f}"
        )

    return LinearSCM(
        model_id=model_id,
        beta=beta_alt,
        alpha=base_model.alpha,
        gamma=gamma_alt,
        sigma_x=base_model.sigma_x,
        sigma_y=math.sqrt(sigma_y_sq_alt),
        lambda_r=base_model.lambda_r,
        nu_r=base_model.nu_r,
        sigma_r=base_model.sigma_r,
    )


def simulate_observational(model: LinearSCM, n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Simulate observational samples from the SCM."""
    z = rng.normal(0.0, 1.0, size=n)
    eps_x = rng.normal(0.0, 1.0, size=n)
    eps_y = rng.normal(0.0, 1.0, size=n)
    eps_r = rng.normal(0.0, 1.0, size=n)

    x = model.alpha * z + model.sigma_x * eps_x
    y = model.beta * x + model.gamma * z + model.sigma_y * eps_y
    r = model.lambda_r * x + model.nu_r * y + model.sigma_r * eps_r

    return {"z": z, "x": x, "y": y, "r": r}


def simulate_intervention(model: LinearSCM, n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Simulate data under do(X): X is randomized and independent of Z."""
    z = rng.normal(0.0, 1.0, size=n)
    x = rng.normal(0.0, 1.0, size=n)
    eps_y = rng.normal(0.0, 1.0, size=n)
    eps_r = rng.normal(0.0, 1.0, size=n)

    y = model.beta * x + model.gamma * z + model.sigma_y * eps_y
    r = model.lambda_r * x + model.nu_r * y + model.sigma_r * eps_r

    return {"z": z, "x": x, "y": y, "r": r}


def empirical_covariance_xyzr(x: np.ndarray, y: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Return empirical covariance matrix for [X, Y, R]."""
    stacked = np.column_stack([x, y, r])
    centered = stacked - stacked.mean(axis=0, keepdims=True)
    return centered.T @ centered / centered.shape[0]


def run_nonidentifiability_block(
    model_a: LinearSCM,
    model_b: LinearSCM,
    n_observational: int,
    rng: np.random.Generator,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]], Dict[str, float]]:
    """Run and summarize the non-identifiability demonstration."""
    analytic_cov_a = analytic_covariance_matrix(model_a)
    analytic_cov_b = analytic_covariance_matrix(model_b)
    analytic_diff = np.abs(analytic_cov_a - analytic_cov_b)

    # This check encodes the theorem setup: the two SCMs are observationally equivalent.
    analytic_max_abs_diff = float(analytic_diff.max())
    if analytic_max_abs_diff > 1e-10:
        raise RuntimeError(
            "Construction error: matched models are not analytically equivalent for [X,Y,R]. "
            f"max_abs_diff={analytic_max_abs_diff:.3e}"
        )

    sim_a = simulate_observational(model_a, n_observational, rng)
    sim_b = simulate_observational(model_b, n_observational, rng)

    empirical_cov_a = empirical_covariance_xyzr(sim_a["x"], sim_a["y"], sim_a["r"])
    empirical_cov_b = empirical_covariance_xyzr(sim_b["x"], sim_b["y"], sim_b["r"])

    empirical_diff = empirical_cov_a - empirical_cov_b

    model_rows: List[Dict[str, float]] = []
    for model in (model_a, model_b):
        moms = analytic_moments(model)
        model_rows.append(
            {
                "model_id": model.model_id,
                "true_beta": model.beta,
                "alpha": model.alpha,
                "gamma": model.gamma,
                "sigma_x": model.sigma_x,
                "sigma_y": model.sigma_y,
                "sigma_r": model.sigma_r,
                "analytic_var_x": moms["var_x"],
                "analytic_cov_xy": moms["cov_xy"],
                "analytic_var_y": moms["var_y"],
                "analytic_obs_slope": moms["obs_slope"],
                "sample_obs_slope": ols_slope(
                    sim_a["x"], sim_a["y"]
                )
                if model.model_id == model_a.model_id
                else ols_slope(sim_b["x"], sim_b["y"]),
            }
        )

    variables = ["X", "Y", "R"]
    diff_rows: List[Dict[str, float]] = []
    for i in range(3):
        for j in range(3):
            diff_rows.append(
                {
                    "row_var": variables[i],
                    "col_var": variables[j],
                    "analytic_diff": float(analytic_cov_a[i, j] - analytic_cov_b[i, j]),
                    "sample_diff": float(empirical_diff[i, j]),
                }
            )

    diagnostics = {
        "analytic_max_abs_cov_diff": analytic_max_abs_diff,
        "sample_max_abs_cov_diff": float(np.abs(empirical_diff).max()),
        "model_a_sample_obs_slope": ols_slope(sim_a["x"], sim_a["y"]),
        "model_b_sample_obs_slope": ols_slope(sim_b["x"], sim_b["y"]),
    }
    return model_rows, diff_rows, diagnostics


def summarize(values: Sequence[float]) -> Tuple[float, float]:
    """Return (mean, standard deviation) for a sequence."""
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1))


def run_restoration_blocks(
    model_a: LinearSCM,
    model_b: LinearSCM,
    n_observational: int,
    n_intervention: int,
    reps: int,
    anchor_noises: Sequence[float],
    rng: np.random.Generator,
) -> List[Dict[str, float]]:
    """Run intervention and anchor restoration experiments."""
    result_rows: List[Dict[str, float]] = []

    for model in (model_a, model_b):
        naive_estimates: List[float] = []
        intervention_estimates: List[float] = []
        anchor_estimates: Dict[float, List[float]] = {noise: [] for noise in anchor_noises}

        for _ in range(reps):
            obs = simulate_observational(model, n_observational, rng)
            naive_estimates.append(ols_slope(obs["x"], obs["y"]))

            interventional = simulate_intervention(model, n_intervention, rng)
            intervention_estimates.append(ols_slope(interventional["x"], interventional["y"]))

            for noise_scale in anchor_noises:
                # Anchor variable A approximates Z with additive measurement noise.
                anchor = obs["z"] + noise_scale * rng.normal(0.0, 1.0, size=n_observational)
                anchor_estimates[noise_scale].append(
                    ols_coef_x_given_anchor(obs["x"], anchor, obs["y"])
                )

        naive_mean, naive_sd = summarize(naive_estimates)
        result_rows.append(
            {
                "block": "restoration",
                "scenario": "observational_naive",
                "model_id": model.model_id,
                "true_beta": model.beta,
                "estimate_mean": naive_mean,
                "estimate_sd": naive_sd,
                "bias_mean": naive_mean - model.beta,
                "n_reps": reps,
            }
        )

        intervention_mean, intervention_sd = summarize(intervention_estimates)
        result_rows.append(
            {
                "block": "restoration",
                "scenario": "intervention_doX",
                "model_id": model.model_id,
                "true_beta": model.beta,
                "estimate_mean": intervention_mean,
                "estimate_sd": intervention_sd,
                "bias_mean": intervention_mean - model.beta,
                "n_reps": reps,
            }
        )

        for noise_scale in anchor_noises:
            estimate_mean, estimate_sd = summarize(anchor_estimates[noise_scale])
            result_rows.append(
                {
                    "block": "restoration",
                    "scenario": f"anchor_adjust_noise_{noise_scale:.2f}",
                    "model_id": model.model_id,
                    "true_beta": model.beta,
                    "estimate_mean": estimate_mean,
                    "estimate_sd": estimate_sd,
                    "bias_mean": estimate_mean - model.beta,
                    "n_reps": reps,
                }
            )

    return result_rows


def two_env_invariance_beta(slopes: Sequence[float], variances_x: Sequence[float]) -> float:
    """Solve beta from two-environment equations s_e = beta + c / Var(X|E=e)."""
    if len(slopes) != 2 or len(variances_x) != 2:
        raise ValueError("Two-environment estimator expects exactly two slopes and two variances.")

    s1, s2 = float(slopes[0]), float(slopes[1])
    v1, v2 = float(variances_x[0]), float(variances_x[1])

    denom = (1.0 / v1) - (1.0 / v2)
    if abs(denom) < 1e-12:
        raise ValueError("Environment variances are too similar for stable estimation.")
    c = (s1 - s2) / denom
    return s1 - (c / v1)


def simulate_environment(
    beta: float,
    gamma: float,
    alpha_by_env: Sequence[float],
    sigma_x_by_env: Sequence[float],
    sigma_y: float,
    n_per_env: int,
    rng: np.random.Generator,
) -> Dict[str, object]:
    """Simulate two environments and return naive and invariance estimates."""
    if len(alpha_by_env) != 2 or len(sigma_x_by_env) != 2:
        raise ValueError("Environment simulation currently supports exactly two environments.")

    slopes: List[float] = []
    var_xs: List[float] = []
    pooled_x: List[np.ndarray] = []
    pooled_y: List[np.ndarray] = []

    for alpha, sigma_x in zip(alpha_by_env, sigma_x_by_env):
        z = rng.normal(0.0, 1.0, size=n_per_env)
        eps_x = rng.normal(0.0, 1.0, size=n_per_env)
        eps_y = rng.normal(0.0, 1.0, size=n_per_env)

        x = alpha * z + sigma_x * eps_x
        y = beta * x + gamma * z + sigma_y * eps_y

        slopes.append(ols_slope(x, y))
        var_xs.append(float(np.var(x)))
        pooled_x.append(x)
        pooled_y.append(y)

    pooled_x_arr = np.concatenate(pooled_x)
    pooled_y_arr = np.concatenate(pooled_y)

    pooled_naive = ols_slope(pooled_x_arr, pooled_y_arr)
    invariance_est = two_env_invariance_beta(slopes, var_xs)
    return {
        "pooled_naive": pooled_naive,
        "invariance_est": invariance_est,
        "slopes": slopes,
        "var_xs": var_xs,
    }


def run_environment_block(
    beta: float,
    gamma: float,
    sigma_y: float,
    n_per_env: int,
    reps: int,
    rng: np.random.Generator,
) -> List[Dict[str, float]]:
    """Run environment-based identification with one valid and one violated scenario."""
    scenarios = {
        # Valid assumptions: alpha and gamma are invariant across environments,
        # while X-noise changes by environment.
        "assumptions_hold": {
            "alpha_by_env": (1.0, 1.0),
            "sigma_x_by_env": (0.45, 1.75),
        },
        # Violation: alpha changes across environments, so the confounding term is
        # not of the form c / Var(X|E=e) with a single shared c.
        "alpha_shift_violation": {
            "alpha_by_env": (1.0, 1.7),
            "sigma_x_by_env": (0.45, 1.75),
        },
    }

    rows: List[Dict[str, float]] = []
    for scenario, cfg in scenarios.items():
        pooled_naive_estimates: List[float] = []
        invariance_estimates: List[float] = []
        env1_slope_values: List[float] = []
        env2_slope_values: List[float] = []

        for _ in range(reps):
            sim = simulate_environment(
                beta=beta,
                gamma=gamma,
                alpha_by_env=cfg["alpha_by_env"],
                sigma_x_by_env=cfg["sigma_x_by_env"],
                sigma_y=sigma_y,
                n_per_env=n_per_env,
                rng=rng,
            )
            pooled_naive_estimates.append(float(sim["pooled_naive"]))
            invariance_estimates.append(float(sim["invariance_est"]))
            env1_slope_values.append(float(sim["slopes"][0]))
            env2_slope_values.append(float(sim["slopes"][1]))

        pooled_mean, pooled_sd = summarize(pooled_naive_estimates)
        inv_mean, inv_sd = summarize(invariance_estimates)
        env1_mean, env1_sd = summarize(env1_slope_values)
        env2_mean, env2_sd = summarize(env2_slope_values)

        rows.extend(
            [
                {
                    "block": "environment",
                    "scenario": scenario,
                    "model_id": "env_model",
                    "true_beta": beta,
                    "estimate_name": "pooled_naive",
                    "estimate_mean": pooled_mean,
                    "estimate_sd": pooled_sd,
                    "bias_mean": pooled_mean - beta,
                    "env1_slope_mean": env1_mean,
                    "env1_slope_sd": env1_sd,
                    "env2_slope_mean": env2_mean,
                    "env2_slope_sd": env2_sd,
                    "n_reps": reps,
                },
                {
                    "block": "environment",
                    "scenario": scenario,
                    "model_id": "env_model",
                    "true_beta": beta,
                    "estimate_name": "two_env_invariance",
                    "estimate_mean": inv_mean,
                    "estimate_sd": inv_sd,
                    "bias_mean": inv_mean - beta,
                    "env1_slope_mean": env1_mean,
                    "env1_slope_sd": env1_sd,
                    "env2_slope_mean": env2_mean,
                    "env2_slope_sd": env2_sd,
                    "n_reps": reps,
                },
            ]
        )

    return rows


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    """Write a list of dict rows to CSV with stable column order."""
    rows_list = list(rows)
    if not rows_list:
        raise ValueError(f"No rows provided for CSV output: {path}")

    fieldnames = list(rows_list[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_list:
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs",
        help="Directory for CSV/JSON outputs.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--n-observational", type=int, default=6000)
    parser.add_argument("--n-intervention", type=int, default=6000)
    parser.add_argument("--n-environment", type=int, default=5000)
    parser.add_argument("--reps", type=int, default=250)
    parser.add_argument(
        "--anchor-noises",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 1.0],
        help="Anchor-noise scales for A = Z + noise * eps.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Base and matched SCMs for the non-identifiability theorem demonstration.
    model_a = LinearSCM(
        model_id="edge_present",
        beta=0.80,
        alpha=1.00,
        gamma=-0.30,
        sigma_x=1.00,
        sigma_y=1.00,
        lambda_r=0.55,
        nu_r=0.40,
        sigma_r=0.70,
    )
    model_b = matched_observational_model(model_a, beta_alt=0.00, model_id="edge_absent")

    nonid_model_rows, nonid_diff_rows, nonid_diag = run_nonidentifiability_block(
        model_a=model_a,
        model_b=model_b,
        n_observational=args.n_observational,
        rng=rng,
    )

    restoration_rows = run_restoration_blocks(
        model_a=model_a,
        model_b=model_b,
        n_observational=args.n_observational,
        n_intervention=args.n_intervention,
        reps=args.reps,
        anchor_noises=args.anchor_noises,
        rng=rng,
    )

    environment_rows = run_environment_block(
        beta=0.60,
        gamma=0.90,
        sigma_y=1.00,
        n_per_env=args.n_environment,
        reps=args.reps,
        rng=rng,
    )

    write_csv(output_dir / "nonidentifiability_model_summary.csv", nonid_model_rows)
    write_csv(output_dir / "nonidentifiability_covariance_diff.csv", nonid_diff_rows)
    write_csv(output_dir / "restoration_estimator_summary.csv", restoration_rows)
    write_csv(output_dir / "environment_invariance_summary.csv", environment_rows)

    metadata = {
        "seed": args.seed,
        "n_observational": args.n_observational,
        "n_intervention": args.n_intervention,
        "n_environment_per_env": args.n_environment,
        "reps": args.reps,
        "anchor_noises": args.anchor_noises,
        "nonidentifiability": nonid_diag,
        "models": {
            "edge_present": {
                "beta": model_a.beta,
                "alpha": model_a.alpha,
                "gamma": model_a.gamma,
                "sigma_x": model_a.sigma_x,
                "sigma_y": model_a.sigma_y,
            },
            "edge_absent": {
                "beta": model_b.beta,
                "alpha": model_b.alpha,
                "gamma": model_b.gamma,
                "sigma_x": model_b.sigma_x,
                "sigma_y": model_b.sigma_y,
            },
        },
    }

    with (output_dir / "run_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)


if __name__ == "__main__":
    main()
