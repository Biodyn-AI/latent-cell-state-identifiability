# Identifiability of Regulatory Edges Under Latent Cell-State Confounding

## 1. Scope and goal
This note executes Proposal 4 from `market_research/ambitious_paper_questions/theory_first_low_compute_proposals.md`.

Goal: determine when a mechanistic claim about `TF -> target` is identifiable from model-internal signals versus when latent cell-state confounding makes the claim fundamentally ambiguous.

## 2. Structural setup
We use a linear-Gaussian SCM:

- `Z`: latent cell-state confounder
- `X`: TF activity proxy
- `Y`: target proxy
- `R`: model-internal readout proxy

Equations:
- `X = alpha * Z + eps_x`
- `Y = beta * X + gamma * Z + eps_y`
- `R = lambda * X + nu * Y + eps_r`

Question: can `beta` (the causal `TF -> target` effect) be identified from observational distributions over `(X, Y, R)`?

## 3. Theoretical results
### 3.1 Non-identifiability under unrestricted latent confounding
For observational data, the slope satisfies:
- `s_obs = Cov(X,Y) / Var(X) = beta + gamma * alpha / Var(X)`

Therefore multiple `(beta, gamma)` pairs can produce the same observed slope.
With Gaussian noise, matching `Cov(X,Y)` and `Var(Y)` is enough to match the full observational law over `(X,Y)`, and therefore over `(X,Y,R)` when `R` is a fixed linear function of `X,Y` plus noise.

Constructive reparameterization (holding `alpha, sigma_x` fixed):
- choose any `beta'`
- set `gamma' = (Cov_target - beta' * Var(X)) / alpha`
- set `sigma_y'^2 = VarY_target - [beta'^2 Var(X) + gamma'^2 + 2 beta' gamma' alpha]`

If `sigma_y'^2 > 0`, the alternative model is observationally equivalent but has different `beta`.

### 3.2 Identifiability restoration via intervention sufficiency
Under randomized `do(X)` interventions, `X` is independent of `Z`. Then:
- `E[Y | do(X=x)] = beta * x + const`

So regression slope of `Y` on intervened `X` identifies `beta`.

### 3.3 Identifiability restoration via environment invariance (two-environment case)
Assume two environments `e in {1,2}` with:
- invariant `beta, alpha, gamma`
- environment-dependent `sigma_x,e`

Then:
- `s_e = beta + c / Var(X|E=e)` where `c = alpha * gamma`

With two environments:
- `c = (s_1 - s_2) / (1/VarX_1 - 1/VarX_2)`
- `beta = s_1 - c / VarX_1`

So `beta` is identified if those invariance assumptions hold and environment variances are distinct.

### 3.4 Identifiability restoration via anchor variables
If an anchor `A` measures confounder state (`A = Z + noise`) and noise is small, regression `Y ~ X + A` approximately deconfounds and recovers `beta`.

If anchor noise is large, residual confounding remains and bias returns.

## 4. Toy simulation execution
Script executed:

```bash
python3 subproject_10_identifiability_latent_cell_state_confounding/implementation/scripts/run_toy_identifiability_study.py
```

Artifacts:
- `subproject_10_identifiability_latent_cell_state_confounding/implementation/outputs/nonidentifiability_model_summary.csv`
- `subproject_10_identifiability_latent_cell_state_confounding/implementation/outputs/nonidentifiability_covariance_diff.csv`
- `subproject_10_identifiability_latent_cell_state_confounding/implementation/outputs/restoration_estimator_summary.csv`
- `subproject_10_identifiability_latent_cell_state_confounding/implementation/outputs/environment_invariance_summary.csv`

## 5. Empirical sanity checks (toy only)
### 5.1 Non-identifiability construction succeeds
Two SCMs were constructed:
- `edge_present`: `beta = 0.8`, `gamma = -0.3`
- `edge_absent`: `beta = 0.0`, `gamma = 1.3`

Key result from `run_metadata.json`:
- `analytic_max_abs_cov_diff = 0.0` for `[X,Y,R]`
- sample observational slopes are nearly identical:
  - `edge_present`: `0.6446`
  - `edge_absent`: `0.6427`

Interpretation: observational internals cannot distinguish strong-edge and no-edge worlds in this setting.

### 5.2 Restoration via interventions
From `restoration_estimator_summary.csv`:
- `edge_present` (`true beta = 0.8`)
  - naive observational estimate: `0.6503` (bias `-0.1497`)
  - intervention estimate: `0.7997` (bias `-0.0003`)
- `edge_absent` (`true beta = 0.0`)
  - naive observational estimate: `0.6504` (bias `+0.6504`)
  - intervention estimate: `0.0002` (bias `+0.0002`)

### 5.3 Restoration via anchors
From `restoration_estimator_summary.csv`:
- `edge_present`, anchor-adjusted estimate moves from `0.8017` (noise `0.00`) to `0.7007` (noise `1.00`)
- `edge_absent`, anchor-adjusted estimate moves from `0.0002` (noise `0.00`) to `0.4336` (noise `1.00`)

Interpretation: good anchors can restore identification; noisy anchors only partially help.

### 5.4 Restoration via environment invariance
From `environment_invariance_summary.csv` with true `beta = 0.6`:

- Assumptions hold:
  - pooled naive estimate: `0.9417` (bias `+0.3417`)
  - two-env invariance estimate: `0.6002` (bias `+0.0002`)

- Assumptions violated (alpha shifts by environment):
  - pooled naive estimate: `0.9397` (bias `+0.3397`)
  - two-env invariance estimate: `0.7328` (bias `+0.1328`)

Interpretation: environment-based identification is powerful but assumption-sensitive.

## 6. Practical claim checklist
Use this checklist before claiming mechanistic edge recovery from internals.

| Tier | Minimum assumptions/evidence | Allowed claim |
|---|---|---|
| Weak | Observational internals only (`R`, no interventions, no anchor quality evidence) | Association only. No directional `TF -> target` claim. |
| Medium | Anchor available with quantified noise, plus sensitivity analysis to anchor degradation | Conditional directional claim with explicit residual-confounding caveat. |
| Medium+ | Multiple environments with tested invariance assumptions and variance separation | Directional claim only within the declared invariance regime. |
| Strong | Randomized or strongly quasi-randomized `do(X)`-style perturbation sufficiency checks | Directional mechanistic claim with strongest identifiability support. |

## 7. Limitations
- Linear-Gaussian toy SCM is intentionally simplified.
- No biological benchmark claims are made from these simulations.
- Environment-based identification can fail under realistic assumption violations.

## 8. Next executable step
Evaluate whether existing cross-tissue artifacts in this repository satisfy the invariance preconditions (constant mechanism and adequate environment variance separation) before applying the environment estimator to real outputs.
