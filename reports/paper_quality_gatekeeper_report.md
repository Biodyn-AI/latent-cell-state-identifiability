## A) VERDICT: READY (workshop / methods-track submission)

### Why
The previous submission blockers were addressed with new analyses: targeted perturbation overlap, pathway validation, and cell-state consistency. The manuscript now has explicit claim-tier calibration and no longer relies on proxy-only evidence for all real-data conclusions. Evidence is strong for the identifiability theory and diagnostic framework, and moderate for broad biological confirmation in real data.

### Central claim (one sentence)
Directional TF-target claims from observational model internals are generally non-identifiable under latent cell-state confounding, and should be promoted to directional language only when explicit identification assumptions survive stress tests and are supplemented by perturbation evidence when available.

### Main contributions and supporting evidence
1. Constructive non-identifiability under latent confounding.
- Evidence: observationally equivalent model continuum plus indistinguishability classifier (AUC 0.4994, accuracy 0.4995).

2. Identification restoration map with adversarial failure analysis.
- Evidence: synthetic benchmark across heavy tails, nonlinear interaction, measurement error, and selection bias.

3. Cross-tissue real-data directional plausibility diagnostics.
- Evidence: 22/76 edges pass conservative invariance criteria; null control enrichment (empirical p = 0.001996); stronger LOEO stability for pass edges.

4. Gap-closure biological validation.
- Evidence: targeted perturbation overlap (13 measurements, 7 edges), Hallmark/GO enrichment profiles for pass/fail target sets, and state-level slope consistency analysis.

5. Scientist-readable claim-tier protocol.
- Evidence: revised paper uses explicit Evidence/Inference/Hypothesis/Scientific implication framing per Results subsection.

### Top 5 likely reviewer objections
1. Perturbation overlap is still small, so direct causal confirmation remains limited.
2. Pass vs fail perturbation separation is not statistically significant (p = 0.217).
3. Cell-state consistency differences are directionally favorable but non-significant (p = 0.671).
4. Linear invariance assumptions may miss nonlinear regulatory effects.
5. Pathway enrichment can be inflated by ontology redundancy and background choices.

---

## B) Top 5 reviewer objections and how the revised draft addresses them
1. "You still do not have enough direct perturbation evidence."
- Addressed by adding targeted perturbation-backed analysis for exact manuscript edges and explicitly limiting causal language to covered settings.

2. "Your pass/fail split may be threshold-driven."
- Addressed by threshold sweep (pass count 9 to 29; baseline 22) and permutation null test showing pass enrichment beyond chance.

3. "Cross-environment estimates may be unstable."
- Addressed by LOEO analysis showing lower beta-range instability for pass edges (0.027 vs 0.144 median).

4. "Biological interpretation is still superficial."
- Addressed by pathway-level and cell-state-level analyses and explicit narrative framing in each Results subsection.

5. "The writing is too technical and reads like a pipeline."
- Addressed by full rewrite with scientist-readable structure and mandatory narrative-gate labels: Evidence, Inference, Hypothesis, Scientific implication.

---

## C) Revised paper in full (submission-style draft)
Updated manuscript:
- `subproject_10_identifiability_latent_cell_state_confounding/reports/identifiability_latent_confounding_research_paper.md`
- `subproject_10_identifiability_latent_cell_state_confounding/reports/identifiability_latent_confounding_research_paper.pdf`

This revision now includes:
- perturbation/pathway/cell-state gap-closure analyses,
- stronger scientist-facing interpretation,
- explicit evidence-calibrated claim tiers,
- updated limitations tied to observed uncertainty.

---

## D) Figure/table audit list
### Figure 1 (`fig01_nonidentifiability_continuum.png`)
- Status: retained.
- Interpretation role: demonstrates multiple causal worlds with identical observational fit.

### Figure 2 (`fig02_nonidentifiability_classifier_auc.png`)
- Status: retained.
- Interpretation role: confirms empirical indistinguishability under observational equivalence.

### Figure 3 (`fig03_estimator_rmse_heatmap.png`)
- Status: retained.
- Interpretation role: shows method fragility under measurement error/selection bias.

### Figure 4 (`fig04_anchor_phase_diagram.png`)
- Status: retained.
- Interpretation role: defines when anchors help versus hurt.

### Figure 5 (`fig05_environment_stress_rmse_gain.png`)
- Status: retained.
- Interpretation role: makes variance-separation boundary explicit.

### Figure 6 (`fig06_tissue_invariance_r2_hist.png`)
- Status: retained.
- Interpretation role: global fit-quality distribution for real edges.

### Figure 7 (`fig07_tissue_observed_vs_fitted_slopes.png`)
- Status: retained.
- Interpretation role: fit agreement, not causal confirmation.

### Figure 8 (`fig08_tissue_edge_examples.png`)
- Status: retained.
- Interpretation role: illustrative edge-level cases; aggregate controls remain primary.

### Figure 9 (`fig09_threshold_sensitivity.png`)
- Status: retained.
- Interpretation role: threshold robustness.

### Figure 10 (`fig10_permutation_null_pass_count.png`)
- Status: retained.
- Interpretation role: falsification against chance matching.

### Figure 11 (`fig11_targeted_perturbation_beta_vs_delta.png`)
- New gap-closure figure.
- Interpretation role: direct perturbation alignment check for pass/fail edges.

### Figure 12 (`fig12_hallmark_enrichment_pass_fail.png`)
- New gap-closure figure.
- Interpretation role: pathway-level biological context for pass/fail target sets.

### Figure 13 (`fig13_cellstate_sign_consistency_pass_fail.png`)
- New gap-closure figure.
- Interpretation role: within-environment state robustness of directional slopes.

### Table 1 (adversarial estimator benchmark)
- Status: retained and reframed.
- Interpretation role: links method behavior to biological/technical failure modes.

---

## E) Claims table
| Main claim | Support location | Strength |
|---|---|---|
| Observational internals do not identify direction under unrestricted confounding. | Theorem 1; Fig 1-2; Section 4.1 | Strong |
| Interventions and high-quality anchors recover effects in well-specified regimes. | Section 4.2-4.3; Fig 3-4; Table 1 | Strong |
| Environment invariance is powerful but has a stability boundary. | Section 4.4; Fig 5 | Strong |
| A minority of real tissue edges satisfy conservative directional plausibility diagnostics. | Section 4.5; Fig 6-10; quality summary JSON | Medium-Strong |
| Direct perturbation overlap partially supports but does not fully confirm pass/fail separation. | Section 4.6; Fig 11; perturbation summary JSON | Medium |
| Pathway/cell-state analyses add biological context but limited statistical separation. | Section 4.7; Fig 12-13; pathway/cellstate outputs | Medium |
| Broad causal confirmation for all pass edges is established. | Not claimed | Not supported / intentionally excluded |

---

## F) References list
Adebayo et al. (2018), Alain and Bengio (2016), Belinkov (2022), Cui et al. (2024), Geiger et al. (2021), Hewitt and Liang (2019), Hicks et al. (2018), Jain and Wallace (2019), Luecken and Theis (2019), Meng et al. (2022), Olsson et al. (2022), Pearl (2009), Peters et al. (2017), Theodoris et al. (2023).

---

## Residual risks (not blockers for workshop-level submission)
1. Expand matched perturbation overlap beyond 7 edges for stronger causal confirmation.
2. Add nonlinear invariance variants to stress-test linear-model sensitivity.
3. Increase cell-state-stratified sample size to raise power for pass/fail separation.
