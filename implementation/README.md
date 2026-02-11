# Implementation

Scripts:
- `subproject_10_identifiability_latent_cell_state_confounding/implementation/scripts/run_toy_identifiability_study.py`
- `subproject_10_identifiability_latent_cell_state_confounding/implementation/scripts/run_research_grade_identifiability_suite.py`
- `subproject_10_identifiability_latent_cell_state_confounding/implementation/scripts/run_tissue_invariance_diagnostics.py`
- `subproject_10_identifiability_latent_cell_state_confounding/implementation/scripts/run_paper_quality_checks.py`
- `subproject_10_identifiability_latent_cell_state_confounding/implementation/scripts/run_gap_closure_validations.py`

Run commands:
```bash
python3 subproject_10_identifiability_latent_cell_state_confounding/implementation/scripts/run_toy_identifiability_study.py
python3 subproject_10_identifiability_latent_cell_state_confounding/implementation/scripts/run_research_grade_identifiability_suite.py
python3 subproject_10_identifiability_latent_cell_state_confounding/implementation/scripts/run_tissue_invariance_diagnostics.py
python3 subproject_10_identifiability_latent_cell_state_confounding/implementation/scripts/run_paper_quality_checks.py
python3 subproject_10_identifiability_latent_cell_state_confounding/implementation/scripts/run_gap_closure_validations.py
bash subproject_10_identifiability_latent_cell_state_confounding/scripts/reproduce_paper.sh
```

What they do:
- `run_toy_identifiability_study.py`: baseline non-identifiability and restoration sanity checks.
- `run_research_grade_identifiability_suite.py`: adversarial synthetic benchmark suite and paper figures.
- `run_tissue_invariance_diagnostics.py`: real-data cross-tissue invariance diagnostics from existing `.h5ad` assets.
- `run_paper_quality_checks.py`: gatekeeper controls (threshold sensitivity, permutation null, leave-one-environment-out stability).
- `run_gap_closure_validations.py`: gap-closure analyses (targeted perturbation overlap, pathway enrichment, cell-state consistency) and Figures 11-13 generation.

Key output directories:
- `subproject_10_identifiability_latent_cell_state_confounding/implementation/outputs/`
- `subproject_10_identifiability_latent_cell_state_confounding/implementation/outputs/research_grade_synthetic/`
- `subproject_10_identifiability_latent_cell_state_confounding/implementation/outputs/tissue_invariance/`
- `subproject_10_identifiability_latent_cell_state_confounding/implementation/outputs/tissue_invariance_quality/`
- `subproject_10_identifiability_latent_cell_state_confounding/implementation/outputs/gap_closure_validations/`

Key figure directory:
- `subproject_10_identifiability_latent_cell_state_confounding/reports/figures/`


## Agent Completion Tracking
- Do this for your project: `tracking/prompt.md`.
- When you finish meaningful work, update `tracking/prompt.md` with the project status (progress, decisions, blockers, and next step).
- Skip the update only if nothing meaningful changed.
