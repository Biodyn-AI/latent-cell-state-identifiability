#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
cd "${WORKSPACE_ROOT}"

run() {
  echo "+ $*"
  "$@"
}

echo "[reproduce] workspace: ${WORKSPACE_ROOT}"

run python3 subproject_10_identifiability_latent_cell_state_confounding/implementation/scripts/run_toy_identifiability_study.py
run python3 subproject_10_identifiability_latent_cell_state_confounding/implementation/scripts/run_research_grade_identifiability_suite.py

REALDATA_FILES=(
  "single_cell_mechinterp/outputs/invariant_causal_edges/kidney/processed.h5ad"
  "single_cell_mechinterp/outputs/invariant_causal_edges/lung/processed.h5ad"
  "single_cell_mechinterp/outputs/invariant_causal_edges/immune/processed.h5ad"
  "single_cell_mechinterp/outputs/invariant_causal_edges/external_lung/processed.h5ad"
  "single_cell_mechinterp/data/perturb/adamson/perturb_processed_symbols.h5ad"
  "single_cell_mechinterp/data/perturb/dixit/perturb_processed_symbols.h5ad"
  "single_cell_mechinterp/data/perturb/dixit_7_days/perturb_processed_symbols.h5ad"
)

missing_count=0
for f in "${REALDATA_FILES[@]}"; do
  if [[ ! -f "${WORKSPACE_ROOT}/${f}" ]]; then
    missing_count=$((missing_count + 1))
  fi
done

if [[ ${missing_count} -eq 0 ]]; then
  echo "[reproduce] real-data assets found: running full real-data regeneration."
  run python3 subproject_10_identifiability_latent_cell_state_confounding/implementation/scripts/run_tissue_invariance_diagnostics.py
  run python3 subproject_10_identifiability_latent_cell_state_confounding/implementation/scripts/run_paper_quality_checks.py
  run python3 subproject_10_identifiability_latent_cell_state_confounding/implementation/scripts/run_gap_closure_validations.py
else
  echo "[reproduce] ${missing_count} real-data assets missing: using cached real-data outputs."
  if [[ ! -f "${WORKSPACE_ROOT}/subproject_10_identifiability_latent_cell_state_confounding/implementation/outputs/tissue_invariance/edge_fit_results.csv" ]]; then
    echo "[reproduce] ERROR: cached tissue invariance outputs missing."
    exit 1
  fi
  run python3 subproject_10_identifiability_latent_cell_state_confounding/implementation/scripts/run_paper_quality_checks.py
fi

if [[ -f "${WORKSPACE_ROOT}/reports/scripts/validate_scientific_writing.py" ]]; then
  run python3 reports/scripts/validate_scientific_writing.py --file subproject_10_identifiability_latent_cell_state_confounding/reports/identifiability_latent_confounding_research_paper.md
else
  echo "[reproduce] WARNING: narrative validator not found; skipping manuscript validation."
fi

run bash -lc 'cd subproject_10_identifiability_latent_cell_state_confounding/reports && pandoc identifiability_latent_confounding_research_paper.md -o identifiability_latent_confounding_research_paper.pdf --pdf-engine=xelatex'

echo "[reproduce] complete"
