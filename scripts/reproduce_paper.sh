#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -d "${ROOT_DIR}/../single_cell_mechinterp" ]]; then
  WORKSPACE_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
else
  WORKSPACE_ROOT="${ROOT_DIR}"
fi
cd "${WORKSPACE_ROOT}"

run() {
  echo "+ $*"
  "$@"
}

PROJECT_DIR="${ROOT_DIR}"
SCRIPT_DIR="${PROJECT_DIR}/implementation/scripts"
REPORTS_DIR="${PROJECT_DIR}/reports"
PAPER_MD="${REPORTS_DIR}/identifiability_latent_confounding_research_paper.md"

echo "[reproduce] workspace: ${WORKSPACE_ROOT}"

run python3 "${SCRIPT_DIR}/run_toy_identifiability_study.py"
run python3 "${SCRIPT_DIR}/run_research_grade_identifiability_suite.py"

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
  run python3 "${SCRIPT_DIR}/run_tissue_invariance_diagnostics.py"
  run python3 "${SCRIPT_DIR}/run_paper_quality_checks.py"
  run python3 "${SCRIPT_DIR}/run_gap_closure_validations.py"
else
  echo "[reproduce] ${missing_count} real-data assets missing: using cached real-data outputs."
  if [[ ! -f "${PROJECT_DIR}/implementation/outputs/tissue_invariance/tissue_edge_environment_stats.csv" ]]; then
    echo "[reproduce] ERROR: cached tissue invariance outputs missing."
    exit 1
  fi
  run python3 "${SCRIPT_DIR}/run_paper_quality_checks.py"
fi

if [[ -f "${WORKSPACE_ROOT}/reports/scripts/validate_scientific_writing.py" ]]; then
  run python3 "${WORKSPACE_ROOT}/reports/scripts/validate_scientific_writing.py" --file "${PAPER_MD}"
elif [[ -f "${PROJECT_DIR}/reports/scripts/validate_scientific_writing.py" ]]; then
  run python3 "${PROJECT_DIR}/reports/scripts/validate_scientific_writing.py" --file "${PAPER_MD}"
else
  echo "[reproduce] WARNING: narrative validator not found; skipping manuscript validation."
fi

run bash -lc "cd \"${REPORTS_DIR}\" && pandoc identifiability_latent_confounding_research_paper.md -o identifiability_latent_confounding_research_paper.pdf --pdf-engine=xelatex"

echo "[reproduce] complete"
