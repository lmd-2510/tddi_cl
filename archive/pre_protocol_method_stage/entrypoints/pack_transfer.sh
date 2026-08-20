#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/data/uyen/data_splits"
PACKAGE_NAME="${PACKAGE_NAME:-ddi2025_cil_transfer}"
STAMP="$(date '+%Y%m%d_%H%M%S')"
DIST_DIR="${PROJECT_ROOT}/dist"
STAGING_DIR="${DIST_DIR}/${PACKAGE_NAME}_${STAMP}"
ARCHIVE_PATH="${DIST_DIR}/${PACKAGE_NAME}_${STAMP}.tar.gz"

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

copy_tree() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "$dst")"
  cp -r "$src" "$dst"
}

cleanup_pycache() {
  find "$1" -type d -name "__pycache__" -prune -exec rm -rf {} +
  find "$1" -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
}

write_manifest() {
  cat > "${STAGING_DIR}/TRANSFER_README.md" <<EOF
# DDI2025-CIL Transfer Package

This package is prepared for moving the project to another SSH machine.

## Included

- \`plans/\`
- \`scripts/\`
- \`src/\`
- \`requirements.txt\`
- \`run_smoke.sh\`
- \`run_full.sh\`
- \`pack_transfer.sh\`
- selected lightweight runtime artifacts under \`outputs/\`:
  - \`audit/\`
  - \`class_distribution/\`
  - \`figures/\`
  - \`leakage/\`
  - \`preprocess/\`
  - \`tasks/\`

## Excluded

- raw datasets: \`*.csv\`, \`*.parquet\`
- heavy run artifacts: \`outputs/runs*\`
- smoke/debug duplicate outputs
- local caches: \`__pycache__/\`, \`*.pyc\`
- repo metadata: \`.git/\`, \`.codex/\`, \`.agents/\`

## Suggested Remote Steps

\`\`\`bash
tar -xzf $(basename "${ARCHIVE_PATH}")
cd $(basename "${STAGING_DIR}")
conda create -n ddi2025-cil python=3.11 -y
conda activate ddi2025-cil
pip install -r requirements.txt
\`\`\`

Copy the real dataset Parquet files into the project root on the remote machine before running:

\`\`\`bash
./run_smoke.sh
./run_full.sh
\`\`\`
EOF
}

main() {
  mkdir -p "${DIST_DIR}"
  rm -rf "${STAGING_DIR}"
  mkdir -p "${STAGING_DIR}"

  log "Copying project sources"
  copy_tree "${PROJECT_ROOT}/plans" "${STAGING_DIR}/plans"
  copy_tree "${PROJECT_ROOT}/scripts" "${STAGING_DIR}/scripts"
  copy_tree "${PROJECT_ROOT}/src" "${STAGING_DIR}/src"

  log "Copying top-level runtime files"
  cp "${PROJECT_ROOT}/requirements.txt" "${STAGING_DIR}/requirements.txt"
  cp "${PROJECT_ROOT}/run_smoke.sh" "${STAGING_DIR}/run_smoke.sh"
  cp "${PROJECT_ROOT}/run_full.sh" "${STAGING_DIR}/run_full.sh"
  cp "${PROJECT_ROOT}/pack_transfer.sh" "${STAGING_DIR}/pack_transfer.sh"

  log "Copying lightweight artifacts"
  mkdir -p "${STAGING_DIR}/outputs"
  for artifact_dir in audit class_distribution figures leakage preprocess tasks; do
    if [[ -d "${PROJECT_ROOT}/outputs/${artifact_dir}" ]]; then
      copy_tree "${PROJECT_ROOT}/outputs/${artifact_dir}" "${STAGING_DIR}/outputs/${artifact_dir}"
    fi
  done

  cleanup_pycache "${STAGING_DIR}"
  write_manifest

  log "Creating archive ${ARCHIVE_PATH}"
  tar -czf "${ARCHIVE_PATH}" -C "${DIST_DIR}" "$(basename "${STAGING_DIR}")"

  log "Package created"
  du -sh "${ARCHIVE_PATH}"
  printf '%s\n' "${ARCHIVE_PATH}"
}

main "$@"
