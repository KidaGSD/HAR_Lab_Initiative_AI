#!/usr/bin/env bash
# End-to-end pipeline runner for Layer-1 Trigger data prep.
set -euo pipefail

LIMIT=${LIMIT:-40}
WINDOW_SIZE=${WINDOW_SIZE:-3.0}
HOP_SIZE=${HOP_SIZE:-0.2}
SEQ_LEN=${SEQ_LEN:-50}
CONFIG_PATH=${CONFIG_PATH:-config/default.yaml}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
RUN_ROOT="runs/${RUN_ID}"
LOGS_DIR="${RUN_ROOT}/state/logs"

mkdir -p "${LOGS_DIR}"
mkdir -p "${RUN_ROOT}"

echo "== Run ID: ${RUN_ID} (artifacts: ${RUN_ROOT}) ==" 

echo "== Step 1: refresh candidates =="
python analyze_cache.py

echo "== Step 2: download modalities (limit=${LIMIT}) =="
python scripts/download_modalities.py \
  --config "${CONFIG_PATH}" \
  --run-root "${RUN_ROOT}" \
  --uids-file candidate_takes.csv \
  --batch-size 20 \
  --limit "${LIMIT}" \
  --yes

LATEST_LOG=$(ls -t "${LOGS_DIR}"/download_*.json | head -n 1)
if [[ -z "${LATEST_LOG}" ]]; then
  echo "No download log found; aborting." >&2
  exit 1
fi

echo "== Step 3: update download status =="
python scripts/update_candidate_status.py \
  --log-file "${LATEST_LOG}" \
  --candidates candidate_takes.csv

echo "== Step 4: extract ready_for_alignment list =="
python scripts/extract_downloaded_uids.py \
  --log-file "${LATEST_LOG}" \
  --output ready_for_alignment.txt

echo "== Step 5: alignment + QA =="
python scripts/run_alignment.py \
  --config "${CONFIG_PATH}" \
  --run-root "${RUN_ROOT}" \
  --uids-file ready_for_alignment.txt \
  --ready-file state/active/ready_for_windowing.txt \
  --overwrite

echo "== Step 6: window extraction =="
python scripts/make_windows.py \
  --config "${CONFIG_PATH}" \
  --run-root "${RUN_ROOT}" \
  --uids-file ready_for_windowing.txt \
  --window "${WINDOW_SIZE}" \
  --hop "${HOP_SIZE}" \
  --seq-len "${SEQ_LEN}"

echo "== Step 7: weak labeling =="
python scripts/generate_weak_labels.py \
  --config "${CONFIG_PATH}" \
  --run-root "${RUN_ROOT}"

echo "== Step 8: write manifest =="
python tools/create_manifest.py \
  --run-id "${RUN_ID}" \
  --output-root runs \
  --uids-file ready_for_windowing.txt \
  --window "${WINDOW_SIZE}" \
  --hop "${HOP_SIZE}" \
  --seq-len "${SEQ_LEN}" \
  --weak-label-rule "linear_angspeed_gazestd_v1" \
  --notes "Generated via pipeline.sh"

echo "Pipeline completed. Weak labels stored in weak_labels.parquet"
