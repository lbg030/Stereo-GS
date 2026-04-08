#!/bin/bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
  EUROC_ROOT="$1"
  shift
else
  EUROC_ROOT="${EUROC_ROOT:-}"
fi

if [[ -z "${EUROC_ROOT}" ]]; then
  echo "Usage: $0 /path/to/euroc_root [extra stereo_demo args]"
  echo "Or set EUROC_ROOT in the environment."
  exit 1
fi

DATASETS=(
  "MH_01_easy"
  "MH_02_easy"
  "MH_03_medium"
  "MH_04_difficult"
  "MH_05_difficult"
  "V1_01_easy"
  "V1_02_medium"
  "V1_03_difficult"
  "V2_01_easy"
  "V2_02_medium"
  "V2_03_difficult"
)

for SEQ in "${DATASETS[@]}"
do
    echo "======= ${SEQ} 실행 ======="
    python stereo_demo.py \
        --dataset euroc \
        --datapath "${EUROC_ROOT}/${SEQ}" \
        --calib "calib/euroc.txt" \
        --baseline 0.1101 \
        --disable_vis \
        --stride 3 \
        --stereo \
        --final_refinement_iters 26000 \
        --buffer 256 \
        --upsample \
        "$@"
done
