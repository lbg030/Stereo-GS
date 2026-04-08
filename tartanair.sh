#!/bin/bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
  TARTANAIR_ROOT="$1"
  shift
else
  TARTANAIR_ROOT="${TARTANAIR_ROOT:-}"
fi

if [[ -z "${TARTANAIR_ROOT}" ]]; then
  echo "Usage: $0 /path/to/tartanair_stereo_root [extra stereo_demo args]"
  echo "Or set TARTANAIR_ROOT in the environment."
  exit 1
fi

DATASETS=(
  # "SE000"
  # "SE001"
  # "SE002"
  # "SE003"
  "SE004"
  "SE005"
  "SE006"
  "SE007"
  # "SH000"
  # "SH001"
  # "SH002"
  # "SH003"
  # "SH004"
  # "SH005"
  # "SH006"
  # "SH007"
)

for SEQ in "${DATASETS[@]}"
do
    echo "======= ${SEQ} 실행 ======="
    python stereo_demo.py \
        --dataset tartan \
        --datapath "${TARTANAIR_ROOT}/${SEQ}" \
        --calib "calib/tartan.txt" \
        --baseline 0.25 \
        --disable_vis \
        --stride 4 \
        --stereo \
        --final_refinement_iters 26000 \
        --buffer 256 \
        --upsample \
        "$@"
done
