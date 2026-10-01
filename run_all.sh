#!/usr/bin/env bash
# Step 4: 依序執行 01 抓取 -> 02 深度+手部姿態 -> 03 可視化。
# 用法: ./run_all.sh [--skip-pull] [--force] [--max-frames N] [--no-viz]
#   --skip-pull  不連手機,改用 data/raw 內最新的 session
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

skip_pull=0; force=(); max_frames=(); viz=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-pull)  skip_pull=1 ;;
    --force)      force=(--force) ;;
    --max-frames) max_frames=(--max-frames "$2"); shift ;;
    --no-viz)     viz=0 ;;
    -h|--help)    sed -n '2,5p' "$0"; exit 0 ;;
    *) echo "未知參數: $1" >&2; exit 1 ;;
  esac
  shift
done

step2=()
if [[ $skip_pull -eq 0 ]]; then
  echo "===== 01_pull_latest.py ====="
  raw="$(python3 01_pull_latest.py "${force[@]}" | tail -n 1)"
  step2+=("$raw")
fi

echo "===== 02_depth_wilor.py ====="
out="$(python3 02_depth_wilor.py "${step2[@]}" "${max_frames[@]}" | tee /dev/stderr | tail -n 1)"

if [[ $viz -eq 1 ]]; then
  echo "===== 03_visualize.py ====="
  python3 03_visualize.py "$out"
fi
