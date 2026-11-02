#!/usr/bin/env bash
set -euo pipefail
REPO=/home/jovyan/iskhakov_anvar_local/rare-event-ssl
JOB_ID=${JOB_ID:-marine_ssl_v1}
LOG=/home/jovyan/iskhakov_anvar_local/logs/rare-event-ssl/$JOB_ID
mkdir -p "$LOG"
if ! mkdir "$LOG/owner.claim" 2>/dev/null; then
  for ((i=0; i<17280; i++)); do
    [ ! -f "$LOG/owner.failed" ] || exit 1
    [ ! -f "$LOG/owner.done" ] || exit 0
    sleep 5
  done
  exit 1
fi
trap 'rc=$?; if [ "$rc" = 0 ]; then touch "$LOG/owner.done"; else touch "$LOG/owner.failed"; fi' EXIT
exec > >(tee -a "$LOG/run.log") 2>&1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=$REPO/src
cd "$REPO"
date -u
python3 -m pip freeze > "$LOG/dependencies.txt"
git rev-parse HEAD > "$LOG/code_revision.txt"
command -v ffmpeg
command -v ffprobe
python3 -m unittest discover -s tests -v
python3 -u -m marine_ssl.collect \
  --config "$REPO/configs/pretrain.yaml" \
  --catalog-dir /mnt/shared_ru.ml.SZ-2_000180/Iskhakov/reports/rare-event-ssl/data-audit-20261102/catalogs \
  --output /mnt/shared_ru.ml.SZ-2_000180/Iskhakov/datasets/rare-event-ssl/marine-ssl-v1 \
  --workers 2
