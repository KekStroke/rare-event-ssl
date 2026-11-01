# Marine Mammal SSL data

| Pretraining source | Target hours |
| --- | ---: |
| DORI-ONC | 900 |
| Orcasound | 300 |
| Pacific Sound | 250 |
| NOAA SanctSound | 150 |

The collector stops at actual retained duration. Output is mono PCM16 at 16 kHz,
in non-overlapping five-second WAV windows inside TARs: 1,152,000 windows and
184.32 GB of sample bytes at quota. Worker count bounds temporary raw downloads;
the configured free-space reserve is 400 GB. Whole native archives are unnecessary.

Run acquisition and full validation in MLS CPU jobs. Install the independent data
package with `pip install -e .`; FFmpeg and ffprobe must be available in the image.
The existing detector environment in `requirements.txt` is separate.

```bash
REPO=/home/jovyan/iskhakov_anvar_local/rare-event-ssl
DATA=/mnt/shared_ru.ml.SZ-2_000180/Iskhakov/datasets/rare-event-ssl
REPORT=/mnt/shared_ru.ml.SZ-2_000180/Iskhakov/reports/rare-event-ssl/2026-11-02
cd "$REPO"
python -m marine_ssl.inventory --shared-root /mnt/shared_ru.ml.SZ-2_000180/Iskhakov \
  --output "$REPORT/inventory" --benchmark-config "$REPO/configs/benchmarks.yaml"
python -m marine_ssl.collect --config "$REPO/configs/pretrain.yaml" \
  --catalog-dir "$REPORT/inventory/catalogs" --output "$DATA/marine-ssl-v1" --workers 2
```

`scripts/run_marine_ssl_data.sh` and `configs/mls-data.yaml` provide the MLS launch
template. Copy them together as `run_job.sh` and `config.yaml` under a unique
`/home/jovyan/iskhakov_anvar_local/jobs/<job_id>/`, update the config script path,
and use a fresh `JOB_ID` for each launch. Submit with the workspace `mls-job` skill.
The job records dependencies and UTC progress, runs the tests, and reports errors
and quota shortfalls. Repeated runs verify existing artifacts before reusing them.

Leakage checks use only benchmark recordings selected for acquisition or already
present. ONC-Benchmark is also published inside DORI-ONC/test; Orcasound episodes
recur across DORI, Hackathon and continuous archives. Source recordings and
sessions are excluded before segmentation. Unknown shared-source provenance is
rejected. Add any later selected audio via planner `--protected-manifest`; rerun
the exclusions before using that benchmark. Exact PCM hashes supplement source
metadata; they do not identify arbitrary shifted or re-encoded excerpts.

Benchmark plans live in `configs/benchmarks.yaml`. `marine_ssl.benchmarks` resolves
download allowlists and `marine_ssl.acquire` transfers and fully decodes selected
public files. Avacha/ANIMAL-SPOT reuse the verified `2026-10-30` collection;
Dominica and Manual reuse local originals with SHA-256 transfer receipts. DCLDE
2027 needs an explicitly chosen audio manifest and has no automatic full-corpus
download. DCLDE 2022/2024 and reserve OOI are outside this collection.

Keep source versions, licenses, annotation joins and upstream splits with the data.
At the pinned ONC-Benchmark revision, 386 CSV rows reference 385 available audio
files; `ICLISTENHF1353_20230731T235507.000Z.flac` is missing. Do not invent it or
declare the annotation set complete. Keep Voices in the Sea reuse conditions with
its research copy. A completed transfer or quota does not establish a finished
evaluation protocol; inspect the validation summaries and original source groups.

Audio, catalogs, manifests and runtime outputs stay on shared storage, outside Git.
Run `python -m unittest discover -s tests -v` to verify the data tools without network.

## Legacy WAV containers

Only legacy Orcasound may use explicit container recovery. Complete PCM followed
by malformed non-audio metadata is retained unchanged. A tiny missing tail is
accepted only below both 100 ms and 0.1% of declared duration; incomplete physical
frames and incomplete five-second windows are discarded. Audio is never padded.
Original source SHA-256, declared sizes, omitted bytes/frames and recovery quality
are recorded in each receipt. Benchmark audio uses strict validation.
