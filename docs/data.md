# Marine SSL data

The collection contains four pretraining sources and the benchmarks listed in
`configs/benchmarks.yaml`. Optional DORI-OOI and historical DCLDE corpora are outside
this collection. Downloads, conversion, full decoding, and inventories run in MLS
CPU jobs; the Jupyter host only prepares and submits jobs.

| Pretraining source | Target hours | Selection |
| --- | ---: | --- |
| DORI-ONC | 900 | Pinned HF revision; preserve original recording IDs and metadata |
| Orcasound | 300 | Unique source recordings; DORI-Orcasound supplies metadata, not a second corpus |
| Pacific Sound | 250 | Dated `pacific-sound-16khz` objects; reuse verified existing files first |
| SanctSound | 150 | CI01, MB01, SB01/SB02, FK01; retain deployment and detection provenance |
| Total | 1,600 | Actual verified duration, not estimated file counts |

Output is mono PCM16 at 16 kHz, five-second windows without overlap. At quota the
sample payload is **184.32 GB** (171.66 GiB), or 1,152,000 windows. TAR shards keep
these windows out of a million-file directory; headers, manifests, and receipts add
overhead. Original downloads and benchmark audio need separate capacity accounting.
The collector bounds prefetch by worker count, checks known object sizes against
free space, retains a configured 400 GB reserve, and removes only its owned temporary
sources after verified conversion. Do not reserve or download the entire native
DORI/DCLDE archives to satisfy these quotas. Check the inventory and largest queued
objects before choosing concurrency.

## Inventory and acquisition

Install the data dependencies from `pyproject.toml` in the MLS job environment.
Run these commands from the repository in an MLS CPU job with Internet access when
needed. Paths below refer to the shared mount; `Y:` is its Windows mapping.

```bash
REPO=/home/jovyan/iskhakov_anvar_local/rare-event-ssl
DATA=/mnt/shared_ru.ml.SZ-2_000180/Iskhakov/datasets/rare-event-ssl
REPORT=/mnt/shared_ru.ml.SZ-2_000180/Iskhakov/reports/rare-event-ssl/2026-11-02
cd "$REPO"
python -m marine_ssl.inventory --shared-root /mnt/shared_ru.ml.SZ-2_000180/Iskhakov \
  --output "$REPORT/inventory" --benchmark-config "$REPO/configs/benchmarks.yaml"
python -m marine_ssl.acquire --plan "$REPORT/inventory/benchmark_plan.json" \
  --output "$DATA/2026-11-02/benchmarks" --workers 8 --reserve-gb 400 \
  --datasets onc_benchmark orcasound_hackathon2024 beans_watkins beans_hiceas voices_in_the_sea
```

The default HF inventory covers DORI-ONC, DORI-Orcasound, ONC-Benchmark, and
Orcasound Hackathon 2024. It scans existing files only inside this project's dataset
directory. Metadata discovery and a successful transfer do not establish benchmark
readiness: inspect source checksums, full decoding, annotation joins, durations,
sample rates/channels, original splits, missing files, and duplicate reports.

Avacha and ANIMAL-SPOT use the existing verified collection under
`/mnt/shared_ru.ml.SZ-2_000180/Iskhakov/datasets/rare-event-ssl/2026-10-30/`.
Dominica and the manual sound/noise set use authorized local originals with source
SHA-256 transfer receipts. Preserve their annotations and original split files.
Manual audio without matching annotations cannot enter labeled evaluation.

ONC-Benchmark has 386 metadata rows but 385 published audio files at the pinned
revision; `ICLISTENHF1353_20230731T235507.000Z.flac` is missing. Keep that discrepancy
visible. Hackathon's published audio tree and metadata must be joined explicitly;
the card's approximate count is not a validation target. Voices in the Sea reuse
terms remain unconfirmed: retain provenance and keep the research copy private.

## Selected benchmark exclusions and pretraining

DCLDE 2027 acquisition requires an **explicit selected audio manifest**. Its default
config has no selected subset and cannot trigger a full native-corpus download.
Freeze the chosen benchmark recordings and their source/session metadata, then
generate a completeness receipt for those selected records. Do not exclude unrelated
historical archives or all published DCLDE recordings merely because they exist.
The official challenge test is scheduled for 2027 and cannot yet be verified locally.

Every published ONC-Benchmark audio filename also appears in DORI-ONC/test. Orcasound
episodes can recur in DORI, Hackathon, and continuous archives. Exclude complete
selected source recordings/sessions before segmentation, including upstream source
files behind crops; a different five-second filename does not establish independence.
Freeze the independent Orcasound deployment/session holdout before declaring training
ready. Unknown provenance remains unresolved rather than being treated as disjoint.

After the selected DCLDE manifest and completeness receipt exist, prepare candidates:

```bash
python -m marine_ssl.planning --config "$REPO/configs/pretrain.yaml" \
  --catalog-dir "$REPORT/inventory/catalogs" --output "$REPORT/pretrain" \
  --dclde-manifest "$REPORT/dclde_selected.jsonl" \
  --dclde-receipt "$REPORT/dclde_selected_receipt.json"
python -m marine_ssl.prepare --config "$REPO/configs/pretrain.yaml" \
  --candidates "$REPORT/pretrain/pretrain_candidates.jsonl" \
  --output "$DATA/marine-ssl-v1" --workers 8
```

Pass `--holdout-manifest` to the planner when the independent labeled Orcasound
holdout is frozen. Acquisition may proceed for screened sources while protocol
blockers remain explicit. Reaching duration quotas does not clear those blockers.
Decoded PCM hashes identify exact duplicates; resampled or shifted excerpts still
require source/time provenance or a separate audio overlap investigation. Keep the
candidate exclusions, preparation issues, source revisions, checksums, and final
validation summaries beside the resulting manifests.
