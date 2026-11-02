# Marine SSL data

The collection targets four pretraining sources and the selected benchmarks listed in
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
the card's approximate count is not a validation target.

Hackathon revision `b5aefd514724650a4e21b117a0715e1baa7a746f` publishes
`test/negative/000/rpi-orcasound-lab_2021_11_16_20_35_19.flac` as **zero bytes**.
Both the pinned catalog and the downloaded transfer receipt confirm that size;
the empty-file SHA-256 is `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
This is an upstream audio gap, not an unsupported codec or a transfer failure.
Retain its upstream row/split identity and failed validation; do not invent
replacement audio or change labels. The benchmark cannot be marked fully ready.
No verified mirror of this original recording has been established.

Voices in the Sea reuse terms remain unconfirmed: retain provenance and keep the
research copy private.

## Selected benchmark exclusions and pretraining

The current protected catalogs are ONC-Benchmark and Orcasound Hackathon 2024.
Every published ONC-Benchmark audio filename also appears in DORI-ONC/test.
Orcasound episodes can recur in DORI, Hackathon, and continuous archives. Exclude
selected original source recordings and acquisition sessions before segmentation;
a different five-second filename does not establish independence. The sensor ID
alone does not reserve every recording from that hydrophone.

Legacy Orcasound calendar dates with an unspecified local clock remain local dates,
not invented UTC timestamps. They clear the coarse time guard only when their year
is nonadjacent to every resolved year of the actually selected Orcasound test audio.
Same/adjacent years and unknown test-year coverage remain unresolved.

DCLDE 2027 currently has metadata only in this collection: no benchmark audio subset
is selected, downloaded, or claimed ready. Its default config cannot trigger a full
native-audio download. If benchmark audio is explicitly selected later, preserve
its original source/session metadata and add those actual records to the planner
with `--protected-manifest`. Unselected DCLDE corpora, reserve datasets, and the
undefined cross-deployment holdout do not block this collection. The latter remains
an undefined task, not an independent test result.

Collect bounded configured sources using the selected cached benchmark catalogs:

```bash
python -m marine_ssl.collect --config "$REPO/configs/pretrain.yaml" \
  --catalog-dir "$REPORT/inventory/catalogs" \
  --output "$DATA/marine-ssl-v1" --workers 8
```

Inspect `pretrain_plan.json`, candidate exclusions, preparation issues, and actual
verified duration. Reaching quotas establishes neither complete benchmark
availability nor absence of all possible acoustic duplicates. Final metadata audit
uses the actually prepared manifest and explicitly supplied, verified benchmark
manifests, enriching their transport IDs with original source/session metadata.
Missing manifests or unresolved shared-source provenance must remain partial.

Original-file SHA-256 and source/session/time exclusions detect known overlaps.
Native int32 PCM hashes and resampled int16 PCM hashes use different byte contracts
and cannot be compared directly. A metadata-clear result is not a global proof
against shifted, cropped, or cross-codec duplicates. Keep source revisions,
checksums, exclusions, and validation summaries beside the resulting manifests;
report upstream missing/empty benchmark audio separately from leakage clearance.

For dated Pacific Sound PCM WAVs, `use_range_prefix: true` acquires only the configured first six hours using verified HTTP ranges and a pinned ETag. The original object identity stays in provenance; the materialized prefix has its own SHA-256 and is never reported as the full original hash. Unsupported containers fall back to full-object acquisition.

Verified benchmark audio participates in original-file SHA-256 screening even when its source family is unknown. Missing origin evidence keeps the audit partial; an assignment based only on the local dataset directory or filename is not verified source provenance.
