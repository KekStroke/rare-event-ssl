# Marine SSL data work

The detector paper and its existing scripts remain an independent experiment.
New data tooling lives in `src/marine_ssl`, with configuration in `configs`.

- Pretraining quotas: DORI-ONC 900 h, Orcasound 300 h, Pacific Sound 250 h,
  SanctSound 150 h. Use actual retained duration, not estimated source sizes.
- Output: mono PCM16, 16 kHz, non-overlapping 5-second windows inside TARs.
- Check leakage only against the benchmark recordings actually selected for
  acquisition or already present. Do not expand to unrelated/reserve corpora.
- Exclude original recordings and correlated sessions before segmentation.
  Treat Orcasound publications as one source family. Preserve upstream labels.
- Keep audio, catalogs, manifests, validation and runtime logs outside Git.
  Record source revisions, licenses, exclusions and preparation parameters.
- Run downloads, decoding and inventories in MLS CPU jobs; follow workspace
  `mls-job` and `download-dataset`. Never install a training stack on Jupyter.
- A submitted job or metadata listing is not a completed dataset. Report actual
  terminal status, usable hours, missing audio, errors and any remaining quota.

Validate changes with `python -m unittest discover -s tests -v`. FFmpeg and
ffprobe are required for preparation; tests do not need network access.
