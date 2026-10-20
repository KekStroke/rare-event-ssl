# tsssl: compact single-pass detection of overlapping killer-whale calls

Code, experiment database and manuscripts behind the draft *A Compact Single-Pass Detector of Overlapping Killer-Whale Calls Trained on Synthetic-Mixture Labels: What the Components Contribute* (papers/paper3) and the companion study *What Moves Detection and What Moves the Boundary?* (papers/paper1).

## Layout
- `common.py` audio front-end (16 kHz, 80-mel, 20 ms tokens), synthetic-mixture construction (`synth`, overlap / paired no-overlap arms), encoders (vanilla and Gemma-style "modern" block), event targets, decoders, scorer. Dataset profile via `TSSSL_DATASET=whale|ozf`.
- `prep.py`, `prep_ozf.py` data preparation (whale lab clips + field recordings; OZF from Zenodo 15507508, slowed x2).
- `pretrain.py` HuBERT-style / event-branch pretraining (`--event_head`, `--no_hubert` = regime R3). `finetune.py` field fine-tuning with the event branch. `probe.py` frozen probes.
- `decode_instances.py` instance decoder (onset peaks + log-duration, no box NMS) and frame decoder, thresholds chosen on development recordings under a false-alarm budget.
- `bootstrap_instances.py`, `bootstrap_h24.py`, `bootstrap_ozf.py`, `bootstrap_ozf_vox.py` paired recording-level bootstraps (recording carries all seeds).
- `vox_field_eval.py`, `vox_ozf_eval.py`, `voxaboxen_eval.py`, `map_eval.py` Voxaboxen comparison under our protocol and mAP in Voxaboxen's definition.
- `analyze_misses.py`, `analyze_onset_proximity.py`, `check_aggregation.py`, `audit_metric.py` diagnostics.
- `run_*.sh` the exact launch scripts of every series (H21: `run_approved.sh`-era scripts; H22 `run_r3van*.sh`; H23 `run_vox_*.sh`; H24 `run_noov.sh`; H26 `run_ozf_arm.sh`, `run_vox_ozf.sh`).
- `expdb/` experiment database: `expdb.py` (SQLite `expdb.sqlite`: hardware fingerprint, code snapshot, data checksums, curves, results per run), `knowledge/hypotheses.json` (H1-H26 with verdicts and run ids), `docs/FINDINGS.md` (auto-generated).
- `results/*.json` every reported number (decode outputs, bootstraps, mAP); per-recording tracks (`*.npz`) are not in git (96 MB) and can be regenerated with `finetune.py`.
- `papers/paper3/main.tex` main manuscript (v0.6); `papers/paper1/main.tex` companion study (reserve).

## Reproducing the main tables
Whale: `python decode_instances.py fte_R3_seed0 100` per run, then `python bootstrap_instances.py 100 2000` and `python bootstrap_h24.py 100 2000`.
OZF: `TSSSL_DATASET=ozf python bootstrap_ozf.py 400 2000`, `TSSSL_DATASET=ozf python bootstrap_ozf_vox.py 400 2000`, `TSSSL_DATASET=ozf python map_eval.py fte_ozf_ov_seed0 400`.
Both need the saved tracks in `results/` (regenerate with `finetune.py` from the checkpoints) and the data layout from `prep.py` / `prep_ozf.py`. Protocols: `TSSSL_voxaboxen_protocol_ru.md`, `TSSSL_OZF_protocol_ru.md` (in `docs/`).

Data are not included: whale recordings and annotations are the MarineMammalBench field set; OZF is public (Zenodo 15507508). Voxaboxen 0.2.0 (commit 20cf966) was run with four documented one-line inference patches (see the protocol document).
