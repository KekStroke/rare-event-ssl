# tsssl: a compact single-pass detector of overlapping killer-whale calls

Code, launch scripts, saved results, experiment database and manuscript for
*A Compact Single-Pass Detector of Overlapping Killer-Whale Calls Trained on Synthetic-Mixture Labels: What the Components Contribute* (`paper/main.tex`, draft v0.7).

The detector is a 17.8M-parameter encoder with one event branch (presence, onset, log-duration per 20 ms token). The branch is trained already during pretraining on synthetic mixtures whose event positions are known by construction, fine-tuned on field labels, and decoded by onset-peak picking without box NMS. The paper measures what each component contributes to class-agnostic recall at matched false-alarm rate, on 17 killer-whale field recordings and on the public Overlapping Zebra Finch data (OZF).

## Layout

```
common.py              audio front-end, synthetic-mixture construction (overlap / paired no-overlap arms), encoders
                       (vanilla and Gemma-style "modern" block), event targets, decoders, scorer. TSSSL_DATASET=whale|ozf
expdb.py               experiment database: hardware fingerprint, code snapshot, data checksums, curves, results per run
recall_far.py          class-agnostic recall at a false-alarm budget (shared scorer)
prep.py, prep_ozf.py   data preparation (whale lab clips + field recordings; OZF from Zenodo 15507508, slowed x2)
pretrain.py            pretraining; --event_head adds the event branch (R2), --no_hubert removes the unit loss (R3)
finetune.py            field fine-tuning with the event branch; saves per-recording tracks to results/*_tracks.npz
decode_instances.py    instance decoder (onset peaks + log-duration) and frame decoder, thresholds chosen on dev under a FA budget
bootstrap_instances.py paired recording-level bootstrap of R1/R2/R3, their frame decoders and Voxaboxen (C1-C3, C5)
bootstrap_h24.py       the same for one pair of regimes: R3 no-overlap vs R3 (C6, default) or R3 vanilla vs R3 (C4)
bootstrap_ozf.py, bootstrap_ozf_vox.py, map_eval.py, eval_synth_tests.py   OZF: paired arms, Voxaboxen, mAP, synthetic test sets
vox_field_eval.py, vox_ozf_eval.py, voxaboxen_eval.py   Voxaboxen under our protocol (field FT, OZF, synthetic-only checkpoint)
analyze_onset_proximity.py, check_aggregation.py        diagnostics quoted in the paper (remaining misses; op vs matched-FAR aggregates)
scripts/               the launch scripts of every reported series (see below)
results/               every reported number: decode outputs, bootstraps, mAP (per-recording tracks *.npz are not in git, 96 MB)
expdb/                 expdb.sqlite (all runs), knowledge/hypotheses.json (H1-H26 with verdicts), docs/FINDINGS.md (generated)
docs/                  the protocols fixed before the Voxaboxen and OZF runs (Russian)
paper/                 main.tex, main.pdf
archive/               companion study on HuBERT-style targets (code, results, scripts, draft) and a deferred proposal
```

## Series and launch scripts

| Claim | Series | Script |
|---|---|---|
| C1-C3 | H21: regimes R1 (units), R2 (units + event branch), R3 (event branch only), 3 seeds | `scripts/run_proto.sh` |
| C4 | H22: R3 on the vanilla block | `scripts/run_r3van.sh`, `scripts/run_r3van_seeds.sh` |
| C5 | H23: Voxaboxen 0.2.0 fine-tuned on the field folds | `scripts/run_vox_field.sh` (+ `run_vox_fold1.sh` retry of fold 1), `run_vox_infer0.sh`, `run_vox_best.sh` |
| C6 | H24: R3 on the no-overlap synthetic corpus | `scripts/run_noov.sh` |
| OZF | H26: both arms on OZF, `scripts/run_ozf_arm.sh ov|noov`; Voxaboxen on OZF, `scripts/run_vox_ozf.sh` | |

Scripts assume the repository at `~/tsssl` with `venv/`; Voxaboxen scripts assume a separate `~/voxenv` and the CETACEANS checkout with the April-2026 synthetic-only checkpoint.

## Reproducing the tables

Whale (tracks in `results/` from `finetune.py`):

```
for n in fte_R1 fte_R2 fte_R3 fte_R3van fte_R3noov; do for s in 0 1 2; do python decode_instances.py ${n}_seed$s 100; done; done
python bootstrap_instances.py 100 2000            # Tables 1-2, 4-5: R1/R2/R3, frame decoders, Voxaboxen
python bootstrap_h24.py 100 2000                  # C6: R3 no-overlap vs R3
python bootstrap_h24.py 100 2000 R3van R3 h22     # C4: R3 vanilla vs R3
python analyze_onset_proximity.py                 # remaining misses on overlapping calls
```

OZF:

```
export TSSSL_DATASET=ozf
python bootstrap_ozf.py 400 2000; python bootstrap_ozf_vox.py 400 2000
for a in ov noov; do for s in 0 1 2; do python map_eval.py fte_ozf_${a}_seed$s 400; done; done
python eval_synth_tests.py
```

## Data

Not included. Whale recordings and annotations are the MarineMammalBench field set (17 recordings, 613 calls) and the lab clip corpus (45,429 clips); OZF is public (Zenodo 15507508). Voxaboxen 0.2.0 (commit 20cf966) was run with four documented one-line inference patches (`docs/TSSSL_voxaboxen_protocol_ru.md`).

## Environment

`requirements.txt` pins the versions recorded in `expdb/expdb.sqlite` (torch 2.11.0+cu128, Python 3.14, RTX 5090 Laptop GPU).
