# archive

Material that is not part of the detector paper but was produced in the same repository. Kept for the record; nothing here is needed to reproduce `paper/main.tex`.

## companion/

The companion study *What Moves Detection and What Moves the Boundary? Target Source, Encoder Block and Fine-Tuning in Self-Supervised Detection of Killer-Whale Calls under Limited Data* (draft v0.3, `companion/paper/`), hypotheses H1-H20 in `expdb/knowledge/hypotheses.json`.

- `code/` frozen probes (`probe.py`), F1-bootstrap (`bootstrap.py`), miss decomposition (`analyze_misses.py`, `audit_metric.py`), decoder variants on the onset-head models (`decode_variants.py`), unit-quality analysis (`analysis_units.py`), result table and F1(tau) plot (`summarize.py`). These import `common.py`, `recall_far.py` and `prep.py` from the repository root; run them from the root (`python archive/companion/code/probe.py ...`) or copy them there.
- `results/` probe and fine-tuning results of the target-source comparison (log-mel k-means, AVES k-means, data2vec, BEST-RQ, JEPA, hybrid, oracle-boundary and random-segment units, iteration-2 units, scratch), their bootstraps, recall-at-FAR table, miss analyses and example figures.
- `scripts/` the launch scripts of those series (`run_all.sh` ... `run_stride.sh`) and `run_ozf.sh`, the sequential predecessor of `scripts/run_ozf_arm.sh`.
- `docs/` the H21 report and the manuscript plan (Russian).

## proposals/boundary_units/

*Learning Where Units End* (proposal draft v0.1, 18 October 2026), deferred by scope decision; kept as a record of its hypotheses and gates.

## misc/

`bootstrap_overlap_matched.log`: console log of an early bootstrap run with the Voxaboxen field-FT variants, superseded by `results/bootstrap_instances_b100.json`.
