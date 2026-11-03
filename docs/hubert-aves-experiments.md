# HuBERT / AVES transfer on the retained Marine Mammal SSL corpus

This experiment compares published HuBERT-base-ls960, published AVES-base-bio,
the retained historical AVES marine checkpoint, and two new domain-adapted
encoders initialized separately from the published models. The historical best
is an exploratory reference: downstream test results influenced its selection.
Its original pretraining exposure is not certified by the new corpus audit.

Published AVES-bio declares FSD50K and subsets of AudioSet/VGGSound as its SSL
sources; Watkins and HICEAS are instead named as BEANS downstream evaluation
and development tasks. Their use during development of the AVES family must
remain visible in the comparison. The publications do not provide a recording
deduplication audit against the web corpora, so they establish neither sample
overlap nor certified absence of overlap. See the [official model inventory](https://github.com/earthspecies/aves#pretrained-models)
and [AVES data/evaluation description](https://arxiv.org/html/2210.14493v1#S3).

The primary comparison freezes each encoder and fits the same supervised heads.
All models share immutable cohorts, preprocessing, head search and matching
rules. Selection uses validation only. Test predictions never select layers,
regularization, thresholds, pretraining checkpoints or stopping time. Published
HuBERT retains its published waveform normalization; AVES retains its own.

| Task | Most informative metric | Additional measurements / limitations |
|---|---|---|
| Avacha call types | macro F1 | balanced accuracy, per-class support; conservative year groups, one class absent in validation |
| Watkins species | macro F1 | balanced accuracy; preserve official splits, quarantine conflicting exact duplicates |
| Avacha presence | average precision (AP) | ROC-AUC and F1 at validation-selected threshold |
| HICEAS presence | AP | ROC-AUC/F1, 10 s windows with 5 s stride; uncertainty clustered by recording |
| Hackathon presence | AP | ROC-AUC/F1; presence labels do not establish SRKW ecotype |
| ONC external presence | AP | reuse Avacha presence head, max five-second chunk probability; no ONC fitting |
| Avacha continuous field | recall at dev-set 50/100/300 false alarms/hour | actual test false alarms/hour, event F1 and AP; existing recording folds |
| Dominica clicks | event AP/F1 at 1/5/10 ms | localization error and false alarms/hour; a 20 ms encoder grid is an explicit coarse baseline |
| Manual sound events | event AP | event F1/false alarms/hour; auxiliary because provenance remains unresolved |
| Voices examples | shared-ontology macro F1 | accuracy, ontology coverage; reuse Watkins head, auxiliary small external set |

DCLDE2027 and a proposed Orcasound deployment holdout are N/A in this stage,
as agreed. The 300 h main Orcasound pretraining pool is unlabeled. Hackathon is
a separate labeled downstream cohort, with source/session exclusions from SSL.

Audio uses channel zero, physical 16 kHz resampling and unpadded five-second
contexts (a shorter-than-convolution clip receives explicitly reported minimum
padding). Transformer layers 6, 9 and 12 are one-based. Clip heads use frame mean
and standard deviation, training-only standardization and balanced logistic
regression with C in 0.1/1/10. No examples move between upstream splits. Unknown
manual annotations are never converted to negative labels. Native UEM rounding
is tolerated only within the declared half-quantum; source files are unchanged.

Feature extraction runs on GPU, supervised head fitting on MLS CPU/payg. A
single-owner, resumable SQLite cache avoids one feature file per clip. Cache
identity includes source hashes/intervals, the full registered model identity,
layers and preprocessing. Per-model cache allowance is 128 GiB. Temporal heads
operate on physical frame centers and retain the five-second context boundaries.

The old `scratch` run was actually a fresh AVES-initialized marine adaptation.
Its relevant recipe is HuBERT cosine NCE (K=200, projection dimension 256,
temperature 0.1), masked loss only plus feature penalty 10; frozen AVES block 6
provides clustering features. The new codebook is fitted on training audio only.
New SSL validation holds out source sessions/deployments; it replaces the old
run's test-influenced checkpoint selection. Resume must restore optimizer,
scheduler, AMP, RNG and data position, rather than silently restarting a run.

Reports retain per-example predictions and class/recording support for paired
model comparisons. Bootstrap units are source groups/recordings, not overlapping
windows. Very few independent source groups and tiny external cohorts must stay
visible next to intervals. Metrics from different tasks are not collapsed into
an unqualified overall score. The historical 0.9284 composite is not a result
under this protocol and must not appear as one.

HICEAS retains its published recording splits; some recording dates occur in
more than one split. Bootstrap uncertainty clusters by UTC source day (nine
test days), rather than treating the 1,485 overlapping test windows or all 135
recordings as independent deployments. This does not establish performance on
an unseen deployment. Watkins numeric names do not establish tape independence.

Run the staged pipeline in this order (absolute paths in MLS configs):

1. `python -m marine_ssl.experiments.run freeze --config EXPERIMENT.json`
2. GPU: `python -m marine_ssl.experiments.run features --config EXPERIMENT.json`
3. CPU: `python -m marine_ssl.experiments.run evaluate --config EXPERIMENT.json`
4. CPU: `python -m marine_ssl.experiments.compare REPORT_ROOT`

The optional `experiments` dependency set records the verified image API. Use
the MLS image's Torch installation; installing a training stack on Jupyter is
outside this workflow. Each submitted job freezes and hashes its small source
snapshot. Later repository edits do not mutate a queued/running job's source.

Retained historical Fairseq files must be read in an isolated CPU job with
genuine Fairseq Dictionary/configuration classes. The verified conversion used
official Fairseq commit `4a388e64cd646ed7d7ad8de8fae55df2b8eea91d`, OmegaConf
2.0.6 and Hydra 1.0.7. Record every backbone mapping and explicit training-head
exclusion. PyTorch's documented positional weight-norm g/v aliases may migrate
to `parametrizations.weight.original0/original1` without changing tensors;
require full key/shape coverage and an exact encoder-output roundtrip. Never
replace missing classes with pickle stubs or accept a partially initialized
encoder. The converted historical encoder remains subject to its selection bias.
