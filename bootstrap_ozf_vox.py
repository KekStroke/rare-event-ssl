"""OZF: paired recording-level bootstrap of our R3 (both pretraining arms, instance decoder, 3 seeds) against Voxaboxen trained on OZF (1 model),
at the operating points (val-chosen thresholds, budget 400 FA per slowed hour) and at matched global FAR 200 (the only point both reach in >80% of resamples).
Usage: TSSSL_DATASET=ozf python bootstrap_ozf_vox.py [budget=400] [B=2000]"""
import sys, os, json, numpy as np
assert os.environ.get('TSSSL_DATASET') == 'ozf'
import bootstrap_instances as bi
from bootstrap_instances import *
z = np.load(ROOT / 'results' / 'voxozf_counts.npz'); VOX = dict(op=z['op'], sw_tp=z['sw_tp'], sw_np=z['sw_np'], sw_tpo=z['sw_tpo'])
NAMES = {'ov': [f'fte_ozf_ov_seed{s}' for s in range(3)], 'noov': [f'fte_ozf_noov_seed{s}' for s in range(3)]}
C = {g: [run_counts(n)['instance'] for n in names] for g, names in NAMES.items()}
groups = {'R3 ov/inst': C['ov'], 'R3 noov/inst': C['noov'], 'VOX-OZF': [VOX]}
rng = np.random.default_rng(0); k = len(RECS); full = np.arange(k); IDX = [rng.integers(0, k, k) for _ in range(NB)]
def row(grp, ix):
    r, f, ro, fs, rs, rov = pooled(grp, ix)
    return [r, f, ro, at_far(fs, rs, 200.0), at_far(fs, rov, 200.0), at_far(fs, rs, 400.0), at_far(fs, rov, 400.0)]
LAB = ['op_recall', 'op_far', 'op_overlap', 'R@200', 'OV@200', 'R@400', 'OV@400']
S = {}
for g, grp in groups.items():
    fr = row(grp, full); boots = np.array([row(grp, ix) for ix in IDX]); S[g] = boots
    ci = lambda j: np.nanpercentile(boots[:, j], [2.5, 97.5]); nn = lambda j: float(np.isnan(boots[:, j]).mean())
    print(f"{g:13s} " + '  '.join(f"{lab} {fr[j]:.3f} [{ci(j)[0]:.3f},{ci(j)[1]:.3f}]" + (f" undef {nn(j):.2f}" if nn(j) > 0.01 else '') for j, lab in enumerate(LAB)))
res = {}
print('\npaired differences (13 test recordings, recording carries all seeds; Voxaboxen one model):')
for a, b in [('R3 ov/inst', 'VOX-OZF'), ('R3 noov/inst', 'VOX-OZF')]:
    d = S[a] - S[b]; rr = {}
    for j, lab in enumerate(LAB):
        dj = d[:, j]; ok = ~np.isnan(dj); dj = dj[ok]
        rr[lab] = (round(float(dj.mean()), 3), *[round(float(x), 3) for x in np.percentile(dj, [2.5, 97.5])], round(float((dj > 0).mean()), 3), int(ok.sum())) if ok.sum() >= 100 else (float('nan'),) * 4 + (int(ok.sum()),)
    print(f"  {a} - {b}: " + '  '.join(f"{lab} {rr[lab][0]:+.3f} [{rr[lab][1]:+.3f},{rr[lab][2]:+.3f}] n={rr[lab][4]}" for lab in LAB))
    res[f'{a} - {b}'] = rr
json.dump(dict(budget=BUDGET, B=NB, diffs=res), open(ROOT / 'results' / f'bootstrap_ozf_vox_b{BUDGET}.json', 'w'), indent=1); print('saved')
