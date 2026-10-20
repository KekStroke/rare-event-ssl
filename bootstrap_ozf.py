"""H26 (OZF): paired recording-level bootstrap, R3 pretrained on the paired no-overlap corpus (fte_ozf_noov_seed0-2) vs the overlap corpus
(fte_ozf_ov_seed0-2). Same machinery as bootstrap_instances.py. Readouts: operating point (val-chosen thresholds, budget 400 FA per slowed hour),
matched global FAR 200/400/800 per slowed hour (= 100/200/400 per real hour), overlapping and isolated subsets, instance and frame decoders.
Usage: TSSSL_DATASET=ozf python bootstrap_ozf.py [budget=400] [B=2000]"""
import sys, os, json, numpy as np
assert os.environ.get('TSSSL_DATASET') == 'ozf'
import bootstrap_instances as bi
from bootstrap_instances import *
FARS = [200.0, 400.0, 800.0]
NAMES = {'ov': [f'fte_ozf_ov_seed{s}' for s in range(3)], 'noov': [f'fte_ozf_noov_seed{s}' for s in range(3)]}
C = {g: [run_counts(n) for n in names] for g, names in NAMES.items()}
groups = {f'{g}/inst': [c['instance'] for c in C[g]] for g in NAMES} | {f'{g}/frame': [c['frame'] for c in C[g]] for g in NAMES}
rng = np.random.default_rng(0); k = len(RECS); full = np.arange(k); IDX = [rng.integers(0, k, k) for _ in range(NB)]
LAB = ['op_recall', 'op_far', 'op_overlap', 'op_isolated'] + [f'R@{int(f)}' for f in FARS] + [f'OV@{int(f)}' for f in FARS] + [f'ISO@{int(f)}' for f in FARS]

def pooled_iso(group, ix):
    n = len(group); nref = NREF[ix].sum() * n; nov = NOV[ix].sum() * n; niso = nref - nov; h = hours[ix].sum() * n
    op = sum(g['op'][ix].sum(0) for g in group); tp = sum(g['sw_tp'][:, ix].sum(1) for g in group); npd = sum(g['sw_np'][:, ix].sum(1) for g in group); tpo = sum(g['sw_tpo'][:, ix].sum(1) for g in group)
    fars = (npd - tp) / h; o = np.argsort(fars)
    return op[0] / nref, (op[1] - op[0]) / h, op[2] / max(1, nov), (op[0] - op[2]) / max(1, niso), fars[o], (tp / nref)[o], (tpo / max(1, nov))[o], ((tp - tpo) / max(1, niso))[o]

def row(grp, ix):
    r, f, ro, ri, fs, rs, rov, ris = pooled_iso(grp, ix)
    return [r, f, ro, ri] + [at_far(fs, rs, F) for F in FARS] + [at_far(fs, rov, F) for F in FARS] + [at_far(fs, ris, F) for F in FARS]

print(f'OZF test: {len(RECS)} recordings, {NREF.sum()} references, {NOV.sum()} overlapping ({NOV.sum()/NREF.sum():.2f}), {hours.sum():.2f} slowed hours; budget {BUDGET} FA per slowed hour')
S = {}; per_seed = {}; fullrows = {}
for g, grp in groups.items():
    fr = row(grp, full); boots = np.array([row(grp, ix) for ix in IDX]); S[g] = boots; fullrows[g] = [round(float(x), 3) for x in fr]
    ci = lambda j: np.nanpercentile(boots[:, j], [2.5, 97.5]); nn = lambda j: float(np.isnan(boots[:, j]).mean())
    print(f"{g:11s} " + '  '.join(f"{lab} {fr[j]:.3f} [{ci(j)[0]:.3f},{ci(j)[1]:.3f}]" + (f" undef {nn(j):.2f}" if nn(j) > 0.01 else '') for j, lab in enumerate(LAB) if lab != 'op_far') + f"  op_FAR {fr[1]:.1f}")
    per_seed[g] = [[round(float(x), 3) for x in row([c], full)] for c in grp]
    for i, r in enumerate(per_seed[g]): print(f"    seed{i}: op R {r[0]:.3f} FAR {r[1]:.0f} ov {r[2]:.3f} iso {r[3]:.3f} | " + ' '.join(f"{LAB[j]} {r[j]:.3f}" for j in range(4, len(LAB))))
res = {}
print('\npaired differences (recording-level bootstrap over 13 test recordings, recording carries all seeds; conditional on the trained models):')
for a, b in [('noov/inst', 'ov/inst'), ('noov/frame', 'ov/frame'), ('ov/inst', 'ov/frame'), ('noov/inst', 'noov/frame')]:
    d = S[a] - S[b]; rr = {}
    for j, lab in enumerate(LAB):
        dj = d[:, j]; ok = ~np.isnan(dj); dj = dj[ok]
        rr[lab] = (round(float(dj.mean()), 3), *[round(float(x), 3) for x in np.percentile(dj, [2.5, 97.5])], round(float((dj > 0).mean()), 3), int(ok.sum())) if ok.sum() >= 100 else (float('nan'),) * 4 + (int(ok.sum()),)
    fmt = lambda r: f"{r[0]:+.3f} [{r[1]:+.3f},{r[2]:+.3f}] P>0 {r[3]:.2f} n={r[4]}"
    print(f"  {a} - {b}:")
    for lab in LAB: print(f"      {lab:12s} {fmt(rr[lab])}")
    res[f'{a} - {b}'] = rr
json.dump(dict(budget=BUDGET, B=NB, fars=FARS, labels=LAB, full=fullrows, per_seed=per_seed, diffs=res, n_rec=len(RECS), n_ref=int(NREF.sum()), n_overlap=int(NOV.sum())),
          open(ROOT / 'results' / f'bootstrap_ozf_b{BUDGET}.json', 'w'), indent=1)
print('saved results/bootstrap_ozf_b%d.json' % BUDGET)
