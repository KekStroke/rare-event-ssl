"""H24: paired recording-level bootstrap, R3 pretrained on the no-overlap synthetic corpus (fte_R3noov_seed0-2) vs the default corpus (fte_R3_seed0-2).
Same machinery as bootstrap_instances.py (imported). Primary readout: recall on the overlapping-reference subset at matched FAR; secondary: R@100/150/61, isolated recall.
Usage: python bootstrap_h24.py [budget=100] [B=2000] [groupA=R3noov] [groupB=R3] [out=h24]
H22 (vanilla block vs modern block) is the same script with groups R3van R3 and out h22."""
import sys, json, numpy as np
import bootstrap_instances as bi
from bootstrap_instances import *

GA = sys.argv[3] if len(sys.argv) > 3 else 'R3noov'; GB = sys.argv[4] if len(sys.argv) > 4 else 'R3'; OUT = sys.argv[5] if len(sys.argv) > 5 else 'h24'
NAMES = {g: [f'fte_{g}_seed{s}' for s in range(3)] for g in (GB, GA)}
C = {g: [run_counts(n) for n in names] for g, names in NAMES.items()}
groups = {f'{g}/inst': [c['instance'] for c in C[g]] for g in NAMES} | {f'{g}/frame': [c['frame'] for c in C[g]] for g in NAMES}
rng = np.random.default_rng(0); k = len(RECS); full = np.arange(k); IDX = [rng.integers(0, k, k) for _ in range(NB)]
vfar = 61.4
LAB = ['op_recall', 'op_far', 'overlap_recall', 'R@100', 'R@150', 'R@61', 'R@300', 'OV@100', 'OV@61', 'OV@150', 'ISO@100', 'op_isolated']

def pooled_iso(group, ix):
    n = len(group); nref = NREF[ix].sum() * n; nov = NOV[ix].sum() * n; niso = nref - nov; h = hours[ix].sum() * n
    op = sum(g['op'][ix].sum(0) for g in group); tp = sum(g['sw_tp'][:, ix].sum(1) for g in group); npd = sum(g['sw_np'][:, ix].sum(1) for g in group); tpo = sum(g['sw_tpo'][:, ix].sum(1) for g in group)
    fars = (npd - tp) / h; o = np.argsort(fars)
    return op[0] / nref, (op[1] - op[0]) / h, op[2] / max(1, nov), (op[0] - op[2]) / max(1, niso), fars[o], (tp / nref)[o], (tpo / max(1, nov))[o], ((tp - tpo) / max(1, niso))[o]

def row(grp, ix):
    r, f, ro, ri, fs, rs, rov, ris = pooled_iso(grp, ix)
    return [r, f, ro, at_far(fs, rs, 100), at_far(fs, rs, 150), at_far(fs, rs, vfar), at_far(fs, rs, 300), at_far(fs, rov, 100), at_far(fs, rov, vfar), at_far(fs, rov, 150), at_far(fs, ris, 100), ri]

S = {}; per_seed = {}
for g, grp in groups.items():
    fullrow = row(grp, full); boots = np.array([row(grp, ix) for ix in IDX]); S[g] = boots
    ci = lambda j: np.nanpercentile(boots[:, j], [2.5, 97.5])
    print(f"{g:13s} " + '  '.join(f"{lab} {fullrow[j]:.3f} [{ci(j)[0]:.3f},{ci(j)[1]:.3f}]" for j, lab in enumerate(LAB) if lab != 'op_far') + f"  op_FAR {fullrow[1]:.1f}")
    per_seed[g] = [[round(float(x), 3) for x in row([c], full)] for c in grp]
    for i, r in enumerate(per_seed[g]): print(f"    seed{i}: op R {r[0]:.3f} FAR {r[1]:.0f} ov {r[2]:.3f} iso {r[11]:.3f} | R@100 {r[3]:.3f} R@61 {r[5]:.3f} | OV@100 {r[7]:.3f} OV@61 {r[8]:.3f} ISO@100 {r[10]:.3f}")
res = {}
print('\npaired differences (recording-level bootstrap, recording carries all seeds):')
for a, b in [(f'{GA}/inst', f'{GB}/inst'), (f'{GA}/frame', f'{GB}/frame'), (f'{GA}/inst', f'{GA}/frame'), (f'{GB}/inst', f'{GB}/frame')]:
    d = S[a] - S[b]; rr = {}
    for j, lab in enumerate(LAB):
        dj = d[:, j]; ok = ~np.isnan(dj); dj = dj[ok]
        rr[lab] = (round(float(dj.mean()), 3), *[round(float(x), 3) for x in np.percentile(dj, [2.5, 97.5])], round(float((dj > 0).mean()), 3), int(ok.sum()))
    fmt = lambda r: f"{r[0]:+.3f} [{r[1]:+.3f},{r[2]:+.3f}] P>0 {r[3]:.2f} n={r[4]}"
    print(f"  {a} - {b}:"); 
    for lab in LAB: print(f"      {lab:14s} {fmt(rr[lab])}")
    res[f'{a} - {b}'] = rr
json.dump(dict(budget=BUDGET, B=NB, labels=LAB, per_seed=per_seed, diffs=res, full={g: [round(float(x), 3) for x in row(grp, full)] for g, grp in groups.items()}), open(ROOT / 'results' / f'bootstrap_{OUT}_b{BUDGET}.json', 'w'), indent=1)
print(f'saved results/bootstrap_{OUT}_b{BUDGET}.json')
