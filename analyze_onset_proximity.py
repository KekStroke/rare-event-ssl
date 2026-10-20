"""Does the one-token/one-instance limitation explain the remaining misses on overlapping calls?
For each reference (R3 instance decoder, dev<=100 operating point per fold, 3 seeds): distance from its onset to the nearest
other reference onset in the same recording; miss categories: 'same-token' (<=20 ms: two onsets in one token, unrepresentable),
'within d_min' (<= fold's chosen d_min: the second peak is suppressed by the minimum peak distance), 'other'.
Also: for missed overlapping refs, was there an onset peak (>=0.05) within +-1 token of the reference onset (the branch saw it)?"""
import sys, json, numpy as np
from scipy.signal import find_peaks
from common import ROOT, DATA, SR, FOLDS, ann_for, NOISE
import common
from recall_far import TAU
names = sys.argv[1:] or [f'fte_R3_seed{s}' for s in range(3)]
MIN_DUR = 0.15
def inst_events(e, th_on, d_min, min_score, tok_s):
    pres, on, ld = e[:, 0], e[:, 1], e[:, 2]; pk, _ = find_peaks(on, height=th_on, distance=max(1, int(d_min / tok_s))); out = []
    for p in pk:
        a = p * tok_s; b = a + max(MIN_DUR, min(float(np.exp(ld[p])) * tok_s, 5.0)); i1 = min(len(pres), int(b / tok_s) + 1)
        if float(on[p] * pres[p:i1].mean()) < min_score: continue
        out.append((a, b))
    return out
tot = dict(n_ov=0, miss_ov=0, same_tok=0, within_dmin=0, other=0, miss_other_peak=0, miss_other_nopeak=0, n_iso=0, miss_iso=0)
gap_all = []; gap_missed = []
for name in names:
    z = np.load(ROOT / 'results' / f'{name}_tracks.npz'); tok_s = float(z['tok_s']); common.TOK_S = tok_s
    EV = {k[4:]: z[k].astype(np.float32) for k in z.files if k.startswith('on__')}
    thr = json.load(open(ROOT / 'results' / f'decode_instances_{name}_b100.json'))['instance']['thresholds']
    for k in range(3):
        th_on, d_min, ms = thr[k]
        for s in FOLDS[k]:
            R = ann_for(s); preds = inst_events(EV[s], th_on, d_min, ms, tok_s); on = EV[s][:, 1]
            used = np.zeros(len(preds), bool); matched = set()
            for ri, (r0, r1, _) in enumerate(R):
                best, bd = -1, 1e9
                for j, (p0, p1) in enumerate(preds):
                    if used[j] or abs(p0 - r0) > TAU or abs(p1 - r1) > max(TAU, 0.5 * (r1 - r0)): continue
                    if abs(p0 - r0) < bd: best, bd = j, abs(p0 - r0)
                if best >= 0: used[best] = True; matched.add(ri)
            for ri, (r0, r1, _) in enumerate(R):
                ov = any(j != ri and r0 < R[j][1] and R[j][0] < r1 for j in range(len(R)))
                gap = min([abs(R[j][0] - r0) for j in range(len(R)) if j != ri] or [9e9])
                if not ov:
                    tot['n_iso'] += 1; tot['miss_iso'] += ri not in matched; continue
                tot['n_ov'] += 1; gap_all.append(gap)
                if ri in matched: continue
                tot['miss_ov'] += 1; gap_missed.append(gap)
                if gap <= tok_s: tot['same_tok'] += 1
                elif gap <= d_min: tot['within_dmin'] += 1
                else:
                    tot['other'] += 1; i = int(r0 / tok_s)
                    if on[max(0, i - 1): i + 2].max() >= 0.05: tot['miss_other_peak'] += 1
                    else: tot['miss_other_nopeak'] += 1
n = len(names)
print(f"{n} runs (R3 instance, dev<=100 op points). Overlapping refs per run: {tot['n_ov']//n}, missed per run: {tot['miss_ov']/n:.1f} (recall {1-tot['miss_ov']/tot['n_ov']:.3f}); isolated missed per run {tot['miss_iso']/n:.1f} of {tot['n_iso']//n}")
print(f"  missed overlapping by nearest-other-onset distance: same token (<=20 ms) {tot['same_tok']/n:.1f}  within fold d_min {tot['within_dmin']/n:.1f}  other {tot['other']/n:.1f} (of which onset peak>=0.05 present at ref onset {tot['miss_other_peak']/n:.1f}, absent {tot['miss_other_nopeak']/n:.1f})")
ga = np.array(gap_all); gm = np.array(gap_missed)
for d in [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]:
    print(f"  nearest other onset <= {d:.2f}s: all overlapping refs {np.mean(ga<=d):.2f}  missed overlapping refs {np.mean(gm<=d):.2f}")
print('  d_min chosen per run/fold:', end=' ')
for name in names: print(json.load(open(ROOT / 'results' / f'decode_instances_{name}_b100.json'))['instance']['thresholds'], end=' ')
print()
json.dump(dict(runs=names, totals=tot, gaps_all=ga.tolist(), gaps_missed=gm.tolist()), open(ROOT / 'results' / 'onset_proximity_R3.json', 'w'))
