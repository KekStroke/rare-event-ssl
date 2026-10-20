"""Instance decoding from the event branch (presence, onset, log-duration) saved in results/<name>_tracks.npz (on__ = [T,3]).

Boxes: onset peaks (height th_on, min distance d_min) -> start = peak time, end = start + exp(logdur[peak]) * tok_s,
score = onset prob * mean presence inside the box; overlapping boxes are allowed; class = argmax of summed call probabilities inside.
Also the frame decoder baseline (two-threshold on p_call) from the same run, for a same-model comparison.
Protocol: thresholds chosen on the two dev recordings per fold (max class-agnostic recall s.t. dev FAR <= budget), test pooled.
Reports: strict recall (tau 0.5), actual test FAR, coverage>=50%, boundary/absorbed counts, FAR-matched recall, and recall on the
OVERLAPPING subset (references whose annotation overlaps another reference) vs the isolated subset.
Usage: python decode_instances.py <name> [budget]
"""
import sys, json, itertools, numpy as np
from scipy.signal import find_peaks
from common import ROOT, DATA, SR, FOLDS, match, ann_for, NOISE, CLASSES, splits
import common
from recall_far import events_from_pcall, stats, TAU

name = sys.argv[1]; budget = int(sys.argv[2]) if len(sys.argv) > 2 else 100
z = np.load(ROOT / 'results' / f'{name}_tracks.npz'); tok_s = float(z['tok_s']); common.TOK_S = tok_s
P = {k[3:]: z[k].astype(np.float32) for k in z.files if k.startswith('P__')}
EV = {k[4:]: z[k].astype(np.float32) for k in z.files if k.startswith('on__')}
refs = {s: ann_for(s) for s in P}; hours = {s: len(np.load(DATA / 'field16k' / f'{s}.npy')) / SR / 3600 for s in P}
MIN_DUR = 0.15

def overlapping_flags(R):
    f = []
    for i, (a, b, _) in enumerate(R):
        f.append(any(j != i and a < R[j][1] and R[j][0] < b for j in range(len(R))))
    return f

def instance_events(s, th_on, d_min, min_score=0.0):
    e = EV[s]; pres, on, ld = e[:, 0], e[:, 1], e[:, 2]
    pk, _ = find_peaks(on, height=th_on, distance=max(1, int(d_min / tok_s))); out = []
    for p in pk:
        dur = float(np.exp(ld[p])) * tok_s; a = p * tok_s; b = a + max(MIN_DUR, min(dur, 5.0))
        i0, i1 = p, min(len(pres), int(b / tok_s) + 1); score = float(on[p] * pres[i0:i1].mean())
        if score < min_score: continue
        cls = CLASSES[int(P[s][i0:i1, :NOISE].sum(0).argmax())]
        out.append((a, b, cls))
    return out

def frame_events(s, det, edge): return events_from_pcall(1.0 - P[s][:, NOISE], det, edge)

def analyse(recs, flags):
    nref = sum(len(r) for _, r in recs); tp = 0; vis = 0; bonly = 0; merged = 0
    tp_ov = 0; n_ov = 0; tp_iso = 0; n_iso = 0
    for (preds, R), fl in zip(recs, flags):
        used = np.zeros(len(preds), bool); matched = set()
        for ri, (r0, r1, _) in enumerate(R):
            best, bd = -1, 1e9
            for j, (p0, p1, _) in enumerate(preds):
                if used[j] or abs(p0 - r0) > TAU or abs(p1 - r1) > max(TAU, 0.5 * (r1 - r0)): continue
                if abs(p0 - r0) < bd: best, bd = j, abs(p0 - r0)
            if best >= 0: used[best] = True; matched.add(ri)
        tp += len(matched)
        for ri, r in enumerate(R):
            if fl[ri]: n_ov += 1; tp_ov += ri in matched
            else: n_iso += 1; tp_iso += ri in matched
            if sum(max(0, min(p[1], r[1]) - max(p[0], r[0])) for p in preds) / max(1e-6, r[1] - r[0]) >= 0.5: vis += 1
            if ri in matched: continue
            ov = [j for j, p in enumerate(preds) if p[0] < r[1] and p[1] > r[0]]
            if ov and any(not used[j] for j in ov): bonly += 1
            elif ov: merged += 1
    return dict(recall=tp / nref, visible50=vis / nref, boundary_only=bonly, merged=merged, recall_overlapping=tp_ov / max(1, n_ov), n_overlapping=n_ov, recall_isolated=tp_iso / max(1, n_iso))

GRID_FRAME = list(itertools.product([0.1, 0.2, 0.3, 0.5, 0.7, 0.9], [0.05, 0.1, 0.2, 0.3, 0.5]))
D_MIN = [0.1, 0.2, 0.3, 0.5] if common.DATASET == 'whale' else [0.04, 0.06, 0.1, 0.2, 0.3]   # OZF protocol: shorter calls
GRID_INST = list(itertools.product([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], D_MIN, [0.0, 0.1, 0.2, 0.3]))
results = {}
for variant in ['frame', 'instance']:
    pooled = []; flags = []; chosen = []
    for k, (_, devs, test) in enumerate(splits()):
        best, br = None, -1; dev_h = sum(hours[s] for s in devs)
        grid = GRID_FRAME if variant == 'frame' else GRID_INST
        for prm in grid:
            dec = (lambda s: frame_events(s, *prm)) if variant == 'frame' else (lambda s: instance_events(s, *prm))
            recs = [(dec(s), [(a, b, 'c') for a, b, _ in refs[s]]) for s in devs]
            rec, far, _, _ = stats(recs, dev_h)
            if far <= budget and rec > br: br, best = rec, prm
        if best is None: best = grid[-1]
        chosen.append(best)
        dec = (lambda s: frame_events(s, *best)) if variant == 'frame' else (lambda s: instance_events(s, *best))
        for s in test:
            pooled.append((dec(s), [(a, b, 'c') for a, b, _ in refs[s]])); flags.append(overlapping_flags(refs[s]))
    test_h = sum(hours[s] for _, _, te in splits() for s in te)      # whale: all 17 recordings; OZF: the 13 test recordings only (dev hours must not enter the test FAR)
    A = analyse(pooled, flags); rec, far, prec, f1 = stats(pooled, test_h)
    curve = []
    for prm0 in ([0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9] if variant == 'frame' else [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]):
        prm = (prm0,) + tuple(chosen[0][1:])
        dec = (lambda s: frame_events(s, *prm)) if variant == 'frame' else (lambda s: instance_events(s, *prm))
        r_, f_, _, _ = stats([(dec(s), [(a, b, 'c') for a, b, _ in refs[s]]) for s in P], sum(hours.values())); curve.append((f_, r_))
    curve.sort(); fars = np.array([c[0] for c in curve]); rr = np.array([c[1] for c in curve])
    interp = {B: (round(float(np.interp(B, fars, rr)), 3) if fars.min() <= B <= fars.max() else None) for B in [50, 100, 300]}
    results[variant] = dict(recall=round(rec, 3), test_far=round(far, 1), precision=round(prec, 3), visible50=round(A['visible50'], 3), boundary_only=A['boundary_only'], merged=A['merged'],
                            recall_overlapping=round(A['recall_overlapping'], 3), n_overlapping=A['n_overlapping'], recall_isolated=round(A['recall_isolated'], 3), thresholds=[list(c) for c in chosen], far_matched=interp)
    print(f"{name:26s} {variant:8s} dev≤{budget}/h → test FAR {far:5.0f}  strict R {rec:.3f}  overlapping-subset R {A['recall_overlapping']:.3f} (n={A['n_overlapping']})  isolated R {A['recall_isolated']:.3f}  cov≥50% {A['visible50']:.3f}  bnd {A['boundary_only']:3d} abs {A['merged']:3d} | FAR-matched R@50/100/300 {interp[50]} / {interp[100]} / {interp[300]}")
json.dump(results, open(ROOT / 'results' / f'decode_instances_{name}_b{budget}.json', 'w'), indent=1)
