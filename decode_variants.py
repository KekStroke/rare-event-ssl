"""Four decoder variants on saved full tracks (12-class softmax + onset posterior), tuned on dev only.

  base        two-threshold on p_call = 1 - p_noise (current decoder)
  class       base events split where the median-filtered argmax over call classes changes (cannot represent two simultaneous calls)
  onset       base events split at peaks of the onset posterior (peak >= th_on, min distance d_min); offset of the previous piece = next onset
  both        class split, then onset split inside each piece
Selection on the two dev recordings per fold: maximise strict class-agnostic recall subject to dev FAR <= budget.
Report on test: strict recall, actual FAR, boundary-only / merged counts, visible recall (>= 50 % of the reference covered),
and the FAR-matched recall from a detect-threshold sweep (interpolated at 50/100/300 FA/h) as the final curve characterisation.
Usage: python decode_variants.py <name> [budget]
"""
import sys, json, itertools, numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks
from common import ROOT, DATA, SR, FOLDS, match, ann_for, NOISE, CLASSES
import common
from recall_far import events_from_pcall, stats, TAU

name = sys.argv[1]; budget = int(sys.argv[2]) if len(sys.argv) > 2 else 100
z = np.load(ROOT / 'results' / f'{name}_tracks.npz'); tok_s = float(z['tok_s']); common.TOK_S = tok_s
P = {k[3:]: z[k].astype(np.float32) for k in z.files if k.startswith('P__')}
ON = {k[4:]: z[k].astype(np.float32) for k in z.files if k.startswith('on__')}
refs = {s: ann_for(s) for s in P}; hours = {s: len(np.load(DATA / 'field16k' / f'{s}.npy')) / SR / 3600 for s in P}
MIN_DUR = 0.15

def base_events(s, det, edge):
    return events_from_pcall(1.0 - P[s][:, NOISE], det, edge)

def class_split(s, evs, med=5):
    am = median_filter(P[s][:, :NOISE].argmax(1), size=med, mode='nearest'); out = []
    for a, b, _ in evs:
        i0, i1 = int(a / tok_s), max(int(a / tok_s) + 1, int(b / tok_s)); seg = am[i0:i1]
        cuts = [0] + [j for j in range(1, len(seg)) if seg[j] != seg[j - 1]] + [len(seg)]
        for c0, c1 in zip(cuts[:-1], cuts[1:]):
            if (c1 - c0) * tok_s >= MIN_DUR: out.append(((i0 + c0) * tok_s, (i0 + c1) * tok_s, CLASSES[int(seg[c0])]))
    return out

def onset_split(s, evs, th_on, d_min):
    on = ON[s]; out = []
    for a, b, c in evs:
        i0, i1 = int(a / tok_s), max(int(a / tok_s) + 1, int(b / tok_s)); seg = on[i0:i1]
        pk, _ = find_peaks(seg, height=th_on, distance=max(1, int(d_min / tok_s)))
        pk = [p for p in pk if p * tok_s >= MIN_DUR and (len(seg) - p) * tok_s >= MIN_DUR]     # a cut must leave two valid pieces
        cuts = [0] + sorted(pk) + [len(seg)]
        for c0, c1 in zip(cuts[:-1], cuts[1:]):
            if (c1 - c0) * tok_s >= MIN_DUR: out.append(((i0 + c0) * tok_s, (i0 + c1) * tok_s, c))
    return out

def decode_variant(s, variant, det, edge, th_on=0.5, d_min=0.3):
    ev = base_events(s, det, edge)
    if variant in ('class', 'both'): ev = class_split(s, ev)
    if variant in ('onset', 'both'): ev = onset_split(s, ev, th_on, d_min)
    return ev

def overlaps(p, r): return p[0] < r[1] and p[1] > r[0]
def coverage(preds, r): return sum(max(0, min(p[1], r[1]) - max(p[0], r[0])) for p in preds) / max(1e-6, r[1] - r[0])

def analyse(recs):
    """strict recall, FAR, visible(>=50% covered), boundary-only, merged"""
    nref = sum(len(r) for _, r in recs); tp = 0; vis = 0; bonly = 0; merged = 0; npred = sum(len(p) for p, _ in recs)
    for preds, R in recs:
        used = np.zeros(len(preds), bool); matched = set()
        for ri, (r0, r1, _) in enumerate(R):
            best, bd = -1, 1e9
            for j, (p0, p1, _) in enumerate(preds):
                if used[j] or abs(p0 - r0) > TAU or abs(p1 - r1) > max(TAU, 0.5 * (r1 - r0)): continue
                if abs(p0 - r0) < bd: best, bd = j, abs(p0 - r0)
            if best >= 0: used[best] = True; matched.add(ri)
        tp += len(matched)
        for ri, r in enumerate(R):
            if coverage(preds, r) >= 0.5: vis += 1
            if ri in matched: continue
            ov = [j for j, p in enumerate(preds) if overlaps(p, r)]
            if ov and any(not used[j] for j in ov): bonly += 1
            elif ov: merged += 1
    return dict(recall=tp / nref, visible50=vis / nref, boundary_only=bonly, merged=merged, npred=npred, tp=tp)

GRID_BASE = list(itertools.product([0.1, 0.2, 0.3, 0.5, 0.7, 0.9], [0.05, 0.1, 0.2, 0.3, 0.5]))
GRID_ON = list(itertools.product([0.2, 0.35, 0.5, 0.7], [0.2, 0.3, 0.5]))
results = {}
for variant in ['base', 'class', 'onset', 'both']:
    if variant in ('onset', 'both') and not ON: continue
    pooled = []; chosen = []
    for k in range(3):
        test = FOLDS[k]; others = [f for i, f in enumerate(FOLDS) if i != k]; devs = [others[0][0], others[1][0]]
        best, br = None, -1; dev_h = sum(hours[s] for s in devs)
        for det, edge in GRID_BASE:
            for th_on, d_min in (GRID_ON if variant in ('onset', 'both') else [(0.5, 0.3)]):
                recs = [(decode_variant(s, variant, det, edge, th_on, d_min), [(a, b, 'c') for a, b, _ in refs[s]]) for s in devs]
                rec, far, _, _ = stats(recs, dev_h)
                if far <= budget and rec > br: br, best = rec, (det, edge, th_on, d_min)
        if best is None: best = (0.9, 0.5, 0.5, 0.3)
        chosen.append(best); pooled += [(decode_variant(s, variant, *best), [(a, b, 'c') for a, b, _ in refs[s]]) for s in test]
    A = analyse(pooled); rec, far, prec, f1 = stats(pooled, sum(hours.values()))
    # FAR-matched curve: sweep det with the chosen edge/onset params of fold 0, all recordings
    curve = []
    for det in [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        _, e, t, d = chosen[0]
        recs = [(decode_variant(s, variant, det, e, t, d), [(a, b, 'c') for a, b, _ in refs[s]]) for s in P]
        r_, f_, _, _ = stats(recs, sum(hours.values())); curve.append((f_, r_))
    curve.sort(); fars = np.array([c[0] for c in curve]); rr = np.array([c[1] for c in curve])
    interp = {B: (round(float(np.interp(B, fars, rr)), 3) if fars.min() <= B <= fars.max() else None) for B in [50, 100, 300]}
    results[variant] = dict(recall=round(rec, 3), test_far=round(far, 1), precision=round(prec, 3), visible50=round(A['visible50'], 3),
                            boundary_only=A['boundary_only'], merged=A['merged'], thresholds=chosen, far_matched=interp)
    print(f"{name:22s} {variant:6s} dev≤{budget}/h → test FAR {far:5.0f}  strict R {rec:.3f}  visible≥50% {A['visible50']:.3f}  boundary-only {A['boundary_only']:3d}  merged {A['merged']:3d}  | FAR-matched R@50/100/300 {interp[50]} / {interp[100]} / {interp[300]}")
json.dump(results, open(ROOT / 'results' / f'decode_variants_{name}_b{budget}.json', 'w'), indent=1)
