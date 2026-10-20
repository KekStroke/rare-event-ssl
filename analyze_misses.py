"""Where do the misses come from? Decomposition on saved call-probability tracks, no training.

For each reference event at the operating point chosen for the dev FA budget (per fold):
  detected          matched by the strict event scorer (class-agnostic, tau=0.5, offset tol max(tau, .5 dur))
  boundary-only     a predicted event overlaps the reference but fails the collar/offset test  -> Olga would still see it
  merged            overlapping predicted event exists but is already matched to a neighbouring reference
  decoder           track peak >= detect threshold inside the event, yet no predicted event overlaps (median/min_dur loss)
  below-threshold   0.1 <= peak < detect threshold (recoverable by trading false alarms)
  absent            peak < 0.1: the track carries no evidence of the call
Also: test recall-vs-FAR curve (same decoder, detect threshold swept, edge = detect/2), interpolated at 50/100/300 FA/h,
per-class and per-duration breakdown of misses, and a CSV of examples for Olga.
Usage: python analyze_misses.py <name> [budget=100]
"""
import sys, json, itertools, numpy as np, csv
from pathlib import Path
from common import ROOT, DATA, SR, FOLDS, match, decode, ann_for, CLASSES
import common
from recall_far import events_from_pcall, stats, TAU

name = sys.argv[1]; budget = int(sys.argv[2]) if len(sys.argv) > 2 else 100
z = np.load(ROOT / 'results' / f'{name}_pcall.npz'); tok_s = float(z['tok_s']); common.TOK_S = tok_s
pc = {k[3:]: z[k].astype(np.float32) for k in z.files if k.startswith('p__')}
refs = {s: ann_for(s) for s in pc}
hours = {s: len(np.load(DATA / 'field16k' / f'{s}.npy')) / SR / 3600 for s in pc}
RF = json.load(open(ROOT / 'results' / 'recall_far.json'))[name][str(budget)]['thresholds']

def overlaps(p, r): return p[0] < r[1] and p[1] > r[0]

rows = []
for k in range(3):
    det, edge = RF[k]
    for s in FOLDS[k]:
        preds = events_from_pcall(pc[s], det, edge); R = [(a, b, 'call') for a, b, _ in refs[s]]
        # strict matching with bookkeeping of which pred matched which ref
        used = np.zeros(len(preds), bool); matched_ref = {}
        for ri, (r0, r1, _) in enumerate(R):
            best, bd = -1, 1e9
            for j, (p0, p1, _) in enumerate(preds):
                if used[j] or abs(p0 - r0) > TAU or abs(p1 - r1) > max(TAU, 0.5 * (r1 - r0)): continue
                if abs(p0 - r0) < bd: best, bd = j, abs(p0 - r0)
            if best >= 0: used[best] = True; matched_ref[ri] = best
        for ri, (r0, r1, cls) in enumerate(refs[s]):
            a, b = int(r0 / tok_s), max(int(r0 / tok_s) + 1, int(np.ceil(r1 / tok_s))); peak = float(pc[s][a:b].max()) if b <= len(pc[s]) else float(pc[s][a:].max())
            if ri in matched_ref: cat = 'detected'
            else:
                ov = [j for j, p in enumerate(preds) if overlaps(p, (r0, r1))]
                if ov and any(not used[j] for j in ov): cat = 'boundary-only'
                elif ov: cat = 'merged'
                elif peak >= det: cat = 'decoder'
                elif peak >= 0.1: cat = 'below-threshold'
                else: cat = 'absent'
            rows.append(dict(rec=s, fold=k, onset=round(r0, 2), offset=round(r1, 2), dur=round(r1 - r0, 2), cls=cls, peak=round(peak, 3), det=det, cat=cat))

cats = ['detected', 'boundary-only', 'merged', 'decoder', 'below-threshold', 'absent']
n = len(rows); cnt = {c: sum(r['cat'] == c for r in rows) for c in cats}
print(f'{name}: operating point for dev budget {budget} FA/h; {n} reference events')
print('  ' + '  '.join(f'{c} {cnt[c]} ({100*cnt[c]/n:.0f}%)' for c in cats))
miss = [r for r in rows if r['cat'] != 'detected']
print(f'  misses {len(miss)}; "visible to Olga" (detected + boundary-only + merged): {100*(cnt["detected"]+cnt["boundary-only"]+cnt["merged"])/n:.1f}%')
print('  misses by class: ' + ', '.join(f'{c} {sum(r["cls"]==c for r in miss)}/{sum(r["cls"]==c for r in rows)}' for c in CLASSES))
for lo, hi in [(0, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 2.0), (2.0, 99)]:
    sub = [r for r in rows if lo <= r['dur'] < hi]
    if sub: print(f'  dur {lo}-{hi}s: n={len(sub)} miss-rate {100*sum(r["cat"]!="detected" for r in sub)/len(sub):.0f}%  absent {sum(r["cat"]=="absent" for r in sub)}  below-thr {sum(r["cat"]=="below-threshold" for r in sub)}  boundary {sum(r["cat"]=="boundary-only" for r in sub)}')
print('  misses by category x class (top):')
for c in ['absent', 'below-threshold', 'boundary-only', 'decoder', 'merged']:
    sub = [r for r in miss if r['cat'] == c]
    if sub: print(f'    {c:16s} ' + ', '.join(f'{k} {v}' for k, v in sorted(((k, sum(r["cls"]==k for r in sub)) for k in CLASSES), key=lambda x: -x[1]) if v))

# test recall-vs-FAR curve: sweep detect over all recordings (no dev selection), edge = det/2
print('  test recall-FAR curve (detect sweep, edge=detect/2):')
curve = []
for det in [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    recs = [(events_from_pcall(pc[s], det, det / 2), [(a, b, 'call') for a, b, _ in refs[s]]) for s in pc]
    rec, far, prec, f1 = stats(recs, sum(hours.values())); curve.append((far, rec, det))
curve.sort()
for far, rec, det in curve: print(f'    det {det:.2f}: FAR {far:6.0f}/h  recall {rec:.3f}')
fars = np.array([c[0] for c in curve]); recs_ = np.array([c[1] for c in curve])
interp = {B: float(np.interp(B, fars, recs_)) if fars.min() <= B <= fars.max() else None for B in [50, 100, 300]}
print('  interpolated TEST recall at FAR 50/100/300:', {k: (round(v, 3) if v is not None else None) for k, v in interp.items()})

out = ROOT / 'results' / f'misses_{name}.json'
json.dump(dict(name=name, budget=budget, counts=cnt, n=n, curve=curve, interp=interp, rows=rows), open(out, 'w'), indent=1)
ex = sorted([r for r in miss if r['cat'] in ('absent', 'below-threshold')], key=lambda r: r['peak'])[:12] + sorted([r for r in miss if r['cat'] == 'boundary-only'], key=lambda r: -r['dur'])[:6]
with open(ROOT / 'results' / f'misses_{name}_examples.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(ex[0].keys())); w.writeheader(); w.writerows(ex)
print('  saved', out.name, 'and examples csv')
