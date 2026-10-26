"""Audit of the (D, tau50) decomposition on saved per-recording events.
 (1) plateau check: class-aware and class-agnostic F1/recall at tau = 1, 2, 5, 10 s
 (2) class-aware vs class-agnostic onset/offset errors
 (3) onset errors restricted to the COMMON set of reference events matched by all listed models (tau=1, class-aware)
"""
import json, sys, numpy as np
from common import ROOT, match, prf

def load(n): return json.load(open(ROOT / 'results' / f'{n}_events.json'))['recordings']

def matched_pairs(preds, refs, tau, ca):
    used = np.zeros(len(preds), bool); out = {}
    for ri, (r0, r1, rl) in enumerate(refs):
        best, bd = -1, 1e9
        for j, (p0, p1, pl) in enumerate(preds):
            if used[j] or (ca and pl != rl) or abs(p0 - r0) > tau or abs(p1 - r1) > max(tau, 0.5 * (r1 - r0)): continue
            if abs(p0 - r0) < bd: best, bd = j, abs(p0 - r0)
        if best >= 0: used[best] = True; out[ri] = (preds[best][0] - r0, preds[best][1] - r1)
    return out

names = sys.argv[1:] or ['mel', 'aves', 'kmaves', 'kmaves_modern', 'd2v', 'd2v_modern', 'ft_kmaves', 'ft_d2v_modern']
R = {n: load(n) for n in names}
print('(1) plateau check: F1 cls / recall cls / F1 any at large collars')
for n in names:
    row = []
    for tau in [1, 2, 5, 10]:
        tp = sum(match(p, r, tau, True) for p, r in R[n]); npred = sum(len(p) for p, _ in R[n]); nref = sum(len(r) for _, r in R[n])
        P, Rc, F = prf(tp, npred, nref); tpa = sum(match(p, r, tau, False) for p, r in R[n]); Fa = prf(tpa, npred, nref)[2]
        row.append(f'τ={tau}: {F:.3f}/{Rc:.3f}/{Fa:.3f}')
    print(f'  {n:16s} ' + '  '.join(row))
print('(2) onset|offset abs error p50/p90 (ms), class-aware vs class-agnostic matching at τ=1')
for n in names:
    for ca in (True, False):
        on, off = [], []
        for p, r in R[n]:
            m = matched_pairs(p, r, 1.0, ca); on += [abs(v[0]) for v in m.values()]; off += [abs(v[1]) for v in m.values()]
        q = lambda x: np.percentile(np.array(x) * 1000, [50, 90]).round(0) if x else None
        print(f'  {n:16s} {"cls" if ca else "any"}  n={len(on):3d}  onset {q(on)}  offset {q(off)}')
print('(3) onset error on the common matched set (class-aware, τ=1) across all listed models')
common_sets = None; per_model = {}
for n in names:
    ms = [matched_pairs(p, r, 1.0, True) for p, r in R[n]]; per_model[n] = ms
    keys = {(i, k) for i, m in enumerate(ms) for k in m}
    common_sets = keys if common_sets is None else common_sets & keys
print(f'  common matched events: {len(common_sets)} of {sum(len(r) for _, r in R[names[0]])}')
for n in names:
    on = [abs(per_model[n][i][k][0]) for (i, k) in common_sets]
    print(f'  {n:16s} onset p50/p75/p90 ms: {np.percentile(np.array(on) * 1000, [50, 75, 90]).round(0)}')
