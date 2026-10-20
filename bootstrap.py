"""Paired bootstrap over recordings: CI of F1(tau) per model and of the difference between two models.
Uses results/<name>_events.json (per-recording predicted/reference events, pooled over the 3 test folds)."""
import json, sys, numpy as np
from common import ROOT, match, prf, TAUS
def load(n):
    j = json.load(open(ROOT / 'results' / f'{n}_events.json')); return [(p, r) for p, r in j['recordings']]
def f1(recs, tau, ca=True):
    tp = sum(match(p, r, tau, ca) for p, r in recs); return prf(tp, sum(len(p) for p, _ in recs), sum(len(r) for _, r in recs))[2]
def boot(names, tau=0.2, B=2000, seed=0):
    R = {n: load(n) for n in names}; k = len(next(iter(R.values()))); rng = np.random.default_rng(seed)
    idx = [rng.integers(0, k, k) for _ in range(B)]
    out = {n: np.array([f1([R[n][i] for i in ix], tau) for ix in idx]) for n in names}
    return out
if __name__ == '__main__':
    names = sys.argv[1:]; res = {}
    for tau in [0.05, 0.2, 1.0]:
        out = boot(names, tau)
        for n in names:
            lo, hi = np.percentile(out[n], [2.5, 97.5]); res[f'{n}@{tau}'] = (round(float(out[n].mean()), 3), round(float(lo), 3), round(float(hi), 3))
            print(f'tau={tau} {n:22s} F1 {out[n].mean():.3f}  95% CI [{lo:.3f}, {hi:.3f}]')
        if len(names) == 2:
            d = out[names[0]] - out[names[1]]; lo, hi = np.percentile(d, [2.5, 97.5])
            print(f'tau={tau} diff {names[0]} - {names[1]}: {d.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  P(diff>0)={np.mean(d > 0):.3f}')
            res[f'diff@{tau}'] = (round(float(d.mean()), 3), round(float(lo), 3), round(float(hi), 3), round(float(np.mean(d > 0)), 3))
    json.dump(res, open(ROOT / 'results' / f'bootstrap_{"_vs_".join(names)}.json', 'w'), indent=1)
