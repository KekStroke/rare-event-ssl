"""Main criterion for the detector: class-agnostic recall at fixed false-alarm budgets (FA per hour).

For each test fold: thresholds (detect, edge) are chosen on the two dev recordings as the pair with the highest
dev recall subject to dev FAR <= budget; test recall / FAR / F1 are reported at that pair, pooled over folds.
Matching: class-agnostic, onset collar tau (default 0.5 s), offset tolerance max(tau, 0.5*dur).
A false alarm is a predicted event not matched to any reference event.
Usage: python recall_far.py <name> [<name> ...]   (needs results/<name>_pcall.npz)
"""
import sys, json, itertools, numpy as np
from common import ROOT, DATA, SR, FOLDS, match, decode, ann_for, NOISE
import common

BUDGETS = [50, 100, 300]          # false alarms per hour; call rate on these recordings is ~500/h
TAU = common.TAU_MATCH          # 0.5 s whale, 0.1 s OZF (slowed)
DET = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]; EDGE = [0.05, 0.1, 0.2, 0.3, 0.5]

def events_from_pcall(pc, det, edge):
    probs = np.stack([pc, 1 - pc], 1).astype(np.float32)          # fake 2-class track: [call, noise]
    saved = (common.NOISE, common.CLASSES)
    common.NOISE = 1; common.CLASSES = ['call']
    try: ev = decode(probs, det, edge, med=5, min_dur=0.15)
    finally: common.NOISE, common.CLASSES = saved                  # never leak the override into ann_for()/read_raven()
    return ev

def stats(recs, hours):
    tp = sum(match(p, r, TAU, False) for p, r in recs); npred = sum(len(p) for p, _ in recs); nref = sum(len(r) for _, r in recs)
    rec = tp / max(nref, 1); far = (npred - tp) / hours; prec = tp / max(npred, 1)
    return rec, far, prec, (2 * prec * rec / (prec + rec) if prec + rec else 0.0)

def evaluate(name):
    z = np.load(ROOT / 'results' / f'{name}_pcall.npz'); tok_s = float(z['tok_s'])
    pc = {k[3:]: z[k].astype(np.float32) for k in z.files if k.startswith('p__')}
    refs = {s: [(a, b, 'call') for a, b, _ in ann_for(s)] for s in pc}
    hours = {s: len(np.load(DATA / 'field16k' / f'{s}.npy')) / SR / 3600 for s in pc}
    common.TOK_S = tok_s
    out = {}
    for B in BUDGETS:
        pooled = []; chosen = []
        for k in range(3):
            test = FOLDS[k]; others = [f for i, f in enumerate(FOLDS) if i != k]; devs = [others[0][0], others[1][0]]
            best, br = None, -1
            for det, edge in itertools.product(DET, EDGE):
                recs = [(events_from_pcall(pc[s], det, edge), refs[s]) for s in devs]
                rec, far, _, _ = stats(recs, sum(hours[s] for s in devs))
                if far <= B and rec > br: br, best = rec, (det, edge)
            if best is None: best = (0.9, 0.5)
            chosen.append(best); pooled += [(events_from_pcall(pc[s], *best), refs[s]) for s in test]
        rec, far, prec, f1 = stats(pooled, sum(hours[s] for f in FOLDS for s in f))
        out[B] = dict(recall=round(rec, 3), far_per_h=round(far, 1), precision=round(prec, 3), f1_any=round(f1, 3), thresholds=chosen)
    # per-class recall at the 300/h point (class-agnostic matching, grouped by reference class)
    B = 300; pooled_cls = []
    for k in range(3):
        test = FOLDS[k]; det, edge = out[B]['thresholds'][k]
        pooled_cls += [(events_from_pcall(pc[s], det, edge), ann_for(s)) for s in test]
    per_class = {}
    for c in ['K1', 'K4', 'K5', 'K7', 'K10', 'K12', 'K13', 'K14', 'K17', 'K21', 'K27']:
        sub = [(p, [e for e in r if e[2] == c]) for p, r in pooled_cls]; n = sum(len(r) for _, r in sub)
        if n: per_class[c] = round(sum(match(p, [(a, b, 'call') for a, b, _ in r], TAU, False) for p, r in sub) / n, 3)
    out['per_class_recall_at_300'] = per_class
    return out

if __name__ == '__main__':
    res = {}
    for n in sys.argv[1:]:
        try: res[n] = evaluate(n)
        except FileNotFoundError: print(n, 'no pcall file (re-run probe/finetune)'); continue
        r = res[n]; print(f"{n:24s} " + '  '.join(f"FA≤{B}/h: R {r[B]['recall']:.3f} (FAR {r[B]['far_per_h']:.0f}, P {r[B]['precision']:.2f})" for B in BUDGETS)
              + f"  | K12@300 {r['per_class_recall_at_300'].get('K12')}  K21 {r['per_class_recall_at_300'].get('K21')}")
    json.dump(res, open(ROOT / 'results' / 'recall_far.json', 'w'), indent=1)
