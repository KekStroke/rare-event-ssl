"""Field-fine-tuned Voxaboxen under our protocol. Reads the staff decoder's combined boxes (inference at detection threshold
0.05, comb_discard 0) per fold: test recordings + the two dev recordings. Operating point: Detection Prob threshold with the
highest dev recall s.t. dev FAR <= budget (per fold); reports pooled test strict recall, FAR, overlapping-subset recall,
coverage>=50%, and the FAR-matched curve. Also writes results/voxft_counts.npz for the paired bootstrap (bootstrap_instances.py).
Usage: python vox_field_eval.py [budget=100] [inference_dir_glob]
"""
import sys, json, glob, numpy as np, pandas as pd
from pathlib import Path
from common import ROOT, DATA, SR, FOLDS, ann_for
from recall_far import stats, TAU
BUDGET = int(sys.argv[1]) if len(sys.argv) > 1 else 100
import os; KF = [int(x) for x in os.environ.get('VOX_FOLDS', '0,1,2').split(',')]   # restrict folds for a dry run only
EXP = os.environ.get('VOX_EXP', 'vox_ft_fold'); TAG = os.environ.get('VOX_TAG', 'voxft')   # vox_ft_fold = staff final-model (last epoch); vox_ftb_fold = best-val checkpoint
RECS = [s for f in FOLDS for s in f]; FOLD = {s: k for k in range(3) for s in FOLDS[k]}
refs = {s: ann_for(s) for s in RECS}; hours = {s: len(np.load(DATA / 'field16k' / f'{s}.npy')) / SR / 3600 for s in RECS}
TH = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]
def ov_flags(R): return [any(j != i and a < R[j][1] and R[j][0] < b for j in range(len(R))) for i, (a, b, _) in enumerate(R)]
def count(preds, s):
    R = refs[s]; fl = ov_flags(R); used = np.zeros(len(preds), bool); tp = 0; tpo = 0; vis = 0
    for ri, (r0, r1, _) in enumerate(R):
        best, bd = -1, 1e9
        for j, (p0, p1, _) in enumerate(preds):
            if used[j] or abs(p0 - r0) > TAU or abs(p1 - r1) > max(TAU, 0.5 * (r1 - r0)): continue
            if abs(p0 - r0) < bd: best, bd = j, abs(p0 - r0)
        if best >= 0: used[best] = True; tp += 1; tpo += fl[ri]
        if sum(max(0, min(p[1], r1) - max(p[0], r0)) for p in preds) / max(1e-6, r1 - r0) >= 0.5: vis += 1
    return tp, tpo, vis, sum(fl)
def load_boxes(k, s):
    pat = str(ROOT / 'vox_field' / f'fold{k}' / 'project' / f'{EXP}{k}' / 'inference' / f'peaks_pred_{s}-detthresh0.05-comb.txt')
    d = pd.read_csv(pat, sep='\t'); return [(float(a), float(b), float(p)) for a, b, p in zip(d['Begin Time (s)'], d['End Time (s)'], d['Detection Prob'])]
ev = lambda boxes, t: [(a, b, 'c') for a, b, p in boxes if p > t]
boxes = {}
for k in KF:
    others = [f for i, f in enumerate(FOLDS) if i != k]; devs = [others[0][0], others[1][0]]
    for s in FOLDS[k] + devs: boxes[(k, s)] = load_boxes(k, s)
pooled = []; chosen = []; op = np.zeros((len(RECS), 3)); sw_tp = np.zeros((len(TH), len(RECS))); sw_np = np.zeros((len(TH), len(RECS))); sw_tpo = np.zeros((len(TH), len(RECS))); vis_tot = 0; nov = 0; tpo_tot = 0
for k in KF:
    others = [f for i, f in enumerate(FOLDS) if i != k]; devs = [others[0][0], others[1][0]]; best, br = TH[-1], -1
    for t in TH:
        rec, far, _, _ = stats([(ev(boxes[(k, s)], t), [(a, b, 'c') for a, b, _ in refs[s]]) for s in devs], sum(hours[s] for s in devs))
        if far <= BUDGET and rec > br: br, best = rec, t
    chosen.append(best)
    for s in FOLDS[k]:
        i = RECS.index(s); e = ev(boxes[(k, s)], best); tp, tpo, vis, n_ov = count(e, s); op[i] = (tp, len(e), tpo); vis_tot += vis; nov += n_ov; tpo_tot += tpo
        pooled.append((e, [(a, b, 'c') for a, b, _ in refs[s]]))
        for j, t in enumerate(TH): e2 = ev(boxes[(k, s)], t); c2 = count(e2, s); sw_tp[j, i] = c2[0]; sw_tpo[j, i] = c2[1]; sw_np[j, i] = len(e2)
TR = [s for k in KF for s in FOLDS[k]]; rec, far, prec, f1 = stats(pooled, sum(hours[s] for s in TR)); nref = sum(len(refs[s]) for s in TR); H = sum(hours[s] for s in TR)
fars = (sw_np.sum(1) - sw_tp.sum(1)) / H; recs = sw_tp.sum(1) / nref; o = np.argsort(fars)
interp = {B: (round(float(np.interp(B, fars[o], recs[o])), 3) if fars.min() <= B <= fars.max() else None) for B in [50, 61, 100, 150, 300]}
print(f"Voxaboxen field-FT [{TAG}]  dev≤{BUDGET}/h thresholds {chosen} → test FAR {far:5.0f}  strict R {rec:.3f}  overlapping-subset R {tpo_tot / max(1, nov):.3f} (n={nov})  cov≥50% {vis_tot / nref:.3f}  | FAR-matched R@50/61/100/150/300 {interp}")
print('  curve:', ' '.join(f'{f:.0f}/{r:.3f}' for f, r in zip(fars[o], recs[o])))
if len(KF) == 3: np.savez(ROOT / 'results' / f'{TAG}_counts.npz', op=op, sw_tp=sw_tp, sw_np=sw_np, sw_tpo=sw_tpo, TH=np.array(TH), thresholds=np.array(chosen))
if len(KF) == 3: json.dump(dict(budget=BUDGET, thresholds=chosen, recall=round(rec, 3), test_far=round(far, 1), recall_overlapping=round(tpo_tot / max(1, nov), 3), n_overlapping=nov, visible50=round(vis_tot / nref, 3), far_matched=interp,
               curve=[(float(f), float(r)) for f, r in zip(fars[o], recs[o])]), open(ROOT / 'results' / f'{TAG}_b{BUDGET}.json', 'w'), indent=1)
