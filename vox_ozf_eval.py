"""Voxaboxen trained on OZF (run_vox_ozf.sh) scored under our OZF protocol. Reads the staff decoder's combined boxes (detection threshold 0.05,
comb discard 0) for test + val recordings, converts real seconds to the pipeline's slowed seconds (x2), chooses the Detection Prob threshold on val
under the FA budget, reports test recall / FAR / overlapping-subset recall at that operating point and recall at matched global FAR; also mAP@0.5/0.8
in Voxaboxen's definition (real time) from the same boxes. Saves results/voxozf_counts.npz for the paired bootstrap.
Usage: TSSSL_DATASET=ozf python vox_ozf_eval.py [budget=400]"""
import sys, os, json, numpy as np, pandas as pd
assert os.environ.get('TSSSL_DATASET') == 'ozf'
from common import ROOT, DATA, SR, FOLDS, ann_for, splits
from recall_far import TAU
BUDGET = int(sys.argv[1]) if len(sys.argv) > 1 else 400; SLOW = 2.0
INF = ROOT / 'vox_ozf' / 'project' / 'vox_ozf' / 'inference'
train, devs, test = splits()[0]
def boxes(s):
    d = pd.read_csv(INF / f'peaks_pred_{s}-detthresh0.05-comb.txt', sep='\t')
    return [(float(a) * SLOW, float(b) * SLOW, float(p)) for a, b, p in zip(d['Begin Time (s)'], d['End Time (s)'], d['Detection Prob'])]
B = {s: boxes(s) for s in devs + test}; refs = {s: [(a, b) for a, b, _ in ann_for(s)] for s in devs + test}
hours = {s: len(np.load(DATA / 'field16k' / f'{s}.npy')) / SR / 3600 for s in devs + test}
flags = {s: [any(j != i and a < R[j][1] and R[j][0] < b for j in range(len(R))) for i, (a, b) in enumerate(R)] for s, R in refs.items()}
def count(preds, s):
    R = refs[s]; used = np.zeros(len(preds), bool); tp = 0; tpo = 0
    for ri, (r0, r1) in enumerate(R):
        best, bd = -1, 1e9
        for j, (p0, p1) in enumerate(preds):
            if used[j] or abs(p0 - r0) > TAU or abs(p1 - r1) > max(TAU, 0.5 * (r1 - r0)): continue
            if abs(p0 - r0) < bd: best, bd = j, abs(p0 - r0)
        if best >= 0: used[best] = True; tp += 1; tpo += flags[s][ri]
    return tp, tpo
TH = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
ev = lambda s, t: [(a, b) for a, b, p in B[s] if p >= t]
best, br = TH[-1], -1; dev_h = sum(hours[s] for s in devs); ndev = sum(len(refs[s]) for s in devs)
for t in TH:
    tp = 0; npred = 0
    for s in devs: a, _ = count(ev(s, t), s); tp += a; npred += len(ev(s, t))
    far = (npred - tp) / dev_h
    if far <= BUDGET and tp / ndev > br: br, best = tp / ndev, t
op = np.zeros((len(test), 3)); sw_tp = np.zeros((len(TH), len(test))); sw_np = np.zeros((len(TH), len(test))); sw_tpo = np.zeros((len(TH), len(test)))
for i, s in enumerate(test):
    e = ev(s, best); tp, tpo = count(e, s); op[i] = (tp, len(e), tpo)
    for j, t in enumerate(TH): e = ev(s, t); a, b = count(e, s); sw_tp[j, i] = a; sw_tpo[j, i] = b; sw_np[j, i] = len(e)
nref = sum(len(refs[s]) for s in test); nov = sum(sum(flags[s]) for s in test); h = sum(hours[s] for s in test)
rec = op[:, 0].sum() / nref; far = (op[:, 1].sum() - op[:, 0].sum()) / h; rov = op[:, 2].sum() / nov
fars = (sw_np.sum(1) - sw_tp.sum(1)) / h; o = np.argsort(fars); R = (sw_tp.sum(1) / nref)[o]; RO = (sw_tpo.sum(1) / nov)[o]; F = fars[o]
atf = lambda arr, x: float(np.interp(x, F, arr)) if F.min() <= x <= F.max() else float('nan')
print(f'Voxaboxen on OZF (staff recipe, 1 model): val-chosen Detection Prob threshold {best} under {BUDGET} FA/slowed h -> test recall {rec:.3f} at FAR {far:.0f}/slowed h, overlapping-subset {rov:.3f} (n={nov}); curve max FAR {F.max():.0f}')
print('  matched FAR 200/400/800: R', [round(atf(R, x), 3) for x in (200, 400, 800)], 'overlap R', [round(atf(RO, x), 3) for x in (200, 400, 800)])
np.savez(ROOT / 'results' / 'voxozf_counts.npz', op=op, sw_tp=sw_tp, sw_np=sw_np, sw_tpo=sw_tpo, TH=np.array(TH), thresholds=np.array([best]))
# mAP (Voxaboxen definition) from the same boxes, real time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
def match_count(Rr, Pb, iou_min):
    if not Rr or not Pb: return 0
    ra = np.array([r[0] for r in Rr]); rb = np.array([r[1] for r in Rr]); pa = np.array([p[0] for p in Pb]); pb_ = np.array([p[1] for p in Pb])
    inter = np.maximum(0, np.minimum(rb[:, None], pb_[None]) - np.maximum(ra[:, None], pa[None])); union = (rb - ra)[:, None] + (pb_ - pa)[None] - inter
    m = maximum_bipartite_matching(csr_matrix((inter / np.maximum(union, 1e-9)) >= iou_min), perm_type='column'); return int((m >= 0).sum())
out = {}
for iou in [0.5, 0.8]:
    precs, recs = [], []
    for t in np.linspace(0, 1, 201):
        tp = 0; npred = 0
        for s in test:
            Pb = [(a / SLOW, b / SLOW) for a, b, p in B[s] if p >= t]; npred += len(Pb); tp += match_count([(a / SLOW, b / SLOW) for a, b in refs[s]], Pb, iou)
        precs.append(tp / max(npred, 1)); recs.append(tp / nref)
    pr = [0.0] + precs + [1.0]; rc = [1.0] + recs + [0.0]; bestp = {}
    for r, p in zip(rc, pr): bestp[r] = max(p, bestp.get(r, 0))
    rn = np.array(list(bestp.keys())); pn = np.array(list(bestp.values())); ap = float(np.mean([pn[rn >= lv].max() for lv in np.linspace(0, 1, 1001)]))
    out[f'mAP@{iou}'] = round(ap, 4); print(f'  mAP@{iou} IoU (our computation, same test split, real time) = {ap*100:.2f}')
json.dump(dict(budget=BUDGET, threshold=best, recall=round(rec, 3), far=round(far, 1), overlap_recall=round(rov, 3), n_overlap=int(nov), R_at=[round(atf(R, x), 3) for x in (200, 400, 800)], OV_at=[round(atf(RO, x), 3) for x in (200, 400, 800)], **out), open(ROOT / 'results' / f'voxozf_b{BUDGET}.json', 'w'), indent=1)
