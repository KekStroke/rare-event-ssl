"""mAP at IoU 0.5 / 0.8 in Voxaboxen's definition, for our OZF instance detections on the official TEST split, in REAL time (slowed times / 2).
Definition copied from voxaboxen/evaluation/evaluation.py::mean_average_precision: for each detection threshold, maximum bipartite matching between
references and predictions over pairs with IoU >= iou (we use scipy's maximum_bipartite_matching; same cardinality as Hopcroft-Karp), precision and
recall; then AP = mean over 1001 recall levels of the max precision at recall >= level, with (recall 1, precision 0) and (recall 0, precision 1) added.
Detection threshold sweep: our instance score (onset prob x mean presence) over 200 levels; secondary decoder params = the val-chosen ones.
Usage: TSSSL_DATASET=ozf python map_eval.py <run_name> [budget=400]   (pooled over the run's boxes; per-seed numbers reported by calling per run)"""
import sys, os, json, numpy as np
assert os.environ.get('TSSSL_DATASET') == 'ozf'
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.signal import find_peaks
from common import ROOT, DATA, SR, FOLDS, ann_for, NOISE
import common
name = sys.argv[1]; BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 400; SLOW = 2.0; MIN_DUR = 0.15
z = np.load(ROOT / 'results' / f'{name}_tracks.npz'); tok_s = float(z['tok_s']); common.TOK_S = tok_s
P = {k[3:]: z[k].astype(np.float32) for k in z.files if k.startswith('P__')}; EV = {k[4:]: z[k].astype(np.float32) for k in z.files if k.startswith('on__')}
thr = json.load(open(ROOT / 'results' / f'decode_instances_{name}_b{BUDGET}.json'))['instance']['thresholds'][0]
test = FOLDS[0]

def boxes(s, th_on=0.0, d_min=thr[1]):
    e = EV[s]; pres, on, ld = e[:, 0], e[:, 1], e[:, 2]; pk, _ = find_peaks(on, height=max(th_on, 1e-4), distance=max(1, int(d_min / tok_s))); out = []
    for p in pk:
        a = p * tok_s; b = a + max(MIN_DUR, min(float(np.exp(ld[p])) * tok_s, 5.0)); i1 = min(len(pres), int(b / tok_s) + 1)
        out.append((a / SLOW, b / SLOW, float(on[p] * pres[p:i1].mean())))
    return out

def match_count(R, Pb, iou_min):
    if not R or not Pb: return 0
    ra = np.array([r[0] for r in R]); rb = np.array([r[1] for r in R]); pa = np.array([p[0] for p in Pb]); pb_ = np.array([p[1] for p in Pb])
    inter = np.maximum(0, np.minimum(rb[:, None], pb_[None]) - np.maximum(ra[:, None], pa[None])); union = (rb - ra)[:, None] + (pb_ - pa)[None] - inter
    G = csr_matrix((inter / np.maximum(union, 1e-9)) >= iou_min)
    m = maximum_bipartite_matching(G, perm_type='column'); return int((m >= 0).sum())

REF = {s: [(a / SLOW, b / SLOW) for a, b, _ in ann_for(s)] for s in test}
ALL = {s: boxes(s) for s in test}; nref = sum(len(v) for v in REF.values())
levels = np.linspace(0, 1, 201); out = {}
for iou in [0.5, 0.8]:
    precs, recs = [], []
    for t in levels:
        tp = 0; npred = 0
        for s in test:
            Pb = [b for b in ALL[s] if b[2] >= t]; npred += len(Pb); tp += match_count(REF[s], Pb, iou)
        if npred + nref == 0: continue
        precs.append(tp / max(npred, 1)); recs.append(tp / nref)
    pr = [0.0] + precs + [1.0]; rc = [1.0] + recs + [0.0]
    best = {}
    for r, p in zip(rc, pr): best[r] = max(p, best.get(r, 0))
    rn = np.array(list(best.keys())); pn = np.array(list(best.values()))
    ap = float(np.mean([pn[rn >= lv].max() for lv in np.linspace(0, 1, 1001)]))
    out[f'mAP@{iou}'] = round(ap, 4); print(f'{name}: mAP@{iou} IoU = {ap*100:.2f}  (test refs {nref}, 13 recordings, real time)')
json.dump(out, open(ROOT / 'results' / f'map_{name}.json', 'w'), indent=1)
