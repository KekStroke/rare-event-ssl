"""Voxaboxen (BEATs backbone, synthetic-trained, never saw field audio) under OUR protocol, from its stored per-recording
peak predictions (fwd/bck box sets, Detection Prob >= 0.5 already applied by the tool). Fusion approximated by: all fwd boxes
+ bck boxes whose IoU with every fwd box < 0.5. Threshold sweep only possible upward from 0.5.
Outputs: dev-budget operating points (recall, test FAR), recall-FAR curve, coverage>=50%, boundary-only/merged counts, sed_eval F1."""
import json, numpy as np, itertools
from pathlib import Path
from common import ROOT, DATA, SR, FOLDS, ann_for, match, CLASSES
from recall_far import stats, TAU
INF = Path('/mnt/c/Users/Iaroslav/CETACEANS/voxaboxen_whale/project/whale_aves/inference')
def read(p):
    out = []
    for line in open(p, encoding='utf-8', errors='ignore').read().splitlines()[1:]:
        f = line.split('\t')
        if len(f) >= 4: out.append((float(f[0]), float(f[1]), f[2], float(f[3])))
    return out
def iou(a, b):
    i = max(0, min(a[1], b[1]) - max(a[0], b[0])); u = (a[1] - a[0]) + (b[1] - b[0]) - i; return i / u if u > 0 else 0
stems = sorted({p.name[len('peaks_pred_'):-8] for p in INF.glob('peaks_pred_*-fwd.txt')})
boxes = {}
for st in stems:
    fwd = read(INF / f'peaks_pred_{st}-fwd.txt'); bck = read(INF / f'peaks_pred_{st}-bck.txt')
    fused = list(fwd) + [b for b in bck if all(iou(b, f) < 0.5 for f in fwd)]
    boxes[st] = {'fwd': fwd, 'bck': bck, 'fused': fused}
# map stems to annotation stems (case-insensitive)
ann_stems = {s.lower(): s for s in [p.stem for p in (DATA / 'field16k').glob('*.npy')]}
hours = {st: len(np.load(DATA / 'field16k' / f'{ann_stems[st.lower()]}.npy')) / SR / 3600 for st in stems}
refs = {st: ann_for(ann_stems[st.lower()]) for st in stems}
fold_of = {s.lower(): k for k in range(3) for s in FOLDS[k]}
def events(st, mode, thr): return [(a, b, c) for a, b, c, p in boxes[st][mode] if p >= thr]
def overlaps(p, r): return p[0] < r[1] and p[1] > r[0]
def analyse(recs):
    nref = sum(len(r) for _, r in recs); tp = 0; vis = 0; bonly = 0; merged = 0
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
            if sum(max(0, min(p[1], r[1]) - max(p[0], r[0])) for p in preds) / max(1e-6, r[1] - r[0]) >= 0.5: vis += 1
            if ri in matched: continue
            ov = [j for j, p in enumerate(preds) if overlaps(p, r)]
            if ov and any(not used[j] for j in ov): bonly += 1
            elif ov: merged += 1
    return tp / nref, vis / nref, bonly, merged
THR = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
for mode in ['fused', 'fwd', 'bck']:
    print(f'== Voxaboxen {mode}')
    print('  curve: ' + '  '.join(f'thr {t:.2f}: FAR {stats([(events(s, mode, t), [(a,b,"c") for a,b,_ in refs[s]]) for s in stems], sum(hours.values()))[1]:.0f}/h R {stats([(events(s, mode, t), [(a,b,"c") for a,b,_ in refs[s]]) for s in stems], sum(hours.values()))[0]:.3f}' for t in THR[::3]))
    for B in [50, 100, 300]:
        pooled = []
        for k in range(3):
            test = [s for s in stems if fold_of[s.lower()] == k]; others = [f for i, f in enumerate(FOLDS) if i != k]; devs = [others[0][0].lower(), others[1][0].lower()]
            devs = [s for s in stems if s.lower() in devs]
            best, br = None, -1
            for t in THR:
                rec, far, _, _ = stats([(events(s, mode, t), [(a, b, 'c') for a, b, _ in refs[s]]) for s in devs], sum(hours[s] for s in devs))
                if far <= B and rec > br: br, best = rec, t
            if best is None: best = 0.95
            pooled += [(events(s, mode, best), [(a, b, 'c') for a, b, _ in refs[s]]) for s in test]
        rec, far, prec, f1 = stats(pooled, sum(hours.values())); r_, vis, bo, mg = analyse(pooled)
        print(f'  dev≤{B}/h → test FAR {far:.0f}  strict R {rec:.3f}  coverage≥50% {vis:.3f}  boundary-only {bo}  merged {mg}')
# sed_eval class-aware at thr 0.5 fused (sanity vs the 65.5 reported earlier)
try:
    import sed_eval, dcase_util
    m = sed_eval.sound_event.EventBasedMetrics(event_label_list=CLASSES, t_collar=0.2, percentage_of_length=0.5)
    for st in stems:
        to = lambda evs: dcase_util.containers.MetaDataContainer([dict(event_onset=a, event_offset=b, event_label=c, filename=st) for a, b, c in evs])
        m.evaluate(reference_event_list=to(refs[st]), estimated_event_list=to(events(st, 'fused', 0.5)))
    r = m.results(); print(f"sed_eval fused@0.5: overall F1 {100*r['overall']['f_measure']['f_measure']:.1f}  class-wise {100*r['class_wise_average']['f_measure']['f_measure']:.1f}  (reported in CETACEANS: 65.5 standalone)")
except Exception as e: print('sed_eval failed', e)
