"""H21: paired bootstrap over recordings (recording carries all seeds) for the three pretraining regimes with the instance
decoder, the frame decoder on the same runs, and Voxaboxen (stored fused boxes) under the same protocol.

Per recording and run we store, at the dev-selected operating point (budget B, fold-wise thresholds from decode_instances),
tp / npred / nref / tp_overlapping / n_overlapping, and a threshold sweep (main threshold swept, the fold's other decoder
parameters kept) giving tp / npred per threshold. A bootstrap sample resamples the 17 recordings; within a sample counts are
pooled over recordings and seeds, the recall-FAR curve is interpolated at matched FAR (100/h and Voxaboxen's own FAR).
Usage: python bootstrap_instances.py [budget=100] [B=2000]
"""
import sys, json, numpy as np
from pathlib import Path
from scipy.signal import find_peaks
from common import ROOT, DATA, SR, FOLDS, ann_for, NOISE, CLASSES
import common
from recall_far import events_from_pcall, stats, TAU

BUDGET = int(sys.argv[1]) if len(sys.argv) > 1 else 100; NB = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
REG = {r: [f'fte_{r}_seed{s}' for s in range(3)] for r in ['R1', 'R2', 'R3']}
RECS = [s for f in FOLDS for s in f]; FOLD = {s: k for k in range(len(FOLDS)) for s in FOLDS[k]}
refs = {s: ann_for(s) for s in RECS}; hours = np.array([len(np.load(DATA / 'field16k' / f'{s}.npy')) / SR / 3600 for s in RECS])
MIN_DUR = 0.15
TH_INST = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
TH_FRAME = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98, 0.99]

def ov_flags(R): return [any(j != i and a < R[j][1] and R[j][0] < b for j in range(len(R))) for i, (a, b, _) in enumerate(R)]
FLAGS = {s: ov_flags(refs[s]) for s in RECS}; NREF = np.array([len(refs[s]) for s in RECS]); NOV = np.array([sum(FLAGS[s]) for s in RECS])

def count(preds, s):
    """strict class-agnostic matching -> tp, tp on overlapping refs"""
    R = refs[s]; used = np.zeros(len(preds), bool); tp = 0; tpo = 0
    for ri, (r0, r1, _) in enumerate(R):
        best, bd = -1, 1e9
        for j, (p0, p1, _) in enumerate(preds):
            if used[j] or abs(p0 - r0) > TAU or abs(p1 - r1) > max(TAU, 0.5 * (r1 - r0)): continue
            if abs(p0 - r0) < bd: best, bd = j, abs(p0 - r0)
        if best >= 0: used[best] = True; tp += 1; tpo += FLAGS[s][ri]
    return tp, tpo

def inst_events(e, P, th_on, d_min, min_score, tok_s):
    pres, on, ld = e[:, 0], e[:, 1], e[:, 2]; pk, _ = find_peaks(on, height=th_on, distance=max(1, int(d_min / tok_s))); out = []
    for p in pk:
        a = p * tok_s; b = a + max(MIN_DUR, min(float(np.exp(ld[p])) * tok_s, 5.0)); i1 = min(len(pres), int(b / tok_s) + 1)
        if float(on[p] * pres[p:i1].mean()) < min_score: continue
        out.append((a, b, 'c'))
    return out

def run_counts(name):
    z = np.load(ROOT / 'results' / f'{name}_tracks.npz'); tok_s = float(z['tok_s']); common.TOK_S = tok_s
    P = {k[3:]: z[k].astype(np.float32) for k in z.files if k.startswith('P__')}; EV = {k[4:]: z[k].astype(np.float32) for k in z.files if k.startswith('on__')}
    dj = json.load(open(ROOT / 'results' / f'decode_instances_{name}_b{BUDGET}.json'))
    out = {}
    for var, TH in [('instance', TH_INST), ('frame', TH_FRAME)]:
        thr = dj[var]['thresholds']; op = np.zeros((len(RECS), 3)); sw_tp = np.zeros((len(TH), len(RECS))); sw_np = np.zeros((len(TH), len(RECS))); sw_tpo = np.zeros((len(TH), len(RECS)))
        for i, s in enumerate(RECS):
            prm = thr[FOLD[s]]
            dec = (lambda t: inst_events(EV[s], P[s], t, prm[1], prm[2], tok_s)) if var == 'instance' else (lambda t: events_from_pcall(1.0 - P[s][:, NOISE], t, prm[1]))
            ev = dec(prm[0]); tp, tpo = count(ev, s); op[i] = (tp, len(ev), tpo)
            for j, t in enumerate(TH):
                ev = dec(t); tp_, tpo_ = count(ev, s); sw_tp[j, i] = tp_; sw_tpo[j, i] = tpo_; sw_np[j, i] = len(ev)
        out[var] = dict(op=op, sw_tp=sw_tp, sw_np=sw_np, sw_tpo=sw_tpo)
    return out

def vox_counts():
    INF = Path('/mnt/c/Users/Iaroslav/CETACEANS/voxaboxen_whale/project/whale_aves/inference')
    def read(p):
        o = []
        for line in open(p, encoding='utf-8', errors='ignore').read().splitlines()[1:]:
            f = line.split('\t')
            if len(f) >= 4: o.append((float(f[0]), float(f[1]), f[2], float(f[3])))
        return o
    def iou(a, b):
        i = max(0, min(a[1], b[1]) - max(a[0], b[0])); u = (a[1] - a[0]) + (b[1] - b[0]) - i; return i / u if u > 0 else 0
    low = {s.lower(): s for s in RECS}; boxes = {}
    for p in INF.glob('peaks_pred_*-fwd.txt'):
        st = p.name[len('peaks_pred_'):-8]
        if st.lower() not in low: continue
        fwd = read(p); bck = read(INF / f'peaks_pred_{st}-bck.txt'); boxes[low[st.lower()]] = fwd + [b for b in bck if all(iou(b, f) < 0.5 for f in fwd)]
    assert set(boxes) == set(RECS), (set(RECS) - set(boxes))
    TH = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    ev = lambda s, t: [(a, b, 'c') for a, b, c, p in boxes[s] if p >= t]
    chosen = []
    for k in range(3):
        others = [f for i, f in enumerate(FOLDS) if i != k]; devs = [others[0][0], others[1][0]]; best, br = 0.95, -1
        for t in TH:
            rec, far, _, _ = stats([(ev(s, t), [(a, b, 'c') for a, b, _ in refs[s]]) for s in devs], sum(hours[RECS.index(s)] for s in devs))
            if far <= BUDGET and rec > br: br, best = rec, t
        chosen.append(best)
    op = np.zeros((len(RECS), 3)); sw_tp = np.zeros((len(TH), len(RECS))); sw_np = np.zeros((len(TH), len(RECS))); sw_tpo = np.zeros((len(TH), len(RECS)))
    for i, s in enumerate(RECS):
        e = ev(s, chosen[FOLD[s]]); tp, tpo = count(e, s); op[i] = (tp, len(e), tpo)
        for j, t in enumerate(TH): e = ev(s, t); tp_, tpo_ = count(e, s); sw_tp[j, i] = tp_; sw_tpo[j, i] = tpo_; sw_np[j, i] = len(e)
    return dict(op=op, sw_tp=sw_tp, sw_np=sw_np, sw_tpo=sw_tpo, thresholds=chosen)

def pooled(group, ix):
    """group: list of count dicts (seeds). Returns recall, FAR, recall_overlapping at op, and the curves (fars, recs, recs_overlap)."""
    n = len(group); nref = NREF[ix].sum() * n; nov = NOV[ix].sum() * n; h = hours[ix].sum() * n
    op = sum(g['op'][ix].sum(0) for g in group); tp = sum(g['sw_tp'][:, ix].sum(1) for g in group); npd = sum(g['sw_np'][:, ix].sum(1) for g in group)
    tpo = sum(g['sw_tpo'][:, ix].sum(1) for g in group) if all('sw_tpo' in g for g in group) else np.full(len(tp), np.nan)
    fars = (npd - tp) / h; recs = tp / nref; reco = tpo / max(1, nov); o = np.argsort(fars)
    return op[0] / nref, (op[1] - op[0]) / h, op[2] / max(1, nov), fars[o], recs[o], reco[o]

def at_far(fars, recs, F): return float(np.interp(F, fars, recs)) if fars.min() <= F <= fars.max() else np.nan

if __name__ == '__main__':
    C = {r: [run_counts(n) for n in names] for r, names in REG.items()}
    V = vox_counts()
    groups = {f'{r}/inst': [c['instance'] for c in C[r]] for r in REG} | {f'{r}/frame': [c['frame'] for c in C[r]] for r in REG} | {'VOX': [V]}
    for tag, g in [('voxft', 'VOXFT'), ('voxftb', 'VOXFTB')]:
        vf = ROOT / 'results' / f'{tag}_counts.npz'
        if vf.exists():
            z = np.load(vf); groups[g] = [dict(op=z['op'], sw_tp=z['sw_tp'], sw_np=z['sw_np'], **({'sw_tpo': z['sw_tpo']} if 'sw_tpo' in z.files else {}))]; print(f'field-fine-tuned Voxaboxen included ({g})')
    rng = np.random.default_rng(0); k = len(RECS); full = np.arange(k); IDX = [rng.integers(0, k, k) for _ in range(NB)]
    # Voxaboxen's own operating FAR (full data) as second matching point
    _, vopfar, _, vfs, vrs, _ = pooled([V], full); vfar = float(vfs.max()); FARS = {'100': 100.0, '150': 150.0, 'vox': vfar}
    print(f'Voxaboxen dev≤{BUDGET}/h operating FAR {vopfar:.1f}/h (thresholds {V["thresholds"]}); its maximal FAR (thr 0.5) {vfar:.1f}/h = matching point')
    S = {}
    for g, grp in groups.items():
        rec, far, rov, fs, rs, ro = pooled(grp, full)
        boots = []
        for ix in IDX:
            r_, f_, o_, fs_, rs_, ro_ = pooled(grp, ix); boots.append([r_, f_, o_, at_far(fs_, rs_, 100.0), at_far(fs_, rs_, 150.0), at_far(fs_, rs_, vfar), at_far(fs_, rs_, 300.0), at_far(fs_, ro_, 100.0), at_far(fs_, ro_, vfar)])
        boots = np.array(boots)
        S[g] = boots
        ci = lambda a: np.nanpercentile(a, [2.5, 97.5]); nn = lambda a: np.isnan(a).mean()
        print(f"{g:9s} op: R {rec:.3f} [{ci(boots[:,0])[0]:.3f},{ci(boots[:,0])[1]:.3f}] FAR {far:5.1f}  R_overlap {rov:.3f} [{ci(boots[:,2])[0]:.3f},{ci(boots[:,2])[1]:.3f}]"
              f"  | R@100 {at_far(fs, rs, 100):.3f} [{ci(boots[:,3])[0]:.3f},{ci(boots[:,3])[1]:.3f}] undef {nn(boots[:,3]):.2f}  R@150 {at_far(fs, rs, 150):.3f} [{ci(boots[:,4])[0]:.3f},{ci(boots[:,4])[1]:.3f}] undef {nn(boots[:,4]):.2f}  R@{vfar:.0f} {at_far(fs, rs, vfar):.3f} [{ci(boots[:,5])[0]:.3f},{ci(boots[:,5])[1]:.3f}] undef {nn(boots[:,5]):.2f}  R@300 {at_far(fs, rs, 300):.3f} [{ci(boots[:,6])[0]:.3f},{ci(boots[:,6])[1]:.3f}] undef {nn(boots[:,6]):.2f}  | OVERLAP R@100 {at_far(fs, ro, 100):.3f} [{ci(boots[:,7])[0]:.3f},{ci(boots[:,7])[1]:.3f}]  OVERLAP R@{vfar:.0f} {at_far(fs, ro, vfar):.3f} [{ci(boots[:,8])[0]:.3f},{ci(boots[:,8])[1]:.3f}]")
    res = {}
    pairs = [('R2/inst', 'R1/inst'), ('R3/inst', 'R1/inst'), ('R3/inst', 'R2/inst'), ('R2/inst', 'R2/frame'), ('R1/inst', 'R1/frame'), ('R3/inst', 'R3/frame'),
             ('R2/inst', 'VOX'), ('R3/inst', 'VOX'), ('R1/inst', 'VOX'), ('R2/frame', 'R1/frame'), ('R3/frame', 'R1/frame')]
    if 'VOXFT' in groups: pairs += [('R2/inst', 'VOXFT'), ('R3/inst', 'VOXFT'), ('R1/inst', 'VOXFT'), ('VOXFT', 'VOX')]
    if 'VOXFTB' in groups: pairs += [('R2/inst', 'VOXFTB'), ('R3/inst', 'VOXFTB'), ('R1/inst', 'VOXFTB'), ('VOXFTB', 'VOXFT')]
    print('\npaired differences (recording-level bootstrap, recording carries all seeds):')
    for a, b in pairs:
        d = S[a] - S[b]; row = {}
        for j, lab in enumerate(['op_recall', 'op_far', 'overlap_recall', 'R@100', 'R@150', f'R@{vfar:.0f}', 'R@300', 'OV@100', f'OV@{vfar:.0f}']):
            dj = d[:, j]; ok = ~np.isnan(dj)
            if ok.sum() < 100: row[lab] = (float('nan'),) * 4 + (int(ok.sum()),); continue
            dj = dj[ok]; m = dj.mean(); lo, hi = np.percentile(dj, [2.5, 97.5]); p = (dj > 0).mean(); row[lab] = (round(float(m), 3), round(float(lo), 3), round(float(hi), 3), round(float(p), 3), int(ok.sum()))
        fmt = lambda r: f"{r[0]:+.3f} [{r[1]:+.3f},{r[2]:+.3f}] P>0 {r[3]:.2f} n={r[4]}"
        print(f"  {a:9s} - {b:9s}  op ΔR {row['op_recall'][0]:+.3f} [{row['op_recall'][1]:+.3f},{row['op_recall'][2]:+.3f}] (ΔFAR {row['op_far'][0]:+.0f})  Δoverlap {row['overlap_recall'][0]:+.3f} [{row['overlap_recall'][1]:+.3f},{row['overlap_recall'][2]:+.3f}]")
        print(f"               ΔR@100 {fmt(row['R@100'])}   ΔR@150 {fmt(row['R@150'])}   ΔR@{vfar:.0f} {fmt(row[f'R@{vfar:.0f}'])}   ΔR@300 {fmt(row['R@300'])}")
        print(f"               Δoverlap@100 {fmt(row['OV@100'])}   Δoverlap@{vfar:.0f} {fmt(row[f'OV@{vfar:.0f}'])}")
        res[f'{a} - {b}'] = row
    json.dump(dict(budget=BUDGET, B=NB, vox_far=vfar, diffs=res, groups={g: dict(op_recall=round(float(np.mean(S[g][:,0])),3), R_at_100=round(float(np.nanmean(S[g][:,3])),3)) for g in groups}),
              open(ROOT / 'results' / f'bootstrap_instances_b{BUDGET}.json', 'w'), indent=1)
