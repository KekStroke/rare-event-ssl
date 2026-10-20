"""H26 secondary analysis: saved OZF fine-tuned models (ckpt/ft_fte_ozf_{arm}_seed{s}_split0.pt) run by inference only on the TEST split of the
six OZF Synthetic sets (overlap ratio 0 .. 1). Decoder thresholds = those chosen on the real val set (decode_instances json), frozen.
Per set and arm (pooled over 3 seeds): strict recall, FAR per slowed hour, overlapping-subset recall at the frozen operating point, and
recall / overlap recall at matched global FAR (th_on sweep, fold-0 secondary params). Usage: TSSSL_DATASET=ozf python eval_synth_tests.py [budget=400] [far=400]"""
import sys, os, json, numpy as np, torch, torch.nn as nn
assert os.environ.get('TSSSL_DATASET') == 'ozf'
from common import *
from recall_far import events_from_pcall, stats, TAU
from bootstrap_instances import inst_events, TH_INST
BUDGET = int(sys.argv[1]) if len(sys.argv) > 1 else 400; FAR_M = float(sys.argv[2]) if len(sys.argv) > 2 else 400.0
dev = 'cuda'; stats_ = [s.to(dev) for s in load_stats()]; MIN_DUR = 0.15

class Model(nn.Module):
    def __init__(self, block):
        super().__init__(); self.enc = build_encoder(block); self.head = nn.Linear(self.enc.d, NOISE + 1); self.ev = nn.Linear(self.enc.d, 3)
    def forward(self, wave):
        tok = tokens_from_mel(melspec(wave), stats_); h = self.enc(tok); return self.head(h), self.ev(h)

@torch.no_grad()
def predict(m, w, chunk_s=10.0):
    n = int(chunk_s * SR); P, E = [], []
    for s0 in range(0, len(w), n):
        c = w[s0:s0 + n]
        if len(c) < SR // 2: break
        with torch.autocast('cuda', dtype=torch.bfloat16): o = m(torch.tensor(c)[None].to(dev))
        P.append(torch.softmax(o[0].float(), -1)[0].cpu().numpy()); e = o[1].float()[0]
        E.append(np.stack([torch.sigmoid(e[:, 0]).cpu().numpy(), torch.sigmoid(e[:, 1]).cpu().numpy(), e[:, 2].cpu().numpy()], 1))
    return np.concatenate(P), np.concatenate(E)

def count(preds, R):
    used = np.zeros(len(preds), bool); fl = [any(j != i and a < R[j][1] and R[j][0] < b for j in range(len(R))) for i, (a, b) in enumerate(R)]; tp = 0; tpo = 0
    for ri, (r0, r1) in enumerate(R):
        best, bd = -1, 1e9
        for j, (p0, p1, _) in enumerate(preds):
            if used[j] or abs(p0 - r0) > TAU or abs(p1 - r1) > max(TAU, 0.5 * (r1 - r0)): continue
            if abs(p0 - r0) < bd: best, bd = j, abs(p0 - r0)
        if best >= 0: used[best] = True; tp += 1; tpo += fl[ri]
    return tp, tpo, sum(fl)

def read_ann(p):
    out = []
    for line in open(p).read().splitlines()[1:]:
        f = line.split('\t'); out.append((float(f[1]), float(f[2])))
    return out

SETS = ['0', '0.2', '0.4', '0.6', '0.8', '1']; out = {}
for arm in ['ov', 'noov']:
    models = []
    for s in range(3):
        ck = torch.load(ROOT / 'ckpt' / f'ft_fte_ozf_{arm}_seed{s}_split0.pt', map_location='cpu'); m = Model(ck['block']).to(dev); m.load_state_dict(ck['model']); m.eval()
        thr = json.load(open(ROOT / 'results' / f'decode_instances_fte_ozf_{arm}_seed{s}_b{BUDGET}.json'))
        models.append((m, thr['instance']['thresholds'][0], thr['frame']['thresholds'][0]))
    for st in SETS:
        d = DATA / 'synth_test' / f'overlap_{st}'; recs = sorted(p.stem for p in (d / 'field16k').glob('*.npy'))
        agg = dict(nref=0, nov=0, hours=0.0, tp=0, npred=0, tpo=0, ftp=0, fnp=0, ftpo=0); sw = np.zeros((len(TH_INST), 3))
        for rec in recs:
            w = np.load(d / 'field16k' / f'{rec}.npy'); R = read_ann(d / 'ann' / f'{rec}.txt'); h = len(w) / SR / 3600
            for m, ti, tf in models:
                P, E = predict(m, w)
                ev = inst_events(E, P, ti[0], ti[1], ti[2], TOK_S); tp, tpo, nov = count(ev, R)
                fv = events_from_pcall(1.0 - P[:, NOISE], tf[0], tf[1]); ftp, ftpo, _ = count(fv, R)
                agg['nref'] += len(R); agg['nov'] += nov; agg['hours'] += h; agg['tp'] += tp; agg['npred'] += len(ev); agg['tpo'] += tpo; agg['ftp'] += ftp; agg['fnp'] += len(fv); agg['ftpo'] += ftpo
                for j, t in enumerate(TH_INST):
                    e2 = inst_events(E, P, t, ti[1], ti[2], TOK_S); a, b, _ = count(e2, R); sw[j] += (a, len(e2), b)
        fars = (sw[:, 1] - sw[:, 0]) / agg['hours']; o = np.argsort(fars)
        rm = float(np.interp(FAR_M, fars[o], (sw[:, 0] / agg['nref'])[o])) if fars.min() <= FAR_M <= fars.max() else float('nan')
        om = float(np.interp(FAR_M, fars[o], (sw[:, 2] / max(1, agg['nov']))[o])) if fars.min() <= FAR_M <= fars.max() else float('nan')
        r = dict(recall=agg['tp'] / agg['nref'], far=(agg['npred'] - agg['tp']) / agg['hours'], overlap_recall=agg['tpo'] / max(1, agg['nov']), overlap_frac=agg['nov'] / agg['nref'],
                 frame_recall=agg['ftp'] / agg['nref'], frame_far=(agg['fnp'] - agg['ftp']) / agg['hours'], frame_overlap_recall=agg['ftpo'] / max(1, agg['nov']), R_at_far=rm, OV_at_far=om, n_ref=agg['nref'] // 3)
        out[f'{arm}/overlap_{st}'] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
        print(f"{arm:5s} synthetic overlap {st:3s}: refs {r['n_ref']} (overlapping {r['overlap_frac']:.2f})  inst op R {r['recall']:.3f} FAR {r['far']:.0f} ovR {r['overlap_recall']:.3f} | R@{FAR_M:.0f} {rm:.3f} OV@{FAR_M:.0f} {om:.3f} | frame op R {r['frame_recall']:.3f} FAR {r['frame_far']:.0f} ovR {r['frame_overlap_recall']:.3f}", flush=True)
json.dump(out, open(ROOT / 'results' / f'ozf_synth_tests_b{BUDGET}.json', 'w'), indent=1); print('saved')
