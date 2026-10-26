"""Frozen frame-level probe on the 17 Avacha recordings with 3 recording-held-out folds,
two-threshold decoding and event F1 over an onset-collar sweep.

--feats: path to a pretrain checkpoint | 'aves' | 'mel' | 'scratch'
"""
import argparse, json, itertools, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from common import *

ap = argparse.ArgumentParser()
ap.add_argument('--feats', required=True)
ap.add_argument('--name', required=True)
ap.add_argument('--steps', type=int, default=3000)
ap.add_argument('--chunk_s', type=float, default=10.0)
a = ap.parse_args()
dev = 'cuda'; torch.manual_seed(0)
res_dir = ROOT / 'results'; res_dir.mkdir(exist_ok=True)

# ------------------------------------------------------------------ feature extraction
stats = [s.to(dev) for s in load_stats()]
if a.feats in ('mel', 'scratch', 'aves'):
    kind = a.feats
else:
    kind = 'ckpt'
if kind in ('ckpt', 'scratch'):
    block = 'vanilla'
    if kind == 'ckpt':
        ck = torch.load(a.feats, map_location='cpu'); block = ck['args'].get('block', 'vanilla')
    enc = build_encoder(block).to(dev).eval()
    if kind == 'ckpt':
        enc.load_state_dict(ck['enc'])
        print('loaded', a.feats, 'step', ck['step'])
if kind == 'aves':
    from prep import load_aves
    aves = load_aves(dev)

@torch.no_grad()
def extract(wave_np):
    """-> float16 tensor [L, T, D] on GPU, T at 50 Hz."""
    n = int(a.chunk_s * SR); outs = []
    for s in range(0, len(wave_np), n):
        w = torch.tensor(wave_np[s:s + n])[None].to(dev)
        if len(w[0]) < SR // 2: break
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if kind == 'aves':
                hs, _ = aves.extract_features(w); layers = torch.stack(hs)[:, 0]
                n_exp = w.shape[1] // (HOP * STACK)                      # tokens the 20 ms grid expects for this chunk
                if layers.shape[1] < n_exp:                              # conv front-end yields 499/10 s: pad to keep absolute time
                    layers = torch.cat([layers, layers[:, -1:].expand(-1, n_exp - layers.shape[1], -1)], 1)
                layers = layers[:, :n_exp]
            else:
                tok = tokens_from_mel(melspec(w), stats)
                if kind == 'mel': layers = tok
                else:
                    _, hs = enc(tok, None, return_layers=True); layers = torch.stack(hs)[:, 0]
        outs.append(layers.half())
    return torch.cat(outs, dim=-2)

stems = field_stems(); feats, labels, refs = {}, {}, {}
t0 = time.time()
for st in stems:
    w = np.load(DATA / 'field16k' / f'{st}.npy'); f = extract(w)
    feats[st] = f; refs[st] = ann_for(st); labels[st] = torch.tensor(frame_targets(refs[st], f.shape[-2])).to(dev)
print('features', {k: tuple(v.shape) for k, v in list(feats.items())[:2]}, 'sec', round(time.time() - t0))
L, D = feats[stems[0]].shape[0], feats[stems[0]].shape[-1]

# ------------------------------------------------------------------ probe
class Probe(nn.Module):
    def __init__(self):
        super().__init__(); self.w = nn.Parameter(torch.zeros(L)); self.lin = nn.Linear(D, NOISE + 1)
        self.norm = nn.LayerNorm(D)
    def forward(self, x):  # x [L, N, D]
        x = (torch.softmax(self.w, 0)[:, None, None] * x.float()).sum(0)
        return self.lin(self.norm(x))

def train_probe(train_stems):
    X = torch.cat([feats[s] for s in train_stems], dim=1); Y = torch.cat([labels[s] for s in train_stems])
    cnt = torch.bincount(Y, minlength=NOISE + 1).float(); wcls = (cnt.sum() / (cnt + 1)).sqrt(); wcls /= wcls.mean()
    pr = Probe().to(dev); opt = torch.optim.AdamW(pr.parameters(), lr=2e-3, weight_decay=1e-4)
    N = X.shape[1]
    for it in range(a.steps):
        idx = torch.randint(0, N, (8192,), device=dev)
        loss = F.cross_entropy(pr(X[:, idx]), Y[idx], weight=wcls)
        opt.zero_grad(); loss.backward(); opt.step()
    return pr

@torch.no_grad()
def predict(pr, st):
    return torch.softmax(pr(feats[st]), -1).cpu().numpy()

GRID = list(itertools.product([0.5, 0.6, 0.7, 0.8, 0.9], [0.2, 0.3, 0.4, 0.5]))
pooled = []; per_fold = []; PC_ALL = {}; PC_FOLD = {}
for k in range(3):
    test = FOLDS[k]; others = [f for i, f in enumerate(FOLDS) if i != k]
    devs = [others[0][0], others[1][0]]
    train = [s for f in others for s in f if s not in devs]
    pr = train_probe(train)
    P = {s: predict(pr, s) for s in devs + test}
    best, bf = None, -1
    for det, edge in GRID:
        sc = score([(decode(P[s], det, edge), refs[s]) for s in devs], taus=[0.2])[('cls', 0.2)][2]
        if sc > bf: bf, best = sc, (det, edge)
    recs = [(decode(P[s], *best), refs[s]) for s in test]
    pooled += recs; sc = score(recs)
    PC_ALL.update({s: (1.0 - P[s][:, NOISE]).astype(np.float16) for s in devs + test}); PC_FOLD.update({s: k for s in test})
    lw = torch.softmax(pr.w.detach(), 0).cpu().numpy().round(3).tolist()
    per_fold.append(dict(fold=k, thr=best, dev_f1=round(bf, 4), test_f1_cls_0p2=round(sc[('cls', 0.2)][2], 4),
                         n_pred=sum(len(p) for p, _ in recs), n_ref=sum(len(r) for _, r in recs), layer_weights=lw))
    print(per_fold[-1], flush=True)

def timing_errors(recordings, tau=1.0):
    on, off = [], []
    for preds, rs in recordings:
        used = np.zeros(len(preds), bool)
        for r0, r1, _ in rs:
            best, bd = -1, 1e9
            for j, (p0, p1, _) in enumerate(preds):
                if used[j] or abs(p0 - r0) > tau or abs(p1 - r1) > max(tau, 0.5 * (r1 - r0)): continue
                if abs(p0 - r0) < bd: best, bd = j, abs(p0 - r0)
            if best >= 0: used[best] = True; on.append(preds[best][0] - r0); off.append(preds[best][1] - r1)
    on, off = np.array(on), np.array(off)
    q = lambda x: [round(float(v), 3) for v in np.percentile(np.abs(x), [50, 75, 90])] if len(x) else None
    return dict(n_matched=int(len(on)), abs_onset_p50_75_90=q(on), abs_offset_p50_75_90=q(off),
                onset_bias=round(float(on.mean()), 3) if len(on) else None, offset_bias=round(float(off.mean()), 3) if len(off) else None,
                onset_hist_20ms=np.histogram(on, bins=np.arange(-0.5, 0.52, 0.02))[0].tolist() if len(on) else None)
timing = timing_errors(pooled)
# ---- screening operating point ("Olga mode"): maximise class-agnostic recall s.t. precision >= 0.5 at tau=0.5
RGRID = list(itertools.product([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [0.1, 0.2, 0.3, 0.4, 0.5]))
def rec_eval(recs, tau=0.5):
    tp = sum(match(p, r, tau, False) for p, r in recs); npred = sum(len(p) for p, _ in recs); nref = sum(len(r) for _, r in recs)
    return prf(tp, npred, nref), npred, nref
dur_h = {s: len(np.load(DATA / 'field16k' / f'{s}.npy')) / SR / 3600 for s in stems}
op_pooled, op_folds = [], []
for k in range(3):
    test = FOLDS[k]; others = [f for i, f in enumerate(FOLDS) if i != k]; devs = [others[0][0], others[1][0]]
    train = [s for f in others for s in f if s not in devs]
    pr = train_probe(train); P = {s: predict(pr, s) for s in devs + test}
    best, br = None, -1
    for det, edge in RGRID:
        (p_, r_, _), _, _ = rec_eval([(decode(P[s], det, edge), refs[s]) for s in devs])
        if p_ >= 0.5 and r_ > br: br, best = r_, (det, edge)
    if best is None: best = (0.9, 0.5)
    recs = [(decode(P[s], *best), refs[s]) for s in test]; op_pooled += recs
    op_folds.append(dict(fold=k, thr=best, dev_recall=round(br, 3)))
(p_, r_, f_), npred, nref = rec_eval(op_pooled)
tp_total = sum(match(p, r, 0.5, False) for p, r in op_pooled); hours = sum(dur_h.values())
per_class = {}
for c in CLASSES:
    sub = [(p, [e for e in r if e[2] == c]) for p, r in op_pooled]
    n = sum(len(r) for _, r in sub)
    if n: per_class[c] = dict(n=n, recall=round(sum(match(p, r, 0.5, False) for p, r in sub) / n, 3))
recall_op = dict(precision=round(p_, 3), recall=round(r_, 3), f1=round(f_, 3), far_per_hour=round((npred - tp_total) / hours, 1),
                 n_pred=npred, n_ref=nref, hours=round(hours, 2), folds=op_folds, per_class_recall=per_class,
                 recall_at_tau02=round(rec_eval(op_pooled, 0.2)[0][1], 3))
sc = score(pooled)
table = {f'{m}@{t}': [round(v, 4) for v in sc[(m, t)]] for (m, t) in sc}
res = dict(name=a.name, feats=a.feats, per_fold=per_fold, pooled=table, timing=timing, recall_op=recall_op)
np.savez_compressed(res_dir / f'{a.name}_pcall.npz', **{f'p__{s}': v for s, v in PC_ALL.items()}, **{f'fold__{s}': np.int8(k) for s, k in PC_FOLD.items()}, tok_s=TOK_S)
json.dump({'stems': [s for k in range(3) for s in FOLDS[k]], 'recordings': [[[list(e) for e in pr], [list(e) for e in rf]] for pr, rf in pooled]},
          open(res_dir / f'{a.name}_events.json', 'w'))
json.dump(res, open(res_dir / f'{a.name}.json', 'w'), indent=1)
try:
    import expdb, subprocess, sys as _s; proto = dict(expdb.PROTOCOL); proto['probe_steps'] = a.steps; print('expdb run id', expdb.record_probe(res, proto, a.feats))
    subprocess.run([_s.executable, str(ROOT / 'expdb' / 'make_findings.py')], stdout=subprocess.DEVNULL, stderr=open(ROOT / 'logs' / 'make_findings.err', 'a'))
    subprocess.run(['cp', str(ROOT / 'expdb' / 'docs' / 'FINDINGS.md'), '/mnt/c/Users/Iaroslav/Downloads/TSSSL_FINDINGS.md'])
except Exception as e: print('expdb record failed:', e)
print(a.name)
print('tau      ' + ' '.join(f'{t:>6}' for t in TAUS))
print('F1 cls   ' + ' '.join(f'{sc[("cls", t)][2]:6.3f}' for t in TAUS))
print('F1 any   ' + ' '.join(f'{sc[("any", t)][2]:6.3f}' for t in TAUS))
print('P/R cls@0.2', [round(v, 3) for v in sc[('cls', 0.2)]])
print('timing', {k: v for k, v in timing.items() if k != 'onset_hist_20ms'})
print('recall_op', {k: v for k, v in recall_op.items() if k not in ('folds',)})
