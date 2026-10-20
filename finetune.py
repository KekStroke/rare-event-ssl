"""Full fine-tuning of a pretrained encoder (or scratch) for frame-level call detection, same folds / decoder / scorer as probe.py.
Adaptation axis of the benchmark: frozen probe (probe.py) vs full FT (this).
"""
import argparse, json, itertools, time, copy
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from common import *

ap = argparse.ArgumentParser()
ap.add_argument('--feats', required=True)            # ckpt path | 'scratch'
ap.add_argument('--name', required=True)
ap.add_argument('--steps', type=int, default=1500)
ap.add_argument('--lr', type=float, default=None)
ap.add_argument('--head_lr', type=float, default=1e-3)
ap.add_argument('--chunk_s', type=float, default=10.0)
ap.add_argument('--bs', type=int, default=8)
ap.add_argument('--block', default=None, choices=[None, 'vanilla', 'modern'])
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--onset_head', action='store_true')
ap.add_argument('--onset_w', type=float, default=1.0)
ap.add_argument('--event_head', action='store_true', help='instance branch presence/onset/log-duration; init from ckpt ev_head if present')
a = ap.parse_args()
dev = 'cuda'; torch.manual_seed(a.seed); np.random.seed(a.seed)
res_dir = ROOT / 'results'; res_dir.mkdir(exist_ok=True)
stats = [s.to(dev) for s in load_stats()]
block = 'vanilla'
if a.feats == 'aves': block = 'aves-hubert-base'
elif a.feats != 'scratch':
    ck = torch.load(a.feats, map_location='cpu'); block = ck['args'].get('block', 'vanilla')
if a.block: block = a.block
print('block', block)
if a.lr is None: a.lr = 2e-5 if a.feats == 'aves' else 5e-5

stems = field_stems(); waves = {s: np.load(DATA / 'field16k' / f'{s}.npy') for s in stems}
refs = {s: ann_for(s) for s in stems}
n_chunk = int(a.chunk_s * SR)

def chunks(stem):
    w = waves[stem]
    for s0 in range(0, len(w), n_chunk):
        c = w[s0:s0 + n_chunk]
        if len(c) < SR // 2: break
        yield s0 / SR, c

class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.enc = build_encoder(block); self.head = nn.Linear(self.enc.d, NOISE + 1)
        self.onset = nn.Linear(self.enc.d, 1) if a.onset_head else None
        self.ev = nn.Linear(self.enc.d, 3) if a.event_head else None
    def forward(self, wave):
        tok = tokens_from_mel(melspec(wave), stats); h = self.enc(tok)
        if self.ev is not None: return self.head(h), self.ev(h)
        return (self.head(h), self.onset(h)[..., 0]) if self.onset is not None else self.head(h)

class AvesModel(nn.Module):
    def __init__(self):
        super().__init__()
        from prep import load_aves
        self.enc = load_aves(dev).train(); self.head = nn.Linear(768, NOISE + 1)
        self.onset = nn.Linear(768, 1) if a.onset_head else None
    def forward(self, wave):
        hs, _ = self.enc.extract_features(wave); h = hs[-1]
        n_exp = wave.shape[1] // (HOP * STACK)
        if h.shape[1] < n_exp: h = torch.cat([h, h[:, -1:].expand(-1, n_exp - h.shape[1], -1)], 1)
        h = h[:, :n_exp]
        return (self.head(h), self.onset(h)[..., 0]) if self.onset is not None else self.head(h)

def make_model():
    if a.feats == 'aves': return AvesModel().to(dev)
    m = Model().to(dev)
    if a.feats != 'scratch':
        m.enc.load_state_dict(ck['enc'])
        if a.event_head and ck.get('ev_head') is not None: m.ev.load_state_dict(ck['ev_head']); print('event head initialised from pretraining')
    return m

def train(train_stems):
    m = make_model(); m.train()
    items = [(st, t0, c) for st in train_stems for t0, c in chunks(st)]
    # frame targets per chunk
    Y = {}; ON = {}; EVT = {}
    for st, t0, c in items:
        n_tok = len(c) // (HOP * STACK)
        ev = [(s - t0, e - t0, l) for s, e, l in refs[st] if e > t0 and s < t0 + len(c) / SR]
        Y[(st, t0)] = torch.tensor(frame_targets(ev, n_tok))
        on = torch.zeros(n_tok)
        for s_, e_, l_ in ev:
            i = int(s_ / TOK_S)
            if 0 <= i < n_tok: on[max(0, i - 1): i + 2] = 1.0
        ON[(st, t0)] = on
        if a.event_head:
            from common import event_targets
            ev_tok = [(int(s_ / TOK_S), int(e_ / TOK_S), 0) for s_, e_, l_ in ev]
            EVT[(st, t0)] = torch.tensor(np.stack(event_targets(ev_tok, n_tok), 1))
    cnt = torch.bincount(torch.cat(list(Y.values())), minlength=NOISE + 1).float(); wcls = (cnt.sum() / (cnt + 1)).sqrt(); wcls = (wcls / wcls.mean()).to(dev)
    opt = torch.optim.AdamW([{'params': m.enc.parameters(), 'lr': a.lr}, {'params': m.head.parameters(), 'lr': a.head_lr}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 100) * 0.5 * (1 + np.cos(np.pi * min(1, s / a.steps))))
    rng = np.random.default_rng(a.seed)
    for it in range(a.steps):
        batch = [items[i] for i in rng.integers(0, len(items), a.bs)]
        L = min(len(c) for _, _, c in batch); L = L // (HOP * STACK) * (HOP * STACK)
        w = torch.tensor(np.stack([c[:L] for _, _, c in batch])).to(dev)
        y = torch.stack([Y[(st, t0)][: L // (HOP * STACK)] for st, t0, _ in batch]).to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            outp = m(w)
        logits = (outp[0] if (a.onset_head or a.event_head) else outp).float()
        T = min(logits.shape[1], y.shape[1])
        loss = F.cross_entropy(logits[:, :T].reshape(-1, NOISE + 1), y[:, :T].reshape(-1), weight=wcls)
        if a.event_head:
            e = torch.stack([EVT[(st, t0)][: L // (HOP * STACK)] for st, t0, _ in batch]).to(dev)[:, :T]; eo = outp[1].float()[:, :T]
            l_pres = F.binary_cross_entropy_with_logits(eo[..., 0], e[..., 0]); l_on = F.binary_cross_entropy_with_logits(eo[..., 1], e[..., 1], pos_weight=torch.tensor(10.0, device=dev))
            dm = e[..., 3] > 0; l_dur = F.l1_loss(eo[..., 2][dm], e[..., 2][dm]) if dm.any() else eo.sum() * 0
            loss = loss + l_pres + l_on + 0.5 * l_dur
        if a.onset_head:
            ob = torch.stack([ON[(st, t0)][: L // (HOP * STACK)] for st, t0, _ in batch]).to(dev)[:, :T]
            loss = loss + a.onset_w * F.binary_cross_entropy_with_logits(outp[1].float()[:, :T], ob, pos_weight=torch.tensor(20.0, device=dev))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
    return m.eval()

ONSET_TRACKS = {}
@torch.no_grad()
def predict(m, stem):
    out, ons = [], []
    for t0, c in chunks(stem):
        w = torch.tensor(c)[None].to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o = m(w)
        if a.event_head:
            out.append(torch.softmax(o[0].float(), -1)[0].cpu().numpy()); e = o[1].float()[0]
            ons.append(np.stack([torch.sigmoid(e[:, 0]).cpu().numpy(), torch.sigmoid(e[:, 1]).cpu().numpy(), e[:, 2].cpu().numpy()], 1))
        elif a.onset_head: out.append(torch.softmax(o[0].float(), -1)[0].cpu().numpy()); ons.append(torch.sigmoid(o[1].float())[0].cpu().numpy())
        else: out.append(torch.softmax(o.float(), -1)[0].cpu().numpy())
    if a.onset_head or a.event_head: ONSET_TRACKS[stem] = np.concatenate(ons).astype(np.float16)
    return np.concatenate(out)

GRID = list(itertools.product([0.5, 0.6, 0.7, 0.8, 0.9], [0.2, 0.3, 0.4, 0.5]))
pooled, per_fold = [], []; t_start = time.time(); PC_ALL = {}; PC_FOLD = {}; FULL_P = {}
for k, (train_stems, devs, test) in enumerate(splits()):
    m = train(train_stems); P = {s: predict(m, s) for s in devs + test}
    if DATASET != 'whale': torch.save({'model': m.state_dict(), 'block': block, 'args': vars(a)}, ROOT / 'ckpt' / f'ft_{a.name}_split{k}.pt')
    best, bf = None, -1
    for det, edge in GRID:
        sc = score([(decode(P[s], det, edge), refs[s]) for s in devs], taus=[0.2])[('cls', 0.2)][2]
        if sc > bf: bf, best = sc, (det, edge)
    recs = [(decode(P[s], *best), refs[s]) for s in test]; pooled += recs; sc = score(recs)
    PC_ALL.update({s: (1.0 - P[s][:, NOISE]).astype(np.float16) for s in devs + test}); PC_FOLD.update({s: k for s in test}); FULL_P.update({s: P[s] for s in devs + test})
    per_fold.append(dict(fold=k, thr=best, dev_f1=round(bf, 4), test_f1_cls_0p2=round(sc[('cls', 0.2)][2], 4),
                         n_pred=sum(len(p) for p, _ in recs), n_ref=sum(len(r) for _, r in recs)))
    print(per_fold[-1], 'min', round((time.time() - t_start) / 60, 1), flush=True)
    del m; torch.cuda.empty_cache()

def timing_errors(recordings, tau=1.0):
    on = []
    for preds, rs in recordings:
        used = np.zeros(len(preds), bool)
        for r0, r1, _ in rs:
            best_j, bd = -1, 1e9
            for j, (p0, p1, _) in enumerate(preds):
                if used[j] or abs(p0 - r0) > tau or abs(p1 - r1) > max(tau, 0.5 * (r1 - r0)): continue
                if abs(p0 - r0) < bd: best_j, bd = j, abs(p0 - r0)
            if best_j >= 0: used[best_j] = True; on.append(preds[best_j][0] - r0)
    on = np.abs(np.array(on)); return dict(n_matched=int(len(on)), abs_onset_p50_75_90=[round(float(v), 3) for v in np.percentile(on, [50, 75, 90])] if len(on) else None)

sc = score(pooled); timing = timing_errors(pooled)
table = {f'{mm}@{t}': [round(v, 4) for v in sc[(mm, t)]] for (mm, t) in sc}
res = dict(name=a.name, feats=a.feats, mode='full_ft', per_fold=per_fold, pooled=table, timing=timing, args=vars(a))
json.dump(res, open(res_dir / f'{a.name}.json', 'w'), indent=1)
json.dump({'recordings': [[[list(e) for e in pr], [list(e) for e in rf]] for pr, rf in pooled]}, open(res_dir / f'{a.name}_events.json', 'w'))
np.savez_compressed(res_dir / f'{a.name}_pcall.npz', **{f'p__{s}': v for s, v in PC_ALL.items()}, **{f'fold__{s}': np.int8(k) for s, k in PC_FOLD.items()}, tok_s=TOK_S)
if a.onset_head or a.event_head:
    np.savez_compressed(res_dir / f'{a.name}_tracks.npz', **{f'P__{s}': v.astype(np.float16) for s, v in FULL_P.items()}, **{f'on__{s}': v for s, v in ONSET_TRACKS.items()}, tok_s=TOK_S)
print(a.name); print('tau      ' + ' '.join(f'{t:>6}' for t in TAUS))
print('F1 cls   ' + ' '.join(f'{sc[("cls", t)][2]:6.3f}' for t in TAUS)); print('F1 any   ' + ' '.join(f'{sc[("any", t)][2]:6.3f}' for t in TAUS))
print('timing', timing)
try:
    import expdb; proto = dict(expdb.PROTOCOL); proto.update(adaptation='full fine-tuning', block=block, event_head=a.event_head, onset_head=a.onset_head, ft_steps=a.steps, lr=a.lr, head_lr=a.head_lr, bs=a.bs, chunk_s=a.chunk_s)
    print('expdb run id', expdb.record_probe(res, proto, a.feats))
except Exception as e: print('expdb record failed:', e)
