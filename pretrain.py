"""Masked pretraining on synthesized 5 s mixtures.
variants: kmmel (HuBERT iter-0 targets: k-means on log-mel), kmaves (k-means on AVES-bio layer-9),
          d2v (continuous EMA-teacher latent regression, data2vec/JEPA-style)."""
import argparse, copy, math, time, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from common import *

ap = argparse.ArgumentParser()
ap.add_argument('--variant', required=True, choices=['kmmel', 'kmaves', 'd2v', 'rq', 'kmseg', 'kmboth', 'hybrid', 'jepa', 'kmoracle', 'kmrandseg', 'kmiter2'])
ap.add_argument('--steps', type=int, default=20000)
ap.add_argument('--bs', type=int, default=32)
ap.add_argument('--lr', type=float, default=5e-4)
ap.add_argument('--layers', type=int, default=8)
ap.add_argument('--d', type=int, default=384)
ap.add_argument('--n_seq', type=int, default=24000)
ap.add_argument('--tag', default='')
ap.add_argument('--block', default='vanilla', choices=['vanilla','modern'])
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--jepa_p', type=float, default=0.015)
ap.add_argument('--jepa_span', type=int, default=40)
ap.add_argument('--event_head', action='store_true', help='instance branch: presence/onset/log-duration from synthetic construction (extra supervision)')
ap.add_argument('--no_hubert', action='store_true', help='control: event branch only, no unit prediction loss')
ap.add_argument('--ev_w', type=float, default=1.0)
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = 'cuda'
name = a.variant + a.tag
out = ROOT / 'ckpt'; out.mkdir(exist_ok=True)

if a.variant in ('d2v', 'rq', 'jepa'): targets = None
elif a.variant == 'hybrid': targets = np.load(DATA / 'targets_kmaves.npy')[: a.n_seq]
elif a.variant == 'kmboth':
    t1, t2 = np.load(DATA / 'targets_kmmel.npy'), np.load(DATA / 'targets_kmaves.npy'); T = min(t1.shape[1], t2.shape[1])
    targets = np.stack([t1[:, :T], t2[:, :T]], -1)[: a.n_seq]
elif a.no_hubert and not (DATA / f'targets_{a.variant}.npy').exists(): targets = None   # event-only control without unit targets (OZF)
else: targets = np.load(DATA / f'targets_{a.variant}.npy')[: a.n_seq]
ds = SynthDataset(a.n_seq, targets, events=a.event_head)
dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=8, drop_last=True,
                                 persistent_workers=True, prefetch_factor=4, multiprocessing_context='fork')
stats = [s.to(dev) for s in load_stats()]
enc = build_encoder(a.block, d=a.d, layers=a.layers).to(dev)
K = 100
if a.variant == 'jepa':
    # predictor: 2-layer transformer over context embeddings + learned mask tokens at target positions
    class Predictor(nn.Module):
        def __init__(s, d):
            super().__init__(); s.mask = nn.Parameter(torch.randn(d) * 0.02)
            s.layers = nn.ModuleList([nn.TransformerEncoderLayer(d, 6, 4 * d, 0.1, activation='gelu', batch_first=True, norm_first=True) for _ in range(2)])
            s.out = nn.Linear(d, d)
        def forward(s, ctx, mask):
            x = torch.where(mask[..., None], s.mask.to(ctx.dtype), ctx)
            for l in s.layers: x = l(x)
            return s.out(x)
    head = Predictor(a.d).to(dev); teacher = copy.deepcopy(enc).eval()
    for p in teacher.parameters(): p.requires_grad_(False)
if a.variant in ('d2v', 'hybrid'):
    head = nn.Linear(a.d, a.d).to(dev); teacher = copy.deepcopy(enc).eval()
    for p in teacher.parameters(): p.requires_grad_(False)
    if a.variant == 'hybrid': head = nn.ModuleDict({'r': head, 'u': nn.Linear(a.d, K).to(dev)})
elif a.variant != 'jepa':
    head = nn.Linear(a.d, K).to(dev)
if a.variant == 'kmboth': head2 = nn.Linear(a.d, K).to(dev)
if a.variant == 'kmseg': head2 = nn.Linear(a.d, 1).to(dev)      # segment-start (boundary) head
if a.variant in ('kmboth', 'kmseg'): head = nn.ModuleDict({'u': head, 'b': head2})
ev_head = nn.Linear(a.d, 3).to(dev) if a.event_head else None     # presence, onset, log-duration
if a.variant == 'rq':   # BEST-RQ: fixed random projection + fixed random codebook, targets online
    g = torch.Generator().manual_seed(0)
    rq_proj = torch.randn(N_MELS * STACK, 16, generator=g).to(dev)
    rq_cb = F.normalize(torch.randn(K, 16, generator=g), dim=-1).to(dev)
params = list(enc.parameters()) + list(head.parameters()) + (list(ev_head.parameters()) if ev_head is not None else [])
print('params M', sum(p.numel() for p in params) / 1e6)
opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.98), weight_decay=0.01)
warm = 1500
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps))))

def ema_update(step):
    tau = 0.999 + (0.9999 - 0.999) * min(1, step / a.steps)
    with torch.no_grad():
        for pt, ps in zip(teacher.parameters(), enc.parameters()):
            pt.mul_(tau).add_(ps.detach(), alpha=1 - tau)

step = 0; t0 = time.time(); log = []
enc.train()
while step < a.steps:
    for batch_ in dl:
        wave, tgt = batch_[0], batch_[1]; et = batch_[2].to(dev, non_blocking=True) if a.event_head else None
        wave = wave.to(dev, non_blocking=True); tgt = tgt.to(dev, non_blocking=True)
        with torch.no_grad():
            tok = tokens_from_mel(melspec(wave), stats)
            if a.variant == 'rq':
                z = F.normalize(tok.float() @ rq_proj, dim=-1); tgt = torch.cdist(z, rq_cb[None]).argmin(-1)
            elif a.variant not in ('d2v', 'jepa'): T = min(tok.shape[1], tgt.shape[1]); tok, tgt = tok[:, :T], tgt[:, :T]
        B, T, _ = tok.shape
        mask = span_mask(B, T, device=dev) if a.variant != 'jepa' else span_mask(B, T, p=a.jepa_p, span=a.jepa_span, device=dev)   # JEPA masking: blocks by default, short spans for the control
        with torch.autocast('cuda', dtype=torch.bfloat16):
            x = enc(tok, mask)
            if a.variant == 'jepa':
                # context encoder sees ONLY unmasked tokens (masked positions zeroed, no mask embedding leakage), predictor fills targets
                ctx = enc(torch.where(mask[..., None], torch.zeros_like(tok), tok), None)
                with torch.no_grad():
                    _, hs = teacher(tok, None, return_layers=True)
                    y = F.instance_norm(hs[-1].float().transpose(1, 2)).transpose(1, 2)
                pred = head(ctx, mask).float(); loss = F.mse_loss(pred[mask], y[mask]); acc = torch.tensor(0.)
            elif a.variant == 'hybrid':
                with torch.no_grad():
                    _, hs = teacher(tok, None, return_layers=True)
                    y = torch.stack([F.instance_norm(h.float().transpose(1, 2)).transpose(1, 2) for h in hs[-4:]]).mean(0)
                logits = head['u'](x).float()
                loss = F.cross_entropy(logits[mask], tgt[mask]) + F.mse_loss(head['r'](x).float()[mask], y[mask])
                acc = (logits[mask].argmax(-1) == tgt[mask]).float().mean()
            elif a.variant == 'd2v':
                with torch.no_grad():
                    _, hs = teacher(tok, None, return_layers=True)
                    y = torch.stack([F.instance_norm(h.float().transpose(1, 2)).transpose(1, 2) for h in hs[-4:]]).mean(0)
                pred = head(x).float()
                loss = F.mse_loss(pred[mask], y[mask])
                acc = torch.tensor(0.)
            elif a.variant == 'kmboth':
                l1, l2 = head['u'](x).float(), head['b'](x).float()
                loss = F.cross_entropy(l1[mask], tgt[..., 0][mask]) + F.cross_entropy(l2[mask], tgt[..., 1][mask])
                acc = (l2[mask].argmax(-1) == tgt[..., 1][mask]).float().mean()
            elif a.variant == 'kmseg':
                logits, bl = head['u'](x).float(), head['b'](x).float()[..., 0]
                u, b = tgt[..., 0], tgt[..., 1].float()
                loss = F.cross_entropy(logits[mask], u[mask]) + F.binary_cross_entropy_with_logits(bl[mask], b[mask], pos_weight=torch.tensor(3.0, device=dev))
                acc = (logits[mask].argmax(-1) == u[mask]).float().mean()
            else:
                logits = head(x).float()
                loss = F.cross_entropy(logits[mask], tgt[mask])
                acc = (logits[mask].argmax(-1) == tgt[mask]).float().mean()
        if a.event_head:
            Tm = min(x.shape[1], et.shape[1]); eo = ev_head(x[:, :Tm].float()); e = et[:, :Tm]
            l_pres = F.binary_cross_entropy_with_logits(eo[..., 0], e[..., 0])
            l_on = F.binary_cross_entropy_with_logits(eo[..., 1], e[..., 1], pos_weight=torch.tensor(10.0, device=dev))
            dm = e[..., 3] > 0; l_dur = F.l1_loss(eo[..., 2][dm], e[..., 2][dm]) if dm.any() else eo.sum() * 0
            l_ev = l_pres + l_on + 0.5 * l_dur
            loss = (loss * 0 if a.no_hubert else loss) + a.ev_w * l_ev
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); step += 1
        if a.variant in ('d2v', 'hybrid', 'jepa'): ema_update(step)
        if step % 100 == 0:
            rec = dict(step=step, loss=round(loss.item(), 4), acc=round(acc.item(), 4), lr=sched.get_last_lr()[0],
                       ev=(round(l_ev.item(), 4) if a.event_head else None),
                       mask_frac=round(mask.float().mean().item(), 3), min=round((time.time() - t0) / 60, 1))
            if step % 1000 == 0:
                try:
                    import expdb; rec['gpu'] = expdb.telemetry()
                except Exception: pass
            log.append(rec); print(json.dumps(rec), flush=True)
        if step % 5000 == 0 or step == a.steps:
            torch.save({'enc': enc.state_dict(), 'head': head.state_dict(), 'args': vars(a), 'step': step,
                        'ev_head': (ev_head.state_dict() if ev_head is not None else None)},
                       out / f'{name}_{step}.pt')
        if step >= a.steps: break
json.dump(log, open(out / f'{name}_log.json', 'w'))
print('done', name, 'min', round((time.time() - t0) / 60, 1))
try:
    import expdb; rid = expdb.record_pretrain(vars(a), log, sum(p.numel() for p in params) / 1e6, out / f'{name}_{a.steps}.pt', (time.time() - t0) / 60)
    print('expdb run id', rid)
except Exception as e: print('expdb record failed:', e)
