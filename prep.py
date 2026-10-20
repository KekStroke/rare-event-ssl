"""Data preparation. Stages: lab | field | stats | kmmel | kmaves  (run in that order)."""
import sys, json, time
from pathlib import Path
import numpy as np, torch, soundfile as sf
import torch.nn.functional as F
from scipy.signal import resample_poly
from tqdm import tqdm
from common import *

N_SEQ = 24000
K = 100
FIT_SEQS = 3000
AVES_PT = Path('/mnt/c/Users/Iaroslav/CETACEANS/voxaboxen_weights/aves-base-bio.torchaudio.pt')
AVES_CFG = AVES_PT.with_suffix('.model_config.json')
AVES_LAYER = 9

def to16k(x, sr):
    if x.ndim > 1: x = x.mean(1)
    if sr == SR: return x.astype(np.float32)
    from math import gcd
    g = gcd(sr, SR); return resample_poly(x, SR // g, sr // g).astype(np.float32)

def stage_lab():
    files = sorted((DATA / 'lab').glob('*.wav'))
    print('lab files', len(files))
    audio, off, cls = [], [0], []
    for p in tqdm(files):
        x, sr = sf.read(p, dtype='float32'); x = to16k(x, sr)
        audio.append(x.astype(np.float16)); off.append(off[-1] + len(x))
        pre = p.name.split('-')[0].upper()
        cls.append(CLASSES.index(pre) if pre in CLASSES else -1)
    np.savez(DATA / 'lab16k.npz', audio=np.concatenate(audio), off=np.array(off), cls=np.array(cls))
    print('hours', off[-1] / SR / 3600, 'noise', sum(c < 0 for c in cls))

def stage_field():
    (DATA / 'field16k').mkdir(exist_ok=True)
    for p in sorted((DATA / 'field').glob('*.wav')):
        x, sr = sf.read(p, dtype='float32'); x = to16k(x, sr)
        np.save(DATA / 'field16k' / f'{p.stem}.npy', x); print(p.stem, sr, len(x) / SR)

def stage_stats():
    lab = Lab(); dev = 'cuda'
    acc = []
    for i in range(0, 2000, 100):
        w = torch.tensor(np.stack([synth(lab, j) for j in range(i, i + 100)])).to(dev)
        acc.append(melspec(w).reshape(-1, N_MELS).cpu())
    m = torch.cat(acc); np.savez(DATA / 'mel_stats.npz', mu=m.mean(0).numpy(), sd=m.std(0).numpy() + 1e-5)
    print('stats done')

class TorchKMeans:
    """Lloyd k-means on the GPU: k-means++ init on a subsample, `restarts` restarts, best inertia kept."""
    def __init__(self, k, iters=60, restarts=3, seed=0, dev='cuda'):
        self.k, self.iters, self.restarts, self.seed, self.dev = k, iters, restarts, seed, dev
    def _init(self, X, g):
        n = X.shape[0]; sub = X[torch.randperm(n, generator=g, device=self.dev)[:min(n, 200000)]]
        C = sub[torch.randint(0, sub.shape[0], (1,), generator=g, device=self.dev)]
        for _ in range(1, self.k):
            d = torch.cdist(sub, C).min(1).values ** 2
            C = torch.cat([C, sub[torch.multinomial(d / d.sum(), 1, generator=g)]])
        return C
    def fit(self, X):
        X = torch.as_tensor(X, device=self.dev, dtype=torch.float32); best = None
        for r in range(self.restarts):
            g = torch.Generator(device=self.dev).manual_seed(self.seed + r); C = self._init(X, g)
            for _ in range(self.iters):
                a = self._assign(X, C)
                Cn = torch.zeros_like(C).index_add_(0, a, X); cnt = torch.bincount(a, minlength=self.k).float()
                empty = cnt == 0
                C = torch.where(empty[:, None], X[torch.randint(0, X.shape[0], (self.k,), generator=g, device=self.dev)], Cn / cnt.clamp(min=1)[:, None])
            inertia = ((X - C[self._assign(X, C)]) ** 2).sum().item()
            if best is None or inertia < best[0]: best = (inertia, C.clone())
        self.inertia_, self.cluster_centers_ = best[0], best[1].cpu().numpy(); return self
    def _assign(self, X, C):
        out = torch.empty(X.shape[0], dtype=torch.long, device=self.dev)
        for i in range(0, X.shape[0], 65536):
            out[i:i + 65536] = torch.cdist(X[i:i + 65536], C).argmin(1)
        return out
    def predict(self, X):
        return self._assign(torch.as_tensor(X, device=self.dev, dtype=torch.float32),
                            torch.as_tensor(self.cluster_centers_, device=self.dev)).cpu().numpy()

def _fit_assign(feat_fn, name, dim_t):
    """feat_fn(batch_idx list) -> np.float32 [B, T, D]. Fits k-means on FIT_SEQS, assigns all N_SEQ."""
    bs = 64
    fit = np.concatenate([feat_fn(list(range(i, i + bs))) for i in range(0, FIT_SEQS, bs)]).reshape(-1, dim_t[1])
    print(name, 'fit frames', fit.shape, flush=True); t = time.time()
    km = TorchKMeans(K).fit(fit); print(name, 'kmeans', round(time.time() - t, 1), 's inertia', round(km.inertia_ / len(fit), 3), flush=True)
    T = dim_t[0]; out = np.zeros((N_SEQ, T), np.int16)
    for i in tqdm(range(0, N_SEQ, bs)):
        f = feat_fn(list(range(i, min(i + bs, N_SEQ))))
        out[i:i + len(f)] = km.predict(f.reshape(-1, f.shape[-1])).reshape(len(f), T)
    np.save(DATA / f'targets_{name}.npy', out)
    np.save(DATA / f'centroids_{name}.npy', km.cluster_centers_.astype(np.float32))
    h = np.bincount(out.ravel(), minlength=K) / out.size
    print(name, 'done; cluster entropy (bits)', float(-(h[h > 0] * np.log2(h[h > 0])).sum()), 'of', np.log2(K), flush=True)

def stage_kmmel():
    lab = Lab(); stats = [s.cuda() for s in load_stats()]
    def feat(ids):
        w = torch.tensor(np.stack([synth(lab, j) for j in ids])).cuda()
        return tokens_from_mel(melspec(w), stats).float().cpu().numpy()
    _fit_assign(feat, 'kmmel', (SEQ_N // (HOP * STACK), N_MELS * STACK))

def load_aves(dev='cuda'):
    import torchaudio
    cfg = json.load(open(AVES_CFG))
    m = torchaudio.models.wav2vec2_model(**cfg, aux_num_out=None)
    m.load_state_dict(torch.load(AVES_PT, map_location='cpu')); return m.to(dev).eval()

def stage_kmaves():
    lab = Lab(); aves = load_aves()
    @torch.no_grad()
    def feat(ids):
        w = torch.tensor(np.stack([synth(lab, j) for j in ids])).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            hs, _ = aves.extract_features(w, num_layers=AVES_LAYER)
        return hs[-1].float().cpu().numpy()
    T = feat([0]).shape[1]; print('aves tokens per 5 s', T)
    _fit_assign(feat, 'kmaves', (T, 768))


# ----------------------------------------------------------------------------- segment (variable-length) units
SEG_BOUND_FRAC = 0.25     # fraction of tokens that start a new segment -> mean segment ~4 tokens (80 ms)

def stage_kmseg():
    """Boundary-aware targets: merge consecutive AVES frames into segments by cosine change-point,
    one unit per segment (nearest kmaves centroid to the segment mean), plus a segment-start flag."""
    lab = Lab(); aves = load_aves(); cent = torch.tensor(np.load(DATA / 'centroids_kmaves.npy')).cuda()
    @torch.no_grad()
    def feats(ids):
        w = torch.tensor(np.stack([synth(lab, j) for j in ids])).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            hs, _ = aves.extract_features(w, num_layers=AVES_LAYER)
        return hs[-1].float()
    bs = 64
    sims = torch.cat([F.cosine_similarity(f[:, 1:], f[:, :-1], dim=-1).flatten()
                      for f in (feats(list(range(i, i + bs))) for i in range(0, 1024, bs))])
    thr = torch.quantile(sims, SEG_BOUND_FRAC).item(); print('kmseg boundary threshold cos', round(thr, 4), flush=True)
    T = feats([0]).shape[1]; units = np.zeros((N_SEQ, T), np.int16); bounds = np.zeros((N_SEQ, T), np.int8); seglens = []
    for i in tqdm(range(0, N_SEQ, bs)):
        f = feats(list(range(i, min(i + bs, N_SEQ))))
        sim = F.cosine_similarity(f[:, 1:], f[:, :-1], dim=-1)
        b = torch.cat([torch.ones(f.shape[0], 1, dtype=torch.bool, device=f.device), sim < thr], 1)   # [B, T]
        seg_id = torch.cumsum(b.long(), 1) - 1
        for k in range(f.shape[0]):
            n = int(seg_id[k, -1]) + 1
            means = torch.zeros(n, f.shape[-1], device=f.device).index_add_(0, seg_id[k], f[k])
            cnt = torch.bincount(seg_id[k], minlength=n).float()
            u = torch.cdist(means / cnt[:, None], cent).argmin(1)
            units[i + k] = u[seg_id[k]].cpu().numpy(); bounds[i + k] = b[k].cpu().numpy(); seglens.append(cnt.cpu().numpy())
    np.save(DATA / 'targets_kmseg.npy', np.stack([units, bounds.astype(np.int16)], -1))
    sl = np.concatenate(seglens); print('kmseg done; segment len tokens mean/median/p90', sl.mean().round(2), np.median(sl), np.percentile(sl, 90),
          'unit switch rate', float((units[:, 1:] != units[:, :-1]).mean()).__round__(3), flush=True)

def stage_kmoracle():
    """Causal test of boundary alignment. For each synthetic sequence (labels known by construction):
       oracle  : merge AVES frames within true label runs (event segments), one unit per segment
       randseg : same segment-length multiset, randomly permuted and offset -> same granularity, no alignment
       Units = nearest kmaves centroid to the segment-mean feature. Test recordings are never touched."""
    lab = Lab(); aves = load_aves(); cent = torch.tensor(np.load(DATA / 'centroids_kmaves.npy')).cuda()
    bs = 64; T = None; U_or, U_rs = [], []; stats_align = []
    rng = np.random.RandomState(123)
    for i in tqdm(range(0, N_SEQ, bs)):
        ids = list(range(i, min(i + bs, N_SEQ)))
        waves, labs = zip(*[synth(lab, j, with_labels=True) for j in ids])
        w = torch.tensor(np.stack(waves)).cuda()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            hs, _ = aves.extract_features(w, num_layers=AVES_LAYER)
        f = hs[-1].float(); T = f.shape[1]
        for k, y in enumerate(labs):
            y = y[:T]; b = np.r_[True, y[1:] != y[:-1]]                         # true boundaries
            seg = np.cumsum(b) - 1; lens = np.bincount(seg)
            # random control: permute segment lengths, random circular offset
            rl = rng.permutation(lens); starts = np.r_[0, np.cumsum(rl)[:-1]]; off = rng.randint(0, T)
            b2 = np.zeros(T, bool); b2[(starts + off) % T] = True; b2[0] = True; seg2 = np.cumsum(b2) - 1
            for S, store in ((seg, U_or), (seg2, U_rs)):
                S_t = torch.tensor(S, device=f.device); n = int(S.max()) + 1
                means = torch.zeros(n, f.shape[-1], device=f.device).index_add_(0, S_t, f[k]) / torch.bincount(S_t, minlength=n).float()[:, None]
                u = torch.cdist(means, cent).argmin(1)[S_t].cpu().numpy().astype(np.int16); store.append(u)
            near = b | np.roll(b, 1) | np.roll(b, -1); stats_align.append((b2[near].mean(), b2[~near].mean(), b.mean()))
    np.save(DATA / 'targets_kmoracle.npy', np.stack(U_or)); np.save(DATA / 'targets_kmrandseg.npy', np.stack(U_rs))
    a = np.array(stats_align); print('oracle/randseg done; boundary rate', a[:, 2].mean().round(3),
          'randseg P(start|near true boundary)', a[:, 0].mean().round(3), 'P(start|inside)', a[:, 1].mean().round(3), flush=True)

ITER2_SRC = 'ckpt/kmaves_20000.pt'; ITER2_LAYER = 6     # HuBERT iteration 2: k-means on the student's own intermediate layer

def stage_kmiter2():
    """Second HuBERT iteration: cluster layer-6 features of the iteration-1 student (km(AVES), vanilla) -> new targets."""
    from common import Encoder
    lab = Lab(); stats = [s.cuda() for s in load_stats()]
    ck = torch.load(ROOT / ITER2_SRC, map_location='cpu'); enc = Encoder().cuda().eval(); enc.load_state_dict(ck['enc'])
    @torch.no_grad()
    def feat(ids):
        w = torch.tensor(np.stack([synth(lab, j) for j in ids])).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, hs = enc(tokens_from_mel(melspec(w), stats), None, return_layers=True)
        return hs[ITER2_LAYER].float().cpu().numpy()
    T = feat([0]).shape[1]; print('iter2 source', ITER2_SRC, 'layer', ITER2_LAYER, 'tokens', T, flush=True)
    _fit_assign(feat, 'kmiter2', (T, enc.d))

if __name__ == '__main__':
    for st in sys.argv[1:]:
        t = time.time(); globals()['stage_' + st](); print(st, 'took', round(time.time() - t), 's', flush=True)
