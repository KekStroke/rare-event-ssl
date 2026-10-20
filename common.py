"""Shared pieces for the HuBERT-vs-continuous-target pilot on Avacha killer-whale audio.

Everything runs at 16 kHz. Student input: 80-bin log-mel at 10 ms hop, stacked in
pairs -> 50 tokens/s (20 ms), a Transformer encoder, and a pretext head.
"""
import math, random, json, os
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio

ROOT = Path.home() / 'tsssl'
DATASET = os.environ.get('TSSSL_DATASET', 'whale')          # 'whale' (default, unchanged) | 'ozf' (Overlapping Zebra Finch, see TSSSL_OZF_protocol_ru.md)
DATA = ROOT / 'data' if DATASET == 'whale' else ROOT / 'data' / DATASET
SR = 16000
N_MELS, N_FFT, HOP = 80, 400, 160          # 25 ms window, 10 ms hop
STACK = int(os.environ.get('TSSSL_STACK', 2))   # mel frames per token: 1 -> 10 ms, 2 -> 20 ms, 4 -> 40 ms
NOOVERLAP = os.environ.get('TSSSL_NOOVERLAP', '0') == '1'   # synthetic mixtures without overlapping calls (ablation H24)
NOOV_GAP = 1600                                                # 0.1 s minimum gap between calls in that corpus
_LAST_DRAWS = []
PAIRED = os.environ.get('TSSSL_PAIRED', '0') == '1'           # OZF protocol: both arms use the identical call list (prefix that fits sequentially)
N_CALLS = (1, 4) if DATASET == 'whale' else (1, 10)           # calls per 5 s mixture: randint(lo, hi+1)
TAU_MATCH = 0.5 if DATASET == 'whale' else 0.1               # onset collar of the strict scorer (s, in the pipeline's time scale)
TOK_S = HOP * STACK / SR                   # 0.02 s
SEQ_S = 5.0
SEQ_N = int(SEQ_S * SR)
CLASSES = ['K1', 'K4', 'K5', 'K7', 'K10', 'K12', 'K13', 'K14', 'K17', 'K21', 'K27'] if DATASET == 'whale' else ['VOC']
NOISE = len(CLASSES)                       # index 11
FOLDS = [
    ['010816-1409_K1K5K7K10K17', '280716-1838_K5K7K17', '070716-1731_K1K12',
     '280616-1355_K1K5K12K17', '060816-1435_K27', '220616-1109_K7K17'],
    ['230716-1546_distantK5K17', '070816-1632_K10K13K21', '010816-1816_K4K10K12K13',
     '040816-1836_K13K27', '010816-1633_K5K7K17', '080816-1600_K4K7'],
    ['010816-1611_K5K7K10K17', '010816-1341_K4', '070816-1640_K1K10',
     '040816-1906_K13K27', '010816-1322_k1k13k14'],
]
DEV_PER_FOLD = 2   # recordings taken from the training folds for threshold selection
_SPLITS = None
if DATASET != 'whale' and (DATA / 'splits.json').exists():
    import json as _json; _SPLITS = _json.load(open(DATA / 'splits.json')); FOLDS = [_SPLITS['test']]

def splits():
    """List of (train_stems, dev_stems, test_stems). Whale: 3 recording-held-out folds, dev = first recording of each other fold.
    OZF: the official single split (train 46, val 6 as dev, test 13)."""
    if _SPLITS is not None: return [(list(_SPLITS['train']), list(_SPLITS['val']), list(_SPLITS['test']))]
    out = []
    for k in range(3):
        test = FOLDS[k]; others = [f for i, f in enumerate(FOLDS) if i != k]; devs = [others[0][0], others[1][0]]
        out.append(([s for f in others for s in f if s not in devs], devs, list(test)))
    return out

# ----------------------------------------------------------------------------- audio
_mel = {}
def melspec(wave: torch.Tensor) -> torch.Tensor:
    """wave [B, N] float -> log-mel [B, T, 80] (T = N // HOP)."""
    key = wave.device
    if key not in _mel:
        _mel[key] = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, f_min=20, f_max=8000,
            center=True, power=2.0).to(wave.device)
    m = _mel[key](wave)[..., : wave.shape[-1] // HOP]           # [B, 80, T]
    return torch.log(m + 1e-6).transpose(1, 2)

def tokens_from_mel(mel: torch.Tensor, stats) -> torch.Tensor:
    """[B, T, 80] -> normalised, stacked [B, T//STACK, 80*STACK]."""
    mu, sd = stats
    mel = (mel - mu) / sd
    B, T, D = mel.shape
    T = T // STACK * STACK
    return mel[:, :T].reshape(B, T // STACK, D * STACK)

def load_stats():
    s = np.load(DATA / 'mel_stats.npz')
    return torch.tensor(s['mu']), torch.tensor(s['sd'])

# ----------------------------------------------------------------------------- lab corpus / synthesis
class Lab:
    """All lab clips at 16 kHz in one float16 array with offsets and class ids."""
    def __init__(self):
        z = np.load(DATA / 'lab16k.npz')
        self.audio = z['audio']; self.off = z['off']; self.cls = z['cls']   # cls: -1 noise else CLASSES idx
        self.noise_ids = np.flatnonzero(self.cls < 0)
        self.call_ids = np.flatnonzero(self.cls >= 0)
    def clip(self, i):
        return self.audio[self.off[i]:self.off[i + 1]].astype(np.float32)

def _fade(c, n=320):
    n = min(n, len(c) // 2); r = np.linspace(0, 1, n, dtype=np.float32)
    c = c.copy(); c[:n] *= r; c[-n:] *= r[::-1]; return c

def synth(lab: Lab, idx: int, with_labels=False):
    """Deterministic 5 s mixture: cross-faded noise background + 1..4 faded calls (~40-60 % call time)."""
    rng = np.random.RandomState(idx)
    out = np.zeros(SEQ_N, np.float32)
    pos = 0; xf = 800                                      # 50 ms cross-fade between background clips
    bg_gain = rng.uniform(0.5, 1.0)
    while pos < SEQ_N:
        c = _fade(lab.clip(rng.choice(lab.noise_ids)), xf) * bg_gain
        n = min(len(c), SEQ_N - pos); out[pos:pos + n] += c[:n]; pos += max(n - xf, 1)
    lab_tok = np.full(SEQ_N // (HOP * STACK), NOISE, np.int64); events = []
    placed = []; rng2 = np.random.RandomState(idx + 10_000_019)      # NOOVERLAP ablation: separate stream so clip ids / gains stay those of the default corpus
    draws = []
    for _ in range(rng.randint(N_CALLS[0], N_CALLS[1] + 1)):
        i = rng.choice(lab.call_ids); c = _fade(lab.clip(i))
        if len(c) >= SEQ_N: continue
        s = rng.randint(0, SEQ_N - len(c)); g = rng.uniform(0.4, 1.6)
        draws.append((i, c, s, g))
    if PAIRED:                                                        # identical call list in both arms: the longest prefix that fits sequentially with NOOV_GAP
        total = 0; keep = 0
        for i, c, s, g in draws:
            if total + len(c) + (NOOV_GAP if keep else 0) > SEQ_N: break
            total += len(c) + (NOOV_GAP if keep else 0); keep += 1
        draws = draws[:keep]
    global _LAST_DRAWS; _LAST_DRAWS = [(int(i), float(g)) for i, c, s, g in draws]
    if PAIRED and NOOVERLAP and draws:                                # guaranteed-fit sequential placement: random order, random gaps >= NOOV_GAP summing to the slack
        order = rng2.permutation(len(draws)); slack = SEQ_N - sum(len(draws[j][1]) for j in order) - NOOV_GAP * (len(draws) - 1)
        cuts = np.sort(rng2.randint(0, slack + 1, size=len(draws))); pos = 0; prev = 0; new = []
        for j, cpt in zip(order, cuts):
            pos += (cpt - prev) + (NOOV_GAP if new else 0); prev = cpt
            i, c, _, g = draws[j]; new.append((i, c, pos, g)); pos += len(c)
        draws = new
    for i, c, s, g in draws:
        if NOOVERLAP and not PAIRED:                                  # sequential-style corpus (as Voxaboxen's synthetic data): calls never overlap, min gap NOOV_GAP
            free = []; cur = 0
            for a0, b0 in sorted(placed):
                if a0 - NOOV_GAP - cur >= len(c): free.append((cur, a0 - NOOV_GAP - len(c)))
                cur = max(cur, b0 + NOOV_GAP)
            if SEQ_N - cur >= len(c): free.append((cur, SEQ_N - len(c)))
            if not free: continue                                     # does not fit: the call is dropped
            lo, hi = free[rng2.randint(len(free))]; s = rng2.randint(lo, hi + 1); placed.append((s, s + len(c)))
        out[s:s + len(c)] += c * g
        a, b = s // (HOP * STACK), (s + len(c)) // (HOP * STACK)
        lab_tok[a:b + 1] = lab.cls[i]; events.append((a, b, int(lab.cls[i])))
    peak = np.abs(out).max() + 1e-6
    out = out / peak * rng.uniform(0.3, 0.95)
    if with_labels == 'events': return out, lab_tok, events
    return (out, lab_tok) if with_labels else out

def event_targets(events, n_tok):
    """Instance targets per token from the inserted-call list (overlaps allowed):
       presence [T] (any call covers token), onset [T] (+-1 token around each start), logdur [T] at onset tokens, dur_mask [T]."""
    pres = np.zeros(n_tok, np.float32); on = np.zeros(n_tok, np.float32); ld = np.zeros(n_tok, np.float32); dm = np.zeros(n_tok, np.float32)
    for a, b, c in events:
        a2, b2 = max(0, a), min(n_tok - 1, b); pres[a2:b2 + 1] = 1
        if 0 <= a < n_tok:
            on[max(0, a - 1): a + 2] = 1; ld[a] = np.log(max(1, b - a + 1)); dm[a] = 1
    return pres, on, ld, dm

class SynthDataset(torch.utils.data.Dataset):
    def __init__(self, n, targets=None, events=False):
        self.lab = Lab(); self.n = n; self.targets = targets; self.events = events
    def __len__(self): return self.n
    def __getitem__(self, i):
        if self.events:
            w, _, ev = synth(self.lab, i, with_labels='events'); n_tok = SEQ_N // (HOP * STACK)
            et = torch.from_numpy(np.stack(event_targets(ev, n_tok), 1))          # [T, 4]
            t = torch.from_numpy(self.targets[i].astype(np.int64)) if self.targets is not None else torch.zeros(n_tok, dtype=torch.long)
            return torch.from_numpy(w), t, et
        w = torch.from_numpy(synth(self.lab, i))
        if self.targets is None: return w, torch.zeros(1)
        return w, torch.from_numpy(self.targets[i].astype(np.int64))

# ----------------------------------------------------------------------------- model
class ConvPos(nn.Module):
    def __init__(self, d, k=63, groups=16):
        super().__init__(); self.c = nn.Conv1d(d, d, k, padding=k // 2, groups=groups)
    def forward(self, x):  # [B, T, d]
        return x + F.gelu(self.c(x.transpose(1, 2))).transpose(1, 2)

class Encoder(nn.Module):
    def __init__(self, d_in=N_MELS * STACK, d=384, layers=8, heads=6, ff=1536, drop=0.1):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, d), nn.LayerNorm(d), nn.Dropout(drop))
        self.mask_emb = nn.Parameter(torch.randn(d) * 0.02)
        self.pos = ConvPos(d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, drop, activation='gelu', batch_first=True, norm_first=True)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d, heads, ff, drop, activation='gelu',
                                                                batch_first=True, norm_first=True) for _ in range(layers)])
        self.norm = nn.LayerNorm(d); self.d = d
    def forward(self, tok, mask=None, return_layers=False):
        x = self.inp(tok)
        if mask is not None: x = torch.where(mask[..., None], self.mask_emb.to(x.dtype), x)
        x = self.pos(x); hs = [x]
        for l in self.layers:
            x = l(x); hs.append(x)
        x = self.norm(x); hs[-1] = x
        return (x, hs) if return_layers else x

def span_mask(B, T, p=0.08, span=10, device='cpu'):
    """HuBERT-style: each token is a span start with prob p; spans of `span` tokens."""
    starts = torch.rand(B, T, device=device) < p
    m = starts.clone()
    for s in range(1, span):
        m[:, s:] |= starts[:, :-s]
    return m

# ----------------------------------------------------------------------------- field data / labels
def read_raven(path):
    rows = []
    with open(path, encoding='utf-8', errors='ignore') as f:
        hdr = f.readline().rstrip('\r\n').split('\t')
        ib, ie, ia = hdr.index('Begin Time (s)'), hdr.index('End Time (s)'), hdr.index('Annotation')
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            if len(p) <= ia or not p[ia].strip(): continue
            lab = p[ia].strip().upper()
            if lab in CLASSES: rows.append((float(p[ib]), float(p[ie]), lab))
    return rows

def field_stems():
    return sorted(p.stem for p in (DATA / 'field16k').glob('*.npy'))

def ann_for(stem):
    cands = {p.stem.lower(): p for p in (DATA / 'ann').glob('*.txt')}
    return read_raven(cands[stem.lower()])

def frame_targets(events, n_tok):
    y = np.full(n_tok, NOISE, np.int64)
    for s, e, lab in events:
        a, b = int(s / TOK_S), int(math.ceil(e / TOK_S))
        y[a:min(b, n_tok)] = CLASSES.index(lab)
    return y

# ----------------------------------------------------------------------------- decoding + scoring
def median1d(x, k):
    if k <= 1: return x
    from scipy.ndimage import median_filter
    return median_filter(x, size=k, mode='nearest')

def decode(probs, detect=0.7, edge=0.3, med=5, min_dur=0.15):
    """probs [T, 12] softmax -> list of (start, end, cls). Two-threshold on p_call = 1 - p_noise."""
    pc = median1d(1.0 - probs[:, NOISE], med)
    T = len(pc); ev = []; i = 0
    while i < T:
        if pc[i] > detect:
            a = i
            while a > 0 and pc[a - 1] > edge: a -= 1
            b = i
            while b + 1 < T and pc[b + 1] > edge: b += 1
            seg = probs[a:b + 1, :NOISE].sum(0); cls = CLASSES[int(seg.argmax())]
            if (b + 1 - a) * TOK_S >= min_dur: ev.append((a * TOK_S, (b + 1) * TOK_S, cls))
            i = b + 1
        else: i += 1
    return ev

def match(preds, refs, tau, class_aware=True):
    """Greedy onset-nearest matching. Offset tolerance max(tau, 0.5*ref_dur). Returns TP."""
    used = np.zeros(len(preds), bool); tp = 0
    for rs, re_, rl in refs:
        best, bd = -1, 1e9
        for j, (ps, pe, pl) in enumerate(preds):
            if used[j] or (class_aware and pl != rl): continue
            if abs(ps - rs) <= tau and abs(pe - re_) <= max(tau, 0.5 * (re_ - rs)):
                d = abs(ps - rs)
                if d < bd: best, bd = j, d
        if best >= 0: used[best] = True; tp += 1
    return tp

def prf(tp, npred, nref):
    p = tp / max(npred, 1); r = tp / max(nref, 1)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)

TAUS = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]

def score(recordings, taus=TAUS):
    """recordings: list of (pred_events, ref_events). Returns {tau: (P, R, F1)} class-aware and agnostic."""
    out = {}
    for ca in (True, False):
        for tau in taus:
            tp = sum(match(p, r, tau, ca) for p, r in recordings)
            npred = sum(len(p) for p, _ in recordings); nref = sum(len(r) for _, r in recordings)
            out[('cls' if ca else 'any', tau)] = prf(tp, npred, nref)
    return out

# ----------------------------------------------------------------------------- modern (Gemma-3 / Conformer style) block
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        xf = x.float(); return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.w.float()).to(x.dtype)

def rope(x, base=10000.0):
    """x [B, H, T, Dh] -> rotary-embedded."""
    B, H, T, Dh = x.shape
    half = Dh // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=x.device).float() / half))
    ang = torch.arange(T, device=x.device).float()[:, None] * freqs[None]        # [T, half]
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).to(x.dtype)

class GemmaAttn(nn.Module):
    """Multi-head attention with RoPE, QK-norm, optional sliding window (local layers)."""
    def __init__(self, d, heads, window=None):
        super().__init__(); self.h = heads; self.dh = d // heads; self.window = window
        self.qkv = nn.Linear(d, 3 * d, bias=False); self.o = nn.Linear(d, d, bias=False)
        self.qn = RMSNorm(self.dh); self.kn = RMSNorm(self.dh)
    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k = rope(self.qn(q)), rope(self.kn(k))
        mask = None
        if self.window is not None and self.window < T:
            i = torch.arange(T, device=x.device); mask = (i[:, None] - i[None, :]).abs() <= self.window
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(y.transpose(1, 2).reshape(B, T, -1))

class GatedConv(nn.Module):
    """Conformer convolution module: pointwise -> GLU gate -> depthwise conv -> SiLU -> pointwise."""
    def __init__(self, d, k=15):
        super().__init__(); self.n = RMSNorm(d); self.pw1 = nn.Linear(d, 2 * d, bias=False)
        self.dw = nn.Conv1d(d, d, k, padding=k // 2, groups=d, bias=False); self.pw2 = nn.Linear(d, d, bias=False)
    def forward(self, x):
        y = F.glu(self.pw1(self.n(x)), dim=-1)
        y = F.silu(self.dw(y.transpose(1, 2)).transpose(1, 2))
        return x + self.pw2(y)

class GemmaBlock(nn.Module):
    def __init__(self, d, heads, ff, window, drop):
        super().__init__()
        self.n1 = RMSNorm(d); self.attn = GemmaAttn(d, heads, window)
        self.conv = GatedConv(d)
        self.n2 = RMSNorm(d); self.gate = nn.Linear(d, ff, bias=False); self.up = nn.Linear(d, ff, bias=False)
        self.down = nn.Linear(ff, d, bias=False); self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = x + self.drop(self.attn(self.n1(x)))
        x = self.conv(x)
        h = self.n2(x); x = x + self.drop(self.down(F.gelu(self.gate(h), approximate='tanh') * self.up(h)))  # GeGLU
        return x

class ModernEncoder(nn.Module):
    """Gemma-3-style stack: RMSNorm, GeGLU, RoPE, QK-norm, local(window)/global attention 3:1,
    plus a Conformer gated-conv module per layer. Same interface as Encoder."""
    def __init__(self, d_in=N_MELS * STACK, d=384, layers=8, heads=6, ff=1024, drop=0.1, window=64):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, d), RMSNorm(d), nn.Dropout(drop))
        self.mask_emb = nn.Parameter(torch.randn(d) * 0.02)
        self.layers = nn.ModuleList([GemmaBlock(d, heads, ff, None if (i + 1) % 4 == 0 else window, drop)
                                     for i in range(layers)])
        self.norm = RMSNorm(d); self.d = d
    def forward(self, tok, mask=None, return_layers=False):
        x = self.inp(tok)
        if mask is not None: x = torch.where(mask[..., None], self.mask_emb.to(x.dtype), x)
        hs = [x]
        for l in self.layers:
            x = l(x); hs.append(x)
        x = self.norm(x); hs[-1] = x
        return (x, hs) if return_layers else x

def build_encoder(block='vanilla', **kw):
    return ModernEncoder(**kw) if block == 'modern' else Encoder(**kw)
