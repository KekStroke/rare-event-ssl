"""OZF (Overlapping Zebra Finch, Zenodo 15507508) -> our data layout, under data/ozf. Fixed by TSSSL_OZF_protocol_ru.md before any training.
Time handling: slow-down by factor 2 via the sample-rate trick (Voxaboxen make_slowed_version.py): samples are kept, the declared rate is
halved (44100 -> 22050) and then resampled to 16 kHz; annotation times are multiplied by 2. All times below are in SLOWED seconds.
Outputs: field16k/<fn>.npy (65 recordings), ann/<fn>.txt (Raven, label VOC), lab16k.npz (calls + backgrounds from TRAIN recordings only),
mel_stats.npz, splits.json, synth_test/<overlap>/{field16k,ann} for the six synthetic sets (test split only), corpus_stats.json."""
import os, json, numpy as np, soundfile as sf, pandas as pd, torch
from pathlib import Path
from scipy.signal import resample_poly
os.environ['TSSSL_DATASET'] = 'ozf'
from common import ROOT, DATA, SR, melspec, N_MELS
RAW = ROOT / 'data' / 'ozf_raw' / 'OZF_unz' / 'OZF' / 'formatted'; RAWS = ROOT / 'data' / 'ozf_raw' / 'OZFS_unz' / 'OZF_synthetic'
SLOW = 2; BG_MIN = 1.0; BG_MARGIN = 0.05        # slowed seconds: minimum annotation-free segment used as background, margin from annotations
(DATA / 'field16k').mkdir(parents=True, exist_ok=True); (DATA / 'ann').mkdir(exist_ok=True)

def load_slowed(path):
    x, sr = sf.read(path, dtype='float32')
    if x.ndim > 1: x = x.mean(1)
    sr2 = sr // SLOW; from math import gcd; g = gcd(sr2, SR)
    return resample_poly(x, SR // g, sr2 // g).astype(np.float32)

def read_table(path):
    t = pd.read_csv(path, sep='\t')
    ev = sorted(zip(t['Begin Time (s)'] * SLOW, t['End Time (s)'] * SLOW))
    return [(float(a), float(b)) for a, b in ev]

def write_raven(path, ev):
    with open(path, 'w') as f:
        f.write('Selection\tBegin Time (s)\tEnd Time (s)\tAnnotation\n')
        for i, (a, b) in enumerate(ev): f.write(f'{i+1}\t{a:.6f}\t{b:.6f}\tVOC\n')

splits = {}
for sp in ['train', 'val', 'test']:
    d = pd.read_csv(RAW / f'{sp}_info.csv'); splits[sp] = list(d.fn)
    for _, r in d.iterrows():
        x = load_slowed(RAW / r.audio_fp); np.save(DATA / 'field16k' / f'{r.fn}.npy', x)
        write_raven(DATA / 'ann' / f'{r.fn}.txt', read_table(RAW / r.selection_table_fp))
json.dump(splits, open(DATA / 'splits.json', 'w'), indent=1); print({k: len(v) for k, v in splits.items()})

# lab corpus from TRAIN recordings only: calls = annotated intervals (exact bounds), backgrounds = annotation-free segments >= BG_MIN
audio, off, cls = [], [0], []; n_call = 0; bg_total = 0.0; durs = []
for fn in splits['train']:
    x = np.load(DATA / 'field16k' / f'{fn}.npy'); ev = read_table(RAW / f'selection_tables/{fn}.txt')
    for a, b in ev:
        c = x[int(a * SR): int(b * SR)]
        if len(c) < 2 * 320: continue
        audio.append(c.astype(np.float16)); off.append(off[-1] + len(c)); cls.append(0); n_call += 1; durs.append(b - a)
    cur = 0.0; segs = []
    for a, b in ev + [(len(x) / SR, len(x) / SR)]:
        if a - BG_MARGIN - cur >= BG_MIN: segs.append((cur, a - BG_MARGIN))
        cur = max(cur, b + BG_MARGIN)
    for a, b in segs:
        c = x[int(a * SR): int(b * SR)]; audio.append(c.astype(np.float16)); off.append(off[-1] + len(c)); cls.append(-1); bg_total += b - a
np.savez(DATA / 'lab16k.npz', audio=np.concatenate(audio), off=np.array(off), cls=np.array(cls))
stats = dict(n_calls=n_call, n_bg=int(sum(c < 0 for c in cls)), bg_total_s=round(bg_total, 1), call_dur_slowed_pct=dict(zip(['p1', 'p5', 'p50', 'p95', 'p99'], np.percentile(durs, [1, 5, 50, 95, 99]).round(3).tolist())),
             rec_len_slowed_s=120.0, call_rate_per_slowed_h=round(8504 / (65 * 120 / 3600), 0))
print(stats)

# six synthetic sets: TEST split only, same slow-down (inference-only use)
for ov in ['0', '0.2', '0.4', '0.6', '0.8', '1']:
    base = RAWS / f'overlap_{ov}'; d = pd.read_csv(base / 'test_info.csv'); out = DATA / 'synth_test' / f'overlap_{ov}'
    (out / 'field16k').mkdir(parents=True, exist_ok=True); (out / 'ann').mkdir(exist_ok=True)
    for _, r in d.iterrows():
        np.save(out / 'field16k' / f'{r.fn}.npy', load_slowed(base / r.audio_fp)); write_raven(out / 'ann' / f'{r.fn}.txt', read_table(base / r.selection_table_fp))
    print('synthetic', ov, len(d))

# mel stats from the default synthetic mixtures (as prep.stage_stats)
from common import Lab, synth
lab = Lab(); acc = []
for i in range(0, 2000, 100):
    w = torch.tensor(np.stack([synth(lab, j) for j in range(i, i + 100)])).cuda(); acc.append(melspec(w).reshape(-1, N_MELS).cpu())
m = torch.cat(acc); np.savez(DATA / 'mel_stats.npz', mu=m.mean(0).numpy(), sd=m.std(0).numpy() + 1e-5)
json.dump(stats, open(DATA / 'corpus_stats.json', 'w'), indent=1); print('done')
