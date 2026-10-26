"""Unit-quality analysis (HuBERT-paper style) for the k-means target sets.
For each target set: cluster purity and PNMI against call-type labels, and the unit-switch rate at
event boundaries vs. elsewhere.  (a) on 2000 synthesized sequences with exact token labels,
(b) on the real field recordings: AVES layer-9 features -> nearest kmaves centroid, labels from Raven."""
import json, numpy as np, torch
from common import *
from prep import load_aves, AVES_LAYER

def purity_pnmi(units, labels, K=100):
    """labels: token classes (noise included). Purity = sum_k max_c n_kc / N ; PNMI = I(u;c)/H(c)."""
    C = int(labels.max()) + 1
    M = np.zeros((K, C)); np.add.at(M, (units, labels), 1)
    N = M.sum(); pu = M.max(1).sum() / N
    pk, pc = M.sum(1) / N, M.sum(0) / N; pkc = M / N
    with np.errstate(divide='ignore', invalid='ignore'):
        mi = np.nansum(pkc * np.log(pkc / (pk[:, None] * pc[None, :])))
    Hc = -np.sum(pc[pc > 0] * np.log(pc[pc > 0]))
    return float(pu), float(mi / Hc)

def switch_stats(units, labels):
    """P(unit switch) at label boundaries (±1 token) vs. inside segments."""
    sw = np.concatenate([[False], units[1:] != units[:-1]])
    bnd = np.concatenate([[False], labels[1:] != labels[:-1]])
    near = bnd | np.roll(bnd, 1) | np.roll(bnd, -1)
    return float(sw[near].mean()), float(sw[~near].mean()), float(sw.mean())

out = {}
lab = Lab()
for name in ['kmmel', 'kmaves']:
    T = np.load(DATA / f'targets_{name}.npy')
    U, L = [], []
    for i in range(2000):
        _, y = synth(lab, i, with_labels=True); n = min(len(y), T.shape[1])
        U.append(T[i, :n].astype(np.int64)); L.append(y[:n])
    U, L = np.concatenate(U), np.concatenate(L)
    pu, pn = purity_pnmi(U, L); s_b, s_in, s_all = switch_stats(U, L)
    out[f'{name}_synth'] = dict(purity=round(pu, 3), pnmi=round(pn, 3), switch_at_boundary=round(s_b, 3),
                                switch_inside=round(s_in, 3), switch_overall=round(s_all, 3), n_tokens=int(len(U)))
    print(name, 'synth', out[f'{name}_synth'], flush=True)

# real field recordings with kmaves units
aves = load_aves(); cent = torch.tensor(np.load(DATA / 'centroids_kmaves.npy')).cuda()
U, L = [], []
with torch.no_grad():
    for st in field_stems():
        w = np.load(DATA / 'field16k' / f'{st}.npy'); n = 10 * SR
        us = []
        for s in range(0, len(w), n):
            x = torch.tensor(w[s:s + n])[None].cuda()
            if x.shape[1] < SR // 2: break
            with torch.autocast('cuda', dtype=torch.bfloat16):
                hs, _ = aves.extract_features(x, num_layers=AVES_LAYER)
            u_c = torch.cdist(hs[-1][0].float(), cent).argmin(1).cpu().numpy(); n_exp = x.shape[1] // (HOP * STACK)
            u_c = np.r_[u_c, np.repeat(u_c[-1], max(0, n_exp - len(u_c)))][:n_exp]; us.append(u_c)
        u = np.concatenate(us); y = frame_targets(ann_for(st), len(u)); U.append(u); L.append(y)
U, L = np.concatenate(U), np.concatenate(L)
pu, pn = purity_pnmi(U, L); s_b, s_in, s_all = switch_stats(U, L)
out['kmaves_field'] = dict(purity=round(pu, 3), pnmi=round(pn, 3), switch_at_boundary=round(s_b, 3),
                           switch_inside=round(s_in, 3), switch_overall=round(s_all, 3), n_tokens=int(len(U)),
                           call_token_frac=round(float((L != NOISE).mean()), 3))
print('kmaves field', out['kmaves_field'])
# chance reference: purity if units ignore classes = majority-class share
out['majority_share_field'] = round(float(np.bincount(L).max() / len(L)), 3)
json.dump(out, open(ROOT / 'results' / 'units.json', 'w'), indent=1)
