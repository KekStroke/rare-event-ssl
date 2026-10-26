"""Aggregate results/*.json into a table and an F1(tau) plot."""
import json, glob
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import ROOT, TAUS

ORDER = ['mel', 'scratch', 'aves', 'aves_8k', 'kmmel', 'rq', 'd2v', 'kmaves', 'kmaves_modern', 'd2v_modern', 'd2v_seed1', 'd2v_modern_seed1', 'rq_s1', 'rq_s4', 'kmseg', 'kmboth', 'kmseg_modern']
LABEL = {'mel': 'log-mel linear (no context)', 'scratch': 'random encoder', 'aves': 'AVES-bio frozen (94M)',
         'aves_8k': 'AVES-bio frozen, 8k-step probe', 'kmmel': 'HuBERT, k-means(mel) targets',
         'kmaves': 'HuBERT, k-means(AVES L9) targets', 'rq': 'BEST-RQ random-quantizer targets',
         'd2v': 'data2vec/JEPA-style continuous targets', 'kmaves_modern': 'HuBERT km(AVES) + Gemma-style block',
         'd2v_modern': 'data2vec + Gemma-style block', 'd2v_seed1': 'data2vec, seed 1', 'd2v_modern_seed1': 'data2vec + Gemma block, seed 1',
         'rq_s1': 'BEST-RQ, 10 ms tokens', 'rq_s4': 'BEST-RQ, 40 ms tokens', 'kmseg': 'HuBERT, segment units + boundary head',
         'kmboth': 'HuBERT, two codebooks (mel + AVES)', 'kmseg_modern': 'segment units + Gemma block', 'rq_s1_half': 'BEST-RQ, 10 ms tokens, 10k steps (compute-matched)', 'ft_scratch': 'full FT, random init', 'ft_kmaves': 'full FT, HuBERT km(AVES)', 'ft_d2v_modern': 'full FT, data2vec + Gemma block'}
R = {}
for f in glob.glob(str(ROOT / 'results' / '*.json')):
    j = json.load(open(f))
    if 'name' in j: R[j['name']] = j
names = [n for n in ORDER if n in R] + [n for n in R if n not in ORDER]
lines = ['| model | ' + ' | '.join(f'F1 cls τ={t}' for t in TAUS) + ' | F1 any τ=0.2 | F1 any τ=1.0 | P/R cls τ=0.2 |', '|' + '---|' * (len(TAUS) + 4)]
for n in names:
    p = R[n]['pooled']
    lines.append(f'| {LABEL.get(n, n)} | ' + ' | '.join(f"{p[f'cls@{t}'][2]:.3f}" for t in TAUS)
                 + f" | {p['any@0.2'][2]:.3f} | {p['any@1.0'][2]:.3f} | {p['cls@0.2'][0]:.2f}/{p['cls@0.2'][1]:.2f} |")
md = '\n'.join(lines); print(md)
(ROOT / 'results' / 'summary.md').write_text(md + '\n\nPer-fold (test F1 cls τ=0.2):\n' +
    '\n'.join(f"- {n}: " + ', '.join(f"F{d['fold']} {d['test_f1_cls_0p2']:.3f} thr={d['thr']}" for d in R[n]['per_fold']) for n in names))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for n in names:
    p = R[n]['pooled']
    ax[0].plot(TAUS, [p[f'cls@{t}'][2] for t in TAUS], marker='o', label=LABEL.get(n, n))
    ax[1].plot(TAUS, [p[f'any@{t}'][2] for t in TAUS], marker='o', label=LABEL.get(n, n))
for a, t in zip(ax, ['event F1, class-aware', 'event F1, class-agnostic']):
    a.set_xscale('log'); a.set_xlabel('onset collar τ, s'); a.set_title(t); a.grid(alpha=.3)
ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(ROOT / 'results' / 'f1_vs_tau.png', dpi=120)
print('saved', ROOT / 'results' / 'f1_vs_tau.png')
