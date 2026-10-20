"""FINDINGS.md from expdb.sqlite + knowledge/hypotheses.json + results/units.json."""
import json, sqlite3
from pathlib import Path
R = Path.home() / 'tsssl'; c = sqlite3.connect(R / 'expdb' / 'expdb.sqlite')
fp = json.loads(c.execute("SELECT fp_json FROM runs ORDER BY id DESC LIMIT 1").fetchone()[0])
rows = c.execute("""SELECT id,name,valid,f1_cls_0p05,f1_cls_0p2,f1_cls_1p0,f1_any_0p2,f1_any_1p0,onset_p50,onset_p90,recall_op_recall,recall_op_precision,stack
                    FROM probe_view ORDER BY f1_cls_0p2 DESC""") if False else c.execute(
    """SELECT id,name,valid,f1_cls_0p05,f1_cls_0p2,f1_cls_1p0,f1_any_0p2,f1_any_1p0,onset_p50,onset_p90,recall_op_recall,recall_op_precision,stack
       FROM runs WHERE kind='probe' AND id IN (SELECT MAX(id) FROM runs WHERE kind='probe' GROUP BY name) ORDER BY f1_cls_0p2 DESC""").fetchall()
pre = c.execute("SELECT id,name,valid,block,stack,seed,steps,final_loss,final_acc,minutes,params_m,invalid_reason FROM runs WHERE kind='pretrain' ORDER BY id").fetchall()
H = json.load(open(R / 'expdb' / 'knowledge' / 'hypotheses.json'))
units = json.load(open(R / 'results' / 'units.json')) if (R / 'results' / 'units.json').exists() else {}
L = []
L.append(f"# Findings — TS-SSL pilot, Avacha killer-whale events ({fp.get('gpu')}, torch {fp.get('torch')}, {fp.get('python')}, WSL2)\n")
L.append("Protocol: 16 kHz, 80-mel/10 ms, 2-frame tokens (20 ms unless stated), 8-layer d=384 encoder (14.8M vanilla / 17.8M Gemma-style), 20k steps x 32 x 5 s synthesized mixtures from 45,429 lab clips; "
         "frozen weighted-layer linear probe, 3 recording-held-out folds (6/6/5), thresholds on 2 dev recordings, event F1 with onset collar tau, offset tolerance max(tau, 0.5 dur). **One seed per row** unless stated; fold spread is ~±0.05 F1.\n")
L.append("## 1. Probe results (pooled over the 3 test folds)\n")
L.append("Rows are the latest record per model name (re-probes replace earlier ones); P(A>B) in §3b is the share of positive bootstrap differences, not a p-value.\n")
L.append("| id | model | F1 cls τ=.05 | τ=.2 | τ=1 (plateau D) | F1 any τ=.2 | τ=1 | onset p50 / p90 ms | screening R / P | tok ms | valid |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|")
f = lambda v: '-' if v is None else f"{v:.3f}"
for r in rows:
    L.append(f"| {r[0]} | {r[1]} | {f(r[3])} | {f(r[4])} | {f(r[5])} | {f(r[6])} | {f(r[7])} | {'-' if r[8] is None else int(r[8]*1000)} / {'-' if r[9] is None else int(r[9]*1000)} | {f(r[10])} / {f(r[11])} | {'-' if r[12] is None else r[12]*10} | {'✓' if r[2] else '✗'} |")
import re, statistics as st
fam = {}
for r in rows:
    base = re.sub(r'_seed\d+$', '', r[1]); fam.setdefault(base, []).append(r)
L.append("\n### 1b. Seed statistics (mean ± sd over seeds, families with ≥2 seeds)\n")
L.append("| family | n | F1 cls τ=.2 | plateau τ=1 | F1 any τ=.2 | onset p50 ms | onset p90 ms |"); L.append("|---|---|---|---|---|---|---|")
for b, rs in sorted(fam.items(), key=lambda kv: -st.mean(x[4] for x in kv[1])):
    if len(rs) < 2: continue
    g = lambda i, k=1: f"{st.mean(x[i]*k for x in rs):.3f} ± {st.stdev(x[i]*k for x in rs):.3f}" if k == 1 else f"{st.mean(x[i]*k for x in rs):.0f} ± {st.stdev(x[i]*k for x in rs):.0f}"
    L.append(f"| {b} | {len(rs)} | {g(4)} | {g(5)} | {g(6)} | {g(8,1000)} | {g(9,1000)} |")
L.append("\n## 2. Pretraining runs\n")
L.append("| id | name | block | tok ms | seed | steps | final loss | unit acc | min | M params | valid |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|")
for r in pre:
    L.append(f"| {r[0]} | {r[1]} | {r[3]} | {r[4]*10} | {r[5]} | {r[6]} | {f(r[7])} | {f(r[8])} | {r[9]} | {r[10]} | {'✓' if r[2] else '✗ ' + str(r[11])} |")
L.append("\nMinutes were measured with other jobs sharing the GPU (manual probes, target prep); they are upper bounds, not benchmarks.\n")
L.append("## 3. Unit quality (analysis_units.py)\n")
for k, v in units.items():
    L.append(f"- `{k}`: {v}")
import glob as _g
bs = sorted(_g.glob(str(R / 'results' / 'bootstrap_*.json')))
if bs:
    L.append("\n## 3b. Paired bootstrap over recordings (2000 resamples of the 17 test recordings, seed-0 models)\n")
    L.append("| comparison | τ | F1 A [95% CI] | F1 B [95% CI] | A − B [95% CI] | P(A>B) |"); L.append("|---|---|---|---|---|---|")
    for b in bs:
        j = json.load(open(b)); names = Path(b).stem[len('bootstrap_'):].split('_vs_')
        if len(names) != 2: continue
        for tau in ['0.05', '0.2', '1.0']:
            a_, b_, d = j[f'{names[0]}@{tau}'], j[f'{names[1]}@{tau}'], j[f'diff@{tau}']
            L.append(f"| {names[0]} vs {names[1]} | {tau} | {a_[0]:.3f} [{a_[1]:.3f}, {a_[2]:.3f}] | {b_[0]:.3f} [{b_[1]:.3f}, {b_[2]:.3f}] | {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] | {d[3]:.3f} |")
rf = R / 'results' / 'recall_far.json'
if rf.exists():
    J = json.load(open(rf))
    L.append("\n## 3c. Main criterion: class-agnostic recall at fixed false-alarm budgets (thresholds from dev per fold; test pooled; τ=0.5)\n")
    L.append("| model | R @≤50/h (test FAR) | R @≤100/h | R @≤300/h | K12 @300 | K21 @300 |"); L.append("|---|---|---|---|---|---|")
    for n, r in J.items():
        pc = r.get('per_class_recall_at_300', {})
        L.append(f"| {n} | {r['50']['recall']:.3f} ({r['50']['far_per_h']:.0f}) | {r['100']['recall']:.3f} ({r['100']['far_per_h']:.0f}) | {r['300']['recall']:.3f} ({r['300']['far_per_h']:.0f}) | {pc.get('K12','-')} | {pc.get('K21','-')} |")
L.append("\n## 4. Hypotheses → verdicts\n")
for h in H:
    L.append(f"**{h['id']}** [{h['status']}] {h['statement']}  \n  contract: {h['contract']}; expected: {h['expect']}")
    for v in h['verdicts']:
        L.append(f"  - {v['date']} **{v['status']}** (runs {v['runs']}): {v['note']}")
    L.append('')
L.append("## 5. Open items\n- second seed for d2v / d2v_modern (queued)\n- compute-matched 10 ms run (queued)\n- bootstrap CIs over recordings\n- boundary-aligned segmentation (spectral flux / self-iteration) with an alignment gate before training\n- 32 kHz front-end for K12\n- FAR-budget threshold selection for the screening mode\n")
(R / 'expdb' / 'docs' / 'FINDINGS.md').write_text('\n'.join(L)); print('\n'.join(L[:12]))
