"""Why 0.873-0.801=0.072 (seed means of decode_instances far_matched) vs +0.065 (bootstrap mean of dR@100)?
Computes, with the bootstrap script's machinery, four aggregations of R@FAR100 per regime:
 A per-run curve, fold-0 secondary params for all recordings (decode_instances convention), mean over seeds
 B per-run curve, fold-wise secondary params, mean over seeds
 C pooled curve over seeds (ratio of sums), fold-wise params, full data  (bootstrap point estimate)
 D bootstrap mean over defined samples (reported), and the fraction of samples where R@100 is undefined
"""
import sys; sys.argv = ['x', '100', '400']
import numpy as np, json
import bootstrap_instances as bi
C = {r: [bi.run_counts(n)['instance'] for n in names] for r, names in bi.REG.items()}
full = np.arange(len(bi.RECS)); rng = np.random.default_rng(0); k = len(bi.RECS); IDX = [rng.integers(0, k, k) for _ in range(2000)]
out = {}
for r in ['R1', 'R2', 'R3']:
    B = np.mean([bi.at_far(*bi.pooled([g], full)[3:], 100) for g in C[r]])
    _, _, _, fs, rs = bi.pooled(C[r], full); Cc = bi.at_far(fs, rs, 100)
    A = np.mean([json.load(open(f'results/decode_instances_fte_{r}_seed{s}_b100.json'))['instance']['far_matched']['100'] for s in range(3)])
    boots = np.array([bi.at_far(*bi.pooled(C[r], ix)[3:], 100) for ix in IDX]); D = np.nanmean(boots); und = np.isnan(boots).mean()
    # curve support on full data
    out[r] = dict(A=A, B=B, C=Cc, D=D, undef=und, far_min=float(fs.min()), far_max=float(fs.max()))
    print(f"{r}: A(seed-mean, fold0 params) {A:.3f}  B(seed-mean, fold-wise) {B:.3f}  C(pooled, full data) {Cc:.3f}  D(bootstrap mean) {D:.3f} undefined {und:.2f}  curve FAR range [{fs.min():.0f},{fs.max():.0f}]")
for a, b in [('R2', 'R1'), ('R3', 'R1'), ('R3', 'R2')]:
    print(f"  {a}-{b}: A {out[a]['A']-out[b]['A']:+.3f}  B {out[a]['B']-out[b]['B']:+.3f}  C {out[a]['C']-out[b]['C']:+.3f}  D(mean of diffs over defined samples, from json) see bootstrap_instances_b100.json")
# paired bootstrap restricted to samples where BOTH are defined vs the full-data estimate C
for a, b in [('R2', 'R1'), ('R3', 'R1'), ('R3', 'R2')]:
    da = np.array([bi.at_far(*bi.pooled(C[a], ix)[3:], 100) for ix in IDX]); db = np.array([bi.at_far(*bi.pooled(C[b], ix)[3:], 100) for ix in IDX])
    ok = ~np.isnan(da) & ~np.isnan(db); d = (da - db)[ok]
    print(f"  {a}-{b}: full-data C {out[a]['C']-out[b]['C']:+.3f}; bootstrap mean {d.mean():+.3f} [{np.percentile(d,2.5):+.3f},{np.percentile(d,97.5):+.3f}] on {ok.mean():.2f} of samples; "
          f"mean FAR-op in defined vs undefined samples: {np.mean([bi.pooled(C[a], ix)[1] for ix in np.array(IDX)[ok][:200]]):.0f} vs {np.mean([bi.pooled(C[a], ix)[1] for ix in np.array(IDX)[~ok][:200]]) if (~ok).sum() else float('nan'):.0f}")
