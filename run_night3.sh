#!/bin/bash
# complete the 2x2 (block x target) with 3 seeds per cell, then the hybrid objective (H9)
cd ~/tsssl; P=venv/bin/python; L=logs
for s in 1 2; do
  $P pretrain.py --variant kmaves --block modern --seed $s --tag _modern_seed$s > $L/pretrain_kmaves_modern_seed$s.log 2>&1 && echo PRETRAIN_OK_kmaves_modern_seed$s
  $P probe.py --feats ckpt/kmaves_modern_seed${s}_20000.pt --name kmaves_modern_seed$s > $L/probe_kmaves_modern_seed$s.log 2>&1; echo PROBE_kmaves_modern_seed$s
done
$P pretrain.py --variant d2v --seed 2 --tag _seed2 > $L/pretrain_d2v_seed2.log 2>&1 && echo PRETRAIN_OK_d2v_seed2
$P probe.py --feats ckpt/d2v_seed2_20000.pt --name d2v_seed2 > $L/probe_d2v_seed2.log 2>&1; echo PROBE_d2v_seed2
for s in 0 1; do
  $P pretrain.py --variant hybrid --block modern --seed $s --tag _modern_seed$s > $L/pretrain_hybrid_modern_seed$s.log 2>&1 && echo PRETRAIN_OK_hybrid_seed$s || echo PRETRAIN_FAIL_hybrid_seed$s
  $P probe.py --feats ckpt/hybrid_modern_seed${s}_20000.pt --name hybrid_modern_seed$s > $L/probe_hybrid_modern_seed$s.log 2>&1; echo PROBE_hybrid_seed$s
done
echo NIGHT3_DONE
