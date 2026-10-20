#!/bin/bash
# robustness of the two headline models: 3 seeds each (seed 0 exists)
cd ~/tsssl; P=venv/bin/python; L=logs
for s in 1 2; do
  $P pretrain.py --variant kmaves --seed $s --tag _seed$s > $L/pretrain_kmaves_seed$s.log 2>&1 && echo PRETRAIN_OK_kmaves_seed$s
  $P probe.py --feats ckpt/kmaves_seed${s}_20000.pt --name kmaves_seed$s > $L/probe_kmaves_seed$s.log 2>&1; echo PROBE_kmaves_seed$s
done
$P pretrain.py --variant d2v --block modern --seed 2 --tag _modern_seed2 > $L/pretrain_d2v_modern_seed2.log 2>&1 && echo PRETRAIN_OK_d2v_modern_seed2
$P probe.py --feats ckpt/d2v_modern_seed2_20000.pt --name d2v_modern_seed2 > $L/probe_d2v_modern_seed2.log 2>&1; echo PROBE_d2v_modern_seed2
echo NIGHT2_DONE
