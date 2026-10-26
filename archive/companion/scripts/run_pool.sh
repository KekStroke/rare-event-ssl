#!/bin/bash
# fixed comparison pool on the modern block, 3 seeds: HuBERT-km(AVES), data2vec, BEST-RQ, TS-JEPA-style; supervised = ft_scratch_modern
cd ~/tsssl; P=venv/bin/python; L=logs
until grep -q ORACLE_DONE logs/run_oracle.log; do sleep 60; done
for s in 0 1 2; do
  $P pretrain.py --variant jepa --block modern --seed $s --tag _modern_seed$s > $L/pretrain_jepa_modern_seed$s.log 2>&1 && echo PRE_jepa_$s
  $P probe.py --feats ckpt/jepa_modern_seed${s}_20000.pt --name jepa_modern_seed$s > $L/probe_jepa_modern_seed$s.log 2>&1; echo PROBE_jepa_$s
  $P finetune.py --feats ckpt/jepa_modern_seed${s}_20000.pt --name ft_jepa_modern_seed$s > $L/ft_jepa_modern_seed$s.log 2>&1; echo FT_jepa_$s
done
for s in 0 1 2; do
  $P pretrain.py --variant rq --block modern --seed $s --tag _modern_seed$s > $L/pretrain_rq_modern_seed$s.log 2>&1 && echo PRE_rq_$s
  $P probe.py --feats ckpt/rq_modern_seed${s}_20000.pt --name rq_modern_seed$s > $L/probe_rq_modern_seed$s.log 2>&1; echo PROBE_rq_$s
  $P finetune.py --feats ckpt/rq_modern_seed${s}_20000.pt --name ft_rq_modern_seed$s > $L/ft_rq_modern_seed$s.log 2>&1; echo FT_rq_$s
done
for s in 1 2; do $P finetune.py --feats ckpt/kmaves_modern_seed${s}_20000.pt --name ft_kmaves_modern_seed$s > $L/ft_kmaves_modern_seed$s.log 2>&1; echo FT_kmaves_modern_$s; done
$P finetune.py --feats ckpt/kmaves_modern_20000.pt --name ft_kmaves_modern > $L/ft_kmaves_modern.log 2>&1; echo FT_kmaves_modern_0
for s in 1 2; do $P finetune.py --feats scratch --block modern --seed $s --name ft_scratch_modern_seed$s > $L/ft_scratch_modern_seed$s.log 2>&1; echo FT_scratch_modern_$s; done
echo POOL_DONE
