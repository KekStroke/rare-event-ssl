#!/bin/bash
# approved scope (2026-10-18 evening): scratch-modern seeds, HuBERT iteration 2 (vanilla, 3 seeds), BEST-RQ on modern (3 seeds)
cd ~/tsssl; P=venv/bin/python; L=logs
for s in 1 2; do $P finetune.py --feats scratch --block modern --seed $s --name ft_scratch_modern_seed$s > $L/ft_scratch_modern_seed$s.log 2>&1; echo FT_scratch_modern_$s; done
$P prep.py kmiter2 > $L/prep_kmiter2.log 2>&1 && echo PREP_iter2 || { echo PREP_iter2_FAIL; exit 1; }
for s in 0 1 2; do
  $P pretrain.py --variant kmiter2 --seed $s --tag _seed$s > $L/pretrain_kmiter2_seed$s.log 2>&1 && echo PRE_iter2_$s
  $P probe.py --feats ckpt/kmiter2_seed${s}_20000.pt --name kmiter2_seed$s > $L/probe_kmiter2_seed$s.log 2>&1; echo PROBE_iter2_$s
  $P finetune.py --feats ckpt/kmiter2_seed${s}_20000.pt --name ft_kmiter2_seed$s > $L/ft_kmiter2_seed$s.log 2>&1; echo FT_iter2_$s
done
for s in 0 1 2; do
  $P pretrain.py --variant rq --block modern --seed $s --tag _modern_seed$s > $L/pretrain_rq_modern_seed$s.log 2>&1 && echo PRE_rq_$s
  $P probe.py --feats ckpt/rq_modern_seed${s}_20000.pt --name rq_modern_seed$s > $L/probe_rq_modern_seed$s.log 2>&1; echo PROBE_rq_$s
  $P finetune.py --feats ckpt/rq_modern_seed${s}_20000.pt --name ft_rq_modern_seed$s > $L/ft_rq_modern_seed$s.log 2>&1; echo FT_rq_$s
done
echo APPROVED_DONE
