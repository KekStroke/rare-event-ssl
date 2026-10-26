#!/bin/bash
# Prototype decisive test (H21): same instance head + same FT; pretraining regimes
#   R1 HuBERT units only (existing kmaves_modern seeds 0-2)
#   R2 HuBERT units + event-structure branch on synthetic mixtures (extra supervision from construction)
#   R3 event-structure branch only (control, no unit loss)
cd ~/tsssl; P=venv/bin/python; L=logs
for s in 0 1 2; do
  $P pretrain.py --variant kmaves --block modern --event_head --seed $s --tag _modern_ev_seed$s > $L/pretrain_kmaves_modern_ev_seed$s.log 2>&1 && echo PRE_R2_$s || echo PRE_R2_FAIL_$s
  $P pretrain.py --variant kmaves --block modern --event_head --no_hubert --seed $s --tag _modern_evonly_seed$s > $L/pretrain_kmaves_modern_evonly_seed$s.log 2>&1 && echo PRE_R3_$s || echo PRE_R3_FAIL_$s
done
for s in 0 1 2; do
  ck1=ckpt/kmaves_modern_20000.pt; [ $s -gt 0 ] && ck1=ckpt/kmaves_modern_seed${s}_20000.pt
  $P finetune.py --feats $ck1 --event_head --seed $s --name fte_R1_seed$s > $L/fte_R1_seed$s.log 2>&1; echo FT_R1_$s
  $P finetune.py --feats ckpt/kmaves_modern_ev_seed${s}_20000.pt --event_head --seed $s --name fte_R2_seed$s > $L/fte_R2_seed$s.log 2>&1; echo FT_R2_$s
  $P finetune.py --feats ckpt/kmaves_modern_evonly_seed${s}_20000.pt --event_head --seed $s --name fte_R3_seed$s > $L/fte_R3_seed$s.log 2>&1; echo FT_R3_$s
done
for n in fte_R1_seed0 fte_R1_seed1 fte_R1_seed2 fte_R2_seed0 fte_R2_seed1 fte_R2_seed2 fte_R3_seed0 fte_R3_seed1 fte_R3_seed2; do $P decode_instances.py $n 100 >> $L/decode_instances.log 2>&1; done
echo PROTO_DONE
