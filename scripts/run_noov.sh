#!/bin/bash
# H24: does overlap in the synthetic construction matter? R3 (event branch only, modern) pretrained on the no-overlap corpus
# (TSSSL_NOOVERLAP=1: same clips/gains/call counts, calls placed without overlap, 0.1 s min gap, ~5% of calls dropped), same FT and decoder.
cd ~/tsssl; P=venv/bin/python; L=logs
for s in 0 1 2; do
  TSSSL_NOOVERLAP=1 $P pretrain.py --variant kmaves --block modern --event_head --no_hubert --seed $s --tag _modern_evonly_noov_seed$s > $L/pretrain_kmaves_modern_evonly_noov_seed$s.log 2>&1 && echo PRE_NOOV_$s || { echo PRE_NOOV_FAIL_$s; continue; }
  $P finetune.py --feats ckpt/kmaves_modern_evonly_noov_seed${s}_20000.pt --event_head --seed $s --name fte_R3noov_seed$s > $L/fte_R3noov_seed$s.log 2>&1; echo FT_NOOV_$s
  $P decode_instances.py fte_R3noov_seed$s 100 >> $L/decode_instances.log 2>&1
done
echo NOOV_DONE
