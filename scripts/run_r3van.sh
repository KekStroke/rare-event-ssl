#!/bin/bash
# H22: does the modern block matter for the event-only detector? R3 on the vanilla block, seed 0 (screening), same FT and decoder.
cd ~/tsssl; P=venv/bin/python; L=logs; s=0
$P pretrain.py --variant kmaves --block vanilla --event_head --no_hubert --seed $s --tag _vanilla_evonly_seed$s > $L/pretrain_kmaves_vanilla_evonly_seed$s.log 2>&1 && echo PRE_R3V_$s || echo PRE_R3V_FAIL_$s
$P finetune.py --feats ckpt/kmaves_vanilla_evonly_seed${s}_20000.pt --block vanilla --event_head --seed $s --name fte_R3van_seed$s > $L/fte_R3van_seed$s.log 2>&1; echo FT_R3V_$s
$P decode_instances.py fte_R3van_seed$s 100 >> $L/decode_instances.log 2>&1
echo R3VAN_DONE
