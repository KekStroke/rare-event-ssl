#!/bin/bash
# H22 repeats (seeds 1,2) after the Voxaboxen comparison frees the GPU
until grep -q "VOX_DONE" ~/tsssl/logs/run_vox_field.log; do sleep 120; done
cd ~/tsssl; P=venv/bin/python; L=logs
for s in 1 2; do
  $P pretrain.py --variant kmaves --block vanilla --event_head --no_hubert --seed $s --tag _vanilla_evonly_seed$s > $L/pretrain_kmaves_vanilla_evonly_seed$s.log 2>&1 && echo PRE_R3V_$s || { echo PRE_R3V_FAIL_$s; continue; }
  $P finetune.py --feats ckpt/kmaves_vanilla_evonly_seed${s}_20000.pt --block vanilla --event_head --seed $s --name fte_R3van_seed$s > $L/fte_R3van_seed$s.log 2>&1; echo FT_R3V_$s
  $P decode_instances.py fte_R3van_seed$s 100 >> $L/decode_instances.log 2>&1
done
echo R3VAN_SEEDS_DONE
