#!/bin/bash
# approved decoder experiment: FT with auxiliary onset head (tracks saved) for the student (3 seeds) and the teacher (3 seeds)
cd ~/tsssl; P=venv/bin/python; L=logs
until grep -q RECALLSEEDS_DONE logs/run_recall_seeds.log; do sleep 60; done
for s in 0 1 2; do
  ck=ckpt/d2v_modern_20000.pt; [ $s -gt 0 ] && ck=ckpt/d2v_modern_seed${s}_20000.pt
  $P finetune.py --feats $ck --onset_head --seed $s --name fto_d2v_modern_seed$s > $L/fto_d2v_modern_seed$s.log 2>&1; echo O_d2v_$s
done
for s in 0 1 2; do $P finetune.py --feats aves --onset_head --seed $s --name fto_aves_seed$s > $L/fto_aves_seed$s.log 2>&1; echo O_aves_$s; done
echo ONSET_DONE
for n in fto_d2v_modern_seed0 fto_d2v_modern_seed1 fto_d2v_modern_seed2 fto_aves_seed0 fto_aves_seed1 fto_aves_seed2; do $P decode_variants.py $n 100 >> $L/decode_variants.log 2>&1; done
echo VARIANTS_DONE
