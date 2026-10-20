#!/bin/bash
# One arm of the OZF protocol (ov | noov), seeds 0-2; the two arms run concurrently (data synthesis is CPU-bound, 24 cores).
arm=$1; cd ~/tsssl; P=~/tsssl/venv/bin/python; L=logs; export TSSSL_DATASET=ozf TSSSL_PAIRED=1
if [ $arm = noov ]; then export TSSSL_NOOVERLAP=1; else export TSSSL_NOOVERLAP=0; fi
for s in 0 1 2; do
  T=_ozf_${arm}_seed$s
  while pgrep -f "tag $T\$" > /dev/null; do sleep 30; done          # an earlier launch of this exact pretraining may still be running
  [ -f ckpt/kmaves${T}_20000.pt ] || $P pretrain.py --variant kmaves --block modern --event_head --no_hubert --seed $s --tag $T > $L/pretrain_kmaves$T.log 2>&1; echo PRE_$T
  $P finetune.py --feats ckpt/kmaves${T}_20000.pt --event_head --seed $s --name fte_ozf_${arm}_seed$s > $L/fte_ozf_${arm}_seed$s.log 2>&1; echo FT_$T
  $P decode_instances.py fte_ozf_${arm}_seed$s 400 >> $L/decode_instances_ozf_$arm.log 2>&1; echo DEC_$T
done
echo ARM_DONE_$arm
