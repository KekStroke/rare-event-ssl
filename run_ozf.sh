#!/bin/bash
# OZF protocol (TSSSL_OZF_protocol_ru.md): R3 modern event-only, paired synthetic corpora with / without overlaps, 3 seeds, identical FT + decode.
cd ~/tsssl; P=~/tsssl/venv/bin/python; L=logs; export TSSSL_DATASET=ozf TSSSL_PAIRED=1
for s in 0 1 2; do
  for arm in ov noov; do
    if [ $arm = noov ]; then export TSSSL_NOOVERLAP=1; else export TSSSL_NOOVERLAP=0; fi
    T=_ozf_${arm}_seed$s
    [ -f ckpt/kmaves${T}_20000.pt ] || $P pretrain.py --variant kmaves --block modern --event_head --no_hubert --seed $s --tag $T > $L/pretrain_kmaves$T.log 2>&1; echo PRE_$T
    $P finetune.py --feats ckpt/kmaves${T}_20000.pt --event_head --seed $s --name fte_ozf_${arm}_seed$s > $L/fte_ozf_${arm}_seed$s.log 2>&1; echo FT_$T
    $P decode_instances.py fte_ozf_${arm}_seed$s 400 >> $L/decode_instances_ozf.log 2>&1; echo DEC_$T
  done
done
echo OZF_DONE
