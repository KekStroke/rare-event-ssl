#!/bin/bash
cd ~/tsssl; P=venv/bin/python; L=logs
true
true
echo PREP_OK
for v in kmmel kmaves rq d2v; do
  $P pretrain.py --variant $v > $L/pretrain_$v.log 2>&1 && echo PRETRAIN_OK_$v || echo PRETRAIN_FAIL_$v
done
$P probe.py --feats mel --name mel > $L/probe_mel.log 2>&1; echo PROBE_mel
$P probe.py --feats scratch --name scratch > $L/probe_scratch.log 2>&1; echo PROBE_scratch
$P probe.py --feats aves --name aves > $L/probe_aves.log 2>&1; echo PROBE_aves
for v in kmmel kmaves rq d2v; do
  $P probe.py --feats ckpt/${v}_20000.pt --name $v > $L/probe_$v.log 2>&1; echo PROBE_$v
done
echo ALL_DONE
# --- appended: modern block ablation on the two most informative target types
cd ~/tsssl; P=venv/bin/python; L=logs
for v in kmaves d2v; do
  $P pretrain.py --variant $v --block modern --tag _modern > $L/pretrain_${v}_modern.log 2>&1 && echo PRETRAIN_OK_${v}_modern
  $P probe.py --feats ckpt/${v}_modern_20000.pt --name ${v}_modern > $L/probe_${v}_modern.log 2>&1; echo PROBE_${v}_modern
done
echo ALL_DONE_2
