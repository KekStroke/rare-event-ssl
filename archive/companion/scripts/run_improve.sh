#!/bin/bash
cd ~/tsssl; P=venv/bin/python; L=logs
until grep -q STRIDE_DONE logs/run_stride.log; do sleep 60; done
for v in kmseg kmboth; do
  $P pretrain.py --variant $v > $L/pretrain_$v.log 2>&1 && echo PRETRAIN_OK_$v || echo PRETRAIN_FAIL_$v
  $P probe.py --feats ckpt/${v}_20000.pt --name $v > $L/probe_$v.log 2>&1; echo PROBE_$v
done
$P pretrain.py --variant kmseg --block modern --tag _modern > $L/pretrain_kmseg_modern.log 2>&1 && echo PRETRAIN_OK_kmseg_modern
$P probe.py --feats ckpt/kmseg_modern_20000.pt --name kmseg_modern > $L/probe_kmseg_modern.log 2>&1; echo PROBE_kmseg_modern
echo IMPROVE_DONE
