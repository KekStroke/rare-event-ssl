#!/bin/bash
cd ~/tsssl; P=venv/bin/python; L=logs
$P pretrain.py --variant hybrid --block modern --seed 2 --tag _modern_seed2 > $L/pretrain_hybrid_modern_seed2.log 2>&1 && echo PRETRAIN_OK_hybrid_seed2
$P probe.py --feats ckpt/hybrid_modern_seed2_20000.pt --name hybrid_modern_seed2 > $L/probe_hybrid_modern_seed2.log 2>&1; echo PROBE_hybrid_seed2
echo NIGHT4_DONE
