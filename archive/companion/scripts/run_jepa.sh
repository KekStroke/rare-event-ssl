#!/bin/bash
# screening seed: JEPA (separate 2-layer predictor, block masks ~45%, context encoder with masked tokens zeroed, EMA target encoder, last-layer instance-normed targets)
cd ~/tsssl; P=venv/bin/python; L=logs
$P pretrain.py --variant jepa --block modern --seed 0 --tag _modern_seed0 > $L/pretrain_jepa_modern_seed0.log 2>&1 && echo PRE_jepa_0 || echo PRE_jepa_FAIL
$P probe.py --feats ckpt/jepa_modern_seed0_20000.pt --name jepa_modern_seed0 > $L/probe_jepa_modern_seed0.log 2>&1; echo PROBE_jepa_0
$P finetune.py --feats ckpt/jepa_modern_seed0_20000.pt --name ft_jepa_modern_seed0 > $L/ft_jepa_modern_seed0.log 2>&1; echo FT_jepa_0
echo JEPA_DONE
