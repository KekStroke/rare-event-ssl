#!/bin/bash
# control: same JEPA (predictor, EMA target encoder) but HuBERT-style short spans (p .08, span 10) -> tests the block-mask explanation
cd ~/tsssl; P=venv/bin/python; L=logs
until grep -q JEPA_DONE logs/run_jepa.log; do sleep 60; done
$P pretrain.py --variant jepa --block modern --seed 0 --jepa_p 0.08 --jepa_span 10 --tag _modern_short_seed0 > $L/pretrain_jepa_modern_short_seed0.log 2>&1 && echo PRE_jepa_short || echo PRE_jepa_short_FAIL
$P probe.py --feats ckpt/jepa_modern_short_seed0_20000.pt --name jepa_modern_short_seed0 > $L/probe_jepa_modern_short_seed0.log 2>&1; echo PROBE_jepa_short
$P finetune.py --feats ckpt/jepa_modern_short_seed0_20000.pt --name ft_jepa_modern_short_seed0 > $L/ft_jepa_modern_short_seed0.log 2>&1; echo FT_jepa_short
echo JEPA2_DONE
