#!/bin/bash
cd ~/tsssl; P=venv/bin/python; L=logs
TSSSL_STACK=1 $P pretrain.py --variant d2v --block modern --tag _modern_s1 > $L/pretrain_d2v_modern_s1.log 2>&1 && echo PRETRAIN_OK_d2v_modern_s1
TSSSL_STACK=1 $P probe.py --feats ckpt/d2v_modern_s1_20000.pt --name d2v_modern_s1 > $L/probe_d2v_modern_s1.log 2>&1; echo PROBE_d2v_modern_s1
TSSSL_STACK=1 $P finetune.py --feats ckpt/d2v_modern_s1_20000.pt --name ft_d2v_modern_s1 > $L/ft_d2v_modern_s1.log 2>&1; echo FT_d2v_modern_s1
echo H11_DONE
