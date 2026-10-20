#!/bin/bash
cd ~/tsssl; P=venv/bin/python; L=logs
until grep -q IMPROVE_DONE logs/run_improve.log; do sleep 60; done
$P pretrain.py --variant d2v --seed 1 --tag _seed1 > $L/pretrain_d2v_seed1.log 2>&1 && echo PRETRAIN_OK_d2v_seed1
$P probe.py --feats ckpt/d2v_seed1_20000.pt --name d2v_seed1 > $L/probe_d2v_seed1.log 2>&1; echo PROBE_d2v_seed1
$P pretrain.py --variant d2v --block modern --seed 1 --tag _modern_seed1 > $L/pretrain_d2v_modern_seed1.log 2>&1 && echo PRETRAIN_OK_d2v_modern_seed1
$P probe.py --feats ckpt/d2v_modern_seed1_20000.pt --name d2v_modern_seed1 > $L/probe_d2v_modern_seed1.log 2>&1; echo PROBE_d2v_modern_seed1
echo SEED_DONE
