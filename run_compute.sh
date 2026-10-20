#!/bin/bash
# compute-matched stride check: 10 ms tokens with half the steps (same token-updates as 20 ms / 20k)
cd ~/tsssl; P=venv/bin/python; L=logs
until grep -q SEED_DONE logs/run_seed.log; do sleep 60; done
TSSSL_STACK=1 $P pretrain.py --variant rq --steps 10000 --tag _s1_half > $L/pretrain_rq_s1_half.log 2>&1 && echo PRETRAIN_OK_rq_s1_half
TSSSL_STACK=1 $P probe.py --feats ckpt/rq_s1_half_10000.pt --name rq_s1_half > $L/probe_rq_s1_half.log 2>&1; echo PROBE_rq_s1_half
echo COMPUTE_DONE
