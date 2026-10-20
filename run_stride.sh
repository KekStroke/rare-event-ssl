#!/bin/bash
# stride ablation: BEST-RQ targets (online, stride-agnostic) at 10 / 40 ms tokens; 20 ms is the main 'rq' run
cd ~/tsssl; P=venv/bin/python; L=logs
until grep -q ALL_DONE_2 logs/run_all.log; do sleep 60; done
for S in 1 4; do
  TSSSL_STACK=$S $P pretrain.py --variant rq --tag _s$S > $L/pretrain_rq_s$S.log 2>&1 && echo PRETRAIN_OK_rq_s$S
  TSSSL_STACK=$S $P probe.py --feats ckpt/rq_s${S}_20000.pt --name rq_s$S > $L/probe_rq_s$S.log 2>&1; echo PROBE_rq_s$S
done
echo STRIDE_DONE
