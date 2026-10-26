#!/bin/bash
cd ~/tsssl; P=venv/bin/python; L=logs
$P probe.py --feats mel --name mel > $L/probe_mel.log 2>&1; echo R_mel
$P probe.py --feats aves --name aves > $L/probe_aves.log 2>&1; echo R_aves
for v in kmmel rq d2v kmaves; do $P probe.py --feats ckpt/${v}_20000.pt --name $v > $L/probe_$v.log 2>&1; echo R_$v; done
echo RECALL_DONE
