#!/bin/bash
# causal test: oracle-boundary units vs random-segment units vs frame units (kmaves, 3 seeds exist), vanilla block, same compute
cd ~/tsssl; P=venv/bin/python; L=logs
until grep -q BM_PROPER_DONE logs/bm_proper.log; do sleep 60; done
$P prep.py kmoracle > $L/prep_kmoracle.log 2>&1 && echo PREP_oracle || { echo PREP_oracle_FAIL; exit 1; }
for v in kmoracle kmrandseg; do for s in 0 1; do
  $P pretrain.py --variant $v --seed $s --tag _seed$s > $L/pretrain_${v}_seed$s.log 2>&1 && echo PRE_${v}_$s
  $P probe.py --feats ckpt/${v}_seed${s}_20000.pt --name ${v}_seed$s > $L/probe_${v}_seed$s.log 2>&1; echo PROBE_${v}_$s
  $P finetune.py --feats ckpt/${v}_seed${s}_20000.pt --name ft_${v}_seed$s > $L/ft_${v}_seed$s.log 2>&1; echo FT_${v}_$s
done; done
echo ORACLE_DONE
