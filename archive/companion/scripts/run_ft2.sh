#!/bin/bash
cd ~/tsssl; P=venv/bin/python; L=logs
for s in 1 2; do
  $P finetune.py --feats ckpt/kmaves_seed${s}_20000.pt --name ft_kmaves_seed$s > $L/ft_kmaves_seed$s.log 2>&1; echo FT_kmaves_seed$s
  $P finetune.py --feats ckpt/d2v_modern_seed${s}_20000.pt --name ft_d2v_modern_seed$s > $L/ft_d2v_modern_seed$s.log 2>&1; echo FT_d2v_modern_seed$s
done
$P finetune.py --feats ckpt/aves_dummy.pt --name _skip > /dev/null 2>&1 || true
echo FT2_DONE
