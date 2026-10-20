#!/bin/bash
cd ~/tsssl; P=venv/bin/python; L=logs
$P finetune.py --feats scratch --name ft_scratch > $L/ft_scratch.log 2>&1; echo FT_scratch
$P finetune.py --feats ckpt/kmaves_20000.pt --name ft_kmaves > $L/ft_kmaves.log 2>&1; echo FT_kmaves
$P finetune.py --feats ckpt/d2v_modern_20000.pt --name ft_d2v_modern > $L/ft_d2v_modern.log 2>&1; echo FT_d2v_modern
echo FT_DONE
