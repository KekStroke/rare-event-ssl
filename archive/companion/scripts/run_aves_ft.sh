#!/bin/bash
cd ~/tsssl; P=venv/bin/python; L=logs
$P finetune.py --feats aves --name ft_aves > $L/ft_aves.log 2>&1; echo FT_aves
$P finetune.py --feats aves --name ft_aves_seed1 --seed 1 > $L/ft_aves_seed1.log 2>&1; echo FT_aves_seed1
echo AVESFT_DONE
