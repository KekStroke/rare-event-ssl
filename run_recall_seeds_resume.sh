#!/bin/bash
cd ~/tsssl; P=venv/bin/python; L=logs
$P finetune.py --feats aves --name ft_aves_seed1 --seed 1 > $L/ft_aves_seed1.log 2>&1; echo S_aves_1
$P recall_far.py ft_d2v_modern ft_d2v_modern_seed1 ft_d2v_modern_seed2 ft_kmaves ft_kmaves_seed1 ft_kmaves_seed2 ft_kmaves_modern ft_kmaves_modern_seed1 ft_kmaves_modern_seed2 ft_scratch_modern ft_scratch_modern_seed1 ft_scratch_modern_seed2 ft_aves ft_aves_seed1 ft_kmiter2_seed0 aves > $L/recall_far_seeds.log 2>&1
echo RECALLSEEDS_DONE
