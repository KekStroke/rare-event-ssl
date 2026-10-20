#!/bin/bash
# re-run the main FT models (and frozen AVES probe) so call-probability tracks are saved, then recall-vs-FAR
cd ~/tsssl; P=venv/bin/python; L=logs
$P finetune.py --feats ckpt/d2v_modern_20000.pt --name ft_d2v_modern > $L/ft_d2v_modern.log 2>&1; echo R1
$P finetune.py --feats ckpt/kmaves_20000.pt --name ft_kmaves > $L/ft_kmaves.log 2>&1; echo R2
$P finetune.py --feats ckpt/kmaves_modern_20000.pt --name ft_kmaves_modern > $L/ft_kmaves_modern.log 2>&1; echo R3
$P finetune.py --feats scratch --block modern --name ft_scratch_modern > $L/ft_scratch_modern.log 2>&1; echo R4
$P finetune.py --feats ckpt/kmiter2_seed0_20000.pt --name ft_kmiter2_seed0 > $L/ft_kmiter2_seed0.log 2>&1; echo R5
$P probe.py --feats aves --name aves > $L/probe_aves.log 2>&1; echo R6
$P finetune.py --feats aves --name ft_aves > $L/ft_aves.log 2>&1; echo R7
$P recall_far.py ft_d2v_modern ft_kmaves ft_kmaves_modern ft_scratch_modern ft_kmiter2_seed0 aves ft_aves > $L/recall_far.log 2>&1; echo RECALLFAR_DONE
