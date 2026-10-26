#!/bin/bash
cd ~/tsssl; P=venv/bin/python; L=logs
$P probe.py --feats aves --name aves > $L/probe_aves.log 2>&1; echo A_aves
$P analysis_units.py > $L/analysis_units.log 2>&1; echo A_units
$P finetune.py --feats scratch --block modern --name ft_scratch_modern > $L/ft_scratch_modern.log 2>&1; echo A_ft_scratch_modern
$P finetune.py --feats scratch --block vanilla --name ft_scratch --seed 0 > $L/ft_scratch.log 2>&1; echo A_ft_scratch
echo AUDIT_DONE
