#!/bin/bash
# seeds 1-2 for the recall-at-FAR table (FT re-runs dump probability tracks); AVES FT seed 1 too
cd ~/tsssl; P=venv/bin/python; L=logs
for s in 1 2; do
  $P finetune.py --feats ckpt/d2v_modern_seed${s}_20000.pt --name ft_d2v_modern_seed$s > $L/ft_d2v_modern_seed$s.log 2>&1; echo S_d2v_$s
  $P finetune.py --feats ckpt/kmaves_seed${s}_20000.pt --name ft_kmaves_seed$s > $L/ft_kmaves_seed$s.log 2>&1; echo S_kmaves_$s
  $P finetune.py --feats ckpt/kmaves_modern_seed${s}_20000.pt --name ft_kmaves_modern_seed$s > $L/ft_kmaves_modern_seed$s.log 2>&1; echo S_kmavesm_$s
  $P finetune.py --feats scratch --block modern --seed $s --name ft_scratch_modern_seed$s > $L/ft_scratch_modern_seed$s.log 2>&1; echo S_scratch_$s
done
$P finetune.py --feats aves --name ft_aves_seed1 --seed 1 > $L/ft_aves_seed1.log 2>&1; echo S_aves_1
$P recall_far.py ft_d2v_modern ft_d2v_modern_seed1 ft_d2v_modern_seed2 ft_kmaves ft_kmaves_seed1 ft_kmaves_seed2 ft_kmaves_modern ft_kmaves_modern_seed1 ft_kmaves_modern_seed2 ft_scratch_modern ft_scratch_modern_seed1 ft_scratch_modern_seed2 ft_aves ft_aves_seed1 ft_kmiter2_seed0 aves > $L/recall_far_seeds.log 2>&1
echo RECALLSEEDS_DONE
