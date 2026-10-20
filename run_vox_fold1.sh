#!/bin/bash
export MPLBACKEND=Agg
cd ~/tsssl; PY=~/voxenv/bin/python; MAIN=/mnt/c/Users/Iaroslav/CETACEANS/voxaboxen_main.py; W=/mnt/c/Users/Iaroslav/CETACEANS/voxaboxen_weights
CK=/mnt/c/Users/Iaroslav/CETACEANS/voxaboxen_whale/project/whale_aves/final-model.pt; k=1; D=~/tsssl/vox_field/fold$k; N=vox_ft_fold$k
$PY $MAIN train-model --project-config-fp $D/project/project_config.yaml --name $N --encoder-type beats --beats-checkpoint-fp $W/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
  --previous-checkpoint-fp $CK --bidirectional --early-stopping --n-epochs 50 --min-epochs 5 --patience 10 --lr 5e-5 --batch-size 16 --clip-duration 10 --clip-hop 5 \
  --unfreeze-encoder-epoch 3 --end-mask-perc 0.1 --lamb 0.04 --rho 0.01 --pos-loss-weight 1 --n-val-fit 19 --nms soft_nms --soft-nms-sigma 0.5 --nms-thresh 0.5 --peak-distance 5 \
  --rnn-hidden-size 2048 --num-workers 4 --seed 0 --val-during-training > logs/vox_ft_fold${k}_retry.log 2>&1 && echo VOX_TRAIN_1 || { echo VOX_TRAIN_FAIL_1; echo VOX1_DONE; exit; }
$PY $MAIN inference --model-args-fp $D/project/$N/params.yaml --file-info-for-inference ~/tsssl/vox_field/fold${k}_infer_info.csv --detection-threshold 0.05 --comb-discard-threshold 0.0 --comb-iou-threshold 0.5 > logs/vox_inf0_fold${k}_retry.log 2>&1 && echo VOX_INF0_1 || echo VOX_INF0_FAIL_1
echo VOX1_DONE
