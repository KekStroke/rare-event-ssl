#!/bin/bash
# Voxaboxen on OZF under its own reproduction recipe (scripts/voxaboxen_experiments.sh: natural OZF, BEATs, bidirectional, 50 epochs, batch 4, other args default),
# trained on the official train split with val for its model selection; then staff inference on test + val at detection threshold 0.05, comb discard 0.
cd ~/tsssl; PY=~/voxenv/bin/python; MAIN=/mnt/c/Users/Iaroslav/CETACEANS/voxaboxen_main.py; W=/mnt/c/Users/Iaroslav/CETACEANS/voxaboxen_weights
DATA=/home/yarix/tsssl/data/ozf_raw/OZF_unz/OZF/formatted; PRJ=/home/yarix/tsssl/vox_ozf/project
$PY $MAIN project-setup --data-dir $DATA --project-dir $PRJ > logs/vox_ozf_setup.log 2>&1 && echo VOXOZF_SETUP || { echo VOXOZF_SETUP_FAIL; exit 1; }
$PY $MAIN train-model --project-config-fp $PRJ/project_config.yaml --name vox_ozf --n-epochs 50 --batch-size 4 --encoder-type beats --beats-checkpoint-fp $W/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt --bidirectional --num-workers 8 --seed 0 > logs/vox_ozf_train.log 2>&1 && echo VOXOZF_TRAIN || { echo VOXOZF_TRAIN_FAIL; exit 1; }
$PY $MAIN inference --model-args-fp $PRJ/vox_ozf/params.yaml --file-info-for-inference /home/yarix/tsssl/vox_ozf/infer_info.csv --detection-threshold 0.05 --comb-discard-threshold 0.0 --comb-iou-threshold 0.5 > logs/vox_ozf_inf.log 2>&1 && echo VOXOZF_INF || echo VOXOZF_INF_FAIL
echo VOXOZF_DONE
