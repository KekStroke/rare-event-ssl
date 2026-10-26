#!/bin/bash
# After the Voxaboxen field FT: rerun staff inference with comb-discard 0.0 so the recall-FAR curve can be swept over the
# combined boxes' Detection Prob (the discard filter is the last step of the staff decoder, so thresholding the discard-0 file
# at t reproduces the staff decoder with comb_discard = t exactly).
until grep -q "VOX_DONE" ~/tsssl/logs/run_vox_field.log; do sleep 120; done
cd ~/tsssl; PY=~/voxenv/bin/python; MAIN=/mnt/c/Users/Iaroslav/CETACEANS/voxaboxen_main.py
for k in 0 1 2; do D=~/tsssl/vox_field/fold$k/project/vox_ft_fold$k; [ -f $D/params.yaml ] || { echo VOX_INF0_SKIP_$k; continue; }
  $PY $MAIN inference --model-args-fp $D/params.yaml --file-info-for-inference ~/tsssl/vox_field/fold${k}_infer_info.csv --detection-threshold 0.05 --comb-discard-threshold 0.0 --comb-iou-threshold 0.5 > logs/vox_inf0_fold$k.log 2>&1 && echo VOX_INF0_$k || echo VOX_INF0_FAIL_$k
done; echo VOX_INF0_DONE
