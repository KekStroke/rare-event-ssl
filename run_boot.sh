#!/bin/bash
# re-probe the headline models so per-recording events are saved, then paired bootstrap
cd ~/tsssl; P=venv/bin/python; L=logs
$P probe.py --feats aves --name aves > $L/probe_aves.log 2>&1; echo R_aves
$P probe.py --feats mel --name mel > $L/probe_mel.log 2>&1; echo R_mel
for v in kmaves kmaves_seed1 kmaves_seed2 kmaves_modern kmaves_modern_seed1 kmaves_modern_seed2 d2v d2v_modern d2v_modern_seed1 d2v_modern_seed2; do
  $P probe.py --feats ckpt/${v}_20000.pt --name $v > $L/probe_$v.log 2>&1; echo R_$v
done
$P bootstrap.py d2v_modern kmaves > $L/boot_d2vm_vs_kmaves.log 2>&1
$P bootstrap.py kmaves aves > $L/boot_kmaves_vs_aves.log 2>&1
$P bootstrap.py d2v_modern d2v > $L/boot_d2vm_vs_d2v.log 2>&1
echo BOOT_DONE
