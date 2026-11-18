#!/bin/bash
# RSNA subset: crop+ingest, then embed with the 4 paper encoders + Mammo-FM (one GPU job at a time).
cd /f/Development/MamoRAC
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
L=cache/rsna_chain.log
echo "$(date +%H:%M) prep start" >> $L
python prep_rsna.py >> cache/rsna_prep.log 2>&1; echo "$(date +%H:%M) prep exit $?" >> $L
for e in mammo_clip_b5 dinov2_vitl14_r518_noclahe rad_dino_noclahe mammoscreen_feat mammo_fm; do
  echo "$(date +%H:%M) embed $e" >> $L
  python -c "import embed_all; embed_all.embed_all('$e', dataset='rsna_sub')" >> cache/rsna_embed_$e.log 2>&1
  echo "$(date +%H:%M) embed $e exit $?" >> $L
done
echo "$(date +%H:%M) ALL DONE" >> $L
