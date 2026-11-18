#!/bin/bash
# INbreast through the RSNA pipeline (crop + 512 px), then embed with the 4 paper encoders.
cd /f/Development/MamoRAC
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
L=cache/inbreast512_chain.log
echo "$(date +%H:%M) prep start" >> $L
python prep_inbreast512.py >> $L 2>&1; echo "$(date +%H:%M) prep exit $?" >> $L
for e in mammo_clip_b5 dinov2_vitl14_r518_noclahe rad_dino_noclahe mammoscreen_feat; do
  echo "$(date +%H:%M) embed $e" >> $L
  python -c "import embed_all; embed_all.embed_all('$e', dataset='inbreast_512')" >> cache/inbreast512_embed_$e.log 2>&1
  echo "$(date +%H:%M) embed $e exit $?" >> $L
done
echo "$(date +%H:%M) ALL DONE" >> $L
