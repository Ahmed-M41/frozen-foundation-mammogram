#!/bin/bash
# One GPU job at a time (memory guard stopped parallel jobs on 2026-11-14 04:00).
cd /f/Development/MamoRAC
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
python overnight_queue.py dinov3_vitb16_r512_noclahe radimagenet_densenet121 dinov2_vitb14_r518_noclahe \
  convnextv2_base_noclahe convnext_base_noclahe davit_base_noclahe vit_base_noclahe resnet50_noclahe \
  densenet121_noclahe dinov2_vits14_r518_noclahe
python summarize_encoder_evals.py > cache/encoder_summary.txt 2>&1
echo "$(date +%H:%M) chain3 done"
