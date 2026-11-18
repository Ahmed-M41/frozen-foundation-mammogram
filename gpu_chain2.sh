#!/bin/bash
cd /f/Development/MamoRAC
export HF_HUB_OFFLINE=1
until grep -q "^TOTAL" cache/lora_dinov2.log; do sleep 60; done
echo "$(date +%H:%M) start LoRA raddino"; python -u finetune_lora.py --model raddino > cache/lora_raddino.log 2>&1
echo "$(date +%H:%M) evaluating LoRA"; python -u finetune_lora_eval.py > cache/lora_eval.log 2>&1
echo "$(date +%H:%M) chain2 done"
