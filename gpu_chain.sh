#!/bin/bash
# Run the GPU training jobs one after another (each waits for the previous to finish).
cd /f/Development/MamoRAC
export HF_HUB_OFFLINE=1
until grep -q "TOTAL" cache/resnet50_e2e.log || grep -q "Traceback" cache/resnet50_e2e.log; do sleep 30; done
echo "$(date +%H:%M) resnet50 finished; evaluating"; python -u finetune_resnet50_eval.py > cache/resnet50_e2e_eval.log 2>&1
echo "$(date +%H:%M) start LoRA dinov2"; python -u finetune_lora.py --model dinov2 > cache/lora_dinov2.log 2>&1
echo "$(date +%H:%M) start LoRA raddino"; python -u finetune_lora.py --model raddino > cache/lora_raddino.log 2>&1
echo "$(date +%H:%M) evaluating LoRA"; python -u finetune_lora_eval.py > cache/lora_eval.log 2>&1
echo "$(date +%H:%M) GPU chain done"
