#!/bin/bash
# DINOv3 ViT-L (fair match for our DINOv2 ViT-L), then embed + evaluate it (one GPU job).
cd /f/hf_weights
U=https://huggingface.co/timm/vit_large_patch16_dinov3.lvd1689m/resolve/main/model.safetensors
F=dinov3_vitl16.safetensors; N=1212347640
for i in $(seq 1 200); do
  have=$(stat -c %s $F 2>/dev/null || echo 0); [ "$have" -ge "$N" ] && break
  code=$(curl -sSL -r "${have}-" --max-time 600 -o part_l.tmp -w '%{http_code}' "$U" 2>/dev/null)
  if [ "$code" = "206" ] && [ -s part_l.tmp ]; then cat part_l.tmp >> $F; fi; rm -f part_l.tmp
  echo "$(date +%H:%M) $F $(stat -c %s $F)/$N (http $code)"
done
[ "$(stat -c %s $F)" -eq "$N" ] || { echo "download incomplete"; exit 1; }
echo "$(date +%H:%M) DONE $F"
