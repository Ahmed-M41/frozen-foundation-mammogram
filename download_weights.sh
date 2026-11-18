#!/bin/bash
# Resumable downloads over a slow line: curl -C - retries until each file is complete.
# Files land in F:/hf_weights/<name>; log in cache/download_weights.log
cd /f/hf_weights
get() {  # name url expected_bytes
  for i in $(seq 1 500); do
    have=$(stat -c %s "$1" 2>/dev/null || echo 0)
    [ "$have" -ge "$3" ] && { echo "$(date +%H:%M) DONE $1 $have bytes"; return 0; }
    # fetch the next chunk as an explicit byte range; append only on a 206 partial response
    code=$(curl -sSL -r "${have}-" --max-time 300 -o part.tmp -w '%{http_code}' "$2" 2>/dev/null)
    if [ "$code" = "206" ] && [ -s part.tmp ]; then cat part.tmp >> "$1"; fi
    rm -f part.tmp
    echo "$(date +%H:%M) $1 $(stat -c %s "$1" 2>/dev/null || echo 0)/$3 bytes (http $code)"
  done
}
size() { curl -sIL "$1" | grep -i '^x-linked-size\|^content-length' | tail -1 | awk '{print $2}' | tr -d '\r'; }
U1=https://huggingface.co/timm/vit_base_patch16_dinov3.lvd1689m/resolve/main/model.safetensors
U2=https://huggingface.co/batmanLab/Mammo-FM/resolve/main/Mammo-FM_BatmanlabTrained_CLIP.tar
S1=$(size $U1); S2=$(size $U2); echo "$(date +%H:%M) sizes: dinov3_vitb16=$S1 mammofm=$S2"
get dinov3_vitb16.safetensors $U1 $S1
get mammofm_clip.tar $U2 $S2
