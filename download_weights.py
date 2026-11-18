"""Resumable, retrying weight downloads over a slow connection (one file at a time).
Run: python download_weights.py  (log: cache/download_weights.log)"""
import os
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import hf_hub_download

QUEUE = [  # (repo, filename) in priority order
    ("timm/vit_base_patch16_dinov3.lvd1689m", "model.safetensors"),
    ("batmanLab/Mammo-FM", "Mammo-FM_BatmanlabTrained_CLIP.tar"),
    ("timm/vit_large_patch16_dinov3.lvd1689m", "model.safetensors"),
]

for repo, fn in QUEUE:
    for attempt in range(200):
        try:
            t = time.time()
            p = hf_hub_download(repo, fn, etag_timeout=60)
            print(f"{time.strftime('%H:%M')} DONE {repo}/{fn} {os.path.getsize(p)/1e6:.0f} MB "
                  f"in {(time.time()-t)/60:.1f} min", flush=True)
            break
        except Exception as e:  # noqa: BLE001
            print(f"{time.strftime('%H:%M')} retry {attempt} {repo}: {type(e).__name__}", flush=True)
            time.sleep(5)
