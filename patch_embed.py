"""Tile each breast image into patches and embed every patch with a FROZEN
encoder -> one bag of patch embeddings per image, cached to .npz.

This is the input for the attention-MIL head (mil.py): a small trained head that
learns to weight the suspicious patches, concentrating the localized malignancy
signal that a whole-breast embedding dilutes. The backbone stays frozen.

Multi-scale grid (2x2 + 4x4 = 20 patches) captures both context and detail.
No CLAHE (it hurts frozen encoders). Run:
    python patch_embed.py --embedder dinov2_vitb14_noclahe --dataset cbis
"""
import argparse
import os

import numpy as np

import config
import db
import embedders
from preprocessing import load_image

CACHE_DIR = os.path.join(config.PROJECT_DIR, "patch_cache")
GRIDS = (2, 4)          # 2x2 + 4x4 = 20 patches/image
N_PATCHES = sum(g * g for g in GRIDS)


def tiles(img, grid):
    w, h = img.size
    tw, th = w // grid, h // grid
    return [img.crop((c * tw, r * th, (c + 1) * tw, (r + 1) * th))
            for r in range(grid) for c in range(grid)]


def patch_embed(encoder, dataset="cbis"):
    os.makedirs(CACHE_DIR, exist_ok=True)
    conn = db.connect()
    rows = conn.execute("SELECT id, file_path FROM samples WHERE dataset=? ORDER BY id",
                        (dataset,)).fetchall()
    emb = embedders.load_embedder(encoder)
    print(f"  encoder={encoder} dim={emb.dim} images={len(rows)} patches/img={N_PATCHES}")

    ids, bags = [], []
    for i, r in enumerate(rows, 1):
        img = load_image(r["file_path"])
        if img is None:
            continue
        img = img.convert("RGB")
        t = []
        for g in GRIDS:
            t += tiles(img, g)
        vec = emb.embed_batch(t)              # [N_PATCHES, dim]
        ids.append(r["id"])
        bags.append(vec.astype(np.float32))
        if i % 200 == 0:
            print(f"\r  {i}/{len(rows)}", end="", flush=True)
    bags = np.stack(bags)                      # [n, N_PATCHES, dim]
    ids = np.array(ids, dtype=np.int64)
    out = os.path.join(CACHE_DIR, f"{encoder}__{dataset}.npz")
    np.savez_compressed(out, ids=ids, bags=bags)
    ok = not bool(np.isnan(bags).any()) and bags.shape[0] == len(ids)
    print(f"\r  saved {bags.shape} -> {out}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="dinov2_vitb14_noclahe")
    ap.add_argument("--dataset", default="cbis")
    args = ap.parse_args()
    import sys
    sys.exit(0 if patch_embed(args.embedder, args.dataset) else 1)
