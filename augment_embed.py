"""Orientation-augmented embeddings for laterality-invariant retrieval.

Frozen encoders are NOT flip-invariant, so a RIGHT-breast lesion and a similar
LEFT-breast lesion produce different embeddings and can't retrieve each other.
Two remedies (both training-free — just more/aligned embeddings of the SAME images):

  mode=mir    embed the horizontal MIRROR of each image, stored under E__mir.
              Gallery = E (train) + E__mir (train)  ->  covers both orientations.
  mode=canon  flip only RIGHT-laterality images so ALL breasts share one
              orientation, stored under E__canon. Used for BOTH gallery and query.
              (Cleaner: no gallery doubling, perfect alignment; uses known laterality.)

Run:  python augment_embed.py --embedder dinov2_vitb14_noclahe --mode mir
      python augment_embed.py --embedder dinov2_vitb14_noclahe --mode canon --dataset cbis
"""
import argparse
import sys

import numpy as np
from PIL import ImageOps

import config
import db
import embedders
from preprocessing import load_image


def _rows(conn, dataset):
    if dataset is not None:
        return conn.execute("SELECT id, file_path, laterality FROM samples WHERE dataset=? ORDER BY id",
                            (dataset,)).fetchall()
    return conn.execute("SELECT id, file_path, laterality FROM samples ORDER BY id").fetchall()


def augment_embed(encoder, mode, dataset=None, batch_size=config.BATCH_SIZE):
    conn = db.init_db()
    store = f"{encoder}__{mode}"
    done = db.embedded_sample_ids(conn, store)
    rows = [r for r in _rows(conn, dataset) if r["id"] not in done]
    print(f"  store={store} to_embed={len(rows)}")
    if not rows:
        print("RESULT: PASS (nothing to do)")
        return True

    emb = embedders.load_embedder(encoder)
    buf_ids, buf_imgs, n = [], [], 0

    def flush():
        nonlocal n
        if not buf_imgs:
            return
        vecs = emb.embed_batch(buf_imgs)
        for sid, v in zip(buf_ids, vecs):
            db.insert_embedding(conn, sid, store, v)
            n += 1
        conn.commit()
        buf_ids.clear()
        buf_imgs.clear()

    for r in rows:
        img = load_image(r["file_path"])
        if img is None:
            continue
        if mode == "mir":
            img = ImageOps.mirror(img)
        elif mode == "canon":
            if str(r["laterality"]).upper() == "RIGHT":
                img = ImageOps.mirror(img)   # flip R -> L orientation; leave LEFT as-is
        buf_ids.append(r["id"])
        buf_imgs.append(img)
        if len(buf_imgs) >= batch_size:
            flush()
    flush()

    ids, vecs, _ = db.load_embeddings(conn, store, dataset=dataset)
    ok = len(ids) > 0 and not bool(np.isnan(vecs).any()) and \
        float(np.linalg.norm(vecs, axis=1).min()) > 0
    print(f"  {store}: embedded {n}, rows={len(ids)}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", required=True)
    ap.add_argument("--mode", choices=["mir", "canon"], required=True)
    ap.add_argument("--dataset", default=None)
    args = ap.parse_args()
    sys.exit(0 if augment_embed(args.embedder, args.mode, args.dataset) else 1)
