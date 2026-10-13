"""Stage 3: embed every sample (train AND test) once and cache in the DB.

Resumable: skips sample_ids already embedded for the chosen embedder, so a
re-run after a crash is a no-op and adding a new encoder only embeds the gap.

Run:
    python embed_all.py --embedder resnet50 --limit 16   # smoke gate
    python embed_all.py --embedder resnet50              # full
"""
import argparse
import sys

import numpy as np

import config
import db
import embedders
from preprocessing import load_image


def embed_all(embedder_name, limit=None, batch_size=config.BATCH_SIZE, dataset=None):
    conn = db.init_db()
    done = db.embedded_sample_ids(conn, embedder_name)

    if dataset is not None:
        rows = conn.execute("SELECT id, file_path FROM samples WHERE dataset=? ORDER BY id",
                            (dataset,)).fetchall()
    else:
        rows = conn.execute("SELECT id, file_path FROM samples ORDER BY id").fetchall()
    todo = [(r["id"], r["file_path"]) for r in rows if r["id"] not in done]
    if limit is not None:
        todo = todo[:limit]

    print(f"  embedder            : {embedder_name}")
    print(f"  total samples       : {len(rows)}")
    print(f"  already embedded    : {len(done)}")
    print(f"  to embed (this run) : {len(todo)}")
    if not todo:
        print("RESULT: PASS (nothing to do — idempotent)")
        return True

    emb = embedders.load_embedder(embedder_name)
    print(f"  loaded {emb.name} dim={emb.dim} device={emb.device}")

    n_written = 0
    buf_ids, buf_imgs = [], []

    def flush():
        nonlocal n_written
        if not buf_imgs:
            return
        vecs = emb.embed_batch(buf_imgs)
        for sid, vec in zip(buf_ids, vecs):
            db.insert_embedding(conn, sid, embedder_name, vec)
            n_written += 1
        conn.commit()
        buf_ids.clear()
        buf_imgs.clear()

    for sid, path in todo:
        img = load_image(path)
        if img is None:
            continue
        buf_ids.append(sid)
        buf_imgs.append(img)
        if len(buf_imgs) >= batch_size:
            flush()
            print(f"\r  embedded {n_written}/{len(todo)}", end="", flush=True)
    flush()
    print(f"\r  embedded {n_written}/{len(todo)}      ")

    # --- gate checks ---
    ids, vecs, _ = db.load_embeddings(conn, embedder_name, dataset=dataset)
    has_nan = bool(np.isnan(vecs).any()) if vecs.size else False
    min_norm = float(np.linalg.norm(vecs, axis=1).min()) if vecs.size else 0.0
    total_emb = len(ids)
    total_samples = len(rows)

    print(f"  rows in embeddings: {total_emb}")
    print(f"  dim: {vecs.shape[1] if vecs.size else 'n/a'}  nan: {has_nan}  "
          f"min_norm: {min_norm:.4f}")

    if limit is not None:
        ok = (n_written == len(todo)) and not has_nan and min_norm > 0
        print("RESULT (smoke):", "PASS" if ok else "FAIL")
    else:
        ok = (total_emb == total_samples) and not has_nan and min_norm > 0
        print(f"  full coverage: {total_emb}/{total_samples}")
        print("RESULT (full):", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default=config.DEFAULT_EMBEDDER)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--dataset", default=None, help="only embed samples from this dataset")
    args = ap.parse_args()
    ok = embed_all(args.embedder, limit=args.limit, batch_size=args.batch_size,
                   dataset=args.dataset)
    sys.exit(0 if ok else 1)
