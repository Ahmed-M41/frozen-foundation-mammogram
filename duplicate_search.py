"""E3 (reviewer R2.4): exhaustive search for films shared between CBIS-DDSM and Mini-DDSM.

Every CBIS-DDSM image (1,592; breast-cropped versions, plus horizontal mirror) is compared
with every Mini-DDSM image (5,328; normal, benign and cancer folders) by three signals:
  1. 256-bit DCT perceptual hash (the method of Section III-C), Hamming distance;
  2. cosine similarity of frozen DINOv2 ViT-L/14 features (same cropped images);
  3. cosine similarity of frozen Mammo-CLIP features.
A pair is a candidate if its hash distance is unusually small or if it is the mutual nearest
neighbour in feature space with an outlying similarity. Candidates are written with contact
sheets for visual confirmation (cache/duplicates/).

Run: python duplicate_search.py  ->  cache/duplicate_search.json, cache/duplicates/*.png
"""
import json
import os

import numpy as np
from PIL import Image
from scipy.fft import dctn

import config
import db

OUT_DIR = os.path.join(config.PROJECT_DIR, "cache", "duplicates")
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def phash(img):
    g = np.asarray(img.convert("L").resize((64, 64), Image.LANCZOS), dtype=np.float64)
    d = dctn(g, norm="ortho")[:16, :16].flatten()
    return np.packbits(d > np.median(d[1:]))


def hashes(paths, flip=False):
    out = []
    for k, p in enumerate(paths):
        im = Image.open(p)
        if flip:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        out.append(phash(im))
        if k % 1000 == 0:
            print(f"  hashed {k}/{len(paths)}", flush=True)
    return np.stack(out)


def hamming(A, B):
    D = np.empty((len(A), len(B)), dtype=np.uint16)
    for i in range(0, len(A), 64):
        D[i:i + 64] = POP[np.bitwise_xor(A[i:i + 64, None, :], B[None, :, :])].sum(-1)
    return D


def feats(conn, enc, ds, ids):
    eids, X, _ = db.load_embeddings(conn, enc, dataset=ds)
    pos = {int(i): k for k, i in enumerate(eids)}
    X = X[[pos[int(i)] for i in ids]]
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    conn = db.connect()
    cb = conn.execute("SELECT id, file_path, file_name FROM samples WHERE dataset='cbis' ORDER BY id").fetchall()
    mi = conn.execute("SELECT id, file_path, file_name, dataset, pathology FROM samples "
                      "WHERE dataset IN ('miniddsm','miniddsm_normal') ORDER BY id").fetchall()
    print(f"CBIS {len(cb)} x Mini-DDSM {len(mi)} = {len(cb) * len(mi):,} pairs", flush=True)
    Hc = hashes([r["file_path"] for r in cb]); Hcf = hashes([r["file_path"] for r in cb], flip=True)
    Hm = hashes([r["file_path"] for r in mi])
    D = np.minimum(hamming(Hc, Hm), hamming(Hcf, Hm))
    print(f"hash distances: min {D.min()}  1st pct {np.percentile(D, 1):.0f}  median {np.median(D):.0f}", flush=True)

    # feature similarity on the SAME cropped CBIS images that were hashed
    cb_ids = np.array([r["id"] for r in cb])
    mi_ids = np.array([r["id"] for r in mi])
    sims = {}
    for enc in ("dinov2_vitl14_noclahe", "mammo_clip_b5"):
        A = feats(conn, enc, "cbis", cb_ids)
        Bm = np.concatenate([feats(conn, enc, "miniddsm", mi_ids[np.array([r["dataset"] == "miniddsm" for r in mi])]),
                             feats(conn, enc, "miniddsm_normal", mi_ids[np.array([r["dataset"] == "miniddsm_normal" for r in mi])])])
        order = np.concatenate([np.where([r["dataset"] == "miniddsm" for r in mi])[0],
                                np.where([r["dataset"] == "miniddsm_normal" for r in mi])[0]])
        S = np.empty((len(A), len(mi)), dtype=np.float32); S[:, order] = A @ Bm.T
        sims[enc] = S
    # candidate pairs
    cand = {}
    hz = D.astype(float)
    hash_thr = max(40, np.percentile(D, 0.001))
    for i, j in zip(*np.where(D <= hash_thr)):
        cand[(int(i), int(j))] = "hash"
    for enc, S in sims.items():
        z = (S - S.mean(1, keepdims=True)) / S.std(1, keepdims=True)
        bi = S.argmax(1); bj = S.argmax(0)
        for i in range(S.shape[0]):
            j = bi[i]
            if bj[j] == i and z[i, j] > 6:
                cand.setdefault((i, int(j)), f"mutual-NN {enc}")
    rows = []
    for (i, j), why in cand.items():
        rows.append(dict(cbis=cb[i]["file_name"], cbis_path=cb[i]["file_path"],
                         mini=mi[j]["file_name"], mini_path=mi[j]["file_path"],
                         mini_folder=mi[j]["dataset"], mini_pathology=mi[j]["pathology"],
                         hash_dist=int(D[i, j]),
                         sim_dinov2=float(sims["dinov2_vitl14_noclahe"][i, j]),
                         sim_mammoclip=float(sims["mammo_clip_b5"][i, j]), why=why))
    rows.sort(key=lambda r: (r["hash_dist"], -r["sim_dinov2"]))
    json.dump(dict(n_pairs=len(cb) * len(mi), hash_threshold=float(hash_thr),
                   hash_min=int(D.min()), hash_median=float(np.median(D)), candidates=rows),
              open(os.path.join(config.PROJECT_DIR, "cache", "duplicate_search.json"), "w"), indent=2)
    print(f"{len(rows)} candidate pairs", flush=True)
    # contact sheets, 10 pairs per sheet
    for s in range(0, min(len(rows), 60), 10):
        chunk = rows[s:s + 10]
        sheet = Image.new("L", (10 * 200, 2 * 260 + 20), 255)
        for k, r in enumerate(chunk):
            for row, key in enumerate(("cbis_path", "mini_path")):
                im = Image.open(r[key]).convert("L"); im.thumbnail((200, 260))
                sheet.paste(im, (k * 200, row * 270))
        sheet.save(os.path.join(OUT_DIR, f"candidates_{s // 10:02d}.png"))
    for k, r in enumerate(rows[:60]):
        print(f"{k:2d} hash {r['hash_dist']:3d} dino {r['sim_dinov2']:.3f} mclip {r['sim_mammoclip']:.3f} "
              f"{r['why'][:30]:30s} {r['cbis'][:28]} <-> {r['mini']} ({r['mini_pathology']})")


if __name__ == "__main__":
    main()
