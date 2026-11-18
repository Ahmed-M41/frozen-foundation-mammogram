"""E3 step 2: pixel-level verification of CBIS-DDSM / Mini-DDSM duplicate films.

For every Mini-DDSM image, the 30 CBIS-DDSM images (and their mirrors) closest in
perceptual hash are compared by normalised cross-correlation (NCC) of 128x128 grayscale
versions. A shared film gives NCC close to 1; different breasts give much lower values.
Then duplicates are grouped to breast level (case) on both sides.

Run after duplicate_search.py:  python duplicate_verify.py -> cache/duplicate_verify.json
"""
import json
import os

import numpy as np
from PIL import Image

import config
import db
from duplicate_search import hashes, hamming

K = 30
OUT_DIR = os.path.join(config.PROJECT_DIR, "cache", "duplicates")


def small(path, flip=False):
    im = Image.open(path).convert("L")
    if flip:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    a = np.asarray(im.resize((128, 128), Image.BILINEAR), dtype=np.float32).ravel()
    a -= a.mean()
    return a / (np.linalg.norm(a) + 1e-8)


def main():
    conn = db.connect()
    cb = conn.execute("SELECT id, file_path, file_name, case_id, split, pathology FROM samples "
                      "WHERE dataset='cbis' ORDER BY id").fetchall()
    mi = conn.execute("SELECT id, file_path, file_name, dataset, case_id, pathology FROM samples "
                      "WHERE dataset IN ('miniddsm','miniddsm_normal') ORDER BY id").fetchall()
    Hc = hashes([r["file_path"] for r in cb]); Hcf = hashes([r["file_path"] for r in cb], flip=True)
    Hm = hashes([r["file_path"] for r in mi])
    D0, D1 = hamming(Hc, Hm), hamming(Hcf, Hm)
    Sc = np.stack([small(r["file_path"]) for r in cb])
    Scf = np.stack([small(r["file_path"], True) for r in cb])
    print("cbis thumbnails ready", flush=True)
    best = []
    for j, r in enumerate(mi):
        m = small(r["file_path"])
        cand0 = np.argsort(D0[:, j])[:K]; cand1 = np.argsort(D1[:, j])[:K]
        n0 = Sc[cand0] @ m; n1 = Scf[cand1] @ m
        if n0.max() >= n1.max():
            i, ncc, flip = int(cand0[n0.argmax()]), float(n0.max()), False
        else:
            i, ncc, flip = int(cand1[n1.argmax()]), float(n1.max()), True
        best.append((j, i, ncc, flip))
        if j % 1000 == 0:
            print(f"  verified {j}/{len(mi)}", flush=True)
    nccs = np.array([b[2] for b in best])
    print("best-NCC percentiles (per Mini image):",
          {p: round(float(np.percentile(nccs, p)), 3) for p in (50, 90, 99, 99.5, 99.9)}, flush=True)
    rows = []
    for j, i, ncc, flip in sorted(best, key=lambda b: -b[2])[:400]:
        rows.append(dict(ncc=ncc, mirrored=flip, mini=mi[j]["file_name"], mini_path=mi[j]["file_path"],
                         mini_case=mi[j]["case_id"], mini_folder=mi[j]["dataset"],
                         mini_pathology=mi[j]["pathology"], cbis=cb[i]["file_name"],
                         cbis_path=cb[i]["file_path"], cbis_case=cb[i]["case_id"],
                         cbis_split=cb[i]["split"], cbis_pathology=cb[i]["pathology"],
                         hash_dist=int(min(D0[i, j], D1[i, j]))))
    json.dump(dict(k=K, ncc_percentiles={p: float(np.percentile(nccs, p)) for p in (50, 90, 99, 99.9)},
                   top=rows), open(os.path.join(config.PROJECT_DIR, "cache", "duplicate_verify.json"), "w"),
              indent=2)
    # contact sheets of the ranked list at NCC bands, for visual threshold setting
    for name, sel in (("ncc_top", rows[:10]), ("ncc_rank_40_50", rows[40:50]),
                      ("ncc_rank_80_90", rows[80:90]), ("ncc_rank_150_160", rows[150:160])):
        sheet = Image.new("L", (10 * 200, 2 * 210 + 30), 255)
        for k, r in enumerate(sel):
            for row, key in enumerate(("cbis_path", "mini_path")):
                im = Image.open(r[key]).convert("L").resize((190, 200))
                if row == 0 and r["mirrored"]:
                    im = im.transpose(Image.FLIP_LEFT_RIGHT)
                sheet.paste(im, (k * 200, row * 220))
        sheet.save(os.path.join(OUT_DIR, f"{name}.png"))
    for k in (0, 10, 20, 40, 60, 80, 100, 120, 150, 200, 300, 399):
        if k < len(rows):
            r = rows[k]
            print(f"rank {k:3d} NCC {r['ncc']:.3f} hash {r['hash_dist']} {r['cbis']} ({r['cbis_split']}) <-> {r['mini']} ({r['mini_pathology']})")


if __name__ == "__main__":
    main()
