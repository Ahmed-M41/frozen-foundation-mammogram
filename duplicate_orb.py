"""E3 step 3: decide shared films by keypoint matching with geometric verification.

Candidates: for every Mini-DDSM image, the 20 CBIS-DDSM images closest in perceptual hash
(original and mirrored), i.e. ~213k pairs out of 8.5M. For each pair, ORB keypoints are
matched (cross-checked Hamming) and a similarity transform is fitted with RANSAC; the
number of geometrically consistent inliers is the duplicate score. The same film yields
many inliers; different breasts yield few. The decision threshold is read off the
(bimodal) inlier distribution and confirmed visually.

Run: python duplicate_orb.py -> cache/duplicate_orb.json, cache/duplicates/orb_*.png
"""
import json
import os

import cv2
import numpy as np
from PIL import Image

import config
import db
from duplicate_search import hashes, hamming

K = 20
OUT_DIR = os.path.join(config.PROJECT_DIR, "cache", "duplicates")
orb = cv2.ORB_create(nfeatures=1500, fastThreshold=10)


def load(path, flip=False):
    im = Image.open(path).convert("L")
    if flip:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    w, h = im.size
    im = im.resize((int(512 * w / h), 512), Image.BILINEAR)
    return np.asarray(im)


def desc(img):
    kp, d = orb.detectAndCompute(img, None)
    return np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32), d


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def inliers(a, b):
    (pa, da), (pb, db_) = a, b
    if da is None or db_ is None or len(da) < 8 or len(db_) < 8:
        return 0
    m = bf.match(da, db_)
    if len(m) < 8:
        return 0
    src = pa[[x.queryIdx for x in m]]; dst = pb[[x.trainIdx for x in m]]
    M, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    return int(mask.sum()) if mask is not None else 0


def main():
    conn = db.connect()
    cb = conn.execute("SELECT id, file_path, file_name, case_id, split, pathology FROM samples "
                      "WHERE dataset='cbis' ORDER BY id").fetchall()
    mi = conn.execute("SELECT id, file_path, file_name, dataset, case_id, pathology FROM samples "
                      "WHERE dataset IN ('miniddsm','miniddsm_normal') ORDER BY id").fetchall()
    Hc = hashes([r["file_path"] for r in cb]); Hcf = hashes([r["file_path"] for r in cb], flip=True)
    Hm = hashes([r["file_path"] for r in mi])
    D0, D1 = hamming(Hc, Hm), hamming(Hcf, Hm)
    print("hashes done; computing ORB descriptors", flush=True)
    Dc = [desc(load(r["file_path"])) for r in cb]
    Dcf = [desc(load(r["file_path"], True)) for r in cb]
    print("cbis descriptors done", flush=True)
    res = []
    for j, r in enumerate(mi):
        dm = desc(load(r["file_path"]))
        best = (0, -1, False)
        for flip, Dmat, Dd in ((False, D0, Dc), (True, D1, Dcf)):
            for i in np.argsort(Dmat[:, j])[:K]:
                n = inliers(Dd[i], dm)
                if n > best[0]:
                    best = (n, int(i), flip)
        res.append((j,) + best)
        if j % 500 == 0:
            print(f"  matched {j}/{len(mi)}", flush=True)
    n_in = np.array([x[1] for x in res])
    hist = np.histogram(n_in, bins=[0, 10, 15, 20, 25, 30, 40, 50, 75, 100, 200, 2000])
    print("best-inlier histogram per Mini image:", list(zip(hist[1][:-1].tolist(), hist[0].tolist())), flush=True)
    rows = []
    for j, n, i, flip in sorted(res, key=lambda x: -x[1]):
        rows.append(dict(inliers=n, mirrored=flip, mini=mi[j]["file_name"], mini_path=mi[j]["file_path"],
                         mini_case=mi[j]["case_id"], mini_folder=mi[j]["dataset"],
                         mini_pathology=mi[j]["pathology"], cbis=cb[i]["file_name"],
                         cbis_path=cb[i]["file_path"], cbis_case=cb[i]["case_id"],
                         cbis_split=cb[i]["split"], cbis_pathology=cb[i]["pathology"]))
    json.dump(dict(k=K, histogram=list(zip(hist[1][:-1].tolist(), hist[0].tolist())), pairs=rows),
              open(os.path.join(config.PROJECT_DIR, "cache", "duplicate_orb.json"), "w"), indent=2)
    # sheets at inlier bands for visual threshold confirmation
    bands = [(200, 2000), (75, 200), (40, 75), (25, 40), (20, 25), (15, 20), (10, 15)]
    for lo, hi in bands:
        sel = [r for r in rows if lo <= r["inliers"] < hi][:10]
        if not sel:
            continue
        sheet = Image.new("L", (10 * 200, 2 * 210 + 30), 255)
        for k, r in enumerate(sel):
            for row, key in enumerate(("cbis_path", "mini_path")):
                im = Image.open(r[key]).convert("L").resize((190, 200))
                if row == 0 and r["mirrored"]:
                    im = im.transpose(Image.FLIP_LEFT_RIGHT)
                sheet.paste(im, (k * 200, row * 220))
        sheet.save(os.path.join(OUT_DIR, f"orb_{lo:04d}_{hi:04d}.png"))
        print(f"band {lo}-{hi}: {sum(lo <= r['inliers'] < hi for r in rows)} Mini images", flush=True)


if __name__ == "__main__":
    main()
