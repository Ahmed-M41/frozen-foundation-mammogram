"""User-suggested test: does CLAHE help the FROZEN-feature probe on CBIS (cbis_full)?

The current pipeline uses NO CLAHE. The user's idea (which helped their trained models):
random-range CLAHE as TRAIN augmentation + fixed CLAHE at TEST. In the frozen setting
this augments the PROBE's training features rather than the encoder. Tested unbiased.

Three conditions, 3-encoder multi-view probe, paired-test AUC (comparable to the current
no-CLAHE 0.851):
  (1) no CLAHE          -> cached DB features (baseline)
  (2) fixed CLAHE       -> clip 4.0 on train and test
  (3) CLAHE augmentation-> train re-encoded K times at random clip in [2,6]; test clip 4.0

Encodes the full-resolution cbis_full mammograms on the fly (slow) for (2) and (3).
"""
import numpy as np
import cv2
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import db, evaluate
import embedders as EMB
import mammo_clip
from mil import _best_threshold, _metrics

TILE = (6, 6)
KAUG = 3
AUG_CLIPS = (2.0, 6.0)      # random uniform range for augmentation
TEST_CLIP = 4.0
SEEDS = [0, 1, 2, 3]
POS = ["MALIGNANT", "CANCER"]


def clahe_pil(path, clip):
    """Open full-res mammogram, apply CLAHE(clip) on 8-bit grayscale, return RGB PIL."""
    g = np.asarray(Image.open(path).convert("L"), np.uint8)
    if clip and clip > 0:
        g = cv2.createCLAHE(clipLimit=clip, tileGridSize=TILE).apply(g)
    return Image.fromarray(np.repeat(g[:, :, None], 3, 2), "RGB")


def load_meta():
    conn = db.connect()
    rows = conn.execute(
        "SELECT id,file_path,split,pathology,view,patient_id,case_id "
        "FROM samples WHERE dataset='cbis_full' AND view IN ('CC','MLO') ORDER BY id").fetchall()
    return conn, rows


CACHED = ["mammo_clip_b5", "dinov2_vitl14_noclahe", "rad_dino_noclahe"]


def build_encoders():
    mc = mammo_clip.build("mammo_clip_b5")                       # use_clahe=False
    dv = EMB.load_embedder("dinov2_vitl14_noclahe")
    rd = EMB.load_embedder("rad_dino_noclahe")
    return [mc, dv, rd]


def load_cached(conn, ids):
    """Current no-CLAHE features from the DB, aligned to `ids` (for a fair baseline)."""
    mats = []
    for e in CACHED:
        eid, v, _ = db.load_embeddings(conn, e, dataset=["cbis_full"])
        idx = {int(i): k for k, i in enumerate(eid)}
        mats.append(_l2(v[[idx[int(i)] for i in ids]]))
    return np.concatenate(mats, 1).astype(np.float32)


def _l2(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def encode(encs, paths, clip, bs=8):
    """Concat L2-normed features of the 3 encoders for CLAHE(clip)-processed images."""
    mats = [[] for _ in encs]
    for i in range(0, len(paths), bs):
        imgs = [clahe_pil(p, clip) for p in paths[i:i+bs]]
        for j, e in enumerate(encs):
            mats[j].append(e.embed_batch(imgs))
    return np.concatenate([_l2(np.concatenate(m, 0)) for m in mats], 1).astype(np.float32)


def encode_random(encs, paths, lo, hi, seed, bs=8):
    rng = np.random.default_rng(seed)
    mats = [[] for _ in encs]
    for i in range(0, len(paths), bs):
        chunk = paths[i:i+bs]
        imgs = [clahe_pil(p, float(rng.uniform(lo, hi))) for p in chunk]
        for j, e in enumerate(encs):
            mats[j].append(e.embed_batch(imgs))
    return np.concatenate([_l2(np.concatenate(m, 0)) for m in mats], 1).astype(np.float32)


def case_auc(conn, ids, X, y, pat, view, tr_rows, te_rows, seed, aug_rows=None, aug=None):
    """View-agnostic head, patient-grouped 5-fold OOF, case-fused (mean) paired AUC."""
    Xtr, ytr, ptr = X[tr_rows], y[tr_rows], pat[tr_rows]
    if aug is not None:                       # append augmented train copies (same patients/labels)
        Xtr = np.concatenate([Xtr, aug["X"]], 0)
        ytr = np.concatenate([ytr, aug["y"]], 0)
        ptr = np.concatenate([ptr, aug["pat"]], 0)
    sg = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    te_pred = []
    for trf, _ in sg.split(Xtr, ytr, groups=ptr):
        s = StandardScaler().fit(Xtr[trf])
        c = LogisticRegression(max_iter=4000, C=0.05, class_weight="balanced")
        c.fit(s.transform(Xtr[trf]), ytr[trf])
        te_pred.append(c.predict_proba(s.transform(X[te_rows]))[:, 1])
    te_score = np.mean(te_pred, 0)
    br = [dict(ids=ids[te_rows], score_pos=te_score)]
    cf = evaluate.case_fuse(conn, br, method="mean")
    yv = np.array(cf["y_true"]); sv = np.array(cf["y_score"]); pidx = np.array(cf["paired_idx"], int)
    return roc_auc_score(yv[pidx], sv[pidx])


def main():
    conn, rows = load_meta()
    ids = np.array([r["id"] for r in rows])
    paths = [r["file_path"] for r in rows]
    y = np.array([1 if str(r["pathology"]).upper() in POS else 0 for r in rows])
    pat = np.array([r["patient_id"] for r in rows])
    view = np.array([r["view"] for r in rows])
    split = np.array([r["split"] for r in rows])
    tr_rows, te_rows = split == "train", split == "test"
    print(f"cbis_full: {len(rows)} imgs  train={tr_rows.sum()} test={te_rows.sum()}", flush=True)

    encs = build_encoders()

    print("encoding FIXED CLAHE (clip 4.0) ...", flush=True)
    Xfix = encode(encs, paths, TEST_CLIP)

    print(f"encoding AUG train ({KAUG} random-clip copies) ...", flush=True)
    tr_idx = np.where(tr_rows)[0]
    tr_paths = [paths[i] for i in tr_idx]
    augX, augy, augp = [], [], []
    for k in range(KAUG):
        augX.append(encode_random(encs, tr_paths, AUG_CLIPS[0], AUG_CLIPS[1], seed=100+k))
        augy.append(y[tr_idx]); augp.append(pat[tr_idx])
        print(f"  aug copy {k+1}/{KAUG} done", flush=True)
    aug = dict(X=np.concatenate(augX, 0), y=np.concatenate(augy, 0), pat=np.concatenate(augp, 0))

    Xno = load_cached(conn, ids)              # no-CLAHE baseline, SAME probe for fairness
    res = {"none": [], "fixed": [], "aug": []}
    for seed in SEEDS:
        res["none"].append(case_auc(conn, ids, Xno, y, pat, view, tr_rows, te_rows, seed))
        res["fixed"].append(case_auc(conn, ids, Xfix, y, pat, view, tr_rows, te_rows, seed))
        res["aug"].append(case_auc(conn, ids, Xfix, y, pat, view, tr_rows, te_rows, seed, aug=aug))
    print("\n================ CBIS-DDSM 3-encoder probe (view-agnostic head), paired AUC ================")
    print(f"  (1) no CLAHE  (current)              = {np.mean(res['none']):.4f} +- {np.std(res['none']):.4f}")
    print(f"  (2) fixed CLAHE (clip 4.0)           = {np.mean(res['fixed']):.4f} +- {np.std(res['fixed']):.4f}")
    print(f"  (3) CLAHE augmentation (your idea)   = {np.mean(res['aug']):.4f} +- {np.std(res['aug']):.4f}")


if __name__ == "__main__":
    main()
