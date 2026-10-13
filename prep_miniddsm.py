"""Prepare & ingest Mini-DDSM (film, JPEG-8 light version) as a SELF-CONTAINED
benign-vs-malignant dataset — its own database + held-out test, no CBIS involved.

Why a fresh dataset: the frozen `ianpan/mammoscreen` score in Ensembler B was
pretrained on CBIS-DDSM, so B's CBIS numbers carry train-on-test leakage through
the score head. Re-running the whole pipeline inside Mini-DDSM (independent
curation + patient split) is a generalization/robustness check. (Caveat: Mini-DDSM
abnormals are the SAME DDSM films, so this is NOT leakage-free for the score head —
it is an independent-split check, documented as such.)

LABELLING TRAP handled here: Mini-DDSM's `Status` (Benign/Cancer/Normal) is
per-PATIENT-FOLDER, but each folder holds BOTH breasts x BOTH views. The healthy
contralateral breast inherits the folder label. We use the `Tumour_Contour`
columns to keep ONLY the lesion-bearing breast(s):
  case  = (patient stem, Side) that has a contour on >=1 of its 2 views
  label = 1 if Status==Cancer else 0 (Benign)
  drop  = Normal folder entirely + contralateral non-lesion breasts
Metadata analysis (DataWMask.xlsx, 1952 patients x 4 imgs): yields 1460 lesion
breasts (Benign 720 / Cancer 740, ~49/51), every one with BOTH CC+MLO. Zero
Benign/Cancer patients lack a contour, so no lesion breast is unlocatable.

Preprocessing matches CBIS exactly: YOLO/bbox breast-crop (mammo_crop =
ianpan/mammo-crop) -> 512x512 grayscale JPEG on disk; CLAHE is added later at
embed time (preprocessing.CLAHETransform), NOT baked in here.

Ingested as dataset='miniddsm', patient_id=<stem> (e.g. C_0029_1, globally
unique — groups both breasts of bilateral patients so the split stays leak-free),
case_id='miniddsm:<stem>:<SIDE>', split assigned here (patient-level, class-
stratified, fixed seed).

Run (after unzipping MINI-DDSM-Complete-JPEG-8):
    python prep_miniddsm.py --img-root "<...>/MINI-DDSM-Complete-JPEG-8" \
        --limit 16     # smoke: crop+ingest 16 images, verify, then drop --limit
    python prep_miniddsm.py --img-root "<...>/MINI-DDSM-Complete-JPEG-8"
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
from PIL import Image

import db

DATASET = "miniddsm"
DEFAULT_XLSX = r"C:\Users\pc\Downloads\DataWMask.xlsx"
DEFAULT_OUT = r"F:\Development\MasterNotebooks\preproc_jpg_miniddsm"
CROP_SIZE = 512


def build_cases(xlsx):
    """Return a DataFrame of lesion-bearing breast-view images to ingest."""
    df = pd.read_excel(xlsx)
    df.columns = [c.strip() for c in df.columns]
    df["stem"] = df["fileName"].str.split(".").str[0]            # C_0029_1 (patient id)
    c1 = df["Tumour_Contour"].astype(str).str.strip()
    c2 = df["Tumour_Contour2"].astype(str).str.strip()
    df["has_contour"] = (c1 != "-") | (c2 != "-")
    df["rel"] = df["fullPath"].apply(lambda p: re.split(r"[\\/]", str(p)))  # [Status, pdir, file]

    # affected breast = (stem, Side) with a contour on >=1 view
    aff = (df.groupby(["stem", "Side"])["has_contour"].max()
             .rename("breast_affected").reset_index())
    df = df.merge(aff, on=["stem", "Side"], how="left")

    keep = df[(df["Status"].isin(["Benign", "Cancer"])) & (df["breast_affected"])].copy()
    keep["binary_label"] = (keep["Status"] == "Cancer").astype(int)
    keep["case_id"] = "miniddsm:" + keep["stem"] + ":" + keep["Side"]
    keep["dataset"] = "miniddsm"
    return keep


def build_normals(xlsx):
    """Normal breasts (both sides of Normal-folder patients) for the normal-vs-cancer
    task. Ingested as a separate dataset 'miniddsm_normal', binary_label=0. Combined
    with the malignant subset of 'miniddsm' (label 1) at experiment time."""
    df = pd.read_excel(xlsx)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["Status"] == "Normal"].copy()
    df["stem"] = df["fileName"].str.split(".").str[0]
    df["rel"] = df["fullPath"].apply(lambda p: re.split(r"[\\/]", str(p)))
    df["binary_label"] = 0
    df["case_id"] = "miniddsm_normal:" + df["stem"] + ":" + df["Side"]
    df["dataset"] = "miniddsm_normal"
    return df


def patient_split(keep, test_frac, seed):
    """Class-stratified split over PATIENTS (stem). Bilateral patients (both
    breasts kept) stay wholly in one split. Patient class = max case label."""
    plabel = keep.groupby("stem")["binary_label"].max()
    rng = np.random.default_rng(seed)
    split = {}
    for cls in (0, 1):
        pats = plabel[plabel == cls].index.to_numpy()
        rng.shuffle(pats)
        n_test = int(round(test_frac * len(pats)))
        for p in pats[:n_test]:
            split[p] = "test"
        for p in pats[n_test:]:
            split[p] = "train"
    return keep["stem"].map(split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-root", required=True,
                    help="path to the extracted MINI-DDSM-Complete-JPEG-8 folder")
    ap.add_argument("--xlsx", default=DEFAULT_XLSX)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="smoke: process only N images")
    ap.add_argument("--no-crop", action="store_true",
                    help="debug: skip breast crop (store resized full film)")
    ap.add_argument("--lesions-only", action="store_true",
                    help="skip the normal breasts (ingest only the benign/malignant lesion set)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    lesion = build_cases(args.xlsx)
    lesion["split"] = patient_split(lesion, args.test_frac, args.seed)
    parts = [lesion]
    if not args.lesions_only:
        normal = build_normals(args.xlsx)
        normal["split"] = patient_split(normal, args.test_frac, args.seed)
        parts.append(normal)
    keep = pd.concat(parts, ignore_index=True)
    keep = keep.sort_values(["dataset", "split", "stem", "Side", "View"]).reset_index(drop=True)
    if args.limit is not None:
        keep = keep.iloc[:args.limit].copy()

    crop_array = None
    if not args.no_crop:
        import mammo_crop
        crop_array = mammo_crop.crop_array

    conn = db.init_db()
    datasets = sorted(keep["dataset"].unique())
    for d in datasets:
        conn.execute("DELETE FROM embeddings WHERE sample_id IN "
                     "(SELECT id FROM samples WHERE dataset=?)", (d,))
        conn.execute("DELETE FROM samples WHERE dataset=?", (d,))
    conn.commit()

    n_ok = n_missing = n_err = 0
    for _, r in keep.iterrows():
        src = os.path.join(args.img_root, *r["rel"])
        if not os.path.exists(src):
            n_missing += 1
            continue
        try:
            a = np.array(Image.open(src).convert("L"), dtype=np.uint8)
            if crop_array is not None:
                a = crop_array(a)
            im = Image.fromarray(a).resize((CROP_SIZE, CROP_SIZE), Image.BILINEAR)
            out_name = f"{r['stem']}_{r['Side']}_{r['View']}.jpg"
            out_path = os.path.join(args.out_dir, out_name)
            im.save(out_path, quality=95)
        except Exception as e:  # noqa: BLE001
            print(f"  [err] {src}: {e}")
            n_err += 1
            continue
        conn.execute(
            "INSERT OR IGNORE INTO samples "
            "(file_path,file_name,dataset,patient_id,laterality,view,abnormality,"
            " pathology,binary_label,case_id,split) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (out_path, out_name, r["dataset"], r["stem"], r["Side"], r["View"], None,
             str(r["Status"]).upper(), int(r["binary_label"]), r["case_id"], r["split"]))
        n_ok += 1
        if n_ok % 200 == 0:
            conn.commit()
            print(f"\r  cropped+ingested {n_ok}", end="", flush=True)
    conn.commit()
    print(f"\r  cropped+ingested {n_ok}      ")
    print(f"Mini-DDSM ingest: ok={n_ok} missing={n_missing} err={n_err} datasets={datasets}")

    for d in datasets:
        print(f" [{d}]")
        for r in conn.execute(
                "SELECT split,pathology,COUNT(*) c, COUNT(DISTINCT case_id) cases,"
                " COUNT(DISTINCT patient_id) pats FROM samples WHERE dataset=? "
                "GROUP BY split,pathology ORDER BY split,pathology", (d,)):
            print(f"   {r['split']:5s} {r['pathology']:9s} imgs={r['c']:4d} "
                  f"cases={r['cases']:4d} pats={r['pats']:4d}")
        # leak check: no patient in both splits (within dataset)
        bad = conn.execute(
            "SELECT COUNT(*) c FROM (SELECT patient_id FROM samples WHERE dataset=? "
            "GROUP BY patient_id HAVING COUNT(DISTINCT split)>1)", (d,)).fetchone()["c"]
        print(f"   patient-split leak (must be 0): {bad}")
    # cross-dataset patient collision (stems must be globally unique)
    xbad = conn.execute(
        "SELECT COUNT(*) c FROM (SELECT patient_id FROM samples WHERE dataset IN "
        "('miniddsm','miniddsm_normal') GROUP BY patient_id HAVING "
        "COUNT(DISTINCT dataset)>1)").fetchone()["c"]
    print(f" cross-dataset patient collision (must be 0): {xbad}")


if __name__ == "__main__":
    main()
