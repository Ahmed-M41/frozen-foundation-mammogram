"""Stage 1: read the official split CSVs and populate the `samples` table.

CBIS-DDSM uses the dataset's ORIGINAL train/test split (patient-disjoint,
verified 0 patient overlap for mass-only). No re-splitting, no k-fold. InBreast
is optional and can be added either as extra gallery (train) data or as an
external zero-shot test set.

Run:
    python ingest.py                         # CBIS mass only (headline)
    python ingest.py --inbreast gallery      # + InBreast added to the gallery
    python ingest.py --inbreast test         # + InBreast as external test set
    python ingest.py --reset                 # wipe samples/embeddings first
"""
import argparse
import os
import sys

import pandas as pd

import config
import db


def _norm_lat(v):
    v = str(v).strip().upper()
    if v in ("L", "LEFT"):
        return "LEFT"
    if v in ("R", "RIGHT"):
        return "RIGHT"
    return v


def _insert(conn, **row):
    conn.execute(
        "INSERT OR IGNORE INTO samples "
        "(file_path,file_name,dataset,patient_id,laterality,view,abnormality,"
        " pathology,binary_label,case_id,split) "
        "VALUES (:file_path,:file_name,:dataset,:patient_id,:laterality,:view,"
        ":abnormality,:pathology,:binary_label,:case_id,:split)",
        row,
    )


def ingest_cbis(conn, csv_path, split, dataset="cbis"):
    """One sample per unique preprocessed image; label = malignant if ANY
    abnormality on that image is malignant."""
    df = pd.read_csv(csv_path)
    df["cancer"] = df["cancer"].astype(int)
    agg = df.groupby("image_path", as_index=False).agg({
        "cancer": "max",
        "patient_id": "first",
        "left or right breast": "first",
        "image view": "first",
        "abnormality type": "first",
        "pathology": "first",
    })
    n, missing = 0, 0
    for _, r in agg.iterrows():
        fp = str(r["image_path"])
        if not os.path.exists(fp):
            missing += 1
            continue
        lat = _norm_lat(r["left or right breast"])
        view = str(r["image view"]).strip().upper()
        abn = str(r["abnormality type"]).strip().lower()
        patient = str(r["patient_id"]).strip()
        _insert(conn, file_path=fp, file_name=os.path.basename(fp), dataset=dataset,
                patient_id=patient, laterality=lat, view=view, abnormality=abn,
                pathology=str(r["pathology"]), binary_label=int(r["cancer"]),
                case_id=f"{dataset}_{patient}_{lat}_{abn}", split=split)
        n += 1
    conn.commit()
    return n, missing


def ingest_cbis_full(conn, csv_path, split, dataset="cbis_full"):
    """Same cases as `cbis` but pointing at the ORIGINAL full-mammogram jpeg
    ('image file path') instead of the YOLO-cropped preproc — used to give
    Mammo-CLIP its native full-image input (no crop, no square-squash)."""
    df = pd.read_csv(csv_path)
    df["cancer"] = df["cancer"].astype(int)
    agg = df.groupby("image file path", as_index=False).agg({
        "cancer": "max", "patient_id": "first", "left or right breast": "first",
        "image view": "first", "abnormality type": "first", "pathology": "first",
    })
    n, missing = 0, 0
    for _, r in agg.iterrows():
        fp = str(r["image file path"])
        if not os.path.exists(fp):
            missing += 1
            continue
        lat = _norm_lat(r["left or right breast"])
        view = str(r["image view"]).strip().upper()
        abn = str(r["abnormality type"]).strip().lower()
        patient = str(r["patient_id"]).strip()
        _insert(conn, file_path=fp, file_name=os.path.basename(fp), dataset=dataset,
                patient_id=patient, laterality=lat, view=view, abnormality=abn,
                pathology=str(r["pathology"]), binary_label=int(r["cancer"]),
                case_id=f"{dataset}_{patient}_{lat}_{abn}", split=split)
        n += 1
    conn.commit()
    return n, missing


def ingest_inbreast_full(conn, csv_path, split="train", dataset="inbreast_full"):
    """InBreast NATIVE full-resolution PNGs (to match Mammo-CLIP native input),
    for the cross-dataset experiments. Patient id anonymized -> synthetic per-image
    id for CV grouping; each image is its own case (no CC/MLO pairing available)."""
    df = pd.read_csv(csv_path)
    df["cancer"] = df["cancer"].astype(int)
    n, missing = 0, 0
    for _, r in df.iterrows():
        fp = str(r["full_img_path"])
        if not os.path.exists(fp):
            missing += 1
            continue
        lat = _norm_lat(r["Laterality"])
        view = str(r["image view"]).strip().upper()
        fname = str(r["File Name"]).strip()
        _insert(conn, file_path=fp, file_name=os.path.basename(fp), dataset=dataset,
                patient_id=f"inb_{fname}", laterality=lat, view=view, abnormality=None,
                pathology=f"BIRADS_{r['Bi-Rads']}", binary_label=int(r["cancer"]),
                case_id=f"inbreast_{fname}", split=split)
        n += 1
    conn.commit()
    return n, missing


def ingest_inbreast(conn, csv_path, split, dataset="inbreast"):
    """InBreast: patient id is anonymized, so case_id is per-image (no CC/MLO
    pairing here — image-level and external-test use only)."""
    df = pd.read_csv(csv_path)
    df["cancer"] = df["cancer"].astype(int)
    n, missing = 0, 0
    for _, r in df.iterrows():
        fp = str(r["image_path"])
        if not os.path.exists(fp):
            missing += 1
            continue
        lat = _norm_lat(r["Laterality"])
        view = str(r["image view"]).strip().upper()
        fname = str(r["File Name"]).strip()
        _insert(conn, file_path=fp, file_name=os.path.basename(fp), dataset=dataset,
                patient_id="NA", laterality=lat, view=view, abnormality=None,
                pathology=f"BIRADS_{r['Bi-Rads']}", binary_label=int(r["cancer"]),
                case_id=f"inbreast_{fname}", split=split)
        n += 1
    conn.commit()
    return n, missing


def enforce_case_labels(conn):
    """Case-consistency rule (user): a breast case is malignant if ANY of its
    views/images is malignant -> overwrite every benign image of that case to
    malignant. Returns how many image labels were flipped."""
    flipped = conn.execute(
        "SELECT COUNT(*) c FROM samples s WHERE s.binary_label=0 AND EXISTS "
        "(SELECT 1 FROM samples s2 WHERE s2.case_id=s.case_id AND s2.binary_label=1)"
    ).fetchone()["c"]
    conn.execute(
        "UPDATE samples SET binary_label=1 WHERE binary_label=0 AND case_id IN "
        "(SELECT case_id FROM samples WHERE binary_label=1)")
    conn.commit()
    return flipped


def _report(conn):
    def q(sql, *a):
        return conn.execute(sql, a).fetchall()
    total = q("SELECT COUNT(*) c FROM samples")[0]["c"]
    print(f"  total samples: {total}")
    print("  by split/dataset/view/class:")
    for r in q("SELECT split,dataset,view,binary_label,COUNT(*) c FROM samples "
               "GROUP BY split,dataset,view,binary_label ORDER BY split,dataset,view,binary_label"):
        print(f"    split={r['split']:5s} ds={r['dataset']:8s} view={str(r['view']):4s} "
              f"label={r['binary_label']} n={r['c']}")
    print("  cases (test, cbis):")
    rows = q("SELECT COUNT(DISTINCT case_id) c FROM samples WHERE split='test' AND dataset='cbis'")
    print(f"    distinct test cbis cases: {rows[0]['c']}")
    # leakage guard: no patient in both train and test (cbis)
    leak = q("SELECT COUNT(*) c FROM (SELECT patient_id FROM samples WHERE split='train' AND dataset='cbis' "
             "INTERSECT SELECT patient_id FROM samples WHERE split='test' AND dataset='cbis')")
    print(f"  LEAKAGE CHECK (cbis train/test patient overlap): {leak[0]['c']} "
          f"{'OK' if leak[0]['c'] == 0 else 'LEAK!'}")
    return total, leak[0]["c"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inbreast", choices=["none", "gallery", "test"], default="none",
                    help="add InBreast as extra gallery (train) or as external test set")
    ap.add_argument("--cbis-full", action="store_true",
                    help="also ingest cbis_full (original full-mammogram jpegs) for Mammo-CLIP native input")
    ap.add_argument("--inbreast-full", action="store_true",
                    help="also ingest inbreast_full (native full PNGs) for cross-dataset experiments")
    ap.add_argument("--reset", action="store_true", help="wipe samples+embeddings first")
    args = ap.parse_args()

    conn = db.init_db()
    if args.reset:
        conn.execute("DELETE FROM embeddings")
        conn.execute("DELETE FROM samples")
        conn.commit()
        print("  reset: cleared samples + embeddings")

    ntr, mtr = ingest_cbis(conn, config.CBIS_TRAIN_CSV, "train")
    nte, mte = ingest_cbis(conn, config.CBIS_TEST_CSV, "test")
    print(f"  CBIS train inserted={ntr} missing_files={mtr}")
    print(f"  CBIS test  inserted={nte} missing_files={mte}")

    if args.inbreast != "none":
        split = "train" if args.inbreast == "gallery" else "test"
        nib, mib = ingest_inbreast(conn, config.INBREAST_CSV, split)
        print(f"  InBreast ({args.inbreast}, split={split}) inserted={nib} missing_files={mib}")

    if args.cbis_full:
        nft, mft = ingest_cbis_full(conn, config.CBIS_TRAIN_CSV, "train")
        nfe, mfe = ingest_cbis_full(conn, config.CBIS_TEST_CSV, "test")
        print(f"  cbis_full train inserted={nft} missing={mft}; test inserted={nfe} missing={mfe}")

    if args.inbreast_full:
        nif, mif = ingest_inbreast_full(conn, config.INBREAST_CSV, "train")
        print(f"  inbreast_full inserted={nif} missing={mif}")

    flipped = enforce_case_labels(conn)
    print(f"  case-label rule: flipped {flipped} benign images -> malignant "
          f"(any-view-malignant => case malignant)")
    db.set_meta(conn, seed=config.SEED, dataset_scope="cbis_mass",
                inbreast=args.inbreast)
    total, leak = _report(conn)
    ok = total > 0 and leak == 0
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
