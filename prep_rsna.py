"""Prepare an RSNA Screening Mammography subset as TRAINING-ONLY data for the INbreast transfer
experiments (RSNA is never used as a test set: MammoScreen was fine-tuned on an unpublished 80%
of RSNA).

Source: Kaggle competition train.csv (labels) + Theo Viel's 512x512 PNG export (CC0) of the same
images, both in F:/Downloads/rsna. Subset: every patient with a cancer (biopsy-proven, 486) plus
N_NEG randomly drawn cancer-free patients (seed 0); CC and MLO views only.
Preprocessing identical to Mini-DDSM: breast crop with ianpan/mammo-crop, then 512x512 grayscale
JPEG. Ingested as dataset='rsna_sub', patient_id='rsna_<id>', case = patient + laterality,
label = cancer, split='train' (all of it is training data).

Run: python prep_rsna.py [--limit 20]
"""
import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image

import db

CSV = r"F:\Downloads\rsna\train.csv"
PNG = r"F:\Downloads\rsna\png512"
OUT = r"F:\Development\MasterNotebooks\preproc_jpg_rsna"
N_NEG = 2500
DATASET = "rsna_sub"


def select():
    d = pd.read_csv(CSV)
    d = d[d.view.isin(["CC", "MLO"])]
    pos = d.groupby("patient_id").cancer.max()
    rng = np.random.default_rng(0)
    neg_pool = np.array(sorted(pos[pos == 0].index))
    keep = set(pos[pos == 1].index) | set(rng.choice(neg_pool, N_NEG, replace=False).tolist())
    return d[d.patient_id.isin(keep)].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int); args = ap.parse_args()
    sel = select()
    if args.limit:
        sel = sel.iloc[:args.limit]
    print(f"images {len(sel)}, patients {sel.patient_id.nunique()}, cancer patients "
          f"{sel[sel.cancer == 1].patient_id.nunique()}, cancer images {int(sel.cancer.sum())}", flush=True)
    import mammo_crop
    os.makedirs(OUT, exist_ok=True)
    conn = db.init_db()
    conn.execute(f"DELETE FROM embeddings WHERE sample_id IN (SELECT id FROM samples WHERE dataset='{DATASET}')")
    conn.execute(f"DELETE FROM samples WHERE dataset='{DATASET}'"); conn.commit()
    ok = err = 0
    for _, r in sel.iterrows():
        src = os.path.join(PNG, f"{r.patient_id}_{r.image_id}.png")
        side = "LEFT" if r.laterality == "L" else "RIGHT"
        try:
            a = np.array(Image.open(src).convert("L"), dtype=np.uint8)
            a = mammo_crop.crop_array(a)
            out = os.path.join(OUT, f"{r.patient_id}_{r.image_id}.jpg")
            Image.fromarray(a).resize((512, 512), Image.BILINEAR).save(out, quality=95)
        except Exception as e:  # noqa: BLE001
            print(f"  [err] {src}: {e}"); err += 1; continue
        conn.execute("INSERT OR IGNORE INTO samples (file_path,file_name,dataset,patient_id,laterality,view,"
                     "abnormality,pathology,binary_label,case_id,split) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                     (out, os.path.basename(out), DATASET, f"rsna_{r.patient_id}", side, r["view"], None,
                      "CANCER" if r.cancer else "NORMAL", int(r.cancer), f"rsna_{r.patient_id}_{side}", "train"))
        ok += 1
        if ok % 500 == 0:
            conn.commit(); print(f"  {ok}", flush=True)
    conn.commit()
    print(f"RSNA subset ingested: ok={ok} err={err}")


if __name__ == "__main__":
    main()
