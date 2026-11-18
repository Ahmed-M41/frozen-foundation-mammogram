"""INbreast through the SAME pipeline as the RSNA subset (and Mini-DDSM): breast crop with
ianpan/mammo-crop, then 512x512 grayscale JPEG. Tests whether the RSNA-trained regimes lose
accuracy on INbreast because of a preprocessing mismatch (RSNA: cropped, 512 px; inbreast_full:
uncropped, native resolution). Copies every inbreast_full row (same patient_id, case_id, labels)
into dataset='inbreast_512'. Never load inbreast_512 together with inbreast_full (duplicates).
Run: python prep_inbreast512.py"""
import os

import numpy as np
from PIL import Image

import db

OUT = r"F:\Development\MasterNotebooks\preproc_jpg_inbreast512"
DATASET = "inbreast_512"
COLS = "laterality,view,abnormality,pathology,binary_label,case_id,split,patient_id"


def main():
    import mammo_crop
    os.makedirs(OUT, exist_ok=True)
    conn = db.init_db()
    conn.execute(f"DELETE FROM embeddings WHERE sample_id IN (SELECT id FROM samples WHERE dataset='{DATASET}')")
    conn.execute(f"DELETE FROM samples WHERE dataset='{DATASET}'"); conn.commit()
    rows = conn.execute(f"SELECT file_path,{COLS} FROM samples WHERE dataset='inbreast_full'").fetchall()
    for r in rows:
        a = np.array(Image.open(r[0]).convert("L"), dtype=np.uint8)
        a = mammo_crop.crop_array(a)
        out = os.path.join(OUT, os.path.splitext(os.path.basename(r[0]))[0] + ".jpg")
        Image.fromarray(a).resize((512, 512), Image.BILINEAR).save(out, quality=95)
        conn.execute(f"INSERT INTO samples (file_path,file_name,dataset,{COLS}) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                     (out, os.path.basename(out), DATASET, *tuple(r)[1:]))
    conn.commit()
    print("inbreast_512 ingested:", len(rows))


if __name__ == "__main__":
    main()
