"""Prepare & ingest mini-MIAS (film mammography) as an extra database source.

MIAS = 322 digitized-film images (1024x1024 PGM), ALL MLO view, no CC/MLO pairing.
Info.txt line: REFNUM BG CLASS [SEVERITY X Y RADIUS]; NORM lines have no severity.
Per image: malignant(1) if ANY abnormality is 'M', else benign(0). We keep only
ABNORMAL images (benign+malignant) to match CBIS's biopsied composition and drop
the 207 NORM images. Ingested as dataset='mias_full', split='train' (augmentation
only), one case per image (like inbreast_full).

Run:  python prep_mias.py
"""
import os

import db

MIAS_DIR = (r"C:\Users\pc\AppData\Local\Temp\claude\F--Development-MamoRAC"
            r"\b46da560-ebb4-476e-933e-9d4079dd7234\scratchpad\mias\all-mias")
INFO = os.path.join(MIAS_DIR, "Info.txt")
DATASET = "mias_full"


def parse_info():
    """refnum -> dict(severity in {B,M} or None, classes set). Only mdb lines."""
    rec = {}
    with open(INFO) as f:
        for line in f:
            parts = line.split()
            if not parts or not parts[0].startswith("mdb"):
                continue
            ref = parts[0]
            cls = parts[2] if len(parts) > 2 else "NORM"
            sev = parts[3] if len(parts) > 3 and parts[3] in ("B", "M") else None
            r = rec.setdefault(ref, {"sev": set(), "cls": set()})
            r["cls"].add(cls)
            if sev:
                r["sev"].add(sev)
    return rec


def main():
    rec = parse_info()
    conn = db.init_db()
    conn.execute("DELETE FROM embeddings WHERE sample_id IN "
                 "(SELECT id FROM samples WHERE dataset=?)", (DATASET,))
    conn.execute("DELETE FROM samples WHERE dataset=?", (DATASET,))
    conn.commit()

    n_b = n_m = n_norm = missing = 0
    for i, (ref, r) in enumerate(sorted(rec.items())):
        if "NORM" in r["cls"] and not r["sev"]:
            n_norm += 1
            continue                                   # drop normals
        label = 1 if "M" in r["sev"] else 0            # malignant if any M
        fp = os.path.join(MIAS_DIR, f"{ref}.pgm")
        if not os.path.exists(fp):
            missing += 1
            continue
        lat = "RIGHT" if int(ref[3:]) % 2 == 1 else "LEFT"   # MIAS pairs (nominal)
        conn.execute(
            "INSERT OR IGNORE INTO samples "
            "(file_path,file_name,dataset,patient_id,laterality,view,abnormality,"
            " pathology,binary_label,case_id,split) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (fp, f"{ref}.pgm", DATASET, f"mias_{ref}", lat, "MLO",
             ",".join(sorted(r["cls"])), "MALIGNANT" if label else "BENIGN",
             label, f"mias_{ref}", "train"))
        n_m += label
        n_b += (1 - label)
    conn.commit()
    print(f"MIAS ingested: benign={n_b} malignant={n_m} (dropped {n_norm} NORM, "
          f"missing={missing})  dataset={DATASET}")
    rows = conn.execute("SELECT view,binary_label,COUNT(*) c FROM samples WHERE dataset=? "
                        "GROUP BY view,binary_label", (DATASET,)).fetchall()
    for r in rows:
        print(f"   view={r['view']} label={r['binary_label']} n={r['c']}")


if __name__ == "__main__":
    main()
