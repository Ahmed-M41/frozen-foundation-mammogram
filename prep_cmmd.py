"""Prepare & ingest CMMD (digital FFDM, D1 biopsy cohort) as an extra database source.

CMMD = Chinese Mammography Database (GE Senographe, DIGITAL). We use the D1 cohort
only: 1026 patients / 1026 breasts / 2214 images with clean per-breast biopsy
labels (Benign/Malignant), each breast having CC+MLO. D2 (molecular-subtype subset)
is dropped — its mirror labels nearly everything malignant (contralateral breasts
mislabeled). PNGs are already YOLO breast-cropped (Method_crop=Yolo).

View: this PNG mirror has no explicit CC/MLO column; per breast we assign views by
sorted filename order (CC,MLO,CC,MLO...) — a documented heuristic that only affects
the CC/MLO specialist heads mildly (the view-agnostic head is unaffected).

Ingested as dataset='cmmd_full', split='train', patient_id=cmmd_<ID> (both breasts
grouped), case_id=cmmd_<ID>_<L/R> (one breast = one 2-view case).

Run:  python prep_cmmd.py
"""
import os

import pandas as pd

import db

SP = (r"C:\Users\pc\AppData\Local\Temp\claude\F--Development-MamoRAC"
      r"\b46da560-ebb4-476e-933e-9d4079dd7234\scratchpad\cmmd")
IMG_DIR = os.path.join(SP, "CMMD", "Cancer")
CSV = os.path.join(SP, "description.csv")
DATASET = "cmmd_full"


def main():
    df = pd.read_csv(CSV)
    df = df[df["ID"].str.startswith("D1")].copy()        # clean biopsy cohort
    df["base"] = df["Path"].str.split("/").str[-1]        # D1-0033_1-2.png

    conn = db.init_db()
    conn.execute("DELETE FROM embeddings WHERE sample_id IN "
                 "(SELECT id FROM samples WHERE dataset=?)", (DATASET,))
    conn.execute("DELETE FROM samples WHERE dataset=?", (DATASET,))
    conn.commit()

    n_b = n_m = missing = 0
    for (pid, lr), sub in df.groupby(["ID", "LeftRight"]):
        rows = sub.sort_values("base").reset_index(drop=True)
        lat = "LEFT" if lr == "L" else "RIGHT"
        label = 1 if str(rows.iloc[0]["Classification"]).strip() == "Malignant" else 0
        for i, r in rows.iterrows():
            fp = os.path.join(IMG_DIR, r["base"])
            if not os.path.exists(fp):
                missing += 1
                continue
            view = "CC" if i % 2 == 0 else "MLO"          # heuristic per-breast split
            conn.execute(
                "INSERT OR IGNORE INTO samples "
                "(file_path,file_name,dataset,patient_id,laterality,view,abnormality,"
                " pathology,binary_label,case_id,split) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (fp, r["base"], DATASET, f"cmmd_{pid}", lat, view,
                 str(r["Abnormality"]), str(r["Classification"]).upper(),
                 label, f"cmmd_{pid}_{lr}", "train"))
            n_m += label
            n_b += (1 - label)
    conn.commit()
    print(f"CMMD-D1 ingested: benign_imgs={n_b} malignant_imgs={n_m} missing={missing} "
          f"dataset={DATASET}")
    for r in conn.execute("SELECT view,binary_label,COUNT(*) c FROM samples WHERE dataset=? "
                          "GROUP BY view,binary_label", (DATASET,)).fetchall():
        print(f"   view={r['view']} label={r['binary_label']} n={r['c']}")
    nc = conn.execute("SELECT COUNT(DISTINCT case_id) c FROM samples WHERE dataset=?",
                      (DATASET,)).fetchone()["c"]
    npat = conn.execute("SELECT COUNT(DISTINCT patient_id) c FROM samples WHERE dataset=?",
                        (DATASET,)).fetchone()["c"]
    print(f"   distinct cases(breasts)={nc}  patients={npat}")


if __name__ == "__main__":
    main()
