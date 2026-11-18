"""Rebuild INbreast patient and breast grouping (the public metadata has Patient ID 'removed').

INbreast.xls lists images exam by exam: consecutive rows of one woman share the acquisition
date and breast density (ACR) and cover each laterality/view at most once. Patient identifiers
are anonymised, so a new exam is started only on certain evidence: a new acquisition date, or
the same laterality/view again. This yields 116 exams against the documented 115 cases (at most
one woman split in two); all INbreast splits are by these exams.

  python inbreast_groups.py            # dry run: print the grouping check
  python inbreast_groups.py --apply    # back up the database, then write patient_id/case_id
                                       # (case = patient + laterality, i.e. one breast)
"""
import os
import shutil
import sys
from collections import Counter

import pandas as pd

import config
import db

XLS = r"F:\Development\MasterNotebooks\inbreast\INbreast.xls"
MAX_GAP = 2000


def groups():
    d = pd.read_excel(XLS).dropna(subset=["File Name"])
    d["fn"] = d["File Name"].astype(int)
    out, cur, seen, prev = {}, 0, set(), None
    for _, r in d.iterrows():
        key = (r["Laterality"], r["View"])
        # split only on certain evidence of a new exam: a new acquisition date, or the same
        # laterality/view again (ACR has a missing value; file-number gaps occur inside exams)
        if prev is not None and (r["Acquisition date"] != prev["Acquisition date"] or key in seen):
            cur += 1; seen = set()
        seen.add(key); prev = r
        out[str(r["fn"])] = (f"inbreast_p{cur:03d}", r["Laterality"])
    return out


def main():
    g = groups()
    sizes = Counter(Counter(p for p, _ in g.values()).values())
    n_pat = len({p for p, _ in g.values()})
    print(f"images {len(g)}, patients {n_pat}, patients by image count {dict(sorted(sizes.items()))}")
    ok = len(g) == 410 and 115 <= n_pat <= 118          # documented: 115 cases
    print("410 images, close to the documented 115 cases:", ok)
    if "--apply" not in sys.argv:
        return
    if not ok:
        sys.exit("grouping does not match the documented composition; not applied")
    src = config.DB_PATH if hasattr(config, "DB_PATH") else None
    if src and os.path.exists(src):
        shutil.copy2(src, src + ".bak_before_inbreast_groups")
        print("backup:", src + ".bak_before_inbreast_groups")
    conn = db.connect()
    rows = conn.execute("SELECT id, file_name FROM samples WHERE dataset IN ('inbreast','inbreast_full')").fetchall()
    n = 0
    for r in rows:
        fn = r["file_name"].split(".")[0]
        if fn in g:
            p, lat = g[fn]
            side = "RIGHT" if lat == "R" else "LEFT"
            conn.execute("UPDATE samples SET patient_id=?, case_id=? WHERE id=?", (p, f"{p}_{side}", r["id"]))
            n += 1
    conn.commit()
    print("updated rows:", n)


if __name__ == "__main__":
    main()
