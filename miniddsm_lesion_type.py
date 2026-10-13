"""Parse DDSM .OVERLAY files to tag each Mini-DDSM lesion breast as
mass / calc / mixed, and write it into samples.abnormality (dataset='miniddsm').

Lets us restrict benign-vs-malignant to MASS-ONLY cases for a fair comparison to
our CBIS work (which is mass-only, abnormality='mass', 932 cases).

Each .OVERLAY (e.g. Cancer/0001/C_0001_1.RIGHT_CC.OVERLAY) has one or more
'ABNORMALITY' blocks, each with a line 'LESION_TYPE MASS ...' or
'LESION_TYPE CALCIFICATION ...'. A breast case = (stem, Side); its type is the
set of lesion types across its (up to 2) view OVERLAYs:
  all MASS -> 'mass' ; all CALCIFICATION -> 'calc' ; both -> 'mixed'.

Run:  python miniddsm_lesion_type.py --img-root "<...>/MINI-DDSM-Complete-JPEG-8"
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import db


def lesion_types_in(path):
    out = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = re.match(r"\s*LESION_TYPE\s+(\w+)", line)
            if m:
                t = m.group(1).upper()
                out.append("mass" if t == "MASS" else
                            "calc" if t.startswith("CALC") else t.lower())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-root", required=True)
    args = ap.parse_args()

    case_types = defaultdict(set)     # (stem, Side) -> {mass, calc, ...}
    for status in ("Benign", "Cancer"):
        for ov in glob.glob(os.path.join(args.img_root, status, "*", "*.OVERLAY")):
            base = os.path.basename(ov)                 # C_0001_1.RIGHT_CC.OVERLAY
            parts = base.split(".")
            stem = parts[0]                             # C_0001_1
            side = parts[1].split("_")[0]              # RIGHT
            for t in lesion_types_in(ov):
                case_types[(stem, side)].add(t)

    def classify(ts):
        ts = {t for t in ts if t in ("mass", "calc")}
        if ts == {"mass"}:
            return "mass"
        if ts == {"calc"}:
            return "calc"
        if ts == {"mass", "calc"}:
            return "mixed"
        return None

    conn = db.connect()
    counts = defaultdict(int)
    n_upd = n_nolabel = 0
    for (stem, side), ts in case_types.items():
        abn = classify(ts)
        if abn is None:
            n_nolabel += 1
            continue
        cid = f"miniddsm:{stem}:{side}"
        cur = conn.execute("UPDATE samples SET abnormality=? WHERE dataset='miniddsm' "
                           "AND case_id=?", (abn, cid))
        if cur.rowcount:
            n_upd += cur.rowcount
            counts[abn] += 1
    conn.commit()

    print(f"OVERLAY-tagged cases: {sum(counts.values())}  (rows updated: {n_upd}, "
          f"unclassified: {n_nolabel})")
    for k in ("mass", "calc", "mixed"):
        print(f"   {k:6s}: {counts[k]} cases")
    # coverage vs ingested lesion cases + mass benign/malignant split
    print(" ingested miniddsm cases by abnormality x label:")
    for r in conn.execute(
            "SELECT abnormality, binary_label, COUNT(DISTINCT case_id) c "
            "FROM samples WHERE dataset='miniddsm' GROUP BY abnormality,binary_label "
            "ORDER BY abnormality,binary_label"):
        lab = "malignant" if r["binary_label"] else "benign"
        print(f"   abn={str(r['abnormality']):6s} {lab:9s} cases={r['c']}")


if __name__ == "__main__":
    main()
