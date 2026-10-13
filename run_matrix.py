"""Stage 5: sweep the retrieval grid for every embedded encoder and dump a
results table (case-level + per-view image-level) for analysis.

Reads cached embeddings only — no re-embedding. Each (embedder, k, distance,
head) writes experiment rows via evaluate.run_headline.

NOTE: the "best" printed here is the TEST-optimal config (an oracle upper bound
for landscape analysis). The HONEST headline number comes from select_hparams.py,
which picks the config by CV on the train set only.

Run:
    python run_matrix.py                       # all embedded encoders
    python run_matrix.py --embedder dinov2_vitl14 --four-way
"""
import argparse
import csv
import itertools
import os
import sys

import config
import db
from evaluate import run_headline


def present_embedders(conn):
    rows = conn.execute(
        "SELECT DISTINCT embedder_name FROM embeddings ORDER BY embedder_name"
    ).fetchall()
    return [r["embedder_name"] for r in rows]


def run(embedders, four_way=False, quick=False):
    conn = db.connect()
    ks = [config.DEFAULT_K] if quick else config.K_GRID
    dists = ["cosine"] if quick else config.DISTANCE_GRID
    heads = [config.DEFAULT_HEAD] if quick else config.HEAD_GRID

    results = []
    for emb in embedders:
        present = conn.execute("SELECT COUNT(*) c FROM embeddings WHERE embedder_name=?",
                               (emb,)).fetchone()["c"]
        if present == 0:
            print(f"[skip] no embeddings for {emb}")
            continue
        # idempotent: clear prior baseline experiments for this embedder
        conn.execute("DELETE FROM predictions WHERE experiment_id IN "
                     "(SELECT id FROM experiments WHERE embedder_name=? AND variant='baseline')",
                     (emb,))
        conn.execute("DELETE FROM experiments WHERE embedder_name=? AND variant='baseline'",
                     (emb,))
        conn.commit()

        best_row, best_auc = None, -1.0
        for k, dist, head in itertools.product(ks, dists, heads):
            res = run_headline(conn, emb, k, dist, head, four_way=four_way,
                               persist=True, verbose=False, save_predictions=False)
            if res is None:
                continue
            cm = res["case"]["metrics"]
            pm = res["case"]["paired_metrics"] or {}
            row = dict(embedder=emb, k=k, distance=dist, head=head,
                       n_cases=res["case"]["n_cases"],
                       case_acc=round(cm["accuracy"], 4), case_f1=round(cm["f1"], 4),
                       case_auc=round(cm["auc"], 4),
                       case_sens=round(cm["sensitivity"], 4),
                       case_spec=round(cm["specificity"], 4),
                       paired_acc=round(pm.get("accuracy", 0.0), 4),
                       paired_auc=round(pm.get("auc", 0.0), 4))
            results.append(row)
            if cm["auc"] > best_auc:
                best_auc, best_row = cm["auc"], row
        if best_row:
            print(f"  {emb:16s} best-test case: acc={best_row['case_acc']:.4f} "
                  f"auc={best_row['case_auc']:.4f} (k={best_row['k']} "
                  f"{best_row['distance']} {best_row['head']})")

    if not results:
        print("RESULT: FAIL (no configs ran — embed some encoders first)")
        return False

    out = os.path.join(config.OUTPUT_DIR, "results_matrix.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\n  wrote {len(results)} rows -> {out}")
    best = max(results, key=lambda r: r["case_auc"])
    print(f"  overall best-test (case AUC): {best}")
    print("RESULT: PASS")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", action="append", dest="embedders", default=None)
    ap.add_argument("--four-way", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    conn = db.connect()
    embs = args.embedders or present_embedders(conn)
    ok = run(embs, four_way=args.four_way, quick=args.quick)
    sys.exit(0 if ok else 1)
