"""Summarize results from the experiments table into leaderboards.

  python report.py                 # case-level leaderboard + honest (CV) picks
  python report.py --level image   # per-view image-level leaderboard
"""
import argparse

import db

TARGET_CBIS_CASE = 0.8333   # prior TRAINED 4-model case-level accuracy (to beat)
TARGET_INBREAST = 0.9659    # prior TRAINED InBreast zero-shot accuracy


def _fmt(r):
    return (f"acc={r['accuracy']:.4f} f1={r['f1']:.4f} auc={r['auc']:.4f} "
            f"sens={r['sensitivity']:.4f} spec={r['specificity']:.4f}")


def leaderboard(conn, level="case", limit=25):
    print(f"\n=== {level.upper()}-LEVEL LEADERBOARD (top {limit} by accuracy) ===")
    rows = conn.execute(
        "SELECT * FROM experiments WHERE level=? ORDER BY accuracy DESC, auc DESC LIMIT ?",
        (level, limit)).fetchall()
    for r in rows:
        tag = "" if r["variant"] == "baseline" else f" [{r['variant']}]"
        print(f"  {r['accuracy']:.4f} auc={r['auc']:.4f} f1={r['f1']:.4f} "
              f"| {r['embedder_name']:28s} k={r['k']:<3} {r['distance']:6s} {r['head']:8s} "
              f"g={r['gallery_view']}->q={r['query_view']} n={r['n_test']} {r['dataset_scope']}{tag}")


def per_encoder_best(conn, level="case"):
    print(f"\n=== BEST {level.upper()} PER ENCODER (by AUC, single encoders) ===")
    rows = conn.execute(
        "SELECT embedder_name, MAX(auc) mauc FROM experiments "
        "WHERE level=? AND embedder_name NOT LIKE 'ens_%' "
        "GROUP BY embedder_name ORDER BY mauc DESC", (level,)).fetchall()
    for er in rows:
        b = conn.execute(
            "SELECT * FROM experiments WHERE level=? AND embedder_name=? "
            "ORDER BY auc DESC, accuracy DESC LIMIT 1", (level, er["embedder_name"])).fetchone()
        print(f"  {b['embedder_name']:28s} {_fmt(b)} (k={b['k']} {b['distance']} {b['head']})")


def honest_cv(conn):
    rows = conn.execute(
        "SELECT * FROM experiments WHERE variant='cv_selected' AND level='case' "
        "ORDER BY accuracy DESC").fetchall()
    if not rows:
        return
    print("\n=== HONEST CV-SELECTED CASE-LEVEL (config chosen on train only) ===")
    for r in rows:
        flag = "  <-- beats prior 83.33%" if r["accuracy"] > TARGET_CBIS_CASE else ""
        print(f"  {r['embedder_name']:28s} {_fmt(r)} (k={r['k']} {r['distance']} {r['head']}){flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="case", choices=["case", "image"])
    ap.add_argument("--limit", type=int, default=25)
    args = ap.parse_args()
    conn = db.connect()
    n = conn.execute("SELECT COUNT(*) c FROM experiments").fetchone()["c"]
    print(f"experiments recorded: {n}")
    print(f"TARGETS to beat — CBIS case-level acc {TARGET_CBIS_CASE:.4f}, "
          f"InBreast zero-shot acc {TARGET_INBREAST:.4f}")
    leaderboard(conn, args.level, args.limit)
    per_encoder_best(conn, args.level)
    honest_cv(conn)


if __name__ == "__main__":
    main()
