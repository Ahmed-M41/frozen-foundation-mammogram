"""Case-level (target) evaluation with an HONESTLY tuned decision threshold.

The malignant class is the minority, so the default 0.5 threshold under-calls
malignancy and understates accuracy. We pick the threshold on the TRAIN set only:
each train case is scored by leave-one-out retrieval that excludes same-PATIENT
gallery neighbours (no leakage), then we choose the threshold maximizing balanced
accuracy. That threshold is applied once to the official test set.

Reports CASE-LEVEL metrics only (full set and the paired CC+MLO subset). The
case-label rule (any-view-malignant => case malignant) is already enforced in the DB.
"""
import argparse

import numpy as np
from sklearn.metrics import balanced_accuracy_score

import config
import db
import fusion
import evaluate
from retrieve import RetrievalIndex


def _patient_map(conn, ids):
    meta = db.sample_meta(conn, ids)
    return {int(i): meta[int(i)]["patient_id"] for i in ids}


def _branch(conn, e, k, distance, head, view, g_view, g_split, q_split, ds, loo, pmap):
    gview = None if g_view == "all" else view
    gi, gv, gl = db.load_embeddings(conn, e, split=g_split, view=gview, dataset=ds)
    qi, qv, _ = db.load_embeddings(conn, e, split=q_split, view=view, dataset=ds)
    if len(gi) == 0 or len(qi) == 0:
        return None
    index = RetrievalIndex(gi, gv, gl, distance=distance)
    kk = min(k + 25, len(gi)) if loo else min(k, len(gi))
    sc, nid, nl = index.search(qv, kk)
    if loo:
        rs, rl = [], []
        for r in range(len(qi)):
            qpat = pmap[int(qi[r])]
            keep = np.array([pmap.get(int(n)) != qpat for n in nid[r]])
            s, l = sc[r][keep][:k], nl[r][keep][:k]
            if len(s) < k:  # pad (rare)
                s = np.pad(s, (0, k - len(s))); l = np.pad(l, (0, k - len(l)))
            rs.append(s); rl.append(l)
        sc, nl = np.stack(rs), np.stack(rl)
    else:
        sc, nl = sc[:, :k], nl[:, :k]
    _, sp = fusion.fuse(sc, nl, distance, head)
    return dict(ids=qi, score_pos=sp)


def case_scores(conn, encoders, k, distance, head, method, four_way,
                g_split, q_split, ds, loo):
    ids_all = [int(r["id"]) for r in conn.execute(
        "SELECT id FROM samples WHERE dataset=? AND split IN (?,?)",
        (ds, g_split, q_split)).fetchall()]
    pmap = _patient_map(conn, ids_all) if loo else {}
    specs = [(v, None) for v in config.VIEWS]
    if four_way:
        specs += [(v, "all") for v in config.VIEWS]
    branches = []
    for view, gv in specs:
        for e in encoders:
            b = _branch(conn, e, k, distance, head, view, gv, g_split, q_split, ds, loo, pmap)
            if b:
                branches.append(b)
    if not branches:
        return None
    return evaluate.case_fuse(conn, branches, method=method)


def tune_and_eval(conn, encoders, k=40, distance="cosine", head="tip",
                  method="mean", four_way=True, ds="cbis"):
    tr = case_scores(conn, encoders, k, distance, head, method, four_way, "train", "train", ds, loo=True)
    yt, st = tr["y_true"], tr["y_score"]
    thr, best_ba = 0.5, -1
    for t in np.unique(st):
        ba = balanced_accuracy_score(yt, (st >= t).astype(int))
        if ba > best_ba:
            best_ba, thr = ba, float(t)
    te = case_scores(conn, encoders, k, distance, head, method, four_way, "train", "test", ds, loo=False)
    y, s, pidx = te["y_true"], te["y_score"], te["paired_idx"]

    def m(mask):
        yy, ss = y[mask], s[mask]
        d = evaluate.compute_metrics(yy, (ss >= thr).astype(int), ss)
        d["bal_acc"] = 0.5 * (d["sensitivity"] + d["specificity"])
        return d
    return thr, m(np.arange(len(y))), m(pidx), len(pidx)


def _show(name, thr, full, paired, npaired):
    print(f"  {name}  (train-tuned thr={thr:.3f})")
    print(f"    FULL(210):   acc={full['accuracy']:.4f} bal_acc={full['bal_acc']:.4f} "
          f"auc={full['auc']:.4f} sens={full['sensitivity']:.4f} spec={full['specificity']:.4f}")
    print(f"    PAIRED({npaired}): acc={paired['accuracy']:.4f} bal_acc={paired['bal_acc']:.4f} "
          f"auc={paired['auc']:.4f} sens={paired['sensitivity']:.4f} spec={paired['specificity']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--distance", default="cosine")
    ap.add_argument("--head", default="tip")
    ap.add_argument("--method", default="mean", choices=["mean", "max", "min", "noisy_or"])
    ap.add_argument("--four-way", action="store_true")
    ap.add_argument("--ds", default="cbis")
    args = ap.parse_args()
    conn = db.connect()
    thr, full, paired, npaired = tune_and_eval(
        conn, args.encoders, k=args.k, distance=args.distance, head=args.head,
        method=args.method, four_way=args.four_way, ds=args.ds)
    name = f"{'4way' if args.four_way else '2way'} {args.method} [{'+'.join(args.encoders)}]"
    _show(name, thr, full, paired, npaired)
