"""Flexible best-config evaluator: any encoder set, fusion head, CC/MLO fusion
method, gallery/query (split, dataset), and orientation handling — all training-free.

Orientation (--orient):
  none    plain (E) gallery and query.
  mirror  gallery = E + E__mir (both laterality orientations); query = E.
  canon   gallery and query both use E__canon (all breasts flipped to one side).
(Requires augment_embed.py to have produced the __mir / __canon embeddings.)

Examples:
  python eval_best.py --encoders dinov2_vitb14_noclahe dinov2_vitl14_noclahe rad_dino_noclahe --method max --orient mirror
  python eval_best.py --encoders mammo_clip_b5 --g-ds cbis_full --q-ds cbis_full --method max --orient canon
"""
import argparse

import numpy as np

import config
import db
import fusion
import evaluate
from retrieve import RetrievalIndex


def _l2(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def _stack_gallery(conn, names, split, view, ds, l2=False):
    """Stack (vstack) train rows across embedder-name variants (orientation)."""
    ids, vecs, labs = [], [], []
    for nm in names:
        i, v, l = db.load_embeddings(conn, nm, split=split, view=view, dataset=ds)
        if len(i) == 0:
            return None, None, None
        ids.append(i)
        vecs.append(_l2(v) if l2 else v)
        labs.append(l)
    return np.concatenate(ids), np.concatenate(vecs, 0), np.concatenate(labs)


def _view_branches(conn, encoders, k, distance, head, view, g_split, g_ds,
                   q_split, q_ds, mode, gal_suffixes=("",), qry_suffix="",
                   g_view=None):
    """One retrieval branch: query = test images of `view`; gallery = train images
    of `g_view` (None -> same as query view; 'all' -> both views = the ALL-model
    analog). g_view='all' with the two query views gives the old 4-way ensemble."""
    gview = view if g_view is None else (None if g_view == "all" else g_view)
    if mode == "concat":
        g_parts = []
        for suf in gal_suffixes:
            per_enc, ids_s, lab_s = [], None, None
            for e in encoders:
                i, v, l = db.load_embeddings(conn, e + suf, split=g_split, view=gview, dataset=g_ds)
                if len(i) == 0:
                    return []
                if ids_s is None:
                    ids_s, lab_s = i, l
                else:
                    assert np.array_equal(i, ids_s), f"gallery misalign {e}{suf}"
                per_enc.append(_l2(v))
            g_parts.append((ids_s, np.concatenate(per_enc, 1), lab_s))
        g_ids = np.concatenate([p[0] for p in g_parts])
        g_vecs = np.concatenate([p[1] for p in g_parts], 0)
        g_lab = np.concatenate([p[2] for p in g_parts])
        q_per, q_ids = [], None
        for e in encoders:
            i, v, _ = db.load_embeddings(conn, e + qry_suffix, split=q_split, view=view, dataset=q_ds)
            if len(i) == 0:
                return []
            if q_ids is None:
                q_ids = i
            else:
                assert np.array_equal(i, q_ids), f"query misalign {e}"
            q_per.append(_l2(v))
        q_vecs = np.concatenate(q_per, 1)
        index = RetrievalIndex(g_ids, g_vecs, g_lab, distance=distance)
        assert not (set(q_ids.tolist()) & set(g_ids.tolist())), "DATA LEAK"
        sc, _, nl = index.search(q_vecs, min(k, len(g_ids)))
        _, sp = fusion.fuse(sc, nl, distance, head)
        return [dict(ids=q_ids, score_pos=sp)]

    # late: one branch per encoder
    out = []
    for e in encoders:
        g_names = [e + suf for suf in gal_suffixes]
        g_ids, g_vecs, g_lab = _stack_gallery(conn, g_names, g_split, gview, g_ds)
        if g_ids is None:
            continue
        qi, qv, _ = db.load_embeddings(conn, e + qry_suffix, split=q_split, view=view, dataset=q_ds)
        if len(qi) == 0:
            continue
        index = RetrievalIndex(g_ids, g_vecs, g_lab, distance=distance)
        assert not (set(qi.tolist()) & set(g_ids.tolist())), "DATA LEAK"
        sc, _, nl = index.search(qv, min(k, len(g_ids)))
        _, sp = fusion.fuse(sc, nl, distance, head)
        out.append(dict(ids=qi, score_pos=sp))
    return out


_ORIENT = {"none": (("",), ""), "mirror": (("", "__mir"), ""), "canon": (("__canon",), "__canon")}


def evaluate_config(conn, encoders, k=40, distance="cosine", head="tip",
                    method="max", mode="late", g_split="train", g_ds="cbis",
                    q_split="test", q_ds="cbis", orient="none", four_way=False):
    gal_suffixes, qry_suffix = _ORIENT[orient]
    branches = []
    for view in config.VIEWS:            # specialists: CC-gallery->CC, MLO-gallery->MLO
        branches.extend(_view_branches(conn, encoders, k, distance, head, view,
                                       g_split, g_ds, q_split, q_ds, mode,
                                       gal_suffixes, qry_suffix, g_view=None))
    if four_way:                          # ALL-model analog: all-gallery->CC, all-gallery->MLO
        for view in config.VIEWS:
            branches.extend(_view_branches(conn, encoders, k, distance, head, view,
                                           g_split, g_ds, q_split, q_ds, mode,
                                           gal_suffixes, qry_suffix, g_view="all"))
    if not branches:
        return None
    return evaluate.case_fuse(conn, branches, method=method)


def _split_ds(s):
    return [x for x in s.split(",")] if "," in s else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--distance", default="cosine")
    ap.add_argument("--head", default="tip")
    ap.add_argument("--method", default="max", choices=["mean", "max", "min", "noisy_or"])
    ap.add_argument("--mode", default="late", choices=["late", "concat"])
    ap.add_argument("--orient", default="none", choices=["none", "mirror", "canon"])
    ap.add_argument("--four-way", action="store_true",
                    help="add ALL-gallery branches (mimics the old 4-model ensemble)")
    ap.add_argument("--g-split", default="train")
    ap.add_argument("--g-ds", default="cbis")
    ap.add_argument("--q-split", default="test")
    ap.add_argument("--q-ds", default="cbis")
    args = ap.parse_args()

    conn = db.connect()
    cf = evaluate_config(conn, args.encoders, k=args.k, distance=args.distance,
                         head=args.head, method=args.method, mode=args.mode,
                         g_split=args.g_split, g_ds=_split_ds(args.g_ds),
                         q_split=args.q_split, q_ds=_split_ds(args.q_ds),
                         orient=args.orient, four_way=args.four_way)
    if cf is None:
        print("RESULT: FAIL (missing embeddings for this config)")
        return
    m = cf["metrics"]
    bacc = 0.5 * (m["sensitivity"] + m["specificity"])
    print(f"  {args.mode}[{'+'.join(args.encoders)}] {args.method} orient={args.orient} "
          f"four_way={args.four_way} k={args.k} {args.distance} {args.head}")
    print(f"    gallery={args.g_ds}({args.g_split}) query={args.q_ds}({args.q_split}) "
          f"n_cases={cf['n_cases']} (paired={cf['n_paired']})")
    print(f"    ALL:    acc={m['accuracy']:.4f} bal_acc={bacc:.4f} f1={m['f1']:.4f} "
          f"auc={m['auc']:.4f} sens={m['sensitivity']:.4f} spec={m['specificity']:.4f}")
    if cf["paired_metrics"]:
        pm = cf["paired_metrics"]
        pbacc = 0.5 * (pm["sensitivity"] + pm["specificity"])
        print(f"    PAIRED: acc={pm['accuracy']:.4f} bal_acc={pbacc:.4f} f1={pm['f1']:.4f} "
              f"auc={pm['auc']:.4f} sens={pm['sensitivity']:.4f} spec={pm['specificity']:.4f}")


if __name__ == "__main__":
    main()
