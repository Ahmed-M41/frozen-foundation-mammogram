"""Compute EVERY headline number for the BASE4 early-fusion method (the new paper).

BASE4 = three frozen encoders + MammoScreen with its classification head removed, used
as a fourth frozen feature extractor; a single multi-view logistic probe on the
concatenated features outputs the decision (NO percentile score-fusion, NO weight w).

For CBIS-DDSM (benign/malignant) and Mini-DDSM (normal/cancer), on the same paired-case
test subset and 4-seed protocol as the paper, this reports for four configs:
  zero_shot : MammoScreen scalar score alone            (context / component)
  probe3    : BASE3 probe alone  (3 encoders)           (the old probe, ~0.854/0.973)
  late3     : BASE3 probe + MammoScreen score, w=0.6    (the OLD method, ~0.896/0.961)
  base4     : BASE4 early-fusion probe  (THE NEW METHOD, ~0.896/0.980)

Each config: AUC (mean+-std over seeds); at the ROC-optimal (Youden = max balanced
accuracy) threshold picked on TEST -> acc, bal_acc, sens, spec and summed confusion;
at the honest threshold picked on TRAIN -> bal_acc. Written to cache/base4_numbers.json
for the figures, and printed as a readable table.
"""
import json
import os
import numpy as np
from sklearn.metrics import roc_auc_score

import config, db
import miniddsm_B as M
from data_scaling_extend import head_case_scores, align_B
from data_scaling import _pct
from mil import _best_threshold, _metrics
from exp_msfeat_fusion import load_enc, BASE3, MSFEAT, cbis_ms

W = 0.6
SEEDS = [0, 1, 2, 3]
CFG_ORDER = ["zero_shot", "probe3", "late3", "base4"]
FULL_CFGS = ["probe3", "base4"]


def _scores(kind, Rtr, Mtr, Rte, Mte):
    """Return (train_score, test_score) for a config given probe R and mammoscreen M."""
    if kind == "zero_shot":
        return Mtr, Mte
    if kind == "probe":                       # probe alone (R already the right BASE)
        return Rtr, Rte
    if kind == "late":                        # percentile fusion w=0.6
        return (W * _pct(Rtr, Rtr) + (1 - W) * _pct(Mtr, Mtr),
                W * _pct(Rte, Rtr) + (1 - W) * _pct(Mte, Mtr))
    raise ValueError(kind)


def _one_config(ytr, str_, yte, ste):
    """Metrics for one config on one seed.

    REPORTED operating point: Youden threshold chosen on the pooled out-of-fold TRAINING
    case scores only, applied unchanged to test (the protocol stated in the paper).
    The test-chosen threshold is kept ONLY as `*_testthr` fields, to document the
    correction against the submitted version; it must never be reported as a result.
    """
    auc = roc_auc_score(yte, ste)
    thr = config.DECISION_THRESHOLD           # fixed 0.5, untuned (revision)
    m_yd = _metrics(yte, ste, _best_threshold(ytr, str_))  # v2: train-OOF Youden, for the record
    m = _metrics(yte, ste, thr)
    tp, fp, fn = m["tp"], m["fp"], m["fn"]
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    m_or = _metrics(yte, ste, _best_threshold(yte, ste))   # submitted-version (test-picked)
    return dict(auc=auc, acc=m["accuracy"], bal_acc=m["bal_acc"],
                sens=m["sensitivity"], spec=m["specificity"], prec=prec, f1=f1,
                tn=m["tn"], fp=fp, fn=fn, tp=tp, thr=thr,
                acc_testthr=m_or["accuracy"], bal_testthr=m_or["bal_acc"],
                sens_testthr=m_or["sensitivity"], spec_testthr=m_or["specificity"],
                acc_youden=m_yd["accuracy"], bal_youden=m_yd["bal_acc"])


def run_dataset(conn, label, datasets, task, split_fn, ms_map, apply_nc=False):
    raw3 = load_enc(conn, datasets, BASE3)
    raw4 = load_enc(conn, datasets, BASE3 + [MSFEAT])
    if apply_nc:
        ids, X3, y, split, grp, view, ds, patho, abn = M.apply_task("nc", *raw3)
        POS = ["CANCER", "MALIGNANT"]
        keep = np.isin(raw4[7], ["NORMAL"] + POS)   # raw4[7] = patho
        X4 = raw4[1][keep]
        assert np.array_equal(ids, raw4[0][keep]), "BASE3/BASE4 row mismatch (nc)"
    else:
        ids, X3, y, split, grp, view, ds, patho, abn = raw3
        X4 = raw4[1]
        assert np.array_equal(ids, raw4[0]), "BASE3/BASE4 row mismatch"

    per = {c: [] for c in CFG_ORDER}
    per_full = {c: [] for c in FULL_CFGS}    # all test cases incl. single-view (no MS score needed)
    for seed in SEEDS:
        tr, te = split_fn(seed, split, grp, y)
        # BASE3 case scores
        rcf_tr3, rcf_te3 = head_case_scores(conn, ids, X3, y, grp, view, tr, te, M.C_HEAD, seed)
        ytr, R3tr, Mtr, yte, R3te, Mte = align_B(rcf_tr3, rcf_te3, ms_map, ms_map)
        # BASE4 case scores (same cases)
        rcf_tr4, rcf_te4 = head_case_scores(conn, ids, X4, y, grp, view, tr, te, M.C_HEAD, seed)
        ytr4, R4tr, Mtr4, yte4, R4te, Mte4 = align_B(rcf_tr4, rcf_te4, ms_map, ms_map)
        assert np.array_equal(yte, yte4), "case alignment differs between BASE3/BASE4"

        strain, stest = {}, {}
        strain["zero_shot"], stest["zero_shot"] = _scores("zero_shot", R3tr, Mtr, R3te, Mte)
        strain["probe3"], stest["probe3"] = _scores("probe", R3tr, Mtr, R3te, Mte)
        strain["late3"], stest["late3"] = _scores("late", R3tr, Mtr, R3te, Mte)
        strain["base4"], stest["base4"] = _scores("probe", R4tr, Mtr, R4te, Mte)
        for c in CFG_ORDER:
            per[c].append(_one_config(ytr, strain[c], yte, stest[c]))
        # full test population (every case, single-view included): probe configs only
        for c, (cf_tr, cf_te) in (("probe3", (rcf_tr3, rcf_te3)), ("base4", (rcf_tr4, rcf_te4))):
            per_full[c].append(_one_config(np.array(cf_tr["y_true"]), np.array(cf_tr["y_score"]),
                                           np.array(cf_te["y_true"]), np.array(cf_te["y_score"])))

    def _agg(rows):
        keys = ("auc", "acc", "bal_acc", "sens", "spec", "prec", "f1", "thr",
                "acc_testthr", "bal_testthr", "sens_testthr", "spec_testthr",
                "acc_youden", "bal_youden")
        agg = {k: float(np.mean([r[k] for r in rows])) for k in keys}
        agg["std"] = {k: float(np.std([r[k] for r in rows]))
                      for k in ("auc", "acc", "bal_acc", "sens", "spec", "prec", "f1")}
        for cc in ("tn", "fp", "fn", "tp"):
            agg[cc] = int(np.sum([r[cc] for r in rows]))
        agg["per_seed"] = [{k: (float(r[k]) if not isinstance(r[k], (int, np.integer)) else int(r[k]))
                            for k in ("auc", "acc", "bal_acc", "sens", "spec", "tn", "fp", "fn", "tp")}
                           for r in rows]
        agg["n_test_cases"] = int(np.mean([r["tn"] + r["fp"] + r["fn"] + r["tp"] for r in rows]))
        return agg

    out = {"paired": {c: _agg(per[c]) for c in CFG_ORDER},
           "full": {c: _agg(per_full[c]) for c in FULL_CFGS}}

    print(f"\n=== {label} ===  (X3={X3.shape[1]}d  X4={X4.shape[1]}d  {len(SEEDS)} seeds)")
    print("    fixed 0.5 threshold (REPORTED) | [submitted: test-picked threshold]")
    for pop in ("full", "paired"):
        for c, a in out[pop].items():
            print(f"    {pop:6s} {c:10s} n={a['n_test_cases']:4d} AUC {a['auc']:.4f}+-{a['std']['auc']:.4f} "
                  f"acc {a['acc']*100:5.1f} bal {a['bal_acc']*100:5.1f} sens {a['sens']*100:5.1f} "
                  f"spec {a['spec']*100:5.1f} prec {a['prec']*100:5.1f} F1 {a['f1']*100:5.1f} | "
                  f"[acc {a['acc_testthr']*100:5.1f} bal {a['bal_testthr']*100:5.1f} "
                  f"sens {a['sens_testthr']*100:5.1f} spec {a['spec_testthr']*100:5.1f}]")
    return out


def main():
    conn = db.connect()
    result = {}

    result["cbis_bm"] = run_dataset(
        conn, "CBIS-DDSM benign/malignant (official split)", ["cbis_full"], "bm",
        lambda seed, split, grp, y: (split == "train", split == "test"),
        cbis_ms(), apply_nc=False)

    result["miniddsm_nc"] = run_dataset(
        conn, "Mini-DDSM normal/cancer (20% patient-disjoint)",
        ["miniddsm", "miniddsm_normal"], "nc",
        lambda seed, split, grp, y: (~M.patient_test_mask(grp, y, 0.2, seed),
                                     M.patient_test_mask(grp, y, 0.2, seed)),
        M.load_ms(["miniddsm", "miniddsm_normal"]), apply_nc=True)

    # v1 (cache/base4_numbers.json) is kept: it holds the submitted test-threshold numbers
    path = os.path.join(config.PROJECT_DIR, "cache", "base4_numbers_v3.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
