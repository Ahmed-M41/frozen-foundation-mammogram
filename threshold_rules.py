"""Honest ways to set the operating-point cutoff (no test labels used).

Every rule below picks the cutoff from TRAINING out-of-fold case scores only. The rule
itself is chosen by an inner check on training data (pick the cutoff on 4/5 of the
training OOF cases, measure accuracy on the other 1/5, repeated), never by test results.

Also tests a 4-seed ensemble for CBIS-DDSM: the 4 seeds only change the cross-validation
fold assignment, so averaging their scores is one model (an ensemble of 20 fold models).

Run: python threshold_rules.py   ->  cache/threshold_rules.json
"""
import json
import os

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import config
import db
import miniddsm_B as M
from data_scaling_extend import head_case_scores, align_B
from exp_msfeat_fusion import load_enc, BASE3, MSFEAT, cbis_ms
from mil import _metrics

SEEDS = [0, 1, 2, 3]


# ---------------------------------------------------------------- cutoff rules (train only)
def _sweep(y, s):
    """All cutoffs at once: candidate t = each unique score (predict 1 if s >= t).
    Returns (cutoffs, sensitivity, specificity, accuracy) arrays, ascending cutoff."""
    o = np.argsort(-s, kind="mergesort")
    s_d, y_d = s[o], y[o]
    tp = np.cumsum(y_d)
    fp = np.cumsum(1 - y_d)
    last = np.r_[s_d[1:] != s_d[:-1], True]          # last index of each tied block
    t, tp, fp = s_d[last], tp[last], fp[last]
    P, N = y.sum(), len(y) - y.sum()
    sens, spec = tp / P, 1 - fp / N
    acc = (tp + (N - fp)) / len(y)
    return t, sens, spec, acc                       # descending cutoff order


def r_balacc(y, s):                      # current rule: maximise balanced accuracy
    t, se, sp, _ = _sweep(y, s)
    return float(t[np.argmax(se + sp)])


def r_acc(y, s):                         # maximise plain accuracy
    t, _, _, a = _sweep(y, s)
    return float(t[np.argmax(a)])


def r_half(y, s):                        # fixed 0.5 (balanced class weights -> natural midpoint)
    return 0.5


def r_prev(y, s):                        # predicted-positive rate = training prevalence
    return float(np.quantile(s, 1 - y.mean()))


def r_boot(y, s, B=200, seed=0):         # bagged balanced-accuracy cutoff (more stable)
    rng = np.random.default_rng(seed)
    ts = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        ts.append(r_balacc(y[i], s[i]))
    return float(np.median(ts))


def r_boot_acc(y, s, B=200, seed=0):     # bagged accuracy cutoff
    rng = np.random.default_rng(seed)
    ts = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        ts.append(r_acc(y[i], s[i]))
    return float(np.median(ts))


RULES = {"bal_acc (current)": r_balacc, "max_acc": r_acc, "fixed_0.5": r_half,
         "prevalence": r_prev, "bagged_bal_acc": r_boot, "bagged_acc": r_boot_acc}


def inner_check(y, s, rule, reps=5):
    """Accuracy of a rule measured INSIDE training data only (selection criterion)."""
    accs, bals = [], []
    for r in range(reps):
        skf = StratifiedKFold(5, shuffle=True, random_state=r)
        for a, b in skf.split(s, y):
            t = rule(y[a], s[a])
            p = (s[b] >= t).astype(int)
            accs.append(np.mean(p == y[b]))
            bals.append(balanced_accuracy_score(y[b], p))
    return float(np.mean(accs)), float(np.mean(bals))


def evaluate_rules(ytr, str_, yte, ste):
    out = {}
    for name, rule in RULES.items():
        t = rule(ytr, str_)
        m = _metrics(yte, ste, t)
        ia, ib = inner_check(ytr, str_, rule)
        out[name] = dict(thr=float(t), auc=float(roc_auc_score(yte, ste)),
                         acc=m["accuracy"], bal=m["bal_acc"],
                         sens=m["sensitivity"], spec=m["specificity"],
                         inner_acc=ia, inner_bal=ib)
    return out


# ---------------------------------------------------------------- case scores per seed
def case_scores(conn, datasets, split_fn, ms_map, apply_nc):
    raw4 = load_enc(conn, datasets, BASE3 + [MSFEAT])
    if apply_nc:
        raw3 = load_enc(conn, datasets, BASE3)
        ids, _, y, split, grp, view, *_ = M.apply_task("nc", *raw3)
        keep = np.isin(raw4[7], ["NORMAL", "CANCER", "MALIGNANT"])
        X4 = raw4[1][keep]
        assert np.array_equal(ids, raw4[0][keep])
    else:
        ids, X4, y, split, grp, view = raw4[:6]
    per_seed = []
    for seed in SEEDS:
        tr, te = split_fn(seed, split, grp, y)
        cf_tr, cf_te = head_case_scores(conn, ids, X4, y, grp, view, tr, te, M.C_HEAD, seed)
        ytr_p, Rtr_p, _, yte_p, Rte_p, _ = align_B(cf_tr, cf_te, ms_map, ms_map)
        per_seed.append(dict(
            full=(np.array(cf_tr["y_true"]), np.array(cf_tr["y_score"]),
                  np.array(cf_te["y_true"]), np.array(cf_te["y_score"]),
                  [str(c) for c in cf_tr["cids"]], [str(c) for c in cf_te["cids"]]),
            paired=(ytr_p, Rtr_p, yte_p, Rte_p)))
    return per_seed


def _avg(rows):
    keys = rows[0][next(iter(rows[0]))].keys()
    return {r: {k: float(np.mean([row[r][k] for row in rows])) for k in keys} for r in rows[0]}


def _print(title, res):
    print(f"\n{title}")
    print(f"  {'rule':20s} {'cutoff':>6s} | inner(train) acc  bal | TEST AUC   acc   bal  sens  spec")
    for r, a in res.items():
        print(f"  {r:20s} {a['thr']:6.3f} | {a['inner_acc']*100:12.1f} {a['inner_bal']*100:5.1f} | "
              f"{a['auc']:8.3f} "
              f"{a['acc']*100:8.1f} {a['bal']*100:5.1f} {a['sens']*100:5.1f} {a['spec']*100:5.1f}")


def main():
    import pickle
    conn = db.connect()
    result = {}

    pkl = os.path.join(config.PROJECT_DIR, "cache", "threshold_rules_scores.pkl")
    if os.path.exists(pkl):
        with open(pkl, "rb") as f:
            cbis, mini = pickle.load(f)
    else:
        cbis = case_scores(conn, ["cbis_full"],
                           lambda s, sp, g, y: (sp == "train", sp == "test"),
                           cbis_ms(), apply_nc=False)
        print("cbis scores done", flush=True)
        mini = case_scores(conn, ["miniddsm", "miniddsm_normal"],
                           lambda s, sp, g, y: (~M.patient_test_mask(g, y, 0.2, s),
                                                M.patient_test_mask(g, y, 0.2, s)),
                           M.load_ms(["miniddsm", "miniddsm_normal"]), apply_nc=True)
        with open(pkl, "wb") as f:
            pickle.dump((cbis, mini), f)
        print("mini scores done (cached)", flush=True)

    for name, per_seed in (("cbis", cbis), ("mini", mini)):
        for pop in ("full", "paired"):
            rows = [evaluate_rules(*d[pop][:4]) for d in per_seed]
            result[f"{name}_{pop}_mean4seeds"] = _avg(rows)
            _print(f"{name} {pop}: mean of 4 seeds (single models)", result[f"{name}_{pop}_mean4seeds"])

    # CBIS: same official test set for every seed -> average the 4 seeds' scores (ensemble)
    ytr0, _, yte0, _, cid_tr, cid_te = cbis[0]["full"]
    for d in cbis[1:]:
        assert d["full"][4] == cid_tr and d["full"][5] == cid_te, "case order differs"
    s_tr = np.mean([d["full"][1] for d in cbis], axis=0)
    s_te = np.mean([d["full"][3] for d in cbis], axis=0)
    ens = evaluate_rules(ytr0, s_tr, yte0, s_te)
    print(f"\n  ensemble AUC full: test {roc_auc_score(yte0, s_te):.4f}  "
          f"(train OOF {roc_auc_score(ytr0, s_tr):.4f})")
    result["cbis_full_ensemble"] = ens
    _print("cbis full: 4-seed ensemble", ens)

    ytr_p, _, yte_p, _ = cbis[0]["paired"]
    s_trp = np.mean([d["paired"][1] for d in cbis], axis=0)
    s_tep = np.mean([d["paired"][3] for d in cbis], axis=0)
    ens_p = evaluate_rules(ytr_p, s_trp, yte_p, s_tep)
    print(f"\n  ensemble AUC paired: test {roc_auc_score(yte_p, s_tep):.4f}")
    result["cbis_paired_ensemble"] = ens_p
    _print("cbis paired: 4-seed ensemble", ens_p)

    path = os.path.join(config.PROJECT_DIR, "cache", "threshold_rules.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
