"""INbreast: MammoScreen AS RELEASED vs MammoScreen + our classifier vs MammoFF, on exactly the
same test breasts (regime B test patients, seeds 0-3 pooled), with paired bootstrap CIs.
MammoScreen as released = its own cancer score (CC+MLO averaged by the model itself), from
cache/mammoscreen_inbreast_zeroshot.npz. Run: python inbreast_released_compare.py
-> cache/inbreast_released_compare.json"""
import json
import os

import numpy as np

import ablation_e1 as A
import bootstrap_ci as BC
import config
import db
import inbreast_eval as IE

MCLIP, DINO, RAD, MS = A.ALL4


def main():
    conn = db.connect()
    z = np.load(os.path.join(config.PROJECT_DIR, "cache", "mammoscreen_inbreast_zeroshot.npz"), allow_pickle=True)
    released = {str(c): float(s) for c, s in zip(z["cids"], z["s"])}
    rows = {"MammoFF": A.ALL4, "MammoScreen + our classifier": [MS]}
    te_all = {k: {} for k in rows}
    for name, encs in rows.items():
        ids, X, y, split, grp, view, *_ = A.load(conn, ["inbreast_full"], encs, "bm", True)
        for seed in IE.SEEDS:
            tr, te = IE.masks("B_within", seed, split, grp, y)
            H = A.heads(ids, X, y, grp, view, tr, te, seed)
            ted, _ = A.fuse(conn, H, ids, view, te, "default", "te")
            te_all[name].update({f"{seed}:{c}": v for c, v in ted.items()})
    ref = te_all["MammoFF"]
    te_all["MammoScreen as released"] = {k: (v[0], released[k.split(":", 1)[1]]) for k, v in ref.items()
                                         if k.split(":", 1)[1] in released}
    shared = sorted(set.intersection(*(set(d) for d in te_all.values())))
    out = {"n_pooled_breasts": len(shared)}
    for name, d in te_all.items():
        y = np.array([d[c][0] for c in shared]); s = np.array([d[c][1] for c in shared])
        ci, _ = BC.ci(y, s, [c.split(":", 1)[1] for c in shared])   # case-level: a breast in several partitions is one unit
        dl, dci = (0.0, [0.0, 0.0]) if name == "MammoFF" else A.delta_ci(ref, d, shared)
        out[name] = dict(auc=ci["auc"], delta_vs_mammoff=dl, delta_ci=dci)
        print(f"{name:30s} AUC {ci['auc']['est']:.3f} [{ci['auc']['lo']:.3f}, {ci['auc']['hi']:.3f}]  "
              f"vs MammoFF {dl:+.3f} [{dci[0]:+.3f}, {dci[1]:+.3f}]  (n={len(shared)})")
    a = te_all["MammoScreen + our classifier"]; b = te_all["MammoScreen as released"]
    dl, dci = A.delta_ci(b, a, shared)
    out["ours_minus_released"] = dict(delta=dl, delta_ci=dci)
    print(f"MammoScreen + our classifier minus as released: {dl:+.3f} [{dci[0]:+.3f}, {dci[1]:+.3f}]")
    json.dump(out, open(os.path.join(config.PROJECT_DIR, "cache", "inbreast_released_compare.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
