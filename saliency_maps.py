"""Occlusion-sensitivity saliency for the frozen-encoder + linear-probe pipeline.

"Where does the model base its judgment?" Our decision is a linear probe on the pooled
features of four frozen backbones, so there is no single conv layer to Grad-CAM. Occlusion
sensitivity is the faithful, model-agnostic answer: slide a neutral patch over the image,
re-encode the occluded image through all four frozen backbones, run the probe, and record
how far the predicted probability of the true class falls. A large fall marks a region the
decision relies on. No gradients, no backbone training; it attributes the whole pipeline.

One confident, correctly-classified example is shown per class:
  CBIS-DDSM: benign + malignant (mass);  Mini-DDSM: normal + cancer.
Outputs: paper_ieee_access/figures/fig_saliency_cbis.png and fig_saliency_mini.png
(occlusion maps are cached to cache/saliency_panels.pkl; `--render` re-plots without re-encoding).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import zoom, gaussian_filter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import config, db
import miniddsm_B as M
import embedders as EMB
from exp_msfeat_fusion import load_enc, BASE3, MSFEAT
from data_scaling_extend import _l2

ENC_NAMES = BASE3 + [MSFEAT]
BATCH = {"mammo_clip_b5": 8, "dinov2_vitl14_r518_noclahe": 16,
         "rad_dino_noclahe": 16, "mammoscreen_feat": 16}
GRID = 12            # occluder grid (GRID x GRID positions)
OCC_FRAC = 0.18      # occluder side as a fraction of the shorter image side
FIGDIR = os.path.join(config.PROJECT_DIR, "paper_ieee_access", "figures")

_ENCODERS = None
def encoders():
    global _ENCODERS
    if _ENCODERS is None:
        _ENCODERS = [(n, EMB.load_embedder(n)) for n in ENC_NAMES]
    return _ENCODERS


def _chunked(enc, pils, bs):
    out = []
    for i in range(0, len(pils), bs):
        out.append(enc.embed_batch(pils[i:i + bs]))
    return np.concatenate(out, 0)


def features(pils):
    """List[PIL] -> [N, 7680] features, L2-normed per encoder then concatenated (train order)."""
    mats = []
    for name, enc in encoders():
        mats.append(_l2(_chunked(enc, pils, BATCH[name])))
    return np.concatenate(mats, 1).astype(np.float32)


def fit_scorer(Xtr, ytr):
    """View-agnostic head: standardise + L2-regularised logistic regression (C=0.5, balanced)."""
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=4000, C=M.C_HEAD, class_weight="balanced")
    clf.fit(sc.transform(Xtr), ytr)
    return sc, clf


def prob(sc, clf, feats):
    return clf.predict_proba(sc.transform(feats))[:, 1]


MAXSIDE = 1536      # every encoder resizes to <=1520 anyway; this is faithful and far cheaper


def _downscale(gray):
    H, W = gray.shape
    s = MAXSIDE / max(H, W)
    if s >= 1.0:
        return gray
    return np.array(Image.fromarray(gray).resize((int(W * s), int(H * s)), Image.LANCZOS))


def occlusion_map(gray, sc, clf):
    """gray: HxW uint8. Returns (P0, gray_used, saliency in [0,1]) via occlusion sensitivity.
    Saliency = |change in P(positive)| when a neutral patch hides each region: the regions
    the decision is most sensitive to, whichever way the case is called."""
    gray = _downscale(gray)
    H, W = gray.shape
    side = max(8, int(OCC_FRAC * min(H, W)))
    ys = np.linspace(0, H - side, GRID).astype(int)
    xs = np.linspace(0, W - side, GRID).astype(int)
    fill = int(gray.mean())

    variants = [Image.fromarray(gray).convert("RGB")]
    for y in ys:
        for x in xs:
            g = gray.copy()
            g[y:y + side, x:x + side] = fill
            variants.append(Image.fromarray(g).convert("RGB"))
    Fs = sc.transform(features(variants))
    score = clf.decision_function(Fs)          # LOGIT: unsaturated, this is what the probe computes
    p0 = float(clf.predict_proba(Fs[:1])[0, 1])  # probability, for the human-readable title
    s0, sg = score[0], score[1:].reshape(GRID, GRID)
    change = np.abs(s0 - sg)                    # |change in decision score| when a region is hidden
    sal = zoom(change, (H / GRID, W / GRID), order=1)
    sal = gaussian_filter(sal, sigma=min(H, W) * 0.012)
    return p0, gray, sal                        # sal in logit units, NOT per-image normalised


def pick_examples(ids, X, y, sc, clf, split_mask):
    """Rank test cases: most confident true positives, most confident true negatives."""
    p = prob(sc, clf, X[split_mask])
    ids_m, y_m = ids[split_mask], y[split_mask]
    pos_idx = np.where(y_m == 1)[0]
    neg_idx = np.where(y_m == 0)[0]
    pos_sorted = pos_idx[np.argsort(-p[pos_idx])]   # highest prob first
    neg_sorted = neg_idx[np.argsort(p[neg_idx])]    # lowest prob first
    return ids_m, p, pos_sorted, neg_sorted


def load_gray(path):
    return np.array(Image.open(path).convert("L"))


def path_view_patho(conn, sid):
    r = conn.execute("SELECT file_path,view,pathology FROM samples WHERE id=?", (int(sid),)).fetchone()
    return r["file_path"], r["view"], str(r["pathology"])


def choose_clean(conn, ids_m, ranked, k=12):
    """Among the top-k most confident correct candidates, prefer the image with the fewest
    burned-in near-white pixels (labels/markers), so the example is not dominated by an
    artefact. Returns the most confident essentially-clean id, else the cleanest of the k."""
    best, best_frac = ids_m[ranked[0]], 1.0
    for idx in ranked[:k]:
        sid = ids_m[idx]
        fp, _, _ = path_view_patho(conn, sid)
        frac = float((load_gray(fp) > 245).mean())
        if frac < best_frac:
            best, best_frac = sid, frac
        if frac < 0.002:          # <0.2% white -> clean; keep the most confident such one
            return sid
    return best


def compute_panel(conn, sid, sc, clf, title):
    fp, _, _ = path_view_patho(conn, sid)
    p0, gray, sal = occlusion_map(load_gray(fp), sc, clf)
    return dict(gray=gray, sal=sal, p0=p0, title=title)


def render_pair(panels, suptitle, outname, figw, figh):
    """Two panels on a shared absolute colour scale (per-figure vmax, in decision-score
    units). Alpha is proportional to the signal, so a negative case stays faint and only
    true evidence lights up, rather than a per-image-normalised map exaggerating noise.
    Manual layout with a reserved top margin so the suptitle never overlaps panel titles."""
    vmax = max(p["sal"].max() for p in panels)
    fig, axes = plt.subplots(1, 2, figsize=(figw, figh))
    fig.subplots_adjust(top=0.80, bottom=0.17, left=0.03, right=0.97, wspace=0.05)
    im = None
    for ax, pan in zip(axes, panels):
        ax.imshow(pan["gray"], cmap="gray", aspect="equal")
        a = np.clip(pan["sal"] / (vmax + 1e-9), 0, 1) * 0.7      # signal-proportional opacity
        im = ax.imshow(pan["sal"], cmap="jet", vmin=0, vmax=vmax, alpha=a, aspect="equal")
        ax.set_title(f"{pan['title']}\n$p$(positive) = {pan['p0']:.2f}", fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])
    cax = fig.add_axes([0.25, 0.08, 0.5, 0.028])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("change in the probe's decision score when the region is hidden", fontsize=8)
    cb.set_ticks([0, vmax]); cb.set_ticklabels(["none", "large"])
    fig.suptitle(suptitle, fontsize=12, y=0.965)
    out = os.path.join(FIGDIR, outname)
    fig.savefig(out, dpi=200); plt.close(fig)
    print("saved", out)


def build():
    conn = db.connect()
    TITLE = "Occlusion sensitivity of the frozen-encoder probe"

    # ---------- CBIS-DDSM benign/malignant ----------
    ids_c, Xc, yc0, split_c, grp_c, view_c, ds_c, patho_c, abn_c = load_enc(conn, ["cbis_full"], ENC_NAMES)
    yc = np.isin(patho_c, ["MALIGNANT", "CANCER"]).astype(int)
    sc_c, clf_c = fit_scorer(Xc[split_c == "train"], yc[split_c == "train"])
    idsm, _, pos_c, neg_c = pick_examples(ids_c, Xc, yc, sc_c, clf_c, split_c == "test")
    cbis_mal, cbis_ben = choose_clean(conn, idsm, pos_c), choose_clean(conn, idsm, neg_c)
    cbis = [compute_panel(conn, cbis_ben, sc_c, clf_c, "CBIS-DDSM: benign (mass)"),
            compute_panel(conn, cbis_mal, sc_c, clf_c, "CBIS-DDSM: malignant (mass)")]

    # ---------- Mini-DDSM normal/cancer ----------
    raw = load_enc(conn, ["miniddsm", "miniddsm_normal"], ENC_NAMES)
    ids_n, Xn, yn, split_n, grp_n, view_n, ds_n, patho_n, abn_n = M.apply_task("nc", *raw)
    te_n = M.patient_test_mask(grp_n, yn, 0.2, 0)
    sc_n, clf_n = fit_scorer(Xn[~te_n], yn[~te_n])
    idsn, _, pos_n, neg_n = pick_examples(ids_n, Xn, yn, sc_n, clf_n, te_n)
    mini_can, mini_nor = choose_clean(conn, idsn, pos_n), choose_clean(conn, idsn, neg_n)
    mini = [compute_panel(conn, mini_nor, sc_n, clf_n, "Mini-DDSM: normal"),
            compute_panel(conn, mini_can, sc_n, clf_n, "Mini-DDSM: cancer")]

    import pickle
    pickle.dump({"cbis": cbis, "mini": mini},
                open(os.path.join(config.PROJECT_DIR, "cache", "saliency_panels.pkl"), "wb"))
    render_all()
    print(f"  CBIS  benign id={cbis_ben}  malignant id={cbis_mal}")
    print(f"  Mini  normal id={mini_nor}  cancer id={mini_can}")


def render_all():
    """Render both figures from cached occlusion maps (fast; no re-encoding)."""
    import pickle
    TITLE = "Occlusion sensitivity of the frozen-encoder probe"
    d = pickle.load(open(os.path.join(config.PROJECT_DIR, "cache", "saliency_panels.pkl"), "rb"))
    render_pair(d["cbis"], TITLE, "fig_saliency_cbis.png", 7.4, 6.8)
    render_pair(d["mini"], TITLE, "fig_saliency_mini.png", 8.6, 5.8)


if __name__ == "__main__":
    import sys
    render_all() if "--render" in sys.argv else build()
