"""Inference cost per image and per case (R1.5-c/d/e): latency and peak GPU memory.
Timings INCLUDE reading the image from disk, decoding, each encoder's own preprocessing
(resize, normalisation) and the forward pass; GPU work is synchronised before each clock read.
Images: CBIS-DDSM official test set at native resolution (as in the paper).
  Part 1, per encoder: model load time; weights memory; batch-1 latency per image (median and
          IQR over N_IMG images after warm-up); throughput at batch 32; peak GPU memory.
  Part 2, deployment: all four encoders resident at once; one CASE = CC + MLO of one breast:
          read + preprocess + 4 encoders x 2 images + L2/concat + the classifier (3 heads x 5
          folds, predicted on CPU); median latency per case and peak GPU memory.
Run: python inference_timing.py -> cache/inference_timing.json"""
import json
import os
import statistics as st
import time

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import ablation_e1 as A
import config
import db
import embedders

N_IMG, WARM, N_CASES = 60, 5, 40
OUT = os.path.join(config.PROJECT_DIR, "cache", "inference_timing.json")
MB = 1024 ** 2


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def clock(fn):
    sync(); t = time.perf_counter(); r = fn(); sync()
    return time.perf_counter() - t, r


def q(xs):
    xs = sorted(xs)
    return dict(median_ms=1000 * st.median(xs), p25_ms=1000 * xs[len(xs) // 4],
                p75_ms=1000 * xs[(3 * len(xs)) // 4])


def main():
    conn = db.connect()
    rows = conn.execute("SELECT id,file_path,case_id,view FROM samples WHERE dataset='cbis_full' "
                        "AND split='test' ORDER BY id").fetchall()
    paths = [r["file_path"] for r in rows]
    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "encoders": {}}
    t_dec = [clock(lambda p=p: Image.open(p).convert("RGB"))[0] for p in paths[:N_IMG]]
    out["decode_only"] = q(t_dec)
    print(f"read+decode only: {out['decode_only']['median_ms']:.1f} ms/image", flush=True)

    # ---------------- Part 1: each encoder alone ----------------
    for name in A.ALL4:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        t_load, emb = clock(lambda: embedders.load_embedder(name))
        w_mem = (torch.cuda.memory_allocated() - base) / MB
        for p in paths[:WARM]:
            emb.embed_batch([Image.open(p)])
        torch.cuda.reset_peak_memory_stats()
        lat = [clock(lambda p=p: emb.embed_batch([Image.open(p)]))[0] for p in paths[WARM:WARM + N_IMG]]
        peak1 = torch.cuda.max_memory_allocated() / MB
        torch.cuda.reset_peak_memory_stats()
        b = paths[:32]
        emb.embed_batch([Image.open(p) for p in b])                     # warm-up at batch 32
        tb = [clock(lambda: emb.embed_batch([Image.open(p) for p in b]))[0] for _ in range(3)]
        peak32 = torch.cuda.max_memory_allocated() / MB
        nparam = sum(p.numel() for p in emb.model.parameters()) if hasattr(emb, "model") else None
        out["encoders"][name] = dict(load_s=t_load, weights_mb=w_mem, params=nparam,
                                     batch1=q(lat), batch32_img_per_s=32 / st.median(tb),
                                     peak_mb_batch1=peak1, peak_mb_batch32=peak32)
        e = out["encoders"][name]
        print(f"{name:28s} load {t_load:5.1f}s  weights {w_mem:6.0f} MB  batch-1 {e['batch1']['median_ms']:6.1f} ms/img  "
              f"batch-32 {e['batch32_img_per_s']:5.1f} img/s  peak {peak1:6.0f} / {peak32:6.0f} MB", flush=True)
        del emb; torch.cuda.empty_cache()

    # ---------------- Part 2: deployment, all four resident, one case = CC + MLO ----------------
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    embs = [embedders.load_embedder(n) for n in A.ALL4]
    resident = torch.cuda.memory_allocated() / MB
    ids, X, y, split, grp, view, *_ = A.load(conn, ["cbis_full"], A.ALL4, "bm", True)
    tr = split == "train"
    heads = {}
    for hname, m in (("all", np.ones(len(view), bool)), ("cc", view == "CC"), ("mlo", view == "MLO")):
        fits = []
        for k in range(5):                                    # 5 fold models per head, as in the paper
            sel = np.where(tr & m)[0][k::5]
            keep = np.setdiff1d(np.where(tr & m)[0], sel)
            s = StandardScaler().fit(X[keep])
            fits.append((s, LogisticRegression(max_iter=4000, C=0.5, class_weight="balanced")
                         .fit(s.transform(X[keep]), y[keep])))
        heads[hname] = fits
    by_case = {}
    for r in rows:
        by_case.setdefault(r["case_id"], {})[r["view"]] = r["file_path"]
    cases = [v for v in by_case.values() if "CC" in v and "MLO" in v][:N_CASES + WARM]

    def l2(v):
        return v / max(np.linalg.norm(v), 1e-8)

    def run_case(c):
        feats = {}
        for vw in ("CC", "MLO"):
            im = Image.open(c[vw])
            feats[vw] = np.concatenate([l2(e.embed_batch([im])[0]) for e in embs])[None]
        sc = []
        for hname, vw in (("all", "CC"), ("all", "MLO"), ("cc", "CC"), ("mlo", "MLO")):
            sc.append(np.mean([lr.predict_proba(s.transform(feats[vw]))[0, 1] for s, lr in heads[hname]]))
        return float(np.mean(sc))

    for c in cases[:WARM]:
        run_case(c)
    torch.cuda.reset_peak_memory_stats()
    lat = [clock(lambda c=c: run_case(c))[0] for c in cases[WARM:]]
    t_cls = [clock(lambda: [lr.predict_proba(s.transform(X[:1])) for h in heads.values() for s, lr in h])[0]
             for _ in range(50)]
    out["deployment"] = dict(resident_weights_mb=resident, per_case=q(lat),
                             classifier_only=q(t_cls), peak_mb=torch.cuda.max_memory_allocated() / MB,
                             n_cases=len(lat))
    d = out["deployment"]
    print(f"ALL FOUR resident {resident:.0f} MB; per case (CC+MLO, read->score) {d['per_case']['median_ms']:.0f} ms "
          f"[IQR {d['per_case']['p25_ms']:.0f}-{d['per_case']['p75_ms']:.0f}]; classifier alone "
          f"{d['classifier_only']['median_ms']:.2f} ms; peak GPU {d['peak_mb']:.0f} MB", flush=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print("saved", OUT)


if __name__ == "__main__":
    main()
