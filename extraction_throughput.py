"""End-to-end batch-32 extraction throughput (read + decode + preprocess + forward) per encoder,
on CBIS-DDSM (native resolution) and Mini-DDSM (512 px), 96 images each after warm-up."""
import time, json, torch, statistics as st
from PIL import Image
import ablation_e1 as A, db, embedders
conn = db.connect()
P = {ds: [r[0] for r in conn.execute(f"SELECT file_path FROM samples WHERE dataset IN ({q}) ORDER BY RANDOM() LIMIT 128")]
     for ds, q in (("cbis", "'cbis_full'"), ("mini", "'miniddsm','miniddsm_normal'"))}
N = {ds: conn.execute(f"SELECT COUNT(*) FROM samples WHERE dataset IN ({q})").fetchone()[0]
     for ds, q in (("cbis", "'cbis_full'"), ("mini", "'miniddsm','miniddsm_normal'"))}
res = {"n_images": N, "img_per_s": {}}
for name in A.ALL4:
    e = embedders.load_embedder(name); res["img_per_s"][name] = {}
    for ds, ps in P.items():
        e.embed_batch([Image.open(p) for p in ps[:32]])
        ts = []
        for k in range(1, 4):
            b = ps[32 * k:32 * (k + 1)]
            torch.cuda.synchronize(); t = time.perf_counter()
            e.embed_batch([Image.open(p) for p in b]); torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
        res["img_per_s"][name][ds] = 32 / st.median(ts)
    print(name, {k: round(v, 1) for k, v in res["img_per_s"][name].items()}, flush=True)
    del e; torch.cuda.empty_cache()
for ds in P:
    s = sum(N[ds] / res["img_per_s"][n][ds] for n in A.ALL4)
    res[f"total_s_{ds}"] = s
    print(ds, N[ds], "images: all four encoders", round(s), "s =", round(s / 60, 1), "min")
json.dump(res, open("cache/extraction_throughput.json", "w"), indent=2)
