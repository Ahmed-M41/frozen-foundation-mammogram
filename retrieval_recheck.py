"""Re-measure the kNN-retrieval negative-result baseline at the CURRENT probe resolution.

The paper reports per-encoder retrieval AUC (Mammo-CLIP 0.71, RAD-DINO 0.69, DINOv2 0.64),
but the DINOv2 number was measured at 224 while the probe now uses DINOv2 at 518. This
re-runs the same retrieval machinery (evaluate.run_headline: CC/MLO branches, case-fused,
paired subset) on cbis_full for the exact three encoders the probe uses, sweeping the
fusion head {vote, weighted, tip} and neighbourhood k, and prints the BEST paired-case AUC
per encoder. No DB writes (persist=False).
"""
import numpy as np
import db
import evaluate as E

ENCODERS = ["mammo_clip_b5", "dinov2_vitl14_r518_noclahe", "rad_dino_noclahe"]
KS = [25, 75, 150]
HEADS = ["vote", "weighted", "tip"]
DIST = "cosine"


def best_auc(conn, emb):
    best = (-1.0, None)
    for k in KS:
        for head in HEADS:
            res = E.run_headline(conn, emb, k, DIST, head, dataset="cbis_full",
                                 four_way=True, persist=False, verbose=False,
                                 save_predictions=False)
            if res is None:
                continue
            cf = res["case"]
            pm = cf.get("paired_metrics") or cf["metrics"]
            auc = pm["auc"]
            if auc > best[0]:
                best = (auc, f"k={k} {head}")
    return best


def main():
    conn = db.connect()
    print(f"Retrieval negative-result re-check on cbis_full (best over k{KS} x {HEADS}):")
    for emb in ENCODERS:
        n = conn.execute("SELECT COUNT(*) c FROM embeddings WHERE embedder_name=?",
                         (emb,)).fetchone()["c"]
        if n == 0:
            print(f"  {emb:32s} NO EMBEDDINGS")
            continue
        auc, cfg = best_auc(conn, emb)
        print(f"  {emb:32s} best paired-case AUC = {auc:.4f}  ({cfg})", flush=True)


if __name__ == "__main__":
    main()
