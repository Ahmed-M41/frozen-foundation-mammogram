"""R1.7: full operating-point metrics and confusion counts for every model in the paper's
comparison tables, at the fixed 0.5 threshold (Supplementary Table S8).

Part A (Table finetune models, CBIS-DDSM full official test set, n=210, single fixed fold
assignment as in that table): Frozen, fine-tuned 1 of 4, fine-tuned 3 of 4, LoRA, and the
end-to-end ResNet-50 (from cache/resnet50_e2e_eval.json, scored on its own).
Part B (Table components models): MammoScreen score as released, 3-encoder classifier, late
fusion, early fusion (MammoFF) on CBIS-DDSM paired cases and Mini-DDSM, four-seed means,
confusion counts from seed 0 (cache/base4_numbers_v3.json, shared-film patients excluded).

Run: python full_metrics_s8.py  ->  cache/full_metrics_s8.json
"""
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

import config
import finetune_eval_full4 as F4
import finetune_eval_phase2 as P2
from finetune_paired_bootstrap import _cf

CACHE = os.path.join(config.PROJECT_DIR, "cache")


def metrics(cf):
    y = np.array([v[0] for v in cf.values()])
    s = np.array([v[1] for v in cf.values()])
    p = s >= 0.5
    tp, fn = int((p & (y == 1)).sum()), int((~p & (y == 1)).sum())
    tn, fp = int((~p & (y == 0)).sum()), int((p & (y == 0)).sum())
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    return dict(n=len(y), auc=float(roc_auc_score(y, s)), acc=(tp + tn) / len(y),
                bal=(sens + spec) / 2, sens=sens, spec=spec, prec=prec,
                f1=2 * prec * sens / (prec + sens), tp=tp, fp=fp, fn=fn, tn=tn)


def main():
    out = {"A": {}, "B": {}}
    frozen, _ = _cf(F4.run, C=0.5, mc_fmt="mammo_clip_b5", label="Frozen 4")
    ft1, _ = _cf(F4.run, C=0.5)
    ft3, _ = _cf(P2.run, C=0.5)
    P2.FT_PREFIXES = ["dinov2_vitl14_r518_lora", "rad_dino_lora"]
    P2.FROZEN_ENCODERS = ["mammo_clip_b5", "mammoscreen_feat"]
    lora, _ = _cf(P2.run, C=0.5)
    for name, cf in (("Frozen", frozen), ("Fine-tuned (1 of 4)", ft1),
                     ("Fine-tuned (3 of 4)", ft3), ("LoRA (DINOv2, RAD-DINO)", lora)):
        out["A"][name] = metrics(cf)
    r = json.load(open(os.path.join(CACHE, "resnet50_e2e_eval.json")))["full"]["resnet50"]
    n, npos = 210, out["A"]["Frozen"]["tp"] + out["A"]["Frozen"]["fn"]
    tp = round(r["sens"]["est"] * npos); tn = round(r["spec"]["est"] * (n - npos))
    out["A"]["ResNet-50, end to end"] = dict(
        n=n, **{k: r[k]["est"] for k in ("auc", "acc", "bal", "sens", "spec", "prec", "f1")},
        tp=tp, fn=npos - tp, tn=tn, fp=n - npos - tn)
    b4 = json.load(open(os.path.join(CACHE, "base4_numbers_v3.json")))
    names = {"zero_shot": "MammoScreen score", "probe3": "3-encoder classifier",
             "late3": "Late fusion", "base4": "Early fusion (MammoFF)"}
    for ds, key in (("CBIS-DDSM (B/M), paired", "cbis_bm"), ("Mini-DDSM (N/C)", "miniddsm_nc")):
        out["B"][ds] = {}
        for k, name in names.items():
            m = b4[key]["paired"][k]
            s0 = m["per_seed"][0]
            out["B"][ds][name] = dict(n=s0["tn"] + s0["fp"] + s0["fn"] + s0["tp"], auc=m["auc"],
                                      acc=m["acc"], bal=m["bal_acc"], sens=m["sens"],
                                      spec=m["spec"], prec=m["prec"], f1=m["f1"],
                                      **{c: s0[c] for c in ("tp", "fp", "fn", "tn")})
    for part in out.values():
        for k, v in part.items():
            for name, m in (v.items() if "auc" not in v else [(k, v)]):
                print(f"{name:28s} AUC {m['auc']:.3f} acc {m['acc']*100:.1f} sens {m['sens']*100:.1f} "
                      f"spec {m['spec']*100:.1f} prec {m['prec']*100:.1f} F1 {m['f1']*100:.1f} "
                      f"TP {m['tp']} FP {m['fp']} FN {m['fn']} TN {m['tn']}")
    json.dump(out, open(os.path.join(CACHE, "full_metrics_s8.json"), "w"), indent=2)
    print("saved cache/full_metrics_s8.json")


if __name__ == "__main__":
    main()
