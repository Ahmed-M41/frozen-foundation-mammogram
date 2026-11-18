"""Print one compact table over every cache/encoder_eval_*.json (candidate encoders)."""
import glob
import json

print(f"{'row':38s} | CBIS full AUC  diff [95% CI]          | Mini AUC  diff [95% CI]")
for p in sorted(glob.glob("cache/encoder_eval_*.json")):
    d = json.load(open(p))
    for name in d["cbis"]:
        if name.startswith("All four"):
            continue
        c = d["cbis"][name]["full"]; m = d["mini"][name]["full"]
        print(f"{name:38s} | {c['auc']:.3f} {c['delta']:+.3f} [{c['delta_ci'][0]:+.3f},{c['delta_ci'][1]:+.3f}] | "
              f"{m['auc']:.3f} {m['delta']:+.3f} [{m['delta_ci'][0]:+.3f},{m['delta_ci'][1]:+.3f}]")
