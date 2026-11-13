"""Regenerate every number, table and figure in the paper from the cached embeddings.

Runs the analysis scripts in dependency order and stops at the first failure. Each
script's output goes to output/make_paper_numbers/<script>.log; results land in cache/.

  python make_paper_numbers.py            # every CPU step (about 20 minutes)
  python make_paper_numbers.py --gpu      # also the occlusion maps (needs a GPU)
  python make_paper_numbers.py --list     # show the steps

Prerequisite: the embedding database built by the extraction scripts (see README).
"""
import os
import subprocess
import sys
import time

import config

STEPS = [
    ("base4_numbers.py",             "Table III, Table components, Table sota (MammoFF rows)"),
    ("bootstrap_ci.py",              "95% confidence intervals for every main-result metric"),
    ("threshold_rules.py",           "Supplementary Table S1 (choice of the 0.5 threshold)"),
    ("exp_cbis_enhance.py",          "Supplementary Table S2 (sensitivity to C)"),
    ("finetune_eval_full4.py",       "Table finetune: frozen and 1-of-4 fine-tuned rows"),
    ("finetune_eval_phase2.py",      "Table finetune: 3-of-4 fine-tuned row and GPU-hours"),
    ("finetune_paired_bootstrap.py", "Paired AUC-difference CIs, frozen vs fine-tuned"),
    ("scanner_confound_check.py",    "Section IV-E site check and Table scanner"),
    ("base4_scaling.py",             "Database-growth curves (both datasets)"),
    ("make_confusion_json.py",       "Confusion-matrix counts (representative seed)"),
    ("retrieval_recheck.py",         "Section IV-A retrieval baseline (nearest-neighbour AUCs)"),
    ("benchmark_mvlp.py",            "Table cost: classifier fit time, parameters, memory"),
    ("paper_figures.py",             "All figures except occlusion maps"),
]
GPU_STEPS = [
    ("saliency_maps.py",             "Occlusion-sensitivity figures (re-encodes images: needs GPU)"),
]


def main():
    steps = STEPS + (GPU_STEPS if "--gpu" in sys.argv else [])
    if "--list" in sys.argv:
        for i, (f, what) in enumerate(STEPS + GPU_STEPS, 1):
            print(f"{i:2d}. {f:30s} {what}")
        return
    logdir = os.path.join(config.PROJECT_DIR, "output", "make_paper_numbers")
    os.makedirs(logdir, exist_ok=True)
    stale = os.path.join(config.PROJECT_DIR, "cache", "threshold_rules_scores.pkl")
    if os.path.exists(stale):
        os.remove(stale)              # force threshold_rules.py to recompute its scores
    t_all = time.time()
    for i, (f, what) in enumerate(steps, 1):
        t = time.time()
        print(f"[{i}/{len(steps)}] {f}  ({what})", flush=True)
        with open(os.path.join(logdir, f.replace(".py", ".log")), "w") as log:
            r = subprocess.run([sys.executable, f], cwd=config.PROJECT_DIR,
                               stdout=log, stderr=subprocess.STDOUT)
        if r.returncode:
            sys.exit(f"FAILED: {f} (exit {r.returncode}); see {logdir}")
        print(f"      done in {time.time() - t:.0f}s", flush=True)
    print(f"All {len(steps)} steps finished in {(time.time() - t_all) / 60:.1f} min; "
          f"logs in {logdir}")


if __name__ == "__main__":
    main()
