"""Regenerate every number, table and figure in the paper from the cached embeddings.

Runs the analysis scripts in dependency order and stops at the first failure. Each
script's output goes to output/make_paper_numbers/<script>.log; results land in cache/.

  python make_paper_numbers.py            # every CPU step
  python make_paper_numbers.py --from 14  # resume from step 14
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
    # --- added in the revision (reviewer comments in brackets) ---
    ("dataset_composition.py",       "Table data: patients/breasts/images per partition, overlap check (R1.1, R1.2)"),
    ("high_sensitivity_points.py",   "Limitations: 90% / 95% sensitivity operating points (R1.10)"),
    ("components_ci.py",             "Table components: paired differences vs early fusion (R1.8)"),
    ("ablation_e1.py",               "Table ablation: single extractors, leave-one-out, components (R2.2, R2.3, R2.7)"),
    ("clean_ablation.py",            "Supplementary Table S3: leave-one-out within the three native encoders"),
    ("mammofm_combos2.py",           "Mammo-FM in place of MammoScreen (R2.2)"),
    ("bootstrap_ci_mammofm.py",      "Table sota: Mammo-FM row with CI (R2.2)"),
    ("finetune_lora_eval.py",        "Table finetune: LoRA row (R2.6)"),
    ("finetune_resnet50_eval.py",    "Table finetune: end-to-end ResNet-50 row (R2.5)"),
    ("full_metrics_s8.py",           "Supplementary Table S8: full metrics and confusion counts (R1.7)"),
    ("inbreast_eval.py",             "Table inbreast-regimes, Supplementary Table S5 (R1.9, R2.4)"),
    ("inbreast_released_compare.py", "Section IV-F: MammoScreen as released vs with the classifier (INbreast)"),
    ("inbreast_external_full.py",    "Table inbreast-sota: INbreast as a fully external test"),
    ("inbreast_rsna_relabel.py",     "Supplementary Table S4a: RSNA label check"),
    ("inbreast512_eval.py",          "Supplementary Table S4b: INbreast through the RSNA pipeline"),
    ("rsna_train_eval.py",           "Supplementary Table S6: film test sets with added training data"),
] + [(f"encoder_eval.py {e}", "Supplementary Table S7: further frozen encoders (R2.2)") for e in (
    "convnext_base_noclahe", "convnextv2_base_noclahe", "davit_base_noclahe", "densenet121_noclahe",
    "dinov2_vitb14_r518_noclahe", "dinov2_vits14_r518_noclahe", "dinov3_vitb16_r512_noclahe",
    "dinov3_vitl16_r512_noclahe", "efficientnet_b5_noclahe", "mammo_clip_b2", "mammo_fm",
    "radimagenet_densenet121", "radimagenet_resnet50", "resnet50_noclahe", "vit_base_noclahe")]

# Steps that read the image files rather than the cached features (duplicate search, R2.4)
# or need a GPU; run with --gpu.
GPU_STEPS = [
    ("saliency_maps.py",             "Occlusion-sensitivity figures (re-encodes images: needs GPU)"),
    ("duplicate_search.py",          "Section III-C: perceptual-hash search for shared films"),
    ("duplicate_verify.py",          "Section III-C: pixel-level verification"),
    ("duplicate_orb.py",             "Section III-C: keypoint matching with geometric verification"),
    ("inference_timing.py",          "Section III-J: inference latency and memory (needs GPU)"),
    ("extraction_throughput.py",     "Table cost: extraction throughput (needs GPU)"),
]


def main():
    steps = STEPS + (GPU_STEPS if "--gpu" in sys.argv else [])
    if "--list" in sys.argv:
        for i, (f, what) in enumerate(STEPS + GPU_STEPS, 1):
            print(f"{i:2d}. {f:30s} {what}")
        return
    if "--from" in sys.argv:          # resume: --from <step number>
        steps = steps[int(sys.argv[sys.argv.index("--from") + 1]) - 1:]
    logdir = os.path.join(config.PROJECT_DIR, "output", "make_paper_numbers")
    os.makedirs(logdir, exist_ok=True)
    stale = os.path.join(config.PROJECT_DIR, "cache", "threshold_rules_scores.pkl")
    if os.path.exists(stale):
        os.remove(stale)              # force threshold_rules.py to recompute its scores
    t_all = time.time()
    for i, (f, what) in enumerate(steps, 1):
        t = time.time()
        print(f"[{i}/{len(steps)}] {f}  ({what})", flush=True)
        with open(os.path.join(logdir, f.replace(".py", "").replace(" ", "_") + ".log"), "w") as log:
            r = subprocess.run([sys.executable] + f.split(), cwd=config.PROJECT_DIR,
                               stdout=log, stderr=subprocess.STDOUT)
        if r.returncode:
            sys.exit(f"FAILED: {f} (exit {r.returncode}); see {logdir}")
        print(f"      done in {time.time() - t:.0f}s", flush=True)
    print(f"All {len(steps)} steps finished in {(time.time() - t_all) / 60:.1f} min; "
          f"logs in {logdir}")


if __name__ == "__main__":
    main()
