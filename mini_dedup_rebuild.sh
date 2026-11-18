#!/bin/bash
# Recompute every Mini-DDSM number with shared-film patients removed (config.MINI_EXCLUDE_SHARED_FILMS).
# CBIS-DDSM numbers are unaffected. All-patient outputs backed up in cache/allpatients_2026-11-14/.
cd /f/Development/MamoRAC
export PYTHONIOENCODING=utf-8
L=cache/mini_dedup_rebuild.log
rm -f cache/threshold_rules_scores.pkl          # cached scores were computed on all patients
for s in base4_numbers bootstrap_ci threshold_rules base4_scaling make_confusion_json \
         scanner_confound_check components_ci high_sensitivity_points ablation_e1; do
  echo "$(date +%H:%M) START $s" >> $L
  python $s.py > cache/dedup_$s.out 2>&1
  echo "$(date +%H:%M) $s exit $?" >> $L
done
echo "$(date +%H:%M) ALL DONE" >> $L
