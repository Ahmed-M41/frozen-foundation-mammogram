"""Spearman rank correlation between the three-encoder classifier score and MammoScreen's
scalar score on the test cases (Section on components). CBIS-DDSM paired seed 0; Mini-DDSM
four seeds pooled. Run: python rank_corr.py [--all-patients]  ->  prints both."""
import sys

from scipy.stats import spearmanr

import config
if "--all-patients" in sys.argv:
    config.MINI_EXCLUDE_SHARED_FILMS = False
import components_ci as CC
import db
import miniddsm_B as M
from exp_msfeat_fusion import cbis_ms

conn = db.connect()
c = CC.collect(conn, ["cbis_full"], lambda s, sp, g, y: (sp == "train", sp == "test"), cbis_ms(), False, [0])
m = CC.collect(conn, ["miniddsm", "miniddsm_normal"],
               lambda s, sp, g, y: (~M.patient_test_mask(g, y, 0.2, s), M.patient_test_mask(g, y, 0.2, s)),
               M.load_ms(["miniddsm", "miniddsm_normal"]), True, [0, 1, 2, 3])
print(f"exclude_shared={config.MINI_EXCLUDE_SHARED_FILMS}  CBIS rho={spearmanr(c['probe3'], c['ms'])[0]:.3f}  "
      f"Mini rho={spearmanr(m['probe3'], m['ms'])[0]:.3f}")
