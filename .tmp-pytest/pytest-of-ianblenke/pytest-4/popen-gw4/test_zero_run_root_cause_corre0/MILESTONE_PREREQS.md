
## Milestone 2026.04.70 Pre-flight

.69 zero-run root cause: yaml_key_error_title

Open RETROs entering .70:
  - RETRO-MANIFEST-FULL-SCOPE: HUMAN_REQUIRED
  - RETRO-SVAMP-ZERO-AUC: TARGETED (Exp 907+908)
  - RETRO-XILINX-TOOLS-UNAVAILABLE: HUMAN_REQUIRED
  - RETRO-INERTIA-SWEEPS-TARGET-MISSED: TARGETED (Exp 914)

Gates:
  Exp 906 (code repair 50q): GATED on results/experiment_905_iterative_self_repair_v1.json
    signed_improvement > 0
  Exp 908 (EstimationVerifier): GATED on results/experiment_907_svamp_root_cause_v2.json
    labeling_mismatch_confirmed == True
  Exp 914 (PIMI sparse final): ABORTS if ops/exclusion_manifest.yaml contains
    experiment_scope matching "iCE40 PIMI research"
