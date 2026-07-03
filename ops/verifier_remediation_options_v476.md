# Verifier Remediation Options V476

Prepared for the operator-decides step in CLAUDE.md's Verifier Authenticity Discipline. No verifier source was modified by this task.

## Audit Context

- Audit report: `ops/verifier_authenticity_audit_report.md` (2026-07-01).
- Audit summary read for context: 20 files scanned, 11 AUTHENTIC, 6 HONEST_HEURISTIC, 2 DISHONEST_NAMING, 1 CANNOT_DETERMINE, 0 ADVERSARIAL_GAMING, 0 OUTRIGHT_FAKE.
- This document reconfirms the two flagged findings from current source instead of relying only on the audit quotes.

## Independent Reconfirmation

```json
{
  "and_composition_verifier": {
    "default_ensemble_exported": true,
    "default_k5_includes_soskan": true,
    "exceptions_masked_as_pass_energy": true,
    "pipeline_records_advisory_certificate": true,
    "pipeline_uses_default_ensemble": true,
    "soskan_score_cap_present": true,
    "untrained_soskan_returns_neutral_0_5": true
  },
  "claim_isolation_uncertainty_router": {
    "artifact_only_json_inputs": true,
    "claim_isolated_accept_copied_from_manifest_or_validator": true,
    "fixed_uncertainty_scores": true,
    "imported_modules": [
      "__future__",
      "collections",
      "collections.abc",
      "json",
      "pathlib",
      "typing"
    ],
    "model_substrate_imports": [],
    "no_actual_claim_verifier_call": true,
    "no_model_substrate_imported": true,
    "routing_threshold_policy": true
  }
}
```

## and_composition_verifier.py

### RENAME_TO_REFLECT_REALITY

RENAME_TO_REFLECT_REALITY: rename the public surface to `AdvisoryK5VerifierAdapterHarness` (or `advisory_k5_verifier_adapter_harness.py`). That name matches the current behavior: an advisory adapter/certificate harness over mixed heuristic and model-shaped members, not a production trained k=5 energy ensemble with guaranteed exponential null-space shrinkage.

### RETIRE

RETIRE: remove the default AND-compose certificate from VerifyRepairPipeline and update tests/evals that import `build_default_verifier_ensemble()`. The final pipeline verdict should not change because the current certificate is advisory and explicitly does not short-circuit `result.verified`. What breaks is the `and_compose_k5` certificate, k=5 tests, and analysis scripts that partition SOSKAN/SemEnergy versus AST/Semantic/Z3. A remaining k-1 composition can be meaningful only if it is renamed as k=4/k-1 advisory composition and its correlations/thresholds are remeasured; it cannot inherit the current k=5 Exp 1108 claim.

### REIMPLEMENT_PROPERLY

REIMPLEMENT_PROPERLY: load or train a real trained SOSKANEnergyV3 checkpoint on a declared FoVer split, persist the feature normalization stats, calibrate raw energy instead of capping `raw / 2.0`, and remove exception-to-pass masking in favor of an explicit degraded or fail-closed certificate. This also requires recomputing pairwise correlations, thresholds, FoVer AUROC, and the default ensemble tests against the actual energy-model substrate the current adapter only pretends to provide by default.

### Recommendation

REIMPLEMENT_PROPERLY. The repo already wires the ensemble into VerifyRepairPipeline as an advisory certificate, so retiring it would remove useful integration context, while a pure rename would leave an inert SOSKAN member in the default k=5 path. Until reimplementation is funded, quarantine headline claims and describe the current file as an advisory harness only.

## claim_isolation_uncertainty_router.py

### RENAME_TO_REFLECT_REALITY

RENAME_TO_REFLECT_REALITY: rename the public surface to `ClaimIsolationArtifactRoutingLedger` (or `claim_isolation_artifact_routing_ledger.py`). That name states what the code does now: read JSON/JSONL artifacts, copy existing accept booleans, apply uncertainty/prefix-risk routing, and write a ledger.

### RETIRE

RETIRE: remove Exp 1541's artifact generator and update downstream `claim_isolation_router_scale.py`, milestone-retro references, and tests that expect `results/experiment_1541_claim_isolation_uncertainty_router_v2.json`. Live verification should not lose a model call because this module currently performs none, but the larger claim-router cost/safety lineage would lose its bridge from Exp 1525/1537 artifacts into Exp 1553.

### REIMPLEMENT_PROPERLY

REIMPLEMENT_PROPERLY: for every extracted claim selected by the router, perform an actual model call per claim or invoke a real isolated-claim verifier with the original answer hidden, record the prompt/model/cache provenance, compare full-context and isolated decisions on the same case, and keep deterministic SAT/product-line/runtime validators as false-accept authority. Fixed uncertainty constants and copied `claim_isolated_accept` booleans would have to be replaced by measured per-claim verifier outputs.

### Recommendation

RENAME_TO_REFLECT_REALITY. The current module is useful artifact routing glue, and downstream Exp 1553-style scale work may still consume that ledger. Reimplementation would be a larger live-verifier project, while retirement would discard the routing lineage without improving verifier truthfulness as much as an honest name/docstring would.

## Decision Boundary

This package prepares the operator's decision. It does not rename, retire, reimplement, or edit either flagged verifier.
