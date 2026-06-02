## Verification Pipeline Tiers

| Tier | Name | Class | Cost | Signal Source | Skip Condition |
|------|------|-------|------|---------------|----------------|
| 2 | VJEPA v2 | `VariationalJEPAPredictor` | ~10 ms | CoT violation prediction (variational, KL-regularised, OOD AUC=0.6640, Exp 883/884, deployed 2026-04-25) | `energy < vjepa_threshold` |
| 3 | Ising | `VerifyRepairPipeline` | ~0.006 ms/constraint | Full constraint verification | Always runs if tiers 0-2 pass |

 Tier 2 updated to VJEPA v2 (VariationalJEPAPredictor, OOD AUC=0.6640) by Exp 884 on 2026-04-25 (REQ-VERIFY-145); prior Tier 2 was EORMModel (55M-param CoT energy reward model, trained in Exps 340/341/355/359). Each tier returns early if it can clear the response, avoiding subsequent more expensive tiers.
