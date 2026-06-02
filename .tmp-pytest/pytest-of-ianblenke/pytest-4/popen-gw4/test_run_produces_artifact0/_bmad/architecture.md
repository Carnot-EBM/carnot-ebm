## Verification Pipeline Tiers

| Tier | Name | Class | Cost | Signal Source | Skip Condition |
|------|------|-------|------|---------------|----------------|
| 2 | VJEPA v2 | `VariationalJEPAPredictor` | ~10 ms | CoT violation prediction (variational, KL-regularised, OOD AUC=0.4561, Exp 883/884, deployed 2026-06-02) | `energy < vjepa_threshold` |

 Tier 2 updated to VJEPA v2 (VariationalJEPAPredictor, OOD AUC=0.4561) by Exp 884 on 2026-06-02 (REQ-VERIFY-145); prior Tier 2 was EORMModel (55M-param CoT energy reward model, trained in Exps 340/341/355/359). Each tier returns early if it can clear the response.
