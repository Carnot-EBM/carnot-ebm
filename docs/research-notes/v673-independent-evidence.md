# V673 independent evidence audit

Exp7734 reads the V673 producer artifacts and the saved Exp7732 prediction,
event, and complete-static bank files. It invokes no experimental model. The
terminal JSON records exact byte hashes and the cold-reader commands.

| Gate | Source | Audit rule | Current interpretation |
|---|---|---|---|
| Producer custody | Exp7727–Exp7733 terminal JSON | Exact byte hash, terminal verdict, class, and adversarial flag | Exp7731 and Exp7733 are conductor pre-gate blocks; their science is unavailable. Exp7730 and Exp7732 are disqualified producers. |
| Prior exposure | Exp7727 development corpus | Preserve exposed RAGTruth family identities | Development only; no fresh generalization. |
| Pilot model custody | Exp7729 Qwen pilot | Record pilot receipts as upstream history | The audit itself loads no model and spends zero tokens. |
| Paired predictions | Exp7732 raw `rows.json` | Same family, input hash, source hash and all three arms | Each family contributes one independent unit. |
| Pre-label order | Exp7732 raw `rows.json` and `event_rows.json` | Prediction precedes feedback; one prediction and feedback per family and arm | Future labels cannot justify a prediction. |
| Admission and retention | Exp7732 raw events and producer frozen decision rows | One-use admission, disjoint roles, frozen Brier arithmetic, false-accept bounds and restart parity | Fixture accounting only while the producer remains disqualified. |
| Complete-static closure | Exp7732 `bank_complete_static.json` and producer dictionary | Rebuild eight primitives and 28 pairs; compare saved weights and reject dynamic templates or admissions | Exact fixture certification is circular. |
| Terminal validity | Exp7734 private mutations, cold CLI, adversarial reader, strict row reader | Reject leaked label, swapped source, missing arm, contradictory Brier aggregate and flagged upstream input | Failed readers open no readiness gate. |

The audit's observed 96 Exp7732 fixture families are useful for checking
accounting. They do not repair Exp7732's failed validation or create missing
Exp7731 and Exp7733 development evidence. Brier scores from those fixtures
remain descriptive. Decision cost, retention benefit and efficiency readiness
are unmeasured. The V672 alignment fixture remains circular positive.
