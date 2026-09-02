# V600 method-change contract

Date: 2026-09-02  
Requirement: `REQ-REPORT-6861`

## Purpose

V600 starts from an immutable V599 evidence root. The root closes failed claim
mechanisms without deleting reusable infrastructure. Exp6861 reads terminal
JSON and conductor gate records only. It does not rerun V599 work, import the
Exp6860 reducer, invoke an LLM, load a GGUF model, or enter a live ARC game.

The controlling artifact is
`results/experiment_6861_v600_branch_retirement_evidence_contract.json`.
Exp6862, Exp6867, and Exp6870 each require the exact field
`v600_evidence_contract_ready_score == 1`.

## V599 disposition replay

The fresh reducer preserves infrastructure and scientific claims as separate
records.

| Branch | Recomputed state | Controlling fact |
|---|---|---|
| Typed authority | Positive infrastructure | Exact typed rows retain parity, unique identities, and exact labels. |
| Compatibility | Null scientific claim | All three local model streams completed, but 78 shortcut rows and 8 of 24 isomorphic direction groups fail. |
| Self-learning | Disqualified scientific claim | The held policy abstains on every opportunity. Fresh per-write reduction finds 13 harmful writes. The controlling sealed source is flagged. |
| Supervisor | Blocked claim | There are zero authentic live rows with nonzero pre-action headroom. Exp6858 has an explicit conductor gate record. |
| Tool gap | Partial | The first-party receipt contract is complete on fixtures. It has zero authentic live chains. |
| V599 capstone | Positive procedure only | Exp6860 completed disposition work. It advanced zero scientific branches. |

The self-learning replay does not discard the transaction kernel. Restart and
rollback receipts remain reusable. The compatibility replay does not discard
local scoring. Model, tokenizer, process-owner, lease, and teardown receipts
remain reusable.

## Narrow retirements

V600 retires three claim mechanisms:

1. Raw fixed-sequence margins cannot support a compatibility claim.
2. Supervisor credit cannot rerun without a new nonzero-headroom opportunity.
3. The V599 risk-sensitive policy cannot rerun unchanged.

V600 preserves five infrastructure surfaces:

1. Exact typed authority.
2. Three-family local forced-sequence scoring.
3. Memory transactions.
4. Restart and rollback.
5. First-party tool-gap receipt transport.

## Changed mechanisms

The compatibility branch now uses dual-side, nuisance-matched semantic
contrasts. An exact structure checker and an independent exact solution checker
must agree. Calibration and held groups freeze before scoring. A positive held
claim requires a confidence interval above zero in both model families. It must
also exceed every matched nuisance effect.

The memory branch now uses a decision-time observability firewall. Outcome-only
fields cannot select the action they later label. The controller must use
bounded exploration. It quarantines a write unless pessimistic benefit clears
the harm cost. A positive result requires nondegenerate held actions, positive
exact later effect, fewer harmful writes, durability, portability, and zero
leakage.

The live branch now uses resumable first-party receipt checkpoints. It must
capture an authentic adapter-disabled tool-gap chain before any effect test.
Delivery and withholding can be compared only after the same pre-gap state hash
replays. This replaces the unchanged supervisor rerun. It makes no level-solve
claim.

## Split and provenance contract

Every downstream artifact uses `carnot.v600.evidence_provenance.v1`. The schema
requires these identity classes:

- Local source SHA-256.
- Calibration and held split SHA-256 values.
- Model artifact and embedded tokenizer SHA-256 values.
- Process owner PID, process start time, and lease identity.
- Exact structure, solution, and later-outcome authorities.
- Live receipt identity.
- Conductor skip identity.

The calibration hash field is `calibration_split_sha256`. The held hash field is
`held_split_sha256`. The disjointness field is
`calibration_held_group_overlap_count`. Unavailable evidence stays null. It does
not become numeric zero.

## Primary reference verification

The verification boundary is the primary arXiv abstract and metadata page. No
paper code, private data, model weights, or unreported implementation behavior
is treated as available.

- [arXiv:2606.10616](https://arxiv.org/abs/2606.10616) verifies OSL-MR's strict
  split between online-observable inputs and offline supervision. V600 applies
  that split at each memory decision.
- [arXiv:2605.29556](https://arxiv.org/abs/2605.29556) verifies structure-side
  and solution-side verification. The primary page lists ICML 2026. This
  corrects the V600 planning note's ICLR label.
- [arXiv:2604.09459](https://arxiv.org/abs/2604.09459) now reports 69 papers:
  56 core credit-assignment methods and 13 boundary enablers. This corrects the
  planning note's 47-method count. V600 uses restored-state comparisons and
  binds each credit claim to provenance and falsification.
- [arXiv:2608.15008](https://arxiv.org/abs/2608.15008) verifies a controlled
  memory-substrate harness with three backbones, four suites, and 26 metrics.
  No substrate dominates. Excess retrieval can harm sequential decisions.

These references motivate changed mechanisms. They do not authorize model
self-verification or replace Carnot's exact authorities.
