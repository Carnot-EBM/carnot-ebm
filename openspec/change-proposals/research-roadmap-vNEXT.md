# Carnot Research Roadmap vNEXT: Identifiable Constraint Energy and Production-Shadow Self-Learning

**Created:** 2026-08-19

**Milestone:** 2026.08.557

**Status:** Planned

**Supersedes:** milestone 2026.08.556, experiments 6460-6472

**Primary evidence:** V556 artifacts, `ops/conductor-log.md`,
`ops/exclusion_manifest.yaml`, and the V557 source refresh in
`research-references.md`

The architecture document was last reconciled on 2026-07-03. It is stale under
the repository's 30-day rule. This roadmap uses it only for stable component
names. V556 artifacts and current operational records define live behavior.

## 1. What milestone 2026.08.556 proved

V556 completed all 13 tasks. It repaired the unique-event continuous-learning
evidence path and produced a narrow ARC safety result. It did not earn the
planned constraint-routing science claim.

| Area | V556 evidence | What it proves | Remaining limit |
|---|---|---|---|
| Queue and handoff | Exp6460 preserved the V555 terminal facts but set the queue-integrity score to zero after the full Python suite reported 209 failures and 14 errors. | A queue artifact can fail closed without suppressing independent science branches. | Exp6460 repeated a retired transition shape. V557 does not propose another queue-transition experiment. |
| Source receipt | Exp6461 preserved primary-source rows. Its verdict began `blocked_primary_source_receipt` because sources are not execution oracles. | Source ingestion stayed separate from product, model, hardware, and ARC claims. | The latest arXiv release and Extropic's August stack update still need a new dated receipt. |
| Raw persistence | Exp6462 passed its live canary across all three mandatory GGUF families. | Atomic nonzero raw files and one-event/one-path/one-hash binding work. | This proof did not repair the corpus label contract. |
| SOTA corpus | Exp6463 wrote 12 MB of evidence and retained selection headroom in every partition. It set `sota_corpus_ready_score=0` because `label_commitments` failed. | The generation path, rows, and headroom exist. | No result may treat the corpus as held evidence unless an independent forensic proves that labels and membership were sealed before inference. |
| Constraint routing | Exp6464 and Exp6466 were gate-blocked. Exp6465 and Exp6467 were skipped. The capstone retired the repeated Exp6464 and Exp6466 blocked shapes. | The gate cascade prevented invalid science. | The monolithic fixed-policy corpus lineage has no eligible causal result. It must be forensically salvaged or retired, not regenerated again. |
| Continuous self-learning | Exp6468, Exp6469, and Exp6470 passed. The verifier-bounded arm achieved 15/18 development, 30/30 prospective-update, and 24/24 future-held exact outcomes while controls stayed at zero. Fifteen corrupt events were blocked after restart. | A unique-event external factor learner can improve future exact yield with exact veto, rollback, restart, and independent row recomputation. | The result is experiment-local and uses one controlled task family. It is not wired into the production verification pipeline and has no cross-domain capacity or interference result. |
| ARC safety | Exp6471 passed without a solve claim. Mean objective reachability was 0.45 with and without the shield. The shield restored the `g50t` fallback and improved the safety roster from 8/16 to 9/16. | A generic fallback can prevent a narrow safety regression without inventing a solve. | The shield was evaluated in an experiment harness. It is not an opt-in component on the executed E3 decision path. |
| Capstone | Exp6472 set `science_claim_eligible=false`, `continuous_learning_claim_eligible=true`, and `arc_claim_eligible=true`. | The eligible claims are narrow and replayable. | Carnot still lacks a production-shadow self-learning result and an identifiable held exact-energy decision gain. |

The main V556 lesson is contractual. Candidate headroom does not rescue a
broken held-label commitment. A later gate cannot prove an earlier seal.

## 2. The three largest gaps to the PRD vision

### Gap 1: Continuous self-learning is not yet a production capability

FR-11 requires a closed loop that observes verifier outcomes, updates useful
constraints, preserves prior capability, and improves later decisions. V556
proved this in one experiment harness. V557 must add a default-off shadow
adapter to `VerifyRepairPipeline`, then test a chronological cross-domain
stream with bounded memory, restart, corruption, retention, and exact write
admission.

### Gap 2: Exact constraint energy lacks an identifiable, backend-neutral gain

The PRD calls for symbolic constraints, scalar energy, and deterministic
verification. Current exact checkers are domain-specific. The failed V556
corpus cannot support a held claim. V557 therefore separates two questions:

1. Can one typed finite-domain record translate to exact backends without
   semantic drift?
2. Does its frozen scalar violation energy improve held final selection over
   matched controls under an identifying observation protocol?

This branch uses immutable solver-grounded units. It does not depend on the
failed SOTA corpus and does not grant learned energy release authority.

### Gap 3: ARC safety is not on the executed live decision path

V556 produced a valid shield experiment, but the live agent cannot yet opt in
to that shield through the E3 path. V557 must wire one default-off component,
preserve baseline behavior when disabled, and record whether the shield
actually changes an executed action. The held test uses runtime traces only.
It makes no game or level solve claim.

## 3. Research hooks from the V557 refresh

| Source | V557 hook | Boundary |
|---|---|---|
| Sampling-Verification Danger Law, arXiv:2608.17956 | Measure support coverage and preserve rare-mode collision witnesses. | A sampled pass certifies sampled support only. |
| Protocol-Level Identifiability Audit, arXiv:2608.13326 | Add a zero-inference audit before causal routing experiments. | A non-identifying protocol blocks inference and the claim. |
| CPMpy transformation waterfall, arXiv:2608.15143 | Define a small backend-neutral finite-domain record and exact translation receipts. | Exact backend results remain authoritative. |
| MoE hallucination signals, arXiv:2608.17687 | Record router diagnostics only if the existing local runner exposes them. | LLM-judge-trained signals cannot release answers and do not justify runner surgery. |
| Constrained-decoding decomposition, arXiv:2608.13959 | Separate readable form from semantic correctness. | Retired grammar and finite-ID result lanes stay closed. |
| EB-CaP online energy cache, arXiv:2608.06467 | Test confidence, diversity, capacity, and interference gates on an exact-admitted factor cache. | Cache scores may rank or abstain only. |
| ER-KAN, arXiv:2608.14773 | Retain noise-degradation ratio for later compact-energy work. | No KAN retraining is scheduled in V557. |
| Extropic August stack update | Preserve a software/API capability receipt and fixed-width factor ABI assumptions. | No authenticated TSU path exists, so V557 makes no hardware performance claim. |

Semantic Scholar exposed 30 arXiv-indexed EBT citations and seven ARM-EBM
citations on 2026-08-19. The new records do not supply an exact local verifier.
OpenReview, Hugging Face Papers, GitHub Trending, and Logical Intelligence do
not change the authority boundary or dependency choices.

## 4. V557 architecture

```text
                                      exact authority boundary
                                                │
                                                ▼
┌──────────────────────────┐        ┌──────────────────────────┐
│ Mandatory local GGUFs    │        │ Exact local checker      │
│ unique raw event stream  │───────▶│ pass / veto / abstain    │
└────────────┬─────────────┘        └────────────┬─────────────┘
             │ immutable receipt                  │
             ▼                                    │ verified event
┌──────────────────────────┐                       ▼
│ VerifyRepairPipeline     │        ┌──────────────────────────┐
│ shadow adapter, default  │◀───────│ Bounded factor cache     │
│ off; baseline preserved  │        │ confidence + diversity   │
└────────────┬─────────────┘        │ exact admission only     │
             │ proposal scores      └────────────┬─────────────┘
             ▼                                    │ future units only
┌──────────────────────────┐                       │
│ Candidate rank / abstain │───────────────────────┘
│ never release authority  │
└──────────────────────────┘

Independent exact-energy lane:

typed finite-domain record ─▶ translation waterfall ─▶ exact backends
            │                                                │
            └──────── scalar violation energy ───────────────┘
                              │
                              ▼
                 identifying held selection A/B

Independent ARC lane:

runtime E3 features ─▶ default-off generic shield ─▶ executed action receipt
       │                         │
       └──────── baseline ───────┴─▶ held safety and reachability reducer
                                      no source, adapter, or solve claim
```

Only deterministic local checkers authorize release and cache writes. The
factor cache, scalar energy, LLM, and ARC shield may propose, rank, veto, or
abstain within their declared scope.

## 5. Phase 0 - Evidence boundary and method integrity

### Exp6473 - V556 terminal evidence and retirement boundary

Freeze the 13 V556 terminal artifacts. Recompute the capstone eligibility
states and materialize the retirement boundary for the failed corpus chain.
Do not validate or activate a staged queue. This is the first infrastructure
slot.

Deliverable:
`results/experiment_6473_v556_terminal_evidence_and_retirement_boundary.json`

### Exp6474 - Protocol-identifiability and receipt-conformance preflight

Implement a reusable zero-inference audit over a finite policy class,
observation support, and target effect. Emit constructive collision witnesses
when the protocol is non-identifying. Validate the task-scoped receipt phases
needed by V557. This is the second infrastructure slot.

Deliverable:
`results/experiment_6474_protocol_identifiability_and_receipt_preflight.json`

### Exp6475 - V557 primary-source and product-state receipt

Record the 2026-08-14 through 2026-08-18 arXiv release, EBT and ARM-EBM
citation trails, OpenReview, Hugging Face Papers, GitHub Trending, Extropic,
Logical Intelligence, and the rendered ARC leaderboard. Sources are not
execution oracles.

Deliverable:
`results/experiment_6475_v557_primary_source_and_product_state.json`

### Exp6476 - V556 corpus label-commitment forensic

Inspect the existing Exp6463 bytes without new inference. Compare label,
partition, prompt, raw-output, checkpoint, file-time, and git receipts. Set a
salvage score only if the on-disk evidence proves that labels and membership
were committed before the first generation. Otherwise retire the corpus
lineage. No downstream task depends on salvage.

Deliverable:
`results/experiment_6476_v556_corpus_label_commitment_forensic.json`

## 6. Phase 1 - Identifiable exact constraint energy

### Exp6477 - Backend-neutral finite-domain constraint record

Define a small typed record for finite integer domains, logical composition,
negation, arithmetic comparisons, and explicit objective terms. Translate it
to two existing exact backends where available. Compare exact satisfiability,
witnesses, violation sets, and scalar violation energy on immutable cases.

Deliverable:
`results/experiment_6477_backend_neutral_exact_constraint_record.json`

### Exp6478 - Held exact-energy final-selection A/B

Run only after Exp6474 proves the protocol identifying and Exp6477 proves
backend parity. Freeze the scalar energy, tie rules, candidate bytes, and held
manifest. Compare first candidate, random, shuffled energy, violation count,
and exact-energy selection. The final exact checker supplies the headline.

Deliverable:
`results/experiment_6478_identifiable_held_exact_energy_selection.json`

## 7. Phase 2 - Production-shadow continuous self-learning

### Exp6479 - Default-off factor-cache shadow adapter

Wire the V556 unique-event learner into `VerifyRepairPipeline` as an additive,
default-off shadow adapter. Disabled behavior must be byte-for-byte compatible.
Enabled shadow mode records proposed rank changes but cannot release an answer
or admit a write without the exact checker.

Deliverable:
`results/experiment_6479_verify_repair_factor_cache_shadow_adapter.json`

### Exp6480 - Cross-domain prospective continuous self-learning

This is the required continuous self-learning experiment. Use all three
mandatory GGUF families on a chronological arithmetic, code, and finite-domain
constraint stream. Compare frozen, ungated cache, and exact-admitted bounded
cache arms on future units. Use one raw generation event once. Updates affect
future units only.

Deliverable:
`results/experiment_6480_cross_domain_production_shadow_csl.json`

### Exp6481 - Capacity, interference, corruption, and restart stress

If Exp6480 is ready, freeze its update rule. Test several cache capacities,
authority conflicts, supersession, retrieval collisions, forged passes,
wrong-unit bindings, rollback, and a real process restart. Run new held future
generations with the mandatory GGUF families.

Deliverable:
`results/experiment_6481_bounded_factor_cache_interference_restart.json`

### Exp6482 - Independent production-shadow CSL audit

Remain ungated. Recompute Exp6479 through Exp6481 from raw files, receipts, and
event logs. Check default-off compatibility, unique events, exposure order,
exact-veto order, capacity accounting, rollback, restart, protected retention,
and row aggregates.

Deliverable:
`results/experiment_6482_independent_production_shadow_csl_audit.json`

## 8. Phase 3 - Live ARC safety and capstone

### Exp6483 - Default-off ARC E3 shield integration

Move the Exp6471 generic shield into the live E3 decision surface behind an
explicit default-off option. Preserve baseline imports, actions, and latency
when disabled. Record a structured decision receipt when enabled. Do not add
game-specific thresholds or adapters.

Deliverable:
`results/experiment_6483_arc_e3_default_off_safety_shield_integration.json`

### Exp6484 - Held executed-policy ARC influence A/B

This is the milestone's ARC generalization slot. Run only after Exp6483 passes.
Use hidden runtime traces and leave-one-game-out splits. Compare disabled,
enabled, ablated, and shuffled shields. Credit only rows where the live E3 path
imports the component and the decision receipt proves an executed action
changed. Make no game or level solve claim.

Deliverable:
`results/experiment_6484_arc_held_executed_shield_influence_ab.json`

### Exp6485 - V557 independent adversarial capstone

Recompute every requested claim from rows and raw evidence. Check retirement,
identifiability, exact backend parity, held selection, production-shadow
compatibility, CSL chronology, lifecycle safety, ARC live reachability, and
protected files. Reconcile specs and operations documents only from eligible
determinations.

Deliverable:
`results/experiment_6485_v557_adversarial_capstone.json`

## 9. Dependency graph

```text
Exp6473 evidence boundary ──────────────────────────────────────────┐
Exp6475 source receipt ─────────────────────────────────────────────┤
Exp6476 corpus forensic ────────────────────────────────────────────┤
                                                                   │
Exp6474 identifiability ───────┐                                   │
                              ├─▶ Exp6478 held exact-energy A/B     │
Exp6477 exact record parity ───┘                                   │
                                                                   │
Exp6479 shadow adapter ─▶ Exp6480 cross-domain CSL ─▶ Exp6481 stress
            │                      │                       │        │
            └──────────────────────┴──────────────┬────────┘        │
                                                  ▼                 │
                                          Exp6482 audit             │
                                                                   │
Exp6483 ARC live wiring ─▶ Exp6484 held executed influence ─────────┤
all terminal evidence ─────────────────────────────────────▶ Exp6485 capstone
```

Exp6482 and Exp6485 remain conductor-ungated. They must report missing,
blocked, or failed upstream evidence instead of disappearing with a cascade.
Exp6476 has no downstream gate because a forensic cannot create a missing
pre-inference commitment.

## 10. Hardware requirements

| Experiments | Hardware | Requirement |
|---|---|---|
| Exp6480, Exp6481 | Two local RTX 3090 GPUs | Use cached `unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and `unsloth/gemma-4-26B-A4B-it-GGUF` files with embedded tokenizers. Record device, runner, model-file, tokenizer, phase, and CPU-fallback receipts. |
| Exp6473, Exp6474, Exp6476-Exp6479, Exp6482-Exp6485 | CPU and local SSD | Run receipt audits, exact solver parity, immutable replay, production wiring tests, ARC runtime traces, and capstone reducers. |
| Exp6475 | Network plus local source cache | Hash primary pages and preserve timestamps. Network or authentication failure must produce a source-specific blocked row. |
| Extropic, FPGA, NPU, Ising board | Not required | Z1 hardware is planned for 2027. Carnot has no authenticated TSU route. Existing FPGA and NPU states have not changed. No board redesign, unchanged probe, power claim, or latency claim is scheduled. |

The dual RTX 3090 pair and current CPU/storage are sufficient. No wishlist
purchase blocks V557.

## 11. Promotion gates and stop rules

- Exp6473 may record a queue problem, but it does not validate a staged queue
  and cannot suppress independent tasks.
- Exp6476 may salvage the corpus only from evidence that already existed before
  inference. A new manifest or reconstructed timestamp is not a commitment.
- A non-identifying protocol emits a collision witness and blocks Exp6478
  before candidate evaluation.
- Backend parity requires matching satisfiability, witness validity, and
  violation semantics. Matching scalar totals alone are insufficient.
- The exact checker is the final oracle for Exp6478 and every cache write.
- The production shadow adapter is default off. Disabled behavior must remain
  compatible with the pre-V557 path.
- A held CSL unit must not appear in development, update events, cache entries,
  prompt examples, or raw-output hashes.
- Cache capacity includes tombstones and quarantine state. Eviction cannot
  resurrect a revoked factor.
- Every comparative task emits one row per unit, arm, model, seed, or condition
  needed to recompute the headline.
- A `blocked_*` verdict names the failed check and observed value in
  `gate_check_summary`.
- Exp6484 cannot read game source, use a hand adapter, update the ARC solve
  registry, or claim a game or level solve.
- Learned scores, LLM judges, energy, cache confidence, and source pages never
  become release oracles.
- If a changed rerun repeats its declared prior failure, its
  `retire_if_same_verdict` rule retires that scope.

## 12. Expected milestone decision

V557 can end in four honest states:

1. The V556 corpus lacks a historical label commitment. Retire that lineage
   and preserve its raw bytes only as non-held development evidence.
2. The exact record has backend parity, but scalar energy gives no held final-
   decision gain. Preserve the null and retain the record as verification
   infrastructure.
3. The production-shadow learner improves future exact yield across domains
   without retention or lifecycle failure. Promote only that bounded external-
   memory result.
4. The live ARC shield prevents a held safety regression but does not improve
   mean reachability. Promote it only as a default-off safety fallback.

The milestone succeeds if it produces trustworthy, independently recomputable
determinations. Positive effects are not required.
