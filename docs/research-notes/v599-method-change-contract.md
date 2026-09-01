# V599 Method-Change Evidence Contract

Date: 2026-09-01

Spec: `REQ-REPORT-6848`

## Scope

This contract freezes terminal V598 evidence. It does not rerun a V598
mechanism. It does not invoke an LLM. The inference substrate is deterministic
CPU evidence replay.

The machine-readable contract is
`results/experiment_6848_v599_method_change_evidence_contract.json`. Its
`v599_evidence_contract_ready_score` is the exact gate for Exp6849, Exp6850,
Exp6853, and Exp6857. The score measures evidence completeness. It does not
measure scientific benefit.

## Controlling V598 Results

Exp6836 reports compile parity and two readiness scores as ready. Its rows
contain 16 candidate occurrences but only 8 unique candidate identities. Each
identity occurs twice. Its compile manifest also contains 16 receipts for 8
unique candidate identities.

The independent Exp6847 capstone recomputed compile parity as false. It
recomputed both readiness scores as zero. Exp6847 controls V599 when the two
artifacts disagree. The semantic checks in each stored compile receipt still
pass. That local result does not repair the duplicated identity authority.

The following terminal results stay frozen:

| Source | Frozen result | V599 boundary |
|---|---|---|
| Exp6837 | No scientific rows; exclusive lease and live canary failed | Separate resource admission from later scoring |
| Exp6842 | 540 held-future rows; 5 wins; 69 losses; mean effect -0.118519 | Retire the nonselective residual-memory rule |
| Exp6844 | 60 action rows; all 60 have zero headroom | Require nonzero matched headroom before effect credit |
| Exp6845 | 0 first-party obligations across 20 zero-obligation strata | Require a first-party obligation before transport or utility credit |
| Exp6838 | Artifact absent after a conductor `GATE_BLOCK` | Preserve the skip; do not invent an audit result |

Audit completeness is not scientific benefit. Exp6842 keeps its terminal null
class and its harmful scientific disposition. Exp6844 and Exp6845 keep their
blocked effect dispositions.

## Frozen Sources And Dynamic Discovery

Frozen V598 artifacts use their repo-relative path and SHA-256 hash. Hash drift
blocks the contract. Missing, invalid, or nonterminal required sources also
block it. The Exp6838 absence is allowed only with the hash-bound conductor
skip line.

Future ARC evidence follows different rules. An audit discovers candidates at
execution time. A stored absolute path is descriptive only. It cannot select a
future source. The execution-time manifest must provide a repo-relative path,
artifact family, terminal class, and file hash.

Each qualified record also carries these provenance groups:

- Generator identity: generator ID, implementation hash, and configuration hash.
- Model identity: model ID, model artifact hash, and tokenizer hash.
- Process ownership: PID, parent PID, start time, command hash, task lease, and teardown receipt.
- Exact outcome authority: receipt ID, time, before and after state hashes, outcome, and source hash.
- Conductor skip: task, time, outcome, reason, upstream task, and exact log-line hash.

Missing provenance makes a dynamic record ineligible. Records from different
games, models, policies, budgets, supervisors, or tool-loop states cannot be
pooled.

## Changed Mechanisms

V599 changes the failed mechanism in each open branch:

- Exp6849 uses fresh identities and isomorphic invariance for typed authority.
- Exp6850 owns GPU leases and runs one canary per model before scientific scoring.
- Exp6853 introduces verified-memory, no-memory, and abstain opportunities.
- Exp6857 discovers provenance-qualified ARC receipts at execution time.

The contract adds no novelty dependency. Exact checkers and exact later
outcomes remain external authorities.

## Primary Reference Verification

Each identifier below resolved on its primary arXiv page on 2026-09-01.
Access is context-only. No package, repository, model, or service dependency was
added.

| Identifier and primary title | Date | Method delta | Carnot hook | Access boundary |
|---|---|---|---|---|
| [arXiv:2604.27283 — Learning When to Remember: Risk-Sensitive Contextual Bandits for Abstention-Aware Memory Retrieval in LLM-Based Coding Agents](https://arxiv.org/abs/2604.27283) | 2026-04-30 | Memory, no-memory, and abstention are risk-sensitive actions | Replace the harmful residual rule with a bounded exact-outcome selector | Context only; no dependency |
| [arXiv:2604.15149 — LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking](https://arxiv.org/abs/2604.15149) | 2026-04-16 | Isomorphic Perturbation Testing detects extensional shortcuts | Require identity, label, permutation, and duplicate-removal invariance | Context only; no dependency |
| [arXiv:2607.16999 — Counterfactual Shapley Credit Assignment](https://arxiv.org/abs/2607.16999) | 2026-07-18 | Counterfactual coalitions separate policy effects from luck | Credit only valid coalitions with exact outcomes and headroom | Context only; no dependency |
| [arXiv:2608.11994 — Claim-Level Reliability Assessment for Efficient Test-Time Reasoning](https://arxiv.org/abs/2608.11994) | 2026-08-12 | Targeted falsification verifies decision-critical claims | Keep atoms and action receipts as evidence units | Context only; no dependency |
| [arXiv:2606.19808 — Think Again or Think Longer? Selective Verification for Budget-Aware Reasoning](https://arxiv.org/abs/2606.19808) | 2026-06-18 | Selective verification compares preserve and intervention | Compare intervention, preserve, and abstain on matched opportunities | Context only; no dependency |
| [arXiv:2608.31046 — Does On-Policy Distillation Really Distill? From Noisy Teacher to Self-Improvement](https://arxiv.org/abs/2608.31046) | 2026-08-31 | Some teacher-attributed gains can come from tail-token suppression | Keep model updates as controls and freeze GGUF weights | Context only; no dependency |
| [arXiv:2608.30461 — From Final Artifacts to Trajectories: Retrospective Process Supervision for Evidence-Grounded Long-Form Generation](https://arxiv.org/abs/2608.30461) | 2026-08-31 | RetroGen reconstructs candidate traces from final artifacts | Reconstruction can diagnose but cannot replace live receipts | Context only; no dependency |
| [arXiv:2608.29596 — Towards a Systems Foundation for Agentic Skills: Architecture, Lifecycle, and Security](https://arxiv.org/abs/2608.29596) | 2026-08-30 | The paper defines a lifecycle for procedural skill artifacts | Keep source, reachability, intervention, outcome, and retirement together | Context only; no dependency |

Two planner labels did not match their primary records. The planner called
`2608.30461` “RetroGen: Scaling Retrospective Process Supervision for Reliable
Agentic Reasoning.” It called `2608.29596` “Agentic Skills in the Wild: A
Comprehensive Study of Reusable Skill Artifacts” and dated it 2026-08-28. The
primary pages show the titles and dates in the table. The identifiers still
resolve. The JSON preserves both mismatches.
