# Research Roadmap vNEXT — Milestone 2026.07.510

**Title:** Prospective Exact-Stream Self-Learning, Live Relational Goal Energy, and One-Axis Rust Portability
**Status:** Proposed
**Task range:** Exp5706–Exp5716
**Execution manifest:** `research-roadmap-next.yaml`
**Supersedes:** Milestone `2026.07.509` planning document after its terminal conductor run

## Milestone thesis

Milestone `.509` closed one major scientific gap and two tempting but negative extensions. The
active-spline KAN controller passed a genuinely independent anytime-valid audit: 4,464 conformal
rows and 14,400 decision rows were recomputed from immutable receipts, worst powered-group
coverage was `0.904762`, the largest pathwise risk upper bound was `0.076237` against a `0.1`
limit, every paired benefit lower bound was positive, poison/restart/retention controls passed,
and exact unsafe accepts remained zero. A disabled-by-default `VerifyRepairPipeline` shadow
adapter also passed replay and rollback checks. That is strong FR-11 evidence, but it is still
derived from a synthetic replay stream and has not governed a prospective stream of outputs from
a current local model.

The ARC counterexample-patched transition model accepted 51 safe patches but failed its
preregistered utility gate, so it is terminally retired; its downstream A/B was correctly skipped.
The required live attempt banked no new level and the registry remained at 177. Independent
outer-loop diagnosis then exposed a narrower live defect: on placement game `sp80`, the submitted
agent's `GoalSatisfactionEnergy` returned exactly `1.0` on all 771 live frontier calls, immediate
candidate guidance was also constant, and the router never changed an ordering. Carnot already
has a relational target-match representation that separates the observed `sp80` win state from
near-wins, but that representation has not been routed through the current `E3AgentPolicy` goal
path with a zero-variance fallback.

Finally, the two-axis temperature-by-penalty sampler was exact but scientifically worse than the
promoted one-axis baseline: mean ESS was `42.89` versus `63.88`, mean autocorrelation was `8.89`
versus `6.12`, and first feasible discovery took `194.4` versus `123.0` corrected transitions.
The extension and its gated Rust port are retired. The promoted one-axis temperature-exchange
sampler remains exact and quality-positive, but it still lacks the Rust/PyO3 boundary required by
the PRD's dual-language, hardware-portable energy core.

`.510` follows those terminal facts. It generates a new immutable exact-constraint canary with a
mandated SOTA local GGUF through the supported Python llama.cpp CUDA API, consumes the committed
stream once in chronological order, and evaluates the FR-11 shadow controller prequentially before
allowing an isolated act-on-advice canary. It repairs the live ARC goal-state representation at
GAP-5703, validates it on exact and reproduced controls, and runs a matched live-path A/B before
the unconditional registry-rotated level attempt. It ports only the promoted one-axis sampler to
Rust and proves exact, restart, and hard-instance parity without a timing claim.

No native three-model runtime certificate, JSON-grammar endpoint, external generated-text scorer,
two-axis tempering, counterexample transition patcher, relational move-pruner, generic exploration
signal, offline ARC solve, board execution, TSU/Kona claim, or hardware speedup is reopened.

## What milestone `.509` proved

| Evidence | Terminal fact | Consequence for `.510` |
|---|---|---|
| Exp5636 transition | `.508` evidence was archived into a collision-free Exp5636–Exp5647 graph. | Allocate above all later outer-loop artifacts: Exp5706–Exp5716. |
| Exp5637 source delta | One non-duplicate source, Baba in Wonderland, sharpened preservation-conflict controls. | Keep one bounded execution-time freshness slot; a deduplicated no-op is success. |
| Exp5638 schema corrigendum | The immutable structured unsafe-count artifact now has a hash-bound scalar gate contract. | Preserve the source artifact and contract; do not rerun the synthetic learner. |
| Exp5639 independent FR-11 audit | `fr11_independent_promotion_ready_score=1`; exact safety, benefit, retention, poison, checkpoint, group coverage, and anytime pathwise risk all passed. | Promote the certified controller to prospective shadow evaluation. |
| Exp5640 shadow adapter | `fr11_shadow_ready_not_default_enabled`; default path equivalence held and unsafe update accepts were zero. | Keep default disabled; exercise the adapter on a new exact-label stream before any stronger canary. |
| Exp5641 executable ARC model | Exact replay and zero unsafe accepts held, but utility was insufficient and the terminal verdict was blocked/retired. | Add the missing manifest retirement and do not tune or reuse the transition patcher. |
| Exp5642 ARC model A/B | Correctly gate-skipped because Exp5641 did not promote. | Use a mechanistically different goal-representation repair. |
| Exp5643 live ARC attempt | `lf52` L8 banked no new level; registry stayed 177 and no solve credit was granted. | Exclude recent failed targets, rotate from live registry state, and preserve full duration/method receipts. |
| Exp5644 two-axis exact audit | The beta×lambda label-exchange implementation was exact. | Exactness alone is not utility; retain only as negative-method evidence. |
| Exp5645 two-axis quality | All arms found feasible states, but two-axis quality regressed on ESS, autocorrelation, validity, and discovery cost; ready score was zero. | Retire the extension and update the exclusion manifest. |
| Exp5646 two-axis Rust parity | Correctly gate-skipped. | Port the already-promoted one-axis algorithm instead. |
| Exp5647 capstone | FR-11 promoted; ARC delta zero; one-axis preserved; two-axis quality and Rust parity not promoted; broad Python/spec checks remained non-clean while focused checks passed. | Begin with retirement and ledger reconciliation; keep broad-test debt explicit rather than laundering it into experiment verdicts. |

## The three largest gaps to the PRD vision

### Gap 1 — FR-11 is certified on replay, not on a prospective current-model stream

The PRD requires continuous improvement on real workloads without corrupting prior capability.
`.509` proved causal learning, exact rollback, conditional coverage, and stopping-time safety on an
immutable synthetic stream, then proved a disabled shadow adapter. It did not show that the same
controller can handle the format failures, salient shortcuts, constraint-family shifts, and delayed
labels produced by a current local flagship GGUF. `.510` therefore creates a sealed exact-label
canary using `unsloth/gemma-4-26B-A4B-it-GGUF`, freezes its prefix/suffix hashes before learning,
and consumes it chronologically with prequential current-versus-previous-controller telemetry. The
controller may adapt only its small KAN state; model weights remain frozen and exact validators
remain final authority.

### Gap 2 — ARC's live goal energy is blind to placement and relational configurations

The live agent has search machinery and a useful candidate stack, but GAP-5703 proves that its goal
signal can be mathematically constant on a real placement episode. A constant sort key cannot guide
the frontier. The missing capability is a generic visible-state representation for progress toward
within-frame relational targets, plus an explicit zero-variance self-audit. Carnot already has
relational target-match code with exact separation evidence, but earlier relational-energy,
move-pruning, and MAP experiments did not enumerate a new winning trajectory. `.510` does not
repeat those solve claims. It asks the narrower unanswered question: does routing the relational
representation through the current submitted agent change live candidate orderings and improve
known-level efficiency without regressions? The required novel-level attempt remains separate and
unconditional.

### Gap 3 — the promoted stochastic energy core has not crossed the production-language boundary

The one-axis corrected cDLS replica-exchange sampler has exact enumerable-target evidence and
paired hard-instance mixing gains. The PRD and architecture require a Rust core with Python
bindings for deterministic deployment and future accelerator work. `.509` showed why a new method
must not be ported merely because it is exact. `.510` ports the promoted algorithm as-is, proves
energy, transition, exchange, distribution, serialization, and restart parity, then checks that
hard-instance quality survives the boundary. Timing may be collected diagnostically but cannot be
reported as a speedup.

## 2025–2026 research incorporated

The `V510 Planner Refresh - 20260714` block was appended to `research-references.md` before this
roadmap was designed.

| Source or finding | Executable use in `.510` |
|---|---|
| Understanding Why Language Models Hallucinate / TrapQA, arXiv:2607.00447 | Add exact prompt-supported rows with salient shortcut answers to the local-GGUF canary; score only independent constraint outcomes. |
| OEUVRE, AISTATS 2026 OpenReview | Record prequential loss under current and previous controller state before each label-conditioned update; do not replace anytime-valid or exact safety gates. |
| Anytime-Valid Conformal Risk Control, arXiv:2602.04364 | Preserve `.509`'s stopping-time certificate as the prospective release authority. |
| GAP-5703 live mechanism trace | Add zero-variance goal-energy fallback and route placement/spatial frames through a generic relational representation in the submitted live path. |
| Earlier GAP-4891 relational-energy ladder | Reuse only its proven goal-discrimination representation; do not repeat the failed claim that energy ordering or relational pruning alone solves the trajectory-enumeration wall. |
| Temperature-label replica exchange, Exp5633/5634 | Port the exact, quality-positive one-axis kernel without changing its scientific method. |

Semantic Scholar's newly visible EBT citation arXiv:2607.11555 concerns a learned relaxation for
neural set-function optimization and does not supersede Carnot's exact validators or sampler.
ARM-EBM citations added no stronger local dependency. OpenReview and Hugging Face repeated already
bounded verifier/memory families. GitHub supplied no sampler or KAN package that displaces the local
implementation. Extropic still exposes no authenticated Carnot-accessible TSU, and Logical
Intelligence still exposes no local Kona weights or reproducible comparator.

## Target architecture

```text
MANDATED LOCAL SOTA GGUF (weights frozen)
unsloth/gemma-4-26B-A4B-it-GGUF via llama-cpp-python CUDA
                         |
        exact FSM / arithmetic / hard-soft / TrapQA rows
                         v
        +----------------------------------------------+
        | sealed prospective canary                    |
        | raw response + exact label + family + hashes |
        | committed prefix + unopened suffix           |
        +----------------------+-----------------------+
                               |
                     chronological one-pass stream
                               v
        +----------------------------------------------+
        | .509 FR-11 shadow adapter                    |
        | current/previous prequential loss            |
        | conformal retain/adapt/reset/abstain          |
        | exact validator always wins                   |
        +----------------------+-----------------------+
                               | prospective gate
                               v
        +----------------------------------------------+
        | isolated act-on-advice canary                |
        | atomic KAN checkpoint + delayed labels       |
        | poison/retention/rollback/restart             |
        | production default remains OFF               |
        +----------------------------------------------+

ARC AGENT-VISIBLE FRAMES + OWN ACTION/OUTCOME RECEIPTS ONLY
                         |
              connected components / region pairs
                         v
        +----------------------------------------------+
        | goal-energy router                           |
        | zero-variance audit -> safe fallback         |
        | count/fraction OR relational placement energy|
        +----------------------+-----------------------+
                               | exact discrimination gate
                               v
                  current E3AgentPolicy live A/B
                               | advisory promotion only
current baseline + registry precheck ------------------+--> live +1 attempt
                                                            |
                                                    reproduce -> registry

EXP5633/5634 PROMOTED ONE-AXIS TEMPERATURE EXCHANGE
                         |
                         v
        +----------------------------------------------+
        | Rust corrected cDLS + temperature labels     |
        | minimal PyO3 binding + Python fallback       |
        | exact distribution / swap / restart parity   |
        +----------------------+-----------------------+
                               | parity gate
                               v
                 matched hard-instance quality replay
                 (portability only; no speedup claim)
```

## Phase 0 — terminal continuity and source freshness (Exp5706–Exp5707)

**Exp5706 — `.509` to `.510` transition and retirement closure.** Archive all `.509` artifacts,
preserve the FR-11 promotion and one-axis sampler, record the ARC null, add the missing exclusion
entries for the Exp5641 counterexample model and Exp5645 two-axis extension, snapshot current
outer-loop Exp5700–Exp5705 evidence, allocate Exp5706–Exp5716, and emit a collision-free dependency
map. This is infrastructure slot one.

**Exp5707 — execution-time V510 source delta.** Search every mandated primary and secondary source
after the V510 planner marker, deduplicate against the full reference ledger and exclusion manifest,
and map only new local exact hooks. A clean no-op is terminal success. This is infrastructure slot
two and the SOTA-ingestion slot.

## Phase 1 — prospective exact-stream continuous self-learning (Exp5708–Exp5710)

**Exp5708 — local-SOTA exact-constraint canary.** Use
`unsloth/gemma-4-26B-A4B-it-GGUF` through `llama-cpp-python` with authenticated CUDA offload and no
grammar endpoint. Freeze a preregistered panel of finite-state, arithmetic, hard/soft constraint,
and TrapQA-style shortcut rows. Store raw responses and independent exact-validator labels, count
missing or malformed rows as failures, and seal a chronological shadow prefix plus unopened canary
suffix. This is a runtime-and-data receipt, not a verifier-quality claim.

**Exp5709 — gated prospective FR-11 shadow stream.** If the canary has sufficient valid rows and
zero validator disagreement, consume the committed prefix exactly once in chronological order.
Before each delayed label reveal, record the current and previous KAN controller decisions and
losses; then apply only shadow adaptation allowed by the `.509` anytime-valid contract. Compare with
frozen, last-window, no-memory, and corrupted-order controls. Promotion requires pathwise risk,
group coverage, positive paired benefit, exact safety, retention, and checkpoint/restart gates.

**Exp5710 — gated isolated act-on-advice canary.** Open only the precommitted suffix. In an isolated
copy of the real verify/repair path, allow the controller's retain/adapt/reset/abstain advice to alter
the small KAN state while model weights and the production default remain frozen. Run delayed-label,
poison, contradiction, crash/restart, regression, rollback, and old-family retention controls. This
is the milestone's continuous self-learning experiment. It can establish canary readiness, never
automatic production enablement.

## Phase 2 — live relational goal energy and ARC level attempt (Exp5711–Exp5713)

**Exp5711 — placement/spatial goal-energy live-path qualification.** Add an explicit variance audit
to live goal scoring and fail safely to the existing no-goal-bias order when scores are constant.
Route only generically detected placement/spatial frames through the existing relational
target-match representation; target values must be read from the current frame or agent-owned
runtime receipts, never hard-coded by game. Prove exact separation on synthetic positive/negative
controls and already reproduced levels, demonstrate reachability from `E3AgentPolicy`, and reject
corrupted masks, translated distractors, unsupported classes, and per-game leakage. This claims no
new solve and uses `solve_provenance=development_proxy`.

**Exp5712 — gated known-level live-path A/B.** If Exp5711 is exact, nondegenerate, and leak-free,
compare the current full candidate stack with the same stack plus the relational route on a
preregistered set of reproduced placement/spatial levels under identical seeds and environment
action budgets. Measure candidate-order changes, level retention, actions per reproduced level,
frontier expansions, invalid actions, and regression intervals. Promotion requires a known-level
efficiency or retained-level benefit with an interval excluding zero, zero level regressions, and
no unsafe route accepts. This does not claim that goal energy solves the known trajectory-generation
wall.

**Exp5713 — unconditional registry-rotated live-agent `+1` attempt.** Registry-precheck at execution,
exclude every reproduced level and recent failed target, then freeze one eligible target before
interaction. Use Exp5712 only if it promoted and the target-local relational hypothesis is learned
from the agent's own runtime observations; otherwise use the unchanged baseline. The task is never
structured-gated. Only live-agent self-discovery followed by independent generic reproduction and a
true registry update counts.

## Phase 3 — one-axis Rust portability and reconciliation (Exp5714–Exp5716)

**Exp5714 — one-axis Rust/Python exact parity.** Verify the immutable Exp5633/5634 hashes, add or
extend an OpenSpec requirement, then implement the smallest Rust corrected-cDLS plus
temperature-label exchange core and PyO3 binding consistent with the workspace. Prove deterministic
energy, proposal, acceptance, swap, target-replica, enumerable-distribution, error, serialization,
and Python-fallback parity. Broken swap and stale-label controls must fail. No timing or hardware
claim is admissible.

**Exp5715 — gated hard-instance quality and restart parity.** If exact parity passes, compare Rust
and Python one-axis exchange on the original preregistered hard-instance families and paired seeds
with matched corrected-kernel transitions. Require quality preservation for validity, energy, ESS,
autocorrelation, barrier crossings, and solve probability; checkpoint mid-run, cross the language
boundary, resume, and compare deterministic suffixes and distributions. This is portability and
operational-state evidence only, not a speed benchmark.

**Exp5716 — `.510` capstone reconciliation.** Aggregate every artifact, enforce structured gates and
retirement rules, reconcile OpenSpec, traceability, completion, references, status, changelog,
known issues, exclusions, ARC registry, and conductor evidence, then run roadmap schema/gate lints,
focused and applicable broad tests, spec coverage, adversarial verification, root-clutter sweep, and
the applicable E2E plan. Blocked, skipped, flagged, proxy, unauthenticated-offload, or unaudited
evidence cannot promote.

## Dependency graph

```text
Exp5706 transition + retirement closure --------------------------+
Exp5707 source delta ---------------------------------------------+----> Exp5716 capstone

Exp5708 sealed SOTA exact canary
    └──[canary ready + validator disagreement=0]──> Exp5709 prospective shadow
                                                       └──[shadow ready + unsafe=0]──> Exp5710 isolated canary

GAP-5703 + existing relational representation
    └──> Exp5711 exact/live-path goal-energy qualification
             └──[exact + nondegenerate + leak-free]──> Exp5712 known-level live A/B
                                                           └── advisory only ─┐
current live baseline + registry precheck -----------------------------------> Exp5713 +1 attempt

Exp5633/5634 promoted one-axis exchange
    └──> Exp5714 Rust/Python exact parity
             └──[parity ready + broken controls rejected]──> Exp5715 quality/restart parity

Exp5706–Exp5715 --------------------------------------------------> Exp5716 reconciliation
```

Exp5713 is intentionally not structured-gated because the ARC standing floor requires a genuine
live attempt even if the new mechanism fails. Exp5710 and Exp5715 are gated because stronger canary
or portability claims are meaningful only after their exact prerequisites pass.

## Hardware and model requirements

| Resource | Tasks | Requirement and claim boundary |
|---|---|---|
| Dual RTX 3090 GPUs | Exp5708 only | At least one positive CUDA-offload receipt through `llama-cpp-python`; record model path/hash, quantization, device, offloaded layers, peak memory, runtime, and raw response hashes. CPU-only fallback is blocked, not headline evidence. |
| Mandated local GGUF | Exp5708 | `MODEL_SPECS` must include `unsloth/gemma-4-26B-A4B-it-GGUF`. A cached second mandated model may be diagnostic, but no task depends on three-model availability. Legacy 0.8B/E4B models are smoke-only and cannot satisfy the canary. |
| CPU and system RAM | Exp5706–Exp5716 | Exact validation, KAN adaptation, ARC offline arcade, enumerable sampler checks, compilation, and reconciliation. Record CPU identity, memory, and wall time where relevant. |
| NVMe | Exp5708–Exp5710, Exp5714–Exp5715 | Immutable GGUF, sealed canary rows, KAN ledgers/checkpoints, Rust build outputs, sampler checkpoints, and content hashes. |
| Rust/PyO3 toolchain | Exp5714–Exp5715 | Use the existing workspace, binding, feature, and fallback conventions. Toolchain and ABI identities are required artifacts. |
| KV260 / GateMate / PolarFire | None | No board is needed to answer this milestone's scientific questions. Rust portability is a software-boundary receipt and cannot be presented as board readiness or speedup. |
| Extropic TSU / Kona | None | No authenticated local access exists. Public claims are context only. |

## Promotion and retirement rules

1. Exp5708 blocks if no mandated model is locally resolved, CUDA offload is unauthenticated, raw
   response provenance is incomplete, exact validators disagree, or sealed split hashes are absent.
   The native three-model/JSON-grammar scope remains retired regardless.
2. Exp5709 promotes prospective shadow evidence only if all decisions precede labels, the stream is
   consumed in committed order, exact unsafe accepts are zero, pathwise risk and worst-group coverage
   pass, paired benefit is positive, and restart/retention controls pass.
3. Exp5710 can promote only an isolated canary. Production remains disabled by default; any unsafe
   update, poison acceptance, unrecovered regression, checkpoint mismatch, or model-weight mutation
   blocks and retires this stronger canary scope.
4. Exp5711 is a representation qualification, not a solve. It must reject per-game constants and
   fail safely on zero variance. If the relational route cannot change scores/orderings on exact
   positive controls, retire the live route; do not tune the old pruner or patcher.
5. Exp5712 promotes only with matched-budget known-level benefit, zero level regressions, live
   reachability, and zero unsafe route accepts. A null does not block Exp5713.
6. Exp5713 receives solve credit only for `live_agent_self_discovery`, independent generic
   reproduction, and a true registry delta. Development-proxy or outer-loop evidence is never
   headline solve evidence.
7. Exp5714/5715 may port only the one-axis algorithm fixed by Exp5633/5634. Any semantic, exact-target,
   quality, serialization, or restart mismatch blocks portability. Timing and hardware speedup stay
   false even if diagnostic durations differ.
8. Exp5706 must apply the missing Exp5641 and Exp5645 manifest retirements before downstream scope
   claims. Exp5716 must mechanically retire any repeated same-verdict scope declared in YAML.
9. Every terminal artifact uses an `honest_verdict` beginning with `complete:` or `blocked:`.
   Gate-skipped artifacts remain blocked and cannot be promoted by the capstone.

## Expected outputs

- `results/experiment_5706_transition_v510.json`
- `results/experiment_5707_v510_source_delta_ingestion.json`
- `results/experiment_5708_sota_exact_constraint_canary.json`
- `results/experiment_5709_fr11_prospective_shadow_stream.json`
- `results/experiment_5710_fr11_isolated_act_on_advice_canary.json`
- `results/experiment_5711_arc_relational_goal_energy_live_qualification.json`
- `results/experiment_5712_arc_relational_goal_energy_live_ab.json`
- `results/experiment_5713_arc_live_self_discovery_levelup_v510.json`
- `results/experiment_5714_one_axis_tempering_rust_parity.json`
- `results/experiment_5715_one_axis_tempering_rust_quality_restart.json`
- `results/experiment_5716_v510_capstone_reconciliation.json`

Implementation tasks must add or update the relevant `openspec/capabilities/*/spec.md` REQ-* anchors
before source changes. The capstone reconciles `research-complete.yaml`, `_bmad/traceability.md`,
`ops/status.md`, `ops/changelog.md`, `ops/known-issues.md`, `ops/exclusion_manifest.yaml`, and the ARC
registry with observed evidence. It must not modify `research-roadmap.yaml` or
`scripts/research_conductor.py`.
