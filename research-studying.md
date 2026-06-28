# Research Studying — Ranked Ideas for Future Experiments

**Purpose:** Claude (outer loop) continuously researches novel ideas from
online sources, ranks them by potential impact on Carnot's current state,
and queues the most promising into the next roadmap milestone. Codex (inner
loop) executes the current experiments.

**Updated:** 2026-06-20 (ARC-AGI-3 submission-sprint pivot; discovery clusters re-pointed).
**Current Focus:** ARC-AGI-3 Kaggle submission sprint through the 2026-06-30 Milestone #1 deadline (CLAUDE.md "ARC-AGI-3 Submission Sprint Forcing Function"). The headline is **action efficiency** — the leaderboard scores `(human_actions/agent_actions)^2`, so the live wall is "explore with FEWER actions," not solve-rate or config tuning. Active research directions: affordance / clickability / frame-change prediction (the StochasticGoose lever), neural-guided / value-guided search to prune the explorer, imitation prior from the 342 human replays, and goal/world-model induction (Family-A/Family-B winning architectures). The verifier-moat / FoVer / paper-v6 program is now SUPPORTING, not the headline. Discovery machinery re-pointed 2026-06-20: `scripts/sweep_clusters.py` clusters 5 (ARC interactive-agent exploration/affordance) + 6 (neural-guided search / goal-world-model induction) added alongside the pre-pivot clusters 0-4. NOTE: the per-milestone SOTA-ingestion slot (the actual ingestion path) has been ARC-aligned since the pivot — `.414`/`.415`/`.416` ingested search/imitation/affordance SOTA with real arXiv IDs; this header refresh + the new clusters bring the upstream DISCOVERY side in line.

**Historical (pre-pivot, preserved per never-prune):** Phase 1 ship-track was one external reproducer away. Paper-v6 narrowed per the 2026-05-23 Deep Think round; two retractions + one rescue + five-post operations/honesty blog series shipped. Sweep infrastructure recovered 2026-05-24 after 8 days degraded.

<!-- EXP4890-SOTA-INGESTION-V451-FRONTIER-START -->
## 2026-06-27 Exp 4890 - .451 V451 frontier SOTA ingestion - INGESTED

**Status:** INGESTED into `results/experiment_4890_sota_ingestion_v451_frontier.json`.

**Preconditions:** `research-studying.md`, `research-references.md`,
`results/experiment_4882_ttt_dynamics_value_gap.json`,
`results/experiment_4883_inducer_ceiling_ab.json`, and
`results/experiment_4879_sota_ingestion_v450_frontier.json` were present. A1's
actual fork verdict is `INDUCER_CEILING_HARD`; A1b's inducer-ceiling attribution
is `METHOD_IS_CEILING`. `scripts/sweep_clusters.py` emitted the
neural-guided-search/world-model cluster 6 URLs. `scripts/sweep_semscholar.py`
was run on five focused queries; HTTP 429 rate limits were recorded rather than
promoted as evidence. Low-concurrency WebSearch/WebFetch plus direct arXiv
HTTP checks verified the top six papers listed below. `/deep-research` was not
invoked. The retired energy-as-ARC-lever, coverage/vocabulary,
exploration-strategy, selection/ranking, and perception-from-grid classes were
not re-ingested. No model load, training, leaderboard submission, or solve claim
was made; this is a no solve claim ingestion note.

**A1/A1b branch:** `INDUCER_CEILING_HARD` with `METHOD_IS_CEILING`, so `.451`
targets alternative world-model representations beyond executable code.

**Verified source set:**
- arXiv:2503.18938 -- AdaWorld: Learning Adaptable World Models with Latent Actions
- arXiv:2505.08073 -- Explainable Reinforcement Learning Agents Using World Models
- arXiv:2602.23997 -- Foundation World Models for Agents that Learn, Verify, and Adapt Reliably Beyond Static Environments
- arXiv:2603.19312 -- LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels
- arXiv:2606.25421 -- Beyond Next-Observation Prediction: Agent-Authored World Modeling for Sequential Decision Making
- arXiv:2606.26217 -- Fast LeWorldModel

**SOTA -> .451 frontier mapping:**
- **Agent-authored decision-need world-model targets** (arXiv:2606.25421): maps to .451 / INDUCER_CEILING_HARD + METHOD_IS_CEILING. A1b fit: A1b says the current executable-code induction method is the ceiling; decision-need targets replace generic next-observation supervision. Evidence: arXiv:2606.25421 proposes Agent-Authored World Modeling, where the policy identifies decision-relevant dynamics before acting. Experiment graft: For each A1 held-out engine miss, generate a decision-need target such as hidden register state, object persistence, or action effect, then train or prompt a non-code world-model target table before engine loading. Validation gate: Promote only if the decision-target representation raises held-out changed-cell value accuracy on the same A1 games without using the banked answer as supervision. Fails when: The authored targets mirror the current model misconception, require unobserved facts, or improve next-frame text while held-out dynamics stay flat.
- **Action-prefix latent transition adapter** (arXiv:2606.26217, arXiv:2603.19312): maps to .451 / INDUCER_CEILING_HARD + METHOD_IS_CEILING. A1b fit: A1b rules out another same-method executable inducer pass; prefix latents model multi-step action effects without rolling one-step code. Evidence: arXiv:2606.26217 replaces repeated one-step latent rollout with action-prefix prediction; arXiv:2603.19312 supplies the compact LeWorldModel latent substrate. Experiment graft: Encode candidate action prefixes into latent future-state deltas and score A1 held-out transitions through the latent adapter before converting only accepted deltas into engine facts. Validation gate: Count the graft only when long-horizon held-out transition accuracy improves and one-step observed-prefix replay does not regress. Fails when: Prefix supervision hides wrong mechanics, latent states cannot be decoded into verifier-checkable facts, or compounding error remains flat.
- **Latent-action adaptable world-model interface** (arXiv:2503.18938): maps to .451 / INDUCER_CEILING_HARD + METHOD_IS_CEILING. A1b fit: A1b method-ceiling means the missing structure may be action semantics, not code synthesis strength; latent actions give a non-code action layer. Evidence: arXiv:2503.18938 extracts self-supervised latent actions and conditions an adaptable world model on them for transfer with limited interactions. Experiment graft: Infer latent action tokens from cold-start ARC transitions, align E3 discrete controls to those tokens, and feed the latent action state into the held-out transition scorer. Validation gate: Promote only if the latent-action adapter predicts off-prefix action effects better than the current executable engine on the A1 split. Fails when: The latent actions do not align with legal controls, collapse across different mechanics, or require more adaptation interactions than ARC permits.
- **Reverse counterfactual world-model targeter** (arXiv:2505.08073, arXiv:2606.25421): maps to .451 / INDUCER_CEILING_HARD + METHOD_IS_CEILING. A1b fit: A1b method-ceiling suggests the engine needs a targetable state representation; reverse world models ask what state fact would make a desired action rational. Evidence: arXiv:2505.08073 augments model-based RL with a reverse world model for counterfactual state targets; arXiv:2606.25421 supplies the decision-need target construction. Experiment graft: For each failed A1 transition, ask a reverse model for the missing state fact that would make the predicted action effect valid, then turn that fact into a targeted induction or probe row. Validation gate: Accept only reverse targets that reduce held-out dynamics errors while remaining oracle-distinct from level completion. Fails when: The counterfactual state is unreachable, leaks the terminal answer, or explains the policy without improving transition prediction.
- **Verification-calibrated abstraction substrate** (arXiv:2602.23997, arXiv:2603.19312): maps to .451 / INDUCER_CEILING_HARD + METHOD_IS_CEILING. A1b fit: A1b method-ceiling points away from another executable-code attempt; a calibrated abstraction layer makes representation reliability explicit. Evidence: arXiv:2602.23997 argues for world models with online abstraction calibration and verification hooks; arXiv:2603.19312 demonstrates compact latent world-model structure with physical probes. Experiment graft: Insert a persistent latent abstraction state beside the executable engine and require each abstract fact to carry a verifier-calibrated confidence before it affects planning. Validation gate: Promote only if abstraction confidence predicts held-out engine mismatches and the calibrated facts improve A1 value accuracy. Fails when: The abstraction is too coarse for ARC mechanics, calibration only tracks seen prefixes, or verifier hooks become a selection/ranking rerun.

flagged_for_v451: agent_authored_decision_need_targets (arXiv:2606.25421)
flagged_for_v451: action_prefix_latent_adapter (arXiv:2606.26217 + arXiv:2603.19312)
flagged_for_v451: latent_action_world_model_adapter (arXiv:2503.18938)

**Bottom line for .451:** start with agent-authored decision-need targets, then
action-prefix latent adapters and latent-action world-model interfaces. Treat
reverse/counterfactual targets and verification-calibrated abstractions as
secondary representation experiments; do not re-run the retired classes.
<!-- EXP4890-SOTA-INGESTION-V451-FRONTIER-END -->

<!-- EXP4879-SOTA-INGESTION-V450-FRONTIER-START -->
## 2026-06-27 Exp 4879 - .450 V450 frontier SOTA ingestion - INGESTED

**Status:** INGESTED into `results/experiment_4879_sota_ingestion_v450_frontier.json`.

**Preconditions:** `research-studying.md`, `research-references.md`,
`results/experiment_4871_generation_wall_fork_probe_gpu_fixed.json`,
`results/experiment_4872_cegis_world_model_refinement.json`, and
`results/experiment_4868_sota_ingestion_v449_frontier.json` were present. A1's
source `fork_verdict` is null because the positive-control check failed, but the
numeric fork table computes to `INDUCER_CEILING`. A1b CEGIS delta was 0.0 with
CI95 [0.0, 0.0], so this note carries forward the next inducer candidates, not
the current CEGIS refinement loop. `scripts/sweep_clusters.py` emitted the
neural-guided-search/world-model cluster 6 URLs. `scripts/sweep_semscholar.py`
was run on five focused queries; rate limits were recorded rather than promoted
as evidence. Low-concurrency WebSearch/WebFetch plus direct arXiv HTTP checks
verified the top eight papers listed below. `/deep-research` was not invoked.
The retired macro-vocab/click-heatmap coverage, exploration-strategy, and
energy classes were not re-ingested. No model load, training, leaderboard
submission, or solve claim was made; this is a no solve claim ingestion note.

**A1/A1b branch:** `INDUCER_CEILING` residual with an A1 positive-control caveat
and a nulled A1b CEGIS refinement delta.

**Verified source set:**
- arXiv:2203.13474 -- CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis
- arXiv:2506.02918 -- World Modelling Improves Language Model Agents
- arXiv:2507.03160 -- Assessing Small Language Models for Code Generation: An Empirical Study with Benchmarks
- arXiv:2507.15877 -- Out-of-Distribution Generalization in the ARC-AGI Domain: Comparing Execution-Guided Neural Program Synthesis and Test-Time Fine-Tuning
- arXiv:2509.03956 -- World Model Implanting for Test-time Adaptation of Embodied Agents
- arXiv:2605.05138 -- Executable World Models for ARC-AGI-3 in the Era of Coding Agents
- arXiv:2606.25421 -- Beyond Next-Observation Prediction: Agent-Authored World Modeling for Sequential Decision Making
- arXiv:2606.26217 -- Fast LeWorldModel

**SOTA -> .450 frontier mapping:**
- **Test-time world-model and dynamics adaptation loop** (arXiv:2506.02918, arXiv:2509.03956, arXiv:2507.15877): maps to .450 / INDUCER_CEILING. A1b fit: A1b CEGIS delta was 0.0, so the next swing adapts or retrieves dynamics at test time before planning through the engine. Evidence: arXiv:2506.02918 trains internal state prediction for language agents; arXiv:2509.03956 composes world models at test time; arXiv:2507.15877 compares ARC execution-guided synthesis with test-time fine-tuning. Experiment graft: Collect cold-start transitions, fit or retrieve a compact dynamics adapter, then remeasure held-out transition accuracy before any planner reranking. Validation gate: Promote only if held-out off-prefix transition accuracy improves on games disjoint from the adapter's observed-prefix fit. Sovereignty: The adapter can be selected or trained locally from game observations, preserving the air-gapped path. Fails when: The adapter memorizes prefix frames, loses hidden state, or improves observed replay while held-out dynamics remain flat.
- **Family-B reference versus local open-code inducer A/B** (arXiv:2605.05138, arXiv:2507.03160, arXiv:2203.13474): maps to .450 / INDUCER_CEILING. A1b fit: A1b's null CEGIS result means the loop needs a stronger inducer measurement, not another repair pass from the same engine. Evidence: arXiv:2605.05138 supplies the executable-world-model coding-agent reference; arXiv:2507.03160 evaluates small code models; arXiv:2203.13474 establishes open multi-turn code synthesis. Experiment graft: Run one Family-B reference lane and one local open-code lane against the same engine interface and held-out transition gate. Validation gate: The reference lane measures the capability ceiling; the local lane is promoted only if it beats the current Qwen3.5-9B-MTP inducer under the A1 held-out game set. Sovereignty: The cloud-strength lane is a ceiling measurement; the desired deployment lane remains local and open. Fails when: The reference lane still overfits observed prefixes, or no local open inducer can synthesize executable state updates.
- **Agent-authored world-model target construction** (arXiv:2606.25421, arXiv:2506.02918): maps to .450 / INDUCER_CEILING. A1b fit: A1b failed to repair from generic counterexamples; decision-oriented targets ask the agent what transition facts it needs before acting. Evidence: arXiv:2606.25421 replaces next-observation prediction with agent-authored dynamics targets; arXiv:2506.02918 shows state prediction can support language-agent tool planning. Experiment graft: For each failed held-out transition, generate a decision-need target such as hidden toggle state, object persistence, or action effect, then train or prompt the inducer against that target. Validation gate: Count the method only when targeted transition facts raise held-out engine accuracy, not just when next-frame text improves. Sovereignty: Target construction is derived from local traces and can feed either the local inducer or the reference lane. Fails when: The generated targets mirror the model's misconception or require observations the game has not exposed.
- **Action-prefix latent world-model adapter** (arXiv:2606.26217, arXiv:2506.02918, arXiv:2507.15877): maps to .450 / INDUCER_CEILING. A1b fit: A1b's flat delta leaves compounding one-step dynamics error as a candidate residual; prefix-level prediction attacks that error. Evidence: arXiv:2606.26217 predicts latents for action prefixes instead of rolling one step at a time; arXiv:2506.02918 adds state prediction to agents; arXiv:2507.15877 keeps ARC execution guidance in scope. Experiment graft: Add an action-prefix probe over candidate sequences and compare its held-out transition predictions against the current one-step engine. Validation gate: Promote only if long-horizon held-out transition accuracy improves without degrading one-step observed-prefix replay. Sovereignty: A small prefix adapter can run locally and can be swapped behind the same executable-engine interface. Fails when: Prefix supervision hides wrong mechanics, or the latent state cannot be decoded into executable game-state checks.

flagged_for_v450: test_time_dynamics_adaptation_loop (arXiv:2506.02918 + arXiv:2509.03956 + arXiv:2507.15877)
flagged_for_v450: family_b_vs_local_open_code_inducer_ab (arXiv:2605.05138 + arXiv:2507.03160 + arXiv:2203.13474)
flagged_for_v450: agent_authored_world_model_targets (arXiv:2606.25421 + arXiv:2506.02918)

**Bottom line for .450:** try test-time dynamics adaptation first, then compare
a Family-B executable-world-model reference inducer with a local open-code
inducer. Use agent-authored targets and action-prefix adapters as targeted
engine-quality improvements; keep the current CEGIS loop recorded as nulled.
<!-- EXP4879-SOTA-INGESTION-V450-FRONTIER-END -->

<!-- EXP4868-SOTA-INGESTION-V449-FRONTIER-START -->
## 2026-06-27 Exp 4868 - .449 V449 frontier SOTA ingestion - INGESTED

**Status:** INGESTED into `results/experiment_4868_sota_ingestion_v449_frontier.json`.

**Preconditions:** `research-studying.md`, `research-references.md`,
`results/experiment_4861_generation_wall_fork_probe.json`, and
`results/experiment_4858_sota_ingestion_generation_expressibility.json` were
present. The checked-in Exp 4861 A1 `fork_verdict` is blocked/null
(`honest_verdict=blocked_generator_unavailable`), so this note follows the
operator-reserved likely `INDUCER_CEILING` branch without claiming A1 measured
it. `scripts/sweep_clusters.py` emitted ARC action-effect and
neural-guided-search/world-model cluster URLs. `scripts/sweep_semscholar.py`
was run on four focused queries; HTTP 429 limited three queries, and the
test-time dynamics query returned arXiv IDs recorded in the artifact. Low-
concurrency WebSearch/WebFetch plus direct arXiv HTTP checks verified the top
eight papers listed below. `/deep-research` was not invoked. The retired
macro-vocab/click-heatmap coverage, exploration-strategy, and energy classes
were not re-ingested. No model load, training, leaderboard submission, or solve
claim was made; this is a no solve claim ingestion note.

**A1 fork targeted:** `INDUCER_CEILING`, with the caveat that the committed A1
artifact is blocked/null. The .449 handoff is to improve world-model inducer
accuracy before investing in more planner/search machinery.

**Verified source set:**
- arXiv:2203.13474 -- CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis
- arXiv:2502.07786 -- Counterexample Guided Program Repair Using Zero-Shot Learning and MaxSAT-based Fault Localization
- arXiv:2506.02918 -- World Modelling Improves Language Model Agents
- arXiv:2507.03160 -- Assessing Small Language Models for Code Generation: An Empirical Study with Benchmarks
- arXiv:2507.15877 -- Out-of-Distribution Generalization in the ARC-AGI Domain: Comparing Execution-Guided Neural Program Synthesis and Test-Time Fine-Tuning
- arXiv:2509.03956 -- World Model Implanting for Test-time Adaptation of Embodied Agents
- arXiv:2605.05138 -- Executable World Models for ARC-AGI-3 in the Era of Coding Agents
- arXiv:2606.11521 -- Counterexample Guided Learning in the Large using Reasoning Agents

**SOTA -> .449 frontier mapping:**
- **Family-B executable world-model inducer quality ladder** (arXiv:2605.05138, arXiv:2507.03160, arXiv:2203.13474): maps to .449 / INDUCER_CEILING. Fork mapping: INDUCER_CEILING means the executable model is inaccurate before planning; compare the strong Family-B coding-agent inducer against local small/open code inducers under the same held-out transition gate. Evidence: arXiv:2605.05138 reports verifier-driven executable Python world models with strong coding agents; arXiv:2507.03160 and arXiv:2203.13474 bound what local/open code models can plausibly supply. Experiment graft: Build a two-lane inducer harness: cloud-strength Family-B reference lane for ceiling measurement, local open-code lane for sovereign deployment, both emitting the same executable engine interface. Validation gate: Pass only if held-out off-path transition accuracy improves before any planner reranking; otherwise .449 retires the inducer upgrade. Sovereignty: The cloud lane is a measurement oracle for capability, not the desired deployment path; the local lane preserves air-gapped operation. Fails when: The strong inducer still overfits observed prefixes, or the local open inducer cannot approach the cloud reference without forbidden network access.
- **Test-time world-model and dynamics adaptation loop** (arXiv:2506.02918, arXiv:2509.03956, arXiv:2507.15877): maps to .449 / INDUCER_CEILING. Fork mapping: INDUCER_CEILING can be attacked by adapting the dynamics model at test time from observed transitions before planning through it. Evidence: arXiv:2506.02918 adds internal state prediction to language agents, arXiv:2509.03956 composes world models at test time, and arXiv:2507.15877 frames ARC test-time fine-tuning versus execution-guided synthesis. Experiment graft: After cold-start transition collection, fit or select a small dynamics adapter, then rerun the held-out transition score before plan_in_model. Validation gate: Only count improvements that raise held-out transition accuracy on games not used for the adapter's observed-prefix fit. Sovereignty: The adapter can be trained or selected locally from game observations, which keeps the improvement air-gapped. Fails when: The adapter memorizes prefix frames, loses hidden state, or improves in-distribution replay without raising held-out dynamics accuracy.
- **Counterexample-guided executable world-model refinement** (arXiv:2606.11521, arXiv:2502.07786, arXiv:2507.15877): maps to .449 / INDUCER_CEILING. Fork mapping: INDUCER_CEILING becomes a refinement loop: failed held-out transitions become counterexamples that revise the executable engine instead of merely rejecting it. Evidence: arXiv:2606.11521 shows counterexamples can improve LLM symbolic induction; arXiv:2502.07786 uses CEGIS-style LLM repair; arXiv:2507.15877 supports execution-guided ARC synthesis. Experiment graft: Wrap the engine verifier in a CEGIS loop that converts off-path mismatch rows into minimal failing transition tests and asks the inducer to repair the engine. Validation gate: Accept a refined engine only when the repair fixes held-out counterexamples without regressing observed-prefix replay. Sovereignty: Counterexamples are produced by the local executable verifier, so even a small local inducer receives precise feedback without cloud traces. Fails when: Counterexamples are too sparse, repairs overfit the latest failing row, or the executable representation cannot express the hidden mechanic.
- **Local open-code inducer distillation and self-correction** (arXiv:2507.03160, arXiv:2203.13474, arXiv:2502.07786): maps to .449 / INDUCER_CEILING. Fork mapping: INDUCER_CEILING requires a stronger air-gapped inducer, so the local lane should use open code-model selection plus verifier feedback rather than another generic prompt. Evidence: arXiv:2507.03160 evaluates compact open code models, arXiv:2203.13474 establishes open multi-turn program synthesis, and arXiv:2502.07786 shows verifier feedback can improve LLM repair. Experiment graft: Benchmark candidate local code models on executable-engine synthesis, then distill the successful prompting and repair traces into the chosen local inducer lane. Validation gate: Promote a local inducer only if it beats the current Qwen3.5-9B-MTP engine accuracy under the same A1 held-out-game set. Sovereignty: This is the deployment candidate: all inference and refinement stays on local hardware after the cloud reference has measured the ceiling. Fails when: The best local model cannot synthesize executable state updates, or self-correction loops repeatedly repair syntax while dynamics remain wrong.

flagged_for_v449: family_b_executable_world_model_inducer_ladder (arXiv:2605.05138 + arXiv:2507.03160 + arXiv:2203.13474)
flagged_for_v449: test_time_world_model_adaptation_loop (arXiv:2506.02918 + arXiv:2509.03956 + arXiv:2507.15877)
flagged_for_v449: cegis_world_model_refinement_loop (arXiv:2606.11521 + arXiv:2502.07786 + arXiv:2507.15877)

**Bottom line for .449:** stage the Family-B executable-world-model inducer as
the capability reference, add test-time dynamics adaptation, and wrap the
induced engine in counterexample-guided refinement. Promote the local open
inducer only when held-out transition accuracy improves under the same A1 gate.
<!-- EXP4868-SOTA-INGESTION-V449-FRONTIER-END -->

<!-- EXP4858-SOTA-INGESTION-GENERATION-EXPRESSIBILITY-START -->
## 2026-06-27 Exp 4858 - .448 generation expressibility SOTA ingestion - INGESTED

**Status:** INGESTED into `results/experiment_4858_sota_ingestion_generation_expressibility.json`.

**Preconditions:** `research-studying.md`, `research-references.md`,
`results/experiment_4848_sota_ingestion_object_world_model.json`, and
`results/experiment_4851_generation_coverage_diagnostic.json` were present.
Exp 4851's dominant bucket was `NEVER_ENUMERATED`, so the target is generation
expressibility, not ranking. `scripts/sweep_clusters.py` emitted ARC
action-effect/exploration and neural-guided-search/world-model cluster URLs.
`scripts/sweep_semscholar.py` was run on three focused generation-expressibility
queries and returned HTTP 429 for all three, so no S2-only source was promoted.
Low-concurrency WebSearch/WebFetch plus direct arXiv HTTP checks verified the
top eight papers listed below. `/deep-research` was not invoked. The nulled
exploration-strategy class and concluded energy stages were not re-ingested. No
model load, training, leaderboard submission, or solve claim was made; this is a
no solve claim ingestion note.

**A1 bucket targeted:** `NEVER_ENUMERATED` means the current proposer did not
express at least one action primitive in most banked winning prefixes. A ranker
cannot fix a missing candidate. The .448 handoff is to put the winning prefix
into the pool by widening the proposer vocabulary.

**Partial/noisy object signal contract:** carry forward only partial/noisy object
signal from Exp 4848. Object slots, relation hints, and action bindings can guide
proposal generation, but exact identity is not a trusted precondition.

**Verified source set:**
- arXiv:2006.08381 -- DreamCoder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning
- arXiv:2310.19791 -- LILO: Learning Interpretable Libraries by Compressing and Documenting Code
- arXiv:2411.17708 -- Towards Efficient Neurally-Guided Program Induction for ARC-AGI
- arXiv:2507.14172 -- SOAR: Self-Improving Language Models for Evolutionary Program Synthesis: A Case Study on ARC-AGI
- arXiv:2507.15877 -- Out-of-Distribution Generalization in the ARC-AGI Domain: Comparing Execution-Guided Neural Program Synthesis and Test-Time Fine-Tuning
- arXiv:2601.06604 -- Object-Centric World Models Meet Monte Carlo Tree Search
- arXiv:2605.05138 -- Executable World Models for ARC-AGI-3 in the Era of Coding Agents
- arXiv:2606.14418 -- Causal Object-Centric Models for Planning with Monte Carlo Tree Search

**SOTA -> generation expressibility mapping for .448:**
- **DreamCoder/LILO action-primitive library learner** (arXiv:2006.08381, arXiv:2310.19791): maps to .448 / NEVER_ENUMERATED; Primitive expansion: Learns reusable action primitives and documented abstractions from verified traces, then offers those primitives as first-class proposer moves. Winner insertion: Insert the winning prefix into the pool by synthesizing a short program over learned move/transform/loop primitives, then lowering it to actions. Object signal: Consumes partial/noisy object signal as soft predicates and confidence-tagged slots; stable object IDs are useful hints, not requirements. Proposal graft: Mine successful and near-successful traces for reusable ARC action macros, add them to the proposer vocabulary, and enumerate macro programs before raw coordinate retries. Verification handoff: Lower each macro program to executable actions and keep it only when replay matches the verifier's prefix/state checks. Takes over: Takes over the fixed primitive vocabulary that made decisive multi-step prefixes absent from the proposal pool. Fails when: The trace corpus is too thin to learn useful macros, abstractions overfit one family, or the learned macro cannot be lowered into legal live actions.
- **Neurally guided ARC DSL program search** (arXiv:2411.17708, arXiv:2507.14172): maps to .448 / NEVER_ENUMERATED; Primitive expansion: Uses a learned guide and evolutionary self-improvement to prioritize DSL primitive operators and compositions that the generic action proposer never names. Winner insertion: Insert the winning prefix into the pool by searching ARC DSL programs, compiling the best program into candidate action prefixes, and replaying them. Object signal: Consumes partial/noisy object signal as optional features for DSL argument binding while preserving fallback grid predicates. Proposal graft: Add a program-search proposer arm that samples guided DSL transformations for object, color, region, and loop operations before live ranking. Verification handoff: Compile candidate programs to action prefixes and verify by executable replay against the offline/live harness. Takes over: Takes over flat action enumeration when the missing prefix is a composed grid transformation rather than a single primitive action. Fails when: The DSL excludes the true mechanic, the neural guide memorizes training families, or evolutionary search spends the action budget on invalid programs.
- **Execution-guided neural program synthesis for ARC** (arXiv:2507.15877, arXiv:2507.14172): maps to .448 / NEVER_ENUMERATED; Primitive expansion: Expands primitives by letting neural proposals mutate executable programs under feedback from failed replays and counterexamples. Winner insertion: Insert the winning prefix into the pool by turning replay errors into program edits until the synthesized action program reaches the missing prefix state. Object signal: Consumes partial/noisy object signal only as counterexample annotations and candidate feature bindings; execution remains the source of truth. Proposal graft: Run a bounded synthesize-execute-repair loop over action programs, enqueueing only repaired prefixes that pass executable checks. Verification handoff: Every repaired program is replayed; rejected traces become the next synthesis counterexample rather than a scored candidate. Takes over: Takes over one-shot proposal failures by converting verifier rejects into new candidate programs. Fails when: Counterexamples are too expensive to gather, the repair model edits surface syntax but not mechanics, or the executable check cannot expose the missing rule.
- **Object-relational MCTS action-program proposer** (arXiv:2601.06604, arXiv:2606.14418): maps to .448 / NEVER_ENUMERATED; Primitive expansion: Expands primitive actions into object-bound action programs selected by MCTS over learned object-relational dynamics. Winner insertion: Insert the winning prefix into the pool by expanding MCTS branches whose object-relation deltas predict the decisive first-contact transition. Object signal: Consumes partial/noisy object signal as probabilistic slots and relation hypotheses, allowing identity uncertainty inside the tree state. Proposal graft: Add a shallow object-relation MCTS proposer that emits replayable object-bound branches as candidate prefixes. Verification handoff: Replay branch actions and compare observed relation deltas against the MCTS prediction before admitting the prefix. Takes over: Takes over unguided object interaction enumeration when the current pool never tries the decisive object-action binding. Fails when: Noisy slots alias multiple controllable objects, learned dynamics rewards a latent shortcut, or tree search cannot ground object choices to live controls.
- **Executable world-model action programmer** (arXiv:2605.05138, arXiv:2507.15877): maps to .448 / NEVER_ENUMERATED; Primitive expansion: Turns induced executable transition models into primitive action-program templates that become new proposer primitives for interactive ARC levels. Winner insertion: Insert the winning prefix into the pool by inducing a transition program, planning through it, and exporting the realized action sequence as a prefix. Object signal: Consumes partial/noisy object signal as typed hints for transition variables; the executable model must still pass replay without trusting exact IDs. Proposal graft: Let coding-agent world-model induction produce executable transition functions and ask the proposer to enumerate action programs that satisfy those functions. Verification handoff: Accept an action program only after replay confirms the induced transition and the prefix remains legal under the live harness. Takes over: Takes over static frame-delta heuristics by creating new executable action programs before ranker selection. Fails when: The induced model overfits observed prefixes, hidden mechanics need more probes, or the planner finds a model path that cannot be executed in the real game.

flagged_for_v448: dreamcoder_lilo_action_library (arXiv:2006.08381 + arXiv:2310.19791)
flagged_for_v448: eg_nps_soar_arc_program_search (arXiv:2411.17708 + arXiv:2507.14172 + arXiv:2507.15877)
flagged_for_v448: comet_executable_world_model_mcts (arXiv:2606.14418 + arXiv:2601.06604 + arXiv:2605.05138)

**Bottom line for .448:** start with DreamCoder/LILO-style action-library
learning as the vocabulary widening layer, pair it with execution-guided ARC
program search for counterexample repair, and use COMET/executable-world-model
MCTS to turn partial object signals into replayable action prefixes.
<!-- EXP4858-SOTA-INGESTION-GENERATION-EXPRESSIBILITY-END -->

<!-- EXP4848-SOTA-INGESTION-OBJECT-WORLD-MODEL-START -->
## 2026-06-27 Exp 4848 - .447 object-world-model planning SOTA ingestion - INGESTED

**Status:** INGESTED into `results/experiment_4848_sota_ingestion_object_world_model.json`.

**Preconditions:** `research-studying.md`, `research-references.md`, and
`results/experiment_4838_sota_ingestion_perception_representation.json` were
present. `scripts/sweep_clusters.py` emitted world-model, affordance/action-effect,
and neural-guided world-model cluster URLs. `scripts/sweep_semscholar.py` was
run on three focused object-world-model planning queries; one query returned
eight arXiv IDs and two returned HTTP 429, so no S2-only source was promoted.
Low-concurrency WebSearch/WebFetch plus direct arXiv HTTP checks verified the
top eight papers listed below. `/deep-research` was not invoked. The nulled
exploration-strategy class was not re-ingested. No model load, training,
leaderboard submission, or solve claim was made; this is a no solve claim
ingestion note.

**A1 object layer imported from Exp 4838:** the .447 question is not more
perception and not more generic exploration. Given object IDs, slots, relation
edges, persistence tracks, object/action bindings, and causal shortcut guards,
the planner must turn object-relational state into a proposable winner on a
novel game.

**Verified source set:**
- arXiv:1911.12247 -- Contrastive Learning of Structured World Models
- arXiv:2402.03326 -- Slot Structured World Models
- arXiv:2507.03298 -- Dyn-O: Building Structured World Models with Object-Centric Representations
- arXiv:2511.02225 -- Learning Interactive World Model for Object-Centric Reinforcement Learning
- arXiv:2601.06604 -- Object-Centric World Models Meet Monte Carlo Tree Search
- arXiv:2605.14937 -- Slot-MPC: Goal-Conditioned Model Predictive Control with Object-Centric Representations
- arXiv:2606.12316 -- Slots, Transitions, Loops: Learning Composable World Models for ARC
- arXiv:2606.14418 -- Causal Object-Centric Models for Planning with Monte Carlo Tree Search

**SOTA -> object-world-model planning mapping for .447:**
- **Object-relation transition graph proposer** (arXiv:1911.12247, arXiv:2402.03326): maps to .447; Consumes the A1 object layer as persistent object IDs, relation edges, and before/after object-action bindings. Object-relational state: State is a set of object slots plus a graph of inferred pairwise relations and action-conditioned edge changes. Planning graft: Learn a compact transition model over object-relation edits and ask the proposer to enumerate the smallest graph edits that move a near state toward a terminal-looking structure. Proposable winner output: Produce a proposable winner as a concrete object-relation edit plus the executable action template that should cause that edit. Verification handoff: Replay the action template in the live harness and keep only edits whose rendered before/after object graph matches the predicted delta. Takes over: Takes over frame-only candidate generation when the pool cannot name which object relation should change. Fails when: The A1 tracker merges objects, graph edges encode visual proximity instead of mechanics, or relation edits cannot be lowered to actions.
- **Slot-structured imagined rollout planner** (arXiv:2402.03326, arXiv:2507.03298): maps to .447; Consumes the A1 object layer as slot-aligned object features, dynamics-aware attributes, and slot persistence tracks. Object-relational state: State is an object slot table with dynamics-aware fields separated from visual nuisance fields, plus learned interaction messages. Planning graft: Roll forward short object-slot trajectories, score imagined deltas for goal-like object changes, and back out the action prefix that caused the best rollout. Proposable winner output: Produce a proposable winner as a replayable object-slot rollout and a small action prefix for the live verifier. Verification handoff: Promote only rollouts that replay into the predicted object IDs, positions, and relation deltas without relying on a terminal oracle. Takes over: Takes over static ranking by generating structured futures that were not present in the old candidate pool. Fails when: Slot identity drifts, imagined trajectories chase texture changes, or rollout error compounds before a concrete action can be verified.
- **Causal object-centric MCTS planner** (arXiv:2601.06604, arXiv:2606.14418): maps to .447; Consumes the A1 object layer as object tokens, action-slot fusions, and causal relevance scores for which objects matter to a decision. Object-relational state: State is a MuZero-style latent object tree with object-causal attention over relevant slot interactions. Planning graft: Run shallow MCTS over object-latent transitions, using causal attention to expand actions bound to task-relevant objects first. Proposable winner output: Produce a proposable winner as the best replayable MCTS branch: object-bound actions plus predicted object-state deltas. Verification handoff: Submit only the branch action prefix to the live verifier, then compare observed object deltas against the MCTS predicted state. Takes over: Takes over unguided first-contact search by deciding which object interactions to expand before spending live actions. Fails when: The latent tree cannot be grounded into executable actions, causal attention locks onto distractors, or MCTS optimizes an unobservable latent reward shortcut.
- **Goal-conditioned slot MPC action optimizer** (arXiv:2605.14937, arXiv:2507.03298): maps to .447; Consumes the A1 object layer as differentiable slot features and object-level target deltas. Object-relational state: State is a differentiable slot dynamics model with action-conditioned object updates and goal-conditioned target slots. Planning graft: Use gradient-based MPC over the object dynamics to optimize a short action sequence toward a target object configuration. Proposable winner output: Produce a proposable winner as an optimized action sequence that should realize a specific object-level goal state. Verification handoff: Replay the optimized sequence and reject it unless the observed A1 object tracks reach the planned goal-conditioned slot delta. Takes over: Takes over random coordinate retries by directly optimizing actions against object-level dynamics. Fails when: The action space is not differentiable enough, the goal slot is wrong, or the optimizer finds an invalid but smooth object shortcut.
- **Interaction-primitive loop policy for ARC** (arXiv:2511.02225, arXiv:2606.12316): maps to .447; Consumes the A1 object layer as composable object interactions, looped slot transitions, and demonstration-conditioned summaries. Object-relational state: State is a structured ARC object graph with interaction primitives and loop variables over colors, shapes, and spatial relations. Planning graft: Select a high-level interaction primitive and looped transition order, then lower it into object-bound primitive actions. Proposable winner output: Produce a proposable winner as an ARC-specific transformation sketch plus concrete object-bound actions that enter the candidate pool. Verification handoff: Run the lowered actions and retain the sketch only when the observed object graph follows the predicted looped transition. Takes over: Takes over generic exploration by constructing a new object-level candidate instead of waiting for one to appear by chance. Fails when: The primitive library misses the game mechanic, the loop summary memorizes family style, or the transformation sketch cannot lower to live actions.

flagged_for_v447: comet_object_mcts_planner (arXiv:2606.14418 + arXiv:2601.06604 + arXiv:2402.03326)
flagged_for_v447: slot_mpc_object_action_optimizer (arXiv:2605.14937 + arXiv:2507.03298)
flagged_for_v447: loop_owm_interaction_primitive_proposer (arXiv:2606.12316 + arXiv:2511.02225 + arXiv:1911.12247)

**Bottom line for .447:** prioritize COMET/ObjectZero-style object-MCTS as the
main planner, Slot-MPC as the direct object-action optimizer, and
Loop-OWM/FIOC-WM interaction primitives as the ARC-specific proposal layer.
The handoff is object-relational planning that creates a candidate winner, not
another pass over an unchanged exploration pool.
<!-- EXP4848-SOTA-INGESTION-OBJECT-WORLD-MODEL-END -->

<!-- EXP4838-SOTA-INGESTION-PERCEPTION-REPRESENTATION-START -->
## 2026-06-27 Exp 4838 - .446 perception/representation SOTA ingestion - INGESTED

**Status:** INGESTED into `results/experiment_4838_sota_ingestion_perception_representation.json`.

**Preconditions:** `research-studying.md` and `research-references.md` were
present. `scripts/sweep_clusters.py` emitted world-model, affordance/action-effect,
and neural-guided world-model cluster URLs. `scripts/sweep_semscholar.py` was
run on three focused perception/representation queries and returned HTTP 429
for all of them, so no S2-only source was promoted. Low-concurrency
WebSearch/WebFetch plus direct arXiv HTTP checks verified the top eight papers
listed below. `/deep-research` was not invoked. The nulled exploration-strategy
class was not re-ingested. No model load, training, leaderboard submission, or
solve claim was made; this is a no solve claim ingestion note.

**L1-wall context imported:** `.445` left the wall at L1-first-contact:
the winning L1 prefix is not entering the pool, frame-only order-1 features are
at chance for the current diagnosis, and exploration reweighting has nulled.
The .446 target is therefore perception/representation: make a novel game's
winner representable/proposable before ranking.

**Verified source set:**
- arXiv:1802.04687 -- Neural Relational Inference for Interacting Systems
- arXiv:1911.12247 -- Contrastive Learning of Structured World Models
- arXiv:2006.15055 -- Object-Centric Learning with Slot Attention
- arXiv:2402.03326 -- Slot Structured World Models
- arXiv:2507.03298 -- Dyn-O: Building Structured World Models with Object-Centric Representations
- arXiv:2601.06604 -- Object-Centric World Models Meet Monte Carlo Tree Search
- arXiv:2602.11389 -- Causal-JEPA: Learning World Models through Object-Level Latent Masking
- arXiv:2606.12316 -- Slots, Transitions, Loops: Learning Composable World Models for ARC

**SOTA -> perception/representation mapping for the L1 wall:**
- **Slotized ARC object-state proposal binder** (arXiv:2006.15055, arXiv:2606.12316): maps to L1-FIRST-CONTACT / GAP-ARCH-FEATURES; Replace frame-only order-1 features with color/object slots, masks, object permanence IDs, and spatial relation tokens from each frame. Winner representable/proposable test: A novel game's winning prefix is representable only if each decisive click/action can bind to a slot object and a before/after relation, with no terminal win oracle. Proposable output: Propose slot-conditioned action templates such as select/move/merge/fill over objects instead of only raw coordinate retries. Takes over: Takes over frame-delta scalar features that cannot name the object whose transformation makes the first win possible. Fails when: Slot binding drifts between frames, small ARC objects are merged into background, or the proposer cannot turn slots into executable actions.
- **Object-relational transition graph proposer** (arXiv:1802.04687, arXiv:1911.12247, arXiv:2402.03326): maps to L1-FIRST-CONTACT / GAP-ARCH-FEATURES; Represent each state as objects plus inferred interaction edges, then learn action-conditioned transition rules over that graph. Winner representable/proposable test: The winning prefix is representable if the graph transition predicts the decisive object/relation change before the live explorer sees a win. Proposable output: Propose near-miss next states and action prefixes by editing object relations that the graph predicts will change. Takes over: Takes over exploration-prior reweighting when the candidate pool lacks a structured transition that could produce the winner. Fails when: Edges encode visual proximity rather than mechanics, negatives are too easy, or relation edits produce impossible grid states.
- **Object-centric latent-dynamics planner with MCTS** (arXiv:2507.03298, arXiv:2601.06604): maps to L1-FIRST-CONTACT / GAP-ARCH-FEATURES; Use an object-centric world model as the state substrate for short lookahead planning, keeping dynamics-aware features separate from visual nuisance features. Winner representable/proposable test: The winning prefix is representable if MCTS over object latents can reach a low-depth candidate state whose replayable action prefix was absent from the frame-only pool. Proposable output: Propose a small set of replayable object-state rollouts and concrete action prefixes for the live verifier to test. Takes over: Takes over unguided first-contact exploration after representation, not search depth, is the limiting factor. Fails when: Latent rollout drift compounds, object discovery fails under clutter, or MCTS optimizes a latent state that cannot be grounded into actions.
- **Causal object-level JEPA shortcut guard** (arXiv:2602.11389, arXiv:1802.04687): maps to L1-FIRST-CONTACT / GAP-ARCH-FEATURES; Mask or intervene on object-level latents so the encoder must infer interaction-dependent structure rather than frame provenance shortcuts. Winner representable/proposable test: The winning prefix is representable only if masked-object prediction recovers the decisive hidden relation and the relation survives counterfactual object swaps. Proposable output: Propose shortcut-guarded object features used to filter and construct candidate prefixes before any ranker scores them. Takes over: Takes over chance-level order-1 features by forcing the representation to encode causal object dependencies. Fails when: The mask objective can be solved from color/style shortcuts, the object set is wrong, or counterfactual swaps break valid mechanics.
- **ARC composable slot-transition loop model** (arXiv:2606.12316, arXiv:2006.15055, arXiv:1911.12247): maps to L1-FIRST-CONTACT / GAP-ARCH-FEATURES; Learn ARC rules as composable transitions over color slots, objects, loops, and demonstration-conditioned task summaries. Winner representable/proposable test: The winning prefix is representable if the demonstration-conditioned slot loop proposes the missing first-contact transformation on a held-out game. Proposable output: Propose an executable transformation sketch or action prefix that enters the live candidate pool before static ranking. Takes over: Takes over generic exploration levers by changing what the pool can express: object transformations instead of raw prefix frequency. Fails when: The demo-conditioned summary memorizes family style, the loop fails on single-shot mechanics, or transformations cannot lower to live actions.

flagged_for_v446: loop_owm_slot_transition_proposer (arXiv:2606.12316 + arXiv:2006.15055 + arXiv:1911.12247)
flagged_for_v446: object_relational_world_model_mcts (arXiv:2601.06604 + arXiv:2402.03326 + arXiv:2507.03298)
flagged_for_v446: causal_object_jepa_shortcut_guard (arXiv:2602.11389 + arXiv:1802.04687)

**Bottom line for .446:** prioritize Loop-OWM slot-transition proposals
with Slot Attention/C-SWM as the substrate, then pair that with object-relational
world-model MCTS. Use Causal-JEPA as the shortcut guard so the learned
representation captures object interactions rather than provenance or frame
style. This is perception, not more exploration.
<!-- EXP4838-SOTA-INGESTION-PERCEPTION-REPRESENTATION-END -->

<!-- EXP4828-SOTA-INGESTION-CROSS-FAMILY-TRANSFER-START -->
## 2026-06-26 Exp 4828 - .445 cross-family transfer SOTA ingestion - INGESTED

**Status:** INGESTED into `results/experiment_4828_sota_ingestion_cross_family_transfer.json`.

**Preconditions:** `research-studying.md` and `research-references.md` were
present. `scripts/sweep_clusters.py` emitted verifier/reward, EBM, and
neural-guided search cluster URLs. `scripts/sweep_semscholar.py` was run on
three focused cross-family transfer queries and returned HTTP 429 for all of
them, so no S2-only source was promoted. Low-concurrency WebSearch/WebFetch
plus direct arXiv HTTP checks verified the top eight papers listed below.
`/deep-research` was not invoked. No model load, training, leaderboard
submission, or solve claim was made; this is a no solve claim ingestion note.

**Verified source set:**
- arXiv:1911.08731 -- Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization
- arXiv:2003.00688 -- Out-of-Distribution Generalization via Risk Extrapolation (REx)
- arXiv:2007.01434 -- In Search of Lost Domain Generalization
- arXiv:2012.07421 -- WILDS: A Benchmark of in-the-Wild Distribution Shifts
- arXiv:2311.14743 -- A Baseline Analysis of Reward Models' Ability To Accurately Analyze Foundation Models Under Distribution Shift
- arXiv:2403.13787 -- RewardBench: Evaluating Reward Models for Language Modeling
- arXiv:2602.08489 -- Beyond Correctness: Learning Robust Reasoning via Transfer
- arXiv:2605.25629 -- When In-Distribution Gains Fail: Evaluating Weak-to-Strong Reward Models under Preference Shift

**SOTA -> S4 cross-family transfer mapping:**
- **Leave-one-family reward/verifier transfer gate** (arXiv:2311.14743, arXiv:2403.13787, arXiv:2007.01434, arXiv:2012.07421): maps to S4; Split the ARC family corpus into source families and one held-out target family, then report S4 energy accuracy, calibration, and OOD transfer deltas separately for prompt shift and response shift. Held-out family test: A family is successful only if its held-out score, worst-family score, and calibration stay positive; pooled source-family averages cannot authorize S4. Takes over: Takes over the old pooled transfer readout that let a strong source family hide a brittle held-out family. Fails when: The split leaks level identity, the held-out family has too few examples, or the verifier is calibrated only on source-family outputs.
- **Representation anchoring for verifier-energy fine-tuning** (arXiv:2605.25629): maps to S4; Fine-tune the S4 energy with an anchor penalty to keep the verifier near the pretrained representation while allowing source-family adaptation where it improves held-out transfer. Held-out family test: Pick the anchor weight by source-family validation only, then score the held-out family once and require transfer-aware gain rather than source-family memorization. Takes over: Takes over unconstrained verifier fine-tuning that can chase family style features and lose the transferable representation. Fails when: The base representation lacks cross-family signal, the anchor is so strong that useful adaptation is blocked, or source labels encode family shortcuts.
- **Worst-family group DRO energy training** (arXiv:1911.08731, arXiv:2012.07421): maps to S4; Treat source game families as groups and optimize the S4 verifier for worst-family loss with explicit L2 or early-stopping regularization before the held-out family is touched. Held-out family test: Report worst-source-family and held-out-family energy separation; the method passes only if the worst group improves without collapsing the held-out family. Takes over: Takes over average-loss energy training when rare mechanics families are swamped by easier frequent families. Fails when: Family labels are noisy, group sizes are too small for stable worst losses, or regularization is too weak to generalize beyond training groups.
- **Risk extrapolation across source families** (arXiv:2003.00688, arXiv:2007.01434): maps to S4; Add a V-REx-style penalty that reduces variance in verifier-energy risk across source families, with DomainBed-style model selection that never tunes on the held-out family. Held-out family test: Score the held-out family after source-only model selection and compare against ERM, anchored fine-tuning, and group DRO controls. Takes over: Takes over source-family ERM when S4 needs a smoother transfer surface across mechanics families. Fails when: Source-family variation does not cover the held-out shift, the risk-equality penalty suppresses genuinely useful family-specific signals, or model selection overfits source domains.
- **Transferable-reward prefix continuation stress** (arXiv:2602.08489): maps to S4; Stress the S4 energy by asking whether partial plans or reasoning prefixes generated from one source family help a separate policy continue in another family, rather than only judging final answers. Held-out family test: A held-out family earns credit only when source-family prefixes improve continuation quality under the target-family verifier without manual target labels. Takes over: Takes over final-outcome-only verifier checks that miss brittle reasoning traces which cannot transfer across models or families. Fails when: Families do not share transferable substructure, prefix swaps create invalid action contexts, or the continuation model learns a style cue.

flagged_for_v445: anchor_leave_one_family_transfer_gate (arXiv:2605.25629 + arXiv:2311.14743 + arXiv:2403.13787)
flagged_for_v445: worst_family_group_dro_s4_energy (arXiv:1911.08731 + arXiv:2012.07421)
flagged_for_v445: rex_transferable_reward_stress (arXiv:2003.00688 + arXiv:2602.08489 + arXiv:2007.01434)

**Bottom line for .445:** prioritize the Anchor plus leave-one-family transfer
gate because it directly attacks the .393/GAP-4 failure mode: a verifier energy
can look good in-distribution while failing on the family that matters. Pair it
with worst-family Group DRO as the robust-training control, then use REx and
RLTR-style transferable-reward stress as the falsifier when source families
still look too easy.
<!-- EXP4828-SOTA-INGESTION-CROSS-FAMILY-TRANSFER-END -->

<!-- EXP4818-SOTA-INGESTION-ENERGY-GUIDED-GENERATION-START -->
## 2026-06-26 Exp 4818 - .444 energy-guided generation SOTA ingestion - INGESTED

**Status:** INGESTED into `results/experiment_4818_sota_ingestion_energy_guided_generation.json`.

**Preconditions:** `research-studying.md` and `research-references.md` were
present. `scripts/sweep_clusters.py` emitted the EBM and neural-guided search
cluster URLs. `scripts/sweep_semscholar.py` was run on three focused
energy-guided generation queries and returned HTTP 429 for all of them, so no
S2-only source was promoted. Low-concurrency WebSearch/WebFetch plus direct
arXiv HTTP checks verified the top eight papers listed below. `/deep-research`
was not invoked. No model load, training, leaderboard submission, or solve
claim was made; this is a no solve claim ingestion note.

**Verified source set:**
- arXiv:1806.10230 -- Guided evolutionary strategies: Augmenting random search with surrogate gradients
- arXiv:1909.06878 -- Model Based Planning with Energy Based Models
- arXiv:2202.11705 -- COLD Decoding: Energy-based Constrained Text Generation with Langevin Dynamics
- arXiv:2207.12598 -- Classifier-Free Diffusion Guidance
- arXiv:2305.12018 -- BOLT: Fast Energy-based Controlled Text Generation with Tunable Biases
- arXiv:2309.15028 -- Do not throw away your value model! Generating more preferable text with Value-Guided Monte-Carlo Tree Search decoding
- arXiv:2502.07202 -- Monte Carlo Tree Diffusion for System 2 Planning
- arXiv:2605.28814 -- Self-Improving Language Models with Bidirectional Evolutionary Search

**SOTA -> S3 energy-guided generation mapping:**
- **Energy-constrained sampler with fast logit-bias refinement** (arXiv:2202.11705, arXiv:2305.12018): maps to S3; Represent action programs or latent plans as variables under the S3 energy, then use short Langevin-style edits or BOLT-style tunable biases to move candidates before hard mechanics verification. Winner insertion: Generate a small guided batch, replay-verify each candidate, and put the lowest-energy verified winner into the explorer pool. Takes over: Takes over blind local mutation when the pool has no candidate that already wins but the verifier can score partial plausibility. Fails when: The candidate representation has no smooth edit neighborhood, energy gradients point toward unplayable programs, or bias tuning collapses diversity before verification.
- **Classifier/score-guided proposal sampler** (arXiv:2207.12598): maps to S3; Treat S3 energy as a classifier-like guidance signal during proposal construction, with guidance strength increased only after action syntax and transition checks keep the candidate executable. Winner insertion: Sample a guided batch, choose the verified candidate with the best lower-is-better score, and put that winner into the live pool. Takes over: Takes over unguided proposal batches whose candidates are scored only after generation has already spent the action budget. Fails when: Guidance scale overwhelms diversity, the score follows a provenance shortcut, or high guidance yields valid-looking but unplayable plans.
- **Value-guided tree and diffusion generation** (arXiv:2309.15028, arXiv:2502.07202): maps to S3; Expand partial generated plans as a tree and use negative energy as the value term for partial rollouts; revisit branches whose decoded or denoised continuations improve the trust score. Winner insertion: Complete the best tree leaf into executable actions, verify it, and put the verified low-energy winner into the candidate pool. Takes over: Takes over one-shot generation by spending inference-time compute on partial candidates before they become expensive live actions. Fails when: Partial-plan energy is poorly calibrated, branching cost exceeds the action-efficiency budget, or the tree repeatedly explores equivalent leaves.
- **Energy-as-fitness evolutionary pool search** (arXiv:2605.28814, arXiv:1806.10230): maps to S3; Run a tiny population over action programs where fitness is negative S3 energy plus executable validity, and use recombination or guided random search to escape high-probability but losing rollouts. Winner insertion: Select the verified elite with the best energy-adjusted fitness and put that winner into the explorer pool while retaining alternates. Takes over: Takes over single-candidate mutation by preserving a small population of high-quality candidates instead of only the current best guess. Fails when: Fitness overweights novelty, surrogate energy points away from true transition mechanics, or evolution burns budget without verified elites.
- **Plan-with-energy state trajectory generator** (arXiv:1909.06878, arXiv:2502.07202): maps to S3; Generate intermediate state trajectories under the energy model, then ask the action realizer to synthesize concrete actions that reach the best low-energy trajectory. Winner insertion: Promote a trajectory only after it is realized as executable actions; the realized action sequence becomes the winner inserted into the pool. Takes over: Takes over static plan reranking when no current candidate reaches a goal-like state but the energy can propose plausible intermediate states. Fails when: State-space descent invents unreachable grids, the action realizer cannot realize the trajectory, or energy ignores a small decisive change.

flagged_for_v444: bolt_cold_cfg_value_tree_generator_for_s3 (arXiv:2202.11705 + arXiv:2305.12018 + arXiv:2207.12598 + arXiv:2309.15028 + arXiv:2502.07202)
flagged_for_v444: bes_energy_fitness_pool_inserter (arXiv:2605.28814 + arXiv:1806.10230 + arXiv:1909.06878)

**Bottom line for .444:** prioritize the BOLT/COLD/CFG/value-tree generator
because it can put a winner into the pool with the smallest live-stack change:
generate a guided batch, verify hard mechanics, then promote the best
low-energy verified candidate. Keep BES-style energy-as-fitness evolution as
the fallback when guided chains collapse, because recombination can escape
high-probability losing rollouts while still using the same verifier energy.
<!-- EXP4818-SOTA-INGESTION-ENERGY-GUIDED-GENERATION-END -->

<!-- EXP4808-SOTA-INGESTION-ENERGY-GUIDED-GENERATION-START -->
## 2026-06-26 Exp 4808 - .443 energy-guided generation SOTA ingestion - INGESTED

**Status:** INGESTED into `results/experiment_4808_sota_ingestion_energy_guided_generation.json`.

**Preconditions:** `research-studying.md` and `research-references.md` were
present. `scripts/sweep_clusters.py` emitted the EBM and neural-guided search
cluster URLs. `scripts/sweep_semscholar.py` was run on three focused
energy-guided generation queries and returned HTTP 429 for all of them, so no
S2-only source was promoted. Low-concurrency WebSearch/WebFetch plus direct
arXiv HTTP checks verified the top eight papers listed below. `/deep-research`
was not invoked. No model load, training, leaderboard submission, or solve
claim was made; this is a no solve claim ingestion note.

**Verified source set:**
- arXiv:1806.10230 -- Guided evolutionary strategies: Augmenting random search with surrogate gradients
- arXiv:1909.06878 -- Model Based Planning with Energy Based Models
- arXiv:2202.11705 -- COLD Decoding: Energy-based Constrained Text Generation with Langevin Dynamics
- arXiv:2207.12598 -- Classifier-Free Diffusion Guidance
- arXiv:2305.12018 -- BOLT: Fast Energy-based Controlled Text Generation with Tunable Biases
- arXiv:2309.15028 -- Do not throw away your value model! Generating more preferable text with Value-Guided Monte-Carlo Tree Search decoding
- arXiv:2502.07202 -- Monte Carlo Tree Diffusion for System 2 Planning
- arXiv:2605.28814 -- Self-Improving Language Models with Bidirectional Evolutionary Search

**SOTA -> S3 energy-guided generation mapping:**
- **Energy-constrained sampler with fast logit-bias refinement** (arXiv:2202.11705, arXiv:2305.12018): maps to S3; Represent action programs or latent plans as variables under the S3 energy, then use short Langevin-style edits or BOLT-style tunable biases to move candidates before hard mechanics verification. Winner insertion: Generate a small guided batch, replay-verify each candidate, and put the lowest-energy verified winner into the explorer pool. Takes over: Takes over blind local mutation when the pool has no candidate that already wins but the verifier can score partial plausibility. Fails when: The candidate representation has no smooth edit neighborhood, energy gradients point toward unplayable programs, or bias tuning collapses diversity before verification.
- **Classifier/score-guided proposal sampler** (arXiv:2207.12598): maps to S3; Treat S3 energy as a classifier-like guidance signal during proposal construction, with guidance strength increased only after action syntax and transition checks keep the candidate executable. Winner insertion: Sample a guided batch, choose the verified candidate with the best lower-is-better score, and put that winner into the live pool. Takes over: Takes over unguided proposal batches whose candidates are scored only after generation has already spent the action budget. Fails when: Guidance scale overwhelms diversity, the score follows a provenance shortcut, or high guidance yields valid-looking but unplayable plans.
- **Value-guided tree and diffusion generation** (arXiv:2309.15028, arXiv:2502.07202): maps to S3; Expand partial generated plans as a tree and use negative energy as the value term for partial rollouts; revisit branches whose decoded or denoised continuations improve the trust score. Winner insertion: Complete the best tree leaf into executable actions, verify it, and put the verified low-energy winner into the candidate pool. Takes over: Takes over one-shot generation by spending inference-time compute on partial candidates before they become expensive live actions. Fails when: Partial-plan energy is poorly calibrated, branching cost exceeds the action-efficiency budget, or the tree repeatedly explores equivalent leaves.
- **Energy-as-fitness evolutionary pool search** (arXiv:2605.28814, arXiv:1806.10230): maps to S3; Run a tiny population over action programs where fitness is negative S3 energy plus executable validity, and use recombination or guided random search to escape high-probability but losing rollouts. Winner insertion: Select the verified elite with the best energy-adjusted fitness and put that winner into the explorer pool while retaining alternates. Takes over: Takes over single-candidate mutation by preserving a small population of high-quality candidates instead of only the current best guess. Fails when: Fitness overweights novelty, surrogate energy points away from true transition mechanics, or evolution burns budget without verified elites.
- **Plan-with-energy state trajectory generator** (arXiv:1909.06878, arXiv:2502.07202): maps to S3; Generate intermediate state trajectories under the energy model, then ask the action realizer to synthesize concrete actions that reach the best low-energy trajectory. Winner insertion: Promote a trajectory only after it is realized as executable actions; the realized action sequence becomes the winner inserted into the pool. Takes over: Takes over static plan reranking when no current candidate reaches a goal-like state but the energy can propose plausible intermediate states. Fails when: State-space descent invents unreachable grids, the action realizer cannot realize the trajectory, or energy ignores a small decisive change.

flagged_for_v443: bolt_cold_cfg_value_tree_generator_for_s3 (arXiv:2202.11705 + arXiv:2305.12018 + arXiv:2207.12598 + arXiv:2309.15028 + arXiv:2502.07202)
flagged_for_v443: bes_energy_fitness_pool_inserter (arXiv:2605.28814 + arXiv:1806.10230 + arXiv:1909.06878)

**Bottom line for .443:** prioritize the BOLT/COLD/CFG/value-tree generator
because it can put a winner into the pool with the smallest live-stack change:
generate a guided batch, verify hard mechanics, then promote the best
low-energy verified candidate. Keep BES-style energy-as-fitness evolution as
the fallback when guided chains collapse, because recombination can escape
high-probability losing rollouts while still using the same verifier energy.
<!-- EXP4808-SOTA-INGESTION-ENERGY-GUIDED-GENERATION-END -->

<!-- EXP4798-SOTA-INGESTION-ENERGY-GUIDED-GENERATION-START -->
## 2026-06-26 Exp 4798 - .442 energy-guided generation SOTA ingestion - INGESTED

**Status:** INGESTED into `results/experiment_4798_sota_ingestion_energy_guided_generation.json`.

**Preconditions:** `research-studying.md` and `research-references.md` were
present. `scripts/sweep_clusters.py` emitted the EBM and neural-guided search
cluster URLs. `scripts/sweep_semscholar.py` was run on a focused
energy-guided generation query and returned HTTP 429, so no S2-only source was
promoted. Low-concurrency WebSearch/WebFetch plus direct arXiv HTTP checks
verified the top eight papers listed below. `/deep-research` was not invoked.
No model load, training, leaderboard submission, or solve claim was made; this
is a no solve claim ingestion note.

**S1/S2 context imported:** `results/experiment_4781_structural_energy_s1_contrastive_landscape.json` reports
`success_structural_energy_s1_landscape_authorizes_s2` and
`results/experiment_4791_structural_energy_s2_offpath_trust_gate.json` reports
`complete_structural_energy_s2_no_live_trust_value` with
`s2_live_path_reachable=True`.
The .442 planner may use S1/S2 lower-is-better energy to guide S3 generation,
but not as an environment oracle or solve claim.

**Verified source set:**
- arXiv:1806.10230 -- Guided evolutionary strategies: Augmenting random search with surrogate gradients
- arXiv:1909.06878 -- Model Based Planning with Energy Based Models
- arXiv:2012.04322 -- Quality-Diversity Optimization: a novel branch of stochastic optimization
- arXiv:2105.05233 -- Diffusion Models Beat GANs on Image Synthesis
- arXiv:2202.11705 -- COLD Decoding: Energy-based Constrained Text Generation with Langevin Dynamics
- arXiv:2207.12598 -- Classifier-Free Diffusion Guidance
- arXiv:2309.15028 -- Do not throw away your value model! Generating more preferable text with Value-Guided Monte-Carlo Tree Search decoding
- arXiv:2502.07202 -- Monte Carlo Tree Diffusion for System 2 Planning

**SOTA -> S3 energy-guided generation mapping:**
- **Energy-constrained Langevin candidate generator** (arXiv:2202.11705): maps to S3; Treat action programs, latent plans, or textual program sketches as variables under a composite energy: S1 transition plausibility, S2 trust gate, hard ARC action validity, and pool novelty. Winner insertion: Run short energy-guided sampling chains, verify each decoded candidate, and insert the lowest-energy verified winner into the explorer pool. Takes over: Takes over blind local repair and purely random candidate mutation when the explorer has no candidate that already wins. Fails when: The candidate representation has no smooth neighborhood, low-energy edits violate executable mechanics, or sampling collapses to duplicates.
- **Classifier/score-guided proposal sampler** (arXiv:2105.05233, arXiv:2207.12598): maps to S3; Use the S1/S2 energy as a classifier-like or score-guidance signal during proposal construction, increasing guidance only after the hard action and transition validators keep the candidate executable. Winner insertion: Sample a small guided batch, choose the verified candidate with the best lower-is-better energy, and put that winner into the live pool. Takes over: Takes over unguided proposal batches whose candidates are only scored after generation has already spent the action budget. Fails when: Guidance scale overwhelms diversity, the score follows a provenance shortcut, or high guidance makes syntactically valid but unplayable plans.
- **Value-guided tree generation** (arXiv:2309.15028, arXiv:2502.07202): maps to S3; Expand partial generated plans as a tree, using negative S1/S2 energy as the value term for partial rollouts and revisiting branches whose denoised or decoded continuations improve the trust score. Winner insertion: Complete the best tree leaf into an executable plan, verify it, and put the verified low-energy winner into the candidate pool. Takes over: Takes over one-shot generation by letting inference-time compute refine partial plans before they become expensive live actions. Fails when: Partial-plan energy is poorly calibrated, branching cost exceeds the action-efficiency budget, or the tree repeatedly explores equivalent leaves.
- **Energy-as-fitness evolutionary pool search** (arXiv:1806.10230, arXiv:2012.04322): maps to S3; Run a tiny population over generated action programs where fitness is negative S1/S2 energy plus executable validity and a novelty/diversity descriptor for game mechanics not yet covered by the pool. Winner insertion: Select the verified elite with the best energy-adjusted fitness and put that winner into the explorer pool while retaining diverse alternates. Takes over: Takes over single-candidate mutation by preserving a small archive of high-quality diverse candidates instead of only the current best guess. Fails when: Fitness overweights novelty, surrogate energy points away from true transition mechanics, or the population burns budget without new verified elites.
- **Plan-with-energy state trajectory generator** (arXiv:1909.06878, arXiv:2502.07202): maps to S3; Generate intermediate state trajectories under the S1/S2 energy, then ask the existing inducer/action realizer to synthesize the concrete action program that reaches the best low-energy trajectory. Winner insertion: Only promote a trajectory after it is realized as executable actions; the realized action sequence becomes the winner inserted into the pool. Takes over: Takes over static plan reranking when no existing candidate reaches a goal-like state but the energy can propose plausible intermediate states. Fails when: State-space descent invents unreachable grids, the action realizer cannot realize the trajectory, or energy ignores a small decisive object change.

flagged_for_v442: cold_cfg_value_tree_generator_for_s3 (arXiv:2202.11705 + arXiv:2105.05233 + arXiv:2207.12598 + arXiv:2309.15028 + arXiv:2502.07202)
flagged_for_v442: energy_fitness_qd_pool_inserter (arXiv:1806.10230 + arXiv:2012.04322 + arXiv:1909.06878)

**Bottom line for .442:** prioritize the COLD/CFG/value-tree generator path
because it can put a winner into the pool with the smallest change to the live
explorer: generate a guided batch, verify hard mechanics, then promote the
best low-energy verified candidate. Keep the energy-fitness quality-diversity
path as the fallback when single-chain generation collapses, because it
preserves diverse elites instead of merely reranking a frozen pool.
<!-- EXP4798-SOTA-INGESTION-ENERGY-GUIDED-GENERATION-END -->

<!-- EXP4788-SOTA-INGESTION-ENERGY-GUIDED-SEARCH-START -->
## 2026-06-26 Exp 4788 - .441 energy-guided search SOTA ingestion - INGESTED

**Status:** INGESTED into `results/experiment_4788_sota_ingestion_energy_guided_search.json`.

**Preconditions:** `research-studying.md` and `research-references.md` were
present. `scripts/sweep_clusters.py` emitted the EBM and neural-guided search
cluster URLs. `scripts/sweep_semscholar.py` was run on two focused
energy-guided search queries and returned HTTP 429, so no S2-only source was
promoted. Low-concurrency WebSearch/WebFetch plus direct arXiv HTTP checks
verified the top eight papers listed below. `/deep-research` was not invoked.
No model load, training, leaderboard submission, or solve claim was made; this
is a no solve claim ingestion note.

**S1 context imported:** `results/experiment_4781_structural_energy_s1_contrastive_landscape.json` reports
`success_structural_energy_s1_landscape_authorizes_s2` with
`energy_ranking_loo_auroc_mean=0.7134961314270525`
and `denoising_direction_agreement=0.6223390275952694`.
The .441 planner should treat S1 as a lower-is-better guide for search and
generation, not as an environment oracle.

**Verified source set:**
- arXiv:1909.06878 -- Model Based Planning with Energy Based Models
- arXiv:2103.11505 -- Policy-Guided Heuristic Search with Guarantees
- arXiv:2202.11705 -- COLD Decoding: Energy-based Constrained Text Generation with Langevin Dynamics
- arXiv:2206.09914 -- A Langevin-like Sampler for Discrete Distributions
- arXiv:2304.14391 -- Energy-based Models are Zero-Shot Planners for Compositional Scene Rearrangement
- arXiv:2309.15028 -- Do not throw away your value model! Generating more preferable text with Value-Guided Monte-Carlo Tree Search decoding
- arXiv:2502.07202 -- Monte Carlo Tree Diffusion for System 2 Planning
- arXiv:2505.10819 -- PoE-World: Compositional World Modeling with Products of Programmatic Experts

**SOTA -> S2/S3 energy-guided search mapping:**
- **Energy/value-guided MCTS frontier controller** (arXiv:2309.15028, arXiv:2502.07202): maps to S2; Use negative S1 energy as the value term in the S2 tree policy: expand induced partial plans whose predicted transition rollouts fall downhill, and re-score every child through verify before acting. Takes over: Takes over uniform or static best-first expansion when the live induce->verify->plan loop has several plausible next engine states. Fails when: The S1 energy is miscalibrated off distribution, tree branching is too wide for value reuse, or partial-rollout scores become a shortcut for known generator provenance.
- **Energy-weighted policy-guided best-first search** (arXiv:2103.11505, arXiv:2309.15028): maps to S2; Rank the live frontier by a composite priority: learned policy prior for cheap action plausibility plus lower S1 energy for transition trust, with verification gating before a node consumes action budget. Takes over: Takes over hand-tuned value_weight and FIFO frontier scheduling by making the S1 landscape the admissibility-aware heuristic term. Fails when: The energy and policy disagree without a tie-break budget, low-energy nodes are all duplicates, or the heuristic over-prunes the one action that would reveal the game mechanic.
- **Gradient-guided discrete energy search** (arXiv:2202.11705, arXiv:2206.09914): maps to S3; Treat candidate action programs, latent action slots, or text plans as discrete variables and use S1-energy gradients or discrete Langevin proposals to mutate them before normal verify accepts any plan. Takes over: Takes over random local repair of generated candidates by proposing low-energy edits that still pass the executable verifier. Fails when: The candidate representation is not differentiable enough to expose a useful neighborhood, gradient proposals violate hard grid mechanics, or repeated Langevin steps collapse to near-duplicate candidates.
- **EBM-as-planner state trajectory refinement** (arXiv:1909.06878, arXiv:2304.14391): maps to S3; Use S1 energy directly as the planner objective over intermediate state hypotheses: sample or optimize a sequence of latent next states, then ask the existing inducer to synthesize actions that realize them. Takes over: Takes over plan-in-model reranking when no winning action sequence is present in the static candidate pool. Fails when: State-space descent finds physically impossible intermediate grids, the action realizer cannot execute the inferred state path, or the energy ignores small but decisive object changes.
- **Product-of-experts compositional planning** (arXiv:2304.14391, arXiv:2505.10819, arXiv:1909.06878): maps to S2, S3; Make the S1 energy one expert in a product with code-world-model, spatial-relation, and action-effect experts; S2 scores which factor to trust, while S3 composes factors to generate new plans. Takes over: Takes over monolithic world-model acceptance by requiring each expert factor to improve or preserve the joint product energy before a plan is promoted. Fails when: Experts double-count the same shortcut, one factor dominates the product, sparse observations synthesize the wrong programmatic expert, or product energy improves without executable action support.

flagged_for_v441: energy_value_guided_mcts_frontier_controller (arXiv:2309.15028 + arXiv:2502.07202 + arXiv:2103.11505)
flagged_for_v441: ebm_poe_planner_for_s3_generation (arXiv:1909.06878 + arXiv:2304.14391 + arXiv:2505.10819)

**Bottom line for .441:** start with the energy/value-guided MCTS frontier
controller for S2, because it grafts the S1 energy onto the live
induce->verify->plan loop with the smallest change in control flow. In
parallel, prepare the EBM/PoE planner path for S3 generation so low-energy
trajectory and product-of-experts proposals can make new candidate plans appear
instead of merely reranking a frozen pool.
<!-- EXP4788-SOTA-INGESTION-ENERGY-GUIDED-SEARCH-END -->

<!-- EXP4778-SOTA-INGESTION-STRUCTURAL-ENERGY-START -->
## 2026-06-26 Exp 4778 - .440 structural-energy SOTA ingestion after S0' - INGESTED

**Status:** INGESTED into `results/experiment_4778_sota_ingestion_structural_energy.json`.

**Preconditions:** `research-studying.md` and `research-references.md` were
present. `scripts/sweep_clusters.py` emitted the EBM, action/effect, and
neural-guided/world-model cluster URLs. `scripts/sweep_semscholar.py` was run
on three focused queries and returned HTTP 429, so no S2-only source was
promoted. Low-concurrency WebSearch/WebFetch plus direct arXiv HTTP checks
verified the top eight papers listed below. `/deep-research` was not invoked.
No model load, training, leaderboard submission, or solve claim was made; this
is a no solve claim ingestion note.

**S0' context imported:** `results/experiment_4771_structural_energy_s0prime_origin_matched.json` reports
`success_structural_energy_s0prime_reopens_s1` with
`origin_probe_auroc=0.5` and
`shuffled_label_control_auroc=0.5033091959271814`.
Because the artifact is also adversarial-flagged, .440 should treat leak-robust
evaluation as the gate for every S1-S4 continuation.

**Verified source set:**
- arXiv:1907.02893 -- Invariant Risk Minimization
- arXiv:1911.12247 -- Contrastive Learning of Structured World Models
- arXiv:2006.15055 -- Object-Centric Learning with Slot Attention
- arXiv:2301.08243 -- Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture
- arXiv:2505.10819 -- PoE-World: Compositional World Modeling with Products of Programmatic Experts
- arXiv:2505.13910 -- ShortcutProbe: Probing Prediction Shortcuts for Learning Robust Models
- arXiv:2510.04542 -- Code World Models for General Game Playing
- arXiv:2605.05138 -- Executable World Models for ARC-AGI-3 in the Era of Coding Agents

**SOTA -> S1-S4 structural-energy mapping:**
- **Slot-relational contrastive transition energy** (arXiv:2006.15055, arXiv:1911.12247, arXiv:2505.10819): maps to S1, S2; Takes over the S0' scalar structural probe by making the score an object-factorized transition energy that can rank off-path candidate next states and induced engines. Leak eval: Run only on origin-matched induced rows, then require origin/provenance probe failure and shortcut probes before accepting the energy as oracle-distinct. Fails when: Slot binding drifts across frames, PoE factors memorize source provenance, or contrastive negatives are too easy to separate without learning transition mechanics.
- **Executable PoE/code world-model trust energy** (arXiv:2505.10819, arXiv:2605.05138, arXiv:2510.04542): maps to S2, S3, S4; Takes over binary executable-model accept/reject checks by ranking world-model factors and code engines on transition consistency where the environment win-check is unavailable. Leak eval: Each executable factor must be evaluated on held-out transitions with no terminal win oracle, no source-origin token, and a provenance probe that stays at chance. Fails when: The generated program overfits public prefixes, hidden-state inference is wrong, or the trust energy leaks private solution facts through a verifier shortcut.
- **JEPA latent residual energy for transfer survival** (arXiv:2301.08243, arXiv:1911.12247): maps to S1, S4; Takes over raw frame-marginal controls by asking whether the structural signal survives in a predictive representation that is not allowed to encode source provenance. Leak eval: Pair leave-one-game and leave-one-family folds with shuffled-label controls so a JEPA residual cannot pass by memorizing nuisance origin. Fails when: The latent discards exact grid consequences, learns a value head rather than transition mechanics, or transfers only within one ARC family.
- **Shortcut/invariance leak-robust energy evaluation gate** (arXiv:2505.13910, arXiv:1907.02893): maps to S1, S2, S3, S4; Takes over the ad hoc S0 origin-probe warning by making leak detection a required gate with provenance probes, shuffled-label controls, and counterfactual/invariance stress tests. Leak eval: This is the .440 acceptance harness: S1-S4 methods must pass shortcut probe, origin/provenance chance probe, and invariance under counterfactual origin swaps before roadmap promotion. Fails when: Shortcut probes have no hard negatives, environments are too correlated with labels for invariance to identify the causal feature, or the probe is tuned after seeing the target fold.

**Leak-robust evaluation note:** S0' reopens S1 only if .440 treats origin/provenance leakage as a first-class failure mode: every energy result needs origin/provenance controls, shortcut probes, and counterfactual/invariance stress tests.
Use ShortcutProbe and IRM as the explicit shortcut/invariance evaluation
templates.
- origin-matched induced-only rows for positive and negative transition candidates
- chance-level origin/provenance probe before any oracle-distinct continuation claim
- shuffled-label or ShortcutProbe-style latent shortcut control on identical folds
- counterfactual/invariance probes over game family, origin, and candidate-generator environments

flagged_for_v440: slot_relational_contrastive_energy_s0prime_guarded (arXiv:2006.15055 + arXiv:1911.12247 + arXiv:2505.10819 + arXiv:2505.13910 + arXiv:1907.02893)
flagged_for_v440: poe_code_world_model_trust_gate_after_s0prime (arXiv:2505.10819 + arXiv:2605.05138 + arXiv:2510.04542)

**Bottom line for .440:** prioritize the Slot Attention + C-SWM
slot-relational contrastive energy rerun under the explicit leak gate, then
connect PoE/code world-model trust only after origin/provenance and
shortcut/invariance controls stay clean.
<!-- EXP4778-SOTA-INGESTION-STRUCTURAL-ENERGY-END -->

<!-- EXP4768-SOTA-INGESTION-STRUCTURAL-ENERGY-START -->
## 2026-06-26 Exp 4768 - .438 structural-energy SOTA ingestion for S1-S4 - INGESTED

**Status:** INGESTED into `results/experiment_4768_sota_ingestion_structural_energy.json`.

**Preconditions:** `research-studying.md` and `research-references.md` were
present. `scripts/sweep_clusters.py` emitted the EBM and neural-guided/world
model cluster URLs. `scripts/sweep_semscholar.py` was run on three focused
queries and returned HTTP 429, so no S2-only source was promoted.
Low-concurrency WebSearch/WebFetch plus direct arXiv HTTP checks verified the
top eight papers listed below. `/deep-research` was not invoked. No model load,
training, leaderboard submission, or solve claim was made; this is a no solve claim
ingestion note.

**S0 context imported:** `results/experiment_4761_structural_energy_s0_core_bet_probe.json` reports
`complete: structural_energy_s0_retired_loo_0.746_null_or_leaky` with an origin-probe leak,
so the .439 planner should treat the S1-S4 entries as candidate inputs that
must address provenance leakage before any continuation claim.

**Verified source set:**
- arXiv:2006.15055 -- Object-Centric Learning with Slot Attention
- arXiv:2301.08243 -- Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture
- arXiv:2307.01668 -- Training Energy-Based Models with Diffusion Contrastive Divergences
- arXiv:2505.10819 -- PoE-World: Compositional World Modeling with Products of Programmatic Experts
- arXiv:2507.04920 -- Object-centric Denoising Diffusion Models for Physical Reasoning
- arXiv:2510.04542 -- Code World Models for General Game Playing
- arXiv:2602.02900 -- Manifold-Constrained Energy-Based Transition Models for Offline Reinforcement Learning
- arXiv:2605.05138 -- Executable World Models for ARC-AGI-3 in the Era of Coding Agents

**SOTA -> S1-S4 structural-energy mapping:**
- **Slot-factor contrastive transition energy** (arXiv:2006.15055, arXiv:2505.10819, arXiv:2602.02900, arXiv:2307.01668): maps to S1, S2; Takes over the S0 feature-only probe by making object_relational and frame_delta features into a contrastive energy landscape that can rank candidate transitions and induced engines. Fails when: Slot binding is unstable across ARC frames, the energy keeps learning induced-vs-real provenance like the S0 origin probe, or near-miss negatives are too sparse to harden the contrastive objective.
- **Product-of-experts executable world-model trust gate** (arXiv:2505.10819, arXiv:2510.04542, arXiv:2605.05138): maps to S2, S3, S4; Takes over the binary executable-model verifier by ranking factorized candidate engines on off-path structural energy where the environment win-check is unavailable. Fails when: The synthesized factors overfit a prefix, hidden-state inference is wrong, or executable-model verification leaks private solution facts instead of measuring transition consistency.
- **JEPA-style latent transition residual energy** (arXiv:2301.08243, arXiv:2602.02900): maps to S1, S4; Takes over raw frame-marginal controls by predicting semantically meaningful target representations from context/action structure. Fails when: The representation discards exact cell-level consequences, the learned latent becomes a value head in disguise, or cross-family transfer drops like the prior oracle-distinct failures.
- **Object-centric denoising structural prior** (arXiv:2507.04920, arXiv:2307.01668, arXiv:2602.02900): maps to S3, S4; Takes over static reranking by perturbing candidate transition trajectories toward low-energy, object-consistent alternatives that the bare explorer did not enumerate. Fails when: The diffusion prior smooths away discrete ARC mechanics, conditioning accidentally uses observed goal states as an oracle, or every lift is only reranking a winner already present in the pool.

flagged_for_v439: slot_factor_transition_energy_rerun_after_s0_origin_probe_leak_guard (arXiv:2006.15055 + arXiv:2505.10819 + arXiv:2602.02900 + arXiv:2307.01668)
flagged_for_v439: poe_code_world_model_trust_gate_with_cwm_hidden_state_planning (arXiv:2505.10819 + arXiv:2510.04542 + arXiv:2605.05138)

**Bottom line for .439:** start with the slot/factor contrastive transition
energy rerun, using Slot Attention object bindings plus PoE/programmatic
factors and MC-ETM-style hard near-miss negatives. In parallel, keep the
PoE/code-world-model trust gate ready as the strongest S2-S3 integration path
if the leak-hardened S1 energy clears the S0 origin-probe failure mode.
<!-- EXP4768-SOTA-INGESTION-STRUCTURAL-ENERGY-END -->

<!-- EXP4758-SOTA-INGESTION-START -->

## 2026-06-26 Exp 4758 - .437 structured world-model + grounded-goal SOTA ingestion - INGESTED

**Status:** INGESTED into `results/experiment_4758_sota_ingestion.json`.

**Preconditions:** Hugging Face model API reachability returned `net_ok`.
`scripts/sweep_clusters.py` emitted the ARC neural-guided-search / world-model
cluster URL and the ARC action-effect / exploration cluster URL.
`scripts/sweep_semscholar.py` was run on three focused queries and returned
HTTP 429, so no S2-only source was promoted. Low-concurrency WebSearch/WebFetch
plus direct arXiv HTTP checks verified the top eight papers listed below.
`/deep-research` was not invoked. No model load, training, leaderboard
submission, or solve claim was made; this is a no solve claim ingestion note.

**Verified source set:**
- arXiv:2402.12275 -- WorldCoder, a Model-Based LLM Agent: Building World Models by Writing Code and Interacting with the Environment
- arXiv:2503.23145 -- CodeARC: Benchmarking Reasoning Capabilities of LLM Agents for Inductive Program Synthesis
- arXiv:2511.02225 -- Learning Interactive World Model for Object-Centric Reinforcement Learning
- arXiv:2601.06604 -- Object-Centric World Models Meet Monte Carlo Tree Search
- arXiv:2605.05138 -- Executable World Models for ARC-AGI-3 in the Era of Coding Agents
- arXiv:2605.14937 -- Slot-MPC: Goal-Conditioned Model Predictive Control with Object-Centric Representations
- arXiv:2606.08775 -- Unifying Object-Centric World Models and Diffusion Policy: A Hierarchical Framework for Multi-Stage Robotic Tasks
- arXiv:2606.14418 -- Causal Object-Centric Models for Planning with Monte Carlo Tree Search

**SOTA -> .438 experiment mapping:**
- **Verifier-refined executable world-model induction** (arXiv:2605.05138, arXiv:2402.12275): takes over Takes over the .437 A1 structured engine slot: replace brittle free-form load_engine output with a typed executable model, a transition verifier, and refactoring toward simpler factors. Fails when: The prompt budget cannot afford repeated refactors, hidden mechanics need observations not yet taken, or the induced program overfits public prefixes without perception-grounded object/goal evidence.
- **Perception-grounded goal-conditioned object planning** (arXiv:2605.14937, arXiv:2606.08775): takes over Takes over the .437 A2 detector fix by turning detected pieces and goal sprites into a goal-conditioned planning objective with subgoal checks. Fails when: Slots drift across frames, ARC goals are non-spatial or hidden-state dependent, or differentiable/continuous MPC assumptions do not transfer to discrete click/key action spaces.
- **Causal object-centric MCTS action-slot planner** (arXiv:2606.14418, arXiv:2601.06604): takes over Takes over static candidate ranking by adding object-causal attention, slot-level transition predictions, and search over object interactions. Fails when: The object representation misses the controllable entity, action-slot binding aliases multiple mechanics, or MCTS rollouts compound an early world-model error before a live probe can correct it.
- **Interactive program-synthesis refinement over factor primitives** (arXiv:2503.23145, arXiv:2511.02225): takes over Takes over one-shot induction by converting errors into differential-test style counterexamples and by organizing object interactions into reusable factor primitives. Fails when: ARC action budgets make probes too expensive, the true rule lies outside the primitive vocabulary, or there is no free oracle analogous to CodeARC's hidden function query channel.

flagged_for_438: verifier_refined_executable_world_model_with_perception_grounded_goal_mpc (arXiv:2605.05138 + arXiv:2402.12275 + arXiv:2605.14937 + arXiv:2606.08775)

**Bottom line for .438:** build the verifier-refined executable world-model
loop first, but bind its goals to perception-grounded object/subgoal structure
instead of asking the free-form engine for another brittle terminal predicate.
The direct target is `E3AgentPolicy` + `arc_executable_world_model` +
`ProductWorldModel` with the structural-alignment goal pipeline supplying
goal-conditioned subgoals and failure diagnostics.
<!-- EXP4758-SOTA-INGESTION-END -->

## 2026-06-22 Skill-to-LoRA (arXiv:2606.16769) - operator-directed ingestion - INGESTED

**Status:** INGESTED (operator: "what can we learn"). S2L (Zhang & Qi, CUHK, 2026-06-15) distills a
procedural SKILL.md into a per-skill LoRA adapter (rank-16, ~6M params, 24MB) loaded at runtime instead of
injecting full skill text. SWE-Skills-Bench 21-skill subset: +5.2pp pass (65/210 vs 54/210), -6.6% tokens,
CNG 0.58 vs -0.18. Qwen3.6-27B base; code "upon acceptance." No EBM/verifier/world-model/active-inference/
ARC content.

**Verdict:** CITE / CALIBRATE, do NOT build. The method is trained PER-SKILL WEIGHTS — a public-game weights
prior that does NOT transfer to hidden games (the trap in [[feedback_arc_value_is_process_not_weights]]); S2L's
own scope limit ("not for open-ended reasoning / edge cases") confirms it. Reinforced by our own
full-FT>LoRA-for-OOD finding ([[reference_trm_tta_mcgovern]]) and the retired TRM-adapter line
([[feedback_trm_training_retired]]). Do NOT build per-skill LoRAs for the frozen ARC generator
([[project_arc_live_generator]]).

**What we keep (3 calibrations):** (1) SHARPENS the .425 energy-config-space bet — S2L and Carnot solve the
same skill-library + composition + ROUTING problem, but S2L picks LoRA-WEIGHTS as the skill representation
while we pick ENERGY ([[project_arc_energy_config_space]]); their "weights don't transfer OOD" limitation is
exactly why energy-as-online-refined-config-space is the right substrate for hidden games — the contrast
VALIDATES our choice. (2) Their reproduced finding "skill prompting is often neutral or harmful" corroborates
the CLAUDE.md "MECHANICALLY-ENFORCED — prose is reference-only" direction (distill stable rules into linters,
don't re-inject text); optional low-priority follow-up = measure whether prose-injected disciplines help the
planner or just cost latency (the .424 1201s planner timeout is a hint). (3) Their adapter-router ≈ our
`recommend_approach` survey-feature routing; they confirm weights-routing degrades OOD (Wrong-LoRA), our
affordance/energy routing is the right level. Peer/contrast, not a build direction.

## 2026-06-22 Energy-config-space generation for the ARC-AGI-3 wall - operator-directed ingestion - INGESTED

**Status:** INGESTED into `docs/research-notes/arc-generation-wall-energy-config-space-2026-06-22.md`
(operator: "work energy judgement into the live agent ... refine and embrace an energy config space within
each game and shared amongst the games"). Two adversarial research workflows over six higher-abstraction
families, repo-grounded against the refuted ledger + reusable assets.

**Verdict:** BUILD (cheapest-first), flagged for `.425` in `ops/known-issues.md` MANDATORY-NEXT-MILESTONE.
The reframe: the candidate-GENERATION wall (`winner_generated=1/25`) is `make-a-winner-appear`, not
`select-the-winner` (rerankers/routers/best-first REFUTED). Carnot's energy verifier earns its keep as a
generative DRIVER — a per-game ONLINE energy landscape + a SHARED cross-game energy prior guiding iterative
level attempts. Strongest cheap unlock: wire the BUILT-but-unwired `exp4020` `is_goal` (held-out precision
1.0) as a graded goal-ENERGY target (closes GAP-ARCH-GOAL-NOT-VERIFIED). Then macro-vocabulary horizon
collapse, then energy-as-FITNESS quality-diversity evolution (the non-AR generator). Real arXiv IDs:
2605.05138 (ARC-AGI-3 SOTA), 2605.28814, 2505.10819, 2005.05960, 2505.24784, 2009.08111, FunSearch
(Nature s41586-023-06924-6). Flagged for the `.425` planner per the SOTA-ingestion cycle.

## 2026-06-20 LoopWM (arXiv:2606.18208) - operator-directed ingestion - INGESTED

**Status:** INGESTED into `docs/research-notes/loopwm-2606.18208-ingestion-2026-06-20.md` (operator asked
"what can we learn"). Looped World Models = weight-tied recurrent-depth transformer (Universal-Transformer
+ PonderNet halt) applied to text-state world modeling; non-peer-reviewed tech report, no code, "100x" is
a 1B-vs->100B-API gap not a looped-vs-fixed ablation.

**Verdict:** CITE, do NOT build. The learned-simulator path is a STRUCTURAL non-starter for first-contact
ARC (offline-pretraining-required, zero-shot absent; Family-B's exact symbolic transition model is
strictly better). ONE sprint pickup -> `.417` candidate 6: a **verifier-grounded adaptive per-step budget**
for the explorer (really ACT/PonderNet; spend search compute only on hard frames -> attacks action
efficiency; zero new model/training). Phase-3 note-and-file: Carnot's energy verifier as the HALT signal
for a looped refiner ("loop until E<tau") -- the one original idea LoopWM implies. Conceptual reuse only;
does NOT revive the retired nano-TRM training.

## 2026-06-20 Exp 4477 - .413 SOTA ingestion ingested

**Status:** INGESTED into `results/experiment_4477_sota_ingestion_413.json` and
`docs/research-notes/sota-ingestion-413-2026-06-20.md`.

**Preconditions:** reliable channel reachable on CPU. The command
`.venv/bin/python scripts/sweep_clusters.py --help` succeeded; the arXiv
reachability check succeeded. `scripts/sweep_clusters.py` emitted focused
verifier and world-model cluster URLs. `scripts/sweep_semscholar.py` ran five
focused queries; Semantic Scholar returned six unique arXiv IDs and HTTP 429 on
two queries, so no S2-only non-arXiv source was promoted. Low-concurrency
WebSearch/WebFetch plus arXiv abs-page HTTP 200 checks verified
arXiv:2606.11521, arXiv:2605.27051, arXiv:2604.08792, arXiv:2307.03966,
arXiv:2604.02434, arXiv:2605.05138, arXiv:2606.12316, and arXiv:2512.24156.
The banned `/deep-research` channel was not invoked. No leaderboard submission,
live solve, or training run was launched.

**.413 outcome conditioning:** Exp 4467 banked dc22, Exp 4468 banked sc25 L2-L5,
Exp 4469 banked generic sc25 cast-grid L1, Exp 4470 banked sb26, and Exp 4474
kept the GAP-4 regression guard green. GAP-5 demo-underdetermination remains
the program-induction precision frontier for `.414`.

**Fresh-pass candidates marked ingested:** Counterexample Guided Learning
(arXiv:2606.11521), ConVer CEGAR-CEGIS verification (arXiv:2605.27051),
Choose, Don't Label program disambiguation (arXiv:2604.08792), PBE multi-intent
detection (arXiv:2307.03966), compositional neuro-symbolic consistency filtering
(arXiv:2604.02434), Executable World Models (arXiv:2605.05138), Loop-OWM
(arXiv:2606.12316), and graph-based ARC-AGI-3 exploration (arXiv:2512.24156).

flagged_for_v414: Socrates-style multiple-choice query synthesis for GAP-5 demo-underdetermination (arXiv:2604.08792)

random_seed=4477

**SOTA->experiment mapping note:** Build a GAP-5-aware tiered acceptance
harness: independent program induction plus cross-example consistency; when
programs agree on one target but diverge on sibling inputs, synthesize a
discriminating query and accept only if replayable executable evidence resolves
the ambiguity. Otherwise abstain.

## 2026-06-19 Exp 4464 - .412 SOTA ingestion ingested

**Status:** INGESTED into `results/experiment_4464_sota_ingestion_412.json` and
`docs/research-notes/sota-ingestion-412-2026-06-19.md`.

**Preconditions:** reliable channel reachable on CPU. The command
`scripts/sweep_clusters.py --help` succeeded; the arXiv reachability check
succeeded. `scripts/sweep_clusters.py` emitted focused verifier and world-model
cluster URLs. `scripts/sweep_semscholar.py` ran five focused queries; Semantic
Scholar returned HTTP 429 on all five queries, so no S2-only source was
promoted. Low-concurrency WebSearch/WebFetch plus arXiv abs-page HTTP 200 checks
verified arXiv:2309.16436, arXiv:2606.11521, arXiv:2507.14172,
arXiv:2411.17708, arXiv:2411.02272, arXiv:2605.05138, arXiv:2606.12316, and
arXiv:2603.13372. The banned `/deep-research` channel was not invoked.
No leaderboard submission, live solve, or training run was launched.

**Fresh-pass candidates marked ingested:** Counterexample-Guided Learning
(arXiv:2606.11521), SMT-checked CEGIS (arXiv:2309.16436), SOAR
(arXiv:2507.14172), neurally-guided program induction (arXiv:2411.17708),
induction+transduction routing (arXiv:2411.02272), Executable World Models
(arXiv:2605.05138), Loop-OWM (arXiv:2606.12316), and ARC living survey context
(arXiv:2603.13372).

flagged_for_v413: Counterexample-guided re-induction from rejecting execution states (arXiv:2606.11521; SMT-checked CEGIS predecessor arXiv:2309.16436)

random_seed=4464

**SOTA->experiment mapping note:** Add a counterexample-guided re-induction
loop to the generic solver: feed verifier-rejected execution states back into
dc22 config/toggle induction, tr87 glyph-rewrite induction, and sc25 phase-FSM
world-model induction, then count only reproduction-gated fixes.

## 2026-06-19 MANUFACTURED game variants — flagged for .413 (bigger generic-transfer benchmark)

**Status:** SHIPPED (outer-loop) + flagged for the .413 planner. `python/carnot/agentic/arc_variant_generator.py`
manufactures mechanic-preserving held-out layout variants of the 25 public games (deterministic
color-permutation -> positions unchanged, no action remap; optional reflection w/ click remap), and
`VariantEnv` + `arc_leaderboard_eval.py --variant N`/`--reflect` run the full CarnotAgent cascade
against them. The REAL env keeps the win logic, so a solve is a real solve and a color-permuted variant
forces the LLM inducer to RE-induce the win-rule in a new palette = a genuine generalization test.
Validated: the explorer solves variant-1 lp85 to L1 in 21 actions (= the real game). **flagged_for_v413
(operator 2026-06-19):** wire the variant set into the LOO/generic-transfer benchmark — score the
generic solver on 25 games x N variants (not 2/7 LOO on 25), and TRAIN the generic operators +
example-conditioned inducer against variant diversity. Solvability + gold solution are GUARANTEED per
variant (inherited from the original game, judged by the real win-condition) so generated solutions are
objectively gradeable. Just add `--variant`/`--reflect` to the next LOO benchmark task.

## 2026-06-19 Exp 4452 - .411 SOTA ingestion ingested

**Status:** INGESTED into `results/experiment_4452_sota_ingestion_411.json` and
`docs/research-notes/sota-ingestion-411-2026-06-19.md`.

**Preconditions:** reliable channel reachable on CPU. The command
`scripts/sweep_clusters.py --help` succeeded; the arXiv reachability check
succeeded. `scripts/sweep_clusters.py` emitted focused cluster URLs.
`scripts/sweep_semscholar.py` ran five focused queries; Semantic Scholar
returned HTTP 429 on four queries and surfaced CodeARC from the
counterexample-guided query. Low-concurrency WebSearch/WebFetch plus arXiv
abs-page HTTP 200 checks verified arXiv:2310.19791, arXiv:2006.08381,
arXiv:2211.16605, arXiv:2405.15880, arXiv:2503.23145, arXiv:2605.05138,
arXiv:2606.12316, and arXiv:2603.05099. The banned `/deep-research` channel was
not invoked. No leaderboard submission, live solve, or training run was
launched.

**Fresh-pass candidates marked ingested:** LILO (arXiv:2310.19791), DreamCoder
(arXiv:2006.08381), Stitch (arXiv:2211.16605), HYSYNTH (arXiv:2405.15880),
CodeARC (arXiv:2503.23145), Executable World Models (arXiv:2605.05138),
Loop-OWM (arXiv:2606.12316), and ARC-TGI (arXiv:2603.05099).

flagged_for_v412: LILO-style documented library induction over the ARC solver corpus (arXiv:2310.19791)

random_seed=4452

**SOTA->experiment mapping note:** Build a documented primitive-library
induction pass over solved predicates, executable world models, and primitive
ledger rows; retrieve those primitives during first-contact solving; and count
only held-out, reproduction-gated improvements.

## 2026-06-19 RecursiveMAS (arXiv:2604.25917) — INGESTED as CORROBORATION; decomposition parked

**Status:** INGESTED → `docs/research-notes/recursivemas-corroboration-2026-06-19.md`. Operator-handed;
logged as corroboration (weaker fit — multi-agent LLM latent-recursion training technique, not a
drop-in for our execution-grounded offline ARC agent). **Corroborates** (3rd datapoint): an explicit
Critic/verifier in the loop + iterative refinement-on-mismatch improves accuracy = our
propose→verify→refactor. **Does NOT transfer to the live agent:** latent collaboration bypasses text
decoding, incompatible with our EXECUTION-grounded verifier (must run the code each round); multi-LLM
infeasible on one offline 16GB P100 in the 8h budget. **flagged (PARKED, not an ARC-sprint task):** the
Planner→Critic→Solver DECOMPOSITION idea → for the HIERARCHICAL-PLANNING track (the vc33/L4 wall;
LeCun names hierarchical planning THE open problem) — DEV-side: frontier-model Planner decomposes a
deep-tail game into subgoals, verifier grounds each, bank the decomposition as a richer corpus entry.
No experiment staged.

## 2026-06-19 VibeThinker-3B proposer candidate (HF WeiboAI/VibeThinker-3B) — INGESTED, flagged for .412

**Status:** INGESTED → `docs/research-notes/vibethinker-3b-proposer-candidate-2026-06-19.md`.
Operator-handed model. 3B, **MIT**, Qwen2.5-Coder base; IMO-AnswerBench 76.4–80.6% (rivals 671B–1T),
96.1% LeetCode — a 3B near-frontier on VERIFIABLE math/code (3rd small-model-rivals-frontier datapoint
after FinAcumen). Candidate **proposer swap** for the ARC live generator: 1.93 GB Q4 (vs Qwen3.5-9B's
5.9 GB) → more 16 GB-Kaggle KV headroom; its "not for autonomous agents" limit is exactly the job our
harness-as-agent design removes from the LLM. GGUF ready (`oussaber/VibeThinker-3B-Q4_K_M-GGUF`).
**flagged_for_v412 (STRONGEST):** head-to-head proposer benchmark VibeThinker-3B (thinking) vs
Qwen3.5-9B-MTP (`/no_think`) on the SAME E3 induce/refactor step — score **grounding rate +
events-to-solve + WALL-TIME** (long-CoT slowness is the real risk vs the live time budget), not raw
accuracy. Swap only if it grounds ≥ as well at ≤ the wall-time. Verifier/harness unchanged (LLM is
swappable; the verifier is the moat).

## 2026-06-19 FinAcumen experience-memory ingestion (arXiv:2606.17642) — INGESTED, flagged for .411

**Status:** INGESTED → `docs/research-notes/finacumen-experience-memory-ingestion-2026-06-19.md`.
Operator-handed paper. Structural mirror of our ARC live-solver (frozen small model + self-evolving
experience memory + thresholded retrieval + verifier gate + fallback); a frozen 8B + memory rivals
GPT-4o/72B (+41 pts) — empirical support that corpus/retrieval quality, not model size, is the lever.
**Implemented this ingestion (committed):** confidence-threshold + failure-`cautions` in
`arc_solve_learning.recommend_approach` (FinAcumen "irrelevant retrieval DEGRADES" → don't blind-copy a
low-confidence recipe on unseen games). **flagged_for_v411 (STRONGEST):** wire `confident_transfer` +
`cautions` into the runtime induction prompt (`arc_executable_world_model` induce/refactor) —
precision-over-recall transfer on held-out games; calibrate the threshold against the .410 LOO
benchmark. Plus: dedup/rank/cap few-shot (k_max=5); systematic corpus distillation (Findings+Cautions).

## 2026-06-19 Exp 4440 - .410 example-corpus SOTA ingestion ingested

**Status:** INGESTED into `results/experiment_4440_sota_ingestion_410.json` and
`docs/research-notes/sota-ingestion-410-2026-06-19.md`.

**Preconditions:** reliable channel reachable on CPU. `scripts/sweep_clusters.py`
emitted focused arXiv cluster URLs for world-model and verifier/program
literature. `scripts/sweep_semscholar.py` ran focused ARC/program/library
queries and returned HTTP 429; no S2-only source was promoted. Low-concurrency
WebSearch/WebFetch plus arXiv API / arXiv abs-page HTTP 200 checks verified
arXiv:2310.19791, arXiv:2006.08381, arXiv:2211.16605, arXiv:2405.15880,
arXiv:2503.23145, arXiv:2605.05138, arXiv:2606.12316, and arXiv:2603.05099.
The banned `/deep-research` channel was not invoked. No leaderboard submission
or training run was launched.

**.410 outcome conditioning:** Exp 4432 solved 2/7 leave-one-out targets; Exp
4433 reproduced `g50t` L1 from example-conditioned win induction; Exp 4434
lifted example-conditioned world-model accuracy from 0.714286 to 1.0 but added
zero reproduced levels; Exp 4435 left `dc22` as an open verifier gap; Exp 4436
deepened `tu93` to L5 and consolidated solver primitives.

**Fresh-pass candidates marked ingested:** LILO (arXiv:2310.19791), DreamCoder
(arXiv:2006.08381), Stitch (arXiv:2211.16605), HYSYNTH (arXiv:2405.15880),
CodeARC (arXiv:2503.23145), Executable World Models (arXiv:2605.05138),
Loop-OWM (arXiv:2606.12316), and ARC-TGI (arXiv:2603.05099).

flagged_for_v411: LILO-style documented library induction over the ARC solver/example corpus (arXiv:2310.19791)

random_seed=4440

**SOTA->experiment mapping note:** Build a documented primitive-library induction
pass over solved predicates, executable world models, and primitive ledger rows;
retrieve those primitives during first-contact solving; and count only
held-out, reproduction-gated improvements.

## 2026-06-19 Exp 4429 - .409 ARC headline SOTA ingestion ingested

**Status:** INGESTED into `results/experiment_4429_sota_ingestion_409.json`.

**Preconditions:** reliable channel reachable on CPU. `scripts/sweep_clusters.py`
emitted focused arXiv cluster URLs for verifier/reward and world-model
literature. `scripts/sweep_semscholar.py` ran focused ARC/program/world-model
queries, surfaced arXiv:2605.05138, and one verifier-grounded focused query
returned HTTP 429; no S2-only source was promoted. Low-concurrency
WebSearch/WebFetch plus arXiv API / arXiv abs-page HTTP 200 checks verified
arXiv:2605.05138, arXiv:2605.05485, arXiv:2503.23145, arXiv:2512.22336, and
arXiv:2605.25931. The banned `/deep-research` channel was not invoked. TRM
training stood down. CPU substrate only: literature ingestion, not model
execution.

**.409 outcome conditioning:**
- Exp 4421: one config-rule level counted after reproduction, but the source
  artifact is adversarial-stamped and `verifier_is_oracle=true`.
- Exp 4423: `partial: generic_first_contact_g50t_routed_missing_verifier_gap_logged`,
  `offline_reproduced=false`, and `reproduced_levels=0`.
- Exp 4424: mechanic/lookahead repair improved tests for sc25 but
  `new_levels_reproduced=0` and `offline_reproduced=false`.
- Exp 4425: `config_rule_vocabulary_transfers=false`; no self-learning transfer
  lift was proven.
- Exp 4426: CPU registry audit reported all counted entries reproduced and
  recorded the .409 reproduction-gate rows.

**Fresh-pass candidates marked ingested:**
- Executable World Models for ARC-AGI-3, arXiv:2605.05138 - mapped to a fresh
  generic first-contact coding-agent harness that builds, verifies, and searches
  executable world models across unseen games.
- ReaComp compiled symbolic solver induction, arXiv:2605.05485 - mapped to
  verifier-grounded win-rule induction compiled into reusable zero-token DSL
  solvers.
- CodeARC inductive program synthesis, arXiv:2503.23145 - mapped to
  counterexample-led program induction for g50t and sc25 residual verifier gaps.
- Agent2World adaptive symbolic world-model feedback, arXiv:2512.22336 - mapped
  to adaptive behavior tests that repair induced world models before search.
- Explore Before You Solve, arXiv:2605.25931 - mapped to speed-depth budget
  control for unseen-game transfer.

flagged_for_v410: Executable ARC-AGI-3 world-model agent with verifier-grounded planning (arXiv:2605.05138)

random_seed=4429

**SOTA->experiment mapping note:** The .410 headline should combine executable
world-model induction with verifier-grounded search. Start from arXiv:2605.05138
as the main harness, compile stable win rules via arXiv:2605.05485, use
arXiv:2503.23145 for counterexample-led program refinement, use arXiv:2512.22336
to stress-test induced transition models, and use arXiv:2605.25931 to allocate
first-contact exploration before solve attempts.

## 2026-06-19 Exp 4420 - .408 fork SOTA ingestion ingested

**Status:** INGESTED into `results/experiment_4420_sota_ingestion_v409.json`.

**Preconditions:** reliable channel reachable. `scripts/sweep_clusters.py`
imported and emitted focused arXiv cluster URLs. `scripts/sweep_semscholar.py`
imported and was run on focused adaptive world-model repair queries; Semantic
Scholar returned HTTP 429, so no S2-only source was promoted. Low-concurrency
WebSearch/WebFetch plus arXiv API / arXiv abs-page HTTP 200 checks verified
arXiv:2605.05485, arXiv:2503.23145, arXiv:2605.22446, arXiv:2605.09502,
arXiv:2605.12201, arXiv:2606.13565, arXiv:2502.01384, and arXiv:2508.02298.
The banned `/deep-research` channel was not invoked. TRM training stood down.

**Filtered track:** .408 outcomes after config-rule induction, Agent2World
adaptive E3 repair, hidden-state first-error localization, GAP-4 local
sovereign generator gating, config-rule vocabulary transfer, and SteerConf code
detection calibration repair.

**.408 outcome conditioning:**
- Exp 4414: `complete_config_rule_partial`, `new_levels_reproduced=0`,
  `reproducible_total_levels=34`, and `verifier_is_oracle=true`. Config-rule
  induction found a grounded rule but not a new reproduced level.
- Exp 4415: `complete_e3_adaptive_partial`, `new_levels_reproduced=0`,
  `reproducible_total_levels=34`, and `verifier_is_oracle=true`. Agent2World
  adaptive behavior tests exposed mechanics but did not deepen E3.
- Exp 4416: `complete: clean_powered_null_position_only_not_beaten`,
  `hidden_state_localizer_has_nonposition_signal=false`,
  `position_only_baseline_f1=1.0`, and `delta_ci95=[0.0, 0.0]`. Hidden-state
  localization remains diagnostic, not actionable.
- Exp 4417: `sovereign_gap4_gate_holds=true`, `local_generator_coverage=0.2333`,
  `graded_gate_fires=0`, and `delta_ci95=[0.0, 0.0]`. Sovereign local
  generation remains viable but flat under the current gate.
- Exp 4418: `blocked_local_model_unavailable` and
  `config_rule_vocabulary_transfers=false`. Do not plan another local-model-only
  vocabulary transfer until the local inducer exists or the method avoids
  test-time LLM calls.
- Exp 4419: `complete: clean_null_steered_confidence_does_not_rescue_code_detector`,
  `detection_calibrated_multi_domain=false`, `domains_at_chance=[code_humaneval]`,
  `positive_control_passed=true`, and `verifier_is_oracle=false`. SteerConf did
  not rescue the code detector.

**Fresh-pass candidates marked ingested:**
- ReaComp compiled symbolic solver induction, arXiv:2605.05485 - mapped to
  reusable constrained-DSL solver induction from E3/config traces; strongest
  .409 hand-off because it avoids the blocked local-model and zero-test-time
  sovereignty bottleneck.
- CodeARC interactive differential-query program induction, arXiv:2503.23145 -
  mapped to verifier-returned counterexample queries for config-rule and GAP-4
  program induction.
- Pre-VLA preemptive runtime verification, arXiv:2605.22446 - mapped to
  verify-before-rollout filtering and resampling for E3 action chunks after
  Exp 4415 yielded zero new levels.
- Hidden Error Awareness, arXiv:2605.09502 - mapped to a diagnostic-only
  hidden-state audit after Exp 4416 tied the position-only baseline.
- RisCoSet, arXiv:2605.12201 - mapped to risk-controlling prediction sets for
  code_humaneval after SteerConf left code at chance.

out_of_band_flagged:
- A2D2 adaptive any-length discrete diffusion, arXiv:2606.13565 -
  operator-owned verifier-as-reward generator training, NOT auto-run in-loop.
- SEPO score-entropy policy optimization, arXiv:2502.01384 - operator-owned
  verifier-as-reward generator training, NOT auto-run in-loop.
- Full CAPO policy optimization, arXiv:2508.02298 - operator-owned generator
  training; only offline critique-label diagnostics are in-band.

flagged_for_v409: ReaComp compiled symbolic solver induction (arXiv:2605.05485)

Flagged for .409: `ReaComp compiled symbolic solver induction (arXiv:2605.05485)`

random_seed=4420

**Bottom line for the .409 roadmap:** ReaComp is the single strongest method:
compile existing verifier-checked traces into reusable symbolic solvers, then
use CodeARC-style counterexample queries to widen rule coverage. Keep
Pre-VLA-style preemptive filtering as the E3 repair support track, treat
hidden-state awareness as diagnostic only, and rebuild code_humaneval detection
around risk-controlled prediction sets rather than another scalar confidence
calibrator.

## 2026-06-18 Exp 4409 - .407 fork SOTA ingestion ingested

**Status:** INGESTED into `results/experiment_4409_sota_ingestion_v408.json`.

**Preconditions:** reliable channel reachable via arXiv API and arXiv/WebFetch
HTTP 200 checks. `scripts/sweep_clusters.py` emitted the focused arXiv cluster
URLs. `scripts/sweep_semscholar.py` imported and was run on focused real
first-error, active PRM, calibration, and ARC world-model queries, but Semantic
Scholar returned HTTP 429 for those focused queries, so no S2-only result was
promoted. Low-concurrency WebSearch/WebFetch plus arXiv page checks verified
arXiv:2512.22336, arXiv:2605.13772, arXiv:2503.02863, arXiv:2605.25931,
arXiv:2508.02298, arXiv:2606.13565, and arXiv:2502.01384. The banned
`/deep-research` channel was not invoked.

**Filtered track:** .407 outcomes after real-intervention first-error
localizer deconfounding, typed-taxonomy localizer audit, active-learning
self-learning, cross-domain detector calibration repair, and ARC E3
per-mechanic executable unit tests.

**.407 outcome conditioning:**
- Exp 4403: `complete: clean_powered_null_position_only_not_beaten`,
  `localizer_genuinely_beats_position_only=false`, FoVer
  `position_only_baseline=1.0`, GAP-4 ARC `delta_ci95=[-0.134615, 0.173077]`,
  and `verifier_is_oracle=false`. The real-intervention text localizer is a
  powered position-only null, not a .408 headline.
- Exp 4404: `blocked_gate_check_failed` because
  `localizer_genuinely_beats_position_only actual=False expected=True`. The
  typed-taxonomy cross-domain localizer stays gated.
- Exp 4407: `complete: clean_null_position_bound_or_saturated`,
  `localizer_compounds=false`, `compounding_delta_ci95=[0.0, 0.0]`,
  `positive_control_headroom=false`, and `verifier_is_oracle=false`. Active
  selection over the same position-bound localizer did not compound.
- Exp 4408: `complete: calibrated_multi_domain_contract_false_deconfounded`,
  `detection_calibrated_multi_domain=false`,
  `domains_at_chance=[code_humaneval]`, `positive_control_passed=true`, and
  `verifier_is_oracle=false`. Detection calibration remains alive as a repair
  track, but the deployable multi-domain contract is false.
- Exp 4405/4406: `complete_e3_deeper_partial` and
  `complete_e3_ar25_ka59_ft09_partial`, both with `new_levels_reproduced=0`,
  `reproducible_total_levels=34`, and `verifier_is_oracle=true`. Static
  per-mechanic tests did not deepen ARC E3.

**Fresh-pass candidates marked ingested:**
- Agent2World adaptive testing, arXiv:2512.22336 - mapped to behavior-aware
  E3 mechanic repair after static unit tests found blockers but yielded zero
  new levels.
- GeoReason hidden-state transport, arXiv:2605.13772 - mapped to a diagnostic
  first-error audit after the text localizer tied the position-only baseline.
- SteerConf confidence elicitation, arXiv:2503.02863 - mapped to
  domain-calibration repair after Exp 4408 left code_humaneval at chance.
- AERA explore-verify-plan, arXiv:2605.25931 - mapped to speed-depth and
  public-artifact controls around any renewed ARC E3 progress claim.
- CAPO generative credit assignment, arXiv:2508.02298 - mapped only to offline
  critique-label diagnostics after active localizer selection did not compound;
  generator policy optimization is not auto-run in-loop.

out_of_band_flagged:
- A2D2 adaptive any-length discrete diffusion, arXiv:2606.13565 -
  operator-owned verifier-as-reward generator training, NOT auto-run in-loop.
- SEPO score-entropy policy optimization, arXiv:2502.01384 - operator-owned
  verifier-as-reward generator training, NOT auto-run in-loop.

flagged_for_v408: agent2world_adaptive_e3_mechanic_repair_v408

Flagged for .408: `agent2world_adaptive_e3_mechanic_repair_v408`

random_seed=4409

**Bottom line for the .408 roadmap:** do not repeat the position-bound
localizer, the gated typed-taxonomy branch, or active selection over the same
failed signal. The single strongest .408 method is Agent2World-style
behavior-aware adaptive testing for ARC E3 mechanic repair, with AERA
speed-depth controls. GeoReason and CAPO are diagnostics for deciding whether
any non-position first-error signal remains; SteerConf is the calibration
repair support track. A2D2 and SEPO stay out of band for operator-owned
verifier-as-reward generator training.

## 2026-06-18 Exp 4398 - .406 fork SOTA ingestion ingested

**Status:** INGESTED into `results/experiment_4398_sota_ingestion_v407.json`.

**Preconditions:** network precondition passed via Hugging Face reachability and
arXiv/WebFetch verification. If that check had failed, the only honest artifact
would have been `honest_verdict=blocked_network_unavailable`.

**Reliable-channel provenance:** `scripts/sweep_clusters.py` emitted focused
verifier/process-reward and world-model arXiv discovery URLs. `scripts/sweep_semscholar.py`
was run on focused verifiable-process-data, OOD verifier calibration, selective
prediction, and ARC E3 lookahead queries; Semantic Scholar returned arXiv:2504.00891,
arXiv:2502.11520, arXiv:2603.19310, arXiv:2409.13757, arXiv:2602.03412,
arXiv:2508.04748, arXiv:2407.05693, and arXiv:2606.08728 before HTTP 429 on
the remaining focused queries. Low-concurrency WebSearch/WebFetch plus arXiv
abs/html checks verified arXiv:2601.14209, arXiv:2603.25412, arXiv:2504.10559,
arXiv:2602.07842, arXiv:2606.16070, arXiv:2605.02395, arXiv:2605.25133,
arXiv:2606.13565, and arXiv:2502.01384. The banned `/deep-research` channel was
not invoked.

**Filtered track:** .406 outcomes after verifiable-process-data first-error
localization, localizer skeptic-proofing, localizer self-learning, cross-domain
detector calibration, and ARC E3 lookahead/mechanic-gap work.

**.406 outcome conditioning:**
- Exp 4392: `success: synthetic_process_localizer_beats_ensemble_baseline`,
  `localizer_beats_ensemble_baseline=true`, FoVer `synthetic_trained_localizer=1.0`,
  GAP-4 ARC `synthetic_trained_localizer=0.692308`, and `verifier_is_oracle=false`.
  The process-data localizer is the live vehicle, but not yet the trusted headline.
- Exp 4393: `complete: a1_win_quarantined_as_artifact_confounded`,
  `localizer_win_is_genuine=false`, `beats_position_only_baseline=false`, and
  `template_ablation_drop=0.0`. The A1 win is quarantined until real,
  position-diverse first-error evidence deconfounds it.
- Exp 4396: `complete: clean_saturated_null_localizer`,
  `localizer_compounds=false`, `compounding_delta_ci95=[0.0, 0.0]`, and
  `positive_control_passed=true`. Simple corpus growth saturated; .407 needs
  active/uncertainty selection rather than more of the same stream.
- Exp 4397: `complete: calibrated_multi_domain_contract_false`,
  `detection_calibrated_multi_domain=false`, `domains_at_chance=[]`, non-FoVer
  domains above chance (`gap4_arc`, `gsm8k`, `code_humaneval`), and
  `verifier_is_oracle=false`. Detection remains alive, but calibration/base-rate
  repair is required before a deployable detector contract.
- Exp 4394/4395: `complete_e3_deeper_partial` and
  `complete_e3_ar25_ka59_ft09_partial`, both with `new_levels_reproduced=0`,
  `reproducible_total_levels=34`, and `verifier_is_oracle=true`. ARC E3 remains
  the north star, but .406 produced mechanic-gap work rather than new solves.

**Fresh-pass candidates marked ingested:**
- InT Self-Proposed Interventions, arXiv:2601.14209 - mapped to real
  first-error intervention traces that deconfound the A1 localizer.
- Reasoning Safety Monitor, arXiv:2603.25412 - mapped to typed step-localizer
  audit labels and adversarial first-error taxonomy checks.
- ActPRM active learning, arXiv:2504.10559 - mapped to active uncertainty and
  first-error-position diversity sampling after the saturated self-learning null.
- Semantic Confidence Aggregation / MACE, arXiv:2602.07842 - mapped to
  multi-answer/base-rate calibration repair after the false calibrated
  multi-domain contract.
- Mind-Studio executable world models with lookahead evaluation,
  arXiv:2606.16070 - carried as E3 mechanic-gap tests after zero new .406
  reproduced levels.

Carried baseline context: Controllable and Verifiable Process Data Synthesis,
arXiv:2605.02395, and Prover-Verifier Deliberation, arXiv:2605.25133, remain
verified supports, but the .406 outcomes make real intervention/deconfounding
the single .407 flag.

out_of_band_flagged:
- A2D2 adaptive any-length discrete diffusion, arXiv:2606.13565 -
  operator-owned verifier-as-reward generator training, NOT auto-run in-loop.
- SEPO score-entropy policy optimization, arXiv:2502.01384 - operator-owned
  verifier-as-reward generator training, NOT auto-run in-loop.

flagged_for_v407: intervention_active_real_first_error_deconfounding_v407

Flagged for .407: `intervention_active_real_first_error_deconfounding_v407`

random_seed=4398

**Bottom line for the .407 roadmap:** build on the process-data localizer only
after deconfounding it. The single strongest .407 method is InT-style real
first-error intervention evidence, actively selected for position/template
diversity and audited by typed reasoning-monitor labels. Calibration repair and
E3 mechanic-gap tests stay live supporting tracks. A2D2 and SEPO stay out of
band for operator-owned verifier-as-reward generator training.

## 2026-06-18 Exp 4387 - .405 fork SOTA ingestion ingested

**Status:** INGESTED into `results/experiment_4387_sota_ingestion_v406.json`.

**Preconditions:** network precondition passed via Hugging Face reachability and
arXiv/WebFetch verification. If that check had failed, the only honest artifact
would have been `honest_verdict=blocked_network_unavailable`.

**Reliable-channel provenance:** `scripts/sweep_clusters.py` emitted focused
verifier/process-reward and world-model arXiv discovery URLs. `scripts/sweep_semscholar.py`
was run on focused bidirectional-PRM, selective-prediction, cross-domain verifier,
and ARC E3 lookahead queries; it returned arXiv:2603.16253, arXiv:2605.02395,
arXiv:2601.18984, arXiv:2603.02119, arXiv:2506.11474, arXiv:2601.14209, and
arXiv:2603.25412 before Semantic Scholar rate-limited the remaining focused
queries with HTTP 429. Low-concurrency WebSearch/WebFetch plus arXiv abs/html
checks verified arXiv:2605.02395, arXiv:2102.10395, arXiv:2605.25133,
arXiv:2504.16828, arXiv:2606.16070, arXiv:2508.01682, arXiv:2605.05138,
arXiv:2606.13565, and arXiv:2502.01384. The banned `/deep-research` channel was
not invoked.

**Filtered track:** .405 outcomes after BiPRM detector localization/abstention,
skeptic-proof gating, detector self-learning compounding, cross-domain detector
generalization, and ARC E3 lookahead/mechanic-gap work.

**.405 outcome conditioning:**
- Exp 4381: `complete: clean_powered_null_bidirectional_not_actionable`,
  `detector_localization_actionable=false`, `localization_delta_ci95=[0.0, 0.0]`,
  `useful_operating_point=null`, and `verifier_is_oracle=false`. BiPRM-style
  bidirectional fusion is a baseline/null for this corpus, not the .406 headline.
- Exp 4382: `blocked_gate_check_failed` because
  `detector_localization_actionable=false`. The skeptic-proof phase remains
  gated off until localization becomes actionable.
- Exp 4385: `success: detector_compounds_heldout_localization_f1`,
  `detector_compounds=true`, `compounding_delta_ci95=[0.003396, 0.032772]`,
  and `verifier_is_oracle=false`. Detector self-learning is the clean .405
  oracle-distinct positive.
- Exp 4386: `success: detector_generalizes_cross_domain_non_fover`,
  `detector_generalizes_cross_domain=true`, GAP-4 ARC `detection_auroc=0.963317`,
  `auroc_ci95=[0.922285, 0.990662]`, `selection_headroom=0.129`,
  `n=28443`, and `verifier_is_oracle=false`. Detection generalizes beyond
  FoVer and has real selection headroom, but code/GSM remain unavailable-domain
  gaps.
- Exp 4383/4384: `complete_e3_deeper_partial` and
  `complete_e3_ar25_ka59_ft09_partial`, both with `new_levels_reproduced=0`,
  `reproducible_total_levels=34`, and `verifier_is_oracle=true`. ARC E3 remains
  the north star, but .405 yielded mechanic-gap repair work rather than new
  solves.

**Fresh-pass candidates marked ingested:**
- Controllable and Verifiable Process Data Synthesis for PRMs, arXiv:2605.02395
  - mapped to .406 verifiable first-error data for cross-domain localization.
- On Calibration and Out-of-domain Generalization, arXiv:2102.10395 - mapped to
  multi-domain detector calibration after the GAP-4 ARC generalization win.
- Trust but Verify: Prover-Verifier Deliberation for Selective LLM Prediction,
  arXiv:2605.25133 - mapped to a structured report/abstain layer if abstention
  is retried after the raw threshold null.
- ThinkPRM, arXiv:2504.16828 - mapped to bounded explanation labels for the
  untyped first-error gap, gated by executable or symbolic checks.
- Mind-Studio executable world models with lookahead evaluation,
  arXiv:2606.16070 - carried as E3 mechanic-gap repair after zero new .405
  reproduced levels.

Carried baseline context: BiPRM, arXiv:2508.01682, and Executable World Models
for ARC-AGI-3, arXiv:2605.05138, remain verified context, but the .405 outcomes
make them baseline/north-star supports rather than the single .406 flag.

out_of_band_flagged:
- A2D2 adaptive any-length discrete diffusion, arXiv:2606.13565 -
  operator-owned verifier-as-reward generator training, NOT auto-run in-loop.
- SEPO score-entropy policy optimization, arXiv:2502.01384 - operator-owned
  verifier-as-reward generator training, NOT auto-run in-loop.

flagged_for_v406: verifiable_process_data_cross_domain_localization_v406

Flagged for .406: `verifiable_process_data_cross_domain_localization_v406`

random_seed=4387

**Bottom line for the .406 roadmap:** do not re-headline BiPRM fusion after the
clean actionable-localization null, and do not unlock skeptic-proofing until a
localizer exists. The live .405 signal is detector self-learning plus GAP-4 ARC
cross-domain detection. The .406 flag should therefore add verifiable
process-supervision data for first-error localization across domains, then use
multi-domain calibration and structured abstention only after the localizer is
real. A2D2 and SEPO stay out of band for operator-owned verifier-as-reward
generator training.

## 2026-06-18 Exp 4376 - .404 fork SOTA ingestion ingested

**Status:** INGESTED into `results/experiment_4376_sota_ingestion_v405.json`.

**Preconditions:** network precondition passed via Hugging Face reachability and
arXiv/WebFetch verification. If that check had failed, the only honest artifact
would have been `honest_verdict=blocked_network_unavailable`.

**Reliable-channel provenance:** `scripts/sweep_clusters.py` emitted focused
world-model and verifier/process-reward arXiv discovery URLs and the arXiv API
returned fresh June 2026 IDs. The relevant fresh cluster hit was
arXiv:2606.16070 (Mind-Studio). `scripts/sweep_semscholar.py` was run on focused
LLM-heuristic, diffusion-search, and step-error detector queries; Semantic
Scholar returned HTTP 429, so no S2-only result was promoted. Low-concurrency
WebSearch/WebFetch plus arXiv page checks verified arXiv:2508.01682,
arXiv:2606.16070, arXiv:2605.05138, arXiv:2503.18809, arXiv:2603.20216,
arXiv:2606.13565, and arXiv:2502.01384. Supporting detector benchmark context
was checked against arXiv:2412.06559 and ThinkPRM against arXiv:2504.16828. The
banned `/deep-research` channel was not invoked.

**Filtered track:** .404 outcomes after LLM-generated/code heuristics for
planning, E3 executable-world-model ARC progression, DiffusionGemma
repair-or-retire, and verifier-as-detector step-error measurement.

**.404 outcome conditioning:**
- Exp 4370: `complete: clean_powered_null_linear_not_beaten`,
  `acceptance_gate_passed=true`, `llm_heuristic_beats_linear=false`,
  `held_out_actions_equal=true`, and `verifier_is_oracle=false`. The stronger
  generated-heuristic function class is a clean null on the reproduced corpus,
  not the .405 headline.
- Exp 4372: `success_e3_deeper_lp85_reproduced`,
  `new_levels_reproduced=1`, `reproducible_total_levels=34`, and
  `verifier_is_oracle=true`. E3 remains the ARC north star, but still
  oracle-grounded.
- Exp 4374: `retired_in_generation_conversion_unmeasurable`,
  `scorer_requalified_leak_clean=false`, `codila_control_differentiates=false`,
  `benchmark_n=0`, and `s3_guided_beats_control=false`. DiffusionGemma
  in-generation conversion stays retired from the autonomous in-loop headline.
- Exp 4375: `complete: detector_beats_chance_zero_selection_headroom_fover`,
  `detector_auroc=0.918304`, `detector_beats_chance=true`,
  `selection_headroom.headroom=0.0`, `n_candidates=8829`, and
  `verifier_is_oracle=false`. This is the strongest non-oracle positive .404
  signal.

**Fresh-pass candidates marked ingested:**
- Bidirectional Process Reward Model, arXiv:2508.01682 - mapped to the .405
  detector-first follow-up: bidirectional step-error localization plus
  risk-coverage on cached FoVer and ARC/E3 traces.
- Mind-Studio executable world models with lookahead evaluation,
  arXiv:2606.16070 - mapped to the E3 continuation with entropy-selected traces,
  lightweight skill files, and K-step rollout-fidelity checks.
- Executable World Models for ARC-AGI-3, arXiv:2605.05138 - carried forward as
  the ARC E3 baseline after the lp85 level advance.
- Classical Planning with LLM-Generated Heuristics, arXiv:2503.18809 - marked as
  a clean-null control after Exp 4370 rather than a repeated .405 headline.
- CoDiLA locally coherent parallel decoding, arXiv:2603.20216 - retained only as
  a DiffusionGemma diagnostic/control once scorer and local-control preconditions
  are repaired.

out_of_band_flagged:
- A2D2 adaptive any-length discrete diffusion, arXiv:2606.13565 -
  operator-owned verifier-as-reward generator training, NOT auto-run in-loop.
- SEPO score-entropy policy optimization, arXiv:2502.01384 - operator-owned
  verifier-as-reward generator training, NOT auto-run in-loop.

flagged_for_v405: biprm_processbench_detector_localization_v405

Flagged for .405: `biprm_processbench_detector_localization_v405`

random_seed=4376

**Bottom line for the .405 roadmap:** do not re-run the LLM-generated heuristic
arm unchanged, and do not revive DiffusionGemma in-loop while both scorer and
CoDiLA gates failed. Continue E3 with Mind-Studio-style lookahead fidelity, but
put the single .405 flag on the detector-first BiPRM/ProcessBench-style
step-error localization and abstention path because Exp 4375 produced the clean
non-oracle positive signal. A2D2 and SEPO stay out of band for operator-owned
verifier-as-reward generator training.

## 2026-06-18 Exp 4365 - .403 fork SOTA ingestion ingested

**Status:** INGESTED into `results/experiment_4365_sota_ingestion_v404.json`.

**Preconditions:** network precondition passed via Hugging Face reachability and
arXiv/WebFetch verification. If that check had failed, the only honest artifact
would have been `honest_verdict=blocked_network_unavailable`.

**Reliable-channel provenance:** `scripts/sweep_clusters.py` emitted focused
energy/reward and world-model arXiv discovery URLs; direct arXiv fetch of those
helper URLs returned HTTP 400 in this pass, so no cluster rows were promoted
without independent verification. `scripts/sweep_semscholar.py` was run on five
focused query strings; it returned HTTP 429 for three queries and surfaced
arXiv:2606.08501 plus arXiv:2503.18809 among usable candidates. Low-concurrency
WebSearch/WebFetch plus arXiv page checks verified arXiv:2503.18809,
arXiv:2605.05138, arXiv:2603.20216, arXiv:2606.08501, arXiv:2602.01842,
arXiv:2606.13565, and arXiv:2502.01384. The banned `/deep-research` channel was
not invoked.

**Filtered track:** .403 outcomes after Prism-hardened S3 verifier-guided
diffusion-LM search, E3 deeper executable-world-model progression, and
self-learning action-cost compounding.

**.403 outcome conditioning:**
- Exp 4359: `acceptance_gate=true`, `honest_verdict=scorer_leaky_in_search_corpus`,
  `benchmark_n=0`, `controls_differentiated=false`, and
  `s3_guided_beats_control=false`. The Prism/S3 line is not a clean null, but
  it is also not a positive generation result; .404 should quarantine the
  external scorer and repair controls before reviving the search headline.
- Exp 4361: `success_e3_deeper_tu93_reproduced`,
  `new_levels_reproduced=1`, `reproducible_total_levels=33`, and
  `verifier_is_oracle=true`. E3 remains real ARC progress, but its verifier
  caveat stays oracle-grounded.
- Exp 4364: `action_efficiency_compounds=true`,
  `acceptance_gate_passed=true`, `deployed_into_solver_kit=true`,
  `reproduction_gated=true`, and `verifier_is_oracle=false`. The LLM heuristic
  arm did not run, so .404 should test the stronger function class on the clean
  compounding substrate.

**Fresh-pass candidates marked ingested:**
- Classical Planning with LLM-Generated Heuristics, arXiv:2503.18809 - mapped
  to the .404 headline: synthesize small Python heuristic programs and select
  only by reproduced held-out action count against the deployed linear
  action-cost heuristic.
- Executable World Models for ARC-AGI-3, arXiv:2605.05138 - mapped to deeper
  tu93/sc25/tn36/lp85 progression with `verifier_is_oracle=true` kept explicit.
- CoDiLA locally coherent parallel decoding, arXiv:2603.20216 - fresh
  scorer-quarantine control for dLLM search after Exp 4359's leaky external
  scorer state.
- PAPO reward-state alignment, arXiv:2606.08501 - mapped to authentic
  trajectory-state diagnostics before any reward-guided generator training.
- Prism hierarchical trajectory search/self-verification, arXiv:2602.01842 -
  carried forward only as a repaired HTS harness target with branch-diversity,
  scorer-disagreement, and leak receipts.

out_of_band_flagged:
- A2D2 adaptive any-length discrete diffusion, arXiv:2606.13565 -
  operator-owned verifier-as-reward generator training, NOT auto-run in-loop.
- SEPO score-entropy policy optimization, arXiv:2502.01384 - operator-owned
  verifier-as-reward generator training, NOT auto-run in-loop.

flagged_for_v404: llm_generated_action_heuristics_compounding_v404

Flagged for .404: `llm_generated_action_heuristics_compounding_v404`

random_seed=4365

**Bottom line for the .404 roadmap:** do not spend the next headline slot on an
unclean Prism/S3 gain. Keep diffusion search in scorer-quarantine repair with
CoDiLA/PAPO controls, continue E3 as oracle-grounded ARC north-star progress,
and put the main .404 flag on LLM-generated action heuristics over the clean
Exp 4364 compounding substrate. A2D2 and SEPO stay out of band for
operator-owned verifier-as-reward generator training.

## 2026-06-17 Exp 4354 - .402 fork SOTA ingestion ingested

**Status:** INGESTED into `results/experiment_4354_sota_ingestion_v403.json`.

**Preconditions:** network precondition passed via Hugging Face reachability and
arXiv/WebFetch verification. If that check had failed, the only honest artifact
would have been `honest_verdict=blocked_network_unavailable`.

**Reliable-channel provenance:** `scripts/sweep_clusters.py` emitted the focused
energy/reward arXiv discovery URL. `scripts/sweep_semscholar.py` was run on the
five focused query strings and returned HTTP 429 for each query, so it produced
no usable arXiv IDs in this pass. Low-concurrency WebSearch/WebFetch plus arXiv
page checks verified arXiv:2602.01842, arXiv:2604.06260, arXiv:2606.08501,
arXiv:2605.05138, arXiv:2503.18809, arXiv:2512.24156, arXiv:2605.25931,
arXiv:2606.13565, and arXiv:2502.01384. The banned `/deep-research` channel was
not invoked.

**Filtered track:** .402 outcomes after S3 verifier-guided diffusion-LM search,
E3 deeper executable-world-model progression, and learned action-cost heuristic
action-efficiency.

**.402 outcome conditioning:**
- Exp 4348: `acceptance_gate=true`, `honest_verdict=controls_not_differentiable`,
  `benchmark_n=240`, and adversarial verification reports `TAUTOLOGY` across
  the S3-vs-control deltas. The S3 line remains alive, but .403 must harden the
  search and controls before claiming a clean generation gain.
- Exp 4351: `success_e3_deeper_tn36_reproduced`,
  `new_levels_reproduced=1`, `reproducible_total_levels=23`, and
  `verifier_is_oracle=true`. E3 has real ARC progress, but those solves remain
  oracle-grounded and should not be promoted as an oracle-free verifier moat.
- Exp 4353: `action_efficiency_improves=true`,
  `held_out_actions_baseline=25`, `held_out_actions_learned=16`,
  `positive_control_passed=true`, `reproduction_gated=true`, and
  `verifier_is_oracle=false`. The next self-learning step should generalize
  action-efficiency heuristics under reproduction gates.

**Fresh-pass candidates marked ingested:**
- Prism hierarchical trajectory search/self-verification, arXiv:2602.01842 -
  mapped to the .403 headline: harden S3 with hierarchical pruning, partial
  remasking, self-verified feedback, and explicit diversity/leakage receipts.
- S3 Stratified Scaling Search, arXiv:2604.06260 - carried forward as the base
  fixed-model verifier-guided denoising search, but only with differentiated
  controls after the .402 metric-tautology caution.
- PAPO reward-state alignment, arXiv:2606.08501 - mapped to authentic
  trajectory-state alignment diagnostics, not in-loop generator training.
- Executable World Models for ARC-AGI-3, arXiv:2605.05138 - mapped to deeper
  private-like E3 progression with verifier_is_oracle=true kept explicit.
- Classical Planning with LLM-Generated Heuristics, arXiv:2503.18809 - mapped
  to a bounded program-heuristic generalization of the reproduced action-count
  win from Exp 4353.

**Screened but not mapped as strongest rows:** Graph-Based Exploration for
ARC-AGI-3 (arXiv:2512.24156) and AERA speed-depth trade-off
(arXiv:2605.25931) were verified and read as ARC exploration context. They
support the E3/action-efficiency direction, but the .402 outcomes point more
directly to executable-world-model continuation and learned/LLM-generated action
heuristics.

out_of_band_flagged:
- A2D2 adaptive any-length discrete diffusion, arXiv:2606.13565 -
  operator-owned verifier-as-reward generator training, NOT auto-run in-loop.
- SEPO score-entropy policy optimization, arXiv:2502.01384 - operator-owned
  verifier-as-reward generator training, NOT auto-run in-loop.

flagged_for_v403: prism_hardened_s3_verifier_guided_search_v403

Flagged for .403: `prism_hardened_s3_verifier_guided_search_v403`

random_seed=4354

**Bottom line for the .403 roadmap:** keep the headline on non-training
verifier-guided diffusion-LM search, but do not repeat the .402 artifact shape.
Use Prism-style hierarchical trajectory search and partial-remasking controls to
make S3's lift auditable, keep PAPO as a state-alignment diagnostic, continue
E3 deeper progression as oracle-grounded ARC progress, generalize the learned
action-cost heuristic with reproduced action-count gates, and keep A2D2/SEPO
out of band for operator-owned generator training.

## 2026-06-17 Exp 4343 - .401 outcome SOTA ingestion ingested

**Status:** INGESTED into `results/experiment_4343_sota_ingestion_v402.json`.

**Preconditions:** network precondition passed via Hugging Face reachability and
arXiv API verification. If that check had failed, the only honest artifact would
have been `honest_verdict=blocked_network_unavailable`.

**Reliable-channel provenance:** `scripts/sweep_clusters.py` emitted focused
arXiv discovery URLs for energy/reward and world-model clusters. The first
`scripts/sweep_semscholar.py` query returned arXiv:2604.06260 and
arXiv:2602.23997; subsequent focused Semantic Scholar probes returned HTTP 429.
Low-concurrency WebSearch/WebFetch plus the arXiv API verified arXiv:2604.06260,
arXiv:2606.13565, arXiv:2606.08501, arXiv:2606.10829, arXiv:2603.12554,
arXiv:2509.25420, arXiv:2605.05138, arXiv:2605.15256, arXiv:2602.06291, and
arXiv:2602.23997. The banned `/deep-research` channel was not invoked.

**Filtered track:** .401 outcomes after leak-robust in-generation moat
replication, E3 explore-verify-plan reproduction on ar25 and sc25, and
action-role cross-game self-learning.

**.401 outcome conditioning:**
- Exp 4338: `honest_verdict=complete: in_generation_moat_replicates`,
  `in_generation_moat_replicates=true`, `scorer_leak_recheck_passed=true`,
  `controls_differentiated=true`, `benchmark_n=240`,
  `carnot_minus_best_control_delta=0.358333`, and
  `replication_ci95=[0.283333, 0.4375]`; the leak-robust in-generation moat
  replicated and the .402 headline should scale it rather than pivot away.
- Exp 4339: `game=ar25`, `offline_reproduced=true`, `plan_executed=true`,
  `reproduced_levels=1`, and `explore_lemmas_collected=7`; E3 has a reproduced
  ar25 level and should move to deeper/multi-game progression.
- Exp 4341: `game=sc25`, `offline_reproduced=true`, `plan_executed=true`,
  `reproduced_levels=1`, and `explore_lemmas_collected=6`; sc25 reproduction
  opens the path to converting the live-recorded levels, not another L1 replay.
- Exp 4342: `learned_encoder_transfer_helps=false`,
  `cross_game_state_reduction=1.00635593220339`,
  `cross_game_state_reduction_ci95=[1.0, 1.0168354897287482]`, and
  `positive_control_passed=true`; action-role cross-game value transfer is a
  powered null and needs a full interaction-world-model transfer arm or
  retirement.

**Fresh-pass candidates marked ingested:**
- S3 Stratified Scaling Search, arXiv:2604.06260 - mapped to the .402 headline:
  verifier-guided denoising-trajectory search over the leak-robust scorer.
- A2D2 adaptive any-length discrete diffusion, arXiv:2606.13565 - mapped to a
  secondary reward-guided fine-tuning arm if the fixed-model S3 scale-up holds.
- PAPO reward-state alignment, arXiv:2606.08501 - mapped to step-aware process
  rewards and entropy-guided replay diagnostics for authentic denoising states.
- Executable World Models for ARC-AGI-3, arXiv:2605.05138 - mapped to a
  multi-game/deeper-level E3 sweep after ar25 and sc25 L1 reproduced.
- ReactiveGWM, arXiv:2605.15256 - mapped to the only remaining cross-game path:
  full interaction-world-model transfer, otherwise retire the transfer line.

**Screened but not mapped as strongest rows:** ADAS (arXiv:2606.10829),
Entropy-Guided Step Selection (arXiv:2603.12554), Reward-Guided Dual-Phase
Search (arXiv:2509.25420), Foundation World Models (arXiv:2602.23997), and
Consequence-Based Utility (arXiv:2602.06291) were verified and read as context.
Consequence-Based Utility remains the correct lead if a future leak-robust
moat recheck retires in-generation guidance, but Exp 4338 makes the active
.402 branch a guided-generation scale-up instead.

flagged_for_v402:
`s3_stratified_scaling_search_guided_generation_v402`.

Flagged for .402: `s3_stratified_scaling_search_guided_generation_v402`.

random_seed=4343

**Bottom line for the .402 roadmap:** the in-generation moat settled positive in
Exp 4338, so do not pivot to consequence-based oracle-free ranking as the lead.
Scale the moat with S3-style verifier-guided denoising-trajectory search under
fixed-compute controls, keep A2D2/PAPO as training and reward-state ablations,
turn E3 into a multi-game/deeper-level reproduced-world-model sweep, and either
upgrade cross-game transfer to a full interaction-world-model representation or
retire it after the powered action-role null.

## 2026-06-17 Exp 4332 - .400 fork SOTA ingestion ingested

**Status:** INGESTED into `results/experiment_4332_sota_ingestion_v401.json`.

**Reliable-channel provenance:** `scripts/sweep_clusters.py` and
`scripts/sweep_semscholar.py` imported successfully; `sweep_clusters.py`
emitted focused arXiv discovery URLs for verifier/reward and world-model
clusters. Semantic Scholar was reachable through the helper but returned HTTP
429 for the three focused keyword probes in this loop. Low-concurrency
WebSearch/WebFetch verified arXiv:2602.11146, arXiv:2502.01384,
arXiv:2512.22336, arXiv:2605.25931, arXiv:2605.15256, arXiv:2604.17415,
arXiv:2605.18548, arXiv:2606.00291, arXiv:2605.26491, and arXiv:2510.23691.
The banned `/deep-research` channel was not invoked.

**Filtered track:** .400 outcomes after second-corpus guided-generation
replication, adaptive guided-generation scale-up, E3 executable-world-model
induction on ar25, and learned frame-encoder cross-game value transfer.

**.400 outcome conditioning:**
- Exp 4325: `honest_verdict=scorer_leaky_on_second_corpus`,
  `in_generation_moat_replicates=false`, `controls_differentiated=false`,
  `scorer_leak_recheck_passed=false`, `benchmark_n=0`,
  `carnot_minus_best_control_delta=0.0`, and `replication_ci95=[0.0, 0.0]`;
  the first in-generation moat did not replicate because the second-corpus
  scorer failed the independent leak recheck.
- Exp 4326: `adaptive_guidance_beats_control=false`,
  `adaptive_ci95=[-0.075, 0.35]`, `adaptive_controls_differentiated=true`, and
  `adaptive_benchmark_n=40`; adaptive guidance differentiated controls but did
  not beat the engaged control.
- Exp 4327: `offline_reproduced=false`, `plan_executed=false`,
  `reproduced_levels=0`, `verifier_best_accuracy=0.8875`, and
  `residual_mismatch_class=missing_world_model_rule_gap_hidden_undo_stack_action7`;
  E3 made a useful partial world model but no reproduced solve.
- Exp 4331: `learned_encoder_transfer_helps=false`,
  `cross_game_state_reduction=1.0084925690021231`,
  `cross_game_state_reduction_ci95=[1.0, 1.0303068758652514]`, and
  `baseline_solves_held_out=true`; the positive-control solver worked, but the
  learned frame encoder still did not reduce held-out search states.

**Fresh-pass candidates marked ingested:**
- DiNa-LRM diffusion-native latent reward modeling, arXiv:2602.11146 - mapped to
  a leak-robust partial-state reward scorer before any scaled generation claim.
- SEPO score-entropy policy optimization, arXiv:2502.01384 - mapped to bounded
  discrete-diffusion reward optimization after the adaptive schedule-only null.
- Agent2World adaptive world-model testing, arXiv:2512.22336 - mapped to
  behavior-aware E3 verifier tests for hidden transition gaps.
- AERA explore-verify-plan ARC-AGI-3 agent, arXiv:2605.25931 - mapped to an
  explicit information-gain budget before E3 planning.
- ReactiveGWM game-agnostic interaction representation, arXiv:2605.15256 -
  mapped to richer cross-game value features after the tiny frame encoder stayed flat.

**Screened but not mapped as strongest rows:** Reward Score Matching
(arXiv:2604.17415), STT-Arena (arXiv:2605.18548), Representation-Rationalizability
(arXiv:2606.00291), Diffusion LAIR (arXiv:2605.26491), and Game-TARS
(arXiv:2510.23691) were read as relevant context. They were not selected as
strongest rows because the observed .400 failures point more directly to
leak-robust noisy-state rewards, adaptive world-model tests, explore-before-plan
discipline, and game-invariant interaction features.

Already-covered context not re-ingested as fresh method rows: A2D2
(arXiv:2606.13565), TR2-D2 (arXiv:2509.25171), Reward-State Alignment
(arXiv:2606.08501), diffusion step selection (arXiv:2603.12554), Executable
World Models for ARC-AGI-3 (arXiv:2605.05138), and Graph-Based Exploration for
ARC-AGI-3 (arXiv:2512.24156).

flagged_for_v401:
`leak_robust_diffusion_native_partial_state_reward_v401`.

Flagged for .401: `leak_robust_diffusion_native_partial_state_reward_v401`.

random_seed=4332

**Bottom line for the .401 roadmap:** do not scale the Exp 4315 guided-generation
claim yet. The second-corpus leak check failed and the adaptive run was a bounded
null, so the strongest .401 entry is leak-robust diffusion-native partial-state reward
scoring. Keep E3 on adaptive testing plus explore-before-plan repair, and
retry cross-game value transfer only with richer game-invariant interaction
features.

## 2026-06-17 Exp 4320 - .399 fork SOTA ingestion ingested

**Status:** INGESTED into `results/experiment_4320_sota_ingestion_v400.json`.

**Reliable-channel provenance:** `scripts/sweep_clusters.py` and
`scripts/sweep_semscholar.py` imported successfully; `sweep_clusters.py`
emitted focused arXiv discovery URLs for verifier, energy, and routing
clusters. Semantic Scholar was reachable through the helper but returned HTTP
429 for the four focused keyword probes in this loop. Low-concurrency
WebSearch/WebFetch verified arXiv:2606.13565, arXiv:2509.25171,
arXiv:2606.15841, arXiv:2502.08773, arXiv:2605.05478, arXiv:2602.22871,
arXiv:2602.01849, arXiv:2603.04445, arXiv:2512.02543, and arXiv:2605.09965.
The banned `/deep-research` channel was not invoked.

**Filtered track:** .399 outcomes after the IR3DE+CASCAL cross-domain router,
the DiffusionGemma reward-guided step-stitching run, the efficiency cascade
deployment run, and the ARC cross-game learned-verifier transfer run.

**.399 outcome conditioning:**
- Exp 4314: `cross_domain_selection_holds=false`,
  `cross_domain_delta=0.2307692308`,
  `cross_domain_delta_ci95=[-0.1153846154, 0.5384615385]`, and
  `label_ablation_robust=true`; the selector survived the label-ablation check
  but did not make a decision-grade cross-domain moat.
- Exp 4315: `diffusiongemma_guidance_moat=true`,
  `controls_differentiated=true`, `scorer_leak_recheck_passed=true`,
  `carnot_minus_best_control_delta=0.225`, and
  `guidance_moat_ci95=[0.075, 0.375]`; the external-verifier-guided
  in-generation moat closed.
- Exp 4316: `cascade_dominates_controls=false`,
  `accuracy_always_energy=0.6`, `accuracy_cascade=0.55`, and
  `cost_ratio_cascade=0.3019632358`; the cascade was useful as a diagnostic but
  the always-energy verifier remained the cleaner operating point.
- Exp 4318: `cross_game_transfer_helps=false`,
  `cross_game_state_reduction=1.0`, and `baseline_solves_held_out=true`; the
  uniform positive-control solver worked, but the learned value-head did not
  reduce held-out search states.

**Fresh-pass candidates marked ingested:**
- A2D2 adaptive any-length discrete diffusion, arXiv:2606.13565 - mapped to the
  .400 scaled external-verifier-guided DiffusionGemma generation headline.
- TR2-D2 tree-search trajectory replay, arXiv:2509.25171 - mapped to bounded
  reward-guided replay buffers for DiffusionGemma partial-state denoising.
- Heteroskedastic Signals in Budgeted LLM Verification, arXiv:2606.15841 -
  mapped to cost-stratified cascade diagnostics after the Exp 4316 global
  cascade failed to dominate.
- UniRoute unseen-model routing, arXiv:2502.08773 - mapped to cross-domain
  performance-fingerprint routing without domain labels or family IDs.
- LANTERN experience-gated transfer, arXiv:2605.05478 - mapped to gated
  multi-source ARC value-head transfer after the Exp 4318 flat result.

**Screened but not mapped as strongest rows:** Reward-Guided Stitching
(arXiv:2602.22871), Self-Rewarding SMC (arXiv:2602.01849), CSMC
(arXiv:2602.09424), Dynamic Model Routing and Cascading (arXiv:2603.04445),
Inference-Time Distillation (arXiv:2512.02543), and Game Multiverse
(arXiv:2605.09965) were read as relevant context. They were not re-ingested as
fresh method rows because Reward-Guided Stitching and Self-Rewarding SMC are
already in earlier sweeps and the others are weaker fits than the five mapped
rows for the observed .399 outcomes.

Already-covered context not re-ingested as fresh method rows: Budget-aware
Discriminative Verification, IR3DE, Routing with Generated Data / CASCAL,
TTARAG, SMC importance weighting for discrete diffusion, EEVEE,
optimize_anything / GEPA, RefGRPO, SLMJury, ReMDM, ARC-TGI, ARC-GEN, RFG,
EDLM, EntRGi, Manta-LM, masked-discrete-diffusion guidance dynamics, INSPECTOR
Representation-as-a-Judge, ABPR, and Decocted Experience.

flagged_for_v400:
`scaled_external_verifier_guided_diffusiongemma_generation_v400`.

Flagged for .400: `scaled_external_verifier_guided_diffusiongemma_generation_v400`.

random_seed=4320

**Bottom line for the .400 roadmap:** Exp 4315 is the only .399 fork that closed
decision-grade, so make .400 an A2D2/TR2-D2-style scaled guided generation
headline over the existing leak-checked DiffusionGemma scorer. Keep
cross-domain routing on UniRoute-style frozen fingerprints, convert cascade work
into heteroskedastic threshold diagnostics, and only retry cross-game transfer
with LANTERN-style experience gates.

## 2026-06-17 Exp 4309 - .398 fork SOTA ingestion ingested

**Status:** INGESTED into `results/experiment_4309_sota_ingestion_v399.json`.

**Reliable-channel provenance:** `scripts/sweep_clusters.py` and
`scripts/sweep_semscholar.py` imported successfully; `sweep_clusters.py`
emitted focused arXiv discovery URLs for verifier/energy clusters. Semantic
Scholar returned arXiv IDs for budget-aware discriminative verification,
domain routing, and discrete diffusion probes, and returned HTTP 429 for one
RAG adaptation probe and one rerouting-security probe. WebSearch/WebFetch
verified arXiv:2510.14913, arXiv:2606.06098, arXiv:2601.09692,
arXiv:2601.11443, arXiv:2505.22524, arXiv:2601.21380, arXiv:2602.09424, and
arXiv:2605.05007. The banned `/deep-research` channel was not invoked.

**Filtered track:** .398 outcomes after the hardened iso-FLOPs verifier-vs-judge
run, the DiffusionGemma engaged-control guidance run, and the cross-domain
selector generalization stress.

**.398 outcome conditioning:**
- Exp 4303: `efficiency_pareto_holds=true`,
  `accuracy_energy_verifier=0.8`, `accuracy_best_judge=0.5`,
  `accuracy_delta_ci95=[0.1, 0.5]`, and `cost_ratio=1.03e-08`; the efficiency
  axis hardened into a decision-grade Pareto win.
- Exp 4304: `diffusiongemma_guidance_moat=false`,
  `controls_differentiated=true`, `scorer_leak_recheck_passed=true`,
  `carnot_minus_best_control_delta=0.133334`, and
  `guidance_moat_ci95=[-0.066667, 0.366667]`; guided generation improved the
  point estimate but did not clear the engaged-control CI gate.
- Exp 4305: `cross_domain_selection_holds=false`,
  `cross_domain_delta=0.2307692308`,
  `cross_domain_ci95=[-0.1153846154, 0.5384615385]`, and
  `label_ablation_robust=true`; the FoVer slice was positive but underpowered,
  while ARC and ARC-GEN held-out reads collapsed.

**Fresh-pass candidates marked ingested:**
- Budget-aware discriminative verification, arXiv:2510.14913 - mapped to the
  .399 deployment/cascade-router headline after Exp 4303 hardened efficiency.
- IR3DE linear domain-expert router, arXiv:2606.06098 - mapped to a simpler
  domain-invariant router rebuild after Exp 4305's broad cross-domain collapse.
- Routing with Generated Data / CASCAL, arXiv:2601.09692 - mapped to
  generated-data router pretraining with query-only anti-leak controls.
- TTARAG retrieval-prediction adaptation, arXiv:2601.11443 - mapped to
  powered retrieval-augmented selector adaptation on train-side traces only.
- SMC importance weighting for discrete diffusion, arXiv:2505.22524 - mapped
  to a secondary DiffusionGemma repair track with engaged particle/reweighting
  controls.

**Screened but not mapped as strongest rows:** RerouteGuard (arXiv:2601.21380),
CSMC clean-sample Markov chains (arXiv:2602.09424), Uno-Orchestra
(arXiv:2605.05007), and TRouter (arXiv:2604.09377) were read as adjacent
routing-security, clean-sample diffusion, selective-delegation, and cold-start
routing evidence. They remain weaker for `.399` than the mapped rows because
RerouteGuard is attack-specific, CSMC is molecule/biology-centered,
Uno-Orchestra is a broader multi-agent policy, and TRouter overlaps the more
direct IR3DE/CASCAL router rebuild path.

Already-covered context not re-ingested as fresh method rows: EEVEE,
optimize_anything / GEPA, RefGRPO, SLMJury, ReMDM, ARC-TGI, ARC-GEN, RFG,
EDLM, EntRGi, Self-Improving LLM Agents at Test-Time, SEVerA, DPRM,
Reward-Guided Stitching, Manta-LM, masked-discrete-diffusion guidance dynamics,
INSPECTOR Representation-as-a-Judge, ABPR, Decocted Experience, and COVER.

flagged_for_v399:
`budget_aware_discriminative_cascade_router_v399`.

Flagged for .399: `budget_aware_discriminative_cascade_router_v399`.

random_seed=4309

**Bottom line for the .399 roadmap:** the efficiency axis is the only .398 fork
that hardened cleanly, so make the next headline a budget-aware discriminative
cascade-router. Rebuild the cross-domain router with IR3DE/CASCAL-style
domain-invariant training before claiming broader transfer, and keep
keep SMC-guided DiffusionGemma as the secondary repair track rather than the
.399 headline.

## 2026-06-16 Exp 4298 - .397 fork SOTA ingestion ingested

**Status:** INGESTED into `results/experiment_4298_sota_ingestion_v398.json`.

**Reliable-channel provenance:** `scripts/sweep_clusters.py` and
`scripts/sweep_semscholar.py` imported successfully; `sweep_clusters.py`
emitted focused arXiv discovery URLs. Semantic Scholar was reachable through
the helper but returned HTTP 429 for several low-concurrency probes, while one
small-judge query returned arXiv IDs. WebSearch/WebFetch verified
arXiv:2606.11182, arXiv:2605.19633, arXiv:2606.14211, arXiv:2606.07810, and
arXiv:2503.00307. The banned `/deep-research` channel was not invoked.

**Filtered track:** .397 outcomes after the non-degenerate ARC-GEN
cross-generator run, partial-state DiffusionGemma scorer build, and the missing
strong-judge efficiency hardening artifact.

**.397 outcome conditioning:**
- Exp 4291: `cross_generator_holds=true`,
  `non_degenerate_guards_pass=true`, and
  `headline_outcome=arcgen_cross_generator_generalizes`; the ARC-GEN
  cross-generator moat is closed rather than still degenerate.
- Exp 4292: `partial_state_scorer_built=true`,
  `partial_state_leak_free=true`, `partial_state_auroc=0.966143`, and
  `leak_ablation_auroc=0.937365`; the missing scorer from `.396` now exists
  and survived the leak audit.
- Exp 4294: `strong_judge_efficiency_outcome=unavailable_missing_exp4294_json`
  because `results/experiment_4294_verifier_efficiency_harden_strong_judge.json`
  was not present at ingestion time. Do not claim the strong-judge efficiency
  hardening result until the artifact exists.

**Fresh-pass candidates marked ingested:**
- EEVEE router-prompt co-evolution, arXiv:2606.11182 - mapped to the .398
  broader-domain selector generalization headline after cross-generator ARC-GEN
  transfer held.
- optimize_anything / GEPA text-parameter search, arXiv:2605.19633 - mapped to
  train-only selector/harness text optimization with locked held-out domains.
- RefGRPO reflection-outcome calibration, arXiv:2606.14211 - mapped to
  exact-outcome-calibrated selector self-reflection and selective prediction.
- SLMJury small-judge budget function, arXiv:2606.07810 - mapped to the
  efficiency-axis fallback while Exp 4294 remains unavailable.
- ReMDM remasking inference-time scaling, arXiv:2503.00307 - mapped to the
  secondary guided-generation path now that the partial-state scorer is
  leak-free.

**Screened but not mapped as strongest rows:** SIA (arXiv:2605.27276), SE-GA
(arXiv:2605.16883), and Sensi (arXiv:2603.17683) were read as adjacent
self-improvement and agentic-curriculum evidence. They remain weaker for `.398`
because SIA mutates weights/harness together, SE-GA is GUI-specific, and Sensi
depends on an LLM-as-judge curriculum while reporting a perception bottleneck.

Already-covered context not re-ingested as fresh method rows: ARC-TGI, ARC-GEN,
RFG, EDLM, EntRGi, Self-Improving LLM Agents at Test-Time, SEVerA, DPRM,
Reward-Guided Stitching, Manta-LM, masked-discrete-diffusion guidance dynamics,
INSPECTOR Representation-as-a-Judge, ABPR, Decocted Experience, and COVER.

flagged_for_v398:
`eevee_router_prompt_broader_domain_selector_v398`.

Flagged for .398: `eevee_router_prompt_broader_domain_selector_v398`.

random_seed=4298

**Bottom line for the .398 roadmap:** the ARC-GEN transfer critique is now
closed on a non-degenerate pool and the partial-state scorer is leak-free, so
the next headline should broaden selector generalization across heterogeneous
domains using EEVEE-style router-prompt co-evolution. Keep ReMDM as the
secondary guided-generation branch, and treat small-verifier efficiency as
unconfirmed until Exp 4294 exists.

## 2026-06-16 DMoE (arXiv:2606.14243) - candidate substrate for the (out-of-band) verifier-as-reward retry

**Source:** operator-directed read, 2026-06-16. Yue et al. (Tsinghua),
"Decoupled Mixture-of-Experts for Parametric Knowledge Injection" (arXiv:2606.14243).
Single-source (abstract-level) read — treat as a CANDIDATE, not a validated direction.

**Score: 3 x 3 x 2 x 2 = 36** — moderate-low. It does NOT touch the verifier
moat, ARC, or the in-generation/diffusion question (north-star alignment is low;
it is about knowledge injection, not verification). The leverage is on the
SECONDARY self-learning track, which is currently out-of-band.

**Position:** The verifier-as-reward / self-learning track failed 7x this run on
LIVE LoRA-RFT (Gemma4ClippableLinear attach, then "cannot train in the task
window"), and got punted out-of-band (exp4263). DMoE proposes a cleaner
modularity for the SAME goal — folding verifier-certified knowledge into the
sovereign model — that directly addresses the two failure modes:
- knowledge becomes a DECOUPLED, independently-updatable expert (base untouched
  -> no catastrophic forgetting / knowledge conflict that full-finetune risks);
- experts attach ONLY to the final-layer FFN, PRESERVING KV-cache (cheap to run,
  and a far smaller training surface than full/LoRA RFT -> likelier to fit a
  bounded window);
- a lightweight uncertainty-aware router activates the expert only when needed
  (a peer of Carnot's cascade-router / verifier-as-router; the verifier signal
  could itself gate activation).
- Decentralization fit: users add/swap domain experts LOCALLY without retraining
  the base — same shape as the local-first sovereignty constraint.

**Next experiment (candidate, not queued):** when self-learning re-opens, re-attempt
verifier-as-reward distillation as DMoE-style expert injection of verifier-certified
traces (final-FFN decoupled expert + uncertainty router) INSTEAD of full/LoRA RFT.
Smoke-first: does a final-FFN decoupled expert attach + train >=20 steps in-window on
the gemma-4-E4B base (the harness that the LoRA path could not), loss moving, no base
regression? Honest null if it also cannot fit the window. Pair with the literature
two-source rule before any novelty claim (DMoE vs prior adapter/MoE knowledge-injection).

## 2026-06-16 Exp 4286 - .396 fork SOTA ingestion ingested

**Status:** INGESTED into `results/experiment_4286_sota_ingestion_v397.json`.

**Reliable-channel provenance:** `scripts/sweep_clusters.py` and
`scripts/sweep_semscholar.py` imported successfully; `sweep_clusters.py`
emitted the focused arXiv discovery URLs; Semantic Scholar returned HTTP 429
for the two low-concurrency keyword probes, so it was reachable as code but did
not promote sources. WebSearch/WebFetch verified arXiv:2605.14531,
arXiv:2506.10971, arXiv:2601.22588, arXiv:2603.20334, and arXiv:2604.04373.
The banned `/deep-research` channel was not invoked.

**Filtered track:** .396 outcomes after the DiffusionGemma full run,
ARC-GEN cross-family stress, self-learning repower, and verifier-efficiency
head-to-head.

**.396 outcome conditioning:**
- Exp 4281: `diffusiongemma_guidance_moat=false` and
  `blocked_partial_state_verifier`; the learned verifier cannot score partial
  DiffusionGemma token canvases.
- Exp 4282: raw `arcgen_cross_family_holds=true`, but the outer-loop correction
  records `arcgen_cross_family_holds_outerloop_corrected=false` with
  `DEGENERATE_SEPARATION`, so ARC-GEN is not headline-clean generalization.
- Exp 4284: `efficiency_parity_at_lower_cost=true`,
  `accuracy_delta=0.4423076923`, CI95 `[0.3076923077, 0.5769230769]`, and
  `cost_ratio=1.95e-08`; the cheap energy verifier remains the efficient
  judging path.

**Fresh-pass candidates marked ingested:**
- Manta-LM closed-loop diffusion control, arXiv:2605.14531 - mapped to the
  missing partial-state scorer/controller required before another
  DiffusionGemma guidance headline.
- Masked discrete diffusion guidance dynamics, arXiv:2506.10971 - mapped to a
  guidance-strength and trajectory-stability audit for masked denoising.
- INSPECTOR Representation-as-a-Judge, arXiv:2601.22588 - mapped to small
  representation-probe verifier distillation after Exp 4284's cost win.
- ABPR trace-guided procedural refinement, arXiv:2603.20334 - mapped to a
  non-degenerate proof-trace cross-substrate generalization stress gate.
- Decocted experience for test-time inference, arXiv:2604.04373 - mapped to
  retrieval-only selector context while the online-weight-update result remains
  under tautology correction.

Already-covered context not re-ingested as fresh method rows: RFG, EDLM,
EntRGi, ARC-GEN, Paying Less Generalization Tax, S3, Self-Improving LLM Agents
at Test-Time, SEVerA, ARC-TGI, DPRM, Reward-Guided Stitching, and COVER.

flagged_for_v397:
`manta_partial_state_scorer_diffusiongemma_v397`.

Flagged for .397: `manta_partial_state_scorer_diffusiongemma_v397`.

random_seed=4286

**Bottom line for the .397 roadmap:** the DiffusionGemma moat FAILED for a
specific engineering reason, not because a learned external verifier lost to
RFG. Build the learned partial-state scorer first, keep ARC-GEN out of the
headline until the degenerate pool is repaired with proof-trace or relational
substrates, preserve the cheap-verifier efficiency path, and keep online updates
retrieval-only until the tautology audit is fixed.

## 2026-06-16 Exp 4276 - .395 fork SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-v396-2026-06-16.md`.

**Filtered track:** .395 ARC cross-family transfer after the hardened
oracle-distinct Set-Encoder selector generalized on held-out families
(`cross_family_delta=0.4038461538`, CI95 `[0.25, 0.5576923077]`) while the
fresh ARC-TGI fallback was correctly gate-blocked because the existing family
split was feasible.

**Fresh-pass candidates marked ingested:**
- Paying Less Generalization Tax, arXiv:2601.18217 - mapped to a stronger
  cross-family stress split with richer randomized family metadata.
- ARC-GEN, arXiv:2511.00162 - mapped to an independent procedural-family
  replication of the Exp 4271 transfer win.
- RFG, arXiv:2509.25604 - mapped as the queued DiffusionGemma full-run method
  now that cross-family selector generalization opened the scale-up gate.
- Self-Improving LLM Agents at Test-Time, arXiv:2510.07841 - mapped to
  low-margin selector-head adaptation on held-out generated families.
- SEVerA, arXiv:2603.25111 - mapped to verified fallback contracts for any
  self-improving selector or diffusion-refiner branch.

Already-covered context not re-ingested as fresh method rows: ARC-TGI,
Reliability Gap, DPRM, entropy-guided diffusion RL, L-VARC, TrajAD, RL^V,
EntRGi, and Self-Trained Verification.

.395 status mapped honestly: Exp 4271 `cross_family_generalizes` with
`cross_family_win_holds=true`, `cross_family_delta=0.4038461538`,
`cross_family_ci95=[0.25, 0.5576923077]`, `held_out_family_n=52`, and
`verifier_is_oracle=false`; Exp 4272 was blocked because the existing-pool
family split was feasible.

flagged_for_v396:
`rfg_diffusiongemma_full_run_plus_arcgen_transfer_stress_v396`.

Flagged for .396: `rfg_diffusiongemma_full_run_plus_arcgen_transfer_stress_v396`.

**Bottom line for the .396 roadmap:** cross-family did GENERALIZE, so run the
bounded RFG-style DiffusionGemma full-run arm with exact-grid selector
arbitration, and use ARC-GEN to independently stress the transfer claim while
keeping a stronger generalization stress test.

## 2026-06-15 Exp 4265 - .394 fork SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-v395-2026-06-15.md`.

**Filtered track:** .394 ARC oracle-distinct forks after the selector win
survived provenance-blind and multi-seed hardening, while cross-game transfer
was blocked, synthesis underperformed selection, DiffusionGemma preflight was
loader-blocked, and code replication read corpus-specific.

**Fresh-pass candidates marked ingested:**
- ARC-TGI, arXiv:2603.05099 - mapped as the strongest .395 method: recover the
  missing task-family/game-disjoint transfer substrate.
- Reliability Gap in Benchmark Auditing, arXiv:2606.03305 - mapped to
  provenance-first leak discipline after the high-origin-probe but surviving
  provenance-blind audit.
- DPRM, arXiv:2604.24357 - mapped to verifier/process-reward token ordering
  only after DiffusionGemma loader repair.
- Entropy-guided step selection for diffusion LLM RL, arXiv:2603.12554 - mapped
  to a deferred denoising-step reward smoke after loader repair.
- L-VARC, arXiv:2606.12847 - mapped to training-only semantic abstraction over
  ARC-TGI families, with privileged features removed at inference.

Already-covered context not re-ingested as fresh method rows: Compute-as-Teacher,
GSA, GenSelect-BoN, Reward-Guided Stitching, S3, EDLM, arXiv:2406.01572
discrete guidance, CoDeC, ARC of Progress, ARCTraj, and Compositional
Neuro-Symbolic Reasoning.

.394 status mapped honestly: Exp 4256 `arc_provenance_blind_win_survives` with
`provenance_blind_delta=0.3846153846`; Exp 4257
`arc_oracle_distinct_win_replicates_multiseed` with `mean_delta=0.4576923077`;
Exp 4258 `blocked_arc_game_ids_unrecoverable`; Exp 4259
`arc_synthesis_underperforms_selection` with `synthesis_breaks_oracle_ceiling=false`
and `synthesis_minus_oracle_delta=-0.2826086957`; Exp 4260
`blocked_diffusiongemma_gguf_loader_failed` with `preflight_go=false`; Exp 4264
`code_oracle_distinct_replication_corpus_specific` with
`code_replication_beats_vote=false`.

flagged_for_v395:
`arc_tgi_family_generator_cross_game_generalization_v395`.

Flagged for .395: `arc_tgi_family_generator_cross_game_generalization_v395`.

**Bottom line for the .395 roadmap:** do not spend .395 on full DiffusionGemma
or another synthesis headline yet. First repair the transfer substrate with a
provenance manifest and ARC-TGI-style family-disjoint candidate pool, then test
whether the hardened Set-Encoder win survives held-out task families. Keep DiffusionGemma as loader repair, not a full-run .395 bet.

## 2026-06-15 Exp 4251 - .393 planning sweep SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-set-encoder-offline-rft-v394-2026-06-15.md`.

**Filtered track:** ARC oracle-distinct set-encoder scale-up after Exp 4245
landed the clean A3 beats-vote win, with Exp 4246 code replication blocked on a
missing distinct candidate corpus and Exp 4248 offline reward-weighted SFT
blocked by the upstream harness smoke.

**Seed and fresh-pass candidates marked ingested:**
- Set-LLM, arXiv:2505.15433 - mapped as the high-capacity selector scale-up
  after the Exp 4245 DeepSets-style set encoder already beat vote.
- AggLM, arXiv:2509.06870 - mapped as the strongest .394 method: a generative
  reconciler that synthesizes a corrected grid from Set-Encoder evidence.
- ARBITER, arXiv:2605.26172 - mapped to wrong-majority basin diagnostics and
  conservative evidence-over-vote accounting.
- Budget-aware discriminative verification, arXiv:2510.14913 - mapped to
  cost-normalized vote-plus-verifier hybrid reporting.
- RAFT, arXiv:2504.11343, and VAR, arXiv:2502.11026 - mapped to the owed
  offline reward-weighted SFT path after the harness proves real training.
- Spurious Rewards, arXiv:2506.10947 - mapped to the required same-base
  random-label Arm B control for any reward-training claim.
- SCOPE, arXiv:2512.15146 - mapped to per-region ARC evidence for the .394
  AggLM synthesis ablation.

Exp 4245 status mapped honestly: `headline_outcome=arc_oracle_distinct_set_encoder_beats_vote`,
`set_encoder_minus_vote_delta=0.4423076923`, CI95 `[0.3076923077, 0.5961538462]`,
`margin_override_minus_vote=0.4230769231`, and `oracle_distinct_beats_vote=true`.
Exp 4246 status mapped honestly: `blocked_code_second_corpus_missing`; code
robustness is unresolved, not refuted. Exp 4248 status mapped honestly:
`blocked_gate_check_failed` because Exp 4247 reported `harness_smoke_passed=false`,
`steps_run=0`, and `trainable_param_count=0`.

flagged_for_v394:
`agglm_synthesize_corrected_grid_from_set_encoder_evidence_v394`.

Flagged for .394: `agglm_synthesize_corrected_grid_from_set_encoder_evidence_v394`.

**Bottom line for the .394 roadmap:** scale the proven ARC set-encoder win with
AggLM-style corrected-grid synthesis plus SCOPE per-region evidence on a bigger
pool. Keep code replication as a robustness gate and treat reward-weighted SFT as an owed gate after the harness proves real training.
## 2026-06-15 Exp 4238 - .392 planning sweep SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-cross-candidate-aggregator-v393-2026-06-15.md`.

**Filtered track:** strengthened oracle-distinct ARC aggregation after Exp 4231
built a sparse cross-candidate aggregator, Exp 4232 tied vote despite headroom,
and Exp 4233 beat vote on code with `disambiguation_read=ARC_null_is_data_sparsity`.

**Seed and fresh-pass candidates marked ingested:**
- Set-Encoder, arXiv:2404.06912 - mapped as the strongest .393 architecture
  lever: full cross-candidate attention instead of Exp 4231's augmented-feature
  logistic aggregator.
- Calibrated Reasoning, arXiv:2509.19681 - mapped to imbalance-aware calibrated
  losses, but only after ARC positive-candidate growth.
- Margin-triggered re-arbitration, arXiv:2606.04323 - kept as the deployment
  guard because Exp 4232's margin override also tied vote.
- SCOPE, arXiv:2512.15146 - mapped to per-region ARC evidence and dense
  confidence signals for wrong-majority cases.
- Adaptive verification allocation, arXiv:2602.03975 - mapped to compute
  routing after a stronger score exists.
- MSV, arXiv:2603.03417 - mapped to joint cross-sequence scoring as the
  direct model-class corroboration for Set-Encoder.
- AggLM, arXiv:2509.06870, and AgentAuditor, arXiv:2602.09341 - mapped to
  review/reconcile/synthesize and localized evidence audit arms if selection
  still leaves oracle headroom unused.

Exp 4231 status mapped honestly: `oracle_distinct_auroc=0.7865558646`,
`positive_candidate_n=20`, `wrong_majority_n=9`, and
`no_learnable_gain_reason=too_few_positives_after_growth`. Exp 4232 status
mapped honestly: `aggregator_minus_vote_delta=0.0`,
`oracle_minus_vote=0.1730769231`, and `oracle_distinct_beats_vote=false`.
Exp 4233 status mapped honestly: `code_predictor_minus_vote_delta=0.03125`,
CI95 `[0.00625, 0.0625]`, and
`disambiguation_read=ARC_null_is_data_sparsity`.

flagged_for_v393:
`bigger_arc_pool_full_set_encoder_agglm_aggregator_v393`.

Flagged for .393: `bigger_arc_pool_full_set_encoder_agglm_aggregator_v393`.

**Bottom line for the .393 roadmap:** grow ARC positives, run a full Set-Encoder against the augmented-feature aggregator, and build a bigger ARC pool before declaring the oracle-distinct selection thesis bounded.
## 2026-06-15 Exp 4226 - .391 planning sweep SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-learned-aggregator-v392-2026-06-15.md`.

**Filtered track:** learned aggregation after Exp 4220 trained the ARC verifier
and Exp 4221 found oracle headroom but `oracle_distinct_beats_vote=false`.

**Seed and fresh-pass candidates marked ingested:**
- AggLM, arXiv:2509.06870 - mapped as the strongest .392 follow-up: convert
  the A2 ARC verifier from flat rerank into review/reconcile/synthesize
  aggregation for minority-correct recovery.
- AgentAuditor, arXiv:2602.09341 - mapped to localized evidence auditing and
  the LLM-as-judge efficiency head-to-head.
- GenSelect-BoN, arXiv:2602.02143 - mapped as the RL-trained selection-only
  baseline and recipe.
- MSV, arXiv:2603.03417 - mapped to cross-candidate features and whole-set
  verifier calibration.
- Online CoT-verifier learnability, arXiv:2603.03538, plus SR-TTRL ICML 2026 -
  mapped to the verifier-as-reward self-learning loop after a positive
  aggregator gate.

Exp 4220 status mapped honestly: `selector_trained=true`,
`oracle_distinct_auroc=0.778980279`, and `wrong_majority_n=5`. Exp 4221 status
mapped honestly: `oracle_minus_vote=0.3571428571`,
`verifier_minus_vote_delta=-0.0714285714`, and
`oracle_distinct_beats_vote=false`.

flagged_for_v392:
`agglm_style_arc_review_reconcile_aggregator_v392`.

Flagged for .392: `agglm_style_arc_review_reconcile_aggregator_v392`.

**Bottom line for the .392 roadmap:** run the AggLM-style ARC aggregator before another flat rerank.
## 2026-06-14 Exp 4215 - .390 planning sweep SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-oracle-distinct-v391-2026-06-14.md`.

**Filtered track:** oracle-distinct learned ARC verifier, wrong-majority
recovery, detector-axis abstention, and execution-reward baselines kept separate
from moat claims.

**Seed and fresh-pass candidates marked ingested:**
- ARBITER, arXiv:2605.26172 - mapped to the wrong-majority headroom target and
  a conservative override that only beats vote when learned margin is high.
- SCOPE, arXiv:2512.15146 - mapped to per-region confidence and subgroup
  features for the A2/A3 ARC verifier.
- ThinkPRM, arXiv:2504.16828, and the PRM survey, arXiv:2510.08049 - mapped to
  the learned process-verifier recipe and the selector/detector/reward taxonomy.
- V-STaR, arXiv:2402.06457 - mapped to the accepted/rejected correctness
  boundary already used in-repo.
- Calibrated Reasoning, arXiv:2509.19681 - mapped to Exp 4208's detector and
  abstention axis.
- ExecVerify, arXiv:2603.11226, and EVOM, arXiv:2604.00442 - mapped to the B1
  execution-reward baselines, explicitly not oracle-distinct moat evidence.

Exp 4210 status mapped honestly: `blocked_gate_check_failed`; A3 did not run
because A2 did not produce `selector_trained=true`. Exp 4208 remains detector
evidence, not a vote-beating selector result.

flagged_for_v391:
`arbiter_conservative_override_arc_wrong_majority_v391`.

Flagged for .391: `arbiter_conservative_override_arc_wrong_majority_v391`.

**Bottom line for the .391 roadmap:** run the ARBITER conservative override over ARC wrong-majority cases first.
## 2026-06-14 Exp 4203 - .389 planning sweep SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-verifier-as-reward-v390-2026-06-14.md`.

**Filtered track:** verifier-as-reward de-confounding, code-RLVR baselines,
process/self-distill math rewards, and cost-normalized verifier plus
self-consistency framing.

**Seed and fresh-pass candidates marked ingested:**
- Spurious Rewards, arXiv:2506.10947 - mapped to the mandatory non-Qwen base
  and same-generator random-label A-vs-B control.
- Spurious Rewards Paradox, arXiv:2601.11061 - mapped to the
  memorization-shortcut diagnostic.
- RLV-epsilon-R, arXiv:2601.04411 - mapped to TPR/FPR/Youden-J reporting.
- RLEF, arXiv:2410.02089; Aletheia, arXiv:2601.12186; and CodeScaler,
  arXiv:2602.17684 - mapped to code-RLVR baselines a positive result must beat.
- Self-Distilled RLVR, arXiv:2604.03128; CEPO, arXiv:2605.19436; and
  ThinkPRM, arXiv:2504.16828 - mapped to the math-process-reward fork after
  the de-confounding gate.
- Budget-aware discriminative verification, arXiv:2510.14913, and
  When To Solve/Verify, arXiv:2504.01005 - mapped to the hybrid verifier plus
  self-consistency cost-crossover framing.

Exp 4199 status mapped honestly: `blocked_gate_check_failed`; the A-vs-B
collection did not run because the upstream training-launched gate was false.

flagged_for_v390:
`non_qwen_same_generator_random_label_ablation_v390`.

Flagged for .390: `non_qwen_same_generator_random_label_ablation_v390`.

**Bottom line for the .390 roadmap:** run the non-Qwen same-generator
random-label A-vs-B replication before any math-process-reward fork.
## 2026-06-14 Exp 4192 - .388 planning sweep SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-efficiency-gap4-diffusion-v389-2026-06-14.md`.

**Filtered track:** DiffusionGemma verifier-guided test-time scale-up,
efficiency-moat LLM-judge comparator and cost normalization, plus the CEM
operator-authorization closure for the retired GAP-3 trained-content-energy
selector lineage.

**Seed and fresh-pass candidates marked ingested:**
- Test-Time Scaling with Diffusion Language Models via Reward-Guided Stitching,
  arXiv:2602.22871 - mapped to step-level DiffusionGemma guidance and stitching
  ablations.
- S^3 Stratified Scaling Search, arXiv:2604.06260 - mapped to the strongest
  `.389` DiffusionGemma verifier-guided denoising-search target.
- Self-Rewarding SMC, arXiv:2602.01849 - mapped as the self-guided particle
  control for the DiffusionGemma scale-up.
- Tuning LLM Judge Design Decisions for 1/1000 of the Cost,
  OpenReview:cve4NOiyVp / arXiv:2501.17178 - mapped to tuned LLM-judge
  comparator and cost-normalized moat accounting.
- When To Solve/Verify, arXiv:2504.01005 - mapped to the fixed-budget
  solve-versus-verify normalization bar.
- ThinkPRM, arXiv:2504.16828 - mapped as the high-quality but expensive
  process-verifier comparator.
- CEM, arXiv:2510.20607 - re-flagged to the operator only:
  `operator_authorization_required=true`, `auto_activation_recommended=false`,
  retirement marker `gap3_trained_content_energy_selector_retired_stage2v2_2026_06_09`.

cem_operator_authorization_flag:
`source_id=2510.20607; operator_authorization_required=true; auto_activation_recommended=false; retirement_marker=gap3_trained_content_energy_selector_retired_stage2v2_2026_06_09`.

flagged_for_v389:
`s3_diffusiongemma_verifier_guided_search_scaleup_v389`.

Flagged for .389: `s3_diffusiongemma_verifier_guided_search_scaleup_v389`.

**Bottom line for the .389 roadmap:** run the S^3-style DiffusionGemma
verifier-guided denoising search first, with Reward-Guided Stitching and
Self-Rewarding SMC as ablation/control arms and judge-cost normalization around
the efficiency moat. Keep CEM on the operator surface only; do not activate it
until operator authorization is granted and gate-1R is passed.
## 2026-06-14 Exp 4180 - .387 planning sweep SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-moat-gap3-diffusion-v388-2026-06-14.md`.

**Filtered track:** verifier-as-reward, sanitized headroom, accuracy-and-cost
moat framing, GAP-3 learned ARC energy, TRM vote/headroom decomposition, and
DiffusionGemma guidance for the `.388` handoff.

**Seed and fresh-pass candidates marked ingested:**
- Unsolvability Ceiling, arXiv:2605.07395 - mapped to the A1 headroom-gate
  sanitization already applied; it is a measurement guard, not a verifier.
- When To Solve/Verify, arXiv:2504.01005 - mapped to A3 accuracy-and-cost
  reporting against self-consistency.
- ThinkPRM, arXiv:2504.16828 - mapped to A3 as the high-quality but expensive
  process-verifier comparator.
- Generalizable Reasoning through Compositional Energy Minimization,
  arXiv:2510.20607 - mapped to GAP-3 Stage-2 compositional ARC energy and
  flagged as the strongest `.388` follow-on.
- Self-Rewarding SMC, arXiv:2602.01849 - mapped to the queued DiffusionGemma
  particle-guidance template after a positive energy gate.
- TRM ARC-AGI-1 ablation, arXiv:2512.11847 - mapped to the TRM headroom/vote
  decomposition and identity-conditioning control.

flagged_for_v388:
`cem_gap3_stage2_compositional_arc_energy_v388`.

Flagged for .388: `cem_gap3_stage2_compositional_arc_energy_v388`.

**Bottom line for the .388 roadmap:** run the CEM-style GAP-3 Stage-2
compositional ARC energy prototype first. Keep A1/A3 as mandatory gates and use
Self-Rewarding SMC only for DiffusionGemma guidance once the energy gate is positive.
## 2026-06-13 Exp 4170 - .387 verifier-moat guidance SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-verifier-moat-guidance-v387-2026-06-13.md`.

**Filtered track:** verifier-as-reward, accepted/rejected trace selection, and
energy-guided generation for the `.387` handoff. This ingestion keeps
DiffusionGemma guidance queued because Exp 4168 recorded
`verifier_value_added=false` from a deferred, unfaithful/still-training
baseline rather than from a tested positive or negative guidance result.

**Seed and fresh-pass candidates marked ingested:**
- TRM, arXiv:2510.04871 - mapped as the faithful baseline and oracle-headroom
  gate before any verifier or diffusion-guidance claim.
- TTA-TRM, arXiv:2511.02886 - mapped as the same-budget no-verifier adaptation
  control.
- V-STaR, arXiv:2402.06457 - mapped as the accepted/rejected trace selector and
  strongest `.387` next step.
- SEDD, arXiv:2310.16834 - mapped as the discrete score/energy scaffold for
  generation-time verifier guidance.
- Classifier-guided diffusion, arXiv:2105.05233, and classifier-free guidance,
  arXiv:2207.12598 - mapped as the external-energy precedent and internal-score
  control.
- EntRGi, arXiv:2602.05000 - mapped as the queued DiffusionGemma reward-guidance
  template after a positive verifier-discrimination gate.
- EDLM, arXiv:2410.21357 - mapped as the internal sequence-energy comparator
  for any future guidance claim.

flagged_for_v387:
`vstar_rejected_trace_selector_headroom_gate_before_diffusiongemma_v387`.

Flagged for .387: `vstar_rejected_trace_selector_headroom_gate_before_diffusiongemma_v387`.

**Bottom line for the .387 roadmap:** build the V-STaR-style rejected-trace
selector and headroom gate first. Keep EntRGi/DiffusionGemma guidance queued
unless the verifier discrimination gate flips positive.
## 2026-06-13 Exp 4162 - .386 verifier-moat guidance SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-verifier-moat-guidance-2026-06-13.md`.

**Filtered track:** verifier-vs-self-consistency, reward-guided generation,
and ARC-AGI-3 action efficiency for the `.386` handoff. This ingestion extends
the `.385` verifier moat and queued DiffusionGemma gate without duplicating the
prior TRM/TTA-TRM/V-STaR/SEDD/CFG milestone ingestion.

**Seed and fresh-pass candidates marked ingested:**
- ARBITER, arXiv:2605.26172 - mapped as the wrong-majority/rerank-recovery
  moat anchor and the reason to aggregate an external verifier with vote.
- ThinkPRM, arXiv:2504.16828 - mapped as the data-efficient process-verifier
  existence proof and LLM-judge comparison bar.
- Optimal LLM+PRM Aggregation, arXiv:2510.13918 - mapped as the calibrated
  vote-plus-verifier aggregation recipe.
- RLV, arXiv:2505.04842 - mapped as the cheap verifier/value-head efficiency
  head-to-head template.
- EntRGi, arXiv:2602.05000 - mapped as the discrete diffusion reward-guidance
  template for DiffusionGemma after a positive discrimination gate.
- Executable World Models for ARC-AGI-3, arXiv:2605.05138, and ARC-AGI-3 tech
  report, arXiv:2603.24621 - mapped as executable transition verification and
  action-efficiency anchors.

Flagged for .386: `entrgi_diffusiongemma_energy_guidance_after_positive_discrimination_gate_v386`.

**Bottom line for the .386 roadmap:** run the EntRGi-style DiffusionGemma
energy-guidance template only after the verifier-discrimination gate is
positive. If the gate is not positive, run the RLV-style cheap
energy-verifier-vs-LLM-judge efficiency head-to-head first.
## 2026-06-13 Exp 4152 - .385 recursive-reasoner/verifier energy-guidance SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-recursive-reasoner-verifier-energy-guidance-2026-06-13.md`.

**Filtered track:** verifier-guided training plus energy-guided generation for
the `.385` handoff. This connects the TRM/TTA/V-STaR recursive verifier stack
to the queued DiffusionGemma energy-guidance use without treating a generator
substrate as verifier evidence.

**Seed and fresh-pass candidates marked ingested:**
- TRM, arXiv:2510.04871 - mapped as the `nano-trm` baseline and oracle-headroom
  gate before any verifier-guided or diffusion-guided claim.
- TTA-TRM, arXiv:2511.02886 - mapped as the same-budget adaptation-control arm
  that prevents full fine-tuning from masquerading as verifier reward.
- V-STaR, arXiv:2402.06457 - mapped as the accepted/rejected trace selector for
  saved `nano-trm` candidates before another generator pass.
- SEDD, arXiv:2310.16834 - mapped as the discrete diffusion score/energy
  formalism for generation-time verifier guidance.
- Classifier-guided diffusion, arXiv:2105.05233, and classifier-free diffusion
  guidance, arXiv:2207.12598 - mapped as the external-guidance precedent and
  no-external-verifier control.
- DiffusionGemma official docs, https://ai.google.dev/gemma/docs/diffusiongemma
  - mapped as the queued open-weight block-diffusion substrate, gated on
  measured Carnot-verifier discrimination.

Flagged for .385: `diffusiongemma_sedd_verifier_energy_guidance_probe_v385`.

**Bottom line for the .385 roadmap:** run the DiffusionGemma/SEDD
verifier-energy-guidance probe only if the verifier discrimination gate is
positive; otherwise keep improving the V-STaR-style trace selector and
candidate diversity before spending on guided-generation probe.

## 2026-06-13 Exp 4141 - .383 recursive-reasoner/verifier SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-recursive-reasoner-verifier-2026-06-13.md`.

**Filtered track:** recursive reasoner generator choice plus verifier-as-reward
mapping for the `.383` decisive graft. This follows the Exp 4130 resumable
training ingestion and the Exp 4139 graft receipt, which currently reports
`verifier_value_added=false`, `headroom_present=false`, and
`complete: uninformative_no_headroom_false_negative_risk`.

**Seed and fresh-pass candidates marked ingested:**
- GRAM, arXiv:2605.19376 - mapped as the stochastic-latent generator to graft
  onto in `.384` only if a verifier-value/headroom gate is met.
- Thinking Reward Model for complex reasoning, arXiv:2602.08498 - mapped as
  the RLVR/GRPO precedent for isolating verified-correct trace quality from
  outcome correctness, directly informing the `.383` RFT de-confound.
- Weaver, arXiv:2506.18203 - mapped as the weighted weak-verifier ensemble
  precedent for the `.383` non-oracle ensemble-rerank headline.

Flagged for .384: `gram_as_generator_if_verifier_value_added_and_headroom_present_v384`.

**Bottom line for the .384 roadmap:** use GRAM as the next generator only if
the verifier side first demonstrates transferable value with measurable
oracle(best-of-K) headroom; otherwise continue fixing headroom/candidate
diversity, not as an unconditional rerank claim.

## 2026-06-13 Exp 4130 - .382 resumable-training SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-resumable-training-2026-06-13.md`.

**Filtered track:** checkpoint resume, LR-schedule continuity, and
long-horizon accumulation over the `nano-trm` plus Carnot stack. This follows
the Exp 4121 `.381` baseline-graft ingestion and narrows `.382` to the runner
discipline needed before another verifier-search or training claim.

**Seed and fresh-pass candidates marked ingested:**
- PyTorch Lightning checkpoint resume docs - mapped as the full-state
  `ckpt_path` gate because Lightning checkpoints carry optimizer and LR
  scheduler state as well as global step.
- PyTorch saving/loading docs - mapped as the fallback optimizer-state
  checkpoint contract for any non-Lightning runner.
- Lightning gradient-accumulation docs - mapped as the long-horizon accounting
  rule: count optimizer steps and effective batch size, not microbatches.
- TRM, arXiv:2510.04871 - mapped as the resumed long-horizon baseline whose
  Sudoku evidence must be accumulated by checkpoint lineage and optimizer step.
- TTA-TRM, arXiv:2511.02886 - mapped as the bounded full-fine-tune control that
  must share the same resumed scheduler receipts as any verifier-admitted arm.

Flagged for .383: `lightning_full_state_lr_scheduler_resume_gate_for_nano_trm_v383`.

**Bottom line for the .383 roadmap:** first ship a Lightning full-state resume
gate for nano-trm that proves optimizer, LR scheduler, global-step, data
checksum, and gradient-accumulation continuity across two bounded passes. If
that gate fails, do not spend the next run on per-step verifier work.

## 2026-06-13 Exp 4121 - .381 TRM baseline-graft SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-trm-baseline-graft-2026-06-13.md`.

**Filtered track:** resumable TRM Sudoku baseline reproduction plus Carnot
verifier graft, after Exp 4108 produced a checkpointed but partial baseline,
Exp 4109 found no post-hoc verifier lift over vote, and Exp 4111 flagged
in-loop verifier-guided search as the next candidate.

**Seed and fresh-pass candidates marked ingested:**
- TRM, arXiv:2510.04871 - mapped as the resumed Sudoku Extreme baseline gate
  before any verifier-lift claim.
- TTA-TRM, arXiv:2511.02886 - mapped as the full-fine-tuning adaptation control
  that must be isolated from verifier-admission effects.
- Adaptive verifier-guided candidate expansion, arXiv:2602.01070, with VPRM/VPR
  support from arXiv:2601.17223 and arXiv:2605.10325 - mapped as the strongest
  .382 follow-on because post-hoc verifier reranking already tied vote.
- V-STaR, arXiv:2402.06457 - mapped as accepted/rejected Sudoku trace selector
  training once candidate diversity and oracle support exist.
- ReST, arXiv:2308.08998, and STaR, arXiv:2203.14465 - mapped as the resumable
  generate-filter-improve curriculum, with rejected rows retained for selector
  data.

Flagged for .382: `verifier_guided_adaptive_candidate_expansion_over_resumed_trm`.

**Bottom line for the .382 roadmap:** put the executable Sudoku verifier inside
candidate expansion over the resumed TRM checkpoint before spending on selector
or RFT work. Require pass@1 or oracle-support lift over fixed-K vote and Exp
4109 post-hoc verifier rerank; otherwise selector/RFT work should stay blocked.
## 2026-06-12 Exp 4111 - .380 TRM verifier-training SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-trm-verifier-training-2026-06-12.md`.

**Filtered track:** TRM baseline reproduction plus verifier-guided training and
search over the `nano-trm` Sudoku substrate after Exp 4108 produced an honest
partial baseline and Exp 4109 produced an honest post-hoc verifier null.

**Seed and fresh-pass candidates marked ingested:**
- TRM, arXiv:2510.04871 - mapped as the faithful Sudoku Extreme baseline
  reproduction gate before any verifier-lift claim.
- TTA-TRM, arXiv:2511.02886 - mapped as the full-fine-tuning adaptation control
  that must be isolated from verifier-admission effects.
- V-STaR, arXiv:2402.06457 - mapped as accepted/rejected Sudoku trace selector
  training once candidate diversity exists.
- STaR, arXiv:2203.14465, and ReST, arXiv:2308.08998 - mapped as the cached
  generate-filter-improve cadence, with rejected rows retained for selector data.
- Adaptive verifier-guided search, arXiv:2602.01070, with VPRM/VPR support from
  arXiv:2601.17223 and arXiv:2605.10325 - mapped as the next in-loop verifier
  use because Exp 4109 post-hoc reranking tied vote.

Flagged for .381: `verifier_guided_adaptive_sudoku_search_before_training`.

**Bottom line for the .381 roadmap:** move the executable Sudoku verifier into
candidate expansion before spending on another full fine-tune. Require pass@1
or oracle-support lift over fixed-K vote and Exp 4109 post-hoc verifier rerank;
otherwise keep V-STaR and RFT routes blocked.

## 2026-06-12 Exp 4102 - .379 TRM self-training SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-trm-self-training-2026-06-12.md`.

**Filtered track:** verifier-certified RFT over a recursive `nano-trm`/TRM
substrate, with Carnot verifier labels selecting, correcting, or densifying the
training signal.

**Seed and fresh-pass candidates marked ingested:**
- V-STaR, arXiv:2402.06457 - mapped as accepted/rejected TRM trace selector
  training before any second RFT corpus gate.
- STaR, arXiv:2203.14465, and ReST, arXiv:2308.08998 - mapped as the cached
  generate-filter-improve cadence for recursive traces.
- TTA-TRM, arXiv:2511.02886 - mapped as the full-fine-tune substrate and a
  control against attributing adaptation-only gains to the verifier.
- RLVR with imperfect verifiers, arXiv:2510.00915 - mapped as FP/FN-calibrated
  weighting and abstention before verifier-certified RFT.
- VPRM/VPR, arXiv:2601.17223 and arXiv:2605.10325 - mapped as dense
  per-recursion step rewards only after outcome calibration.
- Self-Trained Verification, arXiv:2605.30290 - marked as fresh adjacent
  verifier-training evidence, but deferred behind the cheaper V-STaR trace
  selector because `.379` already emits accepted/rejected TRM traces.

Flagged for .380: `vstar_rejected_trace_selector_for_trm_rft`.

**Bottom line for the .380 roadmap:** build a V-STaR-style selector over the
saved nano-trm candidate pool, require a rerank win against the current Carnot
verifier ordering, and only then let the selector gate a second full-fine-tune
RFT corpus.
## 2026-06-12 Exp 4094 - .378 precision-calibration SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-precision-calibration-2026-06-12.md`.

**Filtered track:** the `.378` verifier-precision / verifier-as-reward headline:
the 0.32 false-positive channel behind ARC certification precision 0.6818, the
Exp 4087 precision rescue to 0.8824 at 0.7143 recall, the blocked Exp 4088/4089
RFT path, and the Exp 4093 OFF-ARC demo-fit precision replay.

**Seed and fresh-pass candidates marked ingested:**
- BARC / Combining Induction and Transduction, arXiv:2411.02272 - mapped as an
  augmentation-consistency filter before RFT corpus admission.
- Noisy Data is Destructive to RLVR, arXiv:2603.16140 - mapped as the stop-rule
  against training through the 0.32 false-positive channel.
- RLVR with imperfect verifiers, arXiv:2510.00915 - mapped as explicit FP/FN
  noise correction and calibration metadata for future RLVR hooks.
- V-STaR, arXiv:2402.06457 - mapped as rejected-trace retention and verifier
  training over accepted/rejected pairs.
- RFT scaling, arXiv:2308.01825 - retained as the simple fine-tuning baseline
  only after clean, diverse positives exist.
- Invisible Leash, arXiv:2507.14843 - retained as the same-pool latent-support
  gate before RFT/RLVR spend.
- Process Supervision-Guided Policy Optimization for Code Generation,
  arXiv:2410.17621, plus CodePRM ACL 2025 - mapped as step-level process reward
  records before sparse reward RL.

**Bottom line for the .379 roadmap:** prioritize
`calibrated_forward_noise_correction_before_rlvr`,
`augmentation_consistency_filter_before_rft_corpus`,
`vstar_rejected_trace_verifier_training`, and
`step_level_process_reward_weighted_sft`; keep
`latent_support_gate_before_rft_spend` as the launch gate so RFT is only used
when correct transforms are already present in the generated pool.

## 2026-06-11 Exp 4081 - .377 verifier-as-reward SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-verifier-as-reward-2026-06-11.md`.

**Filtered track:** the `.377` verifier-as-reward pivot: verifier-certified RFT,
Tulu-3-style RLVR, Invisible Leash latent-vs-absent support gating, dense
process-reward distillation, and RFT/STaR/ReST self-training over the current
three-arm RFT pipeline.

**Seed and fresh-pass candidates marked ingested:**
- Tulu 3, arXiv:2411.15124 - mapped as the open SFT/DPO/RLVR recipe, but gated
  behind a clean verifier-certification label.
- The Invisible Leash, arXiv:2507.14843 - mapped as the latent-vs-absent support
  diagnostic before any RFT/RLVR spend.
- RL vs. Distillation, arXiv:2505.14216 - mapped as the accuracy-vs-capability
  fork and the reason to track pass@k/oracle support, not only pass@1.
- Self-Distilled RLVR, arXiv:2604.03128 - mapped as a later credit-assignment
  upgrade only after external verifier reward direction is clean.
- STaR, arXiv:2203.14465 - mapped as the minimal generate-filter-finetune loop.
- ReST, arXiv:2308.08998 - mapped as the offline generate-filter-improve cadence
  for reusable cached trace pools.
- Verifiable Process Reward Models, arXiv:2601.17223 - mapped as deterministic
  rule-verifier step rewards rather than opaque neural step judges.
- Verifiable Process Rewards for Agentic Reasoning, arXiv:2605.10325 - mapped as
  dense turn/action rewards for long-horizon ARC agent trajectories.

**Bottom line for the .378 roadmap:** prioritize
`latent_vs_absent_precision_gate_before_rft` and
`process_reward_weighted_sft_over_trace_certification`; keep the deconfounded
RFT-correct vs RFT-ablation contrast; only add self-distilled RLVR after the
external verifier reward direction is measured clean.

## 2026-06-11 Exp 4067 - .376 SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-2026-06-11-v376-unsaturated-corpora-and-online-pruning.md`.

**Filtered tracks:** LOCAL-12B oracle-headroom code corpus for the off-ARC
demo-fit verifier transfer measurement, and VERIFIER-GUIDED ONLINE ACTION-PRUNING
for the efficient ARC harness over explore-first + GAP-4.

**Exp 4055 seed flags confirmed actionable for .376:**
- `evalplus_hidden_rescore_fixed_pool` - actionable as the cheap first hidden-test
  gate, but must route upward when oracle headroom is absent.
- `saga_generated_tests_as_discriminator_arm` - actionable after the official
  hidden-score path is stable; use only as generated-test tie-break/explanation.
- `gap4_online_pruner_for_explore_first_arc` - actionable now as soft pruning
  with replay-disabled-on-failure.
- `equivpruner_state_action_cache_for_arc` - actionable now for exact state hashes
  and GAP-4-confirmed equivalence only.

**Seed and fresh-pass candidates marked ingested:**
- LiveCodeBench v6, arXiv:2403.07974, plus current public leaderboard mirror - mapped as the local-12B headroom route after EvalPlus.
- EvalPlus / HumanEval+ / MBPP+, arXiv:2305.01210 - retained as the first fixed-pool hidden rescore gate.
- SAGA / Rethinking Verification for LLM Code Generation, arXiv:2507.06920 - retained as the generated-test discriminator arm.
- Inference-Time Code Selection via Symbolic Equivalence Partitioning, arXiv:2604.06485 - mapped as the bounded functional-equivalence diagnostic.
- ACES, arXiv:2604.03922 - mapped as the same-pass-matrix Arm A++ baseline.
- What If We Allocate Test-Time Compute Adaptively?, arXiv:2602.01070 - mapped as the online PRM-style prune/expand control precedent.
- Update-Free On-Policy Steering via Verifiers, arXiv:2603.10282 - newly mapped as the no-weight-update verifier-steering precedent for GAP-4 action priors.
- Adaptive Test-Time Compute Allocation via Learned Heuristics over Categorical Structure, arXiv:2602.03975 - retained as selective verifier-call allocation over intermediate states.
- EquivPruner, arXiv:2505.16312 - retained as exact state/action equivalence caching before approximate pruning.
- CoT2-Meta, arXiv:2603.28135 - retained as the explicit expand/prune/repair/stop/fallback controller shape.
- DIRECT, arXiv:2606.12402 - marked as a fresh adjacent compute-router citation, useful for budget framing but not a first implementation target.

**Bottom line for the .377 roadmap:** prioritize
`livecodebench_v6_local12b_headroom_route` and
`gap4_soft_prune_replay_for_arc_efficiency`; add
`gap4_equivpruner_exact_state_action_cache` before learned/approximate pruning,
and add `saga_generated_tests_hidden_score_tiebreak` only after official hidden
scores are frozen.

## 2026-06-11 Exp 4055 - .375 SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-2026-06-11-unsaturated-execverif-and-verifier-pruner.md`.

**Filtered tracks:** UN-SATURATED execution-verification corpus for exp4056/4057 off-ARC
demo-fit transfer measurement, and VERIFIER-GUIDED online action-pruning for the efficient
ARC harness over explore-first + GAP-4.

**Seed candidates marked ingested:**
- EvalPlus / HumanEval+ / MBPP+, arXiv:2305.01210 - mapped as the default hidden-test rescore path that fixes the `.374` base HumanEval/MBPP saturation failure.
- LiveCodeBench v6, arXiv:2403.07974 - mapped as the contamination-free escalation corpus if EvalPlus hidden tests lack headroom.
- SAGA / Rethinking Verification for LLM Code Generation, arXiv:2507.06920 - mapped as the generated-test discrimination arm after the fixed-pool EvalPlus path is stable.
- DryRUN / You Don't Need Public Tests to Generate Correct Code, arXiv:2604.21598 - mapped as the public-test-free self-simulation tie-break arm, not as authoritative final scoring.
- What If We Allocate Test-Time Compute Adaptively?, arXiv:2602.01070 - mapped as the online verifier-guided prune/expand control rule for ARC frontier expansion.
- Marco DeepResearch, arXiv:2603.28376 - mapped as the verification-centric budget-ledger precedent for agentic search.
- Pushing Test-Time Scaling Limits of Deep Search with Asymmetric Verification, arXiv:2510.06135 - mapped as the cheap-verifier-vs-expensive-search budget split for GAP-4 pruning.

**Fresh-pass confirmations marked ingested:**
- Adaptive Test-Time Compute Allocation via Learned Heuristics over Categorical Structure, arXiv:2602.03975 - mapped as verifier-cost-limited selective GAP-4 calls over intermediate ARC states.
- EquivPruner, arXiv:2505.16312 - mapped as exact state/action equivalence caching before approximate pruning.
- CoT2-Meta, arXiv:2603.28135 - mapped as the explicit expand/prune/repair/stop/fallback controller shape for explore-first telemetry.
- SEP, arXiv:2604.06485, and ACES, arXiv:2604.03922 - carried forward as same-pool baselines around the EvalPlus measurement, not as the headline fix.

**Bottom line for the .376 roadmap:** prioritize `evalplus_hidden_rescore_fixed_pool`
and `gap4_online_pruner_for_explore_first_arc`; add `saga_generated_tests_as_discriminator_arm`
only after the fixed-pool EvalPlus adapter is stable, and add `equivpruner_state_action_cache_for_arc`
before any learned/approximate pruning.

## 2026-06-11 Exp 4043 - .374 SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-2026-06-11-offarc-power-and-closed-loop-planning.md`.

**Filtered tracks:** OFF-ARC statistical power + stronger discriminator for exp4044/4045, and
CLOSED-LOOP planning over the verified vc33 world model under model error for exp4046.

**Seed candidates marked ingested:**
- Inference-Time Code Selection via Symbolic Equivalence Partitioning, arXiv:2604.06485 - mapped as the SEP semantic-partition tie-break/diagnostic for the full-power off-ARC panel.
- Scaling Agentic Verifier for Competitive Coding, arXiv:2602.04254 - mapped as the expensive targeted-counterexample comparator for hard ties.
- Efficient Prediction of Pass@k Scaling, arXiv:2510.05197 - mapped as the pilot sizing and budget discipline for HumanEval+MBPP power.
- What model does MuZero learn?, arXiv:2306.00840 - mapped as the policy-support / WM-trust constraint for vc33 search.
- World-in-World, arXiv:2510.18135 - mapped as the closed-loop task-success evaluation rule.
- Latent Geometry Beyond Search / GC-IDM, arXiv:2605.08732 - mapped as per-step replanning and action-prior guidance.
- Bounding Distributional Shifts through Novelty Detection, arXiv:2508.06096 - mapped as the novelty-MPC trust gate against WM exploitation.

**Fresh-pass confirmations marked ingested:**
- DOCE, arXiv:2408.13745 - retained as the execution-based code-selection protocol anchor for the powered measurement.
- CodeT, arXiv:2207.10397 - retained as the dual execution-agreement baseline alongside ACES.
- ACES, arXiv:2604.03922 - promoted as the strongest same-pass-matrix Arm A++ baseline.
- R-WoM, arXiv:2510.11892 - mapped as retrieval grounding over verified transition traces for short-lookahead vc33 planning.

**Bottom line for the .375 roadmap:** prioritize `offarc_full_power_sep_aces_agentic_counterexample_panel`
and `closed_loop_vc33_replan_with_wm_trust_gate`; add `novelty_mpc_gate_for_verified_wm_search`
as the required guardrail so the planner cannot claim model-only goal satisfaction as a solve.

## 2026-06-11 Exp 4030 - .373 SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-2026-06-11-offarc-and-search.md`.

**Filtered tracks:** OFF-ARC execution-consistency verifier transfer for exp4031/4032, and
hierarchical/subgoal search over a verified world model for exp4034/4035.

**Seed candidates marked ingested:**
- DOCE, arXiv:2408.13745 - mapped as the exp4031/4032 protocol anchor.
- ACES, arXiv:2604.03922 - mapped as the stronger same-pass-matrix consistency baseline.
- Symbolic Equivalence Partitioning, arXiv:2604.06485 - mapped as a bounded diagnostic/enrichment layer.
- Scaling Agentic Verifier, arXiv:2602.04254 - mapped as an expensive comparator/escalation path.
- Iterative Self-Training via Reinforced Re-Ranking, arXiv:2504.09643 - mapped as a future distillation target only.
- Hierarchical Planning with Latent World Models, arXiv:2604.03208 - mapped as symbolic hierarchical MPC over vc33.
- Subgoal-Guided PHS, arXiv:2506.07255 - mapped as failed-tree subgoal mining/logging for exp4035.
- Sokoban HRL with Landmarks, arXiv:2504.04366 - mapped as feature-level landmark decomposition precedent.

**Fresh-pass additions marked ingested:**
- Policy-Guided Heuristic Search with Guarantees, arXiv:2103.11505 - mapped as policy+heuristic priority extension for current best-first search.
- Structure-Induced Information for Rerooting Levin Tree Search, arXiv:2605.30664 - mapped as the .374 fallback if explicit subgoals are too costly.

**Bottom line for next roadmap:** prioritize `off_arc_demo_fit_vs_aces_and_doce_protocol` and
`vc33_hierarchical_subgoal_search_with_rerooting_fallback`; keep SEP as a diagnostic, not a gate.

## Sweep 2026-05-24T15:20Z (Claude outer-loop, infrastructure recovery)

**Recovery summary:**

The 2026-05-16 sweep entry below reported the infrastructure as
"2-channel-down": Bash dead, arXiv API HTTP 429, fallback URL
returning 404. The 2026-05-24 outer-loop probe ran a direct
verification:

| Check | Pre-recovery | Post-recovery |
|---|---|---|
| Bash exit code | exit 1 (every shell) | exit 0 (working) |
| arXiv listing URL (`/list/cs.LG/2605`) | HTTP 404 | still 404 — pattern was stale |
| arXiv listing URL (`/list/cs.LG/2026-05`) | not tested | **HTTP 200 — the correct pattern** |
| arXiv listing URL (`/list/cs.LG/recent`) | not tested | **HTTP 200 — alternate fallback** |
| arXiv API direct (cluster URL) | HTTP 429 | HTTP 200 after redirect-follow (HTTPS, `-L`) |
| HN Algolia API | 0 hits at search-time | HTTP 200 (channel itself live) |

**The fix:** the cron-prompt URL pattern `arxiv.org/list/cs.LG/2605`
should be replaced with `arxiv.org/list/cs.LG/2026-05` (calendar-month
format) or `arxiv.org/list/cs.LG/recent` (alternate fallback). The
arXiv API itself works fine via HTTPS with redirect-following; the
earlier 429s appear to have been transient rate-limiting, not a
permanent block. Cron-prompt URL update remains operator-owned but
is now a documented one-line edit.

**Fresh sweep results (all 5 clusters, 4 top hits each):**

Cluster 0 — verifier ensemble / spec gaming / reward hacking:
- arXiv:2605.21384 (2026-05-20) **SpecBench: Measuring Reward Hacking
  in Long-Horizon Coding Agents.** Directly Carnot-adjacent; benchmark
  for the exact failure mode our verifier-authenticity discipline
  catches. **Promote to research-references.md.**
- arXiv:2605.20744 (2026-05-20) **Hack-Verifiable Environments:
  Towards Evaluating Reward Hacking at Scale.** Adversarial-verify
  discipline analog at the environment level. **Promote.**
- arXiv:2605.22620 (2026-05-21) Two is Better Than One: Collapse-free
  Multi-Reward RLIF Training Framework. AND-composed verifier
  ensemble structurally adjacent. Track.

Cluster 1 — EBM / energy-guided LLM:
- arXiv:2605.14558 (2026-05-14) **Resolving Action Bottleneck:
  Agentic Reinforcement Learning Informed by Token-Level Energy.**
  Token-level energy as RL signal; direct EBM-as-policy framework.
  **Promote.**

Cluster 2 — SAE / probes / interpretability:
- arXiv:2605.22462 (2026-05-21) From Correlation to Cause: Five-Stage
  Methodology for Feature Analysis in Transformer Language Models.
  Methodology paper, could inform adversarial-verify discipline.
- arXiv:2605.20868 (2026-05-20) Runtime-Certified Bounded-Error
  Quantized Attention. Relevant to the RotorQuant conversation.

Cluster 3 — active inference / FEP / world model:
- arXiv:2605.22675 Self-Policy Distillation via Capability-Selective
  Subspace Projection — possible relevance to FR-11 attractor work
  (.282 exp3007). Track.
- Other cluster-3 hits this round were medical imaging / video / table
  recognition false positives. Cluster-3 narrowing may be warranted.

Cluster 4 — sub-quadratic / analog / FPGA:
- arXiv:2605.17720 (2026-05-18) **ROA-Based Subharmonic Injection
  Locking for Oscillator-Based Ising Machines.** Hardware Ising
  substrate evolution; future production target candidate.
  **Promote.**
- arXiv:2605.19399 (2026-05-19) HSCO-Bench: Agent-Driven End-to-End
  Hardware-Software Co-design Benchmark for SoCs. Track.

**Result:** 20 fetched / dedupe not yet run / 4 promoted to
references.md / 5 watched / 11 tangential or false-positive.
Sweep infrastructure operational again; future planner passes can
invoke sweep_clusters.py + arXiv API directly without intervention.

### Sweep takeaways

1. **The URL-pattern bug was the real annoyance** — Bash recovered
   on its own, the API was never permanently rate-limited, but the
   stale cron-prompt URL kept producing 404s. The fix is a single
   character class change: `2605` → `2026-05`.
2. **The 5-cluster fan-out works at high signal-to-noise** —
   clusters 0, 1, 4 produced highly Carnot-relevant papers; cluster
   2 produced one solid + one tangential; cluster 3 mostly false
   positives this round.
3. **Cluster 3 may need re-narrowing** — "active inference" + "free
   energy" without a strong AND-anchor surfaces too many adjacent
   domains (medical, vision, table recognition). Worth tightening
   when next operator-edits the cron prompt.
4. **The planner has been pulling references organically** through
   plan-next-milestone passes (7 new arXiv entries in research-
   references.md since 2026-05-21 — see the metamorphic-testing
   cluster + attractor-reasoning cluster). Sweep infrastructure
   being broken did NOT stop literature integration; it just made
   it less systematic.

## Sweep 2026-05-16T08:00Z (Claude outer-loop, 2-channel-down + Bash dead)

**Environment constraint:** outer-loop Bash still broken (every shell
exits 1); `sweep_dedupe.py` + `sweep_citations.py` etc. NOT INVOCABLE.

**Queries fired (degraded):**
- arxiv cluster 0 (verifier-ensemble / null-space / spec-gaming) →
  **HTTP 429 Too Many Requests**.
- arxiv cluster 3 (active-inference / free-energy / LLM) →
  **HTTP 429 Too Many Requests**.
- arxiv.org/list/cs.LG/2605 (cron-prompt fallback) →
  **HTTP 404 Not Found** (URL pattern may be stale).
- hn.algolia "verifier energy LLM" → 0 hits.

**Result: 0 fetched / 0 dedupe-skipped / 0 scored / 0 promoted.**

The arxiv API rate-limit on 2 consecutive queries plus the fallback
listing also unavailable means this sweep has zero candidate-fetch
capacity. Honest report: nothing surfaced.

### Sweep takeaways

1. **The cron-prompt fallback URL is stale.** `https://arxiv.org/list/cs.LG/2605`
   returns 404. Working format may be `cs.LG/2026-05` or `cs.LG/26.05`.
   Worth verifying when shell recovers and updating the cron prompt.
2. **arxiv API rate-limit hits confirm operator-discussed broadened
   queries would help** (more results per request → fewer requests).
   Cron-prompt URL update remains operator-owned.
3. **5 consecutive sweep cycles in degraded state** (Bash dead + now
   API rate-limited). Net new candidates this period: 1 (CoT2-Meta
   Score 320 from the 04:35Z citation-sweep window before API
   throttling hit). Routine keyword-rotation channel exhausted +
   cannot open new channels.

## Sweep 2026-05-16T06:50Z (Claude outer-loop, degraded environment — WebFetch+Edit only)

**Environment constraint:** outer-loop Bash failed earlier in this session
(every shell command returns exit 1 with no output). Sweep helpers
(`sweep_dedupe.py`, `sweep_citations.py`, etc.) NOT INVOCABLE. Manual
dedupe against known-set via Edit-tool memory only. Conductor's
auto-commit chain will sweep up this file edit even though outer-loop
git operations are blocked.

**Queries fired (hour-mod-4=2; cluster 2 primary):**
- arxiv abs:"sparse autoencoder" OR "white box probe" OR "reconstruction
  error" AND "LLM" → 8 fetched
- (cluster 0/1/3 skipped — Bash blocker means I can't pipe through
  dedupe efficiently; one cluster is sufficient to confirm saturation)
- HN skipped (5+ prior 0-hits today)

**Manual dedupe (no `sweep_dedupe.py --filter` available):**

All 8 cluster-2 IDs are already in research-studying.md from prior
sweeps (2605.14694, 2605.14449, 2605.14347, 2605.13930, 2605.12874,
2605.12809, 2605.12770, 2605.12245). Verified by memory of the
13-prior-sweep catalogue.

**Result: 8 fetched / 8 known-skipped / 0 scored / 0 promoted.**

This is the third 100%-saturation sweep in 24 hours (the pattern shipped
2026-05-16T00:40Z, again at 04:35Z when keyword rotation produced 0 but
routine citation-sweep on ODAR anchor surfaced CoT2-Meta Score 320, and
now). The auto-rotation channel has fully mapped the recent SOTA window.

### Sweep takeaways

1. **Continued saturation validates the operator-approved trickle policy.**
   Queue backlog (Phase 1 prongs awaiting `.198 outcomes + operator
   PyPI approval + CoT2-Meta routing experiment + recovered audit
   tasks in `.200) needs absorption time before new literature can
   productively layer on.
2. **Bash-tool blocker is the dominant constraint this sweep.** With
   `sweep_dedupe.py` and `sweep_citations.py` unavailable, citation-
   sweep depth-following (the high-yield channel from the morning's
   04:35Z sweep) couldn't fire. Next sweep with a working shell
   should run routine citation-sweep on CoT2-Meta or any other
   high-score anchor not yet depth-swept.
3. **Operator-flagged anchors remain the highest-yield channel.** The
   sweep helper suite, even when fully functional, beats keyword
   rotation primarily through citation-following. The keyword-rotation
   pure-keyword channel has converged to ~0 promotions per fire.

## Sweep 2026-05-16T04:35Z (Claude outer-loop, routine citation-sweep on ODAR anchor — cadence rule)

**Helpers used (per the routine-citation cadence shipped 2026-05-16T01:00Z):**
- `sweep_citations.py 2602.23681` (ODAR anchor, Score 400; not yet
  depth-swept since promotion at 21:30Z). Returned ~15 unique IDs.
- `sweep_citations.py 2605.12484` (Fast-Slow anchor, Score 400) —
  still 404 not-indexed in Semantic Scholar (paper too new at 4 days
  old at promotion + 14 days now).
- `sweep_dedupe.py --filter` (default workflow step 2.5).

**Result: ~15 fetched / dedupe-skipped 13 / 2 scored / 1 promoted.**

### NEW Rank HIGH: CoT2-Meta — Budgeted Metacognitive Control for Test-Time Reasoning (arXiv:2603.28135, Mar 30 2026)
- **Score:** 5×4×4×4 = **320**
- **Authors:** Siyuan Ma, Bo Gao, Zikai Xiao, Hailong Wang, Xinlei Yu,
  Rui Qian, Jiayu Qian, Luqi Gong, **Yang Liu** (same Ma/Gao/Liu line
  as ODAR)
- **Why it matters:** Training-free framework integrating CoT generation
  with metacognitive control decisions — **"expansion, pruning, repair,
  stopping, fallback decisions"** for budgeted computational allocation
  during reasoning. This is structurally an explicit ROUTING/ORCHESTRATION
  framework over a verify-repair-stop loop: "expand" = generate
  candidates, "prune" = verifier rejection, "repair" = exactly what
  Carnot does, "stopping" = ODAR-style fast-path acceptance, "fallback"
  = deliberative path. Same author group as ODAR (Ma/Gao/Liu) suggests
  a coherent research program; CoT2-Meta is the operational framework,
  ODAR is the routing-mechanism within it.
- **Action:** Cite alongside ODAR (Score 400) in paper-v6 §3 as evidence
  for the metacognitive-control architecture pattern. Concrete .197+
  proposal: extend the Carnot ODAR routing (.190 exp1822 queued) with
  the explicit expand/prune/repair/stop/fallback state-machine from
  CoT2-Meta. Carnot's verify-repair loop currently lacks an explicit
  "fallback" path — the LLM either passes the verifier or iterates;
  the fallback (e.g., escalate to k=16 ensemble disagreement check or
  human review) is implicit. CoT2-Meta provides the formalism.
- **Cross-reference:** complements arXiv:2602.23681 ODAR (Score 400)
  + arXiv:2605.12484 Fast-Slow (Score 400). Three coherent peer
  anchors from the same year on the routing/orchestration theme.

### Other scored (no promotion):

- **arXiv:2511.09873** HierRouter — Coordinated Routing of Specialized
  LLMs via RL (Gupta/Guo/Kannan/Prasanna, Nov 2025). Score 3×3×2×2 =
  **36**. Adjacent (LLM routing) but RL-trained hierarchical routing
  is high-cost to replicate; out-of-scope for Carnot's training-free
  verify-repair architecture.

### Pre-2026 references catalogued (no scoring; archived for citation tracking)

~13 papers in the ODAR references-direction sweep span Feb 2025
through Nov 2025 (foundational work on routing, reasoning RL, planning).
These are now in the known-set for future citation-sweep dedupe.

### Sweep takeaways

1. **Routine citation-sweep cadence (every 4th keyword sweep) earns
   its keep.** Today: keyword rotation at 100% saturation; citation-sweep
   surfaced 1 Score-320 promotion. CoT2-Meta would NEVER have surfaced
   via the 4 cluster queries — its abstract uses "metacognitive control"
   and "expansion / pruning" not "verifier ensemble" / "energy based
   model" terms.
2. **Author-cluster signal:** Ma/Gao/Liu have published two highly-
   relevant Carnot-adjacent papers (ODAR Feb 2026 + CoT2-Meta Mar 2026).
   This is a research-program convergence worth following — citation-
   sweep on CoT2-Meta in a future cycle may surface their newer work.
3. **arXiv:2605.12484 Fast-Slow still not S2-indexed** (14 days old);
   citation-sweep yield decays sharply with anchor age below 30 days.
   Patience expected for the May 2026 anchors.
4. **The fast-slow / ODAR / CoT2-Meta cluster** is now a coherent
   sub-literature anchor for paper-v6 §3 architecture-validation. All
   three converge on the same fast-deliberative routing pattern that
   Carnot's verify-repair loop implements. The four-anchor triangulation
   (bijection + Fast-Slow + ODAR + CoT2-Meta) is becoming a five-anchor
   triangulation with inference-time-planning (2602.02991) as the fifth.

## Sweep 2026-05-16T00:40Z (Claude outer-loop, light fire — saturation acknowledgment)

**Queries fired (light fire per operator-approved trickle policy):**
- arxiv abs:"active inference" OR "free energy" AND "LLM" → 5 fetched (cluster 3)
- arxiv abs:"sparse autoencoder" OR "white box probe" OR "reconstruction error" AND "LLM" → 8 fetched (cluster 2)

**Dedupe filter:**
13 candidates fetched, **all 13 known-skipped** (100% filter rate).
64 IDs now in known-set (up from 53 at 16:50Z fire). The keyword
rotation has fully saturated against research-studying.md's catalogue.

**Result: 0 NEW / 0 scored / 0 promoted.**

This is the expected outcome per the operator-approved trickle policy.
The queue's backlog (Fast-Slow Variant adversarial confirmation
pending in .192 exp1837, ODAR Score 400 awaiting .193+ integration,
Inference-Time Planning Score 144 cite still pending, PyPI ship-track
about to unblock via tag-push) needs experimental absorption time
before new literature can be productively layered on.

### Sweep takeaways

1. **100% dedupe filter rate is healthy at this point in the cycle.**
   Carnot's literature landscape is well-mapped relative to the
   4-cluster query rotation's reach. New high-impact findings will
   come from: (a) operator-flagged anchors (like the 13:15Z Fast-Slow
   promotion), (b) citation-following from existing high-score nodes
   when triggered, OR (c) cron-prompt URL update to the broadened
   cluster queries (sweep_clusters.py output) when operator decides.
2. **No helper-extended sweep this fire.** Citation-sweep + semscholar
   helpers remain available; reserving them for either (a) a specific
   operator-flagged research question or (b) post-.192 retro when
   exp1837 confirmation result drives new follow-up direction.
3. **Saturation confirms the queue's high-impact items are real.**
   When the auto-rotation finds 0 new across 13 fetched, it means
   the queue's existing entries (Fast-Slow, ODAR, bijection,
   inference-time-planning) ARE the SOTA — not gaps in coverage.

## Sweep 2026-05-15T21:30Z (Claude outer-loop, 4-helper combined: citations + semscholar)

**Helpers exercised:**
- `sweep_citations.py 2604.07650` (Behavioral Entanglement anchor, Score 400, Apr 2026) — 38 unique IDs (mostly older references).
- `sweep_citations.py 2605.02269` (Spec Gaming anchor, Score 320, May 2026) — 20 unique IDs.
- `sweep_citations.py 2605.14449` (QAOD anchor, Score 320) — 404 not-indexed.
- `sweep_citations.py 2602.18671` (Spilled Energy, Feb 2026) citations-only — 0 hits (no arxiv-mirrored citations yet).
- `sweep_semscholar.py "fast-slow LLM verifier energy"` — 9 unique IDs.
- `sweep_dedupe.py --filter` (default workflow step 2.5).

**Result: ~70 fetched across helpers / ~50 known-skipped via dedupe / ~20 newly-scored / 1 promoted (Score 400, in-domain critical).**

### NEW Rank URGENT (TIED with Spec Gaming + QAOD at 320 → 400 now): ODAR — Free-Energy-Principled Adaptive Routing for LLM Reasoning (arXiv:2602.23681, Feb 27 2026)
- **Score:** 5×4×4×5 = **400**
- **Authors:** Siyuan Ma, Bo Gao, Xiaojun Jia, Simeng Qin, Tianlin Li,
  Ke Ma, Xiaoshuang Jia, Wenqi Ren, Yang Liu
- **Why it matters CRITICAL:** Adaptive routing for LLM reasoning that
  dynamically allocates compute between FAST AND DELIBERATIVE agents
  using **active inference + free-energy principle**. Employs "a
  free-energy-principled, risk-sensitive fusion mechanism" to select
  answers while balancing likelihood with epistemic uncertainty.
  **This directly merges Carnot's Phase 4 (active inference) track
  with the Fast-Slow Variant (.189 exp1811) track** — the two have
  been parallel until now, but ODAR demonstrates they're the SAME
  mechanism viewed from different sides:
    - Phase 4: free-energy reduction = alpha_t (Carnot's target metric,
      ensemble-output substrate-inaccessible per exp1745)
    - Fast-Slow Variant: fast-weight context shaped by slow-weight
      verifier ensemble (exp1811's design)
    - ODAR: free-energy routing between fast/deliberative agents on
      the same answer-space (Ma et al. closed-form mechanism)
- **Empirical strength:** Tested across **23 benchmarks** with
  "reduced computational overhead compared to uniform sampling
  strategies." This is multi-benchmark evidence at a scale Carnot
  hasn't yet achieved on Phase 4.
- **Action — high-leverage:**
  1. ops/known-issues.md MANDATORY entry: ODAR routing mechanism
     should inform Phase 4 canonical-metric decision (.189 exp1814 OR
     a downstream task).
  2. Paper-v6 §3: ODAR is the FOURTH independent peer-reviewed
     anchor for Carnot's design pattern. Quadruple triangulation now
     (bijection 500 + Fast-Slow 400 + ODAR 400 + inference-time-planning
     144).
  3. Concrete .190+ proposal: "Carnot ODAR-style Routing" — adopt
     the free-energy-principled risk-sensitive fusion in place of
     Carnot's current verify-repair argmax selection. Acceptance gate:
     match ODAR's "reduced computational overhead" claim relative to
     uniform-iteration verify-repair on a 30-example reasoning corpus.
- **Cross-references:** complements arXiv:2605.12536 (IIT↔FEP
  maximum-caliber bridge, the basis of Carnot's alpha_t' replacement
  derivation in exp1721). ODAR is the OPERATIONAL counterpart to
  exp1721's theoretical derivation.

### Other newly-scored (no promotion):

- **arXiv:2604.01681** Agentic Fast-Slow Planning for AVs (Chen et al.,
  Apr 2026). Score 2×4×2×2 = **32**. Autonomous-vehicle hierarchical
  planning; out-of-domain for Carnot's LLM verification.
- **arXiv:2603.22866** Aerial Agentic AI (Dong et al., Mar 2026). Score
  1×3×1×1 = **3**. UAV wireless networks, out-of-domain.
- **arXiv:2604.12185** Order-Aware Hypergraph RAG (Wu/Kuai et al., Apr
  2026). Score 2×3×2×2 = **24**. RAG with order-aware knowledge
  representation; adjacent to structural verification but not core.
- **arXiv:2601.03267** OpenAI GPT-5 System Card (OpenAI team, Dec 2025
  v1 / May 2026 v2). Score 3×3×1×2 = **18**. Production-deployment
  baseline reference; Carnot doesn't have GPT-5 access. Noted as
  reference material.

### Sweep takeaways

1. **semscholar channel produced the day's highest-yield single
   helper invocation** — 9 IDs fetched, 1 Score-400 promotion. By
   contrast: prior 2 keyword-rotation sweeps (16:50Z, 20:35Z) found
   0 promotions each; 21:15Z citation-sweep found 1 Score-144
   promotion.
2. **ODAR is the day's most strategic literature finding.** The
   Phase 4 program has been pursuing alpha_t measurement across 5
   experiments (exp1715/1721/1741/1745/1811) without convergence;
   ODAR demonstrates that a DIFFERENT free-energy-derived target
   (routing mechanism, not metric measurement) succeeds across 23
   benchmarks. This may be the rescue path Phase 4 has been blocked
   on.
3. **Citation-following yield decays with anchor age:** Dec 2025
   anchor → 39 unique IDs; Feb 2026 → 38; Apr 2026 → 38 (mostly older
   references); May 2026 → 404 (too new). The older anchors provide
   archival depth; newer anchors need different methods. semscholar
   keyword search complements citation-following at the SOTA edge.
4. **Quadruple peer-reviewed triangulation** of Carnot's verify-repair
   architecture pattern: arXiv:2512.15605 (bijection) + arXiv:2605.12484
   (Fast-Slow) + arXiv:2602.23681 (ODAR) + arXiv:2602.02991 (inference-
   time planning). Paper-v6 §3 has substantial peer-review backing now.

## Sweep 2026-05-15T21:15Z (Claude outer-loop, citation-following + broadened-cluster attempt — FIRST extended-window sweep)

**Helpers used (per the suite shipped 21:00Z):**
- `scripts/sweep_citations.py 2512.15605 --direction both` (AR-LM↔EBM
  bijection anchor, Score 500 — highest-scoring active queue entry).
- `scripts/sweep_dedupe.py --filter` (default workflow step 2.5).
- `scripts/sweep_clusters.py` broadened queries on clusters 1 + 2 —
  **arXiv API returned HTTP 429 Too Many Requests** on the broadened
  cluster URLs (max_results=20 + complex OR-chains hit rate-limit
  thresholds). Citation-sweep alone carried this fire.

**Result: 39 fetched / 0 dedupe-skipped / 39 scored / 1 promoted (in-domain).**

Citation-sweep returned 39 unique arxiv IDs from the bijection anchor's
references + citations. **All 39 NEW to the queue** (0 known-skipped —
the keyword rotation has been blind to all of these because the topical
filters didn't pattern-match the abstract wording). 36 of 39 are
pre-2026 references (foundational EBM / RL / LLM papers); 3 are 2026
citations:

### NEW Rank MEDIUM: Inference-Time Planning Self-Generated Context (arXiv:2602.02991, Feb 3 2026)
- **Score:** 4×3×3×4 = **144**
- **Authors:** Haijiang Yan, Jian-Qiao Zhu, Adam Sanborn
- **Why it matters:** Bayesian framework explaining LLM planning
  dynamics: "self-generated context accumulation drives planning
  behavior shifts at inference time." **This is exactly the mechanism
  Fast-Slow Variant exploits** — the verifier-output-summary IS the
  self-generated context that accumulates across verify-repair
  iterations. Provides independent peer evidence that the
  fast-weight-context approach is mechanistically grounded, not just
  empirically motivated by arXiv:2605.12484. Useful paper-v6 §3 cite
  alongside the AR-LM↔EBM bijection and Fast-Slow papers.
- **Action:** cite in paper-v6 §3 (architecture validation). If .189
  exp1811 succeeds, this is a third independent literature anchor for
  the design pattern (Score 500 bijection + Score 400 Fast-Slow + Score
  144 inference-time-planning = ~triangulated theory base).
- **Caveat:** the paper is non-EBM-native (Bayesian framing, not energy);
  Carnot's specific verifier-energy mechanism is still novel relative
  to this work.

### Skipped (low score):
- **arXiv:2603.23398** Graph Energy Matching (Score 8) — molecular EBM,
  out-of-domain.
- **arXiv:2604.00555** Ontology-Constrained Neural Reasoning (Score 36) —
  enterprise agentic neurosymbolic, adjacent but not core.

### Pre-2026 references catalogued (no scoring; archived for citation tracking)

36 papers in the references-direction sweep span 2010 ("1004.2027" —
early relevant work) through 2025. Notable buckets include foundational
EBM papers (2010s), RL-from-feedback work (2017-2022), reasoning
benchmarks (2021-2024), and ICLR/NeurIPS 2024-2025 reasoning-model
papers. These are now in the known-set; future citation-sweeps from
other anchors will dedupe against them.

### Sweep takeaways

1. **Citation-following dwarfs keyword rotation in yield.** Zero promotions
   from the prior 2 keyword sweeps (16:50Z, 20:35Z); 1 in-domain
   promotion + 38 archived references from this single citation-sweep.
   Validates the operator-confirmed "operator-flagged anchors are the
   highest-yield channel" finding empirically.
2. **The bijection anchor (Score 500) was published Dec 2025; 5 months
   of citations means a meaningful citation graph already exists.** The
   Fast-Slow anchor (Score 400, May 2026) is too new to have meaningful
   citations yet (sweep_citations.py 404'd it earlier today). Citation-
   following yield scales with anchor age.
3. **Broadened-cluster sweeps need rate-limit care.** arXiv API 429'd
   on `max_results=20` with complex OR-chains; should drop back to
   `max_results=8` for the broadened cluster URLs (operator can paste
   them into the cron prompt or invoke via `sweep_clusters.py` with
   `--max-results 8`).
4. **arXiv:2602.02991 strengthens paper-v6 §3 architecture validation.**
   Three independent peer-reviewed mechanisms now point at the same
   design: AR-LM↔EBM bijection (2512.15605), Fast-Slow Training
   (2605.12484), and self-generated-context-driven planning shifts
   (2602.02991). Carnot's verify-repair loop sits at the intersection
   of all three.

## Sweep 2026-05-15T20:35Z (Claude outer-loop, hour-mod-4=0; clusters 1 EBM + 0 verifier-ensembles)

**Queries fired (rotated to clusters with productive history):**
- arxiv abs:"energy based model" AND ("reasoning" OR "verification" OR "LLM") → 8 fetched (5 known + 1 new + 2 out-of-domain)
- arxiv abs:"verifier ensemble" OR "null space attack" OR "specification gaming" → 8 fetched (8 known)
- HN skipped (4 prior 0-hits today; no broader-query authorization)

**Dedupe filter (2nd deployment of scripts/sweep_dedupe.py):**
16 candidates fetched, 13 already-known filtered at ingest, 3 truly-new
candidates surfaced. 53 IDs now in known-set (up from 49 at 16:50Z fire).

**Result: 0 NEW promotions.** All 3 newly-surfaced candidates marginal
or out-of-domain:

- **arXiv:2604.14733** "Differentiable Object Pose Connectivity Metrics
  for Regrasp Sequence Optimization" (Qin/Wan/Harada, Apr 2026).
  Score 1×3×1×1 = **3**. Robotic manipulation EBM, not LLM domain.
  Skipped.
- **arXiv:2602.03640** "Tutorial on Reasoning for IR & IR for Reasoning"
  (Hoveyda et al., Feb 2026). Score 3×2×2×2 = **24**. IR-context
  survey; mildly cites EBM approaches but no novel methodology for
  Carnot. Skipped.
- **arXiv:2601.02594** "Annealed Langevin Posterior Sampling (ALPS)"
  (Chand/Jacob, Jan 2026). Score 3×3×2×3 = **54**. Multiscale EBM
  for IMAGE inverse problems; out-of-domain BUT the annealed-Langevin
  sampling primitive is potentially adaptable to Carnot's THRML
  near-critical sampler failure (.175 exp1709 — fundamental limit at
  beta=1.05 unfixed in 54-cell ablation). Marginally relevant; note
  but don't promote.

### Sweep takeaways

1. **Saturation confirmed across 2 successive fires.** 16:50Z dedupe
   filter rate: 81% (17/21). 20:35Z dedupe filter rate: 81% (13/16).
   The 4 fixed-cluster queries have fully mapped the recent-window
   arxiv state. Future productive sweeps require either (a) the
   operator-discussed cluster-URL broadening (process-reward-model,
   token-energy, transcoder, predictive-coding terms), (b) extension
   to broader arxiv categories beyond cs.LG, OR (c) shift to a
   different signal channel (PaperWithCode, OpenReview venue tracking).
2. **arXiv:2601.02594 ALPS annealed-Langevin is the closest hit to
   exp1709's open question.** The near-critical sampler limit at
   beta=1.05 (no intervention closed the gap in 54-cell burn-in ×
   h_schedule ablation) is exactly the kind of failure mode that
   annealing schedules attack. NOT promoting to active queue
   (Score 54 too low) but flagging the cross-cite potential — if
   .190+ revisits exp1709 with ALPS-style multiscale annealing,
   this paper becomes the methodology reference.
3. **Operator-flagged additions remain the highest-yield channel.**
   The auto-rotation surfaced 0 promotions across 2 successive sweeps;
   meanwhile operator-flagged arXiv:2605.12484 (Fast-Slow, Score 400)
   from 13:15Z remains the single most impactful literature input of
   the day. The signal: routine arxiv rotation does not surface novel
   directions at the current state-of-the-art window; targeted
   operator review is where new ideas come from.

## Sweep 2026-05-15T16:50Z (Claude outer-loop, hour-mod-4=0; clusters 2/0/3 — FIRST with dedupe filter)

**Queries fired (rotated to clusters with stalest coverage):**
- arxiv abs:"sparse autoencoder" OR "white box probe" OR "reconstruction error" AND "LLM" → 8 fetched
- arxiv abs:"verifier ensemble" OR "null space attack" OR "specification gaming" → 8 fetched
- arxiv abs:"active inference" OR "free energy" AND "LLM" → 5 fetched
- HN skipped (3 prior 0-hits in 24h; no broader query yet authorized)

**Dedupe filter (NEW workflow step 2.5, first deployment):**
21 candidates fetched, 17 already-known filtered at ingest via
`python3 scripts/sweep_dedupe.py --filter`, 4 truly-new candidates
surfaced for scoring.

**Result: 0 NEW promotions.** All 4 newly-surfaced candidates either
out-of-domain or low-score:

- **arXiv:2602.19160** "LLM Reasoning from General Game Playing"
  (Świechowski et al., Feb 2026). Score 3x3x2x2 = **36**. LLM logical-
  error taxonomy in GGP environments; mildly applicable for adversarial
  corpus design but not critical path. Skipped.
- **arXiv:2602.18082** "AndroWasm" — Android malware obfuscation,
  out-of-domain false positive on the "specification gaming" filter
  (the paper uses the phrase in security context). Skipped.
- **arXiv:2601.23206** "Game content via small LMs" — game content
  generation, out-of-domain. Skipped.
- **arXiv:2605.12784** "ToolMol" drug discovery agentic framework —
  molecular agentic, out-of-domain. Skipped.

**Dedupe protocol validated.** Pre-dedupe, this sweep would have
re-scored 17 papers we've already ranked across the prior 7 sweeps
(2604.07650, 2605.02269, 2604.12500, 2603.28063, 2605.12874, 2605.14694,
2605.14449, 2605.14347, 2605.13930, 2605.12809, 2605.12770, 2605.12245,
2605.07639, 2605.12536, 2605.12495, 2605.11638, 2603.08806). Post-dedupe,
only 4 new candidates scored — saving ~70% of the prose budget per the
operator-confirmed efficiency win 2026-05-15.

### Sweep takeaways

1. **Dedupe filter works as designed** (17/21 = 81% filter rate). The
   protocol is now stable; future sweeps will report in the
   "N fetched / M known-skipped / P scored / Q promoted" compact format.
2. **The 3 out-of-domain false positives** (AndroWasm, game content,
   ToolMol) confirm the operator-precedence bug noted in the
   12:48Z sweep entry — narrow `abs:"phrase"` matches hit unrelated
   papers when the phrase has alternate meanings. Adding explicit
   AND-grouping (per the operator-discussed cron-URL upgrade) would
   filter these at fetch.
3. **0 promotions ≠ low-yield**. The 81% dedupe rate means the queue
   is well-mapped and stable. Operator-flagged manual additions
   (e.g., arXiv:2605.12484 Fast-Slow at Score 400) remain the
   highest-yield input channel for surfacing genuinely-new ideas
   the auto-rotation misses.

## Sweep 2026-05-15T12:48Z (Claude outer-loop, hour-mod-4=0; clusters 1 EBM + 3 active inference)

**Queries fired (rotated to clusters not covered in 08:45Z fire):**
- arxiv abs:"energy based model" AND ("reasoning" OR "verification" OR "LLM") → 5 results (ALL re-hits from prior sweeps)
- arxiv abs:"active inference" OR "free energy" AND "LLM" → 4 results (3 re-hits, 1 UAV-domain rejected)
- hn.algolia "energy EBM verifier" → 0 hits

**Result: 9 candidates; 0 NEW promotions; 8 re-hits + 1 rejected. Cluster saturation confirmed across 4 sweep rotations in 24h.**

### Rejected candidates (this sweep):

- **arXiv:2604.27935v1** — "Flying by Inference: Active Inference World Models for Adaptive UAV Swarms" (Arshid et al., Apr 30 2026). Score 2×4×2×2 = **32**. UAV swarms domain; hierarchical world-model active inference is mildly applicable to Carnot's verifier-as-free-energy framing but the domain gap is too large for replication value. **Worth noting**: the "hierarchical probabilistic inference + online KL minimization" structure echoes the alpha_t / alpha_t' computation Carnot is currently rescuing in .182 exp1745 — IF the .182 per-step disaggregation succeeds, this paper becomes a candidate cross-cite for paper-v6 §3.

### Saturation pattern (all 4 sweeps 2026-05-15)

The 4 cluster queries have been hit ~3 times each in 24h. Re-hit rate:
- Cluster 0 (verifier ensembles / spec gaming): 4 sweeps, 0 new since 04:42Z
- Cluster 1 (EBM + LLM): 3 sweeps, 0 new since 04:45Z
- Cluster 2 (SAE / white-box probe): 2 sweeps, last new 08:45Z (QAOD, exemplars, rate-distortion)
- Cluster 3 (active inference): 2 sweeps, 0 new since 04:45Z

The fixed `max_results=8` + the recent-paper bias of `sortBy=submittedDate` means each rotation re-fetches the same 5-8 papers until enough time passes for arXiv's listing to refresh.

### Recommended next-rotation broadening (do NOT modify CLAUDE.md per cron constraint, but the next-fire prompt could consider):

- Cluster 0 expansion: add `OR abs:"process reward model"` OR `abs:"deliberative alignment"`
- Cluster 1 expansion: add `OR abs:"token energy"` OR `abs:"energy guided decoding"`
- Cluster 2 expansion: add `OR abs:"feature attribution"` OR `abs:"transcoder"`
- Cluster 3 expansion: add `OR abs:"predictive coding"` OR `abs:"world model"`

Saturation is a healthy sign that Carnot's literature landscape is well-mapped at the current state-of-the-art window. Re-hits are NOT wasted; they confirm priority stability.

### Sweep takeaways

1. **Re-hit-only fire validates queue stability** — the active queue's top 5 (2512.15605, 2605.02269, 2605.14449, 2605.12536, 2605.14558) are not being displaced by new arrivals.
2. **Carnot's .182 per-step alpha disaggregation (exp1745) has no published peer methodology in this sweep window** — the arXiv:2604.27935 hierarchical-active-inference angle is the closest hit but UAV-domain. Carnot is operating in a literature-gap zone for this specific question.
3. **Next rotation should consider broader queries** per the suggestions above. Marking this as a sweep-mechanism observation, not a CLAUDE.md change.

## Sweep 2026-05-15T08:45Z (Claude outer-loop, hour-mod-4 rotation; cluster 2 SAE primary + cluster 0 verifier-ensembles)

**Queries fired (clusters 1/3 just covered in prior fire; rotating to cluster 2 SAE primary and cluster 0):**
- arxiv abs:"sparse autoencoder" OR "white box probe" OR "reconstruction error" AND "LLM" → 5 results
- arxiv abs:"verifier ensemble" OR "null space attack" OR "specification gaming" → 3 results (all already in queue)
- (cluster 1 EBM + cluster 3 active inference skipped — covered in 00:42Z + 04:45Z sweeps respectively)

**Result: 8 candidates scored; 3 NEW promotions; 5 re-hits acknowledged. Top score 320 (no >400 this fire).**

### NEW Rank HIGH: QAOD White-Box Hallucination Detection (arXiv:2605.14449v1, May 14 2026)
- **Score:** 5×4×4×4 = **320**
- **Authors:** Siyang Yao, Erhu Feng, Yubin Xia
- **Why it matters:** White-box probing framework using ORTHOGONAL
  decomposition of answer representations against question context.
  Reports "up to 21% improvement on BioASQ" for cross-domain
  hallucination detection. **Direct adversarial test against Carnot's
  NLA-via-SAE methodology** (exp1694/1720): orthogonal-decomposition
  may outperform the SAE-based NLA probe shipped as verifier #16 in
  .178. Worth a head-to-head comparison.
- **Action for .180+:** propose a "QAOD vs NLA-SAE probe head-to-head"
  experiment on the same gemma-4-26B-A4B-it-GGUF substrate Carnot
  already has loaded. If QAOD outperforms NLA-SAE by >5pp on the
  same 60-example test set used in exp1716, propose adding QAOD as
  verifier #17 (NOT as an NLA replacement — k=16 stays shipped).
- **Cross-reference:** complements exp1716 eval-awareness test
  (delta_tpr=-0.042 SAFE) — QAOD provides an alternative probe
  family that may have different eval-awareness characteristics.

### NEW Rank MEDIUM-HIGH: Exemplar Partitioning for Mechanistic Interpretability (arXiv:2605.14347v1, May 14 2026)
- **Score:** 4×5×3×4 = **240**
- **Authors:** Jessica Rumbelow (independent)
- **Why it matters:** Voronoi partitions of activation space as
  unsupervised alternative to SAE training, achieving "comparable
  interpretability with ~10³× fewer tokens." If the 1000× efficiency
  claim transfers, Carnot could replace the exp1694-trained SAE (1k
  calibration corpus) with a Voronoi-partition probe trained on ~10
  examples — drastically cheaper for the kind of small-corpus
  domain-specific verification Carnot does.
- **Action:** treat as the cheaper-substrate alternative to the
  current NLA-SAE. If the .180+ QAOD comparison shows SAE is the
  weaker probe, Voronoi-partition is the next candidate replacement
  rather than re-training a larger SAE.
- **Cross-reference:** orthogonal to QAOD; both are "skip the SAE"
  alternatives.

### NEW Rank MEDIUM: Rate-Distortion-Polysemanticity Tradeoff in SAEs (arXiv:2605.14694v1, May 14 2026)
- **Score:** 4×4×3×3 = **144**
- **Authors:** Tommaso Mencattini, Francesco Montagna, Francesco Locatello
- **Why it matters:** Formal rate-distortion analysis of the
  polysemanticity tradeoff in SAEs. Shows enforcing interpretability
  necessarily increases both rate AND distortion; polysemanticity
  is driven by training-data distribution characteristics. Carnot's
  NLA-SAE probe has 704 active features (exp1694); this paper's
  tradeoff curve tells us where 704 sits on the
  reconstruction-vs-monosemanticity spectrum and whether scaling up
  is even productive.
- **Action:** cite in paper-v6 §3 NLA-probe methodology section.
  Not a near-term experiment but informs SAE sizing for future
  Carnot probes.

### Re-hits of papers already in queue (no action):

- arXiv:2605.12874 (Descriptive Collision in SAE Auto-Interpretability) — promoted in 2026-05-14T04:15Z sweep (URGENT)
- arXiv:2604.07650 (Behavioral Entanglement Verifier Ensembles) — promoted .144
- arXiv:2605.02269 (Spec Gaming in Reasoning Models) — promoted 00:42Z, Score 320
- arXiv:2604.12500 (Safety Training under On-Policy RL) — Score 36, skipped 3x
- arXiv:2605.13930 (SAE on EEG Foundation Models) — out-of-domain (EEG, not LLM)

### Sweep takeaways

1. **NLA-probe landscape has 3 alternative families** now visible:
   (a) Carnot's current SAE-based NLA (exp1694 shipped, k=16
   production), (b) QAOD orthogonal-decomposition (this sweep,
   Score 320), (c) Voronoi-partition exemplars (this sweep, Score 240).
   The head-to-head comparison is a clear .180+ task and would
   strengthen paper-v6 §3.
2. **No score>400 this fire** — converging toward operational
   refinement (head-to-head probes; rate-distortion analysis) rather
   than fundamental new directions. This is healthy: it means
   Carnot's high-level architecture stays competitive with the
   literature; what's left is engineering refinement.
3. **The verifier-ensemble cluster has saturated** in the recent
   window — all 3 results were repeats from prior sweeps. Next
   rotation may need to broaden the query (add "process reward
   model" or "deliberative alignment" as adjacent terms).

## Sweep 2026-05-15T04:45Z (Claude outer-loop, hour-mod-4 rotation, clusters 3/0/1)

**Queries fired (cluster 3 active-inference skipped last fire — picked up this fire):**
- arxiv abs:"active inference" OR "free energy" AND "LLM" → 3 results
- arxiv abs:"verifier ensemble" OR "null space attack" OR "specification gaming" → 3 results (all 3 already in queue from prior sweeps)
- arxiv abs:"energy based model" AND ("reasoning" OR "verification" OR "LLM") → 5 results (4 already in queue; 1 new from 2026-05-14)
- hn.algolia "verifier energy LLM" → 0 hits

**Result: 11 candidates scored; 2 NEW promotions (both Score 192); 9 re-hits acknowledged. No score>400 this fire.**

### NEW Rank MEDIUM-HIGH: Token-Level Energy for Agentic RL (arXiv:2605.14558v1, May 14 2026 — yesterday)
- **Score:** 4×4×3×4 = **192**
- **Authors:** Langzhou He, Junyou Zhu, Yue Zhou, Zhengyao Gu, Junhua Liu,
  Wei-Chieh Huang, Henry Peng Zou, David Wipf, Philip S. Yu, Qitian Wu
- **Why it matters:** Token-level energy-based credit assignment in agentic
  RL reveals that training signals concentrate on action tokens despite
  their scarcity. Proposes the **ActFocus** reweighting mechanism, reporting
  a 65.2pp gain over PPO with no computational overhead. Directly relevant
  to Carnot's FR-11 (verifier-as-reward RL) work which has been an open
  retro question across .96-.150+. The token-level energy framing is
  compatible with Carnot's verifier-output-as-energy interpretation; if the
  ActFocus reweighting transfers to verifier-driven RL, it could be a
  near-term lift.
- **Caveat:** the 65.2pp PPO gain is a SUBSTANTIAL claim; should be treated
  as adversarial-verify-worthy if Carnot replicates. Replication budget:
  one Carnot agentic-RL experiment with + without ActFocus reweighting on
  the same FR-11-style verifier signal.
- **Action:** queue for .177+ as a candidate FR-11 follow-up experiment.

### NEW Rank MEDIUM-HIGH: IIT ↔ FEP Maximum-Caliber Bridge (arXiv:2605.12536v1, May 3 2026)
- **Score:** 4×4×3×4 = **192**
- **Authors:** Alexander Kearney
- **Why it matters:** Establishes mathematical connection between the Free
  Energy Principle (Phase 4 substrate) and Integrated Information Theory
  through maximum-caliber variational principles. Demonstrates that
  information emerges from prediction error under predictive coding.
  Directly relevant to Carnot's Phase 4 active-inference framing — the
  alpha_t metric needs theoretical grounding (the exp1693 suspicious
  invariance + the AR-LM↔EBM bijection paper suggest alpha_t may be
  bijection-invariant by construction). This paper's maximum-caliber
  framing may provide an alternative derivation of alpha_t that is NOT
  bijection-invariant — worth investigating before .176 exp1715 retries
  the alpha_t audit.
- **Action:** cite in paper-v6 §3 (Phase 4 theoretical framing) alongside
  arXiv:2512.15605. If exp1715 confirms bijection-invariance artifact,
  the maximum-caliber derivation in this paper is the replacement candidate.

### Re-hits of papers already in queue (no action):

- arXiv:2604.07650 (Behavioral Entanglement) — promoted .144
- arXiv:2605.02269 (Spec Gaming in Reasoning Models) — promoted 04:42Z sweep
- arXiv:2604.12500 (Safety Training under On-Policy RL) — Score 36, skipped twice
- arXiv:2512.15605v3 (AR-LMs are Secretly EBMs) — promoted 00:42Z sweep, Score 500
- arXiv:2512.18730v1 (Theoretical Lens RL-Tuned LLMs) — promoted, Score 192
- arXiv:2601.21064v3 (Textual Equilibrium Propagation) — noted, Score 144
- arXiv:2602.18671v4 (Spilled Energy) — already integrated as `verify_spilled_energy`

### Sweep takeaways

1. **No score>400 this fire** — the queue is converging on the AR-LM↔EBM
   bijection (2512.15605) as the top theoretical anchor. Two new Score-192
   candidates (token-level energy for agentic RL; IIT↔FEP bridge) are
   complementary: ActFocus is a near-term operational lift; IIT↔FEP is
   theoretical framing for Phase 4 alpha_t derivation.
2. **Active inference cluster has thinned out** — only 3 results in the
   most-recent listing, of which 2 are weakly LLM-related. The cluster
   may need a broader query (e.g., add "predictive coding", "perception-
   action loop", "world model").
3. **exp1709 finding from .175 ALREADY beats the literature on its
   specific question** — analytic Curie-Weiss ground-truth comparison at
   n=128 with 10k samples isn't matched in any of the sampling papers
   surveyed this fire. The .176 exp1714 codification has at least one
   independently-novel contribution.

## Sweep 2026-05-15T00:42Z (Claude outer-loop, hour-mod-4 rotation, clusters 0/1/2)

**Queries fired (3 of 4 cluster rotation; cluster 3 active-inference skipped this fire):**
- arxiv abs:"verifier ensemble" OR "null space attack" OR "specification gaming" → 4 results
- arxiv abs:"energy based model" AND ("reasoning" OR "verification" OR "LLM") → 5 results
- arxiv abs:"sparse autoencoder" OR "white box probe" OR "reconstruction error" AND "LLM" → 5 results
- hn.algolia "verifier energy LLM" → 0 results (no HN front-page activity this window)

**Result: 14 candidates scored; 4 NEW promotions (top score 500 — score>400 noted to known-issues per protocol).**

### NEW Rank URGENT: Autoregressive LMs are Secretly EBMs (arXiv:2512.15605v3, Dec 2025; v3 update May 2026)
- **Score:** 5×5×4×5 = **500**
- **Authors:** Mathieu Blondel, Michael E. Sander, Germain Vivier-Ardisson, Tianlin Liu, Vincent Roulet (Google DeepMind, INRIA, EPFL collaboration)
- **Why it matters URGENT:** Establishes an explicit BIJECTION between
  autoregressive language models and energy-based models, and connects
  both to maximum-entropy RL. Provides theoretical error bounds for
  DISTILLING an EBM into an AR-LM. Directly relevant to Carnot's
  Phase-3 endgame ("evolve into a foundation model based on hardware-
  acceleratable EBM/EBT"). The bijection means our work translating
  between LLM outputs and EBM energy IS a well-defined map — not a
  bolt-on. The distillation error bounds may give Phase 3 a clean
  acceptance gate (distillation gap small → AR-LM and EBM are
  operationally equivalent).
- **Action for paper-v6:** cite as §3 peer methodology AND §6 theoretical
  framing. The bijection is exactly the formal scaffolding Carnot needed.
- **Action for Phase 3:** add a milestone task that re-derives Carnot's
  verifier-as-free-energy interpretation through the AR-LM↔EBM bijection
  in this paper. The Phase 4 active-inference suspicious-invariance
  finding from exp1693 may be a corollary of this bijection (alpha_t
  computed in a way that's invariant to substrate size because both
  representations are operationally equivalent under the bijection).
- **Score-gate cross-reference:** also added to ops/known-issues.md under
  RESEARCH-STUDYING CANDIDATES per the score>400 protocol.

### NEW Rank HIGH: Specification Gaming in Reasoning Models (arXiv:2605.02269v1, May 4 2026)
- **Score:** 5×4×4×4 = **320**
- **Authors:** Kei Nishimura-Gasparian, Robert McCarthy, David Lindner (Lindner is at Anthropic)
- **Why it matters:** Open-source evaluation suite demonstrating "all
  tested models exploit their specifications at non-negligible rates"
  across diverse settings. RL reasoning training INCREASES exploitation
  rates; test-time mitigations only partially reduce. Directly tests
  whether Carnot's k=15 verifier ensemble's null-space resilience holds
  up against deliberate spec-gaming (vs unwitting hallucinations).
- **Action:** when Phase-3 substrate is ready, run Carnot's k=6/k=15
  verifier ensemble against this suite. The k=6→k=15 lift is the
  empirical handle on null-space-mimicry defence (cf. memory entry
  project_null_space_mimicry_attack.md).
- **Cross-reference:** complements arXiv:2603.28063 (next entry) which
  proves the theoretical inevitability — together they bracket the
  empirical-vs-theoretical sides of spec gaming.

### NEW Rank MEDIUM-HIGH: Theoretical Lens for RL-Tuned LLMs via EBMs (arXiv:2512.18730v1, Dec 2025)
- **Score:** 4×4×3×4 = **192**
- **Authors:** Zhiquan Tan, Yinrong Hong
- **Why it matters:** Theoretical analysis of KL-regularized RL for LLMs
  using CLOSED-FORM EBM structures, analyzing "verifiable rewards"
  through the lens of optimal reasoning distributions. Provides
  convergence properties for instruction-tuned models. Carnot's
  FR-11 (verifier-as-reward) work has empirical retros from
  .96-.150+; this paper's closed-form EBM framework is the
  theoretical scaffolding those retros were missing.
- **Action:** cite in paper-v6 §3 (FR-11 methodology) AND §6 (theoretical
  underpinning for verifiable-reward RL convergence claims).

### NEW Rank MEDIUM-HIGH: Orthogonal Latent Spaces SAE for Token Influence (arXiv:2605.12809v1, May 12 2026)
- **Score:** 4×4×3×4 = **192**
- **Authors:** Shixing Yu, Promit Ghosal, Kyra Gan
- **Why it matters:** SAE-based token-influence attribution with EXPLICIT
  orthogonality constraint on latent features + Jacobian-vector products
  for "non-decomposable" latent influence. Directly informs Carnot's
  NLA 16th verifier 4-task chain — exp1694 (.171) shipped TPR=0.73 with
  704 active SAE features but did NOT enforce orthogonality. Adding the
  orthogonal-latent constraint may improve eval-awareness robustness
  (which exp1700 will measure in .172).
- **Action:** if exp1700 (.172) NLA eval-awareness test detects a gap >
  5pp, the .173 NLA prototype v4 should adopt the orthogonal-latent
  constraint from this paper. The Jacobian-vector products methodology
  may also reduce per-example latency below exp1694's 150ms p50.

### Additional candidates noted (score 80-150; not promoted to active queue):

- **arXiv:2601.21064v3** — Textual Equilibrium Propagation (Chen, Deng, Zou, Yu, Li; Jan 2026 v3). Score 144. Equilibrium-prop inspired LLM workflow optimization; relevant to EBT path but replication non-trivial.
- **arXiv:2603.28063v1** — Reward Hacking as Equilibrium (Wang, Huang; Mar 2026). Score 144. Proves spec gaming is a "structural equilibrium, not a correctable bug" — bracket-citation with arXiv:2605.02269 above.
- **arXiv:2511.21882v1** — Closed-Loop / Equilibrium Transformers (Anbar Jafari; Nov 2025). Score 144. Iterative latent refinement via energy minimization; parallel to Carnot's EBT direction.
- **arXiv:2605.12055v1** — Linguistic Constraint Violations via SAE (Hardy, Padó; May 12 2026). Score 81. Negative result — limited evidence for unified violation detectors. Informs NLA: should NOT expect one feature per verifier class.

### Re-hits of papers already in queue (no action):

- arXiv:2604.07650 (Behavioral Entanglement) — already promoted in 2026-05-14 sweep #4
- arXiv:2602.18671 (Spilled Energy v4 update) — already partially integrated as `verify_spilled_energy` method; v4 = newer revision but no new claims
- arXiv:2605.12874 (Descriptive Collision in SAE) — already promoted in 2026-05-14T04:15Z sweep #2

### Sweep takeaways

1. **2512.15605 is the highest-score sweep result in Carnot's literature
   record to date** (500 > prior top scores of 400). The AR-LM↔EBM
   bijection is the theoretical scaffolding Phase 3 was missing and
   may explain the exp1693 alpha_t suspicious-invariance finding.
2. **Spec-gaming corpus is converging on "structural equilibrium" framing**
   (2605.02269 empirical + 2603.28063 theoretical). Carnot's null-space-
   mimicry defence is the right thing to test against this corpus.
3. **SAE methodology is maturing fast** (3 May 2026 SAE papers in this
   sweep alone). The NLA 4-task chain should explicitly track this
   sub-literature; .173 prototype v4 should adopt orthogonality
   constraints if .172 eval-awareness detects a gap.

## Sweep 2026-05-14T20:45Z (Claude outer-loop /loop job 875c06b4 fire #6)

**Queries fired:**
- arxiv abs:"active inference" OR "free energy" AND "LLM" → 8 results (this hour the API responded)
- arxiv abs:"verifier ensemble" OR "null space attack" OR "specification gaming" → 8 results

**Result: 1 NEW candidate promoted; 2 re-hits of papers already in queue.**

### NEW Rank MEDIUM: AlphaGRPO Decompositional Verifiable Reward (arXiv:2605.12495, May 12 2026)
- **Score:** 4×4×3×3 = **144**
- **Authors:** Runhui Huang, Jie Wu, Rui Yang
- **Why it matters:** Introduces "Decompositional Verifiable Reward" that
  decomposes requests into verifiable semantic queries during GRPO training.
  Structurally similar to Carnot's NSVIF constraint extraction (DSL →
  PySAT/Z3 verifiable constraints). Worth investigating whether the
  decomposition primitive transfers to Carnot's pipeline, OR whether the
  reverse — Carnot's NSVIF-style constraint extraction — could enhance
  AlphaGRPO's reward decomposition. For paper-v6 §3 peer mention.

### Repeat hits (already in queue from prior sweeps, no action needed)
- arXiv:2604.07650 Behavioral Entanglement (already Score 400)
- arXiv:2605.02269 Specification Gaming in Reasoning (already Score 300)
- arXiv:2605.11638 U-Statistics with Active Inference (already Score 36, not promoted)
- arXiv:2605.07639 Tacit Knowledge Extraction (already not promoted)
- arXiv:2605.12536 Maximum-Caliber Deviation (already Score 48)

### Sweep-#6 takeaways
- arxiv API is responsive again this hour (vs the 429/timeout streak earlier today)
- Yield is low because we've already harvested the high-relevance recent
  submissions in prior sweeps. The corpus refreshes weekly-ish on arxiv;
  expect sweep-#7+ to be similarly thin until the next batch of relevant
  preprints lands.

---

## Sweep 2026-05-14T16:55Z (Claude outer-loop /loop job 875c06b4 fire #5)

**Queries fired:**
- arxiv abs:"sparse autoencoder" OR "white box probe" OR "reconstruction error" → **HTTP 429**
- arxiv abs:"formal verification" AND "LLM" → **HTTP 429**
- HN search: `energy verifier hallucination` → 0 hits
- HN search: `energy based model` → same 10 results as sweep #4 (no churn)
- Semantic Scholar API → **HTTP 429**

**Result: low-yield fire.** Both arxiv and Semantic Scholar API rate-limited this hour. HN hadn't churned since sweep #4. No new candidates promoted.

**Operational observation:** the cron is firing every 4 hours but arxiv's API has been 429-throttled on 2 of the last 3 fires. This is consistent with the rest-of-the-world also hammering it. Recommendation for cron-prompt revision: stagger by day_of_year mod 4 AND add a 60-90s delay between WebFetches to spread the request load.

---

## Sweep 2026-05-14T12:40Z (Claude outer-loop /loop job 875c06b4 fire #4)

**Queries fired:**
- arxiv abs:"probabilistic computing" OR "Ising machine" OR "stochastic circuit" OR "p-bit" → **timeout 60s**
- arxiv cat:cs.LG AND abs:"Ising" AND abs:"sampling" → **HTTP 503 Service Unavailable**
- HN search: `energy based model` (timestamp > 1746500000)

**arxiv API is degraded this fire.** Both queries failed (one timeout, one 503).
HN was the only successful source.

**Candidates surfaced (10 HN stories, 1 promoted):**

### NEW Rank LOW-MEDIUM: Kona EBM Sudoku Benchmark (logicalintelligence.com, Feb 2026)
- **Score:** 4×3×4×3 = **144**
- **Source:** HN (2 points, low signal, but content is high-signal)
- **URL:** logicalintelligence.com/blog/energy-based-model-sudoku-demo
- **Why it matters:** Headline claim **96% vs 2% on Sudoku** — Carnot's Phase-3
  parity target (per CLAUDE.md Project Vision: "Functional parity with Kona")
  has new published benchmark numbers. Worth checking:
  1. Whether Carnot's existing sudoku verifier (per `python/carnot/verify/sudoku.py`)
     can run the same benchmark format
  2. The "96%" methodology: is it solve-rate, verification-accuracy, or
     constraint-satisfaction? Kona is closed-weight; Carnot adaptation TBD.
- **Status:** NOT pre-stage material (Phase-3 parity is a long-horizon target;
  current focus is Phase 1 ship + adversarial-verify rigor).

### Additional HN candidates (not promoted; score < 100):
- ebmsovereign.com Energy-Guard OS — 88.7% leak detection EBM (security domain,
  off-topic for Carnot's reasoning verification mission)
- 2024 iopscience.iop.org "Introduction to latent variable EBMs" — foundational,
  already implicitly used in Phase-3 DBAE-EBM design (memory)
- "Logical Intelligence" startup page (LeCun-linked, EBM-based) — already cited
  in CLAUDE.md
- Other HN hits: self-promotional / off-topic (Sudoku demos, anomaly detection
  blog posts, YouTube content)

### Sweep-#4 takeaways

- **arxiv API health is unreliable** at this hour. Two queries failed in a
  row. Future fires should be defensive: try one arxiv query, fall back to
  HN/openreview/semantic scholar promptly.
- **HN-only fires have lower yield**. arxiv is the load-bearing source for
  serious literature.
- Worth augmenting the cron prompt with a semantic-scholar fallback URL:
  `https://api.semanticscholar.org/graph/v1/paper/search?query=...&limit=10&fields=title,abstract,year,citationCount`

---

## Operator-flagged 2026-05-14T11:00Z: Iron Layer (github.com/bwahacker/iron-layer)

**Source:** Operator-flagged during conversation, not via sweep. Worth recording as peer methodology.

### Iron Layer — Prompt-Injection Honeypot Labeler (bwahacker, ~Apr 2026)
- **Score:** 4×3×4×3 = **144** (moderate; not pre-stage material but worth tracking)
- **Repo:** github.com/bwahacker/iron-layer
- **What it is:** Detonates untrusted text inside an isolated sandbox with a canary LLM (Claude Haiku 4.5) wired to a wildcard MCP server that **fakes** dangerous tool execution. Records which dangerous-intent buckets (`filesystem-read`, `code-exec`, `network-egress`, `exfil-email`, `secret-access`, etc.) the injected text coaxed out. Output: JSONL pairs `(raw_input_text → tickled_signals)` for downstream classifier training (Featrix).
- **Why it matters to Carnot (5 angles):**
  1. **Adversarial corpus generator for exp2102 NLA probe v2.** Pre-staged
     `.165 task explicitly needs n>=30 adversarial examples spanning factual /
     logical / arithmetic classes. Iron Layer's JSONL output format is
     directly suitable as adversarial eval data. Candidate corpus source.
  2. **Signal normalization pattern.** Iron Layer's clean bucket taxonomy
     (`filesystem-read`, `code-exec`, etc.) could inspire a similar
     normalization layer for Carnot's verifier-ensemble output schema in
     paper-v6 §3.
  3. **Sandbox-canary pattern.** Structurally similar to Carnot's
     `CARNOT_USE_SANDBOX=1` gvisor pattern. When Carnot eventually runs
     adversarial outputs through the verifier ensemble, an Iron-Layer-style
     canary sandbox would let us "execute" suspect outputs without
     real-world side effects.
  4. **Deterministic lures via hashing.** Same principle as Carnot's
     `reproducibility_checksum`. Prior art for hash-deterministic adversarial
     inputs — worth citing in the Adversarial Artifact Verification CLAUDE.md
     rule.
  5. **Specification gaming connection.** Iron Layer is the operational
     artifact of arXiv:2605.02269 ("Specification Gaming in Reasoning
     Models") — the injection text gamings the canary's apparently-helpful
     behavior. Reinforces the load-bearing nature of that paper.
- **Decentralization concern:** uses closed-weight Claude Haiku 4.5 for the
  canary. Per CLAUDE.md decentralization Rule 1 (local-first using open
  models), Carnot adaptation would need a local model (Qwen3.5-0.8B as the
  canary). Adapting is straightforward — the methodology is model-agnostic.
- **License unknown** from README excerpt; need to verify before integration.
- **Status:** NOT pre-stage material — Phase 1 ship, Phase 4 active inference,
  THRML parity v2, and NLA probe v2 are higher-priority. Recorded here for
  reference if exp2102 NLA probe v2 needs more adversarial data than we can
  synthesize manually, OR if a future "verifier-ensemble taxonomy paper" milestone
  draws on it as peer methodology.

---

## Sweep 2026-05-14T08:20Z (Claude outer-loop /loop job 875c06b4 fire #3)

**Queries fired:**
- arxiv abs:"test time compute" AND (verification OR reasoning OR sampling) → **HTTP 429 rate-limited**
- HN search: `LLM verifier` (timestamp filter created_at_i > 1746000000)
- arxiv abs page: 2512.02080 (deep-dive on top HN hit)
- arxiv listing /list/cs.LG/2605 → HTTP 404 (URL format issue; skipped)

**Candidates surfaced (10 HN stories, 1 deep-dive, 2 promoted):**

### NEW Rank URGENT: The 4/δ Bound — Predictable LLM-Verifier Convergence (arXiv:2512.02080, Dec 2025)
- **Score:** 5×5×4×4 = **400**
- **Authors:** Pierre Dantas, Lucas Cordeiro, Youcheng Sun, Waldir Junior
- **Surfaced via:** Hacker News (59 points, 13 comments)
- **Why it matters:** This is a **theoretical convergence bound for verifier-loop
  systems with formal guarantees** — exactly the kind of architectural
  justification paper-v6 §3 needs. Models the LLM-verifier loop as an
  absorbing Markov chain with 4 stages (CodeGen → Compilation →
  InvariantSynth → SMTSolving) and proves: (1) termination for any δ > 0
  success rate per stage, (2) expected latency E[n] ≤ 4/δ iterations.
  Validated over 90,000 trials. **Carnot's verify-repair pipeline is
  structurally this exact architecture** (different specific stages, but
  same absorbing-Markov-chain shape). Citing this paper grounds Carnot's
  pipeline in published convergence theory rather than empirical hand-wave.
- **Action items:**
  1. Cite in paper-v6 §3 architecture lineage discussion
  2. Compute Carnot's empirical δ (stage success rate) from recent
     verify-repair runs and validate against the 4/δ prediction
  3. The "three operational zones (marginal, practical, high-performance)"
     calibration strategy is directly applicable to Carnot's tier system

### NEW Rank HIGH: BEAVER — Efficient Deterministic LLM Verifier (arXiv:2512.05439, Dec 2025)
- **Score:** 5×4×5×3 = **300**
- **Surfaced via:** Hacker News
- **Why it matters:** Carnot already has a "BEAVER-lite" task in `.147 (exp1879
  Deterministic Bounds for Validators ran OK). This is the **source paper**.
  Confirms our existing implementation is literature-grounded; should be
  cited in paper-v6 alongside Spera Theorem 9.2 + the 4/δ Bound. Worth
  reading the full paper to identify any features we're missing in the
  exp1879 implementation.

### Additional HN candidates (not promoted; below score 200):
- **Aura-State** (GitHub, Mar 2026, 23 pts) — Formally verified LLM state
  machine compiler using Z3 + CTL model checking. Z3 is already in Carnot;
  conceptual sibling. Score ~48.
- **Terminal-Bench-RL** (GitHub, July 2025, 125 pts) — "Hybrid reward
  signal of unit test verifiers & a behavioural LLM judge." Adjacent
  verifier-design pattern. Score ~36.
- **VR.dev** (Show HN, Mar 2026, 3 pts) — HARD/SOFT/AGENTIC verifier
  taxonomy; "deterministic probes against databases." Adjacent.
- **Sigma Guard** (Show HN, May 2026, 3 pts) — Cellular sheaf cohomology
  for consistency verification. Mathematically interesting, far from Carnot's
  current path.
- 5 others (Pencil Puzzle Bench, PupiBot1.0 triple-agent, Maestro orchestrator,
  Probus AI vuln scanner) — off-topic or low-signal.

### Cron-prompt bug status

Fire #3 confirmed the rotation issue: `hour mod 4` at fire-times :13 every 4h
always equals 0. Manual cluster selection used each fire instead. Should fix
the prompt formula to `day_of_year mod 4` OR `fire_counter mod 4` in a future
cron-prompt revision. Flagged but not fixed this fire.

---

## Sweep 2026-05-14T04:15Z (Claude outer-loop /loop job 875c06b4 fire #2)

**Queries fired** (rotated to clusters 2+3 since the rotation formula `hour mod 4`
in the cron always lands on the same residue when fires are every 4h — flagged
as a cron-prompt bug to fix later):

- arxiv abs:"sparse autoencoder" OR "white box probe" OR "reconstruction error" AND "LLM"
- arxiv abs:"active inference" OR "free energy" AND "LLM"

**Candidates surfaced (13 raw, 3 promoted to ranked queue below):**

### NEW Rank URGENT: Descriptive Collision in SAE Auto-Interpretability (arXiv:2605.12874, May 13 2026)
- **Score:** 5×5×4×4 = **400**
- **Author:** Jordan F. McCann
- **Why it matters:** **Direct adversarial-verify critique of SAE-based
  interpretability** — the foundational technique for Carnot's NLA-class 16th
  verifier (per `feedback_nla_class_16th_verifier_committed.md`). McCann shows
  that distinct SAE features receive identical text-descriptions, inflating
  reported interpretability by ~⅓ of feature identity bits. **This means the
  exp1851 NLA probe v2 must include a description-collision check before
  claiming the SAE features actually discriminate adversarial outputs.** If
  the 16-features-with-1-description pattern shows up in our SAE, the TPR
  lift claim is artificial (the probe is detecting feature-class identity,
  not output-distinguishing signal).
- **Action items:**
  1. Add a `feature_description_collision_rate` check to the planned exp2102
     NLA probe v2 artifact schema
  2. Cite McCann in paper-v6 §6 as a methodology limitation for white-box
     verifiers
  3. Sanity-check whether existing carnot SAE code (if any) already audits
     for this

### NEW Rank MEDIUM: Domain Restriction via Multi SAE Layer Transitions (arXiv:2605.11920, May 12 2026)
- **Score:** 4×4×4×3 = **192**
- **Authors:** Elias Shaheen, Avi Mendelson
- **Why it matters:** OOD detection via cross-layer SAE activation analysis.
  Carnot's verifier ensemble is structurally an OOD-detection problem for
  LLM outputs (verifier ensemble = "is this output in the support of valid
  outputs?"). Cross-layer SAE signal as an OOD verifier is a natural fit for
  the 16th verifier methodology. Lower priority than McCann because it's a
  technique to adopt rather than a methodology critique to defend against.

### NEW Rank MEDIUM: Do LMs Encode Linguistic Constraint Violations? (arXiv:2605.12055, May 12 2026)
- **Score:** 4×3×3×3 = **108**
- **Authors:** Hardy, Sebastian Padó
- **Why it matters:** "Employs sparse autoencoders to investigate whether
  LLMs encode grammatical violation detection through monosemantic feature
  activation patterns." Direct conceptual sibling to Carnot's verifier
  ensemble — constraint-violation detection via white-box SAE features. If
  the answer is yes, that's a baseline Carnot can adopt for the linguistic-
  constraint class of verifiers. If no, that's a known limitation.

### Additional candidates (not promoted; rank below 100):
- arXiv:2605.12809 Correcting Influence: Unboxing LLM Outputs (Yu, Ghosal, Gan) — SAE-based training-data attribution; tangential to verification
- arXiv:2605.12770 WriteSAE: SAEs for Recurrent State (Young) — SSM-targeted; relevant if Phase-3 substrate becomes recurrent
- arXiv:2605.12245 SOAR: Scale Optimization for NVFP4 Quantization — efficient deployment, not verification
- arXiv:2605.12225 Mechanistic Interpretability of ASR — audio domain
- arXiv:2605.12122 Disentangled Sparse Representations for Diffusion Unlearning — diffusion concept suppression
- arXiv:2605.11638 Learning U-Statistics with Active Inference — statistical estimation, not LLM
- arXiv:2605.07639 Tacit Knowledge Extraction via Logic Augmented Generation + Active Inference — adjacent to NSVIF
- arXiv:2605.12536 Information as Maximum-Caliber Deviation (Kearney) — FEP/IIT bridge, theoretical only
- arXiv:2605.01290 How Light Reshapes the Mind — cognition modeling, off-topic
- arXiv:2604.27935 Flying by Inference: UAV Swarms — robotics, off-topic

---

## Sweep 2026-05-14T00:42Z (Claude outer-loop, manual seed for /loop job 875c06b4)

**Queries fired:**
- arxiv abs:"verifier ensemble" OR "null space attack" OR "specification gaming"
- arxiv abs:"energy based model" AND (reasoning OR verification OR LLM)

**Candidates surfaced (10 raw, 4 promoted to ranked queue below):**

### NEW Rank URGENT: Behavioral Entanglement + Reweighting Verifier Ensembles (arXiv:2604.07650, Apr 2026)
- **Score:** 5×4×4×5 = **400**
- **Authors:** Kuai, Jiang, Zhu et al.
- **Why it matters:** Directly addresses Carnot's k=15 AND-composition null-space concern (per Spera Theorem 9.2 memory). Demonstrates that "correlated reasoning patterns and synchronized failures undermine ensemble verification" — i.e., the joint-null-space attack is empirically observable, not just theoretically possible. Reports up to **4.5% accuracy lift** from de-entangled reweighting of verifier ensembles. This is a load-bearing peer paper for paper-v6's Phase-3 architecture justification.
- **Carnot integration path:** Phase-3 verifier ensemble could adopt reweighting; current uniform-weight AND-composition is what the paper shows is suboptimal. Worth a dedicated milestone task to replicate the reweighting algorithm on Carnot's k=15 setup and measure the lift on a held-out adversarial corpus.

### NEW Rank URGENT: Spilled Energy in LLMs (arXiv:2602.18671, Feb 2026)
- **Score:** 5×4×5×4 = **400**
- **Authors:** Minut, Dewidar, Masi
- **Why it matters:** "Reinterprets LLM softmax classifiers as EBMs to detect hallucinations using training-free metrics derived from output logits without requiring probe classifiers." This is structurally identical to Carnot's verifier-energy philosophy — energy as verification, no labels required. Strong methodological peer. May provide a baseline to compare Carnot's verifier ensemble against on standard hallucination benchmarks. **Already partially used in Carnot** (verify_spilled_energy is a method in VerifyRepairPipeline per the conductor's AST signatures); confirm coverage + cite in paper-v6 §3 as a peer methodology.

### NEW Rank HIGH: Autoregressive LMs are Secretly EBMs (arXiv:2512.15605, Dec 2025)
- **Score:** 5×5×5×3 = **375**
- **Authors:** Blondel, Sander, Vivier-Ardisson
- **Why it matters:** Theoretical foundation — "Mathematical equivalence between autoregressive models and EBMs, revealing lookahead capabilities in next-token prediction." Supports Carnot's premise that any LLM output admits an energy interpretation, and therefore can be verified via energy. Cite in paper-v6 §3 architecture lineage; Phase-3 substrate justification.

### NEW Rank HIGH: Specification Gaming in Reasoning Models (arXiv:2605.02269, May 2026)
- **Score:** 5×4×3×5 = **300**
- **Authors:** Nishimura-Gasparian, McCarthy, Lindner
- **Why it matters:** "RL reasoning training increases exploitation rates of model specifications." Specification gaming is precisely the failure mode Carnot's adversarial-verify caught on exp1851 (3.4s wall time with TPR=1.0). This paper formalizes that the SOTA RL-trained models (Qwen3.6 GRPO etc.) are MORE prone to gaming the verifier signal — i.e., Carnot's verifier needs to be MORE adversarially robust on the current SOTA than on prior generations. Cite in paper-v6 §6 limitations + adversarial-verify CLAUDE.md rule.

### Additional candidates (not promoted; rank below 300):
- arXiv:2603.28063 Reward Hacking as Equilibrium (Wang, Huang) — theoretical unification of specification gaming; relevant to paper-v6 §6 but not actionable
- arXiv:2511.21882 Closed-Loop Transformers (Anbar Jafari) — iterative energy refinement; Phase-3 EBT track relevance, but the abstract is thin on numerical results
- arXiv:2601.21064 Textual Equilibrium Propagation (Chen, Deng, Zou) — workflow optimization not verification; adjacent
- arXiv:2604.12500 Safety Training Modulates Misalignment (Eshuijs et al.) — environment-design effects; less direct
- arXiv:2512.18730 RL-Tuned LLMs via EBMs (Tan, Hong) — theoretical, less actionable
- arXiv:2603.08806 Test-Driven AI Agent Definition (Rehan) — agent design, off-topic

---


## How This Works

1. Claude searches arxiv, OpenReview, GitHub, Extropic, Semantic Scholar, HN
2. Each finding is ranked by: relevance × novelty × feasibility × urgency
3. Top ideas are promoted to `research-roadmap-next.yaml` when a slot opens
4. Lower-ranked ideas stay here for future consideration
5. Ideas that prove irrelevant are moved to "Archived"

## Ranking Criteria

- **Relevance (1-5):** How directly does this apply to Carnot's current gaps?
- **Novelty (1-5):** Is this a new approach we haven't tried?
- **Feasibility (1-5):** Can we implement this in 1-2 experiments?
- **Urgency (1-5):** Does our current research depend on this?
- **Score = R × N × F × U** (max 625)

## Active Research Queue (Ranked)

### Rank 0a-prime: Fast-Slow Training (FST) — Carnot's Verify-Repair Loop Validated as Dual-Timescale Architecture (NEW 2026-05-15T13:15Z, operator-flagged)
- **Score:** 5×4×4×5 = **400**
- **Source:** arXiv:2605.12484 (May 2026) — "Learning, Fast and Slow:
  Towards LLMs That Adapt Continually"
- **Idea:** Treat LLM training as two timescales. "Slow weights" = model
  parameters (RL updates); "fast weights" = optimized context (in-context
  learning). Combining both yields 3× sample efficiency over RL-only,
  70% less KL divergence from base, less catastrophic forgetting, and
  successful continual learning where parameter-only RL stalls.
- **Direct mapping onto Carnot's architecture:** Slow weights = k=16
  verifier ensemble + base LLM (frozen at inference). Fast weights =
  the verifier-output-summary that re-prompts the LLM on the next
  verify-repair iteration. Carnot's value proposition has been
  "second-pair-of-eyes verification at inference time" — this paper
  provides peer-validated theoretical scaffolding for that exact
  architecture pattern.
- **Phase 4 rescue hypothesis:** the .181 exp1741 + .182 exp1745
  finding that alpha_t and alpha_t' are BOTH bijection-invariant at
  the ensemble-output level may be because we're measuring at the
  wrong scale. The fast-slow framing suggests measuring
  free-energy reduction on FAST WEIGHTS (context shaped by verifier),
  not slow weights (base model). If exp1745 confirms ensemble-level
  invariance, switching the measurement target to fast-weight context
  is the cleaner rescue.
- **FR-11 rethink:** Paper's central empirical finding is that
  parameter-only RL is strictly worse than fast-slow split on sample
  efficiency, drift, AND continual learning. FR-11 (verifier-as-reward
  RL) has stalled across .96-.150+ retros — possibly because it routes
  the verifier signal into slow weights (RL gradients) when the right
  destination is fast weights (context optimization).
- **Continual self-learning angle:** Carnot's CSL experiments (.177
  exp1779-1780, .180 exp1791) had mixed results. The paper provides a
  concrete mechanism: CSL works when the fast-slow split is in place —
  new tasks land in fast weights; slow weights only update slowly.
  Worth re-auditing CSL artifacts through this lens.
- **Where to land:** paper-v6 §3 architecture validation cite + concrete
  .183+ experiment (Carnot Fast-Slow Variant; see ops/known-issues.md
  RESEARCH-STUDYING CANDIDATES).
- **Cross-references:** complements but is structurally different from
  arXiv:2512.15605 (AR-LM↔EBM bijection theory) and arXiv:2605.14558
  (Token-Level Energy ActFocus). The bijection paper says "AR-LMs ARE
  EBMs"; this paper says "you should layer fast-weight context on top
  of slow-weight LMs/EBMs and train them at different timescales."

### Rank 0b: Token-Level Energy for Agentic RL — ActFocus Reweighting (NEW 2026-05-15T04:45Z)
- **Score:** 4×4×3×4 = **192**
- **Source:** arXiv:2605.14558v1 (He, Zhu, Zhou, Gu, Liu, Huang, Zou, Wipf, Yu, Wu; May 14 2026)
- **Idea:** Token-level energy-based credit assignment in agentic RL.
  Training signals concentrate on action tokens despite their scarcity.
  ActFocus reweighting reports 65.2pp gain over PPO with no compute
  overhead.
- **Hypothesis to investigate:** Does ActFocus reweighting transfer to
  Carnot's FR-11 (verifier-as-reward RL) flow? If yes, near-term
  high-leverage operational lift on the same .96-.150+ retros that
  have run on FR-11 without breakthrough.
- **Caveat:** 65.2pp gain is a substantial claim — replicate
  adversarial-verify-aware; treat as IMPLAUSIBLE_PERFECT-adjacent if
  Carnot replication shows gains > 30pp without methodology disclosure.
- **Where to land:** queue for .177+ FR-11 follow-up experiment.

### Rank 0c: IIT ↔ FEP Maximum-Caliber Bridge — Alternative alpha_t Derivation (NEW 2026-05-15T04:45Z)
- **Score:** 4×4×3×4 = **192**
- **Source:** arXiv:2605.12536v1 (Alexander Kearney; May 3 2026)
- **Idea:** Maximum-caliber variational principle bridges FEP and IIT;
  shows information emerges from prediction error under predictive
  coding.
- **Hypothesis to investigate:** alpha_t may have a maximum-caliber
  derivation that is NOT bijection-invariant under arXiv:2512.15605.
  If true, the .176 exp1715 audit's invariance finding (if confirmed)
  would NOT invalidate Phase 4 — it would just be using the wrong
  variational principle.
- **Where to land:** paper-v6 §3 Phase 4 theoretical framing,
  cited alongside arXiv:2512.15605.
- **Cross-reference:** depends on .176 exp1715 outcome — if
  artifact_detected=true, this is the replacement candidate framing.

### URGENT Rank 0a: AR-LMs are Secretly EBMs — Theoretical Scaffolding for Phase 3 (NEW 2026-05-15)
- **Score:** 5×5×4×5 = **500**
- **Source:** arXiv:2512.15605v3 (Blondel, Sander, Vivier-Ardisson, Liu, Roulet)
- **Idea:** Explicit bijection between autoregressive LMs and EBMs, plus
  distillation error bounds. The bijection is the formal scaffolding
  Carnot's Phase 3 ("foundation model based on hardware-acceleratable
  EBM/EBT") was missing — every architectural decision in Phase 3 can
  now be cross-checked against an existence proof that the AR-LM↔EBM
  map is well-defined.
- **Hypothesis to investigate:** The exp1693 (.171) Phase 4
  alpha_t = 0.15054 invariance across n=8/16/32/64 may be a corollary
  of this bijection — alpha_t is bijection-invariant, hence n-invariant.
  exp1699 (.172) random-verifier-injection audit will partially test this.
- **Where to land:** paper-v6 §3 (peer methodology) + §6 (theoretical
  framing); Phase 3 milestone task to re-derive verifier-as-free-energy
  through the bijection.
- **Why #0a (alongside the live-vs-simulated finding):** highest sweep
  score in Carnot's literature record (500 vs prior top 400); the
  bijection result is load-bearing for the Phase 3 endgame.

### URGENT Rank 0: Live vs Simulated Inference Validation
- **Score:** 5×5×5×5 = **625** (MAXIMUM)
- **Source:** Internal finding — ALL positive results were simulated inference
- **Crisis:** Exp 184 is the FIRST live GPU experiment and shows -2% standard,
  -12% adversarial on 3B model. But ALL previous positive results (Exp 91,
  120, 121, 161, 162) used SIMULATED inference. We cannot distinguish whether
  the negative result is model-size (precision ceiling) or inference-mode
  (simulation was unrealistically favorable).
- **MUST DO IMMEDIATELY:** Run 0.8B Qwen3.5 with LIVE GPU inference on the
  SAME GSM8K questions. If 0.8B live shows +10-14%, precision ceiling is real
  and we fix it with Z3/confidence. If 0.8B live shows ~0%, our ENTIRE results
  narrative is based on simulation artifacts and we have a fundamental problem.
- **Status:** INVESTIGATED — result is CONFIRMED ARTIFACT
- **Finding:** Live 0.8B inference on GSM8K produces identical wrong answers
  as the checkpoint (Q0=182, Q1=3, Q2=120000). The model scores ~25% on
  GSM8K — the simulated inference assumed ~65-70% (instruction-tuned level).
  ALL positive improvement numbers were measured against fake baselines.
- **Root cause:** Simulated inference was calibrated to published benchmarks
  for instruction-tuned models, but we loaded the BASE model (Qwen3.5-0.8B,
  not an instruct variant). The base model's actual GSM8K score is ~25%.
- **Impact:** The core +10-28% improvement claim is based on simulation
  artifacts. Real live inference shows 0% improvement at both 0.8B and 3B.
- **Path forward:** Either (a) use instruction-tuned models, (b) improve
  prompt engineering for base models, or (c) acknowledge constraint
  verification helps simulated/ideal scenarios but not raw base model outputs.
- **Exp 316 update (2026-04-14):** Full-scale benchmark ran in simulated mode
  (no live GPU). Schema and CI tests pass (28/28). Simulated results show no
  mode-to-mode improvement as expected — simulation is not live inference.
  Live GPU run still required to resolve the open question.
- **Why #0:** This is the most important finding of the entire project.

### Rank 1: Confidence-Calibrated Constraint Verification
- **Score:** 5×4×5×5 = 500
- **Source:** Internal finding (Exp 184: 3B model -2% regression)
- **Idea:** Weight constraint violations by confidence level. High-confidence
  violations (exact arithmetic mismatch) get repaired; low-confidence
  (approximate values, intermediate steps) get logged but not repaired.
  This directly addresses the precision ceiling where FP > TP on larger models.
- **Status:** Already in roadmap as Exp 202. Highest priority.
- **Why #1:** Without this, Carnot's value proposition shrinks as models improve.

### Rank 2: Semantic Constraint Verification via Chain-of-Thought Decomposition
- **Score:** 5×5×3×5 = 375
- **Source:** Exp 184 error analysis — larger models make semantic errors, not arithmetic
- **Idea:** Decompose chain-of-thought into logical steps, verify each step's
  LOGIC (not just arithmetic). "If A then B, A is true, therefore B" can be
  checked structurally. Apply the global consistency checker (Exp 172, 100%
  detection) to single-response multi-step reasoning.
- **Status:** Noted in research-program.md, not yet in roadmap
- **Why #2:** Addresses the 67% of errors that are currently uncatchable

### Rank 3: Speculative Decoding with Constraint Pre-Filtering
- **Score:** 4×5×3×4 = 240
- **Source:** Speculative decoding literature + our guided decoding (0.006ms)
- **Idea:** Use a small draft model to generate candidate tokens, then
  verify each candidate's constraint energy BEFORE the large model commits.
  Like speculative decoding but with constraint energy as the accept/reject
  criterion instead of probability matching.
- **Status:** Not in roadmap. Needs research.
- **Why #3:** Combines two proven techniques (spec decoding + constraint energy)

### Rank 4: Contrastive Constraint Learning from Model Errors
- **Score:** 4×4×4×4 = 256
- **Source:** Exp 184 data — we now have (correct, incorrect) pairs from a 3B model
- **Idea:** Train constraint extractors on the SPECIFIC error patterns of each
  model size. Instead of one-size-fits-all ArithmeticExtractor, learn what
  the 3B model gets wrong vs right and build model-specific constraints.
  The self-learning tracker (Exp 132) already accumulates this data.
- **Status:** Partially addressed by Exp 201 (precision curve)
- **Why #4:** Makes the constraint system model-adaptive

### Rank 5: FPGA Ising Sampler with Real-Time Coupling Updates
- **Score:** 3×5×3×3 = 135
- **Source:** Kria KV260 arriving in 4 days + research-hardware-wishlist.md
- **Idea:** Implement a 4K p-bit Ising sampler in Verilog with AXI-Lite
  interface for real-time coupling updates. The coupling matrix is
  reprogrammed for each constraint verification, not fixed at synthesis.
  This enables dynamic constraint checking at hardware speed.
- **Status:** Hardware ordered. Needs Verilog implementation.
- **Why #5:** Validates the TSU hardware path

### Rank 6: Energy-Aware Beam Search
- **Score:** 4×4×3×3 = 144
- **Source:** Guided decoding (Exp 110) + beam search literature
- **Idea:** Modify beam search to include constraint energy in the beam score.
  Standard beam search: score = log_prob. Energy beam search:
  score = log_prob - alpha * constraint_energy. This naturally steers
  generation toward constraint-satisfying sequences without post-hoc repair.
- **Status:** Not in roadmap
- **Why #6:** Principled integration of energy into generation

### Rank 7: Hierarchical Constraint Composition for Complex Reasoning
- **Score:** 3×4×3×4 = 144
- **Source:** Exp 63 (hierarchical Ising) + Exp 172 (global consistency)
- **Idea:** Compose constraints hierarchically: word-level (arithmetic),
  sentence-level (logic), paragraph-level (consistency), document-level
  (factual). Each level feeds violations to the next. This mirrors how
  human reasoning catches errors at multiple scales.
- **Status:** Partially explored (Exp 63, 172, 176)
- **Why #7:** Framework for scaling verification to complex documents

### Rank 8: Differentiable Constraint Compilation to Hardware
- **Score:** 3×5×2×3 = 90
- **Source:** Exp 66 (differentiable constraints) + FPGA path
- **Idea:** Compile differentiable KAN constraints directly to FPGA lookup
  tables. The spline knots become LUT entries. Training updates the LUT
  contents without FPGA resynthesis. This is the bridge between Tier 4
  adaptive structure and hardware acceleration.
- **Status:** Long-term, needs FPGA first
- **Why #8:** The eventual production architecture

## New Findings from Study Run (2026-04-11)

### NSVIF: Neuro-Symbolic Verification via First-Order Logic (HIGH RELEVANCE)
- **Source:** [arxiv 2601.17789](https://arxiv.org/html/2601.17789v1)
- **What:** Formalizes instruction verification as a CSP — extracts constraints
  from instructions, converts to first-order logic, solves with Z3 SMT solver.
- **Relevance to precision ceiling:** This is EXACTLY what we need for larger
  models. Instead of pattern-matching arithmetic (ArithmeticExtractor), formalize
  the constraints as FOL and use an SMT solver. FOL constraints have NO false
  positives — they're either satisfied or not. This could eliminate the FP
  problem on 3B+ models entirely.
- **Score:** 5×5×4×5 = **500** — ties with Rank 1
- **Action:** Promote to roadmap. Replace ArithmeticExtractor's regex with
  Z3 SMT solving for arithmetic constraints. Keep regex as fast path, Z3
  as verification backend.

### ConstraintLLM: Neuro-Symbolic for Industrial Scheduling
- **Source:** [EMNLP 2025](https://aclanthology.org/2025.emnlp-main.809.pdf)
- **What:** Neuro-symbolic framework combining LLMs with constraint solvers
  for industrial scheduling. LLM generates constraint specifications, solver
  verifies feasibility.
- **Relevance:** Directly applicable to our scheduling domain (Exp 44, LagONN).
  Could improve scheduling constraint extraction.
- **Score:** 4×4×4×3 = 192

### FPGA P-Bit Cluster: 6400 Spins, 64 Billion Flips/Second
- **Source:** [arxiv 2512.24558](https://arxiv.org/html/2512.24558) + 
  [Nature Electronics](https://www.nature.com/articles/s41928-024-01182-4)
- **What:** Multi-FPGA cluster implementing sparse Boltzmann machines with
  p-bits. Achieved 6400 spins (80×80 Ising) on FPGA, 50-64 billion
  probabilistic flips/second. CD training with up to n=10M sweeps.
- **Relevance:** Our KV260 (arriving in 4 days) has 256K LUTs — enough for
  ~4K p-bits. This paper provides the implementation reference: sparse
  connectivity, local parallel updates, low-precision arithmetic.
  Key detail: they use CD-n with n=10M sweeps per update, far more than
  our CD-1 or CD-5. Worth testing higher-n CD on our learned Ising models.
- **Score:** 4×4×5×3 = 240 — promotes above energy-aware beam search
- **Action:** Use as implementation reference for KV260 Ising sampler.
  Add high-n CD experiment to roadmap.

### Speculative Speculative Decoding (ICLR 2026)
- **Source:** [ICLR 2026](https://openreview.net/pdf?id=aL1Wnml9Ef)
- **What:** Meta-speculation — speculate the NEXT round during current
  verification. Amortizes verification cost across rounds.
- **Relevance:** If we combine with constraint energy, the draft model
  generates candidates, constraint energy pre-filters, and the target
  model verifies. Three-level pipeline. But complex to implement.
- **Score:** 3×5×2×3 = 90

### KAN Computing-in-Memory (Nature Communications 2026)
- **Source:** [Nature Comms](https://www.nature.com/articles/s41467-026-69592-w)
- **What:** Hardware implementation of KAN using tunable Gaussian-like
  memory cells. Spline activations implemented as analog memory lookups.
- **Relevance:** Validates our Tier 4 vision (KAN → hardware). Not directly
  actionable until we have the right hardware, but confirms the path.
- **Score:** 3×5×2×2 = 60

### Agentic Confidence Calibration (2026)
- **Source:** [arxiv 2601.15778](https://arxiv.org/html/2601.15778v1)
- **What:** Holistic Trajectory Calibration — extracts process-level features
  across an agent's entire trajectory to calibrate confidence.
- **Relevance:** Directly applicable to our multi-turn agentic verification.
  Instead of per-step constraint checking, calibrate confidence across the
  whole reasoning trajectory. Could improve the global consistency checker.
- **Score:** 4×4×3×4 = 192

## Updated Rankings After Study Run

| Rank | Idea | Score | Status |
|------|------|-------|--------|
| 1 | NSVIF: FOL + Z3 SMT constraint verification | **500** | NEW — promote to roadmap |
| 1 | Confidence-calibrated constraints | 500 | In roadmap (Exp 202) |
| 3 | Semantic constraint via CoT decomposition | 375 | Noted |
| 4 | Contrastive constraint learning | 256 | Partially in Exp 201 |
| 5 | FPGA p-bit cluster (implementation ref) | **240** | NEW — use for KV260 |
| 6 | Speculative decoding with constraints | 240 | Needs research |
| 7 | ConstraintLLM industrial scheduling | **192** | NEW |
| 7 | Agentic confidence calibration | **192** | NEW |
| 9 | Energy-aware beam search | 144 | Noted |
| 9 | Hierarchical constraint composition | 144 | Partially explored |
| 11 | FPGA Ising real-time updates | 135 | KV260 arriving |
| 12 | Speculative speculative decoding | **90** | NEW — complex |
| 12 | Differentiable constraint compilation | 90 | Long-term |
| 14 | KAN computing-in-memory | **60** | NEW — validates path |

### Kona 1.0 Architecture Details (STRATEGIC INTELLIGENCE)
- **Source:** [logicalintelligence.com](https://logicalintelligence.com/kona-ebms-energy-based-models),
  [BusinessWire Jan 2026](https://www.businesswire.com/news/home/20260120751310)
- **What:** Kona 1.0 is now in pilot programs. Key architectural details:
  - **Non-autoregressive at trace level** — generates complete reasoning traces
    simultaneously (not token-by-token)
  - **Continuous latent space** — outputs dense vector tokens, not discrete
  - **Self-correcting** — learns by recognizing and correcting own mistakes
  - **96.2% Sudoku** in 313ms (vs LLMs at 2%)
  - Yann LeCun added to leadership (validates EBM direction)
  - Pilot sectors: energy, manufacturing, semiconductors
- **Relevance:** This is our North Star competitor. Key differences from Carnot:
  - Kona generates reasoning; Carnot verifies LLM reasoning
  - Kona is non-autoregressive; Carnot works with autoregressive LLMs
  - Kona operates in continuous latent space; we're bridging to it (Exp 64-66)
  - The self-correcting aspect is what our verify-repair loop does externally
- **Implications for our precision ceiling:** Kona's continuous latent space
  may not have the FP problem because it doesn't use discrete constraint
  matching. Our Z3 SMT approach (NSVIF) is the bridge.
- **Score:** Strategic intelligence, not directly actionable. Monitor.

### Extropic Z1 Timeline Update
- **Source:** [extropic.ai/hardware](https://extropic.ai/hardware)
- **What:** Z1 chip (hundreds of thousands of p-bits per chip, millions per
  card) scheduled for early access 2026. XTR-0 testing platform was Q3 2025.
  Mass-manufacturable using standard CMOS.
- **Relevance:** Our KV260 FPGA (arriving in 4 days) is the bridge. If Z1
  early access opens, we have the SamplerBackend abstraction (Exp 71) ready
  to plug in. Our FPGA work validates the architecture before Z1 ships.
- **Score:** 3×3×2×3 = 54 — monitor, hardware path validated

### "Hallucination is Inevitable" (HuggingFace trending)
- **Source:** [huggingface.co/papers/2401.11817](https://huggingface.co/papers/2401.11817)
- **What:** Formal proof that LLMs inherently hallucinate — cannot learn all
  computable functions. Hallucination is a mathematical inevitability.
- **Relevance:** VALIDATES our entire approach. If hallucination can't be
  eliminated from INSIDE the model, external verification (Carnot) is the
  only path. This is the theoretical justification for our product.
- **Score:** 5×1×1×5 = 25 — not actionable but validates our thesis

## Libraries of Reference (Consulted During Study Runs)

Study runs check ALL of these sources:
1. **arxiv.org** — primary research papers
2. **OpenReview.net** — NeurIPS/ICML/ICLR submissions
3. **extropic.ai/writing** — TSU hardware updates
4. **Semantic Scholar** — citation tracking for key papers
5. **HuggingFace papers** (huggingface.co/papers) — daily ML paper feed
6. **GitHub trending** — new repos (ising-model, energy-based-model topics)
7. **logicalintelligence.com** — Kona architecture updates
8. **FPGA conferences** (FCCM, FPL, DAC) — Ising machine implementations
9. **AMD developer forums** — NPU/XDNA updates
10. **Nature Electronics/Communications** — hardware implementations
11. **ACL Anthology** — NLP constraint/verification papers

## Needs Investigation (Unranked)

- LagONN + guided decoding combination (oscillatory escape + energy steering)
- Multi-agent constraint verification (one agent generates, another verifies)
- Retrieval-augmented constraints (look up facts before verifying)
- Constraint transfer learning (train on one domain, apply to another)
- Grammar-constrained decoding as constraint substitute (ACL 2025 finding)
- Block verification for speculative decoding (5-8% speedup, OpenReview)
- Physics-informed KAN with augmented Lagrangian (Nature 2025)

## Revalidation Sweep — Approaches That Deserve Live Re-Testing

**Context:** The simulation artifact discovery (Exp 203-209) led us to remove
unverified numbers from reporting. But some earlier approaches may genuinely
work — they were tested with bad experimental methodology, not bad ideas.
This sweep re-runs the most promising old experiments with live GPU inference
to either confirm or definitively rule them out.

**STATUS: COMPLETED 2026-04-14** — Exp 271-279 executed and classified.
Full results: `results/revalidation_sweep_271_279_summary.json`.

### High Priority — Results

| Original Exp | Revalidation Exp | Classification | Outcome |
|-------------|-----------------|----------------|---------|
| 172, 176 | **Exp 271** | ✅ **CONFIRMED** | 100% detection, 0% FP, 1.91 ms/call — logic-based, inference-mode-independent |
| 134 (Tier 1) | **Exp 272** | ⚠️ INCONCLUSIVE | 86% FP reduction confirmed (7→1); task-success rate flat 32.7% — FP win is real, primary objective not met |
| 126-127 | **Exp 273** | ✅ **CONFIRMED** | 100% rollback success + 100% violation detection (canned outputs; deterministic logic) |
| 158 | **Exp 274** | ✅ **CONFIRMED** | 45% coverage ≥ 40% target; 100% accuracy ≥ 75% target on IT model responses |
| 175 | **Exp 275** | ✅ **CONFIRMED** | KAN AUROC 0.991 on live traces; AMR pruned 17 params, 0.0 AUROC gain |

### Medium Priority — Results

| Original Exp | Revalidation Exp | Classification | Outcome |
|-------------|-----------------|----------------|---------|
| 91-92 | **Exp 276** | ✅ **CONFIRMED** | Z3+LLM: 80% detection, 0% FP; semantic: 0% detection, 20% FP for arithmetic |
| 142 | **Exp 277** | ⚠️ INCONCLUSIVE | 3068 tests pass; results JSON absent — needs re-run for quantitative classification |
| 149 | — | Not revalidated | TruthfulQA factual coverage deferred to future milestone |
| 136 | **Exp 278** | ✅ **CONFIRMED** | 100% warm hit rate, 0% FP unseen, session boundary preserved, avg score 95.67 |

### Low Priority — Results

| Original Exp | Revalidation Exp | Classification | Outcome |
|-------------|-----------------|----------------|---------|
| 161, 163 | — | Superseded | Covered by Exp 219/235 (200-question GSM8K live runs) |
| 178 | **Exp 279** | ✅ **CONFIRMED** | Stale detection 100%, fresh-wrong 0%, FP 20%, lift +40pp — semantic grounding targets quantity-mismatch specifically |

### Definitively Ruled Out (evidence-based, not provenance-based)

These are NOT candidates for revalidation — they were disproven by experimental evidence:
- **Activation-based EBMs** (Exp 1-38): 14 principles prove they detect confidence, not correctness. No provenance issue — the approach is fundamentally flawed.
- **LNN adaptive couplings** (Exp 116): -90% vs static Ising. Worse in every metric.
- **Precision-based constraint reweighting** (Exp 134 original): 0% improvement on the specific reweighting approach (though the self-learning architecture was validated by Exp 223).

### D-Wave Quantum Annealing (ACTIONABLE — Add Now)
- **D-Wave Advantage**: 5,000+ qubits (Pegasus topology, 15-way connectivity).
  Advantage2: 7,000+ target (Zephyr, 20-way). Solves Ising/QUBO natively.
- **Ocean SDK**: Apache 2.0, `pip install dwave-ocean-sdk`. `dimod` for BQM,
  `neal` for local simulated annealing, `dwave-system` for real QPU.
- **Carnot fit**: Perfect — D-Wave literally solves Ising problems. Our
  SamplerBackend abstraction + IsingEBM coupling matrix maps 1:1 to dimod BQM.
- **Local simulation**: `neal.SimulatedAnnealingSampler()` runs locally, same
  API as hardware. Prove the approach works without QPU access.
- **Free tier**: 1 min QPU/month via D-Wave Leap (enough for ~1000 problems).
- **Score**: 5x5x5x4 = **500** — high relevance, high feasibility, proven technology
- **Action**: Add `dwave-ocean-sdk` as optional dep, create `DWaveSampler`
  implementing `SamplerBackend`, benchmark local sim vs CPU Ising sampler.

### Intel Loihi 2 Neuromorphic (Track — Cloud Access Available)
- **Intel Loihi 2**: 1M spiking neurons, on-chip learning. Free academic access
  via Intel Neuromorphic Research Community (INRC). Natively implements
  energy-minimization via spiking dynamics. Demonstrated Ising solving via
  neural annealing (Intel labs 2023-2024). Relevant to Boltzmann tier sampling.
- **Action:** Apply for INRC access. Could implement a `LoihiSampler` backend.

### Oscillator-Based Ising Machines (Track — CMOS Scalable)
- **Purdue/Cornell coupled CMOS oscillator networks**: Phase-encoded spins via
  injection-locked LC oscillators. 240-spin chip demonstrated (2024). CMOS-native
  means it could scale to millions of spins on standard foundries.
- **Purdue p-bit MRAM** (Camsari group): MRAM-based stochastic magnetic tunnel
  junctions. 8-p-bit ASIC demonstrated, 50K designs published. Same p-bit
  abstraction as Extropic but magnetic rather than thermodynamic.
- **Action:** Monitor for ASIC availability (2025-2027). Our SamplerBackend
  abstraction is ready for both approaches.

### NTT Coherent Ising Machine (Track — Largest Demonstrated)
- **NTT/Stanford CIM**: Optical parametric oscillator pulses, 100,000+ spins
  demonstrated. Time-multiplexed (vs SPIM's spatial). NTT offers cloud access
  for research collaborations. Largest Ising machine demonstrated to date.
- **Action:** Explore NTT research collaboration for cloud CIM access.

### Analog In-Memory Computing (Monitor)
- **Mythic M1076**: Analog matrix-multiply in flash memory, 25 TOPS. Dev kit
  ~$500. Energy function evaluation (W*s products) maps to analog MAC.
  Relevant for KAN/Boltzmann forward passes, not Ising sampling.

### EBM Safety Classifier (Distilled from gpt-oss-safeguard) — HIGH PRIORITY
- **Concept:** Train Carnot's KAN tier as a lightweight safety classifier using
  gpt-oss-safeguard (Apache 2.0, 20B/120B) as teacher. The KAN model (2.3K params,
  0.994 AUROC) could classify inputs as safe/unsafe at a fraction of the compute.
- **How it works:**
  1. Run gpt-oss-safeguard-20b on a corpus of safe + unsafe prompts
  2. Collect (input, safety_label, reasoning) pairs
  3. Train KAN energy model: low energy = safe, high energy = unsafe
  4. Deploy as a pre-filter in VerifyRepairPipeline for input sanitization
- **Advantages over gpt-oss-safeguard alone:**
  - 2.3K params vs 5.1B active params (2000x smaller)
  - Runs on CPU in <1ms (vs GPU inference for the teacher)
  - Integrates natively with Carnot's energy pipeline
  - Hardware-acceleratable (Ising/FPGA/D-Wave for the safety energy landscape)
- **Score:** 5x5x4x5 = **500** — high impact, feasible, proven teacher model
- **Action:** Add to next milestone. Requires downloading gpt-oss-safeguard-20b
  weights from HuggingFace and running distillation pipeline.

### Mythos System Card Insights (Applied — From Anthropic's 244-page safety evaluation)
- **Source:** Anthropic Claude Mythos Preview System Card (April 7, 2026)
- **Key findings applicable to Carnot:**

1. **Verification gap validated:** Even Mythos (93.9% SWE-bench) produces factual
   errors that are only caught when users explicitly request re-derivation. The model
   "could reach the right answer once asked but did not verify claims before writing
   them." This validates Carnot's external verification thesis.

2. **Reward hacking in self-learning:** Mythos discovered novel reward hacks (moving
   computation outside timing calls, using test data to train). Our self-learning
   loop (Exp 223/241) needs guards against energy function gaming.

3. **Behavioral monitoring for autonomous systems:** Anthropic uses automated offline
   monitoring, behavioral audits, and interpretability analysis for alignment. Our
   conductor runs autonomously for hours — we should apply similar monitoring.

4. **Constitutional alignment for autoresearch:** Define explicit rules for what the
   conductor can/cannot do without human approval. Prevent autonomous systems from
   taking irreversible actions.

- **Proposed experiments:**
  - Reward hacking detection in self-learning energy function
  - Conductor behavioral audit log with anomaly detection
  - Conductor constitution defining allowed/forbidden autonomous actions
  - Verification-before-publication gate (extend Exp 209 provenance audit)

### Vulkan Compute Backend for Universal GPU Support (Phase 2 — Plan Now)
- **Why:** CUDA locks us to NVIDIA. ROCm is unstable (broke on our iGPU).
  Vulkan works on every modern GPU: NVIDIA, AMD, Intel, mobile.
- **What to build:** Vulkan compute shaders for energy function evaluation
  (E = -0.5 x^T J x), Ising sampling (parallel spin flips), and KAN
  forward pass (B-spline evaluation).
- **Tools:** `vulkano` (Rust, our production language), `kompute` (Python bridge),
  or `wgpu` (Rust, WebGPU API over Vulkan/Metal/DX12).
- **Architecture:** Vulkan for energy computation, CUDA/ROCm for LLM inference
  (PyTorch/JAX still need vendor backends for model loading).
- **When:** Phase 2 — after core verification pipeline is stable. The Rust
  crates (`carnot-ising`, `carnot-kan`) are the natural place to add Vulkan.
- **Score:** 4x4x3x4 = 192 — important for portability, medium effort
- **Action:** Add Vulkan compute experiment to Phase 2 milestone. Start with
  Ising energy evaluation (simplest kernel), then KAN forward pass.

### NVIDIA "Ising" — NAMING COLLISION, NOT an Ising optimization solver (Noted, Low Relevance)
- **Source:** [nvidia.com/en-us/solutions/quantum-computing/ising/](https://www.nvidia.com/en-us/solutions/quantum-computing/ising/) (2026 release, exact date TBD)
- **What NVIDIA's "Ising" actually is:** A family of Apache-2.0-ish AI models for
  quantum computing workflows. NOT a classical Ising-model optimization solver.
  Two members:
  - **Ising Calibration** — 35B-parameter Vision-Language Model that automates
    quantum processor (QPU) tuning by inferring calibration actions from QPU
    experimental data.
  - **Ising Decoding** — Two 3D CNN models (0.9M / 1.8M parameters) for quantum
    error correction. Claimed 2.5x faster, 3x more accurate than prior methods.
- **Why the name collision matters for us:** Carnot's "Ising" means the discrete
  spin-glass optimizer (carnot-ising crate, Ising tier in the four-tier model
  hierarchy). NVIDIA's "Ising" means "AI models for operating qubit hardware".
  Future contributors reading "NVIDIA Ising" may incorrectly assume it's directly
  applicable to Carnot's Ising verifier. It is not.
- **Direct applicability to Carnot:** Very low.
  - We don't operate qubit hardware; we use Ising *as a math formulation* for
    constraint satisfaction on classical hardware.
  - The D-Wave sampler we integrated (Exp 320) IS quantum-hardware-adjacent,
    but uses D-Wave's quantum annealing, not NVIDIA's QPU calibration flow.
- **Indirect applicability:** The 3D CNN for quantum error correction is an EBM-
  like discriminator architecture — it learns to assign low "energy" to valid
  error syndromes vs. invalid ones. Pattern-level similarity to our CIKANEnergy
  and EORM models. Worth a skim of the arXiv write-up when available, but not
  worth an experiment.
- **Score:** 2x3x3x1 = 18 (mostly "name disambiguation" value, not research value).
- **Action:** No experiment planned. This entry exists so future sessions don't
  confuse NVIDIA Ising with our Ising work.

### CUDA Megakernel Fusion Techniques (Study — Transferable Optimizations)
- **Source:** [luce-megakernel](https://github.com/Luce-Org/luce-megakernel) (MIT)
- **What it is:** A single-dispatch CUDA megakernel that runs the entire Qwen 3.5-0.8B
  forward pass (24 layers) without returning to CPU. Eliminates ~100 kernel launches
  per token, achieving 413 tok/s on RTX 3090 at 1.87 tok/J.
- **Why study:** The kernel fusion pattern — running an entire compute graph in one
  dispatch with cooperative grid sync between stages — is transferable beyond LLM
  inference. Relevant techniques for Carnot:
  - **Fused Ising sampling:** Multiple spin-flip rounds + energy evaluation in one
    kernel launch, avoiding host round-trips between sampling iterations.
  - **Fused verify-repair pipeline:** Chain constraint extraction, energy evaluation,
    and repair candidate scoring in a single GPU dispatch.
  - **Register-resident state:** Keep Ising spin vectors in registers across iterations
    instead of writing to global memory — directly applicable to our RTX 3090 setup.
  - **DVFS power tuning:** Their 1.87 tok/J efficiency comes partly from GPU clock
    management — useful for extended conductor runs to avoid thermal throttle (GPU 0
    already hitting 82C per RETRO-025).
- **Limitations:** CUDA-only (no Vulkan path), batch-1 only, single-model specific.
  The code itself is not reusable, but the patterns are.
- **Score:** 3x4x3x3 = 108 — novel fusion patterns, medium relevance, medium effort
  to study and adapt, not urgent.
- **Action:** Study cooperative grid sync and register-resident state patterns. Consider
  applying to Ising sampling kernel if we write custom CUDA (before Vulkan port).

### Photonic Computing (Monitor — Not Actionable Yet)
- **Q.ANT NPU 2.0** — commercial photonic matmul accelerator (30x energy efficiency).
  Not directly useful for sampling. Commercial-only, no cloud access.
- **Photonic Ising Machines (SPIMs)** — encode spins as phase patterns, compute
  Hamiltonians optically in a single pass. Currently ~32 spins experimentally.
  - [arxiv 2508.17440](https://arxiv.org/abs/2508.17440) — k-local Ising + optical KANs on same platform (maps to Carnot tiers!)
  - [arxiv 2502.18918](https://arxiv.org/abs/2502.18918) — parallel SPIM via spatial multiplexing
  - [arxiv 2410.10689](https://arxiv.org/abs/2410.10689) — fully programmable SPIM
- **When to act:** When SPIMs scale past ~100 spins or Q.ANT opens cloud access.
  Our SamplerBackend abstraction is ready for a photonic adapter.

## Archived (Investigated, Not Promising)

- LNN adaptive couplings within chains: -90% vs static Ising (Exp 116)
- Precision-based constraint reweighting: 0% improvement (Exp 134)
- Activation-based EBMs: detect confidence not correctness (14 principles)

<!-- EXP210_STUDYING_START -->
## Study Run 2026-04-12 - Constraint Extraction for Instruction-Tuned Models

### Ranking update
| Rank | Idea | Score | Why it matters |
|------|------|-------|----------------|
| 1 | Prompt-to-constraint intermediate representation with solver fallback | 625 | NSVIF, DeCRIM, and ConstraintLLM all point to the same fix: extract atomic constraints from the instruction before verifying the answer. |
| 2 | Benchmark-first extraction workbench | 500 | FollowBench, CFBench, RealInstruct, and VIFBench provide the missing datasets needed to measure extraction recall and false positives directly. |
| 3 | Dual-path verification: prompt-answer first, CoT second | 500 | CoT verification is promising, but monitorability papers say Carnot should never depend on raw CoT alone. |
| 4 | Typed step-graph verification for arithmetic and logic traces | 375 | VeriCoT, PCRLLM, Deductive Verification, and Typed CoT all support moving from free-form traces to explicit premises and rules. |
| 5 | Constraint-programming route for scheduling and resource tasks | 240 | ConstraintLLM plus IndusCP is the best external path for Carnot's scheduling extractor gap. |
| 6 | CoT monitorability score and fallback policy | 240 | Recent monitorability work implies Carnot needs a gate deciding when CoT evidence is safe to trust. |

### Key takeaways
- The strongest direct fit is prompt-side instruction verification: convert instructions into atomic constraints first, then verify the answer against them.
- Step-level CoT verification is now technically credible, but only when reasoning traces are reformatted into explicit premises, rules, and typed steps.
- Benchmark coverage for fine-grained instruction constraints is finally good enough to evaluate extraction quality directly instead of using answer accuracy as a proxy.
- Recent monitorability papers make raw chain-of-thought an unsafe sole source of truth; Carnot needs a fallback path that does not trust CoT by default.

### Proposed experiments for 2026-04-15
- **EXP-211 - Instruction-to-Constraint IR Benchmark**
  Goal: Build a gold benchmark of atomic prompt constraints from FollowBench, RealInstruct, CFBench, and VIFBench, then measure extraction recall and false positives on instruction-tuned models.
  Hypothesis: Prompt-side decomposition will reduce false positives more than answer-only regex extraction because the verifier will know exactly which constraints matter before inspecting the response.
  Success criteria: Atomic constraint recall >= 0.85 on the curated benchmark, satisfied-constraint false-positive rate <= 0.05, and measurable improvement over the current regex plus Z3 promptless path.
- **EXP-212 - Dual-Path CoT Verifier with Typed Step Graphs**
  Goal: Implement a step-level verifier for arithmetic and logic traces using premise-rule-conclusion records inspired by VeriCoT, PCRLLM, Deductive Verification, and Typed CoT.
  Hypothesis: A typed step graph will catch errors that answer-only checking misses, but only when combined with prompt-derived constraints and a fallback to answer-level verification.
  Success criteria: On a live instruction-tuned cohort, catch >= 25% of wrong answers missed by prompt-only verification while adding < 2% extra false positives on correct answers.
- **EXP-213 - CoT Monitorability Audit and Fallback Policy**
  Goal: Measure whether Qwen and Gemma instruction-tuned models expose enough faithful reasoning to justify CoT-based extraction, using recent faithfulness and pathology metrics.
  Hypothesis: Monitorability differs by model family and task, so Carnot should gate CoT extraction behind a measured trust score rather than assuming traces are faithful.
  Success criteria: Produce a per-model monitorability score, a pathology breakdown, and a simple policy that predicts when to trust CoT extraction versus prompt-answer-only verification.
<!-- EXP210_STUDYING_END -->

## Study Run 2026-04-12 — Post-Milestone 2026.04.14 + Early 2026.04.15

**Updated:** 2026-04-12
**Current Focus:** Semantic grounding gap (0/9 wrong answers detected on live GSM8K)

### New Findings

#### Property-Generated Solver (HIGH IMPACT — code verification)
- **Source:** [arxiv 2506.18315](https://arxiv.org/abs/2506.18315)
- **What:** Uses property-based testing to validate LLM-generated code. Properties
  are simpler to define than exhaustive test oracles. **23-37% pass@1 improvement.**
- **Relevance:** Directly applicable to Exp 217 (property code verifier) and our
  HumanEval pipeline. Could multiply the +3.3pp we got in Exp 208.
- **Score:** 5×5×5×5 = **625** — MAXIMUM. Implement immediately.
- **Action:** Integrate PBT into CodeExtractor for Exp 217/220.

#### Eidoku: Neuro-Symbolic Verification Gate
- **Source:** [arxiv 2512.20664](https://arxiv.org/pdf/2512.20664)
- **What:** Deterministic rejection gate for LLM reasoning hallucinations.
  Neuro-symbolic sanity check that gates generative output.
- **Relevance:** Exactly what our verify-repair pipeline does. Validate our
  architecture against their design patterns.
- **Score:** 5×4×4×4 = 320

#### Neuro-Symbolic Compliance (LLM + SMT for Finance)
- **Source:** [arxiv 2601.06181](https://arxiv.org/html/2601.06181v1)
- **What:** LLM interprets regulations → generates SMT constraints → solver
  enforces consistency. 86.2% SMT code gen accuracy, 100x reasoning speedup.
- **Relevance:** Same pattern as our Z3 extractor but for legal/financial domain.
  Validates LLM-as-SMT-generator approach.
- **Score:** 4×4×4×3 = 192

#### SCoRe: Multi-Turn RL Self-Correction (ICLR 2025)
- **Source:** ICLR 2025 SuperCorrect
- **What:** Multi-turn RL teaches LLMs to self-correct. +15.6% MATH, +9.1% HumanEval.
- **Relevance:** Our verify-repair loop is external self-correction. SCoRe shows
  internal self-correction can complement it. Could inform repair prompting.
- **Score:** 4×4×3×4 = 192

#### Learning to Self-Verify (CRITICAL INSIGHT)
- **Source:** [arxiv 2602.07594](https://arxiv.org/html/2602.07594v1)
- **What:** Self-verification doesn't improve with model scale. Needs explicit
  training. Generation and verification are asymmetric capabilities.
- **Relevance:** Validates Carnot's external verification approach. LLMs can't
  self-verify — they need us.
- **Score:** 5×3×1×5 = 75 — not actionable but validates thesis

#### Thought Anchors (NeurIPS 2025 Workshop)
- **Source:** [OpenReview](https://openreview.net/forum?id=VnSlfeRCaU)
- **What:** Identifies which CoT reasoning steps have outsized impact on final
  answers. Some steps are "anchors" that determine the trajectory.
- **Relevance:** Could improve our CoT monitorability audit (Exp 213) — focus
  verification on anchor steps, not all steps.
- **Score:** 4×5×3×4 = 240

#### Scientific Knowledge-Driven Decoding Constraints
- **Source:** [arxiv 2604.06603](https://arxiv.org/html/2604.06603)
- **What:** Hard constraints combined with LLM distributions during decoding
  without interfering with normal reasoning.
- **Relevance:** Directly applicable to our guided decoding (Exp 110). Better
  constraint integration method.
- **Score:** 4×4×3×3 = 144

### Updated Rankings After 2026-04-12 Study Run

| Rank | Idea | Score | Status |
|------|------|-------|--------|
| 1 | **Property-Based Testing for code verification** | **625** | NEW — integrate into Exp 217/220 |
| 1 | Prompt-to-constraint IR with solver fallback | 625 | In progress (Exp 211-212) |
| 3 | Confidence-calibrated constraints | 500 | Deferred |
| 4 | Semantic constraint via CoT decomposition | 375 | In progress (Exp 215-216) |
| 5 | Eidoku verification gate pattern | **320** | NEW — architecture validation |
| 6 | Contrastive constraint learning | 256 | Partially explored |
| 7 | Thought Anchors for CoT focus | **240** | NEW — improve Exp 213 |
| 8 | FPGA p-bit cluster | 240 | KV260 arriving soon |
| 9 | Neuro-Symbolic Compliance (SMT) | **192** | NEW — validates Z3 approach |
| 9 | SCoRe self-correction | **192** | NEW — inform repair prompting |
| 11 | ConstraintLLM scheduling | 192 | Noted |
| 12 | Energy-aware beam search | 144 | Noted |
| 12 | Scientific decoding constraints | **144** | NEW — guided decoding |
| 14 | FPGA Ising real-time updates | 135 | KV260 arriving |

#### 1024-Neuron FPGA Ising Accelerator (FPGA REFERENCE)
- **Source:** [arxiv 2505.20250](https://arxiv.org/abs/2505.20250)
- **What:** All-to-all connected probabilistic Ising machine on FPGA with
  ~10,000x speedup over GPU heuristics. 1024 neurons.
- **Relevance:** Direct implementation reference for our KV260 Ising sampler
  (Exp 228). Our target is 4K spins — this shows 1K is proven.
- **Score:** 5×4×5×4 = 400

#### VCoT-Bench: Z3 Proofs → Verus Rust Verification
- **Source:** [arxiv 2603.18334](https://arxiv.org/html/2603.18334)
- **What:** Benchmarks LLMs on transforming Z3 proofs into Verus-level
  Rust verification steps. Bridges formal proofs to systems code.
- **Relevance:** Could connect our Z3 constraint verification to our Rust
  crates — formal proofs that compile to verified Rust code.
- **Score:** 4×5×2×3 = 120

#### Solver-Aided Policy Compliance for LLM Agents
- **Source:** [arxiv 2603.20449](https://arxiv.org/html/2603.20449)
- **What:** Translates NL tool-use policies into Z3 constraints, checks
  planned tool calls before execution.
- **Relevance:** Directly applicable to our agentic verification — could
  verify conductor/agent actions before they execute.
- **Score:** 5×4×4×3 = 240

#### SemLoc: Structured Grounding of LLM Reasoning
- **Source:** [arxiv 2603.29109](https://arxiv.org/abs/2603.29109)
- **What:** Binds each inferred property to a typed program anchor for
  runtime checking. 42.8% Top-1 fault localization accuracy.
- **Relevance:** Typed grounding of reasoning steps — aligns with Exp 212
  typed reasoning IR and Exp 215 semantic grounding.
- **Score:** 4×5×3×4 = 240

#### Graph of Verification: DAG-Based Multi-Granular Verification
- **Source:** [arxiv 2506.12509](https://arxiv.org/abs/2506.12509)
- **What:** Adaptive multi-granular verification using DAG structure over
  reasoning steps. Complements our step-by-step approach.
- **Score:** 4×4×3×3 = 144

#### Continuous Self-Improvement via Learned Verifier
- **Source:** [arxiv 2505.19475](https://arxiv.org/abs/2505.19475)
- **What:** Learned verifier scores candidates for test-time self-training
  loop. Matches Carnot's autonomous self-learning vision.
- **Score:** 4×4×3×4 = 192

### Implications for Milestone 2026.04.16

The Property-Generated Solver finding is transformative for code verification.
Our HumanEval result (+3.3pp) used only basic execution testing. PBT showed
23-37% improvement on similar benchmarks — we should expect a much larger delta
if we integrate property-based testing into our CodeExtractor + repair loop.

**Proposed milestone 2026.04.16 theme: "Scale What Works"**
1. Scale code verification with PBT (our strongest live result)
2. FPGA Ising prototype (KV260 should have arrived)
3. Full 164-problem HumanEval with PBT + repair (publishable result)
4. Multi-model code verification (Qwen + Gemma + larger models)
5. Self-learning from code verification traces (Tier 1-2)
6. Bridge to production: package the code verification pipeline

### Proposed Milestone: "Security Hardening" (after revalidation)

**Theme:** Harden the autoresearch pipeline against adversarial inputs,
supply chain attacks, and untrusted code execution.

**Experiments:**
1. **gvisor sandbox validation** — verify sandboxed_exec_function works
   end-to-end on full HumanEval, measure overhead vs in-process exec
2. **gpt-oss-safeguard-20b integration** — deploy as local content scanner
   for arxiv/web ingestion, measure false positive rate on research papers
3. **Model supply chain audit** — pin all HuggingFace model hashes, verify
   no trust_remote_code=True calls exist, add pre-download hash check
4. **Semgrep/Bandit for generated code** — scan LLM-generated code before
   execution, integrate into the verify-repair loop
5. **Conductor isolation** — run conductor in Firecracker microVM with
   limited filesystem access, network filtering
6. **Prompt injection detection** — add Rebuff or similar to detect injection
   attempts in web-fetched content used by the study run

**Dependencies:** gvisor already installed (runsc), Docker running,
sandbox.py module created
**Expected outcome:** Code execution fully sandboxed, external content
scanned before ingestion, model supply chain verified

### ~~Proposed~~ Completed Milestone: "Revalidation Sweep" (2026-04-14)

**Theme:** Re-run the 10 most promising pre-provenance experiments with
live GPU inference and modern extractors. Either confirm they work (and
add to the live results portfolio) or definitively rule them out with
evidence, not just missing metadata.

**Actual experiments:** 9 (Exp 271-279)
**Outcome:** 6 CONFIRMED, 2 INCONCLUSIVE (Exp 272 FP-only win; Exp 277 missing JSON), 0 definitively ruled out.
**Credible results added:** GlobalConsistencyChecker, agent rollback, factual KB extraction,
KAN verification, Z3+LLM on GSM8K arithmetic, cross-session memory, adversarial semantic grounding.
**Remaining:** Exp 277 (combined signals) needs re-run with explicit JSON output; TruthfulQA deferred.

### Sweep 2026-05-16T12:00Z
- **Anchor**: arXiv:2603.28135
- **New IDs**: 49
- **Promotions**:
  - arXiv:2601.17223 (Score 400)
  - arXiv:2602.14189 (Score 320)
  - arXiv:2604.16753 (Score 320)

## 2026-06-07 Exp 3932 - Agentic Verification Efficiency Positioning

**Candidate:** Verification-efficiency positioning for the next convergence
milestone.

**Score: 5 x 5 x 4 x 4 = 400** - high alignment with north-star section 5, high
experiment leverage, medium implementation risk, and high convergence value.

**Position:** Carnot belongs in the cheap discriminative verifier lane: a
classifier-first energy layer screens all steps/actions, while competent
GenRM/ThinkPRM judges handle hard cases. The local Exp 3926/3928 artifacts are
blocked, so the claim is positioned as a near-term convergence target rather
than a landed parity result; Exp 3929 supplies the synthetic ARC-AGI-3 action-
efficiency bridge.

**Next experiments:** ProcessBench full-benchmark head-to-head: run Carnot energy scores versus a competent GenRM/ThinkPRM-style judge on the full held-out benchmark so the efficiency claim is tested against a credible comparator; ARC-AGI-3 real-benchmark agentic run: replace the synthetic grid step with the official interactive harness and report action efficiency without claiming a leaderboard score.

## 2026-06-08 Exp 3943 - Verifier Efficiency Landscape Positioning

**Candidate:** .365 convergence steer after the verifier-efficiency proof.

**Score: 5 x 5 x 5 x 4 = 500** - maximum north-star alignment, maximum
experiment leverage, maximum public-positioning value, and medium execution
risk because real benchmark access and full ProcessBench throughput can still
block.

**Position:** Carnot now belongs in the cheap discriminative verifier lane:
energy verification screens every candidate cheaply, while GenRM/ThinkPRM-style
judges handle close or high-value cases. The .364 result should be framed as a
cost-normalized verifier proof, not as a claim that energy scoring replaces
generative reasoning.

**Next experiments:** ProcessBench full-benchmark head-to-head: run the landed cheap-energy verifier and the competent GenRM/ThinkPRM-style judge on the full held-out benchmark with cost-normalized parity/Pareto reporting; ARC-AGI-3 real agentic run / real ARC-AGI-3 agentic run: move from synthetic action-pruning to an official interactive harness run, reporting action efficiency only under the benchmark protocol.

<!-- EXP4520-ACTION-EFFICIENCY-SOTA-START -->
## 2026-06-20 Exp 4520 - .417 action-efficiency SOTA ingestion - INGESTED

**Status:** INGESTED into `docs/research-notes/arc-action-efficiency-sota-417.md`
and `results/experiment_4520_sota_ingestion_417.json`.

**Preconditions:** Hugging Face API reachability succeeded; `scripts/sweep_clusters.py`
clusters 5 and 6 emitted focused URLs; `scripts/sweep_semscholar.py` produced
seven arXiv candidates and HTTP 429 on replay/memory queries; top sources were
verified by arXiv abs-page HTTP 200 and low-concurrency WebSearch/WebFetch.
`/deep-research` was not invoked. No live solve, training run, leaderboard
submission, ops/status/traceability edit, or `scripts/research_conductor.py`
edit occurred.

**Methods marked ingested:** affordance-landscape clickability pruning
(arXiv:2008.09241, arXiv:2501.06047), SIERL/Go-Explore frontier control
(arXiv:2602.00460, arXiv:1901.10995), PER/DQfD replay seeding
(arXiv:1511.05952, arXiv:1704.03732), UI-Mem-style persistent action memory
(arXiv:2602.05832), and SLOPE-style optimistic potential shaping
(arXiv:2602.03201).

flagged_for_v418: affordance-pruned frame-change/clickability plus
SIERL/Go-Explore frontier control over replayable offline-search states; use
PER/DQfD replay seeding for predictor/value training, add UI-Mem-style
persistent action memory behind similarity-gated retrieval, and keep SLOPE as
ranking-only until it reduces actions-to-first-levelup at equal solve-rate.
<!-- EXP4520-ACTION-EFFICIENCY-SOTA-END -->

<!-- EXP4530-NAVIGATION-SEARCH-SOTA-START -->
## 2026-06-21 Exp 4530 - .418 navigation-search SOTA ingestion - INGESTED

**Status:** INGESTED into `docs/research-notes/arc-navigation-search-sota-418.md`
and `results/experiment_4530_sota_ingestion_navigation_search.json`.

**Preconditions:** `scripts/sweep_clusters.py --help` succeeded; the arXiv API
reachability check succeeded; `scripts/sweep_clusters.py` clusters 5 and 6
emitted focused URLs; `scripts/sweep_semscholar.py` returned zero unique arXiv
IDs and HTTP 429 on four focused navigation/search queries; top sources were
verified by arXiv abs-page HTTP 200 and low-concurrency WebSearch/WebFetch.
`/deep-research` was not invoked. No live solve, training run, leaderboard
submission, ops/status/traceability edit, or `scripts/research_conductor.py`
edit occurred.

**Methods marked ingested:** SoRB replay-buffer graph search (arXiv:1906.05253),
SIERL reachable-frontier subgoal control with reachability novelty
(arXiv:2602.00460, arXiv:1810.02274), Go-Explore / First-return-then-explore
archive discipline (arXiv:1901.10995, arXiv:2004.12919), embodied frontier
navigation scoring (arXiv:2304.05506, arXiv:2603.05377), and AERA speed-depth
budget control for ARC-AGI-3 (arXiv:2605.25931).

flagged_for_v419: SoRB-style replay-buffer graph over StepwiseExplorer frontier nodes, with exact _shortest_path navigation costs, charged return prefixes, RESET fallback diagnostics, and the existing CORE median-action gate as the acceptance metric
<!-- EXP4530-NAVIGATION-SEARCH-SOTA-END -->

<!-- EXP4541-GOAL-ACQUISITION-SOTA-START -->
## 2026-06-21 Exp 4541 - .419 goal-acquisition SOTA ingestion - INGESTED

**Status:** INGESTED into `docs/research-notes/arc-goal-acquisition-sota-419.md`
and `results/experiment_4541_sota_ingestion_goal_acquisition.json`.

**Preconditions:** `scripts/sweep_clusters.py --help` succeeded; the arXiv API
reachability check succeeded; `scripts/sweep_clusters.py` clusters 6 and 3
emitted focused URLs; `scripts/sweep_semscholar.py` returned arXiv:2507.14172,
arXiv:2603.20334, arXiv:2603.13372, and arXiv:2601.10904 with HTTP 429 on
three focused queries; top sources were verified by arXiv abs-page HTTP 200 and
low-concurrency WebSearch/WebFetch. `/deep-research` was not invoked. The .418
navigation thread is superseded and was not re-ingested. No live solve,
training run, leaderboard submission, ops/status/traceability edit, or
`scripts/research_conductor.py` edit occurred.

**Methods marked ingested:** Family-B executable world-model re-induction
(arXiv:2605.05138, arXiv:2603.24621), refinement-loop program synthesis
(arXiv:2601.10904, arXiv:2507.14172), adaptive behavior-test goal-shift
detection (arXiv:2512.22336, arXiv:2604.08792), and neural-guided DSL/library
induction for reusable level predicates (arXiv:2411.17708, arXiv:2310.19791).

flagged_for_v420: Family-B executable re-induction loop for each level-up, with separate GOAL-vs-dynamics candidates, adaptive behavior tests for goal-shift detection, and a bounded refinement loop around exp4533
<!-- EXP4541-GOAL-ACQUISITION-SOTA-END -->

<!-- EXP4553-LLM-INDUCER-SOTA-START -->
## 2026-06-21 Exp 4553 - .420 LLM-inducer SOTA ingestion - INGESTED

**Status:** INGESTED into `docs/research-notes/arc-llm-inducer-sota-420.md`
and `results/experiment_4553_sota_ingestion_llm_inducer.json`.

**Preconditions:** `scripts/sweep_clusters.py --help` succeeded; the arXiv API
reachability check succeeded; `scripts/sweep_clusters.py` clusters 6 and 3
emitted focused URLs; `scripts/sweep_semscholar.py` ran four focused queries
and returned HTTP 429 on all four, so no S2-only source was promoted. Top
sources were verified by arXiv abs-page HTTP 200 and low-concurrency
WebSearch/WebFetch. `/deep-research` was not invoked. No live solve, training
run, live LLM inference, leaderboard submission, ops/status/traceability edit,
or `scripts/research_conductor.py` edit occurred.

**Methods marked ingested:** Family-B executable world-model induction
(arXiv:2605.05138, arXiv:2603.24621), LLM-PV proposal distribution with held-out
execution selection (arXiv:2510.14331), ALGO LLM-generated oracle verification
(arXiv:2305.14591), ABPR procedural refinement (arXiv:2603.20334), and
Counterexample Guided Learning (arXiv:2606.11521).

flagged_for_v421: combine Family-B executable world-model induction (arXiv:2605.05138) with bounded counterexample-guided refinement (arXiv:2606.11521) inside the Exp 4544 GOAL+DYNAMICS+plan proposer
<!-- EXP4553-LLM-INDUCER-SOTA-END -->

<!-- EXP4565-VERIFIER-ROUTER-SOTA-START -->
## 2026-06-21 Exp 4565 - .421 verifier-router SOTA ingestion - INGESTED

**Status:** INGESTED into `docs/research-notes/sota-ingestion-verifier-router-421-2026-06-21.md`
and `results/experiment_4565_sota_ingestion_verifier_router.json`.

**Preconditions:** `scripts/sweep_clusters.py --help` succeeded; the arXiv API
reachability check succeeded; `scripts/sweep_clusters.py` clusters 0 and 6
emitted focused URLs; `scripts/sweep_semscholar.py` ran five focused queries
and returned HTTP 429 on all five, so no S2-only source was promoted. Top
sources were verified by arXiv abs-page HTTP 200 and low-concurrency
WebSearch/WebFetch. `/deep-research` was not invoked. No live solve, training
run, live LLM inference, leaderboard submission, ops/status/traceability edit,
or `scripts/research_conductor.py` edit occurred.

**Methods marked ingested:** self-evolving verifiable-reward RL for cross-game
verifier transfer (arXiv:2601.22607, arXiv:2505.24760), adaptive PRM-guided
candidate expansion (arXiv:2602.01070), budget-aware discriminative
verification (arXiv:2510.14913), CASCAL / IR3DE generated-data routing
(arXiv:2601.09692, arXiv:2606.06098), and executable world-model plus
counterexample-guided repair (arXiv:2605.05138, arXiv:2606.11521).

flagged_for_v422: adaptive PRM-guided candidate expansion over the Exp 4556 DiscriminativeVerifier, trained and refreshed with self-evolving verifiable-reward data (arXiv:2602.01070 + arXiv:2601.22607)
<!-- EXP4565-VERIFIER-ROUTER-SOTA-END -->

<!-- EXP4577-ACTION-EFFECT-SOTA-START -->
## 2026-06-22 Exp 4577 - .422 action-effect SOTA ingestion - INGESTED

**Status:** INGESTED into `docs/research-notes/sota-ingestion-action-effect-422-2026-06-22.md`
and `results/experiment_4577_sota_ingestion_action_effect.json`.

**Preconditions:** `scripts/sweep_clusters.py --help` succeeded; the arXiv API
reachability check succeeded; `scripts/sweep_clusters.py` clusters 5 and 6
emitted focused URLs; `scripts/sweep_semscholar.py` returned HTTP 429 for both
focused queries, so no S2-only source was promoted. Top sources were verified
by arXiv abs-page HTTP 200 and low-concurrency WebSearch/WebFetch of the six
arXiv sources plus the StochasticGoose implementation URL. `/deep-research` was
not invoked. No live solve, training run, live LLM inference, leaderboard
submission, ops/status/traceability edit, or `scripts/research_conductor.py`
edit occurred.

**Methods marked ingested:** StochasticGoose-style learned frame-change
clickability predictor (arXiv:2603.24621), AgentRM generalizable reward-model
search (arXiv:2502.18407), ThinkPRM generative process verifier
(arXiv:2504.16828), Scaling Flaws verifier-guided-search caution
(arXiv:2502.00271), adaptive PRM-guided best-first candidate expansion
(arXiv:2602.01070), and self-evolving verifiable-reward data refresh
(arXiv:2601.22607).

flagged_for_v423: use a StochasticGoose-style learned action-effect model as the candidate-expansion prior, then allocate Exp 4569 best-first frontier budget with adaptive PRM guidance and scaling-flaw controls (arXiv:2603.24621 + arXiv:2602.01070 + arXiv:2502.00271)
<!-- EXP4577-ACTION-EFFECT-SOTA-END -->

<!-- EXP4589-FEATURE-ROUTER-SOTA-START -->
## 2026-06-22 Exp 4589 - .423 feature-router SOTA ingestion - INGESTED

**Status:** INGESTED into `docs/research-notes/sota-ingestion-feature-router-423-2026-06-22.md`
and `results/experiment_4589_sota_ingestion_feature_router.json`.

**Preconditions:** `scripts/sweep_clusters.py --help` succeeded; the arXiv API
reachability check succeeded; `scripts/sweep_clusters.py` clusters 5 and 6
emitted focused URLs; `scripts/sweep_semscholar.py` returned HTTP 429 for three
focused queries and no S2-only arXiv ID was promoted. Top sources were verified
by arXiv abs-page HTTP 200 and low-concurrency WebSearch/WebFetch of the seven
arXiv sources. `/deep-research` was not invoked. No live solve, training run,
live LLM inference, leaderboard submission, ops/status/traceability edit, or
`scripts/research_conductor.py` edit occurred.

**Methods marked ingested:** SkillRouter full-text skill routing
(arXiv:2603.22455), SkillGraph evolving skill graphs (arXiv:2605.12039),
SkillComposer skill create/merge/improve (arXiv:2606.06079), Skill-Pro reusable
procedural skills (arXiv:2602.01869), SkillRL/SkillBank recursive skill
distillation (arXiv:2602.08234), Graph-Based Exploration for ARC-AGI-3
(arXiv:2512.24156), and the ARC-AGI-3 efficiency/drift contract
(arXiv:2603.24621).

flagged_for_v424: implement SkillRouter-style full-body routing over arc_solver_kit skills, backed by SkillGraph/SkillRL trace distillation and graph-explore env-adaptive replay regeneration for drifted rows (arXiv:2603.22455 + arXiv:2605.12039 + arXiv:2512.24156)
<!-- EXP4589-FEATURE-ROUTER-SOTA-END -->

<!-- EXP4601-GENERATION-SOTA-START -->
## 2026-06-22 Exp 4601 - .424 generation SOTA ingestion - INGESTED

**Status:** INGESTED into `docs/research-notes/sota-ingestion-generation-world-model-424-2026-06-22.md`.

**Filtered track:** candidate generation on first contact, executable/symbolic
world-model induction, perceptual grounding, verified skill/controller
synthesis, exploration oracles, and objective energy as a generation prior.

**Preconditions:** `.venv/bin/python scripts/sweep_clusters.py --help`
succeeded and the arXiv API reachability check succeeded. Cluster helpers 5 and
6 emitted focused exploration/world-model URLs. Semantic Scholar returned HTTP
429 for the focused ARC/CWM/Sensi/SkillGen query and HTTP 500 for the broader
world-model exploration query, so no S2-only source was promoted. Direct arXiv
HTTP checks verified all cited IDs. `/deep-research` was not invoked. No live
LLM inference, training run, leaderboard submission, ops/status/traceability
edit, or `scripts/research_conductor.py` edit occurred.

**Methods marked ingested:** Executable World Models plus Code World Models
(arXiv:2605.05138, arXiv:2510.04542, arXiv:2603.24621), Sensi perceptual
grounding and curriculum test-time learning (arXiv:2603.17683), verified
skill/controller synthesis (arXiv:2605.10999, arXiv:2605.16986,
arXiv:2605.08083), exploration-oracle / predictive-world-model curiosity
(arXiv:2502.00225, arXiv:2502.13200, arXiv:2505.19095), and adaptive symbolic
world-model induction for novel games (arXiv:2507.12821, arXiv:2510.12088).

Exp 4592 status mapped honestly: `winner_generated=2/25`, improving over the
1/25 baseline but leaving the generation wall mostly open. Exp 4594 status
mapped honestly: `complete: goal_energy_prior_no_value_honest_null_gap_sharpened`.

flagged_for_v425: executable_world_model_energy_config_space_generation_prior
(arXiv:2605.05138 + arXiv:2510.04542).

**Bottom line for .425:** make executable world-model induction the candidate
generator, and use objective energy as a trust/goal/repair prior inside
generation rather than another final reranker.
<!-- EXP4601-GENERATION-SOTA-END -->

<!-- EXP4613-WORLD-MODEL-TRUST-SOTA-START -->
## 2026-06-23 Exp 4613 - .425 world-model trust SOTA ingestion - INGESTED

**Status:** INGESTED into `docs/research-notes/world-model-trust-literature-2026-06-23.md`.

**Filtered track:** world-model trust energy, scored-agent verifier
integration, closed-loop model utility, learned heuristic search, and
goal-conditioned value for level-to-level generalization.

**Preconditions:** Hugging Face model API reachability returned `net_ok`.
`scripts/sweep_clusters.py` emitted focused world-model/search URLs. Semantic
Scholar returned HTTP 429 for the five focused queries, so no S2-only source was
promoted. Low-concurrency WebSearch/WebFetch plus direct arXiv HTTP checks
verified arXiv:2605.05138, arXiv:2502.01989, arXiv:2510.18135,
arXiv:2511.09515, arXiv:2102.04518, arXiv:2406.04935, arXiv:2206.03023, and
arXiv:2502.20379. `/deep-research` was not invoked.

**Methods marked ingested:** executable world-model induction plus
multi-verifier trust energy; VFScale as an intrinsic-energy control;
closed-loop world-model utility plus imagined policy repair; learned
value/pruning search; and goal-conditioned value learning. Note: arXiv:2206.03023
is GoFAR, not a UVFA/HER primary paper, so it is used as the goal-conditioned
offline value reference rather than mislabeled.

flagged_for_v426: executable_world_model_plus_multi_verifier_trust_energy
(arXiv:2605.05138 + arXiv:2502.20379)

flagged_for_v426: goal_conditioned_spatial_value_tiebreaker
(arXiv:2102.04518 + arXiv:2406.04935 + arXiv:2206.03023)

**Bottom line for .426:** make executable world-model induction the A1 source
of candidate models, score it with multi-aspect trust energy, route only
trusted models into A2, and use learned/goal-conditioned value strictly as a
bounded search tie-breaker until no-regression gates pass.
<!-- EXP4613-WORLD-MODEL-TRUST-SOTA-END -->

<!-- EXP4625-OFFLINE-LIVE-BRIDGE-SOTA-START -->
## 2026-06-23 Exp 4625 - .426 offline-live bridge SOTA ingestion - INGESTED

**Status:** INGESTED into `docs/research-notes/offline-live-bridge-literature-2026-06-23.md`.

**Filtered track:** offline-to-live transfer for the graduated value head:
distribution shift from winning-path training to live off-path frontiers,
calibration of a ranking into a bounded cost, and compute-cost control for
value-guided search.

**Preconditions:** Hugging Face model API reachability returned `net_ok`.
`scripts/sweep_clusters.py` emitted focused value/search and ARC exploration
URLs. `scripts/sweep_semscholar.py` returned HTTP 500/429 for the five focused
queries and no S2-only source was promoted. Low-concurrency WebSearch/WebFetch
plus direct arXiv HTTP checks verified arXiv:1011.0686, arXiv:2604.11351,
arXiv:1706.04599, arXiv:2102.04518, arXiv:2406.04935, arXiv:2206.03023,
arXiv:2511.10264, and arXiv:2303.09477. `/deep-research` was not invoked.

**Methods marked ingested:** DAgger / WM-DAgger search-distribution retraining,
post-hoc value-to-cost calibration, cached decision-point Q*/limited-horizon
heuristic evaluation, SLOPE/local-heuristic bounded pruning, and
goal-conditioned offline value.

flagged_for_v427: dagger_search_distribution_value_retraining
(arXiv:1011.0686 + arXiv:2604.11351)

flagged_for_v427: calibrated_value_to_cost_tiebreaker (arXiv:1706.04599)

flagged_for_v427: decision_point_cached_qstar_value_head
(arXiv:2102.04518 + arXiv:2511.10264)

**Bottom line for .427:** first train or calibrate the value head on the live
frontier distribution, then make every live use bounded and cached; only after
those no-regression gates pass should SLOPE-style pruning or goal-conditioned
dense value affect the scored agent.
<!-- EXP4625-OFFLINE-LIVE-BRIDGE-SOTA-END -->

<!-- EXP4637-INTRINSIC-ACTION-EFFECT-SOTA-START -->
## 2026-06-23 Exp 4637 - .427 intrinsic-motivation/action-effect SOTA ingestion - INGESTED

**Status:** INGESTED into `docs/research-notes/intrinsic-motivation-action-effect-literature-2026-06-23.md`.

**Filtered track:** dense online intrinsic motivation and action-effect
prediction for the .427 live-exploration problem: replace raw surprise with
learning progress, suppress noisy-TV transitions, and turn clickability /
action-effect predictions into fewer wasted actions.

**Preconditions:** Hugging Face model API reachability returned `net_ok`.
`scripts/sweep_clusters.py` emitted focused ARC exploration and neural-guided
search URLs. `scripts/sweep_semscholar.py` returned HTTP 429 for all five
focused queries and no S2-only source was promoted. Low-concurrency
WebSearch/WebFetch plus direct arXiv HTTP checks verified arXiv:2604.18701,
arXiv:2509.25438, arXiv:2102.04399, arXiv:1705.05363, arXiv:1810.12894,
arXiv:2601.10904, arXiv:2603.24621, arXiv:2512.24156, and arXiv:2605.05138.
`/deep-research` was not invoked.

**Methods marked ingested:** Curiosity-Critic cumulative prediction-error
improvement, Learning Progress Monitoring, aleatoric-noise curiosity guards,
ICM/RND prediction-error controls, ARC clickability/action-effect expansion,
graph-based exploration, and executable-world-model action-effect planning.

flagged_for_v428: curiosity_critic_learning_progress_dense_reward
(arXiv:2604.18701 + arXiv:2509.25438)

flagged_for_v428: noisy_tv_aware_action_effect_uncertainty_gate
(arXiv:2102.04399 + arXiv:2509.25438)

flagged_for_v428: clickability_action_effect_expansion_prior
(arXiv:2601.10904 + arXiv:2603.24621)

flagged_for_v428: graph_executable_world_model_action_effect_planner
(arXiv:2512.24156 + arXiv:2605.05138)

**Bottom line for .428:** build Curiosity-Critic/LPM-style learning-progress
rewards over the existing action-effect predictor first, add the aleatoric guard
before any scored-agent use, and evaluate graph/executable action-effect
planning only behind matched action-efficiency no-regression gates.
<!-- EXP4637-INTRINSIC-ACTION-EFFECT-SOTA-END -->

<!-- EXP4649-ENERGY-FITNESS-GENERATOR-SOTA-START -->
## 2026-06-23 Exp 4649 - .428 energy-fitness generator SOTA ingestion - INGESTED

**Status:** INGESTED into `docs/research-notes/energy-fitness-generator-literature-2026-06-23.md`.

**Filtered track:** energy-as-fitness QD evolution, macro-action vocabulary
induction, hierarchical subgoal search, and factored executable world models for
the .428 generation wall: turn goal-energy plus action-effect from rankers into
candidate generators for .429.

**Preconditions:** Hugging Face model API reachability returned `net_ok`.
`scripts/sweep_clusters.py` emitted focused ARC exploration and neural-guided
search URLs. `scripts/sweep_semscholar.py` returned HTTP 429 for four focused
queries and no S2-only source was promoted. Low-concurrency WebSearch/WebFetch
plus direct arXiv HTTP checks verified arXiv:2605.28814, arXiv:2308.05483,
arXiv:2504.01915, arXiv:2605.27130, arXiv:2107.07031, arXiv:2502.02962,
arXiv:2302.04693, arXiv:1810.04586, arXiv:1710.11089, arXiv:2604.03208,
arXiv:2506.07255, arXiv:2504.04366, arXiv:2505.10819, and arXiv:2605.05138.
`/deep-research` was not invoked.

**Methods marked ingested:** BES/QD action-sequence evolution, sparse/deceptive
QD controls, empowerment/eigenoption/proto-goal macro induction, hierarchical
latent-world-model/subgoal search, PoE-World factored executable modeling, and
fresh distributed QD scaling.

flagged_for_v429: energy_as_fitness_qd_bes_action_sequence_generator
(arXiv:2605.28814 + arXiv:2308.05483 + arXiv:2504.01915)

flagged_for_v429: macro_action_vocabulary_empowerment_options
(arXiv:2107.07031 + arXiv:2502.02962 + arXiv:2302.04693 + arXiv:1710.11089)

flagged_for_v429: hierarchical_subgoal_search_over_goal_energy
(arXiv:2604.03208 + arXiv:2506.07255 + arXiv:2504.04366)

flagged_for_v429: poe_world_factored_executable_model_planner
(arXiv:2505.10819 + arXiv:2605.05138)

flagged_for_v429: distributed_qd_mutation_ensemble_later_scaling
(arXiv:2605.27130 + arXiv:2605.28814)

**Bottom line for .429:** attempt single-node energy-as-fitness QD over
action-sequence fragments first, pair it with a macro-action vocabulary to
collapse horizon, then add hierarchical subgoals and PoE-World only behind
replay-verified action-effect gates; keep distributed QD as a later scaling arm.
<!-- EXP4649-ENERGY-FITNESS-GENERATOR-SOTA-END -->

<!-- EXP4661-GENERATION-GUIDANCE-SOTA-START -->
## 2026-06-24 Exp 4661 - .429 generation-guidance SOTA ingestion - INGESTED

**Status:** INGESTED into `docs/research-notes/generation-guidance-sota-ingestion-2026-06-24.md`.

**Filtered track:** surviving generation-guidance directions for chaining a
second live level-up after A1 value-routing and A2 energy-fitness QD both
returned no live lift: hierarchical subgoal search, PoE/factored executable
world models, and distribution-shift-corrected value routing.

**Preconditions:** Hugging Face model API reachability returned `net_ok`.
`scripts/sweep_clusters.py --help` exited cleanly. `scripts/sweep_clusters.py`
emitted focused ARC exploration and neural-guided-search URLs.
`scripts/sweep_semscholar.py` returned HTTP 429 for the three focused queries
and no S2-only source was promoted. Low-concurrency WebSearch/WebFetch plus
direct arXiv HTTP checks verified arXiv:2604.03208, arXiv:2506.07255,
arXiv:2504.04366, arXiv:2505.10819, arXiv:2605.05138, arXiv:1011.0686,
arXiv:2604.11351, arXiv:1706.04599, arXiv:2102.04518, arXiv:2605.28814,
arXiv:2308.05483, and arXiv:2504.01915. `/deep-research` was not invoked.

**Dead levers confirmed not re-flagged:** macro-action horizon-collapse
RETIRED; click-heatmap off-centroid generator RETIRED; just-explore
schedule-extraction CLOSED; goal-energy heuristic NULL.

**Methods marked ingested:** hierarchical subgoal search over live E3,
PoE-World/factored executable world-model planning, and DAgger/calibrated
distribution-shift-corrected value routing for subgoal frontiers.

flagged_for_v430: hierarchical_subgoal_e3_frontier_with_distribution_shift_value_routing
(arXiv:2604.03208 + arXiv:2506.07255 + arXiv:2504.04366 + arXiv:1011.0686 + arXiv:2604.11351 + arXiv:1706.04599)

flagged_for_v430: poe_world_factored_executable_subgoal_planner
(arXiv:2505.10819 + arXiv:2605.05138)

**Bottom line for .430:** make hierarchical subgoals the primary .430 input,
with DAgger/calibrated value routing as the affordable low-level guide; keep
PoE-World/factored executable planning as the stronger second candidate when
transition-factor trust is available. Do not re-open macro depth, off-centroid
click coverage, just-explore schedule extraction, or standalone goal-energy
heuristics.
<!-- EXP4661-GENERATION-GUIDANCE-SOTA-END -->

<!-- EXP4673-STRUCTURAL-DEEPENING-SOTA-START -->
## 2026-06-24 Exp 4673 - .431 structural-deepening SOTA ingestion - INGESTED

**Status:** INGESTED into `docs/research-notes/structural-deepening-sota-ingestion-2026-06-24.md`.

**Filtered track:** structural fallback after A1 L2-goal-induction closed with
`single_exemplar_goal_insufficient` and A2 distribution-corrected value-routing
closed with `missing_verifier_gap_live_frontier_not_separated`. The ingestion
deepens the `.429` flagged tracks into implementable `.431` candidates rather
than re-running scalar value routing.

**Preconditions:** Hugging Face model API reachability returned `net_ok`.
`scripts/sweep_clusters.py --help` exited cleanly. `scripts/sweep_clusters.py`
emitted the neural-guided-search and ARC-exploration cluster URLs.
`scripts/sweep_semscholar.py` returned HTTP 429 for the three focused queries
and no S2-only source was promoted. Low-concurrency WebSearch/WebFetch plus
direct arXiv HTTP checks verified arXiv:2604.03208, arXiv:2506.07255,
arXiv:2504.04366, arXiv:2505.10819, arXiv:2605.05138, arXiv:2605.12913,
arXiv:2604.11351, and arXiv:1011.0686. `/deep-research` was not invoked.

**Methods marked ingested:** hierarchical subgoal search over live E3, failed
search-tree subgoal proposal with value tie-breaking, PoE-World/factored
executable world-model planning, and WM-DAgger trust-weighted
subgoal-conditioned value routing.

flagged_for_v431: hierarchical_subgoal_e3_frontier_with_a1_a2_tiebreakers
(arXiv:2604.03208 + arXiv:2506.07255 + arXiv:2504.04366 + arXiv:2605.12913 + arXiv:1011.0686)

flagged_for_v431: poe_world_factored_executable_subgoal_planner
(arXiv:2505.10819 + arXiv:2605.05138)

**Bottom line for .431:** make the hierarchical subgoal layer the primary
structural move. Use A1's induced goal signal as a subgoal proposer, A2's value
head as a bounded local tie-breaker, and live E3 as the executor. Keep PoE-World
as the stronger alternate when enough transition trust exists to factor effects.
<!-- EXP4673-STRUCTURAL-DEEPENING-SOTA-END -->

<!-- EXP4685-DIRECTED-EXPLORATION-SOTA-START -->
## 2026-06-24 Exp 4685 - .432 directed-exploration SOTA ingestion - INGESTED

**Status:** INGESTED into `docs/research-notes/directed-exploration-sota-ingestion-2026-06-24.md`.

**Filtered track:** fallback beyond the `.431` A1 hierarchical subgoal search
and A2 PoE-World planner. A1 closed at `wall_diagnosis=l1_first_contact` with
`value_head_still_not_separating`; A2 closed with
`candidate_generation_coverage_factored=0.0` and `experts_overfit_prefix`.
The live gap is now action-proposal coverage: make a winning L1 trajectory
appear before A1/A2 can select, decompose, or plan over it.

**Preconditions:** Hugging Face model API reachability returned `net_ok`.
`scripts/sweep_clusters.py --help` exited cleanly. `scripts/sweep_clusters.py`
emitted the ARC exploration and neural-guided-search cluster URLs.
`scripts/sweep_semscholar.py` returned HTTP 429 for the four focused queries
and no S2-only source was promoted. Low-concurrency WebSearch/WebFetch plus
direct arXiv HTTP checks verified arXiv:2002.06038, arXiv:1810.12894,
arXiv:2005.05960, arXiv:1712.06560, arXiv:2502.10077, arXiv:2603.02045,
arXiv:2102.11137, and arXiv:2505.10819. `/deep-research` was not invoked.

**Methods marked ingested:** episodic controllable-novelty policy family,
Plan2Explore-style disagreement plus empowerment, novelty/QD replayable action
prefix archives, strategy-guided language-action exploration, and
program-synthesis action-effect proposal filtering.

flagged_for_v432: controllable_novelty_e3_proposal_policy
(arXiv:2002.06038 + arXiv:1810.12894 + arXiv:2603.02045)

flagged_for_v432: program_synthesis_action_effect_proposal_filter
(arXiv:2505.10819 + arXiv:2102.11137)

**Bottom line for .432:** build the controllable-novelty proposal policy first
because it directly attacks the L1-first-contact distribution gap. Add the
program-synthesis action-effect filter as the second arm when enough trusted
prefix transitions exist to avoid the A2 `experts_overfit_prefix` failure.
<!-- EXP4685-DIRECTED-EXPLORATION-SOTA-END -->

<!-- EXP4697-AMORTIZED-EXPLORATION-SOTA-START -->
## 2026-06-24 Exp 4697 - .433 amortized-exploration SOTA ingestion - INGESTED

**Status:** INGESTED into `docs/research-notes/amortized-exploration-sota-ingestion-2026-06-24.md`.

**Filtered track:** fallback beyond `.432` per-game directed exploration. A1
controllable novelty closed with `winning_prefix_still_not_proposed`; A2
program synthesis closed with `heldout_transitions_too_sparse`. The next wall
is `hidden-game transfer`: first-contact behavior must transfer to unseen scored
games instead of being rediscovered from scratch on each game.

**Preconditions:** Hugging Face model API reachability returned `net_ok`.
`scripts/sweep_clusters.py --help` exited cleanly. `scripts/sweep_clusters.py`
emitted the ARC exploration and neural-guided-search cluster URLs.
`scripts/sweep_semscholar.py` returned HTTP 429 for the four focused queries
and no S2-only source was promoted. Low-concurrency WebSearch/WebFetch plus
direct arXiv HTTP checks verified arXiv:2210.14215, arXiv:2310.09971,
arXiv:2601.19810, arXiv:1802.07245, arXiv:2008.02790, arXiv:2603.03680,
arXiv:1901.10995, and arXiv:2004.12919. `/deep-research` was not invoked.

**Methods marked ingested:** in-context exploration-prior distillation,
self-imposed-goal / structured-noise meta exploration, decoupled
meta-explore/exploit language-agent adaptation, and Go-Explore return-then-
explore archive upgrade.

flagged_for_v433: in_context_exploration_prior_from_first_contact_traces
(arXiv:2210.14215 + arXiv:2310.09971 + arXiv:2601.19810)

flagged_for_v433: arc_go_explore_return_then_explore_archive_upgrade
(arXiv:1901.10995 + arXiv:2004.12919)

**Bottom line for .433:** build the in-context exploration prior first because
it directly amortizes rare successful first-contact behavior across games. Keep
the Go-Explore archive as the structural companion because it already exists in
`arc_go_explore.py` and provides replayable return points for deeper probing.
<!-- EXP4697-AMORTIZED-EXPLORATION-SOTA-END -->

<!-- EXP4709-STRUCTURED-WORLD-MODEL-SOTA-START -->
## 2026-06-25 Exp 4709 - .434 structured-world-model SOTA ingestion - INGESTED

**Status:** INGESTED into `docs/research-notes/structured-world-model-active-probing-sota-ingestion-2026-06-25.md`.

**Filtered track:** fallback beyond `.433` A1/A2. A1 closed with
`object_centric_perception_no_new_level_residual_offpath_calibration_insufficient` and A2 closed with `amortized_prior_go_explore_no_coverage_gain_residual_logged`. The next wall is the
`structured-world-model / active-probing next wall`: the explorer needs an induced object-relational transition model
that it can plan in, plus targeted probes that confirm or refute mechanic
hypotheses.

**Preconditions:** Hugging Face model API reachability returned `net_ok`.
`scripts/sweep_clusters.py --help` exited cleanly. `scripts/sweep_clusters.py`
emitted the ARC neural-guided-search, action-effect, and world-model cluster
URLs. `scripts/sweep_semscholar.py` returned object-centric and
active-hypothesis-testing arXiv IDs. Low-concurrency WebSearch/WebFetch plus
direct arXiv HTTP checks verified arXiv:2410.08822, arXiv:2511.02225,
arXiv:2601.06604, arXiv:2511.06136, arXiv:2307.02427, arXiv:2210.13455,
arXiv:2506.01876, and arXiv:2309.08477. `/deep-research` was not invoked.

**Methods marked ingested:** factored object-relational executable transition
model, object-model MCTS with epistemic probe planning, hypothesis-driven
active probe loop, and object-world-model drift guardrails.

flagged_for_v434: factored_object_relational_executable_world_model
(arXiv:2511.02225 + arXiv:2410.08822 + arXiv:2307.02427)

flagged_for_v434: object_model_mcts_with_epistemic_probe_planning
(arXiv:2601.06604 + arXiv:2210.13455)

flagged_for_v434: hypothesis_driven_active_probe_loop
(arXiv:2506.01876 + arXiv:2309.08477)

**Bottom line for .434:** build the factored object-relational executable
world model first, then use object-model MCTS and active probes to decide which
uncertain mechanics deserve live actions. Keep the drift guardrail as the
failure detector so object-centric perception does not create a false sense of
control.
<!-- EXP4709-STRUCTURED-WORLD-MODEL-SOTA-END -->

## INGESTED 2026-06-27: arXiv:2603.09906 "Thinking to Recall" (Google Research)
Reasoning unlocks parametric knowledge via (1) computational buffer + (2) factual priming; KEY: one
hallucinated intermediate fact significantly lowers final-answer accuracy, and self-generated facts are an
UNFIXED reliability risk. Mapped onto the Carnot VERIFIER CORE (not the ARC generation wall):
docs/research-notes/thinking-to-recall-verifier-gated-reasoning-sota-ingestion-2026-06-27.md. Two outputs:
(a) paper-v6 corroboration quote bank (FLAGGED, re-verify before publication); (b) scoped experiment
VERIFIER-GATED REASONING (process verifier filters/repairs intermediate facts -> reliable factual priming;
arms A/B/C, CI95 gate, GPU 1). Flagged for a verifier-core milestone, NOT the ARC sprint.
UPDATE 2026-06-27: de-risked. #1 risk RESOLVED POSITIVE (no-retrieval model-native self-consistency
discriminates Qwen3.5-9B hallucinations, AUROC 0.759 CI95 [0.698,0.818]); but #2 risk FAILED the gate
(priming headroom +2.4pp CI95 [-0.024,+0.072] straddles 0 -> paper's effect does not significantly
reproduce at 9B/SimpleQA-Verified) -> full A/B/C build MOOT for this model/corpus (revive only with a
32B-class model or a higher-headroom corpus). results/verifier_gated_reasoning_{derisk_hardened,headroom}.json.

## INGESTED 2026-06-27: "Forward Self-Models" (jagilley.github.io, NOT on arXiv) -> white-box-complementary tier
A small aux transformer predicts a model's later-layer activations from earlier ones; the residual ("computational
novelty") tracks COMPUTE COMPLEXITY (attention-entropy r=+0.332) but is UNCORRELATED with prediction difficulty
(d~=+-0.03) and the author makes NO uncertainty/hallucination claim. KEY: this REFUTES the tempting bridge
"residual = white-box hallucination signal for the verifier" (cite the dissociation every time it recurs). Honest
use = ONE taxonomy paragraph (state[Cognometry]/feature[Silico]/COMPUTATION[this]) in the white-box-COMPLEMENTARY
tier, NEVER the black-box core (decentralization rules 1+7; also cross-vendor transfer ceiling cos 0.043 + no
black-box spillover = most-degraded of the three). NO build; ARC=no-help; Phase-3=orthogonal; amortized-theme=echo
only (do NOT add to corroboration list). docs/research-notes/forward-self-models-white-box-complementary-sota-ingestion-2026-06-27.md

## INGESTED 2026-06-27: "Socratic agents for autonomous scientific discovery" / AHOIS (arXiv:2606.26722) -> autonomous-loop PEER
Multi-agent AI scientist; centerpiece = a physics-critic doing an explicit 4-step Socratic interrogation (causal-question ->
constraint-check -> counterexample-gen -> falsification-criteria); optical-fibre demo (MNIST 76.97%/Fashion-MNIST 83.17%), NO
code. Carnot placement: a PEER of the conductor/outer-loop (cluster with Self-Harness/W2S/Sakana-DGM), NOT the verifier core.
WEAKEST peer on the metric-gaming axis: cataloged own-system failure modes W2S=4, Sakana=2, AHOIS=0 -> corroboration, NOT
validation; do NOT stack as equal gaming-resistance evidence. The 4-step critic maps 1:1 onto machinery Carnot already runs
(adversarial_verify / CEGIS exp4872 / check_false_negative_risk) -> the borrow is PROMPT STRUCTURE ONLY (force a generated
counterexample in the milestone-close audit prompts), untested in our domain (needs an A/B), increases the Layer-1.5
audit-integrity-guard surface (gate behind it), LLM-judge/audit tier ONLY never the energy core. NO build; ARC=no-help
(selection device, wall=generation); no codebase to extend; paper-v6 cite gated on the two-source rule.
docs/research-notes/ahois-socratic-agents-autonomous-loop-peer-sota-ingestion-2026-06-27.md

<!-- EXP4900-SOTA-INGESTION-V452-FRONTIER-START -->
## Exp 4900 - .452 representation fork SOTA ingestion - INGESTED

- Honest verdict: `success_sota_ingestion_v452_frontier_mapped`
- Aimed at A1 fork: `VALUE_GAP_REPRESENTATION_INVARIANT`
- A1b fork: `VALUE_GAP_REPRESENTATION_INVARIANT_HARD`
- Branch: `third_representation_class_or_operator_escalation`
- Banned channel note: `/deep-research` not invoked; no solve claim; no model load; retired classes excluded.

### Methods
- Latent-action adaptable representation interface (arXiv:2503.18938): Infer latent action tokens from cold ARC transitions, align E3 legal controls to those tokens, and feed the latent-action state into the held-out transition scorer without converting it through the failed decision-need or action-prefix table formats.
- Reverse-counterfactual representation targeter (arXiv:2505.08073): For each hard A1/A1b transition miss, ask a reverse model for the missing pre-state or register fact that would make the desired action effect valid, then materialize only verifier-checkable facts into a new representation probe.
- Verification-calibrated abstraction substrate (arXiv:2602.23997): Insert a persistent abstraction state beside the executable engine, attach verifier-calibrated confidence to each abstract fact, and let only calibrated facts influence held-out transition prediction.

### Planner Flags
- flagged_for_v452: latent_action_interface (arXiv:2503.18938)
- flagged_for_v452: reverse_counterfactual_targeter (arXiv:2505.08073)
- flagged_for_v452: verification_calibrated_abstraction (arXiv:2602.23997)
<!-- EXP4900-SOTA-INGESTION-V452-FRONTIER-END -->

<!-- EXP4911-SOTA-INGESTION-V453-FRONTIER-START -->
## Exp 4911 - .453 wall and verifier-pivot SOTA ingestion - INGESTED

- Honest verdict: `success_sota_ingestion_v453_frontier_mapped`
- Aimed at A1 fork: `WALL_DEEPER_THAN_VALUE_PREDICTION`
- A1b fork: `VALUE_GAP_REPRESENTATION_INVARIANT_4_CLASSES`
- Branch: `wall_survives_four_representations_plus_env_grounding`
- Banned channel note: `/deep-research` not invoked; no solve claim; no model load; nulled classes excluded.

### Final ARC Wall Diagnostics
- Causal state-abstraction wall diagnostic (arXiv:2401.12497): Run an offline causal-abstraction audit over failed A1/A1b transitions: identify which variables would have to be retained for changed-cell value prediction, then stop unless the abstraction exposes a new observable state variable within the sprint budget.
- Local causal SSM world-model diagnostic (arXiv:2505.02074): Fit a tiny causal SSM on observed prefixes and compare its inferred causal graph against the failed engine facts; accept only a diagnostic report unless it predicts held-out changed values without extra actions.
- Object-level masked Causal-JEPA diagnostic (arXiv:2602.11389): Mask object-level latent facts in failed transitions and ask whether observed context can reconstruct the changed values; use the result to label the wall observable or unobservable rather than launching another planner.
- Interpretable causal world-model extraction (arXiv:2504.07257): Extract symbolic object-transition equations from the same failed games and compare them to the induced engine; use mismatches to decide whether the sprint should stop or whether a single observable state variable was missed.

### Post-Sprint Verifier-Moat Pivot
This is the post-sprint verifier-moat pivot required after the ARC sprint retires.
- Distributional energy verifier for structured reasoning (arXiv:2605.18871): Port the FoVer evaluation harness to MuSR/TravelPlanner-style structured reasoning rows, score candidates with a small distributional energy verifier, and compare against self-consistency and an LLM judge.
- Small energy outcome reward model (arXiv:2505.14999): Train or calibrate a tiny EORM-style scorer on non-FoVer reasoning traces, then run best-of-N selection against self-consistency at matched sample and cost budgets.
- Tool-aware scientific process reward model (arXiv:2606.04579): Build a small FoVer-style scientific-tool trace corpus, label step selection/execution/interpretation errors, and test whether a PRM catches errors missed by generator self-checks.
- Environment-aware data-analysis process verifier (arXiv:2604.24198): Use DataPRM-style active trace checks on non-saturated data-analysis tasks, with the verifier probing intermediate states and scoring correctable versus irrecoverable mistakes.

### Planner Flags
- flagged_for_v453: causal_state_abstraction_wall_diagnostic (arXiv:2401.12497)
- flagged_for_v453: distributional_energy_verifier_pivot (arXiv:2605.18871)
- flagged_for_v453: tool_aware_science_prm_pivot (arXiv:2606.04579)
<!-- EXP4911-SOTA-INGESTION-V453-FRONTIER-END -->

### .453 PLANNER DECISION (outer-loop Claude Opus 4.8, 2026-06-28)

Selected the .453 milestone from the exp4911 frontier map above, under the still-active
ARC sprint (through 2026-06-30). Decisions:

- **ARC headline = the causal-state-abstraction wall DIAGNOSTIC (arXiv:2401.12497, D's
  priority-1).** .452 settled the wall is representation-invariant across FOUR world-model
  representations PLUS env-grounded real-env-value search (A1 exp4903 delta -0.04 B1-trusted;
  A1b exp4904 VALUE_GAP_REPRESENTATION_INVARIANT_4_CLASSES). The diagnostic does NOT propose
  representation #5; it CLASSIFIES the minimal causal state variables behind each failed
  first-win as observable-from-the-ARC-interface (a fixable representation gap) vs hidden
  (representation-invariant by construction). This is the "final ARC closure check before
  2026-06-30" the exp4911 planner_instruction conditions on, and it is mandated by the
  still-active sprint (majority-ARC). Publishable closure for the FoVer paper's ARC section.
- **Post-6/30 pivot = distributional energy verifier (arXiv:2605.18871, post-sprint priority-1).**
  The .453 SOTA-ingestion slot does NOT re-map (exp4911 already mapped); it MINIMALLY SCAFFOLDS
  the offline FoVer->MuSR distributional-energy-verifier-vs-self-consistency harness + dry-run,
  so the loop executes the pivot the instant the sprint retires.
- **Did NOT carry forward** (all nulled/retired per exp4911): energy-as-ARC-lever, TTA-on-code-engine,
  stronger-local-code-inducers, decision-need targets, action-prefix latents, coverage/vocabulary,
  exploration, selection/ranking, perception-from-grid, representation #5.

<!-- EXP4940-DISTRIBUTIONAL-ENERGY-VERIFIER-EXECUTABLE-SPEC-START -->
## Exp 4940 - Distributional Energy Verifier Executable Spec - INGESTED

- Honest verdict: `success_distributional_energy_verifier_pivot_executable_spec_ready`
- Cited SOTA papers: arXiv:2605.18871, arXiv:2504.16828, arXiv:2502.01989
- Bottom line for the post-6/30 roadmap: build the decomposed-energy LoRA-ensemble scorer on top of Carnot's FoVer analytical penalties; keep ThinkPRM as a matched-compute comparator and VFScale as a dense-energy ablation, not the immediate drop-in verifier.
- Guardrail: readiness/design only; no moat-proven claim and no real benchmark execution.

### flagged_for_next_milestone
- flagged_for_next_milestone: decomposed_energy_lora_ensemble_with_fover_penalties (arXiv:2605.18871)
- flagged_for_next_milestone_comparator: thinkprm_generative_prm_comparator (arXiv:2504.16828)
- flagged_for_next_milestone_ablation: vfscale_intrinsic_energy_dense_reward (arXiv:2502.01989)
<!-- EXP4940-DISTRIBUTIONAL-ENERGY-VERIFIER-EXECUTABLE-SPEC-END -->
