# Carnot Research Roadmap v634: Executable Grounding and Useful Online Learning

**Created:** 2026-09-10

**Milestone:** `2026.09.634`

**Status:** Planned; V633 is complete. Activation remains the conductor's next transition.

**Supersedes:** the stale V632 design used during V633 execution

**Task contract:** exactly 13 tasks, `exp7192` through `exp7204`, in the order below

**Execution authority:** `research-roadmap-next.yaml` before activation; matching `research-roadmap.yaml` afterward

## What V633 Proved

All thirteen tasks reached terminal records. That does not mean thirteen scientific successes.
The primary artifacts, rather than conductor OK labels, establish the following:

| Evidence | Finding | Next decision |
|---|---|---|
| Exp7179 | The active YAML had thirteen V633 tasks. Its design still named V632. | Write and validate both complete files before handoff. No science branch depends on the advisory audit. |
| Exp7180/7181 | A 192-row exact fixture and real Qwen3.8 capture completed. | Reuse the runtime and fixture builder. Readiness proves no verifier benefit. |
| Exp7182 | Energy accuracy was 54/128 (0.421875), versus direct 72/128 (0.5625). Paired CI95 for the difference was [-0.1796875, -0.1015625]. There were 50 parse failures and 24 harmful flips. | Retire the frozen threshold rule. Test separate extraction, explicit execution and unknown states. |
| Exp7183/7184/7185 | Delayed feedback, template addition, revocation and rollback ran. Adaptive future error was 0.20 versus static 0.00. Shuffled credit changed no decisions. | Preserve the null. Test unknown-parameter acquisition under equal initial information and bounded pending feedback. |
| Exp7186 | ARC blocked on the nonexistent/empty `arc_eval_runner.py` prerequisite. | Use the existing scored policy and `scripts/arc_leaderboard_eval.py:run_game`. The later known-issues diagnosis says the named runner never existed; do not try to restore it. |
| Exp7187/7188 | Pair swaps preserved enumerated slice laws. Naive quantization changed 162/162 precision-conditioned laws; corrected transitions preserved the intended target. | Preserve CPU correctness. Measure real deployment and correction costs. |
| Exp7189 | Compiled Rust matched explicit Python transitions. End-to-end subprocess timings missed the PRD's 10x requirement; no PyO3 execution was claimed. | Test a persistent compiled boundary, with phase attribution and the old bridge retained as a control. |
| Exp7190/7191 | Board visibility was documented. The capstone finished its matrix with blocked ARC science. | Retain per-board dispositions and ungated synthesis. A complete blocked matrix is terminal. |

Source artifacts are `results/experiment_7179_v633_contract_receipt.json` through
`results/experiment_7191_v633_capstone.json`, using their exact task-declared names
in the completed V633 YAML. No historical artifact is rewritten by this plan.

## Three Largest Gaps to the PRD Vision

1. **Correct extraction is still missing.** FR-12 requires constraints that describe
   the actual source. A valid schema and a low score did not establish this in V633.
   Phase 2 tests separate source/claim extraction and executable relations with abstention.
   The authority stays outside the generation and selection processes.
2. **Learning does not yet earn its complexity.** FR-11 requires improvements on future
   inputs with immutable evaluation and rollback. V633's static baseline had no future
   error headroom. Phase 3 tests acquisition of genuinely unknown parameters and pays
   for delayed feedback. It distinguishes common learning benefit from admission-policy benefit.
3. **Research mechanisms lack measured deployment value.** FR-05/FR-08 and NFR-01 require
   a usable Rust/Python boundary with measured performance. Live ARC also needs reachable,
   tested mechanisms. Phase 1 measures the queued live tool path; Phase 4 measures the
   in-process sampler boundary and correction costs. Public-game transfer remains a proxy.

## Research Inputs and Selection

The V634 source refresh was added to `research-references.md` before this design.
It records all requested discovery channels and access limitations.

| Primary source | Local hypothesis | Limit |
|---|---|---|
| [Capacity-constrained delayed feedback](https://arxiv.org/abs/2606.11711) | Pending-label capacity can determine what constraint memory learns. | Finite hypothesis elimination does not inherit the paper's convex regret theorem. |
| [Delayed-feedback evaluation diagnostics](https://arxiv.org/abs/2608.11560) | Test alignment and actual policy dependence before reporting adaptation value. | A CPU controlled stream is not a real-model learning result. |
| [Symbolic grounding](https://arxiv.org/abs/2609.05025) | Separate source relations from claim parsing, then execute them. | SQL/typed syntax does not itself certify source fidelity. |
| [Structure snowballing](https://arxiv.org/abs/2604.06066) | Grammar-only controls distinguish formatting from semantics. | A newer prompt alone cannot reopen a failed mechanism. |
| [Coupling distortion](https://arxiv.org/abs/2609.07347) | Price the exact correction needed after quantization. | Host measurements are not soft-spin device results. |
| [KAN abstraction](https://arxiv.org/abs/2602.06737), [KAN-SAs](https://arxiv.org/abs/2512.00055) | Preserve certification and spline-deployment options. | Defer another ranker until extracted features show useful signal. |
| [Extropic Z1T](https://extropic.ai/writing/z1t) | Evaluate host correction and transfer break-even conditions. | Fixed degree alone does not prove graph embedding; vendor estimates are not local timings. |

EBT, ARM/EBM equivalence, neural constraint methods, OpenReview, Semantic Scholar
citation trails, Hugging Face Papers, GitHub trending and Kona were checked.
The bibliography records exact URLs. OpenReview forum access hit browser checks.
Neither EBT nor Kona supplies a verified drop-in Qwen constraint authority.
The retired generated-text scorer and repair-stack programs remain closed.

## V634 Architecture

```text
                      immutable public source/claim panel
                                    |
                          independent Qwen3.8 calls
                         source / claim / direct (7196)
                                    |
                 typed executor + unknown state (7195)
                                    |
                  independent authority audit (7197)
                         full denominator + cost

  real ARC env -> scored E3 policy -> selfparse tool loop (7193)
                 per-game knowledge withheld         |
                                            real tool/gap receipts
                                                      |
                                      causal generalization audit (7194)

  sealed hidden-parameter stream (7198) -> bounded pending queue
                                                      |
                                      constraint acquisition (7199)
                                                      |
                                 cold reload / poison / rollback (7200)

  existing exact slice kernel -> persistent PyO3 (7201)
                                           |
                                   cost + sample quality (7202)

  existing corrected kernel -> CPU cost envelope + board receipts (7203)

  source/contract audit (7192) and final matrix (7204) observe all branches.
```

Energy certifies or scores representations. It does not generate answers in this design.
Exact execution gains are marked circular when they use the correctness authority.
An independent implementation does not by itself make an exact check oracle-distinct.

## Phase 1: Live tool discovery and execution contracts

Run the overdue direct selfparse task early. Its bounded session uses the real scored policy. An independent audit counts actual calls and banked progress. The source/contract task is advisory.

### Exp7192: V634 source ingestion and exact execution contract

V633 activated thirteen tasks against a V632 document. This plan ships both sources together. Audit them independently and ingest only post-planning source changes. This advisory task gates no science.

- Select the YAML whose milestone equals 2026.09.634. Before activation use research-roadmap-next.yaml; afterward use research-roadmap.yaml. Freeze its bytes and the design bytes. Independently parse the exact contract table. Compare thirteen IDs, order, titles, deliverables and gates. Do not compare the staged design against the still-active V633 YAML.
- Run schema, prior-failure, exclusion, invented-path, ARC-floor and gate-declaration checks. Confirm producer fields exist in each upstream REQUIRED ARTIFACT FIELDS block. Exercise the real conductor gate evaluator on temporary synthetic artifacts in /tmp. Label those fixtures as validation inputs; they are not research results. Check both field=1 and field=0, missing field and quarantined input. Record the current quarantine behavior separately: evaluate_gates currently evaluates fields and does not reject flagged_adversarial by itself. The experiment-level precondition must reject quarantined upstream data. Do not claim this pre-gate provides that protection or edit conductor/QA code.
- Read the V634 planning refresh in research-references.md. Check at most five primary papers through low-concurrency requests. Include arXiv:2606.11711, 2608.11560, 2609.05025 and 2609.07347. Record URL, version/date, access outcome, method boundary and target task. Check Extropic and Kona for a dated change. Record no delta when appropriate. Update references and research-studying.md only for real changes. Do not launch a research-agent fan-out.
- Emit source_contract_complete_score=1 only after both source mapping and structural checks complete. A readable mismatch is disqualified. An inaccessible external paper may be an explicit source-limit row if cached primary evidence suffices; do not invent its contents. Preserve the chosen contract under results/raw/experiment_7192/ for the capstone.

**Deliverable:** `results/experiment_7192_v634_source_contract.json`.

### Exp7193: Live ARC direct-tool generalization measurement

The 2026-09-10 known-issues entry queues a direct selfparse tool-loop run. The supervisor tool arm is a no-op. Exp7186 instead required an invented arc_eval_runner.py. Use the shipped scored policy and real offline environment with per-game knowledge withheld.

- Trace the actual E3AgentPolicy/make_carnot_agent induction call and tool_gap_events attachment. Use scripts/arc_leaderboard_eval.py:run_game or its real environment loop. Verify the path exists and imports. Do not create a replacement named arc_eval_runner.py. Set CARNOT_ARC_INDUCE_TOOL_LOOP=selfparse directly. Keep CARNOT_ARC_SUPERVISOR_TOOL_ARM unset; do not change ARM_ORDER or repair that separate arm here.
- Freeze one r11l session and seed 7193001. Use an adapter-withheld live policy view. The evaluator may read the registry before launch to record previously reproduced levels. Deny the policy access to per-game adapters, game source, registry contents and solved trajectories. Allow the environment process its own executable source. Verify isolation before model work. This run measures direct tool discovery on a known public game with its solution knowledge withheld; it is not a new solve.
- Set CARNOT_ARC_INDUCE_N_CTX=49152 before proposer construction for one single-stream selfparse runner. The shipped 106496 default does not fit a 24 GiB card. Verify actual context slots, prompt size, KV allocation and free VRAM through the existing fit check. Preserve the validated 4096 completion budget and current sampler settings. Allow up to 4000 environment actions and 3600 seconds for this one session, with 2400-second induction-call deadlines. The shipped Qwen path has recorded roughly 1730-second median induction latency. Do not replace this with short forced cancellations. Reserve the remaining task budget for setup and checks.
- Capture task-owned model identity, real completions, tool calls, induction IDs, raw gap events, banked levels, actions and elapsed time. Count new inductions separately from cited historical receipts. Ten accumulated real tool-loop inductions remains the evidence target; this bounded task does not promise ten new inductions. Empty gaps after real calls are valid. A zero-call or timed-out run is a complete limited-evidence null, not a success claim or retryable partial. Set arc_tool_measurement_complete_score=1 when the scheduled session has its terminal receipt. Set arc_tool_engagement_score=1 only when a real tool loop executes and returns a terminal induction result. solve_provenance=live_agent_self_discovery. Report no paired efficacy or official leaderboard score. Do not submit anything.

**Deliverable:** `results/experiment_7193_v634_arc_direct_tool.json`.

### Exp7194: ARC tool-gap and banked-progress causal audit

Tool engagement and banked progress are different outcomes. The operator queue requires running the refinement tools on the new receipts. Audit generalization evidence without promoting shared supervisor credit into causal benefit.

- Load Exp7193 from its declared deliverable and raw manifests. Run both existing refinement tools against only these new receipts, using scratch output paths. Keep earlier r11l two-call evidence as historical context, not pooled new volume.
- Recompute the new session's progress and cost rows. A supervisor helped flag is not banked progress. Join gaps to actual induction IDs and tool calls. Preserve zero-gap rows and parser failures. Audit whether a real missing capability was requested. Do not infer gaps from model prose. This one-session volume probe supports no paired causal efficacy estimate.
- Produce a bounded generalization recommendation: a reusable tool candidate with an exact runtime counterexample, or an honest no-gap count. Preserve new versus historical induction denominators and whether engagement completed. Do not modify the curated arm table or invent a new tool to satisfy the slot. Set arc_gap_audit_complete_score=1 after all joins and recomputations, regardless of sign. A zero-engagement upstream remains an audited null, not live discovery evidence.

**Deliverable:** `results/experiment_7194_v634_arc_gap_audit.json`.

## Phase 2: Executable source grounding

Prototype typed semantics and a new immutable panel before capture. Measure real Qwen extraction once. Audit full-denominator correctness and total cost independently. No tuning uses held-out labels.

### Exp7195: Typed source execution and abstention prototype

Exp7182 lost 14.06 percentage points against direct decisions. It made 24 harmful flips and had 50/128 parse failures. Replace the frozen overlap/threshold rule with an explicit source-relation executor and unknown semantics. Use old data for diagnosis only.

- Reconstruct the old errors by parse failure, source omission, entity binding, relation orientation, negation and unresolved evidence. Quote raw-row hashes for each diagnosed category. Do not tune or rescore the frozen old rule into a win.
- Implement a small typed relation executor under python/carnot/verify/experiment_7195_source_relation_executor.py. Accept explicit entity IDs, relation operators, polarity and source offsets. Check offsets against raw source bytes. Execute direction and negation explicitly. Missing, contradictory or ambiguous mappings yield unknown. Unknown does not mean false. Keep all uncertainty states visible.
- Generate a new sealed evaluation panel using the shipped exact fixture builder: 16 calibration base cases and 32 held-out base cases, each with four paired variants, total 192 rows. Use new seed 7195001 and fresh entity identities. Freeze family and template grouping before generation. Old V633 rows are development evidence only. Separate public source/claim views from evaluator-only truth and edit metadata.
- Before any LLM run, test the executor on exact typed inputs and adverse mutations: reversed arguments, negation, removed evidence, duplicate names and invalid offsets. Require zero deterministic semantic disagreements for the supported fragment. Unsupported expressions must abstain. Emit typed_executor_ready_score=1 only when this passes. Freeze the three atomic prompts, the grammar-only control, scoring policy and sample budget. No corpus label, expected edit or canonical answer may enter the producer view.

**Deliverable:** `results/experiment_7195_v634_typed_grounding.json`.

### Exp7196: Qwen3.8 independent source and claim capture

The typed executor is ready. Test whether short independent extraction calls preserve semantic content better than the old joint response. The capture task measures transport and cost, not verifier value.

- Load Exp7195's frozen public view and prompt hashes. Generate source tuples from the source alone, claim tuples from the claim alone, and a separate direct support judgment from both. Do not let the source call see the claim or let either extraction call see the direct judgment. Use grammar constraints only to enforce the shipped typed output syntax.
- Run all 192 rows, one draw per call, with fixed output budgets of 128 source tokens, 64 claim tokens and 16 judgment tokens. Use the shipped non-thinking configuration only if the actual server supports it; record exact parameters. These are atomic extraction budgets, not a token-budget escalation. No parser retries or regenerated missing rows. Count invalid, truncated and unknown outputs on the full denominator.
- Cap capture at 2400 seconds and requests at 60 seconds. Persist every raw completion and request failure before the next unit. Reuse identical source calls only by exact public-source hash and record cache hits. Report cold and amortized costs separately. Use one leased model instance; no unowned server reuse.
- Set atomic_capture_complete_score=1 when every scheduled row has a terminal call receipt. A completed parse-poor bank is a scientific null available for audit, not blocked or partial. Seal raw hashes and the full scoring denominator. Preserve checkpoints if the task's own work remains unfinished; never put a running shell in the terminal deliverable.

**Deliverable:** `results/experiment_7196_v634_qwen_atomic_capture.json`.

### Exp7197: Typed grounding value and independent semantic audit

A parser success is not semantic correctness. Compare typed execution with direct decisions and syntax-only controls. Test whether abstention avoids harmful flips without hiding errors by reducing coverage.

- Use Exp7195's sealed calibration/evaluation split and Exp7196's raw calls. Compare direct judgment, grammar-validity-only, typed execution with unknown abstention, lexical overlap, and shuffled-source typed execution. Keep all 128 evaluation rows including parse failures. No tuning after evaluation labels open.
- Report end-to-end correct/128, parse rate, abstention, conditional accuracy, false accepts, false rejects, harmful flips, rename consistency, edit sensitivity and cold/amortized latency. Bootstrap 10000 paired draws by 32 base cases, stratified by family. Repeated variants are not independent samples.
- The preregistered value gate requires a strictly positive paired CI lower bound for correct/128 over direct judgment, no false-accept increase, and at least 0.60 coverage. An efficiency claim instead requires accuracy noninferiority within 0.02 with its CI, equal coverage, and at least 2x measured latency improvement including extraction. If neither passes, retain a complete null. Label pilot scope; these gates do not establish broad model performance.
- In a fresh evaluator process, recompute raw predictions with the independent authority and repeat argument reversal, semantic deletion and shuffled-source controls. Confirm label bytes cannot enter extraction or policy selection. Explicitly decide verifier_is_oracle: different code alone does not make equivalent exact checks oracle-distinct. Any gain using the same complete correctness authority is circular_positive and execution-grounded, never a learned-verifier moat. Set grounding_audit_complete_score=1 after complete audit, even for a null.

**Deliverable:** `results/experiment_7197_v634_grounding_value_audit.json`.

## Phase 3: Continuous acquisition with delayed feedback

Build an equal-information stream before the learner. Add constraints from released evidence under fixed capacity. Check cold persistence and causal dependence. Keep the full-information oracle separate.

### Exp7198: Sealed constraint stream with bounded pending feedback

Exp7184 could not beat its zero-error static future baseline. arXiv:2606.11711 motivates a different question: learning unknown constraint parameters when pending feedback consumes finite capacity. Do not relabel the old ceiling as headroom.

- Keep the original static-rule result unchanged. Build a new CPU mechanism stream with four numeric predicate families and hidden parameters from a finite 33-value domain. The public family grammar is shared. Parameters, future labels, change times and feedback delays are evaluator-only. Neither the learner nor the frozen baseline may read them.
- Freeze ten independent stream seeds, 1024 events each: 128 warmup, 128 online validation, 512 prospective events, 128 recurrence events and 128 poison/rollback events. Use stable, shifted and recurrent parameter regimes. Keep the exact full-information oracle as an explicitly unattainable upper bound. Generate all data from fixed seeds before running any learner; never condition corpus selection on a method winning. Use fixed seed-derived family ordering with equal family counts in each window. The initial 128 warmup events obey the same admission and delay budget for every deployable arm; freeze their shared initialized state at the boundary. The designated validation segment only supplies labels actually requested and released.
- All deployable arms receive identical warmup evidence and the same public observations. Freeze the static policy after warmup. Online arms can request a label for at most one of every four arriving events, within 64 KiB memory and pending capacities C in {1,4,16}. Test constant delays 0,4,16 and one frozen burst-delay schedule. Pending records occupy space until release or explicit eviction. Evicted labels are lost to that arm; the evaluator retains them for scoring only. Divide arriving observations into fixed four-event blocks. Predict each event as it arrives, then hold only its public features until the block ends. Select at most one event for a feedback request at that boundary. Random admission samples uniformly; disagreement admission selects the largest fraction of disagreeing hypotheses. Both use the same seeded tie rule. Delays start at request time. If capacity is full, both discard the new request; never evict a pending record. A release makes capacity available only for the next boundary. These rules are identical across admission arms.
- Implement separated public-event and feedback-release iterators. Labels appear only after a prediction is committed. A scheduler cannot inspect the future delay. Emit stream_capacity_ready_score=1 after determinism, access isolation, disjoint windows and budget mutation tests. Include sample_size_budget and positive-control headroom diagnostics; a no-headroom slice remains in the report.

**Deliverable:** `results/experiment_7198_v634_feedback_capacity_stream.json`.

### Exp7199: Continuous self-learning by bounded constraint acquisition

Test FR-11 constraint addition through exact version-space elimination from released evidence. This replaces family credit that changed no decisions with a finite hypothesis set whose updates can change predictions. It is CPU self-learning, not LLM training.

- Build an opt-in finite version-space controller. Retain the hypotheses consistent with released feedback. Commit a constraint template only when it has distinct support and passes the past online-validation window. If feedback contradicts all active hypotheses, revoke the affected template and reopen that family's candidate set. Keep the superseded template and rollback hash. Model weights remain immutable. Predict by majority vote over surviving hypotheses, ties reject. An empty set abstains. Count every abstention as an error in the full prospective/recurrence denominator. A committed singleton supplies its exact predicate. Before singleton commitment require at least three distinct released support labels plus zero mistakes on eight subsequent requested-and-released validation labels for that family. Validation labels consume the same quota and pending capacity as all other labels. Never inspect the full evaluator validation sidecar. On an empty-set contradiction, revoke the template, reset that family's validation buffer and candidate set, then apply the newly released counterexample only if its frozen role is support. Previously observed labels remain archived but cannot validate the new epoch. Require fresh support after reset. Evaluator-known drift times cannot trigger a reset. If support is insufficient, keep the uncommitted vote and report no commit. Assign each event a support or validation role by a frozen public hash rule before any label is revealed. Balance roles within each four-event block. Only support labels eliminate hypotheses. Validation labels never fit the hypothesis set. Freeze a singleton candidate using support alone, then test it on eight distinct validation labels requested after the freeze; require zero mistakes. Earlier labels cannot be reused as its validation. All requests remain quota- and capacity-charged. A validation failure rejects that candidate and resets the epoch without using that validation label to fit its replacement. Thus the validation gate is not true by construction.
- Compare warmup-frozen constraints, FIFO feedback, uniform-random admission and disagreement-priority admission. Both adaptive admission arms use the identical controller, memory, pending capacity and label quota. Disagreement priority uses only current hypothesis predictions; use a frozen seeded tie rule. Also report the all-information oracle as an upper bound. Charge selection, storage, updates and lookups to the same end-to-end cost. Use Exp7198's four-event admission boundary and identical drop-new behavior when pending capacity is full. All arms receive the same public block; neither can select using labels or future delays. Charge the four-event staging buffer to the 64 KiB memory budget. FIFO admission selects the first public event in each four-event block. It uses the same version-space controller, support rule, prediction policy, query quota, drop-new capacity rule and memory limit as random and priority admission. Only the admission choice differs. The warmup-frozen arm stops updating after warmup.
- Run all ten stream seeds and all Exp7198 capacity/delay cells. Make predictions before requesting or releasing feedback. Estimate error and false-accept differences on the prospective and recurrence windows. Bootstrap by stream seed, not event or capacity cell. Record all comparisons; do not select a winning cell after seeing outcomes.
- The primary cell is C=4 with the frozen burst-delay schedule. acquisition_value_score=1 requires lower paired CI95 error than both frozen and FIFO controls, false-accept increase <=0, recurrence error increase <=0.02 and zero memory/capacity violations. Priority-specific benefit additionally requires lower error than random admission. Otherwise report a complete null. Measure update and lookup p50/p95. The hardware path is bitset intersections and bounded counters on CPU, then FPGA lookup/bitset logic. Target <1 ms lookup; no 100x hardware speedup is claimed without measurement. Set acquisition_run_complete_score=1 when every planned cell has terminal rows. Report version-space acquisition benefit separately from template-commit benefit. Majority prediction can already use a learned singleton; committing its identical predicate is persistence, not an additional accuracy gain. Do not claim that storage alone caused improved decisions.

**Deliverable:** `results/experiment_7199_v634_bounded_acquisition.json`.

### Exp7200: Cold feedback-causality and memory rollback audit

Exp7185 found shuffled family credit changed no decisions. The new acquisition policy must survive cold reload, causality and deletion tests before any claim of continuous improvement.

- Reconstruct Exp7199 from Exp7198's immutable stream in a fresh process. Match per-event predictions, budget counters and state hashes. The auditor can read truth only after reproducing the action. No producer comparison routine may supply the expected metric.
- On sealed controls, delay all feedback until after the last decision, shuffle admitted-feedback identities within matched capacity cells, delete learned templates, and replay from cold checkpoints. Future-label access and fake no-op updates must fail. If priority and random admission see identical evidence or decisions, report no scheduling benefit even when the common learner improves. Run two distinct deletion controls. Template-only deletion diagnoses whether commitment changes behavior. Whole-learning deletion resets version spaces, support records, validation buffers and committed templates to the common warmup checkpoint. Only the whole-state intervention can establish acquisition dependence if the singleton version space already controls predictions. Reapply neither archived future feedback nor pending released labels after reset. Verify role separation and candidate-freeze timestamps under both interventions.
- Run poison and recurrence windows without exposing their future outcomes to commit decisions. Roll back rejected changes and verify byte and decision parity with the prior checkpoint. Require zero unreleased-label reads, zero memory/capacity violations and exact metric recomputation. Promote memory only if the producer value gate and causal checks both pass. Set acquisition_audit_complete_score=1 when the audit finishes; external upstream absence is blocked, never partial.

**Deliverable:** `results/experiment_7200_v634_acquisition_cold_audit.json`.

## Phase 4: Production boundary, correction cost and synthesis

Reuse the validated sampler law. Change only its process boundary, then measure sample quality and end-to-end cost. Keep attached boards visible without repeating unchanged bring-up. Finish an ungated evidence matrix.

### Exp7201: Persistent PyO3 fixed-cardinality sampler prototype

Exp7189 established compiled parity but used a subprocess/JSON bridge for millisecond work. It claimed no PyO3 execution and missed the 10x target. Test whether a persistent in-process boundary removes measured deployment overhead.

- Profile setup, serialization, process launch, kernel and parsing separately on the shipped Exp7189 workload. Preserve raw timings and Python control. State whether the bridge-overhead hypothesis is supported; do not assume all cost is launch time.
- Expose the already shipped fixed_cardinality kernel through a persistent PyO3 object. Accept explicit proposal/uniform tapes for parity checks and independent RNG streams for distribution checks. Reuse input buffers over repeated calls. Make batching explicit and retain a batch-size-one path. Do not change the sampler law or remove the subprocess baseline.
- Test exact tape decisions, energy deltas <=1e-12, magnetization preservation, invalid shapes, buffer lifetime and cross-language state serialization. Exercise E2E-003 and E2E-004 with the compiled binding. A Python fallback cannot count as compiled execution. Emit pyo3_slice_ready_score=1 only after genuine compiled parity. A missing compiler/binding prerequisite yields blocked with its actual error. Do not introduce a new 10x gate or claim speed from this prototype.

**Deliverable:** `results/experiment_7201_v634_slice_pyo3.json`.

### Exp7202: Sampler boundary cost and sample-quality comparison

A faster kernel need not make an application faster. Compare persistent and subprocess boundaries under equal sampling work and wall budgets. Keep the NFR-01 target unchanged, even if the new path only meets a smaller local gate.

- Freeze n={32,64,128}, k={2,4}, batch={1,16,64}, ten independent seeds, and Python, subprocess Rust and persistent PyO3 arms. Use 160 proposals per chain for equal-work runs. Add a 50 ms equal-wall window with measured deadline overshoot. Alternate arm order and separate cold initialization from warm repeated calls. Charge data transfer and synchronization.
- Use exact enumeration at n=8 and n=12 with k=2 for the law check. For larger cells report effective samples per second, lag correlations and sector violations, not an exact-law claim. Include constant-chain and biased-transition controls so bogus ESS or wrong-target speed cannot pass. ESS uses a declared estimator and enough draws; otherwise record insufficient ESS evidence. Separately run quality panels with 1024 burn-in and 8192 retained proposals per chain across all ten seeds. Require at least 4096 retained draws and ESS >=100 for energy and one occupation observable per seed/cell before quality is sufficient. Compare mean energy within a prespecified 0.02 standardized tolerance and paired ESS-rate ratio CI lower bound >=0.90. If any required cell lacks these conditions, report sample_quality_sufficient=false and boundary_value_score=0; short 160-proposal timings alone cannot establish quality.
- The primary deployment cell is n=64,k=4,batch=1 under equal work. boundary_value_score=1 requires parity, zero sector violations, and a paired latency-speedup CI95 lower bound >1 over the subprocess bridge without worse sample quality. Report Python speedup and nfr_01_10x_met separately. Batch amortization cannot masquerade as single-query speed. Set slice_comparison_complete_score=1 on a fully measured null as well as a positive.

**Deliverable:** `results/experiment_7202_v634_slice_cost_quality.json`.

### Exp7203: Board continuity and quantized-correction cost envelope

Exp7188 proved CPU target correction. Exp7190 retained hardware visibility with topology unknown. Test the host correction cost relevant to future sampling hardware while keeping each attached board visible.

- Read the latest authentic KV260, GateMate and PolarFire receipts. Record each terminal criterion, date, source hash and unresolved prerequisite. Preserve KV260 graduation only when the cited receipt supports it. Preserve uncertainty about PolarFire dispatch. Historical SSH or board CPU execution is not programmable-logic sampling.
- Compare the latest operator-authored GateMate physical-state attestation against Exp6559. Without a newer receipt issue zero JTAG, reset, flash or power operations. Even if changed, record the authorized next action without performing it in this host study. Do not repeat an unchanged hardware bring-up. Any future KV260 check uses SSH reachability, never host storage discovery.
- Reuse Exp7188's corrected transition law on fixed n={16,32,64} graphs, ten seeds and 4/8/16-bit coupling quantization. Compare exact float and delayed-acceptance correction at matched proposals and matched wall budgets. Count cheap-stage rejects, full-energy calls, host latency and acceptance. Preserve rejected states in the sampling chain. Recompute the small-law check before making a cost claim. Report every precision/seed row, not only the fastest cell. Freeze ten seeds 7203001..7203010 and beta=1. Reuse make_frustrated_instance(n, seed) from Exp7187, including its nonzero fields and frustrated triangle. Record actual edge count and degree; do not silently sparsify it. Require at least one changed quantized coefficient in each precision condition before interpreting a distortion result. Exact-grid conditions remain explicit negative controls. Use k=2, 1024 proposals per chain for equal-work runs and 100 ms per chain for equal-wall runs. Cap all host measurements at 900 seconds. Include exact-law n=8 checks separately. The break-even envelope is per proposed transition; it cannot establish equal effective-sample throughput or mixing speed.
- Compute a break-even envelope from measured host costs and explicitly hypothetical device/transfer latency axes. Mark unavailable device timing unknown. Keep topology_unknown without a mapping to the published parent graph. Degree<=16 is only necessary. Do not claim TSU, FPGA or soft-spin execution or power savings. Set hardware_envelope_complete_score=1 when all board dispositions and the CPU envelope finish. Per-board blocks do not erase completed host evidence.

**Deliverable:** `results/experiment_7203_v634_hardware_correction.json`.

### Exp7204: V634 independent evidence matrix and branch decisions

Read all thirteen planned tasks, including external blocks and complete nulls. Preserve the PRD boundaries: source semantics, useful continual learning, real live generalization and measured production deployment. The capstone has no upstream success gate.

- Load the frozen V634 contract captured by Exp7192 if available. Otherwise use the staged or activated file with matching milestone identity. Enumerate exactly exp7192 through exp7204, marking this row self. Read each declared deliverable first. Resolve canonical conductor gate-block artifacts by full task ID if needed. Record actual missing paths rather than inventing alternate outputs.
- Build an evidence matrix with verdict class, free verdict, substrate, raw-row counts, hashes, authentication flags, acceptance gates and limits. Recompute every proposed numeric claim from per-unit rows. Quarantined artifacts are visible but excluded from promoted evidence. Keep measurement completion, method value, execution-grounded circular gains and PRD completion separate.
- Issue continue, retire or needs_changed_prerequisite for each mechanism. A repeated failure needs its exact prior ID and verdict plus the stated retirement signal. Do not infer identical verdicts from different text or retire unrelated branches. Do not mutate the exclusion manifest or protected verifiers. New source/claim extraction and memory experiments are pilots, not general verification or foundation-model claims.
- Run scripts/publication_gate.py --json for the unchanged G1-G4 publication gate. Publication remains operator-only. Reconcile planned versus completed specs, _bmad/traceability.md, ops/status.md and ops/changelog.md additively. capstone_complete_score=1 means a complete matrix. If external absence prevents a scientific conclusion, verdict_class=blocked with gate_check_summary, never retryable partial.

**Deliverable:** `results/experiment_7204_v634_capstone.json`.

## Exact Task Contract

| Order | Task ID | Exact title | Deliverable | Structured gate |
|---:|---|---|---|---|
| 1 | `exp7192-source-contract` | V634 source ingestion and exact execution contract | `results/experiment_7192_v634_source_contract.json` | none |
| 2 | `exp7193-arc-direct-tool` | Live ARC direct-tool generalization measurement | `results/experiment_7193_v634_arc_direct_tool.json` | none |
| 3 | `exp7194-arc-gap-audit` | ARC tool-gap and banked-progress causal audit | `results/experiment_7194_v634_arc_gap_audit.json` | `exp7193-arc-direct-tool.arc_tool_measurement_complete_score == 1` |
| 4 | `exp7195-typed-grounding` | Typed source execution and abstention prototype | `results/experiment_7195_v634_typed_grounding.json` | none |
| 5 | `exp7196-qwen-atomic-capture` | Qwen3.8 independent source and claim capture | `results/experiment_7196_v634_qwen_atomic_capture.json` | `exp7195-typed-grounding.typed_executor_ready_score == 1` |
| 6 | `exp7197-grounding-value-audit` | Typed grounding value and independent semantic audit | `results/experiment_7197_v634_grounding_value_audit.json` | `exp7195-typed-grounding.typed_executor_ready_score == 1` AND `exp7196-qwen-atomic-capture.atomic_capture_complete_score == 1` |
| 7 | `exp7198-feedback-capacity-stream` | Sealed constraint stream with bounded pending feedback | `results/experiment_7198_v634_feedback_capacity_stream.json` | none |
| 8 | `exp7199-bounded-acquisition` | Continuous self-learning by bounded constraint acquisition | `results/experiment_7199_v634_bounded_acquisition.json` | `exp7198-feedback-capacity-stream.stream_capacity_ready_score == 1` |
| 9 | `exp7200-acquisition-cold-audit` | Cold feedback-causality and memory rollback audit | `results/experiment_7200_v634_acquisition_cold_audit.json` | `exp7199-bounded-acquisition.acquisition_run_complete_score == 1` |
| 10 | `exp7201-slice-pyo3` | Persistent PyO3 fixed-cardinality sampler prototype | `results/experiment_7201_v634_slice_pyo3.json` | none |
| 11 | `exp7202-slice-cost-quality` | Sampler boundary cost and sample-quality comparison | `results/experiment_7202_v634_slice_cost_quality.json` | `exp7201-slice-pyo3.pyo3_slice_ready_score == 1` |
| 12 | `exp7203-hardware-correction` | Board continuity and quantized-correction cost envelope | `results/experiment_7203_v634_hardware_correction.json` | none |
| 13 | `exp7204-capstone` | V634 independent evidence matrix and branch decisions | `results/experiment_7204_v634_capstone.json` | none |

## Dependency Graph

```text
7192 (advisory source/contract audit; gates none)
7193 -> 7194
7195 -> 7196 -> 7197
  +--------------^
7198 -> 7199 -> 7200
7201 -> 7202
7203 (independent host correction/board continuity)
7204 (ungated matrix of every task, including missing or blocked inputs)
```

Only real data/readiness dependencies have structured gates. The upstream task
spells each gate field identically in REQUIRED ARTIFACT FIELDS. Readiness gates
never require a positive scientific outcome. The longest dependent branch has
three tasks. CPU branches do not depend on the model or ARC run.

## Model and Substrate Contract

Only Exp7193 and Exp7196 need an LLM. Each declares
`MODEL_SPECS` with `unsloth/Qwen3.8-27B-GGUF`, Q4_K_M, through its local
GGUF path. Both use real generation, CUDA, `CARNOT_FORCE_LIVE=1` and
`inference_mode=live_gpu`. The runtime records resolved revision, file hash,
embedded tokenizer, chat template, lease, process ownership and model count.
A cache miss or resource conflict produces a diagnosed block.

| Actual work | Substrate class | Duration floor |
|---|---|---:|
| Full multi-row generation, including the planned ARC sessions | `model_full_generation` | 60 seconds |
| Eight-token canary or a run that stops after that canary | `model_bounded_generation` | 10 seconds |
| Loading or embedding extraction without generation | `model_load_no_generation` | 2 seconds |
| CPU execution of constraint or sampler algorithms | `cpu_exact_solver_or_simulator` | Existing CPU policy |
| Read-only evidence analysis | `aggregation` | Existing aggregation policy |
| A prerequisite prevents all qualifying work | `blocked_no_run` | No invented execution |

No new load-only experiment is planned. The distinctions above also apply to
partial runs. Never pad runtime. CPU tasks declare no model invocation.
Qwen3.5-0.8B and gemma-4-E4B-it remain CPU smoke models only; no headline
comparison uses them. The old four-model list is superseded.

## Hardware Requirements and Budgets

| Tasks | Required resources | Bounded workload | Acquisition consequence |
|---|---|---|---|
| 7193, 7196 | One task-owned RTX 3090 allocation from the existing two-card host; cached ~17.1 GB Qwen GGUF | One model instance; ARC session cap 3600 seconds, extraction cap 2400 seconds; retain setup/test reserve inside 4800 seconds | No purchase. Record free VRAM and the actual context allocation before load. |
| 7195, 7197 | CPU, Python, exact relation executor | 192-row panel; 32 independent held-out base cases | No new hardware. |
| 7198-7200 | CPU/system memory | Ten 1024-event streams, 64 KiB learner state, fixed queue capacities and delays | Bitsets and counters support later FPGA learning; no device speed claim. |
| 7201-7202 | Existing Rust/PyO3 toolchain and CPU | Tiny exact-law checks plus fixed size/cardinality/batch sweep | No FPGA prerequisite for language-boundary value. |
| 7203 | CPU and existing board receipts | Fixed-graph quantization/correction sweep; no board commands | KV260 preserved, GateMate awaiting physical-state change, PolarFire evidence checked separately. |
| 7192, 7194, 7204 | CPU/filesystem and bounded source access | Parser/gate checks and evidence recomputation | No GPU or new service dependency. |

Per-task estimates in YAML total 455 minutes, including implementation and
validation. They are budgets, not a forecast of measured compute. Each task has
its own fixed bounds and progress heartbeat. Existing GPUs need no multi-model
runner because each live task uses one model. Record runner selection rather
than infer a missing DualGPURunner from a GPU snapshot.

Extropic Z1, Alveo/Agilex, XDNA and additional GPUs are wishlist paths, not assumed
resources. No credentials, hardware purchase, board reset or external submission
is part of this milestone. A changed GateMate attestation permits planning the
next action; this milestone still performs no physical operation.

## Acceptance, Failure and Retirement Discipline

Every task produces rows, sample counts, measured duration, source hashes,
`inference_substrate`, `inference_substrate_class`, `verifier_is_oracle`,
`honest_verdict` and the closed `verdict_class` enum. Every required field has
its own principle annotation. Gate fields mean exactly what their producer says.

Null results finish. Scientific failure is not an unavailable prerequisite.
External absence uses `blocked` with `gate_check_summary`. Only incomplete work
inside the task may use retryable `partial`. Checkpoints never occupy terminal
result paths. A zero-call ARC run cannot establish model/tool efficacy.

Prior-failure blocks name the old experiment, exact verdict, changed mechanism
and `retire_if_same_verdict: true`. No task requires a retired upstream ID.
Versioned scientific continuations change the mechanism rather than only the
name. Hardware continuity uses the standing 2026-05-29 authorization for a
read-only disposition. The original grounding, family-credit and 10x findings
remain unchanged. A local boundary speedup does not redefine NFR-01.

The queued 2026-09-10 direct tool-use pass is covered by Exp7193/7194. The
cancelled supervisor-arm run stays cancelled. No active scope-reduction directive
is bypassed: retired repair, generated-text scoring, public-game re-solves and
unavailable hardware execution are excluded. The capstone records compliance.

## Verification and Documentation Handoff

Before handoff, validate the full staged YAML with the repository schema,
prior-failure and exclusion linters, real Markdown/YAML contract parsers,
invented-path scanner, ARC floor and gate checker. Exercise passing and failing
producer fields through the real conductor gate evaluator in `/tmp` fixtures.
Check every existing source path and every same-milestone dependency.

Implementation tasks must extend their existing capability specs with named
REQ-* and SCENARIO-* entries, write meaningful failing tests, implement, then
run focused lint/type/spec checks. Bindings use E2E-003/004. Memory uses the
applicable E2E-007 invariants. ARC uses E2E-009/010 plus its separately measured
live comparison. Source grounding runs public input through real extraction,
execution and independent adjudication. Planning-only work needs no model run.
Do not run the full repository test suite as an experiment precondition.

Every prompt requires flushed phase lines, before/after lines around long calls,
and a heartbeat inside long loops at least every 60 seconds. A blocking native
call needs an external heartbeat and its own deadline. Keep all output gaps
below 600 seconds to preserve the available 4800-second cap.

Keep `openspec/`, `_bmad/traceability.md`, `ops/status.md` and `ops/changelog.md`
aligned. Planning records remain explicitly planned until experiments execute.
This plan does not modify the active YAML or `scripts/research_conductor.py`.
Do not push. Publication requires the operator's separate action.
