# SemIf EBM and ARC experiment plan

Date: 2026-09-20

Status: planning only. No implementation or scored-path change is part of this note.

## 1. Decision

Test SemIf-style option-logit readout in two tracks. Track A tests it as a calibrated feature and as one term in a product-of-experts energy. Track B tests it at typed decision seams in the ARC live agent. The reason to test it is narrow. It can turn a declared option set into one probability distribution without generated text or JSON parsing. Two corrections are load-bearing. First, on JevBench v1.2, SemIf scores 74.7 and ranks second behind the closed Jev at 75.4. It is the top open system, not number one. The reported 0.7-point gap has no confidence interval that we saw (`/home/ianblenke/jevbench-src/RESULTS-v1.2.md:17-18`, `:39`). JevBench is Benchmark Heaven's own benchmark (`/home/ianblenke/jevbench-src/RESULTS.md:19`). Second, JevBench measures text decisions. Its families include routing, intent, policy/rules, scoring, extraction, and answer judging. Its topics include finance (`/home/ianblenke/jevbench-src/README.md:168-175`; `/home/ianblenke/jevbench-src/scripts/v1.2/topics.py:20-23`). It says nothing about ARC grid-state reasoning. Transfer to ARC is a hypothesis to test, not a finding.

### Decision in one paragraph

Test Needle 3 paired with a Jev-style decision engine, but give each component a narrow role. Needle is initially a grammar-constrained call emitter and a calibrated routing signal, not the mechanic reasoner. SemIf is the first option selector because it needs no task training. A NanoJev-style choice head is a later test-time-training arm that must start fresh for each unseen game and learn only from that game's already-observed trajectory. Qwen3.8-27B remains the expensive world-model generator, and the existing E3 cascade remains the authoritative fallback and verifier-routed controller. The claim that this pairing could "greatly accelerate the solve rate" is an untested hope, not a finding. The public ARC set is already fully solved as a development proxy at 183 reproducible levels across 25 games (`ops/arc_solve_registry.yaml:6332-6333`), while hidden-game performance is unknown. The measurable target is therefore adapter-free levels gained under fixed action, token, and wall-clock budgets, plus actions to progress, decision latency, and tokens per level. No numeric expected speedup is justified until E6 measures what fraction of current live-loop cost is actually replaceable.

## 2. What a readout decision is in Carnot terms

SemIf receives state text, a question or criterion, and 2 to 16 declared options. Each option is assigned a single-token label. The model evaluates the prompt once. SemIf reads the final-position logits for those labels and applies a softmax. It does not generate an answer token. The implementation checks the single-token labels, evaluates one prompt, selects the option logits, and normalizes them in [`direct.py`](https://github.com/TheoLeeCJ/SemIf/blob/master/src/semif_phase1/direct.py#L11-L72). Its shared-state path prefills one state and branches the cache across criteria in [`shared.py`](https://github.com/TheoLeeCJ/SemIf/blob/master/src/semif_phase1/shared.py#L49-L141).

For option `o` and state `s`, let the selected logit be `z_o`. Then:

```text
p(o | s, declared options) = exp(z_o) / sum_j exp(z_j)
E_readout(o | s) = -log p(o | s) = -z_o + log sum_j exp(z_j)
```

For ranking the declared options, `-log p` and `-logit` differ only by the same state-dependent constant. The native option distribution is therefore already an energy over the declared options. It is conditional on that option set. It is not a globally normalized belief about every possible action.

This is compatible with Carnot's energy interface. `GibbsModel.energy` is a learned scalar MLP energy (`python/carnot/models/gibbs/__init__.py:170-204`, `:267-302`). `IsingModel.energy` supplies a quadratic energy (`python/carnot/models/ising/__init__.py:115-127`, `:210-238`). `CompositionalEnergyMinimizer.compute_total_energy` already sums sub-energies (`python/carnot/models/compositional_energy.py:5-19`). A product of experts can therefore use:

```text
E_joint(o | s) = alpha * E_readout(o | s) + beta * E_verifier(o, s)
p_joint(o | s) proportional to exp(-E_joint(o | s))
```

`alpha` and `beta` must be non-negative and fit only on training folds. Raw scores from different sources must not be added before calibration. Carnot already has a diagnostic path that converts mean token log-probability to a semantic energy and adds it to a verifier energy. That path keeps deterministic validators as final authority (`python/carnot/verify/arm_ebm_logprob_telemetry_repair.py:122-193`, `:196-250`).

A readout is not an energy-based verifier in Carnot's stronger sense.

- It does not execute an induced world model.
- It does not check a constraint independently.
- It does not inspect a hidden game rule.
- It only reports the same model's relative preference among supplied labels.
- If the model generated a hypothesis and then scores that hypothesis, proposal and score share one model and one prompt family.

This creates a circularity risk. A readout may repeat the generator's error with high confidence. It must be recorded as `verifier_is_oracle=false`. It must not replace an oracle-distinct checker. On ARC world models, `WorldModelVerifier` executes the candidate against observed transitions and reports exact, cell, and change agreement (`python/carnot/agentic/arc_executable_world_model.py:1849-1933`, `:2043-2139`). That checker remains authoritative. This follows the repository's open oracle-distinct gap: detector AUROC can be high without proving a selector beats vote (`ops/verifier_gaps.md:1005-1010`). The later ARC set-encoder result only closed its measured selection slice with a non-oracle selector and a positive interval (`ops/verifier_gaps.md:1115-1130`).

The SemIf performance facts remain self-reported. Its [README](https://github.com/TheoLeeCJ/SemIf/blob/master/README.md) reports a shared-prefix fixture improving from 2.33 to 20.03 decisions per second, with 5 or 6 argmax flips among 777 decisions under BF16 cache reuse. It also says the returned probabilities are conditional on the options and need target-workload calibration. The JevBench harness reports a SemIf rebuild falling from 72% to 21% after option reversal (`/home/ianblenke/jevbench-src/RESULTS-v1.2.md:90-91`). These are reasons for experiments, not deployment evidence.

## 3. Track A: energy-based-model integration

### Existing substrate and the headroom rule

The calibrated-decision benchmark has 6,548 labeled reasoning-step rows. It derives two PCIB features, splits by question ID, trains a fixed 2-4-1 Gibbs model, and scores held-out AUROC and Brier (`python/carnot/autoresearch/calibrated_decision_benchmark.py:18-46`, `:126-187`, `:251-287`). It rejects an all-equal scorer as degenerate (`python/carnot/autoresearch/calibrated_decision_benchmark.py:62-77`, `:278-287`). The repository also has scalar temperature fitting and ECE (`python/carnot/training/platt_scaler.py:1-17`, `:70-124`, `:143-196`), isotonic group calibration (`python/carnot/verify/group_conditional_calibration.py:7-47`), and risk-coverage fields with five seeds, at least 1,000 examples, and bootstrap intervals (`python/carnot/pipeline/risk_coverage_abstention_3718.py:29-40`, `:55-107`).

The corpus is not valid for every claim. The registered FoVer candidate pool has zero selection headroom even though detector AUROC is high (`ops/verifier_registry.yaml:13-35`). The static decision experiment also found only 22 conflict cells among 3,950 rows and near-universal accept behavior (`ops/verifier_gaps.md:4836-4860`). Therefore:

- Use the 6,548 rows for calibration and detection claims.
- Do not use that saturated pool alone for a selection-lift claim.
- Compute `oracle_at_k - incumbent_at_1` before each selection test.
- If headroom is zero or no decisions flip, report `FALSE_NEGATIVE_RISK`.
- Use a headroom-positive pool for selection. The existing ARC pool has measured wrong-majority rows and later produced a non-oracle positive result at `n=52` (`ops/verifier_gaps.md:1075-1091`, `:1115-1130`). It is a candidate, not an automatic substitute.

### A1. Readout energy and product of experts

**Hypothesis.** A calibrated option-readout energy contains source-grounded information that is not present in the two static PCIB features. A non-negative product of experts will improve held-out Brier score over both the readout-only and verifier-only models. On a headroom-positive candidate pool, it will also improve top-one selection.

**Inputs and future code touch set.** Use `data/fover_corpus_v4.json`. Preserve the question-ID grouping used by `calibrated_decision_benchmark._split_features` (`python/carnot/autoresearch/calibrated_decision_benchmark.py:159-174`). Build one prompt from the row's available source text. Declare `accept`, `reject`, and `escalate`. Read all three logits in one pass. Join `-log p` to the existing PCIB features and Gibbs energy. Add a future experiment driver at `scripts/experiments/semif_readout_ebm_eval.py`, reusable logic under `python/carnot/verify/semif_readout_energy.py`, and focused tests under `tests/python/test_semif_readout_energy.py`. Do not change the standing benchmark until the result passes.

**Metric.** Primary: out-of-fold Brier delta of PoE minus the better single expert. Secondary: log loss, ECE with fixed bins, AUROC, AURC, coverage at 5% selective error, and top-one lift only where headroom is positive. Use paired group bootstrap intervals by question ID. Report the fitted `alpha` and `beta` for every fold.

**Sample size and reason.** Use all 6,548 rows and at least 1,000 held-out predictions. This supports sub-percentage calibration claims under the repository floor. Keep at least 30 independent question groups in every reported fold. If the headroom-positive selection set has fewer than 30 independent tasks, call it exploratory and do not promote it.

**Positive control.** In a separate synthetic lane, add a noisy copy of the independent gold label as a readout feature. The PoE must detect its added signal and improve Brier. Never mix this lane with the real fit.

**Degenerate-case check.** A uniform option distribution must reduce to the verifier-only ranking up to a constant. An all-equal Gibbs output must be rejected. A missing or non-finite option logit must force `escalate`, not `accept`.

**Pass/fail gate.** Pass only if the 95% paired interval for PoE Brier delta is below zero against both single experts, ECE does not worsen by more than 0.01, the positive control passes, and any selection claim has measured headroom with a positive interval. Otherwise fail.

**Kill criterion.** Drop readout-as-an-EBM-term if the positive control passes but `alpha` collapses to zero in every fold, readout-only discrimination does not beat chance, and the PoE has no Brier gain on both the main and headroom-positive sets. Keep an ARC seam test separate only if Track B still has direct evidence.

**Cost.** One model forward per row: 6,548 readouts. Fitting and bootstrap work are CPU-only. Cache the readout logits so A2 and A3 add no model calls.

### A2. Target-workload calibrator tournament

**Hypothesis.** Raw option softmax values are miscalibrated, and one of scalar temperature scaling, true two-parameter Platt scaling, or isotonic regression will reduce Brier and ECE out of fold. Do not call the existing scalar-temperature class full Platt scaling. It fits one temperature (`python/carnot/training/platt_scaler.py:70-124`).

**Inputs and future code touch set.** Use the cached A1 logits and labels. Run a separate calibration for each workload and decision seam. Do not pool JevBench, FoVer, and ARC. Add the comparison to `python/carnot/verify/semif_readout_energy.py` and `scripts/experiments/semif_readout_ebm_eval.py`. Reuse `group_conditional_calibration.py` only for its isotonic pattern (`python/carnot/verify/group_conditional_calibration.py:35-96`). Fit every calibrator inside each training fold.

**Metric.** Brier is primary. Report ECE, log loss, maximum calibration error, reliability tables, and 95% group-bootstrap intervals. Report calibration slope and intercept for the Platt arm. Report performance by option count and source family.

**Sample size and reason.** Require at least 1,000 out-of-fold predictions and at least 30 independent groups per workload. Isotonic needs at least 20 positive and 20 negative examples in each training fold. If that floor fails, omit isotonic rather than silently pooling test data.

**Positive control.** Raise known synthetic probabilities to a fixed temperature. Scalar temperature scaling must recover a lower Brier score. Apply a known affine distortion to logits. Two-parameter Platt scaling must recover it better than temperature-only scaling.

**Degenerate-case check.** Constant probabilities must remain constant. Isotonic must not create a false rank improvement. Empty bins, one-class folds, and probabilities outside `[0,1]` are hard failures.

**Pass/fail gate.** Select a calibrator only when its Brier interval is below raw softmax and the upper bound of its ECE interval is below 0.05. If two methods tie, choose temperature scaling because it has fewer parameters. Otherwise fail and keep probabilities diagnostic-only.

**Kill criterion.** Drop probability-valued use on a workload if no method clears ECE 0.05, Brier is no better than prevalence, or fold-to-fold calibration reverses. Ranking-only use may continue if A1 discrimination passed.

**Cost.** No new model forward. Three calibrator fits per fold, plus grouped bootstrap. This uses cached A1 logits.

### A3. Calibrated accept, reject, or escalate policy

**Hypothesis.** A calibrated readout or PoE can route easy cases to `accept` or `reject` and uncertain cases to `escalate` while reducing selective error at useful coverage.

**Inputs and future code touch set.** Use the out-of-fold A1 and A2 probabilities on the calibrated-decision benchmark. Use the existing decision grammar `accept`, `reject`, `abstain` as a structural reference (`python/carnot/verify/abstention_calibrated_clean_verifier_v15.py:58-69`). Rename `abstain` to `escalate` in the new experiment artifact. Add policy evaluation to `scripts/experiments/semif_readout_ebm_eval.py`. Do not alter `calibrated_decision_benchmark.py` in the first experiment.

**Metric.** Primary: AURC and coverage at 5% selective error. Also report risk at 50%, 80%, and 90% coverage, Brier, ECE, the full confusion matrix, and counts for all three actions. Evaluate a pre-registered grid of escalation costs instead of choosing one cost after seeing results.

**Sample size and reason.** Use all out-of-fold rows, five fixed seeds, at least 1,000 examples, and at least 30 independent question groups. This matches the existing abstention measurement floor (`python/carnot/pipeline/risk_coverage_abstention_3718.py:29-40`, `:89-107`).

**Positive control.** An independent noisy-gold score must produce a better risk-coverage curve than entropy and prevalence controls.

**Degenerate-case check.** Report `always_accept`, `always_reject`, and `always_escalate`. A policy with no examples in one action is not a three-way policy. The prior decision artifact produced 6,613 accepts, 2 escalations, and no rejects for one Gibbs arm (`results/experiment_7385_v648_decision_training.json:128-142`). This shape must fail the balance check unless the registered cost matrix proves it optimal.

**Pass/fail gate.** Pass if the AURC improvement over both entropy and verifier-only controls has a 95% interval below zero, coverage at 5% risk is at least 25%, all three actions occur, and no protected group exceeds the registered risk bound. Otherwise fail.

**Kill criterion.** Drop the policy use if every calibrated arm collapses to one action, if useful coverage stays below 25%, or if escalation saves no errors after its measured cost is charged.

**Cost.** No new model forward. It is a threshold sweep and grouped bootstrap over cached out-of-fold rows.

### A4. Option-order and prompt-template adversarial probe

**Hypothesis.** A usable readout is stable under meaning-preserving option permutations and prompt templates after labels are mapped back to semantic option IDs.

**Inputs and future code touch set.** Use 1,000 stratified rows from A1. For each row, test original order, reverse order, and four seeded permutations. Test three templates: terse, rubric-first, and state-first. Keep option wording unchanged. Add the probe to `scripts/experiments/semif_readout_ebm_eval.py` and its fixtures to `tests/python/test_semif_readout_energy.py`. E1 already contains an original-versus-reversed option probe in `/home/ianblenke/carnot-wt-jev/scripts/jevbench_readout_eval.py:459-475`.

**Metric.** Semantic-option total variation distance, argmax flip rate, accuracy delta, Brier delta, ECE delta, and worst-row examples. Bootstrap by source group. Report each template and order separately.

**Sample size and reason.** Use at least 1,000 decisions. This supports sub-percentage flip-rate claims. Cover every observed option-count stratum. Each row receives six orderings under three templates, for 18 mapped distributions.

**Positive control.** A deliberately position-biased mock scorer must fail the test. A scorer keyed by stable semantic option ID must pass.

**Degenerate-case check.** Uniform probabilities are invariant but useless. They must fail the accuracy and Brier gates. Duplicate labels, multi-token labels, and a missing option probability are hard errors.

**Pass/fail gate.** Pass if the upper 95% bound on argmax flips is below 1%, median total variation is at most 0.02, worst-template accuracy loss is at most one point, and A2's calibration gate still passes. These bounds are fixed before inference. Otherwise fail.

**Kill criterion.** Drop the method if any meaning-preserving reversal causes a large collapse like the reported 72% to 21% case, or if no stable template exists across folds. Do not repair it by choosing one order on the evaluation labels.

**Cost.** 18,000 one-pass readouts for 1,000 rows. Shared-prefix reuse may reduce latency within a row. It does not reduce the counted decisions. The speed claim must be measured on our model and hardware.

## 4. Track B: the ARC live agent

### Live path and decision seams

The scored cascade is real and reachable. `make_carnot_agent` constructs the policy and raises the submitted action bound to 2,000 (`python/carnot/agentic/arc_competition_agent.py:9611-9650`). `E3AgentPolicy` implements explore, induce, verify, plan, and execute (`python/carnot/agentic/arc_competition_agent.py:5182-5193`). Its action choke point calls the routed loop (`python/carnot/agentic/arc_competition_agent.py:6846-6883`, `:7304-7315`). The adapter-free offline entrypoint constructs `E3AgentPolicy` under `--mechanism e3` (`scripts/arc_loop_solve.py:530-539`). The leaderboard entrypoint also constructs `E3AgentPolicy` and records `solve_provenance=live_agent_self_discovery` (`scripts/arc_leaderboard_eval.py:116-129`, `:1217-1234`).

`arc_live_ttt` is supporting evaluation code, not a separate scored policy. It builds a live test-time model from observed transitions, applies the same verifier and planner pattern, and declares the verifier non-oracle (`python/carnot/agentic/arc_live_ttt.py:1-43`, `:257-430`). Its transition outcomes can feed E2 only when their game and episode provenance are intact.

Use leave-one-game-out calibration. Hold out every row from the target game. Use no per-game adapters. Keep `solve_provenance=live_agent_self_discovery`. Public offline games remain a development proxy for hidden games. The final metric is the live level counter and actions to progress. REQ-5720 names `levels_gained` and `actions_to_first_solve` as ground truth and says holdout induction accuracy is indirect (`openspec/capabilities/arc-human-replay-frame-change/spec.md:9208-9228`). Do not use its adapter-based hand-verifier distance in this work.

The current live artifacts already record the selected action, branch, phase, induction result, verifier fields, level before and after, frame change, and transition count (`python/carnot/agentic/arc_competition_agent.py:7174-7246`). They do not record the full candidate option set. The belief fields record whether ranking changed and the final action, not all candidate rows (`python/carnot/agentic/arc_competition_agent.py:7259-7277`). E2 must therefore add shadow telemetry before B1 can be trained. Existing games need not be replayed to label decision points already present, but missing option sets cannot be reconstructed honestly.

| Decision seam | Declared options | State text | Existing outcome label | Live reachability |
|---|---|---|---|---|
| Candidate-action selection | The incumbent top 15 action candidates plus `delegate_to_incumbent_tail`. Rich candidates include keyboard actions and object clicks. The generator can admit up to 48 click targets (`python/carnot/agentic/arc_graph_explore.py:189-252`). | Logical grid summary; level; recent transition summaries; path depth; stall state; each action ID, coordinates, and incumbent features. Never include game source. | `action`, `frame_changed_since_last_action`, `level_before`, `level_after`, phase, and later actions to the next level-up from provenance (`python/carnot/agentic/arc_competition_agent.py:7174-7246`). | `make_carnot_agent -> E3AgentPolicy.next_move -> _next_move_routed -> StepwiseExplorer._candidates`. Candidate construction and ranker order are in `python/carnot/agentic/arc_competition_agent.py:2808-2832`, `:2982-3032`. |
| Induction timing | `continue_explore`, `induce_now`, `reinduce_now`, `delegate_to_current_gate`. | Level, transition count, explored-out flag, stall state, prior induction count, new transitions since induction, prior skip reason, and last verifier outcome. | `induction_ran_this_action`, `induction_reason`, `induction_skipped`, `induction_planned`, trust and verify fields, then level progress (`python/carnot/agentic/arc_competition_agent.py:7219-7243`). | `_next_move_routed` computes stall, calls `_should_enter_induction`, and runs the timed induction seam (`python/carnot/agentic/arc_competition_agent.py:7464-7505`). A readout can gate the expensive generation. |
| World-model hypothesis gate | For one candidate: `accept_candidate`, `reject_and_explore`, `escalate_for_alternative`. For a bounded multi-candidate run: candidate IDs plus reject and escalate, capped at 16. | Observed transition summaries; candidate source; `exact_accuracy`, `cell_recall`, `change_accuracy`; trust energy; off-path checks; and plan termination status. Do not include hidden source or future frames. | Existing attempts record trust energy, held-out and verify accuracy, cell recall, binary gate, skip reason, whether a plan was found, and later actions to progress (`python/carnot/agentic/arc_competition_agent.py:7207-7237`). `WorldModelVerifier` supplies the independent execution checks (`python/carnot/agentic/arc_executable_world_model.py:1849-1933`, `:2043-2139`). | The ordinary gate creates `WorldModelVerifier`, checks accuracy and change, then either skips or plans (`python/carnot/agentic/arc_competition_agent.py:8693-8759`, `:8822-8834`). The hidden-state selection path also records trust and gate outcomes (`python/carnot/agentic/arc_competition_agent.py:8603-8689`). |
| Supervisor arm selection | `no_redirect` plus the eligible members of `drop_goal_bias`, `allow_reinduction`, `force_exploration_diversity`, and the default-off `tool_loop_reinduction`. | The existing `TrajectorySnapshot`: level, goal-bias status, induced latch, induction attempts, new transitions, and diversity-active state (`python/carnot/agentic/arc_competition_agent.py:5979-6002`). | `resolved_by_levelup`, `actions_to_levelup`, per-arm `fired` and `helped`, split credit, and unredirected stagnation windows (`python/carnot/agentic/arc_trajectory_supervisor.py:308-349`; `openspec/capabilities/arc-world-model-trust-energy/spec.md:27986-28015`). | `_maybe_supervise_trajectory` is called once per routed action and applies through existing seams (`python/carnot/agentic/arc_competition_agent.py:5961-6029`). The current selector is a fixed order over the curated arm set (`python/carnot/agentic/arc_trajectory_supervisor.py:36-64`, `:263-306`). This is Generalization-Testing Floor activity 4: selection, not generation. |

Candidate-action and supervisor choices are not generated today. A readout would replace or gate an algorithmic choice there. It can save generation only at induction timing or when it prevents an unhelpful reinduction. Its quality benefit must still clear actions-to-progress.

### Scored serving blocker and budget

The local Python path can read logits. `LlamaCppCompletionLogprobProvider` constructs `Llama(..., logits_all=True)` and requests completion log-probabilities (`python/carnot/verify/beaver_lite_live.py:140-209`). Another provider evaluates tokens and reads `self._llama.scores[-1]` directly (`python/carnot/verify/beaver_lite.py:198-205`, `:275-289`). This proves a local library route exists. It does not prove parity with the standing llama.cpp server route.

**BLOCKER: the current scored vLLM wrapper cannot return an option distribution to the agent.** `_vllm_raw_completion` sends only model, prompt, maximum tokens, temperature, optional seed, and stop. It then discards everything except text, finish reason, and completion-token usage (`python/carnot/agentic/arc_executable_world_model.py:7362-7410`). The ordinary generation payload also has no log-probability field (`python/carnot/agentic/arc_executable_world_model.py:8054-8059`, `:8340-8368`). The server launch has no `--max-logprobs` setting (`python/carnot/agentic/arc_executable_world_model.py:7216-7272`).

Current upstream vLLM documents `logprobs`, `prompt_logprobs`, `allowed_token_ids`, and `logprob_token_ids` on [completion requests](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/protocol.html). Its server also exposes a [`--max-logprobs` setting](https://docs.vllm.ai/en/latest/cli/serve.html#--max-logprobs). The exact wheel attached to the offline kernel is not pinned in this repository. The kernel finds any `vllm-*.whl` and installs it offline (`scripts/kaggle/submission_kernel/main.py:295-311`). E0 must inspect that wheel's actual help and request schema.

If the attached version supports the fields, the minimum future change is a dedicated readout request beside `_vllm_raw_completion`:

1. Use `/tokenize` to prove each option label is one token. The kernel already calls that endpoint (`scripts/kaggle/submission_kernel/main.py:813-825`).
2. Send `max_tokens: 1`, `logprobs: N`, and the option token IDs. Prefer `logprob_token_ids` for exact declared tokens. Use `allowed_token_ids` only if the installed version returns the unwarped scores needed for comparison.
3. Preserve `choices[0].logprobs` in the returned structure.
4. Add `--max-logprobs 16` to `_ensure_vllm_server` if the installed server requires an explicit cap. Sixteen matches SemIf's declared-option limit. Do not silently truncate a larger ARC action set. Use the top-15-plus-tail design instead.

If the attached wheel lacks these request fields, scored-path readout is blocked until the wheel closure is rebuilt and attached. `prompt_logprobs` alone is not a substitute unless E0 proves an equivalent single-prefill construction.

The scored budget is material. The kernel sets 131,072 maximum induction tokens and a 2,400-second per-call timeout. Its seven measured induction lengths range from 49,244 to 83,544 tokens, with median 61,284 (`scripts/kaggle/submission_kernel/main.py:595-627`). The vLLM server uses a default concurrency cap of eight; the kernel's preflight reads that value (`scripts/kaggle/submission_kernel/main.py:829-835`), and `_vllm_max_seqs` defines the default and documents the measurements (`python/carnot/agentic/arc_executable_world_model.py:4147-4211`). The full swarm subprocess has 41,400 seconds (`scripts/kaggle/submission_kernel/main.py:1585-1615`). The scored card is an offline NvidiaRtxPro6000 (`scripts/kaggle/submission_kernel/kernel-metadata.json:8-10`).

One correctly avoided median induction therefore removes about 61,284 generated completion tokens from the scored-code sample. It can also remove a long request. The serving code's separate nine-draw estimate is median 62,490 tokens and 1,562 seconds per stream at concurrency eight (`python/carnot/agentic/arc_executable_world_model.py:4160-4169`). That time is an estimate from documented measurements, not a scored-run guarantee. A readout still pays prompt prefill and one output step. Candidate selection at every action may cost more than it saves. E0 and E4 must measure that overhead.

### B1. Shadow candidate-action selector

**Hypothesis.** A readout over the incumbent top candidates can improve actions to real progress without removing the explorer's tail actions.

**Inputs and future code touch set.** Add shadow-only capture at `StepwiseExplorer._candidates` in `python/carnot/agentic/arc_competition_agent.py`. Record stable candidate IDs, the exact state text, option text, logits, incumbent order, chosen action, and later level outcome. Extend action provenance in the same file. Add an offline reducer at `scripts/experiments/semif_arc_readout_eval.py` and tests at `tests/python/test_semif_arc_readout.py`. Options are the incumbent top 15 plus `delegate_to_incumbent_tail`.

**Metric.** Primary A/B metric: levels gained, then actions to first level-up among runs with equal progress. Report right-censoring at 2,000 actions. Calibration metrics are Brier and ECE for whether an option leads to progress within fixed horizons. Immediate frame change is diagnostic only.

**Sample size and reason.** E2 must collect at least 1,000 eligible decision rows. E4 must use at least 30 paired episode units across at least 10 games and at least three seeds. Bootstrap and randomization are grouped by game. This supports a run-level percentage-point claim. It does not create 30 independent games. Any cross-game generalization claim stays sample-limited until 30 distinct games exist.

**Positive control.** An oracle selector that reads the already-recorded future level outcome must beat the incumbent when top-16 headroom exists. It is an analysis-only ceiling.

**Degenerate-case check.** Uniform, missing, or invalid probabilities delegate to the incumbent exact order. If the oracle top-16 cannot beat the incumbent, mark `FALSE_NEGATIVE_RISK` and do not interpret a learned null.

**Pass/fail gate.** Pass only if the paired interval improves actions to progress, loses no levels, the leave-one-game-out result has the same sign, and readout overhead stays below 5% of episode wall time. Otherwise fail.

**Kill criterion.** Drop B1 if there is no top-16 headroom, if gains disappear under leave-one-game-out, or if per-action prefill cost consumes the gain.

**Cost.** E2 adds no games and runs shadow readouts only where stored state is complete. E4 costs 60 bounded episodes at the minimum design: 30 pairs, two arms, at most 2,000 actions each. The readout can run once per eligible action, so this is the most expensive readout seam.

### B2. Induction timing gate

**Hypothesis.** A typed gate can avoid inductions that will be skipped, rejected, or fail to produce progress, while preserving useful inductions.

**Inputs and future code touch set.** Capture the `_should_enter_induction` state and the timed attempt row in `python/carnot/agentic/arc_competition_agent.py`. Use `continue_explore`, `induce_now`, `reinduce_now`, and `delegate_to_current_gate`. The state and existing labels are listed in the seam table. Add evaluation to `scripts/experiments/semif_arc_readout_eval.py`.

**Metric.** Primary: generated induction tokens avoided subject to non-inferior levels gained and actions to progress. Secondary: induction precision, recall of attempts that produced a verifier-passing plan, wall time, timeout count, Brier, and ECE.

**Sample size and reason.** Require at least 1,000 gate opportunities for calibration, at least 100 actual induction attempts, and the same 30 paired episode floor for A/B. Group all intervals by game and episode. If fewer than 100 attempts exist, report feasibility only.

**Positive control.** An analysis-only oracle that sees the attempt's final `planned`, verifier, and later progress fields must save tokens without losing progress.

**Degenerate-case check.** `never_induce` and `always_induce` are explicit controls. Missing logits delegate to the current gate. A policy that saves tokens only by suppressing all progress fails.

**Pass/fail gate.** Pass if generated induction tokens fall by at least 10%, the 95% interval excludes zero, levels do not fall, actions to progress do not worsen, and recall of progress-producing inductions is at least 95%.

**Kill criterion.** Drop B2 if useful and useless inductions are not separable out of game, or if one missed useful induction removes a level gain without a compensating replicated benefit.

**Cost.** One readout at each induction opportunity. A successful skip can avoid about 61,284 median generated tokens under the scored-code sample. The A/B generation budget stays identical between arms; only the gate may use less of it.

### B3. World-model accept, reject, or escalate gate

**Hypothesis.** Readout energy can add a calibrated soft feature to the independent world-model verifier and improve routing among accept, reject, and escalate. It cannot replace execution checks.

**Inputs and future code touch set.** Add readout fields beside the ordinary and hidden world-model trust gates in `python/carnot/agentic/arc_competition_agent.py`. Reuse verifier metrics from `python/carnot/agentic/arc_executable_world_model.py`. Add PoE composition and calibration to `python/carnot/verify/semif_readout_energy.py`. Add the offline reducer and tests named in B1. Keep `WorldModelVerifier` as final authority.

**Metric.** Primary: paired actions to progress and levels gained. Secondary: Brier and ECE for later plan success, false-accept rate for verifier-failing engines, escalation rate, plan-found rate, and tokens spent on alternatives.

**Sample size and reason.** Require at least 1,000 candidate decisions for sub-percentage calibration, at least 30 accepted and 30 rejected engines, and 30 paired A/B episodes. Split leave-one-game-out. If accepted or rejected counts fail, do not fit a three-way gate.

**Positive control.** Inject syntactically valid but transition-wrong engines. `WorldModelVerifier` must reject them. The PoE may not override that rejection.

**Degenerate-case check.** Uniform readout probabilities must reproduce the existing verifier decision. Same-model confidence without verifier evidence cannot accept an engine. Missing checker fields force reject or escalate.

**Pass/fail gate.** Pass only if false accepts do not increase, the PoE improves out-of-game Brier, and the paired live metric improves actions to progress without losing levels. A readout-only win does not pass.

**Kill criterion.** Drop B3 if readout confidence is highly correlated with the proposer but adds no conditional signal after verifier metrics, or if the fitted policy ever overrides an independent hard failure.

**Cost.** One readout per induced candidate after generation. It cannot recover the cost of the candidate already generated. Savings can only come from avoiding alternative generation or bad plan execution.

### B4. Curated supervisor-arm selector

**Hypothesis.** A typed readout over the curated arm set can choose a better eligible redirect than the fixed order after stagnation. This is selection over existing methods. It is not a request to invent a new strategy.

**Inputs and future code touch set.** Use `TrajectorySnapshot`, eligible arms, and existing outcome receipts. Add a shadow readout beside `TrajectorySupervisor._first_eligible_arm` in `python/carnot/agentic/arc_trajectory_supervisor.py`. Preserve the fixed selector as fallback. Extend `scripts/experiments/semif_arc_readout_eval.py` and `tests/python/test_semif_arc_readout.py`. REQ-6600 fixes the current arms and application seams (`openspec/capabilities/arc-world-model-trust-energy/spec.md:27618-27645`). REQ-6640 fixes the measurable outcomes (`openspec/capabilities/arc-world-model-trust-energy/spec.md:27970-28015`).

**Metric.** Primary: levels gained and actions from redirect to level-up. Secondary: arm fired/helped counts, split credit, unredirected windows, reinduction tokens, Brier, and ECE. Scripted qualification rows test plumbing only and are excluded from efficacy.

**Sample size and reason.** Require at least 30 eligible stagnation episodes and at least 10 observed outcomes for every arm that can be selected. For calibration claims, accumulate at least 1,000 shadow decisions across runs or label the result coarse. Use leave-one-game-out folds.

**Positive control.** Replay the existing fixed order and a scripted state where exactly one arm is eligible. Both must select the legal arm. A future-outcome oracle supplies the ceiling but never ships.

**Degenerate-case check.** An illegal arm is masked before normalization. Uniform or missing scores use the fixed order. `no_redirect` cannot hide an exhausted-arm condition from the ledger.

**Pass/fail gate.** Pass if the paired interval improves actions to progress, no level is lost, and no arm's harm interval crosses a pre-registered safety bound. The result must repeat under leave-one-game-out.

**Kill criterion.** Drop B4 if too few arms fire, if one game supplies most positive outcomes, or if the selector only learns the current fixed order. The existing bounded exposure artifact reports no arm promotion, so E2 must first prove outcome coverage rather than assume it (`results/experiment_7457_v653_arc_exposure.json:27139-27306`).

**Cost.** One readout only at a stagnation boundary. There is no generation cost unless the chosen existing arm invokes reinduction. This is the cheapest ARC decision seam after telemetry exists.

## 5. Roles

These roles split decision, emission, generation, and verification. They do not assume that a small model can discover a hidden mechanic by itself.

| Component | What it decides or emits in the live loop | Live seam and reachability |
|---|---|---|
| Needle 3 | Emit one schema-valid ARC action or tool call and return its own confidence. In the confidence-gate arm it also proposes `act`, `confirm`, or `refuse`. It does not get authority to accept a world model or invent a mechanic. The supplied README claims byte-level grammar, structured calls, and a calibrated confidence head (`/home/ianblenke/semif-refs/needle_README.md:3-11`, `:44-50`). | Candidate actions already exist at `StepwiseExplorer._candidates` (`python/carnot/agentic/arc_competition_agent.py:2808-2832`, `:2982-3032`), and every emitted action passes through `E3AgentPolicy.next_move` (`python/carnot/agentic/arc_competition_agent.py:6846-6883`). Needle is not imported there today, so it is not currently reachable. A future injected `decision_router` or `call_emitter` on `E3AgentPolicy`, passed through `make_carnot_agent`, would make it live-path reachable. A standalone Needle experiment would not count. |
| SemIf | Score the finite candidate set and emit a conditional probability distribution or an abstention signal. It may select an action, an induction timing option, or a supervisor arm only after target-workload calibration. It does not emit free-form code. | Candidate selection belongs after the existing rankers and before `_candidates` returns (`python/carnot/agentic/arc_competition_agent.py:2962-3032`). Induction timing belongs beside `_should_enter_induction` (`python/carnot/agentic/arc_competition_agent.py:6120-6145`) and the routed escalation (`:7464-7487`). SemIf is not reachable today. It becomes reachable only when the scorer is owned by `E3AgentPolicy` and the selected action still exits through `next_move`. |
| NanoJev-style runtime head | From state, question, and the current 2-to-255 candidates, emit a probability distribution without decoding. The clean ARC arm fits only a small choice head at runtime, prequentially, from the current unseen game's own observed action/outcome trajectory. It does not ship a head trained on the 25 public ARC games. | The input candidates are available at `_candidates`; the chosen action and later level/frame outcomes are available in action provenance (`python/carnot/agentic/arc_competition_agent.py:7174-7246`). It is not reachable today. A head reset in `E3AgentPolicy.__init__`, updated only after the next observed frame, and called from `_candidates` would be reachable from both `make_carnot_agent` and `E3AgentPolicy`. Persisting a fitted head across games would violate this plan. |
| Qwen3.8-27B generator | Generate or repair executable world-model code and goal predicates after the cheap routes cannot make progress. It is not a per-action serializer in this plan. | `_proposer` constructs the offline `LocalGGUFProposer` (`python/carnot/agentic/arc_competition_agent.py:6607-6651`); `_induce_and_plan` sends the agent's own transitions to `induce` (`:8444-8502`); the pin is `unsloth/Qwen3.8-27B-GGUF` (`python/carnot/agentic/arc_executable_world_model.py:3906-3914`). This role is already reachable. |
| Existing E3 cascade | Own the phase machine, deterministic exploration, escalation timing, independent world-model checks, planning, fallback, and final action. It masks illegal candidates and retains hard vetoes. | `E3AgentPolicy` is explicitly explore, induce, verify, plan, execute (`python/carnot/agentic/arc_competition_agent.py:5182-5193`). The default factory constructs it (`:9611-9676`), and `choose_action` delegates to it (`:9794-9814`). This is the incumbent and the baseline arm. |

Needle's two JevBench v1.2 entries are partial runs. Their reported intelligence scores are 22.4 and 39.5, with overall scores 16.8 and 19.2 (`/home/ianblenke/jevbench-src/RESULTS-v1.2.md:67-68`). That evidence argues against making frozen Needle the sole decision engine. Its plausible contribution is well-formed emission and a confidence signal, subject to ARC calibration.

### Pairing designs

| Design | Live flow | Possible saving | Main failure and reachability |
|---|---|---|---|
| Jev selector, grammar emitter | E3 enumerates legal candidates. SemIf or the runtime NanoJev-style head selects one. Needle receives only the selected option and its schema, or a plain deterministic grammar serializes it. E3 validates and emits the action. | Avoid free-form decoding and malformed-call recovery. If selection happens at a boundary that otherwise invokes Qwen3.8, it may also avoid a long generation. A plain grammar is the cost floor. | The second model may add latency without adding correctness. Needle may change the selected semantics while formatting them. The current ARC action is already a typed `(kind, data)` tuple, so Needle may have no advantage over a plain grammar. Not reachable today; reachable after injection at `_candidates` and `next_move`, without changing `choose_action`'s final `GameAction` conversion (`python/carnot/agentic/arc_competition_agent.py:9794-9814`). |
| Needle confidence routes act, confirm, or refuse | High calibrated confidence may act through E3. The `confirm` band and low-confidence/refuse cases go to the chosen Jev engine. If that engine is also uncertain, control returns to the incumbent cascade; Qwen3.8 is reached only through the existing stall-to-induction seam, not as a new per-action chatbot. | Easy cases can avoid the Jev call; hard cases can avoid Qwen only when the Jev decision restores progress before induction. | Needle confidence may be confident about syntax rather than action correctness, and its partial JevBench results warn that wrong high-confidence actions are plausible. A direct per-action 27B fallback would multiply cost and is outside the incumbent mechanism. Not reachable today; reachable if the routing thresholds live in `E3AgentPolicy` and `_should_enter_induction` remains the only 27B admission point (`python/carnot/agentic/arc_competition_agent.py:6120-6145`, `:7464-7487`). |
| Calibrated Needle then SemIf then 27B | Needle acts only above a leave-one-game-out risk threshold. Otherwise SemIf scores the same stable option IDs. SemIf acts only above its own margin and calibration threshold. Remaining cases delegate to the existing E3 cascade, which may later invoke Qwen3.8 and must still pass `WorldModelVerifier`. | Most potential savings come from preventing unnecessary 27B inductions. Needle can also spare SemIf calls on easy cases. | Correlated errors can make two apparent confirmations one repeated mistake. Threshold tuning can leak the held-out game. Sequential model calls can be slower than the baseline on common actions. Not reachable today; reachable through one E3-owned router. The independent ordinary verifier remains at `python/carnot/agentic/arc_competition_agent.py:8693-8759`; neither confidence source may override it. |

The same three designs may substitute the runtime NanoJev-style head for SemIf only after E9 passes. The published NanoJev result is not evidence for zero-training discovery: its README says the shared checkpoint was trained with complete-question cross entropy on task data, including 18,760 decision questions per variant and expert episodes (`/home/ianblenke/semif-refs/nanojev_README.md:64-89`). A runtime-fit head fits the ARC framing only if it is initialized without game-specific ARC knowledge, reset for every game, predicts before seeing the outcome, updates only after its own action produces an observation, reads no source or adapter, and does not carry fitted state between games. Under those conditions it is test-time training from the agent's own trajectory, like the framing of `arc_live_ttt`, rather than a trained known-game adapter. A public-game-trained or cross-game-persistent head would not qualify.

## 6. What "accelerate" means and how we measure it

"Greatly accelerate the solve rate" is an untested hope. In this plan, the **budgeted solve rate** is the fraction of pre-registered adapter-free game-seed episodes with at least one real level-counter advance before all shared caps expire. Also report total `levels_gained` and mean levels gained per episode because one binary solve can hide multiple levels. A level counts only when `frame.levels_completed` advances. This follows REQ-5720's ground truth (`openspec/capabilities/arc-human-replay-frame-change/spec.md:9208-9213`). It is not the 183-level registry total, held-out induction accuracy, immediate frame change, or a replay of a banked solution.

Every comparison freezes one budget tuple before outcomes are read:

- `A`: 2,000 environment actions per game, matching the submitted bound (`python/carnot/agentic/arc_competition_agent.py:9632-9650`).
- `T`: one run-wide prompt-plus-completion token budget measured and frozen by E6. The same cap applies to every arm. A skipped generation consumes zero completion tokens but the gate's prompt tokens still count.
- `W`: one run-wide wall-clock budget measured and frozen by E6, never exceeding the scored subprocess bound of 41,400 seconds (`scripts/kaggle/submission_kernel/main.py:1585-1615`). Per-request timeouts remain part of the cost, not censored away.

The primary metrics are:

1. Adapter-free budgeted solve rate, total `levels_gained`, and levels gained per episode under the identical `(A, T, W)` tuple.
2. `actions_to_first_solve` and actions between later level advances, with unsolved episodes right-censored at `A`, as required by REQ-5720.
3. Decision-point latency at p50, p90, and p99, split into candidate construction, Needle, SemIf or NanoJev, 27B generation, verifier, planner, environment wait, and total `E3AgentPolicy.next_move` wall time. Report cold and warm model cases separately.
4. Prompt, completion, and total tokens per episode and per level gained. If an arm gains no level, tokens per level is infinite, not omitted.
5. Scored-run feasibility: projected aggregate time must fit `W`, aggregate tokens must fit `T`, and the per-call 131,072-token and 2,400-second generation bounds remain visible (`python/carnot/agentic/arc_executable_world_model.py:4019-4028`).

The baseline arm is the unchanged current E3 cascade. Use exact paired game, seed, action cap, token cap, wall cap, model revisions, and episode order. Freeze all 25 public games and three seeds per game for 75 paired episode units. This clears the repository's 30-example floor, covers the full public roster, and gives repeated stochastic observations. It still supplies only 25 independent game clusters, so it cannot support a sub-percentage claim or prove hidden-game transfer. Use game-clustered bootstrap intervals and an exact paired randomization test. If intervals are wide, report an underpowered null rather than a negative finding.

Use leave-one-game-out fitting for SemIf calibration and any global thresholds. The runtime NanoJev-style arm is stricter: its head may learn only within its current episode and is evaluated prequentially. No per-game adapter, game source, banked trajectory, hand verifier, or previous run of that game is available to any arm. Keep `solve_provenance=live_agent_self_discovery`.

A "faster" arm must not win by receiving easier games. Every arm runs every identical game-seed unit. Report within-game paired deltas, each game's baseline difficulty, and results by baseline-unsolved, baseline-one-level, and baseline-multilevel strata. Report the worst five games and leave-one-game-out influence. An aggregate win fails if it comes only from dropping hard games, timing out hard cells, or changing the roster.

The shared positive control is an analysis-only future-outcome selector over the same enumerated candidates. It must demonstrate selectable headroom and improve actions to progress on the frozen logs. A separate latency control replaces a measured expensive decision with a zero-cost correct stub and must recover the corresponding measured time. These controls test outcome headroom and timing attribution; neither is deployable.

The null-result plan is to preserve the current cascade unchanged. If the positive control fails, label the learned null `FALSE_NEGATIVE_RISK`. If the positive control passes but a pairing arm has no positive interval, worsens any level count, shifts benefit to easier games, or increases tokens or wall time at equal progress, do not promote it. A null never triggers threshold retuning on the held-out games.

The speed ceiling follows Amdahl's law. If E6 measures replaceable decision work as fraction `f` of total wall time and the replacement is `r` times faster with fractional overhead `h`, then:

```text
speedup <= 1 / ((1 - f) + f / r + h)
perfect-replacement ceiling = 1 / (1 - f)
```

The code contains a rough estimate that Qwen3.8 induction could consume about 49% of the scored budget (`python/carnot/agentic/arc_executable_world_model.py:3906-3911`). If that estimate were the measured `f` and every induction vanished at zero overhead, the conditional ceiling would be about 1.96x. It is not an expected speedup: the estimate is not an end-to-end scored cost profile, useful inductions cannot all be removed, and the pairing adds work. Existing artifacts do not justify a numeric expected upper bound. E6 must either replace this conditional ceiling with measured shares or state that no bound can be estimated.

## 7. Experiments E6 onward

### E6. Current live-loop decision cost profile

**Hypothesis.** A bounded set of decision points, especially 27B induction, accounts for enough current wall time and tokens to make a small routing model relevant.

**Code touched.** First pass touches no code: reduce existing action provenance, induction attempts, request manifests, and phase spans. The timed induction wrapper already writes `wall_s` (`python/carnot/agentic/arc_competition_agent.py:7770-7817`), and the proposer already normalizes generated and prompt token counts when the backend returns them (`python/carnot/agentic/arc_executable_world_model.py:6966-7000`, `:7362-7410`). If existing artifacts lack a field, record the gap; a later implementation may add shadow timers around `E3AgentPolicy.next_move` and its subphases, but this planning pass does not.

The existing inputs are incomplete rather than empty. `results/experiment_7234_v637_arc_scored_dryrun.json` is adversarially flagged and failed its `actual_dispatch_engine_write_policy_consumption` gate, so exclude it from every quantitative profile. `results/experiment_5972_arc_llm_on_budget2000_feasibility.json` is clean and has per-cell calls, completion tokens, actions, and elapsed time, but it used Qwen3.6-35B-A3B rather than the current generator. `results/experiment_7457_v653_arc_exposure.json` is clean and has request timings under a 256-token request cap, not full inductions. E6 may use the two clean artifacts as reducer and schema controls. It may not merge them into a current-loop speed claim.

**Metric.** Fraction of end-to-end episode wall time and tokens at each decision point, field coverage, cold/warm split, and the Amdahl ceiling. Reconcile subphase time to episode time and request token totals to backend usage.

**Sample size and reason.** Use every unflagged authentic current-Qwen3.8 artifact with compatible fields. Require at least 30 complete episodes across at least 10 games before publishing a numeric share; otherwise report coverage only. The flagged dry run is excluded; short-request and older-model cost artifacts may test parsing but cannot establish the current share.

**Positive control.** Sum known synthetic phase spans and token rows with one injected delay and one injected token count; the reducer must recover both exactly. On real rows, reconciled subphase time must not exceed episode time except for explicitly concurrent spans.

**Gate.** Advance if at least 95% of episode wall time and 99% of model tokens are attributed, at least one replaceable seam has a positive lower interval on cost share, and no model-version mixing remains. Freeze `T` and `W` from this artifact before later A/B runs.

**Kill criterion.** Stop speed claims if the current artifacts cannot distinguish candidate selection, generation, verifier, planner, and environment time, or if all replaceable decision work is below 5% of wall time. Quality experiments may continue without an acceleration claim.

**Cost.** CPU-only reduction of existing artifacts. No new model call, game, GPU run, or submission.

### E7. Frozen Needle confidence as a gate

**Hypothesis.** Frozen Needle confidence, calibrated out of game, can identify a high-precision subset of action calls that may execute without a Jev or 27B escalation.

**Code touched.** Future work would add an injected shadow `call_emitter` and confidence fields to `E3AgentPolicy`, thread it through `make_carnot_agent`, and extend action provenance at `python/carnot/agentic/arc_competition_agent.py:7174-7246`. It must not change the incumbent action in the shadow stage.

**Metric.** Selective action error and risk-coverage, calibration error, false high-confidence actions, escalation rate, gate latency, malformed calls, then budgeted solve rate, actions to progress, tokens per level, and wall time against current E3.

**Sample size and reason.** At least 1,000 shadow decisions with at least 100 progress-relevant outcomes, covering all 25 games. Only after that, 75 paired episode units. This is enough for coarse routing and episode deltas, not a sub-percentage confidence claim.

**Positive control.** An analysis-only oracle confidence equal to the recorded future progress label must route useful actions and reduce escalation. A deliberately inverted confidence must fail.

**Gate.** Freeze thresholds inside leave-one-game-out folds. Pass only if the high-confidence slice has at least 25% coverage, its upper 95% selective-error bound is below the registered risk ceiling, no held-out game loses a level, and total wall time or tokens improve at equal progress.

**Kill criterion.** Drop Needle as a decision gate if confidence does not predict ARC correctness, if high-confidence errors cluster on hard games, if calibration reverses out of game, or if its call cost exceeds the avoided downstream cost. It may remain a grammar emitter.

**Cost.** One frozen Needle call per eligible shadow decision, then 150 bounded episodes for baseline versus gate at the minimum paired design. The reported model binary is 8-29 MB, but our runtime size and latency remain unmeasured.

### E8. SemIf option-readout selector

**Hypothesis.** A zero-training SemIf readout can reorder enumerated legal actions and improve actions to progress without paying autoregressive output cost.

**Code touched.** Future shadow scorer after the incumbent rankers in `StepwiseExplorer._candidates` (`python/carnot/agentic/arc_competition_agent.py:2962-3032`), injected through `E3AgentPolicy` and `make_carnot_agent`; reducer and tests remain the B1 files named above.

**Metric.** Top-one progress lift where oracle headroom exists, option-order stability, Brier and ECE, decision latency, tokens, budgeted solve rate, and actions to progress.

**Sample size and reason.** At least 1,000 shadow decisions, at least 30 progress-positive decision groups, and 75 paired episode units after calibration. Group by game and use leave-one-game-out folds.

**Positive control.** The future-outcome option oracle must beat the incumbent on the same top-k pool. A stable-ID mock scorer must survive option permutation.

**Gate.** Pass if the paired lower interval for actions-to-progress improvement is positive, no levels are lost, option reversal remains inside A4's bound, and added readout time is less than the time it avoids.

**Kill criterion.** Drop the selector if candidate headroom is zero, it only copies incumbent order, gains vanish out of game, or the scored backend cannot expose exact option scores.

**Cost.** One readout per eligible action in shadow and 150 bounded episodes for the paired test. This is likely more expensive than boundary-only routing because the grid changes each action.

### E9. Runtime-fit NanoJev-style choice head

**Hypothesis.** A small choice head fitted online from the current game's own trajectory can learn action-effect preferences quickly enough to improve later decisions in that same unseen game.

**Code touched.** Future per-game head state owned by `E3AgentPolicy.__init__`; candidate features from `_candidates`; pre-outcome prediction recorded at `next_move`; update only when the next frame joins the prior action to its observed outcome (`python/carnot/agentic/arc_competition_agent.py:6846-6905`, `:7174-7246`). Reset head, optimizer, replay buffer, and calibration state at every new agent construction.

**Metric.** Prequential log loss and Brier, regret against the incumbent and future-outcome ceiling, time to useful calibration, runtime fit and inference latency, memory, budgeted solve rate, actions to progress, and tokens per level. Immediate frame change is an auxiliary training target, not solve credit.

**Sample size and reason.** At least 1,000 strictly prequential decisions across all 25 games, at least 30 decisions after warm-up in every promoted episode, and 75 paired episode units. Report the learning curve by decision index. The 25-game cluster limit remains.

**Positive control.** A synthetic changing contextual bandit with delayed labels must be learned after reset, and a label-shuffled control must remain at chance. On ARC logs, the future-outcome oracle must show candidate headroom.

**Gate.** Pass only if prediction precedes every label, no state crosses a game boundary, prequential loss beats the frozen incumbent score out of game, runtime overhead stays below 5% of episode wall time, and paired progress improves without a level loss.

**Kill criterion.** Drop the runtime head if useful signal arrives only after the episode is effectively over, unchosen-action counterfactuals make the fit unstable, the head memorizes game IDs, or a warm start from published NanoJev weights is required for benefit. That would be known-task training, not the proposed discovery mechanism.

**Cost.** One small-backbone forward plus bounded head updates per eligible decision and 150 bounded episodes. Exact checkpoint bytes, simultaneous-memory fit, and training latency are blockers to measure before the paired run.

### E10. Jev selector plus grammar-constrained emitter

**Hypothesis.** Separating selection from serialization preserves Jev decision quality while eliminating malformed action or tool calls; Needle adds value only if it beats a plain grammar at equal selected actions.

**Code touched.** Future selector at `_candidates`, emitter immediately before the selected tuple leaves `E3AgentPolicy.next_move`, and final validation before `choose_action` maps it to `GameAction` (`python/carnot/agentic/arc_competition_agent.py:6846-6883`, `:9794-9814`). Record selector choice, serialized call, parsed call, confidence, and rejection reason.

**Metric.** Exact selected-option preservation, parse and schema success, invalid-call refusal, added latency and tokens, budgeted solve rate, and actions to progress. Compare current typed tuple, plain grammar, and Needle emitter.

**Sample size and reason.** At least 1,000 serialized shadow calls across every action schema, followed by 75 units per arm for three arms, 225 bounded episodes total. Pair all three arms by game and seed.

**Positive control.** Inject malformed coordinates, an illegal action ID, duplicate calls, and an empty call. Both emitters must reject them; the deterministic grammar must reproduce every valid selected option exactly.

**Gate.** Needle passes only if it has zero semantic substitutions, no worse schema validity than the plain grammar, and a positive latency or robustness benefit. The pairing passes live only if it improves cost at equal progress over baseline.

**Kill criterion.** Drop Needle emission if the existing typed tuple or plain grammar is already perfect, if Needle changes a selected action, or if its confidence adds no calibrated routing signal.

**Cost.** Shadow serialization first. The live comparison costs 225 bounded episodes only if E7 or E8/E9 supplies a passing selector; otherwise stop before live A/B.

### E11. Needle act-confirm-refuse routing into a Jev engine, then E3/Qwen

**Hypothesis.** A calibrated Needle gate can handle easy calls, send only uncertain calls to the best passing Jev selector, and reduce 27B inductions without suppressing progress.

**Code touched.** Future E3-owned routing state at `next_move`, calibrated gate before the selected action, and a low-confidence delegation bit consumed by the existing induction admission at `_should_enter_induction` and `_next_move_routed` (`python/carnot/agentic/arc_competition_agent.py:6120-6145`, `:7464-7505`). Qwen remains behind `_proposer`; no new direct action-generation path is added.

**Metric.** Coverage of all three Needle routes, Jev escalation rate, 27B induction calls avoided, recall of progress-producing inductions, decision latency, tokens per level, budgeted solve rate, and actions to progress.

**Sample size and reason.** At least 1,000 shadow routes with at least 100 rows in each route or a declared merged band, then 75 paired episodes against current E3. Thresholds are fitted leave-one-game-out.

**Positive control.** A constructed routing fixture must exercise act, confirm, refuse, Jev escalation, and eventual E3 induction. An oracle route must save at least one known expensive unhelpful induction without losing its paired level outcome.

**Gate.** Pass only if progress-producing induction recall is at least 95%, budgeted solve rate is non-inferior, actions to progress do not worsen, and both tokens and wall time have positive paired intervals.

**Kill criterion.** Drop the cascade if `act` contains confident errors, `refuse` becomes a blanket no-op, the Jev stage merely repeats Needle, or avoided inductions were already cheap or useful.

**Cost.** Up to one Needle and one Jev call per eligible action plus the unchanged 27B fallback, over 150 bounded episodes. Early exits must be counted; reporting only the calls that reached Qwen is invalid.

### E12. Calibrated Needle then SemIf then 27B cascade

**Hypothesis.** Two calibrated, differently constructed cheap signals can route easy and medium cases while reserving Qwen3.8 induction for the residual hard set.

**Code touched.** The same single E3-owned router as E11, with stable option IDs, two separately fitted thresholds, an agreement feature, and a mandatory delegate-to-incumbent outcome. The ordinary and hidden-state verifier gates at `python/carnot/agentic/arc_competition_agent.py:8598-8689` and `:8693-8759` remain unchanged.

**Metric.** Incremental coverage and conditional error after each rung, error correlation, disagreement rate, 27B calls and tokens avoided, end-to-end decision latency, budgeted solve rate, actions to progress, and tokens per level.

**Sample size and reason.** At least 1,000 shadow decisions with at least 100 Needle/SemIf disagreements, then 75 paired episodes. If fewer than 100 disagreements exist, the second rung has no measurable independent role and the live test does not start.

**Positive control.** A synthetic ladder with known easy, medium, and hard cases must route each stratum to its intended rung. A duplicated SemIf signal must be detected as correlated and must not receive false two-vote credit.

**Gate.** Pass only if SemIf adds conditional signal after Needle, the hard residual still reaches E3/Qwen, no verifier veto is overridden, no level is lost, and equal-progress token and wall costs both improve with positive paired intervals.

**Kill criterion.** Drop the second cheap rung if its errors are effectively identical to Needle's, sequential overhead erases savings, the vLLM score surface is absent, or benefits disappear leave-one-game-out.

**Cost.** Up to two cheap-model calls per eligible decision and the existing 27B residual, over 150 bounded episodes after both single-component gates pass. This is last in the ladder because it is uninterpretable until E7 and E8 work separately.

## 8. Sequencing

1. **E0: prove readout access and runtime parity.** Use at least 100 frozen prompts spanning two to 16 options. Obtain mapped option distributions from the local llama.cpp path and from the exact scored vLLM wheel. Record option tokenization, logits or log-probabilities, total variation, argmax agreement, latency, and prompt-token count. Pass if both paths return every declared option, the lower 95% bound on argmax agreement is at least 95%, and median total variation is at most 0.05. Stop the ladder if either path lacks exact declared-option scores, labels are not single tokens, or parity fails. Do not tune the bounds after seeing the result.

2. **E1: run the JevBench public-task readout evaluation.** Use `/home/ianblenke/carnot-wt-jev/scripts/jevbench_readout_eval.py`. The inspected script targets 231 public tasks, reads next-token GGUF logits, runs a reverse-order probe, fits out-of-fold temperature scaling, and uses 1,000 bootstrap resamples (`/home/ianblenke/carnot-wt-jev/scripts/jevbench_readout_eval.py:26-32`, `:216-239`, `:459-475`, `:558-567`, `:746-747`). Treat this as a text-decision transport test only. Stop if the option readout fails, calibration is no better than prevalence, the order probe is unstable, or bootstrap intervals do not support the registered claim.

3. **E2: build offline ARC telemetry.** Parse existing live-agent artifacts. Add shadow logging only where existing artifacts lack the option set or state text. Do not start new games for label construction. Require stable IDs, exact prompt bytes, declared options, raw logits, chosen decision, and the existing outcome fields. Stop if fewer than 1,000 labeled decisions are available for a seam, fewer than 30 grouped episode units exist for an effect test, or the candidate set cannot be reconstructed. Log each missing field in `ops/verifier_gaps.md` when implementation begins.

4. **E3: calibrate by seam.** Fit temperature, true Platt, and isotonic methods inside leave-one-game-out folds. Never share a game between fit and test. Keep no per-game adapter or per-game fit. Stop a seam if calibration does not beat prevalence and raw softmax, if ECE remains above 0.05, or if option-order sensitivity fails A4.

5. **E4: paired A/B in the adapter-free offline arcade.** Run the real E3 cascade from `scripts/arc_loop_solve.py --mechanism e3`. Use at least 30 paired game-seed episode units. Set `solve_provenance=live_agent_self_discovery`. Read no game source. Use no per-game adapters. Decide on levels gained and actions to progress, not held-out world-model accuracy. Stop if any arm loses levels, worsens actions to progress, fails leave-one-game-out, exceeds the time budget, or lacks a positive interval on its registered primary metric.

6. **E5: only then consider a scored-path change.** First add shadow-only scored telemetry. Require E0 parity, an E4 pass, unchanged hard verifier authority, bounded overhead, and a rollback flag. Submission is operator-only. Stop if the exact offline wheel cannot return the needed scores, the Blackwell shadow distribution drifts from E0, or the operator does not approve a scored run. For the new pairing request, E5 is held until E6 through E12 finish; SemIf-only evidence does not authorize Needle or NanoJev on the scored path.

7. **E6: measure the incumbent cost share from existing artifacts.** Attribute wall time and tokens before claiming an acceleration opportunity. Freeze the later A/B budgets here. Stop numeric speed claims if fewer than 30 compatible current-model episodes exist or attribution coverage fails.

8. **E7: test frozen Needle confidence in shadow, then paired A/B.** Keep Needle weights frozen and use leave-one-game-out calibration. If confidence does not identify a safe high-coverage slice, retain only grammar emission and skip E11/E12.

9. **E8: test the SemIf option selector.** Require candidate headroom, order stability, out-of-game calibration, and paired actions-to-progress lift. A JevBench win alone does not advance this rung.

10. **E9: test the runtime-fit NanoJev-style head.** Reset at every game and evaluate prequentially. Do not use the published task-trained head as the live discovery mechanism. Advance it as an alternative Jev selector only if it beats the incumbent after its online training cost.

11. **E10: separate selector from emitter.** Compare the winning Jev selector with the current typed tuple, a plain grammar, and Needle emission. Stop Needle emission if it changes semantics or cannot beat the plain grammar.

12. **E11: test Needle act-confirm-refuse routing into the winning Jev engine and then the incumbent E3/Qwen boundary.** Preserve useful-induction recall and require equal-budget progress before crediting saved calls.

13. **E12: test the specific Needle then SemIf then 27B cascade.** Run it only if E7 and E8 pass separately and at least 100 Needle/SemIf disagreements exist. This is the last offline rung before any new scored shadow request.

Each rung preserves its artifact even when it stops the ladder. A stopped ladder is an informative result.

## 9. Blockers and licenses

| Item | Known state | Blocker before an offline Kaggle bundle or scored-path test |
|---|---|---|
| Needle 3 code and engine | The operator-supplied repository fact is Apache-2.0. The supplied README says the 8-29 MB weights and platform engines are hosted on Hugging Face and that telemetry is on by default (`/home/ianblenke/semif-refs/needle_README.md:3`, `:27`, `:73-87`). | The weight license is unverified from the supplied reference. Apache-2.0 on the repository does not by itself license separately hosted weights. Verify the model card, redistribution, commercial/competition use, the prebuilt engine license, transitive native dependencies, exact hash, and an offline telemetry-disabled mode before bundling. |
| NanoJev code, weights, and data | The operator-supplied repository fact is MIT. The README identifies a public 0.6B checkpoint and dataset and says the release contains a trained step-400 checkpoint (`/home/ianblenke/semif-refs/nanojev_README.md:5-11`, `:80-93`). | The supplied reference does not state the checkpoint or dataset license, and the `nanojev_files.txt` listing proves only that a repository `LICENSE` file exists. Verify code, weights, dataset, tokenizer, and base-model licenses separately. Exact download bytes and the inference/training dependency closure are also unverified. Prefer a NanoJev-style runtime head over redistributing the task-trained checkpoint until this is settled. |
| SemIf code and weights | SemIf code is MIT. Its README says model weights are not included and upstream models retain their own licenses (`/home/ianblenke/semif-refs/semif_README.md:165-172`). | There is no single "SemIf weights license." Verify the chosen Qwen, MiniCPM, or other upstream weight license, tokenizer, quantization, and redistribution terms. A separate 4B model also adds storage and simultaneous-memory pressure. |
| Offline Kaggle bundle | The current scored kernel already declares six dataset sources for code, runtime, Qwen weights, vLLM wheels, and compiled kernels (`scripts/kaggle/submission_kernel/kernel-metadata.json:11-18`). The existing path discovers attached assets under `/kaggle/input`. | No current total attached-data limit, per-dataset limit, or allowed-source-count limit was verified from the offline repository. Before adding a model, record exact compressed and mounted bytes, source count, import/load time, host RAM, device memory, and whether the competition rerun can attach and read it with internet disabled. Do not assume a small parameter count means the full dependency closure fits. |
| SemIf logprob access | The first pass found that the scored vLLM wrapper drops logprobs and prompt-token usage, and the exact offline wheel schema is unverified (`python/carnot/agentic/arc_executable_world_model.py:7362-7410`). | This blocks SemIf-over-the-current-Qwen designs, including E8 and E12, until E0 proves exact declared-option scores. `prompt_logprobs` alone is insufficient. A separate SemIf model can bypass this API blocker, but only by clearing the added license, dataset, memory, and latency blockers. |
| Needle and NanoJev score access | Needle supplies its own confidence, and NanoJev supplies its own distribution, so neither inherently needs Qwen vLLM logprobs. | They still need offline-loadable runtimes, stable schemas, full artifacts, licenses, and measured coexistence with the Qwen server. Needle's grammar does not repair the separate Qwen tool-parser blocker; in this plan it serializes ARC actions after selection. |

Decentralization implications: all proposed components are local and must run with internet disabled. No closed Jev service is a dependency. Before publication or distribution, any new weights, trained runtime-head artifact, dataset, and native package must have independently mirrored, content-addressed distribution as required by the repository's decentralization rule. Until the licenses permit redistribution, a hash and a download recipe are not a substitute for a legal offline competition bundle.

## 10. Risks and reasons to drop the approach

- **Hidden-game distribution shift.** Calibration learned on public games may fail on hidden mechanics. Leave-one-game-out reduces leakage. It does not prove hidden-game calibration. Drop probability-valued decisions if shadow ECE drifts beyond the registered bound.
- **Option-order sensitivity.** The reported 72% to 21% reversal is large. Drop the method if semantic distributions are not stable under A4.
- **GGUF versus vLLM numerics.** Quantization, tokenizer behavior, kernels, and API normalization may change the distribution. Stop at E0 if semantic-option parity fails.
- **3090 versus Blackwell.** Local development and scored serving use different hardware and model formats. The code itself warns that induction lengths came from dev hardware and that scored concurrency behavior is estimated (`python/carnot/agentic/arc_executable_world_model.py:4199-4203`). Require Blackwell shadow receipts before control authority.
- **Circularity.** The same model may propose and score a world model. Keep the readout soft. Never let it override an independent execution failure. Drop B3 if it adds no conditional signal after verifier fields.
- **JevBench-to-ARC non-transfer.** Text decision accuracy is not grid-state reasoning. A strong E1 result does not advance B1 through B4 without ARC evidence.
- **Time budget.** A one-pass readout over a long changing state can still be expensive. Shared prefix reuse is limited when the grid changes every action. Drop per-action B1 if overhead consumes its progress gain. Prefer boundary-only B2 or B4.
- **Saturated data.** High detector accuracy with zero selection headroom can create a clean but meaningless null. Require `oracle_at_k > incumbent_at_1` or report `FALSE_NEGATIVE_RISK`.
- **Small game count.** Thirty episode units are not thirty independent games. Keep cross-game claims sample-limited until the distinct-game floor is met.
- **Decentralization.** Use the local open model and offline serving paths. Do not make a closed service a dependency. Keep fallbacks local and auditable.
- **Needle role confusion.** Grammar validity and confidence are not ARC mechanic intelligence. The partial JevBench runs are weak decision evidence. Drop Needle decision authority before dropping its narrower emitter role.
- **Runtime-training leakage.** A NanoJev-style head that carries state across games or trains on public-game trajectories is an adapter. Reset it per game and audit prediction-before-label order.
- **Correlated cascade confidence.** Needle and SemIf may agree because they share prompt artifacts, not because they provide independent evidence. Measure conditional information and disagreement; two correlated votes are not an oracle-distinct verifier.
- **Easy-game substitution.** Aggregate speed can rise while hard games time out or disappear. Exact paired rosters and difficulty-stratified reporting are mandatory.
- **Packaging pressure.** A small model can still require a large tokenizer, runtime, compiler, or wheel closure. Clear mounted-size, load-time, RAM, and device-memory checks before a scored shadow test.

## 11. What was not verified and operator questions

### Not verified

- The SemIf repository was reachable. Its README and `src/semif_phase1/direct.py`, `core.py`, and `shared.py` were read from current `master` on 2026-09-20. SemIf was not cloned, installed, or executed. Its exact commit was not pinned. Its speed and accuracy claims were not reproduced.
- The read-only JevBench clone was inspected. Nothing in it was executed. No confidence interval for the 75.4 versus 74.7 gap was found.
- The E1 script exists in the separate worktree and was inspected. It was not run. Its results do not yet exist here.
- The exact version and schema of the vLLM wheel attached to the scored kernel were not verified. Current upstream API fields may not match that offline wheel.
- No local llama.cpp server readout and no scored vLLM readout were executed. E0 parity is open.
- Existing ARC artifacts were not proven to contain 1,000 usable decisions for any seam. Candidate option sets are not present in the current action provenance.
- No 3090-to-Blackwell numeric comparison was run.
- No hidden game, GPU, scored kernel, or submission was used.
- Needle 3 was not downloaded, installed, or run. Its ARC action quality, confidence calibration, grammar behavior on coordinate calls, offline engine behavior, telemetry-disable behavior, exact binary size, and weight license were not verified. The two JevBench v1.2 rows are partial and were not reproduced here.
- NanoJev was not cloned or run. Its reported ViZDoom, maze, and Snake results, checkpoint identity, model and dataset licenses, exact bytes, and runtime dependency closure were not verified. The supplied file list is not license evidence for separately hosted weights.
- A runtime-fit NanoJev-style head was not implemented or tested. Its compliance with the ARC framing is conditional on per-game reset, prequential updates, no adapter or source access, and no cross-game fitted state.
- The third-party `liao96312/jev-arena-nanojev` project was not inspected. It is only a lead for a small grid-decision lab, not evidence.
- No artifact established that Needle, SemIf, NanoJev, and Qwen can coexist inside the scored storage, host-memory, device-memory, startup, and wall-clock bounds. Current Kaggle attached-data and source-count limits were not verified.
- The public registry's 183 reproducible levels across 25 games are a solved development proxy, not a hidden-game score. Hidden-game performance for every proposed arm remains unknown.
- The claim that pairing Needle with a Jev engine would greatly accelerate solve rate was not tested. Existing artifacts do not provide a compatible end-to-end current-Qwen3.8 cost profile, so no numeric expected speedup is verified. The 1.96x figure above is only a conditional perfect-removal ceiling derived from an estimated 49% share.

### Questions for the operator

1. What cost matrix should define `accept`, `reject`, and `escalate`? In particular, what is one escalation worth relative to one false accept and one false reject?
2. Should a readout remain a soft PoE feature in all cases, or may it control B1 and B4 after their gates pass? This plan does not allow it to override `WorldModelVerifier`.
3. Is the top-15-plus-incumbent-tail action representation acceptable, or should B1 be dropped when more than 16 actions are legal?
4. May E0 inspect and probe the exact attached vLLM wheel on a Blackwell run before any agent behavior changes?
5. Are the E0 parity bounds acceptable: 95% lower-bound argmax agreement and median total variation at most 0.05?
6. Which public game roster should supply the adapter-free E4 pairs? The roster must be frozen before outcomes are read.
7. If E4 passes, the operator must decide whether and when to perform E5. This plan does not submit.
8. Should E9 use a generic frozen Qwen3-0.6B backbone with a fresh runtime head, or may it use the published NanoJev backbone after its training corpus and license are audited? The fresh-head arm is the cleaner generalization test.
9. If both SemIf and the runtime NanoJev-style head pass separately, which one should E11 treat as the primary Jev engine? The plan defaults to the lower-cost arm at equal progress.


## 12. Operator decisions, 2026-09-20

The operator answered the questions in section 11 with "follow your recommendations". The
outer loop's recommendations are recorded here as decisions.

| # | Decision |
|---|---|
| 1 | No single cost matrix is set yet. Escalation cost is derived from measured token and time costs once E6 has them (its first run was a sample-limited null). Until then every accept, reject, escalate policy is reported over a grid of cost ratios and compared by AUROC, ECE and Brier, not by one assumed cost. |
| 2 | A readout stays a soft feature. It has no authority to control candidate-action or supervisor-arm choices, and it never overrides `WorldModelVerifier`. Revisit only after the relevant gates pass. |
| 3 | The top-15-plus-delegate-to-incumbent action representation is accepted. |
| 4 | Granted: a private, non-scored Kaggle probe of the exact vLLM wheel on a Blackwell run (E0). It must not touch the scored kernel. It does not authorize any scored-path change. |
| 5 | The E0 parity bounds stay (95% lower-bound argmax agreement, median total variation at most 0.05). Do not loosen them to pass. The first E0 run found identical-runtime pairs already missing the bounds, so read those rows first. |
| 6 | The frozen roster is in `semif-e4-frozen-game-roster-2026-09-20.md`. It is nine games chosen by a fixed hash rule, three per action-interface stratum. |
| 7 | Deferred. It depends on E4, which has not run. Submission stays operator-only. |
| 8 | E9 uses the generic frozen Qwen3-0.6B backbone with a fresh runtime head. The published NanoJev backbone stays out until its training corpus and license are audited. |
| 9 | Deferred. It depends on both engines passing separately. |

Also approved the same day: shadow telemetry on the live agent (off by default, no behavior
change), which supplies the missing option sets and per-decision timing for E2 and E6.
