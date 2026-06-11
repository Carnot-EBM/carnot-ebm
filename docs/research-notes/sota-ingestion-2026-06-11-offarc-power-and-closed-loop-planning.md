# SOTA ingestion 2026-06-11: OFF-ARC power + closed-loop planning

**Receipt fields**
- honest_verdict: `complete: sota_ingestion_offarc_power_and_closed_loop_mapped`
- methods_mapped_count: 9
- inference_substrate: `aggregation_from_upstream_artifacts`
- flagged_for_v375:
  - `offarc_full_power_sep_aces_agentic_counterexample_panel`
  - `closed_loop_vc33_replan_with_wm_trust_gate`
  - `novelty_mpc_gate_for_verified_wm_search`

**Citations**
- {arxiv_id_or_url: `https://arxiv.org/abs/2408.13745`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2207.10397`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2604.03922`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2604.06485`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2602.04254`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2510.05197`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2306.00840`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2510.18135`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2605.08732`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2508.06096`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2510.11892`}

**Fresh pass provenance**

Read the seed corpus from `research-references.md` section
"2026-06-11 Post-.373 Planning Sweep (Milestone 2026.06.374)" and filtered
`research-studying.md` to the two active tracks. Ran:

- `python scripts/sweep_clusters.py 0 --max-results 8`
- `python scripts/sweep_clusters.py 3 --max-results 8`
- `python scripts/sweep_semscholar.py "execution based code selection symbolic equivalence agentic verifier pass@k" --limit 8`
- `python scripts/sweep_semscholar.py "closed loop planning world model model error replanning novelty MPC" --limit 8`

Semantic Scholar returned HTTP 429 for both focused queries. Direct arXiv API fetches for the
cluster URLs also hit rate limiting/timeouts, so the content check used low-concurrency primary
arXiv WebSearch/WebFetch of the top papers: DOCE, CodeT, ACES, SEP, Agentic Verifier, pass@k
scaling, MuZero model-error, World-in-World, GC-IDM, novelty-MPC, and R-WoM. The `/deep-research`
skill was not invoked.

---

## OFF-ARC power + stronger discriminator

**Current Carnot substrate:** Exp 4032 already transferred the GAP-4 execution primitive to MBPP:
candidate programs run through the restricted code execution path, visible tests act as demos, and
hidden tests are held out for final scoring. The reusable stack is
`python/carnot/verify/sandbox.py` plus the GAP-4 demo-fit execution-selector pattern, not ARC grid
logic. The .373 result was directional (+5.0pp) but underpowered: n=40, CI95 touched zero.

### 1. Power-calibrated DOCE measurement with pass@k budgeting

**Method:** DOCE gives the clean code-selection protocol: generate a fixed candidate pool, execute
candidate code against trial tests, compare n-best reranking / MBR / self-debugging style selectors,
and evaluate hidden tests only at the end. The pass@k scaling paper adds the missing planning tool:
estimate how much N, k, and task count are needed before spending full generation budget.

**Implementation over Carnot stack:** Make exp4044/4045 a measurement, not another small smoke.
Use a frozen candidate pool for all selectors on full HumanEval (164) plus a predeclared MBPP
slice. Pre-budget with a small pilot using the pass@k estimator, then run 10k task-level bootstrap
CIs on Arm B minus Arm A/A++. Keep the oracle/best-of-pool positive control from exp4032 so a
ceiling-saturated pool is reported as uninformative. Store per-task pass matrices, hidden-test
outcomes, candidate hashes, sandbox mode, and bootstrap seeds.

**Pitfalls / where it fails:** pass@k forecasting predicts candidate-pool capability, not verifier
lift. Dynamic allocation can bias the comparison if it changes per-arm candidate pools, so any
adaptive budget must be pilot-only or applied identically across arms. A 5pp lift may still be
real but not claimable if the CI touches zero again.

### 2. ACES / CodeT / DOCE as the stronger same-pass-matrix baseline

**Method:** CodeT established dual execution agreement: generate tests, execute candidates, and use
candidate-output agreement as a selector. ACES is the newer stronger baseline over the same binary
candidate-by-test matrix: leave one test out, rank candidates with the remaining tests, and weight
tests by whether the held-out pass/fail pattern agrees with that ranking. DOCE supplies the broader
execution-reranking framing.

**Implementation over Carnot stack:** Add Arm A++ before claiming Carnot-specific lift. After
sandbox execution, persist a binary matrix `candidate_id x visible_test_id`. Run vote/pass-count,
CodeT-style dual agreement when generated tests are present, and ACES-C/ACES-O-style test weighting
on the same matrix. The demo-fit verifier must beat or complement this stronger baseline, not just
plain vote.

**Pitfalls / where it fails:** ACES and CodeT remain bounded by test quality. Correlated wrong
solutions can agree with each other, generated tests can be invalid, and visible tests can be too
weak to distinguish semantics. If Arm B only beats plain vote but ties ACES, .375 should frame the
result as execution-consistency parity, not a new discriminator.

### 3. Symbolic Equivalence Partitioning (SEP)

**Method:** SEP filters candidate programs with public examples, uses bounded symbolic execution to
partition survivors by functional equivalence, and selects from the dominant semantic partition.
The key signal is semantic agreement rather than syntactic diversity or raw pass count.

**Implementation over Carnot stack:** Use SEP as the first stronger-discriminator add-on for
exp4045. After visible-test filtering through `sandboxed_exec_function`, apply a bounded symbolic
or property-based equivalence probe only to simple Python signatures. Record equivalence-class
size, representative candidate, public-test pass rate, hidden-test gold rate, and a fail-closed
`sep_unavailable_reason` for unsupported Python. Compare three selectors: ACES, demo-fit, and
demo-fit+SEP tie-break.

**Pitfalls / where it fails:** Arbitrary Python symbolic execution can explode on loops, mutation,
library calls, IO, recursion, floating point, and large domains. Dominant semantic partitions can
still be correlated-wrong. SEP should improve diagnosis and tie-breaking before it becomes a gate.

### 4. Agentic Verifier targeted counterexample generation

**Method:** Agentic Verifier actively searches for discriminating inputs that separate candidate
programs, rather than relying on passive trial tests or random sampling. It is the clean
"stronger discriminator" direction when public tests under-determine hidden semantics.

**Implementation over Carnot stack:** Treat this as an expensive comparator or .375 escalation
branch, not the core .374 proof. If the full-power ACES/SEP/demo-fit panel is tied, run targeted
counterexample generation on the tied candidate clusters. Generated tests must be sandboxed,
proven valid for the problem signature, stored separately from hidden tests, and marked
`inference_time_generated`.

**Pitfalls / where it fails:** It adds training/inference cost and weakens the cheap model-free
verifier story. Generated tests can be invalid, overfit prompt phrasing, or accidentally encode
benchmark assumptions. Its best near-term role is tie explanation and hard-case mining, not a
replacement for the fixed-pool powered measurement.

---

## CLOSED-LOOP planning over a verified world model

**Current Carnot substrate:** Exp 4034 induced the vc33 `is_goal(state)` predicate with held-out
precision/recall 1.0. Exp 4035 then found a plan inside the verified vc33 world model, but the
repeated action plan exploited model error and failed real-env confirmation. The relevant stack is
`python/carnot/agentic/arc_vc33_goal_predicate_induction.py` plus
`python/carnot/agentic/arc_vc33_hierarchical_search.py`: verified simulator, predicate, subgoals,
bounded best-first search, and real-env confirmation.

### 5. MuZero-style model-error discipline: constrain search to the trustworthy policy support

**Method:** MuZero model-error analysis explains the vc33 failure directly: learned/value-equivalent
models can work on the data-collection policy while failing on unseen policies; policy priors keep
search closer to regions where model predictions are more accurate.

**Implementation over Carnot stack:** Add a WM-support score to every expanded vc33 state/action:
distance from observed transition traces, action frequency under real-env collection, simulator
disagreement where replay evidence exists, and horizon depth since last real observation. Penalize
or block branches that leave support. Replace "planner found a model-satisfying path" with "planner
found a path that stays inside verified support and then real-env confirms."

**Pitfalls / where it fails:** A conservative support prior can reject the genuinely new action
needed to solve a level. The gate should have an exploration escape hatch: if all branches are OOD,
return to real-env observation collection rather than forcing a stale model plan.

### 6. World-in-World closed-loop evaluation: execute, observe, revise, replan

**Method:** World-in-World argues that world models should be scored by closed-loop task success,
not open-loop visual or prediction fidelity. Its useful principle for Carnot is the loop: propose,
simulate, execute a bounded step, observe the real state, then replan.

**Implementation over Carnot stack:** Convert exp4046 from open-loop plan replay to receding-horizon
execution. At each step, run bounded search from the current real observation, commit only the next
action or a short certified prefix, execute it in the offline ARC environment, re-encode the new
state, and recompute both `is_goal` and WM trust. Artifact fields should include per-step replans,
trust scores, divergence events, and whether `levels_completed` advanced.

**Pitfalls / where it fails:** Closed-loop replanning costs more environment interaction and can
stall if the observation encoder is brittle. It does not solve model error by itself; without a
trust gate it can still replan repeatedly into the same bad simulator basin.

### 7. GC-IDM / single-step inverse-dynamics replanning

**Method:** GC-IDM replaces expensive online search with a goal-conditioned inverse-dynamics map:
current latent, goal latent, remaining horizon -> next action. The broad lesson is per-decision
amortized planning with fresh current-state encoding.

**Implementation over Carnot stack:** Build the symbolic analog first. Use the verified vc33 world
model and exp4034 predicate to derive action priors from local state deltas: "which next action
most reduces unmet goal components and stays in support?" Feed that prior as a tie-breaker or
expansion-order term into `hierarchical_best_first_search`, but execute one action at a time with
real observation refresh.

**Pitfalls / where it fails:** Pure one-step control can fail on forced multi-step chains where an
early action temporarily worsens the heuristic. Keep a short horizon fallback and report when
single-step replanning cycles.

### 8. Novelty-MPC trust gate

**Method:** Novelty-MPC uses a novelty detector inside model-predictive control so proposed
trajectories that leave the training distribution rely less on the learned model. This maps cleanly
onto the vc33 degenerate-plan failure.

**Implementation over Carnot stack:** Start with a non-learned gate: nearest-neighbor distance to
observed vc33 feature vectors, unseen action-repeat penalties, progress-bar inconsistency, and
simulator-vs-observation residual after each real step. Later replace or augment it with a small
VAE/energy detector if logged states are sufficient. Branches above threshold should trigger
re-observation, shorter horizon, or abstention.

**Pitfalls / where it fails:** The novelty detector becomes a single point of failure. False
negatives let the planner exploit the WM again; false positives block valid novel tactics. Report
coverage/abstention separately from solve rate.

### 9. R-WoM-style retrieval grounding for short lookahead

**Method:** R-WoM shows that long-horizon simulated procedure planning degrades, while grounding
world-model predictions with retrieved external/observed evidence improves longer-horizon use.
For Carnot, the analog is not tutorial retrieval; it is retrieval of verified local transition
traces and action effects.

**Implementation over Carnot stack:** Before expanding a candidate action, retrieve nearest
verified vc33 transitions and require the simulated effect to match one of the known local
transition families or an explicitly explored new transition. Use this as a pruning/trust feature
inside the closed-loop planner. Keep lookahead short and refresh retrieval after every real action.

**Pitfalls / where it fails:** Retrieval can overfit the observed trace library and reject necessary
new mechanics. It also does not prevent compounding error if k-step rollouts are trusted beyond the
retrieved support.

---

## Bottom line for the .375 roadmap

1. **Strongest OFF-ARC path:** run a full-power fixed-pool HumanEval+MBPP panel with vote, DOCE /
   CodeT agreement, ACES, demo-fit, SEP tie-break, and an Agentic-Verifier hard-tie comparator.
   The headline criterion is not "positive direction"; it is Arm B or Arm B+SEP CI95 lower bound
   above zero against ACES, with oracle headroom present.
2. **Strongest closed-loop path:** replace open-loop vc33 planning with receding-horizon execution:
   one real action or short certified prefix, fresh observation re-encoding, support/novelty trust
   gate, and replanning from the real state. A model-only `is_goal` satisfaction path is no longer
   sufficient evidence.
3. **Do not overclaim:** SEP/ACES can beat naive demo-fit without contradicting Carnot; that would
   mean the demo-fit verifier needs semantic partitioning. Closed-loop replanning can still fail
   honestly if the verified WM substrate lacks the transition family needed for vc33.
