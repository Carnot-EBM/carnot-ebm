# SOTA ingestion 2026-06-11: OFF-ARC verifier transfer + hierarchical search

**Receipt fields**
- honest_verdict: `complete: sota_ingestion_offarc_and_search_mapped`
- methods_mapped_count: 10
- inference_substrate: `aggregation_from_upstream_artifacts`
- flagged_for_v374:
  - `off_arc_demo_fit_vs_aces_and_doce_protocol`
  - `vc33_hierarchical_subgoal_search_with_rerooting_fallback`
  - `sep_symbolic_partition_diagnostic_for_code_transfer`

**Citations**
- {arxiv_id_or_url: `https://arxiv.org/abs/2408.13745`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2604.03922`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2604.06485`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2602.04254`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2504.09643`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2604.03208`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2506.07255`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2103.11505`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2504.04366`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2605.30664`}

**Fresh pass provenance**

Read seed corpus from `research-references.md` section "2026-06-11 (.373 planning sweep)" and
filtered `research-studying.md` for the two active tracks. Ran:

- `python scripts/sweep_clusters.py 0 --max-results 8`
- `python scripts/sweep_clusters.py 3 --max-results 8`
- `python scripts/sweep_semscholar.py "execution based code generation code selection reranking tests" --limit 8`
- `python scripts/sweep_semscholar.py "symbolic equivalence partitioning code generation selection" --limit 8`
- `python scripts/sweep_semscholar.py "hierarchical planning latent world models subgoal heuristic search" --limit 8`
- `python scripts/sweep_semscholar.py "subgoal guided policy heuristic search learned subgoals" --limit 8`

Semantic Scholar returned HTTP 429 on the code-selection queries even after a sequential retry. The
subgoal query returned `2506.07255`, `2103.11505`, `2510.17382`, `2605.10634`, and `2605.30664`;
only `2103.11505` and `2605.30664` were strong enough for this .373 mapping. Low-concurrency
WebSearch/WebFetch then loaded the top arXiv pages. The `/deep-research` skill was not invoked.

---

## OFF-ARC execution-consistency verifier transfer

**Current Carnot substrate:** `python/carnot/verify/sandbox.py` provides restricted function
execution with gVisor when available and in-process fallback otherwise. `python/carnot/agentic/
arc_gap4_execution_verifier.py` holds the GAP-4 execution-consistency shape: induce a candidate
rule/program from demonstrations, execute it, and score candidates by exact output disagreement.
For MBPP/HumanEval, the ARC grid DSL itself should not be reused as-is; the reusable primitive is
the visible-test execution gate and pass-matrix/content-consistency scoring.

### 1. DOCE execution-based reranking protocol

**Method:** DOCE is the clean protocol anchor for exp4031/4032: generate N candidates, execute
visible tests, compare execution-based filtering/reranking/MBR/self-debugging against
execution-free baselines, and evaluate only on hidden tests at the end.

**Implementation over Carnot stack:** Build the OFF-ARC runner as the DOCE-style experiment:
same candidate pool for all arms, visible tests executed through `sandboxed_exec_function`, hidden
tests withheld until scoring, and bootstrap CI on Arm B minus Arm A. Arm A is vote/self-consistency,
Arm A++ is ACES, and Arm B is the GAP-4 demo-fit exact-output selector. Add an oracle/best-of-pool
positive control so a saturated pool is reported as uninformative, not as verifier failure.

**Pitfalls / where it fails:** Visible tests can be too weak, noisy, or ceiling-saturated. If the
oracle and vote are close, demo-fit cannot show a lift. Trial tests also invite overfitting if the
candidate generator sees hidden tests or if repair loops mutate against evaluation tests.

### 2. ACES leave-one-out test consistency

**Method:** ACES weights tests by whether each held-out test's pass/fail pattern agrees with the
candidate ranking induced by the remaining tests. It turns the public-test pass matrix into a
test-quality signal without needing to know which candidate is truly correct.

**Implementation over Carnot stack:** After sandbox execution, store a binary candidate x visible
test matrix. ACES can run as Arm A++ with no extra LLM calls and almost no extra execution. It is
the fair stronger baseline for the GAP-4 selector because both operate on the same pass matrix.

**Pitfalls / where it fails:** ACES is still bounded by the tests. If all public tests are trivial,
correlated-wrong candidates can rank well. It also does not inspect program semantics beyond pass
patterns, so it can tie or beat a naive demo-fit selector when the selector only counts visible
passes.

### 3. Symbolic Equivalence Partitioning (SEP)

**Method:** SEP filters with public examples, partitions candidates into bounded functional
equivalence classes using symbolic execution, then selects from the dominant semantic class. The
key contribution is separating syntactic diversity from functional diversity.

**Implementation over Carnot stack:** Use as a diagnostic/enrichment layer, not the first primary
arm. After the sandbox public-test filter, run a bounded symbolic or property-based equivalence
probe on surviving Python functions where signatures are simple. Record class size, public-test
pass rate, and hidden-test gold rate. This can catch cases where exact-output public tests
over-split equivalent implementations.

**Pitfalls / where it fails:** Symbolic execution can explode on arbitrary Python, library calls,
recursion, mutation, floating point, or IO. Dominant equivalence classes can still be correlated
wrong. Keep it bounded and fail closed to "SEP unavailable" rather than blocking exp4032.

### 4. Agentic Verifier targeted counterexample generation

**Method:** Agentic Verifier actively searches for discriminative tests that separate candidate
solutions, rather than relying on random inputs or static visible tests.

**Implementation over Carnot stack:** Do not put this in the main exp4031 path. Use it as the
expensive comparator or .374 escalation when DOCE/ACES/Arm B are tied. If used, all generated tests
must be marked as inference-time generated, sandboxed, and kept separate from hidden gold.

**Pitfalls / where it fails:** It adds training/inference cost and a new generator/verifier model,
which weakens Carnot's cheap model-free verifier story. Generated tests can be invalid, too narrow,
or accidentally leak benchmark assumptions. Its best use is to explain hard ties, not to replace
the GAP-4 primitive.

### 5. Reinforced / learned reranking

**Method:** Reinforced reranking trains a reward/reranker over generated code candidates and can
improve code selection after iterative self-training.

**Implementation over Carnot stack:** Treat as a future distillation target if exp4032 shows the
model-free selector has signal. Store verifier-certified pass/fail traces as a corpus, but do not
train in exp4031/4032.

**Pitfalls / where it fails:** It needs a training loop, a nontrivial corpus, and careful leakage
controls. It is also the opposite of the immediate thesis: "cheap execution verifier transfers
off-ARC." Use it for .374 only if the cheap selector lands a clean signal worth compressing.

---

## Hierarchical/subgoal search over verified world model

**Current Carnot substrate:** `python/carnot/agentic/arc_heuristic_search_over_verified_wm.py`
already exposes bounded best-first search with `next_states`, `is_goal`, a coded goal-distance
heuristic, `nodes_expanded`, and terminal artifact schema. `python/carnot/agentic/
arc_goal_predicate_separation.py` derives and sandbox-validates `is_goal(state)` from cached
level-up traces without reading the environment's level counter. The .373 target is vc33: reuse the
verified world-model simulator plus the exp4034/exp4020-style goal predicate, then add hierarchy
above the existing best-first loop.

### 6. HWM temporal hierarchy / hierarchical MPC

**Method:** HWM plans at multiple temporal scales so high-level planning proposes coarse latent
targets and low-level planning executes shorter-horizon actions. The relevant lesson is temporal
decomposition, not latent rollouts.

**Implementation over Carnot stack:** Implement a symbolic HWM analog for exp4035: a high-level
planner proposes intermediate state predicates or landmark states over the verified simulator; the
existing `best_first_search` becomes the low-level planner that searches from current state to each
subgoal and finally to `is_goal`. Replan after each real action or confirmed simulator divergence.

**Pitfalls / where it fails:** Carnot has an exact-ish symbolic simulator, not a learned latent
space. Bad subgoals can add overhead or make the problem unsound if accepted without reachability
checks. Every subgoal must be verifier-reachable and must improve the coded distance or frontier
entropy before it is trusted.

### 7. Subgoal-Guided PHS with learned subgoals

**Method:** Subgoal-PHS learns subgoals and policies from search trees, including failed attempts,
so failed search is not wasted. It directly addresses long-horizon planning when complete solution
trajectories are scarce.

**Implementation over Carnot stack:** First run a bounded vc33 search and persist expanded states,
heuristic scores, parent actions, and dead ends. Mine candidate subgoals from frontier states that
reduce `unsatisfied_targets`, distance, or known vc33 progress features. In the first .373 run,
filter these as coded/verified landmarks rather than training a large policy; in .374, train a
small action prior from the logged trees if exp4035 produces enough data.

**Pitfalls / where it fails:** Learned subgoals can overfit one game and can be unreachable from
the real state. Failed-tree mining can select attractive dead ends. Require independent
reachability, exact simulator replay, and real-env confirmation before claiming wall breakage.

### 8. Policy-Guided Heuristic Search (PHS) guarantees

**Method:** PHS combines a policy and heuristic while retaining search-loss guarantees tied to
their quality. It is a better theoretical fit than unconstrained learned planning because Carnot
already has hard legality from the simulator.

**Implementation over Carnot stack:** Extend `best_first_search` priority from `cost + heuristic`
to include a policy/action-prior term learned or hand-derived from solved traces. Keep the hard
`max_expansions <= 50000`, log policy score and heuristic score separately, and compare against the
current coded heuristic-only baseline.

**Pitfalls / where it fails:** The guarantee depends on policy quality; a bad policy can reorder
the frontier toward dead ends. Start with a conservative tie-breaker, not a dominant term, and
report any expansion increase honestly.

### 9. Sokoban HRL landmarks

**Method:** Sokoban-HRL shows recursive learned subgoals and landmarks can scale in a
Sokoban-class puzzle domain. This is the closest domain analogy for the ARC-AGI-3 wall games.

**Implementation over Carnot stack:** Use landmarks as explicit intermediate predicates in vc33:
box/agent/target relation milestones, collision-state deltas, or progress-bar components exposed
by the verified WM. Feed those predicates to repeated bounded best-first calls, not to a learned
end-to-end HRL policy.

**Pitfalls / where it fails:** The paper learns hierarchies from scratch; Carnot has too little
vc33 data for that. Hand-mined landmarks can become r11l-style bespoke macros if they mention game
names or exact action sequences. Require feature-level landmarks and cross-level replay.

### 10. Rerooting Levin Tree Search (fresh pass)

**Method:** The fresh pass surfaced arXiv:2605.30664, which replaces explicit subgoal generation
with learned "rerooters" that implicitly decompose search into soft subtasks and reduce overhead.

**Implementation over Carnot stack:** Keep as the .374 fallback if explicit subgoal generation is
too expensive in vc33. Build a simple rerooter from clustered simulator states or heuristic
cost-to-go estimates: periodically reset the local search root to a verified frontier state that
improves progress, then continue bounded search. This can be implemented on top of the current
frontier logs without changing the verified simulator.

**Pitfalls / where it fails:** Rerooting is easy to turn into hidden macro-selection. It must
preserve provenance: selected reroot state, verified path to it, and proof that the plan still
executes from the real current state. If it cannot replay the root path, it is only a heuristic
diagnostic, not a solve.

---

## Bottom line for the .374 roadmap

1. **Strongest OFF-ARC path:** implement exp4031/4032 as a DOCE-style two-arm protocol with ACES
   as the stronger consistency baseline. This is the cleanest test of whether the GAP-4
   execution-consistency primitive transfers to MBPP/HumanEval. SEP should be a diagnostic, not a
   gate, unless arbitrary-Python symbolic execution is already bounded and stable.
2. **Strongest search path:** implement exp4035 as hierarchical symbolic MPC over the verified vc33
   simulator: high-level subgoals/landmarks, low-level bounded best-first, real-env confirmation.
   Add Subgoal-PHS logging from failed trees. If explicit subgoals are too costly, carry
   rerooting-LTS into .374 as the lower-overhead fallback.
3. **Do not overclaim:** Agentic Verifier and reinforced reranking are useful comparators or
   later distillation targets, but they do not support the immediate cheap-verifier transfer claim.
   Learned HRL is evidence for hierarchy, not permission to ship game-specific macros.
