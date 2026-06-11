# SOTA ingestion 2026-06-11: unsaturated execution verification + verifier pruner

**Receipt fields**
- honest_verdict: `complete: sota_ingestion_unsaturated_execverif_and_pruner_mapped`
- methods_mapped_count: 10
- inference_substrate: `aggregation_from_upstream_artifacts`
- flagged_for_v376:
  - `evalplus_hidden_rescore_fixed_pool`
  - `saga_generated_tests_as_discriminator_arm`
  - `gap4_online_pruner_for_explore_first_arc`
  - `equivpruner_state_action_cache_for_arc`

**Citations**
- {arxiv_id_or_url: `https://arxiv.org/abs/2305.01210`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2403.07974`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2507.06920`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2604.21598`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2604.06485`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2604.03922`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2602.01070`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2602.03975`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2505.16312`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2603.28135`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2603.28376`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2510.06135`}

**Fresh pass provenance**

Read the seed corpus from `research-references.md` section
"2026-06-11 Post-.374 Planning Sweep (Milestone 2026.06.375)" and filtered
`research-studying.md` to the two `.375` tracks. Ran:

- `python scripts/sweep_clusters.py 0 --max-results 8`
- `python scripts/sweep_clusters.py 3 --max-results 8`
- `python scripts/sweep_semscholar.py "EvalPlus HumanEval+ MBPP+ generated tests code verification LiveCodeBench SAGA DryRUN" --limit 8`
- `python scripts/sweep_semscholar.py "online verification adaptive test-time compute action pruning verifier guided search deep research asymmetric verification" --limit 8`

Semantic Scholar returned HTTP 429 for both focused queries, so this pass did
not treat S2 as a discovery source. Direct arXiv API fetches of the two cluster
URLs worked. Cluster 0 surfaced verifier / PRM / reward-hacking papers mostly
outside the two concrete tracks; cluster 3 surfaced current world-model papers
such as `2606.12072` and `2606.09457`, but they do not beat the direct
verifier-pruning references below for `.376` planning. Low-concurrency
WebSearch/WebFetch then verified the top seed papers plus the strongest fresh
adjacent pruner papers: `2602.03975`, `2505.16312`, and `2603.28135`. The
`/deep-research` skill was not invoked.

---

## UN-SATURATED execution-verification corpus

**Current Carnot substrate:** Exp 4056/4057 should reuse the existing
demo-fit code verifier, sandbox, and candidate-generation checkpoint. The core correction
is to stop measuring on base HumanEval/MBPP, where the `.374` run showed every
arm and oracle at 1.0 on the completed subset. Selection can still use visible
examples; final scoring must use EvalPlus-style hidden tests so the visible
tests under-determine the hidden semantics.

### 1. EvalPlus hidden-test rescore as the default .375 corpus

**Method:** EvalPlus extends HumanEval into HumanEval+ with roughly 80x more
tests and provides MBPP+ as the MBPP analog. Its load-bearing claim for Carnot is
not leaderboard rank; it is that base code benchmarks can miss wrong programs
that stronger generated tests catch.

**Implementation over Carnot stack:** Keep the fixed candidate pool and the
same visible-example demo-fit selector. Add an EvalPlus evaluation adapter that
normalizes task ids, candidate hashes, visible-test pass matrices, hidden
EvalPlus outcomes, sandbox timeout/errors, and bootstrap seeds. Compare vote,
demo-fit, symbolic/semantic baselines, and oracle headroom on the same pool.

**Pitfalls / where it fails:** EvalPlus can still saturate if the local
generator is too weak or the candidate pool has no diversity. The artifact must
report oracle headroom first; if oracle hidden pass is also 1.0 or all arms tie,
the result is another non-measurement, not evidence against demo-fit transfer.

### 2. LiveCodeBench v6 as the contamination-free escalation corpus

**Method:** LiveCodeBench continuously collects newer competitive-programming
tasks and includes execution-facing scenarios beyond plain code generation. It
is the escalation route if EvalPlus hidden tests no longer provide headroom.

**Implementation over Carnot stack:** Treat LiveCodeBench as a second-stage
adapter after EvalPlus lands. The sandbox must enforce the benchmark's runtime
and memory limits, and the result schema must preserve task date/window so later
analysis can separate contamination resistance from hardness.

**Pitfalls / where it fails:** LiveCodeBench tasks are heavier than HumanEval+
and MBPP+, so throughput can recreate the `.374` 22-task truncation problem.
Start with EvalPlus for the powered measurement, then escalate only if headroom
is absent or the checkpointed pool clears the minimum N.

### 3. SAGA generated-test discrimination arm

**Method:** SAGA reframes code verification as test generation: produce tests
that expose candidates passing superficial visible checks while failing hidden
semantics. This is the algorithmic analog of the Carnot demo-fit gap, but as an
active generated-test arm.

**Implementation over Carnot stack:** Add a bounded generated-test arm after
the fixed visible-test matrix is saved. Generated inputs must be validated
against the problem signature, executed in the same sandbox, stored separately
from official hidden tests, and never leaked into the final EvalPlus score.
Useful comparisons are demo-fit alone, SAGA tests alone, and demo-fit plus SAGA
tie-break.

**Pitfalls / where it fails:** Generated tests can be invalid, redundant, or
biased toward common wrong submissions. If SAGA tests train on or imitate hidden
EvalPlus distributions too closely, the measurement becomes a benchmark-specific
test generator, not off-ARC transfer.

### 4. DryRUN public-test-free simulation as the decentralization arm

**Method:** DryRUN removes public tests by asking the model to construct valid
inputs and mentally simulate executions for self-correction. The Carnot value is
as a public-test-free discriminator, not as a replacement for sandboxed final
scoring.

**Implementation over Carnot stack:** Run DryRUN-style self-simulation only as
metadata or a tie-break feature on candidate clusters. Keep the authoritative
result path execution-based: sandbox visible tests for selection diagnostics,
EvalPlus hidden tests for final score, and generated self-tests clearly marked
`inference_time_generated`.

**Pitfalls / where it fails:** Mental execution can hallucinate, and removing
public tests also removes the cheap grounding signal that makes the current
demo-fit verifier tractable. This is best for hard cases where visible tests are
absent or misleading, not the primary .375 powered measurement.

### 5. SEP / ACES as the stronger same-pool baseline

**Method:** SEP partitions candidates by bounded functional equivalence, while
ACES weights tests by leave-one-out consistency over the candidate-by-test pass
matrix. Both are strong non-Carnot baselines that can explain whether demo-fit
adds beyond execution consistency.

**Implementation over Carnot stack:** Reuse the sandbox pass matrix from the
EvalPlus adapter. Add ACES as the cheap Arm A++ baseline and SEP as a bounded
semantic tie-break for simple Python signatures. The demo-fit verifier should
be compared against these baselines before any claim of unique transfer.

**Pitfalls / where it fails:** ACES depends on test quality and can reward
correlated wrong candidates. SEP can blow up on loops, mutation, recursion,
floating point, and library-heavy code. Both should be fail-closed diagnostics
when the required program structure is unsupported.

---

## VERIFIER-GUIDED online action-pruning

**Current Carnot substrate:** The banked G2 result retires vc33 verified-world-
model planning, not ARC efficiency work. The forward stack is the proven
explore-first solver plus the model-free GAP-4 verifier. The .376 question is
whether GAP-4 can move from post-hoc confirmation into online pruning: score or
reject actions before expensive environment execution while preserving solve
coverage.

### 6. Adaptive online verification / prune-expand control

**Method:** `2602.01070` argues that verifier-guided online allocation beats
uniform sampling and post-hoc reranking. The important abstraction is using a
process verifier as a control signal inside generation/search, not just after a
candidate is complete.

**Implementation over Carnot stack:** Wrap the explore-first frontier in a
`gap4_prune_score(state, action, trace)` call before expansion. Low-risk
integration is soft pruning first: rank/skip only actions whose GAP-4 evidence
is dominated by already-tested siblings, record every pruned action, and replay
with pruning disabled on failure to estimate false-negative damage.

**Pitfalls / where it fails:** A verifier that is accurate post-hoc can still
be unsafe pre-execution if partial states lack evidence. The first experiment
must report both efficiency and coverage: actions avoided, verifier calls,
solves retained, and any solution path pruned by the gate.

### 7. Verification-cost-limited selective calls

**Method:** `2602.03975` studies verification as the scarce resource and
allocates verifier calls across intermediate states using feasibility gating,
pre-verification ranking, and uncertainty. This matches GAP-4 because the
verifier is cheap relative to LLM judging but still not free when called at
every branch.

**Implementation over Carnot stack:** Add a two-stage ARC gate: deterministic
feasibility filters first, then GAP-4 only for frontier nodes with uncertain
progress or high duplicate risk. The result should include verifier-call budget,
call sites, abstentions, and solve-preservation replay.

**Pitfalls / where it fails:** The "learned state-distance" part may not exist
for ARC-AGI-3 games yet. Start with deterministic features: repeated action
patterns, no-op deltas, already-seen state hashes, and GAP-4 exactness when
available. Do not train a new learned ranker before the logging schema proves
which states need it.

### 8. EquivPruner-style semantic-equivalent action cache

**Method:** EquivPruner prunes semantically equivalent reasoning actions to
reduce redundant search. ARC has a direct analog: many actions lead to identical
or verifier-equivalent states under the current game abstraction.

**Implementation over Carnot stack:** Add a canonical state/action signature
cache beside the explore-first frontier. If two actions produce the same state
hash or the same GAP-4-extracted program/effect signature, keep the cheaper or
more promising representative and mark the rest as pruned-equivalent.

**Pitfalls / where it fails:** Equivalence is game-local. Two states that look
equivalent under a weak signature may diverge after future actions, so the first
version should prune exact hashes and GAP-4-confirmed equivalence only. Approximate
semantic pruning can come later.

### 9. CoT2-Meta-style controller for expand / prune / repair / stop

**Method:** CoT2-Meta provides a training-free controller over expansion,
pruning, repair, stopping, and fallback. Carnot already has these pieces
informally; the missing part is a logged controller state machine.

**Implementation over Carnot stack:** Add an explicit controller record around
explore-first: at each step choose expand, prune, repair/retry, stop, or fallback
to unpruned search. Feed it GAP-4 scores, duplicate-state counts, action cost,
and recent progress. This is mainly orchestration and telemetry, not a new
solver.

**Pitfalls / where it fails:** A controller can hide failure by abstaining or
falling back too late. Acceptance must be paired: solve count with controller,
solve count with pruning disabled, and action/call cost deltas at equal task
sets.

### 10. Marco DeepResearch / asymmetric verification budget split

**Method:** Marco DeepResearch and asymmetric verification both argue that
verification can be cheaper than generation/search and can unlock stronger
test-time scaling when verification is deliberately budgeted. The relevant
transfer is the budget split: spend a modest, explicit verifier budget to avoid
much larger search/tool cost.

**Implementation over Carnot stack:** Define a verifier-budget ledger for ARC:
GAP-4 calls, sandbox/program checks, environment actions, LLM/tool calls if any,
and wall time. The pruner only wins if it reduces expensive actions or search
nodes without spending an unbounded verifier budget.

**Pitfalls / where it fails:** "Verification is easier than solving" is task
dependent. GAP-4 is reliable for exact execution/program checks, not for every
latent ARC subgoal. If the verifier cannot decide, it must abstain and let
explore-first continue rather than manufacturing confidence.

---

## Bottom line for the .376 roadmap

1. **Strongest track-1 method:** `evalplus_hidden_rescore_fixed_pool`. Resume
   the existing candidate-generation checkpoint, keep visible-example demo-fit
   selection, and score against HumanEval+/MBPP+ hidden tests with oracle
   headroom reported first. Add SAGA as a generated-test discriminator only
   after the fixed-pool EvalPlus path is stable.
2. **Strongest track-2 method:** `gap4_online_pruner_for_explore_first_arc`.
   Start with soft pruning plus replay-with-pruning-disabled so efficiency gains
   cannot mask coverage loss. Add exact state/action equivalence caching from
   EquivPruner before any learned or approximate pruning.
3. **Do not overclaim:** Track 1 must call another all-arms-1.0 result
   saturated, even if the code ran cleanly. Track 2 must report solve
   preservation separately from action reduction; a cheaper solver that prunes
   the only solution path is a regression.
