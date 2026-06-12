# SOTA ingestion 2026-06-11: v376 unsaturated corpora + online pruning

**Receipt fields**
- honest_verdict: `complete: sota_ingestion_v376_unsaturated_corpora_and_online_pruning_mapped`
- methods_mapped_count: 10
- inference_substrate: `aggregation_from_upstream_artifacts`
- flagged_for_v377:
  - `livecodebench_v6_local12b_headroom_route`
  - `evalplus_to_livecodebench_resume_accumulate_adapter`
  - `saga_generated_tests_hidden_score_tiebreak`
  - `gap4_soft_prune_replay_for_arc_efficiency`
  - `gap4_equivpruner_exact_state_action_cache`

**Citations**
- {arxiv_id_or_url: `https://arxiv.org/abs/2305.01210`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2403.07974`}
- {arxiv_id_or_url: `https://llm-stats.com/benchmarks/livecodebench-v6`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2604.06485`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2604.03922`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2507.06920`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2602.01070`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2603.10282`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2602.03975`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2505.16312`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2603.28135`}
- {arxiv_id_or_url: `https://arxiv.org/abs/2606.12402`}

**Fresh pass provenance**

Read the seed corpus from
`docs/research-notes/sota-ingestion-2026-06-11-unsaturated-execverif-and-verifier-pruner.md`
and the "2026-06-11 Post-.375 Planning Sweep (Milestone 2026.06.376)"
section of `research-references.md`. The four Exp 4055 `flagged_for_v376`
methods were treated as the seed; this was a confirm-and-extend pass, not a
fresh corpus rebuild.

Ran:

- `python scripts/sweep_clusters.py 0 --max-results 8`
- `python scripts/sweep_clusters.py 3 --max-results 8`
- `python scripts/sweep_semscholar.py "LiveCodeBench v6 verifier code generation 12B oracle headroom EvalPlus generated tests" --limit 5`
- `python scripts/sweep_semscholar.py "online verifier steering action pruning update-free on-policy verifier test-time compute" --limit 5`

The cluster script emitted arXiv API URLs; direct fetches of those URLs returned
mostly broad, off-track recent papers. The only fresh adjacent paper from the
cluster-3 URL that changes the pruning context was `2606.12402` DIRECT, which
confirms test-time compute routing is active but does not displace verifier
pruning. Semantic Scholar returned zero arXiv IDs for the code-headroom query
and HTTP 429 for the verifier-pruning query. Low-concurrency WebSearch/WebFetch
then verified the primary sources named in the seed plus the new online-steering
paper `2603.10282`, the selective-verification paper `2602.03975`, and the
fresh DIRECT adjacent router paper. The `/deep-research` skill was not invoked.

---

## Confirmed .376 actionability from Exp 4055

- `evalplus_hidden_rescore_fixed_pool`: actionable now as the first resumed
  measurement gate, but not sufficient by itself because the .375 smoke reported
  no 12B oracle headroom. Keep it as the cheap hidden-test adapter and escalate
  to LiveCodeBench v6 when headroom is absent.
- `saga_generated_tests_as_discriminator_arm`: actionable after the fixed pool
  and official hidden-score route are stable. It is a discriminator/tie-break,
  not the authoritative final score.
- `gap4_online_pruner_for_explore_first_arc`: actionable now. It should start
  as soft pruning with replay-disabled-on-failure so efficiency cannot hide a
  solve-rate regression.
- `equivpruner_state_action_cache_for_arc`: actionable now for exact state hashes
  and GAP-4-confirmed equivalence only. Approximate semantic equivalence should
  wait until false-negative replay logs exist.

---

## LOCAL-12B oracle-headroom code corpus

**Current Carnot substrate:** Track 1 should sit on the existing demo-fit code
verifier, sandbox, fixed candidate pool, and EvalPlus/LiveCodeBench eval path.
The measurement must report oracle headroom first: if the local 12B best-of-pool
or oracle is saturated on a corpus, the run is a non-measurement and should
route upward rather than claiming verifier transfer failed.

### 1. LiveCodeBench v6 as the hard headroom route

**Method:** LiveCodeBench is the right escalation corpus because it continuously
collects contest problems, supports time-windowed contamination control, and
includes execution-facing code scenarios. The upstream .376 planning sweep
records the key headroom claim: frontier leaderboard models are near the top of
the current LiveCodeBench v6 scale, while the local 12B pool should remain far
from that ceiling. The fresh pass confirmed the official benchmark/tooling and
current public leaderboard mirror still expose a large score spread.

**Implementation over Carnot stack:** Add a LiveCodeBench v6 adapter after the
EvalPlus adapter. It should consume the same fixed candidate hashes, use the
same sandbox execution ledger where possible, preserve release window and
scenario metadata, and support local model output import rather than forcing
generation inside the benchmark runner. Report vote, demo-fit code verifier,
ACES/SEP baselines, and oracle/best-of-pool on exactly the same task/candidate
set.

**Pitfalls / where it fails:** LiveCodeBench tasks are heavier and more variable
than HumanEval+/MBPP+. A small synchronous bounded batch is safer than another
background runner. If local 12B generation produces too few compilable
candidates, the bottleneck is generation quality rather than verifier transfer.

### 2. EvalPlus hidden rescore as the cheap first gate

**Method:** EvalPlus extends HumanEval and MBPP with much larger generated test
suites and shows that weak public tests miss wrong programs. It remains the
cheapest official hidden-test rescore path, even though the .375 smoke found no
local-12B headroom on its attempted subset.

**Implementation over Carnot stack:** Resume the fixed pool, normalize task ids,
candidate hashes, visible-test pass matrices, hidden EvalPlus outcomes, sandbox
timeouts, and bootstrap seeds. Use EvalPlus as the first pass because its task
shape is closest to the existing HumanEval/MBPP adapter.

**Pitfalls / where it fails:** If oracle headroom is absent again, stop and
route to LiveCodeBench v6. Do not treat another all-arms-tie as evidence against
the demo-fit verifier.

### 3. SAGA generated tests as a hidden-score tie-break

**Method:** SAGA reframes code verification as test generation and reports
stronger fault detection than existing code benchmark tests, including a
LiveCodeBench-v6 comparison. It is useful because the Carnot question is
whether verifier-style selection can distinguish candidates that visible tests
under-specify.

**Implementation over Carnot stack:** Run SAGA-style generated tests after the
official hidden-score path is frozen. Generated tests should live in a separate
`inference_time_generated` matrix, never mix with the official final score, and
only break ties or explain hidden failures.

**Pitfalls / where it fails:** Generated tests can be invalid, duplicated, or
biased toward known benchmark failures. If generated tests are tuned to mimic
hidden LiveCodeBench/EvalPlus distributions, the result becomes benchmark
overfitting rather than off-ARC transfer.

### 4. SEP bounded functional-equivalence diagnostic

**Method:** SEP partitions candidates into bounded symbolic equivalence classes
and reports improvements on HumanEval+ and LiveCodeBench without extra LLM
inference. It is the strongest semantic same-pool diagnostic for whether the
demo-fit verifier adds value beyond execution agreement.

**Implementation over Carnot stack:** Reuse the same sandbox pass matrix and
attempt bounded symbolic partitioning only for supported Python signatures.
Report SEP as a baseline/tie-break beside demo-fit, not as a replacement for
hidden execution scoring.

**Pitfalls / where it fails:** Symbolic execution can fail on mutation,
recursion, library-heavy code, floating point, and complex input parsers. Fail
closed and record unsupported reasons instead of silently dropping hard tasks.

### 5. ACES leave-one-out pass-matrix weighting

**Method:** ACES scores generated tests by whether their pass/fail pattern
distinguishes likely-correct from likely-incorrect candidates, avoiding the
simple "more tests passed is always better" trap. It is the cheap Arm A++ that
demo-fit should beat to claim unique transfer.

**Implementation over Carnot stack:** Build ACES directly from the visible and
generated-test pass matrices already produced by the sandbox. Compare plain
vote, ACES, SEP, demo-fit, and oracle on the same candidates.

**Pitfalls / where it fails:** ACES assumes enough candidate/test diversity to
estimate discriminative tests. Correlated wrong candidates can still dominate
the matrix, so hidden official score remains the only final authority.

---

## VERIFIER-GUIDED ONLINE ACTION-PRUNING

**Current Carnot substrate:** Track 2 should use the explore-first solver and
GAP-4 verifier as an online action-pruner, not a post-hoc scorer only. The
acceptance metric must pair efficiency with solve preservation: actions avoided,
frontier nodes avoided, GAP-4 calls spent, solves retained, and replay evidence
for any task that fails under pruning.

### 6. PRM-style adaptive online prune/expand control

**Method:** `2602.01070` makes the core online-verification move: verification
is a control signal during generation/search, not merely a final reranker. Its
PRM scores guide pruning and expansion inside a trajectory.

**Implementation over Carnot stack:** Wrap explore-first expansion in a
`gap4_prune_score(state, action, trace)` decision. Start with soft pruning:
rank or skip only dominated siblings, log every skipped action, and rerun with
pruning disabled if the pruned search fails.

**Pitfalls / where it fails:** Partial ARC states may not contain enough evidence
for a safe GAP-4 decision. A post-hoc accurate verifier can still prune the only
future-solution path online.

### 7. UF-OPS update-free verifier steering

**Method:** `2603.10282` trains verifier functions from policy rollout data and
uses them to steer action choices at execution time without changing base policy
parameters. This matches the `.376` goal: GAP-4 should steer action selection
online without retraining the explore-first solver.

**Implementation over Carnot stack:** Treat GAP-4 outputs and replay labels as
rollout-derived steering evidence. Add an execution-time action prior that
nudges explore-first toward higher verifier-likelihood actions while preserving
an abstain path and an unpruned fallback.

**Pitfalls / where it fails:** UF-OPS is robotics, not ARC. Its verifier learns
from rollout data; Carnot must first log enough ARC frontier/action outcomes to
avoid inventing a learned steering function from too little data.

### 8. Verification-cost-limited selective calls

**Method:** `2602.03975` studies verifier calls as the scarce resource and
combines deterministic feasibility gates, pre-verification scoring, and
uncertainty-aware allocation. This is directly useful because a GAP-4 call is
cheap but not free when invoked at every branch.

**Implementation over Carnot stack:** Put deterministic filters before GAP-4:
no-op deltas, repeated state hashes, impossible action arguments, and already
seen `(state, action)` pairs. Call GAP-4 only where cheap filters cannot decide
or where branch uncertainty is high.

**Pitfalls / where it fails:** Learned distance-to-goal and residual scoring do
not yet exist for ARC-AGI-3. The first version should log deterministic
features and verifier outcomes before training any scorer.

### 9. EquivPruner exact state/action cache

**Method:** EquivPruner removes semantically equivalent reasoning actions to
cut redundant search. ARC has an immediate exact version: multiple actions can
lead to the same state hash or the same GAP-4-confirmed effect signature.

**Implementation over Carnot stack:** Add a state/action equivalence cache next
to the frontier. For exact state hashes, keep one representative. For
GAP-4-confirmed equivalent effects, keep the cheaper or higher-ranked action and
record the pruned sibling as equivalent.

**Pitfalls / where it fails:** Approximate equivalence is dangerous in ARC
because visually similar states can diverge later. The `.376` implementation
should prune exact hashes and verifier-confirmed equivalence only.

### 10. CoT2-Meta controller for expand/prune/repair/stop/fallback

**Method:** CoT2-Meta turns test-time reasoning into an explicit controller over
expansion, pruning, repair, stopping, and fallback. Carnot already has these
behaviors informally; the missing piece is a logged controller state machine.

**Implementation over Carnot stack:** Add controller telemetry around
explore-first: each step records the chosen action class, GAP-4 evidence,
duplicate-state counts, search cost, and whether fallback was triggered. This
keeps pruning decisions auditable.

**Pitfalls / where it fails:** A controller can hide loss of coverage by falling
back too late or abstaining too often. Every run must pair pruned solve count
with unpruned replay solve count on the same tasks.

**Fresh adjacent non-headline method:** DIRECT (`2606.12402`) confirms that
test-time compute routing for embodied planning remains an active line and can
reduce latency, but it routes model/planning effort rather than verifier-guided
action pruning. It supports the budget-ledger framing, not a new first
implementation target.

---

## Bottom line for the .377 roadmap

1. **Strongest track-1 route:** implement `livecodebench_v6_local12b_headroom_route`
   as the escalation after EvalPlus. The `.376` run should resume the fixed
   pool, try EvalPlus first, and immediately route to LiveCodeBench v6 if
   oracle headroom is absent.
2. **Strongest track-2 route:** implement
   `gap4_soft_prune_replay_for_arc_efficiency` first. Soft pruning plus
   replay-disabled-on-failure is the only safe way to measure efficiency without
   hiding solve loss.
3. **Secondary but ready:** add
   `gap4_equivpruner_exact_state_action_cache` before learned steering. Exact
   cache pruning is low-risk and creates the telemetry needed for later UF-OPS
   style learned steering.
4. **Do not overclaim:** Track 1 must label another saturated hidden-test result
   as a measurement failure, not a transfer failure. Track 2 must report solve
   preservation beside every efficiency number.
