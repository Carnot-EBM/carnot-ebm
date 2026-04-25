# Closed-weight frontier-LLM integration: teacher-forced proxy + tool use

**Status:** Draft change proposal.
**Origin:** External architecture review (Google Deep Think, 2026-04-25)
  on integrating Carnot with closed-weight frontier models like Claude
  / GPT / Gemini, where residual-stream hooking is impossible. The
  review proposed four operational paths; this proposal frames them as
  a "closed-weight frontier integration" track that pairs cleanly with
  the open-weight v2 proposal (commit a1d0a338,
  `carnot-v2-langevin-act-rdt-architecture.md`).
**Target milestone:** 2026.05.NN+ — concurrent with or after the v2
  proposal, after the .65/.66 layered diagnostic chain lands.
**Priority:** Medium-high. The closed-weight path is the larger
  near-term commercial surface (most enterprise Carnot consumers will
  have API access only), but the technical risk is higher than the
  open-weight path because cross-vendor transferability is *known to
  fail* per Cognometry's finding (cosine 0.043 across vendors).
**Depends on:**
  - Existing per-token EBMs on Carnot-EBM HuggingFace
    (`Carnot-EBM/per-token-ebm-qwen35-2b-nothink`,
    `Carnot-EBM/per-token-ebm-gemma4-e4b-it-nothink`).
  - Existing MCP server (`python/carnot/mcp/server.py`) for path C.
  - Cognometry reference (`research-references.md`) for the
    transferability prior on path A.
  - `garak-red-team-integration.md`'s 5-vector poisoning
    countermeasures for path D — strict prerequisite, not optional.

## Summary

Carnot's existing per-token EBMs are trained on the internal
activations of *specific open models* (Qwen3.5-0.8B, Gemma4-E4B).
When the consumer's generator is closed-weight (Claude, GPT, Gemini),
those activations are unreachable. Four paths bridge the gap, ordered
by technical maturity and risk:

- **Path A — teacher-forced proxy energy scoring** (foundational).
  Force a local proxy Qwen to *process* the closed-weight model's
  token sequence (not generate freely), extract activations at each
  forced token, score via the existing per-token EBM. Produces a
  per-token energy map of the closed-weight output without ever
  touching its weights.
- **Path B — energy-guided backtracking on streaming API** (depends
  on A). As the closed-weight model streams, score via path A. On
  energy spike, truncate the API history and prompt with a
  "structural anomaly at step N" instruction.
- **Path C — Claude self-routing to Carnot via Tool Use** (extends
  existing MCP). System-prompt-engineer the closed-weight model to
  recognise constraint-satisfaction subtasks and delegate to the
  existing `verify_and_repair` MCP tool. Most of the plumbing exists;
  the novelty is the dispatch-side instruction.
- **Path D — Claude-as-oracle contrastive dataset generation**
  (highest-risk, last). Automate the closed-weight model to produce
  paired clean/subtly-flawed reasoning traces, encode via path A,
  build a contrastive dataset for the next-gen Carnot EBM. Real
  poisoning surface. Strict prerequisite: dogfood-safeguard track
  shipped first.

Each path has falsifiable acceptance gates. Path A is the foundational
primitive — if cross-vendor activations don't transfer (per Cognometry's
prior), paths B and D collapse and only path C survives.

## What this proposal IS NOT

- **Not a replacement for the v2 open-weight architecture proposal**
  (Langevin / ACT-RDT). The two are complementary: v2 is what to do
  when you *can* hook the residual stream; this is what to do when
  you can't. A mature Carnot consumer should run both, picking the
  path based on the LLM's openness.
- **Not a recommendation to train Carnot models on Claude output as
  a default**. Path D is genuinely high-risk and is gated explicitly
  on the dogfood-safeguard track having shipped, with full provenance
  labelling on any resulting trained model.
- **Not for the generative-time safety gate** (issue #2 `budget_ms`).
  Teacher-forcing every token through a proxy doubles inference
  latency at minimum. Same exclusion as v2.

## Proposed experiments

### Exp A — Teacher-forced proxy energy scoring (the foundational primitive)

**Deliverable:**
`python/carnot/closed_weight/teacher_forced_proxy.py` +
`scripts/experiment_<N>_teacher_forced_proxy.py` +
`tests/python/test_teacher_forced_proxy.py` +
`results/experiment_<N>_teacher_forced_proxy.json`.

**What it does:**

1. Build a `TeacherForcedProxy` class that loads a local Qwen3.5-0.8B
   in proxy mode: takes a prompt P and a target token sequence
   T = (t_1, ..., t_n), runs the forward pass with each t_i
   force-fed (no sampling, no generation), and returns the
   residual-stream activation at each step. This is the standard
   teacher-forcing recipe — no novel ML, just plumbing on top of
   the existing loader.
2. Score each per-token activation via the existing per-token Qwen
   EBM (`Carnot-EBM/per-token-ebm-qwen35-2b-nothink`). Returns a
   length-n vector of per-token energies.
3. Build a labelled corpus of 200 reasoning traces, half clean and
   half with known structural flaws (arithmetic, logic, code
   execution). Use multiple LLMs as the trace generator: GPT-4o,
   Claude 3.7, Gemini 2.5, plus our SOTA local models for
   in-distribution control.
4. For each trace, compute the per-token energy, then test whether
   energy at the *flawed* token is statistically separable from
   energies at clean tokens.

**Acceptance gates:**

1. **In-distribution sanity**: on traces from local Qwen3.5-0.8B
   (the proxy's own native model), the per-token EBM produces
   AUROC ≥ 0.85 separating flawed-token energies from clean-token
   energies. Verifies the EBM works on its native distribution.
2. **Within-family transferability**: on traces from
   Qwen3.6-35B-A3B (same family, larger), AUROC drop is ≤ 10
   percentage points relative to (1). The Cognometry within-family
   cosine of 0.464 predicts moderate transfer; this gate quantifies
   it for our setup.
3. **Cross-vendor transferability** (the load-bearing gate): on
   traces from Claude / GPT / Gemini, AUROC stays at or above 0.65.
   If AUROC ≤ 0.55 (random), Cognometry's prior holds and the proxy
   approach **does not work** for cross-vendor — paths B and D both
   collapse, and only path C (Tool Use) survives. This is the
   honest gate that determines whether the rest of this proposal is
   buildable.
4. **Honest-verdict enum**: `proxy_in_dist_strong_within_family_strong_cross_vendor_strong`,
   `proxy_in_dist_strong_within_family_strong_cross_vendor_weak`,
   `proxy_in_dist_strong_within_family_weak_cross_vendor_collapsed`,
   `proxy_in_dist_failed`,
   `proxy_loader_unimplementable`.

This experiment is *the* gate for the whole proposal. If gate (3)
fails, paths B and D are abandoned, and we ship only path C.

### Exp B — Energy-guided backtracking on streaming Claude API (gated on A)

**Deliverable:**
`python/carnot/closed_weight/streaming_backtrack.py` +
`scripts/experiment_<N>_streaming_backtrack.py` +
`tests/python/test_streaming_backtrack.py` +
`results/experiment_<N>_streaming_backtrack.json`.

**What it does:**

1. Wrap the Anthropic streaming API (the
   `messages.stream()` SSE path). At each token-batch boundary,
   run path A's teacher-forced proxy + per-token EBM on the tokens
   accumulated so far. Maintain a sliding-window energy estimate.
2. On a confirmed energy spike (delta over a tunable threshold, with
   a small lookahead to avoid noise), abort the API stream, truncate
   the conversation history to before the spike, and resume with a
   targeted prompt: "A structural logic anomaly was detected at
   the step ending '{snippet}'. Review your constraints and provide
   an alternative reasoning path."
3. Evaluate on a labelled corpus of 50 reasoning traces with
   *injected* errors at known token positions (synthetic, not real
   adversarial inputs).

**Acceptance gates:**

1. **Detection**: 80% of injected errors trigger an energy spike
   that exceeds the tunable threshold within 10 tokens of the actual
   error position.
2. **Backtrack-and-recover**: in 50% of detected cases, Claude
   produces a corrected continuation that passes Z3 / Hypothesis
   verification at the trace tail. (Lower bound: Claude's own
   self-correction rate on flawed traces is the empirical ceiling
   here, so we set 50% as "the wrapper roughly halves the
   hallucination rate" — better is ideal, this is the floor.)
3. **No regression on clean traces**: on the clean half of the
   labelled corpus, the wrapper triggers a backtrack in ≤ 5% of
   cases (false-positive rate). Backtracks are expensive (extra API
   calls and latency); we cannot tolerate frequent false alarms.
4. **Honest-verdict enum**: `backtrack_detection_and_recovery_both_above_gate`,
   `backtrack_detection_above_recovery_below`,
   `backtrack_detection_below_gate`,
   `backtrack_false_positive_rate_above_budget`.

### Exp C — Claude self-routing to Carnot via Tool Use (extends MCP)

**Deliverable:**
`python/carnot/closed_weight/tool_use_router.py` +
updated `python/carnot/mcp/server.py` to expose a
`verify_constraint_subproblem` tool with structured input schema +
system prompt template at
`docs/closed-weight-system-prompts/claude-router-prompt.md` +
`results/experiment_<N>_tool_use_router.json`.

**What it does:**

1. Define a structured tool schema for Claude's Tool Use API:
   `verify_constraint_subproblem({constraint_text, constraint_type:
   "arithmetic" | "code_executable" | "logical_satisfiability" |
   "type_assertion", expected_property})`.
2. Author a system prompt template that teaches Claude (a) when a
   subtask is constraint-satisfiable enough to be worth delegating,
   (b) how to formulate the structured tool call, (c) what to do
   with the verified answer (incorporate as ground truth, don't
   overwrite with its own guess).
3. Run on a benchmark of 100 reasoning traces where the dominant
   bottleneck is constraint satisfaction (math word problems,
   scheduling, type checking, formal-spec equivalence).
4. Compare against three baselines: bare Claude, Claude + manual
   tool-use prompts (no Carnot system prompt), bare Carnot pipeline
   on the same traces.

**Acceptance gates:**

1. **Delegation rate**: Claude calls the tool on ≥ 70% of traces
   the system prompt classifies as constraint-satisfiable. The
   prompt should *teach the routing*, not require manual annotation.
2. **Quality lift**: accuracy on the 100-trace benchmark exceeds
   bare Claude by ≥ +5 percentage points and exceeds Carnot-alone
   by ≥ +10 percentage points (the lift comes from Claude's
   world-knowledge plus Carnot's structural verification, neither
   alone).
3. **Latency budget**: median end-to-end latency ≤ 2x bare Claude
   for the constraint-satisfiable subset. Tool calls add latency;
   we cap the cost at 2x the unverified path.
4. **Honest-verdict enum**: `tool_use_router_lift_above_gate`,
   `tool_use_router_delegation_below_gate`,
   `tool_use_router_quality_lift_below_gate`,
   `tool_use_router_latency_above_budget`.

This experiment extends the existing MCP path; the marginal new code
is the system-prompt template + the tool schema. Lowest-risk of the
four experiments.

### Exp D — Claude-as-oracle contrastive dataset generation (highest-risk, gated on full safeguard track)

**Deliverable:**
`python/carnot/closed_weight/oracle_corpus.py` +
`scripts/experiment_<N>_oracle_corpus_audit.py` +
`results/experiment_<N>_oracle_corpus_audit.json` +
explicit provenance metadata in any resulting trained model card on
HuggingFace.

**What it does:**

1. Automate the closed-weight model to generate paired traces:
   given a problem statement, produce one canonically-correct
   reasoning trace and one trace with a single-step structural
   flaw at a known position. Repeat for ~10K problems across
   arithmetic / code / logic domains.
2. Encode each trace via path A (teacher-forced proxy + activation
   extraction), label with "clean" / "flawed_at_step_N" metadata.
3. Use the resulting (activation, label) pairs to train a next-gen
   per-token EBM with sharper structural-flaw discrimination.

**Strict prerequisites** (not waivable for this experiment):

- The dogfood-safeguard track must be shipped (3 proposals:
  `conductor-self-protection-safeguard`, `generative-time-safety-gate`,
  `garak-red-team-integration`). The Garak proposal's
  poisoning-countermeasures section already covers the
  `training_eligible:false` discipline that this experiment must
  inherit.
- The oracle corpus is quarantined under
  `data/oracle_corpus/<source>/<date>/` with a manifest recording
  source LLM, generation prompt, generation date, and a hash of
  every trace.
- The corpus passes a sanity-check audit before being used for
  training: a held-out random sample of 100 paired traces is
  manually reviewed for genuine flaws (vs the closed-weight model
  generating fake/uninformative flaws to satisfy the prompt).
- Any model trained on this corpus has its training-data
  provenance published in the HuggingFace model card under a clear
  "Trained partly on synthetic Claude-generated data" warning.

**Acceptance gates:**

1. **Corpus quality**: the manual audit on the 100-sample held-out
   slice confirms ≥ 80% of paired traces have a genuine,
   diagnosis-worthy flaw at the labelled position. If lower, the
   corpus is rejected — Claude was generating fake flaws.
2. **Training lift**: the per-token EBM trained on the audited
   corpus + our existing in-distribution data outperforms the
   baseline (in-distribution-only) by AUROC ≥ +5 percentage points
   on the held-out flaw-detection benchmark.
3. **No distribution collapse**: the trained EBM does not regress
   on Carnot-Bench or any existing benchmark by more than -1
   percentage point. Training on synthetic data must not cost
   in-distribution performance.
4. **Honest-verdict enum**: `oracle_corpus_clean_training_lift_no_regression`,
   `oracle_corpus_clean_training_lift_with_regression`,
   `oracle_corpus_quality_below_audit_gate`,
   `oracle_corpus_training_lift_below_gate`,
   `oracle_corpus_blocked_safeguard_track_not_shipped`.

This experiment is the highest-risk and last in priority. Path A
must succeed first. The dogfood-safeguard track must be shipped
first. Even then, the resulting trained model is labelled
"synthetic-data-augmented" indefinitely.

## Risks and honest concerns

- **Cross-vendor transferability is the load-bearing assumption**.
  Cognometry's measured cosine 0.043 across vendors is the prior
  *against* path A working. If gate A.3 fails, only path C survives
  — and even path C is just a system-prompt + Tool Use wrapper, not
  a deep Carnot integration. This proposal honestly contemplates
  the failure case.
- **Self-correction is not free**. Path B's "Claude self-corrects
  on spatial pointers" assumes Claude can localise its own
  structurally invalid reasoning. Empirically, Claude often
  re-justifies invalid reasoning rather than fixing it. Gate B.2
  (50% recovery rate) is the floor — anything lower means the
  primitive is detection-only, not a verifier-corrector.
- **Tool-use latency adds up**. Each `verify_constraint_subproblem`
  call is a full Carnot pipeline run (claim extraction + Z3 /
  Hypothesis / EBM + repair if needed). At 100ms-1s per call, an
  agent that delegates aggressively can quickly exceed
  human-acceptable latency. Gate C.3 caps it at 2x bare Claude.
- **Path D's poisoning surface**. Training on closed-weight
  synthetic data inherits the closed-weight model's biases,
  blindspots, and any subtle adversarial patterns. The Garak
  proposal's countermeasures are necessary but probably not
  sufficient — long-term, Carnot's training data should be
  dominated by real reasoning traces with mechanically-verifiable
  ground truth (Z3 / Hypothesis output), not LLM-generated traces.
  Path D is a stopgap, not a strategy.
- **Tie-in with `recursive-extractor-and-verify-stream-alignment.md`
  (commit 643a21ad)**: that proposal targets long-context
  decomposition. Path C here is the *short*-context delegation
  primitive. They compose — RLM decomposes, then per-snippet calls
  flow through the Tool Use router. The combined surface is
  larger than either alone.

## Tie-ins to other drafted proposals

- **v2 open-weight proposal** (`a1d0a338`): pair this with that.
  Open-weight gets Langevin/ACT/RDT internally. Closed-weight gets
  teacher-forced proxy + Tool Use externally. A mature Carnot
  consumer runs both paths and picks based on the model's openness.
- **Dogfood-safeguard track** (3 proposals queued for 3 milestones
  now): path D depends on this. Must ship first.
- **Cognometry reference** (`research-references.md`): the
  transferability finding is the prior that gates path A.
- **MCP server** (`python/carnot/mcp/server.py`): path C extends it.
  Adding the `verify_constraint_subproblem` tool is mostly schema
  work + the system-prompt template.
- **Phase 3 capability spec** (`openspec/capabilities/phase3-kona/spec.md`):
  closed-weight integration is *not* on the Phase 3 critical path
  (Phase 3 is foundation-model construction, which is open-weight by
  definition). But path A's per-token energy mapping is a useful
  validation tool for Phase 3 generation quality.
