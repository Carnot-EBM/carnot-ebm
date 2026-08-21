# AVO adaptation for a local generator — 2026-08-21

This note records the AVO research findings, the parts that carry over to
this project's weak local generator, and the build decision. The operator
referred to the system as "EVO". The correct name is **AVO — Agentic
Variation Operators** (NVIDIA, arXiv 2603.24517, March 2026).

## Part 1 — what the research established

Primary sources:

- Paper: "AVO: Agentic Variation Operators for Autonomous Evolutionary
  Search", arXiv 2603.24517 (https://arxiv.org/abs/2603.24517).
- NVIDIA developer blog: "NVIDIA AVO Reaches 100% on ARC-AGI-3"
  (developer.nvidia.com/blog/nvidia-avo-reaches-100-on-arc-agi-3-...).
- NVIDIA AI on X, status 2090786258981466231 (the 183-level claim).
- Hacker News item 49387755 (critical discussion).
- Secondary: wccftech, thenewstack, wispaper (add no new facts).

### Confirmed: the CUDA kernel result

- The agent IS the variation operator. The paper replaces the classical
  evolutionary step `Vary(P) = Generate(Sample(P))` with
  `Vary(P) = Agent(P, K, f)`. The agent chooses which prior versions to
  study, which documents to read (CUDA guides, PTX ISA, Blackwell specs,
  FlashAttention-4 source), which lines to edit, when to benchmark, and
  when to commit.
- 7 days of continuous autonomous evolution on multi-head attention on a
  Blackwell B200. The agent explored 500+ optimization directions and
  committed 40 kernel versions. Each version persists as a git commit
  with its score.
- Results: up to +3.5% over cuDNN and up to +10.5% over FlashAttention-4
  (BF16, peak 1668 TFLOPS). The gains transfer to grouped-query
  attention after 30 minutes of autonomous adaptation.
- Fitness is mechanical: a binary correctness gate, then throughput in
  TFLOPS. A kernel that fails correctness scores zero.
- A SUPERVISOR agent watches for two failure modes: stalling after
  exhausted exploration, and unproductive repeat cycles. On either, it
  reviews the whole trajectory and steers the search toward several
  candidate directions.

### Confirmed: the ARC-AGI-3 result

- AVO scored 100.00 RHAE on the ARC-AGI-3 PUBLIC set: all 183 levels
  across all 25 environments, with no instructions, rules, or stated
  goals. RHAE (Relative Human Action Efficiency) weights completion by
  per-level action efficiency against human baselines.
- Bare Claude Opus 5 scores about 30% on the same benchmark. The lift
  from 30% to 100% is the harness contribution NVIDIA claims.
- Action count: 6,624 environment actions, versus 7,542 for VISTA
  (a cross-system comparison, about 12% fewer).
- Base model: Claude Opus 5 for the full run. GPT-5.6 Sol appears only
  in preliminary side experiments.

### Refuted or corrected

1. **The ARC gain did NOT come from the evolutionary variation-operator
   loop.** For ARC, the blog states the agent built no explicit world
   models and issued actions directly: "form a hypothesis, act, observe
   evidence, update state, and continue", over text-only 64x64 grids.
   The components NVIDIA credits for ARC are: persistent memory,
   supervision, and long-horizon scaffolding — wrapped around a frontier
   reasoner. The agent-as-mutation-operator design is the CUDA-kernel
   mechanism, not the ARC mechanism. Any adaptation that copies the
   evolutionary loop "because of the ARC score" copies the wrong part.
2. **Public set only.** NVIDIA states the results "are not results on
   the semi-private or fully private competition sets". Our registry
   also shows 183/183 on the public set — reached with 25 hand-built
   per-game adapters. AVO reached it with one general loop. The gap is
   generality, not score.
3. **No component ablations.** The paper and blog publish no ablation
   isolating supervisor vs memory vs model. The "the model is not the
   entire agent" attribution is plausible but unmeasured. No token,
   cost, or wall-clock figures exist for the ARC run (HN raised this).
4. Related mechanism papers share one thread: mechanical ground-truth
   feedback substitutes for model expertise. KernelPro (arXiv
   2606.26453) packages profiler data as pluggable tools. The CUDA-RL
   agent (arXiv 2602.24286) uses verification plus profiling as reward.
   CudaForge (arXiv 2511.01884) iterates a Coder and a Judge on Nsight
   metrics for $0.30 per kernel. Expert knowledge lives in tools and
   feedback, not in the model.

## Part 2 — the carries-over / does-not-carry-over split

The constraint: AVO wraps Claude Opus 5. Our scored path runs
Qwen3.8-27B offline with no internet. A design that silently assumes
frontier-model reasoning fails here. The split:

### Carries over to a weak local generator

1. **Mechanical fitness as the driver.** AVO's correctness gate plus
   throughput score is model-free. Our analogue already exists: the
   WorldModelVerifier trust gate, offline replay, and live level
   progress. The weaker the generator, the more the harness must lean
   on cheap mechanical scoring to filter noisy proposals. This part
   transfers and is already built.
2. **Persistent lineage with receipts.** AVO persists every committed
   version with its score. Our engine store and induction attempt rows
   are the analogue. Partially built.
3. **Supervision — with a degraded redirect step.** AVO's stagnation
   DETECTION is model-free: it reads trajectory statistics. That
   transfers unchanged. AVO's REDIRECT step is open-ended frontier
   re-planning. That does not transfer. The honest weak-generator
   adaptation is a closed decision table over levers the agent already
   has, applied in a fixed order, bounded, with receipts. This is the
   piece our live path lacks (see Part 3).
4. **Knowledge as tools, not weights** (the KernelPro thread). The
   induction tool loop's retrieval half (REQ-ARC-WMTE-6540 line of
   work) already applies this: the model fetches evidence instead of
   carrying it in context.

### Does not carry over

1. **Per-action frontier reasoning over raw grids.** Opus 5 reads a
   64x64 text grid and forms hypotheses per action. Qwen3.8-27B cannot
   sustain that depth, and per-action LLM calls do not fit the scored
   latency budget (vLLM concurrency is capped at 8; induction already
   presses the scored timeout). Our inversion — classical
   verifier-routed exploration with the LLM only at the induction
   tier — is the correct adaptation, not a compromise.
2. **Open-ended strategy generation by the supervisor.** A weak model
   asked to invent a new strategy produces noise the run cannot
   validate. Enumerate and select instead.
3. **Unbounded budgets.** 7 days and 500 directions have no scored
   analogue. Every adapted mechanism must be bounded per level and per
   run.
4. **Runtime document reading.** The scored run has no internet, and
   reading hidden-game source is forbidden by standing rule.

## Part 3 — build decision

**Chosen: a trajectory supervisor on the live path
(REQ-ARC-WMTE-6600, default OFF).**

The hole it fills, measured in `arc_competition_agent.py`: after the
one-shot induction latch fires and the plan exhausts without a
level-up, the cascade returns to `explore` and never changes strategy
again for that level. The explorer's stall-diversity draw
(`CARNOT_ARC_EXPLORE_STALL`) is a move-level response inside one search
policy. Bounded reinduction exists but is a stall-time lever, default
off. Nothing observes the trajectory as a whole and redirects. The
conductor gained park-and-escalate on 2026-08-21; the live agent has no
counterpart, and AVO's supervisor is precisely that counterpart.

Why not the evolutionary loop over induced world models:

1. The research says the evolutionary operator is not what produced
   AVO's ARC result (Part 1, correction 1).
2. Our closest weak-generator analogue already exists and is in active
   development this week: recall-gated resample plus the repair-mode
   tool loop with a monotone seed floor (REQ-ARC-WMTE-6410/6470) is a
   bounded lineage of engine variants under mechanical selection.
   A population loop would duplicate it and multiply scored-path LLM
   calls we cannot afford.
3. The supervisor is fully testable without Kaggle and without any LLM
   call: detection and redirection are deterministic Python.

### Design

New module `python/carnot/agentic/arc_trajectory_supervisor.py`,
wired into `E3AgentPolicy` behind `CARNOT_ARC_TRAJECTORY_SUPERVISOR=1`
(default off — no scored-path default flips).

- **Detection.** The supervisor observes one snapshot per action:
  current level, induction latch, induction attempt count, transitions
  since the last induction attempt, goal-bias state, and diversity
  state. It counts actions since the last level-up or redirect. When
  the count reaches the window (default 400 actions), the trajectory
  is stagnant.
- **Redirect.** On stagnation it fires the FIRST eligible arm not yet
  used on this level, then starts a fresh window:
  1. `drop_goal_bias` — a goal bias is installed and a full window
     passed without progress under it. The bias is steering and not
     working; a degenerate or inverted induced goal traps the frontier.
     Applied via the existing `set_goal_bias(None)` seam.
  2. `allow_reinduction` — the induction latch is set, at least 200 new
     transitions accumulated since the last attempt, and the per-level
     attempt cap (3) is not reached. Applied by resetting the latch,
     which re-enters the existing induction path with genuinely new
     evidence.
  3. `force_exploration_diversity` — switch the explorer to its
     stall-diversity draw (randomized top-k pop instead of the
     deterministic head). Applied via the existing hybrid-diversity
     seam.
- **Bounds.** Each arm fires at most once per level. A level-up resets
  the arms and counters. The supervisor makes no LLM call.
- **Receipts.** Every redirect records action index, level, arm, and
  diagnosis. `trajectory_supervisor_diagnostics()` exposes the ledger
  so run artifacts carry the evidence (the same receipts doctrine as
  the 2026-08-21 conductor note).

### What stays blocked on Kaggle quota

- Any measurement on the Blackwell / NVFP4 / vLLM scored backend,
  including the `qwen3_xml` tool-call parser trial.
- A live scored A/B of the supervisor. The staged follow-on is an
  LLM-off offline A/B on public games with per-game adapters disabled
  (the generalization-floor shape), on local GPU 1, which exercises
  arms 1 and 3; arm 2 needs a live generator.
