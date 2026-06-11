# Search-layer + decentralization literature survey (2026-06-11) — for the .372 pivot

**Provenance:** the `/deep-research` harness rate-limited 4× (server-side 429s under its 25–75-agent
fan-out; ~6M tokens, zero output). Gathered instead via direct low-concurrency WebSearch (which
worked cleanly) and synthesized by the outer-loop. Sources are search-surfaced; READ the primary
papers before implementation (literature-priority discipline). Maps to the three live `.372` forks:
exp4021 (search), exp4020 (goal induction), exp4022 (decentralization/distillation).

---

## 1. Search over an induced/verified world model — for exp4021 (THE central bet)

**Most directly applicable — our exact setup:**
- **"Learning Discrete World Models for Heuristic Search"** (RLC/RLJ 2024,
  rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_225.pdf) — learns a DISCRETE world model and runs
  heuristic search over it. This is structurally identical to ours (verified program model = the
  discrete transition function; run search over it). The closest literature match to exp4021.
- **"Policy-Guided Heuristic Search with Guarantees"** (AAAI, PHS*) — a learned POLICY guides A*/
  best-first search WITH solution-quality guarantees. The "guarantees" matter: our verifier gives
  exact legality, so a policy/heuristic-guided search over it can retain correctness while cutting
  branching. Strong candidate for the exp4021 search algorithm.
- **MuZero / MuDreamer** (2405.15083) — MCTS over a learned latent dynamics model. The canonical
  "search over a model" method; the lesson for us is the *separation* of model + search + value,
  not the latent representation (ours is symbolic/exact, which is an ADVANTAGE — no model-rollout
  hallucination).

**Sokoban-specific (our wall games are Sokoban-class):**
- **DRC recurrent planner** (arXiv:2407.15421, "Planning in a recurrent NN that plays Sokoban") —
  a 1.29M-param ConvLSTM "paces" to give itself extra compute, learns plan representations, and
  *generalizes to larger OOD Sokoban*. Evidence that learned planning transfers OOD — relevant to
  the modality-mismatch worry.
- **Hierarchical RL with landmarks** (arXiv:2504.04366) — deep recursive goal decomposition
  emerges from learning and scales to hard Sokoban; "substantially improved when combined with
  search." → **subgoal/landmark decomposition is the key tractability lever** for the long-horizon
  games where flat search blows up.

**What makes search tractable despite exponential branching (the actionable answer):**
1. A learned-or-coded HEURISTIC to order the frontier (A*/best-first). For us the heuristic can be
   **CODED** (distance over the exp4020 goal predicate: unmet goal components, manhattan-to-target,
   progress-bar delta) rather than learned — more OOD-robust (avoids the learned-heuristic OOD
   failure, the chief pitfall below).
2. **Subgoal/landmark decomposition** (hierarchical) to break the horizon — the single biggest
   lever for Sokoban-class.
3. Bounded node expansion + iterative deepening to stay tractable.

**Pitfall:** learned heuristics fail on OOD puzzle structure (the recurring finding). Mitigation:
prefer a coded/compositional heuristic over the verified model; if learned, validate OOD before
trusting. Our verifier-as-exact-simulator sidesteps the model-rollout-hallucination failure that
plagues latent world models — a structural advantage worth stating in exp4021.

## 1b. Planning under an imperfect (~99%) model — exp4021 robustness

- **"Investigating Compounding Prediction Errors in Learned Dynamics Models"** (arXiv:2203.09637)
  + the MPC literature: **replanning from each executed state (receding horizon, take only the
  first action) is the standard defense** — errors don't compound because you re-observe and
  re-plan. **Carnot ALREADY does this**: the L1→L3 verifier-validated re-induction (re-perceive →
  re-induce → re-validate → act per level) IS closed-loop MPC / re-induction-on-divergence. The
  literature validates our loop; exp4021 should make the MPC framing explicit and replan per step.
- Multi-step / hierarchical models reduce compounding error further (avoid single-step rollout
  chains). Our model is exact per-transition, so compounding is bounded to the 1% misprediction
  rate per step — replanning absorbs it.

## 1c. ARC-AGI-3 SOTA (the bar + the families) — context for the whole milestone

- **ARC-AGI-3 technical report** (arXiv:2603.24621; arcprize.org/arc-agi/3): interactive,
  >1000 levels / 150+ environments; agents must explore, infer goals on the fly, build world
  models, plan long-horizon. **Frontier LLMs score <1%; preview SOTA = 12.58% (Tufa Labs /
  "StochasticGoose")**; humans 100%. This quantifies our bar — 4 games / 5 levels is already
  meaningfully above the <1% frontier-LLM floor.
- **"Executable World Models for ARC-AGI-3 in the Era of Coding Agents"** (arXiv:2605.05138) — the
  Family-B SOTA, and it IS the Carnot thesis: a coding agent induces an executable Python
  transition model and verifies transitions. (Already in our memory as the thesis-validating
  paper; reported RHAE ~58% with GPT-5.5.) Our exp4021 extends exactly this with a search layer.
- **"Graph-Based Exploration for ARC-AGI-3"** (arXiv:2512.24156) — Family-A, no-induction directed
  exploration (the 3rd-place open-source baseline; our exp4004 explore-first is in this family).
- **Convergence: the two SOTA families are (B) induce+verify executable world models and (A)
  graph-explore — Carnot already runs both. The unsolved gap both leave open is exactly the
  planning/search layer .372 builds.**

## 2. Goal-predicate induction separated from dynamics — for exp4020

- **"Goal Inference as Inverse Planning"** + **Inverse Reward Design** (arXiv:1711.02827) — the
  classical framing: infer the goal/objective from observed (goal-reaching) behavior, treating it
  as a SEPARATE object from the transition model. Validates exp4020's core premise.
- **Eureka** (arXiv:2310.12931, "Human-Level Reward Design via Coding LLMs") — an LLM writes the
  reward/goal function AS CODE from context. This is **exactly exp4020's method**: codex writes
  `is_goal(state)` from the level-up transitions. Strong precedent that LLM-coded goal predicates
  work.
- The **hallucinator → synthesizer → executor** architecture (goals as logical predicates over
  final states; synthesize programs maximizing success probability) — a clean architecture for
  separating goal-spec from dynamics; maps to exp4020 (goal predicate) + exp4021 (search/synthesis
  to reach it) + the verifier (executor).
- **Actionable:** induce `is_goal()` as code (Eureka-style) from the env's own level_completed
  signal; the goal predicate becomes the search target for exp4021. The separation is well-precedented.

## 3. Decentralization via distillation — for exp4022 (branch B), with the critical caveat

- **Tulu 3** (arXiv:2411.15124) — the canonical OPEN post-training recipe (SFT → DPO → RLVR);
  the reference pipeline if we distill verifier-certified traces into a local model.
- **RLVR / generative verifiers** — verifiers distillable to **7B** with minimal degradation;
  RFT/STaR/ReST use verifier-certified positive traces as the training corpus. The verifier-as-
  automated-ground-truth-engine framing (Deep Think Q3) is well-supported: execution/unit-test
  reward is reliable and reward-hacking-resistant vs a learned reward model.
- **THE CRITICAL CAVEAT — "The Invisible Leash: Why RLVR May or May Not Escape Its Origin"**
  (arXiv:2507.14843): **RLVR sharpens what is already in the base model's support but may NOT
  create capability outside it.** This is the literature form of Deep Think's "representational
  deficit" claim AND it directly bounds exp4022: distillation/RLVR can close the 0.26→0.57 gap IF
  it's a SAMPLING/SHARPENING gap (the local model CAN induce the abstraction, just rarely), but
  NOT if it's a true REPRESENTATIONAL gap (the abstraction is absent from the local latent space).
- **Honest assessment for exp4022:** the decentralization branch hinges on which regime we're in —
  and exp4012's best-of-N number is the diagnostic. If best-of-N at high k DOES surface
  demo-perfect inductions (just rarely), the capability is latent → distillation/RLVR can sharpen
  it (viable sovereign path). If best-of-N yields only confident failures even at high k, the
  capability is absent → the invisible leash holds → distillation won't close it (Deep Think
  right; sovereignty needs a stronger base model, not more training on this one). **exp4022's
  branch-on-exp4012 design is exactly the right experiment.**

## 3b. Library learning / abstraction — for exp4020 + exp4025 (ArcMemo upgrade)

- **DreamCoder** (wake-sleep library learning) + **LILO/Stitch** (arXiv:2310.19791; LLM-guided
  synthesis + Stitch λ-abstraction compression + AutoDoc) — LILO beats DreamCoder on harder tasks
  with richer, linguistically-grounded libraries. **Carnot's ArcMemo IS a concept-library-learning
  instance**; LILO's compression + auto-documentation is the concrete upgrade path for ArcMemo v5
  (exp4025): compress recurring induced-program fragments into named, documented abstractions that
  seed future induction.
- **"Neural-guided Bidirectional Program Search for Abstraction and Reasoning"** (arXiv:2110.11536)
  — bidirectional (forward-induce + backward-from-goal) search for ARC; relevant to combining the
  exp4021 forward search with goal-directed backward reasoning.

---

## Bottom line for the .372 roadmap

- **exp4021 (search):** the literature strongly supports the design. Use **policy/heuristic-guided
  search over the verified discrete model (PHS*-style) + subgoal/landmark hierarchical
  decomposition + per-step MPC replanning**. Prefer a CODED heuristic over the goal predicate
  (OOD-robust) over a learned one. Our exact-verifier-as-simulator is a structural advantage
  (no rollout hallucination). Closest reference: Learning Discrete World Models for Heuristic
  Search (RLC 2024).
- **exp4020 (goal induction):** Eureka-style LLM-coded `is_goal()` from level_completed signals is
  well-precedented (goal-inference-as-inverse-planning lineage). Sound.
- **exp4022 (decentralization):** the "Invisible Leash" result is the decisive lens — the
  0.26→0.57 gap is closable by distillation ONLY if it's a sharpening gap, not a representational
  one. exp4012's best-of-N-at-high-k is the diagnostic; the branch-on-exp4012 design is correct.
  Tulu 3 + RLVR is the recipe if branch-B fires.
- **exp4025 (ArcMemo):** LILO/Stitch compression + auto-doc is the concrete upgrade for concept
  memory.

**Primary papers to read in depth (priority order):** 2605.05138 (Executable World Models ARC-3),
RLC-2024-225 (Discrete World Models for Heuristic Search), 2507.14843 (Invisible Leash — the
decentralization decider), 2504.04366 (hierarchical landmarks Sokoban), 2310.12931 (Eureka),
2310.19791 (LILO).
