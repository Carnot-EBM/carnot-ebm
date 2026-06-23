# SOTA ingestion: MATM (Multi-Agent Transactive Memory) — arXiv:2606.19911

**Ingested 2026-06-23 (outer-loop, operator-requested).** Method: read → map-onto-Carnot-tracks
→ adversarial-verify-each-mapping → synthesize (10-agent workflow `matm-ingestion`). All repo
claims below were spot-verified against tracked files (not hallucinated).

## What MATM is

Kim, He, Jain, Agrawal, Arabzadeh, Diaz — submitted 2026-06-18. A **population-level retrieval
system for agent-generated trajectories**: RAG generalized from human artifacts to agent
procedural knowledge. Mechanics:

- **Granularity:** fixed `l=5`-step action–observation **sub-trajectory chunks**, not full
  trajectories. Key = embedding of `[task description x + the last l recent steps]` (i.e. task +
  current partial-trajectory context). Value = the *next* `l` steps (a continuation snippet).
- **Embedding:** frozen shared E5-Base. First-stage top-20 by cosine, then a learned
  Learning-to-Rank reranker (FFN / LambdaMART / SVMRank over 44 features) → top-1, injected into
  the prompt as an in-context "Reference Example."
- **Quality control = success-threshold filtering, NOT verification.** A trajectory enters the
  buffer only if its terminal eval score `s ≥ θ`. The reranker is trained on **marginal-utility**
  labels `ℓ = s_t − s_base` (does injecting the chunk improve the consumer's outcome vs
  no-retrieval). There is **no per-step / invariant quality check**, no repository dedup/curation,
  and **no online feedback loop** from consumer outcomes back into the repo or reranker (the LTR
  head is trained once offline; closing the two-sided market is future work).
- **Heterogeneous population:** ~34 consumer / ~35 producer agents across GPT-4-Turbo,
  Claude-3.5, Qwen3-32B, Llama-3.3-70B, Gemini-2.5, DeepSeek.

**Results (the one transferable claim — step reduction without joint training):**
ALFWorld success 47.08% → 55.11% single-stage (+8.0pp), → **64.31%** with SVMRank reranker
(+17.2pp); steps 11.77 → 10.35. WebArena 18.18% → 20.45% (+2.3pp); steps 22.0 → 19.9.
Cross-agent retrieval works; producer–consumer capability gap barely correlates with gain
(r=+0.04). No reported case of retrieval dropping below baseline. **No k-ablation.** Within-
benchmark retrieval only (no cross-task/cross-domain transfer claim).

## Map onto Carnot — honest verdict (adversarially verified)

### SURVIVES (KEEP_NARROW) — ARC similarity-keyed partial-trajectory retrieval
Carnot's live ARC agent **already** does within-game, self-populated, trajectory-level retrieval
(`StepwiseExplorer.adj` + `_shortest_path` / `_partial_forward_path`, scored by
`navigation_diagnostics.forward_walk_hit_rate`, in `arc_competition_agent.py` /
`arc_graph_explore.py`). So MATM does **not** supply per-run trajectory memory — that exists.

The **one un-subsumed slice**: MATM keys retrieval by a **similarity** embedding of the recent
state, where Carnot keys by **exact frame-hash**. A coarse/LSH state descriptor would let a
*near-match* state inherit a useful action prefix from that hidden game's own earlier rollouts —
a strict generalization of the existing `hud_mask` exact-hash relaxation. Grep confirms no
LSH/similarity-trajectory index exists in the live `arc_*` modules. Ships flag-gated inside
`StepwiseExplorer` (live-by-construction; passes `arc_orphan_solver_lint`).

The energy/EBM role here is **oracle-distinct**: a retrieved prefix is a *candidate*, scored by
the existing learned cross-game value/trust router (`value_head` / `goal_bias` /
`WorldModelVerifier` trust-energy) before the agent commits actions to it. The verifier is the
learned value head, **not** the executable win-check → `verifier_is_oracle: false`. Energy as
router-not-generator: MATM proposes recombined prefixes; the verifier prunes/ranks them.

### KILL_SUBSUMED — energy-verifier as the cross-game retrieval quality gate
Tempting (MATM has no real quality gate; we have an energy verifier), but the graft sits
**downstream of what is actually broken.** Carnot ran the cross-game transfer experiment three
times and hit a triply-replicated null: exp4318 (`cross_game_state_reduction=1.0`), exp4331,
exp4342 — all positive-control-passed, all `transfer_helps: false`, root-caused in-artifact as a
**representation/encoding gap** (a game-invariant ARC value representation is missing), *upstream*
of any scoring gate. You cannot usefully gate a representation that does not transfer. Corroborated
by the 2026-06-23 perception-is-the-binding-constraint finding.

### KILL_SUBSUMED — MATM-style trajectory memory for the conductor's own task-agents
MATM's literal motivating problem ("newly instantiated agents repeatedly rediscover existing
solutions") is real for the conductor, but already owned twice: **FinAcumen (arXiv:2606.17642,
ingested 2026-06-19)** grafted semantic experience-memory retrieval + dedup/rank/`k_max` into
`recommend_approach` (a near-exact structural mirror of MATM, carrying the precise "irrelevant
retrieval DEGRADES — precision>recall" reranker lesson); **Self-Harness (arXiv:2606.09498)** owns
the conductor-self-improvement half (mine failure-signatures → regression-gated harness edits — and
adds a regression gate MATM lacks). MATM adds no new mechanism here.

## Bottom line for the roadmap

One flagged candidate (ARC live action-efficiency — fits the submission-sprint forcing function;
does not displace a level-up attempt):

```yaml
- id: expNNNN-arc-similarity-trajectory-retrieval
  agent_type: codex
  model: gpt-5.5
  title: "StepwiseExplorer similarity-keyed partial-trajectory retrieval (MATM-graft), verifier-routed"
  inference_substrate: verifier_ensemble_against_cached_candidates
  scope: >
    Flag-gated coarse/LSH state-descriptor index on StepwiseExplorer.adj so _shortest_path returns
    an action prefix from a SIMILAR (not bit-identical) prior state; each retrieved prefix scored by
    the existing value_head/goal_bias/WorldModelVerifier router before commit. A/B vs the SUBMITTED
    exact-hash baseline over reproduced games (tu93, lp85, sp80, cn04, m0r0) via
    scripts/arc3_replay_scorecard_metaharness.py.
  required_artifact_fields:
    verifier_is_oracle:
      value: false
      principle: "Retrieved prefixes scored by the LEARNED cross-game value/trust head, not the
                  executable win-check; oracle-distinct per the Circularity discipline."
    forward_walk_hit_rate_delta:
      principle: "Strict increase vs exact-hash adj is the necessary condition that similarity
                  retrieval surfaces a usable prefix at all."
    actions_to_first_levelup_delta:
      principle: "The live action-efficiency metric MATM's step-reduction claim targets."
    offline_reproduced:
      principle: "Only registry-reproduced games count; zero regression on reached_level."
  acceptance_gate:
    condition: "forward_walk_hit_rate strictly up AND actions-to-first-levelup down >=1 on >=2 games
                AND zero reached_level regression AND test_arc_submitted_agent_parity.py green AND
                in-budget (lazy_value_top_k)."
    principle: "Falsifiable efficiency gate; on failure RETIRE (value_weight=5 disposition)."
    retire_if_same_verdict: true
```

**Metric moved:** live action-efficiency (`actions_to_first_levelup` + `forward_walk_hit_rate`).
Does NOT move `reproducible_total_levels` unless a banked sub-sequence reaches a strictly new
offline-reproduced level — expect "efficiency, not new level."

## Do NOT over-claim

MATM is a consumer-side retrieval result on ALFWorld/WebArena with **success-θ filtering, not
verification** — it does **not** prove an energy verifier improves trajectory selection, and it ran
**no k-ablation**, so its step-reductions are not evidence for Carnot's verifier moat. Its retrieval
is **within-benchmark**; it does **not** prove cross-game transfer (Carnot already has a triply-
replicated cross-game null). Scope every borrowed claim to **within-game**.

## Cross-references
- arXiv:2606.19911 (MATM) — the paper
- FinAcumen arXiv:2606.17642 — the prior ingestion that subsumes the conductor-memory graft
- Self-Harness arXiv:2606.09498 — owns the conductor self-improvement half
- exp4318 / exp4331 / exp4342 — the triply-replicated cross-game transfer null
- `arc_competition_agent.py` (`value_weight=0`, reverted from 5.0 2026-06-20) — the retire disposition precedent
- CLAUDE.md "ARC Live-Path Reachability Discipline", "Circularity / Oracle-Distinctness Discipline",
  "SOTA-Ingestion Cycle Discipline"
