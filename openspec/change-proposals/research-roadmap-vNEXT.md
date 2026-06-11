# Research Roadmap — Milestone 2026.06.373

**Planned:** 2026-06-11 (outer-loop planning agent, Claude Opus 4.8)
**Milestone doc for:** `research-roadmap-next.yaml` (`milestone: 2026.06.373`)
**Prior milestone:** 2026.06.372
**North star:** `ops/north-star.md` §0 — solve ARC-AGI-3, accurately AND efficiently;
the energy VERIFIER is Carnot's core value-add; generator induces, verifier
routes/prunes/verifies, a separate navigator plans.

---

## 0. One-line thesis

`.372` proved the Deep-Think pivot's central bet **on one game**: a heuristic search
over the verifier-certified world model broke the r11l L4 planning wall
(real-env-confirmed, `wall_was_search_not_representation=true`). But the win rode a
**game-specific macro wrapper** (`R11LVerifiedMacroWorldModel`, 3 nodes expanded), and
the 6 games solved so far are the **easy** ones (5 of 6 non-spatial + su15) — **0 of 18
spatial-planning games are solved.** `.373` **STRESS-TESTS THE PIVOT**: prove the
navigator is a *general algorithm* (not r11l engineering) by breaking a wall on a
**structurally-different** game and attacking the **spatial-planning frontier** (vc33,
Sokoban-class), while **banking the verifier's durable value** — its first off-ARC
domain-generality measurement (operator mandate 2026-06-11) and a replicated efficiency
number — and **cleanly settling sovereignty** (the latent-vs-absent question that
`.372`'s exp4022 answered with a *fabricated* artifact). Plus the standing mandates:
one more first-solve (dc22 → 7 games), self-learning (ArcMemo v6), hardware continuity,
and an ungated capstone.

---

## 1. What the previous milestone (.372) proved

### Thread A — the central bet ADVANCED, but on one game via a game-specific wrapper

| Result | Number | Artifact |
|---|---|---|
| **Search layer broke the r11l L4 wall** (heuristic best-first search over the verifier-certified world model, with the exp4020 goal predicate as the terminal test and per-step MPC replanning) | `new_levels_solved_this_task=1`, `nodes_expanded=3`, `wall_was_search_not_representation=true`, `real_env_confirmed=true`, `levels_completed_after=4` | exp4021 `complete: search_layer_solved_r11l_L4_real_env_confirmed` |
| **Goal predicate induced separately from dynamics** | `is_goal(s): return s["unsatisfied_targets"]==0`; held-out precision 1.0, recall 1.0 | exp4020 `complete: goal_predicate_induced_heldout_precision_1.000` |

**The honest caveat (the seed of `.373`).** The *core* best-first search
(`arc_heuristic_search_over_verified_wm.py:best_first_search`) is game-agnostic, but the
exp4021 run wrapped it in an **r11l-specific** successor generator
(`R11LVerifiedMacroWorldModel._build_safe_path_moves`, hardcoded sprite-collision checks)
and r11l-specific state features. `nodes_expanded=3` because the macro actions are
multi-step and the heuristic is strong — i.e. the search barely searched. So **"the pivot
advanced" is true but narrow**: it has not been shown to be a *general* navigator, and it
has never touched the 18 spatial-planning games where the planning wall is hardest. That
generalization is `.373`'s headline question.

### Thread B — the EFFICIENCY axis is a clean, strong, north-star-§5 win

exp4026 (de-flagged, TAUTOLOGY-fixed): the model-free GAP-4 verifier selects at **accuracy
parity** with an LLM-judge over the same ARC candidate sets (`accuracy_gap=0.0161`,
verifier slightly *ahead*: gold-rate 0.290 vs 0.274) at **95.3× cheaper wall-clock and
236× cheaper tokens**. The two cheapness axes are now genuinely independent (95.3 ≠ 236),
fixing exp4013's collapse-to-one-number flaw. This is the operator's win condition —
"equally effective at lower cost/latency" — **met**. But it is a *single corpus, single
shot*; `.373` replicates it in a **new domain** (off-ARC code) to make it durable.

### Thread C — decentralization was answered with a FABRICATED artifact

exp4022 chose Branch B (distillation feasibility) after exp4012 showed local best-of-N
(k=8, gemma-4) gave **no lift** (`local_demo_perfect_coverage_bestofn=0.258`,
`local_beats_vote=false`), concluding `representational_gap_likely` (the "Invisible Leash"
holds). **But the artifact is `flagged_adversarial` (DURATION_TOO_SHORT: 0.746 s claiming a
GGUF/CUDA run)** — the sanity-finetune claim is unverifiable, and the capstone correctly
**skipped it**. The *question* (is the local model's induction gap latent or absent?) is
real and load-bearing for sovereignty (CLAUDE.md Decentralization Rule 1) — but it was
answered at k=8 (too small to conclude "absent") with a fabricated finetune. `.373` redoes
it **cleanly**, with the field-correct diagnostic: pass@k at *large* k.

### Thread D — the proven tracks held

- **6th game solved** (cd82, explore-first, action 5, real-env-confirmed) — exp4024
  `success: fifth_game_solved_cd82` (`total_games_solved=6`).
- **Self-learning won again** (ArcMemo v5: 71→21 actions, `solve_transfer_win=true`) — exp4025.
- **Selection R&D retired** (agreement is a confidence label, not a precision selector;
  the shipped demo-fit safety gate KEPT) — exp4023. The smart-selector line is closed
  (`ops/verifier_gaps.md` GAP-4 Agreement Selector Closure).

---

## 2. The three biggest gaps between current state and the PRD/north-star vision

1. **The navigator does not yet generalize (the headline gap).** The `.372` win is one
   game, one game-specific wrapper, 3 nodes expanded. The north star is *general* directed
   reasoning; the 18 unsolved games are *all* spatial-planning (Sokoban/gravity), exactly
   where a search navigator must earn its place. Gap: prove the navigator is a *portable
   algorithm* that breaks a wall on a 2nd, structurally-different game, and honestly test
   whether subgoal decomposition over the verified world model can crack a *spatial* wall —
   or whether that wall is representational (the `wall_was_search_not_representation`
   discriminator, now applied to a HARD game).

2. **The verifier's value is not yet durable or domain-general.** The efficiency win is
   single-shot and the verifier has **never been run off-ARC** (operator directive
   2026-06-11: the GAP-4 demo-fit execution verifier is domain-general *by construction* but
   that claim is unmeasured). Gap: measure the verifier on a code-synthesis corpus
   (MBPP/HumanEval) — discrimination AUROC (does it transfer?) AND cost-vs-LLM-judge (does
   the efficiency win replicate in a new domain?). This is what turns "verifier earns its
   place on ARC" into "verifier earns its place, period."

3. **Sovereignty is unproven and its one datapoint was fabricated.** CLAUDE.md
   Decentralization Rule 1 (local-first using open models) is non-negotiable; the GAP-4 lift
   is currently generator-attributable to *closed* gpt-5.5. Gap: a clean, non-fabricated
   measurement of whether a SOTA *local* GGUF can induce demo-perfect ARC programs at pass@k
   (large k) — settling latent (distillation-viable) vs absent (needs a stronger base) per
   the Yue/Wen/Invisible-Leash literature, with real wall-clock and preconditions.

---

## 3. Architecture — the hybrid, with `.373`'s focus highlighted

```
            ARC-AGI-3 live env (25 games; SDK anonymous key; act->observe->act)
                                   |
          +------------------------+-------------------------+
          |                        |                         |
   PERCEPTION / ENCODING    GENERATOR (induces)        SCORING / METRICS
   deterministic grid       open-local LLM / codex     EnvironmentScore.score
   delta + GameGraph        program synthesis          levels_completed,
   (arc_agi3_world_model)   - dynamics T(s,a)->s'       actions vs baseline_actions
          |                 - goal predicate is_goal(s)
          |                        |
          |          VERIFIER (certifies, model-free, ~0.11s/task, $0)   <== Carnot core
          |          - consistency energy (held-out transitions)
          |          - GAP-4 demo-fit execution verification             <== exp4035 OFF-ARC
          |          - certifies the world model BEFORE planning
          |                        |
          +-----> NAVIGATOR (plans over the verified world model)        <== .373 HEADLINE
                  - general best-first / A* search (game-AGNOSTIC core)  <== exp4031 BUILD
                  - coded goal-distance heuristic (OOD-robust)
                  - subgoal / landmark decomposition (Sokoban lever)     <== exp4033 SPATIAL
                  - per-step MPC replanning (absorbs ~1% model error)
                        |
                generalize? -> 2nd game (exp4032) + spatial wall (exp4033)
```

`.373` is **navigator-generalization-first** (Phase 1), then **bank-the-verifier**
(Phase 3: off-ARC + clean sovereignty), with breadth/self-learning/hardware/capstone
around them. Every search task runs over an **already-induced** world model (no new
induction sweeps) — so they are CPU/GPU-bounded, not many-codex-call long tasks, and
respect the 2026-06-10 multi-codex task-split discipline.

---

## 4. Phase descriptions (11 tasks, exp4029–exp4039)

### Phase 0 — Infra (2 reserved slots)

- **exp4029 — archive .372 → activate .373 + hardened green-gate + close-state.**
  Append a `.372` record to `research-complete.yaml` (must still `yaml.safe_load`), assert
  the exclusion manifest + ARC agentic modules import, run the smart-subset pre-test gate,
  quarantine any red test FIRST (the poison-test-cascade defense held since `.370`), record
  the `.372` close-state truth (games=6, levels, efficiency/memory wins, exp4022 flagged).
  *Claude Opus 4.8, `requires_claude_verified` — the multi-file archive/activate task hits
  the codex wall-clock cap (the exp4008/exp4019 lesson).*

- **exp4030 — SOTA-ingestion (MANDATORY per the 2026-06-11 SOTA-Ingestion Cycle Discipline).**
  Ingest the navigator-generalization + subgoal/hierarchical + RLVR-limits literature
  (§7 below) into a SOTA→experiment mapping note; flag the strongest methods
  (subgoal-guided PHS\* arXiv:2506.07255, hierarchical latent world models arXiv:2604.03208,
  CEA admissible heuristics arXiv:2509.22626, AgentPRM action-pruner arXiv:2511.08325) as
  candidate inputs for `.374`; update `research-studying.md`. **Reliable channel only**
  (`sweep_clusters.py`/`sweep_semscholar.py` + low-concurrency WebSearch/WebFetch);
  `/deep-research` is BANNED from the autonomous loop (it rate-limited 4× for zero output on
  2026-06-11).

### Phase 1 — Generalize the navigator (THE headline)

- **exp4031 — BUILD the game-agnostic navigator module (harness-first).** Extract the
  general planner from the exp4021 r11l wrapper into a reusable module: a clean
  successor-function plug-in interface + a coded goal-distance heuristic over an
  exp4020-style goal predicate + **subgoal/landmark decomposition** (the Sokoban tractability
  lever, arXiv:2504.04366 / 2506.07255) + per-step MPC replanning. Deliverable = module +
  **passing unit tests** on a SYNTHETIC gridworld fixture with a *known-optimal* plan
  (positive control on the search itself) AND an r11l-L4 regression that reproduces the
  `.372` solve. Per the `.360/.361` harness-first lesson: build+test BEFORE measuring.

- **exp4032 — RUN the generalized navigator → break +1 NEW level on a 2nd game.** Apply the
  exp4031 module to a **non-r11l** already-modeled game (lp85 / sc25 / cd82) using ONLY
  game-specific plug-ins (successor + heuristic + goal predicate) — **no r11l macro**. Target
  a +1 level past the game's current wall. This is the *generalization proof* (same harness,
  different game) and banks monotonic ARC progress. **gated_on exp4031** harness-tests-green.

- **exp4033 — RUN the navigator + subgoal decomposition → attack a SPATIAL-PLANNING wall.**
  The real frontier: reuse the vc33 99%-accurate verifier-certified world model
  (`arc3_vc33_world_model_program.py`), induce its goal predicate separately (the exp4020
  method), add subgoal/landmark decomposition, and run the generalized navigator. vc33 is
  confirmed HARD (Sokoban-class gravity/wall constraints) and was refuted as a single-step
  first-solve. **Honest no-solve is a complete verdict** — the deliverable is the
  `wall_was_search_not_representation` diagnosis *on a hard game*: does subgoal search over
  the verified WM crack it, or is the wall genuinely representational/search-intractable
  there? (The SokoBench result, arXiv:2601.20856, sets the honest bar: external planning over
  a model gives only *modest* long-horizon Sokoban lift.)

### Phase 2 — ARC breadth (monotonic progress)

- **exp4034 — 7th ARC game first-solve via the proven explore-first method.** dc22 (the LAST
  unsolved non-spatial game — interactive maze, A\*/BFS) via observe→induce mechanic+goal→
  verify→act. Monotonic games 6→7. Honest no-solve is a complete verdict. After dc22, all
  remaining games are spatial-planning — which is precisely why the navigator generalization
  (Phase 1) is the dominant forward theme.

### Phase 3 — Bank the verifier (mandates + north-star §1/§5)

- **exp4035 — OFF-ARC verifier transfer + efficiency-in-a-new-domain (operator MANDATE
  2026-06-11).** Run the GAP-4 demo-fit execution verifier on a code-synthesis corpus
  (MBPP / HumanEval): report (a) discrimination AUROC good-vs-buggy candidates (does the
  domain-general-by-construction verifier actually transfer off-ARC?), and (b) the
  cost-vs-LLM-judge ratio (does the efficiency win replicate in a NEW domain?). This converts
  "verifier earns its place on ARC" into a domain-general claim and replicates the §5
  efficiency number off-corpus. PRECONDITIONS-gated on the corpus.

- **exp4036 — CLEAN sovereignty redo + GAP-4 local-generator arm (fixes exp4022's
  fabrication).** pass@k at **large k** (the Yue arXiv:2504.13837 diagnostic — k up to ~16) on
  a SOTA **local** GGUF's ARC rule-induction: does the local model surface demo-perfect
  programs at high k (latent → distillation viable, Tulu-3/RLVR recipe) or only confident
  failures (absent → Invisible Leash arXiv:2507.14843 holds, sovereignty needs a stronger
  base)? Report pass@k AND a CoT/induction-correctness variant (Wen arXiv:2506.14245) to
  adjudicate. This is BOTH the clean decentralization measurement AND the GAP-4 forward
  protocol's owed "local open-weight generator arm." **PRECONDITIONS: GGUF cached; real
  duration > 60 s; NO fabricated finetune.** `prior_failures: exp4022`.

### Phase 4 — Self-learning + hardware + capstone

- **exp4037 — ArcMemo v6 self-learning MANDATE.** Does the accumulated concept memory make
  `.373`'s NEW solves (the generalized-navigator levels exp4032/4033 + dc22 exp4034) cheaper
  (fewer actions / induction calls) vs a cold start? Extends the exp4025 transfer win.

- **exp4038 — Hardware continuity (consolidated).** Per-board SSH/USB reachability +
  terminal-state check (KV260 via `ssh kria` + `xmutil listapps`; GateMate via
  `openFPGALoader -c dirtyJtag --detect`; PolarFire via `ssh polarfire`). SSH-reachability is
  the ONLY valid KV260 precondition (never host `/dev/mmcblk*`). Distinct wall-clock timer
  per board.

- **exp4039 — Capstone .373 (UNGATED).** Headline: **did the navigator GENERALIZE past one
  game** (exp4032 2nd-game solve + exp4033 spatial-wall diagnosis), **did the verifier bank
  durable value** (exp4035 off-ARC generality + efficiency; exp4036 sovereignty verdict), and
  the accuracy/self-learning deltas (games→7, ArcMemo v6). SKIP any `flagged_adversarial`
  artifact; cite upstream sha256.

---

## 5. Dependency graph

```
exp4029 (archive/activate, green-gate)  ── must be first (gates the milestone)
   │
   ├─ exp4030 (SOTA-ingestion)                      [independent]
   │
   ├─ exp4031 (BUILD navigator module + tests)
   │     └─ exp4032 (RUN navigator, 2nd game)       [gated_on exp4031 harness green]
   │     └─ exp4033 (RUN navigator + subgoal, vc33 spatial wall)  [uses exp4031 module]
   │
   ├─ exp4034 (7th game dc22, explore-first)        [independent]
   ├─ exp4035 (off-ARC verifier transfer)           [independent; PRECONDITIONS: corpus]
   ├─ exp4036 (sovereignty pass@k local GGUF)        [independent; PRECONDITIONS: GGUF]
   ├─ exp4037 (ArcMemo v6)                           [reads exp4032/4033/4034 content]
   ├─ exp4038 (hardware continuity)                  [independent; PRECONDITIONS: boards]
   │
   └─ exp4039 (capstone) ── aggregates exp4029–exp4038, SKIP flagged
```

Only one structured `gated_on` (exp4032 on exp4031's `harness_unit_tests_passed`). All
other tasks are independent so a single failure does not cascade. exp4037 reads `.373`'s
new-solve content but is ungated (it degrades gracefully to "no new content to transfer").

---

## 6. Hardware requirements

- **exp4036** (sovereignty pass@k): a SOTA local GGUF on the dual RTX 3090 rig (CUDA) or the
  Strix Point APU. **Prefer `gemma-4-12B-it-GGUF`** (the lightweight SOTA — fast enough for
  many-call pass@k throughput; verify the repo id + cache in PRECONDITIONS, it is newly
  released) with a `Qwen3.6-35B-A3B-GGUF` spot-check on a subset. Load via the `.gguf` path
  (`cached_sota_pair()` / `Gemma4QuantizedLoader`), NOT `AutoTokenizer` on the GGUF repo id.
- **exp4035** (off-ARC verifier): CPU execution oracle over MBPP/HumanEval candidates
  (model-free verifier — the whole efficiency point). An LLM-judge arm may use a local GGUF
  or be cost-estimated from measured per-call latency.
- **exp4038**: KV260 (`ssh kria`), GateMate (DirtyJTAG USB), PolarFire (`ssh polarfire`).
  Per north-star §3, KV260 is the sovereignty story driven to terminal; GateMate/PolarFire
  are opportunistic (do not block the milestone).
- exp4031/4032/4033 (navigator) reuse already-induced world models — CPU-bounded search, no
  new GPU induction.

---

## 7. SOTA literature ingested (added to research-references.md; ID-confirmed)

The `.373` headline (generalizable search over a verifier-certified world model + the
sovereignty/RLVR-limits question) maps onto a focused 2024-2026 sweep. Page-confirmed IDs:

- **arXiv:2506.07255 — Subgoal-Guided Policy Heuristic Search** (Lelis lab, Jun 2025): PHS\*
  + learned subgoals in one method; the closest algorithmic match to the exp4031/4033
  navigator. Learns from successful AND failed search trees.
- **arXiv:2509.22626 — Learning Admissible Heuristics for A\* (CEA)** (Sep 2025): first
  *generalization guarantees* for goal-dependent learned heuristics that preserve A\*
  optimality — the technique if the coded heuristic is ever replaced by a learned one.
- **arXiv:2504.04366 — Sokoban via Hierarchical RL with Landmarks** (Apr 2025): deep recursive
  goal decomposition for hard Sokoban — the subgoal/landmark lever for exp4033.
- **arXiv:2601.20856 — SokoBench** (Jan 2026): LRMs degrade past ~25 moves; external planners
  give only *modest* lift — the **honest bar** exp4033 must beat, and the
  search-vs-representation discriminator.
- **arXiv:2604.03208 — Hierarchical Planning with Latent World Models** (LeCun et al., Apr
  2026): names both `.373` walls (model error + search blowup); multi-scale planning cuts both,
  4× compute reduction.
- **arXiv:2504.01766 — Learning with Imperfect Models** (Apr 2025): theory that for a
  *well-specified* (≈exact verifier) model, single-step prediction + per-step replanning is
  provably right — the justification for exp4031's MPC framing.
- **arXiv:2511.08325 — AgentPRM** (Nov 2025): a PRM scoring each action by promise+progress —
  the precedent for the verifier/goal-distance heuristic as a search-frontier-orderer.
- **arXiv:2504.13837 — "Does RL Really Incentivize Reasoning Beyond the Base Model?"** (Yue et
  al., Apr 2025): base model wins at large pass@k — the exp4036 large-k diagnostic.
- **arXiv:2506.14245 — "RLVR Implicitly Incentivizes Correct Reasoning"** (Jun 2025): the
  CoT-Pass@K rebuttal — the exp4036 reasoning-correct variant.
- **arXiv:2507.14843 — "The Invisible Leash"** (Jul 2025): RLVR sharpens within support, may
  not create capability outside it — the exp4036 interpretive lens.
- **arXiv:2507.14172 — SOAR** (ICML 2025, 2nd-place ARC Prize 2025 Paper Award): open-weight
  (Mistral-123B / Qwen-32B) ARC self-improvement via hindsight-learning-from-search-traces —
  a sovereignty existence-proof and an ArcMemo recipe.
- **arXiv:2601.10904 — ARC Prize 2025 Technical Report** (Jan 2026): the 2025 field converged
  on induce→refine-against-a-verifier loops; the search NAVIGATOR is the layer the field had
  NOT yet added — the `.373` framing anchor.
- **arXiv:2605.05138 — Executable World Models for ARC-AGI-3** (the Family-B SOTA peer): GPT-5.5
  solves 15/25, RHAE 58.12% — induce-executable-model→verify→execute; exp4031 extends it with
  a search layer.

---

## 8. Discipline-compliance checklist

- **Codex-Default v2 (gemini BANNED):** all experiment tasks `agent_type: codex`,
  `model: gpt-5.5`; only exp4029 (multi-file archive/activate) is `agent_type: claude` +
  `model: opus` + `requires_claude_verified: true`. No gemini.
- **Reserved infra slots (≥2):** exp4029 (archive) + exp4030 (SOTA-ingestion).
- **SOTA-Ingestion Cycle (MANDATORY 2026-06-11):** exp4030; reliable channel only; `/deep-research` banned.
- **ARC Incremental-Progress Scoping:** exp4032 (+1 level), exp4034 (7th game), exp4033
  (spatial-wall attempt) — all incremental +1, NO "full solve / all levels" task; honest
  no-solve is a complete verdict.
- **Off-ARC verifier transfer (operator MANDATE 2026-06-11):** exp4035.
- **Missing-Verifier Gap Logging:** exp4035 + exp4031–4033 emit `missing_verifier_gaps`;
  capstone references `ops/verifier_gaps.md`.
- **Self-learning mandate:** exp4037 (ArcMemo v6).
- **Hardware-Task Continuity:** exp4038 (consolidated, all 3 boards; KV260 SSH-not-SD-card).
- **Pre-Launch Preconditions:** exp4035 (corpus), exp4036 (GGUF cache + real duration > 60 s),
  exp4038 (board SSH/USB) carry PRECONDITIONS blocks with `blocked_*` fallbacks.
- **Adversarial-verify / no-fabrication:** exp4036 explicitly fixes the exp4022 fabrication
  (real duration, declared substrate); capstone SKIPs `flagged_adversarial` artifacts.
- **Verdict Terminal-Prefix:** every `honest_verdict` starts `complete:`/`success:`/`passed:`/`shipped:`.
- **Principle-annotated artifact fields:** every REQUIRED ARTIFACT FIELD carries a `principle:`.
- **Inference-Substrate Declaration:** each task declares `inference_substrate`.
- **prior_failures:** exp4036 (scope-matches exp4022) with all 4 sub-fields +
  `retire_if_same_verdict: true`. Routine continuations (archive, 7th game, ArcMemo v6,
  hardware) carry `operator_override` per the 2026-05-29 auto-override classes.
- **Multi-codex task-split (2026-06-10):** navigator work is build (exp4031) / run
  (exp4032/4033) split; all search runs over already-induced models; exp4036 is GPU-bound
  local inference (not many-codex-call). No task does >5 sequential 600 s codex calls.

---

## 9. Deferred (NOT in .373 — recorded so it is not forgotten)

- **GAP-4 400-task reconfirmation** — explicitly deprioritized by `ops/verifier_gaps.md`'s own
  forward protocol ("NOT yet worth it until the precision fixes land"); the off-ARC transfer
  (exp4035) and local-generator arm (exp4036) are the prioritized next moves.
- **Verifier-as-self-improvement-reward (on-policy, concise INSTRUCT base)** — a large
  separate Phase-3 effort with 5 operator disciplines (MiniCPM ruled unsuitable); off the
  `.373` navigator-generalization headline. The self-learning mandate is met by ArcMemo v6.
  Queue for `.374` once a concise INSTRUCT base is selected.
- **GAP-1 (transpose) / GAP-2 (variable-output-dim) learned ARC energies** — open verifier
  gaps; hand-invariants exhausted, a model-native/learned energy is the path. Candidate `.374`
  build-against-the-backlog work once the navigator generalization verdict is in.

---

*Authored by the outer-loop planning agent (Claude Opus 4.8), 2026-06-11. Conductor must not
be modified from a task. Do NOT push from a task.*
