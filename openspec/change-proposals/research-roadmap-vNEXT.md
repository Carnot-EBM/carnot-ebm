# Research Roadmap — Milestone 2026.06.366

**Planned:** 2026-06-09 (outer-loop planning agent, Claude Opus 4.8)
**Milestone doc for:** `research-roadmap-next.yaml` (`milestone: 2026.06.366`)
**Prior milestone:** 2026.06.365
**North star:** `ops/north-star.md` §0 — solve ARC-AGI-3, accurately and efficiently.

---

## 0. One-line thesis

**.365 landed Carnot's FIRST ARC-AGI-3 solve (r11l level-1, real-env-confirmed,
4 actions). .366 turns one solve into a verifier-load-bearing offline run:**
solve more levels and more games (accuracy), run the three owed generalization
experiments that .365 skipped, and — the load-bearing one — prove the verifier
ACTION-PRUNER cuts actions on REAL solved games with a clean no-verifier ablation
(M3). That M3 result is the project's existential "verifier earns its place"
proof, now measurable in the one venue (ARC-AGI-3 / RHAE) where the metric IS
action efficiency and where self-consistency is NOT already near-optimal.

---

## 1. What .365 proved — and where it stalled

### Landed (genuine, real-env-confirmed)
- **exp3946 — FIRST ARC-AGI-3 SOLVE.** r11l level-1 solved offline, confirmed by
  the real env (`real_env_confirmed: true`, `ACCURACY_levels_solved: 1`,
  `first_solve_at_action: 4`). The induced mechanic: a click selects a piece, a
  second click places it so the pieces' average position aligns on the target
  centroid. This is the decisive M1 result the ARC plan ladder was built to reach
  — a persistent perception+planner beat the 0/183 reactive ceiling.
- **exp3945 — activation green-gate** clean (ARC substrate tests green, agentic
  modules import, YAMLs parse).

### Stalled / owed / broken (the .366 work list)
- **exp3947 / exp3948 / exp3949 NEVER RAN.** The conductor ran exp3945 → exp3946,
  then jumped to the capstone, skipping the three Phase-2 generalization tasks:
  the 6-non-spatial-game pipeline sweep, goal-predicate induction, and
  latent-register augmentation. These are owed-not-failed (no verdict produced).
- **exp3950 hardware continuity** produced no artifact (empty).
- **exp3951 capstone BLOCKED by a conductor-config bug.** The capstone's
  `gated_on` used `op: exists`, which is NOT a supported op (supported: `==`,
  `!=`, `>`, `>=`, `<`, `<=`, `in`, `not_in`, `contains`, `not_contains`). The
  gate threw `unknown op 'exists'`, emitted three identical pre-gate-block commits,
  and the operational retro then miscounted the milestone as "0 experiments"
  (a git-heuristic artifact of the repeated block commits). **LESSON BAKED INTO
  .366: never use `op: exists`; the capstone is UNGATED so it aggregates whatever
  landed and cannot stall the milestone.**

### Carry-forward from the SOTA scan (verified 2026-06-09, research-references.md)
- **EWM is now SOTA v2: 58.12% mean RHAE, 15/25 games (GPT-5.5)** — the
  induce-world-model + verify-transitions + stop-on-divergence architecture IS
  Carnot's division of labor. Cite v2, not the old 32.58%.
- **Graph-Based Exploration (no induction) solves median 30/52 levels** — the
  efficiency floor Carnot's pruner competes against.
- **TRM has NO ARC-AGI-3 result** (static ARC-1/2 only) — there is no published
  TRM ARC-3 number to "beat"; the offline quota-gate's operative comparators are
  our prior 0/183, a no-induction baseline, and frontier-LLM <0.4%.
- New energy-as-planner corroborator **Planning-as-Descent (2512.17846)** and
  stochastic-world-model **CASSANDRA (2601.18620)** strengthen the hidden-state
  and pruner experiments.

---

## 2. The three biggest gaps (current state vs north star)

1. **ACCURACY is one level on one game.** RHAE scores nothing on an unsolved
   level and weights later levels more. We have r11l-L1. The gap: more r11l
   levels + ≥1 more game, and an honest count of how many of the 6 non-spatial
   games reach a plan-able (trustworthy) world-model.
2. **The verifier's efficiency value is proven only on a SYNTHETIC env.** exp3929
   measured the action-pruner at 1.96x (CI 1.74-2.19) on a synthetic ARC-style
   env. The north-star §5 existential question — does the verifier earn its place
   — is INCONCLUSIVE on FoVer (capstone .363/.364: `earns=false`). ARC-AGI-3 is
   the headroom venue; the gap is a REAL-game WITH-vs-WITHOUT ablation.
3. **Hidden latent state is the named direction-killer, untested on live games.**
   11/25 games are non-Markov on the visible grid. A grid→grid model is
   under-determined there. The gap: does latent-register augmentation recover
   Markov dynamics (lower consistency energy) on the real hidden-state games?

---

## 3. Architecture (unchanged hybrid; this milestone exercises each role on REAL games)

```
            ARC-AGI-3 offline env (arc_agi SDK, OPERATION_MODE=OFFLINE, 25 games)
                                   |  act -> observe -> act
                 +-----------------+------------------+
   GENERATOR (induces the rule)            VERIFIER  (Carnot's value-add)
   ---------------------------            ------------------------------
   - deterministic numpy perception        - consistency_energy(model, held_out)
     (objects, compute_grid_delta)           = misprediction rate (load-bearing,
   - program/DSL synthesis from the           accuracy-side: certifies a model)
     observed sparse local delta            - action-PRUNER (efficiency-side, the
   - codex (gpt-5.5) = rare heavy             RHAE multiplier): prune looping /
     inducer at escalation points            null-effect / deadly actions
   - local Gemma-4 (SOTA GGUF) = cheap      - goal-predicate / task_potential =
     sovereign perception + proposer         per-step progress (state-value)
                          |                            |
                          +----------> GameGraph <-----+
                   (persistent per-game state-action graph + transition store;
                    cross-episode persistence + cross-game DSL library = SELF-LEARNING)
```

The generator induces; the verifier (a) certifies which induced model is
trustworthy (consistency energy), (b) prunes wasteful actions (RHAE multiplier),
and (c) supplies the goal-distance progress signal. .366 measures each on real
games, with the M3 ablation isolating the pruner's contribution.

---

## 4. Phases & experiments (11 tasks, exp3952-exp3962)

### Phase 0 - Activation (1)
- **exp3952** archive .365 -> activate .366; green-gate (ARC substrate tests, module
  imports, YAML parse). *codex.*

### Phase 1 - ACCURACY: solve more levels and more games (4)
- **exp3953** r11l FULL solve: take r11l from 1/6 to as many of levels 2-6 as the
  induced select/place mechanic + perception reach; real-env-confirmed; report
  per-level actions vs `baseline_actions` (RHAE-style efficiency). *claude/opus.*
- **exp3954** SECOND game solve: solve level-0 of the next-easiest non-spatial game
  (lp85 known-tractable per EWM; su15 / sc25 / tn36 candidates) via the
  perception -> induced-model -> goal-predicate loop; real-env-confirmed.
  *claude/opus.* (prior_failures: vc33 no-solve.)
- **exp3955** the OWED pipeline sweep: active-data -> codex program synthesis ->
  consistency-energy verification across the 6 non-spatial games; per-game
  trustworthy-model table (held-out energy <= 0.15 vs vc33's 0.005). *claude/opus.*
- **exp3956** the OWED goal-predicate induction: induce `goal_predicate(grid)->bool`
  from observed level-ups; precision/recall on held-out win-vs-non-win states.
  *claude/opus.*

### Phase 2 - hidden state + self-learning (2)
- **exp3957** the OWED latent-register augmentation: add latent boolean registers
  (counter / collected-flag / phase) to the 11 hidden-state games; does
  consistency energy DROP vs grid-only? *claude/opus.*
- **exp3958** CROSS-GAME DSL TRANSFER (SELF-LEARNING mandate, research-program.md
  Tier-2 constraint memory): build a DreamCoder-style DSL fragment library across
  the modeled games; measure whether reusing the library makes the Nth game's
  induction CHEAPER (fewer codex calls / lower energy at equal data) than the 1st.
  *claude/opus.*

### Phase 3 - EFFICIENCY: the project's real venue (1)
- **exp3959** M3 EFFICIENCY THESIS (load-bearing): on the games solved this
  milestone (r11l + 2nd game), run the verifier action-pruner WITH vs WITHOUT
  (ablation); measure `EFFICIENCY_mean_action_ratio_on_solved` with bootstrap 95%
  CIs; non-overlapping CIs in the exp3929 direction (1.96x) = the efficiency
  thesis transfers from synthetic to REAL. *claude/opus.* This is the strongest
  available external evidence that the verifier earns its place.

### Phase 4 - M4 readiness + mandates + capstone (3)
- **exp3960** OFFLINE ACCURACY-vs-BASELINE sweep (M4 quota-gate readiness): run the
  hybrid policy through `arc3_offline_eval.py` across the start_here_top8 games;
  report `ACCURACY_total_levels_solved` + `EFFICIENCY` vs the random/object_click
  baselines and the documented frontier-LLM / graph-explore numbers; emit an
  HONEST quota-gate verdict (offline must beat our prior 0 AND a no-induction
  baseline before an online run is justified). Prepare - do NOT submit - the
  operator-only scored-run assessment. *claude/opus.*
- **exp3961** hardware continuity (consolidated; the OWED .365 task): KV260
  (`ssh kria`, SSH-not-SD-card) / GateMate (`openFPGALoader --detect`) / PolarFire
  (`ssh polarfire`) reachability + next step. *codex.*
- **exp3962** capstone .366 (UNGATED - the .365 `op: exists` lesson): aggregate the
  accuracy push (levels solved, trustworthy models, goal-predicate P/R), the
  hidden-state energy delta, the self-learning transfer result, the M3 efficiency
  CIs, and the M4 readiness verdict. SKIP any `flagged_adversarial` artifact; cite
  upstream sha256. *codex.*

---

## 5. Dependency graph

```
exp3952 (activate)
   |-> exp3953 r11l full -------------+
   |-> exp3954 2nd game --------------+
   |-> exp3955 pipeline sweep --------+
   |-> exp3956 goal predicate --------+
   |-> exp3957 latent registers ------+
   |-> exp3958 DSL transfer ----------+
   |-> exp3959 M3 efficiency (reads solved games from exp3953/3954; falls back
   |            to the .365 r11l solve if new solves miss - NOT hard-gated)
   |-> exp3960 M4 offline sweep
   +-> exp3961 hardware
        +-> exp3962 capstone (UNGATED; aggregates whatever landed, skips flagged)
```

No task is hard-`gated_on` another, by design - the .365 stall came from a gated
capstone. Each task self-checks its preconditions and emits an honest `blocked_*`
on a missing prerequisite rather than cascade-blocking the milestone.

---

## 6. Models & substrate

- **Generator/inducer:** codex (gpt-5.5) as the rare heavy inducer; local **Gemma-4
  SOTA GGUF (`unsloth/gemma-4-26B-A4B-it-GGUF`)** as the sovereign multimodal
  perception/proposer, with `gemma-4-E4B-it` as the fast per-step fallback.
  Deterministic numpy perception (`compute_grid_delta`, `objects`) is primary -
  the LLM is used only for ambiguous object semantics / DSL proposal.
- **Verifier:** CPU energy/consistency ensemble (`consistency_energy`,
  `arc_agi3_action_efficiency.select_verifier_pruned_action`, the SAT/AST/AND
  composition verifiers) - no GPU required.
- **Substrate declarations:** ARC solve/induction tasks declare
  `offline_arc_agi3_*` substrates (low duration floor, real-env-confirmed);
  pipeline sweep declares the codex-synthesis substrate; capstone/activate declare
  `aggregation_from_upstream_artifacts`; hardware declares `hardware_smoke`.

## 7. Hardware requirements

- **None blocking.** All ARC work is CPU + the offline `arc_agi` SDK +
  `environment_files/`; Gemma perception is GPU-optional (small model). The
  consolidated hardware-continuity task (exp3961) keeps KV260/GateMate/PolarFire
  visible per the Hardware-Task Continuity Discipline (KV260 SSH-not-SD-card).

## 8. Disciplines honored

- **Capstone UNGATED** (the .365 `op: exists` fix); no task hard-`gated_on` another.
- **PRECONDITIONS-first** on every compute/offline-env/codex task; honest
  `blocked_<resource>` on a miss (no fabrication).
- **Verdict terminal-prefix** (`complete:` / `success:` / `blocked_`).
- **inference_substrate** declared per task; real `duration_s`, `random_seed`.
- **Principle-annotated** REQUIRED ARTIFACT FIELDS + gate conditions.
- **operator_override** on every routine/continuation/owed task (standing
  2026-06-08 ARC directive) + **prior_failures** on the 2nd-game task (vc33
  no-solve root cause). Exclusion-manifest scan: no ARC scope-matches retired ids.
- **Self-learning** experiment present (exp3958 cross-game DSL transfer).
- **Operator-only external publication:** exp3960 PREPARES the scored-run
  assessment; the operator triggers any online run.
