# Research Roadmap — Milestone 2026.06.370

**Planned:** 2026-06-10 (outer-loop planning agent, Claude Opus 4.8)
**Milestone doc for:** `research-roadmap-next.yaml` (`milestone: 2026.06.370`)
**Prior milestone:** 2026.06.369
**North star:** `ops/north-star.md` §0 — solve ARC-AGI-3, accurately AND efficiently;
the energy VERIFIER is Carnot's core value-add.

---

## 0. One-line thesis

`.369` was designed to turn the GAP-4 **program-induction execution verifier** from a
PRELIMINARY positive into a CONFIRMED + DECENTRALIZED + DEPLOYED one — but a single
**agent-shipped poison pre-test** cascade-SKIPped the entire confirm+decentralize+deploy
phase (exp3987–3991) before it ran. **`.370` hardens the green-gate to quarantine any red
test FIRST, then RE-RUNS the five owed tasks** (the operator's four `conductor_followups` +
the local-open-weight generator arm) so they actually execute — while building on `.369`'s
two real wins: r11l **broke the L2 wall and reached L3** via verifier-validated re-induction,
and ArcMemo solve-transfer won again (2668→17 actions). `.370` pushes the level frontier
further (r11l→L4, break lp85/sc25's L2 walls), fixes the 4th-game (induce the dynamics
BEFORE pruning), continues self-learning, hardware, and an ungated capstone.

---

## 1. What the previous milestone (.369) proved (and failed to run)

### Thread A — the load-bearing INFRA failure: a poison-test cascade ate the GAP-4 phase

`.369`'s archive task (exp3986) shipped a test that **failed in its own post-test phase**
(`1 failed, 108 passed`) but was marked OK because the deliverable already existed. That red
test then sat in `tests/python/`, and the conductor's pre-test gate ran it before every
subsequent task → **exp3987, exp3988, exp3989, exp3990, exp3991 all 3-fail-SKIPped**
(`Pre-tests failing, self-heal failed`) before the test was quarantined ~06:23 UTC. The
**entire** GAP-4 CONFIRM + DECENTRALIZE + DEPLOY phase produced **no artifacts**. Capstone
verdict: `gap4_UNCONFIRMED_NOT_DECENTRALIZED_NOT_DEPLOYED ... missing5`.

This is the **4th recurrence** of the agent-shipped poison-test cascade (.325/.326/.332/.369;
memory `incident_agent_shipped_test_cascade`). The conductor cannot be modified from a task,
so **`.370`'s first task (exp3997) is a hardened green-gate**: after writing its deliverable
it runs the FULL `tests/python` suite and QUARANTINES (renames out of pytest collection +
records) any red test, re-running until green — so the owed GAP-4 followups are not
cascade-skipped a second time. The skipped tasks are NOT in the exclusion manifest (an infra
SKIP is not a scientific failure), so re-running them is legitimate forward continuation.

### Thread B — the two REAL wins on the live north star

| Result | Number | Artifact |
|---|---|---|
| **r11l broke the L2 wall → L3** (verifier-validated re-induction IN the solve loop) | `ACCURACY_levels_solved=3`, `new_levels=2`, per-level actions [4,8,12], **2 actions saved vs open-loop**, verifier validated the rule before committing | exp3992 `success: verifier_validated_reinduction_advanced_r11l_to_L3` |
| **ArcMemo solve-transfer v2** (Tier-2 constraint memory compounding) | concept-memory-seeded solve **2668 → 17 actions** | exp3994 `success: arcmemo_solve_transfer_v2_2668to17_actions` |

The L3 win is the **first time the GAP-4 execution verifier earned its place INSIDE a real
ARC-AGI-3 solve**: it validated a re-induced per-level rule (the collision-forbidden mask
exp3980 had only diagnosed) against held-out L2 transitions before any action was spent, and
broke a wall open-loop re-induction could not. This is the mechanism `.370` scales.

### Thread C — the 4th-game miss (and its fix)

exp3993 attempted a 4th distinct game with the verifier as an action-pruner and returned
`fourth_game_no_solve_pruner_rejected_unseen_dynamics` (`induced_mechanic: none`, all of
tn36/su15/dc22 attempted, `duration_s=0.0`). **Root cause: you cannot prune by a model you
have not induced.** The pruner scores actions by consistency with an induced dynamics model;
with the dynamics never observed it rejected everything. `.370`'s fix (exp4004): **active
dynamics exploration FIRST** (probe the env to observe transitions), THEN induce, THEN prune
— the explore→induce→exploit order the ARC-AGI-3 tech report (arXiv:2603.24621) and the
executable-world-model peer (arXiv:2605.05138) both prescribe.

### The five ways the GAP-4 positive is still PRELIMINARY (unchanged — the work never ran)

Per `ops/verifier_gaps.md` GAP-4 "NOT yet ESTABLISHED" + the 2026-06-10 operator handoff:

1. **Statistical significance** — ARC-1 sign test p=0.0625; chain-arms prereg all-gold bar
   NOT met (p=0.07 vs 0.52). → followups **#1 (de-selection coverage)** + **#2 (pre-
   registered precision confirmation)**.
2. **Feedback vs iid resampling** — does a ≤3-iter feedback chain beat 3 independent draws?
   UNRESOLVED. → followup **#3 (feedback-vs-redraw same-run paired control)**.
3. **Decentralization** — the lift is generator-attributable and the generator is closed-
   weight gpt-5.5 (the verifier side is local + model-free). → the owed **local open-weight
   GGUF generator arm** (CLAUDE.md Decentralization Rule 1).
4. **Not yet a registered, reusable verifier.** → followup **#4 (registration + bit-exact
   offline tier-stack re-eval; must reproduce ARC-2 19/31 + ARC-1 28/31)**.
5. **Demo-underdetermination (GAP-5)** — measured by the sibling-input tripwire inside #2/#4.

(NOT pursued: a 400-task scale run — the handoff is explicit it "is NOT yet worth it until
(1)/(2)". Respected.)

---

## 2. The three biggest gaps between current state and the PRD vision

1. **The verifier moat is UNCONFIRMED (existential).** Carnot's whole value-add is the
   verifier (generator is commodity/closed). The GAP-4 positive that would establish it is
   borderline-significant and never got its powered re-test. **Closing this is `.370`'s #1
   job** — the pre-registered precision confirmation (followup #2, exp3999) is the single
   experiment that decides whether independent-induction agreement is a SELECTOR or a
   confidence-label-only signal.
2. **The verifier is not decentralized.** A sovereign headline (CLAUDE.md Rule 1) requires
   the GENERATOR to be local-open-weight too. No published ARC work reports local-open-weight
   program synthesis with execution-verifier rerank — a genuinely novel, ownable number is
   one experiment (exp4002) away, IF the GGUF is cached.
3. **ARC-AGI-3 accuracy is still a thin plateau** (3 games × L1, now r11l→L3). The L3 win
   shows verifier-validated re-induction scales the level frontier; `.370` must convert that
   into monotonic level gains (break lp85/sc25 L2 walls; push r11l→L4) and a 4th game.

---

## 3. Architecture — where `.370` acts

```
  ARC-AGI-3 env (offline Arcade, 25 live environments)
        │  perceive (deterministic objects/targets)
        ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  GENERATOR (induces the rule)                                     │
  │   • codex gpt-5.5 program-induction      ← the proven inducer     │
  │   • LOCAL GGUF program-induction (exp4002)← the OWED sovereign arm │
  └─────────────────────────────────────────────────────────────────┘
        │  candidate def transform(grid) / per-level rule
        ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  ENERGY VERIFIER (Carnot's value-add — model-free, local)        │
  │   demo-fit exact-repro gate → restricted-namespace execution →   │
  │   graded min-hamming snap (τ≤0.005) → gated rerank / vote        │
  │                                                                   │
  │   ROLES exercised in .370:                                        │
  │    • SELECTOR / precision  (followup #2 — IS agreement a selector?)│
  │    • IN-LOOP VALIDATOR     (exp4003 — validate re-induced L-rule)  │
  │    • ACTION-PRUNER         (exp4004 — AFTER dynamics are induced)  │
  │    • REGISTERED STACK      (followup #4 — gap4_program_induction_*)│
  └─────────────────────────────────────────────────────────────────┘
        │  validated rule / pruned action
        ▼
  real env.step → levels_completed (ground truth) ──► ArcMemo concept memory (self-learning)
```

The verifier primitives (`python/carnot/agentic/arc_world_model_synth.py`
`consistency_energy` / `grade_predictions`) are **unchanged and already proven**; `.370`
swaps only the generator (decentralization) and applies the verifier in more roles. The
demo-fit acceptance is execution-guided program synthesis (arXiv:2507.15877); the ≤3-iter
feedback chain is algorithmic-debugging refinement (arXiv:2603.20334) whose value vs iid
resampling followup #3 decides.

---

## 4. Phases & experiments (11 tasks, exp3997–exp4007)

### Phase 0 — Infra: the hardened green-gate (the load-bearing fix)

- **exp3997** archive .369 → activate .370; **GREEN-GATE + POISON-TEST QUARANTINE.** After
  writing the deliverable, run the FULL `tests/python` suite; quarantine any red test (rename
  out of collection into `tests/quarantine/` + record `quarantined_tests`); re-run to green.
  Records `.369` truth (L3 win; GAP-4 confirm-phase poison-skipped; ArcMemo win; 4th-game
  pruner-rejected-unseen-dynamics). This is what stops the cascade recurring. claude/opus.

### Phase 1 — GAP-4 CONFIRM (the four owed `conductor_followups`, re-run after poison-skip)

Queued **verbatim** from `results/arc3_gap4_chain_arms_adversarial_verify.json`
`synthesis.conductor_followups` per the 2026-06-10 TOP-PRIORITY handoff.

- **exp3998 — DE-SELECTION COVERAGE RUN** (codex): k=2 fresh ≤3-iter 600s chains on the 11
  never-chained ARC-2 pool tasks; de-bias the 0.833 coverage estimate; transcripts + gold-leak
  audit; no all-gold bar. (re-issues the poison-skipped exp3987.)
- **exp3999 — PRE-REGISTERED PRECISION CONFIRMATION v2** (codex): k=3 ALL-FRESH chains on NEW
  clean tasks; protocol committed BEFORE any call; **primary binomial critical-value gate
  (n≥19 events, ≥14 gold ⇒ size 0.046 / power 0.837 at p=0.80)**; secondary vs in-run fresh-arm
  rate; **tertiary = task-level unanimity-with-abstention on sibling-input disagreement (the
  GAP-5 tripwire)**; `retire_if_same_verdict` on the precision-uplift claim. **THE confirmatory
  experiment.** (re-issues the poison-skipped exp3988.)
- **exp4000 — FEEDBACK-VS-REDRAW DECIDING CONTROL** (codex): same-run paired — one feedback
  chain vs 3 independent singles, equal 600s, interleaved in ONE run; exact McNemar/Fisher;
  resolves whether feedback content beats iid resampling. (re-issues the poison-skipped exp3989.)
- **exp4001 — HARNESS REGISTRATION + OFFLINE TIER-STACK EVAL** (claude, CPU, zero codex):
  register `gap4_program_induction_stack` in `ops/verifier_registry.yaml`; reusable module;
  **bit-exact offline re-eval must reproduce ARC-2 19/31 and ARC-1 28/31**; append the
  446ef5d2 demo-underdetermination GAP-5 entry to `ops/verifier_gaps.md`. (re-issues the
  poison-skipped exp3990; gemini was the handoff's choice but gemini is BANNED — claude/CPU.)

### Phase 2 — DECENTRALIZE (the owed sovereign generator arm)

- **exp4002 — LOCAL OPEN-WEIGHT GENERATOR ARM** (opus, GGUF): replace the gpt-5.5 inducer with
  a SOTA **local GGUF** program proposer (`Qwen3.6-35B-A3B` / `gemma-4-26B-A4B-it`, loaded via
  the `.gguf` path per the GGUF tokenizer rule). Measure local induction demo-perfect rate +
  the gated rerank on the SAME ARC-1 pool (reproduce the venue) + the **cost** (local-GGUF
  seconds vs codex seconds vs the model-free verifier seconds). PRECONDITION-gated on the GGUF
  cache — NEVER falls back to DSL/codex (the exp3975 silent-fallback lesson). (re-issues the
  poison-skipped exp3991.)

### Phase 3 — ARC-AGI-3 ACCURACY (build on the L3 wall-break)

- **exp4003 — SCALE THE LEVEL FRONTIER** (codex): push r11l L3→L4+ AND apply verifier-validated
  re-induction to break lp85's & sc25's L2 walls. Versioned continuation of the exp3992 WIN.
  Raises the monotonic level counter past the current best (r11l L3 + lp85 L1 + sc25 L1 = 5
  levels). Incremental-Progress Scoping: target +1..+n, never "all levels".
- **exp4004 — 4TH GAME first-solve via active dynamics exploration** (opus): fix exp3993's
  `pruner_rejected_unseen_dynamics` — OBSERVE the dynamics first (probe the env), THEN induce a
  model, THEN verifier-prune. Pick the empirically-easiest non-spatial game by L0 budget. Raises
  games-solved 3→4. Real-env-confirmed.

### Phase 4 — self-learning + hardware + capstone

- **exp4005 — ArcMemo SOLVE-transfer v3** (codex): does banked concept memory cut the cost of
  the NEW level/game from Phase 3 vs cold-start? Extends the exp3994 win to genuinely new
  content. Self-learning MANDATE (research-program.md); positive control (≥2 shared concepts).
- **exp4006 — Hardware continuity** (codex): KV260 (`ssh kria`, toward terminal per north-star
  §3) + GateMate + PolarFire reachability; distinct per-board timers (exp3866 tautology
  corrigendum); SSH-not-SD-card.
- **exp4007 — Capstone .370** (codex, UNGATED): headline question — is the GAP-4 verifier now
  **CONFIRMED** (significance from #1–#3), **DECENTRALIZED** (local arm), **DEPLOYED**
  (registered)? Plus ARC accuracy (games + new levels) and ArcMemo solve-transfer. Skip any
  `flagged_adversarial` artifact; cite upstream sha256; aggregate whatever exists.

---

## 5. Dependency graph

```
exp3997 (archive + POISON-GUARD → green tree)   ← gates everything (must leave tests green)
   ├── exp3998  de-selection coverage         (codex chains; independent)
   ├── exp3999  precision confirmation v2      (codex chains; THE confirmatory)
   ├── exp4000  feedback-vs-redraw             (codex chains; independent)
   ├── exp4001  registration + offline eval    (CPU replay of saved programs; independent)
   ├── exp4002  local-GGUF generator arm       (GGUF-precondition-gated; independent)
   ├── exp4003  scale level frontier           (builds on exp3992 verifier-validated re-induction)
   ├── exp4004  4th game (explore→induce→prune)(builds on exp3993 diagnosis)
   ├── exp4005  ArcMemo solve-transfer v3      (soft-uses exp4003/exp4004 target; falls back to re-held-out)
   ├── exp4006  hardware continuity            (independent)
   └── exp4007  capstone                       (UNGATED aggregator — reads all of the above)
```

No structured `gated_on` chains (the ungated-resilience lesson: gates cascade-block when an
upstream is missing). exp4005 and exp4007 read upstream artifacts but tolerate missing ones.

---

## 6. Routing (gemini BANNED) & hardware

- **gemini is BANNED this milestone.** Every `.367/.368/.369` gemini task stalled (600s/1201s
  silence timeouts; incident_333 quota crash). The two `.369` planner runs were gemini and both
  FAILed — which is why this plan was authored by the outer-loop (Claude Opus 4.8).
- **3 opus** (exp3997 poison-guard infra · exp4002 local-GGUF · exp4004 4th-game — all multi-
  step / bootstrap-risk / hardware-or-GGUF integration per the routing guidance) + **1 claude**
  (exp4001 CPU registration) + **7 codex/gpt-5.5** (program-induction is gpt-5.5;
  aggregation/ARC-planner/registry/hardware are mechanical). 0 gemini.
- **Hardware:** offline ARC-AGI-3 env (Arcade, no GPU for planner/perception tasks). exp4002
  needs a cached SOTA GGUF on the RTX 3090 rig (PRECONDITION-gated; load via the `.gguf` path,
  never `AutoTokenizer` on a GGUF repo id). KV260/GateMate/PolarFire reachability (exp4006).
- **codex (gpt-5.5)** quota for the 3 program-induction confirmatory arms (~3.5k codex-s +
  ~90–135 calls + a same-run paired control; ≥600s timeouts per the handoff hygiene rules).

---

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **Poison-test cascade recurs** (4th time) | exp3997 hardened green-gate quarantines red tests before completing; every task instructed NOT to ship tests asserting on `honest_verdict` strings. |
| **Codex-chain tasks exceed wall-clock** (exp3998/3999/4000 spawn fresh codex chains × many tasks × 600s) | Generous `max_turns` (70-90); followup #2 is the must-run (priority critical); #1 splittable; honest partial coverage allowed (report n + CI, never a bar to clear). |
| **GGUF not cached** (exp4002 decentralization arm) | PRECONDITION check emits `blocked_local_gguf_not_cached` and EXITs — NEVER falls back to DSL/codex. |
| **4th game still unsolved** | Honest `complete: fourth_game_no_solve_<reason>`; the explore-first fix is the new approach (prior_failures documented), not a doomed rerun. |
| **Confirmatory comes back negative** | A powered confidence-label-only retirement IS a confirmed answer (the verifier's value migrates to in-loop validation + cascade-routing, which the L3 win + exp4003 already demonstrate). |

---

## 8. Acceptance — what makes `.370` a win

1. **The GAP-4 confirm+decentralize+deploy phase actually RUNS** (no poison-skip) — exp3998–
   exp4002 land real artifacts.
2. **A powered answer on whether agreement is a selector** (exp3999) — positive or honest
   retirement, either is convergence.
3. **A registered, bit-exact-reproducible verifier** (exp4001: 19/31 + 28/31 replay).
4. **A real local-GGUF induction number** (exp4002) OR an honest `blocked_local_gguf_not_cached`.
5. **Monotonic level progress** past 5 (exp4003 breaks an L2 wall / pushes r11l→L4) and/or a
   4th game (exp4004).
6. **Self-learning compounds** (exp4005) and the boards stay visible (exp4006).

---

## 9. References incorporated (filed in research-references.md)

- **2507.15877** — execution-guided neural program synthesis vs TTFT; the published frame for
  the GAP-4 demo-fit verifier + the ARC-1→ARC-2 transfer probe (OOD strain).
- **2603.20334 ABPR** — LLM-driven algorithmic-debugging feedback refinement; prior art for the
  ≤3-iter chain whose value followup #3 (feedback-vs-redraw) decides.
- **2605.05138 EWM** — induce→verify-program→plan on ARC-AGI-3 (closed GPT-5.x; the local arm is
  the open differentiator).
- **2603.24621** — ARC-AGI-3 tech report (explore→acquire-goal→world-model→adapt; the interactive
  setting exp4004's explore-first fix honors).
