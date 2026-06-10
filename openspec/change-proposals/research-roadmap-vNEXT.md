# Research Roadmap — Milestone 2026.06.371

**Planned:** 2026-06-10 (outer-loop planning agent, Claude Opus 4.8)
**Milestone doc for:** `research-roadmap-next.yaml` (`milestone: 2026.06.371`)
**Prior milestone:** 2026.06.370
**North star (`ops/north-star.md` §0):** solve ARC-AGI-3 accurately AND efficiently; the
energy/model-free VERIFIER is Carnot's core, existential value-add (§5).

---

## 1. What `.370 proved (and what it left owed)

`.370 re-issued the five GAP-4 confirm/decentralize/deploy tasks that the `.369 agent-shipped
poison-test cascade had SKIPped. The hardened green-gate (exp3997) held — the GAP-4 phase RAN
this time. Capstone verdict (exp4007):
`gap4_PHASE_RAN_UNCONFIRMED_DECENTRALIZED_DEPLOYED_local_not_beats_vote_games4_levels5_arcmemo_transfer_win`.

Read via `scripts/summarize_artifact.py`, the honest per-task state:

| Task | Result | Status for `.371 |
|---|---|---|
| exp3997 green-gate | `pretest_suite_green=true`; poison-guard held | KEEP the guard |
| exp3998 de-selection coverage | `0.4091` on 11 never-chained tasks (vs 0.8333 selected); **de-biased coverage 0.6304** | DONE — honest denominator banked |
| **exp3999 precision confirmation** | **`protocol_preregistered_pending_execution`, 0 codex calls, 0 agreement events** | **NEVER RAN — owed a 2nd time** |
| exp4000 feedback-vs-redraw | `feedback_no_better_than_redraw_p1.0_FALSE_NEGATIVE_RISK` | UNDERPOWERED — re-run with power |
| exp4001 registration/deploy | `arc2 19/31 + arc1 28/31 reproduced bit-exact`; registered; GAP-5 appended | DONE — **DEPLOYED** |
| exp4002 local generator | `local_induction 0.2581 / pass2 0.4516 (=vote) / below_codex`; gemma-4-26B, <=3 attempts; failed 20/31 codex-solved tasks | **WEAK — close the induction gap** |
| exp4003 scale frontier | `level_frontier_holds` — r11l L4 re-induction found NO verifier-validated candidate; 36 actions saved; levels held at 5 | walls UNBROKEN — explore-first |
| **exp4004 fourth game** | **`fourth_game_solved_su15 at action 14`** (explore-first fix worked; dynamics_induced before pruning) | **WIN — games 3→4** |
| exp4005 ArcMemo transfer v3 | `solve_transfer 14→10 actions` (win) | WIN — self-learning compounds |

**Two REAL wins:** the explore-first fix solved a 4th game (su15), and ArcMemo self-learning
compounded again. **DEPLOYED + DECENTRALIZED(weak):** the program-induction verifier is registered
and reproduces bit-exact; a local open-weight model CAN drive it but weakly.

**The decisive owed item:** the **GAP-4 precision confirmation has now failed to EXECUTE for two
consecutive milestones** — poison-skipped in `.369, then pre-registered-but-never-run in `.370
(exp3999 emitted `pending_execution` with 0 codex calls). The single question that decides whether
the verifier moat is CONFIRMED — *is independent-induction agreement a real precision selector or
only a confidence label?* — is still unanswered. This is `.371's #1 priority, designed so it
**cannot exit without executing**.

---

## 2. The three biggest gaps (current state → PRD/north-star vision)

1. **The verifier moat is DEPLOYED but UNCONFIRMED.** The agreement-as-selector claim has never
   been tested with statistical power (the confirmatory experiment never ran). Until it does, the
   GAP-4 program-induction verifier — Carnot's ARC-domain instance of "the verifier earns its
   place" (north-star §5) — has no confirmed precision uplift. **Gap → a powered confirmation that
   actually runs, plus the stronger cross-example-consistency discriminator the GAP-5 entry calls
   for.**

2. **The decentralization path is sovereign but weak.** A local open-weight generator induces ARC
   programs at 25.81% (vs codex 94%) and gives zero rerank lift, because exp4002 drew only <=3
   samples from a generic instruct GGUF. The literature (BARC: ~22% of ARC-AGI-1 at a 512-sample
   budget + demo-fit filter) says open-weight induction is a **best-of-N + cheap-verifier-filter**
   regime, not a few-shot one. **Gap → best-of-N local sampling + the free demo-fit verifier
   filter; the verifier is ~0.11 s/task, so spending more local compute + filtering is the
   decentralization-clean efficiency path.**

3. **ARC accuracy: the level walls are unbroken and only 4 games are solved.** `.370 banked a 4th
   game (su15) via explore-first, but the L2 walls (lp85, sc25) and r11l L4 held because exp4003
   re-induced WITHOUT first exploring per-level dynamics. **Gap → apply the proven explore-first
   method (observe per-level dynamics → induce → verifier-validate) to break a wall and to solve a
   5th game — monotonic progress per the Incremental-Progress Scoping rule.**

Plus the standing efficiency axis (north-star §5): **no paper reports a model-free-verifier-vs-
LLM-judge cost ratio on ARC selection** — an open, ownable Carnot number.

---

## 3. Milestone shape — 11 tasks (exp4008–4018), 5 phases

```
Phase 0  INFRA            exp4008  archive .370 -> activate .371 + HARDENED green-gate (KEEP the poison-guard)
Phase 1  CONFIRM the moat exp4009  precision confirmation v3  — MUST EXECUTE (owed 2 milestones)
                          exp4010  GAP-5 cross-example-consistency SELECTOR upgrade (build against the logged gap)
                          exp4011  feedback-vs-redraw v2 — POWERED (exp4000 was FALSE_NEGATIVE_RISK)
Phase 2  DECENTRALIZE     exp4012  local best-of-N + cheap verifier filter (close the 25.81% induction gap; BARC regime)
Phase 3  EFFICIENCY       exp4013  model-free verifier vs LLM-judge cost ratio on ARC selection (north-star §5)
Phase 4  ARC ACCURACY     exp4014  break a level wall via explore-first per-level re-induction (lp85/sc25 L2, r11l L4)
                          exp4015  FIFTH ARC-AGI-3 game first-solve via explore-first (games 4->5)
Phase 5  MANDATES+CAP     exp4016  ArcMemo solve-transfer v4 (self-learning mandate; extends the 14->10 win)
                          exp4017  hardware continuity (consolidated; KV260 toward terminal)
                          exp4018  capstone .371 (UNGATED) — is the verifier now CONFIRMED + decentralized-effective?
```

**Dependency graph (NO structured `gated_on` — the ungated-resilience lesson from the `.365
op:exists block + the `.369/`.370 poison cascades).** Each task handles missing upstream
gracefully so a single failure cannot cascade-skip the milestone:

```
exp4008 (green-gate, KEEP) ── gates nothing structurally; just guarantees a green tree
   │
   ├─ Phase 1 (independent): exp4009, exp4010, exp4011  — each reads the cached GAP-4 pool; no cross-dep
   ├─ Phase 2 (independent): exp4012  — reads the cached ARC-1 pool + model-free verifier
   ├─ Phase 3 (independent): exp4013  — reads the cached candidate sets
   ├─ Phase 4 (independent): exp4014, exp4015  — real offline env; each stops at first failed level
   └─ Phase 5: exp4016 (falls back to a re-held-out level if 4014/4015 didn't advance),
               exp4017 (per-board, independent), exp4018 (aggregates WHATEVER exists)
```

**Routing (Codex-Default v2, 2026-06-10 — gemini BANNED; every `.367–`.370 gemini task stalled and
both `.369/`.370 gemini planner runs FAILed):**
- **3 opus / claude:** exp4008 (green-gate + full-suite quarantine = bootstrap-and-bail +
  multi-step infra), exp4012 (local-GGUF loading + best-of-N inference), exp4015 (5th-game
  perception planner + real-env multi-step coordination). Mirrors the `.370 precedent
  (`agent_type: claude, model: opus, requires_claude: true`), which ran on Claude in `.370.
- **8 codex / gpt-5.5:** all program-induction + execution-verification + aggregation tasks
  (exp4009, 4010, 4011, 4013, 4014, 4016, 4017, 4018). `requires_codex: true`.
- **0 gemini.** A task that would have needed gemini's 1M context (none here) waits or chunks.

---

## 4. Phase detail

### Phase 0 — infra (exp4008)
Archive `.370, activate `.371, and **keep the hardened green-gate**: parse-guard
`research-complete.yaml` + `ops/exclusion_manifest.yaml`, import the ARC agentic modules, then run
the FULL `tests/python` suite and **quarantine any red test** out of collection
(`tests/quarantine/`) until green — the load-bearing fix that stopped the `.370 cascade. Anti-
recursion: never write a test asserting on a per-run `honest_verdict` string (the poison pattern).
Record the `.370 truth (4 games, 5 levels, confirmation-not-run, local-weak).

### Phase 1 — CONFIRM the moat (exp4009, exp4010, exp4011)

- **exp4009 — precision confirmation v3 (THE owed experiment; MUST EXECUTE).** k=3 ALL-FRESH chains
  (no probe/stale arms), equal 600 s, on NEW clean chain-feasible ARC-2 tasks; protocol committed
  BEFORE any call. PRIMARY gate = the handoff's binomial critical-value rule (n>=19 agreement
  events, reject H0 p=0.52 if >=14 gold; size 0.046 / power 0.837 at p=0.80). SECONDARY = precision
  vs the in-run fresh-arm base rate (~0.73). The design closes the `.370 failure mode: an
  EXECUTION-FLOOR precondition (the artifact is INVALID unless `total_codex_calls>0` AND
  `n_agreement_events>0`) means it cannot emit `pending_execution`. `prior_failures` cites exp3999
  (`pending_execution`) with `retire_if_same_verdict: true` — a 3rd non-execution/non-confirmation
  retires the agreement-as-selector claim to confidence-label-only.

- **exp4010 — GAP-5 cross-example-consistency selector upgrade (Missing-Verifier Gap Logging
  mandate).** exp4001 logged GAP-5 (demo-underdetermination) as the open verifier gap. Build a
  STRONGER discriminator than plain output-agreement: filter induced programs by cross-demonstration
  consistency + sibling-input agreement (arXiv:2604.02434), and measure whether it lifts
  precision/coverage over output-agreement on the SAME ARC-2 pool. Closing a logged gap with a new
  registry-eligible discriminator is direct progress on Carnot's core.

- **exp4011 — feedback-vs-redraw v2 (POWERED).** exp4000 returned `p=1.0` but FALSE_NEGATIVE_RISK
  (too few discordant pairs). Re-run same-run interleaved (the 2.2x between-run variance forbids
  cross-run comparison) with enough paired tasks for power; exact McNemar/Fisher. Decides whether to
  deploy expensive feedback chains or cheaper independent redraws. `prior_failures` cites exp4000.

### Phase 2 — DECENTRALIZE (exp4012)
**Close the local induction gap via best-of-N + cheap filter (the BARC regime).** Replace exp4002's
<=3 generic-GGUF attempts with k=8–16 independent local samples per task (fast gemma-4-12B-it-GGUF
for throughput; PRECONDITION the cache — newly released), filtered by the free demo-fit verifier
(~0.11 s/task). Report: best-of-N local demo-perfect coverage (target: approach codex 0.94), gated
pass@2 vs vote 0.4516 / oracle 0.6129 / codex 0.5806, and the COST (local-generator seconds vs codex
seconds; the verifier is ~free). `prior_failures` cites exp4002 (the <=3-attempt weak arm).

### Phase 3 — EFFICIENCY (exp4013)
**The north-star §5 efficiency datum on ARC.** Head-to-head: the model-free GAP-4 verifier
(demo-fit + execution, ~0.11 s/task) vs an LLM-judge (codex/gpt-5.5) selecting from the SAME
candidate set on the ARC-2 pool. Report accuracy parity (selection agreement within CI) AND the
compute/latency ratio — the "parity at Nx cheaper" claim. No paper reports this ratio on ARC; it is
an ownable Carnot contribution and the cleanest statement of "the verifier earns its place on cost."

### Phase 4 — ARC ACCURACY (exp4014, exp4015)

- **exp4014 — break a level wall via explore-first per-level re-induction.** exp4003 held the walls
  because it re-induced WITHOUT observing per-level dynamics. Apply the exp4004 explore-first method
  (a bounded active-exploration phase observes the per-level transitions, THEN induce the rule, THEN
  verifier-validate against held-out transitions before committing actions) to break lp85/sc25 L2
  and/or r11l L4. Incremental-Progress Scoping: target +1..+n levels, STOP at the first level that
  fails; never "all levels." `prior_failures` cites exp4003.

- **exp4015 — FIFTH ARC-AGI-3 game first-solve (games 4->5).** Apply the proven explore-first
  perceive→induce→verifier-prune method to the next-easiest non-spatial game (avoid vc33's PSPACE
  trap). Real-env-confirmed. Versioned continuation of the exp4004 win (which solved su15).

### Phase 5 — mandates + capstone (exp4016, exp4017, exp4018)
- **exp4016 — ArcMemo solve-transfer v4 (self-learning mandate).** Extend the exp4005 14->10 win to
  this milestone's NEW content (the higher level from exp4014 or the 5th game from exp4015; fall
  back to a re-held-out level). Positive control: the target must share structure with >=2 banked
  concepts (else transfer is impossible — FALSE_NEGATIVE_RISK).
- **exp4017 — hardware continuity (consolidated).** SSH/USB-reachability per board (KV260, GateMate,
  PolarFire); KV260 toward terminal per north-star §3. KV260 precondition is SSH-reachability ONLY
  (never the host SD card). Distinct per-board wall-clock timers (the exp3866 TAUTOLOGY corrigendum).
- **exp4018 — capstone .371 (UNGATED).** Headline question: **is the GAP-4 verifier now CONFIRMED
  (did the precision confirmation finally execute and reach a powered answer) and decentralization-
  EFFECTIVE (did best-of-N close the induction gap)?** Plus ARC accuracy (games/levels) and self-
  learning. Aggregate whatever landed; skip any `flagged_adversarial`; cite upstream sha256.

---

## 5. Hardware requirements
- **exp4012** (local best-of-N): a cached SOTA GGUF (gemma-4-12B-it-GGUF preferred for throughput;
  gemma-4-26B-A4B fallback) + `llama_cpp`; 1–2x RTX 3090 or CPU. PRECONDITION the cache — gemma-4-12B
  is newly released (2026-06-05) and may not be cached.
- **exp4017** (hardware continuity): SSH to `kria` + `polarfire`; `openFPGALoader` for GateMate.
- All other tasks are CPU offline (cached ARC pools + the offline `arc_agi` Arcade env) or codex
  network calls; no GPU required.

## 6. Invariants carried
- `paper_ready=TRUE` (G1–G4; frozen FoVer 0.9131 NEVER substituted — `.371 adds CONFIRM/
  DECENTRALIZE/EFFICIENCY lenses to the ARC-domain verifier, not a new headline).
- Verifier math/ARC-domain-bound; facts remain earned-negative.
- Both energy theses (selection P0.1 + generation EBT) remain bounded-negative; the verifier (not an
  energy generator) is the value-add.
- Gated/required fields emitted BARE; no flagged-adversarial artifact aggregated; no external
  publication; KV260 SSH-not-SD-card; Incremental-Progress Scoping (no "all levels").
