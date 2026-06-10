# Research Roadmap — Milestone 2026.06.369

**Planned:** 2026-06-10 (outer-loop planning agent, Claude Opus 4.8)
**Milestone doc for:** `research-roadmap-next.yaml` (`milestone: 2026.06.369`)
**Prior milestone:** 2026.06.368
**North star:** `ops/north-star.md` §0 — solve ARC-AGI-3, accurately AND efficiently.

---

## 0. One-line thesis

The GAP-4 **program-induction execution verifier** produced the project's FIRST positive
verifier result on ARC content/rule selection (ARC-1 rerank vote 0.4516 → gated 0.5806,
~0.80 of the proven oracle headroom, +4/−0, 5/5 adversarially confirmed). It is
**PRELIMINARY** in five named ways. **.369 turns the preliminary positive into a CONFIRMED,
DECENTRALIZED, DEPLOYED verifier** by executing the operator's four verbatim
`conductor_followups` (de-selection coverage, pre-registered precision confirmation,
feedback-vs-redraw control, harness registration), adding the owed **local open-weight
generator arm** (the decentralization tier — no published work reports it), and continuing
the north-star accuracy push (one incremental level + a 4th game, both verifier-assisted),
self-learning (ArcMemo solve-transfer v2), hardware, and an ungated capstone.

---

## 1. What the previous milestone (.368) proved

### Thread A — the conductor's autonomous GAP-4 build FAILED; the outer-loop's SUCCEEDED

The .368 conductor task `exp3975` tried to build the GAP-4 verifier **DSL-only** and came
back `gap4_positive_control_failed_auroc0.00`, `coverage 0.0`, `llm_proposer_used=False` —
the deterministic DSL could not synthesize the rules, and the SOTA GGUF proposer was never
invoked. `exp3976` (the rerank) consequently `blocked_gate_check_failed`. **The DSL-only
path is superseded — .369 does not re-attempt it.**

In parallel, the **outer-loop session** (operator-directed, 2026-06-09/10) built GAP-4 the
right way — **codex (gpt-5.5) program-induction + a model-free execution verifier** — and
landed the program's first positives, all 5/5 adversarially confirmed
(`ops/verifier_gaps.md` GAP-4, memory `project_gap3_verifier_program`):

| Result | Number | Artifact |
|---|---|---|
| **ARC-1 rerank** (contaminated pool — upper bound) | vote 0.4516 → **gated 0.5806** pass@2 (+4/−0; ~0.80 of oracle headroom) | `results/arc3_gap4_rule_exec_verifier.json` |
| **ARC-2 transfer probe** (reduced-exposure) | induction 0.93→**0.57**, precision 0.90→**0.47**; demo-overfit asymmetry ⇒ genuine induction, not recall | `results/arc3_gap4_arc2_rule_exec_verifier.json` |
| **Precision fixes** | graded min-hamming gate production τ=0.005 (ARC-1: 0 losses); k=3 single-shot agreement precise but coverage-collapsed | `results/arc3_gap4_arc2_consistency_ensemble.json` |
| **k=3 CHAIN-arms** | agreement 10/16 entries, gold 8/10; fresh-chain per-arm 0.833; **prereg all-gold bar NOT met**; agreement = a CONFIDENCE LABEL, not a selector | `results/arc3_gap4_chain_arms_adversarial_verify.json` |

**Deployment frontier (measured offline):** graded-snap τ≤0.005 → promote-first-FRESH-chain-
raw-output → vote = ARC-2 pass@1 **19/31 (0.6129)** (vs vote 1/31), ARC-1 28/31.

### Thread B — the other .368 outcomes (the verifier on the two owed axes)

| Question | .368 verdict | Artifact |
|---|---|---|
| **Verifier earns ACCURACY?** | ❌ **No** — the conductor's DSL build failed; the real positive came from the outer-loop (codex), still PRELIMINARY | capstone exp3985 `verifier_earns_accuracy=False` |
| **Verifier earns EFFICIENCY?** | ⚠️ **Yes but n=5** — energy-consistency verifier vs LLM-judge: parity at **8789× cheaper**, verifier provably invoked — but only **5 programs judged** (underpowered) | exp3978 |
| **World-model induction (6 non-spatial games)** | ❌ **0/6** trustworthy AGAIN (second negative, positive control passed) → offline non-spatial world-model induction is **bounded** | exp3979 |
| **Incremental L1→L2** | ❌ r11l L2 wall holds; re-induction found L2 needs a **different rule** (collision-forbidden mask) — diagnosis, not a solve | exp3980 |
| **4th game first-solve** | ❌ `fourth_game_no_solve_budget_exceeded` | exp3981 |
| **Self-learning (ArcMemo solve-transfer)** | ✅ **Big win** — concept-memory-seeded solve **2668 → 17 actions** | exp3982 |
| Hardware continuity / retro detector fix | ✅ boards visible; retro commit-detector bug fixed + self-check added | exp3983 / exp3984 |

### The five ways the GAP-4 positive is PRELIMINARY (the .369 work list)

The `ops/verifier_gaps.md` GAP-4 "NOT yet ESTABLISHED" list + the 2026-06-10 handoff name
exactly five gaps, and .369 closes or advances each:

1. **Statistical significance** — ARC-1 sign test p=0.0625 (borderline); chain-arms prereg
   all-gold bar NOT met (p=0.07 vs 0.52). → **followups #1 (de-selection coverage) + #2
   (pre-registered precision confirmation)**.
2. **Feedback vs iid resampling unresolved** — does a feedback chain beat 3 independent
   singles? (post-feedback per-call 0.43 ≈ iter0 0.46). → **followup #3 (deciding control)**.
3. **Decentralization** — the lift is generator-attributable and the generator is **closed-
   weight gpt-5.5**; the verifier side is local+model-free but the inducer is not. → **the
   LOCAL open-weight generator arm** (CLAUDE.md decentralization Rule 1 + the GAP-4 forward
   protocol "local open-weight generator arm (Gemma-4/Qwen3.6)").
4. **Deployment** — the tiered policy is measured but not registered as a reusable verifier.
   → **followup #4 (harness registration + bit-exact offline re-eval)**.
5. **Demo-underdetermination (GAP-5)** — three disjoint demo-perfect programs can converge
   on the SAME wrong output; the only tripwire is task-level sibling-input disagreement. →
   **carried as followup #2's registered tertiary gate + the GAP-5 entry append in #4**.

(NOT pursued: a 400-task scale run — the handoff is explicit that it "is NOT yet worth it;
it becomes worth it only after (1)/(2)". Respected.)

---

## 2. The three biggest gaps between current state and the PRD vision

1. **The verifier's accuracy moat is real but UNCONFIRMED and CLOSED-WEIGHT.** The PRD/north-
   star require an externally-grounded verifier that works *locally* (decentralization Rule
   1). The GAP-4 positive depends on gpt-5.5 induction and a contaminated pool. .369 confirms
   the significance (followups #1–#3) and replaces the closed-weight inducer with a SOTA
   local GGUF (the decentralization arm) — without which the headline cannot be published as
   a sovereignty result.
2. **ARC-AGI-3 accuracy has stalled at 3×L1.** The method generalizes (3 games) but every
   game stalls at L2 and the 4th game ran out of budget. The PRD's accuracy axis needs
   *monotonic* level progress. .369 uses the now-working GAP-4 execution verifier IN the
   solve loop (validate a re-induced L2 rule before committing actions; prune the 4th-game
   search) — turning the verifier's efficiency value into real accuracy progress.
3. **Self-learning is proven on induction-cost and one solve, not yet as a compounding
   loop.** exp3970 (induction transfer) + exp3982 (solve transfer 2668→17) are two points.
   The PRD's continuous-self-learning goal needs the concept memory to compound across a
   *new* game. .369 extends ArcMemo to the 4th game's solve with a positive control.

---

## 3. Architecture — the GAP-4 program-induction execution verifier (and its decentralization tier)

```
                    ARC task = { demo (input,output) pairs , test_input , candidate pool }
                                              │
            ┌─────────────────────────────────┴──────────────────────────────────┐
            │                          GENERATOR (induces)                         │
            │   escalation tier:  gpt-5.5 via codex exec  ──┐                      │
            │   DECENTRALIZATION tier (NEW, .369):          ├─►  def transform(grid): ...
            │     SOTA local GGUF (Qwen3.6-35B-A3B /        │    (a program = an executable
            │     gemma-4-26B-A4B) proposes the program ────┘     hypothesis of the rule)
            └─────────────────────────────────┬──────────────────────────────────┘
                                              │  k≥1 candidate programs
                    ┌──────────────────────────┴───────────────────────────┐
                    │      VERIFIER  (Carnot's value-add — LOCAL, MODEL-FREE)             │
                    │  (1) DEMO-FIT gate: keep a program only if it exactly reproduces     │
                    │      ALL demo outputs (execution-guided acceptance, arXiv:2507.15877)│
                    │  (2) EXECUTE the demo-perfect program on test_input → predicted*     │
                    │  (3) graded min-hamming snap (τ≤0.005) of predicted* to candidates   │
                    │  (4) AGREEMENT across k fresh inductions = CONFIDENCE LABEL (not a    │
                    │      selector); sibling-input disagreement = underdetermination flag │
                    └──────────────────────────┬───────────────────────────┘
                                              │  promote-first-fresh-raw → else vote
                                       reranked candidate  →  ARC-AGI-3 solve loop / pass@2
```

**What is ESTABLISHED:** the verifier side is fully local + model-free; the gate is safe
(+4/−0 on ARC-1); agreement is a confidence label (agreement-first selection is net −1).
**What .369 adds:** independence/significance confirmation (#1–#3), a local inducer (the
decentralization arm), a registered reusable module (#4), and verifier-in-the-loop solving
(accuracy phase). Corroborated by EWM (2605.05138, induce→verify-program→plan), ABPR
(2603.20334, executable-hypothesis verification), cross-example consistency filtering
(2604.02434), and the must-beat cheap baselines self-certainty/self-consistency (2502.18581).

---

## 4. Phases

### Phase 0 — milestone transition (1 task)
- **exp3986** archive .368 → activate .369; GREEN-GATE (yaml parse, ARC substrate tests,
  ARC agentic-module imports), record the .368 truth (GAP-4 outer-loop positive; conductor
  DSL build failed; efficiency n=5; 0/6 world-model; 3×L1; ArcMemo solve-transfer win).

### Phase 1 — GAP-4 CONFIRMATORY (the operator's four verbatim follow-ups) (4 tasks)
Queued **verbatim** from `results/arc3_gap4_chain_arms_adversarial_verify.json`
`synthesis.conductor_followups` per the 2026-06-10 TOP-PRIORITY handoff.
- **exp3987 — DE-SELECTION COVERAGE RUN** (codex): k=2 fresh ≤3-iter 600s chains on the 11
  never-chained ARC-2 pool tasks; de-bias the 0.833 coverage estimate; transcripts + gold-
  leak audit; no all-gold bar.
- **exp3988 — PRE-REGISTERED PRECISION CONFIRMATION v2** (codex): k=3 ALL-FRESH chains on NEW
  clean tasks; protocol committed BEFORE any call; **primary binomial critical-value gate
  (n≥19 events, ≥14 gold ⇒ size 0.046 / power 0.837 at p=0.80)**; secondary vs in-run fresh-
  arm rate; **tertiary = task-level unanimity-with-abstention on sibling-input disagreement
  (the GAP-5 tripwire)**; `retire_if_same_verdict` on the precision-uplift claim.
- **exp3989 — FEEDBACK-VS-REDRAW DECIDING CONTROL** (codex): same-run paired — one feedback
  chain vs 3 independent singles, equal 600s, interleaved in ONE run; exact McNemar/Fisher;
  resolves whether feedback content beats iid resampling.
- **exp3990 — HARNESS REGISTRATION + OFFLINE TIER-STACK EVAL** (claude, CPU, zero codex):
  register `gap4_program_induction_stack` in `ops/verifier_registry.yaml`; reusable module;
  **bit-exact offline re-eval must reproduce ARC-2 19/31 and ARC-1 28/31**; append the
  446ef5d2 demo-underdetermination GAP-5 entry to `ops/verifier_gaps.md`; fix the committed
  cost line. (Handoff said gemini — gemini is BANNED this milestone, route to claude/CPU.)

### Phase 2 — DECENTRALIZATION (the owed deployment tier) (1 task)
- **exp3991 — LOCAL OPEN-WEIGHT GENERATOR ARM** (opus, GGUF): replace the gpt-5.5 inducer
  with a SOTA **local GGUF** program proposer (`Qwen3.6-35B-A3B` / `gemma-4-26B-A4B-it`,
  loaded via the `.gguf` path per the GGUF tokenizer rule). Measure local induction
  demo-perfect rate + the gated rerank on the SAME ARC-1 pool (reproduce the venue) + the
  **cost** (local-GGUF seconds vs codex seconds vs the model-free verifier seconds — the
  decentralized efficiency datum that strengthens exp3978's n=5 result). CLAUDE.md
  decentralization Rule 1 + the GAP-4 forward-protocol "local open-weight generator arm";
  no published work reports local-open-weight ARC program synthesis with execution-verifier
  reranking — a genuinely novel, ownable number.

### Phase 3 — ARC-AGI-3 ACCURACY (verifier-in-the-loop; monotonic progress) (2 tasks)
- **exp3992 — INCREMENTAL LEVELS via verifier-validated re-induction** (codex): exp3980
  diagnosed r11l L2 needs a different rule; RE-INDUCE the L2 rule from L2 observations and
  **use the GAP-4 execution verifier to validate the candidate L2 rule against L2 transitions
  BEFORE committing actions** (the verifier's efficiency value applied to a real solve).
  Target +1 level on ONE game (Incremental-Progress Scoping). `retire_if_same_verdict`.
- **exp3993 — FOURTH GAME first-solve, verifier-pruned** (opus): exp3981 ran out of budget;
  use the GAP-4 verifier as an action-pruner (verifier-as-free-energy, the Exp1165 ~4×
  precedent) + pick the empirically-easiest non-spatial game by L0 budget. Real-env-
  confirmed; raises games-solved 3→4.

### Phase 4 — self-learning + hardware + capstone (3 tasks)
- **exp3994 — ArcMemo SOLVE-transfer v2** (codex): extend exp3982's win — does the banked
  concept memory make the 4th game's solve cheaper than cold-start? Two arms at equal
  perception; positive control (≥2 same-family games). Self-learning MANDATE.
- **exp3995 — Hardware continuity** (codex): KV260 (`ssh kria`, toward terminal per north-
  star §3) + GateMate + PolarFire reachability; distinct per-board timers (exp3866 tautology
  corrigendum); SSH-not-SD-card.
- **exp3996 — Capstone .369** (codex, UNGATED): headline question — is the GAP-4 verifier now
  **CONFIRMED** (significance from #1–#3), **DECENTRALIZED** (local arm), and **DEPLOYED**
  (registered)? Plus ARC accuracy (games + new levels) and ArcMemo solve-transfer. Skip any
  `flagged_adversarial` artifact; cite upstream sha256; aggregate whatever exists.

---

## 5. Dependency graph

```
exp3986 (archive/activate, green-gate)
   │
   ├─► exp3987 de-selection coverage ─────────────┐
   ├─► exp3988 precision confirmation v2 ──────────┤
   ├─► exp3989 feedback-vs-redraw control ─────────┼─► exp3990 registration + offline re-eval
   │        (the 3 codex confirmatory arms feed #4's tier-stack + GAP-5 entry)
   │
   ├─► exp3991 LOCAL generator arm (GGUF; reuses the model-free verifier from the gap4 artifacts)
   │
   ├─► exp3992 incremental L2 (verifier-validated re-induction)
   ├─► exp3993 fourth game (verifier-pruned) ──────► exp3994 ArcMemo solve-transfer v2
   │                                                     (held-out target = the 4th game)
   ├─► exp3995 hardware continuity
   │
   └─► exp3996 capstone .369  (UNGATED — aggregates whatever landed)
```

No hard prerequisite is allowed to cascade-block: every confirmatory arm runs independently
and the capstone is ungated by design (the .365 op:exists + .366 no-artifact lessons).

---

## 6. Routing (gemini BANNED — every .367/.368 gemini task stalled; `incident_333` quota crash)

| Tasks | Agent | Why |
|---|---|---|
| exp3987 / exp3988 / exp3989 | **codex (gpt-5.5)** + `requires_codex` | the program INDUCER *is* gpt-5.5 via codex exec — these tasks generate programs |
| exp3986 / exp3990 / exp3992 / exp3994 / exp3995 / exp3996 | **codex (gpt-5.5)** + `requires_codex` | mechanical/aggregation/registry/hardware; codex is the available formulaic backend (exp3990 is CPU-only, no induction) |
| exp3991 / exp3993 | **claude opus** + `requires_claude` | LOCAL GGUF supervision + real-env solve = anti-fabrication, multi-step tool choreography, high judgment |

2 opus (exp3991/exp3993) + 9 codex. No gemini.

## 7. Hardware requirements

- **codex (gpt-5.5)** quota for the 3 program-induction confirmatory arms (~3.5k + ~90–135
  calls + a same-run paired control; ≥600s timeouts per the handoff hygiene rules).
- **2× RTX 3090 / local GGUF cache** for exp3991 (the local program proposer —
  `Qwen3.6-35B-A3B-GGUF` / `gemma-4-26B-A4B-it-GGUF`, PRECONDITION the cache; load via the
  `.gguf` path, never `AutoTokenizer` on the GGUF repo id).
- **ARC offline env** (`environment_files/` present; `arc_agi` SDK importable) for the solve
  + transfer tasks.
- **FPGA boards** (KV260 via `ssh kria`, GateMate via `openFPGALoader -c dirtyJtag --detect`,
  PolarFire via `ssh polarfire`) for the continuity task.

## 8. New references incorporated (filed in research-references.md, 2026-06-10 scan)

- **2605.05138 EWM** — the published induce→verify-program→plan instance on ARC-AGI-3 (closed
  GPT-5.x → local arm is an open differentiator).
- **2603.20334 ABPR** — executable-hypothesis verification via algorithmic debugging (richer
  GAP-4 signal; a future gap entry for execution-trace disagreement).
- **2604.02434** — cross-example consistency (consensus) filtering = a concrete agreement-
  selection method + GAP-5 underdetermination remedy.
- **2506.18203 Weaver** — label-free weak-verifier ensembling + distill-to-400M efficiency.
- **2502.18581 self-certainty** + **2509.19681 calibrated reasoning** — the must-beat cheap
  baselines + cost-at-matched-accuracy evidence for the efficiency axis.
