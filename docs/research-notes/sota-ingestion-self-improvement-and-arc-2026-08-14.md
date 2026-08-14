# SOTA ingestion: held-out-gated self-improvement, and where ARC-AGI-3 actually stands

Date: 2026-08-14. Channel: `WebSearch` + `WebFetch`, sequential, per the SOTA-Ingestion Cycle
Discipline. `/deep-research` was not used and must not be.

Two tracks, run because the day's work built a held-out-gated selection loop for the ARC agent and
neither track had been ingested against that specific question. The existing ARC corpus is fresh to
arXiv 2608.12xxx and the two ARC watch notes were updated 2026-08-13, so a generic rescan would
have re-found what is already there.

---

## Finding 1 (LOAD-BEARING): the ARC-AGI-3 field moved, and one of our beliefs is now false

**Claude Opus 5 scores 30.2% on ARC-AGI-3.** GPT-5.6 Sol is second at 7.8%, Claude Opus 4.8 third
at 1.5%. Snapshot dated 2026-08-14.

| rank | system | score |
|---|---|---|
| 1 | Claude Opus 5 | **30.2%** |
| 2 | GPT-5.6 Sol | 7.8% |
| 3 | Claude Opus 4.8 | 1.5% |
| 4 | GPT-5.6 Terra | 0.8% |
| 5-11 | GPT-5.5, Gemini 3.1 Pro, Grok 4.5, GPT-5.4, GPT-5.6 Luna, Opus 4.7, Grok 4.20 | 0.4% and below |

**Carnot's hidden-set score is 0.08.** So a general-purpose frontier model, called through an API
with no ARC-specific harness, scores roughly **four times** what this project's purpose-built
harness scores.

**This refutes `project_arc_leaderboard_leaders_are_source_reading`**, which records that "Carnot
0.08 hidden is comparable-to-ahead of the legit field." That was true when Sol's 7.8% topped the
board. It is not true now. The memory needs correcting, and the strategic read that followed from
it needs revisiting: we are not near the frontier, we are well behind it.

**Sourcing honesty.** The primary leaderboard at `arcprize.org/leaderboard` renders its table in
JavaScript and `WebFetch` cannot read it. The 30.2% figure comes from two independent aggregators
that agree exactly on all eleven entries (`benchlm.ai/benchmarks/arcagi3`,
`theresanaiforthat.com/benchmark/arc-agi-3/`), one of which states it "mirrors the published score
view" and describes the tasks as "private interactive tasks with public aggregate results."
Two agreeing secondaries is better than one and is still not the primary. **Confirm against the
rendered leaderboard before this number is cited anywhere public.**

**What follows if it holds.** The gap is not obviously in the harness. This project's live
generator is frozen to `gemma-4-31B` (an open, ~31B model, chosen deliberately for
decentralization). Opus 5 is a frontier model. A 4x gap between a bespoke harness on a small open
model and a bare API call to a frontier model is most simply read as a **generator-capability**
gap, not a scaffolding gap — which is uncomfortable, because scaffolding is what this project has
been building. It does not mean the scaffolding is worthless; it does mean the headline claim
cannot be "our harness competes" while the number says otherwise. Worth an explicit operator
decision rather than a quiet re-scope.

---

## Finding 2: the self-improvement literature converged on the architecture built today

Four 2026 papers describe held-out-gated, regression-guarded self-improvement loops. This is the
same shape as `scripts/arc_bench.py` + `scripts/arc_flag_ledger.py`, arrived at independently.

| paper | mechanism | gate |
|---|---|---|
| **GRASP** (arXiv:2605.29668) | edits to a bounded skill library | net improvement on a balanced held-out probe under a **hard regression budget** |
| **RSEA** (arXiv:2606.28374) | rewrites all three agent layers from its own trajectories | **strict keep-better**: accept as working state if it does not regress; update frozen best only on strict improvement |
| **Regimes** (arXiv:2606.10241) | typed transform at a seam chosen by failure diagnosis | explicit **regression-bounded acceptance rule** |
| **Self-Harness** (arXiv:2606.09498) | agent optimises its own operating framework | verifier-grounded, regression-gated (already in memory) |

**GRASP's ablation is direct corroboration of today's diagnosis.** They report that skill
acquisition **"without validation is no better than using no skills."** That is the same claim as
this project's measurement: 101 `CARNOT_ARC_*` flags, 13 of the last 16 ARC tasks ending
`ready_no_solve_claim` or `default_off`, and nothing recording which were on or why. Generation
without selection produces nothing. Two independent lines of evidence now say so.

**GRASP's headline**: gpt-oss-120b from 40.6% to 88.8% on MedAgentBench, beating the strongest of
five self-improvement baselines by 21.0 points; other models gained 17.2 to 40.3 points.

**GRASP's limitation is aimed straight at us**: the mechanism improved agents on three of four
non-clinical environments and **"remain[ed] flat only where the action space is open-ended."**
ARC-AGI-3 is an open-ended action space. So the closest published result to our setup predicts a
flat outcome in our setup. That is a prior worth holding while the 20-flag sweep runs.

---

## Finding 3: our design shares RSEA's unsolved hole, and nobody has fixed it

The question this project actually needs answered is **held-out reuse**: we select flag after flag
against the same 25 games, so each promotion is another selection event on one small set.

RSEA does the same thing and does not solve it. Verified by reading the paper:

- a single validation split `D_v`, constant across all `G` generations, **no rotation or
  resampling**
- adaptive overfitting / held-out exhaustion is **not discussed** — a real gap, given many
  candidate rewrites are scored against the same data
- the authors' own stated limit: **"The strict gate guarantees safety on held-out val, not on
  every test draw when val is small."**
- and: "The transfer benchmarks use a single seed of the shuffled split and modest eval sizes."

So the state of the art is: name the risk, do not mitigate it. We are not behind here.

**One thing we have that the published loops do not.** `arc_bench.select()` already rotates
through the roster with a signature that resets the offset when the roster changes. It was built
for cost — cheap steering every milestone, full sweep on promotion — but it is exactly the
freshness mechanism RSEA lacks. Promoting it from a cost optimisation to a stated
anti-overfitting device is nearly free and would be a genuine, if small, improvement on the
published designs.

**The sharper nuance, which the literature does not draw.** Adaptive overfitting is normally a
*noise* phenomenon: repeated selection on a finite sample fits the sample's randomness. Our
benchmark is **deterministic** — dc22 spends exactly 11,737 actions on every run — so there is no
sampling noise to fit. A flag that improves the benchmark genuinely improves the agent *on those
games*.

The risk is therefore different in kind and identical in consequence: we would be selecting for
**public-game-specific behaviour that does not transfer to hidden games**. Not statistical
overfitting, but corpus overfitting. The mitigation is the same (rotation, and treating the number
as a proxy), and the honest framing is different — worth writing into `arc_flag_ledger.py` so the
next reader does not reach for a p-value that would not mean anything here.

---

## Bottom line for the roadmap

1. **Correct the stale memory.** `project_arc_leaderboard_leaders_are_source_reading` says we are
   comparable-to-ahead of the legit field. At 30.2% versus 0.08 we are not. High priority: it is
   load-bearing for how this project talks about itself.
2. **Verify 30.2% against the rendered primary leaderboard** before it is cited publicly. Two
   agreeing aggregators is not the primary source.
3. **Hold GRASP's open-action-space prior** while the 20-flag sweep runs. The closest published
   result predicts flat. If the sweep promotes nothing, that is corroboration, not failure.
4. **Promote rotation from cost optimisation to a named anti-corpus-overfitting device**, and
   write the deterministic-benchmark nuance into the ledger docstring. Small, free, and ahead of
   the published designs.
5. **Open the generator question.** A 4x gap to a bare frontier API call reads as generator
   capability, not scaffolding. That collides with the decentralization constraint (rule 1:
   local-first, open weights) and is an operator decision, not an agent one.

**Cross-references:** arXiv:2605.29668 (GRASP) · arXiv:2606.28374 (RSEA) · arXiv:2606.10241
(Regimes) · arXiv:2606.09498 (Self-Harness, `reference_self_harness`) ·
`benchlm.ai/benchmarks/arcagi3` + `theresanaiforthat.com/benchmark/arc-agi-3/` (the two agreeing
aggregators) · `arcprize.org/leaderboard` (primary, unread — JS-rendered) ·
`scripts/arc_bench.py`, `scripts/arc_flag_ledger.py` (the loop this ingests against) ·
`project_arc_leaderboard_leaders_are_source_reading` (the memory this refutes) ·
`project_arc_live_generator` (the frozen gemma-4-31B stack Finding 1 bears on).
