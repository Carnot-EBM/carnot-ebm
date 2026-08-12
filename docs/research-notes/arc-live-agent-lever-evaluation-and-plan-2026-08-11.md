# ARC live agent: lever evaluation and new-approach plan (2026-08-11)

**Status:** RESEARCH SYNTHESIS. This note changes no live-path file. It answers the
2026-08-11 operator ask: evaluate what we have tried, then propose new approaches to
raise the submission score.

**Score record.** Best on record: 0.12 (old Qwen3.5-9B stack, single L4, pre-2026-07-28).
Current stack first clean data point: 0.09 (gemma-4-31B-qat, RTX PRO 6000, 2026-08-11).
The two numbers come from different stacks. They are not a paired comparison
(`ops/known-issues.md` 2026-08-11).

**New facts from 2026-08-11 (both change the option space):**

1. The scored card is ONE RTX PRO 6000 Blackwell with 96 GB VRAM. It is not 4x L4.
   The generator loads, MTP engages, and 4 parallel requests hold with 3.3x VRAM
   headroom. All prior live-stack sizing assumed a 16-24 GB class card.
2. Kernel v14 added a preview-mode diagnostics channel. We can now run measurement
   code on the exact scored card for free. A preview run costs no submission slot.

## Part 1 — what we tried, with verdicts

Every verdict below was read from the artifact's own `honest_verdict` field, not from
memory. Sources: `docs/research-notes/arc-lever-triangulation-2026-07-23.md` and the
named result files.

### Dead lever classes (do not re-propose without a new mechanism)

| Lever class | Evidence | Verdict |
|---|---|---|
| Candidate ranking / selection | 7-9 A/Bs: tier schedule, small-object-first, frame-change scorer, object-history prior (exp5740), structural-energy rerank. Two more members (exp4556 router, exp4617 value head) agree but carry `flagged_adversarial: true` — cited here as corroboration only, never as load-bearing evidence | Zero live levels moved. Class dead on the unflagged members alone. |
| Search depth / lookahead | Winner audit 2026-07-20: all Milestone-1 winners are greedy. Lookahead signal 0/247 winning actions across two runs | Class dead. Carnot already out-searches the winners. |
| Bigger dense generator alone | exp5722: 31B swap moved 0 live levels. exp5764: 31B held-out induction 0.378 (need near 1.0 for a 14-step plan) | Model size alone does not clear the wall. |
| /think reasoning toggle | exp5714: inert under code-only prompts. exp5726: more completions, no level-up | Null on progress, on the tested stacks. |
| Trajectory retrieval (MATM-style) | actions-to-progress metric: null | Null on progress. |
| Blanket REx refinement | exp6248: 2 of 6 games improved, gate not met | Retired as a universal lever. |
| Counterexample transition patcher | exp5641 | Retired (exclusion manifest). |
| Depth/admission portfolio levers | exp6215/6216/6229/6230/6231 in exp6232 ledger | A/A-identical or never fired. No admissible depth lever. |

### What stands (the confirmed constraints)

1. **Dynamics induction quality is the binding constraint.** The agent must compose a
   13-33 action sequence. That needs near-perfect induced dynamics. Best measured:
   0.378 held-out change-fidelity (exp5764). 0.378^14 is near zero.
2. **The goal-gradient wall is a SECOND, independent gate.** Even perfect dynamics
   cannot plan without a signal to climb. The goal-energy zero-gradient wall
   (2026-08-09 entry) blocks the CNN tier and likely the nav template too.
3. **Per-game wins are real even where blanket wins are not.** exp6248's ka59 arm
   crossed the live trust threshold (0.9792 vs 0.3125) at equal budget.
4. **VALID-score arm selection looks predictive.** Retrospective: 6/6 on exp6248 data.
   Prospective (exp6250, running now): 2/2 games matched so far. Honest read: one
   informative match (lp85, 0.1096 vs 0.0171) plus one tie (dc22, both arms 0.5), so
   the prospective evidence is 1 real hit, not 2.

## Part 2 — proposed new approaches

Ordered by expected information per GPU-hour. Each entry names its prior-failure
framing so the exclusion-manifest lint can pass it.

### P1. Finish and wire the best-of-both ensemble (in flight)

exp6250 is running. If its gate passes, wire `run_rex_ensemble` into live induction
as default-OFF, then run the live shadow A/B. Prior-failure framing: exp6248 retired
blanket REx; the ensemble is the per-game selector its retirement note flagged as
unanswered. Gate: ensemble pooled held-out >= best pure arm on the fresh roster.

### P2. Generalize selection to best-of-N induction (new axis: samples per budget)

The ensemble picks between 2 search shapes. The same VALID-score selector can pick
among N independent induction samples. The scored card holds K=4 concurrent requests
today (v14 proved they SUCCEED; per-stream latency under batching is NOT yet measured
— P5's harness must measure it before "4 for the price of 1" is assumed). This is NOT
the dead ranking class: it selects among GENERATED PROGRAMS with an oracle-distinct
VALID score, not among perceptual click candidates. Gate: pooled held-out fidelity at
N=4 beats N=1 at matched wall-clock on >= 3 of 4 fresh games.

### P3. Replace the BINARY induced goal with a GRADED progress potential

CORRECTED after adversarial review: binary LLM goal induction ALREADY EXISTS. The
live reinduction path has asked the LLM to write `is_level_complete` as code since
milestone .430 (`arc_llm_reinduction.py`, with degenerate-predicate rejection), and
non-LLM goal-predicate induction succeeded offline (exp4020, held-out precision
1.000, wired into `arc_goal_energy_live.py`). That binary predicate is exactly the
mechanism behind the zero-gradient wall: a 0/1 goal gives search nothing to climb
until it is already at the goal.

The NEW sliver: induce a GRADED `progress(grid) -> float` potential, verified
against observed `level_progress` deltas (the live signal the static corpus lacked),
and use it as the search potential. Prior-failure framing must name the full
lineage: the .430 binary `is_level_complete` line (frequently degenerate), exp4020's
binary offline success, the 2026-08-09 zero-gradient wall — and exp5641 (retired,
but it patched DYNAMICS, not the objective). Gate: the graded potential ranks true
level-up states above non-progress states AND assigns strictly increasing values
along >= 1 known winning trajectory, on held-out frames, for >= 3 of 4 games.

### P4. Re-size the live config for the real 96 GB card

Every live-stack constant was tuned for a 16-24 GB assumption. Now measurable for
free via the preview channel: raise slots beyond 4, raise induce max_tokens, and
re-test /think ON the scored-class hardware. External evidence (GPT-5.6 ARC-3:
about 26x score from reasoning effort) directly challenges the frozen /no_think.

The /think prior-failure ledger is longer than two entries. Full list, required for
the rerun-discipline block: exp5594 (null), exp5714 (Qwen3.5-9B, inert under
code-only prompts), exp5726 (ThinkingCap-27B dual-GPU, completions up, no level-up),
exp6229 gemma-31B think determination (`blocked_gate_check_failed`, never measured),
and a pre-gate block on the native gemma-31B think A/B landed 2026-08-11 (commit
a9d4b139). So NO /think null exists on the current 31B generator — the prior nulls
are other models, and the one 31B attempt was gate-blocked, not negative. The real
differentiator here is the scored-card preview substrate plus the unmeasured-on-31B
status, not hardware alone. Gate: any config change must beat the current config on
held-out induction fidelity in a preview-mode A/B before it ships in a submission.

### P5. Build the preview-channel A/B harness (instrument, not lever)

Wrap P2/P4 style induction A/Bs into the kernel's preview branch: run on public-game
transitions, print held-out fidelity, exit. This measures on the exact scored
substrate at zero submission cost. Public-game data only (source-reading stays
dev-tier per the Live-Path Reachability discipline).

### P6. Ask the operator: MoE-many vs dense-few at matched wall-clock

Qwen3.6-35B-A3B is EXPECTED to decode much faster than dense 31B (about 3B active
params; not yet measured on this rig — measure before deciding). More samples per
budget could beat fewer, better samples once P2's selector exists.
The generator pin is operator-settled (2026-07-28, gemma won 11-0-2 per-sample), so
this is a question, not an action: does the operator authorize a matched-wall-clock
samples-vs-quality A/B? The 11-0-2 result measured per-sample quality, not
best-of-N-per-second. That is a genuinely different question.

## Sequencing

1. exp6250 finishes (today) -> P1 wiring decision.
2. P3 goal-induction pilot offline (dual 3090s, no submission dependency).
3. P5 harness, then P4 preview A/Bs on the scored card.
4. P2 best-of-N once P1's selector is validated.
5. P6 goes to the operator as a yes/no.

Submissions stay gated on offline wins per `feedback_arc3_online_gated_on_offline_
beating_baselines`: go live only when an offline result beats the current baseline.
One submission per UTC day is the hard external limit.
