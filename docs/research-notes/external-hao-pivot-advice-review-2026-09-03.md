# External advice to abandon runtime induction for a hypothesize-act-observe loop: reviewed, not adopted

Written 2026-09-03. An external reasoner (Google Deep Think) was asked five architecture
questions about the live ARC-AGI-3 agent. It answered that the program is "structurally
misconceived" and recommended replacing explicit runtime world-model induction with a
hypothesize-act-observe (HAO) loop. An adversarial review checked every load-bearing claim
against this repository. The central thesis does not survive. Two of its subsidiary findings do.

This note exists so the pivot is not re-proposed from the same reasoning in three months.

## What was recommended

Rank-ordered by the external reasoner: (1) state-delta filtering with run-length encoding of the
induce prompt, (2) an "Oracle Trace Test" feeding 3-5 hand-curated transitions to the induce
prompt, (3) supervisor novelty-credit plus an unconditional fallback arm, (4) pivot from explicit
induction to HAO, (5) synthetic out-of-distribution validation games, described as "zero direct
agent EV". Its single recommended measurement was the Oracle Trace Test, to be run immediately.

## The three factual errors

**1. The truncation event was read as attention failure. It is a measured arithmetic wall.**
The opening thesis cites our 19,142-token generation as "a classic symptom of autoregressive
repetition and attention failure". Our own controlled probe says otherwise: a 30,010-token prompt
at `-c 49152` generated exactly 19,142 tokens with `truncated=true`, and prompt+generation summed
to 49,152 — the pool to the token. The SAME prompt at `-c 98304` ran its full 21,000-token budget
with `truncated=false`. A content pathology does not stop at an exact arithmetic boundary and then
vanish when the boundary moves. The mechanism is named in `arc_executable_world_model.py`
(`_limit_diagnostic`, "SHARED-POOL TRUNCATION") and in the server (`ctx_shift=false`).

Fair to the reasoner: repetition WAS a real induce defect here, measured and mitigated by a wired
`repeat_penalty` on 2026-07-31. The prior is reasonable. Its application to this event is wrong.

**2. "Catastrophically failing the budget constraint" was asserted without the budget, and the
actions cited did not come from the tier being blamed.** All 9 eval artifacts ran `--budget 2500`;
the run banked 2 levels in 2,121 raw / 2,311 charged actions, inside its budget. Efficiency versus
human IS poor and nobody disputes it (level 1: 885 charged against a human 22; level 2: 125
against 33; scorer efficiency 0.689). But the run made only 42 LLM completions across 2,121
actions, so roughly 98% of those actions came from the mechanical explore/act/observe tier
(`StepwiseExplorer`) — the closest thing to HAO we already run. The thrash offered as evidence
against induction is HAO's own failure mode.

**3. The AVO precedent is misstated, and correctly read it argues the other way.** The advice says
NVIDIA "chose HAO despite having access to frontier models", implying a tested architectural
choice. They ran HAO *on* a frontier reasoner, with no published ablations and no cost data. There
is no evidence they trialled and rejected explicit induction; intent is being read into an absence
of ablations. The deeper problem: every HAO step is a fresh zero-shot hypothesis call, which is
the operation the advice itself argues a Q4 27B is worst at. The precedent cautions against a
naive HAO pivot at our scale rather than mandating one.

## Why "pivot to HAO" is a change of degree, not kind

`E3AgentPolicy` already runs EXPLORE (collect transitions from its own play) then INDUCE then
VERIFY (`WorldModelVerifier`) then PLAN then EXECUTE, returning to EXPLORE on divergence. The
"a hallucinated constraint becomes ground truth" argument is answered by the verify gate, which
measurement shows working: it rejected memorizing engines in 9 of 11 dynamics failures, and 20 of
25 rounds never reached the planner at all. The verify gate is the one component our data shows
functioning as designed, and the pivot would discard it.

What IS fairly attacked: the induce step is a single global synthesis over ALL transitions since
level-start (`k=None`, 220-299 transitions, 30-46k tokens). That is real, measured, and worth
fixing — as a bounded A/B on transition selection, not as an architecture replacement.

## Two dismissals our own corpus contradicts

The advice dismissed two counter-arguments it had itself raised. Both are present-tense here.

- **Deduplication destroys temporal rules.** sk48 is built from counter mechanics: a 42-step
  counter decrementing 2 per keypress, rings that reset it, pinwheel tiles that cycle player
  colour one step per entry, and a winning line with near-zero counter slack. tu93 has
  parity-toggling reset state; ft09 a local constraint colour-cycle. Deduplicating on
  `(action, state_delta)` and keeping first occurrence destroys exactly the evidence these rules
  live in.
- **Novelty credit is gameable.** The registry documents decoy sockets, decoy pads and dotted
  decoy glyphs across several games, and sk48's pinwheels are literally state-cycling tiles that
  generate endless novel grid hashes while draining a death counter. A novelty-credited supervisor
  arm would score that loop as success.

## What survives

Two real wounds, both of which this project had already diagnosed independently:

- **Supervisor credit assignment is broken, and our own artifact proves it.** All three r11l
  redirects, at charged actions 120, 240 and 360, were credited by the single level-up at 885
  (120+765 = 240+645 = 360+525). One level-up credits every pending redirect, so per-arm `helped`
  is unusable as an absolute rate. The proposed unconditional fallback arm maps onto a trigger
  already written in `arc_trajectory_supervisor.py`: a stagnation with no eligible arm is the
  written specification for a human to propose a new one. r11l logged 14 of them.
- **Unbounded transition growth is real.** 25 transitions is 6,589 tokens, 82 is 20,431, live
  inductions cite 220-299. Right conclusion, wrong mechanism, and the specific dedupe design is
  unsafe per the section above.

Also conceded: the public 25 share an ontological manifold and leave-one-game-out overestimates
transfer. Our own numbers say so — 183 levels with hand-built per-game knowledge against 5-7
adapter-free live levels across 9 games. Mechanics-absent synthetic games would be a genuinely
stronger probe than our current mechanic-preserving variants.

## What the evidence actually says the bottleneck is

Not context collapse. At 25 transitions — about 6,589 tokens, no context pressure at all — induced
engines reach 100% prefix accuracy and 50-75% held-out. That is memorization, and it is the
largest single failure class. Separately, 14 of 21 stored goal predicates never fire on a real
win. Both failures occur at prompt sizes far below any wall. "Lost in the middle" appears nowhere
in our measurements; it is an imported prior.

## The Oracle Trace Test is asymmetric

A failure would be close to decisive. A success would not: the trace is hand-curated to isolate
mechanics perfectly, a condition no achievable runtime filter reproduces, so success proves a
ceiling rather than a reachable operating point. It also has an instrument defect — with 3-5
transitions our held-out split yields 1-2 rows, and a 1-row held-out gate is vacuous. Run today it
would be doubly confounded, because its curated arm dodges the context wall its live comparator
still hits, so any gap it measured would be the wall.

Worth running, cheap, but AFTER the context fix, scored against a large separately-collected
held-out set rather than the internal split, and paired with a live-prompt arm on the same game.
As a generator diagnostic on public games it is permitted; it must never emit a counted solve or a
`live_agent_self_discovery` stamp.

## Corrected ranking

1. Verify `CARNOT_ARC_INDUCE_N_CTX=98304` end-to-end on a live induce — confirm the engine field
   is emitted and `chars_final > 0`, then measure held-out accuracy of fresh inductions. Every
   other measurement is confounded until this lands.
2. Transition-selection A/B: cap k, keep the K most recent, deduplicate only with preserved counts
   AND an exemption for transitions touching counter, toggle or parity state. Note the decode side
   is the larger cost term (median 62,490 tokens, 97.6% reasoning); prompt filtering attacks the
   smaller one, but it is what removes the truncation trigger.
3. Supervisor: unconditional fallback arm, plus a within-window novel-state delta as a SECONDARY
   diagnostic only, never sole credit, guarded against the documented decoy mechanics.
4. Modified Oracle Trace Test, post-fix, paired, externally scored. Low expectations — stuffing
   curated win exemplars into the prompt already measured negative on dynamics.
5. One or two mechanics-absent synthetic games before submission. Not zero-value: it is the
   expected-hidden-score estimate for November, which is the decision variable for an irreversible
   submission.

HAO pivot: rejected as a rewrite. The actionable residue is rebalancing an architecture that
already has both tiers — cheaper incremental hypothesis probes when the verify gate keeps
rejecting global inductions — not replacing the induction whose verify gate is the component that
works.

## Internal contradictions in the advice, recorded because they are instructive

The ranking puts filtering first while naming a test that decides whether induction survives at
all; if that test can moot the work, it belongs first. The thesis calls collapse "mathematically
guaranteed" and then proposes a test to find out, which reveals the confidence as rhetorical. The
HAO argument concedes that HAO without a global model thrashes, then refutes that concern using
thrash produced by our HAO-like tier. And synthetic validation is called "zero direct agent EV"
one section after being described as the thing that tells us whether our expected hidden score is
zero.

## Limits of this note

One external consultation, one reviewer, one repository. The review verified claims against code
and artifacts but did not run new experiments. Nothing here establishes that runtime induction is
the RIGHT architecture — only that the case made for abandoning it rests on a misread probe, a
budget figure nobody looked up, and a blog post with no ablations. The bottleneck our own data
points at, memorization and broken goal predicates at small prompt sizes, is untouched by either
architecture and remains the open problem.
