# Recursive self-improvement levels, and what they say about our fitness target — 2026-09-14

This note records findings from a recursive-self-improvement (RSI) survey
paper, maps this project's autoresearch mutation round (REQ-AUTO-019
through REQ-AUTO-024) onto its autonomy-level framework, and uses its
Headroom-Closed Index to explain a pattern already observed in eight real
production autoresearch fires this session: `accepted > 0` but
`committed = 0` every time after the first round.

Primary source: "The Last AI Built by Humans: Toward Genuine Recursive
Self-Improvement", arXiv 2609.11873 (https://arxiv.org/abs/2609.11873),
33 authors, submitted 2026-09-10.

## Part 1 — what the paper establishes

**Definition.** RSI is "an autonomous, closed-loop process in which an AI
system identifies its own limitations, develops and validates
improvements, and uses the resulting capabilities to improve the
improvement process itself." Three axes: autonomy (how much of
identify-improve-validate the AI owns), efficiency (validated improvement
per unit of resource), innovation (search beyond human-prescribed
strategies).

**Five autonomy levels**, minimal to maximal:

| Level | What the AI owns | What stays fixed | Paper's example |
|---|---|---|---|
| L1 Execution | Runs a prescribed procedure | Everything else | FineWeb-Edu applies human-defined quality labels |
| L2 Strategy | Diagnoses weaknesses, picks the method | Objective, evaluator | Self-Harness edits agent harnesses under a fixed benchmark |
| L3 Experience-acquisition | Decides what to practice | Objective, evaluator, method space | SIMA 2 generates its own practice tasks |
| L4 Environment-adaptation | Revises persistent state from deployment feedback | Governance, oversight | PANDO admits/demotes rules from observed outcomes |
| L5 Recursive inheritance | Revises the improvement mechanism itself | Almost nothing | A-Evolve-Training rewrites its own research policy round to round |

**Headroom-Closed Index (HCI).** Normalizes a benchmark score against the
90th-percentile frontier score in that benchmark's entry year:
`HCI = 100 * (score - F_0) / (100 - F_0)`. HCI 0 = entry-year frontier,
HCI 100 = perfect. By 2026, knowledge-heavy benchmarks are nearly closed
(advanced math 86.4, graduate science 85.8) while interactive/open-ended
ones lag hard (software engineering 52.6, tool agents 39.9). The point:
narrow, well-specified benchmarks close fast; open-ended ones don't, and a
single aggregate "AI progress" number hides this split.

**Three named blockers to genuine RSI**, each with a measured incident:

1. **Safe inheritance** — persistence doesn't guarantee the change was
   good. Gödel Agent trials ended below the starting point 14% of the
   time. Fix: never inherit without a real improvement check.
2. **Autonomy attribution** — a better candidate doesn't prove a better
   *search* mechanism if the search's own rules are still fixed. Darwin
   Gödel Machine lifted SWE-bench 20% to 50%, but "archive maintenance and
   parent-selection rules remain outside self-modification" — that's
   still L2, not L5, whatever the score says.
3. **Reliable verification** — repeated access to the same evaluator
   rewards gaming it over actually improving. Fix: frozen evaluators,
   matched compute budgets, independent ground truth the candidate never
   sees.

## Part 2 — where our autoresearch round actually sits

Carnot's mutation round (`scripts/autoresearch_conductor_round.py` +
`python/carnot/autoresearch/orchestrator.py`) is **L2, Strategy
Autonomy**, cleanly, and should be described that way rather than as
generically "self-improving":

- Fixed by us, never touched by the AI: the benchmark set
  (`double_well`, `rosenbrock`), the 3-gate evaluator, the accept/reject
  policy, the retry budget (REQ-AUTO-023), which generator gets tried
  first (codex, then Fable).
- Owned by the AI: the actual optimization method per hypothesis. Eight
  real production rounds this session proposed L-BFGS-B, plain Adam,
  Levenberg-damped Newton, and Polak-Ribière conjugate gradient — a real,
  if narrow, strategy search.
- This is structurally the same shape as the paper's own Self-Harness
  example (L2): fixed benchmark, AI-chosen method.

**Two of the three RSI blockers are already closed by design, not by
luck:**

- *Safe inheritance*: `commit_accepted_hypothesis` only lands a commit
  when `entry.eval_improvements` shows a real per-benchmark gain over the
  live baseline — never on a self-reported claim. This is the paper's own
  prescribed fix, already shipped before this note was written.
- *Reliable verification*: REQ-AUTO-021's `_energy_verification_patch`
  discards any hypothesis's self-reported `final_energy` and recomputes
  the true value from its claimed `final_state` through a real potential
  function the hypothesis never touches. This is exactly "frozen
  evaluator + independent ground truth" from the paper's own fix list.

*Autonomy attribution* is the honest gap: the benchmark choice, the
evaluator, and the retry/fallback policy are all outside the AI's reach.
By the paper's own standard, a higher accept rate here would not by
itself be evidence of a better search mechanism — it would just mean the
fixed benchmark got easier to solve, which is exactly what happened next.

## Part 3 — HCI explains the eight-fires pattern directly

Every real production fire after the first landed `accepted > 0` but
`committed = 0`, regardless of which generator produced the hypotheses:

| Fire | started (UTC) | accepted/rejected | fable-only? | committed |
|---|---|---|---|---|
| 1 | 2026-09-13 08:20 | 3/0 | yes (codex failed all 5) | 5 (first-ever) |
| 2 | 2026-09-13 08:49 | 5/0 | yes | 0 |
| 3 | 2026-09-13 16:58 | 4/1 | yes | 0 |
| 4 | 2026-09-13 17:32 | 2/3 | yes | 0 |
| 5 (manual) | 2026-09-13 23:42 | 4/1 | no (codex succeeded) | 0 |
| 6 | 2026-09-14 01:41 | 2/3 | yes | 0 |
| 7 | 2026-09-14 02:21 | 3/2 | yes | 0 |
| 8 | 2026-09-14 02:51 | 2/3 | yes | 0 |

Fire 1 drove both benchmarks from their seed baselines
(`double_well` 0.05, `rosenbrock` 0.5) to machine-precision zero in five
iterations, entirely via the Fable fallback (codex failed every iteration
that round too). Every fire since has been proposing genuinely good, real
optimizers (fire 6 combined Adam with a Levenberg-damped Newton polish,
fire 7 combined Adam with Polak-Ribière conjugate gradient, fire 8 used
L-BFGS-B with an explicit `1e-14` early-exit target) against a baseline
that is already at the global minimum, independent of which generator
proposed them. There is no headroom left to close. In the paper's terms:
`double_well`/`rosenbrock` are at HCI ≈ 100 for this method class, exactly
like their own "advanced mathematics: 86.4" — a benchmark this close to
saturated stops discriminating between strategies, so continued rounds
mostly measure "does the fixed evaluator have a hole in it" rather than
"is the AI a better optimizer than last time."

**This is not a bug in what shipped this session** (REQ-AUTO-019 through
REQ-AUTO-024 all did what they were built to do — see
`ops/known-issues.md` 2026-09-12/13 entries). It is the expected, now
externally-corroborated shape of an L2 loop whose fitness target has run
its course.

## Part 4 — what this suggests, not yet acted on

The plan that shipped REQ-AUTO-019 named the next fitness target as
explicitly out of scope: "A verifier-ensemble-AUROC fitness target — no
reusable scoring harness exists yet (confirmed by research); would be a
real, separate build." This note's HCI framing is an independent argument
for the same conclusion: a genuinely headroom-having target (verifier
AUROC, or an ARC-generalization score per the ARC-AGI-3 Generalization
Floor in CLAUDE.md) is what would move this loop past a saturated toy
problem, not more iterations on DoubleWell/Rosenbrock. Reaching L3
(the AI choosing *what to practice*, not just *how*) would additionally
require the loop to pick its own next benchmark from a menu — not
attempted, not scoped here.

Nothing in this note changes the current implementation. It is a record
of the mapping and the corroboration, for whoever next revisits the
fitness-target question.

## Cross-references

- `openspec/capabilities/autoresearch/spec.md` REQ-AUTO-019 through
  REQ-AUTO-024
- `ops/known-issues.md` 2026-09-12/13 entries (the always-run fix, the
  Fable fallback, the retry-on-empty fix, the diagnostic fix, the
  attribution fix)
- `scripts/autoresearch_conductor_round.py`,
  `python/carnot/autoresearch/orchestrator.py`
- CLAUDE.md "ARC-AGI-3 Generalization-Testing Floor" (a candidate
  headroom-having target for a future fitness target #2)
