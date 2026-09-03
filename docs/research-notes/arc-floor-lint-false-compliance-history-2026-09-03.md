# ARC generalization-floor lint: false-compliance history (measured 2026-09-03)

Context: commit 08d63f6d6b fixed `scripts/arc_levelup_guarantee_lint.py` --
its ARC-scope gate was a bare substring test (`"arc" in prompt`), and
"research" contains "arc". Almost every task prompt names research-roadmap /
research_conductor / research-program, so every task passed the scope gate.
One generic phrase ("held-out") then completed a false floor match. The check
could not report zero. Spec: REQ-ARC-6861.

## Method

Replay both predicates (old substring, fixed `_ARC_SCOPE` word-boundary) over
every version of `research-roadmap.yaml` in git history since the floor check
shipped (2026-07-17). 105 distinct blobs parsed into 99 distinct milestones
(2026.07.511 through 2026.09.609). Final blob per milestone used.
`research-complete.yaml` does not retain task prompts, so git history of the
roadmap file is the reconstruction; roadmap versions never committed would be
missed (none are known).

## Result

| Measure | Count |
|---|---|
| Distinct milestones since 2026-07-17 | 99 |
| Reported floor-compliant by the OLD predicate (old >= 1) | 59 |
| FALSE-COMPLIANT: old >= 1 but fixed predicate finds 0 floor tasks | 21 |
| Of those, zero ARC-scoped tasks AT ALL (fully ARC-free roadmap) | 8 |
| Fixed predicate finds MORE than old (under-match check) | 0 |

The 21 false-compliant milestones: 2026.07.515, .519, 2026.08.532, .540,
.544, .550, .552, .554, .562, .563, .566, .583, .584, .588, .589, .593,
2026.09.601, .604, .606, .607, .609.

The 8 with no ARC-scoped task at all: 2026.08.554, .563, .583, .584, .593,
2026.09.604, .606, .607. (2026.09.609's single ARC-scoped task is its
capstone saying "verify that no v609 task made an ARC solve ... claim" -- a
negative mention, so 609 is also ARC-work-free in substance.)

The remaining 13 false-compliants DID carry ARC-scoped tasks, but none with a
generalization signal -- the old predicate still reported floor coverage that
did not exist.

## Reading

The soft floor warning has been unable to fire truthfully for roughly 1 in 5
of the milestones it approved since it shipped. The last 4 milestones before
the fix (.604, .606, .607, .609) were all false-compliant, ~2 months before
the November submission deadline. The check stays WARN-only per the CLAUDE.md
"ARC-AGI-3 Generalization-Testing Floor" rule; whether the streak of
ARC-free milestones warrants promoting it to a hard gate is an operator
question, not decided here.

Reproduce: the replay script is inline in the session record; it walks
`git log --since=2026-07-17 -- research-roadmap.yaml`, dedupes blobs, and
applies both predicates from `scripts/arc_levelup_guarantee_lint.py`.
