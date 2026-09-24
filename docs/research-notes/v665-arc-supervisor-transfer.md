# V665 ARC supervisor transfer

Date: 2026-09-24  
Requirement: REQ-ARC-7625

## Question

Can accumulated trajectory-supervisor receipts support a conservative
cross-game arm-selection recommendation without new gameplay?

This task uses activity 4 of the 2026-08-22 ARC generalization amendment. It
reads existing outcome ledgers. It does not run a collector, load an LLM,
inspect hidden game source, run offline BFS, or build a per-game adapter.

## Historical boundary

Exp7611 remains a null. It found zero naturally selected matched keys. Its
verdict remains
`complete_null_matched_prefix_fixture_ready_empirical_benefit_not_established`.

Exp7612 remains blocked. Its historical protocol equality check recorded
equal-looking expected and observed values but `passed=false`. A current
read-only equality check cannot explain that historical failure. It does not
authorize an unchanged collector rerun. Its verdict remains
`complete_blocked_exp7611_protocol`.

Exp7625 changes the question. It reduces existing supervisor outcomes. It does
not repeat the Exp7611 or Exp7612 measurement.

## Evidence rule

The reducer prefers the six-game adapter-withheld V663 and V664 receipts. Each
raw receipt must match a hash in its producer artifact. Logical episodes are
hashed without seed labels and copy metadata. Byte-equivalent policy behavior
therefore counts once.

The existing supervisor classifier separates applied and shadow receipts.
Shadow `would_have_redirects` are proposals with natural control outcomes.
They are not applied redirects. They cannot enter actual fired or helped
counts. Applied redirects must retain `resolved_by_levelup`,
`actions_to_levelup`, and episode-end censoring.

## Decision rule

An authenticated ledger with zero actual firings is ready but has nothing to
refine. Its terminal result is
`complete_null_no_firings_nothing_to_refine`. This null satisfies the reserved
ARC slot under the amendment.

Arm statistics require at least 20 uncensored actual firings across at least
three games. Eligible arms receive per-game and leave-one-game-out summaries.
A never-helped arm also receives a one-sided binomial upper bound. These are
observational associations. They are not causal treatment effects.

No recommendation changes live defaults. A new-arm specification is permitted
only when an applied receipt shows that every enabled arm was used and
stagnation continued. The specification must remain game-blind and human
curated.

## Current outcome

The preferred accumulated panel contains shadow receipts, not applied
redirects. The reducer preserves their proposed and `would_have` counts while
reporting zero applied and zero actual firings. The selection recommendation is
`no_change`. No solve is claimed. The inherited attempt provenance is
`live_agent_self_discovery`.
