# Agentic Verification Efficiency Positioning - Exp 3932

Method: local document synthesis only. No new inference was run; this note reads
`research-references.md`, `research-studying.md`, `ops/north-star.md`, and the
local Exp 3926/3928/3929 result artifacts.

## Position

Carnot's current thesis matches the 2026 verification-efficiency literature:
use a cheap discriminative verifier first, reserve competent generative judges
for hard or close cases, and measure value as accuracy per unit cost. ProcessBench
is the right held-out step-verification venue because it is explicitly labeled
at the process-step level. ThinkPRM and GenRM define the competent-judge recipe:
verification as structured generation plus a parsed verdict, not a raw yes/no
judge prompt. Budget-aware Discriminative Verification supplies the cost model:
a forward-pass verifier can beat generative verification under a fixed compute
budget even when the generative judge is strong.

The local evidence is not yet a clean efficiency win. Exp 3926 is blocked/flagged,
so the competent-judge parity/Pareto claim remains unlanded on disk. Exp 3928 is
also blocked/flagged, so the independent-corpus moat replication still needs a
clean run. Exp 3929 does land the first ARC-AGI-3 agentic step: a verifier-pruned
synthetic grid agent solved at 1.959x action efficiency, CI95
[1.742, 2.194], while explicitly making no official ARC-AGI-3 score
claim.

## Interpretation

Carnot is best positioned as a classifier-first verification layer rather than
as another generative PRM. The differentiator is not that energy reasoning is
more expressive than ThinkPRM or GenRM; it is that a cheap external verifier can
screen every step and every candidate action, then escalate only the uncertain
cases. That is exactly the north-star win condition: equally effective as the LM
at lower cost, with ARC-AGI-3 as the agentic proof venue after the offline proof.

## Next Experiments

1. ProcessBench full-benchmark head-to-head. Rationale: it converts the blocked
   Exp 3926 efficiency thesis into the decisive comparison against a competent
   GenRM/ThinkPRM-style judge on a standard held-out process-verification corpus.
2. ARC-AGI-3 real-benchmark agentic run. Rationale: it converts the Exp 3929
   synthetic action-pruning result into the intended interactive venue while
   preserving the discipline that no official score is claimed unless the
   benchmark protocol actually runs.
