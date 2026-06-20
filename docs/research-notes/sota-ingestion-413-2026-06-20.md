# SOTA ingestion 2026-06-20: .413 precision and underdetermination map for .414

Reliable channel only: `scripts/sweep_clusters.py`, `scripts/sweep_semscholar.py`,
arXiv abs-page HTTP-200 checks, and low-concurrency WebSearch/WebFetch of the
top abstracts. `.venv/bin/python scripts/sweep_clusters.py --help` succeeded
and arXiv was reachable. `scripts/sweep_clusters.py` emitted focused verifier
and world-model cluster URLs. `scripts/sweep_semscholar.py` ran five focused
queries; Semantic Scholar returned six unique arXiv IDs and HTTP 429 on two
queries, so no S2-only non-arXiv source was promoted. `/deep-research` was not
invoked. No leaderboard submission was made. No live solve or training run was
launched.

## .413 outcome conditioning

Exp 4467 banked dc22 through counterexample-guided config-rule grounding,
Exp 4468 moved sc25 L2-L5 from provisional to reproduced, Exp 4469 banked a
generic sc25 cast-grid phase-FSM L1 operator, and Exp 4470 banked sb26. Exp
4474 kept the GAP-4 execution-verifier regression guard green. The remaining
frontier is not "try CEGIS"; it is program-induction precision, agreement
acceptance, and GAP-5 demo-underdetermination.

## Focused sweep result

- Counterexample Guided Learning in the Large using Reasoning Agents,
  arXiv:2606.11521, remains the clean feedback-loop template for rejected
  executable rules.
- ConVer contract and loop-invariant CEGAR-CEGIS verification, arXiv:2605.27051,
  supplies a scalable generate-check-refine contract pattern.
- Choose, Don't Label, arXiv:2604.08792, is the strongest `.414` method: turn
  ambiguous programs into multiple-choice discriminating behaviors instead of
  trusting a demo-perfect but underdetermined rule.
- Multi-Intent Detection in PBE, arXiv:2307.03966, gives a direct ambiguity
  detector precedent for examples that admit multiple intents.
- Compositional Neuro-Symbolic Reasoning, arXiv:2604.02434, maps to
  cross-example consistency filtering before a candidate reaches the gate.
- Executable World Models for ARC-AGI-3, arXiv:2605.05138, remains the
  replayable phase-FSM world-model substrate.
- Loop-OWM, arXiv:2606.12316, supplies object-centric slot/transition structure.
- Graph-Based Exploration for ARC-AGI-3, arXiv:2512.24156, is the explicit
  state-action graph baseline and source of untested transition queries.

## SOTA->experiment mapping

The `.414` planner should build a GAP-5-aware tiered acceptance harness:
induce multiple candidate programs, execute them on all demos and sibling
inputs, synthesize a Socrates-style discriminating behavior when programs agree
on the target but diverge elsewhere, and abstain when the executable evidence
cannot resolve the ambiguity. Then apply cross-example consistency and exact
replay before promoting the candidate. This feeds the open re86
pattern-match/sprite-resize gap and future manufactured variants without
claiming a live solve.

flagged_for_v414: Socrates-style multiple-choice query synthesis for GAP-5 demo-underdetermination (arXiv:2604.08792)
