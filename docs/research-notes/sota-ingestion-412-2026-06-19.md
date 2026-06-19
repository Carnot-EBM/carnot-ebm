# SOTA ingestion 2026-06-19: .412 counterexample-guided map for .413

Reliable channel only: `scripts/sweep_clusters.py`, `scripts/sweep_semscholar.py`,
arXiv abs-page HTTP-200 checks, and low-concurrency WebSearch/WebFetch of the
top abstracts. `scripts/sweep_clusters.py --help` succeeded and arXiv was
reachable. `scripts/sweep_clusters.py` emitted focused verifier and world-model
cluster URLs. `scripts/sweep_semscholar.py` ran five focused queries; Semantic
Scholar returned HTTP 429 on all five, so no S2-only source was promoted.
`/deep-research` was not invoked. No leaderboard submission was made. No live
solve or training run was launched.

## Focused sweep result

- Counterexample-Guided Learning in the Large, arXiv:2606.11521, is the
  freshest and strongest fit for `.413`: use verifier feedback as a structured
  counterexample, then re-induce from the rejecting execution state.
- Neuro-Symbolic Reasoning for Planning, arXiv:2309.16436, supplies the formal
  CEGIS loop: LLM learner, exact verifier, counterexample, revised candidate.
- SOAR self-improving evolutionary synthesis, arXiv:2507.14172, maps to
  evolutionary repair and hindsight banking after a generic operator fails.
- Towards Efficient Neurally-Guided Program Induction for ARC-AGI,
  arXiv:2411.17708, maps to ordered search over a compact glyph/grid DSL.
- Combining Induction and Transduction for Abstract Reasoning,
  arXiv:2411.02272, maps to routing between exact induced programs and direct
  state predictions.
- Executable World Models for ARC-AGI-3, arXiv:2605.05138, remains the
  interactive phase-FSM verification harness.
- Loop-OWM, arXiv:2606.12316, supplies object-centric slot and transition-loop
  structure for cast-grid and toggle mechanics.
- The ARC of Progress living survey, arXiv:2603.13372, anchors the meta-solver
  context and warns that refinement loops remain load-bearing across ARC
  versions.

## SOTA->experiment mapping

The `.413` planner should implement counterexample-guided re-induction as the
front door for the remaining generic-solver failures. When dc22, tr87, or sc25
rejects a proposed generic rule, record the exact execution state and failed
predicate, cluster related failures, re-prompt the inducer with that
counterexample, and accept only reproduction-gated fixes. SOAR and
neurally-guided induction provide search pressure; induction+transduction
provides routing; Executable World Models and Loop-OWM provide phase-FSM and
object-state verification targets.

flagged_for_v413: Counterexample-guided re-induction from rejecting execution states (arXiv:2606.11521; SMT-checked CEGIS predecessor arXiv:2309.16436)
