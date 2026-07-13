# ARC-AGI-3 Perception-Grounding Audit (2026-07-13)

**Status: diagnostic only, no code changes.** Per `ops/known-issues.md`'s 2026-07-06
ARC standing-floor task queue, item 1 ("perception-grounding audit, cheap, run
first"): audit whatever frame-diff / state-change read the live agent's
perception layer currently relies on for the failure mode described in
arXiv:2603.17683 ("Sensi") — an LLM-as-judge validating its own hallucinated
frame-diff reads as internally self-consistent rather than against ground
truth, producing a self-reinforcing error cascade. This is a report of
findings, not a fix.

## Scope

Audited the two modules named in the task: `arc_frame_change_predictor.py`
and `arc_online_action_effect_scorer.py`, plus their wiring into the live
`E3AgentPolicy` default configuration in `arc_competition_agent.py`.

## Finding 1 (positive): the primary frame-diff scorer is NOT vulnerable to
the literal Sensi failure mode, and the defense is deliberate

`GroundTruthValidatedFrameChangeScorer` (`arc_frame_change_predictor.py`)
wraps the live `LiveActionEffectScorer` and gates its influence on ranking
behind **agreement with real observed pixel deltas**, not self-consistency:

- `observe_transition(before, action_id, data, after)` computes the
  scorer's *predicted* change verdict from `before`, and the *actual*
  observed change from a literal pixel comparison of `before` vs `after`
  (`_transition_frame_delta`: `np.count_nonzero(lhs != rhs) / total`).
- The scorer only contributes to candidate ranking (`candidate_score`
  returns nonzero) once `validated` is `True` — which requires at least
  one predicted/observed agreement **and zero contradictions ever**. A
  single wrong prediction permanently disables that scorer instance for
  the rest of the episode (no partial credit, no majority vote).
- This channel is not an LLM and does not do free-text reasoning at all —
  it is a small supervised CNN (`SmallFrameChangeCNN`) plus a count-based
  memory (`PersistentAEM`), both scored numerically and checked against
  real pixels. It structurally cannot exhibit "the judge validates its own
  hallucinated read as self-consistent" because there is no judge doing
  free-text self-assessment in this channel.
- This wrapper is wired into the SHIPPED default:
  `_load_submitted_frame_change_scorer()` (called whenever `E3AgentPolicy`
  is constructed with its default `frame_change_scorer` sentinel) always
  returns `GroundTruthValidatedFrameChangeScorer(...)`, and
  `SUBMITTED_FRAME_CHANGE_PREDICTOR_ENABLED = True`. `observe_transition`
  is genuinely called on the live path with real `(previous_frame, action,
  latest)` triples (`arc_competition_agent.py` ~line 1457-1475), guarded
  only by `hasattr(fcs, "observe_transition")` and a non-None
  `previous_frame` — confirmed this guard passes for the live default
  wrapper (it has the method), so validation feedback genuinely reaches it
  during play, not just at construction time.

Similarly, tier-3 world-model induction (the LLM-authored `engine()` /
`is_level_complete()` code) is checked by literally running the induced
code against **held-out real transitions** and comparing predicted vs
actual pixels (`WorldModelVerifier`, `heldout_accuracy`,
`binary_gate_pass`) — not by asking the proposer "are you sure" or by
checking the LLM's self-consistency across repeated calls. This session's
own `exp5588`/`exp5589` runs are a live demonstration of this gate working
correctly: an induced engine with `heldout_accuracy=0.0` (7 transitions)
and `0.125` (25 transitions) was correctly rejected both times rather than
trusted on the strength of the LLM's own confidence.

**Conclusion on Finding 1:** the two most LLM-adjacent perception channels
in the live agent (frame-diff scoring, tier-3 induced dynamics) are both
architecturally grounded against real pixel/transition ground truth, not
self-consistency. The Sensi paper's specific failure mode does not appear
to apply to either channel as currently built.

## Finding 2 (real, live-path-active): the CNN term of the frame-change
scorer is silently zero on the default frontier-priority computation

This is unrelated to the Sensi self-consistency pattern, but is exactly
the kind of "perception silently not doing what it appears to do" issue
this audit exists to surface, and it was previously partially discovered
but never fixed on the shipped path.

- `arc_online_action_effect_scorer.py` (a research-only scorer variant,
  **not** wired into the shipped default) documents, as an already-measured
  finding from prior project work: `FrameChangeScorer.candidate_score`
  reads `getattr(candidate, "action_id")`, which raises `AttributeError`
  on a plain dict candidate (`{"action": ..., "data": ...}` — one of the
  two candidate shapes the explorer's frontier uses). `LiveActionEffectScorer
  .candidate_score` wraps the CNN call in a bare `try/except Exception:
  pass`, so this AttributeError is silently swallowed and the CNN term
  contributes exactly zero. The module's own docstring states this was
  measured to occur on "~20/25 games" and explained a genuine false
  negative in a prior distinct experiment ("the online-vs-frozen first-win
  null").
- The fix for this (`_as_action_like`, a dict-to-namespace normalizer) was
  built **only** inside `OnlineActionEffectScorer`
  (`arc_online_action_effect_scorer.py`), which is a research-arm factory
  (`build_online_scorer(arm, root)`) for comparing "frozen" vs
  "online-scratch"/"online-warm" experiment arms. It was never backported
  to `FrameChangeScorer.candidate_score` or `LiveActionEffectScorer
  .candidate_score` in `arc_frame_change_predictor.py` — confirmed via
  `grep`: `_as_action_like` appears nowhere outside
  `arc_online_action_effect_scorer.py`.
- Traced every live call site that reaches `frame_change_scorer
  .candidate_score` (or a wrapper around it) to determine whether
  dict-shaped candidates are actually passed to it on the DEFAULT
  configuration (not just in principle):
  - `_candidates()`'s main path (`rich_action_candidates(...,
    frame_change_scorer=...)`) scores proper `ArcAction`-shaped
    candidates before they are ever converted to dict rows — CNN term
    contributes correctly here.
  - `qd_generator` / `controllable_novelty_policy` consumers (lines ~700,
    705, 1240, 2008) DO pass dict-shaped rows into the scorer, but both
    features are **off by default**
    (`SUBMITTED_CONTROLLABLE_NOVELTY_PROPOSAL_ENABLED = False`; no
    `qd_generator` default), so these specific call sites are inert on
    the shipped agent today.
  - **`ActionEffectExpansionPrior.frontier_priority(frame, node.get
    ("untested") or [])`, called from `_action_effect_frontier_key()`
    (`arc_competition_agent.py:1102-1116`), IS part of the default,
    always-on path** — gated only by
    `SUBMITTED_ACTION_EFFECT_EXPANSION_PRIOR_ENABLED = True`, with no
    other flag. `node["untested"]` is the frontier's dict-row
    representation (`{"action": ..., "data": ...}`), confirmed by the
    dict-construction pattern used to build frontier rows elsewhere in
    the same file (e.g. `rows.append({"action": 6, "data": {...}})`).
    This means: **every single frontier-expansion-priority computation on
    the live default agent silently drops the CNN term of the frame-change
    scorer**, for every game, every turn. The count-based `PersistentAEM`
    memory term (`_action_id_from`, a shape-tolerant helper) is unaffected
    and continues to contribute correctly — this is not a total scorer
    outage, just a silent, always-on loss of the CNN's contribution
    specifically to frontier prioritization.

**Practical magnitude, honestly bounded:** the CNN term already carries a
small blend weight (`cnn_weight = 0.05` vs `memory_weight = 1.0` in
`LiveActionEffectScorer`), so even a fully-functioning CNN term would be a
minor nudge relative to the memory term, not a dominant signal. Combined
with this project's own prior finding (`project_arc_live_agent_learning_gaps`
memory: frame-only order-1 features sit at LOO=chance on held-out games),
it is plausible the CNN term was not carrying much marginal signal on novel
games anyway — this audit does not establish how much capability, if any,
this specific bug costs. What it does establish is that the bug is real,
confirmed on the live-default frontier-priority path (not just a
theoretical or research-arm-only concern), and has a ready-made fix
sitting unused one file away.

## Cross-reference to the project's own prior perception finding

This audit's Finding 1 (the scorer channels ARE grounded, not
self-consistent) and Finding 2 (a real but likely-small silent
degradation) are both consistent with, and do not contradict,
`project_arc_live_agent_learning_gaps`'s standing conclusion that the
binding constraint on the live agent is PERCEPTION — specifically that
frame-only order-1 features sit at LOO=chance on held-out games. This
audit did not find evidence that the perception layer is confidently
*wrong* (hallucinating false signal, the Sensi pattern); it found that one
already-small perception signal is silently absent on the default
frontier-priority path, and that the overall channel's information content
was already known to be weak. Both point toward representation quality
(what the perception layer can encode) rather than validation discipline
(whether its claims are checked) as the dominant open gap — the validation
machinery here is unusually careful for a project of this scope.

## Recommendations (not executed — audit only)

1. **Cheap, mechanical**: backport `_as_action_like`'s dict-normalization
   from `arc_online_action_effect_scorer.py` into
   `FrameChangeScorer.candidate_score` (or wrap call sites), so the CNN
   term contributes on `ActionEffectExpansionPrior.frontier_priority` too.
   Low risk (the existing behavior is "silently contributes zero", so a
   fix can only add signal, not regress the memory term), but should be
   measured (matched-budget A/B) before being trusted as a capability win,
   consistent with CLAUDE.md's Phase Prototype + Empirical Validation
   discipline — a silently-dead 0.05-weight term turning on is not
   guaranteed to help given the LOO=chance finding above.
2. **Documentation**: the comment at `arc_competition_agent.py:1457-1460`
   ("The LiveActionEffectScorer (the frozen shipped scorer) does NOT have
   this method, so the check `hasattr(fcs, ...)` makes this a guaranteed
   no-op for the frozen path") is stale/misleading for the current
   default: `self.frame_change_scorer` on the default path is
   `GroundTruthValidatedFrameChangeScorer`, which DOES have
   `observe_transition`, and it IS called. The comment appears to describe
   the *inner* `LiveActionEffectScorer` rather than the outer wrapper
   actually assigned to `self.frame_change_scorer`. Worth a one-line
   correction so a future reader does not conclude ground-truth validation
   feedback never reaches the live default (it does).
3. Per task 2 in the same known-issues queue entry, the classical
   connected-component/color-blob segmentation lever (arXiv:2512.24156)
   remains the higher-leverage next step for representation quality — this
   audit does not change that priority ordering.

## Sources consulted

- `python/carnot/agentic/arc_frame_change_predictor.py`
- `python/carnot/agentic/arc_online_action_effect_scorer.py`
- `python/carnot/agentic/arc_competition_agent.py` (E3AgentPolicy
  construction, `_load_submitted_frame_change_scorer`,
  `ActionEffectExpansionPrior` wiring, `observe_transition` call sites)
- `results/experiment_5588_tier3_induction_live_path_sanity_check.json`,
  `results/experiment_5589_tier3_induction_normal_budget_capability_check.json`
  (this session's own live evidence that the induction trust gate rejects
  low-confidence induced engines rather than trusting them)
- `ops/known-issues.md` 2026-07-06 entry (task queue, arXiv:2603.17683
  citation)
- `project_arc_live_agent_learning_gaps` memory (prior LOO=chance finding)
