#!/usr/bin/env python3
"""Exp 5904 -- STAGE 1 (OFFLINE) diagnostic: can a coordinate-aware click feature beat the
coordinate-blind incumbent at predicting the outcome of a click?

Spec refs: REQ-ARC-FCP-5904, SCENARIO-ARC-FCP-5904-COORDINATE-BLINDNESS-IS-REPAIRED,
SCENARIO-ARC-FCP-5904-BLIND-ARM-IS-A-LABEL-VALIDITY-CHECK.

WHAT THIS IS, AND WHAT IT IS EXPLICITLY NOT
-------------------------------------------
This is a FEASIBILITY MEASUREMENT of a signal, not a capability claim. It is STAGE 1 only.
The terminal gate for any router change is a LIVE A/B on banked levels, which is a SEPARATE,
LATER experiment. Offline AUROC licenses NOTHING here, and that is measured history, not
caution: exp4545's 0.725-AUROC discriminator REGRESSED live search, which is exactly why the
live agent's ``SUBMITTED_VALUE_WEIGHT`` is pinned at 1e-12. Accordingly the router change
this experiment measures ships DEFAULT OFF
(``arc_discriminative_router.SUBMITTED_ONLINE_CLICK_TARGET_ROUTER_ENABLED``).

THE DEFECT UNDER TEST
---------------------
The live candidate router scores a click via ``_action_id(action)``, which returns the action
TYPE integer -- 6 for every click. ``cross_game_features_v3`` then consumes it through a
7-dimensional one-hot in which coordinates are structurally unrepresentable. Measured: 37
distinct click targets on an lp85 reset frame collapse to ONE router score, and ``rank()``
preserves the input order exactly.

ONLINE / WITHIN-GAME ONLY
-------------------------
The label below is measured by a LABEL-TIME search over the offline env (fork the env, step
the candidate click, read the outcome). That is a development-proxy ORACLE used for corpus
construction, exactly like any training label -- the live agent NEVER calls it at inference.
The discriminator is fitted ONLINE, WITHIN A SINGLE GAME, on a TEMPORAL split (fit on the
earlier states, score the later ones), and is discarded per game. Nothing is trained,
carried, or pooled across games.

That exclusion is deliberate on two grounds: ``ops/exclusion_manifest.yaml`` id
``cross_game_value_transfer_retired_exp4342_v401`` retires cross-game learned value transfer
after three nulls (``operator_reopen_required: true``); and a hidden-game agent has no prior
exposure to the hidden game to transfer from in the first place (CLAUDE.md "ARC-AGI-3 IS a
Live Hidden-Game Discovery Agent").

THE LABEL, AND WHY IT IS THIS ONE
---------------------------------
``label = 1`` iff ACTUALLY STEPPING this click against the offline env changed the settled
grid or advanced ``levels_completed``. It is causally downstream of the click by
construction, because it IS the observed post-click frame.

MEASURED CAVEAT ON THAT DISJUNCTION: on the corpus harvested so far, every level-up ALSO
changed the frame, so the label is IDENTICALLY "the frame changed" and the ``OR levels_up``
disjunct supplies no positives of its own. The artifact reports this as a COUNT
(``label_composition.n_levels_up_without_change`` /
``label_is_measured_identical_to_frame_change``) rather than assuming it either way, because
``n_levels_up`` sitting next to ``n_positive`` otherwise reads as if level-ups contributed
distinct positives. Two consequences: (a) the headline AUROC is a frame-change result, and
frame-change is the EASY component; (b) this repo ALREADY has modules aimed at exactly that
target (``arc_frame_change_predictor``'s ``FrameChangeScorer`` /
``GroundTruthValidatedFrameChangeScorer`` / ``BehaviorActionPrior`` / ...). They are NOT run
as arms here -- ``FrameChangeScorer`` needs a trained ``SmallFrameChangeCNN`` checkpoint and
none exists in ``models/`` (checked at runtime), and the ground-truth-validated scorer is
already DOWNSTREAM of this router rather than a comparable offline arm. The artifact says so
in ``baselines_not_run_and_why``. So "the incumbent carries ZERO signal" is a claim about the
V3 ROUTER specifically, never about this project's frame-change capability in general.

The level-up question is answered separately, by the HARD SLICE: among the clicks that DID
change something, which ones level up (``levels_up`` conditioned on ``changed``)? That arm
previously could never produce a number -- it filtered on ``changed`` while reusing the
pooled label, so every row in it was a positive and AUROC was undefined by construction. Its
label is now ``levels_up``, which is what makes it run.

Rejected alternatives, each for a measured reason:

* ``arc_human_replay_corpus.level_progress(row, step_index)`` -- a pure function of the step
  index (verified at that module's lines 134-145). This is the exp5835 defect: a
  zero-perception step-index predictor scored 0.9234 there while the "perception" arms scored
  0.66-0.69. The human-replay corpus is also structurally unusable for click discrimination:
  it holds exactly ONE action per frame, so there are no within-state sibling negatives.
* Matching a candidate against a banked winning trajectory by exact ``(x, y)`` -- measured
  0/6 matches on lp85 and 0/6 on tn36 (banked routes click a button's true pixel centre;
  the generator proposes component centroids, off by L1 1-4). That silently yields a
  zero-positive corpus, i.e. a false null.

THE FOUR ARMS (a null without a positive control is not a finding)
-----------------------------------------------------------------
* ``blind``  -- the incumbent coordinate-blind featurization. Expected AUROC ~0.5 BY
  CONSTRUCTION (one distinct score). It is the defect demonstration AND a NEGATIVE control:
  a coordinate-blind scorer CANNOT beat chance on a click-dependent label, so a ``blind``
  AUROC outside [0.45, 0.55] means the LABEL LEAKS and no arm is interpretable.
* ``coord``  -- the 21 coordinate-aware features, fit online within-game, scored
  prospectively on held-out later states.
* ``random`` -- ``RandomCandidateRouter``, the already-shipped coordinate-aware-but-
  uninformative control. It isolates "coordinate-aware" from "informative".
* ``step_index`` -- a ZERO-PERCEPTION control (the state index only). ``coord`` must beat it;
  if it does not, the harness or the label is broken. This is the exp5835 lesson as a gate.

THE METRIC OF RECORD IS WITHIN-STATE AUROC
-----------------------------------------
The live router only ever ranks candidates WITHIN one frame's candidate list; it never
compares a click on frame A against a click on frame B. So the metric of record is the
WITHIN-STATE AUROC (per-state, pooled by discriminating-pair count), and the across-state
figure is reported as a labelled reference only.

That is not a stylistic preference -- it was measured in this experiment's own first run. The
incumbent router emits exactly ONE distinct score per state (verified: 38-48 distinct click
targets, 1 distinct score, at every harvested state) but a DIFFERENT constant per state,
because v3's features are frame-level. Pooled across states it therefore scored 0.3105, an
apparent "signal" produced entirely by cross-state base-rate variation that cannot influence
a single live ranking decision. Within-state it is exactly 0.5, as it must be.

TWO CHECKS, AND ONLY THE SECOND IS THE GATE OF RECORD
-----------------------------------------------------
1. ``coordinate_blindness_repair_check`` (originally emitted as ``pre_registered_gate``)::

       within_state_coord_auroc >= 0.60  AND
       (within_state_coord_auroc - within_state_blind_auroc) >= 0.10

   It is honest about the defect and USELESS as evidence of value, so it is no longer called a
   gate and no longer names the verdict. Two measured reasons. (a) ``blind`` is pinned at
   exactly 0.5 within-state by construction, so the second conjunct collapses into the first --
   the "conjunction" is a single threshold. (b) Evaluated against this run's own arms, the
   uninformative controls clear it too: the shipped ``RandomCandidateRouter`` at 0.6308 passes,
   and so does ``static_salience`` at 0.9360 -- i.e. the bar is cleared both by an arm that
   knows nothing and by the ordering that is ALREADY LIVE. The artifact now emits
   ``also_passed_by_arms`` so that fact travels with the number.
   (The band check on ``blind`` remains reported SEPARATELY as ``label_validity_check`` and is
   never folded into any gate -- exp5835 was voided partly for a conjunct that asserted
   something about the baseline arm, making it unpassable for ANY treatment value.)

2. ``informativeness_gate`` -- ADDED POST-HOC AFTER REVIEW, and labelled as such in the
   artifact rather than backdated::

       within_state_coord_auroc >= 0.60  AND
       lower_bound(bootstrap_ci95(coord - static_salience)) > 0   for every contributing game

   This is the gate of record, because check (1)'s pass region CONTAINS REGRESSIONS:
   ``static_salience`` lies inside it, so a coord of 0.70 would have reported "passed" while
   ordering clicks WORSE than the incumbent sort it would replace. Both conjuncts here are
   still properties the treatment can move on its own, so the exp5835 defect is not
   reintroduced.

THE HONEST COMPARATOR IS THE STATIC SALIENCE SORT, AND IT NEEDS AN INTERVAL
--------------------------------------------------------------------------
Beating ``blind`` is trivial -- it is a constant. The number that says what this signal is
actually WORTH is ``coord_minus_static_within_state``: how much the learned head adds over the
STATIC area x colour-rarity salience sort, which is what really orders clicks live once the
router ties. It is the HEADLINE of this artifact.

It ships with a PAIRED WITHIN-STATE BOOTSTRAP CI and ``fraction_replicates_le_zero``, plus a
measured noise floor (the spread of the uninformative random arm's own within-state AUROC over
``RANDOM_NOISE_SEEDS`` seeds on the same corpus). Reporting it as a bare point estimate was
the shape of exp3540, whose paired p=0.135 "advantage" the retro classified as a small-sample
artifact. Measured on the smoke corpus (lp85, 2 scored states, 20 positives): delta +0.0436,
bootstrap CI95 [-0.0302, +0.1245], P(delta<=0) = 0.133, against an uninformative-arm seed-noise
sd of 0.0717 -- the delta is inside the CI's zero and roughly HALF the noise the corpus
produces for free. The smoke run therefore establishes NO gain over the incumbent ordering, and
the verdict says exactly that.

WHAT THE HARD SLICE SAYS (measured, and it is not flattering)
------------------------------------------------------------
lp85 smoke, 20 clicks that changed something, 5 of them level-ups: coord 0.5067 (chance),
static_salience 0.6600, shipped-random 0.7067. So the 0.98 headline is carried by the trivial
"does this click do anything at all" component; on the slice that asks "which effective click
makes PROGRESS" the learned head is at chance and BEHIND the incumbent static sort. With 5
positives none of these three numbers is powered -- which is the point: this is a
feasibility measurement, and what it currently measures is that the level-up question is
UNANSWERED, not answered favourably.

Run:
    .venv/bin/python python/carnot/experiment_5904_click_target_discrimination.py --smoke
    .venv/bin/python python/carnot/experiment_5904_click_target_discrimination.py  # full
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "python") not in sys.path:
    sys.path.insert(0, str(REPO / "python"))

EXPERIMENT_ID = 5904
RANDOM_SEED = 5904
ARTIFACT_PATH = REPO / "results" / "experiment_5904_click_target_discrimination.json"

# Games whose banked routes reach a level-up on a CLICK, so a click-discrimination corpus can
# exist at all. Measured over 12 games: 6 others (sc25, dc22, sk48, m0r0, cd82, sp80) level up
# on KEYBOARD actions and contribute nothing here.
CLICK_GAMES = ("lp85", "vc33", "su15", "r11l", "tn36")
SMOKE_GAMES = ("vc33", "lp85")

# States per game to harvest. Cost is dominated by the banked-prefix replay length, measured
# 4.2-70.6 ms per fully-labelled sample.
DEFAULT_MAX_STATES = 12
SMOKE_MAX_STATES = 4
DEFAULT_MAX_CLICKS = 48  # the live generator's own default cap

# Seeds used to measure the uninformative arm's own spread on this corpus -- the noise floor
# any claimed delta has to clear. 200 is enough for a stable sd at this corpus size and costs
# well under a second (the random arm is a hash, not a model).
RANDOM_NOISE_SEEDS = 200


# --------------------------------------------------------------------------- preconditions


def check_preconditions() -> list[dict[str, Any]]:
    """PRECONDITIONS, checked BEFORE any measurement.

    Per CLAUDE.md's Pre-Launch Preconditions Discipline: if a resource is missing, the honest
    move is a ``blocked_*`` verdict, never a synthesized number. Every confirmed fabrication
    in this project's history shared the root cause of an agent silently lacking a resource
    and proceeding anyway.
    """

    checks: list[dict[str, Any]] = []

    env_dir = REPO / "environment_files"
    checks.append(
        {
            "resource": "environment_files_present",
            "available": env_dir.is_dir() and any(env_dir.iterdir()),
            "detail": str(env_dir),
        }
    )

    try:
        import numpy  # noqa: F401
        from scipy import ndimage  # noqa: F401

        checks.append({"resource": "numpy_scipy_importable", "available": True, "detail": "ok"})
    except Exception as exc:  # pragma: no cover - environment guard
        checks.append(
            {"resource": "numpy_scipy_importable", "available": False, "detail": repr(exc)}
        )

    try:
        from carnot.agentic.arc_solver_kit import offline_arcade

        arcade = offline_arcade()
        checks.append(
            {
                "resource": "offline_arcade_constructible",
                "available": arcade is not None,
                "detail": "OperationMode.OFFLINE, no network",
            }
        )
    except Exception as exc:  # pragma: no cover - environment guard
        checks.append(
            {"resource": "offline_arcade_constructible", "available": False, "detail": repr(exc)}
        )

    try:
        from carnot.agentic.arc_discriminative_router import (
            SUBMITTED_ONLINE_CLICK_TARGET_ROUTER_ENABLED,
            load_cross_game_discriminative_router,
        )

        checks.append(
            {
                "resource": "incumbent_v3_router_loadable",
                "available": load_cross_game_discriminative_router() is not None,
                "detail": "models/arc_discriminative_verifier_v3.json",
            }
        )
        checks.append(
            {
                "resource": "online_click_router_default_off",
                "available": SUBMITTED_ONLINE_CLICK_TARGET_ROUTER_ENABLED is False,
                "detail": "live parity is preserved by the default-off flag",
            }
        )
    except Exception as exc:  # pragma: no cover - environment guard
        checks.append(
            {"resource": "incumbent_v3_router_loadable", "available": False, "detail": repr(exc)}
        )

    return checks


# ------------------------------------------------------------------------- banked routes


def _load_banked_route(game: str) -> list[tuple[int, dict[str, Any] | None]]:
    """The game's banked winning action sequence, normalized to (action_id, data).

    Reuses ``scripts/arc3_replay_scorecard_metaharness.py``'s tables and parsers rather than
    re-deriving them: those already encode which artifact holds each game's replayable
    trajectory and the three different on-disk formats they use.
    """

    scripts_dir = str(REPO / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from arc3_replay_scorecard_metaharness import (  # type: ignore[import-not-found]
        GAME_ARTIFACTS,
        RESOLVED_ARTIFACTS,
        load_actions,
        normalize,
    )

    source = RESOLVED_ARTIFACTS.get(game, GAME_ARTIFACTS.get(game))
    if not source:
        return []
    route: list[tuple[int, dict[str, Any] | None]] = []
    for raw in load_actions(source):
        action_id, data = normalize(raw)
        if action_id is None:
            continue
        route.append((int(action_id), data))
    return route


def _resolve_game_id(arcade: Any, game: str) -> str:
    """Map a short game key ('lp85') to the arcade's full id ('lp85-305b61c3')."""

    env_dir = REPO / "environment_files" / game
    if env_dir.is_dir():
        for child in sorted(env_dir.iterdir()):
            if child.is_dir():
                return f"{game}-{child.name}"
    return game


# ------------------------------------------------------------------------------ harvest


class _Harvester:
    """Forks the offline env to MEASURE each candidate click's outcome.

    Every fork is a fresh ``arcade.make() + reset()`` plus a replay of the banked prefix.
    Measured 4.1 ms for make+reset and 0.19 ms per step -- and crucially ~5x cheaper than
    ``env.reset()`` on an existing env (21.5 ms), which is why a fresh env is used per fork.
    """

    def __init__(self, game: str) -> None:
        from arcengine import GameAction

        from carnot.agentic.arc_solver_kit import offline_arcade

        self.game = game
        self.arcade = offline_arcade()
        self.scorecard_id = self.arcade.open_scorecard()
        self.game_id = _resolve_game_id(self.arcade, game)
        self.GameAction = GameAction
        self.route = _load_banked_route(game)
        self.n_forks = 0

    def _action_enum(self, action_id: int) -> Any:
        return getattr(self.GameAction, f"ACTION{int(action_id)}")

    def fork(self, prefix_n: int, extra_clicks: Sequence[tuple[int, int]] = ()) -> Any:
        """Replay ``prefix_n`` banked actions, then the given clicks; return the last frame."""

        env = self.arcade.make(self.game_id, scorecard_id=self.scorecard_id)
        frame = env.reset()
        self.n_forks += 1
        for action_id, data in self.route[:prefix_n]:
            frame = env.step(self._action_enum(action_id), data=data, reasoning=None)
            if frame is None:
                return None
        for x, y in extra_clicks:
            frame = env.step(self.GameAction.ACTION6, data={"x": int(x), "y": int(y)})
            if frame is None:
                return None
        return frame

    def close(self) -> None:
        try:
            self.arcade.close_scorecard(self.scorecard_id)
        except Exception:
            pass


def harvest_game(
    game: str, *, max_states: int, max_clicks: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the labelled (state, click, label) corpus for one game.

    Returns ``(rows, diagnostics)``. Each row carries the ORIGINAL salience rank (the
    candidate's index in the un-reordered generator output, which IS the incumbent static
    baseline's ordering) so the triviality baseline can be reported alongside the arms.
    """

    from carnot.agentic.arc_click_target_features import (
        click_target_features,
        click_target_frame_context,
    )
    from carnot.agentic.arc_discriminative_router import load_cross_game_discriminative_router
    from carnot.agentic.arc_graph_explore import rich_action_candidates
    from carnot.agentic.arc_solver_kit import frame_level, settled_grid

    # The REAL incumbent router, so the defect is MEASURED per state rather than asserted.
    incumbent = load_cross_game_discriminative_router()
    harvester = _Harvester(game)
    diagnostics: dict[str, Any] = {
        "game": game,
        "game_id": harvester.game_id,
        "route_length": len(harvester.route),
        "states_considered": 0,
        "states_kept": 0,
        "states_excluded_no_positive": 0,
        "states_excluded_no_negative": 0,
        "states_excluded_no_clicks": 0,
        "n_forks": 0,
    }
    rows: list[dict[str, Any]] = []
    if not harvester.route:
        diagnostics["note"] = "no banked route on disk"
        harvester.close()
        return rows, diagnostics

    try:
        # One full replay records where levels_completed increments -- the level-up
        # boundaries. Measured 92 ms for tn36's 102 actions.
        boundaries: list[int] = []
        base_frame = harvester.fork(0)
        if base_frame is None:
            diagnostics["note"] = "reset returned no frame"
            return rows, diagnostics
        env = harvester.arcade.make(harvester.game_id, scorecard_id=harvester.scorecard_id)
        frame = env.reset()
        level = frame_level(frame)
        for index, (action_id, data) in enumerate(harvester.route):
            frame = env.step(harvester._action_enum(action_id), data=data, reasoning=None)
            if frame is None:
                break
            new_level = frame_level(frame)
            if new_level > level:
                boundaries.append(index)
            level = new_level
        diagnostics["level_up_boundaries"] = boundaries
        diagnostics["banked_levels"] = level

        # Candidate states: the state immediately BEFORE each level-up whose banked action is
        # a click (true distance 1 -- the only place the cheap k=1 label has positives;
        # measured 14/14 d=1 states have >= 1 level-up click, 0/29 states at d=2-3 do), then
        # earlier click states as filler so the corpus has non-boundary negatives too.
        boundary_states = [b for b in boundaries if harvester.route[b][0] == 6]
        filler = [i for i, (aid, _d) in enumerate(harvester.route) if aid == 6]
        ordered_states: list[int] = []
        for index in boundary_states + filler:
            if index not in ordered_states:
                ordered_states.append(index)
        ordered_states = ordered_states[:max_states]

        for state_index in ordered_states:
            diagnostics["states_considered"] += 1
            frame = harvester.fork(state_index)
            if frame is None:
                continue
            before_grid = np.array(settled_grid(frame), copy=True)  # MUST copy: shared buffer
            before_level = frame_level(frame)
            candidates = [
                action
                for action in rich_action_candidates(frame, max_click=max_clicks)
                if int(getattr(action, "action_id", 0)) == 6
                and isinstance(getattr(action, "data", None), dict)
            ]
            if not candidates:
                diagnostics["states_excluded_no_clicks"] += 1
                continue

            # MEASURE the incumbent's collapse at this exact state: how many distinct scores
            # does the live router actually produce across these candidates?
            incumbent_scores: list[float] = []
            if incumbent is not None:
                try:
                    incumbent_scores = [
                        float(incumbent.score(frame, action)) for action in candidates
                    ]
                except Exception:
                    incumbent_scores = []

            context = click_target_frame_context(frame, use_cache=False)
            state_rows: list[dict[str, Any]] = []
            outcome_classes: dict[bytes, int] = {}
            for salience_rank, action in enumerate(candidates):
                x, y = int(action.data["x"]), int(action.data["y"])
                after = harvester.fork(state_index, [(x, y)])
                if after is None:
                    continue
                after_grid = np.array(settled_grid(after), copy=True)
                changed = after_grid.shape != before_grid.shape or bool(
                    np.any(after_grid != before_grid)
                )
                levels_up = int(frame_level(after)) > int(before_level)
                key = after_grid.tobytes()
                outcome_class = outcome_classes.setdefault(key, len(outcome_classes))
                state_rows.append(
                    {
                        "game": game,
                        "state_index": state_index,
                        "at_boundary": state_index in boundary_states,
                        "x": x,
                        "y": y,
                        "salience_rank": salience_rank,
                        "changed": changed,
                        "levels_up": levels_up,
                        "outcome_class": outcome_class,
                        "label": 1.0 if (changed or levels_up) else 0.0,
                        "features": click_target_features(context, x, y),
                        "incumbent_score": (
                            incumbent_scores[salience_rank]
                            if salience_rank < len(incumbent_scores)
                            else None
                        ),
                    }
                )

            positives = sum(1 for row in state_rows if row["label"] >= 0.5)
            negatives = len(state_rows) - positives
            if positives == 0 or negatives == 0:
                # MANDATORY exclusion: a state with no positive (no candidate can win) or no
                # negative (EVERY candidate wins) is a DEGENERATE test -- keeping it injects
                # pure negatives or pure positives and inflates any AUROC without testing
                # anything (FALSE_NEGATIVE_RISK discipline).
                #
                # The two are counted SEPARATELY because they mean opposite things and the
                # distinction is real in this corpus: vc33 has a measured frame-change base
                # rate of 1.000, so its states are excluded for having NO NEGATIVE (nothing to
                # discriminate against), not for being unwinnable. Reporting both under one
                # "no_positive" label would misdescribe the corpus.
                if positives == 0:
                    diagnostics["states_excluded_no_positive"] += 1
                else:
                    diagnostics["states_excluded_no_negative"] += 1
                continue
            diagnostics["states_kept"] += 1
            diagnostics["distinct_outcome_classes"] = diagnostics.get(
                "distinct_outcome_classes", []
            ) + [len(outcome_classes)]
            # The measured defect, per state: N candidates, N distinct (6,x,y) keys, and how
            # many distinct scores the LIVE router actually emits.
            diagnostics.setdefault("incumbent_collapse", []).append(
                {
                    "state_index": state_index,
                    "n_candidates": len(candidates),
                    "n_distinct_click_targets": len(
                        {(int(a.data["x"]), int(a.data["y"])) for a in candidates}
                    ),
                    "n_distinct_incumbent_scores": len(set(incumbent_scores))
                    if incumbent_scores
                    else None,
                }
            )
            rows.extend(state_rows)
    finally:
        diagnostics["n_forks"] = harvester.n_forks
        harvester.close()

    return rows, diagnostics


# ---------------------------------------------------------------------------------- AUROC


def auroc(scores: Sequence[float], labels: Sequence[float]) -> float | None:
    """Rank-based AUROC with proper tie handling.

    Tie handling is load-bearing here, not pedantry: the ``blind`` arm produces ONE distinct
    score for every candidate, so a tie-unaware implementation would report an arbitrary
    number instead of exactly 0.5, and the label-validity check would be meaningless.
    """

    pairs = [(float(s), float(l)) for s, l in zip(scores, labels)]
    n_pos = sum(1 for _s, l in pairs if l >= 0.5)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][0])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and pairs[order[j + 1]][0] == pairs[order[i]][0]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average_rank
        i = j + 1
    positive_rank_sum = sum(ranks[i] for i, (_s, l) in enumerate(pairs) if l >= 0.5)
    return float((positive_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def stratified_auroc(
    rows: Sequence[dict[str, Any]], scores: Sequence[float]
) -> tuple[float | None, int]:
    """WITHIN-STATE AUROC, pooled over states. This is the PRIMARY metric, and here is why.

    The live router only ever ranks candidates WITHIN ONE state (one frame's candidate list).
    It never compares a click on frame A against a click on frame B, so an AUROC pooled ACROSS
    states measures something the live path never does.

    That distinction is not academic -- it was MEASURED in this experiment's own first run. The
    incumbent v3 router emits exactly ONE distinct score per state (verified: 38-48 distinct
    click targets, 1 distinct score, at every harvested state) but a DIFFERENT constant per
    state, because the v3 features are frame-level. Pooled across states, where the positive
    base rate differs per state, that per-state constant scored AUROC 0.3105 -- an apparent
    "signal" that is entirely an artifact of cross-state base-rate variation and cannot
    influence a single live ranking decision. Within-state, it is exactly 0.5 by construction,
    which is the correct reading and the right basis for the label-validity check.

    Aggregation is a pooled Mann-Whitney statistic: sum of per-state U divided by the sum of
    per-state (n_pos * n_neg), so states contribute in proportion to how many discriminating
    pairs they actually contain.
    """

    by_state: dict[Any, list[tuple[float, float]]] = {}
    for row, score in zip(rows, scores):
        by_state.setdefault((row["game"], row["state_index"]), []).append(
            (float(score), float(row["label"]))
        )

    u_total = 0.0
    pair_total = 0.0
    n_states = 0
    for pairs in by_state.values():
        n_pos = sum(1 for _s, l in pairs if l >= 0.5)
        n_neg = len(pairs) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
        value = auroc([s for s, _l in pairs], [l for _s, l in pairs])
        if value is None:  # pragma: no cover - guarded by the class check above
            continue
        u_total += value * n_pos * n_neg
        pair_total += n_pos * n_neg
        n_states += 1
    if pair_total <= 0:
        return None, 0
    return float(u_total / pair_total), n_states


def paired_within_state_delta_bootstrap(
    rows: Sequence[dict[str, Any]],
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    n_boot: int = 4000,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Uncertainty on ``within_state_AUROC(a) - within_state_AUROC(b)``, PAIRED and WITHIN-state.

    WHY THIS EXISTS AND WHY IT IS NOT OPTIONAL. The decision-relevant number in this experiment
    is ``coord - static_salience``: the delta against the ordering that is ALREADY LIVE. It was
    previously reported as a bare point estimate (+0.0436) with no uncertainty at all, while
    every ``*_auroc`` block honestly carried ``ci95: null, n_games: 1``. A point estimate on a
    2-state / 20-positive corpus cannot distinguish a real improvement from noise, and this
    project has already paid for that exact mistake once: exp3540's paired p=0.135 "advantage"
    was retro-classified as a small-sample artifact.

    Resampling is done WITHIN each scored state, with replacement, preserving the design (the
    live router only ever ranks within one state). Both arms are recomputed on the SAME
    resample, so the delta is paired and the shared corpus noise cancels. A resampled state
    that comes out single-class contributes no discriminating pairs and is skipped by
    ``stratified_auroc`` for both arms alike; a replicate with no usable state at all is
    dropped and counted.
    """

    by_state: dict[Any, list[int]] = {}
    for index, row in enumerate(rows):
        by_state.setdefault((row["game"], row["state_index"]), []).append(index)
    if not by_state:
        return {"n_bootstrap": 0, "note": "no scored states"}

    labels = np.asarray([float(row["label"]) for row in rows], dtype=np.float64)
    array_a = np.asarray([float(value) for value in scores_a], dtype=np.float64)
    array_b = np.asarray([float(value) for value in scores_b], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    state_indices = [np.asarray(idx, dtype=np.int64) for idx in by_state.values()]

    deltas: list[float] = []
    dropped = 0
    for _replicate in range(int(n_boot)):
        u_a = u_b = pair_total = 0.0
        for idx in state_indices:
            picked = idx[rng.integers(0, len(idx), len(idx))]
            state_labels = labels[picked]
            n_pos = float((state_labels >= 0.5).sum())
            n_neg = float(len(state_labels) - n_pos)
            if n_pos <= 0 or n_neg <= 0:
                continue
            value_a = auroc(array_a[picked].tolist(), state_labels.tolist())
            value_b = auroc(array_b[picked].tolist(), state_labels.tolist())
            if value_a is None or value_b is None:  # pragma: no cover - guarded above
                continue
            weight = n_pos * n_neg
            u_a += value_a * weight
            u_b += value_b * weight
            pair_total += weight
        if pair_total <= 0:
            dropped += 1
            continue
        deltas.append(float(u_a / pair_total - u_b / pair_total))

    if not deltas:
        return {"n_bootstrap": int(n_boot), "note": "every replicate was degenerate"}
    array = np.asarray(deltas, dtype=np.float64)
    return {
        "n_bootstrap": int(n_boot),
        "n_replicates_used": int(array.size),
        "n_replicates_dropped_degenerate": int(dropped),
        "delta_mean": float(array.mean()),
        "delta_ci95": [float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))],
        "fraction_replicates_le_zero": float((array <= 0.0).mean()),
        "excludes_zero": bool(float(np.percentile(array, 2.5)) > 0.0),
        "resampling": "rows resampled with replacement WITHIN each scored state; arms paired",
    }


# ----------------------------------------------------------------------------------- arms


def _temporal_split(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit on the game's EARLIER states, score the LATER ones.

    Prospective by construction: no row is ever scored by a model that has seen it. That is
    the offline stand-in for what the live agent actually does -- it can only have learned
    from what it has already tried.
    """

    state_indices = sorted({int(row["state_index"]) for row in rows})
    if len(state_indices) < 2:
        return [], []
    cut = state_indices[max(1, len(state_indices) // 2) - 1]
    fit = [row for row in rows if int(row["state_index"]) <= cut]
    score = [row for row in rows if int(row["state_index"]) > cut]
    return fit, score


def run_arms_for_game(rows: Sequence[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    """Score all four arms on one game's held-out later states."""

    from carnot.agentic.arc_click_target_features import (
        CLICK_TARGET_FEATURE_DIM,
        OnlineClickTargetDiscriminator,
    )

    fit_rows, score_rows = _temporal_split(rows)
    result: dict[str, Any] = {
        "n_fit": len(fit_rows),
        "n_score": len(score_rows),
        "n_states_fit": len({r["state_index"] for r in fit_rows}),
        "n_states_score": len({r["state_index"] for r in score_rows}),
    }
    if not fit_rows or not score_rows:
        result["note"] = "insufficient states for a temporal split"
        return result

    labels = [float(row["label"]) for row in score_rows]
    result["n_pos"] = int(sum(1 for l in labels if l >= 0.5))
    result["n_neg"] = int(len(labels) - result["n_pos"])

    # LABEL COMPOSITION -- load-bearing for interpretation, not decoration. The label is
    # (changed OR leveled_up), and the two components are NOT equally hard. The frame-change
    # component is the EASY one: the static salience sort already reaches AUROC 0.934 on it
    # on lp85 (measured). The level-up component is the hard, scarce one. Reporting the split
    # is what stops a strong pooled number being read as a strong level-up result.
    result["n_levels_up"] = int(sum(1 for row in score_rows if bool(row["levels_up"])))
    result["n_changed_only"] = int(
        sum(1 for row in score_rows if bool(row["changed"]) and not bool(row["levels_up"]))
    )
    result["label_is_predominantly_frame_change"] = bool(
        result["n_pos"] > 0 and result["n_levels_up"] < result["n_pos"]
    )
    # MEASURED, not assumed: is the ``OR levels_up`` disjunct doing any work at all? If every
    # level-up also changed the frame then the label is IDENTICALLY "the frame changed", and
    # reporting n_levels_up alongside n_pos would misread as level-ups supplying positives of
    # their own. Counted rather than asserted so a future corpus where it differs is visible.
    result["n_levels_up_without_change"] = int(
        sum(1 for row in score_rows if bool(row["levels_up"]) and not bool(row["changed"]))
    )
    result["n_label_equals_changed"] = int(
        sum(1 for row in score_rows if bool(row["label"] >= 0.5) == bool(row["changed"]))
    )
    result["label_is_measured_identical_to_frame_change"] = bool(
        result["n_label_equals_changed"] == len(score_rows)
    )

    # arm BLIND: the incumbent v3 router's OWN scores, measured during harvest -- not a
    # stand-in. Falls back to a constant only if the checkpoint was unavailable, in which case
    # the constant is itself the honest representation of a coordinate-blind scorer.
    measured = [row.get("incumbent_score") for row in score_rows]
    if all(value is not None for value in measured):
        blind_scores = [float(value) for value in measured]
        result["blind_scores_are_measured_from_the_live_router"] = True
    else:
        blind_scores = [1.0 for _row in score_rows]
        result["blind_scores_are_measured_from_the_live_router"] = False
    result["blind"] = auroc(blind_scores, labels)
    result["blind_within_state"], result["blind_n_states"] = stratified_auroc(
        score_rows, blind_scores
    )
    result["blind_distinct_scores"] = len(set(blind_scores))
    result["blind_distinct_scores_per_state"] = [
        len({row.get("incumbent_score") for row in score_rows if row["state_index"] == index})
        for index in sorted({row["state_index"] for row in score_rows})
    ]

    # arm COORD: the 21 coordinate-aware features, fit online on the earlier states only.
    head = OnlineClickTargetDiscriminator(dim=CLICK_TARGET_FEATURE_DIM)
    for row in fit_rows:
        head.observe(row["features"], row["label"])
    fitted = head.fit()
    result["coord_head_fitted"] = bool(fitted)
    result["coord_head_stats"] = head.stats()
    if fitted:
        coord_scores = [head.proba(row["features"]) for row in score_rows]
        result["coord"] = auroc(coord_scores, labels)
        result["coord_within_state"], result["coord_n_states"] = stratified_auroc(
            score_rows, coord_scores
        )
        result["coord_distinct_scores"] = len(set(coord_scores))
    else:
        result["coord"] = None
        result["coord_within_state"] = None
        result["coord_distinct_scores"] = 0

    # arm RANDOM: coordinate-aware but uninformative -- isolates "sees coordinates" from
    # "knows something". This calls the ACTUALLY-SHIPPED RandomCandidateRouter, not a bare RNG,
    # so the control is the same object the repo already uses as its positive control: it hashes
    # candidate_action_key, which carries (6, x, y), giving distinct scores per target.
    from carnot.agentic.arc_discriminative_router import RandomCandidateRouter

    class _KeyedAction:
        """Minimal ArcAction-shaped view so RandomCandidateRouter sees the real (6, x, y) key."""

        def __init__(self, x: int, y: int) -> None:
            self.action_id = 6
            self.data = {"x": int(x), "y": int(y)}

    class _StateFrame:
        """Distinct per-state frame stand-in: RandomCandidateRouter also hashes the frame."""

        def __init__(self, game: str, state_index: int) -> None:
            self.frame = np.full((4, 4), int(state_index) % 32767, dtype=np.int16)
            self.game_id = str(game)

    random_router = RandomCandidateRouter(seed=int(seed))
    random_scores = [
        random_router._score(
            _StateFrame(row["game"], row["state_index"]), _KeyedAction(row["x"], row["y"])
        )
        for row in score_rows
    ]
    result["random"] = auroc(random_scores, labels)
    result["random_within_state"], _ = stratified_auroc(score_rows, random_scores)
    result["random_arm_uses_shipped_router"] = True
    result["random_distinct_scores"] = len(set(random_scores))

    # arm STEP_INDEX: zero perception. coord must beat this or the harness is broken.
    step_scores = [float(row["state_index"]) for row in score_rows]
    result["step_index"] = auroc(step_scores, labels)
    # Within-state this is a CONSTANT (every row in a state shares its index), so it is
    # exactly 0.5 -- which is the point: a zero-perception feature cannot rank candidates.
    result["step_index_within_state"], _ = stratified_auroc(score_rows, step_scores)

    # The incumbent STATIC salience sort, which is what actually orders clicks live once the
    # router ties. Lower rank should mean "more likely", hence the negation.
    static_scores = [-float(row["salience_rank"]) for row in score_rows]
    result["static_salience"] = auroc(static_scores, labels)
    result["static_salience_within_state"], _ = stratified_auroc(score_rows, static_scores)

    # THE DECISION-RELEVANT NUMBER, WITH UNCERTAINTY. coord - static is what says whether this
    # head would improve the ordering that is already live; a point estimate alone cannot tell
    # a real gain from resampling noise on a 2-state corpus (exp3540's lesson).
    if fitted:
        coord_within = result.get("coord_within_state")
        static_within = result.get("static_salience_within_state")
        result["coord_minus_static_within_state"] = (
            None
            if coord_within is None or static_within is None
            else float(coord_within) - float(static_within)
        )
        result["coord_minus_static_bootstrap"] = paired_within_state_delta_bootstrap(
            score_rows, coord_scores, static_scores, seed=int(seed)
        )

    # NOISE FLOOR, measured on THIS corpus: how far does a provably-uninformative but
    # coordinate-aware arm wander when only its seed changes? If the coord-minus-static delta is
    # smaller than this spread, the delta is inside the noise the corpus generates for free.
    noise_values: list[float] = []
    for offset in range(RANDOM_NOISE_SEEDS):
        probe = RandomCandidateRouter(seed=int(seed) + 1000 + offset)
        probe_scores = [
            probe._score(
                _StateFrame(row["game"], row["state_index"]), _KeyedAction(row["x"], row["y"])
            )
            for row in score_rows
        ]
        value, _n = stratified_auroc(score_rows, probe_scores)
        if value is not None:
            noise_values.append(float(value))
    if noise_values:
        noise_array = np.asarray(noise_values, dtype=np.float64)
        result["random_arm_seed_noise"] = {
            "n_seeds": int(noise_array.size),
            "mean": float(noise_array.mean()),
            "sd": float(noise_array.std(ddof=1)) if noise_array.size > 1 else None,
            "min": float(noise_array.min()),
            "max": float(noise_array.max()),
            "metric": "within_state_auroc of the shipped RandomCandidateRouter, seed varied only",
        }

    # HARD SLICE: among the clicks that DID change something, which ones LEVEL UP?
    #
    # The label used by every arm above is ``changed OR levels_up``, so the easy component
    # ("does this click do anything at all") can carry the whole headline. This slice removes
    # that component: it conditions on ``changed`` and asks the question that actually matters
    # live -- which of the clicks that DO something make PROGRESS.
    #
    # THE PRIOR VERSION OF THIS ARM WAS STRUCTURALLY DEAD and produced nothing. It used
    # ``hard_labels = row["label"]`` on rows filtered by ``changed``; since
    # ``label = 1.0 if (changed or levels_up)``, every hard row had label 1.0, ``auroc``
    # returned None on the single class, and the artifact shipped ``n_hard == n_pos`` with
    # ``coord_hard_only: null`` / ``pooled_coord_hard_only_auroc.n_games: 0``. The comment that
    # stood here also asserted "pooled 0.8747 collapses to 0.5544 hard-only, 84.6% trivial
    # no-op negatives" -- figures the shipped label could not produce; they were stale numbers
    # from an earlier label presented as current measurements, and are deleted rather than
    # re-cited. The label below (``levels_up`` among ``changed``) is what makes the arm run.
    hard = [row for row in score_rows if bool(row["changed"])]
    hard_labels = [float(bool(row["levels_up"])) for row in hard]
    result["n_hard"] = len(hard)
    result["n_hard_pos"] = int(sum(1 for value in hard_labels if value >= 0.5))
    result["n_hard_neg"] = int(len(hard_labels) - result["n_hard_pos"])
    result["hard_slice_label"] = "levels_up, conditioned on changed (NOT the pooled label)"
    if hard and fitted:
        result["coord_hard_only"] = auroc(
            [head.proba(row["features"]) for row in hard], hard_labels
        )
        result["static_salience_hard_only"] = auroc(
            [-float(row["salience_rank"]) for row in hard], hard_labels
        )
        result["random_hard_only"] = auroc(
            [
                random_router._score(
                    _StateFrame(row["game"], row["state_index"]), _KeyedAction(row["x"], row["y"])
                )
                for row in hard
            ],
            hard_labels,
        )
    return result


def _random_router_control_distinct_scores() -> int:
    """Confirm the shipped RandomCandidateRouter really is coordinate-AWARE.

    It keys on ``candidate_action_key``, which carries (6, x, y) -- so distinct targets get
    distinct scores. That is exactly the property the real router lacks, which is why it is
    the right positive control for "coordinate-aware".
    """

    from carnot.agentic.arc_discriminative_router import RandomCandidateRouter

    class _F:
        def __init__(self) -> None:
            self.frame = np.zeros((8, 8), dtype=np.int16)

    class _A:
        def __init__(self, x: int, y: int) -> None:
            self.action_id = 6
            self.data = {"x": x, "y": y}

    router = RandomCandidateRouter()
    frame = _F()
    return len({router._score(frame, _A(x, x + 1)) for x in range(10)})


def repair_check_passes(value: float | None, blind: float | None) -> bool:
    """The ORIGINAL pre-registered expression, as a pure function of one arm's AUROC.

    Module-level and applied to EVERY arm, not just the treatment, because that is how the
    artifact's ``also_passed_by_arms`` field is computed -- the fact that the shipped
    uninformative ``RandomCandidateRouter`` clears the same bar is measured, not asserted, and a
    test pins it. Within-state ``blind`` is 0.5 by construction, so this reduces to
    ``value >= 0.60``; keeping the second conjunct here keeps the function faithful to the
    expression that was actually pre-registered.
    """

    return bool(
        value is not None and blind is not None and value >= 0.60 and (value - blind) >= 0.10
    )


# ---------------------------------------------------------------------------------- main


def _pool(per_game: dict[str, dict[str, Any]], key: str) -> dict[str, Any]:
    """Aggregate one metric over games -- and SAY SO when there is only one game.

    The ``pooled_*`` / ``within_state_*`` key names are historical, and with a single
    contributing game they are NOT pooled: the "mean" is that one game's value and ``ci95`` is
    null because a one-game sample has no between-game spread. Emitting
    ``single_game_not_pooled`` makes that unmissable instead of leaving a reader to notice
    ``n_games: 1``. This run's smoke corpus is exactly that case (vc33 contributes zero states:
    its measured frame-change base rate is 1.000, so every state is excluded for having no
    negative).
    """

    values = [row[key] for row in per_game.values() if isinstance(row.get(key), (int, float))]
    if not values:
        return {
            "mean": None,
            "n_games": 0,
            "ci95": None,
            "values": [],
            "single_game_not_pooled": False,
            "no_contributing_games": True,
        }
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    if len(values) > 1:
        # Normal-approximation CI over GAMES (the unit of independence), honestly labelled --
        # with 2-5 games this is wide and should not be read as tight.
        half = 1.96 * float(array.std(ddof=1)) / float(np.sqrt(len(values)))
        ci = [mean - half, mean + half]
    else:
        ci = None
    return {
        "mean": mean,
        "n_games": len(values),
        "ci95": ci,
        "values": [float(v) for v in values],
        "single_game_not_pooled": len(values) == 1,
        "no_contributing_games": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="2 games, 4 states each")
    parser.add_argument("--games", nargs="*", default=None)
    parser.add_argument("--max-states", type=int, default=None)
    parser.add_argument("--max-clicks", type=int, default=DEFAULT_MAX_CLICKS)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    started = time.time()
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    preconditions = check_preconditions()
    missing = [check["resource"] for check in preconditions if not check["available"]]

    games = tuple(args.games) if args.games else (SMOKE_GAMES if args.smoke else CLICK_GAMES)
    max_states = args.max_states or (SMOKE_MAX_STATES if args.smoke else DEFAULT_MAX_STATES)
    out_path = Path(args.out) if args.out else ARTIFACT_PATH

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT_ID,
        "experiment_id": EXPERIMENT_ID,
        "title": "Coordinate-aware online click-target discrimination (stage 1, offline)",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "mode": "smoke" if args.smoke else "full",
        "games_requested": list(games),
        "max_states_per_game": max_states,
        "max_clicks_per_state": int(args.max_clicks),
        "random_seed": RANDOM_SEED,
        "preconditions_checked": preconditions,
        "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "verifier_is_oracle": False,
        "solve_provenance": "development_proxy",
        "online_within_game_only": True,
        "cross_game_checkpoint_loaded": False,
        "field_provenance": {
            "inference_substrate": {
                "principle": (
                    "Pure Python env-stepping plus numpy; no GGUF load, no CUDA, no LLM call. "
                    "Declaring the substrate explicitly is what selects the correct duration "
                    "floor (0.01 s) instead of the 60 s live-inference floor, and prevents a "
                    "false DURATION_TOO_SHORT flag on an honest run."
                )
            },
            "verifier_is_oracle": {
                "principle": (
                    "The DISCRIMINATOR scores from perception features only and never executes "
                    "a candidate. The LABEL is produced by an executable oracle (fork the env "
                    "and step the click) at CORPUS-CONSTRUCTION time only -- the live agent "
                    "never calls it at inference. Declared false with that circularity stated "
                    "explicitly per the Circularity/Oracle-Distinctness discipline."
                )
            },
            "solve_provenance": {
                "principle": (
                    "development_proxy: this is an offline measurement over public games, not "
                    "a live-agent self-discovery solve. It emits no solve claim at all."
                )
            },
            "online_within_game_only": {
                "principle": (
                    "The retired direction is cross-game learned value transfer "
                    "(exclusion_manifest id cross_game_value_transfer_retired_exp4342_v401, "
                    "operator_reopen_required). Fitting per game on a temporal split keeps this "
                    "outside that scope -- and is the only thing a hidden-game agent could do."
                )
            },
            "random_seed": {
                "principle": (
                    "Determinism is the precondition for reproducibility; without a seed no "
                    "third party can re-run and confirm or refute the numbers."
                )
            },
            "reproducibility_checksum": {
                "principle": (
                    "Content-addressed hash of the harvested corpus catches silent env or "
                    "generator drift between this artifact and a future replication."
                )
            },
            "duration_s": {
                "principle": (
                    "Real compute takes wall-clock time; a missing or implausibly-short "
                    "duration is the load-bearing fabrication signal."
                )
            },
            "label_definition": {
                "principle": (
                    "A label that is not causally downstream of the CLICK is learnable with "
                    "zero perception -- the exp5835 defect (level_progress is a pure function "
                    "of step_index). Stating the label explicitly makes that auditable."
                )
            },
            "preconditions_checked": {
                "principle": (
                    "Records WHICH resources were verified before measuring; pre-empts the "
                    "fabrication mode where a missing resource is silently papered over."
                )
            },
        },
        "label_definition": (
            "label = 1 iff ACTUALLY STEPPING this click against the offline env changed the "
            "settled grid or advanced levels_completed (measured by execution, causally "
            "downstream of the click). Explicitly NOT arc_human_replay_corpus.level_progress, "
            "which is a pure function of step_index (the exp5835 defect), and explicitly NOT "
            "an exact-(x,y) match against a banked route (measured 0/6 on lp85 and tn36 -- a "
            "silent zero-positive corpus)."
        ),
        "stage": "stage_1_offline_feasibility_only",
        "offline_auc_licenses_nothing": (
            "Offline AUROC licenses NOTHING about live search. exp4545's 0.725-AUROC "
            "discriminator REGRESSED live search, which is why the live agent's "
            "SUBMITTED_VALUE_WEIGHT is pinned at 1e-12. The terminal gate for the router "
            "change is a LIVE A/B on banked levels, stratified by "
            "frame_change_scorer.as_dict()['frame_diff_ground_truth_validated'] per game "
            "(the router only fully owns click order where that scorer is unvalidated). That "
            "A/B is a SEPARATE, LATER experiment; this one does not attempt it."
        ),
        "prior_work_extended": [
            {
                "id": "REQ-ARC-FCP-5758",
                "verdict": "clean NULL; gap logged as a Missing-Verifier Gap",
                "difference": (
                    "5758 reordered the SAME static salience formula (small-object-first) and "
                    "found no monotonic-in-area formula could surface the winners. This builds "
                    "the missing discriminating SIGNAL that entry called for, rather than "
                    "another ordering of the same one."
                ),
            },
            {
                "id": "exp4545",
                "verdict": "0.725 offline AUROC, REGRESSED live search",
                "difference": (
                    "Cross-game trained value head applied live at full weight. This is an "
                    "ONLINE within-episode head, DEFAULT OFF, with a live A/B as the terminal "
                    "gate rather than an offline AUROC."
                ),
            },
            {
                "id": "exp5835",
                "verdict": "VOIDED (unpassable gate conjunction + step-index label)",
                "difference": (
                    "The label here is measured by executing the click, and the gate contains "
                    "no conjunct asserting anything about another arm's value. A "
                    "zero-perception step_index arm is included as an explicit control."
                ),
            },
            {
                "id": "arc_inert_click_pruner.InertClickSigPruner",
                "verdict": "shipped, default off",
                "difference": (
                    "That is a BINARY VETO keyed on an EXACT 4-tuple signature and never "
                    "orders survivors; this is a CONTINUOUS 21-feature ranker that "
                    "generalizes across similar-but-not-identical blobs. Complementary."
                ),
            },
        ],
    }

    if missing:
        artifact["honest_verdict"] = "blocked_" + "_and_".join(missing)
        artifact["duration_s"] = round(time.time() - started, 4)
        artifact["reproducibility_checksum"] = "sha256:" + hashlib.sha256(b"blocked").hexdigest()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        print(f"BLOCKED: missing preconditions {missing}; wrote {out_path}")
        return 1

    from carnot.agentic.arc_click_target_features import CLICK_TARGET_FEATURE_NAMES

    artifact["feature_names"] = list(CLICK_TARGET_FEATURE_NAMES)
    artifact["random_router_control_distinct_scores"] = _random_router_control_distinct_scores()

    all_rows: list[dict[str, Any]] = []
    per_game: dict[str, dict[str, Any]] = {}
    harvest_diagnostics: list[dict[str, Any]] = []

    for game in games:
        print(f"[{game}] harvesting ...", flush=True)
        try:
            rows, diagnostics = harvest_game(
                game, max_states=max_states, max_clicks=int(args.max_clicks)
            )
        except Exception as exc:
            harvest_diagnostics.append({"game": game, "error": repr(exc)})
            print(f"[{game}] harvest error: {exc!r}", flush=True)
            continue
        harvest_diagnostics.append(diagnostics)
        print(
            f"[{game}] states_kept={diagnostics['states_kept']} "
            f"excluded_no_positive={diagnostics['states_excluded_no_positive']} "
            f"excluded_no_negative={diagnostics['states_excluded_no_negative']} "
            f"rows={len(rows)} forks={diagnostics['n_forks']}",
            flush=True,
        )
        if not rows:
            continue
        all_rows.extend(rows)
        arms = run_arms_for_game(rows, seed=RANDOM_SEED + len(per_game))
        arms["n_rows"] = len(rows)
        per_game[game] = arms
        print(
            f"[{game}] blind={arms.get('blind')} coord={arms.get('coord')} "
            f"random={arms.get('random')} step_index={arms.get('step_index')} "
            f"static={arms.get('static_salience')}",
            flush=True,
        )

    artifact["harvest_diagnostics"] = harvest_diagnostics
    artifact["per_game"] = per_game
    artifact["n_labelled_rows_total"] = len(all_rows)
    artifact["n_positives_total"] = int(sum(1 for r in all_rows if r["label"] >= 0.5))
    artifact["n_states_total"] = len({(r["game"], r["state_index"]) for r in all_rows})

    for arm in ("blind", "coord", "random", "step_index", "static_salience", "coord_hard_only"):
        artifact[f"pooled_{arm}_auroc"] = _pool(per_game, arm)
    for arm in ("blind", "coord", "random", "step_index", "static_salience"):
        artifact[f"within_state_{arm}_auroc"] = _pool(per_game, f"{arm}_within_state")

    artifact["primary_metric"] = {
        "name": "within_state_auroc",
        "principle": (
            "The live router only ever ranks candidates WITHIN one frame's candidate list, so "
            "the within-state AUROC is the only figure that corresponds to a decision the live "
            "path actually makes. The across-state pooled figure is reported alongside it for "
            "completeness and is NOT the metric of record -- measured here, the incumbent's "
            "per-state CONSTANT score reaches an across-state AUROC of ~0.31 purely from "
            "cross-state base-rate variation while being provably 0.5 within-state, i.e. "
            "incapable of influencing any single ranking."
        ),
    }

    blind_values = artifact["within_state_blind_auroc"]["values"]
    coord_mean = artifact["within_state_coord_auroc"]["mean"]
    blind_mean = artifact["within_state_blind_auroc"]["mean"]
    random_mean = artifact["within_state_random_auroc"]["mean"]
    step_mean = artifact["within_state_step_index_auroc"]["mean"]
    artifact["across_state_reference"] = {
        "blind": artifact["pooled_blind_auroc"]["mean"],
        "coord": artifact["pooled_coord_auroc"]["mean"],
        "note": (
            "Reference only, NOT the metric of record -- see primary_metric. An across-state "
            "figure can move on cross-state base-rate variation alone."
        ),
    }

    # Label-validity check, reported SEPARATELY from the gate (the exp5835 lesson).
    label_valid = bool(blind_values) and all(0.45 <= value <= 0.55 for value in blind_values)
    artifact["label_validity_check"] = {
        "status": "valid" if label_valid else "invalid",
        "blind_per_game": blind_values,
        "band": [0.45, 0.55],
        "metric": "within_state_auroc",
        "principle": (
            "A coordinate-blind scorer CANNOT beat chance at ranking candidates WITHIN a "
            "state, because it emits one identical score for all of them (measured: 1 distinct "
            "score across 38-48 distinct targets at every harvested state). A within-state "
            "blind AUROC outside the band therefore means the label leaks, in which case NO "
            "arm is interpretable -- reported as invalid rather than as a pass or a fail. The "
            "check is deliberately WITHIN-state: the across-state figure legitimately deviates "
            "from 0.5 on base-rate variation alone (measured 0.31), which would make an "
            "across-state band check reject valid runs."
        ),
    }

    static_mean = artifact["within_state_static_salience_auroc"]["mean"]

    # ------------------------------------------------------------------ the repair CHECK
    #
    # This was originally emitted as ``pre_registered_gate`` and its ``passed: true`` was the
    # artifact's headline. That was misleading and the key is deliberately GONE: the expression
    # is a single threshold in disguise (``blind`` is pinned at exactly 0.5 within-state by
    # construction, so the second conjunct reduces to the first) and -- MEASURED below, against
    # this run's own numbers -- the very controls introduced to isolate "coordinate-aware" from
    # "informative" also clear it. A check that the shipped uninformative RandomCandidateRouter
    # passes cannot be the evidence that a change is worth shipping. It is retained VERBATIM,
    # renamed to what it actually tests: the coordinate-blindness defect is repaired.
    def _repair_expression(value: float | None) -> bool:
        return repair_check_passes(value, blind_mean)

    arm_means = {
        "coord": coord_mean,
        "random": random_mean,
        "static_salience": static_mean,
        "step_index": step_mean,
        "blind": blind_mean,
    }
    artifact["coordinate_blindness_repair_check"] = {
        "expression": (
            "within_state_coord_auroc >= 0.60 AND "
            "(within_state_coord_auroc - within_state_blind_auroc) >= 0.10"
        ),
        "metric": "within_state_auroc",
        "renamed_from": "pre_registered_gate",
        "is_not_an_informativeness_gate": True,
        "coord_auroc": coord_mean,
        "blind_auroc": blind_mean,
        "delta": (None if coord_mean is None or blind_mean is None else coord_mean - blind_mean),
        "passed": _repair_expression(coord_mean),
        "also_passed_by_arms": [
            name
            for name, value in arm_means.items()
            if name != "coord" and _repair_expression(value)
        ],
        "arm_means_evaluated": arm_means,
        "principle": (
            "Within-state, the blind arm is a per-state CONSTANT, so its AUROC is exactly 0.5 "
            "by arithmetic and the second conjunct collapses into the first: the expression is "
            "'coord >= 0.60' and nothing more. It confirms the defect (one identical score for "
            "every click) is gone. It does NOT show the new scores are BETTER than what already "
            "orders clicks live -- ``also_passed_by_arms`` names the arms in THIS run that clear "
            "the same bar, including uninformative ones. Read informativeness_gate for that."
        ),
    }

    # ------------------------------------------------------- the informativeness GATE
    #
    # ADDED POST-HOC, after review, and labelled as such rather than backdated. The reason is
    # that the original expression's pass region CONTAINS REGRESSIONS: static_salience sits
    # inside it, so a coord of 0.70 would have reported "passed" while ordering clicks WORSE
    # than the incumbent static sort it would replace. This gate is the decision-relevant one:
    # does the head beat the ordering that is already live, by more than resampling noise?
    per_game_bootstraps = {
        game: arms["coord_minus_static_bootstrap"]
        for game, arms in per_game.items()
        if isinstance(arms.get("coord_minus_static_bootstrap"), dict)
        and arms["coord_minus_static_bootstrap"].get("delta_ci95") is not None
    }
    excludes_zero_per_game = {
        game: bool(boot.get("excludes_zero")) for game, boot in per_game_bootstraps.items()
    }
    informativeness_passed = bool(
        coord_mean is not None
        and coord_mean >= 0.60
        and excludes_zero_per_game
        and all(excludes_zero_per_game.values())
    )
    artifact["informativeness_gate"] = {
        "expression": (
            "within_state_coord_auroc >= 0.60 AND "
            "lower_bound(bootstrap_ci95(within_state_coord_auroc - "
            "within_state_static_salience_auroc)) > 0 for every contributing game"
        ),
        "metric": "within_state_auroc",
        "added_post_hoc_after_review": True,
        "why_added": (
            "The originally pre-registered expression's pass region contains REGRESSIONS: "
            "static_salience (the incumbent ordering) lies inside it, so a coord value worse "
            "than the incumbent could report 'passed'. A gate that cannot distinguish "
            "improvement from regression carries almost no information."
        ),
        "coord_auroc": coord_mean,
        "static_salience_auroc": static_mean,
        "delta": (None if coord_mean is None or static_mean is None else coord_mean - static_mean),
        "per_game_bootstrap": per_game_bootstraps,
        "per_game_ci_excludes_zero": excludes_zero_per_game,
        "passed": informativeness_passed,
        "is_gate_of_record": True,
        "principle": (
            "The only decision this experiment can inform is 'would this reorder clicks BETTER "
            "than what already does'. That is coord - static, and it needs an interval, not a "
            "point estimate: exp3540's paired p=0.135 'advantage' was retro-classified as a "
            "small-sample artifact at this exact shape. Passing this gate is still NOT a licence "
            "to flip the live flag -- the terminal gate is a LIVE A/B (exp4545)."
        ),
    }
    artifact["gate_of_record"] = "informativeness_gate"
    artifact["informativeness_established"] = informativeness_passed

    artifact["honest_comparator"] = {
        "name": "coord_minus_static_within_state",
        "within_state_coord_auroc": coord_mean,
        "within_state_static_salience_auroc": static_mean,
        "delta": (None if coord_mean is None or static_mean is None else coord_mean - static_mean),
        "delta_has_uncertainty": bool(per_game_bootstraps),
        "per_game_bootstrap": per_game_bootstraps,
        "single_game_bootstrap": (
            next(iter(per_game_bootstraps.values())) if len(per_game_bootstraps) == 1 else None
        ),
        "random_arm_seed_noise_per_game": {
            game: arms.get("random_arm_seed_noise")
            for game, arms in per_game.items()
            if arms.get("random_arm_seed_noise")
        },
        "informativeness_established": informativeness_passed,
        "principle": (
            "Beating the BLIND arm is trivial -- it is a constant within a state. The static "
            "area x colour-rarity salience sort is what actually orders clicks live once the "
            "router ties, so the delta against IT is what says whether this signal is worth "
            "anything, and it is the HEADLINE figure of this artifact -- not the large delta "
            "against the constant blind arm. Report it WITH its bootstrap interval: a delta "
            "whose CI includes 0, or that is smaller than the seed-noise spread of the "
            "uninformative random arm on the same corpus, establishes NOTHING over the "
            "incumbent ordering. Read exp4545 before concluding even a significant delta here "
            "will survive live search."
        ),
    }
    artifact["controls"] = {
        "coord_beats_random": (
            None if coord_mean is None or random_mean is None else coord_mean > random_mean
        ),
        "coord_beats_step_index": (
            None if coord_mean is None or step_mean is None else coord_mean > step_mean
        ),
        "principle": (
            "coord must beat BOTH the coordinate-aware-but-uninformative control (isolating "
            "'informative' from 'coordinate-aware') and the zero-perception step-index "
            "control (the exp5835 lesson as a gate). Failing either means the harness or the "
            "label is broken, not that the signal is strong."
        ),
    }

    # What the label actually consists of, aggregated. A pooled AUROC on a
    # predominantly-frame-change label is NOT evidence about level-up prediction, and the
    # incumbent static salience sort is already strong on the frame-change component (measured
    # 0.934 on lp85). Say so in the artifact rather than letting a big number speak.
    total_levels_up = int(sum(1 for r in all_rows if bool(r["levels_up"])))
    n_label_equals_changed = int(
        sum(1 for r in all_rows if bool(r["label"] >= 0.5) == bool(r["changed"]))
    )
    label_is_frame_change = bool(all_rows) and n_label_equals_changed == len(all_rows)
    artifact["label_composition"] = {
        "n_rows": len(all_rows),
        "n_positive": artifact["n_positives_total"],
        "n_levels_up": total_levels_up,
        "n_levels_up_without_change": int(
            sum(1 for r in all_rows if bool(r["levels_up"]) and not bool(r["changed"]))
        ),
        "n_changed": int(sum(1 for r in all_rows if bool(r["changed"]))),
        "n_changed_only": int(
            sum(1 for r in all_rows if bool(r["changed"]) and not bool(r["levels_up"]))
        ),
        "n_label_equals_changed": n_label_equals_changed,
        "label_is_measured_identical_to_frame_change": label_is_frame_change,
        "level_up_positives_meet_n30_floor": total_levels_up >= 30,
        "interpretation": (
            "The label is WRITTEN as (frame changed OR leveled up), but MEASURED on this corpus "
            "the disjunct does no work: n_levels_up_without_change is the count of level-ups "
            "that did NOT also change the frame, and when "
            "label_is_measured_identical_to_frame_change is true the label is IDENTICALLY 'did "
            "the frame change'. So n_levels_up must NOT be read as supplying positives of its "
            "own -- every level-up positive is already a frame-change positive. Frame-change is "
            "the EASY component (the incumbent STATIC salience sort reaches 0.936 within-state "
            "on it here). A strong AUROC on this label therefore shows only that "
            "coordinate-aware features carry signal where the V3 ROUTER SPECIFICALLY carries "
            "none -- scoped to that router, NOT to this project's frame-change capability in "
            "general, which arc_frame_change_predictor already targets (see "
            "baselines_not_run_and_why). It is not a level-up-prediction result; per exp4545 it "
            "is not a live-search result; and per informativeness_gate it is not by itself "
            "evidence of any gain over the ordering already live. For the level-up question "
            "specifically read the hard slice (levels_up conditioned on changed)."
        ),
    }

    # BASELINES DELIBERATELY NOT RUN, named rather than silently omitted. The label above is
    # measured-identical to frame-change, and this repo already ships modules purpose-built for
    # frame-change prediction from (frame, candidate). Leaving them unmentioned would let the
    # artifact read as if no prior capability existed for this exact target.
    frame_change_checkpoints = sorted(
        str(path.relative_to(REPO))
        for path in (REPO / "models").glob("*")
        if "frame" in path.name.lower() and "change" in path.name.lower()
    )
    artifact["baselines_not_run_and_why"] = {
        "module": "python/carnot/agentic/arc_frame_change_predictor.py",
        "scorers": [
            "FrameChangeScorer",
            "LiveActionEffectScorer",
            "GroundTruthValidatedFrameChangeScorer",
            "BehaviorActionPrior",
            "ActionEffectExpansionPrior",
        ],
        "frame_change_checkpoints_found_in_models_dir": frame_change_checkpoints,
        "reason": (
            "FrameChangeScorer requires a trained SmallFrameChangeCNN checkpoint and none "
            "exists in models/ (measured above -- the list is empty), so it cannot be run as an "
            "arm here without first training one, which is out of this experiment's scope. "
            "GroundTruthValidatedFrameChangeScorer is not a comparable offline arm at all: it "
            "returns 0.0 until it self-validates against observed transitions, and it is "
            "ALREADY DOWNSTREAM of this router in arc_graph_explore (which is exactly why the "
            "live A/B must stratify on frame_diff_ground_truth_validated). The honest "
            "consequence: this experiment does NOT establish that coordinate-aware features "
            "beat the project's existing frame-change capability -- only that they beat the V3 "
            "ROUTER, which carries no coordinate signal at all."
        ),
    }

    # THE HARD SLICE, PROMOTED TO TOP LEVEL. It is the load-bearing result: it strips the easy
    # "did anything happen at all" component out of the label and asks the question the live
    # router exists to answer. It is reported whether or not it flatters the treatment.
    artifact["hard_slice_summary"] = {
        "label": "levels_up, conditioned on changed",
        "why": (
            "The pooled label is measured-identical to frame-change, so a high pooled AUROC can "
            "be carried entirely by 'does this click do anything at all'. Conditioning on "
            "changed removes that component and leaves the discrimination that matters live: "
            "among clicks that DO something, which ones make PROGRESS."
        ),
        "per_game": {
            game: {
                "n_hard": arms.get("n_hard"),
                "n_hard_pos": arms.get("n_hard_pos"),
                "n_hard_neg": arms.get("n_hard_neg"),
                "coord": arms.get("coord_hard_only"),
                "static_salience": arms.get("static_salience_hard_only"),
                "random": arms.get("random_hard_only"),
            }
            for game, arms in per_game.items()
        },
        "previously_inoperative": (
            "This arm shipped dead: it filtered rows on `changed` while reusing the pooled "
            "label (`changed OR levels_up`), so every row in it was a positive, AUROC was "
            "undefined by construction, and the artifact emitted n_hard == n_pos with "
            "coord_hard_only: null. Its source comment also cited 'pooled 0.8747 collapses to "
            "0.5544 hard-only, 84.6% trivial no-op negatives' -- figures that label could not "
            "produce. The label is now levels_up and those stale figures are deleted."
        ),
        "interpretation": (
            "Read this BEFORE the headline AUROC. Where coord sits at ~0.5 here while the "
            "pooled figure is high, the pooled figure is carried by the trivial component and "
            "the head has NOT been shown to predict progress. Positive counts on this slice are "
            "tiny (single digits on the smoke corpus), so a number ABOVE 0.5 here is equally "
            "unpowered -- neither direction is a finding at this n, and the uninformative random "
            "arm is reported alongside precisely so its wander is visible."
        ),
    }

    artifact["measured_incumbent_collapse"] = [
        entry
        for diagnostics in harvest_diagnostics
        for entry in diagnostics.get("incumbent_collapse", [])
    ]

    artifact["missing_verifier_gaps"] = [
        {
            "gap": "click-target progress discrimination at true distance >= 2",
            "failure_mode": (
                "Measured on tn36 states 13-14 (1-2 clicks before a level-up): ZERO of 48 "
                "generated candidates can reach the level-up within 2 further clicks, while "
                "the banked route does it in 1-2. No router change can order a candidate that "
                "was never generated."
            ),
            "missing_discriminator": (
                "a goal-conditioned candidate GENERATOR, not a ranker. Reported here as a "
                "measured exclusion only -- the generation-exploration axis is retired "
                "(exclusion_manifest id "
                "generation_axis_exploration_signal_retired_exp5154_v473)."
            ),
            "priority": "reported_not_pursued_here",
        },
        {
            "gap": "positive supply for the level-up label",
            "failure_mode": (
                "Across 12 audited games only 14 states at true distance 1 exist with 27 "
                "positives total -- below CLAUDE.md's N>=30 floor for a percentage-point "
                "delta claim. Only 5 of 12 games contribute at all (6 level up on KEYBOARD "
                "actions). MEASURED CONSEQUENCE on the smoke corpus: the hard slice (levels_up "
                "among changed) has 5 positives, where coord scores 0.5067 -- chance -- against "
                "static 0.6600. So the level-up discrimination this router would need is "
                "currently UNMEASURED rather than demonstrated, and the pooled headline comes "
                "from the frame-change component."
            ),
            "missing_discriminator": (
                "self-play state generation (harvest every visited state from a solver run) "
                "rather than banked-route states, which is also better distribution-matched "
                "to what the live agent sees."
            ),
            "priority": "high -- blocks a powered level-up-labelled claim",
        },
    ]

    corpus_digest = hashlib.sha256(
        json.dumps(
            [
                [r["game"], r["state_index"], r["x"], r["y"], r["label"], r["salience_rank"]]
                for r in all_rows
            ],
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    artifact["reproducibility_checksum"] = f"sha256:{corpus_digest}"
    artifact["duration_s"] = round(time.time() - started, 4)

    if not per_game:
        artifact["honest_verdict"] = "complete_no_informative_states_harvested_corpus_degenerate"
    elif not label_valid:
        artifact["honest_verdict"] = "complete_label_validity_check_invalid_arms_uninterpretable"
    elif informativeness_passed:
        artifact["honest_verdict"] = (
            "complete_coordinate_aware_click_features_beat_static_baseline_live_ab_still_required"
        )
    elif artifact["coordinate_blindness_repair_check"]["passed"]:
        # The honest reading when the repair check passes but the gate of record does not: the
        # DEFECT is repaired (measurably), and the head is NOT shown to add anything over the
        # ordering already live. Both halves belong in the verdict; reporting only the first
        # (as "pass stage1 gate") is what a downstream capstone would misread as a win.
        artifact["honest_verdict"] = (
            "complete_coordinate_blindness_repaired_but_gain_over_static_baseline_unestablished"
        )
    else:
        artifact["honest_verdict"] = "complete_coordinate_aware_click_features_below_stage1_gate"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print("\n== EXP 5904 STAGE-1 SUMMARY ==")
    print(f"  rows={len(all_rows)} states={artifact['n_states_total']} games={list(per_game)}")
    if len(per_game) <= 1:
        print("  NOTE: one contributing game -- every 'pooled_*' value below is a SINGLE game's")
    print("  -- WITHIN-STATE AUROC (metric of record) --")
    print(f"  blind      = {blind_mean}")
    print(f"  coord      = {coord_mean}")
    print(f"  random     = {random_mean}")
    print(f"  step_index = {step_mean}")
    print(f"  static     = {artifact['within_state_static_salience_auroc']['mean']}")
    print("  -- across-state reference (NOT the metric of record) --")
    print(f"  blind      = {artifact['pooled_blind_auroc']['mean']}")
    print(f"  coord      = {artifact['pooled_coord_auroc']['mean']}")
    print(f"  label_validity    = {artifact['label_validity_check']['status']}")
    print("  -- HEADLINE: coord vs the ordering already live --")
    print(f"  coord - static    = {artifact['honest_comparator']['delta']}")
    for game, boot in artifact["informativeness_gate"]["per_game_bootstrap"].items():
        print(
            f"    [{game}] bootstrap ci95={boot.get('delta_ci95')} "
            f"P(delta<=0)={boot.get('fraction_replicates_le_zero')} "
            f"excludes_zero={boot.get('excludes_zero')}"
        )
    for game, noise in artifact["honest_comparator"]["random_arm_seed_noise_per_game"].items():
        print(f"    [{game}] uninformative-arm seed-noise sd={noise.get('sd')}")
    print("  -- HARD SLICE (levels_up among clicks that changed something) --")
    for game, arms in per_game.items():
        print(
            f"    [{game}] n={arms.get('n_hard')} pos={arms.get('n_hard_pos')} "
            f"coord={arms.get('coord_hard_only')} static={arms.get('static_salience_hard_only')} "
            f"random={arms.get('random_hard_only')}"
        )
    print(
        f"  repair check      = {artifact['coordinate_blindness_repair_check']['passed']} "
        f"(also passed by {artifact['coordinate_blindness_repair_check']['also_passed_by_arms']})"
    )
    print(f"  GATE OF RECORD    = {artifact['informativeness_gate']['passed']}  (informativeness)")
    print(f"  verdict           = {artifact['honest_verdict']}")
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
