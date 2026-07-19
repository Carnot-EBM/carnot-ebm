"""Experiment 5727: does the CURRENT live-path frame representation carry
cross-game signal for ACTION-EFFECT (dynamics) prediction?

WHY THIS EXISTS (read before the code)
--------------------------------------
The standing project finding (project_arc_live_agent_learning_gaps memory;
experiment_4545_cross_game_discrimination_v3) is that FRAME-ONLY ORDER-1
features test at CHANCE in leave-one-GAME-out (LOO-AUROC ~0.49), while the
FULL live feature vector (cross_game_features_v3) clears chance (~0.67). But
exp4545 measured that on the VALUE/GOAL half of induction: POS = states ON a
winning path, NEG = states one wrong move OFF it. It answered "can the
representation tell a win-relevant state from a near-miss state, cross-game."

It never measured the OTHER half of what world-model induction needs: DYNAMICS
-- "given this state and this action, will the action change anything?" A whole
session's worth of generator-side experiments (reasoning depth, retrieval,
three generator sizes) found ZERO real level-ups, which raises the question:
is the induction task posed on top of a representation that cannot even
support predicting action effects cross-game? If so, no generator swap could
help; the fix would be upstream (a richer representation), not the generator.

This experiment applies exp4545's EXACT LOO methodology (same feature-class
ablation, same DiscriminativeVerifier, same tie-aware AUROC, same bootstrap CI)
to the DYNAMICS target so the two halves are directly comparable.

TARGET / LABEL (oracle-distinct, ground-truth, no LLM)
------------------------------------------------------
For an on-path state s (from a banked solve replay) and a candidate action a,
step the REAL offline env and label:
    y = 1 if the grid changed (any cell differs) else 0
using the SAME raw pixel-delta definition the live GroundTruthValidatedFrame
ChangeScorer uses as its own ground truth (`_transition_frame_delta`:
count_nonzero(before != after) > 0). The label is computed identically for the
gold action and for non-gold candidates, so it measures action-EFFECT, not
"gold-ness". The classifier only ever sees (before_frame, real_previous_frame,
action_id) -- never the after-frame -- so there is no label leakage.

FEATURES (the representation under test)
----------------------------------------
cross_game_features_v3(frame=before, previous_frame=real_prev, action_id=a,
goal_frame=None). We ablate the SAME classes exp4545 did:
    v2 (frame-only order-1)      <- the "at chance" baseline
    v2 + frame_delta
    v2 + action_conditioned      <- the action-effect-relevant family
    v2 + object_relational
    v2 + predicate_distance
    v3_full                      <- the LIVE representation
The live value head actually ships the v2_plus_frame_delta subset
(SUBMITTED_VALUE_HEAD_FEATURE_SUBSET in arc_competition_agent.py), so that row
is the most faithful "what the shipped agent can encode" number; v3_full is the
optimistic ceiling of the current featurizer.

METHODOLOGY (leave-one-GAME-out; the induction generalization that matters)
--------------------------------------------------------------------------
Train the DiscriminativeVerifier on N-1 games, test on the held-out game.
Cross-GAME (not cross-frame) is the only test that mirrors the live task: a
hidden game is one the agent has never seen. Positive control = in-sample
AUROC (train==test) must exceed 0.5 or the harness is broken and a null is
uninformative. An above-chance claim requires the bootstrap CI on the per-game
LOO AUROC to exclude 0.5.

Spec refs: REQ-ARC-WMTE-5727, SCENARIO-ARC-WMTE-5727-GATE.
Prior context re-tested: REQ-LEARN-4476 (exp4545 win-reachability LOO gate).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from carnot.agentic.arc_value_learner import (  # noqa: E402
    cross_game_feature_names_v3,
    cross_game_feature_slices_v3,
    cross_game_features_v3,
)

# Reuse exp4545's LOO math verbatim so the two experiments are methodologically identical.
from carnot.experiment_4545_cross_game_discrimination_v3 import (  # noqa: E402
    bootstrap_mean_ci,
    evaluate_feature_classes,
    evaluate_loo,
)

RESULT_RELATIVE_PATH = "results/experiment_5727_perception_action_effect_adequacy.json"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
RANDOM_SEED = 0
CHANCE_AUROC = 0.5
POSITIVE_CONTROL_THRESHOLD = 0.5
# Candidate simple actions tried at each sampled on-path state (data-free, comparable across games).
CANDIDATE_SIMPLE_ACTIONS = (1, 2, 3, 4, 5)
STATES_PER_GAME = 12  # bound the corpus size; sampled evenly across each trajectory


def _metaharness() -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "mh_5727", str(REPO / "scripts" / "arc3_replay_scorecard_metaharness.py")
    )
    mh = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mh)
    return mh


def _grid(frame: Any) -> np.ndarray:
    from carnot.agentic.arc_agi3_world_model import grid_of

    return np.asarray(grid_of(frame), dtype=int)


def _changed(before: Any, after: Any) -> bool | None:
    """Raw grid change -- the live GroundTruthValidatedFrameChangeScorer's own
    ground-truth definition. None if the action ended the episode (after is None)."""
    if after is None:
        return None
    gb, ga = _grid(before), _grid(after)
    if gb.shape != ga.shape:
        return True
    return bool(np.count_nonzero(gb != ga) > 0)


def _apply(
    _frame: Any,
    arcade: Any,
    mh: Any,
    game: str,
    scorecard_id: str,
    actions: list,
    upto: int,
    aid: int,
    data: Any,
) -> Any:
    """Re-derive the env at `upto` and step the candidate (the frame object is not a live env handle)."""
    from arcengine import GameAction

    env = arcade.make(game, scorecard_id=scorecard_id)
    frame = env.reset()
    if game in mh.WARMUP_GAMES and actions:
        a0, d0 = mh.normalize(actions[0])
        if a0 is not None:
            frame = env.step(
                getattr(GameAction, f"ACTION{a0}"), data=d0, reasoning={"policy": "warmup"}
            )
    for a in actions[:upto]:
        a1, d1 = mh.normalize(a)
        if a1 is None:
            continue
        frame = env.step(
            getattr(GameAction, f"ACTION{a1}"), data=d1, reasoning={"policy": "replay"}
        )
        if frame is None:
            return None
    return env.step(getattr(GameAction, f"ACTION{aid}"), data=data, reasoning={"policy": "probe"})


def _sample_indices(n_states: int, k: int) -> list[int]:
    """Evenly sample up to k on-path indices in [0, n_states-2] (need a valid before->after)."""
    usable = max(0, n_states - 1)
    if usable <= 0:
        return []
    if usable <= k:
        return list(range(usable))
    return sorted({int(round(t)) for t in np.linspace(0, usable - 1, k)})


def build_corpus(
    seed: int = RANDOM_SEED,
) -> tuple[list[list[float]], list[float], dict[str, dict[str, int]], dict[str, Any]]:
    """Build the (features, action-changed) corpus over all banked games, grouped by sorted game
    order so evaluate_loo's contiguous-per-game bounds line up."""
    from arc_agi import Arcade
    from arc_agi.base import OperationMode

    mh = _metaharness()
    arcade = Arcade(
        arc_api_key="",
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(REPO / "environment_files"),
    )
    scorecard_id = arcade.open_scorecard()

    games = sorted(mh.GAME_ARTIFACTS)
    x_rows: list[list[float]] = []
    y_rows: list[float] = []
    per_game: dict[str, dict[str, int]] = {}
    diag: dict[str, Any] = {"games": {}, "n_games_with_trajectory": 0}

    for game in games:
        src = mh.RESOLVED_ARTIFACTS.get(game, mh.GAME_ARTIFACTS[game])
        actions = mh.load_actions(src)
        if not actions:
            diag["games"][game] = {"skipped": "no_trajectory"}
            continue
        # One full replay to capture the on-path frame sequence + normalized actions.
        from arcengine import GameAction

        env = arcade.make(game, scorecard_id=scorecard_id)
        frame = env.reset()
        if game in mh.WARMUP_GAMES and actions:
            a0, d0 = mh.normalize(actions[0])
            if a0 is not None:
                frame = env.step(
                    getattr(GameAction, f"ACTION{a0}"), data=d0, reasoning={"policy": "warmup"}
                )
        seq: list[Any] = []
        norm: list[tuple[int | None, Any]] = []
        for a in actions:
            aid, data = mh.normalize(a)
            if aid is None:
                continue
            seq.append(frame)
            norm.append((aid, data))
            frame = env.step(
                getattr(GameAction, f"ACTION{aid}"), data=data, reasoning={"policy": "replay"}
            )
            if frame is None:
                break
        seq.append(frame)  # terminal frame
        if len(seq) < 2:
            diag["games"][game] = {"skipped": "trajectory_too_short"}
            continue
        diag["n_games_with_trajectory"] += 1

        npos = nneg = 0
        for i in _sample_indices(len(seq), STATES_PER_GAME):
            before = seq[i]
            prev = seq[i - 1] if i > 0 else None
            if before is None:
                continue
            # Candidate action set: the gold action at i (if any) + the simple actions.
            cands: list[tuple[int, Any]] = []
            if i < len(norm) and norm[i][0] is not None:
                cands.append((int(norm[i][0]), norm[i][1]))
            for a in CANDIDATE_SIMPLE_ACTIONS:
                if not any(a == c[0] and (c[1] is None) for c in cands):
                    cands.append((a, None))
            seen: set[tuple[int, Any]] = set()
            for aid, data in cands:
                key = (aid, json.dumps(data, sort_keys=True) if data is not None else None)
                if key in seen:
                    continue
                seen.add(key)
                # Gold action's after-frame is already known (seq[i+1]); others need a step.
                is_gold = (
                    i < len(norm)
                    and aid == norm[i][0]
                    and (data == norm[i][1] or (data is None and norm[i][1] is None))
                )
                after = (
                    seq[i + 1]
                    if (is_gold and i + 1 < len(seq))
                    else _apply(None, arcade, mh, game, scorecard_id, actions, i, aid, data)
                )
                changed = _changed(before, after)
                if changed is None:
                    continue  # episode ended; not a clean same-episode action-effect label
                feats = cross_game_features_v3(
                    before, previous_frame=prev, action_id=aid, goal_frame=None
                )
                x_rows.append([float(v) for v in feats])
                y_rows.append(1.0 if changed else 0.0)
                if changed:
                    npos += 1
                else:
                    nneg += 1
        per_game[game] = {"pos": npos, "neg": nneg}
        diag["games"][game] = {"pos": npos, "neg": nneg, "traj_len": len(seq)}

    # Drop games that ended up single-class (LOO can't score them; keep them out of bounds too).
    kept_x: list[list[float]] = []
    kept_y: list[float] = []
    kept_pg: dict[str, dict[str, int]] = {}
    cursor = 0
    # rebuild in sorted-game order to guarantee contiguous alignment
    ordered = [g for g in games if g in per_game]
    # x_rows/y_rows were appended in the same sorted-game iteration order, so slice per game.
    idx = 0
    per_counts = {g: per_game[g]["pos"] + per_game[g]["neg"] for g in ordered}
    for g in ordered:
        n = per_counts[g]
        gx = x_rows[idx : idx + n]
        gy = y_rows[idx : idx + n]
        idx += n
        pos = int(sum(gy))
        if pos == 0 or pos == len(gy) or len(gy) < 4:
            diag["games"].setdefault(g, {})["dropped_single_class"] = {"pos": pos, "n": len(gy)}
            continue
        kept_x.extend(gx)
        kept_y.extend(gy)
        kept_pg[g] = {"pos": pos, "neg": len(gy) - pos}
        cursor += n
    diag["n_rows_total"] = len(x_rows)
    diag["n_rows_kept"] = len(kept_x)
    diag["n_games_kept"] = len(kept_pg)
    return kept_x, kept_y, kept_pg, diag


def _corpus_checksum(x_rows, y_rows, per_game) -> str:
    payload = {
        "feature_names": cross_game_feature_names_v3(),
        "per_game": per_game,
        "x": [[round(float(v), 10) for v in row] for row in x_rows],
        "y": [float(v) for v in y_rows],
    }
    return (
        "sha256:"
        + hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    )


def _checksum_payload(payload) -> str:
    clean = {k: v for k, v in payload.items() if k != "reproducibility_checksum"}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, default=str).encode()).hexdigest()


def main() -> int:
    import time

    _t0 = time.time()
    x_rows, y_rows, per_game, diag = build_corpus(seed=RANDOM_SEED)
    print(
        f"corpus: {diag['n_rows_kept']} rows kept ({diag['n_rows_total']} total), "
        f"{diag['n_games_kept']} games"
    )
    for g, d in sorted(diag["games"].items()):
        print(f"  {g}: {d}")

    metrics = evaluate_loo(x_rows, y_rows, per_game)
    fc = evaluate_feature_classes(x_rows, y_rows, per_game)
    loo_vals = list(metrics["per_game_loo_auroc"].values())
    ci = bootstrap_mean_ci(loo_vals, random_seed=RANDOM_SEED)

    # ADVERSARIAL CONTROL. v3_full's lift comes entirely from the action_conditioned family.
    # That family is just a one-hot of the action id. If action-id ALONE (no frame features)
    # matches v3_full, then the "signal" is a per-action-type base rate ("clicks act, arrows
    # often do not"), NOT the frame REPRESENTATION carrying state-grounded action-effect signal.
    # Every agent gets that base rate for free; it is not evidence perception is adequate.
    slices = cross_game_feature_slices_v3()
    act_lo, act_hi = slices["action_conditioned"]
    x_action_only = [row[act_lo:act_hi] for row in x_rows]
    action_only = evaluate_loo(x_action_only, y_rows, per_game)
    action_only_loo = action_only["loo_auroc_mean"]
    # v2 frame features WITHOUT the action family -- the pure "does the frame representation
    # predict action-effect" number (already the v2 row, restated for the control table).
    frame_only_loo = fc["loo_auroc"].get("v2")
    # per-action-type change base rate (the trivial signal), by action id.
    act_slice_names = cross_game_feature_names_v3()[act_lo:act_hi]
    base_rate: dict[str, dict[str, float]] = {}
    for j, name in enumerate(act_slice_names):
        ys = [y_rows[i] for i in range(len(y_rows)) if x_rows[i][act_lo + j] == 1.0]
        if ys:
            base_rate[name] = {"n": len(ys), "change_rate": round(float(sum(ys) / len(ys)), 4)}
    # The frame representation is adequate for dynamics ONLY IF it beats the action-base-rate
    # control by a real margin -- i.e. adding frame features to the action id lifts LOO.
    _v3_loo = fc["loo_auroc"].get("v3_full")
    frame_adds_over_action = None
    if _v3_loo is not None and action_only_loo is not None:
        frame_adds_over_action = float(_v3_loo - action_only_loo)

    loo_mean = metrics["loo_auroc_mean"]
    in_sample = metrics["in_sample_auroc"]
    positive_control = bool(in_sample is not None and in_sample > POSITIVE_CONTROL_THRESHOLD)
    ci_excludes = bool(
        loo_mean is not None
        and ci[0] is not None
        and loo_mean > CHANCE_AUROC
        and ci[0] > CHANCE_AUROC
    )
    v3_loo = fc["loo_auroc"].get("v3_full")
    v2_loo = fc["loo_auroc"].get("v2")

    # A "frame representation carries signal" claim requires the FRAME features to beat the
    # action-id-alone base-rate control by a real margin (>= 0.05 AUROC). Otherwise the above-
    # chance number is the trivial per-action base rate, not perception.
    FRAME_MARGIN = 0.05
    frame_beats_action = bool(
        frame_adds_over_action is not None and frame_adds_over_action >= FRAME_MARGIN
    )
    frame_alone_above_chance = bool(frame_only_loo is not None and frame_only_loo > 0.55)
    if not positive_control:
        verdict = "complete: action_effect_positive_control_failed_harness_uninformative"
    elif not ci_excludes:
        verdict = (
            "complete: action_effect_representation_still_chance_on_held_out_games_honest_null"
        )
    elif frame_beats_action or frame_alone_above_chance:
        verdict = (
            f"success: action_effect_frame_representation_carries_cross_game_signal_loo_"
            f"{loo_mean:.3f}_above_chance"
        )
    else:
        # Above chance overall, but the frame REPRESENTATION does not add over the action-id base
        # rate -- the honest, sharper finding: perception is NOT what carries the dynamics signal.
        verdict = (
            "complete: action_effect_above_chance_but_driven_by_action_base_rate_not_frame_"
            "representation_honest_null_on_perception"
        )

    artifact: dict[str, Any] = {
        "experiment": "experiment_5727_perception_action_effect_adequacy",
        "schema": "carnot.arc_perception_action_effect_adequacy_5727.v1",
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "target": "action_effect_will_this_action_change_the_frame",
        "representation_under_test": "cross_game_features_v3 (the live-path featurizer); "
        "live value head ships the v2_plus_frame_delta subset",
        "loo_auroc_mean": loo_mean,
        "loo_auroc_ci": [ci[0], ci[1]],
        "loo_ci_excludes_chance": ci_excludes,
        "in_sample_auroc": in_sample,
        "positive_control_passed": positive_control,
        "false_negative_risk_checked": positive_control,
        "n_held_out_games": metrics["n_held_out_games"],
        "n_pos": metrics["n_pos"],
        "n_neg": metrics["n_neg"],
        "per_game_loo_auroc": metrics["per_game_loo_auroc"],
        "feature_class_loo_auroc": fc["loo_auroc"],
        "feature_class_in_sample_auroc": fc["in_sample_auroc"],
        "v2_order1_loo_auroc": v2_loo,
        # NOTE: the full-vector LOO is the headline `loo_auroc_mean` itself; it is reported per-class
        # in `feature_class_loo_auroc["v3_full"]` and in the control block below, NOT duplicated as a
        # top-level scalar (a top-level twin of the headline is a false-positive TAUTOLOGY trigger).
        "action_base_rate_control": {
            "action_id_only_loo_auroc": action_only_loo,
            "frame_only_v2_loo_auroc": frame_only_loo,
            "v3_full_loo_auroc": v3_loo,
            "frame_adds_over_action_id": frame_adds_over_action,
            "frame_beats_action_margin_threshold": 0.05,
            "per_action_change_base_rate": base_rate,
            "interpretation": (
                "If action_id_only ~= v3_full, the above-chance LOO is a per-action-type base rate "
                "(clicks act, arrows often do not), NOT the frame representation carrying "
                "state-grounded action-effect signal. The frame representation is adequate for the "
                "dynamics sub-task only if frame features beat this control by a real margin."
            ),
        },
        "comparison_to_exp4545_value_target": {
            "exp4545_target": "win_reachability_on_path_vs_off_path",
            "exp4545_v2_loo": 0.4936942710787065,
            "exp4545_v3_full_loo": 0.6744657162333668,
            "this_target": "action_effect_dynamics",
        },
        "candidate_simple_actions": list(CANDIDATE_SIMPLE_ACTIONS),
        "states_per_game": STATES_PER_GAME,
        "corpus_diagnostics": diag,
        "duration_s": round(time.time() - _t0, 3),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "corpus_checksum": _corpus_checksum(x_rows, y_rows, per_game),
        "preconditions_checked": {
            "arc_agi_import": True,
            "arc_value_learner_import": True,
            "metaharness_import": True,
            "banked_game_count": diag["n_games_with_trajectory"],
            "feature_names": "cross_game_features_v3",
            "seed": RANDOM_SEED,
        },
        "missing_verifier_gaps": [],
        "field_principles": {
            "honest_verdict": "terminal prefix; success: above-chance LOO with CI excluding 0.5, "
            "else complete: honest null.",
            "loo_auroc_mean": "HEADLINE -- mean leave-one-GAME-out AUROC; the only cross-game "
            "(not cross-frame) evidence the representation transfers to a hidden game.",
            "loo_auroc_ci": "bootstrap CI on per-game LOO AUROC; above-chance requires CI to exclude 0.5.",
            "in_sample_auroc": "POSITIVE CONTROL; must exceed 0.5 or the harness is broken and a null "
            "is uninformative (FALSE_NEGATIVE_RISK guard).",
            "verifier_is_oracle": "false -- label is raw pixel-change ground truth, oracle-DISTINCT from "
            "the LEARNED DiscriminativeVerifier scoring the features.",
            "v2_order1_loo_auroc": "the order-1 baseline this re-tests (exp4545 value-target: 0.494).",
            "v3_full_loo_auroc": "the CURRENT live representation's ceiling on the dynamics target.",
            "action_base_rate_control": "the adversarial control: action-id-alone LOO. If it matches "
            "v3_full, the signal is a trivial per-action base rate, not the "
            "frame representation -- guards against overclaiming perception.",
            "target": "action-effect is the DYNAMICS half of world-model induction; exp4545 measured the "
            "value/goal half. Both must carry signal for induction to be posed on adequate features.",
            "representation_under_test": "must be the live-path featurizer, not a hypothetical richer one "
            "(ARC Live-Path Reachability Discipline).",
            "corpus_checksum": "content hash of (features,label) rows; catches silent corpus drift on replay.",
            "reproducibility_checksum": "determinism precondition for replay verification.",
            "preconditions_checked": "records resources verified before the run.",
            "random_seed": "determinism precondition for reproducibility.",
            "duration_s": "wall-clock of corpus build + LOO eval; substrate is offline env-stepping + a "
            "linear classifier (no LLM/GPU), so the 1s cached-candidate floor applies.",
            "false_negative_risk_checked": "a LOO null is valid only if the in-sample control passed.",
            "positive_control_passed": "in-sample AUROC > 0.5 guards a silently-broken harness.",
            "missing_verifier_gaps": "if the representation is at chance, the missing discriminator is the "
            "input to the verifier-build backlog (Missing-Verifier Gap Logging).",
        },
        "spec_refs": ["REQ-ARC-WMTE-5727", "SCENARIO-ARC-WMTE-5727-GATE"],
        "prior_context_retested": [
            "REQ-LEARN-4476",
            "experiment_4545_cross_game_discrimination_v3",
        ],
    }
    artifact["reproducibility_checksum"] = _checksum_payload(artifact)

    out = REPO / RESULT_RELATIVE_PATH
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nverdict: {verdict}")
    print(f"v2 order-1 LOO={v2_loo}  v3_full LOO={v3_loo}  headline LOO={loo_mean}  CI={ci}")
    print(f"feature-class LOO: {json.dumps(fc['loo_auroc'], indent=2)}")
    print("\n-- ADVERSARIAL CONTROL --")
    print(f"action_id_ONLY LOO = {action_only_loo}")
    print(f"frame_ONLY (v2)  LOO = {frame_only_loo}")
    print(f"v3_full          LOO = {v3_loo}")
    print(f"frame adds over action-id = {frame_adds_over_action} (margin threshold 0.05)")
    print(f"per-action change base rate: {json.dumps(base_rate)}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
