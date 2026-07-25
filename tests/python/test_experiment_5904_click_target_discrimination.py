"""Tests for exp5904's ARMS, GATE and UNCERTAINTY reporting -- not for its harvest.

Spec refs: REQ-ARC-FCP-5904, SCENARIO-ARC-FCP-5904-HARD-SLICE-IS-OPERATIVE,
SCENARIO-ARC-FCP-5904-INFORMATIVENESS-GATE.

WHY THESE TESTS EXIST. Every one of them pins a defect that shipped in the first version of
this experiment and was found by review, not by the suite:

* The HARD-SLICE arm was STRUCTURALLY DEAD. It filtered rows on ``changed`` while reusing the
  pooled label ``(changed OR levels_up)``, so every row in the slice was a positive, ``auroc``
  returned ``None`` for any score vector, and the artifact shipped ``coord_hard_only: null`` /
  ``n_hard == n_pos``. The arm existed precisely to test whether the signal is more than
  "distinguish a no-op from any effect" -- the live triviality risk -- so its silence was the
  worst possible failure. ``test_the_old_hard_slice_label_was_undefined_by_construction``
  encodes the defect; the two tests after it pin the working arm.
* The pre-registered gate was cleared by the experiment's OWN uninformative control. Within a
  state the blind arm is a constant, hence exactly 0.5, so the "conjunction" reduced to a
  single threshold that the shipped ``RandomCandidateRouter`` (0.6308) and the incumbent
  ``static_salience`` sort (0.9360) both cleared -- while the artifact's verdict read as a pass.
* The one decision-relevant number (coord minus static) shipped as a bare point estimate with
  no interval, on a 2-state corpus. ``exp3540`` is the precedent: paired p=0.135, retro verdict
  "advantage was small-sample artifact".

The harvest path (forking the offline arcade) is deliberately NOT tested here: it needs the
real env and is exercised by running the experiment's own ``--smoke`` mode.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "python") not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(REPO / "python"))

from carnot.agentic.arc_click_target_features import CLICK_TARGET_FEATURE_DIM  # noqa: E402


def _load_experiment() -> Any:
    """Import the experiment module by path (it is a script, not an importable package member)."""

    path = REPO / "python" / "carnot" / "experiment_5904_click_target_discrimination.py"
    spec = importlib.util.spec_from_file_location("exp5904_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXP = _load_experiment()


# --------------------------------------------------------------------------- synthetic corpus


def _features(signal: float, index: int) -> list[float]:
    """A 21-wide vector whose FIRST entry carries the signal and the rest vary mildly.

    The head standardizes with a variance floor, so a column of pure constants contributes
    nothing; giving the other columns small deterministic variation keeps the fit realistic
    without making any of them predictive.
    """

    vector = [0.0] * CLICK_TARGET_FEATURE_DIM
    vector[0] = float(signal)
    vector[1] = float((index % 5) / 5.0)
    vector[2] = float((index % 3) / 3.0)
    vector[19] = float((index % 7) / 7.0)
    vector[20] = float((index % 11) / 11.0)
    return vector


def _corpus(
    *,
    n_states: int = 4,
    per_state: int = 10,
    n_level_ups_per_state: int = 2,
    level_up_without_change: bool = False,
) -> list[dict[str, Any]]:
    """A labelled corpus shaped exactly like ``harvest_game``'s output.

    Within each state the first ``n_level_ups_per_state`` rows level up, the next two change the
    frame without levelling up, and the remainder are inert -- so the pooled label has both
    classes AND the hard slice (levels_up among changed) has both classes too.
    """

    rows: list[dict[str, Any]] = []
    for state in range(n_states):
        for position in range(per_state):
            levels_up = position < n_level_ups_per_state
            changed = position < n_level_ups_per_state + 2
            if levels_up and level_up_without_change:
                changed = False
            label = 1.0 if (changed or levels_up) else 0.0
            rows.append(
                {
                    "game": "gm00",
                    "state_index": state,
                    "at_boundary": state % 2 == 0,
                    "x": position,
                    "y": state,
                    "salience_rank": position,
                    "changed": changed,
                    "levels_up": levels_up,
                    "outcome_class": position,
                    "label": label,
                    "features": _features(1.0 if label >= 0.5 else 0.0, position + state),
                    "incumbent_score": 0.4 + 0.01 * state,  # per-state CONSTANT, as measured
                }
            )
    return rows


# ------------------------------------------------------------------------------ hard slice


def test_the_old_hard_slice_label_was_undefined_by_construction() -> None:
    """The shipped arm could NEVER produce a number. Encoded so it cannot silently return.

    ``hard = [r for r in rows if r['changed']]`` combined with ``label = changed or levels_up``
    means every hard row is a positive, so ``auroc`` sees one class and returns None whatever
    the scores are.
    """

    rows = _corpus()
    hard = [row for row in rows if bool(row["changed"])]
    assert hard, "fixture must contain frame-changing clicks"
    old_labels = [float(row["label"]) for row in hard]
    assert set(old_labels) == {1.0}
    assert EXP.auroc([float(i) for i in range(len(hard))], old_labels) is None
    assert EXP.auroc([1.0] * len(hard), old_labels) is None


def test_hard_slice_uses_levels_up_and_reports_both_classes() -> None:
    """SCENARIO-ARC-FCP-5904-HARD-SLICE-IS-OPERATIVE: the arm must actually run."""

    arms = EXP.run_arms_for_game(_corpus(), seed=5904)
    assert arms["hard_slice_label"].startswith("levels_up")
    assert arms["n_hard"] > 0
    assert arms["n_hard_pos"] > 0, "the hard slice must contain level-ups"
    assert arms["n_hard_neg"] > 0, "the hard slice must contain changed-but-no-level-up rows"
    assert arms["n_hard"] == arms["n_hard_pos"] + arms["n_hard_neg"]
    # The dead arm's signature was not ``n_hard == n_pos`` (that is EXPECTED whenever the label
    # is identical to frame-change) but a hard slice with NO negatives in it. Pin that instead.
    assert arms["n_hard_pos"] < arms["n_hard"], "the slice must not be all-positive again"
    for key in ("coord_hard_only", "static_salience_hard_only", "random_hard_only"):
        assert isinstance(arms[key], float), f"{key} must be a number, not None"
        assert 0.0 <= arms[key] <= 1.0


def test_hard_slice_separates_a_signal_that_only_the_pooled_label_would_show() -> None:
    """A head that only knows "did anything happen" must NOT look good on the hard slice.

    This is the triviality risk in one assertion: score by ``changed`` (a perfect pooled-label
    predictor) and the hard slice sees a constant, i.e. exactly 0.5.
    """

    rows = _corpus()
    hard = [row for row in rows if bool(row["changed"])]
    hard_labels = [float(bool(row["levels_up"])) for row in hard]
    trivial_scores = [1.0 for _row in hard]  # "it changed something" -- true for every hard row
    assert EXP.auroc(trivial_scores, hard_labels) == pytest.approx(0.5)
    oracle_scores = [float(bool(row["levels_up"])) for row in hard]
    assert EXP.auroc(oracle_scores, hard_labels) == pytest.approx(1.0)


# ------------------------------------------------------------------------- label composition


def test_label_identity_to_frame_change_is_measured_not_assumed() -> None:
    """The ``OR levels_up`` disjunct does no work when level-ups always change the frame.

    Measured on the real smoke corpus: 182/182 rows have ``label == changed`` and
    ``n_levels_up_without_change == 0``, so reporting ``n_levels_up`` next to ``n_positive``
    would read as if level-ups supplied positives of their own. The flag must flip when a
    corpus DOES contain a level-up with no frame change.
    """

    identical = EXP.run_arms_for_game(_corpus(), seed=5904)
    assert identical["n_levels_up_without_change"] == 0
    assert identical["n_label_equals_changed"] == identical["n_score"]
    assert identical["label_is_measured_identical_to_frame_change"] is True

    distinct = EXP.run_arms_for_game(_corpus(level_up_without_change=True), seed=5904)
    assert distinct["n_levels_up_without_change"] > 0
    assert distinct["label_is_measured_identical_to_frame_change"] is False


# -------------------------------------------------------------------------------- the gate


def test_repair_check_is_cleared_by_the_uninformative_random_arm() -> None:
    """The originally pre-registered expression cannot be evidence of value.

    Values are this run's own measured within-state AUROCs: coord 0.9797, shipped random
    0.6308, incumbent static 0.9360, step_index 0.5, blind 0.5.
    """

    blind = 0.5
    assert EXP.repair_check_passes(0.9796511627906976, blind) is True  # coord
    assert EXP.repair_check_passes(0.6308139534883721, blind) is True  # UNINFORMATIVE control
    assert EXP.repair_check_passes(0.936046511627907, blind) is True  # the INCUMBENT ordering
    assert EXP.repair_check_passes(0.5, blind) is False  # step_index
    assert EXP.repair_check_passes(blind, blind) is False  # blind itself
    assert EXP.repair_check_passes(None, blind) is False
    assert EXP.repair_check_passes(0.99, None) is False


def test_repair_check_pass_region_contains_regressions_against_the_incumbent() -> None:
    """Why it is not the gate of record: 0.70 passes while ordering worse than the incumbent."""

    incumbent = 0.936046511627907
    regression = 0.70
    assert regression < incumbent
    assert EXP.repair_check_passes(regression, 0.5) is True


# ------------------------------------------------------------------------------- bootstrap


def _score_rows() -> list[dict[str, Any]]:
    rows = _corpus()
    _fit, score_rows = EXP._temporal_split(rows)
    assert score_rows
    return score_rows


def test_bootstrap_ci_includes_zero_when_the_two_arms_are_identical() -> None:
    rows = _score_rows()
    scores = [float(row["salience_rank"]) for row in rows]
    result = EXP.paired_within_state_delta_bootstrap(rows, scores, scores, n_boot=300, seed=1)
    assert result["delta_mean"] == pytest.approx(0.0)
    assert result["delta_ci95"] == [pytest.approx(0.0), pytest.approx(0.0)]
    assert result["excludes_zero"] is False
    assert result["fraction_replicates_le_zero"] == pytest.approx(1.0)


def test_bootstrap_ci_excludes_zero_for_a_genuinely_better_arm() -> None:
    rows = _score_rows()
    perfect = [float(row["label"]) for row in rows]
    inverted = [-float(row["label"]) for row in rows]
    result = EXP.paired_within_state_delta_bootstrap(rows, perfect, inverted, n_boot=300, seed=2)
    assert result["delta_mean"] > 0.9
    assert result["delta_ci95"][0] > 0.0
    assert result["excludes_zero"] is True
    assert result["fraction_replicates_le_zero"] == pytest.approx(0.0)


def test_bootstrap_reports_its_own_bookkeeping() -> None:
    rows = _score_rows()
    a = [float(row["label"]) for row in rows]
    b = [float(row["salience_rank"]) for row in rows]
    result = EXP.paired_within_state_delta_bootstrap(rows, a, b, n_boot=200, seed=3)
    assert result["n_bootstrap"] == 200
    assert result["n_replicates_used"] + result["n_replicates_dropped_degenerate"] == 200
    assert 0.0 <= result["fraction_replicates_le_zero"] <= 1.0
    assert result["delta_ci95"][0] <= result["delta_mean"] <= result["delta_ci95"][1]
    assert "WITHIN each scored state" in result["resampling"]


def test_bootstrap_on_an_empty_corpus_says_so_instead_of_raising() -> None:
    result = EXP.paired_within_state_delta_bootstrap([], [], [], n_boot=10)
    assert result["n_bootstrap"] == 0
    assert "no scored states" in result["note"]


def test_run_arms_attaches_uncertainty_and_a_noise_floor_to_the_headline_delta() -> None:
    """The decision-relevant delta must never ship as a bare point estimate again."""

    arms = EXP.run_arms_for_game(_corpus(), seed=5904)
    assert arms["coord_head_fitted"] is True
    assert isinstance(arms["coord_minus_static_within_state"], float)
    boot = arms["coord_minus_static_bootstrap"]
    assert boot["delta_ci95"][0] <= boot["delta_mean"] <= boot["delta_ci95"][1]
    assert isinstance(boot["fraction_replicates_le_zero"], float)
    noise = arms["random_arm_seed_noise"]
    assert noise["n_seeds"] == EXP.RANDOM_NOISE_SEEDS
    assert noise["sd"] is not None and noise["sd"] > 0.0
    assert noise["min"] <= noise["mean"] <= noise["max"]


# ----------------------------------------------------------------------------- pooling label


def test_pool_flags_a_single_game_value_as_not_pooled() -> None:
    """``pooled_*`` with one contributing game is one game's number, and must say so."""

    one = EXP._pool({"lp85": {"coord_within_state": 0.9}}, "coord_within_state")
    assert one["n_games"] == 1
    assert one["single_game_not_pooled"] is True
    assert one["ci95"] is None
    assert one["mean"] == pytest.approx(0.9)

    two = EXP._pool(
        {"lp85": {"coord_within_state": 0.9}, "tn36": {"coord_within_state": 0.7}},
        "coord_within_state",
    )
    assert two["n_games"] == 2
    assert two["single_game_not_pooled"] is False
    assert two["ci95"] is not None

    none = EXP._pool({"lp85": {"coord_within_state": None}}, "coord_within_state")
    assert none["n_games"] == 0
    assert none["no_contributing_games"] is True
    assert none["single_game_not_pooled"] is False


# -------------------------------------------------------------------- artifact-level honesty


def test_shipped_artifact_reports_the_headline_with_uncertainty_and_no_stale_gate_key() -> None:
    """Guards the artifact a downstream capstone would actually read.

    A capstone reading ``pre_registered_gate.passed: true`` would have banked a positive result
    that the numbers do not support, so that key must be GONE and replaced by the two explicitly
    named blocks.
    """

    import json

    path = REPO / "results" / "experiment_5904_click_target_discrimination.json"
    if not path.exists():  # pragma: no cover - artifact is committed alongside the experiment
        pytest.fail(f"missing artifact {path}; run the experiment's --smoke mode")
    artifact = json.loads(path.read_text(encoding="utf-8"))

    assert "pre_registered_gate" not in artifact
    assert artifact["gate_of_record"] == "informativeness_gate"

    repair = artifact["coordinate_blindness_repair_check"]
    assert repair["renamed_from"] == "pre_registered_gate"
    assert repair["is_not_an_informativeness_gate"] is True
    # The measured fact that motivated the rename must travel WITH the number.
    assert "random" in repair["also_passed_by_arms"]

    gate = artifact["informativeness_gate"]
    assert gate["added_post_hoc_after_review"] is True
    assert gate["passed"] == artifact["informativeness_established"]
    assert gate["passed"] is False, "the smoke corpus does not establish a gain over static"

    comparator = artifact["honest_comparator"]
    assert comparator["delta_has_uncertainty"] is True
    boot = comparator["single_game_bootstrap"]
    assert boot["delta_ci95"][0] < 0.0 < boot["delta_ci95"][1], "smoke delta must show a null CI"
    assert boot["fraction_replicates_le_zero"] > 0.0

    # The verdict must name BOTH halves: defect repaired, gain unestablished.
    verdict = artifact["honest_verdict"]
    assert verdict.startswith("complete_")
    assert "repaired" in verdict and "unestablished" in verdict

    # Label honesty: the disjunct is measured to be a no-op on this corpus.
    composition = artifact["label_composition"]
    assert composition["n_levels_up_without_change"] == 0
    assert composition["label_is_measured_identical_to_frame_change"] is True

    # The hard slice must carry real numbers, and the not-run baselines must be named.
    hard = artifact["hard_slice_summary"]["per_game"]
    assert hard, "hard slice must report per-game numbers"
    for game_values in hard.values():
        assert isinstance(game_values["coord"], float)
        assert game_values["n_hard_pos"] > 0 and game_values["n_hard_neg"] > 0
    baselines = artifact["baselines_not_run_and_why"]
    assert "FrameChangeScorer" in baselines["scorers"]
    assert baselines["frame_change_checkpoints_found_in_models_dir"] == []
