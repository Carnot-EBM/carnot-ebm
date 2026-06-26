"""Tests for Exp 4761 structural-energy S0 transition-correctness probe.

Spec refs: REQ-ARC-WMTE-4761, SCENARIO-ARC-WMTE-4761-S0-GATE,
SCENARIO-ARC-WMTE-4761-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_4761_structural_energy_s0_core_bet_probe as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"


def _spec_section() -> str:
    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-ARC-WMTE-4761")
    end = spec.index("### REQ-ARC-WMTE-4751", start)
    return spec[start:end]


def _grid(changed: bool = False) -> np.ndarray:
    g = np.zeros((4, 4), dtype=np.int16)
    g[1, 1] = 2
    if changed:
        g[1, 2] = 2
        g[1, 1] = 0
    return g


def _fixture_rows() -> mod.S0Dataset:
    rows: list[mod.TransitionCandidateRow] = []
    origin: list[mod.OriginProbeRow] = []
    for gi, game in enumerate(("g0", "g1", "g2")):
        for i in range(4):
            rows.append(
                mod.TransitionCandidateRow(
                    game=game,
                    label=1.0,
                    structural=[2.0 + gi, 1.0 + i, 0.0],
                    marginal=[0.0],
                    family_features={
                        "object_relational": [2.0 + gi, 1.0 + i],
                        "frame_delta": [0.0],
                    },
                    cell_change_fraction=0.0,
                    near_miss_negative=False,
                )
            )
            rows.append(
                mod.TransitionCandidateRow(
                    game=game,
                    label=0.0,
                    structural=[-2.0 - gi, -1.0 - i, 0.0],
                    marginal=[0.0],
                    family_features={
                        "object_relational": [-2.0 - gi, -1.0 - i],
                        "frame_delta": [0.0],
                    },
                    cell_change_fraction=0.03,
                    near_miss_negative=True,
                )
            )
            # Origin rows are deliberately feature-identical across origins so
            # SCENARIO-ARC-WMTE-4761-S0-GATE can verify the leak audit passes.
            origin.append(mod.OriginProbeRow(game=game, label=1.0, structural=[1.0, 0.0]))
            origin.append(mod.OriginProbeRow(game=game, label=0.0, structural=[1.0, 0.0]))
    return mod.S0Dataset(
        candidate_rows=rows,
        origin_rows=origin,
        per_game={g: {"candidate_rows": 8, "origin_rows": 8} for g in ("g0", "g1", "g2")},
    )


def test_req_arc_wmte_4761_spec_declares_s0_contract() -> None:
    """REQ-ARC-WMTE-4761: OpenSpec declares the S0 probe contract first."""

    section = _spec_section()

    assert "REQ-ARC-WMTE-4761" in section
    assert "SCENARIO-ARC-WMTE-4761-S0-GATE" in section
    assert "SCENARIO-ARC-WMTE-4761-BLOCKED-PRECONDITION" in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert "object_relational" in section
    assert "frame_delta" in section
    assert "action_conditioned" in section
    assert "predicate_distance" in section
    assert "v2 frame-marginal" in section
    assert "origin-probe" in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert field in section
        assert principle["principle"] in section


def test_scenario_arc_wmte_4761_features_exclude_marginals_and_dead_families() -> None:
    """SCENARIO-ARC-WMTE-4761-S0-GATE: structural rows use only allowed v3 slices."""

    previous = _grid(False)
    candidate = _grid(True)

    view = mod.structural_feature_view(previous, candidate, action_key=(6, 1, 1))

    assert set(view) == {"structural", "marginal", "family_features", "feature_names"}
    assert len(view["structural"]) == len(view["family_features"]["object_relational"]) + len(
        view["family_features"]["frame_delta"]
    )
    assert len(view["marginal"]) > 0
    assert view["structural"] != view["marginal"]
    assert all(name.startswith(("object_relational_", "frame_delta_")) for name in view["feature_names"])
    assert not any(name.startswith(("v2_", "action_conditioned_", "predicate_distance_")) for name in view["feature_names"])


def test_req_arc_wmte_4761_prediction_rows_do_not_corrupt_ground_truth() -> None:
    """REQ-ARC-WMTE-4761: negatives are induced predictions, not synthetic corruptions."""

    state = np.zeros((10, 10), dtype=np.int16)
    state[1, 1] = 2
    real_next = state.copy()
    real_next[1, 1] = 0
    real_next[1, 2] = 2
    model_prediction = state.copy()

    dataset = mod.dataset_from_heldout_predictions(
        "toy",
        [(state, (6, 1, 1), real_next)],
        predict_fn=lambda _s, _a: model_prediction,
    )

    assert len(dataset.candidate_rows) == 1
    row = dataset.candidate_rows[0]
    assert row.label == 0.0
    assert row.near_miss_negative is True
    assert row.cell_change_fraction == pytest.approx(2 / 100)
    assert dataset.per_game["toy"]["negative_rows"] == 1
    assert dataset.per_game["toy"]["ground_truth_corruptions"] == 0
    assert len(dataset.origin_rows) == 2
    assert dataset.origin_rows[0].label == 1.0
    assert dataset.origin_rows[1].label == 0.0


def test_scenario_arc_wmte_4761_builds_passing_artifact_with_controls(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4761-S0-GATE: artifact reports the matched gate measurements."""

    artifact = mod.build_artifact_from_dataset(
        _fixture_rows(),
        preconditions_checked={
            "offline_arcade": True,
            "cross_game_features_v3_import": True,
            "banked_game_count": 3,
        },
        random_seed=4761,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["verifier_is_oracle"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["loo_auroc_structural"] == pytest.approx(1.0)
    assert artifact["loo_auroc_marginal_control"] == pytest.approx(0.5)
    assert artifact["loo_auroc_majority_control"] == pytest.approx(0.5)
    assert artifact["structural_minus_marginal_delta_ci95"][0] > 0.0
    assert artifact["per_family_loo"]["object_relational"] == pytest.approx(1.0)
    assert artifact["origin_probe_auroc"] == pytest.approx(0.5)
    assert artifact["near_miss_negative_fraction"] == pytest.approx(1.0)
    assert artifact["in_sample_auroc"] == pytest.approx(1.0)
    assert artifact["s0_gate_passed"] is True

    out = mod.write_artifact(artifact, root=tmp_path)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == artifact
    assert out == tmp_path / mod.RESULT_RELATIVE_PATH


def test_scenario_arc_wmte_4761_blocked_precondition_has_no_claimed_auroc() -> None:
    """SCENARIO-ARC-WMTE-4761-BLOCKED-PRECONDITION: blocked artifacts fail closed."""

    artifact = mod.build_blocked_artifact(
        "blocked_structural_features_missing",
        {"offline_arcade": True, "cross_game_features_v3_import": False},
        random_seed=4761,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_structural_features_missing"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["loo_auroc_structural"] is None
    assert artifact["loo_auroc_ci95"] is None
    assert artifact["s0_gate_passed"] is False
    assert artifact["reproducibility_checksum"].startswith("sha256:")


def test_req_arc_wmte_4761_metric_edges_and_validation_fail_closed() -> None:
    """REQ-ARC-WMTE-4761: metric edge cases and malformed artifacts fail closed."""

    assert mod._clean_float("not-a-float") is None
    assert mod._clean_float(float("nan")) is None
    assert mod._cell_change_fraction(np.zeros((1, 1)), np.zeros((1, 2))) == 1.0
    assert mod._auroc([0.1], [1.0]) is None
    assert mod._logistic_scores([], [], []) == []
    assert mod._logistic_scores([[0.0]], [1.0], [[2.0], [3.0]]) == [0.5, 0.5]
    assert mod._bootstrap_mean_ci([], seed=1) is None
    assert mod._bootstrap_mean_ci([0.7], seed=1) == [0.7, 0.7]
    assert mod._delta_ci({"solo": None}, {"solo": 0.5}, 1) is None
    assert mod._ci_lower_gt([0.6, 0.7], 0.5) is True
    assert mod._ci_lower_gt(None, 0.5) is False
    assert mod._delta_ci_excludes_zero_positive([0.1, 0.2]) is True
    assert mod._delta_ci_excludes_zero_positive([-0.1, 0.2]) is False

    one_row = mod.TransitionCandidateRow(
        game="solo",
        label=1.0,
        structural=[1.0],
        marginal=[0.0],
        family_features={"object_relational": [1.0], "frame_delta": [0.0]},
        cell_change_fraction=0.0,
        near_miss_negative=False,
    )
    one_origin = mod.OriginProbeRow(game="solo", label=1.0, structural=[1.0])
    assert mod._in_sample_auroc([], "structural") is None
    assert mod._in_sample_auroc([one_row], "structural") is None
    assert mod._loo_metrics_candidate([one_row], "structural")["per_game"]["solo"] is None
    assert mod._loo_metrics_origin([one_origin])["per_game"]["solo"] is None

    merged = mod._merge_datasets([_fixture_rows(), mod.S0Dataset([], [], {"extra": {}})])
    assert len(merged.candidate_rows) == len(_fixture_rows().candidate_rows)
    assert "extra" in merged.per_game

    metrics = mod.evaluate_dataset(_fixture_rows())
    retire_metrics = copy.deepcopy(metrics)
    retire_metrics["loo_auroc_ci95"] = [0.4, 0.7]
    assert mod._artifact_verdict(retire_metrics).startswith("complete: structural_energy_s0_retired")
    null_metrics = copy.deepcopy(metrics)
    null_metrics["structural"]["loo_auroc"] = 0.58
    assert mod._artifact_verdict(null_metrics).startswith("complete: structural_energy_s0_honest_null")

    preconditions = mod.check_preconditions()
    assert preconditions["offline_arcade"] is True
    assert preconditions["cross_game_features_v3_import"] is True
    assert preconditions["ok"] is True

    artifact = mod.build_artifact_from_dataset(
        _fixture_rows(),
        preconditions_checked={"offline_arcade": True, "cross_game_features_v3_import": True},
    )
    bad_artifacts = [
        {},
        artifact | {"reproducibility_checksum": "sha256:bad"},
        artifact | {"verifier_is_oracle": True},
        artifact | {"inference_substrate": "wrong"},
        artifact | {"field_principles": {}},
        artifact | {"honest_verdict": ""},
        artifact | {"honest_verdict": "oops"},
        artifact | {"n_candidate_rows": 0},
        artifact | {"n_pos": 0},
        artifact | {"loo_auroc_majority_control": 0.4},
    ]
    blocked = mod.build_blocked_artifact(
        "blocked_structural_features_missing",
        {"offline_arcade": True, "cross_game_features_v3_import": False},
    )
    bad_artifacts.extend(
        [
            blocked | {"loo_auroc_structural": 0.5},
            blocked | {"s0_gate_passed": True},
        ]
    )
    for bad in bad_artifacts:
        if bad and bad.get("reproducibility_checksum") != "sha256:bad":
            bad["reproducibility_checksum"] = mod._checksum_payload(bad)
        with pytest.raises(ValueError):
            mod.validate_artifact(bad)
