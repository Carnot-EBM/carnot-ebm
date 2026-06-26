"""Tests for Exp 4771 S0' origin-matched structural-energy re-test.

Spec refs: REQ-ARC-WMTE-4771, SCENARIO-ARC-WMTE-4771-ORIGIN-MATCHED-GATE,
SCENARIO-ARC-WMTE-4771-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_4771_structural_energy_s0prime_origin_matched as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"


def _normalise(text: str) -> str:
    return " ".join(text.split())


def _spec_section() -> str:
    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-ARC-WMTE-4771")
    end = spec.index("### REQ-ARC-WMTE-4768", start)
    return spec[start:end]


def _grid(offset: int = 0) -> np.ndarray:
    g = np.zeros((10, 10), dtype=np.int16)
    g[2, 2 + offset] = 3
    return g


def _fixture_dataset() -> mod.S0PrimeDataset:
    rows: list[mod.OriginMatchedPredictionRow] = []
    per_game: dict[str, dict[str, int | bool]] = {}
    for gi, game in enumerate(("g0", "g1", "g2")):
        correct = 0
        wrong = 0
        for i in range(4):
            rows.append(
                mod.OriginMatchedPredictionRow(
                    game=game,
                    label=1.0,
                    prediction_origin="induced",
                    structural=[3.0 + gi, 1.0 + i, 0.0],
                    marginal=[0.0],
                    family_features={
                        "object_relational": [3.0 + gi, 1.0 + i],
                        "frame_delta": [0.0],
                    },
                    cell_change_fraction=0.0,
                    near_miss_negative=False,
                )
            )
            correct += 1
            rows.append(
                mod.OriginMatchedPredictionRow(
                    game=game,
                    label=0.0,
                    prediction_origin="induced",
                    structural=[-3.0 - gi, -1.0 - i, 0.0],
                    marginal=[0.0],
                    family_features={
                        "object_relational": [-3.0 - gi, -1.0 - i],
                        "frame_delta": [0.0],
                    },
                    cell_change_fraction=0.02,
                    near_miss_negative=True,
                )
            )
            wrong += 1
        per_game[game] = {
            "correct": correct,
            "wrong": wrong,
            "rows": correct + wrong,
            "contributes_to_loo": True,
        }
    return mod.S0PrimeDataset(rows=rows, per_game=per_game)


def test_req_arc_wmte_4771_spec_declares_origin_matched_contract() -> None:
    """REQ-ARC-WMTE-4771: OpenSpec declares the S0' contract before code."""

    section = _normalise(_spec_section())

    assert "REQ-ARC-WMTE-4771" in section
    assert "SCENARIO-ARC-WMTE-4771-ORIGIN-MATCHED-GATE" in section
    assert "SCENARIO-ARC-WMTE-4771-FIELD-PRINCIPLES" in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert "origin-matched" in section
    assert "object_relational" in section
    assert "frame_delta" in section
    assert "shuffled-label" in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert field in section
        assert _normalise(principle["principle"]) in section


def test_scenario_arc_wmte_4771_features_are_structural_only() -> None:
    """SCENARIO-ARC-WMTE-4771-ORIGIN-MATCHED-GATE: v3 view excludes marginals and dead families."""

    view = mod.structural_feature_view(_grid(0), _grid(1), action_key=(6, 2, 3))

    assert set(view) == {"structural", "marginal", "family_features", "feature_names"}
    assert set(view["family_features"]) == {"object_relational", "frame_delta"}
    assert len(view["structural"]) == len(view["family_features"]["object_relational"]) + len(
        view["family_features"]["frame_delta"]
    )
    assert len(view["marginal"]) > 0
    assert all(name.startswith(("object_relational_", "frame_delta_")) for name in view["feature_names"])
    assert not any(name.startswith(("v2_", "action_conditioned_", "predicate_distance_")) for name in view["feature_names"])


def test_req_arc_wmte_4771_dataset_rows_are_all_induced_origin() -> None:
    """REQ-ARC-WMTE-4771: positives and negatives are both induced predictions."""

    state = _grid(0)
    real_correct = _grid(1)
    real_wrong = _grid(2)
    model_outputs = [real_correct.copy(), _grid(0)]

    def predict_fn(_state: np.ndarray, _action: tuple[int, ...]) -> np.ndarray:
        return model_outputs.pop(0)

    dataset = mod.dataset_from_origin_matched_predictions(
        "toy",
        [
            (state, (6, 2, 3), real_correct),
            (state, (6, 3, 3), real_wrong),
        ],
        predict_fn=predict_fn,
    )

    assert [row.label for row in dataset.rows] == [1.0, 0.0]
    assert {row.prediction_origin for row in dataset.rows} == {"induced"}
    assert dataset.per_game["toy"]["correct"] == 1
    assert dataset.per_game["toy"]["wrong"] == 1
    assert dataset.per_game["toy"]["contributes_to_loo"] is True
    assert dataset.per_game["toy"]["ground_truth_corruptions"] == 0
    assert mod.origin_probe_audit(dataset)["loo_auroc"] == pytest.approx(0.5)


def test_scenario_arc_wmte_4771_builds_success_artifact_with_leak_controls(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4771-ORIGIN-MATCHED-GATE: artifact reports all decisive controls."""

    artifact = mod.build_artifact_from_dataset(
        _fixture_dataset(),
        preconditions_checked={
            "offline_arcade": True,
            "cross_game_features_v3_import": True,
            "banked_game_count": 3,
        },
        random_seed=4771,
        shuffle_resamples=48,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "success_structural_energy_s0prime_reopens_s1"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["loo_auroc_structural"] == pytest.approx(1.0)
    assert artifact["loo_auroc_marginal_control"] == pytest.approx(0.5)
    assert artifact["loo_auroc_majority_control"] == pytest.approx(0.5)
    assert artifact["loo_auroc_ci95"][0] > 0.5
    assert artifact["structural_minus_marginal_delta_ci95"][0] > 0.0
    assert artifact["per_family_loo"]["object_relational"] == pytest.approx(1.0)
    assert artifact["origin_probe_auroc"] == pytest.approx(0.5)
    assert artifact["shuffled_label_control_auroc"] <= 0.55
    assert artifact["in_sample_auroc"] == pytest.approx(1.0)
    assert artifact["s0prime_gate_passed"] is True
    assert artifact["retire_if_same_verdict"] is True
    assert artifact["per_game_class_balance"]["g0"]["correct"] == 4
    assert artifact["origin_probe"]["status"] == "origin_matched_single_origin_all_induced"

    out = mod.write_artifact(artifact, root=tmp_path)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == artifact
    assert out == tmp_path / mod.RESULT_RELATIVE_PATH


def test_scenario_arc_wmte_4771_blocked_precondition_has_no_claimed_auroc() -> None:
    """SCENARIO-ARC-WMTE-4771-ORIGIN-MATCHED-GATE: blocked artifacts fail closed."""

    artifact = mod.build_blocked_artifact(
        "blocked_offline_arcade_missing",
        {"offline_arcade": False, "cross_game_features_v3_import": False},
        random_seed=4771,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_offline_arcade_missing"
    assert artifact["loo_auroc_structural"] is None
    assert artifact["origin_probe_auroc"] is None
    assert artifact["shuffled_label_control_auroc"] is None
    assert artifact["s0prime_gate_passed"] is False
    assert artifact["reproducibility_checksum"].startswith("sha256:")


def test_req_arc_wmte_4771_retirement_and_validation_edges() -> None:
    """REQ-ARC-WMTE-4771: collapse, leak-control, and schema edges fail closed."""

    dataset = _fixture_dataset()
    metrics = mod.evaluate_dataset(dataset, random_seed=4771, shuffle_resamples=24)
    assert mod._artifact_verdict(metrics) == "success_structural_energy_s0prime_reopens_s1"

    collapse = copy.deepcopy(metrics)
    collapse["loo_auroc_ci95"] = [0.45, 0.7]
    assert mod._artifact_verdict(collapse) == "complete_structural_energy_s0prime_retired_was_origin_leak"

    shuffled_leak = copy.deepcopy(metrics)
    shuffled_leak["shuffled_label_control_auroc"] = 0.75
    assert mod._gate_passed(shuffled_leak) is False
    assert mod._artifact_verdict(shuffled_leak) == "complete_structural_energy_s0prime_retired_leak_control_failed"

    mixed_origin = mod.S0PrimeDataset(
        rows=[
            dataset.rows[0],
            mod.OriginMatchedPredictionRow(
                game="bad",
                label=0.0,
                prediction_origin="real",
                structural=[0.0],
                marginal=[0.0],
                family_features={"object_relational": [0.0], "frame_delta": [0.0]},
                cell_change_fraction=0.0,
                near_miss_negative=False,
            ),
        ],
        per_game={"bad": {"correct": 1, "wrong": 1, "rows": 2, "contributes_to_loo": True}},
    )
    assert mod.origin_probe_audit(mixed_origin)["loo_auroc"] == pytest.approx(1.0)

    one_class = mod.S0PrimeDataset(
        rows=[dataset.rows[0]],
        per_game={"solo": {"correct": 1, "wrong": 0, "rows": 1, "contributes_to_loo": False}},
    )
    assert mod.evaluate_dataset(one_class)["structural"]["per_game"]["g0"]["skipped"] is True

    assert mod._bootstrap_mean_ci([], seed=1) is None
    assert mod._bootstrap_mean_ci([0.7], seed=1) == [0.7, 0.7]
    assert mod._delta_ci({"solo": {"auroc": None}}, {"solo": {"auroc": 0.5}}, 1) is None
    assert mod._ci_lower_gt([0.6, 0.7], 0.5) is True
    assert mod._ci_lower_gt(None, 0.5) is False
    assert mod._delta_ci_excludes_zero_positive([0.1, 0.2]) is True
    assert mod._delta_ci_excludes_zero_positive([-0.1, 0.2]) is False

    preconditions = mod.check_preconditions()
    assert preconditions["offline_arcade"] is True
    assert preconditions["cross_game_features_v3_import"] is True
    assert preconditions["ok"] is True

    artifact = mod.build_artifact_from_dataset(
        dataset,
        preconditions_checked={"offline_arcade": True, "cross_game_features_v3_import": True},
        shuffle_resamples=24,
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
        artifact | {"origin_probe_auroc": 0.7},
        artifact | {"shuffled_label_control_auroc": 0.7},
    ]
    blocked = mod.build_blocked_artifact(
        "blocked_structural_features_missing",
        {"offline_arcade": True, "cross_game_features_v3_import": False},
    )
    bad_artifacts.extend(
        [
            blocked | {"loo_auroc_structural": 0.5},
            blocked | {"s0prime_gate_passed": True},
        ]
    )
    for bad in bad_artifacts:
        if bad and bad.get("reproducibility_checksum") != "sha256:bad":
            bad["reproducibility_checksum"] = mod._checksum_payload(bad)
        with pytest.raises(ValueError):
            mod.validate_artifact(bad)
