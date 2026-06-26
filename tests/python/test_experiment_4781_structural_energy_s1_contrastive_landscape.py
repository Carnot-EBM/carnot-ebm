"""Tests for Exp 4781 S1 contrastive structural-energy landscape.

Spec refs: REQ-ARC-WMTE-4781,
SCENARIO-ARC-WMTE-4781-CONTRASTIVE-LANDSCAPE-GATE.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_4781_structural_energy_s1_contrastive_landscape as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"


def _normalise(text: str) -> str:
    return " ".join(text.split())


def _spec_section() -> str:
    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-ARC-WMTE-4781")
    end = spec.index("### REQ-ARC-WMTE-4768", start)
    return spec[start:end]


def _grid(offset: int = 0) -> np.ndarray:
    grid = np.zeros((10, 10), dtype=np.int16)
    grid[2, 2 + offset] = 3
    return grid


def _fixture_dataset(*, frame_delta_collapsed: bool = False) -> mod.S1EnergyDataset:
    rows: list[mod.EnergyCandidateRow] = []
    per_game: dict[str, dict[str, int | bool]] = {}
    for gi, game in enumerate(("g0", "g1", "g2")):
        correct = 0
        wrong = 0
        for i in range(4):
            pos_frame = [1.5 + gi, 0.25 + i]
            neg_frame = [-1.5 - gi, -0.25 - i]
            if frame_delta_collapsed:
                pos_frame = [0.0, 0.0]
                neg_frame = [0.0, 0.0]
            rows.append(
                mod.EnergyCandidateRow(
                    game=game,
                    label=1.0,
                    prediction_origin="induced",
                    structural=[3.0 + gi, 1.0 + i, *pos_frame],
                    marginal=[0.0],
                    family_features={
                        "object_relational": [3.0 + gi, 1.0 + i],
                        "frame_delta": pos_frame,
                    },
                    cell_change_fraction=0.0,
                    near_miss_negative=False,
                )
            )
            correct += 1
            rows.append(
                mod.EnergyCandidateRow(
                    game=game,
                    label=0.0,
                    prediction_origin="induced",
                    structural=[-3.0 - gi, -1.0 - i, *neg_frame],
                    marginal=[0.0],
                    family_features={
                        "object_relational": [-3.0 - gi, -1.0 - i],
                        "frame_delta": neg_frame,
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
            "ground_truth_corruptions": 0,
        }
    return mod.S1EnergyDataset(rows=rows, per_game=per_game)


def test_req_arc_wmte_4781_spec_declares_contrastive_landscape_contract() -> None:
    """REQ-ARC-WMTE-4781: OpenSpec declares the S1 contract before code."""

    section = _normalise(_spec_section())

    assert "REQ-ARC-WMTE-4781" in section
    assert "SCENARIO-ARC-WMTE-4781-CONTRASTIVE-LANDSCAPE-GATE" in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert "contrastive" in section
    assert "leave-one-GAME-out AUROC ranking by -E" in section
    assert "object_relational" in section
    assert "frame_delta" in section
    assert "denoising-direction" in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert field in section
        assert _normalise(principle["principle"]) in section


def test_scenario_arc_wmte_4781_features_are_structural_only() -> None:
    """SCENARIO-ARC-WMTE-4781-CONTRASTIVE-LANDSCAPE-GATE: v3 view excludes dead families."""

    view = mod.structural_feature_view(_grid(0), _grid(1), action_key=(6, 2, 3))

    assert set(view) == {"structural", "marginal", "family_features", "feature_names"}
    assert set(view["family_features"]) == {"object_relational", "frame_delta"}
    assert len(view["structural"]) == len(view["family_features"]["object_relational"]) + len(
        view["family_features"]["frame_delta"]
    )
    assert len(view["marginal"]) > 0
    assert all(name.startswith(("object_relational_", "frame_delta_")) for name in view["feature_names"])
    assert not any(name.startswith(("v2_", "action_conditioned_", "predicate_distance_")) for name in view["feature_names"])


def test_req_arc_wmte_4781_margin_energy_ranks_correct_lower() -> None:
    """REQ-ARC-WMTE-4781: contrastive training learns E(correct) < E(wrong)."""

    dataset = _fixture_dataset()
    model = mod.fit_contrastive_energy(
        dataset.rows,
        feature_attr="structural",
        seed=4781,
        config=mod.EnergyTrainingConfig(epochs=40, max_pairs_per_epoch=64),
    )
    positive = next(row for row in dataset.rows if row.label == 1.0)
    negative = next(row for row in dataset.rows if row.label == 0.0)

    assert model.energy(positive.structural) < model.energy(negative.structural)
    assert model.score(positive.structural) > model.score(negative.structural)
    assert mod.denoising_direction_agreement(dataset.rows, seed=4781) == pytest.approx(1.0)


def test_scenario_arc_wmte_4781_builds_success_artifact_with_controls(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4781-CONTRASTIVE-LANDSCAPE-GATE: artifact reports all gates."""

    seeds = list(range(4781, 4791))
    artifact = mod.build_artifact_from_dataset(
        _fixture_dataset(),
        preconditions_checked={
            "offline_arcade": True,
            "cross_game_features_v3_import": True,
            "banked_game_count": 3,
        },
        random_seeds_used=seeds,
        training_config=mod.EnergyTrainingConfig(epochs=30, max_pairs_per_epoch=64),
        shuffle_resamples=8,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "success_structural_energy_s1_landscape_authorizes_s2"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["energy_ranking_loo_auroc_mean"] == pytest.approx(1.0)
    assert artifact["energy_ranking_loo_auroc_ci95"][0] > 0.5
    assert artifact["energy_ranking_loo_auroc_marginal_control_mean"] == pytest.approx(0.5)
    assert artifact["denoising_direction_agreement"] == pytest.approx(1.0)
    assert artifact["origin_probe_auroc"] == pytest.approx(0.5)
    assert artifact["shuffled_label_control_auroc"] <= 0.55
    assert artifact["per_family_loo"]["object_relational"] == pytest.approx(1.0)
    assert artifact["per_family_loo"]["frame_delta"] == pytest.approx(1.0)
    assert artifact["in_sample_auroc"] == pytest.approx(1.0)
    assert artifact["n_seeds"] == 10
    assert artifact["random_seeds_used"] == seeds
    assert artifact["s1_gate_passed"] is True

    out = mod.write_artifact(artifact, root=tmp_path)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == artifact
    assert out == tmp_path / mod.RESULT_RELATIVE_PATH


def test_scenario_arc_wmte_4781_blocked_precondition_has_no_claimed_auroc() -> None:
    """SCENARIO-ARC-WMTE-4781-CONTRASTIVE-LANDSCAPE-GATE: blocked artifacts fail closed."""

    artifact = mod.build_blocked_artifact(
        "blocked_structural_features_missing",
        {"offline_arcade": True, "cross_game_features_v3_import": False},
        random_seeds_used=list(range(4781, 4791)),
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_structural_features_missing"
    assert artifact["energy_ranking_loo_auroc_mean"] is None
    assert artifact["energy_ranking_loo_auroc_ci95"] is None
    assert artifact["origin_probe_auroc"] is None
    assert artifact["s1_gate_passed"] is False
    assert artifact["reproducibility_checksum"].startswith("sha256:")


def test_req_arc_wmte_4781_bounded_result_and_validation_edges() -> None:
    """REQ-ARC-WMTE-4781: failed gates and malformed artifacts fail closed."""

    seeds = list(range(4781, 4791))
    metrics = mod.evaluate_dataset(
        _fixture_dataset(),
        random_seeds_used=seeds,
        training_config=mod.EnergyTrainingConfig(epochs=20, max_pairs_per_epoch=64),
        shuffle_resamples=4,
    )
    assert mod._artifact_verdict(metrics) == "success_structural_energy_s1_landscape_authorizes_s2"

    one_family = copy.deepcopy(metrics)
    one_family["per_family_loo"]["frame_delta"] = 0.5
    assert mod._s1_gate_passed(one_family) is False
    assert mod._artifact_verdict(one_family) == "complete_structural_energy_s1_discriminates_no_usable_landscape"

    leak = copy.deepcopy(metrics)
    leak["origin_probe"]["loo_auroc"] = 0.7
    assert mod._s1_gate_passed(leak) is False

    collapsed = mod.build_artifact_from_dataset(
        _fixture_dataset(frame_delta_collapsed=True),
        preconditions_checked={"offline_arcade": True, "cross_game_features_v3_import": True},
        random_seeds_used=seeds,
        training_config=mod.EnergyTrainingConfig(epochs=20, max_pairs_per_epoch=64),
        shuffle_resamples=4,
    )
    mod.validate_artifact(collapsed)
    assert collapsed["honest_verdict"] == "complete_structural_energy_s1_discriminates_no_usable_landscape"
    assert collapsed["s1_gate_passed"] is False
    assert collapsed["per_family_loo"]["frame_delta"] == pytest.approx(0.5)

    assert mod._bootstrap_mean_ci([], seed=1) is None
    assert mod._bootstrap_mean_ci([0.7], seed=1) == [0.7, 0.7]
    assert mod._ci_lower_gt([0.6, 0.7], 0.5) is True
    assert mod._ci_lower_gt(None, 0.5) is False
    assert mod._origin_probe_audit(mod.S1EnergyDataset(rows=[], per_game={}))["loo_auroc"] is None

    artifact = mod.build_artifact_from_dataset(
        _fixture_dataset(),
        preconditions_checked={"offline_arcade": True, "cross_game_features_v3_import": True},
        random_seeds_used=seeds,
        training_config=mod.EnergyTrainingConfig(epochs=20, max_pairs_per_epoch=64),
        shuffle_resamples=4,
    )
    bad_artifacts = [
        {},
        artifact | {"reproducibility_checksum": "sha256:bad"},
        artifact | {"verifier_is_oracle": True},
        artifact | {"inference_substrate": "wrong"},
        artifact | {"field_principles": {}},
        artifact | {"honest_verdict": ""},
        artifact | {"honest_verdict": "oops"},
        artifact | {"n_seeds": 9},
        artifact | {"n_candidate_rows": 0},
        artifact | {"n_pos": 0},
        artifact | {"origin_probe_auroc": 0.7},
        artifact | {"shuffled_label_control_auroc": 0.7},
    ]
    blocked = mod.build_blocked_artifact(
        "blocked_offline_arcade_missing",
        {"offline_arcade": False, "cross_game_features_v3_import": True},
        random_seeds_used=seeds,
    )
    bad_artifacts.extend(
        [
            blocked | {"energy_ranking_loo_auroc_mean": 0.5},
            blocked | {"s1_gate_passed": True},
        ]
    )
    for bad in bad_artifacts:
        if bad and bad.get("reproducibility_checksum") != "sha256:bad":
            bad["reproducibility_checksum"] = mod._checksum_payload(bad)
        with pytest.raises(ValueError):
            mod.validate_artifact(bad)


def test_req_arc_wmte_4781_numeric_and_dataset_edge_paths() -> None:
    """REQ-ARC-WMTE-4781: defensive metric and conversion edges are deterministic."""

    empty_model = mod.ContrastiveEnergyModel([], [], [])
    assert empty_model.score([1.0]) == 0.0
    assert empty_model.energy([1.0]) == -0.0

    short_feature_model = mod.ContrastiveEnergyModel([0.0, 0.0], [1.0, 1.0], [1.0, 0.0])
    assert short_feature_model.score([2.0]) == pytest.approx(2.0)
    assert mod._clean_float("bad") is None
    assert mod._align_features([1.0], 3).tolist() == [1.0, 0.0, 0.0]
    assert mod._auroc([1.0], [1.0]) is None
    assert mod._feature_matrix([], "structural").shape == (0, 0)
    assert mod.fit_contrastive_energy([], feature_attr="structural").weights == []

    one_class = [row for row in _fixture_dataset().rows if row.label == 1.0][:2]
    assert mod.fit_contrastive_energy(one_class, feature_attr="structural").score([1.0, 2.0, 3.0, 4.0]) == 0.0
    assert mod._in_sample_auroc([], "structural", random_seeds_used=[4781], training_config=mod.EnergyTrainingConfig()) is None
    assert mod._in_sample_auroc(one_class, "structural", random_seeds_used=[4781], training_config=mod.EnergyTrainingConfig()) is None

    clip_rows = [
        mod.EnergyCandidateRow("g0", 1.0, "induced", [1.0], [0.0], {"object_relational": [1.0], "frame_delta": [1.0]}, 0.0, False),
        mod.EnergyCandidateRow("g0", 0.0, "induced", [-1.0], [0.0], {"object_relational": [-1.0], "frame_delta": [-1.0]}, 0.0, True),
    ]
    clipped = mod.fit_contrastive_energy(
        clip_rows,
        config=mod.EnergyTrainingConfig(margin=100.0, epochs=1, learning_rate=1000.0, l2=0.0),
    )
    assert np.linalg.norm(np.asarray(clipped.weights)) == pytest.approx(100.0)

    single_class_train_rows = [
        mod.EnergyCandidateRow("g0", 1.0, "induced", [1.0], [0.0], {"object_relational": [1.0], "frame_delta": [1.0]}, 0.0, False),
        mod.EnergyCandidateRow("g0", 0.0, "induced", [-1.0], [0.0], {"object_relational": [-1.0], "frame_delta": [-1.0]}, 0.0, True),
        mod.EnergyCandidateRow("g1", 1.0, "induced", [2.0], [0.0], {"object_relational": [2.0], "frame_delta": [2.0]}, 0.0, False),
    ]
    fold_metrics = mod._loo_energy_metrics(
        single_class_train_rows,
        "structural",
        random_seeds_used=[4781],
        training_config=mod.EnergyTrainingConfig(epochs=1),
    )
    assert fold_metrics["per_game_by_seed"][0]["g0"]["skip_reason"] == "train_fold_single_class"

    assert mod.denoising_direction_agreement(clip_rows) is None
    assert mod.denoising_direction_agreement(single_class_train_rows) is None
    no_pair_rows = [
        *clip_rows,
        mod.EnergyCandidateRow("g1", 1.0, "induced", [2.0], [0.0], {"object_relational": [2.0], "frame_delta": [2.0]}, 0.0, False),
    ]
    assert mod.denoising_direction_agreement(no_pair_rows) is None

    mixed_origin = mod.S1EnergyDataset(
        rows=[replace_row := _fixture_dataset().rows[0], mod.EnergyCandidateRow(
            "bad",
            0.0,
            "real",
            [0.0],
            [0.0],
            {"object_relational": [0.0], "frame_delta": [0.0]},
            0.0,
            False,
        )],
        per_game={"bad": {"correct": 1, "wrong": 1, "rows": 2}},
    )
    assert replace_row.prediction_origin == "induced"
    assert mod._origin_probe_audit(mixed_origin)["loo_auroc"] == 1.0
    assert mod._shuffled_label_control([], random_seeds_used=[4781], training_config=mod.EnergyTrainingConfig(), shuffle_resamples=1)[
        "loo_auroc"
    ] is None

    preconditions = mod.check_preconditions()
    assert preconditions["offline_arcade"] is True
    assert preconditions["cross_game_features_v3_import"] is True
    assert preconditions["ok"] is True

    source = mod.s0prime.S0PrimeDataset(
        rows=[
            mod.s0prime.OriginMatchedPredictionRow(
                game="toy",
                label=1.0,
                prediction_origin="induced",
                structural=[1.0],
                marginal=[0.0],
                family_features={"object_relational": [1.0], "frame_delta": [1.0]},
                cell_change_fraction=0.0,
                near_miss_negative=False,
            )
        ],
        per_game={"toy": {"correct": 1, "wrong": 0, "rows": 1}},
    )
    converted = mod._from_s0prime_dataset(source)
    assert converted.rows[0].game == "toy"
    assert converted.rows[0].prediction_origin == "induced"
    assert converted.per_game["toy"]["correct"] == 1
