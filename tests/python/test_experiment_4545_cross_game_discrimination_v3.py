"""Tests for Exp 4545 cross-game discriminative verifier v3.

Spec refs: REQ-LEARN-4476, SCENARIO-LEARN-4476-GATE.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_4545_cross_game_discrimination_v3 as mod


def _success_metrics() -> dict[str, object]:
    return {
        "loo_auroc_mean": 0.6744657162333668,
        "loo_auroc_ci": [0.612, 0.731],
        "in_sample_auroc": 0.8710834214701216,
        "per_game_loo_auroc": {"a": 0.65, "b": 0.70},
        "n_held_out_games": 2,
        "n_pos": 12,
        "n_neg": 12,
    }


def _feature_metrics() -> dict[str, dict[str, float]]:
    return {
        "loo_auroc": {
            "v2": 0.503096152732577,
            "v2_plus_action_conditioned": 0.5065398442857277,
            "v2_plus_frame_delta": 0.6798088810175374,
            "v2_plus_object_relational": 0.6314590310571959,
            "v2_plus_predicate_distance": 0.5193000293549701,
            "v3_full": 0.6744657162333668,
        },
        "in_sample_auroc": {
            "v2": 0.7256356865855809,
            "v2_plus_action_conditioned": 0.759,
            "v2_plus_frame_delta": 0.850,
            "v2_plus_object_relational": 0.821,
            "v2_plus_predicate_distance": 0.781,
            "v3_full": 0.8710834214701216,
        },
    }


def test_req_learn_4476_success_artifact_has_required_principle_fields(tmp_path: Path) -> None:
    """REQ-LEARN-4476: Exp 4545 records the oracle-distinct LOO win schema."""

    artifact = mod.build_artifact(
        metrics=_success_metrics(),
        feature_class_metrics=_feature_metrics(),
        preconditions_checked={
            "arc_value_learner_import": True,
            "cached_cross_game_candidate_corpus": True,
        },
        random_seed=0,
        corpus_checksum="sha256:corpus",
        tests_pass=True,
    )

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
        assert field in artifact["field_principles"]
    assert artifact["honest_verdict"] == (
        "success: cross_game_discrimination_loo_auroc_0.674_above_chance"
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["loo_auroc_mean"] == pytest.approx(0.6744657162333668)
    assert artifact["loo_auroc_ci"] == [0.612, 0.731]
    assert artifact["positive_control_passed"] is True
    assert artifact["false_negative_risk_checked"] is True
    assert artifact["loo_ci_excludes_chance"] is True
    assert artifact["missing_verifier_gaps"] == []
    assert artifact["feature_families_added"] == [
        "object_relational",
        "frame_delta",
        "action_conditioned",
        "predicate_distance",
    ]
    assert mod.artifact_schema_errors(artifact) == []

    out = mod.write_artifact(artifact, root=tmp_path)
    assert out == tmp_path / mod.RESULT_RELATIVE_PATH
    assert json.loads(out.read_text(encoding="utf-8")) == artifact


def test_scenario_learn_4476_honest_null_logs_in_sample_transfer_gap() -> None:
    """SCENARIO-LEARN-4476-GATE: chance LOO with positive control sharpens the gap."""

    null_metrics = {
        **_success_metrics(),
        "loo_auroc_mean": 0.503,
        "loo_auroc_ci": [0.451, 0.557],
        "in_sample_auroc": 0.812,
    }
    feature_metrics = _feature_metrics()
    feature_metrics["loo_auroc"] = {
        "v2": 0.503,
        "v2_plus_frame_delta": 0.511,
        "v2_plus_object_relational": 0.509,
        "v2_plus_action_conditioned": 0.505,
        "v2_plus_predicate_distance": 0.507,
        "v3_full": 0.503,
    }

    artifact = mod.build_artifact(
        metrics=null_metrics,
        feature_class_metrics=feature_metrics,
        preconditions_checked={"cached_cross_game_candidate_corpus": True},
        random_seed=0,
        corpus_checksum="sha256:corpus",
        tests_pass=True,
    )

    assert artifact["honest_verdict"] == (
        "complete: cross_game_discrimination_still_chance_gap_sharpened_honest_null"
    )
    assert artifact["positive_control_passed"] is True
    assert artifact["false_negative_risk_checked"] is True
    assert artifact["loo_ci_excludes_chance"] is False
    assert artifact["missing_verifier_gaps"]
    assert "frame_delta" in artifact["missing_verifier_gaps"][0]
    assert mod.artifact_schema_errors(artifact) == []


def test_req_learn_4476_bootstrap_ci_is_deterministic_and_chance_aware() -> None:
    """REQ-LEARN-4476: LOO AUROC CI is deterministic for reproducible gate claims."""

    ci1 = mod.bootstrap_mean_ci([0.60, 0.70, 0.80], random_seed=7, n_bootstrap=300)
    ci2 = mod.bootstrap_mean_ci([0.60, 0.70, 0.80], random_seed=7, n_bootstrap=300)

    assert ci1 == ci2
    assert ci1[0] > 0.5
    assert mod.bootstrap_mean_ci([], random_seed=7, n_bootstrap=10) == [None, None]


def test_req_learn_4476_evaluate_loo_scores_unseen_games() -> None:
    """REQ-LEARN-4476: LOO scoring trains on every game except the held-out one."""

    x_rows: list[list[float]] = []
    y_rows: list[float] = []
    per_game: dict[str, dict[str, int]] = {}
    for game_index, game in enumerate(("a", "b", "c")):
        pos = [[2.0 + game_index * 0.1], [2.2 + game_index * 0.1]]
        neg = [[-2.0 - game_index * 0.1], [-2.2 - game_index * 0.1]]
        x_rows.extend(pos + neg)
        y_rows.extend([1.0, 1.0, 0.0, 0.0])
        per_game[game] = {"pos": 2, "neg": 2}

    metrics = mod.evaluate_loo(x_rows, y_rows, per_game, iters=250, lr=0.25)

    assert metrics["loo_auroc_mean"] == pytest.approx(1.0)
    assert metrics["in_sample_auroc"] == pytest.approx(1.0)
    assert metrics["n_held_out_games"] == 3
    assert metrics["n_pos"] == 6
    assert metrics["n_neg"] == 6


def test_req_learn_4476_helper_edge_cases_are_stable() -> None:
    """REQ-LEARN-4476: checksum, tie, skip, and positive-control branches are stable."""

    checksum = mod.corpus_checksum([[1.2345678901234]], [1.0], {"a": {"pos": 1, "neg": 0}})
    skipped = mod.evaluate_loo([[1.0], [-1.0]], [1.0, 0.0], {"a": {"pos": 1, "neg": 1}}, iters=80)
    weak = mod.build_artifact(
        metrics={
            "loo_auroc_mean": 0.62,
            "loo_auroc_ci": [0.56, 0.70],
            "in_sample_auroc": 0.49,
        },
        feature_class_metrics={"loo_auroc": {"v2": "bad"}, "in_sample_auroc": {"v2": "bad"}},
        preconditions_checked={"cached_cross_game_candidate_corpus": True},
        random_seed=0,
        corpus_checksum=checksum,
        tests_pass=False,
    )
    generic_gap = mod.build_artifact(
        metrics={
            "loo_auroc_mean": 0.51,
            "loo_auroc_ci": [0.48, 0.55],
            "in_sample_auroc": 0.62,
        },
        feature_class_metrics={
            "loo_auroc": {"v2": 0.50, "v2_plus_frame_delta": 0.57},
            "in_sample_auroc": {"v2_plus_frame_delta": 0.49},
        },
        preconditions_checked={"cached_cross_game_candidate_corpus": True},
        random_seed=0,
        corpus_checksum=checksum,
        tests_pass=False,
    )

    assert mod._clean_float("not-a-number") is None
    assert mod._loo_ci_excludes_chance(None, [0.6, 0.7]) is False
    assert mod.tie_aware_auroc([0.1, 0.2], [1.0, 1.0]) == pytest.approx(0.5)
    assert checksum.startswith("sha256:")
    assert skipped["n_held_out_games"] == 0
    assert skipped["loo_auroc_mean"] is None
    assert weak["honest_verdict"] == (
        "complete: cross_game_discrimination_positive_control_failed_harness_uninformative"
    )
    assert weak["positive_control_passed"] is False
    assert generic_gap["missing_verifier_gaps"] == [
        "v3 feature families did not produce a reliable in-sample or LOO discriminator; missing stronger "
        "negative labels and mechanic-specific predicate abstractions."
    ]


def test_req_learn_4476_feature_class_metrics_use_stable_slices() -> None:
    """REQ-LEARN-4476: feature-class ablations use the exported v3 slices."""

    width = len(mod.cross_game_feature_names_v3())
    x_rows: list[list[float]] = []
    y_rows: list[float] = []
    per_game: dict[str, dict[str, int]] = {}
    for game_index, game in enumerate(("a", "b", "c")):
        for sign, label in ((2.0, 1.0), (2.2, 1.0), (-2.0, 0.0), (-2.2, 0.0)):
            row = [0.0] * width
            row[0] = sign + game_index * 0.1
            x_rows.append(row)
            y_rows.append(label)
        per_game[game] = {"pos": 2, "neg": 2}

    metrics = mod.evaluate_feature_classes(x_rows, y_rows, per_game, iters=250, lr=0.25)

    assert set(metrics["loo_auroc"]) == {
        "v2",
        "v2_plus_action_conditioned",
        "v2_plus_frame_delta",
        "v2_plus_object_relational",
        "v2_plus_predicate_distance",
        "v3_full",
    }
    assert metrics["loo_auroc"]["v3_full"] == pytest.approx(1.0)
    assert metrics["in_sample_auroc"]["v2"] == pytest.approx(1.0)


def test_req_learn_4476_schema_rejects_oracle_or_nonterminal_claims() -> None:
    """REQ-LEARN-4476: fabricated oracle and non-terminal artifacts fail closed."""

    artifact = mod.build_artifact(
        metrics=_success_metrics(),
        feature_class_metrics=_feature_metrics(),
        preconditions_checked={"cached_cross_game_candidate_corpus": True},
        random_seed=0,
        corpus_checksum="sha256:corpus",
        tests_pass=True,
    )
    oracle = {**artifact, "verifier_is_oracle": True}
    nonterminal = {**artifact, "honest_verdict": "loo won"}
    missing_principle = {
        **artifact,
        "field_principles": {k: v for k, v in artifact["field_principles"].items() if k != "loo_auroc_ci"},
    }
    invalids = [
        ({**artifact, "inference_substrate": "live_llm_inference"}, "inference_substrate"),
        ({**artifact, "preconditions_checked": []}, "preconditions_checked"),
        ({**artifact, "loo_auroc_ci": [0.6]}, "loo_auroc_ci"),
        (
            {**artifact, "positive_control_passed": False, "false_negative_risk_checked": True},
            "false_negative_risk_checked",
        ),
        (
            {**artifact, "positive_control_passed": False, "loo_ci_excludes_chance": True},
            "above-chance",
        ),
        ({k: v for k, v in artifact.items() if k != "reproducibility_checksum"}, "reproducibility_checksum"),
        ({k: v for k, v in artifact.items() if k != "honest_verdict"}, "missing required field"),
    ]

    assert "verifier_is_oracle must be false" in mod.artifact_schema_errors(oracle)
    assert "honest_verdict must start with a terminal prefix" in mod.artifact_schema_errors(nonterminal)
    assert "field_principles must annotate every required artifact field" in mod.artifact_schema_errors(
        missing_principle
    )
    for invalid, expected in invalids:
        assert any(expected in error for error in mod.artifact_schema_errors(invalid))
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.write_artifact(oracle)


def test_req_learn_4476_blocked_artifact_is_terminal_and_schema_valid() -> None:
    """REQ-LEARN-4476: missing corpus exits as a terminal blocked artifact."""

    artifact = mod._blocked_artifact(
        "cached_cross_game_candidate_corpus",
        {"cached_cross_game_candidate_corpus": False},
        random_seed=0,
    )

    assert artifact["honest_verdict"] == "complete: blocked_cached_cross_game_candidate_corpus"
    assert artifact["missing_verifier_gaps"] == [
        "blocked_cached_cross_game_candidate_corpus: required cross-game corpus unavailable"
    ]
    assert mod.artifact_schema_errors(artifact) == []
