"""Tests for source-grounded calibration against matched controls.

Spec refs: REQ-AUTO-7413 and SCENARIO-AUTO-7413-01 through
SCENARIO-AUTO-7413-07.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7413_v650_source_calibration as mod


ROOT = Path(__file__).resolve().parents[2]


def _row(
    partition: str,
    index: int,
    label: int | None,
    *,
    group: str | None = None,
) -> dict[str, Any]:
    """Build one compact row with the same predictor/evaluator separation."""

    value = 0.85 if label == 1 else 0.15
    return {
        "row_key": f"{partition}-row-{index}",
        "group_id": group or f"{partition}-group-{index}",
        "partition": partition,
        "source_features": {
            "numeric_novelty_with_context": value,
            "falsifiability_score": value,
            "normalized_number_token_overlap": 1.0 - value,
            "normalized_content_token_overlap": 1.0 - value,
            "max_answer_source_sentence_overlap": 1.0 - value,
            "missing_or_empty_source": 0.0,
        },
        "response_only_ablation": {
            "entity_uptake": value,
            "falsifiability_score": value,
        },
        "label": label,
        "label_authority": "machine_annotation",
    }


def _fixture_rows() -> list[dict[str, Any]]:
    """Return disjoint roles with both classes and one unscored test row."""

    rows: list[dict[str, Any]] = []
    for partition in ("train", "probability_calibration", "policy_calibration"):
        rows.extend(_row(partition, index, index % 2) for index in range(6))
    rows.extend(_row("final_test", index, index % 2) for index in range(8))
    rows.append(_row("final_test", 8, None))
    return rows


def _passing_receipts() -> list[dict[str, Any]]:
    """Provide all frozen affected and terminal receipt names."""

    return [
        {"name": name, "passed": True, "exit_code": 0, "timed_out": False}
        for name in (*mod.AFFECTED_CHECK_NAMES, *mod.TERMINAL_CHECK_NAMES)
    ]


def test_req_auto_7413_spec_precedes_implementation() -> None:
    """REQ-AUTO-7413 owns all required fields and seven scenarios."""

    text = (ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-AUTO-7413") :]
    for suffix in ("01", "02", "03", "04", "05", "06", "07"):
        assert f"SCENARIO-AUTO-7413-{suffix}" in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section or field in {
            "experiment_id",
            "milestone",
            "status",
        }


def test_scenario_7413_01_preconditions_and_feature_boundary() -> None:
    """SCENARIO-AUTO-7413-01 authenticates inputs and rejects feature leaks."""

    checks, hashes, loaded = mod.collect_preconditions(ROOT)
    assert checks and hashes and set(loaded) == {"upstream", "protocol", "features", "corpus"}
    assert all(row["passed"] for row in checks)
    row = _row("train", 0, 0)
    assert len(mod.feature_vector(row, "l2_logistic_six_input")) == 6
    assert len(mod.feature_vector(row, "response_only_2_4_1_gibbs")) == 2
    changed = deepcopy(row)
    changed["source_features"]["teacher_score"] = 0.2
    with pytest.raises(ValueError, match="feature keys"):
        mod.feature_vector(changed, "source_aware_6_4_1_gibbs")
    with pytest.raises(ValueError, match="registered arm"):
        mod.feature_vector(row, "unknown")


def test_scenario_7413_01_fit_uses_registered_roles_and_five_seeds() -> None:
    """SCENARIO-AUTO-7413-01 fits all 20 units with disjoint role receipts."""

    units = mod.fit_registered_units(_fixture_rows(), steps=3)
    assert len(units) == 4 * 5
    assert {(unit["arm"], unit["seed"]) for unit in units} == {
        (arm, seed) for arm in mod.ARMS for seed in mod.TRAINING_SEEDS
    }
    for unit in units:
        assert unit["fit_partition"] == "train"
        assert unit["affine"]["fitting_partition"] == "probability_calibration"
        assert unit["selected_policy"]["selection_partition"] == "policy_calibration"
        assert unit["optimizer_work"]["maximum_steps"] == 3
        assert unit["input_hashes"]["training"] != unit["input_hashes"]["calibration"]
    bad = _fixture_rows()
    for row in bad:
        if row["partition"] == "train" and row["label"] == 0:
            row["partition"] = "final_test"
    with pytest.raises(ValueError, match="partition support"):
        mod.fit_registered_units(bad, steps=1)


def test_scenario_7413_02_paired_scoring_preserves_unscored_rows() -> None:
    """SCENARIO-AUTO-7413-02 emits every official row for every arm and seed."""

    rows = _fixture_rows()
    units = mod.fit_registered_units(rows, steps=2)
    paired, receipt = mod.score_official_rows(units, rows)
    assert len(paired) == 9 * 4 * 5
    identities = {(row["row_key"], row["arm"], row["seed"]) for row in paired}
    assert len(identities) == len(paired)
    unscored = [row for row in paired if row["label"] is None]
    assert len(unscored) == 4 * 5
    assert all(row["brier_contribution"] is None for row in unscored)
    assert receipt == {
        "official_rows": 9,
        "eligible_scored_rows": 8,
        "unscored_rows": 1,
        "arm_seed_units": 20,
        "paired_rows": 180,
    }


def test_scenario_7413_03_metrics_report_both_classes_and_decisions() -> None:
    """SCENARIO-AUTO-7413-03 recomputes proper, ranking, and action metrics."""

    paired = []
    for seed in mod.TRAINING_SEEDS:
        for index, (label, probability) in enumerate(((0, 0.1), (1, 0.9), (0, 0.2), (1, 0.8))):
            paired.append(
                mod.metric_row_for_test(
                    arm="source_aware_6_4_1_gibbs",
                    seed=seed,
                    row_key=f"r{index}",
                    group_id=f"g{index}",
                    label=label,
                    raw_probability=0.5,
                    probability=probability,
                    decision="escalate",
                )
            )
    metrics = mod.reduce_metrics(paired)
    aggregate = metrics["by_arm"]["source_aware_6_4_1_gibbs"]
    assert aggregate["brier"] == pytest.approx(0.025)
    assert aggregate["auroc"] == 1.0
    assert aggregate["incorrect_pr_auc"] == 1.0
    assert aggregate["correct_pr_auc"] == 1.0
    assert aggregate["macro_f1"] == 1.0
    assert aggregate["decision_changes"]["raw_to_calibrated_binary"] == 10


def test_scenario_7413_03_simultaneous_bootstrap_detects_source_signal() -> None:
    """SCENARIO-AUTO-7413-03 averages seeds then bootstraps connected groups."""

    rows = []
    controls = {
        "training_prevalence": 0.4,
        "l2_logistic_six_input": 0.3,
        "response_only_2_4_1_gibbs": 0.2,
        "source_aware_6_4_1_gibbs": 0.05,
    }
    for group_index in range(8):
        label = group_index % 2
        for arm, error in controls.items():
            probability = error if label == 0 else 1.0 - error
            for seed in mod.TRAINING_SEEDS:
                rows.append(
                    mod.metric_row_for_test(
                        arm=arm,
                        seed=seed,
                        row_key=f"r{group_index}",
                        group_id=f"g{group_index}",
                        label=label,
                        raw_probability=probability,
                        probability=probability,
                        decision="escalate",
                    )
                )
    intervals = mod.paired_group_bootstrap(rows, draws=200, seed=6501307)
    assert intervals["draws"] == 200
    assert intervals["seed"] == 6501307
    assert intervals["seed_average_before_group_resampling"] is True
    assert all(
        row["brier_delta"]["simultaneous_ci95"][1] < 0 for row in intervals["contrasts"].values()
    )


def test_scenario_7413_04_small_support_disables_actions() -> None:
    """SCENARIO-AUTO-7413-04 converts uncertified threshold actions to escalation."""

    unit = mod.fit_registered_units(_fixture_rows(), steps=1)[0]
    policy = unit["selected_policy"]
    assert policy["accept_enabled"] is False
    assert policy["reject_enabled"] is False
    decision = mod.certified_decision(0.0, policy, "fixture")
    assert decision["decision"] == "escalate"
    assert "uncertified" in decision["reason"]


def test_scenario_7413_05_source_ablation_rules_are_fixed() -> None:
    """SCENARIO-AUTO-7413-05 removes or permutes source without reading labels."""

    row = _row("train", 0, 1)
    removed = mod.source_removed_row(row)
    assert list(removed["source_features"].values()) == [0.0, 0.85, 0.0, 0.0, 0.0, 1.0]
    assert removed["label"] == 1

    predictors = [
        {
            "row_key": "a",
            "group_id": "a",
            "partition": "train",
            "question": "q",
            "answer": "The count is 1.",
            "sentence": "The count is 1.",
            "context": "The count is 1.",
            "label": 0,
        },
        {
            "row_key": "b",
            "group_id": "b",
            "partition": "train",
            "question": "q",
            "answer": "The count is 1.",
            "sentence": "The count is 1.",
            "context": "The count is 2.",
            "label": 1,
        },
    ]
    permuted = mod.cross_group_source_permutation(predictors)
    assert [row["row_key"] for row in permuted] == ["a", "b"]
    assert permuted[0]["source_features"]["normalized_number_token_overlap"] == 0.0
    assert permuted[1]["source_features"]["normalized_number_token_overlap"] == 1.0
    assert [row["label"] for row in permuted] == [0, 1]
    with pytest.raises(ValueError, match="two groups"):
        mod.cross_group_source_permutation(predictors[:1])


def test_scenario_7413_06_value_gate_requires_every_registered_check() -> None:
    """SCENARIO-AUTO-7413-06 keeps complete capture separate from benefit."""

    good_intervals = {
        "contrasts": {
            baseline: {
                "brier_delta": {"simultaneous_ci95": [-0.2, -0.01]},
                "coverage_delta": {"ci95": [0.0, 0.1]},
            }
            for baseline in mod.BASELINE_ARMS
        }
    }
    metrics = {
        "by_arm": {
            arm: {"log_loss": 0.2 if arm == mod.PRIMARY_ARM else 0.3, "coverage": 0.3}
            for arm in mod.ARMS
        }
    }
    result = mod.reduce_calibration_value(good_intervals, metrics)
    assert result["passed"] is True and result["calibration_value_score"] == 1
    changed = deepcopy(good_intervals)
    changed["contrasts"]["training_prevalence"]["brier_delta"]["simultaneous_ci95"][1] = 0.0
    result = mod.reduce_calibration_value(changed, metrics)
    assert result["passed"] is False and result["calibration_value_score"] == 0


def test_scenario_7413_07_artifact_validation_detects_mutations() -> None:
    """SCENARIO-AUTO-7413-07 independently rejects row and gate drift."""

    artifact = mod.build_fixture_artifact()
    assert mod.validate_artifact(artifact) == []
    assert artifact["calibration_capture_complete_score"] == 1
    assert artifact["calibration_value_score"] == 0
    assert artifact["verdict_class"] == "null"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["label_authority"] == "machine_annotation"
    changed = deepcopy(artifact)
    changed["promotion_score"] = 1
    changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
    assert "declaration_mismatch:promotion_score" in mod.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["paired_metric_rows"].pop()
    changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
    assert "paired_row_budget_mismatch" in mod.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["calibration_value_score"] = 1
    changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
    assert "calibration_value_score_mismatch" in mod.validate_artifact(changed)


def test_blocked_artifact_and_cli_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-AUTO-7413 keeps external absence blocked and the wrapper thin."""

    failed = {
        "check": "upstream_ready",
        "upstream": mod.UPSTREAM_PATH.as_posix(),
        "path": mod.UPSTREAM_PATH.as_posix(),
        "field": "source_feature_protocol_ready_score",
        "operator": "==",
        "expected": 1,
        "observed": None,
        "passed": False,
    }
    blocked = mod.build_blocked_artifact(failed)
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"]["blocked_observed"] is None
    assert mod.validate_artifact(blocked) == []

    args = mod.parse_args(["--date", mod.RUN_DATE])
    assert args.date == mod.RUN_DATE
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(mod.build_fixture_artifact()), encoding="utf-8")
    monkeypatch.setattr(mod, "cold_replay", lambda _path: [])
    assert mod.main(["--date", mod.RUN_DATE, "--cold-replay", str(candidate)]) == 0
    monkeypatch.setattr(mod, "cold_replay", lambda _path: ["drift"])
    assert mod.main(["--date", mod.RUN_DATE, "--cold-replay", str(candidate)]) == 1
    calls: list[tuple[Path, str, Path]] = []
    monkeypatch.setattr(
        mod,
        "run_experiment",
        lambda root, date, output_path: calls.append((root, date, output_path)),
    )
    output = tmp_path / "out.json"
    assert mod.main(["--date", mod.RUN_DATE, "--output", str(output)]) == 0
    assert calls == [(mod.REPO_ROOT, mod.RUN_DATE, output)]


def test_numeric_metrics_reject_missing_classes_and_invalid_probabilities() -> None:
    """REQ-AUTO-7413 fails closed instead of inventing undefined ranking metrics."""

    with pytest.raises(ValueError, match="both labels"):
        mod.binary_auroc([1], [0.5])
    with pytest.raises(ValueError, match="both labels"):
        mod.binary_pr_auc([0], [0.5], positive_label=1)
    assert math.isfinite(mod.log_loss_contribution(1, 1.0))
    with pytest.raises(ValueError, match="probability"):
        mod.log_loss_contribution(0, float("nan"))


def test_defensive_helpers_and_progress_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-AUTO-7413 covers malformed inputs and observable loop checkpoints."""

    mod.progress(0.0, "fixture", "boundary", completed=1)
    assert "phase=fixture" in capsys.readouterr().out
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert mod._load_object(malformed) == {}
    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    assert mod._load_object(array) == {}

    row = _row("train", 0, 0)
    with pytest.raises(ValueError, match="frozen width"):
        mod._finite_vector([0.0, float("nan")], 2)
    changed = deepcopy(row)
    changed["source_features"] = {}
    with pytest.raises(ValueError, match="source feature keys"):
        mod.feature_vector(changed, mod.PRIMARY_ARM)
    changed = deepcopy(row)
    changed["response_only_ablation"] = {}
    with pytest.raises(ValueError, match="response feature keys"):
        mod.feature_vector(changed, mod.PRIMARY_ARM)
    changed = deepcopy(row)
    changed["response_only_ablation"]["teacher_score"] = 0.1
    with pytest.raises(ValueError, match="prohibited"):
        mod.feature_vector(changed, mod.PRIMARY_ARM)
    with pytest.raises(ValueError, match="identity"):
        mod._label_blind_representatives([{"group_id": "", "row_key": ""}])
    with pytest.raises(ValueError, match="policy labels"):
        mod._select_policy(
            mod.fit_registered_units(_fixture_rows(), steps=1)[0],
            [row for row in _fixture_rows() if row["partition"] != "policy_calibration"],
        )
    with pytest.raises(ValueError, match="registered"):
        mod.fit_registered_units(_fixture_rows(), condition="unknown", steps=1)

    units = mod.fit_registered_units(_fixture_rows(), steps=0, emit_progress=True)
    paired, _ = mod.score_official_rows(units, _fixture_rows(), emit_progress=True)
    assert paired
    output = capsys.readouterr().out
    assert "small_ebm_training" in output and "official_scoring" in output
    reject = mod.certified_decision(
        1.0,
        {
            "accept_threshold": 0.01,
            "reject_threshold": 0.9,
            "accept_enabled": False,
            "reject_enabled": False,
        },
        "fixture",
    )
    assert reject["decision"] == "escalate" and "reject_action_uncertified" in reject["reason"]


def test_reducer_and_ablation_error_paths() -> None:
    """SCENARIO-AUTO-7413-03/05 fail closed on incomplete paired evidence."""

    with pytest.raises(ValueError, match="scored rows"):
        mod._unit_metrics([])
    with pytest.raises(ValueError, match="positive"):
        mod.paired_group_bootstrap([], draws=0)
    with pytest.raises(ValueError, match="scored groups"):
        mod.paired_group_bootstrap([], draws=1)
    rows = []
    for arm in mod.ARMS[:-1]:
        rows.append(
            mod.metric_row_for_test(
                arm=arm,
                seed=mod.TRAINING_SEEDS[0],
                row_key="r",
                group_id="g",
                label=0,
                raw_probability=0.1,
                probability=0.1,
                decision="escalate",
            )
        )
    with pytest.raises(ValueError, match="arm coverage"):
        mod.paired_group_bootstrap(rows, draws=1)
    changed = _row("train", 0, 0)
    changed["response_only_ablation"] = {}
    with pytest.raises(ValueError, match="response feature keys"):
        mod.source_removed_row(changed)


def test_checkpoint_validation_and_cold_reader_defenses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-AUTO-7413-07 binds checkpoint bytes and raw row reductions."""

    units = mod.fit_registered_units(_fixture_rows(), steps=1)[:1]
    monkeypatch.setattr(mod, "RAW_DIR", Path("raw"))
    manifest = mod.write_checkpoints(tmp_path, units)
    assert manifest[0]["byte_size"] > 0
    assert manifest[0]["sha256"] == mod.sha256_file(tmp_path / manifest[0]["path"])
    assert mod._validation_passed(_passing_receipts()) is True
    assert mod._validation_passed([]) is False
    blocked = mod.build_blocked_artifact(
        {
            "check": "missing",
            "upstream": "u",
            "path": "p",
            "field": "f",
            "expected": 1,
            "observed": None,
        }
    )
    assert mod.independent_reduce(blocked)["capture"] == 0

    artifact = mod.build_fixture_artifact()
    changed = deepcopy(artifact)
    changed["verdict_class"] = "unknown"
    changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
    assert "verdict_class_invalid" in mod.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["field_principles"].pop("schema")
    changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
    assert "field_principles_mismatch" in mod.validate_artifact(changed)
    changed = deepcopy(blocked)
    changed["calibration_capture_complete_score"] = 1
    changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
    assert "blocked_disposition_invalid" in mod.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["source_ablation_rows"].pop()
    changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
    assert "ablation_row_budget_mismatch" in mod.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["paired_metric_rows"][0]["brier_contribution"] = 999.0
    changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
    errors = mod.validate_artifact(changed)
    assert "brier_contribution_mismatch:0" in errors
    assert "calibration_metrics_mismatch" in errors
    changed = deepcopy(artifact)
    changed["paired_metric_rows"][0]["log_loss_contribution"] = 999.0
    changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
    assert "log_loss_contribution_mismatch:0" in mod.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["calibration_value_reduction"] = {}
    changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
    assert "calibration_value_reduction_mismatch" in mod.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:wrong"
    assert "reproducibility_checksum_mismatch" in mod.validate_artifact(changed)

    candidate = tmp_path / "candidate.json"
    mod.atomic_json(candidate, artifact)
    assert mod.cold_replay(candidate) == []
    assert mod.cold_replay(tmp_path / "missing.json") == ["artifact_unreadable_or_not_object"]
    span = mod._span("fixture", 0.0, 0.0, 1)
    assert span["completed_units"] == 1

    nonfixture = deepcopy(artifact)
    nonfixture["fixture_artifact"] = False
    nonfixture["source_artifact_hashes"] = {
        "invalid": "not-a-row",
        "missing": {"path": "missing.file", "sha256": "sha256:absent"},
    }
    nonfixture["reproducibility_checksum"] = mod.artifact_checksum(nonfixture)
    nonfixture_errors = mod.validate_artifact(nonfixture)
    assert any(error.startswith("checkpoint_bytes_mismatch:") for error in nonfixture_errors)
    assert "source_artifact_hash_row_invalid" in nonfixture_errors
    assert "source_artifact_hash_mismatch:missing.file" in nonfixture_errors


def test_row_integrity_and_independent_cli_defenses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-AUTO-7413-07 rejects duplicate, malformed, and invented rows."""

    row = mod.metric_row_for_test(
        arm=mod.PRIMARY_ARM,
        seed=mod.TRAINING_SEEDS[0],
        row_key="r",
        group_id="g",
        label=1,
        raw_probability=0.5,
        probability=0.8,
        decision="escalate",
    )
    duplicate_errors = mod._row_integrity_errors([row, row])
    assert "duplicate_metric_row:1" in duplicate_errors
    changed = deepcopy(row)
    changed["probability"] = float("nan")
    assert "probability_invalid:0" in mod._row_integrity_errors([changed])
    changed = deepcopy(row)
    changed["label"] = None
    assert "scored_label_invalid:0" in mod._row_integrity_errors([changed])
    changed = deepcopy(row)
    changed["scored"] = False
    assert "unscored_row_invented_metric:0" in mod._row_integrity_errors([changed])

    candidate = tmp_path / "candidate.json"
    mod.atomic_json(candidate, mod.build_fixture_artifact())
    assert mod.main(["--date", mod.RUN_DATE, "--independent-reduce", str(candidate)]) == 0
    malformed = tmp_path / "bad.json"
    malformed.write_text("{", encoding="utf-8")
    assert mod.main(["--date", mod.RUN_DATE, "--independent-reduce", str(malformed)]) == 1
    monkeypatch.setattr(mod, "run_experiment", lambda *_args, **_kwargs: {})
    assert mod.main(["--date", mod.RUN_DATE]) == 0
