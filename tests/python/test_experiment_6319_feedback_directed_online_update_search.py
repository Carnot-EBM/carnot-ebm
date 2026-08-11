"""Tests for Exp6319 feedback-directed online update search.

Spec refs: REQ-CSL-6319, REQ-CSL-6319-PROTECTED-SEAL,
REQ-CSL-6319-DENSE-SIGNAL, REQ-CSL-6319-MATCHED-ARMS,
REQ-CSL-6319-READY, REQ-CSL-6319-PROVENANCE,
SCENARIO-CSL-6319-PROTECTED-LEAKAGE,
SCENARIO-CSL-6319-BUDGET-PARITY,
SCENARIO-CSL-6319-ONE-TIME-OPEN,
SCENARIO-CSL-6319-SIGNAL-TAMPERING,
SCENARIO-CSL-6319-DETERMINISTIC-REPLAY.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_6319_feedback_directed_online_update_search as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _artifact(tmp_path: Path, *, write: bool = True) -> dict[str, object]:
    return mod.run(
        date="20260811",
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        duration_s=1.0,
        test_exit_codes=_passing_exit_codes(),
        write=write,
    )


def _read_json(receipt: dict[str, object]) -> dict[str, object]:
    return json.loads(Path(str(receipt["path"])).read_text(encoding="utf-8"))


def _refresh(artifact: dict[str, object]) -> dict[str, object]:
    mod.refresh_terminal_fields(artifact)
    return artifact


def test_req_csl_6319_spec_declares_artifact_contract() -> None:
    """REQ-CSL-6319-PROVENANCE: OpenSpec owns fields and scenarios."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-CSL-6319") :]

    for token in (
        "REQ-CSL-6319-PROTECTED-SEAL",
        "REQ-CSL-6319-DENSE-SIGNAL",
        "REQ-CSL-6319-MATCHED-ARMS",
        "REQ-CSL-6319-READY",
        "SCENARIO-CSL-6319-PROTECTED-LEAKAGE",
        "SCENARIO-CSL-6319-BUDGET-PARITY",
        "SCENARIO-CSL-6319-ONE-TIME-OPEN",
        "SCENARIO-CSL-6319-SIGNAL-TAMPERING",
        "SCENARIO-CSL-6319-DETERMINISTIC-REPLAY",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        *mod.SEARCH_ARMS,
    ):
        assert token in section
    normalized = " ".join(section.split())
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_csl_6319_protected_leakage_and_one_time_open(
    tmp_path: Path,
) -> None:
    """SCENARIO-CSL-6319-PROTECTED-LEAKAGE: search sees no protected labels."""

    artifact = _artifact(tmp_path)
    protected_manifest = _read_json(artifact["protected_validation_manifest_path_and_hash"])
    development = artifact["development_progress_by_candidate_and_arm"]
    lineage = artifact["candidate_lineage_and_intervention_receipts"]
    access_log = artifact["protected_partition_seal_and_access_log"]

    assert artifact["upstream_path_hash_and_terminal_class"]["ready_score"] == 1.0
    assert protected_manifest["target_states_hidden_from_manifest"] is True
    assert all("target_state" not in row for row in protected_manifest["events"])
    assert access_log["sealed_before_search"] is True
    assert access_log["open_count"] == 1
    assert access_log["opened_after_both_searches_terminated"] is True
    assert access_log["protected_feedback_after_open"] is False
    assert artifact["protected_validation_reuse_count"] == 0
    assert type(artifact["protected_validation_reuse_count"]) is int
    assert artifact["progress_signal_release_authority_count"] == 0
    assert type(artifact["progress_signal_release_authority_count"]) is int
    assert artifact["source_model_weight_mutation_count"] == 0
    assert type(artifact["source_model_weight_mutation_count"]) is int
    assert all(row["protected_target_visible_before_search_stop"] is False for row in lineage)
    assert all(row["selected_before_candidate_execution"] is True for row in lineage)
    assert all(row["protected_exact_visible"] is False for row in development["rows"])
    assert artifact["dense_progress_signal_definition_and_cost"]["uses_protected_validation"] is False
    assert artifact["dense_progress_signal_definition_and_cost"]["release_authority"] == "none"


def test_scenario_csl_6319_budget_parity_and_ready(tmp_path: Path) -> None:
    """SCENARIO-CSL-6319-BUDGET-PARITY: arms share all search budgets."""

    artifact = _artifact(tmp_path)
    budgets = artifact["matched_candidate_update_verifier_time_and_movement_budgets"]
    outcomes = artifact["validated_improvements_false_discoveries_and_regressions_by_arm"]
    per_cost = artifact["validated_improvements_per_cost_by_arm"]
    predictiveness = artifact["signal_predictiveness_intervals_and_sample_sizes"]

    repeated = budgets[mod.REPEATED_SAMPLING_ARM]
    directed = budgets[mod.FEEDBACK_DIRECTED_ARM]
    for key in (
        "candidate_count",
        "update_operation_count",
        "development_exact_verifier_call_count",
        "wall_time_ceiling_s",
        "movement_budget_ceiling",
        "candidate_pool_hash",
    ):
        assert repeated[key] == directed[key]
    assert budgets["parity_passed"] is True
    assert predictiveness["protected_improvement_correlation"]["mean_delta"] > 0.0
    assert predictiveness["protected_improvement_correlation"]["n"] >= 2
    assert outcomes[mod.FEEDBACK_DIRECTED_ARM]["validated_improvement_count"] > outcomes[
        mod.REPEATED_SAMPLING_ARM
    ]["validated_improvement_count"]
    assert outcomes[mod.FEEDBACK_DIRECTED_ARM]["protected_regression_count"] <= outcomes[
        mod.REPEATED_SAMPLING_ARM
    ]["protected_regression_count"]
    assert outcomes[mod.FEEDBACK_DIRECTED_ARM]["false_discovery_count"] <= outcomes[
        mod.REPEATED_SAMPLING_ARM
    ]["false_discovery_count"]
    assert per_cost[mod.FEEDBACK_DIRECTED_ARM]["improvements_per_cost"] > per_cost[
        mod.REPEATED_SAMPLING_ARM
    ]["improvements_per_cost"]
    assert artifact["feedback_directed_search_ready_score"] == 1.0
    assert artifact["status"] == "complete_positive"
    assert artifact["honest_verdict"].startswith("complete_positive:")


def test_scenario_csl_6319_cli_schema_and_tamper_failures(tmp_path: Path) -> None:
    """SCENARIO-CSL-6319-SIGNAL-TAMPERING: readiness fails on signal abuse."""

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    assert mod.main(["--date", "20260811", "--output", str(output), "--validate"]) == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"])
    assert artifact["verifier_is_oracle"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) is None

    missing = dict(artifact)
    missing.pop("field_principles")
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(missing)

    bad_zero = json.loads(json.dumps(artifact))
    bad_zero["protected_validation_reuse_count"] = True
    _refresh(bad_zero)
    with pytest.raises(ValueError, match="protected_validation_reuse_count"):
        mod.validate_artifact(bad_zero)

    leaked = json.loads(json.dumps(artifact))
    leaked["dense_progress_signal_definition_and_cost"]["uses_protected_validation"] = True
    _refresh(leaked)
    assert leaked["feedback_directed_search_ready_score"] == 0.0

    release_abuse = json.loads(json.dumps(artifact))
    release_abuse["progress_signal_release_authority_count"] = 1
    _refresh(release_abuse)
    assert release_abuse["feedback_directed_search_ready_score"] == 0.0

    reopened = json.loads(json.dumps(artifact))
    reopened["protected_partition_seal_and_access_log"]["open_count"] = 2
    _refresh(reopened)
    assert reopened["feedback_directed_search_ready_score"] == 0.0

    bad_status = json.loads(json.dumps(leaked))
    bad_status["status"] = "complete_positive"
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(bad_status)

    bad_checksum = dict(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)


def test_scenario_csl_6319_deterministic_replay_and_helper_paths(
    tmp_path: Path,
) -> None:
    """SCENARIO-CSL-6319-DETERMINISTIC-REPLAY: checksums are stable."""

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    first = mod.run(
        date="20260811",
        result_path=output,
        duration_s=1.0,
        test_exit_codes=_passing_exit_codes(),
        write=False,
    )
    second = mod.run(
        date="20260811",
        result_path=output,
        duration_s=3.0,
        test_exit_codes=_passing_exit_codes(),
        write=False,
    )
    assert first["reproducibility_checksum"] == second["reproducibility_checksum"]
    assert mod._paired_interval([]) == {"n": 0, "mean_delta": 0.0, "lower": 0.0, "upper": 0.0}
    assert mod._paired_interval([1.0]) == {
        "n": 1,
        "mean_delta": 1.0,
        "lower": 1.0,
        "upper": 1.0,
    }
    assert mod._paired_interval([0.0, 1.0])["n"] == 2
    assert mod._pearson([], []) == 0.0
    assert mod._pearson([1.0, 1.0], [0.0, 1.0]) == 0.0
    assert mod._path_receipt(tmp_path / "missing.json")["present"] is False
    with pytest.raises(ValueError, match="forced"):
        mod._require(False, "forced")

    no_tests = json.loads(json.dumps(first))
    no_tests["test_exit_codes"] = {mod.DEFAULT_TEST_COMMANDS[0]: 1}
    _refresh(no_tests)
    assert no_tests["feedback_directed_search_ready_score"] == 0.0

    for field in (
        "upstream_path_hash_and_terminal_class",
        "structured_gate_receipt",
        "protected_partition_seal_and_access_log",
        "matched_candidate_update_verifier_time_and_movement_budgets",
        "signal_predictiveness_intervals_and_sample_sizes",
        "validated_improvements_false_discoveries_and_regressions_by_arm",
        "validated_improvements_per_cost_by_arm",
        "movement_memory_and_wall_time_by_arm",
        "test_exit_codes",
        "protected_files_unchanged",
    ):
        malformed = json.loads(json.dumps(first))
        malformed[field] = []
        assert mod.ready_score(malformed) == 0.0
