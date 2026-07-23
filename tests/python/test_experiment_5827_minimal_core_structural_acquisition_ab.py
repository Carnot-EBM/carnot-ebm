"""Tests for Exp5827 minimal-core structural acquisition.

Spec refs: REQ-LEARN-5827, SCENARIO-LEARN-5827-ACTIVE-CORE,
SCENARIO-LEARN-5827-MATCHED-ARMS, SCENARIO-LEARN-5827-READY-GATE,
SCENARIO-LEARN-5827-FAIL-CLOSED.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5827_minimal_core_structural_acquisition_ab as mod


REPO = Path(__file__).resolve().parents[2]
SELF_LEARNING_SPEC = REPO / "openspec/capabilities/self-learning/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5827_minimal_core_structural_acquisition_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5827_minimal_core_structural_acquisition_ab.py "
    "-m pytest tests/python/test_experiment_5827_minimal_core_structural_acquisition_ab.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5827_minimal_core_structural_acquisition_ab.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5827_minimal_core_structural_acquisition_ab.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
TEST_COMMANDS = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
]
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


def _preconditions(tmp_path: Path) -> dict[str, Any]:
    return mod.collect_preconditions(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        checkpoint_dir=tmp_path / "checkpoints",
        memory_probe=lambda: {"available_mb": 8192, "required_mb": 512, "ok": True},
        disk_probe=lambda root: {"available_mb": 8192, "required_mb": 512, "ok": True},
    )


def _walk(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return list(value.keys()) + [item for sub in value.values() for item in _walk(sub)]
    if isinstance(value, list):
        return [item for sub in value for item in _walk(sub)]
    return [value]


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """REQ-LEARN-5827: build the deterministic structural learner artifact once."""

    base = tmp_path_factory.mktemp("exp5827")
    return mod.run(
        result_path=base / mod.RESULT_RELATIVE_PATH.name,
        checkpoint_dir=base / "checkpoints",
        preconditions_checked=_preconditions(base),
        duration_s=1.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )


def test_req_learn_5827_spec_declares_required_contract() -> None:
    """REQ-LEARN-5827: OpenSpec names the artifact fields and principles."""

    text = SELF_LEARNING_SPEC.read_text(encoding="utf-8")
    section = text[text.index("## REQ-LEARN-5827") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5827",
        "SCENARIO-LEARN-5827-ACTIVE-CORE",
        "SCENARIO-LEARN-5827-MATCHED-ARMS",
        "SCENARIO-LEARN-5827-READY-GATE",
        "SCENARIO-LEARN-5827-FAIL-CLOSED",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`structural_learner_ready_score`",
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_learn_5827_active_core_artifact_is_terminal_and_replayable(
    artifact: dict[str, Any],
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-5827-ACTIVE-CORE: artifact is complete and hash-bound."""

    destination = tmp_path / mod.RESULT_RELATIVE_PATH.name
    replay = mod.run(
        result_path=destination,
        checkpoint_dir=tmp_path / "ckpt",
        preconditions_checked=_preconditions(tmp_path),
        duration_s=1.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )
    loaded = json.loads(destination.read_text(encoding="utf-8"))

    assert artifact == replay == loaded
    assert mod.validate_artifact(artifact) is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["structural_learner_ready_score"] == pytest.approx(1.0)
    assert isinstance(artifact["structural_learner_ready_score"], float)
    assert artifact["preconditions_checked"]["preconditions_ready"] is True
    assert artifact["preconditions_checked"]["row_replay"]["row_count"] == 360
    assert artifact["preconditions_checked"]["headroom_witnesses"]["headroom_present_row_count"] == 324
    assert artifact["preconditions_checked"]["structured_gate_replay"]["ok"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["oracle_boundary_violation_count"] == 0
    assert artifact["test_commands"] == TEST_COMMANDS
    assert artifact["test_exit_codes"] == TEST_EXIT_CODES
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)


def test_scenario_learn_5827_matched_arms_have_budget_parity_and_no_label_leakage(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-LEARN-5827-MATCHED-ARMS: deployable arms share evidence budgets."""

    arms = artifact["arm_definitions_and_budget_parity"]
    assert set(arms["arms"]) == set(mod.CONTROL_ARMS)
    assert arms["science_labels_assigned_after_arm_freeze"] is True
    assert arms["budget_parity_passed"] is True
    assert arms["upper_bound_arm"] == mod.UPPER_BOUND_ARM
    assert arms["definitions"][mod.UPPER_BOUND_ARM]["deployable"] is False

    deployable = [arm for arm in mod.CONTROL_ARMS if arm != mod.UPPER_BOUND_ARM]
    budgets = {arms["definitions"][arm]["query_budget_per_row"] for arm in deployable}
    operation_hashes = {arms["definitions"][arm]["candidate_operations_hash"] for arm in deployable}
    stopping_rules = {arms["definitions"][arm]["stopping_rule"] for arm in deployable}
    assert budgets == {mod.QUERY_BUDGET_PER_ROW}
    assert len(operation_hashes) == 1
    assert stopping_rules == {mod.STOPPING_RULE}

    receipts = artifact["query_and_minimal_core_receipts"]
    assert receipts["all_receipts_hash"].startswith("sha256:")
    assert receipts["oracle_boundary"] == "exact_membership_outcome_only"
    assert receipts["deployable_label_leakage_count"] == 0
    assert receipts["active"]["query_count"] <= (
        artifact["structural_recovery_and_headroom"]["headroom_present_row_count"]
        * mod.QUERY_BUDGET_PER_ROW
    )
    leak_tokens = {
        "ground_truth_structure",
        "target_structure",
        "target_structure_seal",
        "future_label",
        "exact_structure_upper_bound",
    }
    active_sample_values = [str(item) for item in _walk(receipts["active"]["sample_receipts"])]
    assert not leak_tokens.intersection(active_sample_values)
    assert all(
        receipt["oracle_boundary"] == "exact_membership_outcome_only"
        for receipt in receipts["active"]["sample_receipts"]
    )


def test_req_learn_5827_structural_grammar_strictly_exceeds_exp5762_overlap(
    artifact: dict[str, Any],
) -> None:
    """REQ-LEARN-5827: grammar records overlap and out-of-template expressivity."""

    grammar = artifact["structural_hypothesis_grammar"]
    assert grammar["frozen_before_replay"] is True
    assert grammar["strictly_exceeds_exp5762_library"] is True
    assert grammar["signature_overlap_with_exp5762"]["overlap_count"] > 0
    assert grammar["signature_overlap_with_exp5762"]["new_signature_count"] >= 4
    assert grammar["max_arity"] >= 3
    assert {"hard_forbid", "hard_require", "soft_penalty", "soft_preference"}.issubset(
        set(grammar["role_operations"])
    )
    assert {"forall", "exists", "count_eq", "sequence_exists"}.issubset(
        set(grammar["quantification"])
    )
    target_relations = {
        row["relation"] for row in grammar["out_of_template_signatures"]
    }
    assert target_relations == {
        "cyclic_order",
        "cardinality_eq",
        "weighted_sum_lte",
        "forbidden_subsequence",
    }
    assert all(count > 0 for count in grammar["candidate_hypothesis_count_by_family"].values())


def test_scenario_learn_5827_ready_gate_uses_disaggregated_positive_bounds(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-LEARN-5827-READY-GATE: credit follows headroom and CI gates."""

    recovery = artifact["structural_recovery_and_headroom"]
    paired = artifact["paired_deltas_and_ci95"]
    safety = artifact["protected_prefix_and_safety"]
    metrics = artifact["per_arm_family_change_metrics"]

    assert recovery["out_of_template_row_count"] == 360
    assert recovery["headroom_present_row_count"] == 324
    assert recovery["credited_family_count"] == 4
    assert recovery["families_with_positive_lcb"] == list(mod.PRIMARY_FAMILIES)
    assert recovery["precision_floor"] == pytest.approx(0.95)
    assert recovery["active_precision"] >= 0.95
    assert recovery["protected_prefix_regression_count"] == 0
    assert paired["pooled"]["heterogeneity_check"]["pooled_reporting_allowed"] is True
    assert paired["pooled"]["active_minus_exp5762_template"]["ci95"][0] > 0.0
    assert paired["pooled"]["active_minus_exp5762_template"]["mean_delta"] > 0.0
    assert safety["protected_prefix_regression_count"] == 0
    assert safety["unsafe_propagation_count"] == 0

    for family in mod.PRIMARY_FAMILIES:
        assert paired["family"][family]["active_minus_exp5762_template"]["ci95"][0] > 0.0
        for change in mod.CHANGE_ORDER:
            active = metrics[mod.ACTIVE_ARM][family][change]
            baseline = metrics[mod.TEMPLATE_BASELINE_ARM][family][change]
            assert active["exact_behavioral_recovery"] == pytest.approx(1.0)
            assert active["constraint_precision"] >= 0.95
            assert active["wrong_structure_acceptance_rate"] == pytest.approx(0.0)
            assert active["complexity"]["mean_predicate_count"] >= 1.0
            assert active["exact_behavioral_recovery"] > baseline["exact_behavioral_recovery"]


def test_scenario_learn_5827_fail_closed_for_missing_inputs_exits_and_leakage(
    artifact: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-LEARN-5827-FAIL-CLOSED: bad evidence cannot look ready."""

    blocked = mod.run(
        root=tmp_path,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        checkpoint_dir=tmp_path / "checkpoints",
        duration_s=1.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )
    assert blocked["status"] == "blocked"
    assert blocked["structural_learner_ready_score"] == 0.0
    assert blocked["honest_verdict"].startswith("blocked:")
    assert "missing_upstream_artifact" in blocked["preconditions_checked"]["blocked_reasons"]
    assert mod.validate_artifact(blocked) is True

    failed_exits = mod.build_artifact(
        preconditions_checked=_preconditions(tmp_path / "failed"),
        duration_s=1.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes={**TEST_EXIT_CODES, FULL_TEST_COMMAND: 2},
    )
    assert failed_exits["status"] == "blocked"
    assert failed_exits["structural_learner_ready_score"] == 0.0
    assert "failed_test_exit_codes" in mod.blocked_reasons(failed_exits)

    for mutate, match in (
        (lambda item: item.update({"inference_substrate": "live_llm"}), "inference_substrate"),
        (lambda item: item.update({"verifier_is_oracle": False}), "verifier_is_oracle"),
        (lambda item: item.update({"oracle_boundary_violation_count": 1}), "structural_learner_ready_score"),
        (
            lambda item: item["query_and_minimal_core_receipts"].update(
                {"deployable_label_leakage_count": 1}
            ),
            "structural_learner_ready_score",
        ),
        (
            lambda item: item["protected_prefix_and_safety"].update(
                {"protected_prefix_regression_count": 1}
            ),
            "structural_learner_ready_score",
        ),
        (lambda item: item.update({"reproducibility_checksum": mod.sha256_text("wrong")}), "reproducibility_checksum"),
    ):
        bad = deepcopy(artifact)
        mutate(bad)
        with pytest.raises(ValueError, match=match):
            mod.validate_artifact(bad)

    monkeypatch.setattr(
        mod,
        "_read_json",
        lambda path: (_ for _ in ()).throw(ValueError("corrupt")),
    )
    corrupt = mod.collect_preconditions(
        result_path=tmp_path / "corrupt.json",
        checkpoint_dir=tmp_path / "corrupt-ckpt",
        memory_probe=lambda: {"available_mb": 0, "required_mb": 512, "ok": False},
        disk_probe=lambda root: {"available_mb": 0, "required_mb": 512, "ok": False},
    )
    assert set(corrupt["blocked_reasons"]) >= {
        "corrupt_upstream_artifact",
        "insufficient_free_ram",
        "insufficient_free_disk",
    }


def test_req_learn_5827_low_level_helpers_and_validation_edges(
    artifact: dict[str, Any],
    tmp_path: Path,
) -> None:
    """REQ-LEARN-5827: helper edge cases fail closed and remain deterministic."""

    scalar_json = tmp_path / "scalar.json"
    scalar_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        mod._read_json(scalar_json)

    blank_jsonl = tmp_path / "blank.jsonl"
    blank_jsonl.write_text('\n{"ok": true}\n', encoding="utf-8")
    scalar_jsonl = tmp_path / "scalar.jsonl"
    scalar_jsonl.write_text("1\n", encoding="utf-8")
    assert mod._read_jsonl(blank_jsonl) == [{"ok": True}]
    with pytest.raises(ValueError, match="JSONL object required"):
        mod._read_jsonl(scalar_jsonl)
    assert mod.read_row_file(tmp_path / "missing.rows.jsonl") == []
    assert mod.fixture_preconditions()["preconditions_ready"] is True

    assert mod._bootstrap_ci95([]) == [0.0, 0.0]
    assert mod._bootstrap_ci95([0.25]) == [0.25, 0.25]
    assert mod._heterogeneity_check({"a": [0.1, 0.2], "b": [0.11, 0.19]})[
        "pooled_reporting_allowed"
    ] is True
    assert mod._heterogeneity_check({"a": [0.0], "b": [1.0]})[
        "pooled_reporting_allowed"
    ] is False

    rows = mod.read_row_file(REPO / mod.EXP5826_ROWS_RELATIVE_PATH)
    headroom = next(row for row in rows if mod._headroom_present(row))
    candidates = mod._candidate_domain(headroom)
    hypotheses = mod._hypothesis_space(str(headroom["family"]))
    assert candidates
    assert hypotheses
    assert any(mod._hypothesis_accepts(hypotheses[0], candidate["assignment"]) in {True, False} for candidate in candidates)

    stream_artifact = mod._read_json(REPO / mod.EXP5826_ARTIFACT_RELATIVE_PATH)
    tampered_rows = deepcopy(rows)
    tampered_rows[0]["row_hash"] = mod.sha256_text("tamper")
    replay_receipt = mod._row_replay_receipt(
        tampered_rows,
        stream_artifact,
        REPO / mod.EXP5826_ROWS_RELATIVE_PATH,
    )
    assert replay_receipt["replay_ok"] is False

    missing = deepcopy(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        mod.validate_artifact(missing)

    blocked_reasons_probe = deepcopy(artifact)
    blocked_reasons_probe["inference_substrate"] = "wrong"
    blocked_reasons_probe["verifier_is_oracle"] = False
    blocked_reasons_probe["oracle_boundary_violation_count"] = 1
    blocked_reasons_probe["query_and_minimal_core_receipts"]["deployable_label_leakage_count"] = 1
    blocked_reasons_probe["protected_prefix_and_safety"]["protected_prefix_regression_count"] = 1
    blocked_reasons_probe["protected_prefix_and_safety"]["unsafe_propagation_count"] = 1
    assert set(mod.blocked_reasons(blocked_reasons_probe)) >= {
        "inference_substrate",
        "verifier_is_oracle",
        "oracle_boundary_violation_count",
        "deployable_label_leakage_count",
        "protected_prefix_regression_count",
        "unsafe_propagation_count",
    }

    generic_score_probe = deepcopy(artifact)
    generic_score_probe["arm_definitions_and_budget_parity"]["budget_parity_passed"] = False
    assert "structural_learner_ready_score" in mod.blocked_reasons(generic_score_probe)

    invalid_provenance_shape = deepcopy(artifact)
    invalid_provenance_shape["field_provenance"] = []
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(invalid_provenance_shape)

    invalid_provenance_principle = deepcopy(artifact)
    invalid_provenance_principle["field_provenance"]["status"]["principle"] = "wrong"
    with pytest.raises(ValueError, match="field_provenance:status"):
        mod.validate_artifact(invalid_provenance_principle)

    invalid_status = deepcopy(artifact)
    invalid_status["status"] = "blocked"
    invalid_status["reproducibility_checksum"] = mod.reproducibility_checksum(invalid_status)
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(invalid_status)

    invalid_complete_verdict = deepcopy(artifact)
    invalid_complete_verdict["honest_verdict"] = "blocked: forced"
    invalid_complete_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(
        invalid_complete_verdict
    )
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(invalid_complete_verdict)

    invalid_blocked_verdict = mod.build_artifact(
        preconditions_checked=_preconditions(tmp_path / "blocked-verdict"),
        duration_s=1.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes={**TEST_EXIT_CODES, FULL_TEST_COMMAND: 2},
    )
    invalid_blocked_verdict["honest_verdict"] = "ready"
    invalid_blocked_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(
        invalid_blocked_verdict
    )
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(invalid_blocked_verdict)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = mod.sha256_text("wrong")
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)

    no_write = mod.run(
        result_path=tmp_path / "no-write.json",
        checkpoint_dir=tmp_path / "no-write-ckpt",
        preconditions_checked=_preconditions(tmp_path / "no-write"),
        duration_s=1.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=False,
    )
    assert no_write["status"] == "complete"
    assert not (tmp_path / "no-write.json").exists()
