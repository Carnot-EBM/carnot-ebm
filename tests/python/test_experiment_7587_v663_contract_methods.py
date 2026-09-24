"""Tests for the V663 contract and method advisory.

Spec refs: REQ-REPORT-7587 and SCENARIO-REPORT-7587-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_7587_v663_contract_methods as subject


ROOT = Path(__file__).resolve().parents[2]


def _roadmap() -> dict[str, object]:
    """Load the real active V663 authority used by every contract test."""

    value = yaml.safe_load((ROOT / subject.ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _passing_receipts() -> list[dict[str, object]]:
    """Build complete synthetic custody without running repository commands."""

    return [
        {
            "name": name,
            "command": ["unit", name],
            "worktree": str(ROOT),
            "exit_code": 0,
            "passed": True,
            "log_path": f"unit/{name}.log",
            "log_sha256": "sha256:" + "1" * 64,
        }
        for name in (
            *subject.REQUIRED_CHECK_NAMES,
            *subject.REPOSITORY_CHECK_NAMES,
            *subject.TERMINAL_CHECK_NAMES,
        )
    ]


def test_authority_resolution_before_and_after_activation(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7587-AUTHORITY accepts staged and consumed states."""

    active = tmp_path / subject.ACTIVE_ROADMAP_PATH
    active.write_text((ROOT / subject.ACTIVE_ROADMAP_PATH).read_text(), encoding="utf-8")
    selected, value, candidates = subject.resolve_v663_roadmap(tmp_path)
    assert selected == active
    assert value["milestone"] == subject.MILESTONE
    assert candidates[0]["observed"] == "absent"

    staged = tmp_path / subject.NEXT_ROADMAP_PATH
    staged.write_text(active.read_text(encoding="utf-8"), encoding="utf-8")
    assert subject.resolve_v663_roadmap(tmp_path)[0] == staged

    changed = yaml.safe_load(staged.read_text(encoding="utf-8"))
    changed["milestone"] = "2026.09.662"
    staged.write_text(yaml.safe_dump(changed), encoding="utf-8")
    assert subject.resolve_v663_roadmap(tmp_path)[0] == active

    active.unlink()
    with pytest.raises(ValueError, match="V663 roadmap authority"):
        subject.resolve_v663_roadmap(tmp_path)


def test_exact_fourteen_task_contract_and_private_mutations() -> None:
    """SCENARIO-REPORT-7587-AUTHORITY compares every conjunctive field."""

    design = (ROOT / subject.DESIGN_PATH).read_text(encoding="utf-8")
    comparison = subject.compare_contract_authorities(design, _roadmap())
    assert comparison["passed"] is True
    assert [row["unit_id"] for row in comparison["rows"]] == list(subject.EXPECTED_TASK_IDS)
    assert all(row["raw_numerator"] == row["raw_denominator"] == 7 for row in comparison["rows"])

    controls = subject.run_contract_mutation_controls(design, _roadmap())
    assert len(controls) == 8
    assert {row["mutation"] for row in controls} == {"count", "order", "path", "field"}
    assert all(row["qualified"] is True for row in controls)

    with pytest.raises(ValueError, match="unknown mutation"):
        subject.mutate_yaml_for_test(_roadmap(), "unknown")


@pytest.mark.parametrize("mutation", ("count", "order", "path", "field"))
def test_each_yaml_mutation_fails(mutation: str) -> None:
    """REQ-REPORT-7587 rejects each private YAML authority mutation."""

    design = (ROOT / subject.DESIGN_PATH).read_text(encoding="utf-8")
    changed = subject.mutate_yaml_for_test(_roadmap(), mutation)
    assert subject.compare_contract_authorities(design, changed)["passed"] is False


def test_v662_dispositions_preserve_exact_boundaries() -> None:
    """SCENARIO-REPORT-7587-CUSTODY preserves all fourteen V662 outcomes."""

    rows = subject.collect_v662_dispositions(ROOT)
    assert [row["task_id"] for row in rows] == list(subject.V662_TASK_IDS)
    assert all(row["authenticated"] is True for row in rows)
    by_id = {row["task_id"]: row for row in rows}
    assert by_id["exp7579-decision-learning-audit"]["static_group_count"] == 80
    assert by_id["exp7579-decision-learning-audit"]["static_brier"] == 0.146249673
    assert by_id["exp7579-decision-learning-audit"]["raw_brier"] == 0.145635636
    assert by_id["exp7578-continuous-proper-loss"]["retention_passed"] is False
    assert by_id["exp7581-arc-bounded-canary"]["inference_started"] is False
    assert by_id["exp7582-arc-panel-a"]["evidence_kind"] == "conductor_pre_gate"
    assert by_id["exp7583-arc-panel-b"]["evidence_kind"] == "missing_producer"
    assert by_id["exp7585-portable-service"]["warm_ratio"] == pytest.approx(33.59268279121724)
    assert by_id["exp7585-portable-service"]["warm_lower95"] == pytest.approx(31.122338001012775)
    assert by_id["exp7585-portable-service"]["execution_venue"] == "host"


def test_method_rows_and_records_preserve_claim_limits(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7587-METHODS maps primary sections to bounded uses."""

    rows = subject.method_rows()
    assert {row["method_family"] for row in rows} == {
        "eaev_evidence_alignment",
        "rt4chart_hierarchical_verification",
        "u_calibration",
        "on_chip_kan_locality",
    }
    limits = " ".join(str(row["claim_limit"]) for row in rows)
    assert "not an EAEV replication" in limits
    assert "not the U-Calibration algorithm" in limits
    assert all(row["external_result_is_local_measurement"] is False for row in rows)

    subject.write_method_records(tmp_path)
    first = (tmp_path / subject.NOTE_PATH).read_text(encoding="utf-8")
    subject.write_method_records(tmp_path)
    assert (tmp_path / subject.NOTE_PATH).read_text(encoding="utf-8") == first
    studying = (tmp_path / subject.STUDY_PATH).read_text(encoding="utf-8")
    assert studying.count(subject.STUDY_MARKER) == 1
    assert "Applicable" in studying and "Deferred" in studying


def test_lifecycle_rejects_duplicate_after_durable_reload(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7587-VALIDATION exercises legal delayed feedback."""

    row = subject.run_lifecycle_control(tmp_path)
    assert row["events"] == ["predict", "release", "update", "persist", "reload", "duplicate"]
    assert row["prediction_before_release"] is True
    assert row["reload_equal"] is True
    assert row["duplicate_rejected"] is True
    assert row["passed"] is True


def test_artifact_is_complete_null_advisory(tmp_path: Path) -> None:
    """REQ-REPORT-7587 keeps contract readiness separate from benefit."""

    artifact = subject.build_artifact_for_test(
        validation_receipts=_passing_receipts(), lifecycle=subject.run_lifecycle_control(tmp_path)
    )
    assert artifact["honest_verdict"] == "complete_null_v663_contract_methods_ingested"
    assert artifact["verdict_class"] == "null"
    assert artifact["contract_ready_score"] == 1
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert all(value == 0 for value in artifact["invocation_counts"].values())
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["planned_inference_substrate_class"] == "aggregation"
    assert artifact["random_seed"] == 7587
    assert len(artifact["rows"]) == 14
    assert len(artifact["v662_dispositions"]) == 14
    assert {row["category"] for row in artifact["acceptance_gate_results"]} == {
        "validity",
        "readiness",
        "benefit",
        "retention",
        "freshness",
    }
    assert artifact["sample_size_budget"]["observed"] == 14
    assert subject.independent_reduce(artifact)["passed"] is True
    assert subject.validate_artifact(artifact, root=ROOT, require_terminal=False) == []


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        ("contract_ready_score", True, "contract_ready_score_bare_integer_required"),
        ("honest_verdict", "success", "terminal_verdict_prefix_required"),
        ("verdict_class", "success", "verdict_class_invalid"),
        ("MODEL_SPECS", ["model"], "model_specs_must_be_empty"),
        ("flagged_adversarial", True, "independent_reduction_failed"),
        ("reproducibility_checksum", "sha256:" + "0" * 64, "checksum_mismatch"),
    ],
)
def test_artifact_mutations_fail_closed(field: str, replacement: object, error: str) -> None:
    """REQ-REPORT-7587 rejects protected terminal-field mutations."""

    artifact = subject.build_artifact_for_test(validation_receipts=_passing_receipts())
    artifact[field] = replacement
    assert error in subject.validate_artifact(artifact, root=ROOT, require_terminal=False)


def test_rows_and_custody_fail_independent_reduction() -> None:
    """SCENARIO-REPORT-7587-CUSTODY binds comparison and disposition rows."""

    artifact = subject.build_artifact_for_test(validation_receipts=_passing_receipts())
    artifact["rows"][0]["observed"]["deliverable"] = "changed.json"
    assert subject.independent_reduce(artifact)["passed"] is False

    artifact = subject.build_artifact_for_test(validation_receipts=_passing_receipts())
    artifact["v662_dispositions"][10]["evidence_kind"] = "producer"
    assert subject.independent_reduce(artifact)["passed"] is False


def test_blocked_artifact_names_every_required_operand() -> None:
    """REQ-REPORT-7587 emits a complete blocked result for missing input."""

    artifact = subject.build_blocked_artifact(
        upstream="research-roadmap.yaml",
        path="research-roadmap.yaml",
        field="milestone",
        expected=subject.MILESTONE,
        observed="absent",
    )
    failure = artifact["gate_check_summary"]["first_failure"]
    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["verdict_class"] == "blocked"
    assert set(("check", "upstream", "path", "field", "op", "expected", "observed")) <= set(failure)


def test_validation_plans_and_receipts_are_scoped(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7587-VALIDATION freezes paths and private storage."""

    assert subject.affected_file_manifest() == {
        "tests": [subject.TEST_PATH.as_posix()],
        "modules": [subject.MODULE_PATH.as_posix()],
        "static": [subject.WRAPPER_PATH.as_posix(), subject.SPEC_PATH.as_posix()],
    }
    commands = subject.build_validation_plan(ROOT, tmp_path / "private")
    assert subject.validate_validation_plan(ROOT, commands) == []
    assert [row.name for row in commands] == list(subject.REQUIRED_CHECK_NAMES)
    assert all(parent.is_dir() for parent in subject.private_basetemp_parents(commands))

    guards = subject.build_repository_check_plan(ROOT, ROOT / subject.ACTIVE_ROADMAP_PATH)
    assert [row.name for row in guards] == list(subject.REPOSITORY_CHECK_NAMES)
    receipts = _passing_receipts()
    assert subject.reduce_validation_receipts(receipts)["passed"] is True
    for key in ("command", "worktree", "exit_code", "log_sha256"):
        changed = deepcopy(receipts)
        changed[0].pop(key)
        assert subject.reduce_validation_receipts(changed)["passed"] is False


def test_operator_state_cli_and_thin_wrapper(tmp_path: Path) -> None:
    """REQ-REPORT-7587 preserves E0/E6 and exposes bounded replay modes."""

    queue = subject.operator_queue(ROOT)
    assert queue["E0"]["status"] == "operator_blocked"
    assert queue["E6"]["status"] == "resolved"
    with pytest.raises(SystemExit):
        subject.parse_args(["--date", "20260923"])

    artifact = subject.build_artifact_for_test(validation_receipts=_passing_receipts())
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(artifact), encoding="utf-8")
    assert subject.main(["--root", str(ROOT), "--cold-validate", str(candidate)]) == 0
    assert subject.main(["--root", str(ROOT), "--independent-reduce", str(candidate)]) == 0
    assert subject.main(["--root", str(ROOT), "--lifecycle-replay", str(tmp_path / "life")]) == 0

    wrapper = ROOT / subject.WRAPPER_PATH
    if wrapper.exists():
        text = wrapper.read_text(encoding="utf-8")
        assert "experiment_7587_v663_contract_methods" in text
        assert len(text.splitlines()) <= 20


def test_malformed_contract_and_disposition_shapes_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-7587 rejects malformed authorities and capstone rows."""

    comparison = subject.compare_contract_authorities("bad", None)
    assert comparison["passed"] is False
    assert "markdown_task_table_missing" in comparison["errors"]
    assert "yaml_task_table_missing" in comparison["errors"]

    capstone = tmp_path / subject.V662_CAPSTONE_PATH
    capstone.parent.mkdir(parents=True)
    capstone.write_text(json.dumps({"task_dispositions": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="dispositions are unavailable"):
        subject.collect_v662_dispositions(tmp_path)
    capstone.write_text(
        json.dumps({"task_dispositions": [{} for _ in subject.V662_TASK_IDS]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="order is invalid"):
        subject.collect_v662_dispositions(tmp_path)


def test_validation_plan_drift_is_rejected(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7587-VALIDATION rejects every scope expansion."""

    commands = subject.build_validation_plan(ROOT, tmp_path / "private")
    assert "validation_command_names_changed" in subject.validate_validation_plan(ROOT, [])

    changed = list(commands)
    original = changed[1]
    changed[1] = subject.CommandSpec(original.name, ("pytest",), original.scope, original.timeout_s)
    assert "pytest_target_changed" in subject.validate_validation_plan(ROOT, changed)

    for parent in set(subject.private_basetemp_parents(commands)):
        parent.rmdir()
    assert "private_basetemp_parent_missing" in subject.validate_validation_plan(ROOT, commands)

    changed = list(commands)
    original = changed[0]
    changed[0] = subject.CommandSpec(
        original.name,
        (*original.argv, str(ROOT / "tests/python")),
        original.scope,
        original.timeout_s,
    )
    assert "unscoped_test_directory_forbidden" in subject.validate_validation_plan(ROOT, changed)


def test_cold_validator_and_source_hash_failures() -> None:
    """REQ-REPORT-7587 cold validation reports every protected boundary."""

    assert subject.validate_artifact([], root=ROOT) == ["artifact_mapping_required"]
    assert subject._source_hashes_match({}, ROOT) is False  # noqa: SLF001
    assert subject._source_hashes_match({"source_artifact_hashes": [1]}, ROOT) is False  # noqa: SLF001
    malformed_hash = {"source_artifact_hashes": [{"path": 1, "sha256": None}]}
    assert subject._source_hashes_match(malformed_hash, ROOT) is False  # noqa: SLF001
    missing_hash = {"source_artifact_hashes": [{"path": "absent", "sha256": "sha256:x"}]}
    assert subject._source_hashes_match(missing_hash, ROOT) is False  # noqa: SLF001

    artifact = subject.build_artifact_for_test(validation_receipts=_passing_receipts())
    artifact.pop("schema")
    artifact["milestone"] = "wrong"
    artifact["invocation_counts"] = {}
    artifact["inference_substrate_class"] = "live"
    artifact["acceptance_gate_results"] = []
    artifact["field_principles"] = {}
    artifact["method_map_path"] = "wrong"
    artifact["source_artifact_hashes"] = []
    errors = subject.validate_artifact(artifact, root=ROOT, require_terminal=False)
    assert any(error.startswith("required_fields_missing") for error in errors)
    assert "schema_mismatch" in errors
    assert "experiment_identity_mismatch" in errors
    assert "current_model_calls_nonzero" in errors
    assert "substrate_class_invalid" in errors
    assert "acceptance_gate_shape_invalid" in errors
    assert "field_principles_incomplete" in errors
    assert "method_map_path_invalid" in errors
    assert "source_hash_mismatch" in errors

    blocked = subject.build_blocked_artifact(
        upstream="u", path="p", field="f", expected=1, observed=0
    )
    blocked["gate_check_summary"]["first_failure"] = {}
    blocked["reproducibility_checksum"] = subject.reproducibility_checksum(blocked)
    assert "blocked_gate_summary_incomplete" in subject.validate_artifact(
        blocked, root=ROOT, require_terminal=False
    )

    artifact = subject.build_artifact_for_test(validation_receipts=[])
    assert "terminal_validation_incomplete" in subject.validate_artifact(artifact, root=ROOT)


def test_terminal_plan_prerequisites_and_valid_date(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7587-VALIDATION binds terminal commands and prerequisites."""

    commands = subject._terminal_commands(ROOT, tmp_path / "candidate.json", tmp_path / "life")  # noqa: SLF001
    assert [row.name for row in commands] == list(subject.TERMINAL_CHECK_NAMES)
    assert "--strict" in commands[-1].argv
    required = subject._required_inputs(ROOT)  # noqa: SLF001
    assert required and all(expected == observed for _, _, expected, observed in required)
    assert subject.date_argument(subject.RUN_DATE) == subject.RUN_DATE
