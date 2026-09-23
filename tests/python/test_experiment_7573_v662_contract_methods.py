"""Tests for the V662 contract and method advisory.

Spec refs: REQ-REPORT-7573 and SCENARIO-REPORT-7573-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_7573_v662_contract_methods as subject


ROOT = Path(__file__).resolve().parents[2]


def _roadmap() -> dict[str, object]:
    value = yaml.safe_load((ROOT / subject.ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _passing_receipts() -> list[dict[str, object]]:
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
    """SCENARIO-REPORT-7573-AUTHORITY accepts staged and consumed states."""

    active = tmp_path / subject.ACTIVE_ROADMAP_PATH
    active.write_text((ROOT / subject.ACTIVE_ROADMAP_PATH).read_text(), encoding="utf-8")
    selected, value, candidates = subject.resolve_v662_roadmap(tmp_path)
    assert selected == active
    assert value["milestone"] == subject.MILESTONE
    assert candidates[0]["observed"] == "absent"

    staged = tmp_path / subject.NEXT_ROADMAP_PATH
    staged.write_text(active.read_text(encoding="utf-8"), encoding="utf-8")
    selected, _, candidates = subject.resolve_v662_roadmap(tmp_path)
    assert selected == staged
    assert all(row["matches_milestone"] for row in candidates)

    changed = yaml.safe_load(staged.read_text(encoding="utf-8"))
    changed["milestone"] = "2026.09.661"
    staged.write_text(yaml.safe_dump(changed), encoding="utf-8")
    assert subject.resolve_v662_roadmap(tmp_path)[0] == active


def test_missing_authority_fails_without_substitute(tmp_path: Path) -> None:
    """REQ-REPORT-7573 does not synthesize a planning authority."""

    with pytest.raises(ValueError, match="V662 roadmap authority"):
        subject.resolve_v662_roadmap(tmp_path)


def test_exact_fourteen_task_contract_and_private_mutations() -> None:
    """SCENARIO-REPORT-7573-AUTHORITY compares every conjunctive field."""

    design = (ROOT / subject.DESIGN_PATH).read_text(encoding="utf-8")
    comparison = subject.compare_contract_authorities(design, _roadmap())
    assert comparison["passed"] is True
    assert comparison["errors"] == []
    assert [row["unit_id"] for row in comparison["contract_rows"]] == list(
        subject.EXPECTED_TASK_IDS
    )
    assert len(comparison["rows"]) == 14
    assert all(row["matched"] is True for row in comparison["rows"])

    controls = subject.run_contract_mutation_controls(design, _roadmap())
    assert len(controls) == 2 * len(subject.mutation_names())
    assert {row["mutation"] for row in controls} == {"count", "order", "path", "gate"}
    assert all(row["qualified"] is True for row in controls)


@pytest.mark.parametrize("mutation", ("count", "order", "path", "gate"))
def test_each_yaml_mutation_fails(mutation: str) -> None:
    """REQ-REPORT-7573 rejects each private YAML authority mutation."""

    design = (ROOT / subject.DESIGN_PATH).read_text(encoding="utf-8")
    changed = subject.mutate_yaml_for_test(_roadmap(), mutation)
    assert subject.compare_contract_authorities(design, changed)["passed"] is False


def test_v661_dispositions_keep_five_named_boundaries() -> None:
    """SCENARIO-REPORT-7573-CUSTODY preserves the terminal V661 record."""

    rows = subject.collect_v661_dispositions(ROOT)
    assert len(rows) == 13
    assert all(row["authenticated"] is True for row in rows)
    by_id = {row["task_id"]: row for row in rows}
    assert by_id["exp7561-recalibration-prototype"]["strict_row_lint_exit_code"] == 1
    assert by_id["exp7567-source-evaluation"]["verdict_class"] == "null"
    assert by_id["exp7567-source-evaluation"]["fresh_confirmatory_claim_allowed"] is False
    assert by_id["exp7568-continuous-recalibration"]["evidence_kind"] == "pre_gate"
    assert by_id["exp7570-arc-live-lineage"]["flagged_adversarial"] is True
    assert by_id["exp7570-arc-live-lineage"]["inference_started"] is False
    assert by_id["exp7571-portable-calibration"]["board_continuity_complete_score"] == 1
    assert by_id["exp7571-portable-calibration"]["service_measurement_complete_score"] == 0


def test_method_rows_and_records_preserve_access_limits(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7573-METHODS maps four primary sections to bounded tests."""

    rows = subject.method_rows()
    assert {row["method_family"] for row in rows} == {
        "proper_loss_recalibration",
        "local_kan_learning",
        "structured_decoding_semantics",
        "fpga_asic_cost_accounting",
    }
    assert all(row["applicable_test_or_deferral"] for row in rows)
    assert all(row["external_result_is_local_measurement"] is False for row in rows)
    access = subject.secondary_access_rows()
    assert {row["channel"] for row in access} == {"Semantic Scholar", "OpenReview"}
    assert all(row["complete"] is False for row in access)

    subject.write_method_records(tmp_path)
    first = (tmp_path / subject.NOTE_PATH).read_text(encoding="utf-8")
    subject.write_method_records(tmp_path)
    assert (tmp_path / subject.NOTE_PATH).read_text(encoding="utf-8") == first
    study = (tmp_path / subject.STUDY_PATH).read_text(encoding="utf-8")
    assert study.count(subject.STUDY_MARKER) == 1


def test_lifecycle_control_predicts_releases_persists_and_reloads(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7573-VALIDATION exercises the advisory E2E lifecycle."""

    row = subject.run_lifecycle_control(tmp_path)
    assert row["events"] == ["predict", "release", "update", "persist", "reload"]
    assert row["passed"] is True
    assert row["prediction_before_release"] is True
    assert row["reload_equal"] is True


def test_artifact_is_complete_advisory_with_required_fields(tmp_path: Path) -> None:
    """REQ-REPORT-7573 keeps contract readiness separate from benefit."""

    artifact = subject.build_artifact_for_test(
        validation_receipts=_passing_receipts(), lifecycle=subject.run_lifecycle_control(tmp_path)
    )
    assert artifact["honest_verdict"] == "complete_null_v662_contract_methods_ingested"
    assert artifact["verdict_class"] == "null"
    assert artifact["flagged_adversarial"] is False
    assert artifact["contract_ready_score"] == 1
    assert artifact["method_ingestion_complete_score"] == 1
    assert artifact["positive_claim"] is False
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert all(value == 0 for value in artifact["invocation_counts"].values())
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["planned_inference_substrate_class"] == "aggregation"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["completed_archive_ends_at"] == "2026.09.660"
    assert len(artifact["rows"]) == 14
    assert len(artifact["v661_dispositions"]) == 13
    assert artifact["method_map_path"] == subject.NOTE_PATH.as_posix()
    assert {row["category"] for row in artifact["acceptance_gate_results"]} == {
        "validity",
        "readiness",
        "benefit",
    }
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
    """REQ-REPORT-7573 rejects protected terminal-field mutations."""

    artifact = subject.build_artifact_for_test(validation_receipts=_passing_receipts())
    artifact[field] = replacement
    assert error in subject.validate_artifact(artifact, root=ROOT, require_terminal=False)


def test_row_and_disposition_mutations_fail_independent_reduction() -> None:
    """REQ-REPORT-7573 binds comparative rows and V661 custody."""

    artifact = subject.build_artifact_for_test(validation_receipts=_passing_receipts())
    artifact["rows"][0]["observed"]["path"] = "changed.json"
    assert subject.independent_reduce(artifact)["passed"] is False

    artifact = subject.build_artifact_for_test(validation_receipts=_passing_receipts())
    artifact["v661_dispositions"][8]["evidence_kind"] = "producer"
    assert subject.independent_reduce(artifact)["passed"] is False


def test_blocked_artifact_names_every_required_operand() -> None:
    """REQ-REPORT-7573 emits a complete blocked result for missing input."""

    artifact = subject.build_blocked_artifact(
        upstream="research-roadmap.yaml",
        path="research-roadmap.yaml",
        field="milestone",
        expected=subject.MILESTONE,
        observed="absent",
    )
    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["contract_ready_score"] == 0
    assert artifact["gate_check_summary"]["first_failure"] == {
        "check": "required_input",
        "upstream": "research-roadmap.yaml",
        "path": "research-roadmap.yaml",
        "field": "milestone",
        "op": "==",
        "expected": subject.MILESTONE,
        "observed": "absent",
        "passed": False,
    }


def test_validation_manifest_and_plans_are_scoped(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7573-VALIDATION freezes files and private parents."""

    manifest = subject.affected_file_manifest()
    assert manifest == {
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
    selected = str(ROOT / subject.ACTIVE_ROADMAP_PATH)
    assert all(selected in row.argv or row.name == "publication_gate_json" for row in guards)


def test_validation_receipts_bind_command_worktree_exit_and_log() -> None:
    """REQ-REPORT-7573 requires exact current validation custody."""

    receipts = _passing_receipts()
    assert subject.reduce_validation_receipts(receipts)["passed"] is True
    for key in ("command", "worktree", "exit_code", "log_sha256"):
        changed = deepcopy(receipts)
        changed[0].pop(key)
        assert subject.reduce_validation_receipts(changed)["passed"] is False


def test_operator_queue_keeps_e0_blocked_and_e6_resolved() -> None:
    """REQ-REPORT-7573 preserves operator queue boundaries."""

    queue = subject.operator_queue(ROOT)
    assert queue["E0"]["status"] == "operator_blocked"
    assert queue["E6"]["status"] == "resolved"


def test_cli_modes_and_thin_wrapper(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7573-VALIDATION exposes bounded replay modes."""

    with pytest.raises(SystemExit):
        subject.parse_args(["--date", "20260922"])
    artifact = subject.build_artifact_for_test(validation_receipts=_passing_receipts())
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(artifact), encoding="utf-8")
    assert subject.main(["--root", str(ROOT), "--cold-validate", str(candidate)]) == 0
    assert subject.main(["--root", str(ROOT), "--independent-reduce", str(candidate)]) == 0

    wrapper = ROOT / subject.WRAPPER_PATH
    if wrapper.exists():
        text = wrapper.read_text(encoding="utf-8")
        assert "experiment_7573_v662_contract_methods" in text
        assert len(text.splitlines()) <= 20


def test_invalid_shapes_fail_cleanly() -> None:
    """REQ-REPORT-7573 rejects malformed evidence without exceptions."""

    assert subject.validate_artifact([], root=ROOT) == ["artifact_mapping_required"]
    assert subject.compare_contract_authorities("bad", None)["passed"] is False
    with pytest.raises(ValueError, match="unknown mutation"):
        subject.mutate_yaml_for_test(_roadmap(), "unknown")


def test_private_loaders_and_contract_shapes_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-7573 rejects non-mapping authority and evidence shapes."""

    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text("- not-a-mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        subject._load_yaml(yaml_path)  # noqa: SLF001
    json_path = tmp_path / "bad.json"
    json_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        subject._load_json(json_path)  # noqa: SLF001

    with pytest.raises(ValueError, match="task mapping"):
        subject._normalize_yaml_contract({"tasks": [1]})  # noqa: SLF001
    with pytest.raises(ValueError, match="gate list"):
        subject._normalize_yaml_contract(  # noqa: SLF001
            {"tasks": [{"id": "x", "gated_on": [1]}]}
        )


def test_malformed_v661_capstone_fails_closed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7573-CUSTODY rejects missing and reordered dispositions."""

    path = tmp_path / subject.V661_CAPSTONE_PATH
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="dispositions are unavailable"):
        subject.collect_v661_dispositions(tmp_path)
    rows = [{"task_id": "wrong"} for _ in subject.V661_TASK_IDS]
    path.write_text(json.dumps({"task_dispositions": rows}), encoding="utf-8")
    with pytest.raises(ValueError, match="order is invalid"):
        subject.collect_v661_dispositions(tmp_path)


def test_each_validation_plan_drift_is_rejected(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7573-VALIDATION rejects command and directory expansion."""

    commands = subject.build_validation_plan(ROOT, tmp_path / "private")
    changed = list(commands)
    changed[0] = subject.CommandSpec("changed", changed[0].argv, changed[0].scope)
    assert "validation_command_names_changed" in subject.validate_validation_plan(ROOT, changed)

    changed = list(commands)
    changed[1] = subject.CommandSpec(
        changed[1].name,
        tuple(arg for arg in changed[1].argv if arg != subject.TEST_PATH.as_posix()),
        changed[1].scope,
    )
    assert "pytest_target_changed" in subject.validate_validation_plan(ROOT, changed)

    parent = subject.private_basetemp_parents(commands)[0]
    parent.rmdir()
    assert "private_basetemp_parent_missing" in subject.validate_validation_plan(ROOT, commands)

    changed = list(commands)
    changed[0] = subject.CommandSpec(
        changed[0].name, (*changed[0].argv, str(ROOT / "tests/python")), changed[0].scope
    )
    assert "unscoped_test_directory_forbidden" in subject.validate_validation_plan(ROOT, changed)


def test_receipt_reducer_rejects_absence_and_failure() -> None:
    """REQ-REPORT-7573 keeps missing and failed checks distinct."""

    assert subject.reduce_validation_receipts([])["passed"] is False
    receipts = _passing_receipts()
    receipts[0]["passed"] = False
    receipts[0]["exit_code"] = 1
    assert any(
        error.startswith("receipt_failed:")
        for error in subject.reduce_validation_receipts(receipts)["errors"]
    )


def test_source_hash_reader_rejects_malformed_rows(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7573-VALIDATION rejects unbound source claims."""

    assert subject._source_hashes_match({}, ROOT) is False  # noqa: SLF001
    assert (
        subject._source_hashes_match(  # noqa: SLF001
            {"source_artifact_hashes": ["bad"]}, ROOT
        )
        is False
    )
    assert (
        subject._source_hashes_match(  # noqa: SLF001
            {"source_artifact_hashes": [{"path": None, "sha256": None}]}, ROOT
        )
        is False
    )
    assert (
        subject._source_hashes_match(  # noqa: SLF001
            {"source_artifact_hashes": [{"path": str(tmp_path / "missing"), "sha256": "x"}]},
            ROOT,
        )
        is False
    )


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        ("schema", "bad", "schema_mismatch"),
        ("experiment_id", "bad", "experiment_identity_mismatch"),
        ("invocation_counts", {}, "current_model_calls_nonzero"),
        ("inference_substrate_class", "live", "substrate_class_invalid"),
        ("acceptance_gate_results", [], "acceptance_gate_shape_invalid"),
        ("field_principles", {}, "field_principles_incomplete"),
        ("method_map_path", "bad", "method_map_path_invalid"),
        ("source_artifact_hashes", [], "source_hash_mismatch"),
    ],
)
def test_cold_validator_rejects_schema_boundaries(
    field: str, replacement: object, error: str
) -> None:
    """REQ-REPORT-7573 checks each protected schema boundary."""

    artifact = subject.build_artifact_for_test(validation_receipts=_passing_receipts())
    artifact[field] = replacement
    assert error in subject.validate_artifact(artifact, root=ROOT, require_terminal=False)


def test_cold_validator_requires_fields_blocked_summary_and_terminal_receipts() -> None:
    """REQ-REPORT-7573 validates missing, blocked, and terminal receipt shapes."""

    artifact = subject.build_artifact_for_test(validation_receipts=_passing_receipts())
    del artifact["schema"]
    assert any(
        error.startswith("required_fields_missing:")
        for error in subject.validate_artifact(artifact, root=ROOT, require_terminal=False)
    )

    blocked = subject.build_blocked_artifact(
        upstream="x", path="x", field="bytes", expected="present", observed="absent"
    )
    blocked["gate_check_summary"] = {}
    assert "blocked_gate_summary_incomplete" in subject.validate_artifact(
        blocked, root=ROOT, require_terminal=False
    )

    artifact = subject.build_artifact_for_test(validation_receipts=_passing_receipts())
    artifact["validation_receipts"] = []
    artifact["reproducibility_checksum"] = subject.reproducibility_checksum(artifact)
    assert "terminal_validation_incomplete" in subject.validate_artifact(artifact, root=ROOT)


def test_terminal_plan_receipt_normalizer_and_prerequisites(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7573-VALIDATION binds the exact E2E command and root."""

    plan = subject._terminal_commands(  # noqa: SLF001
        ROOT, tmp_path / "candidate.json", tmp_path / "lifecycle"
    )
    assert [row.name for row in plan] == list(subject.TERMINAL_CHECK_NAMES)
    normalized = subject._normalize_receipts(  # noqa: SLF001
        [{"name": "x", "exit_code": 0}], ROOT
    )
    assert normalized[0]["worktree"] == str(ROOT)
    prerequisites = subject._required_inputs(ROOT)  # noqa: SLF001
    assert prerequisites
    assert all(expected == observed for _, _, expected, observed in prerequisites)
    assert subject.date_argument(subject.RUN_DATE) == subject.RUN_DATE
