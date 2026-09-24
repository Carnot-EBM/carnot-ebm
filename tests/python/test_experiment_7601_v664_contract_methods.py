"""Tests for the V664 contract and method advisory.

Spec refs: REQ-REPORT-7601 and SCENARIO-REPORT-7601-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_7601_v664_contract_methods as subject


ROOT = Path(__file__).resolve().parents[2]


def _roadmap() -> dict[str, object]:
    """Load the selected V664 authority without changing it."""

    value = yaml.safe_load((ROOT / subject.ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _passing_receipts() -> list[dict[str, object]]:
    """Build complete synthetic command custody for pure artifact tests."""

    return [
        {
            "name": name,
            "command_argv": ["unit", name],
            "worktree": str(ROOT),
            "exit_code": 0,
            "passed": True,
            "timed_out": False,
            "log_path": f"/tmp/{name}.log",
            "log_sha256": "sha256:" + "1" * 64,
        }
        for name in (
            *subject.REQUIRED_CHECK_NAMES,
            *subject.REPOSITORY_CHECK_NAMES,
            *subject.TERMINAL_CHECK_NAMES,
        )
    ]


def test_authority_resolution_accepts_staged_and_consumed_staging(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7601-AUTHORITY accepts both lifecycle states."""

    active = tmp_path / subject.ACTIVE_ROADMAP_PATH
    active.write_text((ROOT / subject.ACTIVE_ROADMAP_PATH).read_text(), encoding="utf-8")
    selected, value, candidates = subject.resolve_v664_roadmap(tmp_path)
    assert selected == active
    assert value["milestone"] == subject.MILESTONE
    assert candidates[0]["observed"] == "absent"

    staged = tmp_path / subject.NEXT_ROADMAP_PATH
    staged.write_text(active.read_text(encoding="utf-8"), encoding="utf-8")
    assert subject.resolve_v664_roadmap(tmp_path)[0] == staged

    changed = yaml.safe_load(staged.read_text(encoding="utf-8"))
    changed["milestone"] = "2026.09.663"
    staged.write_text(yaml.safe_dump(changed), encoding="utf-8")
    assert subject.resolve_v664_roadmap(tmp_path)[0] == active

    active.unlink()
    with pytest.raises(ValueError, match="V664 roadmap authority"):
        subject.resolve_v664_roadmap(tmp_path)


def test_exact_contract_and_all_private_mutations() -> None:
    """SCENARIO-REPORT-7601-AUTHORITY checks every conjunctive field."""

    design = (ROOT / subject.DESIGN_PATH).read_text(encoding="utf-8")
    comparison = subject.compare_contract_authorities(design, _roadmap())
    assert comparison["passed"] is True
    assert [row["unit_id"] for row in comparison["rows"]] == list(subject.EXPECTED_TASK_IDS)
    assert all(row["raw_numerator"] == row["raw_denominator"] == 7 for row in comparison["rows"])

    controls = subject.run_contract_mutation_controls(design, _roadmap())
    assert len(controls) == 9
    assert {row["mutation"] for row in controls} == {
        "removed_row",
        "reordered_row",
        "changed_path",
        "misspelled_field",
        "consumed_staging",
    }
    assert all(row["qualified"] is True for row in controls)


@pytest.mark.parametrize(
    "mutation", ("removed_row", "reordered_row", "changed_path", "misspelled_field")
)
def test_each_yaml_mutation_fails(mutation: str) -> None:
    """REQ-REPORT-7601 rejects each private YAML authority mutation."""

    design = (ROOT / subject.DESIGN_PATH).read_text(encoding="utf-8")
    changed = subject.mutate_yaml_for_test(_roadmap(), mutation)
    assert subject.compare_contract_authorities(design, changed)["passed"] is False

    with pytest.raises(ValueError, match="unknown mutation"):
        subject.mutate_yaml_for_test(_roadmap(), "unknown")


def test_v663_dispositions_preserve_eight_one_five_boundary() -> None:
    """SCENARIO-REPORT-7601-CUSTODY preserves all prior outcomes literally."""

    rows = subject.collect_v663_dispositions(ROOT)
    assert [row["task_id"] for row in rows] == list(subject.V663_TASK_IDS)
    assert all(row["authenticated"] is True for row in rows)
    counts = subject.disposition_kind_counts(rows)
    assert counts == {"actual_producer": 8, "conductor_pre_gate": 1, "absent_producer": 5}
    by_id = {row["task_id"]: row for row in rows}
    assert by_id["exp7590-evidence-pilot"]["evidence_kind"] == "conductor_pre_gate"
    assert by_id["exp7591-fit-evidence"]["evidence_kind"] == "absent_producer"
    assert by_id["exp7598-rust-consumer"]["honest_verdict"].endswith("speed_gate_failed")
    assert by_id["exp7600-capstone"]["evidence_kind"] == "actual_producer"


def test_selector_check_is_only_a_changed_readiness_hypothesis() -> None:
    """SCENARIO-REPORT-7601-CUSTODY does not correct the blocked protocol."""

    row = subject.planning_selector_hypothesis(ROOT)
    assert row["failed_preconditions"] == []
    assert row["source_group_count"] == 480
    assert row["scored_group_count"] == 240
    assert row["pilot_group_count"] == 8
    assert row["classification"] == "hypothesis_about_changed_executable_readiness"
    assert row["historical_artifact_corrected"] is False
    assert row["scientific_result"] is False


def test_method_rows_and_records_preserve_claim_limits(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7601-METHODS maps primary methods and deferred leads."""

    rows = subject.method_rows()
    by_family = {row["method_family"]: row for row in rows}
    assert {
        "eaev_evidence_alignment",
        "on_chip_kan_locality",
        "proper_calibeating",
        "u_calibration",
        "spintronic_ising_future_lead",
        "sparse_transformer_future_lead",
    } == set(by_family)
    assert "not the U-Calibration algorithm" in by_family["u_calibration"]["claim_limit"]
    assert by_family["spintronic_ising_future_lead"]["status"] == "future_lead"
    assert by_family["sparse_transformer_future_lead"]["status"] == "future_lead"
    assert all(row["external_result_is_local_measurement"] is False for row in rows)

    subject.write_method_records(tmp_path)
    first = (tmp_path / subject.NOTE_PATH).read_text(encoding="utf-8")
    subject.write_method_records(tmp_path)
    assert (tmp_path / subject.NOTE_PATH).read_text(encoding="utf-8") == first
    studying = (tmp_path / subject.STUDY_PATH).read_text(encoding="utf-8")
    assert studying.count(subject.STUDY_MARKER) == 1
    assert "guarded delayed learner" in studying
    assert "future leads" in studying


def test_artifact_is_complete_null_advisory() -> None:
    """REQ-REPORT-7601 keeps administrative readiness separate from benefit."""

    artifact = subject.build_artifact_for_test(validation_receipts=_passing_receipts())
    assert artifact["honest_verdict"] == "complete_null_v664_contract_methods_ingested"
    assert artifact["verdict_class"] == "null"
    assert artifact["contract_ready_score"] == 1
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert all(value == 0 for value in artifact["invocation_counts"].values())
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["planned_inference_substrate_class"] == "aggregation"
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["random_seed"] == 7601
    assert len(artifact["rows"]) == 14
    assert len(artifact["v663_dispositions"]) == 14
    assert artifact["v663_disposition_counts"] == {
        "actual_producer": 8,
        "conductor_pre_gate": 1,
        "absent_producer": 5,
    }
    assert {row["category"] for row in artifact["acceptance_gate_results"]} == {
        "validity",
        "readiness",
        "benefit",
        "retention",
        "freshness",
    }
    assert artifact["capability_e2e"]["numbered_runtime_e2e_applicable"] is False
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
    """REQ-REPORT-7601 rejects protected terminal-field mutations."""

    artifact = subject.build_artifact_for_test(validation_receipts=_passing_receipts())
    artifact[field] = replacement
    assert error in subject.validate_artifact(artifact, root=ROOT, require_terminal=False)


def test_rows_and_custody_fail_independent_reduction() -> None:
    """SCENARIO-REPORT-7601-CUSTODY binds comparison and source rows."""

    artifact = subject.build_artifact_for_test(validation_receipts=_passing_receipts())
    artifact["rows"][0]["observed"]["deliverable"] = "changed.json"
    assert subject.independent_reduce(artifact)["passed"] is False

    artifact = subject.build_artifact_for_test(validation_receipts=_passing_receipts())
    artifact["v663_dispositions"][3]["evidence_kind"] = "actual_producer"
    assert subject.independent_reduce(artifact)["passed"] is False


def test_blocked_artifact_names_every_required_operand() -> None:
    """REQ-REPORT-7601 emits complete blocked evidence for absent inputs."""

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
    assert {"check", "upstream", "path", "field", "operator", "expected", "observed"} <= set(
        failure
    )
    assert subject.validate_artifact(artifact, root=ROOT, require_terminal=False) == []


def test_validation_plans_and_receipts_are_scoped(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7601-VALIDATION freezes private bounded commands."""

    manifest = subject.affected_file_manifest()
    assert manifest["tests"] == [subject.TEST_PATH.as_posix()]
    assert manifest["modules"] == [subject.MODULE_PATH.as_posix()]
    commands = subject.build_validation_plan(ROOT, tmp_path / "private")
    assert subject.validate_validation_plan(ROOT, commands) == []
    assert [row.name for row in commands] == list(subject.REQUIRED_CHECK_NAMES)
    assert all(parent.is_dir() for parent in subject.private_basetemp_parents(commands))

    guards = subject.build_repository_check_plan(ROOT, ROOT / subject.ACTIVE_ROADMAP_PATH)
    assert [row.name for row in guards] == list(subject.REPOSITORY_CHECK_NAMES)
    receipts = _passing_receipts()
    assert subject.reduce_validation_receipts(receipts)["passed"] is True
    for key in ("command_argv", "worktree", "exit_code", "log_sha256"):
        changed = deepcopy(receipts)
        changed[0].pop(key)
        assert subject.reduce_validation_receipts(changed)["passed"] is False


def test_operator_state_replay_modes_and_thin_wrapper(tmp_path: Path) -> None:
    """REQ-REPORT-7601 preserves E0/E6 and bounded replay surfaces."""

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

    wrapper = ROOT / subject.WRAPPER_PATH
    if wrapper.exists():
        text = wrapper.read_text(encoding="utf-8")
        assert "experiment_7601_v664_contract_methods" in text
        assert len(text.splitlines()) <= 20


def test_malformed_authorities_and_capstone_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-7601 rejects malformed authority and custody shapes."""

    comparison = subject.compare_contract_authorities("bad", None)
    assert comparison["passed"] is False
    assert "markdown_task_table_missing" in comparison["errors"]
    assert "yaml_task_table_missing" in comparison["errors"]

    capstone = tmp_path / subject.V663_CAPSTONE_PATH
    capstone.parent.mkdir(parents=True)
    capstone.write_text(json.dumps({"task_dispositions": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="dispositions are unavailable"):
        subject.collect_v663_dispositions(tmp_path)
    capstone.write_text(
        json.dumps({"task_dispositions": [{} for _ in subject.V663_TASK_IDS]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="order is invalid"):
        subject.collect_v663_dispositions(tmp_path)


def test_cold_validator_reports_protected_boundaries() -> None:
    """REQ-REPORT-7601 reports malformed terminal evidence explicitly."""

    assert subject.validate_artifact([], root=ROOT) == ["artifact_mapping_required"]
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

    artifact = subject.build_artifact_for_test(validation_receipts=[])
    assert "terminal_validation_incomplete" in subject.validate_artifact(artifact, root=ROOT)


def test_terminal_plan_prerequisites_and_valid_date(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7601-VALIDATION binds readers and named inputs."""

    candidate = tmp_path / "candidate.json"
    commands = subject._terminal_commands(ROOT, candidate)  # noqa: SLF001
    assert [row.name for row in commands] == list(subject.TERMINAL_CHECK_NAMES)
    assert "--strict" in commands[-1].argv
    required = subject._required_inputs(ROOT)  # noqa: SLF001
    assert required and all(expected == observed for _, _, expected, observed in required)
    assert subject.date_argument(subject.RUN_DATE) == subject.RUN_DATE


def test_selector_failure_stays_a_non_scientific_hypothesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-7601-CUSTODY preserves failed planning observations."""

    monkeypatch.setattr(
        subject.protocol,
        "collect_preconditions",
        lambda _root: ([{"check": "source_exists", "passed": False}], [], {"fit": []}),
    )
    row = subject.planning_selector_hypothesis(ROOT)
    assert row["failed_preconditions"] == ["source_exists"]
    assert row["scored_group_count"] == row["pilot_group_count"] == 0
    assert row["historical_artifact_corrected"] is False


def test_validation_plan_rejects_each_scope_drift(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7601-VALIDATION rejects broad or unowned commands."""

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


def test_source_hash_reader_rejects_malformed_present_and_changed_absence(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7601-CUSTODY authenticates both bytes and absence."""

    assert subject._source_hashes_match({}, tmp_path) is False  # noqa: SLF001
    assert (
        subject._source_hashes_match(  # noqa: SLF001
            {"source_artifact_hashes": [1]}, tmp_path
        )
        is False
    )
    missing = tmp_path / "missing.json"
    absent = {"source_artifact_hashes": [{"path": str(missing), "exists": False, "sha256": None}]}
    assert subject._source_hashes_match(absent, tmp_path) is True  # noqa: SLF001
    missing.write_text("changed", encoding="utf-8")
    assert subject._source_hashes_match(absent, tmp_path) is False  # noqa: SLF001
    wrong = {
        "source_artifact_hashes": [{"path": str(missing), "exists": True, "sha256": "sha256:wrong"}]
    }
    assert subject._source_hashes_match(wrong, tmp_path) is False  # noqa: SLF001


def test_blocked_summary_and_terminal_receipt_comparison_fail_closed() -> None:
    """REQ-REPORT-7601 rejects incomplete blockers and changed terminal logs."""

    blocked = subject.build_blocked_artifact(
        upstream="u", path="p", field="f", expected=1, observed=0
    )
    blocked["gate_check_summary"]["first_failure"] = {}
    blocked["reproducibility_checksum"] = subject.reproducibility_checksum(blocked)
    assert "blocked_gate_summary_incomplete" in subject.validate_artifact(
        blocked, root=ROOT, require_terminal=False
    )

    left = [
        {"name": "reader", "command_argv": ["x"], "exit_code": 0, "passed": True, "log_sha256": "h"}
    ]
    assert subject._terminal_receipts_equal(left, deepcopy(left)) is True  # noqa: SLF001
    changed = deepcopy(left)
    changed[0]["log_sha256"] = "changed"
    assert subject._terminal_receipts_equal(left, changed) is False  # noqa: SLF001
    assert subject._terminal_receipts_equal(left, []) is False  # noqa: SLF001
