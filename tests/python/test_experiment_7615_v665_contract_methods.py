"""Tests for the V665 contract and schema-constrained method advisory.

Spec refs: REQ-REPORT-7615 and SCENARIO-REPORT-7615-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_7615_v665_contract_methods as subject


ROOT = Path(__file__).resolve().parents[2]


def _roadmap() -> dict[str, object]:
    """Load the activated V665 authority without changing its bytes."""

    value = yaml.safe_load((ROOT / subject.ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _passing_receipts() -> list[dict[str, object]]:
    """Provide complete synthetic command custody for pure artifact tests."""

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


def test_authority_resolution_handles_staged_and_consumed_staging(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7615-AUTHORITY accepts both authority lifecycle states."""

    active = tmp_path / subject.ACTIVE_ROADMAP_PATH
    active.write_text((ROOT / subject.ACTIVE_ROADMAP_PATH).read_text(), encoding="utf-8")
    selected, value, candidates = subject.resolve_v665_roadmap(tmp_path)
    assert selected == active
    assert value["milestone"] == subject.MILESTONE
    assert candidates[0]["observed"] == "absent"

    staged = tmp_path / subject.NEXT_ROADMAP_PATH
    staged.write_text(active.read_text(encoding="utf-8"), encoding="utf-8")
    assert subject.resolve_v665_roadmap(tmp_path)[0] == staged
    staged.unlink()
    assert subject.resolve_v665_roadmap(tmp_path)[0] == active

    active.unlink()
    with pytest.raises(ValueError, match="V665 roadmap authority"):
        subject.resolve_v665_roadmap(tmp_path)


def test_exact_contract_and_five_required_controls() -> None:
    """SCENARIO-REPORT-7615-AUTHORITY checks every contract field and lifecycle case."""

    design = (ROOT / subject.DESIGN_PATH).read_text(encoding="utf-8")
    comparison = subject.compare_contract_authorities(design, _roadmap())
    assert comparison["passed"] is True
    assert [row["unit_id"] for row in comparison["rows"]] == list(subject.EXPECTED_TASK_IDS)
    assert all(row["raw_numerator"] == row["raw_denominator"] == 7 for row in comparison["rows"])

    controls = subject.run_contract_mutation_controls(design, _roadmap())
    assert {row["mutation"] for row in controls} == {
        "missing_task",
        "reordered_task",
        "altered_path",
        "misspelled_field",
        "consumed_staging",
    }
    assert len(controls) == 9
    assert all(row["qualified"] is True for row in controls)


@pytest.mark.parametrize(
    "mutation", ("missing_task", "reordered_task", "altered_path", "misspelled_field")
)
def test_each_yaml_contract_mutation_fails(mutation: str) -> None:
    """REQ-REPORT-7615 rejects each private YAML authority corruption."""

    design = (ROOT / subject.DESIGN_PATH).read_text(encoding="utf-8")
    changed = subject.mutate_yaml_for_test(_roadmap(), mutation)
    assert subject.compare_contract_authorities(design, changed)["passed"] is False
    with pytest.raises(ValueError, match="unknown mutation"):
        subject.mutate_yaml_for_test(_roadmap(), "unknown")


def test_v664_dispositions_preserve_terminal_pre_gate_and_absence() -> None:
    """SCENARIO-REPORT-7615-CUSTODY preserves fourteen literal V664 outcomes."""

    rows = subject.collect_v664_dispositions(ROOT)
    assert [row["task_id"] for row in rows] == list(subject.V664_TASK_IDS)
    assert all(row["authenticated"] is True for row in rows)
    assert subject.disposition_kind_counts(rows) == {
        "actual_producer": 9,
        "conductor_pre_gate": 2,
        "absent_producer": 3,
    }
    by_id = {row["task_id"]: row for row in rows}
    assert by_id["exp7603-guarded-update-fixture"]["verdict_class"] == "circular_positive"
    assert by_id["exp7605-fit-evidence"]["evidence_kind"] == "conductor_pre_gate"
    assert by_id["exp7607-evidence-energy"]["evidence_kind"] == "absent_producer"
    assert by_id["exp7614-capstone"]["evidence_kind"] == "actual_producer"


def test_method_records_map_primary_methods_without_reproduction_claim(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7615-METHODS keeps paper assumptions and access limits."""

    rows = subject.method_rows()
    assert {row["method_family"] for row in rows} == {
        "jsonschemabench",
        "eaev_evidence_alignment",
        "proper_calibeating",
    }
    assert all(row["external_result_is_local_measurement"] is False for row in rows)
    assert all("not a reproduction" in row["claim_limit"].lower() for row in rows)
    assert "GCT" in rows[0]["mechanism_read"] and "TPOT" in rows[0]["mechanism_read"]
    assert "counterfactual" in rows[1]["mechanism_read"].lower()
    assert "bounded proper scoring rules" in rows[2]["mechanism_read"]

    subject.write_method_records(tmp_path)
    note = (tmp_path / subject.NOTE_PATH).read_text(encoding="utf-8")
    assert "not a reproduction" in note.lower()
    assert "OpenReview PDF browser challenge" in note
    subject.write_method_records(tmp_path)
    studying = (tmp_path / subject.STUDY_PATH).read_text(encoding="utf-8")
    assert studying.count(subject.STUDY_MARKER) == 1
    assert "inaccessible" in studying.lower()


def test_artifact_is_complete_null_administrative_advisory() -> None:
    """REQ-REPORT-7615 separates exact contract readiness from scientific benefit."""

    artifact = subject.build_artifact_for_test(validation_receipts=_passing_receipts())
    assert artifact["honest_verdict"] == "complete_null_v665_contract_methods_ingested"
    assert artifact["verdict_class"] == "null"
    assert artifact["contract_ready_score"] == 1
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert all(value == 0 for value in artifact["invocation_counts"].values())
    assert artifact["planned_inference_substrate_class"] == "aggregation"
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["execution_venue"] == "host"
    assert artifact["execution_venue_details"]["host"]
    assert artifact["execution_venue_details"]["gpu_used"] is False
    assert len(artifact["rows"]) == len(artifact["task_contract_rows"]) == 14
    assert len(artifact["v664_dispositions"]) == 14
    assert {row["category"] for row in artifact["acceptance_gate_results"]} == {
        "validity",
        "readiness",
        "benefit",
        "retention",
        "freshness",
    }
    benefit = next(
        row for row in artifact["acceptance_gate_results"] if row["category"] == "benefit"
    )
    assert benefit["passed"] is False
    assert artifact["publication_gates"]["independent_science_gated"] is False
    assert subject.independent_reduce(artifact)["passed"] is True
    assert subject.validate_artifact(artifact, root=ROOT, require_terminal=False) == []


def test_protected_artifact_mutations_fail_closed() -> None:
    """SCENARIO-REPORT-7615-TERMINAL rejects false readiness and changed rows."""

    artifact = subject.build_artifact_for_test(validation_receipts=_passing_receipts())
    artifact["rows"][0]["observed"]["deliverable"] = "changed.json"
    assert subject.independent_reduce(artifact)["passed"] is False

    artifact = subject.build_artifact_for_test(validation_receipts=_passing_receipts())
    artifact["flagged_adversarial"] = True
    assert "independent_reduction_failed" in subject.validate_artifact(
        artifact, root=ROOT, require_terminal=False
    )

    artifact = subject.build_artifact_for_test(validation_receipts=_passing_receipts())
    artifact["contract_ready_score"] = True
    assert "contract_ready_score_bare_integer_required" in subject.validate_artifact(
        artifact, root=ROOT, require_terminal=False
    )


def test_blocked_artifact_and_scoped_validation_contract(tmp_path: Path) -> None:
    """REQ-REPORT-7615 names blocker operands and freezes private scoped checks."""

    blocked = subject.build_blocked_artifact(
        upstream="research-roadmap.yaml",
        path="research-roadmap.yaml",
        field="milestone",
        expected=subject.MILESTONE,
        observed="absent",
    )
    failure = blocked["gate_check_summary"]["first_failure"]
    assert blocked["honest_verdict"].startswith("complete_blocked_")
    assert blocked["verdict_class"] == "blocked"
    assert {"check", "upstream", "path", "field", "operator", "expected", "observed"} <= set(
        failure
    )
    assert subject.validate_artifact(blocked, root=ROOT, require_terminal=False) == []

    manifest = subject.affected_file_manifest()
    assert manifest["tests"] == [subject.TEST_PATH.as_posix()]
    assert manifest["modules"] == [subject.MODULE_PATH.as_posix()]
    commands = subject.build_validation_plan(ROOT, tmp_path / "private")
    assert subject.validate_validation_plan(ROOT, commands) == []
    assert [row.name for row in commands] == list(subject.REQUIRED_CHECK_NAMES)


def test_cold_modes_required_inputs_and_thin_wrapper(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7615-TERMINAL binds exact inputs and fresh readers."""

    assert all(expected == observed for _, _, expected, observed in subject._required_inputs(ROOT))
    artifact = subject.build_artifact_for_test(validation_receipts=_passing_receipts())
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(artifact), encoding="utf-8")
    assert subject.main(["--root", str(ROOT), "--cold-validate", str(candidate)]) == 0
    assert subject.main(["--root", str(ROOT), "--independent-reduce", str(candidate)]) == 0
    with pytest.raises(SystemExit):
        subject.parse_args(["--date", "20260923"])
    args = subject.parse_args(
        ["--date", subject.RUN_DATE, "--output", subject.RESULT_PATH.as_posix()]
    )
    assert args.output == subject.RESULT_PATH
    with pytest.raises(SystemExit):
        subject.parse_args(["--date", subject.RUN_DATE, "--output", "results/wrong.json"])

    text = (ROOT / subject.WRAPPER_PATH).read_text(encoding="utf-8")
    assert "experiment_7615_v665_contract_methods" in text
    assert len(text.splitlines()) <= 20


def test_validation_receipts_and_checksum_are_bound() -> None:
    """SCENARIO-REPORT-7615-TERMINAL rejects incomplete command custody."""

    receipts = _passing_receipts()
    assert subject.reduce_validation_receipts(receipts)["passed"] is True
    changed = deepcopy(receipts)
    changed[0].pop("log_sha256")
    assert subject.reduce_validation_receipts(changed)["passed"] is False

    artifact = subject.build_artifact_for_test(validation_receipts=receipts)
    artifact["reproducibility_checksum"] = "sha256:" + "0" * 64
    assert "checksum_mismatch" in subject.validate_artifact(
        artifact, root=ROOT, require_terminal=False
    )


def test_malformed_authorities_and_capstone_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-7615 rejects malformed authority and custody shapes."""

    comparison = subject.compare_contract_authorities("bad", None)
    assert comparison["passed"] is False
    assert {"markdown_task_table_missing", "yaml_task_table_missing"} <= set(comparison["errors"])

    capstone = tmp_path / subject.V664_CAPSTONE_PATH
    capstone.parent.mkdir(parents=True)
    capstone.write_text(json.dumps({"task_dispositions": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="dispositions are unavailable"):
        subject.collect_v664_dispositions(tmp_path)
    capstone.write_text(
        json.dumps({"task_dispositions": [{} for _ in subject.V664_TASK_IDS]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="order is invalid"):
        subject.collect_v664_dispositions(tmp_path)


def test_validation_plan_and_receipt_failures_are_explicit(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7615-TERMINAL rejects scope and receipt drift."""

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

    receipts = _passing_receipts()
    assert subject.reduce_validation_receipts([])["passed"] is False
    receipts[0]["passed"] = False
    receipts[0]["exit_code"] = 1
    assert subject.reduce_validation_receipts(receipts)["passed"] is False


def test_source_hash_reader_and_terminal_helpers_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7615-CUSTODY authenticates present and absent evidence."""

    assert subject._source_hashes_match({}, tmp_path) is False
    assert subject._source_hashes_match({"source_artifact_hashes": [1]}, tmp_path) is False
    missing = tmp_path / "missing.json"
    absent = {"source_artifact_hashes": [{"path": str(missing), "exists": False, "sha256": None}]}
    assert subject._source_hashes_match(absent, tmp_path) is True
    missing.write_text("changed", encoding="utf-8")
    assert subject._source_hashes_match(absent, tmp_path) is False
    wrong = {
        "source_artifact_hashes": [{"path": str(missing), "exists": True, "sha256": "sha256:wrong"}]
    }
    assert subject._source_hashes_match(wrong, tmp_path) is False

    commands = subject._terminal_commands(ROOT, tmp_path / "candidate.json")
    assert [row.name for row in commands] == list(subject.TERMINAL_CHECK_NAMES)
    left = [{"name": "r", "command_argv": ["x"], "exit_code": 0, "passed": True, "log_sha256": "h"}]
    assert subject._terminal_receipts_equal(left, deepcopy(left)) is True
    assert subject._terminal_receipts_equal(left, []) is False
    assert subject.date_argument(subject.RUN_DATE) == subject.RUN_DATE
    span = subject._span("unit", 1.0, 1.0, 1, 1)
    assert span["completed_units"] == span["checkpoint_position"] == 1


def test_cold_validator_reports_all_protected_boundaries() -> None:
    """REQ-REPORT-7615 reports malformed terminal evidence explicitly."""

    assert subject.validate_artifact([]) == ["artifact_mapping_required"]
    artifact = subject.build_artifact_for_test(validation_receipts=_passing_receipts())
    artifact.pop("schema")
    artifact["milestone"] = "wrong"
    artifact["honest_verdict"] = "success"
    artifact["verdict_class"] = "success"
    artifact["MODEL_SPECS"] = ["model"]
    artifact["model_invoked"] = True
    artifact["invocation_counts"] = {}
    artifact["inference_substrate_class"] = "live"
    artifact["acceptance_gate_results"] = []
    artifact["field_principles"] = {}
    artifact["method_map_path"] = "wrong"
    artifact["source_artifact_hashes"] = []
    errors = subject.validate_artifact(artifact, root=ROOT, require_terminal=False)
    assert any(error.startswith("required_fields_missing") for error in errors)
    assert {
        "schema_mismatch",
        "experiment_identity_mismatch",
        "terminal_verdict_prefix_required",
        "verdict_class_invalid",
        "model_specs_must_be_empty",
        "model_invoked_must_be_false",
        "current_model_calls_nonzero",
        "substrate_class_invalid",
        "acceptance_gate_shape_invalid",
        "field_principles_incomplete",
        "method_map_path_invalid",
        "source_hash_mismatch",
    } <= set(errors)

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


def test_missing_tool_version_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-7615 records an unavailable declared tool instead of fabricating it."""

    real_version = subject.importlib.metadata.version

    def fake_version(package: str) -> str:
        if package == "ruff":
            raise subject.importlib.metadata.PackageNotFoundError(package)
        return real_version(package)

    monkeypatch.setattr(subject.importlib.metadata, "version", fake_version)
    row = next(item for item in subject.preconditions_checked(ROOT) if item["upstream"] == "ruff")
    assert row["observed"] == "absent"
    assert row["passed"] is False
