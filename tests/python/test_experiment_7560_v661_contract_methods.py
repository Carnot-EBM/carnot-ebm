"""Tests for the V661 contract and method advisory.

Spec refs: REQ-REPORT-7560 and SCENARIO-REPORT-7560-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest
import yaml

from carnot import experiment_7560_v661_contract_methods as subject


ROOT = Path(__file__).resolve().parents[2]


def _roadmap() -> dict[str, object]:
    value = yaml.safe_load((ROOT / subject.ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _passing_validation() -> dict[str, object]:
    names = (*subject.REQUIRED_CHECK_NAMES, *subject.TERMINAL_CHECK_NAMES)
    return {
        "required_checks_passed": True,
        "terminal_validation_passed": True,
        "validation_receipts": [
            {
                "name": name,
                "required": True,
                "passed": True,
                "exit_code": 0,
                "timed_out": False,
                "duration_s": 0.0,
                "log_path": f"unit-test/{name}.log",
                "log_sha256": "sha256:" + "1" * 64,
            }
            for name in names
        ],
        "repository_health": {"status": "outside_current_affected_validity"},
    }


def test_authority_resolution_accepts_staged_and_consumed_states(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7560-AUTHORITY selects the matching existing authority."""

    active = tmp_path / subject.ACTIVE_ROADMAP_PATH
    active.write_text((ROOT / subject.ACTIVE_ROADMAP_PATH).read_text(), encoding="utf-8")

    selected, value, candidates = subject.resolve_v661_roadmap(tmp_path)
    assert selected == active
    assert value["milestone"] == subject.MILESTONE
    assert candidates[0]["exists"] is False

    staged = tmp_path / subject.NEXT_ROADMAP_PATH
    staged.write_text(active.read_text(encoding="utf-8"), encoding="utf-8")
    selected, _, candidates = subject.resolve_v661_roadmap(tmp_path)
    assert selected == staged
    assert [row["matches_milestone"] for row in candidates] == [True, True]

    changed = yaml.safe_load(staged.read_text(encoding="utf-8"))
    changed["milestone"] = "2026.09.660"
    staged.write_text(yaml.safe_dump(changed), encoding="utf-8")
    selected, _, _ = subject.resolve_v661_roadmap(tmp_path)
    assert selected == active


def test_authority_resolution_rejects_missing_or_stale_inputs(tmp_path: Path) -> None:
    """REQ-REPORT-7560 does not fabricate either planning authority."""

    with pytest.raises(ValueError, match="V661 roadmap authority"):
        subject.resolve_v661_roadmap(tmp_path)


def test_exact_contract_and_private_mutations() -> None:
    """SCENARIO-REPORT-7560-AUTHORITY compares all fields and rejects drift."""

    design = (ROOT / subject.DESIGN_PATH).read_text(encoding="utf-8")
    roadmap = _roadmap()
    comparison = subject.compare_contract_authorities(design, roadmap)
    assert comparison["passed"] is True
    assert comparison["errors"] == []
    assert [row["unit_id"] for row in comparison["contract_rows"]] == list(
        subject.EXPECTED_TASK_IDS
    )
    assert all(all(row["checks"].values()) for row in comparison["contract_rows"])

    controls = subject.run_contract_mutation_controls(design, roadmap)
    assert len(controls) == 2 * len(subject.mutation_names())
    assert {row["mutation"] for row in controls} >= {"count", "order", "path", "gate_field"}
    assert all(row["qualified"] is True for row in controls)


@pytest.mark.parametrize("mutation", subject.mutation_names())
def test_each_yaml_mutation_fails(mutation: str) -> None:
    """REQ-REPORT-7560 rejects every registered YAML authority mutation."""

    design = (ROOT / subject.DESIGN_PATH).read_text(encoding="utf-8")
    comparison = subject.compare_contract_authorities(
        design, subject.mutate_yaml_for_test(_roadmap(), mutation)
    )
    assert comparison["passed"] is False


def test_v660_custody_preserves_nine_producers_and_five_absences() -> None:
    """SCENARIO-REPORT-7560-CUSTODY keeps null, blocked, and absent distinct."""

    rows = subject.collect_v660_dispositions(ROOT)
    assert len(rows) == 14
    assert sum(row["producer_present"] is True for row in rows) == 9
    assert sum(row["producer_present"] is False for row in rows) == 5
    by_id = {row["task_id"]: row for row in rows}
    assert by_id["exp7546-contract-methods"]["flagged_adversarial"] is True
    assert by_id["exp7548-capture-runner"]["verdict_class"] == "blocked"
    assert by_id["exp7549-count-learning"]["verdict_class"] == "null"
    assert by_id["exp7550-count-audit"]["verdict_class"] == "null"
    assert by_id["exp7551-native-pilot"]["evidence_kind"] == "pre_gate_diagnostic"
    assert by_id["exp7551-native-pilot"]["successful_pilot"] is False
    assert all(row["authenticated"] is True for row in rows)


def test_method_map_names_tests_deferrals_and_access_limits(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7560-METHODS keeps external claims bounded."""

    methods = subject.method_rows()
    assert 3 <= len(methods) <= 5
    assert {row["method_family"] for row in methods} == {
        "calibeating_proper_loss",
        "proper_scoring_limits",
        "textual_source_interventions",
        "fpga_asic_service_cost",
    }
    assert all(row["local_test_or_deferral"] for row in methods)
    assert all(row["external_result_is_local_measurement"] is False for row in methods)
    assert subject.secondary_access_rows()
    assert subject.retired_mechanisms()

    subject.write_method_records(tmp_path)
    first = (tmp_path / subject.NOTE_PATH).read_text(encoding="utf-8")
    subject.write_method_records(tmp_path)
    assert (tmp_path / subject.NOTE_PATH).read_text(encoding="utf-8") == first
    study = (tmp_path / subject.STUDY_PATH).read_text(encoding="utf-8")
    assert study.count(subject.STUDY_MARKER) == 1


def test_operator_queue_keeps_e0_blocked_and_e6_resolved() -> None:
    """REQ-REPORT-7560 preserves the current operator queue states."""

    queue = subject.operator_queue(ROOT)
    assert queue["E0"]["status"] == "operator_blocked"
    assert queue["E6"]["status"] == "resolved"


def test_artifact_is_advisory_complete_and_independently_reducible() -> None:
    """SCENARIO-REPORT-7560-VALIDATION separates readiness from benefit."""

    artifact = subject.build_artifact_for_test(validation=_passing_validation())
    assert artifact["contract_ready_score"] == 1
    assert artifact["method_ingestion_complete_score"] == 1
    assert type(artifact["contract_ready_score"]) is int
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["positive_claim"] is False
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert all(value == 0 for value in artifact["invocation_counts"].values())
    assert artifact["execution_venue"] == "host"
    assert artifact["selected_roadmap_path"] == "research-roadmap.yaml"
    assert artifact["completed_ledger_ends_at"] == "2026.09.659"
    assert len(artifact["rows"]) == 13
    assert len(artifact["prior_dispositions"]) == 14
    assert artifact["sample_size_budget"]["completed"] == 13
    assert artifact["publication_gates"]["publication_performed"] is False
    assert subject.independent_reduce(artifact)["passed"] is True
    assert subject.validate_artifact(artifact, require_terminal=False) == []


@pytest.mark.parametrize(
    ("field", "replacement", "expected_error"),
    [
        ("contract_ready_score", True, "contract_ready_score_bare_integer_required"),
        ("honest_verdict", "success", "terminal_verdict_prefix_required"),
        ("selected_roadmap_path", "missing.yaml", "selected_roadmap_path_invalid"),
        ("reproducibility_checksum", "sha256:" + "0" * 64, "checksum_mismatch"),
    ],
)
def test_artifact_rejects_protected_field_mutations(
    field: str, replacement: object, expected_error: str
) -> None:
    """REQ-REPORT-7560 rejects mutations of terminal identity and readiness."""

    artifact = subject.build_artifact_for_test(validation=_passing_validation())
    artifact[field] = replacement
    assert expected_error in subject.validate_artifact(artifact, require_terminal=False)


def test_artifact_rejects_prior_row_and_model_count_mutations() -> None:
    """REQ-REPORT-7560 binds historical custody and zero current calls."""

    artifact = subject.build_artifact_for_test(validation=_passing_validation())
    artifact["prior_dispositions"][0]["flagged_adversarial"] = False
    assert "independent_reduction_failed" in subject.validate_artifact(
        artifact, require_terminal=False
    )

    artifact = subject.build_artifact_for_test(validation=_passing_validation())
    artifact["invocation_counts"]["model_loads_attempted"] = 1
    assert "current_model_calls_nonzero" in subject.validate_artifact(
        artifact, require_terminal=False
    )


def test_blocked_artifact_names_exact_missing_input() -> None:
    """REQ-REPORT-7560 reports external absence as complete blocked evidence."""

    artifact = subject.build_blocked_artifact(
        "research-roadmap.yaml",
        field="milestone",
        expected=subject.MILESTONE,
        observed="absent",
    )
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("complete_blocked_")
    first = artifact["gate_check_summary"]["first_failure"]
    assert first == {
        "check": "required_input",
        "upstream": "research-roadmap.yaml",
        "path": "research-roadmap.yaml",
        "field": "milestone",
        "expected": subject.MILESTONE,
        "observed": "absent",
        "op": "==",
        "passed": False,
    }


def test_cli_date_and_cold_validation(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7560-VALIDATION exposes a thin read-only replay."""

    with pytest.raises(SystemExit):
        subject.parse_args(["--date", "20260922"])
    artifact = subject.build_artifact_for_test(validation=_passing_validation())
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(artifact), encoding="utf-8")
    assert subject.main(["--cold-validate", str(candidate)]) == 0


def test_source_manifest_has_no_missing_current_inputs() -> None:
    """REQ-REPORT-7560 checks current inputs before measurement."""

    rows = subject.collect_preconditions(ROOT, ROOT / subject.ACTIVE_ROADMAP_PATH)
    failed = [row for row in rows if row["passed"] is not True]
    assert failed == []
    assert any(row["field"] == "cpu_and_storage" for row in rows)


def test_wrapper_is_thin() -> None:
    """REQ-REPORT-7560 keeps reusable behavior outside the public CLI."""

    wrapper = ROOT / subject.WRAPPER_PATH
    if wrapper.exists():
        text = wrapper.read_text(encoding="utf-8")
        assert "experiment_7560_v661_contract_methods" in text
        assert len(text.splitlines()) <= 20


def test_malformed_authorities_and_private_helper_errors(tmp_path: Path) -> None:
    """REQ-REPORT-7560 fails closed on malformed private inputs."""

    assert subject.compare_contract_authorities("bad", None)["passed"] is False
    comparison = subject.compare_contract_authorities("**Milestone:** 2026.09.661", _roadmap())
    assert "markdown_task_table_missing" in comparison["errors"]
    declarations = subject._producer_declarations(  # noqa: SLF001
        {"tasks": [42, {"id": "x", "gated_on": [42]}]}
    )
    assert declarations == {"x": []}
    with pytest.raises(ValueError, match="unknown mutation"):
        subject.mutate_yaml_for_test(_roadmap(), "unknown")
    controls = subject.run_contract_mutation_controls("bad", _roadmap())
    assert all(row["observed_errors"] == ["baseline_authorities_incomplete"] for row in controls)

    design_path = tmp_path / subject.V660_DESIGN_PATH
    design_path.parent.mkdir(parents=True)
    text = (ROOT / subject.V660_DESIGN_PATH).read_text(encoding="utf-8")
    design_path.write_text(text.replace("exp7546-contract-methods", "exp9999-changed", 1))
    with pytest.raises(ValueError, match="preserved V660 design"):
        subject._v660_tasks(tmp_path)  # noqa: SLF001

    capstone = tmp_path / subject.V660_PRODUCER_PATHS["exp7559-capstone"]
    capstone.parent.mkdir(parents=True, exist_ok=True)
    capstone.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="capstone dispositions"):
        subject._capstone_disposition_index(tmp_path)  # noqa: SLF001


def test_validation_and_repository_plans_are_frozen(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7560-VALIDATION keeps all command targets scoped."""

    private = tmp_path / "private"
    commands = subject.build_validation_plan(ROOT, private)
    assert subject.validate_validation_plan(ROOT, commands) == []
    assert [command.name for command in commands] == list(
        subject.validation_scope.REQUIRED_CHECK_NAMES
    )
    guards = subject.build_repository_check_plan(ROOT, ROOT / subject.ACTIVE_ROADMAP_PATH)
    assert [row.spec.name for row in guards] == list(subject.REQUIRED_REPOSITORY_CHECK_NAMES)
    selected = str(ROOT / subject.ACTIVE_ROADMAP_PATH)
    assert all(
        selected in row.spec.argv or row.spec.name == "publication_gate_json" for row in guards
    )


def test_precondition_and_publication_receipt_error_paths(tmp_path: Path) -> None:
    """REQ-REPORT-7560 records unreadable authority and G1-G4 logs exactly."""

    roadmap = tmp_path / "bad.yaml"
    roadmap.write_text("not: [valid", encoding="utf-8")
    rows = subject.collect_preconditions(tmp_path, roadmap)
    selected = next(row for row in rows if row["check"] == "selected_roadmap_authority")
    assert selected["observed"] == "unreadable"

    log = tmp_path / "publication.json"
    log.write_text(
        json.dumps({"gates": {name: {"pass": True} for name in ("G1", "G2", "G3", "G4")}}),
        encoding="utf-8",
    )
    receipt = {
        "name": "publication_gate_json",
        "log_path": str(log),
        "command_argv": ["publication_gate.py", "--json"],
        "exit_code": 0,
        "log_sha256": "sha256:" + "2" * 64,
    }
    publication = subject._publication_from_receipts(tmp_path, [receipt])  # noqa: SLF001
    assert publication["paper_ready"] is True
    receipt["log_path"] = log.name
    assert subject._publication_from_receipts(tmp_path, [receipt])["paper_ready"] is True  # noqa: SLF001
    log.write_text("bad", encoding="utf-8")
    assert subject._publication_from_receipts(tmp_path, [receipt])["paper_ready"] is False  # noqa: SLF001
    assert subject._publication_from_receipts(tmp_path, [])["unmet_gates"] == [  # noqa: SLF001
        "G1",
        "G2",
        "G3",
        "G4",
    ]


def test_invalid_validation_builds_disqualified_advisory() -> None:
    """REQ-REPORT-7560 disqualifies failed owned validation."""

    validation = {**_passing_validation(), "affected_checks_passed": False}
    artifact = subject.build_artifact_for_test(validation=validation)
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["contract_ready_score"] == 0
    supplied = {"gates": {name: {"pass": True} for name in ("G1", "G2", "G3", "G4")}}
    assert (
        subject._publication_for_artifact(ROOT, {"publication_gates": supplied})[  # noqa: SLF001
            "publication_performed"
        ]
        is False
    )


def test_source_hash_reader_rejects_malformed_rows() -> None:
    """SCENARIO-REPORT-7560-VALIDATION rejects missing or changed custody."""

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
            {"source_artifact_hashes": [{"path": "missing", "sha256": "sha256:" + "0" * 64}]},
            ROOT,
        )
        is False
    )


@pytest.mark.parametrize(
    ("field", "replacement", "expected"),
    [
        ("schema", "bad", "schema_mismatch"),
        ("experiment_id", "bad", "experiment_identity_mismatch"),
        ("verdict_class", "bad", "verdict_class_invalid"),
        ("MODEL_SPECS", ["bad"], "model_specs_must_be_empty"),
        ("inference_substrate_class", "bad", "substrate_class_invalid"),
        ("inference_substrate", "bad", "substrate_invalid"),
        ("execution_venue", "host_cpu", "execution_venue_invalid"),
        ("acceptance_gate_results", [], "acceptance_gate_shape_invalid"),
        ("publication_gates", {}, "publication_gates_invalid"),
        ("field_principles", {}, "field_principles_incomplete"),
        ("source_artifact_hashes", [], "source_hash_mismatch"),
    ],
)
def test_cold_validator_rejects_each_schema_boundary(
    field: str, replacement: object, expected: str
) -> None:
    """REQ-REPORT-7560 validates each protected schema boundary."""

    artifact = subject.build_artifact_for_test(validation=_passing_validation())
    artifact[field] = replacement
    assert expected in subject.validate_artifact(artifact, require_terminal=False)


def test_cold_validator_defensive_shapes_and_size(tmp_path: Path) -> None:
    """REQ-REPORT-7560 rejects incomplete, invalid, or oversized evidence."""

    assert subject.validate_artifact([]) == ["artifact_mapping_required"]
    artifact = subject.build_artifact_for_test(validation=_passing_validation())
    del artifact["schema"]
    errors = subject.validate_artifact(artifact, require_terminal=False)
    assert any(error.startswith("required_fields_missing:") for error in errors)

    artifact = subject.build_artifact_for_test(validation=_passing_validation())
    artifact["field_principles"]["schema"] = ""
    assert "field_principles_invalid" in subject.validate_artifact(artifact, require_terminal=False)

    artifact = subject.build_artifact_for_test(validation=_passing_validation())
    artifact["oversized"] = "x" * (20 * 1024 * 1024)
    assert "terminal_artifact_size_limit" in subject.validate_artifact(
        artifact, require_terminal=False
    )

    stale = tmp_path / subject.ACTIVE_ROADMAP_PATH
    stale.write_text("milestone: 2026.09.660\n", encoding="utf-8")
    artifact = subject.build_artifact_for_test(validation=_passing_validation())
    artifact["selected_roadmap_path"] = subject.ACTIVE_ROADMAP_PATH.as_posix()
    assert "selected_roadmap_milestone_invalid" in subject.validate_artifact(
        artifact, root=tmp_path, require_terminal=False
    )
    stale.write_text("not: [valid", encoding="utf-8")
    assert "selected_roadmap_milestone_invalid" in subject.validate_artifact(
        artifact, root=tmp_path, require_terminal=False
    )


def test_phase_terminal_plan_and_valid_date(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7560-VALIDATION freezes terminal command identities."""

    span = subject._phase_span("unit", 1.0, 0.0, units=2, checkpoint="done")  # noqa: SLF001
    assert span["completed_units"] == 2
    plan = subject._terminal_commands(ROOT, tmp_path / "candidate.json")  # noqa: SLF001
    assert [row.spec.name for row in plan] == list(subject.TERMINAL_CHECK_NAMES)
    assert subject.date_argument(subject.RUN_DATE) == subject.RUN_DATE
