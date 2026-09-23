"""Tests for the V660 fourteen-disposition capstone.

Spec refs: REQ-REPORT-7559 and SCENARIO-REPORT-7559-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7559_v660_capstone as capstone


def test_contract_and_inventory_preserve_fourteen_exact_rows() -> None:
    """REQ-REPORT-7559; SCENARIO-REPORT-7559-INVENTORY."""

    contract = capstone.load_contract(capstone.REPO_ROOT)
    assert contract["comparison_passed"] is True
    assert [row["id"] for row in contract["tasks"]] == list(capstone.EXPECTED_TASK_IDS)

    evidence = capstone.collect_evidence(capstone.REPO_ROOT, contract["tasks"])
    assert list(evidence) == list(capstone.EXPECTED_TASK_IDS[:-1])
    assert evidence["exp7546-contract-methods"]["advisory_only"] is True
    assert evidence["exp7546-contract-methods"]["flagged_adversarial"] is True
    assert evidence["exp7551-native-pilot"]["evidence_state"] == "conductor_gate_blocked"
    assert evidence["exp7551-native-pilot"]["diagnostic_path"] == (
        "results/experiment_7551_native_pilot.json"
    )
    assert evidence["exp7552-fit-capture"]["source_status"] == "GATE_BLOCK"
    assert evidence["exp7555-source-evaluation"]["payload"] == {}


def test_authority_mutations_fail_closed() -> None:
    """REQ-REPORT-7559; SCENARIO-REPORT-7559-INVENTORY."""

    contract = capstone.load_contract(capstone.REPO_ROOT)
    markdown = (capstone.REPO_ROOT / capstone.DESIGN_PATH).read_text()
    for mutate in (
        lambda road: road["tasks"].pop(),
        lambda road: road["tasks"].reverse(),
        lambda road: road["tasks"].__setitem__(0, {**road["tasks"][0], "title": "changed"}),
        lambda road: road["tasks"].__setitem__(5, {**road["tasks"][5], "gated_on": []}),
    ):
        roadmap = deepcopy(contract["roadmap"])
        mutate(roadmap)
        assert capstone.compare_authorities(markdown, roadmap)["passed"] is False


def test_conductor_diagnostic_preserves_exact_failed_gate() -> None:
    """REQ-REPORT-7559; SCENARIO-REPORT-7559-INVENTORY/FAILURES."""

    contract = capstone.load_contract(capstone.REPO_ROOT)
    evidence = capstone.collect_evidence(capstone.REPO_ROOT, contract["tasks"])
    native = evidence["exp7551-native-pilot"]
    failure = native["gate_check_summary"]
    assert failure["upstream"] == "exp7548-capture-runner"
    assert failure["path"].endswith("results/experiment_7548_v660_capture_runner.json")
    assert failure["field"] == "gpu_capacity_observed_score"
    assert failure["expected"] == 1
    assert failure["observed"] == 0
    assert native["honest_verdict"] == "blocked_gate_check_failed"

    fit = evidence["exp7552-fit-capture"]
    assert fit["gate_check_summary"] == {
        "check": "conductor_pre_gate",
        "upstream": "exp7551-native-pilot",
        "path": "results/experiment_7551_v660_native_pilot.json",
        "field": "native_tool_ready_score",
        "op": "==",
        "expected": 1,
        "observed": "absent",
        "passed": False,
        "status": "GATE_BLOCK",
    }


def test_missing_without_diagnostic_is_explicit(tmp_path: Path) -> None:
    """REQ-REPORT-7559; SCENARIO-REPORT-7559-INVENTORY."""

    task = {
        "id": "exp7552-fit-capture",
        "title": "Capture complete tool-source fitting and calibration evidence",
        "deliverable": "results/missing.json",
        "gated_on": [],
    }
    source = capstone.load_producer(tmp_path, task, {}, {})
    assert source["evidence_state"] == "missing"
    assert source["verdict_class"] == "blocked"
    assert source["gate_check_summary"]["path"] == "results/missing.json"
    assert source["gate_check_summary"]["observed"] is False


def test_four_branch_conclusions_remain_independent() -> None:
    """REQ-REPORT-7559; SCENARIO-REPORT-7559-BRANCHES."""

    contract = capstone.load_contract(capstone.REPO_ROOT)
    evidence = capstone.collect_evidence(capstone.REPO_ROOT, contract["tasks"])
    branches = {row["branch"]: row for row in capstone.build_branch_dispositions(evidence)}

    count = branches["exploratory_cached_count_learning"]
    assert count["verdict_class"] == "null"
    assert count["independent_audit_required"] is True
    assert count["independent_audit_complete"] is True
    assert count["readiness"] == 1
    assert count["benefit"] == 0
    assert count["confirmatory_benefit"] == 0

    source = branches["fresh_injected_tool_decisions"]
    assert source["verdict_class"] == "blocked"
    assert source["independent_source_reduction_required"] is True
    assert source["independent_source_reduction_complete"] is False
    assert source["benefit"] is None

    arc = branches["corrected_live_agent_generalization"]
    assert arc["verdict_class"] == "null"
    assert arc["custody_ready"] == 1
    assert arc["analysis_complete"] == 1
    assert arc["benefit"] == 0

    service = branches["cpu_durability_and_board_continuity"]
    assert service["verdict_class"] == "null"
    assert service["service_complete"] == 1
    assert service["board_continuity_complete"] == 1
    assert service["hardware_benefit"] == 0


@pytest.mark.parametrize(
    ("affected", "terminal", "expected_class", "expected_prefix"),
    [
        (True, False, "partial", "partial_retryable_"),
        (False, True, "disqualified", "complete_disqualified_"),
    ],
)
def test_owned_validation_precedes_scientific_classification(
    affected: bool,
    terminal: bool,
    expected_class: str,
    expected_prefix: str,
) -> None:
    """REQ-REPORT-7559; SCENARIO-REPORT-7559-FAILURES."""

    result = capstone.classify_terminal(
        [{"evidence_state": "terminal", "verdict_class": "null", "advisory_only": False}],
        affected_complete=affected,
        terminal_complete=terminal,
    )
    assert result["verdict_class"] == expected_class
    assert result["honest_verdict"].startswith(expected_prefix)


def test_external_absence_blocks_once_and_advisory_flag_does_not_disqualify() -> None:
    """REQ-REPORT-7559; SCENARIO-REPORT-7559-FAILURES."""

    advisory = {
        "evidence_state": "invalid",
        "verdict_class": "disqualified",
        "advisory_only": True,
    }
    blocked = {
        "evidence_state": "conductor_gate_blocked",
        "verdict_class": "blocked",
        "advisory_only": False,
    }
    result = capstone.classify_terminal(
        [advisory, blocked], affected_complete=True, terminal_complete=True
    )
    assert result == {
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_required_v660_source_science_externally_gated",
        "status": "complete_blocked_required_v660_source_science_externally_gated",
    }

    invalid = {**blocked, "evidence_state": "invalid", "verdict_class": "disqualified"}
    assert (
        capstone.classify_terminal([invalid], affected_complete=True, terminal_complete=True)[
            "verdict_class"
        ]
        == "disqualified"
    )


def test_terminal_artifact_has_complete_accounting_and_blocked_science() -> None:
    """REQ-REPORT-7559; SCENARIO-REPORT-7559-INVENTORY/BRANCHES/FAILURES."""

    artifact = capstone.build_artifact_for_test()
    assert artifact["schema"] == "carnot.exp7559.v660.capstone.v1"
    assert artifact["experiment_id"] == "exp7559-capstone"
    assert artifact["milestone"] == "2026.09.660"
    assert artifact["run_date"] == "20260923"
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == capstone.ZERO_INVOCATION_COUNTS
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["planned_inference_substrate_class"] == "aggregation"
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["execution_venue"] == "host"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["positive_claim"] is False
    assert artifact["flagged_adversarial"] is False
    assert artifact["retained_source_adversarial_flags"] == [
        {
            "task_id": "exp7546-contract-methods",
            "flagged_adversarial": True,
            "excluded_from_scientific_gate": True,
        }
    ]
    assert artifact["capstone_complete_score"] == 1
    assert len(artifact["rows"]) == 14
    assert artifact["rows"] == artifact["task_dispositions"]
    assert artifact["rows"][-1]["evidence_state"] == "current_terminal"
    assert artifact["rows"][-1]["artifact_path"] is None
    assert len(artifact["branch_dispositions"]) == 4
    assert artifact["sample_size_budget"]["planned"] == 14
    assert artifact["sample_size_budget"]["completed"] == 14
    assert artifact["gate_check_summary"]["first_failure"]["field"] == (
        "gpu_capacity_observed_score"
    )
    assert capstone.validate_artifact(artifact) == []


def test_publication_gates_are_exact_conjunction() -> None:
    """REQ-REPORT-7559; SCENARIO-REPORT-7559-VALIDATION."""

    result = capstone.evaluate_publication_gates(capstone.REPO_ROOT)
    gates = result["gates"]
    assert list(gates) == ["G1", "G2", "G3", "G4"]
    assert result["paper_ready"] is all(gates[name]["pass"] is True for name in gates)
    assert result["unmet_gates"] == [name for name in gates if gates[name]["pass"] is not True]
    assert result["publication_performed"] is False


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda value: value.update(model_invoked=True), "model_contract_invalid"),
        (
            lambda value: value.update(inference_substrate="live_llm_inference"),
            "substrate_invalid",
        ),
        (lambda value: value["task_dispositions"].pop(), "task_dispositions_invalid"),
        (
            lambda value: value["branch_dispositions"][0].update(benefit=1),
            "branch_dispositions_invalid",
        ),
        (
            lambda value: value["retirement_rows"].append({"task_id": "fake"}),
            "retirement_rows_invalid",
        ),
        (
            lambda value: value.update(capstone_complete_score=True),
            "capstone_score_invalid",
        ),
        (
            lambda value: value["source_artifact_hashes"][0].update(sha256="sha256:bad"),
            "source_hash_mismatch",
        ),
        (
            lambda value: value["field_principles"].pop("schema"),
            "field_principles_invalid",
        ),
        (
            lambda value: value.update(reproducibility_checksum="sha256:bad"),
            "reproducibility_checksum_invalid",
        ),
    ],
)
def test_protected_artifact_mutations_fail(mutation: object, expected: str) -> None:
    """REQ-REPORT-7559; SCENARIO-REPORT-7559-VALIDATION."""

    artifact = capstone.build_artifact_for_test()
    assert callable(mutation)
    mutation(artifact)  # type: ignore[operator]
    assert expected in capstone.validate_artifact(artifact)


def test_running_self_row_never_reads_future_artifact() -> None:
    """REQ-REPORT-7559; SCENARIO-REPORT-7559-INVENTORY."""

    contract = capstone.load_contract(capstone.REPO_ROOT)
    evidence = capstone.collect_evidence(capstone.REPO_ROOT, contract["tasks"])
    terminal = capstone.classify_terminal(
        list(evidence.values()), affected_complete=True, terminal_complete=False
    )
    rows = capstone.task_dispositions(
        contract["tasks"], evidence, terminal, current_validation_complete=False
    )
    assert rows[-1]["task_id"] == "exp7559-capstone"
    assert rows[-1]["artifact_path"] is None
    assert rows[-1]["artifact_sha256"] is None
    assert rows[-1]["evidence_state"] == "current_running"
    assert rows[-1]["completed"] is False


def test_prior_failures_do_not_retire_external_absence() -> None:
    """REQ-REPORT-7559; SCENARIO-REPORT-7559-FAILURES."""

    contract = capstone.load_contract(capstone.REPO_ROOT)
    evidence = capstone.collect_evidence(capstone.REPO_ROOT, contract["tasks"])
    rows = capstone.reduce_prior_failures(contract["tasks"], evidence)
    source_rows = [
        row for row in rows if row["task_id"] in {"exp7551-native-pilot", "exp7552-fit-capture"}
    ]
    assert source_rows
    assert all(row["external_absence"] is True for row in source_rows)
    assert all(row["retirement_triggered"] is False for row in source_rows)
    assert capstone.retirement_rows(rows) == []

    repeated = deepcopy(rows[0])
    repeated.update(
        exact_text_match=True,
        same_completed_mechanism=True,
        external_absence=False,
        retirement_triggered=True,
    )
    retired = capstone.retirement_rows([repeated])
    assert retired[0]["scope"] == "same_completed_scientific_mechanism_only"


def test_continuations_name_changed_evidence_or_mechanism() -> None:
    """REQ-REPORT-7559; SCENARIO-REPORT-7559-BRANCHES/FAILURES."""

    rows = capstone.continuation_rows()
    assert {row["branch"] for row in rows} == {
        "exploratory_cached_count_learning",
        "fresh_injected_tool_decisions",
        "corrected_live_agent_generalization",
        "cpu_durability_and_board_continuity",
    }
    assert all(bool(row.get("changed_evidence") or row.get("changed_mechanism")) for row in rows)
    assert all(row["activation_authorized"] is False for row in rows)


def test_validation_plan_is_private_scoped_and_complete(tmp_path: Path) -> None:
    """REQ-REPORT-7559; SCENARIO-REPORT-7559-VALIDATION."""

    private = tmp_path / "private"
    private.mkdir()
    commands = capstone.build_validation_plan(capstone.REPO_ROOT, private)
    assert capstone.validate_validation_plan(capstone.REPO_ROOT, commands) == []
    assert [row.name for row in commands] == list(capstone.REQUIRED_CHECK_NAMES)
    focused = next(row for row in commands if row.name == "focused_pytest")
    assert "-n" in focused.argv and "0" in focused.argv
    assert "--no-cov" in focused.argv
    assert any(str(private) in arg for arg in focused.argv)

    changed = list(commands)
    changed[1] = capstone.validation_scope.CommandSpec(
        "focused_pytest", ("pytest", "tests/python"), "too_broad"
    )
    assert capstone.validate_validation_plan(capstone.REPO_ROOT, changed)


def test_preconditions_include_requirement_authority_and_resource() -> None:
    """REQ-REPORT-7559; SCENARIO-REPORT-7559-INVENTORY/VALIDATION."""

    contract = capstone.load_contract(capstone.REPO_ROOT)
    evidence = capstone.collect_evidence(capstone.REPO_ROOT, contract["tasks"])
    rows = capstone.collect_preconditions(capstone.REPO_ROOT, contract, evidence)
    by_check = {row["check"]: row for row in rows}
    assert by_check["driving_requirement"]["observed"] == "REQ-REPORT-7559"
    assert by_check["roadmap_authority"]["observed"] == "research-roadmap.yaml"
    assert by_check["aggregation_resource"]["passed"] is True
    assert by_check["producer_state:exp7555-source-evaluation"]["observed"] == (
        "conductor_gate_blocked"
    )


def test_missing_fields_non_mapping_and_date_fail_closed() -> None:
    """REQ-REPORT-7559; SCENARIO-REPORT-7559-VALIDATION."""

    assert capstone.validate_artifact([]) == ["artifact_mapping_required"]
    artifact = capstone.build_artifact_for_test()
    artifact.pop("schema")
    assert capstone.validate_artifact(artifact) == ["missing_required_field:schema"]
    assert capstone.date_argument("20260923") == "20260923"
    with pytest.raises(ValueError, match="20260923"):
        capstone.date_argument("20260922")


def test_publication_reader_malformed_json_is_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-7559; SCENARIO-REPORT-7559-VALIDATION."""

    class Result:
        stdout = "not json"
        stderr = "reader error"
        returncode = 3

    capstone.evaluate_publication_gates.cache_clear()
    monkeypatch.setattr(capstone.subprocess, "run", lambda *args, **kwargs: Result())
    result = capstone.evaluate_publication_gates(Path("/tmp"))
    assert result["unmet_gates"] == ["reader_failed"]
    assert result["exit_code"] == 3
    capstone.evaluate_publication_gates.cache_clear()


def test_result_json_shape_if_present() -> None:
    """REQ-REPORT-7559; SCENARIO-REPORT-7559-VALIDATION."""

    path = capstone.REPO_ROOT / capstone.RESULT_PATH
    if path.exists():
        value = json.loads(path.read_text())
        assert capstone.validate_artifact(value, require_terminal=True) == []


def test_defensive_contract_and_compact_helper_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-7559; SCENARIO-REPORT-7559-INVENTORY/VALIDATION."""

    monkeypatch.setattr(
        capstone,
        "resolve_v660_roadmap",
        lambda root: (root / "research-roadmap.yaml", {"tasks": None}, []),
    )
    monkeypatch.setattr(capstone, "compare_authorities", lambda text, road: {"passed": True})
    with pytest.raises(ValueError, match="task list"):
        capstone.load_contract(capstone.REPO_ROOT)

    assert capstone._receipt_rows({}) == []
    assert capstone._required_receipts_pass({}) is False
    assert capstone._acceptance_rows({}) == []
    assert capstone._receipt_set_passed({}, capstone.REQUIRED_CHECK_NAMES) is False
    assert capstone._ready_value_fields({"ready_score": 1, "bool_score": True, "label": "x"}) == {
        "ready_score": 1
    }


def test_path_status_and_gate_defensive_branches(tmp_path: Path) -> None:
    """REQ-REPORT-7559; SCENARIO-REPORT-7559-INVENTORY/FAILURES."""

    assert capstone._path_label(Path("relative.json"), tmp_path) == "relative.json"
    assert capstone._path_label(Path("/outside/file.json"), tmp_path) == "/outside/file.json"
    assert capstone.parse_conductor_statuses(tmp_path, []) == {}
    assert capstone._gate_satisfied("null", "in", ["null", "positive"]) is True
    assert capstone._gate_satisfied("null", "bad", ["null"]) is False

    fallback = capstone._cascade_failure(
        {"id": "task", "deliverable": "results/task.json", "gated_on": ["bad"]}, {}, {}
    )
    assert fallback["check"] == "producer_artifact_exists"


def test_malformed_present_producer_and_failure_causes(tmp_path: Path) -> None:
    """REQ-REPORT-7559; SCENARIO-REPORT-7559-INVENTORY."""

    task = {"id": "exp7552-fit-capture", "deliverable": "results/bad.json"}
    path = tmp_path / "results/bad.json"
    path.parent.mkdir()
    path.write_text("[]")
    source = capstone.load_producer(tmp_path, task, {}, {})
    assert source["evidence_state"] == "invalid"
    assert source["gate_check_summary"]["field"] == "json_object"

    receipt = {
        "validation_receipts": [
            {
                "name": "strict",
                "required": True,
                "passed": False,
                "exit_code": 2,
                "log_path": "strict.log",
            }
        ]
    }
    failed = capstone._producer_failure("task", "result.json", receipt, {})
    assert failed["field"] == "strict"
    flagged = capstone._producer_failure("task", "result.json", {"flagged_adversarial": True}, {})
    assert flagged["field"] == "flagged_adversarial"
    invalid = capstone._producer_failure(
        "task", "result.json", {"verdict_class": "disqualified"}, {"failures": ["x"]}
    )
    assert invalid["validation_failures"] == ["x"]


def test_null_classification_and_bad_prior_shape() -> None:
    """REQ-REPORT-7559; SCENARIO-REPORT-7559-FAILURES."""

    valid = {"evidence_state": "terminal", "verdict_class": "null", "advisory_only": False}
    assert (
        capstone.classify_terminal([valid], affected_complete=True, terminal_complete=True)[
            "verdict_class"
        ]
        == "null"
    )
    with pytest.raises(ValueError, match="prior failure"):
        capstone.reduce_prior_failures([{"id": "task"}], {})


@pytest.mark.parametrize("stdout", ["[]", '{"gates": null}'])
def test_publication_reader_rejects_non_object_shapes(
    monkeypatch: pytest.MonkeyPatch, stdout: str
) -> None:
    """REQ-REPORT-7559; SCENARIO-REPORT-7559-VALIDATION."""

    class Result:
        stderr = ""
        returncode = 0

    result = Result()
    result.stdout = stdout
    capstone.evaluate_publication_gates.cache_clear()
    monkeypatch.setattr(capstone.subprocess, "run", lambda *args, **kwargs: result)
    assert capstone.evaluate_publication_gates(Path("/tmp"))["unmet_gates"] == ["reader_failed"]
    capstone.evaluate_publication_gates.cache_clear()


def test_source_hash_defensive_shapes_and_note_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-7559; SCENARIO-REPORT-7559-VALIDATION."""

    assert (
        capstone._source_hashes_match({"source_artifact_hashes": {}}, capstone.REPO_ROOT) is False
    )
    assert (
        capstone._source_hashes_match({"source_artifact_hashes": [1]}, capstone.REPO_ROOT) is False
    )
    assert (
        capstone._source_hashes_match(
            {"source_artifact_hashes": [{"path": "missing", "sha256": None}]}, capstone.REPO_ROOT
        )
        is True
    )
    contract = capstone.load_contract(capstone.REPO_ROOT)
    evidence = capstone.collect_evidence(capstone.REPO_ROOT, contract["tasks"])
    monkeypatch.setattr(capstone, "NOTE_PATH", capstone.TEST_PATH)
    rows = capstone._source_hashes(capstone.REPO_ROOT, contract, evidence)
    assert any(row["path"] == capstone.TEST_PATH.as_posix() for row in rows)


def test_failure_rows_cover_fallback_contract_current_and_invalid() -> None:
    """REQ-REPORT-7559; SCENARIO-REPORT-7559-FAILURES."""

    blocked = {
        "task_id": "blocked",
        "artifact_path": None,
        "expected_path": "results/blocked.json",
        "evidence_state": "conductor_gate_blocked",
        "verdict_class": "blocked",
        "honest_verdict": "GATE_BLOCK",
        "advisory_only": False,
        "gate_check_summary": None,
    }
    assert capstone._normalized_blocked_failure(blocked)["field"] == "verdict_class"
    invalid = {
        **blocked,
        "task_id": "invalid",
        "evidence_state": "invalid",
        "verdict_class": "disqualified",
        "gate_check_summary": {"check": "bad"},
    }
    failures = capstone.failure_rows(
        {"comparison_passed": False, "selected_roadmap_path": "roadmap.yaml"},
        {"blocked": blocked, "invalid": invalid},
        {"required_checks_passed": False, "terminal_validation_passed": False},
    )
    assert [row["check"] for row in failures] == [
        "contract_authorities_agree",
        "current_required_checks_passed",
        "current_terminal_validation_passed",
        "producer_terminal_disposition",
        "bad",
    ]


def test_capstone_note_contains_all_scopes() -> None:
    """REQ-REPORT-7559; SCENARIO-REPORT-7559-BRANCHES."""

    text = capstone.capstone_markdown(capstone.build_artifact_for_test())
    assert "# V660 capstone" in text
    assert "exp7559-capstone" in text
    assert "exploratory_cached_count_learning" in text
    assert "fresh_injected_tool_decisions" in text
    assert "corrected_live_agent_generalization" in text
    assert "cpu_durability_and_board_continuity" in text
    assert "No activation, publication, submission, external contact, or push occurred." in text


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda value: value.update(schema="wrong"), "identity_invalid"),
        (
            lambda value: value.update(honest_verdict="blocked_without_prefix"),
            "terminal_identity_invalid",
        ),
        (lambda value: value.update(positive_claim=True), "current_claim_contract_invalid"),
        (
            lambda value: value["prior_failure_rows"][0].update(exact_text_match=True),
            "prior_failure_rows_invalid",
        ),
        (lambda value: value["continuation_rows"].pop(), "continuation_rows_invalid"),
        (
            lambda value: value["gate_check_summary"].update(passed=True),
            "terminal_reduction_invalid",
        ),
        (
            lambda value: value["acceptance_gate_results"].pop(),
            "acceptance_gates_invalid",
        ),
        (
            lambda value: value["sample_size_budget"].update(planned=13),
            "sample_size_budget_invalid",
        ),
        (
            lambda value: value["publication_gates"].update(
                paper_ready=not value["publication_gates"]["paper_ready"]
            ),
            "publication_gates_invalid",
        ),
        (
            lambda value: value["retained_source_adversarial_flags"].clear(),
            "source_adversarial_flags_invalid",
        ),
    ],
)
def test_additional_cold_reduction_mutations_fail(mutation: object, expected: str) -> None:
    """REQ-REPORT-7559; SCENARIO-REPORT-7559-VALIDATION."""

    artifact = capstone.build_artifact_for_test()
    assert callable(mutation)
    mutation(artifact)  # type: ignore[operator]
    assert expected in capstone.validate_artifact(artifact)


def test_terminal_requirement_independent_reducer_and_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-7559; SCENARIO-REPORT-7559-VALIDATION."""

    artifact = capstone.build_artifact_for_test()
    artifact["validation_receipts"] = [
        row
        for row in artifact["validation_receipts"]
        if row["name"] not in capstone.TERMINAL_CHECK_NAMES
    ]
    assert "terminal_validation_incomplete" in capstone.validate_artifact(
        artifact, require_terminal=True
    )

    good = capstone.build_artifact_for_test()
    assert capstone.independent_reduce(good) == []
    monkeypatch.setattr(
        capstone, "load_contract", lambda root: (_ for _ in ()).throw(ValueError("bad"))
    )
    assert "independent_reduction_failed" in capstone.independent_reduce(good)
