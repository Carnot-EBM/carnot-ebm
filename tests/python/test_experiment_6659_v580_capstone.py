"""Focused tests for the V580 terminal capstone.

Spec refs: REQ-REPORT-6659, REQ-REPORT-6659-PRECONDITIONS,
REQ-REPORT-6659-SCHEMAS, REQ-REPORT-6659-GATES,
REQ-REPORT-6659-ROWS, REQ-REPORT-6659-VERDICTS,
REQ-REPORT-6659-BRANCHES, REQ-REPORT-6659-RETIREMENT,
REQ-REPORT-6659-DISPOSITION, REQ-REPORT-6659-RECONCILIATION,
REQ-REPORT-6659-ATOMIC, SCENARIO-REPORT-6659-AVAILABILITY,
SCENARIO-REPORT-6659-EXACT-GATES,
SCENARIO-REPORT-6659-ROW-RECOMPUTATION,
SCENARIO-REPORT-6659-CLOSED-CLAIMS,
SCENARIO-REPORT-6659-RETIREMENT-AND-BOUNDARIES, and
SCENARIO-REPORT-6659-ATOMIC-TERMINAL.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6659_v580_capstone as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH

# The ARC source is 20 MB and expands during JSON parsing. It is built ONCE at import, so
# the per-test RSS guard measures test leaks rather than fixture loading -- moving the build
# into the fixture instead attributes ~798 MB to whichever test asks for it first, and the
# guard errors at teardown.
#
# BUT THE IMPORT MUST NOT RAISE (2026-08-27). `build_artifact` reads the roadmap and can
# fail. An exception at module scope is a COLLECTION error, and pytest answers one by
# aborting the ENTIRE run: `Interrupted: 1 error during collection` against 57,917 collected
# tests. One stale capstone therefore took down the whole repository suite, which failed
# every conductor task shelling out to `pytest tests/python` -- exp6682's
# `verification_failure` among them. So the build happens at import for the RSS reason, the
# failure is captured rather than raised, and the fixture re-raises it. Both properties kept.
_STORED_ARTIFACT: dict[str, object] | None = None
_BUILD_ERROR: Exception | None = None

try:
    _STORED_ARTIFACT = mod.build_artifact(
        REPO,
        run_date="20260827",
        duration_s=1.25,
        tests_run=[{"command": "focused-exp6659", "exit_code": 0, "summary": "passed"}],
    )
except Exception as exc:  # noqa: BLE001 - confined to this file, re-raised in the fixture
    _BUILD_ERROR = exc


@pytest.fixture(scope="module")
def artifact() -> dict[str, object]:
    """Build one deterministic report from the stored V580 evidence."""

    if _BUILD_ERROR is not None:
        raise _BUILD_ERROR
    assert _STORED_ARTIFACT is not None
    return _STORED_ARTIFACT


def test_req_report_6659_spec_declares_terminal_contract() -> None:
    """REQ-REPORT-6659: the reporting spec owns every durable field."""

    section = SPEC.read_text(encoding="utf-8").split("REQ-REPORT-6659", 1)[1]
    for marker in (
        "SCENARIO-REPORT-6659-AVAILABILITY",
        "SCENARIO-REPORT-6659-EXACT-GATES",
        "SCENARIO-REPORT-6659-ROW-RECOMPUTATION",
        "SCENARIO-REPORT-6659-CLOSED-CLAIMS",
        "SCENARIO-REPORT-6659-RETIREMENT-AND-BOUNDARIES",
        "SCENARIO-REPORT-6659-ATOMIC-TERMINAL",
        mod.INFERENCE_SUBSTRATE,
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert marker in section
    assert all(field in section for field in mod.REQUIRED_FIELDS)


def test_scenario_report_6659_preserves_availability_and_schema(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-REPORT-6659-AVAILABILITY keeps absence distinct from zero."""

    rows = artifact["artifact_availability_rows"]
    assert len(rows) == 12
    assert [row["experiment_number"] for row in rows] == list(range(6647, 6659))
    by_number = {row["experiment_number"]: row for row in rows}
    assert by_number[6652]["missing"] is True
    assert by_number[6652]["present"] is False
    assert by_number[6652]["artifact_sha256"] is None
    assert by_number[6652]["blocked"] is True
    assert by_number[6651]["schema_state"] == "valid_gate_block"
    assert by_number[6658]["schema_state"] == "valid_gate_block"
    assert by_number[6647]["hard_cap"] is True
    assert by_number[6649]["hard_cap"] is True
    for number in (6647, 6648, 6649, 6650, 6653, 6654, 6655, 6656, 6657):
        assert by_number[number]["schema_state"] == "valid"
        assert by_number[number]["artifact_sha256"].startswith("sha256:")
        assert by_number[number]["schema_errors"] == []


def test_scenario_report_6659_recomputes_exact_roadmap_gates(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-REPORT-6659-EXACT-GATES retains exact fields and values."""

    rows = artifact["gate_recomputation_rows"]
    assert len(rows) == 9
    assert all(row["owner_declares_exact_field"] for row in rows)
    assert all(row["contract_state"] == "valid" for row in rows)
    headroom = next(row for row in rows if row["artifact_field"] == "regeneration_headroom_count")
    assert headroom["actual"] == 2
    assert headroom["operator"] == ">="
    assert headroom["expected"] == 8
    assert headroom["recomputed_passed"] is False
    intervention = next(row for row in rows if row["consumer"].startswith("exp6652-"))
    assert intervention["actual"] is None
    assert intervention["field_present"] is False
    assert intervention["recomputed_passed"] is False
    schedule = next(row for row in rows if row["consumer"].startswith("exp6658-"))
    assert schedule["actual"] is False
    assert schedule["recomputed_passed"] is False


def test_scenario_report_6659_rebuilds_all_headlines_from_rows(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-REPORT-6659-ROW-RECOMPUTATION rebuilds every branch metric."""

    headline = artifact["headline_recomputation"]
    assert headline["admission"] == {
        "task_owned_checks_passed": 13,
        "task_owned_checks_total": 13,
        "mandated_model_families_admitted": 3,
        "mandated_model_families_total": 3,
    }
    assert headline["corpus"] == {
        "candidate_row_count": 48,
        "parsed_row_count": 10,
        "parse_failure_count": 38,
        "direct_exact_success_count": 8,
        "direct_exact_success_rate": pytest.approx(8 / 48),
        "regeneration_headroom_count": 2,
    }
    verifier = {row["unit_id"]: row for row in headline["verifier_units"]}
    assert verifier["one_step"]["catch_count"] == 0
    assert verifier["two_steps"]["catch_count"] == 8
    assert verifier["two_steps"]["false_reject_count"] == 0
    assert verifier["full_remaining_suffix"]["false_reject_count"] == 2
    assert headline["suffix_regeneration"]["denominator"] is None
    assert headline["memory"]["producer_order_delta_mean"] == pytest.approx(1 / 18)
    assert headline["memory"]["independent_order_delta_interval_95"][0] < 0
    assert headline["memory"]["independent_order_delta_interval_95"][1] > 0
    assert headline["arc"]["paired_action_row_count"] == 4916
    assert headline["arc"]["changed_action_count"] == 551
    assert headline["arc"]["changed_action_exact_outcome_count"] == 0
    assert headline["arc"]["arc_solve_credit"] == 0
    assert headline["exact_reference"]["supported_fixture_count"] == 12
    assert headline["exact_reference"]["tests_all_passed"] is False
    assert headline["schedule"]["denominator"] is None
    diagnostics = headline["diagnostics"]
    assert diagnostics["parser_failures"][0]["count"] == 38
    assert {row["metric"] for row in diagnostics["missing_denominators"]} == {
        "suffix_regeneration_comparison",
        "thermodynamic_schedule_comparison",
    }
    assert diagnostics["null_only_groups"] == [
        {
            "source": "experiment_6650",
            "row_type": "rejected_pair",
            "count": 40,
            "disposition": "explicit_non_pairable_rows_not_zero_measurements",
        }
    ]
    assert diagnostics["sign_flips"][0]["observed_delta"] == -5
    assert {row["kind"] for row in diagnostics["contradictions"]} == {
        "producer_positive_vs_independent_null",
        "reference_rows_pass_vs_test_gate_block",
    }


def test_scenario_report_6659_keeps_closed_claims_and_branches_independent(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-REPORT-6659-CLOSED-CLAIMS prevents pooled promotion."""

    claims = artifact["claim_classification_rows"]
    assert all(row["verdict_class"] in mod.CLOSED_VERDICT_CLASSES for row in claims)
    by_id = {row["claim_id"]: row for row in claims}
    assert by_id["direct_corpus"]["verdict_class"] == "positive"
    assert by_id["verifier_unit"]["verdict_class"] == "positive"
    assert by_id["memory_general_benefit"]["verdict_class"] == "null"
    assert by_id["arc_live_benefit"]["verdict_class"] == "blocked"
    assert by_id["arc_live_benefit"]["arc_solve_claim"] is False
    assert by_id["ising_schedule"]["verdict_class"] == "blocked"
    # REQ-CONDUCTOR-VERDICT-4: the finished capstone declares null, not partial.
    assert artifact["verdict_class"] == "null"
    assert artifact["status"] == "complete_terminal_null"
    assert artifact["honest_verdict"].startswith("complete_null:")
    assert "pooled" not in artifact

    branches = {row["branch"]: row for row in artifact["branch_summary_rows"]}
    assert set(branches) == {"admission_and_verification", "memory", "arc", "ising"}
    assert branches["admission_and_verification"]["verdict_class"] == "partial"
    assert branches["memory"]["verdict_class"] == "null"
    assert branches["arc"]["verdict_class"] == "blocked"
    assert branches["ising"]["verdict_class"] == "blocked"
    assert all("lesson" in row for row in branches.values())

    assert (
        mod.classify_source_claim(
            present=True,
            schema_valid=True,
            status="complete_positive",
            declared_class="positive",
            verifier_is_oracle=True,
        )
        == "circular_positive"
    )
    assert (
        mod.classify_source_claim(
            present=False,
            schema_valid=False,
            status=None,
            declared_class=None,
            verifier_is_oracle=False,
        )
        == "blocked"
    )
    assert (
        mod.classify_source_claim(
            present=True,
            schema_valid=False,
            status="complete",
            declared_class="positive",
            verifier_is_oracle=False,
        )
        == "disqualified"
    )


def test_scenario_report_6659_retires_repeated_failures_and_preserves_boundaries(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-REPORT-6659-RETIREMENT-AND-BOUNDARIES narrows follow-ups."""

    retirement = artifact["prior_failure_retirement_rows"]
    assert retirement
    repeated = [row for row in retirement if row["retirement_recommended"]]
    assert {row["task_id"] for row in repeated} >= {
        "exp6651-failure-localized-suffix-regeneration",
        "exp6652-constraint-intervention-audit",
        "exp6655-repair-memory-safety-audit",
        "exp6656-arc-trace-automaton-live-loo",
        "exp6658-thermodynamic-schedule-ab",
    }
    assert all(row["disposition"].startswith("recommend_") for row in repeated)
    gaps = {row["gap"]: row for row in artifact["prd_gap_matrix"]}
    assert set(gaps) == {
        "FR-11 autonomous self-learning",
        "FR-12 verifiable reasoning",
        "hardware acceleration",
    }
    assert gaps["FR-11 autonomous self-learning"]["movement"] == "narrow_fixture_only"
    assert gaps["hardware acceleration"]["movement"] == "not_advanced"
    boundaries = artifact["hardware_claim_boundary"]
    assert boundaries["measured_local_paths"] == [
        "dual_RTX_3090_CUDA_GGUF_admission_and_generation",
        "CPU_no_LLM_verifier_memory_ARC_and_Ising_replay",
    ]
    assert set(boundaries["unsupported_claims"]) == {
        "KV260_or_other_FPGA_execution",
        "TSU_Extropic_execution",
        "photonic_execution",
        "hardware_speedup",
        "production_schedule_improvement",
    }
    dispositions = {
        row["component"]: row["disposition"] for row in artifact["architecture_disposition"]
    }
    assert dispositions["two_step_advisory_verifier_unit"] == "adopt"
    assert dispositions["failure_localized_suffix_regeneration"] == "retire"
    assert dispositions["prospective_repair_memory"] == "keep experimental"
    assert dispositions["ARC_trace_automaton_supervisor"] == "narrow"
    assert dispositions["thermodynamic_schedule"] == "defer"


def test_req_report_6659_provenance_reconciliation_and_protection(
    artifact: dict[str, object],
) -> None:
    """REQ-REPORT-6659-PRECONDITIONS records every source and deferred document."""

    assert set(mod.REQUIRED_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_FIELDS) <= set(artifact["field_provenance"])
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["random_seed"] == 6659
    protected = artifact["protected_files_unchanged"]
    assert protected["all_unchanged"] is True
    assert {row["path"] for row in protected["rows"]} == {
        "research-roadmap.yaml",
        "scripts/research_conductor.py",
    }
    receipts = {row["path"]: row for row in artifact["reconciliation_receipts"]}
    assert receipts[mod.SPEC_RELATIVE_PATH.as_posix()]["action"] == "implemented"
    for path in ("_bmad/traceability.md", "ops/status.md", "ops/changelog.md"):
        assert receipts[path]["action"] == "deferred_to_conductor"
    assert receipts["ops/exclusion_manifest.yaml"]["action"] == "recommendations_only"
    assert artifact["preconditions_checked"]["expected_upstream_count"] == 12
    assert artifact["preconditions_checked"]["missing_upstream_ids"] == [
        "exp6652-constraint-intervention-audit"
    ]

    command_receipts = {row["command"]: row for row in mod.DEFAULT_TESTS_RUN}
    required_command_fragments = {
        ".venv/bin/pytest tests/python -q": 3,
        "scripts/audit_roadmap_gates.py research-roadmap.yaml": 1,
        "scripts/validate_prior_failures.py research-roadmap.yaml": 0,
        "scripts/exclusion_manifest_lint.py research-roadmap.yaml": 0,
        "scripts/harness_fit_lint.py research-roadmap.yaml": 1,
        "Roadmap.model_validate": 0,
        "scripts/validate-reconciliation.sh": 1,
        "carnot.experiment_6659_v580_capstone --date 20260827": 0,
        "carnot.experiment_6659_v580_capstone --validate": 0,
        "git status --short": 0,
    }
    for fragment, expected_exit in required_command_fragments.items():
        matches = [row for command, row in command_receipts.items() if fragment in command]
        assert len(matches) == 1
        assert matches[0]["exit_code"] == expected_exit


def test_req_report_6659_validation_checksum_and_atomic_write(
    artifact: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6659-ATOMIC-TERMINAL rejects each durable mutation."""

    mod.validate_artifact(artifact)
    assert mod.unwrap_value({"value": 2, "principle": "owner"}) == 2
    ordinary = {"value": 2, "task": "ordinary"}
    assert mod.unwrap_value(ordinary) is ordinary

    bad = deepcopy(artifact)
    bad["honest_verdict"] = "mutated"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(bad)
    for field, value, message in (
        ("verdict_class", "positive", "capstone verdict"),
        # REQ-CONDUCTOR-VERDICT-4 / SCENARIO-CONDUCTOR-VERDICT-5: partial is
        # the may-retry class and a finished capstone may not declare it.
        ("verdict_class", "partial", "capstone verdict"),
        ("inference_substrate", "wrong", "inference substrate"),
        ("verifier_is_oracle", True, "oracle"),
        ("random_seed", 1, "random seed"),
    ):
        changed = deepcopy(artifact)
        changed[field] = value
        changed["reproducibility_checksum"] = mod.reproducibility_checksum(changed)
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(changed)

    missing = deepcopy(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="required fields"):
        mod.validate_artifact(missing)
    wrong_rows = deepcopy(artifact)
    wrong_rows["artifact_availability_rows"].pop()
    wrong_rows["reproducibility_checksum"] = mod.reproducibility_checksum(wrong_rows)
    with pytest.raises(ValueError, match="availability"):
        mod.validate_artifact(wrong_rows)
    wrong_class = deepcopy(artifact)
    wrong_class["claim_classification_rows"][0]["verdict_class"] = "outside"
    wrong_class["reproducibility_checksum"] = mod.reproducibility_checksum(wrong_class)
    with pytest.raises(ValueError, match="claim class"):
        mod.validate_artifact(wrong_class)
    changed_protected = deepcopy(artifact)
    changed_protected["protected_files_unchanged"]["all_unchanged"] = False
    changed_protected["reproducibility_checksum"] = mod.reproducibility_checksum(changed_protected)
    with pytest.raises(ValueError, match="protected"):
        mod.validate_artifact(changed_protected)
    wrong_gates = deepcopy(artifact)
    wrong_gates["gate_recomputation_rows"].pop()
    wrong_gates["reproducibility_checksum"] = mod.reproducibility_checksum(wrong_gates)
    with pytest.raises(ValueError, match="gate matrix"):
        mod.validate_artifact(wrong_gates)
    oracle_positive = deepcopy(artifact)
    oracle_positive["claim_classification_rows"][0].update(
        {"verdict_class": "positive", "verifier_is_oracle": True}
    )
    oracle_positive["reproducibility_checksum"] = mod.reproducibility_checksum(oracle_positive)
    with pytest.raises(ValueError, match="oracle evidence"):
        mod.validate_artifact(oracle_positive)
    wrong_branches = deepcopy(artifact)
    wrong_branches["branch_summary_rows"].pop()
    wrong_branches["reproducibility_checksum"] = mod.reproducibility_checksum(wrong_branches)
    with pytest.raises(ValueError, match="branch matrix"):
        mod.validate_artifact(wrong_branches)
    wrong_provenance = deepcopy(artifact)
    wrong_provenance["field_provenance"].pop("status")
    wrong_provenance["reproducibility_checksum"] = mod.reproducibility_checksum(wrong_provenance)
    with pytest.raises(ValueError, match="field provenance"):
        mod.validate_artifact(wrong_provenance)
    coerced_denominator = deepcopy(artifact)
    coerced_denominator["headline_recomputation"]["suffix_regeneration"]["denominator"] = 0
    coerced_denominator["reproducibility_checksum"] = mod.reproducibility_checksum(
        coerced_denominator
    )
    with pytest.raises(ValueError, match="denominator"):
        mod.validate_artifact(coerced_denominator)

    path = tmp_path / "nested" / "capstone.json"
    mod.write_artifact_atomic(path, artifact)
    assert json.loads(path.read_text(encoding="utf-8")) == artifact
    assert not list(path.parent.glob("*.tmp"))

    failed_path = tmp_path / "failed" / "capstone.json"
    monkeypatch.setattr(mod.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("stop")))
    with pytest.raises(OSError, match="stop"):
        mod.write_artifact_atomic(failed_path, artifact)
    assert not list(failed_path.parent.glob("*.tmp"))


def test_req_report_6659_input_failures_and_cli(
    artifact: dict[str, object],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-6659-ATOMIC keeps malformed inputs and CLI failures explicit."""

    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact root"):
        mod.read_json(invalid_json)
    roadmap = tmp_path / mod.ROADMAP_RELATIVE_PATH
    roadmap.write_text("milestone: wrong\ntasks: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="milestone"):
        mod.load_roadmap(tmp_path)
    roadmap.write_text(f"milestone: '{mod.MILESTONE}'\ntasks: wrong\n", encoding="utf-8")
    with pytest.raises(ValueError, match="task list"):
        mod.load_roadmap(tmp_path)
    with pytest.raises(ValueError, match="invalid task id"):
        mod._experiment_number("wrong")
    assert mod.compare_gate(2, ">=", 1) is True
    assert mod.compare_gate(1, "<=", 2) is True
    assert mod.compare_gate(None, ">=", 1) is False
    assert mod.compare_gate(1, "unknown", 1) is False
    assert mod.declared_artifact_fields({"prompt": "no contract"}) == set()
    malformed_gate_block = {
        "schema": "blocked_gate_check_v1",
        "status": "wrong",
        "honest_verdict": "wrong",
        "gates_evaluated": None,
    }
    assert set(mod.validate_gate_block_artifact(malformed_gate_block)) == {
        "gate_block_required_fields_missing",
        "gate_block_status_mismatch",
        "gate_block_verdict_mismatch",
        "gate_block_gate_rows_missing",
    }
    assert mod.schema_validation(9999, {}, present=True) == (
        "validator_missing",
        ["owner_validator_missing"],
    )

    class InvalidValidator:
        @staticmethod
        def validate_artifact(_: object) -> None:
            raise ValueError("invalid source")

    monkeypatch.setattr(mod.importlib, "import_module", lambda _: InvalidValidator)
    state, errors = mod.schema_validation(6647, {}, present=True)
    assert state == "invalid"
    assert errors == ["ValueError: invalid source"]
    assert mod._order_interval([]) is None
    assert (
        mod.classify_source_claim(
            present=True,
            schema_valid=True,
            status="blocked_gate",
            declared_class=None,
            verifier_is_oracle=False,
        )
        == "blocked"
    )
    assert (
        mod.classify_source_claim(
            present=True,
            schema_valid=True,
            status="complete",
            declared_class="positive",
            verifier_is_oracle=False,
        )
        == "positive"
    )
    assert (
        mod.classify_source_claim(
            present=True,
            schema_valid=True,
            status="complete",
            declared_class="outside",
            verifier_is_oracle=False,
        )
        == "disqualified"
    )
    assert (
        mod.classify_source_claim(
            present=True,
            schema_valid=True,
            status="complete_partial",
            declared_class=None,
            verifier_is_oracle=False,
        )
        == "partial"
    )
    assert (
        mod.classify_source_claim(
            present=True,
            schema_valid=True,
            status="complete",
            declared_class=None,
            verifier_is_oracle=False,
        )
        == "null"
    )

    output = tmp_path / "cli.json"
    assert mod.main(["--date", "20260827", "--repo-root", str(REPO), "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "complete_terminal_null"
    assert mod.main(["--validate", "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1]) == {"valid": True}
    missing_output = tmp_path / "missing.json"
    assert mod.main(["--validate", "--output", str(missing_output)]) == 1
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["valid"] is False
    invalid_output = tmp_path / "invalid-output.json"
    invalid_output.write_text("not-json", encoding="utf-8")
    assert mod.main(["--validate", "--output", str(invalid_output)]) == 1
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["valid"] is False

    assert mod.validate_gate_block_artifact({}) == ["gate_block_schema_mismatch"]
    assert mod.schema_validation(6652, {}, present=False) == ("missing", [])
    assert mod.sha256_file(tmp_path / "absent") is None
