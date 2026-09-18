"""Tests for the preserved V648 Ising evidence reduction.

Spec refs: REQ-SAMPLER-7392 and SCENARIO-SAMPLER-7392-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7392_v648_ising_reduction as exp


ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.memory_watchdog_skip


@pytest.fixture(scope="module")
def preserved() -> dict[str, object]:
    """Load and authenticate the preserved laws, audit, logs, and traces once."""

    return exp.load_preserved_inputs(ROOT)


@pytest.fixture(scope="module")
def reduction(preserved: dict[str, object]) -> dict[str, object]:
    """Reduce the authentic archive without launching a replacement chain."""

    return exp.reduce_preserved_evidence(
        preserved["law_artifact"],
        preserved["audit_artifact"],
        preserved["trace_records"],
    )


def _passing_receipts() -> list[dict[str, object]]:
    """Return one successful receipt for every current required command."""

    return [
        {
            "name": name,
            "exit_code": 0,
            "passed": True,
            "timed_out": False,
            "command_argv": [name],
            "command_environment": {},
            "scope": "unit_fixture",
            "duration_s": 0.01,
            "log_sha256": "sha256:" + "a" * 64,
        }
        for name in (*exp.REQUIRED_CHECK_NAMES, *exp.TERMINAL_CHECK_NAMES)
    ]


def test_support_enumeration_separates_three_clamp_classes(
    preserved: dict[str, object],
) -> None:
    """SCENARIO-SAMPLER-7392-SUPPORT: positive minimum energy keeps a law."""

    law = preserved["law_artifact"]
    rows = exp.enumerate_support_rows(law["frozen_formulas"])

    assert len(rows) == 24 * 3 * 3
    counts = {
        name: sum(row["support_class"] == name for row in rows) for name in exp.SUPPORT_CLASSES
    }
    assert counts == {
        "contradictory_clamp_empty_support": 27,
        "satisfiable_clamp_no_zero_energy_state": 54,
        "ordinary_nonempty_support": 135,
    }
    positive_minimum = [
        row for row in rows if row["support_class"] == "satisfiable_clamp_no_zero_energy_state"
    ]
    assert all(row["law_defined"] is True for row in positive_minimum)
    assert all(row["support_size"] > 0 for row in positive_minimum)
    assert all(row["normalizer"] > 0.0 for row in positive_minimum)
    assert all(row["minimum_source_energy"] > 0.0 for row in positive_minimum)
    assert max(row["max_abs_energy_residual"] for row in rows) <= 1e-12
    assert max(row["exact_total_variation"] for row in rows) <= 1e-10


def test_contradictory_support_has_null_quality_not_fake_zero(
    reduction: dict[str, object],
) -> None:
    """SCENARIO-SAMPLER-7392-EMPTY: undefined laws have null statistics."""

    empty = [
        row
        for row in reduction["sample_quality_rows"]
        if row["support_status"] == "empty_conditioned_support"
    ]

    assert len([row for row in empty if row["row_type"] == "chain_quality"]) == 72
    assert len([row for row in empty if row["row_type"] == "cell_quality"]) == 18
    assert all(row["sample_size"] == 0 for row in empty)
    assert all(row["max_observable_error"] is None for row in empty)
    assert all(row["minimum_effective_sample_size"] is None for row in empty)
    assert all(row["autocorrelation_defined"] is False for row in empty)
    assert all(row["qualified"] is False for row in empty)


def test_authentic_archive_recomputes_every_chain_and_cell(
    preserved: dict[str, object], reduction: dict[str, object]
) -> None:
    """REQ-SAMPLER-7392-TRACES: every preserved unit is reduced from raw bytes."""

    assert all(row["passed"] is True for row in preserved["preconditions"])
    assert len(preserved["validation_logs"]) == 14
    assert reduction["passed"] is True
    assert reduction["errors"] == []
    assert reduction["trace_record_count"] == 540
    assert reduction["chain_row_count"] == 612
    assert reduction["cell_row_count"] == 153
    assert len(reduction["sample_quality_rows"]) == 765
    assert reduction["exact_law_conclusions"]["defined_row_count"] == 189
    assert reduction["exact_law_conclusions"]["undefined_row_count"] == 27
    assert reduction["exact_law_conclusions"]["law_preservation_confirmed"] is True
    finite = reduction["finite_chain_conclusions"]
    assert finite["original_all_cell_gate_passed"] is False
    assert finite["nonempty_source_cell_count"] == 126
    assert finite["nonempty_source_cells_qualified"] == 126
    assert finite["empty_source_cell_count"] == 18
    assert finite["prospective_pass_claimed"] is False

    again = exp.reduce_preserved_evidence(
        preserved["law_artifact"],
        preserved["audit_artifact"],
        preserved["trace_records"],
    )
    assert again["reduction_checksum"] == reduction["reduction_checksum"]
    assert again == reduction


def test_historical_validator_rejection_and_gates_are_preserved(
    preserved: dict[str, object], reduction: dict[str, object]
) -> None:
    """SCENARIO-SAMPLER-7392-PRESERVE: V647 remains disqualified history."""

    audit = preserved["audit_artifact"]
    assert audit["execution_venue"] == "host_cpu"
    assert audit["verdict_class"] == "disqualified"
    assert audit["flagged_adversarial"] is True
    assert "execution_venue_invalid" in preserved["historical_validator_errors"]

    artifact = exp.build_artifact_for_test(preserved, reduction, _passing_receipts())
    assert artifact["execution_venue"] == "host"
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["ising_reduction_complete_score"] == 1
    assert artifact["law_preservation_confirmed_score"] == 1
    assert artifact["promotion_score"] == 0
    assert artifact["flagged_adversarial"] is False
    assert (
        artifact["original_gate_results"]["acceptance_gate_results"]
        == audit["acceptance_gate_results"]
    )
    assert artifact["original_gate_results"]["gate_check_summary"] == audit["gate_check_summary"]
    assert artifact["historical_trace_manifest"]["flagged_adversarial"] is True
    assert artifact["historical_trace_manifest"]["eligible_for_current_readiness"] is False
    assert exp.validate_artifact(artifact, ROOT) == []

    changed = deepcopy(artifact)
    changed["original_gate_results"]["acceptance_gate_results"][7]["observed"] = True
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    assert "original_gate_results_changed" in exp.validate_artifact(changed, ROOT)


def test_archive_attacks_fail_closed(preserved: dict[str, object]) -> None:
    """SCENARIO-SAMPLER-7392-ATTACKS: six evidence mutations are detected."""

    controls = exp.run_reducer_attack_controls(
        preserved["law_artifact"],
        preserved["audit_artifact"],
        preserved["trace_records"],
    )

    assert [row["control_id"] for row in controls] == [
        "omitted_cell",
        "inconsistent_clamp",
        "stale_source_version",
        "altered_beta",
        "forged_sample",
        "missing_chain_cost",
    ]
    assert all(row["detected"] is True for row in controls)
    assert all(row["expected_detector"] in row["observed_errors"] for row in controls)


def test_structural_validator_names_additional_roster_and_trace_attacks(
    preserved: dict[str, object],
) -> None:
    """SCENARIO-SAMPLER-7392-ATTACKS: secondary corruptions also fail closed."""

    law = preserved["law_artifact"]
    audit = preserved["audit_artifact"]
    traces = preserved["trace_records"]

    short_law = dict(law)
    short_law["frozen_formulas"] = list(law["frozen_formulas"])[1:]
    errors = exp.validate_preserved_structure(short_law, audit, traces)
    assert "formula_roster_incomplete" in errors
    assert "stale_source_version" in errors
    assert "forged_sample" in errors

    hostile_audit = dict(audit)
    hostile_chains = list(audit["per_chain_results"])[1:]
    changed_chain = dict(hostile_chains[0])
    changed_chain["condition"] = "unknown_arm"
    hostile_chains[0] = changed_chain
    hostile_audit["per_chain_results"] = hostile_chains
    hostile_gates = deepcopy(audit["acceptance_gate_results"])
    next(row for row in hostile_gates if row["check"] == "all_source_cells_qualified")[
        "observed"
    ] = True
    hostile_audit["acceptance_gate_results"] = hostile_gates

    hostile_traces = list(traces)
    duplicate = dict(hostile_traces[0])
    changed_trace = dict(hostile_traces[0])
    changed_trace["cell_id"] = "forged-cell"
    changed_trace["source_hash"] = "sha256:" + "f" * 64
    changed_trace["beta"] = 9.0
    hostile_traces.extend([duplicate, changed_trace])
    errors = exp.validate_preserved_structure(law, hostile_audit, hostile_traces)
    assert "chain_roster_incomplete" in errors
    assert "unknown_chain_condition" in errors
    assert "duplicate_trace" in errors
    assert "trace_without_chain" in errors
    assert "trace_roster_incomplete" in errors
    assert "altered_beta" in errors
    assert "stale_source_version" in errors
    assert "original_gate_results_changed" in errors


def test_metric_replay_detects_changed_public_summaries(preserved: dict[str, object]) -> None:
    """REQ-SAMPLER-7392-TRACES: raw samples override changed public summaries."""

    audit = dict(preserved["audit_artifact"])
    chains = list(audit["per_chain_results"])
    changed_chain = deepcopy(chains[0])
    changed_chain["observable_results"][0]["observed_mean"] += 0.25
    chains[0] = changed_chain
    audit["per_chain_results"] = chains
    cells = list(audit["cell_results"])
    changed_cell = dict(cells[0])
    changed_cell["qualified"] = not changed_cell["qualified"]
    cells[0] = changed_cell
    audit["cell_results"] = cells

    reduction = exp.reduce_preserved_evidence(
        preserved["law_artifact"], audit, preserved["trace_records"]
    )

    assert any(error.startswith("chain_metric_mismatch:") for error in reduction["errors"])
    assert any(error.startswith("cell_metric_mismatch:") for error in reduction["errors"])
    assert reduction["passed"] is False


def test_validator_rejects_current_declaration_and_evidence_drift(
    preserved: dict[str, object], reduction: dict[str, object]
) -> None:
    """REQ-SAMPLER-7392-ARTIFACT: current declarations and rows fail closed."""

    artifact = exp.build_artifact_for_test(preserved, reduction, _passing_receipts())

    venue = deepcopy(artifact)
    venue["execution_venue"] = "host_cpu"
    venue["reproducibility_checksum"] = exp.reproducibility_checksum(venue)
    assert "execution_venue_invalid" in exp.validate_artifact(venue, ROOT)

    omitted = deepcopy(artifact)
    omitted["support_rows"].pop()
    omitted["reproducibility_checksum"] = exp.reproducibility_checksum(omitted)
    errors = exp.validate_artifact(omitted, ROOT)
    assert "support_row_roster_invalid" in errors
    assert "stored_reduction_mismatch" in errors

    checksum = deepcopy(artifact)
    checksum["reproducibility_checksum"] = "sha256:" + "0" * 64
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(checksum, ROOT)


def test_validator_covers_all_closed_declarations_and_blocked_guards(
    tmp_path: Path, preserved: dict[str, object], reduction: dict[str, object]
) -> None:
    """REQ-SAMPLER-7392-ARTIFACT: every closed declaration has a rejecting path."""

    artifact = exp.build_artifact_for_test(preserved, reduction, _passing_receipts())
    invalid = dict(artifact)
    invalid.update(
        {
            "schema": "wrong",
            "MODEL_SPECS": ["forbidden/model"],
            "model_invoked": True,
            "invocation_counts": {"current": {}},
            "inference_substrate_class": "gpu_full_model",
            "execution_venue": "host_cpu",
            "verdict_class": "unknown",
            "promotion_score": 1,
            "verifier_is_oracle": False,
            "field_principles": {},
            "support_rows": [],
            "sample_quality_rows": [],
            "archived_reduction_checksum": "sha256:" + "0" * 64,
            "original_gate_results": {},
            "independent_reduction": {},
            "flagged_adversarial": True,
            "ising_reduction_complete_score": 1,
            "law_preservation_confirmed_score": 1,
        }
    )
    invalid.pop("historical_trace_manifest")
    errors = exp.validate_artifact(invalid, ROOT)
    assert {
        "identity_mismatch",
        "current_model_declaration_invalid",
        "current_invocation_counts_invalid",
        "substrate_class_invalid",
        "execution_venue_invalid",
        "verdict_class_invalid",
        "promotion_nonzero",
        "circularity_declaration_invalid",
        "field_principles_incomplete",
        "required_artifact_field_missing",
        "support_row_roster_invalid",
        "sample_quality_row_roster_invalid",
        "cold_support_rows_mismatch",
        "cold_sample_quality_rows_mismatch",
        "cold_reduction_checksum_mismatch",
        "original_gate_results_changed",
        "stored_reduction_mismatch",
        "score_mismatch",
        "adversarial_readiness_nonzero",
        "reproducibility_checksum_mismatch",
    } <= set(errors)

    unavailable = exp.validate_artifact(artifact, tmp_path)
    assert "preserved_evidence_unavailable" in unavailable

    failed = exp.precondition_row("missing", "upstream", "bytes", "present", "missing")
    blocked = exp.build_blocked_artifact_for_test([failed])
    hostile_blocked = dict(blocked)
    hostile_blocked["rows"] = [{"row_type": "forbidden"}]
    hostile_blocked["validation_receipts"] = [{"name": "forbidden"}]
    hostile_blocked["gate_check_summary"] = {}
    hostile_blocked["ising_reduction_complete_score"] = 1
    errors = exp.validate_artifact(hostile_blocked, ROOT)
    assert "blocked_artifact_has_dependent_work" in errors
    assert "blocked_gate_summary_missing" in errors
    assert "blocked_scores_nonzero" in errors


def test_blocked_classification_names_the_exact_failed_precondition() -> None:
    """REQ-SAMPLER-7392-PREFLIGHT: external absence is blocked, not partial."""

    failed = exp.precondition_row(
        "trace_archive_bytes",
        "results/raw/missing.jsonl.gz",
        "bytes",
        "readable_nonempty_bytes",
        "missing",
    )
    artifact = exp.build_blocked_artifact_for_test([failed])

    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["gate_check_summary"]["first_failure"] == failed
    assert artifact["ising_reduction_complete_score"] == 0
    assert artifact["law_preservation_confirmed_score"] == 0
    assert artifact["promotion_score"] == 0
    assert artifact["rows"] == []
    assert exp.validate_artifact(artifact, ROOT) == []
    assert exp.independent_reduce(artifact)["verdict_class"] == "blocked"


def test_scoped_plan_has_private_paths_and_no_numbered_e2e(tmp_path: Path) -> None:
    """REQ-SAMPLER-7392-VALIDATION: Exp7358 provides the exact affected plan."""

    commands = exp.build_validation_plan(ROOT, tmp_path)

    assert [command.name for command in commands] == list(exp.REQUIRED_CHECK_NAMES)
    assert exp.validate_validation_plan(ROOT, commands) == []
    assert not any(
        argument.rstrip("/") in {"tests", "tests/python"}
        for command in commands
        for argument in command.argv
    )
    report = next(
        command for command in commands if command.name == "changed_module_coverage_report"
    )
    assert "COVERAGE_FILE" in dict(report.command_environment)

    expanded = [*commands, commands[0]]
    assert "duplicate_command:worktree_imports" in exp.validate_validation_plan(ROOT, expanded)

    assert exp._gate("minimum", "unit", 2, 3, operator=">=", terminal_blocking=False)["passed"]
    with pytest.raises(ValueError, match="unsupported_gate_operator"):
        exp._gate("bad", "unit", 1, 1, operator="!=", terminal_blocking=False)


def test_input_parsing_and_missing_preconditions_fail_before_reduction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-SAMPLER-7392-PREFLIGHT: malformed or missing inputs stop before replay."""

    array_path = tmp_path / "array.json"
    array_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="json_object_required"):
        exp._read_object(array_path)

    monkeypatch.setattr(exp, "REQUIRED_SOURCE_PATHS", (Path("missing.json"),))
    checks, hashes = exp.collect_preconditions(tmp_path)
    assert hashes == {}
    assert checks[0]["observed"] == "missing"
    with pytest.raises(ValueError, match="preserved_input_precondition_failed"):
        exp.load_preserved_inputs(tmp_path)

    law_path = Path("law.json")
    audit_path = Path("audit.json")
    trace_path = Path("trace.gz")
    monkeypatch.setattr(exp, "LAW_PATH", law_path)
    monkeypatch.setattr(exp, "AUDIT_PATH", audit_path)
    monkeypatch.setattr(exp, "TRACE_PATH", trace_path)
    monkeypatch.setattr(exp, "REQUIRED_SOURCE_PATHS", (law_path, audit_path, trace_path))
    (tmp_path / law_path).write_text("{bad", encoding="utf-8")
    (tmp_path / audit_path).write_text("{}", encoding="utf-8")
    (tmp_path / trace_path).write_bytes(b"trace")
    checks, _hashes = exp.collect_preconditions(tmp_path)
    assert checks[-1]["check"] == "preserved_json_parse"


def test_cli_replay_and_required_field_principles(
    tmp_path: Path,
    preserved: dict[str, object],
    reduction: dict[str, object],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-SAMPLER-7392-TERMINAL: a cold CLI replay accepts the null."""

    artifact = exp.build_artifact_for_test(preserved, reduction, _passing_receipts())
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    assert exp.main(["--date", exp.RUN_DATE, "--replay-artifact", str(path)]) == 0
    replay = json.loads(capsys.readouterr().out)
    assert replay["cold_artifact_replay_errors"] == []
    assert set(artifact["field_principles"]) == set(artifact)
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact

    with pytest.raises(SystemExit, match=exp.RUN_DATE):
        exp.main(["--date", "20260917", "--replay-artifact", str(path)])

    called: dict[str, object] = {}

    def fake_run(root: Path, run_date: str, *, output_path: Path) -> dict[str, object]:
        called.update(root=root, run_date=run_date, output_path=output_path)
        return {}

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(exp, "run_experiment", fake_run)
    output = tmp_path / "out.json"
    try:
        assert exp.main(["--date", exp.RUN_DATE, "--output", str(output)]) == 0
    finally:
        monkeypatch.undo()
    assert called == {"root": exp.REPO_ROOT, "run_date": exp.RUN_DATE, "output_path": output}
