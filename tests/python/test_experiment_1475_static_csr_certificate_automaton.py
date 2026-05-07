"""Tests for Exp 1475 STATIC CSR certificate automaton smoke."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.eval import static_csr_certificate_automaton as exp


def test_req_verify_1475_csr_automaton_matches_existing_parser_cases() -> None:
    """REQ-VERIFY-1475: CSR acceptance is exact on the measured schema cases."""

    cases = exp.tiny_certificate_cases()
    automaton = exp.build_static_csr_automaton(
        case.text for case in cases if exp.existing_path_accepts(case.text)
    )
    evaluation = exp.evaluate_equivalence(cases, automaton)

    assert automaton.state_count == len(automaton.row_offsets) - 1
    assert len(automaton.labels) == len(automaton.targets)
    assert evaluation["schema_cases_evaluated"] == len(cases)
    assert evaluation["exact_acceptance_equivalent"] is True
    assert evaluation["false_accepts"] == 0
    assert evaluation["false_rejects"] == 0
    assert all(
        record["existing_accepts"] == record["csr_accepts"] for record in evaluation["cases"]
    )


def test_req_verify_1475_diagnostics_report_false_accepts_and_rejects() -> None:
    """REQ-VERIFY-1475: mismatched automata expose false accept/reject counts."""

    cases = exp.tiny_certificate_cases()
    invalid_case = next(case for case in cases if not exp.existing_path_accepts(case.text))
    mismatched = exp.build_static_csr_automaton([invalid_case.text])
    evaluation = exp.evaluate_equivalence(cases, mismatched)

    assert evaluation["exact_acceptance_equivalent"] is False
    assert evaluation["false_accepts"] == 1
    assert evaluation["false_rejects"] >= 1
    assert invalid_case.name in evaluation["false_accept_case_names"]
    assert mismatched.accepts(invalid_case.text + " ") is False


def test_req_verify_1475_existing_parser_path_rejects_malformed_inputs() -> None:
    """REQ-VERIFY-1475: the baseline is the existing JSON parser/regex validator."""

    assert exp.existing_path_accepts("{") is False
    assert exp.existing_path_accepts("[]") is False
    assert exp.existing_path_accepts(exp.tiny_certificate_cases()[0].text) is True
    with pytest.raises(ValueError, match="accepted_strings"):
        exp.build_static_csr_automaton([])
    automaton = exp.build_static_csr_automaton([exp.tiny_certificate_cases()[0].text])
    with pytest.raises(ValueError, match="repeats"):
        exp.benchmark_acceptors(exp.tiny_certificate_cases(), automaton, repeats=0)


def test_scenario_verify_1475_runner_writes_required_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-1475: runner writes in-progress then complete artifact."""

    output_path = tmp_path / "experiment_1475_static_csr_certificate_automaton_smoke.json"
    statuses: list[str] = []
    original = exp.write_in_progress_artifact

    def tracking_write(path: Path | str, *, run_date: str = exp.RUN_DATE) -> dict[str, object]:
        artifact = original(path, run_date=run_date)
        statuses.append(json.loads(Path(path).read_text(encoding="utf-8"))["status"])
        return artifact

    monkeypatch.setattr(exp, "write_in_progress_artifact", tracking_write)
    artifact = exp.run_smoke(
        output_path=output_path,
        run_date="20260507",
        repeats=3,
        tests_run=["focused pytest"],
        timer=exp.ConstantStepTimer(step_ns=1000),
    )
    persisted = json.loads(output_path.read_text(encoding="utf-8"))

    assert statuses == ["in_progress"]
    assert persisted == artifact
    assert persisted["status"] == "complete"
    assert persisted["schema_cases_evaluated"] == len(exp.tiny_certificate_cases())
    assert persisted["csr_automaton_path"].endswith(
        "python/carnot/eval/static_csr_certificate_automaton.py"
    )
    assert persisted["exact_acceptance_equivalent"] is True
    assert persisted["existing_path_latency_ms_p50"] == pytest.approx(0.001)
    assert persisted["csr_latency_ms_p50"] == pytest.approx(0.001)
    assert persisted["speedup_ratio"] == pytest.approx(1.0)
    assert persisted["tests_run"] == ["focused pytest"]
    assert persisted["llm_inference_run"] is False
    assert persisted["repair_loop_run"] is False
    assert "bounded" in persisted["honest_verdict"]
