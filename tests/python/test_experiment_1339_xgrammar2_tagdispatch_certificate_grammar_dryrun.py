"""Tests for Exp 1339 dynamic certificate grammar dispatch dry-run.

Spec: REQ-VERIFY-1339, SCENARIO-VERIFY-1339
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import xgrammar2_tagdispatch_certificate_grammar_dryrun as mod


def test_req1339_dispatch_preserves_all_certificate_states() -> None:
    """REQ-VERIFY-1339-2/3: synthetic branches dispatch without losing UNKNOWN."""

    grammar = mod.compile_branch_grammars(timer=mod.ConstantStepTimer())
    summary = mod.evaluate_synthetic_cases(
        mod.synthetic_certificate_cases(),
        grammar=grammar,
        mask_timer=mod.ConstantStepTimer(),
    )

    assert summary["certificate_states_supported"] == ["REPAIR_HINT", "SAT", "UNKNOWN", "UNSAT"]
    assert summary["unknown_state_supported"] is True
    assert summary["state_transition_error_count"] == 0
    assert summary["dynamic_parse_rate"] == pytest.approx(1.0)
    assert summary["static_gbnf_proxy_parse_rate"] == pytest.approx(0.75)
    assert summary["parse_rate_delta_over_static_gbnf_proxy"] == pytest.approx(0.25)
    assert summary["mask_generation_ms_per_token_proxy"] > 0.0

    repair_row = next(row for row in summary["dispatch_results"] if row["expected_state"] == "REPAIR_HINT")
    assert repair_row["dispatched_state"] == "REPAIR_HINT"
    assert repair_row["dynamic_parseable"] is True
    assert repair_row["existing_parser_parseable"] is False


def test_req1339_unsupported_text_counts_as_transition_error() -> None:
    """REQ-VERIFY-1339-3: wrong branch decisions are counted explicitly."""

    grammar = mod.compile_branch_grammars(timer=mod.ConstantStepTimer())
    bad_case = mod.SyntheticCertificateCase(
        name="missing_label",
        expected_state="UNKNOWN",
        text="The solver timed out without emitting a bounded label.",
    )

    summary = mod.evaluate_synthetic_cases([bad_case], grammar=grammar, mask_timer=mod.ConstantStepTimer())
    result = mod.dispatch_certificate_text(bad_case.text, grammar)

    assert result.dispatched_state == "UNSUPPORTED"
    assert result.parseable is False
    assert "no_dynamic_branch_match" in result.errors
    assert summary["state_transition_error_count"] == 1
    assert summary["unknown_state_supported"] is False


def test_req1339_backend_candidates_report_xgrammar_absence_honestly() -> None:
    """REQ-VERIFY-1339-4: backend records distinguish XGrammar from the shim."""

    candidates = mod.grammar_backend_candidates(
        import_checker=lambda _name: False,
        cli_finder=lambda _name: None,
        help_runner=lambda _path: "",
    )

    by_name = {candidate["name"]: candidate for candidate in candidates}
    assert by_name["xgrammar2_tagdispatch_native"]["available"] is False
    assert by_name["xgrammar2_tagdispatch_native"]["failure_reason"] == "xgrammar_import_absent"
    assert by_name["pure_python_tagdispatch_shim"]["available"] is True
    assert by_name["pure_python_tagdispatch_shim"]["fallback_only"] is True


def test_req1339_artifact_contains_required_fields_and_honest_verdict() -> None:
    """REQ-VERIFY-1339-4/5: artifact readiness follows state-transition results."""

    artifact = mod.build_dryrun_artifact(
        run_date="20260505",
        project_root="/repo",
        import_checker=lambda _name: False,
        cli_finder=lambda _name: None,
        help_runner=lambda _path: "",
        timer_factory=lambda: mod.ConstantStepTimer(),
    )

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["artifact_metadata"]["project_root"] == "/repo"
    assert artifact["artifact_metadata"]["run_date"] == "20260505"
    assert artifact["llm_inference_run"] is False
    assert artifact["sota_model_called"] is False
    assert artifact["dynamic_grammar_compile_ms"] > 0.0
    assert artifact["mask_generation_ms_per_token_proxy"] > 0.0
    assert artifact["state_transition_error_count"] == 0
    assert artifact["dynamic_grammar_ready"] is True
    assert artifact["honest_verdict"] == "dryrun_ready_pure_python_tagdispatch_xgrammar_absent"


def test_scenario1339_run_experiment_writes_in_progress_then_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-1339: runner persists the bootstrap and final artifacts."""

    output_path = tmp_path / "experiment_1339.json"
    writes: list[dict[str, Any]] = []
    real_write = mod._write_json

    def recording_write(path: Path, payload: dict[str, Any]) -> None:
        writes.append(payload)
        real_write(path, payload)

    monkeypatch.setattr(mod, "_write_json", recording_write)

    artifact = mod.run_experiment(
        output_path=output_path,
        run_date="20260505",
        project_root=tmp_path,
        import_checker=lambda name: name == "xgrammar",
        cli_finder=lambda _name: None,
        help_runner=lambda _path: "",
        timer_factory=lambda: mod.ConstantStepTimer(),
    )

    persisted = json.loads(output_path.read_text(encoding="utf-8"))

    assert [write["status"] for write in writes] == ["in_progress", "complete"]
    assert persisted == artifact
    assert artifact["honest_verdict"] == "dryrun_ready_native_xgrammar_importable"
    assert mod._module_available("json") is True
