"""Tests for Exp1551 automata/SAT unified contract gate.

Spec: REQ-VERIFY-1551, SCENARIO-VERIFY-1551.
"""

from __future__ import annotations

import json
import builtins
from pathlib import Path
from typing import Any

import pytest

from carnot.inference import sota_models
from carnot.verify import unified_contract_gate as exp


def test_req_verify_1551_routes_masks_repair_and_validator_in_order() -> None:
    """REQ-VERIFY-1551: gate stages run in syntax, repair, validator order."""

    events: list[str] = []
    case = exp.UnifiedGateCase(
        case_id="sat-route",
        source_family="satquest",
        raw_output='{"answer":"SAT"}',
        expected_accept=True,
        syntax_mask=_stage(events, "automata_mask", passed=True),
        semantic_repair=_stage(
            events,
            "semantic_repair",
            passed=True,
            output='{"answer":"SAT","assignment":{"x1":true}}',
            repair_applied=True,
            repair_success=True,
        ),
        deterministic_validator=_stage(
            events,
            "sat_oracle",
            passed=True,
            oracle_agrees=True,
            deterministic_accept=True,
        ),
    )

    result = exp.UnifiedContractGate().evaluate(case)

    assert events == ["automata_mask", "semantic_repair", "sat_oracle"]
    assert result.final_accept is True
    assert result.false_accept is False
    assert [stage.stage for stage in result.stages] == [
        "automata_mask",
        "semantic_repair",
        "sat_oracle",
    ]
    assert result.automata_masks_used is True
    assert result.semantic_repair_layer_used is True
    assert result.sat_oracle_used is True


def test_scenario_verify_1551_rejects_solver_mismatch_after_repair() -> None:
    """SCENARIO-VERIFY-1551: deterministic solver mismatch rejects the output."""

    case = exp.UnifiedGateCase(
        case_id="pl-mismatch",
        source_family="product_line",
        raw_output='{"selected_features":["BogusFeature"],"verifier":{"accept":true}}',
        expected_accept=True,
        soft_accept=True,
        syntax_mask=lambda _case, text: exp.GateStageResult(
            "automata_mask",
            True,
            text,
            reason="syntactically_json",
        ),
        semantic_repair=lambda _case, _text: exp.GateStageResult(
            "semantic_repair",
            True,
            '{"selected_features":["StillWrong"],"verifier":{"accept":true}}',
            repair_applied=True,
            repair_success=True,
        ),
        deterministic_validator=lambda _case, text: exp.GateStageResult(
            "product_line_oracle",
            False,
            text,
            reason="oracle_mismatch",
            oracle_agrees=False,
            deterministic_accept=False,
        ),
    )

    result = exp.UnifiedContractGate().evaluate(case)

    assert result.final_accept is False
    assert result.oracle_agrees is False
    assert result.rejected_by == "product_line_oracle"
    assert result.soft_accept is True
    assert result.product_line_oracle_used is True
    assert result.false_accept is False


def test_req_verify_1551_soft_signal_cannot_override_runtime_contract() -> None:
    """REQ-VERIFY-1551: model-declared accept remains advisory only."""

    case = exp.UnifiedGateCase(
        case_id="runtime-soft-override",
        source_family="runtime_contract",
        raw_output='{"contract_case_id":"runtime-soft-override","final_deterministic_decision":"accept"}',
        expected_accept=False,
        soft_accept=True,
        syntax_mask=_stage([], "automata_mask", passed=True),
        semantic_repair=_stage([], "semantic_repair", passed=True),
        deterministic_validator=lambda _case, text: exp.GateStageResult(
            "runtime_contracts",
            False,
            text,
            reason="runtime_contract_rejected",
            oracle_agrees=False,
            deterministic_accept=False,
        ),
    )

    result = exp.UnifiedContractGate().evaluate(case)

    assert result.final_accept is False
    assert result.false_accept is False
    assert result.soft_signal_overrode_validator is False
    assert result.runtime_contracts_used is True


def test_scenario_verify_1551_runner_writes_ready_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1551: runner writes mixed-case zero-false-accept metrics."""

    output = tmp_path / "experiment_1551.json"
    manifest = tmp_path / "unified_gate.jsonl"
    predecessor_paths = _write_predecessors(tmp_path)
    cases = [
        _passing_case("sat-1", "satquest", syntax_passed=False),
        _passing_case("pl-1", "product_line", syntax_passed=True),
        _passing_case("rt-1", "runtime_contract", syntax_passed=False),
    ]

    artifact = exp.run_experiment(
        project_root=tmp_path,
        output_path=output,
        manifest_path=manifest,
        predecessor_paths=predecessor_paths,
        case_builder_fn=lambda _root: cases,
        model_probe_fn=lambda _root: {
            "live_sota_model_inference_used": False,
            "models_used": [],
            "availability_blockers": ["no_mandated_sota_gguf_runtime"],
            "legacy_small_models_excluded_from_headline_metrics": True,
        },
        focused_tests_passed=True,
    )
    persisted = json.loads(output.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert artifact == persisted
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["unified_contract_gate_ready"] is True
    assert artifact["cases_attempted"] == 3
    assert artifact["automata_masks_used"] is True
    assert artifact["semantic_repair_layer_used"] is True
    assert artifact["sat_oracle_used"] is True
    assert artifact["product_line_oracle_used"] is True
    assert artifact["runtime_contracts_used"] is True
    assert artifact["syntax_accept_rate"] == pytest.approx(1 / 3)
    assert artifact["semantic_repair_success_rate"] == pytest.approx(1.0)
    assert artifact["oracle_agreement_rate"] == pytest.approx(1.0)
    assert artifact["false_accept_rate"] == pytest.approx(0.0)
    assert artifact["focused_tests_passed"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert rows[-1]["row_type"] == "summary"
    assert rows[-1]["false_accept_rate"] == pytest.approx(0.0)


def test_req_verify_1551_predecessor_gate_requires_satquest_zero_false_accepts(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-1551: Exp1549 zero-false-accept evidence gates SAT authority."""

    predecessor_paths = _write_predecessors(tmp_path, sat_false_accepts_after=1)

    artifact = exp.run_experiment(
        project_root=tmp_path,
        output_path=tmp_path / "experiment_1551.json",
        manifest_path=tmp_path / "unified_gate.jsonl",
        predecessor_paths=predecessor_paths,
        case_builder_fn=lambda _root: [_passing_case("sat-1", "satquest")],
        model_probe_fn=lambda _root: {
            "live_sota_model_inference_used": False,
            "models_used": [],
            "availability_blockers": ["no_mandated_sota_gguf_runtime"],
        },
        focused_tests_passed=True,
    )

    assert artifact["status"] == "blocked"
    assert artifact["unified_contract_gate_ready"] is False
    assert artifact["cases_attempted"] == 0
    assert "exp1549_satquest_repaired_false_accepts_nonzero:1" in artifact["blockers"]


def test_req_verify_1551_default_cases_use_real_deterministic_validators() -> None:
    """REQ-VERIFY-1551: built-in mixed cases replay the real local validators."""

    cases = exp.build_bounded_mixed_cases(Path.cwd())
    evaluations = [exp.UnifiedContractGate().evaluate(case) for case in cases]
    summary = exp.summarize_gate_evaluations(evaluations)

    assert {case.source_family for case in cases} == {
        "satquest",
        "product_line",
        "runtime_contract",
    }
    assert all(item.final_accept for item in evaluations)
    assert summary["cases_attempted"] == 3
    assert summary["syntax_accept_rate"] == pytest.approx(1 / 3)
    assert summary["semantic_repair_success_rate"] == pytest.approx(1.0)
    assert summary["oracle_agreement_rate"] == pytest.approx(1.0)
    assert summary["false_accept_rate"] == pytest.approx(0.0)


def test_req_verify_1551_helpers_and_blocker_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-1551: edge helpers fail closed without changing authority."""

    assert exp.summarize_gate_evaluations([])["cases_attempted"] == 0
    assert exp.json_syntax_mask(_passing_case("ok", "satquest"), '{"ok": true}').passed is True
    assert exp.constant_repair('{"ok":true}')(
        _passing_case("ok", "satquest"),
        '{"ok":true}',
    ).reason == "already_validator_shaped"
    assert exp._expected_accept({"final_deterministic_accept": False}) is False  # noqa: SLF001
    assert exp._completion_text("plain") == "plain"  # noqa: SLF001
    assert exp._completion_text(None) == ""  # noqa: SLF001
    assert exp._completion_text({"choices": []}) == ""  # noqa: SLF001
    assert exp._completion_text({"choices": ["bad"]}) == ""  # noqa: SLF001
    assert exp._completion_text({"choices": [{"text": "done"}]}) == "done"  # noqa: SLF001
    assert exp._completion_text({"choices": [{"text": 1}]}) == ""  # noqa: SLF001
    assert exp._resolve_under_root(tmp_path, Path("/tmp/example")).is_absolute()  # noqa: SLF001
    assert exp._display_path(Path("/tmp/outside"), tmp_path) == "/tmp/outside"  # noqa: SLF001

    missing_loaded, missing_blockers = exp.load_predecessor_artifacts(
        exp.PredecessorPaths(
            tmp_path / "missing1535.json",
            tmp_path / "missing1549.json",
            tmp_path / "missing1540.json",
        ),
        project_root=tmp_path,
    )
    assert missing_loaded == {}
    assert len(missing_blockers) == 3

    predecessor_paths = _write_predecessors(tmp_path)
    (tmp_path / "experiment_1535.json").write_text(
        json.dumps({"status": "complete", "false_accept_rate": 0.5}),
        encoding="utf-8",
    )
    (tmp_path / "experiment_1540.json").write_text(
        json.dumps({"status": "complete", "false_accept_rate": 0.1, "oracle_agreement_rate": 0.5}),
        encoding="utf-8",
    )
    _loaded, blockers = exp.load_predecessor_artifacts(predecessor_paths, project_root=tmp_path)
    assert "exp1535_false_accept_rate_nonzero:0.5" in blockers
    assert "exp1540_false_accept_rate_nonzero:0.1" in blockers
    assert "exp1540_oracle_agreement_below_one:0.5" in blockers

    blocked = exp.run_experiment(
        project_root=tmp_path,
        output_path=tmp_path / "blocked_empty.json",
        manifest_path=tmp_path / "blocked_empty.jsonl",
        predecessor_paths=_write_predecessors(tmp_path / "fresh"),
        case_builder_fn=lambda _root: [],
        model_probe_fn=lambda _root: {"availability_blockers": []},
        focused_tests_passed=False,
    )
    assert blocked["status"] == "blocked"
    assert "no_unified_contract_gate_cases" in blocked["blockers"]
    assert "focused_tests_not_passed" in blocked["blockers"]

    monkeypatch.setattr(sota_models, "resolve_cached_gguf", lambda _hf_id: None)
    unavailable = exp.probe_headline_model_availability(tmp_path)
    assert unavailable["live_sota_model_inference_used"] is False
    assert "no_mandated_sota_gguf_runtime" in unavailable["availability_blockers"]

    real_import = builtins.__import__

    def fail_llama_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "llama_cpp":
            raise ModuleNotFoundError("forced llama_cpp absence")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(sota_models, "resolve_cached_gguf", lambda _hf_id: "/models/headline.gguf")
    monkeypatch.setattr(builtins, "__import__", fail_llama_import)
    cached_without_llama = exp.probe_headline_model_availability(tmp_path)
    assert cached_without_llama["live_sota_model_inference_used"] is False
    assert cached_without_llama["cached_mandated_models"]
    assert any(
        blocker.startswith("llama_cpp_import_failed:")
        for blocker in cached_without_llama["availability_blockers"]
    )


def test_req_verify_1551_validator_negative_edges() -> None:
    """REQ-VERIFY-1551: built-in validators reject malformed or mismatched rows."""

    product_case = exp.product_line1540.build_staged_product_line_cases(target_count=1)[0]
    product_validator = exp.product_line_validator(product_case)
    malformed_product = product_validator(_passing_case("pl", "product_line"), "not json")
    assert malformed_product.passed is False
    assert malformed_product.reason == "no_json_object"

    sat_case = exp.satquest1536.build_prompt_cases()[0]
    sat_validator = exp.satquest_validator(sat_case)
    malformed_sat = sat_validator(_passing_case("sat", "satquest"), '{"answer":"UNSAT","verifier":{"accept":true}}')
    assert malformed_sat.passed is False
    assert malformed_sat.false_accept is True

    runtime_case = exp.automata1535.select_contract_cases(
        Path("results/runtime_contract_e2e_manifest_1520.jsonl"),
        per_family=1,
    )[0]
    runtime_validator = exp.runtime_contract_validator(runtime_case)
    wrong_runtime = runtime_validator(
        _passing_case("rt", "runtime_contract"),
        '{"contract_case_id":"wrong","final_deterministic_decision":"accept"}',
    )
    assert wrong_runtime.passed is False
    assert wrong_runtime.reason == "contract_case_id_mismatch"


def _stage(
    events: list[str],
    name: str,
    *,
    passed: bool,
    output: str | None = None,
    repair_applied: bool = False,
    repair_success: bool = False,
    oracle_agrees: bool | None = None,
    deterministic_accept: bool | None = None,
) -> exp.GateStageFn:
    def _fn(_case: exp.UnifiedGateCase, text: str) -> exp.GateStageResult:
        events.append(name)
        return exp.GateStageResult(
            name,
            passed,
            text if output is None else output,
            repair_applied=repair_applied,
            repair_success=repair_success,
            oracle_agrees=oracle_agrees,
            deterministic_accept=deterministic_accept,
        )

    return _fn


def _passing_case(
    case_id: str,
    source_family: str,
    *,
    syntax_passed: bool = True,
) -> exp.UnifiedGateCase:
    return exp.UnifiedGateCase(
        case_id=case_id,
        source_family=source_family,
        raw_output="not-json" if not syntax_passed else '{"ok":true}',
        expected_accept=True,
        syntax_mask=lambda _case, text: exp.GateStageResult(
            "automata_mask",
            syntax_passed,
            text,
            reason="syntax_seed",
        ),
        semantic_repair=lambda _case, _text: exp.GateStageResult(
            "semantic_repair",
            True,
            '{"ok":true}',
            repair_applied=True,
            repair_success=True,
        ),
        deterministic_validator=lambda _case, text: exp.GateStageResult(
            {
                "satquest": "sat_oracle",
                "product_line": "product_line_oracle",
                "runtime_contract": "runtime_contracts",
            }[source_family],
            True,
            text,
            oracle_agrees=True,
            deterministic_accept=True,
        ),
    )


def _write_predecessors(
    tmp_path: Path,
    *,
    sat_false_accepts_after: int = 0,
) -> exp.PredecessorPaths:
    tmp_path.mkdir(parents=True, exist_ok=True)
    exp1535 = tmp_path / "experiment_1535.json"
    exp1549 = tmp_path / "experiment_1549.json"
    exp1540 = tmp_path / "experiment_1540.json"
    exp1535.write_text(
        json.dumps(
            {
                "status": "complete",
                "contract_decoder_adapter_ready": True,
                "false_accept_rate": 0.0,
            }
        ),
        encoding="utf-8",
    )
    exp1549.write_text(
        json.dumps(
            {
                "status": "complete",
                "satquest_oracle_repair_ready": sat_false_accepts_after == 0,
                "satquest_zero_false_accepts": sat_false_accepts_after == 0,
                "solver_oracle_false_accepts_after": sat_false_accepts_after,
            }
        ),
        encoding="utf-8",
    )
    exp1540.write_text(
        json.dumps(
            {
                "status": "complete",
                "product_line_scale_ready": True,
                "false_accept_rate": 0.0,
                "oracle_agreement_rate": 1.0,
            }
        ),
        encoding="utf-8",
    )
    return exp.PredecessorPaths(
        exp1535_artifact=exp1535,
        exp1549_artifact=exp1549,
        exp1540_artifact=exp1540,
    )
