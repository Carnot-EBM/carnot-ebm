"""Tests for Exp 1535 XGrammar/ABS contract decoder adapter.

Spec: REQ-VERIFY-1535, SCENARIO-VERIFY-1535.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import xgrammar_abs_contract_decoder_adapter as exp


def test_req_verify_1535_selects_families_and_compiles_abs_dfa(tmp_path: Path) -> None:
    """REQ-VERIFY-1535: selected cases cover runtime-contract families and DFA masks."""

    manifest = tmp_path / "runtime_contract.jsonl"
    rows = [
        _case("safe-1", "safe_dsl", expected_label=True, final_accept=True),
        _case("safe-duplicate", "safe_dsl", expected_label=True, final_accept=True),
        _case("cert-1", "grammar_certificate", expected_label=False, final_accept=False),
        _case("ignored", "unknown_family", expected_label=True, final_accept=True),
        _case("monitor-1", "monitor_event", expected_label=None, final_accept=True),
        _case("struct-1", "structural_contract", expected_label=False, final_accept=False),
        {"row_type": "summary", "contract_cases_total": 4},
    ]
    _write_jsonl(manifest, rows)

    selected = exp.select_contract_cases(manifest, per_family=1)
    dfa = exp.compile_contract_dfa(selected[0])
    target = exp.canonical_contract_json(selected[0])

    assert [case["source_family"] for case in selected] == [
        "grammar_certificate",
        "safe_dsl",
        "monitor_event",
        "structural_contract",
    ]
    assert dfa.allowed_next_chars("") == frozenset({target[0]})
    assert dfa.allowed_next_chars(target[:12]) == frozenset({target[12]})
    assert dfa.allowed_next_chars('{"contract_case_id":"wrong"') == frozenset()
    assert dfa.allowed_next_chars(target) == frozenset()
    assert dfa.generate() == target
    assert dfa.accepts(target) is True
    assert dfa.accepts(target + " ") is False


def test_req_verify_1535_validator_handoff_blocks_false_accepts() -> None:
    """REQ-VERIFY-1535: deterministic validators remain final acceptance authority."""

    case = _case("reject-me", "structural_contract", expected_label=False, final_accept=False)
    false_accept = exp.validate_decoded_output(
        case,
        raw_output='{"contract_case_id":"reject-me","final_deterministic_decision":"accept"}',
        decoder_mode="baseline_post_decode",
        model_spec={"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"},
        latency_seconds=0.25,
    )
    guided = exp.validate_decoded_output(
        case,
        raw_output=exp.compile_contract_dfa(case).generate(),
        decoder_mode="automata_guided",
        model_spec={"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"},
        latency_seconds=0.05,
    )
    malformed = exp.validate_decoded_output(
        case,
        raw_output='reject {this is not json} {"contract_case_id":"wrong","final_deterministic_decision":"reject"}',
        decoder_mode="baseline_post_decode",
        model_spec={"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"},
        latency_seconds=0.01,
    )
    by_bool = exp.validate_decoded_output(
        case,
        raw_output='{"contract_case_id":"reject-me","final_deterministic_accept":false}',
        decoder_mode="automata_guided",
        model_spec={"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"},
        latency_seconds=0.02,
    )
    missing_decision = exp.validate_decoded_output(
        case,
        raw_output='{"contract_case_id":"reject-me"}',
        decoder_mode="baseline_post_decode",
        model_spec={"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"},
        latency_seconds=0.03,
    )
    summary = exp.summarize_decoder_rows([false_accept, guided, malformed])

    assert false_accept["parse_status"] == "ok"
    assert false_accept["false_accept"] is True
    assert false_accept["deterministic_validator_accept"] is False
    assert guided["parse_status"] == "ok"
    assert guided["false_accept"] is False
    assert guided["deterministic_validator_accept"] is True
    assert malformed["parse_status"] == "contract_case_id_mismatch"
    assert by_bool["parse_status"] == "ok"
    assert by_bool["deterministic_validator_accept"] is True
    assert missing_decision["parse_status"] == "missing_final_decision"
    assert summary["baseline_parse_rate"] == pytest.approx(0.5)
    assert summary["automata_parse_rate"] == pytest.approx(1.0)
    assert summary["baseline_contract_accept_rate"] == pytest.approx(0.0)
    assert summary["automata_contract_accept_rate"] == pytest.approx(1.0)
    assert summary["false_accept_rate"] == pytest.approx(1 / 3)


def test_scenario_verify_1535_runner_writes_ready_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1535: runner compares baseline and automata decoder rows."""

    source_manifest = tmp_path / "runtime_contract.jsonl"
    output = tmp_path / "experiment_1535.json"
    decoder_manifest = tmp_path / "decoder_rows.jsonl"
    _write_jsonl(
        source_manifest,
        [
            _case("cert-1", "grammar_certificate", expected_label=False, final_accept=False),
            _case("safe-1", "safe_dsl", expected_label=True, final_accept=True),
            _case("monitor-1", "monitor_event", expected_label=None, final_accept=True),
            _case("struct-1", "structural_contract", expected_label=False, final_accept=False),
        ],
    )

    artifact = exp.run_experiment(
        project_root=tmp_path,
        run_date="20260508",
        source_manifest_path=source_manifest,
        output_path=output,
        decoder_manifest_path=decoder_manifest,
        cached_pair_fn=lambda **_: [
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                "gpu": 0,
                "model_path": "/models/qwen.gguf",
            }
        ],
        baseline_generator_fn=lambda _prompt, _model, _case: "not strict JSON",
        xgrammar_probe_fn=lambda: False,
        gpu_probe_fn=lambda: {"cuda_available": True, "gpu_count": 1},
        focused_tests_passed=True,
    )
    rows = _read_jsonl(decoder_manifest)

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == ".118"
    assert artifact["contract_decoder_adapter_ready"] is True
    assert artifact["live_sota_model_inference_used"] is True
    assert artifact["cases_attempted"] == 4
    assert artifact["baseline_parse_rate"] == pytest.approx(0.0)
    assert artifact["automata_parse_rate"] == pytest.approx(1.0)
    assert artifact["baseline_contract_accept_rate"] == pytest.approx(0.0)
    assert artifact["automata_contract_accept_rate"] == pytest.approx(1.0)
    assert artifact["false_accept_rate"] == pytest.approx(0.0)
    assert artifact["xgrammar_available"] is False
    assert artifact["abs_dfa_masks_used"] is True
    assert artifact["focused_tests_passed"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(rows) == 9
    assert rows[-1]["row_type"] == "summary"
    assert {row["decoder_mode"] for row in rows[:-1]} == {
        "baseline_post_decode",
        "automata_guided",
    }


def test_req_verify_1535_runner_blocks_without_cases_or_models(tmp_path: Path) -> None:
    """REQ-VERIFY-1535: missing case/model gates produce terminal blockers."""

    source_manifest = tmp_path / "missing.jsonl"
    artifact = exp.run_experiment(
        project_root=tmp_path,
        run_date="20260508",
        source_manifest_path=source_manifest,
        output_path=tmp_path / "experiment_1535.json",
        decoder_manifest_path=tmp_path / "decoder_rows.jsonl",
        cached_pair_fn=lambda **_: None,
        resolver_fn=lambda _hf_id: None,
        xgrammar_probe_fn=lambda: False,
        gpu_probe_fn=lambda: {},
        focused_tests_passed=False,
    )

    assert artifact["status"] == "blocked"
    assert artifact["contract_decoder_adapter_ready"] is False
    assert artifact["live_sota_model_inference_used"] is False
    assert artifact["cases_attempted"] == 0
    assert any(blocker.startswith("missing_runtime_contract_manifest:") for blocker in artifact["blockers"])
    assert "no_mandated_sota_gguf_runtime" in artifact["blockers"]
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_verify_1535_empty_manifest_blocks_after_source_load(tmp_path: Path) -> None:
    """REQ-VERIFY-1535: an empty runtime-contract manifest cannot claim readiness."""

    source_manifest = tmp_path / "empty.jsonl"
    source_manifest.write_text("", encoding="utf-8")

    artifact = exp.run_experiment(
        project_root=tmp_path,
        run_date="20260508",
        source_manifest_path=source_manifest,
        output_path=tmp_path / "experiment_1535.json",
        decoder_manifest_path=tmp_path / "decoder_rows.jsonl",
        cached_pair_fn=lambda **_: [
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                "gpu": 0,
                "model_path": "/models/qwen.gguf",
            }
        ],
        baseline_generator_fn=lambda _prompt, _model, _case: "must not be called",
        xgrammar_probe_fn=lambda: False,
        gpu_probe_fn=lambda: {},
        focused_tests_passed=True,
    )

    assert artifact["status"] == "blocked"
    assert "no_runtime_contract_cases_selected" in artifact["blockers"]


def test_req_verify_1535_probe_helpers_cover_xgrammar_and_resolver() -> None:
    """REQ-VERIFY-1535: optional XGrammar probing and cache fallback are bounded."""

    assert exp.probe_xgrammar_available(lambda _name: object()) is True
    assert exp.probe_xgrammar_available(lambda _name: (_ for _ in ()).throw(ImportError())) is False
    assert exp.canonical_contract_payload({"contract_case_id": "fallback", "final_deterministic_accept": True}) == {
        "contract_case_id": "fallback",
        "final_deterministic_decision": "accept",
    }

    def broken_pair(**_: Any) -> None:
        raise RuntimeError("pair probe failed")

    models = exp.resolve_runtime_models(
        broken_pair,
        lambda hf_id: f"/models/{hf_id.rsplit('/', 1)[-1]}.gguf"
        if hf_id == "unsloth/Qwen3.6-35B-A3B-GGUF"
        else None,
        max_models=1,
    )

    assert models == [
        {
            "name": "Qwen3.6-35B-A3B",
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "role": "flagship_moe_contract_decoder",
            "gpu": 0,
            "model_path": "/models/Qwen3.6-35B-A3B-GGUF.gguf",
        }
    ]


def _case(
    case_id: str,
    source_family: str,
    *,
    expected_label: bool | None,
    final_accept: bool,
) -> dict[str, Any]:
    return {
        "row_type": "contract_case",
        "contract_schema_version": "runtime-contract-e2e/v1",
        "contract_case_id": case_id,
        "prompt_or_case_id": case_id,
        "proposed_output": f"output for {case_id}",
        "certificate_parse_result": {"linked": source_family == "grammar_certificate"},
        "safe_dsl_verifier_result": {"linked": source_family == "safe_dsl"},
        "monitor_event_result": {"linked": source_family == "monitor_event"},
        "structural_contract_result": {"linked": source_family == "structural_contract"},
        "expected_label": expected_label,
        "final_deterministic_accept": final_accept,
        "final_deterministic_decision": "accept" if final_accept else "reject",
        "source_family": source_family,
        "source_path": "synthetic.jsonl",
        "source_line": 1,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
