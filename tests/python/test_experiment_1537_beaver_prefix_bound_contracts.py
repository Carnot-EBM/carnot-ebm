"""Tests for Exp 1537 BEAVER-lite prefix-bound contract audit.

Spec: REQ-VERIFY-1537, SCENARIO-VERIFY-1537.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import beaver_prefix_bound_contracts as exp


def test_req_verify_1537_prefix_bounds_are_monotone_for_canonical_targets() -> None:
    """REQ-VERIFY-1537: structural prefix bounds are monotone and reject invalid prefixes."""

    target = exp.canonical_contract_target("case-a", "accept")
    trie = exp.PrefixFrontierTrie.from_targets([target])
    series = exp.build_prefix_bound_series(target, prefix_stride=1)
    bounds = [state.unsafe_upper_bound for state in series]
    invalid = exp.score_prefix(target, target[:10] + "X", source="observed")

    assert exp.canonical_contract_target("case-a", "maybe").endswith('"reject"}')
    assert trie.allowed_next_chars("") == frozenset({target[0]})
    assert trie.allowed_next_chars(target[:10]) == frozenset({target[10]})
    assert trie.allowed_next_chars(target) == frozenset()
    assert trie.contains_prefix(target[:12]) is True
    assert trie.contains_prefix(target[:12] + "X") is False
    assert bounds == sorted(bounds, reverse=True)
    assert bounds[0] == pytest.approx(1.0)
    assert bounds[-1] == pytest.approx(0.0)
    assert invalid.prefix_consistent is False
    assert invalid.unsafe_upper_bound == pytest.approx(1.0)


def test_req_verify_1537_audit_ranks_risk_but_keeps_validator_authority() -> None:
    """REQ-VERIFY-1537: high-risk rankings do not replace deterministic validators."""

    reject_case = _contract_case("reject-1", "grammar_certificate", False, False)
    target = exp.canonical_contract_target("reject-1", "reject")
    extra_reject_case = _contract_case("reject-2", "grammar_certificate", False, False)
    rows = [
        _decoder_row(
            reject_case,
            decoder_mode="baseline_post_decode",
            raw_output=exp.canonical_contract_target("reject-1", "accept"),
            proposed_final_accept=True,
            deterministic_validator_accept=False,
        ),
        _decoder_row(
            reject_case,
            decoder_mode="automata_guided",
            raw_output=target,
            proposed_final_accept=False,
            deterministic_validator_accept=True,
            token_logprob=-0.125,
            topk_logprobs=[-0.125, -2.5],
        ),
        _decoder_row(
            extra_reject_case,
            decoder_mode="baseline_post_decode",
            raw_output="not json",
            proposed_final_accept=False,
            deterministic_validator_accept=False,
        ),
    ]

    audit = exp.audit_decoder_rows(
        rows,
        focused_tests_passed=True,
        prefix_stride=8,
        max_cases_per_family=1,
    )
    fallback_audit = exp.audit_decoder_rows(
        [
            _decoder_row(
                _contract_case("fallback-1", "monitor_event", None, False),
                decoder_mode="baseline_post_decode",
                raw_output="not json",
                proposed_final_accept=False,
                deterministic_validator_accept=False,
            )
        ],
        focused_tests_passed=True,
        prefix_stride=16,
    )

    assert audit["deterministic_validator_final_authority"] is True
    assert audit["token_logprob_available"] is True
    assert audit["topk_available"] is True
    assert audit["false_accept_rate"] == pytest.approx(0.5)
    assert audit["bound_violations"] == []
    assert audit["high_risk_instances"][0]["decoder_mode"] == "baseline_post_decode"
    assert audit["high_risk_instances"][0]["deterministic_false_accept"] is True
    assert audit["high_risk_instances"][0]["bound_used_as_authority"] is False
    assert fallback_audit["bounded_prefixes"] > 0


def test_scenario_verify_1537_runner_writes_structural_audit_artifacts(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-1537: runner writes a ready artifact with structural fallback."""

    decoder_manifest = tmp_path / "decoder_1535.jsonl"
    runtime_manifest = tmp_path / "runtime_1520.jsonl"
    source_artifact = tmp_path / "experiment_1535.json"
    output = tmp_path / "experiment_1537.json"
    audit_manifest = tmp_path / "prefix_bounds.jsonl"
    cases = [
        _contract_case("cert-1", "grammar_certificate", False, False),
        _contract_case("safe-1", "safe_dsl", True, True),
    ]
    decoder_rows = [
        _decoder_row(
            cases[0],
            decoder_mode="baseline_post_decode",
            raw_output="not json",
            proposed_final_accept=False,
            deterministic_validator_accept=False,
        ),
        _decoder_row(
            cases[0],
            decoder_mode="automata_guided",
            raw_output=exp.canonical_contract_target("cert-1", "reject"),
            proposed_final_accept=False,
            deterministic_validator_accept=True,
        ),
        _decoder_row(
            cases[1],
            decoder_mode="baseline_post_decode",
            raw_output="",
            proposed_final_accept=False,
            deterministic_validator_accept=False,
        ),
        _decoder_row(
            cases[1],
            decoder_mode="automata_guided",
            raw_output=exp.canonical_contract_target("safe-1", "accept"),
            proposed_final_accept=True,
            deterministic_validator_accept=True,
        ),
    ]
    _write_jsonl(decoder_manifest, [*decoder_rows, {"row_type": "summary"}])
    _write_jsonl(runtime_manifest, cases)
    _write_json(
        source_artifact,
        {
            "status": "complete",
            "model_specs": [{"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"}],
            "live_sota_model_inference_used": True,
        },
    )

    artifact = exp.run_experiment(
        project_root=tmp_path,
        decoder_manifest_path=decoder_manifest,
        runtime_manifest_path=runtime_manifest,
        source_artifact_path=source_artifact,
        output_path=output,
        audit_manifest_path=audit_manifest,
        focused_tests_passed=True,
        prefix_stride=16,
    )
    audit_rows = _read_jsonl(audit_manifest)

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["beaver_bound_ready"] is True
    assert artifact["model_specs"] == [{"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"}]
    assert artifact["live_sota_model_inference_used"] is True
    assert artifact["bounded_prefixes"] == len(audit_rows) - 1
    assert artifact["token_logprob_available"] is False
    assert artifact["topk_available"] is False
    assert artifact["deterministic_validator_final_authority"] is True
    assert artifact["false_accept_rate"] == pytest.approx(0.0)
    assert artifact["focused_tests_passed"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["bound_audit_path"].endswith("prefix_bounds.jsonl")
    assert audit_rows[-1]["row_type"] == "summary"


def test_req_verify_1537_runner_blocks_empty_sources(tmp_path: Path) -> None:
    """REQ-VERIFY-1537: missing source manifests produce explicit terminal blockers."""

    artifact = exp.run_experiment(
        project_root=tmp_path,
        decoder_manifest_path=tmp_path / "missing_decoder.jsonl",
        runtime_manifest_path=tmp_path / "missing_runtime.jsonl",
        source_artifact_path=tmp_path / "missing_1535.json",
        output_path=tmp_path / "experiment_1537.json",
        audit_manifest_path=tmp_path / "prefix_bounds.jsonl",
        focused_tests_passed=False,
    )

    assert artifact["status"] == "blocked"
    assert artifact["beaver_bound_ready"] is False
    assert "focused_tests_not_passed" in artifact["blockers"]
    assert any(
        blocker.startswith("missing_or_empty_decoder_manifest:")
        for blocker in artifact["blockers"]
    )
    assert any(
        blocker.startswith("missing_or_empty_runtime_manifest:")
        for blocker in artifact["blockers"]
    )


def _contract_case(
    case_id: str,
    source_family: str,
    expected_label: bool | None,
    final_accept: bool,
) -> dict[str, Any]:
    return {
        "row_type": "contract_case",
        "contract_schema_version": "runtime-contract-e2e/v1",
        "contract_case_id": case_id,
        "prompt_or_case_id": case_id,
        "proposed_output": case_id,
        "certificate_parse_result": {"linked": source_family == "grammar_certificate"},
        "safe_dsl_verifier_result": {"linked": source_family == "safe_dsl"},
        "monitor_event_result": {"linked": source_family == "monitor_event"},
        "structural_contract_result": {"linked": source_family == "structural_contract"},
        "expected_label": expected_label,
        "final_deterministic_accept": final_accept,
        "final_deterministic_decision": "accept" if final_accept else "reject",
        "source_family": source_family,
        "source_path": "fixture.jsonl",
        "source_line": 1,
    }


def _decoder_row(
    case: dict[str, Any],
    *,
    decoder_mode: str,
    raw_output: str,
    proposed_final_accept: bool,
    deterministic_validator_accept: bool,
    token_logprob: float | None = None,
    topk_logprobs: list[float] | None = None,
) -> dict[str, Any]:
    validation_row = dict(case)
    validation_row["final_deterministic_accept"] = bool(proposed_final_accept)
    validation_row["final_deterministic_decision"] = (
        "accept" if proposed_final_accept else "reject"
    )
    row = {
        "row_type": "decoder_result",
        "contract_case_id": case["contract_case_id"],
        "prompt_or_case_id": case["prompt_or_case_id"],
        "source_family": case["source_family"],
        "decoder_mode": decoder_mode,
        "model_hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "raw_output_excerpt": raw_output,
        "expected_label": case["expected_label"],
        "proposed_final_deterministic_accept": bool(proposed_final_accept),
        "deterministic_validator_accept": bool(deterministic_validator_accept),
        "false_accept": bool(case["expected_label"] is False and proposed_final_accept),
        "contract_validation_row": validation_row,
    }
    if token_logprob is not None:
        row["token_logprob"] = token_logprob
    if topk_logprobs is not None:
        row["topk_logprobs"] = topk_logprobs
    return row


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
