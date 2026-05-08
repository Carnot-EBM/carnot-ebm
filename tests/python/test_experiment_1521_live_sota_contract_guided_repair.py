"""Tests for Exp 1521 live SOTA contract-guided repair.

Spec: REQ-VERIFY-1521, SCENARIO-VERIFY-1521.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import live_sota_contract_guided_repair as exp


def test_req_verify_1521_selects_bounded_explicit_failing_or_marginal_cases(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-1521: only deterministic failing or marginal rows are selected."""

    manifest = tmp_path / "runtime_contract.jsonl"
    _write_jsonl(
        manifest,
        [
            {"row_type": "summary", "contract_cases_total": 4},
            _case("accept-ok", expected_label=True, final_accept=True),
            _case("unlabeled-reject", expected_label=None, final_accept=False),
            _case("reject-bad", expected_label=False, final_accept=False),
            _case("marginal-false-reject", expected_label=True, final_accept=False),
        ],
    )

    cases = exp.select_repair_cases(manifest, limit=2)

    assert [case["contract_case_id"] for case in cases] == [
        "reject-bad",
        "marginal-false-reject",
    ]
    assert all(isinstance(case["expected_label"], bool) for case in cases)
    assert all(case["row_type"] == "contract_case" for case in cases)


def test_req_verify_1521_validation_uses_exp1520_false_accept_ledger() -> None:
    """REQ-VERIFY-1521: generated prose is never the acceptance authority."""

    case = _case("reject-bad", expected_label=False, final_accept=False)
    correct = exp.validate_repair_output(
        case,
        model_spec={"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF", "name": "Qwen"},
        mode="grammar_only",
        raw_output='{"contract_case_id":"reject-bad","final_deterministic_decision":"reject"}',
    )
    false_accept = exp.validate_repair_output(
        case,
        model_spec={"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF", "name": "Qwen"},
        mode="draft_conditioned",
        raw_output='draft\n{"contract_case_id":"reject-bad","final_deterministic_decision":"accept"}',
    )
    prose = exp.validate_repair_output(
        case,
        model_spec={"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF", "name": "Qwen"},
        mode="baseline",
        raw_output="I think the invalid certificate should be rejected.",
    )

    summary = exp.summarize_repair_rows([correct, false_accept, prose])

    assert correct["repair_outcome"] == "accepted"
    assert correct["deterministic_validator_accept"] is True
    assert correct["false_accept"] is False
    assert false_accept["repair_outcome"] == "false_accept"
    assert false_accept["deterministic_validator_accept"] is False
    assert false_accept["false_accept"] is True
    assert prose["repair_outcome"] == "invalid_structure"
    assert prose["deterministic_validator_accept"] is False
    assert prose["false_accept"] is False
    assert summary["false_accept_count"] == 1
    assert summary["false_accept_rate"] == pytest.approx(1 / 3)


def test_req_verify_1521_validation_rejects_mismatched_or_incomplete_json() -> None:
    """REQ-VERIFY-1521: strict JSON parsing still requires the matching case ID."""

    case = _case("reject-bad", expected_label=False, final_accept=False)
    by_bool = exp.validate_repair_output(
        case,
        model_spec={"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"},
        mode="grammar_only",
        raw_output='broken {not json} {"contract_case_id":"reject-bad","final_deterministic_accept":false}',
    )
    mismatch = exp.validate_repair_output(
        case,
        model_spec={"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"},
        mode="grammar_only",
        raw_output='{"contract_case_id":"other","final_deterministic_decision":"reject"}',
    )
    missing_decision = exp.validate_repair_output(
        case,
        model_spec={"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"},
        mode="grammar_only",
        raw_output='{"contract_case_id":"reject-bad"}',
    )
    unlabeled = exp.validate_repair_output(
        _case("unlabeled", expected_label=None, final_accept=False),
        model_spec={"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"},
        mode="grammar_only",
        raw_output='{"contract_case_id":"unlabeled","final_deterministic_decision":"reject"}',
    )

    assert by_bool["repair_outcome"] == "accepted"
    assert mismatch["repair_outcome"] == "contract_case_id_mismatch"
    assert missing_decision["repair_outcome"] == "missing_final_decision"
    assert unlabeled["repair_outcome"] == "unlabeled"
    with pytest.raises(ValueError, match="unknown repair mode"):
        exp.build_mode_prompt(case, "unknown")


def test_scenario_verify_1521_runner_writes_ready_manifest_and_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-1521: live SOTA rows compare baseline, grammar, and DCCD."""

    source_manifest = tmp_path / "runtime_contract.jsonl"
    output = tmp_path / "experiment_1521.json"
    repair_manifest = tmp_path / "repair.jsonl"
    _write_jsonl(source_manifest, [_case("reject-bad", expected_label=False, final_accept=False)])

    def fake_generate(prompt: str, model: dict[str, Any], mode: str, case: dict[str, Any]) -> str:
        del prompt, model
        if mode == "baseline":
            return "This should be rejected, but no strict JSON is emitted."
        decision = "reject" if case["expected_label"] is False else "accept"
        return (
            "draft: identify the contract failure first.\n"
            if mode == "draft_conditioned"
            else ""
        ) + json.dumps(
            {
                "contract_case_id": case["contract_case_id"],
                "final_deterministic_decision": decision,
            },
            sort_keys=True,
        )

    artifact = exp.run_experiment(
        project_root=tmp_path,
        run_date="20260508",
        source_manifest_path=source_manifest,
        output_path=output,
        repair_manifest_path=repair_manifest,
        cached_pair_fn=lambda **_: [
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                "gpu": 0,
                "model_path": "/models/qwen.gguf",
            }
        ],
        generator_fn=fake_generate,
        gpu_probe_fn=lambda: {"cuda_available": True, "gpu_count": 1},
        case_limit=1,
    )
    rows = _read_jsonl(repair_manifest)

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["live_sota_model_inference_used"] is True
    assert artifact["contract_guided_repair_ready"] is True
    assert artifact["e2e_cases_loaded"] == 1
    assert artifact["repair_cases_attempted"] == 1
    assert artifact["baseline_accept_rate"] == pytest.approx(0.0)
    assert artifact["grammar_only_accept_rate"] == pytest.approx(1.0)
    assert artifact["draft_conditioned_accept_rate"] == pytest.approx(1.0)
    assert artifact["repair_accept_rate_delta"] == pytest.approx(0.0)
    assert artifact["false_accept_count"] == 0
    assert artifact["false_accept_rate"] == pytest.approx(0.0)
    assert artifact["models_used"] == ["unsloth/Qwen3.6-35B-A3B-GGUF"]
    assert artifact["blockers"] == []
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(rows) == 3
    assert {row["mode"] for row in rows} == {
        "baseline",
        "grammar_only",
        "draft_conditioned",
    }
    assert all(row["raw_output_sha256"] for row in rows)


def test_req_verify_1521_runner_blocks_missing_or_empty_e2e_manifest(tmp_path: Path) -> None:
    """REQ-VERIFY-1521: missing or empty Exp 1520 manifests are terminal blockers."""

    output = tmp_path / "experiment_1521.json"
    repair_manifest = tmp_path / "repair.jsonl"
    model = {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "gpu": 0,
        "model_path": "/models/qwen.gguf",
    }

    missing = exp.run_experiment(
        project_root=tmp_path,
        run_date="20260508",
        source_manifest_path=tmp_path / "missing.jsonl",
        output_path=output,
        repair_manifest_path=repair_manifest,
        cached_pair_fn=lambda **_: [model],
        generator_fn=lambda *_args, **_kwargs: "must not be called",
        gpu_probe_fn=lambda: {},
    )
    empty_manifest = tmp_path / "empty.jsonl"
    empty_manifest.write_text("", encoding="utf-8")
    empty = exp.run_experiment(
        project_root=tmp_path,
        run_date="20260508",
        source_manifest_path=empty_manifest,
        output_path=output,
        repair_manifest_path=repair_manifest,
        cached_pair_fn=lambda **_: None,
        resolver_fn=lambda _hf_id: None,
        generator_fn=lambda *_args, **_kwargs: "must not be called",
        gpu_probe_fn=lambda: {},
    )

    assert any(str(blocker).startswith("missing_runtime_contract_manifest:") for blocker in missing["blockers"])
    assert "no_mandated_sota_model_completed_live_inference" in missing["blockers"]
    assert "no_deterministic_contract_failing_or_marginal_cases" in empty["blockers"]
    assert "no_mandated_sota_gguf_runtime" in empty["blockers"]


def test_req_verify_1521_runner_blocks_ready_on_false_accepts(tmp_path: Path) -> None:
    """REQ-VERIFY-1521: any deterministic false accept prevents readiness."""

    source_manifest = tmp_path / "runtime_contract.jsonl"
    output = tmp_path / "experiment_1521.json"
    repair_manifest = tmp_path / "repair.jsonl"
    _write_jsonl(source_manifest, [_case("reject-bad", expected_label=False, final_accept=False)])

    artifact = exp.run_experiment(
        project_root=tmp_path,
        run_date="20260508",
        source_manifest_path=source_manifest,
        output_path=output,
        repair_manifest_path=repair_manifest,
        cached_pair_fn=lambda **_: [
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                "gpu": 0,
                "model_path": "/models/qwen.gguf",
            }
        ],
        generator_fn=lambda _prompt, _model, _mode, case: json.dumps(
            {
                "contract_case_id": case["contract_case_id"],
                "final_deterministic_decision": "accept",
            }
        ),
        gpu_probe_fn=lambda: {},
        case_limit=1,
    )

    assert artifact["contract_guided_repair_ready"] is False
    assert artifact["false_accept_count"] == 3
    assert artifact["false_accept_rate"] == pytest.approx(1.0)
    assert "false_accept_rate_nonzero" in artifact["blockers"]


def test_req_verify_1521_resolves_single_cached_model_after_pair_exception() -> None:
    """REQ-VERIFY-1521: the local cache resolver can supply one mandated GGUF."""

    def broken_pair(**_: Any) -> None:
        raise RuntimeError("pair probe failed")

    models = exp._resolve_runtime_models(
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
            "role": "flagship_moe_primary_contract_guided_repair",
            "gpu": 0,
            "model_path": "/models/Qwen3.6-35B-A3B-GGUF.gguf",
        }
    ]


def test_req_verify_1521_blocks_without_mandated_sota_runtime(tmp_path: Path) -> None:
    """REQ-VERIFY-1521: missing mandated SOTA GGUFs block instead of falling back."""

    source_manifest = tmp_path / "runtime_contract.jsonl"
    output = tmp_path / "experiment_1521.json"
    repair_manifest = tmp_path / "repair.jsonl"
    _write_jsonl(source_manifest, [_case("reject-bad", expected_label=False, final_accept=False)])

    artifact = exp.run_experiment(
        project_root=tmp_path,
        run_date="20260508",
        source_manifest_path=source_manifest,
        output_path=output,
        repair_manifest_path=repair_manifest,
        cached_pair_fn=lambda **_: None,
        resolver_fn=lambda _hf_id: None,
        generator_fn=lambda *_args, **_kwargs: "must not be called",
        gpu_probe_fn=lambda: {"cuda_available": False, "gpu_count": 0},
        case_limit=1,
    )

    assert artifact["status"] == "blocked"
    assert artifact["live_sota_model_inference_used"] is False
    assert artifact["contract_guided_repair_ready"] is False
    assert artifact["e2e_cases_loaded"] == 1
    assert artifact["repair_cases_attempted"] == 0
    assert artifact["models_used"] == []
    assert "no_mandated_sota_gguf_runtime" in artifact["blockers"]
    assert artifact["honest_verdict"].startswith("complete:")
    assert repair_manifest.read_text(encoding="utf-8") == ""


def _case(contract_case_id: str, *, expected_label: bool | None, final_accept: bool) -> dict[str, Any]:
    return {
        "row_type": "contract_case",
        "contract_schema_version": "runtime-contract-e2e/v1",
        "contract_case_id": contract_case_id,
        "prompt_or_case_id": contract_case_id,
        "proposed_output": "candidate output",
        "certificate_parse_result": {"linked": False},
        "safe_dsl_verifier_result": {"linked": False},
        "monitor_event_result": {"linked": False},
        "structural_contract_result": {"linked": True, "contract_family": "unit"},
        "expected_label": expected_label,
        "final_deterministic_accept": final_accept,
        "final_deterministic_decision": "accept" if final_accept else "reject",
        "source_family": "structural_contract",
        "source_path": "source.jsonl",
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
