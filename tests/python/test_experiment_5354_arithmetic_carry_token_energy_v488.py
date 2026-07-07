"""Tests for Exp5354 arithmetic carry token-energy diagnostic.

Spec refs: REQ-VERIFY-5354, SCENARIO-VERIFY-5354.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5354_arithmetic_carry_token_energy_v488 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_gguf(path: Path) -> Path:
    path.write_bytes(b"GGUF\x03\x00\x00\x00" + b"\x00" * 32)
    return path


def _model_specs(model_path: Path) -> dict[str, Any]:
    specs: dict[str, Any] = {}
    for spec in exp.MANDATED_MODEL_SPECS:
        role = str(spec["role"])
        specs[role] = {
            "role": role,
            "hf_id": spec["hf_id"],
            "quantization": "Q4_K_M",
            "model_path": str(model_path) if role == "flagship_dense" else None,
            "status": "local_gguf_resolved" if role == "flagship_dense" else "missing_local_gguf",
            "autotokenizer_used": False,
            "file_receipts": None,
            "metadata": None,
        }
    return specs


def _exp5353_artifact(model_path: Path, server_path: Path, *, ready: bool = True) -> dict[str, Any]:
    selected = _model_specs(model_path)["flagship_dense"]
    return {
        "schema": "carnot.experiment_5353.tokenprob_feature_audit_corrigendum.v488",
        "experiment_id": {"value": "experiment_5353_tokenprob_feature_audit_corrigendum_v488"},
        "status": {"value": "complete" if ready else "blocked"},
        "honest_verdict": {
            "value": "complete: tokenprob_feature_rows_ready"
            if ready
            else "blocked_tokenprob_features_unavailable"
        },
        "inference_substrate": {"value": "live_llm_inference"},
        "MODEL_SPECS": {"value": _model_specs(model_path)},
        "preconditions_checked": {
            "value": {
                "gpu_visible": ready,
                "selected_backend_kind": "llama-server",
                "selected_backend_path": str(server_path),
                "selected_model_hf_id": selected["hf_id"],
                "selected_model_path": str(model_path),
                "selected_model_file_present": ready,
                "token_probability_api_available": ready,
                "external_text_scorer_reopened": False,
                "retired_scope_check": {
                    "phase_d_external_text_scorer_retired_marker_present": True,
                    "retired_scope_reopened": False,
                    "external_text_scorer_reopened": False,
                },
                "blocked_preconditions": [] if ready else ["per_token_logprob"],
            }
        },
        "selected_model_spec": {"value": selected},
        "per_token_logprob_available": ready,
        "topk_alternatives_available": ready,
        "tokenprob_feature_row_count": 3 if ready else 0,
        "tokenprob_feature_rows_ready": ready,
        "external_text_scorer_reopened": False,
        "no_quality_claim": True,
    }


def _paths(tmp_path: Path, *, exp5353_ready: bool = True) -> dict[str, Path]:
    model_path = _write_gguf(tmp_path / "gemma-4-31B-it-Q4_K_M.gguf")
    server_path = tmp_path / "llama-server"
    server_path.write_text("#!/bin/sh\n", encoding="utf-8")
    server_path.with_name("llama-cli").write_text("#!/bin/sh\n", encoding="utf-8")
    exp5353_path = _write_json(
        tmp_path / exp.exp5353.RESULT_RELATIVE_PATH,
        _exp5353_artifact(model_path, server_path, ready=exp5353_ready),
    )
    return {"model": model_path, "server": server_path, "exp5353": exp5353_path}


def _preconditions(server_path: Path, *, gpu_visible: bool = True) -> dict[str, Any]:
    return {
        "gpu_visible": gpu_visible,
        "nvidia_smi": {"ok": gpu_visible, "stdout": "0, NVIDIA RTX 3090, 24576, 24120"},
        "free_vram_mb": 24120 if gpu_visible else 0,
        "binary_paths": {"llama-server": str(server_path)},
        "cuda_backend_evidence": gpu_visible,
    }


def _case_receipt(case: exp.AdditionCase, *, unsafe_control: bool = False) -> dict[str, Any]:
    correct = case.correct_aliases[0]
    perturbed = case.perturbed_aliases[0]
    correct_logprob = -0.15 - 0.05 * len(case.carry_positions)
    perturbed_logprob = correct_logprob - 2.0
    if unsafe_control and case.is_perturbed_answer_control:
        correct_logprob = -2.40
        perturbed_logprob = -0.10
    return {
        "case_id": case.case_id,
        "prompt": case.prompt,
        "response_json": {
            "content": f" {correct}",
            "timings": {"predicted_per_token_ms": 21.0},
            "completion_probabilities": [
                {
                    "token": f" {correct}",
                    "logprob": correct_logprob,
                    "top_logprobs": [
                        {"token": f" {correct}", "logprob": correct_logprob},
                        {"token": f" {perturbed}", "logprob": perturbed_logprob},
                    ],
                }
            ],
        },
    }


def _complete_probe(**kwargs: Any) -> dict[str, Any]:
    return {
        "status": "completed",
        "backend_kind": "llama-server",
        "endpoint": "/completion",
        "wall_clock_s": 66.0,
        "round_count": 1,
        "case_receipts": [_case_receipt(case) for case in kwargs["diagnostic_cases"]],
    }


def _missing_perturbed_probe(**kwargs: Any) -> dict[str, Any]:
    payload = _complete_probe(**kwargs)
    for receipt in payload["case_receipts"]:
        rows = receipt["response_json"]["completion_probabilities"][0]["top_logprobs"]
        receipt["response_json"]["completion_probabilities"][0]["top_logprobs"] = [rows[0]]
    return payload


def _unsafe_false_accept_probe(**kwargs: Any) -> dict[str, Any]:
    return {
        "status": "completed",
        "backend_kind": "llama-server",
        "endpoint": "/completion",
        "wall_clock_s": 66.0,
        "round_count": 1,
        "case_receipts": [
            _case_receipt(case, unsafe_control=True) for case in kwargs["diagnostic_cases"]
        ],
    }


def test_req_verify_5354_spec_declares_carry_token_energy_contract() -> None:
    """REQ-VERIFY-5354: OpenSpec anchors the carry token-energy diagnostic."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5354") :]
    normalized_section = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5354",
        "SCENARIO-VERIFY-5354",
        str(exp.RESULT_RELATIVE_PATH),
        "tokenprob_feature_rows_ready=true",
        "answer-token negative logprob",
        "carry-position contrast",
        "perturbed-vs-correct",
        "unsafe false accepts",
        "no_broad_hallucination_claim=true",
        "external_text_scorer_reopened=false",
        "live_llm_inference",
        "aggregation_from_upstream_artifacts",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "scripts/research_conductor.py",
    ):
        assert marker in section

    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized_section


def test_req_verify_5354_cases_are_balanced_bounded_and_deterministic() -> None:
    """REQ-VERIFY-5354: deterministic addition cases cover carry/control strata."""

    counts = Counter(case.category for case in exp.ADDITION_CASES)

    assert 12 <= len(exp.ADDITION_CASES) <= 20
    assert counts == {
        "no_carry": 4,
        "single_carry": 4,
        "multi_carry": 4,
        "perturbed_control": 4,
    }
    assert exp._carry_positions(58, 67) == (0, 1)
    assert all(case.prompt.endswith("Answer:") for case in exp.ADDITION_CASES)
    assert any(case.is_perturbed_answer_control for case in exp.ADDITION_CASES)
    assert any(case.carry_kind == "multi_carry" for case in exp.ADDITION_CASES)


def test_scenario_verify_5354_clean_live_carry_energy_signal(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5354: complete token rows open the tiny carry signal gate."""

    paths = _paths(tmp_path)
    artifact = exp.run(
        root=tmp_path,
        result_path=tmp_path / exp.RESULT_RELATIVE_PATH,
        exp5353_artifact_path=paths["exp5353"],
        preconditions_provider=lambda: _preconditions(paths["server"]),
        token_probability_probe=_complete_probe,
        tests_run=[{"command": "unit exp5354", "outcome": "passed"}],
    )

    assert json.loads((tmp_path / exp.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    exp.validate_artifact(artifact)
    assert artifact["status"]["value"] == "complete"
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["inference_substrate"]["value"] == exp.INFERENCE_SUBSTRATE_LIVE
    assert artifact["diagnostic_case_count"] == len(exp.ADDITION_CASES)
    assert artifact["carry_case_count"] > 0
    assert artifact["feature_complete_rate"] == pytest.approx(1.0)
    assert artifact["correct_vs_perturbed_margin"] > 0
    assert artifact["unsafe_false_accepts"] == 0
    assert artifact["external_text_scorer_reopened"] is False
    assert artifact["no_broad_hallucination_claim"] is True
    assert artifact["carry_token_energy_signal_ready"] is True
    assert all(
        row["feature_complete"] is True
        for row in artifact["carry_token_energy_feature_rows"]["value"]
    )
    assert {row["category"] for row in artifact["carry_token_energy_feature_rows"]["value"]} == {
        "no_carry",
        "single_carry",
        "multi_carry",
        "perturbed_control",
    }


def test_scenario_verify_5354_blocks_before_probe_when_exp5353_not_ready(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-5354: Exp5353 readiness gates all carry probing."""

    paths = _paths(tmp_path, exp5353_ready=False)
    calls: list[str] = []
    artifact = exp.run(
        root=tmp_path,
        result_path=tmp_path / "blocked.json",
        exp5353_artifact_path=paths["exp5353"],
        preconditions_provider=lambda: _preconditions(paths["server"]),
        token_probability_probe=lambda **kwargs: calls.append(kwargs["selected_model_spec"]["hf_id"])
        or _complete_probe(**kwargs),
        tests_run=[{"command": "unit blocked", "outcome": "passed"}],
    )

    exp.validate_artifact(artifact)
    assert calls == []
    assert artifact["status"]["value"] == "blocked"
    assert artifact["honest_verdict"]["value"].startswith("blocked_")
    assert artifact["inference_substrate"]["value"] == exp.INFERENCE_SUBSTRATE_AGGREGATION
    assert artifact["diagnostic_case_count"] == 0
    assert artifact["carry_case_count"] == 0
    assert artifact["feature_complete_rate"] == 0.0
    assert artifact["carry_token_energy_signal_ready"] is False
    assert "exp5353_tokenprob_feature_rows_not_ready" in artifact["missing_feature_names"]


def test_scenario_verify_5354_blocks_when_target_toplogprob_rows_are_missing(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5354: missing per-target rows prevent hidden completeness."""

    paths = _paths(tmp_path)
    artifact = exp.run(
        root=tmp_path,
        result_path=tmp_path / "missing-features.json",
        exp5353_artifact_path=paths["exp5353"],
        preconditions_provider=lambda: _preconditions(paths["server"]),
        token_probability_probe=_missing_perturbed_probe,
        tests_run=[{"command": "unit missing features", "outcome": "passed"}],
    )

    exp.validate_artifact(artifact)
    assert artifact["status"]["value"] == "blocked"
    assert artifact["inference_substrate"]["value"] == exp.INFERENCE_SUBSTRATE_LIVE
    assert artifact["feature_complete_rate"] < 1.0
    assert artifact["correct_vs_perturbed_margin"] == 0.0
    assert artifact["unsafe_false_accepts"] == 0
    assert artifact["carry_token_energy_signal_ready"] is False
    assert any(
        name.endswith(":perturbed_target_logprob") for name in artifact["missing_feature_names"]
    )


def test_scenario_verify_5354_blocks_unsafe_false_accepts(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5354: wrong arithmetic accepted by token energy blocks readiness."""

    paths = _paths(tmp_path)
    artifact = exp.run(
        root=tmp_path,
        result_path=tmp_path / "unsafe.json",
        exp5353_artifact_path=paths["exp5353"],
        preconditions_provider=lambda: _preconditions(paths["server"]),
        token_probability_probe=_unsafe_false_accept_probe,
        tests_run=[{"command": "unit unsafe", "outcome": "passed"}],
    )

    exp.validate_artifact(artifact)
    assert artifact["status"]["value"] == "blocked"
    assert artifact["feature_complete_rate"] == pytest.approx(1.0)
    assert artifact["correct_vs_perturbed_margin"] < 0
    assert artifact["unsafe_false_accepts"] > 0
    assert artifact["carry_token_energy_signal_ready"] is False
    assert any(
        row["unsafe_false_accept"] is True
        for row in artifact["carry_token_energy_feature_rows"]["value"]
    )


def test_req_verify_5354_repository_artifact_is_schema_valid() -> None:
    """REQ-VERIFY-5354: checked-in deliverable keeps the required schema stable."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    exp.validate_artifact(artifact)
    assert artifact["experiment_id"]["value"] == exp.EXPERIMENT_ID
    assert artifact["honest_verdict"]["value"].startswith(("complete:", "blocked_"))
    assert artifact["external_text_scorer_reopened"] is False
    assert artifact["no_broad_hallucination_claim"] is True


def test_req_verify_5354_validation_rejects_contract_drift(tmp_path: Path) -> None:
    """REQ-VERIFY-5354: validation rejects scorer, schema, and signal drift."""

    paths = _paths(tmp_path)
    artifact = exp.run(
        root=tmp_path,
        result_path=tmp_path / "clean.json",
        exp5353_artifact_path=paths["exp5353"],
        preconditions_provider=lambda: _preconditions(paths["server"]),
        token_probability_probe=_complete_probe,
        tests_run=[{"command": "unit schema", "outcome": "passed"}],
    )

    malformed_cases = [
        (lambda a: (a["honest_verdict"].__setitem__("value", "done"), a)[1], "honest_verdict"),
        (
            lambda a: (a["inference_substrate"].__setitem__("value", "feature_audit_only"), a)[1],
            "inference_substrate",
        ),
        (
            lambda a: (a.__setitem__("external_text_scorer_reopened", True), a)[1],
            "external_text_scorer_reopened",
        ),
        (
            lambda a: (a.__setitem__("no_broad_hallucination_claim", False), a)[1],
            "no_broad_hallucination_claim",
        ),
        (
            lambda a: (a.__setitem__("diagnostic_case_count", True), a)[1],
            "diagnostic_case_count",
        ),
        (
            lambda a: (a.__setitem__("feature_complete_rate", "1.0"), a)[1],
            "feature_complete_rate",
        ),
        (
            lambda a: (a.__setitem__("correct_vs_perturbed_margin", "bad"), a)[1],
            "correct_vs_perturbed_margin",
        ),
        (
            lambda a: (a.__setitem__("unsafe_false_accepts", False), a)[1],
            "unsafe_false_accepts",
        ),
        (
            lambda a: (
                a["MODEL_SPECS"]["value"]["flagship_dense"].__setitem__("hf_id", "wrong"),
                a,
            )[1],
            "MODEL_SPECS hf_id",
        ),
        (
            lambda a: (a["tests_run"].__setitem__("value", []), a)[1],
            "ready artifact requires tests_run",
        ),
    ]

    for mutate, expected in malformed_cases:
        bad = mutate(deepcopy(artifact))
        with pytest.raises(ValueError, match=expected):
            exp.validate_artifact(bad)


def test_req_verify_5354_feature_helpers_fail_closed() -> None:
    """REQ-VERIFY-5354: feature helpers expose missing rows instead of guessing."""

    rows = exp.build_carry_token_energy_rows([])
    first = rows[0]

    assert first["feature_complete"] is False
    assert "completion_probabilities" in first["missing_features"]
    assert "correct_target_logprob" in first["missing_features"]
    assert exp._target_logprob([{"token": " YES,", "logprob": -0.5}], ("yes",)) == -0.5
    assert exp._target_logprob([{"token": "maybe", "logprob": -0.5}], ("yes",)) is None
    assert exp._normalise_token(" yes!") == "yes"
    assert exp._feature_complete_rate([]) == 0.0
