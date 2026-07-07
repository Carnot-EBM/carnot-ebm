"""Tests for Exp5345 token-probability energy corrigendum.

Spec refs: REQ-VERIFY-5345, SCENARIO-VERIFY-5345.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5345_tokenprob_energy_corrigendum_v487 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def _principle_wrap(principle: str, value: Any) -> dict[str, Any]:
    return {"principle": principle, "value": value}


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


def _runtime_artifact(model_path: Path, server_path: Path, *, clean: bool = True) -> dict[str, Any]:
    selected = _model_specs(model_path)["flagship_dense"]
    return {
        "experiment_id": {"value": "experiment_5337_sota_runtime_corrigendum_multimodel_v487"},
        "status": {"value": "complete" if clean else "blocked"},
        "honest_verdict": {"value": "complete: clean" if clean else "blocked_runtime"},
        "inference_substrate": {"value": "live_llm_inference"},
        "methodology_duration_s": 62.5 if clean else 12.0,
        "sota_runtime_clean_receipt_ready": clean,
        "runtime_unblocked_min_one_mandated": clean,
        "quality_claim_permitted": False,
        "MODEL_SPECS": {"value": _model_specs(model_path)},
        "selected_backend_command": {
            "value": {
                "backend_kind": "llama-cli",
                "backend_variant": "llama-cli-single-turn-batch512",
                "command": [str(server_path.with_name("llama-cli")), "-m", str(model_path)],
                "model_path": str(model_path),
                "model_role": "flagship_dense",
                "prompt": "Write eight lowercase color words separated by spaces.",
                "n_predict": 8,
                "timeout_s": 240.0,
            }
        },
        "runtime_corrigendum_receipt": {
            "value": {
                "model_role": "flagship_dense",
                "hf_id": selected["hf_id"],
                "model_path": str(model_path),
                "clean_receipt_ready": clean,
            }
        },
        "preconditions_checked": {
            "value": {
                "gpu_visible": clean,
                "blocked_preconditions": [] if clean else ["gpu_not_visible"],
                "binary_paths": {
                    "llama-server": str(server_path),
                    "llama-cli": str(server_path.with_name("llama-cli")),
                },
            }
        },
    }


def _internal_artifacts(
    tmp_path: Path,
    model_path: Path,
    *,
    token_probability_available: bool = True,
) -> tuple[Path, Path, Path]:
    tiny_path = tmp_path / exp.exp5331.TINY_RECEIPT_RELATIVE_PATH
    schema_path = tmp_path / exp.exp5331.RECEIPT_SCHEMA_RELATIVE_PATH
    harness_path = tmp_path / exp.exp5331.RESULT_RELATIVE_PATH
    tiny = {
        "schema": exp.exp5331.TINY_RECEIPT_SCHEMA,
        "receipt_kind": "token_probability",
        "model_role": "flagship_dense",
        "model_hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "model_path": str(model_path),
        "backend_kind": "llama-server",
        "endpoint": "/completion",
        "completion_probabilities": [
            {
                "id": 1,
                "token_checksum": "yes",
                "logprob": -0.1,
                "top_logprobs": [{"id": 1, "token_checksum": "yes", "logprob": -0.1}],
            }
        ]
        if token_probability_available
        else [],
        "token_probability": {
            "availability": "available" if token_probability_available else "capability_absent",
            "completion_probability_count": 1 if token_probability_available else 0,
            "top_logprob_row_count": 1 if token_probability_available else 0,
        },
        "logits": {"availability": "capability_absent", "top_logits": []},
        "attention": {"availability": "capability_absent", "summary": {}},
        "token_timing": {"availability": "available", "timings": {"predicted_per_token_ms": 1.0}},
        "quality_interpretation": None,
    }
    schema = {
        "schema": exp.exp5331.RECEIPT_SCHEMA,
        "internal_signal_receipt_ready": token_probability_available,
        "receipt_path": str(tiny_path),
        "receipt_kind": "token_probability" if token_probability_available else "none",
        "availability": {
            "token_probability_available": token_probability_available,
            "logits_available": False,
            "attention_available": False,
            "token_timing_available": True,
        },
        "missing_backend_features": [] if token_probability_available else ["token_probability"],
        "external_text_scorer_reopened": False,
        "no_quality_claim": True,
    }
    harness = {
        "status": {"value": "complete" if token_probability_available else "blocked"},
        "honest_verdict": {
            "value": "complete: token_probability_receipt_ready"
            if token_probability_available
            else "blocked_internal_signal_unavailable"
        },
        "token_probability_available": token_probability_available,
        "logits_available": False,
        "attention_available": False,
        "internal_signal_receipt_ready": token_probability_available,
        "external_text_scorer_reopened": False,
        "no_quality_claim": True,
        "selected_model_spec": {"value": _model_specs(model_path)["flagship_dense"]},
        "tiny_receipt_path": {"value": str(tiny_path)},
        "receipt_schema_path": {"value": str(schema_path)},
    }
    _write_json(tiny_path, tiny)
    _write_json(schema_path, schema)
    _write_json(harness_path, harness)
    return harness_path, schema_path, tiny_path


def _preconditions(server_path: Path, *, gpu_visible: bool = True) -> dict[str, Any]:
    return {
        "gpu_visible": gpu_visible,
        "nvidia_smi": {"ok": gpu_visible, "stdout": "0, NVIDIA RTX 3090, 24576, 24120"},
        "free_vram_mb": 24120 if gpu_visible else 0,
        "binary_paths": {
            "llama-server": str(server_path),
            "llama-cli": str(server_path.with_name("llama-cli")),
        },
        "cuda_backend_evidence": gpu_visible,
    }


def _live_probe(**_kwargs: Any) -> dict[str, Any]:
    receipts = []
    for case in exp.DIAGNOSTIC_CASES:
        correct = case.correct_aliases[0]
        perturbed = case.perturbed_aliases[0]
        receipts.append(
            {
                "case_id": case.case_id,
                "prompt": case.prompt,
                "wall_clock_s": 16.5,
                "response_json": {
                    "content": f" {correct}",
                    "timings": {"predicted_per_token_ms": 22.0},
                    "completion_probabilities": [
                        {
                            "token": f" {correct}",
                            "logprob": -0.10,
                            "top_logprobs": [
                                {"token": f" {correct}", "logprob": -0.10},
                                {"token": f" {perturbed}", "logprob": -2.40},
                            ],
                        }
                    ],
                },
            }
        )
    return {
        "status": "completed",
        "backend_kind": "llama-server",
        "endpoint": "/completion",
        "wall_clock_s": 66.0,
        "round_count": 4,
        "case_receipts": receipts,
    }


def _missing_perturbed_probe(**_kwargs: Any) -> dict[str, Any]:
    payload = _live_probe()
    for receipt in payload["case_receipts"]:
        receipt["response_json"]["completion_probabilities"][0]["top_logprobs"] = [
            receipt["response_json"]["completion_probabilities"][0]["top_logprobs"][0]
        ]
    return payload


def _empty_probability_probe(**_kwargs: Any) -> dict[str, Any]:
    return {
        "status": "completed",
        "backend_kind": "llama-server",
        "endpoint": "/completion",
        "wall_clock_s": 66.0,
        "round_count": 1,
        "case_receipts": [],
    }


def _short_probe(**_kwargs: Any) -> dict[str, Any]:
    payload = _live_probe()
    payload["wall_clock_s"] = 12.0
    return payload


def test_req_verify_5345_spec_declares_corrigendum_contract() -> None:
    """REQ-VERIFY-5345: OpenSpec anchors the token-probability corrigendum."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5345") :]
    normalized_section = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5345",
        "SCENARIO-VERIFY-5345",
        str(exp.RESULT_RELATIVE_PATH),
        "live_llm_inference",
        "aggregation_from_upstream_artifacts",
        "token_probability_available",
        "token_energy_feature_rows",
        "internal_energy_corrigendum_clean",
        "external_text_scorer_reopened=false",
        "no_quality_claim=true",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "scripts/research_conductor.py",
    ):
        assert marker in section

    for field, principle in exp.FIELD_PRINCIPLES.items():
        if field in exp.REQUIRED_WRAPPED_FIELDS:
            assert f"`{field}`" in section
            assert " ".join(principle.split()) in normalized_section


def test_scenario_verify_5345_clean_live_tokenprob_energy_diagnostic(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5345: clean live top-logprobs yield transparent energy rows."""

    model_path = _write_gguf(tmp_path / "gemma-4-31B-it-Q4_K_M.gguf")
    server_path = tmp_path / "llama-server"
    server_path.write_text("#!/bin/sh\n", encoding="utf-8")
    server_path.with_name("llama-cli").write_text("#!/bin/sh\n", encoding="utf-8")
    runtime_path = _write_json(
        tmp_path / exp.exp5337.RESULT_RELATIVE_PATH, _runtime_artifact(model_path, server_path)
    )
    harness_path, schema_path, tiny_path = _internal_artifacts(tmp_path, model_path)

    artifact = exp.run(
        root=tmp_path,
        result_path=tmp_path / exp.RESULT_RELATIVE_PATH,
        exp5337_artifact_path=runtime_path,
        exp5331_artifact_path=harness_path,
        exp5331_schema_path=schema_path,
        exp5331_tiny_receipt_path=tiny_path,
        preconditions_provider=lambda: _preconditions(server_path),
        token_probability_probe=_live_probe,
        tests_run=[{"command": "unit exp5345", "outcome": "passed"}],
    )

    assert json.loads((tmp_path / exp.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    exp.validate_artifact(artifact)
    assert artifact["status"]["value"] == "complete"
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["inference_substrate"]["value"] == exp.INFERENCE_SUBSTRATE_LIVE
    assert artifact["selected_model_spec"]["value"]["hf_id"] == "unsloth/gemma-4-31B-it-GGUF"
    assert artifact["token_probability_available"] is True
    assert artifact["logits_available"] is False
    assert artifact["attention_available"] is False
    assert artifact["diagnostic_case_count"] == len(exp.DIAGNOSTIC_CASES)
    assert artifact["methodology_duration_s"] >= 60.0
    assert artifact["external_text_scorer_reopened"] is False
    assert artifact["no_quality_claim"] is True
    assert artifact["internal_energy_corrigendum_clean"] is True
    rows = artifact["token_energy_feature_rows"]["value"]
    assert len(rows) == len(exp.DIAGNOSTIC_CASES)
    assert all(row["feature_complete"] is True for row in rows)
    assert all(row["energy_margin_perturbed_minus_correct"] > 0 for row in rows)
    assert set(artifact["MODEL_SPECS"]["value"]) == {
        spec["role"] for spec in exp.MANDATED_MODEL_SPECS
    }


def test_scenario_verify_5345_blocks_before_live_generation_when_runtime_unclean(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-5345: unclean Exp5337 uses aggregation-only blocked substrate."""

    model_path = _write_gguf(tmp_path / "gemma-4-31B-it-Q4_K_M.gguf")
    server_path = tmp_path / "llama-server"
    server_path.write_text("#!/bin/sh\n", encoding="utf-8")
    runtime_path = _write_json(
        tmp_path / exp.exp5337.RESULT_RELATIVE_PATH,
        _runtime_artifact(model_path, server_path, clean=False),
    )
    harness_path, schema_path, tiny_path = _internal_artifacts(tmp_path, model_path)
    calls: list[str] = []

    artifact = exp.run(
        root=tmp_path,
        result_path=tmp_path / "blocked.json",
        exp5337_artifact_path=runtime_path,
        exp5331_artifact_path=harness_path,
        exp5331_schema_path=schema_path,
        exp5331_tiny_receipt_path=tiny_path,
        preconditions_provider=lambda: _preconditions(server_path),
        token_probability_probe=lambda **kwargs: (
            calls.append(kwargs["selected_model_spec"]["hf_id"]) or _live_probe()
        ),
        tests_run=[{"command": "unit blocked", "outcome": "passed"}],
    )

    exp.validate_artifact(artifact)
    assert calls == []
    assert artifact["status"]["value"] == "blocked"
    assert artifact["honest_verdict"]["value"].startswith("blocked_")
    assert artifact["inference_substrate"]["value"] == exp.INFERENCE_SUBSTRATE_AGGREGATION
    assert artifact["diagnostic_case_count"] == 0
    assert artifact["token_probability_available"] is False
    assert artifact["internal_energy_corrigendum_clean"] is False
    assert "exp5337_clean_runtime_unavailable" in artifact["missing_feature_names"]


def test_scenario_verify_5345_blocks_when_target_token_features_are_missing(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5345: token probabilities alone are insufficient without target rows."""

    model_path = _write_gguf(tmp_path / "gemma-4-31B-it-Q4_K_M.gguf")
    server_path = tmp_path / "llama-server"
    server_path.write_text("#!/bin/sh\n", encoding="utf-8")
    server_path.with_name("llama-cli").write_text("#!/bin/sh\n", encoding="utf-8")
    runtime_path = _write_json(
        tmp_path / exp.exp5337.RESULT_RELATIVE_PATH, _runtime_artifact(model_path, server_path)
    )
    harness_path, schema_path, tiny_path = _internal_artifacts(tmp_path, model_path)

    artifact = exp.run(
        root=tmp_path,
        result_path=tmp_path / "missing-features.json",
        exp5337_artifact_path=runtime_path,
        exp5331_artifact_path=harness_path,
        exp5331_schema_path=schema_path,
        exp5331_tiny_receipt_path=tiny_path,
        preconditions_provider=lambda: _preconditions(server_path),
        token_probability_probe=_missing_perturbed_probe,
        tests_run=[{"command": "unit missing features", "outcome": "passed"}],
    )

    exp.validate_artifact(artifact)
    assert artifact["status"]["value"] == "blocked"
    assert artifact["inference_substrate"]["value"] == exp.INFERENCE_SUBSTRATE_LIVE
    assert artifact["token_probability_available"] is True
    assert artifact["methodology_duration_s"] >= 60.0
    assert artifact["internal_energy_corrigendum_clean"] is False
    assert any(
        name.endswith(":perturbed_target_logprob") for name in artifact["missing_feature_names"]
    )
    assert all(
        row["perturbed_target_logprob"] is None
        for row in artifact["token_energy_feature_rows"]["value"]
    )


def test_req_verify_5345_repository_artifact_is_schema_valid() -> None:
    """REQ-VERIFY-5345: checked-in deliverable keeps the required schema stable."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    exp.validate_artifact(artifact)
    assert artifact["experiment_id"]["value"] == exp.EXPERIMENT_NAME
    assert artifact["honest_verdict"]["value"].startswith(("complete:", "blocked_"))
    assert artifact["external_text_scorer_reopened"] is False
    assert artifact["no_quality_claim"] is True


def test_req_verify_5345_validation_rejects_contract_drift(tmp_path: Path) -> None:
    """REQ-VERIFY-5345: schema validation rejects scorer, duration, and field drift."""

    model_path = _write_gguf(tmp_path / "gemma-4-31B-it-Q4_K_M.gguf")
    server_path = tmp_path / "llama-server"
    server_path.write_text("#!/bin/sh\n", encoding="utf-8")
    server_path.with_name("llama-cli").write_text("#!/bin/sh\n", encoding="utf-8")
    runtime_path = _write_json(
        tmp_path / exp.exp5337.RESULT_RELATIVE_PATH, _runtime_artifact(model_path, server_path)
    )
    harness_path, schema_path, tiny_path = _internal_artifacts(tmp_path, model_path)
    artifact = exp.run(
        root=tmp_path,
        result_path=tmp_path / "clean.json",
        exp5337_artifact_path=runtime_path,
        exp5331_artifact_path=harness_path,
        exp5331_schema_path=schema_path,
        exp5331_tiny_receipt_path=tiny_path,
        preconditions_provider=lambda: _preconditions(server_path),
        token_probability_probe=_live_probe,
        tests_run=[{"command": "unit schema", "outcome": "passed"}],
    )

    malformed_cases = [
        (lambda a: (a["honest_verdict"].__setitem__("value", "done"), a)[1], "honest_verdict"),
        (
            lambda a: (
                a["inference_substrate"].__setitem__(
                    "value", "verifier_ensemble_against_cached_candidates"
                ),
                a,
            )[1],
            "inference_substrate",
        ),
        (
            lambda a: (a.__setitem__("external_text_scorer_reopened", True), a)[1],
            "external_text_scorer_reopened",
        ),
        (lambda a: (a.__setitem__("no_quality_claim", False), a)[1], "no_quality_claim"),
        (
            lambda a: (a.__setitem__("diagnostic_case_count", 1.5), a)[1],
            "diagnostic_case_count",
        ),
        (
            lambda a: (a.__setitem__("methodology_duration_s", 59.0), a)[1],
            "clean artifact requires methodology_duration_s",
        ),
        (
            lambda a: (a.__setitem__("token_energy_feature_rows", []), a)[1],
            "token_energy_feature_rows",
        ),
        (
            lambda a: (
                a["MODEL_SPECS"]["value"]["flagship_dense"].__setitem__("hf_id", "wrong"),
                a,
            )[1],
            "MODEL_SPECS hf_id",
        ),
        (
            lambda a: (
                a["MODEL_SPECS"]["value"]["flagship_dense"].__setitem__("autotokenizer_used", True),
                a,
            )[1],
            "autotokenizer_used",
        ),
    ]

    for mutate, expected in malformed_cases:
        bad = mutate(deepcopy(artifact))
        with pytest.raises(ValueError, match=expected):
            exp.validate_artifact(bad)
