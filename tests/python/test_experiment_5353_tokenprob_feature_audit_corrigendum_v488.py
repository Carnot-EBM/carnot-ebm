"""Tests for Exp5353 token-probability feature audit corrigendum.

Spec refs: REQ-VERIFY-5353, SCENARIO-VERIFY-5353.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5353_tokenprob_feature_audit_corrigendum_v488 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
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


def _runtime_artifact(model_path: Path, server_path: Path) -> dict[str, Any]:
    return {
        "experiment_id": {"value": "experiment_5337_sota_runtime_corrigendum_multimodel_v487"},
        "status": {"value": "complete"},
        "honest_verdict": {"value": "complete: clean"},
        "inference_substrate": {"value": "live_llm_inference"},
        "methodology_duration_s": 62.0,
        "sota_runtime_clean_receipt_ready": True,
        "runtime_unblocked_min_one_mandated": True,
        "MODEL_SPECS": {"value": _model_specs(model_path)},
        "selected_backend_command": {
            "value": {
                "backend_kind": "llama-cli",
                "backend_variant": "llama-cli-single-turn-batch512",
                "command": [str(server_path.with_name("llama-cli")), "-m", str(model_path)],
                "model_path": str(model_path),
                "model_role": "flagship_dense",
            }
        },
        "runtime_corrigendum_receipt": {
            "value": {
                "model_role": "flagship_dense",
                "hf_id": "unsloth/gemma-4-31B-it-GGUF",
                "model_path": str(model_path),
                "clean_receipt_ready": True,
            }
        },
    }


def _internal_artifact(model_path: Path, *, tokenprob: bool = True) -> dict[str, Any]:
    return {
        "status": {"value": "complete" if tokenprob else "blocked"},
        "honest_verdict": {
            "value": "complete: token_probability_receipt_ready"
            if tokenprob
            else "blocked_internal_signal_unavailable"
        },
        "token_probability_available": tokenprob,
        "logits_available": False,
        "attention_available": False,
        "hidden_state_proxy_available": False,
        "token_timing_available": True,
        "internal_signal_receipt_ready": tokenprob,
        "external_text_scorer_reopened": False,
        "no_quality_claim": True,
        "selected_model_spec": {
            "value": {
                "role": "flagship_dense",
                "hf_id": "unsloth/gemma-4-31B-it-GGUF",
                "model_path": str(model_path),
                "status": "local_gguf_resolved",
                "autotokenizer_used": False,
            }
        },
        "backend_option_surface": {
            "value": {
                "option_flags": {
                    "token_probability_option": tokenprob,
                    "logit_export_option": False,
                    "attention_export_option": False,
                    "hidden_state_proxy_option": False,
                    "aggregate_timing_option": True,
                    "raw_output_option": True,
                }
            }
        },
        "missing_backend_features": {
            "value": [] if tokenprob else ["token_probability_rows_unavailable"]
        },
    }


def _receipt_schema(*, tokenprob: bool = True) -> dict[str, Any]:
    return {
        "schema": exp.exp5331.RECEIPT_SCHEMA,
        "internal_signal_receipt_ready": tokenprob,
        "receipt_kind": "token_probability" if tokenprob else "none",
        "availability": {
            "token_probability_available": tokenprob,
            "logits_available": False,
            "attention_available": False,
            "hidden_state_proxy_available": False,
            "token_timing_available": True,
            "raw_output_receipt_available": True,
        },
        "missing_backend_features": [] if tokenprob else ["token_probability_rows_unavailable"],
        "external_text_scorer_reopened": False,
        "no_quality_claim": True,
    }


def _tiny_receipt(model_path: Path, *, tokenprob: bool = True) -> dict[str, Any]:
    completion_probabilities = (
        [
            {
                "id": 10,
                "token_checksum": "tok-a",
                "logprob": -0.25,
                "top_logprobs": [
                    {"id": 10, "token_checksum": "tok-a", "logprob": -0.25},
                    {"id": 11, "token_checksum": "tok-b", "logprob": -2.0},
                ],
            }
        ]
        if tokenprob
        else []
    )
    return {
        "schema": exp.exp5331.TINY_RECEIPT_SCHEMA,
        "receipt_kind": "token_probability" if tokenprob else "none",
        "model_role": "flagship_dense",
        "model_hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "model_path": str(model_path),
        "backend_kind": "llama-server",
        "endpoint": "/completion",
        "completion_probabilities": completion_probabilities,
        "token_probability": {
            "availability": "available" if tokenprob else "capability_absent",
            "completion_probability_count": len(completion_probabilities),
            "top_logprob_row_count": 2 if tokenprob else 0,
        },
        "logits": {"availability": "capability_absent", "top_logits": []},
        "attention": {"availability": "capability_absent", "summary": {}},
        "hidden_state_proxy": {"availability": "capability_absent", "summary": {}},
        "token_timing": {
            "availability": "available",
            "timings": {
                "prompt_n": 5,
                "predicted_n": 1,
                "predicted_per_token_ms": 1.5,
            },
        },
        "raw_output": {
            "availability": "available",
            "tokens_evaluated": 5,
            "tokens_predicted": 1,
            "content_checksum": "raw",
        },
        "quality_interpretation": None,
    }


def _base_paths(tmp_path: Path, *, tokenprob: bool = True) -> dict[str, Path]:
    model_path = tmp_path / "gemma-4-31B-it-Q4_K_M.gguf"
    model_path.write_bytes(b"GGUF\x03\x00\x00\x00" + b"\x00" * 32)
    server_path = tmp_path / "llama-server"
    server_path.write_text("#!/bin/sh\n", encoding="utf-8")
    server_path.with_name("llama-cli").write_text("#!/bin/sh\n", encoding="utf-8")
    runtime_path = _write_json(
        tmp_path / exp.exp5337.RESULT_RELATIVE_PATH, _runtime_artifact(model_path, server_path)
    )
    internal_path = _write_json(
        tmp_path / exp.exp5331.RESULT_RELATIVE_PATH,
        _internal_artifact(model_path, tokenprob=tokenprob),
    )
    schema_path = _write_json(
        tmp_path / exp.exp5331.RECEIPT_SCHEMA_RELATIVE_PATH,
        _receipt_schema(tokenprob=tokenprob),
    )
    tiny_path = _write_json(
        tmp_path / exp.exp5331.TINY_RECEIPT_RELATIVE_PATH,
        _tiny_receipt(model_path, tokenprob=tokenprob),
    )
    return {
        "model": model_path,
        "server": server_path,
        "runtime": runtime_path,
        "internal": internal_path,
        "schema": schema_path,
        "tiny": tiny_path,
    }


def _preconditions(paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "gpu_visible": True,
        "nvidia_smi": {"ok": True, "stdout": "0, RTX 3090, 24576, 24120"},
        "free_vram_mb": 24120,
        "binary_paths": {
            "llama-server": str(paths["server"]),
            "llama-cli": str(paths["server"].with_name("llama-cli")),
        },
        "cuda_backend_evidence": True,
    }


def _clean_feature_probe(**kwargs: Any) -> dict[str, Any]:
    receipts = []
    for index, prompt in enumerate(kwargs["prompts"]):
        receipts.append(
            {
                "prompt_id": prompt["prompt_id"],
                "prompt_checksum": exp.sha16(prompt["prompt"]),
                "wall_clock_s": 0.75 + index,
                "response_json": {
                    "tokens_evaluated": 7 + index,
                    "tokens_predicted": 1,
                    "timings": {"predicted_per_token_ms": 2.5 + index},
                    "completion_probabilities": [
                        {
                            "id": 100 + index,
                            "token_checksum": f"token-{index}",
                            "logprob": -0.125 - index,
                            "top_logprobs": [
                                {
                                    "id": 100 + index,
                                    "token_checksum": f"token-{index}",
                                    "logprob": -0.125 - index,
                                },
                                {
                                    "id": 200 + index,
                                    "token_checksum": f"alt-{index}",
                                    "logprob": -2.25 - index,
                                },
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
        "wall_clock_s": 9.25,
        "feature_audit_wall_clock_s": 3.75,
        "prompt_receipts": receipts,
    }


def test_req_verify_5353_spec_declares_feature_audit_contract() -> None:
    """REQ-VERIFY-5353: OpenSpec anchors the feature-audit receipt."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5353") :]
    normalized_section = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5353",
        "SCENARIO-VERIFY-5353",
        str(exp.RESULT_RELATIVE_PATH),
        "blocked_tokenprob_features_unavailable",
        "tokenprob_feature_rows_ready",
        "per-token logprob rows",
        "top-k token alternatives",
        "raw logits",
        "attention",
        "hidden states",
        "token timing",
        "prompt/completion token split",
        "feature_audit_only",
        "live_llm_inference",
        "external_text_scorer_reopened=false",
        "no_quality_claim=true",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "scripts/research_conductor.py",
    ):
        assert marker in section

    for field in exp.REQUIRED_WRAPPED_FIELDS:
        assert f"`{field}`" in section
        assert " ".join(exp.FIELD_PRINCIPLES[field].split()) in normalized_section


def test_scenario_verify_5353_emits_tiny_clean_feature_receipt(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5353: real per-token rows open the feature receipt gate."""

    paths = _base_paths(tmp_path, tokenprob=True)
    artifact = exp.run(
        root=tmp_path,
        result_path=tmp_path / exp.RESULT_RELATIVE_PATH,
        exp5337_artifact_path=paths["runtime"],
        exp5331_artifact_path=paths["internal"],
        exp5331_schema_path=paths["schema"],
        exp5331_tiny_receipt_path=paths["tiny"],
        preconditions_provider=lambda: _preconditions(paths),
        feature_probe=_clean_feature_probe,
        tests_run=[{"command": "unit exp5353", "outcome": "passed"}],
    )

    assert json.loads((tmp_path / exp.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    exp.validate_artifact(artifact)
    assert artifact["status"]["value"] == "complete"
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["inference_substrate"]["value"] == "live_llm_inference"
    assert artifact["selected_model_spec"]["value"]["hf_id"] == "unsloth/gemma-4-31B-it-GGUF"
    assert artifact["per_token_logprob_available"] is True
    assert artifact["topk_alternatives_available"] is True
    assert artifact["logits_available"] is False
    assert artifact["attention_available"] is False
    assert artifact["hidden_states_available"] is False
    assert artifact["token_timing_available"] is True
    assert artifact["prompt_completion_token_split_available"] is True
    assert artifact["tokenprob_feature_row_count"] == len(exp.AUDIT_PROMPTS)
    assert artifact["methodology_duration_s"] == pytest.approx(9.25)
    assert artifact["feature_audit_duration_s"] == pytest.approx(3.75)
    assert artifact["methodology_duration_s"] != artifact["feature_audit_duration_s"]
    assert artifact["missing_feature_names"] == []
    assert artifact["tokenprob_feature_rows_ready"] is True
    assert artifact["external_text_scorer_reopened"] is False
    assert artifact["no_quality_claim"] is True
    assert {row["hf_id"] for row in artifact["MODEL_SPECS"]["value"].values()} == set(
        exp.EXPECTED_MODEL_IDS
    )


def test_scenario_verify_5353_blocks_when_per_token_rows_unavailable(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5353: aggregate-only backends block without live probing."""

    paths = _base_paths(tmp_path, tokenprob=False)
    calls: list[str] = []
    artifact = exp.run(
        root=tmp_path,
        result_path=tmp_path / "blocked.json",
        exp5337_artifact_path=paths["runtime"],
        exp5331_artifact_path=paths["internal"],
        exp5331_schema_path=paths["schema"],
        exp5331_tiny_receipt_path=paths["tiny"],
        preconditions_provider=lambda: _preconditions(paths),
        feature_probe=lambda **kwargs: calls.append(kwargs["selected_model_spec"]["hf_id"]) or {},
        tests_run=[{"command": "unit blocked", "outcome": "passed"}],
    )

    exp.validate_artifact(artifact)
    assert calls == []
    assert artifact["status"]["value"] == "blocked"
    assert artifact["honest_verdict"]["value"] == "blocked_tokenprob_features_unavailable"
    assert artifact["inference_substrate"]["value"] == "feature_audit_only"
    assert artifact["per_token_logprob_available"] is False
    assert artifact["topk_alternatives_available"] is False
    assert artifact["tokenprob_feature_row_count"] == 0
    assert "per_token_logprob" in artifact["missing_feature_names"]
    assert "topk_alternatives" in artifact["missing_feature_names"]
    assert artifact["tokenprob_feature_rows_ready"] is False
    assert artifact["preconditions_checked"]["value"]["token_probability_api_available"] is False


def test_req_verify_5353_repository_artifact_is_schema_valid() -> None:
    """REQ-VERIFY-5353: checked-in deliverable keeps the schema stable."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    exp.validate_artifact(artifact)
    assert artifact["experiment_id"]["value"] == exp.EXPERIMENT_ID
    assert artifact["honest_verdict"]["value"].startswith(("complete:", "blocked_"))
    assert artifact["external_text_scorer_reopened"] is False
    assert artifact["no_quality_claim"] is True


def test_req_verify_5353_validation_rejects_contract_drift(tmp_path: Path) -> None:
    """REQ-VERIFY-5353: validation rejects scorer, timing, and readiness drift."""

    paths = _base_paths(tmp_path, tokenprob=True)
    artifact = exp.run(
        root=tmp_path,
        result_path=tmp_path / "clean.json",
        exp5337_artifact_path=paths["runtime"],
        exp5331_artifact_path=paths["internal"],
        exp5331_schema_path=paths["schema"],
        exp5331_tiny_receipt_path=paths["tiny"],
        preconditions_provider=lambda: _preconditions(paths),
        feature_probe=_clean_feature_probe,
        tests_run=[{"command": "unit schema", "outcome": "passed"}],
    )

    malformed_cases = [
        (lambda a: (a["honest_verdict"].__setitem__("value", "done"), a)[1], "honest_verdict"),
        (
            lambda a: (a["inference_substrate"].__setitem__("value", "cached_text"), a)[1],
            "inference_substrate",
        ),
        (
            lambda a: (a.__setitem__("external_text_scorer_reopened", True), a)[1],
            "external_text_scorer_reopened",
        ),
        (lambda a: (a.__setitem__("no_quality_claim", False), a)[1], "no_quality_claim"),
        (
            lambda a: (a.__setitem__("tokenprob_feature_row_count", 0), a)[1],
            "ready artifact requires tokenprob_feature_row_count",
        ),
        (
            lambda a: (
                a.__setitem__("feature_audit_duration_s", a["methodology_duration_s"]),
                a,
            )[1],
            "duration fields must be independent",
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
                a["MODEL_SPECS"]["value"]["flagship_dense"].__setitem__(
                    "autotokenizer_used", True
                ),
                a,
            )[1],
            "autotokenizer_used",
        ),
    ]

    for mutate, expected in malformed_cases:
        bad = mutate(deepcopy(artifact))
        with pytest.raises(ValueError, match=expected):
            exp.validate_artifact(bad)


def test_req_verify_5353_feature_surface_helpers_are_precise(tmp_path: Path) -> None:
    """REQ-VERIFY-5353: helper audits distinguish each backend feature surface."""

    paths = _base_paths(tmp_path, tokenprob=True)
    tiny = _tiny_receipt(paths["model"], tokenprob=True)
    schema = _receipt_schema(tokenprob=True)
    internal = _internal_artifact(paths["model"], tokenprob=True)
    audit = exp.audit_backend_features(tiny, schema, internal)

    assert audit["per_token_logprob_available"] is True
    assert audit["topk_alternatives_available"] is True
    assert audit["logits_available"] is False
    assert audit["attention_available"] is False
    assert audit["hidden_states_available"] is False
    assert audit["token_timing_available"] is True
    assert audit["prompt_completion_token_split_available"] is True
    assert audit["missing_feature_names"] == ["logits", "attention", "hidden_states"]


def test_req_verify_5353_helper_branches_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-5353: defensive helper branches keep feature claims narrow."""

    paths = _base_paths(tmp_path, tokenprob=True)
    model_specs = {"flagship_dense": {"hf_id": "wrong", "autotokenizer_used": False}}
    internal_selected = {
        "selected_model_spec": {
            "value": {
                "role": "flagship_dense",
                "hf_id": "unsloth/gemma-4-31B-it-GGUF",
                "model_path": str(paths["model"]),
            }
        }
    }
    assert (
        exp._selected_model_from_sources({}, internal_selected, {}, model_specs)["hf_id"]
        == "unsloth/gemma-4-31B-it-GGUF"
    )
    tiny_selected = {
        "model_role": "flagship_dense",
        "model_hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "model_path": str(paths["model"]),
    }
    assert (
        exp._selected_model_from_sources({}, {}, tiny_selected, model_specs)["model_path"]
        == str(paths["model"])
    )
    assert exp._selected_model_from_sources({}, {}, {}, model_specs) is None
    assert exp._selected_backend_command({}) is None
    assert exp._raw_or_wrapped_value({"x": 1}, "x") == 1
    assert exp._numeric("not numeric") is None
    assert exp._schema_availability({}, "token_probability_available") is False
    assert exp._token_checksum({"token": "hello"}) == exp.sha16("hello")
    assert exp._token_checksum({"id": 42}) == exp.sha16("42")
    assert exp._token_checksum({}) is None
    assert exp._top_logprob_rows({"completion_probabilities": [{"id": 1, "logprob": -1.0}]}) == [
        {"id": 1, "logprob": -1.0}
    ]
    assert exp._normalise_top_alternatives([{"id": 2, "logprob": "bad"}]) == []
    assert exp.audit_backend_features({}, {}, {})["missing_feature_names"] == [
        "per_token_logprob",
        "topk_alternatives",
        "logits",
        "attention",
        "hidden_states",
        "token_timing",
        "prompt_completion_token_split",
    ]

    fallback_rows = exp.build_feature_rows(
        {
            "case_receipts": [
                "not a receipt",
                {"case_id": "missing_logprob", "completion_probabilities": [{"id": 1}]},
                {
                    "case_id": "fallback_top",
                    "completion_probabilities": [{"id": 2, "token": "z", "logprob": -1.25}],
                },
            ]
        },
        prompts=({"prompt_id": "fallback_top", "prompt": "fallback prompt"},),
    )
    assert fallback_rows == [
        {
            "completion_tokens_predicted": None,
            "feature_source": "backend_completion_probabilities",
            "logprob": -1.25,
            "prompt_checksum": exp.sha16("fallback prompt"),
            "prompt_id": "fallback_top",
            "prompt_tokens_evaluated": None,
            "quality_interpretation": None,
            "token_checksum": exp.sha16("z"),
            "token_id": 2,
            "token_index": 0,
            "token_timing_ms": None,
            "top_alternative_count": 1,
            "top_alternatives": [
                {
                    "logprob": -1.25,
                    "rank": 0,
                    "token_checksum": exp.sha16("z"),
                    "token_id": 2,
                }
            ],
        }
    ]

    assert "selected_model_spec_unavailable" in exp._precondition_blockers(
        selected_model_spec=None,
        preconditions={"gpu_visible": False},
        feature_audit={"per_token_logprob_available": False, "topk_alternatives_available": False},
    )
    assert exp._precondition_blockers(
        selected_model_spec={"hf_id": "not mandated"},
        preconditions={"gpu_visible": True},
        feature_audit={"per_token_logprob_available": True, "topk_alternatives_available": True},
    ) == ["selected_model_not_mandated"]
    assert exp._precondition_blockers(
        selected_model_spec={
            "hf_id": "unsloth/gemma-4-31B-it-GGUF",
            "model_path": str(tmp_path / "missing.gguf"),
            "autotokenizer_used": True,
        },
        preconditions={"gpu_visible": True},
        feature_audit={"per_token_logprob_available": True, "topk_alternatives_available": True},
    ) == ["selected_model_file_missing", "autotokenizer_used_for_gguf"]
    assert exp._honest_verdict(False, ["gpu_not_visible"], False).startswith("blocked_preconditions")
    assert exp._honest_verdict(False, [], True) == "blocked_tokenprob_feature_rows_not_ready"


def test_req_verify_5353_blocked_live_probe_names_missing_rows(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5353: attempted probes with no rows remain blocked."""

    paths = _base_paths(tmp_path, tokenprob=True)
    artifact = exp.run(
        root=tmp_path,
        result_path=tmp_path / "no-rows.json",
        exp5337_artifact_path=paths["runtime"],
        exp5331_artifact_path=paths["internal"],
        exp5331_schema_path=paths["schema"],
        exp5331_tiny_receipt_path=paths["tiny"],
        preconditions_provider=lambda: _preconditions(paths),
        feature_probe=lambda **_kwargs: {
            "status": "completed",
            "backend_kind": "llama-server",
            "endpoint": "/completion",
            "wall_clock_s": 0.0,
            "feature_audit_wall_clock_s": -1.0,
            "prompt_receipts": [],
        },
        tests_run=[],
    )

    exp.validate_artifact(artifact)
    assert artifact["status"]["value"] == "blocked"
    assert "per_token_logprob" in artifact["missing_feature_names"]
    assert "topk_alternatives" in artifact["missing_feature_names"]
    assert "methodology_duration_s" in artifact["missing_feature_names"]
    assert "feature_audit_duration_s" in artifact["missing_feature_names"]
    assert "tests_run" in artifact["missing_feature_names"]


def test_req_verify_5353_validation_aggregates_schema_errors(tmp_path: Path) -> None:
    """REQ-VERIFY-5353: validation reports malformed schema fields fail-closed."""

    paths = _base_paths(tmp_path, tokenprob=True)
    artifact = exp.run(
        root=tmp_path,
        result_path=tmp_path / "clean.json",
        exp5337_artifact_path=paths["runtime"],
        exp5331_artifact_path=paths["internal"],
        exp5331_schema_path=paths["schema"],
        exp5331_tiny_receipt_path=paths["tiny"],
        preconditions_provider=lambda: _preconditions(paths),
        feature_probe=_clean_feature_probe,
        tests_run=[{"command": "unit aggregate validation", "outcome": "passed"}],
    )

    bad = deepcopy(artifact)
    del bad["per_token_logprob_available"]
    bad["experiment_id"] = {"principle": "bad", "value": "wrong"}
    bad["milestone"]["value"] = "wrong"
    bad["status"]["value"] = "weird"
    bad["honest_verdict"]["value"] = 123
    bad["inference_substrate"]["value"] = "cached_text"
    bad["topk_alternatives_available"] = "yes"
    bad["tokenprob_feature_row_count"] = True
    bad["missing_feature_names"] = "bad"
    bad["methodology_duration_s"] = False
    bad["feature_audit_duration_s"] = "bad"
    bad["MODEL_SPECS"]["value"] = "bad"
    bad["selected_model_spec"]["value"] = "bad"
    bad["tests_run"]["value"] = "bad"
    bad["tokenprob_feature_rows_ready"] = "bad"

    with pytest.raises(ValueError) as exc_info:
        exp.validate_artifact(bad)

    message = str(exc_info.value)
    for expected in (
        "missing required field: per_token_logprob_available",
        "experiment_id must be principle wrapped",
        "experiment_id mismatch",
        "milestone mismatch",
        "status must be complete or blocked",
        "honest_verdict",
        "inference_substrate",
        "topk_alternatives_available must be a bare boolean",
        "tokenprob_feature_row_count must be a bare integer",
        "missing_feature_names must be a bare list",
        "methodology_duration_s must be numeric",
        "feature_audit_duration_s must be numeric",
        "MODEL_SPECS must be an object",
        "selected_model_spec must be null or object",
        "tests_run must be a list",
        "tokenprob_feature_rows_ready must be a bare boolean",
    ):
        assert expected in message
