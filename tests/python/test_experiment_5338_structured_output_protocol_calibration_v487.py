"""Tests for Exp 5338 structured-output protocol calibration.

Spec refs: REQ-VERIFY-5338, SCENARIO-VERIFY-5338.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5338_structured_output_protocol_calibration_v487 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_prior_runtime(path: Path, binary: Path, model_path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(binary),
        "-m",
        str(model_path),
        "-p",
        "Write eight lowercase color words separated by spaces.",
        "-n",
        "8",
        "-c",
        "512",
        "-b",
        "512",
        "-ub",
        "128",
        "-ngl",
        "all",
        "-sm",
        "layer",
        "--temp",
        "0",
        "--seed",
        "5337",
        "--no-display-prompt",
        "--simple-io",
        "-st",
        "--perf",
    ]
    payload = {
        "status": {"value": "complete", "principle": "prior status"},
        "honest_verdict": {"value": "complete: clean runtime", "principle": "prior verdict"},
        "inference_substrate": {"value": "live_llm_inference", "principle": "prior substrate"},
        "sota_runtime_clean_receipt_ready": True,
        "runtime_unblocked_min_one_mandated": True,
        "selected_backend_command": {
            "value": {
                "backend_kind": "llama-cli",
                "backend_variant": "llama-cli-single-turn-batch512",
                "command": command,
                "model_path": str(model_path),
                "model_role": "flagship_dense",
                "n_predict": 8,
                "timeout_s": 240.0,
            },
            "principle": "selected command",
        },
        "MODEL_SPECS": {
            "value": {
                "flagship_moe": {
                    "role": "flagship_moe",
                    "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                    "model_path": str(path.parent / "qwen.gguf"),
                    "status": "local_gguf_resolved",
                    "autotokenizer_used": False,
                },
                "flagship_dense": {
                    "role": "flagship_dense",
                    "hf_id": "unsloth/gemma-4-31B-it-GGUF",
                    "model_path": str(model_path),
                    "status": "local_gguf_resolved",
                    "autotokenizer_used": False,
                },
                "middle_moe": {
                    "role": "middle_moe",
                    "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                    "model_path": str(path.parent / "gemma-moe.gguf"),
                    "status": "local_gguf_resolved",
                    "autotokenizer_used": False,
                },
            },
            "principle": "model specs",
        },
        "preconditions_checked": {
            "value": {
                "gpu_visible": True,
                "nvidia_smi": {"ok": True, "stdout": "0, RTX 3090, 24576, 24120, 0"},
                "blocked_preconditions": [],
            },
            "principle": "preconditions",
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_rewrite_fixture(path: Path, *, ready: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": {"value": "complete" if ready else "blocked"},
        "rewrite_state_fixture_ready": ready,
        "fixture_path": {"value": "data/rewrite_state_fixture_v486.json"},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _base_kwargs(tmp_path: Path) -> dict[str, Any]:
    binary = tmp_path / "llama-cli"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    model_path = tmp_path / "gemma-4-31B-it-Q4_K_M.gguf"
    model_path.write_text("GGUF", encoding="utf-8")
    prior_path = _write_prior_runtime(tmp_path / mod.exp5337.RESULT_RELATIVE_PATH, binary, model_path)
    rewrite_path = _write_rewrite_fixture(tmp_path / "results/experiment_5325.json")
    return {
        "root": tmp_path,
        "artifact_path": tmp_path / mod.RESULT_RELATIVE_PATH,
        "prior_runtime_path": prior_path,
        "rewrite_fixture_artifact_path": rewrite_path,
        "tests_run": [{"command": "unit exp5338", "outcome": "passed"}],
    }


def _final_json_generation(**kwargs: Any) -> dict[str, Any]:
    prompt_id = kwargs["prompt_spec"]["prompt_id"]
    if prompt_id == "protocol_fact_probe":
        final = {
            "id": prompt_id,
            "answer": "47",
            "facts": {"duration_minutes": 47, "subject": "aster-9 battery"},
        }
    else:
        final = {
            "id": prompt_id,
            "answer": "orange",
            "facts": {"code_word": "orange"},
        }
    return {
        "completed": True,
        "timed_out": False,
        "returncode": 0,
        "stdout": (
            "Loading model...\n\navailable commands:\n  /exit\n\n> prompt echo\n"
            "[Start thinking]\nbrief scratch that must stay outside final\n"
            f"{kwargs['variant']['sentinel']} "
            f"{json.dumps(final, separators=(',', ':'))}\n"
            "END_FINAL_JSON\n[ Prompt: 200.0 t/s | Generation: 40.0 t/s ]\nExiting..."
        ),
        "stderr": "",
        "wall_clock_s": 0.2,
    }


def test_req_verify_5338_spec_declares_protocol_contract() -> None:
    """REQ-VERIFY-5338: OpenSpec anchors the parse-only protocol gate."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5338") :]
    normalized_section = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5338",
        "SCENARIO-VERIFY-5338",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "live_llm_inference",
        "parse_success_rate",
        "final_json_extraction_rate",
        "thinking_text_outside_final_rate",
        "unsafe_false_accepts",
        "structured_output_protocol_ready",
        "no_quality_claim=true",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "scripts/research_conductor.py",
    ):
        assert marker in section

    for field in mod.REQUIRED_WRAPPED_FIELDS:
        assert f"`{field}`" in section
        assert " ".join(mod.FIELD_PRINCIPLES[field].split()) in normalized_section


def test_scenario_verify_5338_final_sentinel_variant_opens_protocol_gate(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5338: final-sentinel JSON after thinking opens the gate."""

    calls: list[tuple[str, str, list[str]]] = []

    def probe(**kwargs: Any) -> dict[str, Any]:
        calls.append(
            (
                kwargs["variant"]["variant_id"],
                kwargs["prompt_spec"]["prompt_id"],
                kwargs["command"],
            )
        )
        return _final_json_generation(**kwargs)

    artifact = mod.run(
        **_base_kwargs(tmp_path),
        protocol_variants=mod.DEFAULT_PROTOCOL_VARIANTS[:1],
        generation_probe=probe,
        write=True,
    )

    mod.validate_artifact(artifact)
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert len(calls) == len(mod.DEFAULT_CALIBRATION_PROMPTS)
    assert all("-p" in command and "-n" in command and "--seed" in command for _, _, command in calls)
    assert artifact["status"]["value"] == "complete"
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["inference_substrate"]["value"] == "live_llm_inference"
    assert artifact["selected_model_spec"]["value"]["hf_id"] == "unsloth/gemma-4-31B-it-GGUF"
    assert {row["hf_id"] for row in artifact["MODEL_SPECS"]["value"].values()} == set(
        mod.EXPECTED_MODEL_IDS
    )
    assert artifact["parse_success_rate"] == pytest.approx(1.0)
    assert artifact["final_json_extraction_rate"] == pytest.approx(1.0)
    assert artifact["thinking_text_outside_final_rate"] == pytest.approx(1.0)
    assert artifact["unsafe_false_accepts"] == 0
    assert artifact["structured_output_protocol_ready"] is True
    assert artifact["no_quality_claim"] is True
    variant = artifact["protocol_variants"]["value"][0]
    assert variant["ready"] is True
    assert variant["increased_token_budget"] is True
    assert variant["explicit_final_only_sentinel"] is True
    assert variant["post_think_json_extraction"] is True
    assert variant["forbids_analysis_in_final"] is True
    assert variant["parser_side_strips_llama_cpp_banners"] is True
    assert variant["stop_sequences_requested"] == ["END_FINAL_JSON"]


def test_req_verify_5338_blocks_before_generation_when_preconditions_missing(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-5338: missing clean Exp5337 runtime blocks calibration generation."""

    kwargs = _base_kwargs(tmp_path)
    kwargs["prior_runtime_path"].write_text(json.dumps({"sota_runtime_clean_receipt_ready": False}))
    calls: list[str] = []

    artifact = mod.run(
        **kwargs,
        generation_probe=lambda **kw: calls.append(kw["prompt_spec"]["prompt_id"]) or {},
        write=False,
    )

    mod.validate_artifact(artifact)
    assert calls == []
    assert artifact["status"]["value"] == "blocked"
    assert artifact["honest_verdict"]["value"].startswith("blocked_")
    assert artifact["structured_output_protocol_ready"] is False
    assert artifact["no_quality_claim"] is True
    assert "exp5337_runtime_receipt_not_clean" in artifact["preconditions_checked"]["value"][
        "blocked_preconditions"
    ]

    rewrite_blocked_kwargs = _base_kwargs(tmp_path)
    _write_rewrite_fixture(rewrite_blocked_kwargs["rewrite_fixture_artifact_path"], ready=False)
    rewrite_blocked = mod.run(
        **rewrite_blocked_kwargs,
        generation_probe=lambda **kw: calls.append(kw["prompt_spec"]["prompt_id"]) or {},
        write=False,
    )
    assert "rewrite_state_fixture_unavailable" in rewrite_blocked["preconditions_checked"]["value"][
        "blocked_preconditions"
    ]


def test_req_verify_5338_parser_rejects_unsafe_final_json_shapes() -> None:
    """REQ-VERIFY-5338: parser refuses draft, malformed, and thinking-leak finals."""

    variant = mod.DEFAULT_PROTOCOL_VARIANTS[0]
    prompt = mod.DEFAULT_CALIBRATION_PROMPTS[0]
    assert mod._rate([], "parse_success") == 0.0
    assert mod.unsafe_false_accept_count(variant, ()) == 0
    assert mod._select_best_variant([]) is None
    stripped = mod.strip_llama_cpp_banners(
        "Loading model...\n/clear history\n▄▄ ▄▄\n> echoed prompt\nkept line"
    )
    assert stripped.strip() == "kept line"

    no_sentinel = mod.score_protocol_output(
        prompt_spec=prompt,
        variant=variant,
        output_text='{"id":"x","answer":"47","facts":{}}',
        completed=True,
    )
    assert no_sentinel["parse_success"] is False

    draft_then_bad_final = (
        '{"id":"draft","answer":"wrong","facts":{}}\n'
        f"{variant['sentinel']} {{not json}}"
    )
    score = mod.score_protocol_output(
        prompt_spec=prompt,
        variant=variant,
        output_text=draft_then_bad_final,
        completed=True,
    )
    assert score["parse_success"] is False
    assert score["accepted_for_protocol"] is False
    assert score["unsafe_false_accept"] is False

    missing_key = f"{variant['sentinel']} {{\"id\":\"x\",\"answer\":\"47\"}}"
    score = mod.score_protocol_output(
        prompt_spec=prompt,
        variant=variant,
        output_text=missing_key,
        completed=True,
    )
    assert score["parse_success"] is True
    assert score["schema_keys_present"] is False
    assert score["accepted_for_protocol"] is False

    thinking_inside = (
        f"{variant['sentinel']} "
        '{"id":"x","answer":"[Start thinking] leaked","facts":{"duration_minutes":47}}'
    )
    score = mod.score_protocol_output(
        prompt_spec=prompt,
        variant=variant,
        output_text=thinking_inside,
        completed=True,
    )
    assert score["parse_success"] is True
    assert score["thinking_text_outside_final"] is False
    assert score["accepted_for_protocol"] is False
    assert mod.unsafe_false_accept_count(variant, mod.DEFAULT_CALIBRATION_PROMPTS) == 0
    assert mod._contains_thinking_marker(["clean", "[Start thinking] leak"]) is True
    assert (
        mod._thinking_text_outside_final(
            "ignored",
            {"cleaned_text": "[Start thinking]\nFINAL_JSON: {}", "json_start": None},
            {"id": "x"},
        )
        is False
    )

    appended = mod.command_for_protocol(
        ["llama-cli"],
        "prompt",
        n_predict=32,
        seed=7,
        variant={**variant, "stop_sequences_supported": True},
    )
    assert appended[1:] == [
        "-p",
        "prompt",
        "-n",
        "32",
        "--seed",
        "7",
        "--reverse-prompt",
        "END_FINAL_JSON",
    ]

    bad_context = {
        "status": {"value": "complete"},
        "inference_substrate": {"value": "live_llm_inference"},
        "sota_runtime_clean_receipt_ready": True,
        "selected_backend_command": {
            "value": {"command": ["/missing/llama-cli"], "model_role": "flagship_dense"}
        },
        "MODEL_SPECS": {
            "value": {
                "flagship_dense": {
                    "hf_id": "wrong",
                    "model_path": "/missing/model.gguf",
                },
                "flagship_moe": {"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"},
            }
        },
        "preconditions_checked": {"value": {"gpu_visible": True}},
    }
    _selected_command, _model_specs, _selected_model, blockers = mod._selected_runtime_context(
        bad_context
    )
    assert "model_specs_missing_or_drift" in blockers
    assert "selected_binary_missing" in blockers
    assert "selected_model_file_missing" in blockers

    bad_hf_context = json.loads(json.dumps(bad_context))
    bad_hf_context["MODEL_SPECS"]["value"]["middle_moe"] = {
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF"
    }
    _selected_command, _model_specs, _selected_model, blockers = mod._selected_runtime_context(
        bad_hf_context
    )
    assert "model_specs_missing_or_drift" in blockers


def test_req_verify_5338_schema_validation_rejects_contract_drift(tmp_path: Path) -> None:
    """REQ-VERIFY-5338: schema validation catches malformed calibration artifacts."""

    artifact = mod.run(
        **_base_kwargs(tmp_path),
        protocol_variants=mod.DEFAULT_PROTOCOL_VARIANTS[:1],
        generation_probe=_final_json_generation,
        write=False,
    )

    def clone() -> dict[str, Any]:
        return json.loads(json.dumps(artifact))

    malformed_cases = [
        (lambda a: (a.pop("MODEL_SPECS"), a)[1], "missing required fields"),
        (lambda a: (a.__setitem__("experiment_id", mod.EXPERIMENT_ID), a)[1], "principle-wrapped"),
        (lambda a: (a["honest_verdict"].__setitem__("value", "done"), a)[1], "honest_verdict"),
        (lambda a: (a["milestone"].__setitem__("value", "wrong"), a)[1], "milestone mismatch"),
        (lambda a: (a["status"].__setitem__("value", "running"), a)[1], "status must be complete or blocked"),
        (
            lambda a: (a["inference_substrate"].__setitem__("value", "cached"), a)[1],
            "inference_substrate mismatch",
        ),
        (
            lambda a: (a.__setitem__("parse_success_rate", "1.0"), a)[1],
            "parse_success_rate must be in [0, 1]",
        ),
        (
            lambda a: (a.__setitem__("final_json_extraction_rate", 1.5), a)[1],
            "final_json_extraction_rate must be in [0, 1]",
        ),
        (
            lambda a: (a.__setitem__("thinking_text_outside_final_rate", -0.1), a)[1],
            "thinking_text_outside_final_rate must be in [0, 1]",
        ),
        (
            lambda a: (a.__setitem__("unsafe_false_accepts", "0"), a)[1],
            "unsafe_false_accepts must be a bare integer",
        ),
        (
            lambda a: (a.__setitem__("structured_output_protocol_ready", "yes"), a)[1],
            "structured_output_protocol_ready must be a bare boolean",
        ),
        (
            lambda a: (a.__setitem__("no_quality_claim", False), a)[1],
            "no_quality_claim must be bare true",
        ),
        (
            lambda a: (a["MODEL_SPECS"]["value"].pop("middle_moe"), a)[1],
            "MODEL_SPECS roles mismatch",
        ),
        (
            lambda a: (a["MODEL_SPECS"]["value"]["flagship_dense"].__setitem__("hf_id", "wrong"), a)[1],
            "hf_id mismatch",
        ),
        (
            lambda a: (a["tests_run"].__setitem__("value", "bad"), a)[1],
            "tests_run must be a list",
        ),
        (
            lambda a: (a["tests_run"].__setitem__("principle", "wrong"), a)[1],
            "tests_run must be principle-wrapped",
        ),
        (
            lambda a: (a["selected_model_spec"].__setitem__("value", "bad"), a)[1],
            "selected_model_spec must be an object or null",
        ),
        (
            lambda a: (a["protocol_variants"].__setitem__("value", "bad"), a)[1],
            "protocol_variants must be a list",
        ),
        (
            lambda a: (a["status"].__setitem__("value", "blocked"), a)[1],
            "ready artifact must have complete status",
        ),
        (
            lambda a: (
                [row.__setitem__("ready", False) for row in a["protocol_variants"]["value"]],
                a,
            )[1],
            "ready artifact must include a ready protocol variant",
        ),
        (
            lambda a: (a.__setitem__("structured_output_protocol_ready", False), a)[1],
            "blocked artifact must have blocked status",
        ),
    ]

    for mutate, expected in malformed_cases:
        joined = "; ".join(mod.artifact_schema_errors(mutate(clone())))
        assert expected in joined

    with pytest.raises(AssertionError, match="unsafe_false_accepts"):
        bad = clone()
        bad["unsafe_false_accepts"] = 1
        mod.validate_artifact(bad)
