"""Tests for Exp 5351 trigger-then-constrain structured protocol calibration.

Spec refs: REQ-VERIFY-5351, SCENARIO-VERIFY-5351.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5351_trigger_constrain_structured_protocol_v488 as mod


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
                "context": 512,
                "batch": 512,
                "ubatch": 128,
                "gpu_layers": "all",
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
                "free_vram_mb": 48240,
                "nvidia_smi": {"ok": True, "stdout": "0, RTX 3090, 24576, 24120, 0"},
                "blocked_preconditions": [],
            },
            "principle": "preconditions",
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _preconditions_probe(
    *,
    selected_command: dict[str, Any] | None,
    model_specs: dict[str, Any],
    selected_model: dict[str, Any] | None,
    prior_runtime_path: Path,
) -> dict[str, Any]:
    return {
        "exp5337_runtime_artifact_path": str(prior_runtime_path),
        "exp5337_runtime_receipt_clean": True,
        "gpu_visible": True,
        "raw_nvidia_smi": {"ok": True, "stdout": "fixture nvidia-smi"},
        "nvidia_smi": {
            "ok": True,
            "stdout": "0, NVIDIA GeForce RTX 3090, 610.43.02, 24576, 24120, 0",
        },
        "free_vram_mb": 24120,
        "llama_cpp_command": list((selected_command or {}).get("command") or [])[:1],
        "llama_cpp_version": {"ok": True, "stderr": "version: 9606 (9b4dae81f)"},
        "model_file_presence": {
            role: bool(spec.get("model_path")) for role, spec in model_specs.items()
        },
        "selected_model_file_present": bool(selected_model and selected_model.get("model_path")),
        "backend_cuda_evidence": True,
        "blocked_preconditions": [],
    }


def _base_kwargs(tmp_path: Path) -> dict[str, Any]:
    binary = tmp_path / "llama-cli"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    model_path = tmp_path / "gemma-4-31B-it-Q4_K_M.gguf"
    model_path.write_text("GGUF", encoding="utf-8")
    prior_path = _write_prior_runtime(tmp_path / mod.exp5337.RESULT_RELATIVE_PATH, binary, model_path)
    return {
        "root": tmp_path,
        "artifact_path": tmp_path / mod.RESULT_RELATIVE_PATH,
        "prior_runtime_path": prior_path,
        "preconditions_probe": _preconditions_probe,
        "tests_run": [{"command": "unit exp5351", "outcome": "passed"}],
    }


def _trigger_constrain_generation(**kwargs: Any) -> dict[str, Any]:
    prompt_spec = kwargs["prompt_spec"]
    final = {
        "id": prompt_spec["expected"]["id"],
        "answer": prompt_spec["expected"]["answer"],
        "facts": [{"source": prompt_spec["prompt_id"], "checked": True}],
    }
    return {
        "completed": True,
        "timed_out": False,
        "returncode": 0,
        "stdout": (
            "Loading model...\n\navailable commands:\n  /exit\n\n> prompt echo\n"
            '{"id":"draft","answer":"wrong","facts":[]}\n'
            "[Start thinking]\nfree reasoning stays outside the final object\n"
            f"{kwargs['variant']['sentinel']} "
            f"{json.dumps(final, separators=(',', ':'))} "
            f"{kwargs['variant']['end_sentinel']}\n"
            "[ Prompt: 200.0 t/s | Generation: 40.0 t/s ]\nExiting..."
        ),
        "stderr": "",
        "wall_clock_s": 15.25,
    }


def test_req_verify_5351_spec_declares_trigger_constrain_contract() -> None:
    """REQ-VERIFY-5351: OpenSpec anchors the v488 trigger-constrain gate."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5351") : spec.index("### REQ-VERIFY-5339")]
    normalized_section = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5351",
        "SCENARIO-VERIFY-5351",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "trigger-then-constrain",
        "parse_success_rate",
        "schema_success_rate",
        "final_json_extraction_rate",
        "unsafe_false_accepts",
        "methodology_duration_s",
        "structured_protocol_clean",
        "no_quality_claim=true",
        "no_autotokenizer_used=true",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "scripts/research_conductor.py",
    ):
        assert marker in section

    for field in mod.REQUIRED_WRAPPED_FIELDS:
        assert f"`{field}`" in section
        assert " ".join(mod.FIELD_PRINCIPLES[field].split()) in normalized_section


def test_scenario_verify_5351_trigger_constrain_run_writes_clean_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5351: trigger-then-constrain opens the clean protocol gate."""

    calls: list[tuple[str, str, list[str]]] = []

    def probe(**kwargs: Any) -> dict[str, Any]:
        calls.append(
            (
                kwargs["variant"]["variant_id"],
                kwargs["prompt_spec"]["prompt_id"],
                kwargs["command"],
            )
        )
        return _trigger_constrain_generation(**kwargs)

    artifact = mod.run(
        **_base_kwargs(tmp_path),
        protocol_variants=mod.DEFAULT_PROTOCOL_VARIANTS[:1],
        generation_probe=probe,
        write=True,
    )

    mod.validate_artifact(artifact)
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert len(calls) == len(mod.DEFAULT_CALIBRATION_PROMPTS)
    assert all(mod.FREE_REASONING_TRIGGER in command[command.index("-p") + 1] for _, _, command in calls)
    assert artifact["status"]["value"] == "complete"
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["inference_substrate"]["value"] == "live_llm_inference"
    assert artifact["selected_model_spec"]["value"]["hf_id"] == "unsloth/gemma-4-31B-it-GGUF"
    assert {row["hf_id"] for row in artifact["MODEL_SPECS"]["value"].values()} == set(
        mod.EXPECTED_MODEL_IDS
    )
    assert artifact["prompt_count"] == 4
    assert artifact["parse_success_rate"] == pytest.approx(1.0)
    assert artifact["schema_success_rate"] == pytest.approx(1.0)
    assert artifact["final_json_extraction_rate"] == pytest.approx(1.0)
    assert artifact["unsafe_false_accepts"] == 0
    assert artifact["methodology_duration_s"] >= mod.MIN_CLEAN_METHODOLOGY_DURATION_S
    assert artifact["structured_protocol_clean"] is True
    assert artifact["no_quality_claim"] is True
    assert artifact["no_autotokenizer_used"] is True
    variant = artifact["protocol_variants"]["value"][0]
    assert variant["ready"] is True
    assert variant["free_reasoning_trigger_token"] == mod.FREE_REASONING_TRIGGER
    assert variant["final_json_sentinel"] == "FINAL_JSON:"
    assert variant["final_only_extraction"] is True
    assert variant["strict_schema_validation"] is True
    assert variant["stop_sequences_requested"] == ["END_FINAL_JSON"]
    assert variant["token_counts"]["generated_token_count"] > 0
    assert len(variant["command_lines"]) == 4


def test_req_verify_5351_blocks_preconditions_and_invalid_prompt_scope(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-5351: blocked preconditions or wrong prompt count prevent generation."""

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
    assert artifact["structured_protocol_clean"] is False
    assert artifact["no_quality_claim"] is True
    assert "exp5337_runtime_receipt_not_clean" in artifact["preconditions_checked"]["value"][
        "blocked_preconditions"
    ]

    invalid_prompt_count = mod.run(
        **_base_kwargs(tmp_path),
        calibration_prompts=mod.DEFAULT_CALIBRATION_PROMPTS[:3],
        generation_probe=lambda **kw: calls.append(kw["prompt_spec"]["prompt_id"]) or {},
        write=False,
    )
    assert "prompt_count_outside_4_to_6" in invalid_prompt_count["preconditions_checked"]["value"][
        "blocked_preconditions"
    ]

    def gpu_blocked_probe(**kwargs: Any) -> dict[str, Any]:
        payload = _preconditions_probe(**kwargs)
        payload["gpu_visible"] = False
        return payload

    gpu_kwargs = _base_kwargs(tmp_path)
    gpu_kwargs["preconditions_probe"] = gpu_blocked_probe
    gpu_blocked = mod.run(
        **gpu_kwargs,
        generation_probe=lambda **kw: calls.append(kw["prompt_spec"]["prompt_id"]) or {},
        write=False,
    )
    assert "current_gpu_not_visible" in gpu_blocked["preconditions_checked"]["value"][
        "blocked_preconditions"
    ]


def test_req_verify_5351_parser_schema_and_false_accept_guards() -> None:
    """REQ-VERIFY-5351: parser rejects draft, malformed, schema-drift, and leak cases."""

    variant = mod.DEFAULT_PROTOCOL_VARIANTS[0]
    prompt = mod.DEFAULT_CALIBRATION_PROMPTS[0]
    assert mod._rate([], "parse_success") == 0.0
    assert mod.unsafe_false_accept_count(variant, ()) == 0
    assert mod._select_best_variant([]) is None
    assert mod._estimate_generated_tokens("") == 0
    assert mod.strip_llama_cpp_banners("Loading model...\n/clear history\n> echoed\nkept").strip() == "kept"
    assert mod.schema_errors(7, {"type": "integer"}) == []
    assert mod.schema_errors(3.5, {"type": "number"}) == []
    assert mod.schema_errors(True, {"type": "boolean"}) == []
    assert mod.schema_errors([], {"type": "array"}) == []
    assert mod.schema_errors("x", {"type": "mystery"}) == ["payload must be mystery"]

    prior_with_autotokenizer = {
        "status": {"value": "complete"},
        "inference_substrate": {"value": "live_llm_inference"},
        "sota_runtime_clean_receipt_ready": True,
        "selected_backend_command": {
            "value": {"command": ["/missing/llama-cli"], "model_role": "flagship_dense"}
        },
        "MODEL_SPECS": {
            "value": {
                "flagship_moe": {
                    "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                    "autotokenizer_used": False,
                },
                "flagship_dense": {
                    "hf_id": "unsloth/gemma-4-31B-it-GGUF",
                    "autotokenizer_used": True,
                },
                "middle_moe": {
                    "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                    "autotokenizer_used": False,
                },
            }
        },
        "preconditions_checked": {"value": {"gpu_visible": True}},
    }
    _selected_command, _model_specs, _selected_model, blockers = mod._selected_runtime_context(
        prior_with_autotokenizer
    )
    assert "autotokenizer_used_for_gguf" in blockers

    no_sentinel = mod.score_protocol_output(
        prompt_spec=prompt,
        variant=variant,
        output_text='{"id":"battery_probe","answer":"47 minutes","facts":[]}',
        completed=True,
    )
    assert no_sentinel["parse_success"] is False

    draft_then_bad_final = (
        '{"id":"battery_probe","answer":"draft","facts":[]}\n'
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

    missing_key = f"{variant['sentinel']} {{\"id\":\"battery_probe\",\"answer\":\"47 minutes\"}}"
    score = mod.score_protocol_output(
        prompt_spec=prompt,
        variant=variant,
        output_text=missing_key,
        completed=True,
    )
    assert score["parse_success"] is True
    assert score["schema_success"] is False
    assert score["accepted_for_protocol"] is False

    wrong_type = f"{variant['sentinel']} {{\"id\":\"battery_probe\",\"answer\":47,\"facts\":[]}}"
    score = mod.score_protocol_output(
        prompt_spec=prompt,
        variant=variant,
        output_text=wrong_type,
        completed=True,
    )
    assert score["parse_success"] is True
    assert score["schema_success"] is False
    assert score["accepted_for_protocol"] is False

    thinking_inside = (
        f"{variant['sentinel']} "
        '{"id":"battery_probe","answer":"[Start thinking] leaked","facts":[]}'
    )
    score = mod.score_protocol_output(
        prompt_spec=prompt,
        variant=variant,
        output_text=thinking_inside,
        completed=True,
    )
    assert score["thinking_text_outside_final"] is False
    assert score["accepted_for_protocol"] is False
    assert mod.unsafe_false_accept_count(variant, mod.DEFAULT_CALIBRATION_PROMPTS) == 0


def test_req_verify_5351_artifact_validation_rejects_contract_drift(tmp_path: Path) -> None:
    """REQ-VERIFY-5351: schema validation catches malformed v488 artifacts."""

    artifact = mod.run(
        **_base_kwargs(tmp_path),
        protocol_variants=mod.DEFAULT_PROTOCOL_VARIANTS[:1],
        generation_probe=_trigger_constrain_generation,
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
        (lambda a: (a.__setitem__("prompt_count", 3), a)[1], "prompt_count must be 4 to 6"),
        (
            lambda a: (a.__setitem__("prompt_count", "4"), a)[1],
            "prompt_count must be a bare integer",
        ),
        (
            lambda a: (a.__setitem__("parse_success_rate", "1.0"), a)[1],
            "parse_success_rate must be in [0, 1]",
        ),
        (
            lambda a: (a.__setitem__("schema_success_rate", 1.5), a)[1],
            "schema_success_rate must be in [0, 1]",
        ),
        (
            lambda a: (a.__setitem__("final_json_extraction_rate", -0.1), a)[1],
            "final_json_extraction_rate must be in [0, 1]",
        ),
        (
            lambda a: (a.__setitem__("unsafe_false_accepts", "0"), a)[1],
            "unsafe_false_accepts must be a bare integer",
        ),
        (
            lambda a: (a.__setitem__("methodology_duration_s", "61"), a)[1],
            "methodology_duration_s must be numeric",
        ),
        (
            lambda a: (a.__setitem__("structured_protocol_clean", "yes"), a)[1],
            "structured_protocol_clean must be a bare boolean",
        ),
        (
            lambda a: (a.__setitem__("no_quality_claim", False), a)[1],
            "no_quality_claim must be bare true",
        ),
        (
            lambda a: (a.__setitem__("no_autotokenizer_used", False), a)[1],
            "no_autotokenizer_used must be bare true",
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
            lambda a: (a["MODEL_SPECS"]["value"]["flagship_dense"].__setitem__("autotokenizer_used", True), a)[1],
            "autotokenizer_used must stay false",
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
            "clean artifact must have complete status",
        ),
        (
            lambda a: (
                [row.__setitem__("ready", False) for row in a["protocol_variants"]["value"]],
                a,
            )[1],
            "clean artifact must include a ready protocol variant",
        ),
        (
            lambda a: (a.__setitem__("structured_protocol_clean", False), a)[1],
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
