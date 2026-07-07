"""Tests for Exp 5366 live grammar-budgeted SOTA structured protocol.

Spec refs: REQ-VERIFY-5366, SCENARIO-VERIFY-5366.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5351_trigger_constrain_structured_protocol_v488 as exp5351
from carnot import experiment_5365_grammar_budget_protocol_preflight_v489 as exp5365
from carnot import experiment_5366_live_grammar_budgeted_sota_protocol_v489 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5366_live_grammar_budgeted_sota_protocol_v489.py -q"
)


def _minimal_gguf(path: Path) -> Path:
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 17, 5))
    return path


def _ready_exp5365() -> dict[str, Any]:
    return exp5365.run(
        root=REPO,
        artifact_path=Path("/tmp/unused-exp5365.json"),
        exp5351_path=REPO / exp5351.RESULT_RELATIVE_PATH,
        tests_run=["fixture exp5365"],
        started_s=1.0,
        now_s=2.0,
        write=False,
    )


def _runtime_receipt(*, blocked: list[str] | None = None) -> dict[str, Any]:
    return {
        "gpu_visible": not blocked,
        "gguf_runtime_available": not blocked,
        "gguf_loader_family": "llama.cpp/llama-cpp-python",
        "llama_cpp_gpu_offload_supported": not blocked,
        "offload_evidence": not blocked,
        "non_retired_gpu_or_offload_path": not blocked,
        "blocked_preconditions": list(blocked or []),
        "nvidia_smi": {"ok": not blocked, "stdout": "0, NVIDIA RTX 3090, 24576, 24000"},
    }


def test_req_verify_5366_spec_declares_live_grammar_budget_contract() -> None:
    """REQ-VERIFY-5366: OpenSpec anchors the live grammar-budgeted SOTA gate."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5366") : spec.index("### REQ-VERIFY-5365")]

    for marker in (
        "REQ-VERIFY-5366",
        "SCENARIO-VERIFY-5366",
        str(mod.RESULT_RELATIVE_PATH),
        "grammar_budget_protocol_ready=true",
        "non-retired GPU/offload evidence",
        "llama.cpp/GGUF",
        "AutoTokenizer.from_pretrained",
        "AutoModel",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "parse_success_rate>=0.95",
        "schema_success_rate>=0.90",
        "final_json_extraction_rate>=0.95",
        "unsafe_false_accepts=0",
        "methodology_duration_s>=60",
        "scripts/research_conductor.py",
    ):
        assert marker in section

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert mod.FIELD_PRINCIPLES[field] in mod.field_provenance()[field]["principle"]


def test_req_verify_5366_scores_schema_semantic_and_truncation_separately() -> None:
    """REQ-VERIFY-5366: scoring separates parse, schema, semantic, and truncation failures."""

    variant = mod.protocol_settings_from_exp5365(_ready_exp5365())
    prompt = exp5351.DEFAULT_CALIBRATION_PROMPTS[0]
    good = (
        f'{variant["sentinel"]} '
        '{"answer":"47 minutes","facts":["checked"],"id":"battery_probe"} '
        f'{variant["end_sentinel"]}'
    )
    wrong_type = (
        f'{variant["sentinel"]} '
        '{"answer":47,"facts":["checked"],"id":"battery_probe"} '
        f'{variant["end_sentinel"]}'
    )
    wrong_answer = (
        f'{variant["sentinel"]} '
        '{"answer":"48 minutes","facts":["checked"],"id":"battery_probe"} '
        f'{variant["end_sentinel"]}'
    )
    truncated = f'{variant["sentinel"]} {{"answer":"47 minutes","facts":["checked"]'
    no_final = '{"answer":"47 minutes","facts":["checked"],"id":"battery_probe"}'

    good_score = mod.score_live_output(
        prompt_spec=prompt,
        variant=variant,
        output_text=good,
        completed=True,
    )
    schema_score = mod.score_live_output(
        prompt_spec=prompt,
        variant=variant,
        output_text=wrong_type,
        completed=True,
    )
    semantic_score = mod.score_live_output(
        prompt_spec=prompt,
        variant=variant,
        output_text=wrong_answer,
        completed=True,
    )
    truncation_score = mod.score_live_output(
        prompt_spec=prompt,
        variant=variant,
        output_text=truncated,
        completed=False,
    )
    timeout_score = mod.score_live_output(
        prompt_spec=prompt,
        variant=variant,
        output_text=no_final,
        completed=False,
        timed_out=True,
    )
    token_budget_score = mod.score_live_output(
        prompt_spec=prompt,
        variant=variant,
        output_text=no_final,
        completed=True,
        generated_token_count=variant["max_tokens"],
    )
    parse_score = mod.score_live_output(
        prompt_spec=prompt,
        variant=variant,
        output_text=no_final,
        completed=True,
    )

    assert good_score["failure_class"] == "accepted"
    assert good_score["semantic_success"] is True
    assert schema_score["failure_class"] == "schema"
    assert schema_score["parse_success"] is True
    assert semantic_score["failure_class"] == "semantic"
    assert semantic_score["schema_success"] is True
    assert semantic_score["semantic_success"] is False
    assert truncation_score["failure_class"] == "truncation"
    assert truncation_score["truncation_failure"] is True
    assert timeout_score["failure_class"] == "truncation"
    assert token_budget_score["failure_class"] == "truncation"
    assert parse_score["failure_class"] == "parse"
    assert mod._honest_verdict(  # noqa: SLF001
        status="complete",
        grammar_ready=True,
        structured_clean=False,
        blockers=[],
    ).startswith("blocked_structured_protocol_clean_false")


def test_scenario_verify_5366_blocks_without_non_retired_gpu_offload_path(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5366: CPU-only headline path writes blocked artifact with no prompts."""

    gguf = _minimal_gguf(tmp_path / "gemma-4-31B-it-Q4_K_M.gguf")
    calls: list[str] = []

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        exp5365_artifact=_ready_exp5365(),
        model_resolver=lambda hf_id, _quant: (
            str(gguf) if hf_id == "unsloth/gemma-4-31B-it-GGUF" else None
        ),
        runtime_probe=lambda **_kwargs: _runtime_receipt(blocked=["llama_cpp_cpu_only"]),
        generation_probe=lambda **kwargs: calls.append(kwargs["prompt_spec"]["prompt_id"]) or {},
        tests_run=[TEST_COMMAND],
        write=True,
    )

    assert calls == []
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "blocked"
    assert artifact["grammar_budget_protocol_ready"] is True
    assert artifact["structured_protocol_clean"] is False
    assert artifact["selected_model_spec"] is None
    assert artifact["prompt_count"] == 0
    assert artifact["gpu_or_offload_receipt"]["non_retired_gpu_or_offload_path"] is False
    assert "llama_cpp_cpu_only" in artifact["gpu_or_offload_receipt"]["blocked_preconditions"]
    assert artifact["honest_verdict"].startswith("blocked_")
    mod.validate_artifact(artifact)


def test_scenario_verify_5366_mocked_live_run_writes_clean_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5366: live mocked SOTA prompt rows can open the structured gate."""

    gguf = _minimal_gguf(tmp_path / "gemma-4-31B-it-Q4_K_M.gguf")
    calls: list[tuple[str, str, int]] = []

    def generation_probe(**kwargs: Any) -> dict[str, Any]:
        prompt_spec = kwargs["prompt_spec"]
        variant = kwargs["variant"]
        assert kwargs["model_spec"]["hf_id"] == "unsloth/gemma-4-31B-it-GGUF"
        assert kwargs["max_tokens"] == 1024
        calls.append((prompt_spec["prompt_id"], kwargs["model_spec"]["hf_id"], kwargs["max_tokens"]))
        payload = prompt_spec["target_final_object"]
        return {
            "completed": True,
            "timed_out": False,
            "returncode": 0,
            "stdout": (
                "[Start thinking]\n"
                "free reasoning remains outside the final JSON\n"
                f"{variant['sentinel']} "
                f"{json.dumps(payload, sort_keys=True, separators=(',', ':'))} "
                f"{variant['end_sentinel']}"
            ),
            "stderr": "",
            "wall_clock_s": 15.25,
            "generated_token_count": 42,
            "gpu_memory_receipts": {
                "before": {"used_mb": 4},
                "after_load": {"used_mb": 8192},
                "after_generate": {"used_mb": 8192},
                "max_memory_delta_mb": 8188,
                "offload_evidence": True,
            },
        }

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "experiment_5366.json",
        exp5365_artifact=_ready_exp5365(),
        model_resolver=lambda hf_id, _quant: (
            str(gguf) if hf_id == "unsloth/gemma-4-31B-it-GGUF" else None
        ),
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
        generation_probe=generation_probe,
        tests_run=[TEST_COMMAND],
        write=True,
    )

    mod.validate_artifact(artifact)
    assert len(calls) == len(exp5351.DEFAULT_CALIBRATION_PROMPTS)
    assert artifact["status"] == "complete"
    assert artifact["structured_protocol_clean"] is True
    assert artifact["selected_model_spec"]["hf_id"] == "unsloth/gemma-4-31B-it-GGUF"
    assert [row["hf_id"] for row in artifact["MODEL_SPECS"]] == list(mod.MANDATED_HF_IDS)
    assert artifact["inference_substrate"]["loader_family"] == "llama.cpp/llama-cpp-python"
    assert artifact["inference_substrate"]["live_local_sota_inference_ran"] is True
    assert artifact["no_autotokenizer_used"] is True
    assert artifact["prompt_count"] == 4
    assert artifact["parse_success_rate"] == pytest.approx(1.0)
    assert artifact["schema_success_rate"] == pytest.approx(1.0)
    assert artifact["final_json_extraction_rate"] == pytest.approx(1.0)
    assert artifact["semantic_success_rate"] == pytest.approx(1.0)
    assert artifact["truncation_failure_rate"] == pytest.approx(0.0)
    assert artifact["unsafe_false_accepts"] == 0
    assert artifact["completion_slack_min_tokens"] == _ready_exp5365()[
        "completion_slack_min_tokens"
    ]
    assert artifact["methodology_duration_s"] == pytest.approx(61.0)
    assert artifact["protocol_settings"]["max_tokens"] == 1024
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_verify_5366_blocks_when_exp5365_gate_not_ready(tmp_path: Path) -> None:
    """REQ-VERIFY-5366: Exp5365 readiness is a hard gate before generation."""

    not_ready = _ready_exp5365()
    not_ready["grammar_budget_protocol_ready"] = False
    calls: list[str] = []

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "blocked_gate.json",
        exp5365_artifact=not_ready,
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
        generation_probe=lambda **kwargs: calls.append(kwargs["prompt_spec"]["prompt_id"]) or {},
        tests_run=[TEST_COMMAND],
        write=False,
    )

    assert calls == []
    assert artifact["status"] == "blocked"
    assert artifact["grammar_budget_protocol_ready"] is False
    assert artifact["MODEL_SPECS"] == mod.default_model_specs_unresolved()
    assert artifact["selected_model_spec"] is None
    assert artifact["gpu_or_offload_receipt"]["blocked_preconditions"] == [
        "exp5365_grammar_budget_protocol_not_ready"
    ]
    assert artifact["honest_verdict"].startswith("blocked_exp5365")
    mod.validate_artifact(artifact)

    missing_gate = mod.run(
        root=tmp_path,
        artifact_path=Path("relative-blocked.json"),
        exp5365_path=tmp_path / "missing-exp5365.json",
        tests_run=[TEST_COMMAND],
        write=True,
    )
    assert (tmp_path / "relative-blocked.json").is_file()
    assert missing_gate["grammar_budget_protocol_ready"] is False

    gguf = _minimal_gguf(tmp_path / "runtime-fails.gguf")

    def raising_generation_probe(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("fixture generation failure")

    generation_failed = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "generation-failed.json",
        exp5365_artifact=_ready_exp5365(),
        model_resolver=lambda hf_id, _quant: (
            str(gguf) if hf_id == "unsloth/gemma-4-31B-it-GGUF" else None
        ),
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
        generation_probe=raising_generation_probe,
        tests_run=[TEST_COMMAND],
        write=False,
    )
    assert generation_failed["status"] == "blocked"
    assert "live_generation_failed" in generation_failed["gpu_or_offload_receipt"][
        "blocked_preconditions"
    ]
    assert generation_failed["prompt_count"] == 4


def test_req_verify_5366_artifact_validation_rejects_contract_drift(tmp_path: Path) -> None:
    """REQ-VERIFY-5366: artifact validation rejects malformed live-gate fields."""

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "unused.json",
        exp5365_artifact=_ready_exp5365(),
        model_resolver=lambda _hf_id, _quant: None,
        runtime_probe=lambda **_kwargs: _runtime_receipt(blocked=["no_mandated_sota_gguf"]),
        tests_run=[TEST_COMMAND],
        write=False,
    )
    missing_path = tmp_path / "missing.gguf"
    invalid_path = tmp_path / "invalid.gguf"
    invalid_path.write_bytes(b"NOPE" + struct.pack("<IQQ", 3, 1, 1))
    truncated_path = tmp_path / "truncated.gguf"
    truncated_path.write_bytes(b"GGUF")
    bad_version_path = tmp_path / "bad-version.gguf"
    bad_version_path.write_bytes(b"GGUF" + struct.pack("<IQQ", 9, 1, 1))

    resolved = mod.resolve_model_specs(
        lambda hf_id, _quant: (
            str(missing_path)
            if hf_id == "unsloth/Qwen3.6-35B-A3B-GGUF"
            else str(invalid_path)
            if hf_id == "unsloth/gemma-4-31B-it-GGUF"
            else None
        )
    )
    assert resolved[0]["blocked_preconditions"] == ["model_file_missing"]
    assert resolved[1]["status"] == "blocked_metadata_unreadable"
    with pytest.raises(ValueError, match="truncated GGUF"):
        mod.read_gguf_header(truncated_path)
    with pytest.raises(ValueError, match="not a GGUF"):
        mod.read_gguf_header(invalid_path)
    with pytest.raises(ValueError, match="unsupported GGUF version"):
        mod.read_gguf_header(bad_version_path)

    def clone() -> dict[str, Any]:
        return json.loads(json.dumps(artifact))

    malformed_cases = [
        (lambda a: (a.pop("status"), a)[1], "missing required fields"),
        (lambda a: (a.__setitem__("status", "running"), a)[1], "status must be complete or blocked"),
        (
            lambda a: (a.__setitem__("grammar_budget_protocol_ready", "yes"), a)[1],
            "grammar_budget_protocol_ready must be boolean",
        ),
        (
            lambda a: (a.__setitem__("structured_protocol_clean", "false"), a)[1],
            "structured_protocol_clean must be boolean",
        ),
        (
            lambda a: (a.__setitem__("MODEL_SPECS", []), a)[1],
            "MODEL_SPECS must contain all mandated SOTA GGUF specs",
        ),
        (
            lambda a: (a.__setitem__("MODEL_SPECS", "bad"), a)[1],
            "MODEL_SPECS must contain all mandated SOTA GGUF specs",
        ),
        (
            lambda a: (a.__setitem__("selected_model_spec", {"hf_id": "legacy/smoke"}), a)[1],
            "selected_model_spec must be null or one mandated model spec",
        ),
        (
            lambda a: (a.__setitem__("inference_substrate", "live"), a)[1],
            "inference_substrate must be object",
        ),
        (
            lambda a: (a.__setitem__("gpu_or_offload_receipt", []), a)[1],
            "gpu_or_offload_receipt must be object",
        ),
        (
            lambda a: (a.__setitem__("no_autotokenizer_used", False), a)[1],
            "no_autotokenizer_used must be true",
        ),
        (
            lambda a: (a.__setitem__("prompt_count", "0"), a)[1],
            "prompt_count must be non-negative integer",
        ),
        (
            lambda a: (a.__setitem__("parse_success_rate", 2.0), a)[1],
            "parse_success_rate must be in [0, 1]",
        ),
        (
            lambda a: (a.__setitem__("unsafe_false_accepts", -1), a)[1],
            "unsafe_false_accepts must be non-negative integer",
        ),
        (
            lambda a: (a.__setitem__("completion_slack_min_tokens", "982"), a)[1],
            "completion_slack_min_tokens must be integer",
        ),
        (
            lambda a: (a.__setitem__("methodology_duration_s", "0"), a)[1],
            "methodology_duration_s must be numeric",
        ),
        (
            lambda a: (a.__setitem__("honest_verdict", "done"), a)[1],
            "honest_verdict must start with complete: or blocked_",
        ),
        (
            lambda a: (
                a.__setitem__("status", "complete"),
                a.__setitem__("grammar_budget_protocol_ready", False),
                a["inference_substrate"].__setitem__("live_local_sota_inference_ran", True),
                a.__setitem__("selected_model_spec", {"hf_id": "unsloth/gemma-4-31B-it-GGUF"}),
                a,
            )[4],
            "complete status requires Exp5365 grammar budget readiness",
        ),
        (
            lambda a: (
                a.__setitem__("status", "complete"),
                a["inference_substrate"].__setitem__("live_local_sota_inference_ran", False),
                a,
            )[2],
            "complete status requires live local SOTA inference",
        ),
        (
            lambda a: (
                a.__setitem__("structured_protocol_clean", True),
                a.__setitem__("parse_success_rate", 0.94),
                a,
            )[2],
            "structured_protocol_clean thresholds are not satisfied",
        ),
        (
            lambda a: (a["field_provenance"].pop("status"), a)[1],
            "field_provenance must cover required fields",
        ),
    ]

    for mutate, expected in malformed_cases:
        joined = "; ".join(mod.artifact_schema_errors(mutate(clone())))
        assert expected in joined

    with pytest.raises(AssertionError, match="no_autotokenizer_used"):
        bad = clone()
        bad["no_autotokenizer_used"] = False
        mod.validate_artifact(bad)
