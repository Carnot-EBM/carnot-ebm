"""Tests for Exp 5378 structured methodology-duration receipt.

Spec refs: REQ-VERIFY-5378, SCENARIO-VERIFY-5378.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5365_grammar_budget_protocol_preflight_v489 as exp5365
from carnot import experiment_5366_live_grammar_budgeted_sota_protocol_v489 as exp5366
from carnot import experiment_5378_structured_methodology_duration_receipt_v490 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5378_structured_methodology_duration_receipt_v490.py -q"
)


def _minimal_gguf(path: Path) -> Path:
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 17, 5))
    return path


def _ready_exp5365() -> dict[str, Any]:
    return json.loads((REPO / exp5365.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))


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


def _resolved_model_specs(gguf: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base in exp5366.MANDATED_MODEL_SPECS:
        rows.append(
            {
                "role": base["role"],
                "hf_id": base["hf_id"],
                "quantization": base["quantization"],
                "cache_path": str(gguf.parent),
                "model_path": str(gguf),
                "status": "local_gguf_resolved",
                "headline_eligible": True,
                "gguf_loader_family": "llama.cpp",
                "autotokenizer_used": False,
                "file_receipts": {"path": str(gguf), "size_bytes": gguf.stat().st_size},
                "metadata": {"magic": "GGUF", "version": 3, "tensor_count": 17, "metadata_kv_count": 5},
                "blocked_preconditions": [],
            }
        )
    return rows


def _live_exp5366_artifact(gguf: Path, *, methodology_duration_s: float = 19.445366) -> dict[str, Any]:
    specs = _resolved_model_specs(gguf)
    selected = next(row for row in specs if row["hf_id"] == "unsloth/gemma-4-31B-it-GGUF")
    return {
        "schema": exp5366.SCHEMA,
        "experiment_id": exp5366.EXPERIMENT_ID,
        "status": "complete",
        "grammar_budget_protocol_ready": True,
        "structured_protocol_clean": False,
        "MODEL_SPECS": specs,
        "selected_model_spec": selected,
        "inference_substrate": {
            "kind": "live_llm_inference",
            "loader_family": "llama.cpp/llama-cpp-python",
            "gguf_loader_family": "llama.cpp/llama-cpp-python",
            "gpu_or_offload_status": "non_retired_gpu_or_offload_path",
            "live_local_sota_inference_ran": True,
            "selected_model_hf_id": selected["hf_id"],
        },
        "gpu_or_offload_receipt": _runtime_receipt(),
        "no_autotokenizer_used": True,
        "prompt_count": 4,
        "parse_success_rate": 1.0,
        "schema_success_rate": 1.0,
        "final_json_extraction_rate": 1.0,
        "semantic_success_rate": 1.0,
        "truncation_failure_rate": 0.0,
        "unsafe_false_accepts": 0,
        "completion_slack_min_tokens": 982,
        "methodology_duration_s": methodology_duration_s,
        "duration_s": 70.723275,
        "generation_receipts": [
            {
                "completed": True,
                "prompt_id": "battery_duration_probe",
                "wall_clock_s": methodology_duration_s,
                "gpu_memory_receipts": {
                    "max_memory_delta_mb": 21102,
                    "offload_evidence": True,
                },
            }
        ],
        "prompt_results": [{"failure_class": "accepted"}],
        "honest_verdict": "blocked_structured_protocol_clean_false: live SOTA inference ran but clean gate failed",
    }


def test_req_verify_5378_spec_declares_methodology_receipt_contract() -> None:
    """REQ-VERIFY-5378: OpenSpec anchors the structured duration receipt."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5378") : spec.index("### REQ-VERIFY-5366")]

    for marker in (
        "REQ-VERIFY-5378",
        "SCENARIO-VERIFY-5378",
        str(mod.RESULT_RELATIVE_PATH),
        "Exp 5365 grammar-budget preflight",
        "Exp 5366 live structured protocol",
        "without changing the Exp 5366 acceptance thresholds",
        "CUDA/GPU visibility",
        "non-retired GPU/offload evidence",
        "llama.cpp/GGUF",
        "AutoTokenizer.from_pretrained",
        "AutoModel",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "methodology_duration_s>=60",
        "active_roadmap_modified=false",
        "conductor_modified=false",
        "scripts/research_conductor.py",
    ):
        assert marker in section

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert mod.FIELD_PRINCIPLES[field] in mod.field_provenance()[field]["principle"]


def test_scenario_verify_5378_repairs_receipt_duration_without_threshold_drift(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5378: live runtime duration repairs the clean structured receipt."""

    gguf = _minimal_gguf(tmp_path / "gemma-4-31B-it-Q4_K_M.gguf")
    live_calls: list[dict[str, Any]] = []

    def live_runner(**kwargs: Any) -> dict[str, Any]:
        live_calls.append(kwargs)
        return _live_exp5366_artifact(gguf)

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        exp5365_artifact=_ready_exp5365(),
        model_resolver=lambda _hf_id, _quant: str(gguf),
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
        live_runner=live_runner,
        tests_run=[TEST_COMMAND],
        write=True,
    )

    assert live_calls and live_calls[0]["write"] is False
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"
    assert artifact["live_sota_receipt_ready"] is True
    assert artifact["grammar_budget_protocol_ready"] is True
    assert artifact["structured_protocol_clean"] is True
    assert [row["hf_id"] for row in artifact["MODEL_SPECS"]] == list(mod.MANDATED_HF_IDS)
    assert artifact["selected_model_spec"]["hf_id"] == "unsloth/gemma-4-31B-it-GGUF"
    assert artifact["inference_substrate"]["kind"] == "live_llm_inference"
    assert artifact["inference_substrate"]["loader_family"] == "llama.cpp/llama-cpp-python"
    assert artifact["gpu_or_offload_receipt"]["non_retired_gpu_or_offload_path"] is True
    assert artifact["no_autotokenizer_used"] is True
    assert artifact["prompt_count"] == 4
    assert artifact["parse_success_rate"] == pytest.approx(1.0)
    assert artifact["schema_success_rate"] == pytest.approx(1.0)
    assert artifact["final_json_extraction_rate"] == pytest.approx(1.0)
    assert artifact["semantic_success_rate"] == pytest.approx(1.0)
    assert artifact["truncation_failure_rate"] == pytest.approx(0.0)
    assert artifact["completion_slack_min_tokens"] == 982
    assert artifact["unsafe_false_accepts"] == 0
    assert artifact["methodology_duration_s"] == pytest.approx(70.723275)
    assert artifact["active_roadmap_modified"] is False
    assert artifact["conductor_modified"] is False
    assert artifact["acceptance_thresholds"] == {
        "parse_success_rate": exp5366.MIN_PARSE_SUCCESS_RATE,
        "schema_success_rate": exp5366.MIN_SCHEMA_SUCCESS_RATE,
        "final_json_extraction_rate": exp5366.MIN_FINAL_JSON_EXTRACTION_RATE,
        "methodology_duration_s": exp5366.MIN_CLEAN_METHODOLOGY_DURATION_S,
        "unsafe_false_accepts": 0,
    }
    assert artifact["honest_verdict"].startswith("complete:")
    mod.validate_artifact(artifact)


def test_scenario_verify_5378_blocks_before_live_runner_on_cpu_only_runtime(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5378: CPU-only GGUF runtime fails closed before prompts."""

    gguf = _minimal_gguf(tmp_path / "gemma-4-31B-it-Q4_K_M.gguf")

    def live_runner(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("live runner must not be called on CPU-only runtime")

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "blocked.json",
        exp5365_artifact=_ready_exp5365(),
        model_resolver=lambda _hf_id, _quant: str(gguf),
        runtime_probe=lambda **_kwargs: _runtime_receipt(blocked=["llama_cpp_cpu_only"]),
        live_runner=live_runner,
        tests_run=[TEST_COMMAND],
        write=True,
    )

    assert json.loads((tmp_path / "blocked.json").read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "blocked"
    assert artifact["live_sota_receipt_ready"] is False
    assert artifact["grammar_budget_protocol_ready"] is True
    assert artifact["selected_model_spec"] is None
    assert artifact["prompt_count"] == 0
    assert artifact["methodology_duration_s"] == pytest.approx(0.0)
    assert artifact["active_roadmap_modified"] is False
    assert artifact["conductor_modified"] is False
    assert "llama_cpp_cpu_only" in artifact["gpu_or_offload_receipt"]["blocked_preconditions"]
    assert artifact["honest_verdict"].startswith("blocked_")
    mod.validate_artifact(artifact)


def test_req_verify_5378_blocks_when_exp5365_or_model_preconditions_fail(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-5378: missing grammar or model preconditions prevent live receipt claims."""

    not_ready = _ready_exp5365()
    not_ready["grammar_budget_protocol_ready"] = False
    gate_block = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "gate-block.json",
        exp5365_artifact=not_ready,
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
        live_runner=lambda **_kwargs: pytest.fail("Exp5365 block must skip live runner"),
        tests_run=[TEST_COMMAND],
        write=False,
    )
    assert gate_block["status"] == "blocked"
    assert gate_block["grammar_budget_protocol_ready"] is False
    assert gate_block["MODEL_SPECS"] == exp5366.default_model_specs_unresolved()
    assert gate_block["honest_verdict"].startswith("blocked_exp5365")

    model_block = mod.run(
        root=tmp_path,
        artifact_path=Path("model-block.json"),
        exp5365_artifact=_ready_exp5365(),
        model_resolver=lambda _hf_id, _quant: None,
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
        live_runner=lambda **_kwargs: pytest.fail("missing model block must skip live runner"),
        tests_run=[TEST_COMMAND],
        write=True,
    )
    assert (tmp_path / "model-block.json").is_file()
    assert model_block["status"] == "blocked"
    assert model_block["live_sota_receipt_ready"] is False
    assert "no_mandated_sota_gguf_resolved" in model_block["gpu_or_offload_receipt"][
        "blocked_preconditions"
    ]


def test_req_verify_5378_artifact_validation_rejects_contract_drift(tmp_path: Path) -> None:
    """REQ-VERIFY-5378: artifact validation rejects malformed receipt fields."""

    gguf = _minimal_gguf(tmp_path / "gemma-4-31B-it-Q4_K_M.gguf")
    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "unused.json",
        exp5365_artifact=_ready_exp5365(),
        model_resolver=lambda _hf_id, _quant: str(gguf),
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
        live_runner=lambda **_kwargs: _live_exp5366_artifact(gguf),
        tests_run=[TEST_COMMAND],
        write=False,
    )

    def clone() -> dict[str, Any]:
        return json.loads(json.dumps(artifact))

    malformed_cases = [
        (lambda a: (a.pop("status"), a)[1], "missing required fields"),
        (lambda a: (a.__setitem__("status", "running"), a)[1], "status must be complete or blocked"),
        (
            lambda a: (a.__setitem__("live_sota_receipt_ready", "yes"), a)[1],
            "live_sota_receipt_ready must be boolean",
        ),
        (
            lambda a: (a.__setitem__("grammar_budget_protocol_ready", "yes"), a)[1],
            "grammar_budget_protocol_ready must be boolean",
        ),
        (
            lambda a: (a.__setitem__("MODEL_SPECS", []), a)[1],
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
            lambda a: (a.__setitem__("prompt_count", "4"), a)[1],
            "prompt_count must be non-negative integer",
        ),
        (
            lambda a: (a.__setitem__("parse_success_rate", 1.2), a)[1],
            "parse_success_rate must be in [0, 1]",
        ),
        (
            lambda a: (a.__setitem__("completion_slack_min_tokens", "982"), a)[1],
            "completion_slack_min_tokens must be integer",
        ),
        (
            lambda a: (a.__setitem__("unsafe_false_accepts", -1), a)[1],
            "unsafe_false_accepts must be non-negative integer",
        ),
        (
            lambda a: (a.__setitem__("methodology_duration_s", "70"), a)[1],
            "methodology_duration_s must be numeric",
        ),
        (
            lambda a: (a.__setitem__("active_roadmap_modified", True), a)[1],
            "active_roadmap_modified must be false",
        ),
        (
            lambda a: (a.__setitem__("conductor_modified", True), a)[1],
            "conductor_modified must be false",
        ),
        (
            lambda a: (a.__setitem__("honest_verdict", "done"), a)[1],
            "honest_verdict must start with complete: or blocked_",
        ),
        (
            lambda a: (
                a.__setitem__("live_sota_receipt_ready", True),
                a.__setitem__("methodology_duration_s", 59.99),
                a,
            )[2],
            "live_sota_receipt_ready requires methodology_duration_s>=60",
        ),
        (
            lambda a: (
                a.__setitem__("status", "complete"),
                a.__setitem__("selected_model_spec", None),
                a,
            )[2],
            "complete status requires selected_model_spec",
        ),
        (
            lambda a: (
                a["gpu_or_offload_receipt"].__setitem__("non_retired_gpu_or_offload_path", False),
                a,
            )[1],
            "ready receipt requires non-retired GPU/offload evidence",
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


def test_req_verify_5378_fallback_helpers_preserve_blocked_receipt_shape(tmp_path: Path) -> None:
    """REQ-VERIFY-5378: helper fallbacks keep blocked receipts auditable."""

    gguf = _minimal_gguf(tmp_path / "fallback.gguf")
    specs = _resolved_model_specs(gguf)
    selected = specs[0]
    preflight_receipt = {"blocked_preconditions": ["fixture"], "gpu_visible": False}
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not-json", encoding="utf-8")

    assert mod._merge_runtime_receipt(preflight_receipt, {})["blocked_preconditions"] == ["fixture"]  # noqa: SLF001
    assert mod._selected_model_from_live({}, selected) == selected  # noqa: SLF001
    assert mod._model_specs_from_live({"MODEL_SPECS": "bad"}, specs) == specs  # noqa: SLF001
    assert mod._model_specs_cover_mandated("bad") is False  # noqa: SLF001
    assert mod._load_json(tmp_path / "missing.json") == {}  # noqa: SLF001
    assert mod._load_json(malformed) == {}  # noqa: SLF001

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "structured-false.json",
        exp5365_artifact=_ready_exp5365(),
        model_resolver=lambda _hf_id, _quant: str(gguf),
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
        live_runner=lambda **_kwargs: {
            **_live_exp5366_artifact(gguf),
            "parse_success_rate": 0.5,
        },
        tests_run=[TEST_COMMAND],
        write=False,
    )
    assert artifact["live_sota_receipt_ready"] is True
    assert artifact["structured_protocol_clean"] is False
    assert artifact["honest_verdict"].startswith("complete: live SOTA runtime receipt ready")
