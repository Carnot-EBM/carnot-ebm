"""Tests for Exp 5379 live structured clean gate rerun.

Spec refs: REQ-VERIFY-5379, SCENARIO-VERIFY-5379.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5366_live_grammar_budgeted_sota_protocol_v489 as exp5366
from carnot import experiment_5378_structured_methodology_duration_receipt_v490 as exp5378
from carnot import experiment_5379_live_structured_clean_gate_rerun_v490 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_5379_live_structured_clean_gate_rerun_v490.py -q"
)


def _minimal_gguf(path: Path) -> Path:
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 17, 5))
    return path


def _ready_exp5378() -> dict[str, Any]:
    return json.loads((REPO / exp5378.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))


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
                "metadata": {
                    "magic": "GGUF",
                    "version": 3,
                    "tensor_count": 17,
                    "metadata_kv_count": 5,
                },
                "blocked_preconditions": [],
            }
        )
    return rows


def _live_exp5366_artifact(
    gguf: Path,
    *,
    parse_success_rate: float = 1.0,
    schema_success_rate: float = 1.0,
    final_json_extraction_rate: float = 1.0,
    semantic_success_rate: float = 1.0,
    unsafe_false_accepts: int = 0,
    methodology_duration_s: float = 19.445366,
    duration_s: float = 70.723275,
    prompt_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    specs = _resolved_model_specs(gguf)
    selected = next(row for row in specs if row["hf_id"] == "unsloth/gemma-4-31B-it-GGUF")
    rows = prompt_results or [
        {
            "failure_class": "accepted",
            "parse_success": True,
            "schema_success": True,
            "final_json_extraction_success": True,
            "semantic_success": True,
            "truncation_failure": False,
        }
        for _ in range(4)
    ]
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
        "prompt_count": len(rows),
        "parse_success_rate": parse_success_rate,
        "schema_success_rate": schema_success_rate,
        "final_json_extraction_rate": final_json_extraction_rate,
        "semantic_success_rate": semantic_success_rate,
        "truncation_failure_rate": 0.0,
        "unsafe_false_accepts": unsafe_false_accepts,
        "completion_slack_min_tokens": 982,
        "methodology_duration_s": methodology_duration_s,
        "duration_s": duration_s,
        "prompt_results": rows,
        "generation_receipts": [{"completed": True, "wall_clock_s": methodology_duration_s}],
        "honest_verdict": "complete: structured fixture rerun",
    }


def test_req_verify_5379_spec_declares_canonical_clean_gate_contract() -> None:
    """REQ-VERIFY-5379: OpenSpec anchors the canonical .490 clean gate."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5379") : spec.index("### REQ-VERIFY-5378")]

    for marker in (
        "REQ-VERIFY-5379",
        "SCENARIO-VERIFY-5379",
        str(mod.RESULT_RELATIVE_PATH),
        "live_sota_receipt_ready=true",
        "methodology_duration_s>=60",
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
        "wrong_valid_count",
        "scripts/research_conductor.py",
    ):
        assert marker in section

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert mod.FIELD_PRINCIPLES[field] in mod.field_provenance()[field]["principle"]


def test_scenario_verify_5379_gates_on_exp5378_then_writes_clean_truth_source(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5379: ready Exp5378 permits the canonical live clean rerun."""

    gguf = _minimal_gguf(tmp_path / "gemma-4-31B-it-Q4_K_M.gguf")
    live_calls: list[dict[str, Any]] = []
    upstream = _ready_exp5378()

    def live_runner(**kwargs: Any) -> dict[str, Any]:
        live_calls.append(kwargs)
        return _live_exp5366_artifact(gguf)

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        exp5378_artifact=upstream,
        model_resolver=lambda _hf_id, _quant: str(gguf),
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
        live_runner=live_runner,
        tests_run=[TEST_COMMAND],
        write=True,
    )

    assert live_calls and live_calls[0]["write"] is False
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"
    assert artifact["upstream_receipt_ready"] is True
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
    assert artifact["wrong_valid_count"] == 0
    assert artifact["truncation_failure_rate"] == pytest.approx(0.0)
    assert artifact["unsafe_false_accepts"] == 0
    assert artifact["methodology_duration_s"] == pytest.approx(upstream["methodology_duration_s"])
    assert artifact["methodology_duration_sources"]["exp5378_receipt_s"] == pytest.approx(
        upstream["methodology_duration_s"]
    )
    assert artifact["acceptance_thresholds"] == mod.ACCEPTANCE_THRESHOLDS
    assert artifact["honest_verdict"].startswith("complete:")
    mod.validate_artifact(artifact)


def test_scenario_verify_5379_blocks_before_live_runner_when_exp5378_not_ready(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5379: Exp5378 readiness and duration gate live prompts."""

    upstream = _ready_exp5378()
    upstream["live_sota_receipt_ready"] = False

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "blocked-not-ready.json",
        exp5378_artifact=upstream,
        live_runner=lambda **_kwargs: pytest.fail("Exp5378 block must skip live runner"),
        tests_run=[TEST_COMMAND],
        write=True,
    )

    assert json.loads((tmp_path / "blocked-not-ready.json").read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "blocked"
    assert artifact["upstream_receipt_ready"] is False
    assert artifact["structured_protocol_clean"] is False
    assert artifact["selected_model_spec"] is None
    assert artifact["prompt_count"] == 0
    assert (
        "exp5378_live_sota_receipt_not_ready"
        in artifact["gpu_or_offload_receipt"]["blocked_preconditions"]
    )
    assert artifact["honest_verdict"].startswith("blocked_")

    short_upstream = _ready_exp5378()
    short_upstream["methodology_duration_s"] = 59.99
    short_block = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "blocked-short.json",
        exp5378_artifact=short_upstream,
        live_runner=lambda **_kwargs: pytest.fail("short Exp5378 block must skip live runner"),
        tests_run=[TEST_COMMAND],
        write=False,
    )
    assert short_block["upstream_receipt_ready"] is True
    assert short_block["status"] == "blocked"
    assert (
        "exp5378_methodology_duration_lt_60"
        in short_block["gpu_or_offload_receipt"]["blocked_preconditions"]
    )
    mod.validate_artifact(short_block)


def test_scenario_verify_5379_blocks_cpu_only_runtime_before_prompt_generation(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5379: CPU-only GGUF runtime fails closed before prompts."""

    gguf = _minimal_gguf(tmp_path / "gemma-4-31B-it-Q4_K_M.gguf")

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "blocked-cpu.json",
        exp5378_artifact=_ready_exp5378(),
        model_resolver=lambda _hf_id, _quant: str(gguf),
        runtime_probe=lambda **_kwargs: _runtime_receipt(blocked=["llama_cpp_cpu_only"]),
        live_runner=lambda **_kwargs: pytest.fail("CPU-only runtime must skip live runner"),
        tests_run=[TEST_COMMAND],
        write=False,
    )

    assert artifact["status"] == "blocked"
    assert artifact["upstream_receipt_ready"] is True
    assert artifact["structured_protocol_clean"] is False
    assert artifact["selected_model_spec"] is None
    assert artifact["prompt_count"] == 0
    assert "llama_cpp_cpu_only" in artifact["gpu_or_offload_receipt"]["blocked_preconditions"]
    assert artifact["honest_verdict"].startswith("blocked_")
    mod.validate_artifact(artifact)


def test_req_verify_5379_wrong_valid_count_and_thresholds_are_canonical(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-5379: wrong-valid rows are counted but not hidden in parse/schema rates."""

    gguf = _minimal_gguf(tmp_path / "semantics.gguf")
    rows = [
        {
            "failure_class": "semantic",
            "parse_success": True,
            "schema_success": True,
            "final_json_extraction_success": True,
            "semantic_success": False,
            "truncation_failure": False,
        },
        {
            "failure_class": "accepted",
            "parse_success": True,
            "schema_success": True,
            "final_json_extraction_success": True,
            "semantic_success": True,
            "truncation_failure": False,
        },
    ]

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "wrong-valid.json",
        exp5378_artifact=_ready_exp5378(),
        model_resolver=lambda _hf_id, _quant: str(gguf),
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
        live_runner=lambda **_kwargs: _live_exp5366_artifact(
            gguf,
            semantic_success_rate=0.5,
            prompt_results=rows,
        ),
        tests_run=[TEST_COMMAND],
        write=False,
    )

    assert artifact["wrong_valid_count"] == 1
    assert artifact["semantic_success_rate"] == pytest.approx(0.5)
    assert artifact["structured_protocol_clean"] is True

    blocked = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "parse-block.json",
        exp5378_artifact=_ready_exp5378(),
        model_resolver=lambda _hf_id, _quant: str(gguf),
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
        live_runner=lambda **_kwargs: _live_exp5366_artifact(gguf, parse_success_rate=0.94),
        tests_run=[TEST_COMMAND],
        write=False,
    )
    assert blocked["status"] == "complete"
    assert blocked["structured_protocol_clean"] is False
    assert blocked["honest_verdict"].startswith("blocked_structured_protocol_clean_false")
    mod.validate_artifact(blocked)


def test_req_verify_5379_artifact_validation_rejects_contract_drift(tmp_path: Path) -> None:
    """REQ-VERIFY-5379: artifact validation rejects malformed canonical gate fields."""

    gguf = _minimal_gguf(tmp_path / "validation.gguf")
    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "unused.json",
        exp5378_artifact=_ready_exp5378(),
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
        (
            lambda a: (a.__setitem__("status", "running"), a)[1],
            "status must be complete or blocked",
        ),
        (
            lambda a: (a.__setitem__("upstream_receipt_ready", "yes"), a)[1],
            "upstream_receipt_ready must be boolean",
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
            lambda a: (a.__setitem__("wrong_valid_count", -1), a)[1],
            "wrong_valid_count must be non-negative integer",
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
            lambda a: (a.__setitem__("honest_verdict", "done"), a)[1],
            "honest_verdict must start with complete: or blocked_",
        ),
        (
            lambda a: (
                a.__setitem__("structured_protocol_clean", True),
                a.__setitem__("schema_success_rate", 0.89),
                a,
            )[2],
            "structured_protocol_clean thresholds are not satisfied",
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
                a.__setitem__("status", "complete"),
                a["inference_substrate"].__setitem__("live_local_sota_inference_ran", True),
                a.__setitem__("selected_model_spec", None),
                a,
            )[3],
            "complete status requires selected_model_spec",
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


def test_req_verify_5379_fallback_helpers_preserve_blocked_receipt_shape(tmp_path: Path) -> None:
    """REQ-VERIFY-5379: helper fallbacks keep blocked clean-gate receipts auditable."""

    gguf = _minimal_gguf(tmp_path / "fallback.gguf")
    specs = _resolved_model_specs(gguf)
    selected = specs[0]
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not-json", encoding="utf-8")
    valid_json = tmp_path / "valid.json"
    valid_json.write_text('{"ok": true}', encoding="utf-8")

    assert mod._merge_runtime_receipt({"blocked_preconditions": ["fixture"]}, {})[  # noqa: SLF001
        "blocked_preconditions"
    ] == ["fixture"]
    assert mod._selected_model_from_live({}, selected) == selected  # noqa: SLF001
    assert mod._model_specs_from_live({"MODEL_SPECS": "bad"}, specs) == specs  # noqa: SLF001
    assert mod._wrong_valid_count({"wrong_valid_count": 2}) == 2  # noqa: SLF001
    assert mod._wrong_valid_count({"prompt_results": "bad"}) == 0  # noqa: SLF001
    assert mod._model_specs_cover_mandated("bad") is False  # noqa: SLF001
    assert mod._load_json(valid_json) == {"ok": True}  # noqa: SLF001
    assert mod._load_json(malformed) == {}  # noqa: SLF001
    assert mod._load_json(tmp_path / "missing.json") == {}  # noqa: SLF001

    model_block = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "model-block.json",
        exp5378_artifact=_ready_exp5378(),
        model_resolver=lambda _hf_id, _quant: None,
        runtime_probe=lambda **_kwargs: _runtime_receipt(),
        live_runner=lambda **_kwargs: pytest.fail("missing model block must skip live runner"),
        tests_run=[TEST_COMMAND],
        write=False,
    )
    assert model_block["status"] == "blocked"
    assert (
        "no_mandated_sota_gguf_resolved"
        in model_block["gpu_or_offload_receipt"]["blocked_preconditions"]
    )


def test_req_verify_5379_checked_in_deliverable_satisfies_schema() -> None:
    """REQ-VERIFY-5379: committed deliverable is the canonical .490 truth source."""

    path = REPO / mod.RESULT_RELATIVE_PATH
    assert path.is_file()
    artifact = json.loads(path.read_text(encoding="utf-8"))
    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["upstream_receipt_ready"] is True
    assert artifact["structured_protocol_clean"] is True
    assert artifact["no_autotokenizer_used"] is True
    assert artifact["methodology_duration_s"] >= exp5366.MIN_CLEAN_METHODOLOGY_DURATION_S
