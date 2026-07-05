"""Tests for Exp 5271 SOTA GGUF telemetry receipt harness.

Spec refs: REQ-VERIFY-5271, SCENARIO-VERIFY-5271.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5271_sota_telemetry_receipt_harness_v482 as mod


SPEC_PATH = Path("openspec/capabilities/verification/spec.md")


def _fake_gpu_receipts() -> dict[str, Any]:
    return {
        "value": {
            "gpu_visible": True,
            "nvidia_smi": {"ok": True, "stdout": "0, NVIDIA RTX 3090, 610.43.02"},
            "torch_cuda": {"available": True, "device_count": 2},
            "llama_cpp": {
                "import_ok": True,
                "version": "0.3.29",
                "origin": "/venv/llama_cpp/__init__.py",
            },
            "offload_settings": {"n_gpu_layers": -1, "n_ctx": 512, "max_tokens": 2},
        },
        "principle": mod.FIELD_PRINCIPLES["gpu_offload_receipts"],
    }


def _cached_pair_provider(*, gpu_indices: tuple[int, int]) -> list[dict[str, Any]]:
    assert gpu_indices == (0, 1)
    return [
        {"hf_id": mod.MANDATED_MODEL_SPECS[0]["hf_id"], "model_path": "/cache/qwen.gguf"},
        {"hf_id": mod.MANDATED_MODEL_SPECS[1]["hf_id"], "model_path": "/cache/gemma31.gguf"},
    ]


def _write_minimal_gguf(path: Path) -> Path:
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 17, 5))
    return path


def test_req_verify_5271_spec_declares_receipt_only_contract() -> None:
    """REQ-VERIFY-5271: OpenSpec anchors the telemetry receipt gate."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5271") :]

    for marker in (
        "REQ-VERIFY-5271",
        "SCENARIO-VERIFY-5271",
        str(mod.RESULT_RELATIVE_PATH),
        "live_llm_internal_telemetry_local_gguf_sota",
        "telemetry_harness_ready",
        "capability_absent",
        "no_quality_claim.value=true",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_verify_5271_blocks_without_local_mandated_model(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5271: no local SOTA GGUF blocks without fake telemetry."""

    probe_calls: list[str] = []

    def telemetry_probe(**kwargs: Any) -> dict[str, Any]:
        probe_calls.append(str(kwargs["model_spec"]["model_path"]))
        raise AssertionError("telemetry probe must not run without a local GGUF")

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        model_resolver=lambda _hf_id, _quant: None,
        cached_pair_provider=lambda *, gpu_indices: [],
        gpu_receipts_provider=_fake_gpu_receipts,
        telemetry_probe=telemetry_probe,
        tests_run=[{"command": "unit blocked", "outcome": "passed"}],
        write=True,
    )

    assert probe_calls == []
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"]["value"].startswith("blocked_")
    assert artifact["telemetry_harness_ready"] is False
    assert artifact["no_quality_claim"]["value"] is True
    assert (
        artifact["preconditions_checked"]["value"][
            "at_least_one_mandated_model_resolved_without_autotokenizer"
        ]
        is False
    )
    assert artifact["MODEL_SPECS"]["value"]["flagship_moe"]["status"] == "missing_local_gguf"
    mod.validate_artifact(artifact)


def test_scenario_verify_5271_records_available_and_absent_telemetry(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5271: live receipts distinguish available and absent fields."""

    qwen = _write_minimal_gguf(tmp_path / "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
    gemma = _write_minimal_gguf(tmp_path / "gemma-4-31B-it-Q4_K_M.gguf")
    calls: list[str] = []

    def resolver(hf_id: str, _quant: str) -> str | None:
        paths = {
            mod.MANDATED_MODEL_SPECS[0]["hf_id"]: qwen,
            mod.MANDATED_MODEL_SPECS[1]["hf_id"]: gemma,
        }
        path = paths.get(hf_id)
        return str(path) if path else None

    def telemetry_probe(**kwargs: Any) -> dict[str, Any]:
        role = kwargs["model_spec"]["role"]
        calls.append(role)
        return {
            "runtime_ready": True,
            "status": "telemetry_ready",
            "wall_clock_s": 1.25 if role == "flagship_moe" else 1.75,
            "command": ["llama_cpp.Llama", kwargs["model_spec"]["model_path"]],
            "config": dict(kwargs["offload_config"]),
            "prompt_checksum": mod.sha16(kwargs["prompt"]),
            "output_checksum": mod.sha16(f"{role}:OK"),
            "output_text_preview": "OK",
            "logits": {
                "availability": "available",
                "steps": 3,
                "vocab_size": 128,
                "top_k_count": 8,
            },
            "token_logprobs": {
                "availability": "available",
                "token_count": 2,
                "top_logprobs_count": 2,
            },
            "hidden_states": {
                "availability": "capability_absent",
                "reason": "llama_cpp_api_no_hidden_state_export",
            },
            "attention_summaries": {
                "availability": "capability_absent",
                "reason": "llama_cpp_api_no_attention_export",
            },
        }

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "ready.json",
        model_resolver=resolver,
        cached_pair_provider=_cached_pair_provider,
        gpu_receipts_provider=_fake_gpu_receipts,
        telemetry_probe=telemetry_probe,
        tests_run=[{"command": "unit ready", "outcome": "passed"}],
        write=True,
    )

    mod.validate_artifact(artifact)
    assert calls == ["flagship_moe", "flagship_dense"]
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["telemetry_harness_ready"] is True
    assert "flagship_moe" in artifact["telemetry_harness_ready_principle"]
    assert (
        artifact["MODEL_SPECS"]["value"]["flagship_moe"]["file_receipts"]["size_bytes"]
        == qwen.stat().st_size
    )
    assert artifact["MODEL_SPECS"]["value"]["middle_moe"]["status"] == "missing_local_gguf"
    fields = artifact["exposed_telemetry_fields"]["value"]["flagship_moe"]
    assert fields["logits"]["availability"] == "available"
    assert fields["token_logprobs"]["availability"] == "available"
    assert fields["hidden_states"]["availability"] == "capability_absent"
    assert fields["attention_summaries"]["availability"] == "capability_absent"
    assert artifact["duration_receipts"]["value"]["per_model"]["flagship_dense"][
        "wall_clock_s"
    ] == pytest.approx(1.75)
    assert artifact["prompt_output_checksums"]["value"]["flagship_moe"]["prompt_checksum"]
    assert artifact["prompt_output_checksums"]["value"]["flagship_moe"]["output_checksum"]


def test_req_verify_5271_fails_closed_on_subsecond_live_duration(tmp_path: Path) -> None:
    """REQ-VERIFY-5271: live-model telemetry cannot be claimed with sub-second receipts."""

    gguf = _write_minimal_gguf(tmp_path / "gemma-4-31B-it-Q4_K_M.gguf")

    def telemetry_probe(**kwargs: Any) -> dict[str, Any]:
        return {
            "runtime_ready": True,
            "status": "telemetry_ready",
            "wall_clock_s": 0.25,
            "prompt_checksum": mod.sha16(kwargs["prompt"]),
            "output_checksum": mod.sha16("too-fast"),
            "logits": {"availability": "available", "steps": 1},
            "token_logprobs": {"availability": "capability_absent"},
            "hidden_states": {"availability": "capability_absent"},
            "attention_summaries": {"availability": "capability_absent"},
        }

    with pytest.raises(ValueError, match="sub-second live telemetry duration"):
        mod.run(
            root=tmp_path,
            artifact_path=tmp_path / "too-fast.json",
            model_resolver=lambda hf_id, _quant: (
                str(gguf) if hf_id == mod.MANDATED_MODEL_SPECS[1]["hf_id"] else None
            ),
            cached_pair_provider=_cached_pair_provider,
            gpu_receipts_provider=_fake_gpu_receipts,
            telemetry_probe=telemetry_probe,
            tests_run=[],
            write=False,
        )


def test_req_verify_5271_schema_errors_and_helpers_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-5271: schema, GGUF header, and checksum helpers reject bad receipts."""

    gguf = _write_minimal_gguf(tmp_path / "tiny.gguf")
    receipts = mod._file_receipts(gguf)

    assert receipts["checksum_sha256"]
    assert receipts["checksum_head_1m_sha256"]
    assert mod.read_gguf_header(gguf)["magic"] == "GGUF"

    short = tmp_path / "short.gguf"
    short.write_bytes(b"GGUF")
    pointer = tmp_path / "pointer.gguf"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\n", encoding="utf-8")
    unsupported = tmp_path / "unsupported.gguf"
    unsupported.write_bytes(b"GGUF" + struct.pack("<IQQ", 99, 0, 0))
    two_chunk = tmp_path / "two_chunk.gguf"
    two_chunk.write_bytes(b"a" * (1024 * 1024 + 1))
    large = tmp_path / "large.gguf"
    with large.open("wb") as handle:
        handle.seek(65 * 1024 * 1024)
        handle.write(b"x")

    with pytest.raises(ValueError, match="truncated GGUF header"):
        mod.read_gguf_header(short)
    with pytest.raises(ValueError, match="not a GGUF file"):
        mod.read_gguf_header(pointer)
    with pytest.raises(ValueError, match="unsupported GGUF version"):
        mod.read_gguf_header(unsupported)

    assert mod._file_receipts(two_chunk)["checksum_note"] == "full_sha256_recorded"
    assert (
        mod._file_receipts(large)["checksum_note"]
        == "full_sha256_skipped_for_large_file_head_1m_recorded"
    )
    assert mod._field_receipt(True)["availability"] == "available"
    assert mod._field_receipt(None)["availability"] == "capability_absent"
    assert mod._receipt_has_usable_telemetry({"runtime_ready": False}) is False
    assert "logits=available" in mod._availability_summary(
        {"logits": {"availability": "available"}}
    )
    blocked_spec = mod._model_spec_receipt(
        mod.MANDATED_MODEL_SPECS[0],
        lambda _hf_id, _quant: str(pointer),
    )
    assert blocked_spec["status"] == "blocked_metadata_unreadable"

    artifact = mod.build_artifact(
        root=tmp_path,
        gpu_receipts=_fake_gpu_receipts(),
        model_specs={
            spec["role"]: mod._missing_model_spec(spec) for spec in mod.MANDATED_MODEL_SPECS
        },
        telemetry_receipts={},
        cached_pair_provider=lambda *, gpu_indices: [],
        tests_run=[],
        duration_s=2.0,
    )
    mod.validate_artifact(artifact)

    for mutation, message in (
        (
            lambda art: {key: value for key, value in art.items() if key != "honest_verdict"},
            "missing required field",
        ),
        (
            lambda art: (
                art
                | {
                    "honest_verdict": {
                        "value": "pending",
                        "principle": mod.FIELD_PRINCIPLES["honest_verdict"],
                    }
                }
            ),
            "honest_verdict",
        ),
        (
            lambda art: (
                art
                | {
                    "inference_substrate": {
                        "value": "live_llm_inference",
                        "principle": mod.FIELD_PRINCIPLES["inference_substrate"],
                    }
                }
            ),
            "inference_substrate",
        ),
        (lambda art: art | {"telemetry_harness_ready": "false"}, "bare bool"),
        (
            lambda art: art | {"telemetry_harness_ready_principle": ""},
            "telemetry_harness_ready_principle",
        ),
        (
            lambda art: (
                art
                | {
                    "no_quality_claim": {
                        "value": False,
                        "principle": mod.FIELD_PRINCIPLES["no_quality_claim"],
                    }
                }
            ),
            "no_quality_claim",
        ),
        (lambda art: art | {"tests_run": "unit"}, "tests_run"),
        (
            lambda art: (
                art
                | {"MODEL_SPECS": {"value": [], "principle": mod.FIELD_PRINCIPLES["MODEL_SPECS"]}}
            ),
            "MODEL_SPECS.value must be an object",
        ),
        (
            lambda art: (
                art
                | {"MODEL_SPECS": {"value": {}, "principle": mod.FIELD_PRINCIPLES["MODEL_SPECS"]}}
            ),
            "MODEL_SPECS.value missing role",
        ),
        (
            lambda art: (
                art
                | {
                    "MODEL_SPECS": {
                        "value": art["MODEL_SPECS"]["value"]
                        | {
                            "flagship_moe": art["MODEL_SPECS"]["value"]["flagship_moe"]
                            | {"hf_id": "wrong", "autotokenizer_used": True}
                        },
                        "principle": mod.FIELD_PRINCIPLES["MODEL_SPECS"],
                    }
                }
            ),
            "hf_id mismatch",
        ),
        (
            lambda art: (
                art
                | {
                    "duration_receipts": {
                        "value": {"per_model": []},
                        "principle": mod.FIELD_PRINCIPLES["duration_receipts"],
                    }
                }
            ),
            "duration_receipts.value.per_model",
        ),
        (
            lambda art: (
                art
                | {
                    "duration_receipts": {
                        "value": [],
                        "principle": mod.FIELD_PRINCIPLES["duration_receipts"],
                    }
                }
            ),
            "duration_receipts.value must be an object",
        ),
        (
            lambda art: (
                art
                | {
                    "duration_receipts": {
                        "value": {
                            "per_model": {
                                "flagship_moe": {"runtime_ready": True, "wall_clock_s": 0.1}
                            }
                        },
                        "principle": mod.FIELD_PRINCIPLES["duration_receipts"],
                    }
                }
            ),
            "below live duration floor",
        ),
        (
            lambda art: (
                art
                | {
                    "exposed_telemetry_fields": {
                        "value": [],
                        "principle": mod.FIELD_PRINCIPLES["exposed_telemetry_fields"],
                    }
                }
            ),
            "exposed_telemetry_fields.value must be an object",
        ),
        (
            lambda art: (
                art
                | {
                    "exposed_telemetry_fields": {
                        "value": {"flagship_moe": []},
                        "principle": mod.FIELD_PRINCIPLES["exposed_telemetry_fields"],
                    }
                }
            ),
            "exposed_telemetry_fields.value.flagship_moe must be an object",
        ),
        (
            lambda art: (
                art
                | {
                    "exposed_telemetry_fields": {
                        "value": {"flagship_moe": {"logits": {}}},
                        "principle": mod.FIELD_PRINCIPLES["exposed_telemetry_fields"],
                    }
                }
            ),
            "availability missing",
        ),
    ):
        with pytest.raises(AssertionError, match=message):
            mod.validate_artifact(mutation(artifact))

    recovered = mod.build_artifact(
        root=tmp_path,
        gpu_receipts={"value": {}, "principle": mod.FIELD_PRINCIPLES["gpu_offload_receipts"]},
        model_specs={
            spec["role"]: mod._missing_model_spec(spec) for spec in mod.MANDATED_MODEL_SPECS
        },
        telemetry_receipts={},
        cached_pair_provider=lambda *, gpu_indices: (_ for _ in ()).throw(
            RuntimeError("cache boom")
        ),
        tests_run=[],
        duration_s=2.0,
    )
    assert (
        recovered["preconditions_checked"]["value"]["cached_sota_pair_preview"][0]["status"]
        == "cached_sota_pair_error"
    )


def test_req_verify_5271_module_does_not_use_autotokenizer_or_research_conductor() -> None:
    """REQ-VERIFY-5271: GGUF repos use local paths and the conductor is untouched."""

    source = Path(mod.__file__).read_text(encoding="utf-8")

    assert "AutoTokenizer.from_pretrained" not in source
    assert "transformers" not in source
    assert "scripts/research_conductor.py" not in source
