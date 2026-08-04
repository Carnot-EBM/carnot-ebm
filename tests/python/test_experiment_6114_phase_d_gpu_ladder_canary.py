"""Tests for Exp6114 Phase D GPU ladder generation canary.

Spec refs: REQ-VERIFY-6114, SCENARIO-VERIFY-6114-LADDER-REPLAY,
SCENARIO-VERIFY-6114-MEASURED-FIT-MODEL, SCENARIO-VERIFY-6114-REAL-GENERATION,
SCENARIO-VERIFY-6114-LIFECYCLE, SCENARIO-VERIFY-6114-RETIRED-SCOPE.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6103_phase_d_difficulty_ladder_fixture as ladder_mod
from carnot import experiment_6114_phase_d_gpu_ladder_canary as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/verifiable-reasoning/spec.md"
LADDER_ARTIFACT = REPO / ladder_mod.RESULT_RELATIVE_PATH
LADDER_ROWS = REPO / ladder_mod.ROW_FILE_RELATIVE_PATH
LADDER_SPLITS = REPO / ladder_mod.SPLIT_MANIFEST_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6114_phase_d_gpu_ladder_canary.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6114_phase_d_gpu_ladder_canary.py "
    "-m pytest tests/python/test_experiment_6114_phase_d_gpu_ladder_canary.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6114_phase_d_gpu_ladder_canary.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6114_phase_d_gpu_ladder_canary.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6114_phase_d_gpu_ladder_canary.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_COMMAND = (
    "git status --short -- scripts/research_conductor.py "
    "ops/exclusion_manifest.yaml ops/changelog.md ops/status.md _bmad/traceability.md"
)
TEST_COMMANDS = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
    PROTECTED_COMMAND,
]
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


class FakeGenerationBackend:
    """Deterministic stand-in for the live task-owned llama.cpp worker."""

    def __init__(self, *, engaged: bool = True, release: bool = True) -> None:
        self.engaged = engaged
        self.release = release
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        *,
        model_spec: dict[str, Any],
        selected_gpu: int,
        prompts: list[dict[str, Any]],
        decode_config: dict[str, Any],
        baseline_devices: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """SCENARIO-VERIFY-6114-REAL-GENERATION: return natural text rows."""

        self.calls.append(
            {
                "hf_id": model_spec["hf_id"],
                "selected_gpu": selected_gpu,
                "prompt_count": len(prompts),
                "max_new_tokens": decode_config["max_new_tokens"],
                "baseline_devices": baseline_devices,
            }
        )
        rows: list[dict[str, Any]] = []
        for index, prompt in enumerate(prompts):
            text = (
                "I compare the public facts against each answer choice, then keep "
                "the option that satisfies the stated finite rule. Final answer: "
                f"{chr(65 + (index % 4))}"
            )
            rows.append(
                {
                    "row_id": prompt["row_id"],
                    "raw_generation": text,
                    "normalized_generation": text,
                    "generated_token_count": 24 + index,
                    "decode_time_s": round(0.25 + index / 100, 6),
                    "finish_reason": "length" if index % 2 else "stop",
                    "seed": prompt["seed"],
                }
            )
        memory_delta = 18_432 if self.engaged else 0
        return {
            "server_pid": 611400,
            "server_exit_code": 0,
            "pid_exited": True,
            "cuda_sync_method": "llama_cpp_backend_close_plus_vram_probe",
            "worker_exit_observed": True,
            "vram_release_observed": self.release,
            "timeline": [
                {
                    "phase": "pre_load",
                    "task_pid": None,
                    "devices": baseline_devices,
                    "timestamp_monotonic_s": 1.0,
                },
                {
                    "phase": "decode",
                    "task_pid": 611400,
                    "devices": [
                        {
                            "index": selected_gpu,
                            "memory_total_mb": 24576,
                            "memory_free_mb": 5600,
                            "memory_used_mb": memory_delta,
                            "temperature_c": 61,
                        }
                    ],
                    "timestamp_monotonic_s": 2.0,
                },
                {
                    "phase": "post_release",
                    "task_pid": None,
                    "devices": baseline_devices if self.release else [],
                    "timestamp_monotonic_s": 3.0,
                },
            ],
            "gpu_engagement": {
                "attributable": self.engaged,
                "task_pid": 611400,
                "selected_gpu": selected_gpu,
                "selected_gpu_memory_delta_mb": memory_delta,
            },
            "rows": rows,
        }


def _preconditions(tmp_path: Path, *, free_mb: int = 24_120) -> dict[str, Any]:
    before_hashes = {
        str(path): mod.sha256_file(REPO / path)
        for path in mod.PROTECTED_FILES
        if (REPO / path).exists()
    }
    return {
        "schema": "fixture.preconditions",
        "run_date": mod.RUN_DATE,
        "preconditions_ready": True,
        "blocked_reasons": [],
        "gpu": {
            "gpu_count": 2,
            "ok": True,
            "devices": [
                {
                    "index": 0,
                    "name": "RTX 3090",
                    "memory_total_mb": 24576,
                    "memory_free_mb": 12_000,
                    "memory_used_mb": 12_576,
                    "temperature_c": 52,
                },
                {
                    "index": 1,
                    "name": "RTX 3090",
                    "memory_total_mb": 24576,
                    "memory_free_mb": free_mb,
                    "memory_used_mb": 24576 - free_mb,
                    "temperature_c": 53,
                },
            ],
        },
        "resources": {
            "memory": {"available_mb": 96_000, "required_mb": 16_384, "ok": True},
            "disk": {"available_mb": 512_000, "required_mb": 10_240, "ok": True},
            "swap": {"total_mb": 128_000, "free_mb": 96_000, "used_mb": 32_000},
        },
        "runtime": {
            "cuda_build": {"python": "3.12.fixture", "nvcc": {"ok": True}},
            "task_owned_pid_leases": {
                "current_pid": 1000,
                "child_pids": [],
                "lease_scope": "task_owned_processes_only",
            },
        },
        "output_paths": {"result_path": str(tmp_path / mod.RESULT_RELATIVE_PATH.name), "ok": True},
        "root_clutter": {"root_python_files": [], "root_python_file_count": 0, "ok": True},
        "protected_file_hashes_before": before_hashes,
    }


def _exp6102_artifact(tmp_path: Path) -> Path:
    model_path = tmp_path / "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
    model_path.write_bytes(b"GGUF fixture bytes for exp6114")
    model_sha = mod.sha256_file(model_path)
    model_record = {
        "hf_id": mod.MODEL_HF_ID,
        "family": "gemma-4-26b-a4b-it",
        "model_path": str(model_path),
        "model_sha256": model_sha,
        "local_model_present": True,
        "primary_model_file": True,
        "local_path_hash": mod.sha256_text(str(model_path.resolve())),
        "quantization": "Q4_K_M",
        "min_vram_gb": 16,
        "headline_eligible": True,
    }
    artifact = {
        "status": "blocked",
        "honest_verdict": "blocked: insufficient_free_vram",
        "runtime_cuda_vram_thermal_and_pid_lease_receipts": {
            "capacity_verdicts": {
                "unsloth/Qwen3.6-35B-A3B-GGUF": {
                    "fits": False,
                    "required_mb": 24576,
                    "reason": "insufficient_free_vram",
                },
                "unsloth/gemma-4-31B-it-GGUF": {
                    "fits": False,
                    "required_mb": 24576,
                    "reason": "insufficient_free_vram",
                },
                mod.MODEL_HF_ID: {
                    "fits": True,
                    "required_mb": mod.MEASURED_FIT_REQUIRED_MB,
                    "reason": "fits",
                    "selected_gpu": 1,
                },
            }
        },
        "model_specs_and_exact_file_hashes": {
            "records": {mod.MODEL_HF_ID: model_record},
            "all_mandated_files_present": True,
        },
        "quantization_and_embedded_tokenizer_receipts": {
            "records": {
                mod.MODEL_HF_ID: {
                    "auto_tokenizer_used": False,
                    "gguf_embedded_tokenizer_only": True,
                    "quantization": "Q4_K_M",
                    "embedded_tokenizer_receipt": {
                        "source": "embedded_gguf_llama_cpp_vocab_only",
                        "loadable": True,
                        "detail": "embedded tokenizer fixture",
                    },
                }
            },
            "all_embedded_tokenizers_loadable": True,
            "auto_tokenizer_used": False,
        },
    }
    path = tmp_path / "experiment_6102_sota_atom_corpus_vram_recovery.json"
    path.write_text(json.dumps(artifact, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_req_verify_6114_spec_declares_canary_contract() -> None:
    """REQ-VERIFY-6114: OpenSpec names required fields and principles."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-VERIFY-6114") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-6114",
        "SCENARIO-VERIFY-6114-LADDER-REPLAY",
        "SCENARIO-VERIFY-6114-MEASURED-FIT-MODEL",
        "SCENARIO-VERIFY-6114-REAL-GENERATION",
        "SCENARIO-VERIFY-6114-LIFECYCLE",
        "SCENARIO-VERIFY-6114-RETIRED-SCOPE",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        mod.MODEL_HF_ID,
        "17,186 MiB",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_verify_6114_ladder_replay_refuses_tamper() -> None:
    """SCENARIO-VERIFY-6114-LADDER-REPLAY: exact Exp6103 fixture only."""

    artifact = json.loads(LADDER_ARTIFACT.read_text(encoding="utf-8"))
    rows = ladder_mod.read_row_file(LADDER_ROWS)
    splits = json.loads(LADDER_SPLITS.read_text(encoding="utf-8"))
    receipt = mod.verify_sealed_ladder(
        ladder_artifact=artifact,
        ladder_rows=rows,
        split_manifest=splits,
        row_file_path=LADDER_ROWS,
        split_manifest_path=LADDER_SPLITS,
    )

    assert receipt["sealed_ladder_ready"] is True
    assert receipt["calibration_row_count"] == 600
    assert receipt["selected_split_policy"] == "calibration_only"
    sampled = mod.sample_calibration_rows(rows, per_family=4)
    assert len(sampled) == 12
    assert {row["split"] for row in sampled} == {"calibration"}
    assert {row["family"] for row in sampled} == set(ladder_mod.FAMILIES)

    tampered = deepcopy(artifact)
    tampered["phase_d_ladder_fixture_ready_score"] = 0.0
    with pytest.raises(mod.CanaryGateError, match="phase_d_ladder_fixture_ready_score"):
        mod.verify_sealed_ladder(
            ladder_artifact=tampered,
            ladder_rows=rows,
            split_manifest=splits,
            row_file_path=LADDER_ROWS,
            split_manifest_path=LADDER_SPLITS,
        )


def test_scenario_verify_6114_fake_run_is_complete_ready(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6114-REAL-GENERATION: natural rows and schema receipts."""

    backend = FakeGenerationBackend()
    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        ladder_artifact_path=LADDER_ARTIFACT,
        ladder_rows_path=LADDER_ROWS,
        ladder_split_manifest_path=LADDER_SPLITS,
        exp6102_artifact_path=_exp6102_artifact(tmp_path),
        preconditions_checked=_preconditions(tmp_path),
        generation_backend=backend,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=6.114,
        write=True,
    )

    assert backend.calls == [
        {
            "hf_id": mod.MODEL_HF_ID,
            "selected_gpu": 1,
            "prompt_count": 12,
            "max_new_tokens": 512,
            "baseline_devices": _preconditions(tmp_path)["gpu"]["devices"],
        }
    ]
    assert artifact["status"] == "complete_ready"
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert artifact["phase_d_compute_and_ladder_ready_score"] == pytest.approx(1.0)
    assert artifact["retirement_triggered"] is False
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["duration_s"] == pytest.approx(6.114)
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert mod.validate_artifact(artifact) is True
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text(encoding="utf-8")) == artifact

    rows = artifact["generated_calibration_canary_rows_and_hashes"]["rows"]
    assert len(rows) == 12
    assert {row["family"] for row in rows} == set(ladder_mod.FAMILIES)
    assert {row["source_split"] for row in rows} == {"calibration"}
    assert all("Final answer:" in row["raw_generation"] for row in rows)
    assert all(row["row_hash"] == mod.canary_row_hash(row) for row in rows)
    assert all(row["generated_token_count"] > 0 for row in rows)
    assert all(row["decode_time_s"] > 0 for row in rows)

    decode = artifact["prompt_decode_seed_and_token_receipts"]
    assert decode["json_grammar_used"] is False
    assert decode["finite_id_transport_used"] is False
    assert decode["deterministic_answer_builder_used"] is False
    assert decode["cpu_headline_fallback_used"] is False
    assert decode["sleep_substitute_used"] is False
    assert decode["decode_config"]["max_new_tokens"] >= 512

    model_receipt = artifact["model_specs_and_exact_file_hashes"]
    assert model_receipt["selected_model_hf_id"] == mod.MODEL_HF_ID
    assert model_receipt["measured_fit_required_mb"] == mod.MEASURED_FIT_REQUIRED_MB
    assert model_receipt["records"][mod.MODEL_HF_ID]["quantization"] == "Q4_K_M"
    assert artifact["quantization_and_embedded_tokenizer_receipt"]["auto_tokenizer_used"] is False
    assert artifact["gpu_engagement_attribution"]["attributable"] is True
    assert artifact["server_exit_cuda_sync_pid_exit_and_vram_release_receipts"][
        "vram_release_toward_baseline"
    ] is True
    assert artifact["retired_representation_scope_untouched"][
        "exp6102_representation_row_shards_read"
    ] is False
    assert artifact["retired_representation_scope_untouched"][
        "exp5964_representation_row_shards_read"
    ] is False
    assert artifact["protected_files_unchanged"]["all_unchanged"] is True
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert artifact["field_provenance"][field]["principle"] == principle


def test_scenario_verify_6114_lifecycle_failure_retires_shape(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6114-LIFECYCLE: engagement failure cannot be ready."""

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        ladder_artifact_path=LADDER_ARTIFACT,
        ladder_rows_path=LADDER_ROWS,
        ladder_split_manifest_path=LADDER_SPLITS,
        exp6102_artifact_path=_exp6102_artifact(tmp_path),
        preconditions_checked=_preconditions(tmp_path),
        generation_backend=FakeGenerationBackend(engaged=False),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=1.0,
        write=False,
    )

    assert artifact["status"] == "retired"
    assert artifact["honest_verdict"].startswith("retired:")
    assert artifact["retirement_triggered"] is True
    assert artifact["phase_d_compute_and_ladder_ready_score"] == pytest.approx(0.0)
    assert artifact["gpu_engagement_attribution"]["attributable"] is False
    assert mod.validate_artifact(artifact) is True


def test_scenario_verify_6114_capacity_failure_retires_without_backend(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6114-MEASURED-FIT-MODEL: no tiny-model fallback."""

    backend = FakeGenerationBackend()
    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        ladder_artifact_path=LADDER_ARTIFACT,
        ladder_rows_path=LADDER_ROWS,
        ladder_split_manifest_path=LADDER_SPLITS,
        exp6102_artifact_path=_exp6102_artifact(tmp_path),
        preconditions_checked=_preconditions(tmp_path, free_mb=12_000),
        generation_backend=backend,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=1.0,
        write=False,
    )

    assert backend.calls == []
    assert artifact["status"] == "retired"
    assert artifact["honest_verdict"].startswith("retired:")
    assert artifact["retirement_triggered"] is True
    assert artifact["task_owned_gpu_server_and_pid_lease"]["selected_gpu"] is None
    assert "insufficient_free_vram" in artifact["preconditions_checked"]["blocked_reasons"]
    assert artifact["model_specs_and_exact_file_hashes"]["tiny_model_substituted"] is False
    assert mod.validate_artifact(artifact) is True
