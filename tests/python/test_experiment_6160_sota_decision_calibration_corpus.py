"""Tests for Exp6160 fresh SOTA decision calibration corpus.

Spec refs: REQ-VERIFY-6160, REQ-VERIFY-6160-1, REQ-VERIFY-6160-2,
REQ-VERIFY-6160-3, REQ-VERIFY-6160-4, REQ-VERIFY-6160-5,
REQ-VERIFY-6160-6, REQ-VERIFY-6160-7, REQ-VERIFY-6160-8,
REQ-VERIFY-6160-9, REQ-VERIFY-6160-10,
SCENARIO-VERIFY-6160-GATE, SCENARIO-VERIFY-6160-ORDERING,
SCENARIO-VERIFY-6160-NO-MEMORY, SCENARIO-VERIFY-6160-NONOVERLAP.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6160_sota_decision_calibration_corpus as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/verifiable-reasoning/spec.md"


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _preconditions(tmp_path: Path, *, ready: bool = True) -> dict[str, Any]:
    before = {
        relative.as_posix(): mod.sha256_file(REPO / relative)
        for relative in mod.PROTECTED_FILES
        if (REPO / relative).exists()
    }
    return {
        "schema": mod.SCHEMA + ".preconditions",
        "run_date": mod.RUN_DATE,
        "preconditions_ready": ready,
        "blocked_reasons": [] if ready else ["fixture_precondition_block"],
        "checks": {
            "exp6159_ready": ready,
            "two_cuda_gpus_available": ready,
            "no_inherited_model_server": ready,
            "output_paths_writable": True,
            "root_clutter_absent": True,
            "protected_files_present": True,
        },
        "hashed_input_receipts": [],
        "gpu": {
            "gpu_count": 2,
            "ok": ready,
            "devices": [
                {
                    "index": 0,
                    "name": "RTX 3090",
                    "memory_total_mb": 24576,
                    "memory_used_mb": 16,
                    "memory_free_mb": 24560,
                    "temperature_c": 44,
                    "power_draw_w": 21.0,
                },
                {
                    "index": 1,
                    "name": "RTX 3090",
                    "memory_total_mb": 24576,
                    "memory_used_mb": 16,
                    "memory_free_mb": 24560,
                    "temperature_c": 45,
                    "power_draw_w": 22.0,
                },
            ],
        },
        "compute_apps_before": [],
        "lease_state": {
            "task_owned_pid": 616000,
            "parent_pid": 1,
            "lease_scope": "task_owned_child_workers_only",
            "no_inherited_model_server": ready,
        },
        "output_paths": {
            "result_path": str(tmp_path / mod.RESULT_RELATIVE_PATH.name),
            "row_sidecar_dir": str(tmp_path),
            "parent_writable": True,
        },
        "protected_file_hashes_before": before,
        "root_clutter": {"root_python_file_count": 0, "ok": True},
        "principle": mod.FIELD_PRINCIPLES["preconditions_checked"],
    }


def _model_resolution(tmp_path: Path, *, gemma_ready: bool = True) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for index, hf_id in enumerate(mod.MANDATED_MODEL_IDS):
        slug = mod.model_slug(hf_id)
        path = tmp_path / f"{slug}-Q4_K_M.gguf"
        path.write_bytes(b"GGUF" + slug.encode("ascii"))
        records.append(
            {
                "name": "Qwen3.6-35B-A3B" if index == 0 else "Gemma4-26B-A4B-it",
                "hf_id": hf_id,
                "gpu": index,
                "model_path": str(path),
                "real_path": str(path),
                "revision": f"fixture-revision-{index}",
                "quantization": "Q4_K_M",
                "sha256": mod.sha256_file(path),
                "size_bytes": path.stat().st_size,
                "exists": True,
                "is_projector_gguf": False,
                "embedded_tokenizer_loadable": gemma_ready if index == 1 else True,
                "embedded_tokenizer_detail": "embedded tokenizer OK",
                "chat_template_present": gemma_ready if index == 1 else True,
                "chat_template_sha256": mod.sha256_text(f"chat-{index}"),
                "chat_template_keys": ["tokenizer.chat_template"],
                "metadata_summary_sha256": mod.sha256_text(f"meta-{index}"),
                "loader": "llama_cpp.Llama",
                "n_gpu_layers": -1,
                "expected_offload": "full_cuda",
                "actual_use_count": 0,
            }
        )
    return {
        "schema": mod.SCHEMA + ".model_resolution",
        "records": records,
        "blocked_reasons": [] if gemma_ready else ["gemma_embedded_tokenizer_unloadable"],
    }


class FakeDecisionBackend:
    """REQ-VERIFY-6160-8: deterministic task-owned native-chat backend."""

    def __init__(self, *, invalid_first: bool = False) -> None:
        self.invalid_first = invalid_first
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        *,
        model_spec: dict[str, Any],
        prompts: list[dict[str, Any]],
        decode_config: dict[str, Any],
        baseline_devices: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "hf_id": model_spec["hf_id"],
                "prompt_count": len(prompts),
                "temperature": decode_config["temperature"],
                "grammar": decode_config["grammar"],
                "memory": decode_config["memory"],
                "label_conditioned_retry": decode_config["label_conditioned_retry"],
            }
        )
        rows: list[dict[str, Any]] = []
        for index, prompt in enumerate(prompts):
            payload = json.dumps(prompt, sort_keys=True)
            assert "exact_answer" not in payload
            assert "current_outcome" not in payload
            assert "future_label" not in payload
            assert "held_label" not in payload
            assert "post_outcome" not in payload
            raw = (
                ""
                if self.invalid_first and index == 0
                else (
                    "STRATEGY_ID: visible_graph_v1\n"
                    "STRATEGY: inspect the decision-time graph only.\n"
                    f"SOLUTION: proposed terminal answer for {prompt['event_id']}"
                )
            )
            rows.append(
                {
                    "event_id": prompt["event_id"],
                    "raw_response": raw,
                    "generated_token_count": 0 if raw == "" else 41 + index,
                    "decode_time_s": round(0.02 + index / 10_000, 6),
                    "finish_reason": "stop",
                    "seed": prompt["seed"],
                }
            )
        gpu = int(model_spec["gpu"])
        pid = 616000 + gpu
        return {
            "model_hf_id": model_spec["hf_id"],
            "worker_pid": pid,
            "worker_exit_code": 0,
            "pid_exited": True,
            "cuda_sync_method": "fixture_worker_exit",
            "vram_release_observed": True,
            "orphan_task_owned_pid_count": 0,
            "retained_task_owned_vram_mb": 0,
            "unrelated_processes_killed": [],
            "timeline": [
                {
                    "phase": "before_load",
                    "task_pid": pid,
                    "devices": baseline_devices,
                    "compute_apps": [],
                    "timestamp_monotonic_s": 1.0,
                },
                {
                    "phase": "decode",
                    "task_pid": pid,
                    "devices": [
                        {
                            "index": gpu,
                            "name": "RTX 3090",
                            "memory_total_mb": 24576,
                            "memory_used_mb": 18100,
                            "memory_free_mb": 6476,
                            "temperature_c": 63,
                            "power_draw_w": 260.0,
                        }
                    ],
                    "compute_apps": [{"pid": pid, "used_memory_mb": 18100}],
                    "timestamp_monotonic_s": 2.0,
                },
                {
                    "phase": "release",
                    "task_pid": pid,
                    "devices": baseline_devices,
                    "compute_apps": [],
                    "timestamp_monotonic_s": 3.0,
                },
            ],
            "gpu_engagement": {
                "attributable": True,
                "task_pid": pid,
                "selected_gpu": gpu,
                "selected_gpu_memory_delta_mb": 18084,
                "n_gpu_layers": -1,
            },
            "rows": rows,
        }


def _run_artifact(
    tmp_path: Path,
    *,
    backend: FakeDecisionBackend | None = None,
    preconditions_ready: bool = True,
    gemma_ready: bool = True,
    write: bool = False,
) -> dict[str, Any]:
    return mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        row_sidecar_dir=tmp_path,
        preconditions_checked=_preconditions(tmp_path, ready=preconditions_ready),
        model_resolution=_model_resolution(tmp_path, gemma_ready=gemma_ready),
        generation_backend=backend or FakeDecisionBackend(),
        test_exit_codes=_passing_exit_codes(),
        duration_s=6.160,
        write=write,
    )


def test_req_6160_spec_declares_sota_decision_corpus_contract() -> None:
    """REQ-VERIFY-6160: OpenSpec names requirements, fields, and principles."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-VERIFY-6160") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-6160-1",
        "REQ-VERIFY-6160-2",
        "REQ-VERIFY-6160-3",
        "REQ-VERIFY-6160-4",
        "REQ-VERIFY-6160-5",
        "REQ-VERIFY-6160-6",
        "REQ-VERIFY-6160-7",
        "REQ-VERIFY-6160-8",
        "REQ-VERIFY-6160-9",
        "REQ-VERIFY-6160-10",
        "SCENARIO-VERIFY-6160-GATE",
        "SCENARIO-VERIFY-6160-ORDERING",
        "SCENARIO-VERIFY-6160-NO-MEMORY",
        "SCENARIO-VERIFY-6160-NONOVERLAP",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.MANDATED_MODEL_IDS[0],
        mod.MANDATED_MODEL_IDS[1],
        mod.LIVE_INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_6160_complete_ready_conserves_rows_and_orders_outcomes(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6160-ORDERING/NONOVERLAP: authentic rows pass."""

    backend = FakeDecisionBackend()
    artifact = _run_artifact(tmp_path, backend=backend, write=True)

    assert [call["hf_id"] for call in backend.calls] == list(mod.MANDATED_MODEL_IDS)
    assert all(call["prompt_count"] == 240 for call in backend.calls)
    assert all(call["grammar"] is None for call in backend.calls)
    assert all(call["memory"] == "none" for call in backend.calls)
    assert all(call["label_conditioned_retry"] is False for call in backend.calls)

    assert artifact["status"] == "complete_ready"
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert artifact["sota_decision_corpus_ready_score"] == 1.0
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["inference_substrate"] == mod.LIVE_INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert mod.validate_artifact(artifact) is True
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text()) == artifact

    assert artifact["MODEL_SPECS"] == artifact["model_specs"]
    assert [row["hf_id"] for row in artifact["MODEL_SPECS"]] == list(mod.MANDATED_MODEL_IDS)
    assert "Qwen/Qwen3.5-0.8B" not in json.dumps(artifact["MODEL_SPECS"])
    assert "google/gemma-4-E4B-it" not in json.dumps(artifact["MODEL_SPECS"])
    assert all(row["actual_use_count"] == 240 for row in artifact["MODEL_SPECS"])
    assert all(row["n_gpu_layers"] == -1 for row in artifact["MODEL_SPECS"])

    conservation = artifact["row_conservation_and_prior_corpus_nonoverlap"]
    assert conservation["expected_event_count"] == 240
    assert conservation["all_models_conserved"] is True
    assert conservation["prior_exp6146_nonoverlap"]["all_overlap_counts_zero"] is True
    assert conservation["prompt_outcome_leakage_count"] == 0

    for hf_id in mod.MANDATED_MODEL_IDS:
        path = tmp_path / mod.row_sidecar_filename(hf_id)
        rows = _load_jsonl(path)
        assert len(rows) == 240
        assert rows[0]["event_id"] == "exp6159-event-000000"
        assert rows[0]["post_outcome_attached_after_decision"] is True
        assert rows[0]["decision_record_hash"].startswith("sha256:")
        assert rows[0]["post_outcome_id"] == rows[0]["event_id"]
        assert rows[0]["raw_response_hash"] == mod.sha256_text(rows[0]["raw_response"])
        assert rows[0]["answer_parse_state"] == "complete"

    row_paths = artifact["per_model_row_paths_hashes_and_counts"]["per_model"]
    assert set(row_paths) == set(mod.MANDATED_MODEL_IDS)
    assert all(receipt["row_count"] == 240 for receipt in row_paths.values())
    assert all(receipt["sha256"].startswith("sha256:") for receipt in row_paths.values())

    outcomes = artifact["exact_post_decision_outcome_receipts"]
    assert outcomes["post_decision_outcome_attachment_count"] == 480
    assert outcomes["validator_input_absent_from_model_inputs"] is True
    assert outcomes["all_outcomes_attached_after_decision"] is True

    counts = artifact["raw_response_strategy_answer_and_invalid_output_counts"]
    assert counts["total_invalid_output_count"] == 0
    assert counts["per_model"][mod.MANDATED_MODEL_IDS[0]]["raw_response_count"] == 240

    split_counts = artifact["chronological_split_family_and_shift_counts"]
    assert split_counts["source_event_count"] == 240
    assert split_counts["chronological_order_matches_exp6159"] is True
    assert split_counts["structural_shift_event_count"] > 0

    assert artifact["label_conditioned_retry_count"] == 0
    assert artifact["memory_read_and_write_counts"]["memory_read_count"] == 0
    assert artifact["memory_read_and_write_counts"]["memory_write_count"] == 0
    lifecycle = artifact["gpu_offload_pid_lifecycle_and_cleanup_receipts"]
    assert lifecycle["all_models_release_ready"] is True
    assert lifecycle["all_models_gpu_engaged"] is True
    assert artifact["protected_files_unchanged"]["unchanged"] is True


def test_scenario_6160_invalid_terminal_output_is_conserved(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-6160-7: invalid terminal output is honest, not retried."""

    artifact = _run_artifact(tmp_path, backend=FakeDecisionBackend(invalid_first=True))

    counts = artifact["raw_response_strategy_answer_and_invalid_output_counts"]
    assert counts["total_invalid_output_count"] == 2
    assert counts["per_model"][mod.MANDATED_MODEL_IDS[0]]["invalid_output_count"] == 1
    assert counts["per_model"][mod.MANDATED_MODEL_IDS[1]]["invalid_output_count"] == 1
    assert artifact["label_conditioned_retry_count"] == 0
    assert artifact["memory_read_and_write_counts"]["memory_read_count"] == 0
    assert artifact["sota_decision_corpus_ready_score"] == 1.0
    assert artifact["status"] == "complete_ready"
    assert mod.validate_artifact(artifact) is True


def test_scenario_6160_gate_blocks_missing_tokenizer_before_backend(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6160-GATE: missing mandated evidence blocks."""

    backend = FakeDecisionBackend()
    artifact = _run_artifact(tmp_path, backend=backend, gemma_ready=False)

    assert backend.calls == []
    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked:")
    assert "gemma_embedded_tokenizer_unloadable" in artifact["honest_verdict"]
    assert artifact["structured_gate_receipt"]["model_load_permitted"] is False
    assert artifact["sota_decision_corpus_ready_score"] == 0.0
    assert artifact["inference_substrate"] != mod.LIVE_INFERENCE_SUBSTRATE
    assert mod.validate_artifact(artifact) is True

    blocked_preconditions = _run_artifact(tmp_path, preconditions_ready=False)
    assert blocked_preconditions["status"] == "blocked"
    assert "fixture_precondition_block" in blocked_preconditions["honest_verdict"]


def test_req_6160_validation_rejects_leaks_retry_memory_and_checksum_drift(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-6160-5/9/10: readiness guards fail closed."""

    artifact = _run_artifact(tmp_path)

    assert mod._load_jsonl(tmp_path / "missing.jsonl") == []
    (tmp_path / "one.json").write_text('{"ok": true}', encoding="utf-8")
    assert mod._read_json(tmp_path / "one.json") == {"ok": True}
    assert mod._read_json(tmp_path / "missing.json") == {}
    assert mod._file_receipt(tmp_path, Path("missing.gguf"))["exists"] is False

    bad_order = deepcopy(artifact)
    bad_order["exact_post_decision_outcome_receipts"][
        "validator_input_absent_from_model_inputs"
    ] = False
    bad_order["reproducibility_checksum"] = mod.reproducibility_checksum(bad_order)
    assert "exact_post_decision_outcome_receipts" in mod._blocked_reasons(bad_order)
    with pytest.raises(ValueError, match="exact_post_decision_outcome_receipts"):
        mod.validate_artifact(bad_order)

    bad_retry = deepcopy(artifact)
    bad_retry["label_conditioned_retry_count"] = 1
    bad_retry["reproducibility_checksum"] = mod.reproducibility_checksum(bad_retry)
    assert "label_conditioned_retry_count" in mod._blocked_reasons(bad_retry)
    with pytest.raises(ValueError, match="label_conditioned_retry_count"):
        mod.validate_artifact(bad_retry)

    bad_memory = deepcopy(artifact)
    bad_memory["memory_read_and_write_counts"]["memory_read_count"] = 1
    bad_memory["reproducibility_checksum"] = mod.reproducibility_checksum(bad_memory)
    assert "memory_read_and_write_counts" in mod._blocked_reasons(bad_memory)
    with pytest.raises(ValueError, match="memory_read_and_write_counts"):
        mod.validate_artifact(bad_memory)

    bad_leak = deepcopy(artifact)
    bad_leak["row_conservation_and_prior_corpus_nonoverlap"]["prompt_outcome_leakage_count"] = 1
    bad_leak["reproducibility_checksum"] = mod.reproducibility_checksum(bad_leak)
    with pytest.raises(ValueError, match="row_conservation_and_prior_corpus_nonoverlap"):
        mod.validate_artifact(bad_leak)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = mod.sha256_text("wrong")
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)

    bad_provenance_type = deepcopy(artifact)
    bad_provenance_type["field_provenance"] = []
    bad_provenance_type["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_provenance_type
    )
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(bad_provenance_type)

    bad_provenance = deepcopy(artifact)
    bad_provenance["field_provenance"]["status"]["principle"] = "wrong"
    bad_provenance["reproducibility_checksum"] = mod.reproducibility_checksum(bad_provenance)
    with pytest.raises(ValueError, match="field_provenance:status"):
        mod.validate_artifact(bad_provenance)

    bad_model_specs = deepcopy(artifact)
    bad_model_specs["MODEL_SPECS"][1]["hf_id"] = "google/gemma-4-E4B-it"
    bad_model_specs["model_specs"] = deepcopy(bad_model_specs["MODEL_SPECS"])
    bad_model_specs["reproducibility_checksum"] = mod.reproducibility_checksum(bad_model_specs)
    with pytest.raises(ValueError, match="MODEL_SPECS"):
        mod.validate_artifact(bad_model_specs)

    bad_model_specs_mirror = deepcopy(artifact)
    bad_model_specs_mirror["model_specs"][0]["actual_use_count"] = -1
    bad_model_specs_mirror["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_model_specs_mirror
    )
    with pytest.raises(ValueError, match="model_specs"):
        mod.validate_artifact(bad_model_specs_mirror)

    bad_score = deepcopy(artifact)
    bad_score["sota_decision_corpus_ready_score"] = 0.0
    bad_score["reproducibility_checksum"] = mod.reproducibility_checksum(bad_score)
    with pytest.raises(ValueError, match="sota_decision_corpus_ready_score"):
        mod.validate_artifact(bad_score)

    bad_status = deepcopy(artifact)
    bad_status["status"] = "blocked"
    bad_status["reproducibility_checksum"] = mod.reproducibility_checksum(bad_status)
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(bad_status)

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "complete_ready: wrong"
    bad_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(bad_verdict)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = mod.BLOCKED_INFERENCE_SUBSTRATE
    bad_substrate["reproducibility_checksum"] = mod.reproducibility_checksum(bad_substrate)
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    bad_verifier = deepcopy(artifact)
    bad_verifier["verifier_is_oracle"] = False
    bad_verifier["reproducibility_checksum"] = mod.reproducibility_checksum(bad_verifier)
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.validate_artifact(bad_verifier)

    missing = dict(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(missing)

    row = mod.source_stream_bundle().rows[0]
    tampered = deepcopy(row)
    tampered["pre_decision"]["task_descriptor"]["exact_answer"] = "leak"
    assert mod.prompt_for_row(tampered)["contains_forbidden_token"] is True


def test_req_6160_worker_nonzero_exit_becomes_partial_not_ready(tmp_path: Path) -> None:
    """REQ-VERIFY-6160-8: backend transport failure is not credited."""

    class NonzeroBackend(FakeDecisionBackend):
        def generate(self, **kwargs: Any) -> dict[str, Any]:
            receipt = super().generate(**kwargs)
            receipt["worker_exit_code"] = 7
            return receipt

    artifact = _run_artifact(tmp_path, backend=NonzeroBackend())

    assert artifact["status"] == "complete_partial"
    assert artifact["sota_decision_corpus_ready_score"] == 0.0
    assert "worker_nonzero_exit" in artifact["honest_verdict"]
    assert "worker_nonzero_exit" in ",".join(
        artifact["gpu_offload_pid_lifecycle_and_cleanup_receipts"][
            "model_specific_transport_failures"
        ]
    )
    assert mod.validate_artifact(artifact) is True
