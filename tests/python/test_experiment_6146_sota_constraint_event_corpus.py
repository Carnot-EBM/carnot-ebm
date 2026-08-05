"""Tests for Exp6146 SOTA constraint event corpus.

Spec refs: REQ-VERIFY-6146, REQ-VERIFY-6146-1, REQ-VERIFY-6146-2,
REQ-VERIFY-6146-3, REQ-VERIFY-6146-4, REQ-VERIFY-6146-5,
REQ-VERIFY-6146-6, REQ-VERIFY-6146-7, REQ-VERIFY-6146-8,
REQ-VERIFY-6146-9, SCENARIO-VERIFY-6146-GATE,
SCENARIO-VERIFY-6146-ORDERING, SCENARIO-VERIFY-6146-NO-MEMORY,
SCENARIO-VERIFY-6146-LIFECYCLE.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6145_constraint_shift_stream as exp6145
from carnot import experiment_6146_sota_constraint_event_corpus as mod


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
        "hashed_input_receipts": [],
        "gpu": {
            "gpu_count": 2,
            "ok": True,
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
            "task_owned_pid": 614600,
            "parent_pid": 1,
            "lease_scope": "task_owned_child_workers_only",
            "no_inherited_model_server": True,
        },
        "output_paths": {
            "result_path": str(tmp_path / mod.RESULT_RELATIVE_PATH.name),
            "row_sidecar_dir": str(tmp_path),
            "parent_writable": True,
        },
        "protected_file_hashes_before": before,
        "root_clutter": {"root_python_file_count": 0, "ok": True},
    }


def _model_resolution(tmp_path: Path, *, gemma_ready: bool = True) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for index, hf_id in enumerate(mod.MANDATED_MODEL_IDS):
        slug = mod.model_slug(hf_id)
        path = tmp_path / f"{slug}-Q4_K_M.gguf"
        path.write_bytes(b"GGUF" + slug.encode("ascii"))
        records.append(
            {
                "name": "Qwen3.6-35B-A3B" if index == 0 else "Gemma4-31B-it",
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
                "loader": "llama_cpp.Llama",
                "n_gpu_layers": -1,
                "actual_use_count": 0,
            }
        )
    return {
        "schema": mod.SCHEMA + ".model_resolution",
        "records": records,
        "blocked_reasons": [] if gemma_ready else ["gemma_embedded_tokenizer_unloadable"],
    }


class FakeSotaBackend:
    """REQ-VERIFY-6146-7: deterministic task-owned native-chat backend."""

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
                "finite_id_transport": decode_config["finite_id_transport"],
                "memory": decode_config["memory"],
            }
        )
        rows: list[dict[str, Any]] = []
        for index, prompt in enumerate(prompts):
            payload = json.dumps(prompt, sort_keys=True)
            assert "exact_answer" not in payload
            assert "current_validator_result" not in payload
            assert "post_outcome" not in payload
            raw = (
                ""
                if self.invalid_first and index == 0
                else (
                    "STRATEGY_ID: visible_graph_v1\n"
                    "STRATEGY: inspect only the task descriptor and graph summary.\n"
                    f"SOLUTION: proposed terminal solution for {prompt['event_id']}"
                )
            )
            rows.append(
                {
                    "event_id": prompt["event_id"],
                    "raw_response": raw,
                    "generated_token_count": 0 if raw == "" else 37 + index,
                    "decode_time_s": round(0.01 + index / 10_000, 6),
                    "finish_reason": "stop",
                    "seed": prompt["seed"],
                }
            )
        gpu = int(model_spec["gpu"])
        pid = 614600 + gpu
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
                            "memory_used_mb": 18000,
                            "memory_free_mb": 6576,
                            "temperature_c": 63,
                            "power_draw_w": 260.0,
                        }
                    ],
                    "compute_apps": [{"pid": pid, "used_memory_mb": 18000}],
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
                "selected_gpu_memory_delta_mb": 17984,
                "n_gpu_layers": -1,
            },
            "rows": rows,
        }


def test_req_6146_spec_declares_mandated_model_corpus_contract() -> None:
    """REQ-VERIFY-6146: OpenSpec names requirements, fields, and principles."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-VERIFY-6146") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-6146-1",
        "REQ-VERIFY-6146-2",
        "REQ-VERIFY-6146-3",
        "REQ-VERIFY-6146-4",
        "REQ-VERIFY-6146-5",
        "REQ-VERIFY-6146-6",
        "REQ-VERIFY-6146-7",
        "REQ-VERIFY-6146-8",
        "REQ-VERIFY-6146-9",
        "SCENARIO-VERIFY-6146-GATE",
        "SCENARIO-VERIFY-6146-ORDERING",
        "SCENARIO-VERIFY-6146-NO-MEMORY",
        "SCENARIO-VERIFY-6146-LIFECYCLE",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.MANDATED_MODEL_IDS[0],
        mod.MANDATED_MODEL_IDS[1],
        mod.LIVE_INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_6146_complete_ready_conserves_rows_and_orders_outcomes(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6146-ORDERING/LIFECYCLE: authentic rows pass."""

    backend = FakeSotaBackend()
    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        row_sidecar_dir=tmp_path,
        preconditions_checked=_preconditions(tmp_path),
        model_resolution=_model_resolution(tmp_path),
        generation_backend=backend,
        test_exit_codes=_passing_exit_codes(),
        duration_s=6.146,
        write=True,
    )

    assert [call["hf_id"] for call in backend.calls] == list(mod.MANDATED_MODEL_IDS)
    assert all(call["prompt_count"] == 240 for call in backend.calls)
    assert all(call["grammar"] is None for call in backend.calls)
    assert all(call["finite_id_transport"] is False for call in backend.calls)
    assert all(call["memory"] == "none" for call in backend.calls)

    assert artifact["status"] == "complete_ready"
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert artifact["sota_constraint_event_corpus_ready_score"] == 1
    assert artifact["inference_substrate"] == mod.LIVE_INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert mod.validate_artifact(artifact) is True
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text()) == artifact

    assert len(artifact["model_specs"]) == 2
    assert [row["hf_id"] for row in artifact["model_specs"]] == list(mod.MANDATED_MODEL_IDS)
    assert all(row["actual_use_count"] == 240 for row in artifact["model_specs"])
    assert all(row["n_gpu_layers"] == -1 for row in artifact["model_specs"])

    conservation = artifact["per_model_event_row_conservation"]
    assert conservation["expected_event_count"] == 240
    assert conservation["all_models_conserved"] is True
    assert set(conservation["per_model"]) == set(mod.MANDATED_MODEL_IDS)

    for hf_id in mod.MANDATED_MODEL_IDS:
        path = tmp_path / mod.row_sidecar_filename(hf_id)
        rows = _load_jsonl(path)
        assert len(rows) == 240
        assert rows[0]["event_id"] == "exp6145-event-000000"
        assert rows[0]["post_outcome_attached_after_decision"] is True
        assert rows[0]["decision_record_hash"].startswith("sha256:")
        assert rows[0]["post_outcome_id"] == rows[0]["event_id"]
        assert rows[0]["raw_response_hash"] == mod.sha256_text(rows[0]["raw_response"])

    outcomes = artifact["post_decision_exact_outcome_receipts"]
    assert outcomes["post_decision_outcome_attachment_count"] == 480
    assert outcomes["validator_input_absent_from_model_inputs"] is True
    assert outcomes["all_outcomes_attached_after_decision"] is True

    no_retry = artifact["no_memory_and_no_adaptive_retry_receipts"]
    assert no_retry["memory_policy"] == "none"
    assert no_retry["adaptive_retry_count"] == 0
    assert no_retry["grammar_count"] == 0
    assert no_retry["parser_repair_count"] == 0
    assert no_retry["hidden_label_in_prompt_count"] == 0

    lifecycle = artifact["lifecycle_timing_and_cleanup_receipts"]
    assert lifecycle["all_models_release_ready"] is True
    assert lifecycle["orphan_task_owned_pid_count"] == 0
    assert lifecycle["retained_task_owned_vram_mb"] == 0
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert artifact["field_provenance"][field]["principle"] == principle


def test_scenario_6146_invalid_terminal_output_is_honest_not_retry(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-6146-6: invalid terminal output is conserved."""

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        row_sidecar_dir=tmp_path,
        preconditions_checked=_preconditions(tmp_path),
        model_resolution=_model_resolution(tmp_path),
        generation_backend=FakeSotaBackend(invalid_first=True),
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
        write=False,
    )

    counts = artifact["strategy_terminal_solution_and_invalid_output_counts"]
    assert counts["total_invalid_output_count"] == 2
    assert counts["per_model"][mod.MANDATED_MODEL_IDS[0]]["invalid_output_count"] == 1
    assert counts["per_model"][mod.MANDATED_MODEL_IDS[1]]["invalid_output_count"] == 1
    assert artifact["no_memory_and_no_adaptive_retry_receipts"]["adaptive_retry_count"] == 0
    assert artifact["sota_constraint_event_corpus_ready_score"] == 1
    assert artifact["status"] == "complete_ready"
    assert mod.validate_artifact(artifact) is True


def test_scenario_6146_gate_blocks_missing_tokenizer_before_backend(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6146-GATE: missing mandated evidence blocks."""

    backend = FakeSotaBackend()
    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        row_sidecar_dir=tmp_path,
        preconditions_checked=_preconditions(tmp_path),
        model_resolution=_model_resolution(tmp_path, gemma_ready=False),
        generation_backend=backend,
        test_exit_codes=_passing_exit_codes(),
        duration_s=0.5,
        write=False,
    )

    assert backend.calls == []
    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked:")
    assert "gemma_embedded_tokenizer_unloadable" in artifact["honest_verdict"]
    assert artifact["structured_gate_receipt"]["model_load_permitted"] is False
    assert artifact["sota_constraint_event_corpus_ready_score"] == 0
    assert artifact["inference_substrate"] != mod.LIVE_INFERENCE_SUBSTRATE
    assert artifact["tiny_model_smoke_rows_excluded_from_headline"]["headline_use_count"] == 0
    assert mod.validate_artifact(artifact) is True


def test_req_6146_validation_rejects_ordering_retry_and_checksum_drift(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-6146-8/9: readiness and provenance guards fail closed."""

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        row_sidecar_dir=tmp_path,
        preconditions_checked=_preconditions(tmp_path),
        model_resolution=_model_resolution(tmp_path),
        generation_backend=FakeSotaBackend(),
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
        write=False,
    )

    bad_order = deepcopy(artifact)
    bad_order["post_decision_exact_outcome_receipts"][
        "validator_input_absent_from_model_inputs"
    ] = False
    bad_order["reproducibility_checksum"] = mod.reproducibility_checksum(bad_order)
    with pytest.raises(ValueError, match="post_decision_exact_outcome_receipts"):
        mod.validate_artifact(bad_order)

    bad_retry = deepcopy(artifact)
    bad_retry["no_memory_and_no_adaptive_retry_receipts"]["adaptive_retry_count"] = 1
    bad_retry["sota_constraint_event_corpus_ready_score"] = mod.ready_score(bad_retry)
    bad_retry["status"] = mod.status(bad_retry)
    bad_retry["honest_verdict"] = mod.honest_verdict(bad_retry)
    bad_retry["reproducibility_checksum"] = mod.reproducibility_checksum(bad_retry)
    assert bad_retry["sota_constraint_event_corpus_ready_score"] == 0
    with pytest.raises(ValueError, match="adaptive_retry_count"):
        mod.validate_artifact(bad_retry)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = mod.sha256_text("wrong")
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)

    missing = dict(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(missing)

    bundle = exp6145.build_stream_bundle()
    tampered = deepcopy(bundle.rows[0])
    tampered["pre_decision"]["post_outcome"] = {"exact_answer": ["leak"]}
    assert mod.prompt_for_row(tampered)["contains_forbidden_token"] is True


def test_req_6146_helper_receipts_and_validation_guard_branches(tmp_path: Path) -> None:
    """REQ-VERIFY-6146-2/8: helper receipts and fail-closed guards are covered."""

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        row_sidecar_dir=tmp_path,
        preconditions_checked=_preconditions(tmp_path),
        model_resolution=_model_resolution(tmp_path),
        generation_backend=FakeSotaBackend(),
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
        write=False,
    )

    snapshot_path = tmp_path / "snapshots" / "abc123" / "model-UD-Q8_XL.gguf"
    snapshot_path.parent.mkdir(parents=True)
    snapshot_path.write_bytes(b"GGUF")
    local_path = tmp_path / "local-model.gguf"
    local_path.write_bytes(b"GGUF")
    assert mod._extract_revision(snapshot_path) == "abc123"
    assert mod._extract_revision(local_path) == "project-local"
    assert mod._extract_quantization(snapshot_path) == "UD-Q8_XL"
    assert mod._extract_quantization(local_path) == "unknown"
    assert mod._is_projector_gguf(tmp_path / "MTP" / "mtp-model-Q8_0.gguf") is True
    assert mod._is_projector_gguf(snapshot_path) is False
    assert mod._file_receipt(tmp_path, Path("local-model.gguf"))["exists"] is True
    assert mod._file_receipt(tmp_path, Path("missing.gguf"))["exists"] is False

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

    bad_repair = deepcopy(artifact)
    bad_repair["no_memory_and_no_adaptive_retry_receipts"]["parser_repair_count"] = 1
    bad_repair["reproducibility_checksum"] = mod.reproducibility_checksum(bad_repair)
    with pytest.raises(ValueError, match="hidden_retry_or_repair"):
        mod.validate_artifact(bad_repair)

    bad_score = deepcopy(artifact)
    bad_score["sota_constraint_event_corpus_ready_score"] = 0
    bad_score["reproducibility_checksum"] = mod.reproducibility_checksum(bad_score)
    with pytest.raises(ValueError, match="sota_constraint_event_corpus_ready_score"):
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

    hidden_prompt = deepcopy(artifact)
    hidden_prompt["no_memory_and_no_adaptive_retry_receipts"]["hidden_label_in_prompt_count"] = 1
    assert "hidden_label_in_prompt" in mod._blocked_reasons(hidden_prompt)
    post_false = deepcopy(artifact)
    post_false["post_decision_exact_outcome_receipts"][
        "validator_input_absent_from_model_inputs"
    ] = False
    assert "post_decision_exact_outcome_receipts" in mod._blocked_reasons(post_false)


def test_req_6146_worker_nonzero_exit_becomes_partial_not_ready(tmp_path: Path) -> None:
    """REQ-VERIFY-6146-7: backend lifecycle failure is not credited."""

    class NonzeroBackend(FakeSotaBackend):
        def generate(self, **kwargs: Any) -> dict[str, Any]:
            receipt = super().generate(**kwargs)
            receipt["worker_exit_code"] = 7
            return receipt

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        row_sidecar_dir=tmp_path,
        preconditions_checked=_preconditions(tmp_path),
        model_resolution=_model_resolution(tmp_path),
        generation_backend=NonzeroBackend(),
        test_exit_codes=_passing_exit_codes(),
        duration_s=1.0,
        write=False,
    )

    assert artifact["status"] == "complete_partial"
    assert artifact["sota_constraint_event_corpus_ready_score"] == 0
    assert "worker_nonzero_exit" in artifact["honest_verdict"]
    assert mod.validate_artifact(artifact) is True
