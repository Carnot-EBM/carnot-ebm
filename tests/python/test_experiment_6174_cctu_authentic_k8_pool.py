"""Tests for Exp6174 authentic CCTU K8 candidate-pool generation.

Spec refs: REQ-VERIFY-6174, SCENARIO-VERIFY-6174-GATE,
SCENARIO-VERIFY-6174-RAW-BEFORE-LABEL, SCENARIO-VERIFY-6174-RETENTION,
SCENARIO-VERIFY-6174-RESUME.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6174_cctu_authentic_k8_pool as mod
from carnot.verify import cctu_item_bank_6173 as exp6173


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/verification/spec.md"


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _preconditions(tmp_path: Path, *, ready: bool = True) -> dict[str, Any]:
    protected_before = {
        relative.as_posix(): mod.sha256_file(REPO / relative)
        for relative in mod.PROTECTED_FILES
        if (REPO / relative).exists()
    }
    return {
        "schema": mod.SCHEMA + ".preconditions",
        "run_date": mod.RUN_DATE,
        "preconditions_ready": ready,
        "blocked_reasons": [] if ready else ["fixture_gate_block"],
        "checks": {
            "structured_exp6173_gate": ready,
            "item_bank_hash_verified": ready,
            "split_hash_verified": ready,
            "validator_hash_verified": ready,
            "held_seal_verified": ready,
            "mandatory_gemma31b_cached": ready,
            "embedded_tokenizer_ok": ready,
            "embedded_chat_template_ok": ready,
            "llama_cpp_backend_ok": ready,
            "dual_gpu_capacity_ok": ready,
            "task_owned_process": ready,
            "output_paths_writable": True,
            "protected_files_present": True,
            "root_clutter_absent": True,
        },
        "held_access": {
            "generation_held_label_access_count": 0,
            "calibration_label_access_log_path": str(tmp_path / "calibration_access.json"),
            "held_label_access_log_path": str(tmp_path / "held_access.json"),
        },
        "output_paths": {
            "result_path": str(tmp_path / mod.RESULT_RELATIVE_PATH.name),
            "raw_trace_path": str(tmp_path / mod.RAW_TRACE_RELATIVE_PATH.name),
            "checkpoint_path": str(tmp_path / mod.CHECKPOINT_RELATIVE_PATH.name),
            "parent_writable": True,
        },
        "gpu": {
            "gpu_count": 2 if ready else 0,
            "ok": ready,
            "devices": [
                {"index": 0, "name": "RTX 3090", "memory_total_mb": 24576, "memory_free_mb": 24000},
                {"index": 1, "name": "RTX 3090", "memory_total_mb": 24576, "memory_free_mb": 24000},
            ]
            if ready
            else [],
            "compute_apps_before": [],
        },
        "protected_file_hashes_before": protected_before,
        "exclusion_manifest": {"path": "ops/exclusion_manifest.yaml", "exists": True},
    }


def _model_resolution(tmp_path: Path, *, ready: bool = True) -> dict[str, Any]:
    model_path = tmp_path / "gemma-4-31B-it-Q4_K_M.gguf"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"GGUF fixture gemma31b")
    record = {
        "name": "Gemma4-31B-it",
        "hf_id": mod.MANDATORY_MODEL_ID,
        "model_path": str(model_path),
        "real_path": str(model_path),
        "revision": "fixture-snapshot",
        "quantization": "Q4_K_M",
        "sha256": mod.sha256_file(model_path),
        "size_bytes": model_path.stat().st_size,
        "exists": ready,
        "loader": "llama_cpp.Llama",
        "n_gpu_layers": -1,
        "gpu_assignment": {"main_gpu": 0, "visible_devices": [0, 1], "split_mode": "layer"},
        "embedded_tokenizer_loadable": ready,
        "embedded_tokenizer_detail": "embedded GGUF tokenizer OK",
        "chat_template_present": ready,
        "chat_template_sha256": mod.sha256_text("fixture chat template"),
        "chat_template_source": "tokenizer.chat_template",
        "metadata_summary_sha256": mod.sha256_text("fixture metadata"),
        "actual_use_count": 0,
    }
    return {
        "schema": mod.SCHEMA + ".model_resolution",
        "records": [record],
        "blocked_reasons": [] if ready else ["fixture_model_block"],
    }


class FakeK8Backend:
    """SCENARIO-VERIFY-6174-RETENTION: emit valid, invalid, duplicate rows."""

    def __init__(self) -> None:
        self.calls = 0
        self.requested_keys: list[str] = []

    def generate(
        self,
        *,
        model_spec: dict[str, Any],
        public_cases: list[dict[str, Any]],
        sample_plan: list[dict[str, Any]],
        decode_policy: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls += 1
        assert model_spec["hf_id"] == mod.MANDATORY_MODEL_ID
        assert decode_policy["correctness_conditioned_retry"] is False
        assert decode_policy["parser_repair"] is False
        cases = {case.case_id: case for case in exp6173.build_item_bank()}
        rows: list[dict[str, Any]] = []
        public_by_id = {case["case_id"]: case for case in public_cases}
        for entry in sample_plan:
            self.requested_keys.append(entry["sample_key"])
            case = cases[entry["case_id"]]
            assert "expected_steps" not in public_by_id[entry["case_id"]]
            raw = json.dumps(exp6173.known_valid_trace(case), sort_keys=True)
            finish_reason = "stop"
            timeout = False
            refusal = False
            if entry["sample_index"] == 1:
                raw = "{not json"
            elif entry["sample_index"] == 2:
                raw = json.dumps(exp6173.known_valid_trace(case), sort_keys=True)
            elif entry["sample_index"] == 3:
                raw = "I cannot provide a tool trace for this request."
                refusal = True
            elif entry["sample_index"] == 4:
                raw = ""
                finish_reason = "timeout"
                timeout = True
            elif entry["sample_index"] == 5:
                raw = raw[:80]
                finish_reason = "length"
            elif entry["sample_index"] == 6:
                raw = json.dumps(
                    exp6173.mutate_trace(exp6173.known_valid_trace(case), "wrong_final"),
                    sort_keys=True,
                )
            rows.append(
                {
                    "case_id": entry["case_id"],
                    "sample_index": entry["sample_index"],
                    "sample_key": entry["sample_key"],
                    "raw_completion_text": raw,
                    "finish_reason": finish_reason,
                    "timeout": timeout,
                    "refusal": refusal,
                    "truncated": finish_reason == "length",
                    "completion_token_count": 0 if not raw else 17 + entry["sample_index"],
                    "prompt_token_count": 211,
                    "native_tool_calls": [],
                    "native_logprobs": None,
                    "timing": {"decode_time_s": 0.001, "started_monotonic_s": 1.0},
                }
            )
        return {
            "schema": mod.SCHEMA + ".backend_generation",
            "rows": rows,
            "lifecycle_receipt": {
                "worker_pid": 617400,
                "worker_exit_code": 0,
                "pid_exited": True,
                "vram_release_observed": True,
                "orphan_task_owned_pid_count": 0,
                "retained_task_owned_vram_mb": 0,
                "gpu_engagement": {
                    "attributable": True,
                    "selected_gpus": [0, 1],
                    "max_memory_delta_mb": 18100,
                },
                "timeline": [
                    {"phase": "before_load", "devices": []},
                    {"phase": "decode", "devices": []},
                    {"phase": "release", "devices": []},
                ],
            },
        }


def _run_artifact(
    tmp_path: Path,
    *,
    backend: FakeK8Backend | None = None,
    ready: bool = True,
    write: bool = True,
) -> dict[str, Any]:
    return mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        raw_trace_path=tmp_path / mod.RAW_TRACE_RELATIVE_PATH.name,
        calibration_label_path=tmp_path / mod.CALIBRATION_LABEL_RELATIVE_PATH.name,
        held_label_path=tmp_path / mod.HELD_LABEL_RELATIVE_PATH.name,
        checkpoint_path=tmp_path / mod.CHECKPOINT_RELATIVE_PATH.name,
        calibration_access_log_path=tmp_path / mod.CALIBRATION_ACCESS_LOG_RELATIVE_PATH.name,
        held_access_log_path=tmp_path / mod.HELD_ACCESS_LOG_RELATIVE_PATH.name,
        preconditions_checked=_preconditions(tmp_path, ready=ready),
        model_resolution=_model_resolution(tmp_path, ready=ready),
        generation_backend=backend,
        test_exit_codes=_passing_exit_codes(),
        duration_s=6.174,
        write=write,
    )


def test_req_6174_spec_declares_authentic_k8_contract() -> None:
    """REQ-VERIFY-6174: OpenSpec declares model, K8, raw-before-label rules."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-VERIFY-6174") :]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-VERIFY-6174-GATE",
        "SCENARIO-VERIFY-6174-RAW-BEFORE-LABEL",
        "SCENARIO-VERIFY-6174-RETENTION",
        "SCENARIO-VERIFY-6174-RESUME",
        mod.MANDATORY_MODEL_ID,
        "llama_cpp_local_gemma4_31b_gguf_native_chat_tool_trace_generation",
    ):
        assert marker in normalized


def test_scenario_6174_gate_blocks_without_generation(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6174-GATE: missing preconditions block and do not call backend."""

    backend = FakeK8Backend()
    artifact = _run_artifact(tmp_path, backend=backend, ready=False)

    assert backend.calls == 0
    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["cctu_candidate_pool_integrity_score"] == 0.0
    assert artifact["raw_trace_corpus_path_hash_count_and_schema"]["count"] == 0
    assert artifact["model_specs"][0]["hf_id"] == mod.MANDATORY_MODEL_ID
    assert "Qwen/Qwen3.5-0.8B" not in json.dumps(artifact["model_specs"])


def test_scenario_6174_raw_before_label_and_retention_counts(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6174-RAW-BEFORE-LABEL: raw corpus precedes label sidecars."""

    backend = FakeK8Backend()
    artifact = _run_artifact(tmp_path, backend=backend)

    raw_rows = _load_jsonl(tmp_path / mod.RAW_TRACE_RELATIVE_PATH.name)
    calibration_labels = _load_jsonl(tmp_path / mod.CALIBRATION_LABEL_RELATIVE_PATH.name)
    held_labels = _load_jsonl(tmp_path / mod.HELD_LABEL_RELATIVE_PATH.name)
    expected_rows = len(exp6173.build_item_bank()) * mod.K_SAMPLES

    assert backend.calls == 1
    assert len(backend.requested_keys) == expected_rows
    assert len(set(backend.requested_keys)) == expected_rows
    assert len(raw_rows) == expected_rows
    assert len(calibration_labels) == expected_rows // 2
    assert len(held_labels) == expected_rows // 2
    assert artifact["status"] == "complete_ready"
    assert artifact["cctu_candidate_pool_integrity_score"] == 1.0
    assert artifact["raw_before_label_commit_receipts"]["validation_started_after_raw_commit"] is True
    assert artifact["raw_before_label_commit_receipts"]["raw_corpus_sha256"] == mod.sha256_file(
        tmp_path / mod.RAW_TRACE_RELATIVE_PATH.name
    )
    counts = artifact["parse_failure_duplicate_refusal_timeout_and_truncation_counts"]
    assert counts["parse_failure_count"] >= 120
    assert counts["duplicate_raw_completion_count"] >= 120
    assert counts["refusal_count"] == 120
    assert counts["timeout_count"] == 120
    assert counts["truncation_count"] == 120
    assert artifact["no_correctness_conditioned_retry_or_replacement_receipt"] == {
        "correctness_conditioned_retry_count": 0,
        "parser_repair_count": 0,
        "model_judge_count": 0,
        "candidate_replacement_count": 0,
        "preserved_all_raw_rows": True,
    }
    assert all(row["raw_committed_before_validation"] is True for row in calibration_labels[:5])
    assert all(row["split"] == "held" for row in held_labels)


def test_scenario_6174_resume_reuses_matching_raw_rows(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6174-RESUME: matching immutable rows are reused."""

    first = _run_artifact(tmp_path, backend=FakeK8Backend())
    backend = FakeK8Backend()
    second = _run_artifact(tmp_path, backend=backend)

    assert backend.calls == 0
    assert second["resume_idempotence_and_checkpoint_receipts"]["resume_mode"] == "reused_raw_corpus"
    assert second["resume_idempotence_and_checkpoint_receipts"]["generated_new_rows"] == 0
    assert second["raw_trace_corpus_path_hash_count_and_schema"] == first[
        "raw_trace_corpus_path_hash_count_and_schema"
    ]


def test_scenario_6174_resume_blocks_conflicting_duplicate_key(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6174-RESUME: conflicting immutable rows block overwrite."""

    _run_artifact(tmp_path, backend=FakeK8Backend())
    raw_path = tmp_path / mod.RAW_TRACE_RELATIVE_PATH.name
    first_row = deepcopy(_load_jsonl(raw_path)[0])
    first_row["raw_completion_text"] = "conflicting duplicate key"
    raw_path.write_text(
        raw_path.read_text(encoding="utf-8") + json.dumps(first_row, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    artifact = _run_artifact(tmp_path, backend=FakeK8Backend())

    assert artifact["status"] == "blocked"
    assert artifact["resume_idempotence_and_checkpoint_receipts"]["conflicting_key_count"] == 1
    assert artifact["cctu_candidate_pool_integrity_score"] == 0.0


def test_req_6174_helper_branches_and_artifact_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-6174: deterministic helper branches remain covered."""

    assert mod.neutral_json_parse_receipt('{"ok": true} trailing')["error"] == (
        "trailing_content_or_multiple_json_values"
    )
    assert mod.honest_verdict(
        "complete_partial",
        {"case_count": 1, "raw_row_count": 8},
        {"hf_id": "fixture/model"},
        {},
    ).startswith("complete_partial:")
    assert mod.snapshot_revision(Path("/cache/models--x/snapshots/rev123/model.gguf")) == "rev123"
    assert mod.snapshot_revision(Path("/flat/model.gguf")) == "local-flat-cache"
    assert mod.observed_quantization(Path("gemma-4-31B-it-Q5_K_M.gguf")) == "Q5_K_M"
    assert mod.observed_quantization(Path("gemma.gguf")) == "unknown"
    assert mod.model_slug(mod.MANDATORY_MODEL_ID) == "gemma-4-31b-it"
    assert mod.file_receipts([Path("definitely-missing-exp6174.txt")])[0]["exists"] is False
    assert mod._model_specs_from_resolution({})[0]["hf_id"] == mod.MANDATORY_MODEL_ID
    assert mod._run_command(("definitely-missing-exp6174-command",))["returncode"] == 127
    assert mod._parent_writable(tmp_path / "nested" / "artifact.json") is True

    monkeypatch.setattr(mod, "resolve_cached_gguf", lambda *_args, **_kwargs: None)
    missing = mod.resolve_mandatory_model()
    assert missing["blocked_reasons"] == ["mandatory_gemma31b_gguf_not_cached"]

    model_path = tmp_path / "gemma-4-31B-it-Q4_K_M.gguf"
    model_path.write_bytes(b"GGUF fixture")
    monkeypatch.setattr(mod, "resolve_cached_gguf", lambda *_args, **_kwargs: str(model_path))
    monkeypatch.setattr(
        mod,
        "gguf_tokenizer_loadable",
        lambda _path: (False, "fixture tokenizer failure"),
    )
    monkeypatch.setattr(
        mod,
        "gguf_metadata_receipt",
        lambda _path: {
            "chat_template_present": False,
            "chat_template_sha256": None,
            "metadata_summary_sha256": mod.sha256_text("fixture metadata"),
        },
    )
    unresolved = mod.resolve_mandatory_model()
    assert unresolved["blocked_reasons"] == [
        "embedded_tokenizer_unloadable",
        "embedded_chat_template_missing",
    ]

    original_root = mod.REPO_ROOT
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    prereg = tmp_path / "results" / "experiment_6173_cctu_item_bank_preregistration.json"
    prereg.parent.mkdir(parents=True)
    prereg.write_text("{not json", encoding="utf-8")
    assert mod.upstream_bank_split_validator_and_preregistration_hashes()["preregistration"][
        "status"
    ] is None
    monkeypatch.setattr(mod, "REPO_ROOT", original_root)

    artifact = _run_artifact(tmp_path / "validate", backend=FakeK8Backend())
    assert artifact["status"] == "complete_ready"
    validation = mod.validate_existing_artifact(
        tmp_path / "validate" / mod.RESULT_RELATIVE_PATH.name
    )
    assert validation["ok"] is True
