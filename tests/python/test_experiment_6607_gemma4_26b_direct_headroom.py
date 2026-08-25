"""Tests for REQ-REPORT-6607 and its direct-baseline scenarios."""

from __future__ import annotations

import base64
from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6607_gemma4_26b_direct_headroom as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/research-reporting/spec.md"
CORPUS_PATH = REPO / "results/experiment_6604_exact_two_level_plan_corpus.json"


@pytest.fixture(scope="module")
def corpus() -> dict:
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


def _model_identity() -> dict:
    spec = deepcopy(mod.MODEL_SPECS[0])
    return {
        "MODEL_SPECS": [spec],
        "hub_id": mod.GEMMA26_HUB_ID,
        "model_path": spec["model_path"],
        "model_sha256": "sha256:gemma26-model",
        "gguf_shards": [
            {
                "path": spec["model_path"],
                "sha256": "sha256:gemma26-model",
                "byte_count": 22_134_528_992,
            }
        ],
        "quantization": "UD-Q4_K_M",
        "architecture": mod.GEMMA26_ARCHITECTURE,
        "context_length": 262_144,
        "embedded_tokenizer": {
            "source": "embedded_gguf",
            "loadable": True,
            "identity_sha256": "sha256:tokenizer",
            "token_count": 248_320,
        },
        "embedded_chat_template": {
            "source": "tokenizer.chat_template",
            "present": True,
            "sha256": "sha256:chat-template",
        },
        "llama_cpp": {"cuda_linked": True, "version": "9606"},
        "auto_tokenizer_used": False,
        "download_performed": False,
        "legacy_headline_row_count": 0,
    }


def _process_binding() -> dict:
    return {
        "session_id": "session-1",
        "pid": 777,
        "parent_pid": 42,
        "owned_child": True,
        "repository_id": mod.GEMMA26_HUB_ID,
        "model_sha256": "sha256:gemma26-model",
        "command_sha256": "sha256:command",
        "selected_gpu": 0,
        "gpu_uuid": "GPU-0",
        "cpu_fallback": False,
        "cuda_offload": True,
        "offloaded_layers": 65,
        "tokenizer_source": "embedded_gguf",
        "chat_template_sha256": "sha256:chat-template",
    }


def _gpu_receipts(row_count: int) -> dict:
    return {
        "sessions": [
            {
                "session_id": "session-1",
                "pid": 777,
                "parent_pid": 42,
                "owned_child": True,
                "repository_id": mod.GEMMA26_HUB_ID,
                "model_sha256": "sha256:gemma26-model",
                "command_sha256": "sha256:command",
                "selected_gpu": 0,
                "gpu_uuid": "GPU-0",
                "cuda_visible_devices": "0",
                "cpu_fallback": False,
                "cuda_offload": True,
                "offloaded_layers": 65,
                "server_healthy": True,
                "row_count": row_count,
                "samples": [
                    {"stage": "before", "memory_used_mb": 4, "utilization_pct": 0},
                    {
                        "stage": "during",
                        "memory_used_mb": 21_000,
                        "utilization_pct": 70,
                        "worker_pid_present": True,
                    },
                    {
                        "stage": "during",
                        "memory_used_mb": 21_010,
                        "utilization_pct": 72,
                        "worker_pid_present": True,
                    },
                    {"stage": "after", "memory_used_mb": 4, "utilization_pct": 0},
                ],
                "shutdown_requested": True,
                "normal_shutdown": True,
                "worker_absent_after_exit": True,
                "port_closed": True,
                "memory_recovered": True,
                "signals_sent_to_unrelated_pids": [],
            }
        ],
        "all_sessions_authentic": True,
    }


def _protected() -> dict:
    return {
        "all_unchanged": True,
        "rows": [
            {
                "path": path.as_posix(),
                "before_sha256": "sha256:same",
                "after_sha256": "sha256:same",
                "unchanged": True,
            }
            for path in mod.PROTECTED_RELATIVE_PATHS
        ],
    }


def _generation(raw: bytes, *, failure: str | None = None, finish: str = "stop") -> dict:
    return {
        "raw_response_bytes": raw,
        "raw_api_response_sha256": "sha256:api",
        "prompt_tokens": 100,
        "completion_tokens": 12,
        "finish_reason": finish,
        "http_status": 200 if failure is None else 124,
        "failure_kind": failure,
        "started_monotonic_ns": 100,
        "finished_monotonic_ns": 200,
        "latency_s": 0.25,
    }


def _rows_with_half_held_success(corpus: dict) -> tuple[list[dict], dict]:
    contract = mod.build_prompt_and_decode_contract(corpus)
    jobs = mod.build_generation_jobs(corpus, contract)
    held_successes = 0
    rows = []
    for job in jobs:
        witness = job["task"].get("gold_witness")
        use_witness = job["split"] == "held" and witness and held_successes < 54
        if use_witness:
            held_successes += 1
        raw = str(witness).encode("utf-8") if use_witness else b"NOT_A_CANONICAL_PLAN"
        rows.append(mod.build_per_unit_row(job, _generation(raw), _process_binding()))
    assert held_successes == 54
    return rows, contract


def _complete_artifact(corpus: dict) -> dict:
    rows, contract = _rows_with_half_held_success(corpus)
    return mod.assemble_artifact(
        run_date="20260825",
        exp6604_artifact=corpus,
        prompt_contract=contract,
        per_unit_rows=rows,
        model_identity=_model_identity(),
        gpu_receipts=_gpu_receipts(len(rows)),
        checkpoint_receipts={
            "accepted_row_count": len(rows),
            "completed_prefix_hash": mod.sha256_json(rows),
            "prompt_contract_sha256": contract["contract_sha256"],
            "model_sha256": "sha256:gemma26-model",
            "atomic_replace": True,
            "directory_fsync": True,
        },
        preconditions={
            "all_required_preconditions_available": True,
            "failed_preconditions": [],
            "checks": {"upstream_gate": True, "gpu": True, "model": True},
        },
        protected_files=_protected(),
        tests_run=[{"command": "focused", "exit_code": 0, "duration_s": 0.1}],
        duration_s=123.0,
    )


def test_req_report_6607_spec_declares_direct_baseline_contract() -> None:
    """REQ-REPORT-6607 declares every direct baseline safeguard."""

    text = SPEC.read_text(encoding="utf-8")
    for anchor in (
        "REQ-REPORT-6607-PRECONDITIONS",
        "REQ-REPORT-6607-MODEL",
        "REQ-REPORT-6607-PROMPT",
        "REQ-REPORT-6607-ROWS",
        "REQ-REPORT-6607-FAILURES",
        "REQ-REPORT-6607-ORACLE",
        "REQ-REPORT-6607-REDUCER",
        "REQ-REPORT-6607-HEADROOM",
        "REQ-REPORT-6607-CHECKPOINT",
        "REQ-REPORT-6607-LIFECYCLE",
        "REQ-REPORT-6607-ATTACKS",
        "REQ-REPORT-6607-ATOMIC",
        "SCENARIO-REPORT-6607-BLOCKED",
        "SCENARIO-REPORT-6607-FROZEN",
        "SCENARIO-REPORT-6607-RAW-AND-ORACLE",
        "SCENARIO-REPORT-6607-HEADROOM",
        "SCENARIO-REPORT-6607-RESUME",
        "SCENARIO-REPORT-6607-ATTACKS-AND-ATOMIC",
    ):
        assert anchor in text


def test_scenario_report_6607_frozen_jobs_cover_all_task_seed_pairs(corpus: dict) -> None:
    """SCENARIO-REPORT-6607-FROZEN keeps exact prompt bytes and three seeds."""

    contract = mod.build_prompt_and_decode_contract(corpus)
    jobs = mod.build_generation_jobs(corpus, contract)
    assert len(jobs) == 216
    assert len({row["row_id"] for row in jobs}) == 216
    assert {row["seed"] for row in jobs} == set(mod.SEED_SCHEDULE)
    assert sum(row["split"] == "calibration" for row in jobs) == 108
    assert sum(row["split"] == "held" for row in jobs) == 108
    assert all(base64.b64decode(row["prompt_bytes_b64"]) == row["prompt_bytes"] for row in jobs)
    assert all(row["prompt_sha256"] == mod.sha256_bytes(row["prompt_bytes"]) for row in jobs)
    assert contract["grammar"] is None
    assert contract["semantic_mask"] is None
    assert contract["repair"] is False
    assert contract["cross_family_context"] is False


def test_scenario_report_6607_raw_bytes_precede_independent_oracle(corpus: dict) -> None:
    """SCENARIO-REPORT-6607-RAW-AND-ORACLE scores unmodified raw bytes once."""

    contract = mod.build_prompt_and_decode_contract(corpus)
    job = next(
        row
        for row in mod.build_generation_jobs(corpus, contract)
        if row["task"].get("gold_witness")
    )
    raw = job["task"]["gold_witness"].encode("utf-8")
    row = mod.build_per_unit_row(job, _generation(raw), _process_binding())
    assert base64.b64decode(row["raw_response_bytes_b64"]) == raw
    assert row["parsed_plan"] == raw.decode("utf-8")
    assert row["exact_executor_result"]["valid"] is True
    assert row["exact_success"] is True
    assert row["failure_class"] is None
    assert row["attempt_count"] == 1
    assert row["regeneration_count"] == 0
    assert row["response_regenerated"] is False
    assert row["row_hash"] == mod.row_hash(row)


@pytest.mark.parametrize(
    ("raw", "failure", "finish", "expected"),
    [
        (b"BROKEN", None, "stop", "syntax_failure"),
        (b"PICK(parcel_c00)", None, "stop", "semantic_failure"),
        (b"OPEN(crate_c00)", None, "stop", "unmet_goal"),
        (b"I cannot provide that plan.", None, "stop", "refusal"),
        (b"\xff", None, "stop", "invalid_generation"),
        (b"", "timeout", "timeout", "timeout"),
        (b"", "process_failure", "request_failure", "process_failure"),
        (b"OPEN(crate_c00)", None, "length", "invalid_generation"),
    ],
)
def test_req_report_6607_failures_remain_charged(
    corpus: dict, raw: bytes, failure: str | None, finish: str, expected: str
) -> None:
    """REQ-REPORT-6607-FAILURES keeps every invalid output as one charged row."""

    contract = mod.build_prompt_and_decode_contract(corpus)
    job = mod.build_generation_jobs(corpus, contract)[0]
    row = mod.build_per_unit_row(
        job, _generation(raw, failure=failure, finish=finish), _process_binding()
    )
    assert row["failure_class"] == expected
    assert row["charged_failure"] is True
    assert row["exact_success"] is False
    assert row["attempt_count"] == 1
    assert row["regeneration_count"] == 0


def test_scenario_report_6607_headroom_uses_complete_row_reduction(corpus: dict) -> None:
    """SCENARIO-REPORT-6607-HEADROOM separates completion from headroom."""

    rows, _contract = _rows_with_half_held_success(corpus)
    summary = mod.family_headroom_summary(rows)
    assert summary["held"]["row_count"] == 108
    assert summary["held"]["exact_success_count"] == 54
    assert summary["held"]["exact_success_rate"] == 0.5
    assert summary["held"]["charged_failure_rate"] == 0.5
    assert sum(summary["held"]["failure_counts"].values()) == 54
    low, high = summary["held"]["exact_success_interval_95"]
    assert 0.4 < low < 0.5 < high < 0.6
    assert mod.gemma26_headroom_ready_score(True, 20, 100) == 1.0
    assert mod.gemma26_headroom_ready_score(True, 80, 100) == 1.0
    assert mod.gemma26_headroom_ready_score(True, 19, 100) == 0.0
    assert mod.gemma26_headroom_ready_score(True, 81, 100) == 0.0
    assert mod.gemma26_headroom_ready_score(False, 50, 100) == 0.0


def test_scenario_report_6607_resume_accepts_only_exact_prefix(
    corpus: dict, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-6607-RESUME preserves bytes and rejects drift."""

    rows, contract = _rows_with_half_held_success(corpus)
    checkpoint = tmp_path / "checkpoint.json"
    receipt = mod.atomic_write_checkpoint(
        checkpoint,
        rows[:3],
        prompt_contract_sha256=contract["contract_sha256"],
        exp6604_artifact_sha256=mod.sha256_file(CORPUS_PATH),
        model_sha256="sha256:gemma26-model",
        process_sessions=_gpu_receipts(3)["sessions"],
    )
    resumed = mod.load_checkpoint(
        checkpoint,
        expected_row_ids=[row["row_id"] for row in rows],
        prompt_contract_sha256=contract["contract_sha256"],
        exp6604_artifact_sha256=mod.sha256_file(CORPUS_PATH),
        model_sha256="sha256:gemma26-model",
    )
    assert receipt["atomic_replace"] is True
    assert resumed["accepted"] is True
    assert resumed["completed_row_count"] == 3
    assert resumed["rows"][0]["raw_response_sha256"] == rows[0]["raw_response_sha256"]
    rejected = mod.load_checkpoint(
        checkpoint,
        expected_row_ids=[row["row_id"] for row in rows],
        prompt_contract_sha256="sha256:drift",
        exp6604_artifact_sha256=mod.sha256_file(CORPUS_PATH),
        model_sha256="sha256:gemma26-model",
    )
    assert rejected["accepted"] is False
    assert rejected["gate_check_summary"]["failed_condition"] == "prompt_contract_sha256"


def test_scenario_report_6607_attacks_fail_closed_and_aggregates_agree(corpus: dict) -> None:
    """SCENARIO-REPORT-6607-ATTACKS-AND-ATOMIC rejects all ten mutations."""

    artifact = _complete_artifact(corpus)
    reduction = mod.integrity_reducer(artifact)
    assert reduction["complete"] is True
    assert artifact["gemma26_headroom_ready_score"] == 1.0
    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "null"
    assert set(mod.REQUIRED_ATTACK_IDS) == {
        "cross_family_tuning",
        "prompt_drift",
        "seed_drift",
        "split_leakage",
        "omitted_failures",
        "cpu_fallback",
        "fake_cuda_offload",
        "wrong_model",
        "tokenizer_substitution",
        "response_regeneration",
        "aggregate_disagreement",
        "protected_file_mutation",
    }
    assert {row["attack_id"] for row in artifact["attack_rows"]} == set(mod.REQUIRED_ATTACK_IDS)
    assert all(row["failed_closed"] is True for row in artifact["attack_rows"])
    assert mod.validate_artifact(artifact) == []

    tampered = deepcopy(artifact)
    tampered["family_headroom_summary"]["held"]["exact_success_count"] += 1
    tampered["reproducibility_checksum"] = mod.artifact_checksum(tampered)
    assert "family_headroom_summary_mismatch" in mod.validate_artifact(tampered)


def test_scenario_report_6607_blocked_and_atomic_terminal_artifacts(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6607-BLOCKED and ATOMIC retain the failed value."""

    artifact = mod.build_blocked_artifact(
        run_date="20260825",
        failed_condition="gpu_free_vram_mb",
        expected=22_000,
        observed=2_887,
        model_identity=_model_identity(),
        preconditions={
            "all_required_preconditions_available": False,
            "failed_preconditions": ["gpu_free_vram_mb"],
        },
        protected_files=_protected(),
        tests_run=[{"command": "focused", "exit_code": 0, "duration_s": 0.1}],
        duration_s=1.0,
    )
    assert artifact["status"] == "blocked_gpu_free_vram_mb"
    assert artifact["honest_verdict"].startswith("blocked_gpu_free_vram_mb")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["per_unit_rows"] == []
    assert artifact["gemma26_headroom_ready_score"] == 0.0
    assert artifact["gate_check_summary"]["observed"] == 2_887
    assert mod.validate_artifact(artifact) == []

    target = tmp_path / "artifact.json"
    receipt = mod.atomic_write_artifact(target, artifact)
    assert receipt["atomic_replace"] is True
    assert receipt["directory_fsync"] is True
    assert not list(tmp_path.glob(".exp6607-*"))
    assert json.loads(target.read_text(encoding="utf-8")) == artifact


def test_req_report_6607_validation_rejects_identity_and_checksum_tampering(corpus: dict) -> None:
    """REQ-REPORT-6607-ATTACKS validates identity, rows, and final content hash."""

    artifact = _complete_artifact(corpus)
    wrong_model = deepcopy(artifact)
    wrong_model["model_spec_and_identity"]["hub_id"] = "Qwen/Qwen3.5-0.8B"
    wrong_model["reproducibility_checksum"] = mod.artifact_checksum(wrong_model)
    assert "model_identity_mismatch" in mod.validate_artifact(wrong_model)

    bad_checksum = deepcopy(artifact)
    bad_checksum["tests_run"] = [{"command": "changed", "outcome": "failed"}]
    assert "reproducibility_checksum_mismatch" in mod.validate_artifact(bad_checksum)


def test_req_report_6607_integrity_helpers_fail_closed(corpus: dict, tmp_path: Path) -> None:
    """REQ-REPORT-6607-REDUCER rejects missing GPU and malformed row receipts."""

    assert mod._gpu_receipts_ready({}, 1) is False
    bad_gpu = _gpu_receipts(1)
    bad_gpu["sessions"][0]["cuda_offload"] = False
    assert mod._gpu_receipts_ready(bad_gpu, 1) is False

    contract = mod.build_prompt_and_decode_contract(corpus)
    job = mod.build_generation_jobs(corpus, contract)[0]
    row = mod.build_per_unit_row(job, _generation(b"NOT_A_CANONICAL_PLAN"), _process_binding())
    row["prompt_bytes_b64"] = "%%%"
    assert (
        mod._row_ready(
            row,
            job,
            _model_identity(),
            {"session-1": _gpu_receipts(1)["sessions"][0]},
        )
        is False
    )

    missing = mod.load_checkpoint(
        tmp_path / "missing.json",
        expected_row_ids=[job["row_id"]],
        prompt_contract_sha256=contract["contract_sha256"],
        exp6604_artifact_sha256=mod.sha256_file(CORPUS_PATH),
        model_sha256="sha256:gemma26-model",
    )
    assert missing["accepted"] is True
    assert missing["completed_row_count"] == 0


def test_req_report_6607_gpu_ownership_is_scoped_to_selected_device() -> None:
    """REQ-REPORT-6607-PRECONDITIONS allows only its controller on GPU 0."""

    sample = {
        "device": {"uuid": "GPU-0"},
        "compute_processes": [
            {"gpu_uuid": "GPU-0", "pid": 123, "process_name": "controller"},
            {"gpu_uuid": "GPU-1", "pid": 456, "process_name": "unrelated"},
        ],
    }
    receipt = mod.gpu_ownership_receipt(sample, controller_pid=123)
    assert receipt["available"] is True
    assert [row["pid"] for row in receipt["selected_gpu_processes"]] == [123]
    assert receipt["foreign_selected_gpu_processes"] == []

    sample["compute_processes"].append({"gpu_uuid": "GPU-0", "pid": 789, "process_name": "foreign"})
    blocked = mod.gpu_ownership_receipt(sample, controller_pid=123)
    assert blocked["available"] is False
    assert [row["pid"] for row in blocked["foreign_selected_gpu_processes"]] == [789]


def test_req_report_6607_complete_without_headroom_is_valid(corpus: dict) -> None:
    """REQ-REPORT-6607-HEADROOM reports a complete null when held success is zero."""

    contract = mod.build_prompt_and_decode_contract(corpus)
    jobs = mod.build_generation_jobs(corpus, contract)
    rows = [
        mod.build_per_unit_row(job, _generation(b"NOT_A_CANONICAL_PLAN"), _process_binding())
        for job in jobs
    ]
    artifact = mod.assemble_artifact(
        run_date="20260825",
        exp6604_artifact=corpus,
        prompt_contract=contract,
        per_unit_rows=rows,
        model_identity=_model_identity(),
        gpu_receipts=_gpu_receipts(len(rows)),
        checkpoint_receipts={
            "accepted_row_count": len(rows),
            "completed_prefix_hash": mod.sha256_json(rows),
            "prompt_contract_sha256": contract["contract_sha256"],
            "model_sha256": "sha256:gemma26-model",
            "atomic_replace": True,
            "directory_fsync": True,
        },
        preconditions={
            "all_required_preconditions_available": True,
            "failed_preconditions": [],
        },
        protected_files=_protected(),
        tests_run=[{"command": "focused", "exit_code": 0, "duration_s": 0.1}],
        duration_s=1.0,
    )
    assert artifact["status"] == "complete"
    assert artifact["gemma26_headroom_ready_score"] == 0.0
    assert "outside the frozen headroom interval" in artifact["honest_verdict"]
    assert mod.validate_artifact(artifact) == []


def test_req_report_6607_assemble_blocks_incomplete_rows(corpus: dict) -> None:
    """REQ-REPORT-6607-ATOMIC retains rows while a terminal integrity block fails closed."""

    rows, contract = _rows_with_half_held_success(corpus)
    rows = rows[:-1]
    artifact = mod.assemble_artifact(
        run_date="20260825",
        exp6604_artifact=corpus,
        prompt_contract=contract,
        per_unit_rows=rows,
        model_identity=_model_identity(),
        gpu_receipts=_gpu_receipts(len(rows)),
        checkpoint_receipts={
            "accepted_row_count": len(rows),
            "completed_prefix_hash": mod.sha256_json(rows),
            "prompt_contract_sha256": contract["contract_sha256"],
            "model_sha256": "sha256:gemma26-model",
            "atomic_replace": True,
            "directory_fsync": True,
        },
        preconditions={
            "all_required_preconditions_available": True,
            "failed_preconditions": [],
        },
        protected_files=_protected(),
        tests_run=[{"command": "focused", "exit_code": 0, "duration_s": 0.1}],
        duration_s=1.0,
    )
    assert artifact["status"].startswith("blocked_")
    assert artifact["per_unit_rows"] == rows


def test_req_report_6607_validation_reports_every_schema_class(corpus: dict) -> None:
    """REQ-REPORT-6607-ATOMIC names malformed complete and blocked artifacts."""

    complete = _complete_artifact(corpus)
    missing = deepcopy(complete)
    missing.pop("status")
    assert mod.validate_artifact(missing)[0].startswith("missing_required_fields:")

    wrong_complete = deepcopy(complete)
    wrong_complete["verdict_class"] = "positive"
    wrong_complete["reproducibility_checksum"] = mod.artifact_checksum(wrong_complete)
    assert "complete_baseline_verdict_class_mismatch" in mod.validate_artifact(wrong_complete)

    blocked = mod.build_blocked_artifact(
        run_date="20260825",
        failed_condition="model_path",
        expected="present",
        observed="missing",
        model_identity=_model_identity(),
        preconditions={"all_required_preconditions_available": False},
        protected_files=_protected(),
        tests_run=[],
        duration_s=1.0,
    )
    malformed = deepcopy(blocked)
    malformed.update(
        {
            "inference_substrate": "wrong",
            "verifier_is_oracle": False,
            "verdict_class": "alien",
            "status": "partial",
            "honest_verdict": "not blocked",
            "field_provenance": {},
            "gate_check_summary": {"failed_condition": None},
            "gemma26_headroom_ready_score": 1.0,
        }
    )
    malformed["reproducibility_checksum"] = mod.artifact_checksum(malformed)
    errors = mod.validate_artifact(malformed)
    assert {
        "inference_substrate_mismatch",
        "verifier_is_oracle_mismatch",
        "verdict_class_invalid",
        "field_provenance_mismatch",
        "blocked_status_prefix_missing",
        "blocked_verdict_prefix_missing",
        "blocked_verdict_class_mismatch",
        "blocked_gate_condition_missing",
        "blocked_ready_score_nonzero",
    } <= set(errors)
