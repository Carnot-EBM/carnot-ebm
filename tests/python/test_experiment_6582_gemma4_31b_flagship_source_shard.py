"""Tests for the Exp6582 dense Gemma one-family source shard."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6581_qwen36_flagship_source_shard as shared
from carnot import experiment_6582_gemma4_31b_flagship_source_shard as mod


def _protocol() -> dict:
    """Build the four-unit frozen contract with the Exp6582 family mapping."""

    prompt = "Use only source bytes. Return claims."
    units = []
    for index, case_kind in enumerate(("single_hop", "multi_hop", "unsupported", "ambiguity")):
        source = f"Source unit {index} states one bounded fact."
        units.append(
            {
                "unit_id": mod.sha256_json({"index": index, "case_kind": case_kind}),
                "fixture_id": f"fixture-{index}",
                "case_kind": case_kind,
                "split": "held" if index > 1 else "train",
                "exact_source_bytes": source,
                "source_bytes_sha256": mod.sha256_text(source),
                "content_hash": mod.sha256_json({"fixture": index}),
                "row_hash": mod.sha256_json({"unit": index}),
            }
        )
    manifest_body = {
        "schema": "carnot.v572.source_unit_manifest.v1",
        "bounded_unit_count": len(units),
        "max_units": len(units),
        "selected_without_model_outcomes": True,
        "required_case_kinds": ["single_hop", "multi_hop", "unsupported", "ambiguity"],
        "split_names": ["train", "calibration", "held"],
        "units": units,
    }
    budget = {
        "max_prompt_tokens": 4096,
        "max_output_tokens": 512,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    return {
        "v572_source_method_ready_score": 1.0,
        "source_unit_manifest": {
            **manifest_body,
            "manifest_hash": mod.sha256_json(manifest_body),
        },
        "prompt_seed_budget_contract": {
            "family_neutral_prompt": prompt,
            "prompt_sha256": mod.sha256_text(prompt),
            "token_budget": budget,
            "stop_rules": ["<|eot_id|>", "<stop>"],
            "timeout_s": 4200,
            "raw_before_derived_write_order": True,
            "failure_retention_required": True,
            "fresh_process_per_family": True,
            "one_family_task_mapping": {mod.TASK_ID: mod.GEMMA_REPOSITORY_ID},
            "family_rows": [
                {
                    "task_id": mod.TASK_ID,
                    "model_family": mod.GEMMA_REPOSITORY_ID,
                    "prompt_sha256": mod.sha256_text(prompt),
                    "token_budget_hash": mod.sha256_json(budget),
                    "seed": mod.RANDOM_SEED,
                    "family_specific_prompt_allowed": False,
                    "task_timeout_s": 4200,
                    "per_source_unit_timeout_s": 720,
                }
            ],
        },
    }


def _metadata() -> dict:
    """Build a content-derived dense Gemma identity receipt."""

    trusted = "sha256:" + "b" * 64
    blob = "/cache/blobs/" + "b" * 64
    return {
        "repository_id": mod.GEMMA_REPOSITORY_ID,
        "trusted_sha256": trusted,
        "selected_blob_path": blob,
        "admitted": True,
        "rejection_reasons": [],
        "content_metadata": {
            "architecture": mod.GEMMA_ARCHITECTURE,
            "quantization": "Q4_K_M",
            "is_language_model": True,
            "tensor_count": 833,
            "tokenizer_metadata": {
                "token_count": 262144,
                "chat_template_present": True,
                "model": "gemma4",
            },
            "bounded_read_receipt": {"tensor_payload_bytes_read": 0},
        },
        "provenance": {
            "valid": True,
            "repository_id": mod.GEMMA_REPOSITORY_ID,
            "revision": "fixture-revision",
            "snapshot_filename": "gemma-4-31B-it-Q4_K_M.gguf",
            "trusted_sha256": trusted,
            "trusted_hash_matches_blob_key": True,
            "resolved_blob_path": blob,
            "symlink_target_matches_blob": True,
            "ordered_shards": [{"shard_number": 1, "shard_count": 1, "blob_key": "b" * 64}],
        },
    }


def _process() -> dict:
    """Build one measured fresh-process and CUDA receipt."""

    blob = _metadata()["selected_blob_path"]
    command = ["llama-server", "--model", blob, "--n-gpu-layers", "all"]
    return {
        "pid": 6582,
        "parent_pid": 6500,
        "fresh_process": True,
        "os_pid_verified": True,
        "os_parent_pid_verified": True,
        "command": command,
        "os_command": command,
        "command_sha256": mod.sha256_json(command),
        "os_command_sha256": mod.sha256_json(command),
        "command_matches_os": True,
        "selected_blob_path": blob,
        "gguf_sha256": _metadata()["trusted_sha256"],
        "cuda_visible_devices": "0",
        "selected_gpu": 0,
        "offloaded_layers": 63,
        "server_healthy": True,
        "http_status": 200,
        "started_monotonic_ns": 10,
        "ended_monotonic_ns": 1000,
        "shutdown_requested": True,
        "exit_code": 0,
        "normal_shutdown": True,
        "worker_alive_after_exit": False,
        "stdout_sha256": mod.sha256_bytes(b"stdout"),
        "stderr_sha256": mod.sha256_bytes(b"stderr"),
        "evidence_mode": "measured",
        "gpu_samples": [
            {
                "stage": "before",
                "selected_gpu": 0,
                "device": {"memory_used_mb": 10, "utilization_pct": 0},
                "compute_processes": [],
            },
            {
                "stage": "during",
                "selected_gpu": 0,
                "device": {"memory_used_mb": 19000, "utilization_pct": 80},
                "compute_processes": [{"pid": 6582, "used_memory_mb": 18990}],
            },
            {
                "stage": "during",
                "selected_gpu": 0,
                "device": {"memory_used_mb": 19100, "utilization_pct": 70},
                "compute_processes": [{"pid": 6582, "used_memory_mb": 19090}],
            },
            {
                "stage": "after",
                "selected_gpu": 0,
                "device": {"memory_used_mb": 12, "utilization_pct": 0},
                "compute_processes": [],
            },
        ],
        "resident_model_families": [mod.GEMMA_REPOSITORY_ID],
        "signals_sent_to_unrelated_pids": [],
    }


def _unload() -> dict:
    """Build one clean unload and bounded memory-recovery row."""

    return {
        "worker_pid": 6582,
        "shutdown_requested": True,
        "normal_shutdown": True,
        "exit_code": 0,
        "worker_absent_from_proc": True,
        "worker_absent_from_nvidia_smi": True,
        "port_closed": True,
        "baseline_memory_used_mb": 10,
        "recovered_memory_used_mb": 12,
        "memory_delta_from_baseline_mb": 2,
        "recovery_tolerance_mb": mod.RECOVERY_TOLERANCE_MB,
        "no_task_worker_remains": True,
        "recovery_bounded": True,
        "recovery_complete": True,
        "signals_sent_to_unrelated_pids": [],
    }


def _ready_report(tmp_path: Path, *, failed_index: int | None = None) -> dict:
    """Create a complete synthetic shard through the public row lifecycle."""

    protocol = _protocol()
    rows = []
    checkpoints = []
    diagnostics = []
    prior_response = b""
    for index, unit in enumerate(protocol["source_unit_manifest"]["units"]):
        response = b"" if index == failed_index else f"Claim {index} is supported.".encode()
        request = mod.compose_request_bytes(
            protocol["prompt_seed_budget_contract"]["family_neutral_prompt"],
            unit["exact_source_bytes"],
        )
        assert prior_response not in request or not prior_response
        flags = mod.classify_raw_response(response, timed_out=index == failed_index)
        raw_row = mod.build_raw_terminal_row(
            unit=unit,
            order_index=index,
            protocol=protocol,
            metadata_receipt=_metadata(),
            process_receipt=_process(),
            raw_response_bytes=response,
            raw_api_response_sha256=mod.sha256_bytes(response),
            prompt_tokens=20,
            response_tokens=0 if index == failed_index else 8,
            latency_s=1.0 + index,
            stop_reason="timeout" if index == failed_index else "stop",
            request_exit_code=124 if index == failed_index else 0,
            stderr_sha256_at_terminal=mod.sha256_bytes(f"stderr-{index}".encode()),
            failure_flags=flags,
            raw_response_recorded_monotonic_ns=100 + index * 10,
        )
        checkpoint = mod.write_raw_checkpoint(tmp_path, raw_row)
        diagnostic = mod.build_parser_diagnostic(
            raw_row, parser_started_monotonic_ns=101 + index * 10
        )
        rows.append(mod.finalize_terminal_row(raw_row, checkpoint, diagnostic, _process()))
        checkpoints.append(checkpoint)
        diagnostics.append(diagnostic)
        prior_response = response
    gates = [
        {
            "upstream": "exp6579",
            "path": "results/exp6579.json",
            "field": "v572_decomposition_contract_ready_score",
            "expected_value": 1.0,
            "observed_value": 1.0,
            "passed": True,
        },
        {
            "upstream": "exp6580",
            "path": "results/exp6580.json",
            "field": "v572_source_method_ready_score",
            "expected_value": 1.0,
            "observed_value": 1.0,
            "passed": True,
        },
    ]
    return mod.build_report(
        gates=gates,
        protocol=protocol,
        metadata_receipt=_metadata(),
        negative_fixture_rows=[
            {"unit_id": fixture_id, "passed": True}
            for fixture_id in mod.REQUIRED_NEGATIVE_FIXTURE_IDS
        ],
        rows=rows,
        checkpoint_receipts=checkpoints,
        parser_diagnostic_rows=diagnostics,
        process_receipt=_process(),
        unload_rows=[_unload()],
        attack_rows=mod.build_attack_rows(),
        preconditions={"all_required_preconditions_available": True, "checks": {}},
        protected={"all_unchanged": True, "rows": []},
        duration_s=65.0,
        tests_run=[{"command": "focused", "exit_code": 0, "duration_s": 1.0}],
        run_date="20260824",
    )


def test_spec_declares_exp6582_requirements_and_scenarios() -> None:
    """REQ-REPORT-6582: the dense family has executable spec anchors."""

    text = (mod.REPO_ROOT / mod.SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    for anchor in (
        "REQ-REPORT-6582-GATES",
        "REQ-REPORT-6582-IDENTITY",
        "REQ-REPORT-6582-SOURCE",
        "REQ-REPORT-6582-RAW-FIRST",
        "REQ-REPORT-6582-FRESH-CONTEXT",
        "REQ-REPORT-6582-FAILURES",
        "REQ-REPORT-6582-PROCESS",
        "REQ-REPORT-6582-UNLOAD",
        "REQ-REPORT-6582-ATTACKS",
        "REQ-REPORT-6582-REDUCER",
        "REQ-REPORT-6582-ATOMIC",
        "SCENARIO-REPORT-6582-GATE-BLOCK",
        "SCENARIO-REPORT-6582-RAW-FIRST",
        "SCENARIO-REPORT-6582-FRESH-CONTEXT",
        "SCENARIO-REPORT-6582-UNLOAD",
    ):
        assert anchor in text


def test_family_configuration_binds_gemma_and_restores_shared_state() -> None:
    """REQ-REPORT-6582-IDENTITY: family wrappers cannot leak into Exp6581."""

    original = (shared.TASK_ID, shared.QWEN_REPOSITORY_ID, shared.QWEN_ARCHITECTURE)
    assert mod.validate_frozen_protocol(_protocol()) == []
    assert mod.metadata_receipt_passes(_metadata()) is True
    assert (shared.TASK_ID, shared.QWEN_REPOSITORY_ID, shared.QWEN_ARCHITECTURE) == original
    drift = deepcopy(_protocol())
    drift["prompt_seed_budget_contract"]["family_rows"][0]["seed"] = 1
    assert "gemma4_31b_family_contract_mismatch" in mod.validate_frozen_protocol(drift)
    wrong = deepcopy(_metadata())
    wrong["content_metadata"]["architecture"] = "qwen35moe"
    assert mod.metadata_receipt_passes(wrong) is False
    with mod._family_configuration():
        pair = shared.cached_sota_pair()
    assert pair is not None
    assert [row["hf_id"] for row in pair] == [
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        mod.GEMMA_REPOSITORY_ID,
    ]


def test_metadata_negative_fixtures_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-6582-IDENTITY: all five content attacks fail closed."""

    source_rows = [
        {"unit_id": fixture_id, "passed": True, "record": {"admitted": False}}
        for fixture_id in mod.REQUIRED_NEGATIVE_FIXTURE_IDS[:-1]
    ]
    monkeypatch.setattr(shared.gguf_fixtures, "build_negative_fixture_rows", lambda: source_rows)
    observed = mod.build_negative_metadata_fixture_rows()
    assert [row["unit_id"] for row in observed] == list(mod.REQUIRED_NEGATIVE_FIXTURE_IDS)
    assert all(row["passed"] for row in observed)


def test_ready_report_preserves_failure_rows_and_recomputes(tmp_path: Path) -> None:
    """REQ-REPORT-6582-REDUCER: retained raw rows alone derive readiness."""

    report = _ready_report(tmp_path, failed_index=1)
    assert report[mod.READINESS_FIELD] == 1.0
    assert report["verdict_class"] is None
    assert report["model_specs"] == [
        {
            "repository_id": mod.GEMMA_REPOSITORY_ID,
            "expected_architecture": mod.GEMMA_ARCHITECTURE,
        }
    ]
    assert report["aggregate_row_recomputation"]["failure_row_count"] == 1
    assert report["rows"][1]["failure_flags"]["timeout"] is True
    assert report["rows"][1]["attempt_count"] == 1
    assert report["rows"][1]["retry_count"] == 0
    assert set(report["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert report["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert report["verifier_is_oracle"] is False
    assert all(mod.process_and_gpu_checks(_process()).values())
    assert mod.validate_report(report) == []

    partial = mod._normalize_report(
        {"qwen36_family_source_shard_ready_score": 0.0, "status": "partial"}
    )
    assert partial["honest_verdict"].startswith("partial_gemma4_31b_runtime")


def test_attacks_incomplete_rows_and_gate_blocks_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6582-ATTACKS: substitutions and row loss cannot pass."""

    report = _ready_report(tmp_path)
    assert {row["attack_id"] for row in report["attack_rows"]} == set(mod.REQUIRED_ATTACK_IDS)
    missing = deepcopy(report)
    missing["rows"].pop()
    assert mod.recompute_aggregate(missing)["ready_score"] == 0.0
    substituted = deepcopy(report)
    substituted["rows"][0]["repository_id"] = "google/gemma-4-E4B-it"
    assert mod.recompute_aggregate(substituted)["ready_score"] == 0.0

    gates = deepcopy(report["gate_check_summary"]["rows"])
    gates[0]["observed_value"] = 0.0
    gates[0]["passed"] = False
    blocked = mod.build_blocked_report(
        gates=gates,
        protocol=_protocol(),
        metadata_receipt=_metadata(),
        negative_fixture_rows=[
            {"unit_id": fixture_id, "passed": True}
            for fixture_id in mod.REQUIRED_NEGATIVE_FIXTURE_IDS
        ],
        preconditions={
            "all_required_preconditions_available": False,
            "model_process_started": False,
        },
        protected={"all_unchanged": True, "rows": []},
        duration_s=1.0,
        tests_run=[],
        reason="structured_gate_failed",
        run_date="20260824",
    )
    assert blocked["status"] == "blocked"
    assert blocked["gate_check_summary"]["first_failure"]["observed_value"] == 0.0
    assert blocked[mod.READINESS_FIELD] == 0.0
    assert blocked["rows"] == []
    assert blocked["model_revision_and_hash_receipt"] == _metadata()
    assert len(blocked["negative_metadata_fixture_rows"]) == 5
    assert mod.validate_report(blocked) == []


def test_atomic_writer_and_cli_validate_family_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-REPORT-6582-ATOMIC: one same-directory artifact validates."""

    report = _ready_report(tmp_path / "raw")
    output = tmp_path / "artifact.json"
    receipt = mod.atomic_write_report(output, report)
    assert receipt["atomic_replace"] is True
    assert mod.load_json(output) == report
    assert mod.main(["--validate", "--output", str(output)]) == 0
    assert '"valid": true' in capsys.readouterr().out
    bad = deepcopy(report)
    bad.pop(mod.READINESS_FIELD)
    with pytest.raises(ValueError, match="missing_required_fields"):
        mod.atomic_write_report(output, bad)

    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "run_experiment", lambda root, date: report)
    assert mod.main(["--date", "20260824"]) == 0
    cli = json.loads(capsys.readouterr().out)
    assert cli[mod.READINESS_FIELD] == 1.0
    assert mod.RESULT_RELATIVE_PATH.name in cli["artifact"]


def test_validator_names_every_terminal_mutation(tmp_path: Path) -> None:
    """REQ-REPORT-6582-ATOMIC: malformed terminal fields fail with exact errors."""

    report = _ready_report(tmp_path)
    bad = deepcopy(report)
    bad["inference_substrate"] = "replay"
    bad["verifier_is_oracle"] = True
    bad["verdict_class"] = "positive"
    bad["model_specs"] = []
    bad["field_provenance"] = {}
    bad["reproducibility_checksum"] = "sha256:bad"
    assert set(mod.validate_report(bad)) == {
        "inference_substrate_mismatch",
        "verifier_is_oracle_mismatch",
        "verdict_class_invalid",
        "model_specs_mismatch",
        "field_provenance_mismatch",
        "ready_score_mismatch",
        "aggregate_ready_score_mismatch",
        "reproducibility_checksum_mismatch",
    }

    bad_scores = deepcopy(report)
    bad_scores[mod.READINESS_FIELD] = 0.0
    bad_scores["aggregate_row_recomputation"]["ready_score"] = 0.0
    score_errors = mod.validate_report(bad_scores)
    assert "ready_score_mismatch" in score_errors
    assert "aggregate_ready_score_mismatch" in score_errors

    null_incomplete = deepcopy(report)
    null_incomplete["rows"].pop()
    assert "null_verdict_without_ready_shard" in mod.validate_report(null_incomplete)
    blocked_with_rows = deepcopy(report)
    blocked_with_rows["verdict_class"] = "blocked"
    assert "blocked_report_started_rows" in mod.validate_report(blocked_with_rows)


def test_precondition_key_normalization_is_family_specific() -> None:
    """REQ-REPORT-6582-GATES: dense-family gate names cannot claim Qwen work."""

    preconditions = {
        "checks": {
            "positive_qwen_metadata": True,
            "cached_sota_pair_contains_qwen": True,
            "fresh_qwen_process": True,
        },
        "failed_preconditions": ["positive_qwen_metadata"],
    }
    normalized = mod.normalize_preconditions(preconditions)
    assert normalized["checks"] == {
        "positive_gemma4_31b_metadata": True,
        "cached_sota_pair_contains_gemma4_31b": True,
        "fresh_gemma4_31b_process": True,
    }
    assert normalized["failed_preconditions"] == ["positive_gemma4_31b_metadata"]
    assert "qwen" not in json.dumps(normalized).lower()


def test_verification_is_bounded_without_spending_family_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-6582-VERIFICATION-BUDGET: tests do not shorten model work."""

    calls: list[tuple[str, Path, float]] = []

    def fake_run(command: str, repo_root: Path, timeout_s: float) -> dict:
        calls.append((command, repo_root, timeout_s))
        return {"command": command, "exit_code": 0, "duration_s": 0.0}

    root = Path("/bounded-exp6582-fixture")
    monkeypatch.setattr(shared, "_run_named_test", fake_run)
    receipts = mod._checkpoint_tests(root)
    full_suite = [call for call in calls if call[0] == mod.FULL_PYTEST_COMMAND]

    assert len(receipts) == 7
    assert len(full_suite) == 1
    assert full_suite[0][1] == root
    assert full_suite[0][2] >= 4801.0
    assert mod.family_task_deadline(_protocol(), now=1000.0) == 5200.0
