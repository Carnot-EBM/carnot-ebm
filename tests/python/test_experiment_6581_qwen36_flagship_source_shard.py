"""Tests for the Exp6581 one-family Qwen source shard."""

from __future__ import annotations

import base64
from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6581_qwen36_flagship_source_shard as mod


def _protocol() -> dict:
    prompt = "Use only source bytes. Return claims."
    source = "Ada is older than Ben."
    units = []
    for index, case_kind in enumerate(("single_hop", "multi_hop", "unsupported", "ambiguity")):
        unit = {
            "unit_id": mod.sha256_json({"index": index, "case_kind": case_kind}),
            "fixture_id": f"fixture-{index}",
            "case_kind": case_kind,
            "split": "held" if index > 1 else "train",
            "exact_source_bytes": source,
            "source_bytes_sha256": mod.sha256_text(source),
            "content_hash": mod.sha256_json({"fixture": index}),
            "row_hash": mod.sha256_json({"unit": index}),
        }
        units.append(unit)
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
            "one_family_task_mapping": {mod.TASK_ID: mod.QWEN_REPOSITORY_ID},
            "family_rows": [
                {
                    "task_id": mod.TASK_ID,
                    "model_family": mod.QWEN_REPOSITORY_ID,
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
    return {
        "repository_id": mod.QWEN_REPOSITORY_ID,
        "trusted_sha256": "sha256:" + "a" * 64,
        "selected_blob_path": "/cache/blobs/" + "a" * 64,
        "admitted": True,
        "rejection_reasons": [],
        "content_metadata": {
            "architecture": mod.QWEN_ARCHITECTURE,
            "quantization": "Q4_K_M",
            "is_language_model": True,
            "tensor_count": 10,
            "tokenizer_metadata": {
                "token_count": 100,
                "chat_template_present": True,
                "model": "qwen2",
            },
            "bounded_read_receipt": {"tensor_payload_bytes_read": 0},
        },
        "provenance": {
            "valid": True,
            "repository_id": mod.QWEN_REPOSITORY_ID,
            "revision": "fixture-revision",
            "snapshot_filename": "qwen.gguf",
            "trusted_sha256": "sha256:" + "a" * 64,
            "trusted_hash_matches_blob_key": True,
            "resolved_blob_path": "/cache/blobs/" + "a" * 64,
            "symlink_target_matches_blob": True,
            "ordered_shards": [{"shard_number": 1, "shard_count": 1, "blob_key": "a" * 64}],
        },
    }


def _process() -> dict:
    command = ["llama-server", "--model", "/cache/blobs/" + "a" * 64, "--n-gpu-layers", "all"]
    return {
        "pid": 4242,
        "parent_pid": 4000,
        "fresh_process": True,
        "os_pid_verified": True,
        "os_parent_pid_verified": True,
        "command": command,
        "os_command": command,
        "command_sha256": mod.sha256_json(command),
        "os_command_sha256": mod.sha256_json(command),
        "command_matches_os": True,
        "selected_blob_path": "/cache/blobs/" + "a" * 64,
        "cuda_visible_devices": "0",
        "selected_gpu": 0,
        "offloaded_layers": 41,
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
                "device": {"memory_used_mb": 8000, "utilization_pct": 80},
                "compute_processes": [{"pid": 4242, "used_memory_mb": 7990}],
            },
            {
                "stage": "during",
                "selected_gpu": 0,
                "device": {"memory_used_mb": 8100, "utilization_pct": 70},
                "compute_processes": [{"pid": 4242, "used_memory_mb": 8090}],
            },
            {
                "stage": "after",
                "selected_gpu": 0,
                "device": {"memory_used_mb": 12, "utilization_pct": 0},
                "compute_processes": [],
            },
        ],
        "resident_model_families": [mod.QWEN_REPOSITORY_ID],
        "signals_sent_to_unrelated_pids": [],
    }


def _unload() -> dict:
    return {
        "worker_pid": 4242,
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


def _raw_rows(
    protocol: dict, tmp_path: Path, *, failure_index: int | None = None
) -> tuple[list[dict], list[dict], list[dict]]:
    rows = []
    checkpoints = []
    diagnostics = []
    prompt_contract = protocol["prompt_seed_budget_contract"]
    for index, unit in enumerate(protocol["source_unit_manifest"]["units"]):
        request = mod.compose_request_bytes(
            prompt_contract["family_neutral_prompt"], unit["exact_source_bytes"]
        )
        response = (
            b""
            if index == failure_index
            else json.dumps(
                {
                    "claim_id": f"claim-{index}",
                    "supported_spans": [unit["exact_source_bytes"][:3]],
                    "unsupported_reason": "",
                    "release_action": "propose",
                }
            ).encode()
        )
        flags = mod.classify_raw_response(
            response,
            timed_out=index == failure_index,
            process_failure=False,
        )
        raw_row = mod.build_raw_terminal_row(
            unit=unit,
            order_index=index,
            protocol=protocol,
            metadata_receipt=_metadata(),
            process_receipt=_process(),
            raw_response_bytes=response,
            raw_api_response_sha256=mod.sha256_bytes(response),
            prompt_tokens=20,
            response_tokens=0 if index == failure_index else 30,
            latency_s=1.0 + index,
            stop_reason="timeout" if index == failure_index else "stop",
            request_exit_code=124 if index == failure_index else 0,
            stderr_sha256_at_terminal=mod.sha256_bytes(f"stderr-{index}".encode()),
            failure_flags=flags,
            raw_response_recorded_monotonic_ns=100 + index * 10,
        )
        checkpoint = mod.write_raw_checkpoint(tmp_path, raw_row)
        diagnostic = mod.build_parser_diagnostic(
            raw_row,
            parser_started_monotonic_ns=101 + index * 10,
        )
        rows.append(mod.finalize_terminal_row(raw_row, checkpoint, diagnostic, _process()))
        checkpoints.append(checkpoint)
        diagnostics.append(diagnostic)
    return rows, checkpoints, diagnostics


def _ready_report(tmp_path: Path, *, failure_index: int | None = None) -> dict:
    protocol = _protocol()
    rows, checkpoints, diagnostics = _raw_rows(protocol, tmp_path, failure_index=failure_index)
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
    negative_rows = [
        {"unit_id": fixture_id, "passed": True} for fixture_id in mod.REQUIRED_NEGATIVE_FIXTURE_IDS
    ]
    return mod.build_report(
        gates=gates,
        protocol=protocol,
        metadata_receipt=_metadata(),
        negative_fixture_rows=negative_rows,
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


def test_spec_declares_exp6581_requirements_and_scenarios() -> None:
    """REQ-REPORT-6581: the executable contract has named spec anchors."""

    text = (mod.REPO_ROOT / mod.SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    for anchor in (
        "REQ-REPORT-6581-GATES",
        "REQ-REPORT-6581-IDENTITY",
        "REQ-REPORT-6581-SOURCE",
        "REQ-REPORT-6581-RAW-FIRST",
        "REQ-REPORT-6581-FAILURES",
        "REQ-REPORT-6581-PROCESS",
        "REQ-REPORT-6581-UNLOAD",
        "REQ-REPORT-6581-ATTACKS",
        "REQ-REPORT-6581-REDUCER",
        "REQ-REPORT-6581-ATOMIC",
        "SCENARIO-REPORT-6581-GATE-BLOCK",
        "SCENARIO-REPORT-6581-RAW-FIRST",
        "SCENARIO-REPORT-6581-UNLOAD",
    ):
        assert anchor in text


def test_hash_and_json_helpers_are_stable(tmp_path: Path) -> None:
    """REQ-REPORT-6581-ATOMIC: hashes use stable bytes and exclude self-fields."""

    assert mod.canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert mod.sha256_text("x") == mod.sha256_bytes(b"x")
    assert mod.sha256_json({"x": 1}).startswith("sha256:")
    assert mod.sha256_file(tmp_path / "missing") == "missing"
    target = tmp_path / "value.bin"
    target.write_bytes(b"value")
    assert mod.sha256_file(target) == mod.sha256_bytes(b"value")
    row = {"value": 1, "row_hash": "old"}
    assert mod.row_hash(row) == mod.row_hash({"value": 1})
    payload = {"value": 1, "reproducibility_checksum": "old"}
    assert mod.artifact_checksum(payload) == mod.artifact_checksum({"value": 1})
    target.write_text("not-json", encoding="utf-8")
    assert mod.load_json(target) == {}
    target.write_text("[]", encoding="utf-8")
    assert mod.load_json(target) == {}


def test_gate_and_protocol_receipts_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6581-GATE-BLOCK: exact fields and protocol hashes bind work."""

    exp6579 = tmp_path / "exp6579.json"
    exp6580 = tmp_path / "exp6580.json"
    exp6579.write_text(json.dumps({"v572_decomposition_contract_ready_score": 1.0}))
    exp6580.write_text(json.dumps(_protocol()))
    gates = mod.build_gate_receipts(
        tmp_path,
        gate_contracts=(
            ("exp6579", Path("exp6579.json"), "v572_decomposition_contract_ready_score"),
            ("exp6580", Path("exp6580.json"), "v572_source_method_ready_score"),
        ),
    )
    assert all(row["passed"] for row in gates)
    assert all(row["sha256"].startswith("sha256:") for row in gates)
    exp6579.write_text("{}")
    failed = mod.build_gate_receipts(
        tmp_path,
        gate_contracts=(
            ("exp6579", Path("exp6579.json"), "v572_decomposition_contract_ready_score"),
        ),
    )
    assert failed[0]["observed_value"] is None
    assert failed[0]["passed"] is False

    protocol = _protocol()
    assert mod.validate_frozen_protocol(protocol) == []
    drift = deepcopy(protocol)
    drift["prompt_seed_budget_contract"]["family_rows"][0]["seed"] = 9
    assert "qwen_family_contract_mismatch" in mod.validate_frozen_protocol(drift)
    missing = deepcopy(protocol)
    missing["source_unit_manifest"]["units"].pop()
    assert "source_manifest_count_mismatch" in mod.validate_frozen_protocol(missing)
    bad_source = deepcopy(protocol)
    bad_source["source_unit_manifest"]["units"][0]["source_bytes_sha256"] = "sha256:bad"
    assert "source_hash_mismatch" in mod.validate_frozen_protocol(bad_source)


def test_metadata_and_bounded_negative_fixtures(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-6581-IDENTITY: Qwen admits and five attacks fail closed."""

    assert mod.metadata_receipt_passes(_metadata()) is True
    wrong = deepcopy(_metadata())
    wrong["content_metadata"]["architecture"] = "gemma4"
    assert mod.metadata_receipt_passes(wrong) is False

    source_rows = [
        {"unit_id": fixture_id, "passed": True, "record": {"admitted": False}}
        for fixture_id in mod.REQUIRED_NEGATIVE_FIXTURE_IDS[:-1]
    ]
    monkeypatch.setattr(mod.gguf_fixtures, "build_negative_fixture_rows", lambda: source_rows)
    monkeypatch.setattr(
        mod,
        "_wrong_architecture_fixture_row",
        lambda: {"unit_id": "wrong_architecture", "passed": True, "record": {"admitted": False}},
    )
    observed = mod.build_negative_metadata_fixture_rows()
    assert [row["unit_id"] for row in observed] == list(mod.REQUIRED_NEGATIVE_FIXTURE_IDS)
    assert all(row["passed"] for row in observed)

    monkeypatch.undo()
    wrong_architecture = mod._wrong_architecture_fixture_row()
    assert wrong_architecture["unit_id"] == "wrong_architecture"
    assert wrong_architecture["passed"] is True


def test_segmentation_and_failure_taxonomy_are_bounded() -> None:
    """REQ-REPORT-6581-FAILURES: parsing is diagnostic and preserves failures."""

    response = b"Claim one. Claim two! Claim three?"
    assert mod.segment_claim_sentences(response, max_segments=2) == ["Claim one.", "Claim two!"]
    assert mod.classify_raw_response(b"", timed_out=True, process_failure=False) == {
        "timeout": True,
        "malformed_output": False,
        "refusal": False,
        "empty_output": True,
        "no_claim": True,
        "process_failure": False,
    }
    malformed = mod.classify_raw_response(b"\xff", timed_out=False, process_failure=True)
    assert malformed["malformed_output"] is True
    assert malformed["process_failure"] is True
    refusal = mod.classify_raw_response(b"I cannot comply with this request.")
    assert refusal["refusal"] is True
    assert refusal["no_claim"] is True
    assert mod.segment_claim_sentences(b"\xff") == []
    assert mod.segment_claim_sentences(b"claim", max_segments=0) == []


def test_invalid_raw_receipts_fail_closed() -> None:
    """REQ-REPORT-6581-RAW-FIRST: malformed receipt encoding cannot become evidence."""

    assert mod._decode_b64("not base64!") is None
    diagnostic = mod.build_parser_diagnostic(
        {
            "unit_id": "invalid",
            "raw_response_bytes_b64": "not base64!",
            "failure_flags": {"malformed_output": True},
            "raw_response_recorded_monotonic_ns": 1,
        },
        parser_started_monotonic_ns=2,
    )
    assert diagnostic["claim_bearing"] is False
    assert diagnostic["segment_count"] == 0


def test_raw_checkpoint_precedes_parser_and_is_content_addressed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6581-RAW-FIRST: raw bytes land before parser diagnostics."""

    protocol = _protocol()
    rows, checkpoints, diagnostics = _raw_rows(protocol, tmp_path)
    assert len(rows) == len(checkpoints) == len(diagnostics) == 4
    for row, checkpoint, diagnostic in zip(rows, checkpoints, diagnostics, strict=True):
        assert checkpoint["atomic_replace"] is True
        assert checkpoint["checkpoint_sha256"] == mod.sha256_file(checkpoint["absolute_path"])
        assert checkpoint["raw_row_hash"] == row["raw_checkpoint_row_hash"]
        assert row["raw_response_recorded_monotonic_ns"] < diagnostic["parser_started_monotonic_ns"]
        assert diagnostic["diagnostic_only"] is True
        assert "claim_sentences" not in json.loads(Path(checkpoint["absolute_path"]).read_text())


def test_ready_report_recomputes_every_required_field(tmp_path: Path) -> None:
    """REQ-REPORT-6581-REDUCER: readiness derives only from authentic emitted rows."""

    report = _ready_report(tmp_path)
    assert report["qwen36_family_source_shard_ready_score"] == 1.0
    assert report["verdict_class"] is None
    assert report["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert report["verifier_is_oracle"] is False
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(report)
    assert set(report["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert report["aggregate_row_recomputation"]["expected_unit_count"] == 4
    assert report["aggregate_row_recomputation"]["claim_bearing_row_count"] == 4
    assert report["aggregate_row_recomputation"]["all_costs_recomputed"] is True
    assert mod.validate_report(report) == []


def test_failures_remain_visible_without_forcing_row_loss(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6581-FAILURES: a terminal timeout remains one authentic row."""

    report = _ready_report(tmp_path, failure_index=1)
    assert len(report["rows"]) == 4
    assert report["aggregate_row_recomputation"]["failure_row_count"] == 1
    assert report["aggregate_row_recomputation"]["failure_class_counts"]["timeout"] == 1
    assert report["rows"][1]["failure_flags"]["timeout"] is True
    assert report["rows"][1]["attempt_count"] == 1
    assert report["rows"][1]["retry_count"] == 0
    assert report["qwen36_family_source_shard_ready_score"] == 1.0


def test_process_unload_and_attack_checks_fail_closed() -> None:
    """SCENARIO-REPORT-6581-UNLOAD: CUDA lifecycle and attacks are rechecked."""

    assert all(mod.process_and_gpu_checks(_process()).values())
    assert all(mod.unload_checks(_unload()).values())
    failed_unload = deepcopy(_unload())
    failed_unload["memory_delta_from_baseline_mb"] = mod.RECOVERY_TOLERANCE_MB + 1
    assert mod.unload_checks(failed_unload)["memory_recovered"] is False
    attacks = mod.build_attack_rows()
    assert {row["attack_id"] for row in attacks} == set(mod.REQUIRED_ATTACK_IDS)
    assert all(row["candidate_ready_score"] == 0.0 and row["passed"] for row in attacks)


def test_tampering_and_incomplete_manifest_change_readiness(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6581-ATTACKS: drift, retry, bytes, and row loss fail closed."""

    report = _ready_report(tmp_path)
    mutations = []
    for mutate in (
        lambda value: value["rows"][0].update(prompt_sha256="sha256:drift"),
        lambda value: value["rows"][0].update(retry_count=1),
        lambda value: value["rows"][0].update(raw_response_bytes_b64=None),
        lambda value: value["rows"].pop(),
        lambda value: value["process_and_gpu_receipts"].update(offloaded_layers=0),
        lambda value: value["unload_and_recovery_rows"][0].update(recovery_complete=False),
    ):
        changed = deepcopy(report)
        mutate(changed)
        mutations.append(changed)
    for changed in mutations:
        aggregate = mod.recompute_aggregate(changed)
        assert aggregate["ready_score"] == 0.0

    checksum = deepcopy(report)
    checksum["status"] = "changed"
    assert "reproducibility_checksum_mismatch" in mod.validate_report(checksum)
    score = deepcopy(report)
    score["qwen36_family_source_shard_ready_score"] = 0.0
    score["reproducibility_checksum"] = mod.artifact_checksum(score)
    assert "ready_score_mismatch" in mod.validate_report(score)


def test_blocked_report_names_exact_gate_and_starts_no_model() -> None:
    """SCENARIO-REPORT-6581-GATE-BLOCK: blocked output names exact observed value."""

    gates = [
        {
            "upstream": "exp6579",
            "path": "results/exp6579.json",
            "field": "v572_decomposition_contract_ready_score",
            "expected_value": 1.0,
            "observed_value": None,
            "passed": False,
        }
    ]
    report = mod.build_blocked_report(
        gates=gates,
        protocol={},
        preconditions={"model_process_started": False},
        protected={"all_unchanged": True, "rows": []},
        duration_s=0.1,
        tests_run=[],
        reason="structured_gate_failed",
    )
    assert report["honest_verdict"].startswith("blocked_")
    assert report["gate_check_summary"]["first_failure"]["field"] == (
        "v572_decomposition_contract_ready_score"
    )
    assert report["gate_check_summary"]["first_failure"]["observed_value"] is None
    assert report["qwen36_family_source_shard_ready_score"] == 0.0
    assert report["rows"] == []
    assert mod.validate_report(report) == []


def test_partial_disqualified_and_validator_branches_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-6581-REDUCER: every non-ready verdict and schema drift closes."""

    ready = _ready_report(tmp_path)
    kwargs = {
        "gates": ready["gate_check_summary"]["rows"],
        "protocol": ready["source_protocol"],
        "metadata_receipt": ready["model_revision_and_hash_receipt"],
        "negative_fixture_rows": ready["negative_metadata_fixture_rows"],
        "rows": [],
        "checkpoint_receipts": [],
        "parser_diagnostic_rows": [],
        "process_receipt": ready["process_and_gpu_receipts"],
        "unload_rows": ready["unload_and_recovery_rows"],
        "attack_rows": ready["attack_rows"],
        "preconditions": ready["preconditions_checked"],
        "protected": ready["protected_files_unchanged"],
        "duration_s": 1.0,
        "tests_run": [],
        "run_date": "20260824",
    }
    partial = mod.build_report(**kwargs)
    assert partial["verdict_class"] == "partial"
    disqualified = mod.build_report(**{**kwargs, "protected": {"all_unchanged": False, "rows": []}})
    assert disqualified["verdict_class"] == "disqualified"

    cases = (
        ("inference_substrate", "bad", "inference_substrate_mismatch"),
        ("verifier_is_oracle", True, "verifier_is_oracle_mismatch"),
        ("verdict_class", "positive", "verdict_class_invalid"),
        ("model_specs", [], "model_specs_mismatch"),
        ("field_provenance", {}, "field_provenance_mismatch"),
    )
    for key, value, expected in cases:
        changed = deepcopy(ready)
        changed[key] = value
        changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
        assert expected in mod.validate_report(changed)

    aggregate = deepcopy(ready)
    aggregate["aggregate_row_recomputation"]["ready_score"] = 0.0
    aggregate["reproducibility_checksum"] = mod.artifact_checksum(aggregate)
    assert "aggregate_ready_score_mismatch" in mod.validate_report(aggregate)
    null_incomplete = deepcopy(partial)
    null_incomplete["verdict_class"] = None
    null_incomplete["reproducibility_checksum"] = mod.artifact_checksum(null_incomplete)
    assert "null_verdict_without_ready_shard" in mod.validate_report(null_incomplete)
    blocked_rows = deepcopy(ready)
    blocked_rows["verdict_class"] = "blocked"
    blocked_rows["reproducibility_checksum"] = mod.artifact_checksum(blocked_rows)
    assert "blocked_report_started_rows" in mod.validate_report(blocked_rows)


def test_raw_receipt_recovery_and_cost_recompute(tmp_path: Path) -> None:
    """REQ-REPORT-6581-RAW-FIRST: bytes and charged components replay losslessly."""

    report = _ready_report(tmp_path)
    for row, receipt in zip(report["rows"], report["raw_response_receipts"], strict=True):
        raw = base64.b64decode(row["raw_response_bytes_b64"], validate=True)
        assert mod.sha256_bytes(raw) == row["raw_response_sha256"]
        assert receipt["raw_response_sha256"] == row["raw_response_sha256"]
        assert mod.cost_from_components(row["charged_cost_components"]) == row["charged_cost"]


def test_atomic_write_validation_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-REPORT-6581-ATOMIC: the writer validates before same-directory replace."""

    report = _ready_report(tmp_path / "checkpoints")
    output = tmp_path / "artifact.json"
    receipt = mod.atomic_write_report(output, report)
    assert receipt["atomic_replace"] is True
    assert mod.load_json(output) == report
    assert mod.main(["--validate", "--output", str(output)]) == 0
    assert "valid" in capsys.readouterr().out
    bad = deepcopy(report)
    bad.pop("rows")
    with pytest.raises(ValueError, match="missing_required_fields"):
        mod.atomic_write_report(output, bad)
    output.write_text("{}", encoding="utf-8")
    assert mod.main(["--validate", "--output", str(output)]) == 1

    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "run_experiment", lambda root, date: report)
    assert mod.main(["--date", "20260824"]) == 0
    assert mod.RESULT_RELATIVE_PATH.name in capsys.readouterr().out
