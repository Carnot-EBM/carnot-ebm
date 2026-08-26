"""Contract tests for the three independent local GGUF canaries.

Spec refs: REQ-INFER-SOTA-6648,
SCENARIO-INFER-SOTA-6648-ALL-FAMILIES,
SCENARIO-INFER-SOTA-6648-NO-SUBSTITUTION, REQ-INFRA-6648,
SCENARIO-INFRA-6648-INDEPENDENT-PROCESSES,
SCENARIO-INFRA-6648-LIFECYCLE-BLOCK, REQ-REPORT-6648,
SCENARIO-REPORT-6648-READY, SCENARIO-REPORT-6648-BLOCKED, and
SCENARIO-REPORT-6648-ATTACKS-AND-ATOMIC.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6648_three_family_gguf_canaries as exp


REPO = Path(__file__).resolve().parents[2]


def _passing_row(spec: dict, pid: int, started_ns: int) -> dict:
    device_uuid = f"GPU-test-{spec['device_index']}"
    model_pid = pid + 100
    row = {
        "row_kind": "mandated_model_family",
        "family_id": spec["family_id"],
        "role": spec["role"],
        "hf_id": spec["hf_id"],
        "quantization": spec["quantization"],
        "device_index": spec["device_index"],
        "device_uuid": device_uuid,
        "resolution_method": spec["resolution_method"],
        "model_path": f"/cache/{spec['family_id']}.gguf",
        "model_sha256": f"sha256:{pid:064x}",
        "tokenizer": {
            "source": "llama.cpp_embedded_gguf",
            "loadable": True,
            "auto_tokenizer_used": False,
            "prompt_token_count": 7,
            "detail": "embedded tokenizer loaded",
        },
        "worker_process": {
            "pid": pid,
            "pid_start_ticks": pid * 10,
            "executable": "/venv/python",
            "argv": ["/venv/python", "-m", exp.MODULE_NAME, "--worker"],
            "argv_sha256": exp.sha256_json(["/venv/python", "-m", exp.MODULE_NAME, "--worker"]),
            "started_monotonic_ns": started_ns,
            "ended_monotonic_ns": started_ns + 90,
            "exit_code": 0,
            "absent_after_exit": True,
        },
        "model_process": {
            "pid": model_pid,
            "pid_start_ticks": model_pid * 10,
            "parent_pid": pid,
            "executable": "/cache/llama-server",
            "argv": ["/cache/llama-server", "--model", f"/cache/{spec['family_id']}.gguf"],
            "argv_sha256": exp.sha256_json(
                ["/cache/llama-server", "--model", f"/cache/{spec['family_id']}.gguf"]
            ),
            "started_monotonic_ns": started_ns + 10,
            "ended_monotonic_ns": started_ns + 80,
            "exit_code": -15,
            "absent_after_exit": True,
        },
        "lease": {
            "owner": {
                "pid": pid,
                "pid_start_ticks": pid * 10,
                "device_uuid": device_uuid,
                "expected_model": f"/cache/{spec['family_id']}.gguf",
                "token_opaque": True,
            },
            "journal_path": f"/run/{spec['family_id']}.json",
            "journal_checksum": f"sha256:{(pid + 1):064x}",
            "phase_sequence": list(exp.COMPLETE_PHASE_SEQUENCE),
            "phase_history": [
                {"phase": phase, "event_checksum": f"sha256:{index + pid:064x}"}
                for index, phase in enumerate(exp.COMPLETE_PHASE_SEQUENCE)
            ],
            "release": {
                "released": True,
                "phase": "terminal_complete",
                "device_uuid": device_uuid,
                "pid": pid,
                "pid_start_ticks": pid * 10,
            },
            "owner_bound": True,
        },
        "accelerator": {
            "before": {
                "device_uuid": device_uuid,
                "memory_used_mb": 100,
                "model_pid_present": False,
            },
            "resident": {
                "device_uuid": device_uuid,
                "memory_used_mb": 12000,
                "model_pid_present": True,
            },
            "after": {
                "device_uuid": device_uuid,
                "memory_used_mb": 110,
                "model_pid_present": False,
            },
            "cuda_offload": True,
            "resident_vram_delta_mb": 11900,
        },
        "prompt": {
            "sha256": exp.sha256_text(exp.FIXED_PROMPT),
            "random_seed": exp.RANDOM_SEED,
        },
        "output": {
            "text": "READY",
            "sha256": exp.sha256_text("READY"),
            "non_empty": True,
            "prompt_token_count": 7,
            "output_token_count": 1,
            "http_status": 200,
            "finish_reason": "stop",
        },
        "unload": {
            "observed": True,
            "model_process_absent": True,
            "vram_recovered": True,
        },
        "errors": [],
        "admitted": True,
        "failed_checks": [],
    }
    row["row_sha256"] = exp.family_row_hash(row)
    return row


def _passing_rows() -> list[dict]:
    return [
        _passing_row(spec, 2000 + index, 1000 + index * 100)
        for index, spec in enumerate(exp.MODEL_SPECS)
    ]


def _upstream(observed: object = 1.0) -> dict:
    return {
        "path": exp.UPSTREAM_PATH.as_posix(),
        "sha256": "sha256:" + "a" * 64,
        "field": "task_owned_admission_ready_score",
        "expected_value": 1.0,
        "observed_value": observed,
        "validator_errors": [],
        "passed": observed == 1.0,
    }


def _resolution_rows() -> list[dict]:
    return [
        {
            **spec,
            "model_path": f"/cache/{spec['family_id']}.gguf",
            "model_sha256": f"sha256:{index + 1:064x}",
            "byte_count": 10_000 + index,
            "resolved": True,
        }
        for index, spec in enumerate(exp.MODEL_SPECS)
    ]


def _preconditions() -> dict:
    return {
        "all_required_preconditions_available": True,
        "checks": {
            "upstream_gate": True,
            "model_resolution": True,
            "model_hashes": True,
            "two_gpu_uuids": True,
            "llama_cpp_cuda": True,
            "resources": True,
            "protected_hashes": True,
        },
        "failed_preconditions": [],
        "cache": _resolution_rows(),
        "hardware": {"gpus": [{"uuid": "GPU-test-0"}, {"uuid": "GPU-test-1"}]},
        "tools": {"llama_cpp": {"cuda_linked": True}},
        "resources": {"cpu_count": 24, "ram_bytes": 1, "disk_free_bytes": 1},
    }


def _artifact(tmp_path: Path, rows: list[dict] | None = None) -> dict:
    before = exp.protected_hashes(REPO)
    return exp.build_artifact(
        date="20260826",
        root=REPO,
        duration_s=180.0,
        upstream_gate_receipt=_upstream(),
        resolution_rows=_resolution_rows(),
        admission_rows=_passing_rows() if rows is None else rows,
        preconditions=_preconditions(),
        protected_before=before,
        tests_run=exp.DEFAULT_TESTS_RUN,
    )


def test_req_6648_specs_and_exact_model_policy() -> None:
    """REQ-INFER-SOTA-6648 freezes IDs, roles, devices, and resolver paths."""

    anchors = {
        exp.INFERENCE_SPEC_PATH: (
            "REQ-INFER-SOTA-6648",
            "SCENARIO-INFER-SOTA-6648-ALL-FAMILIES",
            "SCENARIO-INFER-SOTA-6648-NO-SUBSTITUTION",
        ),
        exp.INFRA_SPEC_PATH: (
            "REQ-INFRA-6648",
            "SCENARIO-INFRA-6648-INDEPENDENT-PROCESSES",
            "SCENARIO-INFRA-6648-LIFECYCLE-BLOCK",
        ),
        exp.REPORT_SPEC_PATH: (
            "REQ-REPORT-6648",
            "SCENARIO-REPORT-6648-READY",
            "SCENARIO-REPORT-6648-BLOCKED",
            "SCENARIO-REPORT-6648-ATTACKS-AND-ATOMIC",
        ),
    }
    for path, expected in anchors.items():
        text = path.read_text(encoding="utf-8")
        assert all(anchor in text for anchor in expected)
    assert exp.MODEL_SPECS == [
        {
            "family_id": "qwen36_flagship_moe",
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "role": "flagship_moe",
            "quantization": "Q4_K_M",
            "device_index": 0,
            "resolution_method": "cached_sota_pair",
        },
        {
            "family_id": "gemma4_26b_middle_moe",
            "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
            "role": "middle_moe",
            "quantization": "Q4_K_M",
            "device_index": 1,
            "resolution_method": "cached_sota_pair",
        },
        {
            "family_id": "gemma4_31b_flagship_dense",
            "hf_id": "unsloth/gemma-4-31B-it-GGUF",
            "role": "flagship_dense",
            "quantization": "Q4_K_M",
            "device_index": 0,
            "resolution_method": "resolve_cached_gguf",
        },
    ]


def test_req_infer_6648_resolution_calls_exact_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-INFER-SOTA-6648-ALL-FAMILIES uses both required resolvers."""

    calls: list[tuple] = []

    def pair(**kwargs: object) -> list[dict]:
        calls.append(("pair", kwargs))
        return [
            {"hf_id": exp.MODEL_SPECS[0]["hf_id"], "gpu": 0, "model_path": "/q.gguf"},
            {"hf_id": exp.MODEL_SPECS[1]["hf_id"], "gpu": 1, "model_path": "/m.gguf"},
        ]

    def dense(hf_id: str, preferred_quant: str = "Q4_K_M") -> str:
        calls.append(("dense", hf_id, preferred_quant))
        return "/d.gguf"

    monkeypatch.setattr(exp, "cached_sota_pair", pair)
    monkeypatch.setattr(exp, "resolve_cached_gguf", dense)
    monkeypatch.setattr(exp, "sha256_file", lambda path: f"sha256:{Path(path).name:0>64}")
    monkeypatch.setattr(
        Path, "is_file", lambda self: str(self) in {"/q.gguf", "/m.gguf", "/d.gguf"}
    )
    monkeypatch.setattr(Path, "stat", lambda self: type("S", (), {"st_size": 123})())
    rows = exp.resolve_model_specs()
    assert calls[0] == (
        "pair",
        {"gpu_indices": (0, 1), "model_indices": (0, 1)},
    )
    assert calls[1] == ("dense", exp.MODEL_SPECS[2]["hf_id"], "Q4_K_M")
    assert [row["model_path"] for row in rows] == ["/q.gguf", "/m.gguf", "/d.gguf"]
    assert all(row["resolved"] is True for row in rows)


def test_scenario_infra_6648_independent_rows_recompute_ready(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6648-INDEPENDENT-PROCESSES requires distinct sequential workers."""

    rows = _passing_rows()
    failures, aggregate = exp.reduce_model_admission_rows(rows)
    assert failures == []
    assert aggregate["all_mandated_models_admitted"] is True
    assert aggregate["family_row_count"] == 3
    assert aggregate["distinct_worker_identity_count"] == 3
    assert aggregate["sequential_launch_order"] is True
    assert all(exp.family_row_failures(row) == [] for row in rows)
    artifact = _artifact(tmp_path, rows)
    assert exp.validate_artifact(artifact) == []
    assert artifact["status"] == "complete_ready"
    assert artifact["verdict_class"] is None
    assert artifact["all_mandated_models_admitted"] is True
    assert "quality" not in artifact["honest_verdict"].lower()
    assert "performance" not in artifact["honest_verdict"].lower()


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda row: row.update(hf_id="legacy/smoke"), "model_identity_mismatch"),
        (lambda row: row["tokenizer"].update(auto_tokenizer_used=True), "auto_tokenizer_used"),
        (lambda row: row["output"].update(text="", non_empty=False), "output_empty"),
        (lambda row: row["accelerator"].update(cuda_offload=False), "cuda_offload_missing"),
        (lambda row: row["lease"].update(phase_sequence=[]), "phase_sequence_mismatch"),
        (lambda row: row["unload"].update(observed=False), "unload_missing"),
    ],
)
def test_scenario_infer_6648_substitution_and_lifecycle_fail_closed(
    mutation: object, reason: str
) -> None:
    """SCENARIO-INFER-SOTA-6648-NO-SUBSTITUTION rejects false admission evidence."""

    row = _passing_rows()[0]
    mutation(row)  # type: ignore[operator]
    row["row_sha256"] = exp.family_row_hash(row)
    assert reason in exp.family_row_failures(row)
    failures, aggregate = exp.reduce_model_admission_rows([row, *_passing_rows()[1:]])
    assert aggregate["all_mandated_models_admitted"] is False
    assert any(failure["check"].endswith(reason) for failure in failures)


def test_scenario_report_6648_missing_duplicate_reused_and_upstream_block(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6648-BLOCKED preserves exact missing and reused values."""

    rows = _passing_rows()
    duplicate = [rows[0], rows[0], rows[2]]
    failures, aggregate = exp.reduce_model_admission_rows(duplicate)
    assert aggregate["all_mandated_models_admitted"] is False
    assert any(item["reason"] == "duplicate_family_row" for item in failures)

    reused = _passing_rows()
    reused[1]["worker_process"]["pid"] = reused[0]["worker_process"]["pid"]
    reused[1]["worker_process"]["pid_start_ticks"] = reused[0]["worker_process"]["pid_start_ticks"]
    reused[1]["lease"]["owner"]["pid"] = reused[0]["worker_process"]["pid"]
    reused[1]["lease"]["owner"]["pid_start_ticks"] = reused[0]["worker_process"]["pid_start_ticks"]
    reused[1]["row_sha256"] = exp.family_row_hash(reused[1])
    failures, _ = exp.reduce_model_admission_rows(reused)
    assert any(item["reason"] == "reused_worker_identity" for item in failures)

    before = exp.protected_hashes(REPO)
    blocked = exp.build_artifact(
        date="20260826",
        root=REPO,
        duration_s=1.0,
        upstream_gate_receipt=_upstream(None),
        resolution_rows=_resolution_rows(),
        admission_rows=[],
        preconditions={
            **_preconditions(),
            "all_required_preconditions_available": False,
            "failed_preconditions": ["upstream_gate"],
            "checks": {**_preconditions()["checks"], "upstream_gate": False},
        },
        protected_before=before,
        tests_run=exp.DEFAULT_TESTS_RUN,
    )
    assert blocked["status"].startswith("blocked_")
    assert blocked["honest_verdict"].startswith("blocked_")
    assert blocked["verdict_class"] == "blocked"
    assert blocked["all_mandated_models_admitted"] is False
    assert blocked["gate_check_summary"][0]["observed_value"] is None
    assert exp.validate_artifact(blocked) == []


def test_scenario_report_6648_attack_rows_detect_all_mutations(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6648-ATTACKS-AND-ATOMIC detects every required attack."""

    attacks = exp.build_attack_rows(_passing_rows())
    assert [row["attack_id"] for row in attacks] == list(exp.REQUIRED_ATTACK_IDS)
    assert all(row["detected"] is True for row in attacks)
    artifact = _artifact(tmp_path)
    assert len(artifact["per_unit_rows"]) == 3 + len(exp.REQUIRED_ATTACK_IDS)
    assert all(row["row_kind"] == "attack" for row in artifact["per_unit_rows"][3:])


def test_scenario_report_6648_validator_rejects_an_undetected_attack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6648-ATTACKS-AND-ATOMIC fails closed on attack drift."""

    artifact = _artifact(tmp_path)
    attacks = exp.build_attack_rows(_passing_rows())
    attacks[0]["detected"] = False
    monkeypatch.setattr(exp, "build_attack_rows", lambda rows: attacks)
    errors = exp.validate_artifact(artifact)
    assert "per_unit_rows_mismatch" in errors
    assert "attack_rows_invalid" in errors


def test_req_report_6648_field_provenance_hashes_and_checksum(tmp_path: Path) -> None:
    """REQ-REPORT-6648 gives every field replayable lineage and one final hash."""

    artifact = _artifact(tmp_path)
    assert set(artifact) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_provenance"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert all(
        set(receipt) == {"source", "hash", "reducer", "schema"}
        for receipt in artifact["field_provenance"].values()
    )
    assert artifact["reproducibility_checksum"] == exp.payload_checksum(artifact)
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["protected_files_unchanged"]["all_unchanged"] is True
    assert len(artifact["embedded_tokenizer_rows"]) == 3
    assert len(artifact["lease_and_unload_receipts"]) == 3


def test_scenario_report_6648_validator_rejects_row_aggregate_and_content_mutation(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6648-ATTACKS-AND-ATOMIC rejects changed durable evidence."""

    artifact = _artifact(tmp_path)
    bad_row = deepcopy(artifact)
    bad_row["model_admission_rows"][0]["output"]["text"] = "changed"
    bad_row["reproducibility_checksum"] = exp.payload_checksum(bad_row)
    assert "row_hash_mismatch:qwen36_flagship_moe" in exp.validate_artifact(bad_row)

    aggregate = deepcopy(artifact)
    aggregate["all_mandated_models_admitted"] = False
    aggregate["reproducibility_checksum"] = exp.payload_checksum(aggregate)
    assert "aggregate_admission_mismatch" in exp.validate_artifact(aggregate)

    protected = deepcopy(artifact)
    protected["protected_files_unchanged"]["all_unchanged"] = False
    protected["reproducibility_checksum"] = exp.payload_checksum(protected)
    assert "protected_files_changed" in exp.validate_artifact(protected)

    checksum = deepcopy(artifact)
    checksum["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(checksum)


def test_req_infra_6648_worker_commands_are_sequential_and_rows_are_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6648 waits for each worker and proves its later absence."""

    launches: list[str] = []

    def launch(spec: dict, runtime_dir: Path, repo_root: Path) -> dict:
        launches.append(spec["family_id"])
        assert launches == [row["family_id"] for row in exp.MODEL_SPECS[: len(launches)]]
        row = _passing_row(spec, 3000 + len(launches), len(launches) * 100)
        row["model_path"] = spec["model_path"]
        row["model_sha256"] = spec["model_sha256"]
        row["worker_process"]["absent_after_exit"] = True
        row["row_sha256"] = exp.family_row_hash(row)
        return row

    monkeypatch.setattr(exp, "launch_family_worker", launch)
    resolved = _resolution_rows()
    rows = exp.run_family_workers(resolved, tmp_path, REPO)
    assert launches == [spec["family_id"] for spec in exp.MODEL_SPECS]
    assert len(rows) == 3
    assert all(row["worker_process"]["absent_after_exit"] is True for row in rows)


def test_req_report_6648_run_and_validate_cli_without_live_hardware(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-6648 publishes one valid atomic JSON document."""

    output = tmp_path / "exp6648.json"
    monkeypatch.setattr(exp, "resolve_model_specs", _resolution_rows)
    monkeypatch.setattr(exp, "collect_preconditions", lambda *args, **kwargs: _preconditions())
    monkeypatch.setattr(exp, "run_family_workers", lambda *args, **kwargs: _passing_rows())
    monkeypatch.setattr(exp, "run_verification_commands", lambda root: list(exp.DEFAULT_TESTS_RUN))
    artifact = exp.run(date="20260826", root=REPO, result_path=output, work_dir=tmp_path / "work")
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert exp.validate_artifact(artifact) == []
    assert exp.main(["--validate", "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1]) == {"errors": [], "valid": True}

    missing = tmp_path / "missing.json"
    assert exp.main(["--validate", "--output", str(missing)]) == 1
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["errors"] == ["artifact_missing"]


def test_req_report_6648_preconditions_preserve_upstream_tools_and_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6648 records the upstream gate, cache, GPUs, tools, and host."""

    monkeypatch.setattr(
        exp, "gpu_inventory", lambda: [{"index": 0, "uuid": "GPU-0"}, {"index": 1, "uuid": "GPU-1"}]
    )
    monkeypatch.setattr(
        exp,
        "llama_cpp_receipt",
        lambda: {"path": "/llama-server", "exists": True, "cuda_linked": True},
    )
    rows = _resolution_rows()
    before = exp.protected_hashes(REPO)
    receipt = exp.collect_preconditions(
        REPO,
        tmp_path,
        _upstream(),
        rows,
        before,
    )
    assert receipt["all_required_preconditions_available"] is True
    assert receipt["cache"] == rows
    assert [row["uuid"] for row in receipt["hardware"]["gpus"]] == ["GPU-0", "GPU-1"]
    assert receipt["tools"]["llama_cpp"]["cuda_linked"] is True
    assert receipt["resources"]["cpu_count"]
    assert receipt["resources"]["ram_bytes"] > 0
    assert receipt["resources"]["disk_free_bytes"] > 0
    assert receipt["worker_runtime_dir"] == str(tmp_path.resolve())
    assert receipt["auto_tokenizer_allowed"] is False
    assert receipt["download_allowed"] is False


def test_req_infra_6648_gpu_snapshot_normalizes_live_nvidia_uuid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-INFRA-6648 gives live accelerator snapshots the reducer's UUID key."""

    monkeypatch.setattr(
        exp,
        "gpu_inventory",
        lambda: [{"index": 0, "uuid": "GPU-live-0", "memory_used_mb": 12000}],
    )
    monkeypatch.setattr(
        exp,
        "_compute_processes",
        lambda: [{"gpu_uuid": "GPU-live-0", "pid": 4242}],
    )
    snapshot = exp._gpu_snapshot(0, 4242)
    assert snapshot["uuid"] == "GPU-live-0"
    assert snapshot["device_uuid"] == "GPU-live-0"
    assert snapshot["model_pid_present"] is True


def test_scenario_infra_6648_all_malformed_receipts_and_order_drift_fail_closed() -> None:
    """SCENARIO-INFRA-6648-LIFECYCLE-BLOCK covers every receipt boundary."""

    malformed = _passing_rows()[0]
    malformed.update(
        {
            "row_kind": "legacy_smoke",
            "model_path": "",
            "model_sha256": "missing",
            "tokenizer": {
                "source": "wrong",
                "loadable": False,
                "auto_tokenizer_used": False,
                "prompt_token_count": 0,
            },
            "worker_process": {
                "pid": 0,
                "pid_start_ticks": None,
                "executable": "",
                "argv": [],
                "argv_sha256": "bad",
                "exit_code": 1,
                "absent_after_exit": False,
            },
            "model_process": {
                "pid": 0,
                "parent_pid": 99,
                "argv": [],
                "argv_sha256": "bad",
                "exit_code": None,
                "absent_after_exit": False,
            },
            "prompt": {"sha256": "bad", "random_seed": 1},
            "errors": ["runtime failed"],
        }
    )
    malformed["lease"]["journal_checksum"] = "missing"
    malformed["accelerator"]["resident"].update(
        {"model_pid_present": False, "device_uuid": "GPU-wrong"}
    )
    malformed["accelerator"]["after"]["model_pid_present"] = True
    malformed["accelerator"]["resident_vram_delta_mb"] = 0
    malformed["output"].update(
        {
            "sha256": "bad",
            "prompt_token_count": 0,
            "output_token_count": 0,
            "http_status": 500,
        }
    )
    malformed["unload"]["vram_recovered"] = False
    malformed = exp.seal_family_row(malformed)
    failures = exp.family_row_failures(malformed)
    assert {
        "row_kind_mismatch",
        "model_file_receipt_invalid",
        "embedded_tokenizer_missing",
        "tokenizer_token_count_missing",
        "worker_identity_missing",
        "worker_start_missing",
        "worker_command_missing",
        "worker_argv_hash_mismatch",
        "worker_exit_or_absence_missing",
        "model_process_identity_missing",
        "model_process_not_owned_child",
        "model_argv_hash_mismatch",
        "model_exit_or_absence_missing",
        "journal_checksum_missing",
        "device_uuid_mismatch",
        "resident_vram_missing",
        "post_gpu_process_presence",
        "prompt_contract_mismatch",
        "output_hash_mismatch",
        "output_token_count_missing",
        "inference_exit_invalid",
        "vram_recovery_missing",
        "runtime_errors_present",
    }.issubset(failures)
    assert exp.family_row_failures({"family_id": "unknown"})[0] == "unknown_family"

    overlap = _passing_rows()
    overlap[1]["worker_process"]["started_monotonic_ns"] = overlap[0]["worker_process"][
        "ended_monotonic_ns"
    ]
    overlap[1]["row_sha256"] = exp.family_row_hash(overlap[1])
    failures, _ = exp.reduce_model_admission_rows(overlap)
    assert any(row["reason"] == "worker_process_overlap_or_order_drift" for row in failures)
    failures, _ = exp.reduce_model_admission_rows([*_passing_rows(), {"family_id": "extra"}])
    assert any(row["reason"] == "unexpected_family_row" for row in failures)


def test_req_report_6648_validator_reports_each_top_level_contract_mutation(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6648-ATTACKS-AND-ATOMIC covers top-level validation."""

    artifact = _artifact(tmp_path)

    def errors_for(**changes: object) -> list[str]:
        changed = deepcopy(artifact)
        changed.update(changes)
        changed["reproducibility_checksum"] = exp.payload_checksum(changed)
        return exp.validate_artifact(changed)

    missing = deepcopy(artifact)
    missing.pop("duration_s")
    missing["reproducibility_checksum"] = exp.payload_checksum(missing)
    assert "required_fields_mismatch" in exp.validate_artifact(missing)
    assert "defined_model_specs_mismatch" in errors_for(defined_model_specs=[])
    assert "ready_status_mismatch" in errors_for(status="wrong")
    assert "ready_verdict_class_mismatch" in errors_for(verdict_class="blocked")
    assert "embedded_tokenizer_rows_mismatch" in errors_for(embedded_tokenizer_rows=[])
    assert "lease_and_unload_receipts_mismatch" in errors_for(lease_and_unload_receipts=[])
    assert "inference_substrate_mismatch" in errors_for(inference_substrate="wrong")
    assert "verifier_is_oracle_mismatch" in errors_for(verifier_is_oracle=True)
    assert "random_seed_mismatch" in errors_for(random_seed=1)
    assert "field_provenance_mismatch" in errors_for(field_provenance={})

    blocked = deepcopy(artifact)
    blocked["upstream_gate_receipt"]["passed"] = False
    blocked["upstream_gate_receipt"]["observed_value"] = 0.0
    blocked = exp.build_artifact(
        date="20260826",
        root=REPO,
        duration_s=1.0,
        upstream_gate_receipt=blocked["upstream_gate_receipt"],
        resolution_rows=_resolution_rows(),
        admission_rows=_passing_rows(),
        preconditions=_preconditions(),
        protected_before=exp.protected_hashes(REPO),
        tests_run=exp.DEFAULT_TESTS_RUN,
    )

    def blocked_errors(**changes: object) -> list[str]:
        changed = deepcopy(blocked)
        changed.update(changes)
        changed["reproducibility_checksum"] = exp.payload_checksum(changed)
        return exp.validate_artifact(changed)

    assert "blocked_status_mismatch" in blocked_errors(status="wrong")
    assert "blocked_verdict_mismatch" in blocked_errors(honest_verdict="wrong")
    assert "blocked_verdict_class_mismatch" in blocked_errors(verdict_class=None)


def test_req_report_6648_command_and_blocked_worker_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6648 preserves command failure and absent-worker evidence."""

    receipt = exp._command_receipt("printf ok", REPO, 10.0)
    assert receipt["exit_code"] == 0
    assert receipt["summary"] == "ok"

    def timeout(*args: object, **kwargs: object) -> object:
        raise __import__("subprocess").TimeoutExpired("cmd", 2.0)

    monkeypatch.setattr(exp.subprocess, "run", timeout)
    timed_out = exp._command_receipt("slow", REPO, 2.0)
    assert timed_out["exit_code"] == 124
    assert "TimeoutExpired" in timed_out["summary"]

    blocked = exp._blocked_worker_row(_resolution_rows()[0], ["python", "-m"], "missing")
    assert blocked["admitted"] is False
    assert "worker_identity_missing" in blocked["failed_checks"]
    assert blocked["row_sha256"] == exp.family_row_hash(blocked)
    assert exp.sha256_file(tmp_path / "missing") == "missing"
    assert exp._read_json(tmp_path / "missing") == {}
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    assert exp._read_json(invalid) == {}


def test_req_report_6648_failed_commands_run_refusal_and_cli_run_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-6648 blocks failed checks and refuses an invalid final write."""

    failed_tests = [{"command": "focused", "exit_code": 5, "summary": "failed"}]
    preconditions = {
        **_preconditions(),
        "all_required_preconditions_available": False,
        "failed_preconditions": ["resources"],
        "checks": {**_preconditions()["checks"], "resources": False},
    }
    artifact = exp.build_artifact(
        date="20260826",
        root=REPO,
        duration_s=1.0,
        upstream_gate_receipt=_upstream(),
        resolution_rows=_resolution_rows(),
        admission_rows=[],
        preconditions=preconditions,
        protected_before=exp.protected_hashes(REPO),
        tests_run=failed_tests,
    )
    assert any(row["check"] == "precondition.resources" for row in artifact["gate_check_summary"])
    assert any(
        row["reason"] == "verification_command_failed" for row in artifact["gate_check_summary"]
    )

    monkeypatch.setattr(exp, "resolve_model_specs", _resolution_rows)
    monkeypatch.setattr(exp, "collect_preconditions", lambda *args, **kwargs: _preconditions())
    monkeypatch.setattr(exp, "run_family_workers", lambda *args, **kwargs: _passing_rows())
    monkeypatch.setattr(exp, "run_verification_commands", lambda root: list(exp.DEFAULT_TESTS_RUN))
    monkeypatch.setattr(exp, "validate_artifact", lambda artifact: ["forced_invalid"])
    with pytest.raises(ValueError, match="forced_invalid"):
        exp.run(
            date="20260826",
            root=REPO,
            result_path=tmp_path / "refused.json",
            work_dir=tmp_path / "work",
        )
    assert not (tmp_path / "refused.json").exists()

    unreadable = tmp_path / "unreadable.json"
    unreadable.write_text("{", encoding="utf-8")
    assert exp.main(["--validate", "--output", str(unreadable)]) == 1
    assert "artifact_unreadable:JSONDecodeError" in capsys.readouterr().out

    monkeypatch.setattr(
        exp,
        "run",
        lambda **kwargs: {
            "status": "complete_ready",
            "all_mandated_models_admitted": True,
        },
    )
    assert exp.main(["--output", str(tmp_path / "unused.json")]) == 0
    assert (
        json.loads(capsys.readouterr().out.splitlines()[-1])["all_mandated_models_admitted"] is True
    )
