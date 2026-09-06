"""Artifact tests for the V620 GPU lease migration.

Spec refs: REQ-INFRA-7078, SCENARIO-INFRA-7078-PRESERVE-AND-PUBLISH,
SCENARIO-INFRA-7078-IDEMPOTENT, and
SCENARIO-INFRA-7078-DUAL-DEVICE-PREFLIGHT.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import carnot
from carnot import experiment_7078_v620_gpu_lease_migration as exp
from carnot import gpu_lease_phase_journal as lease_api


def _legacy_terminal(runtime_dir: Path, device_uuid: str) -> None:
    lease = lease_api.GpuLease.acquire(
        runtime_dir=runtime_dir,
        task_id=f"old-task:{device_uuid}",
        device_uuid=device_uuid,
        expected_model="fixture/model.gguf",
        vram_before_mb=4,
    )
    lease_api._complete_fixture(lease)
    lease.release()
    document = json.loads(lease.journal_path.read_text(encoding="utf-8"))
    document.pop("lease_id")
    document["checksum"] = lease_api.journal_checksum(document)
    lease_api.write_json_atomic(lease.journal_path, document)


def _gpu_probe() -> dict:
    return {
        "query_ok": True,
        "devices": [
            {"index": 0, "uuid": "GPU-a", "name": "NVIDIA GeForce RTX 3090"},
            {"index": 1, "uuid": "GPU-b", "name": "NVIDIA GeForce RTX 3090"},
        ],
        "processes": [],
    }


def _preflight_probe(
    devices: Sequence[Mapping[str, Any]], runtime_dir: Path
) -> list[dict[str, Any]]:
    return [
        {
            "device_uuid": row["uuid"],
            "classification": "available",
            "journal_path": str(lease_api.journal_path_for(runtime_dir, row["uuid"])),
            "signals_sent": [],
        }
        for row in devices
    ]


def test_req_infra_7078_artifact_has_every_principled_field(tmp_path: Path) -> None:
    """REQ-INFRA-7078: a valid run records all migration and safety evidence."""

    runtime = tmp_path / "runtime"
    _legacy_terminal(runtime, "GPU-a")
    _legacy_terminal(runtime, "GPU-b")
    output = tmp_path / "artifact.json"
    artifact = exp.run(
        date="20260906",
        runtime_dir=runtime,
        result_path=output,
        gpu_probe=_gpu_probe,
        process_match=lambda _pid, _ticks: False,
        lease_id_factory=lambda: "lease:artifact-migration",
        preflight_probe=_preflight_probe,
    )
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert set(artifact) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["inference_substrate"] == "deterministic_os_lease_recovery_no_llm"
    assert artifact["gpu_lease_compatibility_ready_score"] == 1
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("null_")
    assert artifact["signals_sent"] == []
    assert artifact["files_removed"] == []
    assert len(artifact["migration_rows"]) >= 2
    assert all(
        row["classification"] == "available" for row in artifact["post_migration_preflight_rows"]
    )
    assert exp.validate_artifact(artifact) == []


def test_scenario_infra_7078_artifact_blocks_before_mutation(tmp_path: Path) -> None:
    """A live owner blocks both devices before the experiment changes either journal."""

    runtime = tmp_path / "runtime"
    _legacy_terminal(runtime, "GPU-a")
    _legacy_terminal(runtime, "GPU-b")
    paths = [lease_api.journal_path_for(runtime, device) for device in ("GPU-a", "GPU-b")]
    before = [path.read_bytes() for path in paths]
    artifact = exp.run(
        date="20260906",
        runtime_dir=runtime,
        result_path=tmp_path / "blocked.json",
        gpu_probe=_gpu_probe,
        process_match=lambda _pid, _ticks: True,
        preflight_probe=_preflight_probe,
    )
    assert artifact["gpu_lease_compatibility_ready_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["gate_check_summary"]["failed_check"] == "preconditions"
    assert [path.read_bytes() for path in paths] == before
    assert exp.validate_artifact(artifact) == []


def test_scenario_infra_7078_validator_rejects_projection_drift(tmp_path: Path) -> None:
    """The cold validator rejects score, row, principle, prefix, and hash drift."""

    runtime = tmp_path / "runtime"
    _legacy_terminal(runtime, "GPU-a")
    _legacy_terminal(runtime, "GPU-b")
    artifact = exp.run(
        date="20260906",
        runtime_dir=runtime,
        result_path=tmp_path / "valid.json",
        gpu_probe=_gpu_probe,
        process_match=lambda _pid, _ticks: False,
        preflight_probe=_preflight_probe,
    )

    def errors_for(**changes: object) -> list[str]:
        changed = deepcopy(artifact)
        changed.update(changes)
        changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
        return exp.validate_artifact(changed)

    assert "readiness_score_mismatch" in errors_for(gpu_lease_compatibility_ready_score=0)
    assert "verdict_class_mismatch" in errors_for(verdict_class="positive")
    assert "honest_verdict_prefix_mismatch" in errors_for(honest_verdict="complete_wrong")
    assert "field_principles_mismatch" in errors_for(field_principles={})
    assert "post_migration_preflight_mismatch" in errors_for(post_migration_preflight_rows=[])
    bad = deepcopy(artifact)
    bad.pop("rows")
    assert "required_fields_mismatch" in exp.validate_artifact(bad)
    bad = deepcopy(artifact)
    bad["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(bad)


def test_scenario_infra_7078_cli_validation_uses_temporary_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI validator reads an explicit artifact and never rewrites it."""

    missing = tmp_path / "missing.json"
    assert exp.main(["--validate", "--output", str(missing)]) == 1
    assert json.loads(capsys.readouterr().out)["valid"] is False

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}", encoding="utf-8")
    before = invalid.read_bytes()
    assert exp.main(["--validate", "--output", str(invalid)]) == 1
    assert json.loads(capsys.readouterr().out)["valid"] is False
    assert invalid.read_bytes() == before


def test_scenario_infra_7078_precondition_diagnostics_cover_unreadable_inputs(
    tmp_path: Path,
) -> None:
    """Read-only inspection retains absent, non-object, and kernel-table failures."""

    assert exp._lock_observed_held(tmp_path / "absent") is False
    lock_path = tmp_path / "lock"
    lock_path.touch()
    assert exp._lock_observed_held(lock_path, tmp_path) is None
    locks = tmp_path / "locks"
    locks.write_text(f"1: FLOCK ADVISORY WRITE 1 00:00:{lock_path.stat().st_ino} 0 EOF\n")
    assert exp._lock_observed_held(lock_path, locks) is True

    runtime = tmp_path / "runtime"
    list_path = lease_api.journal_path_for(runtime, "GPU-list")
    list_path.parent.mkdir(parents=True)
    list_path.write_text("[]", encoding="utf-8")
    result = exp.collect_preconditions(
        [{"uuid": "GPU-list"}, {"uuid": "GPU-missing"}],
        runtime,
        process_match=lambda _pid, _ticks: False,
    )
    assert result["all_passed"] is False
    assert [row["kind"] for row in result["legacy_validation_rows"]] == [
        "unreadable",
        "unreadable",
    ]
    assert all(row["identity_absent_or_reused"] is False for row in result["owner_liveness_rows"])


def test_scenario_infra_7078_synthetic_rejection_success_paths_are_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Synthetic rows turn false acceptance into visible failed evidence."""

    runtime = tmp_path / "source"
    _legacy_terminal(runtime, "GPU-source")
    source = json.loads(
        lease_api.journal_path_for(runtime, "GPU-source").read_text(encoding="utf-8")
    )
    original = lease_api.migrate_legacy_journal

    def accept_required_blocks(**kwargs: object) -> dict:
        case_root = Path(kwargs["runtime_dir"])
        if case_root.name in {
            "bad-checksum",
            "missing-owner",
            "live",
            "held",
            "wrong-uuid",
            "nonterminal",
            "atomic",
        }:
            return {"action": "accepted"}
        return original(**kwargs)

    monkeypatch.setattr(lease_api, "migrate_legacy_journal", accept_required_blocks)
    rows = exp.run_synthetic_migration_rows(source)
    assert len(rows) == len(exp.SYNTHETIC_CASES)
    assert {row["case"] for row in rows if row["passed"] is False} == {
        "bad_checksum",
        "missing_owner_fields",
        "live_pid",
        "held_lock",
        "wrong_uuid",
        "nonterminal_phase",
        "atomic_write_interruption",
    }


def test_scenario_infra_7078_current_source_and_exp7065_probe_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repeat run reloads preserved legacy bytes and calls the shipped probe."""

    runtime = tmp_path / "runtime"
    _legacy_terminal(runtime, "GPU-a")
    migrated = lease_api.migrate_legacy_journal(
        runtime_dir=runtime,
        device_uuid="GPU-a",
        process_match=lambda _pid, _ticks: False,
    )
    discovered = [
        {
            "journal_path": str(lease_api.journal_path_for(runtime, "GPU-a")),
            "source_sha256": migrated["target_sha256"],
        }
    ]
    legacy = exp._legacy_source_for_synthetic(runtime, discovered)
    assert legacy is not None
    assert "lease_id" not in legacy
    assert (
        exp._legacy_source_for_synthetic(runtime, [{"journal_path": str(tmp_path / "missing")}])
        is None
    )

    fake = SimpleNamespace(
        LEASE_RUNTIME_DIR=Path("original"),
        _lease_probe=lambda devices: [
            {"device_uuid": row["uuid"], "classification": "available"} for row in devices
        ],
    )
    module_name = "carnot.experiment_7065_v619_three_family_entrance_bank"
    monkeypatch.setitem(sys.modules, module_name, fake)
    monkeypatch.setattr(
        carnot, "experiment_7065_v619_three_family_entrance_bank", fake, raising=False
    )
    rows = exp._post_migration_probe([{"uuid": "GPU-a"}], runtime)
    assert rows == [{"device_uuid": "GPU-a", "classification": "available"}]
    assert fake.LEASE_RUNTIME_DIR == Path("original")


def test_scenario_infra_7078_run_defensive_and_default_cli_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Default probe loading, strict-reader drift, invalid builds, and CLI output are covered."""

    fake_inventory = ModuleType("carnot.experiment_6966_gguf_load_envelope_canary")
    fake_inventory.gpu_inventory = lambda: {"devices": [], "processes": [], "query_ok": True}
    monkeypatch.setitem(
        sys.modules, "carnot.experiment_6966_gguf_load_envelope_canary", fake_inventory
    )
    blocked = exp.run(
        date="20260906",
        runtime_dir=tmp_path / "empty",
        result_path=tmp_path / "default-probe.json",
    )
    assert blocked["verdict_class"] == "blocked"

    runtime = tmp_path / "strict"
    _legacy_terminal(runtime, "GPU-a")
    _legacy_terminal(runtime, "GPU-b")
    original_reader = lease_api.read_journal

    def accept_preserved(path: object) -> dict:
        if ".legacy-journal.json" in str(path):
            return {}
        return original_reader(path)

    monkeypatch.setattr(lease_api, "read_journal", accept_preserved)
    strict_block = exp.run(
        date="20260906",
        runtime_dir=runtime,
        result_path=tmp_path / "strict-block.json",
        gpu_probe=_gpu_probe,
        process_match=lambda _pid, _ticks: False,
        preflight_probe=_preflight_probe,
    )
    assert strict_block["gpu_lease_compatibility_ready_score"] == 0
    monkeypatch.setattr(lease_api, "read_journal", original_reader)

    original_validator = exp.validate_artifact
    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["forced_invalid"])
    with pytest.raises(ValueError, match="artifact_invalid:forced_invalid"):
        exp.run(
            date="20260906",
            runtime_dir=tmp_path / "invalid-build",
            result_path=tmp_path / "invalid-build.json",
            gpu_probe=lambda: {"devices": [], "processes": []},
        )
    monkeypatch.setattr(exp, "validate_artifact", original_validator)

    monkeypatch.setattr(
        exp,
        "run",
        lambda **_kwargs: {
            "gpu_lease_compatibility_ready_score": 1,
            "honest_verdict": "null_test",
        },
    )
    assert exp.main(["--date", "20260906", "--output", str(tmp_path / "cli.json")]) == 0
    assert json.loads(capsys.readouterr().out)["honest_verdict"] == "null_test"


def test_scenario_infra_7078_module_entrypoint_validates_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The direct module entrypoint can validate a temporary artifact."""

    invalid = tmp_path / "entrypoint.json"
    invalid.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["experiment_7078_v620_gpu_lease_migration", "--validate", "--output", str(invalid)],
    )
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("carnot.experiment_7078_v620_gpu_lease_migration", run_name="__main__")
    assert exc.value.code == 1
