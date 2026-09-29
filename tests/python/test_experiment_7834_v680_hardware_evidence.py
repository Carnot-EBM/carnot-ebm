"""REQ-REPORT-7834: preserve board custody while checking current service cost."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.reporting.experiment_7834_v680_hardware_evidence import (
    build_audit,
    cold_reduce,
    dispatch,
    main,
)
from carnot.reporting.experiment_7820_v679_hardware_evidence import run_child


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    ROOT / "results/raw/experiment_7834_v680_hardware_evidence/validation_command_manifest.json"
)


def test_missing_service_keeps_authenticated_board_history() -> None:
    """SCENARIO-REPORT-7834-SERVICE: missing service blocks only the service gate."""
    audit = build_audit(ROOT, "20260928", MANIFEST)
    assert audit["experiment_id"] == 7834
    assert audit["milestone"] == "2026.09.680"
    assert audit["honest_verdict"] == "complete_blocked_missing_service_evidence"
    assert audit["verdict_class"] == "blocked"
    assert audit["gate_check_summary"] == [
        {
            "upstream_id": "Exp7833",
            "artifact_path": "results/experiment_7833_v680_service_cost.json",
            "artifact_hash": None,
            "field": "service_evidence_ready_score",
            "operator": "==",
            "expected": 1,
            "observed": None,
            "passed": False,
        }
    ]
    boards = {row["board"]: row for row in audit["board_rows"]}
    assert set(boards) == {"KV260", "PolarFire", "GateMate"}
    assert boards["KV260"]["k_max"] == 5
    assert boards["PolarFire"]["processor_class"] == "linux_cpu"
    assert boards["GateMate"]["blocker"] == "0xffffffff"
    assert all(row["evidence_age_days"] >= 13 for row in boards.values())
    assert audit["acceleration_bound"] is None
    assert audit["hardware_advantage_claimed"] is False
    assert (
        audit["stage_opportunity_map"]["sampler_integration"] == "deferred_no_measured_sampler_work"
    )
    assert audit["acceptance_gate_results"]["readiness"] == 0


def test_qualified_service_recomputes_fraction_and_rejects_flag(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7834-SERVICE: only accepted Exp7833 timing opens a bound."""
    old = json.loads((ROOT / "results/experiment_7820_v679_hardware_evidence.json").read_text())
    source = tmp_path / "results/experiment_7820_v679_hardware_evidence.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(old))
    for row in old["board_rows"]:
        path = tmp_path / row["source_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / row["source_path"]).read_bytes())
    service_path = tmp_path / "results/experiment_7833_v680_service_cost.json"
    service = {
        "experiment_id": 7833,
        "schema": "carnot.experiment_7833.v1",
        "run_date": "20260928",
        "service_evidence_ready_score": 1,
        "flagged_adversarial": False,
        "verdict_class": "null",
        "stage_times_ms": {"host_stage": 2, "whole_service": 10},
    }
    service_path.write_text(json.dumps(service))
    audit = build_audit(tmp_path, "20260928", MANIFEST)
    assert audit["service_source_status"] == "qualified"
    assert audit["host_stage_fraction"] == pytest.approx(0.2)
    assert audit["acceleration_bound"] == pytest.approx(1.25)
    assert audit["verdict_class"] == "null"
    assert audit["hardware_advantage_claimed"] is False
    service["flagged_adversarial"] = True
    service_path.write_text(json.dumps(service))
    blocked = build_audit(tmp_path, "20260928", MANIFEST)
    assert blocked["verdict_class"] == "blocked"
    assert blocked["acceleration_bound"] is None
    assert any(check["field"] == "flagged_adversarial" for check in blocked["gate_check_summary"])


def test_board_mutation_is_rejected(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7834-CUSTODY: prior bytes alone cannot authenticate a changed board."""
    old = json.loads((ROOT / "results/experiment_7820_v679_hardware_evidence.json").read_text())
    source = tmp_path / "results/experiment_7820_v679_hardware_evidence.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(old))
    for row in old["board_rows"]:
        path = tmp_path / row["source_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / row["source_path"]).read_bytes())
    target = tmp_path / old["board_rows"][0]["source_path"]
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="board_hash_mismatch"):
        build_audit(tmp_path, "20260928", MANIFEST)


def test_dispatch_rejects_undeclared_and_cli_records_classes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7834-DISPATCH: exact names, argv and classes reach executor."""
    manifest = json.loads(MANIFEST.read_text())
    seen = []

    def record(root: Path, spec: dict, durable: Path) -> dict:
        seen.append((spec["name"], spec["argv"], spec["classification"]))
        return {
            "name": spec["name"],
            "command_argv": spec["argv"],
            "classification": spec["classification"],
            "passed": True,
            "exit_code": 0,
            "log_path": "recorded",
            "log_sha256": "sha256:recorded",
        }

    assert len(dispatch(ROOT, MANIFEST, tmp_path, executor=record)) == len(manifest["commands"])
    assert seen == [(c["name"], c["argv"], c["classification"]) for c in manifest["commands"]]
    changed = deepcopy(manifest)
    changed["commands"][0]["argv"] += ["--surprise"]
    other = tmp_path / "changed.json"
    other.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="undeclared"):
        dispatch(ROOT, other, tmp_path, executor=record)


def test_real_cli_preserves_diagnostic_failure(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7834-DISPATCH: a health failure does not erase blocked science."""
    manifest = json.loads(MANIFEST.read_text())
    manifest["run_root"] = str(tmp_path / "attempt")
    manifest["candidate_path"] = str(tmp_path / "attempt/candidate.json")
    original = json.loads(MANIFEST.read_text())["run_root"]
    for index, spec in enumerate(manifest["commands"]):
        spec["private_root"] = str(tmp_path / f"attempt/command_{index:02d}_{spec['name']}")
        spec["argv"] = [arg.replace(original, manifest["run_root"]) for arg in spec["argv"]]
    frozen = tmp_path / "manifest.json"
    frozen.write_text(json.dumps(manifest))

    def record(root: Path, spec: dict, durable: Path) -> dict:
        failed = spec["name"] == "repository_health"
        return {
            "name": spec["name"],
            "command_argv": spec["argv"],
            "classification": spec["classification"],
            "passed": not failed,
            "exit_code": int(failed),
            "log_path": "recorded",
            "log_sha256": "sha256:recorded",
        }

    output = tmp_path / "out.json"
    assert (
        main(
            ["--date", "20260928", "--manifest", str(frozen), "--output", str(output)],
            executor=record,
        )
        == 0
    )
    result = json.loads(output.read_text())
    assert result["observed_child_commands"] == manifest["commands"]
    assert result["repository_health"]["status"] == "failed_diagnostic"
    assert result["honest_verdict"] == "complete_blocked_missing_service_evidence"
    assert result["validation_receipts"]["required_checks_passed"] is True


def test_sealed_log_retry_and_mutation(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7834-CUSTODY: closed logs get unique paths and cold hashes."""
    command = {
        "name": "probe",
        "argv": [str(ROOT / ".venv/bin/python"), "-c", "print('ok')"],
        "classification": "required",
        "private_root": str(tmp_path / "first/probe"),
        "timeout_s": 30,
    }
    first = run_child(ROOT, command, tmp_path / "sealed")
    command["private_root"] = str(tmp_path / "second/probe")
    second = run_child(ROOT, command, tmp_path / "sealed")
    assert first["passed"] and second["passed"]
    assert first["log_path"] != second["log_path"]
    audit = build_audit(ROOT, "20260928", MANIFEST)
    audit["validation_receipts"] = {"checks": [first]}
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(audit))
    assert cold_reduce(candidate, ROOT, MANIFEST)["row_count"] == 3
    Path(first["log_path"]).write_bytes(b"mutated")
    with pytest.raises(ValueError, match="log_hash_mismatch"):
        cold_reduce(candidate, ROOT, MANIFEST)


def test_required_failure_disqualifies(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7834-DISPATCH: a required exit zeros readiness."""
    manifest = json.loads(MANIFEST.read_text())
    manifest["candidate_path"] = str(tmp_path / "candidate.json")
    frozen = tmp_path / "manifest.json"
    frozen.write_text(json.dumps(manifest))

    def fail(root: Path, spec: dict, durable: Path) -> dict:
        failed = spec["name"] == "ruff_format"
        return {
            "name": spec["name"],
            "command_argv": spec["argv"],
            "classification": spec["classification"],
            "passed": not failed,
            "exit_code": int(failed),
            "log_path": "recorded",
            "log_sha256": "sha256:recorded",
        }

    output = tmp_path / "out.json"
    assert (
        main(
            ["--date", "20260928", "--manifest", str(frozen), "--output", str(output)],
            executor=fail,
        )
        == 0
    )
    result = json.loads(output.read_text())
    assert result["verdict_class"] == "disqualified"
    assert result["acceptance_gate_results"]["readiness"] == 0
    assert result["validation_receipts"]["required_checks_passed"] is False


def _copy_inventory(tmp_path: Path) -> dict:
    old = json.loads((ROOT / "results/experiment_7820_v679_hardware_evidence.json").read_text())
    source = tmp_path / "results/experiment_7820_v679_hardware_evidence.json"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(json.dumps(old))
    for row in old["board_rows"]:
        path = tmp_path / row["source_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / row["source_path"]).read_bytes())
    return old


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("schema", "invalid_exp7820_schema"),
        ("inventory", "board_inventory_mismatch"),
        ("kv260", "kv260_scope_mismatch"),
        ("polarfire", "polarfire_scope_mismatch"),
        ("gatemate", "gatemate_scope_mismatch"),
    ],
)
def test_inventory_scope_guards(tmp_path: Path, change: str, message: str) -> None:
    """SCENARIO-REPORT-7834-CUSTODY: a changed historical scope cannot be carried forward."""
    old = _copy_inventory(tmp_path)
    if change == "schema":
        old["schema"] = "wrong"
    elif change == "inventory":
        old["board_rows"].pop()
    elif change == "kv260":
        old["board_rows"][0]["k_max"] = 15
    elif change == "polarfire":
        next(row for row in old["board_rows"] if row["board"] == "PolarFire")["processor_class"] = (
            "fpga"
        )
    else:
        next(row for row in old["board_rows"] if row["board"] == "GateMate")["blocker"] = None
    (tmp_path / "results/experiment_7820_v679_hardware_evidence.json").write_text(json.dumps(old))
    with pytest.raises(ValueError, match=message):
        build_audit(tmp_path, "20260928", MANIFEST)


def test_missing_inventory_is_named(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7834-CUSTODY: there is no fabricated replacement inventory."""
    with pytest.raises(ValueError, match="missing_exp7820_board_inventory"):
        build_audit(tmp_path, "20260928", MANIFEST)


@pytest.mark.parametrize(
    "bad", ["readiness", "experiment_id", "schema", "verdict", "times", "bool_time"]
)
def test_service_contract_rejects_bad_operands(tmp_path: Path, bad: str) -> None:
    """SCENARIO-REPORT-7834-SERVICE: no malformed science producer opens timing."""
    _copy_inventory(tmp_path)
    data = {
        "experiment_id": 7833,
        "schema": "carnot.experiment_7833.v1",
        "run_date": "20260928",
        "service_evidence_ready_score": 1,
        "flagged_adversarial": False,
        "verdict_class": "null",
        "stage_times_ms": {"host_stage": 2, "whole_service": 10},
    }
    if bad == "readiness":
        data["service_evidence_ready_score"] = 0
    elif bad == "experiment_id":
        data["experiment_id"] = 7832
    elif bad == "schema":
        data["schema"] = "wrong"
    elif bad == "verdict":
        data["verdict_class"] = "blocked"
    elif bad == "times":
        data["stage_times_ms"] = {"host_stage": 11, "whole_service": 10}
    else:
        data["stage_times_ms"] = {"host_stage": True, "whole_service": 10}
    (tmp_path / "results/experiment_7833_v680_service_cost.json").write_text(json.dumps(data))
    audit = build_audit(tmp_path, "20260928", MANIFEST)
    assert audit["verdict_class"] == "blocked"
    assert audit["gate_check_summary"]


def test_whole_host_stage_has_no_finite_speedup_bound(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7834-SERVICE: a fully host-bound service is not a device result."""
    _copy_inventory(tmp_path)
    service = {
        "experiment_id": 7833,
        "schema": "carnot.experiment_7833.v1",
        "service_evidence_ready_score": 1,
        "flagged_adversarial": False,
        "verdict_class": "null",
        "stage_times_ms": {"host_stage": 10, "whole_service": 10},
    }
    (tmp_path / "results/experiment_7833_v680_service_cost.json").write_text(json.dumps(service))
    audit = build_audit(tmp_path, "20260928", MANIFEST)
    assert audit["service_source_status"] == "qualified"
    assert audit["acceleration_bound"] is None


def test_manifest_names_count_and_receipt_are_guarded(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7834-DISPATCH: no child or fabricated receipt enters the roster."""
    original = json.loads(MANIFEST.read_text())

    def record(root: Path, spec: dict, durable: Path) -> dict:
        return {
            "name": "wrong",
            "command_argv": spec["argv"],
            "classification": spec["classification"],
        }

    with pytest.raises(ValueError, match="undeclared child receipt"):
        dispatch(ROOT, MANIFEST, tmp_path, executor=record)
    for mutation, message in [
        ("name", "undeclared child name"),
        ("count", "undeclared child name"),
    ]:
        changed = deepcopy(original)
        if mutation == "name":
            changed["commands"][0]["name"] = "surprise"
        else:
            changed["commands"].append(deepcopy(changed["commands"][-1]))
        path = tmp_path / f"{mutation}.json"
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match=message):
            dispatch(ROOT, path, tmp_path, executor=record)


def test_cold_replay_rejects_rows_gates_and_raw_bytes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7834-CUSTODY: replay opens raw rows and exact gate operands."""
    audit = build_audit(ROOT, "20260928", MANIFEST)
    candidate = tmp_path / "candidate.json"
    raw = tmp_path / "rows.json"
    raw.write_text(json.dumps(audit["rows"]))
    from carnot.reporting.current_work_receipt import sha256_file

    audit["raw_rows_path"] = str(raw)
    audit["raw_rows_sha256"] = sha256_file(raw)
    candidate.write_text(json.dumps(audit))
    assert cold_reduce(candidate, ROOT, MANIFEST)["row_count"] == 3
    changed = deepcopy(audit)
    changed["board_rows"][0]["k_max"] = 6
    candidate.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="cold_replay_mismatch:board_rows"):
        cold_reduce(candidate, ROOT, MANIFEST)
    changed = deepcopy(audit)
    changed["gate_check_summary"] = []
    candidate.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="cold_replay_mismatch:gate_check_summary"):
        cold_reduce(candidate, ROOT, MANIFEST)
    candidate.write_text(json.dumps(audit))
    raw.write_bytes(b"tamper")
    with pytest.raises(ValueError, match="raw_hash_mismatch"):
        cold_reduce(candidate, ROOT, MANIFEST)
    raw.write_text(json.dumps([{"wrong": True}]))
    audit["raw_rows_sha256"] = sha256_file(raw)
    candidate.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="raw_row_mismatch"):
        cold_reduce(candidate, ROOT, MANIFEST)


def test_cli_prepare_and_cold_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7834-DISPATCH: the real entrypoint can prepare and replay."""
    prepared = tmp_path / "prepared.json"
    assert main(["--date", "20260928", "--prepare", str(prepared)]) == 0
    assert prepared.is_file()
    assert main(["--date", "20260928", "--cold-replay", str(prepared)]) == 0


def test_cli_import_is_inert() -> None:
    """SCENARIO-REPORT-7834-DISPATCH: importing the CLI never starts validation."""
    import runpy

    namespace = runpy.run_path(
        str(ROOT / "scripts/experiments/experiment_7834_v680_hardware_evidence.py"),
        run_name="experiment_7834_import_check",
    )
    assert namespace["main"] is main


def test_cold_replay_uses_candidate_attempt_manifest(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7834-CUSTODY: a retry replays its own frozen attempt bytes."""
    manifest = tmp_path / "retry_manifest.json"
    value = json.loads(MANIFEST.read_text())
    prior_root = value["run_root"]
    value["run_root"] = str(tmp_path / "attempt")
    value["candidate_path"] = str(tmp_path / "attempt/candidate.json")
    for command in value["commands"]:
        command["private_root"] = command["private_root"].replace(prior_root, value["run_root"])
        command["argv"] = [arg.replace(prior_root, value["run_root"]) for arg in command["argv"]]
    manifest.write_text(json.dumps(value))
    candidate = tmp_path / "retry_candidate.json"
    candidate.write_text(json.dumps(build_audit(ROOT, "20260928", manifest)))
    assert main(["--cold-replay", str(candidate)]) == 0
    assert main(["--cold-replay", str(candidate), "--manifest", str(manifest)]) == 0


def test_manifest_count_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7834-DISPATCH: an appended command fails even with matching names."""
    import carnot.reporting.experiment_7834_v680_hardware_evidence as module

    changed = json.loads(MANIFEST.read_text())
    changed["commands"].append(deepcopy(changed["commands"][-1]))
    monkeypatch.setattr(
        module, "COMMAND_NAMES", (*module.COMMAND_NAMES, changed["commands"][-1]["name"])
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="undeclared child count"):
        dispatch(ROOT, path, tmp_path)


def test_real_cli_cold_seal_and_raw_immutability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7834-CUSTODY: publication cold-checks and cannot overwrite raw bytes."""
    import carnot.reporting.experiment_7834_v680_hardware_evidence as module
    from carnot.reporting.current_work_receipt import canonical_hash, sha256_file

    _copy_inventory(tmp_path)
    manifest = json.loads(MANIFEST.read_text())
    manifest["candidate_path"] = str(tmp_path / "candidate.json")
    private_manifest = tmp_path / "manifest.json"
    private_manifest.write_text(json.dumps(manifest))
    log = tmp_path / "sealed.log"
    log.write_bytes(b"closed\n")

    def record(
        root: Path, path: Path, durable: Path, executor: object, before_child: object
    ) -> list[dict]:
        return [
            {
                "name": c["name"],
                "command_argv": c["argv"],
                "classification": c["classification"],
                "passed": True,
                "exit_code": 0,
                "log_path": str(log),
                "log_sha256": sha256_file(log),
            }
            for c in manifest["commands"]
        ]

    monkeypatch.setattr(module, "dispatch", record)
    output = tmp_path / "out.json"
    assert (
        main(
            [
                "--date",
                "20260928",
                "--root",
                str(tmp_path),
                "--manifest",
                str(private_manifest),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    audit = json.loads(output.read_text())
    assert audit["raw_rows_sha256"] == sha256_file(Path(audit["raw_rows_path"]))
    raw = Path(audit["raw_rows_path"])
    assert raw.name == canonical_hash(audit["rows"]).split(":", 1)[1] + ".json"
    raw.write_bytes(b"changed")
    with pytest.raises(ValueError, match="immutable raw rows changed"):
        main(
            [
                "--date",
                "20260928",
                "--root",
                str(tmp_path),
                "--manifest",
                str(private_manifest),
                "--output",
                str(output),
            ]
        )
