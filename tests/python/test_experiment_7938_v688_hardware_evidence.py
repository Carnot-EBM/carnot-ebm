"""REQ-REPORT-7938-V688: workload estimates cannot expand board evidence."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
from typing import Any

from coverage import CoverageData
import pytest

from carnot.reporting import experiment_7938_v688_hardware_evidence as q
from carnot.reporting import validation_7938 as plan
from carnot.reporting import qualification_7926 as runner
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from scripts.experiments import experiment_7938_v688_hardware_evidence as cli

ROOT = Path(__file__).resolve().parents[2]


def authority(private: Path) -> Path:
    """Copy exact upstream bytes before a rejection test damages its authority."""
    prior = json.loads((ROOT / q.PRIOR).read_text())
    for name in {q.PRIOR, *prior["source_artifact_hashes"]}:
        source = ROOT / name
        if source.is_file():
            target = private / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    return private


def service() -> dict[str, Any]:
    """Known timings test reduction mechanics, not measured hardware benefit."""
    return {
        "experiment_id": 7937,
        "task_id": "exp7937-service-cost",
        "milestone": "2026.09.688",
        "run_date": "20260930",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "service_measurement_ready_score": 1,
        "validation_receipts": {"required_checks_passed": True},
        "rows": [
            {
                "operation": "source_head_forward",
                "duration_s": 0.01,
                "kernel_share": 0.2,
                "host_fraction": 0.8,
                "transfer_bytes": 64,
                "topology": "sparse",
                "representation": "neural_head",
            },
            {
                "operation": "energy_evaluation",
                "duration_s": 0.02,
                "kernel_share": 0.5,
                "host_fraction": 0.5,
                "transfer_bytes": 128,
                "topology": "sparse",
                "representation": "quadratic_ising",
                "k_max": 5,
            },
        ],
    }


def test_custody(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7938-CUSTODY: optional workload failure preserves custody."""
    value = q.read_evidence(authority(tmp_path), "20260930")
    assert (value["experiment_id"], value["task_id"], value["milestone"]) == (
        7938,
        "exp7938-hardware-evidence",
        "2026.09.688",
    )
    assert value["verdict_class"] == "null", value["gate_check_summary"]
    assert value["hardware_evidence_ready_score"] == 1
    assert [r["board"] for r in value["rows"]] == ["KV260", "PolarFire", "GateMate"]
    assert value["rows"][0]["k_max"] == 5
    assert value["rows"][1]["processor_class"] == "linux_cpu"
    assert value["rows"][2]["blocker"] == "0xffffffff"
    assert not value["workload_attachment_available"]
    assert value["operation_map"] and not value["measured_offload_ceiling_available"]
    assert value["historical_required_failures"]
    assert value["current_device_execution_count"] == 0
    assert not value["hardware_speedup_claimed"] and value["MODEL_SPECS"] == []
    assert set(value) <= set(value["field_principles"])
    assert q.cold_reduce(tmp_path, value)["row_count"] == 3
    with pytest.raises(ValueError, match="run_date_mismatch"):
        q.read_evidence(tmp_path, "20260929")


@pytest.mark.parametrize("damage", ["missing", "schema", "identity", "terminal", "receipt"])
def test_custody_failure(tmp_path: Path, damage: str) -> None:
    """SCENARIO-REPORT-7938-CUSTODY: source failures have exact gate operands."""
    root = authority(tmp_path)
    path = root / q.PRIOR
    if damage == "missing":
        path.unlink()
    elif damage == "schema":
        path.write_text("[]")
    else:
        value = json.loads(path.read_text())
        if damage == "identity":
            value["experiment_id"] = 0
        elif damage == "terminal":
            value["terminal_validation_sidecar_path"] = str(tmp_path / "absent.json")
        else:
            value["terminal_receipt_hashes"][next(iter(value["terminal_receipt_hashes"]))] = "wrong"
        atomic_json(path, value)
    value = q.read_evidence(root, "20260930")
    assert value["verdict_class"] == "blocked"
    assert value["honest_verdict"].startswith("complete_blocked_")
    assert all(
        {
            "upstream_id",
            "artifact_path",
            "artifact_hash",
            "artifact_field",
            "op",
            "expected",
            "observed",
        }
        <= set(r)
        for r in value["gate_check_summary"]
    )


def test_workload(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7938-WORKLOAD: explicit representation limits offload."""
    root = authority(tmp_path)
    original = service()
    atomic_json(root / q.SERVICE, original)
    value = q.read_evidence(root, "20260930")
    assert value["workload_attachment_available"]
    assert len(value["workload_feasibility_rows"]) == 6
    neural, quadratic = value["workload_feasibility_rows"][::3]
    assert neural["offload_fraction"] == 0
    assert neural["amdahl_estimate_upper_bound"] == 1
    assert quadratic["offload_fraction"] == 0.5
    assert quadratic["amdahl_estimate_upper_bound"] == 2
    assert all(r["comparison_class"] == "estimate" for r in value["workload_feasibility_rows"])
    assert value["workload_feasibility_rows"][1]["offload_fraction"] == 0
    assert value["workload_feasibility_rows"][2]["amdahl_estimate_upper_bound"] is None
    assert q.cold_reduce(root, value)["row_count"] == 3
    for key, bad in [
        ("rows", []),
        ("rows", [None]),
        ("run_date", "20260929"),
        ("flagged_adversarial", True),
        ("verdict_class", "disqualified"),
        ("validation_receipts", {}),
        ("service_measurement_ready_score", 0),
    ]:
        atomic_json(root / q.SERVICE, {**original, key: bad})
        rejected = q.read_evidence(root, "20260930")
        assert not rejected["workload_attachment_available"]
        assert rejected["hardware_evidence_ready_score"] == 1


@pytest.mark.parametrize(
    "field,bad",
    [
        ("duration_s", -1),
        ("duration_s", float("nan")),
        ("kernel_share", 1),
        ("kernel_share", True),
        ("transfer_bytes", -1),
    ],
)
def test_invalid_measurement(tmp_path: Path, field: str, bad: Any) -> None:
    """SCENARIO-REPORT-7938-WORKLOAD: malformed timing cannot support estimates."""
    root = authority(tmp_path)
    value = service()
    value["rows"][0][field] = bad
    atomic_json(root / q.SERVICE, value)
    assert not q.read_evidence(root, "20260930")["workload_attachment_available"]


def test_missing_share_and_large_k(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7938-WORKLOAD: unknown shares and large chains stay bounded."""
    root = authority(tmp_path)
    source = service()
    source["rows"][0].pop("kernel_share")
    source["rows"][1]["k_max"] = 6
    atomic_json(root / q.SERVICE, source)
    value = q.read_evidence(root, "20260930")
    assert value["workload_feasibility_rows"][0]["amdahl_estimate_upper_bound"] is None
    assert value["workload_feasibility_rows"][3]["offload_fraction"] == 0


@pytest.mark.parametrize(
    "field",
    [
        "rows",
        "operation_map",
        "source_artifact_hashes",
        "measured_offload_ceiling_available",
        "hardware_speedup_claimed",
        "gate_check_summary",
        "sample_size_budget",
        "terminal_receipt_hashes",
        "workload_attachment_operands",
    ],
)
def test_negative_replay(tmp_path: Path, field: str) -> None:
    """SCENARIO-REPORT-7938-TERMINAL: cold reduction rejects altered claims."""
    value = q.read_evidence(authority(tmp_path), "20260930")
    value[field] = "changed"
    with pytest.raises(ValueError, match="claims_changed"):
        q.cold_reduce(tmp_path, value)


def test_manifest_and_coverage(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7938-VALIDATION: freeze dates, failures and coverage scope."""
    commands = plan.manifest(tmp_path, "20260930")
    for spec in commands:
        if spec["name"] in {"e2e_016_fixture", "e2e_016_replay"}:
            assert spec["argv"][spec["argv"].index("--date") + 1] == "20260929"
        for arg in spec["argv"]:
            if arg.startswith("--include="):
                assert arg == plan.INCLUDE
    assert len(plan.terminal_manifest(tmp_path)) == 3
    with pytest.raises(ValueError, match="private_tmp_required"):
        plan.manifest(ROOT / "results", "20260930")
    with pytest.raises(ValueError, match="empty_coverage"):
        q.check_coverage_shards([])
    for mode in ("missing", "empty", "foreign", "valid"):
        path = tmp_path / (mode + ".coverage")
        if mode != "missing":
            data = CoverageData(basename=str(path))
            data.add_lines(
                {
                    str(ROOT / plan.MODULE) if mode != "foreign" else "/tmp/foreign.py": {1}
                    if mode == "valid"
                    else set()
                }
            )
            data.write()
        if mode == "valid":
            q.check_coverage_shards([path])
        else:
            with pytest.raises(ValueError, match="empty_coverage"):
                q.check_coverage_shards([path])


@pytest.mark.parametrize("mode", ["passing", "validator_failure", "reader_failure"])
def test_private_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    """SCENARIO-REPORT-7938-TERMINAL: real consumers read the checked primary."""

    def qualified(root: Path, date: str, output: Path, raw: Path, **kwargs: Any) -> int:
        assert kwargs["evidence"] is q and kwargs["validation"] is plan
        value = q.read_evidence(root, date)
        value["validation_receipts"]["required_checks_passed"] = True
        atomic_json(output, value)
        return 0

    def child(
        spec: dict[str, Any], private: Path, durable: Path, started: float, units: int
    ) -> dict[str, Any]:
        log = durable / (spec["name"] + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(json.dumps({"flagged_count": 0}))
        return {
            **spec,
            "passed": mode != "validator_failure",
            "log_path": str(log),
            "log_sha256": sha256_file(log),
            "exit_code": 0,
            "timed_out": False,
        }

    monkeypatch.setattr(runner, "qualify", qualified)
    monkeypatch.setattr(runner, "run_child", child)
    output = tmp_path / "results/experiment_7938_v688_hardware_evidence.json"
    raw = tmp_path / "results/raw/experiment_7938_v688_hardware_evidence"
    if mode == "reader_failure":
        monkeypatch.setattr(q, "reader_receipt", lambda *a, **k: {"passed": False})
    if mode != "passing":
        with pytest.raises(ValueError, match="candidate_rejected|reader_identity"):
            q.qualify(ROOT, "20260930", output, raw)
        return
    assert q.qualify(ROOT, "20260930", output, raw) == 0
    value = json.loads(output.read_text())
    receipt = json.loads(Path(value["primary_resolution_receipt"]["path"]).read_text())
    assert receipt["passed"]
    assert receipt["gate_sha256"] == receipt["document_sha256"] == sha256_file(output)
    assert receipt["gate_path"] == receipt["document_path"] == str(output)
    terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_text())
    assert terminal["candidate_sha256"] == sha256_file(output)
    monkeypatch.setattr(cli, "qualify", q.qualify)
    assert cli.main(["--output", str(output), "--raw-root", str(raw)]) == 0


def test_cli(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7938-VALIDATION: private CLI replay preserves zero compute."""
    output = tmp_path / "fixture.json"
    assert cli.main(["--output", str(output), "--evidence-only"]) == 0
    assert cli.main(["--cold-replay", str(output)]) == 0
    assert cli.main(["--terminal-recheck", str(output)]) == 0
    with pytest.raises(SystemExit):
        cli.main(["--date", "20260929"])
