"""REQ-REPORT-7964: failed custody must remain typed and cannot imply placement."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
from coverage import CoverageData

from carnot.reporting import experiment_7964_v690_hardware_evidence as q
from carnot.reporting import validation_7964 as plan
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from scripts.experiments import experiment_7964_v690_hardware_evidence as cli
from test_experiment_7938_v688_hardware_evidence import service
import shutil


def authority(private: Path) -> Path:
    """REQ-HW-7964: copy original custody bytes before a rejection damages them."""
    prior = json.loads((ROOT / q.PRIOR).read_text())
    for name in {
        q.PRIOR,
        prior["terminal_validation_sidecar_path"],
        *prior["source_artifact_hashes"],
    }:
        source = ROOT / name
        if source.is_file():
            target = private / source.relative_to(ROOT) if source.is_relative_to(ROOT) else source
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    return private


ROOT = Path(__file__).resolve().parents[2]


def workload() -> dict[str, Any]:
    """Fixture costs exercise equations without claiming device measurements."""
    value = service()
    value.update(
        experiment_id=7963,
        task_id="exp7963-service-cost",
        milestone="2026.09.690",
        run_date="20261001",
    )
    for row in value["rows"]:
        row.update(start_s=0.0, end_s=row["duration_s"])
    return value


def test_custody(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7964-CUSTODY: optional absence preserves qualified history."""
    value = q.read_evidence(authority(tmp_path), "20261001")
    assert value["experiment_id"] == 7964 and value["milestone"] == "2026.09.690"
    assert value["required_custody_valid"] and value["hardware_evidence_ready_score"] == 1
    assert not value["typed_blocker_rows"] and value["verdict_class"] == "null"
    assert not value["workload_attachment_available"]
    assert value["current_device_execution_count"] == 0
    assert value["MODEL_SPECS"] == [] and not value["hardware_speedup_claimed"]
    assert value["rows"][0]["k_max"] == 5
    assert value["rows"][1]["processor_class"] == "linux_cpu"
    assert value["rows"][2]["blocker"] == "0xffffffff"
    assert all(r["amdahl_estimate_upper_bound"] is None for r in value["operation_map"])
    assert set(value) <= set(value["field_principles"])
    assert q.cold_reduce(tmp_path, value)["row_count"] == 3
    with pytest.raises(ValueError, match="run_date_mismatch"):
        q.read_evidence(tmp_path, "20260929")


@pytest.mark.parametrize("damage", ["missing", "schema", "rows", "row", "scope", "source_hash"])
def test_failed_custody(tmp_path: Path, damage: str) -> None:
    """SCENARIO-REPORT-7964-CUSTODY: each failed operand survives typed reduction."""
    root = authority(tmp_path)
    path = root / q.PRIOR
    value = json.loads(path.read_text())
    if damage == "missing":
        path.unlink()
    elif damage == "schema":
        path.write_text("[]")
    else:
        if damage == "rows":
            value["board_rows"] = None
        elif damage == "row":
            value["board_rows"][0] = None
        else:
            value["board_rows"][0].pop(damage)
        atomic_json(path, value)
    result = q.read_evidence(root, "20261001")
    assert result["verdict_class"] == "blocked" and not result["required_custody_valid"]
    assert result["honest_verdict"].startswith("complete_blocked_")
    assert result["hardware_evidence_ready_score"] == 0
    assert result["typed_blocker_rows"] and len(result["board_rows"]) == 3
    assert all(not row["fabric_compatible"] for row in result["operation_map"])
    assert all(row["placement_constraint"] is None for row in result["operation_map"])
    assert all(row["modeled_100x_kernel_bound"] is None for row in result["operation_map"])
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
        <= set(row)
        for row in result["gate_check_summary"]
    )
    assert q.cold_reduce(root, result)["row_count"] == 3


@pytest.mark.parametrize("board", ["KV260", "PolarFire", "GateMate"])
def test_each_board_alone(tmp_path: Path, board: str) -> None:
    """SCENARIO-REPORT-7964-CUSTODY: one board cannot close three obligations."""
    root = authority(tmp_path)
    path = root / q.PRIOR
    value = json.loads(path.read_text())
    row = next(row for row in value["board_rows"] if row["board"] == board)
    value["board_rows"] = [row]
    atomic_json(path, value)
    assert q.read_evidence(root, "20261001")["verdict_class"] == "blocked"
    mapped = q.operation_map([row], workload()["rows"], "fixture")
    assert len(mapped) == 2 and all(r["board"] == board for r in mapped)
    for missing in ("scope", "source_hash"):
        bad = deepcopy(row)
        bad.pop(missing)
        assert all(
            r["placement_constraint"] is None
            for r in q.operation_map([bad], workload()["rows"], "fixture")
        )


def test_workload(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7964-WORKLOAD: exact rows and compatibility bound gains."""
    root = authority(tmp_path)
    source = workload()
    atomic_json(root / q.SERVICE, source)
    value = q.read_evidence(root, "20261001")
    assert value["workload_attachment_available"]
    neural, quadratic = value["operation_map"][::3]
    assert neural["offload_fraction"] == 0 and neural["amdahl_estimate_upper_bound"] == 1
    assert quadratic["offload_fraction"] == 0.5
    assert quadratic["modeled_100x_kernel_bound"] == pytest.approx(1 / 0.505)
    assert quadratic["amdahl_estimate_upper_bound"] == 2
    assert quadratic["transfer_bytes"] == 128
    assert value["operation_map"][2]["amdahl_estimate_upper_bound"] is None
    for fraction, expected, ideal in ((0, 1, 1), (1, 100, None)):
        source["rows"][1]["kernel_share"] = fraction
        atomic_json(root / q.SERVICE, source)
        row = q.read_evidence(root, "20261001")["operation_map"][3]
        assert row["modeled_100x_kernel_bound"] == expected
        assert row["amdahl_estimate_upper_bound"] == ideal
    source["rows"][1].pop("duration_s")
    source["rows"][1].pop("start_s")
    source["rows"][1].pop("end_s")
    atomic_json(root / q.SERVICE, source)
    assert (
        q.read_evidence(root, "20261001")["operation_map"][3]["modeled_100x_kernel_bound"] is None
    )
    for field, bad in (
        ("rows", [None]),
        ("rows", []),
        ("verdict_class", "disqualified"),
        ("flagged_adversarial", True),
        ("service_measurement_ready_score", 0),
        ("experiment_id", 0),
        ("validation_receipts", {}),
    ):
        atomic_json(root / q.SERVICE, {**workload(), field: bad})
        rejected = q.read_evidence(root, "20261001")
        assert not rejected["workload_attachment_available"]
        assert rejected["hardware_evidence_ready_score"] == 1
    atomic_json(root / q.SERVICE, workload())
    (root / q.PRIOR).unlink()
    assert all(
        r["placement_constraint"] is None
        for r in q.read_evidence(root, "20261001")["operation_map"]
    )


@pytest.mark.parametrize(
    "field",
    [
        "rows",
        "typed_blocker_rows",
        "required_custody_valid",
        "operation_map_validity",
        "operation_map",
        "gate_check_summary",
        "hardware_speedup_claimed",
    ],
)
def test_replay_mutations(tmp_path: Path, field: str) -> None:
    """SCENARIO-REPORT-7964-VALIDATION: fresh reduction rejects claim mutations."""
    value = q.read_evidence(authority(tmp_path), "20261001")
    value[field] = "changed"
    with pytest.raises(ValueError, match="claims_changed"):
        q.cold_reduce(tmp_path, value)


def test_manifest_and_shards(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7964-VALIDATION: exact includes, private routes and dates."""
    commands = plan.manifest(tmp_path, "20261001")
    for spec in commands:
        if spec["name"] in {"e2e_016_fixture", "e2e_016_replay"}:
            assert spec["argv"][spec["argv"].index("--date") + 1] == "20260929"
        assert all(arg == plan.INCLUDE for arg in spec["argv"] if arg.startswith("--include="))
    assert len(plan.terminal_manifest(tmp_path)) == 3
    with pytest.raises(ValueError, match="private_tmp_required"):
        plan.manifest(ROOT / "results", "20261001")
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
def test_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    """SCENARIO-REPORT-7964-VALIDATION: actual readers bind checked primary bytes."""
    from carnot.reporting import publication_7964 as pub

    def qualified(root: Path, date: str, output: Path, raw: Path, **kwargs: Any) -> int:
        assert kwargs["evidence"] is q and kwargs["validation"] is plan
        atomic_json(output, q.read_evidence(root, date))
        return 0

    def child(
        spec: dict[str, Any], private: Path, durable: Path, started: float, units: int
    ) -> dict[str, Any]:
        log = durable / (spec["name"] + ".log")
        atomic_json(log, {"flagged_count": 0})
        return {
            **spec,
            "passed": mode != "validator_failure",
            "log_path": str(log),
            "log_sha256": sha256_file(log),
            "exit_code": 0,
            "timed_out": False,
        }

    monkeypatch.setattr(pub.runner, "qualify", qualified)
    monkeypatch.setattr(pub.runner, "run_child", child)
    if mode == "reader_failure":
        monkeypatch.setattr(pub, "reader_receipt", lambda *a, **k: {"passed": False})
    output = tmp_path / "results/experiment_7964_v690_hardware_evidence.json"
    raw = output.parent / "raw" / output.stem
    if mode != "passing":
        with pytest.raises(ValueError, match="candidate_rejected|reader_identity"):
            q.qualify(ROOT, "20261001", output, raw)
        return
    assert cli.main(["--output", str(output), "--raw-root", str(raw)]) == 0
    value = json.loads(output.read_text())
    receipt = json.loads(Path(value["primary_resolution_receipt"]["path"]).read_text())
    assert receipt["passed"]
    assert receipt["gate_sha256"] == receipt["document_sha256"] == sha256_file(output)
    assert Path(receipt["newer_sidecar_path"]).stat().st_mtime_ns > output.stat().st_mtime_ns
    assert json.loads(Path(value["terminal_validation_sidecar_path"]).read_text())[
        "candidate_sha256"
    ] == sha256_file(output)


def test_cli(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7964-CUSTODY: blocked output is the artifact cold-replayed."""
    output = tmp_path / "success/experiment_7964_evidence.json"
    assert cli.main(["--output", str(output), "--evidence-only"]) == 0
    assert cli.main(["--cold-replay", str(output)]) == 0
    assert cli.main(["--terminal-recheck", str(output)]) == 0
    missing = tmp_path / "blocked/experiment_7964_evidence.json"
    assert (
        cli.main(["--root", str(tmp_path / "absent"), "--output", str(missing), "--evidence-only"])
        == 0
    )
    assert cli.main(["--root", str(tmp_path / "absent"), "--cold-replay", str(missing)]) == 0
    with pytest.raises(SystemExit):
        cli.main(["--date", "20260929"])


@pytest.mark.parametrize(
    "field,bad",
    [
        ("duration_s", -1),
        ("duration_s", True),
        ("kernel_share", -1),
        ("kernel_share", 1.1),
        ("host_fraction", float("nan")),
        ("transfer_bytes", -1),
        ("operation", ""),
    ],
)
def test_bad_costs(tmp_path: Path, field: str, bad: Any) -> None:
    """SCENARIO-REPORT-7964-WORKLOAD: malformed costs cannot support estimates."""
    root = authority(tmp_path)
    source = workload()
    source["rows"][0][field] = bad
    atomic_json(root / q.SERVICE, source)
    assert not q.read_evidence(root, "20261001")["workload_attachment_available"]


@pytest.mark.parametrize(
    "field,bad",
    [
        ("start_s", None),
        ("end_s", 10),
        ("end_s", -1),
        ("transfer_bytes", None),
        ("transfer_bytes", True),
    ],
)
def test_span_authentication(tmp_path: Path, field: str, bad: Any) -> None:
    """SCENARIO-REPORT-7964-WORKLOAD: timings require spans and integer bytes."""
    root = authority(tmp_path)
    source = workload()
    source["rows"][0][field] = bad
    atomic_json(root / q.SERVICE, source)
    result = q.read_evidence(root, "20261001")
    assert not result["workload_attachment_available"]
    assert result["hardware_evidence_ready_score"] == 1


def test_changed_receipt(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7964-CUSTODY: changed physical bytes cannot reopen scope."""
    root = authority(tmp_path)
    prior = json.loads((root / q.PRIOR).read_text())
    name = next(iter(prior["terminal_receipt_hashes"]))
    (root / name).write_text("{}")
    result = q.read_evidence(root, "20261001")
    assert result["verdict_class"] == "blocked"
    assert result["changed_physical_receipts"] == []
    assert all(row["scope"] is None for row in result["board_rows"])
    assert q.cold_reduce(root, result)["row_count"] == 3


def test_current_runtime_identity(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7964-VALIDATION: old runtime pointers remain historical."""
    result = q.read_evidence(authority(tmp_path), "20261001")
    assert result["current_execution_date"] == "20261001"
    assert "raw_rows_path" not in result and "validation_evidence_files" not in result
    assert result["historical_fixture_date"] == "20260929"
    assert result["cited_upstream_artifacts"][0]["experiment_id"] == 7951
