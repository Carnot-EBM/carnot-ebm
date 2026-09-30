"""REQ-REPORT-7913-V686: guard-safe private validation and dated custody."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

from coverage import CoverageData
import pytest

from carnot.reporting import experiment_7913_v686_hardware_evidence as q
from carnot.reporting import validation_7913 as plan
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from scripts.experiments import experiment_7913_v686_hardware_evidence as cli

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def private() -> Any:
    """Keep every fixture off the guarded results filesystem."""
    with tempfile.TemporaryDirectory(prefix="carnot-7913-test-", dir="/tmp") as folder:
        yield Path(folder)


def authority(private: Path) -> Path:
    """Copy history before damaging a source; historical authorities stay fixed."""
    prior = json.loads((ROOT / q.PRIOR).read_text())
    names = {q.PRIOR, *prior["source_artifact_hashes"]}
    for name in names:
        source = ROOT / name
        if source.is_file():
            target = private / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    return private


def test_scenario_report_7913_custody(private: Path) -> None:
    """SCENARIO-REPORT-7913-CUSTODY: preserve scope and exact failure history."""
    root = authority(private)
    result = q.read_evidence(root, "20260930")
    assert result["experiment_id"] == 7913
    assert result["task_id"] == "exp7913-hardware-evidence"
    assert result["milestone"] == "2026.09.686"
    assert result["verdict_class"] == "null", result["gate_check_summary"]
    assert result["hardware_evidence_ready_score"] == 1
    assert result["current_device_execution_count"] == 0
    assert result["MODEL_SPECS"] == []
    assert [row["board"] for row in result["rows"]] == ["KV260", "PolarFire", "GateMate"]
    assert result["board_rows"][0]["k_max"] == 5
    assert result["board_rows"][1]["processor_class"] == "linux_cpu"
    assert result["board_rows"][2]["blocker"] == "0xffffffff"
    assert all(row["custody_date"] == "20260930" for row in result["rows"])
    assert len(result["historical_7901_affected_failures"]) == 5
    assert result["historical_7901_coverage"] == {"covered": 141, "statements": 218}
    assert result["workload_attachment_available"] is False
    assert q.cold_reduce(root, result)["row_count"] == 3
    assert set(result) <= set(result["field_principles"])
    assert result["sample_size_budget"]["independent"] == 0


@pytest.mark.parametrize("damage", ["missing", "malformed", "changed", "schema"])
def test_scenario_report_7913_sources(private: Path, damage: str) -> None:
    """SCENARIO-REPORT-7913-CUSTODY: missing bytes differ from wrong bytes."""
    root = authority(private)
    path = root / q.PRIOR
    if damage == "missing":
        path.unlink()
    elif damage == "malformed":
        path.write_text("{")
    elif damage == "schema":
        path.write_text("[]")
    else:
        row = json.loads(path.read_text())
        row["verdict_class"] = "positive"
        atomic_json(path, row)
    result = q.read_evidence(root, "20260930")
    assert result["verdict_class"] == "blocked"
    assert result["honest_verdict"].startswith("complete_blocked_")
    assert result["gate_check_summary"]
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
        <= set(x)
        for x in result["gate_check_summary"]
    )


def test_scenario_report_7913_workload(private: Path) -> None:
    """SCENARIO-REPORT-7913-CUSTODY: optional service cannot widen board claims."""
    root = authority(private)
    service = root / q.SERVICE
    value = {
        "experiment_id": 7912,
        "task_id": "exp7912-service-cost",
        "run_date": "20260930",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "service_measurement_ready_score": 1,
        "validation_receipts": {"required_checks_passed": True},
        "rows": [
            {"operation": "verify", "transfer_bytes": 64, "host_fraction": 0.8, "duration_s": 0.002}
        ],
    }
    atomic_json(service, value)
    good = q.read_evidence(root, "20260930")
    assert good["workload_attachment_available"] is True
    assert len(good["workload_feasibility_rows"]) == 3
    assert all(x["hardware_execution_measured"] is False for x in good["workload_feasibility_rows"])
    assert q.cold_reduce(root, good)["row_count"] == 3
    for key, bad in (
        ("flagged_adversarial", True),
        ("rows", []),
        ("rows", [None]),
        ("run_date", "20260929"),
        ("service_measurement_ready_score", 0),
        ("validation_receipts", {}),
    ):
        atomic_json(service, {**value, key: bad})
        rejected = q.read_evidence(root, "20260930")
        assert not rejected["workload_attachment_available"]
        assert rejected["hardware_evidence_ready_score"] == 1
        assert rejected["workload_attachment_operands"]
    service.write_text("{")
    assert not q.read_evidence(root, "20260930")["workload_attachment_available"]


@pytest.mark.parametrize(
    "field",
    [
        "rows",
        "terminal_receipt_hashes",
        "workload_feasibility_rows",
        "gate_check_summary",
        "source_artifact_hashes",
        "sample_size_budget",
    ],
)
def test_scenario_report_7913_negative_replay(private: Path, field: str) -> None:
    """SCENARIO-REPORT-7913-PRIVATE: replay reduces primitive claims again."""
    root = authority(private)
    changed = q.read_evidence(root, "20260930")
    changed[field] = "changed"
    with pytest.raises(ValueError, match="claims_changed"):
        q.cold_reduce(root, changed)


def test_scenario_report_7913_private_coverage(private: Path) -> None:
    """SCENARIO-REPORT-7913-PRIVATE: reject empty and foreign owned shards."""
    paths = [private / "missing", private / "empty", private / "foreign", private / "valid"]
    for path, measured in zip(
        paths[1:],
        (
            {str(ROOT / plan.MODULE): set()},
            {str(private / "foreign.py"): {1}},
            {str(ROOT / plan.MODULE): {1}},
        ),
        strict=True,
    ):
        data = CoverageData(basename=str(path))
        data.add_lines(measured)
        data.write()
    for path in paths[:3]:
        with pytest.raises(ValueError, match="empty_coverage_shard"):
            q.check_coverage_shards([path])
    with pytest.raises(ValueError, match="empty_coverage_shards"):
        q.check_coverage_shards([])
    q.check_coverage_shards([paths[3]])
    with pytest.raises(ValueError, match="private_tmp_required"):
        plan.manifest(ROOT / "results", "20260930")
    commands = plan.manifest(private, "20260930")
    for spec in commands:
        assert spec["deadline_s"] <= 180
        assert spec["expected_exit"] in (0, 1)
        for arg in spec["argv"]:
            if arg.startswith(("--basetemp=", "--data-file=")):
                assert Path(arg.split("=", 1)[1]).is_relative_to(private)
        if spec["name"].startswith("e2e_016"):
            assert spec["argv"][spec["argv"].index("--date") + 1] == "20260930"
    assert (
        next(x for x in commands if x["name"] == "negative_replay")["required_reason"]
        == "claims_changed"
    )
    assert next(x for x in commands if x["name"] == "scoped_spec")["argv"][-5:] == [
        plan.TEST,
        *plan.CONSUMERS,
    ]


def test_scenario_report_7913_owned_child(private: Path) -> None:
    """SCENARIO-REPORT-7913-PRIVATE: child environment retains active guards."""
    from carnot.testing.child_results_guard import CHILD_REPO_ROOT_ENV

    code = "import json, os; print(json.dumps({k:os.environ.get(k) for k in ['PYTHONPATH','TMPDIR','COVERAGE_FILE','CARNOT_CHILD_GUARD_REPO_ROOT']}))"
    spec = plan.command("environment", [sys.executable, "-u", "-c", code])
    receipt = q.run_child(spec, private, private / "sealed", time.monotonic(), 0)
    env = json.loads(Path(receipt["log_path"]).read_text())
    assert env["TMPDIR"] == str(private)
    assert env["COVERAGE_FILE"].startswith(str(private))
    assert str(ROOT / "python") in env["PYTHONPATH"]
    assert env[CHILD_REPO_ROOT_ENV] == str(ROOT)
    assert receipt["argv"] == spec["argv"]
    assert receipt["passed"]
    assert sha256_file(Path(receipt["log_path"])) == receipt["log_sha256"]
    bad = q.run_child(
        plan.command(
            "wrong_reason",
            [sys.executable, "-c", "raise SystemExit(1)"],
            expected=1,
            reason="missing_reason",
        ),
        private,
        private / "sealed",
        time.monotonic(),
        1,
    )
    assert not bad["passed"]


def test_scenario_report_7913_real_cli(private: Path) -> None:
    """SCENARIO-REPORT-7913-PRIVATE: drive real guarded private CLI routes."""
    root = authority(private / "authority")
    commands = plan.manifest(private, "20260930")
    for name in ("cli_success", "cli_missing_input", "cold_replay", "negative_replay"):
        spec = deepcopy(next(x for x in commands if x["name"] == name))
        if "--root" in spec["argv"] and name != "cli_missing_input":
            spec["argv"][spec["argv"].index("--root") + 1] = str(root)
        if name == "negative_replay":
            value = json.loads((private / "success.json").read_text())
            value["rows"] = []
            atomic_json(private / "changed.json", value)
        receipt = q.run_child(spec, private, private / "sealed", time.monotonic(), 0)
        assert receipt["passed"], receipt["output_tail"]
    assert json.loads((private / "missing.json").read_text())["verdict_class"] == "blocked"
    assert (
        cli.main(
            [
                "--date",
                "20260930",
                "--root",
                str(root),
                "--output",
                str(private / "direct.json"),
                "--evidence-only",
            ]
        )
        == 0
    )
    assert (
        cli.main(
            [
                "--date",
                "20260930",
                "--root",
                str(root),
                "--cold-replay",
                str(private / "direct.json"),
            ]
        )
        == 0
    )


@pytest.mark.parametrize(
    "mode",
    [
        "passing",
        "required_failure",
        "first_flag",
        "invalid_json",
        "terminal_failure",
        "publication_failure",
    ],
)
def test_scenario_report_7913_terminal(
    private: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """SCENARIO-REPORT-7913-TERMINAL: bind final bytes and honest readiness."""
    counts: dict[str, int] = {}

    def child(
        spec: dict[str, Any], scratch: Path, durable: Path, started: float, units: int
    ) -> dict[str, Any]:
        assert scratch.is_relative_to(Path("/tmp")) and not scratch.is_relative_to(ROOT / "results")
        name = spec["name"]
        counts[name] = counts.get(name, 0) + 1
        if name == "cli_success":
            atomic_json(scratch / "success.json", q.read_evidence(ROOT, "20260930"))
        if name == "adversarial_verify":
            content = (
                "malformed"
                if mode == "invalid_json" and counts[name] == 1
                else json.dumps({"flagged_count": int(mode == "first_flag" and counts[name] == 1)})
            )
        else:
            content = "claims_changed" if name == "negative_replay" else "ok"
        log = scratch / f"{name}.log"
        log.write_text(content)
        sealed = q.seal(log, durable / "logs" / name)
        exit_code = (
            2
            if (mode == "required_failure" and name == "changed_coverage")
            or (mode == "terminal_failure" and name == "strict_rows")
            else spec["expected_exit"]
        )
        return {
            **spec,
            "exit_code": exit_code,
            "passed": exit_code == spec["expected_exit"],
            "timed_out": False,
            "duration_s": 0.001,
            "log_path": str(sealed),
            "log_sha256": sha256_file(sealed),
        }

    monkeypatch.setattr(q, "run_child", child)
    output = private / "final.json"
    if mode == "publication_failure":
        copy = shutil.copyfile

        def damaged_copy(source: Any, target: Any, **kwargs: Any) -> Any:
            written = copy(source, target, **kwargs)
            if str(target).endswith(".checked.tmp"):
                Path(target).write_text("changed after copy")
            return written

        monkeypatch.setattr(q.shutil, "copyfile", damaged_copy)
        with pytest.raises(ValueError, match="publication_hash_changed"):
            q.qualify(ROOT, "20260930", output, private / "durable")
        assert not output.exists()
        return
    if mode == "terminal_failure":
        with pytest.raises(ValueError, match="terminal_validation_failed"):
            q.qualify(ROOT, "20260930", output, private / "durable")
        assert not output.exists()
        return
    assert q.qualify(ROOT, "20260930", output, private / "durable") == 0
    value = json.loads(output.read_text())
    assert value["verdict_class"] == ("null" if mode == "passing" else "disqualified")
    assert value["flagged_adversarial"] is False
    assert value["hardware_evidence_ready_score"] == int(mode == "passing")
    sidecar = json.loads(Path(value["terminal_validation_sidecar_path"]).read_text())
    assert sidecar["candidate_sha256"] == sha256_file(output)
    assert all(x["passed"] for x in sidecar["reports"])
    assert counts["adversarial_verify"] == (2 if mode in {"first_flag", "invalid_json"} else 1)
    assert (
        cli.main(
            [
                "--date",
                "20260930",
                "--output",
                str(private / "cli-final.json"),
                "--raw-root",
                str(private / "cli-raw"),
            ]
        )
        == 0
    )
    with pytest.raises(ValueError, match="worktree root"):
        q.qualify(private, "20260930", output, private / "raw")


def test_scenario_report_7913_gate_replay(private: Path) -> None:
    """SCENARIO-REPORT-7913-TERMINAL: syntactically valid gate drift also fails."""
    root = authority(private)
    value = q.read_evidence(root, "20260930")
    value["gate_check_summary"] = [{"upstream_id": "different_source"}]
    with pytest.raises(ValueError, match="claims_changed:gate_operands"):
        q.cold_reduce(root, value)
