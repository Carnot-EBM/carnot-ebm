"""REQ-REPORT-7926-V687: preserve custody without issuing device commands."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import time
from typing import Any

from coverage import CoverageData
import pytest

from carnot.reporting import experiment_7926_v687_hardware_evidence as q
from carnot.reporting import validation_7926 as plan
from carnot.reporting import qualification_7926 as runner
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from scripts.experiments import experiment_7926_v687_hardware_evidence as cli

ROOT = Path(__file__).resolve().parents[2]


def authority(private: Path) -> Path:
    """Copy inputs so rejection tests cannot change the historical authorities."""
    prior = json.loads((ROOT / q.PRIOR).read_text())
    for name in {q.PRIOR, *prior["source_artifact_hashes"]}:
        source = ROOT / name
        if source.is_file():
            target = private / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    return private


def test_custody(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7926-CUSTODY: all three original limits remain binding."""
    value = q.read_evidence(authority(tmp_path), "20260930")
    assert (value["experiment_id"], value["task_id"], value["milestone"]) == (
        7926,
        "exp7926-hardware-evidence",
        "2026.09.687",
    )
    assert value["verdict_class"] == "null", value["gate_check_summary"]
    assert value["hardware_evidence_ready_score"] == 1
    assert [r["board"] for r in value["rows"]] == ["KV260", "PolarFire", "GateMate"]
    assert value["rows"][0]["k_max"] == 5
    assert value["rows"][1]["processor_class"] == "linux_cpu"
    assert value["rows"][2]["blocker"] == "0xffffffff"
    assert value["MODEL_SPECS"] == [] and value["current_device_execution_count"] == 0
    assert value["claim_scope"]["AMD_NPU"] == "unqualified"
    assert any(r["name"] == "e2e_016_fixture" for r in value["historical_required_failures"])
    assert not value["workload_attachment_available"]
    assert q.cold_reduce(tmp_path, value)["row_count"] == 3
    assert set(value) <= set(value["field_principles"])
    with pytest.raises(ValueError, match="run_date_mismatch"):
        q.read_evidence(tmp_path, "20260929")


@pytest.mark.parametrize("damage", ["missing", "malformed", "identity", "log"])
def test_custody_failure(tmp_path: Path, damage: str) -> None:
    """SCENARIO-REPORT-7926-CUSTODY: missing and changed evidence must block."""
    root = authority(tmp_path)
    path = root / q.PRIOR
    if damage == "missing":
        path.unlink()
    elif damage == "malformed":
        path.write_text("[]")
    else:
        value = json.loads(path.read_text())
        if damage == "identity":
            value["experiment_id"] = 0
        else:
            value["validation_receipts"]["checks"][0]["log_sha256"] = "sha256:wrong"
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


def service() -> dict[str, Any]:
    """The fixture supplies full-service timing rather than a board benchmark."""
    return {
        "experiment_id": 7925,
        "task_id": "exp7925-service-cost",
        "run_date": "20260930",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "service_measurement_ready_score": 1,
        "validation_receipts": {"required_checks_passed": True},
        "rows": [
            {
                "operation": "verify",
                "duration_s": 0.01,
                "transfer_bytes": 64,
                "kernel_share": 0.2,
                "host_fraction": 0.8,
                "topology": "sparse",
            }
        ],
    }


def test_workload(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7926-WORKLOAD: estimates cannot qualify new hardware."""
    root = authority(tmp_path)
    path = root / q.SERVICE
    original = service()
    atomic_json(path, original)
    value = q.read_evidence(root, "20260930")
    assert value["workload_attachment_available"]
    assert len(value["workload_feasibility_rows"]) == 3
    row = value["workload_feasibility_rows"][0]
    assert row["amdahl_estimate_upper_bound"] == 1.25
    assert row["comparison_class"] == "estimate"
    assert row["speedup"] is None and not value["hardware_speedup_claimed"]
    assert q.cold_reduce(root, value)["row_count"] == 3
    for key, bad in [
        ("flagged_adversarial", True),
        ("run_date", "20260929"),
        ("rows", []),
        ("rows", [None]),
        ("validation_receipts", {}),
        ("verdict_class", "disqualified"),
    ]:
        atomic_json(path, {**original, key: bad})
        rejected = q.read_evidence(root, "20260930")
        assert not rejected["workload_attachment_available"]
        assert rejected["hardware_evidence_ready_score"] == 1
    path.write_text("[]")
    assert not q.read_evidence(root, "20260930")["workload_attachment_available"]
    atomic_json(path, {**original, "rows": [{"operation": "verify", "duration_s": 1}]})
    assert (
        q.read_evidence(root, "20260930")["workload_feasibility_rows"][0][
            "amdahl_estimate_upper_bound"
        ]
        is None
    )


@pytest.mark.parametrize(
    "field",
    [
        "rows",
        "sample_size_budget",
        "current_device_execution_count",
        "hardware_speedup_claimed",
        "claim_scope",
        "gate_check_summary",
    ],
)
def test_replay_rejects_drift(tmp_path: Path, field: str) -> None:
    """SCENARIO-REPORT-7926-TERMINAL: a cold reducer rejects widened claims."""
    root = authority(tmp_path)
    value = q.read_evidence(root, "20260930")
    value[field] = "changed"
    with pytest.raises(ValueError, match="claims_changed"):
        q.cold_reduce(root, value)


def test_manifest(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7926-VALIDATION: fixture dates differ from execution dates."""
    commands = plan.manifest(tmp_path, "20260930")
    for spec in commands:
        if spec["name"] in {"e2e_016_fixture", "e2e_016_replay"}:
            assert spec["argv"][spec["argv"].index("--date") + 1] == "20260929"
        if spec["name"] == "e2e_016_wrong_date":
            assert spec["expected_exit"] == 1
            assert spec["required_reason"] == "run_date_mismatch"
        if "--include=" in " ".join(spec["argv"]):
            assert next(x for x in spec["argv"] if x.startswith("--include=")) == plan.INCLUDE
    assert set(plan.MEASURED) <= set(next(x for x in commands if x["name"] == "ruff_check")["argv"])
    assert next(x for x in commands if x["name"] == "full_pytest")["classification"] == "diagnostic"
    assert next(x for x in commands if x["name"] == "scoped_spec")["argv"][
        -len(plan.TESTS) :
    ] == list(plan.TESTS)
    assert len(plan.terminal_manifest(tmp_path)) == 3
    with pytest.raises(ValueError, match="private_tmp_required"):
        plan.manifest(ROOT / "results", "20260930")


def test_coverage(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7926-VALIDATION: coverage files must contain owned lines."""
    with pytest.raises(ValueError, match="empty_coverage"):
        q.check_coverage_shards([])
    for mode in ("missing", "empty", "foreign"):
        path = tmp_path / f"{mode}.coverage"
        if mode != "missing":
            data = CoverageData(basename=str(path))
            data.add_lines(
                {str(ROOT / plan.MODULE) if mode == "empty" else "/tmp/foreign.py": set()}
            )
            data.write()
        with pytest.raises(ValueError, match="empty_coverage"):
            q.check_coverage_shards([path])
    path = tmp_path / "nonempty.coverage"
    data = CoverageData(basename=str(path))
    data.add_lines({str(ROOT / plan.MODULE): {1}})
    data.write()
    q.check_coverage_shards([path])


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
def test_qualification(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    """SCENARIO-REPORT-7926-TERMINAL: checked bytes and required failures govern readiness."""
    counts: dict[str, int] = {}

    def child(
        spec: dict[str, Any], private: Path, durable: Path, started: float, units: int
    ) -> dict[str, Any]:
        name = spec["name"]
        counts[name] = counts.get(name, 0) + 1
        if name == "cli_success":
            atomic_json(private / "success.json", q.read_evidence(ROOT, "20260930"))
        if name == "coverage_json":
            atomic_json(
                private / "coverage.json",
                {
                    "files": {
                        str(ROOT / p): {"summary": {"covered_lines": 1, "num_statements": 1}}
                        for p in plan.MEASURED
                    }
                },
            )
        text = (
            json.dumps({"flagged_count": int(mode == "first_flag" and counts[name] == 1)})
            if name == "adversarial_verify"
            else "ok"
        )
        if mode == "invalid_json" and name == "adversarial_verify" and counts[name] == 1:
            text = "malformed"
        log = private / f"{name}.log"
        log.write_text(text)
        sealed = runner.old.seal(log, durable / "logs" / name)
        bad = (mode == "required_failure" and name == "changed_coverage") or (
            mode == "terminal_failure" and name == "strict_rows"
        )
        return {
            **spec,
            "passed": not bad,
            "exit_code": 2 if bad else spec["expected_exit"],
            "timed_out": False,
            "duration_s": 0.001,
            "log_path": str(sealed),
            "log_sha256": sha256_file(sealed),
        }

    monkeypatch.setattr(runner, "run_child", child)
    output = tmp_path / "final.json"
    if mode == "publication_failure":
        real_hash = runner.sha256_file
        monkeypatch.setattr(
            runner,
            "sha256_file",
            lambda p: "wrong" if p.name.endswith(".checked.tmp") else real_hash(p),
        )
        with pytest.raises(ValueError, match="publication_hash_changed"):
            runner.qualify(ROOT, "20260930", output, tmp_path / "raw")
        assert not output.exists()
        return
    if mode == "terminal_failure":
        with pytest.raises(ValueError, match="terminal_validation_failed"):
            runner.qualify(ROOT, "20260930", output, tmp_path / "raw")
        assert not output.exists()
        return
    assert runner.qualify(ROOT, "20260930", output, tmp_path / "raw") == 0
    value = json.loads(output.read_text())
    assert value["verdict_class"] == ("null" if mode == "passing" else "disqualified")
    assert value["hardware_evidence_ready_score"] == int(mode == "passing")
    assert value["flagged_adversarial"] is False
    sidecar = json.loads(Path(value["terminal_validation_sidecar_path"]).read_text())
    assert sidecar["candidate_sha256"] == sha256_file(output)
    assert all(r["passed"] for r in sidecar["reports"])
    with pytest.raises(ValueError, match="worktree root"):
        runner.qualify(tmp_path, "20260930", output, tmp_path / "raw")
    monkeypatch.setattr(cli, "qualify", runner.qualify)
    assert (
        cli.main(
            ["--output", str(tmp_path / "via-cli.json"), "--raw-root", str(tmp_path / "via-raw")]
        )
        == 0
    )


def test_cli_private(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7926-VALIDATION: direct private reduction, replay and date rejection."""
    output = tmp_path / "private.json"
    assert cli.main(["--output", str(output), "--evidence-only"]) == 0
    assert cli.main(["--cold-replay", str(output)]) == 0
    assert cli.main(["--terminal-recheck", str(output)]) == 0
    with pytest.raises(SystemExit) as error:
        cli.main(["--date", "20260929"])
    assert error.value.code == 2
