"""REQ-REPORT-9093-V775 and SCENARIO-REPORT-9093-V775-ACCOUNTING."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest

from carnot.reporting import service_cost_9093 as d
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def source(root: Path, n: int = 9082, **extra: object) -> Path:
    """Small receipts exercise accounting, never scientific performance."""
    path = root / d.SOURCES[n]
    atomic_json(path, dict(experiment_id=n, verdict_class="null", **extra))
    return path


def operation(uid: str = "u", **extra: object) -> dict:
    return dict(
        unit_id=uid,
        arm="static",
        group="g",
        family="graph",
        seed=1,
        operation="generation",
        status="completed",
        scope="science",
        start_ns=10,
        end_ns=30,
        clock_domain="host-boot",
        model="qwen",
        device="cpu",
        parent_id=None,
        **extra,
    )


def test_intervals_and_reduction() -> None:
    """REQ-REPORT-9093-V775: nested clocks pay once and missing clocks stay null."""
    assert d.interval_union([(10, 30), (15, 20), (25, 40), (50, 60)]) == 40
    assert d.interval_union([]) == 0
    rows = [operation(), operation("child")]
    rows[1].update(start_ns=15, end_ns=20, parent_id="u")
    result = d.summarize(rows)
    assert result["observed_latency_summary"][0]["charged_s"] == 20 / 1e9
    assert result["complete_service_ratios"] == []
    assert result["energy_sensor_coverage"]["joules"] is None
    rows[1]["end_ns"] = None
    assert d.summarize(rows)["unavailable_operands"]
    with pytest.raises(ValueError, match="duplicate"):
        d.summarize([rows[0], rows[0]])
    rows[0]["end_ns"] = 0
    with pytest.raises(ValueError, match="clock"):
        d.summarize(rows[:1])


def test_missing_and_blocked_inventory(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9093-V775: a gate diagnostic is not a zero-time service."""
    source(tmp_path, 9083, custody_ready_score=0)
    diagnostic = tmp_path / "results/experiment_9085_collect_a.json"
    atomic_json(
        diagnostic,
        dict(
            experiment=9085,
            schema="blocked_gate_check_v1",
            duration_s=0,
            started_at="now",
            finished_at="now",
            failed_field="custody_ready_score",
            failed_expected=1,
            failed_observed=0,
            failed_upstream="exp9083-prospective-custody",
            failed_evidence_path=str(tmp_path / d.SOURCES[9083]),
            failed_evidence_sha256=d.ref(tmp_path / d.SOURCES[9083])["sha256"],
        ),
    )
    manifest = d.freeze(tmp_path, tmp_path / "private")
    result = d.inventory(manifest)
    assert len(result["invocation_dispositions"]) >= 11
    checks = result["gate_check_summary"]
    assert any(r["observed"] == "missing_file" for r in checks)
    assert any(r["observed"] == 0 and r["expected"] == 1 for r in checks)
    skipped = [r for r in result["rows"] if r["task"] == 9085]
    assert skipped and all(r["latency_s"] is None for r in skipped)
    assert result["complete_service_ratios"] == []
    assert result["unavailable_operands"]
    assert all(r["joules"] is None for r in result["operation_cost_rows"])


def test_hash_and_receipt_authentication(tmp_path: Path) -> None:
    """REQ-REPORT-9093-V775: bind producer, raw clocks and all failed validation."""
    receipt = dict(
        current_run_id="run",
        current_owner_pid=1,
        MODEL_SPECS=[],
        started_monotonic_ns=10,
        ended_monotonic_ns=30,
        duration_s=20 / 1e9,
        execution_venue="host",
        inference_substrate_class="no_model_load",
    )
    source(
        tmp_path,
        current_work_receipts=[receipt],
        validation_receipts=[
            dict(
                name="partial",
                started_monotonic_ns=15,
                ended_monotonic_ns=20,
                exit_code=1,
                passed=False,
            )
        ],
    )
    manifest = d.freeze(tmp_path, tmp_path / "private")
    result = d.inventory(manifest)
    assert any(r["status"] == "failed" for r in result["rows"])
    assert any(r["scope"] == "administrative" for r in result["rows"])
    saved = Path(next(r["path"] for r in manifest["entries"] if r.get("path")))
    saved.chmod(0o600)
    saved.write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        d.inventory(manifest)


def test_fixture_mechanics(tmp_path: Path) -> None:
    """REQ-REPORT-9093-V775: CPU instrumentation is a fixture, never production cost."""
    rows = d.fixture_replay(tmp_path)
    assert rows and all(r["scope"] == "fixture" for r in rows)
    assert all(r["end_ns"] >= r["start_ns"] for r in rows)
    assert d.summarize(rows)["complete_service_ratios"] == []


def test_cli_foreign_replay_and_tamper(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9093-V775: actual private CLI and cold replay need no PYTHONPATH."""
    output = tmp_path / (d.NAME + ".json")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    argv = [str(d.ROOT / ".venv/bin/python"), str(d.ROOT / d.CLI)]
    result = subprocess.run(
        argv + ["--date", "20261115", "--sources", str(tmp_path), "--private-output", str(output)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert value["current_llm_invocations"] == 0
    assert value["cost_inventory_ready_score"] == 0
    replay = subprocess.run(
        argv + ["--cold-replay", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        timeout=60,
    )
    assert replay.returncode == 0, replay.stdout + replay.stderr
    value["rows"] = []
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    assert (
        subprocess.run(
            argv + ["--cold-replay", str(output)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            timeout=60,
        ).returncode
        == 1
    )


def test_producer_hash_and_unpublished_work(tmp_path: Path) -> None:
    """REQ-REPORT-9093-V775: failed original binding and partial validation remain visible."""
    producer = tmp_path / "producer.py"
    producer.write_text("original producer")
    source(tmp_path, code_config_hashes=[dict(path=str(producer), sha256="sha256:wrong")])
    work = tmp_path / "results/raw/experiment_9082_contract/invocations/early/work.json"
    atomic_json(work, dict(start_ns=10, end_ns=20, owner_pid=1))
    result = d.inventory(d.freeze(tmp_path, tmp_path / "private"))
    assert any(
        r["check"] == "producer_hash" and not r["passed"] for r in result["gate_check_summary"]
    )
    assert any(
        r["status"] == "censored" and r["operation"] == "qualification" for r in result["rows"]
    )


def test_runner_validation_and_rehashed_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-9093-V775: command, log and primitive identity all matter."""
    from carnot.reporting import service_cost_runner_9093 as runner

    scope = [dict(name="unit", argv=["test"])]
    work = runner.measure(tmp_path / "missing", tmp_path / "raw", scope)
    assert not runner.qualify(work, {})
    log = tmp_path / "log"
    log.write_text("passed")
    receipt = dict(
        name="unit",
        command_argv=["test"],
        exit_code=0,
        passed=True,
        timed_out=False,
        log_path=str(log),
        log_sha256=d.ref(log)["sha256"],
    )
    validation = dict(receipts=[receipt], coverage={})
    monkeypatch.setattr(runner, "verify_measured", lambda *args: True)
    assert runner.qualify(work, validation)
    changed = deepcopy(validation)
    changed["receipts"][0]["exit_code"] = 1
    assert not runner.qualify(work, changed)
    log.write_text("tampered")
    assert not runner.qualify(work, validation)
    log.write_text("passed")
    value = runner.build(work, validation, False)
    assert value["cost_inventory_ready_score"] == 1
    assert value["complete_service_cost_ready_score"] == 0
    assert runner.build(work, {}, False)["verdict_class"] == "disqualified"
    atomic_json(tmp_path / "raw/validation.json", validation)
    output = tmp_path / (d.NAME + ".json")
    atomic_json(output, value)
    assert runner.replay(output)["passed"]
    bad = deepcopy(value)
    bad["duration_s"] = 999
    atomic_json(output, bad)
    assert not runner.replay(output)["passed"]
    atomic_json(output, value)
    reference = value["raw_shard_hashes"][0]
    path = Path(reference["path"])
    path.chmod(0o600)
    altered = json.loads(path.read_bytes())
    altered["result"]["rows"] = []
    atomic_json(path, altered)
    value["raw_shard_hashes"][0] = d.ref(path)
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    assert not runner.replay(output)["passed"]
    assert not runner.replay(tmp_path / "absent")["passed"]


@pytest.mark.parametrize("qualified", [False, True])
def test_main_validation_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qualified: bool
) -> None:
    """REQ-REPORT-9093-V775: run production orchestration without recursively launching tests."""
    from carnot.reporting import service_cost_runner_9093 as runner

    private = tmp_path / "private"
    private.mkdir()
    monkeypatch.setattr(runner.tempfile, "mkdtemp", lambda **kwargs: str(private))
    monkeypatch.setattr(runner, "run_commands", lambda *args, **kwargs: [])
    monkeypatch.setattr(runner, "qualify", lambda *args: qualified)
    monkeypatch.setattr(runner, "measured", lambda *args: {})
    (private / "checks").mkdir()
    (private / "checks/.coverage").write_text("test-only orchestration sentinel")
    output = tmp_path / "output" / (d.NAME + ".json")
    original = d.ROOT
    monkeypatch.setattr(d, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "plan", lambda p: [])
    assert runner.main(["--date", "20261115", "--sources", str(tmp_path / "absent")]) == (
        0 if qualified else 1
    )
    produced = tmp_path / "results" / (d.NAME + ".json")
    assert produced.is_file() is qualified
    monkeypatch.setattr(d, "ROOT", original)
    with pytest.raises(SystemExit):
        runner.main(["--date", "wrong"])
    with pytest.raises(SystemExit):
        runner.main(
            ["--date", "20261115", "--private-output", str(original / "results" / output.name)]
        )


def test_missing_field_and_original_failures(tmp_path: Path) -> None:
    """REQ-REPORT-9093-V775: retain exact original gates and distinguish missing fields."""
    original = d.gate("/historical", "required", 1, 0, "original_failure", "old")
    source(tmp_path, 9083, gate_check_summary=[original])
    atomic_json(
        tmp_path / "results/experiment_9085_collect_a.json",
        dict(
            experiment=9085,
            failed_field="absent",
            failed_expected=1,
            failed_evidence_path=str(tmp_path / d.SOURCES[9083]),
            failed_evidence_sha256=d.ref(tmp_path / d.SOURCES[9083])["sha256"],
        ),
    )
    value = d.inventory(d.freeze(tmp_path, tmp_path / "private"))
    assert original in value["gate_check_summary"]
    assert any(r["observed"] == "missing_field" for r in value["gate_check_summary"])


def test_receipt_formats_deduplication_and_clock_drift(tmp_path: Path) -> None:
    """REQ-REPORT-9093-V775: shared parent clocks are counted once across sidecar forms."""
    receipt = dict(
        current_owner_pid=3,
        started_monotonic_ns=10,
        ended_monotonic_ns=20,
        duration_s=10 / 1e9,
        execution_venue="host",
        MODEL_SPECS=[],
    )
    path = tmp_path / "current_work.json"
    atomic_json(path, receipt)
    source(tmp_path, 9082, current_work_receipts=[d.ref(path), receipt, receipt])
    source(tmp_path, 9083, current_work_receipts=dict(receipt, duration_s=99))
    manifest = d.freeze(tmp_path, tmp_path / "private")
    value = d.inventory(manifest)
    actual = [r for r in value["rows"] if r["operation"] == "qualification"]
    assert len(actual) == 2
    assert sum(r["latency_s"] is not None for r in actual) == 1
    assert any(
        r["check"] == "raw_clock_duration" and not r["passed"] for r in value["gate_check_summary"]
    )


def test_receipt_and_fixture_tamper(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9093-V775: independent bytes remain bound after headline rehashing."""
    from carnot.reporting import service_cost_runner_9093 as runner

    work = runner.measure(tmp_path / "missing", tmp_path / "raw", [])
    atomic_json(tmp_path / "raw/validation.json", {})
    value = runner.build(work, {}, True)
    output = tmp_path / (d.NAME + ".json")
    atomic_json(output, value)
    assert runner.replay(output)["passed"]
    path = Path(work["fixture_rows"][0]["source_reference"]["path"])
    previous = path.read_bytes()
    path.write_text("tamper")
    assert runner.replay(output)["reason"] == "fixture_hash_drift"
    path.write_bytes(previous)
    path = Path(work["current_work_reference"]["path"])
    path.chmod(0o600)
    path.write_text("tamper")
    assert runner.replay(output)["reason"] == "receipt_hash_drift"


def test_shared_producer_bytes(tmp_path: Path) -> None:
    """REQ-REPORT-9093-V775: shared producer bytes have one immutable custody object."""
    code = tmp_path / "producer.py"
    code.write_text("shared producer")
    binding = d.ref(code)
    source(tmp_path, 9082, code_config_hashes=[binding, binding])
    source(tmp_path, 9083, code_config_hashes=[binding])
    manifest = d.freeze(tmp_path, tmp_path / "private")
    copies = [e for e in manifest["entries"] if e["source_path"] == str(code)]
    assert len(copies) == 3 and len({e["path"] for e in copies}) == 1
    assert d.inventory(manifest)["invocation_dispositions"]
