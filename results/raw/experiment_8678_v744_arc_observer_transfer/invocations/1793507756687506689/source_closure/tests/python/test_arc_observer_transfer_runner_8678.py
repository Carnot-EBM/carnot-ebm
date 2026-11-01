"""REQ-REPORT-8678, REQ-VERIFY-8678: fresh transfer never recasts censored work."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import arc_observer_transfer_8678 as e
from carnot.reporting import arc_observer_transfer_runner_8678 as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.v709_execution import child


def test_actual_private_cli_and_cold_rejection(tmp_path):
    """REQ-VERIFY-8678: real script paths publish blocked evidence and reject substitutions."""
    output = tmp_path / (e.NAME + ".json")
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    args = ["--root", str(tmp_path / "absent"), "--output", str(output), "--private-control"]
    receipt = child("private", argv + args, tmp_path / "logs", deadline=120)
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["arc_rows_ready_score"] == 0
    assert child("cold", argv + ["--cold-replay", str(output)], tmp_path / "logs")["passed"]
    changed = deepcopy(value)
    changed["arc_rows_ready_score"] = 1
    changed["reproducibility_checksum"] = canonical_hash(changed)
    target = tmp_path / "changed.json"
    atomic_json(target, changed)
    assert child("rehashed", argv + ["--cold-replay", str(target)], tmp_path / "logs", expected=1)[
        "passed"
    ]
    changed["work_reference"]["path"] = str(tmp_path / "missing")
    atomic_json(target, changed)
    assert child("missing", argv + ["--cold-replay", str(target)], tmp_path / "logs", expected=1)[
        "passed"
    ]
    assert child("owned_failure", argv + args + ["--owned-failure"], tmp_path / "logs")["passed"]
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"
    assert child("bad_date", argv + ["--date", "20261301"], tmp_path / "logs", expected=2)["passed"]
    assert runner.main(["--private-control"]) == 2


def test_real_qualified_wrapper_and_children(tmp_path):
    """SCENARIO-REPORT-8678-TRANSFER: actual wrapper callbacks preserve recorder parity."""
    from carnot.reporting import arc_observation_live_8677 as qualified
    from carnot.reporting import arc_observer_transfer_live_8678 as live

    ref = e.authenticate()["actual_transfer_references"][0]
    spec = dict(
        e.units()[0],
        max_actions=8,
        directory=str(tmp_path / "episode"),
        kind="episode",
        fixture_reference=ref,
    )
    off = live.collect(spec, qualified.trace_sdk(ref), tmp_path / "off")
    on = live.collect(dict(spec, reset_provenance="on"), qualified.trace_sdk(ref), tmp_path / "on")
    assert off["actions"] == on["actions"] and off["observer_rows"] == on["observer_rows"]
    assert on["observer_issues"] == [] and on["recorder_error_metadata"] == []
    assert off["reset_provenance"]["reset_request_rows"] == []
    assert on["reset_provenance"]["reset_request_rows"]
    assert any(r["status"] == "duplicate_callback" for r in on["delivery_rows"])
    assert [r["action_index"] for r in on["delivery_rows"] if r["status"] == "canonical"] == list(
        range(1, 9)
    )
    work = e.empty_work("20261101", [])
    runner.run_child(spec, tmp_path, work)
    assert work["measurement_receipts"][0]["passed"]
    del spec["fixture_reference"]
    spec["directory"] = str(tmp_path / "installed")
    runner.run_child(spec, tmp_path, work)
    assert work["measurement_receipts"][-1]["passed"]
    assert json.loads((tmp_path / "installed/transfer_stream.json").read_bytes())["sdk_identity"][
        "official"
    ]


def test_measurement_branches(tmp_path, monkeypatch):
    """REQ-VERIFY-8678: missing children and exhausted budgets cannot acquire readiness."""
    protocol = e.authenticate()
    monkeypatch.setattr(e, "authenticate", lambda: protocol)

    def recorded(spec, raw, work):
        work["measurement_receipts"].append(dict(name="recorded", passed=True, duration_s=0.01))
        atomic_json(Path(spec["directory"]) / "transfer_stream.json", dict(unit=spec))

    monkeypatch.setattr(runner, "run_child", recorded)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    result = runner.measure(e.ROOT, tmp_path / "complete", scratch, "20261101")
    assert (
        len(result["references"]) == 16 and result["registry_precheck"]["checked_before_execution"]
    )
    assert result["preconditions_checked"]["writable"]

    def killed(spec, raw, work):
        work["measurement_receipts"].append(
            dict(name="killed", passed=False, duration_s=0.01, exit_code=-9)
        )

    monkeypatch.setattr(runner, "run_child", killed)
    assert runner.measure(e.ROOT, tmp_path / "killed", scratch, "20261101")["references"] == []
    monkeypatch.setattr(
        e, "authenticate", lambda: (_ for _ in ()).throw(ValueError("changed_authority"))
    )
    assert (
        runner.measure(e.ROOT, tmp_path / "auth", scratch, "20261101")["failures"][-1]["check"]
        == "upstream_authentication"
    )
    monkeypatch.setattr(e, "authenticate", lambda: protocol)
    digest = runner.sha256_file
    monkeypatch.setattr(
        runner, "sha256_file", lambda p: "changed" if p.suffix == ".pt" else digest(p)
    )
    assert (
        runner.measure(e.ROOT, tmp_path / "head", scratch, "20261101")["failures"][-1]["check"]
        == "numerical_checkpoint"
    )
    monkeypatch.setattr(runner, "sha256_file", digest)
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda p: type("Usage", (), {"free": 0})())
    assert (
        runner.measure(e.ROOT, tmp_path / "resources", scratch, "20261101")["failures"][-1]["check"]
        == "cpu_resources"
    )


def test_measurement_cap_and_rejected_qualification(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8678-REPLAY: never expand the fixed quota to fill a horizon."""
    original = e.read
    for field in ("required_checks_passed", "flagged_adversarial", "observer_delivery_ready_score"):

        def changed(ref):
            value = original(ref)
            if "experiment_8677" in ref["path"] and "work.json" not in ref["path"]:
                value[field] = not bool(value[field])
            return value

        monkeypatch.setattr(e, "read", changed)
        with pytest.raises(ValueError, match="observer_qualification"):
            e.authenticate()
    monkeypatch.setattr(e, "read", original)
    monkeypatch.setattr(e, "read_bound_sidecar", lambda *args: {"report": {"passed": False}})
    with pytest.raises(ValueError, match="upstream_publication"):
        e.authenticate()


def test_source_closure_rejection_and_budget_limit(tmp_path, monkeypatch):
    """REQ-VERIFY-8678: authenticate runtime dependencies, then enforce the cumulative cap."""
    original_hash = e.sha256_file
    monkeypatch.setattr(
        e,
        "sha256_file",
        lambda p: "changed" if "/source_closure/python/" in str(p) else original_hash(p),
    )
    with pytest.raises(ValueError, match="qualified_source_closure"):
        e.authenticate()
    monkeypatch.setattr(e, "sha256_file", original_hash)
    protocol = e.authenticate()
    monkeypatch.setattr(e, "authenticate", lambda: protocol)
    ticks = iter([0, 2_500_000_000_000, 2_500_000_000_001])
    monkeypatch.setattr(runner.time, "monotonic_ns", lambda: next(ticks))
    work = runner.measure(e.ROOT, tmp_path / "bounded", tmp_path, "20261101")
    assert work["references"] == [] and work["duration_s"] > 2400


def test_real_child_rejection(tmp_path):
    """SCENARIO-VERIFY-8678-REPLAY: failed SDK requests retain an actual child exit one."""
    spec = dict(
        e.units()[0],
        game="private-unavailable-game",
        max_actions=1,
        directory=str(tmp_path / "rejected"),
        kind="episode",
    )
    target = tmp_path / "unit.json"
    atomic_json(target, spec)
    receipt = child(
        "rejected_sdk",
        [sys.executable, "-u", str(e.ROOT / e.CLI), "--episode", str(target)],
        tmp_path / "logs",
        expected=1,
    )
    assert receipt["passed"]
    result = json.loads((tmp_path / "rejected/transfer_stream.json").read_bytes())
    assert result["termination"]["error"]
    assert result["actions"] == [dict(kind="RESET", data=None)]
    assert result["observations"] == []
