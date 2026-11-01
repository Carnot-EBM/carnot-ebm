"""REQ-REPORT-8705 / REQ-VERIFY-8705: repaired code needs fresh owned evidence."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import arc_decision_qualification_8705 as e
from carnot.reporting import arc_decision_qualification_runner_8705 as runner


def cli(*args):
    """SCENARIO-VERIFY-8705-SEAL: use the actual standalone CLI outside the checkout."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)],
        cwd="/var/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )


@pytest.fixture(scope="module")
def private(tmp_path_factory):
    """SCENARIO-REPORT-8705-CLOSED: only the historical ledger is read."""
    path = tmp_path_factory.mktemp("arc8705") / (e.NAME + ".json")
    result = cli("--private-control", "--output", path, "--date", "20261101")
    assert result.returncode == 0, result.stdout + result.stderr
    return path, json.loads(path.read_bytes())


def test_private_and_replay(private):
    """SCENARIO-REPORT-8705-CLOSED: qualification is separate from scientific benefit."""
    path, value = private
    assert cli("--cold-replay", path).returncode == 0
    assert value["experiment_id"] == 8705 and value["milestone"] == "2026.11.746"
    assert value["completed_count"] == value["censored_count"] == 16
    assert value["independent_count"] == 4 and value["complete_horizon_count"] == 0
    assert value["policy_decision"] == "no_supported_policy_change"
    assert (
        value["arc_decision_ready_score"] == 0
    )  # private receipts cannot grant production readiness
    assert value["diagnostic_closed"] is True
    assert value["model_invocation_counts"]["language_model"] == 0
    assert value["headline_solve_credit"] == 0
    assert value["historical_quarantines"]["exp8691_flagged_adversarial"] is True
    assert set(value) <= set(value["field_principles"])


@pytest.mark.parametrize(
    "mode", ["promotion", "primitive", "missing", "source", "clock", "receipts", "log"]
)
def test_replay_rejections(private, tmp_path, monkeypatch, mode):
    """SCENARIO-VERIFY-8705-SEAL: rehashed local envelopes cannot replace sealed evidence."""
    _, original = private
    value = deepcopy(original)
    work = deepcopy(e.read(value["primitive_reference"]))
    receipts = deepcopy(value["validation_receipts"])
    if mode == "promotion":
        value["policy_decision"] = "raise_priority"
    elif mode in {"primitive", "missing"}:
        work["run_date"] = "20261102"
        operand = tmp_path / "measurement.json"
        e.atomic_json(operand, work)
        value["primitive_reference"] = e.reference(operand)
        e.atomic_json(
            tmp_path / "measurement_seal.json", {"measurement": value["primitive_reference"]}
        )
        if mode == "missing":
            operand.unlink()
    else:
        if mode == "source":
            work["code_refs"] = []
        elif mode == "clock":
            work["end_ns"] += 1
        elif mode == "receipts":
            receipts = []
        else:
            receipts[0].update(stdout_path=str(tmp_path / "absent"), stdout_sha256="bad")
            value["validation_receipts"] = receipts
        read = e.read

        def altered(ref):
            if ref == value["primitive_reference"]:
                return work
            if ref == value["validation_reference"]:
                return {"receipts": receipts}
            return read(ref)

        monkeypatch.setattr(e, "read", altered)
    value["reproducibility_checksum"] = e.checksum(value)
    path = tmp_path / "candidate.json"
    e.atomic_json(path, value)
    if mode in {"promotion", "primitive", "missing"}:
        assert cli("--cold-replay", path).returncode == 1
    with pytest.raises((ValueError, OSError)):
        runner.replay(path)


@pytest.mark.parametrize("control", ["--private-control", "--failed-child"])
def test_private_output_and_failed_child(tmp_path, control):
    """REQ-VERIFY-8705: aliases reject before writes; a real failed child disqualifies."""
    alias = tmp_path / "checkout"
    alias.symlink_to(e.ROOT, target_is_directory=True)
    target = alias / "results" / (e.NAME + ".json")
    assert cli(control, "--output", target).returncode == 1
    output = tmp_path / (e.NAME + ".json")
    if control == "--failed-child":
        result = cli("--private-control", control, "--output", output)
        assert result.returncode == 1
        value = json.loads((tmp_path / "raw" / e.NAME / "terminal_candidate.json").read_bytes())
        assert value["verdict_class"] == "disqualified"
        assert value["arc_decision_ready_score"] == 0
        receipt = next(r for r in value["validation_receipts"] if r["name"] == "failed_child")
        assert receipt["actual_exit"] == 1 and not receipt["passed"]


def test_authority_and_adapter(tmp_path, monkeypatch):
    """REQ-VERIFY-8705: exercise the real adapter and independent full-object authority."""
    assert e.authority(e.ROOT)["ready"]
    with pytest.raises((OSError, ValueError)):
        e.authority(tmp_path)
    monkeypatch.setattr(runner.validation, "validate", lambda *a: [dict(passed=True)])
    assert runner.validate(tmp_path) == [dict(passed=True)]
    monkeypatch.setattr(runner.validation, "terminal", lambda *a: dict(passed=True))
    assert runner.terminal(tmp_path, tmp_path) == dict(passed=True)


def test_owned_runner_controls(private, tmp_path, monkeypatch):
    """REQ-VERIFY-8705: original success, failed delegated checks and date guards stay real."""
    path, value = private
    work = e.read(value["primitive_reference"])
    for args in [
        ("--date", "123"),
        ("--date", "20260230"),
        ("--private-control",),
        ("--cold-replay", tmp_path / "absent"),
    ]:
        assert cli(*args).returncode == 1
    monkeypatch.setattr(runner, "measure", lambda *a: work)
    monkeypatch.setattr(runner.validation, "validate", lambda *a: [dict(name="bad", passed=False)])
    monkeypatch.setattr(runner, "publish_primary", lambda *a: {})
    monkeypatch.setattr(runner, "reader_receipt", lambda *a, **kw: dict(passed=True))
    assert runner.run(tmp_path / (e.NAME + ".json"), "20261101", False, False) == 1
    result = e.build(
        work,
        [
            dict(name=name, passed=True)
            for name in ("original_disjoint_union", "preserved_raw_child_sqlite", "coverage_report")
        ],
        path,
    )
    assert result["arc_decision_ready_score"] == 1


def test_blocked_and_deadline(tmp_path, monkeypatch):
    """REQ-VERIFY-8705: absent external evidence blocks; an expired measurement cannot qualify."""
    resources = runner.resources

    def unavailable(path):
        result = resources(path)
        result.update(disk_backed=False, memory_available_bytes=0)
        return result

    monkeypatch.setattr(runner, "resources", unavailable)
    monkeypatch.setattr(
        e.original, "authenticate", lambda: (_ for _ in ()).throw(FileNotFoundError("operand"))
    )
    raw = tmp_path / "raw" / e.NAME / "invocations" / "blocked"
    work = runner.measure(raw, "20261101")
    e.atomic_json(raw / "validation.json", dict(receipts=[]))
    result = e.build(work, [dict(name="private", passed=True)], tmp_path / (e.NAME + ".json"))
    assert result["verdict_class"] == "blocked" and result["arc_decision_ready_score"] == 0
    assert result["gate_check_summary"] and result["completed_count"] == 0
    monkeypatch.setattr(runner.time, "monotonic_ns", iter([0, 3601_000_000_000]).__next__)
    with pytest.raises(TimeoutError, match="measurement_deadline"):
        runner.measure(tmp_path / "timeout", "20261101")


@pytest.mark.parametrize("mode", ["contract", "historical_log", "operand", "log"])
def test_remaining_immutable_boundaries(private, tmp_path, monkeypatch, mode):
    """SCENARIO-VERIFY-8705-SEAL: changed original evidence cannot be rescued by aggregate hashes."""
    _, value = private
    work = deepcopy(e.read(value["primitive_reference"]))
    if mode == "contract":
        design = tmp_path / e.DESIGN
        design.parent.mkdir(parents=True)
        design.write_text((e.ROOT / e.DESIGN).read_text().replace(e.TASKS_PIN, "bad"))
        (tmp_path / "research-roadmap.yaml").write_bytes(
            (e.ROOT / "research-roadmap.yaml").read_bytes()
        )
        with pytest.raises(ValueError, match="v746_full_object_authority"):
            e.authority(tmp_path)
        return
    digest = e.sha256_file
    if mode == "historical_log":
        history = json.loads((e.ROOT / "results" / (e.original.NAME + ".json")).read_bytes())
        selected = Path(
            next(r["stdout_path"] for r in history["validation_receipts"] if "stdout_path" in r)
        )
        monkeypatch.setattr(e, "sha256_file", lambda p: "changed" if p == selected else digest(p))
        with pytest.raises(ValueError, match="historical_log_changed"):
            runner.measure(tmp_path / "raw" / e.NAME / "invocations" / "bad", "20261101")
        return
    if mode == "operand":
        selected = Path(work["source_refs"][0]["path"])
    else:
        work_receipt = dict(
            name="stream", passed=True, stdout_path=str(tmp_path / "stream"), stdout_sha256="bad"
        )
        Path(work_receipt["stdout_path"]).write_text("actual stream")
        value = deepcopy(value)
        value["validation_receipts"] = [work_receipt]
        read = e.read
        monkeypatch.setattr(
            e,
            "read",
            lambda r: (
                dict(receipts=[work_receipt]) if r == value["validation_reference"] else read(r)
            ),
        )
        selected = Path(work_receipt["stdout_path"])
    monkeypatch.setattr(e, "sha256_file", lambda p: "changed" if p == selected else digest(p))
    candidate = tmp_path / "candidate.json"
    e.atomic_json(candidate, value)
    with pytest.raises(ValueError):
        runner.replay(candidate)
