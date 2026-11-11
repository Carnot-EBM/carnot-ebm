"""REQ-REPORT-8999 / REQ-VERIFY-8999: private authority, science and CLI custody."""

from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

from coverage import Coverage, CoverageData
import pytest
import yaml

from carnot.reporting import v769_branch_contract as c


def tree(root):
    """Private copies permit refusal controls without touching operator evidence."""
    for name in [c.DESIGN, c.ACTIVE, *c.e.PINS]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((c.ROOT / name).read_bytes())
    return root


def validation(passed=True):
    """Synthetic checks test refusal only; published checks are real subprocesses."""
    return dict(
        receipts=[dict(name="fixture", passed=passed)],
        coverage=dict(complete=passed, percent=100 if passed else 0),
        global_test_debt={},
    )


def test_authority(tmp_path):
    """REQ-REPORT-8999: active and staged copies must match every full object."""
    root = tree(tmp_path / "root")
    assert c.authority(root)["ready"]
    queue = yaml.safe_load((root / c.ACTIVE).read_bytes())
    (root / c.STAGED).write_text(yaml.safe_dump(queue))
    old = deepcopy(queue)
    old["milestone"] = "2026.11.768"
    (root / c.ACTIVE).write_text(yaml.safe_dump(old))
    assert c.authority(root)["comparison_mode"] == "staged"
    assert c.authority(root)["ready"]
    (root / c.STAGED).unlink()
    assert not c.authority(root)["ready"]
    assert not c.authority(tmp_path / "missing")["ready"]
    (root / c.ACTIVE).write_text(yaml.safe_dump(queue))
    for field in ["title", "prompt", "deliverable", "phase"]:
        changed = deepcopy(queue)
        changed["tasks"][0][field] = "forged"
        (root / c.ACTIVE).write_text(yaml.safe_dump(changed))
        assert not c.authority(root)["ready"]
    (root / c.ACTIVE).write_text(yaml.safe_dump(queue))
    assert all(r["passed"] for r in c.mutations(root, tmp_path / "controls"))


def test_frozen_evidence(tmp_path):
    """REQ-REPORT-8999: historical criteria, states, roles and bounds stay frozen."""
    result = c.e.science(c.ROOT)
    assert result["frozen_graph_authority_ready_score"] == 1
    assert result["online_protocol_ready_score"] == 1
    assert result["role_counts"] == dict(calibration=128, test=128, online=768)
    assert len(result["development_admission_rows"]) == 4
    assert all(r["passed"] for r in result["development_admission_rows"])
    absent = c.e.science(tmp_path)
    assert absent["frozen_graph_authority_ready_score"] == 0
    assert absent["online_protocol_ready_score"] == 0
    assert any(r["observed"] == "missing_file" for r in absent["checks"])


def test_build_and_replay(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8999-REPLAY: independently recompute rows and readiness."""
    work = c.freeze(tree(tmp_path / "root"), tmp_path / "raw", "20261111")
    value = c.build(work, validation())
    assert value["contract_ready_score"] == 1
    assert value["arc_new_outcomes_ready_score"] == 0
    assert value["verdict_class"] == "null"
    assert value["completed_count"] == value["intended_count"] == 14
    assert not any(value["model_invocation_counts"].values())
    candidate = tmp_path / (c.NAME + ".json")
    c.atomic_json(candidate, value)
    assert not c.replay(candidate)["passed"]
    monkeypatch.setattr(c, "verify_measured", lambda *args: True)
    assert c.replay(candidate)["passed"]
    for field in ["contract_ready_score", "completed_count", "frozen_graph_authority_ready_score"]:
        forged = dict(value, **{field: 99})
        forged["reproducibility_checksum"] = c.checksum(forged)
        c.atomic_json(candidate, forged)
        assert not c.replay(candidate)["passed"]
    assert c.build(work, validation(False))["verdict_class"] == "disqualified"
    changed = deepcopy(work)
    changed["science"]["frozen_graph_authority_ready_score"] = 0
    assert c.build(changed, validation())["contract_ready_score"] == 1
    with pytest.raises(FileExistsError):
        c.freeze(tmp_path, Path(work["raw"]), "20261111")
    blocked = c.build(c.freeze(tmp_path / "absent", tmp_path / "blocked", "20261111"), validation())
    assert blocked["verdict_class"] == "blocked"
    assert blocked["contract_ready_score"] == 0
    assert any(r["observed"] == "missing_file" for r in blocked["gate_check_summary"])
    assert not c.replay(tmp_path / "absent.json")["passed"]


def test_real_cli(tmp_path, monkeypatch):
    """E2E-018/021 / REQ-VERIFY-8999: execute outside checkout without PYTHONPATH."""
    from carnot.reporting import v755_branch_runner as runner

    config = tmp_path / "coverage.ini"
    config.write_text(
        "[run]\ninclude =\n"
        + "".join(f"    {c.ROOT / p}\n" for p in c.OWNED)
        + "[report]\nexclude_lines =\n"
    )
    for args, expected in [
        (["--date", "bad"], 2),
        (["--cold-replay", str(tmp_path / "absent")], 1),
    ]:
        print("before real CLI", args, flush=True)
        child = subprocess.run(
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--rcfile",
                str(config),
                "--data-file",
                str(tmp_path / ".coverage"),
                "--parallel-mode",
                str(c.ROOT / c.CLI),
                *args,
            ],
            cwd="/var/tmp",
            env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
            capture_output=True,
            text=True,
            timeout=60,
        )
        print("after real CLI", child.returncode, flush=True)
        assert child.returncode == expected, child.stdout + child.stderr
    tracer = Coverage.current()
    if tracer:
        for path in tmp_path.glob(".coverage.*"):
            data = CoverageData(basename=str(path))
            data.read()
            tracer.get_data().update(data)

    def checked(raw, tracer):
        tracer.stop()
        tracer.save()
        return validation()

    monkeypatch.setattr(runner, "validate", checked)
    monkeypatch.setattr(runner, "terminal", lambda candidate, raw: c.replay(candidate))
    monkeypatch.setattr(c, "verify_measured", lambda *args: True)
    output = tmp_path / (c.NAME + ".json")
    for args in [["--date", "20261111", "--output", str(output)], ["--cold-replay", str(output)]]:
        monkeypatch.setattr(sys, "argv", [c.CLI, *args])
        with pytest.raises(SystemExit) as error:
            runpy.run_path(str(c.ROOT / c.CLI), run_name="__main__")
        assert error.value.code == 0


def test_replay_operand_refusals(tmp_path, monkeypatch):
    """REQ-VERIFY-8999: rehashed custody forgeries must fail independently."""
    work = c.freeze(tree(tmp_path / "root"), tmp_path / "raw", "20261111")
    value = c.build(work, validation())
    candidate = tmp_path / (c.NAME + ".json")
    monkeypatch.setattr(c, "verify_measured", lambda *args: True)
    for mode in ["operand", "code", "contract", "control"]:
        forged = deepcopy(value)
        if mode == "operand":
            forged["raw_shard_hashes"][0]["sha256"] = "sha256:forged"
        elif mode == "code":
            forged["code_config_hashes"][0]["source"]["sha256"] = "sha256:forged"
        else:
            changed = deepcopy(work)
            changed.pop("raw_reference")
            if mode == "contract":
                changed["contract"]["ready"] = False
            else:
                changed["controls"][0]["passed"] = False
            path = tmp_path / (mode + ".json")
            c.atomic_json(path, changed)
            forged["raw_shard_hashes"][0] = c.reference(path)
        forged["reproducibility_checksum"] = c.checksum(forged)
        c.atomic_json(candidate, forged)
        assert not c.replay(candidate)["passed"]


def test_subprocess_wrapper_and_frontier_refusal(tmp_path, monkeypatch):
    """REQ-VERIFY-8999: pytest options stay with pytest, stale ARC stays unready."""
    monkeypatch.setattr(c, "_run_check", lambda name, argv, *args: dict(argv=argv, passed=True))
    assert "--noconftest" in c.run_check("test", [sys.executable, "-m", "pytest"], tmp_path)["argv"]
    assert c.run_check("lint", ["ruff", "check"], tmp_path)["argv"] == ["ruff", "check"]
    original = c.e.pinned

    def forged_frontier(root, name, checks):
        value = original(root, name, checks)
        value["arc_frontier"]["frontier_after"]["ledger_receipt_identities"] = ["forged"]
        return value

    monkeypatch.setattr(c.e, "pinned", forged_frontier)
    assert not c.e.frontier(c.ROOT)["ready"]
