"""REQ-REPORT-7851-V682: V682 authority and CLI regressions."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from carnot.reporting import roadmap_contract as contract


ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / "openspec/change-proposals/research-roadmap-vNEXT.md"
ACTIVE = ROOT / "research-roadmap.yaml"
V681 = ROOT / "docs/research-notes/v681-authority-snapshots"
CLI = ROOT / "scripts/experiments/experiment_7851_v682_contract_methods.py"


@pytest.fixture
def authorities():
    """SCENARIO-REPORT-7851-V682-CONTRACT: read exact current sources."""
    return DESIGN.read_text(), yaml.safe_load(ACTIVE.read_text())


def test_active_contract_and_v681_regression(authorities):
    """SCENARIO-REPORT-7851-V682-CONTRACT: activation and old bytes both bind."""
    design, active = authorities
    result = contract.compare_contract(
        design, active, active, milestone="2026.09.682", first_id=7851, count=14
    )
    assert result["passed"]
    assert [row["unit_id"].split("-")[0] for row in result["rows"]] == [
        f"exp{i}" for i in range(7851, 7865)
    ]
    old_design = (V681 / "design.md").read_text()
    old_staged = yaml.safe_load((V681 / "staged.yaml").read_text())
    old_active = yaml.safe_load((V681 / "active.yaml").read_text())
    old = contract.compare_contract(old_design, old_staged, old_active)
    assert old["passed"] and len(old["rows"]) == 14


@pytest.mark.parametrize(
    "mutation", ["drop", "reorder", "stale", "model", "retirement", "gate", "document"]
)
def test_mutations_fail(authorities, mutation):
    """SCENARIO-REPORT-7851-V682-CONTRACT: each private defect is rejected."""
    design, active = deepcopy(authorities)
    staged = deepcopy(active)
    if mutation == "drop":
        staged["tasks"].pop()
    elif mutation == "reorder":
        staged["tasks"][0], staged["tasks"][1] = staged["tasks"][1], staged["tasks"][0]
    elif mutation == "stale":
        staged["milestone"] = "2026.09.681"
    elif mutation == "model":
        staged["tasks"][0]["MODEL_SPECS"] = ["wrong/model"]
    elif mutation == "retirement":
        staged["tasks"][0]["prior_failures"][0]["retire_if_same_verdict"] = False
    elif mutation == "gate":
        staged["tasks"][4]["gated_on"][0]["artifact_field"] = "misspelled"
    else:
        design = design.replace(
            "Bind fourteen tasks and register", "Bind thirteen tasks and register", 1
        )
    result = contract.compare_contract(
        design, staged, active, milestone="2026.09.682", first_id=7851, count=14
    )
    assert not result["passed"]


def test_snapshot_and_cold_replay(authorities, tmp_path):
    """SCENARIO-REPORT-7851-V682-CONTRACT: bytes and row identities are stable."""
    design, active = authorities
    result = contract.compare_contract(
        design, active, active, milestone="2026.09.682", first_id=7851, count=14
    )
    snapshots = contract.snapshot_authorities((DESIGN, ACTIVE, ACTIVE), tmp_path / "snap")
    assert contract.verify_snapshots(snapshots)
    Path(snapshots[0]["path"]).write_bytes(b"changed")
    assert not contract.verify_snapshots(snapshots)
    raw = tmp_path / "rows.json"
    candidate = tmp_path / "candidate.json"
    raw.write_text(json.dumps(result["rows"]))
    candidate.write_text(
        json.dumps(
            {"experiment_id": 7851, "task_id": "exp7851-contract-methods", "rows": result["rows"]}
        )
    )
    assert contract.cold_replay(candidate, raw, experiment_id=7851)
    changed = json.loads(candidate.read_text())
    changed["rows"][0]["absolute_metric"] = 0
    candidate.write_text(json.dumps(changed))
    assert not contract.cold_replay(candidate, raw, experiment_id=7851)


def test_private_real_cli_and_verdict(authorities, tmp_path):
    """SCENARIO-REPORT-7851-V682-VALIDATION: CLI emits a consistent private state."""
    output = tmp_path / "private.json"
    raw = tmp_path / "rows.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--date",
            "20260929",
            "--output",
            str(output),
            "--raw",
            str(raw),
            "--no-validation",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    value = json.loads(output.read_text())
    assert value["experiment_id"] == 7851
    assert value["verdict_class"] == "partial"
    assert value["honest_verdict"].startswith("partial_")
    assert value["contract_ready_score"] == 0
    assert len(value["rows"]) == 14
    assert contract.cold_replay(output, raw, experiment_id=7851)


def load_cli():
    """Load the owned CLI for controlled error-route tests."""
    spec = importlib.util.spec_from_file_location("v682_contract_cli", CLI)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reader_errors_and_snapshot_conflict(authorities, tmp_path):
    """SCENARIO-REPORT-7851-V682-CONTRACT: malformed and changed bytes fail."""
    design, _ = authorities
    with pytest.raises(ValueError, match="JSON contract missing"):
        contract.parse_design(
            design.replace("V682_TASK_CONTRACT_START", "BROKEN"), milestone="2026.09.682"
        )
    with pytest.raises(ValueError, match="milestone mismatch"):
        contract.parse_design(
            design.replace('"milestone": "2026.09.682"', '"milestone": "wrong"', 1),
            milestone="2026.09.682",
        )
    with pytest.raises(ValueError, match="three authorities"):
        contract.snapshot_authorities((ACTIVE,), tmp_path / "snap")
    contract.snapshot_authorities((DESIGN, ACTIVE, ACTIVE), tmp_path / "snap")
    (tmp_path / "snap/design.md").write_bytes(b"changed")
    with pytest.raises(ValueError, match="immutable"):
        contract.snapshot_authorities((DESIGN, ACTIVE, ACTIVE), tmp_path / "snap")


def test_child_exits_logs_and_deadline(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7851-V682-VALIDATION: real exits and sealed logs survive."""
    cli = load_cli()
    monkeypatch.setattr(cli, "SEALED", tmp_path / "sealed")
    valid = {
        "name": "worktree_imports",
        "argv": [
            sys.executable,
            "-c",
            "import json; print(json.dumps({'resolved_imports': {'reader': "
            + repr(str(ROOT / "python/carnot/reporting/roadmap_contract.py"))
            + "}}))",
        ],
        "timeout_s": 5,
        "classification": "required",
    }
    good = cli.run_child(valid, 0, tmp_path / "work")
    assert good["passed"] and good["resolved_imports"]
    assert cli.run_child(valid, 1, tmp_path / "work")["passed"]
    invalid = dict(valid, argv=[sys.executable, "-c", "print('no imports')"])
    assert not cli.run_child(invalid, 2, tmp_path / "work")["passed"]
    fail = dict(valid, name="failed", argv=[sys.executable, "-c", "raise SystemExit(3)"])
    assert cli.run_child(fail, 3, tmp_path / "work")["exit_code"] == 3
    slow = dict(
        valid,
        name="deadline",
        argv=[sys.executable, "-c", "import time; time.sleep(2)"],
        timeout_s=0.1,
    )
    assert cli.run_child(slow, 4, tmp_path / "work")["timed_out"]
    target = Path(good["log_path"])
    target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="collision"):
        cli.run_child(valid, 5, tmp_path / "work")


def test_cli_private_block_mismatch_and_errors(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7851-V682-VALIDATION: failed preconditions grant no readiness."""
    cli = load_cli()
    output, raw = tmp_path / "out.json", tmp_path / "rows.json"
    args = ["--output", str(output), "--raw", str(raw)]
    absent = ROOT / "docs/research-notes/absent-v682-input.md"
    monkeypatch.setattr(cli, "SOURCES", (*cli.SOURCES, absent))
    assert cli.main(args) == 0
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "blocked"
    assert value["gate_check_summary"][0]["path"] == str(absent)
    monkeypatch.setattr(cli, "SOURCES", cli.SOURCES[:-1])
    raw.unlink()
    original = cli.compare_contract

    def mismatch(*items, **kwargs):
        result = original(*items, **kwargs)
        result["passed"] = False
        return result

    monkeypatch.setattr(cli, "compare_contract", mismatch)
    assert cli.main(args) == 1
    assert json.loads(output.read_text())["contract_ready_score"] == 0
    with pytest.raises(SystemExit):
        cli.main(["--no-validation"])
    with pytest.raises(SystemExit):
        cli.main(["--cold-replay", str(output)])
    raw.write_text("[]")
    with pytest.raises(ValueError, match="checkpoint content differs"):
        cli.main(args)


@pytest.mark.parametrize("mode", ["clean", "required_fail", "flagged", "bad_report"])
def test_cli_terminal_reduction(tmp_path, monkeypatch, mode):
    """SCENARIO-REPORT-7851-V682-VALIDATION: required exits and flags close readiness."""
    cli = load_cli()
    manifest_path = tmp_path / "manifest.json"
    commands = [
        {
            "name": "worktree_imports",
            "argv": [sys.executable, "-c", "pass"],
            "classification": "required",
            "timeout_s": 5,
        },
        {
            "name": "adversarial_verify",
            "argv": [sys.executable, "-c", "pass"],
            "classification": "required",
            "timeout_s": 5,
        },
    ]
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "test_v682",
                "frozen_hashes": {},
                "commands": commands,
                "repository_health": {
                    "name": "repository_health_180s",
                    "argv": [sys.executable, "-c", "pass"],
                    "classification": "diagnostic",
                    "timeout_s": 180,
                },
                "historical_required_failures": ["Exp7837 remains disqualified"],
            }
        )
    )
    monkeypatch.setattr(cli, "MANIFEST", manifest_path)
    monkeypatch.setattr(cli, "SNAPSHOTS", tmp_path / "snapshots")
    monkeypatch.setenv("CARNOT_7851_TMP", str(tmp_path / "owned"))
    report = tmp_path / "verifier.json"
    report.write_text(
        "not json"
        if mode == "bad_report"
        else json.dumps({"flagged_count": int(mode == "flagged")})
    )

    def child(command, index, scratch_dir):
        return {
            "name": command["name"],
            "command_argv": command["argv"],
            "classification": command["classification"],
            "passed": not (mode == "required_fail" and command["name"] == "worktree_imports"),
            "log_path": str(report),
            "exit_code": 0,
            "timed_out": False,
        }

    monkeypatch.setattr(cli, "run_child", child)
    output, raw = tmp_path / "result.json", tmp_path / "rows.json"
    rc = cli.main(["--output", str(output), "--raw", str(raw)])
    value = json.loads(output.read_text())
    assert rc == (0 if mode == "clean" else 1)
    assert value["verdict_class"] == ("circular_positive" if mode == "clean" else "disqualified")
    assert value["contract_ready_score"] == int(mode == "clean")
    assert value["flagged_adversarial"] == (mode in {"flagged", "bad_report"})
    assert value["historical_required_failures"] == ["Exp7837 remains disqualified"]
    assert cli.main(["--cold-replay", str(output), "--raw", str(raw)]) == 0
    changed = json.loads(raw.read_text())
    changed["rows"][0]["absolute_metric"] = 0
    raw.write_text(json.dumps(changed))
    assert cli.main(["--cold-replay", str(output), "--raw", str(raw)]) == 1


def test_cli_frozen_and_snapshot_errors(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7851-V682-VALIDATION: frozen inputs cannot drift."""
    cli = load_cli()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "test_v682",
                "frozen_hashes": {"python/carnot/reporting/roadmap_contract.py": "sha256:wrong"},
                "commands": [],
            }
        )
    )
    monkeypatch.setattr(cli, "MANIFEST", manifest_path)
    monkeypatch.setattr(cli, "SNAPSHOTS", tmp_path / "snapshots")
    monkeypatch.setenv("CARNOT_7851_TMP", str(tmp_path / "owned"))
    args = ["--output", str(tmp_path / "result.json"), "--raw", str(tmp_path / "rows.json")]
    with pytest.raises(ValueError, match="frozen source"):
        cli.main(args)
    (tmp_path / "snapshots/design.md").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="immutable snapshot"):
        cli.main(args)


def test_source_rows_and_current_work_receipt(tmp_path):
    """SCENARIO-REPORT-7851-V682-CONTRACT: missing and historical roles stay distinct."""
    cli = load_cli()
    missing = ROOT / "docs/research-notes/absent-v682-input.md"
    rows = cli.source_rows(
        (ROOT / "results/experiment_7837_v681_contract_methods.json", missing), "20260929"
    )
    assert rows[0]["source_role"] == "historical_disqualified"
    assert rows[1]["sha256"] is None and not rows[1]["eligible"]
    assert cli.path_label(tmp_path) == str(tmp_path)


def test_child_force_kill_after_grace(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7851-V682-VALIDATION: a child that ignores terminate is killed."""
    cli = load_cli()
    monkeypatch.setattr(cli, "SEALED", tmp_path / "sealed")

    class StubbornChild:
        killed = False

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired("stubborn", timeout)
            return -9

        def kill(self):
            self.killed = True

    child = StubbornChild()
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *args, **kwargs: child)
    command = {
        "name": "stubborn",
        "argv": [sys.executable],
        "timeout_s": 0,
        "classification": "required",
    }
    receipt = cli.run_child(command, 0, tmp_path / "scratch")
    assert child.killed and receipt["timed_out"] and not receipt["passed"]


def test_snapshot_cold_verification_refuses_false(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7851-V682-CONTRACT: a failed snapshot replay stops compute."""
    cli = load_cli()
    monkeypatch.setattr(cli, "SNAPSHOTS", tmp_path / "snapshots")
    monkeypatch.setattr(cli, "verify_snapshots", lambda snapshots: False)
    with pytest.raises(ValueError, match="cold verification failed"):
        cli.main(["--output", str(tmp_path / "private.json"), "--no-validation"])
