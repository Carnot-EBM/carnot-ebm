"""REQ-REPORT-7837: V681 authority and executable receipt tests."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from carnot.reporting import roadmap_contract as subject


ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / "openspec/change-proposals/research-roadmap-vNEXT.md"
STAGED = ROOT / "research-roadmap-next.yaml"
ACTIVE = ROOT / "research-roadmap.yaml"
CLI = ROOT / "scripts/experiments/experiment_7837_v681_contract_methods.py"


@pytest.fixture
def authorities():
    """SCENARIO-REPORT-7837-AUTHORITY: use the actual planned bytes."""
    return (
        DESIGN.read_text(),
        yaml.safe_load(STAGED.read_text()),
        yaml.safe_load(ACTIVE.read_text()),
    )


def test_exact_four_way_contract(authorities):
    """SCENARIO-REPORT-7837-AUTHORITY: every task and gate agrees."""
    design, staged, active = authorities
    result = subject.compare_contract(design, staged, active)
    assert result["passed"]
    assert len(result["rows"]) == 14
    assert [r["unit_id"].split("-")[0] for r in result["rows"]] == [
        f"exp{i}" for i in range(7837, 7851)
    ]
    assert all(r["status"] == "completed" and r["matched"] for r in result["rows"])


@pytest.mark.parametrize(
    "mutation", ("drop", "reorder", "retirement", "model", "gate", "stale_doc")
)
def test_private_mutations_rejected(authorities, mutation):
    """SCENARIO-REPORT-7837-AUTHORITY: reject each owned contract defect."""
    design, staged, active = deepcopy(authorities)
    if mutation == "drop":
        staged["tasks"].pop()
    elif mutation == "reorder":
        staged["tasks"][0], staged["tasks"][1] = staged["tasks"][1], staged["tasks"][0]
    elif mutation == "retirement":
        del staged["tasks"][0]["prior_failures"][0]["retire_if_same_verdict"]
    elif mutation == "model":
        staged["tasks"][5]["MODEL_SPECS"] = ["wrong/model"]
    elif mutation == "gate":
        staged["tasks"][3]["gated_on"][0]["artifact_field"] = "typo_ready_score"
    else:
        design = design.replace(
            "Bind fourteen tasks and ingest", "Bind thirteen tasks and ingest", 1
        )
    result = subject.compare_contract(design, staged, active)
    assert not result["passed"]
    assert any(not row["matched"] for row in result["rows"]) or result["errors"]


def test_immutable_snapshot_and_cold_replay(authorities, tmp_path):
    """SCENARIO-REPORT-7837-REPLAY: changed bytes invalidate custody."""
    design, staged, active = authorities
    result = subject.compare_contract(design, staged, active)
    sources = (DESIGN, STAGED, ACTIVE)
    snapshots = subject.snapshot_authorities(sources, tmp_path / "snapshots")
    assert len(snapshots) == 3
    assert subject.verify_snapshots(snapshots)
    subject.snapshot_authorities(sources, tmp_path / "snapshots")
    path = Path(snapshots[0]["path"])
    path.write_bytes(path.read_bytes() + b"tampered")
    assert not subject.verify_snapshots(snapshots)
    with pytest.raises(ValueError, match="immutable"):
        subject.snapshot_authorities(sources, tmp_path / "snapshots")
    rows_path = tmp_path / "rows.json"
    rows_path.write_text(json.dumps(result["rows"]))
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps(
            {"experiment_id": 7837, "task_id": "exp7837-contract-methods", "rows": result["rows"]}
        )
    )
    assert subject.cold_replay(candidate, rows_path)
    changed = json.loads(candidate.read_text())
    changed["experiment_id"] = "exp7837-contract-methods"
    candidate.write_text(json.dumps(changed))
    assert not subject.cold_replay(candidate, rows_path)


def test_real_private_cli_e2e(tmp_path):
    """SCENARIO-REPORT-7837-REPLAY: execute the CLI without nested validation."""
    output = tmp_path / "candidate.json"
    run = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--date",
            "20260928",
            "--output",
            str(output),
            "--no-validation",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(output.read_text())
    assert value["experiment_id"] == 7837
    assert value["task_id"] == "exp7837-contract-methods"
    assert value["contract_ready_score"] == 0
    assert value["verdict_class"] == "partial"
    assert value["MODEL_SPECS"] == []
    assert value["acceptance_gate_results"]["decision_benefit"] is None
    assert len(value["rows"]) == 14


def load_cli():
    """Load the actual entrypoint for bounded dispatch tests."""
    spec = importlib.util.spec_from_file_location("v681_contract_cli", CLI)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_design_and_milestone_errors(authorities):
    """SCENARIO-REPORT-7837-AUTHORITY: malformed contracts fail closed."""
    design, staged, active = deepcopy(authorities)
    with pytest.raises(ValueError, match="JSON contract missing"):
        subject.parse_design(design.replace("V681_TASK_CONTRACT_START", "BROKEN"))
    with pytest.raises(ValueError, match="milestone mismatch"):
        subject.parse_design(
            design.replace('"milestone": "2026.09.681"', '"milestone": "wrong"', 1)
        )
    active["milestone"] = "wrong"
    assert "roadmap_milestone" in subject.compare_contract(design, staged, active)["errors"]


def test_snapshot_arity_and_raw_object(authorities, tmp_path):
    """SCENARIO-REPORT-7837-REPLAY: snapshots need three authorities."""
    with pytest.raises(ValueError, match="three authorities"):
        subject.snapshot_authorities((DESIGN,), tmp_path)
    rows = subject.compare_contract(*authorities)["rows"]
    candidate = tmp_path / "candidate.json"
    raw = tmp_path / "rows.json"
    candidate.write_text(
        json.dumps({"experiment_id": 7837, "task_id": "exp7837-contract-methods", "rows": rows})
    )
    raw.write_text(json.dumps({"rows": rows}))
    assert subject.cold_replay(candidate, raw)
    rows[0]["absolute_metric"] = 0
    candidate.write_text(
        json.dumps({"experiment_id": 7837, "task_id": "exp7837-contract-methods", "rows": rows})
    )
    raw.write_text(json.dumps(rows))
    assert not subject.cold_replay(candidate, raw)


def test_owned_children_and_sealed_logs(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7837-REPLAY: real exits, imports and deadlines survive."""
    cli = load_cli()
    monkeypatch.setattr(cli, "RAW", tmp_path)
    python = sys.executable
    valid = {
        "name": "worktree_imports",
        "argv": [
            python,
            "-c",
            f'import json; print(json.dumps({{"resolved_imports": {{"carnot": "{ROOT}/python/carnot/__init__.py"}}}}))',
        ],
        "timeout_s": 5,
        "classification": "required",
    }
    receipt = cli.run_child(valid, 0)
    assert receipt["passed"] and receipt["resolved_imports"]
    assert cli.run_child(valid, 1)["passed"]
    bad = dict(valid, argv=[python, "-c", 'print("worktree_imports_ok")'])
    assert not cli.run_child(bad, 2)["passed"]
    failed = dict(valid, name="failed", argv=[python, "-c", "raise SystemExit(3)"])
    assert cli.run_child(failed, 3)["exit_code"] == 3
    slow = dict(
        valid, name="deadline", argv=[python, "-c", "import time; time.sleep(2)"], timeout_s=0.1
    )
    assert cli.run_child(slow, 4)["timed_out"]
    source = tmp_path / "source.log"
    source.write_bytes(b"sealed")
    seal = cli.seal_log(source, "explicit")
    Path(seal["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="collision"):
        cli.seal_log(source, "explicit")


def test_private_terminal_states(tmp_path, authorities, monkeypatch):
    """SCENARIO-REPORT-7837-REPLAY: blocked and failed checks grant no readiness."""
    cli = load_cli()
    monkeypatch.setattr(cli, "RAW", tmp_path)
    output, raw = tmp_path / "result.json", tmp_path / "rows.json"
    args = ["--date", "20260928", "--output", str(output), "--raw", str(raw)]
    missing = ROOT / "docs/research-notes/absent-v681-input.md"
    monkeypatch.setattr(cli, "REQUIRED_SOURCES", (*cli.REQUIRED_SOURCES, missing))
    assert cli.main(args) == 0
    blocked = json.loads(output.read_text())
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"][0]["path"] == str(missing)
    monkeypatch.setattr(cli, "REQUIRED_SOURCES", cli.REQUIRED_SOURCES[:-1])
    original_compare = cli.compare_contract

    def mismatch(*items):
        result = original_compare(*items)
        result["passed"] = False
        return result

    monkeypatch.setattr(cli, "compare_contract", mismatch)
    assert cli.main(args) == 1
    assert json.loads(output.read_text())["contract_ready_score"] == 0


def test_frozen_manifest_and_private_mode(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7837-REPLAY: a changed roster never runs."""
    cli = load_cli()
    monkeypatch.setattr(cli, "RAW", tmp_path)
    monkeypatch.setattr(cli, "MANIFEST_SHA256", "bad")
    with pytest.raises(ValueError, match="manifest changed"):
        cli.main(["--output", str(tmp_path / "out.json")])
    with pytest.raises(SystemExit):
        cli.main(["--no-validation"])


def test_frozen_dispatch_terminal_reader(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7837-REPLAY: only the frozen successful roster qualifies."""
    cli = load_cli()
    monkeypatch.setattr(cli, "RAW", tmp_path)
    output, raw = tmp_path / "result.json", tmp_path / "rows.json"
    args = ["--date", "20260928", "--output", str(output), "--raw", str(raw)]
    state = {"failed": None, "flagged": 0}

    def fake_child(command, index):
        if command["name"] == "adversarial_verify":
            report = tmp_path / "verifier.json"
            report.write_text(json.dumps({"flagged_count": state["flagged"]}))
            log_path = str(report)
        else:
            log_path = str(tmp_path / "ordinary.log")
        return {
            "name": command["name"],
            "command_argv": command["argv"],
            "classification": command["classification"],
            "passed": command["name"] != state["failed"],
            "log_path": log_path,
            "exit_code": int(command["name"] == state["failed"]),
        }

    monkeypatch.setattr(cli, "run_child", fake_child)
    assert cli.main(args) == 0
    good = json.loads(output.read_text())
    assert good["contract_ready_score"] == 1
    assert good["verdict_class"] == "circular_positive"
    assert len(good["observed_child_commands"]) == 19
    state["failed"] = "ruff_check"
    assert cli.main(args) == 1
    failed = json.loads(output.read_text())
    assert failed["verdict_class"] == "disqualified"
    assert failed["contract_ready_score"] == 0
    state["failed"] = None
    state["flagged"] = 1
    assert cli.main(args) == 1
    assert json.loads(output.read_text())["flagged_adversarial"] is True
    state["flagged"] = 0
    verifier = tmp_path / "verifier.json"
    verifier.write_text("invalid JSON")

    def invalid_child(command, index):
        receipt = fake_child(command, index)
        if command["name"] == "adversarial_verify":
            verifier.write_text("invalid JSON")
        return receipt

    monkeypatch.setattr(cli, "run_child", invalid_child)
    assert cli.main(args) == 1
    assert json.loads(output.read_text())["flagged_adversarial"] is True


def test_cli_cold_replay_mode(authorities, tmp_path):
    """SCENARIO-REPORT-7837-REPLAY: CLI cold mode rejects identity mutation."""
    cli = load_cli()
    rows = subject.compare_contract(*authorities)["rows"]
    raw, candidate = tmp_path / "rows.json", tmp_path / "candidate.json"
    raw.write_text(json.dumps(rows))
    candidate.write_text(
        json.dumps({"experiment_id": 7837, "task_id": "exp7837-contract-methods", "rows": rows})
    )
    assert cli.main(["--cold-replay", str(candidate), "--raw", str(raw)]) == 0
    candidate.write_text(
        json.dumps(
            {
                "experiment_id": "exp7837-contract-methods",
                "task_id": "exp7837-contract-methods",
                "rows": rows,
            }
        )
    )
    assert cli.main(["--cold-replay", str(candidate), "--raw", str(raw)]) == 1


def test_stubborn_owned_child_and_snapshot_failure(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7837-REPLAY: kill only a child past deadline."""
    cli = load_cli()
    monkeypatch.setattr(cli, "RAW", tmp_path)
    child = {
        "name": "stubborn",
        "argv": [
            sys.executable,
            "-c",
            "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(10)",
        ],
        "timeout_s": 0.3,
        "classification": "required",
    }
    assert cli.run_child(child, 0)["timed_out"]
    monkeypatch.setattr(cli, "verify_snapshots", lambda snapshots: False)
    with pytest.raises(ValueError, match="snapshot cold verification failed"):
        cli.main(["--output", str(tmp_path / "result.json"), "--raw", str(tmp_path / "rows.json")])
