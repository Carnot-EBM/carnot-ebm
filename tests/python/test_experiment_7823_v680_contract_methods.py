"""REQ-REPORT-7823: V680 exact authority and terminal receipt tests."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_7823_v680_contract_methods as subject

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts/experiments/experiment_7823_v680_contract_methods.py"


def load_cli():
    """Load the entrypoint itself so dispatch checks its real functions."""
    spec = importlib.util.spec_from_file_location("exp7823_cli", CLI)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def authority():
    """SCENARIO-REPORT-7823-CONTRACT: inspect independent source bytes."""
    return (
        (ROOT / subject.DESIGN).read_text(),
        yaml.safe_load((ROOT / "research-roadmap.yaml").read_text()),
    )


def test_exact_contract_and_private_mutations(authority):
    """SCENARIO-REPORT-7823-CONTRACT: refuse each changed operand."""
    text, roadmap = authority
    result = subject.compare_contract(text, roadmap)
    assert result["passed"]
    assert len(result["rows"]) == 14
    assert [r["unit_id"].split("-")[0] for r in result["rows"]] == [
        f"exp{i}" for i in range(7823, 7837)
    ]
    for name in (
        "drop",
        "reorder",
        "title",
        "phase",
        "deliverable",
        "model",
        "substrate",
        "gate_field",
        "unknown_producer",
        "prior_retirement",
    ):
        private = subject.mutate(roadmap, name)
        assert private != roadmap
        assert not subject.compare_contract(text, private)["passed"], name
    assert not subject.compare_contract(
        text.replace(
            "Bind fourteen tasks and register shortcut and abstention hypotheses", "Stale title", 1
        ),
        roadmap,
    )["passed"]


def test_matching_authority_and_cold_snapshot(authority, tmp_path):
    """SCENARIO-REPORT-7823-CONTRACT: later active YAML cannot rewrite history."""
    text, roadmap = authority
    choice, actual, candidates = subject.resolve_authority(ROOT)
    assert choice.name == "research-roadmap.yaml"
    assert actual["milestone"] == subject.MILESTONE
    assert candidates[0]["exists"] is False
    later = deepcopy(roadmap)
    later["milestone"] = "2026.09.681"
    (tmp_path / "research-roadmap.yaml").write_text(yaml.safe_dump(later))
    with pytest.raises(ValueError, match="V680 authority"):
        subject.resolve_authority(tmp_path)
    rows = subject.compare_contract(text, roadmap)["rows"]
    raw = tmp_path / "rows.json"
    candidate = tmp_path / "candidate.json"
    raw.write_text(json.dumps(rows))
    candidate.write_text(json.dumps({"rows": rows}))
    assert subject.cold_validate(candidate, raw, ROOT)
    rows[0]["matched"] = False
    candidate.write_text(json.dumps({"rows": rows}))
    assert not subject.cold_validate(candidate, raw, ROOT)


def test_v679_custody():
    """SCENARIO-REPORT-7823-CUSTODY: queue receipts cannot fill absent outputs."""
    rows = subject.prior_inventory(ROOT)
    assert len(rows) == 14
    assert sum(r["producer_state"] == "missing" for r in rows) == 5
    assert sum(r["producer_state"] == "circular_positive" for r in rows) == 2
    assert sum(r["producer_state"] == "disqualified" for r in rows) == 4
    assert sum(r["producer_state"] == "blocked" for r in rows) == 3
    assert all(r["pre_gate_path"] != r["producer_path"] for r in rows if r["pre_gate_path"])


def test_real_dispatch_and_log_seal(tmp_path):
    """SCENARIO-REPORT-7823-TERMINAL: frozen argv and immutable exited logs."""
    cli = load_cli()
    manifest = cli.load_manifest(ROOT / subject.MANIFEST)
    observed = []

    def record(command):
        observed.append((command["name"], command["argv"], command["classification"]))
        return {"name": command["name"]}

    cli.dispatch(manifest, record)
    assert observed == [(c["name"], c["argv"], c["classification"]) for c in manifest["commands"]]
    assert len(observed) == 21
    bad = deepcopy(manifest)
    bad["commands"].append(dict(bad["commands"][0], name="undeclared"))
    with pytest.raises(ValueError, match="manifest"):
        cli.dispatch(bad, record)
    missing = tmp_path / "nested" / "pytest" / "run"
    with pytest.raises(ValueError, match="parent"):
        cli.require_pytest_parents([{"argv": [f"--basetemp={missing}"]}])
    missing.parent.mkdir(parents=True)
    cli.require_pytest_parents([{"argv": [f"--basetemp={missing}"]}])
    source = tmp_path / "child.log"
    source.write_bytes(b"first\n")
    first = cli.seal_log(source, tmp_path / "durable", "one")
    source.write_bytes(b"later\n")
    second = cli.seal_log(source, tmp_path / "durable", "two")
    assert first["path"] != second["path"]
    assert cli.verify_seal(first) and cli.verify_seal(second)
    Path(first["path"]).write_bytes(b"firsT\n")
    assert not cli.verify_seal(first)


def test_cli_modes_and_candidate(authority, tmp_path, monkeypatch):
    """SCENARIO-REPORT-7823-TERMINAL: real CLI modes keep evidence separate."""
    cli = load_cli()
    assert cli.main(["--check-authority"]) == 0
    text, roadmap = authority
    rows = subject.compare_contract(text, roadmap)["rows"]
    raw = tmp_path / "rows.json"
    candidate = tmp_path / "candidate.json"
    raw.write_text(json.dumps(rows))
    candidate.write_text(json.dumps({"rows": rows}))
    assert cli.main(["--cold-validate", str(candidate), "--raw", str(raw)]) == 0
    candidate.write_text(json.dumps({"rows": []}))
    assert cli.main(["--cold-validate", str(candidate), "--raw", str(raw)]) == 1
    monkeypatch.setattr(
        cli,
        "run_experiment",
        lambda date: {
            "run_date": date,
            "honest_verdict": "complete_circular_positive_v680_contract_methods",
            "contract_ready_score": 1,
        },
    )
    assert cli.main(["--date", "20260928"]) == 0


def test_negative_reader_boundaries(authority, tmp_path):
    """SCENARIO-REPORT-7823-CONTRACT: malformed archived inputs fail closed."""
    text, roadmap = authority
    with pytest.raises(ValueError, match="JSON contract missing"):
        subject.parse_design(text.replace("<!-- V680_TASK_CONTRACT_START -->", "<!-- absent -->"))
    with pytest.raises(ValueError, match="milestone mismatch"):
        subject.parse_design(
            text.replace('"milestone": "2026.09.680"', '"milestone": "2026.09.681"', 1)
        )
    changed = deepcopy(roadmap)
    changed["milestone"] = "2026.09.681"
    assert not subject.compare_contract(text, changed)["passed"]
    with pytest.raises(ValueError, match="unknown mutation"):
        subject.mutate(roadmap, "bogus")
    archive = yaml.safe_load(
        (ROOT / "docs/research-notes/v679-authority-snapshots/roadmap.yaml").read_text()
    )
    archive["tasks"][0]["id"] = "wrong-id"
    preserved = tmp_path / "docs/research-notes/v679-authority-snapshots"
    preserved.mkdir(parents=True)
    (preserved / "roadmap.yaml").write_text(yaml.safe_dump(archive))
    with pytest.raises(ValueError, match="order changed"):
        subject.prior_inventory(tmp_path)
    raw = tmp_path / "rows.json"
    candidate = tmp_path / "candidate.json"
    raw.write_text("[]")
    candidate.write_text('{"rows": []}')
    assert not subject.cold_validate(candidate, raw, tmp_path)


def test_execution_replay_and_verdict_boundaries(authority, tmp_path, monkeypatch):
    """SCENARIO-REPORT-7823-TERMINAL: execute complete roster with fake exited children."""
    cli = load_cli()
    manifest = deepcopy(cli.load_manifest(ROOT / subject.MANIFEST))
    monkeypatch.setattr(cli, "RAW", tmp_path / "raw")
    manifest["private_root"] = str(tmp_path / "private")
    manifest["commands"] = cli.runtime_plan(manifest)
    callbacks = []

    def fake_run(root, commands, *, log_dir, heartbeat_s):
        command = commands[0]
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_dir / (command.name + ".log")
        log.write_text("exited 0\n")
        return [
            {
                "name": command.name,
                "command_argv": list(command.argv),
                "exit_code": 0,
                "passed": True,
                "log_path": str(log),
                "log_sha256": cli.sha256_file(log),
                "timed_out": False,
            }
        ]

    monkeypatch.setattr(cli, "run_commands", fake_run)
    receipts = cli.execute_plan(manifest, lambda done: callbacks.append(len(done)))
    assert len(receipts) == 21
    assert callbacks == [17]
    assert all(
        cli.verify_seal({"path": r["log_path"], "sha256": r["log_sha256"]}) for r in receipts
    )
    first = tmp_path / "source.log"
    first.write_text("once")
    cli.seal_log(first, tmp_path / "sealed", "same-attempt")
    with pytest.raises(ValueError, match="already exists"):
        cli.seal_log(first, tmp_path / "sealed", "same-attempt")
    changed_manifest = tmp_path / "manifest.json"
    changed_manifest.write_text("{}")
    with pytest.raises(ValueError, match="byte hash changed"):
        cli.load_manifest(changed_manifest)

    monkeypatch.setattr(cli, "RESULT", tmp_path / "result.json")
    monkeypatch.setattr(cli, "load_manifest", lambda path: manifest)
    monkeypatch.setattr(cli, "execute_plan", lambda plan, callback: (callback([]), [])[1])
    value = cli.run_experiment("20260928")
    assert value["contract_ready_score"] == 1
    assert (tmp_path / "result.json").is_file()
    assert (Path(manifest["private_root"]) / "candidate.json").is_file()
    with pytest.raises(ValueError, match="run date"):
        cli.run_experiment("20260927")

    text, roadmap = authority
    comparison = subject.compare_contract(text, roadmap)
    prior = subject.prior_inventory(ROOT)
    authority_path = ROOT / "research-roadmap.yaml"
    source = {"path": "missing-external", "exists": False, "sha256": None}
    blocked = cli.build_artifact(comparison, prior, [source], [], [], [], authority_path, [])
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"][0]["field"] == "exists"
    failed = {
        "name": "adversarial_verify",
        "classification": "required",
        "passed": False,
        "log_path": "log",
        "log_sha256": "sha256:bad",
        "exit_code": 1,
        "command_argv": ["bad"],
    }
    disqualified = cli.build_artifact(comparison, prior, [], [failed], [], [], authority_path, [])
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["flagged_adversarial"] is True
    assert disqualified["contract_ready_score"] == 0


def test_remaining_rejection_paths(authority, tmp_path, monkeypatch):
    """SCENARIO-REPORT-7823-TERMINAL: altered bytes and diagnostic exits stay distinct."""
    import runpy
    import sys

    text, roadmap = authority
    cli = load_cli()
    comparison = subject.compare_contract(text, roadmap)
    comparison["rows"][0]["checks"]["title"] = False
    comparison["rows"][0]["matched"] = False
    diagnostic = {
        "name": "repository_health_full_python_suite",
        "classification": "diagnostic",
        "passed": False,
        "exit_code": 1,
        "log_path": "old",
        "log_sha256": "sha256:old",
        "command_argv": ["pytest"],
    }
    value = cli.build_artifact(
        comparison,
        subject.prior_inventory(ROOT),
        [],
        [diagnostic],
        [],
        [],
        ROOT / "research-roadmap.yaml",
        [],
    )
    assert value["verdict_class"] == "disqualified"
    assert all(row["upstream_id"] != diagnostic["name"] for row in value["gate_check_summary"])
    assert any(row["field"] == "title" for row in value["gate_check_summary"])

    stale = tmp_path / "stale-design.md"
    stale.write_text(
        text.replace(
            "Bind fourteen tasks and register shortcut and abstention hypotheses", "Stale title", 1
        )
    )
    monkeypatch.setattr(cli, "DESIGN_SNAPSHOT", stale)
    with pytest.raises(ValueError, match="live authorities differ"):
        cli.run_experiment("20260928")

    preserved = tmp_path / "docs/research-notes/v679-authority-snapshots"
    preserved.mkdir(parents=True)
    (preserved / "roadmap.yaml").write_text(
        (ROOT / "docs/research-notes/v679-authority-snapshots/roadmap.yaml").read_text()
    )
    results = tmp_path / "results"
    results.mkdir()
    (results / "experiment_7812_other.json").write_text('{"schema":"unrelated"}')
    assert len(subject.prior_inventory(tmp_path)) == 14

    monkeypatch.setattr(sys, "argv", [str(CLI), "--check-authority"])
    with pytest.raises(SystemExit) as exited:
        runpy.run_path(str(CLI), run_name="__main__")
    assert exited.value.code == 0
