"""REQ-REPORT-7809: authentic snapshots and real dispatcher behavior."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_7809_v679_contract_methods as subject

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts/experiments/experiment_7809_v679_contract_methods.py"


def load_cli():
    """Import the actual command entrypoint so dispatch tests cover its code."""
    spec = importlib.util.spec_from_file_location("exp7809_cli", CLI)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if (
        yaml.safe_load((ROOT / "research-roadmap.yaml").read_text())["milestone"]
        != subject.MILESTONE
    ):
        preserved = ROOT / subject.YAML_SNAPSHOT
        module.resolve_authority = lambda root: (
            preserved,
            yaml.safe_load(preserved.read_text()),
            [{"path": str(subject.YAML_SNAPSHOT), "exists": True, "milestone": subject.MILESTONE}],
        )
        module.DESIGN = subject.DESIGN_SNAPSHOT
    return module


@pytest.fixture
def authorities():
    """SCENARIO-REPORT-7809-CONTRACT: use immutable historical bytes."""
    return (
        (ROOT / subject.DESIGN_SNAPSHOT).read_text(),
        yaml.safe_load((ROOT / subject.YAML_SNAPSHOT).read_text()),
    )


def test_authentic_three_way_contract(authorities):
    """SCENARIO-REPORT-7809-CONTRACT: all independent fields agree."""
    design, roadmap = authorities
    result = subject.compare_contract(design, roadmap)
    assert result["passed"]
    assert len(result["rows"]) == 14
    assert [r["unit_id"].split("-")[0] for r in result["rows"]] == [
        f"exp{i}" for i in range(7809, 7823)
    ]
    assert all(all(r["checks"].values()) for r in result["rows"])


@pytest.mark.parametrize(
    "mutation",
    [
        "drop",
        "reorder",
        "title",
        "phase",
        "deliverable",
        "model",
        "substrate",
        "unknown_producer",
        "gate_field",
        "prior_experiment_id",
        "prior_verdict",
        "prior_addressed_by",
        "prior_retirement",
    ],
)
def test_private_mutations_rejected(authorities, mutation):
    """SCENARIO-REPORT-7809-CONTRACT: each altered field fails alone."""
    design, roadmap = authorities
    changed = subject.mutate(roadmap, mutation)
    assert not subject.compare_contract(design, changed)["passed"]
    assert changed != roadmap


def test_stale_design_and_rollover(authorities, tmp_path):
    """SCENARIO-REPORT-7809-CONTRACT: later active YAML cannot replace history."""
    design, roadmap = authorities
    assert not subject.compare_contract(
        design.replace(
            "Bind fourteen tasks to immutable validation and research protocols", "Stale title", 1
        ),
        roadmap,
    )["passed"]
    assert not subject.compare_contract(
        design.replace(
            '"MODEL_SPECS": ["unsloth/Qwen3.8-27B-GGUF"]', '"MODEL_SPECS": ["wrong/model"]', 1
        ),
        roadmap,
    )["passed"]
    later = deepcopy(roadmap)
    later["milestone"] = "2026.09.680"
    (tmp_path / "research-roadmap.yaml").write_text(yaml.safe_dump(later))
    with pytest.raises(ValueError, match="V679 authority"):
        subject.resolve_authority(tmp_path)
    assert subject.compare_contract(design, roadmap)["passed"]


def test_v678_custody_and_queue_distinction():
    """SCENARIO-REPORT-7809-CUSTODY: missing science stays missing."""
    rows = subject.prior_inventory(ROOT)
    assert len(rows) == 14
    assert sum(r["producer_state"] == "missing" for r in rows) == 6
    assert sum(r["producer_state"] == "circular_positive" for r in rows) == 1
    assert sum(r["producer_state"] == "disqualified" for r in rows) == 6
    assert sum(r["producer_state"] == "blocked" for r in rows) == 1
    assert all(r["pre_gate_path"] != r["producer_path"] for r in rows if r["pre_gate_path"])


def test_real_dispatcher_matches_frozen_manifest():
    """SCENARIO-REPORT-7809-TERMINAL: record every child including appended guards."""
    cli = load_cli()
    manifest = cli.load_manifest(ROOT / subject.MANIFEST)
    observed = []

    def record(command):
        observed.append((command["name"], command["argv"], command["classification"]))
        return {
            "name": command["name"],
            "command_argv": command["argv"],
            "exit_code": 0,
            "passed": True,
            "log_path": "sealed",
            "log_sha256": "sha256:ok",
        }

    cli.dispatch(manifest, record)
    assert observed == [(c["name"], c["argv"], c["classification"]) for c in manifest["commands"]]
    bad = deepcopy(manifest)
    bad["commands"].append(dict(bad["commands"][0], name="undeclared"))
    with pytest.raises(ValueError, match="manifest"):
        cli.dispatch(bad, record)
    assert len(observed) == 21


def test_missing_parent_retry_and_byte_mutation(tmp_path):
    """SCENARIO-REPORT-7809-TERMINAL: logs seal only after a completed attempt."""
    cli = load_cli()
    source = tmp_path / "child.log"
    source.write_bytes(b"first\n")
    missing = tmp_path / "nested" / "pytest" / "run"
    with pytest.raises(ValueError, match="parent"):
        cli.require_pytest_parents([{"argv": [f"--basetemp={missing}"]}])
    missing.parent.mkdir(parents=True)
    cli.require_pytest_parents([{"argv": [f"--basetemp={missing}"]}])
    first = cli.seal_log(source, tmp_path / "durable", "attempt-1")
    source.write_bytes(b"later\n")
    second = cli.seal_log(source, tmp_path / "durable", "attempt-2")
    assert first["path"] != second["path"]
    assert first["sha256"] != second["sha256"]
    assert Path(first["path"]).read_bytes() == b"first\n"
    Path(first["path"]).write_bytes(b"firsT\n")
    assert not cli.verify_seal(first)


def test_cold_rows_reject_one_byte(authorities, tmp_path):
    """SCENARIO-REPORT-7809-TERMINAL: summary bytes must match raw units."""
    design, roadmap = authorities
    rows = subject.compare_contract(design, roadmap)["rows"]
    raw = tmp_path / "rows.json"
    candidate = tmp_path / "candidate.json"
    raw.write_text(json.dumps(rows))
    candidate.write_text(json.dumps({"rows": rows}))
    assert subject.cold_validate(candidate, raw, ROOT)
    rows[0]["matched"] = False
    candidate.write_text(json.dumps({"rows": rows}))
    assert not subject.cold_validate(candidate, raw, ROOT)


def test_orchestration_with_recording_children(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7809-TERMINAL: orchestration records every phase without live children."""
    cli = load_cli()
    frozen = cli.load_manifest(ROOT / subject.MANIFEST)
    private = tmp_path / "private"
    copied = deepcopy(frozen)
    copied["private_root"] = str(private)
    monkeypatch.setattr(cli, "load_manifest", lambda path: copied)
    monkeypatch.setattr(cli, "RAW", tmp_path / "raw")
    monkeypatch.setattr(cli, "RESULT", tmp_path / "result.json")
    monkeypatch.setattr(
        cli,
        "dispatch",
        lambda manifest, executor: [executor(command) for command in manifest["commands"]],
    )

    def fake_run(root, commands, *, log_dir, heartbeat_s):
        command = commands[0]
        log_dir.mkdir(parents=True)
        log = log_dir / ("00_" + command.name + ".log")
        log.write_text("child exited 0\n")
        return [
            {
                "name": command.name,
                "command_argv": list(command.argv),
                "passed": True,
                "exit_code": 0,
                "log_path": str(log),
                "log_sha256": cli.sha256_file(log),
                "timed_out": False,
            }
        ]

    monkeypatch.setattr(cli, "run_commands", fake_run)
    value = cli.run_experiment("20260928")
    assert value["verdict_class"] == "circular_positive"
    assert len(value["validation_receipts"]) == 21
    assert all(
        cli.verify_seal({"path": str(ROOT / row["log_path"]), "sha256": row["log_sha256"]})
        for row in value["validation_receipts"]
    )
    assert value["repository_health"]["classification"] == "diagnostic"
    assert json.loads((tmp_path / "result.json").read_text())["rows"] == value["rows"]
    with pytest.raises(ValueError, match="run date"):
        cli.run_experiment("20260927")


def test_cli_modes_and_contract_errors(authorities, tmp_path, monkeypatch):
    """SCENARIO-REPORT-7809-TERMINAL: each real CLI mode reports its own boundary."""
    cli = load_cli()
    assert cli.main(["--check-authority"]) == 0
    design, roadmap = authorities
    raw = tmp_path / "rows.json"
    candidate = tmp_path / "candidate.json"
    rows = subject.compare_contract(design, roadmap)["rows"]
    raw.write_text(json.dumps(rows))
    candidate.write_text(json.dumps({"rows": rows}))
    assert cli.main(["--cold-validate", str(candidate), "--raw", str(raw)]) == 0
    candidate.write_text('{"rows": []}')
    assert cli.main(["--cold-validate", str(candidate), "--raw", str(raw)]) == 1
    monkeypatch.setattr(
        cli,
        "run_experiment",
        lambda date: {
            "honest_verdict": "complete_disqualified_fixture",
            "verdict_class": "disqualified",
        },
    )
    assert cli.main(["--date", "20260928"]) == 0
    with pytest.raises(ValueError, match="JSON contract missing"):
        subject.parse_design(design.replace("V679_TASK_CONTRACT_START", "BROKEN_START"))
    with pytest.raises(ValueError, match="JSON milestone mismatch"):
        subject.parse_design(
            design.replace('"milestone": "2026.09.679"', '"milestone": "2026.09.680"')
        )
    with pytest.raises(ValueError, match="unknown mutation"):
        subject.mutate(roadmap, "unknown")


def test_failed_receipts_do_not_open_readiness(authorities):
    """SCENARIO-REPORT-7809-TERMINAL: required failure zeros administrative readiness."""
    cli = load_cli()
    design, roadmap = authorities
    comparison = subject.compare_contract(design, roadmap)
    prior = subject.prior_inventory(ROOT)
    receipts = [
        {
            "name": "focused_pytest",
            "classification": "required",
            "passed": False,
            "exit_code": 1,
            "log_path": "failed.log",
            "log_sha256": "sha256:bad",
            "command_argv": ["pytest"],
        },
        {
            "name": "repository_health_full_python_suite",
            "classification": "diagnostic",
            "passed": False,
            "exit_code": -15,
            "log_path": "broad.log",
            "log_sha256": "sha256:broad",
            "command_argv": ["pytest"],
        },
    ]
    value = cli.build_artifact(
        comparison, prior, [], receipts, [], [], ROOT / "research-roadmap.yaml", []
    )
    assert value["verdict_class"] == "disqualified"
    assert value["acceptance_gate_results"]["readiness"] == 0
    assert len(value["gate_check_summary"]) == 1
    assert value["repository_health"]["exit_code"] == -15


def test_remaining_refusal_paths(authorities, tmp_path, monkeypatch):
    """SCENARIO-REPORT-7809-CONTRACT: malformed custody and bytes fail closed."""
    import runpy
    import sys

    cli = load_cli()
    design, roadmap = authorities
    wrong_milestone = deepcopy(roadmap)
    wrong_milestone["milestone"] = "2026.09.680"
    assert "roadmap_milestone" in subject.compare_contract(design, wrong_milestone)["errors"]
    archive = tmp_path / "docs/research-notes/v678-authority-snapshots/roadmap.yaml"
    archive.parent.mkdir(parents=True)
    previous = yaml.safe_load(
        (ROOT / "docs/research-notes/v678-authority-snapshots/roadmap.yaml").read_text()
    )
    previous["tasks"][0], previous["tasks"][1] = previous["tasks"][1], previous["tasks"][0]
    archive.write_text(yaml.safe_dump(previous))
    with pytest.raises(ValueError, match="V678 task order"):
        subject.prior_inventory(tmp_path)
    previous["tasks"][0], previous["tasks"][1] = previous["tasks"][1], previous["tasks"][0]
    archive.write_text(yaml.safe_dump(previous))
    (tmp_path / "results").mkdir()
    (tmp_path / "results/experiment_7798_wrong.json").write_text('{"schema":"not_a_receipt"}')
    assert sum(r["producer_state"] == "missing" for r in subject.prior_inventory(tmp_path)) == 14
    raw = tmp_path / "rows.json"
    candidate = tmp_path / "candidate.json"
    raw.write_text("[]")
    candidate.write_text('{"rows": []}')
    assert not subject.cold_validate(candidate, raw, tmp_path)

    changed_manifest = tmp_path / "manifest.json"
    changed_manifest.write_text("{}")
    with pytest.raises(ValueError, match="manifest byte hash"):
        cli.load_manifest(changed_manifest)
    source = tmp_path / "child.log"
    source.write_bytes(b"once")
    cli.seal_log(source, tmp_path / "durable", "attempt")
    with pytest.raises(ValueError, match="already exists"):
        cli.seal_log(source, tmp_path / "durable", "attempt")
    mismatch = subject.compare_contract(design, subject.mutate(roadmap, "title"))
    blocked = cli.build_artifact(
        mismatch,
        [],
        [{"path": "missing", "exists": False}],
        [],
        [],
        [],
        ROOT / "research-roadmap.yaml",
        [],
    )
    assert blocked["verdict_class"] == "blocked"
    assert any(r["field"] == "title" for r in blocked["gate_check_summary"])

    altered = tmp_path / "valid_but_different_design.md"
    altered.write_text(design + "\n")
    monkeypatch.setattr(cli, "DESIGN", altered)
    with pytest.raises(ValueError, match="immutable snapshots"):
        cli.run_experiment("20260928")
    monkeypatch.setattr(sys, "argv", [str(CLI), "--check-authority"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(CLI), run_name="__main__")
    assert exc.value.code == 0
