"""REQ-REPORT-7879-V684: current authority and executable scope tests."""

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from carnot.reporting.roadmap_contract import compare_contract


ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts/experiments/experiment_7879_v684_contract_methods.py"
DESIGN = ROOT / "openspec/change-proposals/research-roadmap-vNEXT.md"
ACTIVE = ROOT / "research-roadmap.yaml"


def owned():
    """Load only the current owned module by its real path."""
    spec = importlib.util.spec_from_file_location("v684_contract_methods", CLI)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_stage_is_terminal_and_active_is_real(tmp_path):
    """SCENARIO-REPORT-7879-V684-AUTHORITY preserves actual activation state."""
    result = owned().inspect_authorities(DESIGN, tmp_path / "absent.yaml", ACTIVE)
    assert result["active_milestone"] == "2026.09.684"
    assert result["staged_present"] is False
    assert result["contract_ready_score"] == 0
    assert result["gate_check_summary"][0]["artifact_field"] == "exists"
    assert result["gate_check_summary"][0]["observed"] is False
    assert len(result["contract_rows"]) == 12


@pytest.mark.parametrize(
    "field",
    [
        "count",
        "order",
        "id",
        "title",
        "phase",
        "deliverable",
        "MODEL_SPECS",
        "inference_substrate_class",
        "gated_on",
        "prior_failures",
    ],
)
def test_private_contract_mutations_fail(field):
    """SCENARIO-REPORT-7879-V684-SCOPE rejects changed contract bytes."""
    active = yaml.safe_load(ACTIVE.read_text())
    staged = deepcopy(active)
    tasks = staged["tasks"]
    if field == "count":
        tasks.pop()
    elif field == "order":
        tasks[0], tasks[1] = tasks[1], tasks[0]
    elif field == "prior_failures":
        tasks[0][field][0].pop("addressed_by")
    elif field == "gated_on":
        tasks[3][field][0]["artifact_field"] = "wrong"
    else:
        tasks[0][field] = "wrong"
    comparison = compare_contract(
        DESIGN.read_text(),
        staged,
        active,
        milestone="2026.09.684",
        first_id=7879,
        count=12,
    )
    assert not comparison["passed"]


def test_manifest_rejects_nonexistent_arc_empty_coverage_and_venue(tmp_path):
    """SCENARIO-REPORT-7879-V684-SCOPE checks paths before dispatch."""
    cli = owned()
    manifest = cli.build_manifest(ROOT, tmp_path)
    assert cli.validate_manifest(manifest, ROOT) == []
    bad = deepcopy(manifest)
    bad["task_scopes"]["exp7887-arc-supervisor-delta"]["tests"] = [
        "tests/python/test_arc_supervisor_delta_7874_nonexistent.py"
    ]
    assert "missing_test" in cli.validate_manifest(bad, ROOT)
    assert "empty_coverage" in cli.validate_coverage([], 0)
    assert "illegal_venue" in cli.validate_venue("host_cpu")


def test_private_cli_cold_replay_and_fields(tmp_path):
    """SCENARIO-REPORT-7879-V684-AUTHORITY exercises the real CLI."""
    output, raw = tmp_path / "candidate.json", tmp_path / "rows.json"
    clean_env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    command = [
        sys.executable,
        str(CLI),
        "--date",
        "20260929",
        "--output",
        str(output),
        "--raw",
        str(raw),
        "--no-validation",
    ]
    run = subprocess.run(
        command, cwd=tmp_path, env=clean_env, capture_output=True, text=True, timeout=60
    )
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "blocked"
    assert value["honest_verdict"].startswith("complete_blocked_")
    assert value["contract_ready_score"] == 0
    assert value["execution_venue"] == "host"
    assert value["MODEL_SPECS"] == value["model_specs"] == []
    assert len(value["rows"]) == len(value["contract_rows"]) == 12
    replay = subprocess.run(
        [sys.executable, str(CLI), "--cold-replay", str(output), "--raw", str(raw)],
        cwd=tmp_path,
        env=clean_env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert replay.returncode == 0, replay.stdout + replay.stderr
    value["rows"][0]["absolute_metric"] = 1 - value["rows"][0]["absolute_metric"]
    output.write_text(json.dumps(value))
    replay = subprocess.run(
        [sys.executable, str(CLI), "--cold-replay", str(output), "--raw", str(raw)],
        cwd=tmp_path,
        env=clean_env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert replay.returncode != 0


def test_staged_match_and_wrong_active_are_distinct(tmp_path):
    """SCENARIO-REPORT-7879-V684-AUTHORITY tests both genuine authority routes."""
    cli = owned()
    staged = tmp_path / "staged.yaml"
    staged.write_bytes(ACTIVE.read_bytes())
    matched = cli.inspect_authorities(DESIGN, staged, ACTIVE)
    assert matched["contract_ready_score"] == 1
    assert all(row["matched"] for row in matched["contract_rows"])
    wrong = tmp_path / "active.yaml"
    data = yaml.safe_load(ACTIVE.read_text())
    data["milestone"] = "2026.09.683"
    wrong.write_text(yaml.safe_dump(data))
    result = cli.inspect_authorities(DESIGN, staged, wrong)
    assert result["contract_ready_score"] == 0
    assert {x["artifact_field"] for x in result["gate_check_summary"]} >= {"milestone"}


def test_mutation_receipts_and_immutable_snapshots(tmp_path):
    """SCENARIO-REPORT-7879-V684-SCOPE retains defects and exact source bytes."""
    cli = owned()
    rows = cli.mutation_rows(DESIGN, ACTIVE)
    assert len(rows) >= 10 and all(row["rejected"] for row in rows)
    snapshots = cli.snapshot_sources([(DESIGN, "design"), (ACTIVE, "active")], tmp_path)
    assert len(snapshots) == 2
    assert all(
        Path(row["snapshot_path"]).read_bytes() == (ROOT / row["path"]).read_bytes()
        for row in snapshots
    )
    Path(snapshots[0]["snapshot_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="immutable"):
        cli.snapshot_sources([(DESIGN, "design")], tmp_path)


def test_manifest_import_and_coverage_failure_routes(tmp_path):
    """SCENARIO-REPORT-7879-V684-SCOPE rejects silent validation defects."""
    cli = owned()
    manifest = cli.build_manifest(ROOT, tmp_path)
    bad = deepcopy(manifest)
    bad["commands"][0]["argv"][0] = str(tmp_path / "missing")
    bad["module_roots"]["scripts.roadmap_schema"] = "scripts/missing.py"
    assert set(cli.validate_manifest(bad, ROOT)) == {"invalid_command", "missing_module"}
    assert cli.validate_coverage(["current.py"], 1) == []
    assert cli.validate_venue("host") == []
    assert cli.resolve_imports(manifest, ROOT)
    with pytest.raises(ValueError, match="import path mismatch"):
        cli.resolve_imports(bad | {"module_roots": {"scripts.roadmap_schema": "CODEX.md"}}, ROOT)


def test_terminal_manifest_and_sealed_child_receipt(tmp_path):
    """SCENARIO-REPORT-7879-V684-SCOPE seals completed logs and exact argv."""
    cli = owned()
    manifest = cli.build_manifest(ROOT, tmp_path)
    names = {row["name"] for row in manifest["commands"]}
    assert {
        "coverage_combine",
        "coverage_report",
        "cold_replay",
        "adversarial_verify",
        "strict_row_lint",
    } <= names
    log = tmp_path / "child.log"
    log.write_text("done\n")
    receipt = {
        "name": "probe",
        "log_path": str(log),
        "log_sha256": cli.sha256_file(log),
        "passed": True,
        "command_argv": [sys.executable, "-V"],
    }
    sealed = cli.seal_receipts([receipt], tmp_path / "sealed")
    assert Path(sealed[0]["log_path"]).read_bytes() == b"done\n"
    assert sealed[0]["classification"] == "required"


def test_artifact_classifies_owned_failure_separately(tmp_path):
    """SCENARIO-REPORT-7879-V684-AUTHORITY preserves failed owned checks."""
    cli = owned()
    manifest = cli.build_manifest(ROOT, tmp_path)
    path = tmp_path / "manifest.json"
    cli.atomic_json(path, manifest)
    authority = cli.inspect_authorities(DESIGN, ROOT / "research-roadmap-next.yaml", ACTIVE)
    sources = [cli.source_row(DESIGN, "design", "20260929")]
    failed = {
        "name": "affected_pytest",
        "classification": "required",
        "passed": False,
        "command_argv": [sys.executable, "-V"],
        "exit_code": 1,
    }
    result = cli.artifact("20260929", authority, manifest, path, sources, [failed], {})
    assert result["verdict_class"] == "disqualified"
    assert result["acceptance_gate_results"]["readiness"] == 0
    assert len(result["historical_required_failures"]) >= 2


def test_sealed_log_rejects_byte_changes_and_existing_collision(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7879-V684-SCOPE checks receipt bytes before sealing."""
    cli = owned()
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    original = tmp_path / "child.log"
    original.write_bytes(b"first")
    receipt = {
        "name": "probe",
        "log_path": "child.log",
        "log_sha256": cli.sha256_file(original),
        "passed": True,
    }
    directory = tmp_path / "sealed"
    sealed = cli.seal_receipts([receipt], directory)
    assert Path(sealed[0]["log_path"]).read_bytes() == b"first"
    original.write_bytes(b"changed")
    with pytest.raises(ValueError, match="child log changed"):
        cli.seal_receipts([receipt], directory)
    original.write_bytes(b"first")
    Path(sealed[0]["log_path"]).write_bytes(b"collision")
    with pytest.raises(ValueError, match="immutable child log"):
        cli.seal_receipts([receipt], directory)


def test_supervised_validation_and_terminal_fail_closed(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7879-V684-SCOPE runs both validation phases in private scratch."""
    cli = owned()
    scratch = tmp_path / "scratch"
    output, raw = tmp_path / "candidate.json", tmp_path / "rows.json"
    terminal_calls = 0
    coverage_present = False

    def fake_run(root, commands, *, log_dir):
        nonlocal terminal_calls
        assert root == ROOT
        log_dir.mkdir(parents=True, exist_ok=True)
        terminal = any(command.name == "cold_replay" for command in commands)
        if terminal:
            terminal_calls += 1
        elif coverage_present:
            cli.atomic_json(
                scratch / "coverage.json",
                {"files": {"current.py": {"summary": {"num_statements": 1}}}},
            )
        receipts = []
        for command in commands:
            path = log_dir / f"{command.name}.log"
            flagged = command.name == "adversarial_verify" and terminal_calls == 1
            path.write_text(json.dumps({"flagged_count": int(flagged)}))
            receipts.append(
                {
                    "name": command.name,
                    "command_argv": list(command.argv),
                    "classification": "required",
                    "passed": True,
                    "log_path": str(path),
                    "log_sha256": cli.sha256_file(path),
                    "exit_code": 0,
                }
            )
        return receipts

    monkeypatch.setattr(cli, "run_commands", fake_run)
    args = [
        "--date",
        "20260929",
        "--output",
        str(output),
        "--raw",
        str(raw),
        "--scratch",
        str(scratch),
    ]
    assert cli.main(args) == 0
    first = json.loads(output.read_text())
    assert first["verdict_class"] == "disqualified"
    assert any(
        row.get("coverage_error") == "empty_coverage" for row in first["validation_receipts"]
    )
    assert cli.main(["--cold-replay", str(output), "--raw", str(raw)]) == 0
    assert cli.main(["--cold-replay", str(output)]) == 1
    coverage_present = True
    terminal_calls = 0
    assert cli.main(args) == 0
    second = json.loads(output.read_text())
    assert second["flagged_adversarial"] is True
    assert second["honest_verdict"] == "complete_disqualified_v684_terminal_validation"
    assert terminal_calls == 2
    assert not any(row.get("coverage_error") for row in second["validation_receipts"])
    assert (scratch / "terminal-validation-receipts.json").is_file()
    original_atomic_json = cli.atomic_json

    def corrupt_published(path, value):
        original_atomic_json(path, value)
        if path == output:
            path.write_text("changed")

    monkeypatch.setattr(cli, "atomic_json", corrupt_published)
    with pytest.raises(ValueError, match="published bytes differ"):
        cli.main(args)


def test_private_cli_rejects_invalid_scope_and_public_private_output(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7879-V684-SCOPE checks scope before dispatch."""
    cli = owned()
    with pytest.raises(SystemExit):
        cli.main(["--no-validation"])
    monkeypatch.setattr(cli, "validate_manifest", lambda manifest, root: ["missing_test"])
    with pytest.raises(ValueError, match="invalid declared scope"):
        cli.main(
            ["--output", str(tmp_path / "candidate.json"), "--scratch", str(tmp_path / "scratch")]
        )
