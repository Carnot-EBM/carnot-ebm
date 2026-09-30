"""REQ-REPORT-7915-V687: preserve historical authority and fixture identity."""

from copy import deepcopy
import gzip
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from carnot.reporting import v687_contract_methods as methods
from carnot.reporting import v687_contract_validation as validation
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.v685_authority_lifecycle import assess_authorities
from scripts.experiments import experiment_7915_v687_contract_methods as cli

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/experiment_7892_v685_source_boundary.json"


@pytest.fixture
def authorities(tmp_path):
    """SCENARIO-REPORT-7915-AUTHORITY: freeze thirteen private task bytes."""
    validation.prepare(ROOT, tmp_path)
    staged = tmp_path / "stage.yaml"
    staged.write_bytes((tmp_path / "active.yaml").read_bytes())
    return tmp_path / "design.md", staged, tmp_path / "active.yaml", tmp_path / "snapshots"


def test_lifecycle_defaults_and_explicit_count(authorities, tmp_path):
    """SCENARIO-REPORT-7915-AUTHORITY: historical consumers keep twelve slots."""
    design, staged, active, snapshots = authorities
    result = methods.assess(*authorities)
    assert result["activated"] and len(result["contract_rows"]) == 13
    assert not assess_authorities(
        design, staged, active, snapshots, milestone=methods.MILESTONE, first_id=7915
    )["activated"]
    old = ROOT / "tests/fixtures/v685"
    design.write_bytes((old / "design.md").read_bytes())
    active.write_bytes(gzip.decompress((old / "active.yaml.gz").read_bytes()))
    staged.write_bytes(active.read_bytes())
    assert assess_authorities(design, staged, active, snapshots)["activated"]


def test_staging_and_twelve_mutations(authorities, tmp_path):
    """SCENARIO-REPORT-7915-AUTHORITY: missing staging never borrows active bytes."""
    design, staged, active, snapshots = authorities
    staged.unlink()
    result = methods.assess(*authorities)
    assert result["activated"] and not result["authority_snapshots"]["staged"]["exists"]
    staged.write_text("milestone: 2026.10.688\ntasks: []\n")
    assert methods.assess(*authorities)["activated"]
    active.write_text("milestone: 2026.09.686\ntasks: []\n")
    assert not methods.assess(*authorities)["activated"]
    validation.prepare(ROOT, tmp_path)
    rows = methods.mutations(design, active, SOURCE, tmp_path / "mutations")
    assert len(rows) == 12 and all(row["passed"] for row in rows)
    assert {row["unit_id"] for row in rows} >= {"count", "date", "digest", "order", "gate"}


def test_freeze_and_historical_dates(tmp_path):
    """SCENARIO-REPORT-7915-DATES: both historical routes use the frozen fixture date."""
    manifest = validation.manifest(ROOT, tmp_path)
    e2e = [row for row in manifest["commands"] if row["name"].startswith("e2e_016")]
    assert len(e2e) == 3
    assert all(
        row["argv"][row["argv"].index("--date") + 1] == "20260929"
        for row in e2e
        if row["expected_exit"] == 0
    )
    assert e2e[-1]["expected_exit"] == 1
    assert e2e[-1]["failure_reason"] == "run_date_mismatch"
    assert manifest["coverage_includes"] == validation.CHANGED
    expected_include = "--include=" + ",".join(str(ROOT / name) for name in validation.CHANGED)
    assert all(
        expected_include in row["argv"]
        for row in manifest["commands"]
        if row["name"]
        in {"coverage_unit", "coverage_cli_success", "coverage_report", "coverage_json"}
    )
    assert all(name in manifest["dependency_hashes"] for name in validation.CHANGED)
    freeze = methods.method_freeze(ROOT)
    assert len(freeze["future_validation_scopes"]) == 13
    assert freeze["fragility"]["mask_seed"] == 68721
    assert freeze["delayed_aci"]["total_delays"] == [21, 24, 36]


def test_retain_actual_repository_health_observation(tmp_path):
    """SCENARIO-REPORT-7915-VALIDATION: rerun owned checks without repeating repository debt."""
    log = tmp_path / "full-suite.log"
    log.write_text("observed repository failure")
    path = tmp_path / "observation.json"
    row = {
        "name": "repository_full_suite",
        "argv": [str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"],
        "actual_exit": 1,
        "passed": False,
        "log_path": str(log),
        "log_sha256": sha256_file(log),
    }
    prior = tmp_path / "prior.json"
    atomic_json(
        prior,
        {
            "experiment_id": 7915,
            "honest_verdict": "complete_disqualified_required_validation",
            "gate_check_summary": [{"artifact_field": "per_file_statement_coverage"}],
        },
    )
    row.update(prior_candidate_path=str(prior), prior_candidate_sha256=sha256_file(prior))
    atomic_json(path, row)
    value = validation.manifest(ROOT, tmp_path, repository_health_receipt=path)
    assert value["repository_health_prior_observation"] == row
    assert all(item["name"] != "repository_full_suite" for item in value["commands"])
    frozen = tmp_path / "manifest.json"
    atomic_json(frozen, value)
    validation.prepare(ROOT, tmp_path)
    candidate = methods.candidate(
        ROOT,
        methods.assess(
            tmp_path / "design.md",
            tmp_path / "consumed.yaml",
            tmp_path / "active.yaml",
            tmp_path / "snapshots",
        ),
        methods.source_custody(SOURCE, methods.SOURCE_SHA256),
        methods.method_freeze(ROOT),
        [],
        [{"passed": True, "argv": ["true"], "command_argv": ["true"]}],
        frozen,
        1,
        2,
    )
    assert candidate["historical_required_failures"][-1]["experiment_id"] == 7915
    assert candidate["observed_child_commands"] == [["true"]]
    log.write_text("drift")
    with pytest.raises(ValueError, match="repository_health_receipt_drift"):
        validation.manifest(ROOT, tmp_path, repository_health_receipt=path)


def test_candidate_and_cold_reduction(authorities, tmp_path):
    """SCENARIO-REPORT-7915-VALIDATION: reduce thirteen rows and preserve failures."""
    manifest_path = tmp_path / "manifest.json"
    atomic_json(manifest_path, validation.manifest(ROOT, tmp_path))
    value = methods.candidate(
        ROOT,
        methods.assess(*authorities),
        methods.source_custody(SOURCE, methods.SOURCE_SHA256),
        methods.method_freeze(ROOT),
        [],
        [{"passed": True, "argv": ["true"]}],
        manifest_path,
        1,
        2,
    )
    assert value["contract_ready_score"] == 1 and value["sample_size_budget"]["intended"] == 13
    assert value["historical_fixture_date"] == "20260929"
    assert value["execution_date"] == "20260930" and value["lifecycle_task_count"] == 13
    assert value["source_sample_size_budget"]["eligible"] == 604
    assert any(row.get("experiment_id") == 7903 for row in value["historical_required_failures"])
    output, raw = tmp_path / "candidate.json", tmp_path / "rows.json"
    atomic_json(output, value)
    atomic_json(raw, methods.primitive_rows(value))
    assert methods.cold_replay(output, raw)
    changed = deepcopy(value)
    changed["sample_size_budget"]["intended"] = 12
    atomic_json(output, changed)
    assert not methods.cold_replay(output, raw)
    atomic_json(output, value)
    value["execution_date"] = "20260929"
    atomic_json(output, value)
    assert not methods.cold_replay(output, raw)
    failed = methods.candidate(
        ROOT,
        methods.assess(*authorities),
        methods.source_custody(SOURCE, methods.SOURCE_SHA256),
        methods.method_freeze(ROOT),
        [],
        [{"passed": False, "argv": ["false"]}],
        manifest_path,
        1,
        2,
    )
    assert failed["verdict_class"] == "disqualified" and failed["contract_ready_score"] == 0


def test_real_cli_success_failure_and_replay(authorities, tmp_path, monkeypatch):
    """SCENARIO-REPORT-7915-DATES: real CLI enforces producer and fixture dates."""
    design, staged, active, _ = authorities
    output, raw = tmp_path / "cli.json", tmp_path / "cli-rows.json"
    argv = [
        sys.executable,
        str(Path(cli.__file__)),
        "--date",
        "20260930",
        "--fixture-e2e",
        "--design",
        str(design),
        "--staged",
        str(staged),
        "--active",
        str(active),
        "--output",
        str(output),
        "--raw",
        str(raw),
    ]
    done = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr
    replay = [
        sys.executable,
        str(Path(cli.__file__)),
        "--cold-replay",
        str(output),
        "--raw",
        str(raw),
    ]
    assert subprocess.run(replay, cwd=ROOT, capture_output=True, timeout=60).returncode == 0
    terminal = [
        sys.executable,
        str(Path(cli.__file__)),
        "--terminal-recheck",
        str(output),
        "--output",
        str(tmp_path / "checked.json"),
        "--raw",
        str(raw),
    ]
    done = subprocess.run(terminal, cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr
    assert sha256_file(tmp_path / "checked.json") == sha256_file(output)
    atomic_json(output, {"experiment_id": 0})
    assert subprocess.run(replay, cwd=ROOT, capture_output=True, timeout=60).returncode == 1
    wrong = [
        sys.executable,
        str(ROOT / "scripts/experiments/experiment_7868_v683_intervention_protocol.py"),
        "--date",
        "20260930",
        "--fixture-e2e",
        str(tmp_path / "wrong.json"),
    ]
    done = subprocess.run(wrong, cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert done.returncode == 1 and "run_date_mismatch" in done.stdout + done.stderr
    monkeypatch.setattr(sys, "argv", ["exp7915", "--fixture-e2e"])
    with pytest.raises(SystemExit):
        cli.main()
    monkeypatch.setattr(sys, "argv", ["exp7915", "--date", "wrong"])
    with pytest.raises(SystemExit):
        cli.main()
    monkeypatch.setattr(sys, "argv", ["exp7915", "--terminal-recheck", str(output)])
    with pytest.raises(SystemExit):
        cli.main()


@pytest.mark.parametrize("role", ["design", "staged", "active"])
def test_missing_and_malformed_authority_blocks(authorities, role):
    """SCENARIO-REPORT-7915-AUTHORITY: malformed outside evidence cannot qualify."""
    index = {"design": 0, "staged": 1, "active": 2}[role]
    authorities[index].write_text("tasks: [invalid" if role != "design" else "missing")
    value = methods.assess(*authorities)
    assert not value["activated"] and value["gate_check_summary"]
    assert len(value["contract_rows"]) == 13


def test_real_execution_orchestration(authorities, tmp_path, monkeypatch):
    """SCENARIO-REPORT-7915-VALIDATION: wrappers use the callable supervisor."""
    design, staged, active, _ = authorities
    args = SimpleNamespace(
        design=design,
        staged=staged,
        active=active,
        source=SOURCE,
        output=tmp_path / "result.json",
        raw=tmp_path / "raw/rows.json",
        fixture_e2e=False,
    )
    original = validation.manifest

    def manifest(root, private):
        frozen = original(root, private)
        frozen["commands"] = [
            {"name": "fixture_check", "argv": ["true"], "expected_exit": 0, "deadline_s": 1}
        ]
        return frozen

    def child(root, spec, private, durable):
        atomic_json(
            private / "coverage.json",
            {
                "files": {
                    name: {
                        "summary": {"num_statements": 1, "covered_lines": 1},
                        "missing_lines": [],
                    }
                    for name in validation.CHANGED
                }
            },
        )
        (private / ".coverage.combined").write_bytes(b"fixture data")
        return {**spec, "passed": True, "actual_exit": 0}

    monkeypatch.setattr(validation, "manifest", manifest)
    monkeypatch.setattr(validation.shared, "run_check", child)
    monkeypatch.setattr(
        validation.shared, "publish", lambda root, value, output, *a: atomic_json(output, value)
    )
    value = validation.execute(ROOT, args, tmp_path / "private")
    assert value["contract_ready_score"] == 1 and methods.cold_replay(args.output, args.raw)
    monkeypatch.setattr(cli, "execute", lambda *a: value)
    monkeypatch.setattr(sys, "argv", ["exp7915", "--output", str(args.output)])
    assert cli.main() == 0
    assert validation.coverage_complete(tmp_path / "private/coverage.json")
    assert not validation.coverage_complete(tmp_path / "missing")
