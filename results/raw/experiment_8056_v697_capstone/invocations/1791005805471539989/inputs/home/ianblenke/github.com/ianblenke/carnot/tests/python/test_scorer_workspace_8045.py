"""REQ-REPORT-8045: private fixtures qualify a venue without model evidence."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot import experiment_8033_v696_scoring_isolation as old
from carnot import experiment_8045_v697_scorer_workspace as e
from carnot.reporting.current_work_receipt import atomic_json


def cli(tmp_path, *args):
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    prefix = [sys.executable]
    if env.get("CARNOT_8045_COVERAGE_CONFIG"):
        prefix += ["-m", "coverage", "run", "--rcfile=" + env["CARNOT_8045_COVERAGE_CONFIG"]]
    return subprocess.run(
        [*prefix, str(e.ROOT / e.CLI), *map(str, args)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )


def test_workspace_repair(tmp_path):
    """SCENARIO-REPORT-8045-WORKSPACE: old command builder creates pytest's parent."""
    specs = old.commands(tmp_path)
    assert (tmp_path / "pytest").is_dir()
    assert any("--basetemp=" in arg for spec in specs for arg in spec.argv)
    evidence = e.reproduce(tmp_path / "private")
    assert evidence["before"]["actual_exit_code"] == 1
    assert evidence["after"]["actual_exit_code"] == 0
    assert evidence["before"]["missing_parent_detected"]
    assert evidence["after"]["parent_exists_before_launch"]


def test_fresh_context_and_rejection_controls():
    """SCENARIO-REPORT-8045-FIXTURE: copied scores, positions and resets are checked."""
    rows = e.fixture_rows()
    assert e.reduce_rows(rows)["passed"]
    assert len({r["context_identity"] for r in rows}) == len(rows)
    assert all(r["copied_scores_survived"] and r["context_closed"] for r in rows)
    for key in ("copied_scores_survived", "reset_observed", "context_closed"):
        bad = copy.deepcopy(rows)
        bad[0][key] = False
        assert not e.reduce_rows(bad)["passed"]
    bad = copy.deepcopy(rows)
    bad[0]["conditional_logit_positions"][0] += 1
    assert not e.reduce_rows(bad)["passed"]
    bad = copy.deepcopy(rows)
    bad[0]["target_logprobs"][0] += 0.1
    assert not e.reduce_rows(bad)["passed"]
    assert not e.reduce_rows(rows[:1])["passed"]


def test_precondition_operands(tmp_path):
    """SCENARIO-REPORT-8045-WORKSPACE: missing resources name actual operands."""
    assert all(r["passed"] for r in e.preconditions(e.ROOT))
    missing = e.preconditions(tmp_path)
    assert any(not r["passed"] and r["observed"] == "missing_resource" for r in missing)
    for row in missing:
        assert set(
            [
                "upstream_id",
                "path",
                "sha256",
                "artifact_field",
                "check_name",
                "expected",
                "observed",
                "passed",
            ]
        ) <= set(row)


def test_real_cli_fixture_null_blocked_and_tampered(tmp_path):
    """SCENARIO-REPORT-8045-TERMINAL: outside-repo CLI exits normally on all routes."""
    out = tmp_path / "fixture.json"
    valid = cli(tmp_path, "--fixture-output", out)
    assert valid.returncode == 0, valid.stdout + valid.stderr
    assert e.reduce_rows(json.loads(out.read_text())["rows"])["passed"]
    null = cli(tmp_path, "--fixture-output", out, "--shift-position")
    assert null.returncode == 0
    assert not e.reduce_rows(json.loads(out.read_text())["rows"])["passed"]
    primary = tmp_path / "results" / (e.NAME + ".json")
    blocked = cli(tmp_path, "--root", tmp_path / "absent", "--output", primary)
    assert blocked.returncode == 0, blocked.stdout + blocked.stderr
    value = json.loads(primary.read_text())
    assert value["verdict_class"] == "blocked" and value["scorer_fixture_ready_score"] == 0
    assert cli(tmp_path, "--cold-replay", primary).returncode == 0
    value["current_model_invocation_count"] = 1
    atomic_json(primary, value)
    assert cli(tmp_path, "--cold-replay", primary).returncode == 1
    assert cli(tmp_path, "--cold-replay", tmp_path / "missing").returncode == 1
    assert cli(tmp_path, "--date", "bad").returncode == 2


def test_manifest_and_main_seal(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8045-TERMINAL: complete owned checks seal replayable bytes."""
    specs = e.commands(tmp_path)
    assert (tmp_path / "pytest").is_dir()
    assert any("--strict" in s.argv for s in specs)
    assert any(s.scope == "repository_health" for s in specs)
    monkeypatch.setattr(
        e,
        "commands",
        lambda scratch: [e.CommandSpec("private_check", (sys.executable, "-c", "pass"), "owned")],
    )

    def private_validation(scratch, raw):
        log = raw / "private_check.log"
        log.write_text("private test-only validation receipt\n")
        ref = e.reference(log)
        return dict(
            receipts=[
                dict(
                    name="private_check",
                    scope="owned",
                    passed=True,
                    exit_code=0,
                    log_path=ref["path"],
                    log_sha256=ref["sha256"],
                )
            ],
            coverage=dict(percent_covered=100, num_statements=1, covered_lines=1, missing_lines=0),
        )

    monkeypatch.setattr(e, "validate", private_validation)
    output = tmp_path / "results" / (e.NAME + ".json")
    assert e.main(["--output", str(output)]) == 0
    value = json.loads(output.read_text())
    assert value["scorer_fixture_ready_score"] == 1
    assert value["current_model_invocation_count"] == 0
    assert e.main(["--output", str(output)]) == 1
    e.replay(value)
    raw = Path(value["terminal_validation_sidecar_path"]).parent
    fixture = raw / "fixture.json"
    original = fixture.read_bytes()
    fixture.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="hash"):
        e.replay(value)
    fixture.write_bytes(original)
    monkeypatch.setattr(e, "validate", lambda scratch, raw: dict(receipts=[], coverage={}))
    other = tmp_path / "null/results" / (e.NAME + ".json")
    assert e.main(["--output", str(other)]) == 0
    assert json.loads(other.read_text())["verdict_class"] == "null"


def test_validation_counts_only_added_statements(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8045-TERMINAL: old uncovered lines do not inflate task scope."""
    line = next(
        i
        for i, value in enumerate((e.ROOT / old.OWNED[0]).read_text().splitlines(), 1)
        if '(scratch / "pytest").mkdir' in value
    )

    def child(root, specs, **kwargs):
        atomic_json(
            tmp_path / "coverage.json",
            dict(
                files={
                    e.MODULE: dict(summary=dict(num_statements=2, covered_lines=2)),
                    e.CLI: dict(summary=dict(num_statements=1, covered_lines=1)),
                    old.OWNED[0]: dict(executed_lines=[line], missing_lines=[1]),
                }
            ),
        )
        return []

    monkeypatch.setattr(e, "run_commands", child)
    raw = tmp_path / "raw"
    measured = e.validate(tmp_path, raw)
    assert measured["coverage"]["percent_covered"] == 100
    assert measured["coverage"]["num_statements"] == 4
    (tmp_path / "coverage.json").unlink()
    monkeypatch.setattr(e, "run_commands", lambda *a, **k: [])
    assert e.validate(tmp_path, raw)["coverage"]["percent_covered"] == 0


def test_contract_errors_and_primitive_drift(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8045-TERMINAL: missing fields and rehashed drift are rejected."""
    atomic_json(tmp_path / e.HISTORY, {})
    checks = e.preconditions(tmp_path)
    assert any(r["observed"] == "missing_field_contract_error" for r in checks)
    output = tmp_path / "results" / (e.NAME + ".json")
    assert e.main(["--root", str(tmp_path / "absent"), "--output", str(output)]) == 0
    value = json.loads(output.read_text())
    raw = Path(value["terminal_validation_sidecar_path"]).parent
    atomic_json(raw / "fixture.json", dict(rows=[dict(stale=True)]))
    value["raw_shard_hashes"] = [e.reference(Path(r["path"])) for r in value["raw_shard_hashes"]]
    value["checkpoint_references"] = [e.reference(raw / "fixture.json")]
    with pytest.raises(ValueError, match="fixture_row_drift"):
        e.replay(value)
    real_run = e.run_commands

    def bad_cold(root, specs, **kwargs):
        if specs[0].name == "cold_reduction":
            return [dict(passed=False)]
        return real_run(root, specs, **kwargs)

    monkeypatch.setattr(e, "run_commands", bad_cold)
    failed = tmp_path / "cold/results" / (e.NAME + ".json")
    assert e.main(["--root", str(tmp_path / "absent"), "--output", str(failed)]) == 1
    monkeypatch.setattr(e, "run_commands", real_run)
    monkeypatch.setattr(
        e, "terminal", lambda path: dict(passed=path.name == "terminal_candidate.json")
    )
    failed = tmp_path / "published/results" / (e.NAME + ".json")
    assert e.main(["--root", str(tmp_path / "absent"), "--output", str(failed)]) == 1
