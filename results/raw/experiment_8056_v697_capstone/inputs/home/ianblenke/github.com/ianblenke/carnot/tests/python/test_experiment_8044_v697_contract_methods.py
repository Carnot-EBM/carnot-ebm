"""REQ-REPORT-8044: private authority and authentic terminal history qualification."""

from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from carnot.reporting import v697_contract_methods as methods
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from scripts.experiments import experiment_8044_v697_contract_methods as cli
from test_experiment_8031_v696_contract_methods import private_reader_corpus  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]


def inputs(private):
    """Frozen authority keeps tests independent of later roadmap activation."""
    private.mkdir(parents=True, exist_ok=True)
    for name in ("design.md", "active.yaml"):
        (private / name).write_bytes(
            gzip.decompress((ROOT / "tests/fixtures/v697" / (name + ".gz")).read_bytes())
        )
    return SimpleNamespace(
        design=private / "design.md",
        active=private / "active.yaml",
        staged=private / "absent.yaml",
        fixture_e2e=True,
        output=private / "experiment_8044_fixture.json",
        raw=private / "raw/rows.json",
        date="20261003",
    )


def test_authority(tmp_path):
    """SCENARIO-REPORT-8044-AUTHORITY: every prompt and all twelve controls bind."""
    args = inputs(tmp_path)
    value = methods.assess(args.design, args.staged, args.active, tmp_path / "snapshots")
    assert value["activated"] and len(value["contract_rows"]) == 13
    assert value["staging_custody_status"] == "unknown_consumed"
    assert all(
        r["passed"]
        for r in methods.mutations(args.design, args.active, args.staged, tmp_path / "mutations")
    )
    text = args.design.read_text()
    args.design.write_text(text.replace('"prompt":"', '"prompt":"CHANGED ', 1))
    assert not methods.assess(args.design, args.staged, args.active, tmp_path / "drift")[
        "activated"
    ]
    args.design.write_text(text)
    args.staged.write_text("milestone: 2026.10.696\ntasks: []\n")
    assert not methods.assess(args.design, args.staged, args.active, tmp_path / "stage")[
        "activated"
    ]
    args.design.unlink()
    assert not methods.assess(args.design, args.staged, args.active, tmp_path / "missing")[
        "activated"
    ]


def test_history_and_methods(tmp_path):
    """SCENARIO-REPORT-8044-HISTORY: final bytes and historical contradictions coexist."""
    args = inputs(tmp_path)
    tasks = yaml.safe_load(args.active.read_bytes())["tasks"]
    evidence = methods.history(ROOT, tasks, tmp_path / "history")
    assert evidence["ready"], evidence["gate_check_summary"]
    rows = evidence["historical_disposition_rows"]
    for n in (8032, 8038):
        selected = [r for r in rows if r["experiment_id"] == n]
        assert any(
            r["evidence_kind"] == "final_primary" and r["verdict_class"] == "null" for r in selected
        )
        assert any(
            r["evidence_kind"] == "intermediate_log" and "disqualified" in r["honest_verdict"]
            for r in selected
        )
    assert next(r for r in rows if r["experiment_id"] == 8034)["original_path"].endswith(
        "experiment_8034_fit_likelihood_capture.json"
    )
    assert all(
        r["honest_verdict"] is None for r in rows if r["experiment_id"] in (8035, 8036, 8037)
    )
    assert all(Path(r["path"]).is_relative_to(tmp_path) for r in evidence["hashes"])
    missing = methods.history(tmp_path / "absent", tasks, tmp_path / "missing")
    assert not missing["ready"] and missing["gate_check_summary"]
    freeze = methods.method_freeze(args.design, tasks)
    assert len(freeze["task_contracts"]) == 13 and not freeze["science_pre_gate"]
    assert {"GASP", "LILAC+", "FedProTIP"} <= {r["method"] for r in freeze["method_source_map"]}
    assert all(g["upstream"] != tasks[0]["id"] for t in tasks for g in t["gated_on"])


def test_execution_and_tamper(tmp_path):
    """SCENARIO-REPORT-8044-TERMINAL: independent primitive replay rejects edits."""
    args = inputs(tmp_path)
    args.owned_checks_only = True
    value = methods.execute(args, tmp_path / "work")
    frozen = json.loads(Path(value["validation_command_manifest_path"]).read_text())
    assert all(s["classification"] != "diagnostic" for s in frozen["commands"])
    assert value["contract_ready_score"] == 1 and value["verdict_class"] == "circular_positive"
    assert value["run_date"] == "20261003" and value["independent_count"] == 0
    assert value["MODEL_SPECS"] == value["trained_head_specs"] == []
    assert methods.cold_replay(args.output, args.raw)
    original = args.output.read_bytes()
    for field in ("contract_ready_score", "historical_disposition_rows", "canonical_tasks_sha256"):
        changed = deepcopy(value)
        changed[field] = None
        atomic_json(args.output, changed)
        assert not methods.cold_replay(args.output, args.raw)
    args.output.write_bytes(original)
    Path(value["checkpoint_references"][0]["path"]).write_bytes(b"changed")
    assert not methods.cold_replay(args.output, args.raw)
    assert not methods.cold_replay(tmp_path / "missing", args.raw)
    args.output.write_text("invalid json")
    assert not methods.cold_replay(args.output, args.raw)


def test_blocked_and_failed_checks(tmp_path, monkeypatch):
    """REQ-REPORT-8044: missing external resources block; owned failures disqualify."""
    args = inputs(tmp_path)
    args.design.unlink()
    value = methods.execute(args, tmp_path / "work")
    assert value["verdict_class"] == "blocked" and value["gate_check_summary"]
    args = inputs(tmp_path / "failure")
    args.fixture_e2e = False
    monkeypatch.setattr(
        methods,
        "manifest",
        lambda *a: dict(
            commands=[dict(name="failure", argv=["private"], expected_exit=0, deadline_s=1)],
            dependency_hashes={},
        ),
    )
    monkeypatch.setattr(
        methods.validation,
        "run_check",
        lambda *a, **k: dict(name="failure", argv=["private"], passed=False, actual_exit=1),
    )
    monkeypatch.setattr(methods, "publish", lambda *a: None)
    value = methods.execute(args, tmp_path / "failure-work")
    assert value["verdict_class"] == "disqualified" and value["contract_ready_score"] == 0


def test_private_cli_and_publication(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8044-TERMINAL: real CLI exits and both consumers qualify bytes."""
    args = inputs(tmp_path)
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    base = [str(ROOT / ".venv/bin/python"), "-u", str(ROOT / methods.CLI)]
    fixture = [
        "--date",
        "20261003",
        "--fixture-e2e",
        "--design",
        str(args.design),
        "--active",
        str(args.active),
        "--staged",
        str(args.staged),
        "--output",
        str(args.output),
        "--raw",
        str(args.raw),
    ]
    run = subprocess.run(
        base + fixture, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
    )
    assert run.returncode == 0, run.stdout + run.stderr
    replay = base + ["--cold-replay", str(args.output), "--raw", str(args.raw)]
    assert (
        subprocess.run(replay, cwd=tmp_path, env=env, capture_output=True, timeout=60).returncode
        == 0
    )
    assert cli.main(["--cold-replay", str(args.output), "--raw", str(args.raw)]) == 0
    value = json.loads(args.output.read_text())
    methods.publish(value, args.output, args.raw, tmp_path / "terminal", args.raw.parent)
    report = json.loads(Path(value["terminal_validation_sidecar_path"]).read_text())
    assert report["passed"] and report["primary_sha256"] == sha256_file(args.output)
    assert all(r["passed"] for r in report["reports"])
    value["contract_ready_score"] = 0
    atomic_json(args.output, value)
    assert (
        subprocess.run(replay, cwd=tmp_path, env=env, capture_output=True, timeout=60).returncode
        == 1
    )
    with pytest.raises(SystemExit):
        cli.main(["--fixture-e2e"])
    with pytest.raises(SystemExit):
        cli.main(["--date", "20261004"])
    assert cli.main(["--cold-replay", str(tmp_path / "missing"), "--raw", str(args.raw)]) == 1
    monkeypatch.setattr(methods, "execute", lambda *a: {})
    assert cli.main(fixture) == 0
    assert cli.main(["--owned-checks-only", *fixture]) == 0


def test_owned_measurement_and_terminal_faults(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8044-TERMINAL: actual process logs and coverage enter custody."""
    args = inputs(tmp_path)
    args.fixture_e2e = False
    private = tmp_path / "work"
    private.mkdir()
    counts = {
        p: dict(summary=dict(num_statements=1, covered_lines=1), missing_lines=[])
        for p in methods.OWNED
    }
    atomic_json(private / "coverage.json", dict(files=counts))
    real = methods.manifest
    monkeypatch.setattr(
        methods,
        "manifest",
        lambda *a: dict(
            real(*a),
            commands=[
                dict(
                    name="private_fixture",
                    argv=[str(ROOT / ".venv/bin/python"), "-c", "print('owned check', flush=True)"],
                    expected_exit=0,
                    deadline_s=60,
                )
            ],
        ),
    )
    value = methods.execute(args, private)
    assert value["contract_ready_score"] == 1 and value["coverage_statement_counts"]
    assert methods.cold_replay(args.output, args.raw)
    monkeypatch.setattr(
        methods.validation, "run_check", lambda root, spec, *a, **k: dict(spec, passed=False)
    )
    with pytest.raises(ValueError, match="terminal_validation_failed"):
        methods.publish(value, args.output, args.raw, tmp_path / "fail", args.raw.parent)
    monkeypatch.setattr(
        methods.validation, "run_check", lambda root, spec, *a, **k: dict(spec, passed=True)
    )
    monkeypatch.setattr(
        methods,
        "reader_receipt",
        lambda *a, **k: dict(gate_sha256="wrong", document_sha256="wrong"),
    )
    with pytest.raises(ValueError, match="primary_reader_drift"):
        methods.publish(value, args.output, args.raw, tmp_path / "drift", args.raw.parent)


def test_history_tamper_and_absence_paths(tmp_path):
    """SCENARIO-REPORT-8044-HISTORY: missing or wrong operands cannot qualify custody."""
    args = inputs(tmp_path)
    tasks = yaml.safe_load(args.active.read_bytes())["tasks"]
    root = tmp_path / "corpus"
    (root / "results").mkdir(parents=True)
    atomic_json(
        root / "results/experiment_8032_wrong.json",
        dict(
            honest_verdict="complete_null_fixture",
            terminal_validation_sidecar_path=str(root / "missing"),
        ),
    )
    record = methods.history(root, tasks, tmp_path / "evidence")
    assert not record["ready"]
    assert any(
        g["artifact_field"] == "terminal_primary_sha256" for g in record["gate_check_summary"]
    )
    assert all(
        g["observed"] is None
        for g in record["gate_check_summary"]
        if g["artifact_field"] == "honest_verdict"
        and g["upstream_id"] == "exp8039-learning-benefit-audit"
    )
    args.active.unlink()
    failed = methods.assess(args.design, args.staged, args.active, tmp_path / "inactive")
    assert not failed["activated"]


def test_independent_history_reduction_and_missing_authority_replay(tmp_path):
    """SCENARIO-REPORT-8044-HISTORY: reduction checks original bytes beyond claim seals."""
    args = inputs(tmp_path)
    value = methods.execute(args, tmp_path / "work")
    assert methods.evidence.replay_history(
        value["historical_disposition_rows"], value["checkpoint_references"]
    )
    changed = deepcopy(value["historical_disposition_rows"])
    next(r for r in changed if r["evidence_kind"] == "final_primary")["honest_verdict"] = (
        "complete_null_invented"
    )
    assert not methods.evidence.replay_history(changed, value["checkpoint_references"])
    changed = deepcopy(value["historical_disposition_rows"])
    next(r for r in changed if r["evidence_kind"] == "intermediate_log")["original_line"] = (
        "invented log"
    )
    assert not methods.evidence.replay_history(changed, value["checkpoint_references"])
    changed = deepcopy(value["historical_disposition_rows"])
    next(r for r in changed if r["evidence_kind"] == "absent_evidence")["honest_verdict"] = (
        "complete_null_invented"
    )
    assert not methods.evidence.replay_history(changed, value["checkpoint_references"])
    args = inputs(tmp_path / "blocked")
    args.design.unlink()
    blocked = methods.execute(args, tmp_path / "blocked-work")
    assert blocked["verdict_class"] == "blocked" and methods.cold_replay(args.output, args.raw)


def test_external_custody_block_and_current_code_drift(tmp_path, monkeypatch):
    """REQ-REPORT-8044: external absence names a resource and code changes invalidate replay."""
    args = inputs(tmp_path)
    value = methods.execute(args, tmp_path / "work")
    original = tmp_path / "producer.py"
    original.write_text("original")
    saved = tmp_path / "producer.bin"
    saved.write_bytes(original.read_bytes())
    value["checkpoint_references"].append(
        dict(
            path=str(saved),
            original_path=str(original),
            sha256=sha256_file(saved),
            role="current_code",
        )
    )
    atomic_json(args.output, value)
    atomic_json(args.raw, methods.primitive_rows(value))
    assert methods.cold_replay(args.output, args.raw)
    original.write_text("changed")
    assert not methods.cold_replay(args.output, args.raw)
    real = methods.history
    missing = tmp_path / "missing_required_resource.json"
    monkeypatch.setattr(
        methods,
        "history",
        lambda *a: dict(
            real(*a),
            ready=False,
            gate_check_summary=[
                methods.evidence.operand(
                    missing, "required_resource", "resource_exists", True, False
                )
            ],
        ),
    )
    args = inputs(tmp_path / "external")
    blocked = methods.execute(args, tmp_path / "external-work")
    assert (
        blocked["verdict_class"] == "blocked"
        and "missing_required_resource" in blocked["honest_verdict"]
    )


def test_frozen_mypy_invocation(tmp_path):
    """SCENARIO-REPORT-8044-TERMINAL: the frozen strict command must parse and exit normally."""
    spec = next(
        s for s in methods.manifest(ROOT, tmp_path)["commands"] if s["name"] == "mypy_strict"
    )
    run = subprocess.run(spec["argv"], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
