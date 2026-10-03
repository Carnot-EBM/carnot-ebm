"""REQ-REPORT-7979-V692 and REQ-VERIFY-7979-V692: audit without science claims."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest
import yaml

from carnot.reporting import v692_contract_methods as methods
from carnot.reporting import v692_contract_validation as validation
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from scripts.experiments import experiment_7979_v692_contract_methods as cli

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/experiment_7892_v685_source_boundary.json"


def inputs(private):
    """Separate authority copies keep tests from changing conductor inputs."""
    validation.prepare(ROOT, private)
    return tuple(private / n for n in ("design.md", "stage.yaml", "active.yaml", "snapshots"))


def build(private):
    """Use authenticated historical evidence with private current outputs."""
    started = time.monotonic_ns()
    paths = inputs(private)
    manifest = private / "manifest.json"
    atomic_json(manifest, validation.manifest(ROOT, private))
    tasks = yaml.safe_load(paths[2].read_bytes())["tasks"]
    args = (
        ROOT,
        methods.assess(*paths),
        methods.source_custody(SOURCE, methods.SOURCE_SHA256),
        methods.method_freeze(ROOT, tasks=tasks),
        [],
        [{"passed": True, "argv": ["true"]}],
        manifest,
        started,
        time.monotonic_ns(),
    )
    return methods.candidate(*args), args


def test_authority_mutations_and_methods(tmp_path):
    """SCENARIO-REPORT-7979-AUTHORITY: accept activation and reject private drift."""
    paths = inputs(tmp_path)
    result = methods.assess(*paths)
    assert result["activated"] and len(result["contract_rows"]) == 13
    assert result["staging_custody_status"] == "unknown_consumed"
    controls = methods.mutations(paths[0], paths[2], SOURCE, tmp_path / "mutations")
    assert len(controls) == 16 and all(r["passed"] for r in controls)
    assert {"duplicate_id", "source_hash", "producer_rollover", "forged_date"} <= {
        r["unit_id"] for r in controls
    }
    tasks = yaml.safe_load(paths[2].read_bytes())["tasks"]
    freeze = methods.method_freeze(ROOT, tasks=tasks)
    assert sum(freeze["role_allocation"].values()) == 640
    assert freeze["reserved_evaluation_sources"] == 96
    assert freeze["features"] == {"source": 8, "qwen_probability": 1}
    assert freeze["decisions"]["primary_comparisons"] == 6
    assert freeze["learning"]["minimum_inferential_blocks"] == 8
    assert {r["name"] for r in freeze["literature_map"]} >= {
        "EBT",
        "ARM-EBM",
        "evidence alignment",
        "delayed ACI",
        "sparse spline learning",
    }
    assert all(
        r["mechanism"] and r["adaptation"] and r["deferred_scope"] for r in freeze["literature_map"]
    )
    assert freeze["task_contracts"] == tasks and not freeze["science_pre_gate"]


def test_history_and_cold_replay(tmp_path):
    """SCENARIO-VERIFY-7979-CUSTODY: retain failures and exact producer identities."""
    value, args = build(tmp_path)
    assert value["contract_ready_score"] == 1
    assert value["verdict_class"] == "circular_positive"
    assert value["MODEL_SPECS"] == [] and value["model_invocation_counts"]["calls"] == 0
    history = value["historical_dispositions"]
    assert len(history) == 13 and history[-1]["verdict_class"] == "disqualified"
    assert history[-1]["gate_check_summary"]
    assert sum(r["disposition"] == "dispatch_blocked" for r in history) == 4
    assert all(r["planned_producer_path"] != r["dispatch_receipt"].get("path") for r in history)
    output, raw = tmp_path / "candidate.json", tmp_path / "rows.json"
    atomic_json(output, value)
    atomic_json(raw, methods.primitive_rows(value))
    assert methods.cold_replay(output, raw)
    for key, changed in (
        ("experiment_id", 0),
        ("run_date", "20260930"),
        ("rows", []),
        ("contract_ready_score", 0),
        ("method_registry", []),
    ):
        atomic_json(output, {**value, key: changed})
        assert not methods.cold_replay(output, raw)
    assert not methods.cold_replay(tmp_path / "absent", raw)
    atomic_json(output, value)
    saved = Path(value["authority_snapshots"]["active"]["snapshot_path"])
    original = saved.read_bytes()
    saved.write_bytes(b"changed")
    assert not methods.cold_replay(output, raw)
    saved.write_bytes(original)
    failed = methods.candidate(*(*args[:5], [{"passed": False, "argv": ["false"]}], *args[6:]))
    assert failed["verdict_class"] == "disqualified" and failed["contract_ready_score"] == 0
    custody = deepcopy(args[2])
    custody["ready"] = False
    blocked = methods.candidate(*(*args[:2], custody, *args[3:]))
    assert blocked["verdict_class"] == "blocked" and blocked["contract_ready_score"] == 0


def test_missing_historical_input(tmp_path):
    """SCENARIO-REPORT-7979-AUTHORITY: missing required history names exact operands."""
    history = methods.historical_custody(tmp_path)
    assert not history["ready"] and history["gate_check_summary"]
    assert all(
        {"path", "hash", "artifact_field", "op", "expected", "observed"} <= set(r)
        for r in history["gate_check_summary"]
    )


def test_changed_historical_hashes_and_blocked_candidate(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-7979-CUSTODY: history cannot authenticate changed source bytes."""
    original = methods.sha256_file
    monkeypatch.setattr(
        methods, "sha256_file", lambda p: "sha256:changed" if p.suffix == ".bin" else original(p)
    )
    assert not methods.historical_custody(ROOT)["ready"]
    monkeypatch.setattr(
        methods,
        "sha256_file",
        lambda p: "sha256:changed" if p.name == "experiment_7970_energy_fit.json" else original(p),
    )
    assert not methods.historical_custody(ROOT)["ready"]
    monkeypatch.setattr(methods, "sha256_file", original)
    value, args = build(tmp_path)
    monkeypatch.setattr(
        methods,
        "historical_custody",
        lambda root: {
            "ready": False,
            "rows": [],
            "hashes": [],
            "gate_check_summary": [{"artifact_field": "sha256", "observed": None}],
        },
    )
    blocked = methods.candidate(*args)
    assert blocked["verdict_class"] == "blocked" and not blocked["contract_ready_score"]
    assert value["verdict_class"] == "circular_positive"


def test_negative_wrapper_rejects_success(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7979-VALIDATION: only child exit one is an expected rejection."""
    monkeypatch.setattr(
        "sys.argv",
        ["experiment_7979", "--expect-rejection", "--cold-replay", str(tmp_path / "absent")],
    )
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0))
    assert cli.main() == 1


def test_manifest_and_cli_guards(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7979-VALIDATION: all owned routes and includes are frozen."""
    validation.prepare(ROOT, tmp_path)
    frozen = validation.manifest(ROOT, tmp_path)
    assert frozen["coverage_includes"] == validation.OWNED
    commands = {r["name"]: r for r in frozen["commands"]}
    assert {
        "e2e_018_private",
        "repository_full_suite",
        "coverage_expected_failure",
    } <= commands.keys()
    assert commands["coverage_expected_failure"]["expected_exit"] == 0
    assert "--expect-rejection" in commands["coverage_expected_failure"]["argv"]
    if frozen["prior_owned_attempts"]:
        assert commands["repository_full_suite"]["reused_receipt"]["classification"] == "diagnostic"
        assert all(r["sha256"] for r in frozen["prior_owned_attempts"])
    assert not validation.coverage_complete(tmp_path / "absent.json")
    for argv in (["--fixture-e2e"], ["--date", "20260930"]):
        monkeypatch.setattr("sys.argv", ["experiment_7979", *argv])
        with pytest.raises(SystemExit):
            cli.main()
    monkeypatch.setattr("sys.argv", ["experiment_7979", "--cold-replay", str(tmp_path / "absent")])
    assert cli.main() == 1


def test_private_execution_and_publication(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7979-VALIDATION: owned failures still publish honest bytes."""
    paths = inputs(tmp_path / "inputs")
    original_manifest = validation.manifest

    def manifest(root, private):
        value = original_manifest(root, private)
        value["commands"] = [{"name": "injected_failure", "argv": ["false"]}]
        return value

    monkeypatch.setattr(validation, "manifest", manifest)
    monkeypatch.setattr(
        validation,
        "run_check",
        lambda *a: {
            "name": "injected_failure",
            "passed": False,
            "argv": ["false"],
            "actual_exit": 1,
        },
    )
    private = tmp_path / "work"
    private.mkdir()
    atomic_json(private / "coverage.json", {"files": {}})
    private.joinpath(".coverage.combined").write_text("private test data")
    args = SimpleNamespace(
        design=paths[0],
        staged=paths[1],
        active=paths[2],
        source=SOURCE,
        fixture_e2e=False,
        output=tmp_path / "published/experiment_7979_fixture.json",
        raw=tmp_path / "published/raw/experiment_7979_fixture/rows.json",
    )
    value = validation.execute(ROOT, args, private)
    assert value["verdict_class"] == "disqualified" and not value["contract_ready_score"]
    assert methods.cold_replay(args.output, args.raw)
    selected = json.loads((args.raw.parent / "primary_resolution_receipt.json").read_text())
    assert selected["gate_sha256"] == selected["document_sha256"] == sha256_file(args.output)
    monkeypatch.setattr(
        validation,
        "reader_receipt",
        lambda *a, **kw: {"gate_sha256": "wrong", "document_sha256": "wrong"},
    )
    with pytest.raises(ValueError, match="primary_reader_drift"):
        validation.publish(ROOT, value, args.output, args.raw, tmp_path / "again")


def test_real_private_cli_routes(tmp_path):
    """SCENARIO-REPORT-7979-VALIDATION: the real CLI works outside the checkout."""
    import os
    import subprocess

    paths = inputs(tmp_path / "inputs")
    output = tmp_path / "experiment_7979_fixture.json"
    raw = tmp_path / "raw/rows.json"
    script = ROOT / validation.OWNED[-1]
    base = [str(ROOT / ".venv/bin/python"), str(script)]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    success = subprocess.run(
        base
        + [
            "--fixture-e2e",
            "--design",
            str(paths[0]),
            "--active",
            str(paths[2]),
            "--staged",
            str(paths[1]),
            "--output",
            str(output),
            "--raw",
            str(raw),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert success.returncode == 0, success.stdout + success.stderr
    replay = subprocess.run(
        base + ["--cold-replay", str(output), "--raw", str(raw)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert replay.returncode == 0, replay.stdout + replay.stderr
    data = json.loads(output.read_text())
    data["contract_ready_score"] = 0
    atomic_json(output, data)
    rejection = subprocess.run(
        base + ["--expect-rejection", "--cold-replay", str(output), "--raw", str(raw)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert rejection.returncode == 0 and "inner_exit=1" in rejection.stdout
