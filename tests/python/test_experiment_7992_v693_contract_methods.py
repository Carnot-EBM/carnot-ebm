"""REQ-REPORT-7992-V693: qualify owned routes without importing science claims."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest
import yaml

from carnot.reporting import v693_contract_methods as methods
from carnot.reporting import v693_contract_validation as validation
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from scripts.experiments import experiment_7992_v693_contract_methods as cli

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/experiment_7892_v685_source_boundary.json"


def build(private):
    """Private authority and evidence copies prevent tests rewriting historical artifacts."""
    validation.prepare(ROOT, private)
    paths = tuple(private / n for n in ("design.md", "stage.yaml", "active.yaml", "authority"))
    manifest = private / "manifest.json"
    atomic_json(manifest, validation.manifest(ROOT, private))
    args = (
        ROOT,
        methods.assess(*paths),
        methods.source_custody(SOURCE, methods.SOURCE_SHA256),
        methods.method_freeze(yaml.safe_load(paths[2].read_bytes())["tasks"]),
        [],
        [{"passed": True, "argv": ["true"]}],
        manifest,
        time.monotonic_ns(),
        time.monotonic_ns(),
    )
    return methods.candidate(*args), args


def test_authority_and_twelve_mutations(tmp_path):
    """SCENARIO-REPORT-7992-AUTHORITY: each independent executable operand rejects drift."""
    value, args = build(tmp_path)
    assert value["contract_ready_score"] == 1
    assert value["staging_custody_status"] == "unknown_consumed"
    controls = methods.mutations(
        tmp_path / "design.md", tmp_path / "active.yaml", tmp_path / "mutations"
    )
    assert len(controls) == 12 and all(r["passed"] for r in controls)
    assert {r["unit_id"] for r in controls} == set(methods.MUTATIONS)
    assert len(value["method_freeze"]["registry"]) == 13
    assert value["method_freeze"]["task_contracts"] == args[3]["task_contracts"]
    assert not value["method_freeze"]["science_pre_gate"]


def test_custody_and_verdicts(tmp_path):
    """SCENARIO-VERIFY-7992-CUSTODY: preserve producer failures without rechecking mutable code."""
    value, args = build(tmp_path)
    assert value["verdict_class"] == "circular_positive"
    assert value["MODEL_SPECS"] == value["trained_head_specs"] == []
    assert value["model_invocation_counts"]["calls"] == 0
    assert len(value["historical_failure_rows"]) == 13
    assert (
        sum(r["disposition"] == "dispatch_blocked" for r in value["historical_failure_rows"]) == 4
    )
    assert value["historical_failure_rows"][0]["verdict_class"] == "disqualified"
    assert value["historical_failure_rows"][-1]["verdict_class"] == "disqualified"
    assert value["producer_snapshot_rows"]
    output, raw = tmp_path / "candidate.json", tmp_path / "rows.json"
    atomic_json(output, value)
    atomic_json(raw, methods.primitive_rows(value))
    assert methods.cold_replay(output, raw)
    for key, changed in (("experiment_id", 0), ("contract_ready_score", 0), ("rows", [])):
        atomic_json(output, {**value, key: changed})
        assert not methods.cold_replay(output, raw)
    assert not methods.cold_replay(tmp_path / "absent", raw)
    atomic_json(output, value)
    frozen = Path(value["producer_snapshot_rows"][0]["snapshot_path"])
    original_bytes = frozen.read_bytes()
    frozen.write_bytes(b"changed")
    assert not methods.cold_replay(output, raw)
    frozen.write_bytes(original_bytes)
    failed = methods.candidate(*(*args[:5], [{"passed": False, "argv": ["false"]}], *args[6:]))
    assert failed["verdict_class"] == "disqualified" and failed["contract_ready_score"] == 0
    custody = deepcopy(args[2])
    custody["ready"] = False
    blocked = methods.candidate(*(*args[:2], custody, *args[3:]))
    assert blocked["verdict_class"] == "blocked" and blocked["contract_ready_score"] == 0
    diagnostic = methods.candidate(
        *(
            *args[:5],
            [*args[5], {"passed": False, "argv": ["false"], "classification": "diagnostic"}],
            *args[6:],
        )
    )
    assert diagnostic["contract_ready_score"] == 1


def test_missing_history_and_wrong_id(tmp_path):
    """SCENARIO-REPORT-7992-VALIDATION: reproduce the uncovered Exp7979 wrong-ID route."""
    from carnot.reporting import v692_contract_methods

    bad, raw = tmp_path / "bad.json", tmp_path / "rows.json"
    atomic_json(bad, {"experiment_id": 0})
    atomic_json(raw, {"rows": []})
    assert not v692_contract_methods.cold_replay(bad, raw)
    history = methods.historical_custody(tmp_path, tmp_path / "history")
    assert not history["ready"] and history["gate_check_summary"]
    assert all(
        {"path", "hash", "artifact_field", "op", "expected", "observed"} <= set(r)
        for r in history["gate_check_summary"]
    )


def test_manifest_and_wrapper_guards(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7992-VALIDATION: expected rejection must assert the actual child exit."""
    validation.prepare(ROOT, tmp_path)
    frozen = validation.manifest(ROOT, tmp_path)
    assert frozen["coverage_includes"] == validation.OWNED
    assert all("reused_receipt" not in r for r in frozen["commands"])
    assert not validation.coverage_complete(tmp_path / "absent")
    for argv in (["--fixture-e2e"], ["--date", "20260930"]):
        monkeypatch.setattr("sys.argv", ["experiment_7992", *argv])
        with pytest.raises(SystemExit):
            cli.main()
    monkeypatch.setattr(
        "sys.argv",
        ["experiment_7992", "--expect-rejection", "--cold-replay", str(tmp_path / "absent")],
    )
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0))
    assert cli.main() == 1
    monkeypatch.setattr("sys.argv", ["experiment_7992", "--cold-replay", str(tmp_path / "absent")])
    assert cli.main() == 1


def test_private_execution_and_publication_failure(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7992-VALIDATION: publication failure cannot qualify unchecked bytes."""
    validation.prepare(ROOT, tmp_path / "inputs")
    original = validation.manifest

    def manifest(root, private):
        value = original(root, private)
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
    args = SimpleNamespace(
        design=tmp_path / "inputs/design.md",
        staged=tmp_path / "inputs/stage.yaml",
        active=tmp_path / "inputs/active.yaml",
        source=SOURCE,
        fixture_e2e=False,
        output=tmp_path / "published/experiment_7992_fixture.json",
        raw=tmp_path / "published/raw/experiment_7992_fixture/rows.json",
    )
    private = tmp_path / "work"
    private.mkdir()
    atomic_json(private / "coverage.json", {"files": {}})
    private.joinpath(".coverage.combined").write_text("private test data")
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
    monkeypatch.setattr(
        validation,
        "publish_primary",
        lambda *a, **kw: (_ for _ in ()).throw(ValueError("candidate_rejected")),
    )
    with pytest.raises(ValueError, match="candidate_rejected"):
        validation.publish(ROOT, value, args.output, args.raw, tmp_path / "rejected")


def test_real_private_cli_routes(tmp_path):
    """SCENARIO-REPORT-7992-VALIDATION: real success, blocked, invalid and negative replay stay private."""
    import os
    import subprocess

    validation.prepare(ROOT, tmp_path / "inputs")
    output, raw = tmp_path / "experiment_7992_fixture.json", tmp_path / "raw/rows.json"
    base = [str(ROOT / ".venv/bin/python"), str(ROOT / validation.OWNED[-1])]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)

    def run(argv):
        print("[test7992] subprocess_before", flush=True)
        result = subprocess.run(
            base + argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60
        )
        print(f"[test7992] subprocess_after exit={result.returncode}", flush=True)
        return result

    success = [
        "--fixture-e2e",
        "--design",
        str(tmp_path / "inputs/design.md"),
        "--active",
        str(tmp_path / "inputs/active.yaml"),
        "--staged",
        str(tmp_path / "inputs/stage.yaml"),
        "--output",
        str(output),
        "--raw",
        str(raw),
    ]
    result = run(success)
    assert result.returncode == 0, result.stdout + result.stderr
    assert run(["--cold-replay", str(output), "--raw", str(raw)]).returncode == 0
    atomic_json(output, {**json.loads(output.read_text()), "experiment_id": 0})
    rejection = run(["--expect-rejection", "--cold-replay", str(output), "--raw", str(raw)])
    assert rejection.returncode == 0 and "inner_exit=1" in rejection.stdout
    assert run(success + ["--source", str(tmp_path / "absent")]).returncode == 0
    blocked = json.loads(output.read_text())
    assert blocked["verdict_class"] == "blocked" and blocked["contract_ready_score"] == 0
    assert run(["--cold-replay", str(output), "--raw", str(raw)]).returncode == 0
    assert run(["--date", "bad"]).returncode == 2


def test_incomplete_live_authority_and_historical_mismatch(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7992-AUTHORITY: external absence blocks without inventing contract text."""
    value, args = build(tmp_path)
    incomplete = tmp_path / "incomplete.md"
    incomplete.write_text("# V693 external design lacks a machine contract\n")
    assessment = methods.assess(
        incomplete, tmp_path / "absent", tmp_path / "active.yaml", tmp_path / "missing-authority"
    )
    assert not assessment["activated"] and assessment["gate_check_summary"]
    blocked = methods.candidate(*((args[0], assessment) + args[2:]))
    assert blocked["verdict_class"] == "blocked" and blocked["contract_ready_score"] == 0
    output, raw = tmp_path / "blocked.json", tmp_path / "blocked-rows.json"
    atomic_json(output, blocked)
    atomic_json(raw, methods.primitive_rows(blocked))
    assert methods.cold_replay(output, raw)
    historical_root = tmp_path / "historical"
    capstone = historical_root / "results/experiment_7991_v692_capstone.json"
    old = json.loads((ROOT / "results/experiment_7991_v692_capstone.json").read_text())
    old["authority_snapshots"]["active"]["sha256"] = "sha256:changed"
    atomic_json(capstone, old)
    assert not methods.historical_custody(historical_root, tmp_path / "bad-history")["ready"]
    monkeypatch.setattr(
        methods,
        "historical_custody",
        lambda *a: {
            "ready": False,
            "rows": [],
            "hashes": [],
            "gate_check_summary": [],
            "publication": {},
        },
    )
    assert methods.candidate(*args)["honest_verdict"] == "complete_blocked_historical_custody"
    assert value["contract_ready_score"] == 1


def test_missing_historical_authority_is_terminal_blocked(tmp_path):
    """SCENARIO-VERIFY-7992-CUSTODY: a missing external snapshot is a named gate operand."""
    root = tmp_path / "historical"
    old = json.loads((ROOT / "results/experiment_7991_v692_capstone.json").read_text())
    old["authority_snapshots"]["active"]["snapshot_path"] = str(tmp_path / "missing-v692.bin")
    atomic_json(root / "results/experiment_7991_v692_capstone.json", old)
    custody = methods.historical_custody(root, tmp_path / "saved")
    assert not custody["ready"]
    assert custody["gate_check_summary"][0]["observed"] is False


@pytest.mark.parametrize("live", ['tasks: [{{"id": "changed"}}]\n', "tasks: []\n"])
def test_private_fixture_ignores_mutable_roadmap(tmp_path, live):
    """SCENARIO-REPORT-7992-FROZEN-FIXTURE: live edits cannot change the private oracle."""
    import shutil

    root = tmp_path / "checkout"
    fixtures = root / "tests/fixtures/v693"
    fixtures.mkdir(parents=True)
    shutil.copyfile(ROOT / "tests/fixtures/v693/active.yaml.gz", fixtures / "active.yaml.gz")
    (root / "research-roadmap.yaml").write_text(live)
    (root / "research-roadmap-next.yaml").write_text(live)
    private = tmp_path / "private"
    validation.prepare(root, private)
    active = private / "active.yaml"
    assert sha256_file(active) == (
        "sha256:ae3a0133d3251da33b197553765f08ecbdd4630305cf2c9f1475d154f09cf56c"
    )
    assert not (private / "stage.yaml").exists()
    assessed = methods.assess(
        private / "design.md", private / "stage.yaml", active, private / "snapshots"
    )
    assert assessed["activated"] and len(assessed["contract_rows"]) == 13
    assert (root / "research-roadmap.yaml").read_text() == live


def test_malformed_live_authority_publishes_blocked_bytes(tmp_path):
    """SCENARIO-REPORT-7992-FROZEN-FIXTURE: malformed live YAML never borrows the oracle."""
    inputs = tmp_path / "inputs"
    validation.prepare(ROOT, inputs)
    active = inputs / "active.yaml"
    malformed = b'tasks: [{{"id": "changed"}}]\n'
    active.write_bytes(malformed)
    args = SimpleNamespace(
        design=inputs / "design.md",
        staged=inputs / "stage.yaml",
        active=active,
        source=SOURCE,
        fixture_e2e=True,
        output=tmp_path / "blocked.json",
        raw=tmp_path / "raw/rows.json",
    )
    value = validation.execute(ROOT, args, tmp_path / "work")
    assert value["verdict_class"] == "blocked" and value["contract_ready_score"] == 0
    assert not value["observed_activation"] and value["task_contract"] == []
    failure = next(
        r for r in value["gate_check_summary"] if r["artifact_field"] == "authority_yaml"
    )
    assert failure["path"] == str(active) and failure["hash"] == sha256_file(active)
    saved = Path(value["authority_snapshots"]["active"]["snapshot_path"])
    assert saved.read_bytes() == malformed and active.read_bytes() == malformed
    assert methods.cold_replay(args.output, args.raw)
