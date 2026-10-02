"""REQ-REPORT-8018: private immutable authority and terminal reader qualification."""

from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from carnot.reporting import v695_contract_methods as methods
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from scripts.experiments import experiment_8018_v695_contract_methods as cli

ROOT = Path(__file__).resolve().parents[2]


def inputs(private):
    """Frozen oracle inputs keep later activation and UTC dates out of old tests."""
    private.mkdir(parents=True, exist_ok=True)
    for name in ("design.md", "active.yaml"):
        (private / name).write_bytes(
            gzip.decompress((ROOT / "tests/fixtures/v695" / (name + ".gz")).read_bytes())
        )
    return SimpleNamespace(
        design=private / "design.md",
        active=private / "active.yaml",
        staged=private / "absent.yaml",
        fixture_e2e=True,
        output=private / "experiment_8018_fixture.json",
        raw=private / "raw/rows.json",
        date="20261002",
    )


def test_authority_and_mutations(tmp_path):
    """SCENARIO-REPORT-8018-AUTHORITY: all thirteen tasks and twelve mutations bind."""
    args = inputs(tmp_path)
    result = methods.assess(args.design, args.staged, args.active, tmp_path / "snapshots")
    assert result["activated"] and len(result["contract_rows"]) == 13
    assert result["staging_custody_status"] == "unknown_consumed"
    assert all(r["passed"] for r in result["lineage_applicability_rows"])
    controls = methods.mutations(args.design, args.active, args.staged, tmp_path / "mutations")
    assert len(controls) == 12 and all(r["passed"] for r in controls)
    original = args.design.read_text()
    for token in (
        "## Exact task contract",
        "V695_TASK_CONTRACT_START",
        "Canonical full-task SHA-256",
    ):
        args.design.write_text(original.replace(token, "removed", 1))
        failed = methods.assess(args.design, args.staged, args.active, tmp_path / "missing")
        assert not failed["activated"] and failed["gate_check_summary"]
    args.design.write_text(original.replace('"prompt": "CONTEXT:', '"prompt": "CHANGED:', 1))
    assert not methods.assess(args.design, args.staged, args.active, tmp_path / "prompt")[
        "activated"
    ]
    args.design.write_text(original)
    args.staged.write_text("milestone: 2026.10.694\ntasks: []\n")
    assert not methods.assess(args.design, args.staged, args.active, tmp_path / "stage")[
        "activated"
    ]
    args.staged.unlink()
    changed = yaml.safe_load(args.active.read_bytes())
    next(t for t in changed["tasks"] if t["gated_on"])["gated_on"][0]["artifact_field"] = "typo"
    args.active.write_text(yaml.safe_dump(changed, sort_keys=False))
    assert not methods.assess(args.design, args.staged, args.active, tmp_path / "gate")["activated"]


def test_private_execution_and_configuration_drift(tmp_path):
    """SCENARIO-REPORT-8018-TERMINAL: frozen configuration and primitive drift reject replay."""
    args = inputs(tmp_path)
    value = methods.execute(args, tmp_path / "work")
    assert value["contract_ready_score"] == 1 and value["verdict_class"] == "circular_positive"
    assert value["run_date"] == "20261002" and value["model_invocation_counts"]["calls"] == 0
    assert value["MODEL_SPECS"] == value["trained_head_specs"] == []
    assert value["sample_size_budget"]["independent"] == 0
    assert len(value["method_freeze"]["task_contracts"]) == 13
    assert len(value["method_freeze"]["primary_comparisons"]) == 3
    assert value["historical_reader_failure_rows"] and all(
        r["log_authenticated"] for r in value["historical_reader_failure_rows"]
    )
    assert all(
        Path(r["path"]).is_relative_to(args.raw.parent) for r in value["checkpoint_references"]
    )
    assert methods.cold_replay(args.output, args.raw)
    original = args.output.read_bytes()
    for key, replacement in (
        ("experiment_id", 0),
        ("contract_ready_score", 0),
        ("sample_size_budget", {}),
    ):
        changed = deepcopy(value)
        changed[key] = replacement
        atomic_json(args.output, changed)
        assert not methods.cold_replay(args.output, args.raw)
    args.output.write_bytes(original)
    ref = next(r for r in value["checkpoint_references"] if "method_freeze" in r["path"])
    Path(ref["path"]).write_bytes(b"configuration byte drift")
    assert not methods.cold_replay(args.output, args.raw)
    assert not methods.cold_replay(tmp_path / "absent", args.raw)
    atomic_json(args.output, {"experiment_id": 0})
    assert not methods.cold_replay(args.output, args.raw)
    atomic_json(args.output, {})
    assert not methods.cold_replay(args.output, args.raw)


def test_missing_live_design_is_terminal_blocked(tmp_path):
    """REQ-REPORT-8018: private fixture bytes cannot repair missing external design authority."""
    args = inputs(tmp_path)
    args.design.unlink()
    value = methods.execute(args, tmp_path / "work")
    assert value["verdict_class"] == "blocked" and value["contract_ready_score"] == 0
    assert all(r["passed"] is False for r in value["gate_check_summary"])
    assert methods.cold_replay(args.output, args.raw)


def test_manifest_and_owned_failure(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8018-TERMINAL: owned failures disqualify; health remains separate."""
    args = inputs(tmp_path)
    args.fixture_e2e = False
    manifest = methods.manifest(ROOT, tmp_path / "work")
    assert manifest["coverage_includes"] == methods.OWNED
    assert (
        next(r for r in manifest["commands"] if r["name"] == "repository_full_suite")[
            "classification"
        ]
        == "diagnostic"
    )
    monkeypatch.setattr(
        methods,
        "manifest",
        lambda *a: dict(
            manifest, commands=[dict(name="failed", argv=["private"], expected_exit=0)]
        ),
    )
    monkeypatch.setattr(
        methods.validation,
        "run_check",
        lambda *a, **k: dict(name="failed", argv=["private"], passed=False, actual_exit=1),
    )
    monkeypatch.setattr(methods, "publish", lambda *a: None)
    value = methods.execute(args, tmp_path / "work")
    assert value["verdict_class"] == "disqualified" and value["contract_ready_score"] == 0


def test_private_direct_cli(tmp_path):
    """SCENARIO-REPORT-8018-TERMINAL: actual script accepts private work and rejects tampering."""
    args = inputs(tmp_path)
    entry = ROOT / methods.CLI
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    base = [str(ROOT / ".venv/bin/python"), "-u", str(entry)]
    fixture = [
        "--date",
        "20261002",
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
    value = json.loads(args.output.read_text())
    value["contract_ready_score"] = 0
    atomic_json(args.output, value)
    assert (
        subprocess.run(replay, cwd=tmp_path, env=env, capture_output=True, timeout=60).returncode
        == 1
    )
    with pytest.raises(SystemExit):
        cli.main(["--fixture-e2e"])
    assert cli.main(["--cold-replay", str(tmp_path / "missing"), "--raw", str(args.raw)]) == 1
    with pytest.raises(SystemExit):
        cli.main(["--date", "20261003"])


def test_checked_private_publication(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8018-TERMINAL: real readers select exact terminal bytes and bound sidecars."""
    args = inputs(tmp_path)
    args.fixture_e2e = False
    real = methods.manifest
    private = tmp_path / "work"
    private.mkdir()
    counts = {
        p: dict(summary=dict(num_statements=1, covered_lines=1), missing_lines=[])
        for p in methods.OWNED
    }
    atomic_json(private / "coverage.json", dict(files=counts))
    monkeypatch.setattr(
        methods,
        "manifest",
        lambda *a: dict(
            real(*a),
            commands=[
                dict(
                    name="private_fixture_check",
                    argv=[
                        str(ROOT / ".venv/bin/python"),
                        "-c",
                        "print('private protocol check', flush=True)",
                    ],
                    expected_exit=0,
                    deadline_s=60,
                )
            ],
        ),
    )
    value = methods.execute(args, private)
    assert value["contract_ready_score"] == 1
    report = json.loads(Path(value["terminal_validation_sidecar_path"]).read_text())
    assert report["passed"] and report["primary_sha256"] == sha256_file(args.output)
    assert methods.cold_replay(args.output, args.raw)
    assert all(
        r["passed"]
        for r in json.loads((args.raw.parent / "published_recheck.json").read_text())["receipts"]
    )


def test_capstone_retry_preserves_previous_invocation(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8018-TERMINAL: a retry must retain earlier immutable operands."""
    import runpy
    from carnot.reporting import v694_capstone as cap

    fixture = runpy.run_path(str(ROOT / "tests/python/test_experiment_8017_v694_capstone.py"))[
        "fixture"
    ]
    root = tmp_path / "input"
    fixture(root)
    output = tmp_path / "experiment_8017_v694_capstone.json"
    real_manifest, real_run = cap.manifest, cap.run_check

    def manifest(private):
        value = real_manifest(private)
        value["commands"] = [
            s for s in value["commands"] if s["name"] in {"publication_gate", "coverage_json"}
        ]
        return value

    def run(repo, spec, private, durable):
        if spec["name"] not in {"publication_gate", "coverage_json"}:
            return real_run(repo, spec, private, durable)
        log = durable / (spec["name"] + ".json")
        atomic_json(
            log,
            dict(
                paper_ready=False,
                unmet_gates=["G2"],
                gates={f"G{i}": {"pass": i != 2} for i in range(1, 5)},
            ),
        )
        if spec["name"] == "coverage_json":
            atomic_json(
                private.parent / "coverage.json",
                dict(
                    files={
                        p: {"summary": {"num_statements": 1, "covered_lines": 1}} for p in cap.OWNED
                    }
                ),
            )
        return dict(
            spec, passed=True, actual_exit=0, log_path=str(log), log_sha256=sha256_file(log)
        )

    monkeypatch.setattr(cap, "manifest", manifest)
    monkeypatch.setattr(cap, "run_check", run)
    assert cap.qualify(root, root / "design.md", "20261002", output) == 0
    original = json.loads(output.read_text())
    refs = original["checkpoint_references"]
    monkeypatch.setattr(cap, "build", lambda *a: deepcopy(original))
    assert cap.qualify(root, root / "design.md", "20261002", output) == 0
    assert all(sha256_file(Path(r["path"])) == r["sha256"] for r in refs)
    assert cap.cold_replay(json.loads(output.read_text())) == []


def test_terminal_faults_and_malformed_replay(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8018-TERMINAL: drift in final readers cannot silently publish."""
    args = inputs(tmp_path)
    value = methods.execute(args, tmp_path / "work")
    atomic_json(tmp_path / "malformed.json", {"experiment_id": 8018})
    assert not methods.cold_replay(tmp_path / "malformed.json", args.raw)
    (tmp_path / "malformed.json").write_text("not json")
    assert not methods.cold_replay(tmp_path / "malformed.json", args.raw)
    real_reader = methods.reader_receipt
    monkeypatch.setattr(methods.validation, "publish", lambda root, v, out, *a: atomic_json(out, v))
    monkeypatch.setattr(
        methods.validation, "run_check", lambda root, spec, *a: dict(spec, passed=True)
    )
    monkeypatch.setattr(
        methods,
        "reader_receipt",
        lambda *a, **k: dict(gate_sha256="wrong", document_sha256="wrong"),
    )
    with pytest.raises(ValueError, match="primary_reader_drift"):
        methods.publish(value, args.output, args.raw, tmp_path / "terminal", args.raw.parent)
    monkeypatch.setattr(methods, "reader_receipt", real_reader)
    monkeypatch.setattr(
        methods.validation,
        "run_check",
        lambda root, spec, *a: dict(spec, passed=not spec["name"].startswith("published_")),
    )
    with pytest.raises(ValueError, match="published_validation_failed"):
        methods.publish(value, args.output, args.raw, tmp_path / "terminal2", args.raw.parent)
