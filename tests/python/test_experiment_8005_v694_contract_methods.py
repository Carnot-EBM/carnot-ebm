"""REQ-REPORT-8005: private authority and publication checks preserve history."""

from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from scripts.experiments import experiment_8005_v694_contract_methods as cli

ROOT = Path(__file__).resolve().parents[2]


def inputs(private):
    """Frozen fixtures keep later activation from changing this qualification."""
    private.mkdir(parents=True, exist_ok=True)
    for name in ("design.md", "active.yaml"):
        (private / name).write_bytes(
            gzip.decompress((ROOT / "tests/fixtures/v694" / f"{name}.gz").read_bytes())
        )
    return SimpleNamespace(
        design=private / "design.md",
        active=private / "active.yaml",
        staged=private / "absent.yaml",
        fixture_e2e=True,
        output=private / "experiment_8005_fixture.json",
        raw=private / "raw/rows.json",
    )


def test_authority_mutations_and_missing_fields(tmp_path):
    """SCENARIO-REPORT-8005-AUTHORITY: exact gates and complete bytes reject drift."""
    args = inputs(tmp_path)
    result = cli.assess(args.design, args.staged, args.active, tmp_path / "snapshots")
    assert result["activated"] and len(result["contract_rows"]) == 13
    assert result["staging_custody_status"] == "unknown_consumed"
    assert len(result["lineage_applicability_rows"]) == 13
    assert all(r["passed"] for r in result["lineage_applicability_rows"])
    rows = cli.mutations(args.design, args.active, tmp_path / "absent", tmp_path / "mutations")
    assert len(rows) == 12 and all(r["passed"] for r in rows)
    original = args.design.read_text()
    for token in (
        "## Exact task contract",
        "V694_TASK_CONTRACT_START",
        "Canonical full-task SHA-256",
    ):
        args.design.write_text(original.replace(token, "removed", 1))
        failed = cli.assess(args.design, args.staged, args.active, tmp_path / "missing")
        assert not failed["activated"] and failed["gate_check_summary"]
    args.design.write_text(original)
    args.design.write_text(
        original.replace('"prompt": "CONTEXT:', '"prompt": "CHANGED CONTEXT:', 1)
    )
    assert not cli.assess(args.design, args.staged, args.active, tmp_path / "machine-prompt")[
        "activated"
    ]
    args.design.write_text(original)
    baseline = yaml.safe_load(args.active.read_bytes())
    for field in ("title", "prompt", "prior_failures", "artifact_field", "upstream"):
        changed = deepcopy(baseline)
        if field in ("artifact_field", "upstream"):
            next(t for t in changed["tasks"] if t["gated_on"])["gated_on"][0][field] = "changed"
        else:
            changed["tasks"][0][field] = "changed"
        args.active.write_text(yaml.safe_dump(changed, sort_keys=False))
        assert not cli.assess(args.design, args.staged, args.active, tmp_path / field)["activated"]


def test_private_execution_and_cold_custody(tmp_path):
    """SCENARIO-REPORT-8005-PUBLICATION: primitives and checkpoints remain byte bound."""
    args = inputs(tmp_path)
    value = cli.execute(args, tmp_path / "work")
    assert value["contract_ready_score"] == 1 and value["verdict_class"] == "circular_positive"
    assert value["run_date"] == "20261002" and value["model_invocation_counts"]["calls"] == 0
    assert value["MODEL_SPECS"] == value["trained_head_specs"] == []
    assert value["sample_size_budget"]["independent"] == 0
    assert len(value["method_freeze"]["task_contracts"]) == 13
    assert value["historical_failure_rows"][0]["honest_verdict"] == "complete_blocked_authority"
    assert len(value["lineage_rows"]) == 13
    assert all(
        {"exclusion_reason", "censor_reason", "raw_numerator", "raw_denominator"} <= set(row)
        for row in value["rows"]
    )
    assert cli.cold_replay(args.output, args.raw)
    original = args.output.read_bytes()
    for key, replacement in (("experiment_id", 0), ("contract_ready_score", 0), ("rows", [])):
        atomic_json(args.output, {**value, key: replacement})
        assert not cli.cold_replay(args.output, args.raw)
    args.output.write_bytes(original)
    checkpoint = Path(value["checkpoint_references"][0]["path"])
    saved = checkpoint.read_bytes()
    checkpoint.write_bytes(b"changed")
    assert not cli.cold_replay(args.output, args.raw)
    checkpoint.write_bytes(saved)
    assert cli.cold_replay(args.output, args.raw)
    args.design.write_text("# Missing V694 authority\n")
    blocked = cli.execute(args, tmp_path / "blocked-work")
    assert blocked["verdict_class"] == "blocked" and blocked["contract_ready_score"] == 0
    assert all(
        {"path", "hash", "artifact_field", "passed"} <= set(r)
        for r in blocked["gate_check_summary"]
    )
    assert cli.cold_replay(args.output, args.raw)


def test_owned_execution_and_manifest(tmp_path, monkeypatch):
    """REQ-REPORT-8005: real owned runner failures disqualify; static checks stay explicit."""
    args = inputs(tmp_path)
    args.fixture_e2e = False
    frozen = cli.manifest(ROOT, tmp_path / "work")
    assert frozen["coverage_includes"] == [cli.OWNED]
    assert any(r["name"] == "repository_full_suite" for r in frozen["commands"])
    assert not any("e2e_016" in r["name"] for r in frozen["commands"])
    original = cli.manifest

    def manifest(root, private):
        value = original(root, private)
        value["commands"] = [{"name": "owned_failure", "argv": ["false"]}]
        return value

    monkeypatch.setattr(cli, "manifest", manifest)
    runner = cli.validation.run_check
    monkeypatch.setattr(
        cli.validation,
        "run_check",
        lambda *a: (
            dict(name="owned_failure", passed=False, argv=["false"], actual_exit=1)
            if a[1]["name"] == "owned_failure"
            else runner(*a)
        ),
    )
    monkeypatch.setattr(
        cli.validation, "publish", lambda root, value, output, *rest: atomic_json(output, value)
    )
    value = cli.execute(args, tmp_path / "work")
    assert value["verdict_class"] == "disqualified" and value["contract_ready_score"] == 0
    assert (
        sha256_file(args.output)
        == json.loads((args.raw.parent / "primary_resolution_receipt.json").read_text())[
            "gate_sha256"
        ]
    )
    work = tmp_path / "work"
    atomic_json(
        work / "coverage.json",
        {
            "files": {
                cli.OWNED: {
                    "summary": {"num_statements": 1, "covered_lines": 1},
                    "missing_lines": [],
                }
            }
        },
    )
    monkeypatch.setattr(
        cli,
        "reader_receipt",
        lambda *a, **kw: dict(gate_sha256="changed", document_sha256="changed"),
    )
    with pytest.raises(ValueError, match="primary_reader_drift"):
        cli.execute(args, work)


def test_imported_main_and_absent_replay(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8005-PUBLICATION: imported argument dispatch has real coverage."""
    args = inputs(tmp_path)
    argv = [
        "experiment_8005",
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
    monkeypatch.setattr("sys.argv", argv)
    assert cli.main() == 0
    monkeypatch.setattr(
        "sys.argv", ["experiment_8005", "--cold-replay", str(args.output), "--raw", str(args.raw)]
    )
    assert cli.main() == 0
    monkeypatch.setattr("sys.argv", ["experiment_8005", "--cold-replay", str(tmp_path / "absent")])
    assert cli.main() == 1
    monkeypatch.setattr("sys.argv", ["experiment_8005", "--fixture-e2e"])
    with pytest.raises(SystemExit):
        cli.main()


def test_real_cli_success_blocked_and_negative(tmp_path):
    """SCENARIO-REPORT-8005-AUTHORITY: direct imports and fresh replay need no caller PYTHONPATH."""
    args = inputs(tmp_path)
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    base = [str(ROOT / ".venv/bin/python"), "-u", str(ROOT / cli.OWNED)]
    success = [
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

    def run(argv):
        print("[test8005] subprocess_before", flush=True)
        result = subprocess.run(
            base + argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60
        )
        print(f"[test8005] subprocess_after exit={result.returncode}", flush=True)
        return result

    result = run(success)
    assert result.returncode == 0, result.stdout + result.stderr
    replay = ["--cold-replay", str(args.output), "--raw", str(args.raw)]
    assert run(replay).returncode == 0
    atomic_json(args.output, {"experiment_id": 0})
    assert run(replay).returncode == 1
    args.design.write_text("# Missing V694 contract\n")
    assert run(success).returncode == 0
    blocked = json.loads(args.output.read_text())
    assert blocked["verdict_class"] == "blocked" and blocked["contract_ready_score"] == 0
    assert run(replay).returncode == 0
    assert run(["--date", "bad"]).returncode == 2
    assert run(["--fixture-e2e"]).returncode == 2
