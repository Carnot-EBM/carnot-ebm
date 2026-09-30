"""REQ-REPORT-7940-V689: custody and administrative agreement earn no science credit."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from carnot.reporting import v689_contract_methods as methods
from carnot.reporting import v689_contract_validation as validation
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from scripts.experiments import experiment_7940_v689_contract_methods as cli

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/experiment_7892_v685_source_boundary.json"


def inputs(private):
    """Make a declared synthetic oracle so live missing history remains untouched."""
    validation.prepare(ROOT, private)
    return tuple(private / name for name in ("design.md", "stage.yaml", "active.yaml", "snapshots"))


def test_authority_mutations_and_missing_staging(tmp_path):
    """SCENARIO-REPORT-7940-AUTHORITY: reject each mutation from a passing baseline."""
    paths = inputs(tmp_path)
    observed = methods.assess(*paths)
    assert observed["activated"] and len(observed["contract_rows"]) == 13
    assert observed["activation_before_observation"] is False
    rows = methods.mutations(paths[0], paths[2], SOURCE, tmp_path / "mutations")
    assert len(rows) == 12 and all(row["passed"] for row in rows)
    paths[1].unlink()
    assert methods.assess(*paths)["activated"]  # preserved actual staged snapshot
    assert not methods.assess(*(*paths[:3], tmp_path / "unseen"))["activated"]
    paths[0].unlink()
    assert not methods.assess(*paths)["activated"]


def test_freeze_custody_and_manifest(tmp_path):
    """SCENARIO-REPORT-7940-VALIDATION: methods, sources and exact dates bind the run."""
    freeze = methods.method_freeze(ROOT)
    assert len(freeze["future_validation_scopes"]) == 13
    assert len(freeze["training"]["arm_names"]) == 9
    assert freeze["sentence_labels"]["selection_before_labels"]
    assert freeze["delayed_aci"]["total_delays"] == [21, 24, 36]
    assert freeze["qualified_evidence_custody"]["ready"]
    assert any(row["decision"] == "defer" for row in freeze["literature_adoption_decisions"])
    value = validation.manifest(ROOT, tmp_path)
    assert value["coverage_includes"] == validation.OWNED
    commands = {row["name"]: row for row in value["commands"]}
    for name in ("e2e_016_fixture", "e2e_016_replay"):
        argv = commands[name]["argv"]
        assert argv[argv.index("--date") + 1] == "20260929"
    outputs = []
    for row in value["commands"]:
        argv = row["argv"]
        if "--output" in argv:
            outputs.append(Path(argv[argv.index("--output") + 1]).parent)
    assert len(outputs) == len(set(outputs))
    assert not validation.coverage_complete(tmp_path / "absent")


def test_candidate_cold_reduction_and_failures(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7940-VALIDATION: primitive readiness and custody control claims."""
    paths = inputs(tmp_path)
    manifest = tmp_path / "manifest.json"
    atomic_json(manifest, validation.manifest(ROOT, tmp_path))
    args = (
        ROOT,
        methods.assess(*paths),
        methods.source_custody(SOURCE, methods.SOURCE_SHA256),
        methods.method_freeze(ROOT),
        [],
        [{"passed": True, "argv": ["true"]}],
        manifest,
        1,
        2,
    )
    value = methods.candidate(*args)
    assert value["experiment_id"] == 7940 and value["lifecycle_task_count"] == 13
    assert (
        value["contract_ready_score"] == 1 and value["source_sample_size_budget"]["eligible"] == 604
    )
    output, raw = tmp_path / "result.json", tmp_path / "rows.json"
    atomic_json(output, value)
    atomic_json(raw, methods.primitive_rows(value))
    assert methods.cold_replay(output, raw)
    for field, observed in (("experiment_id", 0), ("execution_date", "20260929")):
        changed = deepcopy(value)
        changed[field] = observed
        atomic_json(output, changed)
        assert not methods.cold_replay(output, raw)
    failed = methods.candidate(*(*args[:5], [{"passed": False, "argv": ["false"]}], *args[6:]))
    assert failed["verdict_class"] == "disqualified" and failed["contract_ready_score"] == 0
    monkeypatch.setattr(
        methods,
        "qualified_custody",
        lambda root: {
            "ready": False,
            "hashes": [],
            "rows": [],
            "gate_check_summary": [
                methods.shared.operand(SOURCE, "sha256", "expected", None, "exp7916")
            ],
        },
    )
    blocked = methods.candidate(*args)
    assert blocked["verdict_class"] == "blocked" and blocked["contract_ready_score"] == 0
    fixture = methods.candidate(*args, fixture=True)
    assert fixture["honest_verdict"].startswith("complete_")


def test_cli_guards_and_adapter(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7940-PUBLICATION: fixtures cannot publish historical evidence."""
    for argv in (
        ["exp", "--fixture-e2e"],
        ["exp", "--date", "wrong"],
        ["exp", "--terminal-recheck", str(tmp_path / "missing")],
    ):
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit):
            cli.main()
    monkeypatch.setattr(cli, "execute", lambda *args: {})
    monkeypatch.setattr(sys, "argv", ["exp", "--output", str(tmp_path / "result.json")])
    assert cli.main() == 0


def test_execution_publication_and_live_readers(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7940-PUBLICATION: real consumers select checked final bytes."""
    paths = inputs(tmp_path / "inputs")
    args = SimpleNamespace(
        design=paths[0],
        staged=paths[1],
        active=paths[2],
        source=SOURCE,
        output=tmp_path / "success/experiment_7940_fixture.json",
        raw=tmp_path / "success/raw/experiment_7940_fixture/rows.json",
        fixture_e2e=False,
    )
    private = tmp_path / "private"
    private.mkdir()
    report = {
        "files": {
            name: {"summary": {"num_statements": 1, "covered_lines": 1}, "missing_lines": []}
            for name in validation.OWNED
        }
    }
    atomic_json(private / "coverage.json", report)
    (private / ".coverage.combined").write_bytes(b"test-only synthetic coverage database")
    monkeypatch.setattr(
        validation,
        "manifest",
        lambda *args: {
            "commands": [dict(name="fixture_check", argv=["true"], expected_exit=0, deadline_s=1)]
        },
    )
    monkeypatch.setattr(
        validation.shared, "run_check", lambda *args: dict(passed=True, argv=["true"])
    )
    value = validation.execute(ROOT, args, private)
    assert value["contract_ready_score"] == 1
    terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_text())
    assert terminal["primary_sha256"] == sha256_file(args.output)
    selected = json.loads((args.raw.parent / "primary_resolution_receipt.json").read_text())
    assert selected["gate_sha256"] == selected["document_sha256"] == sha256_file(args.output)
    assert terminal["passed"] and methods.cold_replay(args.output, args.raw)
    assert validation.coverage_complete(private / "coverage.json")
    atomic_json(args.output.parent / "experiment_7940_shadow.json", value)
    with pytest.raises(ValueError, match="conflicting_primary"):
        validation.publish(ROOT, value, args.output, args.raw, tmp_path / "conflict")
    (args.output.parent / "experiment_7940_shadow.json").unlink()
    monkeypatch.setattr(
        validation,
        "reader_receipt",
        lambda *args, **kwargs: {"gate_sha256": "wrong", "document_sha256": "wrong"},
    )
    with pytest.raises(ValueError, match="primary_reader_drift"):
        validation.publish(ROOT, value, args.output, args.raw, tmp_path / "drift")


def test_fixture_execution_and_owned_failure(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7940-VALIDATION: fixture results and owned failures stay distinct."""
    paths = inputs(tmp_path / "inputs")
    args = SimpleNamespace(
        design=paths[0],
        staged=paths[1],
        active=paths[2],
        source=SOURCE,
        output=tmp_path / "fixture.json",
        raw=tmp_path / "rows.json",
        fixture_e2e=True,
    )
    fixture = validation.execute(ROOT, args, tmp_path / "private")
    assert fixture["verdict_class"] == "null" and fixture["contract_ready_score"] == 0
    args.fixture_e2e = False
    monkeypatch.setattr(
        validation,
        "manifest",
        lambda *args: {
            "commands": [dict(name="owned_check", argv=["false"], expected_exit=0, deadline_s=1)]
        },
    )
    monkeypatch.setattr(
        validation.shared, "run_check", lambda *args: dict(passed=False, argv=["false"])
    )
    monkeypatch.setattr(validation, "publish", lambda *args: None)
    failed = validation.execute(ROOT, args, tmp_path / "failure-private")
    assert failed["verdict_class"] == "disqualified" and failed["contract_ready_score"] == 0


def test_live_block_and_preserved_historical_counts(tmp_path):
    """SCENARIO-REPORT-7940-AUTHORITY: current blockers cannot erase older fixture counts."""
    import gzip

    live = methods.assess(
        ROOT / "openspec/change-proposals/research-roadmap-vNEXT.md",
        ROOT / "research-roadmap-next.yaml",
        ROOT / "research-roadmap.yaml",
        tmp_path,
    )
    assert not live["activated"] and live["gate_check_summary"]
    assert not live["contract_rows"][2]["checks"]["prior"]
    for version, count, first in (("v688", 12, 7928), ("v687", 13, 7915)):
        directory = tmp_path / version
        directory.mkdir()
        for name in ("design.md", "active.yaml"):
            (directory / name).write_bytes(
                gzip.decompress((ROOT / f"tests/fixtures/{version}/{name}.gz").read_bytes())
            )
        old = methods.shared.lifecycle.assess_authorities(
            directory / "design.md",
            directory / "staged.yaml",
            directory / "active.yaml",
            directory / "snapshots",
            milestone=f"2026.09.{version[1:]}",
            first_id=first,
            count=count,
        )
        assert old["activated"] and len(old["contract_rows"]) == count
