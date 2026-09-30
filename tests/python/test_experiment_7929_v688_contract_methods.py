"""REQ-REPORT-7929-V688: current authority keeps historical evidence intact."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

from carnot.reporting import v688_contract_methods as methods
from carnot.reporting import v688_contract_validation as validation
from carnot.reporting.current_work_receipt import atomic_json
from scripts.experiments import experiment_7929_v688_contract_methods as cli

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/experiment_7892_v685_source_boundary.json"


def test_authority_and_mutations(tmp_path):
    """SCENARIO-REPORT-7929-AUTHORITY: each complete-task edit must fail."""
    validation.prepare(ROOT, tmp_path)
    paths = (
        tmp_path / "design.md",
        tmp_path / "stage.yaml",
        tmp_path / "active.yaml",
        tmp_path / "snapshots",
    )
    observed = methods.assess(*paths)
    assert observed["activated"] and len(observed["contract_rows"]) == 12
    assert not observed["authority_snapshots"]["staged"]["exists"]
    rows = methods.mutations(paths[0], paths[2], SOURCE, tmp_path / "mutations")
    assert len(rows) == 12 and all(row["passed"] for row in rows)
    paths[1].write_text("milestone: 2026.10.689\ntasks: []\n")
    assert methods.assess(*paths)["activated"]
    paths[2].write_text("milestone: 2026.09.687\ntasks: []\n")
    assert not methods.assess(*paths)["activated"]


def test_custody_and_freeze(tmp_path):
    """SCENARIO-REPORT-7929-CUSTODY: exact primaries and durable logs are operands."""
    custody = methods.qualified_custody(ROOT)
    assert custody["ready"], custody["gate_check_summary"]
    assert len(custody["rows"]) == 2
    assert all(row["score"] == 1 and row["flagged_adversarial"] is False for row in custody["rows"])
    assert methods.qualified_custody(tmp_path)["gate_check_summary"]
    freeze = methods.method_freeze(ROOT)
    assert len(freeze["future_validation_scopes"]) == 12
    assert freeze["delayed_aci"]["total_delays"] == [21, 24, 36]
    assert any("2607.20792" in row["source"] for row in freeze["literature_adoption_decisions"])
    manifest = validation.manifest(ROOT, tmp_path)
    assert manifest["coverage_includes"] == validation.OWNED
    for row in manifest["commands"]:
        if row["name"] in {"e2e_016_fixture", "e2e_016_replay"}:
            assert row["argv"][row["argv"].index("--date") + 1] == "20260929"


def test_candidate_reduction_and_owned_failure(tmp_path):
    """SCENARIO-REPORT-7929-VALIDATION: recompute readiness and producer identity."""
    validation.prepare(ROOT, tmp_path)
    manifest = tmp_path / "manifest.json"
    atomic_json(manifest, validation.manifest(ROOT, tmp_path))
    args = (
        ROOT,
        methods.assess(
            tmp_path / "design.md",
            tmp_path / "stage.yaml",
            tmp_path / "active.yaml",
            tmp_path / "snapshots",
        ),
        methods.source_custody(SOURCE, methods.SOURCE_SHA256),
        methods.method_freeze(ROOT),
        [],
        [{"passed": True, "argv": ["true"]}],
        manifest,
        1,
        2,
    )
    value = methods.candidate(*args)
    assert value["experiment_id"] == 7929 and value["lifecycle_task_count"] == 12
    assert value["source_sample_size_budget"]["excluded"] == 36
    output, raw = tmp_path / "result.json", tmp_path / "rows.json"
    atomic_json(output, value)
    atomic_json(raw, methods.primitive_rows(value))
    assert methods.cold_replay(output, raw)
    changed = deepcopy(value)
    changed["execution_date"] = "20260929"
    atomic_json(output, changed)
    assert not methods.cold_replay(output, raw)
    changed["experiment_id"] = 0
    atomic_json(output, changed)
    assert not methods.cold_replay(output, raw)
    failed = methods.candidate(*(*args[:5], [{"passed": False, "argv": ["false"]}], *args[6:]))
    assert failed["verdict_class"] == "disqualified" and failed["contract_ready_score"] == 0


def test_cli_guards(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7929-VALIDATION: fixture writes stay in private paths."""
    for argv in (
        ["exp7929", "--fixture-e2e"],
        ["exp7929", "--terminal-recheck", str(tmp_path / "missing")],
        ["exp7929", "--date", "wrong"],
    ):
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit):
            cli.main()


def test_execute_adapter(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7929-VALIDATION: the wrapper reuses the supervised runner."""

    def execute(root, args, private, **kwargs):
        assert kwargs["experiment_id"] == 7929 and kwargs["count"] == 12
        return {"done": True}

    monkeypatch.setattr(validation.shared, "execute", execute)
    assert validation.execute(ROOT, SimpleNamespace(fixture_e2e=True), tmp_path) == {"done": True}
    monkeypatch.setattr(cli, "execute", lambda *args: {})
    monkeypatch.setattr(sys, "argv", ["exp7929", "--output", str(tmp_path / "result.json")])
    assert cli.main() == 0


def test_missing_authority_and_external_custody_block(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7929-CUSTODY: outside absence is terminal blocked."""
    validation.prepare(ROOT, tmp_path)
    missing = methods.assess(
        tmp_path / "missing.md",
        tmp_path / "stage.yaml",
        tmp_path / "active.yaml",
        tmp_path / "snapshots",
    )
    assert not missing["activated"] and missing["gate_check_summary"]
    manifest = tmp_path / "manifest.json"
    atomic_json(manifest, validation.manifest(ROOT, tmp_path))
    atomic_json(
        tmp_path / "coverage.json-observed",
        {"files": {"owned.py": {"summary": {"num_statements": 1, "covered_lines": 1}}}},
    )
    monkeypatch.setattr(
        methods,
        "qualified_custody",
        lambda root: {
            "ready": False,
            "rows": [],
            "hashes": [],
            "gate_check_summary": [
                {"upstream_id": "exp7916", "artifact_field": "sha256", "observed": None}
            ],
        },
    )
    value = methods.candidate(
        ROOT,
        missing,
        methods.source_custody(SOURCE, methods.SOURCE_SHA256),
        methods.method_freeze(ROOT),
        [],
        [{"passed": True, "argv": ["true"]}],
        manifest,
        1,
        2,
    )
    assert value["verdict_class"] == "blocked" and value["contract_ready_score"] == 0
    assert value["coverage_statement_counts"]["owned.py"]["num_statements"] == 1


def test_live_readers_after_newer_nested_sidecar(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7929-VALIDATION: both real readers see the primary bytes."""
    from carnot.reporting.current_work_receipt import sha256_file

    output = tmp_path / "experiment_7929_fixture.json"
    raw = tmp_path / "raw/experiment_7929_fixture/rows.json"
    value = dict(
        experiment_id=7929,
        task_id="exp7929-contract-methods",
        honest_verdict="complete_null_fixture",
        verdict_class="null",
        flagged_adversarial=False,
        contract_ready_score=0,
    )

    def execute(*args, **kwargs):
        atomic_json(output, value)
        terminal = raw.parent / "terminal_validation"
        atomic_json(
            terminal / f"terminal-{sha256_file(output)[7:]}.json",
            {"passed": True, "candidate_sha256": sha256_file(output)},
        )
        return value

    monkeypatch.setattr(validation.shared, "execute", execute)
    args = SimpleNamespace(fixture_e2e=False, output=output, raw=raw)
    assert validation.execute(ROOT, args, tmp_path) == value
    receipt = json.loads((raw.parent / "primary_resolution_receipt.json").read_text())
    assert receipt["gate_sha256"] == receipt["document_sha256"] == sha256_file(output)
    assert not validation.coverage_complete(tmp_path / "missing-coverage.json")
    atomic_json(tmp_path / "experiment_7929_shadow.json", {**value, "shadow": True})
    stamp = time.time() + 120
    os.utime(tmp_path / "experiment_7929_shadow.json", (stamp, stamp))
    with pytest.raises(ValueError, match="primary_reader_drift"):
        validation.execute(ROOT, args, tmp_path)
