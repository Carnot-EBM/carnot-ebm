"""REQ-REPORT-7953-V690: verify observed identity without invented custody."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

from carnot.reporting import v690_authority as authority
from carnot.reporting import v690_contract_methods as methods
from carnot.reporting import v690_contract_validation as validation
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from scripts.failure_ledger import FailureLedger, LedgerEntry
from scripts.experiments import experiment_7953_v690_contract_methods as cli

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/experiment_7892_v685_source_boundary.json"


def inputs(private):
    """Copy real independent authorities; never create fake prior history."""
    validation.prepare(ROOT, private)
    return tuple(private / n for n in ("design.md", "stage.yaml", "active.yaml", "snapshots"))


def test_activation_custody_and_mutations(tmp_path):
    """SCENARIO-REPORT-7953-AUTHORITY: active identity survives consumed staging."""
    paths = inputs(tmp_path)
    result = authority.assess(*paths)
    assert result["observed_activation"] and result["planning_ready_score"] == 1
    assert len(result["contract_rows"]) == len(result["lineage_applicability_rows"]) == 13
    paths[1].unlink()
    result = authority.assess(*(*paths[:3], tmp_path / "unseen"))
    assert result["activated"] and result["staging_custody_status"] == "unknown_consumed"
    assert result["planning_ready_score"] == 0
    assert all(r["passed"] for r in methods.mutations(paths[0], paths[2], SOURCE, tmp_path / "m"))
    paths[1].write_bytes(paths[2].read_bytes())
    paths[2].unlink()
    result = authority.assess(*paths)
    assert result["planning_ready_score"] == 1 and not result["observed_activation"]
    paths[0].unlink()
    assert not authority.assess(*paths)["activated"]


@pytest.mark.parametrize(
    "key", ["experiment_id", "verdict", "addressed_by", "retire_if_same_verdict"]
)
def test_conditional_history(tmp_path, key):
    """SCENARIO-REPORT-7953-AUTHORITY: required history differs from legitimate emptiness."""
    ledger = FailureLedger()
    task = {"id": "exp7955-new-unseen-xyz", "title": "New unseen xyz", "prior_failures": []}
    assert authority.lineage(task, ledger, [])["passed"]
    ledger.entries.append(
        LedgerEntry("exp1-new-unseen-xyz", "old", "complete_null_old", "finding", "new-unseen-xyz")
    )
    assert not authority.lineage(task, ledger, [])["passed"]
    task["prior_failures"] = [
        {
            "experiment_id": "exp1",
            "verdict": "complete_null_old",
            "addressed_by": "new target",
            "retire_if_same_verdict": True,
        }
    ]
    assert authority.lineage(task, ledger, [])["passed"]
    task["prior_failures"][0][key] = False if key == "retire_if_same_verdict" else ""
    assert not authority.lineage(task, ledger, [])["passed"]


def test_conflicting_snapshots_and_v689_failures(tmp_path):
    """SCENARIO-REPORT-7953-AUTHORITY: contradictory bytes and old failures stay visible."""
    paths = inputs(tmp_path)
    authority.assess(*paths)
    changed = yaml.safe_load(paths[2].read_bytes())
    changed["tasks"][0]["prompt"] += " changed"
    paths[1].write_text(yaml.safe_dump(changed))
    assert not authority.assess(*paths)["activated"]
    paths[1].unlink()
    raw = yaml.safe_dump(changed).encode()
    import hashlib

    saved = paths[3] / f"staged-{hashlib.sha256(raw).hexdigest()}.bin"
    saved.write_bytes(raw)
    assert not authority.assess(*paths)["activated"]
    saved.write_bytes(b"corrupted")
    assert not authority.assess(*paths)["activated"]
    paths[2].write_bytes(raw)
    assert not authority.assess(*(*paths[:3], tmp_path / "new"))["activated"]
    reproduced = methods.reproduce_v689(ROOT, tmp_path / "v689")
    assert all(reproduced.values())


def test_freeze_manifest_and_custody(tmp_path):
    """SCENARIO-REPORT-7953-VALIDATION: methods, current closure and historical dates bind."""
    freeze = methods.method_freeze(ROOT)
    assert len(freeze["training"]["arm_names"]) == 9
    assert freeze["response_targets"]["implicit_true"]
    assert freeze["confidence_formula"] == "abs(2*p-1)"
    assert methods.qualified_custody(ROOT)["ready"]
    manifest = validation.manifest(ROOT, tmp_path)
    assert manifest["coverage_includes"] == validation.OWNED
    commands = {r["name"]: r for r in manifest["commands"]}
    for name in ("affected_pytest", "coverage_unit", "scoped_spec_coverage"):
        assert commands[name]["argv"].count(validation.TESTS[0]) == 1
    for name in ("e2e_016_fixture", "e2e_016_replay"):
        argv = commands[name]["argv"]
        assert argv[argv.index("--date") + 1] == "20260929"
    outputs = [
        Path(r["argv"][r["argv"].index("--output") + 1]).parent
        for r in manifest["commands"]
        if "--output" in r["argv"]
    ]
    assert len(outputs) == len(set(outputs))


def test_candidate_reduction_and_owned_failure(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7953-VALIDATION: readiness reduces from owned primitive checks."""
    paths = inputs(tmp_path / "inputs")
    manifest = tmp_path / "manifest.json"
    atomic_json(manifest, validation.manifest(ROOT, tmp_path))
    args = (
        ROOT,
        authority.assess(*paths),
        methods.source_custody(SOURCE, methods.SOURCE_SHA256),
        methods.method_freeze(ROOT),
        [],
        [{"passed": True, "argv": ["true"]}],
        manifest,
        1,
        2,
    )
    value = methods.candidate(*args)
    assert value["contract_ready_score"] == 1 and value["verdict_class"] == "circular_positive"
    assert value["source_sample_size_budget"]["intended"] == 640
    for upstream in value["cited_upstream_artifacts"]:
        original = json.loads(Path(upstream["path"]).read_text())
        assert all(field in original for field in upstream["fields_imported"])
    output, raw = tmp_path / "result.json", tmp_path / "rows.json"
    atomic_json(output, value)
    atomic_json(raw, methods.primitive_rows(value))
    assert methods.cold_replay(output, raw)
    for field, observed in (
        ("experiment_id", 0),
        ("execution_date", "20260929"),
        ("contract_ready_score", 0),
        ("observed_activation", False),
        ("staging_custody_status", "contradictory"),
        ("planning_ready_score", 0),
        ("lineage_applicability_rows", []),
    ):
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
    assert methods.candidate(*args)["verdict_class"] == "blocked"
    assert not methods.cold_replay(tmp_path / "absent", raw)


def test_execution_publication_and_live_readers(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7953-PUBLICATION: newer nested sidecars cannot replace primaries."""
    paths = inputs(tmp_path / "inputs")
    args = SimpleNamespace(
        design=paths[0],
        staged=paths[1],
        active=paths[2],
        source=SOURCE,
        output=tmp_path / "success/experiment_7953_fixture.json",
        raw=tmp_path / "success/raw/experiment_7953_fixture/rows.json",
        fixture_e2e=False,
    )
    private = tmp_path / "private"
    private.mkdir()
    atomic_json(
        private / "coverage.json",
        {
            "files": {
                n: {"summary": {"num_statements": 1, "covered_lines": 1}, "missing_lines": []}
                for n in validation.OWNED
            }
        },
    )
    monkeypatch.setattr(
        validation,
        "manifest",
        lambda *args: {
            "commands": [
                {"name": "fixture_check", "argv": ["true"], "expected_exit": 0, "deadline_s": 1}
            ],
            "repository_health_current_receipt": {
                "name": "repository_full_suite",
                "argv": ["diagnostic_fixture"],
                "passed": False,
                "classification": "diagnostic",
            },
        },
    )
    monkeypatch.setattr(
        validation.shared, "run_check", lambda *args: {"passed": True, "argv": ["true"]}
    )
    value = validation.execute(ROOT, args, private)
    assert value["contract_ready_score"] == 1
    terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_text())
    assert terminal["primary_sha256"] == sha256_file(args.output)
    sidecar = args.raw.parent / "newer.json"
    atomic_json(sidecar, {**value, "contract_ready_score": 0})
    os.utime(sidecar, (2000000000, 2000000000))
    selected = validation.reader_receipt(
        value["task_id"], args.output.parent, field="contract_ready_score", expected=1
    )
    assert selected["gate_sha256"] == selected["document_sha256"] == sha256_file(args.output)
    assert methods.cold_replay(args.output, args.raw)
    atomic_json(args.output.parent / "experiment_7953_shadow.json", value)
    with pytest.raises(ValueError, match="conflicting_primary"):
        validation.publish(ROOT, value, args.output, args.raw, tmp_path / "conflict")
    (args.output.parent / "experiment_7953_shadow.json").unlink()
    monkeypatch.setattr(
        validation,
        "reader_receipt",
        lambda *args, **kwargs: {"gate_sha256": "wrong", "document_sha256": "wrong"},
    )
    with pytest.raises(ValueError, match="primary_reader_drift"):
        validation.publish(ROOT, value, args.output, args.raw, tmp_path / "drift")
    assert not validation.coverage_complete(tmp_path / "absent")


def test_cli_guards_and_terminal_routes(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7953-PUBLICATION: real routes are measured by the frozen manifest."""
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


def test_invalid_authority_does_not_repeat_history_scan(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7953-AUTHORITY: reject changed bytes before external reader work."""
    paths = inputs(tmp_path)
    value = yaml.safe_load(paths[2].read_bytes())
    value["tasks"][0]["prompt"] += " changed"
    paths[2].write_text(yaml.safe_dump(value))

    def forbidden(*args):
        raise AssertionError("unchanged history cannot rescue conflicting authority")

    monkeypatch.setattr(authority.FailureLedger, "load_from_artifacts", forbidden)
    assert not authority.assess(*paths)["observed_activation"]


def test_reader_cache_is_input_bound(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7953-VALIDATION: reuse only exact reader input and corpus bytes."""
    calls = []

    def loaded(*args):
        calls.append(1)
        return FailureLedger()

    monkeypatch.setattr(authority.FailureLedger, "load_from_artifacts", loaded)
    raw = yaml.safe_dump(
        {"tasks": [{"id": "exp7955-unique-qxyz", "title": "Unique qxyz", "prior_failures": []}]}
    ).encode()
    first = authority.reader_rows(raw, "private-corpus-a")
    before = len(calls)
    assert authority.reader_rows(raw, "private-corpus-a") == first
    assert len(calls) == before
    authority.reader_rows(raw, "private-corpus-b")
    assert len(calls) > before
    first[0]["passed"] = False
    assert authority.reader_rows(raw, "private-corpus-a")[0]["passed"]


def test_repository_health_is_separate(tmp_path):
    """SCENARIO-REPORT-7953-VALIDATION: one bounded health run never becomes an owned pass."""
    value = validation.manifest(ROOT, tmp_path, repository_health_receipt=tmp_path / "missing")
    assert value["repository_health_current_receipt"] is None
    assert not any(r["name"] == "repository_full_suite" for r in value["commands"])


def test_publication_custody_drift(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7953-VALIDATION: a mismatched exact primary cannot qualify reuse."""
    monkeypatch.setattr(methods, "PUBLICATION_SHA256", "sha256:changed")
    value = methods.qualified_custody(ROOT)
    assert not value["ready"]
    assert any(
        r["upstream_id"] == "exp7941-training-publication" for r in value["gate_check_summary"]
    )


def test_snapshot_replay_drift(tmp_path):
    """SCENARIO-REPORT-7953-PUBLICATION: saved bytes remain required even after consumption."""
    paths = inputs(tmp_path / "inputs")
    manifest = tmp_path / "manifest.json"
    atomic_json(manifest, validation.manifest(ROOT, tmp_path))
    value = methods.candidate(
        ROOT,
        authority.assess(*paths),
        methods.source_custody(SOURCE, methods.SOURCE_SHA256),
        methods.method_freeze(ROOT),
        [],
        [{"passed": True, "argv": ["true"]}],
        manifest,
        1,
        2,
    )
    output, raw = tmp_path / "result.json", tmp_path / "rows.json"
    atomic_json(output, value)
    atomic_json(raw, methods.primitive_rows(value))
    saved = Path(value["authority_snapshots"]["active"]["snapshot_path"])
    saved.write_bytes(b"changed preserved bytes")
    assert not methods.cold_replay(output, raw)


def test_empty_history_requires_a_list():
    """SCENARIO-REPORT-7953-AUTHORITY: malformed emptiness cannot stand in for a list."""
    ledger = FailureLedger()
    for history in (False, "", None):
        assert not authority.lineage(
            {"id": "exp7955-new-xyzq", "prior_failures": history}, ledger, []
        )["passed"]
