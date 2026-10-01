"""REQ-REPORT-7966-V691: preserve authority, date identity and private validation."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest
import yaml

from carnot.reporting import v691_contract_methods as methods
from carnot.reporting import v691_contract_validation as validation
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from scripts.experiments import experiment_7966_v691_contract_methods as cli

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/experiment_7892_v685_source_boundary.json"


def inputs(private):
    """Use separate private copies so tests never rewrite current authorities."""
    validation.prepare(ROOT, private)
    return tuple(private / n for n in ("design.md", "stage.yaml", "active.yaml", "snapshots"))


def test_authority_and_twelve_mutations(tmp_path):
    """SCENARIO-REPORT-7966-AUTHORITY: consumed staging does not erase activation."""
    paths = inputs(tmp_path)
    result = methods.assess(*paths)
    assert result["activated"] and len(result["contract_rows"]) == 13
    assert result["planning_ready_score"] == 0
    assert result["staging_custody_status"] == "unknown_consumed"
    assert all(r["passed"] for r in methods.mutations(paths[0], paths[2], SOURCE, tmp_path / "m"))
    paths[1].write_bytes(paths[2].read_bytes())
    assert methods.assess(*paths)["planning_ready_score"] == 1
    paths[2].unlink()
    result = methods.assess(*paths)
    assert not result["activated"] and result["planning_ready_score"] == 1
    paths[0].unlink()
    assert not methods.assess(*paths)["activated"]


def test_snapshot_conflict(tmp_path):
    """SCENARIO-REPORT-7966-AUTHORITY: preserved conflicting bytes fail."""
    paths = inputs(tmp_path)
    methods.assess(*paths)
    paths[3].joinpath("staged-conflict.bin").write_text("milestone: changed\ntasks: []\n")
    assert not methods.assess(*paths)["activated"]


@pytest.mark.parametrize(
    "field", ["run_date", "experiment_id", "task_id", "milestone", "execution_date"]
)
def test_issued_identity_rollover(tmp_path, field):
    """SCENARIO-REPORT-7966-DATES: own receipt admits rollover and rejects drift."""
    path = tmp_path / "producer.json"
    original = dict(
        experiment_id=7953,
        task_id="exp7953-contract-methods",
        milestone="2026.09.690",
        run_date="20260930",
        execution_date="20260930",
        current_run_id="exp7953-20260930",
    )
    atomic_json(path, original)
    receipt = methods.freeze_invocation(path)
    assert methods.validate_invocation(path, receipt)["passed"]
    changed = {**original, field: "changed"}
    atomic_json(path, changed)
    assert not methods.validate_invocation(path, receipt)["passed"]
    atomic_json(path, original)
    receipt["sha256"] = "sha256:stale"
    assert not methods.validate_invocation(path, receipt)["passed"]
    path.unlink()
    assert not methods.validate_invocation(path, receipt)["passed"]


def test_freeze_and_manifest(tmp_path):
    """SCENARIO-REPORT-7966-METHODS: full tasks bind losses, roles and branch gates."""
    tasks = yaml.safe_load((ROOT / "research-roadmap.yaml").read_bytes())["tasks"]
    freeze = methods.method_freeze(ROOT, tasks=tasks)
    assert sum(freeze["role_allocation"].values()) == 640
    assert freeze["task_contracts"] == tasks and not freeze["science_pre_gate"]
    assert len(freeze["literature_adoption_decisions"]) == 4
    assert freeze["review_access_limits"]["EBT"] == "HTTP 429"
    frozen = validation.manifest(ROOT, tmp_path)
    assert frozen["coverage_includes"] == validation.OWNED
    commands = {r["name"]: r for r in frozen["commands"]}
    assert "e2e_015" in commands and "e2e_018_private" in commands
    for name in ("e2e_016_fixture", "e2e_016_replay"):
        argv = commands[name]["argv"]
        assert argv[argv.index("--date") + 1] == "20260929"
    outputs = [
        r["argv"][r["argv"].index("--output") + 1]
        for r in frozen["commands"]
        if "--output" in r["argv"]
    ]
    assert len(outputs) == len(set(outputs))
    assert not validation.coverage_complete(tmp_path / "absent.json")


def build(private):
    """Build from real custody but keep every output in private scratch."""
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


def test_candidate_replay_and_failures(tmp_path):
    """SCENARIO-REPORT-7966-VALIDATION: readiness follows rows and owned checks."""
    value, args = build(tmp_path)
    assert value["contract_ready_score"] == 1 and value["run_date"] == "20261001"
    assert value["verdict_class"] == "circular_positive"
    output, raw = tmp_path / "candidate.json", tmp_path / "rows.json"
    atomic_json(output, value)
    atomic_json(raw, methods.primitive_rows(value))
    assert methods.cold_replay(output, raw)
    for key, changed in (
        ("experiment_id", 0),
        ("run_date", "20260930"),
        ("planning_ready_score", 1),
        ("producer_date_rows", []),
        ("rows", []),
        ("contract_ready_score", 0),
    ):
        atomic_json(output, {**value, key: changed})
        assert not methods.cold_replay(output, raw)
    assert not methods.cold_replay(tmp_path / "absent", raw)
    atomic_json(output, value)
    snapshot = Path(value["authority_snapshots"]["active"]["snapshot_path"])
    saved = snapshot.read_bytes()
    snapshot.unlink()
    assert not methods.cold_replay(output, raw)
    snapshot.write_bytes(saved)
    failed = methods.candidate(*(*args[:5], [{"passed": False, "argv": ["false"]}], *args[6:]))
    assert failed["verdict_class"] == "disqualified" and not failed["contract_ready_score"]
    missing = deepcopy(args[2])
    missing["ready"] = False
    blocked = methods.candidate(*(*args[:2], missing, *args[3:]))
    assert blocked["verdict_class"] == "blocked" and not blocked["contract_ready_score"]


def test_upstream_custody_failure(tmp_path):
    """SCENARIO-REPORT-7966-DATES: exact primary/hash remains the reuse boundary."""
    value, args = build(tmp_path)
    manifest = json.loads(args[6].read_text())
    manifest["producer_invocations"][0]["sha256"] = "sha256:stale"
    atomic_json(args[6], manifest)
    blocked = methods.candidate(*args)
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"] and not blocked["contract_ready_score"]
    dates = {r["identity"]["run_date"] for r in value["producer_date_rows"]}
    assert dates == {"20260930", "20261001"}


def test_cli_guards_and_replay(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7966-VALIDATION: parameterized CLI protects live paths."""
    for argv in (
        ["--fixture-e2e"],
        ["--terminal-recheck", str(tmp_path / "missing")],
        ["--date", "20260930"],
    ):
        monkeypatch.setattr("sys.argv", ["experiment_7966", *argv])
        with pytest.raises(SystemExit):
            cli.main()
    monkeypatch.setattr("sys.argv", ["experiment_7966", "--cold-replay", str(tmp_path / "missing")])
    assert cli.main() == 1


def test_execute_archives_and_publication(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7966-VALIDATION: failed coverage is owned disqualification."""
    paths = inputs(tmp_path / "inputs")
    original = validation.manifest

    def limited(root, private):
        value = original(root, private)
        value["commands"] = []
        atomic_json(private / "coverage.json", {"files": {}})
        (private / ".coverage.combined").write_bytes(b"private failed coverage fixture")
        return value

    monkeypatch.setattr(validation, "manifest", limited)
    args = SimpleNamespace(
        design=paths[0],
        staged=paths[1],
        active=paths[2],
        source=SOURCE,
        fixture_e2e=False,
        output=tmp_path / "published/experiment_7966_fixture.json",
        raw=tmp_path / "published/raw/rows.json",
    )
    value = validation.execute(ROOT, args, tmp_path / "work")
    assert value["verdict_class"] == "disqualified" and not value["contract_ready_score"]
    assert methods.cold_replay(args.output, args.raw)
    receipt = json.loads((args.raw.parent / "primary_resolution_receipt.json").read_text())
    assert receipt["gate_sha256"] == receipt["document_sha256"] == sha256_file(args.output)


def test_reader_drift_rejected(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7966-VALIDATION: final consumers must select checked bytes."""
    value, _ = build(tmp_path / "inputs")
    monkeypatch.setattr(
        validation,
        "reader_receipt",
        lambda *a, **kw: dict(gate_sha256="wrong", document_sha256="wrong"),
    )
    with pytest.raises(ValueError, match="primary_reader_drift"):
        validation.publish(
            ROOT,
            value,
            tmp_path / "published/experiment_7966_fixture.json",
            tmp_path / "published/raw/rows.json",
            tmp_path / "scratch",
        )
