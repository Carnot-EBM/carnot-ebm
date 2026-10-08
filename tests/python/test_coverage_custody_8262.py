"""REQ-REPORT-8262 / REQ-VERIFY-8262: private evidence must outlive its producer."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from carnot.reporting import coverage_custody_8262 as custody
from carnot.reporting import v714_coverage_custody as q
from carnot.reporting import v714_coverage_runner as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.v709_execution import child


@pytest.fixture(autouse=True)
def isolate_diagnostic_environment(monkeypatch):
    """REQ-VERIFY-8262: private cases cannot inherit the producer's diagnostic receipt."""
    monkeypatch.delenv("CARNOT8262_HEALTH_RECEIPT", raising=False)


def measured(tmp_path):
    """Use an actual coverage child rather than a report inferred from console text."""
    root = tmp_path / "root"
    root.mkdir()
    code = root / "owned.py"
    code.write_text('"""Private measured code."""\nx = 1\nprint(x)\n')
    report = tmp_path / "coverage.json"
    spec = dict(
        name="coverage_json",
        argv=[
            str(q.ROOT / ".venv/bin/coverage"),
            "json",
            "--data-file=" + str(tmp_path / ".coverage"),
            "--include=" + str(code),
            "-o",
            str(report),
        ],
    )
    child(
        "measure",
        [
            str(q.ROOT / ".venv/bin/coverage"),
            "run",
            "--data-file=" + str(tmp_path / ".coverage"),
            str(code),
        ],
        tmp_path / "logs",
    )
    receipt = child("coverage_json", spec["argv"], tmp_path / "logs")
    return root, report, spec, receipt


def test_durable_measured_report(tmp_path):
    """SCENARIO-VERIFY-8262-CUSTODY: measured totals survive scratch removal."""
    root, report, spec, receipt = measured(tmp_path)
    binding = custody.preserve(root, spec, receipt, ["owned.py"], tmp_path / "durable")
    report.unlink()
    assert custody.replay(binding)["owned.py"]["num_statements"] == 2
    assert sha256_file(Path(binding["receipt_path"])) == binding["receipt_sha256"]
    assert binding["source_report_path"] == str(report)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "stale",
        "foreign",
        "partial",
        "excluded",
        "failed",
        "zero",
        "absent_counts",
        "tampered",
        "bad_operand",
    ],
)
def test_invalid_reports(tmp_path, mutation):
    """SCENARIO-VERIFY-8262-CUSTODY: each failed operand stays distinguishable."""
    root, report, spec, receipt = measured(tmp_path)
    value = json.loads(report.read_bytes())
    entry = next(iter(value["files"].values()))
    if mutation == "missing":
        report.unlink()
    elif mutation == "stale":
        os.utime(report, ns=(1, 1))
    elif mutation == "foreign":
        value["files"] = {"wrong.py": entry}
    elif mutation == "partial":
        entry["missing_lines"] = entry["executed_lines"][-1:]
        entry["executed_lines"] = entry["executed_lines"][:-1]
        entry["summary"]["covered_lines"] -= 1
    elif mutation == "excluded":
        entry["excluded_lines"] = [2]
        entry["summary"]["excluded_lines"] = 1
    elif mutation == "failed":
        receipt.update(exit_code=1, actual_exit=1, passed=False)
    elif mutation == "zero":
        entry["summary"]["num_statements"] = 0
    elif mutation == "absent_counts":
        entry.pop("summary")
    elif mutation == "tampered":
        entry["summary"]["covered_lines"] += 1
    else:
        spec["argv"] = [sys.executable, "-c", "pass"]
    if mutation not in {"missing", "stale"}:
        stamp = report.stat().st_mtime_ns
        atomic_json(report, value)
        os.utime(report, ns=(stamp, stamp))
    with pytest.raises((ValueError, OSError, KeyError)):
        custody.preserve(root, spec, receipt, ["owned.py"], tmp_path / "durable")


def test_rehashed_primitive_rejected(tmp_path):
    """REQ-VERIFY-8262: a new envelope hash cannot repair contradictory statements."""
    root, report, spec, receipt = measured(tmp_path)
    binding = custody.preserve(root, spec, receipt, ["owned.py"], tmp_path / "durable")
    durable = Path(binding["report_path"])
    value = json.loads(durable.read_bytes())
    next(iter(value["files"].values()))["summary"]["num_statements"] += 1
    atomic_json(durable, value)
    binding["report_sha256"] = sha256_file(durable)
    saved = json.loads(Path(binding["receipt_path"]).read_bytes())
    saved["report_sha256"] = binding["report_sha256"]
    atomic_json(Path(binding["receipt_path"]), saved)
    binding["receipt_sha256"] = sha256_file(Path(binding["receipt_path"]))
    with pytest.raises(ValueError):
        custody.replay(binding)


def authorities(tmp_path):
    """Copy only authority inputs; planning and execution read independent files."""
    root = tmp_path / "authority"
    root.mkdir()
    for name in [q.DESIGN, q.ACTIVE, q.HISTORY, q.PROTOCOL]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((q.ROOT / name).read_bytes())
    return root


def test_authority_and_science_binding(tmp_path):
    """SCENARIO-REPORT-8262-AUTHORITY: full prompts and science bytes govern readiness."""
    root = authorities(tmp_path)
    work = q.authority_work(root, tmp_path / "raw")
    assert work["contract"]["activated"] and not work["contract"]["planning_matched"]
    assert len(work["tasks"]) == 14
    assert work["execution_contract"]["science_protocol_sha256"] == q.PIN
    assert len(work["execution_contract"]["producers"]) == 14
    active = root / q.ACTIVE
    value = yaml.safe_load(active.read_bytes())
    value["tasks"][0]["prompt"] += " changed"
    active.write_text(yaml.safe_dump(value))
    failed = q.authority_work(root, tmp_path / "changed")
    assert not failed["contract"]["activated"]
    active.unlink()
    assert not q.authority_work(root, tmp_path / "absent")["contract"]["activated"]
    (root / q.PROTOCOL).write_text("{}")
    assert q.authority_work(root, tmp_path / "science")["failures"]
    (root / q.DESIGN).write_text("missing contract")
    assert q.authority_work(root, tmp_path / "bad")["failures"]


def private_plan(private, candidate):
    """Private scripted children exercise the production orchestration branch."""
    code = private / "owned.py"
    code.write_text('"""REQ-VERIFY-8262 private child."""\nx = 1\nassert x == 1\n')
    coverage = str(q.ROOT / ".venv/bin/coverage")
    return dict(
        commands=[
            dict(
                name="coverage_custody_tests",
                argv=[coverage, "run", "--data-file=" + str(private / ".coverage"), str(code)],
                expected=0,
                deadline=30,
                scope="owned",
            ),
            dict(
                name="current_contract_tests",
                argv=[
                    sys.executable,
                    "-c",
                    "from carnot.reporting import v714_coverage_custody as q; "
                    "from pathlib import Path; import sys; "
                    "assert q.authority_work(q.ROOT, Path(sys.argv[1]))['contract']['activated']",
                    str(private / "authority"),
                ],
                expected=0,
                deadline=30,
                scope="owned",
            ),
            dict(
                name="coverage_json",
                argv=[
                    coverage,
                    "json",
                    "--data-file=" + str(private / ".coverage"),
                    "--include=" + str(code),
                    "-o",
                    str(private / "explicit-report.json"),
                ],
                expected=0,
                deadline=30,
                scope="owned",
            ),
        ],
        owned=[str(code)],
    )


def test_real_orchestration_and_cold_cli(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8262-CUSTODY: no fixture bypass can set readiness."""
    output = tmp_path / (q.NAME + ".json")
    monkeypatch.setattr(runner, "manifest", private_plan)
    value = runner.run(q.ROOT, output)
    assert value["coverage_custody_ready_score"] == value["current_contract_ready_score"] == 1
    assert value["scratch_removed"] and value["owned_statement_counts"]
    assert value["generalized_learning_benefit_score"] == 0
    assert len(value["historical_dispositions"]) == 14
    assert all(r["passed"] for r in value["cold_replay_rows"])
    command = [sys.executable, str(q.ROOT / q.CLI), "--cold-replay", str(output)]
    done = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr
    for key in [
        "current_contract_ready_score",
        "coverage_custody_ready_score",
        "rows",
        "historical_dispositions",
        "owned_statement_counts",
        "MODEL_SPECS",
    ]:
        changed = deepcopy(value)
        changed[key] = (
            ["foreign"] if key == "MODEL_SPECS" else [] if isinstance(changed[key], list) else 99
        )
        changed.pop("reproducibility_checksum")
        changed["reproducibility_checksum"] = canonical_hash(changed)
        atomic_json(output, changed)
        assert subprocess.run(command, capture_output=True, timeout=60).returncode == 1
    atomic_json(output, value)
    bad = deepcopy(value)
    bad["reproducibility_checksum"] = "wrong"
    atomic_json(output, bad)
    with pytest.raises(ValueError, match="checksum"):
        runner.replay(output)
    atomic_json(output, value)
    original_assess = runner.assess_authorities
    monkeypatch.setattr(
        runner,
        "assess_authorities",
        lambda *a, **k: dict(original_assess(*a, **k), activated=False),
    )
    with pytest.raises(ValueError, match="authority_reduction"):
        runner.replay(output)
    monkeypatch.setattr(runner, "assess_authorities", original_assess)
    logged = Path(value["validation_receipts"][0]["stdout_path"])
    original_log = logged.read_bytes()
    logged.write_text("changed")
    with pytest.raises(ValueError, match="validation_stream"):
        runner.replay(output)
    logged.write_bytes(original_log)
    work_path = Path(value["work_reference"]["path"])
    work_bytes = work_path.read_bytes()
    for mutation in ["history", "tasks"]:
        work = json.loads(work_bytes)
        changed = deepcopy(value)
        if mutation == "history":
            work["history"][0]["honest_verdict"] = "forged"
            changed["historical_dispositions"] = work["history"]
        else:
            work["tasks"][0]["prompt"] += " forged"
            changed["task_contract"] = work["tasks"]
        atomic_json(work_path, work)
        changed["work_reference"]["sha256"] = sha256_file(work_path)
        changed.pop("reproducibility_checksum")
        changed["reproducibility_checksum"] = canonical_hash(changed)
        atomic_json(output, changed)
        with pytest.raises(ValueError, match="primitive_drift"):
            runner.replay(output)
    work_path.write_bytes(work_bytes)
    atomic_json(output, value)
    assert runner.main(["--cold-replay", str(output)]) == 0
    assert runner.main(["--cold-replay", str(tmp_path / "absent")]) == 1
    with pytest.raises(SystemExit):
        runner.main(["--date", "invalid"])


def test_orchestration_failed_child(tmp_path, monkeypatch):
    """REQ-VERIFY-8262: actual failed children disqualify and preserve zero readiness."""

    def failed(private, candidate):
        plan = private_plan(private, candidate)
        plan["commands"][0]["argv"] = [sys.executable, "-c", "raise SystemExit(3)"]
        return plan

    monkeypatch.setattr(runner, "manifest", failed)
    value = runner.run(q.ROOT, tmp_path / (q.NAME + ".json"))
    assert value["verdict_class"] == "disqualified"
    assert value["coverage_custody_ready_score"] == value["current_contract_ready_score"] == 0
    assert not value["required_checks_passed"]


def test_reduction_precedence(tmp_path):
    """REQ-REPORT-8262: external failure is blocked while owned failure disqualifies."""
    work = q.measure(q.ROOT, tmp_path / "raw")
    receipts = [
        dict(name=name, passed=True, exit_code=0)
        for name in ["coverage_custody_tests", "current_contract_tests"]
    ]
    result = q.reduce(work, receipts, True, True)
    assert result["current_contract_ready_score"] == result["coverage_custody_ready_score"] == 1
    work["failures"].append(q.failure(tmp_path / "missing", "exists", True, None))
    assert q.reduce(work, receipts, True, True)["verdict_class"] == "blocked"
    assert q.reduce(work, [], True, True)["verdict_class"] == "blocked"


@pytest.mark.parametrize(
    "mutation", ["code", "hash", "binding", "exit", "stream", "summary", "totals", "timestamp"]
)
def test_durable_provenance_negatives(tmp_path, mutation):
    """REQ-VERIFY-8262: recomputed primitive and command identity resist new hashes."""
    root, report, spec, receipt = measured(tmp_path)
    binding = custody.preserve(root, spec, receipt, ["owned.py"], tmp_path / "durable")
    saved = json.loads(Path(binding["receipt_path"]).read_bytes())
    if mutation == "code":
        Path(binding["owned_files"][0]["snapshot_path"]).write_text("x = 2\n")
    elif mutation == "hash":
        binding["report_sha256"] = "wrong"
    elif mutation == "binding":
        binding["source_mtime_ns"] += 1
    elif mutation == "exit":
        saved["command_receipt"]["exit_code"] = 1
    elif mutation == "stream":
        Path(receipt["stdout_path"]).write_text("changed")
    elif mutation == "summary":
        binding["owned_statement_counts"]["owned.py"]["num_statements"] += 1
        saved["owned_statement_counts"] = binding["owned_statement_counts"]
    elif mutation == "timestamp":
        source = json.loads(report.read_bytes())
        source["meta"]["timestamp"] = "2000-01-01T00:00:00"
        atomic_json(report, source)
        with pytest.raises(ValueError, match="freshness"):
            custody.preserve(root, spec, receipt, ["owned.py"], tmp_path / "stale")
        return
    else:
        source = json.loads(Path(binding["report_path"]).read_bytes())
        source["totals"]["num_statements"] += 1
        atomic_json(Path(binding["report_path"]), source)
        binding["report_sha256"] = sha256_file(Path(binding["report_path"]))
        saved["report_sha256"] = binding["report_sha256"]
    atomic_json(Path(binding["receipt_path"]), saved)
    binding["receipt_sha256"] = sha256_file(Path(binding["receipt_path"]))
    with pytest.raises(ValueError):
        custody.replay(binding)


def test_authority_negative_controls(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8262-AUTHORITY: every full task field and design digest matter."""
    root = authorities(tmp_path)
    original = (root / q.ACTIVE).read_bytes()
    for field in ["prior_failures", "gated_on", "MODEL_SPECS", "title"]:
        value = yaml.safe_load(original)
        value["tasks"][0][field] = []
        if field == "gated_on":
            value["tasks"][0][field] = [
                dict(upstream="wrong", artifact_field="ready", op="==", value=1)
            ]
        (root / q.ACTIVE).write_text(yaml.safe_dump(value))
        if field == "MODEL_SPECS":
            value["tasks"][0][field] = [dict(name="wrong")]
            (root / q.ACTIVE).write_text(yaml.safe_dump(value))
        assert not q.authority_work(root, tmp_path / field)["contract"]["activated"]
    (root / q.ACTIVE).write_bytes(original)
    text = (root / q.DESIGN).read_text()
    (root / q.DESIGN).write_text(
        text.replace("Work in {project_root}", "Changed in {project_root}", 1)
    )
    assert q.authority_work(root, tmp_path / "digest")["failures"]
    work = q.measure(root, tmp_path / "external-missing")
    assert work["failures"]
    monkeypatch.setattr(q, "read_bound_sidecar", lambda *a: {"report": {"passed": False}})
    work = q.measure(q.ROOT, tmp_path / "unqualified")
    assert any(r["artifact_field"] == "qualified_terminal" for r in work["failures"])
    assert any(r["artifact_field"] == "historical_terminal_binding" for r in work["failures"])
    original_snapshot = q.snapshot

    def absent(path, raw, role):
        ref = original_snapshot(path, raw, role)
        return dict(ref, exists=False) if path.name == "exclusion_manifest.yaml" else ref

    monkeypatch.setattr(q, "snapshot", absent)
    assert any(
        r["artifact_field"] == "exists"
        for r in q.measure(q.ROOT, tmp_path / "missing-exclusion")["failures"]
    )


def test_manifest_and_rejected_terminal(tmp_path, monkeypatch):
    """REQ-VERIFY-8262: private orchestration retains actual terminal failures and health."""
    private = tmp_path / "plan"
    private.mkdir()
    plan = runner.manifest(private, tmp_path / "candidate")
    assert "-o" in next(s for s in plan["commands"] if s["name"] == "coverage_json")["argv"]

    def bounded(private, candidate):
        specs = private_plan(private, candidate)
        specs["repository_health"] = dict(
            name="health",
            argv=[sys.executable, "-c", "raise SystemExit(7)"],
            expected=0,
            deadline=30,
            scope="diagnostic",
        )
        return specs

    monkeypatch.setattr(runner, "manifest", bounded)
    original_terminal = runner.terminal
    reports = []

    def reject_once(candidate, raw):
        if not reports:
            reports.append(True)
            receipt = child(
                "terminal_failure", [sys.executable, "-c", "raise SystemExit(8)"], raw / "negative"
            )
            return dict(passed=False, checks=[receipt])
        return original_terminal(candidate, raw)

    monkeypatch.setattr(runner, "terminal", reject_once)
    output = tmp_path / (q.NAME + ".json")
    original_relative = Path.is_relative_to
    monkeypatch.setattr(
        Path, "is_relative_to", lambda self, other: self == output or original_relative(self, other)
    )
    original_atomic = runner.atomic_json

    def safe_export(path, value):
        return original_atomic(
            tmp_path / "exported_contract.json" if path == q.ROOT / q.EXECUTION else path, value
        )

    monkeypatch.setattr(runner, "atomic_json", safe_export)
    value = runner.run(q.ROOT, output)
    assert value["repository_health"]["exit_code"] == 7
    assert value["verdict_class"] == "disqualified" and value["coverage_custody_ready_score"] == 0
    assert any(r["exit_code"] == 8 for r in value["validation_receipts"])
    original_run = runner.run
    monkeypatch.setattr(runner, "run", lambda *a: value)
    assert runner.main(["--output", str(output)]) == 0

    def conflict(*args, **kwargs):
        raise ValueError("conflicting_primary")

    monkeypatch.setattr(runner, "publish_primary", conflict)
    with pytest.raises(ValueError, match="conflicting_primary"):
        original_run(q.ROOT, tmp_path / "conflict" / (q.NAME + ".json"))


@pytest.mark.parametrize("mutation", ["missing", "stale", "foreign", "partial", "rehashed_tamper"])
def test_real_orchestration_invalid_coverage(tmp_path, monkeypatch, mutation):
    """SCENARIO-VERIFY-8262-CUSTODY: real scripted children cannot bypass custody gates."""

    def mutated(private, candidate):
        specs = private_plan(private, candidate)
        original = specs["commands"][-1]["argv"]
        source = original[-1]
        script = (
            "import json,subprocess,sys; from pathlib import Path; "
            f"subprocess.run({original!r}, check=True); "
            f"p=Path({source!r}); v=json.loads(p.read_bytes()); "
            "e=next(iter(v['files'].values())); "
        )
        changes = {
            "missing": "p.unlink()",
            "stale": "v['meta']['timestamp']='2000-01-01T00:00:00'; p.write_text(json.dumps(v))",
            "foreign": "v['files']={'foreign.py':e}; p.write_text(json.dumps(v))",
            "partial": "e['missing_lines']=e['executed_lines'][-1:]; e['executed_lines']=e['executed_lines'][:-1]; p.write_text(json.dumps(v))",
            "rehashed_tamper": "e['summary']['covered_lines']+=1; p.write_text(json.dumps(v))",
        }
        specs["commands"][-1]["argv"] = [
            sys.executable,
            "-c",
            script + changes[mutation],
            "json",
            "-o",
            source,
        ]
        return specs

    monkeypatch.setattr(runner, "manifest", mutated)
    value = runner.run(q.ROOT, tmp_path / (q.NAME + ".json"))
    assert value["verdict_class"] == "disqualified"
    assert value["current_contract_ready_score"] == value["coverage_custody_ready_score"] == 0
    assert any(
        r["artifact_field"] == "durable_measured_coverage" for r in value["gate_check_summary"]
    )


def test_missing_tool_blocks_and_health_receipt(tmp_path, monkeypatch):
    """REQ-REPORT-8262: missing tools block; imported diagnostic bytes never imply global health."""
    monkeypatch.setattr(runner, "manifest", private_plan)
    original_file = Path.is_file
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda self: False if self == q.ROOT / ".venv/bin/ruff" else original_file(self),
    )
    receipt = child(
        "private_health_control", [sys.executable, "-c", "raise SystemExit(7)"], tmp_path / "health"
    )
    # This private negative receipt exercises import shape; it never claims a full-suite pass.
    receipt["argv"] = [str(q.ROOT / ".venv/bin/pytest"), "tests/python", "-q"]
    receipt["scope"] = "private_negative_shape_control"
    path = tmp_path / "health_receipt.json"
    atomic_json(path, receipt)
    monkeypatch.setenv("CARNOT8262_HEALTH_RECEIPT", str(path))
    value = runner.run(q.ROOT, tmp_path / (q.NAME + ".json"))
    assert value["verdict_class"] == "blocked"
    assert value["repository_health"]["passed"] is False
    assert value["coverage_custody_ready_score"] == value["current_contract_ready_score"] == 0
    receipt["argv"] = ["foreign"]
    atomic_json(path, receipt)
    with pytest.raises(ValueError, match="health_argv"):
        runner.run(q.ROOT, tmp_path / "invalid" / (q.NAME + ".json"))
    receipt["argv"] = [str(q.ROOT / ".venv/bin/pytest"), "tests/python", "-q"]
    receipt["stdout_sha256"] = "wrong"
    atomic_json(path, receipt)
    with pytest.raises(ValueError, match="health_stream"):
        runner.run(q.ROOT, tmp_path / "stream" / (q.NAME + ".json"))
