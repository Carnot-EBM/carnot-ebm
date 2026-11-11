"""REQ-REPORT-9013-V770 / REQ-VERIFY-9013: private custody and receipt controls."""

from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

from coverage import Coverage, CoverageData
import pytest
import yaml

from carnot.reporting import v770_branch_contract as c


def tree(root):
    """Private authority copies keep refusal controls away from operator evidence."""
    for name in [
        c.DESIGN,
        c.ACTIVE,
        "research-complete.yaml",
        "ops/conductor-log.md",
        *c.e.PINS,
        c.e.LEDGER,
        *c.e.operands(c.ROOT),
    ]:
        name = name if isinstance(name, str) else str(Path(name["path"]).relative_to(c.ROOT))
        source, target = c.ROOT / name, root / name
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    return root


def validation(work):
    """Synthetic receipts exercise refusal only; production uses measured children."""
    raw = Path(work["raw"])
    receipts = []
    for name in c.CHECKS:
        log = raw / (name + ".log")
        log.write_text("private receipt control\n")
        receipts.append(
            dict(
                name=name,
                passed=True,
                normal_exit=True,
                timed_out=False,
                exit_code=0,
                started_monotonic_ns=work["end_ns"],
                scope_sha256=c.canonical_hash(work["validation_scope_manifest"]),
                log_path=str(log),
                log_sha256=c.reference(log)["sha256"],
                reference=c.reference(log),
            )
        )
    return dict(receipts=receipts, coverage=dict(complete=True, percent=100), global_test_debt={})


def test_authority(tmp_path):
    """REQ-REPORT-9013-V770: both lifecycle modes bind all full objects and order."""
    root = tree(tmp_path / "root")
    actual = c.authority(root)
    assert actual["ready"] and len(actual["rows"]) == 14
    assert actual["rows"][0]["unit_id"] == c.TASK
    queue = yaml.safe_load((root / c.ACTIVE).read_bytes())
    (root / c.STAGED).write_text(yaml.safe_dump(queue))
    old = deepcopy(queue)
    old["milestone"] = "2026.11.769"
    (root / c.ACTIVE).write_text(yaml.safe_dump(old))
    assert c.authority(root)["comparison_mode"] == "staged"
    assert c.authority(root)["ready"]
    (root / c.STAGED).unlink()
    assert not c.authority(root)["ready"]
    assert not c.authority(tmp_path / "missing")["ready"]
    (root / c.ACTIVE).write_text(yaml.safe_dump(queue))
    assert all(r["passed"] for r in c.mutations(root, tmp_path / "controls"))


def test_freeze_build_and_receipt_controls(tmp_path):
    """SCENARIO-REPORT-9013-SCOPED: current receipts reject absent/stale/failure/timeout."""
    work = c.freeze(tree(tmp_path / "root"), tmp_path / "raw", "20261111")
    checked = validation(work)
    value = c.build(work, checked)
    assert value["verdict_class"] == "null" and value["contract_ready_score"] == 1
    assert value["frozen_graph_authority_ready_score"] == value["online_protocol_ready_score"] == 1
    assert value["arc_new_outcomes_ready_score"] == 0
    assert not value["repository_health_passed"]
    assert len(value["repository_health"]["rows"]) == 3
    assert all(
        r["original_verdict_class"] == "disqualified" for r in value["repository_health"]["rows"]
    )
    assert value["archive_lag"]["plan_time_archive_end"] == "2026.11.768"
    assert value["completed_count"] == value["intended_count"] == 14
    assert not any(value["model_invocation_counts"].values())
    for mode in ["missing", "failed", "timeout", "abnormal", "stale", "scope", "log", "coverage"]:
        changed = deepcopy(checked)
        row = changed["receipts"][0]
        if mode == "missing":
            changed["receipts"].pop()
        if mode == "failed":
            row["passed"] = False
        if mode == "timeout":
            row["timed_out"] = True
        if mode == "abnormal":
            row["normal_exit"] = False
        if mode == "stale":
            row["started_monotonic_ns"] = 0
        if mode == "scope":
            row["scope_sha256"] = "forged"
        if mode == "log":
            row["log_sha256"] = "forged"
        if mode == "coverage":
            changed["coverage"]["complete"] = False
        rejected = c.build(work, changed)
        assert rejected["verdict_class"] == "disqualified", mode
        assert all(rejected[k] == 0 for k in c.READINESS), mode
    independent = deepcopy(work)
    independent["science"]["frozen_graph_authority_ready_score"] = 0
    assert c.build(independent, checked)["contract_ready_score"] == 1
    with pytest.raises(FileExistsError):
        c.freeze(tmp_path, Path(work["raw"]), "20261111")
    blocked_work = c.freeze(tmp_path / "absent", tmp_path / "blocked", "20261111")
    blocked = c.build(blocked_work, validation(blocked_work))
    assert blocked["verdict_class"] == "blocked"
    assert any(r["observed"] == "missing_file" for r in blocked["gate_check_summary"])


def test_replay_refusals(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-9013-REPLAY: independent reductions reject repaired hashes."""
    work = c.freeze(tree(tmp_path / "root"), tmp_path / "raw", "20261111")
    checked = validation(work)
    value = c.build(work, checked)
    candidate = tmp_path / (c.NAME + ".json")
    c.atomic_json(candidate, value)
    assert not c.replay(candidate)["passed"]
    monkeypatch.setattr(c, "verify_measured", lambda *args: True)
    assert c.replay(candidate)["passed"]
    for mode in ["summary", "operand", "code", "contract", "control", "health"]:
        forged = deepcopy(value)
        if mode == "summary":
            forged["contract_ready_score"] = 99
        elif mode == "operand":
            forged["raw_shard_hashes"][0]["sha256"] = "sha256:forged"
        elif mode == "code":
            forged["code_config_hashes"][0]["source"]["sha256"] = "sha256:forged"
        else:
            changed = deepcopy(work)
            changed.pop("raw_reference")
            if mode == "contract":
                changed["contract"]["ready"] = False
            if mode == "control":
                changed["controls"][0]["passed"] = False
            if mode == "health":
                changed["repository_health"]["passed"] = True
            path = tmp_path / (mode + ".json")
            c.atomic_json(path, changed)
            forged["raw_shard_hashes"][0] = c.reference(path)
        forged["reproducibility_checksum"] = c.checksum(forged)
        c.atomic_json(candidate, forged)
        assert not c.replay(candidate)["passed"], mode
    c.atomic_json(Path(work["raw"]) / "validation.json", checked)
    value["raw_shard_hashes"].append(c.reference(Path(work["raw"]) / "validation.json"))
    value["reproducibility_checksum"] = c.checksum(value)
    c.atomic_json(candidate, value)
    assert c.replay(candidate)["passed"]
    assert not c.replay(tmp_path / "absent")["passed"]


def test_child_cli_and_runner_binding(tmp_path, monkeypatch):
    """REQ-VERIFY-9013 / E2E-018/021: real foreign-CWD CLI and bound publication."""
    config = tmp_path / "coverage.ini"
    config.write_text(
        "[run]\ninclude =\n"
        + "".join(f"    {c.ROOT / p}\n" for p in c.OWNED)
        + "[report]\nexclude_lines =\n"
    )
    for args, expected in [
        (["--date", "bad"], 2),
        (["--cold-replay", str(tmp_path / "absent")], 1),
    ]:
        print("before private CLI", args, flush=True)
        child = subprocess.run(
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--rcfile",
                str(config),
                "--data-file",
                str(tmp_path / ".coverage"),
                "--parallel-mode",
                str(c.ROOT / c.CLI),
                *args,
            ],
            cwd="/var/tmp",
            env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
            capture_output=True,
            text=True,
            timeout=60,
        )
        print("after private CLI", child.returncode, flush=True)
        assert child.returncode == expected, child.stdout + child.stderr
    tracer = Coverage.current()
    if tracer:
        for path in tmp_path.glob(".coverage.*"):
            data = CoverageData(basename=str(path))
            data.read()
            tracer.get_data().update(data)

    def checked(raw, tracer):
        tracer.stop()
        tracer.save()
        return validation(json.loads((raw / "work.json").read_bytes()))

    monkeypatch.setattr(c, "_validate", checked)
    monkeypatch.setattr(
        c,
        "run_check",
        lambda name, argv, raw, *args: validation(
            json.loads((raw.parent / "work.json").read_bytes())
        )["receipts"][-1],
    )
    monkeypatch.setattr(c, "_terminal", lambda candidate, raw: c.replay(candidate))
    monkeypatch.setattr(c, "verify_measured", lambda *args: True)
    output = tmp_path / (c.NAME + ".json")
    root = tree(tmp_path / "root")
    for args in [
        ["--date", "20261111", "--root", str(root), "--output", str(output)],
        ["--cold-replay", str(output)],
    ]:
        monkeypatch.setattr(sys, "argv", [c.CLI, *args])
        with pytest.raises(SystemExit) as error:
            runpy.run_path(str(c.ROOT / c.CLI), run_name="__main__")
        assert error.value.code == 0


def test_wrapper_and_optional_frontier(tmp_path, monkeypatch):
    """REQ-REPORT-9013-V770: exact scoped files and unchanged graph/ARC gates."""
    monkeypatch.setattr(
        c.old, "run_check", lambda name, argv, *args: dict(name=name, argv=argv, passed=True)
    )
    for name in ["owned_tests", "private_e2e018_021", "spec_coverage", "ruff_check"]:
        receipt = c.run_check(name, ["python", "-m", "pytest"], tmp_path)
        assert receipt["scope_sha256"] == c.canonical_hash(c.scope_manifest())
    assert c.e.science(tmp_path)["frozen_graph_authority_ready_score"] == 0
    assert not c.e.frontier(tmp_path)["ready"]
    original = c.e.load

    def changed(root, name):
        value = original(root, name)
        value["arc_frontier"]["frontier_after"]["ledger_receipt_identities"] = ["forged"]
        return value

    monkeypatch.setattr(c.e, "load", changed)
    assert not c.e.frontier(c.ROOT)["ready"]


def test_validation_receipt_binding(tmp_path, monkeypatch):
    """SCENARIO-REPORT-9013-SCOPED: bind the measured coverage report after tracing."""
    log = tmp_path / "coverage.json"
    log.write_text("{}")
    data = tmp_path / ".coverage"
    data.write_bytes(b"measured bytes after report auto-combination")
    original = dict(
        receipts=[dict(reference=c.reference(log))],
        coverage=dict(data_reference=dict(path=str(data), sha256="stale_before_report")),
    )
    monkeypatch.setattr(c, "_validate", lambda *args: original)
    monkeypatch.setattr(
        c,
        "run_check",
        lambda name, argv, raw, *args: dict(
            log_path=str(log),
            log_sha256=c.reference(log)["sha256"],
            scope_sha256=c.canonical_hash(c.scope_manifest()),
        ),
    )
    result = c.validate(tmp_path, None)
    row = result["receipts"][0]
    assert row["log_sha256"] == c.reference(log)["sha256"]
    assert row["scope_sha256"] == c.canonical_hash(c.scope_manifest())
    assert result["coverage"]["data_reference"] == c.reference(data)


def test_terminal_tamper_child(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-9013-REPLAY: real child refuses changed aggregates outside checkout."""
    candidate = tmp_path / "candidate.json"
    c.atomic_json(candidate, dict(contract_ready_score=1))
    monkeypatch.setattr(c, "_terminal", lambda *args: dict(passed=True, receipts=[]))
    result = c.terminal(candidate, tmp_path / "terminal")
    assert result["passed"] and result["receipts"][0]["actual_exit"] == 1
    assert (
        json.loads((tmp_path / "terminal/tampered.json").read_bytes())["contract_ready_score"] == 99
    )
