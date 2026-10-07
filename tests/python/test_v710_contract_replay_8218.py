"""REQ-REPORT-8218 / REQ-VERIFY-8218: immutable replay and actual CLI controls."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from carnot.reporting import v710_contract_replay as q
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.roadmap_contract import parse_design


def authority(tmp_path):
    """Private full authority permits controls without activating the checkout."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    d = tmp_path / "design.md"
    _, original = parse_design((q.ROOT / q.HISTORY).read_text(), milestone="2026.10.709")
    tasks = []
    for index in range(14):
        task = deepcopy(original[index % 13])
        task.update(
            id=f"exp{8218 + index}-private-fixture",
            milestone=q.MILESTONE,
            deliverable=f"results/experiment_{8218 + index}_fixture.json",
            gated_on=[],
        )
        tasks.append(task)
    tasks[1]["gated_on"] = [
        dict(upstream=tasks[0]["id"], artifact_field="fixture_ready", op="==", value=1)
    ]
    table = "\n".join(
        f"| {i + 1} | {t['id']} | {t['title']} | {t['phase']} | {t['deliverable']} |"
        for i, t in enumerate(tasks)
    )
    from carnot.reporting.v685_authority_lifecycle import tasks_digest

    d.write_text(
        "## Exact task contract\n"
        + table
        + "\nCanonical full-task SHA256: `"
        + tasks_digest(tasks)
        + "`\n"
        + "<!-- V710_TASK_CONTRACT_START -->\n```json\n"
        + json.dumps(dict(milestone=q.MILESTONE, tasks=tasks))
        + "\n```\n"
    )
    a = tmp_path / "active.yaml"
    a.write_text(yaml.safe_dump(dict(milestone=q.MILESTONE, tasks=tasks)))
    return d, tmp_path / "absent", a


def test_contract_and_mutations(tmp_path):
    """SCENARIO-REPORT-8218-CONTRACT: every independent authority operand matters."""
    d, s, a = authority(tmp_path)
    v = q.assess(d, s, a, tmp_path / "snapshots")
    assert v["activated"] and len(v["contract_rows"]) == 14
    assert not v["planning_matched"]
    controls = q.mutations(d, a, tmp_path / "mutations")
    assert {r["control"] for r in controls} == {
        "count",
        "order",
        "title",
        "prompt",
        "gate",
        "model",
        "deliverable",
        "digest",
        "table",
    }
    assert all(r["rejected"] for r in controls)
    staged = tmp_path / "staged.yaml"
    staged.write_bytes(a.read_bytes())
    v = q.assess(d, staged, tmp_path / "missing", tmp_path / "staged-only")
    assert v["planning_matched"] and not v["activated"]


def test_snapshots_and_missing_bytes(tmp_path):
    """SCENARIO-REPORT-8218-HISTORY: originals are qualified independently of live code."""
    p = tmp_path / "original"
    p.write_bytes(b"original bytes")
    ref = q.snapshot(p, tmp_path / "custody", "original")
    p.write_bytes(b"mutable later bytes")
    assert q.verify_reference(ref)
    Path(ref["snapshot_path"]).write_bytes(b"tampered")
    assert not q.verify_reference(ref)
    missing = q.snapshot(tmp_path / "absent", tmp_path / "custody", "missing")
    assert missing["exists"] is False and missing["sha256"] is None
    assert q.verify_reference(missing)
    with pytest.raises(ValueError):
        q.require_reference(ref)


def test_reduce_failure_precedence(tmp_path):
    """REQ-REPORT-8218: owned failure zeros readiness even with external blocks."""
    d, s, a = authority(tmp_path)
    work = dict(
        contract=q.assess(d, s, a, tmp_path / "authority"),
        failures=[],
        historical_dispositions=[],
        replay_controls=[],
        immutable_code_snapshots=[],
        h1_custody={},
        preconditions_checked=[],
        source_artifact_hashes=[],
    )
    receipt = dict(passed=True, exit_code=0, timed_out=False)
    good = q.reduce(work, [receipt])
    assert good["verdict_class"] == "circular_positive"
    assert good["contract_ready_score"] == 1
    assert good["generalized_learning_benefit_score"] == 0
    work["failures"] = [q.failure(tmp_path / "absent", "original_bytes", True, None)]
    blocked = q.reduce(work, [receipt])
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"][0]["observed"] is None
    bad = q.reduce(work, [dict(receipt, passed=False)])
    assert bad["verdict_class"] == "disqualified"
    assert bad["historical_replay_ready_score"] == bad["contract_ready_score"] == 0


def cli(tmp_path, *args):
    """A fresh real process tests script imports and literal terminal exits."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, str(q.ROOT / q.CLI), *map(str, args)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_real_cli_block_and_tamper(tmp_path):
    """SCENARIO-REPORT-8218-CLI: unavailable evidence publishes an honest blocked result."""
    root = tmp_path / "root"
    root.mkdir()
    output = tmp_path / (q.NAME + ".json")
    run = cli(tmp_path, "--root", root, "--output", output, "--private-fixture")
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["intended_count"] == 14
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    for field in ["rows", "contract_ready_score", "historical_dispositions"]:
        changed = deepcopy(value)
        changed[field] = [{"tampered": True}] if isinstance(value[field], list) else 999
        changed.pop("reproducibility_checksum")
        changed["reproducibility_checksum"] = canonical_hash(changed)
        atomic_json(output, changed)
        assert cli(tmp_path, "--cold-replay", output).returncode == 1
    atomic_json(output, value)
    ref = value["work_reference"]
    Path(ref["path"]).write_text("{}")
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    assert cli(tmp_path, "--date", "20261006").returncode == 2
    assert cli(tmp_path, "--private-fixture").returncode == 1


def test_manifest(tmp_path):
    """SCENARIO-VERIFY-8218-REPLAY: frozen ownership includes real CLI statements."""
    plan = q.commands(tmp_path)
    text = json.dumps(plan)
    assert q.CLI in text and "--strict" in text and "--fail-under=100" in text
    assert any(p["scope"] == "repository_health" for p in plan)
    assert "test_restricted_decision_audit_8210.py" in text


def test_original_primitive_replay(tmp_path):
    """SCENARIO-VERIFY-8218-REPLAY: repaired CLI authenticates copied original H1 operands."""
    from carnot.reporting import v710_replay_history as h

    work = dict(
        source_artifact_hashes=[],
        preconditions_checked=[],
        failures=[],
        immutable_code_snapshots=[],
        replay_controls=[],
    )
    source = json.loads(
        (q.ROOT / "results/experiment_8210_v709_restricted_decision_audit.json").read_bytes()
    )
    h.audit_copy(source, tmp_path, work)
    assert work["h1_custody"]["agreement"]
    assert work["h1_custody"]["completed_count"] == 97
    assert not work["h1_custody"]["H1"]["passed"]
    assert all(r["passed"] for r in work["replay_controls"]), work["replay_controls"]
    assert source["verdict_class"] == "disqualified"
    assert not work["h1_custody"]["scientific_gate"]


def test_original_history_and_unavailable_authority(original_work):
    """SCENARIO-REPORT-8218-HISTORY: all thirteen immutable dispositions remain original."""
    from carnot.reporting import v710_replay_history as h

    work = original_work
    assert len(work["historical_dispositions"]) == 13
    by_id = {r["original_primary"]["experiment_id"]: r for r in work["historical_dispositions"]}
    assert by_id[8210]["verdict_class"] == by_id[8217]["verdict_class"] == "disqualified"
    assert not any(r["historical_misconduct_inferred"] for r in by_id.values())
    assert work["retained_affected_failure"]["errno122_retained"]
    assert not work["contract"]["activated"]
    assert work["h1_custody"]["agreement"]


def test_real_git_custody(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8218-HISTORY: original commit bytes survive later checkout drift."""
    from carnot.reporting import v710_replay_history as h
    from carnot.reporting import v709_execution as x
    from carnot.reporting.current_work_receipt import sha256_file

    repo = tmp_path / "repo"
    repo.mkdir()
    for argv in [
        ["git", "init", "-q"],
        ["git", "config", "user.email", "fixture@example.invalid"],
        ["git", "config", "user.name", "Private fixture"],
    ]:
        subprocess.run(argv, cwd=repo, check=True, capture_output=True)
    path = repo / "code.py"
    path.write_text("original = 1\n")
    digest = sha256_file(path)
    subprocess.run(["git", "add", "code.py"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-qm", "Private original"], cwd=repo, check=True, capture_output=True
    )
    path.write_text("mutable = 2\n")
    monkeypatch.setattr(q, "ROOT", repo)
    monkeypatch.setattr(x, "ROOT", repo)
    ref = dict(path=str(path), sha256=digest)
    cache = {}
    copied = h.recover_code(ref, tmp_path / "custody", cache)
    assert copied["exists"] and q.verify_reference(copied)
    assert h.recover_code(ref, tmp_path / "again", cache) == copied
    assert not h.recover_code(dict(ref, sha256="sha256:" + "0" * 64), tmp_path / "missing", {})[
        "exists"
    ]
    assert not h.recover_code(dict(ref, path=str(tmp_path / "foreign")), tmp_path / "foreign", {})[
        "exists"
    ]


def test_full_authority_and_missing_history(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8218-HISTORY: missing source bytes keep their original task slots."""
    from carnot.reporting import v710_replay_history as h

    root = tmp_path / "root"
    root.mkdir()
    d, s, a = authority(tmp_path / "fixture")
    for source, name in [
        (d, q.DESIGN),
        (a, "research-roadmap.yaml"),
        (q.ROOT / q.HISTORY, q.HISTORY),
    ]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    monkeypatch.setattr(h, "INPUTS", [q.DESIGN, q.HISTORY, "research-roadmap.yaml"])
    work = h.measure(root, tmp_path / "raw")
    assert work["contract"]["activated"] and len(work["historical_dispositions"]) == 13
    assert all(r["original_primary"] is None for r in work["historical_dispositions"])
    assert all(c["rejected"] for c in work["mutation_controls"])
    monkeypatch.setattr(q, "mutations", lambda *args: [dict(control="fault", rejected=False)])
    work = h.measure(root, tmp_path / "fault")
    assert any(g["artifact_field"] == "mutation.fault" for g in work["failures"])


def test_environment_failure_is_real_and_blocked(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8218-CLI: an actual failed environment child cannot grant readiness."""
    from carnot.reporting import v710_replay_runner as run

    monkeypatch.setattr(
        run.qualified,
        "precondition_command",
        lambda: dict(
            name="environment_failure",
            argv=[sys.executable, "-c", "raise SystemExit(7)"],
            deadline=30,
            expected=0,
            scope="preconditions",
        ),
    )
    root = tmp_path / "root"
    root.mkdir()
    output = tmp_path / (q.NAME + ".json")
    assert run.main(["--root", str(root), "--output", str(output), "--private-fixture"]) == 0
    value = json.loads(output.read_bytes())
    assert any(
        g["artifact_field"] == "python_environment_exit" for g in value["gate_check_summary"]
    )
    assert value["precondition_receipts"][0]["actual_exit"] == 7
    assert value["contract_ready_score"] == 0
    value["schema"] = "tampered checksum"
    atomic_json(output, value)
    with pytest.raises(ValueError, match="checksum"):
        run.replay(output)


def test_custody_rejects_changed_h1(tmp_path, monkeypatch):
    """REQ-VERIFY-8218: a changed historical headline is a custody failure, never new science."""
    from carnot.reporting import v710_replay_history as h

    work = dict(
        source_artifact_hashes=[],
        preconditions_checked=[],
        failures=[],
        immutable_code_snapshots=[],
        replay_controls=[],
    )
    source = json.loads(
        (q.ROOT / "results/experiment_8210_v709_restricted_decision_audit.json").read_bytes()
    )
    source["H1"]["passed"] = not source["H1"]["passed"]
    monkeypatch.setattr(h.audit, "CLI", "scripts/experiments/unavailable_audit.py")
    h.audit_copy(source, tmp_path, work)
    assert not work["h1_custody"]["agreement"]
    assert any(g["artifact_field"] == "H1" for g in work["failures"])


@pytest.fixture(scope="module")
def original_work(tmp_path_factory):
    """SCENARIO-REPORT-8218-HISTORY: authenticate real original operands once per suite."""
    from carnot.reporting import v710_replay_history as h

    return h.measure(q.ROOT, tmp_path_factory.mktemp("original-8218"))


def candidate(work, directory):
    """Private candidates use actual measured controls and immutable source operands."""
    from carnot.reporting.current_work_receipt import sha256_file

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "work.json"
    atomic_json(path, work)
    value = q.reduce(work, work["replay_controls"])
    value.update(
        work_reference=dict(path=str(path), sha256=sha256_file(path)),
        raw_shard_hashes=[],
        code_config_hashes=[],
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    output = directory / "candidate.json"
    atomic_json(output, value)
    return output


def test_cold_original_primitives_and_rehashed_failures(tmp_path, original_work):
    """SCENARIO-VERIFY-8218-REPLAY: rehashing claims cannot replace the original operands."""
    path = candidate(original_work, tmp_path / "valid")
    result = cli(tmp_path, "--cold-replay", path)
    assert result.returncode == 0, result.stdout + result.stderr
    changed = deepcopy(original_work)
    changed["historical_dispositions"][0]["original_primary"]["honest_verdict"] = (
        "complete_positive_invented"
    )
    result = cli(tmp_path, "--cold-replay", candidate(changed, tmp_path / "history"))
    assert result.returncode == 1 and "historical_disposition" in result.stdout
    changed = deepcopy(original_work)
    changed["h1_custody"]["H1"]["passed"] = not changed["h1_custody"]["H1"]["passed"]
    result = cli(tmp_path, "--cold-replay", candidate(changed, tmp_path / "h1"))
    assert result.returncode == 1 and "original_H1" in result.stdout


def test_cold_full_authority_and_changed_reduction(tmp_path):
    """SCENARIO-REPORT-8218-CONTRACT: cold authority recomputation ignores altered aggregates."""
    from carnot.reporting import v709_execution as x
    from carnot.reporting import v710_replay_runner as run

    d, s, a = authority(tmp_path / "fixture")
    work = dict(
        contract=q.assess(d, s, a, tmp_path / "authority"),
        failures=[],
        historical_dispositions=[],
        immutable_code_snapshots=[],
        h1_custody={},
        preconditions_checked=[],
        source_artifact_hashes=[],
        replay_controls=x.pytest_controls(tmp_path / "controls", tmp_path / "logs"),
    )
    assert run.replay(candidate(work, tmp_path / "valid"))["passed"]
    changed = deepcopy(work)
    changed["contract"]["canonical_tasks_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="authority_reduction"):
        run.replay(candidate(changed, tmp_path / "changed"))


def test_unavailable_reproduction_and_code(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8218-HISTORY: absent failure resources remain external blocks."""
    from carnot.reporting import v710_replay_history as h

    root = tmp_path / "root"
    d, s, a = authority(tmp_path / "fixture")
    for source, name in [
        (d, q.DESIGN),
        (a, "research-roadmap.yaml"),
        (q.ROOT / q.HISTORY, q.HISTORY),
        (
            q.ROOT / "results/experiment_8217_v709_capstone.json",
            "results/experiment_8217_v709_capstone.json",
        ),
    ]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    monkeypatch.setattr(h, "INPUTS", [q.DESIGN, q.HISTORY])
    monkeypatch.setattr(h, "recover_code", lambda ref, *args: dict(ref, exists=False))
    work = h.measure(root, tmp_path / "raw")
    assert any(g["artifact_field"] == "original_code_bytes" for g in work["failures"])
    assert any(
        g["artifact_field"] == "retained_quota_failure_reproduction" for g in work["failures"]
    )


def test_missing_audit_operands_stop_reduction(tmp_path):
    """REQ-VERIFY-8218: absent original measurement and source copies remain unavailable."""
    from carnot.reporting import v710_replay_history as h
    from carnot.reporting.current_work_receipt import sha256_file

    source = json.loads(
        (q.ROOT / "results/experiment_8210_v709_restricted_decision_audit.json").read_bytes()
    )
    work = dict(source_artifact_hashes=[], preconditions_checked=[], failures=[])
    absent = deepcopy(source)
    absent["measurement_reference"]["path"] = str(tmp_path / "missing.json")
    h.audit_copy(absent, tmp_path / "missing", work)
    assert work["h1_custody"]["agreement"] is None and work["failures"]
    original = source["measurement_reference"]
    data = json.loads(Path(original["path"]).read_bytes())
    data["refs"][0]["path"] = str(tmp_path / "missing_source.json")
    modified = tmp_path / "private_measurement.json"
    atomic_json(modified, data)
    source["measurement_reference"] = dict(path=str(modified), sha256=sha256_file(modified))
    work = dict(
        source_artifact_hashes=[],
        preconditions_checked=[],
        failures=[],
        immutable_code_snapshots=[],
    )
    h.audit_copy(source, tmp_path / "source", work)
    assert work["h1_custody"]["agreement"] is None and work["failures"]
