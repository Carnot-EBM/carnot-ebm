"""REQ-REPORT-9068-V774: private controls cannot supply scientific observations."""

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

from carnot.reporting import v774_dispatch_contract as c


def authority_tree(root):
    """Copy only metadata so refusal controls cannot edit live scheduling."""
    for name in [
        c.DESIGN,
        c.ACTIVE,
        c.LOG,
        c.ARCHIVE,
        c.QUEUE,
        "scripts/conductor_exclusion_manifest.json",
    ]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((c.ROOT / name).read_bytes())
    return root


def test_authority_and_scheduler(tmp_path):
    """SCENARIO-REPORT-9068-V774: exact active/staged objects and real old/new selection."""
    root = authority_tree(tmp_path / "root")
    value = c.authority(root)
    assert value["ready"] and value["comparison_mode"] == "activated"
    assert len(value["rows"]) == len(value["title_key_rows"]) == 14
    assert all(r["passed"] for r in value["title_key_rows"])
    rows = c.scheduler_replay(root, value["tasks"], tmp_path / "scheduler")
    assert rows[0]["selected_id"] is None and rows[0]["expected_failure_observed"]
    assert rows[1]["selected_id"] == "exp9071-collect-b" and rows[1]["passed"]
    assert c.scheduler_replay(root, value["tasks"], tmp_path / "scheduler") == rows
    queue = yaml.safe_load((root / c.ACTIVE).read_bytes())
    (root / c.STAGED).write_text(yaml.safe_dump(queue))
    (root / c.ACTIVE).write_text("milestone: 2026.11.773\ntasks: []\n")
    assert c.authority(root)["ready"]
    assert c.authority(root)["comparison_mode"] == "staged"
    (root / c.STAGED).unlink()
    assert not c.authority(root)["ready"]
    assert not c.authority(tmp_path / "absent")["ready"]


@pytest.mark.parametrize("kind", ["prompt", "model", "title", "gate", "order", "design", "prior"])
def test_authority_mutations(tmp_path, kind):
    """REQ-REPORT-9068-V774: independently pinned full objects defeat coordinated edits."""
    root = authority_tree(tmp_path / "root")
    q = yaml.safe_load((root / c.ACTIVE).read_bytes())
    if kind in {"prompt", "model", "title"}:
        q["tasks"][0][kind] += " forged"
    if kind == "gate":
        q["tasks"][1]["gated_on"][0]["value"] = 99
    if kind == "order":
        q["tasks"].reverse()
    if kind == "design":
        with (root / c.DESIGN).open("a") as f:
            f.write("changed authority\n")
    if kind == "prior":
        with (root / c.LOG).open("a") as f:
            f.write(f"| date | {q['tasks'][0]['title'][:50].strip()} | OK | old |\n")
    (root / c.ACTIVE).write_text(yaml.safe_dump(q))
    assert not c.authority(root)["ready"]


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """Use authenticated historical bytes once; no TEST outcomes are opened."""
    p = tmp_path_factory.mktemp("dispatch9068")
    return c.collect(c.ROOT, p / "raw", p / "private", "20261112")


def checked(work, tmp_path):
    """Constructed receipts exercise reducer refusal only, never production qualification."""
    w = deepcopy(work)
    tmp_path.mkdir(parents=True, exist_ok=True)
    w["validation"] = dict(receipts=[], coverage=dict(complete=True))
    for command in w["validation_scope_manifest"]["commands"]:
        log = tmp_path / (command["name"] + ".log")
        log.write_text("private receipt control\n")
        w["validation"]["receipts"].append(
            dict(
                name=command["name"],
                command_argv=command["argv"],
                passed=True,
                normal_exit=True,
                timed_out=False,
                exit_code=0,
                started_monotonic_ns=w["frozen_ns"],
                ended_monotonic_ns=w["frozen_ns"],
                scope_sha256=c.canonical_hash(w["validation_scope_manifest"]),
                log_path=str(log),
                log_sha256=c.reference(log)["sha256"],
            )
        )
    return w


def test_reduction_and_independent_gates(work, tmp_path, monkeypatch):
    """REQ-REPORT-9068-V774: zero ARC support leaves custody and frozen methods intact."""
    monkeypatch.setattr(c, "verify_measured", lambda *_: True)
    w = checked(work, tmp_path)
    v = c.build(w)
    assert v["contract_ready_score"] == v["frozen_methods_ready_score"] == 1
    assert v["arc_new_outcomes_ready_score"] == 0 and v["verdict_class"] == "null"
    assert v["completed_count"] == v["intended_count"] == len(v["rows"]) == 14
    assert v["arc_frontier"]["baseline_experiment_id"] == 9054
    assert v["arc_frontier"]["ledger_reference"]["sha256"]
    assert all(r["check"] == "optional_arc_support" for r in v["arc_frontier"]["checks"])
    assert v["archive_lag"]["completed_scheduling_frontier"] == "2026.11.773"
    assert v["model_invocation_counts"]["model_loads"] == 0
    assert v["MODEL_SPECS"] == v["trained_head_specs"] == []
    assert not v["repository_health"]["passed"]
    assert v["preserved_v773"]["missing_B_is_proof_of_no_attempt"] is False
    w["reduction"]["science"]["frozen_graph_authority_ready_score"] = 0
    assert c.build(w)["contract_ready_score"] == 1
    assert c.build(w)["frozen_methods_ready_score"] == 0


@pytest.mark.parametrize(
    "kind",
    ["missing", "stale", "failed", "timeout", "log", "command", "duplicate", "code", "coverage"],
)
def test_receipt_refusal(work, tmp_path, monkeypatch, kind):
    """REQ-VERIFY-9068: every failed current check zeros all readiness gates."""
    monkeypatch.setattr(c, "verify_measured", lambda *_: kind != "coverage")
    w = checked(work, tmp_path)
    c.validation.mutate_receipt(w, kind)
    v = c.build(w)
    assert v["verdict_class"] == "disqualified"
    assert not any(v[k] for k in c.READINESS)
    assert any(
        r["check"] == "current_qualification" and r["observed"] is False
        for r in v["gate_check_summary"]
    )


def test_blocked_before_reduction(tmp_path, monkeypatch):
    """REQ-REPORT-9068-V774: absent prerequisites stop computation and retain exact failures."""
    monkeypatch.setattr(c, "reduce", lambda *_: pytest.fail("reduced missing sources"))
    w = c.collect(tmp_path / "absent", tmp_path / "raw", tmp_path / "private", "20261112")
    v = c.build(w)
    assert v["verdict_class"] == "blocked" and v["honest_verdict"].startswith("complete_blocked_")
    assert any(r["observed"] == "missing_file" for r in v["gate_check_summary"])
    assert v["censored_count"] == 14 and not any(v[k] for k in c.READINESS)


def child(tmp_path, *args):
    """Exercise the real CLI without checkout cwd or ambient Python import paths."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(c.ROOT / c.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("[test9068] child_before", args, flush=True)
    result = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90)
    print(
        "[test9068] child_after",
        result.returncode,
        result.stdout[-600:],
        result.stderr[-600:],
        flush=True,
    )
    return result


def test_foreign_cli_and_rehashed_replay(work, tmp_path):
    """SCENARIO-VERIFY-9068-REPLAY: honest private diagnostics replay; forged gates fail."""
    output = tmp_path / (c.NAME + ".json")
    source = tmp_path / "work.json"
    c.atomic_json(source, work)
    assert child(tmp_path, "--from-work", source, "--output", output).returncode == 0
    assert child(tmp_path, "--cold-replay", output).returncode == 0
    value = json.loads(output.read_bytes())
    value["contract_ready_score"] = 99
    value["reproducibility_checksum"] = c.checksum(value)
    c.atomic_json(output, value)
    assert child(tmp_path, "--cold-replay", output).returncode == 1
    assert child(tmp_path, "--date", "bad").returncode == 2
    assert child(tmp_path, "--from-work", source).returncode == 2
    assert child(tmp_path, "--cold-replay", tmp_path / "absent").returncode == 1


def test_replay_rejects_rehashed_evidence(work, tmp_path):
    """REQ-VERIFY-9068: changing both the work and envelope cannot invent an authority."""
    w = deepcopy(work)
    w["raw"] = str(tmp_path)
    c.atomic_json(tmp_path / "work.json", w)
    value = c.build(w)
    candidate = tmp_path / (c.NAME + ".json")
    c.atomic_json(candidate, value)
    assert c.replay(candidate)["passed"]
    for kind in ["reduction", "scope", "operand"]:
        changed = deepcopy(w)
        if kind == "reduction":
            changed["reduction"]["contract"]["ready"] = False
        if kind == "scope":
            changed["validation_scope_manifest"]["statement_percent"] = 99
        c.atomic_json(tmp_path / "work.json", changed)
        v = c.build(changed)
        if kind == "operand":
            v["raw_shard_hashes"][0]["sha256"] = "sha256:forged"
        v["reproducibility_checksum"] = c.checksum(v)
        c.atomic_json(candidate, v)
        assert not c.replay(candidate)["passed"], kind


def test_authority_missing_field_and_scheduler_bytes(tmp_path):
    """REQ-REPORT-9068-V774: missing fields and changed replay fixtures remain visible."""
    root = authority_tree(tmp_path / "root")
    (root / c.ACTIVE).write_text("tasks: []\n")
    assert any(r["observed"] == "missing_field" for r in c.authority(root)["checks"])
    root = authority_tree(root)
    tasks = c.authority(root)["tasks"]
    c.scheduler_replay(root, tasks, tmp_path / "scheduler")
    with (tmp_path / "scheduler/new" / c.ACTIVE).open("a") as f:
        f.write("forged: true\n")
    with pytest.raises(ValueError, match="fixture_drift"):
        c.scheduler_replay(root, tasks, tmp_path / "scheduler")


def test_validation_uses_exact_frozen_commands(work, monkeypatch):
    """REQ-VERIFY-9068: qualification forwards every frozen command and real coverage path."""
    calls = []
    monkeypatch.setattr(
        c.validation,
        "run_check",
        lambda spec, *_: calls.append(spec) or dict(name=spec.name, passed=True),
    )
    monkeypatch.setattr(
        c,
        "measured",
        lambda paths, owned, raw: dict(complete=False, rows=[], paths=list(map(str, paths))),
    )
    result = c.validate(work)
    assert [s.name for s in calls] == work["validation_scope_manifest"]["required_check_names"]
    assert result["coverage"]["complete"] is False
    assert "rows" not in result["coverage"]


def test_main_current_qualification_and_refusal(work, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-9068-REPLAY: the production path refuses failed owned checks."""

    def collect(root, raw, private, date):
        w = deepcopy(work)
        w["raw"] = str(raw)
        return w

    monkeypatch.setattr(c, "collect", collect)
    monkeypatch.setattr(c, "verify_measured", lambda *_: True)
    monkeypatch.setattr(c, "validate", lambda w: checked(w, tmp_path / "receipts")["validation"])
    monkeypatch.setattr(c, "terminal", lambda *_: dict(passed=True))
    output = tmp_path / "accepted" / (c.NAME + ".json")
    assert c.main(["--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["contract_ready_score"] == 1
    monkeypatch.setattr(c, "validate", lambda w: dict(receipts=[], coverage=dict(complete=False)))
    failed = tmp_path / "failed" / (c.NAME + ".json")
    assert c.main(["--output", str(failed)]) == 1
    assert not failed.exists()
    delayed = deepcopy(work)
    delayed["start_ns"] -= 3301 * 10**9
    monkeypatch.setattr(c, "collect", lambda *_: delayed)
    with pytest.raises(TimeoutError, match="measurement_deadline"):
        c.main(["--output", str(tmp_path / "late" / (c.NAME + ".json"))])


def test_actual_missing_input_cli(tmp_path):
    """REQ-REPORT-9068-V774: missing external input publishes an honest blocked diagnostic."""
    output = tmp_path / (c.NAME + ".json")
    result = child(tmp_path, "--root", tmp_path / "absent", "--output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(output.read_bytes())["verdict_class"] == "blocked"


def test_staged_snapshot_and_separate_current_health(work, tmp_path, monkeypatch):
    """REQ-REPORT-9068-V774: stage exact bytes and retain failed broader health separately."""
    root = tmp_path / "root"
    shutil.copytree(work["retained"], root)
    (root / c.STAGED).write_bytes((root / c.ACTIVE).read_bytes())
    (root / c.ACTIVE).write_text("milestone: 2026.11.773\ntasks: []\n")
    health = tmp_path / "health.json"
    c.atomic_json(health, dict(passed=False, exit_code=1, scope="broader_repository_health"))
    monkeypatch.setattr(c, "HEALTH", health)
    monkeypatch.setattr(c, "reduce", lambda *_: dict(fixture_only=True))
    w = c.collect(root, tmp_path / "raw", tmp_path / "private", "20261112")
    assert w["reduction"] == dict(fixture_only=True)
    assert any(r["source"]["path"] == str(root / c.STAGED) for r in w["snapshots"])
    current = w["repository_health"]["current_suite"]
    assert current["result"]["passed"] is False
    assert current["references"][0] == c.reference(health)


def test_qualified_arc_support_boundaries():
    """REQ-REPORT-9068-V774: thirty unique episodes, three families and paired arms are required."""
    from carnot.agentic.arc_trajectory_supervisor import ARM_ORDER
    from carnot.reporting.v766_branch_evidence import arc_support

    rows = [
        dict(
            status="completed",
            selected=True,
            applied=True,
            arm=arm,
            invocation_id="private",
            game=f"family{family}",
            seed=seed,
        )
        for family in range(3)
        for seed in range(10)
        for arm in ARM_ORDER[:2]
    ]
    assert arc_support(rows)["ready"]
    assert arc_support(rows + rows)["new_uncensored_episode_count"] == 30
    assert not arc_support([r for r in rows if not (r["game"] == "family2" and r["seed"] == 9)])[
        "ready"
    ]
    assert not arc_support(
        [r for r in rows if not (r["game"] == "family2" and r["arm"] == ARM_ORDER[1])]
    )["ready"]
    censored = [dict(r, status="censored") if r["game"] == "family2" else r for r in rows]
    assert not arc_support(censored)["ready"]
