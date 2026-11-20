"""REQ-REPORT-9082-V775: administrative readiness grants no scientific credit."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from carnot.reporting import v775_contract as c


def tree(root):
    """Private metadata copies keep mutation controls away from live scheduling."""
    for name in c.AUTHORITIES:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((c.ROOT / name).read_bytes())
    return root


def test_authority_and_scheduler(tmp_path):
    """SCENARIO-REPORT-9082-V775: all objects and actual selector agree."""
    root = tree(tmp_path / "root")
    a = c.authority(root)
    assert a["ready"] and a["comparison_mode"] == "activated"
    assert len(a["rows"]) == 14 and all(r["matched"] for r in a["rows"])
    rows = c.scheduler_replay(root, a["tasks"], tmp_path / "scheduler")
    assert rows[0]["expected_failure_observed"] and rows[0]["passed"]
    assert rows[1]["selected_id"] == "exp9086-collect-b" and rows[1]["passed"]
    assert c.scheduler_replay(root, a["tasks"], tmp_path / "scheduler") == rows
    queue = yaml.safe_load((root / c.ACTIVE).read_bytes())
    (root / c.STAGED).write_text(yaml.safe_dump(queue))
    (root / c.ACTIVE).write_text("milestone: 2026.11.774\ntasks: []\n")
    assert c.authority(root)["ready"]
    assert c.authority(root)["comparison_mode"] == "staged"
    (root / c.STAGED).unlink()
    assert not c.authority(root)["ready"]
    assert not c.authority(tmp_path / "absent")["ready"]
    with (tmp_path / "scheduler/new" / c.ACTIVE).open("a") as stream:
        stream.write("forged: true\n")
    with pytest.raises(ValueError, match="fixture_drift"):
        c.scheduler_replay(c.ROOT, a["tasks"], tmp_path / "scheduler")


@pytest.mark.parametrize("kind", ["prompt", "model", "gate", "order", "design", "prior", "id"])
def test_authority_refuses_changes(tmp_path, kind):
    """REQ-REPORT-9082-V775: coordinated task edits cannot replace frozen authority."""
    root = tree(tmp_path / "root")
    queue = yaml.safe_load((root / c.ACTIVE).read_bytes())
    if kind in {"prompt", "model"}:
        queue["tasks"][0][kind] += " forged"
    elif kind == "gate":
        queue["tasks"][1]["gated_on"][0]["artifact_field"] = "invented_score"
    elif kind == "order":
        queue["tasks"].reverse()
    elif kind == "design":
        with (root / c.DESIGN).open("a") as stream:
            stream.write("forged design\n")
    elif kind == "prior":
        with (root / c.LOG).open("a") as stream:
            stream.write(f"| date | {queue['tasks'][0]['title'][:50].strip()} | OK | old |\n")
    else:
        with (root / "research-complete.yaml").open("a") as stream:
            stream.write("\nused_id: exp9083-prospective-custody\n")
    (root / c.ACTIVE).write_text(yaml.safe_dump(queue))
    assert not c.authority(root)["ready"]


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """Authenticate real historical inputs once without opening protected truth."""
    p = tmp_path_factory.mktemp("contract9082")
    return c.collect(c.ROOT, p / "raw", p / "private", "20261115")


def qualify(work, tmp_path, monkeypatch):
    """Constructed receipts test rejection paths; production uses real subprocess exits."""
    w = deepcopy(work)
    monkeypatch.setattr(c, "verify_measured", lambda *_: True)
    w["validation"] = dict(receipts=[], coverage=dict(complete=True))
    for command in w["validation_scope_manifest"]["commands"]:
        path = tmp_path / (command["name"] + ".log")
        path.write_text("private receipt fixture\n")
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
                log_path=str(path),
                log_sha256=c.reference(path)["sha256"],
            )
        )
    return w


def test_independent_readiness(work, tmp_path, monkeypatch):
    """REQ-REPORT-9082-V775: historical nulls do not become current model work."""
    assert work["reduction"], work["preconditions_checked"]
    w = qualify(work, tmp_path, monkeypatch)
    value = c.build(w)
    assert value["contract_ready_score"] == value["frozen_methods_ready_score"] == 1
    assert value["verdict_class"] == "null" and value["independent_generalization_score"] == 0
    assert value["current_llm_invocations"] == 0 and value["MODEL_SPECS"] == []
    assert value["intended_count"] == value["attempted_count"] == value["completed_count"] == 14
    assert len(value["previous_outcome_inventory"]) == 14
    assert value["archive_lag"]["planning_archive_end"] == "2026.11.773"
    for ref in value["frozen_protocol_references"]:
        frozen = json.loads(Path(ref["path"]).read_bytes())
        original = json.loads(Path(ref["original_reference"]["path"]).read_bytes())
        assert frozen["method"] == original
        assert frozen["access_authority"] == "exp9083-prospective-custody"
    w["reduction"]["science"]["online_protocol_ready_score"] = 0
    value = c.build(w)
    assert value["contract_ready_score"] == 1 and value["frozen_methods_ready_score"] == 0
    w["reduction"]["science"]["online_protocol_ready_score"] = 1
    w["reduction"]["contract"]["ready"] = False
    assert c.build(w)["frozen_methods_ready_score"] == 1


@pytest.mark.parametrize(
    "kind", ["missing", "stale", "failed", "timeout", "log", "command", "duplicate", "code"]
)
def test_receipt_refusal(work, tmp_path, monkeypatch, kind):
    """REQ-REPORT-9082-V775: each affected-check failure zeros every readiness claim."""
    w = qualify(work, tmp_path, monkeypatch)
    c.validation.mutate_receipt(w, kind)
    value = c.build(w)
    assert value["verdict_class"] == "disqualified"
    assert value["contract_ready_score"] == value["frozen_methods_ready_score"] == 0


def test_missing_inputs(tmp_path, monkeypatch):
    """SCENARIO-REPORT-9082-V775: unavailable inputs stop dependent work exactly once."""
    monkeypatch.setattr(c, "reduce", lambda *_: pytest.fail("reduced absent inputs"))
    w = c.collect(tmp_path / "absent", tmp_path / "raw", tmp_path / "private", "20261115")
    value = c.build(w)
    assert value["verdict_class"] == "blocked"
    assert value["honest_verdict"].startswith("complete_blocked_")
    assert any(r["observed"] == "missing_file" for r in value["gate_check_summary"])
    assert value["censored_count"] == 14


def child(tmp_path, *args):
    """Real child processes prove imports and replay without checkout cwd or PYTHONPATH."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(c.ROOT / c.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("[test9082] child_before", args, flush=True)
    result = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90)
    print(
        "[test9082] child_after",
        result.returncode,
        result.stdout[-500:],
        result.stderr[-500:],
        flush=True,
    )
    return result


def test_foreign_cli(work, tmp_path):
    """SCENARIO-REPORT-9082-V775: private publication, block, cold replay and tamper refusal."""
    source, output = tmp_path / "source.json", tmp_path / (c.NAME + ".json")
    c.atomic_json(source, work)
    assert child(tmp_path, "--from-work", source, "--output", output).returncode == 0
    assert child(tmp_path, "--cold-replay", output).returncode == 0
    value = json.loads(output.read_bytes())
    value["contract_ready_score"] = 99
    value["reproducibility_checksum"] = c.checksum(value)
    c.atomic_json(output, value)
    assert child(tmp_path, "--cold-replay", output).returncode == 1
    blocked = tmp_path / "blocked" / (c.NAME + ".json")
    assert child(tmp_path, "--root", tmp_path / "absent", "--output", blocked).returncode == 0
    assert child(tmp_path, "--cold-replay", blocked).returncode == 0
    assert json.loads(blocked.read_bytes())["verdict_class"] == "blocked"
    assert child(tmp_path, "--date", "bad").returncode == 2
    assert child(tmp_path, "--from-work", source).returncode == 2
    assert child(tmp_path, "--cold-replay", tmp_path / "absent").returncode == 1


def test_replay_rebuilds(work, tmp_path):
    """REQ-REPORT-9082-V775: rehashing forged aggregate or retained work still fails."""
    w = deepcopy(work)
    w["raw"] = str(tmp_path)
    c.atomic_json(tmp_path / "work.json", w)
    output = tmp_path / (c.NAME + ".json")
    c.atomic_json(output, c.build(w))
    assert c.replay(output)["passed"]
    for kind in ["reduction", "scope", "operand"]:
        changed = deepcopy(w)
        if kind == "reduction":
            changed["reduction"]["contract"]["ready"] = False
        elif kind == "scope":
            changed["validation_scope_manifest"]["statement_percent"] = 99
        c.atomic_json(tmp_path / "work.json", changed)
        value = c.build(changed)
        if kind == "operand":
            value["raw_shard_hashes"][0]["sha256"] = "sha256:forged"
        value["reproducibility_checksum"] = c.checksum(value)
        c.atomic_json(output, value)
        assert not c.replay(output)["passed"], kind


def test_validation_orchestration(work, monkeypatch):
    """REQ-REPORT-9082-V775: each frozen command runs once and owns only new coverage."""
    calls = []
    monkeypatch.setattr(
        c.validation, "run_check", lambda spec, *_: calls.append(spec.name) or dict(name=spec.name)
    )
    monkeypatch.setattr(c, "measured", lambda *_: dict(complete=True, rows=[]))
    assert c.validate(work)["coverage"] == dict(complete=True)
    assert calls == [r["name"] for r in work["validation_scope_manifest"]["commands"]]
    monkeypatch.setattr(c, "collect", lambda *_: deepcopy(work))
    monkeypatch.setattr(c, "validate", lambda *_: dict(receipts=[], coverage=dict(complete=False)))
    assert c.main(["--output", "/tmp/carnot9082-disqualified-test/" + c.NAME + ".json"]) == 1
    monkeypatch.setattr(c.time, "monotonic_ns", lambda: work["start_ns"] + 3301_000_000_000)
    with pytest.raises(TimeoutError, match="measurement_deadline"):
        c.main(["--output", "/tmp/carnot9082-deadline-test/" + c.NAME + ".json"])


def test_novel_scope_and_existing_id(tmp_path, monkeypatch):
    """REQ-REPORT-9082-V775: novel scope needs no invented failure, but used IDs are refused."""
    import re

    root = tree(tmp_path / "root")
    queue = yaml.safe_load((root / c.ACTIVE).read_bytes())
    queue["tasks"][0]["prior_failures"] = []
    design = (root / c.DESIGN).read_text()
    design = re.sub(
        r"(<!-- V775_TASK_CONTRACT_START -->\s*```json\s*)(.*?)(\s*```)",
        lambda m: m[1] + json.dumps(dict(milestone=c.MILESTONE, tasks=queue["tasks"])) + m[3],
        design,
        flags=re.S,
    )
    (root / c.DESIGN).write_text(design)
    (root / c.ACTIVE).write_text(yaml.safe_dump(queue))
    monkeypatch.setitem(c.PINS, c.DESIGN, c.reference(root / c.DESIGN)["sha256"])
    assert c.authority(root)["ready"]
    collision = root / "results/experiment_9086_already_used.json"
    c.atomic_json(collision, dict(experiment_id=9086))
    assert not c.authority(root)["ready"]


def test_immutable_copies(work, tmp_path):
    """REQ-REPORT-9082-V775: existing source and protocol copies cannot be rewritten."""
    raw = tmp_path / "raw"
    target = raw / "authority" / c.DESIGN
    target.parent.mkdir(parents=True)
    target.write_text("forged source\n")
    raw.chmod(0o700)
    with pytest.raises(ValueError, match="immutable_snapshot_drift"):
        c.collect(c.ROOT, raw, tmp_path / "private", "20261115")
    w = deepcopy(work)
    w["evidence_raw"] = str(tmp_path)
    c.atomic_json(tmp_path / "frozen_protocols/v775-spline-protocol.json", dict(method="forged"))
    with pytest.raises(ValueError, match="immutable_protocol_drift"):
        c.reduce(w)


def test_main_success_orchestration(work, tmp_path, monkeypatch):
    """REQ-REPORT-9082-V775: normal validated runs retain one full-suite receipt before publication."""
    w = qualify(work, tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(c, "collect", lambda *_: deepcopy(w))
    monkeypatch.setattr(c, "validate", lambda _: w["validation"])
    monkeypatch.setattr(
        c.validation,
        "run_check",
        lambda spec, *_: calls.append(spec.argv) or dict(passed=False, normal_exit=True),
    )
    monkeypatch.setattr(c, "terminal", lambda *_: dict(passed=True))
    output = tmp_path / (c.NAME + ".json")
    assert c.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["contract_ready_score"] == value["frozen_methods_ready_score"] == 1
    assert not value["repository_health"]["passed"]
    assert list(calls[0]) == w["validation_scope_manifest"]["global_suite_command"]


def test_collect_staged_snapshot(tmp_path):
    """SCENARIO-REPORT-9082-V775: staged authority is retained even when methods are unavailable."""
    root = tree(tmp_path / "root")
    (root / c.STAGED).write_bytes((root / c.ACTIVE).read_bytes())
    (root / c.ACTIVE).write_text("milestone: 2026.11.774\ntasks: []\n")
    w = c.collect(root, tmp_path / "raw", tmp_path / "private", "20261115")
    assert any(r["source"]["path"] == str(root / c.STAGED) for r in w["snapshots"])
    assert not w["reduction"]


@pytest.mark.parametrize("kind", ["missing", "command", "log", "stale", "timeout", "malformed"])
def test_activation_receipts_are_bound(work, tmp_path, monkeypatch, kind):
    """REQ-REPORT-9082-V775: missing or forged activation receipts cannot qualify readiness."""
    w = qualify(work, tmp_path, monkeypatch)
    if kind == "missing":
        w["activation_receipts"].pop()
    elif kind == "malformed":
        w["activation_receipts"][0] = {}
    elif kind == "command":
        w["activation_receipts"][0]["command_argv"] = ["unmeasured"]
    elif kind == "log":
        w["activation_receipts"][0]["log_sha256"] = "sha256:forged"
    elif kind == "stale":
        w["activation_receipts"][0]["scope_sha256"] = "sha256:stale"
    else:
        w["activation_receipts"][0]["timed_out"] = True
    value = c.build(w)
    assert value["contract_ready_score"] == value["frozen_methods_ready_score"] == 0
    assert value["verdict_class"] == "disqualified"


def test_replay_pins_cannot_be_rehashed(work, tmp_path):
    """REQ-REPORT-9082-V775: changing a historical source and its envelope still fails fixed pins."""
    import shutil

    w = deepcopy(work)
    retained = tmp_path / "retained"
    shutil.copytree(Path(w["retained"]), retained)
    w.update(raw=str(tmp_path), retained=str(retained))
    (retained / c.evidence.PROTOCOL).write_text("{}\n")
    c.atomic_json(tmp_path / "work.json", w)
    output = tmp_path / (c.NAME + ".json")
    c.atomic_json(output, c.build(w))
    assert c.replay(output)["reason"] == "pinned_authority_drift"


def test_live_id_collision_precedes_snapshot(tmp_path):
    """REQ-REPORT-9082-V775: an unarchived live artifact still burns its experiment ID."""
    root = tree(tmp_path / "root")
    c.atomic_json(root / "results/experiment_9086_existing.json", dict(experiment_id=9086))
    w = c.collect(root, tmp_path / "raw", tmp_path / "private", "20261115")
    assert any(
        r["check"] == "full_contract"
        and r["path"] == str(root / c.ACTIVE)
        and r["observed"] is False
        for r in w["preconditions_checked"]["checks"]
    )
    assert c.build(w)["contract_ready_score"] == 0
