"""REQ-REPORT-9096-V776: custody and planning do not establish scientific benefit."""

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

from carnot.reporting import v776_contract as c


def tree(root):
    """Private authority copies let mutation tests leave real scheduling alone."""
    for name in c.AUTHORITIES:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((c.ROOT / name).read_bytes())
    return root


def test_authority(tmp_path):
    """SCENARIO-REPORT-9096-V776: activated and staged full objects both qualify."""
    root = tree(tmp_path / "root")
    result = c.authority(root)
    assert result["ready"] and result["comparison_mode"] == "activated"
    assert len(result["rows"]) == 14 and all(r["matched"] for r in result["rows"])
    queue = yaml.safe_load((root / c.ACTIVE).read_bytes())
    (root / c.STAGED).write_text(yaml.safe_dump(queue))
    (root / c.ACTIVE).write_text("milestone: 2026.11.775\ntasks: []\n")
    assert c.authority(root)["ready"]
    assert c.authority(root)["comparison_mode"] == "staged"
    (root / c.STAGED).unlink()
    assert not c.authority(root)["ready"]
    assert not c.authority(tmp_path / "absent")["ready"]


@pytest.mark.parametrize(
    "kind", ["prompt", "gate", "order", "title", "design", "used", "completed", "model", "count"]
)
def test_authority_refuses_drift(tmp_path, kind):
    """REQ-REPORT-9096-V776: coordinated edits cannot replace the pinned design."""
    root = tree(tmp_path / "root")
    queue = yaml.safe_load((root / c.ACTIVE).read_bytes())
    if kind == "prompt":
        queue["tasks"][0]["prompt"] += " forged"
    elif kind == "gate":
        queue["tasks"][1]["gated_on"][0]["artifact_field"] = "undeclared_score"
    elif kind == "order":
        queue["tasks"].reverse()
    elif kind == "title":
        queue["tasks"][1]["title"] = queue["tasks"][0]["title"]
    elif kind == "design":
        with (root / c.DESIGN).open("a") as stream:
            stream.write("forged\n")
    elif kind == "used":
        c.atomic_json(root / "results/experiment_9100_other.json", {})
    elif kind == "completed":
        with (root / c.LOG).open("a") as stream:
            stream.write(f"| date | {queue['tasks'][0]['title'][:50].strip()} | OK | old |\n")
    elif kind == "model":
        queue["tasks"][0]["MODEL_SPECS"] = [{"hf_id": "invented"}]
    else:
        queue["tasks"].pop()
    (root / c.ACTIVE).write_text(yaml.safe_dump(queue))
    assert not c.authority(root)["ready"]


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """Real byte authentication runs once before controls alter private envelopes."""
    path = tmp_path_factory.mktemp("contract9096")
    return c.collect(c.ROOT, path / "raw", path / "private", "20261115")


def test_preserved_outcomes_and_protocols(work):
    """REQ-REPORT-9096-V776: preserve all historical outcomes and unchanged methods."""
    assert work["reduction"], work["preconditions_checked"]
    value = c.build(work)
    assert len(value["full_task_objects"]) == len(value["previous_outcome_rows"]) == 14
    assert value["canonical_task_hash"] == c.canonical_hash(value["full_task_objects"])
    assert value["MODEL_SPECS"] == [] and value["current_llm_invocations"] == 0
    assert value["independent_generalization_score"] == value["effective_independent_groups"] == 0
    assert value["archive_lag"]["planning_archive_end"] == "2026.11.774"
    archive = yaml.safe_load((c.ROOT / "research-complete.yaml").read_bytes())
    assert value["archive_lag"]["observed_archive_end"] == archive["milestones"][-1]["id"]
    assert value["open_gaps"] == ["GAP-ORACLE-DISTINCT", "GAP-DETECTOR-AUROC-4208"]
    rows = {r["experiment_id"]: r for r in value["previous_outcome_rows"]}
    assert rows[9083]["honest_verdict"] == "complete_blocked_historical_attempt_topology_coverage"
    assert rows[9084]["honest_verdict"] == "complete_blocked_historical_producer_hash"
    assert rows[9085]["primary_status"] == "missing_file"
    assert rows[9085]["scheduler_honest_verdict"] == "blocked_gate_check_failed"
    assert rows[9093]["honest_verdict"] == "complete_null_inventory_unavailable_service"
    assert rows[9094]["verdict_class"] == "circular_positive"
    assert len(value["protocol_references"]) == 5
    static = json.loads(Path(value["protocol_references"][0]["path"]).read_bytes())
    human = json.loads((c.ROOT / c.HUMAN).read_bytes())
    assert static["protocol"] == human["protocol"]
    assert len(value["deferred_graph_bounds"]["seven_bound_rows"]) == 7
    assert len(value["deferred_graph_bounds"]["ten_bound_rows"]) == 10
    assert value["preserved_results"]["custody"]["fixture_custody_ready_score"] == 1
    assert value["preserved_results"]["human"]["question_count"] == 300


def test_missing_inputs(tmp_path):
    """SCENARIO-REPORT-9096-V776: an unchanged external absence is terminal blocked."""
    work = c.collect(tmp_path / "absent", tmp_path / "raw", tmp_path / "private", "20261115")
    value = c.build(work)
    assert value["verdict_class"] == "blocked" and value["censored_count"] == 14
    assert value["honest_verdict"].startswith("complete_blocked_")
    assert any(r["observed"] == "missing_file" for r in value["gate_check_summary"])
    assert value["contract_ready_score"] == 0


def child(tmp_path, *args):
    """Actual script imports and replay must work with no checkout import environment."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(c.ROOT / c.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("[test9096] child_before", args, flush=True)
    result = subprocess.run(
        argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
    )
    print(
        "[test9096] child_after",
        result.returncode,
        result.stdout[-600:],
        result.stderr[-600:],
        flush=True,
    )
    return result


def qualify(work, tmp_path, monkeypatch):
    """Private synthetic receipts exercise refusal rules; production runs real commands."""
    changed = deepcopy(work)
    monkeypatch.setattr(c, "verify_measured", lambda *_: True)
    receipts = []
    for command in changed["validation_scope_manifest"]["commands"]:
        path = tmp_path / (command["name"] + ".log")
        path.write_text("receipt-control fixture only\n")
        receipts.append(
            dict(
                name=command["name"],
                command_argv=command["argv"],
                passed=True,
                normal_exit=True,
                timed_out=False,
                exit_code=0,
                started_monotonic_ns=changed["frozen_ns"],
                ended_monotonic_ns=changed["frozen_ns"],
                scope_sha256=c.canonical_hash(changed["validation_scope_manifest"]),
                log_path=str(path),
                log_sha256=c.reference(path)["sha256"],
            )
        )
    changed["validation"] = dict(receipts=receipts, coverage=dict(complete=True))
    return changed


@pytest.mark.parametrize(
    "kind",
    ["valid", "missing", "stale", "failed", "timeout", "log", "command", "duplicate", "code"],
)
def test_current_receipts(work, tmp_path, monkeypatch, kind):
    """SCENARIO-REPORT-9096-V776-VALIDATION: affected failures disqualify readiness."""
    changed = qualify(work, tmp_path, monkeypatch)
    c.validation.mutate_receipt(changed, kind)
    value = c.build(changed)
    assert value["contract_ready_score"] == int(kind == "valid")
    assert value["verdict_class"] == ("null" if kind == "valid" else "disqualified")


def test_foreign_cli(work, tmp_path):
    """SCENARIO-REPORT-9096-V776: actual private success/block/replay/tamper paths."""
    source, output = tmp_path / "source.json", tmp_path / (c.NAME + ".json")
    c.atomic_json(source, work)
    result = child(tmp_path, "--from-work", source, "--output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert child(tmp_path, "--cold-replay", output).returncode == 0
    value = json.loads(output.read_bytes())
    value["contract_ready_score"] = 99
    value["reproducibility_checksum"] = c.checksum(value)
    c.atomic_json(output, value)
    assert child(tmp_path, "--cold-replay", output).returncode == 1
    blocked = c.collect(
        tmp_path / "absent", tmp_path / "blocked-raw", tmp_path / "private", "20261115"
    )
    c.atomic_json(source, blocked)
    output = tmp_path / "blocked" / (c.NAME + ".json")
    assert child(tmp_path, "--from-work", source, "--output", output).returncode == 0
    assert child(tmp_path, "--cold-replay", output).returncode == 0
    assert json.loads(output.read_bytes())["verdict_class"] == "blocked"
    assert child(tmp_path, "--date", "bad").returncode == 2
    assert child(tmp_path, "--from-work", source).returncode == 2
    assert child(tmp_path, "--cold-replay", tmp_path / "absent").returncode == 1


@pytest.mark.parametrize("kind", ["valid", "reduction", "scope", "operand", "aggregate", "pin"])
def test_replay(work, tmp_path, kind):
    """REQ-REPORT-9096-V776: rehashing an envelope cannot invent valid evidence."""
    changed = deepcopy(work)
    changed["raw"] = str(tmp_path)
    if kind == "reduction":
        changed["reduction"]["contract"]["ready"] = False
    elif kind == "scope":
        changed["validation_scope_manifest"]["statement_percent"] = 99
    elif kind == "pin":
        retained = tmp_path / "retained"
        shutil.copytree(Path(changed["retained"]), retained)
        changed["retained"] = str(retained)
        (retained / c.QUEUE).write_text("forged\n")
    c.atomic_json(tmp_path / "work.json", changed)
    value = c.build(changed)
    if kind == "operand":
        value["raw_shard_hashes"][0]["sha256"] = "sha256:forged"
    elif kind == "aggregate":
        value["completed_count"] = 1
    value["reproducibility_checksum"] = c.checksum(value)
    output = tmp_path / (c.NAME + ".json")
    c.atomic_json(output, value)
    assert c.replay(output)["passed"] == (kind == "valid")


def test_immutable_protocol(work, tmp_path):
    """REQ-REPORT-9096-V776: frozen protocols cannot be overwritten after inspection."""
    changed = deepcopy(work)
    changed["evidence_raw"] = str(tmp_path)
    c.atomic_json(tmp_path / "frozen_protocols/static.json", {})
    with pytest.raises(ValueError, match="immutable_protocol_drift"):
        c.reduce(changed)


def test_orchestration(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-9096-V776-VALIDATION: every frozen command runs once."""
    calls = []
    monkeypatch.setattr(
        c.validation, "run_check", lambda spec, *_: calls.append(spec.name) or dict(passed=False)
    )
    monkeypatch.setattr(
        c.previous, "validate", lambda _: dict(receipts=[], coverage=dict(complete=False))
    )
    assert not c.validate(work)["coverage"]["complete"]
    changed = qualify(work, tmp_path, monkeypatch)
    monkeypatch.setattr(c, "collect", lambda *_: deepcopy(changed))
    monkeypatch.setattr(c, "validate", lambda _: changed["validation"])
    monkeypatch.setattr(c, "terminal", lambda *_: dict(passed=True))
    output = tmp_path / (c.NAME + ".json")
    assert c.main(["--output", str(output)]) == 0
    assert calls == ["global_python_suite"]
    assert json.loads(output.read_bytes())["contract_ready_score"] == 1
    assert not json.loads(output.read_bytes())["repository_health"]["passed"]
    monkeypatch.setattr(c, "validate", lambda _: dict(receipts=[], coverage=dict(complete=False)))
    assert c.main(["--output", str(output)]) == 1
    monkeypatch.setattr(c.time, "monotonic_ns", lambda: changed["start_ns"] + 3301_000_000_000)
    with pytest.raises(TimeoutError, match="measurement_deadline"):
        c.main(["--output", str(output)])
    times = iter([changed["start_ns"], changed["start_ns"] + 4801_000_000_000])
    monkeypatch.setattr(c.time, "monotonic_ns", lambda: next(times))
    with pytest.raises(TimeoutError, match="hard_deadline"):
        c.main(["--output", str(output)])


def test_owned_validation_delegation(work, monkeypatch):
    """REQ-REPORT-9096-V776: retain the shared coverage and normal-exit policy."""
    calls = []
    monkeypatch.setattr(c.validation, "run_check", lambda spec, *_: calls.append(spec.name) or {})
    monkeypatch.setattr(c.previous, "measured", lambda *_: dict(complete=True, rows=[]))
    assert c.validate(work)["coverage"] == dict(complete=True)
    assert calls == [r["name"] for r in work["validation_scope_manifest"]["commands"]]


def test_missing_field_and_exact_encoding(tmp_path):
    """REQ-REPORT-9096-V776: missing keys are distinct from zero; brace encoding is exact."""
    root = tree(tmp_path / "root")
    queue = yaml.safe_load((root / c.ACTIVE).read_bytes())
    tasks = c.parse_design((root / c.DESIGN).read_text(), milestone=c.MILESTONE)[1]
    for encoded in [False, True]:
        for actual, expected in zip(queue["tasks"], tasks, strict=True):
            actual["prompt"] = c.encode(expected["prompt"]) if encoded else expected["prompt"]
        (root / c.ACTIVE).write_text(yaml.safe_dump(queue))
        assert c.authority(root)["ready"]
    c.atomic_json(root / "results" / (c.NAME + ".json"), dict(experiment_id=9096))
    assert c.authority(root)["ready"]
    (root / c.ACTIVE).write_text("tasks: []\n")
    assert c.authority(root)["checks"][-1]["observed"] == "missing_field"
