"""REQ-VERIFY-8213 / REQ-REPORT-8213: record real boundaries without model credit."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.verify import request_recorder_8213 as e
from carnot.reporting import recorder_execution_8213 as runner


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    """SCENARIO-VERIFY-8213-SCHEDULE: authenticate original source custody."""
    return runner.inputs(e.ROOT, tmp_path_factory.mktemp("inputs-8213"))


def synthetic():
    """REQ-VERIFY-8213: synthetic operands carry no historical clocks."""
    return e.envelope(
        dict(
            source_cluster_id="fixture-source",
            source_bytes=b"source".hex(),
            answer_bytes=b"answer".hex(),
            condition="original",
        ),
        dict(
            model="fixture",
            messages=[dict(role="user", content="fixture prompt")],
            max_tokens=128,
            seed=7098213,
            temperature=0,
            stream=False,
        ),
        dict(
            model_sha256=e.key("model"),
            chat_template_sha256=e.key("template"),
            runtime_sha256=e.key("runtime"),
        ),
    )


def test_semantics_and_journal_restart(tmp_path):
    """SCENARIO-VERIFY-8213-JOURNAL: event identity is separate from semantics."""
    path = tmp_path / "events.jsonl"
    journal = e.Journal(path)
    original = synthetic()
    first = journal.issue("a", original)
    assert first["issue_sequence"] == 1 and first["status"] == "pending"
    assert first["issued_at"].endswith("+00:00") and first["issued_monotonic_ns"] > 0
    restart = e.Journal(path)
    assert list(restart.pending) == ["a"]
    end = restart.finish("a", "censored", dict(reason="restart"))
    assert end["semantic_key"] == first["semantic_key"] and not restart.pending
    second = restart.issue("b", original, parent_id="a", retry_id="a")
    assert second["semantic_key"] == first["semantic_key"] and second["issue_sequence"] == 2
    restart.finish("b", "completed", {"text": "fixture"})
    with pytest.raises(ValueError, match="duplicate"):
        restart.issue("a", original)
    with pytest.raises(ValueError, match="terminal"):
        restart.finish("b", "error", {})
    with pytest.raises(ValueError, match="status"):
        restart.finish("missing", "invented", {})
    for field in original:
        changed = deepcopy(original)
        changed[field] = "changed"
        assert e.key(changed) != e.key(original)
    assert len(e.Journal(path).events) == 4
    for field in ["seed", "max_tokens"]:
        changed = deepcopy(original)
        changed["payload"][field] += 1
        assert e.key(changed) != e.key(original)


def test_partial_tail_and_hash_drift(tmp_path):
    """SCENARIO-VERIFY-8213-JOURNAL: incomplete or changed journals fail closed."""
    path = tmp_path / "events.jsonl"
    journal = e.Journal(path)
    journal.issue("a", synthetic())
    original = path.read_bytes()
    path.write_bytes(original + b'{"partial":')
    with pytest.raises(ValueError, match="partial_tail"):
        e.Journal(path)
    assert path.read_bytes() == original + b'{"partial":'
    row = json.loads(original)
    row["envelope"]["condition"] = "changed"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="event_hash"):
        e.Journal(path)
    row = json.loads(original)
    row["sequence"] = 3
    row["event_sha256"] = e.key({k: v for k, v in row.items() if k != "event_sha256"})
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="event_order"):
        e.Journal(path)
    path.write_bytes(original)
    with pytest.raises(ValueError, match="envelope_hash"):
        journal.issue("bad", dict(synthetic(), source_sha256="wrong"))


def test_inputs_and_frozen_schedule(data, tmp_path):
    """SCENARIO-VERIFY-8213-SCHEDULE: choose identity order before current outcomes."""
    assert data["ready"] and len(data["schedule"]["rows"]) == 24
    rows = data["schedule"]["rows"]
    assert [s["source_cluster_id"] for s in rows] == sorted(s["source_cluster_id"] for s in rows)
    assert len({s["source_cluster_id"] for s in rows}) == 24
    assert all(s["envelope"]["payload"]["max_tokens"] == 128 for s in rows)
    assert all(s["transport_attempts"] == 1 for s in rows)
    assert not runner.inputs(tmp_path / "absent", tmp_path / "blocked")["ready"]
    missing = deepcopy(data["roster"][:23])
    with pytest.raises(ValueError, match="source_count"):
        e.schedule(missing, data["identity"])
    duplicate = deepcopy(data["roster"][:24])
    duplicate[1] = duplicate[0]
    with pytest.raises(ValueError, match="source_count"):
        e.schedule(duplicate, data["identity"])


def test_http_boundary_and_real_service_joins(data, tmp_path):
    """SCENARIO-VERIFY-8213-JOIN: scripted peer joins actual service signatures."""
    work = e.qualify(data, tmp_path)
    assert all(r["passed"] for r in work["fixture_rows"])
    assert work["non_generation_count"] == 1 and work["pending_count"] == 0
    assert {r["status"] for r in work["envelope_roundtrip_rows"]} == {
        "completed",
        "error",
        "censored",
    }
    assert all(
        r["python_probability"] == pytest.approx(r["rust_probability"], abs=1e-10)
        for r in work["service_join_rows"]
    )
    assert all(
        r["exclusive_duration_ns"] == sum(b - a for a, b in r["spans"].values())
        for r in work["service_join_rows"]
    )
    assert work["service_configuration"]["native_loaded"] is True


def test_build_replay_and_gate_failures(data, tmp_path):
    """SCENARIO-REPORT-8213-CLI: headline mutations fail independent reopening."""
    work = e.qualify(data, tmp_path / "work")
    receipts = [dict(name="owned", passed=True, normal_exit=True, exit_code=0)]
    value = runner.build(data, work, tmp_path, receipts, 0.1)
    path = tmp_path / (e.NAME + ".json")
    e.atomic_json(path, value)
    assert e.replay(path) and value["verdict_class"] == "circular_positive"
    assert value["request_recorder_ready_score"] == 1 and value["MODEL_SPECS"] == []
    for field in ["completed_count", "request_recorder_ready_score", "deployment_demand_observed"]:
        changed = deepcopy(value)
        changed[field] = 99
        changed["reproducibility_checksum"] = runner.checksum(changed)
        e.atomic_json(path, changed)
        assert not e.replay(path)
    assert not e.replay(tmp_path / "absent")
    failed = runner.build(data, work, tmp_path, [dict(passed=False)], 0.1)
    assert failed["verdict_class"] == "disqualified" and failed["request_recorder_ready_score"] == 0
    blocked = runner.inputs(tmp_path / "absent", tmp_path / "blocked")
    value = runner.build(blocked, {}, tmp_path / "blocked", [], 0.1)
    e.atomic_json(path, value)
    assert e.replay(path) and value["verdict_class"] == "blocked"
    assert value["gate_check_summary"][0]["observed"] is False


def test_real_cli_and_negative_replay(tmp_path):
    """SCENARIO-REPORT-8213-CLI: direct children use only private scratch."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    output = tmp_path / (e.NAME + ".json")
    command = [sys.executable, "-u", str(e.ROOT / e.CLI)]

    def run(args):
        return subprocess.run(
            command + args, env=env, cwd=tmp_path, capture_output=True, text=True, timeout=90
        )

    completed = run(["--fixture-e2e", str(output)])
    assert completed.returncode == 0, completed.stdout + completed.stderr
    value = json.loads(output.read_text())
    assert value["request_recorder_ready_score"] == 1
    assert run(["--cold-replay", str(output)]).returncode == 0
    work_path = Path(value["work_path"])
    work = json.loads(work_path.read_text())
    work["service_join_rows"][0]["semantic_key"] = "forged"
    work_path.chmod(0o600)
    e.atomic_json(work_path, work)
    value["work_sha256"] = e.sha256_file(work_path)
    for ref in value["raw_shard_hashes"]:
        if ref["path"] == str(work_path):
            ref["sha256"] = value["work_sha256"]
    value["reproducibility_checksum"] = runner.checksum(value)
    e.atomic_json(output, value)
    assert run(["--cold-replay", str(output)]).returncode == 1
    assert run(["--date", "20261005"]).returncode == 2
    assert run(["--fixture-e2e", str(e.ROOT / "results" / (e.NAME + ".json"))]).returncode == 2
    missing = tmp_path / "missing" / (e.NAME + ".json")
    assert run(["--fixture-e2e", str(missing), "--root", str(tmp_path / "absent")]).returncode == 0
    assert json.loads(missing.read_text())["verdict_class"] == "blocked"


def test_command_manifest_and_deadline(tmp_path):
    """REQ-REPORT-8213: freeze exact scopes and kill a timed-out process group."""
    commands = runner.validation_plan(tmp_path)
    assert any("--fail-under=100" in c.argv for c in commands)
    assert any(c.name == "scoped_spec_coverage" and "--files" in c.argv for c in commands)
    assert {"e2e015", "e2e019"}.issubset({c.name for c in commands})
    rows = runner.execute(
        [
            runner.CommandSpec("ok", (sys.executable, "-c", "print('ok')"), "private", 2),
            runner.CommandSpec(
                "timeout", (sys.executable, "-c", "import time; time.sleep(10)"), "private", 0.05
            ),
        ],
        tmp_path,
    )
    assert rows[0]["passed"] and rows[1]["timed_out"] and not rows[1]["passed"]
    assert all(r["stdout_sha256"] and r["stderr_sha256"] for r in rows)


def test_rehashed_terminal_and_identity_tampering(tmp_path):
    """SCENARIO-VERIFY-8213-JOURNAL: rehashing cannot change event state rules."""
    path = tmp_path / "journal.jsonl"
    journal = e.Journal(path)
    journal.issue("a", synthetic())
    journal.finish("a", "completed", {"text": "fixture"})
    original = [json.loads(line) for line in path.read_text().splitlines()]

    def rehash(rows):
        previous = e.key(e.SCHEMA)
        for row in rows:
            row.pop("event_sha256", None)
            row["previous_sha256"] = previous
            previous = e.key(row)
            row["event_sha256"] = previous
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    rows = deepcopy(original)
    rows[0]["semantic_key"] = "wrong"
    rehash(rows)
    with pytest.raises(ValueError, match="duplicate_or_semantic"):
        e.Journal(path)
    rows = deepcopy(original)
    rows[1]["request_id"] = "unknown"
    rehash(rows)
    with pytest.raises(ValueError, match="terminal_without_issue"):
        e.Journal(path)
    rows = deepcopy(original)
    rows[1]["result"] = {"text": "changed"}
    rehash(rows)
    with pytest.raises(ValueError, match="terminal_custody"):
        e.Journal(path)


def test_independent_primitive_mutations(data, tmp_path):
    """SCENARIO-VERIFY-8213-JOIN: changed clocks, scores or stored IDs fail replay."""
    from carnot.verify import recorder_fixtures_8213 as fixtures

    work = e.qualify(data, tmp_path / "work")
    assert fixtures.validate_work(work)
    mutations = [
        lambda w: w["envelope_roundtrip_rows"].pop(),
        lambda w: w.update(non_generation_count=99),
        lambda w: w["service_join_rows"][0].update(semantic_key="wrong"),
        lambda w: w["service_join_rows"][0]["spans"].update(encode=[2, 1]),
        lambda w: w["service_join_rows"][0].update(exclusive_duration_ns=0),
        lambda w: w["service_join_rows"][0].update(python_probability=9),
        lambda w: w["service_join_rows"][0]["store"].update(sha256="wrong"),
    ]
    for mutate in mutations:
        changed = deepcopy(work)
        mutate(changed)
        assert not fixtures.validate_work(changed)
    changed = deepcopy(work)
    row = changed["service_join_rows"][0]
    from carnot.verify import durable_batch_8159 as host

    store = host.Store(tmp_path / "other-store.json", row["head_hash"])
    store.commit(
        [
            dict(
                request_id="wrong",
                input_hash=row["semantic_key"],
                values=row["values"],
                probability=row["rust_probability"],
            )
        ]
    )
    row["store"] = e.reference(store.path)
    assert not fixtures.validate_work(changed)
    value = runner.build(data, work, tmp_path, [dict(passed=True)], 0.1)
    path = tmp_path / (e.NAME + ".json")
    changed = deepcopy(value)
    changed["raw_shard_hashes"][0]["sha256"] = "wrong"
    changed["reproducibility_checksum"] = runner.checksum(changed)
    e.atomic_json(path, changed)
    assert not e.replay(path)
    changed = deepcopy(value)
    changed["reproducibility_checksum"] = "wrong"
    e.atomic_json(path, changed)
    assert not e.replay(path)
    changed = deepcopy(value)
    changed["schedule_path"] = str(tmp_path / "schedule.json")
    frozen = deepcopy(data["schedule"])
    frozen["rows"].reverse()
    e.atomic_json(Path(changed["schedule_path"]), frozen)
    changed["reproducibility_checksum"] = runner.checksum(changed)
    e.atomic_json(path, changed)
    assert not e.replay(path)


def test_owned_failure_and_natural_runner(data, tmp_path, monkeypatch):
    """REQ-REPORT-8213: owned failures disqualify even before boundary execution."""
    failed = runner.build(data, {}, tmp_path, [dict(passed=False)], 0.1)
    assert failed["verdict_class"] == "disqualified"
    path = tmp_path / (e.NAME + ".json")
    e.atomic_json(path, failed)
    assert e.replay(path)
    execute = runner.execute
    monkeypatch.setattr(
        runner,
        "validation_plan",
        lambda private: [
            runner.CommandSpec(
                "private_test_import",
                (sys.executable, "-c", "print('private test')"),
                "private_test",
                5,
            )
        ],
    )

    def private_execute(commands, raw):
        if commands[0].name == "repository_health_once":
            commands = [
                runner.CommandSpec(
                    "private_health_fixture",
                    (sys.executable, "-c", "print('synthetic health fixture')"),
                    "private_test_only",
                    5,
                )
            ]
        return execute(commands, raw)

    monkeypatch.setattr(runner, "execute", private_execute)
    assert runner.main(["--output", str(tmp_path / "natural" / (e.NAME + ".json"))]) == 0


def test_malformed_required_input(tmp_path, monkeypatch):
    """REQ-REPORT-8213: malformed input is a blocked operand, never invented data."""
    path = tmp_path / runner.UPSTREAM
    path.parent.mkdir(parents=True)
    path.write_text("{broken")
    blocked = runner.inputs(tmp_path, tmp_path / "blocked")
    assert (
        not blocked["ready"]
        and blocked["checks"][-1]["artifact_field"] == "required_operand_structure"
    )


@pytest.mark.parametrize(
    "operand", ["fit_primary", "source_plan", "service_primary", "service_config", "library"]
)
def test_authentication_stops_dependent_work(data, tmp_path, monkeypatch, operand):
    """REQ-REPORT-8213: hash drift stops at its exact owned prerequisite."""
    original = e.sha256_file
    selected = {
        "fit_primary": e.ROOT / runner.UPSTREAM,
        "source_plan": next(
            Path(r["path"]) for r in data["refs"] if Path(r["path"]).name == "source_plan.json"
        ),
        "service_primary": e.ROOT / "results/experiment_8174_v706_complete_request_cost.json",
        "service_config": next(
            Path(r["path"]) for r in data["refs"] if Path(r["path"]).name == "input_data.json"
        ),
        "library": Path(data["library"]["path"]),
    }[operand]
    monkeypatch.setattr(
        e, "sha256_file", lambda path: "sha256:drift" if Path(path) == selected else original(path)
    )
    blocked = runner.inputs(e.ROOT, tmp_path / "blocked")
    assert not blocked["ready"]
    assert not (tmp_path / "blocked" / "schedule.json").exists()
    assert any(not c["passed"] and c["path"] == str(selected) for c in blocked["checks"])
