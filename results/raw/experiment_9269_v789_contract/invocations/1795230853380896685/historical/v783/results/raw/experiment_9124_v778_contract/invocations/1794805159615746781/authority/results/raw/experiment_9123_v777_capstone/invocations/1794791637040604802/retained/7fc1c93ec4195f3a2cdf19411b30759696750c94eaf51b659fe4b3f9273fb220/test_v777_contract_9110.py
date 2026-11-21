"""REQ-REPORT-9110-V777: authenticated methods are administrative evidence only."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from carnot.reporting import v777_contract as c


def tree(root):
    """Private scheduling copies allow rejection controls without changing the queue."""
    for name in c.AUTHORITIES:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((c.ROOT / name).read_bytes())
    return root


def test_authority(tmp_path):
    """SCENARIO-REPORT-9110-V777: staged and active complete objects both qualify."""
    root = tree(tmp_path / "root")
    assert c.authority(root)["ready"]
    queue = yaml.safe_load((root / c.ACTIVE).read_bytes())
    if queue["milestone"] != c.MILESTONE:
        queue = yaml.safe_load((root / c.STAGED).read_bytes())
    (root / c.ACTIVE).write_text(yaml.safe_dump(queue))
    assert c.authority(root)["comparison_mode"] == "activated"
    tasks = c.parse_design((root / c.DESIGN).read_text(), milestone=c.MILESTONE)[1]
    for encoded in [False, True]:
        for actual, expected in zip(queue["tasks"], tasks, strict=True):
            actual["prompt"] = c.encode(expected["prompt"]) if encoded else expected["prompt"]
        (root / c.ACTIVE).write_text(yaml.safe_dump(queue))
        assert c.authority(root)["ready"]
    c.atomic_json(root / "results" / (c.NAME + ".json"), dict(experiment_id=9110))
    assert c.authority(root)["ready"]
    (root / c.STAGED).write_text(yaml.safe_dump(queue))
    (root / c.ACTIVE).write_text("milestone: 2026.11.776\ntasks: []\n")
    assert c.authority(root)["comparison_mode"] == "staged"
    (root / c.STAGED).unlink()
    assert not c.authority(root)["ready"]
    (root / c.ACTIVE).write_text("tasks: []\n")
    assert c.authority(root)["checks"][-1]["observed"] == "missing_field"
    assert not c.authority(tmp_path / "absent")["ready"]


@pytest.mark.parametrize(
    "kind", ["prompt", "gate", "order", "title", "design", "used", "completed", "model", "count"]
)
def test_authority_refuses_drift(tmp_path, kind):
    """REQ-REPORT-9110-V777: titles cannot hide prompt, gate or collision drift."""
    root = tree(tmp_path / "root")
    source = (
        c.ACTIVE
        if yaml.safe_load((root / c.ACTIVE).read_bytes())["milestone"] == c.MILESTONE
        else c.STAGED
    )
    queue = yaml.safe_load((root / source).read_bytes())
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
        c.atomic_json(root / "results/experiment_9111_other.json", {})
    elif kind == "completed":
        with (root / c.LOG).open("a") as stream:
            stream.write(f"| date | {queue['tasks'][0]['title'][:50].strip()} | OK | old |\n")
    elif kind == "model":
        queue["tasks"][0]["MODEL_SPECS"] = [{"hf_id": "invented"}]
    else:
        queue["tasks"].pop()
    (root / source).write_text(yaml.safe_dump(queue))
    assert not c.authority(root)["ready"]


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """Real source authentication runs once; controls change only private envelopes."""
    path = tmp_path_factory.mktemp("contract9110")
    return c.collect(c.ROOT, path / "raw", path / "private", "20261115")


def test_preserved_outcomes_and_protocols(work):
    """REQ-REPORT-9110-V777: original roles and failed science survive new scheduling."""
    assert work["reduction"], [
        r for r in work["preconditions_checked"]["checks"] if not r["passed"]
    ]
    value = c.build(work)
    assert len(value["full_task_objects"]) == len(value["previous_outcome_rows"]) == 14
    assert value["canonical_task_hash"] == c.canonical_hash(value["full_task_objects"])
    assert (
        value["MODEL_SPECS"] == []
        and value["current_llm_invocations"] == value["generated_tokens"] == 0
    )
    assert value["requested_inference_substrate_class"] == "no_model_load"
    assert value["independent_generalization_score"] == value["effective_independent_groups"] == 0
    assert value["archive_lag"]["planning_archive_end"] == "2026.11.775"
    assert (
        value["archive_lag"]["observed_archive_end"]
        == yaml.safe_load((c.ROOT / "research-complete.yaml").read_bytes())["milestones"][-1]["id"]
    )
    rows = {r["experiment_id"]: r for r in value["previous_outcome_rows"]}
    assert rows[9097]["readiness"]["source_ready_score"] == 1
    assert rows[9100]["primary_status"] == "missing_file" and rows[9100]["scheduler_alternative"]
    assert rows[9107]["verdict_class"] == "disqualified" and not rows[9107]["qualified"]
    assert rows[9109]["primary_status"] == "present"
    assert value["preserved_results"]["runtime_summary"]["embedding_pass_count"] == 4
    human = value["preserved_results"]["human_summary"]
    assert len(human["rows"]) == 900 and human["question_count"] == 300
    assert human["role_counts"] == dict(TRAIN=450, CAL=150, TEST=300)
    assert len(value["protocol_references"]) == 5
    for key in [
        "source_manifest_reference",
        "evaluator_label_reference",
        "shard_a_roster_reference",
        "shard_b_roster_reference",
        "static_protocol_reference",
        "online_protocol_reference",
        "energy_protocol_reference",
    ]:
        ref = value[key]
        assert c.reference(Path(ref["path"]))["sha256"] == ref["sha256"]
    assert "logsumexp" in Path(value["energy_protocol_reference"]["path"]).read_text()
    assert len(value["deferred_graph_bounds"]["seven_bound_rows"]) == 7
    assert len(value["deferred_graph_bounds"]["ten_bound_rows"]) == 10
    assert (
        value["preserved_results"]["original_exp8435_null"]["honest_verdict"]
        == "complete_null_v726_capacity_benefit_audit"
    )


def test_missing_inputs(tmp_path):
    """SCENARIO-REPORT-9110-V777: unchanged absence is terminal blocked, never partial."""
    work = c.collect(tmp_path / "absent", tmp_path / "raw", tmp_path / "private", "20261115")
    value = c.build(work)
    assert value["verdict_class"] == "blocked" and value["censored_count"] == 14
    assert value["honest_verdict"].startswith("complete_blocked_")
    assert any(r["observed"] == "missing_file" for r in value["gate_check_summary"])
    assert (
        value["contract_ready_score"]
        == value["source_ready_score"]
        == value["online_methods_ready_score"]
        == 0
    )


def child(tmp_path, *args):
    """The actual CLI must import and replay outside the checkout without PYTHONPATH."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(c.ROOT / c.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("[test9110] child_before", args, flush=True)
    result = subprocess.run(
        argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
    )
    print(
        "[test9110] child_after",
        result.returncode,
        result.stdout[-500:],
        result.stderr[-500:],
        flush=True,
    )
    return result


def qualify(work, tmp_path, monkeypatch):
    """Synthetic receipt controls cannot qualify production; production runs real commands."""
    changed = deepcopy(work)
    monkeypatch.setattr(c.previous, "verify_measured", lambda *_: True)
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
    """SCENARIO-REPORT-9110-V777: failed owned checks disqualify all readiness scores."""
    changed = qualify(work, tmp_path, monkeypatch)
    c.validation.mutate_receipt(changed, kind)
    value = c.build(changed)
    assert (
        value["contract_ready_score"]
        == value["source_ready_score"]
        == value["online_methods_ready_score"]
        == int(kind == "valid")
    )
    assert value["verdict_class"] == ("null" if kind == "valid" else "disqualified")


def test_foreign_cli(work, tmp_path):
    """SCENARIO-REPORT-9110-V777: real private success/refusal/tamper/cold replay."""
    source, output = tmp_path / "source.json", tmp_path / (c.NAME + ".json")
    c.atomic_json(source, work)
    result = child(tmp_path, "--from-work", source, "--output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert child(tmp_path, "--cold-replay", output).returncode == 0
    value = json.loads(output.read_bytes())
    value["source_ready_score"] = 99
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


def test_immutable_protocol(work, tmp_path):
    """REQ-REPORT-9110-V777: observing results cannot replace frozen methods."""
    changed = dict(work, evidence_raw=str(tmp_path))
    c.atomic_json(tmp_path / "frozen_protocols/static.json", {})
    with pytest.raises(ValueError, match="immutable_protocol_drift"):
        c.reduce(changed)


def test_closure_refuses_drift(work, tmp_path):
    """REQ-REPORT-9110-V777: missing or changed immutable operands stop replay."""
    root = Path(work["retained"])
    pins = c.input_pins(root)
    assert pins[c.CAPSTONE] == c.PINS[c.CAPSTONE]
    assert any(p.endswith("human_online_9099.py") for p in pins)
    assert c.input_pins(tmp_path) == c.PINS
    value = c.build(work)
    output = tmp_path / (c.NAME + ".json")
    c.atomic_json(output, value)
    assert c.replay(output)["passed"]
    changed = deepcopy(work)
    changed["reduction"]["contract"]["tasks"][0]["prompt"] += " forged"
    changed["raw"] = str(tmp_path)
    c.atomic_json(tmp_path / "work.json", changed)
    c.atomic_json(output, c.build(changed))
    assert not c.replay(output)["passed"]


def test_gates_distinguish_absence_and_zero():
    """SCENARIO-REPORT-9110-V777: absent fields cannot masquerade as measured zero."""
    for observed in ["missing_field", "missing_file", 0]:
        row = c.check(Path("/absent"), "source_ready_score", 1, observed, "source_gate")
        assert row["observed"] == observed and not row["passed"]


def test_owned_validation_delegation(work, monkeypatch):
    """REQ-REPORT-9110-V777: reuse every frozen command and real coverage policy."""
    calls = []
    monkeypatch.setattr(c.validation, "run_check", lambda spec, *_: calls.append(spec.name) or {})
    monkeypatch.setattr(c.collection, "measured", lambda *_: dict(complete=True, rows=[]))
    assert c.validate(work)["coverage"] == dict(complete=True)
    assert calls == [r["name"] for r in work["validation_scope_manifest"]["commands"]]


def test_orchestration(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-9110-V777: publication follows validation, with hard deadlines."""
    changed = qualify(work, tmp_path, monkeypatch)
    monkeypatch.setattr(c, "collect", lambda *_: deepcopy(changed))
    monkeypatch.setattr(c, "validate", lambda _: changed["validation"])
    monkeypatch.setattr(c, "terminal", lambda *_: dict(passed=True))
    monkeypatch.setattr(c.validation, "run_check", lambda *_: dict(passed=False))
    output = tmp_path / (c.NAME + ".json")
    assert c.main(["--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["contract_ready_score"] == 1
    monkeypatch.setattr(c, "validate", lambda _: dict(receipts=[], coverage=dict(complete=False)))
    assert c.main(["--output", str(output)]) == 1


@pytest.mark.parametrize("kind", ["zero", "field", "code", "recovery", "primitive"])
def test_source_closure_controls(work, monkeypatch, kind):
    """REQ-REPORT-9110-V777: the complete original code and recovery closure is required."""
    changed = deepcopy(work)
    saved = c.historical(Path(work["retained"]))
    source = saved["values"]["exp9097-human-authority"]["value"]
    methods = saved["values"]["exp9099-online-methods"]["value"]
    if kind == "zero":
        source["source_ready_score"] = 0
    elif kind == "field":
        source.pop("source_ready_score")
    elif kind == "code":
        saved["operands"].pop(methods["code_config_hashes"][0]["path"])
    elif kind == "recovery":
        methods["exact_resume"] = False
    else:
        saved["operands"][source["static_protocol_reference"]["path"]]["sha256"] = "sha256:forged"
    monkeypatch.setattr(c, "historical", lambda _: saved)
    assert c.reduce(changed) == {}
    assert c.build(changed)["verdict_class"] == "blocked"


def test_pin_conflict(work, monkeypatch):
    """SCENARIO-REPORT-9110-V777: two authorities cannot bind different bytes to one input."""
    capstone = json.loads((c.ROOT / c.CAPSTONE).read_bytes())
    relative = str(Path(capstone["work_reference"]["path"]).relative_to(c.ROOT))
    monkeypatch.setattr(c, "PINS", dict(c.PINS, **{relative: "sha256:forged"}))
    with pytest.raises(ValueError, match="historical_pin_conflict"):
        c.input_pins(c.ROOT)


def test_nested_primitive_identity(work, monkeypatch):
    """SCENARIO-REPORT-9110-V777: nested historical reducers cannot forge a row binding."""
    saved = c.historical(Path(work["retained"]))
    source = saved["values"]["exp9097-human-authority"]["value"]
    forged = dict(source["static_protocol_reference"], sha256="sha256:forged")
    monkeypatch.setattr(c.history, "reduce", lambda envelope: c.history.primitive(envelope, forged))
    with pytest.raises(ValueError, match="primitive_identity"):
        c.reduce(deepcopy(work))
