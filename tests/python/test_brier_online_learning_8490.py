"""REQ-REPORT-8490 and REQ-VERIFY-8490: unavailable qualification prevents learning."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import brier_online_learning_8490 as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child


def test_pinned_roster_preserves_all_intended_units():
    """REQ-REPORT-8490: dependent seeds and siblings cannot become independent samples."""
    rows, retention = e.roster()
    assert len(rows) == 6000 and len({r["unit_id"] for r in rows}) == 6000
    assert len({r["question_hash"] for r in rows}) == 100
    assert {r["arm"] for r in rows} == set(e.ARMS)
    assert {r["seed"] for r in rows} == set(e.SEEDS)
    assert len(retention) == 15000 and {r["boundary"] for r in retention} == set(range(5))
    assert len({r["question_hash"] for r in retention}) == 50
    assert all(r["probability"] is None and r["missing_reason"] for r in rows + retention)


@pytest.mark.parametrize(
    "field,observed",
    [
        ("objective_kernel_ready_score", 0),
        ("flagged_adversarial", True),
        ("required_checks_passed", False),
    ],
)
def test_upstream_qualification_is_conjunctive(field, observed):
    """SCENARIO-REPORT-8490-BLOCKED: reported kernel readiness alone cannot admit labels."""
    value = dict(
        objective_kernel_ready_score=1, flagged_adversarial=False, required_checks_passed=True
    )
    assert not e.qualification_failures(Path("input.json"), "sha256:pin", value)
    value[field] = observed
    failures = e.qualification_failures(Path("input.json"), "sha256:pin", value)
    assert failures[0]["field"] == field and failures[0]["observed"] == observed
    assert failures[0]["upstream"] == e.UPSTREAM_TASK
    del value[field]
    assert any(
        r["observed"] is None
        for r in e.qualification_failures(Path("input.json"), "sha256:pin", value)
    )


@pytest.fixture()
def absent_work(tmp_path):
    """SCENARIO-REPORT-8490-BLOCKED: real missing authority is a recorded external absence."""
    return e.measure(tmp_path / "missing", tmp_path / "raw", tmp_path / "scratch")


def test_missing_and_owned_failure_have_different_verdicts(absent_work, tmp_path):
    """SCENARIO-REPORT-8490-BLOCKED: a failed owned child disqualifies an otherwise honest block."""
    output = tmp_path / (e.NAME + ".json")
    good = [child("fixture", [sys.executable, "-c", "print('actual receipt')"], tmp_path / "logs")]
    value = e.build(absent_work, good, output)
    assert value["verdict_class"] == "blocked" and value["objective_stream_ready_score"] == 0
    assert (
        value["completed_count"] == value["failed_count"] == 0 and value["censored_count"] == 6000
    )
    assert value["MODEL_SPECS"] == [] and value["current_llm_invocations"] == 0
    assert value["independent_count"] == value["coefficient_change_count"] == 0
    assert any(r["observed"] is None for r in value["gate_check_summary"])
    bad = child("failure", [sys.executable, "-c", "raise SystemExit(7)"], tmp_path / "logs")
    assert e.build(absent_work, [bad], output)["verdict_class"] == "disqualified"
    assert e.build(absent_work, [], output)["verdict_class"] == "disqualified"


def test_replay_rejects_repaired_envelopes(absent_work, tmp_path):
    """SCENARIO-REPORT-8490-REPLAY: primitive operands govern the terminal reduction."""
    receipt = child("fixture", [sys.executable, "-c", "print('receipt')"], tmp_path / "logs")
    output = tmp_path / (e.NAME + ".json")
    value = e.build(absent_work, [receipt], output)
    atomic_json(output, value)
    assert e.replay(output)
    for key in ("objective_stream_ready_score", "rows", "gate_check_summary"):
        changed = deepcopy(value)
        changed[key] = 1 if key.endswith("score") else []
        changed["reproducibility_checksum"] = e.checksum(changed)
        atomic_json(output, changed)
        assert not e.replay(output)
    assert not e.replay(tmp_path / "absent.json")


def test_unqualified_current_producer_stays_unused(tmp_path):
    """SCENARIO-REPORT-8490-BLOCKED: actual shipped replay failure forbids numerical work."""
    work = e.measure(e.ROOT, tmp_path / "raw", tmp_path / "scratch")
    assert any(f["field"] == "cold_replay" and f["observed"] is False for f in work["failures"])
    assert any(f["field"] == "flagged_adversarial" for f in work["failures"])
    assert work["label_updates"] == 0 and work["source_refs"]
    assert any(
        r["name"] == "upstream_cold_replay" and r["actual_exit"] == 1
        for r in work["external_receipts"]
    )

    assert any(
        r["name"] == "historical_v726_replay" and r["passed"] for r in work["external_receipts"]
    )
    receipt = child(
        "owned_fixture", [sys.executable, "-c", "print('owned check')"], tmp_path / "logs"
    )
    output = tmp_path / (e.NAME + ".json")
    value = e.build(work, [receipt], output)
    atomic_json(output, value)
    assert e.replay(output)
    for field in ("flagged_adversarial", "cold_replay"):
        changed = deepcopy(work)
        changed["failures"] = [r for r in changed["failures"] if r["field"] != field]
        atomic_json(Path(work["raw"]) / "measurement.json", changed)
        atomic_json(output, e.build(changed, [receipt], output))
        assert not e.replay(output)


def test_private_cli_and_controls(tmp_path):
    """SCENARIO-VERIFY-8490-CLI: actual CLI bytes pass readers and fail tamper controls."""
    from carnot.reporting import brier_online_runner_8490 as runner

    assert runner.main(["--deliberate-error"]) == 7
    assert runner.main(["--cold-replay", str(tmp_path / "absent")]) == 1
    with pytest.raises(SystemExit):
        runner.main(["--date", "20261012"])
    with pytest.raises(SystemExit):
        runner.main(["--private-control"])
    output = tmp_path / "results" / (e.NAME + ".json")
    assert (
        runner.main(
            ["--private-control", "--root", str(tmp_path / "absent"), "--output", str(output)]
        )
        == 0
    )
    assert runner.main(["--cold-replay", str(output)]) == 0
    assert json.loads(output.read_bytes())["verdict_class"] == "blocked"


def test_manifest_collects_complete_disjoint_suites(tmp_path):
    """REQ-VERIFY-8490: scoped historical shards retain every collected assertion exactly once."""
    from carnot.reporting import brier_online_runner_8490 as runner

    plan = runner.manifest(tmp_path)
    ledger = json.loads((tmp_path / "collection.json").read_bytes())
    union = [n for r in plan for n in r.get("collected_node_ids", [])]
    assert union == ledger["node_ids"] and len(union) == len(set(union))
    assert all(r["deadline"] <= 600 for r in plan)
    assert all("--no-cov" in r["argv"] for r in plan if r.get("collected_node_ids"))


def test_changed_precondition_cannot_claim_measurement(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8490-BLOCKED: clearing the blocker requires the actual causal study."""
    monkeypatch.setattr(e, "inspect", lambda *args: dict(failures=[]))
    with pytest.raises(ValueError, match="blocking_condition_changed"):
        e.measure(e.ROOT, tmp_path / "raw", tmp_path / "scratch")


def test_environment_authority_and_sidecar_rejections(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8490-BLOCKED: independent resource, task and sidecar gates remain necessary."""
    monkeypatch.setattr(
        e.methods,
        "authority",
        lambda *args: dict(
            activated=False, tasks=[dict(id=e.TASK, deliverable="wrong")], gate_check_summary=[]
        ),
    )
    original = e.methods.base.legacy.resources
    monkeypatch.setattr(
        e.methods.base.legacy, "resources", lambda p: dict(original(p), available_memory_bytes=0)
    )
    monkeypatch.setattr(e, "read_bound_sidecar", lambda *args: dict(report=dict(passed=False)))
    monkeypatch.setattr(e.history, "custody", lambda *args: [])
    original_hash = e.sha256_file
    monkeypatch.setattr(
        e, "sha256_file", lambda p: "wrong" if p.name == "update_adapter.py" else original_hash(p)
    )

    def failed_child(name, argv, logs, **kwargs):
        return child(name, [sys.executable, "-c", "raise SystemExit(7)"], logs)

    monkeypatch.setattr(e, "child", failed_child)
    work = e.measure(e.ROOT, tmp_path / "raw", tmp_path / "scratch")
    fields = {f["field"] for f in work["failures"]}
    assert {
        "activated_task_authority",
        "private_disk_memory_tools",
        "bound_publication_validation",
        "original_authority_replay",
    } <= fields
    assert "qualified_update_adapter" in fields


@pytest.mark.parametrize("attack", ["roster", "code_hash", "saved_receipt", "log", "exit"])
def test_primitive_and_receipt_tampering(absent_work, tmp_path, attack):
    """SCENARIO-REPORT-8490-REPLAY: repaired primitive and envelope hashes cannot repair operands."""
    receipt = child("fixture", [sys.executable, "-c", "print('receipt')"], tmp_path / "logs")
    work = deepcopy(absent_work)
    if attack == "roster":
        roster = e.read(work["roster_reference"])
        roster["rows"][0]["probability"] = 0.5
        atomic_json(tmp_path / "changed_roster.json", roster)
        work["roster_reference"] = e.reference(tmp_path / "changed_roster.json")
    elif attack == "code_hash":
        work["code_refs"][0]["sha256"] = "wrong"
    elif attack == "saved_receipt":
        receipt["duration_s"] = -1
    elif attack == "log":
        Path(receipt["stdout_path"]).write_text("changed\n")
    else:
        receipt["passed"] = False
        atomic_json(Path(receipt["stdout_path"]).with_suffix(".receipt.json"), receipt)
    atomic_json(Path(work["raw"]) / "measurement.json", work)
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, e.build(work, [receipt], output))
    assert not e.replay(output)


def test_private_cli_retains_real_coverage_and_collection(tmp_path):
    """SCENARIO-VERIFY-8490-CLI: collected nodes and real coverage bytes survive private cleanup."""
    from carnot.reporting import brier_online_runner_8490 as runner

    private = tmp_path / "private"
    private.mkdir()
    nodes, _ = runner.collection.collect(private, e.TEST)
    atomic_json(private / "collection.json", dict(node_ids=nodes, shards=[nodes]))
    config = private / "coverage.ini"
    config.write_text(
        "[run]\ninclude=" + str(e.ROOT / e.CLI) + "\ndata_file=" + str(private / ".coverage") + "\n"
    )
    receipt = child(
        "genuine_cli_coverage",
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--rcfile=" + str(config),
            str(e.ROOT / e.CLI),
            "--deliberate-error",
        ],
        private / "coverage_logs",
        expected=7,
    )
    assert receipt["passed"]
    receipt = child(
        "genuine_coverage_json",
        [
            sys.executable,
            "-m",
            "coverage",
            "json",
            "--rcfile=" + str(config),
            "-o",
            str(private / "coverage.json"),
        ],
        private / "coverage_logs",
    )
    assert receipt["passed"]
    output = tmp_path / "results" / (e.NAME + ".json")
    assert runner.run(tmp_path / "absent", output, private, control=True) == 0
    artifact = json.loads(output.read_bytes())
    work = e.read(artifact["primitive_reference"])
    assert work["coverage_refs"] and e.replay(output)
    assert (Path(work["raw"]) / "collection.json").is_file()


def test_duplicate_collection_cannot_hide_assertions(tmp_path, monkeypatch):
    """REQ-VERIFY-8490: duplicate node identities cannot pass a complete shard union."""
    from carnot.reporting import brier_online_runner_8490 as runner

    monkeypatch.setattr(
        runner.collection, "collect", lambda *args: (["duplicate", "duplicate"], {})
    )
    with pytest.raises(ValueError, match="historical_collection_union"):
        runner.manifest(tmp_path)


def test_rehashed_foreign_producer_pin(absent_work, tmp_path):
    """SCENARIO-REPORT-8490-REPLAY: a valid local digest cannot replace the independent producer pin."""
    receipt = child("fixture", [sys.executable, "-c", "print('receipt')"], tmp_path / "logs")
    foreign = tmp_path / (e.upstream.NAME + ".json")
    atomic_json(
        foreign,
        dict(
            objective_kernel_ready_score=1, required_checks_passed=True, flagged_adversarial=False
        ),
    )
    work = deepcopy(absent_work)
    work["source_refs"].append(e.reference(foreign))
    atomic_json(Path(work["raw"]) / "measurement.json", work)
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, e.build(work, [receipt], output))
    assert not e.replay(output)
