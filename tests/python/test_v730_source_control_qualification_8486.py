"""REQ-REPORT-8486, REQ-VERIFY-8486: source qualification preserves original evidence."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import v730_source_control_qualification as e
from carnot.reporting import v730_source_control_runner as r
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    """Real original producers supply controls; no current model or labels are opened."""
    raw = tmp_path_factory.mktemp("v730-controls")
    return e.measure(e.ROOT, raw), raw


def test_controls_and_current_authority(measured):
    """SCENARIO-REPORT-8486-CONTROLS: retain original roles, capacities and source custody."""
    work, raw = measured
    assert not work["failures"]
    assert work["authority"]["canonical_tasks_sha256"] == e.methods.TASKS_PIN
    assert work["methods_consumer"]["passed"] and work["source_consumer"]["passed"]
    assert len(work["panel"]["rows"]) == 4500
    assert {k: len(v) for k, v in work["panel"]["question_roles"].items()} == dict(
        fit=150, tune=50, test=100
    )
    assert work["panel"]["control_parameter_counts"] == dict(
        zip(e.legacy.ARMS, [0, 2, 13, 109, 109], strict=True)
    )
    upstream, imported = (e.read(ref) for ref in work["upstream_refs"])
    heads = e.read(dict(path=imported["frozen_heads_path"], sha256=imported["frozen_heads_sha256"]))
    changed = deepcopy(upstream)
    changed["rows"][0]["x"][1:] = [999.0] * 8
    a = e.legacy.freeze(upstream, heads, e.methods.protocol(), [])
    b = e.legacy.freeze(changed, heads, e.methods.protocol(), [])
    for arm in e.legacy.ARMS[:3]:
        assert [x["x"] for x in a["rows"] if x["arm"] == arm] == [
            x["x"] for x in b["rows"] if x["arm"] == arm
        ]
    value = e.build(work, [dict(passed=True)], raw, raw / (e.NAME + ".json"))
    assert value["source_controls_ready_score"] == 1
    assert value["verdict_class"] == "circular_positive"
    assert value["independent_count"] == 300
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )
    assert value["inference_substrate_class"] == "no_model_load" and value["MODEL_SPECS"] == []
    assert value["experiment_id"] == 8486 and value["run_date"] == "20261013"


def test_original_pin_rejects_self_consistent_forgery(tmp_path):
    """SCENARIO-REPORT-8486-PIN: matching local hashes cannot replace the independent pin."""
    evidence = e.producer_pin_control(tmp_path)
    assert evidence["valid_original_passed"] and evidence["forged_producer_rejected"]
    assert evidence["local_hashes_consistent"]
    assert e.read(evidence["forged_producer_reference"])["forged_pin_fixture"] is True
    assert evidence["original_terminal_reference"]["sha256"] == e.ORIGINAL_PIN


def test_failure_classes_and_missing_inputs(measured, tmp_path):
    """REQ-REPORT-8486: external absence blocks while actual owned validation disqualifies."""
    work, raw = measured
    failed = e.build(work, [dict(passed=False)], raw, raw / (e.NAME + ".json"))
    assert failed["source_controls_ready_score"] == 0 and failed["verdict_class"] == "disqualified"
    missing = e.measure(tmp_path / "absent", tmp_path / "missing")
    value = e.build(
        missing, [dict(passed=True)], tmp_path / "missing", tmp_path / (e.NAME + ".json")
    )
    assert value["verdict_class"] == "blocked" and value["gate_check_summary"]
    assert value["censored_count"] == 4500
    assert not e.replay(tmp_path / "absent.json")


def test_private_cli_and_cold_tampering(tmp_path):
    """SCENARIO-VERIFY-8486-CLI: real children preserve terminal and rejection evidence."""
    output = tmp_path / "results" / (e.NAME + ".json")
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    cases = [
        (["--date", "20261013", "--private-control", "--output", str(output)], 0),
        (["--cold-replay", str(output)], 0),
        (["--cold-replay", str(tmp_path / "missing")], 1),
        (["--deliberate-error"], 7),
        (["--date", "20261012"], 2),
        (["--private-control"], 2),
    ]
    for i, (args, expected) in enumerate(cases):
        result = child(f"cli-{i}", argv + args, tmp_path / "logs", expected=expected, deadline=180)
        assert result["passed"], Path(result["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["source_controls_ready_score"] == 1
    assert Path(value["terminal_validation_sidecar_path"]).is_file()
    changed = deepcopy(value)
    changed["source_controls_ready_score"] = 999
    changed["reproducibility_checksum"] = e.checksum(changed)
    atomic_json(output, changed)
    assert not e.replay(output)
    work_path = Path(value["primitive_reference"]["path"])
    work = json.loads(work_path.read_bytes())
    work["panel"]["rows"][0]["x"][0] = 0.7
    atomic_json(work_path, work)
    atomic_json(output, e.build(work, value["validation_receipts"], work_path.parent, output))
    assert not e.replay(output)


def test_manifest_has_exact_collection_and_coverage(tmp_path):
    """REQ-VERIFY-8486: every historical node belongs to exactly one bounded shard."""
    plan = r.manifest(tmp_path)
    assert {"coverage_report", "strict_mypy", "spec_coverage"} <= {p["name"] for p in plan}
    for collection in [p for p in plan if p["name"].startswith("collect_")]:
        nodes = collection["collected_node_ids"]
        if nodes[0].startswith(e.TEST):
            continue
        executed = [
            n
            for p in plan
            if not p["name"].startswith("collect_")
            for n in p.get("collected_node_ids", [])
            if n.split("::")[0] == nodes[0].split("::")[0]
        ]
        assert executed == nodes and len(executed) == len(set(executed))


def test_methods_consumer_and_owned_pin_failure(tmp_path, monkeypatch):
    """REQ-REPORT-8486: independent consumer and producer rejection failures forbid readiness."""
    monkeypatch.setattr(e, "reader_receipt", lambda *a, **kw: dict(passed=False))
    work = e.measure(e.ROOT, tmp_path / "consumer")
    assert any(
        g["artifact_field"] == "methods_and_source_inputs_ready_score" for g in work["failures"]
    )
    assert e.replay(tmp_path / "nonexistent") is False
    monkeypatch.undo()
    monkeypatch.setattr(
        e,
        "producer_pin_control",
        lambda *_: dict(
            valid_original_passed=True, forged_producer_rejected=False, local_hashes_consistent=True
        ),
    )
    work = e.measure(e.ROOT, tmp_path / "owned")
    value = e.build(work, [dict(passed=True)], tmp_path / "owned", tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "disqualified" and value["source_controls_ready_score"] == 0


def test_original_authority_drift_and_worker_controls(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8486-CLI: original worker rejects missing requests and preserves exits."""
    monkeypatch.setattr(e, "sha256_file", lambda _: "sha256:" + "0" * 64)
    with pytest.raises(ValueError, match="original_authority_pin"):
        with e.original_binding(tmp_path / "bad"):
            pass
    monkeypatch.undo()
    assert r.main(["--original-exec", str(e.ROOT / e.legacy.CLI), "--deliberate-error"]) == 7
    empty = tmp_path / "empty.py"
    empty.write_text("print('normal original worker exit', flush=True)\n")
    assert r.original_worker([str(empty)], tests=False) == 0
    with pytest.raises(SystemExit) as exc:
        r.main(["--history-request", str(tmp_path / "absent")])
    assert exc.value.code == 2


def test_private_blocked_cli(tmp_path):
    """SCENARIO-VERIFY-8486-CLI: unchanged external absence remains a terminal blocked result."""
    output = tmp_path / "blocked" / (e.NAME + ".json")
    result = child(
        "blocked",
        [
            sys.executable,
            "-u",
            str(e.ROOT / e.CLI),
            "--private-control",
            "--root",
            str(tmp_path / "absent"),
            "--output",
            str(output),
        ],
        tmp_path / "logs",
        deadline=180,
    )
    assert result["passed"], Path(result["stderr_path"]).read_text()
    assert json.loads(output.read_bytes())["verdict_class"] == "blocked"
