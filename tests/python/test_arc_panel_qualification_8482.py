"""REQ-REPORT-8482 / REQ-VERIFY-8482: missing authority cannot become live evidence."""

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
import pytest

from carnot.reporting import arc_generalization_panel_8468 as old
from carnot.reporting import arc_panel_qualification_8482 as p
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.v709_execution import child
from carnot.reporting.v721_capstone_evidence import frozen_inputs
from test_arc_supervisor_live_panel_8384 import work


def original_refs():
    """Use independently pinned original inputs so today's authority cannot rewrite history."""
    primary = json.loads((old.ROOT / p.PRIOR).read_bytes())
    return json.loads(Path(primary["work_reference"]["path"]).read_bytes())["source_input_receipts"]


def test_original_registry_rehashed_rejection(tmp_path):
    """SCENARIO-REPORT-8482-REJECTION: original line350 rejects internally consistent drift."""
    with frozen_inputs(original_refs(), tmp_path / "writes"):
        w = work(tmp_path)
        w["panel"] = old.freeze([], [])
        w["preconditions_checked"] = {"task_sha256": old.TASK_PIN}
        value = old.build(w, [], tmp_path, tmp_path / (old.NAME + ".json"))
        with old.bindings():
            assert old._REPLAY(value), "outer seals and summary must be consistent"
        assert not old.replay(value), "independent registry must reject private panel"
        assert value["reproducibility_checksum"] == canonical_hash(w)
        print("[exp8482] original line350 registry rejection qualified", flush=True)


def current_work(tmp_path):
    """Missing current authority remains a bound operand, not a constructed observation."""
    w = work(tmp_path)
    checks, hashes, resources = p.preconditions(tmp_path / "private", p.ROOT)
    registry = yaml.safe_load((p.ROOT / "ops/arc_solve_registry.yaml").read_bytes())["games"]
    w.update(panel=old.freeze([], registry), failures=checks, preconditions_checked=resources)
    w["source_input_receipts"] = [
        p.snapshot(Path(name), tmp_path / "inputs", digest)
        for name, digest in hashes.items()
        if digest
    ]
    return w


def test_current_authority_and_replay(tmp_path):
    """SCENARIO-REPORT-8482-CUSTODY: blocked work still authenticates independent panel operands."""
    w = current_work(tmp_path)
    value = p.build(w, [dict(passed=True, scope="owned")], tmp_path, tmp_path / (p.NAME + ".json"))
    assert value["verdict_class"] == "blocked" and value["experiment_id"] == 8482
    assert value["milestone"] == "2026.10.729" and value["intended_count"] == 16
    assert value["completed_count"] == 0 and value["independent_count"] == 0
    assert all(row["status"] == "unstarted" for row in value["rows"])
    assert value["policy_model_calls"] == 0 and value["MODEL_SPECS"] == []
    assert value["panel_execution_ready_score"] == value["arc_support_ready_score"] == 0
    assert set(value) <= set(value["field_principles"])
    assert p.replay(value)
    assert not p.replay({})
    changed = deepcopy(w)
    changed["panel"] = old.freeze([], [])
    candidate = p.build(changed, [], tmp_path, tmp_path / (p.NAME + ".json"))
    assert not p.replay(candidate)
    candidate["completed_count"] = 1
    assert not p.replay(candidate)
    checks, _, resources = p.preconditions(tmp_path / "missing", tmp_path)
    assert checks and resources["task_sha256"] is None


def test_plan_and_cli(tmp_path):
    """SCENARIO-VERIFY-8482-VALIDATION: collect exact bounded shards and exercise real CLI errors."""
    specs = p.plan(tmp_path / "plan")
    assert not any(s["scope"] == "global" for s in specs)
    original = [s for s in specs if s["name"].startswith("historical_8468_")]
    nodes = [n for s in original for n in s["collected_node_ids"]]
    assert len(nodes) == len(set(nodes)) == 15
    cli = [str(p.ROOT / ".venv/bin/python"), "-u", str(p.ROOT / p.CLI)]
    output = tmp_path / (p.NAME + ".json")
    for name, args, expected in [
        ("private", ["--private-e2e", "--output", str(output)], 0),
        ("valid", ["--cold-replay", str(output)], 0),
        ("missing", ["--cold-replay", str(tmp_path / "absent")], 1),
        ("wrong_date", ["--date", "20261012"], 2),
        ("public_fixture", ["--private-e2e"], 2),
    ]:
        assert child(name, [*cli, *args], tmp_path / "cli", expected=expected, deadline=120)[
            "passed"
        ]


def test_run_adapter(tmp_path):
    """SCENARIO-REPORT-8482-PANEL: invocation bindings restore the original collector after use."""
    with patch.object(p, "_RUN", return_value=0) as run:
        assert p.run(tmp_path / (p.NAME + ".json"), tmp_path) == 0
        run.assert_called_once()
    assert old.NAME == "experiment_8468_v728_arc_generalization_panel"


def test_historical_cli_shard(tmp_path):
    """SCENARIO-REPORT-8482-REJECTION: real original-date child runs sealed original assertions."""
    shard = tmp_path / "shard.json"
    atomic_json(shard, [old.TEST + "::test_frozen_four_games"])
    receipt = child(
        "historical_control",
        [
            str(p.ROOT / ".venv/bin/python"),
            "-u",
            str(p.ROOT / p.CLI),
            "--historical-shard",
            str(shard),
        ],
        tmp_path / "history",
        deadline=120,
    )
    assert receipt["passed"]
    assert sha256_file(p.ROOT / p.PRIOR) == p.PRIOR_PIN


def test_replay_requires_external_gates(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8482-CUSTODY: removing bound missing-authority checks cannot grant readiness."""
    w = current_work(tmp_path)
    w["failures"] = []
    candidate = p.build(w, [], tmp_path, tmp_path / (p.NAME + ".json"))
    assert not p.replay(candidate)
    monkeypatch.setattr(p, "_REPLAY", lambda value: True)
    assert not p.replay({})


def test_validation_guard_failures(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8482-VALIDATION: failed collection and independent pin drift fail closed."""
    from carnot.reporting import arc_panel_validation_8482 as validation

    monkeypatch.setattr(validation, "child", lambda *args, **kwargs: dict(passed=False))
    with pytest.raises(ValueError, match="test_collection_failed"):
        validation.collect(tmp_path, p.TEST)
    log = tmp_path / "collection.stdout"
    log.write_text(p.TEST + "::duplicate\n" + p.TEST + "::duplicate\n")
    monkeypatch.setattr(
        validation, "child", lambda *args, **kwargs: dict(passed=True, stdout_path=str(log))
    )
    with pytest.raises(ValueError, match="empty_or_duplicate"):
        validation.collect(tmp_path, p.TEST)
    real_hash = validation.sha256_file
    primary = json.loads((p.ROOT / p.PRIOR).read_bytes())
    w = json.loads(Path(primary["work_reference"]["path"]).read_bytes())
    for path, expected in [
        (p.ROOT / p.PRIOR, "historical_primary_pin"),
        (Path(primary["work_reference"]["path"]), "historical_work_pin"),
        (Path(w["source_input_receipts"][0]["snapshot_path"]), "historical_snapshot_pin"),
    ]:
        monkeypatch.setattr(
            validation,
            "sha256_file",
            lambda operand, bad=path: "bad" if operand == bad else real_hash(operand),
        )
        with pytest.raises(ValueError, match=expected):
            validation.historical(tmp_path / "request.json")
