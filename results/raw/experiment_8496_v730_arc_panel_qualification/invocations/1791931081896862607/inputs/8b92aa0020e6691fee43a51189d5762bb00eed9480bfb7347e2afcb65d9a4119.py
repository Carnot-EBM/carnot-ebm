"""REQ-REPORT-8496 / REQ-VERIFY-8496: private controls precede public observations."""

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from carnot.reporting import arc_generalization_panel_8468 as old
from carnot.reporting import arc_panel_qualification_8496 as p
from carnot.reporting import arc_panel_validation_8496 as validation
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.v709_execution import child
from carnot.reporting.v721_capstone_evidence import frozen_inputs
from test_arc_supervisor_live_panel_8384 import work


def test_success_then_original_rejection(tmp_path):
    """SCENARIO-REPORT-8496-REJECTION: successful authority precedes panel-only mutation."""
    receipt = validation.rejection(tmp_path)
    assert receipt["successful_replay"] and receipt["outer_replay_after_mutation"]
    assert receipt["original_rejection"] and receipt["original_line"] == 350
    assert receipt["changed_fields"] == ["panel"]
    assert receipt["historical_task_sha256"] == old.TASK_PIN
    assert sha256_file(p.ROOT / p.PRIOR) == p.PRIOR_PIN


def current_work(tmp_path):
    """Independent current authority and a missing ledger exercise custody without science."""
    w = work(tmp_path)
    checks, hashes, resources = p.preconditions(tmp_path / "private", p.ROOT)
    registry = yaml.safe_load((p.ROOT / "ops/arc_solve_registry.yaml").read_bytes())["games"]
    panel = old.freeze([], registry)
    w.update(panel=panel, failures=checks, preconditions_checked=resources)
    w["source_input_receipts"] = [
        p.snapshot(Path(name), tmp_path / "inputs", digest)
        for name, digest in hashes.items()
        if digest
    ]
    return w


def test_current_replay_and_panel_tampering(tmp_path):
    """SCENARIO-REPORT-8496-CUSTODY: outer hashes cannot replace independent metadata."""
    w = current_work(tmp_path)
    value = p.build(w, [], tmp_path, tmp_path / (p.NAME + ".json"))
    assert value["experiment_id"] == 8496 and value["milestone"] == "2026.10.730"
    assert value["intended_count"] == 16 and value["completed_count"] == 0
    assert value["MODEL_SPECS"] == [] and value["policy_model_calls"] == 0
    assert value["new_level_solves"] == value["arc_support_ready_score"] == 0
    assert set(value) <= set(value["field_principles"])
    assert p.replay(value) and not p.replay({})
    changed = deepcopy(w)
    changed["panel"]["registry_precheck"][changed["panel"]["selected_games"][0]][
        "levels_reproduced"
    ] = -1
    candidate = p.build(changed, [], tmp_path, tmp_path / (p.NAME + ".json"))
    assert not p.replay(candidate)
    value["completed_count"] = 1
    assert not p.replay(value)
    checks, _, resources = p.preconditions(tmp_path / "missing", tmp_path)
    assert checks and resources["task_sha256"] is None


def test_plan_exact_baseline_union(tmp_path):
    """SCENARIO-VERIFY-8496-VALIDATION: collect every original node before execution."""
    specs = p.plan(tmp_path / "plan")
    assert all(s["scope"] == "owned" for s in specs)
    proof = json.loads((tmp_path / "plan/coverage_shard_union.json").read_bytes())
    assert len(proof[validation.shipped.HISTORICAL]["original"]) == 15
    assert all(v["exact_union"] for v in proof.values())
    assert json.loads((tmp_path / "plan/coverage_baseline_union_receipt.json").read_bytes())[
        "passed"
    ]
    assert any(s["name"] == "original_rejection_350" for s in specs)


def test_real_cli_and_run_binding(tmp_path):
    """SCENARIO-VERIFY-8496-VALIDATION: bounded real children exercise terminal errors."""
    cli = [str(p.ROOT / ".venv/bin/python"), "-u", str(p.ROOT / p.CLI)]
    output = tmp_path / (p.NAME + ".json")
    for name, args, expected in [
        ("private", ["--private-e2e", "--output", str(output)], 0),
        ("valid", ["--cold-replay", str(output)], 0),
        ("missing", ["--cold-replay", str(tmp_path / "absent")], 1),
        ("wrong_date", ["--date", "20261012"], 2),
        ("public_fixture", ["--private-e2e"], 2),
        ("rejection", ["--qualify-rejection", str(tmp_path / "reject")], 0),
    ]:
        assert child(name, [*cli, *args], tmp_path / "cli", expected=expected, deadline=120)[
            "passed"
        ]
    with patch.object(p, "_RUN", return_value=0) as run:
        assert p.run(output, tmp_path) == 0
        run.assert_called_once()
    assert old.NAME == "experiment_8468_v728_arc_generalization_panel"


def test_historical_validation_failures(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8496-REJECTION: missing or changed original seals close qualification."""
    real = validation.sha256_file
    primary = json.loads((p.ROOT / p.PRIOR).read_bytes())
    ref = primary["work_reference"]
    w = json.loads(Path(ref["path"]).read_bytes())
    for path, expected in [
        (p.ROOT / p.PRIOR, "historical_primary_pin"),
        (Path(ref["path"]), "historical_work_pin"),
        (Path(w["source_input_receipts"][0]["snapshot_path"]), "historical_snapshot_pin"),
    ]:
        monkeypatch.setattr(
            validation,
            "sha256_file",
            lambda operand, bad=path: "bad" if operand == bad else real(operand),
        )
        with pytest.raises(ValueError, match=expected):
            validation.rejection(tmp_path)
