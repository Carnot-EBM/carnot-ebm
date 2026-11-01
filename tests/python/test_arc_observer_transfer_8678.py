"""REQ-REPORT-8678, REQ-VERIFY-8678: fresh transfer never recasts censored work."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import arc_observer_transfer_8678 as e
from carnot.reporting import arc_observer_transfer_runner_8678 as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.v709_execution import child


def panel():
    """SCENARIO-REPORT-8678-TRANSFER: twins and seeds do not multiply game N."""
    episodes = []
    for unit in e.units():
        episodes.append(
            dict(
                unit=unit,
                actions=[{"kind": 1}],
                observations=[{"level": 0}],
                termination={"error": None},
                legacy={},
                rng=[],
                torch_rng=[],
                sdk_calls=[],
                reset_requests=[],
                final_state={"level": 0},
                observer_issues=[],
                model_tripwire_attempts=[],
                llm_calls=0,
                numerical_head_calls=1,
                delivery_rows=[dict(action_index=1, owner="is_done", status="canonical")],
                reset_provenance=dict(reset_request_rows=[], callback_sequence_rows=[], issues=[]),
                recorder_error_metadata=[],
                duration_s=0.1,
                sdk_identity={"official": True},
                observer_rows=[
                    dict(
                        redirect_id="r",
                        arm="diversify",
                        horizon_status="censored",
                        end_reason="reset",
                        observed_horizon=1,
                        overlapping_redirect_ids=[],
                        application_index=1,
                    )
                ],
            )
        )
    return episodes


def test_gate_states_and_qualified_sources(tmp_path, monkeypatch):
    """REQ-VERIFY-8678: each unavailable operand has its own diagnostic state."""
    for (name, pin), field in zip(e.PINS.items(), e.GATE_FIELDS, strict=True):
        path = tmp_path / name
        assert (
            next(r for r in e.gates(tmp_path) if r["field"] == field)["observed_state"]
            == "missing_file"
        )
        for value, state in [
            ({}, "missing_field"),
            ({field: 0}, "observed_zero"),
            ({field: 1}, "changed_bytes"),
        ]:
            atomic_json(path, value)
            assert (
                next(r for r in e.gates(tmp_path) if r["field"] == field)["observed_state"] == state
            )
    assert all(r["passed"] for r in e.gates(e.ROOT))
    assert e.authenticate()["games"] == e.GAMES
    original = e.sha256_file
    monkeypatch.setattr(
        e,
        "sha256_file",
        lambda p: "changed" if p.name == "arc_competition_agent.py" else original(p),
    )
    with pytest.raises(ValueError, match="qualified_source"):
        e.authenticate()


def test_censorship_parity_and_cross_game_support():
    """SCENARIO-REPORT-8678-TRANSFER: a repaired boundary can still censor all windows."""
    episodes = panel()
    result = e.reduce(episodes)
    assert len(result["action_parity_rows"]) == 8
    assert all(r["passed"] for r in result["action_parity_rows"])
    assert result["complete_horizon_count"] == 0
    assert len(result["horizon_rows"]) == 16
    assert len(result["censored_reason_rows"]) == 16
    assert all(r["independent_count"] == 0 for r in result["cross_game_support_rows"])
    episodes[1]["actions"] = []
    assert not e.reduce(episodes)["action_parity_rows"][0]["passed"]
    episodes = panel()
    episodes[1]["observer_issues"] = [{"reason": "conflicting_duplicate"}]
    assert not e.reduce(episodes)["action_parity_rows"][0]["passed"]
    assert not e.reduce([])["action_parity_rows"][0]["passed"]
    episodes = panel()
    episodes[1]["observer_rows"][0].update(
        horizon_status="no_progress_within_horizon",
        end_reason="horizon_completed",
        observed_horizon=120,
    )
    assert e.reduce(episodes)["complete_horizon_count"] == 1


def sealed(tmp_path, episodes, failures):
    """REQ-VERIFY-8678: retain a seal outside the candidate envelope."""
    work = e.empty_work("20261101", failures)
    work["diagnosis_rows"] = [dict(passed=True)] * 16
    work["controls"] = [dict(case="conflict", passed=True)]
    for i, episode in enumerate(episodes):
        target = tmp_path / f"episode-{i}.json"
        atomic_json(target, episode)
        work["references"].append(e.reference(target))
    path = tmp_path / "work.json"
    atomic_json(path, work)
    atomic_json(tmp_path / "measurement_seal.json", dict(work_sha256=sha256_file(path)))
    checks = [dict(name=n, passed=True, duration_s=0) for n in e.REQUIRED_CHECKS]
    atomic_json(tmp_path / "validation_seal.json", dict(receipts_sha256=canonical_hash(checks)))
    return path, checks


def test_sealed_build_and_rehashed_tampering(tmp_path):
    """SCENARIO-VERIFY-8678-REPLAY: rehashing claims cannot replace sealed primitives."""
    path, checks = sealed(tmp_path, panel(), [])
    output = tmp_path / (e.NAME + ".json")
    value = e.build(path, checks, output)
    assert value["arc_rows_ready_score"] == 1 and value["verdict_class"] == "null"
    assert value["diagnostic_closed"] and value["promotion_ready_score"] == 0
    assert value["preserved_history"]["exp8664_flagged"] is True
    assert e.replay(value)
    changed = deepcopy(value)
    changed["complete_horizon_count"] = 100
    changed["reproducibility_checksum"] = canonical_hash(changed)
    assert not e.replay(changed)
    work = json.loads(path.read_bytes())
    work["run_date"] = "20261102"
    atomic_json(path, work)
    changed["work_reference"] = e.reference(path)
    with pytest.raises(ValueError, match="seal"):
        e.replay(changed)
    assert e.build(path, [], output)["verdict_class"] == "disqualified"
