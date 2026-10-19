"""REQ-REPORT-8664, REQ-VERIFY-8664: qualify current joins without policy credit."""

from copy import deepcopy
import json

import pytest
from test_arc_reset_transfer_8650 import panel as prior_panel

from carnot.reporting import arc_reset_transfer_8664 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


def panel():
    """Keep prior action fixtures and add current callback/public-frame operands."""
    episodes = prior_panel()
    for ep in episodes:
        ep["sdk_identity"] = dict(official=True)
        ep["recorder_error_metadata"] = []
        ep["reset_provenance"]["callback_sequence_rows"] = [
            dict(action_index=1, method="returned", owner="sdk_return", invocation=1)
        ]
        ep["reset_observations"] = [
            dict(action_index=1, observation_sha256=canonical_hash({}), public_observation={})
        ]
        for request in ep["reset_provenance"]["reset_request_rows"]:
            request["post_observation_sha256"] = canonical_hash({})
    return episodes


def test_current_panel_and_zero_support():
    """SCENARIO-REPORT-8664-TRANSFER: support counts eight dependent cells once."""
    rows = panel()
    result = e.reduce(rows)
    assert result["diagnostic_transfer_score"] == 1
    assert result["reset_transfer_rows_ready_score"] == 1
    assert sum(r["support_count"] for r in result["cause_support_rows"]) == 8
    assert len(result["callback_sequence_rows"]) == 8
    assert len(e.units()) == 16
    assert all(u["observer"] == "on" and u["max_seconds"] == 120 for u in e.units())
    for ep in rows:
        ep["reset_observations"] = []
        ep["reset_provenance"]["reset_request_rows"] = []
    result = e.reduce(rows)
    assert result["reset_transfer_rows_ready_score"] == 1
    assert result["diagnostic_transfer_score"] == 0
    assert all(r["reset_support_count"] == 0 for r in result["per_game_results"])


def test_censored_redirect_callback_join():
    """SCENARIO-REPORT-8664-TRANSFER: censoring joins request and callback identities."""
    episodes = panel()
    row = dict(
        redirect_id="redirect",
        application_index=0,
        observed_horizon=1,
        horizon_status="censored",
        end_reason="reset",
        overlapping_redirect_ids=[],
    )
    for ep in episodes[:2]:
        ep["observer_rows"] = [deepcopy(row)]
    result = e.reduce(episodes)
    assert result["complete_horizon_count"] == 0
    on = next(r for r in result["redirect_reset_joins"] if r["reset_provenance"] == "on")
    assert on["reset_request_id"] == "1:1" and on["callback_ids"]
    assert on["origin"] == "agent_reset"


def test_prior_observer_issue_gate_is_preserved():
    """REQ-REPORT-8664: authentic causes cannot erase a failed original observer gate."""
    episodes = panel()
    episodes[1]["observer_issues"] = [dict(action_index=1, reason="duplicate_observation")]
    result = e.reduce(episodes)
    assert all(r["passed"] for r in result["action_parity_rows"])
    assert result["missing_cause_rows"] == result["recorder_error_rows"] == []
    assert result["reset_transfer_rows_ready_score"] == result["diagnostic_transfer_score"] == 0


@pytest.mark.parametrize("mutation", ["id", "callback", "duplicate", "error", "sdk", "unknown"])
def test_current_join_rejections(mutation):
    """SCENARIO-REPORT-8664-TRANSFER: stale context cannot authenticate a reset."""
    rows = panel()
    ep = rows[1]
    request = ep["reset_provenance"]["reset_request_rows"][0]
    if mutation == "id":
        request["reset_request_id"] = "99:1"
    elif mutation == "callback":
        ep["reset_provenance"]["callback_sequence_rows"] = []
    elif mutation == "duplicate":
        ep["reset_provenance"]["reset_request_rows"].append(deepcopy(request))
    elif mutation == "error":
        ep["recorder_error_metadata"] = [dict(error="TypeError", message="bounded")]
    elif mutation == "sdk":
        ep["sdk_identity"]["official"] = False
    else:
        request["origin"] = "unknown"
    result = e.reduce(rows)
    assert result["diagnostic_transfer_score"] == 0
    assert result["recorder_error_rows"] or result["missing_cause_rows"] or mutation == "sdk"


def test_exact_upstream_gates(tmp_path, monkeypatch):
    """REQ-VERIFY-8664: missing files, missing fields and zero remain distinct."""
    source = e.authenticate(e.ROOT)
    assert source["reset_provenance_ready_score"] == source["arc_inputs_ready_score"] == 1
    assert source["original_observer_reference"]["sha256"] == e.OBSERVER_PIN
    missing = e.gate_summary(tmp_path)
    assert all(r["observed_state"] == "missing_file" for r in missing)
    for name in e.PINS:
        atomic_json(tmp_path / name, {})
    assert all(r["observed_state"] == "missing_field" for r in e.gate_summary(tmp_path))
    for name, field in zip(e.PINS, e.GATE_FIELDS, strict=True):
        atomic_json(tmp_path / name, {field: 0})
    assert all(r["observed_state"] == "observed_zero" for r in e.gate_summary(tmp_path))
    original = e.checked

    def changed(path, digest):
        value = original(path, digest)
        if "8663" in path.name:
            value["reset_provenance_ready_score"] = 0
        return value

    monkeypatch.setattr(e, "checked", changed)
    with pytest.raises(ValueError, match="readiness"):
        e.authenticate(e.ROOT)
    monkeypatch.undo()
    monkeypatch.setattr(e, "read_bound_sidecar", lambda *a: dict(report=dict(passed=False)))
    with pytest.raises(ValueError, match="publication"):
        e.authenticate(e.ROOT)


def work_file(tmp_path, episodes):
    """Keep fixtures private and retain a measurement seal outside the envelope."""
    references = []
    for index, ep in enumerate(episodes):
        p = tmp_path / f"episode-{index}.json"
        atomic_json(p, ep)
        references.append(dict(path=str(p), sha256=sha256_file(p)))
    path = tmp_path / "work.json"
    atomic_json(
        path,
        dict(
            run_date="20261019",
            failures=[] if episodes else e.gate_summary(tmp_path),
            references=references,
            preconditions_checked={},
            source_artifact_hashes={},
            code_config_hashes={},
            raw_file_hashes={},
            historical_model_provenance={},
            registry_precheck={},
            duration_s=0.01,
            phase_spans=[],
            measurement_receipts=[],
            upstream_gates={},
        ),
    )
    atomic_json(path.with_name("measurement_seal.json"), dict(work_sha256=sha256_file(path)))
    return path


def test_build_replay_and_failure_classes(tmp_path):
    """SCENARIO-VERIFY-8664-REPLAY: no rehashed envelope replaces the independent seal."""
    path = work_file(tmp_path, panel())
    receipts = [dict(name=n, passed=True, duration_s=0.01) for n in e.REQUIRED_CHECKS]
    output = tmp_path / (e.NAME + ".json")
    value = e.build(path, receipts, output)
    assert value["experiment_id"] == 8664 and value["run_date"] == "20261019"
    assert value["verdict_class"] == "null" and value["diagnostic_transfer_score"] == 1
    assert value["policy_refinement_score"] == value["promotion_ready_score"] == 0
    assert value["independent_count"] == 4
    assert e.replay(value)
    value["diagnostic_transfer_score"] = 0
    assert not e.replay(value)
    value = e.build(path, [dict(receipts[0], passed=False)], output)
    assert value["verdict_class"] == "disqualified"
    path = work_file(tmp_path, [])
    value = e.build(path, receipts, output)
    assert value["verdict_class"] == "blocked" and e.replay(value)
    work = json.loads(path.read_bytes())
    work["failures"] = []
    atomic_json(path, work)
    assert e.build(path, receipts, output)["verdict_class"] == "disqualified"
    value["work_reference"]["sha256"] = sha256_file(path)
    with pytest.raises(ValueError, match="measurement_seal"):
        e.replay(value)
    path.unlink()
    value["work_reference"]["sha256"] = json.loads(
        path.with_name("measurement_seal.json").read_bytes()
    )["work_sha256"]
    with pytest.raises(OSError):
        e.replay(value)
