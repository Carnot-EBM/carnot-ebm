"""REQ-REPORT-8650, REQ-VERIFY-8650: reset evidence cannot invent causal support."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.reporting import arc_reset_transfer_8650 as e
from carnot.reporting.current_work_receipt import atomic_json, sha256_file


def episode(unit, *, reset=True):
    """Keep synthetic public returns distinct from official SDK measurements."""
    action = dict(kind="RESET" if reset else 1, data=None)
    request = dict(
        reset_request_id="1:1",
        action_index=1,
        origin="agent_reset",
        initiating_policy_branch="StepwiseExplorer.next_move.bootstrap",
        pre_observation_sha256="before",
        post_observation_sha256="after",
        remaining_actions=512,
        remaining_time_s=119,
        sdk_status=dict(state="NOT_FINISHED", error=None),
    )
    return dict(
        unit=unit,
        actions=[action],
        rng=[1],
        torch_rng=[2],
        observations=[{}],
        termination=dict(error=None, censoring="action_limit"),
        legacy="receipt",
        sdk_calls=[dict(method="reset" if reset else "step")],
        observer_rows=[],
        observer_issues=[],
        recorder_errors=[],
        numerical_head_calls=1,
        llm_calls=0,
        model_tripwire_attempts=[],
        steps=[],
        events=[],
        end_reason="action_limit",
        reset_observations=[dict(action_index=1, observation_sha256="after")] if reset else [],
        reset_provenance=dict(
            reset_request_rows=[request] if reset and unit["reset_provenance"] == "on" else [],
            requester_ownership_rows=[],
            terminal_reason_rows=[],
            missing_cause_rows=[],
            issues=[],
        ),
    )


def panel():
    return [episode(u) for u in e.units()]


def test_fixed_protocol_and_support():
    """SCENARIO-REPORT-8650-TRANSFER: twins do not double eight support cells."""
    values = panel()
    result = e.reduce(values)
    assert len(values) == 16
    assert all(u["observer"] == "on" and u["max_actions"] == 512 for u in e.units())
    assert result["reset_transfer_rows_ready_score"] == 1
    assert result["diagnostic_transfer_score"] == 1
    assert sum(r["support_count"] for r in result["cause_support_rows"]) == 8
    assert len(result["action_parity_rows"]) == 8
    empty = [episode(u, reset=False) for u in e.units()]
    result = e.reduce(empty)
    assert result["reset_transfer_rows_ready_score"] == 1
    assert result["diagnostic_transfer_score"] == 0
    assert all(r["reset_support_count"] == 0 for r in result["per_game_results"])


@pytest.mark.parametrize(
    "mutation",
    ["unknown", "duplicate", "fabricated", "hash", "parity", "missing", "error", "head", "branch"],
)
def test_reject_unowned_and_changed_resets(mutation):
    """REQ-REPORT-8650: observations and requester receipts must agree exactly."""
    values = panel()
    value = values[1]
    request = value["reset_provenance"]["reset_request_rows"][0]
    if mutation == "unknown":
        request["origin"] = "unknown"
    elif mutation == "duplicate":
        value["reset_provenance"]["reset_request_rows"].append(deepcopy(request))
    elif mutation == "fabricated":
        request["action_index"] = 99
    elif mutation == "hash":
        request["post_observation_sha256"] = "forged"
    elif mutation == "parity":
        value["actions"] = []
    elif mutation == "missing":
        values.pop()
    elif mutation == "head":
        value["numerical_head_calls"] = 0
    elif mutation == "branch":
        request["initiating_policy_branch"] = "fabricated_supervisor"
    else:
        value["recorder_errors"] = ["failure"]
    result = e.reduce(values)
    assert not result["diagnostic_transfer_score"]
    assert result["missing_cause_rows"] or mutation in {"parity", "missing", "error", "head"}


def test_censored_join_uses_actual_request():
    """REQ-REPORT-8650: no horizon extension and no default supervisor cause."""
    values = panel()
    for value in values:
        value["observer_rows"] = [
            dict(
                redirect_id="r",
                application_index=1,
                observed_horizon=0,
                horizon_status="censored",
                end_reason="reset",
                sequence=[],
                overlapping_redirect_ids=[],
            )
        ]
    result = e.reduce(values)
    assert len(result["redirect_reset_joins"]) == 16
    assert all(
        r["reset_request_id"] == "1:1"
        for r in result["redirect_reset_joins"]
        if r["reset_provenance"] == "on"
    )
    assert all(
        r["missing_reason"] == "reset_provenance_disabled"
        for r in result["redirect_reset_joins"]
        if r["reset_provenance"] == "off"
    )
    assert result["complete_horizon_count"] == 0
    values[1]["reset_provenance"]["reset_request_rows"] = []
    assert e.reduce(values)["redirect_reset_joins"][0]["origin"] == "unknown"


def test_original_authentication_and_missing(tmp_path):
    """REQ-VERIFY-8650: producer pins survive changed live agent source."""
    source = e.authenticate(e.ROOT)
    assert source["reset_provenance_ready_score"] == 1
    assert source["arc_inputs_ready_score"] == 1
    with pytest.raises(OSError):
        e.authenticate(tmp_path)
    path = tmp_path / next(iter(e.PINS))
    atomic_json(path, {})
    with pytest.raises(ValueError):
        e.authenticate(tmp_path)


def test_build_blocked_disqualified_and_rehashed_replay(tmp_path):
    """SCENARIO-VERIFY-8650-REPLAY: an envelope cannot replace measurement authority."""
    work = dict(
        run_date="20261018",
        failures=[dict(check="missing", observed=None)],
        references=[],
        preconditions_checked={},
        source_artifact_hashes={},
        code_config_hashes={},
        historical_model_provenance={},
        registry_precheck={},
        duration_s=0.01,
        phase_spans=[],
        measurement_receipts=[],
        upstream_gates={},
    )
    path = tmp_path / "work.json"
    atomic_json(path, work)
    atomic_json(path.with_name("measurement_seal.json"), dict(work_sha256=sha256_file(path)))
    receipt = dict(name="private_control", passed=True, duration_s=0.01)
    output = tmp_path / (e.NAME + ".json")
    value = e.build(path, [receipt], output)
    assert value["verdict_class"] == "blocked" and e.replay(value)
    assert e.build(path, [dict(receipt, passed=False)], output)["verdict_class"] == "disqualified"
    changed = deepcopy(value)
    changed["diagnostic_transfer_score"] = 1
    assert not e.replay(changed)
    work["failures"] = []
    atomic_json(path, work)
    changed["work_reference"]["sha256"] = sha256_file(path)
    with pytest.raises(ValueError, match="measurement_seal"):
        e.replay(changed)


@pytest.mark.parametrize("mutation", ["readiness", "publication"])
def test_upstream_semantic_rejections(monkeypatch, mutation):
    """REQ-VERIFY-8650: hash qualification cannot replace the producer's semantic gate."""
    original = e.checked

    def checked(path, pin):
        value = original(path, pin)
        if mutation == "readiness" and "8640" in path.name:
            value["arc_inputs_ready_score"] = 0
        return value

    monkeypatch.setattr(e, "checked", checked)
    if mutation == "publication":
        monkeypatch.setattr(e, "read_bound_sidecar", lambda *args: dict(report=dict(passed=False)))
    with pytest.raises(ValueError):
        e.authenticate(e.ROOT)


def test_complete_panel_build_and_receipt_hash_replay(tmp_path):
    """REQ-VERIFY-8650: publication recomputes science from immutable episode primitives."""
    references = []
    for index, ep in enumerate(panel()):
        path = tmp_path / f"episode-{index}.json"
        atomic_json(path, ep)
        references.append(dict(path=str(path), sha256=sha256_file(path)))
    operand = tmp_path / "operand"
    operand.write_text("immutable")
    work = dict(
        run_date="20261018",
        failures=[],
        references=references,
        preconditions_checked={},
        source_artifact_hashes={str(operand): sha256_file(operand)},
        code_config_hashes={},
        raw_file_hashes={str(operand): sha256_file(operand)},
        historical_model_provenance={},
        registry_precheck={},
        duration_s=0.01,
        phase_spans=[],
        measurement_receipts=[],
        upstream_gates={},
    )
    path = tmp_path / "work.json"
    atomic_json(path, work)
    atomic_json(path.with_name("measurement_seal.json"), dict(work_sha256=sha256_file(path)))
    log = tmp_path / "stdout"
    log.write_text("actual child")
    receipts = [
        dict(
            name=n,
            passed=True,
            duration_s=0.01,
            stdout_path=str(log),
            stdout_sha256=sha256_file(log),
        )
        for n in e.REQUIRED_CHECKS
    ]
    value = e.build(path, receipts, tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "null" and value["diagnostic_transfer_score"] == 1
    assert value["policy_refinement_score"] == value["promotion_ready_score"] == 0
    assert e.replay(value)
    operand.unlink()
    with pytest.raises(OSError):
        e.replay(value)
    full = panel()
    full[1]["observer_rows"] = [dict(horizon_status="no_progress_within_horizon")]
    assert e.reduce(full)["complete_horizon_count"] == 1
