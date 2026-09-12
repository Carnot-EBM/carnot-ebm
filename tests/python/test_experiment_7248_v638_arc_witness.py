"""CPU conformance for the optional transition witness.

Spec: REQ-ARC-WMTE-7248 and SCENARIO-ARC-WMTE-7248-*.
"""

from __future__ import annotations

from copy import deepcopy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.request

import numpy as np
import pytest

from carnot.agentic import arc_competition_agent as competition
from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic import arc_llm_reinduction as reinduce
from carnot.agentic.arc_transition_witness_exp7248 import (
    WITNESS_MARKER,
    build_transition_witness,
    canonical_witness_bytes,
    render_transition_witness,
    witness_feedback_enabled,
)
from carnot.experiment_7248_v638_arc_witness import (
    independently_reduce_conformance_rows,
    reduce_conformance_rows,
)

pytestmark = pytest.mark.memory_watchdog_skip

BAD_CODE = """import numpy as np
def engine(grid, action, data):
    return np.asarray(grid) - 1
def is_level_complete(grid):
    return bool(np.all(np.asarray(grid) >= 1))
"""

GOOD_CODE = """import numpy as np
def engine(grid, action, data):
    return np.add(np.asarray(grid), 1)
def is_level_complete(grid):
    return bool(np.all(np.asarray(grid) >= 1))
"""


def _transition(before, action, after, data=None):
    return e3.Transition(
        np.asarray(before, dtype=np.int16),
        action,
        data,
        np.asarray(after, dtype=np.int16),
        0,
        0,
    )


def _changed_rows(n=9):
    return [_transition(np.full((2, 2), i), 1, np.full((2, 2), i + 1)) for i in range(n)]


def _exec_engine(source):
    namespace = {}
    exec(compile(source, "<exp7248-test-engine>", "exec"), namespace)  # noqa: S102
    return namespace["engine"]


def _reply(source):
    content = (
        "<tool_call>\n<function=run_engine_on_transitions>\n<parameter=code>\n"
        + source
        + "</parameter>\n</function>\n</tool_call>"
    )
    return {
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": {"completion_tokens": 20, "prompt_tokens": 100},
    }


def test_flag_is_default_off(monkeypatch):
    """SCENARIO-7248-DEFAULT-PARITY: no flag cannot activate feedback."""
    monkeypatch.delenv("CARNOT_ARC_TRANSITION_WITNESS", raising=False)
    assert witness_feedback_enabled() is False
    monkeypatch.setenv("CARNOT_ARC_TRANSITION_WITNESS", "0")
    assert witness_feedback_enabled() is False
    monkeypatch.setenv("CARNOT_ARC_TRANSITION_WITNESS", "1")
    assert witness_feedback_enabled() is True


def test_cpu_panel_types_changed_and_unchanged_observations():
    """SCENARIO-7248-TYPED-WITNESSES: failures get executable transition anchors."""
    moved_right = _transition(
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        1,
        [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
    )
    unchanged = _transition([[0, 0], [0, 1]], 2, [[0, 0], [0, 1]])

    identity = build_transition_witness([moved_right, unchanged], lambda g, _a, _d: g)
    assert identity["available"] is True
    assert identity["observation_counts"] == {
        "total": 2,
        "observed_changed": 1,
        "observed_unchanged": 1,
        "correct_unchanged_identity": 1,
        "mismatched": 1,
    }
    row = identity["mismatches"][0]
    assert row["action"] == {"action": 1, "data": None}
    assert row["pre_frame_hash"].startswith("sha256:")
    assert row["observed_changed_coordinates"] == [[1, 1], [1, 2]]
    assert row["predicted_changed_coordinates"] == []
    assert row["typed_mismatch"] == "identity_on_changed_transition"

    def wrong_direction(grid, _action, _data):
        predicted = np.asarray(grid).copy()
        predicted[1, 1] = 0
        predicted[1, 0] = 1
        return predicted

    wrong = build_transition_witness([moved_right], wrong_direction)
    assert wrong["mismatches"][0]["typed_mismatch"] == "changed_coordinate_set_mismatch"
    assert wrong["mismatches"][0]["predicted_changed_coordinates"] == [[1, 0], [1, 1]]

    two_changes = _transition([[0, 0], [0, 0]], 3, [[1, 1], [0, 0]])

    def misses_one(grid, _action, _data):
        predicted = np.asarray(grid).copy()
        predicted[0, 0] = 1
        return predicted

    missing = build_transition_witness([two_changes], misses_one)
    assert missing["mismatches"][0]["typed_mismatch"] == ("missing_observed_change_coordinates")

    malformed = build_transition_witness([moved_right], lambda _g, _a, _d: [1, 2, 3])
    assert malformed["mismatches"][0]["typed_mismatch"] == "malformed_output"

    exact = build_transition_witness([moved_right], lambda _g, _a, _d: moved_right.next_grid)
    assert exact["available"] is True
    assert exact["readiness_score"] == 0
    assert exact["mismatches"] == []

    no_change = build_transition_witness([unchanged], lambda g, _a, _d: g.copy())
    assert no_change["available"] is False
    assert no_change["reason"] == "no_observed_changed_transitions"
    assert no_change["readiness_score"] == 0
    assert no_change["mismatches"] == []
    assert no_change["observation_counts"]["correct_unchanged_identity"] == 1
    assert no_change["valid_no_change_observations"] == [
        {
            "action": {"action": 2, "data": None},
            "pre_frame_hash": no_change["valid_no_change_observations"][0]["pre_frame_hash"],
            "observed_changed_coordinates": [],
            "predicted_changed_coordinates": [],
            "observation_type": "valid_no_change_identity",
        }
    ]


def test_witness_cap_is_diverse_and_order_independent():
    """REQ-ARC-WMTE-7248: eight rows use stable mismatch/action strata."""
    rows = []
    for action in range(1, 13):
        before = np.zeros((3, 3), dtype=np.int16)
        after = before.copy()
        after[action % 3, (action // 3) % 3] = 1
        rows.append(_transition(before, action, after))
    forward = build_transition_witness(rows, lambda g, _a, _d: g)
    reverse = build_transition_witness(list(reversed(rows)), lambda g, _a, _d: g)
    assert len(forward["mismatches"]) == 8
    assert forward == reverse
    assert len({r["action"]["action"] for r in forward["mismatches"]}) == 8
    assert canonical_witness_bytes(forward) == canonical_witness_bytes(reverse)


def test_witness_error_channels_and_action_data_stay_typed():
    """REQ-ARC-WMTE-7248: runtime failures remain evidence, not successful predictions."""
    changed = _transition([[0, 0], [0, 0]], 6, [[1, 0], [0, 0]], {"y": [2], "x": 1})
    unchanged = _transition([[0, 0], [0, 0]], 7, [[0, 0], [0, 0]])

    def wrong_value(grid, action, data):
        del action, data
        predicted = np.asarray(grid).copy()
        predicted[0, 0] = 2
        return predicted

    value_payload = build_transition_witness([changed], wrong_value)
    assert value_payload["mismatches"][0]["typed_mismatch"] == (
        "wrong_values_at_observed_coordinates"
    )
    assert value_payload["mismatches"][0]["action"]["data"] == {"x": 1, "y": [2]}

    def hallucination(grid, action, data):
        del action, data
        predicted = np.asarray(grid).copy()
        predicted[1, 1] = 1
        return predicted

    mixed = build_transition_witness([changed, unchanged], hallucination)
    assert "change_predicted_for_unchanged_transition" in {
        row["typed_mismatch"] for row in mixed["mismatches"]
    }

    def raises(_grid, _action, _data):
        raise RuntimeError("fixture failure")

    errored = build_transition_witness([changed], raises)
    assert errored["mismatches"][0]["typed_mismatch"] == "engine_error"
    assert errored["mismatches"][0]["error_kind"] == "RuntimeError"

    shape_change = _transition([[0, 0]], 8, [[0], [0]], {"value": object()})
    shaped = build_transition_witness([shape_change], lambda g, _a, _d: g)
    assert shaped["mismatches"][0]["typed_mismatch"] == "observed_shape_change"
    assert isinstance(shaped["mismatches"][0]["action"]["data"]["value"], str)
    assert build_transition_witness([changed], wrong_value, max_mismatches=0)["mismatches"] == []


def test_selfparse_refinement_delivers_exact_witness_and_passes_existing_gate(
    monkeypatch, tmp_path
):
    """SCENARIO-7248-RUNTIME-DELIVERY: the scored CEGIS request carries exact bytes."""
    monkeypatch.setattr(e3, "E3_DIR", tmp_path / "engines")
    monkeypatch.setattr(reinduce, "MAX_REFINEMENT_ROUNDS", 2)
    monkeypatch.setenv("CARNOT_ARC_INDUCE_THINK", "0")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "selfparse")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_TURNS", "1")
    monkeypatch.setenv("CARNOT_ARC_CEGIS_TOOL_LOOP", "1")
    monkeypatch.setenv("CARNOT_ARC_TRANSITION_WITNESS", "1")
    monkeypatch.delenv("CARNOT_ARC_INDUCE_TOOL_GRAMMAR", raising=False)
    proposer = e3.LocalGGUFProposer(ffn_cpu_layers=0, mtp=False, max_tokens=1024, tries=1)
    monkeypatch.setattr(proposer, "_ensure_server", lambda: True)
    sent = []
    replies = [_reply(BAD_CODE), _reply(GOOD_CODE)]

    def request(req, timeout=None):
        del timeout
        sent.append(bytes(req.data))
        return io.BytesIO(json.dumps(replies.pop(0)).encode())

    monkeypatch.setattr(urllib.request, "urlopen", request)
    transitions = _changed_rows()
    result = reinduce.execute_bounded_llm_reinduction(
        game="exp7248",
        transitions=transitions,
        cell=1,
        root_grid=np.zeros((2, 2), dtype=np.int16),
        proposer=proposer,
        candidate_provider=lambda engine, goal: [("generated", engine, goal)],
        load_engine=e3.load_engine,
        plan_in_model=lambda _engine, _goal, _root: [{"action": 1, "data": None}],
        max_rounds=2,
        min_heldout_accuracy=1.0,
        transition_witness_enabled=True,
    )

    expected = render_transition_witness(
        build_transition_witness(transitions, _exec_engine(BAD_CODE))
    )
    payloads = [json.loads(item) for item in sent]
    assert len(payloads) == 2
    assert WITNESS_MARKER not in payloads[0]["messages"][0]["content"]
    assert expected in payloads[1]["messages"][0]["content"]
    assert result.planned is True, {
        "rounds": [
            {
                key: row.get(key)
                for key in (
                    "action",
                    "accepted_by_heldout_verifier",
                    "heldout_accuracy",
                    "skipped",
                    "retention_signal_heldout_change_consistency",
                )
            }
            for row in result.rounds
        ],
        "loop_stats": [
            {
                "terminated_by": row.get("tool_loop", {}).get("terminated_by"),
                "mismatch_trajectory": row.get("tool_loop", {}).get("mismatch_trajectory"),
                "best_visible_mismatches": row.get("tool_loop", {}).get("best_visible_mismatches"),
            }
            for row in result.rounds
        ],
    }
    assert result.accepted_by_heldout_verifier is True
    assert result.heldout_accuracy == 1.0
    assert result.rounds[-1]["action"] == "refactor_tool_loop"


def test_disabled_refinement_request_bytes_are_exactly_equal(monkeypatch, tmp_path):
    """SCENARIO-7248-DEFAULT-PARITY: absent and explicit false use identical bytes."""
    monkeypatch.setattr(e3, "E3_DIR", tmp_path / "engines")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_THINK", "0")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_LOOP", "selfparse")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TOOL_TURNS", "1")
    monkeypatch.delenv("CARNOT_ARC_INDUCE_TOOL_GRAMMAR", raising=False)
    proposer = e3.LocalGGUFProposer(ffn_cpu_layers=0, mtp=False, max_tokens=1024, tries=1)
    monkeypatch.setattr(proposer, "_ensure_server", lambda: True)
    path = Path(e3.E3_DIR) / "parity" / "world_model.py"
    path.parent.mkdir(parents=True)
    sent = []

    def request(req, timeout=None):
        del timeout
        sent.append(bytes(req.data))
        return io.BytesIO(json.dumps(_reply(GOOD_CODE)).encode())

    monkeypatch.setattr(urllib.request, "urlopen", request)
    path.write_text(BAD_CODE)
    assert reinduce._tool_loop_refactor(proposer, "parity", _changed_rows(), 1)[0]
    absent = sent.pop()
    path.write_text(BAD_CODE)
    assert reinduce._tool_loop_refactor(
        proposer,
        "parity",
        _changed_rows(),
        1,
        transition_witness_enabled=False,
    )[0]
    disabled = sent.pop()
    assert absent == disabled


def test_e3_policy_is_the_only_runtime_switch_owner(monkeypatch):
    """REQ-ARC-WMTE-7248: the canonical scored wrapper opts the generic loop in."""
    policy = competition.E3AgentPolicy("exp7248", proposer=object(), value_head=None)
    policy.think_arm_fallback_enabled = False
    monkeypatch.setattr(policy, "_configure_producer_evidence", lambda *_args: None)
    captured = []

    def fake_execute(**kwargs):
        captured.append(dict(kwargs))
        return SimpleNamespace(heldout_accuracy=1.0)

    monkeypatch.setattr(competition, "execute_bounded_llm_reinduction", fake_execute)
    monkeypatch.delenv("CARNOT_ARC_TRANSITION_WITNESS", raising=False)
    policy._execute_bounded_llm_reinduction_with_arm_fallback({}, proposer=object(), transitions=[])
    assert "transition_witness_enabled" not in captured[-1]
    monkeypatch.setenv("CARNOT_ARC_TRANSITION_WITNESS", "1")
    policy._execute_bounded_llm_reinduction_with_arm_fallback({}, proposer=object(), transitions=[])
    assert captured[-1]["transition_witness_enabled"] is True


def test_independent_reducer_catches_a_disconnected_adapter_mutation():
    """SCENARIO-7248-NEGATIVE-CONTROLS: a dropped witness cannot earn readiness."""
    rows = [
        {"unit": "identity_changed", "passed": True},
        {"unit": "identity_unchanged", "passed": True},
        {"unit": "wrong_direction", "passed": True},
        {"unit": "missing_effect", "passed": True},
        {"unit": "malformed_output", "passed": True},
        {
            "unit": "runtime_delivery",
            "generated_witness_sha256": "sha256:abc",
            "delivered_witness_sha256": "sha256:abc",
            "accepted_by_existing_policy_gate": True,
        },
        {"unit": "default_parity", "request_bytes_equal": True, "action_equal": True},
        {"unit": "leakage", "agent_observation_history_only": True},
        {"unit": "disconnect_mutation", "caught": True},
    ]
    clean = reduce_conformance_rows(rows)
    assert clean["arc_witness_ready_score"] == 1
    assert independently_reduce_conformance_rows(rows) == clean
    mutant = deepcopy(rows)
    mutant[5]["delivered_witness_sha256"] = None
    caught = reduce_conformance_rows(mutant)
    assert caught["arc_witness_ready_score"] == 0
    assert caught["gates"]["runtime_delivery"] is False
    assert independently_reduce_conformance_rows(mutant) == caught
