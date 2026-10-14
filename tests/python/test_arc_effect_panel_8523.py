"""REQ-REPORT-8523: direct state effects and censored progress stay separate."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.reporting import arc_effect_panel_8523 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def test_bound_upstream_and_frozen_pairs(tmp_path):
    """SCENARIO-REPORT-8523-EFFECTS preserves the qualified panel and authority."""
    source = e.authenticate(e.ROOT, tmp_path)
    units = e.units(source["protocol"])
    assert len(units) == 16
    assert len({u["game"] for u in units}) == 4
    assert [u["mode"] for u in units[:4]] == ["applied", "shadow", "shadow", "applied"]
    assert all(u["max_actions"] == 512 and u["max_seconds"] == 120 for u in units)
    assert source["primary"]["arc_effect_receipts_ready_score"] == 1


def test_missing_upstream_and_wrong_hash(tmp_path):
    """REQ-VERIFY-8523: absence differs from observed zero and refuses execution."""
    with pytest.raises(OSError):
        e.authenticate(tmp_path, tmp_path / "raw")
    path = tmp_path / "wrong.json"
    atomic_json(path, {})
    with pytest.raises(ValueError, match="input_hash"):
        e.checked(path, "sha256:invalid")


def primitive(tmp_path, unit, *, mode="applied", changed=True, resolved=False):
    """Small visible observations qualify reduction; they are never live evidence."""
    before = dict(
        goal_bias_present=False,
        induced=True,
        hybrid_diversity=False,
        steps_since_progress=120,
        stall_threshold=150,
    )
    if not changed and mode == "applied":
        before.update(hybrid_diversity=True, steps_since_progress=151)
    after = dict(before, hybrid_diversity=True, steps_since_progress=151) if changed else before
    observation = dict(pixels=[[[0]]], level=0, available_actions=[1])
    receipt = dict(
        schema=e.SCHEMA,
        action_index=1,
        wrapper_action_index=2,
        observation=observation,
        observation_sha256=canonical_hash(observation),
        snapshot=dict(level=0),
        eligibility_predicates=[dict(arm="force_exploration_diversity", eligible=True)],
        mode=mode,
        arm="force_exploration_diversity",
        selected=True,
        applied=changed and mode == "applied",
        application_disposition="selected_shadow"
        if mode == "shadow"
        else "state_changed"
        if changed
        else "already_active_no_op",
        before_state=before,
        after_state=after,
        before_state_sha256=canonical_hash(before),
        after_state_sha256=canonical_hash(after),
        effective_action_policy_change=changed,
        induction_attempt=None,
        induction_outcome="not_observed",
        model_action_executed=False,
        observation_error=None,
        action=dict(kind=1, data=None),
    )
    directory = tmp_path / unit["episode_id"].replace(":", "_")
    directory.mkdir(parents=True, exist_ok=True)
    terminal = dict(
        unit=unit,
        solve_provenance="development_proxy",
        error=None,
        censoring="action_limit",
        duration_s=0.1,
        model_tripwire_passed=True,
        model_tripwire_attempts=[],
        llm_invocation_count=0,
        actual_wrapper_path=dict(
            factory="make_carnot_agent", policy="E3AgentPolicy", choose_action=3, is_done=3
        ),
        supervisor_receipt=dict(
            direct_effect_receipts=[receipt],
            redirects=[dict(arm=receipt["arm"], action_index=1, resolved_by_levelup=resolved)],
        ),
    )
    steps = []
    for index in range(3):
        action = dict(name="RESET" if index == 0 else "ACTION1", data={})
        visible = dict(observation, level=int(resolved and index == 2))
        steps.append(
            dict(
                index=index + 1,
                action=action,
                action_sha256=canonical_hash(action),
                observation=visible,
                observation_sha256=canonical_hash(visible),
            )
        )
    atomic_json(directory / "episode.json", terminal)
    (directory / "steps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps))
    return dict(
        unit=unit,
        directory=str(directory),
        terminal_sha256=e.sha256_file(directory / "episode.json"),
        steps_sha256=e.sha256_file(directory / "steps.jsonl"),
    )


def test_direct_mutation_shadow_noop_and_rehashed_tamper(tmp_path):
    """REQ-REPORT-8523: mode alone cannot establish application or resolved support."""
    panel = e.units(e.qualification.protocol())
    applied = primitive(tmp_path, panel[0])
    shadow = primitive(tmp_path, panel[1], mode="shadow", changed=False)
    row, effects = e.reduce_episode(applied)
    assert row["mutation_count"] == 1 and effects[0]["applied"]
    assert effects[0]["right_censored"]
    assert not e.reduce_episode(shadow)[1][0]["applied"]
    noop = primitive(tmp_path / "noop", panel[0], changed=False)
    assert e.reduce_episode(noop)[0]["no_op_count"] == 1
    path = Path(applied["directory"]) / "episode.json"
    terminal = json.loads(path.read_bytes())
    terminal["supervisor_receipt"]["direct_effect_receipts"][0]["applied"] = False
    atomic_json(path, terminal)
    applied["terminal_sha256"] = e.sha256_file(path)
    with pytest.raises(ValueError, match="receipt_application"):
        e.reduce_episode(applied)


def test_all_slots_pairs_and_independence(tmp_path):
    """REQ-REPORT-8523: right censoring never becomes failed-arm evidence."""
    panel = e.units(e.qualification.protocol())
    references = [
        primitive(tmp_path, u, mode=u["mode"], changed=u["mode"] == "applied") for u in panel
    ]
    summary = e.reduce(references, panel)
    assert len(summary["rows"]) == 16 and len(summary["action_divergence_rows"]) == 8
    assert summary["independent_count"] == 4
    assert not summary["arm_refinement_authorized"]
    assert all(r["matched_initial_state"] for r in summary["action_divergence_rows"])
    missing = e.reduce([], panel)
    assert len(missing["rows"]) == 16
    assert all(r["missing_reason"] for r in missing["rows"])
    bad = deepcopy(references)
    bad.append(bad[0])
    with pytest.raises(ValueError, match="duplicate_episode"):
        e.reduce(bad, panel)


@pytest.mark.parametrize(
    "case",
    [
        "identity",
        "steps_hash",
        "sequence",
        "boundary",
        "observation",
        "state",
        "arm",
        "substrate",
        "selection",
    ],
)
def test_independent_primitive_refusals(tmp_path, case):
    """REQ-VERIFY-8523: even self-rehashed malformed operands cannot supply effects."""
    reference = primitive(tmp_path, e.units(e.qualification.protocol())[0])
    directory = Path(reference["directory"])
    terminal = json.loads((directory / "episode.json").read_bytes())
    receipt = terminal["supervisor_receipt"]["direct_effect_receipts"][0]
    if case == "identity":
        terminal["unit"] = {}
    elif case == "steps_hash":
        reference["steps_sha256"] = "bad"
    elif case == "sequence":
        steps = [json.loads(s) for s in (directory / "steps.jsonl").read_text().splitlines()]
        steps[0]["index"] = 0
        (directory / "steps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps))
        reference["steps_sha256"] = e.sha256_file(directory / "steps.jsonl")
    elif case == "boundary":
        receipt["mode"] = "shadow"
    elif case == "observation":
        receipt["observation_sha256"] = "bad"
    elif case == "state":
        receipt["before_state_sha256"] = "bad"
    elif case == "arm":
        receipt["arm"] = "invented"
    elif case == "substrate":
        receipt["model_action_executed"] = True
    elif case == "selection":
        receipt["eligibility_predicates"] = []
    atomic_json(directory / "episode.json", terminal)
    reference["terminal_sha256"] = e.sha256_file(directory / "episode.json")
    with pytest.raises(ValueError):
        e.reduce_episode(reference)


@pytest.mark.parametrize("arm", ["drop_goal_bias", "allow_reinduction"])
def test_each_mutation_and_disabled_induction(tmp_path, arm):
    """REQ-REPORT-8523: resetting a latch never establishes completed induction."""
    ref = primitive(tmp_path, e.units(e.qualification.protocol())[0])
    path = Path(ref["directory"]) / "episode.json"
    terminal = json.loads(path.read_bytes())
    receipt = terminal["supervisor_receipt"]["direct_effect_receipts"][0]
    before = dict(receipt["before_state"], goal_bias_present=True)
    after = dict(before)
    after["goal_bias_present" if arm == "drop_goal_bias" else "induced"] = False
    receipt.update(
        arm=arm,
        before_state=before,
        after_state=after,
        before_state_sha256=canonical_hash(before),
        after_state_sha256=canonical_hash(after),
        eligibility_predicates=[dict(arm=arm, eligible=True)],
        effective_action_policy_change=arm == "drop_goal_bias",
    )
    terminal["supervisor_receipt"]["redirects"][0]["arm"] = arm
    if arm == "allow_reinduction":
        receipt.update(
            induction_attempt=dict(skipped="disabled_by_env"), induction_outcome="model_disabled"
        )
    atomic_json(path, terminal)
    ref["terminal_sha256"] = e.sha256_file(path)
    assert e.reduce_episode(ref)[1][0]["applied"]


def test_nonselected_and_real_wrapper_replay_branch(tmp_path, monkeypatch):
    """REQ-VERIFY-8523: nonselections stay observations; live rows cross shipped replay."""
    ref = primitive(tmp_path, e.units(e.qualification.protocol())[0], changed=False)
    path = Path(ref["directory"]) / "episode.json"
    terminal = json.loads(path.read_bytes())
    receipt = terminal["supervisor_receipt"]["direct_effect_receipts"][0]
    receipt.update(selected=False, arm=None, application_disposition="not_selected")
    terminal["solve_provenance"] = "live_agent_self_discovery"
    monkeypatch.setattr(e.panel_runtime, "reduce_episode", lambda *_: {})
    atomic_json(path, terminal)
    ref["terminal_sha256"] = e.sha256_file(path)
    row, events = e.reduce_episode(ref)
    assert events == [] and row["solve_provenance"] == "live_agent_self_discovery"


def test_action_divergence_resolved_and_errors(tmp_path):
    """REQ-REPORT-8523: descriptive action effects remain independent of level progress."""
    panel = e.units(e.qualification.protocol())
    refs = [
        primitive(tmp_path, u, mode=u["mode"], changed=u["mode"] == "applied", resolved=True)
        for u in panel
    ]
    ref = refs[0]
    directory = Path(ref["directory"])
    steps = [json.loads(s) for s in (directory / "steps.jsonl").read_text().splitlines()]
    steps[-1]["action"] = dict(name="ACTION2", data={})
    steps[-1]["action_sha256"] = canonical_hash(steps[-1]["action"])
    (directory / "steps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps))
    ref["steps_sha256"] = e.sha256_file(directory / "steps.jsonl")
    summary = e.reduce(refs, panel)
    assert summary["action_divergence_rows"][0]["divergent_actions"] == 1
    assert summary["censoring_counts"]["resolved_redirects"] == 16
    terminal = json.loads((directory / "episode.json").read_bytes())
    terminal["error"] = "policy:error"
    atomic_json(directory / "episode.json", terminal)
    ref["terminal_sha256"] = e.sha256_file(directory / "episode.json")
    assert e.reduce_episode(ref)[0]["status"] == "failed"


def test_shipped_wrapper_receipts_and_primitive_replay(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8523-REPLAY crosses real wrapper and reducer on private observations."""
    import os
    from types import SimpleNamespace
    from unittest.mock import patch

    frame = SimpleNamespace(
        frame=[[[0] * 8 for _ in range(8)]],
        levels_completed=0,
        available_actions=[1],
        state="NOT_FINISHED",
    )

    class Environment:
        def reset(self):
            return frame

        def step(self, action, data):
            return frame

    arcade = SimpleNamespace(make=lambda *_a, **_k: Environment(), open_scorecard=lambda: "private")
    monkeypatch.setattr(e.runtime.numeric, "load_live_action_effect_scorer", lambda **_: None)
    unit = dict(e.units(e.qualification.protocol())[0], max_actions=4)
    with patch.dict(
        os.environ, {"CARNOT_ARC_EFFECT_RECEIPTS": "1", "CARNOT_ARC_DISABLE_INDUCTION": "1"}
    ):
        terminal = e.runtime.episode(unit, arcade, tmp_path)
    assert terminal["error"] is None
    ref = dict(
        unit=unit,
        directory=str(tmp_path),
        terminal_sha256=e.sha256_file(tmp_path / "episode.json"),
        steps_sha256=e.sha256_file(tmp_path / "steps.jsonl"),
    )
    row, events = e.reduce_episode(ref)
    assert row["actions_used"] == 4 and row["receipt_count"] == 3 and events == []
