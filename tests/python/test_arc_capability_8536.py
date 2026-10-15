"""REQ-VERIFY-8536: unavailable work must not consume a supervisor arm."""

from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from carnot.agentic import arc_supervisor_capability as cap
from carnot.agentic.arc_trajectory_supervisor import TrajectorySnapshot, TrajectorySupervisor


def snapshot(available, **changes):
    """Build visible state independently of the capability predicate under test."""
    fields = dict(
        level=0,
        goal_bias_installed=False,
        induced=True,
        induction_attempts=1,
        new_transitions_since_induction=201,
        diversity_active=False,
    )
    fields.update(changes)
    policy = SimpleNamespace(trajectory_runtime_capabilities=lambda: {"llm_reinduction": available})
    return cap.with_capabilities(policy, TrajectorySnapshot(**fields))


@pytest.mark.parametrize("available", [False, True, None, "unknown"])
def test_capability_predicates_preserve_order_and_baseline(available):
    """SCENARIO-VERIFY-8536-WRAPPER: only explicit absence masks an arm."""
    supervisor = TrajectorySupervisor(window=1)
    state = snapshot(available)
    rows = supervisor.arm_eligibility(state)
    base = supervisor.arm_eligibility(
        TrajectorySnapshot(
            **{k: getattr(state, k) for k in TrajectorySnapshot.__dataclass_fields__}
        )
    )
    assert [r["arm"] for r in rows] == [r["arm"] for r in base]
    allow = next(r for r in rows if r["arm"] == "allow_reinduction")
    assert allow["baseline_eligible"] is True
    assert allow["eligible"] is (available is not False)
    assert allow["capability_eligible"] is (available if type(available) is bool else None)
    assert allow["capability_reason"] == (
        "runtime_explicitly_disabled"
        if available is False
        else "runtime_available"
        if available is True
        else "runtime_unknown"
    )
    redirect = supervisor.observe(state)
    assert redirect.arm == (
        "force_exploration_diversity" if available is False else "allow_reinduction"
    )
    if available is False:
        assert "allow_reinduction" not in supervisor.receipt()["arms_used"]


def test_unknown_missing_changing_and_exhaustion():
    """REQ-VERIFY-8536: availability is sampled afresh and absence is not exhaustion."""
    plain = TrajectorySnapshot(0, False, True, 1, 201, True)
    unknown = cap.with_capabilities(SimpleNamespace(), plain)
    assert unknown.llm_reinduction_available is None
    assert (
        cap.with_capabilities(
            SimpleNamespace(trajectory_runtime_capabilities=lambda: None), plain
        ).llm_reinduction_available
        is None
    )
    supervisor = TrajectorySupervisor(window=1)
    assert supervisor.observe(snapshot(False, diversity_active=True)) is None
    window = supervisor.receipt()["unredirected_windows"][-1]
    assert window["selection_status"] == "no_currently_executable_arm"
    assert window["arms_used"] == []
    assert supervisor.observe(snapshot(True, diversity_active=True)).arm == "allow_reinduction"
    supervisor._arms_used.update(supervisor.receipt()["arms_enabled"])
    assert supervisor.observe(snapshot(False, diversity_active=True)) is None
    assert (
        supervisor.receipt()["unredirected_windows"][-1]["selection_status"] == "all_arms_attempted"
    )
    assert supervisor.observe(snapshot(False, level=1)) is None
    assert supervisor.receipt()["arms_used"] == []
    assert supervisor.observe(snapshot(False, level=1)).arm == "force_exploration_diversity"
    assert set(asdict(plain)) == set(TrajectorySnapshot.__dataclass_fields__)


def assert_disabled_selection():
    """One assertion is shared by the genuine selector and its rejecting mutation."""
    assert (
        TrajectorySupervisor(window=1).observe(snapshot(False)).arm == "force_exploration_diversity"
    )


def test_disabled_capability_mutant_is_killed():
    """SCENARIO-VERIFY-8536-WRAPPER: ignoring model absence must fail the new test."""
    assert_disabled_selection()
    with patch.object(cap, "filter_rows", side_effect=lambda rows, _: rows):
        with pytest.raises(AssertionError):
            assert_disabled_selection()


def test_real_wrapper_selection_and_receipt_parity(tmp_path):
    """SCENARIO-VERIFY-8536-WRAPPER: use actual factory, policy and returned actions."""
    from carnot.reporting import arc_capability_fixtures_8536 as fixtures

    result = fixtures.collect(tmp_path)
    assert all(r["passed"] for r in result["action_parity_rows"])
    cases = {(r["case"], r["masked"]): r for r in result["selection_fixture_rows"]}
    assert cases["disabled", False]["selected_arms"][0] == "allow_reinduction"
    assert cases["disabled", True]["selected_arms"][0] == "force_exploration_diversity"
    for case in ["enabled", "unknown"]:
        assert cases[case, True]["selected_arms"][0] == "allow_reinduction"
    assert cases["active", True]["selected_arms"] == []
    assert cases["shadow", True]["state"]["induced"] is True
    assert all(not r["applied"] for r in cases["shadow", True]["receipts"])
    assert cases["transition", True]["state"]["hybrid_diversity"] is True
    assert cases["changing", True]["selected_arms"] == [
        "force_exploration_diversity",
        "allow_reinduction",
    ]
    assert all(r["llm_calls"] == 0 for r in cases.values())
    assert fixtures.replay(result)


def test_fixture_model_tripwire_is_required(monkeypatch):
    """REQ-VERIFY-8536: attempted model access disqualifies the fixture itself."""
    from contextlib import contextmanager
    from carnot.reporting import arc_capability_fixtures_8536 as fixtures

    @contextmanager
    def denied():
        yield ["attempted_load"]

    monkeypatch.setattr(fixtures, "no_models", denied)
    with pytest.raises(ValueError, match="fixture_model_tripwire"):
        fixtures.run("disabled", True, True)


def test_unreadable_runtime_capability_is_unknown():
    """REQ-VERIFY-8536: a failed capability reader cannot become evidence of absence."""

    def unreadable():
        raise ValueError("unknown_runtime")

    plain = TrajectorySnapshot(0, False, True, 1, 201, False)
    state = cap.with_capabilities(
        SimpleNamespace(trajectory_runtime_capabilities=unreadable), plain
    )
    assert state.llm_reinduction_available is None
    assert TrajectorySupervisor(window=1).observe(state).arm == "allow_reinduction"


def test_default_off_wrapper_bytes_match_original_policy():
    """REQ-VERIFY-8536: execute the original method at the actual wrapper boundary."""
    import ast
    from pathlib import Path
    from carnot.agentic import arc_competition_agent as agent
    from carnot.reporting import arc_capability_fixtures_8536 as fixtures
    from carnot.reporting.current_work_receipt import canonical_hash

    path = (
        Path(__file__).resolve().parents[2]
        / "results/raw/experiment_8536_v733_arc_capability_qualification/historical/arc_competition_agent.py"
    )
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "E3AgentPolicy")
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_maybe_supervise_trajectory"
    )
    namespace = dict(vars(agent))
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    current = fixtures.run("disabled", False, True)
    assert current["actions"][0]["name"] == "RESET"
    assert all("data" in a for a in current["actions"])
    with patch.object(agent.E3AgentPolicy, "_maybe_supervise_trajectory", namespace[method.name]):
        original = fixtures.run("disabled", False, True)
    assert canonical_hash(current) == canonical_hash(original)
