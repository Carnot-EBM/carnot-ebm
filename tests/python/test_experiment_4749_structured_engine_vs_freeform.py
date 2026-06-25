"""Tests for Exp 4749 structured engine vs free-form engine.

Spec refs: REQ-ARC-WMTE-4749,
SCENARIO-ARC-WMTE-4749-STRUCTURED-ENGINE-ADAPTER,
SCENARIO-ARC-WMTE-4749-LIVE-WIRING,
SCENARIO-ARC-WMTE-4749-ACCURACY-ARTIFACT.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


pytestmark = pytest.mark.memory_watchdog_skip

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _fixture_transitions():
    from carnot.agentic.arc_executable_world_model import Transition

    return [
        Transition(
            grid=np.array([[0, 0]], dtype=np.int16),
            action=1,
            data=None,
            next_grid=np.array([[1, 1]], dtype=np.int16),
            level_before=1,
            level_after=1,
        ),
        Transition(
            grid=np.array([[0, 0]], dtype=np.int16),
            action=1,
            data=None,
            next_grid=np.array([[1, 1]], dtype=np.int16),
            level_before=1,
            level_after=1,
        ),
        Transition(
            grid=np.array([[2, 2]], dtype=np.int16),
            action=2,
            data=None,
            next_grid=np.array([[2, 2]], dtype=np.int16),
            level_before=1,
            level_after=1,
        ),
    ]


class FakeProgrammaticProposer:
    model_specs = "Qwen3.5-9B-MTP fake"

    def induce_programmatic_experts(self, **_: Any) -> list[dict[str, Any]]:
        return [
            {
                "name": "stable_zero_to_one",
                "object_class": "color_0",
                "kind": "color_rewrite",
                "action": 1,
                "from_color": 0,
                "to_color": 1,
            }
        ]


def test_req_arc_wmte_4749_spec_declares_artifact_contract() -> None:
    """REQ-ARC-WMTE-4749: OpenSpec anchors the 4749 adapter and artifact."""

    from carnot import experiment_4749_structured_engine_vs_freeform as exp4749

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4749" in spec
    assert "SCENARIO-ARC-WMTE-4749-STRUCTURED-ENGINE-ADAPTER" in spec
    assert "SCENARIO-ARC-WMTE-4749-LIVE-WIRING" in spec
    assert "SCENARIO-ARC-WMTE-4749-ACCURACY-ARTIFACT" in spec
    assert exp4749.RESULT_RELATIVE_PATH in spec
    for field, principle in exp4749.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4749_structured_adapter_returns_product_engine() -> None:
    """SCENARIO-ARC-WMTE-4749-STRUCTURED-ENGINE-ADAPTER: adapter is shaped and live."""

    from carnot.agentic.arc_executable_world_model import ProductWorldModel
    from carnot.agentic.arc_structured_world_model import (
        structured_engine_non_degenerate,
        structured_load_engine,
    )

    goal = lambda grid: bool(np.all(np.asarray(grid) == 1))
    engine, returned_goal = structured_load_engine(
        "fixture",
        transitions=_fixture_transitions(),
        proposer=FakeProgrammaticProposer(),
        cell=1,
        goal=goal,
    )

    assert returned_goal is goal
    assert isinstance(getattr(engine, "__self__", None), ProductWorldModel)
    assert list(inspect.signature(engine).parameters) == ["grid", "action", "data"]

    predicted = engine(np.array([[0, 0]], dtype=np.int16), 1, None)
    assert np.array_equal(predicted, np.array([[1, 1]], dtype=np.int16))
    assert structured_engine_non_degenerate(engine, _fixture_transitions()) is True


def test_scenario_arc_wmte_4749_accuracy_measure_beats_identity_freeform() -> None:
    """SCENARIO-ARC-WMTE-4749-ACCURACY-ARTIFACT: same transitions compare both engines."""

    from carnot.agentic.arc_structured_world_model import (
        StructuredEngineReinductionProposer,
        build_structured_engine,
        heldout_transition_split,
        make_structured_load_engine,
        measure_engine_accuracy,
        normalise_transition,
        structured_engine_non_degenerate,
        structured_load_engine,
    )

    transitions = _fixture_transitions()
    structured_engine, _goal = structured_load_engine(
        "fixture",
        transitions=transitions,
        proposer=FakeProgrammaticProposer(),
        cell=1,
        goal=lambda _grid: False,
    )
    identity_engine = lambda grid, _action, _data=None: np.asarray(grid).copy()

    assert measure_engine_accuracy(identity_engine, transitions[:2]) == 0.0
    assert measure_engine_accuracy(structured_engine, transitions[:2]) == 1.0
    assert measure_engine_accuracy(identity_engine, []) == 0.0
    assert measure_engine_accuracy(lambda *_args: (_ for _ in ()).throw(RuntimeError("boom")), transitions[:1]) == 0.0
    assert structured_engine_non_degenerate(lambda *_args: (_ for _ in ()).throw(RuntimeError("boom")), transitions[:1]) is False
    assert structured_engine_non_degenerate(identity_engine, transitions[:1]) is False
    assert heldout_transition_split(transitions[:1]) == (transitions[:1], transitions[:1])

    as_mapping = {
        "grid": np.array([[0]], dtype=np.int16),
        "action": 3,
        "data": {"x": 1},
        "next_grid": np.array([[1]], dtype=np.int16),
        "level_before": 2,
        "level_after": 3,
    }
    mapped = normalise_transition(as_mapping)
    assert mapped.action == 3
    assert mapped.data == {"x": 1}
    assert mapped.level_after == 3
    as_object = SimpleNamespace(
        grid=np.array([[4]], dtype=np.int16),
        action=4,
        data=None,
        next_grid=np.array([[5]], dtype=np.int16),
        level_before=0,
        level_after=0,
    )
    assert normalise_transition(as_object).action == 4

    built = build_structured_engine(
        "fixture",
        transitions=transitions,
        proposer=FakeProgrammaticProposer(),
        goal=None,
        fallback_goal_loader=lambda _game: (None, lambda grid: bool(np.any(np.asarray(grid) == 1))),
    )
    assert built.expert_trust_weights[0]["kept"] is True
    assert built.goal(np.array([[1]], dtype=np.int16)) is True

    built_false_goal = build_structured_engine(
        "fixture",
        transitions=transitions,
        proposer=FakeProgrammaticProposer(),
        goal=None,
        fallback_goal_loader=lambda _game: (None, "not-callable"),
    )
    assert built_false_goal.goal(np.array([[1]], dtype=np.int16)) is False

    def boom_loader(_game: str):
        raise FileNotFoundError(_game)

    built_missing_goal = build_structured_engine(
        "fixture",
        transitions=transitions,
        proposer=FakeProgrammaticProposer(),
        goal=None,
        fallback_goal_loader=boom_loader,
    )
    assert built_missing_goal.goal(np.array([[1]], dtype=np.int16)) is False

    load = make_structured_load_engine(
        game="fixture",
        transitions=transitions,
        proposer=FakeProgrammaticProposer(),
        goal=lambda _grid: False,
    )
    made_engine, made_goal = load("")
    assert made_goal(np.array([[1]], dtype=np.int16)) is False
    assert measure_engine_accuracy(made_engine, transitions[:2]) == 1.0

    base = SimpleNamespace(repo_substr="Qwen3.5-9B-MTP-GGUF", extra="delegated")
    shim = StructuredEngineReinductionProposer(base)
    assert shim.model_specs == "Qwen3.5-9B-MTP-GGUF"
    assert shim.extra == "delegated"
    assert shim.induce("fixture", transitions, 1)[0] is True
    assert shim.refactor("fixture", None)[0] is True


def test_scenario_arc_wmte_4749_live_reinduction_uses_env_gated_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-WMTE-4749-LIVE-WIRING: env gate swaps loader for level-up."""

    from carnot.agentic import arc_competition_agent as agent
    from carnot.agentic import arc_structured_world_model as structured_wm
    from carnot.agentic.arc_llm_reinduction import LlmReinductionResult

    captured: dict[str, Any] = {}
    marker_loader = lambda _game: (
        lambda grid, _action, _data=None: np.asarray(grid),
        lambda _grid: False,
    )

    def fake_make_structured_load_engine(**kwargs: Any):
        captured["structured_loader_kwargs"] = kwargs
        return marker_loader

    class FakeStructuredProposer:
        def __init__(self, base: Any) -> None:
            captured["structured_proposer_base"] = base

    def fake_reinduction(**kwargs: Any) -> LlmReinductionResult:
        captured.update(kwargs)
        return LlmReinductionResult(planned=False, skipped="no_reachable_plan_after_refinement")

    monkeypatch.setenv("CARNOT_ARC_STRUCTURED_ENGINE", "1")
    monkeypatch.setattr(structured_wm, "make_structured_load_engine", fake_make_structured_load_engine)
    monkeypatch.setattr(structured_wm, "StructuredEngineReinductionProposer", FakeStructuredProposer)
    monkeypatch.setattr(agent, "execute_bounded_llm_reinduction", fake_reinduction)

    policy = agent.E3AgentPolicy(
        "lp85",
        proposer=SimpleNamespace(model_specs="Qwen"),
        target_levels=2,
    )
    policy._pending_induction_reason = "level_up_reinduction"
    policy._start_level = 0
    policy._current_goal_level = 2
    policy._previous_level_complete_grid = np.array([[8]], dtype=np.int16)
    policy.root_grid = np.array([[0]], dtype=np.int16)
    policy.transitions = _fixture_transitions()

    policy._induce_and_plan()

    assert captured["load_engine"] is marker_loader
    assert captured["proposer"].__class__ is FakeStructuredProposer
    assert captured["structured_loader_kwargs"]["game"] == "lp85"
    assert len(captured["structured_loader_kwargs"]["transitions"]) == len(policy._active_transitions())
    assert captured["structured_loader_kwargs"]["transitions"][0] is policy._active_transitions()[0]
    assert captured["structured_loader_kwargs"]["cell"] == 1
    assert policy.induction_attempts[-1]["structured_engine_enabled"] is True


def test_scenario_arc_wmte_4749_artifact_schema_and_verdicts() -> None:
    """SCENARIO-ARC-WMTE-4749-ACCURACY-ARTIFACT: artifact fields are principle-gated."""

    from carnot import experiment_4749_structured_engine_vs_freeform as exp4749

    artifact = exp4749.build_artifact(
        preconditions_checked={
            "qwen3_5_9b_mtp_gguf_cached": True,
            "offline_arcade": True,
            "structured_symbols_importable": True,
        },
        structured_engine_non_degenerate=True,
        freeform_heldout_accuracy=0.12,
        structured_heldout_accuracy=1.0,
        l2_proposer_failed=False,
        offline_reproduced=False,
        solve_provenance="development_proxy",
        live_path_reachable=True,
        duration_s=60.0,
        target_game="lp85",
        expert_trust_weights=[{"name": "stable_zero_to_one", "trust": 1.0, "kept": True}],
    )

    assert exp4749.artifact_schema_errors(artifact) == []
    assert artifact["honest_verdict"].startswith("success_")
    assert artifact["verifier_is_oracle"] is False
    assert artifact["accuracy_delta"] == 0.88
    assert artifact["chosen_submitted_config"]["structured_engine_enabled"] is True

    banked = exp4749.build_artifact(
        preconditions_checked={"qwen3_5_9b_mtp_gguf_cached": True},
        structured_engine_non_degenerate=True,
        freeform_heldout_accuracy=0.12,
        structured_heldout_accuracy=0.2,
        l2_proposer_failed=False,
        offline_reproduced=True,
        solve_provenance="live_agent_self_discovery",
        live_path_reachable=True,
        duration_s=60.0,
        target_game="lp85",
        expert_trust_weights=[],
    )
    assert banked["honest_verdict"] == "success_structured_engine_l2_banked_lp85"

    null = exp4749.build_artifact(
        preconditions_checked={"qwen3_5_9b_mtp_gguf_cached": True},
        structured_engine_non_degenerate=False,
        freeform_heldout_accuracy=0.12,
        structured_heldout_accuracy=0.12,
        l2_proposer_failed=True,
        offline_reproduced=False,
        solve_provenance="development_proxy",
        live_path_reachable=False,
        duration_s=60.0,
        target_game="lp85",
        expert_trust_weights=[],
    )

    assert null["honest_verdict"].startswith("complete_")
    assert null["chosen_submitted_config"] == "unchanged"
    assert null["null_methodology_note"]

    bad = dict(artifact)
    bad["honest_verdict"] = "maybe"
    bad["verifier_is_oracle"] = True
    bad["structured_engine_non_degenerate"] = False
    bad["reproducibility_checksum"] = "sha256:bad"
    errors = exp4749.artifact_schema_errors(bad)
    assert "honest_verdict_terminal_prefix" in errors
    assert "verifier_is_oracle_false" in errors
    assert "structured_engine_non_degenerate" in errors
    assert "reproducibility_checksum" in errors

    bad_provenance = dict(null)
    bad_provenance["solve_provenance"] = "outer_loop_re"
    bad_provenance["null_methodology_note"] = ""
    bad_provenance["reproducibility_checksum"] = exp4749.payload_checksum(bad_provenance)
    errors = exp4749.artifact_schema_errors(bad_provenance)
    assert "solve_provenance" in errors
    assert "null_methodology_note" in errors
