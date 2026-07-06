"""Tests for Exp 4557 executable world-model proposer re-induction.

Spec refs: REQ-ARC-WMTE-4557, SCENARIO-ARC-WMTE-4557-POSITIVE-CONTROL-FIRST.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _levels(lp85: int = 1, m0r0: int = 1, sp80: int = 1, vc33: int = 1) -> dict[str, int]:
    return {"lp85": lp85, "m0r0": m0r0, "sp80": sp80, "vc33": vc33}


def _measurement(
    label: str,
    *,
    levels: dict[str, int] | None = None,
    efficiency: float = 2.0074,
    planned: bool = False,
) -> dict[str, object]:
    levels = dict(levels or _levels())
    return {
        "measurement": label,
        "core_efficiency": efficiency,
        "deepest_level_by_game": levels,
        "per_game": [
            {
                "game": game,
                "best_level": level,
                "efficiency": efficiency / 4.0,
                "diagnostics": {
                    "induction_attempts": [
                        {
                            "reason": "level_up_reinduction",
                            "planned": planned,
                            "skipped": "" if planned else "heldout_transition_verification_failed",
                            "refinement_rounds_used": 2 if planned else 3,
                            "counterexamples": [{"kind": "heldout_transition_verification_failed"}],
                        }
                    ]
                },
            }
            for game, level in levels.items()
        ],
    }


def _preconditions() -> dict[str, object]:
    return {
        "agents_md_read": True,
        "codex_md_read": True,
        "offline_arcade_import_smoke": True,
        "qwen3_5_9b_mtp_gguf_cached": True,
        "qwen3_5_9b_mtp_gguf_path": "/models/Qwen3.5-9B-Q4_K_M.gguf",
        "llama_cpp_import": True,
        "llama_cpp_version": "0.3.29",
        "spec_has_req_4557": True,
        "ok": True,
    }


def _positive_control(passed: bool = True) -> dict[str, object]:
    return {
        "passed": passed,
        "executable_model_verified": passed,
        "reachable_plan": passed,
        "dsl_reachable_plan": False,
        "heldout_accuracy": 1.0 if passed else 0.0,
        "refinement_rounds_used": 1 if passed else 3,
        "source": "live_qwen_executable_world_model_fixture",
    }


def test_req_arc_wmte_4557_spec_declares_executable_proposer_contract() -> None:
    """REQ-ARC-WMTE-4557: OpenSpec anchors the executable proposer artifact."""

    from carnot import experiment_4557_executable_world_model_proposer as exp4557

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4557" in spec
    assert "SCENARIO-ARC-WMTE-4557-POSITIVE-CONTROL-FIRST" in spec
    assert exp4557.RESULT_RELATIVE_PATH in spec
    for field, principle in exp4557.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_req_arc_wmte_4557_rejects_unverified_heldout_candidate_before_planning() -> None:
    """REQ-ARC-WMTE-4557: held-out transition failures become CEGIS counterexamples."""

    from carnot.agentic.arc_executable_world_model import Transition
    from carnot.agentic.arc_llm_reinduction import execute_bounded_llm_reinduction

    class FakeProposer:
        model_specs = "Qwen3.5-9B-MTP GGUF (/models/Qwen3.5-9B-Q4_K_M.gguf)"

        def __init__(self) -> None:
            self.calls: list[str] = []
            self.induction_sizes: list[int] = []

        def induce(self, _game, transitions, _cell):
            self.calls.append("induce")
            self.induction_sizes.append(len(transitions))
            return True, "overfit executable candidate"

        def refactor(self, _game, counterexample):
            # REQ-ARC-WMTE-4544: refactor() must receive REAL per-transition mismatch
            # evidence (BEFORE/PREDICTED/OBSERVED deltas from WorldModelVerifier.score()),
            # not just a scalar heldout_accuracy summary -- this is the CEGIS counterexample
            # refactor_prompt() is built to consume.
            mismatch = counterexample.mismatches[0]
            self.calls.append(
                f"refactor:n={counterexample.n}:n_correct={counterexample.n_correct}:"
                f"mismatch_i={mismatch['i']}"
            )
            return True, "general executable candidate"

    transitions = [
        Transition(
            grid=np.array([[0]], dtype=np.int16),
            action=1,
            data=None,
            next_grid=np.array([[1]], dtype=np.int16),
            level_before=1,
            level_after=1,
        ),
        Transition(
            grid=np.array([[1]], dtype=np.int16),
            action=1,
            data=None,
            next_grid=np.array([[2]], dtype=np.int16),
            level_before=1,
            level_after=1,
        ),
    ]

    def overfit_prefix_only(grid, _action, _data):
        current = np.asarray(grid)
        if int(current[0, 0]) == 0:
            return current + 1
        return current

    def general_increment(grid, _action, _data):
        return np.asarray(grid) + 1

    engines = iter(
        [
            (overfit_prefix_only, lambda grid: bool(np.asarray(grid)[0, 0] >= 2)),
            (general_increment, lambda grid: bool(np.asarray(grid)[0, 0] >= 2)),
        ]
    )
    plan_calls = 0

    def plan_in_model(engine, goal, start_grid):
        nonlocal plan_calls
        plan_calls += 1
        grid = np.asarray(start_grid)
        path = []
        for _ in range(3):
            if bool(goal(grid)):
                return path
            grid = np.asarray(engine(grid.copy(), 1, None))
            path.append({"action": 1, "data": None})
        return path if bool(goal(grid)) else None

    proposer = FakeProposer()
    result = execute_bounded_llm_reinduction(
        game="fixture",
        transitions=transitions,
        cell=1,
        root_grid=np.array([[0]], dtype=np.int16),
        proposer=proposer,
        candidate_provider=lambda engine, goal: [("loaded", engine, goal)],
        load_engine=lambda _game: next(engines),
        plan_in_model=plan_in_model,
        max_rounds=3,
        min_heldout_accuracy=1.0,
    )

    assert result.planned is True
    assert result.refinement_rounds_used == 2
    assert result.counterexamples[0]["kind"] == "heldout_transition_verification_failed"
    # REQ-ARC-WMTE-4544: the counterexample must carry REAL per-transition mismatch evidence,
    # not just the scalar heldout_accuracy -- this is what makes refinement genuinely
    # counterexample-guided rather than "you're wrong, try again" with no concrete detail.
    assert result.counterexamples[0]["real_accuracy"] == 0.5
    assert result.counterexamples[0]["real_n"] == 2
    assert result.counterexamples[0]["real_n_correct"] == 1
    real_mismatches = result.counterexamples[0]["real_mismatches"]
    assert len(real_mismatches) == 1
    assert real_mismatches[0]["i"] == 1
    assert "true_change" in real_mismatches[0]
    assert "your_prediction_was_wrong_at" in real_mismatches[0]
    assert result.rounds[0]["accepted_by_heldout_verifier"] is False
    assert result.rounds[1]["accepted_by_heldout_verifier"] is True
    assert proposer.calls == ["induce", "refactor:n=2:n_correct=1:mismatch_i=1"]
    assert proposer.induction_sizes == [1]
    assert plan_calls == 1


def test_scenario_arc_wmte_4557_positive_control_failure_skips_measurement(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4557-POSITIVE-CONTROL-FIRST: failed gate exits honestly."""

    from carnot import experiment_4557_executable_world_model_proposer as exp4557

    def should_not_measure():
        raise AssertionError("measurement must not run before the positive control passes")

    artifact = exp4557.run(
        root=tmp_path,
        preconditions_checked=_preconditions(),
        measurement_runner=should_not_measure,
        positive_control_runner=lambda: _positive_control(False),
        now=lambda: 1.0,
    )

    assert artifact["positive_control_passed"] is False
    assert artifact["false_negative_risk_checked"] is False
    assert artifact["core_efficiency_best"] is None
    assert artifact["efficiency_delta"] is None
    assert artifact["chosen_submitted_config"] == "unchanged"
    assert artifact["honest_verdict"] == (
        "complete: executable_proposer_positive_control_failed_no_deeper_barrier_refined"
    )
    assert exp4557.artifact_schema_errors(artifact) == []
    assert (
        json.loads((tmp_path / exp4557.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
        == artifact
    )


def test_req_arc_wmte_4557_honest_null_records_value_after_passed_gate() -> None:
    """REQ-ARC-WMTE-4557: a no-deeper null is valid only after the positive control passes."""

    from carnot import experiment_4557_executable_world_model_proposer as exp4557

    artifact = exp4557.build_artifact(
        preconditions_checked=_preconditions(),
        offline_dsl_baseline=_measurement("offline_dsl_baseline"),
        executable_proposer=_measurement("executable_proposer", planned=True),
        llm_proposer_value={"count": 1, "opportunities": 2, "rate": 0.5, "events": ["lp85:L2"]},
        positive_control=_positive_control(True),
        offline_reproduction={},
        model_specs="Qwen3.5-9B-MTP GGUF (/models/Qwen3.5-9B-Q4_K_M.gguf)",
        refinement_rounds_used={"lp85": [2], "m0r0": [3], "sp80": [], "vc33": []},
        barrier_refinement="heldout_verified_plan_did_not_reach_deeper_core_level",
        random_seed=4557,
        duration_s=61.0,
    )

    assert artifact["honest_verdict"] == (
        "complete: executable_proposer_positive_control_passed_no_deeper_barrier_refined"
    )
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["core_efficiency_best"] == 2.0074
    assert artifact["efficiency_delta"] == 0.0
    assert "null_delta_methodology_note" in artifact
    assert artifact["llm_proposer_value"]["count"] == 1
    assert artifact["verifier_is_oracle"] is False
    assert artifact["chosen_submitted_config"] == "unchanged"
    assert exp4557.artifact_schema_errors(artifact) == []


def test_req_arc_wmte_4557_success_requires_l2_efficiency_preservation_and_replay() -> None:
    """REQ-ARC-WMTE-4557: strict CORE improvement is the only submitted-config path."""

    from carnot import experiment_4557_executable_world_model_proposer as exp4557

    artifact = exp4557.build_artifact(
        preconditions_checked=_preconditions(),
        offline_dsl_baseline=_measurement("offline_dsl_baseline", levels=_levels(lp85=1)),
        executable_proposer=_measurement(
            "executable_proposer",
            levels=_levels(lp85=2),
            efficiency=3.125,
            planned=True,
        ),
        llm_proposer_value={"count": 1, "opportunities": 1, "rate": 1.0, "events": ["lp85:L2"]},
        positive_control=_positive_control(True),
        offline_reproduction={"reproduced": True, "game": "lp85", "reached_level": 2},
        model_specs="Qwen3.5-9B-MTP GGUF (/models/Qwen3.5-9B-Q4_K_M.gguf)",
        refinement_rounds_used={"lp85": [1]},
        barrier_refinement="resolved: executable proposer reached L2",
        random_seed=4557,
        duration_s=61.0,
    )

    assert artifact["honest_verdict"] == (
        "success: executable_proposer_lp85_reached_L2_core_efficiency_3.1250_above_2.0074"
    )
    assert artifact["core_efficiency_best"] == 3.125
    assert artifact["efficiency_delta"] == 1.1176
    assert artifact["core_solves_preserved"] is True
    assert artifact["chosen_submitted_config"]["executable_world_model_proposer"] is True
    assert exp4557.artifact_schema_errors(artifact) == []

    dropped = dict(artifact)
    dropped["core_solves_preserved"] = False
    assert any(
        "core_solves_preserved" in error for error in exp4557.artifact_schema_errors(dropped)
    )
