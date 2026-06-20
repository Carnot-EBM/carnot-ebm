"""Tests for Exp 4475 submitted ARC stronger-stack integration artifact.

Spec refs: REQ-REPORT-4475-LIVE-STACK,
SCENARIO-REPORT-4475-LIVE-STACK-PARITY.
"""

from carnot import experiment_4475_wire_stronger_generic_stack as exp
from carnot.agentic.arc_competition_agent import SUBMITTED_AGENT_CONFIG


def test_build_artifact_emits_required_terminal_fields():
    """REQ-REPORT-4475-LIVE-STACK: artifact fields back the submitted default."""
    artifact = exp.build_artifact(
        before_generic_solve_rate=0.25,
        after_generic_solve_rate=0.375,
        before_solved=2,
        after_solved=3,
        attempted_games=8,
        reproduced_levels=34,
        offline_reproduced=True,
        preconditions_checked={
            "arcade_import": True,
            "submitted_agent_config_loaded": True,
            "env_game_blocked": True,
        },
        tests_pass=True,
        duration_s=1.25,
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == "verifier_ensemble_against_cached_candidates"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 34
    assert artifact["generic_solve_rate_delta"] == 0.125
    assert artifact["submitted_agent_config"] == SUBMITTED_AGENT_CONFIG
    assert set(exp.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert artifact["field_principles"][field]["principle"]
    assert exp.validate_artifact(artifact) == []


def test_validate_artifact_rejects_nonterminal_or_missing_required_fields():
    """REQ-REPORT-4475-LIVE-STACK: schema rejects reconciler-hostile artifacts."""
    artifact = exp.build_artifact(
        before_generic_solve_rate=0.0,
        after_generic_solve_rate=0.0,
        before_solved=0,
        after_solved=0,
        attempted_games=1,
        reproduced_levels=0,
        offline_reproduced=False,
        preconditions_checked={"arcade_import": False},
        tests_pass=False,
        duration_s=0.0,
    )
    artifact["honest_verdict"] = "done but not terminal-prefixed"
    del artifact["preconditions_checked"]

    errors = exp.validate_artifact(artifact)
    assert "honest_verdict missing terminal prefix" in errors
    assert "missing required field: preconditions_checked" in errors
