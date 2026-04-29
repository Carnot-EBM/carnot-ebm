"""Unit tests for the roadmap schema + prior-failures linter.

Why these tests exist: the planner Sonnet has dropped a required
`prior_failures` field on 10 occasions across milestones .77/.78/.79,
each time costing the conductor a DOOMED_RERUN_BLOCK and an empty task
slot. These tests exercise the mechanical guard that catches such
omissions before the conductor ever sees the task.

REQ: roadmap-schema-validation.md
SCENARIO: planner emits incomplete YAML; linter rejects.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

# Make scripts/ importable when running pytest from repo root.
_REPO_ROOT = Path(__file__).parent.parent.parent
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from roadmap_schema import GateSpec, PriorFailure, ResearchTask, Roadmap  # noqa: E402
from validate_prior_failures import validate_roadmap  # noqa: E402


def _valid_task_kwargs(**overrides):
    """Build a minimal-valid ResearchTask kwargs dict, with optional overrides.

    Encapsulating this here keeps each test focused on the *one* field it
    is exercising; the rest of the model stays at default-valid values so
    a schema change to an unrelated field doesn't ripple through the suite.
    """
    base = dict(
        id="exp9999-test-task",
        milestone="2026.04.99",
        deliverable="results/experiment_9999_test.json",
        title="Test Task Title For Unit Tests",
        prompt="A non-empty prompt body suffices for schema testing.",
    )
    base.update(overrides)
    return base


def test_valid_research_task_validates_without_error():
    """REQ-INFRA-081: A task with all required fields and well-formed deliverable validates."""
    task = ResearchTask(**_valid_task_kwargs())
    assert task.id == "exp9999-test-task"
    assert task.deliverable.startswith("results/")
    assert task.deliverable.endswith(".json")
    assert task.prior_failures == []  # default
    assert task.gated_on == []  # default


def test_invalid_gate_op_raises_validation_error():
    """REQ-INFRA-081: GateSpec.op must be one of the allowed comparison operators.

    A planner that invents a novel op like '==>' or 'contains' should be
    caught at parse time, not silently fail-open at gate-check time.
    """
    with pytest.raises(ValidationError):
        GateSpec(upstream="exp1.foo", artifact_field="ok", op="===", value=True)
    # Sanity: a known-good op does NOT raise.
    GateSpec(upstream="exp1.foo", artifact_field="ok", op="==", value=True)


def test_deliverable_must_start_with_results_and_end_json():
    """REQ-INFRA-081: Mis-shaped deliverable paths fail validation.

    The conductor's _deliverable_exists() probe assumes the deliverable
    lives under results/ and is JSON; non-conforming paths silently never
    match.
    """
    with pytest.raises(ValidationError):
        ResearchTask(**_valid_task_kwargs(deliverable="docs/exp_9999_test.md"))
    with pytest.raises(ValidationError):
        ResearchTask(**_valid_task_kwargs(deliverable="results/exp_9999_test.txt"))


def test_linter_reports_violations_on_synthetic_79_style_roadmap(tmp_path):
    """REQ-INFRA-081: Linter flags tasks that overlap archived experiments without prior_failures.

    We synthesize a small failure-ledger and a small roadmap, then assert
    that the linter produces at least 2 prior-failure violations — the
    same shape as the .79 issue with Exps 1016 and 1021.
    """
    # Synthesize a failure ledger with two prior experiments whose titles
    # share >=3 distinctive tokens with the roadmap tasks below.
    fake_complete = tmp_path / "research-complete.yaml"
    fake_complete.write_text(
        yaml.safe_dump(
            {
                "milestones": [
                    {
                        "id": "2026.04.79",
                        "title": "Synthetic prior milestone",
                        "tasks": [
                            {
                                "id": "exp1016-prior",
                                "title": "Cascade Routing Verifier Probe Wide Corpus",
                            },
                            {
                                "id": "exp1021-prior",
                                "title": "Energy Selection SSD Continuous Distillation Loop",
                            },
                        ],
                    },
                ],
            }
        )
    )

    # Roadmap with two tasks that match the prior scope but lack prior_failures.
    fake_roadmap = tmp_path / "research-roadmap-79.yaml"
    fake_roadmap.write_text(
        yaml.safe_dump(
            {
                "milestone": "2026.04.79",
                "milestone_title": "Test milestone",
                "milestone_doc": "openspec/change-proposals/test.md",
                "tasks": [
                    {
                        "id": "exp1016-cascade-rerun",
                        "milestone": "2026.04.79",
                        "deliverable": "results/experiment_1016_cascade.json",
                        "title": "Cascade Routing Verifier Probe Wide Corpus v2",
                        "prompt": "Re-attempt cascade routing verifier probe.",
                    },
                    {
                        "id": "exp1021-energy-ssd-rerun",
                        "milestone": "2026.04.79",
                        "deliverable": "results/experiment_1021_energy_ssd.json",
                        "title": "Energy Selection SSD Continuous Distillation Loop v3",
                        "prompt": "Re-attempt energy-selection self-distillation loop.",
                    },
                ],
            }
        )
    )

    schema_errors, prior_violations = validate_roadmap(fake_roadmap, complete_path=fake_complete)
    assert schema_errors == [], f"Unexpected schema errors: {schema_errors}"
    assert len(prior_violations) >= 2, (
        f"Expected >=2 prior-failure violations, got {len(prior_violations)}: {prior_violations}"
    )
    joined = " ".join(prior_violations)
    assert "exp1016-cascade-rerun" in joined
    assert "exp1021-energy-ssd-rerun" in joined


def test_linter_passes_clean_on_active_80_roadmap():
    """REQ-INFRA-081: The current research-roadmap.yaml (milestone .80) must lint clean.

    This is the regression guard: every commit that touches the active
    roadmap or the linter is checked against the real .80 roadmap, so
    accidental scope-overlap (or schema errors introduced by an edit)
    surface immediately.
    """
    active_roadmap = _REPO_ROOT / "research-roadmap.yaml"
    if not active_roadmap.exists():
        pytest.skip("research-roadmap.yaml not present in repo")

    schema_errors, prior_violations = validate_roadmap(active_roadmap)
    assert schema_errors == [], f"Active .80 roadmap has schema errors: {schema_errors}"
    assert prior_violations == [], (
        f"Active .80 roadmap has prior-failure violations:\n  " + "\n  ".join(prior_violations)
    )


def test_roadmap_milestone_must_match_each_task_milestone():
    """REQ-INFRA-081: A task whose milestone field disagrees with the roadmap's is rejected."""
    with pytest.raises(ValidationError):
        Roadmap(
            milestone="2026.04.99",
            milestone_title="Mismatch test",
            milestone_doc="openspec/test.md",
            tasks=[
                ResearchTask(**_valid_task_kwargs(milestone="2026.04.42")),
            ],
        )


def test_prior_failure_round_trips_through_yaml(tmp_path):
    """REQ-INFRA-081: A task with a fully-populated prior_failures block validates and round-trips."""
    pf = PriorFailure(
        experiment_id="exp850-sota-code-repair-v5",
        verdict="model_not_cached",
        addressed_by="Exp 855 LIVE-ENV permanent fix; this attempt downloads model first.",
        retire_if_same_verdict=True,
    )
    task = ResearchTask(**_valid_task_kwargs(prior_failures=[pf]))
    assert task.prior_failures[0].retire_if_same_verdict is True
    # Round-trip via dict (what yaml.safe_dump consumes)
    dumped = task.model_dump()
    rehydrated = ResearchTask.model_validate(dumped)
    assert rehydrated.prior_failures[0].experiment_id == "exp850-sota-code-repair-v5"


# ---------------------------------------------------------------------------
# Differential agent routing — `model` field validation
# ---------------------------------------------------------------------------


def test_model_field_default_is_none():
    """REQ-INFRA-082 / SCENARIO-INFRA-082-A: Absence of model field falls through to conductor's AGENT_MODEL (Sonnet)."""
    task = ResearchTask(**_valid_task_kwargs())
    assert task.model is None


def test_model_field_accepts_sonnet():
    """REQ-INFRA-082 / SCENARIO-INFRA-082-B: Explicit Sonnet routing for routine experiments."""
    task = ResearchTask(**_valid_task_kwargs(model="sonnet"))
    assert task.model == "sonnet"


def test_model_field_accepts_opus():
    """REQ-INFRA-082 / SCENARIO-INFRA-082-C: Pre-emptive Opus routing for complex / hardware-integration / schema tasks."""
    task = ResearchTask(**_valid_task_kwargs(model="opus"))
    assert task.model == "opus"


def test_model_field_rejects_unknown_string():
    """REQ-INFRA-082 / SCENARIO-INFRA-082-D: Typos like 'sonet' or 'gpt-4' must fail validation, not silently default."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ResearchTask(**_valid_task_kwargs(model="sonet"))
    with pytest.raises(ValidationError):
        ResearchTask(**_valid_task_kwargs(model="haiku"))  # retired, see CLAUDE.md
    with pytest.raises(ValidationError):
        ResearchTask(**_valid_task_kwargs(model="gpt-4"))


def test_escalate_on_max_turns_default_true():
    """REQ-INFRA-082 / SCENARIO-INFRA-082-E: C+E pattern is on by default — Sonnet max-turns escalates to Opus."""
    task = ResearchTask(**_valid_task_kwargs())
    assert task.escalate_on_max_turns is True


def test_escalate_on_max_turns_can_disable():
    """REQ-INFRA-082 / SCENARIO-INFRA-082-F: Tasks that genuinely should not retry on max-turns can opt out."""
    task = ResearchTask(**_valid_task_kwargs(escalate_on_max_turns=False))
    assert task.escalate_on_max_turns is False


def test_model_opus_round_trips_through_yaml():
    """REQ-INFRA-082 / SCENARIO-INFRA-082-G: A task with model=opus survives YAML round-trip."""
    task = ResearchTask(**_valid_task_kwargs(model="opus"))
    dumped = task.model_dump()
    rehydrated = ResearchTask.model_validate(dumped)
    assert rehydrated.model == "opus"
