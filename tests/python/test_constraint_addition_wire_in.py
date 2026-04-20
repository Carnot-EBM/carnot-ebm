"""Tests for the ConstraintAdditionFromMemory wire-in to VerifyRepairPipeline.

Verifies REQ-LEARN-053, REQ-LEARN-054 and SCENARIO-LEARN-083/084/085:
- Pipeline constructed without constraint_memory is unaffected (SCENARIO-LEARN-083).
- Pipeline calls observe() for each violation (SCENARIO-LEARN-084).
- Pipeline calls check_and_add() each verify() cycle so patterns cross-activate (SCENARIO-LEARN-085).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pytest

from carnot.pipeline.constraint_addition import ConstraintAdditionFromMemory
from carnot.pipeline.verify_repair import VerifyRepairPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline(**kwargs) -> VerifyRepairPipeline:
    """Build a pipeline with no model, no real extractors — patched for speed."""
    return VerifyRepairPipeline(model=None, **kwargs)


def _verification_result_with_violations(violation_types: list[str]):
    """Return a fake VerificationResult whose violations list matches violation_types."""
    from carnot.pipeline.extract import ConstraintResult

    violations = [
        ConstraintResult(
            constraint_type=vtype,
            description=f"test violation {vtype}",
            metadata={},
        )
        for vtype in violation_types
    ]
    from carnot.pipeline.verify_repair import VerificationResult

    return VerificationResult(
        verified=False,
        constraints=violations,
        energy=1.0,
        violations=violations,
    )


def _verification_result_clean():
    """Return a VerificationResult with no violations."""
    from carnot.pipeline.verify_repair import VerificationResult

    return VerificationResult(
        verified=True,
        constraints=[],
        energy=0.0,
        violations=[],
    )


# ---------------------------------------------------------------------------
# SCENARIO-LEARN-083: constraint_memory=None leaves existing behavior intact
# ---------------------------------------------------------------------------


def test_no_constraint_memory_leaves_behavior_unchanged():
    """Pipeline without constraint_memory must not call any ConstraintAdditionFromMemory methods.

    WHY: The wire-in must be purely additive — existing callers that omit
    constraint_memory should see zero behavioral change (REQ-LEARN-053).
    """
    pipeline = _make_pipeline()
    assert pipeline._constraint_memory is None

    # Patch _evaluate_constraints so no real extraction is needed.
    clean_result = _verification_result_clean()
    with patch.object(pipeline, "_evaluate_constraints", return_value=clean_result), \
         patch.object(pipeline, "extract_constraints", return_value=[]), \
         patch.object(pipeline, "extract_typed_reasoning", return_value=None), \
         patch.object(pipeline, "verify_semantic_grounding", return_value=None), \
         patch.object(pipeline, "verify_semantic_verifier_v2", return_value=None):
        result = pipeline.verify("What is 2+2?", "4", domain="arithmetic")

    assert result.verified is True
    assert result.violations == []


# ---------------------------------------------------------------------------
# SCENARIO-LEARN-084: observe() is called once per violation
# ---------------------------------------------------------------------------


def test_observe_called_for_each_violation():
    """Pipeline must call constraint_memory.observe() for every violation found.

    WHY: ConstraintAdditionFromMemory can only accumulate evidence when it is
    notified of violations — REQ-LEARN-054 requires the pipeline to feed each
    violation into observe() so the pattern counter increments.
    """
    cm = ConstraintAdditionFromMemory(threshold=10)  # high threshold — won't fire
    pipeline = _make_pipeline(constraint_memory=cm)

    viol_result = _verification_result_with_violations(["carry:overflow", "sign:flip"])
    with patch.object(pipeline, "_evaluate_constraints", return_value=viol_result), \
         patch.object(pipeline, "extract_constraints", return_value=[]), \
         patch.object(pipeline, "extract_typed_reasoning", return_value=None), \
         patch.object(pipeline, "verify_semantic_grounding", return_value=None), \
         patch.object(pipeline, "verify_semantic_verifier_v2", return_value=None):
        pipeline.verify("q", "resp", domain="arithmetic")

    counts = cm.get_pattern_counts()
    # 'carry:overflow' → prefix 'carry'; 'sign:flip' → prefix 'sign'
    assert counts.get("carry", 0) == 1
    assert counts.get("sign", 0) == 1


def test_observe_accumulates_across_multiple_verify_calls():
    """Each verify() call accumulates observations — counts are additive.

    WHY: Cross-session learning depends on counts growing monotonically across
    calls within a session (REQ-LEARN-054).
    """
    cm = ConstraintAdditionFromMemory(threshold=10)
    pipeline = _make_pipeline(constraint_memory=cm)

    viol_result = _verification_result_with_violations(["carry:overflow"])
    with patch.object(pipeline, "_evaluate_constraints", return_value=viol_result), \
         patch.object(pipeline, "extract_constraints", return_value=[]), \
         patch.object(pipeline, "extract_typed_reasoning", return_value=None), \
         patch.object(pipeline, "verify_semantic_grounding", return_value=None), \
         patch.object(pipeline, "verify_semantic_verifier_v2", return_value=None):
        pipeline.verify("q1", "resp1", domain="arithmetic")
        pipeline.verify("q2", "resp2", domain="arithmetic")
        pipeline.verify("q3", "resp3", domain="arithmetic")

    assert cm.get_pattern_counts()["carry"] == 3


# ---------------------------------------------------------------------------
# SCENARIO-LEARN-085: check_and_add() is called during verify()
# ---------------------------------------------------------------------------


def test_check_and_add_called_during_verify():
    """Pipeline must call constraint_memory.check_and_add(self) each verify() call.

    WHY: check_and_add() promotes matured patterns into active constraints.
    Without calling it during verify(), patterns that crossed the threshold
    would never activate — the cross-session improvement loop would be broken
    (REQ-LEARN-054, SCENARIO-LEARN-085).
    """
    cm = MagicMock(spec=ConstraintAdditionFromMemory)
    cm.check_and_add.return_value = []
    cm.observe = MagicMock()

    pipeline = _make_pipeline(constraint_memory=cm)
    clean_result = _verification_result_clean()
    with patch.object(pipeline, "_evaluate_constraints", return_value=clean_result), \
         patch.object(pipeline, "extract_constraints", return_value=[]), \
         patch.object(pipeline, "extract_typed_reasoning", return_value=None), \
         patch.object(pipeline, "verify_semantic_grounding", return_value=None), \
         patch.object(pipeline, "verify_semantic_verifier_v2", return_value=None):
        pipeline.verify("q", "r", domain="arithmetic")

    cm.check_and_add.assert_called_once_with(pipeline)


def test_check_and_add_receives_pipeline_reference():
    """The pipeline passed to check_and_add() is the same VerifyRepairPipeline instance.

    WHY: check_and_add() uses the pipeline reference to register new constraints
    with template_library.observe_pattern().  Passing the wrong object breaks the
    activation chain (REQ-LEARN-054).
    """
    captured = []

    class CapturingMemory:
        def check_and_add(self, p):
            captured.append(p)
            return []

        def observe(self, vtype, step):
            pass

    cm = CapturingMemory()
    pipeline = _make_pipeline(constraint_memory=cm)
    clean_result = _verification_result_clean()
    with patch.object(pipeline, "_evaluate_constraints", return_value=clean_result), \
         patch.object(pipeline, "extract_constraints", return_value=[]), \
         patch.object(pipeline, "extract_typed_reasoning", return_value=None), \
         patch.object(pipeline, "verify_semantic_grounding", return_value=None), \
         patch.object(pipeline, "verify_semantic_verifier_v2", return_value=None):
        pipeline.verify("q", "r")

    assert len(captured) == 1
    assert captured[0] is pipeline


def test_no_observe_when_no_violations():
    """observe() must NOT be called when the result has no violations.

    WHY: Calling observe() on clean responses would inflate pattern counters
    with false evidence, causing spurious constraint additions (REQ-LEARN-054).
    """
    cm = MagicMock(spec=ConstraintAdditionFromMemory)
    cm.check_and_add.return_value = []

    pipeline = _make_pipeline(constraint_memory=cm)
    clean_result = _verification_result_clean()
    with patch.object(pipeline, "_evaluate_constraints", return_value=clean_result), \
         patch.object(pipeline, "extract_constraints", return_value=[]), \
         patch.object(pipeline, "extract_typed_reasoning", return_value=None), \
         patch.object(pipeline, "verify_semantic_grounding", return_value=None), \
         patch.object(pipeline, "verify_semantic_verifier_v2", return_value=None):
        pipeline.verify("q", "r")

    cm.observe.assert_not_called()
