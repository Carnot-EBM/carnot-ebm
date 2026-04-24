"""Tests for Exp 802: FR-11 Embedding Relay — 10-session Tier 1 relay.

Every test traces to REQ-LEARN-098 (FR-11 Tier 1 relay MUST use EmbeddingConstraintStore).

Coverage targets:
  - compute_session_precision: tp+fp=0 case (returns 1.0), normal case
  - is_monotonically_non_decreasing: identifies non-decreasing and decreasing sequences
  - compute_honest_verdict: all three branches (works, partial, plateau)
  - build_session_questions: returns correct structure
  - update_store_from_violations: adds constraints for known error types
  - run_session: returns (tp, fp, violation_events) with expected types
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO / "scripts"
for _p in [str(_REPO), str(_SCRIPTS)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# compute_session_precision
# ---------------------------------------------------------------------------


def test_precision_zero_positives_returns_one():
    """compute_session_precision returns 1.0 when tp=0 and fp=0 (no positives).

    Why 1.0: if the pipeline never accepted anything, there were no false alarms —
    precision is technically undefined but treated as perfect to avoid division-by-zero.

    Spec: REQ-LEARN-098
    """
    from experiment_802_fr11_embedding_relay import compute_session_precision

    assert compute_session_precision(0, 0) == 1.0


def test_precision_all_true_positives():
    """compute_session_precision returns 1.0 when fp=0 and tp>0.

    Spec: REQ-LEARN-098
    """
    from experiment_802_fr11_embedding_relay import compute_session_precision

    assert compute_session_precision(5, 0) == 1.0


def test_precision_mixed():
    """compute_session_precision returns tp/(tp+fp) for normal inputs.

    Spec: REQ-LEARN-098
    """
    from experiment_802_fr11_embedding_relay import compute_session_precision

    result = compute_session_precision(3, 1)
    assert abs(result - 0.75) < 1e-9


def test_precision_all_false_positives():
    """compute_session_precision returns 0.0 when tp=0 and fp>0.

    Spec: REQ-LEARN-098
    """
    from experiment_802_fr11_embedding_relay import compute_session_precision

    assert compute_session_precision(0, 5) == 0.0


# ---------------------------------------------------------------------------
# is_monotonically_non_decreasing
# ---------------------------------------------------------------------------


def test_monotonic_non_decreasing_flat():
    """Flat sequence [0.5, 0.5, 0.5] is non-decreasing.

    Spec: REQ-LEARN-098
    """
    from experiment_802_fr11_embedding_relay import is_monotonically_non_decreasing

    assert is_monotonically_non_decreasing([0.5, 0.5, 0.5]) is True


def test_monotonic_non_decreasing_strictly_increasing():
    """Strictly increasing sequence [0.4, 0.6, 0.8] is non-decreasing.

    Spec: REQ-LEARN-098
    """
    from experiment_802_fr11_embedding_relay import is_monotonically_non_decreasing

    assert is_monotonically_non_decreasing([0.4, 0.6, 0.8]) is True


def test_monotonic_non_decreasing_drop():
    """Sequence with a drop [0.5, 0.7, 0.6] is NOT non-decreasing.

    Spec: REQ-LEARN-098
    """
    from experiment_802_fr11_embedding_relay import is_monotonically_non_decreasing

    assert is_monotonically_non_decreasing([0.5, 0.7, 0.6]) is False


def test_monotonic_single_element():
    """Single-element list is trivially non-decreasing.

    Spec: REQ-LEARN-098
    """
    from experiment_802_fr11_embedding_relay import is_monotonically_non_decreasing

    assert is_monotonically_non_decreasing([0.7]) is True


def test_monotonic_two_elements_equal():
    """Two equal elements [0.5, 0.5] is non-decreasing.

    Spec: REQ-LEARN-098
    """
    from experiment_802_fr11_embedding_relay import is_monotonically_non_decreasing

    assert is_monotonically_non_decreasing([0.5, 0.5]) is True


# ---------------------------------------------------------------------------
# compute_honest_verdict
# ---------------------------------------------------------------------------


def test_verdict_tier1_relay_works():
    """compute_honest_verdict returns 'tier1_relay_works' when monotonic AND delta positive.

    Spec: REQ-LEARN-098, SCENARIO-LEARN-145
    """
    from experiment_802_fr11_embedding_relay import compute_honest_verdict

    precision = [0.5, 0.55, 0.60, 0.63, 0.65, 0.67, 0.68, 0.70, 0.71, 0.73]
    assert compute_honest_verdict(precision, monotonic=True, delta_positive_by_s5=True) == "tier1_relay_works"


def test_verdict_tier1_partial_improvement():
    """compute_honest_verdict returns 'tier1_partial_improvement' when delta positive but not monotonic.

    Spec: REQ-LEARN-098, SCENARIO-LEARN-145
    """
    from experiment_802_fr11_embedding_relay import compute_honest_verdict

    precision = [0.5, 0.55, 0.60, 0.63, 0.65, 0.67, 0.68, 0.70, 0.71, 0.73]
    assert compute_honest_verdict(precision, monotonic=False, delta_positive_by_s5=True) == "tier1_partial_improvement"


def test_verdict_tier1_plateau_persists():
    """compute_honest_verdict returns 'tier1_plateau_persists' when no delta by session 5.

    Spec: REQ-LEARN-098, SCENARIO-LEARN-145
    """
    from experiment_802_fr11_embedding_relay import compute_honest_verdict

    precision = [0.5] * 10
    assert compute_honest_verdict(precision, monotonic=True, delta_positive_by_s5=False) == "tier1_plateau_persists"


def test_verdict_plateau_overrides_not_monotonic():
    """When delta_positive_by_s5=False, verdict is plateau even if monotonic=False.

    Spec: REQ-LEARN-098
    """
    from experiment_802_fr11_embedding_relay import compute_honest_verdict

    precision = [0.5, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    assert compute_honest_verdict(precision, monotonic=False, delta_positive_by_s5=False) == "tier1_plateau_persists"


# ---------------------------------------------------------------------------
# build_session_questions
# ---------------------------------------------------------------------------


def test_build_session_questions_count():
    """build_session_questions returns exactly 50 questions per session.

    Spec: SCENARIO-LEARN-145
    """
    from experiment_802_fr11_embedding_relay import build_session_questions

    questions = build_session_questions()
    assert len(questions) == 50


def test_build_session_questions_fields():
    """Each question dict has required keys: question, response, is_correct, error_type.

    Spec: REQ-LEARN-098
    """
    from experiment_802_fr11_embedding_relay import build_session_questions

    questions = build_session_questions()
    for q in questions:
        assert "question" in q
        assert "response" in q
        assert "is_correct" in q
        assert "error_type" in q


def test_build_session_questions_has_both_correct_and_errors():
    """Session questions include both correct (is_correct=True) and error instances.

    Spec: SCENARIO-LEARN-145
    """
    from experiment_802_fr11_embedding_relay import build_session_questions

    questions = build_session_questions()
    correct = [q for q in questions if q["is_correct"]]
    errors = [q for q in questions if not q["is_correct"]]
    assert len(correct) == 25
    assert len(errors) == 25


# ---------------------------------------------------------------------------
# update_store_from_violations
# ---------------------------------------------------------------------------


def test_update_store_adds_constraints_for_known_types():
    """update_store_from_violations adds one SPO per known error_type event.

    Spec: REQ-LEARN-098
    """
    from carnot.pipeline.embedding_constraint_store import EmbeddingConstraintStore
    from experiment_802_fr11_embedding_relay import update_store_from_violations

    store = EmbeddingConstraintStore()
    initial_size = len(store._store)
    events = [
        {"error_type": "carry", "query": "37 + 45 = 72"},
        {"error_type": "sign", "query": "5 - (-3) = 2"},
    ]
    added = update_store_from_violations(store, events)
    assert added == 2
    assert len(store._store) == initial_size + 2


def test_update_store_ignores_unknown_types():
    """update_store_from_violations skips events with unrecognized error_type.

    Spec: REQ-LEARN-098
    """
    from carnot.pipeline.embedding_constraint_store import EmbeddingConstraintStore
    from experiment_802_fr11_embedding_relay import update_store_from_violations

    store = EmbeddingConstraintStore()
    events = [
        {"error_type": "unknown_type", "query": "some response"},
        {"error_type": "another_unknown", "query": "other response"},
    ]
    added = update_store_from_violations(store, events)
    assert added == 0
    assert len(store._store) == 0


def test_update_store_empty_events():
    """update_store_from_violations with empty list returns 0 and leaves store unchanged.

    Spec: REQ-LEARN-098
    """
    from carnot.pipeline.embedding_constraint_store import EmbeddingConstraintStore
    from experiment_802_fr11_embedding_relay import update_store_from_violations

    store = EmbeddingConstraintStore()
    added = update_store_from_violations(store, [])
    assert added == 0
    assert len(store._store) == 0


# ---------------------------------------------------------------------------
# run_session
# ---------------------------------------------------------------------------


def test_run_session_returns_correct_types():
    """run_session returns (int, int, list) with tp and fp non-negative.

    Spec: REQ-LEARN-098
    """
    from carnot.models.ising import IsingConfig, IsingModel
    from carnot.pipeline.embedding_constraint_store import EmbeddingConstraintStore
    from experiment_802_fr11_embedding_relay import build_session_questions, run_session

    store = EmbeddingConstraintStore()
    store.from_casememory_patterns(
        {"carry": 4, "sign": 4, "unit": 4, "comparison": 4, "causal": 4}
    )
    ising = IsingModel(IsingConfig(input_dim=32, coupling_init="xavier_uniform"))
    questions = build_session_questions()

    tp, fp, violation_events = run_session(questions, store, ising)

    assert isinstance(tp, int)
    assert isinstance(fp, int)
    assert isinstance(violation_events, list)
    assert tp >= 0
    assert fp >= 0


def test_run_session_violation_events_have_error_type():
    """Each violation event dict has 'error_type' and 'query' keys.

    Spec: REQ-LEARN-098
    """
    from carnot.models.ising import IsingConfig, IsingModel
    from carnot.pipeline.embedding_constraint_store import EmbeddingConstraintStore
    from experiment_802_fr11_embedding_relay import build_session_questions, run_session

    store = EmbeddingConstraintStore()
    store.from_casememory_patterns({"carry": 4, "sign": 4, "unit": 4})
    ising = IsingModel(IsingConfig(input_dim=32, coupling_init="xavier_uniform"))
    questions = build_session_questions()

    _tp, _fp, violation_events = run_session(questions, store, ising)

    for event in violation_events:
        assert "error_type" in event
        assert "query" in event
