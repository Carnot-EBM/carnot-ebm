"""Unit tests for the pure aggregation/verdict logic of experiment_5717
(the playbook-exemplar stall-induction A/B). Live induction is not exercised here
(that needs a GPU); these pin the honest-reporting logic: floor detection and the
outlier-fragility guard that refuses to call a direction on noise-dominated data.

Spec: REQ-ARC-WMTE-5717, SCENARIO-ARC-WMTE-5717.
"""

from __future__ import annotations

from carnot import experiment_5717_playbook_exemplars_stall_induction_ab as exp


def _rows(control_recalls, treatment_recalls):
    rows = []
    for i, r in enumerate(control_recalls):
        rows.append(
            {
                "exemplars": False,
                "trial": i,
                "induction_ok": True,
                "reproduction_accuracy": 0.0,
                "cell_recall": r,
                "induce_s": 1.0,
            }
        )
    for i, r in enumerate(treatment_recalls):
        rows.append(
            {
                "exemplars": True,
                "trial": i,
                "induction_ok": True,
                "reproduction_accuracy": 0.0,
                "cell_recall": r,
                "induce_s": 1.0,
            }
        )
    return rows


def test_arm_summary_aggregates_ok_rate_and_recall():
    rows = _rows([0.2, 0.0], [0.1, 0.3])
    control = exp._arm_summary(rows, exemplars=False)
    assert control["runs"] == 2
    assert control["induction_ok_rate"] == 1.0
    assert control["mean_cell_recall"] == 0.1
    assert control["max_cell_recall"] == 0.2


def _verdict_for(control_recalls, treatment_recalls):
    rows = _rows(control_recalls, treatment_recalls)
    control = exp._arm_summary(rows, exemplars=False)
    treatment = exp._arm_summary(rows, exemplars=True)
    return exp._verdict(control, treatment, rows, len(rows))


def test_verdict_floored_when_both_arms_near_zero():
    v, delta, floored, fragile = _verdict_for([0.0, 0.01], [0.0, 0.0])
    assert floored is True
    assert "metric_floored_inconclusive" in v


def test_verdict_outlier_fragile_refuses_direction():
    # A single 0.7 control outlier dominates the mean; removing it flips the sign -> not reliable.
    v, delta, floored, fragile = _verdict_for([0.7, 0.0, 0.0, 0.0], [0.02, 0.0, 0.0, 0.0])
    assert floored is False  # control mean 0.175 is above the floor
    assert fragile is True
    assert "no_reliable_signal_high_variance" in v


def test_verdict_calls_improved_when_robust_and_positive():
    # Tight, above-floor recalls with a real positive gap that leave-one-out cannot explain away.
    v, delta, floored, fragile = _verdict_for([0.10, 0.12, 0.11, 0.09], [0.20, 0.22, 0.21, 0.19])
    assert floored is False and fragile is False
    assert delta > 0.02
    assert "improved_cellrecall" in v


def test_verdict_calls_hurt_when_robust_and_negative():
    v, delta, floored, fragile = _verdict_for([0.20, 0.22, 0.21, 0.19], [0.10, 0.12, 0.11, 0.09])
    assert floored is False and fragile is False
    assert delta < -0.02
    assert "hurt_cellrecall" in v


def test_verdict_no_scored_runs():
    v, delta, floored, fragile = exp._verdict(
        exp._arm_summary([], exemplars=False), exp._arm_summary([], exemplars=True), [], 0
    )
    assert delta is None
    assert "no_scored_runs" in v


def test_checksum_is_deterministic():
    a = exp._checksum({"x": [1, 2], "y": "z"})
    b = exp._checksum({"y": "z", "x": [1, 2]})
    assert a == b and a.startswith("sha256:")


def test_verdict_terminal_prefix_on_all_paths():
    # Every verdict this experiment can emit MUST start with a terminal prefix.
    cases = [
        _verdict_for([0.0], [0.0])[0],  # floored
        _verdict_for([0.7, 0.0], [0.0, 0.0])[0],  # fragile
        _verdict_for([0.10, 0.12, 0.11, 0.09], [0.20, 0.22, 0.21, 0.19])[0],  # improved
        exp._verdict(
            exp._arm_summary([], exemplars=False), exp._arm_summary([], exemplars=True), [], 0
        )[0],  # no scored runs
    ]
    for verdict in cases:
        assert verdict.startswith("complete")
