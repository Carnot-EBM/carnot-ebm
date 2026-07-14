"""Tests for the goal_bias degenerate-score self-audit (REQ-ARC-FCP-5703-2,
GAP-5703 candidate design (a)).

Spec refs: REQ-ARC-FCP-5703-2, SCENARIO-ARC-FCP-5703-2-DEGENERATE-SELF-AUDIT.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from carnot.agentic.arc_competition_agent import StepwiseExplorer


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def test_req_arc_fcp_5703_2_spec_declares_degeneracy_audit() -> None:
    """REQ-ARC-FCP-5703-2: OpenSpec declares the self-audit contract, including that
    it is observability-only (does not disable goal_bias mid-episode)."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-FCP-5703-2") :]
    section = section[: section.index("### REQ-ARC-WMTE-5593-4")]

    for marker in (
        "SCENARIO-ARC-FCP-5703-2-DEGENERATE-SELF-AUDIT",
        "OBSERVABILITY ONLY",
        "score_variance",
    ):
        assert marker in section


def _node(value: float) -> dict:
    return {"frame": SimpleNamespace(value=value)}


def test_goal_bias_diagnostics_default_not_degenerate_when_disabled() -> None:
    explorer = StepwiseExplorer()
    diag = explorer.goal_bias_diagnostics()
    assert diag["enabled"] is False
    assert diag["nodes_scored"] == 0
    assert diag["degenerate"] is False
    assert diag["score_variance"] == 0.0
    assert diag["score_min"] is None
    assert diag["score_max"] is None


def test_goal_bias_diagnostics_not_degenerate_below_min_sample_floor() -> None:
    """A short episode that hasn't scored enough nodes must not be misread as
    degenerate -- there simply isn't enough evidence yet."""

    explorer = StepwiseExplorer()
    explorer.set_goal_bias(lambda frame: 1.0, label="constant_test_energy")

    for _ in range(5):  # well below the 20-sample floor
        explorer._goal_bias_score(_node(1.0))

    diag = explorer.goal_bias_diagnostics()
    assert diag["nodes_scored"] == 5
    assert diag["score_variance"] == 0.0
    assert diag["degenerate"] is False


def test_goal_bias_diagnostics_flags_degenerate_constant_score_past_min_samples() -> None:
    """A constant score across enough real invocations is flagged degenerate --
    the exact real-world shape found on sp80 (exp5703)."""

    explorer = StepwiseExplorer()
    explorer.set_goal_bias(lambda frame: 1.0, label="constant_test_energy")

    for _ in range(25):
        explorer._goal_bias_score(_node(1.0))

    diag = explorer.goal_bias_diagnostics()
    assert diag["nodes_scored"] == 25
    assert diag["score_variance"] == 0.0
    assert diag["score_min"] == 1.0
    assert diag["score_max"] == 1.0
    assert diag["degenerate"] is True


def test_goal_bias_diagnostics_does_not_flag_real_varying_signal() -> None:
    """A genuinely varying goal_bias must never be misclassified as degenerate."""

    explorer = StepwiseExplorer()
    explorer.set_goal_bias(lambda frame: float(frame.value), label="real_varying_energy")

    values = [0.1, 0.9, 0.3, 0.7, 0.5] * 6  # 30 samples, real variance
    for v in values:
        explorer._goal_bias_score(_node(v))

    diag = explorer.goal_bias_diagnostics()
    assert diag["nodes_scored"] == 30
    assert diag["score_variance"] > 1e-6
    assert diag["degenerate"] is False


def test_goal_bias_diagnostics_degeneracy_does_not_change_selection() -> None:
    """SCENARIO-ARC-FCP-5703-2-DEGENERATE-SELF-AUDIT: flagging degenerate must not
    itself alter which candidate the explorer selects -- this is observability only."""

    explorer = StepwiseExplorer()
    explorer.cur = "root"
    explorer.set_goal_bias(lambda frame: 1.0, label="constant_test_energy", lower_is_better=True)
    explorer.graph = {
        "shallow": {
            "path": [{"action": 1, "data": None}],
            "untested": [{"action": 2, "data": None}],
            "value": 0.0,
            "frame": SimpleNamespace(value=1.0),
        },
        "deep": {
            "path": [{"action": 1, "data": None}, {"action": 2, "data": None}],
            "untested": [{"action": 3, "data": None}],
            "value": 0.0,
            "frame": SimpleNamespace(value=1.0),
        },
    }

    before = explorer._frontier()
    for _ in range(25):
        explorer._goal_bias_score(_node(1.0))
    assert explorer.goal_bias_diagnostics()["degenerate"] is True
    after = explorer._frontier()

    assert before == after
