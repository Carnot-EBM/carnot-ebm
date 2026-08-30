"""REQ-INFRA-6850: a coerced task never keeps the model of the agent it was coerced away from.

INCIDENT 2026-08-30. `research_conductor.py` read `task_model` at line ~6579, rewrote
`task_agent_type` from gemini to codex in the coercion block below it, and never touched the
model. A task planned as `agent_type: gemini` + `model: gemini-3.1-pro-preview` was correctly
flipped to codex and then invoked with the Gemini model name:

    Codex CLI error: Model metadata for `gemini-3.1-pro-preview` not found

15 occurrences across 5 dates (2026-07-01, 07-05, 08-05, 08-27, 08-30), always exactly 3 per
date -- one task burning its full retry budget and retiring for an environmental reason.

The instructive part: the safety net WORKED and still caused this. Without the coercion the task
fails as gemini and is legible; with it, it fails as CODEX with an error that reads like a codex
problem, which is why five occurrences went untraced.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from research_conductor import coerced_model  # noqa: E402

DEFAULT = "gpt-5.6-sol"


def test_the_incident_shape_gets_the_new_agents_model() -> None:
    """The exact 2026-08-30 case."""
    assert coerced_model("gemini", "codex", "gemini-3.1-pro-preview", DEFAULT) == DEFAULT


def test_an_uncoerced_task_keeps_its_planned_model() -> None:
    """Coercion is the only trigger; a plan nobody overruled is left alone."""
    assert coerced_model("codex", "codex", "gpt-5.5", DEFAULT) == "gpt-5.5"


def test_claude_to_codex_also_drops_the_foreign_model() -> None:
    """The claude->codex branch has the same defect and the same fix."""
    assert coerced_model("claude", "codex", "opus", DEFAULT) == DEFAULT
    assert coerced_model("claude", "codex", "claude-sonnet-5", DEFAULT) == DEFAULT


def test_a_coerced_task_with_no_model_stays_unset() -> None:
    """None lets the callee pick its own default; inventing one would be a silent override."""
    assert coerced_model("gemini", "codex", None, DEFAULT) is None


def test_a_model_already_belonging_to_the_new_agent_survives() -> None:
    """Coercion must not clobber a correct model just because the agent changed."""
    assert coerced_model("gemini", "codex", "gpt-5.5", DEFAULT) == "gpt-5.5"


def test_an_unknown_model_is_kept_because_failing_loud_beats_running_wrong() -> None:
    """Fail toward LEAVING THE PLAN ALONE.

    A wrongly-kept model fails loudly at dispatch, exactly as the incident did. A wrongly-
    replaced one silently runs a different model and reports a result, which is worse: the
    artifact would claim a substrate that never ran.
    """
    assert coerced_model("gemini", "codex", "mystery-7b", DEFAULT) == "mystery-7b"


def test_a_future_model_in_the_same_family_is_not_read_as_foreign() -> None:
    """Prefix matching, not a fixed table -- a table drifts narrower than its concept, which is
    the failure mode this repository keeps rediscovering."""
    assert coerced_model("gemini", "codex", "gpt-6-turbo", DEFAULT) == "gpt-6-turbo"
    assert coerced_model("gemini", "claude", "claude-6-opus", DEFAULT) == "claude-6-opus"


def test_the_call_site_uses_it(monkeypatch) -> None:
    """The rule must be WIRED, not merely implemented beside the dispatch.

    Three times this session a feature's call site was replaced with a constant and the suite
    stayed green. This asserts the source actually threads the helper into model_override.
    """
    src = (REPO / "scripts" / "research_conductor.py").read_text()
    assert "model_override=coerced_model(" in src
    assert "_planned_agent_type" in src
