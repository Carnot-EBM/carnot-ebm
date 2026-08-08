"""Pins the fix for live-agent-adversarial-review-2026-08-08.md, Gaps finding 1.

Spec: REQ-ARC-WMTE-6220

WHAT WAS BROKEN. `SUBMITTED_EARLY_STOP_GRACE = 400` (arc_competition_agent.py) was declared
with a comment claiming it was "Enabled 2026-08-07," but nothing read it: `E3AgentPolicy`
never accepted or forwarded an `early_stop_grace` argument, `StepwiseExplorer`'s own default
stayed `None`, and the key was absent from `SUBMITTED_AGENT_CONFIG` -- so the existing parity
test (`test_arc_submitted_agent_parity.py`) could not have caught the drift even though it
already checks several sibling fields the same way. Every scored game therefore ran its
fruitless post-solve tail to the full action budget, not the 400-frame grace the record
claimed.

WHAT THIS FILE PINS.
  1. `E3AgentPolicy`, constructed with no override, forwards `SUBMITTED_EARLY_STOP_GRACE` to
     its `StepwiseExplorer` -- checked two ways: reading the resulting instance attribute,
     and intercepting the constructor call itself so a future refactor cannot satisfy the
     attribute check by some other side channel.
  2. An explicit `early_stop_grace=` override still reaches the explorer. There is no
     `CARNOT_ARC_*` environment-variable convention for this knob anywhere in the module or
     its sweep scripts (checked before writing this test), so this is the actual override
     mechanism in use -- not a substitute for a missing env var, since none was ever implied.
  3. `SUBMITTED_AGENT_CONFIG["early_stop_grace"]` exists and equals 400, so the shipped
     config and the shipped constant cannot silently diverge again.
  4. The REAL scored entrypoint (`make_carnot_agent` -> `CarnotAgent.__init__`) -- not just
     `E3AgentPolicy`'s own default -- passes `SUBMITTED_AGENT_CONFIG["early_stop_grace"]`
     through to `E3AgentPolicy`. `E3AgentPolicy` is replaced with a spy for this one so the
     test never loads a real value head, candidate router, or model.
"""

from __future__ import annotations

from carnot.agentic import arc_competition_agent as m
from carnot.agentic.arc_competition_agent import (
    E3AgentPolicy,
    StepwiseExplorer,
    SUBMITTED_AGENT_CONFIG,
    SUBMITTED_EARLY_STOP_GRACE,
    make_carnot_agent,
)


def test_submitted_early_stop_grace_pinned_in_agent_config():
    """The shipped constant and the shipped config dict must agree -- this is the exact
    single-source-of-truth guarantee the parity test's own docstring promises for every
    other field, and this key was simply missing from it."""
    assert "early_stop_grace" in SUBMITTED_AGENT_CONFIG
    assert SUBMITTED_AGENT_CONFIG["early_stop_grace"] == SUBMITTED_EARLY_STOP_GRACE
    assert SUBMITTED_AGENT_CONFIG["early_stop_grace"] == 400


def test_e3_agent_policy_default_reaches_explorer_instance_attribute():
    """No override supplied: the explorer INSTANCE must carry the shipped grace, not the
    `None` StepwiseExplorer itself still defaults to when nobody threads a value through."""
    pol = E3AgentPolicy("paritytest", proposer=None, value_head=lambda _frame: 0.0)
    assert pol.explorer.early_stop_grace == SUBMITTED_EARLY_STOP_GRACE
    assert pol.explorer.early_stop_grace == 400


def test_e3_agent_policy_forwards_early_stop_grace_via_constructor_kwarg(monkeypatch):
    """Stronger than reading the attribute above: intercepts the StepwiseExplorer
    constructor call itself, so this fails if a future change set the attribute through
    some other path instead of actually threading the kwarg end to end."""
    captured: dict = {}
    real_init = StepwiseExplorer.__init__

    def _spy_init(self, *args, **kwargs):
        captured.update(kwargs)
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(StepwiseExplorer, "__init__", _spy_init)
    E3AgentPolicy("paritytest", proposer=None, value_head=lambda _frame: 0.0)

    assert "early_stop_grace" in captured
    assert captured["early_stop_grace"] == SUBMITTED_EARLY_STOP_GRACE


def test_e3_agent_policy_explicit_kwarg_override_reaches_explorer():
    """The override mechanism for this knob is an explicit constructor kwarg (there is no
    CARNOT_ARC_* env var for it), so this is what a future A/B harness would actually use to
    flip one arm without touching the module-level SUBMITTED_* constant."""
    pol = E3AgentPolicy(
        "paritytest", proposer=None, value_head=lambda _frame: 0.0, early_stop_grace=77
    )
    assert pol.explorer.early_stop_grace == 77


def test_carnot_agent_scored_entrypoint_passes_submitted_grace_to_e3_policy(monkeypatch):
    """The ACTUAL scored construction site (make_carnot_agent -> CarnotAgent.__init__) reads
    SUBMITTED_AGENT_CONFIG explicitly and must forward early_stop_grace to E3AgentPolicy --
    the review named this exact site as where the value needs to reach the explorer, not just
    E3AgentPolicy's own default. E3AgentPolicy is replaced with a spy here (never invoked for
    real) so the test never constructs a value head, candidate router, or model."""

    captured_kwargs: dict = {}

    class _SpyE3Policy:
        def __init__(self, *args, **kwargs):
            captured_kwargs.update(kwargs)

        def is_done(self, *_args, **_kwargs):
            return True

    class _FakeBase:
        def __init__(self, *args, **kwargs) -> None:
            self.game_id = "paritytest"

    monkeypatch.setattr(m, "E3AgentPolicy", _SpyE3Policy)
    cls = make_carnot_agent(_FakeBase)
    cls()  # runs the real CarnotAgent.__init__, which constructs the (spied) policy

    assert "early_stop_grace" in captured_kwargs
    assert captured_kwargs["early_stop_grace"] == SUBMITTED_AGENT_CONFIG["early_stop_grace"]
    assert captured_kwargs["early_stop_grace"] == 400
