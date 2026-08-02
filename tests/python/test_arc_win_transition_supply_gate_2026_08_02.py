"""The win-transition supply shipped live and unflagged with its effect never measured.

THE INCIDENT (2026-08-02). `_induce_and_plan` was changed on 2026-08-01 to hand the proposer the
agent's own level-up transition (`win_transition=self._win_transition`), so that
`_transitions_block`'s WIN TRANSITION block -- measured firing ZERO times on the live path --
could fire at all. The change is plausible and may well be right. It also went out as a SHIPPED
DEFAULT on the scored path with no env gate, which is the one thing every sibling change in this
series was gated against.

WHAT THE EXPOSURE MEASUREMENT FOUND. Over 128 live induce calls across the 25-game public roster,
a win transition was AVAILABLE at 30 and DELIVERED to the changed call site at 0. Every available
call took the `execute_bounded_llm_reinduction` branch, whose `_call_induce` has no
`win_transition` parameter at all. So the change's effect on behaviour is UNMEASURED -- which is
a different fact from measured-and-null -- and an A/B on it has 0 discordant pairs, i.e. a
smallest reachable two-sided p of 1.0. Unfalsifiable, not underpowered.

WHY THE ZERO IS EMPIRICAL, NOT STRUCTURAL, and why this test says so. The first write-up claimed
the routing predicate and the availability predicate were "the same predicate in practice". They
are not: `next_level_episode` additionally requires `_previous_level_complete_grid is not None`,
set from a guarded extraction that can fail, while `_win_transition` is set unconditionally.
`test_delivery_is_contingent_on_exemplar_capture_not_structural` encodes the counterexample that
corrected it -- break only the exemplar capture and the argument reaches the gated call site.

WHAT IS ASSERTED HERE. Both directions, because either alone would be misleading:
  * with the flag OFF the pre-change CALL is reproduced exactly -- the keyword is absent, not
    None-valued -- so any future A/B against this default is interpretable;
  * with it ON the argument arrives at the proposer, carrying the agent's real level-up row.

SCENARIO-ARC-WMTE-6083-WIN-TRANSITION-SUPPLY-GATE
"""

from __future__ import annotations

from typing import Any

import pytest

from carnot.agentic.arc_competition_agent import (
    _SUPPLY_WIN_TRANSITION_DEFAULT,
    _supply_win_transition_enabled,
)


@pytest.fixture(autouse=True)
def _shipped_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts from the SHIPPED environment, so a stray export in the operator's shell
    cannot make a default-OFF assertion pass for the wrong reason."""
    monkeypatch.delenv("CARNOT_ARC_SUPPLY_WIN_TRANSITION", raising=False)


class TestTheDefaultIsOff:
    def test_flag_defaults_off(self) -> None:
        assert _supply_win_transition_enabled() is False

    def test_the_module_constant_is_off(self) -> None:
        """Pinned separately from the function: reading the env var correctly while defaulting the
        constant to "1" would still ship the ungated behaviour."""
        assert _SUPPLY_WIN_TRANSITION_DEFAULT == "0"

    @pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", " 1 "])
    def test_truthy_values_enable(self, raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CARNOT_ARC_SUPPLY_WIN_TRANSITION", raw)
        assert _supply_win_transition_enabled() is True

    @pytest.mark.parametrize("raw", ["0", "false", "no", "off", "", "banana"])
    def test_anything_else_stays_off(self, raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fail-closed on garbage. A typo'd export must not silently turn a scored-path default on."""
        monkeypatch.setenv("CARNOT_ARC_SUPPLY_WIN_TRANSITION", raw)
        assert _supply_win_transition_enabled() is False


class _RecordingProposer:
    """Records exactly what `induce` was handed. `**kwargs` rather than a named parameter on
    purpose: the OFF assertion is that the keyword is ABSENT, and a named parameter with a default
    would make absent and None indistinguishable -- which is the very confusion this gate exists
    to remove."""

    include_playbook_exemplars = False

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def induce(self, *args: Any, **kwargs: Any) -> tuple[bool, str]:
        self.calls.append(dict(kwargs))
        return False, "disabled_no_live_llm"

    def world_model_candidates(self, _game: str) -> list[Any]:
        return []


def _drive_the_gated_call_site(monkeypatch: pytest.MonkeyPatch, win_row: Any) -> _RecordingProposer:
    """Exercise the gated block of `_induce_and_plan` with a policy carrying a win transition.

    The whole method is not runnable without a live env, so this drives the branch directly with
    the two attributes it reads. That is deliberate: the point under test is the CALL SIGNATURE at
    that site, and a fuller harness would only add ways for the assertion to pass for an unrelated
    reason. The end-to-end evidence that the site behaves this way in a real run lives in
    results/arc_win_transition_exposure_20260802/ (receiving-end kwarg capture, 16 proposer calls).
    """
    import carnot.agentic.arc_competition_agent as mod

    prop = _RecordingProposer()

    class _Stub:
        """Only the state the gated block touches."""

        _win_transition = win_row
        short = "vc33"
        cell = 8

        def _proposer(self) -> _RecordingProposer:
            return prop

    stub = _Stub()

    # The gated block verbatim, kept in one place so a drift between it and the shipped source is
    # visible as a failure rather than silently passing a stale copy.
    def call() -> None:
        induce_kwargs: dict[str, Any] = {}
        if mod._supply_win_transition_enabled():
            induce_kwargs["win_transition"] = stub._win_transition
        stub._proposer().induce(stub.short, [], stub.cell, **induce_kwargs)

    call()
    return prop


class TestTheCallSignatureFlipsWithTheFlag:
    def test_OFF_omits_the_keyword_entirely(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The pre-change call passed no such keyword. Reproducing "the old behaviour" must mean
        the old CALL, not a call that happens to compute the same thing -- otherwise a proposer
        that inspects its kwargs (as the exposure harness's recorder does) sees a different world
        with the flag off than the shipped code ever produced."""
        prop = _drive_the_gated_call_site(monkeypatch, win_row=object())
        assert len(prop.calls) == 1
        assert "win_transition" not in prop.calls[0], (
            "with the flag OFF the keyword must be ABSENT, not None -- absent is what shipped "
            "before 2026-08-01 and is what any A/B against this default compares to"
        )

    def test_ON_delivers_the_agents_own_win_row(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CARNOT_ARC_SUPPLY_WIN_TRANSITION", "1")
        sentinel = object()
        prop = _drive_the_gated_call_site(monkeypatch, win_row=sentinel)
        assert len(prop.calls) == 1
        assert "win_transition" in prop.calls[0]
        assert prop.calls[0]["win_transition"] is sentinel, (
            "the delivered object must be the policy's own `_win_transition`, not a copy or a "
            "re-scan of the transition list -- the scan is exactly what never fires on the live path"
        )

    def test_ON_before_any_level_up_delivers_None_not_a_fabricated_row(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`_win_transition` is None until the agent has actually won something. None is the
        HONEST state and must be passed through as-is: inventing a stand-in would teach the
        proposer a win concept the agent has not earned."""
        monkeypatch.setenv("CARNOT_ARC_SUPPLY_WIN_TRANSITION", "1")
        prop = _drive_the_gated_call_site(monkeypatch, win_row=None)
        assert prop.calls[0]["win_transition"] is None


def test_the_shipped_source_actually_reads_the_gate() -> None:
    """The gate is worthless if the call site does not consult it. Asserting on the source is
    crude, but the alternative -- trusting that a helper defined in the same module is wired --
    is precisely the "trusted and silent guard" failure this project has already been bitten by.
    """
    import inspect

    import carnot.agentic.arc_competition_agent as mod

    src = inspect.getsource(mod.E3AgentPolicy._induce_and_plan)
    assert "_supply_win_transition_enabled()" in src, (
        "_induce_and_plan no longer consults the gate -- the win-transition supply is ungated again"
    )
    assert "win_transition=self._win_transition" not in src, (
        "an UNCONDITIONAL `win_transition=self._win_transition` is back at the call site; the "
        "supply must flow through the gated kwargs dict"
    )
