"""Unit tests for the DEV-GATED playbook-exemplar injection into the stall / first-contact
world-model re-induction prompt (REQ-ARC-WMTE-5717).

Covers the mechanism WITHOUT any live GPU: the prompt-builder gate, the proposer opt-in
field + its pass-through to the prompt, and the agent-side flag-OR-env gate. The end-to-end
induction-quality A/B is
python/carnot/experiment_5717_playbook_exemplars_stall_induction_ab.py (live GGUF).

Spec: REQ-ARC-WMTE-5717, SCENARIO-ARC-WMTE-5717.
"""

from __future__ import annotations

import numpy as np
import pytest

from carnot.agentic import arc_competition_agent as agent
from carnot.agentic.arc_executable_world_model import (
    CodexProposer,
    LocalGGUFProposer,
    Transition,
    _PLAYBOOK_EXEMPLAR_BLOCK,
    induce_prompt,
)

_BLOCK_MARKER = "GENERAL EXPLORATION PRINCIPLES"


def _tiny_transitions():
    g0 = np.zeros((3, 3), dtype=int)
    g1 = g0.copy()
    g1[0, 0] = 5
    return [Transition(g0, 1, None, g1, 0, 0)]


def test_induce_prompt_byte_identical_when_off():
    trans = _tiny_transitions()
    off = induce_prompt("gx", trans, 5)
    off_explicit = induce_prompt("gx", trans, 5, include_playbook_exemplars=False)
    assert off == off_explicit
    assert _BLOCK_MARKER not in off
    assert off.startswith("You are inducing")


def test_induce_prompt_prepends_block_when_on():
    trans = _tiny_transitions()
    off = induce_prompt("gx", trans, 5)
    on = induce_prompt("gx", trans, 5, include_playbook_exemplars=True)
    assert on.startswith(_BLOCK_MARKER)
    assert on == _PLAYBOOK_EXEMPLAR_BLOCK + off  # exact prefix, suffix byte-identical


def test_playbook_block_is_game_agnostic():
    # The block must not smuggle a per-game fact (a specific color/coord/mechanic name).
    lowered = _PLAYBOOK_EXEMPLAR_BLOCK.lower()
    for banned in ("bp35", "lp85", "wa30", "color-15", "color 15"):
        assert banned not in lowered


def test_local_gguf_proposer_default_flag_false():
    assert LocalGGUFProposer(repo_substr="x").include_playbook_exemplars is False


def test_codex_proposer_default_flag_false():
    assert CodexProposer().include_playbook_exemplars is False


@pytest.mark.parametrize("flag", [False, True])
def test_proposer_induce_passes_flag_to_prompt(monkeypatch, flag):
    """The proposer's include_playbook_exemplars flag must reach induce_prompt. Capture the
    first generate() prompt without spawning a server."""

    class _CapturedError(Exception):
        pass

    captured: dict[str, str] = {}

    def _fake_generate(prompt, *_a, **_k):
        captured["prompt"] = prompt
        raise _CapturedError()

    prop = LocalGGUFProposer(repo_substr="x", include_playbook_exemplars=flag)
    monkeypatch.setattr(prop, "generate", _fake_generate)
    with pytest.raises(_CapturedError):
        prop.induce("gx", _tiny_transitions(), 5)
    assert (_BLOCK_MARKER in captured["prompt"]) is flag


def test_agent_gate_off_by_default(monkeypatch):
    monkeypatch.setattr(agent, "SUBMITTED_PLAYBOOK_EXEMPLARS_ENABLED", False)
    monkeypatch.delenv("CARNOT_ARC_PLAYBOOK_EXEMPLARS_ENABLED", raising=False)
    assert agent._playbook_exemplars_gate_on() is False


def test_agent_gate_on_via_env(monkeypatch):
    monkeypatch.setattr(agent, "SUBMITTED_PLAYBOOK_EXEMPLARS_ENABLED", False)
    monkeypatch.setenv("CARNOT_ARC_PLAYBOOK_EXEMPLARS_ENABLED", "1")
    assert agent._playbook_exemplars_gate_on() is True


def test_agent_gate_on_via_module_flag(monkeypatch):
    monkeypatch.setattr(agent, "SUBMITTED_PLAYBOOK_EXEMPLARS_ENABLED", True)
    monkeypatch.delenv("CARNOT_ARC_PLAYBOOK_EXEMPLARS_ENABLED", raising=False)
    assert agent._playbook_exemplars_gate_on() is True
