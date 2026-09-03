"""Spec: REQ-CONDUCTOR-VERDICT-4, SCENARIO-CONDUCTOR-VERDICT-4-A

The planner prompt tells task authors which verdict_class retries.

INCIDENT 2026-09-03. Both the v605 and v608 capstones declared `verdict_class: partial` over
milestones whose upstream tasks had been cascade-blocked. `partial` is the one enum member the
conductor re-runs, so each burned three attempts producing three identical artifacts and was
retired. Two milestones lost their capstone for describing their inputs accurately with the wrong
class -- `blocked`, which is terminal and trusted on first write, was available and correct both
times.

The prompt listed the enum and never said that one member retries. This is a PROMPT fix: the guard
is behaving as designed and must not be changed (see the 2026-09-03 correction in
ops/known-issues.md, which retracts an earlier claim that the guard punished honesty).
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONDUCTOR = REPO / "scripts" / "research_conductor.py"


def _src() -> str:
    """Read the source rather than build the prompt: importing pulls in torch and jax."""
    return CONDUCTOR.read_text(encoding="utf-8")


def test_the_prompt_says_one_member_retries() -> None:
    assert "ONE MEMBER RETRIES" in _src()


def test_the_prompt_forbids_partial_for_external_incompleteness() -> None:
    src = _src()
    assert "do NOT declare `partial` when the incompleteness is EXTERNAL" in src


def test_the_prompt_names_blocked_as_the_alternative() -> None:
    assert "Declare\\n" in _src() or "`blocked` instead" in _src()


def test_the_prompt_carries_the_incident_that_motivated_it() -> None:
    """A rule without its cost gets edited away by someone who does not know the price."""
    src = _src()
    assert "2026-09-03" in src
    assert "capstones declared `partial`" in src


def test_the_guidance_sits_inside_the_verdict_class_section() -> None:
    """Placed elsewhere it is separated from the enum it qualifies."""
    src = _src()
    enum = src.index("A CLOSED enum next to the free-text honest_verdict")
    retries = src.index("ONE MEMBER RETRIES")
    nxt = src.index("solve_provenance: [REQUIRED", enum)
    assert enum < retries < nxt


def test_no_unescaped_brace_was_introduced() -> None:
    """The prompt is f-string fragments; a bare brace raises at format time, not import."""
    import re

    src = _src()
    start = src.index("ONE MEMBER RETRIES")
    end = src.index("solve_provenance: [REQUIRED", start)
    for m in re.finditer(r"(?<!\{)\{(?!\{)|(?<!\})\}(?!\})", src[start:end]):
        raise AssertionError(f"unescaped brace at offset {m.start()}")
