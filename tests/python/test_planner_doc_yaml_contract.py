"""Spec: REQ-RESEARCH-6890, SCENARIO-RESEARCH-6890-A

The planner prompt tells the agent its two output files must describe the same task list.

INCIDENT 2026-09-02, twice in one day. One planner run writes BOTH
`openspec/change-proposals/research-roadmap-vNEXT.md` and `research-roadmap-next.yaml`. It wrote a
document declaring "13 tasks, exp6885 through exp6897" alongside a YAML holding 4, and the day
before, a document claiming 11 tasks (exp6874-exp6884) alongside a YAML holding 4.

Each milestone's own contract experiment caught the mismatch and failed its gate — exp6874 on
`v602_document_yaml_parity`, exp6885 on `document_yaml_task_contract`. In .602 that root failure
cascade-blocked exp6875, exp6876 and exp6877, so 3 of the milestone's 4 real tasks never ran.

This is a PROMPT fix, not a mechanical gate, and that was deliberate. An activation guard that
refuses on mismatch would deadlock the loop if the planner reproduced the mismatch on replan, and
the conductor allows only 2 replans. The existing per-task contract check already detects it; the
prompt aims to stop it being produced.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONDUCTOR = REPO / "scripts" / "research_conductor.py"


def _planner_prompt_text() -> str:
    """The planner prompt is built with f-string fragments; read the source region.

    Reading the SOURCE rather than calling the builder keeps this test free of the conductor's
    import side effects, which pull in torch and jax.
    """
    return CONDUCTOR.read_text(encoding="utf-8")


def test_the_prompt_requires_the_two_files_to_agree() -> None:
    src = _planner_prompt_text()
    assert "THE TWO FILES MUST AGREE ON THE TASK LIST" in src


def test_the_prompt_names_the_failure_it_prevents() -> None:
    """A rule without its incident gets edited away by someone who does not know the cost."""
    src = _planner_prompt_text()
    assert "2026.09.602" in src
    assert "cascade-blocked 3" in src


def test_the_prompt_tells_the_agent_what_to_do_when_it_trims() -> None:
    """The failure mode is deciding mid-plan to emit fewer tasks and leaving the document."""
    src = _planner_prompt_text()
    assert "EDIT FILE 1 to match" in src


def test_the_instruction_sits_before_the_optional_fields_section() -> None:
    """Ordering matters: it must land while the agent is still reading about the two files.

    Placed after OPTIONAL YAML FIELDS it would be separated from FILE 1 / FILE 2 by ~40 lines of
    unrelated field documentation.
    """
    src = _planner_prompt_text()
    two_files = src.index("THE TWO FILES MUST AGREE ON THE TASK LIST")
    optional = src.index("OPTIONAL YAML FIELDS")
    file2 = src.index("FILE 2: research-roadmap-next.yaml")
    assert file2 < two_files < optional


def test_the_prompt_still_creates_both_files() -> None:
    """Guard against an edit that drops one of the two outputs while rewording this section."""
    src = _planner_prompt_text()
    assert "FILE 1: openspec/change-proposals/research-roadmap-vNEXT.md" in src
    assert "FILE 2: research-roadmap-next.yaml" in src


def test_no_stray_unescaped_brace_was_introduced() -> None:
    """The prompt is f-string fragments; a bare { would raise at format time, not at import.

    The incident text contains `exp6885-exp6897`, not a brace, but a future edit adding a literal
    example dict would break the planner at runtime rather than in review.
    """
    src = _planner_prompt_text()
    start = src.index("THE TWO FILES MUST AGREE ON THE TASK LIST")
    end = src.index("OPTIONAL YAML FIELDS", start)
    section = src[start:end]
    # every { or } in an f-string fragment must be doubled
    for m in re.finditer(r"(?<!\{)\{(?!\{)|(?<!\})\}(?!\})", section):
        raise AssertionError(f"unescaped brace at offset {m.start()} in the new prompt section")
