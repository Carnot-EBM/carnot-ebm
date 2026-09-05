"""Spec: REQ-SUBSTRATE-ALIAS-1, SCENARIO-SUBSTRATE-ALIAS-1-B

An alias added to the no-LLM tuple needs evidence however the member is written.

QA-LAYER FINDING 2026-09-04, confirmed 2026-09-05. The guard answered "does this diff add a
quoted `*_no_llm` string". The tuple is not written that way. Measured when this was fixed: of
53 members, 32 are CONSTANT NAMES, one is a starred tuple, and only 20 are bare literals. So the
normal way to add an alias -- define a module constant, then name it in the tuple -- put no
matching literal on any added line and walked straight through.

The named missed input was the staged line `+ LOCAL_SOTA_FIXED_SEQUENCE_REPRESENTATION_SUBSTRATE,`
whose value is `live_local_sota_gguf_fixed_sequence_representation`. That value does not end in
`_no_llm`, so widening the regex alone would not have caught it either. The fix reads the tuple
instead of the diff text.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import substrate_alias_evidence_lint as lint  # noqa: E402

_HEAD = """
A_SUB = "already_present_no_llm"
DETERMINISTIC_VERIFIER_SUBSTRATES = ("det_one", "det two, no LLM")
NO_LLM_SUBSTRATE_ALIASES = (
    A_SUB,
    *DETERMINISTIC_VERIFIER_SUBSTRATES,
    "a_bare_literal_no_llm",
)
"""

_STAGED_BY_NAME = """
A_SUB = "already_present_no_llm"
LOCAL_SOTA_FIXED_SEQUENCE_REPRESENTATION_SUBSTRATE = (
    "live_local_sota_gguf_fixed_sequence_representation"
)
DETERMINISTIC_VERIFIER_SUBSTRATES = ("det_one", "det two, no LLM")
NO_LLM_SUBSTRATE_ALIASES = (
    A_SUB,
    *DETERMINISTIC_VERIFIER_SUBSTRATES,
    "a_bare_literal_no_llm",
    LOCAL_SOTA_FIXED_SEQUENCE_REPRESENTATION_SUBSTRATE,
)
"""


def test_an_alias_added_by_constant_name_is_caught() -> None:
    """The incident. No `_no_llm` literal appears on the added line at all."""
    assert lint.new_aliases_structural(_HEAD, _STAGED_BY_NAME) == [
        "live_local_sota_gguf_fixed_sequence_representation"
    ]


def test_the_literal_scan_alone_would_have_missed_it() -> None:
    """States plainly why the structural read had to be added, not just widened."""
    diff = "+    LOCAL_SOTA_FIXED_SEQUENCE_REPRESENTATION_SUBSTRATE,"
    assert lint.new_aliases(diff, _HEAD) == []


def test_an_alias_added_as_a_bare_literal_is_still_caught() -> None:
    staged = _HEAD.replace(
        '    "a_bare_literal_no_llm",',
        '    "a_bare_literal_no_llm",\n    "brand_new_literal_no_llm",',
    )
    assert lint.new_aliases_structural(_HEAD, staged) == ["brand_new_literal_no_llm"]


def test_an_alias_added_inside_the_starred_tuple_is_caught() -> None:
    """A member can be added one level down without touching the alias tuple at all."""
    staged = _HEAD.replace(
        'DETERMINISTIC_VERIFIER_SUBSTRATES = ("det_one", "det two, no LLM")',
        'DETERMINISTIC_VERIFIER_SUBSTRATES = ("det_one", "det two, no LLM", "det_three")',
    )
    assert lint.new_aliases_structural(_HEAD, staged) == ["det_three"]


def test_an_unchanged_tuple_reports_nothing() -> None:
    assert lint.new_aliases_structural(_HEAD, _HEAD) == []


def test_an_unparseable_staged_file_reports_nothing_rather_than_everything() -> None:
    """A syntax error mid-edit must not spray a refusal naming every existing alias."""
    assert lint.new_aliases_structural(_HEAD, "def broken( :\n") == []


def test_the_real_tuple_resolves_every_member_shape() -> None:
    """Against the live file, not a fixture: names, starred tuple and literals all resolve."""
    members = lint.resolve_alias_members((REPO / "scripts" / "adversarial_verify.py").read_text())
    assert len(members) > 50
    assert "deterministic_verifier" in members
