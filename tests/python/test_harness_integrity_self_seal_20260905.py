"""Spec: REQ-HARNESS-INTEGRITY-1, SCENARIO-HARNESS-INTEGRITY-1-B

Unsealing the integrity lint against its own edit is permitted and announced, never silent.

QA-LAYER FINDING 2026-09-04, oldest of the open set at seven days, confirmed 2026-09-05. A path
named in any active declaration's `unsealed` list is unsealed for that commit. That bypass is
deliberate for ordinary harness files: a glob never unseals, so it costs a typed path and leaves
intent in the record. Naming THIS file, though, permits an uncommitted edit to the check that
decides whether edits may land.

The first fix refused the self-unseal outright. That was wrong, and the guard proved it by
refusing the commit that carried the fix: this file is sealed by the standing declaration, so a
refusal makes it uneditable -- the commit that would move the HEAD baseline is the one being
refused, and the guard's own refusal text advertises an `--unseal` that would no longer work.

The finding is SILENT non-firing. The cure for silence is noise, not a wall. The self-unseal
proceeds and every run that honours it says so.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import harness_integrity_lint as lint  # noqa: E402

SELF = "scripts/harness_integrity_lint.py"


def test_a_declaration_naming_the_lint_still_unseals_it() -> None:
    """Permitted: refusing it made the file unrepairable, which the guard demonstrated."""
    unsealed, _ = lint.effective_unsealed([{"unsealed": [SELF]}])
    assert SELF in unsealed


def test_a_self_unseal_is_reported_back_to_the_caller() -> None:
    """The missing fact. Without this the disarm happened and nothing said so."""
    _, self_unsealed = lint.effective_unsealed([{"unsealed": [SELF, "scripts/other.py"]}])
    assert self_unsealed == {SELF}


def test_an_ordinary_harness_unseal_is_not_announced() -> None:
    """Announcing every unseal would train a reader to skip the line that matters."""
    unsealed, self_unsealed = lint.effective_unsealed([{"unsealed": ["scripts/other.py"]}])
    assert unsealed == {"scripts/other.py"}
    assert self_unsealed == set()


def test_declarations_are_unioned_before_the_check() -> None:
    records = [{"unsealed": ["scripts/a.py"]}, {"unsealed": [SELF]}, None]
    unsealed, self_unsealed = lint.effective_unsealed(records)
    assert unsealed == {"scripts/a.py", SELF}
    assert self_unsealed == {SELF}


def test_no_declarations_unseal_nothing_and_announce_nothing() -> None:
    assert lint.effective_unsealed([None, {}]) == (set(), set())


def test_the_announced_set_is_deliberately_narrow() -> None:
    """A wider set needs its own written reason; drift into it should fail this test."""
    assert lint.SELF_UNSEAL_IS_LOUD == frozenset({SELF})


def test_the_announced_path_exists() -> None:
    """An entry naming a moved or misspelled file announces nothing."""
    assert (REPO / SELF).is_file()
