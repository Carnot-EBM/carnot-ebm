"""REQ-OPS-DETERMINATION-RESTORE-6260: the restore must catch DOWNGRADE, not just DELETION.

Spec: REQ-OPS-DETERMINATION-RESTORE-6260 /
SCENARIO-OPS-DETERMINATION-RESTORE-6260-DOWNGRADE-TO-NULL-IS-DAMAGE

INCIDENT 2026-08-12. Five artifacts reached the git index with `flagged_adversarial`
changed True -> None, lifting their quarantine. `_restore_dropped_determinations` already
existed and ran before every conductor `git add -A`, and it restored NOTHING, because its
test was `k not in cur` -- False when the key is present and merely nulled. The damage was
caught only because `determination-preservation-lint` refused a HUMAN commit, and that lint
never runs on conductor commits (`--no-verify`, deliberately, for the anti-stash-loss reasons
documented in `git_commit_and_push`).

This is the SILENT_NON_FIRING class named in CLAUDE.md's QA-Layer Authenticity Discipline: a
guard whose pattern is narrower than the concept it claims to protect. The decision rule is
now the pure function `determination_damage`, tested here without a git fixture.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from research_conductor import determination_damage  # noqa: E402


def test_a_deleted_determination_is_damage() -> None:
    d = determination_damage(
        {"flagged_adversarial": True, "corrigendum_note": "why"}, {"experiment": "t"}
    )
    assert d == {"flagged_adversarial": True, "corrigendum_note": "why"}


def test_a_determination_downgraded_to_none_is_damage() -> None:
    """THE 2026-08-12 SHAPE. The key is present and nulled, so the original `k not in cur`
    test was False and the helper restored nothing on the case it was written for."""
    d = determination_damage({"flagged_adversarial": True}, {"flagged_adversarial": None})
    assert d == {"flagged_adversarial": True}


def test_a_determination_downgraded_to_false_is_damage() -> None:
    # False re-admits an artifact to headline aggregation exactly as None does.
    d = determination_damage({"flagged_adversarial": True}, {"flagged_adversarial": False})
    assert d == {"flagged_adversarial": True}


def test_a_deliberate_clear_with_a_cleared_note_is_NOT_damage() -> None:
    """The sanctioned route must survive. determination_preservation_lint documents clearing
    as: set the value falsy AND add a `*_cleared_note`. Restoring over that would make an
    auditable decision impossible to express."""
    d = determination_damage(
        {"flagged_adversarial": True},
        {
            "flagged_adversarial": False,
            "flagged_adversarial_cleared_note": "re-verified: the TAUTOLOGY was structural",
        },
    )
    assert d == {}


def test_a_falsy_value_in_head_is_not_protected() -> None:
    # There is nothing meaningful to preserve, so this must not resurrect a False.
    assert determination_damage({"flagged_adversarial": False}, {}) == {}


def test_non_determination_keys_are_ignored() -> None:
    # Only determination tokens are in scope; a dropped measurement is fail-forward, not damage.
    assert determination_damage({"auroc": 0.9, "duration_s": 3}, {}) == {}


def test_unchanged_artifact_has_no_damage() -> None:
    head = {"flagged_adversarial": True, "corrigendum_note": "why"}
    assert determination_damage(head, dict(head)) == {}


def test_new_measurements_do_not_mask_a_lost_determination() -> None:
    # Fail-forward: a re-run's fresh numbers are fine, but the determination still counts as lost.
    d = determination_damage(
        {"flagged_adversarial": True, "auroc": 0.5},
        {"auroc": 0.91, "new_field": 7},
    )
    assert d == {"flagged_adversarial": True}
