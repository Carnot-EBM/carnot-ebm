"""Tests for enum-first verdict_class consumption (conductor + reconciler).

REQ: REQ-CONDUCTOR-VERDICT-2 (openspec/capabilities/research-harnesses/spec.md).
SCENARIOs: SCENARIO-CONDUCTOR-VERDICT-4,
SCENARIO-CONDUCTOR-VERDICT-5,
SCENARIO-CONDUCTOR-VERDICT-6,
SCENARIO-CONDUCTOR-VERDICT-7.

A declared verdict_class inside the closed enum replaces substring
token-list inference in both consumers. The token lists were patched at
least six times; the latest false positive was `disqualified:` — an
honest negative — drawing a critical flag. No test touches tracked
state; everything operates on in-memory dicts.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import in_process_doc_reconcile as reconcile  # noqa: E402
import research_conductor as rc  # noqa: E402


def test_disqualified_declaration_is_trustworthy_terminal() -> None:
    # SCENARIO-CONDUCTOR-VERDICT-4: the exact shape that drew a critical
    # flag this week — an honest negative full of legacy partial tokens.
    payload = {
        "verdict_class": "disqualified",
        "honest_verdict": ("disqualified: eval harness contaminated, no improvement measurable"),
    }
    untrust, verdict = rc._verdict_is_untrustworthy(payload)
    assert untrust is False
    assert verdict is not None
    assert reconcile.classify_artifact(payload) == "⚠️ Research Finding"


def test_partial_declaration_beats_terminal_prefix() -> None:
    # SCENARIO-CONDUCTOR-VERDICT-5: declaration wins in both directions —
    # a partial run cannot launder itself through a `complete:` prefix.
    payload = {
        "verdict_class": "partial",
        "honest_verdict": "complete: ran 3 of 10 corpora before timeout",
    }
    untrust, _ = rc._verdict_is_untrustworthy(payload)
    assert untrust is True


def test_out_of_enum_declaration_falls_through_to_legacy() -> None:
    # SCENARIO-CONDUCTOR-VERDICT-6: an unknown class behaves exactly as
    # if undeclared (the linter already flags it CRITICAL); the terminal
    # prefix then classifies the verdict as trustworthy, legacy-style.
    declared = {
        "verdict_class": "triumphant",
        "honest_verdict": "complete: all corpora processed",
    }
    undeclared = {"honest_verdict": "complete: all corpora processed"}
    assert rc._verdict_is_untrustworthy(declared) == rc._verdict_is_untrustworthy(undeclared)
    assert reconcile.classify_artifact(declared) == reconcile.classify_artifact(undeclared)


def test_circular_positive_never_labels_complete() -> None:
    # SCENARIO-CONDUCTOR-VERDICT-7: the exp6478 gap — an honest circular
    # win must not read as a research win downstream, even when a
    # retro_*_closed upgrade would otherwise promote it.
    payload = {
        "verdict_class": "circular_positive",
        "honest_verdict": "complete_positive: verifier==oracle beats LLM judge",
        "retro_circularity_closed": "RETRO-EXP6478",
    }
    assert reconcile.classify_artifact(payload) == "⚠️ Research Finding"
    # Still trustworthy-terminal for the conductor: no retry churn.
    untrust, _ = rc._verdict_is_untrustworthy(payload)
    assert untrust is False


def test_positive_and_blocked_declarations_map_directly() -> None:
    # REQ-CONDUCTOR-VERDICT-2 rule 2: positive -> Complete, blocked -> Blocked.
    assert (
        reconcile.classify_artifact({"verdict_class": "positive", "honest_verdict": "x"})
        == "✅ Complete"
    )
    assert (
        reconcile.classify_artifact({"verdict_class": "blocked", "honest_verdict": "x"})
        == "⚠️ Blocked"
    )
    for cls in ("positive", "circular_positive", "null", "blocked", "disqualified"):
        untrust, _ = rc._verdict_is_untrustworthy({"verdict_class": cls, "honest_verdict": "x"})
        assert untrust is False, cls


def test_principle_wrapped_declaration_is_unwrapped() -> None:
    # REQ-CONDUCTOR-VERDICT-2: any field may arrive principle-wrapped
    # (QA-layer origin bug #2 shape); the declaration must still bite.
    payload = {
        "verdict_class": {
            "value": "partial",
            "principle": "declared class replaces token inference",
        },
        "honest_verdict": "complete: partial corpus only",
    }
    untrust, _ = rc._verdict_is_untrustworthy(payload)
    assert untrust is True
    label = reconcile.classify_artifact(payload)
    assert label == "⚠️ Research Finding"


def test_consumer_enums_match_the_linter_enum() -> None:
    # REQ-CONDUCTOR-VERDICT-2: the consumers' fallback copies must not
    # drift from adversarial_verify._VERDICT_CLASSES (the source of
    # truth) — the OPERATOR_CURATED_PATHS equality pattern.
    import adversarial_verify as av

    assert rc._VERDICT_CLASSES_FALLBACK == av._VERDICT_CLASSES
    assert reconcile._VERDICT_CLASSES_FALLBACK == av._VERDICT_CLASSES
