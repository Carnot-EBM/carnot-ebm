"""Tests for REQ-PHASE4-039 / SCENARIO-PHASE4-039."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import sota_ingestion_4043 as mod


SPEC_PATH = Path("openspec/capabilities/phase4_active_inference/spec.md")
NOTE_PATH = Path(
    "docs/research-notes/"
    "sota-ingestion-2026-06-11-offarc-power-and-closed-loop-planning.md"
)
RECEIPT_PATH = Path("results/experiment_4043_sota_ingestion_receipt.json")


def _valid_receipt() -> dict[str, object]:
    return mod.build_receipt(
        methods_mapped_count=4,
        citations=[
            {"arxiv_id_or_url": "https://arxiv.org/abs/2604.06485"},
            {"arxiv_id_or_url": "https://arxiv.org/abs/2602.04254"},
            {"arxiv_id_or_url": "https://arxiv.org/abs/2306.00840"},
        ],
        flagged_for_v375=[
            "offarc_power_plus_sep_aces_agentic_verifier",
            "closed_loop_replan_with_wm_trust_gate",
        ],
    )


def test_req_phase4_039_spec_anchor_exists() -> None:
    """REQ-PHASE4-039: OpenSpec declares the Exp 4043 ingestion contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-039" in spec
    assert "SCENARIO-PHASE4-039" in spec
    assert NOTE_PATH.as_posix() in spec
    assert RECEIPT_PATH.as_posix() in spec
    assert "flagged_for_v375" in spec


def test_build_receipt_has_required_schema_fields_for_req_phase4_039() -> None:
    """REQ-PHASE4-039: Exp 4043 emits the machine-checkable receipt schema."""

    receipt = _valid_receipt()

    assert receipt == {
        "honest_verdict": (
            "complete: sota_ingestion_offarc_power_and_closed_loop_mapped"
        ),
        "methods_mapped_count": 4,
        "citations": [
            {"arxiv_id_or_url": "https://arxiv.org/abs/2604.06485"},
            {"arxiv_id_or_url": "https://arxiv.org/abs/2602.04254"},
            {"arxiv_id_or_url": "https://arxiv.org/abs/2306.00840"},
        ],
        "flagged_for_v375": [
            "offarc_power_plus_sep_aces_agentic_verifier",
            "closed_loop_replan_with_wm_trust_gate",
        ],
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }


@pytest.mark.parametrize(
    ("bad_receipt", "message"),
    [
        (_valid_receipt() | {"honest_verdict": "draft"}, "terminal prefix"),
        (_valid_receipt() | {"methods_mapped_count": 0}, "at least three"),
        (_valid_receipt() | {"methods_mapped_count": 2}, "at least three"),
        (_valid_receipt() | {"citations": []}, "non-empty list"),
        (
            _valid_receipt() | {"citations": [{"title": "missing source"}]},
            "arxiv_id_or_url",
        ),
        (_valid_receipt() | {"flagged_for_v375": []}, "flagged_for_v375"),
        (
            _valid_receipt() | {"inference_substrate": "manual_guess"},
            "aggregation_from_upstream_artifacts",
        ),
    ],
)
def test_validate_receipt_rejects_schema_violations_for_scenario_phase4_039(
    bad_receipt: dict[str, object], message: str
) -> None:
    """SCENARIO-PHASE4-039: invalid receipts fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_receipt(bad_receipt)


def test_validate_receipt_rejects_missing_and_extra_fields() -> None:
    """SCENARIO-PHASE4-039: receipt fields are bare and exact."""

    missing_receipt = _valid_receipt()
    missing_receipt.pop("citations")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_receipt(missing_receipt)

    extra_receipt = _valid_receipt()
    extra_receipt["uncited_method"] = "not allowed"
    with pytest.raises(ValueError, match="unexpected fields"):
        mod.validate_receipt(extra_receipt)


def test_validate_markdown_note_checks_scenario_phase4_039_sections() -> None:
    """SCENARIO-PHASE4-039: paired note keeps track/action/failure sections."""

    note = """
    # SOTA ingestion

    ## OFF-ARC power + stronger discriminator
    Implementation over Carnot stack: reuse sandbox.py and demo-fit verifier.
    Pitfalls / where it fails: visible tests may be ceiling-saturated.

    ## CLOSED-LOOP planning over a verified world model
    Implementation over Carnot stack: reuse vc33 WM and exp4034 is_goal.
    Pitfalls / where it fails: model-error exploitation can dominate.

    ## Bottom line for the .375 roadmap
    Flag power+SEP and closed-loop replanning with trust gate.
    """

    mod.validate_markdown_note(note)


def test_validate_markdown_note_rejects_missing_bottom_line() -> None:
    """SCENARIO-PHASE4-039: note must close the discover-to-plan loop."""

    with pytest.raises(ValueError, match="Bottom line"):
        mod.validate_markdown_note(
            "## OFF-ARC power + stronger discriminator\n"
            "## CLOSED-LOOP planning over a verified world model\n"
            "Implementation over Carnot stack\n"
            "Pitfalls / where it fails\n"
        )


def test_deliverable_files_validate_against_req_phase4_039() -> None:
    """REQ-PHASE4-039: committed note and JSON receipt satisfy the contract."""

    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    note = NOTE_PATH.read_text(encoding="utf-8")

    mod.validate_receipt(receipt)
    mod.validate_markdown_note(note)
    assert receipt["methods_mapped_count"] >= 3
