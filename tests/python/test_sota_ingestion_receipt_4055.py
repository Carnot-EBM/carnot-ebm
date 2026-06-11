"""Tests for REQ-REPORT-4055 / SCENARIO-REPORT-4055."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import sota_ingestion_4055 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
NOTE_PATH = Path(
    "docs/research-notes/"
    "sota-ingestion-2026-06-11-unsaturated-execverif-and-verifier-pruner.md"
)
RECEIPT_PATH = Path("results/experiment_4055_sota_ingestion_receipt.json")
STUDYING_PATH = Path("research-studying.md")


def _valid_citations() -> list[dict[str, str]]:
    return [
        {"arxiv_id_or_url": "https://arxiv.org/abs/2305.01210"},
        {"arxiv_id_or_url": "https://arxiv.org/abs/2403.07974"},
        {"arxiv_id_or_url": "https://arxiv.org/abs/2507.06920"},
        {"arxiv_id_or_url": "https://arxiv.org/abs/2604.21598"},
        {"arxiv_id_or_url": "https://arxiv.org/abs/2602.01070"},
        {"arxiv_id_or_url": "https://arxiv.org/abs/2505.16312"},
    ]


def _valid_receipt() -> dict[str, object]:
    return mod.build_receipt(
        methods_mapped_count=6,
        citations=_valid_citations(),
        flagged_for_v376=[
            "evalplus_hidden_rescore_fixed_pool",
            "gap4_online_pruner_for_explore_first_arc",
        ],
    )


def test_req_report_4055_spec_anchor_exists() -> None:
    """REQ-REPORT-4055: OpenSpec declares the ingestion contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4055" in spec
    assert "SCENARIO-REPORT-4055" in spec
    assert NOTE_PATH.as_posix() in spec
    assert RECEIPT_PATH.as_posix() in spec
    assert "flagged_for_v376" in spec


def test_build_receipt_has_required_schema_fields_for_req_report_4055() -> None:
    """REQ-REPORT-4055: Exp 4055 emits the exact receipt schema."""

    receipt = _valid_receipt()

    assert receipt == {
        "honest_verdict": (
            "complete: "
            "sota_ingestion_unsaturated_execverif_and_pruner_mapped"
        ),
        "methods_mapped_count": 6,
        "citations": _valid_citations(),
        "flagged_for_v376": [
            "evalplus_hidden_rescore_fixed_pool",
            "gap4_online_pruner_for_explore_first_arc",
        ],
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }


@pytest.mark.parametrize(
    ("bad_receipt", "message"),
    [
        (_valid_receipt() | {"honest_verdict": "draft"}, "terminal prefix"),
        (_valid_receipt() | {"methods_mapped_count": 0}, "at least six"),
        (_valid_receipt() | {"methods_mapped_count": 5}, "at least six"),
        (_valid_receipt() | {"citations": []}, "non-empty list"),
        (
            _valid_receipt() | {"citations": _valid_citations()[:5]},
            "at least methods_mapped_count",
        ),
        (
            _valid_receipt() | {"citations": [{"title": "missing source"}] * 6},
            "arxiv_id_or_url",
        ),
        (_valid_receipt() | {"flagged_for_v376": []}, "flagged_for_v376"),
        (
            _valid_receipt() | {"inference_substrate": "manual_guess"},
            "aggregation_from_upstream_artifacts",
        ),
    ],
)
def test_validate_receipt_rejects_schema_violations_for_scenario_report_4055(
    bad_receipt: dict[str, object], message: str
) -> None:
    """SCENARIO-REPORT-4055: invalid receipts fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_receipt(bad_receipt)


def test_validate_receipt_rejects_missing_and_extra_fields() -> None:
    """SCENARIO-REPORT-4055: receipt fields are bare and exact."""

    missing_receipt = _valid_receipt()
    missing_receipt.pop("citations")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_receipt(missing_receipt)

    extra_receipt = _valid_receipt()
    extra_receipt["uncited_method"] = "not allowed"
    with pytest.raises(ValueError, match="unexpected fields"):
        mod.validate_receipt(extra_receipt)


def test_validate_markdown_note_checks_scenario_report_4055_sections() -> None:
    """SCENARIO-REPORT-4055: paired note keeps both track mappings."""

    note = """
    # SOTA ingestion

    ## UN-SATURATED execution-verification corpus
    Implementation over Carnot stack: demo-fit code verifier, sandbox, EvalPlus.
    Pitfalls / where it fails: hidden tests can still saturate.

    ## VERIFIER-GUIDED online action-pruning
    Implementation over Carnot stack: explore-first solver and GAP-4 verifier.
    Pitfalls / where it fails: verifier false negatives can prune the solution.

    ## Bottom line for the .376 roadmap
    Flag EvalPlus rescore and GAP-4 online pruner.
    """

    mod.validate_markdown_note(note)


def test_validate_markdown_note_rejects_missing_bottom_line() -> None:
    """SCENARIO-REPORT-4055: note must close the discover-to-plan loop."""

    with pytest.raises(ValueError, match="Bottom line"):
        mod.validate_markdown_note(
            "## UN-SATURATED execution-verification corpus\n"
            "## VERIFIER-GUIDED online action-pruning\n"
            "Implementation over Carnot stack\n"
            "Pitfalls / where it fails\n"
        )


def test_deliverable_files_validate_against_req_report_4055() -> None:
    """REQ-REPORT-4055: committed note and JSON receipt satisfy the contract."""

    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    note = NOTE_PATH.read_text(encoding="utf-8")
    studying = STUDYING_PATH.read_text(encoding="utf-8")

    mod.validate_receipt(receipt)
    mod.validate_markdown_note(note)
    assert receipt["methods_mapped_count"] >= 6
    assert "2026-06-11 Exp 4055 - .375 SOTA ingestion ingested" in studying
    assert "Bottom line for the .376 roadmap" in studying
