"""Tests for REQ-REPORT-4067 / SCENARIO-REPORT-4067."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import sota_ingestion_4067 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
NOTE_PATH = Path(
    "docs/research-notes/"
    "sota-ingestion-2026-06-11-v376-unsaturated-corpora-and-online-pruning.md"
)
RECEIPT_PATH = Path("results/experiment_4067_sota_ingestion_receipt.json")
STUDYING_PATH = Path("research-studying.md")


def _valid_citations() -> list[dict[str, str]]:
    return [
        {"arxiv_id_or_url": "https://arxiv.org/abs/2305.01210"},
        {"arxiv_id_or_url": "https://arxiv.org/abs/2403.07974"},
        {"arxiv_id_or_url": "https://arxiv.org/abs/2604.06485"},
        {"arxiv_id_or_url": "https://arxiv.org/abs/2507.06920"},
        {"arxiv_id_or_url": "https://arxiv.org/abs/2603.10282"},
        {"arxiv_id_or_url": "https://arxiv.org/abs/2602.01070"},
    ]


def _valid_receipt() -> dict[str, object]:
    return mod.build_receipt(
        methods_mapped_count=6,
        citations=_valid_citations(),
        flagged_for_v377=[
            "livecodebench_v6_local12b_headroom_route",
            "gap4_soft_prune_replay_for_arc_efficiency",
        ],
    )


def test_req_report_4067_spec_anchor_exists() -> None:
    """REQ-REPORT-4067: OpenSpec declares the ingestion contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4067" in spec
    assert "SCENARIO-REPORT-4067" in spec
    assert NOTE_PATH.as_posix() in spec
    assert RECEIPT_PATH.as_posix() in spec
    assert "flagged_for_v377" in spec


def test_build_receipt_has_required_schema_fields_for_req_report_4067() -> None:
    """REQ-REPORT-4067: Exp 4067 emits the exact receipt schema."""

    receipt = _valid_receipt()

    assert receipt == {
        "honest_verdict": (
            "complete: "
            "sota_ingestion_v376_unsaturated_corpora_and_online_pruning_mapped"
        ),
        "methods_mapped_count": 6,
        "citations": _valid_citations(),
        "flagged_for_v377": [
            "livecodebench_v6_local12b_headroom_route",
            "gap4_soft_prune_replay_for_arc_efficiency",
        ],
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }


@pytest.mark.parametrize(
    ("bad_receipt", "message"),
    [
        (_valid_receipt() | {"honest_verdict": "draft"}, "terminal prefix"),
        (_valid_receipt() | {"methods_mapped_count": 0}, "at least six"),
        (_valid_receipt() | {"methods_mapped_count": 5}, "at least six"),
        (_valid_receipt() | {"methods_mapped_count": True}, "at least six"),
        (_valid_receipt() | {"citations": []}, "non-empty list"),
        (
            _valid_receipt() | {"citations": _valid_citations()[:5]},
            "at least methods_mapped_count",
        ),
        (
            _valid_receipt() | {"citations": [{"title": "missing source"}] * 6},
            "arxiv_id_or_url",
        ),
        (_valid_receipt() | {"flagged_for_v377": []}, "flagged_for_v377"),
        (
            _valid_receipt() | {"inference_substrate": "manual_guess"},
            "aggregation_from_upstream_artifacts",
        ),
    ],
)
def test_validate_receipt_rejects_schema_violations_for_scenario_report_4067(
    bad_receipt: dict[str, object], message: str
) -> None:
    """SCENARIO-REPORT-4067: invalid receipts fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_receipt(bad_receipt)


def test_validate_receipt_rejects_missing_and_extra_fields() -> None:
    """SCENARIO-REPORT-4067: receipt fields are bare and exact."""

    missing_receipt = _valid_receipt()
    missing_receipt.pop("citations")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_receipt(missing_receipt)

    extra_receipt = _valid_receipt()
    extra_receipt["uncited_method"] = "not allowed"
    with pytest.raises(ValueError, match="unexpected fields"):
        mod.validate_receipt(extra_receipt)


def test_validate_markdown_note_checks_scenario_report_4067_sections() -> None:
    """SCENARIO-REPORT-4067: paired note keeps both track mappings."""

    note = """
    # SOTA ingestion

    ## Confirmed .376 actionability from Exp 4055
    evalplus_hidden_rescore_fixed_pool and gap4_online_pruner_for_explore_first_arc.

    ## LOCAL-12B oracle-headroom code corpus
    Implementation over Carnot stack: demo-fit code verifier, sandbox,
    EvalPlus, and LiveCodeBench v6.
    Pitfalls / where it fails: local-12B oracle headroom can vanish.

    ## VERIFIER-GUIDED ONLINE ACTION-PRUNING
    Implementation over Carnot stack: explore-first solver and GAP-4 verifier.
    Pitfalls / where it fails: false negatives can prune a solution path.

    ## Bottom line for the .377 roadmap
    Flag LiveCodeBench headroom route and GAP-4 soft-prune replay.
    """

    mod.validate_markdown_note(note)


def test_validate_markdown_note_rejects_missing_bottom_line() -> None:
    """SCENARIO-REPORT-4067: note must close the discover-to-plan loop."""

    with pytest.raises(ValueError, match="Bottom line"):
        mod.validate_markdown_note(
            "## Confirmed .376 actionability from Exp 4055\n"
            "## LOCAL-12B oracle-headroom code corpus\n"
            "## VERIFIER-GUIDED ONLINE ACTION-PRUNING\n"
            "Implementation over Carnot stack\n"
            "Pitfalls / where it fails\n"
        )


def test_deliverable_files_validate_against_req_report_4067() -> None:
    """REQ-REPORT-4067: committed note and JSON receipt satisfy the contract."""

    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    note = NOTE_PATH.read_text(encoding="utf-8")
    studying = STUDYING_PATH.read_text(encoding="utf-8")

    mod.validate_receipt(receipt)
    mod.validate_markdown_note(note)
    assert receipt["methods_mapped_count"] >= 6
    assert "2026-06-11 Exp 4067 - .376 SOTA ingestion ingested" in studying
    assert "Bottom line for the .377 roadmap" in studying
