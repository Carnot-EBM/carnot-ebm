"""Tests for REQ-PHASE4-035 / SCENARIO-PHASE4-035."""

import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "carnot"
    / "sota_ingestion_4030.py"
)
_SPEC = importlib.util.spec_from_file_location("sota_ingestion_4030", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

build_receipt = _MODULE.build_receipt
validate_markdown_note = _MODULE.validate_markdown_note
validate_receipt = _MODULE.validate_receipt


def test_build_receipt_has_required_schema_fields_for_req_phase4_035() -> None:
    """REQ-PHASE4-035: Exp 4030 emits the machine-checkable receipt schema."""
    receipt = build_receipt(
        methods_mapped_count=2,
        citations=[
            {"arxiv_id_or_url": "https://arxiv.org/abs/2408.13745"},
            {"arxiv_id_or_url": "https://arxiv.org/abs/2604.03208"},
        ],
        flagged_for_v374=["off_arc_demo_fit_vs_aces", "vc33_subgoal_search"],
    )

    assert set(receipt) == {
        "honest_verdict",
        "methods_mapped_count",
        "citations",
        "flagged_for_v374",
        "inference_substrate",
    }
    assert str(receipt["honest_verdict"]).startswith("complete:")
    assert receipt["methods_mapped_count"] == 2
    assert receipt["inference_substrate"] == "aggregation_from_upstream_artifacts"


@pytest.mark.parametrize(
    ("bad_receipt", "message"),
    [
        (
            {
                "honest_verdict": "draft",
                "methods_mapped_count": 1,
                "citations": [{"arxiv_id_or_url": "https://arxiv.org/abs/2408.13745"}],
                "flagged_for_v374": ["off_arc"],
                "inference_substrate": "aggregation_from_upstream_artifacts",
            },
            "terminal prefix",
        ),
        (
            {
                "honest_verdict": "complete: mapped",
                "methods_mapped_count": 0,
                "citations": [{"arxiv_id_or_url": "https://arxiv.org/abs/2408.13745"}],
                "flagged_for_v374": ["off_arc"],
                "inference_substrate": "aggregation_from_upstream_artifacts",
            },
            "positive integer",
        ),
        (
            {
                "honest_verdict": "complete: mapped",
                "methods_mapped_count": 1,
                "citations": [],
                "flagged_for_v374": ["off_arc"],
                "inference_substrate": "aggregation_from_upstream_artifacts",
            },
            "non-empty list",
        ),
        (
            {
                "honest_verdict": "complete: mapped",
                "methods_mapped_count": 1,
                "citations": [{"title": "missing source"}],
                "flagged_for_v374": ["off_arc"],
                "inference_substrate": "aggregation_from_upstream_artifacts",
            },
            "arxiv_id_or_url",
        ),
        (
            {
                "honest_verdict": "complete: mapped",
                "methods_mapped_count": 1,
                "citations": [{"arxiv_id_or_url": "https://arxiv.org/abs/2408.13745"}],
                "flagged_for_v374": [],
                "inference_substrate": "aggregation_from_upstream_artifacts",
            },
            "flagged_for_v374",
        ),
        (
            {
                "honest_verdict": "complete: mapped",
                "methods_mapped_count": 1,
                "citations": [{"arxiv_id_or_url": "https://arxiv.org/abs/2408.13745"}],
                "flagged_for_v374": ["off_arc"],
                "inference_substrate": "manual_guess",
            },
            "aggregation_from_upstream_artifacts",
        ),
    ],
)
def test_validate_receipt_rejects_schema_violations_for_scenario_phase4_035(
    bad_receipt: dict[str, object], message: str
) -> None:
    """SCENARIO-PHASE4-035: invalid receipts fail closed with actionable errors."""
    with pytest.raises(ValueError, match=message):
        validate_receipt(bad_receipt)


def test_validate_receipt_rejects_missing_and_extra_fields() -> None:
    """SCENARIO-PHASE4-035: receipt fields are bare and exact."""
    valid_receipt = build_receipt(
        methods_mapped_count=1,
        citations=[{"arxiv_id_or_url": "https://arxiv.org/abs/2604.03208"}],
        flagged_for_v374=["vc33_subgoal_search"],
    )

    missing_receipt = dict(valid_receipt)
    missing_receipt.pop("citations")
    with pytest.raises(ValueError, match="missing required fields"):
        validate_receipt(missing_receipt)

    extra_receipt = dict(valid_receipt)
    extra_receipt["uncited_method"] = "not allowed"
    with pytest.raises(ValueError, match="unexpected fields"):
        validate_receipt(extra_receipt)


def test_validate_markdown_note_checks_scenario_phase4_035_sections() -> None:
    """SCENARIO-PHASE4-035: paired note keeps track/action/failure sections."""
    note = """
    # SOTA ingestion

    ## OFF-ARC execution-consistency verifier transfer
    Implementation over Carnot stack: reuse sandbox.py and arc_gap4_execution_verifier.py.
    Pitfalls / where it fails: visible tests may be ceiling-saturated.

    ## Hierarchical/subgoal search over verified world model
    Implementation over Carnot stack: reuse verified-WM simulator and exp4020 is_goal.
    Pitfalls / where it fails: subgoals can be invalid or overfit.

    ## Bottom line for the .374 roadmap
    Flag off-ARC demo-fit vs ACES and vc33 subgoal search.
    """

    validate_markdown_note(note)


def test_validate_markdown_note_rejects_missing_bottom_line() -> None:
    """SCENARIO-PHASE4-035: note must close the discover-to-plan loop."""
    with pytest.raises(ValueError, match="Bottom line"):
        validate_markdown_note(
            "## OFF-ARC execution-consistency verifier transfer\n"
            "## Hierarchical/subgoal search over verified world model\n"
            "Implementation over Carnot stack\n"
            "Pitfalls / where it fails\n"
        )
