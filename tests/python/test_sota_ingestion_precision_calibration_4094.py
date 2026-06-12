"""Tests for REQ-REPORT-4094 / SCENARIO-REPORT-4094."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import sota_ingestion_precision_calibration_4094 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
NOTE_PATH = Path("docs/research-notes/sota-ingestion-precision-calibration-2026-06-12.md")
RECEIPT_PATH = Path("results/experiment_4094_sota_ingestion_precision_calibration_receipt.json")
STUDYING_PATH = Path("research-studying.md")


def _valid_methods() -> list[dict[str, str]]:
    return [
        {
            "arxiv_id": "2411.02272",
            "one_line": "BARC-style augmentation consistency ranks ARC programs before labels become reward data.",
        },
        {
            "arxiv_id": "2603.16140",
            "one_line": "Noisy RLVR says do not train through the 0.32 false-positive channel as if algorithms will absorb it.",
        },
        {
            "arxiv_id": "2510.00915",
            "one_line": "Imperfect-verifier correction gives explicit noise hooks for FP/FN-calibrated RLVR.",
        },
        {
            "arxiv_id": "2402.06457",
            "one_line": "V-STaR keeps rejected traces to train a verifier rather than discarding false-positive evidence.",
        },
        {
            "arxiv_id": "2308.01825",
            "one_line": "RFT scaling says rejection-sampled fine-tuning helps weak models only with clean, diverse positives.",
        },
        {
            "arxiv_id": "2507.14843",
            "one_line": "Invisible Leash makes latent support a gate before RFT/RLVR precision spend.",
        },
        {
            "arxiv_id": "2410.17621",
            "one_line": "Step-level code PRM rewards turn sparse execution outcomes into dense training signal.",
        },
    ]


def _valid_receipt() -> dict[str, object]:
    return mod.build_receipt(
        methods_mapped=_valid_methods(),
        strongest_for_next_roadmap=[
            "calibrated_forward_noise_correction_before_rlvr",
            "augmentation_consistency_filter_before_rft_corpus",
            "vstar_rejected_trace_verifier_training",
            "step_level_process_reward_weighted_sft",
        ],
    )


def test_req_report_4094_spec_anchor_exists() -> None:
    """REQ-REPORT-4094: OpenSpec declares the precision-calibration ingestion."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4094" in spec
    assert "SCENARIO-REPORT-4094" in spec
    assert NOTE_PATH.as_posix() in spec
    assert RECEIPT_PATH.as_posix() in spec
    assert "arXiv:2510.00915" in spec
    assert "Bottom line for the .379 roadmap" in spec


def test_build_receipt_has_required_schema_fields_for_req_report_4094() -> None:
    """REQ-REPORT-4094: receipt exposes only the required artifact fields."""

    receipt = _valid_receipt()

    assert receipt == {
        "honest_verdict": "complete: sota_ingestion_precision_calibration_mapped",
        "methods_mapped": _valid_methods(),
        "strongest_for_next_roadmap": [
            "calibrated_forward_noise_correction_before_rlvr",
            "augmentation_consistency_filter_before_rft_corpus",
            "vstar_rejected_trace_verifier_training",
            "step_level_process_reward_weighted_sft",
        ],
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }


@pytest.mark.parametrize(
    ("bad_receipt", "message"),
    [
        (_valid_receipt() | {"honest_verdict": "draft"}, "terminal prefix"),
        (_valid_receipt() | {"methods_mapped": []}, "non-empty list"),
        (
            _valid_receipt() | {"methods_mapped": _valid_methods()[:4]},
            "five to eight",
        ),
        (
            _valid_receipt()
            | {
                "methods_mapped": [{"arxiv_id": "9999.99999", "one_line": "fake"}]
                + _valid_methods()[1:]
            },
            "verified arxiv",
        ),
        (
            _valid_receipt()
            | {
                "methods_mapped": [{"arxiv_id": "2411.02272", "title": "BARC"}]
                + _valid_methods()[1:]
            },
            "exactly arxiv_id and one_line",
        ),
        (
            _valid_receipt() | {"methods_mapped": [_valid_methods()[0]] + _valid_methods()[:-1]},
            "duplicate method",
        ),
        (
            _valid_receipt()
            | {"methods_mapped": [_valid_methods()[0] | {"one_line": ""}] + _valid_methods()[1:]},
            "one_line",
        ),
        (
            _valid_receipt() | {"strongest_for_next_roadmap": []},
            "strongest_for_next_roadmap",
        ),
        (
            _valid_receipt() | {"inference_substrate": "manual_guess"},
            "aggregation_from_upstream_artifacts",
        ),
    ],
)
def test_validate_receipt_rejects_schema_violations_for_scenario_report_4094(
    bad_receipt: dict[str, object], message: str
) -> None:
    """SCENARIO-REPORT-4094: invalid receipts fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_receipt(bad_receipt)


def test_validate_receipt_rejects_missing_and_extra_fields() -> None:
    """SCENARIO-REPORT-4094: receipt fields are bare and exact."""

    missing_receipt = _valid_receipt()
    missing_receipt.pop("methods_mapped")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_receipt(missing_receipt)

    extra_receipt = _valid_receipt()
    extra_receipt["methods_mapped_count"] = 7
    with pytest.raises(ValueError, match="unexpected fields"):
        mod.validate_receipt(extra_receipt)


def test_validate_markdown_note_checks_scenario_report_4094_sections() -> None:
    """SCENARIO-REPORT-4094: note maps methods to pipeline changes and pitfalls."""

    note = """
    # SOTA ingestion precision calibration

    ## Current precision-rescue + RFT anchor

    ## Augmentation-consistency ranking / BARC
    arXiv:2411.02272.
    Implementation over current precision-rescue + RFT pipeline: add ranking.
    Pitfalls / where it fails: augmentation checks can agree on spurious transforms.

    ## Noisy Data is Destructive to RLVR
    arXiv:2603.16140.
    Implementation over current precision-rescue + RFT pipeline: reverify labels.
    Pitfalls / where it fails: small clean sets may starve training.

    ## Imperfect-verifier noise correction
    arXiv:2510.00915.
    Implementation over current precision-rescue + RFT pipeline: estimate FP/FN.
    Pitfalls / where it fails: rate estimates drift after policy updates.

    ## V-STaR keep rejected traces
    arXiv:2402.06457.
    Implementation over current precision-rescue + RFT pipeline: train verifier pairs.
    Pitfalls / where it fails: rejected traces include hard true negatives and parser misses.

    ## RFT scaling for weak models
    arXiv:2308.01825.
    Implementation over current precision-rescue + RFT pipeline: use clean diverse positives.
    Pitfalls / where it fails: better bases gain less from extra positives.

    ## Invisible Leash support gate
    arXiv:2507.14843.
    Implementation over current precision-rescue + RFT pipeline: measure pass@k support.
    Pitfalls / where it fails: pass@1 gains can hide support shrinkage.

    ## Step-level process reward for code
    arXiv:2410.17621.
    Implementation over current precision-rescue + RFT pipeline: line-level rewards.
    Pitfalls / where it fails: dense local reward can optimize non-progress.

    ## Bottom line for the .379 roadmap
    Prioritize calibrated noise correction and augmentation consistency.
    """

    mod.validate_markdown_note(note)


def test_validate_markdown_note_rejects_missing_bottom_line() -> None:
    """SCENARIO-REPORT-4094: note must close the discover-to-plan loop."""

    with pytest.raises(ValueError, match="Bottom line"):
        mod.validate_markdown_note(
            "## Augmentation-consistency ranking / BARC\n"
            "Implementation over current precision-rescue + RFT pipeline\n"
            "Pitfalls / where it fails\n"
        )


def test_validate_markdown_note_rejects_missing_verified_citations() -> None:
    """SCENARIO-REPORT-4094: every mapped method claim cites a verified ID."""

    note = """
    ## Current precision-rescue + RFT anchor
    ## Augmentation-consistency ranking / BARC
    arXiv:2411.02272.
    Implementation over current precision-rescue + RFT pipeline.
    Pitfalls / where it fails.
    ## Noisy Data is Destructive to RLVR
    arXiv:2603.16140.
    Implementation over current precision-rescue + RFT pipeline.
    Pitfalls / where it fails.
    ## Imperfect-verifier noise correction
    arXiv:2510.00915.
    Implementation over current precision-rescue + RFT pipeline.
    Pitfalls / where it fails.
    ## V-STaR keep rejected traces
    arXiv:2402.06457.
    Implementation over current precision-rescue + RFT pipeline.
    Pitfalls / where it fails.
    ## RFT scaling for weak models
    arXiv:2308.01825.
    Implementation over current precision-rescue + RFT pipeline.
    Pitfalls / where it fails.
    ## Invisible Leash support gate
    arXiv:2507.14843.
    Implementation over current precision-rescue + RFT pipeline.
    Pitfalls / where it fails.
    ## Step-level process reward for code
    Implementation over current precision-rescue + RFT pipeline.
    Pitfalls / where it fails.
    ## Bottom line for the .379 roadmap
    Prioritize calibrated noise correction.
    """

    with pytest.raises(ValueError, match="verified arxiv citations"):
        mod.validate_markdown_note(note)


def test_write_outputs_updates_files_idempotently_for_req_report_4094(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-4094: writer emits note, receipt, and one studying section."""

    note_path = tmp_path / "note.md"
    receipt_path = tmp_path / "receipt.json"
    studying_path = tmp_path / "research-studying.md"
    studying_path.write_text("# Research Studying\n\nExisting body.\n", encoding="utf-8")

    receipt = mod.write_outputs(
        note_path=note_path,
        receipt_path=receipt_path,
        studying_path=studying_path,
    )
    second_receipt = mod.write_outputs(
        note_path=note_path,
        receipt_path=receipt_path,
        studying_path=studying_path,
    )

    mod.validate_receipt(receipt)
    mod.validate_receipt(second_receipt)
    mod.validate_markdown_note(note_path.read_text(encoding="utf-8"))
    saved_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    studying = studying_path.read_text(encoding="utf-8")

    assert saved_receipt == receipt
    assert studying.count("2026-06-12 Exp 4094") == 1
    assert "Bottom line for the .379 roadmap" in studying


def test_studying_section_update_handles_heading_layouts_for_req_report_4094() -> None:
    """REQ-REPORT-4094: studying update works before or between sections."""

    without_marker = "# Research Studying\n\n## Existing\nBody.\n"
    with_marker_and_next = mod._with_studying_section(without_marker)
    refreshed = mod._with_studying_section(with_marker_and_next)

    assert with_marker_and_next.index("2026-06-12 Exp 4094") < with_marker_and_next.index(
        "## Existing"
    )
    assert refreshed.count("2026-06-12 Exp 4094") == 1
    assert "## Existing\nBody." in refreshed


def test_deliverable_files_validate_against_req_report_4094() -> None:
    """REQ-REPORT-4094: committed note and JSON receipt satisfy the contract."""

    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    note = NOTE_PATH.read_text(encoding="utf-8")
    studying = STUDYING_PATH.read_text(encoding="utf-8")

    mod.validate_receipt(receipt)
    mod.validate_markdown_note(note)
    assert len(receipt["methods_mapped"]) == 7
    assert "2026-06-12 Exp 4094 - .378 precision-calibration SOTA ingestion ingested" in studying
    assert "Bottom line for the .379 roadmap" in studying
