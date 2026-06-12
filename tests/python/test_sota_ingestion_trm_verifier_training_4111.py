"""Tests for REQ-REPORT-4111 / SCENARIO-REPORT-4111."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import sota_ingestion_trm_verifier_training_4111 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
NOTE_PATH = Path(
    "docs/research-notes/sota-ingestion-trm-verifier-training-2026-06-12.md"
)
ARTIFACT_PATH = Path("results/experiment_4111_sota_ingestion_trm_verifier_training.json")
STUDYING_PATH = Path("research-studying.md")


def _valid_methods() -> list[dict[str, str]]:
    return [
        {
            "name": "TRM Sudoku baseline reproduction",
            "arxiv_id": "2510.04871",
            "url": "https://arxiv.org/abs/2510.04871",
            "implementation_over_stack": (
                "Reproduce the native nano-trm Sudoku Extreme baseline before claiming "
                "any Carnot verifier lift."
            ),
            "failure_mode": (
                "A partial checkpoint can validate the mechanism while still failing to "
                "match the published accuracy target."
            ),
        },
        {
            "name": "TTA-TRM full fine-tuning control",
            "arxiv_id": "2511.02886",
            "url": "https://arxiv.org/abs/2511.02886",
            "implementation_over_stack": (
                "Use bounded full fine-tuning as the adaptation control so verifier "
                "admission is separated from generic task adaptation."
            ),
            "failure_mode": (
                "Compute leakage or public-task memorization can look like verifier value "
                "unless full-finetune and no-verifier arms are isolated."
            ),
        },
        {
            "name": "V-STaR accepted/rejected trace selector",
            "arxiv_id": "2402.06457",
            "url": "https://arxiv.org/abs/2402.06457",
            "implementation_over_stack": (
                "Train a selector from verifier-valid and verifier-invalid Sudoku "
                "candidate traces sampled from the same TRM checkpoint."
            ),
            "failure_mode": (
                "If candidate pools are near-duplicates or already vote-saturated, the "
                "selector learns surface artifacts without pass@1 lift."
            ),
        },
        {
            "name": "STaR / ReST generate-filter-improve loop",
            "arxiv_id": "2203.14465",
            "url": "https://arxiv.org/abs/2203.14465",
            "implementation_over_stack": (
                "Generate Sudoku traces, filter with the executable Carnot verifier, "
                "fine-tune on unique positives, and repeat from cached batches."
            ),
            "failure_mode": (
                "Filtering cannot teach solutions the TRM never samples, and sparse "
                "positives can collapse the improve step."
            ),
        },
        {
            "name": "Verifier-guided adaptive Sudoku search",
            "arxiv_id": "2602.01070",
            "url": "https://arxiv.org/abs/2602.01070",
            "implementation_over_stack": (
                "Move the Sudoku verifier into candidate expansion so compute is spent "
                "on promising partial completions before post-hoc reranking or RFT."
            ),
            "failure_mode": (
                "Local row, column, and box satisfaction can still prefer near-valid "
                "dead ends unless final exact validity remains authoritative."
            ),
        },
    ]


def _valid_artifact() -> dict[str, object]:
    return mod.build_artifact(
        methods_mapped=_valid_methods(),
        flagged_for_v381="verifier_guided_adaptive_sudoku_search_before_training",
    )


def test_req_report_4111_spec_anchor_exists() -> None:
    """REQ-REPORT-4111: OpenSpec declares the TRM verifier-training ingestion."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4111" in spec
    assert "SCENARIO-REPORT-4111" in spec
    assert NOTE_PATH.as_posix() in spec
    assert ARTIFACT_PATH.as_posix() in spec
    assert "flagged_for_v381" in spec
    assert "arXiv:2510.04871" in spec
    assert "arXiv:2602.01070" in spec


def test_build_artifact_has_required_schema_fields_for_req_report_4111() -> None:
    """REQ-REPORT-4111: artifact exposes the required principle-annotated fields."""

    artifact = _valid_artifact()

    assert artifact == {
        "honest_verdict": "complete: sota_ingestion_trm_verifier_training_mapped",
        "methods_mapped": _valid_methods(),
        "flagged_for_v381": "verifier_guided_adaptive_sudoku_search_before_training",
        "field_principles": {
            "honest_verdict": (
                "Terminal-prefixed. Records ingestion completed with verifiable citations."
            ),
            "methods_mapped": (
                "Each method MUST carry a real arXiv ID/URL; an ingestion note without "
                "verifiable citations is treated as fabrication."
            ),
            "flagged_for_v381": (
                "Closes the discover->ingest->plan loop: names the strongest method "
                "for the next planner."
            ),
        },
    }


@pytest.mark.parametrize(
    ("bad_artifact", "message"),
    [
        (_valid_artifact() | {"honest_verdict": "draft"}, "terminal prefix"),
        (_valid_artifact() | {"methods_mapped": []}, "three to five"),
        (
            _valid_artifact() | {"methods_mapped": _valid_methods()[:2]},
            "three to five",
        ),
        (
            _valid_artifact()
            | {
                "methods_mapped": [
                    {
                        "name": "fake",
                        "arxiv_id": "9999.99999",
                        "url": "https://arxiv.org/abs/9999.99999",
                        "implementation_over_stack": "fake",
                        "failure_mode": "fake",
                    }
                ]
                + _valid_methods()[1:]
            },
            "verified arxiv",
        ),
        (
            _valid_artifact()
            | {
                "methods_mapped": [
                    {
                        "name": "TRM",
                        "arxiv_id": "2510.04871",
                        "url": "https://example.com/2510.04871",
                        "implementation_over_stack": "use it",
                        "failure_mode": "breaks",
                    }
                ]
                + _valid_methods()[1:]
            },
            "url",
        ),
        (
            _valid_artifact()
            | {
                "methods_mapped": [{"name": "TRM", "arxiv_id": "2510.04871"}]
                + _valid_methods()[1:]
            },
            "exactly",
        ),
        (
            _valid_artifact()
            | {"methods_mapped": [_valid_methods()[0]] + _valid_methods()[:-1]},
            "duplicate method",
        ),
        (
            _valid_artifact()
            | {
                "methods_mapped": [
                    _valid_methods()[0] | {"failure_mode": ""}
                ]
                + _valid_methods()[1:]
            },
            "non-empty string",
        ),
        (_valid_artifact() | {"flagged_for_v381": ""}, "flagged_for_v381"),
        (
            _valid_artifact() | {"field_principles": {"honest_verdict": "loose"}},
            "field_principles",
        ),
    ],
)
def test_validate_artifact_rejects_schema_violations_for_scenario_report_4111(
    bad_artifact: dict[str, object], message: str
) -> None:
    """SCENARIO-REPORT-4111: invalid mapping artifacts fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_validate_artifact_rejects_missing_and_extra_fields() -> None:
    """SCENARIO-REPORT-4111: artifact fields are exact."""

    missing_artifact = _valid_artifact()
    missing_artifact.pop("methods_mapped")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing_artifact)

    extra_artifact = _valid_artifact()
    extra_artifact["inference_substrate"] = "aggregation_from_upstream_artifacts"
    with pytest.raises(ValueError, match="unexpected fields"):
        mod.validate_artifact(extra_artifact)


def test_validate_markdown_note_checks_scenario_report_4111_sections() -> None:
    """SCENARIO-REPORT-4111: note maps methods to implementation work and risks."""

    note = """
    # SOTA ingestion TRM verifier training

    ## Current .380 baseline-plus-verifier anchor
    arXiv:2510.04871 and arXiv:2511.02886 define the TRM substrate.

    ## TRM Sudoku baseline reproduction
    arXiv:2510.04871.
    Implementation over nano-trm + Carnot-verifier stack: reproduce baseline.
    Pitfalls / where it fails: partial baseline.

    ## TTA-TRM full fine-tuning control
    arXiv:2511.02886.
    Implementation over nano-trm + Carnot-verifier stack: isolate adaptation.
    Pitfalls / where it fails: leakage.

    ## V-STaR accepted/rejected trace selector
    arXiv:2402.06457.
    Implementation over nano-trm + Carnot-verifier stack: train selector pairs.
    Pitfalls / where it fails: duplicate traces.

    ## STaR / ReST generate-filter-improve loop
    arXiv:2203.14465 and arXiv:2308.08998.
    Implementation over nano-trm + Carnot-verifier stack: generate and filter.
    Pitfalls / where it fails: no same-pool support.

    ## Verifier-guided adaptive Sudoku search
    arXiv:2602.01070, arXiv:2601.17223, and arXiv:2605.10325.
    Implementation over nano-trm + Carnot-verifier stack: in-loop pruning.
    Pitfalls / where it fails: local checks miss final correctness.

    ## Flagged for the .381 roadmap
    verifier_guided_adaptive_sudoku_search_before_training
    """

    mod.validate_markdown_note(note)


def test_validate_markdown_note_rejects_missing_roadmap_flag() -> None:
    """SCENARIO-REPORT-4111: note must close the discover-to-plan loop."""

    with pytest.raises(ValueError, match="Flagged for the .381 roadmap"):
        mod.validate_markdown_note(
            "## Current .380 baseline-plus-verifier anchor\n"
            "## TRM Sudoku baseline reproduction\n"
            "arXiv:2510.04871.\n"
            "Implementation over nano-trm + Carnot-verifier stack\n"
            "Pitfalls / where it fails\n"
        )


def test_validate_markdown_note_rejects_missing_verified_citations() -> None:
    """SCENARIO-REPORT-4111: every mapped method cites a verified paper."""

    note = """
    ## Current .380 baseline-plus-verifier anchor
    ## TRM Sudoku baseline reproduction
    arXiv:2510.04871.
    Implementation over nano-trm + Carnot-verifier stack.
    Pitfalls / where it fails.
    ## TTA-TRM full fine-tuning control
    arXiv:2511.02886.
    Implementation over nano-trm + Carnot-verifier stack.
    Pitfalls / where it fails.
    ## V-STaR accepted/rejected trace selector
    arXiv:2402.06457.
    Implementation over nano-trm + Carnot-verifier stack.
    Pitfalls / where it fails.
    ## STaR / ReST generate-filter-improve loop
    arXiv:2203.14465.
    Implementation over nano-trm + Carnot-verifier stack.
    Pitfalls / where it fails.
    ## Verifier-guided adaptive Sudoku search
    arXiv:2602.01070 and arXiv:2601.17223.
    Implementation over nano-trm + Carnot-verifier stack.
    Pitfalls / where it fails.
    ## Flagged for the .381 roadmap
    verifier_guided_adaptive_sudoku_search_before_training
    """

    with pytest.raises(ValueError, match="verified arxiv citations"):
        mod.validate_markdown_note(note)


def test_write_outputs_updates_files_idempotently_for_req_report_4111(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-4111: writer emits note, artifact, and one studying section."""

    note_path = tmp_path / "note.md"
    artifact_path = tmp_path / "artifact.json"
    studying_path = tmp_path / "research-studying.md"
    studying_path.write_text("# Research Studying\n\nExisting body.\n", encoding="utf-8")

    artifact = mod.write_outputs(
        note_path=note_path,
        artifact_path=artifact_path,
        studying_path=studying_path,
    )
    second_artifact = mod.write_outputs(
        note_path=note_path,
        artifact_path=artifact_path,
        studying_path=studying_path,
    )

    mod.validate_artifact(artifact)
    mod.validate_artifact(second_artifact)
    mod.validate_markdown_note(note_path.read_text(encoding="utf-8"))
    saved_artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    studying = studying_path.read_text(encoding="utf-8")

    assert saved_artifact == artifact
    assert studying.count("2026-06-12 Exp 4111") == 1
    assert "Flagged for .381" in studying


def test_studying_section_update_handles_heading_layouts_for_req_report_4111() -> None:
    """REQ-REPORT-4111: studying update works before or between sections."""

    without_marker = "# Research Studying\n\n## Existing\nBody.\n"
    with_marker_and_next = mod._with_studying_section(without_marker)
    refreshed = mod._with_studying_section(with_marker_and_next)
    no_heading = mod._with_studying_section("# Research Studying\nOnly body.\n")
    marker_at_end = mod._with_studying_section(with_marker_and_next.split("\n## Existing")[0])

    assert with_marker_and_next.index("2026-06-12 Exp 4111") < with_marker_and_next.index(
        "## Existing"
    )
    assert refreshed.count("2026-06-12 Exp 4111") == 1
    assert "## Existing\nBody." in refreshed
    assert no_heading.rstrip().endswith("otherwise keep V-STaR and RFT routes blocked.")
    assert marker_at_end.count("2026-06-12 Exp 4111") == 1


def test_deliverable_files_validate_against_req_report_4111() -> None:
    """REQ-REPORT-4111: committed note and JSON artifact satisfy the contract."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    note = NOTE_PATH.read_text(encoding="utf-8")
    studying = STUDYING_PATH.read_text(encoding="utf-8")

    mod.validate_artifact(artifact)
    mod.validate_markdown_note(note)
    assert len(artifact["methods_mapped"]) >= 3
    assert artifact["flagged_for_v381"] == "verifier_guided_adaptive_sudoku_search_before_training"
    assert "2026-06-12 Exp 4111 - .380 TRM verifier-training SOTA ingestion ingested" in studying
    assert "Flagged for .381: `verifier_guided_adaptive_sudoku_search_before_training`" in studying
