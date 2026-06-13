"""Tests for REQ-REPORT-4121 / SCENARIO-REPORT-4121."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import sota_ingestion_trm_baseline_graft_4121 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
NOTE_PATH = Path("docs/research-notes/sota-ingestion-trm-baseline-graft-2026-06-13.md")
ARTIFACT_PATH = Path("results/experiment_4121_sota_ingestion_trm_baseline_graft.json")
STUDYING_PATH = Path("research-studying.md")


def _valid_methods() -> list[dict[str, str]]:
    return [
        {
            "name": "TRM resumable Sudoku baseline gate",
            "arxiv_id": "2510.04871",
            "url": "https://arxiv.org/abs/2510.04871",
            "implementation_over_stack": (
                "Resume the nano-trm Sudoku Extreme checkpoint until the baseline "
                "reproduction is trustworthy before treating verifier lift as meaningful."
            ),
            "failure_mode": (
                "A checkpoint can reload and still remain far below the published TRM "
                "target, making any verifier-graft conclusion underpowered."
            ),
        },
        {
            "name": "TTA-TRM full-fine-tune control",
            "arxiv_id": "2511.02886",
            "url": "https://arxiv.org/abs/2511.02886",
            "implementation_over_stack": (
                "Keep a no-verifier full-fine-tune arm beside verifier-admitted training "
                "so adaptation compute is not confused with verifier value."
            ),
            "failure_mode": (
                "Full fine-tuning can win by public-task adaptation or leakage unless "
                "checkpoint source, split, and optimizer budget are isolated."
            ),
        },
        {
            "name": "Verifier-guided adaptive candidate expansion",
            "arxiv_id": "2602.01070",
            "url": "https://arxiv.org/abs/2602.01070",
            "implementation_over_stack": (
                "Move exact Sudoku checks into candidate expansion so resumed TRM compute "
                "is spent on recoverable partial boards before post-hoc reranking."
            ),
            "failure_mode": (
                "Local verifier scores can prefer near-valid dead ends, so final exact "
                "validity and prune-error rate must remain authoritative."
            ),
        },
        {
            "name": "V-STaR accepted/rejected Sudoku selector",
            "arxiv_id": "2402.06457",
            "url": "https://arxiv.org/abs/2402.06457",
            "implementation_over_stack": (
                "Train a selector from exact-valid and verifier-rejected Sudoku traces "
                "sampled from the same resumed checkpoint."
            ),
            "failure_mode": (
                "Near-duplicate invalid boards can teach shallow artifacts unless the "
                "pool has real within-puzzle diversity."
            ),
        },
        {
            "name": "ReST resumable generate-filter-improve curriculum",
            "arxiv_id": "2308.08998",
            "url": "https://arxiv.org/abs/2308.08998",
            "implementation_over_stack": (
                "Cache generated Sudoku batches, filter them with the Carnot verifier, "
                "resume improvement from unique positives, and retain rejects for selectors."
            ),
            "failure_mode": (
                "If the baseline checkpoint rarely samples valid completions, the cached "
                "curriculum collapses into memorization or too few positives."
            ),
        },
    ]


def _valid_artifact() -> dict[str, object]:
    return mod.build_artifact(
        methods_mapped=_valid_methods(),
        flagged_for_v382="verifier_guided_adaptive_candidate_expansion_over_resumed_trm",
    )


def test_req_report_4121_spec_anchor_exists() -> None:
    """REQ-REPORT-4121: OpenSpec declares the TRM baseline-graft ingestion."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4121" in spec
    assert "SCENARIO-REPORT-4121" in spec
    assert NOTE_PATH.as_posix() in spec
    assert ARTIFACT_PATH.as_posix() in spec
    assert "flagged_for_v382" in spec
    assert "arXiv:2510.04871" in spec
    assert "arXiv:2602.01070" in spec


def test_build_artifact_has_required_schema_fields_for_req_report_4121() -> None:
    """REQ-REPORT-4121: artifact exposes the required principle-annotated fields."""

    artifact = _valid_artifact()

    assert artifact == {
        "honest_verdict": "complete: sota_ingestion_trm_baseline_graft_mapped",
        "methods_mapped": _valid_methods(),
        "flagged_for_v382": "verifier_guided_adaptive_candidate_expansion_over_resumed_trm",
        "field_principles": {
            "honest_verdict": (
                "Terminal-prefixed. Records ingestion completed with verifiable citations."
            ),
            "methods_mapped": (
                "Each method MUST carry a real arXiv ID/URL; an ingestion note without "
                "verifiable citations is treated as fabrication."
            ),
            "flagged_for_v382": (
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
        (_valid_artifact() | {"methods_mapped": _valid_methods()[:2]}, "three to five"),
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
            | {"methods_mapped": [{"name": "TRM", "arxiv_id": "2510.04871"}] + _valid_methods()[1:]},
            "exactly",
        ),
        (
            _valid_artifact() | {"methods_mapped": [_valid_methods()[0]] + _valid_methods()[:-1]},
            "duplicate method",
        ),
        (
            _valid_artifact()
            | {"methods_mapped": [_valid_methods()[0] | {"failure_mode": ""}] + _valid_methods()[1:]},
            "non-empty string",
        ),
        (_valid_artifact() | {"flagged_for_v382": ""}, "flagged_for_v382"),
        (
            _valid_artifact() | {"field_principles": {"honest_verdict": "loose"}},
            "field_principles",
        ),
    ],
)
def test_validate_artifact_rejects_schema_violations_for_scenario_report_4121(
    bad_artifact: dict[str, object], message: str
) -> None:
    """SCENARIO-REPORT-4121: invalid mapping artifacts fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_validate_artifact_rejects_missing_and_extra_fields() -> None:
    """SCENARIO-REPORT-4121: artifact fields are exact."""

    missing_artifact = _valid_artifact()
    missing_artifact.pop("methods_mapped")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing_artifact)

    extra_artifact = _valid_artifact()
    extra_artifact["inference_substrate"] = "aggregation_from_upstream_artifacts"
    with pytest.raises(ValueError, match="unexpected fields"):
        mod.validate_artifact(extra_artifact)


def test_validate_markdown_note_checks_scenario_report_4121_sections() -> None:
    """SCENARIO-REPORT-4121: note maps methods to implementation work and risks."""

    note = """
    # SOTA ingestion TRM baseline graft

    ## Current .381 resumable baseline-graft anchor
    arXiv:2510.04871 and arXiv:2511.02886 define the TRM substrate.

    ## TRM resumable Sudoku baseline gate
    arXiv:2510.04871.
    Implementation over nano-trm + Carnot-verifier stack: resume baseline.
    Pitfalls / where it fails: partial checkpoint.

    ## TTA-TRM full-fine-tune control
    arXiv:2511.02886.
    Implementation over nano-trm + Carnot-verifier stack: isolate adaptation.
    Pitfalls / where it fails: leakage.

    ## Verifier-guided adaptive candidate expansion
    arXiv:2602.01070, arXiv:2601.17223, and arXiv:2605.10325.
    Implementation over nano-trm + Carnot-verifier stack: prune during expansion.
    Pitfalls / where it fails: local checks miss final correctness.

    ## V-STaR accepted/rejected Sudoku selector
    arXiv:2402.06457.
    Implementation over nano-trm + Carnot-verifier stack: train selector pairs.
    Pitfalls / where it fails: duplicate traces.

    ## ReST resumable generate-filter-improve curriculum
    arXiv:2308.08998 and arXiv:2203.14465.
    Implementation over nano-trm + Carnot-verifier stack: cache, filter, improve.
    Pitfalls / where it fails: sparse positives.

    ## Flagged for the .382 roadmap
    verifier_guided_adaptive_candidate_expansion_over_resumed_trm
    """

    mod.validate_markdown_note(note)


def test_validate_markdown_note_rejects_missing_roadmap_flag() -> None:
    """SCENARIO-REPORT-4121: note must close the discover-to-plan loop."""

    with pytest.raises(ValueError, match="Flagged for the .382 roadmap"):
        mod.validate_markdown_note(
            "## Current .381 resumable baseline-graft anchor\n"
            "## TRM resumable Sudoku baseline gate\n"
            "arXiv:2510.04871.\n"
            "Implementation over nano-trm + Carnot-verifier stack\n"
            "Pitfalls / where it fails\n"
        )


def test_validate_markdown_note_rejects_missing_verified_citations() -> None:
    """SCENARIO-REPORT-4121: every mapped method cites a verified paper."""

    note = """
    ## Current .381 resumable baseline-graft anchor
    ## TRM resumable Sudoku baseline gate
    arXiv:2510.04871.
    Implementation over nano-trm + Carnot-verifier stack.
    Pitfalls / where it fails.
    ## TTA-TRM full-fine-tune control
    arXiv:2511.02886.
    Implementation over nano-trm + Carnot-verifier stack.
    Pitfalls / where it fails.
    ## Verifier-guided adaptive candidate expansion
    arXiv:2602.01070 and arXiv:2601.17223.
    Implementation over nano-trm + Carnot-verifier stack.
    Pitfalls / where it fails.
    ## V-STaR accepted/rejected Sudoku selector
    arXiv:2402.06457.
    Implementation over nano-trm + Carnot-verifier stack.
    Pitfalls / where it fails.
    ## ReST resumable generate-filter-improve curriculum
    arXiv:2308.08998.
    Implementation over nano-trm + Carnot-verifier stack.
    Pitfalls / where it fails.
    ## Flagged for the .382 roadmap
    verifier_guided_adaptive_candidate_expansion_over_resumed_trm
    """

    with pytest.raises(ValueError, match="verified arxiv citations"):
        mod.validate_markdown_note(note)


def test_write_outputs_updates_files_idempotently_for_req_report_4121(tmp_path: Path) -> None:
    """REQ-REPORT-4121: writer emits note, artifact, and one studying section."""

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
    assert studying.count("2026-06-13 Exp 4121") == 1
    assert "Flagged for .382" in studying


def test_studying_section_update_handles_heading_layouts_for_req_report_4121() -> None:
    """REQ-REPORT-4121: studying update works before or between sections."""

    without_marker = "# Research Studying\n\n## Existing\nBody.\n"
    with_marker_and_next = mod._with_studying_section(without_marker)
    refreshed = mod._with_studying_section(with_marker_and_next)
    no_heading = mod._with_studying_section("# Research Studying\nOnly body.\n")
    marker_at_end = mod._with_studying_section(with_marker_and_next.split("\n## Existing")[0])

    assert with_marker_and_next.index("2026-06-13 Exp 4121") < with_marker_and_next.index(
        "## Existing"
    )
    assert refreshed.count("2026-06-13 Exp 4121") == 1
    assert "## Existing\nBody." in refreshed
    assert no_heading.rstrip().endswith("selector/RFT work should stay blocked.")
    assert marker_at_end.count("2026-06-13 Exp 4121") == 1


def test_deliverable_files_validate_against_req_report_4121() -> None:
    """REQ-REPORT-4121: committed note and JSON artifact satisfy the contract."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    note = NOTE_PATH.read_text(encoding="utf-8")
    studying = STUDYING_PATH.read_text(encoding="utf-8")

    mod.validate_artifact(artifact)
    mod.validate_markdown_note(note)
    assert len(artifact["methods_mapped"]) >= 3
    assert artifact["flagged_for_v382"] == (
        "verifier_guided_adaptive_candidate_expansion_over_resumed_trm"
    )
    assert "2026-06-13 Exp 4121 - .381 TRM baseline-graft SOTA ingestion ingested" in studying
    assert (
        "Flagged for .382: "
        "`verifier_guided_adaptive_candidate_expansion_over_resumed_trm`"
    ) in studying
