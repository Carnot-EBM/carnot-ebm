"""Tests for REQ-REPORT-4102 / SCENARIO-REPORT-4102."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import sota_ingestion_trm_self_training_4102 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
NOTE_PATH = Path(
    "docs/research-notes/sota-ingestion-trm-self-training-2026-06-12.md"
)
ARTIFACT_PATH = Path("results/experiment_4102_sota_ingestion_trm_self_training.json")
STUDYING_PATH = Path("research-studying.md")


def _valid_methods() -> list[dict[str, str]]:
    return [
        {
            "name": "V-STaR keep-rejected verifier training",
            "arxiv_id": "2402.06457",
            "url": "https://arxiv.org/abs/2402.06457",
            "implementation_over_stack": (
                "Label nano-trm candidate traces with Carnot verifier outcomes and train a "
                "contrastive selector on accepted and rejected traces."
            ),
            "failure_mode": (
                "False-positive verifier labels turn rejected-trace learning into reward "
                "hacking unless calibration gates hold first."
            ),
        },
        {
            "name": "STaR / ReST generate-filter-improve loop",
            "arxiv_id": "2203.14465",
            "url": "https://arxiv.org/abs/2203.14465",
            "implementation_over_stack": (
                "Iterate nano-trm sampling, verifier filtering, and full fine-tuning from "
                "the reusable cached ARC trace pool."
            ),
            "failure_mode": (
                "The loop only amplifies traces already in the model support and discards "
                "hard negative structure."
            ),
        },
        {
            "name": "TTA-TRM full fine-tuning with verifier admission",
            "arxiv_id": "2511.02886",
            "url": "https://arxiv.org/abs/2511.02886",
            "implementation_over_stack": (
                "Use public-task pretraining plus bounded full fine-tuning, with Carnot "
                "verifier precision gates controlling which task traces enter adaptation."
            ),
            "failure_mode": (
                "It can become task memorization or leakage if public/private splits and "
                "full-finetune budgets are not isolated."
            ),
        },
        {
            "name": "Imperfect-verifier forward correction",
            "arxiv_id": "2510.00915",
            "url": "https://arxiv.org/abs/2510.00915",
            "implementation_over_stack": (
                "Attach FP/FN calibration metadata to verifier-certified TRM rewards and "
                "weight updates instead of treating the verifier as noiseless."
            ),
            "failure_mode": (
                "Noise-rate estimates drift after the TRM policy changes, so stale "
                "correction can bias the next RFT round."
            ),
        },
        {
            "name": "Verifiable process rewards for recursive steps",
            "arxiv_id": "2605.10325",
            "url": "https://arxiv.org/abs/2605.10325",
            "implementation_over_stack": (
                "Score each recursive grid-edit step with deterministic Carnot verifier "
                "checks before outcome-level hidden-test selection."
            ),
            "failure_mode": (
                "Locally valid recursive edits can still fail the final ARC transformation "
                "unless dense rewards are outcome-calibrated."
            ),
        },
    ]


def _valid_artifact() -> dict[str, object]:
    return mod.build_artifact(
        methods_mapped=_valid_methods(),
        flagged_for_v380="vstar_rejected_trace_selector_for_trm_rft",
    )


def test_req_report_4102_spec_anchor_exists() -> None:
    """REQ-REPORT-4102: OpenSpec declares the TRM self-training ingestion."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4102" in spec
    assert "SCENARIO-REPORT-4102" in spec
    assert NOTE_PATH.as_posix() in spec
    assert ARTIFACT_PATH.as_posix() in spec
    assert "flagged_for_v380" in spec
    assert "arXiv:2511.02886" in spec


def test_build_artifact_has_required_schema_fields_for_req_report_4102() -> None:
    """REQ-REPORT-4102: artifact exposes only the required mapping fields."""

    artifact = _valid_artifact()

    assert artifact == {
        "honest_verdict": "complete: sota_ingestion_trm_self_training_mapped",
        "methods_mapped": _valid_methods(),
        "flagged_for_v380": "vstar_rejected_trace_selector_for_trm_rft",
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
                        "name": "V-STaR",
                        "arxiv_id": "2402.06457",
                        "url": "https://example.com/2402.06457",
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
                "methods_mapped": [
                    {"name": "V-STaR", "arxiv_id": "2402.06457"}
                ]
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
        (_valid_artifact() | {"flagged_for_v380": ""}, "flagged_for_v380"),
    ],
)
def test_validate_artifact_rejects_schema_violations_for_scenario_report_4102(
    bad_artifact: dict[str, object], message: str
) -> None:
    """SCENARIO-REPORT-4102: invalid mapping artifacts fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_validate_artifact_rejects_missing_and_extra_fields() -> None:
    """SCENARIO-REPORT-4102: artifact fields are bare and exact."""

    missing_artifact = _valid_artifact()
    missing_artifact.pop("methods_mapped")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing_artifact)

    extra_artifact = _valid_artifact()
    extra_artifact["inference_substrate"] = "aggregation_from_upstream_artifacts"
    with pytest.raises(ValueError, match="unexpected fields"):
        mod.validate_artifact(extra_artifact)


def test_validate_markdown_note_checks_scenario_report_4102_sections() -> None:
    """SCENARIO-REPORT-4102: note maps methods to implementation work and risks."""

    note = """
    # SOTA ingestion TRM self-training

    ## Current .379 TRM verifier-RFT anchor
    arXiv:2605.30290 is adjacent verifier training context.

    ## V-STaR keep-rejected verifier training
    arXiv:2402.06457.
    Implementation over nano-trm + Carnot-verifier stack: train selector pairs.
    Pitfalls / where it fails: verifier false positives poison DPO labels.

    ## STaR / ReST generate-filter-improve loop
    arXiv:2203.14465 and arXiv:2308.08998.
    Implementation over nano-trm + Carnot-verifier stack: generate and filter.
    Pitfalls / where it fails: no same-pool support means no learning target.

    ## TTA-TRM full fine-tuning
    arXiv:2511.02886.
    Implementation over nano-trm + Carnot-verifier stack: full fine-tune.
    Pitfalls / where it fails: task leakage.

    ## Imperfect-verifier correction
    arXiv:2510.00915.
    Implementation over nano-trm + Carnot-verifier stack: FP/FN weighting.
    Pitfalls / where it fails: nonstationary rates.

    ## Verifiable process rewards
    arXiv:2601.17223 and arXiv:2605.10325.
    Implementation over nano-trm + Carnot-verifier stack: dense step reward.
    Pitfalls / where it fails: local reward can miss global correctness.

    ## Flagged for the .380 roadmap
    vstar_rejected_trace_selector_for_trm_rft
    """

    mod.validate_markdown_note(note)


def test_validate_markdown_note_rejects_missing_roadmap_flag() -> None:
    """SCENARIO-REPORT-4102: note must close the discover-to-plan loop."""

    with pytest.raises(ValueError, match="Flagged for the .380 roadmap"):
        mod.validate_markdown_note(
            "## Current .379 TRM verifier-RFT anchor\n"
            "## V-STaR keep-rejected verifier training\n"
            "arXiv:2402.06457.\n"
            "Implementation over nano-trm + Carnot-verifier stack\n"
            "Pitfalls / where it fails\n"
        )


def test_validate_markdown_note_rejects_missing_verified_citations() -> None:
    """SCENARIO-REPORT-4102: every mapped method cites a verified paper."""

    note = """
    ## Current .379 TRM verifier-RFT anchor
    ## V-STaR keep-rejected verifier training
    arXiv:2402.06457.
    Implementation over nano-trm + Carnot-verifier stack.
    Pitfalls / where it fails.
    ## STaR / ReST generate-filter-improve loop
    arXiv:2203.14465 arXiv:2308.08998.
    Implementation over nano-trm + Carnot-verifier stack.
    Pitfalls / where it fails.
    ## TTA-TRM full fine-tuning
    arXiv:2511.02886.
    Implementation over nano-trm + Carnot-verifier stack.
    Pitfalls / where it fails.
    ## Imperfect-verifier correction
    arXiv:2510.00915.
    Implementation over nano-trm + Carnot-verifier stack.
    Pitfalls / where it fails.
    ## Verifiable process rewards
    arXiv:2601.17223.
    Implementation over nano-trm + Carnot-verifier stack.
    Pitfalls / where it fails.
    ## Flagged for the .380 roadmap
    vstar_rejected_trace_selector_for_trm_rft
    """

    with pytest.raises(ValueError, match="verified arxiv citations"):
        mod.validate_markdown_note(note)


def test_write_outputs_updates_files_idempotently_for_req_report_4102(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-4102: writer emits note, artifact, and one studying section."""

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
    assert studying.count("2026-06-12 Exp 4102") == 1
    assert "Flagged for .380" in studying


def test_studying_section_update_handles_heading_layouts_for_req_report_4102() -> None:
    """REQ-REPORT-4102: studying update works before or between sections."""

    without_marker = "# Research Studying\n\n## Existing\nBody.\n"
    with_marker_and_next = mod._with_studying_section(without_marker)
    refreshed = mod._with_studying_section(with_marker_and_next)

    assert with_marker_and_next.index("2026-06-12 Exp 4102") < with_marker_and_next.index(
        "## Existing"
    )
    assert refreshed.count("2026-06-12 Exp 4102") == 1
    assert "## Existing\nBody." in refreshed


def test_deliverable_files_validate_against_req_report_4102() -> None:
    """REQ-REPORT-4102: committed note and JSON artifact satisfy the contract."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    note = NOTE_PATH.read_text(encoding="utf-8")
    studying = STUDYING_PATH.read_text(encoding="utf-8")

    mod.validate_artifact(artifact)
    mod.validate_markdown_note(note)
    assert len(artifact["methods_mapped"]) >= 3
    assert artifact["flagged_for_v380"] == "vstar_rejected_trace_selector_for_trm_rft"
    assert "2026-06-12 Exp 4102 - .379 TRM self-training SOTA ingestion ingested" in studying
    assert "Flagged for .380: `vstar_rejected_trace_selector_for_trm_rft`" in studying
