"""Tests for REQ-REPORT-4130 / SCENARIO-REPORT-4130."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import sota_ingestion_resumable_training_4130 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
NOTE_PATH = Path("docs/research-notes/sota-ingestion-resumable-training-2026-06-13.md")
ARTIFACT_PATH = Path("results/experiment_4130_sota_ingestion_resumable_training.json")
STUDYING_PATH = Path("research-studying.md")


def _valid_methods() -> list[dict[str, str]]:
    return [
        {
            "name": "PyTorch Lightning full-state checkpoint resume",
            "arxiv_id_or_url": "https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html",
            "url": "https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html",
            "implementation_over_stack": (
                "Resume nano-trm bounded passes with Trainer.fit(..., ckpt_path=...) "
                "so global_step, optimizer state, and LR scheduler state continue."
            ),
            "failure_mode": (
                "Loading weights only or restarting the trainer silently rewinds the "
                "warmup/cosine schedule and fabricates long-horizon progress."
            ),
        },
        {
            "name": "PyTorch optimizer-state checkpoint contract",
            "arxiv_id_or_url": "https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html",
            "url": "https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html",
            "implementation_over_stack": (
                "If a run bypasses Lightning, persist model state, optimizer state, "
                "epoch or step, LR scheduler state, data cursor, and RNG receipt."
            ),
            "failure_mode": (
                "A model-only checkpoint loses momentum buffers and parameter-group "
                "learning rates, causing an unreported resume discontinuity."
            ),
        },
        {
            "name": "Lightning gradient-accumulation schedule",
            "arxiv_id_or_url": "https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html",
            "url": "https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html",
            "implementation_over_stack": (
                "Use accumulate_grad_batches or GradientAccumulationScheduler to keep "
                "effective batch size explicit across microbatches and bounded passes."
            ),
            "failure_mode": (
                "Counting microbatches as optimizer steps overstates the training "
                "horizon and moves the LR schedule at the wrong rate."
            ),
        },
        {
            "name": "TRM long-horizon baseline gate",
            "arxiv_id_or_url": "2510.04871",
            "url": "https://arxiv.org/abs/2510.04871",
            "implementation_over_stack": (
                "Accumulate resumed nano-trm Sudoku evidence by global optimizer step "
                "and held-out exact accuracy before any verifier-lift claim."
            ),
            "failure_mode": (
                "A checkpoint can reload correctly while still being an undertrained "
                "partial baseline, especially if LR state was reset."
            ),
        },
        {
            "name": "TTA-TRM bounded full-fine-tune control",
            "arxiv_id_or_url": "2511.02886",
            "url": "https://arxiv.org/abs/2511.02886",
            "implementation_over_stack": (
                "Keep a no-verifier full-fine-tune control with the same accumulated "
                "optimizer-step budget and the same resumed LR schedule receipts."
            ),
            "failure_mode": (
                "Full fine-tuning can win through adaptation compute alone, and a "
                "per-pass LR reset confounds the verifier comparison."
            ),
        },
    ]


def _valid_artifact() -> dict[str, object]:
    return mod.build_artifact(
        methods_mapped=_valid_methods(),
        flagged_for_v383="lightning_full_state_lr_scheduler_resume_gate_for_nano_trm_v383",
    )


def test_req_report_4130_spec_anchor_exists() -> None:
    """REQ-REPORT-4130: OpenSpec declares resumable-training ingestion."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4130" in spec
    assert "SCENARIO-REPORT-4130" in spec
    assert NOTE_PATH.as_posix() in spec
    assert ARTIFACT_PATH.as_posix() in spec
    assert "flagged_for_v383" in spec
    assert "arXiv:2510.04871" in spec
    assert "arXiv:2511.02886" in spec
    assert "PyTorch Lightning checkpoint-resume" in spec


def test_build_artifact_has_required_schema_fields_for_req_report_4130() -> None:
    """REQ-REPORT-4130: artifact exposes the required principle fields."""

    artifact = _valid_artifact()

    assert artifact == {
        "honest_verdict": "complete: sota_ingestion_resumable_training_mapped",
        "methods_mapped": _valid_methods(),
        "flagged_for_v383": "lightning_full_state_lr_scheduler_resume_gate_for_nano_trm_v383",
        "field_principles": {
            "honest_verdict": (
                "Terminal-prefixed. Records ingestion completed with verifiable citations."
            ),
            "methods_mapped": (
                "Each method/source MUST carry a real arXiv ID or canonical doc URL; "
                "an ingestion note without verifiable citations is treated as fabrication."
            ),
            "flagged_for_v383": (
                "Closes the discover->ingest->plan loop: names the strongest method "
                "for the next planner."
            ),
        },
    }


@pytest.mark.parametrize(
    ("bad_artifact", "message"),
    [
        (_valid_artifact() | {"honest_verdict": "draft"}, "terminal prefix"),
        (_valid_artifact() | {"field_principles": {"honest_verdict": "loose"}}, "field_principles"),
        (_valid_artifact() | {"methods_mapped": []}, "three to five"),
        (_valid_artifact() | {"methods_mapped": _valid_methods()[:2]}, "three to five"),
        (
            _valid_artifact()
            | {
                "methods_mapped": [
                    {
                        "name": "fake",
                        "arxiv_id_or_url": "9999.99999",
                        "url": "https://arxiv.org/abs/9999.99999",
                        "implementation_over_stack": "fake",
                        "failure_mode": "fake",
                    }
                ]
                + _valid_methods()[1:]
            },
            "verified arxiv ID or canonical doc URL",
        ),
        (
            _valid_artifact()
            | {
                "methods_mapped": [
                    {
                        "name": "TRM",
                        "arxiv_id_or_url": "2510.04871",
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
                "methods_mapped": [
                    {"name": "Lightning", "arxiv_id_or_url": _valid_methods()[0]["arxiv_id_or_url"]}
                ]
                + _valid_methods()[1:]
            },
            "exactly",
        ),
        (
            _valid_artifact()
            | {"methods_mapped": [_valid_methods()[0]] + _valid_methods()[:-1]},
            "duplicate source",
        ),
        (
            _valid_artifact()
            | {"methods_mapped": [_valid_methods()[0] | {"failure_mode": ""}] + _valid_methods()[1:]},
            "non-empty string",
        ),
        (_valid_artifact() | {"flagged_for_v383": ""}, "flagged_for_v383"),
    ],
)
def test_validate_artifact_rejects_schema_violations_for_scenario_report_4130(
    bad_artifact: dict[str, object], message: str
) -> None:
    """SCENARIO-REPORT-4130: invalid mapping artifacts fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_validate_artifact_rejects_non_dict_missing_and_extra_fields() -> None:
    """SCENARIO-REPORT-4130: artifact fields are exact."""

    missing_artifact = _valid_artifact()
    missing_artifact.pop("methods_mapped")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing_artifact)

    extra_artifact = _valid_artifact()
    extra_artifact["inference_substrate"] = "aggregation_from_docs"
    with pytest.raises(ValueError, match="unexpected fields"):
        mod.validate_artifact(extra_artifact)

    malformed_methods = _valid_artifact() | {"methods_mapped": ["not-a-dict"] + _valid_methods()[1:]}
    with pytest.raises(ValueError, match="exactly"):
        mod.validate_artifact(malformed_methods)


def test_validate_markdown_note_checks_scenario_report_4130_sections() -> None:
    """SCENARIO-REPORT-4130: note maps sources to implementation work and risks."""

    note = """
    # SOTA ingestion resumable training

    ## Current .382 resumable-training anchor
    arXiv:2510.04871 and arXiv:2511.02886 define the TRM substrate.

    ## PyTorch Lightning full-state checkpoint resume
    https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html
    Implementation over nano-trm + Carnot stack: use ckpt_path.
    Pitfalls / where it fails: schedule rewind.

    ## PyTorch optimizer-state checkpoint contract
    https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
    Implementation over nano-trm + Carnot stack: save optimizer state.
    Pitfalls / where it fails: momentum loss.

    ## Lightning gradient-accumulation schedule
    https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html
    Implementation over nano-trm + Carnot stack: count optimizer steps.
    Pitfalls / where it fails: microbatch-count drift.

    ## TRM long-horizon baseline gate
    arXiv:2510.04871.
    Implementation over nano-trm + Carnot stack: accumulate baseline evidence.
    Pitfalls / where it fails: partial checkpoint.

    ## TTA-TRM bounded full-fine-tune control
    arXiv:2511.02886.
    Implementation over nano-trm + Carnot stack: isolate adaptation compute.
    Pitfalls / where it fails: adaptation-only gain.

    ## Flagged for the .383 roadmap
    lightning_full_state_lr_scheduler_resume_gate_for_nano_trm_v383
    """

    mod.validate_markdown_note(note)


def test_validate_markdown_note_rejects_missing_roadmap_flag() -> None:
    """SCENARIO-REPORT-4130: note must close the discover-to-plan loop."""

    with pytest.raises(ValueError, match="Flagged for the .383 roadmap"):
        mod.validate_markdown_note(
            "## Current .382 resumable-training anchor\n"
            "## PyTorch Lightning full-state checkpoint resume\n"
            "https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html\n"
            "Implementation over nano-trm + Carnot stack\n"
            "Pitfalls / where it fails\n"
        )


def test_validate_markdown_note_rejects_missing_verified_sources() -> None:
    """SCENARIO-REPORT-4130: every mapped method cites a verified source."""

    note = """
    ## Current .382 resumable-training anchor
    ## PyTorch Lightning full-state checkpoint resume
    https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html
    Implementation over nano-trm + Carnot stack.
    Pitfalls / where it fails.
    ## PyTorch optimizer-state checkpoint contract
    https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
    Implementation over nano-trm + Carnot stack.
    Pitfalls / where it fails.
    ## Lightning gradient-accumulation schedule
    https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html
    Implementation over nano-trm + Carnot stack.
    Pitfalls / where it fails.
    ## TRM long-horizon baseline gate
    arXiv:2510.04871.
    Implementation over nano-trm + Carnot stack.
    Pitfalls / where it fails.
    ## TTA-TRM bounded full-fine-tune control
    Implementation over nano-trm + Carnot stack.
    Pitfalls / where it fails.
    ## Flagged for the .383 roadmap
    lightning_full_state_lr_scheduler_resume_gate_for_nano_trm_v383
    """

    with pytest.raises(ValueError, match="verified source citations"):
        mod.validate_markdown_note(note)


def test_write_outputs_updates_files_idempotently_for_req_report_4130(tmp_path: Path) -> None:
    """REQ-REPORT-4130: writer emits note, artifact, and one studying section."""

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
    assert studying.count("2026-06-13 Exp 4130") == 1
    assert "Flagged for .383" in studying


def test_studying_section_update_handles_heading_layouts_for_req_report_4130() -> None:
    """REQ-REPORT-4130: studying update works before or between sections."""

    without_marker = "# Research Studying\n\n## Existing\nBody.\n"
    with_marker_and_next = mod._with_studying_section(without_marker)
    refreshed = mod._with_studying_section(with_marker_and_next)
    no_heading = mod._with_studying_section("# Research Studying\nOnly body.\n")
    marker_at_end = mod._with_studying_section(with_marker_and_next.split("\n## Existing")[0])

    assert with_marker_and_next.index("2026-06-13 Exp 4130") < with_marker_and_next.index(
        "## Existing"
    )
    assert refreshed.count("2026-06-13 Exp 4130") == 1
    assert "## Existing\nBody." in refreshed
    assert no_heading.rstrip().endswith("per-step verifier work.")
    assert marker_at_end.count("2026-06-13 Exp 4130") == 1


def test_deliverable_files_validate_against_req_report_4130() -> None:
    """REQ-REPORT-4130: committed note and JSON artifact satisfy the contract."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    note = NOTE_PATH.read_text(encoding="utf-8")
    studying = STUDYING_PATH.read_text(encoding="utf-8")

    mod.validate_artifact(artifact)
    mod.validate_markdown_note(note)
    assert len(artifact["methods_mapped"]) >= 3
    assert artifact["flagged_for_v383"] == (
        "lightning_full_state_lr_scheduler_resume_gate_for_nano_trm_v383"
    )
    assert (
        "2026-06-13 Exp 4130 - .382 resumable-training SOTA ingestion ingested"
        in studying
    )
    assert (
        "Flagged for .383: "
        "`lightning_full_state_lr_scheduler_resume_gate_for_nano_trm_v383`"
    ) in studying
