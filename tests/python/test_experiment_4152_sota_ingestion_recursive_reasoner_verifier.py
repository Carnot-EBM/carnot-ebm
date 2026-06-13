"""Tests for REQ-REPORT-4152 / SCENARIO-REPORT-4152."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_4152_sota_ingestion_recursive_reasoner_verifier as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
NOTE_PATH = Path(
    "docs/research-notes/"
    "sota-ingestion-recursive-reasoner-verifier-energy-guidance-2026-06-13.md"
)
ARTIFACT_PATH = Path(
    "results/experiment_4152_sota_ingestion_recursive_reasoner_verifier.json"
)
STUDYING_PATH = Path("research-studying.md")
DIFFUSIONGEMMA_URL = "https://ai.google.dev/gemma/docs/diffusiongemma"


def _valid_methods() -> list[dict[str, str]]:
    return [
        {
            "name": "TRM nano-trm recursive baseline gate",
            "arxiv_id_or_url": "2510.04871",
            "url": "https://arxiv.org/abs/2510.04871",
            "implementation_over_stack": (
                "Keep nano-trm as the recursive Sudoku substrate and measure "
                "oracle headroom before attributing gains to the Carnot verifier."
            ),
            "failure_mode": (
                "An undertrained or no-headroom baseline makes verifier-guided "
                "training and energy guidance uninformative."
            ),
        },
        {
            "name": "TTA-TRM adaptation-control arm",
            "arxiv_id_or_url": "2511.02886",
            "url": "https://arxiv.org/abs/2511.02886",
            "implementation_over_stack": (
                "Run the same bounded fine-tuning budget without Carnot verifier "
                "labels so adaptation compute is isolated from verifier value."
            ),
            "failure_mode": (
                "Full fine-tuning can improve the tiny model by itself, so a "
                "verifier-labeled arm without this control overclaims causality."
            ),
        },
        {
            "name": "V-STaR accepted/rejected trace selector",
            "arxiv_id_or_url": "2402.06457",
            "url": "https://arxiv.org/abs/2402.06457",
            "implementation_over_stack": (
                "Retain accepted and rejected nano-trm traces and train a selector "
                "or pairwise verifier before spending on another generator pass."
            ),
            "failure_mode": (
                "If the candidate pool has correlated errors or false-positive "
                "labels, the selector learns artifacts rather than correctness."
            ),
        },
        {
            "name": "SEDD discrete diffusion score-energy formalism",
            "arxiv_id_or_url": "2310.16834",
            "url": "https://arxiv.org/abs/2310.16834",
            "implementation_over_stack": (
                "Use score-entropy discrete diffusion as the formal bridge for "
                "adding Carnot verifier energy during denoising instead of after it."
            ),
            "failure_mode": (
                "SEDD is a generator objective, not a verifier; an uncalibrated "
                "external energy can damage fluency or collapse diversity."
            ),
        },
        {
            "name": "Classifier-guided diffusion energy precedent",
            "arxiv_id_or_url": "2105.05233",
            "url": "https://arxiv.org/abs/2105.05233",
            "implementation_over_stack": (
                "Treat Carnot verifier scores as the discrete-token analogue of a "
                "guidance energy that reshapes the denoising choice distribution."
            ),
            "failure_mode": (
                "Over-guidance can trade away diversity and create verifier-shaped "
                "but invalid samples unless guidance weights are ablated."
            ),
        },
        {
            "name": "Classifier-free diffusion guidance control",
            "arxiv_id_or_url": "2207.12598",
            "url": "https://arxiv.org/abs/2207.12598",
            "implementation_over_stack": (
                "Keep a no-external-verifier guidance control so Carnot energy is "
                "compared against ordinary conditional/unconditional score mixing."
            ),
            "failure_mode": (
                "A guidance win can come from generic conditioning strength rather "
                "than the Carnot verifier unless this control is included."
            ),
        },
        {
            "name": "DiffusionGemma queued discrete-text substrate",
            "arxiv_id_or_url": DIFFUSIONGEMMA_URL,
            "url": DIFFUSIONGEMMA_URL,
            "implementation_over_stack": (
                "Queue DiffusionGemma as the open-weight block-diffusion generator "
                "for verifier-energy guidance after the verifier discrimination gate."
            ),
            "failure_mode": (
                "DiffusionGemma is a generator substrate, not evidence that Carnot "
                "verifier guidance works; base-task quality and the gate must be measured."
            ),
        },
    ]


def _valid_artifact() -> dict[str, object]:
    return mod.build_artifact(
        methods_mapped=_valid_methods(),
        flagged_for_v385="diffusiongemma_sedd_verifier_energy_guidance_probe_v385",
    )


def test_req_report_4152_spec_anchor_exists() -> None:
    """REQ-REPORT-4152: OpenSpec declares energy-guidance ingestion."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4152" in spec
    assert "SCENARIO-REPORT-4152" in spec
    assert NOTE_PATH.as_posix() in spec
    assert ARTIFACT_PATH.as_posix() in spec
    assert "flagged_for_v385" in spec
    assert "arXiv:2510.04871" in spec
    assert "arXiv:2511.02886" in spec
    assert "arXiv:2402.06457" in spec
    assert "arXiv:2310.16834" in spec
    assert "arXiv:2105.05233" in spec
    assert "arXiv:2207.12598" in spec
    assert DIFFUSIONGEMMA_URL in spec


def test_build_artifact_has_required_schema_fields_for_req_report_4152() -> None:
    """REQ-REPORT-4152: artifact exposes the required principle fields."""

    artifact = _valid_artifact()

    assert artifact == {
        "honest_verdict": (
            "complete: "
            "sota_ingestion_recursive_reasoner_verifier_energy_guidance_mapped"
        ),
        "methods_mapped": _valid_methods(),
        "flagged_for_v385": "diffusiongemma_sedd_verifier_energy_guidance_probe_v385",
        "field_principles": {
            "honest_verdict": (
                "Terminal-prefixed. Records ingestion completed with verifiable citations."
            ),
            "methods_mapped": (
                "Each method/source MUST carry a real arXiv ID/URL; an ingestion "
                "note without verifiable citations is treated as fabrication."
            ),
            "flagged_for_v385": (
                "Closes the discover->ingest->plan loop: names the strongest method "
                "for the next planner."
            ),
        },
    }


@pytest.mark.parametrize(
    ("bad_artifact", "message"),
    [
        (_valid_artifact() | {"honest_verdict": "draft"}, "terminal prefix"),
        (
            _valid_artifact() | {"field_principles": {"honest_verdict": "loose"}},
            "field_principles",
        ),
        (_valid_artifact() | {"methods_mapped": []}, "at least five"),
        (_valid_artifact() | {"methods_mapped": _valid_methods()[:4]}, "at least five"),
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
            "verified arxiv ID or canonical URL",
        ),
        (
            _valid_artifact()
            | {
                "methods_mapped": [
                    {
                        "name": "SEDD",
                        "arxiv_id_or_url": "2310.16834",
                        "url": "https://example.com/2310.16834",
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
                    {
                        "name": "TRM",
                        "arxiv_id_or_url": "2510.04871",
                    }
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
            | {
                "methods_mapped": [
                    _valid_methods()[0] | {"failure_mode": ""}
                ]
                + _valid_methods()[1:]
            },
            "non-empty string",
        ),
        (_valid_artifact() | {"flagged_for_v385": ""}, "flagged_for_v385"),
    ],
)
def test_validate_artifact_rejects_schema_violations_for_scenario_report_4152(
    bad_artifact: dict[str, object], message: str
) -> None:
    """SCENARIO-REPORT-4152: invalid mapping artifacts fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_validate_artifact_rejects_missing_extra_and_bad_method_rows() -> None:
    """SCENARIO-REPORT-4152: artifact and method fields are exact."""

    missing_artifact = _valid_artifact()
    missing_artifact.pop("methods_mapped")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing_artifact)

    extra_artifact = _valid_artifact()
    extra_artifact["inference_substrate"] = "aggregation_from_upstream_artifacts"
    with pytest.raises(ValueError, match="unexpected fields"):
        mod.validate_artifact(extra_artifact)

    malformed_methods = _valid_artifact() | {
        "methods_mapped": ["not-a-dict"] + _valid_methods()[1:]
    }
    with pytest.raises(ValueError, match="exactly"):
        mod.validate_artifact(malformed_methods)


def test_validate_markdown_note_checks_scenario_report_4152_sections() -> None:
    """SCENARIO-REPORT-4152: note maps sources to implementation work and risks."""

    note = f"""
    # SOTA ingestion recursive reasoner verifier energy guidance

    ## Current .385 verifier-guided-generation anchor
    arXiv:2510.04871, arXiv:2511.02886, arXiv:2402.06457,
    arXiv:2310.16834, arXiv:2105.05233, arXiv:2207.12598, and
    {DIFFUSIONGEMMA_URL} are the source anchors.

    ## TRM nano-trm recursive baseline gate
    arXiv:2510.04871.
    Implementation over nano-trm + Carnot-verifier stack: measure headroom.
    Pitfalls / where it fails: no headroom means no verifier claim.

    ## TTA-TRM adaptation-control arm
    arXiv:2511.02886.
    Implementation over nano-trm + Carnot-verifier stack: isolate adaptation.
    Pitfalls / where it fails: adaptation-only gain.

    ## V-STaR accepted/rejected trace selector
    arXiv:2402.06457.
    Implementation over nano-trm + Carnot-verifier stack: train selector.
    Pitfalls / where it fails: false-positive labels.

    ## SEDD discrete diffusion score-energy formalism
    arXiv:2310.16834.
    Implementation over nano-trm + Carnot-verifier stack: add energy during denoising.
    Pitfalls / where it fails: uncalibrated external energy.

    ## Classifier-guided diffusion energy precedent
    arXiv:2105.05233 and arXiv:2207.12598.
    Implementation over nano-trm + Carnot-verifier stack: guidance control.
    Pitfalls / where it fails: over-guidance.

    ## DiffusionGemma queued discrete-text substrate
    {DIFFUSIONGEMMA_URL}
    Implementation over nano-trm + Carnot-verifier stack: queued generator.
    Pitfalls / where it fails: generator substrate, not verifier evidence.

    ## Flagged for the .385 roadmap
    diffusiongemma_sedd_verifier_energy_guidance_probe_v385
    """

    mod.validate_markdown_note(note)


def test_validate_markdown_note_rejects_missing_roadmap_flag() -> None:
    """SCENARIO-REPORT-4152: note must close the discover-to-plan loop."""

    with pytest.raises(ValueError, match="Flagged for the .385 roadmap"):
        mod.validate_markdown_note(
            "## Current .385 verifier-guided-generation anchor\n"
            "## TRM nano-trm recursive baseline gate\n"
            "arXiv:2510.04871.\n"
            "Implementation over nano-trm + Carnot-verifier stack\n"
            "Pitfalls / where it fails\n"
        )


def test_validate_markdown_note_rejects_missing_verified_sources() -> None:
    """SCENARIO-REPORT-4152: every mapped method cites a verified source."""

    note = f"""
    ## Current .385 verifier-guided-generation anchor
    ## TRM nano-trm recursive baseline gate
    arXiv:2510.04871.
    Implementation over nano-trm + Carnot-verifier stack.
    Pitfalls / where it fails.
    ## TTA-TRM adaptation-control arm
    arXiv:2511.02886.
    Implementation over nano-trm + Carnot-verifier stack.
    Pitfalls / where it fails.
    ## V-STaR accepted/rejected trace selector
    arXiv:2402.06457.
    Implementation over nano-trm + Carnot-verifier stack.
    Pitfalls / where it fails.
    ## SEDD discrete diffusion score-energy formalism
    arXiv:2310.16834.
    Implementation over nano-trm + Carnot-verifier stack.
    Pitfalls / where it fails.
    ## Classifier-guided diffusion energy precedent
    arXiv:2105.05233 and arXiv:2207.12598.
    Implementation over nano-trm + Carnot-verifier stack.
    Pitfalls / where it fails.
    ## DiffusionGemma queued discrete-text substrate
    {DIFFUSIONGEMMA_URL}
    Implementation over nano-trm + Carnot-verifier stack.
    Pitfalls / where it fails.
    ## Flagged for the .385 roadmap
    diffusiongemma_sedd_verifier_energy_guidance_probe_v385
    """

    with pytest.raises(ValueError, match="verified source citations"):
        mod.validate_markdown_note(note.replace("arXiv:2310.16834", "SEDD"))


def test_write_outputs_updates_files_idempotently_for_req_report_4152(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-4152: writer emits note, artifact, and one studying section."""

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
    assert studying.count("2026-06-13 Exp 4152") == 1
    assert "Flagged for .385" in studying


def test_studying_section_update_handles_heading_layouts_for_req_report_4152() -> None:
    """REQ-REPORT-4152: studying update works before or between sections."""

    without_marker = "# Research Studying\n\n## Existing\nBody.\n"
    with_marker_and_next = mod._with_studying_section(without_marker)
    refreshed = mod._with_studying_section(with_marker_and_next)
    no_heading = mod._with_studying_section("# Research Studying\nOnly body.\n")
    marker_at_end = mod._with_studying_section(
        with_marker_and_next.split("\n## Existing")[0]
    )

    assert with_marker_and_next.index("2026-06-13 Exp 4152") < with_marker_and_next.index(
        "## Existing"
    )
    assert refreshed.count("2026-06-13 Exp 4152") == 1
    assert "## Existing\nBody." in refreshed
    assert no_heading.rstrip().endswith("guided-generation probe.")
    assert marker_at_end.count("2026-06-13 Exp 4152") == 1


def test_main_prints_terminal_verdict_for_req_report_4152(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-REPORT-4152: CLI entry point writes default repo-root outputs."""

    calls: dict[str, Path] = {}

    def fake_write_outputs(
        *, note_path: Path, artifact_path: Path, studying_path: Path
    ) -> dict[str, object]:
        calls["note_path"] = note_path
        calls["artifact_path"] = artifact_path
        calls["studying_path"] = studying_path
        return {
            "honest_verdict": (
                "complete: "
                "sota_ingestion_recursive_reasoner_verifier_energy_guidance_mapped"
            )
        }

    monkeypatch.setattr(mod, "write_outputs", fake_write_outputs)

    assert mod.main() == 0

    captured = capsys.readouterr()
    assert (
        captured.out.strip()
        == "complete: sota_ingestion_recursive_reasoner_verifier_energy_guidance_mapped"
    )
    assert calls["note_path"].as_posix().endswith(NOTE_PATH.as_posix())
    assert calls["artifact_path"].as_posix().endswith(ARTIFACT_PATH.as_posix())
    assert calls["studying_path"].as_posix().endswith(STUDYING_PATH.as_posix())


def test_deliverable_files_validate_against_req_report_4152() -> None:
    """REQ-REPORT-4152: committed note and JSON artifact satisfy the contract."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    note = NOTE_PATH.read_text(encoding="utf-8")
    studying = STUDYING_PATH.read_text(encoding="utf-8")

    mod.validate_artifact(artifact)
    mod.validate_markdown_note(note)
    assert len(artifact["methods_mapped"]) >= 5
    assert artifact["flagged_for_v385"] == (
        "diffusiongemma_sedd_verifier_energy_guidance_probe_v385"
    )
    assert (
        "2026-06-13 Exp 4152 - .385 recursive-reasoner/verifier energy-guidance "
        "SOTA ingestion ingested"
    ) in studying
    assert (
        "Flagged for .385: "
        "`diffusiongemma_sedd_verifier_energy_guidance_probe_v385`"
    ) in studying
