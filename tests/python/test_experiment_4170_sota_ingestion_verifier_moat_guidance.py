"""Tests for REQ-REPORT-4170 / SCENARIO-REPORT-4170."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_4170_sota_ingestion_verifier_moat_guidance as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
NOTE_PATH = Path(
    "docs/research-notes/"
    "sota-ingestion-verifier-moat-guidance-v387-2026-06-13.md"
)
ARTIFACT_PATH = Path(
    "results/experiment_4170_sota_ingestion_verifier_moat_guidance.json"
)
STUDYING_PATH = Path("research-studying.md")


def _valid_methods() -> list[dict[str, str]]:
    return [dict(method) for method in mod.DEFAULT_METHODS_MAPPED]


def _valid_artifact() -> dict[str, object]:
    return mod.build_artifact(
        methods_mapped=_valid_methods(),
        flagged_for_v387=mod.DEFAULT_FLAGGED_FOR_V387,
    )


def test_req_report_4170_spec_anchor_exists() -> None:
    """REQ-REPORT-4170: OpenSpec declares the .387 ingestion artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4170" in spec
    assert "SCENARIO-REPORT-4170" in spec
    assert NOTE_PATH.as_posix() in spec
    assert ARTIFACT_PATH.as_posix() in spec
    assert "flagged_for_v387" in spec
    for arxiv_id in mod.VERIFIED_ARXIV_IDS:
        assert f"arXiv:{arxiv_id}" in spec


def test_build_artifact_has_required_schema_fields_for_req_report_4170() -> None:
    """REQ-REPORT-4170: artifact exposes the required principle fields."""

    artifact = _valid_artifact()

    assert artifact == {
        "honest_verdict": mod.DEFAULT_HONEST_VERDICT,
        "methods_mapped": _valid_methods(),
        "flagged_for_v387": mod.DEFAULT_FLAGGED_FOR_V387,
        "field_principles": {
            "honest_verdict": (
                "Terminal-prefixed. Records ingestion completed with verifiable citations."
            ),
            "methods_mapped": (
                "Each method/source MUST carry a real arXiv ID/URL; an ingestion "
                "note without verifiable citations is treated as fabrication."
            ),
            "flagged_for_v387": (
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
        (_valid_artifact() | {"methods_mapped": _valid_methods()[:2]}, "at least three"),
        (
            _valid_artifact()
            | {
                "methods_mapped": [
                    {
                        "name": "fake",
                        "arxiv_id_or_url": "9999.99999",
                        "url": "https://arxiv.org/abs/9999.99999",
                        "carnot_verifier_implication": "fake",
                        "queued_diffusiongemma_implication": "fake",
                        "experiment_mapping": "fake",
                    }
                ]
                + _valid_methods()[1:]
            },
            "verified arxiv ID",
        ),
        (
            _valid_artifact()
            | {
                "methods_mapped": [
                    _valid_methods()[0] | {"url": "https://example.com/2510.04871"}
                ]
                + _valid_methods()[1:]
            },
            "url",
        ),
        (
            _valid_artifact()
            | {"methods_mapped": [{"name": "TRM", "arxiv_id_or_url": "2510.04871"}] + _valid_methods()[1:]},
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
                    _valid_methods()[0] | {"experiment_mapping": ""}
                ]
                + _valid_methods()[1:]
            },
            "non-empty string",
        ),
        (_valid_artifact() | {"flagged_for_v387": ""}, "flagged_for_v387"),
    ],
)
def test_validate_artifact_rejects_schema_violations_for_scenario_report_4170(
    bad_artifact: dict[str, object], message: str
) -> None:
    """SCENARIO-REPORT-4170: invalid mapping artifacts fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_validate_artifact_rejects_missing_extra_and_bad_method_rows() -> None:
    """SCENARIO-REPORT-4170: artifact and method fields are exact."""

    missing_artifact = _valid_artifact()
    missing_artifact.pop("methods_mapped")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing_artifact)

    extra_artifact = _valid_artifact()
    extra_artifact["inference_substrate"] = "manual_ingestion"
    with pytest.raises(ValueError, match="unexpected fields"):
        mod.validate_artifact(extra_artifact)

    malformed_methods = _valid_artifact() | {
        "methods_mapped": ["not-a-dict"] + _valid_methods()[1:]
    }
    with pytest.raises(ValueError, match="exactly"):
        mod.validate_artifact(malformed_methods)


def test_validate_markdown_note_checks_scenario_report_4170_sections() -> None:
    """SCENARIO-REPORT-4170: note maps each source to the required axes."""

    note = """
    ## Fresh-pass provenance
    reliable-channel sweeps and WebSearch/WebFetch verified the sources.
    ## SOTA -> experiment mapping
    ## TRM nano-trm baseline and headroom gate
    arXiv:2510.04871. Carnot-verifier implication.
    Queued DiffusionGemma implication. Experiment mapping.
    ## TTA-TRM adaptation-control arm
    arXiv:2511.02886. Carnot-verifier implication.
    Queued DiffusionGemma implication. Experiment mapping.
    ## V-STaR accepted/rejected trace selector
    arXiv:2402.06457. Carnot-verifier implication.
    Queued DiffusionGemma implication. Experiment mapping.
    ## SEDD discrete diffusion score-energy formalism
    arXiv:2310.16834. Carnot-verifier implication.
    Queued DiffusionGemma implication. Experiment mapping.
    ## Classifier-guided diffusion external-energy precedent
    arXiv:2105.05233. Carnot-verifier implication.
    Queued DiffusionGemma implication. Experiment mapping.
    ## Classifier-free guidance control
    arXiv:2207.12598. Carnot-verifier implication.
    Queued DiffusionGemma implication. Experiment mapping.
    ## EntRGi entropy-aware reward guidance
    arXiv:2602.05000. Carnot-verifier implication.
    Queued DiffusionGemma implication. Experiment mapping.
    ## EDLM sequence-level diffusion energy comparator
    arXiv:2410.21357. Carnot-verifier implication.
    Queued DiffusionGemma implication. Experiment mapping.
    ## Flagged for .387
    vstar_rejected_trace_selector_headroom_gate_before_diffusiongemma_v387
    """

    mod.validate_markdown_note(note)


def test_validate_markdown_note_rejects_missing_sources_or_flag() -> None:
    """SCENARIO-REPORT-4170: note must cite each source and close the loop."""

    with pytest.raises(ValueError, match="Flagged for .387"):
        mod.validate_markdown_note("## Fresh-pass provenance\narXiv:2510.04871\n")

    with pytest.raises(ValueError, match="verified source citations"):
        mod.validate_markdown_note(
            mod.NOTE_MARKDOWN.replace("arXiv:2602.05000", "EntRGi")
        )


def test_write_outputs_updates_files_idempotently_for_req_report_4170(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-4170: writer emits note, artifact, and studying entry."""

    note_path = tmp_path / "note.md"
    artifact_path = tmp_path / "artifact.json"
    studying_path = tmp_path / "research-studying.md"
    studying_path.write_text("# Research Studying\n\n## Existing\nBody.\n", encoding="utf-8")

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
    assert studying.count("2026-06-13 Exp 4170") == 1
    assert "flagged_for_v387" in studying
    assert "Flagged for .387" in studying


def test_section_updates_handle_heading_layouts_for_req_report_4170() -> None:
    """REQ-REPORT-4170: markdown updates work before or between sections."""

    without_marker = "# Doc\n\n## Existing\nBody.\n"
    studying_once = mod._with_studying_section(without_marker)
    starts_with_heading = mod._with_studying_section("## Existing\nBody.\n")
    studying_refreshed = mod._with_studying_section(studying_once)
    marker_at_end = mod._with_studying_section(
        studying_once.split("\n## Existing")[0]
    )
    no_heading = mod._with_studying_section("# Doc\nOnly body.\n")

    assert studying_once.index("2026-06-13 Exp 4170") < studying_once.index("## Existing")
    assert starts_with_heading.startswith("## 2026-06-13 Exp 4170")
    assert studying_refreshed.count("2026-06-13 Exp 4170") == 1
    assert marker_at_end.count("2026-06-13 Exp 4170") == 1
    assert "## Existing\nBody." in studying_refreshed
    assert no_heading.rstrip().endswith(
        "unless the verifier discrimination gate flips positive."
    )


def test_main_prints_terminal_verdict_for_req_report_4170(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-REPORT-4170: CLI entry point writes default repo-root outputs."""

    calls: dict[str, Path] = {}

    def fake_write_outputs(
        *,
        note_path: Path,
        artifact_path: Path,
        studying_path: Path,
    ) -> dict[str, object]:
        calls["note_path"] = note_path
        calls["artifact_path"] = artifact_path
        calls["studying_path"] = studying_path
        return {"honest_verdict": mod.DEFAULT_HONEST_VERDICT}

    monkeypatch.setattr(mod, "write_outputs", fake_write_outputs)

    assert mod.main() == 0

    captured = capsys.readouterr()
    assert captured.out.strip() == mod.DEFAULT_HONEST_VERDICT
    assert calls["note_path"].as_posix().endswith(NOTE_PATH.as_posix())
    assert calls["artifact_path"].as_posix().endswith(ARTIFACT_PATH.as_posix())
    assert calls["studying_path"].as_posix().endswith(STUDYING_PATH.as_posix())


def test_module_main_guard_exits_zero_for_req_report_4170(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-4170: direct script execution exits after writing outputs."""

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(Path(mod.__file__)), run_name="__main__")

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == mod.DEFAULT_HONEST_VERDICT


def test_deliverable_files_validate_against_req_report_4170() -> None:
    """REQ-REPORT-4170: committed note and JSON artifact satisfy the contract."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    note = NOTE_PATH.read_text(encoding="utf-8")
    studying = STUDYING_PATH.read_text(encoding="utf-8")

    mod.validate_artifact(artifact)
    mod.validate_markdown_note(note)
    assert len(artifact["methods_mapped"]) >= 3
    assert artifact["flagged_for_v387"] == mod.DEFAULT_FLAGGED_FOR_V387
    assert (
        "2026-06-13 Exp 4170 - .387 verifier-moat guidance SOTA ingestion ingested"
        in studying
    )
    assert (
        "Flagged for .387: "
        "`vstar_rejected_trace_selector_headroom_gate_before_diffusiongemma_v387`"
    ) in studying
