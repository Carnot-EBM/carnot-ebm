"""Tests for REQ-REPORT-4452 / SCENARIO-REPORT-4452."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_4452_sota_ingestion_411 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ARTIFACT_PATH = Path("results/experiment_4452_sota_ingestion_411.json")
WRAPPER_PATH = Path("results/experiment_4452_sota_ingestion_411.py")
NOTE_PATH = Path("docs/research-notes/sota-ingestion-411-2026-06-19.md")


def _valid_artifact() -> dict[str, object]:
    return mod.build_artifact()


def test_req_report_4452_spec_anchor_exists() -> None:
    """REQ-REPORT-4452: OpenSpec declares the .411 SOTA ingestion."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4452" in spec
    assert "SCENARIO-REPORT-4452" in spec
    assert ARTIFACT_PATH.as_posix() in spec
    assert WRAPPER_PATH.as_posix() in spec
    assert NOTE_PATH.as_posix() in spec
    assert "flagged_for_v412" in spec
    assert "complete: sota_ingestion_411_mapped_for_v412" in spec
    assert "arXiv:2310.19791" in spec
    assert "aggregation_from_upstream_artifacts" in spec


def test_build_artifact_has_required_fields_for_req_report_4452() -> None:
    """REQ-REPORT-4452: artifact exposes the required principle fields."""

    artifact = _valid_artifact()

    assert set(artifact) == mod.REQUIRED_ARTIFACT_FIELDS
    assert artifact["honest_verdict"] == mod.DEFAULT_HONEST_VERDICT
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["methods"] == mod.DEFAULT_METHODS
    assert artifact["flagged_for_v412"] == mod.DEFAULT_FLAGGED_FOR_V412
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert artifact["research_note_path"] == NOTE_PATH.as_posix()
    assert "SOTA->experiment" in str(artifact["sota_to_experiment_mapping_note"])


@pytest.mark.parametrize(
    ("bad_artifact", "message"),
    [
        (_valid_artifact() | {"honest_verdict": "blocked_network"}, "terminal prefix"),
        (_valid_artifact() | {"inference_substrate": "cpu_solve"}, "aggregation"),
        (_valid_artifact() | {"field_principles": {"honest_verdict": "loose"}}, "field_principles"),
        (_valid_artifact() | {"methods": mod.DEFAULT_METHODS[:4]}, "five to eight"),
        (
            _valid_artifact()
            | {
                "methods": [
                    mod.DEFAULT_METHODS[0] | {"arxiv_id": "9999.99999"},
                    *mod.DEFAULT_METHODS[1:],
                ]
            },
            "verified arXiv",
        ),
        (
            _valid_artifact() | {"methods": [mod.DEFAULT_METHODS[0], *mod.DEFAULT_METHODS[:-1]]},
            "duplicate",
        ),
        (
            _valid_artifact()
            | {
                "methods": [
                    mod.DEFAULT_METHODS[0] | {"what_it_takes_over_our_stack": ""},
                    *mod.DEFAULT_METHODS[1:],
                ]
            },
            "non-empty string",
        ),
        (_valid_artifact() | {"methods": ["not-a-dict", *mod.DEFAULT_METHODS[1:]]}, "method"),
        (_valid_artifact() | {"flagged_for_v412": "unverified"}, "flagged_for_v412"),
        (_valid_artifact() | {"random_seed": "4452"}, "random_seed"),
        (
            _valid_artifact()
            | {
                "preconditions_checked": dict(mod.DEFAULT_PRECONDITIONS_CHECKED)
                | {"deep_research_invoked": True}
            },
            "deep-research",
        ),
        (
            _valid_artifact()
            | {
                "preconditions_checked": dict(mod.DEFAULT_PRECONDITIONS_CHECKED)
                | {"leaderboard_submission": True}
            },
            "leaderboard",
        ),
        (
            _valid_artifact()
            | {
                "preconditions_checked": dict(mod.DEFAULT_PRECONDITIONS_CHECKED)
                | {"live_solve_claim": True}
            },
            "live solve",
        ),
        (
            _valid_artifact()
            | {
                "preconditions_checked": dict(mod.DEFAULT_PRECONDITIONS_CHECKED)
                | {"cpu_only": False}
            },
            "CPU",
        ),
        (_valid_artifact() | {"preconditions_checked": []}, "preconditions_checked"),
        (_valid_artifact() | {"sota_to_experiment_mapping_note": "short"}, "mapping note"),
        (
            _valid_artifact() | {"research_note_path": "docs/research-notes/wrong.md"},
            "research_note_path",
        ),
    ],
)
def test_validate_artifact_rejects_schema_violations_for_req_report_4452(
    bad_artifact: dict[str, object], message: str
) -> None:
    """REQ-REPORT-4452: invalid SOTA artifacts fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_validate_artifact_rejects_missing_and_extra_fields() -> None:
    """REQ-REPORT-4452: top-level artifact fields are exact."""

    missing = _valid_artifact()
    missing.pop("methods")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing)

    extra = _valid_artifact() | {"url": "https://example.com"}
    with pytest.raises(ValueError, match="unexpected fields"):
        mod.validate_artifact(extra)


def test_validate_notes_check_citations_and_v412_handoff() -> None:
    """SCENARIO-REPORT-4452: notes keep verified citations and the .412 flag."""

    mod.validate_research_note(mod.RESEARCH_NOTE)
    mod.validate_studying_section(mod.STUDYING_SECTION)

    with pytest.raises(ValueError, match="verified source citations"):
        mod.validate_research_note(mod.RESEARCH_NOTE.replace("arXiv:2310.19791", "LILO"))
    with pytest.raises(ValueError, match="required phrase"):
        mod.validate_research_note(mod.RESEARCH_NOTE.replace("No leaderboard submission", "No LB"))
    with pytest.raises(ValueError, match="flagged_for_v412"):
        mod.validate_studying_section(mod.STUDYING_SECTION.replace("flagged_for_v412", "next flag"))
    with pytest.raises(ValueError, match="verified source citations"):
        mod.validate_studying_section(mod.STUDYING_SECTION.replace("arXiv:2606.12316", "Loop-OWM"))


def test_write_outputs_updates_files_idempotently_for_req_report_4452(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4452: writer emits artifact, research note, and studying entry."""

    artifact_path = tmp_path / ARTIFACT_PATH
    note_path = tmp_path / NOTE_PATH
    studying_path = tmp_path / "research-studying.md"
    studying_path.write_text("# Research Studying\n\n## Existing\nBody.\n", encoding="utf-8")

    artifact = mod.write_outputs(
        artifact_path=artifact_path,
        note_path=note_path,
        studying_path=studying_path,
    )
    second_artifact = mod.write_outputs(
        artifact_path=artifact_path,
        note_path=note_path,
        studying_path=studying_path,
    )

    mod.validate_artifact(artifact)
    assert second_artifact == artifact
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert "SOTA->experiment" in note_path.read_text(encoding="utf-8")
    studying = studying_path.read_text(encoding="utf-8")
    assert studying.count("2026-06-19 Exp 4452") == 1
    assert "flagged_for_v412" in studying


def test_section_updates_handle_heading_layouts_for_req_report_4452() -> None:
    """REQ-REPORT-4452: studying updates work before or between sections."""

    without_marker = "# Doc\n\n## Existing\nBody.\n"
    studying_once = mod._with_studying_section(without_marker)
    starts_with_heading = mod._with_studying_section("## Existing\nBody.\n")
    studying_refreshed = mod._with_studying_section(studying_once)
    marker_at_end = mod._with_studying_section(studying_once.split("\n## Existing")[0])
    no_heading = mod._with_studying_section("# Doc\nOnly body.\n")

    assert studying_once.index("2026-06-19 Exp 4452") < studying_once.index("## Existing")
    assert starts_with_heading.startswith("## 2026-06-19 Exp 4452")
    assert studying_refreshed.count("2026-06-19 Exp 4452") == 1
    assert marker_at_end.count("2026-06-19 Exp 4452") == 1
    assert "## Existing\nBody." in studying_refreshed
    assert "LILO" in no_heading


def test_main_and_wrapper_emit_terminal_verdict_for_req_report_4452(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-4452: module and wrapper write default outputs."""

    monkeypatch.setenv("CARNOT_EXP4452_ROOT", str(tmp_path))

    with pytest.raises(SystemExit) as module_exit:
        runpy.run_path(str(Path(mod.__file__)), run_name="__main__")
    assert module_exit.value.code == 0
    assert capsys.readouterr().out.strip() == mod.DEFAULT_HONEST_VERDICT
    assert (tmp_path / ARTIFACT_PATH).exists()
    assert (tmp_path / NOTE_PATH).exists()

    with pytest.raises(SystemExit) as wrapper_exit:
        runpy.run_path(str(WRAPPER_PATH), run_name="__main__")
    assert wrapper_exit.value.code == 0
    assert capsys.readouterr().out.strip() == mod.DEFAULT_HONEST_VERDICT
    assert (tmp_path / ARTIFACT_PATH).exists()
