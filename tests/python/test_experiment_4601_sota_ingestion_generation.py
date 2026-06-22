"""Tests for REQ-REPORT-4601 / SCENARIO-REPORT-4601."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_4601_sota_ingestion_generation as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ARTIFACT_PATH = Path("results/experiment_4601_sota_ingestion_generation.json")
WRAPPER_PATH = Path("results/experiment_4601_sota_ingestion_generation.py")
NOTE_PATH = Path(
    "docs/research-notes/sota-ingestion-generation-world-model-424-2026-06-22.md"
)
STUDYING_PATH = Path("research-studying.md")


def _valid_artifact() -> dict[str, object]:
    return mod.build_artifact()


def test_req_report_4601_spec_anchor_exists() -> None:
    """REQ-REPORT-4601: OpenSpec declares generation SOTA ingestion."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4601" in spec
    assert "SCENARIO-REPORT-4601" in spec
    assert ARTIFACT_PATH.as_posix() in spec
    assert WRAPPER_PATH.as_posix() in spec
    assert NOTE_PATH.as_posix() in spec
    assert "complete: sota_ingestion_generation_mapped" in spec
    assert "aggregation_from_upstream_artifacts" in spec
    assert "results/experiment_4592_generation_completeness_wiring.json" in spec
    assert "results/experiment_4594_goal_energy_generation_prior.json" in spec
    assert "blocked_network" in spec
    assert "https://export.arxiv.org/api/query?search_query=all:test" in spec
    assert "flagged_for_v425" in spec
    assert "scripts/research_conductor.py" in spec
    for source_id in mod.REQUIRED_VERIFIED_SOURCE_IDS:
        assert source_id in spec


def test_build_artifact_has_required_schema_fields_for_req_report_4601() -> None:
    """REQ-REPORT-4601: artifact exposes required fields and principles."""

    artifact = _valid_artifact()

    assert set(artifact) == mod.REQUIRED_ARTIFACT_FIELDS
    assert set(artifact["field_principles"]) == mod.REQUIRED_ARTIFACT_FIELDS
    assert artifact["honest_verdict"] == mod.DEFAULT_HONEST_VERDICT
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["methods_mapped"] == mod.DEFAULT_METHODS_MAPPED
    assert artifact["citations_verified"] == mod.CITATIONS_VERIFIED
    assert artifact["flagged_for_next_roadmap"] == mod.FLAGGED_FOR_NEXT_ROADMAP
    assert artifact["research_note_path"] == NOTE_PATH.as_posix()
    assert artifact["random_seed"] == mod.DEFAULT_RANDOM_SEED
    assert len(artifact["methods_mapped"]) == 5
    assert artifact["preconditions_checked"]["sweep_clusters_help_exit_0"] is True
    assert artifact["preconditions_checked"]["arxiv_api_reachable"] is True
    assert artifact["preconditions_checked"]["exp4592_artifact_read"] is True
    assert artifact["preconditions_checked"]["exp4594_artifact_read"] is True
    assert artifact["preconditions_checked"]["sweep_semscholar_arxiv_ids"] == []
    assert artifact["preconditions_checked"]["sweep_semscholar_rate_limited_queries"]
    assert artifact["preconditions_checked"]["sweep_semscholar_failed_queries"]
    assert artifact["preconditions_checked"]["deep_research_invoked"] is False
    assert artifact["preconditions_checked"]["live_llm_inference"] is False
    assert artifact["preconditions_checked"]["training_launched"] is False
    assert artifact["preconditions_checked"]["ops_docs_modified"] is False
    assert artifact["preconditions_checked"]["research_conductor_modified"] is False


@pytest.mark.parametrize(
    ("bad_artifact", "message"),
    [
        (_valid_artifact() | {"honest_verdict": "draft"}, "terminal prefix"),
        (_valid_artifact() | {"inference_substrate": "live_llm"}, "aggregation"),
        (_valid_artifact() | {"field_principles": {"honest_verdict": "loose"}}, "field_principles"),
        (_valid_artifact() | {"random_seed": "4601"}, "random_seed"),
        (
            _valid_artifact()
            | {
                "citations_verified": {
                    k: v for k, v in mod.CITATIONS_VERIFIED.items() if k != "2605.05138"
                }
            },
            "citations_verified",
        ),
        (
            _valid_artifact()
            | {
                "citations_verified": mod.CITATIONS_VERIFIED
                | {"2605.05138": mod.CITATIONS_VERIFIED["2605.05138"] | {"http_status": 404}}
            },
            "http_status",
        ),
        (_valid_artifact() | {"methods_mapped": mod.DEFAULT_METHODS_MAPPED[:2]}, "three to five"),
        (
            _valid_artifact()
            | {
                "methods_mapped": [
                    mod.DEFAULT_METHODS_MAPPED[0] | {"source_ids": ["9999.99999"]},
                    *mod.DEFAULT_METHODS_MAPPED[1:],
                ]
            },
            "verified citations",
        ),
        (
            _valid_artifact()
            | {
                "methods_mapped": [
                    mod.DEFAULT_METHODS_MAPPED[0] | {"generation_track": "ranking"},
                    *mod.DEFAULT_METHODS_MAPPED[1:],
                ]
            },
            "generation_track",
        ),
        (
            _valid_artifact()
            | {
                "methods_mapped": [
                    mod.DEFAULT_METHODS_MAPPED[0]
                    | {"takes_over_current_a1_a3_mechanisms": "generic generator"},
                    *mod.DEFAULT_METHODS_MAPPED[1:],
                ]
            },
            "Exp 4592 or Exp 4594",
        ),
        (
            _valid_artifact()
            | {
                "methods_mapped": [
                    mod.DEFAULT_METHODS_MAPPED[0] | {"fails_when": ""},
                    *mod.DEFAULT_METHODS_MAPPED[1:],
                ]
            },
            "non-empty string",
        ),
        (
            _valid_artifact()
            | {"flagged_for_next_roadmap": "flagged_for_v424: old plan"},
            "flagged_for_next_roadmap",
        ),
        (
            _valid_artifact()
            | {
                "preconditions_checked": dict(mod.DEFAULT_PRECONDITIONS_CHECKED)
                | {"sweep_clusters_help_exit_0": False}
            },
            "sweep_clusters.py --help",
        ),
        (
            _valid_artifact()
            | {
                "preconditions_checked": dict(mod.DEFAULT_PRECONDITIONS_CHECKED)
                | {"arxiv_api_reachable": False}
            },
            "arXiv API",
        ),
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
                | {"ops_docs_modified": True}
            },
            "ops docs",
        ),
        (
            _valid_artifact()
            | {
                "preconditions_checked": dict(mod.DEFAULT_PRECONDITIONS_CHECKED)
                | {"research_conductor_modified": True}
            },
            "research_conductor",
        ),
        (_valid_artifact() | {"preconditions_checked": []}, "preconditions_checked"),
        (
            _valid_artifact()
            | {"research_note_path": "docs/research-notes/wrong.md"},
            "research_note_path",
        ),
    ],
)
def test_validate_artifact_rejects_schema_violations_for_req_report_4601(
    bad_artifact: dict[str, object], message: str
) -> None:
    """REQ-REPORT-4601: invalid generation-ingestion artifacts fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_validate_artifact_rejects_missing_extra_and_malformed_methods() -> None:
    """REQ-REPORT-4601: top-level and method fields are exact."""

    missing = _valid_artifact()
    missing.pop("methods_mapped")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing)

    extra = _valid_artifact() | {"offline_reproduced": False}
    with pytest.raises(ValueError, match="unexpected fields"):
        mod.validate_artifact(extra)

    malformed = _valid_artifact() | {
        "methods_mapped": ["not-a-dict", *mod.DEFAULT_METHODS_MAPPED[1:]]
    }
    with pytest.raises(ValueError, match="exactly"):
        mod.validate_artifact(malformed)


def test_note_json_round_trips_and_preserves_mapping_for_scenario_4601() -> None:
    """SCENARIO-REPORT-4601: markdown note embeds the validated JSON artifact."""

    artifact = mod.artifact_from_note(mod.RESEARCH_NOTE)

    assert artifact == _valid_artifact()
    mod.validate_research_note(mod.RESEARCH_NOTE)
    assert "Exp 4592" in mod.RESEARCH_NOTE
    assert "Exp 4594" in mod.RESEARCH_NOTE
    assert "Sensi" in mod.RESEARCH_NOTE
    assert "perceptual grounding" in mod.RESEARCH_NOTE
    assert "winner_generated=2/25" in mod.RESEARCH_NOTE
    assert "goal_energy_prior_no_value_honest_null_gap_sharpened" in mod.RESEARCH_NOTE
    assert "flagged_for_v425" in mod.RESEARCH_NOTE

    with pytest.raises(ValueError, match="machine-readable JSON"):
        mod.artifact_from_note(mod.RESEARCH_NOTE.replace("```json", "```text"))
    json_block_end = mod.RESEARCH_NOTE.find("```\n\n## Fresh-pass provenance") + 3
    note_without_prose_citation = (
        mod.RESEARCH_NOTE[:json_block_end]
        + mod.RESEARCH_NOTE[json_block_end:].replace("arXiv:2605.05138", "EWM")
    )
    with pytest.raises(ValueError, match="verified source citations"):
        mod.validate_research_note(note_without_prose_citation)
    with pytest.raises(ValueError, match="required phrase"):
        mod.validate_research_note(mod.RESEARCH_NOTE.replace("No training", "No run"))
    with pytest.raises(ValueError, match="perceptual-grounding"):
        mod.validate_research_note(mod.RESEARCH_NOTE.replace("perceptual-grounding wall", "grid issue"))


def test_write_outputs_emits_stable_files_for_req_report_4601(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4601: writer emits idempotent JSON, note, and queue update."""

    artifact_path = tmp_path / ARTIFACT_PATH
    note_path = tmp_path / NOTE_PATH
    studying_path = tmp_path / "research-studying.md"
    studying_path.write_text("# Research Studying\n\nExisting queue.\n", encoding="utf-8")

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

    studying = studying_path.read_text(encoding="utf-8")
    assert second_artifact == artifact
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert mod.artifact_from_note(note_path.read_text(encoding="utf-8")) == artifact
    assert studying.count(mod.STUDYING_SECTION_START) == 1
    assert "INGESTED into `docs/research-notes/sota-ingestion-generation-world-model-424-2026-06-22.md`" in studying
    assert "flagged_for_v425" in studying


def test_main_and_wrapper_emit_terminal_verdict_for_req_report_4601(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-4601: module and wrapper write default outputs."""

    monkeypatch.setenv("CARNOT_EXP4601_ROOT", str(tmp_path))
    (tmp_path / "research-studying.md").write_text("# Research Studying\n", encoding="utf-8")

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


def test_deliverable_files_validate_against_req_report_4601() -> None:
    """REQ-REPORT-4601: committed note and JSON artifact satisfy the contract."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    note = NOTE_PATH.read_text(encoding="utf-8")
    studying = STUDYING_PATH.read_text(encoding="utf-8")

    mod.validate_artifact(artifact)
    mod.validate_research_note(note)
    assert mod.artifact_from_note(note) == artifact
    assert artifact["flagged_for_next_roadmap"] == mod.FLAGGED_FOR_NEXT_ROADMAP
    assert "flagged_for_v425" in studying
    assert "Exp 4601 - .424 generation SOTA ingestion - INGESTED" in studying
