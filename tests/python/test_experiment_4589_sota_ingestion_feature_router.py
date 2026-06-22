"""Tests for REQ-REPORT-4589 / SCENARIO-REPORT-4589."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_4589_sota_ingestion_feature_router as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
ARTIFACT_PATH = Path("results/experiment_4589_sota_ingestion_feature_router.json")
WRAPPER_PATH = Path("results/experiment_4589_sota_ingestion_feature_router.py")
NOTE_PATH = Path("docs/research-notes/sota-ingestion-feature-router-423-2026-06-22.md")


def _valid_artifact() -> dict[str, object]:
    return mod.build_artifact()


def test_req_report_4589_spec_anchor_exists() -> None:
    """REQ-REPORT-4589: OpenSpec declares feature-router SOTA ingestion."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4589" in spec
    assert "SCENARIO-REPORT-4589" in spec
    assert ARTIFACT_PATH.as_posix() in spec
    assert WRAPPER_PATH.as_posix() in spec
    assert NOTE_PATH.as_posix() in spec
    assert "complete: sota_ingestion_feature_router_mapped" in spec
    assert "aggregation_from_upstream_artifacts" in spec
    assert "REQ-CAPSTONE-4580" in spec
    assert "REQ-CAPSTONE-4582" in spec
    assert "https://export.arxiv.org/api/query?search_query=all:test" in spec
    for source_id in mod.CITATIONS_VERIFIED:
        assert source_id in spec
    assert "scripts/research_conductor.py" in spec


def test_build_artifact_has_required_schema_fields_for_req_report_4589() -> None:
    """REQ-REPORT-4589: artifact exposes required fields and principles."""

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
    assert {method["target_track"] for method in artifact["methods_mapped"]} == {
        "feature_skill_routing",
        "env_adaptive_replay",
    }
    assert artifact["preconditions_checked"]["sweep_clusters_help_exit_0"] is True
    assert artifact["preconditions_checked"]["arxiv_api_reachable"] is True
    assert artifact["preconditions_checked"]["exp4580_artifact_read"] is True
    assert artifact["preconditions_checked"]["exp4582_artifact_read"] is True
    assert artifact["preconditions_checked"]["sweep_semscholar_rate_limited_queries"]
    assert artifact["preconditions_checked"]["deep_research_invoked"] is False
    assert artifact["preconditions_checked"]["live_llm_inference"] is False
    assert artifact["preconditions_checked"]["research_conductor_modified"] is False


@pytest.mark.parametrize(
    ("bad_artifact", "message"),
    [
        (_valid_artifact() | {"honest_verdict": "blocked_network"}, "terminal prefix"),
        (_valid_artifact() | {"inference_substrate": "live_llm_inference"}, "aggregation"),
        (_valid_artifact() | {"field_principles": {"honest_verdict": "loose"}}, "field_principles"),
        (_valid_artifact() | {"random_seed": "4589"}, "random_seed"),
        (
            _valid_artifact()
            | {
                "citations_verified": {
                    k: v for k, v in mod.CITATIONS_VERIFIED.items() if k != "2603.22455"
                }
            },
            "citations_verified",
        ),
        (_valid_artifact() | {"methods_mapped": mod.DEFAULT_METHODS_MAPPED[:2]}, "three to five"),
        (
            _valid_artifact()
            | {
                "methods_mapped": [
                    mod.DEFAULT_METHODS_MAPPED[0]
                    | {"takes_over_current_a1_a3_mechanisms": "generic router"},
                    *mod.DEFAULT_METHODS_MAPPED[1:],
                ]
            },
            "Exp 4580 or Exp 4582",
        ),
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
                    mod.DEFAULT_METHODS_MAPPED[0] | {"target_track": "old_track"},
                    *mod.DEFAULT_METHODS_MAPPED[1:],
                ]
            },
            "target_track",
        ),
        (
            _valid_artifact() | {"flagged_for_next_roadmap": "flagged_for_v423: old plan"},
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
                | {"ops_docs_modified": True}
            },
            "ops docs",
        ),
        (_valid_artifact() | {"preconditions_checked": []}, "preconditions_checked"),
        (
            _valid_artifact() | {"research_note_path": "docs/research-notes/wrong.md"},
            "research_note_path",
        ),
    ],
)
def test_validate_artifact_rejects_schema_violations_for_req_report_4589(
    bad_artifact: dict[str, object], message: str
) -> None:
    """REQ-REPORT-4589: invalid feature-router artifacts fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_validate_artifact_rejects_missing_and_extra_fields() -> None:
    """REQ-REPORT-4589: top-level artifact fields are exact."""

    missing = _valid_artifact()
    missing.pop("citations_verified")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing)

    extra = _valid_artifact() | {"offline_reproduced": False}
    with pytest.raises(ValueError, match="unexpected fields"):
        mod.validate_artifact(extra)


def test_note_json_round_trips_and_preserves_mapping_for_scenario_4589() -> None:
    """SCENARIO-REPORT-4589: markdown note contains the validated JSON artifact."""

    artifact = mod.artifact_from_note(mod.RESEARCH_NOTE)

    assert artifact == _valid_artifact()
    mod.validate_research_note(mod.RESEARCH_NOTE)
    assert "Exp 4580" in mod.RESEARCH_NOTE
    assert "Exp 4582" in mod.RESEARCH_NOTE
    assert "SkillRouter" in mod.RESEARCH_NOTE
    assert "SkillGraph" in mod.RESEARCH_NOTE
    assert "SkillComposer" in mod.RESEARCH_NOTE
    assert "Skill-Pro" in mod.RESEARCH_NOTE
    assert "SkillRL" in mod.RESEARCH_NOTE
    assert "Graph-Based Exploration" in mod.RESEARCH_NOTE
    assert "env-adaptive replay" in mod.RESEARCH_NOTE
    assert "flagged_for_v424" in mod.RESEARCH_NOTE

    with pytest.raises(ValueError, match="machine-readable JSON"):
        mod.artifact_from_note(mod.RESEARCH_NOTE.replace("```json", "```text"))
    with pytest.raises(ValueError, match="verified source citations"):
        mod.validate_research_note(mod.RESEARCH_NOTE.replace("arXiv:2603.22455", "SkillRouter"))
    with pytest.raises(ValueError, match="required phrase"):
        mod.validate_research_note(mod.RESEARCH_NOTE.replace("No training", "No run"))


def test_write_outputs_emits_stable_files_for_req_report_4589(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4589: writer emits idempotent JSON, note, and queue update."""

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
    assert "INGESTED into `docs/research-notes/sota-ingestion-feature-router-423-2026-06-22.md`" in studying
    assert "flagged_for_v424" in studying


def test_main_and_wrapper_emit_terminal_verdict_for_req_report_4589(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-4589: module and wrapper write default outputs."""

    monkeypatch.setenv("CARNOT_EXP4589_ROOT", str(tmp_path))
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
