"""Tests for Exp 5053 .465 SOTA ingestion.

Spec refs: REQ-REPORT-5053, SCENARIO-REPORT-5053,
SCENARIO-REPORT-5053-DUPLICATE-FILTER.
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

from carnot import experiment_5053_sota_ingestion_v465 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"
ARTIFACT_PATH = REPO / mod.RESULT_RELATIVE_PATH
REFERENCES_PATH = REPO / mod.REFERENCES_RELATIVE_PATH


def _artifact() -> dict[str, object]:
    return mod.build_artifact(research_references_updated=True)


def _spec_section() -> str:
    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-REPORT-5053")
    end = spec.index("### REQ-REPORT-4873", start)
    return spec[start:end]


def test_req_report_5053_spec_declares_artifact_and_duplicate_contract() -> None:
    """REQ-REPORT-5053: OpenSpec declares the .465 ingestion contract."""

    section = _spec_section()

    assert "REQ-REPORT-5053" in section
    assert "SCENARIO-REPORT-5053" in section
    assert "SCENARIO-REPORT-5053-DUPLICATE-FILTER" in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert mod.REFERENCES_RELATIVE_PATH in section
    assert mod.HONEST_VERDICT in section
    assert "scripts/research_conductor.py" in section
    assert "ops/changelog.md" in section
    assert "OpenReview" in section
    assert "Hugging Face Papers" in section
    assert "EBT/ARM-EBM citation trails" in section
    for field, principle in mod.REQUIRED_USER_FIELD_PRINCIPLES.items():
        assert field in section
        assert principle["principle"] in section
    for duplicate_id in sorted(mod.REQUIRED_DUPLICATE_IDS_IN_SPEC):
        assert duplicate_id in section


def test_req_report_5053_artifact_maps_actionable_nonduplicate_sources() -> None:
    """REQ-REPORT-5053: artifact maps checked sources to concrete .465 hooks."""

    artifact = _artifact()

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["honest_verdict"] == mod.HONEST_VERDICT
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["research_references_updated"] is True
    assert artifact["n_sources_checked"] == len(mod.SOURCES_CHECKED)
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES

    selected_ids = [source["source_id"] for source in artifact["selected_sources"]]
    assert selected_ids == mod.SELECTED_SOURCE_IDS
    assert not set(selected_ids).intersection(mod.ALREADY_INGESTED_SOURCE_IDS)
    assert len(selected_ids) >= 5

    reference_ids = [entry["source_id"] for entry in artifact["references_added"]]
    assert reference_ids == mod.SELECTED_SOURCE_IDS
    for entry in artifact["references_added"]:
        assert entry["url"].startswith(("https://arxiv.org/abs/", "https://github.com/"))
        assert entry["carnot_hook"]
        assert ".465" in entry["carnot_hook"]

    required_tracks = {
        "verifier moat",
        "energy-guided decoding",
        "constraint satisfaction",
        "hallucination mitigation",
        "hardware-accelerated decoding",
    }
    observed_tracks = {track for source in artifact["selected_sources"] for track in source["tracks"]}
    assert required_tracks.issubset(observed_tracks)

    duplicate_filter = artifact["duplicate_filter"]
    assert "2512.05439" in duplicate_filter["rejected_duplicate_source_ids"]
    assert "2602.03034" in duplicate_filter["rejected_duplicate_source_ids"]
    assert duplicate_filter["selected_source_ids"] == mod.SELECTED_SOURCE_IDS
    assert duplicate_filter["kan_kanfis_status"].startswith("no_new_nonduplicate")
    assert duplicate_filter["ebt_arm_ebm_status"].startswith("no_new_nonduplicate")

    assert len(artifact["next_milestone_candidates"]) >= 3
    assert all(".465" in row["candidate_flag"] for row in artifact["next_milestone_candidates"])
    mod.validate_artifact(artifact)


def test_scenario_report_5053_duplicate_filter_suppresses_prior_sources() -> None:
    """SCENARIO-REPORT-5053-DUPLICATE-FILTER: duplicate IDs are not selected."""

    candidates = [
        mod.CANDIDATE_SOURCES_BY_ID["2512.05439"],
        mod.CANDIDATE_SOURCES_BY_ID["2504.04718"],
    ]

    result = mod.filter_actionable_sources(candidates, existing_reference_text="")

    assert [source["source_id"] for source in result["selected"]] == ["2504.04718"]
    assert result["rejected"][0]["source_id"] == "2512.05439"
    assert result["rejected"][0]["reason"] == "already_ingested"


def test_research_references_update_is_idempotent_for_scenario_report_5053() -> None:
    """SCENARIO-REPORT-5053: managed research reference section is stable."""

    artifact = _artifact()
    original = "# Research References\n\n## Existing\nOld body\n"
    once = mod.update_research_references_text(original, artifact)
    twice = mod.update_research_references_text(once, artifact)

    assert once == twice
    assert once.count(mod.REFERENCES_SECTION_START) == 1
    assert once.count(mod.REFERENCES_SECTION_END) == 1
    assert "Exp 5053 .465 SOTA ingestion source set" in once
    assert mod.HONEST_VERDICT in once
    for source_id in mod.SELECTED_SOURCE_IDS:
        assert f"arXiv:{source_id}" in once
    assert "arXiv:2512.05439" not in once.split(mod.REFERENCES_SECTION_START, 1)[1].split(
        mod.REFERENCES_SECTION_END, 1
    )[0]
    mod.validate_research_references_text(once, artifact)


def test_write_outputs_is_stable_for_req_report_5053(tmp_path: Path) -> None:
    """REQ-REPORT-5053: writer emits stable artifact and reference update."""

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    references_path = tmp_path / mod.REFERENCES_RELATIVE_PATH
    references_path.write_text("# Research References\n\n", encoding="utf-8")

    artifact = mod.write_outputs(artifact_path=artifact_path, references_path=references_path)
    second = mod.write_outputs(artifact_path=artifact_path, references_path=references_path)

    assert second == artifact
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    mod.validate_research_references_text(references_path.read_text(encoding="utf-8"), artifact)


def test_main_writes_deliverables_for_req_report_5053(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """REQ-REPORT-5053: direct module execution writes default outputs."""

    (tmp_path / mod.REFERENCES_RELATIVE_PATH).write_text(
        "# Research References\n\n", encoding="utf-8"
    )
    monkeypatch.setenv("CARNOT_EXP5053_ROOT", str(tmp_path))

    try:
        runpy.run_path(str(Path(mod.__file__)), run_name="__main__")
    except SystemExit as exc:
        assert exc.code == 0

    assert capsys.readouterr().out.strip() == mod.HONEST_VERDICT
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    mod.validate_artifact(written)
    mod.validate_research_references_text(
        (tmp_path / mod.REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8"), written
    )


def test_deliverable_files_validate_for_req_report_5053() -> None:
    """REQ-REPORT-5053: committed deliverables satisfy the schema."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    mod.validate_research_references_text(REFERENCES_PATH.read_text(encoding="utf-8"), artifact)
    assert artifact["honest_verdict"] == mod.HONEST_VERDICT
