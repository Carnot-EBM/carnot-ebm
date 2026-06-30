"""Tests for Exp 5010 verifier-moat SOTA ingestion.

Spec refs: REQ-REPORT-5010, SCENARIO-REPORT-5010,
SCENARIO-REPORT-5010-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import copy
import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_5010_sota_ingestion_verifier_moat as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"
ARTIFACT_PATH = REPO / mod.RESULT_RELATIVE_PATH
NOTE_PATH = REPO / mod.NOTE_RELATIVE_PATH
STUDYING_PATH = REPO / mod.STUDYING_RELATIVE_PATH
REFERENCES_PATH = REPO / mod.REFERENCES_RELATIVE_PATH


def _artifact() -> dict[str, object]:
    return mod.build_artifact(preconditions_checked=mod.DEFAULT_PRECONDITIONS_CHECKED)


def _bad_artifact(**updates: object) -> dict[str, object]:
    artifact = copy.deepcopy(_artifact())
    artifact.update(updates)
    return artifact


def _spec_section() -> str:
    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-REPORT-5010")
    end = spec.index("### REQ-REPORT-4873", start)
    return spec[start:end]


def test_req_report_5010_spec_declares_sota_ingestion_contract() -> None:
    """REQ-REPORT-5010: OpenSpec declares the Exp 5010 contract."""

    section = _spec_section()

    assert "REQ-REPORT-5010" in section
    assert "SCENARIO-REPORT-5010" in section
    assert "SCENARIO-REPORT-5010-BLOCKED-PRECONDITION" in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert mod.NOTE_RELATIVE_PATH in section
    assert mod.HONEST_VERDICT in section
    assert "/deep-research" in section
    assert "scripts/research_conductor.py" in section
    assert "D1 LoRA-EBM" in section
    assert "D2 uPRM" in section
    assert "D3 EBRM" in section
    for field, principle in mod.REQUIRED_USER_FIELD_PRINCIPLES.items():
        assert field in section
        assert principle["principle"] in section


def test_req_report_5010_artifact_maps_new_sources_to_phase_d() -> None:
    """REQ-REPORT-5010: artifact maps verified new sources onto D1/D2/D3."""

    artifact = _artifact()

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["honest_verdict"] == mod.HONEST_VERDICT
    assert artifact["new_arxiv_ids"] == mod.NEW_ARXIV_IDS
    assert not set(artifact["new_arxiv_ids"]).intersection(mod.ALREADY_INGESTED_ARXIV_IDS)
    assert artifact["note_path"] == mod.NOTE_RELATIVE_PATH
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES

    citations = artifact["citations_verified"]
    assert set(citations) == set(mod.NEW_ARXIV_IDS)
    for source_id, citation in citations.items():
        assert citation["http_status"] == 200
        assert citation["url"] == f"https://arxiv.org/abs/{source_id}"
        assert citation["title"]

    seen_sources: set[str] = set()
    seen_arms: set[str] = set()
    for mapping in artifact["sota_to_phase_d_mapping"]:
        assert set(mapping) == mod.REQUIRED_MAPPING_FIELDS
        assert mapping["arxiv_id"] in mod.NEW_ARXIV_IDS
        assert mapping["url"] == f"https://arxiv.org/abs/{mapping['arxiv_id']}"
        assert set(mapping["phase_d_arms"]).issubset(mod.PHASE_D_ARMS)
        assert ".462" in mapping["candidate_flag"]
        assert mapping["implementation_delta"]
        assert mapping["pitfall"]
        seen_sources.add(mapping["arxiv_id"])
        seen_arms.update(mapping["phase_d_arms"])

    assert 3 <= len(artifact["sota_to_phase_d_mapping"]) <= 5
    assert seen_sources == set(mod.NEW_ARXIV_IDS)
    assert {"D1 LoRA-EBM", "D2 uPRM", "D3 EBRM"}.issubset(seen_arms)
    assert len(artifact["next_milestone_candidates"]) >= 2
    assert all(
        ".462" in candidate["candidate_flag"] for candidate in artifact["next_milestone_candidates"]
    )

    reliable = artifact["reliable_channel_used"]
    assert reliable["sweep_clusters_used"] is True
    assert reliable["sweep_semscholar_used"] is True
    assert reliable["websearch_webfetch_used"] is True
    assert reliable["deep_research_invoked"] is False
    assert "HTTP 429" in reliable["semscholar_result"]

    preconditions = artifact["preconditions_checked"]
    assert preconditions["network_arxiv_reachable"] is True
    assert preconditions["sweep_helpers_importable"] is True
    assert preconditions["deep_research_invoked"] is False
    assert preconditions["research_conductor_modified"] is False
    assert preconditions["ops_docs_modified"] is False


@pytest.mark.parametrize(
    ("bad_artifact", "message"),
    [
        (_bad_artifact(honest_verdict="finished"), "honest_verdict"),
        (_bad_artifact(new_arxiv_ids=["2605.18871", *mod.NEW_ARXIV_IDS[:2]]), "already ingested"),
        (
            _bad_artifact(
                citations_verified=mod.CITATIONS_VERIFIED
                | {"2606.19818": mod.CITATIONS_VERIFIED["2606.19818"] | {"http_status": 404}}
            ),
            "HTTP 200",
        ),
        (_bad_artifact(sota_to_phase_d_mapping=[]), "three to five"),
        (
            _bad_artifact(
                sota_to_phase_d_mapping=[
                    mod.DEFAULT_SOTA_TO_PHASE_D_MAPPING[0] | {"candidate_flag": "flagged_for_v461"},
                    *mod.DEFAULT_SOTA_TO_PHASE_D_MAPPING[1:],
                ]
            ),
            ".462",
        ),
        (
            _bad_artifact(
                reliable_channel_used=mod.RELIABLE_CHANNEL_USED | {"deep_research_invoked": True}
            ),
            "deep-research",
        ),
        (_bad_artifact(inference_substrate="live_llm_inference"), "inference_substrate"),
        (_bad_artifact(field_principles={}), "field_principles"),
    ],
)
def test_validator_rejects_fabricated_or_stale_claims_for_scenario_report_5010(
    bad_artifact: dict[str, object], message: str
) -> None:
    """SCENARIO-REPORT-5010: invalid citation and channel claims fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_blocked_precondition_artifact_for_scenario_report_5010() -> None:
    """SCENARIO-REPORT-5010-BLOCKED-PRECONDITION: missing network blocks."""

    artifact = mod.build_blocked_artifact(
        blocked_resource="network",
        preconditions_checked=mod.DEFAULT_PRECONDITIONS_CHECKED
        | {"network_arxiv_reachable": False},
    )

    assert artifact["honest_verdict"] == "blocked_network"
    assert artifact["new_arxiv_ids"] == []
    assert artifact["sota_to_phase_d_mapping"] == []
    assert artifact["next_milestone_candidates"] == []
    assert artifact["preconditions_checked"]["network_arxiv_reachable"] is False
    mod.validate_artifact(artifact)


def test_research_sections_are_idempotent_for_scenario_report_5010() -> None:
    """SCENARIO-REPORT-5010: research markdown updates are stable."""

    artifact = _artifact()
    studying_original = "# Research Studying\n\nOld body\n"
    studying_once = mod.update_research_studying_text(studying_original, artifact)
    studying_twice = mod.update_research_studying_text(studying_once, artifact)

    assert studying_once == studying_twice
    assert studying_once.count(mod.STUDYING_SECTION_START) == 1
    assert "Exp 5010 - verifier-moat literature SOTA ingestion - INGESTED" in studying_once
    assert "success_sota_ingested_5_new_papers_mapped_to_phase_d" in studying_once
    assert "flagged_for_v462" in studying_once
    mod.validate_research_studying_text(studying_once, artifact)

    references_original = "# Research References\n\n## Existing\nOld body\n"
    references_once = mod.update_research_references_text(references_original, artifact)
    references_twice = mod.update_research_references_text(references_once, artifact)

    assert references_once == references_twice
    assert references_once.count(mod.REFERENCES_SECTION_START) == 1
    assert "Exp 5010 verifier-moat literature source set" in references_once
    for source_id in mod.NEW_ARXIV_IDS:
        assert source_id in references_once
    mod.validate_research_references_text(references_once, artifact)


def test_write_outputs_is_stable_for_req_report_5010(tmp_path: Path) -> None:
    """REQ-REPORT-5010: writer emits stable artifact, note, and research updates."""

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    note_path = tmp_path / mod.NOTE_RELATIVE_PATH
    studying_path = tmp_path / mod.STUDYING_RELATIVE_PATH
    references_path = tmp_path / mod.REFERENCES_RELATIVE_PATH
    studying_path.write_text("# Research Studying\n\n", encoding="utf-8")
    references_path.write_text("# Research References\n\n", encoding="utf-8")

    artifact = mod.write_outputs(
        artifact_path=artifact_path,
        note_path=note_path,
        studying_path=studying_path,
        references_path=references_path,
        preconditions_checked=mod.DEFAULT_PRECONDITIONS_CHECKED,
    )
    second = mod.write_outputs(
        artifact_path=artifact_path,
        note_path=note_path,
        studying_path=studying_path,
        references_path=references_path,
        preconditions_checked=mod.DEFAULT_PRECONDITIONS_CHECKED,
    )

    assert second == artifact
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    mod.validate_markdown_note(note_path.read_text(encoding="utf-8"), artifact)
    mod.validate_research_studying_text(studying_path.read_text(encoding="utf-8"), artifact)
    mod.validate_research_references_text(references_path.read_text(encoding="utf-8"), artifact)


@pytest.mark.parametrize(
    ("precondition", "verdict"),
    [
        ("network_arxiv_reachable", "blocked_network"),
        ("sweep_helpers_importable", "blocked_sweep_helpers"),
    ],
)
def test_write_outputs_blocks_missing_reliable_channel_for_scenario_report_5010(
    tmp_path: Path,
    precondition: str,
    verdict: str,
) -> None:
    """SCENARIO-REPORT-5010-BLOCKED-PRECONDITION: writer fails closed."""

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    note_path = tmp_path / mod.NOTE_RELATIVE_PATH
    studying_path = tmp_path / mod.STUDYING_RELATIVE_PATH
    references_path = tmp_path / mod.REFERENCES_RELATIVE_PATH
    preconditions = mod.DEFAULT_PRECONDITIONS_CHECKED | {precondition: False}

    artifact = mod.write_outputs(
        artifact_path=artifact_path,
        note_path=note_path,
        studying_path=studying_path,
        references_path=references_path,
        preconditions_checked=preconditions,
    )

    assert artifact["honest_verdict"] == verdict
    assert artifact["new_arxiv_ids"] == []
    assert artifact["sota_to_phase_d_mapping"] == []
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert not note_path.exists()
    assert not studying_path.exists()
    assert not references_path.exists()
    mod.validate_artifact(artifact)


def test_main_writes_deliverables_for_req_report_5010(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-5010: direct module execution writes default outputs."""

    (tmp_path / mod.STUDYING_RELATIVE_PATH).write_text("# Research Studying\n\n", encoding="utf-8")
    (tmp_path / mod.REFERENCES_RELATIVE_PATH).write_text(
        "# Research References\n\n", encoding="utf-8"
    )
    monkeypatch.setenv("CARNOT_EXP5010_ROOT", str(tmp_path))
    monkeypatch.setenv("CARNOT_EXP5010_USE_DEFAULT_PREFLIGHT", "1")

    with pytest.raises(SystemExit) as module_exit:
        runpy.run_path(str(Path(mod.__file__)), run_name="__main__")

    assert module_exit.value.code == 0
    assert capsys.readouterr().out.strip() == mod.HONEST_VERDICT
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    mod.validate_artifact(written)
    mod.validate_markdown_note(
        (tmp_path / mod.NOTE_RELATIVE_PATH).read_text(encoding="utf-8"), written
    )


def test_deliverable_files_validate_for_req_report_5010() -> None:
    """REQ-REPORT-5010: committed deliverables satisfy the schema."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    mod.validate_markdown_note(NOTE_PATH.read_text(encoding="utf-8"), artifact)
    mod.validate_research_studying_text(STUDYING_PATH.read_text(encoding="utf-8"), artifact)
    mod.validate_research_references_text(REFERENCES_PATH.read_text(encoding="utf-8"), artifact)
    assert artifact["honest_verdict"] == mod.HONEST_VERDICT
