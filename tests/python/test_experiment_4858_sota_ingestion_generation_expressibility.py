"""Tests for Exp 4858 generation-expressibility SOTA ingestion.

Spec refs: REQ-ARC-WMTE-4858, SCENARIO-ARC-WMTE-4858,
SCENARIO-ARC-WMTE-4858-NO-FABRICATION.
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_4858_sota_ingestion_generation_expressibility as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"
ARTIFACT_PATH = REPO / mod.RESULT_RELATIVE_PATH
STUDYING_PATH = REPO / mod.STUDYING_RELATIVE_PATH
REFERENCES_PATH = REPO / mod.REFERENCES_RELATIVE_PATH


def _artifact() -> dict[str, object]:
    return mod.build_artifact()


def _spec_section() -> str:
    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-ARC-WMTE-4858")
    end = spec.index("### REQ-ARC-WMTE-4855", start)
    return spec[start:end]


def test_req_arc_wmte_4858_spec_declares_generation_expressibility_contract() -> None:
    """REQ-ARC-WMTE-4858: OpenSpec declares the Exp 4858 artifact contract."""

    section = _spec_section()

    assert "REQ-ARC-WMTE-4858" in section
    assert "SCENARIO-ARC-WMTE-4858" in section
    assert "SCENARIO-ARC-WMTE-4858-NO-FABRICATION" in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert mod.STUDYING_RELATIVE_PATH in section
    assert mod.REFERENCES_RELATIVE_PATH in section
    assert mod.UPSTREAM_OBJECT_WORLD_MODEL_ARTIFACT in section
    assert mod.UPSTREAM_GENERATION_DIAGNOSTIC_ARTIFACT in section
    assert mod.HONEST_VERDICT in section
    assert mod.INFERENCE_SUBSTRATE in section
    assert "NEVER_ENUMERATED" in section
    assert "partial or noisy object" in section
    assert "winning prefix enters the pool" in section
    assert "/deep-research" in section
    for field, principle in mod.REQUIRED_USER_FIELD_PRINCIPLES.items():
        assert field in section
        assert principle["principle"] in section
    for source_id in mod.REQUIRED_SOURCE_IDS:
        assert source_id in section


def test_req_arc_wmte_4858_artifact_maps_methods_to_missing_prefix_pool_insertion() -> None:
    """REQ-ARC-WMTE-4858: artifact maps SOTA methods onto proposer expressibility."""

    artifact = _artifact()

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["honest_verdict"] == mod.HONEST_VERDICT
    assert artifact["aimed_at_dominant_bucket"] == "NEVER_ENUMERATED"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] == mod.DURATION_S
    assert artifact["note_path"] == mod.NOTE_PATH
    assert artifact["references_path"] == mod.REFERENCES_PATH
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["reproducibility_checksum"] == mod.REPRODUCIBILITY_CHECKSUM
    assert artifact["arxiv_ids_cited"] == sorted(mod.REQUIRED_SOURCE_IDS)
    assert set(artifact["citations"]) == mod.REQUIRED_SOURCE_IDS
    assert set(artifact["field_principles"]) == set(mod.REQUIRED_PRINCIPLE_FIELDS)

    cited: set[str] = set()
    tracks: set[str] = set()
    for method in artifact["methods_mapped"]:
        assert set(method) == mod.REQUIRED_METHOD_FIELDS
        assert method["source_ids"]
        assert set(method["source_ids"]).issubset(mod.REQUIRED_SOURCE_IDS)
        assert method["maps_to_frontier"] == ".448"
        assert method["targets_bucket"] == "NEVER_ENUMERATED"
        assert "primitive" in method["primitive_expansion"].lower()
        assert "winning prefix" in method["winner_prefix_pool_insertion"]
        assert "pool" in method["winner_prefix_pool_insertion"]
        assert "partial/noisy object signal" in method["partial_object_signal_contract"]
        assert method["proposal_graft"]
        assert method["verification_handoff"]
        assert method["takes_over_from_current_stack"]
        assert method["fails_when"]
        assert "flagged_for_v448" in method["roadmap_candidate"]
        cited.update(method["source_ids"])
        tracks.add(method["track"])

    assert 3 <= len(artifact["methods_mapped"]) <= 5
    assert cited.issubset(set(artifact["arxiv_ids_cited"]))
    assert mod.REQUIRED_TRACKS == tracks
    assert {flag["candidate"] for flag in artifact["flagged_for_v448"]} == {
        "dreamcoder_lilo_action_library",
        "eg_nps_soar_arc_program_search",
        "comet_executable_world_model_mcts",
    }

    preconditions = artifact["preconditions_checked"]
    assert preconditions["research_studying_present"] is True
    assert preconditions["research_references_present"] is True
    assert preconditions["upstream_object_world_model_artifact_present"] is True
    assert preconditions["upstream_generation_diagnostic_artifact_present"] is True
    assert preconditions["a1_dominant_bucket"] == "NEVER_ENUMERATED"
    assert preconditions["sweep_clusters_used"] is True
    assert preconditions["sweep_cluster_ids"] == [5, 6]
    assert preconditions["sweep_semscholar_used"] is True
    assert preconditions["sweep_semscholar_result"].endswith("0 unique arxiv IDs")
    assert preconditions["websearch_webfetch_used"] is True
    assert preconditions["top_source_count"] == 8
    assert preconditions["deep_research_invoked"] is False
    assert preconditions["exploration_strategy_reingested"] is False
    assert preconditions["energy_stage_reingested"] is False
    assert preconditions["model_load"] is False
    assert preconditions["training_launched"] is False
    assert preconditions["leaderboard_submission"] is False
    assert preconditions["solve_claim_made"] is False
    assert preconditions["ops_docs_modified"] is False

    upstream = artifact["upstream_artifacts"]
    assert upstream["object_world_model_handoff"] == mod.UPSTREAM_OBJECT_WORLD_MODEL_ARTIFACT
    assert upstream["generation_diagnostic"] == mod.UPSTREAM_GENERATION_DIAGNOSTIC_ARTIFACT
    assert upstream["dominant_bucket"] == "NEVER_ENUMERATED"
    assert upstream["partial_noisy_object_signal_only"] is True
    assert "exact object identity" in upstream["object_signal_constraint"]

    note = artifact["generation_expressibility_mapping_note"]
    assert set(note) == mod.REQUIRED_MAPPING_NOTE_FIELDS
    assert note["terminal_success"] == mod.HONEST_VERDICT
    assert note["root_cause"] == "generation expressibility"
    assert "NEVER_ENUMERATED" in note["summary"]
    assert "winning prefix into the pool" in note["summary"]


@pytest.mark.parametrize(
    ("bad_artifact", "message"),
    [
        (_artifact() | {"honest_verdict": "complete_wrong"}, "honest_verdict"),
        (_artifact() | {"inference_substrate": "live_llm"}, "inference_substrate"),
        (_artifact() | {"aimed_at_dominant_bucket": "COVERED"}, "NEVER_ENUMERATED"),
        (_artifact() | {"field_principles": {}}, "field_principles"),
        (_artifact() | {"arxiv_ids_cited": []}, "arxiv_ids_cited"),
        (_artifact() | {"citations": {}}, "citations"),
        (
            _artifact()
            | {
                "citations": mod.CITATIONS
                | {
                    "2606.14418": mod.CITATIONS["2606.14418"]
                    | {"url": "https://arxiv.org/abs/9999.99999"}
                }
            },
            "citation url",
        ),
        (
            _artifact()
            | {
                "citations": mod.CITATIONS
                | {"2606.14418": mod.CITATIONS["2606.14418"] | {"http_status": 404}}
            },
            "http_status",
        ),
        (_artifact() | {"methods_mapped": mod.DEFAULT_METHODS_MAPPED[:2]}, "three to five"),
        (
            _artifact()
            | {
                "methods_mapped": [
                    mod.DEFAULT_METHODS_MAPPED[0] | {"source_ids": ["9999.99999"]},
                    *mod.DEFAULT_METHODS_MAPPED[1:],
                ]
            },
            "verified citations",
        ),
        (
            _artifact()
            | {
                "methods_mapped": [
                    mod.DEFAULT_METHODS_MAPPED[0] | {"targets_bucket": "COVERED"},
                    *mod.DEFAULT_METHODS_MAPPED[1:],
                ]
            },
            "NEVER_ENUMERATED",
        ),
        (
            _artifact()
            | {
                "methods_mapped": [
                    mod.DEFAULT_METHODS_MAPPED[0] | {"winner_prefix_pool_insertion": "rerank old pool"},
                    *mod.DEFAULT_METHODS_MAPPED[1:],
                ]
            },
            "winning prefix",
        ),
        (
            _artifact()
            | {
                "methods_mapped": [
                    mod.DEFAULT_METHODS_MAPPED[0]
                    | {"partial_object_signal_contract": "requires exact object identity"},
                    *mod.DEFAULT_METHODS_MAPPED[1:],
                ]
            },
            "partial/noisy object signal",
        ),
        (_artifact() | {"flagged_for_v448": []}, "flagged_for_v448"),
        (
            _artifact()
            | {"flagged_for_v448": [{"candidate": "stale", "flag": "flagged_for_v447"}]},
            "stale",
        ),
        (
            _artifact()
            | {
                "preconditions_checked": dict(mod.DEFAULT_PRECONDITIONS_CHECKED)
                | {"deep_research_invoked": True}
            },
            "deep-research",
        ),
        (
            _artifact()
            | {
                "preconditions_checked": dict(mod.DEFAULT_PRECONDITIONS_CHECKED)
                | {"energy_stage_reingested": True}
            },
            "energy",
        ),
        (
            _artifact()
            | {
                "preconditions_checked": dict(mod.DEFAULT_PRECONDITIONS_CHECKED)
                | {"top_source_count": 4}
            },
            "top five to eight",
        ),
        (
            _artifact()
            | {
                "upstream_artifacts": dict(mod.DEFAULT_UPSTREAM_ARTIFACTS)
                | {"dominant_bucket": "COVERED"}
            },
            "dominant bucket",
        ),
        (
            _artifact()
            | {
                "generation_expressibility_mapping_note": dict(mod.DEFAULT_MAPPING_NOTE)
                | {"source_ids": []}
            },
            "mapping note",
        ),
    ],
)
def test_no_fabrication_validator_rejects_bad_generation_expressibility_artifacts(
    bad_artifact: dict[str, object], message: str
) -> None:
    """SCENARIO-ARC-WMTE-4858-NO-FABRICATION: invalid claims fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_validate_artifact_rejects_exact_schema_violations() -> None:
    """REQ-ARC-WMTE-4858: top-level and method schemas are exact."""

    missing = _artifact()
    missing.pop("methods_mapped")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing)

    extra = _artifact() | {"solve_rate": 0.0}
    with pytest.raises(ValueError, match="unexpected fields"):
        mod.validate_artifact(extra)

    malformed_method = _artifact() | {"methods_mapped": ["not-a-dict", *mod.DEFAULT_METHODS_MAPPED[1:]]}
    with pytest.raises(ValueError, match="method"):
        mod.validate_artifact(malformed_method)

    bad_sweep = _artifact() | {"fresh_sweep": dict(mod.DEFAULT_FRESH_SWEEP) | {"cluster_ids": [6]}}
    with pytest.raises(ValueError, match="clusters"):
        mod.validate_artifact(bad_sweep)


def test_scenario_arc_wmte_4858_research_studying_section_is_idempotent() -> None:
    """SCENARIO-ARC-WMTE-4858: research-studying update is stable and cited."""

    original = "# Research Studying\n\nOld body\n"
    once = mod.update_research_studying_text(original, _artifact())
    twice = mod.update_research_studying_text(once, _artifact())

    assert once == twice
    assert once.count(mod.STUDYING_SECTION_START) == 1
    assert once.count(mod.STUDYING_SECTION_END) == 1
    mod.validate_research_studying_text(once, _artifact())
    assert "SOTA -> generation expressibility mapping" in once
    assert "flagged_for_v448" in once
    assert "arXiv:2006.08381" in once
    assert "arXiv:2606.14418" in once
    assert "NEVER_ENUMERATED" in once
    assert "partial/noisy object signal" in once
    assert "winning prefix into the pool" in once
    assert "no solve claim" in once

    later_section = "# Research Studying\n\n<!-- EXP0000-OLDER-START -->\nOld section\n"
    inserted = mod.update_research_studying_text(later_section, _artifact())
    refreshed = mod.update_research_studying_text(inserted, _artifact())

    assert refreshed == inserted
    assert inserted.index(mod.STUDYING_SECTION_START) < inserted.index("<!-- EXP0000-OLDER-START -->")


def test_scenario_arc_wmte_4858_research_references_section_is_idempotent() -> None:
    """SCENARIO-ARC-WMTE-4858: research-references update is stable and cited."""

    original = "# Research References\n\n## Existing Section\nOld body\n"
    once = mod.update_research_references_text(original, _artifact())
    twice = mod.update_research_references_text(once, _artifact())

    assert once == twice
    assert once.count(mod.REFERENCES_SECTION_START) == 1
    assert once.count(mod.REFERENCES_SECTION_END) == 1
    mod.validate_research_references_text(once, _artifact())
    assert "Exp 4858 generation-expressibility source set" in once
    assert "DreamCoder" in once
    assert "LILO" in once
    assert "SOAR" in once
    assert "Object-Centric World Models Meet Monte Carlo Tree Search" in once


def test_research_markdown_updates_reject_broken_existing_sections() -> None:
    """REQ-ARC-WMTE-4858: malformed existing note markers fail closed."""

    with pytest.raises(ValueError, match="missing end marker"):
        mod.update_research_studying_text("# x\n\n" + mod.STUDYING_SECTION_START + "\n", _artifact())
    with pytest.raises(ValueError, match="missing end marker"):
        mod.update_research_references_text("# x\n\n" + mod.REFERENCES_SECTION_START + "\n", _artifact())


def test_write_outputs_is_idempotent_for_scenario_arc_wmte_4858(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4858: writer emits stable JSON and markdown notes."""

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    studying_path = tmp_path / mod.STUDYING_RELATIVE_PATH
    references_path = tmp_path / mod.REFERENCES_RELATIVE_PATH
    studying_path.parent.mkdir(parents=True, exist_ok=True)
    studying_path.write_text("# Research Studying\n\n", encoding="utf-8")
    references_path.write_text("# Research References\n\n", encoding="utf-8")

    artifact = mod.write_outputs(
        artifact_path=artifact_path,
        studying_path=studying_path,
        references_path=references_path,
    )
    second_artifact = mod.write_outputs(
        artifact_path=artifact_path,
        studying_path=studying_path,
        references_path=references_path,
    )

    assert second_artifact == artifact
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    mod.validate_research_studying_text(studying_path.read_text(encoding="utf-8"), artifact)
    mod.validate_research_references_text(references_path.read_text(encoding="utf-8"), artifact)


def test_main_writes_deliverables_for_req_arc_wmte_4858(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-ARC-WMTE-4858: direct module execution writes default outputs."""

    (tmp_path / mod.STUDYING_RELATIVE_PATH).write_text("# Research Studying\n\n", encoding="utf-8")
    (tmp_path / mod.REFERENCES_RELATIVE_PATH).write_text("# Research References\n\n", encoding="utf-8")
    monkeypatch.setenv("CARNOT_EXP4858_ROOT", str(tmp_path))

    with pytest.raises(SystemExit) as module_exit:
        runpy.run_path(str(Path(mod.__file__)), run_name="__main__")

    assert module_exit.value.code == 0
    assert capsys.readouterr().out.strip() == mod.HONEST_VERDICT
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    mod.validate_artifact(written)
    mod.validate_research_studying_text(
        (tmp_path / mod.STUDYING_RELATIVE_PATH).read_text(encoding="utf-8"),
        written,
    )
    mod.validate_research_references_text(
        (tmp_path / mod.REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8"),
        written,
    )


def test_deliverable_files_validate_for_req_arc_wmte_4858() -> None:
    """REQ-ARC-WMTE-4858: committed deliverables satisfy the schema."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    studying = STUDYING_PATH.read_text(encoding="utf-8")
    references = REFERENCES_PATH.read_text(encoding="utf-8")

    mod.validate_artifact(artifact)
    mod.validate_research_studying_text(studying, artifact)
    mod.validate_research_references_text(references, artifact)
    assert artifact["honest_verdict"] == mod.HONEST_VERDICT
    assert artifact["flagged_for_v448"] == mod.FLAGGED_FOR_V448
