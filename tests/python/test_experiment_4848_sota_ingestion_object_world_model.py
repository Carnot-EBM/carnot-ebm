"""Tests for Exp 4848 object-world-model SOTA ingestion.

Spec refs: REQ-ARC-WMTE-4848, SCENARIO-ARC-WMTE-4848,
SCENARIO-ARC-WMTE-4848-NO-FABRICATION.
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_4848_sota_ingestion_object_world_model as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"
ARTIFACT_PATH = REPO / mod.RESULT_RELATIVE_PATH
STUDYING_PATH = REPO / mod.STUDYING_RELATIVE_PATH


def _artifact() -> dict[str, object]:
    return mod.build_artifact()


def _spec_section() -> str:
    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-ARC-WMTE-4848")
    end = spec.index("### REQ-ARC-WMTE-4838", start)
    return spec[start:end]


def test_req_arc_wmte_4848_spec_declares_object_world_model_contract() -> None:
    """REQ-ARC-WMTE-4848: OpenSpec declares the Exp 4848 artifact contract."""

    section = _spec_section()

    assert "REQ-ARC-WMTE-4848" in section
    assert "SCENARIO-ARC-WMTE-4848" in section
    assert "SCENARIO-ARC-WMTE-4848-NO-FABRICATION" in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert mod.STUDYING_RELATIVE_PATH in section
    assert mod.UPSTREAM_PERCEPTION_ARTIFACT in section
    assert mod.HONEST_VERDICT in section
    assert mod.INFERENCE_SUBSTRATE in section
    assert "flagged_for_v447" in section
    assert "A1 object layer" in section
    assert "proposable winner" in section
    assert "object-centric world-model/planning" in section
    assert "/deep-research" in section
    for field, principle in mod.REQUIRED_USER_FIELD_PRINCIPLES.items():
        assert field in section
        assert principle["principle"] in section
    for source_id in mod.REQUIRED_SOURCE_IDS:
        assert source_id in section


def test_req_arc_wmte_4848_artifact_maps_a1_object_layer_to_proposable_winners() -> None:
    """REQ-ARC-WMTE-4848: artifact maps cited methods onto the .447 handoff."""

    artifact = _artifact()

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["honest_verdict"] == mod.HONEST_VERDICT
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] == mod.DURATION_S
    assert artifact["note_path"] == mod.NOTE_PATH
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
        assert method["maps_to_frontier"] == ".447"
        assert "A1" in method["consumes_a1_object_layer"]
        assert "object" in method["object_relational_state"].lower()
        assert method["planning_graft"]
        assert "propos" in method["proposable_winner_output"].lower()
        assert method["verification_handoff"]
        assert method["takes_over_from_current_stack"]
        assert method["fails_when"]
        assert "flagged_for_v447" in method["roadmap_candidate"]
        cited.update(method["source_ids"])
        tracks.add(method["track"])

    assert 3 <= len(artifact["methods_mapped"]) <= 5
    assert cited.issubset(set(artifact["arxiv_ids_cited"]))
    assert mod.REQUIRED_TRACKS == tracks
    assert {flag["candidate"] for flag in artifact["flagged_for_v447"]} == {
        "comet_object_mcts_planner",
        "slot_mpc_object_action_optimizer",
        "loop_owm_interaction_primitive_proposer",
    }

    preconditions = artifact["preconditions_checked"]
    assert preconditions["research_studying_present"] is True
    assert preconditions["research_references_present"] is True
    assert preconditions["upstream_perception_artifact_present"] is True
    assert preconditions["upstream_flagged_for_v446_read"] is True
    assert preconditions["sweep_clusters_used"] is True
    assert preconditions["sweep_cluster_ids"] == [3, 5, 6]
    assert preconditions["sweep_semscholar_used"] is True
    assert preconditions["semantic_scholar_unique_arxiv_ids"] == [
        "2402.03326",
        "2410.08822",
        "2502.07600",
        "2503.06170",
        "2507.03298",
        "2511.02225",
        "2605.14937",
        "2606.14418",
    ]
    assert preconditions["websearch_webfetch_used"] is True
    assert preconditions["top_source_count"] == 8
    assert preconditions["deep_research_invoked"] is False
    assert preconditions["exploration_strategy_reingested"] is False
    assert preconditions["model_load"] is False
    assert preconditions["training_launched"] is False
    assert preconditions["leaderboard_submission"] is False
    assert preconditions["solve_claim_made"] is False
    assert preconditions["ops_docs_modified"] is False

    a1_input = artifact["a1_perception_layer_input"]
    assert a1_input["source_artifact"] == mod.UPSTREAM_PERCEPTION_ARTIFACT
    assert a1_input["source_honest_verdict"] == "success_sota_ingestion_perception_representation_mapped"
    assert a1_input["target_roadmap"] == ".447"
    assert a1_input["exploration_strategy_class_closed"] is True
    assert "object_ids" in a1_input["consumed_state_fields"]
    assert "relation_edges" in a1_input["consumed_state_fields"]
    assert "object_action_bindings" in a1_input["consumed_state_fields"]

    note = artifact["object_world_model_mapping_note"]
    assert set(note) == mod.REQUIRED_MAPPING_NOTE_FIELDS
    assert note["terminal_success"] == mod.HONEST_VERDICT
    assert "A1 object layer" in note["summary"]
    assert "proposable winner" in note["summary"]


@pytest.mark.parametrize(
    ("bad_artifact", "message"),
    [
        (_artifact() | {"honest_verdict": "complete_wrong"}, "honest_verdict"),
        (_artifact() | {"inference_substrate": "live_llm"}, "inference_substrate"),
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
                    mod.DEFAULT_METHODS_MAPPED[0] | {"maps_to_frontier": ".446"},
                    *mod.DEFAULT_METHODS_MAPPED[1:],
                ]
            },
            ".447",
        ),
        (
            _artifact()
            | {
                "methods_mapped": [
                    mod.DEFAULT_METHODS_MAPPED[0] | {"consumes_a1_object_layer": ""},
                    *mod.DEFAULT_METHODS_MAPPED[1:],
                ]
            },
            "A1 object layer",
        ),
        (
            _artifact()
            | {
                "methods_mapped": [
                    mod.DEFAULT_METHODS_MAPPED[0] | {"proposable_winner_output": "rerank old pool"},
                    *mod.DEFAULT_METHODS_MAPPED[1:],
                ]
            },
            "proposable winner",
        ),
        (_artifact() | {"flagged_for_v447": []}, "flagged_for_v447"),
        (
            _artifact()
            | {"flagged_for_v447": [{"candidate": "stale", "flag": "flagged_for_v446"}]},
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
                | {"exploration_strategy_reingested": True}
            },
            "exploration",
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
                "preconditions_checked": dict(mod.DEFAULT_PRECONDITIONS_CHECKED)
                | {"upstream_perception_artifact_present": False}
            },
            "upstream",
        ),
        (
            _artifact()
            | {
                "a1_perception_layer_input": dict(mod.DEFAULT_A1_PERCEPTION_LAYER_INPUT)
                | {"source_artifact": "results/stale.json"}
            },
            "Exp 4838",
        ),
        (
            _artifact()
            | {
                "object_world_model_mapping_note": dict(mod.DEFAULT_MAPPING_NOTE)
                | {"source_ids": []}
            },
            "mapping note",
        ),
    ],
)
def test_no_fabrication_validator_rejects_bad_object_world_model_artifacts(
    bad_artifact: dict[str, object], message: str
) -> None:
    """SCENARIO-ARC-WMTE-4848-NO-FABRICATION: invalid claims fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_validate_artifact_rejects_exact_schema_violations() -> None:
    """REQ-ARC-WMTE-4848: top-level and method schemas are exact."""

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

    bad_sweep = _artifact() | {"fresh_sweep": dict(mod.DEFAULT_FRESH_SWEEP) | {"cluster_ids": [5, 6]}}
    with pytest.raises(ValueError, match="clusters"):
        mod.validate_artifact(bad_sweep)


def test_scenario_arc_wmte_4848_research_studying_section_is_idempotent() -> None:
    """SCENARIO-ARC-WMTE-4848: research-studying update is stable and cited."""

    original = "# Research Studying\n\nOld body\n"
    once = mod.update_research_studying_text(original, _artifact())
    twice = mod.update_research_studying_text(once, _artifact())

    assert once == twice
    assert once.count(mod.STUDYING_SECTION_START) == 1
    assert once.count(mod.STUDYING_SECTION_END) == 1
    mod.validate_research_studying_text(once, _artifact())
    assert "SOTA -> object-world-model planning mapping" in once
    assert "flagged_for_v447" in once
    assert "arXiv:2606.14418" in once
    assert "arXiv:2605.14937" in once
    assert "A1 object layer" in once
    assert "proposable winner" in once
    assert "no solve claim" in once

    later_section = "# Research Studying\n\n<!-- EXP0000-OLDER-START -->\nOld section\n"
    inserted = mod.update_research_studying_text(later_section, _artifact())
    refreshed = mod.update_research_studying_text(inserted, _artifact())

    assert refreshed == inserted
    assert inserted.index(mod.STUDYING_SECTION_START) < inserted.index("<!-- EXP0000-OLDER-START -->")

    existing_header = "# Research Studying\n\n## Existing Section\nOld body\n"
    inserted_before_header = mod.update_research_studying_text(existing_header, _artifact())
    assert inserted_before_header.index(mod.STUDYING_SECTION_START) < inserted_before_header.index(
        "## Existing Section"
    )


def test_research_studying_update_rejects_broken_existing_section() -> None:
    """REQ-ARC-WMTE-4848: malformed existing note markers fail closed."""

    with pytest.raises(ValueError, match="missing end marker"):
        mod.update_research_studying_text("# x\n\n" + mod.STUDYING_SECTION_START + "\n", _artifact())


def test_write_outputs_is_idempotent_for_scenario_arc_wmte_4848(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4848: writer emits stable JSON and studying note."""

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    studying_path = tmp_path / mod.STUDYING_RELATIVE_PATH
    studying_path.parent.mkdir(parents=True, exist_ok=True)
    studying_path.write_text("# Research Studying\n\n", encoding="utf-8")

    artifact = mod.write_outputs(artifact_path=artifact_path, studying_path=studying_path)
    second_artifact = mod.write_outputs(artifact_path=artifact_path, studying_path=studying_path)

    assert second_artifact == artifact
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    mod.validate_research_studying_text(studying_path.read_text(encoding="utf-8"), artifact)


def test_main_writes_deliverables_for_req_arc_wmte_4848(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-ARC-WMTE-4848: direct module execution writes default outputs."""

    (tmp_path / mod.STUDYING_RELATIVE_PATH).write_text("# Research Studying\n\n", encoding="utf-8")
    monkeypatch.setenv("CARNOT_EXP4848_ROOT", str(tmp_path))

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


def test_deliverable_files_validate_for_req_arc_wmte_4848() -> None:
    """REQ-ARC-WMTE-4848: committed deliverables satisfy the schema."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    studying = STUDYING_PATH.read_text(encoding="utf-8")

    mod.validate_artifact(artifact)
    mod.validate_research_studying_text(studying, artifact)
    assert artifact["honest_verdict"] == mod.HONEST_VERDICT
    assert artifact["flagged_for_v447"] == mod.FLAGGED_FOR_V447
