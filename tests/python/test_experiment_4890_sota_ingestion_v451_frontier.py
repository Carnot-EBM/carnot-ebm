"""Tests for Exp 4890 V451 SOTA ingestion.

Spec refs: REQ-ARC-WMTE-4890,
SCENARIO-ARC-WMTE-4890-V451-FRONTIER-MAPPED,
SCENARIO-ARC-WMTE-4890-BLOCKED-A1,
SCENARIO-ARC-WMTE-4890-NO-FABRICATION.
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_4890_sota_ingestion_v451_frontier as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"
ARTIFACT_PATH = REPO / mod.RESULT_RELATIVE_PATH
STUDYING_PATH = REPO / mod.STUDYING_RELATIVE_PATH
REFERENCES_PATH = REPO / mod.REFERENCES_RELATIVE_PATH


def _artifact() -> dict[str, object]:
    return mod.build_artifact(upstream_context=mod.DEFAULT_UPSTREAM_CONTEXT)


def _spec_section() -> str:
    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-ARC-WMTE-4890")
    end = spec.index("### REQ-ARC-WMTE-4852", start)
    return spec[start:end]


def test_req_arc_wmte_4890_spec_declares_v451_contract() -> None:
    """REQ-ARC-WMTE-4890: OpenSpec declares the Exp 4890 artifact contract."""

    section = _spec_section()

    assert "REQ-ARC-WMTE-4890" in section
    assert "SCENARIO-ARC-WMTE-4890-V451-FRONTIER-MAPPED" in section
    assert "SCENARIO-ARC-WMTE-4890-BLOCKED-A1" in section
    assert "SCENARIO-ARC-WMTE-4890-NO-FABRICATION" in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert mod.A1_ARTIFACT_RELATIVE_PATH in section
    assert mod.A1B_ARTIFACT_RELATIVE_PATH in section
    assert mod.HANDOFF_ARTIFACT_RELATIVE_PATH in section
    assert mod.HONEST_VERDICT in section
    assert mod.BLOCKED_A1_VERDICT in section
    assert mod.INFERENCE_SUBSTRATE in section
    assert "INDUCER_CEILING_HARD" in section
    assert "METHOD_IS_CEILING" in section
    assert "alternative world-model representations" in section
    assert "/deep-research" in section
    assert "scripts/research_conductor.py" in section
    for field, principle in mod.REQUIRED_USER_FIELD_PRINCIPLES.items():
        assert field in section
        assert principle["principle"] in section
    for source_id in mod.REQUIRED_SOURCE_IDS:
        assert source_id in section


def test_req_arc_wmte_4890_artifact_maps_method_ceiling_to_alt_representations() -> None:
    """REQ-ARC-WMTE-4890: artifact maps A1/A1b residuals to V451 methods."""

    artifact = _artifact()

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["honest_verdict"] == mod.HONEST_VERDICT
    assert artifact["aimed_at_fork_verdict"] == mod.AIMED_AT_FORK_VERDICT
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] == mod.DURATION_S
    assert artifact["note_path"] == mod.NOTE_PATH
    assert artifact["references_path"] == mod.REFERENCES_PATH
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["reproducibility_checksum"] == mod.REPRODUCIBILITY_CHECKSUM
    assert artifact["arxiv_ids_cited"] == sorted(mod.REQUIRED_SOURCE_IDS)
    assert set(artifact["citations"]) == mod.REQUIRED_SOURCE_IDS
    assert set(artifact["field_principles"]) == set(mod.REQUIRED_PRINCIPLE_FIELDS)
    assert artifact["banned_channels_excluded"] is True

    cited: set[str] = set()
    tracks: set[str] = set()
    for method in artifact["methods_mapped"]:
        assert set(method) == mod.REQUIRED_METHOD_FIELDS
        assert method["source_ids"]
        assert set(method["source_ids"]).issubset(mod.REQUIRED_SOURCE_IDS)
        assert method["maps_to_frontier"] == ".451"
        assert method["targets_fork_verdict"] == mod.AIMED_AT_FORK_VERDICT
        assert method["targets_inducer_attribution"] == mod.AIMED_AT_INDUCER_ATTRIBUTION
        assert "A1b" in method["a1b_result_fit"]
        assert "arXiv:" in method["evidence"]
        assert method["experiment_graft"]
        assert method["validation_gate"]
        assert method["fails_when"]
        assert "flagged_for_v451" in method["roadmap_candidate"]
        assert method["retired_class_reingested"] is False
        cited.update(method["source_ids"])
        tracks.add(method["track"])

    assert 3 <= len(artifact["methods_mapped"]) <= 5
    assert cited.issubset(set(artifact["arxiv_ids_cited"]))
    assert tracks == mod.REQUIRED_TRACKS
    assert {flag["candidate"] for flag in artifact["flagged_for_v451"]} == {
        "agent_authored_decision_need_targets",
        "action_prefix_latent_adapter",
        "latent_action_world_model_adapter",
    }
    assert all("flagged_for_v451" in flag["flag"] for flag in artifact["flagged_for_v451"])

    preconditions = artifact["preconditions_checked"]
    assert preconditions["a1_artifact_present"] is True
    assert preconditions["a1_fork_verdict_read"] is True
    assert preconditions["a1_fork_verdict"] == "INDUCER_CEILING_HARD"
    assert preconditions["a1b_artifact_present"] is True
    assert preconditions["a1b_inducer_ceiling_attribution"] == "METHOD_IS_CEILING"
    assert preconditions["branch_reason"] == mod.BRANCH_REASON
    assert preconditions["sweep_clusters_used"] is True
    assert preconditions["sweep_cluster_ids"] == [6]
    assert preconditions["sweep_semscholar_used"] is True
    assert preconditions["semantic_scholar_result"] == mod.SEMANTIC_SCHOLAR_RESULT
    assert preconditions["websearch_webfetch_used"] is True
    assert preconditions["top_source_count"] == 6
    assert preconditions["arxiv_http_200_verified_ids"] == [
        f"https://arxiv.org/abs/{source_id}" for source_id in sorted(mod.REQUIRED_SOURCE_IDS)
    ]
    assert preconditions["deep_research_invoked"] is False
    assert preconditions["retired_energy_classes_reingested"] is False
    assert preconditions["coverage_vocabulary_reingested"] is False
    assert preconditions["exploration_strategy_reingested"] is False
    assert preconditions["selection_ranking_reingested"] is False
    assert preconditions["perception_from_grid_reingested"] is False
    assert preconditions["research_conductor_modified"] is False
    assert preconditions["ops_docs_modified"] is False

    upstream = artifact["upstream_artifacts"]
    assert upstream["a1_artifact"] == mod.A1_ARTIFACT_RELATIVE_PATH
    assert upstream["a1b_artifact"] == mod.A1B_ARTIFACT_RELATIVE_PATH
    assert upstream["handoff_artifact"] == mod.HANDOFF_ARTIFACT_RELATIVE_PATH
    assert upstream["a1_fork_verdict"] == mod.AIMED_AT_FORK_VERDICT
    assert upstream["a1b_inducer_ceiling_attribution"] == mod.AIMED_AT_INDUCER_ATTRIBUTION
    assert upstream["handoff_honest_verdict"] == "success_sota_ingestion_v450_frontier_mapped"


@pytest.mark.parametrize(
    ("bad_artifact", "message"),
    [
        (_artifact() | {"honest_verdict": "complete_wrong"}, "honest_verdict"),
        (_artifact() | {"aimed_at_fork_verdict": "PLANNER_GAP"}, "aimed_at_fork_verdict"),
        (_artifact() | {"inference_substrate": "live_llm_inference"}, "inference_substrate"),
        (_artifact() | {"banned_channels_excluded": False}, "banned_channels_excluded"),
        (_artifact() | {"field_principles": {}}, "field_principles"),
        (_artifact() | {"arxiv_ids_cited": []}, "arxiv_ids_cited"),
        (_artifact() | {"citations": {}}, "citations"),
        (
            _artifact()
            | {
                "citations": mod.CITATIONS
                | {
                    "2606.25421": mod.CITATIONS["2606.25421"]
                    | {"url": "https://arxiv.org/abs/9999.99999"}
                }
            },
            "citation url",
        ),
        (
            _artifact()
            | {"citations": mod.CITATIONS | {"2606.26217": mod.CITATIONS["2606.26217"] | {"http_status": 404}}},
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
                    mod.DEFAULT_METHODS_MAPPED[0] | {"targets_fork_verdict": "INDUCER_CEILING_BEATABLE"},
                    *mod.DEFAULT_METHODS_MAPPED[1:],
                ]
            },
            "fork verdict",
        ),
        (
            _artifact()
            | {
                "methods_mapped": [
                    mod.DEFAULT_METHODS_MAPPED[0] | {"track": "energy_as_arc_lever"},
                    *mod.DEFAULT_METHODS_MAPPED[1:],
                ]
            },
            "retired",
        ),
        (_artifact() | {"flagged_for_v451": []}, "flagged_for_v451"),
        (
            _artifact() | {"flagged_for_v451": [{"candidate": "stale", "flag": "flagged_for_v450"}]},
            "stale",
        ),
        (
            _artifact()
            | {"preconditions_checked": dict(mod.DEFAULT_PRECONDITIONS_CHECKED) | {"deep_research_invoked": True}},
            "deep-research",
        ),
        (
            _artifact()
            | {
                "preconditions_checked": dict(mod.DEFAULT_PRECONDITIONS_CHECKED)
                | {"retired_energy_classes_reingested": True}
            },
            "retired energy",
        ),
        (
            _artifact()
            | {
                "preconditions_checked": dict(mod.DEFAULT_PRECONDITIONS_CHECKED)
                | {"a1b_inducer_ceiling_attribution": "LOCAL_MODEL_IS_CEILING"}
            },
            "A1b attribution",
        ),
    ],
)
def test_no_fabrication_validator_rejects_bad_v451_artifacts(
    bad_artifact: dict[str, object], message: str
) -> None:
    """SCENARIO-ARC-WMTE-4890-NO-FABRICATION: invalid claims fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_branch_helpers_cover_v451_redirects_for_req_arc_wmte_4890() -> None:
    """REQ-ARC-WMTE-4890: branch derivation follows A1 and A1b values."""

    assert (
        mod.select_ingestion_track({"fork_verdict": "INDUCER_CEILING_BEATABLE"}, {})
        == "tta_scaling_first_win_conversion"
    )
    assert mod.select_ingestion_track({"fork_verdict": "PLANNER_GAP"}, {}) == "neural_guided_planning_search"
    assert (
        mod.select_ingestion_track(
            {"fork_verdict": "INDUCER_CEILING_HARD"},
            {"inducer_ceiling_attribution": "LOCAL_MODEL_IS_CEILING"},
        )
        == "stronger_local_open_code_inducers"
    )
    assert (
        mod.select_ingestion_track(
            {"fork_verdict": "INDUCER_CEILING_HARD"},
            {"inducer_ceiling_attribution": "METHOD_IS_CEILING"},
        )
        == "alternative_world_model_representations"
    )
    assert (
        mod.select_ingestion_track(
            {"fork_verdict": "INDUCER_CEILING_HARD"},
            {"inducer_ceiling_attribution": "LOCAL_ALREADY_SUFFICIENT"},
        )
        == "local_already_sufficient_scale_conversion"
    )
    with pytest.raises(ValueError, match="unsupported A1 fork"):
        mod.select_ingestion_track({"fork_verdict": "GUIDANCE_WALL"}, {})


def test_scenario_arc_wmte_4890_research_sections_are_idempotent() -> None:
    """SCENARIO-ARC-WMTE-4890-V451-FRONTIER-MAPPED: markdown updates are stable."""

    studying_original = "# Research Studying\n\nOld body\n"
    studying_once = mod.update_research_studying_text(studying_original, _artifact())
    studying_twice = mod.update_research_studying_text(studying_once, _artifact())

    assert studying_once == studying_twice
    assert studying_once.count(mod.STUDYING_SECTION_START) == 1
    assert studying_once.count(mod.STUDYING_SECTION_END) == 1
    assert "SOTA -> .451 frontier mapping" in studying_once
    assert "METHOD_IS_CEILING" in studying_once
    assert "flagged_for_v451" in studying_once
    assert "arXiv:2606.25421" in studying_once
    assert "no solve claim" in studying_once
    mod.validate_research_studying_text(studying_once, _artifact())

    references_original = "# Research References\n\n## Existing Section\nOld body\n"
    references_once = mod.update_research_references_text(references_original, _artifact())
    references_twice = mod.update_research_references_text(references_once, _artifact())

    assert references_once == references_twice
    assert references_once.count(mod.REFERENCES_SECTION_START) == 1
    assert references_once.count(mod.REFERENCES_SECTION_END) == 1
    assert "Exp 4890 V451 frontier source set" in references_once
    assert "Agent-Authored World Modeling" in references_once
    assert "Fast LeWorldModel" in references_once
    mod.validate_research_references_text(references_once, _artifact())


def test_research_markdown_updates_reject_broken_existing_sections() -> None:
    """REQ-ARC-WMTE-4890: malformed existing note markers fail closed."""

    with pytest.raises(ValueError, match="missing end marker"):
        mod.update_research_studying_text("# x\n\n" + mod.STUDYING_SECTION_START + "\n", _artifact())
    with pytest.raises(ValueError, match="missing end marker"):
        mod.update_research_references_text("# x\n\n" + mod.REFERENCES_SECTION_START + "\n", _artifact())


def _write_upstream_fixtures(root: Path) -> None:
    (root / mod.A1_ARTIFACT_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / mod.A1_ARTIFACT_RELATIVE_PATH).write_text(
        json.dumps(
            {
                "honest_verdict": "complete_ttt_dynamics_no_value_lift_INDUCER_CEILING_HARD",
                "fork_verdict": "INDUCER_CEILING_HARD",
                "tta_changed_cell_value_accuracy_delta_median": -0.087781,
                "coverage_migration_count": 0,
                "n_games_measured": 9,
            }
        ),
        encoding="utf-8",
    )
    (root / mod.A1B_ARTIFACT_RELATIVE_PATH).write_text(
        json.dumps(
            {
                "honest_verdict": "complete_inducer_ceiling_neither_lane_lifts_method_is_ceiling",
                "inducer_ceiling_attribution": "METHOD_IS_CEILING",
                "reference_lane_value_accuracy_delta": {"median": -1.0, "ci95": [-1.0, 0.0]},
                "local_lane_value_accuracy_delta": {"median": -0.967828, "ci95": [-1.0, -0.087781]},
            }
        ),
        encoding="utf-8",
    )
    (root / mod.HANDOFF_ARTIFACT_RELATIVE_PATH).write_text(
        json.dumps(
            {
                "honest_verdict": "success_sota_ingestion_v450_frontier_mapped",
                "aimed_at_fork_verdict": "INDUCER_CEILING",
                "flagged_for_v450": [
                    {"candidate": "agent_authored_world_model_targets"},
                    {"candidate": "action_prefix_world_model_adapter"},
                ],
            }
        ),
        encoding="utf-8",
    )


def test_write_outputs_handles_blocked_a1_for_scenario_arc_wmte_4890(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4890-BLOCKED-A1: missing A1 writes a blocked deliverable."""

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    studying_path = tmp_path / mod.STUDYING_RELATIVE_PATH
    references_path = tmp_path / mod.REFERENCES_RELATIVE_PATH
    studying_path.write_text("# Research Studying\n\n", encoding="utf-8")
    references_path.write_text("# Research References\n\n", encoding="utf-8")

    artifact = mod.write_outputs(
        artifact_path=artifact_path,
        studying_path=studying_path,
        references_path=references_path,
        repo_root=tmp_path,
    )

    assert artifact["honest_verdict"] == mod.BLOCKED_A1_VERDICT
    assert artifact["methods_mapped"] == []
    assert artifact["banned_channels_excluded"] is True
    assert artifact["preconditions_checked"]["a1_artifact_present"] is False
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact


def test_write_outputs_is_idempotent_for_scenario_arc_wmte_4890(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4890-V451-FRONTIER-MAPPED: writer emits stable outputs."""

    _write_upstream_fixtures(tmp_path)
    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    studying_path = tmp_path / mod.STUDYING_RELATIVE_PATH
    references_path = tmp_path / mod.REFERENCES_RELATIVE_PATH
    studying_path.write_text("# Research Studying\n\n", encoding="utf-8")
    references_path.write_text("# Research References\n\n", encoding="utf-8")

    artifact = mod.write_outputs(
        artifact_path=artifact_path,
        studying_path=studying_path,
        references_path=references_path,
        repo_root=tmp_path,
    )
    second_artifact = mod.write_outputs(
        artifact_path=artifact_path,
        studying_path=studying_path,
        references_path=references_path,
        repo_root=tmp_path,
    )

    assert second_artifact == artifact
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    mod.validate_research_studying_text(studying_path.read_text(encoding="utf-8"), artifact)
    mod.validate_research_references_text(references_path.read_text(encoding="utf-8"), artifact)


def test_main_writes_deliverables_for_req_arc_wmte_4890(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-ARC-WMTE-4890: direct module execution writes default outputs."""

    _write_upstream_fixtures(tmp_path)
    (tmp_path / mod.STUDYING_RELATIVE_PATH).write_text("# Research Studying\n\n", encoding="utf-8")
    (tmp_path / mod.REFERENCES_RELATIVE_PATH).write_text("# Research References\n\n", encoding="utf-8")
    monkeypatch.setenv("CARNOT_EXP4890_ROOT", str(tmp_path))

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


def test_deliverable_files_validate_for_req_arc_wmte_4890() -> None:
    """REQ-ARC-WMTE-4890: committed deliverables satisfy the schema."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    studying = STUDYING_PATH.read_text(encoding="utf-8")
    references = REFERENCES_PATH.read_text(encoding="utf-8")

    mod.validate_artifact(artifact)
    mod.validate_research_studying_text(studying, artifact)
    mod.validate_research_references_text(references, artifact)
    assert artifact["honest_verdict"] == mod.HONEST_VERDICT
    assert artifact["flagged_for_v451"] == mod.FLAGGED_FOR_V451
