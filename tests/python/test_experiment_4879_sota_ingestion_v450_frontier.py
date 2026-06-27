"""Tests for Exp 4879 V450 SOTA ingestion.

Spec refs: REQ-ARC-WMTE-4879,
SCENARIO-ARC-WMTE-4879-V450-FRONTIER-MAPPED,
SCENARIO-ARC-WMTE-4879-NO-FABRICATION.
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_4879_sota_ingestion_v450_frontier as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"
ARTIFACT_PATH = REPO / mod.RESULT_RELATIVE_PATH
STUDYING_PATH = REPO / mod.STUDYING_RELATIVE_PATH
REFERENCES_PATH = REPO / mod.REFERENCES_RELATIVE_PATH


def _artifact() -> dict[str, object]:
    return mod.build_artifact(upstream_context=mod.DEFAULT_UPSTREAM_CONTEXT)


def _spec_section() -> str:
    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-ARC-WMTE-4879")
    end = spec.index("### REQ-ARC-WMTE-4852", start)
    return spec[start:end]


def test_req_arc_wmte_4879_spec_declares_v450_contract() -> None:
    """REQ-ARC-WMTE-4879: OpenSpec declares the Exp 4879 artifact contract."""

    section = _spec_section()

    assert "REQ-ARC-WMTE-4879" in section
    assert "SCENARIO-ARC-WMTE-4879-V450-FRONTIER-MAPPED" in section
    assert "SCENARIO-ARC-WMTE-4879-NO-FABRICATION" in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert mod.A1_ARTIFACT_RELATIVE_PATH in section
    assert mod.A1B_ARTIFACT_RELATIVE_PATH in section
    assert mod.HANDOFF_ARTIFACT_RELATIVE_PATH in section
    assert mod.HONEST_VERDICT in section
    assert mod.INFERENCE_SUBSTRATE in section
    assert "A1b" in section
    assert "null held-out CEGIS delta" in section
    assert "test-time dynamics" in section
    assert "local-open-code inducer" in section
    assert "positive-control caveat" in section
    assert "/deep-research" in section
    assert "scripts/research_conductor.py" in section
    for field, principle in mod.REQUIRED_USER_FIELD_PRINCIPLES.items():
        assert field in section
        assert principle["principle"] in section
    for source_id in mod.REQUIRED_SOURCE_IDS:
        assert source_id in section


def test_req_arc_wmte_4879_artifact_maps_cegis_null_to_next_inducers() -> None:
    """REQ-ARC-WMTE-4879: artifact maps A1/A1b residuals to V450 methods."""

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

    cited: set[str] = set()
    tracks: set[str] = set()
    for method in artifact["methods_mapped"]:
        assert set(method) == mod.REQUIRED_METHOD_FIELDS
        assert method["source_ids"]
        assert set(method["source_ids"]).issubset(mod.REQUIRED_SOURCE_IDS)
        assert method["maps_to_frontier"] == ".450"
        assert method["targets_fork_verdict"] == mod.AIMED_AT_FORK_VERDICT
        assert "A1b" in method["a1b_result_fit"]
        assert "arXiv:" in method["evidence"]
        assert method["experiment_graft"]
        assert method["validation_gate"]
        assert method["sovereignty_note"]
        assert method["fails_when"]
        assert "flagged_for_v450" in method["roadmap_candidate"]
        assert "cegis_world_model_refinement_loop" not in method["roadmap_candidate"]
        cited.update(method["source_ids"])
        tracks.add(method["track"])

    assert 3 <= len(artifact["methods_mapped"]) <= 5
    assert cited.issubset(set(artifact["arxiv_ids_cited"]))
    assert tracks == mod.REQUIRED_TRACKS
    assert {flag["candidate"] for flag in artifact["flagged_for_v450"]} == {
        "test_time_dynamics_adaptation_loop",
        "family_b_vs_local_open_code_inducer_ab",
        "agent_authored_world_model_targets",
    }
    assert all("flagged_for_v450" in flag["flag"] for flag in artifact["flagged_for_v450"])
    assert all("cegis_world_model_refinement_loop" not in flag["flag"] for flag in artifact["flagged_for_v450"])

    preconditions = artifact["preconditions_checked"]
    assert preconditions["research_studying_present"] is True
    assert preconditions["research_references_present"] is True
    assert preconditions["a1_artifact_present"] is True
    assert preconditions["a1b_artifact_present"] is True
    assert preconditions["handoff_artifact_present"] is True
    assert preconditions["a1_fork_verdict_read"] is True
    assert preconditions["a1_source_fork_verdict"] is None
    assert preconditions["a1_computed_fork_verdict"] == "INDUCER_CEILING"
    assert preconditions["a1_genuinely_diagnostic"] is False
    assert preconditions["a1_positive_control_caveat"] is True
    assert preconditions["a1b_delta_median"] == 0.0
    assert preconditions["a1b_delta_ci95"] == [0.0, 0.0]
    assert preconditions["a1b_cegis_moved_accuracy"] is False
    assert preconditions["current_cegis_promoted"] is False
    assert preconditions["sweep_clusters_used"] is True
    assert preconditions["sweep_cluster_ids"] == [6]
    assert preconditions["sweep_semscholar_used"] is True
    assert preconditions["semantic_scholar_unique_arxiv_ids"] == mod.SEMANTIC_SCHOLAR_UNIQUE_ARXIV_IDS
    assert preconditions["websearch_webfetch_used"] is True
    assert preconditions["top_source_count"] == 8
    assert preconditions["deep_research_invoked"] is False
    assert preconditions["retired_coverage_classes_reingested"] is False
    assert preconditions["exploration_strategy_reingested"] is False
    assert preconditions["energy_classes_reingested"] is False
    assert preconditions["model_load"] is False
    assert preconditions["training_launched"] is False
    assert preconditions["leaderboard_submission"] is False
    assert preconditions["solve_claim_made"] is False
    assert preconditions["research_conductor_modified"] is False
    assert preconditions["ops_docs_modified"] is False

    upstream = artifact["upstream_artifacts"]
    assert upstream["a1_artifact"] == mod.A1_ARTIFACT_RELATIVE_PATH
    assert upstream["a1b_artifact"] == mod.A1B_ARTIFACT_RELATIVE_PATH
    assert upstream["handoff_artifact"] == mod.HANDOFF_ARTIFACT_RELATIVE_PATH
    assert upstream["a1_source_fork_verdict"] is None
    assert upstream["a1_computed_fork_verdict"] == mod.AIMED_AT_FORK_VERDICT
    assert upstream["a1b_cegis_moved_accuracy"] is False
    assert upstream["a1b_delta_median"] == 0.0
    assert upstream["carried_forward_from_4868"] == [
        "test_time_world_model_adaptation_loop",
        "family_b_executable_world_model_inducer_ladder",
        "local_open_code_inducer",
    ]
    assert "cegis_world_model_refinement_loop" in upstream["not_carried_forward_from_4868"]

    note = artifact["sota_to_experiment_mapping_note"]
    assert set(note) == mod.REQUIRED_MAPPING_NOTE_FIELDS
    assert note["terminal_success"] == mod.HONEST_VERDICT
    assert note["root_cause"] == "world-model inducer quality after nulled CEGIS"
    assert "A1b CEGIS delta was 0.0" in note["summary"]
    assert "positive-control" in note["a1_caveat"]
    assert "CEGIS" in note["not_carried_forward"]


@pytest.mark.parametrize(
    ("bad_artifact", "message"),
    [
        (_artifact() | {"honest_verdict": "complete_wrong"}, "honest_verdict"),
        (_artifact() | {"inference_substrate": "live_llm"}, "inference_substrate"),
        (_artifact() | {"aimed_at_fork_verdict": "PLANNER_GAP"}, "aimed_at_fork_verdict"),
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
                    mod.DEFAULT_METHODS_MAPPED[0] | {"targets_fork_verdict": "GUIDANCE_WALL"},
                    *mod.DEFAULT_METHODS_MAPPED[1:],
                ]
            },
            "fork verdict",
        ),
        (
            _artifact()
            | {
                "methods_mapped": [
                    mod.DEFAULT_METHODS_MAPPED[0]
                    | {"roadmap_candidate": "flagged_for_v450: cegis_world_model_refinement_loop"},
                    *mod.DEFAULT_METHODS_MAPPED[1:],
                ]
            },
            "nulled CEGIS",
        ),
        (_artifact() | {"flagged_for_v450": []}, "flagged_for_v450"),
        (
            _artifact()
            | {"flagged_for_v450": [{"candidate": "stale", "flag": "flagged_for_v449"}]},
            "stale",
        ),
        (
            _artifact()
            | {"flagged_for_v450": [{"candidate": "cegis", "flag": "flagged_for_v450: cegis_world_model_refinement_loop"}]},
            "nulled CEGIS",
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
                | {"retired_coverage_classes_reingested": True}
            },
            "retired coverage",
        ),
        (
            _artifact()
            | {
                "preconditions_checked": dict(mod.DEFAULT_PRECONDITIONS_CHECKED)
                | {"a1b_cegis_moved_accuracy": True}
            },
            "A1b CEGIS null",
        ),
        (
            _artifact()
            | {
                "preconditions_checked": dict(mod.DEFAULT_PRECONDITIONS_CHECKED)
                | {"research_conductor_modified": True}
            },
            "research_conductor",
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
                | {"a1b_cegis_moved_accuracy": True}
            },
            "upstream A1b",
        ),
        (
            _artifact()
            | {
                "sota_to_experiment_mapping_note": dict(mod.DEFAULT_MAPPING_NOTE)
                | {"source_ids": []}
            },
            "mapping note",
        ),
    ],
)
def test_no_fabrication_validator_rejects_bad_v450_artifacts(
    bad_artifact: dict[str, object], message: str
) -> None:
    """SCENARIO-ARC-WMTE-4879-NO-FABRICATION: invalid claims fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_validate_artifact_rejects_exact_schema_violations() -> None:
    """REQ-ARC-WMTE-4879: top-level and method schemas are exact."""

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
    with pytest.raises(ValueError, match="cluster"):
        mod.validate_artifact(bad_sweep)


def test_upstream_fork_helpers_cover_defensive_branches_for_req_arc_wmte_4879() -> None:
    """REQ-ARC-WMTE-4879: A1 branch derivation handles reported and computed forks."""

    assert mod._as_float("not-a-number", default=1.25) == 1.25
    assert mod._ci_excludes_zero("bad-ci") is False
    assert mod._ci_excludes_zero([0.1, 0.2]) is True
    assert mod._ci_excludes_zero([-0.2, -0.1]) is True
    assert mod.infer_computed_fork_verdict({"fork_verdict": "PLANNER_GAP"}) == "PLANNER_GAP"
    assert (
        mod.infer_computed_fork_verdict(
            {
                "fork_verdict": None,
                "median_engine_heldout_accuracy": 0.9,
                "coverage_migration_count": 2,
                "induce_plan_config": {"high_accuracy_threshold": 0.5},
            }
        )
        == "GUIDANCE_WALL"
    )
    assert (
        mod.infer_computed_fork_verdict(
            {
                "fork_verdict": None,
                "median_engine_heldout_accuracy": 0.9,
                "coverage_migration_count": 0,
                "induce_plan_config": {"high_accuracy_threshold": 0.5},
            }
        )
        == "PLANNER_GAP"
    )


def test_scenario_arc_wmte_4879_research_studying_section_is_idempotent() -> None:
    """SCENARIO-ARC-WMTE-4879-V450-FRONTIER-MAPPED: studying update is stable."""

    original = "# Research Studying\n\nOld body\n"
    once = mod.update_research_studying_text(original, _artifact())
    twice = mod.update_research_studying_text(once, _artifact())

    assert once == twice
    assert once.count(mod.STUDYING_SECTION_START) == 1
    assert once.count(mod.STUDYING_SECTION_END) == 1
    mod.validate_research_studying_text(once, _artifact())
    assert "SOTA -> .450 frontier mapping" in once
    assert "flagged_for_v450" in once
    assert "arXiv:2506.02918" in once
    assert "arXiv:2606.25421" in once
    assert "INDUCER_CEILING" in once
    assert "A1b CEGIS delta was 0.0" in once
    assert "no solve claim" in once

    later_section = "# Research Studying\n\n<!-- EXP0000-OLDER-START -->\nOld section\n"
    inserted = mod.update_research_studying_text(later_section, _artifact())
    refreshed = mod.update_research_studying_text(inserted, _artifact())

    assert refreshed == inserted
    assert inserted.index(mod.STUDYING_SECTION_START) < inserted.index("<!-- EXP0000-OLDER-START -->")


def test_scenario_arc_wmte_4879_research_references_section_is_idempotent() -> None:
    """SCENARIO-ARC-WMTE-4879-V450-FRONTIER-MAPPED: references update is stable."""

    original = "# Research References\n\n## Existing Section\nOld body\n"
    once = mod.update_research_references_text(original, _artifact())
    twice = mod.update_research_references_text(once, _artifact())

    assert once == twice
    assert once.count(mod.REFERENCES_SECTION_START) == 1
    assert once.count(mod.REFERENCES_SECTION_END) == 1
    mod.validate_research_references_text(once, _artifact())
    assert "Exp 4879 V450 frontier source set" in once
    assert "World Modelling Improves Language Model Agents" in once
    assert "Beyond Next-Observation Prediction" in once
    assert "Fast LeWorldModel" in once
    assert "CodeGen" in once


def test_research_markdown_updates_reject_broken_existing_sections() -> None:
    """REQ-ARC-WMTE-4879: malformed existing note markers fail closed."""

    with pytest.raises(ValueError, match="missing end marker"):
        mod.update_research_studying_text("# x\n\n" + mod.STUDYING_SECTION_START + "\n", _artifact())
    with pytest.raises(ValueError, match="missing end marker"):
        mod.update_research_references_text("# x\n\n" + mod.REFERENCES_SECTION_START + "\n", _artifact())


def _write_upstream_fixtures(root: Path) -> None:
    (root / mod.A1_ARTIFACT_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / mod.A1_ARTIFACT_RELATIVE_PATH).write_text(
        json.dumps(
            {
                "fork_verdict": None,
                "honest_verdict": "complete_generation_wall_fork_probe_retired_positive_control_failed",
                "median_engine_heldout_accuracy": 0.0,
                "coverage_migration_count": 0,
                "positive_control_migrated": False,
                "n_games_measured": 9,
                "induce_plan_config": {"high_accuracy_threshold": 0.5},
            }
        ),
        encoding="utf-8",
    )
    (root / mod.A1B_ARTIFACT_RELATIVE_PATH).write_text(
        json.dumps(
            {
                "honest_verdict": "complete_cegis_no_heldout_accuracy_lift_residual_positive_control_failed",
                "cegis_heldout_accuracy_delta_median": 0.0,
                "cegis_heldout_accuracy_delta_ci95": [0.0, 0.0],
                "positive_control_passed": False,
                "delta_on_truly_heldout_split": True,
            }
        ),
        encoding="utf-8",
    )
    (root / mod.HANDOFF_ARTIFACT_RELATIVE_PATH).write_text(
        json.dumps(
            {
                "honest_verdict": "success_sota_ingestion_v449_frontier_mapped",
                "aimed_at_fork_verdict": "INDUCER_CEILING",
                "flagged_for_v449": [
                    {"candidate": "test_time_world_model_adaptation_loop"},
                    {"candidate": "family_b_executable_world_model_inducer_ladder"},
                    {"candidate": "cegis_world_model_refinement_loop"},
                ],
            }
        ),
        encoding="utf-8",
    )


def test_write_outputs_is_idempotent_for_scenario_arc_wmte_4879(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4879-V450-FRONTIER-MAPPED: writer emits stable outputs."""

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
    mod.validate_research_studying_text(studying_path.read_text(encoding="utf-8"), artifact)
    mod.validate_research_references_text(references_path.read_text(encoding="utf-8"), artifact)


def test_main_writes_deliverables_for_req_arc_wmte_4879(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-ARC-WMTE-4879: direct module execution writes default outputs."""

    _write_upstream_fixtures(tmp_path)
    (tmp_path / mod.STUDYING_RELATIVE_PATH).write_text("# Research Studying\n\n", encoding="utf-8")
    (tmp_path / mod.REFERENCES_RELATIVE_PATH).write_text("# Research References\n\n", encoding="utf-8")
    monkeypatch.setenv("CARNOT_EXP4879_ROOT", str(tmp_path))

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


def test_deliverable_files_validate_for_req_arc_wmte_4879() -> None:
    """REQ-ARC-WMTE-4879: committed deliverables satisfy the schema."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    studying = STUDYING_PATH.read_text(encoding="utf-8")
    references = REFERENCES_PATH.read_text(encoding="utf-8")

    mod.validate_artifact(artifact)
    mod.validate_research_studying_text(studying, artifact)
    mod.validate_research_references_text(references, artifact)
    assert artifact["honest_verdict"] == mod.HONEST_VERDICT
    assert artifact["flagged_for_v450"] == mod.FLAGGED_FOR_V450
