"""Tests for Exp 4911 V453 SOTA ingestion.

Spec refs: REQ-ARC-WMTE-4911,
SCENARIO-ARC-WMTE-4911-V453-WALL-AND-PIVOT-MAPPED,
SCENARIO-ARC-WMTE-4911-BLOCKED-UPSTREAM,
SCENARIO-ARC-WMTE-4911-NO-FABRICATION.
"""

from __future__ import annotations

import copy
import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_4911_sota_ingestion_v453_frontier as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"
ARTIFACT_PATH = REPO / mod.RESULT_RELATIVE_PATH
STUDYING_PATH = REPO / mod.STUDYING_RELATIVE_PATH
REFERENCES_PATH = REPO / mod.REFERENCES_RELATIVE_PATH


def _artifact() -> dict[str, object]:
    return mod.build_artifact(upstream_context=mod.DEFAULT_UPSTREAM_CONTEXT)


def _bad_artifact(**updates: object) -> dict[str, object]:
    artifact = copy.deepcopy(_artifact())
    artifact.update(updates)
    return artifact


def _spec_section() -> str:
    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-ARC-WMTE-4911")
    end = spec.index("## Implementation Status", start)
    return spec[start:end]


def test_req_arc_wmte_4911_spec_declares_v453_contract() -> None:
    """REQ-ARC-WMTE-4911: OpenSpec declares the Exp 4911 artifact contract."""

    section = _spec_section()

    assert "REQ-ARC-WMTE-4911" in section
    assert "SCENARIO-ARC-WMTE-4911-V453-WALL-AND-PIVOT-MAPPED" in section
    assert "SCENARIO-ARC-WMTE-4911-BLOCKED-UPSTREAM" in section
    assert "SCENARIO-ARC-WMTE-4911-NO-FABRICATION" in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert mod.A1_ARTIFACT_RELATIVE_PATH in section
    assert mod.A1B_ARTIFACT_RELATIVE_PATH in section
    assert mod.HONEST_VERDICT in section
    assert mod.BLOCKED_VERDICT in section
    assert mod.AIMED_AT_FORK_VERDICT in section
    assert mod.A1B_NULL_VERDICT in section
    assert "verifier-moat / oracle-distinct" in section
    assert "/deep-research" in section
    assert "scripts/research_conductor.py" in section
    for field, principle in mod.REQUIRED_USER_FIELD_PRINCIPLES.items():
        assert field in section
        assert principle["principle"] in section
    for source_id in mod.REQUIRED_FETCHED_SOURCE_IDS:
        assert source_id in section


def test_req_arc_wmte_4911_artifact_maps_wall_and_post_sprint_pivot() -> None:
    """REQ-ARC-WMTE-4911: artifact maps the wall branch and verifier pivot."""

    artifact = _artifact()

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["honest_verdict"] == mod.HONEST_VERDICT
    assert artifact["aimed_at_fork_verdict"] == mod.AIMED_AT_FORK_VERDICT
    assert artifact["a1b_fork_verdict"] == mod.A1B_NULL_VERDICT
    assert artifact["selected_branch"] == mod.SELECTED_BRANCH
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] == mod.DURATION_S
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["reproducibility_checksum"] == mod.REPRODUCIBILITY_CHECKSUM
    assert artifact["arxiv_ids_cited"] == sorted(mod.REQUIRED_FETCHED_SOURCE_IDS)
    assert set(artifact["citations"]) == mod.REQUIRED_FETCHED_SOURCE_IDS
    assert set(artifact["field_principles"]) == set(mod.REQUIRED_PRINCIPLE_FIELDS)
    assert artifact["banned_channels_excluded"] is True

    wall_sources: set[str] = set()
    wall_tracks: set[str] = set()
    for method in artifact["methods_mapped"]:
        assert set(method) == mod.REQUIRED_METHOD_FIELDS
        assert method["source_ids"]
        assert set(method["source_ids"]).issubset(mod.REQUIRED_FETCHED_SOURCE_IDS)
        assert method["maps_to_frontier"] == ".453"
        assert method["targets_a1_fork_verdict"] == mod.AIMED_AT_FORK_VERDICT
        assert method["targets_a1b_fork_verdict"] == mod.A1B_NULL_VERDICT
        assert "four representations" in method["a1_a1b_result_fit"]
        assert "arXiv:" in method["evidence"]
        assert method["experiment_graft"]
        assert method["fails_when"]
        assert "flagged_for_v453" in method["roadmap_candidate"]
        assert method["nulled_class_reingested"] is False
        wall_sources.update(method["source_ids"])
        wall_tracks.add(method["track"])

    assert 3 <= len(artifact["methods_mapped"]) <= 5
    assert wall_sources == mod.REQUIRED_WALL_SOURCE_IDS
    assert wall_tracks == mod.REQUIRED_WALL_TRACKS

    pivot_sources: set[str] = set()
    pivot_tracks: set[str] = set()
    for method in artifact["post_sprint_pivot_methods"]:
        assert set(method) == mod.REQUIRED_PIVOT_METHOD_FIELDS
        assert method["source_ids"]
        assert set(method["source_ids"]).issubset(mod.REQUIRED_FETCHED_SOURCE_IDS)
        assert method["maps_to_track"] == "post_sprint_verifier_moat"
        assert "north-star §1/§5" in method["north_star_fit"]
        assert method["self_consistency_saturated"] is False
        assert method["oracle_distinct_verifier"] is True
        assert method["experiment_graft"]
        assert method["fails_when"]
        pivot_sources.update(method["source_ids"])
        pivot_tracks.add(method["track"])

    assert 3 <= len(artifact["post_sprint_pivot_methods"]) <= 5
    assert pivot_sources == mod.REQUIRED_PIVOT_SOURCE_IDS
    assert pivot_tracks == mod.REQUIRED_PIVOT_TRACKS
    assert {flag["candidate"] for flag in artifact["flagged_for_v453"]} == {
        "causal_state_abstraction_wall_diagnostic",
        "distributional_energy_verifier_pivot",
        "tool_aware_science_prm_pivot",
    }

    preconditions = artifact["preconditions_checked"]
    assert preconditions["a1_artifact_present"] is True
    assert preconditions["a1b_artifact_present"] is True
    assert preconditions["a1_fork_verdict"] == mod.AIMED_AT_FORK_VERDICT
    assert preconditions["a1b_fork_verdict"] == mod.A1B_NULL_VERDICT
    assert preconditions["selected_branch"] == mod.SELECTED_BRANCH
    assert preconditions["sweep_clusters_used"] is True
    assert preconditions["sweep_cluster_ids"] == [6, 0, 1]
    assert preconditions["sweep_semscholar_used"] is True
    assert preconditions["websearch_webfetch_used"] is True
    assert preconditions["top_source_count"] == 8
    assert preconditions["arxiv_http_200_verified_ids"] == [
        f"https://arxiv.org/abs/{source_id}" for source_id in sorted(mod.REQUIRED_FETCHED_SOURCE_IDS)
    ]
    assert preconditions["deep_research_invoked"] is False
    assert preconditions["research_conductor_modified"] is False
    assert preconditions["ops_docs_modified"] is False
    for banned_key in mod.BANNED_PRECONDITION_KEYS:
        assert preconditions[banned_key] is False


@pytest.mark.parametrize(
    ("bad_artifact", "message"),
    [
        (_bad_artifact(honest_verdict="complete_wrong"), "honest_verdict"),
        (_bad_artifact(aimed_at_fork_verdict="SEARCH_BUDGET_BOUND"), "aimed_at_fork_verdict"),
        (_bad_artifact(a1b_fork_verdict="REPRESENTATION_MATTERS"), "a1b_fork_verdict"),
        (_bad_artifact(selected_branch="search_budget_bound"), "selected_branch"),
        (_bad_artifact(inference_substrate="live_llm_inference"), "inference_substrate"),
        (_bad_artifact(banned_channels_excluded=False), "banned_channels_excluded"),
        (_bad_artifact(field_principles={}), "field_principles"),
        (_bad_artifact(arxiv_ids_cited=[]), "arxiv_ids_cited"),
        (_bad_artifact(citations={}), "citations"),
        (
            _bad_artifact(
                citations=mod.CITATIONS
                | {"2401.12497": mod.CITATIONS["2401.12497"] | {"http_status": 404}}
            ),
            "http_status",
        ),
        (
            _bad_artifact(
                citations=mod.CITATIONS
                | {"2505.02074": mod.CITATIONS["2505.02074"] | {"url": "https://arxiv.org/abs/9999.99999"}}
            ),
            "citation url",
        ),
        (_bad_artifact(methods_mapped=mod.DEFAULT_METHODS_MAPPED[:2]), "three to five"),
        (
            _bad_artifact(
                methods_mapped=[
                    mod.DEFAULT_METHODS_MAPPED[0] | {"source_ids": ["9999.99999"]},
                    *mod.DEFAULT_METHODS_MAPPED[1:],
                ]
            ),
            "verified citations",
        ),
        (
            _bad_artifact(
                methods_mapped=[
                    mod.DEFAULT_METHODS_MAPPED[0] | {"track": "decision_need_targets"},
                    *mod.DEFAULT_METHODS_MAPPED[1:],
                ]
            ),
            "nulled",
        ),
        (_bad_artifact(post_sprint_pivot_methods=[]), "post_sprint_pivot_methods"),
        (
            _bad_artifact(
                post_sprint_pivot_methods=[
                    mod.DEFAULT_POST_SPRINT_PIVOT_METHODS[0] | {"self_consistency_saturated": True},
                    *mod.DEFAULT_POST_SPRINT_PIVOT_METHODS[1:],
                ]
            ),
            "non-saturated",
        ),
        (_bad_artifact(flagged_for_v453=[]), "flagged_for_v453"),
        (
            _bad_artifact(flagged_for_v453=[{"candidate": "stale", "flag": "flagged_for_v452"}]),
            "stale",
        ),
        (
            _bad_artifact(
                preconditions_checked=dict(mod.DEFAULT_PRECONDITIONS_CHECKED)
                | {"deep_research_invoked": True}
            ),
            "deep-research",
        ),
        (
            _bad_artifact(
                preconditions_checked=dict(mod.DEFAULT_PRECONDITIONS_CHECKED)
                | {"decision_need_targets_reingested": True}
            ),
            "decision-need",
        ),
    ],
)
def test_no_fabrication_validator_rejects_bad_v453_artifacts(
    bad_artifact: dict[str, object], message: str
) -> None:
    """SCENARIO-ARC-WMTE-4911-NO-FABRICATION: invalid claims fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_branch_helpers_cover_req_arc_wmte_4911_redirects() -> None:
    """REQ-ARC-WMTE-4911: branch derivation follows A1 and A1b values."""

    assert (
        mod.select_ingestion_branch({"fork_verdict": "ENV_GROUNDED_SEARCH_UNLOCKS_FIRST_WIN"}, {})
        == "scale_env_grounded_first_win_search"
    )
    assert (
        mod.select_ingestion_branch({"fork_verdict": "SEARCH_BUDGET_BOUND"}, {})
        == "cut_env_grounding_action_cost"
    )
    assert (
        mod.select_ingestion_branch(
            {"fork_verdict": "WALL_DEEPER_THAN_VALUE_PREDICTION"},
            {"fork_verdict": "VALUE_GAP_REPRESENTATION_INVARIANT_4_CLASSES"},
        )
        == mod.SELECTED_BRANCH
    )
    with pytest.raises(ValueError, match="unsupported A1/A1b fork"):
        mod.select_ingestion_branch({"fork_verdict": "WALL_DEEPER_THAN_VALUE_PREDICTION"}, {})


def test_research_sections_are_idempotent_for_scenario_arc_wmte_4911() -> None:
    """SCENARIO-ARC-WMTE-4911-V453-WALL-AND-PIVOT-MAPPED: notes are stable."""

    studying_original = "# Research Studying\n\nOld body\n"
    studying_once = mod.update_research_studying_text(studying_original, _artifact())
    studying_twice = mod.update_research_studying_text(studying_once, _artifact())

    assert studying_once == studying_twice
    assert studying_once.count(mod.STUDYING_SECTION_START) == 1
    assert studying_once.count(mod.STUDYING_SECTION_END) == 1
    assert "Exp 4911 - .453 wall and verifier-pivot SOTA ingestion - INGESTED" in studying_once
    assert mod.AIMED_AT_FORK_VERDICT in studying_once
    assert "VALUE_GAP_REPRESENTATION_INVARIANT_4_CLASSES" in studying_once
    assert "flagged_for_v453" in studying_once
    assert "post-sprint verifier-moat pivot" in studying_once
    assert "no solve claim" in studying_once
    mod.validate_research_studying_text(studying_once, _artifact())

    references_original = "# Research References\n\n## Existing Section\nOld body\n"
    references_once = mod.update_research_references_text(references_original, _artifact())
    references_twice = mod.update_research_references_text(references_once, _artifact())

    assert references_once == references_twice
    assert references_once.count(mod.REFERENCES_SECTION_START) == 1
    assert references_once.count(mod.REFERENCES_SECTION_END) == 1
    assert "Exp 4911 V453 wall and verifier-pivot source set" in references_once
    assert "Causal State Abstractions" in references_once
    assert "Distributional Energy-Based Models" in references_once
    mod.validate_research_references_text(references_once, _artifact())


def test_research_markdown_updates_reject_broken_existing_sections() -> None:
    """REQ-ARC-WMTE-4911: malformed existing note markers fail closed."""

    with pytest.raises(ValueError, match="missing end marker"):
        mod.update_research_studying_text("# x\n\n" + mod.STUDYING_SECTION_START + "\n", _artifact())
    with pytest.raises(ValueError, match="missing end marker"):
        mod.update_research_references_text("# x\n\n" + mod.REFERENCES_SECTION_START + "\n", _artifact())


def _write_upstream_fixtures(root: Path) -> None:
    (root / mod.A1_ARTIFACT_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / mod.A1_ARTIFACT_RELATIVE_PATH).write_text(
        json.dumps(
            {
                "honest_verdict": "complete_env_grounded_search_no_first_win_lift_WALL_DEEPER_THAN_VALUE_PREDICTION",
                "fork_verdict": "WALL_DEEPER_THAN_VALUE_PREDICTION",
                "value_grounded_first_win_delta_median": -0.04,
                "value_grounded_first_win_delta_ci95": [-0.04, 0.0],
                "coverage_migration_count": 0,
            }
        ),
        encoding="utf-8",
    )
    (root / mod.A1B_ARTIFACT_RELATIVE_PATH).write_text(
        json.dumps(
            {
                "honest_verdict": "complete_latent_action_no_value_lift_representation_invariant_4_classes",
                "fork_verdict": "VALUE_GAP_REPRESENTATION_INVARIANT_4_CLASSES",
                "latent_action_value_accuracy_delta_median": -0.103162,
                "latent_action_value_accuracy_delta_ci95": [-0.231195, 0.025266],
            }
        ),
        encoding="utf-8",
    )


def test_write_outputs_handles_blocked_upstream_for_scenario_arc_wmte_4911(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4911-BLOCKED-UPSTREAM: missing A1 writes blocked deliverable."""

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

    assert artifact["honest_verdict"] == mod.BLOCKED_VERDICT
    assert artifact["methods_mapped"] == []
    assert artifact["post_sprint_pivot_methods"] == []
    assert artifact["banned_channels_excluded"] is True
    assert artifact["preconditions_checked"]["a1_artifact_present"] is False
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact


def test_write_outputs_is_idempotent_for_scenario_arc_wmte_4911(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4911-V453-WALL-AND-PIVOT-MAPPED: writer is stable."""

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


def test_main_writes_deliverables_for_req_arc_wmte_4911(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-ARC-WMTE-4911: direct module execution writes default outputs."""

    _write_upstream_fixtures(tmp_path)
    (tmp_path / mod.STUDYING_RELATIVE_PATH).write_text("# Research Studying\n\n", encoding="utf-8")
    (tmp_path / mod.REFERENCES_RELATIVE_PATH).write_text("# Research References\n\n", encoding="utf-8")
    monkeypatch.setenv("CARNOT_EXP4911_ROOT", str(tmp_path))

    with pytest.raises(SystemExit) as module_exit:
        runpy.run_path(str(Path(mod.__file__)), run_name="__main__")

    assert module_exit.value.code == 0
    assert capsys.readouterr().out.strip() == mod.HONEST_VERDICT
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    mod.validate_artifact(written)
    mod.validate_research_studying_text((tmp_path / mod.STUDYING_RELATIVE_PATH).read_text(encoding="utf-8"), written)
    mod.validate_research_references_text(
        (tmp_path / mod.REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8"),
        written,
    )


def test_deliverable_files_validate_for_req_arc_wmte_4911() -> None:
    """REQ-ARC-WMTE-4911: committed deliverables satisfy the schema."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    studying = STUDYING_PATH.read_text(encoding="utf-8")
    references = REFERENCES_PATH.read_text(encoding="utf-8")

    mod.validate_artifact(artifact)
    mod.validate_research_studying_text(studying, artifact)
    mod.validate_research_references_text(references, artifact)
    assert artifact["honest_verdict"] == mod.HONEST_VERDICT
    assert artifact["flagged_for_v453"] == mod.FLAGGED_FOR_V453
