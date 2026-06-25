"""Tests for Exp 4746 epistemic-MCTS / causal-probe / MATM SOTA ingestion.

Spec refs: REQ-ARC-WMTE-4746, SCENARIO-ARC-WMTE-4746.
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_4746_sota_ingestion_epistemic_mcts_causal_probe as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"
ARTIFACT_PATH = REPO / mod.RESULT_RELATIVE_PATH
NOTE_PATH = REPO / mod.NOTE_RELATIVE_PATH


def _valid_artifact() -> dict[str, object]:
    return mod.build_artifact()


def _spec_section() -> str:
    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-ARC-WMTE-4746")
    end = spec.index("### REQ-ARC-WMTE-4739", start)
    return spec[start:end]


def test_req_arc_wmte_4746_spec_declares_v437_ingestion_contract() -> None:
    """REQ-ARC-WMTE-4746: OpenSpec declares the 4746 artifact contract."""

    section = _spec_section()

    assert "REQ-ARC-WMTE-4746" in section
    assert "SCENARIO-ARC-WMTE-4746" in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert mod.NOTE_RELATIVE_PATH in section
    assert mod.HONEST_VERDICT in section
    assert mod.INFERENCE_SUBSTRATE in section
    assert "blocked_network" in section
    assert "/deep-research" in section
    assert "flagged_for_v437" in section
    assert "flagged_for_v436" not in section
    assert "E3AgentPolicy" in section
    assert "StepwiseExplorer.adj" in section
    assert "arc_executable_world_model" in section
    assert "action-efficiency candidate" in section
    assert "level-bank" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in section
        assert principle["principle"] in section
    for source_id in mod.REQUIRED_SOURCE_IDS:
        assert source_id in section


def test_req_arc_wmte_4746_artifact_has_required_fields_and_v437_flags() -> None:
    """REQ-ARC-WMTE-4746: artifact exposes required fields and annotations."""

    artifact = _valid_artifact()

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_principles"]) == set(mod.REQUIRED_PRINCIPLE_FIELDS)
    assert artifact["honest_verdict"] == mod.HONEST_VERDICT
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["note_path"] == mod.NOTE_RELATIVE_PATH
    assert artifact["verifier_is_oracle"] is False
    assert artifact["flagged_for_next_roadmap"] == mod.FLAGGED_FOR_NEXT_ROADMAP
    assert all("flagged_for_v437" in item for item in artifact["flagged_for_next_roadmap"])
    assert all("flagged_for_v436" not in item for item in artifact["flagged_for_next_roadmap"])
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["reproducibility_checksum"] == mod.REPRODUCIBILITY_CHECKSUM
    assert 3 <= len(artifact["methods_mapped"]) <= 5
    assert set(artifact["citations"]) == mod.REQUIRED_SOURCE_IDS

    preconditions = artifact["preconditions_checked"]
    assert preconditions["arxiv_reachable"] is True
    assert preconditions["deep_research_invoked"] is False
    assert preconditions["websearch_webfetch_used"] is True
    assert preconditions["top_source_count"] == 7
    assert preconditions["solve_claim_made"] is False
    assert preconditions["training_launched"] is False
    assert preconditions["model_load"] is False
    assert preconditions["leaderboard_submission"] is False
    assert preconditions["research_conductor_modified"] is False
    assert preconditions["ops_docs_modified"] is False
    assert preconditions["matm_bounded_to_within_game_efficiency"] is True
    assert preconditions["cross_game_matm_claim_made"] is False
    assert preconditions["level_bank_claim_made"] is False

    tracks = {method["track"] for method in artifact["methods_mapped"]}
    assert "epistemic_object_model_mcts_probe_planner" in tracks
    assert "factored_interaction_causal_probe_bank" in tracks
    assert "similarity_keyed_partial_trajectory_retrieval" in tracks
    assert "object_world_model_drift_policy_breakage_falsifier" in tracks
    for method in artifact["methods_mapped"]:
        assert set(method) == mod.REQUIRED_METHOD_FIELDS
        assert "E3AgentPolicy" in method["maps_to_current_stack"]
        assert "arc_executable_world_model" in method["maps_to_current_stack"]
        assert method["implement_cost_over_current_stack"]
        assert method["fails_when"]
        assert method["roadmap_candidate"]
    matm = next(
        method
        for method in artifact["methods_mapped"]
        if method["track"] == "similarity_keyed_partial_trajectory_retrieval"
    )
    assert matm["source_ids"] == ["2606.19911"]
    assert "StepwiseExplorer.adj" in matm["maps_to_current_stack"]
    assert "within-game" in matm["fails_when"]
    assert "level-bank" in matm["roadmap_candidate"]


@pytest.mark.parametrize(
    ("bad_artifact", "message"),
    [
        (_valid_artifact() | {"honest_verdict": "draft"}, "terminal prefix"),
        (_valid_artifact() | {"honest_verdict": "success: wrong"}, "honest_verdict"),
        (_valid_artifact() | {"inference_substrate": "live_llm"}, "substrate"),
        (_valid_artifact() | {"field_principles": {"honest_verdict": "loose"}}, "field_principles"),
        (_valid_artifact() | {"note_path": "docs/wrong.md"}, "note_path"),
        (_valid_artifact() | {"verifier_is_oracle": True}, "verifier_is_oracle"),
        (_valid_artifact() | {"random_seed": "4746"}, "random_seed"),
        (_valid_artifact() | {"reproducibility_checksum": "sha256:wrong"}, "checksum"),
        (
            _valid_artifact()
            | {
                "citations": {
                    key: value for key, value in mod.CITATIONS.items() if key != "2606.19911"
                }
            },
            "citations",
        ),
        (
            _valid_artifact()
            | {
                "citations": mod.CITATIONS
                | {
                    "2606.19911": {
                        "url": "https://arxiv.org/abs/2606.19911",
                        "http_status": 200,
                    }
                }
            },
            "exactly title",
        ),
        (
            _valid_artifact()
            | {
                "citations": mod.CITATIONS
                | {
                    "2606.19911": mod.CITATIONS["2606.19911"]
                    | {"url": "https://arxiv.org/abs/0000.00000"}
                }
            },
            "citation url",
        ),
        (
            _valid_artifact()
            | {
                "citations": mod.CITATIONS
                | {"2606.19911": mod.CITATIONS["2606.19911"] | {"http_status": 404}}
            },
            "http_status",
        ),
        (
            _valid_artifact()
            | {
                "citations": mod.CITATIONS
                | {"2606.19911": mod.CITATIONS["2606.19911"] | {"title": ""}}
            },
            "title",
        ),
        (_valid_artifact() | {"methods_mapped": mod.DEFAULT_METHODS_MAPPED[:2]}, "three to five"),
        (
            _valid_artifact()
            | {
                "methods_mapped": [
                    mod.DEFAULT_METHODS_MAPPED[0] | {"source_ids": []},
                    *mod.DEFAULT_METHODS_MAPPED[1:],
                ]
            },
            "source_ids",
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
                    mod.DEFAULT_METHODS_MAPPED[0] | {"maps_to_current_stack": "E3 only"},
                    *mod.DEFAULT_METHODS_MAPPED[1:],
                ]
            },
            "current stack",
        ),
        (
            _valid_artifact()
            | {
                "methods_mapped": [
                    mod.DEFAULT_METHODS_MAPPED[0] | {"implement_cost_over_current_stack": ""},
                    *mod.DEFAULT_METHODS_MAPPED[1:],
                ]
            },
            "implement_cost_over_current_stack",
        ),
        (
            _valid_artifact()
            | {
                "methods_mapped": [
                    mod.DEFAULT_METHODS_MAPPED[0] | {"fails_when": ""},
                    *mod.DEFAULT_METHODS_MAPPED[1:],
                ]
            },
            "fails_when",
        ),
        (
            _valid_artifact()
            | {"flagged_for_next_roadmap": ["flagged_for_v436: stale"]},
            ".437",
        ),
        (
            _valid_artifact()
            | {"flagged_for_next_roadmap": ["flagged_for_v437: cross_game_matm_level_bank"]},
            "MATM",
        ),
        (
            _valid_artifact()
            | {"preconditions_checked": dict(mod.DEFAULT_PRECONDITIONS_CHECKED) | {"arxiv_reachable": False}},
            "network",
        ),
        (_valid_artifact() | {"preconditions_checked": []}, "preconditions_checked"),
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
                | {"top_source_count": 9}
            },
            "top five to eight",
        ),
        (
            _valid_artifact()
            | {
                "preconditions_checked": dict(mod.DEFAULT_PRECONDITIONS_CHECKED)
                | {"cross_game_matm_claim_made": True}
            },
            "cross-game MATM",
        ),
        (
            _valid_artifact()
            | {
                "preconditions_checked": dict(mod.DEFAULT_PRECONDITIONS_CHECKED)
                | {"level_bank_claim_made": True}
            },
            "level-bank",
        ),
        (
            _valid_artifact()
            | {
                "preconditions_checked": dict(mod.DEFAULT_PRECONDITIONS_CHECKED)
                | {"solve_claim_made": True}
            },
            "solve claim",
        ),
        (
            _valid_artifact()
            | {
                "preconditions_checked": dict(mod.DEFAULT_PRECONDITIONS_CHECKED)
                | {"training_launched": True}
            },
            "training",
        ),
        (
            _valid_artifact()
            | {
                "preconditions_checked": dict(mod.DEFAULT_PRECONDITIONS_CHECKED)
                | {"model_load": True}
            },
            "model load",
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
                | {"research_conductor_modified": True}
            },
            "research_conductor",
        ),
        (
            _valid_artifact()
            | {
                "preconditions_checked": dict(mod.DEFAULT_PRECONDITIONS_CHECKED)
                | {"ops_docs_modified": True}
            },
            "ops docs",
        ),
    ],
)
def test_validate_artifact_rejects_schema_violations_for_req_arc_wmte_4746(
    bad_artifact: dict[str, object], message: str
) -> None:
    """REQ-ARC-WMTE-4746: malformed ingestion artifacts fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_validate_artifact_rejects_exact_field_violations() -> None:
    """REQ-ARC-WMTE-4746: top-level and method schemas are exact."""

    missing = _valid_artifact()
    missing.pop("methods_mapped")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing)

    extra = _valid_artifact() | {"offline_reproduced": False}
    with pytest.raises(ValueError, match="unexpected fields"):
        mod.validate_artifact(extra)

    malformed_method = _valid_artifact() | {
        "methods_mapped": ["not-a-dict", *mod.DEFAULT_METHODS_MAPPED[1:]]
    }
    with pytest.raises(ValueError, match="method"):
        mod.validate_artifact(malformed_method)


def test_scenario_arc_wmte_4746_note_round_trips_and_maps_methods() -> None:
    """SCENARIO-ARC-WMTE-4746: note embeds JSON and cites mapped methods."""

    artifact = mod.artifact_from_note(mod.RESEARCH_NOTE)

    assert artifact == _valid_artifact()
    mod.validate_research_note(mod.RESEARCH_NOTE)
    assert "SOTA -> .437 epistemic-MCTS / causal-probe / MATM mapping" in mod.RESEARCH_NOTE
    assert "Bottom line for the .437 roadmap" in mod.RESEARCH_NOTE
    assert "E3AgentPolicy" in mod.RESEARCH_NOTE
    assert "StepwiseExplorer.adj" in mod.RESEARCH_NOTE
    assert "arc_executable_world_model" in mod.RESEARCH_NOTE
    assert "flagged_for_v437" in mod.RESEARCH_NOTE
    assert "no solve claim" in mod.RESEARCH_NOTE
    assert "within-game action-efficiency candidate, not a level-bank" in mod.RESEARCH_NOTE

    with pytest.raises(ValueError, match="machine-readable JSON"):
        mod.artifact_from_note(mod.RESEARCH_NOTE.replace(mod.JSON_BLOCK_MARKER, "## Missing marker"))
    with pytest.raises(ValueError, match="terminator"):
        mod.artifact_from_note(mod.RESEARCH_NOTE.replace(mod.JSON_BLOCK_TERMINATOR, ""))
    with pytest.raises(ValueError, match="required phrase"):
        mod.validate_research_note(
            mod.RESEARCH_NOTE.replace("Bottom line for the .437 roadmap", "Bottom line")
        )
    note_with_forbidden_matm = mod.RESEARCH_NOTE.replace(
        "within-game action-efficiency candidate, not a level-bank",
        "cross-game MATM level-bank",
    )
    with pytest.raises(ValueError, match="MATM"):
        mod.validate_research_note(note_with_forbidden_matm)
    json_block_end = mod.RESEARCH_NOTE.find(mod.JSON_BLOCK_TERMINATOR) + len("```")
    note_without_citation = mod.RESEARCH_NOTE[:json_block_end] + mod.RESEARCH_NOTE[
        json_block_end:
    ].replace("arXiv:2606.19911", "MATM")
    with pytest.raises(ValueError, match="verified source citations"):
        mod.validate_research_note(note_without_citation)


def test_write_outputs_is_idempotent_for_scenario_arc_wmte_4746(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4746: writer emits stable artifact and note."""

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    note_path = tmp_path / mod.NOTE_RELATIVE_PATH

    artifact = mod.write_outputs(artifact_path=artifact_path, note_path=note_path)
    second_artifact = mod.write_outputs(artifact_path=artifact_path, note_path=note_path)

    assert second_artifact == artifact
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert mod.artifact_from_note(note_path.read_text(encoding="utf-8")) == artifact


def test_blocked_network_artifact_for_req_arc_wmte_4746() -> None:
    """REQ-ARC-WMTE-4746: blocked network exits without fabricated claims."""

    blocked = mod.build_blocked_network_artifact()

    assert blocked["honest_verdict"] == "blocked_network"
    assert blocked["preconditions_checked"]["arxiv_reachable"] is False
    assert blocked["methods_mapped"] == []
    assert blocked["citations"] == {}
    assert blocked["flagged_for_next_roadmap"] == []
    mod.validate_artifact(blocked, allow_blocked=True)
    with pytest.raises(ValueError, match="blocked"):
        mod.validate_artifact(blocked)


def test_main_writes_deliverables_for_req_arc_wmte_4746(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-ARC-WMTE-4746: direct module execution writes default outputs."""

    monkeypatch.setenv("CARNOT_EXP4746_ROOT", str(tmp_path))
    monkeypatch.setenv("CARNOT_EXP4746_SKIP_NETWORK_CHECK", "1")

    with pytest.raises(SystemExit) as module_exit:
        runpy.run_path(str(Path(mod.__file__)), run_name="__main__")

    assert module_exit.value.code == 0
    assert capsys.readouterr().out.strip() == mod.HONEST_VERDICT
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()
    assert (tmp_path / mod.NOTE_RELATIVE_PATH).exists()


def test_main_writes_blocked_artifact_when_network_blocked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-ARC-WMTE-4746: direct module execution reports blocked arXiv."""

    monkeypatch.setenv("CARNOT_EXP4746_ROOT", str(tmp_path))
    monkeypatch.setenv("CARNOT_EXP4746_FORCE_BLOCKED_NETWORK", "1")

    with pytest.raises(SystemExit) as module_exit:
        runpy.run_path(str(Path(mod.__file__)), run_name="__main__")

    assert module_exit.value.code == 0
    assert capsys.readouterr().out.strip() == "blocked_network"
    blocked = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert blocked["honest_verdict"] == "blocked_network"
    assert (tmp_path / mod.NOTE_RELATIVE_PATH).exists()


def test_deliverable_files_validate_for_req_arc_wmte_4746() -> None:
    """REQ-ARC-WMTE-4746: committed deliverables satisfy the schema."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    note = NOTE_PATH.read_text(encoding="utf-8")

    mod.validate_artifact(artifact)
    mod.validate_research_note(note)
    assert mod.artifact_from_note(note) == artifact
    assert artifact["honest_verdict"] == mod.HONEST_VERDICT
    assert artifact["verifier_is_oracle"] is False
    assert artifact["flagged_for_next_roadmap"] == mod.FLAGGED_FOR_NEXT_ROADMAP
