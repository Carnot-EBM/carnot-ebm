"""Tests for Exp 4758 structured-world-model / grounded-goal SOTA ingestion.

Spec refs: REQ-ARC-WMTE-4758, SCENARIO-ARC-WMTE-4758.
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_4758_sota_ingestion as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"
ARTIFACT_PATH = REPO / mod.RESULT_RELATIVE_PATH
STUDYING_PATH = REPO / mod.STUDYING_RELATIVE_PATH


def _valid_artifact() -> dict[str, object]:
    return mod.build_artifact()


def _spec_section() -> str:
    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-ARC-WMTE-4758")
    end = spec.index("### REQ-ARC-WMTE-4751", start)
    return spec[start:end]


def test_req_arc_wmte_4758_spec_declares_ingestion_contract() -> None:
    """REQ-ARC-WMTE-4758: OpenSpec declares the 4758 artifact contract."""

    section = _spec_section()

    assert "REQ-ARC-WMTE-4758" in section
    assert "SCENARIO-ARC-WMTE-4758" in section
    assert "SCENARIO-ARC-WMTE-4758-BLOCKED-NETWORK" in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert mod.STUDYING_RELATIVE_PATH in section
    assert mod.HONEST_VERDICT in section
    assert mod.INFERENCE_SUBSTRATE in section
    assert "blocked_network" in section
    assert "/deep-research" in section
    assert "flagged_for_438" in section
    assert "E3AgentPolicy" in section
    assert "arc_executable_world_model" in section
    assert "ProductWorldModel" in section
    for field, principle in mod.REQUIRED_USER_FIELD_PRINCIPLES.items():
        assert field in section
        assert principle["principle"] in section
    for source_id in mod.REQUIRED_SOURCE_IDS:
        assert source_id in section


def test_req_arc_wmte_4758_artifact_has_required_fields_and_real_citations() -> None:
    """REQ-ARC-WMTE-4758: artifact exposes required fields and real arXiv IDs."""

    artifact = _valid_artifact()

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["honest_verdict"] == mod.HONEST_VERDICT
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["note_path"] == mod.NOTE_PATH
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["reproducibility_checksum"] == mod.REPRODUCIBILITY_CHECKSUM
    assert artifact["flagged_for_438"] == mod.FLAGGED_FOR_438
    assert "flagged_for_438" in artifact["flagged_for_438"]
    assert "flagged_for_437" not in artifact["flagged_for_438"]
    assert set(artifact["field_principles"]) == set(mod.REQUIRED_PRINCIPLE_FIELDS)
    assert set(artifact["citations"]) == mod.REQUIRED_SOURCE_IDS
    assert 3 <= len(artifact["methods_mapped"]) <= 5

    for method in artifact["methods_mapped"]:
        assert set(method) == mod.REQUIRED_METHOD_FIELDS
        assert method["source_ids"]
        assert set(method["source_ids"]).issubset(mod.REQUIRED_SOURCE_IDS)
        assert method["maps_to_current_stack"]
        assert method["takes_over_from_current_stack"]
        assert method["fails_when"]

    tracks = {method["track"] for method in artifact["methods_mapped"]}
    assert "verifier_refined_executable_world_model" in tracks
    assert "perception_grounded_goal_conditioned_planning" in tracks
    assert "causal_object_mcts_action_slot_planner" in tracks
    assert "interactive_program_synthesis_refinement" in tracks

    preconditions = artifact["preconditions_checked"]
    assert preconditions["network_hf_models_reachable"] is True
    assert preconditions["research_studying_read"] is True
    assert preconditions["research_references_read"] is True
    assert preconditions["sweep_clusters_used"] is True
    assert preconditions["sweep_semscholar_used"] is True
    assert preconditions["sweep_semscholar_http_429"] is True
    assert preconditions["websearch_webfetch_used"] is True
    assert preconditions["top_source_count"] == 8
    assert preconditions["deep_research_invoked"] is False
    assert preconditions["model_load"] is False
    assert preconditions["training_launched"] is False
    assert preconditions["leaderboard_submission"] is False
    assert preconditions["solve_claim_made"] is False
    assert preconditions["ops_docs_modified"] is False


@pytest.mark.parametrize(
    ("bad_artifact", "message"),
    [
        (_valid_artifact() | {"honest_verdict": "success: wrong"}, "honest_verdict"),
        (_valid_artifact() | {"inference_substrate": "live_llm"}, "inference_substrate"),
        (_valid_artifact() | {"field_principles": {}}, "field_principles"),
        (_valid_artifact() | {"flagged_for_438": "flagged_for_437: stale"}, "flagged_for_438"),
        (_valid_artifact() | {"citations": {}}, "citations"),
        (
            _valid_artifact()
            | {
                "citations": mod.CITATIONS
                | {
                    "2605.05138": mod.CITATIONS["2605.05138"]
                    | {"url": "https://arxiv.org/abs/0000.00000"}
                }
            },
            "citation url",
        ),
        (
            _valid_artifact()
            | {
                "citations": mod.CITATIONS
                | {"2605.05138": mod.CITATIONS["2605.05138"] | {"http_status": 404}}
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
                    mod.DEFAULT_METHODS_MAPPED[0] | {"maps_to_current_stack": ""},
                    *mod.DEFAULT_METHODS_MAPPED[1:],
                ]
            },
            "maps_to_current_stack",
        ),
        (
            _valid_artifact()
            | {
                "methods_mapped": [
                    mod.DEFAULT_METHODS_MAPPED[0] | {"takes_over_from_current_stack": ""},
                    *mod.DEFAULT_METHODS_MAPPED[1:],
                ]
            },
            "takes_over",
        ),
        (
            _valid_artifact()
            | {
                "preconditions_checked": dict(mod.DEFAULT_PRECONDITIONS_CHECKED)
                | {"network_hf_models_reachable": False}
            },
            "network",
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
                | {"top_source_count": 4}
            },
            "top five to eight",
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
def test_validate_artifact_rejects_schema_violations_for_req_arc_wmte_4758(
    bad_artifact: dict[str, object], message: str
) -> None:
    """REQ-ARC-WMTE-4758: malformed ingestion artifacts fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_validate_artifact_rejects_exact_field_violations() -> None:
    """REQ-ARC-WMTE-4758: top-level and method schemas are exact."""

    missing = _valid_artifact()
    missing.pop("methods_mapped")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing)

    extra = _valid_artifact() | {"solve_rate": 0.0}
    with pytest.raises(ValueError, match="unexpected fields"):
        mod.validate_artifact(extra)

    malformed_method = _valid_artifact() | {
        "methods_mapped": ["not-a-dict", *mod.DEFAULT_METHODS_MAPPED[1:]]
    }
    with pytest.raises(ValueError, match="method"):
        mod.validate_artifact(malformed_method)


def test_scenario_arc_wmte_4758_research_studying_section_is_idempotent() -> None:
    """SCENARIO-ARC-WMTE-4758: research-studying update is a stable mapping note."""

    original = "# Research Studying\n\nOld body\n"
    once = mod.update_research_studying_text(original, _valid_artifact())
    twice = mod.update_research_studying_text(once, _valid_artifact())

    assert once == twice
    assert once.count(mod.STUDYING_SECTION_START) == 1
    assert once.count(mod.STUDYING_SECTION_END) == 1
    mod.validate_research_studying_text(once, _valid_artifact())
    assert "SOTA -> .438 experiment mapping" in once
    assert "flagged_for_438" in once
    assert "arXiv:2605.05138" in once
    assert "arXiv:2605.14937" in once
    assert "no solve claim" in once


def test_write_outputs_is_idempotent_for_scenario_arc_wmte_4758(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4758: writer emits stable artifact and studying note."""

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    studying_path = tmp_path / mod.STUDYING_RELATIVE_PATH
    studying_path.write_text("# Research Studying\n\n", encoding="utf-8")

    artifact = mod.write_outputs(artifact_path=artifact_path, studying_path=studying_path)
    second_artifact = mod.write_outputs(artifact_path=artifact_path, studying_path=studying_path)

    assert second_artifact == artifact
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    mod.validate_research_studying_text(studying_path.read_text(encoding="utf-8"), artifact)


def test_blocked_network_artifact_for_req_arc_wmte_4758() -> None:
    """REQ-ARC-WMTE-4758: blocked network exits without fabricated claims."""

    blocked = mod.build_blocked_network_artifact()

    assert blocked["honest_verdict"] == "blocked_network"
    assert blocked["preconditions_checked"]["network_hf_models_reachable"] is False
    assert blocked["methods_mapped"] == []
    assert blocked["citations"] == {}
    assert blocked["flagged_for_438"] == ""
    mod.validate_artifact(blocked, allow_blocked=True)
    with pytest.raises(ValueError, match="blocked"):
        mod.validate_artifact(blocked)


def test_main_writes_deliverables_for_req_arc_wmte_4758(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-ARC-WMTE-4758: direct module execution writes default outputs."""

    (tmp_path / mod.STUDYING_RELATIVE_PATH).write_text("# Research Studying\n\n", encoding="utf-8")
    monkeypatch.setenv("CARNOT_EXP4758_ROOT", str(tmp_path))
    monkeypatch.setenv("CARNOT_EXP4758_SKIP_NETWORK_CHECK", "1")

    with pytest.raises(SystemExit) as module_exit:
        runpy.run_path(str(Path(mod.__file__)), run_name="__main__")

    assert module_exit.value.code == 0
    assert capsys.readouterr().out.strip() == mod.HONEST_VERDICT
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()
    mod.validate_research_studying_text(
        (tmp_path / mod.STUDYING_RELATIVE_PATH).read_text(encoding="utf-8"),
        json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")),
    )


def test_main_writes_blocked_artifact_when_network_blocked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-ARC-WMTE-4758: direct module execution reports blocked network."""

    monkeypatch.setenv("CARNOT_EXP4758_ROOT", str(tmp_path))
    monkeypatch.setenv("CARNOT_EXP4758_FORCE_BLOCKED_NETWORK", "1")

    with pytest.raises(SystemExit) as module_exit:
        runpy.run_path(str(Path(mod.__file__)), run_name="__main__")

    assert module_exit.value.code == 0
    assert capsys.readouterr().out.strip() == "blocked_network"
    blocked = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert blocked["honest_verdict"] == "blocked_network"


def test_deliverable_files_validate_for_req_arc_wmte_4758() -> None:
    """REQ-ARC-WMTE-4758: committed deliverables satisfy the schema."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    studying = STUDYING_PATH.read_text(encoding="utf-8")

    mod.validate_artifact(artifact)
    mod.validate_research_studying_text(studying, artifact)
    assert artifact["honest_verdict"] == mod.HONEST_VERDICT
    assert artifact["flagged_for_438"] == mod.FLAGGED_FOR_438
