"""Tests for Exp 4697 amortized-exploration SOTA ingestion.

Spec refs: REQ-ARC-WMTE-4697, SCENARIO-ARC-WMTE-4697.
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_4697_sota_ingestion_amortized_exploration as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"
ARTIFACT_PATH = REPO / mod.RESULT_RELATIVE_PATH
NOTE_PATH = REPO / mod.NOTE_RELATIVE_PATH
STUDYING_PATH = REPO / "research-studying.md"


def _valid_artifact() -> dict[str, object]:
    return mod.build_artifact()


def test_req_arc_wmte_4697_spec_declares_amortized_exploration_contract() -> None:
    """REQ-ARC-WMTE-4697: OpenSpec declares the 4697 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4697" in spec
    assert "SCENARIO-ARC-WMTE-4697" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert mod.NOTE_RELATIVE_PATH in spec
    assert mod.HONEST_VERDICT in spec
    assert mod.INFERENCE_SUBSTRATE in spec
    assert "blocked_network" in spec
    assert "blocked_sweep_clusters" in spec
    assert "scripts/sweep_clusters.py" in spec
    assert "scripts/sweep_semscholar.py" in spec
    assert "/deep-research" in spec
    assert "flagged_for_v433" in spec
    assert "winning_prefix_still_not_proposed" in spec
    assert "heldout_transitions_too_sparse" in spec
    assert "scored hidden-game transfer" in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec
    for source_id in mod.REQUIRED_SOURCE_IDS:
        assert source_id in spec


def test_req_arc_wmte_4697_artifact_has_required_fields_and_principles() -> None:
    """REQ-ARC-WMTE-4697: artifact exposes required fields and annotations."""

    artifact = _valid_artifact()

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_principles"]) == set(mod.REQUIRED_PRINCIPLE_FIELDS)
    assert artifact["honest_verdict"] == mod.HONEST_VERDICT
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["note_path"] == mod.NOTE_RELATIVE_PATH
    assert artifact["deep_research_not_used"] is True
    assert artifact["flagged_for_next_roadmap"] == mod.FLAGGED_FOR_NEXT_ROADMAP
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert 3 <= len(artifact["methods_mapped"]) <= 5
    assert set(artifact["citations_verified"]) == mod.REQUIRED_SOURCE_IDS
    assert artifact["preconditions_checked"]["network_hf_models_reachable"] is True
    assert artifact["preconditions_checked"]["sweep_clusters_help_ok"] is True
    assert artifact["preconditions_checked"]["sweep_clusters_used"] is True
    assert artifact["preconditions_checked"]["sweep_semscholar_used"] is True
    assert artifact["preconditions_checked"]["deep_research_invoked"] is False
    assert artifact["preconditions_checked"]["research_conductor_modified"] is False
    assert artifact["preconditions_checked"]["ops_docs_modified"] is False
    assert artifact["preconditions_checked"]["training_launched"] is False
    assert artifact["preconditions_checked"]["model_load"] is False

    for method in artifact["methods_mapped"]:
        assert "live E3 explorer" in method["maps_to_current_stack"]
        assert "A1 controllable-novelty proposal" in method["maps_to_current_stack"]
        assert "A2 program-synthesis action-effect filter" in method["maps_to_current_stack"]
        assert "arc_go_explore.py" in method["maps_to_current_stack"]
        assert mod.A1_RESIDUAL in method["residual_scope"]
        assert mod.A2_RESIDUAL in method["residual_scope"]
        assert mod.TRANSFER_WALL in method["residual_scope"]


@pytest.mark.parametrize(
    ("bad_artifact", "message"),
    [
        (_valid_artifact() | {"honest_verdict": "draft"}, "terminal prefix"),
        (_valid_artifact() | {"honest_verdict": "success: wrong"}, "honest_verdict"),
        (_valid_artifact() | {"inference_substrate": "live_llm"}, "substrate"),
        (_valid_artifact() | {"field_principles": {"honest_verdict": "loose"}}, "field_principles"),
        (_valid_artifact() | {"deep_research_not_used": False}, "deep_research"),
        (_valid_artifact() | {"note_path": "docs/wrong.md"}, "note_path"),
        (_valid_artifact() | {"random_seed": "4697"}, "random_seed"),
        (
            _valid_artifact()
            | {
                "citations_verified": {
                    key: value
                    for key, value in mod.CITATIONS_VERIFIED.items()
                    if key != "2310.09971"
                }
            },
            "citations_verified",
        ),
        (
            _valid_artifact()
            | {
                "citations_verified": mod.CITATIONS_VERIFIED
                | {
                    "2310.09971": {
                        "url": "https://arxiv.org/abs/2310.09971",
                        "http_status": 200,
                    }
                }
            },
            "exactly title",
        ),
        (
            _valid_artifact()
            | {
                "citations_verified": mod.CITATIONS_VERIFIED
                | {
                    "2310.09971": mod.CITATIONS_VERIFIED["2310.09971"]
                    | {"url": "https://arxiv.org/abs/0000.00000"}
                }
            },
            "citation url",
        ),
        (
            _valid_artifact()
            | {
                "citations_verified": mod.CITATIONS_VERIFIED
                | {"2310.09971": mod.CITATIONS_VERIFIED["2310.09971"] | {"http_status": 404}}
            },
            "http_status",
        ),
        (
            _valid_artifact()
            | {
                "citations_verified": mod.CITATIONS_VERIFIED
                | {"2310.09971": mod.CITATIONS_VERIFIED["2310.09971"] | {"title": ""}}
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
            (
                "live E3 explorer, A1 controllable-novelty proposal, "
                "A2 program-synthesis action-effect filter, and arc_go_explore.py"
            ),
        ),
        (
            _valid_artifact()
            | {
                "methods_mapped": [
                    mod.DEFAULT_METHODS_MAPPED[0] | {"residual_scope": "per-game only"},
                    *mod.DEFAULT_METHODS_MAPPED[1:],
                ]
            },
            "cross-game transfer residuals",
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
            | {"flagged_for_next_roadmap": ["flagged_for_v432: stale"]},
            ".433",
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
                | {"sweep_clusters_help_ok": False}
            },
            "sweep_clusters",
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
def test_validate_artifact_rejects_schema_violations_for_req_arc_wmte_4697(
    bad_artifact: dict[str, object], message: str
) -> None:
    """REQ-ARC-WMTE-4697: malformed ingestion artifacts fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_validate_artifact_rejects_exact_field_violations() -> None:
    """REQ-ARC-WMTE-4697: top-level and method schemas are exact."""

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


def test_scenario_arc_wmte_4697_note_round_trips_and_maps_methods() -> None:
    """SCENARIO-ARC-WMTE-4697: note embeds JSON and cites mapped methods."""

    artifact = mod.artifact_from_note(mod.RESEARCH_NOTE)

    assert artifact == _valid_artifact()
    mod.validate_research_note(mod.RESEARCH_NOTE)
    assert "SOTA -> .433 amortized-exploration mapping" in mod.RESEARCH_NOTE
    assert "Bottom line for the .433 roadmap" in mod.RESEARCH_NOTE
    assert "live E3 explorer" in mod.RESEARCH_NOTE
    assert "A1 controllable-novelty proposal" in mod.RESEARCH_NOTE
    assert "A2 program-synthesis action-effect filter" in mod.RESEARCH_NOTE
    assert "arc_go_explore.py" in mod.RESEARCH_NOTE
    assert "winning_prefix_still_not_proposed" in mod.RESEARCH_NOTE
    assert "heldout_transitions_too_sparse" in mod.RESEARCH_NOTE
    assert "hidden-game transfer" in mod.RESEARCH_NOTE
    assert "AMAGO" in mod.RESEARCH_NOTE
    assert "Algorithm Distillation" in mod.RESEARCH_NOTE
    assert "Go-Explore" in mod.RESEARCH_NOTE
    assert "flagged_for_v433" in mod.RESEARCH_NOTE

    with pytest.raises(ValueError, match="machine-readable JSON"):
        mod.artifact_from_note(mod.RESEARCH_NOTE.replace("```json", "```text"))
    with pytest.raises(ValueError, match="terminator"):
        mod.artifact_from_note(mod.RESEARCH_NOTE.replace("\n```\n\n## Fresh-pass provenance", ""))
    with pytest.raises(ValueError, match="required phrase"):
        mod.validate_research_note(
            mod.RESEARCH_NOTE.replace("Bottom line for the .433 roadmap", "Bottom line")
        )
    json_block_end = mod.RESEARCH_NOTE.find("```\n\n## Fresh-pass provenance") + 3
    note_without_citation = mod.RESEARCH_NOTE[:json_block_end] + mod.RESEARCH_NOTE[
        json_block_end:
    ].replace("arXiv:2310.09971", "AMAGO")
    with pytest.raises(ValueError, match="verified source citations"):
        mod.validate_research_note(note_without_citation)


def test_write_outputs_is_idempotent_for_scenario_arc_wmte_4697(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4697: writer emits stable artifact, note, and queue update."""

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    note_path = tmp_path / mod.NOTE_RELATIVE_PATH
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
    assert "Exp 4697 - .433 amortized-exploration SOTA ingestion - INGESTED" in studying
    assert "flagged_for_v433" in studying


def test_main_writes_deliverables_for_req_arc_wmte_4697(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-ARC-WMTE-4697: direct module execution writes default outputs."""

    monkeypatch.setenv("CARNOT_EXP4697_ROOT", str(tmp_path))
    (tmp_path / "research-studying.md").write_text("# Research Studying\n", encoding="utf-8")

    with pytest.raises(SystemExit) as module_exit:
        runpy.run_path(str(Path(mod.__file__)), run_name="__main__")

    assert module_exit.value.code == 0
    assert capsys.readouterr().out.strip() == mod.HONEST_VERDICT
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()
    assert (tmp_path / mod.NOTE_RELATIVE_PATH).exists()


def test_deliverable_files_validate_for_req_arc_wmte_4697() -> None:
    """REQ-ARC-WMTE-4697: committed deliverables satisfy the schema."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    note = NOTE_PATH.read_text(encoding="utf-8")
    studying = STUDYING_PATH.read_text(encoding="utf-8")

    mod.validate_artifact(artifact)
    mod.validate_research_note(note)
    assert mod.artifact_from_note(note) == artifact
    assert artifact["honest_verdict"] == mod.HONEST_VERDICT
    assert artifact["deep_research_not_used"] is True
    assert artifact["flagged_for_next_roadmap"] == mod.FLAGGED_FOR_NEXT_ROADMAP
    assert "Exp 4697 - .433 amortized-exploration SOTA ingestion - INGESTED" in studying
