"""Tests for Exp 4722 active-probe world-model SOTA ingestion.

Spec refs: REQ-ARC-WMTE-4722, SCENARIO-ARC-WMTE-4722.
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_4722_sota_ingestion_active_probe_world_model as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"
ARTIFACT_PATH = REPO / mod.RESULT_RELATIVE_PATH
NOTE_PATH = REPO / mod.NOTE_RELATIVE_PATH


def _valid_artifact() -> dict[str, object]:
    return mod.build_artifact()


def test_req_arc_wmte_4722_spec_declares_active_probe_contract() -> None:
    """REQ-ARC-WMTE-4722: OpenSpec declares the 4722 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4722" in spec
    assert "SCENARIO-ARC-WMTE-4722" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert mod.NOTE_RELATIVE_PATH in spec
    assert mod.HONEST_VERDICT in spec
    assert mod.INFERENCE_SUBSTRATE in spec
    assert "blocked_network" in spec
    assert "/deep-research" in spec
    assert "flagged_for_v435" in spec
    assert "E3AgentPolicy" in spec
    assert "arc_executable_world_model" in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec
    for source_id in mod.REQUIRED_SOURCE_IDS:
        assert source_id in spec


def test_req_arc_wmte_4722_artifact_has_required_fields_and_principles() -> None:
    """REQ-ARC-WMTE-4722: artifact exposes required fields and annotations."""

    artifact = _valid_artifact()

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_principles"]) == set(mod.REQUIRED_PRINCIPLE_FIELDS)
    assert artifact["honest_verdict"] == mod.HONEST_VERDICT
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["note_path"] == mod.NOTE_RELATIVE_PATH
    assert artifact["verifier_is_oracle"] is False
    assert artifact["flagged_for_next_roadmap"] == mod.FLAGGED_FOR_NEXT_ROADMAP
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["reproducibility_checksum"] == mod.REPRODUCIBILITY_CHECKSUM
    assert 3 <= len(artifact["methods_mapped"]) <= 5
    assert set(artifact["citations"]) == mod.REQUIRED_SOURCE_IDS
    assert artifact["preconditions_checked"]["arxiv_reachable"] is True
    assert artifact["preconditions_checked"]["deep_research_invoked"] is False
    assert artifact["preconditions_checked"]["solve_claim_made"] is False
    assert artifact["preconditions_checked"]["training_launched"] is False
    assert artifact["preconditions_checked"]["model_load"] is False
    assert artifact["preconditions_checked"]["leaderboard_submission"] is False
    assert artifact["preconditions_checked"]["research_conductor_modified"] is False

    for method in artifact["methods_mapped"]:
        assert set(method) == mod.REQUIRED_METHOD_FIELDS
        assert "E3AgentPolicy" in method["maps_to_current_stack"]
        assert "arc_executable_world_model" in method["maps_to_current_stack"]
        assert "maps_to_current_stack" in method
        assert "implement_cost_over_current_stack" in method
        assert "fails_when" in method
        assert method["fails_when"]
        assert method["implement_cost_over_current_stack"]


@pytest.mark.parametrize(
    ("bad_artifact", "message"),
    [
        (_valid_artifact() | {"honest_verdict": "draft"}, "terminal prefix"),
        (_valid_artifact() | {"honest_verdict": "success: wrong"}, "honest_verdict"),
        (_valid_artifact() | {"inference_substrate": "live_llm"}, "substrate"),
        (_valid_artifact() | {"field_principles": {"honest_verdict": "loose"}}, "field_principles"),
        (_valid_artifact() | {"note_path": "docs/wrong.md"}, "note_path"),
        (_valid_artifact() | {"verifier_is_oracle": True}, "verifier_is_oracle"),
        (_valid_artifact() | {"random_seed": "4722"}, "random_seed"),
        (_valid_artifact() | {"reproducibility_checksum": "sha256:wrong"}, "checksum"),
        (
            _valid_artifact()
            | {
                "citations": {
                    key: value for key, value in mod.CITATIONS.items() if key != "2506.01876"
                }
            },
            "citations",
        ),
        (
            _valid_artifact()
            | {
                "citations": mod.CITATIONS
                | {
                    "2506.01876": {
                        "url": "https://arxiv.org/abs/2506.01876",
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
                    "2506.01876": mod.CITATIONS["2506.01876"]
                    | {"url": "https://arxiv.org/abs/0000.00000"}
                }
            },
            "citation url",
        ),
        (
            _valid_artifact()
            | {
                "citations": mod.CITATIONS
                | {"2506.01876": mod.CITATIONS["2506.01876"] | {"http_status": 404}}
            },
            "http_status",
        ),
        (
            _valid_artifact()
            | {
                "citations": mod.CITATIONS
                | {"2506.01876": mod.CITATIONS["2506.01876"] | {"title": ""}}
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
            "E3AgentPolicy and arc_executable_world_model",
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
        (_valid_artifact() | {"flagged_for_next_roadmap": ["flagged_for_v434: stale"]}, ".435"),
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
    ],
)
def test_validate_artifact_rejects_schema_violations_for_req_arc_wmte_4722(
    bad_artifact: dict[str, object], message: str
) -> None:
    """REQ-ARC-WMTE-4722: malformed ingestion artifacts fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_validate_artifact_rejects_exact_field_violations() -> None:
    """REQ-ARC-WMTE-4722: top-level and method schemas are exact."""

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


def test_scenario_arc_wmte_4722_note_round_trips_and_maps_methods() -> None:
    """SCENARIO-ARC-WMTE-4722: note embeds JSON and cites mapped methods."""

    artifact = mod.artifact_from_note(mod.RESEARCH_NOTE)

    assert artifact == _valid_artifact()
    mod.validate_research_note(mod.RESEARCH_NOTE)
    assert "SOTA -> .435 active-probe world-model mapping" in mod.RESEARCH_NOTE
    assert "Bottom line for the .435 roadmap" in mod.RESEARCH_NOTE
    assert "E3AgentPolicy" in mod.RESEARCH_NOTE
    assert "arc_executable_world_model" in mod.RESEARCH_NOTE
    assert "active-probe / hypothesis-driven world-model induction" in mod.RESEARCH_NOTE
    assert "flagged_for_v435" in mod.RESEARCH_NOTE
    assert "no solve claim" in mod.RESEARCH_NOTE

    with pytest.raises(ValueError, match="machine-readable JSON"):
        mod.artifact_from_note(mod.RESEARCH_NOTE.replace("```json", "```text"))
    with pytest.raises(ValueError, match="terminator"):
        mod.artifact_from_note(mod.RESEARCH_NOTE.replace("\n```\n\n## Fresh-pass provenance", ""))
    with pytest.raises(ValueError, match="required phrase"):
        mod.validate_research_note(
            mod.RESEARCH_NOTE.replace("Bottom line for the .435 roadmap", "Bottom line")
        )
    json_block_end = mod.RESEARCH_NOTE.find("```\n\n## Fresh-pass provenance") + 3
    note_without_citation = mod.RESEARCH_NOTE[:json_block_end] + mod.RESEARCH_NOTE[
        json_block_end:
    ].replace("arXiv:2506.01876", "ICPE")
    with pytest.raises(ValueError, match="verified source citations"):
        mod.validate_research_note(note_without_citation)


def test_write_outputs_is_idempotent_for_scenario_arc_wmte_4722(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4722: writer emits stable artifact and note."""

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    note_path = tmp_path / mod.NOTE_RELATIVE_PATH

    artifact = mod.write_outputs(artifact_path=artifact_path, note_path=note_path)
    second_artifact = mod.write_outputs(artifact_path=artifact_path, note_path=note_path)

    assert second_artifact == artifact
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert mod.artifact_from_note(note_path.read_text(encoding="utf-8")) == artifact


def test_main_writes_deliverables_for_req_arc_wmte_4722(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-ARC-WMTE-4722: direct module execution writes default outputs."""

    monkeypatch.setenv("CARNOT_EXP4722_ROOT", str(tmp_path))

    with pytest.raises(SystemExit) as module_exit:
        runpy.run_path(str(Path(mod.__file__)), run_name="__main__")

    assert module_exit.value.code == 0
    assert capsys.readouterr().out.strip() == mod.HONEST_VERDICT
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()
    assert (tmp_path / mod.NOTE_RELATIVE_PATH).exists()


def test_deliverable_files_validate_for_req_arc_wmte_4722() -> None:
    """REQ-ARC-WMTE-4722: committed deliverables satisfy the schema."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    note = NOTE_PATH.read_text(encoding="utf-8")

    mod.validate_artifact(artifact)
    mod.validate_research_note(note)
    assert mod.artifact_from_note(note) == artifact
    assert artifact["honest_verdict"] == mod.HONEST_VERDICT
    assert artifact["verifier_is_oracle"] is False
    assert artifact["flagged_for_next_roadmap"] == mod.FLAGGED_FOR_NEXT_ROADMAP
