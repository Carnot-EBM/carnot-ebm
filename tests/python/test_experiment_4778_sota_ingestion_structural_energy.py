"""Tests for Exp 4778 structural-energy SOTA ingestion.

Spec refs: REQ-ARC-WMTE-4778, SCENARIO-ARC-WMTE-4778,
SCENARIO-ARC-WMTE-4778-LEAK-ROBUST.
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_4778_sota_ingestion_structural_energy as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"
ARTIFACT_PATH = REPO / mod.RESULT_RELATIVE_PATH
STUDYING_PATH = REPO / mod.STUDYING_RELATIVE_PATH


def _artifact() -> dict[str, object]:
    return mod.build_artifact()


def _spec_section() -> str:
    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-ARC-WMTE-4778")
    end = spec.index("### REQ-ARC-WMTE-4771", start)
    return spec[start:end]


def test_req_arc_wmte_4778_spec_declares_leak_robust_ingestion_contract() -> None:
    """REQ-ARC-WMTE-4778: OpenSpec declares the Exp 4778 artifact contract."""

    section = _spec_section()

    assert "REQ-ARC-WMTE-4778" in section
    assert "SCENARIO-ARC-WMTE-4778" in section
    assert "SCENARIO-ARC-WMTE-4778-LEAK-ROBUST" in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert mod.STUDYING_RELATIVE_PATH in section
    assert mod.HONEST_VERDICT in section
    assert mod.INFERENCE_SUBSTRATE in section
    assert "S1 through S4" in section
    assert "flagged_for_v440" in section
    assert "ShortcutProbe/IRM-style" in section
    assert "/deep-research" in section
    for field, principle in mod.REQUIRED_USER_FIELD_PRINCIPLES.items():
        assert field in section
        assert principle["principle"] in section
    for source_id in mod.REQUIRED_SOURCE_IDS:
        assert source_id in section


def test_req_arc_wmte_4778_artifact_maps_methods_with_real_arxiv_ids() -> None:
    """REQ-ARC-WMTE-4778: artifact maps cited methods onto S1-S4 plus leak eval."""

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

    stages = set()
    cited = set()
    tracks = set()
    for method in artifact["methods_mapped"]:
        assert set(method) == mod.REQUIRED_METHOD_FIELDS
        assert method["source_ids"]
        assert set(method["source_ids"]).issubset(mod.REQUIRED_SOURCE_IDS)
        assert method["maps_to_stages"]
        assert method["takes_over_from_current_stack"]
        assert method["leak_robust_eval_role"]
        assert method["fails_when"]
        assert method["roadmap_candidate"]
        stages.update(method["maps_to_stages"])
        cited.update(method["source_ids"])
        tracks.add(method["track"])

    assert 3 <= len(artifact["methods_mapped"]) <= 5
    assert {"S1", "S2", "S3", "S4"}.issubset(stages)
    assert cited.issubset(set(artifact["arxiv_ids_cited"]))
    assert "shortcut_invariance_leak_evaluation_gate" in tracks
    assert {flag["candidate"] for flag in artifact["flagged_for_v440"]} == {
        "slot_relational_contrastive_energy_s0prime_guarded",
        "poe_code_world_model_trust_gate_after_s0prime",
    }

    preconditions = artifact["preconditions_checked"]
    assert preconditions["research_studying_present"] is True
    assert preconditions["research_references_present"] is True
    assert preconditions["sweep_clusters_used"] is True
    assert preconditions["sweep_cluster_ids"] == [1, 5, 6]
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

    s0prime = artifact["s0prime_context"]
    assert s0prime["source_artifact"] == mod.S0PRIME_SOURCE_RELATIVE_PATH
    assert s0prime["stage"] == "S0'"
    assert s0prime["s0prime_gate_passed"] is True
    assert s0prime["origin_probe_auroc"] == 0.5
    assert s0prime["flagged_adversarial"] is True

    leak_note = artifact["leak_robust_evaluation_note"]
    assert set(leak_note["source_ids"]).issubset(set(artifact["arxiv_ids_cited"]))
    assert "origin/provenance" in leak_note["summary"]
    assert "counterfactual/invariance" in leak_note["summary"]
    assert len(leak_note["required_controls"]) >= 4


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
                    "2505.13910": mod.CITATIONS["2505.13910"]
                    | {"url": "https://arxiv.org/abs/9999.99999"}
                }
            },
            "citation url",
        ),
        (
            _artifact()
            | {
                "citations": mod.CITATIONS
                | {"2505.13910": mod.CITATIONS["2505.13910"] | {"http_status": 404}}
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
                    mod.DEFAULT_METHODS_MAPPED[0] | {"maps_to_stages": ["S0"]},
                    *mod.DEFAULT_METHODS_MAPPED[1:],
                ]
            },
            "S1-S4",
        ),
        (
            _artifact()
            | {
                "methods_mapped": [
                    mod.DEFAULT_METHODS_MAPPED[0] | {"leak_robust_eval_role": ""},
                    *mod.DEFAULT_METHODS_MAPPED[1:],
                ]
            },
            "leak_robust_eval_role",
        ),
        (_artifact() | {"flagged_for_v440": []}, "flagged_for_v440"),
        (
            _artifact()
            | {"flagged_for_v440": [{"candidate": "stale", "flag": "flagged_for_v439"}]},
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
                | {"top_source_count": 9}
            },
            "top five to eight",
        ),
        (
            _artifact()
            | {"s0prime_context": dict(mod.DEFAULT_S0PRIME_CONTEXT) | {"origin_probe_auroc": 0.7}},
            "origin",
        ),
        (
            _artifact()
            | {
                "leak_robust_evaluation_note": dict(mod.DEFAULT_LEAK_ROBUST_EVALUATION_NOTE)
                | {"source_ids": []}
            },
            "leak note",
        ),
    ],
)
def test_leak_robust_validator_rejects_bad_sota_ingestion_artifacts(
    bad_artifact: dict[str, object], message: str
) -> None:
    """SCENARIO-ARC-WMTE-4778-LEAK-ROBUST: invalid claims fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_validate_artifact_rejects_exact_schema_violations() -> None:
    """REQ-ARC-WMTE-4778: top-level and method schemas are exact."""

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


def test_scenario_arc_wmte_4778_research_studying_section_is_idempotent() -> None:
    """SCENARIO-ARC-WMTE-4778: research-studying update is stable and cited."""

    original = "# Research Studying\n\nOld body\n"
    once = mod.update_research_studying_text(original, _artifact())
    twice = mod.update_research_studying_text(once, _artifact())

    assert once == twice
    assert once.count(mod.STUDYING_SECTION_START) == 1
    assert once.count(mod.STUDYING_SECTION_END) == 1
    mod.validate_research_studying_text(once, _artifact())
    assert "SOTA -> S1-S4 structural-energy mapping" in once
    assert "Leak-robust evaluation note" in once
    assert "flagged_for_v440" in once
    assert "arXiv:2505.13910" in once
    assert "arXiv:1907.02893" in once
    assert "no solve claim" in once


def test_write_outputs_is_idempotent_for_scenario_arc_wmte_4778(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4778: writer emits stable JSON and studying note."""

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    studying_path = tmp_path / mod.STUDYING_RELATIVE_PATH
    studying_path.parent.mkdir(parents=True, exist_ok=True)
    studying_path.write_text("# Research Studying\n\n", encoding="utf-8")

    artifact = mod.write_outputs(artifact_path=artifact_path, studying_path=studying_path)
    second_artifact = mod.write_outputs(artifact_path=artifact_path, studying_path=studying_path)

    assert second_artifact == artifact
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    mod.validate_research_studying_text(studying_path.read_text(encoding="utf-8"), artifact)


def test_main_writes_deliverables_for_req_arc_wmte_4778(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-ARC-WMTE-4778: direct module execution writes default outputs."""

    (tmp_path / mod.STUDYING_RELATIVE_PATH).write_text("# Research Studying\n\n", encoding="utf-8")
    monkeypatch.setenv("CARNOT_EXP4778_ROOT", str(tmp_path))

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


def test_deliverable_files_validate_for_req_arc_wmte_4778() -> None:
    """REQ-ARC-WMTE-4778: committed deliverables satisfy the schema."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    studying = STUDYING_PATH.read_text(encoding="utf-8")

    mod.validate_artifact(artifact)
    mod.validate_research_studying_text(studying, artifact)
    assert artifact["honest_verdict"] == mod.HONEST_VERDICT
    assert artifact["flagged_for_v440"] == mod.FLAGGED_FOR_V440
