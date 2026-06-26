"""Tests for Exp 4768 structural-energy SOTA ingestion.

Spec refs: REQ-ARC-WMTE-4768, SCENARIO-ARC-WMTE-4768,
SCENARIO-ARC-WMTE-4768-NO-FABRICATION.
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_4768_sota_ingestion_structural_energy as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"
ARTIFACT_PATH = REPO / mod.RESULT_RELATIVE_PATH
STUDYING_PATH = REPO / mod.STUDYING_RELATIVE_PATH


def _artifact() -> dict[str, object]:
    return mod.build_artifact()


def _spec_section() -> str:
    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-ARC-WMTE-4768")
    end = spec.index("### REQ-ARC-WMTE-4765", start)
    return spec[start:end]


def test_req_arc_wmte_4768_spec_declares_structural_energy_ingestion_contract() -> None:
    """REQ-ARC-WMTE-4768: OpenSpec declares the Exp 4768 artifact contract."""

    section = _spec_section()

    assert "REQ-ARC-WMTE-4768" in section
    assert "SCENARIO-ARC-WMTE-4768" in section
    assert "SCENARIO-ARC-WMTE-4768-NO-FABRICATION" in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert mod.STUDYING_RELATIVE_PATH in section
    assert mod.HONEST_VERDICT in section
    assert mod.INFERENCE_SUBSTRATE in section
    assert "S1 through S4" in section
    assert "flagged_for_v439" in section
    assert "/deep-research" in section
    for field, principle in mod.REQUIRED_USER_FIELD_PRINCIPLES.items():
        assert field in section
        assert principle["principle"] in section
    for source_id in mod.REQUIRED_SOURCE_IDS:
        assert source_id in section


def test_req_arc_wmte_4768_artifact_maps_methods_to_s1_s4_with_real_arxiv_ids() -> None:
    """REQ-ARC-WMTE-4768: artifact maps three to five cited methods onto S1-S4."""

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
    for method in artifact["methods_mapped"]:
        assert set(method) == mod.REQUIRED_METHOD_FIELDS
        assert method["source_ids"]
        assert set(method["source_ids"]).issubset(mod.REQUIRED_SOURCE_IDS)
        assert method["maps_to_stages"]
        assert method["takes_over_from_current_stack"]
        assert method["fails_when"]
        assert method["roadmap_candidate"]
        stages.update(method["maps_to_stages"])
        cited.update(method["source_ids"])

    assert 3 <= len(artifact["methods_mapped"]) <= 5
    assert {"S1", "S2", "S3", "S4"}.issubset(stages)
    assert cited.issubset(set(artifact["arxiv_ids_cited"]))
    assert {flag["candidate"] for flag in artifact["flagged_for_v439"]} == {
        "slot_factor_transition_energy_rerun",
        "poe_code_world_model_trust_gate",
    }

    preconditions = artifact["preconditions_checked"]
    assert preconditions["research_studying_present"] is True
    assert preconditions["research_references_present"] is True
    assert preconditions["sweep_clusters_used"] is True
    assert preconditions["sweep_cluster_ids"] == [1, 6]
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

    s0_context = artifact["s0_context"]
    assert s0_context["source_artifact"] == mod.S0_SOURCE_RELATIVE_PATH
    assert s0_context["stage"] == "S0"
    assert "origin_probe_leak" in s0_context["planning_constraint"]


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
                    "2605.05138": mod.CITATIONS["2605.05138"]
                    | {"url": "https://arxiv.org/abs/9999.99999"}
                }
            },
            "citation url",
        ),
        (
            _artifact()
            | {
                "citations": mod.CITATIONS
                | {"2605.05138": mod.CITATIONS["2605.05138"] | {"http_status": 404}}
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
        (_artifact() | {"flagged_for_v439": []}, "flagged_for_v439"),
        (
            _artifact()
            | {"flagged_for_v439": [{"candidate": "stale", "flag": "flagged_for_v438"}]},
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
                | {"model_load": True}
            },
            "model",
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
            | {"s0_context": dict(mod.DEFAULT_S0_CONTEXT) | {"stage": "S1"}},
            "S0",
        ),
    ],
)
def test_no_fabrication_validator_rejects_bad_structural_energy_artifacts(
    bad_artifact: dict[str, object], message: str
) -> None:
    """SCENARIO-ARC-WMTE-4768-NO-FABRICATION: invalid claims fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_validate_artifact_rejects_exact_schema_violations() -> None:
    """REQ-ARC-WMTE-4768: top-level and method schemas are exact."""

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


def test_scenario_arc_wmte_4768_research_studying_section_is_idempotent() -> None:
    """SCENARIO-ARC-WMTE-4768: research-studying update is stable and cited."""

    original = "# Research Studying\n\nOld body\n"
    once = mod.update_research_studying_text(original, _artifact())
    twice = mod.update_research_studying_text(once, _artifact())

    assert once == twice
    assert once.count(mod.STUDYING_SECTION_START) == 1
    assert once.count(mod.STUDYING_SECTION_END) == 1
    mod.validate_research_studying_text(once, _artifact())
    assert "SOTA -> S1-S4 structural-energy mapping" in once
    assert "flagged_for_v439" in once
    assert "arXiv:2602.02900" in once
    assert "arXiv:2510.04542" in once
    assert "no solve claim" in once


def test_write_outputs_is_idempotent_for_scenario_arc_wmte_4768(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4768: writer emits stable JSON and studying note."""

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    studying_path = tmp_path / mod.STUDYING_RELATIVE_PATH
    studying_path.parent.mkdir(parents=True, exist_ok=True)
    studying_path.write_text("# Research Studying\n\n", encoding="utf-8")

    artifact = mod.write_outputs(artifact_path=artifact_path, studying_path=studying_path)
    second_artifact = mod.write_outputs(artifact_path=artifact_path, studying_path=studying_path)

    assert second_artifact == artifact
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    mod.validate_research_studying_text(studying_path.read_text(encoding="utf-8"), artifact)


def test_main_writes_deliverables_for_req_arc_wmte_4768(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-ARC-WMTE-4768: direct module execution writes default outputs."""

    (tmp_path / mod.STUDYING_RELATIVE_PATH).write_text("# Research Studying\n\n", encoding="utf-8")
    monkeypatch.setenv("CARNOT_EXP4768_ROOT", str(tmp_path))

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


def test_deliverable_files_validate_for_req_arc_wmte_4768() -> None:
    """REQ-ARC-WMTE-4768: committed deliverables satisfy the schema."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    studying = STUDYING_PATH.read_text(encoding="utf-8")

    mod.validate_artifact(artifact)
    mod.validate_research_studying_text(studying, artifact)
    assert artifact["honest_verdict"] == mod.HONEST_VERDICT
    assert artifact["flagged_for_v439"] == mod.FLAGGED_FOR_V439
