"""Tests for Exp 4818 energy-guided generation SOTA ingestion.

Spec refs: REQ-ARC-WMTE-4818, SCENARIO-ARC-WMTE-4818,
SCENARIO-ARC-WMTE-4818-NO-FABRICATION.
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_4818_sota_ingestion_energy_guided_generation as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"
ARTIFACT_PATH = REPO / mod.RESULT_RELATIVE_PATH
STUDYING_PATH = REPO / mod.STUDYING_RELATIVE_PATH


def _artifact() -> dict[str, object]:
    return mod.build_artifact()


def _spec_section() -> str:
    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-ARC-WMTE-4818")
    end = spec.index("### REQ-ARC-WMTE-4781", start)
    return spec[start:end]


def test_req_arc_wmte_4818_spec_declares_fresh_generation_contract() -> None:
    """REQ-ARC-WMTE-4818: OpenSpec declares the Exp 4818 artifact contract."""

    section = _spec_section()

    assert "REQ-ARC-WMTE-4818" in section
    assert "SCENARIO-ARC-WMTE-4818" in section
    assert "SCENARIO-ARC-WMTE-4818-NO-FABRICATION" in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert mod.STUDYING_RELATIVE_PATH in section
    assert mod.HONEST_VERDICT in section
    assert mod.INFERENCE_SUBSTRATE in section
    assert "flagged_for_v444" in section
    assert "put a winner into" in section
    assert "Bidirectional Evolutionary Search" in section
    assert "/deep-research" in section
    for field, principle in mod.REQUIRED_USER_FIELD_PRINCIPLES.items():
        assert field in section
        assert principle["principle"] in section
    for source_id in mod.REQUIRED_SOURCE_IDS:
        assert source_id in section


def test_req_arc_wmte_4818_artifact_maps_s3_methods_with_real_arxiv_ids() -> None:
    """REQ-ARC-WMTE-4818: artifact maps cited methods onto S3 generation."""

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
        assert method["maps_to_stages"] == ["S3"]
        assert "winner" in method["put_winner_into_pool"]
        assert method["graft_to_live_loop"]
        assert method["takes_over_from_current_stack"]
        assert method["fails_when"]
        assert method["roadmap_candidate"]
        cited.update(method["source_ids"])
        tracks.add(method["track"])

    assert 3 <= len(artifact["methods_mapped"]) <= 5
    assert cited.issubset(set(artifact["arxiv_ids_cited"]))
    assert mod.REQUIRED_TRACKS == tracks
    assert {flag["candidate"] for flag in artifact["flagged_for_v444"]} == {
        "bolt_cold_cfg_value_tree_generator_for_s3",
        "bes_energy_fitness_pool_inserter",
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

    context = artifact["s3_context"]
    assert context["s3_generation_allowed"] is True
    assert context["roadmap_target"] == ".444"
    assert "not as an environment oracle" in context["generation_constraint"]


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
                    "2605.28814": mod.CITATIONS["2605.28814"]
                    | {"url": "https://arxiv.org/abs/9999.99999"}
                }
            },
            "citation url",
        ),
        (
            _artifact()
            | {
                "citations": mod.CITATIONS
                | {"2605.28814": mod.CITATIONS["2605.28814"] | {"http_status": 404}}
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
                    mod.DEFAULT_METHODS_MAPPED[0] | {"maps_to_stages": ["S2"]},
                    *mod.DEFAULT_METHODS_MAPPED[1:],
                ]
            },
            "S3",
        ),
        (
            _artifact()
            | {
                "methods_mapped": [
                    mod.DEFAULT_METHODS_MAPPED[0] | {"put_winner_into_pool": ""},
                    *mod.DEFAULT_METHODS_MAPPED[1:],
                ]
            },
            "winner",
        ),
        (_artifact() | {"flagged_for_v444": []}, "flagged_for_v444"),
        (
            _artifact()
            | {"flagged_for_v444": [{"candidate": "stale", "flag": "flagged_for_v443"}]},
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
                | {"top_source_count": 4}
            },
            "top five to eight",
        ),
        (
            _artifact()
            | {"s3_context": dict(mod.DEFAULT_S3_CONTEXT) | {"s3_generation_allowed": False}},
            "S3",
        ),
    ],
)
def test_no_fabrication_validator_rejects_bad_energy_guided_generation_artifacts(
    bad_artifact: dict[str, object], message: str
) -> None:
    """SCENARIO-ARC-WMTE-4818-NO-FABRICATION: invalid claims fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_validate_artifact_rejects_exact_schema_violations() -> None:
    """REQ-ARC-WMTE-4818: top-level and method schemas are exact."""

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


def test_scenario_arc_wmte_4818_research_studying_section_is_idempotent() -> None:
    """SCENARIO-ARC-WMTE-4818: research-studying update is stable and cited."""

    original = "# Research Studying\n\nOld body\n"
    once = mod.update_research_studying_text(original, _artifact())
    twice = mod.update_research_studying_text(once, _artifact())

    assert once == twice
    assert once.count(mod.STUDYING_SECTION_START) == 1
    assert once.count(mod.STUDYING_SECTION_END) == 1
    mod.validate_research_studying_text(once, _artifact())
    assert "SOTA -> S3 energy-guided generation mapping" in once
    assert "flagged_for_v444" in once
    assert "arXiv:2605.28814" in once
    assert "arXiv:2305.12018" in once
    assert "put a winner into the pool" in once
    assert "no solve claim" in once

    later_section = "# Research Studying\n\n<!-- EXP0000-OLDER-START -->\nOld section\n"
    inserted = mod.update_research_studying_text(later_section, _artifact())
    refreshed = mod.update_research_studying_text(inserted, _artifact())

    assert refreshed == inserted
    assert inserted.index(mod.STUDYING_SECTION_START) < inserted.index("<!-- EXP0000-OLDER-START -->")


def test_write_outputs_is_idempotent_for_scenario_arc_wmte_4818(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4818: writer emits stable JSON and studying note."""

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    studying_path = tmp_path / mod.STUDYING_RELATIVE_PATH
    studying_path.parent.mkdir(parents=True, exist_ok=True)
    studying_path.write_text("# Research Studying\n\n", encoding="utf-8")

    artifact = mod.write_outputs(artifact_path=artifact_path, studying_path=studying_path)
    second_artifact = mod.write_outputs(artifact_path=artifact_path, studying_path=studying_path)

    assert second_artifact == artifact
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    mod.validate_research_studying_text(studying_path.read_text(encoding="utf-8"), artifact)


def test_main_writes_deliverables_for_req_arc_wmte_4818(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-ARC-WMTE-4818: direct module execution writes default outputs."""

    (tmp_path / mod.STUDYING_RELATIVE_PATH).write_text("# Research Studying\n\n", encoding="utf-8")
    monkeypatch.setenv("CARNOT_EXP4818_ROOT", str(tmp_path))

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


def test_deliverable_files_validate_for_req_arc_wmte_4818() -> None:
    """REQ-ARC-WMTE-4818: committed deliverables satisfy the schema."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    studying = STUDYING_PATH.read_text(encoding="utf-8")

    mod.validate_artifact(artifact)
    mod.validate_research_studying_text(studying, artifact)
    assert artifact["honest_verdict"] == mod.HONEST_VERDICT
    assert artifact["flagged_for_v444"] == mod.FLAGGED_FOR_V444
