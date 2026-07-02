"""Tests for Exp 5172 V474 SOTA ingestion and MAP deep-read.

Spec refs: REQ-REPORT-5172, SCENARIO-REPORT-5172-MAP-DEEP-READ,
SCENARIO-REPORT-5172-OUTPUTS.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5172_sota_ingestion_diffusion_hierarchical_search_v474 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"


def _sample_upstream() -> dict[str, dict[str, Any] | None]:
    return {
        "exp5171": {
            "experiment": "experiment_5171_harden_set_encoder_cross_corpus_n30_v474",
            "gate_passed": True,
            "held_out_task_n": 30,
            "cross_corpus_delta_n30": 0.5,
            "pass_rates": {"set_encoder_at_1": 0.7666666667, "vote_at_1": 0.2666666667},
            "honest_verdict": "success_arc_set_encoder_cross_corpus_gate_passed_n30",
        },
        "exp5173": None,
        "exp5175": None,
    }


def test_req_report_5172_spec_declares_map_deep_read_contract() -> None:
    """REQ-REPORT-5172: OpenSpec anchors the V474 MAP deep-read workflow."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-REPORT-5172")
    section = spec[start:]

    assert "SCENARIO-REPORT-5172-MAP-DEEP-READ" in section
    assert "SCENARIO-REPORT-5172-OUTPUTS" in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert "V474 Outer-Loop Planner References" in section
    assert "GAP-4891" in section
    assert "arc_relational_mask_pruner.py" in section
    assert "V475 Planner References" in section
    assert "/deep-research" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_build_artifact_records_map_deep_read_and_verified_sources() -> None:
    """SCENARIO-REPORT-5172-MAP-DEEP-READ: MAP is promoted from lead to design note."""

    artifact = mod.build_artifact(
        upstream_artifacts=_sample_upstream(),
        duration_s=3.25,
        tests_run=["tests/python/test_experiment_5172_sota_ingestion_diffusion_hierarchical_search_v474.py"],
    )

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["milestone"] == mod.MILESTONE
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["no_deep_research_used"] is True
    assert artifact["conductor_modified"] is False
    assert artifact["references_md_updated"]["value"] is True
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES

    spot_checked = artifact["v474_citations_spot_checked"]["value"]
    assert len(spot_checked) >= 3
    assert {row["arxiv_id"] for row in spot_checked} >= {
        "2605.18871",
        "2510.16449",
        "2605.20745",
    }
    assert all(row["resolved_correctly"] is True for row in spot_checked)

    deep_read = artifact["map_paper_deep_read"]["value"]
    assert set(mod.REQUIRED_MAP_FIELDS).issubset(deep_read)
    assert "Claude 4.6 Opus" in deep_read["model_architecture"]
    assert "Qwen3-4B-Thinking" in deep_read["model_architecture"]
    assert "8 NVIDIA H800 GPUs" in deep_read["model_architecture"]
    assert deep_read["quantitative_headline_result"]["improved_game_count"] == 22
    assert deep_read["quantitative_headline_result"]["game_count"] == 25
    assert deep_read["quantitative_headline_result"]["mean_react_score"] == pytest.approx(0.774)
    assert deep_read["quantitative_headline_result"]["mean_map_score"] == pytest.approx(4.3268)
    assert len(deep_read["quantitative_headline_result"]["arc_agi3_full_table"]) == 25
    assert deep_read["quantitative_headline_result"]["arc_agi3_full_table"]["CD82"]["map"] == {
        "level": 2,
        "score": 3.08,
    }
    assert "spatial layouts" in deep_read["cognitive_map_structure"]
    assert "object-action affordances" in deep_read["cognitive_map_structure"]
    assert "flat frontier" in deep_read["comparison_vs_relational_mask_pruner"]
    assert "pre-search" in deep_read["comparison_vs_relational_mask_pruner"]

    incremental_ids = {
        row["arxiv_id_or_url"].rsplit("/", 1)[-1]
        for row in artifact["incremental_findings"]["value"]
    }
    assert {"2607.01223", "2607.01224", "2606.09159", "2602.01842", "2510.01591"}.issubset(
        incremental_ids
    )
    outcome_ids = {
        row["arxiv_id_or_url"].rsplit("/", 1)[-1]
        for row in artifact["outcome_conditioned_findings"]["value"]
    }
    assert outcome_ids == {"2603.04304"}
    assert artifact["upstream_outcome_summary"]["exp5171"]["gate_passed"] is True
    assert artifact["upstream_outcome_summary"]["exp5173"]["present"] is False
    assert artifact["upstream_outcome_summary"]["exp5175"]["present"] is False
    assert "MAP-style pre-stage should be prototyped" in artifact[
        "bottom_line_recommendation_for_475"
    ]["value"]


def test_scenario_report_5172_appends_v475_section_idempotently() -> None:
    """SCENARIO-REPORT-5172-OUTPUTS: V475 references append once and lead with MAP."""

    artifact = mod.build_artifact(upstream_artifacts=_sample_upstream())
    original = (
        "# Research References\n\n"
        "## V474 Outer-Loop Planner References - 2026-07-02\nOld V474 content.\n"
    )

    updated = mod.append_v475_section(original, artifact)
    second = mod.append_v475_section(updated, artifact)

    assert updated == second
    assert updated.startswith(original)
    assert updated.count(mod.V475_HEADING) == 1
    after_heading = updated.split(mod.V475_HEADING, 1)[1]
    assert after_heading.lstrip().startswith("Added by Exp5172")
    assert "### MAP Deep-Read" in after_heading
    assert "Claude 4.6 Opus" in after_heading
    assert "Theoria" in after_heading
    assert "AutoMem" in after_heading
    assert "V1: Unifying Generation and Self-Verification" in after_heading
    assert mod.V475_END_MARKER in updated


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda artifact: {
                key: value for key, value in artifact.items() if key != "queries_run"
            },
            "missing required fields",
        ),
        (lambda artifact: artifact | {"honest_verdict": "done"}, "honest_verdict"),
        (
            lambda artifact: artifact | {"inference_substrate": "live_llm_inference"},
            "inference_substrate",
        ),
        (
            lambda artifact: artifact | {"no_deep_research_used": False},
            "deep-research",
        ),
        (lambda artifact: artifact | {"conductor_modified": True}, "conductor"),
        (
            lambda artifact: artifact | {"field_principles": {}},
            "field_principles",
        ),
        (
            lambda artifact: artifact
            | {
                "references_md_updated": {
                    "value": False,
                    "principle": mod.FIELD_PRINCIPLES["references_md_updated"],
                }
            },
            "references_md_updated",
        ),
        (
            lambda artifact: artifact
            | {
                "v474_citations_spot_checked": {
                    "value": [],
                    "principle": mod.FIELD_PRINCIPLES["v474_citations_spot_checked"],
                }
            },
            "spot-check",
        ),
        (
            lambda artifact: artifact
            | {
                "v474_citations_spot_checked": {
                    "value": [
                        artifact["v474_citations_spot_checked"]["value"][0]
                        | {"resolved_correctly": False},
                        *artifact["v474_citations_spot_checked"]["value"][1:],
                    ],
                    "principle": mod.FIELD_PRINCIPLES["v474_citations_spot_checked"],
                }
            },
            "resolve correctly",
        ),
        (
            lambda artifact: artifact | {"v474_citations_spot_checked": []},
            "principle-wrapped",
        ),
        (
            lambda artifact: artifact
            | {
                "v474_citations_spot_checked": {
                    "value": artifact["v474_citations_spot_checked"]["value"],
                    "principle": "wrong",
                }
            },
            "declared principle",
        ),
        (
            lambda artifact: artifact
            | {
                "map_paper_deep_read": {
                    "value": {"model_architecture": "missing the rest"},
                    "principle": mod.FIELD_PRINCIPLES["map_paper_deep_read"],
                }
            },
            "map_paper_deep_read",
        ),
        (
            lambda artifact: artifact
            | {
                "map_paper_deep_read": {
                    "value": artifact["map_paper_deep_read"]["value"]
                    | {"quantitative_headline_result": "bad"},
                    "principle": mod.FIELD_PRINCIPLES["map_paper_deep_read"],
                }
            },
            "quantitative result",
        ),
        (
            lambda artifact: artifact
            | {
                "map_paper_deep_read": {
                    "value": artifact["map_paper_deep_read"]["value"]
                    | {
                        "quantitative_headline_result": artifact[
                            "map_paper_deep_read"
                        ]["value"]["quantitative_headline_result"]
                        | {"arc_agi3_full_table": {}}
                    },
                    "principle": mod.FIELD_PRINCIPLES["map_paper_deep_read"],
                }
            },
            "all 25",
        ),
        (
            lambda artifact: artifact
            | {
                "map_paper_deep_read": {
                    "value": artifact["map_paper_deep_read"]["value"]
                    | {
                        "quantitative_headline_result": artifact[
                            "map_paper_deep_read"
                        ]["value"]["quantitative_headline_result"]
                        | {"improved_game_count": 21}
                    },
                    "principle": mod.FIELD_PRINCIPLES["map_paper_deep_read"],
                }
            },
            "22/25",
        ),
        (
            lambda artifact: artifact
            | {
                "map_paper_deep_read": {
                    "value": artifact["map_paper_deep_read"]["value"]
                    | {"comparison_vs_relational_mask_pruner": "too vague"},
                    "principle": mod.FIELD_PRINCIPLES["map_paper_deep_read"],
                }
            },
            "distinguish",
        ),
        (
            lambda artifact: artifact
            | {
                "incremental_findings": {
                    "value": "none",
                    "principle": mod.FIELD_PRINCIPLES["incremental_findings"],
                }
            },
            "value must be a list",
        ),
        (
            lambda artifact: artifact
            | {
                "incremental_findings": {
                    "value": [{"title": "Incomplete"}],
                    "principle": mod.FIELD_PRINCIPLES["incremental_findings"],
                }
            },
            "rows must include",
        ),
        (
            lambda artifact: artifact
            | {
                "incremental_findings": {
                    "value": [
                        artifact["incremental_findings"]["value"][0]
                        | {"arxiv_id_or_url": "arxiv:made-up"}
                    ],
                    "principle": mod.FIELD_PRINCIPLES["incremental_findings"],
                }
            },
            "verified URL",
        ),
        (lambda artifact: artifact | {"tests_run": []}, "tests_run"),
    ],
)
def test_validate_artifact_rejects_invalid_or_unverified_rows(
    mutate: object, message: str
) -> None:
    """REQ-REPORT-5172: invalid artifacts fail closed before writing."""

    artifact = mod.build_artifact(upstream_artifacts=_sample_upstream())

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(mutate(artifact))


def test_load_default_upstream_artifacts_records_absent_files(tmp_path: Path) -> None:
    """REQ-REPORT-5172: outcome conditioning records missing upstream artifacts honestly."""

    exp5171_path = tmp_path / "results/experiment_5171_harden_set_encoder_cross_corpus_n30_v474.json"
    exp5171_path.parent.mkdir(parents=True)
    exp5171_path.write_text(json.dumps(_sample_upstream()["exp5171"]), encoding="utf-8")

    upstream = mod.load_default_upstream_artifacts(tmp_path)

    assert upstream["exp5171"]["gate_passed"] is True
    assert upstream["exp5173"] is None
    assert upstream["exp5175"] is None


def test_present_exp5173_and_exp5175_are_summarized_when_available() -> None:
    """REQ-REPORT-5172: present upstream outcomes override absent placeholders."""

    upstream = _sample_upstream() | {
        "exp5173": {
            "gate_passed": False,
            "status": "complete",
            "honest_verdict": "complete: diffusiongemma_energy_guidance_no_win",
        },
        "exp5175": {
            "gate_passed": True,
            "status": "complete",
            "honest_verdict": "success_relational_mask_pruner_banks_level",
        },
    }

    artifact = mod.build_artifact(upstream_artifacts=upstream)

    assert artifact["upstream_outcome_summary"]["exp5173"] == {
        "present": True,
        "gate_passed": False,
        "status": "complete",
        "honest_verdict": "complete: diffusiongemma_energy_guidance_no_win",
    }
    assert artifact["upstream_outcome_summary"]["exp5175"] == {
        "present": True,
        "gate_passed": True,
        "status": "complete",
        "honest_verdict": "success_relational_mask_pruner_banks_level",
    }


def test_write_outputs_writes_json_and_references_section(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5172-OUTPUTS: writer emits stable JSON and appended references."""

    references_path = tmp_path / mod.REFERENCES_RELATIVE_PATH
    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    references_path.write_text("# Research References\n", encoding="utf-8")

    artifact = mod.write_outputs(
        root=tmp_path,
        references_path=references_path,
        result_path=result_path,
        upstream_artifacts=_sample_upstream(),
        tests_run=["focused"],
    )
    second = mod.write_outputs(
        root=tmp_path,
        references_path=references_path,
        result_path=result_path,
        upstream_artifacts=_sample_upstream(),
        tests_run=["focused"],
    )

    assert second == artifact
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    references = references_path.read_text(encoding="utf-8")
    assert references.count(mod.V475_HEADING) == 1
    assert artifact["references_md_updated"]["value"] is True
