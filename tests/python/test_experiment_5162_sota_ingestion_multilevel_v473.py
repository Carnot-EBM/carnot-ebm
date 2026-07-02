"""Tests for Exp 5162 V473 multi-level ARC SOTA ingestion.

Spec refs: REQ-REPORT-5162, SCENARIO-REPORT-5162.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5162_sota_ingestion_multilevel_v473 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"


def _sample_upstream() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        {
            "gate_passed": False,
            "warmstart_vs_cold_delta_median": 0.0,
            "honest_verdict": "complete: warmstart_replay_ablation_gate_failed",
        },
        {
            "gate_passed": False,
            "games_improved_count": 1,
            "honest_verdict": "complete: goal_energy_ranker_warmstart_gate_failed",
        },
        {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
        },
    )


def test_req_report_5162_spec_declares_incremental_ingestion_contract() -> None:
    """REQ-REPORT-5162: OpenSpec declares the V473 SOTA-ingestion artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-REPORT-5162")
    section = spec[start:]

    assert "SCENARIO-REPORT-5162" in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert "V473 Outer-Loop Planner References" in section
    assert "V474 Planner References" in section
    assert "scripts/sweep_clusters.py" in section
    assert "scripts/sweep_semscholar.py" in section
    assert "/deep-research" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_build_artifact_records_required_fields_and_null_primary_sweep() -> None:
    """REQ-REPORT-5162: artifact records verified fields without padding findings."""

    artifact = mod.build_artifact(
        upstream_artifacts=_sample_upstream(),
        duration_s=2.5,
        tests_run=["tests/python/test_experiment_5162_sota_ingestion_multilevel_v473.py"],
    )

    mod.validate_artifact(artifact)
    assert set(mod.REQUIRED_PRINCIPLED_FIELDS).issubset(artifact)
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["milestone"] == mod.MILESTONE
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["no_deep_research_used"] is True
    assert artifact["conductor_modified"] is False
    assert artifact["references_md_updated"]["value"] is True
    assert artifact["incremental_findings"]["value"] == []
    assert "zero new post-2026-07-02 primary findings" in artifact["honest_verdict"]
    assert "adaptive retention" in artifact["bottom_line_recommendation"]["value"]
    assert "selective reset" in artifact["bottom_line_recommendation"]["value"]

    spot_checked = artifact["v473_citations_spot_checked"]["value"]
    assert len(spot_checked) >= 3
    assert all(row["resolved_correctly"] is True for row in spot_checked)
    assert {row["arxiv_id"] for row in spot_checked} >= {"2402.15957", "2504.02252", "2202.02405"}

    outcome_ids = {
        row["arxiv_id_or_url"]
        for row in artifact["outcome_conditioned_findings"]["value"]
    }
    assert outcome_ids == {"https://arxiv.org/abs/2601.22647", "https://arxiv.org/abs/2607.00457"}
    secondary_ids = {
        row["arxiv_id_or_url"]
        for row in artifact["secondary_findings"]["value"]
    }
    assert secondary_ids == {"https://arxiv.org/abs/2607.00895"}


def test_scenario_report_5162_appends_v474_section_idempotently() -> None:
    """SCENARIO-REPORT-5162: V474 references append without rewriting history."""

    artifact = mod.build_artifact(upstream_artifacts=_sample_upstream())
    original = "# Research References\n\n## V473 Outer-Loop Planner References - 2026-07-02\nOld.\n"

    updated = mod.append_v474_section(original, artifact)
    second = mod.append_v474_section(updated, artifact)

    assert updated == second
    assert updated.startswith(original)
    assert updated.count(mod.V474_HEADING) == 1
    assert "No new post-V473 primary paper was found" in updated
    assert "Test-Time Mixture of World Models" in updated
    assert "Multi-scale Mixture of World Models" in updated
    assert "Beyond Document Grounding" in updated
    assert mod.V474_END_MARKER in updated


def test_positive_upstream_branch_scales_modular_world_models() -> None:
    """REQ-REPORT-5162: passed upstream carryover changes the roadmap branch."""

    artifact = mod.build_artifact(
        upstream_artifacts=(
            {"gate_passed": True, "warmstart_vs_cold_delta_median": 0.2},
            {"gate_passed": True, "games_improved_count": 3},
            {"status": "complete", "honest_verdict": "complete: live_path_attempted"},
        )
    )

    assert artifact["upstream_outcome_summary"]["carryover_path_passed"] is True
    assert artifact["upstream_outcome_summary"]["recommended_mode"] == (
        "scale_modular_world_model_library"
    )
    assert "scale the successful carryover path" in artifact["bottom_line_recommendation"]["value"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda artifact: {
                key: value for key, value in artifact.items() if key != "queries_run"
            },
            "missing required fields",
        ),
        (
            lambda artifact: artifact | {"honest_verdict": "done"},
            "honest_verdict",
        ),
        (
            lambda artifact: artifact | {"inference_substrate": "live_llm_inference"},
            "inference_substrate",
        ),
        (
            lambda artifact: artifact | {"no_deep_research_used": False},
            "deep-research",
        ),
        (
            lambda artifact: artifact | {"conductor_modified": True},
            "conductor",
        ),
        (
            lambda artifact: artifact | {"field_principles": {}},
            "field_principles",
        ),
        (
            lambda artifact: artifact | {"references_md_updated": {"value": False, "principle": mod.FIELD_PRINCIPLES["references_md_updated"]}},
            "references_md_updated",
        ),
        (
            lambda artifact: artifact | {"v473_citations_spot_checked": {"value": [], "principle": mod.FIELD_PRINCIPLES["v473_citations_spot_checked"]}},
            "spot-check",
        ),
        (
            lambda artifact: artifact
            | {
                "v473_citations_spot_checked": {
                    "value": [
                        artifact["v473_citations_spot_checked"]["value"][0]
                        | {"resolved_correctly": False},
                        *artifact["v473_citations_spot_checked"]["value"][1:],
                    ],
                    "principle": mod.FIELD_PRINCIPLES["v473_citations_spot_checked"],
                }
            },
            "resolve correctly",
        ),
        (
            lambda artifact: artifact | {"incremental_findings": []},
            "principle-wrapped",
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
                "outcome_conditioned_findings": {
                    "value": [{"title": "Incomplete"}],
                    "principle": mod.FIELD_PRINCIPLES["outcome_conditioned_findings"],
                }
            },
            "rows must include",
        ),
        (
            lambda artifact: artifact
            | {
                "outcome_conditioned_findings": {
                    "value": [
                        artifact["outcome_conditioned_findings"]["value"][0]
                        | {"arxiv_id_or_url": "arxiv:made-up"}
                    ],
                    "principle": mod.FIELD_PRINCIPLES["outcome_conditioned_findings"],
                }
            },
            "verified URL",
        ),
        (
            lambda artifact: artifact | {"tests_run": []},
            "tests_run",
        ),
    ],
)
def test_validate_artifact_rejects_fabrication_and_forbidden_claims(
    mutate: object, message: str
) -> None:
    """REQ-REPORT-5162: invalid artifacts fail closed before writing."""

    artifact = mod.build_artifact(upstream_artifacts=_sample_upstream())

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(mutate(artifact))


def test_write_outputs_writes_json_and_references_section(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5162: writer emits stable JSON and appended references."""

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
    assert references.count(mod.V474_HEADING) == 1
    assert artifact["references_md_updated"]["value"] is True
