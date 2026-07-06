"""Tests for Exp 5296 V484 execution-time source delta refresh.

Spec refs: REQ-REPORT-5296, SCENARIO-REPORT-5296-APPEND-DELTAS,
SCENARIO-REPORT-5296-NOOP.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5296_sota_source_delta_v484 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_report_5296_spec_declares_v484_refresh_contract() -> None:
    """REQ-REPORT-5296: OpenSpec anchors the V484 source delta refresh."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-5296") : spec.index("### REQ-REPORT-5283")]

    for marker in (
        "REQ-REPORT-5296",
        "SCENARIO-REPORT-5296-APPEND-DELTAS",
        "SCENARIO-REPORT-5296-NOOP",
        str(mod.RESULT_RELATIVE_PATH),
        "literature_ingestion_network_sources",
        "Semantic Scholar",
        "arXiv:2507.02092",
        "arXiv:2512.15605",
        "/deep-research",
        "research-roadmap.yaml",
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_build_artifact_records_required_fields_and_v484_deltas() -> None:
    """REQ-REPORT-5296: artifact records source-verified actionable deltas."""

    artifact = mod.build_artifact(
        duration_s=6.75,
        tests_run=["tests/python/test_experiment_5296_sota_source_delta_v484.py"],
    )

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["milestone"] == mod.MILESTONE
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert "new actionable" in artifact["honest_verdict"]["value"]
    assert artifact["new_references_added"]["value"] == len(mod.ACTIONABLE_DELTAS)
    assert artifact["references_md_updated"]["value"] is True
    assert artifact["retired_scope_reopened"]["value"] is False
    assert artifact["no_deep_research_used"] is True
    assert artifact["research_roadmap_yaml_modified"] is False
    assert artifact["research_conductor_modified"] is False
    assert artifact["plan_change_required"] is False
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES

    sources = artifact["sources_checked"]["value"]
    assert set(mod.REQUIRED_SOURCE_FAMILIES).issubset(sources)
    assert sources["arxiv"]["status"] == "ok"
    assert sources["semantic_scholar"]["status"] == "ok"
    assert sources["github"]["status"] == "ok"
    assert sources["huggingface_papers"]["status"] == "ok"

    deltas = artifact["actionable_deltas"]["value"]
    assert [row["title"] for row in deltas] == [row["title"] for row in mod.ACTIONABLE_DELTAS]
    assert all(row["source_url"].startswith("https://") for row in deltas)
    assert {row["planned_task_impact"] for row in deltas} == {"no_plan_edit"}
    assert {row["retired_scope_risk"] for row in deltas} == {"none"}
    assert any(row["arxiv_id_or_repo"] == "DOI:10.1109/ISPASS69572.2026.00062" for row in deltas)
    assert any(row["arxiv_id_or_repo"] == "2605.12493" for row in deltas)
    assert any(row["arxiv_id_or_repo"] == "2601.11969" for row in deltas)
    assert any(row["arxiv_id_or_repo"] == "2606.30333" for row in deltas)

    semantic = artifact["semantic_scholar_status"]["value"]
    assert semantic["EBT"]["arxiv_id"] == "2507.02092"
    assert semantic["EBT"]["status"] == "ok"
    assert semantic["EBT"]["citationCount"] == 26
    assert semantic["ARM-EBM"]["arxiv_id"] == "2512.15605"
    assert semantic["ARM-EBM"]["status"] == "ok"
    assert semantic["ARM-EBM"]["citationCount"] == 8


def test_scenario_report_5296_appends_refresh_section_idempotently() -> None:
    """SCENARIO-REPORT-5296-APPEND-DELTAS: references append once."""

    artifact = mod.build_artifact()
    original = "# Research References\n\n## V484 Research Update - 2026-07-06\nOld V484 section.\n"

    updated = mod.append_refresh_section(original, artifact)
    second = mod.append_refresh_section(updated, artifact)

    assert updated == second
    assert updated.startswith(original)
    assert updated.count(mod.REFRESH_HEADING) == 1
    assert "Towards System-2 AI" in updated
    assert "LongMemEval-V2" in updated
    assert "MemoryRewardBench" in updated
    assert "Local-Minima-Preserving Continuous Relaxation" in updated
    assert "No executable `.484` task edit is required" in updated
    assert mod.REFRESH_END_MARKER in updated


def test_scenario_report_5296_noop_leaves_references_unchanged() -> None:
    """SCENARIO-REPORT-5296-NOOP: zero-new refresh does not churn references."""

    artifact = mod.build_artifact(actionable_deltas=[])
    original = "# Research References\n\n## V484 Research Update - 2026-07-06\nOld V484 section.\n"

    mod.validate_artifact(artifact)
    assert artifact["new_references_added"]["value"] == 0
    assert artifact["references_md_updated"]["value"] is False
    assert artifact["actionable_deltas"]["value"] == []
    assert "no new actionable" in artifact["honest_verdict"]["value"]
    assert mod.append_refresh_section(original, artifact) == original


def test_write_outputs_writes_artifact_and_reference_append(tmp_path: Path) -> None:
    """REQ-REPORT-5296: writer emits the JSON result and one references append."""

    references = tmp_path / mod.REFERENCES_RELATIVE_PATH
    result = tmp_path / mod.RESULT_RELATIVE_PATH
    references.write_text(
        "# Research References\n\n## V484 Research Update - 2026-07-06\nOld V484 section.\n",
        encoding="utf-8",
    )

    artifact = mod.write_outputs(
        root=tmp_path,
        references_path=references,
        result_path=result,
        tests_run=["tests/python/test_experiment_5296_sota_source_delta_v484.py"],
    )

    mod.validate_artifact(artifact)
    assert result.exists()
    assert json.loads(result.read_text(encoding="utf-8")) == artifact
    updated = references.read_text(encoding="utf-8")
    assert updated.count(mod.REFRESH_HEADING) == 1
    assert updated.startswith("# Research References")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda artifact: {k: v for k, v in artifact.items() if k != "sources_checked"},
            "missing required fields",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "honest_verdict": {
                        "value": "done",
                        "principle": mod.FIELD_PRINCIPLES["honest_verdict"],
                    }
                }
            ),
            "honest_verdict",
        ),
        (
            lambda artifact: artifact | {"honest_verdict": "complete: unwrapped"},
            "principle-wrapped",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "inference_substrate": {
                        "value": "cached_fixture_replay_no_llm",
                        "principle": mod.FIELD_PRINCIPLES["inference_substrate"],
                    }
                }
            ),
            "inference_substrate",
        ),
        (lambda artifact: artifact | {"no_deep_research_used": False}, "deep-research"),
        (
            lambda artifact: artifact | {"research_roadmap_yaml_modified": True},
            "research-roadmap.yaml",
        ),
        (lambda artifact: artifact | {"research_conductor_modified": True}, "research_conductor"),
        (lambda artifact: artifact | {"field_principles": {}}, "field_principles"),
        (
            lambda artifact: (
                artifact
                | {
                    "references_md_updated": {
                        "value": False,
                        "principle": mod.FIELD_PRINCIPLES["references_md_updated"],
                    }
                }
            ),
            "references_md_updated",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "new_references_added": {
                        "value": 0,
                        "principle": mod.FIELD_PRINCIPLES["new_references_added"],
                    }
                }
            ),
            "new_references_added",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "actionable_deltas": {
                        "value": "none",
                        "principle": mod.FIELD_PRINCIPLES["actionable_deltas"],
                    }
                }
            ),
            "actionable_deltas",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "actionable_deltas": {
                        "value": [
                            artifact["actionable_deltas"]["value"][0] | {"source_url": "arxiv:bad"}
                        ],
                        "principle": mod.FIELD_PRINCIPLES["actionable_deltas"],
                    }
                }
            ),
            "verified URL",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "actionable_deltas": {
                        "value": [
                            artifact["actionable_deltas"]["value"][0]
                            | {"planned_task_impact": "plan_edit"}
                        ],
                        "principle": mod.FIELD_PRINCIPLES["actionable_deltas"],
                    }
                }
            ),
            "active plan",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "actionable_deltas": {
                        "value": [
                            artifact["actionable_deltas"]["value"][0]
                            | {"retired_scope_risk": "reopened"}
                        ],
                        "principle": mod.FIELD_PRINCIPLES["actionable_deltas"],
                    }
                }
            ),
            "retired scopes",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "sources_checked": {
                        "value": {"arxiv": artifact["sources_checked"]["value"]["arxiv"]},
                        "principle": mod.FIELD_PRINCIPLES["sources_checked"],
                    }
                }
            ),
            "sources_checked",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "semantic_scholar_status": {
                        "value": {"EBT": artifact["semantic_scholar_status"]["value"]["EBT"]},
                        "principle": mod.FIELD_PRINCIPLES["semantic_scholar_status"],
                    }
                }
            ),
            "Semantic Scholar",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "semantic_scholar_status": {
                        "value": artifact["semantic_scholar_status"]["value"]
                        | {
                            "EBT": artifact["semantic_scholar_status"]["value"]["EBT"]
                            | {"status": "unknown"}
                        },
                        "principle": mod.FIELD_PRINCIPLES["semantic_scholar_status"],
                    }
                }
            ),
            "Semantic Scholar EBT",
        ),
        (lambda artifact: artifact | {"tests_run": []}, "tests_run"),
        (
            lambda artifact: (
                artifact
                | {
                    "retired_scope_reopened": {
                        "value": True,
                        "principle": mod.FIELD_PRINCIPLES["retired_scope_reopened"],
                    }
                }
            ),
            "retired_scope_reopened",
        ),
    ],
)
def test_validate_artifact_rejects_contract_drift(mutate, message: str) -> None:
    """REQ-REPORT-5296: schema drift fails closed before artifact write."""

    artifact = mod.build_artifact()
    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(mutate(artifact))


def test_deliverable_json_matches_required_schema() -> None:
    """REQ-REPORT-5296: committed deliverable uses the tested schema."""

    payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    mod.validate_artifact(payload)
    assert payload["experiment_id"] == mod.EXPERIMENT_ID
    assert payload["honest_verdict"]["value"].startswith("complete:")
