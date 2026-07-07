"""Tests for Exp 5336 V487 execution-time source delta refresh.

Spec refs: REQ-REPORT-5336, SCENARIO-REPORT-5336-APPEND-DELTAS,
SCENARIO-REPORT-5336-NOOP.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5336_sota_source_delta_v487 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_report_5336_spec_declares_v487_refresh_contract() -> None:
    """REQ-REPORT-5336: OpenSpec anchors the V487 source delta refresh."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-5336") : spec.index("### REQ-REPORT-5109")]

    for marker in (
        "REQ-REPORT-5336",
        "SCENARIO-REPORT-5336-APPEND-DELTAS",
        "SCENARIO-REPORT-5336-NOOP",
        str(mod.RESULT_RELATIVE_PATH),
        "literature_ingestion_network_sources",
        "HuggingFace Papers",
        "Extropic",
        "Logical Intelligence",
        "scripts/research_conductor.py",
        "ops/changelog.md",
        "ops/status.md",
        "_bmad/traceability.md",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_build_artifact_records_required_fields_and_v487_deltas() -> None:
    """REQ-REPORT-5336: artifact records source-verified actionable deltas."""

    artifact = mod.build_artifact(
        methodology_duration_s=15.25,
        tests_run=["tests/python/test_experiment_5336_sota_source_delta_v487.py"],
    )

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"]["value"] == mod.EXPERIMENT_ID
    assert artifact["milestone"]["value"] == mod.MILESTONE
    assert artifact["status"]["value"] == "complete"
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["new_actionable_findings_count"] == len(mod.ACTIONABLE_FINDINGS)
    assert artifact["references_modified"] is True
    assert artifact["retired_scope_reopened"] is False
    assert artifact["methodology_duration_s"] == 15.25
    assert artifact["executable_plan_change_required"] is False
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES

    sources = artifact["sources_checked"]["value"]
    assert set(mod.REQUIRED_SOURCE_FAMILIES).issubset(sources)
    assert sources["arxiv"]["status"] == "ok"
    assert sources["openreview"]["status"] == "ok"
    assert sources["semantic_scholar"]["status"] == "rate_limited"
    assert sources["github"]["status"] == "ok"
    assert sources["huggingface_papers"]["status"] == "ok"
    assert "Stop Hand-Holding Your Coding Agent" in " ".join(sources["huggingface_papers"]["not_promoted"])

    findings = artifact["actionable_findings"]["value"]
    assert [row["title"] for row in findings] == [row["title"] for row in mod.ACTIONABLE_FINDINGS]
    assert {row["arxiv_id_or_repo"] for row in findings} == {
        "2512.18950",
        "openreview:cbtykHVWX9",
    }
    assert all(row["source_url"].startswith("https://") for row in findings)
    assert {row["planned_task_impact"] for row in findings} == {"no_plan_edit"}
    assert {row["retired_scope_risk"] for row in findings} == {"none"}


def test_scenario_report_5336_appends_refresh_section_idempotently() -> None:
    """SCENARIO-REPORT-5336-APPEND-DELTAS: references append once."""

    artifact = mod.build_artifact()
    original = "# Research References\n\n### V487 Planner Refresh - 2026-07-07\nOld V487 section.\n"

    updated = mod.append_refresh_section(original, artifact)
    second = mod.append_refresh_section(updated, artifact)

    assert updated == second
    assert updated.startswith(original)
    assert updated.count(mod.REFRESH_HEADING) == 1
    assert "MACLA" in updated
    assert "Hard-Constrained Graph Generation" in updated
    assert "No executable `.487` task edit is required" in updated
    assert mod.REFRESH_END_MARKER in updated


def test_scenario_report_5336_noop_leaves_references_unchanged() -> None:
    """SCENARIO-REPORT-5336-NOOP: zero-new refresh does not churn references."""

    artifact = mod.build_artifact(actionable_findings=[])
    original = "# Research References\n\n### V487 Planner Refresh - 2026-07-07\nOld V487 section.\n"

    mod.validate_artifact(artifact)
    assert artifact["new_actionable_findings_count"] == 0
    assert artifact["references_modified"] is False
    assert artifact["actionable_findings"]["value"] == []
    assert "no new actionable" in artifact["honest_verdict"]["value"]
    assert mod.append_refresh_section(original, artifact) == original


def test_write_outputs_writes_artifact_and_reference_append(tmp_path: Path) -> None:
    """REQ-REPORT-5336: writer emits the JSON result and one references append."""

    references = tmp_path / mod.REFERENCES_RELATIVE_PATH
    result = tmp_path / mod.RESULT_RELATIVE_PATH
    references.write_text(
        "# Research References\n\n### V487 Planner Refresh - 2026-07-07\nOld V487 section.\n",
        encoding="utf-8",
    )

    artifact = mod.write_outputs(
        root=tmp_path,
        references_path=references,
        result_path=result,
        methodology_duration_s=9.25,
        tests_run=["tests/python/test_experiment_5336_sota_source_delta_v487.py"],
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
            lambda artifact: artifact
            | {
                "experiment_id": {
                    "value": "wrong",
                    "principle": mod.FIELD_PRINCIPLES["experiment_id"],
                }
            },
            "experiment_id",
        ),
        (lambda artifact: artifact | {"milestone": mod.MILESTONE}, "principle-wrapped"),
        (
            lambda artifact: artifact
            | {
                "status": {
                    "value": "running",
                    "principle": mod.FIELD_PRINCIPLES["status"],
                }
            },
            "status",
        ),
        (
            lambda artifact: artifact
            | {
                "honest_verdict": {
                    "value": "done",
                    "principle": mod.FIELD_PRINCIPLES["honest_verdict"],
                }
            },
            "honest_verdict",
        ),
        (
            lambda artifact: artifact
            | {
                "inference_substrate": {
                    "value": "aggregation_from_upstream_artifacts",
                    "principle": mod.FIELD_PRINCIPLES["inference_substrate"],
                }
            },
            "inference_substrate",
        ),
        (
            lambda artifact: artifact
            | {
                "references_section_marker": {
                    "value": "<!-- wrong -->",
                    "principle": mod.FIELD_PRINCIPLES["references_section_marker"],
                }
            },
            "references_section_marker",
        ),
        (
            lambda artifact: artifact
            | {
                "sources_checked": {
                    "value": {"arxiv": artifact["sources_checked"]["value"]["arxiv"]},
                    "principle": mod.FIELD_PRINCIPLES["sources_checked"],
                }
            },
            "sources_checked",
        ),
        (
            lambda artifact: artifact
            | {
                "sources_checked": {
                    "value": artifact["sources_checked"]["value"]
                    | {"github": {"status": "skipped", "queries": []}},
                    "principle": mod.FIELD_PRINCIPLES["sources_checked"],
                }
            },
            "status ok or rate_limited",
        ),
        (lambda artifact: artifact | {"new_actionable_findings_count": 99}, "findings count"),
        (lambda artifact: artifact | {"references_modified": False}, "references_modified"),
        (lambda artifact: artifact | {"retired_scope_reopened": True}, "retired_scope_reopened"),
        (lambda artifact: artifact | {"methodology_duration_s": -1.0}, "methodology_duration_s"),
        (
            lambda artifact: artifact | {"executable_plan_change_required": True},
            "executable plan",
        ),
        (
            lambda artifact: artifact
            | {
                "actionable_findings": {
                    "value": {"not": "a list"},
                    "principle": mod.FIELD_PRINCIPLES["actionable_findings"],
                }
            },
            "must be a list",
        ),
        (
            lambda artifact: artifact
            | {
                "actionable_findings": {
                    "value": [{"title": "missing fields"}],
                    "principle": mod.FIELD_PRINCIPLES["actionable_findings"],
                }
            },
            "rows must include",
        ),
        (
            lambda artifact: artifact
            | {
                "actionable_findings": {
                    "value": [
                        artifact["actionable_findings"]["value"][0] | {"source_url": "arxiv:bad"}
                    ],
                    "principle": mod.FIELD_PRINCIPLES["actionable_findings"],
                }
            },
            "verified URL",
        ),
        (
            lambda artifact: artifact
            | {
                "actionable_findings": {
                    "value": [
                        artifact["actionable_findings"]["value"][0]
                        | {"planned_task_impact": "plan_edit"}
                    ],
                    "principle": mod.FIELD_PRINCIPLES["actionable_findings"],
                }
            },
            "active plan",
        ),
        (
            lambda artifact: artifact
            | {
                "actionable_findings": {
                    "value": [
                        artifact["actionable_findings"]["value"][0]
                        | {"retired_scope_risk": "reopened"}
                    ],
                    "principle": mod.FIELD_PRINCIPLES["actionable_findings"],
                }
            },
            "retired scopes",
        ),
        (
            lambda artifact: artifact | {"field_principles": {"experiment_id": "wrong"}},
            "field_principles",
        ),
        (lambda artifact: artifact | {"no_deep_research_used": False}, "deep-research"),
        (lambda artifact: artifact | {"research_conductor_modified": True}, "research_conductor"),
        (lambda artifact: artifact | {"ops_docs_modified": True}, "ops docs"),
        (lambda artifact: artifact | {"traceability_modified": True}, "traceability"),
        (lambda artifact: artifact | {"tests_run": []}, "tests_run"),
    ],
)
def test_validate_artifact_rejects_contract_drift(mutate, message: str) -> None:
    """REQ-REPORT-5336: schema drift fails closed before artifact write."""

    artifact = mod.build_artifact()
    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(mutate(artifact))


def test_deliverable_json_matches_required_schema() -> None:
    """REQ-REPORT-5336: committed deliverable uses the tested schema."""

    payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    mod.validate_artifact(payload)
    assert payload["experiment_id"]["value"] == mod.EXPERIMENT_ID
    assert payload["honest_verdict"]["value"].startswith("complete:")
