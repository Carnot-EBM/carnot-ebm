"""Tests for Exp5537 V502 execution-time source delta ingestion.

Spec refs: REQ-REPORT-5537, SCENARIO-REPORT-5537-APPEND-DELTAS,
SCENARIO-REPORT-5537-NO-NEW-DELTA, SCENARIO-REPORT-5537-BLOCKED-MARKER.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5537_v502_source_delta_ingestion as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"


def _references_text() -> str:
    return (
        "# Research References\n\n"
        "## V502 Planner Refresh - 2026-07-10\n"
        "Existing V502 section.\n"
    )


def _accepted_finding() -> dict[str, str]:
    return {
        "title": "Example ASP Energy Fixture",
        "url": "https://arxiv.org/abs/2699.50201",
        "source_type": "arXiv preprint",
        "carnot_hook": "Add ASP declarative semantics rows to the finite-state exact fixture.",
        "planned_experiment": "llm_fsm_exact_fixture",
        "mapped_task": "exp5541-llm-fsm-exact-fixture",
    }


def test_req_report_5537_spec_declares_v502_source_delta_contract() -> None:
    """REQ-REPORT-5537: OpenSpec anchors the V502 source-delta receipt."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-5537") :]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-REPORT-5537-APPEND-DELTAS",
        "SCENARIO-REPORT-5537-NO-NEW-DELTA",
        "SCENARIO-REPORT-5537-BLOCKED-MARKER",
        str(mod.RESULT_RELATIVE_PATH),
        "V502 Execution Refresh - 20260710",
        "OpenReview",
        "ARM-EBM `2512.15605`",
        "HuggingFace Papers",
        "GitHub",
        "Logical Intelligence",
        "semantic_scholar_status",
        "closed_scopes_reopened=false",
        "aggregation_from_upstream_artifacts",
        "scripts/research_conductor.py",
        "ops/changelog.md",
        "ops/status.md",
        "_bmad/traceability.md",
    ):
        assert marker in section
    assert "Semantic Scholar public routes for EBT `2507.02092`" in normalized
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert f'principle "{principle}"' in normalized


def test_scenario_report_5537_default_appends_and_maps_live_delta() -> None:
    """SCENARIO-REPORT-5537-APPEND-DELTAS: accepted findings map to .502 lanes."""

    artifact = mod.build_artifact(methodology_duration_s=12.5)
    updated = mod.append_refresh_section(_references_text(), artifact)
    second = mod.append_refresh_section(updated, artifact)

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["task_id"] == mod.TASK_ID
    assert artifact["milestone"] == "2026.07.502"
    assert artifact["status"] == "complete"
    assert artifact["new_actionable_findings_count"] == 1
    assert artifact["research_references_updated"] is True
    assert artifact["prior_refresh_marker_found"] is True
    assert artifact["closed_scopes_reopened"] is False
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["semantic_scholar_status"].startswith("ok:")
    assert artifact["experiment_mappings"] == [
        {
            "title": "Answer Set Programming Energised! End-to-End Neurosymbolic Reasoning and Learning with ASP and Energy Based Models",
            "planned_experiment": "llm_fsm_exact_fixture",
            "mapped_task": "exp5541-llm-fsm-exact-fixture",
            "rationale": "Add ASP declarative semantics and non-monotonic constraint rows to the deterministic finite-state exact fixture; reuse the same rows as richer sparse-FSM descriptors without reopening training or proprietary baselines.",
        }
    ]
    assert updated == second
    assert updated.count(mod.REFRESH_HEADING) == 1
    assert "Answer Set Programming Energised" in updated
    assert "exp5541-llm-fsm-exact-fixture" in updated
    assert "exp5545-gated-sparse-repair-fsm-descriptor-scale" in updated
    assert "No closed scope was reopened" in updated
    assert mod.REFRESH_END_MARKER in updated


def test_scenario_report_5537_no_new_delta_records_noop_receipt() -> None:
    """SCENARIO-REPORT-5537-NO-NEW-DELTA: duplicate-only sweep is stable."""

    artifact = mod.build_artifact(
        new_references_added=[],
        methodology_duration_s=7.0,
        tests_run=["tests/python/test_experiment_5537_v502_source_delta_ingestion.py"],
    )

    mod.validate_artifact(artifact)
    assert artifact["new_references_added"] == []
    assert artifact["new_actionable_findings_count"] == 0
    assert artifact["experiment_mappings"] == []
    assert artifact["research_references_updated"] is False
    assert artifact["methodology_duration_s"] == 7.0
    assert set(mod.REQUIRED_SOURCE_FAMILIES).issubset(artifact["sources_checked"])
    assert "no new actionable" in artifact["honest_verdict"]
    assert mod.render_refresh_section(artifact) == ""
    assert mod.append_refresh_section(_references_text(), artifact) == _references_text()

    duplicate_text = " ".join(artifact["duplicates_suppressed"])
    for duplicate in (
        "LLM-FSM",
        "Gram2Token",
        "2607.07026",
        "Distributional EBMs",
        "EBT",
        "ARM-EBM",
        "Extropic",
        "Logical Intelligence",
    ):
        assert duplicate in duplicate_text
    watch_text = " ".join(row["reason"] for row in artifact["watch_only_or_excluded"])
    assert "fine-tuning" in watch_text
    assert "RLAIF" in watch_text
    assert "non-local" in watch_text


def test_scenario_report_5537_blocks_without_prior_marker(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5537-BLOCKED-MARKER: missing V502 marker fails closed."""

    artifact = mod.build_artifact(prior_refresh_marker_found=False)
    original = "# Research References\n\n## V501 Planner Refresh - 2026-07-10\nOld section.\n"

    mod.validate_artifact(artifact)
    assert artifact["status"] == "blocked"
    assert artifact["prior_refresh_marker_found"] is False
    assert artifact["research_references_updated"] is False
    assert artifact["new_references_added"] == []
    assert artifact["experiment_mappings"] == []
    assert artifact["honest_verdict"].startswith("blocked:")
    assert mod.append_refresh_section(original, artifact) == original

    references = tmp_path / mod.REFERENCES_RELATIVE_PATH
    result = tmp_path / mod.RESULT_RELATIVE_PATH
    references.write_text(original, encoding="utf-8")

    written = mod.write_outputs(root=tmp_path, references_path=references, result_path=result)
    assert written["status"] == "blocked"
    assert written["prior_refresh_marker_found"] is False
    assert references.read_text(encoding="utf-8") == original


def test_write_outputs_writes_artifact_and_reference_append(tmp_path: Path) -> None:
    """REQ-REPORT-5537: writer emits JSON and persists accepted deltas."""

    references = tmp_path / mod.REFERENCES_RELATIVE_PATH
    result = tmp_path / mod.RESULT_RELATIVE_PATH
    references.write_text(_references_text(), encoding="utf-8")

    artifact = mod.write_outputs(
        root=tmp_path,
        references_path=references,
        result_path=result,
        methodology_duration_s=9.5,
        tests_run=["tests/python/test_experiment_5537_v502_source_delta_ingestion.py"],
    )

    mod.validate_artifact(artifact)
    assert result.exists()
    assert json.loads(result.read_text(encoding="utf-8")) == artifact
    assert mod.REFRESH_HEADING in references.read_text(encoding="utf-8")


def test_append_refresh_section_requires_prior_marker() -> None:
    """SCENARIO-REPORT-5537-BLOCKED-MARKER: append requires planner marker."""

    artifact = mod.build_artifact(new_references_added=[_accepted_finding()])
    with pytest.raises(ValueError, match="V502 planner refresh marker"):
        mod.append_refresh_section("# Research References\n", artifact)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda artifact: {k: v for k, v in artifact.items() if k != "sources_checked"},
            "missing required",
        ),
        (lambda artifact: artifact | {"experiment_id": "wrong"}, "experiment_id"),
        (lambda artifact: artifact | {"task_id": "wrong"}, "task_id"),
        (lambda artifact: artifact | {"milestone": "2026.07.501"}, "milestone"),
        (lambda artifact: artifact | {"status": "honest_null"}, "status"),
        (lambda artifact: artifact | {"search_date": "2026-07-10"}, "search_date"),
        (lambda artifact: artifact | {"semantic_scholar_status": ""}, "semantic_scholar_status"),
        (lambda artifact: artifact | {"sources_checked": ["arxiv"]}, "sources_checked"),
        (
            lambda artifact: artifact
            | {"sources_checked": artifact["sources_checked"] + [artifact["sources_checked"][0]]},
            "duplicate source families",
        ),
        (lambda artifact: artifact | {"searched_source_details": []}, "searched_source_details"),
        (
            lambda artifact: artifact
            | {"searched_source_details": artifact["searched_source_details"] | {"github": {"status": "skipped"}}},
            "valid status",
        ),
        (
            lambda artifact: artifact
            | {"searched_source_details": {k: v for k, v in artifact["searched_source_details"].items() if k != "github"}},
            "missing github",
        ),
        (lambda artifact: artifact | {"new_actionable_findings_count": 99}, "references count"),
        (
            lambda artifact: artifact | {"new_references_added": {"not": "a list"}},
            "new_references_added must be a list",
        ),
        (
            lambda artifact: artifact | {"new_references_added": [42]},
            "entries must be mappings",
        ),
        (
            lambda artifact: artifact
            | {"new_references_added": [{k: v for k, v in artifact["new_references_added"][0].items() if k != "source_type"}]},
            "reference missing required fields",
        ),
        (
            lambda artifact: artifact
            | {"new_references_added": [artifact["new_references_added"][0] | {"planned_experiment": "unplanned"}]},
            "unknown planned_experiment",
        ),
        (
            lambda artifact: artifact
            | {"new_references_added": [artifact["new_references_added"][0] | {"mapped_task": "wrong"}]},
            "mapped_task",
        ),
        (lambda artifact: artifact | {"experiment_mappings": []}, "experiment_mappings"),
        (
            lambda artifact: artifact | {"field_principles": artifact["field_principles"] | {"extra": "why"}},
            "field_principles",
        ),
        (lambda artifact: artifact | {"inference_substrate": "live_llm_inference"}, "inference_substrate"),
        (lambda artifact: artifact | {"closed_scopes_reopened": True}, "closed_scopes_reopened"),
        (lambda artifact: artifact | {"research_references_updated": False}, "research_references_updated"),
        (
            lambda artifact: artifact | {"status": "blocked", "honest_verdict": "blocked: test"},
            "missing prior_refresh_marker_found",
        ),
        (
            lambda artifact: artifact
            | {
                "status": "blocked",
                "prior_refresh_marker_found": False,
                "honest_verdict": "blocked: test",
            },
            "research_references_updated must be false when blocked",
        ),
        (
            lambda artifact: artifact | {"prior_refresh_marker_found": False},
            "prior_refresh_marker_found must be true",
        ),
        (lambda artifact: artifact | {"honest_verdict": "done"}, "honest_verdict"),
        (lambda artifact: artifact | {"no_deep_research_used": False}, "no_deep_research_used"),
        (lambda artifact: artifact | {"research_conductor_modified": True}, "protected files"),
        (lambda artifact: artifact | {"ops_docs_modified": True}, "protected files"),
        (lambda artifact: artifact | {"traceability_modified": True}, "protected files"),
    ],
)
def test_validate_artifact_rejects_invalid_receipts(mutate, message: str) -> None:
    """REQ-REPORT-5537: schema validation rejects unsafe source-delta receipts."""

    artifact = mod.build_artifact()
    invalid = mutate(artifact)
    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(invalid)
