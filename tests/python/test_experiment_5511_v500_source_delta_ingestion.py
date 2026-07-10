"""Tests for Exp5511 V500 execution-time source delta ingestion.

Spec refs: REQ-REPORT-5511, SCENARIO-REPORT-5511-APPEND-DELTAS,
SCENARIO-REPORT-5511-NO-NEW-DELTA, SCENARIO-REPORT-5511-BLOCKED-MARKER.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5511_v500_source_delta_ingestion as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _references_text() -> str:
    return (
        "# Research References\n\n"
        "## V500 Planner Refresh - 2026-07-09\n"
        "Existing V500 section.\n"
    )


def _accepted_finding() -> dict[str, object]:
    return {
        "title": "Example Exact Structured Verifier",
        "url": "https://arxiv.org/abs/2699.00001",
        "source_type": "arXiv preprint",
        "carnot_hook": "Use as a structured SOTA control fixture with exact validators.",
        "planned_experiment": "structured_sota_control",
        "mapped_task": "exp5512-structured-output-positive-control",
    }


def test_req_report_5511_spec_declares_v500_source_delta_contract() -> None:
    """REQ-REPORT-5511: OpenSpec anchors the V500 source-delta receipt."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-5511") :]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-REPORT-5511-APPEND-DELTAS",
        "SCENARIO-REPORT-5511-NO-NEW-DELTA",
        "SCENARIO-REPORT-5511-BLOCKED-MARKER",
        str(mod.RESULT_RELATIVE_PATH),
        "V500 Execution Refresh - 20260710",
        "OpenReview",
        "Semantic Scholar-style routes for EBT `2507.02092`",
        "ARM-EBM `2512.15605`",
        "HuggingFace Papers",
        "GitHub",
        "Logical Intelligence",
        "closed_scopes_reopened=false",
        "aggregation_from_upstream_artifacts",
        "scripts/research_conductor.py",
        "ops/changelog.md",
        "ops/status.md",
        "_bmad/traceability.md",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert f'principle "{principle}"' in normalized


def test_scenario_report_5511_no_new_delta_records_noop_receipt() -> None:
    """SCENARIO-REPORT-5511-NO-NEW-DELTA: duplicate-only sweep is stable."""

    artifact = mod.build_artifact(
        methodology_duration_s=12.5,
        tests_run=["tests/python/test_experiment_5511_v500_source_delta_ingestion.py"],
    )

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["task_id"] == mod.TASK_ID
    assert artifact["milestone"] == "2026.07.500"
    assert artifact["status"] == "complete"
    assert artifact["new_references_added"] == []
    assert artifact["new_actionable_findings_count"] == 0
    assert artifact["experiment_mappings"] == []
    assert artifact["research_references_updated"] is False
    assert artifact["prior_refresh_marker_found"] is True
    assert artifact["closed_scopes_reopened"] is False
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["methodology_duration_s"] == 12.5
    assert set(mod.REQUIRED_SOURCE_FAMILIES).issubset(artifact["sources_checked"])
    assert artifact["searched_source_details"]["arxiv_recent_api"]["strict_post_marker_hits"] == 0
    assert "no new actionable" in artifact["honest_verdict"]
    assert mod.render_refresh_section(artifact) == ""
    assert mod.append_refresh_section(_references_text(), artifact) == _references_text()

    duplicate_text = " ".join(artifact["duplicates_suppressed"])
    for duplicate in (
        "Distributional EBMs",
        "Constrained Decoding for Diffusion LMs",
        "Budget-Curated Memory",
        "Probabilistic Memory",
        "EBT",
        "ARM-EBM",
    ):
        assert duplicate in duplicate_text
    watch_text = " ".join(row["reason"] for row in artifact["watch_only_or_excluded"])
    assert "non-local" in watch_text
    assert "no local executable path" in watch_text
    assert "policy-gradient" in watch_text


def test_scenario_report_5511_appends_and_maps_accepted_findings() -> None:
    """SCENARIO-REPORT-5511-APPEND-DELTAS: accepted findings map to .500 lanes."""

    artifact = mod.build_artifact(new_references_added=[_accepted_finding()])
    updated = mod.append_refresh_section(_references_text(), artifact)
    second = mod.append_refresh_section(updated, artifact)

    mod.validate_artifact(artifact)
    assert artifact["research_references_updated"] is True
    assert artifact["new_actionable_findings_count"] == 1
    assert artifact["experiment_mappings"] == [
        {
            "title": "Example Exact Structured Verifier",
            "planned_experiment": "structured_sota_control",
            "mapped_task": "exp5512-structured-output-positive-control",
            "rationale": "Use as a structured SOTA control fixture with exact validators.",
        }
    ]
    assert updated == second
    assert updated.count(mod.REFRESH_HEADING) == 1
    assert "Example Exact Structured Verifier" in updated
    assert "exp5512-structured-output-positive-control" in updated
    assert "No closed scope was reopened" in updated
    assert mod.REFRESH_END_MARKER in updated


def test_scenario_report_5511_blocks_without_prior_marker(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5511-BLOCKED-MARKER: missing V500 marker fails closed."""

    artifact = mod.build_artifact(prior_refresh_marker_found=False)
    original = "# Research References\n\n## V499 Planner Refresh - 2026-07-09\nOld section.\n"

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


def test_write_outputs_writes_noop_artifact_without_reference_churn(tmp_path: Path) -> None:
    """REQ-REPORT-5511: writer emits JSON and preserves references on no delta."""

    references = tmp_path / mod.REFERENCES_RELATIVE_PATH
    result = tmp_path / mod.RESULT_RELATIVE_PATH
    original = _references_text()
    references.write_text(original, encoding="utf-8")

    artifact = mod.write_outputs(
        root=tmp_path,
        references_path=references,
        result_path=result,
        methodology_duration_s=8.75,
        tests_run=["tests/python/test_experiment_5511_v500_source_delta_ingestion.py"],
    )

    mod.validate_artifact(artifact)
    assert result.exists()
    assert json.loads(result.read_text(encoding="utf-8")) == artifact
    assert references.read_text(encoding="utf-8") == original


def test_write_outputs_persists_future_append_when_renderer_changes_refs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-5511-APPEND-DELTAS: writer persists a non-empty append."""

    references = tmp_path / mod.REFERENCES_RELATIVE_PATH
    result = tmp_path / mod.RESULT_RELATIVE_PATH
    original = _references_text()
    references.write_text(original, encoding="utf-8")

    def _fake_append(references_text: str, artifact: dict[str, object]) -> str:
        mod.validate_artifact(artifact)
        return references_text + "\n\nfuture append\n"

    monkeypatch.setattr(mod, "append_refresh_section", _fake_append)

    artifact = mod.write_outputs(
        root=tmp_path,
        references_path=references,
        result_path=result,
        tests_run=["tests/python/test_experiment_5511_v500_source_delta_ingestion.py"],
    )

    mod.validate_artifact(artifact)
    assert references.read_text(encoding="utf-8").endswith("\n\nfuture append\n")


def test_validate_artifact_rejects_blocked_status_with_present_marker() -> None:
    """SCENARIO-REPORT-5511-BLOCKED-MARKER: blocked status needs missing marker."""

    artifact = mod.build_artifact() | {
        "status": "blocked",
        "honest_verdict": "blocked: impossible blocked state.",
    }
    with pytest.raises(ValueError, match="blocked artifacts"):
        mod.validate_artifact(artifact)


def test_validate_artifact_rejects_blocked_artifact_with_reference_payload() -> None:
    """SCENARIO-REPORT-5511-BLOCKED-MARKER: blocked artifacts stay empty."""

    finding = _accepted_finding()
    artifact = mod.build_artifact(prior_refresh_marker_found=False) | {
        "new_actionable_findings_count": 1,
        "new_references_added": [finding],
        "experiment_mappings": [
            {
                "title": finding["title"],
                "planned_experiment": finding["planned_experiment"],
                "mapped_task": finding["mapped_task"],
                "rationale": finding["carnot_hook"],
            }
        ],
    }
    with pytest.raises(ValueError, match="research_references_updated"):
        mod.validate_artifact(artifact)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda artifact: {k: v for k, v in artifact.items() if k != "sources_checked"},
            "missing required",
        ),
        (lambda artifact: artifact | {"experiment_id": "wrong"}, "experiment_id"),
        (lambda artifact: artifact | {"task_id": "wrong"}, "task_id"),
        (lambda artifact: artifact | {"milestone": "2026.07.499"}, "milestone"),
        (lambda artifact: artifact | {"status": "honest_null"}, "status"),
        (lambda artifact: artifact | {"search_date": "2026-07-10"}, "search_date"),
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
            lambda artifact: artifact | {"new_actionable_findings_count": 99},
            "references count",
        ),
        (
            lambda artifact: artifact | {"new_references_added": {"not": "a list"}},
            "new_references_added must be a list",
        ),
        (
            lambda artifact: artifact | {"new_references_added": [{"title": "missing fields"}]},
            "new_references_added rows",
        ),
        (
            lambda artifact: artifact
            | {"new_references_added": [_accepted_finding() | {"url": "arxiv:bad"}]},
            "verified URL",
        ),
        (
            lambda artifact: artifact
            | {"new_references_added": [_accepted_finding() | {"carnot_hook": ""}]},
            "Carnot hook",
        ),
        (
            lambda artifact: artifact
            | {"new_references_added": [_accepted_finding() | {"planned_experiment": "bad_lane"}]},
            "planned experiment lane",
        ),
        (
            lambda artifact: artifact
            | {"new_references_added": [_accepted_finding() | {"mapped_task": "wrong"}]},
            "mapped_task",
        ),
        (lambda artifact: artifact | {"experiment_mappings": [{"title": "extra"}]}, "mappings"),
        (lambda artifact: artifact | {"duplicates_suppressed": []}, "duplicates_suppressed"),
        (
            lambda artifact: artifact
            | {"duplicates_suppressed": artifact["duplicates_suppressed"] + [artifact["duplicates_suppressed"][0]]},
            "duplicate suppressed entries",
        ),
        (lambda artifact: artifact | {"closed_scopes_reopened": True}, "closed_scopes_reopened"),
        (lambda artifact: artifact | {"research_references_updated": True}, "research_references_updated"),
        (lambda artifact: artifact | {"prior_refresh_marker_found": False}, "prior_refresh_marker_found"),
        (lambda artifact: artifact | {"honest_verdict": "done"}, "honest_verdict"),
        (
            lambda artifact: artifact | {"inference_substrate": "literature_ingestion"},
            "inference_substrate",
        ),
        (lambda artifact: artifact | {"watch_only_or_excluded": []}, "watch_only_or_excluded"),
        (lambda artifact: artifact | {"methodology_duration_s": -1.0}, "methodology_duration_s"),
        (lambda artifact: artifact | {"field_principles": {"status": "wrong"}}, "field_principles"),
        (lambda artifact: artifact | {"no_deep_research_used": False}, "deep-research"),
        (lambda artifact: artifact | {"research_conductor_modified": True}, "research_conductor"),
        (lambda artifact: artifact | {"ops_docs_modified": True}, "ops docs"),
        (lambda artifact: artifact | {"traceability_modified": True}, "traceability"),
        (lambda artifact: artifact | {"roadmap_files_modified": True}, "roadmap files"),
        (lambda artifact: artifact | {"tests_run": []}, "tests_run"),
    ],
)
def test_validate_artifact_rejects_contract_drift(mutate, message: str) -> None:
    """REQ-REPORT-5511: schema drift fails closed before artifact write."""

    artifact = mod.build_artifact()
    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(mutate(artifact))


def test_deliverable_json_matches_required_schema() -> None:
    """REQ-REPORT-5511: committed deliverable uses the tested schema."""

    payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    mod.validate_artifact(payload)
    assert payload["experiment_id"] == mod.EXPERIMENT_ID
    assert payload["status"] == "complete"
    assert payload["research_references_updated"] is False
    assert payload["new_references_added"] == []
    assert payload["experiment_mappings"] == []
    assert payload["prior_refresh_marker_found"] is True
    assert payload["inference_substrate"] == "aggregation_from_upstream_artifacts"
