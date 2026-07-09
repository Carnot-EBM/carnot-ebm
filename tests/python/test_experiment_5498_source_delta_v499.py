"""Tests for Exp5498 V499 execution-time source delta refresh.

Spec refs: REQ-REPORT-5498, SCENARIO-REPORT-5498-APPEND-DELTAS,
SCENARIO-REPORT-5498-NO-NEW-DELTA,
SCENARIO-REPORT-5498-BLOCKED-GATE-OR-MARKER.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5498_source_delta_v499 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _write_gate(root: Path, *, resolved: bool = True) -> None:
    path = root / mod.PRETEST_GATE_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "experiment": "experiment_5497_pretest_cascade_diagnostic_v499",
                "pretest_cascade_resolved": resolved,
                "inference_substrate": "aggregation_from_upstream_artifacts",
            }
        ),
        encoding="utf-8",
    )


def _references_text() -> str:
    return "# Research References\n\n## V499 Planner Refresh - 2026-07-09\nExisting V499 section.\n"


def test_req_report_5498_spec_declares_source_delta_contract() -> None:
    """REQ-REPORT-5498: OpenSpec anchors the V499 source-delta receipt."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[
        spec.index("### REQ-REPORT-5498") : spec.index("## Implementation Status (REQ-REPORT-5498)")
    ]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-REPORT-5498-APPEND-DELTAS",
        "SCENARIO-REPORT-5498-NO-NEW-DELTA",
        "SCENARIO-REPORT-5498-BLOCKED-GATE-OR-MARKER",
        str(mod.RESULT_RELATIVE_PATH),
        str(mod.PRETEST_GATE_RELATIVE_PATH),
        "arXiv",
        "OpenReview",
        "HuggingFace Papers",
        "Semantic Scholar-style routes for EBT `2507.02092`",
        "ARM-EBM `2512.15605`",
        "GitHub",
        "Extropic",
        "Logical Intelligence",
        "ops/exclusion_manifest.yaml",
        "scripts/research_conductor.py",
        "ops/changelog.md",
        "ops/status.md",
        "_bmad/traceability.md",
        "closed_scopes_reopened=false",
        "aggregation_from_upstream_artifacts",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert f'principle "{principle}"' in normalized


def test_build_artifact_records_required_fields_and_new_delta() -> None:
    """REQ-REPORT-5498: artifact records source-verified V499 delta."""

    artifact = mod.build_artifact(
        methodology_duration_s=42.5,
        tests_run=["tests/python/test_experiment_5498_source_delta_v499.py"],
    )

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["task_id"] == mod.TASK_ID
    assert artifact["milestone"] == "2026.07.499"
    assert artifact["status"] == "complete"
    assert artifact["new_actionable_findings_count"] == 1
    assert artifact["new_references_added"] == mod.NEW_REFERENCES_ADDED
    assert artifact["closed_scopes_reopened"] is False
    assert artifact["research_references_updated"] is True
    assert artifact["prior_refresh_marker_found"] is True
    assert artifact["pretest_gate_artifact"] == str(mod.PRETEST_GATE_RELATIVE_PATH)
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["methodology_duration_s"] == 42.5
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert set(mod.REQUIRED_SOURCE_FAMILIES).issubset(artifact["sources_checked"])

    details = artifact["searched_source_details"]
    assert details["arxiv"]["status"] == "ok"
    assert details["arxiv_recent_api"]["status"] == "ok"
    assert details["openreview"]["status"] == "ok"
    assert details["huggingface_papers"]["status"] == "partial"
    assert details["github"]["status"] == "partial"
    assert details["logical_intelligence"]["status"] == "ok"
    assert "2607.07026" in details["arxiv_recent_api"]["promoted"][0]

    promoted = artifact["new_references_added"][0]
    assert promoted["title"] == (
        "Constrained Decoding for Diffusion Language Models via Efficient "
        "Inference over Finite Automata"
    )
    assert promoted["url"] == "https://arxiv.org/abs/2607.07026"
    assert "finite-automaton" in promoted["carnot_hook"]
    assert "autoregressive prefix masks" in promoted["carnot_hook"]

    duplicate_text = " ".join(artifact["duplicates_suppressed"])
    for duplicate in (
        "Trajel",
        "RT4CHART",
        "ExpGraph",
        "Evo-Memory",
        "VeryTrace",
        "EBT",
        "ARM-EBM",
    ):
        assert duplicate in duplicate_text

    watch_text = " ".join(row["reason"] for row in artifact["watch_only_or_excluded"])
    assert "policy-gradient" in watch_text
    assert "non-local" in watch_text
    assert "hardware speedup" in watch_text


def test_scenario_report_5498_appends_refresh_section_idempotently() -> None:
    """SCENARIO-REPORT-5498-APPEND-DELTAS: references append once."""

    artifact = mod.build_artifact()
    updated = mod.append_refresh_section(_references_text(), artifact)
    second = mod.append_refresh_section(updated, artifact)

    assert updated == second
    assert updated.count(mod.REFRESH_HEADING) == 1
    assert "Constrained Decoding for Diffusion Language Models" in updated
    assert "No active `.499` roadmap edit is required" in updated
    assert "No closed scope was reopened" in updated
    assert mod.REFRESH_END_MARKER in updated


def test_scenario_report_5498_no_new_delta_leaves_references_unchanged() -> None:
    """SCENARIO-REPORT-5498-NO-NEW-DELTA: zero-new refresh does not churn refs."""

    artifact = mod.build_artifact(new_references_added=[])

    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["new_references_added"] == []
    assert artifact["research_references_updated"] is False
    assert artifact["closed_scopes_reopened"] is False
    assert "no new actionable" in artifact["honest_verdict"]
    assert mod.render_refresh_section(artifact) == ""
    assert mod.append_refresh_section(_references_text(), artifact) == _references_text()


def test_scenario_report_5498_blocks_without_gate_or_marker(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5498-BLOCKED-GATE-OR-MARKER: fails closed."""

    artifact = mod.build_artifact(pretest_gate_resolved=False)
    original = "# Research References\n\n## V498 Planner Refresh - 2026-07-09\nOld section.\n"

    mod.validate_artifact(artifact)
    assert artifact["status"] == "blocked"
    assert artifact["pretest_gate_resolved"] is False
    assert artifact["research_references_updated"] is False
    assert artifact["new_references_added"] == []
    assert artifact["honest_verdict"].startswith("blocked:")
    assert mod.append_refresh_section(original, artifact) == original

    references = tmp_path / mod.REFERENCES_RELATIVE_PATH
    result = tmp_path / mod.RESULT_RELATIVE_PATH
    references.write_text(original, encoding="utf-8")
    _write_gate(tmp_path, resolved=True)

    written = mod.write_outputs(root=tmp_path, references_path=references, result_path=result)
    assert written["status"] == "blocked"
    assert written["prior_refresh_marker_found"] is False
    assert references.read_text(encoding="utf-8") == original


def test_blocked_artifact_consistency_checks_fail_closed() -> None:
    """REQ-REPORT-5498: blocked artifacts cannot carry contradictory gate facts."""

    artifact = mod.build_artifact(pretest_gate_resolved=False)

    with pytest.raises(ValueError, match="missing marker or unresolved pretest gate"):
        mod.validate_artifact(
            artifact | {"prior_refresh_marker_found": True, "pretest_gate_resolved": True}
        )

    with pytest.raises(ValueError, match="research_references_updated"):
        mod.validate_artifact(
            artifact
            | {
                "new_actionable_findings_count": 1,
                "new_references_added": [mod.NEW_REFERENCES_ADDED[0]],
            }
        )


def test_write_outputs_writes_artifact_and_reference_append(tmp_path: Path) -> None:
    """REQ-REPORT-5498: writer emits the JSON result and one references append."""

    references = tmp_path / mod.REFERENCES_RELATIVE_PATH
    result = tmp_path / mod.RESULT_RELATIVE_PATH
    references.write_text(_references_text(), encoding="utf-8")
    _write_gate(tmp_path, resolved=True)

    artifact = mod.write_outputs(
        root=tmp_path,
        references_path=references,
        result_path=result,
        methodology_duration_s=9.25,
        tests_run=["tests/python/test_experiment_5498_source_delta_v499.py"],
    )

    mod.validate_artifact(artifact)
    assert result.exists()
    assert json.loads(result.read_text(encoding="utf-8")) == artifact
    assert references.read_text(encoding="utf-8").count(mod.REFRESH_HEADING) == 1


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda artifact: {k: v for k, v in artifact.items() if k != "sources_checked"},
            "missing required",
        ),
        (lambda artifact: artifact | {"experiment_id": "wrong"}, "experiment_id"),
        (lambda artifact: artifact | {"task_id": "wrong"}, "task_id"),
        (lambda artifact: artifact | {"milestone": "2026.07.498"}, "milestone"),
        (lambda artifact: artifact | {"status": "honest_null"}, "status"),
        (lambda artifact: artifact | {"search_date": "2026-07-09"}, "search_date"),
        (lambda artifact: artifact | {"sources_checked": ["arxiv"]}, "sources_checked"),
        (
            lambda artifact: (
                artifact
                | {
                    "sources_checked": artifact["sources_checked"]
                    + [artifact["sources_checked"][0]]
                }
            ),
            "duplicate source families",
        ),
        (lambda artifact: artifact | {"searched_source_details": []}, "searched_source_details"),
        (
            lambda artifact: (
                artifact
                | {
                    "searched_source_details": artifact["searched_source_details"]
                    | {"github": {"status": "skipped"}}
                }
            ),
            "valid status",
        ),
        (lambda artifact: artifact | {"new_actionable_findings_count": 99}, "references count"),
        (
            lambda artifact: artifact | {"new_references_added": {"not": "a list"}},
            "new_references_added must be a list",
        ),
        (
            lambda artifact: artifact | {"new_references_added": [{"title": "missing fields"}]},
            "new_references_added rows",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "new_references_added": [
                        artifact["new_references_added"][0] | {"url": "arxiv:bad"}
                    ]
                }
            ),
            "verified URL",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "new_references_added": [
                        artifact["new_references_added"][0] | {"carnot_hook": ""}
                    ]
                }
            ),
            "Carnot hook",
        ),
        (lambda artifact: artifact | {"duplicates_suppressed": []}, "duplicates_suppressed"),
        (
            lambda artifact: (
                artifact
                | {
                    "duplicates_suppressed": artifact["duplicates_suppressed"]
                    + [artifact["duplicates_suppressed"][0]]
                }
            ),
            "duplicate suppressed entries",
        ),
        (lambda artifact: artifact | {"closed_scopes_reopened": True}, "closed_scopes_reopened"),
        (
            lambda artifact: artifact | {"research_references_updated": False},
            "research_references_updated",
        ),
        (
            lambda artifact: artifact | {"prior_refresh_marker_found": False},
            "prior_refresh_marker_found",
        ),
        (
            lambda artifact: artifact | {"pretest_gate_artifact": "wrong.json"},
            "pretest_gate_artifact",
        ),
        (lambda artifact: artifact | {"pretest_gate_resolved": False}, "pretest gate"),
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
    """REQ-REPORT-5498: schema drift fails closed before artifact write."""

    artifact = mod.build_artifact()
    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(mutate(artifact))


def test_deliverable_json_matches_required_schema() -> None:
    """REQ-REPORT-5498: committed deliverable uses the tested schema."""

    payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    mod.validate_artifact(payload)
    assert payload["experiment_id"] == mod.EXPERIMENT_ID
    assert payload["status"] == "complete"
    assert payload["milestone"] == "2026.07.499"
    assert payload["prior_refresh_marker_found"] is True
    assert payload["pretest_gate_artifact"] == str(mod.PRETEST_GATE_RELATIVE_PATH)
    assert payload["inference_substrate"] == "aggregation_from_upstream_artifacts"
