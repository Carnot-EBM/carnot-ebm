"""Tests for Exp5429 V494 execution-time source delta refresh.

Spec refs: REQ-REPORT-5429, SCENARIO-REPORT-5429-APPEND-DELTAS,
SCENARIO-REPORT-5429-NO-NEW-DELTA,
SCENARIO-REPORT-5429-BLOCKED-MISSING-PLANNER.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5429_source_delta_v494 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_report_5429_spec_declares_v494_source_delta_contract() -> None:
    """REQ-REPORT-5429: OpenSpec anchors the V494 source-delta refresh."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-5429") : spec.index("### REQ-REPORT-5426")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-REPORT-5429",
        "SCENARIO-REPORT-5429-APPEND-DELTAS",
        "SCENARIO-REPORT-5429-NO-NEW-DELTA",
        "SCENARIO-REPORT-5429-BLOCKED-MISSING-PLANNER",
        str(mod.RESULT_RELATIVE_PATH),
        "arXiv",
        "OpenReview",
        "HuggingFace Papers",
        "Semantic Scholar routes for EBT `2507.02092`",
        "ARM-EBM `2512.15605`",
        "GitHub",
        "Extropic",
        "Logical Intelligence",
        "V489/V490/V491/V492/V493/V494 duplicate history",
        "ops/exclusion_manifest.yaml",
        "external generated-text scorer",
        "token/internal-feature-without-backend",
        "broad fine-tuning",
        "duplicate ARC lane",
        "scripts/research_conductor.py",
        "ops/changelog.md",
        "ops/status.md",
        "_bmad/traceability.md",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert f'principle "{principle}"' in normalized


def test_build_artifact_records_required_fields_and_v494_deltas() -> None:
    """REQ-REPORT-5429: artifact records source-verified actionable deltas."""

    artifact = mod.build_artifact(
        methodology_duration_s=73.5,
        tests_run=["tests/python/test_experiment_5429_source_delta_v494.py"],
    )

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["task_id"] == mod.TASK_ID
    assert artifact["milestone"] == "2026.07.494"
    assert artifact["status"] == "complete"
    assert artifact["new_actionable_findings_count"] == len(mod.NEW_REFERENCES_ADDED)
    assert artifact["new_references_added"] == mod.NEW_REFERENCES_ADDED
    assert artifact["retired_scopes_reopened"] is False
    assert artifact["research_references_updated"] is True
    assert artifact["prior_refresh_marker_found"] is True
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["methodology_duration_s"] == 73.5
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES

    assert set(mod.REQUIRED_SOURCE_FAMILIES).issubset(artifact["sources_checked"])
    details = artifact["searched_source_details"]
    assert details["arxiv"]["status"] == "ok"
    assert details["openreview"]["status"] == "partial"
    assert details["huggingface_papers"]["status"] == "partial"
    assert details["semantic_scholar"]["status"] == "partial"
    assert details["github"]["status"] == "ok"
    assert details["extropic_writing"]["status"] == "ok"
    assert details["logical_intelligence"]["status"] == "ok"
    assert "citationCount=26" in details["semantic_scholar"]["result"]
    assert "ARM-EBM" in details["semantic_scholar"]["result"]

    duplicate_text = " ".join(artifact["duplicates_suppressed"])
    for already_covered in (
        "V494 planner sources",
        "A-MEM",
        "CoVe",
        "When Continual Learning Moves to Memory",
        "Energy-Guided Decoding for Object Hallucination",
        "p-Bit-Based Fully-Connected",
        "LoopUS",
        "EBT",
        "ARM-EBM",
    ):
        assert already_covered in duplicate_text

    titles = [row["title"] for row in artifact["new_references_added"]]
    assert titles == [
        "NeuroSCA: Neuro-Symbolic Constraint Abstraction for Smart Contract Hybrid Fuzzing",
        "Sampling for Quality: Training-Free Reward-Guided LLM Decoding via Sequential Monte Carlo",
        "LLMs versus the Halting Problem: Characterizing Program Termination Reasoning",
    ]
    assert artifact["new_references_added"][0]["url"] == "https://arxiv.org/abs/2603.01272"
    assert "missed-constraint reinsertion ledger" in artifact["new_references_added"][0]["carnot_hook"]
    assert artifact["new_references_added"][1]["url"] == "https://arxiv.org/abs/2604.16453"
    assert "prefix-only SMC" in artifact["new_references_added"][1]["carnot_hook"]
    assert artifact["new_references_added"][2]["url"] == "https://arxiv.org/abs/2601.18987"
    assert "witness construction" in artifact["new_references_added"][2]["carnot_hook"]

    watch_text = " ".join(row["reason"] for row in artifact["watch_only_or_excluded"])
    assert "external generated-text" in watch_text
    assert "non-local TSU" in watch_text
    assert "Kona or Aleph" in watch_text
    assert "broad fine-tuning" in watch_text


def test_scenario_report_5429_appends_refresh_section_idempotently() -> None:
    """SCENARIO-REPORT-5429-APPEND-DELTAS: references append once."""

    artifact = mod.build_artifact()
    original = "# Research References\n\n### V494 Planner Refresh - 2026-07-08\nOld V494 section.\n"

    updated = mod.append_refresh_section(original, artifact)
    second = mod.append_refresh_section(updated, artifact)

    assert updated == second
    assert updated.startswith(original)
    assert updated.count(mod.REFRESH_HEADING) == 1
    assert "NeuroSCA" in updated
    assert "Sampling for Quality" in updated
    assert "Halting Problem" in updated
    assert "No active `.494` roadmap edit is required" in updated
    assert "No retired scope was reopened" in updated
    assert mod.REFRESH_END_MARKER in updated


def test_scenario_report_5429_no_new_delta_leaves_references_unchanged() -> None:
    """SCENARIO-REPORT-5429-NO-NEW-DELTA: zero-new refresh does not churn references."""

    artifact = mod.build_artifact(new_references_added=[])
    original = "# Research References\n\n### V494 Planner Refresh - 2026-07-08\nOld V494 section.\n"

    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["new_actionable_findings_count"] == 0
    assert artifact["new_references_added"] == []
    assert artifact["research_references_updated"] is False
    assert "no new actionable" in artifact["honest_verdict"]
    assert mod.render_refresh_section(artifact) == ""
    assert mod.append_refresh_section(original, artifact) == original


def test_scenario_report_5429_missing_planner_marker_blocks() -> None:
    """SCENARIO-REPORT-5429-BLOCKED-MISSING-PLANNER: no marker means no append."""

    artifact = mod.build_artifact(prior_refresh_marker_found=False)
    original = "# Research References\n\n### V493 Execution Refresh - 20260708\nOld section.\n"

    mod.validate_artifact(artifact)
    assert artifact["status"] == "blocked"
    assert artifact["prior_refresh_marker_found"] is False
    assert artifact["research_references_updated"] is False
    assert artifact["new_references_added"] == []
    assert artifact["honest_verdict"].startswith("blocked:")
    assert mod.render_refresh_section(artifact) == ""
    assert mod.append_refresh_section(original, artifact) == original


def test_blocked_artifact_consistency_checks_fail_closed() -> None:
    """REQ-REPORT-5429: blocked artifacts cannot carry contradictory planner facts."""

    artifact = mod.build_artifact(prior_refresh_marker_found=False)

    with pytest.raises(ValueError, match="prior_refresh_marker_found"):
        mod.validate_artifact(artifact | {"prior_refresh_marker_found": True})

    with pytest.raises(ValueError, match="research_references_updated"):
        mod.validate_artifact(
            artifact
            | {
                "new_actionable_findings_count": 1,
                "new_references_added": [mod.NEW_REFERENCES_ADDED[0]],
            }
        )


def test_write_outputs_writes_artifact_and_reference_append(tmp_path: Path) -> None:
    """REQ-REPORT-5429: writer emits the JSON result and one references append."""

    references = tmp_path / mod.REFERENCES_RELATIVE_PATH
    result = tmp_path / mod.RESULT_RELATIVE_PATH
    references.write_text(
        "# Research References\n\n### V494 Planner Refresh - 2026-07-08\nOld V494 section.\n",
        encoding="utf-8",
    )

    artifact = mod.write_outputs(
        root=tmp_path,
        references_path=references,
        result_path=result,
        methodology_duration_s=9.25,
        tests_run=["tests/python/test_experiment_5429_source_delta_v494.py"],
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
            "missing required",
        ),
        (lambda artifact: artifact | {"experiment_id": "wrong"}, "experiment_id"),
        (lambda artifact: artifact | {"task_id": "wrong"}, "task_id"),
        (lambda artifact: artifact | {"milestone": "2026.07.493"}, "milestone"),
        (lambda artifact: artifact | {"status": "honest_null"}, "status"),
        (lambda artifact: artifact | {"search_date": "2026-07-08"}, "search_date"),
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
            "new_references_added rows must include",
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
        (lambda artifact: artifact | {"retired_scopes_reopened": True}, "retired_scopes_reopened"),
        (
            lambda artifact: artifact | {"research_references_updated": False},
            "research_references_updated",
        ),
        (
            lambda artifact: artifact | {"prior_refresh_marker_found": False},
            "prior_refresh_marker_found",
        ),
        (lambda artifact: artifact | {"honest_verdict": "done"}, "honest_verdict"),
        (lambda artifact: artifact | {"inference_substrate": "wrong"}, "inference_substrate"),
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
    """REQ-REPORT-5429: schema drift fails closed before artifact write."""

    artifact = mod.build_artifact()
    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(mutate(artifact))


def test_deliverable_json_matches_required_schema() -> None:
    """REQ-REPORT-5429: committed deliverable uses the tested schema."""

    payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    mod.validate_artifact(payload)
    assert payload["experiment_id"] == mod.EXPERIMENT_ID
    assert payload["status"] == "complete"
    assert payload["milestone"] == "2026.07.494"
    assert payload["prior_refresh_marker_found"] is True
    assert payload["inference_substrate"] == "aggregation_from_upstream_artifacts"
