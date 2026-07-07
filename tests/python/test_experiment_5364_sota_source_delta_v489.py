"""Tests for Exp 5364 V489 execution-time source delta refresh.

Spec refs: REQ-REPORT-5364, SCENARIO-REPORT-5364-APPEND-DELTAS,
SCENARIO-REPORT-5364-HONEST-NULL.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5364_sota_source_delta_v489 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_report_5364_spec_declares_v489_source_delta_contract() -> None:
    """REQ-REPORT-5364: OpenSpec anchors the V489 source delta refresh."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-5364") : spec.index("### REQ-REPORT-5336")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-REPORT-5364",
        "SCENARIO-REPORT-5364-APPEND-DELTAS",
        "SCENARIO-REPORT-5364-HONEST-NULL",
        str(mod.RESULT_RELATIVE_PATH),
        "HuggingFace Papers",
        "Extropic",
        "Logical Intelligence",
        "retired ARC candidate-exploration-signal reruns",
        "scripts/research_conductor.py",
        "ops/changelog.md",
        "ops/status.md",
        "_bmad/traceability.md",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_build_artifact_records_required_fields_and_v489_deltas() -> None:
    """REQ-REPORT-5364: artifact records source-verified actionable deltas."""

    artifact = mod.build_artifact(
        methodology_duration_s=612.5,
        tests_run=["tests/python/test_experiment_5364_sota_source_delta_v489.py"],
    )

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["milestone"] == mod.MILESTONE
    assert artifact["status"] == "complete"
    assert artifact["search_date"] == "20260707"
    assert artifact["new_actionable_findings_count"] == len(mod.ACTIONABLE_FINDINGS)
    assert artifact["research_references_updated"] is True
    assert artifact["retired_scope_reopened"] is False
    assert artifact["methodology_duration_s"] == 612.5
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES

    assert set(mod.REQUIRED_SOURCE_FAMILIES).issubset(artifact["sources_checked"])
    details = artifact["searched_source_details"]
    assert details["arxiv"]["status"] == "ok"
    assert details["openreview"]["status"] == "challenge_blocked"
    assert details["semantic_scholar"]["status"] in {"ok", "rate_limited"}
    assert details["github"]["status"] == "ok"
    assert "G-RRM" in " ".join(artifact["duplicates_suppressed"])

    findings = artifact["findings"]
    assert [row["title"] for row in findings] == [
        "LLGuidance: Low-level Guidance for Super-fast Structured Outputs",
        "LongMemEval-V2: Evaluating Long-Term Agent Memory Toward Experienced Colleagues",
    ]
    assert findings[0] == {
        "title": "LLGuidance: Low-level Guidance for Super-fast Structured Outputs",
        "url": "https://github.com/guidance-ai/llguidance",
        "source_type": "GitHub implementation",
        "carnot_hook": (
            "Use the llama.cpp llguidance integration as the concrete grammar-budget "
            "probe for Exp5365: record whether the local build has "
            "-DLLAMA_LLGUIDANCE=ON, compile JSON/Lark reachability fixtures, and "
            "measure mask-computation budget before Exp5366 live GGUF generation."
        ),
    }
    assert findings[1]["url"] == "https://arxiv.org/abs/2605.12493"


def test_scenario_report_5364_appends_refresh_section_idempotently() -> None:
    """SCENARIO-REPORT-5364-APPEND-DELTAS: references append once."""

    artifact = mod.build_artifact()
    original = "# Research References\n\n### V489 Planner Refresh - 2026-07-07\nOld V489 section.\n"

    updated = mod.append_refresh_section(original, artifact)
    second = mod.append_refresh_section(updated, artifact)

    assert updated == second
    assert updated.startswith(original)
    assert updated.count(mod.REFRESH_HEADING) == 1
    assert "LLGuidance" in updated
    assert "LongMemEval-V2" in updated
    assert "No active `.489` roadmap edit is required" in updated
    assert mod.REFRESH_END_MARKER in updated


def test_scenario_report_5364_honest_null_leaves_references_unchanged() -> None:
    """SCENARIO-REPORT-5364-HONEST-NULL: zero-new refresh does not churn references."""

    artifact = mod.build_artifact(actionable_findings=[])
    original = "# Research References\n\n### V489 Planner Refresh - 2026-07-07\nOld V489 section.\n"

    mod.validate_artifact(artifact)
    assert artifact["status"] == "honest_null"
    assert artifact["new_actionable_findings_count"] == 0
    assert artifact["research_references_updated"] is False
    assert artifact["findings"] == []
    assert "no new actionable" in artifact["honest_verdict"]
    assert mod.append_refresh_section(original, artifact) == original


def test_write_outputs_writes_artifact_and_reference_append(tmp_path: Path) -> None:
    """REQ-REPORT-5364: writer emits the JSON result and one references append."""

    references = tmp_path / mod.REFERENCES_RELATIVE_PATH
    result = tmp_path / mod.RESULT_RELATIVE_PATH
    references.write_text(
        "# Research References\n\n### V489 Planner Refresh - 2026-07-07\nOld V489 section.\n",
        encoding="utf-8",
    )

    artifact = mod.write_outputs(
        root=tmp_path,
        references_path=references,
        result_path=result,
        methodology_duration_s=9.25,
        tests_run=["tests/python/test_experiment_5364_sota_source_delta_v489.py"],
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
        (lambda artifact: {k: v for k, v in artifact.items() if k != "sources_checked"}, "missing required"),
        (lambda artifact: artifact | {"experiment_id": "wrong"}, "experiment_id"),
        (lambda artifact: artifact | {"milestone": "2026.07.488"}, "milestone"),
        (lambda artifact: artifact | {"status": "blocked"}, "status"),
        (lambda artifact: artifact | {"search_date": "20260706"}, "search_date"),
        (lambda artifact: artifact | {"sources_checked": ["arxiv"]}, "sources_checked"),
        (
            lambda artifact: artifact
            | {"sources_checked": artifact["sources_checked"] + [artifact["sources_checked"][0]]},
            "duplicate source families",
        ),
        (lambda artifact: artifact | {"searched_source_details": []}, "searched_source_details"),
        (
            lambda artifact: artifact
            | {
                "searched_source_details": artifact["searched_source_details"]
                | {"github": {"status": "skipped"}}
            },
            "valid status",
        ),
        (lambda artifact: artifact | {"new_actionable_findings_count": 99}, "findings count"),
        (lambda artifact: artifact | {"findings": {"not": "a list"}}, "findings must be a list"),
        (
            lambda artifact: artifact | {"findings": [{"title": "missing fields"}]},
            "findings rows must include",
        ),
        (
            lambda artifact: artifact
            | {"findings": [artifact["findings"][0] | {"url": "arxiv:bad"}]},
            "verified URL",
        ),
        (
            lambda artifact: artifact
            | {"findings": [artifact["findings"][0] | {"carnot_hook": ""}]},
            "Carnot hook",
        ),
        (lambda artifact: artifact | {"duplicates_suppressed": []}, "duplicates_suppressed"),
        (lambda artifact: artifact | {"retired_scope_reopened": True}, "retired_scope_reopened"),
        (
            lambda artifact: artifact | {"research_references_updated": False},
            "research_references_updated",
        ),
        (lambda artifact: artifact | {"references_section_marker": "wrong"}, "references_section_marker"),
        (lambda artifact: artifact | {"honest_verdict": "done"}, "honest_verdict"),
        (lambda artifact: artifact | {"inference_substrate": "wrong"}, "inference_substrate"),
        (lambda artifact: artifact | {"methodology_duration_s": -1.0}, "methodology_duration_s"),
        (lambda artifact: artifact | {"field_principles": {"status": "wrong"}}, "field_principles"),
        (lambda artifact: artifact | {"task_id": "wrong"}, "task_id"),
        (lambda artifact: artifact | {"no_deep_research_used": False}, "deep-research"),
        (lambda artifact: artifact | {"research_conductor_modified": True}, "research_conductor"),
        (lambda artifact: artifact | {"ops_docs_modified": True}, "ops docs"),
        (lambda artifact: artifact | {"traceability_modified": True}, "traceability"),
        (lambda artifact: artifact | {"roadmap_files_modified": True}, "roadmap files"),
        (lambda artifact: artifact | {"tests_run": []}, "tests_run"),
    ],
)
def test_validate_artifact_rejects_contract_drift(mutate, message: str) -> None:
    """REQ-REPORT-5364: schema drift fails closed before artifact write."""

    artifact = mod.build_artifact()
    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(mutate(artifact))


def test_deliverable_json_matches_required_schema() -> None:
    """REQ-REPORT-5364: committed deliverable uses the tested schema."""

    payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    mod.validate_artifact(payload)
    assert payload["experiment_id"] == mod.EXPERIMENT_ID
    assert payload["status"] == "complete"
    assert payload["search_date"] == "20260707"
