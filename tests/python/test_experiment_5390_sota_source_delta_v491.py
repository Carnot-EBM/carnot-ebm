"""Tests for Exp 5390 V491 execution-time source delta refresh.

Spec refs: REQ-REPORT-5390, SCENARIO-REPORT-5390-APPEND-DELTAS,
SCENARIO-REPORT-5390-NO-NEW-DELTA.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5390_sota_source_delta_v491 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_report_5390_spec_declares_v491_source_delta_contract() -> None:
    """REQ-REPORT-5390: OpenSpec anchors the V491 source-delta refresh."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-5390") : spec.index("### REQ-REPORT-5336")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-REPORT-5390",
        "SCENARIO-REPORT-5390-APPEND-DELTAS",
        "SCENARIO-REPORT-5390-NO-NEW-DELTA",
        str(mod.RESULT_RELATIVE_PATH),
        "HuggingFace Papers",
        "Semantic Scholar routes for EBT `2507.02092`",
        "ARM-EBM `2512.15605`",
        "Extropic",
        "Logical Intelligence",
        "V489/V490/V491 duplicate history",
        "non-local TSU/Kona/Aleph claims",
        "external generated-text scorers",
        "token/internal-feature claims without backend evidence",
        "scripts/research_conductor.py",
        "ops/changelog.md",
        "ops/status.md",
        "_bmad/traceability.md",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_build_artifact_records_required_fields_and_v491_deltas() -> None:
    """REQ-REPORT-5390: artifact records source-verified actionable deltas."""

    artifact = mod.build_artifact(
        methodology_duration_s=711.125,
        tests_run=["tests/python/test_experiment_5390_sota_source_delta_v491.py"],
    )

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["task_id"] == mod.TASK_ID
    assert artifact["milestone"] == "2026.07.491"
    assert artifact["status"] == "complete"
    assert artifact["new_actionable_findings_count"] == len(mod.ACTIONABLE_FINDINGS)
    assert artifact["appended_references_block"] is True
    assert artifact["appended_references_anchor"] == mod.REFRESH_HEADING
    assert artifact["retired_scopes_reopened"] is False
    assert artifact["methodology_duration_s"] == 711.125
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES

    assert set(mod.REQUIRED_SOURCE_FAMILIES).issubset(artifact["sources_checked"])
    details = artifact["searched_source_details"]
    assert details["arxiv"]["status"] == "ok"
    assert details["semantic_scholar"]["status"] == "rate_limited"
    assert details["github"]["status"] == "ok"
    assert details["extropic_writing"]["status"] == "ok"
    assert details["logical_intelligence"]["status"] == "ok"
    assert "KANDy" in " ".join(artifact["duplicates_suppressed"])
    assert "VAGEN" in " ".join(artifact["duplicates_suppressed"])

    assert [row["title"] for row in artifact["findings"]] == [
        "AgentLTL: A Trace-Verification Framework for Procedural Compliance",
        "OEP: Poisoning Self-Evolving LLM Agents via Locally Correct Experiences",
        "CoACT: Action-Preserving Observation Compression for Coding Agents",
        "Succinct QUBO formulations for permutation problems by sorting networks",
    ]
    assert artifact["findings"][0]["url"] == "https://arxiv.org/abs/2607.02599"
    assert "deterministic, judge-free compliance score" in artifact["findings"][0]["carnot_hook"]
    assert artifact["findings"][1]["url"] == "https://arxiv.org/abs/2605.18930"
    assert "non-transferable" in artifact["findings"][1]["carnot_hook"]
    assert artifact["findings"][2]["url"] == "https://arxiv.org/abs/2607.02911"
    assert "next-action preservation" in artifact["findings"][2]["carnot_hook"]
    assert artifact["findings"][3]["url"] == "https://arxiv.org/abs/2603.07579"
    assert "sorting-network QUBO" in artifact["findings"][3]["carnot_hook"]


def test_scenario_report_5390_appends_refresh_section_idempotently() -> None:
    """SCENARIO-REPORT-5390-APPEND-DELTAS: references append once."""

    artifact = mod.build_artifact()
    original = "# Research References\n\n### V491 Planner Refresh - 2026-07-08\nOld V491 section.\n"

    updated = mod.append_refresh_section(original, artifact)
    second = mod.append_refresh_section(updated, artifact)

    assert updated == second
    assert updated.startswith(original)
    assert updated.count(mod.REFRESH_HEADING) == 1
    assert "AgentLTL" in updated
    assert "OEP: Poisoning Self-Evolving" in updated
    assert "CoACT: Action-Preserving" in updated
    assert "Succinct QUBO formulations" in updated
    assert "No active `.491` roadmap edit is required" in updated
    assert mod.REFRESH_END_MARKER in updated


def test_scenario_report_5390_no_new_delta_leaves_references_unchanged() -> None:
    """SCENARIO-REPORT-5390-NO-NEW-DELTA: zero-new refresh does not churn references."""

    artifact = mod.build_artifact(actionable_findings=[])
    original = "# Research References\n\n### V491 Planner Refresh - 2026-07-08\nOld V491 section.\n"

    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["new_actionable_findings_count"] == 0
    assert artifact["appended_references_block"] is False
    assert artifact["appended_references_anchor"] is None
    assert artifact["findings"] == []
    assert "no new actionable" in artifact["honest_verdict"]
    assert mod.append_refresh_section(original, artifact) == original


def test_write_outputs_writes_artifact_and_reference_append(tmp_path: Path) -> None:
    """REQ-REPORT-5390: writer emits the JSON result and one references append."""

    references = tmp_path / mod.REFERENCES_RELATIVE_PATH
    result = tmp_path / mod.RESULT_RELATIVE_PATH
    references.write_text(
        "# Research References\n\n### V491 Planner Refresh - 2026-07-08\nOld V491 section.\n",
        encoding="utf-8",
    )

    artifact = mod.write_outputs(
        root=tmp_path,
        references_path=references,
        result_path=result,
        methodology_duration_s=9.25,
        tests_run=["tests/python/test_experiment_5390_sota_source_delta_v491.py"],
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
        (lambda artifact: artifact | {"milestone": "2026.07.490"}, "milestone"),
        (lambda artifact: artifact | {"status": "honest_null"}, "status"),
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
        (lambda artifact: artifact | {"new_actionable_findings_count": 99}, "findings count"),
        (lambda artifact: artifact | {"findings": {"not": "a list"}}, "findings must be a list"),
        (
            lambda artifact: artifact | {"findings": [{"title": "missing fields"}]},
            "findings rows must include",
        ),
        (
            lambda artifact: (
                artifact | {"findings": [artifact["findings"][0] | {"url": "arxiv:bad"}]}
            ),
            "verified URL",
        ),
        (
            lambda artifact: (
                artifact | {"findings": [artifact["findings"][0] | {"carnot_hook": ""}]}
            ),
            "Carnot hook",
        ),
        (lambda artifact: artifact | {"duplicates_suppressed": []}, "duplicates_suppressed"),
        (lambda artifact: artifact | {"retired_scopes_reopened": True}, "retired_scopes_reopened"),
        (
            lambda artifact: artifact | {"appended_references_block": False},
            "appended_references_block",
        ),
        (
            lambda artifact: artifact | {"appended_references_anchor": "wrong"},
            "appended_references_anchor",
        ),
        (lambda artifact: artifact | {"honest_verdict": "done"}, "honest_verdict"),
        (lambda artifact: artifact | {"inference_substrate": "wrong"}, "inference_substrate"),
        (
            lambda artifact: artifact | {"local_execution_implications": []},
            "local_execution_implications",
        ),
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
    """REQ-REPORT-5390: schema drift fails closed before artifact write."""

    artifact = mod.build_artifact()
    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(mutate(artifact))


def test_deliverable_json_matches_required_schema() -> None:
    """REQ-REPORT-5390: committed deliverable uses the tested schema."""

    payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    mod.validate_artifact(payload)
    assert payload["experiment_id"] == mod.EXPERIMENT_ID
    assert payload["status"] == "complete"
    assert payload["milestone"] == "2026.07.491"
