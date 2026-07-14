"""Tests for Exp5604 V506 execution source-delta ingestion.

Spec refs: REQ-REPORT-5604, SCENARIO-REPORT-5604,
SCENARIO-REPORT-5604-NOOP, SCENARIO-REPORT-5604-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_5604_v506_source_delta_ingestion as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"


def _roadmap() -> str:
    tasks = [
        {
            "id": task_id,
            "milestone": mod.MILESTONE,
            "title": task_id,
            "deliverable": f"results/{task_id}.json",
        }
        for task_id in (
            "exp5603-transition-v506",
            "exp5604-v506-source-delta-ingestion",
            "exp5605-raw-response-evidence-envelope",
            "exp5606-clean-sota-solve-verify-evidence-panel",
            "exp5607-property-template-exact-residual-extension",
            "exp5608-kan-longitudinal-self-learning",
            "exp5609-arc-filter-intermediate-invariance-ab",
            "exp5610-unconditional-live-agent-levelup-attempt",
            "exp5611-cdls-matched-cpu-cuda-benchmark",
            "exp5612-v506-capstone-reconciliation",
        )
    ]
    return yaml.safe_dump({"milestone": mod.MILESTONE, "tasks": tasks}, sort_keys=False)


def _planner_references() -> str:
    return (
        "## V506 Planner Refresh - 20260714\n\n"
        "- **ScientistOne** - arXiv:2605.26340.\n"
        "- **Can Aggregate Invariants Accelerate Continuous Subgraph Matching?** "
        "arXiv:2606.24421.\n"
        "- **Accelerating Discrete Langevin Samplers via Continuous Intermediates** "
        "OpenReview Rgs15piXcl.\n"
        "<!-- V506-PLANNER-REFRESH-20260714-END -->\n"
    )


def _make_repo(
    root: Path,
    references_text: str,
    *,
    complete_text: str = "experiments: []\n",
    proposal_text: str | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for relative in ("AGENTS.md", "CODEX.md", "CLAUDE.md", "research-program.md"):
        (root / relative).write_text("fixture\n", encoding="utf-8")
    (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).write_text(references_text, encoding="utf-8")
    (root / mod.RESEARCH_COMPLETE_RELATIVE_PATH).write_text(complete_text, encoding="utf-8")
    (root / "research-roadmap.yaml").write_text(_roadmap(), encoding="utf-8")
    (root / "openspec/change-proposals").mkdir(parents=True, exist_ok=True)
    (root / mod.VNEXT_RELATIVE_PATH).write_text(
        proposal_text
        or (
            "# Research Roadmap vNEXT\n"
            "**Milestone:** 2026.07.506\n"
            "**Task range:** exp5603-exp5612\n"
            "Exp5608 -- KAN-only longitudinal continuous self-learning.\n"
        ),
        encoding="utf-8",
    )
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / mod.EXCLUSION_MANIFEST_RELATIVE_PATH).write_text(
        "retired: []\nretired_extras: []\n", encoding="utf-8"
    )
    (root / mod.KNOWN_ISSUES_RELATIVE_PATH).write_text(
        "Retired parser, causal-memory, PTRM, SGE, generated-text scoring, "
        "and hardware-board scopes remain closed.\n",
        encoding="utf-8",
    )
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts/research_conductor.py").write_text("# fixture\n", encoding="utf-8")
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / mod.EXP5603_RELATIVE_PATH).write_text(
        json.dumps(
            {
                "experiment_id": "exp5603-transition-v506",
                "current_task_range": "exp5603-exp5612",
                "retired_scopes": [
                    {"key": "causal_memory_pace_policy_chain", "closed": True},
                    {"key": "ptrm_as_generator", "closed": True},
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def test_req_report_5604_spec_declares_v506_source_delta_contract() -> None:
    """REQ-REPORT-5604: OpenSpec anchors V506 source-delta fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5604") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-REPORT-5604",
        "SCENARIO-REPORT-5604",
        "SCENARIO-REPORT-5604-NOOP",
        "SCENARIO-REPORT-5604-FIELD-PRINCIPLES",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "`search_timestamp_utc`",
        "`closed_scopes_reopened`",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_5604_non_duplicate_appends_execution_refresh(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5604: new Exp5608 diagnostic appends one V506 block."""

    root = _make_repo(tmp_path, _planner_references())

    artifact = mod.build_and_write_artifact(
        root=root,
        search_timestamp_utc="2026-07-14T03:29:56Z",
        duration_s=0.25,
    )

    mod.validate_artifact(artifact)
    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")
    result_text = (root / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")

    assert references.count(mod.EXECUTION_REFRESH_HEADING) == 1
    assert "The Equilibrium Is the Initialization" in references
    assert "arXiv:2607.11116" in references
    assert "Exp5608" in references
    assert result_text.endswith("\n")
    assert artifact["planner_marker_found"] is True
    assert artifact["research_references_updated"] is True
    assert artifact["closed_scopes_reopened"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["search_timestamp_utc"] == "2026-07-14T03:29:56Z"
    assert artifact["honest_verdict"].startswith("complete:")
    assert [row["source_id"] for row in artifact["new_references_added"]] == [
        "lazy_identity_deq_2607_11116",
    ]

    mapping = [
        row
        for row in artifact["experiment_mappings"]
        if row["lane"] == "KAN-only longitudinal self-learning"
    ][0]
    assert mapping["source_status"] == "accepted_execution_delta"
    assert mapping["experiment_ids"] == ["exp5608-kan-longitudinal-self-learning"]
    assert mapping["source_ids"] == ["lazy_identity_deq_2607_11116"]

    surfaces = {row["surface"] for row in artifact["sources_checked"]}
    assert {
        "arXiv",
        "OpenReview",
        "Semantic Scholar",
        "Hugging Face Papers",
        "GitHub discovery",
        "Extropic writing",
        "Logical Intelligence public pages",
        "local Carnot ledgers",
    }.issubset(surfaces)


def test_scenario_report_5604_duplicate_in_complete_yaml_is_noop(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5604-NOOP: full-ledger duplicates do not append."""

    root = _make_repo(
        tmp_path,
        _planner_references(),
        complete_text=(
            "experiments:\n"
            "- title: The Equilibrium Is the Initialization\n"
            "  source: arXiv:2607.11116\n"
        ),
    )

    artifact = mod.build_and_write_artifact(
        root=root,
        search_timestamp_utc="2026-07-14T03:29:56Z",
        duration_s=0.25,
    )

    mod.validate_artifact(artifact)
    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")

    assert references == _planner_references()
    assert mod.EXECUTION_REFRESH_HEADING not in references
    assert artifact["research_references_updated"] is False
    assert artifact["new_references_added"] == []
    suppressed_ids = {row["source_id"] for row in artifact["duplicates_suppressed"]}
    assert "lazy_identity_deq_2607_11116" in suppressed_ids
    assert artifact["honest_verdict"].startswith("complete:")


def test_scenario_report_5604_missing_planner_marker_blocks_append(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5604-NOOP: missing marker blocks ledger mutation."""

    references_text = "## Earlier Refresh\n\nKnown references only.\n"
    root = _make_repo(tmp_path, references_text)

    artifact = mod.build_and_write_artifact(
        root=root,
        search_timestamp_utc="2026-07-14T03:29:56Z",
        duration_s=0.25,
    )

    mod.validate_artifact(artifact)
    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")
    assert references == references_text
    assert artifact["planner_marker_found"] is False
    assert artifact["research_references_updated"] is False
    assert artifact["honest_verdict"].startswith("blocked:")


def test_scenario_report_5604_sparse_repo_defaults_to_blocked_noop(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5604-NOOP: absent ledgers and default UTC stay terminal."""

    artifact = mod.build_artifact(root=tmp_path, duration_s=0.25)

    mod.validate_artifact(artifact)
    assert artifact["planner_marker_found"] is False
    assert artifact["research_references_updated"] is False
    assert artifact["new_references_added"] == []
    assert artifact["roadmap_context"] == {
        "source": "research-roadmap.yaml",
        "milestone": "",
        "task_ids": [],
    }
    assert artifact["search_timestamp_utc"].endswith("Z")
    assert artifact["honest_verdict"].startswith("blocked:")


def test_scenario_report_5604_existing_execution_block_is_stable(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5604: existing execution block is not appended twice."""

    existing = (
        _planner_references()
        + "\n"
        + mod.render_execution_refresh_block(mod.CANDIDATE_FINDINGS, run_date="20260714")
    )
    root = _make_repo(tmp_path, existing)

    artifact = mod.build_and_write_artifact(
        root=root,
        search_timestamp_utc="2026-07-14T03:29:56Z",
        duration_s=0.25,
    )

    mod.validate_artifact(artifact)
    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")
    assert references == existing
    assert references.count(mod.EXECUTION_REFRESH_HEADING) == 1
    assert artifact["research_references_updated"] is True
    assert [row["source_id"] for row in artifact["new_references_added"]] == [
        "lazy_identity_deq_2607_11116",
    ]


def test_scenario_report_5604_field_principle_validation_rejects_bad_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5604-FIELD-PRINCIPLES: missing principle fails closed."""

    root = _make_repo(tmp_path, _planner_references())
    artifact = mod.build_artifact(
        root=root,
        search_timestamp_utc="2026-07-14T03:29:56Z",
        duration_s=0.25,
    )
    mod.validate_artifact(artifact)

    broken = dict(artifact)
    broken["field_principles"] = dict(artifact["field_principles"])
    broken["field_principles"].pop("search_timestamp_utc")
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(broken)
