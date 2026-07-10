"""Tests for Exp5551 V503 execution source-delta ingestion.

Spec refs: REQ-REPORT-5551, SCENARIO-REPORT-5551,
SCENARIO-REPORT-5551-NOOP, SCENARIO-REPORT-5551-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from carnot import experiment_5551_v503_source_delta_ingestion as mod


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
            "exp5551-v503-source-delta-ingestion",
            "exp5552-automaton-schema-row-completion-receipt",
            "exp5553-gated-gbnf-forced-sota-row-smoke",
            "exp5555-asp-fsm-nonmonotonic-fixture",
            "exp5558-gated-causal-write-manage-read-csl-memory",
            "exp5560-hardware-and-timing-receipt-hygiene",
            "exp5561-arc-fsm-target-rotation-precheck",
        )
    ]
    return yaml.safe_dump({"milestone": mod.MILESTONE, "tasks": tasks}, sort_keys=False)


def _make_repo(root: Path, references_text: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for relative in ("CLAUDE.md", "CODEX.md", "research-program.md"):
        (root / relative).write_text("fixture\n", encoding="utf-8")
    (root / "research-roadmap.yaml").write_text(_roadmap(), encoding="utf-8")
    (root / "openspec/change-proposals").mkdir(parents=True, exist_ok=True)
    (root / "openspec/change-proposals/research-roadmap-vNEXT.md").write_text(
        "# Research Roadmap vNEXT - 2026.07.503\nMilestone: `2026.07.503`\n",
        encoding="utf-8",
    )
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "ops/exclusion_manifest.yaml").write_text("retired: []\n", encoding="utf-8")
    (root / "ops/changelog.md").write_text("fixture changelog\n", encoding="utf-8")
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts/research_conductor.py").write_text("# fixture\n", encoding="utf-8")
    (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).write_text(references_text, encoding="utf-8")
    return root


def _planner_references() -> str:
    return (
        "## V503 Planner Refresh - 20260710\n\n"
        "- **Answer Set Programming Energised! End-to-End Neurosymbolic Reasoning and "
        "Learning with ASP and Energy Based Models** (https://arxiv.org/abs/2607.08136)\n"
        "- **Mitigating Bias in Locally Constrained Decoding via Tractable Proposals** "
        "(https://arxiv.org/abs/2606.01926)\n"
        "- **Memory for Autonomous LLM Agents** (https://arxiv.org/abs/2603.07670)\n"
        "<!-- V503-PLANNER-REFRESH-20260710-END -->\n"
    )


def test_req_report_5551_spec_declares_source_delta_contract() -> None:
    """REQ-REPORT-5551: OpenSpec anchors source-delta fields and append rules."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5551") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-REPORT-5551",
        "SCENARIO-REPORT-5551",
        "SCENARIO-REPORT-5551-NOOP",
        "SCENARIO-REPORT-5551-FIELD-PRINCIPLES",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "`closed_scopes_reopened`",
        "`research_references_updated`",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_5551_non_duplicate_appends_execution_refresh(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5551: non-duplicate ClassicLogic delta appends one V503 block."""

    root = _make_repo(tmp_path, _planner_references())

    artifact = mod.build_and_write_artifact(
        root=root,
        run_date="20260710",
        duration_s=0.25,
        semantic_scholar_status=mod.DEFAULT_SEMANTIC_SCHOLAR_STATUS,
    )

    mod.validate_artifact(artifact)
    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")
    result_text = (root / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")

    assert references.count(mod.EXECUTION_REFRESH_HEADING) == 1
    assert "ClassicLogic" in references
    assert "arXiv:2607.05185" in references
    assert result_text.endswith("\n")
    assert artifact["prior_refresh_marker_found"] is True
    assert artifact["research_references_updated"] is True
    assert artifact["closed_scopes_reopened"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert [row["source_id"] for row in artifact["new_references_added"]] == [
        "classiclogic_2607_05185"
    ]

    lanes = {row["lane"] for row in artifact["experiment_mappings"]}
    assert lanes == {
        "automaton row completion",
        "GBNF-forced SOTA rows",
        "ASP/FSM exact fixture",
        "causal CSL memory",
        "hardware timing receipts",
        "ARC live-path rotation",
    }
    asp_mapping = [
        row for row in artifact["experiment_mappings"] if row["lane"] == "ASP/FSM exact fixture"
    ][0]
    assert "classiclogic_2607_05185" in asp_mapping["source_ids"]
    assert "exp5555-asp-fsm-nonmonotonic-fixture" in asp_mapping["experiment_ids"]


def test_scenario_report_5551_duplicate_noop_leaves_references_unchanged(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5551-NOOP: duplicate source produces no reference append."""

    duplicate_text = (
        _planner_references()
        + "\nHistorical note: ClassicLogic arXiv:2607.05185 was already indexed.\n"
    )
    root = _make_repo(tmp_path, duplicate_text)

    artifact = mod.build_and_write_artifact(
        root=root,
        run_date="20260710",
        duration_s=0.25,
        semantic_scholar_status=mod.DEFAULT_SEMANTIC_SCHOLAR_STATUS,
    )

    mod.validate_artifact(artifact)
    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")

    assert references == duplicate_text
    assert mod.EXECUTION_REFRESH_HEADING not in references
    assert artifact["prior_refresh_marker_found"] is True
    assert artifact["research_references_updated"] is False
    assert artifact["new_references_added"] == []
    suppressed_ids = {row["source_id"] for row in artifact["duplicates_suppressed"]}
    assert "classiclogic_2607_05185" in suppressed_ids
    assert artifact["honest_verdict"].startswith("complete:")


def test_scenario_report_5551_missing_planner_marker_blocks_append(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5551-NOOP: missing planner marker blocks reference mutation."""

    references_text = "## Earlier Refresh\n\nClassic references only.\n"
    root = _make_repo(tmp_path, references_text)

    artifact = mod.build_and_write_artifact(
        root=root,
        run_date="20260710",
        duration_s=0.25,
        semantic_scholar_status=mod.DEFAULT_SEMANTIC_SCHOLAR_STATUS,
    )

    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")
    mod.validate_artifact(artifact)
    assert references == references_text
    assert artifact["prior_refresh_marker_found"] is False
    assert artifact["research_references_updated"] is False
    assert artifact["honest_verdict"].startswith("blocked:")


def test_scenario_report_5551_existing_execution_block_is_stable(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5551: existing execution block is reported without duplicate append."""

    existing = (
        _planner_references()
        + "\n"
        + mod.render_execution_refresh_block([mod.CLASSICLOGIC_FINDING], run_date="20260710")
    )
    root = _make_repo(tmp_path, existing)

    artifact = mod.build_and_write_artifact(
        root=root,
        run_date="20260710",
        duration_s=0.25,
        semantic_scholar_status=mod.DEFAULT_SEMANTIC_SCHOLAR_STATUS,
    )

    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")
    mod.validate_artifact(artifact)
    assert references == existing
    assert references.count(mod.EXECUTION_REFRESH_HEADING) == 1
    assert artifact["research_references_updated"] is True
    assert [row["source_id"] for row in artifact["new_references_added"]] == [
        "classiclogic_2607_05185"
    ]


def test_scenario_report_5551_field_principle_validation_rejects_bad_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5551-FIELD-PRINCIPLES: invalid field principles fail closed."""

    root = _make_repo(tmp_path, _planner_references())
    artifact = mod.build_artifact(
        root=root,
        run_date="20260710",
        duration_s=0.25,
        semantic_scholar_status=mod.DEFAULT_SEMANTIC_SCHOLAR_STATUS,
    )
    mod.validate_artifact(artifact)

    broken = dict(artifact)
    broken["field_principles"] = dict(artifact["field_principles"])
    broken["field_principles"].pop("honest_verdict")
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(broken)
