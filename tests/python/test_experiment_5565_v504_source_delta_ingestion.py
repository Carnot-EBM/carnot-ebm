"""Tests for Exp5565 V504 execution source-delta ingestion.

Spec refs: REQ-REPORT-5565, SCENARIO-REPORT-5565,
SCENARIO-REPORT-5565-NOOP, SCENARIO-REPORT-5565-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from carnot import experiment_5565_v504_source_delta_ingestion as mod


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
            "exp5565-v504-source-delta-ingestion",
            "exp5566-exact-asp-fsm-near-miss-corpus",
            "exp5567-gated-local-sota-solve-verify-asymmetry",
            "exp5568-gated-verifier-coevolution-trigger",
            "exp5569-causal-memory-policy-tournament",
            "exp5570-spline-local-kan-online-energy",
            "exp5571-gated-reset-free-sota-continual-harness",
            "exp5572-gated-delayed-regression-promotion",
            "exp5573-matched-sampler-hardware-continuity",
            "exp5574-ptrm-stochastic-generator-stage1",
            "exp5575-sge-anti-stagnation-live-precheck",
            "exp5576-gated-sge-live-levelup",
        )
    ]
    return yaml.safe_dump({"milestone": mod.MILESTONE, "tasks": tasks}, sort_keys=False)


def _make_repo(root: Path, references_text: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for relative in ("AGENTS.md", "CODEX.md", "CLAUDE.md", "research-program.md"):
        (root / relative).write_text("fixture\n", encoding="utf-8")
    (root / "research-roadmap.yaml").write_text(_roadmap(), encoding="utf-8")
    (root / "openspec/change-proposals").mkdir(parents=True, exist_ok=True)
    (root / "openspec/change-proposals/research-roadmap-vNEXT.md").write_text(
        "# Research Roadmap vNEXT\n**Milestone:** 2026.07.504\n**Task range:** exp5564-exp5577\n",
        encoding="utf-8",
    )
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "ops/exclusion_manifest.yaml").write_text("retired: []\n", encoding="utf-8")
    (root / "ops/known-issues.md").write_text("fixture known issues\n", encoding="utf-8")
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts/research_conductor.py").write_text("# fixture\n", encoding="utf-8")
    (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).write_text(references_text, encoding="utf-8")
    return root


def _planner_references() -> str:
    return (
        "## V504 Planner Refresh - 20260711\n\n"
        "- **SelfMem: Self-Optimizing Memory for AI Agents** (https://arxiv.org/abs/2607.03726)\n"
        "- **Continual Harness: Online Adaptation for Self-Improving Foundation Agents** "
        "(https://arxiv.org/abs/2605.09998)\n"
        "- **LLM-as-a-Verifier: A General-Purpose Verification Framework** "
        "(https://arxiv.org/abs/2607.05391)\n"
        "- **The Verification Horizon: No Silver Bullet for Coding Agent Rewards** "
        "(https://arxiv.org/abs/2606.26300)\n"
        "<!-- V504-PLANNER-REFRESH-20260711-END -->\n"
    )


def test_req_report_5565_spec_declares_source_delta_contract() -> None:
    """REQ-REPORT-5565: OpenSpec anchors V504 source-delta fields and rules."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5565") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-REPORT-5565",
        "SCENARIO-REPORT-5565",
        "SCENARIO-REPORT-5565-NOOP",
        "SCENARIO-REPORT-5565-FIELD-PRINCIPLES",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "`planner_marker_found`",
        "`closed_scopes_reopened`",
        "`citation_trails_checked`",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_5565_non_duplicate_appends_execution_refresh(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5565: Blind Curator delta appends one V504 block."""

    root = _make_repo(tmp_path, _planner_references())

    artifact = mod.build_and_write_artifact(root=root, run_date="20260711", duration_s=0.25)

    mod.validate_artifact(artifact)
    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")
    result_text = (root / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")

    assert references.count(mod.EXECUTION_REFRESH_HEADING) == 1
    assert "The Blind Curator" in references
    assert "arXiv:2607.07436" in references
    assert result_text.endswith("\n")
    assert artifact["planner_marker_found"] is True
    assert artifact["research_references_updated"] is True
    assert artifact["closed_scopes_reopened"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert [row["source_id"] for row in artifact["new_references_added"]] == [
        "blind_curator_2607_07436"
    ]

    mapped_experiments = {
        experiment_id
        for row in artifact["experiment_mappings"]
        for experiment_id in row["experiment_ids"]
    }
    assert "exp5569-causal-memory-policy-tournament" in mapped_experiments
    assert "exp5572-gated-delayed-regression-promotion" in mapped_experiments
    self_learning_mapping = [
        row
        for row in artifact["experiment_mappings"]
        if row["lane"] == "reset-free continuous self-learning stress"
    ][0]
    assert "blind_curator_2607_07436" in self_learning_mapping["source_ids"]


def test_scenario_report_5565_duplicate_noop_leaves_references_unchanged(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5565-NOOP: duplicate Blind Curator source does not append."""

    duplicate_text = (
        _planner_references()
        + "\nHistorical note: The Blind Curator arXiv:2607.07436 was already indexed.\n"
    )
    root = _make_repo(tmp_path, duplicate_text)

    artifact = mod.build_and_write_artifact(root=root, run_date="20260711", duration_s=0.25)

    mod.validate_artifact(artifact)
    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")

    assert references == duplicate_text
    assert mod.EXECUTION_REFRESH_HEADING not in references
    assert artifact["planner_marker_found"] is True
    assert artifact["research_references_updated"] is False
    assert artifact["new_references_added"] == []
    suppressed_ids = {row["source_id"] for row in artifact["duplicates_suppressed"]}
    assert "blind_curator_2607_07436" in suppressed_ids
    assert artifact["honest_verdict"].startswith("complete:")


def test_scenario_report_5565_missing_planner_marker_blocks_append(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5565-NOOP: missing planner marker blocks mutation."""

    references_text = "## Earlier Refresh\n\nKnown references only.\n"
    root = _make_repo(tmp_path, references_text)

    artifact = mod.build_and_write_artifact(root=root, run_date="20260711", duration_s=0.25)

    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")
    mod.validate_artifact(artifact)
    assert references == references_text
    assert artifact["planner_marker_found"] is False
    assert artifact["research_references_updated"] is False
    assert artifact["honest_verdict"].startswith("blocked:")


def test_scenario_report_5565_existing_execution_block_is_stable(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5565: existing execution block is not appended twice."""

    existing = (
        _planner_references()
        + "\n"
        + mod.render_execution_refresh_block([mod.BLIND_CURATOR_FINDING], run_date="20260711")
    )
    root = _make_repo(tmp_path, existing)

    artifact = mod.build_and_write_artifact(root=root, run_date="20260711", duration_s=0.25)

    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")
    mod.validate_artifact(artifact)
    assert references == existing
    assert references.count(mod.EXECUTION_REFRESH_HEADING) == 1
    assert artifact["research_references_updated"] is True
    assert [row["source_id"] for row in artifact["new_references_added"]] == [
        "blind_curator_2607_07436"
    ]


def test_scenario_report_5565_field_principle_validation_rejects_bad_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5565-FIELD-PRINCIPLES: missing principle fails closed."""

    root = _make_repo(tmp_path, _planner_references())
    artifact = mod.build_artifact(root=root, run_date="20260711", duration_s=0.25)
    mod.validate_artifact(artifact)

    broken = dict(artifact)
    broken["field_principles"] = dict(artifact["field_principles"])
    broken["field_principles"].pop("honest_verdict")
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(broken)
