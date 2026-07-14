"""Tests for Exp5579 V505 execution source-delta ingestion.

Spec refs: REQ-REPORT-5579, SCENARIO-REPORT-5579,
SCENARIO-REPORT-5579-NOOP, SCENARIO-REPORT-5579-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from carnot import experiment_5579_v505_source_delta_ingestion as mod


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
            "exp5579-v505-source-delta-ingestion",
            "exp5580-parser-forensics-positive-control",
            "exp5581-clean-sota-solve-verify-remeasurement",
            "exp5582-exact-counterexample-verifier-extension",
            "exp5583-causal-memory-metric-corrigendum",
            "exp5584-two-timescale-exact-gated-controller",
            "exp5585-reset-free-live-local-sota-sessions",
            "exp5586-delayed-promotion-poisoning-gate",
            "exp5587-ptrm-leave-one-game-out-adjudication",
            "exp5588-eom-mcts-live-precheck",
            "exp5589-gated-ordinary-arc-levelup",
            "exp5590-matched-cpu-cuda-crossover-board-continuity",
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
        "# Research Roadmap vNEXT\n**Milestone:** 2026.07.505\n**Task range:** exp5578-exp5591\n",
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
        "## V505 Planner Refresh - 20260711\n\n"
        "- **PACE: Two-Timescale Self-Evolution for Small Language Model Agents** "
        "(https://arxiv.org/abs/2605.23019)\n"
        "- **EvoPolicyGym: Evaluating Autonomous Policy Evolution in Interactive Environments** "
        "(https://arxiv.org/abs/2607.02440)\n"
        "- **LLM-as-a-Verifier** and **The Verification Horizon** were already indexed.\n"
        "<!-- V505-PLANNER-REFRESH-20260711-END -->\n"
    )


def test_req_report_5579_spec_declares_v505_source_delta_contract() -> None:
    """REQ-REPORT-5579: OpenSpec anchors V505 source-delta fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5579") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-REPORT-5579",
        "SCENARIO-REPORT-5579",
        "SCENARIO-REPORT-5579-NOOP",
        "SCENARIO-REPORT-5579-FIELD-PRINCIPLES",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "`planner_marker_found`",
        "`closed_scopes_reopened`",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_5579_non_duplicate_appends_execution_refresh(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5579: new exact-verifier deltas append one V505 block."""

    root = _make_repo(tmp_path, _planner_references())

    artifact = mod.build_and_write_artifact(root=root, run_date="20260714", duration_s=0.25)

    mod.validate_artifact(artifact)
    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")
    result_text = (root / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")

    assert references.count(mod.EXECUTION_REFRESH_HEADING) == 1
    assert "Agentic Proof and Property-Based Testing" in references
    assert "arXiv:2607.09072" in references
    assert "Deceptive Grounding" in references
    assert "arXiv:2607.09349" in references
    assert result_text.endswith("\n")
    assert artifact["planner_marker_found"] is True
    assert artifact["research_references_updated"] is True
    assert artifact["closed_scopes_reopened"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert [row["source_id"] for row in artifact["new_references_added"]] == [
        "agentic_property_templates_2607_09072",
        "deceptive_grounding_2607_09349",
    ]

    verifier_mapping = [
        row
        for row in artifact["experiment_mappings"]
        if row["lane"] == "exact verifier residual extension"
    ][0]
    assert verifier_mapping["source_status"] == "accepted_plus_planner_context"
    assert "exp5582-exact-counterexample-verifier-extension" in verifier_mapping["experiment_ids"]
    assert "agentic_property_templates_2607_09072" in verifier_mapping["source_ids"]
    assert "deceptive_grounding_2607_09349" in verifier_mapping["source_ids"]

    surfaces = {row["surface"] for row in artifact["sources_checked"]}
    assert {
        "arXiv",
        "OpenReview",
        "Semantic Scholar",
        "Hugging Face Papers",
        "GitHub",
        "Extropic writing",
        "Logical Intelligence public pages",
        "local Carnot reference history",
    }.issubset(surfaces)


def test_scenario_report_5579_duplicate_noop_leaves_references_unchanged(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5579-NOOP: duplicate accepted sources do not append."""

    duplicate_text = (
        _planner_references() + "\nHistorical note: Agentic Proof arXiv:2607.09072 and "
        "Deceptive Grounding arXiv:2607.09349 were already indexed.\n"
    )
    root = _make_repo(tmp_path, duplicate_text)

    artifact = mod.build_and_write_artifact(root=root, run_date="20260714", duration_s=0.25)

    mod.validate_artifact(artifact)
    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")

    assert references == duplicate_text
    assert mod.EXECUTION_REFRESH_HEADING not in references
    assert artifact["planner_marker_found"] is True
    assert artifact["research_references_updated"] is False
    assert artifact["new_references_added"] == []
    suppressed_ids = {row["source_id"] for row in artifact["duplicates_suppressed"]}
    assert "agentic_property_templates_2607_09072" in suppressed_ids
    assert "deceptive_grounding_2607_09349" in suppressed_ids
    assert artifact["honest_verdict"].startswith("complete:")


def test_scenario_report_5579_missing_planner_marker_blocks_append(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5579-NOOP: missing planner marker blocks mutation."""

    references_text = "## Earlier Refresh\n\nKnown references only.\n"
    root = _make_repo(tmp_path, references_text)

    artifact = mod.build_and_write_artifact(root=root, run_date="20260714", duration_s=0.25)

    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")
    mod.validate_artifact(artifact)
    assert references == references_text
    assert artifact["planner_marker_found"] is False
    assert artifact["research_references_updated"] is False
    assert artifact["honest_verdict"].startswith("blocked:")


def test_scenario_report_5579_existing_execution_block_is_stable(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5579: existing execution block is not appended twice."""

    existing = (
        _planner_references()
        + "\n"
        + mod.render_execution_refresh_block(mod.CANDIDATE_FINDINGS, run_date="20260714")
    )
    root = _make_repo(tmp_path, existing)

    artifact = mod.build_and_write_artifact(root=root, run_date="20260714", duration_s=0.25)

    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")
    mod.validate_artifact(artifact)
    assert references == existing
    assert references.count(mod.EXECUTION_REFRESH_HEADING) == 1
    assert artifact["research_references_updated"] is True
    assert [row["source_id"] for row in artifact["new_references_added"]] == [
        "agentic_property_templates_2607_09072",
        "deceptive_grounding_2607_09349",
    ]


def test_scenario_report_5579_field_principle_validation_rejects_bad_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5579-FIELD-PRINCIPLES: missing principle fails closed."""

    root = _make_repo(tmp_path, _planner_references())
    artifact = mod.build_artifact(root=root, run_date="20260714", duration_s=0.25)
    mod.validate_artifact(artifact)

    broken = dict(artifact)
    broken["field_principles"] = dict(artifact["field_principles"])
    broken["field_principles"].pop("honest_verdict")
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(broken)
