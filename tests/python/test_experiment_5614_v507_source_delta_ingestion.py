"""Tests for Exp5614 V507 execution source-delta ingestion.

Spec refs: REQ-REPORT-5614, SCENARIO-REPORT-5614-NOOP,
SCENARIO-REPORT-5614-BLOCKED-MARKER,
SCENARIO-REPORT-5614-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_5614_v507_source_delta_ingestion as mod


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
            "exp5613-transition-v507",
            "exp5614-v507-source-delta-ingestion",
            "exp5615-native-llamacpp-cuda-runtime-certificate",
            "exp5616-exact-nonstationary-constraint-stream",
            "exp5617-kan-critical-task-duration-map",
            "exp5618-predictive-window-kan-self-learning",
            "exp5619-arc-forward-inverse-transition-cycle",
            "exp5620-arc-cycle-guarded-live-update-ab",
            "exp5621-arc-live-self-discovery-levelup-v507",
            "exp5622-cdls-exact-kernel-audit",
            "exp5623-cdls-multiseed-cpu-cuda-crossover",
            "exp5624-v507-capstone-reconciliation",
        )
    ]
    return yaml.safe_dump({"milestone": mod.MILESTONE, "tasks": tasks}, sort_keys=False)


def _planner_references() -> str:
    return (
        "## V507 Planner Refresh - 20260714\n\n"
        "- **To Retain or to Adapt? Generalizing Continual Learning** - "
        "arXiv:2607.05609.\n"
        "- **When Does Continual Learning Require Learning** - "
        "arXiv:2607.07847.\n"
        "- **Loss Smoothing for Continual Adaptation** - OpenReview pUqcOkV69j.\n"
        "- **World Action Verifier** - arXiv:2604.01985.\n"
        "- **cDLS** - OpenReview fNI2fPyAfQ.\n"
        "<!-- V507-PLANNER-REFRESH-20260714-END -->\n"
    )


def _make_repo(root: Path, references_text: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for relative in ("AGENTS.md", "CODEX.md", "CLAUDE.md", "research-program.md"):
        (root / relative).write_text("fixture\n", encoding="utf-8")
    (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).write_text(
        references_text,
        encoding="utf-8",
    )
    (root / mod.RESEARCH_COMPLETE_RELATIVE_PATH).write_text(
        (
            "experiments:\n"
            "- id: exp5604-v506-source-delta-ingestion\n"
            "  title: The Equilibrium Is the Initialization\n"
            "  source: arXiv:2607.11116\n"
        ),
        encoding="utf-8",
    )
    (root / "research-roadmap.yaml").write_text(_roadmap(), encoding="utf-8")
    (root / "openspec/change-proposals").mkdir(parents=True, exist_ok=True)
    (root / mod.VNEXT_RELATIVE_PATH).write_text(
        "# Research Roadmap vNEXT\n"
        "**Milestone:** 2026.07.507\n"
        "**Task range:** Exp5613-Exp5624\n"
        "Exp5614 searches after the V507 planner marker and a no-op is terminal.\n",
        encoding="utf-8",
    )
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / mod.EXCLUSION_MANIFEST_RELATIVE_PATH).write_text(
        "retired: []\nretired_extras: []\n", encoding="utf-8"
    )
    (root / mod.KNOWN_ISSUES_RELATIVE_PATH).write_text(
        "Retired parser, solve-versus-verify, causal-memory, PTRM, SGE, "
        "generated-text scoring, hardware-board, and unmatched hardware-speedup "
        "scopes remain closed.\n",
        encoding="utf-8",
    )
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts/research_conductor.py").write_text("# fixture\n", encoding="utf-8")
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / mod.EXP5613_RELATIVE_PATH).write_text(
        json.dumps(
            {
                "experiment_id": "exp5613-transition-v507",
                "current_task_range": "exp5613-exp5624",
                "dependency_map": {
                    "kan_drift": {
                        "tasks": [
                            "exp5616-exact-nonstationary-constraint-stream",
                            "exp5617-kan-critical-task-duration-map",
                            "exp5618-predictive-window-kan-self-learning",
                        ]
                    }
                },
                "retired_scopes": [
                    {"key": "solve_verify_panel", "closed": True},
                    {"key": "unmatched_cdls_crossover", "closed": True},
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def test_req_report_5614_spec_declares_v507_source_delta_contract() -> None:
    """REQ-REPORT-5614: OpenSpec anchors V507 source-delta fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5614") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-REPORT-5614",
        "SCENARIO-REPORT-5614-NOOP",
        "SCENARIO-REPORT-5614-BLOCKED-MARKER",
        "SCENARIO-REPORT-5614-FIELD-PRINCIPLES",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "`search_timestamp_utc`",
        "`reproducibility_checksum`",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_5614_noop_leaves_references_unchanged(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5614-NOOP: no exact new local hook means no append."""

    root = _make_repo(tmp_path, _planner_references())

    artifact = mod.build_and_write_artifact(
        root=root,
        search_timestamp_utc="2026-07-14T08:25:30Z",
        duration_s=0.25,
    )

    mod.validate_artifact(artifact)
    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")
    result_text = (root / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")

    assert references == _planner_references()
    assert mod.EXECUTION_REFRESH_HEADING not in references
    assert result_text.endswith("\n")
    assert artifact["planner_marker_found"] is True
    assert artifact["research_references_updated"] is False
    assert artifact["new_references_added"] == []
    assert artifact["closed_scopes_reopened"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["search_timestamp_utc"] == "2026-07-14T08:25:30Z"
    assert artifact["honest_verdict"].startswith("complete:")

    suppressed_ids = {row["source_id"] for row in artifact["duplicates_suppressed"]}
    assert {
        "retain_or_adapt_2607_05609",
        "when_cl_requires_learning_2607_07847",
        "loss_smoothing_openreview_puqcokv69j",
        "world_action_verifier_2604_01985",
        "cdls_openreview_fni2fpyafq",
    }.issubset(suppressed_ids)

    watch_ids = {row["source_id"] for row in artifact["watch_only_items"]}
    assert {
        "cycle_world_2607_11836",
        "confidently_wrong_2607_11414",
        "snn_csp_parallel_tempering_2607_08897",
        "logical_intelligence_public_updates",
        "extropic_tsu_xtr_z1",
    }.issubset(watch_ids)

    mapping_ids = {
        experiment_id
        for row in artifact["experiment_mappings"]
        for experiment_id in row["experiment_ids"]
    }
    assert {
        "exp5615-native-llamacpp-cuda-runtime-certificate",
        "exp5616-exact-nonstationary-constraint-stream",
        "exp5617-kan-critical-task-duration-map",
        "exp5618-predictive-window-kan-self-learning",
        "exp5619-arc-forward-inverse-transition-cycle",
        "exp5620-arc-cycle-guarded-live-update-ab",
        "exp5621-arc-live-self-discovery-levelup-v507",
        "exp5622-cdls-exact-kernel-audit",
        "exp5623-cdls-multiseed-cpu-cuda-crossover",
    }.issubset(mapping_ids)
    assert {row["source_status"] for row in artifact["experiment_mappings"]} <= {
        "duplicate_planner_context",
        "duplicate_or_watch_only",
        "watch_only_no_local_hook",
    }

    surfaces = {row["surface"] for row in artifact["sources_checked"]}
    assert {
        "arXiv",
        "OpenReview",
        "Semantic Scholar",
        "Hugging Face Papers",
        "GitHub discovery/trending",
        "Extropic writing",
        "Logical Intelligence public pages",
        "local Carnot ledgers",
    }.issubset(surfaces)


def test_scenario_report_5614_missing_planner_marker_blocks_append(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5614-BLOCKED-MARKER: missing marker blocks mutation."""

    references_text = "## Earlier Refresh\n\nKnown references only.\n"
    root = _make_repo(tmp_path, references_text)

    artifact = mod.build_and_write_artifact(
        root=root,
        search_timestamp_utc="2026-07-14T08:25:30Z",
        duration_s=0.25,
    )

    mod.validate_artifact(artifact)
    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")
    assert references == references_text
    assert artifact["planner_marker_found"] is False
    assert artifact["research_references_updated"] is False
    assert artifact["new_references_added"] == []
    assert artifact["honest_verdict"].startswith("blocked:")


def test_scenario_report_5614_marker_and_duplicate_checks_are_auditable(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5614-NOOP: link, marker, and duplicate checks are recorded."""

    root = _make_repo(tmp_path, _planner_references())

    artifact = mod.build_artifact(
        root=root,
        search_timestamp_utc="2026-07-14T08:25:30Z",
        duration_s=0.25,
    )

    mod.validate_artifact(artifact)
    assert artifact["marker_checks"]["planner_marker"] == mod.PLANNER_MARKER
    assert artifact["marker_checks"]["planner_marker_found"] is True
    assert artifact["marker_checks"]["execution_refresh_present"] is False
    assert artifact["duplicate_checks"]["candidate_count"] == 0
    assert artifact["duplicate_checks"]["accepted_count"] == 0
    assert artifact["duplicate_checks"]["duplicates_suppressed_count"] == len(
        artifact["duplicates_suppressed"]
    )
    assert any(
        row["path"] == mod.RESEARCH_COMPLETE_RELATIVE_PATH.as_posix()
        for row in artifact["dedupe_corpus_checked"]
    )
    assert any(
        row["source_id"] == "semantic_scholar_ebt_route" and row["status"] == "http_429"
        for row in artifact["source_link_checks"]
    )


def test_scenario_report_5614_sparse_repo_defaults_to_blocked_noop(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5614-BLOCKED-MARKER: absent ledgers stay terminal."""

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


def test_scenario_report_5614_field_principle_validation_rejects_bad_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5614-FIELD-PRINCIPLES: missing principle fails closed."""

    root = _make_repo(tmp_path, _planner_references())
    artifact = mod.build_artifact(
        root=root,
        search_timestamp_utc="2026-07-14T08:25:30Z",
        duration_s=0.25,
    )
    mod.validate_artifact(artifact)

    broken = dict(artifact)
    broken["field_principles"] = dict(artifact["field_principles"])
    broken["field_principles"].pop("reproducibility_checksum")
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(broken)


def test_req_report_5614_accepted_verdict_branch_is_terminal() -> None:
    """REQ-REPORT-5614: accepted deltas, if any, still use a terminal prefix."""

    verdict = mod._honest_verdict(True, [{"source_id": "future_local_exact_hook"}])

    assert verdict == (
        "complete: accepted 1 non-duplicate actionable V507 source deltas and "
        "kept retired scopes closed"
    )
