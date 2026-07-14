"""Tests for Exp5626 V508 execution source-delta ingestion.

Spec refs: REQ-REPORT-5626, SCENARIO-REPORT-5626-NOOP,
SCENARIO-REPORT-5626-BLOCKED-MARKER,
SCENARIO-REPORT-5626-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_5626_v508_source_delta_ingestion as mod


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
            "exp5625-transition-v508",
            "exp5626-v508-source-delta-ingestion",
            "exp5627-online-conformal-kan-qualification",
            "exp5628-conformal-active-spline-kan-csl",
            "exp5629-conformal-kan-independent-audit",
            "exp5630-arc-epistemic-object-probe-prototype",
            "exp5631-arc-epistemic-probe-live-ab",
            "exp5632-arc-live-self-discovery-levelup-v508",
            "exp5633-temperature-exchange-cdls-exact-audit",
            "exp5634-temperature-exchange-cdls-quality",
            "exp5635-v508-capstone-reconciliation",
        )
    ]
    return yaml.safe_dump({"milestone": mod.MILESTONE, "tasks": tasks}, sort_keys=False)


def _planner_references() -> str:
    return (
        "## V508 Planner Refresh - 20260714\n\n"
        "- **Parameter-Free and Group Conditional Online Conformal Prediction** - "
        "arXiv:2606.00419.\n"
        "- **Optimal Training-Conditional Regret for Online Conformal Prediction** - "
        "arXiv:2602.16537.\n"
        "- **Breaking Local-Minimum Traps in Spiking Neural Network-Based Solvers for "
        "CSPs via Parallel Tempering** - arXiv:2607.08897.\n"
        "<!-- V508-PLANNER-REFRESH-20260714-END -->\n"
    )


def _make_repo(root: Path, references_text: str, *, roadmap_next: bool = False) -> Path:
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
            "- id: exp5614-v507-source-delta-ingestion\n"
            "  title: Energy-Based Transformers are Scalable Learners and Thinkers\n"
            "  source: arXiv:2507.02092\n"
        ),
        encoding="utf-8",
    )
    (root / "research-roadmap.yaml").write_text(_roadmap(), encoding="utf-8")
    if roadmap_next:
        (root / mod.ROADMAP_NEXT_RELATIVE_PATH).write_text(_roadmap(), encoding="utf-8")
    (root / "openspec/change-proposals").mkdir(parents=True, exist_ok=True)
    (root / mod.VNEXT_RELATIVE_PATH).write_text(
        "# Research Roadmap vNEXT\n"
        "**Milestone:** 2026.07.508\n"
        "**Task range:** Exp5625-Exp5635\n"
        "Exp5626 searches after the V508 planner marker and a no-op is terminal.\n",
        encoding="utf-8",
    )
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / mod.EXCLUSION_MANIFEST_RELATIVE_PATH).write_text(
        "retired: []\nretired_extras: []\n", encoding="utf-8"
    )
    (root / mod.KNOWN_ISSUES_RELATIVE_PATH).write_text(
        "Native runtime, solve-versus-verify, ARC transition-cycle proxy, "
        "unmatched cDLS crossover, generated-text scorer, TSU, Kona, Aleph, "
        "and hardware speedup scopes remain closed.\n",
        encoding="utf-8",
    )
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts/research_conductor.py").write_text("# fixture\n", encoding="utf-8")
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / mod.EXP5625_RELATIVE_PATH).write_text(
        json.dumps(
            {
                "experiment_id": "exp5625-transition-v508",
                "current_task_range": "exp5625-exp5635",
                "dependency_map": {
                    "conformal": {
                        "tasks": [
                            "exp5627-online-conformal-kan-qualification",
                            "exp5628-conformal-active-spline-kan-csl",
                            "exp5629-conformal-kan-independent-audit",
                        ]
                    }
                },
                "retired_scopes": [
                    {"key": "native_runtime_certificate", "closed": True},
                    {"key": "unmatched_cdls_crossover", "closed": True},
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def test_req_report_5626_spec_declares_v508_source_delta_contract() -> None:
    """REQ-REPORT-5626: OpenSpec anchors V508 source-delta fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5626") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-REPORT-5626",
        "SCENARIO-REPORT-5626-NOOP",
        "SCENARIO-REPORT-5626-BLOCKED-MARKER",
        "SCENARIO-REPORT-5626-FIELD-PRINCIPLES",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "`search_timestamp_utc`",
        "`reproducibility_checksum`",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_5626_noop_leaves_references_unchanged(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5626-NOOP: no exact new local hook means no append."""

    root = _make_repo(tmp_path, _planner_references(), roadmap_next=True)

    artifact = mod.build_and_write_artifact(
        root=root,
        search_timestamp_utc="2026-07-14T17:42:10Z",
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
    assert artifact["closed_scopes_reopened"] == mod.CLOSED_SCOPES_REOPENED
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["search_timestamp_utc"] == "2026-07-14T17:42:10Z"
    assert artifact["roadmap_context"]["source"] == "research-roadmap-next.yaml"
    assert artifact["honest_verdict"].startswith("complete:")

    suppressed_ids = {row["source_id"] for row in artifact["duplicates_suppressed"]}
    assert {
        "online_group_conformal_2606_00419",
        "training_conditional_regret_2602_16537",
        "snn_csp_parallel_tempering_2607_08897",
        "ebt_arm_ebm_semantic_scholar_routes",
    }.issubset(suppressed_ids)

    watch_ids = {row["source_id"] for row in artifact["watch_only_items"]}
    assert {
        "logical_intelligence_public_updates",
        "extropic_tsu_xtr_z1",
        "github_claim_level_hallucination_repo",
        "static_conformalized_kans_2504_15240",
    }.issubset(watch_ids)

    allowed_ids = {
        "exp5627-online-conformal-kan-qualification",
        "exp5628-conformal-active-spline-kan-csl",
        "exp5629-conformal-kan-independent-audit",
        "exp5630-arc-epistemic-object-probe-prototype",
        "exp5631-arc-epistemic-probe-live-ab",
        "exp5632-arc-live-self-discovery-levelup-v508",
        "exp5633-temperature-exchange-cdls-exact-audit",
        "exp5634-temperature-exchange-cdls-quality",
    }
    mapping_ids = {
        experiment_id
        for row in artifact["experiment_mappings"]
        for experiment_id in row["experiment_ids"]
    }
    assert mapping_ids == allowed_ids
    assert "exp5635-v508-capstone-reconciliation" not in mapping_ids

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


def test_scenario_report_5626_missing_planner_marker_blocks_append(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5626-BLOCKED-MARKER: missing marker blocks mutation."""

    references_text = "## Earlier Refresh\n\nKnown references only.\n"
    root = _make_repo(tmp_path, references_text)

    artifact = mod.build_and_write_artifact(
        root=root,
        search_timestamp_utc="2026-07-14T17:42:10Z",
        duration_s=0.25,
    )

    mod.validate_artifact(artifact)
    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")
    assert references == references_text
    assert artifact["planner_marker_found"] is False
    assert artifact["research_references_updated"] is False
    assert artifact["new_references_added"] == []
    assert artifact["honest_verdict"].startswith("blocked:")


def test_scenario_report_5626_marker_duplicate_and_link_checks_are_auditable(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5626-NOOP: link, marker, and duplicate checks are recorded."""

    root = _make_repo(tmp_path, _planner_references())

    artifact = mod.build_artifact(
        root=root,
        search_timestamp_utc="2026-07-14T17:42:10Z",
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
        row["source_id"] == "semantic_scholar_ebt_route"
        and row["status"] == "http_200_duplicate_citation_route"
        for row in artifact["source_link_checks"]
    )


def test_scenario_report_5626_sparse_repo_defaults_to_blocked_noop(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5626-BLOCKED-MARKER: absent ledgers stay terminal."""

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


def test_scenario_report_5626_field_principle_validation_rejects_bad_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5626-FIELD-PRINCIPLES: malformed fields fail closed."""

    root = _make_repo(tmp_path, _planner_references())
    artifact = mod.build_artifact(
        root=root,
        search_timestamp_utc="2026-07-14T17:42:10Z",
        duration_s=0.25,
    )
    mod.validate_artifact(artifact)

    broken = dict(artifact)
    broken["field_principles"] = dict(artifact["field_principles"])
    broken["field_principles"].pop("reproducibility_checksum")
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["closed_scopes_reopened"] = True
    with pytest.raises(ValueError, match="closed_scopes_reopened"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["search_timestamp_utc"] = "2026-07-14T17:42:10"
    with pytest.raises(ValueError, match="timestamp"):
        mod.validate_artifact(broken)


def test_scenario_report_5626_defensive_helpers_cover_edge_cases(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5626-FIELD-PRINCIPLES: helper boundaries stay explicit."""

    assert mod._relative_path(tmp_path, Path("/outside/root.txt")) == "/outside/root.txt"
    assert mod._honest_verdict(
        True,
        [{"source_id": "accepted", "title": "accepted source"}],
    ).startswith("complete: accepted 1 non-duplicate actionable V508")
