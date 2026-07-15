"""Tests for Exp5707 V510 execution source-delta ingestion.

Spec refs: REQ-REPORT-5707, SCENARIO-REPORT-5707-NOOP,
SCENARIO-REPORT-5707-BLOCKED-MARKER,
SCENARIO-REPORT-5707-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from carnot import experiment_5707_v510_source_delta_ingestion as mod


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
            "exp5706-transition-v510",
            "exp5707-v510-source-delta-ingestion",
            "exp5708-sota-exact-constraint-canary",
            "exp5709-fr11-prospective-shadow-stream",
            "exp5710-fr11-isolated-act-on-advice-canary",
            "exp5711-placement-spatial-goal-energy-live-path-qualification",
            "exp5712-known-level-live-path-relational-goal-ab",
            "exp5713-arc-live-self-discovery-levelup-v510",
            "exp5714-one-axis-rust-python-exact-parity",
            "exp5715-one-axis-hard-instance-quality-restart-parity",
            "exp5716-v510-capstone-reconciliation",
        )
    ]
    return yaml.safe_dump({"milestone": mod.MILESTONE, "tasks": tasks}, sort_keys=False)


def _planner_references() -> str:
    return (
        "## V510 Planner Refresh - 20260714\n\n"
        "- **Understanding Why Language Models Hallucinate: Testing Reasoning "
        "Against Priors** - arXiv:2607.00447.\n"
        "- **OEUVRE: OnlinE Unbiased Variance-Reduced Loss Estimation** - "
        "OpenReview 5jJnGctZMf.\n"
        "- **GAP-5703: live placement-goal energy is constant on sp80.**\n"
        "<!-- V510-PLANNER-REFRESH-20260714-END -->\n"
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
            "- id: exp5637-v509-source-delta-ingestion\n"
            "  title: Execution-time V509 source delta\n"
            "  source: arXiv:2605.16725\n"
            "- id: exp5645-two-axis-tempering-hard-constraint-quality\n"
            "  title: two-axis quality failed\n"
        ),
        encoding="utf-8",
    )
    (root / mod.ROADMAP_RELATIVE_PATH).write_text(_roadmap(), encoding="utf-8")
    if roadmap_next:
        (root / mod.ROADMAP_NEXT_RELATIVE_PATH).write_text(_roadmap(), encoding="utf-8")
    (root / "openspec/change-proposals").mkdir(parents=True, exist_ok=True)
    (root / mod.VNEXT_RELATIVE_PATH).write_text(
        "# Research Roadmap vNEXT\n"
        "**Milestone:** 2026.07.510\n"
        "**Task range:** Exp5706-Exp5716\n"
        "Exp5707 searches after the V510 planner marker and a no-op is terminal.\n",
        encoding="utf-8",
    )
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / mod.EXCLUSION_MANIFEST_RELATIVE_PATH).write_text(
        (
            "retired: []\n"
            "retired_extras:\n"
            "- id: two_axis_tempering_extension_closed\n"
            "  reason: Exp5645 quality failed\n"
        ),
        encoding="utf-8",
    )
    (root / mod.KNOWN_ISSUES_RELATIVE_PATH).write_text(
        "Native three-model JSON grammar runtime, external generated-text scoring, "
        "token steering, broad RL, retired ARC mechanisms, two-axis tempering, "
        "TSU, Kona, and unsupported speedup scopes remain closed.\n",
        encoding="utf-8",
    )
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts/research_conductor.py").write_text("# fixture\n", encoding="utf-8")
    (root / "results").mkdir(parents=True, exist_ok=True)
    return root


def test_req_report_5707_spec_declares_v510_source_delta_contract() -> None:
    """REQ-REPORT-5707: OpenSpec anchors V510 source-delta fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5707") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-REPORT-5707",
        "SCENARIO-REPORT-5707-NOOP",
        "SCENARIO-REPORT-5707-BLOCKED-MARKER",
        "SCENARIO-REPORT-5707-FIELD-PRINCIPLES",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "`accepted_findings`",
        "`duplicate_findings`",
        "`watch_only_findings`",
        "`excluded_findings`",
        "`roadmap_change_required`",
        "`references_updated`",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_5707_noop_leaves_references_unchanged(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5707-NOOP: no exact new local hook means no append."""

    root = _make_repo(tmp_path, _planner_references(), roadmap_next=True)

    artifact = mod.build_and_write_artifact(
        root=root,
        search_timestamp_utc="2026-07-14T19:21:44Z",
        duration_s=0.5,
    )

    mod.validate_artifact(artifact)
    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")
    result_text = (root / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")

    assert references == _planner_references()
    assert mod.EXECUTION_REFRESH_HEADING not in references
    assert result_text.endswith("\n")
    assert artifact["planner_marker_found"] is True
    assert artifact["planner_marker"] == mod.PLANNER_MARKER
    assert artifact["references_updated"] is False
    assert artifact["roadmap_change_required"] is False
    assert artifact["accepted_findings"] == []
    assert artifact["target_experiment_map"] == []
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["search_timestamp_utc"] == "2026-07-14T19:21:44Z"
    assert artifact["roadmap_context"]["source"] == "research-roadmap-next.yaml"
    assert artifact["honest_verdict"].startswith("complete:")

    duplicate_ids = {row["source_id"] for row in artifact["duplicate_findings"]}
    assert {
        "trapqa_2607_00447",
        "oeuvre_openreview_5jJnGctZMf",
        "gap_5703_sp80_goal_energy",
        "semantic_scholar_ebt_2607_11555",
    }.issubset(duplicate_ids)

    watch_ids = {row["source_id"] for row in artifact["watch_only_findings"]}
    assert {
        "extropic_tsu_xtr_z1",
        "logical_intelligence_kona_aleph",
        "huggingface_papers_trapqa_mirror",
        "github_sampler_and_kan_discovery",
    }.issubset(watch_ids)

    excluded_ids = {row["source_id"] for row in artifact["excluded_findings"]}
    assert {
        "native_three_model_json_grammar_runtime",
        "two_axis_tempering_extension",
        "external_generated_text_scoring",
        "non_local_tsu_kona_execution",
    }.issubset(excluded_ids)

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

    query_surfaces = {row["surface"] for row in artifact["queries"]}
    assert surfaces <= query_surfaces
    assert artifact["semantic_scholar_status"]["route"] == "Semantic Scholar Graph API"
    assert artifact["extropic_status"]["local_execution_available"] is False
    assert artifact["logical_intelligence_status"]["local_execution_available"] is False


def test_scenario_report_5707_missing_planner_marker_blocks_append(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5707-BLOCKED-MARKER: missing marker blocks mutation."""

    references_text = "## Earlier Refresh\n\nKnown references only.\n"
    root = _make_repo(tmp_path, references_text)

    artifact = mod.build_and_write_artifact(
        root=root,
        search_timestamp_utc="2026-07-14T19:21:44Z",
        duration_s=0.5,
    )

    mod.validate_artifact(artifact)
    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")
    assert references == references_text
    assert artifact["planner_marker_found"] is False
    assert artifact["references_updated"] is False
    assert artifact["accepted_findings"] == []
    assert artifact["roadmap_change_required"] is False
    assert artifact["honest_verdict"].startswith("blocked:")


def test_scenario_report_5707_checks_and_boundaries_are_auditable(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5707-NOOP: link, marker, and disposition checks exist."""

    root = _make_repo(tmp_path, _planner_references())
    artifact = mod.build_artifact(
        root=root,
        search_timestamp_utc="2026-07-14T19:21:44Z",
        duration_s=0.5,
    )

    mod.validate_artifact(artifact)
    assert artifact["marker_checks"]["planner_marker"] == mod.PLANNER_MARKER
    assert artifact["marker_checks"]["planner_marker_found"] is True
    assert artifact["marker_checks"]["execution_refresh_present"] is False
    assert artifact["duplicate_checks"]["accepted_count"] == 0
    assert artifact["duplicate_checks"]["duplicate_count"] == len(artifact["duplicate_findings"])
    assert artifact["duplicate_checks"]["watch_only_count"] == len(
        artifact["watch_only_findings"]
    )
    assert artifact["duplicate_checks"]["excluded_count"] == len(artifact["excluded_findings"])
    assert any(
        row["path"] == mod.RESEARCH_COMPLETE_RELATIVE_PATH.as_posix()
        for row in artifact["dedupe_corpus_checked"]
    )
    assert any(
        row["source_id"] == "semantic_scholar_ebt_route"
        and "no_new_dependency" in row["status"]
        for row in artifact["source_link_checks"]
    )
    assert artifact["closed_scope_review"] == {
        "native_three_model_json_grammar_runtime_reopened": False,
        "external_generated_text_scoring_reopened": False,
        "token_steering_or_broad_rl_reopened": False,
        "retired_arc_mechanisms_reopened": False,
        "two_axis_tempering_reopened": False,
        "non_local_tsu_or_kona_execution_reopened": False,
        "unsupported_speedup_reopened": False,
        "operator_authorized_scope_expansion": None,
    }


def test_scenario_report_5707_field_principle_validation_rejects_bad_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5707-FIELD-PRINCIPLES: malformed artifacts fail closed."""

    root = _make_repo(tmp_path, _planner_references())
    artifact = mod.build_artifact(
        root=root,
        search_timestamp_utc="2026-07-14T19:21:44Z",
        duration_s=0.5,
    )
    mod.validate_artifact(artifact)

    broken = dict(artifact)
    broken["field_principles"] = dict(artifact["field_principles"])
    broken["field_principles"].pop("reproducibility_checksum")
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["roadmap_change_required"] = True
    with pytest.raises(ValueError, match="roadmap_change_required"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["references_updated"] = True
    with pytest.raises(ValueError, match="references_updated"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["search_timestamp_utc"] = "2026-07-14T19:21:44"
    with pytest.raises(ValueError, match="timestamp"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["target_experiment_map"] = [
        {"target_experiment": "exp5716-v510-capstone-reconciliation"}
    ]
    with pytest.raises(ValueError, match="target experiment"):
        mod.validate_artifact(broken)
