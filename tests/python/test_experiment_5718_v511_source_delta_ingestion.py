"""Tests for Exp5718 V511 execution source-delta ingestion.

Spec refs: REQ-REPORT-5718, SCENARIO-REPORT-5718-ACCEPT-BOUNDED-DELTA,
SCENARIO-REPORT-5718-BLOCKED-MARKER,
SCENARIO-REPORT-5718-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from carnot import experiment_5718_v511_source_delta_ingestion as mod


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
            "exp5717-transition-v511",
            "exp5718-v511-source-delta-ingestion",
            "exp5719-sota-answer-channel-forensics",
            "exp5720-sota-attested-exact-envelope-canary",
            "exp5721-fr11-memops-lifecycle-shadow-stream",
            "exp5722-fr11-compliance-recovery-rollback-canary",
            "exp5723-one-axis-rust-samplerbackend-integration",
            "exp5724-one-axis-rust-python-matched-crossover",
            "exp5725-arc-epistemic-ledger-live-qualification",
            "exp5726-arc-epistemic-ledger-live-ab",
            "exp5727-arc-live-self-discovery-levelup-v511",
            "exp5728-v511-capstone-reconciliation",
        )
    ]
    return yaml.safe_dump({"milestone": mod.MILESTONE, "tasks": tasks}, sort_keys=False)


def _planner_references() -> str:
    return (
        "## V511 Planner Refresh - 20260715\n\n"
        "- **Evidence-Grounded Verified Agentic Reasoning (EG-VAR)** - "
        "arXiv:2607.12650.\n"
        "- **MemOps: Benchmarking Lifecycle Memory Operations in "
        "Long-Horizon Conversations** - arXiv:2607.12893.\n"
        "- **The Compliance Trap: Diagnosing How AI Agents Consume "
        "Conflicting Memory** - arXiv:2607.10608.\n"
        "<!-- V511-PLANNER-REFRESH-20260715-END -->\n"
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
            "- id: exp5707-v510-source-delta-ingestion\n"
            "  title: V510 source delta\n"
            "  source: arXiv:2607.00447\n"
            "- id: exp5716-v510-capstone-reconciliation\n"
            "  title: no prospective FR-11 promotion, ARC delta 0, Rust positive\n"
        ),
        encoding="utf-8",
    )
    (root / mod.ROADMAP_RELATIVE_PATH).write_text(_roadmap(), encoding="utf-8")
    if roadmap_next:
        (root / mod.ROADMAP_NEXT_RELATIVE_PATH).write_text(_roadmap(), encoding="utf-8")
    (root / "openspec/change-proposals").mkdir(parents=True, exist_ok=True)
    (root / mod.VNEXT_RELATIVE_PATH).write_text(
        "# Research Roadmap vNEXT\n"
        "**Milestone:** 2026.07.511\n"
        "**Task range:** Exp5717-Exp5728\n"
        "Exp5721 records lifecycle operations and Exp5722 tests rollback.\n",
        encoding="utf-8",
    )
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / mod.EXCLUSION_MANIFEST_RELATIVE_PATH).write_text(
        (
            "retired: []\n"
            "retired_extras:\n"
            "- id: native_json_grammar_runtime_closed\n"
            "  reason: retired native grammar route\n"
            "- id: two_axis_exchange_closed\n"
            "  reason: Exp5645 terminal quality-negative result\n"
        ),
        encoding="utf-8",
    )
    (root / mod.KNOWN_ISSUES_RELATIVE_PATH).write_text(
        "JSON grammar, external generated-text scoring, token/logit authority, "
        "model-weight writes, PTRM generation, generic exploration signals, "
        "transition patching, two-axis exchange, TSU/Kona execution, and "
        "unsupported speedups remain closed.\n",
        encoding="utf-8",
    )
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts/research_conductor.py").write_text("# fixture\n", encoding="utf-8")
    (root / "results").mkdir(parents=True, exist_ok=True)
    return root


def test_req_report_5718_spec_declares_v511_source_delta_contract() -> None:
    """REQ-REPORT-5718: OpenSpec anchors V511 source-delta fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5718") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-REPORT-5718",
        "SCENARIO-REPORT-5718-ACCEPT-BOUNDED-DELTA",
        "SCENARIO-REPORT-5718-BLOCKED-MARKER",
        "SCENARIO-REPORT-5718-FIELD-PRINCIPLES",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "`huggingface_status`",
        "`github_status`",
        "`accepted_findings`",
        "`roadmap_change_required`",
        "`references_updated`",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_5718_accept_delta_appends_once(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5718-ACCEPT-BOUNDED-DELTA: one hook appends once."""

    root = _make_repo(tmp_path, _planner_references(), roadmap_next=True)

    artifact = mod.build_and_write_artifact(
        root=root,
        search_timestamp_utc="2026-07-19T19:23:31Z",
        duration_s=0.5,
    )

    mod.validate_artifact(artifact)
    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")
    result_text = (root / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")

    assert result_text.endswith("\n")
    assert references.count(mod.EXECUTION_REFRESH_HEADING) == 1
    assert references.index(mod.EXECUTION_REFRESH_HEADING) > references.index(
        mod.PLANNER_HEADING
    )
    assert "Do Agent Optimizers Compound?" in references
    assert "arXiv:2607.14004" in references
    assert artifact["planner_marker_found"] is True
    assert artifact["references_updated"] is True
    assert artifact["references_mutated_this_run"] is True
    assert artifact["roadmap_change_required"] is False
    assert artifact["accepted_findings"][0]["source_id"] == (
        "do_agent_optimizers_compound_2607_14004"
    )
    assert artifact["accepted_findings"][0]["target_experiments"] == [
        "exp5721-fr11-memops-lifecycle-shadow-stream",
        "exp5722-fr11-compliance-recovery-rollback-canary",
    ]
    assert artifact["target_experiment_map"][0]["validator_boundary"].startswith(
        "exact-label"
    )
    assert artifact["target_experiment_map"][0]["falsifiable_metric"]
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["roadmap_context"]["source"] == "research-roadmap-next.yaml"
    assert artifact["honest_verdict"].startswith("complete:")

    duplicate_ids = {row["source_id"] for row in artifact["duplicate_findings"]}
    assert {"eg_var_2607_12650", "memops_2607_12893"}.issubset(duplicate_ids)
    watch_ids = {row["source_id"] for row in artifact["watch_only_findings"]}
    assert {"byte_exact_kv_grafting_2607_14431", "photonic_ising_2607_13446"}.issubset(
        watch_ids
    )
    excluded_ids = {row["source_id"] for row in artifact["excluded_findings"]}
    assert {
        "generative_compilation_2607_13921",
        "seed_self_evolving_distillation_2607_14777",
        "native_json_grammar_runtime",
        "non_local_tsu_kona_execution",
    }.issubset(excluded_ids)

    artifact_second = mod.build_and_write_artifact(
        root=root,
        search_timestamp_utc="2026-07-19T19:23:32Z",
        duration_s=0.5,
    )
    references_second = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(
        encoding="utf-8"
    )
    assert references_second.count(mod.EXECUTION_REFRESH_HEADING) == 1
    assert artifact_second["references_updated"] is True
    assert artifact_second["references_mutated_this_run"] is False


def test_scenario_report_5718_missing_planner_marker_blocks_append(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5718-BLOCKED-MARKER: missing marker blocks mutation."""

    references_text = "## Earlier Refresh\n\nKnown references only.\n"
    root = _make_repo(tmp_path, references_text)

    artifact = mod.build_and_write_artifact(
        root=root,
        search_timestamp_utc="2026-07-19T19:23:31Z",
        duration_s=0.5,
    )

    mod.validate_artifact(artifact)
    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")
    assert references == references_text
    assert artifact["planner_marker_found"] is False
    assert artifact["references_updated"] is False
    assert artifact["references_mutated_this_run"] is False
    assert artifact["accepted_findings"] == []
    assert artifact["roadmap_change_required"] is False
    assert artifact["honest_verdict"].startswith("blocked:")


def test_scenario_report_5718_secondary_routes_and_boundaries(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5718-ACCEPT-BOUNDED-DELTA: dispositions are explicit."""

    root = _make_repo(tmp_path, _planner_references())
    artifact = mod.build_artifact(
        root=root,
        search_timestamp_utc="2026-07-19T19:23:31Z",
        duration_s=0.5,
        references_updated=True,
    )

    mod.validate_artifact(artifact)
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
    assert artifact["semantic_scholar_status"]["post_marker_citation_count"] == 0
    assert artifact["semantic_scholar_status"]["http_status"] == 200
    assert artifact["extropic_status"]["local_execution_available"] is False
    assert artifact["logical_intelligence_status"]["local_execution_available"] is False
    assert artifact["huggingface_status"]["dates_checked"] == [
        "2026-07-16",
        "2026-07-17",
        "2026-07-18",
        "2026-07-19",
    ]
    assert artifact["github_status"]["direct_repositories_checked"] >= 4
    assert artifact["closed_scope_review"] == {
        "json_grammar_reopened": False,
        "external_generated_text_scoring_reopened": False,
        "token_or_logit_authority_reopened": False,
        "model_weight_writes_reopened": False,
        "ptrm_generation_reopened": False,
        "generic_exploration_signals_reopened": False,
        "transition_patching_reopened": False,
        "two_axis_exchange_reopened": False,
        "non_local_tsu_or_kona_execution_reopened": False,
        "unsupported_speedups_reopened": False,
        "operator_authorized_scope_expansion": None,
    }
    assert any(
        row["source_id"] == "semantic_scholar_ebt_route"
        and row["status"] == "http_200_no_post_marker_citation_delta"
        for row in artifact["source_link_checks"]
    )


def test_scenario_report_5718_field_principle_validation_rejects_bad_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5718-FIELD-PRINCIPLES: malformed artifacts fail closed."""

    root = _make_repo(tmp_path, _planner_references())
    artifact = mod.build_artifact(
        root=root,
        search_timestamp_utc="2026-07-19T19:23:31Z",
        duration_s=0.5,
        references_updated=True,
    )
    mod.validate_artifact(artifact)

    broken = dict(artifact)
    broken["field_principles"] = dict(artifact["field_principles"])
    broken["field_principles"].pop("github_status")
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["references_updated"] = "yes"
    with pytest.raises(ValueError, match="references_updated"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["search_timestamp_utc"] = "2026-07-19T19:23:31"
    with pytest.raises(ValueError, match="timestamp"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["target_experiment_map"] = [
        {"target_experiments": ["exp5728-v511-capstone-reconciliation"]}
    ]
    with pytest.raises(ValueError, match="target experiment"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["accepted_findings"] = [dict(artifact["accepted_findings"][0])]
    broken["accepted_findings"][0].pop("validator_boundary")
    with pytest.raises(ValueError, match="accepted finding"):
        mod.validate_artifact(broken)


def test_scenario_report_5718_defensive_helpers_cover_edge_cases(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-REPORT-5718-FIELD-PRINCIPLES: helper boundaries stay explicit."""

    root = _make_repo(tmp_path, _planner_references())
    original = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")
    assert mod._relative_path(tmp_path, Path("/outside/root.txt")) == "/outside/root.txt"
    assert mod._honest_verdict(False, []).startswith("blocked:")
    assert mod._honest_verdict(True, []).startswith("complete: no new")
    assert mod._honest_verdict(True, list(mod.ACCEPTED_FINDINGS)).startswith(
        "complete: accepted 1 non-duplicate actionable V511"
    )
    assert mod._append_execution_refresh_if_needed(root, False, mod._accepted_findings()) is False
    assert (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8") == original
    fallback_insert = mod._insert_after_planner_block(
        "## V511 Planner Refresh - 20260715\nNo end marker.\n",
        "## V511 Execution Refresh - 20260719\n\nBody.\n",
    )
    assert fallback_insert.endswith("Body.\n")

    cli_root = _make_repo(tmp_path / "cli", _planner_references())
    assert (
        mod.main(
            [
                "--root",
                str(cli_root),
                "--search-timestamp-utc",
                "2026-07-19T19:23:33Z",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert mod.RESULT_RELATIVE_PATH.as_posix() in captured.out


def test_scenario_report_5718_remaining_defensive_branches(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5718-FIELD-PRINCIPLES: fail-closed branches are tested."""

    bare_root = tmp_path / "bare"
    bare_root.mkdir()
    assert mod._read_text_if_present(bare_root / "missing.md") == ""
    assert mod._proposal_paths(bare_root) == []
    assert mod._normalize_timestamp(None).endswith("Z")
    assert (
        mod._insert_after_planner_block("No planner marker.\n", "Refresh.\n")
        == "No planner marker.\nRefresh.\n"
    )
    assert mod._insert_after_planner_block(
        mod.PLANNER_HEADING,
        "Refresh.\n",
    ).startswith(mod.PLANNER_HEADING + "\n")

    root = _make_repo(tmp_path / "default", _planner_references())
    artifact = mod.build_artifact(
        root=root,
        search_timestamp_utc="2026-07-19T19:23:31Z",
    )
    assert artifact["references_updated"] is True

    blocked = dict(artifact)
    blocked["roadmap_change_required"] = True
    blocked["honest_verdict"] = "complete: incorrect nonblocking verdict"
    blocked["reproducibility_checksum"] = mod.payload_checksum(blocked)
    with pytest.raises(ValueError, match="scope expansion"):
        mod.validate_artifact(blocked)
