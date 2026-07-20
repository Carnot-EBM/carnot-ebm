"""Tests for Exp5732 V512 source-delta ingestion.

Spec refs: REQ-REPORT-5732, SCENARIO-REPORT-5732-NOOP,
SCENARIO-REPORT-5732-ACCEPT-BOUNDED-DELTA,
SCENARIO-REPORT-5732-BLOCKED-MARKER,
SCENARIO-REPORT-5732-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest
import yaml

from carnot import experiment_5732_v512_source_delta_ingestion as mod


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
            "exp5731-transition-v512",
            "exp5732-v512-source-delta-ingestion",
            "exp5733-sota-finite-choice-proposal-channel",
            "exp5734-sota-exact-proposal-stream",
            "exp5735-zero-gate-kan-continuous-self-learning",
            "exp5736-csl-lifecycle-conflict-rollback",
            "exp5737-sota-stream-csl-shadow-ingress",
            "exp5738-one-axis-rust-batched-backend",
            "exp5739-one-axis-batched-10x-crossover",
            "exp5740-arc-game-blind-primitive-causal-audit",
            "exp5741-arc-generic-primitive-live-ab",
            "exp5742-v512-capstone-reconciliation",
        )
    ]
    return yaml.safe_dump({"milestone": mod.MILESTONE, "tasks": tasks}, sort_keys=False)


def _planner_references() -> str:
    return (
        "## V512 Planner Refresh - 20260719\n\n"
        "- **Generative Compilation** - arXiv:2607.13921.\n"
        "- **Gate-Zero Growth** - arXiv:2607.14571.\n"
        "- **SMC-ES** - arXiv:2607.15003.\n"
        "- **Campaign Diagrams** - arXiv:2607.15225.\n"
        "- **Bridge Evidence** - arXiv:2607.15253.\n"
        "- **Photonic Ising machines toward and beyond a million spins** - "
        "arXiv:2607.13446.\n"
        "<!-- V512-PLANNER-REFRESH-20260719-END -->\n"
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
            "- id: exp5718-v511-source-delta-ingestion\n"
            "  title: V511 source delta\n"
        ),
        encoding="utf-8",
    )
    (root / mod.ROADMAP_RELATIVE_PATH).write_text(_roadmap(), encoding="utf-8")
    if roadmap_next:
        (root / mod.ROADMAP_NEXT_RELATIVE_PATH).write_text(_roadmap(), encoding="utf-8")
    (root / "openspec/change-proposals").mkdir(parents=True, exist_ok=True)
    (root / mod.VNEXT_RELATIVE_PATH).write_text(
        "# Research Roadmap vNEXT\n"
        "**Milestone:** 2026.07.512\n"
        "**Task range:** Exp5731-Exp5742\n"
        "Exp5732 is bounded bibliographic work with no compute claim.\n",
        encoding="utf-8",
    )
    (root / "openspec/capabilities/research-reporting").mkdir(parents=True, exist_ok=True)
    (root / mod.SPEC_RELATIVE_PATH).write_text(
        "REQ-REPORT-5732\n"
        "SCENARIO-REPORT-5732-NOOP\n"
        "SCENARIO-REPORT-5732-ACCEPT-BOUNDED-DELTA\n"
        "SCENARIO-REPORT-5732-BLOCKED-MARKER\n"
        "SCENARIO-REPORT-5732-FIELD-PRINCIPLES\n",
        encoding="utf-8",
    )
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / mod.EXCLUSION_MANIFEST_RELATIVE_PATH).write_text(
        (
            "retired: []\n"
            "retired_extras:\n"
            "- id: free_form_answer_repair_closed\n"
            "  reason: closed for V512\n"
            "- id: unsupported_hardware_speedups_closed\n"
            "  reason: no authenticated local hardware timing\n"
        ),
        encoding="utf-8",
    )
    (root / mod.KNOWN_ISSUES_RELATIVE_PATH).write_text(
        "Free-form answer repair, JSON grammar, external generated-text scoring, "
        "token/logit semantic authority, model-weight writes, broad RL, PTRM "
        "generation, two-axis exchange, learned ARC value transfer, per-game "
        "adapters, and unsupported hardware speedups remain closed.\n",
        encoding="utf-8",
    )
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts/research_conductor.py").write_text("# fixture\n", encoding="utf-8")
    (root / "results").mkdir(parents=True, exist_ok=True)
    return root


def test_req_report_5732_spec_declares_v512_source_delta_contract() -> None:
    """REQ-REPORT-5732: OpenSpec anchors V512 source-delta fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5732") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-REPORT-5732",
        "SCENARIO-REPORT-5732-NOOP",
        "SCENARIO-REPORT-5732-ACCEPT-BOUNDED-DELTA",
        "SCENARIO-REPORT-5732-BLOCKED-MARKER",
        "SCENARIO-REPORT-5732-FIELD-PRINCIPLES",
        str(mod.RESULT_RELATIVE_PATH),
        mod.PLANNER_MARKER,
        mod.INFERENCE_SUBSTRATE,
        "`preconditions_checked`",
        "`benchmark_compute_claimed`",
        "`huggingface_status`",
        "`github_status`",
        "`accepted_findings`",
        "`roadmap_change_required`",
        "`references_updated`",
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_5732_accept_delta_appends_once(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5732-ACCEPT-BOUNDED-DELTA: hooks append once."""

    root = _make_repo(tmp_path, _planner_references(), roadmap_next=True)

    artifact = mod.build_and_write_artifact(
        root=root,
        search_timestamp_utc="2026-07-20T12:00:00Z",
        duration_s=0.25,
    )

    mod.validate_artifact(artifact)
    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")
    result_text = (root / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")

    assert result_text.endswith("\n")
    assert references.count(mod.EXECUTION_REFRESH_HEADING) == 1
    assert "Hard Rules, Soft Preferences" in references
    assert "arXiv:2607.15562" in references
    assert references.index(mod.EXECUTION_REFRESH_HEADING) > references.index(
        mod.PLANNER_HEADING
    )
    assert artifact["planner_marker_found"] is True
    assert len(artifact["accepted_findings"]) == 3
    assert len(artifact["target_experiment_map"]) == 3
    assert artifact["references_updated"] is True
    assert artifact["references_mutated_this_run"] is True
    assert artifact["roadmap_change_required"] is False
    assert artifact["benchmark_compute_claimed"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["roadmap_context"]["source"] == "research-roadmap-next.yaml"
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml_present"] is True
    assert artifact["honest_verdict"].startswith("complete: accepted 3")

    accepted_ids = {row["source_id"] for row in artifact["accepted_findings"]}
    assert accepted_ids == {
        "hard_rules_soft_preferences_2607_15562",
        "presentation_not_mechanism_2607_16019",
        "arc_ewm_ablation_2607_15439",
    }
    assert artifact["accepted_findings"][0]["target_experiments"] == [
        "exp5733-sota-finite-choice-proposal-channel",
        "exp5734-sota-exact-proposal-stream",
    ]
    assert artifact["target_experiment_map"][1]["authority_boundary"].startswith(
        "exact lifecycle"
    )
    assert artifact["target_experiment_map"][2]["falsifiable_metric"]

    duplicate_ids = {row["source_id"] for row in artifact["duplicate_findings"]}
    assert {"gate_zero_growth_2607_14571", "causal_audit_2607_15281"}.issubset(
        duplicate_ids
    )
    watch_ids = {row["source_id"] for row in artifact["watch_only_findings"]}
    assert {"extropic_writing_index", "nhmc_2607_15682"}.issubset(watch_ids)
    inaccessible_ids = {row["source_id"] for row in artifact["inaccessible_findings"]}
    assert {"arxiv_export_api_429", "openreview_browser_challenge"}.issubset(inaccessible_ids)
    excluded_ids = {row["source_id"] for row in artifact["excluded_findings"]}
    assert {"toolsciver_2607_16131", "lazy_arithmetic_2607_15328"}.issubset(excluded_ids)

    artifact_second = mod.build_and_write_artifact(
        root=root,
        search_timestamp_utc="2026-07-20T12:00:01Z",
        duration_s=0.25,
    )
    references_second = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(
        encoding="utf-8"
    )
    assert references_second.count(mod.EXECUTION_REFRESH_HEADING) == 1
    assert artifact_second["references_updated"] is True
    assert artifact_second["references_mutated_this_run"] is False


def test_scenario_report_5732_missing_planner_marker_blocks_append(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5732-BLOCKED-MARKER: missing marker blocks mutation."""

    references_text = "## V511 Planner Refresh - 20260715\nKnown references only.\n"
    root = _make_repo(tmp_path, references_text)

    artifact = mod.build_and_write_artifact(
        root=root,
        search_timestamp_utc="2026-07-20T12:00:00Z",
        duration_s=0.25,
    )

    mod.validate_artifact(artifact)
    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")
    assert references == references_text
    assert artifact["planner_marker_found"] is False
    assert artifact["references_updated"] is False
    assert artifact["accepted_findings"] == []
    assert artifact["roadmap_change_required"] is False
    assert artifact["honest_verdict"].startswith("blocked:")


def test_scenario_report_5732_secondary_statuses_and_boundaries(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5732-ACCEPT-BOUNDED-DELTA: dispositions are explicit."""

    root = _make_repo(tmp_path, _planner_references())
    artifact = mod.build_artifact(
        root=root,
        search_timestamp_utc="2026-07-20T12:00:00Z",
        duration_s=0.25,
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
    assert artifact["huggingface_status"]["api_result_count_for_2026_07_20"] == 3
    assert "2607.16097" in artifact["huggingface_status"]["source_ids"]
    assert (
        artifact["github_status"]["recent_repository_searches"][
            "energy_based_reasoning_constraint_created_after_2026_07_19"
        ]
        == 0
    )
    assert artifact["closed_scope_review"] == mod.closed_scope_review()
    assert any(
        row["source_id"] == "arxiv_cs_ai_recent"
        and row["status"] == "http_200_latest_public_day_2026_07_20_accepted_and_disposed"
        for row in artifact["source_link_checks"]
    )


def test_scenario_report_5732_field_principle_validation_rejects_bad_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5732-FIELD-PRINCIPLES: malformed artifacts fail closed."""

    root = _make_repo(tmp_path, _planner_references())
    artifact = mod.build_artifact(
        root=root,
        search_timestamp_utc="2026-07-20T12:00:00Z",
        duration_s=0.25,
    )
    mod.validate_artifact(artifact)

    broken = dict(artifact)
    broken["field_principles"] = dict(artifact["field_principles"])
    broken["field_principles"].pop("github_status")
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["references_updated"] = "no"
    with pytest.raises(ValueError, match="references_updated"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["benchmark_compute_claimed"] = True
    with pytest.raises(ValueError, match="benchmark"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["search_timestamp_utc"] = "2026-07-20T12:00:00"
    with pytest.raises(ValueError, match="timestamp"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["target_experiment_map"] = [
        {"source_id": "bad", "target_experiments": ["exp5742-v512-capstone-reconciliation"]}
    ]
    with pytest.raises(ValueError, match="target experiment"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["accepted_findings"] = [
        {
            "source_id": "incomplete",
            "target_experiment": "exp5733-sota-finite-choice-proposal-channel",
        }
    ]
    with pytest.raises(ValueError, match="accepted finding"):
        mod.validate_artifact(broken)


def test_scenario_report_5732_defensive_helpers_and_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-REPORT-5732-FIELD-PRINCIPLES: helper boundaries stay explicit."""

    root = _make_repo(tmp_path, _planner_references())
    assert mod.read_text_if_present(root / "missing.md") == ""
    assert mod.relative_path(tmp_path, Path("/outside/root.txt")) == "/outside/root.txt"
    assert mod.normalize_timestamp("2026-07-20T12:00:00+00:00").endswith("Z")
    assert mod.planner_marker_line("missing") is None
    assert mod.honest_verdict(False, []).startswith("blocked:")
    assert mod.honest_verdict(True, []).startswith("complete: no new")
    block = mod.execution_refresh_block(mod.accepted_findings(True))
    assert mod.insert_after_planner_block("no marker", block).endswith(block)
    fallback_insert = mod.insert_after_planner_block(
        f"{mod.PLANNER_HEADING}\nbody",
        block,
    )
    assert fallback_insert.count(mod.EXECUTION_REFRESH_HEADING) == 1
    assert fallback_insert.endswith(block)

    malformed_root = _make_repo(tmp_path / "malformed", _planner_references())
    (malformed_root / mod.ROADMAP_RELATIVE_PATH).write_text("- just\n- a list\n")
    context = mod.roadmap_context(malformed_root)
    assert context["milestone"] == ""
    assert context["task_ids"] == []

    artifact = mod.build_artifact(
        root=root,
        search_timestamp_utc="2026-07-20T12:00:00Z",
        duration_s=0.25,
    )
    tampered = dict(artifact)
    tampered["honest_verdict"] = "complete: tampered"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(tampered)

    blocked = dict(artifact)
    blocked["roadmap_change_required"] = True
    blocked["status"] = "blocked"
    blocked["honest_verdict"] = "blocked: scope expansion requires operator review"
    blocked["reproducibility_checksum"] = mod.payload_checksum(blocked)
    mod.validate_artifact(blocked)

    cli_root = _make_repo(tmp_path / "cli", _planner_references())
    assert (
        mod.main(
            [
                "--root",
                str(cli_root),
                "--search-timestamp-utc",
                "2026-07-20T12:00:01Z",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert mod.RESULT_RELATIVE_PATH.as_posix() in captured.out


def test_scenario_report_5732_module_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-REPORT-5732-FIELD-PRINCIPLES: module entry exits cleanly."""

    root = _make_repo(tmp_path, _planner_references())
    monkeypatch.setattr(
        "sys.argv",
        [
            "experiment_5732_v512_source_delta_ingestion",
            "--root",
            str(root),
            "--search-timestamp-utc",
            "2026-07-20T12:00:02Z",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module(
            "carnot.experiment_5732_v512_source_delta_ingestion",
            run_name="__main__",
        )

    assert exc_info.value.code == 0
    assert mod.RESULT_RELATIVE_PATH.as_posix() in capsys.readouterr().out
