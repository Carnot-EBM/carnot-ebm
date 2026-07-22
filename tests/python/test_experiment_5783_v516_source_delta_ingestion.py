"""Tests for Exp5783 V516 source-delta ingestion.

Spec refs: REQ-REPORT-5783, SCENARIO-REPORT-5783-ZERO-FINDING,
SCENARIO-REPORT-5783-ACCEPT-BOUNDED-DELTA,
SCENARIO-REPORT-5783-BLOCKED-PROVENANCE,
SCENARIO-REPORT-5783-CLOSED-SCOPE-IMMUTABILITY,
SCENARIO-REPORT-5783-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest
import yaml

from carnot import experiment_5783_v516_source_delta_ingestion as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"
START = "2026-07-22T07:39:54Z"
FINISH = "2026-07-22T07:41:21Z"


def _roadmap() -> str:
    tasks = []
    for task_id in mod.ALLOCATED_TARGET_EXPERIMENTS | {"exp5783-v516-source-delta-ingestion"}:
        row = {
            "id": task_id,
            "milestone": mod.MILESTONE,
            "title": task_id,
            "deliverable": f"results/{task_id}.json",
        }
        if task_id in {
            "exp5785-hardness-surface-prospective-fixture",
            "exp5793-arc-live-world-model-ab",
        }:
            row["gated_on"] = [
                {
                    "upstream": "exp5784-evidence-index-terminal-qualification",
                    "artifact_field": "evidence_index_ready_score",
                    "op": "==",
                    "value": 1.0,
                }
            ]
        tasks.append(row)
    tasks.sort(key=lambda row: row["id"])
    return yaml.safe_dump({"milestone": mod.MILESTONE, "tasks": tasks}, sort_keys=False)


def _planner_references() -> str:
    return (
        "## V516 Planner Refresh - 20260722\n\n"
        "- **Solver-Hard Is Not Model-Hard: A Hardness-Controlled Diagnostic "
        "for LLM Constraint Reasoning** - arXiv:2607.17047.\n"
        "- **Verifiable Self-Evolution for Open-Ended Dialogue Skills via "
        "Future-Feedback Prediction** - arXiv:2607.18973.\n"
        "- Extropic and Logical Intelligence public updates remain context only.\n"
        "<!-- V516-PLANNER-REFRESH-20260722-END -->\n"
    )


def _accepted_fixture() -> mod.JsonDict:
    return {
        "source_id": "post_marker_fixture_control_2607_99999",
        "classification": "accepted",
        "title": "Post-Marker Fixture Control for Exact Evidence Qualification",
        "url": "https://arxiv.org/abs/2607.99999",
        "publication_date": "2026-07-22",
        "source_date": "2026-07-22",
        "search_timestamp": START,
        "search_receipt": "arxiv_fixture_post_marker_control",
        "target_experiment": "exp5784-evidence-index-terminal-qualification",
        "authority_boundary": "Adds a receipt-control idea to the existing evidence-index qualification task only.",
        "carnot_hook": "Require final artifact replay after tests before the evidence index can gate downstream work.",
        "falsifiable_metric": "terminal_artifact_reopened_after_write is true before readiness is consumed",
        "post_marker_or_newly_actionable": True,
        "newly_actionable_after_marker": True,
        "reason": "Fixture accepted finding stays inside Exp5784 and does not change IDs, gates, models, hardware claims, headline claims, or retired scopes.",
    }


def _ordered_candidates(artifact: mod.JsonDict) -> list[mod.JsonDict]:
    return (
        artifact["accepted_findings"]
        + artifact["duplicate_findings"]
        + artifact["watch_only_findings"]
        + artifact["excluded_findings"]
        + artifact["inaccessible_findings"]
    )


def _make_repo(root: Path, references_text: str, *, with_next: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for relative in (
        "AGENTS.md",
        "CODEX.md",
        "CLAUDE.md",
        "research-program.md",
        "research-complete.yaml",
    ):
        (root / relative).write_text("fixture\n", encoding="utf-8")
    (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).write_text(
        references_text,
        encoding="utf-8",
    )
    (root / mod.ROADMAP_RELATIVE_PATH).write_text(_roadmap(), encoding="utf-8")
    if with_next:
        (root / mod.ROADMAP_NEXT_RELATIVE_PATH).write_text(_roadmap(), encoding="utf-8")
    (root / "openspec/change-proposals").mkdir(parents=True, exist_ok=True)
    (root / mod.VNEXT_RELATIVE_PATH).write_text(
        "# Research Roadmap vNEXT\n\n"
        "**Milestone:** `2026.07.516`\n\n"
        "Exp5783 source-delta ingestion preserves V516 ids and gates.\n",
        encoding="utf-8",
    )
    (root / "openspec/capabilities/research-reporting").mkdir(parents=True, exist_ok=True)
    (root / mod.SPEC_RELATIVE_PATH).write_text(
        "\n".join(mod.SPEC_REFS) + "\n",
        encoding="utf-8",
    )
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / mod.EXCLUSION_MANIFEST_RELATIVE_PATH).write_text(
        (
            "retired_extras:\n"
            "- id: cegis_closed\n"
            "  reason: CEGIS remains closed\n"
            "- id: generated_text_scoring_closed\n"
            "  reason: PHASE-D generated-text scoring remains closed\n"
            "- id: kan_scaling_closed\n"
            "  reason: KAN scaling remains closed\n"
            "- id: allocation_free_10x_closed\n"
            "  reason: allocation-free 10x remains closed\n"
            "- id: public_arc_solves_closed\n"
            "  reason: public ARC solves remain closed\n"
            "- id: tsu_kona_execution_closed\n"
            "  reason: TSU and Kona execution need authenticated local receipts\n"
        ),
        encoding="utf-8",
    )
    (root / mod.KNOWN_ISSUES_RELATIVE_PATH).write_text(
        "CEGIS, generated-text scoring, KAN scaling, allocation-free 10x, "
        "public ARC solves, TSU execution, and Kona execution remain closed.\n",
        encoding="utf-8",
    )
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / mod.CONDUCTOR_RELATIVE_PATH).write_text("# conductor fixture\n", encoding="utf-8")
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / mod.PRIOR_SOURCE_DELTA_RELATIVE_PATH).write_text(
        '{"status": "complete"}\n',
        encoding="utf-8",
    )
    return root


def test_req_report_5783_spec_declares_post_marker_contract() -> None:
    """REQ-REPORT-5783: OpenSpec anchors V516 source-delta fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-5783") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-REPORT-5783",
        "SCENARIO-REPORT-5783-ZERO-FINDING",
        "SCENARIO-REPORT-5783-ACCEPT-BOUNDED-DELTA",
        "SCENARIO-REPORT-5783-BLOCKED-PROVENANCE",
        "SCENARIO-REPORT-5783-CLOSED-SCOPE-IMMUTABILITY",
        "SCENARIO-REPORT-5783-FIELD-PRINCIPLES",
        str(mod.RESULT_RELATIVE_PATH),
        mod.PLANNER_MARKER,
        mod.INFERENCE_SUBSTRATE,
        "`primary_source_receipts`",
        "`secondary_source_receipts`",
        "`semantic_scholar_citation_receipts`",
        "`roadmap_ids_unchanged`",
        "`gates_unchanged`",
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_5783_zero_finding_keeps_references_unchanged(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5783-ZERO-FINDING: zero accepted findings are terminal."""

    root = _make_repo(tmp_path, _planner_references())

    artifact = mod.build_and_write_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_findings=[],
        test_commands=["unit"],
        test_exit_codes={"unit": 0},
    )

    mod.validate_artifact(artifact)
    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")
    result_text = (root / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")

    assert result_text.endswith("\n")
    assert mod.EXECUTION_REFRESH_HEADING not in references
    assert artifact["status"] == "complete"
    assert artifact["accepted_findings"] == []
    assert artifact["accepted_finding_count"] == 0
    assert artifact["references_modified"] is False
    assert artifact["search_window"]["boundary_marker"] == mod.PLANNER_MARKER
    assert artifact["preconditions_checked"]["network_search_available"] is True
    assert artifact["preconditions_checked"]["research_roadmap_next_read"] is False
    assert artifact["roadmap_ids_unchanged"] is True
    assert artifact["gates_unchanged"] is True
    assert artifact["closed_scopes_reopened"] is False
    assert artifact["hardware_claim_changed"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete: no accepted")


def test_scenario_report_5783_accept_delta_appends_once(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5783-ACCEPT-BOUNDED-DELTA: accepted controls append once."""

    root = _make_repo(tmp_path, _planner_references(), with_next=True)

    artifact = mod.build_and_write_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_findings=[_accepted_fixture()],
    )

    mod.validate_artifact(artifact)
    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")

    assert references.count(mod.EXECUTION_REFRESH_HEADING) == 1
    assert "Post-Marker Fixture Control" in references
    assert "exp5784-evidence-index-terminal-qualification" in references
    assert references.index(mod.EXECUTION_REFRESH_HEADING) > references.index(
        mod.PLANNER_HEADING
    )
    assert artifact["accepted_finding_count"] == 1
    assert artifact["accepted_findings"][0]["target_experiment"] == (
        "exp5784-evidence-index-terminal-qualification"
    )
    assert artifact["references_modified"] is True
    assert artifact["roadmap_ids_unchanged"] is True
    assert artifact["gates_unchanged"] is True
    assert artifact["honest_verdict"].startswith("complete: accepted 1")

    artifact_second = mod.build_and_write_artifact(
        root=root,
        search_started_at="2026-07-22T07:42:00Z",
        search_finished_at="2026-07-22T07:42:01Z",
        accepted_findings=[_accepted_fixture()],
    )
    references_second = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(
        encoding="utf-8"
    )
    assert references_second.count(mod.EXECUTION_REFRESH_HEADING) == 1
    assert artifact_second["references_modified"] is False


def test_scenario_report_5783_blocked_provenance_paths(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5783-BLOCKED-PROVENANCE: marker and reachability block."""

    references_text = "## V515 Planner Refresh - 20260721\nKnown references only.\n"
    root = _make_repo(tmp_path, references_text)

    artifact = mod.build_and_write_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_findings=[_accepted_fixture()],
    )

    mod.validate_artifact(artifact)
    assert (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(
        encoding="utf-8"
    ) == references_text
    assert artifact["status"] == "blocked"
    assert artifact["preconditions_checked"]["planner_marker_found"] is False
    assert artifact["accepted_findings"] == []
    assert artifact["accepted_finding_count"] == 0
    assert artifact["references_modified"] is False
    assert artifact["honest_verdict"].startswith("blocked:")

    reachable_root = _make_repo(tmp_path / "reachable", _planner_references())
    unreachable_receipts = [
        {
            "receipt_id": "arxiv_down",
            "surface": "arXiv",
            "url": "https://arxiv.org/",
            "queried_at": START,
            "status": "inaccessible_timeout",
            "candidate_ids": [],
            "receipt_summary": "fixture outage",
        }
    ]
    unreachable = mod.build_artifact(
        root=reachable_root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_findings=[],
        primary_source_receipts=unreachable_receipts,
        secondary_source_receipts=[],
        semantic_scholar_citation_receipts=[],
    )
    mod.validate_artifact(unreachable)
    assert unreachable["status"] == "blocked"
    assert "source_reachability_failed" in unreachable["preconditions_checked"][
        "failed_preconditions"
    ]


def test_scenario_report_5783_closed_scope_and_roadmap_guards(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5783-CLOSED-SCOPE-IMMUTABILITY: claim boundaries are strict."""

    root = _make_repo(tmp_path, _planner_references())
    artifact = mod.build_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_findings=[_accepted_fixture()],
    )

    mod.validate_artifact(artifact)
    surfaces = {
        row["surface"]
        for row in (
            artifact["primary_source_receipts"]
            + artifact["secondary_source_receipts"]
            + artifact["semantic_scholar_citation_receipts"]
        )
    }
    assert {
        "arXiv",
        "OpenReview",
        "Hugging Face Papers",
        "GitHub discovery",
        "Extropic writing",
        "Logical Intelligence",
        "Semantic Scholar",
    }.issubset(surfaces)
    assert artifact["duplicate_findings"][0]["classification"] == "duplicate"
    assert artifact["watch_only_findings"][0]["classification"] == "watch_only"
    assert artifact["excluded_findings"][0]["classification"] == "excluded"
    assert artifact["candidate_findings"] == (
        artifact["accepted_findings"]
        + artifact["duplicate_findings"]
        + artifact["watch_only_findings"]
        + artifact["excluded_findings"]
        + artifact["inaccessible_findings"]
    )
    assert "CEGIS" in artifact["closed_scope_review"]["protected_scopes"][0]
    assert artifact["target_experiment_map"][0]["target_experiment"].startswith("exp5784")

    broken = dict(artifact)
    broken["accepted_findings"] = [
        {
            **_accepted_fixture(),
            "target_experiment": "exp5795-v516-capstone-reconciliation",
        }
    ]
    with pytest.raises(ValueError, match="accepted finding"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["accepted_findings"] = [
        {
            **_accepted_fixture(),
            "post_marker_or_newly_actionable": False,
        }
    ]
    with pytest.raises(ValueError, match="post-marker"):
        mod.validate_artifact(broken)

    for field, message in (
        ("roadmap_ids_unchanged", "roadmap ids"),
        ("gates_unchanged", "gates"),
        ("closed_scopes_reopened", "closed scopes"),
        ("hardware_claim_changed", "hardware"),
    ):
        broken = dict(artifact)
        broken[field] = False if field.endswith("unchanged") else True
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(broken)


def test_scenario_report_5783_field_principles_helpers_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-REPORT-5783-FIELD-PRINCIPLES: schema, helpers, and CLI stay stable."""

    root = _make_repo(tmp_path, _planner_references())
    assert mod.read_text_if_present(root / "missing.md") == ""
    assert mod.path_sha256(root / "missing.md") is None
    assert mod.normalize_timestamp("2026-07-22T07:39:54+00:00").endswith("Z")
    assert mod.planner_marker_line("missing") is None
    assert mod.planner_block_hash("missing") is None
    assert mod.target_experiment_map([_accepted_fixture()])[0]["source_id"] == (
        "post_marker_fixture_control_2607_99999"
    )
    assert mod.honest_verdict(False, True, [], False).startswith("blocked:")
    assert mod.honest_verdict(True, False, [], False).startswith("blocked:")
    assert mod.honest_verdict(True, True, [], True).startswith("blocked:")
    assert mod.honest_verdict(True, True, [], False).startswith("complete: no accepted")
    block = mod.execution_refresh_block([_accepted_fixture()])
    assert mod.insert_after_planner_block("no marker", block).endswith(block)
    assert mod.insert_after_planner_block(f"prefix\n{mod.EXECUTION_REFRESH_HEADING}\n", block) == (
        f"prefix\n{mod.EXECUTION_REFRESH_HEADING}\n"
    )
    fallback_insert = mod.insert_after_planner_block(f"{mod.PLANNER_HEADING}\nbody", block)
    assert fallback_insert.count(mod.EXECUTION_REFRESH_HEADING) == 1

    malformed_root = _make_repo(tmp_path / "malformed", _planner_references())
    (malformed_root / mod.ROADMAP_RELATIVE_PATH).write_text("- just\n- a list\n")
    preconditions = mod.preconditions_checked(
        malformed_root,
        marker_found=True,
        source_reachable=True,
    )
    assert preconditions["roadmap_ids_hash"] is None
    (malformed_root / mod.ROADMAP_RELATIVE_PATH).write_text(
        "tasks: not-a-list\nmilestone: fixture\n",
        encoding="utf-8",
    )
    assert mod.preconditions_checked(
        malformed_root,
        marker_found=True,
        source_reachable=True,
    )["active_roadmap_milestone"] == "fixture"
    (malformed_root / mod.ROADMAP_RELATIVE_PATH).write_text(
        "tasks: [\n",
        encoding="utf-8",
    )
    assert mod.preconditions_checked(
        malformed_root,
        marker_found=True,
        source_reachable=True,
    )["roadmap_ids_hash"] is None
    (malformed_root / mod.SPEC_RELATIVE_PATH).write_text("missing\n", encoding="utf-8")
    assert "spec_req_report_5783_missing" in mod.preconditions_checked(
        malformed_root,
        marker_found=True,
        source_reachable=True,
    )["failed_preconditions"]
    (malformed_root / mod.ROADMAP_RELATIVE_PATH).unlink()
    (malformed_root / mod.EXCLUSION_MANIFEST_RELATIVE_PATH).unlink()
    failed_without_hashes = mod.preconditions_checked(
        malformed_root,
        marker_found=True,
        source_reachable=True,
    )["failed_preconditions"]
    assert "active_roadmap_hash_missing" in failed_without_hashes
    assert "exclusion_manifest_hash_missing" in failed_without_hashes

    artifact = mod.build_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
    )
    mod.validate_artifact(artifact)
    broken = dict(artifact)
    broken["field_principles"] = dict(artifact["field_principles"])
    broken["field_principles"].pop("primary_source_receipts")
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["search_finished_at"] = START
    with pytest.raises(ValueError, match="timestamp"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["honest_verdict"] = "complete: tampered"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(broken)

    tests_run_path = tmp_path / "tests_run.json"
    tests_run_path.write_text(
        json.dumps([{"command": "unit", "exit_code": 0}]),
        encoding="utf-8",
    )
    assert (
        mod.main(
            [
                "--root",
                str(root),
                "--search-started-at",
                START,
                "--search-finished-at",
                FINISH,
                "--zero-findings",
                "--tests-run-json",
                str(tests_run_path),
            ]
        )
        == 0
    )
    assert mod.RESULT_RELATIVE_PATH.as_posix() in capsys.readouterr().out

    monkeypatch.setattr(
        "sys.argv",
        [
            "experiment_5783_v516_source_delta_ingestion",
            "--root",
            str(root),
            "--search-started-at",
            START,
            "--search-finished-at",
            FINISH,
            "--zero-findings",
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module(
            "carnot.experiment_5783_v516_source_delta_ingestion",
            run_name="__main__",
        )
    assert exc_info.value.code == 0
    assert mod.RESULT_RELATIVE_PATH.as_posix() in capsys.readouterr().out


def test_scenario_report_5783_validator_rejects_schema_and_provenance_errors(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-5783: validation rejects schema drift and unsupported candidates."""

    root = _make_repo(tmp_path, _planner_references())
    artifact = mod.build_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
    )

    missing = dict(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(missing)

    broken = dict(artifact)
    broken["field_principles"] = "not-a-map"
    with pytest.raises(ValueError, match="field_principles must be a mapping"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["status"] = "done"
    with pytest.raises(ValueError, match="invalid status"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["honest_verdict"] = "unsupported verdict"
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["actual_search_wall_time_s"] = -1
    with pytest.raises(ValueError, match="wall time"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["accepted_finding_count"] = 1
    with pytest.raises(ValueError, match="accepted_finding_count"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["references_modified"] = True
    with pytest.raises(ValueError, match="zero accepted"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["candidate_findings"] = artifact["candidate_findings"][:-1]
    with pytest.raises(ValueError, match="candidate_findings"):
        mod.validate_artifact(broken)

    broken = json.loads(json.dumps(artifact))
    broken["duplicate_findings"][0]["classification"] = "novel"
    broken["candidate_findings"] = _ordered_candidates(broken)
    with pytest.raises(ValueError, match="invalid candidate classification"):
        mod.validate_artifact(broken)

    broken = json.loads(json.dumps(artifact))
    broken["watch_only_findings"][0]["url"] = ""
    broken["candidate_findings"] = _ordered_candidates(broken)
    with pytest.raises(ValueError, match="provenance field"):
        mod.validate_artifact(broken)

    broken = json.loads(json.dumps(artifact))
    broken["excluded_findings"][0].pop("publication_date", None)
    broken["excluded_findings"][0].pop("source_date", None)
    broken["candidate_findings"] = _ordered_candidates(broken)
    with pytest.raises(ValueError, match="publication/source date"):
        mod.validate_artifact(broken)

    for receipt_field, message in (
        ("primary_source_receipts", "primary source receipts"),
        ("secondary_source_receipts", "secondary source receipts"),
        ("semantic_scholar_citation_receipts", "semantic scholar citation receipts"),
    ):
        broken = dict(artifact)
        broken[receipt_field] = []
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(broken)
