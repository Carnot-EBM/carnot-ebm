"""Tests for Exp5770 V515 source-delta ingestion.

Spec refs: REQ-REPORT-5770, SCENARIO-REPORT-5770-ZERO-FINDING,
SCENARIO-REPORT-5770-ACCEPT-BOUNDED-DELTA,
SCENARIO-REPORT-5770-BLOCKED-PROVENANCE,
SCENARIO-REPORT-5770-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest
import yaml

from carnot import experiment_5770_v515_source_delta_ingestion as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"
START = "2026-07-22T01:00:00Z"
FINISH = "2026-07-22T01:03:00Z"


def _roadmap() -> str:
    tasks = [
        {
            "id": task_id,
            "milestone": mod.MILESTONE,
            "title": task_id,
            "deliverable": f"results/{task_id}.json",
        }
        for task_id in (
            "exp5769-transition-v515",
            "exp5770-v515-source-delta-ingestion",
            "exp5771-evidence-index-collision-preflight",
            "exp5772-sota-constraint-drift-stream",
            "exp5773-prospective-constraint-acquisition-ab",
            "exp5774-constraint-transfer-forgetting-audit",
            "exp5775-disabled-online-shadow-integration",
            "exp5776-arc-world-model-admission-contract",
            "exp5777-arc-world-model-family-panel",
            "exp5778-arc-world-model-selector-audit",
            "exp5779-arc-heldout-live-e3-ab",
            "exp5780-v515-capstone-precheck",
            "exp5781-v515-capstone-reconciliation",
        )
    ]
    return yaml.safe_dump({"milestone": mod.MILESTONE, "tasks": tasks}, sort_keys=False)


def _planner_references() -> str:
    return (
        "## V515 Planner Refresh - 20260721\n\n"
        "- **Validate the Dream Before You Trust Its Verdict: Admissibility for "
        "World-Model Simulators** - arXiv:2607.07196.\n"
        "- **Residual Drift Dominates Contradiction in Multi-Turn Constraint "
        "Reasoning** - arXiv:2605.23940.\n"
        "<!-- V515-PLANNER-REFRESH-20260721-END -->\n"
    )


def _accepted_fixture() -> mod.JsonDict:
    return {
        "source_id": "verified_world_model_play_adequacy_2607_14169",
        "classification": "accepted",
        "title": "When a Verified World Model Still Loses: Play-Adequacy vs Prediction-Accuracy in LLM-Synthesized Code World Models",
        "url": "https://arxiv.org/abs/2607.14169",
        "publication_date": "2026-07-15",
        "version_date": "2026-07-19",
        "search_timestamp": START,
        "search_receipt": "arxiv_abs_2607_14169",
        "target_experiment": "exp5776-arc-world-model-admission-contract",
        "authority_boundary": "agent-owned play adequacy and pivotal-transition coverage remain required before simulated rollouts can influence E3",
        "carnot_hook": "Add a play-adequacy control that rejects high transition accuracy when pivotal dynamics are missed.",
        "falsifiable_metric": "heldout_pivotal_transition_miss_rate is zero before any world-model policy influence",
        "post_marker_or_newly_actionable": True,
        "newly_actionable_after_marker": True,
        "reason": "The source sharpens the existing V515 ARC world-model validation boundary without changing task IDs, gates, dependencies, models, hardware claims, or headline claims.",
    }


def _make_repo(root: Path, references_text: str, *, with_next: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for relative in (
        "AGENTS.md",
        "CODEX.md",
        "CLAUDE.md",
        "research-program.md",
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
        "**Milestone:** `2026.07.515`\n\n"
        "Exp5770 source-delta ingestion preserves V515 ids and gates.\n",
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
            "- id: kan_scaleup_closed\n"
            "  reason: V515 preserves KAN scale-up retirement\n"
            "- id: public_arc_solving_closed\n"
            "  reason: public ARC solving remains outside Exp5770\n"
            "- id: unsupported_hardware_claims_closed\n"
            "  reason: no authenticated hardware execution\n"
        ),
        encoding="utf-8",
    )
    (root / mod.KNOWN_ISSUES_RELATIVE_PATH).write_text(
        "Retired proposal scoring, KAN scale-up, CEGIS, PHASE-D text scoring, "
        "Rust 10x, public ARC solving, and unsupported hardware claims remain closed.\n",
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


def test_req_report_5770_spec_declares_family_receipt_contract() -> None:
    """REQ-REPORT-5770: OpenSpec anchors V515 source-delta fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-5770") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-REPORT-5770",
        "SCENARIO-REPORT-5770-ZERO-FINDING",
        "SCENARIO-REPORT-5770-ACCEPT-BOUNDED-DELTA",
        "SCENARIO-REPORT-5770-BLOCKED-PROVENANCE",
        "SCENARIO-REPORT-5770-FIELD-PRINCIPLES",
        str(mod.RESULT_RELATIVE_PATH),
        mod.PLANNER_MARKER,
        mod.INFERENCE_SUBSTRATE,
        "`arxiv_receipts`",
        "`openreview_receipts`",
        "`huggingface_receipts`",
        "`github_receipts`",
        "`logical_intelligence_receipts`",
        "`operator_review_required`",
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_5770_zero_finding_completes_without_reference_append(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5770-ZERO-FINDING: zero accepted findings are terminal."""

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
    assert artifact["references_changed"] is False
    assert artifact["actual_search_wall_time_s"] == pytest.approx(180.0)
    assert artifact["roadmap_scope_change_requested"] is False
    assert artifact["operator_review_required"] is False
    assert artifact["closed_scopes_reopened"] is False
    assert artifact["hardware_claim_changed"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["planner_marker_hash"].startswith("sha256:")
    assert artifact["preconditions_checked"]["network_source_reachability_established"] is True
    assert artifact["preconditions_checked"]["research_roadmap_next_read"] is False
    assert artifact["honest_verdict"].startswith("complete: no new")


def test_scenario_report_5770_accept_delta_appends_once(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5770-ACCEPT-BOUNDED-DELTA: accepted controls append once."""

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
    assert "When a Verified World Model Still Loses" in references
    assert "exp5776-arc-world-model-admission-contract" in references
    assert references.index(mod.EXECUTION_REFRESH_HEADING) > references.index(
        mod.PLANNER_HEADING
    )
    assert artifact["accepted_findings"][0]["newly_actionable_after_marker"] is True
    assert artifact["accepted_findings"][0]["target_experiment"] == (
        "exp5776-arc-world-model-admission-contract"
    )
    assert artifact["references_changed"] is True
    assert artifact["honest_verdict"].startswith("complete: accepted 1")

    artifact_second = mod.build_and_write_artifact(
        root=root,
        search_started_at="2026-07-22T01:04:00Z",
        search_finished_at="2026-07-22T01:04:01Z",
        accepted_findings=[_accepted_fixture()],
    )
    references_second = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(
        encoding="utf-8"
    )
    assert references_second.count(mod.EXECUTION_REFRESH_HEADING) == 1
    assert artifact_second["references_changed"] is False


def test_scenario_report_5770_blocked_provenance_paths(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5770-BLOCKED-PROVENANCE: marker and reachability block."""

    references_text = "## V514 Planner Refresh - 20260720\nKnown references only.\n"
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
    assert artifact["references_changed"] is False
    assert artifact["operator_review_required"] is False
    assert artifact["honest_verdict"].startswith("blocked:")

    reachable_root = _make_repo(tmp_path / "reachable", _planner_references())
    unreachable_receipts = [
        {
            "receipt_id": "all_sources_down",
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
        source_receipts=unreachable_receipts,
    )
    mod.validate_artifact(unreachable)
    assert unreachable["status"] == "blocked"
    assert "source_reachability_failed" in unreachable["preconditions_checked"][
        "failed_preconditions"
    ]


def test_scenario_report_5770_field_principles_and_boundaries(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5770-FIELD-PRINCIPLES: schema guards claim boundaries."""

    root = _make_repo(tmp_path, _planner_references())
    artifact = mod.build_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_findings=[_accepted_fixture()],
    )

    mod.validate_artifact(artifact)
    surfaces = {row["surface"] for row in artifact["source_receipts"]}
    assert {
        "arXiv",
        "OpenReview",
        "Semantic Scholar",
        "Hugging Face Papers",
        "GitHub discovery",
        "Extropic writing",
        "Logical Intelligence",
        "local Carnot ledgers",
    }.issubset(surfaces)
    assert artifact["arxiv_receipts"][0]["receipt_id"].startswith("arxiv_")
    assert artifact["openreview_receipts"][0]["status"].startswith("inaccessible_")
    assert artifact["semantic_scholar_receipts"][0]["paper"] == "arXiv:2507.02092"
    assert artifact["huggingface_receipts"][0]["surface"] == "Hugging Face Papers"
    assert artifact["github_receipts"][0]["surface"] == "GitHub discovery"
    assert artifact["extropic_receipts"][0]["surface"] == "Extropic writing"
    assert artifact["logical_intelligence_receipts"][0]["surface"] == (
        "Logical Intelligence"
    )
    assert artifact["duplicate_findings"][0]["classification"] == "duplicate"
    assert artifact["watch_only_findings"][0]["classification"] == "watch_only"
    assert artifact["excluded_findings"][0]["classification"] == "excluded"
    assert artifact["inaccessible_findings"][0]["classification"] == "inaccessible"

    broken = dict(artifact)
    broken["field_principles"] = dict(artifact["field_principles"])
    broken["field_principles"].pop("arxiv_receipts")
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["hardware_claim_changed"] = True
    with pytest.raises(ValueError, match="hardware"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["closed_scopes_reopened"] = True
    with pytest.raises(ValueError, match="closed scopes"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["search_finished_at"] = START
    with pytest.raises(ValueError, match="timestamp"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["actual_search_wall_time_s"] = 1.0
    with pytest.raises(ValueError, match="wall time"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["accepted_findings"] = [
        {
            **_accepted_fixture(),
            "target_experiment": "exp5771-evidence-index-collision-preflight",
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

    review_needed = mod.build_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_findings=[
            {
                **_accepted_fixture(),
                "gate_change_requested": True,
            }
        ],
    )
    mod.validate_artifact(review_needed)
    assert review_needed["status"] == "blocked"
    assert review_needed["operator_review_required"] is True
    assert review_needed["roadmap_scope_change_requested"] is False


def test_scenario_report_5770_helpers_cli_and_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-REPORT-5770-FIELD-PRINCIPLES: helpers and CLI stay stable."""

    root = _make_repo(tmp_path, _planner_references())
    assert mod.read_text_if_present(root / "missing.md") == ""
    assert mod.path_sha256(root / "missing.md") is None
    assert mod.normalize_timestamp("2026-07-22T01:00:00+00:00").endswith("Z")
    assert mod.planner_marker_line("missing") is None
    assert mod.planner_block_hash("missing") is None
    assert mod.target_experiment_map([_accepted_fixture()])[0]["source_id"] == (
        "verified_world_model_play_adequacy_2607_14169"
    )
    assert mod.honest_verdict(False, True, [], False).startswith("blocked:")
    assert mod.honest_verdict(True, False, [], False).startswith("blocked:")
    assert mod.honest_verdict(True, True, [], False).startswith("complete: no new")
    block = mod.execution_refresh_block([_accepted_fixture()])
    assert mod.insert_after_planner_block("no marker", block).endswith(block)
    fallback_insert = mod.insert_after_planner_block(f"{mod.PLANNER_HEADING}\nbody", block)
    assert fallback_insert.count(mod.EXECUTION_REFRESH_HEADING) == 1

    malformed_root = _make_repo(tmp_path / "malformed", _planner_references())
    (malformed_root / mod.ROADMAP_RELATIVE_PATH).write_text("- just\n- a list\n")
    preconditions = mod.preconditions_checked(
        malformed_root,
        marker_found=True,
        source_reachable=True,
    )
    assert preconditions["active_roadmap_milestone"] == ""
    (malformed_root / mod.SPEC_RELATIVE_PATH).write_text("missing\n", encoding="utf-8")
    assert "spec_req_report_5770_missing" in mod.preconditions_checked(
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
    tampered = dict(artifact)
    tampered["honest_verdict"] = "complete: tampered"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(tampered)

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
            "experiment_5770_v515_source_delta_ingestion",
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
            "carnot.experiment_5770_v515_source_delta_ingestion",
            run_name="__main__",
        )
    assert exc_info.value.code == 0
    assert mod.RESULT_RELATIVE_PATH.as_posix() in capsys.readouterr().out
