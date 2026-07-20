"""Tests for Exp5756 V514 source-delta ingestion.

Spec refs: REQ-REPORT-5756, SCENARIO-REPORT-5756-ZERO-FINDING,
SCENARIO-REPORT-5756-ACCEPT-BOUNDED-DELTA,
SCENARIO-REPORT-5756-BLOCKED-PROVENANCE,
SCENARIO-REPORT-5756-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest
import yaml

from carnot import experiment_5756_v514_source_delta_ingestion as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"
START = "2026-07-20T14:10:54Z"
FINISH = "2026-07-20T14:12:26Z"


def _roadmap() -> str:
    tasks = [
        {
            "id": task_id,
            "milestone": mod.MILESTONE,
            "title": task_id,
            "deliverable": f"results/{task_id}.json",
        }
        for task_id in (
            "exp5755-transition-v514",
            "exp5756-v514-source-delta-ingestion",
            "exp5757-proposal-benchmark-scalar-bridge",
            "exp5758-rust-parity-scalar-bridge",
            "exp5759-sota-exact-proposal-utility-panel",
            "exp5760-selective-exact-feedback-search",
            "exp5761-exact-constraint-acquisition-benchmark",
            "exp5762-query-driven-constraint-lifecycle",
            "exp5763-dependent-task-constraint-acquisition",
            "exp5764-one-axis-profiled-allocation-free-hot-path",
            "exp5765-one-axis-final-10x-crossover",
            "exp5766-arc-loo-component-interaction-audit",
            "exp5767-arc-game-blind-composition-hardening",
            "exp5768-v514-capstone-reconciliation",
        )
    ]
    return yaml.safe_dump({"milestone": mod.MILESTONE, "tasks": tasks}, sort_keys=False)


def _planner_references() -> str:
    return (
        "## V514 Planner Refresh - 20260720\n\n"
        "- **Overcoming Over-Fitting in Constraint Acquisition via Query-Driven "
        "Interactive Refinement** - arXiv:2509.24489.\n"
        "- **Constraint acquisition needs better benchmarks / MPMMine** - "
        "arXiv:2605.26279.\n"
        "<!-- V514-PLANNER-REFRESH-20260720-END -->\n"
    )


def _accepted_fixture() -> mod.JsonDict:
    return {
        "source_id": "post_v514_control_fixture_2607_19999",
        "classification": "accepted",
        "title": "Post-V514 Control Fixture for Exact Validator Boundaries",
        "url": "https://arxiv.org/abs/2607.19999",
        "publication_date": "2026-07-20",
        "search_receipt": "arxiv_fixture_query",
        "target_experiment": "exp5762-query-driven-constraint-lifecycle",
        "authority_boundary": "exact solver membership queries admit lifecycle updates",
        "carnot_hook": "Add a bounded contradictory-query control to Exp5762 validation.",
        "falsifiable_metric": "heldout_constraint_recovery_lcb remains positive with zero unsafe updates",
        "reason": "Fixture source changes only an existing V514 validation boundary.",
    }


def _make_repo(root: Path, references_text: str) -> Path:
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
    (root / "openspec/change-proposals").mkdir(parents=True, exist_ok=True)
    (root / mod.VNEXT_RELATIVE_PATH).write_text(
        "# Research Roadmap vNEXT\n\n**Milestone:** `2026.07.514`\n",
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
            "- id: unauthenticated_hardware_claims_closed\n"
            "  reason: no local timing receipt\n"
            "- id: broad_rl_scope_closed\n"
            "  reason: no V514 reopening\n"
        ),
        encoding="utf-8",
    )
    (root / mod.KNOWN_ISSUES_RELATIVE_PATH).write_text(
        "Free-form answer repair, LLM judges, model-weight writes, broad RL, "
        "per-game ARC adapters, and unauthenticated hardware claims remain closed.\n",
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


def test_req_report_5756_spec_declares_query_receipt_contract() -> None:
    """REQ-REPORT-5756: OpenSpec anchors V514 source-delta fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-5756") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-REPORT-5756",
        "SCENARIO-REPORT-5756-ZERO-FINDING",
        "SCENARIO-REPORT-5756-ACCEPT-BOUNDED-DELTA",
        "SCENARIO-REPORT-5756-BLOCKED-PROVENANCE",
        "SCENARIO-REPORT-5756-FIELD-PRINCIPLES",
        str(mod.RESULT_RELATIVE_PATH),
        mod.PLANNER_MARKER,
        mod.INFERENCE_SUBSTRATE,
        "`source_queries`",
        "`source_receipts`",
        "`semantic_scholar_receipts`",
        "`operator_review_required`",
        "`hardware_claim_changed`",
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_5756_zero_finding_completes_without_reference_append(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5756-ZERO-FINDING: zero accepted findings are terminal."""

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
    assert artifact["actual_search_wall_time_s"] == pytest.approx(92.0)
    assert artifact["roadmap_scope_change_requested"] is False
    assert artifact["operator_review_required"] is False
    assert artifact["closed_scopes_reopened"] is False
    assert artifact["hardware_claim_changed"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["planner_marker_hash"].startswith("sha256:")
    assert artifact["preconditions_checked"]["network_source_reachability_established"] is True
    assert artifact["honest_verdict"].startswith("complete: no new")


def test_scenario_report_5756_accept_delta_appends_once(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5756-ACCEPT-BOUNDED-DELTA: accepted controls append once."""

    root = _make_repo(tmp_path, _planner_references())

    artifact = mod.build_and_write_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_findings=[_accepted_fixture()],
    )

    mod.validate_artifact(artifact)
    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")

    assert references.count(mod.EXECUTION_REFRESH_HEADING) == 1
    assert "Post-V514 Control Fixture" in references
    assert "exp5762-query-driven-constraint-lifecycle" in references
    assert references.index(mod.EXECUTION_REFRESH_HEADING) > references.index(
        mod.PLANNER_HEADING
    )
    assert artifact["accepted_findings"][0]["target_experiment"] == (
        "exp5762-query-driven-constraint-lifecycle"
    )
    assert artifact["references_changed"] is True
    assert artifact["honest_verdict"].startswith("complete: accepted 1")

    artifact_second = mod.build_and_write_artifact(
        root=root,
        search_started_at="2026-07-20T14:12:30Z",
        search_finished_at="2026-07-20T14:12:31Z",
        accepted_findings=[_accepted_fixture()],
    )
    references_second = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(
        encoding="utf-8"
    )
    assert references_second.count(mod.EXECUTION_REFRESH_HEADING) == 1
    assert artifact_second["references_changed"] is False


def test_scenario_report_5756_missing_marker_blocks_append(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5756-BLOCKED-PROVENANCE: missing marker blocks mutation."""

    references_text = "## V513 Planner Refresh - 20260720\nKnown references only.\n"
    root = _make_repo(tmp_path, references_text)

    artifact = mod.build_and_write_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_findings=[_accepted_fixture()],
    )

    mod.validate_artifact(artifact)
    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")
    assert references == references_text
    assert artifact["status"] == "blocked"
    assert artifact["preconditions_checked"]["planner_marker_found"] is False
    assert artifact["accepted_findings"] == []
    assert artifact["references_changed"] is False
    assert artifact["operator_review_required"] is False
    assert artifact["honest_verdict"].startswith("blocked:")


def test_scenario_report_5756_source_reachability_block(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5756-BLOCKED-PROVENANCE: source reachability is required."""

    root = _make_repo(tmp_path, _planner_references())
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

    artifact = mod.build_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_findings=[],
        source_receipts=unreachable_receipts,
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "blocked"
    assert "source_reachability_failed" in artifact["preconditions_checked"][
        "failed_preconditions"
    ]
    assert artifact["honest_verdict"].startswith("blocked:")


def test_scenario_report_5756_validation_boundaries(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5756-FIELD-PRINCIPLES: schema guards claim boundaries."""

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
    assert artifact["semantic_scholar_receipts"][0]["paper"] == "arXiv:2507.02092"
    assert artifact["duplicate_findings"][0]["classification"] == "duplicate"
    assert artifact["watch_only_findings"][0]["classification"] == "watch_only"
    assert artifact["excluded_findings"][0]["classification"] == "excluded"
    assert artifact["inaccessible_findings"][0]["classification"] == "inaccessible"

    broken = dict(artifact)
    broken["field_principles"] = dict(artifact["field_principles"])
    broken["field_principles"].pop("source_receipts")
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
            "source_id": "bad",
            "classification": "accepted",
            "target_experiment": "exp5755-transition-v514",
        }
    ]
    with pytest.raises(ValueError, match="accepted finding"):
        mod.validate_artifact(broken)

    review_needed = mod.build_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_findings=[
            {
                **_accepted_fixture(),
                "roadmap_scope_change_requested_if_pursued": True,
            }
        ],
    )
    mod.validate_artifact(review_needed)
    assert review_needed["status"] == "blocked"
    assert review_needed["operator_review_required"] is True
    assert review_needed["roadmap_scope_change_requested"] is True


def test_scenario_report_5756_helpers_and_cli(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-REPORT-5756-FIELD-PRINCIPLES: helpers and CLI stay stable."""

    root = _make_repo(tmp_path, _planner_references())
    assert mod.read_text_if_present(root / "missing.md") == ""
    assert mod.path_sha256(root / "missing.md") is None
    assert mod.normalize_timestamp("2026-07-20T14:00:00+00:00").endswith("Z")
    assert mod.planner_marker_line("missing") is None
    assert mod.planner_block_hash("missing") is None
    assert mod.target_experiment_map([_accepted_fixture()])[0]["source_id"] == (
        "post_v514_control_fixture_2607_19999"
    )
    assert mod.honest_verdict(False, True, [], False).startswith("blocked:")
    assert mod.honest_verdict(True, False, [], False).startswith("blocked:")
    assert mod.honest_verdict(True, True, [], False).startswith("complete: no new")
    block = mod.execution_refresh_block([_accepted_fixture()])
    assert mod.insert_after_planner_block("no marker", block).endswith(block)
    fallback_insert = mod.insert_after_planner_block(
        f"{mod.PLANNER_HEADING}\nbody",
        block,
    )
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
    assert "spec_req_report_5756_missing" in mod.preconditions_checked(
        malformed_root,
        marker_found=True,
        source_reachable=True,
    )["failed_preconditions"]

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


def test_scenario_report_5756_module_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-REPORT-5756-FIELD-PRINCIPLES: module entry exits cleanly."""

    root = _make_repo(tmp_path, _planner_references())
    monkeypatch.setattr(
        "sys.argv",
        [
            "experiment_5756_v514_source_delta_ingestion",
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
            "carnot.experiment_5756_v514_source_delta_ingestion",
            run_name="__main__",
        )

    assert exc_info.value.code == 0
    assert mod.RESULT_RELATIVE_PATH.as_posix() in capsys.readouterr().out
