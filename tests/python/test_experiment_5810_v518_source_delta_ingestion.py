"""Tests for Exp5810 V518 source-delta ingestion.

Spec refs: REQ-REPORT-5810, SCENARIO-REPORT-5810-ZERO-FINDING,
SCENARIO-REPORT-5810-ACCEPT-BOUNDED-DELTA,
SCENARIO-REPORT-5810-BLOCKED-PRECONDITION,
SCENARIO-REPORT-5810-CLOSED-SCOPE-IMMUTABILITY,
SCENARIO-REPORT-5810-SCHEMA.
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest
import yaml

from carnot import experiment_5810_v518_source_delta_ingestion as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"
START = "2026-07-22T18:24:00Z"
FINISH = "2026-07-22T18:25:12Z"


def _roadmap() -> str:
    tasks = []
    for task_id in mod.ALLOCATED_TARGET_EXPERIMENTS | {
        "exp5810-v518-source-delta-ingestion"
    }:
        row = {
            "id": task_id,
            "milestone": mod.MILESTONE,
            "title": task_id,
            "deliverable": f"results/{task_id}.json",
        }
        if task_id in {
            "exp5813-split-budget-sota-canary",
            "exp5814-channel-qualified-constraint-stream",
            "exp5820-arc-live-heldout-world-model-ab",
        }:
            row["gated_on"] = [
                {
                    "upstream": "exp5811-exp5799-event-provenance-audit",
                    "artifact_field": "canary_evidence_ready_score",
                    "op": "==",
                    "value": 1.0,
                }
            ]
        tasks.append(row)
    tasks.sort(key=lambda row: row["id"])
    return yaml.safe_dump({"milestone": mod.MILESTONE, "tasks": tasks}, sort_keys=False)


def _planner_references() -> str:
    return (
        "## V518 Planner Refresh - 20260722\n\n"
        "- **The Coupling Tax: How Shared Token Budgets Undermine Visible "
        "Chain-of-Thought Under Fixed Output Limits** - arXiv:2605.07686.\n"
        "- **Decode-Time Grammars: Constrained LLM Generation over a Refinement "
        "Order of Grammar Fragments** - arXiv:2607.18357.\n"
        "- **Measuring the Limits of Continual Learning for LLMs / "
        "ImprintBench** - OpenReview CompLearn 2026.\n"
        "<!-- V518-PLANNER-REFRESH-20260722-END -->\n"
    )


def _accepted_fixture() -> mod.JsonDict:
    return {
        "source_id": "post_v518_fixture_2607_99998",
        "classification": "accepted",
        "title": "Post-V518 Fixture for Sealed Candidate Environment Audits",
        "url": "https://arxiv.org/abs/2607.99998",
        "publication_date": "2026-07-22",
        "source_date": "2026-07-22",
        "search_timestamp": START,
        "receipt_id": "arxiv_fixture_post_v518",
        "query": 'all:"sealed candidate environment"',
        "access_outcome": "reachable_fixture",
        "target_experiment": "exp5812-split-budget-channel-contract",
        "source_hook": "Require candidate-environment hashes before split-budget decoding.",
        "authority_boundary": (
            "Adds an audit control inside Exp5812 only; exact solvers still decide "
            "semantic correctness."
        ),
        "post_marker_or_newly_actionable": True,
        "reason": (
            "Fixture accepted finding stays inside Exp5812 and does not alter IDs, "
            "gates, required models, closed scopes, hardware claims, or headline claims."
        ),
    }


def _ordered_candidates(artifact: mod.JsonDict) -> list[mod.JsonDict]:
    classes = artifact["finding_classification"]
    return (
        classes["accepted"]
        + classes["duplicate"]
        + classes["watch_only"]
        + classes["excluded"]
        + classes["inaccessible"]
    )


def _make_repo(root: Path, references_text: str, *, with_next: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for relative in ("AGENTS.md", "CODEX.md", "CLAUDE.md", "research-program.md"):
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
        "**Milestone:** `2026.07.518`\n\n"
        "Exp5810 preserves V518 task identity and gate structure.\n",
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
            "- id: phase_d_closed\n"
            "  reason: PHASE D remains closed\n"
            "- id: grammar_semantic_authority_closed\n"
            "  reason: grammar-as-semantic-authority remains closed\n"
            "- id: shadow_integration_closed\n"
            "  reason: shadow integration remains closed\n"
            "- id: arc_cegis_public_closed\n"
            "  reason: ARC CEGIS and public solves remain closed\n"
            "- id: unchanged_board_probe_closed\n"
            "  reason: unchanged board probes remain closed\n"
            "- id: tsu_kona_execution_closed\n"
            "  reason: TSU and Kona execution require authenticated local receipts\n"
        ),
        encoding="utf-8",
    )
    (root / mod.KNOWN_ISSUES_RELATIVE_PATH).write_text(
        "PHASE D, grammar-as-semantic-authority, shadow integration, ARC CEGIS, "
        "public solves, unchanged board probes, TSU execution, and Kona execution "
        "remain closed.\n",
        encoding="utf-8",
    )
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / mod.CONDUCTOR_RELATIVE_PATH).write_text("# conductor fixture\n", encoding="utf-8")
    (root / "results").mkdir(parents=True, exist_ok=True)
    return root


def test_req_report_5810_spec_declares_v518_source_refresh_contract() -> None:
    """REQ-REPORT-5810: OpenSpec names the V518 field and source contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-5810") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-REPORT-5810",
        "SCENARIO-REPORT-5810-ZERO-FINDING",
        "SCENARIO-REPORT-5810-ACCEPT-BOUNDED-DELTA",
        "SCENARIO-REPORT-5810-BLOCKED-PRECONDITION",
        "SCENARIO-REPORT-5810-CLOSED-SCOPE-IMMUTABILITY",
        "SCENARIO-REPORT-5810-SCHEMA",
        str(mod.RESULT_RELATIVE_PATH),
        mod.PLANNER_MARKER,
        mod.INFERENCE_SUBSTRATE,
        "`planner_marker_and_search_window`",
        "`source_receipts`",
        "`citation_trail_receipts`",
        "`roadmap_immutability_receipts`",
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_5810_zero_finding_keeps_references_unchanged(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5810-ZERO-FINDING: zero accepted deltas are complete."""

    root = _make_repo(tmp_path, _planner_references())

    artifact = mod.build_and_write_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_findings=[],
        test_commands=["unit"],
        test_exit_codes={"unit": 0},
        duration_s=72.0,
    )

    mod.validate_artifact(artifact)
    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")
    result_text = (root / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")

    assert result_text.endswith("\n")
    assert mod.EXECUTION_REFRESH_HEADING not in references
    assert artifact["status"] == "complete"
    assert artifact["finding_classification"]["accepted"] == []
    assert artifact["accepted_finding_count"] == 0
    assert artifact["references_modified"] is False
    assert artifact["planner_marker_and_search_window"]["boundary_marker"] == mod.PLANNER_MARKER
    assert artifact["preconditions_checked"]["network_search_available"] is True
    assert artifact["preconditions_checked"]["research_roadmap_next_read"] is False
    assert artifact["roadmap_immutability_receipts"]["roadmap_ids_unchanged"] is True
    assert artifact["roadmap_immutability_receipts"]["gates_unchanged"] is True
    assert artifact["roadmap_immutability_receipts"]["closed_scopes_reopened"] is False
    assert artifact["roadmap_immutability_receipts"]["hardware_claim_changed"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete: no accepted")


def test_scenario_report_5810_accept_delta_appends_once(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5810-ACCEPT-BOUNDED-DELTA: accepted controls append once."""

    root = _make_repo(tmp_path, _planner_references(), with_next=True)

    artifact = mod.build_and_write_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_findings=[_accepted_fixture()],
        duration_s=72.0,
    )

    mod.validate_artifact(artifact)
    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")

    assert references.count(mod.EXECUTION_REFRESH_HEADING) == 1
    assert "Post-V518 Fixture" in references
    assert "exp5812-split-budget-channel-contract" in references
    assert references.index(mod.EXECUTION_REFRESH_HEADING) > references.index(
        mod.PLANNER_HEADING
    )
    assert artifact["accepted_finding_count"] == 1
    assert artifact["finding_classification"]["accepted"][0]["target_experiment"] == (
        "exp5812-split-budget-channel-contract"
    )
    assert artifact["references_modified"] is True
    assert artifact["roadmap_immutability_receipts"]["roadmap_ids_unchanged"] is True
    assert artifact["roadmap_immutability_receipts"]["gates_unchanged"] is True
    assert artifact["honest_verdict"].startswith("complete: accepted 1")
    assert artifact["field_provenance"]["accepted_findings"][0]["source_id"] == (
        "post_v518_fixture_2607_99998"
    )

    second = mod.build_and_write_artifact(
        root=root,
        search_started_at="2026-07-22T18:26:00Z",
        search_finished_at="2026-07-22T18:26:01Z",
        accepted_findings=[_accepted_fixture()],
        duration_s=1.0,
    )
    references_second = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(
        encoding="utf-8"
    )
    assert references_second.count(mod.EXECUTION_REFRESH_HEADING) == 1
    assert second["references_modified"] is False


def test_scenario_report_5810_blocked_preconditions(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5810-BLOCKED-PRECONDITION: stale marker/routes fail closed."""

    references_text = "## V517 Planner Refresh - 20260722\nKnown references only.\n"
    root = _make_repo(tmp_path, references_text)

    artifact = mod.build_and_write_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_findings=[_accepted_fixture()],
        duration_s=72.0,
    )

    mod.validate_artifact(artifact)
    assert (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(
        encoding="utf-8"
    ) == references_text
    assert artifact["status"] == "blocked"
    assert artifact["preconditions_checked"]["planner_marker_found"] is False
    assert artifact["accepted_finding_count"] == 0
    assert artifact["references_modified"] is False
    assert artifact["honest_verdict"].startswith("blocked:")

    reachable_root = _make_repo(tmp_path / "reachable", _planner_references())
    unreachable_receipts = [
        {
            "receipt_id": "arxiv_down",
            "source_family": "arXiv",
            "source_role": "primary",
            "query": "fixture",
            "url": "https://arxiv.org/",
            "accessed_at": START,
            "access_outcome": "inaccessible_timeout",
            "candidate_ids": [],
        }
    ]
    unreachable = mod.build_artifact(
        root=reachable_root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_findings=[],
        source_receipts=unreachable_receipts,
        citation_trail_receipts=[],
        duration_s=72.0,
    )
    mod.validate_artifact(unreachable)
    assert unreachable["status"] == "blocked"
    assert "source_reachability_failed" in unreachable["preconditions_checked"][
        "failed_preconditions"
    ]


def test_scenario_report_5810_closed_scope_and_roadmap_guards(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5810-CLOSED-SCOPE-IMMUTABILITY: protected boundaries hold."""

    root = _make_repo(tmp_path, _planner_references())
    artifact = mod.build_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_findings=[_accepted_fixture()],
        duration_s=72.0,
    )

    mod.validate_artifact(artifact)
    families = {row["source_family"] for row in artifact["source_receipts"]}
    assert {
        "arXiv",
        "OpenReview",
        "Hugging Face Papers",
        "GitHub discovery",
        "Extropic writing",
        "Logical Intelligence",
    }.issubset(families)
    assert {row["paper"] for row in artifact["citation_trail_receipts"]} == {
        "arXiv:2507.02092",
        "arXiv:2512.15605",
    }
    classes = artifact["finding_classification"]
    assert classes["duplicate"][0]["classification"] == "duplicate"
    assert classes["watch_only"][0]["classification"] == "watch_only"
    assert classes["excluded"][0]["classification"] == "excluded"
    assert artifact["finding_classification"]["all_candidates"] == _ordered_candidates(
        artifact
    )
    protected_scopes = set(artifact["roadmap_immutability_receipts"]["protected_scopes"])
    assert {
        "PHASE D",
        "grammar-as-semantic-authority",
        "shadow integration",
        "ARC CEGIS/public solves",
        "unchanged board probes",
        "TSU execution",
        "Kona execution",
    }.issubset(protected_scopes)

    broken = json.loads(json.dumps(artifact))
    broken["finding_classification"]["accepted"] = [
        {
            **_accepted_fixture(),
            "target_experiment": "exp5822-v518-capstone-reconciliation",
        }
    ]
    broken["finding_classification"]["all_candidates"] = _ordered_candidates(broken)
    broken["accepted_finding_count"] = 1
    with pytest.raises(ValueError, match="accepted finding"):
        mod.validate_artifact(broken)

    broken = json.loads(json.dumps(artifact))
    broken["finding_classification"]["accepted"] = [
        {**_accepted_fixture(), "post_marker_or_newly_actionable": False}
    ]
    broken["finding_classification"]["all_candidates"] = _ordered_candidates(broken)
    broken["accepted_finding_count"] = 1
    with pytest.raises(ValueError, match="post-marker"):
        mod.validate_artifact(broken)

    for field, value, message in (
        ("roadmap_ids_unchanged", False, "roadmap ids"),
        ("gates_unchanged", False, "gates"),
        ("closed_scopes_reopened", True, "closed scopes"),
        ("hardware_claim_changed", True, "hardware"),
        ("headline_claim_changed", True, "headline"),
    ):
        broken = json.loads(json.dumps(artifact))
        broken["roadmap_immutability_receipts"][field] = value
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(broken)


def test_scenario_report_5810_schema_helpers_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-REPORT-5810-SCHEMA: schema, checksum, helpers, and CLI stay stable."""

    root = _make_repo(tmp_path, _planner_references())
    assert mod.read_text_if_present(root / "missing.md") == ""
    assert mod.path_sha256(root / "missing.md") is None
    assert mod.normalize_timestamp("2026-07-22T18:24:00+00:00").endswith("Z")
    assert mod.planner_marker_line("missing") is None
    assert mod.planner_block_hash("missing") is None
    assert mod.honest_verdict(False, True, [], False).startswith("blocked:")
    assert mod.honest_verdict(True, False, [], False).startswith("blocked:")
    assert mod.honest_verdict(True, True, [], True).startswith("blocked:")
    assert mod.honest_verdict(True, True, [], False).startswith("complete: no accepted")
    block = mod.execution_refresh_block([_accepted_fixture()])
    assert mod.insert_after_planner_block("no marker", block).endswith(block)
    assert mod.insert_after_planner_block(f"prefix\n{mod.EXECUTION_REFRESH_HEADING}\n", block) == (
        f"prefix\n{mod.EXECUTION_REFRESH_HEADING}\n"
    )
    assert mod.insert_after_planner_block(f"{mod.PLANNER_HEADING}\nbody", block).count(
        mod.EXECUTION_REFRESH_HEADING
    ) == 1

    malformed_root = _make_repo(tmp_path / "malformed", _planner_references())
    (malformed_root / mod.ROADMAP_RELATIVE_PATH).write_text("- just\n- a list\n")
    assert mod.preconditions_checked(
        malformed_root,
        marker_found=True,
        source_reachable=True,
    )["roadmap_ids_hash"] is None
    (malformed_root / mod.ROADMAP_RELATIVE_PATH).write_text(
        "tasks: not-a-list\nmilestone: fixture\n",
        encoding="utf-8",
    )
    assert mod.preconditions_checked(
        malformed_root,
        marker_found=True,
        source_reachable=True,
    )["active_roadmap_milestone"] == "fixture"
    (malformed_root / mod.ROADMAP_RELATIVE_PATH).write_text("tasks: [\n", encoding="utf-8")
    assert mod.preconditions_checked(
        malformed_root,
        marker_found=True,
        source_reachable=True,
    )["roadmap_ids_hash"] is None
    (malformed_root / mod.SPEC_RELATIVE_PATH).write_text("missing\n", encoding="utf-8")
    assert "spec_req_report_5810_missing" in mod.preconditions_checked(
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
        duration_s=72.0,
    )
    mod.validate_artifact(artifact)
    broken = json.loads(json.dumps(artifact))
    broken["field_provenance"].pop("source_receipts")
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(broken)

    broken = json.loads(json.dumps(artifact))
    broken["search_finished_at"] = START
    with pytest.raises(ValueError, match="timestamp"):
        mod.validate_artifact(broken)

    broken = json.loads(json.dumps(artifact))
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
            "experiment_5810_v518_source_delta_ingestion",
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
            "carnot.experiment_5810_v518_source_delta_ingestion",
            run_name="__main__",
        )
    assert exc_info.value.code == 0
    assert mod.RESULT_RELATIVE_PATH.as_posix() in capsys.readouterr().out


def test_scenario_report_5810_validator_rejects_schema_and_provenance_errors(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-5810: validation rejects schema drift and unsupported receipts."""

    root = _make_repo(tmp_path, _planner_references())
    artifact = mod.build_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        duration_s=72.0,
    )

    missing = json.loads(json.dumps(artifact))
    missing.pop("status")
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(missing)

    broken = json.loads(json.dumps(artifact))
    broken["field_provenance"] = "not-a-map"
    with pytest.raises(ValueError, match="field_provenance must be a mapping"):
        mod.validate_artifact(broken)

    broken = json.loads(json.dumps(artifact))
    broken["status"] = "done"
    with pytest.raises(ValueError, match="invalid status"):
        mod.validate_artifact(broken)

    broken = json.loads(json.dumps(artifact))
    broken["honest_verdict"] = "unsupported verdict"
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(broken)

    broken = json.loads(json.dumps(artifact))
    broken["duration_s"] = -1
    with pytest.raises(ValueError, match="duration"):
        mod.validate_artifact(broken)

    broken = json.loads(json.dumps(artifact))
    broken["inference_substrate"] = "live_llm_inference"
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(broken)

    broken = json.loads(json.dumps(artifact))
    broken["accepted_finding_count"] = 1
    with pytest.raises(ValueError, match="accepted_finding_count"):
        mod.validate_artifact(broken)

    broken = json.loads(json.dumps(artifact))
    broken["references_modified"] = True
    with pytest.raises(ValueError, match="zero accepted"):
        mod.validate_artifact(broken)

    broken = json.loads(json.dumps(artifact))
    broken["finding_classification"]["all_candidates"] = (
        broken["finding_classification"]["all_candidates"][:-1]
    )
    with pytest.raises(ValueError, match="all_candidates"):
        mod.validate_artifact(broken)

    broken = json.loads(json.dumps(artifact))
    broken["finding_classification"]["duplicate"][0]["classification"] = "novel"
    broken["finding_classification"]["all_candidates"] = _ordered_candidates(broken)
    with pytest.raises(ValueError, match="invalid candidate classification"):
        mod.validate_artifact(broken)

    broken = json.loads(json.dumps(artifact))
    broken["finding_classification"]["watch_only"][0]["url"] = ""
    broken["finding_classification"]["all_candidates"] = _ordered_candidates(broken)
    with pytest.raises(ValueError, match="provenance field"):
        mod.validate_artifact(broken)

    broken = json.loads(json.dumps(artifact))
    broken["finding_classification"]["excluded"][0].pop("publication_date", None)
    broken["finding_classification"]["all_candidates"] = _ordered_candidates(broken)
    with pytest.raises(ValueError, match="publication/source date"):
        mod.validate_artifact(broken)

    broken = json.loads(json.dumps(artifact))
    broken["source_receipts"][0]["url"] = ""
    with pytest.raises(ValueError, match="source receipt"):
        mod.validate_artifact(broken)

    broken = json.loads(json.dumps(artifact))
    broken["source_receipts"][0] = "not-a-receipt"
    with pytest.raises(ValueError, match="source receipt"):
        mod.validate_artifact(broken)

    broken = json.loads(json.dumps(artifact))
    broken["citation_trail_receipts"] = broken["citation_trail_receipts"][:1]
    with pytest.raises(ValueError, match="citation trail"):
        mod.validate_artifact(broken)

    broken = json.loads(json.dumps(artifact))
    broken["citation_trail_receipts"][0]["url"] = ""
    with pytest.raises(ValueError, match="citation trail receipt"):
        mod.validate_artifact(broken)

    broken = json.loads(json.dumps(artifact))
    accepted = _accepted_fixture()
    accepted.pop("source_hook")
    broken["finding_classification"]["accepted"] = [accepted]
    broken["finding_classification"]["all_candidates"] = _ordered_candidates(broken)
    broken["accepted_finding_count"] = 1
    with pytest.raises(ValueError, match="accepted finding missing source_hook"):
        mod.validate_artifact(broken)
