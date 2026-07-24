"""Tests for Exp5878 V523 source-delta ingestion.

Spec refs: REQ-REPORT-5878, SCENARIO-REPORT-5878-ZERO-FINDING,
SCENARIO-REPORT-5878-ACCEPT-BOUNDED-DELTA,
SCENARIO-REPORT-5878-BLOCKED-PRECONDITION,
SCENARIO-REPORT-5878-CLOSED-SCOPE-IMMUTABILITY,
SCENARIO-REPORT-5878-SCHEMA.
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest
import yaml

from carnot import experiment_5878_v523_source_delta_ingestion as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"
START = "2026-07-24T15:10:00Z"
FINISH = "2026-07-24T15:11:29Z"


def _roadmap() -> str:
    tasks = []
    for task_id in mod.ALLOCATED_TARGET_EXPERIMENTS | {
        "exp5877-transition-v523",
        "exp5878-v523-source-delta-ingestion",
        "exp5889-v523-capstone-reconciliation",
    }:
        row = {
            "id": task_id,
            "milestone": mod.MILESTONE,
            "title": task_id,
            "deliverable": f"results/{task_id}.json",
            "model": "gpt-5.5",
        }
        if task_id in {
            "exp5880-grounding-shortcut-fixture",
            "exp5883-gguf-intermediate-layer-surface-preflight",
        }:
            row["gated_on"] = [
                {
                    "upstream": "exp5879-hardness-headroom-taxonomy-corrigendum",
                    "artifact_field": "hardness_surface_headroom_ready_score",
                    "op": "==",
                    "value": 1.0,
                }
            ]
        if task_id == "exp5881-one-to-one-grounding-acquisition-ab":
            row["gated_on"] = [
                {
                    "upstream": "exp5880-grounding-shortcut-fixture",
                    "artifact_field": "grounding_shortcut_fixture_ready_score",
                    "op": "==",
                    "value": 1.0,
                }
            ]
        tasks.append(row)
    tasks.sort(key=lambda row: row["id"])
    return yaml.safe_dump({"milestone": mod.MILESTONE, "tasks": tasks}, sort_keys=False)


def _planner_references() -> str:
    return (
        "## V523 Planner Refresh - 20260724\n\n"
        "- **Differentiable Logic Programming to Mitigate Reasoning Shortcuts** "
        "- arXiv:2607.21185.\n"
        "- **PRO-LONG: Programmatic Memory Enables Long-Horizon Reasoning** "
        "- arXiv:2607.20064v2.\n"
        "- Guarded: p-computer, KAN adaptation, EvoLib, HalluScope, Extropic, Kona.\n"
        "<!-- V523-PLANNER-REFRESH-20260724-END -->\n"
    )


def _accepted_fixture() -> mod.JsonDict:
    return {
        "source_id": "post_v523_fixture_2607_99997",
        "classification": "accepted",
        "title": "Post-V523 Fixture for Grounding Shortcut Controls",
        "url": "https://arxiv.org/abs/2607.99997",
        "publication_date": "2026-07-24",
        "source_date": "2026-07-24",
        "search_timestamp": START,
        "receipt_id": "arxiv_fixture_post_v523",
        "query": 'all:"grounding shortcut"',
        "access_outcome": "reachable_fixture",
        "target_experiment": "exp5880-grounding-shortcut-fixture",
        "source_hook": "Add one new semantic inversion fixture before the A/B.",
        "authority_boundary": (
            "Adds a bounded Exp5880 control only; exact solvers and immutable "
            "roadmap gates remain authority."
        ),
        "post_marker_or_newer_primary_source": True,
        "primary_source": True,
        "reopens_retired_scope": False,
        "method_to_task_mapping": {
            "method": "grounding_shortcut_control_fixture",
            "target_experiment": "exp5880-grounding-shortcut-fixture",
            "task_hook": "semantic inversion exact-control row",
            "failure_boundary": "reject on any split leakage or solver disagreement",
        },
        "reason": (
            "Fixture accepted finding stays inside Exp5880 and does not alter IDs, "
            "gates, authority, model policy, retired scopes, hardware claims, or "
            "headline authority."
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
    (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).write_text(
        references_text,
        encoding="utf-8",
    )
    (root / mod.RESEARCH_STUDYING_RELATIVE_PATH).write_text(
        "# Research Studying\n\nNo V523 execution ingestion yet.\n",
        encoding="utf-8",
    )
    (root / mod.ROADMAP_RELATIVE_PATH).write_text(_roadmap(), encoding="utf-8")
    if with_next:
        (root / mod.ROADMAP_NEXT_RELATIVE_PATH).write_text(_roadmap(), encoding="utf-8")
    (root / "openspec/change-proposals").mkdir(parents=True, exist_ok=True)
    (root / mod.VNEXT_RELATIVE_PATH).write_text(
        "# Research Roadmap vNEXT\n\n"
        "**Milestone:** 2026.07.523\n\n"
        "Exp5878 preserves V523 task identity, gates, authority, and model policy.\n",
        encoding="utf-8",
    )
    (root / "openspec/capabilities/research-reporting").mkdir(
        parents=True,
        exist_ok=True,
    )
    (root / mod.SPEC_RELATIVE_PATH).write_text(
        "\n".join(mod.SPEC_REFS) + "\n",
        encoding="utf-8",
    )
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / mod.EXCLUSION_MANIFEST_RELATIVE_PATH).write_text(
        (
            "retired_extras:\n"
            "- id: final_embeddings_closed\n"
            "  reason: final embeddings remain closed\n"
            "- id: phase_d_closed\n"
            "  reason: PHASE D remains closed\n"
            "- id: generated_answer_repair_closed\n"
            "  reason: generated-answer repair remains closed\n"
            "- id: kan_residuals_closed\n"
            "  reason: KAN residual route remains closed\n"
            "- id: active_observation_lookahead_closed\n"
            "  reason: active observation and lookahead remain closed\n"
            "- id: public_arc_solves_closed\n"
            "  reason: public ARC solves remain closed\n"
            "- id: unchanged_board_probe_closed\n"
            "  reason: unchanged board probes remain closed\n"
            "- id: tsu_kona_execution_closed\n"
            "  reason: TSU and Kona execution require authenticated receipts\n"
        ),
        encoding="utf-8",
    )
    (root / mod.KNOWN_ISSUES_RELATIVE_PATH).write_text(
        "Final embeddings, PHASE D, generated-answer repair, KAN residuals, "
        "active observation/lookahead, public ARC solves, unchanged board probes, "
        "TSU execution, and Kona execution remain closed.\n",
        encoding="utf-8",
    )
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / mod.CONDUCTOR_RELATIVE_PATH).write_text("# conductor fixture\n", encoding="utf-8")
    (root / "results").mkdir(parents=True, exist_ok=True)
    return root


def test_req_report_5878_spec_declares_v523_source_refresh_contract() -> None:
    """REQ-REPORT-5878: OpenSpec names the V523 source-refresh contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-5878") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-REPORT-5878",
        "SCENARIO-REPORT-5878-ZERO-FINDING",
        "SCENARIO-REPORT-5878-ACCEPT-BOUNDED-DELTA",
        "SCENARIO-REPORT-5878-BLOCKED-PRECONDITION",
        "SCENARIO-REPORT-5878-CLOSED-SCOPE-IMMUTABILITY",
        "SCENARIO-REPORT-5878-SCHEMA",
        str(mod.RESULT_RELATIVE_PATH),
        mod.PLANNER_MARKER,
        mod.INFERENCE_SUBSTRATE,
        "`guarded_finding_receipts`",
        "`studying_ledger_modified`",
        "`sota_to_experiment_mapping`",
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_5878_zero_finding_keeps_ledgers_unchanged(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5878-ZERO-FINDING: zero accepted deltas are complete."""

    root = _make_repo(tmp_path, _planner_references())

    artifact = mod.build_and_write_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_findings=[],
        test_commands=["unit"],
        test_exit_codes={"unit": 0},
        duration_s=89.0,
    )

    mod.validate_artifact(artifact)
    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")
    studying = (root / mod.RESEARCH_STUDYING_RELATIVE_PATH).read_text(encoding="utf-8")
    result_text = (root / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")

    assert result_text.endswith("\n")
    assert mod.EXECUTION_REFRESH_HEADING not in references
    assert mod.STUDYING_EXECUTION_HEADING not in studying
    assert artifact["status"] == "complete"
    assert artifact["finding_classification"]["accepted"] == []
    assert artifact["accepted_finding_count"] == 0
    assert artifact["references_modified"] is False
    assert artifact["studying_ledger_modified"] is False
    assert artifact["sota_to_experiment_mapping"] == []
    assert artifact["planner_marker_and_search_window"]["boundary_marker"] == (
        mod.PLANNER_MARKER
    )
    assert artifact["preconditions_checked"]["network_available"] is True
    assert artifact["preconditions_checked"]["api_routes_checked"] is True
    assert artifact["preconditions_checked"]["research_roadmap_next_read"] is False
    assert artifact["roadmap_immutability_receipts"]["roadmap_ids_unchanged"] is True
    assert artifact["roadmap_immutability_receipts"]["gates_unchanged"] is True
    assert artifact["roadmap_immutability_receipts"]["authority_unchanged"] is True
    assert artifact["roadmap_immutability_receipts"]["model_policy_unchanged"] is True
    assert artifact["roadmap_immutability_receipts"]["closed_scopes_reopened"] is False
    assert artifact["roadmap_immutability_receipts"]["hardware_claim_changed"] is False
    assert artifact["roadmap_immutability_receipts"]["headline_claim_changed"] is False
    assert artifact["guarded_finding_receipts"]
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete: no accepted")


def test_scenario_report_5878_accept_delta_appends_once_and_maps_studying(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5878-ACCEPT-BOUNDED-DELTA: accepted controls map exactly."""

    root = _make_repo(tmp_path, _planner_references(), with_next=True)

    artifact = mod.build_and_write_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_findings=[_accepted_fixture()],
        duration_s=89.0,
    )

    mod.validate_artifact(artifact)
    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")
    studying = (root / mod.RESEARCH_STUDYING_RELATIVE_PATH).read_text(encoding="utf-8")

    assert references.count(mod.EXECUTION_REFRESH_HEADING) == 1
    assert studying.count(mod.STUDYING_EXECUTION_HEADING) == 1
    assert "Post-V523 Fixture" in references
    assert "grounding_shortcut_control_fixture" in studying
    assert "exp5880-grounding-shortcut-fixture" in references
    assert references.index(mod.EXECUTION_REFRESH_HEADING) > references.index(
        mod.PLANNER_HEADING
    )
    assert artifact["accepted_finding_count"] == 1
    assert artifact["finding_classification"]["accepted"][0]["target_experiment"] == (
        "exp5880-grounding-shortcut-fixture"
    )
    assert artifact["references_modified"] is True
    assert artifact["studying_ledger_modified"] is True
    assert artifact["sota_to_experiment_mapping"][0]["method"] == (
        "grounding_shortcut_control_fixture"
    )
    assert artifact["honest_verdict"].startswith("complete: accepted 1")
    assert artifact["field_provenance"]["accepted_findings"][0]["source_id"] == (
        "post_v523_fixture_2607_99997"
    )

    second = mod.build_and_write_artifact(
        root=root,
        search_started_at="2026-07-24T15:12:00Z",
        search_finished_at="2026-07-24T15:12:01Z",
        accepted_findings=[_accepted_fixture()],
        duration_s=1.0,
    )
    references_second = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(
        encoding="utf-8"
    )
    studying_second = (root / mod.RESEARCH_STUDYING_RELATIVE_PATH).read_text(
        encoding="utf-8"
    )
    assert references_second.count(mod.EXECUTION_REFRESH_HEADING) == 1
    assert studying_second.count(mod.STUDYING_EXECUTION_HEADING) == 1
    assert second["references_modified"] is False
    assert second["studying_ledger_modified"] is False


def test_scenario_report_5878_blocked_preconditions(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5878-BLOCKED-PRECONDITION: stale marker/routes fail closed."""

    references_text = "## V522 Planner Refresh - 20260723\nKnown references only.\n"
    root = _make_repo(tmp_path, references_text)

    artifact = mod.build_and_write_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_findings=[_accepted_fixture()],
        duration_s=89.0,
    )

    mod.validate_artifact(artifact)
    assert (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(
        encoding="utf-8"
    ) == references_text
    assert artifact["status"] == "blocked"
    assert artifact["preconditions_checked"]["planner_marker_found"] is False
    assert artifact["accepted_finding_count"] == 0
    assert artifact["references_modified"] is False
    assert artifact["studying_ledger_modified"] is False
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
        duration_s=89.0,
    )
    mod.validate_artifact(unreachable)
    assert unreachable["status"] == "blocked"
    assert "source_reachability_failed" in unreachable["preconditions_checked"][
        "failed_preconditions"
    ]


def test_scenario_report_5878_closed_scope_and_roadmap_guards(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5878-CLOSED-SCOPE-IMMUTABILITY: protected boundaries hold."""

    root = _make_repo(tmp_path, _planner_references())
    artifact = mod.build_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_findings=[_accepted_fixture()],
        duration_s=89.0,
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
    all_receipts = artifact["source_receipts"] + artifact["citation_trail_receipts"]
    assert not any("pending" in row["access_outcome"] for row in all_receipts)
    assert {
        "arxiv_v523_duplicate_primary_pages",
        "arxiv_hf_relevant_primary_pages_post_v523",
        "github_prolong_duplicate_repo_receipt",
        "local_sweep_semscholar_keyword_programmatic_memory_agent",
    }.issubset({row["receipt_id"] for row in artifact["source_receipts"]})
    assert artifact["citation_trail_receipts"][0]["latest_publication_date"] == "2026-07-19"
    classes = artifact["finding_classification"]
    assert classes["duplicate"][0]["classification"] == "duplicate"
    assert classes["watch_only"][0]["classification"] == "watch_only"
    assert classes["excluded"][0]["classification"] == "excluded"
    assert artifact["finding_classification"]["all_candidates"] == _ordered_candidates(
        artifact
    )
    protected_scopes = set(artifact["roadmap_immutability_receipts"]["protected_scopes"])
    assert {
        "final embeddings",
        "PHASE D",
        "generated-answer repair",
        "KAN residuals",
        "active observation/lookahead",
        "public ARC solves",
        "unchanged board probes",
        "TSU execution",
        "Kona execution",
    }.issubset(protected_scopes)

    broken = json.loads(json.dumps(artifact))
    broken["finding_classification"]["accepted"] = [
        {**_accepted_fixture(), "target_experiment": "exp5889-v523-capstone-reconciliation"}
    ]
    broken["finding_classification"]["all_candidates"] = _ordered_candidates(broken)
    broken["accepted_finding_count"] = 1
    broken["sota_to_experiment_mapping"] = [
        broken["finding_classification"]["accepted"][0]["method_to_task_mapping"]
    ]
    with pytest.raises(ValueError, match="accepted finding"):
        mod.validate_artifact(broken)

    for changed_field, expected_message in (
        ("post_marker_or_newer_primary_source", "newer primary-source"),
        ("primary_source", "primary-source"),
        ("reopens_retired_scope", "retired scope"),
    ):
        broken = json.loads(json.dumps(artifact))
        candidate = _accepted_fixture()
        candidate[changed_field] = changed_field == "reopens_retired_scope"
        broken["finding_classification"]["accepted"] = [candidate]
        broken["finding_classification"]["all_candidates"] = _ordered_candidates(broken)
        broken["accepted_finding_count"] = 1
        broken["sota_to_experiment_mapping"] = [candidate["method_to_task_mapping"]]
        with pytest.raises(ValueError, match=expected_message):
            mod.validate_artifact(broken)

    old = _accepted_fixture()
    old["publication_date"] = "2026-07-23"
    with pytest.raises(ValueError, match="newer primary-source"):
        mod._validate_candidate(old, "accepted")  # noqa: SLF001

    for field, value, message in (
        ("roadmap_ids_unchanged", False, "roadmap ids"),
        ("gates_unchanged", False, "gates"),
        ("authority_unchanged", False, "authority"),
        ("model_policy_unchanged", False, "model policy"),
        ("closed_scopes_reopened", True, "closed scopes"),
        ("hardware_claim_changed", True, "hardware"),
        ("headline_claim_changed", True, "headline"),
    ):
        broken = json.loads(json.dumps(artifact))
        broken["roadmap_immutability_receipts"][field] = value
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(broken)


def test_scenario_report_5878_schema_helpers_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-REPORT-5878-SCHEMA: schema, checksum, helpers, and CLI stay stable."""

    root = _make_repo(tmp_path, _planner_references())
    assert mod.read_text_if_present(root / "missing.md") == ""
    assert mod.path_sha256(root / "missing.md") is None
    assert mod.normalize_timestamp("2026-07-24T15:10:00+00:00").endswith("Z")
    assert mod.planner_marker_line("missing") is None
    assert mod.planner_block_hash("missing") is None
    assert mod.honest_verdict(False, True, [], False).startswith("blocked:")
    assert mod.honest_verdict(True, False, [], False).startswith("blocked:")
    assert mod.honest_verdict(True, True, [], False, False).startswith("blocked:")
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
    study_block = mod.studying_execution_block([_accepted_fixture()])
    assert mod.insert_studying_block("existing", study_block).endswith(study_block)
    assert mod.insert_studying_block(
        f"prefix\n{mod.STUDYING_EXECUTION_HEADING}\n",
        study_block,
    ) == f"prefix\n{mod.STUDYING_EXECUTION_HEADING}\n"

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
    assert "spec_req_report_5878_missing" in mod.preconditions_checked(
        malformed_root,
        marker_found=True,
        source_reachable=True,
    )["failed_preconditions"]
    monkeypatch.setattr(mod.os, "access", lambda _path, _mode: False)
    assert "output_path_unavailable" in mod.preconditions_checked(
        malformed_root,
        marker_found=True,
        source_reachable=True,
    )["failed_preconditions"]
    monkeypatch.undo()
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
        duration_s=89.0,
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
            "experiment_5878_v523_source_delta_ingestion",
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
            "carnot.experiment_5878_v523_source_delta_ingestion",
            run_name="__main__",
        )
    assert exc_info.value.code == 0
    assert mod.RESULT_RELATIVE_PATH.as_posix() in capsys.readouterr().out


def test_scenario_report_5878_validator_rejects_schema_and_provenance_errors(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-5878: validation rejects schema drift and unsupported receipts."""

    root = _make_repo(tmp_path, _planner_references())
    artifact = mod.build_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        duration_s=89.0,
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

    for flag, message in (
        ("references_modified", "zero accepted"),
        ("studying_ledger_modified", "zero accepted"),
    ):
        broken = json.loads(json.dumps(artifact))
        broken[flag] = True
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(broken)

    broken = json.loads(json.dumps(artifact))
    broken["sota_to_experiment_mapping"] = [{"method": "orphan"}]
    with pytest.raises(ValueError, match="sota_to_experiment_mapping"):
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
    broken["sota_to_experiment_mapping"] = [accepted["method_to_task_mapping"]]
    with pytest.raises(ValueError, match="accepted finding missing source_hook"):
        mod.validate_artifact(broken)

    broken = json.loads(json.dumps(artifact))
    accepted = _accepted_fixture()
    accepted.pop("method_to_task_mapping")
    broken["finding_classification"]["accepted"] = [accepted]
    broken["finding_classification"]["all_candidates"] = _ordered_candidates(broken)
    broken["accepted_finding_count"] = 1
    broken["sota_to_experiment_mapping"] = []
    with pytest.raises(ValueError, match="method-to-task mapping"):
        mod.validate_artifact(broken)

    mapped = _accepted_fixture()
    mapped["method_to_task_mapping"] = {
        "method": "",
        "target_experiment": "exp5880-grounding-shortcut-fixture",
        "task_hook": "hook",
        "failure_boundary": "boundary",
    }
    with pytest.raises(ValueError, match="mapping is incomplete"):
        mod._validate_mapping(  # noqa: SLF001 - direct branch coverage for schema guard.
            mapped["method_to_task_mapping"],
            "exp5880-grounding-shortcut-fixture",
        )

    mapped = _accepted_fixture()
    mapped["method_to_task_mapping"]["target_experiment"] = (
        "exp5881-one-to-one-grounding-acquisition-ab"
    )
    with pytest.raises(ValueError, match="target does not match"):
        mod._validate_mapping(  # noqa: SLF001 - direct branch coverage for schema guard.
            mapped["method_to_task_mapping"],
            "exp5880-grounding-shortcut-fixture",
        )

    with pytest.raises(ValueError, match="outside Exp5879-Exp5888"):
        mod._validate_mapping(  # noqa: SLF001 - direct branch coverage for schema guard.
            {
                "method": "orphan",
                "target_experiment": "exp5889-v523-capstone-reconciliation",
                "task_hook": "hook",
                "failure_boundary": "boundary",
            },
            "exp5889-v523-capstone-reconciliation",
        )

    accepted_artifact = mod.build_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_findings=[_accepted_fixture()],
        duration_s=89.0,
    )
    broken = json.loads(json.dumps(accepted_artifact))
    broken["sota_to_experiment_mapping"][0]["method"] = "changed"
    with pytest.raises(ValueError, match="sota_to_experiment_mapping"):
        mod.validate_artifact(broken)
