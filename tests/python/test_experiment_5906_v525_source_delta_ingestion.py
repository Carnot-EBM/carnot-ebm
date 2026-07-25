"""Tests for Exp5906 V525 source-delta ingestion.

Spec refs: REQ-REPORT-5906, SCENARIO-REPORT-5906-ZERO-FINDING,
SCENARIO-REPORT-5906-ACCEPT-BOUNDED-DELTA,
SCENARIO-REPORT-5906-HALLMARK-UNCERTAINTY,
SCENARIO-REPORT-5906-DUPLICATE-AND-RETIRED-SCOPE,
SCENARIO-REPORT-5906-SCHEMA.
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest
import yaml

from carnot import experiment_5906_v525_source_delta_ingestion as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"
START = "2026-07-25T04:10:00Z"
FINISH = "2026-07-25T04:13:00Z"


def _planner_references() -> str:
    return (
        "## V525 Planner Refresh - 20260724\n\n"
        "- **Towards Automated Formal Verification of zkEVMs Using "
        "LLM-Guided Constraint Synthesis** - arXiv:2607.19795.\n"
        "- **Memoir: Should a Model Write to Its Memory While It Thinks?** - "
        "arXiv:2607.20792.\n"
        "- **HALLMARK: Diagnosing Three Failure Modes in LLM Citation "
        "Verifiers** - arXiv:2607.18360.\n"
        "<!-- V525-PLANNER-REFRESH-20260724-END -->\n"
    )


def _roadmap() -> str:
    tasks = [
        {
            "id": "exp5906-v525-source-delta-ingestion",
            "milestone": mod.MILESTONE,
            "title": "source refresh",
            "deliverable": mod.RESULT_RELATIVE_PATH.as_posix(),
            "model": "gpt-5.5",
        }
    ]
    for task_id in mod.ALLOCATED_TARGET_EXPERIMENTS:
        row = {
            "id": task_id,
            "milestone": mod.MILESTONE,
            "title": task_id,
            "deliverable": f"results/{task_id}.json",
            "model": "gpt-5.5",
        }
        if task_id == "exp5909-sota-constraint-synthesis-ab":
            row["gated_on"] = [
                {
                    "upstream": "exp5908-verisynth-constraint-fixture",
                    "artifact_field": "verisynth_constraint_fixture_ready_score",
                    "op": "==",
                    "value": 1.0,
                }
            ]
            row["requires_gpu"] = True
        tasks.append(row)
    return yaml.safe_dump({"milestone": mod.MILESTONE, "tasks": tasks}, sort_keys=False)


def _make_repo(root: Path, references_text: str, *, with_next: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).write_text(
        references_text,
        encoding="utf-8",
    )
    (root / mod.ROADMAP_RELATIVE_PATH).write_text(_roadmap(), encoding="utf-8")
    if with_next:
        (root / mod.ROADMAP_NEXT_RELATIVE_PATH).write_text(_roadmap(), encoding="utf-8")
    (root / "openspec/change-proposals").mkdir(parents=True, exist_ok=True)
    (root / mod.VNEXT_RELATIVE_PATH).write_text(
        "# vNEXT\n\n**Milestone:** 2026.07.525\n",
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
        "retired_extras:\n"
        "- id: final_embedding_scoring_closed\n"
        "  reason: final-embedding scoring remains closed\n"
        "- id: kan_mutation_closed\n"
        "  reason: KAN mutation remains closed\n"
        "- id: generated_answer_repair_closed\n"
        "  reason: generated-answer repair remains closed\n"
        "- id: public_arc_solves_closed\n"
        "  reason: public ARC solves remain closed\n"
        "- id: unchanged_board_probe_closed\n"
        "  reason: unchanged board probes remain closed\n",
        encoding="utf-8",
    )
    for rel_path in (
        mod.AGENTS_RELATIVE_PATH,
        mod.CODEX_RELATIVE_PATH,
        mod.CLAUDE_RELATIVE_PATH,
        mod.KNOWN_ISSUES_RELATIVE_PATH,
        mod.CONDUCTOR_RELATIVE_PATH,
        mod.SWEEP_CLUSTERS_RELATIVE_PATH,
        mod.SWEEP_SEMSCHOLAR_RELATIVE_PATH,
        mod.STATUS_RELATIVE_PATH,
        mod.CHANGELOG_RELATIVE_PATH,
        mod.TRACEABILITY_RELATIVE_PATH,
    ):
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{rel_path.as_posix()} fixture\n", encoding="utf-8")
    (root / "results").mkdir(parents=True, exist_ok=True)
    return root


def _accepted_fixture() -> mod.JsonDict:
    return {
        "source_id": "post_v525_fixture_2607_99999",
        "classification": "accepted",
        "decision_bucket": "accepted",
        "title": "Post-V525 Fixture for Transactional Memory Poison Controls",
        "url": "https://arxiv.org/abs/2607.99999",
        "identifier": "2607.99999",
        "publication_date": "2026-07-25",
        "source_date": "2026-07-25",
        "search_timestamp": START,
        "receipt_id": "arxiv_fixture_post_v525",
        "query_family": "arxiv_primary",
        "query": 'all:"transactional memory poison"',
        "access_outcome": "reachable_fixture",
        "target_experiment": "exp5913-transactional-constraint-memory-fixture",
        "source_hook": "Add a bounded poison-burst quarantine control.",
        "authority_boundary": (
            "Sharpens Exp5913 controls only; exact validators and rollback "
            "remain authoritative."
        ),
        "post_marker_or_newer_primary_source": True,
        "primary_source": True,
        "duplicate_of_existing_reference": False,
        "reopens_retired_scope": False,
        "method_to_task_mapping": {
            "method": "transactional_memory_poison_quarantine_control",
            "target_experiment": "exp5913-transactional-constraint-memory-fixture",
            "task_hook": "bounded poison-burst quarantine control",
            "failure_boundary": "reject on missing rollback or protected-prefix loss",
        },
        "reason": "New primary fixture stays inside an allocated .525 task.",
    }


def _ordered_candidates(artifact: mod.JsonDict) -> list[mod.JsonDict]:
    classes = artifact["accepted_rejected_abstained_findings"]
    return (
        classes["accepted"]
        + classes["rejected"]
        + classes["abstained"]
        + classes["false_positive"]
        + classes["known_false_negative"]
        + classes["cutoff_confound"]
        + classes["duplicate"]
        + classes["retired_scope"]
        + classes["inaccessible"]
    )


def test_req_report_5906_spec_declares_hallmark_source_refresh_contract() -> None:
    """REQ-REPORT-5906: OpenSpec names the V525 source-refresh contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-5906") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-REPORT-5906",
        "SCENARIO-REPORT-5906-ZERO-FINDING",
        "SCENARIO-REPORT-5906-ACCEPT-BOUNDED-DELTA",
        "SCENARIO-REPORT-5906-HALLMARK-UNCERTAINTY",
        "SCENARIO-REPORT-5906-DUPLICATE-AND-RETIRED-SCOPE",
        "SCENARIO-REPORT-5906-SCHEMA",
        str(mod.RESULT_RELATIVE_PATH),
        mod.PLANNER_MARKER,
        mod.INFERENCE_SUBSTRATE,
        "`accepted_rejected_abstained_findings`",
        "`false_positive_false_negative_and_cutoff_receipts`",
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_5906_zero_delta_keeps_references_unchanged(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5906-ZERO-FINDING: zero accepted deltas are complete."""

    root = _make_repo(tmp_path, _planner_references())
    before = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")

    artifact = mod.build_and_write_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_findings=[],
        duration_s=180.0,
        test_commands=["unit"],
        test_exit_codes={"unit": 0},
    )

    mod.validate_artifact(artifact)
    after = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")
    result_text = (root / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")

    assert after == before
    assert result_text.endswith("\n")
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete_null:")
    assert artifact["references_append_receipt"]["appended"] is False
    assert artifact["references_append_receipt"]["accepted_count"] == 0
    assert artifact["accepted_rejected_abstained_findings"]["accepted"] == []
    assert artifact["accepted_rejected_abstained_findings"]["rejected"]
    assert artifact["accepted_rejected_abstained_findings"]["abstained"]
    assert artifact["false_positive_false_negative_and_cutoff_receipts"][
        "false_positive_source_decisions"
    ]
    assert artifact["false_positive_false_negative_and_cutoff_receipts"][
        "known_false_negative_source_decisions"
    ]
    assert artifact["false_positive_false_negative_and_cutoff_receipts"][
        "cutoff_confounds"
    ]
    counts = artifact["primary_secondary_and_official_source_counts"]
    assert counts["primary"] >= 1
    assert counts["secondary"] >= 1
    assert counts["official"] >= 1
    assert artifact["preconditions_checked"]["research_roadmap_next_read"] is False
    assert artifact["search_window_and_marker_receipt"]["boundary_marker"] == (
        mod.PLANNER_MARKER
    )
    assert artifact["source_queries_and_endpoint_receipts"]["endpoint_failures"]
    assert artifact["source_queries_and_endpoint_receipts"]["rate_limits"]
    assert artifact["task_identity_and_gate_immutability"]["task_ids_unchanged"] is True
    assert artifact["task_identity_and_gate_immutability"]["gates_unchanged"] is True
    assert artifact["protected_files_unchanged"]["all_unchanged"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE


def test_scenario_report_5906_accepted_delta_appends_once(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5906-ACCEPT-BOUNDED-DELTA: accepted deltas map exactly."""

    root = _make_repo(tmp_path, _planner_references(), with_next=True)

    artifact = mod.build_and_write_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_findings=[_accepted_fixture()],
        duration_s=180.0,
    )

    mod.validate_artifact(artifact)
    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(
        encoding="utf-8"
    )
    assert references.count(mod.EXECUTION_DELTA_HEADING) == 1
    assert "Post-V525 Fixture" in references
    assert "exp5913-transactional-constraint-memory-fixture" in references
    assert artifact["honest_verdict"].startswith("complete_delta:")
    assert artifact["references_append_receipt"]["appended"] is True
    assert artifact["references_append_receipt"]["accepted_count"] == 1
    assert artifact["references_append_receipt"]["heading"] == mod.EXECUTION_DELTA_HEADING
    assert artifact["accepted_rejected_abstained_findings"]["accepted"][0][
        "target_experiment"
    ] == "exp5913-transactional-constraint-memory-fixture"
    assert artifact["accepted_rejected_abstained_findings"]["all_candidates"] == (
        _ordered_candidates(artifact)
    )

    second = mod.build_and_write_artifact(
        root=root,
        search_started_at="2026-07-25T04:14:00Z",
        search_finished_at="2026-07-25T04:15:00Z",
        accepted_findings=[_accepted_fixture()],
        duration_s=60.0,
    )
    references_second = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(
        encoding="utf-8"
    )
    assert references_second.count(mod.EXECUTION_DELTA_HEADING) == 1
    assert second["references_append_receipt"]["appended"] is False


def test_scenario_report_5906_hallmark_uncertainty_and_retired_filters(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5906-HALLMARK-UNCERTAINTY: uncertainty remains explicit."""

    root = _make_repo(tmp_path, _planner_references())
    artifact = mod.build_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_findings=[_accepted_fixture()],
        duration_s=180.0,
    )

    mod.validate_artifact(artifact)
    classes = artifact["accepted_rejected_abstained_findings"]
    assert {row["classification"] for row in classes["abstained"]} == {"abstained"}
    assert {row["classification"] for row in classes["false_positive"]} == {
        "false_positive"
    }
    assert {row["classification"] for row in classes["known_false_negative"]} == {
        "known_false_negative"
    }
    assert {row["classification"] for row in classes["cutoff_confound"]} == {
        "cutoff_confound"
    }
    assert artifact["false_positive_false_negative_and_cutoff_receipts"][
        "principle"
    ] == mod.REQUIRED_FIELD_PRINCIPLES[
        "false_positive_false_negative_and_cutoff_receipts"
    ]
    filters = artifact["duplicate_and_retired_scope_filter"]
    assert filters["duplicate_dimensions"] == [
        "title",
        "identifier",
        "mechanism",
        "existing_reference_heading",
    ]
    assert "KAN mutation" in filters["retired_scope_rules"]
    assert "final-embedding scoring" in filters["retired_scope_rules"]
    assert filters["accepted_reopens_retired_scope_count"] == 0

    for changed_field, expected_message in (
        ("post_marker_or_newer_primary_source", "newer primary-source"),
        ("primary_source", "primary-source"),
        ("duplicate_of_existing_reference", "duplicate"),
        ("reopens_retired_scope", "retired scope"),
    ):
        broken = json.loads(json.dumps(artifact))
        candidate = _accepted_fixture()
        candidate[changed_field] = changed_field in {
            "duplicate_of_existing_reference",
            "reopens_retired_scope",
        }
        broken["accepted_rejected_abstained_findings"]["accepted"] = [candidate]
        broken["accepted_rejected_abstained_findings"]["all_candidates"] = (
            _ordered_candidates(broken)
        )
        broken["references_append_receipt"]["accepted_count"] = 1
        broken["references_append_receipt"]["accepted_source_ids"] = [
            candidate["source_id"]
        ]
        with pytest.raises(ValueError, match=expected_message):
            mod.validate_artifact(broken)

    old = _accepted_fixture()
    old["publication_date"] = "2026-07-24"
    with pytest.raises(ValueError, match="newer primary-source"):
        mod._validate_finding(old, "accepted")  # noqa: SLF001

    wrong_target = _accepted_fixture()
    wrong_target["target_experiment"] = "exp5917-v525-capstone-reconciliation"
    wrong_target["method_to_task_mapping"]["target_experiment"] = wrong_target[
        "target_experiment"
    ]
    with pytest.raises(ValueError, match="allocated .525 experiment"):
        mod._validate_finding(wrong_target, "accepted")  # noqa: SLF001


def test_scenario_report_5906_blocked_preconditions(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5906-SCHEMA: missing marker or routes fail closed."""

    root = _make_repo(tmp_path, "## V525 Planner Refresh - 20260724\nno marker\n")
    artifact = mod.build_and_write_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_findings=[_accepted_fixture()],
        duration_s=180.0,
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["accepted_rejected_abstained_findings"]["accepted"] == []
    assert artifact["references_append_receipt"]["appended"] is False
    assert "planner_marker_missing" in artifact["preconditions_checked"][
        "failed_preconditions"
    ]

    reachable_root = _make_repo(tmp_path / "routes", _planner_references())
    unreachable = mod.build_artifact(
        root=reachable_root,
        search_started_at=START,
        search_finished_at=FINISH,
        source_receipts=[
            {
                "receipt_id": "arxiv_down",
                "source_family": "arXiv",
                "source_role": "primary",
                "query_family": "arxiv_primary",
                "query": "fixture",
                "url": "https://arxiv.org/",
                "accessed_at": START,
                "access_outcome": "inaccessible_timeout",
                "candidate_ids": [],
                "candidate_count": 0,
                "receipt_summary": "down",
            }
        ],
        duration_s=180.0,
    )
    mod.validate_artifact(unreachable)
    assert unreachable["status"] == "blocked"
    assert "source_reachability_failed" in unreachable["preconditions_checked"][
        "failed_preconditions"
    ]


def test_scenario_report_5906_schema_helpers_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-REPORT-5906-SCHEMA: helpers, schema, checksum, and CLI hold."""

    root = _make_repo(tmp_path, _planner_references())
    assert mod.read_text_if_present(root / "missing.md") == ""
    assert mod.path_sha256(root / "missing.md") is None
    assert mod.normalize_timestamp("2026-07-25T04:10:00+00:00").endswith("Z")
    assert mod.planner_marker_line("missing") is None
    assert mod.planner_block_hash("missing") is None
    assert mod.honest_verdict(False, True, [], False).startswith("blocked:")
    assert mod.honest_verdict(True, False, [], False).startswith("blocked:")
    assert mod.honest_verdict(True, True, [], False).startswith("complete_null:")
    assert mod.honest_verdict(True, True, [_accepted_fixture()], False).startswith(
        "complete_delta:"
    )
    block = mod.execution_delta_block([_accepted_fixture()])
    assert mod.insert_after_planner_block("no marker", block).endswith(block)
    assert mod.insert_after_planner_block(
        f"prefix\n{mod.EXECUTION_DELTA_HEADING}\n",
        block,
    ) == f"prefix\n{mod.EXECUTION_DELTA_HEADING}\n"

    artifact = mod.build_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        duration_s=180.0,
    )
    mod.validate_artifact(artifact)

    malformed = _make_repo(tmp_path / "malformed", _planner_references())
    (malformed / mod.ROADMAP_RELATIVE_PATH).write_text("- just\n- a list\n")
    assert "active_roadmap_identity_unavailable" in mod.preconditions_checked(
        malformed,
        marker_found=True,
        source_reachable=True,
        checked_at=START,
    )["failed_preconditions"]
    (malformed / mod.ROADMAP_RELATIVE_PATH).write_text("tasks: [\n", encoding="utf-8")
    failed = mod.preconditions_checked(
        malformed,
        marker_found=True,
        source_reachable=True,
        checked_at=START,
    )["failed_preconditions"]
    assert "active_roadmap_identity_unavailable" in failed
    (malformed / mod.SPEC_RELATIVE_PATH).write_text("missing\n", encoding="utf-8")
    assert "spec_req_report_5906_missing" in mod.preconditions_checked(
        malformed,
        marker_found=True,
        source_reachable=True,
        checked_at=START,
    )["failed_preconditions"]
    monkeypatch.setattr(mod.os, "access", lambda _path, _mode: False)
    assert "output_path_unavailable" in mod.preconditions_checked(
        malformed,
        marker_found=True,
        source_reachable=True,
        checked_at=START,
    )["failed_preconditions"]
    monkeypatch.undo()
    (malformed / mod.ROADMAP_RELATIVE_PATH).unlink()
    (malformed / mod.EXCLUSION_MANIFEST_RELATIVE_PATH).unlink()
    failed_without_hashes = mod.preconditions_checked(
        malformed,
        marker_found=True,
        source_reachable=True,
        checked_at=START,
    )["failed_preconditions"]
    assert "active_roadmap_hash_missing" in failed_without_hashes
    assert "exclusion_manifest_hash_missing" in failed_without_hashes

    for mutate, message in (
        (lambda a: a.pop("status"), "missing required"),
        (lambda a: a.update(status="done"), "invalid status"),
        (lambda a: a.update(honest_verdict="complete:"), "honest_verdict"),
        (lambda a: a.update(inference_substrate="live_llm_inference"), "substrate"),
        (lambda a: a.update(duration_s=-1), "duration"),
        (lambda a: a.update(search_finished_at=START), "timestamp"),
        (lambda a: a.update(source_queries_and_endpoint_receipts=[]), "source_queries"),
        (lambda a: a["source_queries_and_endpoint_receipts"].update(source_receipts=[]), "source_queries"),
        (
            lambda a: a["source_queries_and_endpoint_receipts"].update(
                source_receipts=["not-a-receipt"]
            ),
            "source receipt entries",
        ),
        (
            lambda a: a["source_queries_and_endpoint_receipts"]["source_receipts"][0].pop(
                "url"
            ),
            "source receipt missing url",
        ),
        (lambda a: a["accepted_rejected_abstained_findings"].update(all_candidates=[]), "all_candidates"),
        (
            lambda a: a.update(accepted_rejected_abstained_findings=[]),
            "accepted_rejected_abstained_findings",
        ),
        (
            lambda a: a["accepted_rejected_abstained_findings"].update(duplicate={}),
            "accepted_rejected_abstained_findings.duplicate",
        ),
        (
            lambda a: a["accepted_rejected_abstained_findings"].update(
                duplicate=["not-a-finding"]
            ),
            "finding classification entries",
        ),
        (
            lambda a: a["accepted_rejected_abstained_findings"]["rejected"][0].update(
                classification="accepted"
            ),
            "invalid finding classification",
        ),
        (
            lambda a: a["accepted_rejected_abstained_findings"]["rejected"][0].pop(
                "reason"
            ),
            "finding provenance field missing",
        ),
        (
            lambda a: a["references_append_receipt"].update(accepted_count=1),
            "references append accepted count",
        ),
        (lambda a: a.update(field_provenance="not-a-map"), "field_provenance"),
        (lambda a: a["field_provenance"].pop("status"), "field_provenance"),
        (lambda a: a.update(primary_secondary_and_official_source_counts={"primary": 0}), "source counts"),
        (
            lambda a: a.update(primary_secondary_and_official_source_counts=[]),
            "source counts",
        ),
        (
            lambda a: a.update(task_identity_and_gate_immutability=[]),
            "task_identity_and_gate_immutability",
        ),
        (lambda a: a["task_identity_and_gate_immutability"].update(gates_unchanged=False), "gates"),
        (lambda a: a["protected_files_unchanged"].update(all_unchanged=False), "protected"),
        (
            lambda a: a.update(
                false_positive_false_negative_and_cutoff_receipts={"principle": "wrong"}
            ),
            "false-positive/cutoff",
        ),
        (
            lambda a: a.update(duplicate_and_retired_scope_filter=[]),
            "duplicate_and_retired_scope_filter",
        ),
        (
            lambda a: a["duplicate_and_retired_scope_filter"].update(
                accepted_reopens_retired_scope_count=1
            ),
            "retired scope",
        ),
        (
            lambda a: a["references_append_receipt"].update(appended=True),
            "zero accepted",
        ),
    ):
        broken = json.loads(json.dumps(artifact))
        mutate(broken)
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(broken)

    accepted_missing_hook = _accepted_fixture()
    accepted_missing_hook.pop("source_hook")
    with pytest.raises(ValueError, match="accepted finding missing source_hook"):
        mod._validate_finding(accepted_missing_hook, "accepted")  # noqa: SLF001

    accepted_bad_mapping = _accepted_fixture()
    accepted_bad_mapping["method_to_task_mapping"]["target_experiment"] = (
        "exp5908-verisynth-constraint-fixture"
    )
    with pytest.raises(ValueError, match="method-to-task mapping"):
        mod._validate_finding(accepted_bad_mapping, "accepted")  # noqa: SLF001

    commands, exit_codes = mod._load_tests_run(None)  # noqa: SLF001
    assert commands
    assert set(commands) == set(exit_codes)

    broken = json.loads(json.dumps(artifact))
    broken["honest_verdict"] = "complete_null: tampered"
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
            "experiment_5906_v525_source_delta_ingestion",
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
            "carnot.experiment_5906_v525_source_delta_ingestion",
            run_name="__main__",
        )
    assert exc_info.value.code == 0
    assert mod.RESULT_RELATIVE_PATH.as_posix() in capsys.readouterr().out

    with pytest.raises(SystemExit) as missing_flag:
        mod.main(
            [
                "--root",
                str(root),
                "--search-started-at",
                START,
                "--search-finished-at",
                FINISH,
            ]
        )
    assert "--zero-findings" in str(missing_flag.value)
