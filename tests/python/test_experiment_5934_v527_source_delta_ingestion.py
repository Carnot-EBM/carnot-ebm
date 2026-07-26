"""Tests for Exp5934 V527 source-delta ingestion.

Spec refs: REQ-REPORT-5934, SCENARIO-REPORT-5934-ZERO-FINDING,
SCENARIO-REPORT-5934-ACCEPT-BOUNDED-DELTA,
SCENARIO-REPORT-5934-SOURCE-UNCERTAINTY,
SCENARIO-REPORT-5934-DUPLICATE-AND-RETIRED-SCOPE,
SCENARIO-REPORT-5934-SCHEMA.
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest
import yaml

from carnot import experiment_5934_v527_source_delta_ingestion as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"
START = "2026-07-26T08:09:29Z"
FINISH = "2026-07-26T08:16:00Z"


def _planner_references() -> str:
    return (
        "## V527 Planner Refresh - 20260726\n\n"
        "- **Finite-Sample Coverage Audits for High-Recall Candidate "
        "Generation: Certification and Learning-Theoretic Design** - "
        "arXiv:2607.21480.\n"
        "- **Anatomy of a Sound Neural Reasoner: One-Shot Amortization, "
        "First-Pass Poisoning, and Search Inertness in Clue-Rich "
        "Completion** - arXiv:2607.19635.\n"
        "- **UMEM: Unified Memory Extraction and Management Framework for "
        "Generalizable Memory** - ICML 2026 OpenReview.\n"
        "<!-- V527-PLANNER-REFRESH-20260726-END -->\n"
    )


def _roadmap() -> str:
    tasks = [
        {
            "id": "exp5934-v527-source-delta-ingestion",
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
        if task_id == "exp5936-sota-atomic-support-union-ab":
            row["gated_on"] = [
                {
                    "upstream": "exp5935-non-pruning-atomic-constraint-support",
                    "artifact_field": "atomic_support_contract_ready_score",
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
        "# vNEXT\n\n**Milestone:** 2026.07.527\n",
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
        "- id: schema_reprompt_closed\n"
        "  reason: schema reprompt remains closed\n"
        "- id: exact_diagnostic_reprompt_closed\n"
        "  reason: exact-diagnostic reprompt remains closed\n"
        "- id: finite_id_transport_closed\n"
        "  reason: finite-ID transport remains closed\n"
        "- id: external_scorer_closed\n"
        "  reason: external scorers remain closed\n"
        "- id: kan_mutation_closed\n"
        "  reason: KAN mutation remains closed\n"
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
        mod.RESEARCH_PROGRAM_RELATIVE_PATH,
        mod.RESEARCH_STUDYING_RELATIVE_PATH,
        mod.KNOWN_ISSUES_RELATIVE_PATH,
        mod.CONDUCTOR_RELATIVE_PATH,
        mod.SWEEP_CLUSTERS_RELATIVE_PATH,
        mod.SWEEP_SEMSCHOLAR_RELATIVE_PATH,
        mod.STATUS_RELATIVE_PATH,
        mod.CHANGELOG_RELATIVE_PATH,
        mod.TRACEABILITY_RELATIVE_PATH,
        mod.PRIOR_SOURCE_RESULT_RELATIVE_PATH,
    ):
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{rel_path.as_posix()} fixture\n", encoding="utf-8")
    (root / "results").mkdir(parents=True, exist_ok=True)
    return root


def _accepted_fixture() -> mod.JsonDict:
    return {
        "source_id": "post_v527_fixture_2607_99998",
        "classification": "accepted",
        "decision_bucket": "accepted",
        "title": "Post-V527 Fixture for Non-Pruning Atomic Support",
        "url": "https://arxiv.org/abs/2607.99998",
        "identifier": "2607.99998",
        "authors": ["Fixture Author"],
        "publication_date": "2026-07-26",
        "source_date": "2026-07-26",
        "search_timestamp": START,
        "receipt_id": "arxiv_fixture_post_v527",
        "query_family": "arxiv_primary",
        "query": 'all:"non-pruning atomic support"',
        "access_outcome": "reachable_fixture_primary",
        "target_experiment": "exp5935-non-pruning-atomic-constraint-support",
        "source_hook": "Add a bounded non-pruning support coverage control.",
        "authority_boundary": (
            "Sharpens Exp5935 controls only; exact fixture semantics remain "
            "the release authority."
        ),
        "post_marker_or_newer_primary_source": True,
        "primary_source": True,
        "duplicate_of_existing_reference": False,
        "reopens_retired_scope": False,
        "method_to_task_mapping": {
            "method": "non_pruning_atomic_support_coverage_control",
            "target_experiment": "exp5935-non-pruning-atomic-constraint-support",
            "task_hook": "bounded non-pruning support coverage control",
            "failure_boundary": "reject on omitted exact atom or protected-prefix loss",
        },
        "reason": "New primary fixture stays inside an allocated .527 task.",
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
        + classes["endpoint_failed"]
        + classes["duplicate"]
        + classes["retired_scope"]
    )


def test_req_report_5934_spec_declares_v527_source_refresh_contract() -> None:
    """REQ-REPORT-5934: OpenSpec names the V527 source-refresh contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-5934") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-REPORT-5934",
        "SCENARIO-REPORT-5934-ZERO-FINDING",
        "SCENARIO-REPORT-5934-ACCEPT-BOUNDED-DELTA",
        "SCENARIO-REPORT-5934-SOURCE-UNCERTAINTY",
        "SCENARIO-REPORT-5934-DUPLICATE-AND-RETIRED-SCOPE",
        "SCENARIO-REPORT-5934-SCHEMA",
        str(mod.RESULT_RELATIVE_PATH),
        mod.PLANNER_MARKER,
        mod.INFERENCE_SUBSTRATE,
        "`semantic_scholar_ebt_and_arm_ebm_receipts`",
        "`extropic_github_huggingface_openreview_and_kona_receipts`",
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_5934_zero_delta_keeps_references_unchanged(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5934-ZERO-FINDING: zero accepted deltas are complete."""

    root = _make_repo(tmp_path, _planner_references())
    before = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")

    artifact = mod.build_and_write_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_findings=[],
        duration_s=391.0,
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
    uncertainty = artifact[
        "false_positive_false_negative_cutoff_and_rate_limit_receipts"
    ]
    assert uncertainty["false_positive_source_decisions"]
    assert uncertainty["known_false_negative_source_decisions"]
    assert uncertainty["cutoff_confounds"]
    assert uncertainty["endpoint_failed_source_decisions"]
    assert uncertainty["rate_limit_receipts"]
    counts = artifact["primary_secondary_and_official_source_counts"]
    assert counts["primary"] >= 1
    assert counts["secondary"] >= 1
    assert counts["official"] >= 1
    assert counts["tooling"] >= 1
    assert artifact["semantic_scholar_ebt_and_arm_ebm_receipts"][
        "ebt_visible_citation_count"
    ] == 20
    assert artifact["semantic_scholar_ebt_and_arm_ebm_receipts"][
        "arm_ebm_visible_citation_count"
    ] == 8
    grouped = artifact["extropic_github_huggingface_openreview_and_kona_receipts"]
    assert grouped["extropic_receipts"]
    assert grouped["github_receipts"]
    assert grouped["huggingface_receipts"]
    assert grouped["openreview_receipts"]
    assert grouped["kona_or_aleph_receipts"]
    assert artifact["preconditions_checked"]["research_roadmap_next_read"] is False
    assert artifact["search_window_and_marker_receipt"]["boundary_marker"] == (
        mod.PLANNER_MARKER
    )
    assert artifact["source_queries_and_endpoint_receipts"]["endpoint_failures"]
    assert artifact["source_queries_and_endpoint_receipts"]["rate_limits"]
    assert artifact["task_identity_gate_and_exclusion_immutability"][
        "task_ids_unchanged"
    ] is True
    assert artifact["task_identity_gate_and_exclusion_immutability"][
        "gates_unchanged"
    ] is True
    assert artifact["task_identity_gate_and_exclusion_immutability"][
        "exclusions_unchanged"
    ] is True
    assert artifact["protected_files_unchanged"]["all_unchanged"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE


def test_scenario_report_5934_accepted_delta_appends_once(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5934-ACCEPT-BOUNDED-DELTA: accepted deltas map exactly."""

    root = _make_repo(tmp_path, _planner_references(), with_next=True)

    artifact = mod.build_and_write_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_findings=[_accepted_fixture()],
        duration_s=391.0,
    )

    mod.validate_artifact(artifact)
    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(
        encoding="utf-8"
    )
    assert references.count(mod.EXECUTION_DELTA_HEADING) == 1
    assert "Post-V527 Fixture" in references
    assert "exp5935-non-pruning-atomic-constraint-support" in references
    assert artifact["honest_verdict"].startswith("complete_delta:")
    assert artifact["references_append_receipt"]["appended"] is True
    assert artifact["references_append_receipt"]["accepted_count"] == 1
    assert artifact["references_append_receipt"]["heading"] == mod.EXECUTION_DELTA_HEADING
    assert artifact["accepted_rejected_abstained_findings"]["accepted"][0][
        "target_experiment"
    ] == "exp5935-non-pruning-atomic-constraint-support"
    assert artifact["accepted_rejected_abstained_findings"]["all_candidates"] == (
        _ordered_candidates(artifact)
    )

    second = mod.build_and_write_artifact(
        root=root,
        search_started_at="2026-07-26T08:17:00Z",
        search_finished_at="2026-07-26T08:18:00Z",
        accepted_findings=[_accepted_fixture()],
        duration_s=60.0,
    )
    references_second = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(
        encoding="utf-8"
    )
    assert references_second.count(mod.EXECUTION_DELTA_HEADING) == 1
    assert second["references_append_receipt"]["appended"] is False


def test_scenario_report_5934_source_uncertainty_and_retired_filters(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5934-SOURCE-UNCERTAINTY: uncertainty remains explicit."""

    root = _make_repo(tmp_path, _planner_references())
    artifact = mod.build_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_findings=[_accepted_fixture()],
        duration_s=391.0,
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
    assert {row["classification"] for row in classes["endpoint_failed"]} == {
        "endpoint_failed"
    }
    assert artifact["false_positive_false_negative_cutoff_and_rate_limit_receipts"][
        "principle"
    ] == mod.REQUIRED_FIELD_PRINCIPLES[
        "false_positive_false_negative_cutoff_and_rate_limit_receipts"
    ]
    filters = artifact["duplicate_and_retired_scope_filter"]
    assert filters["duplicate_dimensions"] == [
        "identifier",
        "title",
        "authors",
        "mechanism",
        "existing_reference_heading",
    ]
    for rule in (
        "schema reprompt",
        "exact-diagnostic reprompt",
        "finite-ID transport",
        "external scorers",
        "KAN mutation",
        "public ARC solves",
        "unchanged board probes",
    ):
        assert rule in filters["retired_scope_rules"]
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
    old["source_date"] = "2026-07-25"
    with pytest.raises(ValueError, match="newer primary-source"):
        mod._validate_finding(old, "accepted")  # noqa: SLF001

    wrong_target = _accepted_fixture()
    wrong_target["target_experiment"] = "exp5934-v527-source-delta-ingestion"
    wrong_target["method_to_task_mapping"]["target_experiment"] = wrong_target[
        "target_experiment"
    ]
    with pytest.raises(ValueError, match="allocated .527 experiment"):
        mod._validate_finding(wrong_target, "accepted")  # noqa: SLF001

    missing_hook = _accepted_fixture()
    missing_hook.pop("source_hook")
    with pytest.raises(ValueError, match="accepted finding missing source_hook"):
        mod._validate_finding(missing_hook, "accepted")  # noqa: SLF001

    bad_mapping = _accepted_fixture()
    bad_mapping["method_to_task_mapping"]["target_experiment"] = (
        "exp5936-sota-atomic-support-union-ab"
    )
    with pytest.raises(ValueError, match="method-to-task mapping"):
        mod._validate_finding(bad_mapping, "accepted")  # noqa: SLF001

    missing_mapping = _accepted_fixture()
    missing_mapping["method_to_task_mapping"] = None
    with pytest.raises(ValueError, match="method-to-task mapping"):
        mod._validate_finding(missing_mapping, "accepted")  # noqa: SLF001

    missing_mapping_hook = _accepted_fixture()
    missing_mapping_hook["method_to_task_mapping"].pop("task_hook")
    with pytest.raises(ValueError, match="method-to-task mapping missing task_hook"):
        mod._validate_finding(missing_mapping_hook, "accepted")  # noqa: SLF001


def test_scenario_report_5934_blocked_preconditions(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5934-SCHEMA: missing marker or routes fail closed."""

    root = _make_repo(tmp_path, "## V527 Planner Refresh - 20260726\nno marker\n")
    artifact = mod.build_and_write_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_findings=[_accepted_fixture()],
        duration_s=391.0,
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
                "source_cutoff": "changed_after_2026-07-26",
                "receipt_summary": "down",
            }
        ],
        duration_s=391.0,
    )
    mod.validate_artifact(unreachable)
    assert unreachable["status"] == "blocked"
    assert "source_reachability_failed" in unreachable["preconditions_checked"][
        "failed_preconditions"
    ]


def test_scenario_report_5934_schema_helpers_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-REPORT-5934-SCHEMA: helpers, schema, checksum, and CLI hold."""

    root = _make_repo(tmp_path, _planner_references())
    assert mod.read_text_if_present(root / "missing.md") == ""
    assert mod.path_sha256(root / "missing.md") is None
    assert mod.normalize_timestamp("2026-07-26T08:09:29+00:00").endswith("Z")
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
        duration_s=391.0,
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
    assert "spec_req_report_5934_missing" in mod.preconditions_checked(
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
        (
            lambda a: a["source_queries_and_endpoint_receipts"].update(
                source_receipts=[]
            ),
            "source_queries",
        ),
        (
            lambda a: a["source_queries_and_endpoint_receipts"].update(
                source_receipts=["not-a-receipt"]
            ),
            "source receipt entries",
        ),
        (
            lambda a: a["source_queries_and_endpoint_receipts"]["source_receipts"][
                0
            ].pop("url"),
            "source receipt missing url",
        ),
        (
            lambda a: a["source_queries_and_endpoint_receipts"]["source_receipts"][
                0
            ].pop("source_cutoff"),
            "source receipt missing source_cutoff",
        ),
        (
            lambda a: a["source_queries_and_endpoint_receipts"].update(
                endpoint_failures={}
            ),
            "endpoint failures",
        ),
        (
            lambda a: a["source_queries_and_endpoint_receipts"].update(rate_limits={}),
            "rate limits",
        ),
        (
            lambda a: a["accepted_rejected_abstained_findings"].update(
                all_candidates=[]
            ),
            "all_candidates",
        ),
        (
            lambda a: a.update(accepted_rejected_abstained_findings=[]),
            "accepted_rejected_abstained_findings",
        ),
        (
            lambda a: a["accepted_rejected_abstained_findings"].update(
                endpoint_failed={}
            ),
            "accepted_rejected_abstained_findings.endpoint_failed",
        ),
        (
            lambda a: a["accepted_rejected_abstained_findings"].update(
                endpoint_failed=["not-a-finding"]
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
            lambda a: a["accepted_rejected_abstained_findings"]["rejected"][0].update(
                decision_bucket="accepted"
            ),
            "invalid finding decision bucket",
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
        (
            lambda a: a.update(primary_secondary_and_official_source_counts={"primary": 0}),
            "source counts",
        ),
        (
            lambda a: a.update(
                primary_secondary_and_official_source_counts={
                    "primary": 0,
                    "secondary": 1,
                    "official": 1,
                    "tooling": 1,
                }
            ),
            "source counts",
        ),
        (
            lambda a: a.update(primary_secondary_and_official_source_counts=[]),
            "source counts",
        ),
        (
            lambda a: a["field_provenance"]["status"].update(principle="wrong"),
            "field_provenance principle",
        ),
        (
            lambda a: a.update(task_identity_gate_and_exclusion_immutability=[]),
            "task_identity_gate_and_exclusion_immutability",
        ),
        (
            lambda a: a["task_identity_gate_and_exclusion_immutability"].update(
                task_ids_unchanged=False
            ),
            "task ids",
        ),
        (
            lambda a: a["task_identity_gate_and_exclusion_immutability"].update(
                gates_unchanged=False
            ),
            "gates",
        ),
        (
            lambda a: a["task_identity_gate_and_exclusion_immutability"].update(
                exclusions_unchanged=False
            ),
            "exclusions",
        ),
        (
            lambda a: a["protected_files_unchanged"].update(all_unchanged=False),
            "protected",
        ),
        (
            lambda a: a.update(
                false_positive_false_negative_cutoff_and_rate_limit_receipts={
                    "principle": "wrong"
                }
            ),
            "false-positive/cutoff",
        ),
        (
            lambda a: a["false_positive_false_negative_cutoff_and_rate_limit_receipts"].update(
                false_positive_source_decisions={}
            ),
            "false-positive/cutoff",
        ),
        (
            lambda a: a.update(semantic_scholar_ebt_and_arm_ebm_receipts={}),
            "semantic scholar",
        ),
        (
            lambda a: a.update(
                extropic_github_huggingface_openreview_and_kona_receipts={}
            ),
            "official/discovery",
        ),
        (
            lambda a: a["extropic_github_huggingface_openreview_and_kona_receipts"].update(
                extropic_receipts=[]
            ),
            "official/discovery",
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

    accepted_artifact = mod.build_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_findings=[_accepted_fixture()],
        duration_s=391.0,
    )
    accepted_artifact["references_append_receipt"]["accepted_source_ids"] = ["wrong"]
    with pytest.raises(ValueError, match="references append accepted source ids"):
        mod.validate_artifact(accepted_artifact)

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
            "experiment_5934_v527_source_delta_ingestion",
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
            "carnot.experiment_5934_v527_source_delta_ingestion",
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
