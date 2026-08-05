"""Tests for Exp6144 V533 source-delta ingestion.

Spec refs: REQ-REPORT-6144, SCENARIO-REPORT-6144-ZERO-DELTA,
SCENARIO-REPORT-6144-ACCEPT-BOUNDED-DELTA,
SCENARIO-REPORT-6144-DUPLICATE-AND-RETIRED-SCOPE,
SCENARIO-REPORT-6144-SCHEMA.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_6144_v533_source_delta_ingestion as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"
START = "2026-08-05T10:40:16Z"
FINISH = "2026-08-05T10:43:07Z"


def _planner_references() -> str:
    return (
        "## V533 Planner Refresh - 20260805\n\n"
        "- **TOOD** - arXiv:2607.29592.\n"
        "- **A Framework for Stochastic Differentiable Programming** - "
        "arXiv:2608.01612.\n"
        "- **Thermalizing Stochastic Programs** - arXiv:2608.01615.\n"
        "<!-- V533-PLANNER-REFRESH-20260805-END -->\n"
    )


def _roadmap() -> str:
    tasks = [
        {
            "id": mod.EXPERIMENT_ID,
            "milestone": mod.MILESTONE,
            "title": "source refresh",
            "deliverable": mod.RESULT_RELATIVE_PATH.as_posix(),
        },
        {
            "id": "exp6145-constraint-shift-stream",
            "milestone": mod.MILESTONE,
            "title": "exact stream",
            "deliverable": "results/experiment_6145_constraint_shift_stream.json",
        },
        {
            "id": "exp6152-typed-stochastic-constraint-ir",
            "milestone": mod.MILESTONE,
            "title": "typed stochastic IR",
            "deliverable": "results/experiment_6152_typed_stochastic_constraint_ir.json",
            "gated_on": [
                {
                    "upstream": "exp6145-constraint-shift-stream",
                    "artifact_field": "constraint_shift_stream_ready_score",
                    "op": "==",
                    "value": 1.0,
                }
            ],
        },
    ]
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
        "# vNEXT\n\n### Exp6145\n### Exp6152\n### Exp6154\n",
        encoding="utf-8",
    )
    (root / "openspec/capabilities/research-reporting").mkdir(parents=True, exist_ok=True)
    (root / mod.SPEC_RELATIVE_PATH).write_text(
        "\n".join(mod.SPEC_REFS) + "\n",
        encoding="utf-8",
    )
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / mod.EXCLUSION_MANIFEST_RELATIVE_PATH).write_text(
        "retired_extras:\n"
        "- id: transport_reopen_closed\n"
        "  reason: transport reopen remains closed\n"
        "- id: phase_d_external_scorer_closed\n"
        "  reason: Phase-D external scorer remains closed\n"
        "- id: generated_answer_transport_closed\n"
        "  reason: generated-answer transport remains closed\n"
        "- id: csl_exact_slot_closed\n"
        "  reason: CSL exact-slot requalification remains closed\n"
        "- id: thrml_parity_closed\n"
        "  reason: THRML parity remains closed\n"
        "- id: kan_mutation_closed\n"
        "  reason: KAN mutation remains closed\n"
        "- id: arc_solve_closed\n"
        "  reason: ARC solve claims remain closed\n"
        "- id: unchanged_hardware_access_closed\n"
        "  reason: unchanged hardware access remains closed\n",
        encoding="utf-8",
    )
    for rel_path in (
        mod.AGENTS_RELATIVE_PATH,
        mod.CODEX_RELATIVE_PATH,
        mod.CLAUDE_RELATIVE_PATH,
        mod.RESEARCH_PROGRAM_RELATIVE_PATH,
        mod.RESEARCH_STUDYING_RELATIVE_PATH,
        mod.KNOWN_ISSUES_RELATIVE_PATH,
        mod.CONDUCTOR_LOG_RELATIVE_PATH,
        mod.SWEEP_CLUSTERS_RELATIVE_PATH,
        mod.SWEEP_SEMSCHOLAR_RELATIVE_PATH,
    ):
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{rel_path.as_posix()} fixture\n", encoding="utf-8")
    (root / "results").mkdir(parents=True, exist_ok=True)
    return root


def _accepted_fixture() -> mod.JsonDict:
    return {
        "source_id": "post_v533_fixture_typed_ir_control",
        "classification": "accepted",
        "decision_bucket": "accepted",
        "title": "Post-V533 Fixture for Typed Stochastic IR Control",
        "url": "https://arxiv.org/abs/2608.99998",
        "identifier": "2608.99998",
        "authors": ["Fixture Author"],
        "publication_date": "2026-08-06",
        "source_date": "2026-08-06",
        "search_timestamp": START,
        "receipt_id": "arxiv_v533_fixture_primary",
        "query_family": "arxiv_primary",
        "query": "certified strategy memory",
        "access_outcome": "reachable_fixture_primary_post_marker",
        "target_experiment": "exp6152-typed-stochastic-constraint-ir",
        "source_hook": "Add a bounded support-probability negative control.",
        "authority_boundary": "Sharpens Exp6152 controls only; no task or gate rewrite.",
        "post_marker_or_newer_primary_source": True,
        "materially_changed_after_marker": True,
        "primary_or_official_source": True,
        "duplicate_of_existing_reference": False,
        "reopens_retired_scope": False,
        "new_mechanism_or_material_change": True,
        "method_to_task_mapping": {
            "method": "strategy_memory_freshness_control",
            "target_experiment": "exp6152-typed-stochastic-constraint-ir",
            "task_hook": "bounded support-probability negative control",
            "failure_boundary": "defer if the control needs hardware access",
        },
        "reason": "New primary fixture stays inside an allocated V533 task.",
    }


def _ordered_candidates(artifact: mod.JsonDict) -> list[mod.JsonDict]:
    classes = artifact["accepted_rejected_duplicate_retired_and_abstained_findings"]
    return [row for bucket in mod.CLASSIFICATION_BUCKETS for row in classes[bucket]]


def test_req_report_6144_spec_declares_v533_source_refresh_contract() -> None:
    """REQ-REPORT-6144: OpenSpec names the V533 source-refresh contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-6144") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-REPORT-6144",
        "SCENARIO-REPORT-6144-ZERO-DELTA",
        "SCENARIO-REPORT-6144-ACCEPT-BOUNDED-DELTA",
        "SCENARIO-REPORT-6144-DUPLICATE-AND-RETIRED-SCOPE",
        "SCENARIO-REPORT-6144-SCHEMA",
        str(mod.RESULT_RELATIVE_PATH),
        mod.PLANNER_MARKER,
        mod.INFERENCE_SUBSTRATE,
        "`sota_to_experiment_mapping`",
        "`semantic_scholar_ebt_and_arm_ebm_receipts`",
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_6144_zero_delta_keeps_references_unchanged(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6144-ZERO-DELTA: zero accepted deltas are complete."""

    root = _make_repo(tmp_path, _planner_references())
    before = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")

    artifact = mod.build_and_write_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_findings=[],
        duration_s=171.0,
        test_commands=["unit"],
        test_exit_codes={"unit": 0},
    )

    mod.validate_artifact(artifact)
    after = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")

    assert after == before
    assert (root / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8").endswith("\n")
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete_null:")
    assert artifact["references_append_receipt"]["appended"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["search_window_and_marker_receipt"]["boundary_marker"] == (
        mod.PLANNER_MARKER
    )
    assert artifact["search_window_and_marker_receipt"]["same_day_ordering_uncertainty"]
    assert artifact["preconditions_checked"]["research_roadmap_next_read"] is False
    classes = artifact["accepted_rejected_duplicate_retired_and_abstained_findings"]
    assert classes["accepted"] == []
    assert classes["all_candidates"] == _ordered_candidates(artifact)
    assert classes["duplicate"]
    assert classes["retired_scope"]
    assert classes["endpoint_failed"]
    assert classes["cutoff_confound"]
    assert artifact["sota_to_experiment_mapping"]["accepted_count"] == 0
    assert artifact["sota_to_experiment_mapping"]["task_ids_mutated"] is False
    semantic = artifact["semantic_scholar_ebt_and_arm_ebm_receipts"]
    assert semantic["ebt_arxiv_id"] == "2507.02092"
    assert semantic["arm_ebm_arxiv_id"] == "2512.15605"
    assert semantic["direct_api_reachable"] in {True, False}
    grouped = artifact["openreview_huggingface_github_extropic_and_kona_receipts"]
    assert grouped["openreview_receipts"]
    assert grouped["huggingface_receipts"]
    assert grouped["github_receipts"]
    assert grouped["extropic_receipts"]
    assert grouped["kona_or_aleph_receipts"]
    uncertainty = artifact[
        "cutoff_rate_limit_and_same_day_uncertainty_receipts"
    ]
    assert uncertainty["rate_limit_receipts"]
    assert uncertainty["endpoint_failed_source_decisions"]
    assert uncertainty["same_day_ordering_uncertainty"] is True
    counts = artifact["primary_secondary_and_official_source_counts"]
    assert all(counts[key] >= 1 for key in ("primary", "secondary", "official", "tooling"))
    immutability = artifact["roadmap_identity_gate_and_exclusion_immutability"]
    assert immutability["task_ids_unchanged"] is True
    assert immutability["gates_unchanged"] is True
    assert immutability["exclusions_unchanged"] is True
    assert artifact["protected_files_unchanged"]["all_unchanged"] is True


def test_scenario_report_6144_accepted_delta_appends_once(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6144-ACCEPT-BOUNDED-DELTA: accepted deltas map exactly."""

    root = _make_repo(tmp_path, _planner_references(), with_next=True)

    artifact = mod.build_and_write_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_findings=[_accepted_fixture()],
        duration_s=171.0,
    )

    mod.validate_artifact(artifact)
    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(
        encoding="utf-8"
    )
    assert references.count(mod.EXECUTION_DELTA_HEADING) == 1
    assert "Post-V533 Fixture" in references
    assert "exp6152-typed-stochastic-constraint-ir" in references
    assert artifact["honest_verdict"].startswith("complete_delta:")
    assert artifact["references_append_receipt"]["appended"] is True
    assert artifact["references_append_receipt"]["accepted_count"] == 1
    assert artifact["sota_to_experiment_mapping"]["accepted_mappings"][0][
        "target_experiment"
    ] == "exp6152-typed-stochastic-constraint-ir"

    second = mod.build_and_write_artifact(
        root=root,
        search_started_at="2026-08-05T10:44:00Z",
        search_finished_at="2026-08-05T10:45:00Z",
        accepted_findings=[_accepted_fixture()],
        duration_s=60.0,
    )
    references_second = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(
        encoding="utf-8"
    )
    assert references_second.count(mod.EXECUTION_DELTA_HEADING) == 1
    assert second["references_append_receipt"]["appended"] is False


def test_scenario_report_6144_source_filters_and_mapping_validation(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6144-DUPLICATE-AND-RETIRED-SCOPE: filters are explicit."""

    root = _make_repo(tmp_path, _planner_references())
    artifact = mod.build_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_findings=[_accepted_fixture()],
        duration_s=171.0,
    )

    mod.validate_artifact(artifact)
    classes = artifact["accepted_rejected_duplicate_retired_and_abstained_findings"]
    for bucket in mod.CLASSIFICATION_BUCKETS:
        assert {row["classification"] for row in classes[bucket]} <= {bucket}
    filters = artifact["duplicate_and_retired_scope_filter"]
    for rule in (
        "retired Phase-D",
        "generated-answer",
        "CSL exact-slot",
        "THRML parity",
        "KAN mutation",
        "ARC",
        "unchanged hardware access",
    ):
        assert rule in filters["retired_scope_rules"]
    assert filters["accepted_reopens_retired_scope_count"] == 0

    for changed_field, expected_message in (
        ("post_marker_or_newer_primary_source", "newer primary-source"),
        ("primary_or_official_source", "primary or official"),
        ("duplicate_of_existing_reference", "duplicate"),
        ("reopens_retired_scope", "retired scope"),
        ("new_mechanism_or_material_change", "new mechanism"),
    ):
        candidate = _accepted_fixture()
        candidate[changed_field] = changed_field in {
            "duplicate_of_existing_reference",
            "reopens_retired_scope",
        }
        with pytest.raises(ValueError, match=expected_message):
            mod._validate_finding(candidate, "accepted")  # noqa: SLF001

    old = _accepted_fixture()
    old["source_date"] = "2026-08-05"
    old["materially_changed_after_marker"] = False
    with pytest.raises(ValueError, match="newer primary-source"):
        mod._validate_finding(old, "accepted")  # noqa: SLF001

    wrong_target = _accepted_fixture()
    wrong_target["target_experiment"] = "exp6144-v533-source-delta-ingestion"
    wrong_target["method_to_task_mapping"]["target_experiment"] = wrong_target[
        "target_experiment"
    ]
    with pytest.raises(ValueError, match="allocated .533 experiment or defer"):
        mod._validate_finding(wrong_target, "accepted")  # noqa: SLF001

    bad_mapping = _accepted_fixture()
    bad_mapping["method_to_task_mapping"]["target_experiment"] = "defer"
    with pytest.raises(ValueError, match="method-to-task mapping target mismatch"):
        mod._validate_finding(bad_mapping, "accepted")  # noqa: SLF001

    missing_mapping_hook = _accepted_fixture()
    missing_mapping_hook["method_to_task_mapping"].pop("task_hook")
    with pytest.raises(ValueError, match="method-to-task mapping missing task_hook"):
        mod._validate_finding(missing_mapping_hook, "accepted")  # noqa: SLF001

    defer = _accepted_fixture()
    defer["target_experiment"] = "defer"
    defer["method_to_task_mapping"]["target_experiment"] = "defer"
    mod._validate_finding(defer, "accepted")  # noqa: SLF001
    mod._validate_finding(classes["rejected"][0], "rejected")  # noqa: SLF001


def test_scenario_report_6144_blocked_preconditions_and_schema_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-REPORT-6144-SCHEMA: helpers, schema, and checksum hold."""

    root = _make_repo(tmp_path, "## V533 Planner Refresh - 20260805\nno marker\n")
    artifact = mod.build_and_write_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_findings=[_accepted_fixture()],
        duration_s=171.0,
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["accepted_rejected_duplicate_retired_and_abstained_findings"][
        "accepted"
    ] == []
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
                "receipt_id": "all_down",
                "source_family": "arXiv",
                "source_role": "primary",
                "query_family": "arxiv_primary",
                "query": "fixture",
                "url": "https://arxiv.org/",
                "accessed_at": START,
                "access_outcome": "inaccessible_timeout",
                "candidate_ids": [],
                "candidate_count": 0,
                "source_cutoff": "changed_after_v533_marker",
                "receipt_summary": "down",
            }
        ],
        duration_s=171.0,
    )
    mod.validate_artifact(unreachable)
    assert "source_reachability_failed" in unreachable["preconditions_checked"][
        "failed_preconditions"
    ]

    root = _make_repo(tmp_path / "helpers", _planner_references())
    assert mod.read_text_if_present(root / "missing.md") == ""
    assert mod.path_sha256(root / "missing.md") is None
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

    malformed = _make_repo(tmp_path / "malformed", _planner_references())
    (malformed / mod.ROADMAP_RELATIVE_PATH).write_text("- just\n- a list\n")
    assert "active_roadmap_identity_unavailable" in mod.preconditions_checked(
        malformed,
        marker_found=True,
        source_reachable=True,
        checked_at=START,
    )["failed_preconditions"]
    (malformed / mod.ROADMAP_RELATIVE_PATH).write_text("tasks: [\n", encoding="utf-8")
    assert "active_roadmap_identity_unavailable" in mod.preconditions_checked(
        malformed,
        marker_found=True,
        source_reachable=True,
        checked_at=START,
    )["failed_preconditions"]
    (malformed / mod.SPEC_RELATIVE_PATH).write_text("missing\n", encoding="utf-8")
    assert "spec_req_report_6144_missing" in mod.preconditions_checked(
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
    missing_hash_failures = mod.preconditions_checked(
        malformed,
        marker_found=True,
        source_reachable=True,
        checked_at=START,
    )["failed_preconditions"]
    assert "active_roadmap_hash_missing" in missing_hash_failures
    assert "exclusion_manifest_hash_missing" in missing_hash_failures

    artifact = mod.build_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        duration_s=171.0,
    )
    mod.validate_artifact(artifact)

    def break_rejected_classification(candidate: mod.JsonDict) -> None:
        classes = candidate[
            "accepted_rejected_duplicate_retired_and_abstained_findings"
        ]
        classes["rejected"][0]["classification"] = "accepted"
        classes["all_candidates"][0]["classification"] = "accepted"

    for mutate, message in (
        (lambda a: a.pop("status"), "missing required"),
        (lambda a: a.update(status="done"), "invalid status"),
        (lambda a: a.update(honest_verdict="complete:"), "honest_verdict"),
        (lambda a: a.update(inference_substrate="live_llm_inference"), "substrate"),
        (lambda a: a.update(duration_s=-1), "duration"),
        (lambda a: a.update(search_finished_at=START), "timestamp"),
        (lambda a: a.update(source_queries_and_endpoint_receipts=[]), "source_queries"),
        (
            lambda a: a["source_queries_and_endpoint_receipts"]["source_receipts"][
                0
            ].pop("url"),
            "source receipt",
        ),
        (break_rejected_classification, "classification"),
        (
            lambda a: a["field_provenance"].pop("status"),
            "missing provenance",
        ),
        (
            lambda a: a["field_principles"].pop("status"),
            "missing principle",
        ),
        (
            lambda a: a["accepted_rejected_duplicate_retired_and_abstained_findings"][
                "all_candidates"
            ].append({"classification": "accepted", "url": "https://example.com"}),
            "all_candidates",
        ),
    ):
        broken = mod._roundtrip(artifact)  # noqa: SLF001
        mutate(broken)
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(broken)

    cli_root = _make_repo(tmp_path / "cli", _planner_references())
    monkeypatch.setattr(mod, "REPO_ROOT", cli_root)
    assert mod.main() == 0
    assert "experiment_6144_v533_source_delta_ingestion.json" in capsys.readouterr().out
    mod.validate_artifact(
        mod._roundtrip(  # noqa: SLF001
            json.loads((cli_root / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
        )
    )
