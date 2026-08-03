"""Tests for Exp5962 V528 source-delta ingestion.

Spec refs: REQ-REPORT-5962, SCENARIO-REPORT-5962-ZERO-FINDING,
SCENARIO-REPORT-5962-ACCEPT-BOUNDED-DELTA,
SCENARIO-REPORT-5962-SOURCE-UNCERTAINTY,
SCENARIO-REPORT-5962-DUPLICATE-AND-RETIRED-SCOPE,
SCENARIO-REPORT-5962-SCHEMA.
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest
import yaml

from carnot import experiment_5962_v528_source_delta_ingestion as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"
START = "2026-08-03T14:00:00Z"
FINISH = "2026-08-03T14:11:00Z"


def _planner_references() -> str:
    return (
        "## V528 Planner Refresh - 20260726\n\n"
        "- **HIDE and Seek: Detecting Hallucinations in Language Models via "
        "Decoupled Representations** - arXiv:2506.17748.\n"
        "- **Learning Tractable Distributions of Language Model Continuations** "
        "- arXiv:2511.16054.\n"
        "- **A Probabilistic Neuro-symbolic Layer for Algebraic Constraint "
        "Satisfaction** - arXiv:2503.19466.\n"
        "- **Monotonic Kolmogorov-Arnold Networks** - arXiv:2606.17886.\n"
        "<!-- V528-PLANNER-REFRESH-20260726-END -->\n"
    )


def _roadmap() -> str:
    tasks = [
        {
            "id": mod.EXPERIMENT_ID,
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
            "requires_gpu": False,
        }
        if task_id == "exp5964-sota-atom-compatibility-corpus":
            row["gated_on"] = [
                {
                    "upstream": "exp5963-exact-atom-pair-fixture",
                    "artifact_field": "fixture_ready",
                    "op": "==",
                    "value": True,
                }
            ]
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
        "# vNEXT\n\n**Milestone:** 2026.07.528\n",
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
        "- id: generated_ir_closed\n"
        "  reason: generated-IR remains closed\n"
        "- id: schema_reprompt_closed\n"
        "  reason: schema reprompt remains closed\n"
        "- id: finite_id_closed\n"
        "  reason: finite-ID remains closed\n"
        "- id: external_text_logprob_closed\n"
        "  reason: external-text/logprob scorer remains closed\n"
        "- id: final_embedding_mmlu_closed\n"
        "  reason: final-embedding MMLU remains closed\n"
        "- id: kan_mutation_closed\n"
        "  reason: KAN mutation remains closed\n"
        "- id: public_arc_closed\n"
        "  reason: public ARC solves remain closed\n"
        "- id: board_probe_closed\n"
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
        mod.PRIOR_SOURCE_RESULT_RELATIVE_PATH,
    ):
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{rel_path.as_posix()} fixture\n", encoding="utf-8")
    (root / "results").mkdir(parents=True, exist_ok=True)
    return root


def _accepted_fixture() -> mod.JsonDict:
    return {
        "source_id": "post_v528_fixture_2608_00001",
        "classification": "accepted",
        "decision_bucket": "accepted",
        "title": "Post-V528 Fixture for Exact Atom Compatibility",
        "url": "https://arxiv.org/abs/2608.00001",
        "identifier": "2608.00001",
        "authors": ["Fixture Author"],
        "publication_date": "2026-08-02",
        "source_date": "2026-08-02",
        "search_timestamp": START,
        "receipt_id": "arxiv_fixture_post_v528",
        "query_family": "arxiv_primary",
        "query": 'all:"exact atom compatibility"',
        "access_outcome": "reachable_fixture_primary",
        "target_experiment": "exp5964-sota-atom-compatibility-corpus",
        "source_hook": "Add a bounded exact-atom representation compatibility control.",
        "authority_boundary": "Sharpens Exp5964 controls only; exact labels remain authoritative.",
        "post_marker_or_newer_primary_source": True,
        "primary_source": True,
        "duplicate_of_existing_reference": False,
        "reopens_retired_scope": False,
        "new_mechanism_or_material_change": True,
        "method_to_task_mapping": {
            "method": "exact_atom_representation_compatibility_control",
            "target_experiment": "exp5964-sota-atom-compatibility-corpus",
            "task_hook": "bounded exact-atom compatibility control",
            "failure_boundary": "reject on norm, length, label, or raw-model shortcuts",
        },
        "reason": "New primary fixture stays inside an allocated .528 task.",
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


def test_req_report_5962_spec_declares_v528_source_refresh_contract() -> None:
    """REQ-REPORT-5962: OpenSpec names the V528 source-refresh contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-5962") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-REPORT-5962",
        "SCENARIO-REPORT-5962-ZERO-FINDING",
        "SCENARIO-REPORT-5962-ACCEPT-BOUNDED-DELTA",
        "SCENARIO-REPORT-5962-SOURCE-UNCERTAINTY",
        "SCENARIO-REPORT-5962-DUPLICATE-AND-RETIRED-SCOPE",
        "SCENARIO-REPORT-5962-SCHEMA",
        str(mod.RESULT_RELATIVE_PATH),
        mod.PLANNER_MARKER,
        mod.INFERENCE_SUBSTRATE,
        "`semantic_scholar_ebt_and_arm_ebm_receipts`",
        "`openreview_huggingface_github_extropic_and_kona_receipts`",
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_5962_zero_delta_keeps_references_unchanged(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5962-ZERO-FINDING: zero accepted deltas are complete."""

    root = _make_repo(tmp_path, _planner_references())
    before = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")

    artifact = mod.build_and_write_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_findings=[],
        duration_s=660.0,
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
    assert artifact["accepted_rejected_abstained_findings"]["accepted"] == []
    assert artifact["accepted_rejected_abstained_findings"]["all_candidates"] == (
        _ordered_candidates(artifact)
    )
    uncertainty = artifact[
        "false_positive_false_negative_cutoff_and_rate_limit_receipts"
    ]
    assert uncertainty["false_positive_source_decisions"]
    assert uncertainty["known_false_negative_source_decisions"]
    assert uncertainty["cutoff_confounds"]
    assert uncertainty["endpoint_failed_source_decisions"]
    assert isinstance(uncertainty["rate_limit_receipts"], list)
    counts = artifact["primary_secondary_and_official_source_counts"]
    assert all(counts[key] >= 1 for key in ("primary", "secondary", "official", "tooling"))
    semantic = artifact["semantic_scholar_ebt_and_arm_ebm_receipts"]
    assert semantic["ebt_arxiv_id"] == "2507.02092"
    assert semantic["arm_ebm_arxiv_id"] == "2512.15605"
    grouped = artifact["openreview_huggingface_github_extropic_and_kona_receipts"]
    assert grouped["openreview_receipts"]
    assert grouped["huggingface_receipts"]
    assert grouped["github_receipts"]
    assert grouped["extropic_receipts"]
    assert grouped["kona_or_aleph_receipts"]
    assert artifact["preconditions_checked"]["research_roadmap_next_read"] is False
    assert artifact["search_window_and_marker_receipt"]["boundary_marker"] == (
        mod.PLANNER_MARKER
    )
    assert artifact["source_queries_and_endpoint_receipts"]["endpoint_failures"]
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


def test_scenario_report_5962_accepted_delta_appends_once(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5962-ACCEPT-BOUNDED-DELTA: accepted deltas map exactly."""

    root = _make_repo(tmp_path, _planner_references(), with_next=True)

    artifact = mod.build_and_write_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_findings=[_accepted_fixture()],
        duration_s=660.0,
    )

    mod.validate_artifact(artifact)
    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(
        encoding="utf-8"
    )
    assert references.count(mod.EXECUTION_DELTA_HEADING) == 1
    assert "Post-V528 Fixture" in references
    assert "exp5964-sota-atom-compatibility-corpus" in references
    assert artifact["honest_verdict"].startswith("complete_delta:")
    assert artifact["references_append_receipt"]["appended"] is True
    assert artifact["references_append_receipt"]["accepted_count"] == 1
    assert artifact["references_append_receipt"]["heading"] == mod.EXECUTION_DELTA_HEADING

    second = mod.build_and_write_artifact(
        root=root,
        search_started_at="2026-08-03T14:12:00Z",
        search_finished_at="2026-08-03T14:13:00Z",
        accepted_findings=[_accepted_fixture()],
        duration_s=60.0,
    )
    references_second = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(
        encoding="utf-8"
    )
    assert references_second.count(mod.EXECUTION_DELTA_HEADING) == 1
    assert second["references_append_receipt"]["appended"] is False


def test_scenario_report_5962_source_uncertainty_and_retired_filters(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5962-SOURCE-UNCERTAINTY: uncertainty remains explicit."""

    root = _make_repo(tmp_path, _planner_references())
    artifact = mod.build_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_findings=[_accepted_fixture()],
        duration_s=660.0,
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
    filters = artifact["duplicate_and_retired_scope_filter"]
    for rule in (
        "generated-IR",
        "schema-reprompt",
        "finite-ID",
        "external-text/logprob scorer",
        "final-embedding MMLU",
        "KAN mutation",
        "public ARC solve",
        "unchanged board-probe",
    ):
        assert rule in filters["retired_scope_rules"]
    assert filters["accepted_reopens_retired_scope_count"] == 0

    for changed_field, expected_message in (
        ("post_marker_or_newer_primary_source", "newer primary-source"),
        ("primary_source", "primary-source"),
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
    old["source_date"] = "2026-07-26"
    old["materially_changed_after_marker"] = False
    with pytest.raises(ValueError, match="newer primary-source"):
        mod._validate_finding(old, "accepted")  # noqa: SLF001

    wrong_target = _accepted_fixture()
    wrong_target["target_experiment"] = "exp5962-v528-source-delta-ingestion"
    wrong_target["method_to_task_mapping"]["target_experiment"] = wrong_target[
        "target_experiment"
    ]
    with pytest.raises(ValueError, match="allocated .528 experiment"):
        mod._validate_finding(wrong_target, "accepted")  # noqa: SLF001

    bad_mapping = _accepted_fixture()
    bad_mapping["method_to_task_mapping"]["target_experiment"] = (
        "exp5965-portable-atom-energy-ranker"
    )
    with pytest.raises(ValueError, match="method-to-task mapping"):
        mod._validate_finding(bad_mapping, "accepted")  # noqa: SLF001

    missing_mapping_hook = _accepted_fixture()
    missing_mapping_hook["method_to_task_mapping"].pop("task_hook")
    with pytest.raises(ValueError, match="method-to-task mapping missing task_hook"):
        mod._validate_finding(missing_mapping_hook, "accepted")  # noqa: SLF001


def test_scenario_report_5962_blocked_preconditions_and_schema_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-REPORT-5962-SCHEMA: helpers, schema, checksum, and CLI hold."""

    root = _make_repo(tmp_path, "## V528 Planner Refresh - 20260726\nno marker\n")
    artifact = mod.build_and_write_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_findings=[_accepted_fixture()],
        duration_s=660.0,
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["accepted_rejected_abstained_findings"]["accepted"] == []
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
        duration_s=660.0,
    )
    mod.validate_artifact(unreachable)
    assert "source_reachability_failed" in unreachable["preconditions_checked"][
        "failed_preconditions"
    ]

    root = _make_repo(tmp_path / "helpers", _planner_references())
    assert mod.read_text_if_present(root / "missing.md") == ""
    assert mod.path_sha256(root / "missing.md") is None
    assert mod.normalize_timestamp("2026-08-03T14:00:00+00:00").endswith("Z")
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
    assert "spec_req_report_5962_missing" in mod.preconditions_checked(
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
        duration_s=660.0,
    )
    mod.validate_artifact(artifact)
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
            "source receipt missing url",
        ),
        (
            lambda a: a["accepted_rejected_abstained_findings"].update(
                all_candidates=[]
            ),
            "all_candidates",
        ),
        (
            lambda a: a["accepted_rejected_abstained_findings"]["rejected"][0].update(
                classification="accepted"
            ),
            "invalid finding classification",
        ),
        (
            lambda a: a["references_append_receipt"].update(accepted_count=1),
            "references append accepted count",
        ),
        (lambda a: a.update(field_provenance="not-a-map"), "field_provenance"),
        (
            lambda a: a["field_provenance"]["status"].update(principle="wrong"),
            "field_provenance principle",
        ),
        (
            lambda a: a.update(primary_secondary_and_official_source_counts={"primary": 0}),
            "source counts",
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
            lambda a: a.update(semantic_scholar_ebt_and_arm_ebm_receipts={}),
            "semantic scholar",
        ),
        (
            lambda a: a.update(
                openreview_huggingface_github_extropic_and_kona_receipts={}
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
            lambda a: a["task_identity_gate_and_exclusion_immutability"].update(
                task_ids_unchanged=False
            ),
            "task ids",
        ),
        (
            lambda a: a["protected_files_unchanged"].update(all_unchanged=False),
            "protected",
        ),
        (lambda a: a.update(reproducibility_checksum="wrong"), "checksum"),
    ):
        broken = json.loads(json.dumps(artifact))
        mutate(broken)
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(broken)

    commands, exit_codes = mod._load_tests_run(None)  # noqa: SLF001
    assert commands
    assert set(commands) == set(exit_codes)
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
            "experiment_5962_v528_source_delta_ingestion",
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
            "carnot.experiment_5962_v528_source_delta_ingestion",
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
