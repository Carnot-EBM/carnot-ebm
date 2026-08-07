"""Tests for Exp6171 V535 source-delta ingestion.

Spec refs: REQ-REPORT-6171,
SCENARIO-REPORT-6171-EXACT-MARKER,
SCENARIO-REPORT-6171-BOUNDED-DATED-SOURCE,
SCENARIO-REPORT-6171-DEDUPLICATE-AND-GUARD-SCOPE,
SCENARIO-REPORT-6171-ZERO-DELTA,
SCENARIO-REPORT-6171-SCHEMA.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_6171_v535_source_delta_ingestion as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"
START = "2026-08-07T03:05:00Z"
FINISH = "2026-08-07T03:18:00Z"


def _references() -> str:
    return (
        "## V535 Planner Refresh - 20260806\n\n"
        "- **CCTU** - arXiv:2603.15309.\n"
        "- **CLUE** - arXiv:2510.01591.\n"
        "- **TrajSelector** - arXiv:2510.16449.\n"
        "<!-- V535-PLANNER-REFRESH-20260806-END -->\n"
    )


def _roadmap() -> str:
    tasks = [
        {
            "id": mod.EXPERIMENT_ID,
            "milestone": mod.MILESTONE,
            "title": "source refresh",
            "deliverable": mod.RESULT_RELATIVE_PATH.as_posix(),
        }
    ]
    for task_id in mod.ALLOCATED_TARGET_EXPERIMENTS:
        row = {
            "id": task_id,
            "milestone": mod.MILESTONE,
            "title": task_id,
            "deliverable": f"results/{task_id}.json",
        }
        if task_id.startswith("exp6174"):
            row["gated_on"] = [
                {
                    "upstream": "exp6173-cctu-item-bank-preregistration",
                    "artifact_field": "cctu_item_bank_ready_score",
                    "op": "==",
                    "value": 1.0,
                }
            ]
        tasks.append(row)
    return yaml.safe_dump({"milestone": mod.MILESTONE, "tasks": tasks}, sort_keys=False)


def _make_repo(root: Path, references_text: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).write_text(
        references_text,
        encoding="utf-8",
    )
    (root / mod.ROADMAP_RELATIVE_PATH).write_text(_roadmap(), encoding="utf-8")
    (root / "openspec/change-proposals").mkdir(parents=True, exist_ok=True)
    (root / mod.VNEXT_RELATIVE_PATH).write_text("Exp6173\nExp6181\n", encoding="utf-8")
    (root / mod.SPEC_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / mod.SPEC_RELATIVE_PATH).write_text("\n".join(mod.SPEC_REFS), encoding="utf-8")
    (root / mod.EXCLUSION_MANIFEST_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / mod.EXCLUSION_MANIFEST_RELATIVE_PATH).write_text(
        "retired_extras:\n"
        "- id: phase_d_external_text_scorer_retired_exp5163_v474\n"
        "  blocked_patterns: [output-only scorer]\n"
        "- id: kan_mutation_closed\n"
        "  blocked_patterns: [KAN mutation]\n"
        "- id: arc_outer_loop_solve_closed\n"
        "  blocked_patterns: [ARC solve]\n",
        encoding="utf-8",
    )
    for rel_path in (
        mod.AGENTS_RELATIVE_PATH,
        mod.CODEX_RELATIVE_PATH,
        mod.CLAUDE_RELATIVE_PATH,
        mod.RESEARCH_PROGRAM_RELATIVE_PATH,
        mod.KNOWN_ISSUES_RELATIVE_PATH,
        mod.STATUS_RELATIVE_PATH,
        mod.CONDUCTOR_RELATIVE_PATH,
    ):
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{rel_path.as_posix()} fixture\n", encoding="utf-8")
    (root / "results").mkdir(parents=True, exist_ok=True)
    return root


def _accepted_fixture() -> mod.JsonDict:
    return {
        "stable_id": "arxiv:2608.99998",
        "content_hash": "sha256:" + "1" * 64,
        "title": "Post-marker fixture for executable tool-trace calibration",
        "url": "https://arxiv.org/abs/2608.99998",
        "date": "2026-08-07",
        "authority": "arXiv",
        "source_kind": "primary",
        "local_reachability": "reachable_primary_fixture",
        "roadmap_task": "exp6173-cctu-item-bank-preregistration",
        "changed_method_or_gate": "adds a bounded calibration negative control",
        "retirement_conflict": "none",
        "reason": "Dated primary fixture stays inside Exp6173.",
        "post_marker": True,
        "dated_reproducible": True,
        "primary_or_first_party": True,
        "duplicate": False,
        "reopens_retired_scope": False,
        "reopens_completed_scope": False,
        "new_applicability": True,
    }


def test_req_report_6171_spec_declares_exact_marker_contract() -> None:
    """REQ-REPORT-6171: OpenSpec names the V535 ingestion contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-6171") :]

    for marker in (
        "REQ-REPORT-6171",
        "SCENARIO-REPORT-6171-EXACT-MARKER",
        "SCENARIO-REPORT-6171-BOUNDED-DATED-SOURCE",
        "SCENARIO-REPORT-6171-DEDUPLICATE-AND-GUARD-SCOPE",
        "SCENARIO-REPORT-6171-ZERO-DELTA",
        "SCENARIO-REPORT-6171-SCHEMA",
        "V535-PLANNER-REFRESH-20260806-END",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_report_6171_zero_delta_preserves_references(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6171-ZERO-DELTA: zero accepted deltas are complete."""

    root = _make_repo(tmp_path, _references())
    before = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_bytes()

    artifact = mod.build_and_write_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_candidates=[],
        duration_s=780.0,
        test_commands=["unit"],
        test_exit_codes={"unit": 0},
    )

    mod.validate_artifact(artifact)
    after = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_bytes()

    assert after == before
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete_null:")
    assert "accepted_count=0" in artifact["honest_verdict"]
    assert artifact["zero_delta_accepted"] is True
    marker = artifact["marker_text_count_hash_offset_and_reference_hash"]
    assert marker["marker_count"] == 1
    assert marker["marker_line"] == 6
    assert marker["post_marker_content_hash"].startswith("sha256:")
    assert marker["reference_hash_before"] == marker["reference_hash_after"]
    assert artifact["bounded_time_window"]["window_start_exclusive"] == ("2026-08-06T23:59:59Z")
    assert artifact["semantic_scholar_ebt_and_arm_ebm_counts"]["ebt_visible_count"] == 32
    assert artifact["semantic_scholar_ebt_and_arm_ebm_counts"]["arm_ebm_visible_count"] == 8
    assert artifact["candidate_and_deduplicated_record_counts"]["accepted_count"] == 0
    assert artifact["reference_hash_before_after_and_append_count"]["append_count"] == 0
    assert artifact["protected_files_unchanged"]["all_unchanged"] is True
    assert artifact["preconditions_checked"]["requested_missing_paths"] == [
        "research-roadmap-next.yaml",
        "scripts/reliable_source_delta_ingestion.py",
    ]
    dispositions = {
        row["disposition"] for row in artifact["accepted_rejected_and_guarded_delta_ledger"]
    }
    assert {"guarded", "rejected", "cutoff_confound", "endpoint_failed"} <= dispositions


def test_scenario_report_6171_acceptance_requires_date_authority_and_mapping(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6171-BOUNDED-DATED-SOURCE: accepted rows are bounded."""

    root = _make_repo(tmp_path, _references())
    artifact = mod.build_and_write_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_candidates=[_accepted_fixture()],
        duration_s=780.0,
    )

    mod.validate_artifact(artifact)
    references = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")
    assert references.count(mod.EXECUTION_DELTA_HEADING) == 1
    assert artifact["honest_verdict"].startswith("complete_delta:")
    assert artifact["zero_delta_accepted"] is False
    assert artifact["candidate_and_deduplicated_record_counts"]["accepted_count"] == 1
    assert artifact["roadmap_task_mapping"]["accepted_mappings"][0]["roadmap_task"] == (
        "exp6173-cctu-item-bank-preregistration"
    )

    rerun = mod.build_and_write_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_candidates=[_accepted_fixture()],
        duration_s=780.0,
    )
    references_second = (root / mod.RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8")
    assert references_second.count(mod.EXECUTION_DELTA_HEADING) == 1
    assert rerun["reference_hash_before_after_and_append_count"]["append_count"] == 0

    dupe_fixture = _accepted_fixture()
    dupe_fixture["title"] = "duplicate accepted fixture"
    dupe_artifact = mod.build_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        accepted_candidates=[_accepted_fixture(), dupe_fixture],
        duration_s=780.0,
    )
    assert dupe_artifact["candidate_and_deduplicated_record_counts"]["duplicate_count"] >= 1


def test_scenario_report_6171_deduplicate_and_guard_retired_scope() -> None:
    """SCENARIO-REPORT-6171-DEDUPLICATE-AND-GUARD-SCOPE: filters hold."""

    first = _accepted_fixture()
    duplicate = dict(first)
    duplicate["title"] = "same stable id duplicate"
    unique, duplicates = mod.deduplicate_candidates([first, duplicate])
    assert unique == [first]
    assert duplicates[0]["duplicate_of"] == "arxiv:2608.99998"

    for changed_field, expected_message in (
        ("post_marker", "post-marker"),
        ("dated_reproducible", "dated reproducible"),
        ("primary_or_first_party", "primary or first-party"),
        ("duplicate", "duplicate"),
        ("reopens_retired_scope", "retired scope"),
        ("reopens_completed_scope", "completed scope"),
        ("new_applicability", "changed applicability"),
    ):
        candidate = _accepted_fixture()
        candidate[changed_field] = changed_field in {
            "duplicate",
            "reopens_retired_scope",
            "reopens_completed_scope",
        }
        with pytest.raises(ValueError, match=expected_message):
            mod.validate_accepted_candidate(candidate)

    cutoff = _accepted_fixture()
    cutoff["date"] = "2026-08-06"
    with pytest.raises(ValueError, match="after the V535 marker"):
        mod.validate_accepted_candidate(cutoff)

    wrong_target = _accepted_fixture()
    wrong_target["roadmap_task"] = "exp6171-v535-source-delta-ingestion"
    with pytest.raises(ValueError, match="Exp6173-Exp6181"):
        mod.validate_accepted_candidate(wrong_target)

    guarded = mod.classify_candidate(
        {
            "stable_id": "github:DorskFR/cctui",
            "content_hash": "sha256:" + "2" * 64,
            "title": "CCTU name collision",
            "url": "https://github.com/DorskFR/cctui",
            "date": "2026-08-07",
            "authority": "GitHub",
            "source_kind": "secondary",
            "local_reachability": "reachable_metadata_only",
            "roadmap_task": "exp6173-cctu-item-bank-preregistration",
            "changed_method_or_gate": "none",
            "retirement_conflict": "merely renames CCTU without CCTU benchmark content",
            "reason": "Repository name collision is not source authority.",
            "post_marker": True,
            "dated_reproducible": True,
            "primary_or_first_party": False,
            "duplicate": False,
            "reopens_retired_scope": False,
            "reopens_completed_scope": False,
            "new_applicability": False,
        }
    )
    assert guarded["disposition"] == "guarded"

    endpoint = _accepted_fixture()
    endpoint["endpoint_failed"] = True
    assert mod.classify_candidate(endpoint)["disposition"] == "endpoint_failed"
    duplicate_candidate = _accepted_fixture()
    duplicate_candidate["duplicate"] = True
    assert mod.classify_candidate(duplicate_candidate)["disposition"] == "duplicate"
    same_day = _accepted_fixture()
    same_day["date"] = "2026-08-06"
    assert mod.classify_candidate(same_day)["disposition"] == "cutoff_confound"
    retired = _accepted_fixture()
    retired["reopens_retired_scope"] = True
    assert mod.classify_candidate(retired)["disposition"] == "guarded"
    assert mod.classify_candidate(_accepted_fixture())["disposition"] == "accepted"


def test_scenario_report_6171_blocked_schema_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-REPORT-6171-SCHEMA: helpers, schema, and CLI hold."""

    blocked_root = _make_repo(tmp_path / "blocked", "no marker\n")
    blocked = mod.build_artifact(
        root=blocked_root,
        search_started_at=START,
        search_finished_at=FINISH,
        duration_s=1.0,
    )
    mod.validate_artifact(blocked)
    assert blocked["status"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked:")
    assert (
        "v535_marker_missing_or_not_unique"
        in blocked["preconditions_checked"]["failed_preconditions"]
    )

    unreachable = mod.build_artifact(
        root=_make_repo(tmp_path / "unreachable", _references()),
        search_started_at=START,
        search_finished_at=FINISH,
        duration_s=1.0,
        source_receipts=[
            {
                "receipt_id": "down",
                "authority": "arXiv",
                "source_role": "primary",
                "query": "fixture",
                "url": "https://arxiv.org/",
                "accessed_at": START,
                "access_outcome": "timeout",
                "endpoint_capability": "fixture",
                "candidate_ids": [],
                "candidate_count": 0,
                "status": None,
            }
        ],
    )
    assert (
        "source_reachability_failed" in unreachable["preconditions_checked"]["failed_preconditions"]
    )

    malformed = _make_repo(tmp_path / "malformed", _references())
    (malformed / mod.ROADMAP_RELATIVE_PATH).write_text("- just\n- a list\n", encoding="utf-8")
    (malformed / mod.SPEC_RELATIVE_PATH).write_text("missing\n", encoding="utf-8")
    monkeypatch.setattr(mod.os, "access", lambda _path, _mode: False)
    malformed_artifact = mod.build_artifact(
        root=malformed,
        search_started_at=START,
        search_finished_at=FINISH,
        duration_s=1.0,
    )
    failures = malformed_artifact["preconditions_checked"]["failed_preconditions"]
    assert "active_roadmap_identity_unavailable" in failures
    assert "spec_req_report_6171_missing" in failures
    assert "output_path_unavailable" in failures
    monkeypatch.undo()

    root = _make_repo(tmp_path / "schema", _references())
    artifact = mod.build_artifact(
        root=root,
        search_started_at=START,
        search_finished_at=FINISH,
        duration_s=1.0,
    )
    mod.validate_artifact(artifact)
    assert mod.path_sha256(root / "missing") is None
    assert mod.read_text_if_present(root / "missing") == ""
    assert mod.insert_after_planner_block("no marker\n", "BLOCK").endswith("BLOCK")
    assert (
        mod.insert_after_planner_block(
            f"{mod.EXECUTION_DELTA_HEADING}\n",
            "BLOCK",
        )
        == f"{mod.EXECUTION_DELTA_HEADING}\n"
    )

    for mutate, message in (
        (lambda a: a.pop("status"), "missing required"),
        (lambda a: a.update(status="done"), "invalid status"),
        (lambda a: a.update(honest_verdict="complete:"), "honest_verdict"),
        (lambda a: a.update(inference_substrate="literature_ingestion"), "substrate"),
        (lambda a: a.update(duration_s=-1), "duration"),
        (lambda a: a.update(zero_delta_accepted="true"), "zero_delta"),
        (
            lambda a: a["accepted_rejected_and_guarded_delta_ledger"][0].pop("stable_id"),
            "ledger row",
        ),
        (lambda a: a["field_provenance"].pop("status"), "missing provenance"),
        (lambda a: a.update(reproducibility_checksum="nope"), "checksum"),
    ):
        broken = mod.roundtrip(artifact)
        mutate(broken)
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(broken)

    monkeypatch.setattr(mod, "REPO_ROOT", root)
    assert mod.main() == 0
    out = capsys.readouterr().out
    assert "experiment_6171_v535_source_delta_ingestion.json" in out
    written = json.loads((root / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    mod.validate_artifact(written)
