"""Tests for the Exp5849 V521 transition receipt.

Spec refs: REQ-REPORT-5849, SCENARIO-REPORT-5849,
SCENARIO-REPORT-5849-PRESERVE-BLOCKED-MIXED,
SCENARIO-REPORT-5849-COLLISION-BLOCK,
SCENARIO-REPORT-5849-FIELD-PROVENANCE.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5849_transition_v521 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"


def _write_json(root: Path, rel_path: Path | str, payload: Any) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(root: Path, rel_path: Path | str, text: str = "context\n") -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _artifact_payload(task_id: str) -> JsonDict:
    payloads: dict[str, JsonDict] = {
        "exp5837-transition-v520": {
            "status": "blocked",
            "honest_verdict": (
                "blocked: exp5837 transition preconditions failed: "
                "test_failures=['.venv/bin/pytest tests/python -q', "
                "'.venv/bin/python scripts/check_spec_coverage.py']"
            ),
            "flagged_evidence_preserved": True,
            "next_range_collision_count": 0,
            "preconditions_checked": {
                "failed_preconditions": [
                    "test_failures=['.venv/bin/pytest tests/python -q', "
                    "'.venv/bin/python scripts/check_spec_coverage.py']"
                ]
            },
            "test_exit_codes": {
                ".venv/bin/pytest tests/python -q": 1,
                ".venv/bin/python scripts/check_spec_coverage.py": 1,
                ".venv/bin/pytest tests/python/test_experiment_5837_transition_v520.py -q --no-cov -n 0": 0,
            },
        },
        "exp5838-v520-source-delta-ingestion": {
            "status": "complete",
            "honest_verdict": "complete: no accepted post-V520 source deltas; references unchanged",
            "accepted_finding_count": 0,
            "references_modified": False,
        },
        "exp5839-v519-evidence-qualification": {
            "status": "complete",
            "honest_verdict": (
                "mixed: constraint_stream_and_structural_qualified_lifecycle_and_replay_disqualified"
            ),
            "constraint_stream_qualified_score": 1.0,
            "structural_acquisition_qualified_score": 1.0,
            "adaptive_memory_lifecycle_qualified_score": 0.0,
            "selective_replay_qualified_score": 0.0,
            "promotion_eligibility_matrix": {
                "constraint_stream": {"class": "qualified_clean", "score": 1.0},
                "structural_acquisition": {"class": "qualified_clean", "score": 1.0},
                "adaptive_memory_lifecycle": {
                    "class": "disqualified_flagged_upstream",
                    "score": 0.0,
                },
                "selective_replay": {"class": "provisional_flagged_upstream", "score": 0.0},
            },
        },
        "exp5840-exact-counterfactual-embedding-fixture": {
            "status": "complete",
            "honest_verdict": "ready: exact_counterfactual_embedding_fixture_ready",
            "counterfactual_fixture_ready_score": 1.0,
            "row_file": "results/experiment_5840_exact_counterfactual_embedding_fixture.rows.jsonl",
        },
    }
    return payloads[task_id]


def _research_complete_payload(*, duplicate_520_blocks: int = 1) -> JsonDict:
    v520_block = {
        "id": mod.MILESTONE_FROM,
        "title": "Terminal V520",
        "doc": "openspec/change-proposals/research-roadmap-vNEXT.md",
        "completed": "2026-07-23",
        "finding": "See conductor log for per-experiment results.",
        "tasks": [
            {
                "id": task_id,
                "title": f"title for {task_id}",
                "deliverable": rel_path.as_posix(),
                "result": "OK (conductor)",
            }
            for task_id, rel_path in mod.TASK_ARTIFACT_PATHS.items()
        ],
    }
    return {"milestones": [v520_block for _ in range(duplicate_520_blocks)]}


def _active_roadmap_payload() -> JsonDict:
    return {
        "milestone": mod.MILESTONE_TO,
        "tasks": [
            {
                "id": task_id,
                "milestone": mod.MILESTONE_TO,
                "deliverable": mod.NEXT_TASK_ARTIFACT_PATHS[task_id].as_posix(),
            }
            for task_id in mod.NEXT_TASK_IDS
        ],
    }


def _vnext_doc() -> str:
    lines = [
        "# Research Roadmap vNEXT",
        "",
        "**Milestone:** 2026.07.521",
        "**Task range:** Exp5849-Exp5862",
        "",
    ]
    lines.extend(
        f"`{mod.NEXT_TASK_ARTIFACT_PATHS[task_id].as_posix()}`"
        for task_id in mod.NEXT_TASK_IDS
    )
    lines.extend(f"proposal-only `{task_id}`" for task_id in mod.RESERVED_UNACTIVATED_TASK_IDS)
    return "\n".join(lines) + "\n"


def _conductor_log() -> str:
    return "\n".join(
        [
            "| 2026-07-23 06:13 UTC | Archive terminal .519 evidence and allocate .520 | FAIL | bootstrap |",
            "| 2026-07-23 07:36 UTC | Archive terminal .519 evidence and allocate .520 | FAIL | timeout |",
            "| 2026-07-23 07:59 UTC | Archive terminal .519 evidence and allocate .520 | FAIL | bootstrap |",
            "| 2026-07-23 08:26 UTC | Dated primary-source and implementation receipt fo | OK | 88 passed |",
            "| 2026-07-23 08:54 UTC | Independent .519 row-level evidence and lifecycle  | OK | 88 passed |",
            "| 2026-07-23 09:14 UTC | Gated on Exp5839 stream qualification: exact causa | OK | 86 passed |",
        ]
    )


def _receipt(
    task_id: str, *, critical: tuple[str, ...] = (), warn: tuple[str, ...] = ()
) -> JsonDict:
    path = mod.TASK_ARTIFACT_PATHS[task_id].as_posix()
    flags = [
        {"kind": kind, "severity": "critical", "detail": f"{kind} detail"} for kind in critical
    ]
    flags.extend({"kind": kind, "severity": "warn", "detail": f"{kind} detail"} for kind in warn)
    report = {
        "reports": [
            {
                "path": path,
                "flag_count": len(flags),
                "flags": flags,
                "max_severity": 2 if critical else 1 if warn else -1,
            }
        ],
        "flagged_count": 1 if flags else 0,
    }
    encoded = json.dumps(report, sort_keys=True).encode("utf-8")
    return {
        "task_id": task_id,
        "artifact_path": path,
        "command": f".venv/bin/python scripts/adversarial_verify.py --json {path}",
        "exit_code": 1 if flags else 0,
        "stdout_json": report,
        "stderr": "",
        "receipt_hash": mod.sha256_bytes(encoded),
    }


def _clean_receipts() -> list[JsonDict]:
    return [_receipt(task_id) for task_id in mod.EXPECTED_TASK_IDS]


def _make_root(
    root: Path,
    *,
    duplicate_520_blocks: int = 1,
    include_research_complete: bool = True,
) -> None:
    for task_id, rel_path in mod.TASK_ARTIFACT_PATHS.items():
        _write_json(root, rel_path, _artifact_payload(task_id))
    _write_text(root, "results/experiment_5840_exact_counterfactual_embedding_fixture.rows.jsonl", "{}\n")
    _write_text(root, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(_active_roadmap_payload()))
    if include_research_complete:
        _write_text(
            root,
            mod.RESEARCH_COMPLETE_RELATIVE_PATH,
            yaml.safe_dump(_research_complete_payload(duplicate_520_blocks=duplicate_520_blocks)),
        )
    _write_text(root, mod.VNEXT_RELATIVE_PATH, _vnext_doc())
    _write_text(root, mod.CONDUCTOR_LOG_RELATIVE_PATH, _conductor_log())
    _write_text(root, mod.EXCLUSION_MANIFEST_RELATIVE_PATH, "retired_experiments: []\n")
    _write_text(root, mod.CONDUCTOR_RELATIVE_PATH, "# conductor fixture\n")
    _write_text(root, mod.EVIDENCE_INDEX_RELATIVE_PATH, "# evidence index fixture\n")
    _write_text(root, mod.DOC_RECONCILE_RELATIVE_PATH, "# reconcile fixture\n")
    _write_text(root, mod.SPEC_RELATIVE_PATH, "### REQ-REPORT-5849\n")


def _clean_build(root: Path, *, receipts: list[JsonDict] | None = None) -> JsonDict:
    return mod.build_report(
        root,
        adversarial_receipts=receipts or _clean_receipts(),
        tests_run=[
            {
                "command": "focused",
                "exit_code": 0,
                "ownership_class": "task_owned",
                "suite_kind": "focused",
            },
            {
                "command": ".venv/bin/pytest tests/python -q",
                "exit_code": 1,
                "ownership_class": "global_baseline",
                "blocking": False,
                "pre_existing": True,
            },
        ],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
            mod.STATUS_RELATIVE_PATH: False,
            mod.CHANGELOG_RELATIVE_PATH: False,
            mod.TRACEABILITY_RELATIVE_PATH: False,
        },
        duration_s=1.25,
    )


def test_spec_contains_req_report_5849_contract() -> None:
    """REQ-REPORT-5849: OpenSpec names exact identity, preservation, and range gates."""

    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-5849") :]

    assert str(mod.RESULT_RELATIVE_PATH) in section
    assert "(milestone, task_id, declared_deliverable)" in section
    assert "SCENARIO-REPORT-5849-PRESERVE-BLOCKED-MIXED" in section
    assert "Exp5841 through Exp5848" in section
    assert "Exp5849 through Exp5862" in section
    assert "next_range_collision_count=0" in section
    for field in mod.REQUIRED_PRINCIPLE_FIELDS:
        assert f"`{field}`" in section


def test_scenario_report_5849_archives_terminal_v520_by_exact_identity(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5849: declared V520 paths and disjoint classes are canonical."""

    _make_root(tmp_path)
    report = _clean_build(tmp_path)

    assert report["status"] == "complete"
    assert report["honest_verdict"].startswith("complete:")
    assert report["milestone_transition"] == {
        "source_milestone": "2026.07.520",
        "destination_milestone": "2026.07.521",
        "task_identity_tuple": ["milestone", "task_id", "declared_deliverable"],
    }
    assert len(report["activated_identity_matrix"]) == 4
    assert report["outcome_classification"]["blocked_task_ids"] == [
        "exp5837-transition-v520"
    ]
    assert report["outcome_classification"]["clean_zero_delta_task_ids"] == [
        "exp5838-v520-source-delta-ingestion"
    ]
    assert report["outcome_classification"]["mixed_qualified_task_ids"] == [
        "exp5839-v519-evidence-qualification"
    ]
    assert report["outcome_classification"]["clean_ready_task_ids"] == [
        "exp5840-exact-counterfactual-embedding-fixture"
    ]
    assert "exp5837-transition-v520" not in report["outcome_classification"][
        "headline_eligible_task_ids"
    ]
    assert "exp5839-v519-evidence-qualification" not in report["outcome_classification"][
        "headline_eligible_task_ids"
    ]

    exp5837 = report["activated_identity_matrix"][0]
    assert exp5837["task_id"] == "exp5837-transition-v520"
    assert exp5837["outcome_class"] == "blocked"
    assert exp5837["historical_blocked_reasons_preserved"]

    exp5839 = report["activated_identity_matrix"][2]
    assert exp5839["task_id"] == "exp5839-v519-evidence-qualification"
    assert exp5839["outcome_class"] == "mixed-qualified"
    assert exp5839["qualified_branches"] == ["constraint_stream", "structural_acquisition"]
    assert exp5839["disqualified_or_provisional_branches"] == [
        "adaptive_memory_lifecycle",
        "selective_replay",
    ]

    assert report["blocked_and_disqualified_evidence_preserved"] is True
    assert report["reserved_unactivated_task_ids"] == list(mod.RESERVED_UNACTIVATED_TASK_IDS)
    assert report["research_complete_append_count"] == 0
    assert report["preconditions_checked"]["roadmaps"]["next"]["present"] is False
    assert report["preconditions_checked"]["resource_receipts"]["disk_free_bytes"] > 0
    assert report["next_task_range"] == "exp5849-exp5862"
    assert report["next_range_collision_count"] == 0
    assert report["pre_existing_repository_debt"]["blocking_transition"] is False
    assert report["pre_existing_repository_debt"]["commands"] == [".venv/bin/pytest tests/python -q"]
    assert report["docs_reconciled"]["operator_owned_docs_deferred"] is True
    assert report["docs_reconciled"]["updated"] == [
        "openspec/capabilities/research-reporting/spec.md",
        "python/carnot/experiment_5849_transition_v521.py",
        "tests/python/test_experiment_5849_transition_v521.py",
        "results/experiment_5849_transition_v521.json",
    ]
    assert report["protected_files"]["research-roadmap.yaml"]["unchanged"] is True
    assert report["protected_files"]["scripts/research_conductor.py"]["unchanged"] is True


def test_scenario_report_5849_verifier_receipts_preserve_blocked_and_mixed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-5849-PRESERVE-BLOCKED-MIXED: receipts cannot launder evidence."""

    _make_root(tmp_path)
    report = _clean_build(tmp_path)
    receipts = {row["task_id"]: row for row in report["adversarial_verifier_receipts"]}

    assert receipts["exp5837-transition-v520"]["headline_eligible"] is False
    assert receipts["exp5837-transition-v520"]["headline_ineligible_reason"] == (
        "historical_blocked_artifact"
    )
    assert receipts["exp5838-v520-source-delta-ingestion"]["headline_eligible"] is True
    assert receipts["exp5839-v519-evidence-qualification"]["headline_eligible"] is False
    assert receipts["exp5839-v519-evidence-qualification"]["headline_ineligible_reason"] == (
        "mixed_qualified_disqualified_branches"
    )
    assert receipts["exp5840-exact-counterfactual-embedding-fixture"]["headline_eligible"] is True

    missing_receipt = _clean_build(tmp_path, receipts=_clean_receipts()[:-1])
    assert (
        "adversarial_receipts_missing=['exp5840-exact-counterfactual-embedding-fixture']"
        in missing_receipt["preconditions_checked"]["failed_preconditions"]
    )

    stale_receipts = _clean_receipts()
    stale_receipts[2] = _receipt("exp5839-v519-evidence-qualification", critical=("UNEXPECTED",))
    stale_report = _clean_build(tmp_path, receipts=stale_receipts)
    assert (
        "exp5839-v519-evidence-qualification_unexpected_critical_findings=['UNEXPECTED']"
        in stale_report["preconditions_checked"]["failed_preconditions"]
    )

    completed = mod.run_adversarial_verifier(
        tmp_path,
        mod.TASK_ARTIFACT_PATHS["exp5840-exact-counterfactual-embedding-fixture"],
        subprocess_run=lambda *_args, **_kwargs: mod.ProcessResult(
            returncode=0,
            stdout=json.dumps({"reports": [{"path": "x", "flags": [], "flag_count": 0}]}),
            stderr="",
        ),
    )
    assert completed["exit_code"] == 0
    assert completed["task_id"] == "exp5840-exact-counterfactual-embedding-fixture"
    assert completed["receipt_hash"].startswith("sha256:")

    failed = mod.run_adversarial_verifier(
        tmp_path,
        mod.TASK_ARTIFACT_PATHS["exp5837-transition-v520"],
        subprocess_run=lambda *_args, **_kwargs: mod.ProcessResult(
            returncode=1,
            stdout="{not-json",
            stderr="broken",
        ),
    )
    assert failed["stdout_parse_error"]

    monkeypatch.setattr(mod, "VERIFIER_TASK_IDS", ("exp9999-other",))
    try:
        assert mod.normalize_adversarial_verifier_receipts([])[0]["task_id"] == "exp9999-other"
    finally:
        monkeypatch.setattr(mod, "VERIFIER_TASK_IDS", mod.EXPECTED_TASK_IDS)


def test_scenario_report_5849_collision_blocks_allocation(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5849-COLLISION-BLOCK: occupied V521 ids fail closed."""

    _make_root(tmp_path)
    _write_json(tmp_path, "results/experiment_5857_existing_collision.json", {"status": "old"})
    payload = _research_complete_payload()
    payload["milestones"].append(
        {
            "id": "2026.07.400",
            "title": "stale next-range reference",
            "finding": "mentions exp5859-adaptive-state-microkernel-parity",
            "tasks": [],
        }
    )
    _write_text(tmp_path, mod.RESEARCH_COMPLETE_RELATIVE_PATH, yaml.safe_dump(payload))
    _write_text(
        tmp_path,
        mod.EXCLUSION_MANIFEST_RELATIVE_PATH,
        "retired_experiments:\n- scope_key: exp5860-live-active-observation-ab\n",
    )
    _write_json(
        tmp_path,
        "results/experiment_5700_transition_alloc.json",
        {"next_task_range": "exp5849-exp5862"},
    )

    report = _clean_build(tmp_path)

    assert report["status"] == "blocked"
    assert report["honest_verdict"].startswith("blocked:")
    assert report["next_range_collision_count"] == 4
    assert [row["path"] for row in report["collision_scan"]["preexisting_collisions"]] == [
        "ops/exclusion_manifest.yaml",
        "research-complete.yaml",
        "results/experiment_5700_transition_alloc.json",
        "results/experiment_5857_existing_collision.json",
    ]

    alias_root = tmp_path / "alias"
    _make_root(alias_root)
    _write_json(alias_root, "results/experiment_5839_same_number_alias.json", {"status": "complete"})
    alias_report = _clean_build(alias_root)
    assert alias_report["same_number_alias_groups"]["5839"]["aliases"][0]["path"] == (
        "results/experiment_5839_same_number_alias.json"
    )


def test_scenario_report_5849_emit_report_field_provenance_checksum_and_append(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5849-FIELD-PROVENANCE: emitted artifact is stable."""

    _make_root(tmp_path)
    history_before = (tmp_path / mod.RESEARCH_COMPLETE_RELATIVE_PATH).read_bytes()
    output = tmp_path / mod.RESULT_RELATIVE_PATH
    report = mod.emit_report(
        tmp_path,
        output_path=output,
        adversarial_receipts=_clean_receipts(),
        tests_run=[{"command": "focused", "exit_code": 0, "ownership_class": "task_owned"}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: False,
            mod.CONDUCTOR_RELATIVE_PATH: False,
            mod.STATUS_RELATIVE_PATH: False,
            mod.CHANGELOG_RELATIVE_PATH: False,
            mod.TRACEABILITY_RELATIVE_PATH: False,
        },
        duration_s=1.25,
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert (tmp_path / mod.RESEARCH_COMPLETE_RELATIVE_PATH).read_bytes() == history_before
    assert written == report
    assert mod.payload_checksum(written) == written["reproducibility_checksum"]
    assert set(mod.REQUIRED_PRINCIPLE_FIELDS).issubset(report["field_principles"])
    assert set(mod.REQUIRED_PRINCIPLE_FIELDS).issubset(report["field_provenance"])
    assert all(report["field_principles"][field] for field in mod.REQUIRED_PRINCIPLE_FIELDS)
    assert all(
        report["field_provenance"][field]["sources"] for field in mod.REQUIRED_PRINCIPLE_FIELDS
    )
    assert all(
        "sha256_by_source" in report["field_provenance"][field]
        for field in mod.REQUIRED_PRINCIPLE_FIELDS
    )
    assert report["duration_s"] == 1.25
    assert report["inference_substrate"] == "aggregation_from_upstream_artifacts"

    append_root = tmp_path / "append"
    _make_root(append_root, include_research_complete=False)
    appended = _clean_build(append_root)
    assert appended["research_complete_append_count"] == 1
    assert mod._research_complete_blocks(append_root)
    second = _clean_build(append_root)
    assert second["research_complete_append_count"] == 0

    original = mod.FIELD_PRINCIPLES.pop("status")
    try:
        with pytest.raises(KeyError, match="missing field principles"):
            mod.build_report(
                tmp_path,
                adversarial_receipts=_clean_receipts(),
                modification_overrides={
                    mod.ROADMAP_RELATIVE_PATH: False,
                    mod.CONDUCTOR_RELATIVE_PATH: False,
                    mod.STATUS_RELATIVE_PATH: False,
                    mod.CHANGELOG_RELATIVE_PATH: False,
                    mod.TRACEABILITY_RELATIVE_PATH: False,
                },
                duration_s=1.25,
            )
    finally:
        mod.FIELD_PRINCIPLES["status"] = original


def test_scenario_report_5849_defensive_preconditions_and_parsers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-5849: malformed inputs, ambiguity, and owned failures fail closed."""

    assert mod._task_signature({"tasks": "not-list"}) == ()
    assert mod._artifact_terminal_status({}, {"exists": False, "loadable": False}) == "missing"
    assert mod._artifact_terminal_status({}, {"exists": True, "loadable": False}) == "malformed"
    assert (
        mod._artifact_terminal_status(
            {"honest_verdict": "blocked: no input"},
            {"exists": True, "loadable": True},
        )
        == "blocked"
    )
    assert (
        mod._artifact_terminal_status(
            {"honest_verdict": "ready: fixture"},
            {"exists": True, "loadable": True},
        )
        == "complete"
    )
    assert mod._artifact_terminal_status({}, {"exists": True, "loadable": True}) == "unknown"
    assert mod._task_number("not-an-exp") is None
    assert mod._conductor_outcomes(tmp_path)[1] == list(mod.EXPECTED_TASK_IDS)
    assert mod._parse_conductor_log(tmp_path / "missing-log-file") == []
    short_log_root = tmp_path / "short-log"
    short_log_root.mkdir()
    _write_text(short_log_root, mod.CONDUCTOR_LOG_RELATIVE_PATH, "| too | short |\n")
    assert mod._parse_conductor_log(short_log_root) == []
    assert mod._task_id_for_artifact_path(Path("results/not_declared.json")) == ""
    assert mod._flag_rows_from_stdout(None) == []
    assert mod._flag_rows_from_stdout({"reports": []}) == []
    assert mod._flag_rows_from_stdout({"reports": ["not-a-map"]}) == []
    assert mod._receipt_kinds({"critical_findings": "not-list"}, "critical_findings") == []
    nonzero = mod.normalize_adversarial_verifier_receipts(
        [
            {
                "task_id": "exp5840-exact-counterfactual-embedding-fixture",
                "artifact_path": "results/experiment_5840_exact_counterfactual_embedding_fixture.json",
                "command": "nonzero",
                "exit_code": 2,
                "stdout_json": {"reports": [{"flags": [], "flag_count": 0}]},
            }
        ]
    )[-1]
    assert nonzero["headline_ineligible_reason"] == "verifier_exit_nonzero"

    assert (
        mod._adversarial_receipt_failures(
            {
                "exp5837-transition-v520": {
                    "present": True,
                    "critical_findings": [],
                    "warn_findings": [],
                    "exit_code": 0,
                },
                "exp5838-v520-source-delta-ingestion": {
                    "present": True,
                    "critical_findings": [],
                    "warn_findings": [],
                    "exit_code": 0,
                },
                "exp5839-v519-evidence-qualification": {
                    "present": True,
                    "critical_findings": [{"kind": "UNEXPECTED", "severity": "critical"}],
                    "warn_findings": [],
                    "exit_code": 1,
                },
                "exp5840-exact-counterfactual-embedding-fixture": {
                    "present": True,
                    "critical_findings": [],
                    "warn_findings": [],
                    "exit_code": 0,
                },
            }
        )[0]
        == "exp5839-v519-evidence-qualification_unexpected_critical_findings=['UNEXPECTED']"
    )

    malformed_yaml = tmp_path / "bad.yaml"
    malformed_yaml.write_text("not: [closed\n", encoding="utf-8")
    _payload, malformed_meta = mod._read_yaml_with_meta(malformed_yaml)
    assert malformed_meta["parsed"] is False
    assert malformed_meta["error"]
    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- item\n", encoding="utf-8")
    _payload, list_meta = mod._read_yaml_with_meta(list_yaml)
    assert list_meta["parsed"] is False
    assert list_meta["error"] == "expected mapping, got list"

    ambiguous_root = tmp_path / "ambiguous-history"
    _make_root(ambiguous_root)
    ambiguous_payload = _research_complete_payload()
    altered = json.loads(json.dumps(ambiguous_payload["milestones"][0]))
    altered["tasks"][0]["deliverable"] = "results/other.json"
    ambiguous_payload["milestones"].append(altered)
    _write_text(
        ambiguous_root, mod.RESEARCH_COMPLETE_RELATIVE_PATH, yaml.safe_dump(ambiguous_payload)
    )
    ambiguous = _clean_build(ambiguous_root)
    assert (
        "ambiguous_research_complete_declared_task_blocks"
        in ambiguous["preconditions_checked"]["failed_preconditions"]
    )

    duplicate_task_root = tmp_path / "duplicate-task"
    _make_root(duplicate_task_root)
    duplicate_payload = _research_complete_payload()
    duplicate_task = dict(duplicate_payload["milestones"][0]["tasks"][0])
    duplicate_task["deliverable"] = "results/conflicting.json"
    duplicate_payload["milestones"][0]["tasks"].append(duplicate_task)
    _write_text(
        duplicate_task_root,
        mod.RESEARCH_COMPLETE_RELATIVE_PATH,
        yaml.safe_dump(duplicate_payload),
    )
    duplicate_task_report = _clean_build(duplicate_task_root)
    assert any(
        item.startswith("duplicate_task_id_conflicts=")
        for item in duplicate_task_report["preconditions_checked"]["failed_preconditions"]
    )
    assert any(
        item.startswith("declared_task_ids_mismatch=")
        for item in duplicate_task_report["preconditions_checked"]["failed_preconditions"]
    )

    missing_artifact_root = tmp_path / "missing-artifact"
    _make_root(missing_artifact_root)
    (missing_artifact_root / mod.TASK_ARTIFACT_PATHS["exp5838-v520-source-delta-ingestion"]).unlink()
    missing_artifact = _clean_build(missing_artifact_root)
    assert (
        "missing_or_malformed_declared_deliverables=['exp5838-v520-source-delta-ingestion']"
        in missing_artifact["preconditions_checked"]["failed_preconditions"]
    )

    no_block_root = tmp_path / "no-block"
    _make_root(no_block_root)
    exp5837 = _artifact_payload("exp5837-transition-v520")
    exp5837["status"] = "complete"
    exp5837["honest_verdict"] = "complete: wrongly laundered"
    _write_json(no_block_root, mod.TASK_ARTIFACT_PATHS["exp5837-transition-v520"], exp5837)
    no_block = _clean_build(no_block_root)
    assert (
        "exp5837_blocked_state_not_preserved"
        in no_block["preconditions_checked"]["failed_preconditions"]
    )

    no_mixed_root = tmp_path / "no-mixed"
    _make_root(no_mixed_root)
    exp5839 = _artifact_payload("exp5839-v519-evidence-qualification")
    exp5839["adaptive_memory_lifecycle_qualified_score"] = 1.0
    _write_json(no_mixed_root, mod.TASK_ARTIFACT_PATHS["exp5839-v519-evidence-qualification"], exp5839)
    no_mixed = _clean_build(no_mixed_root)
    assert (
        "exp5839_mixed_disqualification_not_preserved"
        in no_mixed["preconditions_checked"]["failed_preconditions"]
    )

    bad_active_root = tmp_path / "bad-active"
    _make_root(bad_active_root)
    _write_text(bad_active_root, mod.ROADMAP_RELATIVE_PATH, "bad: [yaml\n")
    bad_active = _clean_build(bad_active_root)
    assert (
        "active_roadmap_unparseable" in bad_active["preconditions_checked"]["failed_preconditions"]
    )

    bad_task_root = tmp_path / "bad-task"
    _make_root(bad_task_root)
    _write_text(
        bad_task_root,
        mod.ROADMAP_RELATIVE_PATH,
        yaml.safe_dump(
            {
                "milestone": mod.MILESTONE_TO,
                "tasks": [{"id": "exp9999-not-allocated", "deliverable": "results/x.json"}],
            }
        ),
    )
    bad_task = _clean_build(bad_task_root)
    assert any(
        item.startswith("active_roadmap_task_ids=")
        for item in bad_task["preconditions_checked"]["failed_preconditions"]
    )

    bad_next_root = tmp_path / "bad-next"
    _make_root(bad_next_root)
    _write_text(bad_next_root, mod.ROADMAP_NEXT_RELATIVE_PATH, "bad: [yaml\n")
    bad_next = _clean_build(bad_next_root)
    assert "next_roadmap_unparseable" in bad_next["preconditions_checked"]["failed_preconditions"]

    bad_complete_root = tmp_path / "bad-complete"
    _make_root(bad_complete_root)
    _write_text(bad_complete_root, mod.RESEARCH_COMPLETE_RELATIVE_PATH, "bad: [yaml\n")
    bad_complete = _clean_build(bad_complete_root)
    assert (
        "research_complete_unparseable"
        in bad_complete["preconditions_checked"]["failed_preconditions"]
    )

    existing_complete_root = tmp_path / "existing-complete"
    _make_root(existing_complete_root, include_research_complete=False)
    _write_text(existing_complete_root, mod.RESEARCH_COMPLETE_RELATIVE_PATH, "milestones:")
    existing_complete = _clean_build(existing_complete_root)
    assert existing_complete["research_complete_append_count"] == 1
    assert mod._research_complete_blocks(existing_complete_root)

    missing_log_root = tmp_path / "missing-log"
    _make_root(missing_log_root)
    _write_text(missing_log_root, mod.CONDUCTOR_LOG_RELATIVE_PATH, "no rows\n")
    missing_log = _clean_build(missing_log_root)
    assert any(
        item.startswith("missing_conductor_outcomes=")
        for item in missing_log["preconditions_checked"]["failed_preconditions"]
    )

    modified_root = tmp_path / "modified"
    _make_root(modified_root)
    modified = mod.build_report(
        modified_root,
        adversarial_receipts=_clean_receipts(),
        tests_run=[{"command": "focused", "exit_code": 1, "ownership_class": "task_owned"}],
        modification_overrides={
            mod.ROADMAP_RELATIVE_PATH: True,
            mod.CONDUCTOR_RELATIVE_PATH: True,
            mod.STATUS_RELATIVE_PATH: True,
            mod.CHANGELOG_RELATIVE_PATH: True,
            mod.TRACEABILITY_RELATIVE_PATH: True,
        },
    )
    failed = modified["preconditions_checked"]["failed_preconditions"]
    assert "research_roadmap_modified" in failed
    assert "research_conductor_modified" in failed
    assert any(item.startswith("focused_transition_test_failures=") for item in failed)
    assert modified["duration_s"] > 0
    assert mod._focused_tests_failed(
        [{"command": "global baseline", "exit_code": 1, "blocking": False}]
    ) == []

    collision_branch_root = tmp_path / "collision-branches"
    _make_root(collision_branch_root)
    (collision_branch_root / "results/experiment_5854_directory").mkdir()
    _write_text(collision_branch_root, "results/experiment_5700_transition_bad.json", "{")
    collision_branch = _clean_build(collision_branch_root)
    assert collision_branch["next_range_collision_count"] == 0

    monkeypatch.setattr(mod, "EXPECTED_TASK_IDS", ("not-an-exp-task",))
    assert mod._same_number_alias_groups(tmp_path, {}) == {}
