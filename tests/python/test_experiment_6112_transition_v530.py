"""Tests for the Exp6112 V530 transition receipt.

Spec refs: REQ-REPORT-6112,
SCENARIO-REPORT-6112-ACTIVATED-MATRIX,
SCENARIO-REPORT-6112-TERMINAL-CLASSES,
SCENARIO-REPORT-6112-UNACTIVATED-PROPOSAL,
SCENARIO-REPORT-6112-RETIREMENT,
SCENARIO-REPORT-6112-DUPLICATE-DEBT-AND-VERIFIER,
SCENARIO-REPORT-6112-RANGE-COLLISION,
SCENARIO-REPORT-6112-SCHEMA.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_6112_transition_v530 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"


def _write_text(root: Path, rel_path: Path | str, text: str = "fixture\n") -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(root: Path, rel_path: Path | str, payload: Any) -> None:
    _write_text(root, rel_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _artifact(task_id: str) -> JsonDict:
    fixtures: dict[str, JsonDict] = {
        "exp6100-transition-v529": {
            "status": "complete_with_terminal_receipts",
            "honest_verdict": "complete: archived terminal .528 identities into .529",
            "inference_substrate": "aggregation_from_upstream_artifacts",
        },
        "exp6101-v529-source-delta-ingestion": {
            "status": "complete",
            "honest_verdict": "complete_null: no accepted post-V529 source deltas",
            "inference_substrate": "aggregation_from_external_primary_sources",
        },
        "exp6102-sota-atom-corpus-vram-recovery": {
            "status": "blocked",
            "honest_verdict": "blocked: insufficient_free_vram",
            "inference_substrate": "live_local_sota_gguf_cuda_representation_extraction",
            "retirement_triggered": True,
        },
        "exp6103-phase-d-difficulty-ladder-fixture": {
            "status": "complete",
            "honest_verdict": "complete_ready: phase_d_difficulty_ladder_fixture_sealed_no_llm",
            "inference_substrate": "deterministic_exact_fixture_generation_no_llm",
        },
    }
    return fixtures[task_id]


def _active_roadmap_payload() -> JsonDict:
    return {
        "milestone": mod.MILESTONE_TO,
        "milestone_title": mod.MILESTONE_TO_TITLE,
        "milestone_doc": mod.ROADMAP_DOC_RELATIVE_PATH.as_posix(),
        "tasks": [
            {
                "id": task_id,
                "milestone": mod.MILESTONE_TO,
                "title": mod.ACTIVE_V530_TASK_TITLES[task_id],
                "deliverable": rel_path.as_posix(),
            }
            for task_id, rel_path in mod.ACTIVE_V530_TASK_ARTIFACT_PATHS.items()
        ],
    }


def _completion_payload(include_529_blocks: int = 1) -> JsonDict:
    block = {
        "id": mod.MILESTONE_FROM,
        "title": mod.MILESTONE_FROM_TITLE,
        "doc": mod.ROADMAP_DOC_RELATIVE_PATH.as_posix(),
        "completed": "2026-08-04",
        "finding": "See conductor log for per-experiment results.",
        "tasks": [
            {
                "id": task_id,
                "title": mod.ACTIVATED_TASK_TITLES[task_id],
                "deliverable": rel_path.as_posix(),
                "result": "OK (conductor)",
            }
            for task_id, rel_path in mod.ACTIVATED_TASK_ARTIFACT_PATHS.items()
        ],
    }
    old_duplicate = {
        "id": "2026.07.510",
        "tasks": [{"id": "exp5706-transition-v510", "deliverable": "results/x.json"}],
    }
    return {
        "milestones": [
            deepcopy(old_duplicate),
            deepcopy(old_duplicate),
            *[deepcopy(block) for _ in range(include_529_blocks)],
        ]
    }


def _make_root(root: Path, *, include_529_blocks: int = 1) -> None:
    for task_id, rel_path in mod.ACTIVATED_TASK_ARTIFACT_PATHS.items():
        _write_json(root, rel_path, _artifact(task_id))
    _write_json(
        root,
        "results/experiment_6102_numeric_prefix_alias_success.json",
        {"honest_verdict": "complete: alias must not reopen the retired block"},
    )
    _write_text(root, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(_active_roadmap_payload()))
    _write_text(
        root,
        mod.ROADMAP_DOC_RELATIVE_PATH,
        "\n".join(
            [
                "# Research Roadmap vNEXT",
                "",
                "**Experiment range:** Exp6112-Exp6123",
                "### Exp6112",
                "**Deliverable:** `results/experiment_6112_transition_v530.json`",
                "proposal-only Exp6104-Exp6111 as unactivated rather than missing experiments.",
            ]
        )
        + "\n",
    )
    _write_text(
        root,
        mod.RESEARCH_COMPLETE_RELATIVE_PATH,
        yaml.safe_dump(_completion_payload(include_529_blocks=include_529_blocks)),
    )
    _write_text(
        root,
        mod.CONDUCTOR_LOG_RELATIVE_PATH,
        "\n".join(
            [
                "| 2026-08-04 10:55 UTC | Plan milestone 2026.08.529 | OK | 4 tasks proposed |",
                "| 2026-08-04 10:57 UTC | Milestone 2026.08.529 activated | OK | 4 tasks queued |",
                "| 2026-08-04 13:00 UTC | Exact terminal-boundary handoff from .528 into .52 | OK | 86 passed |",
                "| 2026-08-04 15:01 UTC | Dated evidence refresh after the V529 planner mark | OK | 86 passed |",
                "| 2026-08-04 17:13 UTC | Checkpointed all-family exact-atom representation  | FAIL | artifact_not_updated |",
                "| 2026-08-04 17:32 UTC | Sealed low-chance Phase-D model-difficulty ladder | OK | 86 passed |",
                "| 2026-08-04 18:32 UTC | Plan milestone 2026.08.530 | OK | 12 tasks proposed |",
                "| 2026-08-04 18:35 UTC | Milestone 2026.08.530 activated | OK | 12 tasks queued |",
            ]
        )
        + "\n",
    )
    _write_text(root, mod.EXCLUSION_MANIFEST_RELATIVE_PATH, "retired: []\nretired_experiments: []\n")
    for rel_path in (
        mod.AGENTS_RELATIVE_PATH,
        mod.CODEX_RELATIVE_PATH,
        mod.CLAUDE_RELATIVE_PATH,
        mod.KNOWN_ISSUES_RELATIVE_PATH,
        mod.STATUS_RELATIVE_PATH,
        mod.CHANGELOG_RELATIVE_PATH,
        mod.TRACEABILITY_RELATIVE_PATH,
        mod.NORTH_STAR_RELATIVE_PATH,
        mod.CONDUCTOR_RELATIVE_PATH,
        mod.ADVERSARIAL_VERIFY_RELATIVE_PATH,
        mod.EVIDENCE_INDEX_RELATIVE_PATH,
        mod.DOC_RECONCILE_RELATIVE_PATH,
        mod.SPEC_RELATIVE_PATH,
    ):
        _write_text(root, rel_path, f"{rel_path.as_posix()} fixture\nREQ-REPORT-6112\n")


def _receipt(task_id: str) -> JsonDict:
    artifact_path = mod.ACTIVATED_TASK_ARTIFACT_PATHS[task_id].as_posix()
    stdout_json = {
        "reports": [
            {
                "artifact": artifact_path,
                "loaded": True,
                "flag_count": 0,
                "max_severity": -1,
                "flags": [],
            }
        ],
        "flagged_count": 0,
    }
    return {
        "task_id": task_id,
        "artifact_path": artifact_path,
        "command": f".venv/bin/python scripts/adversarial_verify.py --json {artifact_path}",
        "exit_code": 0,
        "stdout_json": stdout_json,
        "stderr": "",
        "receipt_hash": mod.sha256_json(stdout_json),
    }


def _receipts() -> dict[str, JsonDict]:
    return {task_id: _receipt(task_id) for task_id in mod.ACTIVATED_TASK_ARTIFACT_PATHS}


def _test_receipts() -> list[JsonDict]:
    rows = [
        {
            "command": f"task-owned {kind}",
            "exit_code": 0,
            "ownership_class": "task_owned",
            "suite_kind": kind,
        }
        for kind in mod.REQUIRED_TASK_OWNED_GATE_KINDS
    ]
    rows.extend(
        [
            {
                "command": ".venv/bin/pytest tests/python -q",
                "exit_code": 1,
                "ownership_class": "global_suite",
                "phase": "before",
                "failure_node_ids": ["tests/python/inherited/test_global.py::test_old"],
            },
            {
                "command": ".venv/bin/pytest tests/python -q",
                "exit_code": 1,
                "ownership_class": "global_suite",
                "phase": "after",
                "failure_node_ids": ["tests/python/inherited/test_global.py::test_old"],
            },
            {
                "command": ".venv/bin/python scripts/check_spec_coverage.py",
                "exit_code": 1,
                "ownership_class": "spec_coverage",
                "phase": "before",
                "missing_node_ids": ["tests/python/inherited/test_spec.py::test_old"],
            },
            {
                "command": ".venv/bin/python scripts/check_spec_coverage.py",
                "exit_code": 1,
                "ownership_class": "spec_coverage",
                "phase": "after",
                "missing_node_ids": ["tests/python/inherited/test_spec.py::test_old"],
            },
            {
                "command": ".venv/bin/python scripts/root_clutter_sweep.py --min-age-min 0",
                "exit_code": 0,
                "ownership_class": "root_clutter",
                "phase": "before",
                "root_clutter_paths": ["old_probe.py"],
            },
            {
                "command": ".venv/bin/python scripts/root_clutter_sweep.py --min-age-min 0",
                "exit_code": 0,
                "ownership_class": "root_clutter",
                "phase": "after",
                "root_clutter_paths": ["old_probe.py"],
            },
        ]
    )
    return rows


def _build(root: Path) -> JsonDict:
    return mod.build_report(
        root,
        adversarial_receipts=_receipts(),
        tests_run=_test_receipts(),
        duration_s=1.25,
    )


def test_req_report_6112_spec_declares_transition_contract() -> None:
    """REQ-REPORT-6112: OpenSpec names activated, proposal, retirement, and collision rules."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-6112") :]

    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    assert "(milestone, task_id, declared_deliverable)" in section
    for scenario in (
        "SCENARIO-REPORT-6112-ACTIVATED-MATRIX",
        "SCENARIO-REPORT-6112-TERMINAL-CLASSES",
        "SCENARIO-REPORT-6112-UNACTIVATED-PROPOSAL",
        "SCENARIO-REPORT-6112-RETIREMENT",
        "SCENARIO-REPORT-6112-DUPLICATE-DEBT-AND-VERIFIER",
        "SCENARIO-REPORT-6112-RANGE-COLLISION",
        "SCENARIO-REPORT-6112-SCHEMA",
    ):
        assert scenario in section
    assert "Exp6104 through Exp6111" in section
    assert "Exp6112 through Exp6123" in section
    assert "global_suite_failure_delta <= 0" in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert mod.FIELD_PRINCIPLES[field] in section


def test_scenario_report_6112_exact_matrix_terminal_proposal_and_retirement(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6112-ACTIVATED-MATRIX: four .529 identities classify once."""

    _make_root(tmp_path, include_529_blocks=2)
    report = _build(tmp_path)

    assert report["status"] == "complete_with_terminal_receipts"
    assert report["honest_verdict"].startswith("complete:")
    assert report["milestone_transition"] == {
        "source_milestone": "2026.08.529",
        "destination_milestone": "2026.08.530",
        "task_identity_tuple": ["milestone", "task_id", "declared_deliverable"],
    }
    matrix = report["activated_task_and_deliverable_matrix"]
    assert list(matrix) == list(mod.ACTIVATED_TASK_ARTIFACT_PATHS)
    assert len(matrix) == 4
    assert matrix["exp6102-sota-atom-corpus-vram-recovery"]["terminal_class"] == "blocked-retired"
    assert matrix["exp6102-sota-atom-corpus-vram-recovery"]["retirement_triggered"] is True
    assert matrix["exp6102-sota-atom-corpus-vram-recovery"]["same_number_aliases_ignored"] == [
        "results/experiment_6102_numeric_prefix_alias_success.json"
    ]

    classes = report["exact_terminal_classification"]
    assert classes["terminal_class_by_task_id"] == mod.EXPECTED_TERMINAL_CLASSES
    assert classes["task_ids_by_terminal_class"]["blocked-retired"] == [
        "exp6102-sota-atom-corpus-vram-recovery"
    ]
    assert classes["all_activated_terminal"] is True

    proposal = report["proposal_only_unactivated_id_receipt"]
    assert proposal["proposal_only_unactivated_task_ids"] == mod.PROPOSAL_ONLY_UNACTIVATED_TASK_IDS
    assert proposal["completed_or_missing_claim_count"] == 0
    assert proposal["adversarial_verification_claim_count"] == 0

    retirement = report["retirement_signal_preserved"]
    assert retirement["task_id"] == "exp6102-sota-atom-corpus-vram-recovery"
    assert retirement["retirement_triggered"] is True
    assert retirement["terminal_class"] == "blocked-retired"
    assert retirement["transition_reopened_retired_task"] is False

    verifier = report["adversarial_verifier_receipts"]
    assert verifier["verified_present_declared_deliverable_count"] == 4
    assert verifier["missing_declared_deliverables_not_verified"] == []
    mod.validate_artifact(report)


def test_scenario_report_6112_append_once_debt_delta_and_collision_blocking(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6112-DUPLICATE-DEBT-AND-VERIFIER: inherited debt is delta-gated."""

    _make_root(tmp_path, include_529_blocks=2)
    report = _build(tmp_path)

    assert report["research_complete_append_count"] == 0
    assert report["duplicate_history_amplification_count"] == 0
    assert report["research_complete_append_receipt"]["reason"] == "exact_milestone_block_present"
    assert report["research_complete_append_receipt"]["before_milestone_block_count"] == 2
    debt = report["inherited_debt_baselines_and_deltas"]
    assert debt["global_suite_failure_delta"] == 0
    assert debt["global_spec_gap_delta"] == 0
    assert debt["root_clutter_delta"] == 0
    assert debt["non_amplification_gate_passed"] is True
    mod.validate_artifact(report)

    absent = tmp_path / "absent"
    _make_root(absent, include_529_blocks=0)
    first = _build(absent)
    assert first["research_complete_append_count"] == 1
    assert first["duplicate_history_amplification_count"] == 0
    second = _build(absent)
    assert second["research_complete_append_count"] == 0

    amplified_rows = deepcopy(_test_receipts())
    for row in amplified_rows:
        if row.get("ownership_class") == "global_suite" and row.get("phase") == "after":
            row["failure_node_ids"].append("tests/python/new/test_owned.py::test_new")
    amplified = mod.build_report(
        tmp_path,
        adversarial_receipts=_receipts(),
        tests_run=amplified_rows,
        duration_s=1.25,
    )
    assert amplified["status"] == "blocked"
    assert "global_suite_debt_amplified" in amplified["preconditions_checked"]["failed_preconditions"]

    _write_json(tmp_path, "results/experiment_6119_stale_collision.json", {"status": "stale"})
    collision = _build(tmp_path)
    assert collision["status"] == "blocked"
    assert collision["honest_verdict"].startswith("blocked:")
    assert collision["next_range_collision_count"] == 1
    assert collision["preconditions_checked"]["range_collision_scan"]["collisions"] == [
        {
            "path": "results/experiment_6119_stale_collision.json",
            "kind": "unexpected_next_range_reference",
            "numbers": [6119],
        }
    ]
    mod.validate_artifact(collision)


def test_scenario_report_6112_schema_validation_and_blocked_preconditions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-6112-SCHEMA: required fields, protection, and checksum hold."""

    _make_root(tmp_path)
    report = _build(tmp_path)

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(report)
    assert report["docs_reconciled"] == {
        "openspec_research_reporting_req_6112_present": True,
        "ops_status_deferred_to_conductor_stop_rule": True,
        "ops_changelog_deferred_to_conductor_stop_rule": True,
        "traceability_deferred_to_conductor_stop_rule": True,
        "ops_conductor_log_deferred_to_conductor_stop_rule": True,
        "principle": mod.FIELD_PRINCIPLES["docs_reconciled"],
    }
    assert report["protected_files_unchanged"]["all_unchanged"] is True
    assert report["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert report["field_provenance"][field]["principle"] == mod.FIELD_PRINCIPLES[field]
    mod.validate_artifact(report)

    mutations = [
        (lambda artifact: artifact.pop("status"), "missing required field"),
        (lambda artifact: artifact.update(inference_substrate="live_llm_inference"), "inference_substrate"),
        (lambda artifact: artifact.update(honest_verdict="complete_null: bad"), "honest_verdict"),
        (lambda artifact: artifact.update(next_range_collision_count="0"), "next_range_collision_count"),
        (
            lambda artifact: artifact.update(next_range_collision_count=1, status="complete_with_terminal_receipts"),
            "next_range_collision_count must be zero",
        ),
        (lambda artifact: artifact.update(research_complete_append_count=2), "research_complete_append_count"),
        (
            lambda artifact: artifact.update(duplicate_history_amplification_count=1),
            "duplicate_history_amplification_count",
        ),
        (
            lambda artifact: artifact["activated_task_and_deliverable_matrix"].pop(
                "exp6103-phase-d-difficulty-ladder-fixture"
            ),
            "exactly four",
        ),
        (
            lambda artifact: artifact["activated_task_and_deliverable_matrix"].update(
                {"exp6103-phase-d-difficulty-ladder-fixture": []}
            ),
            "exactly four",
        ),
        (
            lambda artifact: artifact["activated_task_and_deliverable_matrix"][
                "exp6102-sota-atom-corpus-vram-recovery"
            ].update(identity=["2026.08.529", "exp6102-sota-atom-corpus-vram-recovery", "wrong.json"]),
            "activated identity mismatch",
        ),
        (
            lambda artifact: artifact["exact_terminal_classification"]["terminal_class_by_task_id"].update(
                {"exp6102-sota-atom-corpus-vram-recovery": "complete"}
            ),
            "terminal classes",
        ),
        (lambda artifact: artifact.update(proposal_only_unactivated_id_receipt=[]), "proposal-only"),
        (
            lambda artifact: artifact["proposal_only_unactivated_id_receipt"].update(
                completed_or_missing_claim_count=1
            ),
            "proposal-only",
        ),
        (lambda artifact: artifact.update(retirement_signal_preserved=[]), "retirement"),
        (
            lambda artifact: artifact["retirement_signal_preserved"].update(retirement_triggered=False),
            "retirement",
        ),
        (lambda artifact: artifact.update(adversarial_verifier_receipts=[]), "adversarial verifier"),
        (
            lambda artifact: artifact["adversarial_verifier_receipts"].update(
                verified_present_declared_deliverable_count=1
            ),
            "adversarial verifier",
        ),
        (
            lambda artifact: artifact["adversarial_verifier_receipts"]["reports"][0].pop("receipt_hash"),
            "adversarial verifier receipt",
        ),
        (
            lambda artifact: artifact["adversarial_verifier_receipts"]["reports"][0].update(command="python other.py"),
            "adversarial verifier receipt command",
        ),
        (
            lambda artifact: artifact["task_owned_gate_receipts"].update(
                all_required_gate_kinds_present=False
            ),
            "task-owned gate",
        ),
        (
            lambda artifact: artifact["inherited_debt_baselines_and_deltas"].update(
                non_amplification_gate_passed=False
            ),
            "debt non-amplification",
        ),
        (
            lambda artifact: artifact["protected_files_unchanged"].update(all_unchanged=False),
            "protected file",
        ),
        (
            lambda artifact: (
                artifact["protected_files_unchanged"].update(all_unchanged=True),
                next(iter(artifact["protected_files_unchanged"]["files"].values())).update(
                    unchanged=False
                ),
            ),
            "protected file",
        ),
        (lambda artifact: artifact.update(field_provenance=[]), "field provenance"),
        (
            lambda artifact: artifact["field_provenance"]["status"].update(principle="wrong"),
            "field provenance missing",
        ),
    ]
    for mutate, needle in mutations:
        artifact = deepcopy(report)
        mutate(artifact)
        artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
        with pytest.raises(ValueError, match=needle):
            mod.validate_artifact(artifact)

    checksum_drift = deepcopy(report)
    checksum_drift["duration_s"] = 9.5
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(checksum_drift)

    bad = tmp_path / "bad"
    _make_root(bad)
    _write_text(bad, mod.ROADMAP_RELATIVE_PATH, "a: [\n")
    _write_text(bad, mod.RESEARCH_COMPLETE_RELATIVE_PATH, "a: [\n")
    _write_text(bad, mod.EXCLUSION_MANIFEST_RELATIVE_PATH, "a: [\n")
    _write_text(bad, mod.CONDUCTOR_LOG_RELATIVE_PATH, "no activation\n")
    _write_text(bad, mod.SPEC_RELATIVE_PATH, "missing req\n")
    (bad / mod.ADVERSARIAL_VERIFY_RELATIVE_PATH).unlink()
    rows = [row for row in _test_receipts() if row.get("suite_kind") != "coverage"]
    monkeypatch.setattr(
        mod,
        "_protected_files_unchanged",
        lambda _root, _before: {
            "files": {"research-roadmap.yaml": {"unchanged": False}},
            "all_unchanged": False,
            "principle": mod.FIELD_PRINCIPLES["protected_files_unchanged"],
        },
    )
    monkeypatch.setattr(mod, "_resource_receipts", lambda _root: {"disk": {"ok": False}, "memory": {"ok": True}})
    monkeypatch.setattr(mod, "_atomic_output_receipt", lambda _path: {"ok": False})
    blocked = mod.build_report(bad, adversarial_receipts=_receipts(), tests_run=rows, duration_s=1.25)
    failed = set(blocked["preconditions_checked"]["failed_preconditions"])
    assert {
        "active_roadmap_unloadable",
        "research_complete_unparseable",
        "exclusion_manifest_unparseable",
        "v529_activation_line_missing_or_not_four",
        "v530_activation_line_missing_or_not_twelve",
        "live_verifier_missing",
        "task_owned_gate_missing",
        "openspec_req_6112_missing",
        "protected_file_modified",
        "insufficient_resources",
        "atomic_output_unavailable",
    } <= failed

    many = tmp_path / "many"
    _make_root(many)
    bad_roadmap = _active_roadmap_payload()
    bad_roadmap["milestone"] = "2026.08.999"
    bad_roadmap["tasks"] = bad_roadmap["tasks"][:-1]
    _write_text(many, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(bad_roadmap))
    _write_text(many, mod.ROADMAP_NEXT_RELATIVE_PATH, "a: [\n")
    (many / mod.ROADMAP_DOC_RELATIVE_PATH).unlink()
    _write_json(
        many,
        mod.ACTIVATED_TASK_ARTIFACT_PATHS["exp6102-sota-atom-corpus-vram-recovery"],
        {"status": "blocked", "honest_verdict": "blocked: insufficient_free_vram"},
    )
    sparse_receipts = dict(_receipts())
    sparse_receipts.pop("exp6101-v529-source-delta-ingestion")
    sparse_receipts["exp6103-phase-d-difficulty-ladder-fixture"] = {
        **sparse_receipts["exp6103-phase-d-difficulty-ladder-fixture"],
        "exit_code": 1,
        "stdout_json": {
            "reports": [
                {
                    "artifact": "x",
                    "loaded": True,
                    "flag_count": 1,
                    "max_severity": 3,
                    "flags": [{"kind": "CRITICAL", "severity": "critical"}],
                }
            ]
        },
    }
    many_rows = _test_receipts()
    many_rows[0]["exit_code"] = 1
    for row in many_rows:
        if row.get("ownership_class") == "spec_coverage" and row.get("phase") == "after":
            row["missing_node_ids"].append("tests/python/new/test_spec.py::test_new")
        if row.get("ownership_class") == "root_clutter" and row.get("phase") == "after":
            row["root_clutter_paths"].append("new_probe.py")
    monkeypatch.setattr(
        mod,
        "_append_completion_if_absent",
        lambda _root, _terminal: {
            "append_count": 0,
            "duplicate_history_amplification_count": 1,
        },
    )
    many_blocked = mod.build_report(
        many,
        adversarial_receipts=sparse_receipts,
        tests_run=many_rows,
        duration_s=1.25,
    )
    many_failed = set(many_blocked["preconditions_checked"]["failed_preconditions"])
    assert {
        "active_roadmap_milestone_mismatch",
        "active_roadmap_task_ids_mismatch",
        "roadmap_next_unloadable",
        "vnext_proposal_missing",
        "terminal_outcomes_not_preserved",
        "retirement_signal_not_preserved",
        "missing_adversarial_receipts",
        "adversarial_verifier_failed",
        "task_owned_gate_failed",
        "global_spec_debt_amplified",
        "root_clutter_debt_amplified",
        "duplicate_history_amplified",
    } <= many_failed


def test_req_report_6112_defensive_helpers_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-6112: helper failures produce explicit blocked preconditions."""

    output = tmp_path / "artifact.json"
    mod.write_json(output, {"b": 1})
    assert json.loads(output.read_text(encoding="utf-8")) == {"b": 1}
    assert mod.path_sha256(output).startswith("sha256:")
    assert mod.path_sha256(tmp_path / "missing") is None
    assert mod.sha256_json({"a": 1}) == mod.sha256_bytes(b'{"a":1}')

    _, missing_json_meta = mod._read_json_mapping(tmp_path / "missing.json")
    assert missing_json_meta["error"] == "missing"
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    _, bad_json_meta = mod._read_json_mapping(bad_json)
    assert bad_json_meta["error"].startswith("json_error:")
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    _, list_json_meta = mod._read_json_mapping(list_json)
    assert list_json_meta["error"] == "json_not_mapping"

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("a: [\n", encoding="utf-8")
    _, bad_yaml_meta = mod._read_yaml_mapping(bad_yaml)
    assert bad_yaml_meta["error"].startswith("yaml_error:")
    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- a\n", encoding="utf-8")
    _, list_yaml_meta = mod._read_yaml_mapping(list_yaml)
    assert list_yaml_meta["error"] == "yaml_not_mapping"
    assert mod._history_blocks(tmp_path / "missing-root") == []
    assert mod._task_signature({"tasks": "bad"}) == ()
    list_history = tmp_path / "history-list.yaml"
    mod._write_history_blocks(list_history, [], [mod._completion_block_data()])
    assert isinstance(yaml.safe_load(list_history.read_text(encoding="utf-8")), list)
    nonterminal_append = mod._append_completion_if_absent(tmp_path, terminal=False)
    assert nonterminal_append["reason"] == "nonterminal_identity_present"

    assert (
        mod._classify_task(
            "exp6102-sota-atom-corpus-vram-recovery",
            {"status": "blocked", "retirement_triggered": True},
            {"present": True},
        )
        == "blocked-retired"
    )
    assert (
        mod._classify_task(
            "x",
            {"status": "complete"},
            {"present": True},
        )
        == "complete"
    )
    assert mod._classify_task("x", {"status": "unknown"}, {"present": True}) == "missing"
    assert mod._classify_task("x", {}, {"present": False}) == "missing"
    staging = mod._optional_staging_roadmap_receipt(
        {"milestone": "2026.08.530"},
        {"present": True, "loadable": True},
    )
    assert staging["reason"] == "present_optional_staging"
    assert mod._same_number_aliases(tmp_path / "no-results", "not-an-exp", Path("x")) == []
    assert mod._range_number_mentions("Exp6112 and experiment_6123") == {6112, 6123}
    assert mod._range_number_mentions("value 0.6117 is not an experiment id") == set()
    assert (
        mod._allowed_range_reference_kind(mod.SPEC_RELATIVE_PATH, {6112})
        == "transition_owned_reference"
    )
    assert (
        mod._allowed_range_reference_kind(mod.ROADMAP_DOC_RELATIVE_PATH, {6112, 6123})
        == "allowed_allocation_reference"
    )
    assert mod._root_clutter_inventory(tmp_path / "does-not-exist") == []

    after_only_debt = mod._debt_baselines_and_deltas(
        tmp_path,
        [
            {"ownership_class": "global_suite", "phase": "after", "failure_node_ids": ["global_after"]},
            {"ownership_class": "spec_coverage", "phase": "after", "missing_node_ids": ["spec_after"]},
            {"ownership_class": "root_clutter", "phase": "after", "root_clutter_paths": ["root_after.py"]},
        ],
    )
    assert after_only_debt["global_suite_failure_delta"] == 0
    assert after_only_debt["global_spec_gap_delta"] == 0
    assert after_only_debt["root_clutter_delta"] == 0

    before_only_debt = mod._debt_baselines_and_deltas(
        tmp_path,
        [
            {"ownership_class": "global_suite", "phase": "before", "failure_node_ids": ["global_before"]},
            {"ownership_class": "spec_coverage", "phase": "before", "missing_node_ids": ["spec_before"]},
            {"ownership_class": "root_clutter", "phase": "before", "root_clutter_paths": ["root_before.py"]},
        ],
    )
    assert before_only_debt["global_suite_failure_after_node_ids"] == ["global_before"]
    assert before_only_debt["global_spec_gap_after_node_ids"] == ["spec_before"]
    assert before_only_debt["root_clutter_after_paths"] == ["root_before.py"]
    no_debt_rows = mod._debt_baselines_and_deltas(tmp_path, [])
    assert no_debt_rows["root_clutter_delta"] == 0

    _make_root(tmp_path / "receipt_root")
    receipt_report = _build(tmp_path / "receipt_root")
    receipt_matrix = receipt_report["activated_task_and_deliverable_matrix"]
    sparse_receipts = dict(_receipts())
    sparse_receipts.pop("exp6101-v529-source-delta-ingestion")
    sparse_receipts["exp6103-phase-d-difficulty-ladder-fixture"] = {
        **sparse_receipts["exp6103-phase-d-difficulty-ladder-fixture"],
        "stdout_json": {
            "reports": [
                {
                    "artifact": "x",
                    "loaded": True,
                    "flag_count": 1,
                    "max_severity": 1,
                    "flags": [{"kind": "WARN", "severity": "warn"}],
                }
            ]
        },
    }
    sparse_receipts["exp6102-sota-atom-corpus-vram-recovery"] = {
        **sparse_receipts["exp6102-sota-atom-corpus-vram-recovery"],
        "exit_code": 1,
        "stdout_json": {
            "reports": [
                {
                    "artifact": "x",
                    "loaded": True,
                    "flag_count": 1,
                    "max_severity": 3,
                    "flags": [{"kind": "CRITICAL", "severity": "critical"}],
                }
            ]
        },
    }
    grouped = mod._adversarial_receipts_group(sparse_receipts, receipt_matrix)
    assert grouped["failed_receipt_task_ids"] == ["exp6102-sota-atom-corpus-vram-recovery"]
    assert grouped["warning_receipt_task_ids"] == ["exp6103-phase-d-difficulty-ladder-fixture"]
    absent_matrix = deepcopy(receipt_matrix)
    absent_matrix["exp6100-transition-v529"]["present"] = False
    absent_grouped = mod._adversarial_receipts_group(sparse_receipts, absent_matrix)
    assert absent_grouped["verified_present_declared_deliverable_count"] == 2
    assert mod._tests_run_rows(None)[0]["status"] == "not_recorded"
    assert mod._normalize_adversarial_receipts(None, {}) == {}
    assert mod._normalize_adversarial_receipts([{}, "bad", {"task_id": ""}], {}) == {}
    assert mod._receipt_flags({}) == []
    assert mod._receipt_flags({"stdout_json": {"reports": []}}) == []
    assert mod._receipt_flags({"stdout_json": {"reports": [{"flags": "bad"}]}}) == []
    assert mod._receipt_flag_count({"stdout_json": {"flagged_count": 3}}) == 3
    assert mod._receipt_flag_count({"flag_count": 4}) == 4
    assert mod._receipt_max_severity({"stdout_json": {"reports": [{"max_severity": None}]}}) == -1
    assert mod._receipt_max_severity({"max_severity": 3}) == 3
    dirty = mod._dirty_worktree_receipt(REPO)
    assert dirty["git_present"] is True
    assert dirty["command_exit_code"] == 0
