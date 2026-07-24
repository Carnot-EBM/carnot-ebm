"""Tests for the Exp5877 V523 transition receipt.

Spec refs: REQ-REPORT-5877, SCENARIO-REPORT-5877-EXACT-ARCHIVE,
SCENARIO-REPORT-5877-APPEND-ONCE, SCENARIO-REPORT-5877-UNACTIVATED-PROPOSAL,
SCENARIO-REPORT-5877-RANGE-COLLISION, SCENARIO-REPORT-5877-SCHEMA.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5877_transition_v523 as mod


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


def _artifact(task_id: str) -> JsonDict:
    payloads: dict[str, JsonDict] = {
        "exp5863-transition-v522": {
            "status": "complete",
            "honest_verdict": "complete: archived terminal .521 into .522",
            "next_range_collision_count": 0,
        },
        "exp5864-v522-source-delta-ingestion": {
            "status": "complete",
            "honest_verdict": "complete: no accepted post-V522 source deltas",
            "accepted_finding_count": 0,
            "references_modified": False,
        },
        "exp5865-adaptive-state-kernel-requalification": {
            "status": "retired",
            "honest_verdict": (
                "retired: adaptive_state_requalification_blocked_by_unrelated_global_suite_debt"
            ),
            "adaptive_state_microkernel_requalified_score": 0.0,
        },
        "exp5867-prospective-certified-continuous-learning": {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gate_check_summary": "upstream artifact not found for exp5866",
        },
        "exp5868-hardness-controlled-constraint-fixture": {
            "status": "complete",
            "honest_verdict": "ready: hardness_controlled_exact_constraint_fixture_ready",
            "hardness_controlled_fixture_ready_score": 1.0,
        },
        "exp5869-hardness-surface-headroom-audit": {
            "status": "blocked",
            "honest_verdict": "blocked: test_exit_codes",
            "hardness_surface_headroom_ready_score": 0.0,
        },
    }
    return payloads[task_id]


def _completion_payload(*, include_522: bool = True, duplicate_510_blocks: int = 1) -> JsonDict:
    duplicate_block = {
        "id": "2026.07.510",
        "title": "Historical duplicate",
        "doc": "openspec/change-proposals/research-roadmap-vNEXT.md",
        "completed": "2026-07-17",
        "finding": "fixture",
        "tasks": [{"id": "exp5706-transition-v510", "deliverable": "results/x.json"}],
    }
    milestones = [deepcopy(duplicate_block) for _ in range(duplicate_510_blocks)]
    if include_522:
        milestones.append(
            {
                "id": mod.MILESTONE_FROM,
                "title": mod.MILESTONE_FROM_TITLE,
                "doc": mod.ROADMAP_DOC_RELATIVE_PATH.as_posix(),
                "completed": "2026-07-24",
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
        )
    return {"milestones": milestones}


def _active_roadmap_payload() -> JsonDict:
    return {
        "milestone": mod.MILESTONE_TO,
        "tasks": [
            {
                "id": task_id,
                "milestone": mod.MILESTONE_TO,
                "title": f"title for {task_id}",
                "deliverable": rel_path.as_posix(),
            }
            for task_id, rel_path in mod.NEXT_TASK_ARTIFACT_PATHS.items()
        ],
    }


def _vnext_doc() -> str:
    lines = [
        "# Research Roadmap vNEXT",
        "",
        "**Milestone:** 2026.07.523",
        "**Task range:** Exp5877-Exp5889",
        "",
        "Exp5870-Exp5876 appeared in the prior proposal but were never activated.",
    ]
    lines.extend(path.as_posix() for path in mod.NEXT_TASK_ARTIFACT_PATHS.values())
    return "\n".join(lines) + "\n"


def _conductor_log() -> str:
    return "\n".join(
        [
            "| 2026-07-24 01:26 UTC | Exact terminal-boundary handoff from .521 into .52 | OK | 87 passed |",
            "| 2026-07-24 01:59 UTC | Dated evidence refresh after the V522 planner mark | OK | 88 passed |",
            "| 2026-07-24 05:04 UTC | Adaptive-state microkernel E2E attribution and req | FAIL | artifact_not_updated |",
            "| 2026-07-24 05:11 UTC | Default-off adaptive-state adapter at the verify-r | GATE_BLOCK | upstream retired |",
            "| 2026-07-24 05:11 UTC | Prospective non-forgetting continuous learning thr | GATE_BLOCK | gate failed |",
            "| 2026-07-24 05:38 UTC | Proof-hardness-controlled exact constraint fixture | OK | 88 passed |",
            "| 2026-07-24 06:56 UTC | Constraint headroom, surface leakage, and oracle-d | FAIL | artifact_not_updated |",
        ]
    )


def _make_root(
    root: Path,
    *,
    include_522_complete: bool = True,
    duplicate_510_blocks: int = 1,
) -> None:
    for task_id, rel_path in mod.ACTIVATED_TASK_ARTIFACT_PATHS.items():
        if task_id == "exp5866-adaptive-state-pipeline-shadow-adapter":
            continue
        _write_json(root, rel_path, _artifact(task_id))
    _write_text(root, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(_active_roadmap_payload()))
    _write_text(
        root,
        mod.RESEARCH_COMPLETE_RELATIVE_PATH,
        yaml.safe_dump(
            _completion_payload(
                include_522=include_522_complete,
                duplicate_510_blocks=duplicate_510_blocks,
            )
        ),
    )
    _write_text(root, mod.ROADMAP_DOC_RELATIVE_PATH, _vnext_doc())
    _write_text(root, mod.CONDUCTOR_LOG_RELATIVE_PATH, _conductor_log())
    _write_text(root, mod.EXCLUSION_MANIFEST_RELATIVE_PATH, "retired_experiments: []\n")
    _write_text(root, mod.EVIDENCE_INDEX_RELATIVE_PATH, "# evidence index fixture\n")
    _write_text(root, mod.DOC_RECONCILE_RELATIVE_PATH, "# reconcile fixture\n")
    _write_text(root, mod.ADVERSARIAL_VERIFY_RELATIVE_PATH, "# verifier fixture\n")
    _write_text(
        root,
        "python/carnot/experiment_5877_transition_v523.py",
        "# owned exp5877 transition fixture\n",
    )
    _write_text(
        root,
        "tests/python/test_experiment_5877_transition_v523.py",
        "# owned exp5877 test fixture\n",
    )
    for rel_path in mod.PROTECTED_FILE_PATHS + (mod.SPEC_RELATIVE_PATH,):
        if not (root / rel_path).exists():
            _write_text(root, rel_path, f"{rel_path.as_posix()} fixture\n")


def _receipt(task_id: str, *, flagged: bool = False) -> JsonDict:
    flags = (
        [{"kind": "METHODOLOGY_MISSING", "severity": "warn", "detail": "fixture"}]
        if flagged
        else []
    )
    stdout_json = {
        "reports": [
            {
                "path": mod.ACTIVATED_TASK_ARTIFACT_PATHS[task_id].as_posix(),
                "flag_count": len(flags),
                "flags": flags,
                "max_severity": 1 if flags else -1,
            }
        ],
        "flagged_count": 1 if flags else 0,
    }
    return {
        "task_id": task_id,
        "artifact_path": mod.ACTIVATED_TASK_ARTIFACT_PATHS[task_id].as_posix(),
        "command": (
            ".venv/bin/python scripts/adversarial_verify.py --json "
            f"{mod.ACTIVATED_TASK_ARTIFACT_PATHS[task_id].as_posix()}"
        ),
        "exit_code": 1 if flagged else 0,
        "stdout_json": stdout_json,
        "stderr": "",
        "receipt_hash": mod.sha256_json(stdout_json),
    }


def _receipts() -> dict[str, JsonDict]:
    return {
        task_id: _receipt(
            task_id,
            flagged=task_id == "exp5864-v522-source-delta-ingestion",
        )
        for task_id in mod.ACTIVATED_TASK_ARTIFACT_PATHS
        if task_id != "exp5866-adaptive-state-pipeline-shadow-adapter"
    }


def _build(root: Path) -> JsonDict:
    return mod.build_report(
        root,
        adversarial_receipts=_receipts(),
        tests_run=[
            {
                "command": ".venv/bin/pytest tests/python/test_experiment_5877_transition_v523.py -q",
                "exit_code": 0,
            },
            {"command": ".venv/bin/pytest tests/python -q", "exit_code": 0},
        ],
        modification_overrides={rel_path: False for rel_path in mod.PROTECTED_FILE_PATHS},
        duration_s=1.25,
    )


def test_req_report_5877_spec_declares_exact_transition_contract() -> None:
    """REQ-REPORT-5877: OpenSpec names activated identity, proposal, and range gates."""

    section = SPEC_PATH.read_text(encoding="utf-8")
    section = section[section.index("### REQ-REPORT-5877") :]

    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    assert "(milestone, task_id, declared_deliverable)" in section
    assert "SCENARIO-REPORT-5877-EXACT-ARCHIVE" in section
    assert "SCENARIO-REPORT-5877-APPEND-ONCE" in section
    assert "SCENARIO-REPORT-5877-UNACTIVATED-PROPOSAL" in section
    assert "SCENARIO-REPORT-5877-RANGE-COLLISION" in section
    assert "Exp5877 through Exp5889" in section
    assert "next_range_collision_count=0" in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert mod.FIELD_PRINCIPLES[field] in section


def test_scenario_report_5877_archives_terminal_v522_by_exact_identity(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5877-EXACT-ARCHIVE: activated classes stay disjoint."""

    _make_root(tmp_path)
    report = _build(tmp_path)

    assert report["status"] == "complete"
    assert report["honest_verdict"].startswith("complete:")
    assert report["milestone_transition"] == {
        "source_milestone": "2026.07.522",
        "destination_milestone": "2026.07.523",
        "task_identity_tuple": ["milestone", "task_id", "declared_deliverable"],
    }
    assert len(report["activated_task_and_deliverable_matrix"]) == 7
    exp5866 = report["activated_task_and_deliverable_matrix"][
        "exp5866-adaptive-state-pipeline-shadow-adapter"
    ]
    assert exp5866["present"] is False
    assert exp5866["missing_recorded_explicitly"] is True
    assert exp5866["missing_reason"] == "upstream_retired_exp5865"
    assert exp5866["selection_policy"] == "exact_declared_deliverable"

    classes = report["outcome_classification"]
    assert classes["terminal_class_by_task_id"] == {
        "exp5863-transition-v522": "complete_transition",
        "exp5864-v522-source-delta-ingestion": "no_accepted_source_delta",
        "exp5865-adaptive-state-kernel-requalification": "retired",
        "exp5866-adaptive-state-pipeline-shadow-adapter": "missing_upstream_retired",
        "exp5867-prospective-certified-continuous-learning": "gate_blocked",
        "exp5868-hardness-controlled-constraint-fixture": "ready",
        "exp5869-hardness-surface-headroom-audit": "blocked",
    }
    assert classes["retired_task_ids"] == ["exp5865-adaptive-state-kernel-requalification"]
    assert classes["missing_declared_deliverable_task_ids"] == [
        "exp5866-adaptive-state-pipeline-shadow-adapter"
    ]
    assert classes["gate_blocked_task_ids"] == [
        "exp5867-prospective-certified-continuous-learning"
    ]
    assert classes["ready_task_ids"] == ["exp5868-hardness-controlled-constraint-fixture"]
    assert classes["blocked_task_ids"] == ["exp5869-hardness-surface-headroom-audit"]
    assert classes["verifier_warn_task_ids"] == ["exp5864-v522-source-delta-ingestion"]
    assert report["retired_missing_blocked_and_ready_preserved"] is True
    assert report["next_range_collision_count"] == 0
    assert report["next_task_range"]["start"] == "exp5877"
    assert report["next_task_range"]["end"] == "exp5889"
    assert len(report["adversarial_verifier_receipts"]) == 6
    assert (
        "exp5866-adaptive-state-pipeline-shadow-adapter"
        not in report["adversarial_verifier_receipts"]
    )
    mod.validate_artifact(report)


def test_scenario_report_5877_appends_completion_history_once_when_absent(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5877-APPEND-ONCE: absent .522 history is appended once."""

    _make_root(tmp_path, include_522_complete=False, duplicate_510_blocks=2)
    report = _build(tmp_path)
    after_first = yaml.safe_load((tmp_path / mod.RESEARCH_COMPLETE_RELATIVE_PATH).read_text())
    v522_blocks = [row for row in after_first["milestones"] if row["id"] == mod.MILESTONE_FROM]

    assert report["research_complete_append_count"] == 1
    assert len(v522_blocks) == 1
    assert len(v522_blocks[0]["tasks"]) == 7
    assert not {
        row["id"] for row in v522_blocks[0]["tasks"]
    } & set(mod.UNACTIVATED_PROPOSAL_TASK_IDS)
    assert report["duplicate_history_amplification_count"] == 0
    assert report["preconditions_checked"]["duplicate_history"]["before_duplicate_block_count"] == 1
    assert report["preconditions_checked"]["duplicate_history"]["after_duplicate_block_count"] == 1
    mod.validate_artifact(report)

    second = _build(tmp_path)
    after_second = yaml.safe_load((tmp_path / mod.RESEARCH_COMPLETE_RELATIVE_PATH).read_text())
    assert second["research_complete_append_count"] == 0
    assert [row["id"] for row in after_second["milestones"]].count(mod.MILESTONE_FROM) == 1
    assert second["duplicate_history_amplification_count"] == 0
    mod.validate_artifact(second)


def test_scenario_report_5877_unactivated_proposal_ids_are_not_completed(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5877-UNACTIVATED-PROPOSAL: proposal IDs stay out of history."""

    _make_root(tmp_path)
    report = _build(tmp_path)

    receipt = report["unactivated_proposal_id_receipt"]
    assert receipt["task_ids"] == list(mod.UNACTIVATED_PROPOSAL_TASK_IDS)
    assert receipt["present_in_activated_completion_block"] == []
    assert receipt["appended_as_completed"] is False
    mod.validate_artifact(report)

    data = yaml.safe_load((tmp_path / mod.RESEARCH_COMPLETE_RELATIVE_PATH).read_text())
    v522 = next(row for row in data["milestones"] if row["id"] == mod.MILESTONE_FROM)
    v522["tasks"].append(
        {
            "id": mod.UNACTIVATED_PROPOSAL_TASK_IDS[0],
            "deliverable": "results/experiment_5870_gguf_layer_surface_preflight.json",
        }
    )
    _write_text(tmp_path, mod.RESEARCH_COMPLETE_RELATIVE_PATH, yaml.safe_dump(data))

    laundered = _build(tmp_path)
    assert laundered["status"] == "blocked"
    assert "unactivated_proposal_ids_laundered" in laundered["preconditions_checked"][
        "failed_preconditions"
    ]
    with pytest.raises(ValueError, match="unactivated proposal IDs"):
        mod.validate_artifact(laundered)


def test_scenario_report_5877_unexpected_range_reference_blocks_completion(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5877-RANGE-COLLISION: unexpected Exp5877-Exp5889 hits block."""

    _make_root(tmp_path)
    _write_json(tmp_path, "results/experiment_5883_stale_collision.json", {"status": "stale"})
    report = _build(tmp_path)

    assert report["status"] == "blocked"
    assert report["honest_verdict"].startswith("blocked:")
    assert report["next_range_collision_count"] == 1
    assert report["preconditions_checked"]["range_collision_scan"]["collision_count"] == 1
    assert report["preconditions_checked"]["range_collision_scan"]["collisions"][0]["path"] == (
        "results/experiment_5883_stale_collision.json"
    )
    mod.validate_artifact(report)


def test_scenario_report_5877_schema_checksum_and_protection(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5877-SCHEMA: required fields and protection are enforced."""

    _make_root(tmp_path)
    report = _build(tmp_path)

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(report)
    assert report["docs_reconciled"]["ops_status_md"] == "deferred_by_operator_stop_rule"
    assert report["docs_reconciled"]["ops_changelog_md"] == "deferred_by_operator_stop_rule"
    assert report["docs_reconciled"]["traceability_md"] == "deferred_by_operator_stop_rule"
    assert all(row["unchanged"] for row in report["protected_files_unchanged"].values())
    assert report["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in report["field_provenance"]
        assert field in report["field_principles"]
    mod.validate_artifact(report)

    mutations = [
        (lambda artifact: artifact.pop("status"), "missing required field"),
        (lambda artifact: artifact.update(status="bootstrap"), "status"),
        (lambda artifact: artifact.update(inference_substrate="live_llm_inference"), "inference_substrate"),
        (lambda artifact: artifact.update(honest_verdict="mixed: ambiguous"), "honest_verdict"),
        (lambda artifact: artifact.update(next_range_collision_count="0"), "next_range_collision_count"),
        (
            lambda artifact: artifact.update(status="complete", next_range_collision_count=1),
            "next_range_collision_count must be zero",
        ),
        (lambda artifact: artifact.update(research_complete_append_count=2), "research_complete_append_count"),
        (
            lambda artifact: artifact.update(duplicate_history_amplification_count=1),
            "duplicate_history_amplification_count",
        ),
        (
            lambda artifact: artifact.update(retired_missing_blocked_and_ready_preserved=False),
            "laundered",
        ),
        (
            lambda artifact: artifact["protected_files_unchanged"][
                mod.CONDUCTOR_RELATIVE_PATH.as_posix()
            ].update(unchanged=False),
            "protected file",
        ),
        (lambda artifact: artifact.update(field_provenance=[]), "field provenance"),
        (lambda artifact: artifact["field_provenance"].pop("status"), "field provenance missing"),
        (lambda artifact: artifact.update(outcome_classification=[]), "outcome_classification"),
        (
            lambda artifact: artifact["outcome_classification"]["ready_task_ids"].append(
                "exp5869-hardness-surface-headroom-audit"
            ),
            "terminal classes",
        ),
        (
            lambda artifact: artifact.update(activated_task_and_deliverable_matrix=[]),
            "adversarial verifier receipt matrix",
        ),
        (
            lambda artifact: artifact["outcome_classification"].update(
                terminal_class_by_task_id=[]
            ),
            "terminal class map",
        ),
        (
            lambda artifact: artifact["activated_task_and_deliverable_matrix"].pop(
                "exp5869-hardness-surface-headroom-audit"
            ),
            "exactly seven",
        ),
        (
            lambda artifact: artifact["activated_task_and_deliverable_matrix"].update(
                {"exp5863-transition-v522": []}
            ),
            "malformed matrix row",
        ),
        (
            lambda artifact: artifact["activated_task_and_deliverable_matrix"][
                "exp5863-transition-v522"
            ].update(identity=["2026.07.522", "exp5863-transition-v522", "wrong.json"]),
            "activated identity mismatch",
        ),
        (
            lambda artifact: artifact["activated_task_and_deliverable_matrix"][
                "exp5866-adaptive-state-pipeline-shadow-adapter"
            ].update(present=True),
            "Exp5866 missing deliverable",
        ),
        (
            lambda artifact: artifact["adversarial_verifier_receipts"][
                "exp5863-transition-v522"
            ].pop("receipt_hash"),
            "missing adversarial verifier receipt fields",
        ),
        (
            lambda artifact: artifact["adversarial_verifier_receipts"][
                "exp5863-transition-v522"
            ].update(
                command=(
                    ".venv/bin/python scripts/adversarial_verify.py "
                    "--milestone-range 5863 5869 --json"
                )
            ),
            "adversarial verifier receipt command",
        ),
        (
            lambda artifact: artifact["unactivated_proposal_id_receipt"].update(
                appended_as_completed=True
            ),
            "unactivated proposal IDs",
        ),
        (lambda artifact: artifact.update(next_task_range=[]), "next_task_range"),
        (
            lambda artifact: artifact["next_task_range"].update(end="exp5888"),
            "Exp5889",
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


def test_req_report_5877_helpers_cover_defensive_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-5877: helper branches remain deterministic and auditable."""

    directory = tmp_path / "hashdir"
    _write_text(directory, "child.txt", "content\n")
    assert mod.path_sha256(directory).startswith("sha256:")

    output = tmp_path / "artifact.json"
    mod.write_json(output, {"b": 1})
    assert json.loads(output.read_text(encoding="utf-8")) == {"b": 1}

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

    assert mod._normalize_adversarial_receipts(None, {}) == {}
    assert mod._normalize_adversarial_receipts([{}, "bad", {"task_id": ""}], {}) == {}
    normalized = mod._normalize_adversarial_receipts(
        [_receipt("exp5863-transition-v522")],
        {"exp5863-transition-v522": {"present": True}},
    )
    assert normalized["exp5863-transition-v522"]["flag_count"] == 0
    assert mod._receipt_flag_count({"stdout_json": {"flagged_count": 3}}) == 3
    assert mod._receipt_flags({"stdout_json": {"reports": [{"flags": "bad"}]}}) == []
    assert mod._tests_run_rows(None)[0]["status"] == "not_recorded"
    assert mod._failed_required_test_commands(
        [
            {
                "command": ".venv/bin/python scripts/adversarial_verify.py --json x",
                "exit_code": 1,
            },
            {"command": "nonblocking failure", "exit_code": 1, "blocking": False},
            {"command": "real failure", "exit_code": 2},
        ]
    ) == ["real failure"]
    assert mod._failed_nonblocking_test_commands(
        [
            {"command": "nonblocking failure", "exit_code": 1, "blocking": False},
            {"command": "passing", "exit_code": 0, "blocking": False},
        ]
    ) == ["nonblocking failure"]
    assert mod._status_from_log(None) == "MISSING"
    assert mod._status_from_log("| x | UNKNOWN | y |") == "LOGGED"
    assert mod._task_signature({"tasks": "bad"}) == ()
    assert mod._completion_task_rows(tmp_path / "no-history") == []
    append_root = tmp_path / "append-missing-file"
    append_receipt = mod._append_completion_if_absent(append_root)
    assert append_receipt["append_count"] == 1
    assert (append_root / mod.RESEARCH_COMPLETE_RELATIVE_PATH).exists()

    payloads = {task_id: {"status": "complete"} for task_id in mod.EXPECTED_TASK_IDS}
    metadata = {task_id: {"present": True} for task_id in mod.EXPECTED_TASK_IDS}
    metadata["exp5863-transition-v522"] = {"present": False}
    payloads["exp5864-v522-source-delta-ingestion"] = {"status": "mystery"}
    classes = mod._classify_outcomes(payloads, metadata, {})
    assert "exp5863-transition-v522" in classes["missing_declared_deliverable_task_ids"]
    assert "exp5864-v522-source-delta-ingestion" in classes["off_path_task_ids"]

    real_write_text = Path.write_text

    def broken_probe_write(path: Path, *args: Any, **kwargs: Any) -> int:
        if path.name.endswith(".tmp-probe"):
            raise OSError("fixture")
        return real_write_text(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_text", broken_probe_write)
        receipt = mod._atomic_output_receipt(tmp_path / "artifact.json")
        assert receipt["ok"] is False
        assert receipt["error"].startswith("OSError:")

    _make_root(tmp_path / "missing-receipt")
    report = mod.build_report(
        tmp_path / "missing-receipt",
        adversarial_receipts={},
        tests_run=[{"command": "focused", "exit_code": 1}],
        modification_overrides={rel_path: False for rel_path in mod.PROTECTED_FILE_PATHS},
        duration_s=1.0,
    )
    assert report["status"] == "blocked"
    assert report["preconditions_checked"]["failed_required_test_commands"] == ["focused"]
    assert "missing_adversarial_receipts" in report["preconditions_checked"][
        "failed_preconditions"
    ]
    with pytest.raises(ValueError, match="missing adversarial verifier receipt"):
        mod.validate_artifact(report)

    bad_root = tmp_path / "bad-preconditions"
    _make_root(bad_root)
    _write_text(bad_root, mod.ROADMAP_RELATIVE_PATH, "a: [\n")
    _write_text(bad_root, mod.RESEARCH_COMPLETE_RELATIVE_PATH, "a: [\n")
    (bad_root / mod.ADVERSARIAL_VERIFY_RELATIVE_PATH).unlink()

    with monkeypatch.context() as patch:
        patch.setattr(
            mod,
            "_atomic_output_receipt",
            lambda path: {"ok": False, "declared_path": path.as_posix()},
        )
        patch.setattr(
            mod,
            "_resource_receipts",
            lambda root: {
                "disk": {"ok": False, "available_mb": 1, "required_mb": 512},
                "memory": {"ok": True, "available_mb": 1024, "required_mb": 512},
            },
        )
        bad_report = mod.build_report(
            bad_root,
            adversarial_receipts={},
            tests_run=[{"command": "focused", "exit_code": 0}],
            modification_overrides={
                **{rel_path: False for rel_path in mod.PROTECTED_FILE_PATHS},
                mod.CONDUCTOR_RELATIVE_PATH: True,
            },
            duration_s=1.0,
        )
    assert {
        "active_roadmap_unloadable",
        "live_verifier_missing",
        "atomic_output_unavailable",
        "insufficient_resources",
        "research_complete_unparseable",
        "protected_file_modified",
    } <= set(bad_report["preconditions_checked"]["failed_preconditions"])

    amp_root = tmp_path / "duplicate-amplification"
    _make_root(amp_root)

    def amplified_append(root: Path) -> JsonDict:
        return {
            "append_count": 0,
            "appended": False,
            "reason": "fixture",
            "before_sha256": "sha256:before",
            "after_sha256": "sha256:after",
            "before_duplicate_block_count": 0,
            "after_duplicate_block_count": 1,
            "duplicate_history_amplification_count": 1,
        }

    with monkeypatch.context() as patch:
        patch.setattr(mod, "_append_completion_if_absent", amplified_append)
        amp_report = mod.build_report(
            amp_root,
            adversarial_receipts=_receipts(),
            tests_run=[{"command": "focused", "exit_code": 0}],
            modification_overrides={rel_path: False for rel_path in mod.PROTECTED_FILE_PATHS},
            duration_s=1.0,
        )
    assert "duplicate_history_amplified" in amp_report["preconditions_checked"][
        "failed_preconditions"
    ]

    not_preserved_root = tmp_path / "not-preserved"
    _make_root(not_preserved_root)
    with monkeypatch.context() as patch:
        patch.setattr(mod, "_retired_missing_blocked_and_ready_preserved", lambda classes: False)
        not_preserved = mod.build_report(
            not_preserved_root,
            adversarial_receipts=_receipts(),
            tests_run=[{"command": "focused", "exit_code": 0}],
            modification_overrides={rel_path: False for rel_path in mod.PROTECTED_FILE_PATHS},
            duration_s=1.0,
        )
    assert "terminal_outcomes_not_preserved" in not_preserved["preconditions_checked"][
        "failed_preconditions"
    ]

    mismatch_root = tmp_path / "declared-mismatch"
    _make_root(mismatch_root)
    data = yaml.safe_load((mismatch_root / mod.RESEARCH_COMPLETE_RELATIVE_PATH).read_text())
    v522 = next(row for row in data["milestones"] if row["id"] == mod.MILESTONE_FROM)
    v522["tasks"][0]["deliverable"] = "results/experiment_5863_alias.json"
    _write_json(mismatch_root, "results/experiment_5863_alias.json", _artifact("exp5863-transition-v522"))
    _write_text(mismatch_root, mod.RESEARCH_COMPLETE_RELATIVE_PATH, yaml.safe_dump(data))
    mismatch_report = mod.build_report(
        mismatch_root,
        adversarial_receipts={
            **_receipts(),
            "exp5863-transition-v522": {
                **_receipt("exp5863-transition-v522"),
                "artifact_path": "results/experiment_5863_alias.json",
                "command": (
                    ".venv/bin/python scripts/adversarial_verify.py --json "
                    "results/experiment_5863_alias.json"
                ),
            },
        },
        tests_run=[{"command": "focused", "exit_code": 0}],
        modification_overrides={rel_path: False for rel_path in mod.PROTECTED_FILE_PATHS},
        duration_s=1.0,
    )
    assert "declared_deliverable_mismatch" in mismatch_report["preconditions_checked"][
        "failed_preconditions"
    ]
