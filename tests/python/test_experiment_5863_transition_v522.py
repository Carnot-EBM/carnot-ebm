"""Tests for the Exp5863 V522 transition receipt.

Spec refs: REQ-REPORT-5863, SCENARIO-REPORT-5863-EXACT-ARCHIVE,
SCENARIO-REPORT-5863-APPEND-ONCE, SCENARIO-REPORT-5863-RANGE-COLLISION,
SCENARIO-REPORT-5863-SCHEMA.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_5863_transition_v522 as mod


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
        "exp5849-transition-v521": {
            "status": "complete",
            "honest_verdict": "complete: archived .520 into .521",
            "next_range_collision_count": 0,
        },
        "exp5850-v521-source-delta-ingestion": {
            "status": "complete",
            "honest_verdict": "complete: no accepted post-V521 source deltas",
            "accepted_finding_count": 0,
            "references_modified": False,
        },
        "exp5851-deterministic-replay-provenance-contract": {
            "status": "ready",
            "honest_verdict": "ready: deterministic_replay_provenance_contract_clean",
            "deterministic_replay_contract_ready_score": 1.0,
        },
        "exp5852-three-family-paired-embeddings": {
            "status": "complete",
            "honest_verdict": "ready: paired_embedding_corpus_complete_all_three_models",
            "paired_embedding_corpus_ready_score": 1.0,
        },
        "exp5853-paired-embedding-integrity-audit": {
            "status": "disqualified",
            "honest_verdict": "disqualified: raw_model_dimension_identity_shortcut",
            "paired_embedding_integrity_ready_score": 0.0,
            "surviving_shortcuts": ["raw_model_dimension_identity_shortcut"],
        },
        "exp5854-portable-comparative-energy-controls": {
            "schema": "blocked_gate_check_v1",
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gates_evaluated": [
                {
                    "upstream": "exp5853-paired-embedding-integrity-audit",
                    "artifact_field": "paired_embedding_integrity_ready_score",
                    "expected": 1.0,
                    "actual": 0.0,
                    "passed": False,
                }
            ],
        },
        "exp5856-provenance-correct-lifecycle": {
            "status": "complete",
            "honest_verdict": "complete: provenance_correct_lifecycle",
            "adaptive_memory_lifecycle_ready_score": 1.0,
        },
        "exp5857-clean-transfer-selective-replay": {
            "status": "qualified",
            "honest_verdict": "qualified: clean_replay",
            "selective_replay_qualified_score": 1.0,
            "unsafe_transfer_count": 0,
        },
        "exp5858-reduced-oracle-continuous-self-learning": {
            "status": "ready",
            "honest_verdict": "ready: reduced_oracle_continuous_self_learning",
            "continuous_self_learning_ready_score": 1.0,
            "unsafe_accept_count": 0,
        },
        "exp5859-adaptive-state-microkernel-parity": {
            "status": "blocked",
            "honest_verdict": "blocked: adaptive_state_microkernel_conformance_incomplete",
            "adaptive_state_microkernel_ready_score": 0.0,
            "test_exit_codes": {".venv/bin/pytest tests/python -q": 2},
        },
        "exp5860-live-active-observation-ab": {
            "status": "complete_null",
            "honest_verdict": "complete_null: active_observation_no_positive_bound",
            "flagged_adversarial": True,
            "active_observation_ready_score": 0.0,
            "corrigendum_pending": [
                {"kind": "DURATION_TOO_SHORT", "severity": "critical", "detail": "fixture"}
            ],
        },
        "exp5861-attached-board-state-receipts": {
            "status": "no_change_no_authenticated_state_operation_execution",
            "honest_verdict": "no-change: exp5859_not_ready no_speedup",
            "authenticated_state_operation_parity_score": 0.0,
        },
        "exp5862-v521-capstone-reconciliation": {
            "status": "blocked",
            "honest_verdict": "blocked: required capstone checks failed during V521 reconciliation",
            "outcome_classification": {
                "disqualified": ["exp5853-paired-embedding-integrity-audit"],
                "gated_skip": [
                    "exp5854-portable-comparative-energy-controls",
                    "exp5855-exact-release-shadow-routing",
                ],
                "blocked": ["exp5859-adaptive-state-microkernel-parity"],
                "flagged": ["exp5860-live-active-observation-ab"],
            },
        },
    }
    return payloads[task_id]


def _completion_payload(*, include_521: bool = True, duplicate_510_blocks: int = 1) -> JsonDict:
    duplicate_block = {
        "id": "2026.07.510",
        "title": "Historical duplicate",
        "doc": "openspec/change-proposals/research-roadmap-vNEXT.md",
        "completed": "2026-07-17",
        "finding": "fixture",
        "tasks": [{"id": "exp5706-transition-v510", "deliverable": "results/x.json"}],
    }
    milestones = [deepcopy(duplicate_block) for _ in range(duplicate_510_blocks)]
    if include_521:
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
                        "title": mod.TASK_TITLES[task_id],
                        "deliverable": rel_path.as_posix(),
                        "result": "OK (conductor)",
                    }
                    for task_id, rel_path in mod.TASK_ARTIFACT_PATHS.items()
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
        "**Milestone:** 2026.07.522",
        "**Task range:** Exp5863-Exp5876",
        "",
    ]
    lines.extend(path.as_posix() for path in mod.NEXT_TASK_ARTIFACT_PATHS.values())
    return "\n".join(lines) + "\n"


def _conductor_log() -> str:
    return "\n".join(
        [
            "| 2026-07-23 12:05 UTC | Exact terminal-boundary handoff from .520 into .52 | OK | 87 passed |",
            "| 2026-07-23 16:02 UTC | Claim-flip, evaluator-swap, and identity-shortcut  | OK | 87 passed |",
            "| 2026-07-23 16:09 UTC | Held-model and held-constraint comparative energy  | GATE_BLOCK | gate failed |",
            "| 2026-07-23 16:53 UTC | Exact-authority shadow routing after a portable en | GATE_BLOCK | pre-emptive skip |",
            "| 2026-07-23 19:27 UTC | Accepted adaptive operations ABI conformance | FAIL | pytest exit 2 |",
            "| 2026-07-23 20:05 UTC | Closed-loop visual probing under equal action budg | FLAGGED | verifier critical |",
            "| 2026-07-23 23:34 UTC | Four-branch terminal decision ledger and milestone | FAIL | capstone blocked |",
        ]
    )


def _make_root(
    root: Path,
    *,
    include_521_complete: bool = True,
    duplicate_510_blocks: int = 1,
) -> None:
    for task_id, rel_path in mod.TASK_ARTIFACT_PATHS.items():
        if task_id == "exp5855-exact-release-shadow-routing":
            continue
        _write_json(root, rel_path, _artifact(task_id))
    _write_text(root, mod.ROADMAP_RELATIVE_PATH, yaml.safe_dump(_active_roadmap_payload()))
    _write_text(
        root,
        mod.RESEARCH_COMPLETE_RELATIVE_PATH,
        yaml.safe_dump(
            _completion_payload(
                include_521=include_521_complete,
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
        "python/carnot/experiment_5863_transition_v522.py",
        "# owned exp5863 transition fixture\n",
    )
    _write_text(
        root,
        "tests/python/test_experiment_5863_transition_v522.py",
        "# owned exp5863 test fixture\n",
    )
    for rel_path in mod.PROTECTED_FILE_PATHS + (mod.SPEC_RELATIVE_PATH,):
        if not (root / rel_path).exists():
            _write_text(root, rel_path, f"{rel_path.as_posix()} fixture\n")


def _receipt(task_id: str, *, flagged: bool = False) -> JsonDict:
    flags = (
        [{"kind": "DURATION_TOO_SHORT", "severity": "critical", "detail": "fixture"}]
        if flagged
        else []
    )
    stdout_json = {
        "reports": [
            {
                "path": mod.TASK_ARTIFACT_PATHS[task_id].as_posix(),
                "flag_count": len(flags),
                "flags": flags,
                "max_severity": 2 if flags else -1,
            }
        ],
        "flagged_count": 1 if flags else 0,
    }
    return {
        "task_id": task_id,
        "artifact_path": mod.TASK_ARTIFACT_PATHS[task_id].as_posix(),
        "command": (
            ".venv/bin/python scripts/adversarial_verify.py --json "
            f"{mod.TASK_ARTIFACT_PATHS[task_id].as_posix()}"
        ),
        "exit_code": 1 if flags else 0,
        "stdout_json": stdout_json,
        "stderr": "",
        "receipt_hash": mod.sha256_json(stdout_json),
    }


def _receipts() -> dict[str, JsonDict]:
    return {
        task_id: _receipt(task_id, flagged=task_id == "exp5860-live-active-observation-ab")
        for task_id in mod.TASK_ARTIFACT_PATHS
        if task_id != "exp5855-exact-release-shadow-routing"
    }


def _build(root: Path) -> JsonDict:
    return mod.build_report(
        root,
        adversarial_receipts=_receipts(),
        tests_run=[
            {
                "command": ".venv/bin/pytest tests/python/test_experiment_5863_transition_v522.py -q",
                "exit_code": 0,
            },
            {"command": ".venv/bin/pytest tests/python -q", "exit_code": 0},
        ],
        modification_overrides={rel_path: False for rel_path in mod.PROTECTED_FILE_PATHS},
        duration_s=1.25,
    )


def test_req_report_5863_spec_declares_exact_transition_contract() -> None:
    """REQ-REPORT-5863: OpenSpec names identity, append, collision, and field gates."""

    section = SPEC_PATH.read_text(encoding="utf-8")
    section = section[section.index("### REQ-REPORT-5863") :]

    assert mod.RESULT_RELATIVE_PATH.as_posix() in section
    assert "(milestone, task_id, declared_deliverable)" in section
    assert "SCENARIO-REPORT-5863-EXACT-ARCHIVE" in section
    assert "SCENARIO-REPORT-5863-APPEND-ONCE" in section
    assert "SCENARIO-REPORT-5863-RANGE-COLLISION" in section
    assert "Exp5863 through Exp5876" in section
    assert "next_range_collision_count=0" in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert mod.FIELD_PRINCIPLES[field] in section


def test_scenario_report_5863_archives_terminal_v521_by_exact_identity(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5863-EXACT-ARCHIVE: terminal classes stay disjoint."""

    _make_root(tmp_path)
    report = _build(tmp_path)

    assert report["status"] == "complete"
    assert report["honest_verdict"].startswith("complete:")
    assert report["milestone_transition"] == {
        "source_milestone": "2026.07.521",
        "destination_milestone": "2026.07.522",
        "task_identity_tuple": ["milestone", "task_id", "declared_deliverable"],
    }
    assert len(report["exact_task_and_deliverable_matrix"]) == 14
    exp5855 = report["exact_task_and_deliverable_matrix"][
        "exp5855-exact-release-shadow-routing"
    ]
    assert exp5855["present"] is False
    assert exp5855["missing_recorded_explicitly"] is True
    assert exp5855["selection_policy"] == "exact_declared_deliverable"

    classes = report["outcome_classification"]
    assert classes["disqualified_task_ids"] == ["exp5853-paired-embedding-integrity-audit"]
    assert classes["gate_skipped_task_ids"] == [
        "exp5854-portable-comparative-energy-controls",
        "exp5855-exact-release-shadow-routing",
    ]
    assert classes["blocked_task_ids"] == [
        "exp5859-adaptive-state-microkernel-parity",
        "exp5862-v521-capstone-reconciliation",
    ]
    assert classes["flagged_task_ids"] == ["exp5860-live-active-observation-ab"]
    assert classes["no_change_task_ids"] == ["exp5861-attached-board-state-receipts"]
    assert classes["missing_declared_deliverable_task_ids"] == [
        "exp5855-exact-release-shadow-routing"
    ]
    assert set(classes["clean_promoted_task_ids"]).isdisjoint(
        set(classes["disqualified_task_ids"])
        | set(classes["gate_skipped_task_ids"])
        | set(classes["blocked_task_ids"])
        | set(classes["flagged_task_ids"])
        | set(classes["no_change_task_ids"])
    )
    assert report["blocked_disqualified_skipped_flagged_and_no_change_preserved"] is True
    assert report["next_range_collision_count"] == 0
    assert report["next_task_range"]["start"] == "exp5863"
    assert report["next_task_range"]["end"] == "exp5876"
    assert len(report["adversarial_verifier_receipts"]) == 13
    assert "exp5855-exact-release-shadow-routing" not in report["adversarial_verifier_receipts"]
    mod.validate_artifact(report)


def test_scenario_report_5863_appends_completion_history_once_when_absent(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5863-APPEND-ONCE: absent .521 history is appended once."""

    _make_root(tmp_path, include_521_complete=False, duplicate_510_blocks=2)
    report = _build(tmp_path)
    after_first = yaml.safe_load((tmp_path / mod.RESEARCH_COMPLETE_RELATIVE_PATH).read_text())
    v521_blocks = [row for row in after_first["milestones"] if row["id"] == mod.MILESTONE_FROM]

    assert report["research_complete_append_count"] == 1
    assert len(v521_blocks) == 1
    assert len(v521_blocks[0]["tasks"]) == 14
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


def test_scenario_report_5863_unexpected_range_reference_blocks_completion(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5863-RANGE-COLLISION: unexpected Exp5863-Exp5876 hits block."""

    _make_root(tmp_path)
    _write_json(tmp_path, "results/experiment_5864_stale_collision.json", {"status": "stale"})
    report = _build(tmp_path)

    assert report["status"] == "blocked"
    assert report["honest_verdict"].startswith("blocked:")
    assert report["next_range_collision_count"] == 1
    assert report["preconditions_checked"]["range_collision_scan"]["collision_count"] == 1
    assert report["preconditions_checked"]["range_collision_scan"]["collisions"][0]["path"] == (
        "results/experiment_5864_stale_collision.json"
    )
    mod.validate_artifact(report)


def test_scenario_report_5863_schema_checksum_and_protection(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5863-SCHEMA: required fields and protection are enforced."""

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
        (lambda artifact: artifact.update(inference_substrate="live_llm_inference"), "inference_substrate"),
        (lambda artifact: artifact.update(honest_verdict="mixed: ambiguous"), "honest_verdict"),
        (lambda artifact: artifact.update(next_range_collision_count="0"), "next_range_collision_count"),
        (
            lambda artifact: artifact.update(status="complete", next_range_collision_count=1),
            "next_range_collision_count must be zero",
        ),
        (lambda artifact: artifact.update(research_complete_append_count=2), "research_complete_append_count"),
        (
            lambda artifact: artifact.update(
                duplicate_history_amplification_count=1
            ),
            "duplicate_history_amplification_count",
        ),
        (
            lambda artifact: artifact.update(
                blocked_disqualified_skipped_flagged_and_no_change_preserved=False
            ),
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
            lambda artifact: artifact.update(exact_task_and_deliverable_matrix=[]),
            "adversarial verifier receipt matrix",
        ),
        (
            lambda artifact: artifact["outcome_classification"]["clean_promoted_task_ids"].append(
                "exp5860-live-active-observation-ab"
            ),
            "negative evidence promoted",
        ),
        (
            lambda artifact: artifact["adversarial_verifier_receipts"][
                "exp5849-transition-v521"
            ].pop("receipt_hash"),
            "missing adversarial verifier receipt fields",
        ),
        (
            lambda artifact: artifact["adversarial_verifier_receipts"][
                "exp5849-transition-v521"
            ].update(
                command=(
                    ".venv/bin/python scripts/adversarial_verify.py "
                    "--milestone-range 5849 5862 --json"
                )
            ),
            "adversarial verifier receipt command",
        ),
    ]
    for mutate, needle in mutations:
        artifact = deepcopy(report)
        mutate(artifact)
        artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
        with pytest.raises(ValueError, match=needle):
            mod.validate_artifact(artifact)

    checksum_drift = deepcopy(report)
    checksum_drift["status"] = "changed_after_checksum"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(checksum_drift)


def test_req_report_5863_helpers_cover_defensive_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-5863: helper branches remain deterministic and auditable."""

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

    missing_receipts = mod._normalize_adversarial_receipts(None, {})
    assert missing_receipts == {}
    assert mod._normalize_adversarial_receipts([{}, "bad", {"task_id": ""}], {}) == {}
    normalized = mod._normalize_adversarial_receipts(
        [_receipt("exp5849-transition-v521")],
        {"exp5849-transition-v521": {"present": True}},
    )
    assert normalized["exp5849-transition-v521"]["flag_count"] == 0
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
    metadata["exp5849-transition-v521"] = {"present": False}
    payloads["exp5850-v521-source-delta-ingestion"] = {"status": "mystery"}
    classes = mod._classify_outcomes(payloads, metadata, {})
    assert "exp5849-transition-v521" in classes["off_path_task_ids"]
    assert "exp5850-v521-source-delta-ingestion" in classes["off_path_task_ids"]

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
        patch.setattr(mod, "_negative_evidence_preserved", lambda classes: False)
        not_preserved = mod.build_report(
            not_preserved_root,
            adversarial_receipts=_receipts(),
            tests_run=[{"command": "focused", "exit_code": 0}],
            modification_overrides={rel_path: False for rel_path in mod.PROTECTED_FILE_PATHS},
            duration_s=1.0,
        )
    assert "terminal_negative_evidence_not_preserved" in not_preserved[
        "preconditions_checked"
    ]["failed_preconditions"]
