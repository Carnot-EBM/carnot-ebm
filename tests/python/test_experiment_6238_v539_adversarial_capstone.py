"""Tests for Exp6238 V539 exact-path adversarial capstone.

Spec refs: REQ-INFRA-6238, SCENARIO-INFRA-6238-1,
SCENARIO-INFRA-6238-2, SCENARIO-INFRA-6238-3,
SCENARIO-INFRA-6238-4, SCENARIO-INFRA-6238-5,
SCENARIO-INFRA-6238-6.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest
import yaml

from carnot import experiment_6238_v539_adversarial_capstone as mod
from carnot.experiment_artifacts import ARTIFACT_ROOT_ENV


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/research-harnesses/spec.md"


def _write_artifact(path: Path, status: str, verdict: str | None = None, **extra: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "honest_verdict": verdict if verdict is not None else f"{status}: fixture",
        "duration_s": 1.0,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "verifier_is_oracle": False,
        "reproducibility_checksum": "sha256:fixture",
        **extra,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fake_reviews(task_ids: list[str]) -> dict[str, dict[str, object]]:
    return {
        task_id: {
            "summary": {
                "command": f"summarize {task_id}",
                "exit_code": 0,
                "classification": "passed",
                "stdout_tail": "clean",
                "stderr_tail": "",
            },
            "adversarial": {
                "path": task_id,
                "flag_count": 0,
                "critical_flag_count": 0,
                "warn_flag_count": 0,
                "flags": [],
            },
        }
        for task_id in task_ids
    }


def test_req_infra_6238_spec_declares_capstone_contract() -> None:
    """REQ-INFRA-6238: OpenSpec names the exact-path V539 capstone."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6238") :]

    for token in (
        "REQ-INFRA-6238",
        "SCENARIO-INFRA-6238-1",
        "SCENARIO-INFRA-6238-2",
        "SCENARIO-INFRA-6238-3",
        "SCENARIO-INFRA-6238-4",
        "SCENARIO-INFRA-6238-5",
        "SCENARIO-INFRA-6238-6",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert token in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_exact_task_matrix_uses_declared_paths_and_ignores_aliases(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6238-1: exact paths win over same-number sidecars."""

    exact = tmp_path / "results/experiment_9000_declared.json"
    alias = tmp_path / "results/experiment_9000_sidecar.json"
    _write_artifact(exact, "complete", "complete: exact")
    _write_artifact(alias, "blocked", "blocked_gate_check_failed")

    matrix = mod.build_exact_task_artifact_matrix(
        tmp_path,
        [
            {
                "id": "exp9000-declared",
                "title": "Declared fixture",
                "deliverable": "results/experiment_9000_declared.json",
                "track": "infrastructure",
            }
        ],
    )

    row = matrix["exp9000-declared"]
    assert row["declared_deliverable"] == "results/experiment_9000_declared.json"
    assert row["terminal_class"] == "complete"
    assert row["terminal"] is True
    assert row["same_number_alias_used"] is False
    assert row["same_number_alias_candidates_ignored"] == ["results/experiment_9000_sidecar.json"]

    missing = mod.build_exact_task_artifact_matrix(
        tmp_path,
        [
            {
                "id": "exp9001-missing",
                "title": "Missing fixture",
                "deliverable": "results/experiment_9001_declared.json",
                "track": "verification",
            }
        ],
    )
    assert missing["exp9001-missing"]["terminal_class"] == "missing"
    assert missing["exp9001-missing"]["terminal"] is False


def test_gate_cascade_recomputes_exact_upstream_fields(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6238-3: structured gates fail closed on null or missing fields."""

    _write_artifact(
        tmp_path / "results/upstream.json",
        "complete",
        "complete: upstream",
        ready_score=0,
        null_score=None,
    )
    tasks = [
        {"id": "upstream", "deliverable": "results/upstream.json", "track": "infrastructure"},
        {
            "id": "downstream-pass",
            "deliverable": "results/pass.json",
            "track": "verification",
            "gated_on": [
                {"upstream": "upstream", "artifact_field": "ready_score", "op": "==", "value": 0}
            ],
        },
        {
            "id": "downstream-fail",
            "deliverable": "results/fail.json",
            "track": "verification",
            "gated_on": [
                {"upstream": "upstream", "artifact_field": "ready_score", "op": "==", "value": 1},
                {
                    "upstream": "upstream",
                    "artifact_field": "missing_score",
                    "op": "exists",
                    "value": True,
                },
                {
                    "upstream": "missing-upstream",
                    "artifact_field": "ready_score",
                    "op": "==",
                    "value": 1,
                },
            ],
        },
    ]

    receipts = mod.evaluate_gate_cascades(tmp_path, tasks)

    assert receipts["passed_count"] == 1
    assert receipts["failed_count"] == 3
    assert receipts["gates"][0]["passed"] is True
    assert receipts["gates"][1]["actual"] == 0
    assert receipts["gates"][1]["passed"] is False
    assert receipts["gates"][2]["actual"] is None
    assert receipts["gates"][2]["reason"] == "missing_upstream_field"
    assert receipts["gates"][3]["reason"] == "missing_upstream_artifact"
    assert receipts["gates"][3]["principle"] == mod.GATE_PRINCIPLE


def test_current_v539_report_preserves_missing_flagged_and_hardware_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-INFRA-6238-4: branch summaries use exact upstream evidence."""

    roadmap = mod.load_roadmap(REPO)
    task_ids = [str(task["id"]) for task in roadmap["tasks"]]
    reviews = _fake_reviews(task_ids)
    reviews["exp6225-v539-terminal-transition"]["adversarial"] = {
        "path": "results/experiment_6225_v539_terminal_transition.json",
        "flag_count": 2,
        "critical_flag_count": 1,
        "warn_flag_count": 1,
        "flags": [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}],
    }
    monkeypatch.setattr(mod, "live_artifact_reviews", lambda root, matrix: reviews)
    monkeypatch.setattr(
        mod,
        "determination_preservation_receipt",
        lambda root: {
            "command": mod.PRESERVATION_COMMAND,
            "exit_code": 0,
            "classification": "passed",
            "stdout_tail": "OK",
            "stderr_tail": "",
        },
    )

    report = mod.build_report(
        REPO,
        date="20260809",
        command_receipts=[{"command": "focused", "exit_code": 0}],
        before_hashes=mod.protected_hashes(REPO),
        started_at=0.0,
    )

    assert mod.validate_report(report) == []
    assert report["roadmap_path_hash_and_task_ids"]["task_ids"] == list(mod.EXPECTED_V539_TASK_IDS)
    counts = report["missing_blocked_skipped_partial_null_flagged_retired_and_ready_counts"]
    assert counts["missing"] == 2
    assert counts["flagged"] >= 3
    assert counts["hardware_claim_count"] == 0
    assert (
        counts["count_principles"]["hardware_claim_count"]
        == mod.COUNT_PRINCIPLES["hardware_claim_count"]
    )
    assert (
        report["runtime_final_status_and_family_scores"]["three_family_runtime_ready_score"] is None
    )
    assert report["runtime_final_status_and_family_scores"]["claim_allowed"] is False
    assert (
        report["arc_provenance_registry_hash_level_depth_and_promotion_summary"][
            "portfolio_promotion_ready_score"
        ]
        == 0.0
    )
    assert report["code_parse_recovery_and_content_margin_summary"]["content_margin_claim"] is False
    assert report["fresh_stream_and_continuous_learning_summary"]["fresh_stream_claim"] is False
    assert report["shadow_consumer_summary"]["shadow_reachability_claim"] is False
    sampler = report["sampler_activation_quality_equivalence_and_default_summary"]
    assert sampler["treatment_activation_score"] == 1.0
    assert sampler["decision"] == "equivalence_supported"
    assert sampler["default_off_preserved"] is True
    assert report["hardware_boundary_and_claim_count"]["hardware_claim_count"] == 0
    assert (
        report["spec_traceability_status_changelog_known_issues_updates"][
            "ops_status_changelog_traceability_touched"
        ]
        is False
    )


def test_prior_failure_retirement_records_candidates_without_manifest_update(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6238-5: retire-if-same-verdict is recorded without guessing."""

    tasks = [
        {
            "id": "same",
            "deliverable": "results/same.json",
            "prior_failures": [
                {
                    "experiment_id": "exp1",
                    "verdict": "complete: same failure",
                    "addressed_by": "changed",
                    "retire_if_same_verdict": True,
                }
            ],
        },
        {
            "id": "different",
            "deliverable": "results/different.json",
            "prior_failures": [
                {
                    "experiment_id": "exp2",
                    "verdict": "complete: old failure",
                    "addressed_by": "changed",
                    "retire_if_same_verdict": True,
                }
            ],
        },
        {
            "id": "missing",
            "deliverable": "results/missing.json",
            "prior_failures": [
                {
                    "experiment_id": "exp3",
                    "verdict": "missing: old failure",
                    "addressed_by": "changed",
                    "retire_if_same_verdict": True,
                }
            ],
        },
    ]
    _write_artifact(tmp_path / "results/same.json", "complete", "complete: same failure")
    _write_artifact(tmp_path / "results/different.json", "complete", "complete: new outcome")
    matrix = mod.build_exact_task_artifact_matrix(tmp_path, tasks)

    actions = mod.prior_failure_retirement_actions(tasks, matrix)

    assert actions["manifest_update_count"] == 0
    assert actions["rule_fired_count"] == 1
    assert actions["actions"][0]["task_id"] == "same"
    assert actions["actions"][0]["would_update_exclusion_manifest"] is False
    assert actions["actions"][1]["action"] == "no_retirement_current_verdict_differs"
    assert actions["actions"][2]["action"] == "no_retirement_exact_artifact_missing"


def test_report_validation_requires_principles_checksum_and_bare_hardware_zero() -> None:
    """SCENARIO-INFRA-6238-6: report schema is machine-checkable."""

    report = {field: f"fixture-{field}" for field in mod.REQUIRED_ARTIFACT_FIELDS}
    report["status"] = "complete"
    report["inference_substrate"] = mod.INFERENCE_SUBSTRATE
    report["verifier_is_oracle"] = False
    report["field_principles"] = dict(mod.FIELD_PRINCIPLES)
    report["field_provenance"] = {
        field: {"sources": ["REQ-INFRA-6238"], "principle": mod.FIELD_PRINCIPLES[field]}
        for field in mod.REQUIRED_ARTIFACT_FIELDS
    }
    report["missing_blocked_skipped_partial_null_flagged_retired_and_ready_counts"] = {
        **{key: 0 for key in mod.COUNT_PRINCIPLES},
        "count_principles": dict(mod.COUNT_PRINCIPLES),
    }
    report["gate_cascade_receipts"] = {
        "gates": [{"principle": mod.GATE_PRINCIPLE}],
        "passed_count": 0,
        "failed_count": 0,
    }
    report["hardware_boundary_and_claim_count"] = {
        "hardware_claim_count": 0,
        "principle": mod.COUNT_PRINCIPLES["hardware_claim_count"],
    }
    report["duration_s"] = 1.0
    report["honest_verdict"] = "complete: fixture"
    report["reproducibility_checksum"] = ""
    report["reproducibility_checksum"] = mod.payload_checksum(report)

    assert mod.validate_report(report) == []

    broken = dict(report)
    broken["hardware_boundary_and_claim_count"] = {"hardware_claim_count": 1}
    assert "hardware_claim_count must be bare 0" in mod.validate_report(broken)

    broken = dict(report)
    broken["field_principles"] = {}
    assert "missing field_principles entry: status" in mod.validate_report(broken)


def test_write_report_uses_artifact_root_override(tmp_path: Path) -> None:
    """REQ-INFRA-6238: writes are atomic and test-isolated."""

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    report = mod.build_report(
        REPO,
        date="20260809",
        command_receipts=[],
        artifact_reviews=_fake_reviews(list(mod.EXPECTED_V539_TASK_IDS)),
        preservation_receipt={
            "command": mod.PRESERVATION_COMMAND,
            "exit_code": 0,
            "classification": "passed",
            "stdout_tail": "OK",
            "stderr_tail": "",
        },
        before_hashes=mod.protected_hashes(REPO),
        started_at=0.0,
    )

    path = mod.write_report(report, REPO, env={ARTIFACT_ROOT_ENV: str(artifact_root)})

    assert path == artifact_root / mod.RESULT_RELATIVE_PATH.name
    assert json.loads(path.read_text(encoding="utf-8")) == report


def test_fixture_root_builds_report_with_minimal_inputs(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6238-1: preconditions and docs are hashed before classification."""

    root = tmp_path / "repo"
    root.mkdir()
    shutil.copyfile(REPO / "research-roadmap.yaml", root / "research-roadmap.yaml")
    for rel in mod.PRECONDITION_RELATIVE_PATHS:
        path = root / rel
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{rel.as_posix()} fixture\n", encoding="utf-8")
    (root / "results").mkdir(exist_ok=True)

    report = mod.build_report(
        root,
        date="20260809",
        command_receipts=[],
        artifact_reviews=_fake_reviews(list(mod.EXPECTED_V539_TASK_IDS)),
        preservation_receipt={
            "command": mod.PRESERVATION_COMMAND,
            "exit_code": 0,
            "classification": "passed",
            "stdout_tail": "OK",
            "stderr_tail": "",
        },
        before_hashes=mod.protected_hashes(root),
        started_at=0.0,
    )

    assert report["preconditions_checked"]["declared_deliverable_hashes"]
    assert report["research_complete_reconciliation"]["action"] == "recorded_only_no_mutation"
    assert report["protected_files_unchanged"]["unchanged"] is True
    assert (
        report["exact_task_artifact_matrix"]["exp6225-v539-terminal-transition"]["terminal_class"]
        == "missing"
    )


def test_helper_branches_are_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6238-3: malformed JSON, operators, and log gaps stay explicit."""

    yaml_path = tmp_path / "not_mapping.yaml"
    yaml_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    assert mod.load_yaml_object(tmp_path / "missing.yaml") == {}
    assert mod.load_yaml_object(yaml_path) == {}

    malformed = tmp_path / "results/bad.json"
    malformed.parent.mkdir(parents=True)
    malformed.write_text("{", encoding="utf-8")
    list_json = tmp_path / "results/list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert mod.load_json_object(malformed)[1]["error"].startswith("json_error:")
    assert mod.load_json_object(list_json)[1]["error"] == "json_not_mapping"
    assert mod.load_json_object(tmp_path / "missing.json")[1]["error"] == "missing"

    assert mod.evaluate_operator(1, "!=", 2) is True
    assert mod.evaluate_operator(2, ">", 1) is True
    assert mod.evaluate_operator(1, ">=", 1) is True
    assert mod.evaluate_operator(1, "<", 2) is True
    assert mod.evaluate_operator(2, "<=", 2) is True
    assert mod.evaluate_operator("a", "in", ["a"]) is True
    assert mod.evaluate_operator(None, "exists", True) is False
    assert mod.evaluate_operator("a", ">", 1) is False
    assert mod.evaluate_operator(1, "bad", 1) is False

    _write_artifact(tmp_path / "results/upstream.json", "complete", ready_score=1)
    bad_gate_receipts = mod.evaluate_gate_cascades(
        tmp_path,
        [
            {"id": "upstream", "deliverable": "results/upstream.json"},
            {"id": "downstream", "deliverable": "results/downstream.json", "gated_on": ["bad"]},
        ],
    )
    assert bad_gate_receipts["gates"][0]["reason"] == "gate_not_mapping"

    assert (
        mod.latest_conductor_receipts(tmp_path, [{"id": "x", "title": "No log"}])["x"][
            "receipt_found"
        ]
        is False
    )
    assert mod.count_terminal_classes({"x": {"terminal_class": "complete"}})["complete"] == 1

    assert mod._score({"wrapped": {"value": 3, "principle": "fixture"}}, "wrapped") == 3
    retirement = mod.prior_failure_retirement_actions(
        [
            {
                "id": "x",
                "prior_failures": [
                    "bad",
                    {"verdict": "complete: old", "retire_if_same_verdict": False},
                ],
            }
        ],
        {"x": {"terminal_class": "complete", "honest_verdict_raw": "complete: old"}},
    )
    assert retirement["actions"] == []
    assert retirement["rule_fired_count"] == 0

    invalid = {"status": "complete"}
    validation_errors = mod.validate_report(invalid)
    assert "missing required field: roadmap_path_hash_and_task_ids" in validation_errors
    assert "field_principles is not a mapping" in validation_errors
    assert "field_provenance is not a mapping" in validation_errors
    assert "counts field is not a mapping" in validation_errors
    assert "gate_cascade_receipts.gates is not a list" in validation_errors
    assert "wrong inference_substrate" in validation_errors
    assert "verifier_is_oracle must be false" in validation_errors
    assert "honest_verdict lacks terminal prefix" in validation_errors
    assert "reproducibility_checksum missing" in validation_errors

    invalid_counts = {field: None for field in mod.REQUIRED_ARTIFACT_FIELDS}
    invalid_counts.update(
        {
            "status": "complete",
            "field_principles": dict(mod.FIELD_PRINCIPLES),
            "field_provenance": {},
            "missing_blocked_skipped_partial_null_flagged_retired_and_ready_counts": {
                "count_principles": {}
            },
            "gate_cascade_receipts": {"gates": [{}]},
            "hardware_boundary_and_claim_count": {"hardware_claim_count": 0},
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
            "verifier_is_oracle": False,
            "honest_verdict": "complete: fixture",
            "reproducibility_checksum": "sha256:bad",
        }
    )
    count_errors = mod.validate_report(invalid_counts)
    assert "missing field_provenance entry: status" in count_errors
    assert "missing count principle: missing" in count_errors
    assert "gate missing principle" in count_errors

    with pytest.raises(ValueError, match="invalid Exp6238 report"):
        mod.write_report({"status": "complete"}, tmp_path)
