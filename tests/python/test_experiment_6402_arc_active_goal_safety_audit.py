"""Tests for Exp6402 active-goal safety audit.

Spec refs: REQ-ARC-ARM-6402,
SCENARIO-ARC-ARM-6402-REGISTRATION-FIRST,
SCENARIO-ARC-ARM-6402-READINESS-RECOMPUTE,
SCENARIO-ARC-ARM-6402-ATTACKS-FAIL-CLOSED,
SCENARIO-ARC-ARM-6402-MODEL-POLICY-SUBSTRATE,
SCENARIO-ARC-ARM-6402-ARTIFACT-NO-PROMOTION.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6402_arc_active_goal_safety_audit as exp6402


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/arc-agi/spec.md"


def _registered_chain() -> tuple[dict[str, Any], dict[str, Any]]:
    registration = exp6402.register_expected_scope(REPO)
    artifacts = exp6402.load_registered_artifacts(registration, REPO)
    return registration, artifacts


def _artifact(tmp_path: Path) -> dict[str, Any]:
    return exp6402.run(
        date="20260813",
        repo_root=REPO,
        result_path=tmp_path / exp6402.RESULT_RELATIVE_PATH.name,
        duration_s=1.25,
        tests_run=tuple(exp6402.DEFAULT_TEST_COMMANDS),
        test_exit_codes={command: 0 for command in exp6402.DEFAULT_TEST_COMMANDS},
        write=True,
    )


def _with_checksum(payload: dict[str, Any]) -> dict[str, Any]:
    payload["reproducibility_checksum"] = exp6402.payload_checksum(payload)
    return payload


def test_req_arc_arm_6402_spec_declares_safety_audit_contract() -> None:
    """REQ-ARC-ARM-6402: OpenSpec names the audit fields and no-promotion scope."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-ARC-ARM-6402") :]
    for marker in (
        "SCENARIO-ARC-ARM-6402-REGISTRATION-FIRST",
        "SCENARIO-ARC-ARM-6402-READINESS-RECOMPUTE",
        "SCENARIO-ARC-ARM-6402-ATTACKS-FAIL-CLOSED",
        "SCENARIO-ARC-ARM-6402-MODEL-POLICY-SUBSTRATE",
        "SCENARIO-ARC-ARM-6402-ARTIFACT-NO-PROMOTION",
        "register expected paths",
        "public_arc_claim_eligibility",
        exp6402.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert marker in section
    for field in exp6402.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_arc_arm_6402_registration_first_preserves_absent_paths() -> None:
    """SCENARIO-ARC-ARM-6402-REGISTRATION-FIRST: hashes are pinned first."""

    registration, artifacts = _registered_chain()
    matrix = exp6402.present_absent_blocked_skipped_null_flagged_and_retired_artifact_matrix(
        registration,
        artifacts,
    )
    hashes = exp6402.source_entrypoint_policy_reward_machine_model_window_and_registry_hash_matrix(
        registration,
        artifacts,
        repo_root=REPO,
    )

    assert registration["registered_before_reading_conclusions"] is True
    assert registration["expected_scope"]["model_ids"] == list(exp6402.MANDATED_MODEL_IDS)
    assert registration["paths"]["ops/arc_solve_claims.yaml"]["state"] == "absent"
    assert registration["paths"]["python/carnot/arc_agi/agent.py"]["state"] == "absent"
    assert matrix["artifact_states"]["exp6400"]["state"] == "clean"
    assert matrix["artifact_states"]["exp6401"]["state"] == "clean"
    assert matrix["path_state_counts"]["absent"] >= 1
    assert hashes["all_registered_hashes_match_current"] is True
    assert hashes["embedded_receipt_comparisons"]["exp6401_exp6400_manifest_file_hash"]["matches"] is True

    missing = exp6402._artifact_state(
        "missing",
        Path("results/missing.json"),
        None,
        {"state": "absent", "sha256": None},
    )
    assert missing["state"] == "absent"

    blocked = exp6402._artifact_state(
        "blocked",
        Path("results/blocked.json"),
        {"status": "blocked", "honest_verdict": "blocked_x"},
        {"state": "present", "sha256": "sha256:x"},
    )
    assert blocked["state"] == "blocked"


def test_scenario_arc_arm_6402_registration_helpers_cover_edge_states(tmp_path: Path) -> None:
    """SCENARIO-ARC-ARM-6402-REGISTRATION-FIRST: edge states stay explicit."""

    assert exp6402._read_json(tmp_path / "missing.json") is None
    not_object = tmp_path / "list.json"
    not_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="top-level JSON"):
        exp6402._read_json(not_object)

    inside = tmp_path / "inside.txt"
    inside.write_text("x", encoding="utf-8")
    assert exp6402._display_path(inside, tmp_path) == "inside.txt"
    assert exp6402._display_path(Path("/tmp/outside.txt"), tmp_path) == "/tmp/outside.txt"

    with pytest.raises(ValueError, match="registration"):
        exp6402.load_registered_artifacts({"registered_before_reading_conclusions": False}, REPO)

    states = {
        "flagged": {"status": "complete", "honest_verdict": "complete:", "flagged_adversarial": True},
        "skipped": {"status": "skipped", "honest_verdict": "skipped"},
        "retired": {"status": "complete", "honest_verdict": "complete_retired"},
        "null": {"status": "complete", "honest_verdict": "complete: null route value"},
        "unknown": {"status": "partial", "honest_verdict": "partial"},
    }
    for expected, payload in states.items():
        state = exp6402._artifact_state(
            expected,
            Path(f"results/{expected}.json"),
            payload,
            {"state": "present", "sha256": f"sha256:{expected}"},
        )
        assert state["state"] == expected

    assert exp6402._compare("not-a-number", "==", 1.0) is False
    assert exp6402._compare(1.0, ">=", 1.0) is True
    with pytest.raises(ValueError, match="unsupported"):
        exp6402._compare(1.0, "!=", 1.0)


def test_scenario_arc_arm_6402_readiness_recompute_keeps_public_claim_false(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-ARM-6402-READINESS-RECOMPUTE: route value is not a public claim."""

    artifact = _artifact(tmp_path)
    loaded = json.loads((tmp_path / exp6402.RESULT_RELATIVE_PATH.name).read_text(encoding="utf-8"))

    assert loaded == artifact
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["recomputed_scalar_gates_and_readiness"]["exp6393"]["ready"] is True
    assert artifact["recomputed_scalar_gates_and_readiness"]["exp6400"]["ready"] is True
    assert artifact["recomputed_scalar_gates_and_readiness"]["exp6401"]["ready"] is True
    assert artifact["recomputed_scalar_gates_and_readiness"]["scientific_readiness"] is True
    assert artifact["route_promotion_count"] == 0
    assert artifact["solve_claim_count"] == 0
    assert artifact["solve_registry_modified"] is False
    assert artifact["claims_ledger_modified"] is False
    assert artifact["public_arc_claim_eligibility"] is False
    assert artifact["verifier_is_oracle"] is False
    assert artifact["field_principles"]["public_arc_claim_eligibility"]
    assert artifact["field_principles"]["route_promotion_count"]
    assert artifact["reproducibility_checksum"] == exp6402.payload_checksum(artifact)
    exp6402.validate_artifact(artifact)


def test_scenario_arc_arm_6402_attack_groups_fail_closed() -> None:
    """SCENARIO-ARC-ARM-6402-ATTACKS-FAIL-CLOSED: clean controls pass."""

    registration, artifacts = _registered_chain()
    _ = registration
    hidden = exp6402.hidden_source_search_bfs_adapter_proxy_and_outer_loop_attack_results(artifacts)
    oracle = exp6402.oracle_timing_freeze_window_duplicate_model_state_legal_work_and_firing_attack_results(
        artifacts
    )
    goal = exp6402.goal_false_accept_abstention_missing_progress_enablement_solve_and_registry_attack_results(
        artifacts
    )
    default = exp6402.default_off_reachability_and_executed_action_integrity_checks(artifacts)

    assert all(row["passed"] for row in hidden["checks"])
    assert all(row["passed"] for row in oracle["checks"])
    assert all(row["passed"] for row in goal["checks"])
    assert all(row["passed"] for row in default["checks"])
    assert goal["upstream_route_promotion_eligible_count"] == 1
    assert default["executed_action_change_count"] == 0

    bad = copy.deepcopy(artifacts)
    bad["exp6401"]["oracle_before_action_count"] = 1
    bad["exp6401"]["pre_action_goal_probe_and_action_freeze_records"][0][
        "environment_result_visible_before_freeze"
    ] = True
    attacked = exp6402.oracle_timing_freeze_window_duplicate_model_state_legal_work_and_firing_attack_results(
        bad
    )
    assert any(not row["passed"] for row in attacked["checks"])

    bad = copy.deepcopy(artifacts)
    bad["exp6400"]["offline_ground_truth_search_count"] = 1
    hidden_bad = exp6402.hidden_source_search_bfs_adapter_proxy_and_outer_loop_attack_results(bad)
    assert hidden_bad["forbidden_access_clean"] is False

    assert exp6402._freeze_receipts_match([{}]) is True
    forged = copy.deepcopy(artifacts["exp6401"]["pre_action_goal_probe_and_action_freeze_records"])
    forged[0]["freeze_receipt_sha256"] = "sha256:forged"
    assert exp6402._freeze_receipts_match(forged) is False

    findings = exp6402._finding_groups(
        (
            {
                "checks": [
                    {
                        "name": "failed_without_severity",
                        "passed": False,
                        "detail": "default major",
                    }
                ]
            },
        )
    )
    assert findings["major"][0]["name"] == "failed_without_severity"


def test_scenario_arc_arm_6402_model_policy_and_substrate_checks() -> None:
    """SCENARIO-ARC-ARM-6402-MODEL-POLICY-SUBSTRATE: receipts are audited."""

    registration, artifacts = _registered_chain()
    checks = exp6402.model_policy_and_inference_substrate_checks(artifacts)
    hashes = exp6402.source_entrypoint_policy_reward_machine_model_window_and_registry_hash_matrix(
        registration,
        artifacts,
        repo_root=REPO,
    )

    assert checks["models_used"] == list(exp6402.MANDATED_MODEL_IDS)
    assert checks["all_mandated_model_ids_present"] is True
    assert checks["cached_sota_receipts_present"] is True
    assert checks["embedded_tokenizers_all_ok"] is True
    assert checks["autotokenizer_usage_count"] == 0
    assert checks["task_linked_gpu_evidence_terminal"] is True
    assert checks["legacy_headline_cell_present"] is False
    assert hashes["model_hashes_consistent_across_present_artifacts"] is True

    bad = copy.deepcopy(artifacts)
    bad["exp6401"]["autotokenizer_usage_count"] = 1
    bad_checks = exp6402.model_policy_and_inference_substrate_checks(bad)
    assert bad_checks["autotokenizer_usage_count"] == 1
    assert bad_checks["model_policy_substrate_clean"] is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("route_promotion_count", 1, "route_promotion_count"),
        ("solve_claim_count", 1, "solve_claim_count"),
        ("solve_registry_modified", True, "solve_registry_modified"),
        ("claims_ledger_modified", True, "claims_ledger_modified"),
        ("public_arc_claim_eligibility", True, "public_arc_claim_eligibility"),
        ("upstream_artifacts_modified", True, "upstream_artifacts_modified"),
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("verifier_is_oracle", True, "verifier_is_oracle"),
        ("status", "blocked", "status"),
    ],
)
def test_scenario_arc_arm_6402_validation_rejects_forbidden_drift(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    """SCENARIO-ARC-ARM-6402-ARTIFACT-NO-PROMOTION: forbidden drift is rejected."""

    artifact = _artifact(tmp_path)
    bad = copy.deepcopy(artifact)
    bad[field] = value
    _with_checksum(bad)

    with pytest.raises(ValueError, match=message):
        exp6402.validate_artifact(bad)


def test_req_arc_arm_6402_validation_rejects_missing_nested_and_checksum(
    tmp_path: Path,
) -> None:
    """REQ-ARC-ARM-6402: missing fields and nested failures fail validation."""

    artifact = _artifact(tmp_path)

    checksum = dict(artifact)
    checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp6402.validate_artifact(checksum)

    missing = dict(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing fields"):
        exp6402.validate_artifact(missing)

    drift_cases = [
        (
            lambda a: a["source_entrypoint_policy_reward_machine_model_window_and_registry_hash_matrix"].__setitem__(
                "all_registered_hashes_match_current",
                False,
            ),
            "hash_matrix",
        ),
        (
            lambda a: a["source_entrypoint_policy_reward_machine_model_window_and_registry_hash_matrix"].__setitem__(
                "all_embedded_receipts_match_registration",
                False,
            ),
            "hash_matrix",
        ),
        (
            lambda a: a["default_off_reachability_and_executed_action_integrity_checks"].__setitem__(
                "default_off_and_integrity_clean",
                False,
            ),
            "default_off",
        ),
        (
            lambda a: a["model_policy_and_inference_substrate_checks"].__setitem__(
                "model_policy_substrate_clean",
                False,
            ),
            "model_policy",
        ),
        (
            lambda a: a["protected_files_unchanged"]["ops/arc_solve_registry.yaml"].__setitem__(
                "unchanged",
                False,
            ),
            "protected_files_unchanged",
        ),
        (
            lambda a: a.__setitem__("field_principles", {}),
            "field_principles",
        ),
        (
            lambda a: a.__setitem__("honest_verdict", "blocked"),
            "honest_verdict",
        ),
    ]
    for mutate, message in drift_cases:
        bad = copy.deepcopy(artifact)
        mutate(bad)
        _with_checksum(bad)
        with pytest.raises(ValueError, match=message):
            exp6402.validate_artifact(bad)


def test_req_arc_arm_6402_build_artifact_uses_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-ARM-6402: build_artifact validates the runner output."""

    artifact = _artifact(tmp_path)

    def fake_run(**kwargs):
        assert kwargs["date"] == "20260813"
        assert kwargs["write"] is True
        return artifact

    monkeypatch.setattr(exp6402, "run", fake_run)

    built = exp6402.build_artifact(
        tmp_path,
        date="20260813",
        output_path=tmp_path / "out.json",
    )
    assert built["public_arc_claim_eligibility"] is False
