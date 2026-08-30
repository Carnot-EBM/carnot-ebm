"""Tests for the independent Exp6792 CSL causal and safety cold audit.

Spec refs: REQ-CL-6792, SCENARIO-CL-6792-PRECONDITIONS,
SCENARIO-CL-6792-COLD-RECOMPUTE, SCENARIO-CL-6792-CAUSAL,
SCENARIO-CL-6792-SAFETY, SCENARIO-CL-6792-DURABILITY,
SCENARIO-CL-6792-TERMINAL.
"""

from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path

import pytest

from carnot import experiment_6792_csl_causal_safety_cold_audit as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / exp.SPEC_RELATIVE_PATH
SOURCE_PATHS = {
    name: REPO_ROOT / relative_path for name, relative_path in exp.SOURCE_RELATIVE_PATHS.items()
}


@pytest.fixture(scope="module")
def sources() -> dict[str, dict]:
    """Load each checked-in source once because the audit never mutates it."""

    return {name: exp.read_json_object(path) for name, path in SOURCE_PATHS.items()}


@pytest.fixture(scope="module")
def blocked_artifact() -> dict:
    """Build the real fail-closed result once for all schema assertions."""

    return exp.build_artifact(
        source_paths=SOURCE_PATHS,
        run_date=exp.RUN_DATE,
        duration_s=0.25,
    )


def test_req_cl_6792_spec_owns_the_cold_audit_contract() -> None:
    """REQ-CL-6792 anchors the independent audit before implementation."""

    section = SPEC_PATH.read_text(encoding="utf-8").split("## REQ-CL-6792", 1)[1]
    for marker in (
        "SCENARIO-CL-6792-PRECONDITIONS",
        "SCENARIO-CL-6792-COLD-RECOMPUTE",
        "SCENARIO-CL-6792-CAUSAL",
        "SCENARIO-CL-6792-SAFETY",
        "SCENARIO-CL-6792-DURABILITY",
        "SCENARIO-CL-6792-TERMINAL",
        "complete_blocked_csl_causal_audit",
        exp.INFERENCE_SUBSTRATE,
        exp.MODULE_RELATIVE_PATH.as_posix(),
        exp.SCRIPT_RELATIVE_PATH.as_posix(),
        exp.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert marker in section
    for field in exp.REQUIRED_AUDIT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_cl_6792_preconditions_find_only_missing_raw_bytes(
    sources: dict[str, dict],
) -> None:
    """SCENARIO-CL-6792-PRECONDITIONS stops on absent transaction bytes."""

    summary = exp.evaluate_preconditions(sources, SOURCE_PATHS)

    assert summary["all_passed"] is False
    assert summary["failed_checks"] == ["transaction_byte_snapshots"]
    failure = summary["failures"][0]
    assert failure["expected"] == {
        "committed_receipts": 3189,
        "parent_byte_snapshots": 3189,
        "new_state_byte_snapshots": 3189,
        "byte_hash_matches": 3189,
    }
    assert failure["observed"] == {
        "committed_receipts": 3189,
        "parent_byte_snapshots": 0,
        "new_state_byte_snapshots": 0,
        "byte_hash_matches": 0,
    }
    checks = {row["check"]: row for row in summary["checks"]}
    assert checks["compositional_csl_completed"]["passed"] is True
    assert checks["complete_per_event_rows"]["observed"]["row_count"] == 4800
    assert checks["all_five_orders"]["observed"]["order_count"] == 5
    assert checks["source_hashes"]["passed"] is True
    assert checks["exact_receipt_hashes"]["observed"] == {
        "expected_rows": 4800,
        "matching_rows": 4800,
    }


def test_source_evidence_is_read_without_producer_or_headline_imports() -> None:
    """SCENARIO-CL-6792-COLD-RECOMPUTE keeps source producers outside the audit."""

    source = inspect.getsource(exp)
    assert "experiment_6791_compositional_online_constraint_routing_ab import" not in source
    assert "from carnot import experiment_6791" not in source
    for field in (
        "online_minus_frozen_lcb",
        "online_minus_frozen_order_effects",
        "component_action_attribution",
    ):
        assert f'source["{field}"]' not in source


def test_blocked_artifact_has_all_fields_and_no_replay_evidence(
    blocked_artifact: dict,
) -> None:
    """SCENARIO-CL-6792-TERMINAL emits one valid complete blocked artifact."""

    artifact = blocked_artifact
    assert set(artifact) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete_blocked_csl_causal_audit"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("complete_blocked_csl_causal_audit:")
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["rows"] == []
    assert artifact["cold_recomputed_metrics"] == {}
    assert artifact["headline_differences"] == {}
    assert artifact["credited_factor_count"] == 0
    assert artifact["factors_with_changed_action_witness"] == []
    assert artifact["retrieval_disable_effects"] == []
    assert artifact["poison_attack_results"] == []
    assert artifact["capacity_eviction_receipts"] == []
    assert artifact["retention_after_phase"] == {}
    assert artifact["hard_case_harm_after_phase"] == {}
    assert artifact["restart_byte_identity"] is None
    assert artifact["restart_action_identity"] is None
    assert artifact["rollback_byte_identity"] is None
    assert artifact["rollback_action_identity"] is None
    assert artifact["admitted_poison_count"] == 0
    assert artifact["influenced_poison_count"] == 0
    assert artifact["source_verdict_supported"] is False
    assert artifact["csl_causal_audit_completed"] is False
    assert artifact["verifier_is_oracle"] is False
    assert artifact["source_artifact_hashes"] == exp.EXPECTED_SOURCE_HASHES
    assert exp.validate_artifact(artifact) == []


def test_precondition_tampering_names_each_observed_failure(
    sources: dict[str, dict], tmp_path: Path
) -> None:
    """REQ-CL-6792 preserves exact observed values for every failed gate."""

    tampered = deepcopy(sources)
    tampered["experiment_6791"]["compositional_csl_completed"] = False
    tampered["experiment_6791"]["rows"] = tampered["experiment_6791"]["rows"][:-1]
    tampered["experiment_6791"]["frozen_manifest"]["order_hashes"].pop("order_5")
    tampered["experiment_6791"]["source_artifact_hash"] = "sha256:wrong"
    tampered["experiment_6791"]["rows"][0]["hidden_receipt_hash"] = "sha256:wrong"
    copied_paths = dict(SOURCE_PATHS)
    copied = tmp_path / "experiment_6791.json"
    copied.write_text(json.dumps(tampered["experiment_6791"]), encoding="utf-8")
    copied_paths["experiment_6791"] = copied

    summary = exp.evaluate_preconditions(tampered, copied_paths)

    assert summary["failed_checks"] == [
        "compositional_csl_completed",
        "complete_per_event_rows",
        "source_hashes",
        "transaction_byte_snapshots",
        "exact_receipt_hashes",
        "all_five_orders",
    ]
    assert all("observed" in row for row in summary["failures"])


def test_transaction_byte_checker_rejects_malformed_and_accepts_exact_bytes() -> None:
    """SCENARIO-CL-6792-PRECONDITIONS verifies bytes instead of receipt claims."""

    exact = b"{}\n"
    source = {
        "transaction_receipts": [
            {
                "committed": True,
                "parent_bytes_b64": "e30K",
                "new_state_bytes_b64": "e30K",
                "parent_hash": exp.sha256_bytes(exact),
                "new_state_hash": exp.sha256_bytes(exact),
            },
            {
                "committed": True,
                "parent_bytes_b64": "not-base64",
                "new_state_bytes_b64": "é",
                "parent_hash": "sha256:wrong",
                "new_state_hash": "sha256:wrong",
            },
            {"committed": False},
        ]
    }

    assert exp._transaction_byte_observation(source) == {
        "committed_receipts": 2,
        "parent_byte_snapshots": 2,
        "new_state_byte_snapshots": 2,
        "byte_hash_matches": 1,
    }


def test_complete_replay_guard_and_internal_validation_are_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CL-6792 never substitutes partial work after prerequisite checks."""

    real_evaluate = exp.evaluate_preconditions
    monkeypatch.setattr(
        exp,
        "evaluate_preconditions",
        lambda sources, paths: {"all_passed": True},
    )
    with pytest.raises(RuntimeError, match="complete replay is required"):
        exp.build_artifact(source_paths=SOURCE_PATHS, duration_s=0.1)
    monkeypatch.setattr(exp, "evaluate_preconditions", real_evaluate)

    real_validate = exp.validate_artifact
    monkeypatch.setattr(exp, "validate_artifact", lambda artifact: ["forced invalid"])
    with pytest.raises(ValueError, match="forced invalid"):
        exp.build_artifact(source_paths=SOURCE_PATHS, duration_s=0.1)
    monkeypatch.setattr(exp, "validate_artifact", real_validate)


def test_validation_and_atomic_writer_fail_closed(blocked_artifact: dict, tmp_path: Path) -> None:
    """REQ-CL-6792 validates exact fields before one atomic artifact write."""

    target = tmp_path / "artifact.json"
    receipt = exp.write_artifact(target, blocked_artifact)
    assert receipt["atomic_rename"] is True
    assert receipt["sha256"] == exp.sha256_file(target)
    assert json.loads(target.read_text(encoding="utf-8")) == blocked_artifact

    invalid = deepcopy(blocked_artifact)
    invalid["verdict_class"] = "null"
    invalid["reproducibility_checksum"] = exp.reproducibility_checksum(invalid)
    assert "blocked verdict_class mismatch" in exp.validate_artifact(invalid)
    with pytest.raises(ValueError, match="blocked verdict_class mismatch"):
        exp.write_artifact(tmp_path / "invalid.json", invalid)

    general = deepcopy(blocked_artifact)
    general["unexpected"] = True
    general["field_principles"] = {}
    general["inference_substrate"] = "wrong"
    general["random_seed"] = -1
    general["verifier_is_oracle"] = True
    general["verdict_class"] = "wrong"
    general["honest_verdict"] = "wrong"
    general["reproducibility_checksum"] = "wrong"
    general["csl_causal_audit_completed"] = True
    general["rows"] = [{}]
    general["gate_check_summary"] = {"all_passed": True, "failures": []}
    assert set(exp.validate_artifact(general)) == {
        "required field set mismatch",
        "field principle coverage mismatch",
        "inference substrate mismatch",
        "random seed mismatch",
        "verifier_is_oracle must be false",
        "verdict class is outside the closed enum",
        "honest verdict lacks a terminal prefix",
        "reproducibility checksum mismatch",
        "blocked verdict_class mismatch",
        "blocked audit cannot be complete",
        "blocked audit contains replay rows",
        "blocked audit lacks a failed gate",
    }

    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        exp.read_json_object(malformed)


def test_cli_writes_and_validates_task_owned_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-CL-6792 keeps the required command on the validated writer path."""

    output = tmp_path / "result.json"
    assert exp.main(["--date", exp.RUN_DATE, "--output", str(output)]) == 0
    stored = exp.read_json_object(output)
    assert stored["status"] == "complete_blocked_csl_causal_audit"
    assert exp.main(["--validate", "--output", str(output)]) == 0
    assert "complete_blocked_csl_causal_audit:" in capsys.readouterr().out
    invalid = json.loads(output.read_text(encoding="utf-8"))
    invalid["verdict_class"] = "null"
    invalid["reproducibility_checksum"] = exp.reproducibility_checksum(invalid)
    output.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="blocked verdict_class mismatch"):
        exp.main(["--validate", "--output", str(output)])
    with pytest.raises(ValueError, match="YYYYMMDD"):
        exp.build_artifact(source_paths=SOURCE_PATHS, run_date="2026-08-30")
