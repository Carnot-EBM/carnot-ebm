"""Tests for Exp6474 protocol identifiability and receipt preflight.

Spec refs: REQ-VERIFY-6474, SCENARIO-VERIFY-6474-FINITE-AUDIT,
SCENARIO-VERIFY-6474-MINIMUM-SUPPORT, SCENARIO-VERIFY-6474-ATTACKS,
SCENARIO-VERIFY-6474-RECEIPTS, SCENARIO-VERIFY-6474-ROWS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6474_protocol_identifiability_and_receipt_preflight as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _passing_tests() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def test_req_verify_6474_spec_declares_fields_and_scenarios() -> None:
    """REQ-VERIFY-6474: OpenSpec owns the Exp6474 audit contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-VERIFY-6474") :]
    for marker in (
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "SCENARIO-VERIFY-6474-FINITE-AUDIT",
        "SCENARIO-VERIFY-6474-MINIMUM-SUPPORT",
        "SCENARIO-VERIFY-6474-ATTACKS",
        "SCENARIO-VERIFY-6474-RECEIPTS",
        "SCENARIO-VERIFY-6474-ROWS",
        "deterministic_synthetic_protocol_audit_no_llm",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in mod.FIELD_PRINCIPLES


def test_scenario_verify_6474_finite_audit_emits_constructive_witnesses() -> None:
    """SCENARIO-VERIFY-6474-FINITE-AUDIT: equal observations can collide."""

    policies = mod.declared_policy_class()
    support = mod.declared_observation_support()
    declared = mod.audit_support(
        policy_class=policies,
        support=support,
        estimand=mod.DECLARED_ESTIMAND,
        condition_id="declared_minimum_support",
    )
    assert declared["identifying"] is True
    assert declared["collision_count"] == 0
    assert declared["duplicate_observation_count"] == 0
    assert len(declared["pair_rows"]) == 6

    empty = mod.audit_support(
        policy_class=policies,
        support=[],
        estimand=mod.DECLARED_ESTIMAND,
        condition_id="empty_support",
    )
    assert empty["identifying"] is False
    assert empty["collision_count"] > 0
    witness = empty["collision_witnesses"][0]
    assert witness["observed_signature_left"] == witness["observed_signature_right"]
    assert witness["target_effect_left"] != witness["target_effect_right"]
    assert witness["target_effect_delta"] != 0
    assert witness["observed_outcomes_left"] == witness["observed_outcomes_right"]

    duplicate = mod.audit_support(
        policy_class=policies,
        support=[support[0], support[1], support[0]],
        estimand=mod.DECLARED_ESTIMAND,
        condition_id="duplicated_declared_support",
    )
    assert duplicate["identifying"] is True
    assert duplicate["collision_count"] == 0
    assert duplicate["canonical_support"] == support
    assert duplicate["duplicate_observation_count"] == 1

    constant = mod.audit_support(
        policy_class=policies,
        support=[],
        estimand=mod.CONSTANT_ESTIMAND,
        condition_id="constant_estimand_control",
    )
    assert constant["identifying"] is True
    assert constant["collision_count"] == 0


def test_scenario_verify_6474_minimum_support_and_attacks() -> None:
    """SCENARIO-VERIFY-6474-MINIMUM-SUPPORT: search proves minimality."""

    policies = mod.declared_policy_class()
    support = mod.declared_observation_support()
    minimum = mod.synthesize_minimum_identifying_support(
        policy_class=policies,
        candidate_cells=mod.CANDIDATE_OBSERVATION_CELLS,
        estimand=mod.DECLARED_ESTIMAND,
    )
    assert minimum["support"] == support
    assert minimum["size"] == len(support)
    assert minimum["verified_by_exhaustive_enumeration"] is True
    assert all(row["identifying"] is False for row in minimum["smaller_support_rows"])

    leave_one = mod.leave_one_support_out_rows(
        policy_class=policies,
        support=support,
        estimand=mod.DECLARED_ESTIMAND,
    )
    assert {row["removed_cell"] for row in leave_one} == set(support)
    assert all(row["identifying"] is False for row in leave_one)
    assert all(row["witness_count"] > 0 for row in leave_one)

    attacks = mod.build_attack_matrix(policies, support)
    by_id = {row["attack_id"]: row for row in attacks["rows"]}
    assert set(by_id) == set(mod.ATTACK_IDS)
    assert attacks["all_required_controls_passed"] is True
    assert by_id["empty_support"]["witness_required"] is True
    assert by_id["empty_support"]["witness_count"] > 0
    assert by_id["leave_one_support_out"]["witness_count"] == len(support)
    assert by_id["duplicated_observation"]["identifying"] is True
    assert by_id["constant_estimand"]["identifying"] is True
    assert by_id["changed_policy_class"]["identifying"] is False
    assert by_id["changed_policy_class"]["witness_count"] > 0

    impossible = mod.synthesize_minimum_identifying_support(
        policy_class=mod.changed_policy_class(policies),
        candidate_cells=["held_control_outcome"],
        estimand=mod.DECLARED_ESTIMAND,
    )
    assert impossible["verified_by_exhaustive_enumeration"] is False
    assert impossible["support"] == []


def test_scenario_verify_6474_receipt_rows_validate_without_llm() -> None:
    """SCENARIO-VERIFY-6474-RECEIPTS: fixture phase rows fail closed."""

    rows = mod.build_task_scoped_receipt_rows(start_ns=1_000_000)
    report = mod.validate_task_scoped_receipt_rows(rows)
    assert report["accepted"] is True
    assert report["reasons"] == []
    assert report["required_phase_count"] == len(mod.REQUIRED_RECEIPT_PHASES)
    assert {row["phase"] for row in rows} == set(mod.REQUIRED_RECEIPT_PHASES)
    assert all(row["llm_invocation"] is False for row in rows)
    assert all(row["cpu_fallback"] is False for row in rows)

    missing_phase = rows[:-1]
    assert (
        "missing_phase:artifact_write"
        in mod.validate_task_scoped_receipt_rows(missing_phase)["reasons"]
    )

    bad = deepcopy(rows)
    bad[1]["monotonic_end_ns"] = bad[1]["monotonic_start_ns"] - 1
    assert "negative_interval" in mod.validate_task_scoped_receipt_rows(bad)["reasons"]

    bad = deepcopy(rows)
    bad[2]["runner_selection"]["binary_sha256"] = "sha256:" + "0" * 64
    assert "runner_selection_hash_mismatch" in mod.validate_task_scoped_receipt_rows(bad)["reasons"]

    bad = deepcopy(rows)
    bad[2]["cpu_fallback"] = True
    assert "unexpected_cpu_fallback" in mod.validate_task_scoped_receipt_rows(bad)["reasons"]

    bad = deepcopy(rows)
    bad[2]["exit_status"] = {}
    assert "exit_status_missing_returncode" in mod.validate_task_scoped_receipt_rows(bad)["reasons"]

    truncated = deepcopy(rows)
    del truncated[0]["exit_status"]
    assert "truncated_receipt" in mod.validate_task_scoped_receipt_rows(truncated)["reasons"]

    malformed = deepcopy(rows)
    malformed[0]["wall_clock_start"] = ""
    malformed[0]["parent_pid"] = 1
    malformed[0]["no_child_fixture_receipt"] = False
    malformed[0]["command_hash"] = "bad"
    malformed[0]["config_hash"] = "bad"
    malformed[0]["model_hash"] = "bad"
    malformed[0]["raw_output_hash"] = "bad"
    malformed[0]["model_identity"]["model_identity_bound"] = False
    malformed[0]["runner_selection"]["selected"] = False
    malformed[0]["llm_invocation"] = True
    malformed[0]["attribution_confidence"] = 0.5
    malformed[1]["monotonic_start_ns"] = malformed[0]["monotonic_start_ns"]
    malformed[1]["monotonic_end_ns"] = malformed[0]["monotonic_end_ns"]
    report = mod.validate_task_scoped_receipt_rows(malformed)
    assert {
        "wall_clock_interval_missing",
        "parent_pid_invalid",
        "no_child_fixture_receipt_missing",
        "command_hash_missing",
        "config_hash_missing",
        "model_hash_missing",
        "raw_output_hash_missing",
        "model_identity_unbound",
        "runner_not_selected",
        "llm_invocation_not_allowed",
        "low_attribution_confidence",
        "overlap_unexplained",
    } <= set(report["reasons"])


def test_scenario_verify_6474_artifact_rows_recompute_and_validate(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6474-ROWS: terminal fields recompute from rows."""

    artifact = mod.build_artifact(
        root=REPO,
        run_date="20260821",
        duration_s=0.25,
        tests_run=_passing_tests(),
    )
    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete"
    assert artifact["protocol_identifying_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["gate_check_summary"]["all_gates_passed"] is True
    assert artifact["collision_witnesses"] == []
    assert artifact["minimum_identifying_support"]["support"] == mod.declared_observation_support()
    assert artifact["task_scoped_receipt_rows"]["accepted"] is True
    assert artifact["aggregate_row_recomputation"] == mod.recompute_aggregates_from_rows(
        artifact["per_unit_rows"]
    )
    assert set(artifact["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) == []

    bad = deepcopy(artifact)
    bad["protocol_identifying_score"] = 0.0
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "protocol_identifying_score mismatch" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["inference_substrate"] = "live_llm_inference"
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "inference_substrate mismatch" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["verifier_is_oracle"] = False
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "verifier_is_oracle must be true" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["per_unit_rows"] = bad["per_unit_rows"][:-1]
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "aggregate_row_recomputation mismatch" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    del bad["status"]
    assert "missing required field: status" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["field_provenance"] = {}
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "field_provenance must cover exactly required fields" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["field_principles"] = {}
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "missing field_principles entry: status" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["honest_verdict"] = "done"
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "honest_verdict lacks required terminal prefix" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(bad)

    path = tmp_path / "artifact.json"
    mod.write_artifact(artifact, path)
    assert json.loads(path.read_text(encoding="utf-8")) == artifact


def test_req_verify_6474_blocked_edges_and_dependency_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-6474: blocked gates and dependency receipts are explicit."""

    assert (
        mod._status(0.0, {"all_gates_passed": False})
        == "blocked_protocol_identifiability_preflight"
    )
    assert mod._honest_verdict("blocked_protocol_identifiability_preflight").startswith(
        "complete_blocked:"
    )

    original_version = mod.metadata.version

    def fake_version(name: str) -> str:
        if name == "missing-package":
            raise mod.metadata.PackageNotFoundError(name)
        return original_version(name)

    with monkeypatch.context() as mp:
        mp.setattr(mod.metadata, "version", fake_version)
        assert mod._package_version("missing-package") == "not_installed"

    with monkeypatch.context() as mp:
        mp.setattr(
            mod,
            "_gate_check_summary",
            lambda **_: {
                "checks": {"forced": False},
                "all_gates_passed": False,
                "failed_gates": ["forced"],
            },
        )
        blocked = mod.build_artifact(
            root=REPO,
            run_date="20260821",
            duration_s=0.25,
            tests_run=_passing_tests(),
        )
    assert blocked["status"] == "blocked_protocol_identifiability_preflight"
    assert blocked["protocol_identifying_score"] == 0.0


def test_req_verify_6474_run_write_and_validate_mode(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-VERIFY-6474: CLI writes and validates the terminal artifact."""

    result = tmp_path / "experiment_6474.json"
    artifact = mod.run(
        date="20260821",
        result_path=result,
        test_exit_codes=_passing_tests(),
    )
    assert json.loads(result.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"

    assert mod.main(["--date", "20260821", "--result-path", str(result)]) == 0
    written = json.loads(result.read_text(encoding="utf-8"))
    assert written["status"] == "complete"

    assert mod.main(["--validate", "--result-path", str(result)]) == 0
    validate_out = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert validate_out["ok"] is True

    missing = tmp_path / "missing.json"
    assert mod.main(["--validate", "--result-path", str(missing)]) == 1
    missing_out = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert missing_out["ok"] is False
    assert missing_out["errors"] == ["artifact missing"]
