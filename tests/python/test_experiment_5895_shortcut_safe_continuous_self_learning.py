"""Tests for Exp5895 shortcut-safe continuous self-learning.

Spec refs: REQ-LEARN-5895, SCENARIO-LEARN-5895-PRECONDITIONS,
SCENARIO-LEARN-5895-SEALED-SPLITS, SCENARIO-LEARN-5895-LIFECYCLE,
SCENARIO-LEARN-5895-METRICS, SCENARIO-LEARN-5895-HARDWARE-MAPPING,
SCENARIO-LEARN-5895-FAIL-CLOSED.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5895_shortcut_safe_continuous_self_learning as mod


REPO = Path(__file__).resolve().parents[2]
SELF_LEARNING_SPEC = REPO / "openspec/capabilities/self-learning/spec.md"
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5895_shortcut_safe_continuous_self_learning.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5895_shortcut_safe_continuous_self_learning.py "
    "-m pytest tests/python/test_experiment_5895_shortcut_safe_continuous_self_learning.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5895_shortcut_safe_continuous_self_learning.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5895_shortcut_safe_continuous_self_learning.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py --json "
    "results/experiment_5895_shortcut_safe_continuous_self_learning.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_FILE_COMMAND = (
    '.venv/bin/python -c "from pathlib import Path; '
    "assert Path('scripts/research_conductor.py').exists()\""
)
TEST_COMMANDS = [
    FOCUSED_TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
    PROTECTED_FILE_COMMAND,
]
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


def _preconditions(tmp_path: Path) -> dict[str, Any]:
    return mod.collect_preconditions(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        memory_probe=lambda: {
            "available_mb": 16384,
            "required_mb": mod.RAM_FLOOR_MB,
            "ok": True,
        },
        disk_probe=lambda root: {
            "available_mb": 16384,
            "required_mb": mod.DISK_FLOOR_MB,
            "ok": True,
        },
    )


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """REQ-LEARN-5895: build the deterministic 5895 artifact once."""

    base = tmp_path_factory.mktemp("exp5895")
    return mod.run(
        result_path=base / mod.RESULT_RELATIVE_PATH.name,
        preconditions_checked=_preconditions(base),
        duration_s=4.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )


def test_req_learn_5895_spec_declares_required_contract() -> None:
    """REQ-LEARN-5895: OpenSpec declares fields, scenarios, and principles."""

    text = SELF_LEARNING_SPEC.read_text(encoding="utf-8")
    section = text[text.index("## REQ-LEARN-5895") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5895",
        "SCENARIO-LEARN-5895-PRECONDITIONS",
        "SCENARIO-LEARN-5895-SEALED-SPLITS",
        "SCENARIO-LEARN-5895-LIFECYCLE",
        "SCENARIO-LEARN-5895-METRICS",
        "SCENARIO-LEARN-5895-HARDWARE-MAPPING",
        "SCENARIO-LEARN-5895-FAIL-CLOSED",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`shortcut_resistant_csl_ready_score`",
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_learn_5895_preconditions_and_artifact_are_hash_bound(
    artifact: dict[str, Any],
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-5895-PRECONDITIONS: gate, rows, hashes, and chain exclusion."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH.name
    replay = mod.run(
        result_path=result_path,
        preconditions_checked=_preconditions(tmp_path),
        duration_s=4.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )
    loaded = json.loads(result_path.read_text(encoding="utf-8"))

    assert replay == loaded
    assert mod.validate_artifact(replay) is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(replay)
    assert replay["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    assert replay["reproducibility_checksum"] == mod.reproducibility_checksum(replay)
    assert replay["status"] == "complete_positive"
    assert replay["honest_verdict"].startswith("complete_positive:")
    assert replay["continuous_self_learning_task"] is True
    assert replay["shortcut_resistant_csl_ready_score"] == pytest.approx(1.0)
    assert isinstance(replay["shortcut_resistant_csl_ready_score"], float)
    assert replay["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert replay["verifier_is_oracle"] is True
    assert replay["no_model_weight_mutation"]["all_unchanged"] is True
    assert replay["test_commands"] == TEST_COMMANDS
    assert replay["test_exit_codes"] == TEST_EXIT_CODES

    preconditions = replay["preconditions_checked"]
    upstream = replay["upstream_gate_and_hash_receipts"]
    assert preconditions["preconditions_ready"] is True
    assert upstream["exp5894_gate"]["ok"] is True
    assert upstream["exp5894_replay"]["validates"] is True
    assert upstream["exp5893_rows"]["row_count"] == 72
    assert upstream["retired_chain_exclusion"]["dependency_used"] is False
    assert {"experiment_5865", "experiment_5867"} <= set(
        upstream["retired_chain_exclusion"]["retired_context_artifacts"]
    )
    assert upstream["protected_files_unchanged"]["all_unchanged"] is True
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert replay["field_provenance"][field]["principle"] == principle


def test_scenario_learn_5895_sealed_splits_budget_parity_and_lifecycle(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-LEARN-5895-SEALED-SPLITS/LIFECYCLE: commit-before-reveal holds."""

    sealed = artifact["sealed_chronological_split_and_visibility"]
    parity = artifact["frozen_arms_and_budget_parity"]
    policy = artifact["exact_query_policy_and_budget"]
    state = artifact["verified_evidence_and_unresolved_constraint_state"]
    lifecycle = artifact["versioned_promotion_quarantine_rejection_and_rollback"]
    rejected = artifact["rejected_update_non_propagation"]
    certs = artifact["per_update_non_forgetting_certificates"]

    assert sealed["event_count"] == 72
    assert sealed["batches_sealed_before_decision"] is True
    assert sealed["commit_before_reveal"] is True
    assert sealed["future_evidence_visible_before_current_update_count"] == 0
    assert sealed["direct_label_visible_before_prediction_count"] == 0
    assert sealed["batch_counts"]["train"] == 36
    assert sealed["batch_counts"]["future_test"] == 9
    assert sealed["sample_commit_receipts"][0]["label_visible_before_commit"] is False

    assert parity["arms"] == list(mod.ARM_NAMES)
    assert parity["event_budget_parity"] is True
    assert parity["state_budget_parity"] is True
    assert parity["replay_budget_parity"] is True
    assert parity["applicable_query_budget_parity"] is True
    assert (
        policy["reduced_oracle"]["exact_queries_used"] < policy["full_oracle"]["exact_queries_used"]
    )
    assert policy["verifier_authority"] == "exact_semantic_and_constraint_validators"

    assert state["state_type"] == "verified_evidence_plus_unresolved_constraints"
    assert state["verified_evidence_count"] > 0
    assert state["unresolved_constraint_count"] > 0
    assert state["unresolved_constraints_accepted_as_evidence"] is False
    assert state["max_records"] == mod.MEMORY_CAP
    assert state["canonical_state_hash"].startswith("sha256:")

    assert lifecycle["versioned_proposals_enabled"] is True
    assert lifecycle["promoted_update_count"] == state["verified_evidence_count"]
    assert lifecycle["quarantined_update_count"] >= lifecycle["rejected_update_count"]
    assert lifecycle["rollback_mismatch_count"] == 0
    assert lifecycle["prospective_promotion_authority"] == "exact_validator"
    assert lifecycle["sample_promotion_receipts"][0]["validation_authority"] == "exact_validator"

    assert rejected["rejected_update_count"] == lifecycle["rejected_update_count"]
    assert rejected["promoted_rejected_update_count"] == 0
    assert rejected["future_context_rejected_update_count"] == 0
    assert rejected["replay_context_rejected_update_count"] == 0
    assert rejected["rejected_update_ids_disjoint_from_promoted"] is True

    assert certs["certificate_rate"] == pytest.approx(1.0)
    assert certs["failed_certificate_count"] == 0
    assert certs["certificate_count"] == lifecycle["promoted_update_count"]


def test_scenario_learn_5895_metrics_safety_controls_and_hardware_contract(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-LEARN-5895-METRICS/HARDWARE-MAPPING: positive, safe, bounded."""

    metrics = artifact["prospective_semantic_and_constraint_metrics"]
    shortcuts = artifact["shortcut_false_accept_metrics"]
    transfer = artifact["forward_transfer_recurrence_retention_and_regret"]
    bounds = artifact["family_grounding_hardness_lower_bounds"]
    accounting = artifact["replay_query_resource_and_latency_accounting"]
    memory = artifact["memory_cap_accounting"]
    restart = artifact["rollback_restart_and_state_hashes"]
    weights = artifact["no_model_weight_mutation"]
    controls = artifact["null_and_ablation_controls"]
    hardware = artifact["hardware_mapping_contract"]

    primary = metrics["arm_metrics"][mod.PRIMARY_ARM]
    soft = metrics["arm_metrics"]["soft_grounding"]
    shuffled = metrics["arm_metrics"]["shuffled_grounding"]
    assert primary["future_semantic_accuracy"] > soft["future_semantic_accuracy"]
    assert primary["future_semantic_accuracy"] > shuffled["future_semantic_accuracy"]
    assert metrics["primary_minus_best_shortcut_control"]["ci95"][0] > 0.0
    assert metrics["constraint_accuracy_reported_separately"] is True
    assert metrics["exact_validator_retains_authority"] is True

    assert shortcuts["primary_zero_false_accepts"] is True
    assert shortcuts["unsafe_accept_count"] == 0
    assert shortcuts["by_arm"][mod.PRIMARY_ARM]["total"] == 0
    assert shortcuts["by_arm"]["soft_grounding"]["total"] > 0

    assert transfer["forward_transfer"]["primary_minus_best_shortcut_control"]["ci95"][0] > 0.0
    assert transfer["recurrence"]["semantic_accuracy"] == pytest.approx(1.0)
    assert transfer["retention"]["protected_prefix_retention"] == pytest.approx(1.0)
    assert transfer["retention"]["retention_regression_count"] == 0
    assert (
        transfer["dynamic_regret"][mod.PRIMARY_ARM] < transfer["dynamic_regret"]["soft_grounding"]
    )

    assert bounds["all_group_lower_bounds_positive"] is True
    assert bounds["minimum_credited_lcb"] > 0.0
    for axis in ("family", "grounding", "hardness"):
        assert bounds["group_bootstrap_intervals"][axis]["ci95"][0] > 0.0

    assert accounting["event_budget_parity"] is True
    assert (
        accounting["query_efficiency"]["primary_lift_per_exact_query"]
        > accounting["query_efficiency"]["full_oracle_lift_per_exact_query"]
    )
    assert accounting["latency_accounting"]["claim"] == "descriptive_only_no_speedup_claim"
    assert memory["cap_compliance"] is True
    assert memory["max_state_records"] <= mod.MEMORY_CAP
    assert restart["restart_equivalence"] == pytest.approx(1.0)
    assert restart["rollback_hash_mismatch_count"] == 0
    assert restart["full_state_hash"] == restart["resumed_state_hash"]
    assert weights["gguf_weight_mutation_count"] == 0
    assert weights["model_execution_loaded"] is False
    assert controls["all_controls_passed"] is True
    assert controls["soft_grounding_control_detects_shortcuts"] is True
    assert controls["shuffled_grounding_control_detects_shortcuts"] is True
    assert controls["no_memory_not_credited_for_promotion"] is True

    expected_ops = {
        "insert",
        "quarantine",
        "lookup",
        "supersede",
        "rollback",
        "sparse_ranking",
        "fixed_width_ids",
        "bounded_records",
        "deterministic_hashes",
        "update_counts",
        "precision_ranges",
    }
    assert expected_ops <= set(hardware["operations"])
    assert hardware["backend_neutral"] is True
    assert hardware["board_execution_performed"] is False
    assert hardware["speedup_claimed"] is False
    assert hardware["falsifiable"] is True


def test_scenario_learn_5895_fail_closed_for_bad_inputs_and_tampering(
    artifact: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-LEARN-5895-FAIL-CLOSED: unsafe, null, or blocked evidence cannot promote."""

    blocked = mod.run(
        root=tmp_path,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        duration_s=4.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=False,
    )
    assert blocked["status"] == "blocked"
    assert blocked["shortcut_resistant_csl_ready_score"] == 0.0
    assert blocked["honest_verdict"].startswith("blocked:")
    assert "missing_exp5894_artifact" in blocked["preconditions_checked"]["blocked_reasons"]
    assert mod.validate_artifact(blocked) is True

    failed_tests = mod.run(
        result_path=tmp_path / "failed.json",
        preconditions_checked=_preconditions(tmp_path / "failed"),
        duration_s=4.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes={**TEST_EXIT_CODES, FULL_TEST_COMMAND: 1},
        write=False,
    )
    assert failed_tests["status"] == "complete_null"
    assert "failed_test_exit_codes" in mod.blocked_reasons(failed_tests)
    assert mod.validate_artifact(failed_tests) is True

    unsafe = deepcopy(artifact)
    unsafe["shortcut_false_accept_metrics"]["unsafe_accept_count"] = 1
    unsafe["shortcut_resistant_csl_ready_score"] = mod.shortcut_resistant_csl_ready_score(unsafe)
    unsafe["status"] = mod.status(unsafe)
    unsafe["honest_verdict"] = mod.honest_verdict(unsafe)
    unsafe["reproducibility_checksum"] = mod.reproducibility_checksum(unsafe)
    assert unsafe["status"] == "unsafe"
    assert unsafe["honest_verdict"].startswith("unsafe:")
    assert mod.validate_artifact(unsafe) is True

    propagated = deepcopy(artifact)
    propagated["rejected_update_non_propagation"]["promoted_rejected_update_count"] = 1
    propagated["shortcut_resistant_csl_ready_score"] = mod.shortcut_resistant_csl_ready_score(
        propagated
    )
    propagated["status"] = mod.status(propagated)
    propagated["honest_verdict"] = mod.honest_verdict(propagated)
    propagated["reproducibility_checksum"] = mod.reproducibility_checksum(propagated)
    assert propagated["status"] == "unsafe"
    assert "rejected_update_propagation" in mod.blocked_reasons(propagated)
    assert mod.validate_artifact(propagated) is True

    null_artifact = deepcopy(artifact)
    null_artifact["family_grounding_hardness_lower_bounds"]["all_group_lower_bounds_positive"] = (
        False
    )
    null_artifact["shortcut_resistant_csl_ready_score"] = mod.shortcut_resistant_csl_ready_score(
        null_artifact
    )
    null_artifact["status"] = mod.status(null_artifact)
    null_artifact["honest_verdict"] = mod.honest_verdict(null_artifact)
    null_artifact["reproducibility_checksum"] = mod.reproducibility_checksum(null_artifact)
    assert null_artifact["status"] == "complete_null"
    assert null_artifact["honest_verdict"].startswith("complete_null:")
    assert mod.validate_artifact(null_artifact) is True

    missing = deepcopy(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        mod.validate_artifact(missing)

    for mutate, match in (
        (
            lambda item: item.update({"inference_substrate": "wrong"}),
            "inference_substrate",
        ),
        (
            lambda item: item.update({"verifier_is_oracle": False}),
            "verifier_is_oracle",
        ),
        (
            lambda item: item["frozen_arms_and_budget_parity"].update(
                {"event_budget_parity": False}
            ),
            "ready_score",
        ),
        (
            lambda item: item["hardware_mapping_contract"].update({"speedup_claimed": True}),
            "ready_score",
        ),
        (
            lambda item: item.update({"shortcut_resistant_csl_ready_score": 0.0}),
            "ready_score",
        ),
        (
            lambda item: item.update({"status": "blocked"}),
            "status",
        ),
        (
            lambda item: item.update({"honest_verdict": "complete_positive: wrong"}),
            "honest_verdict",
        ),
        (
            lambda item: item.update({"reproducibility_checksum": mod.sha256_text("wrong")}),
            "reproducibility_checksum",
        ),
    ):
        bad = deepcopy(artifact)
        mutate(bad)
        if "reproducibility_checksum" not in match:
            bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
        with pytest.raises(ValueError, match=match):
            mod.validate_artifact(bad)

    bad_provenance = deepcopy(artifact)
    bad_provenance["field_provenance"]["status"]["principle"] = "wrong"
    with pytest.raises(ValueError, match="field_provenance:status"):
        mod.validate_artifact(bad_provenance)

    provenance_not_mapping = deepcopy(artifact)
    provenance_not_mapping["field_provenance"] = []
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(provenance_not_mapping)

    combined = deepcopy(artifact)
    combined["continuous_self_learning_task"] = False
    combined["inference_substrate"] = "wrong"
    combined["verifier_is_oracle"] = False
    combined["frozen_arms_and_budget_parity"]["event_budget_parity"] = False
    combined["per_update_non_forgetting_certificates"]["failed_certificate_count"] = 1
    combined["memory_cap_accounting"]["cap_compliance"] = False
    combined["rollback_restart_and_state_hashes"]["restart_equivalence"] = 0.0
    combined["rollback_restart_and_state_hashes"]["rollback_hash_mismatch_count"] = 1
    combined["no_model_weight_mutation"]["all_unchanged"] = False
    combined["hardware_mapping_contract"]["board_execution_performed"] = True
    combined["hardware_mapping_contract"]["speedup_claimed"] = True
    combined_reasons = set(mod.blocked_reasons(combined))
    assert {
        "continuous_self_learning_task",
        "inference_substrate",
        "verifier_is_oracle",
        "budget_parity",
        "nonforgetting_failure",
        "memory_cap",
        "restart_mismatch",
        "rollback_mismatch",
        "gguf_weight_mutation",
        "hardware_claim",
    } <= combined_reasons

    with monkeypatch.context() as scoped:
        scoped.setattr(mod, "shortcut_resistant_csl_ready_score", lambda item: 0.0)
        assert mod.blocked_reasons(artifact) == ["ready_score"]

    with monkeypatch.context() as scoped:
        scoped.setattr(
            mod.exp5894,
            "validate_artifact",
            lambda item: (_ for _ in ()).throw(ValueError("bad 5894")),
        )
        gate = mod._exp5894_gate(REPO)
        assert gate["validates"] is False
        assert gate["ok"] is False

    with pytest.raises(ValueError, match="unknown arm"):
        mod._predict_arm("unknown", mod.load_fixture_rows(REPO)[0])

    scalar_json = tmp_path / "scalar.json"
    scalar_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        mod._read_json(scalar_json)
    bad_jsonl = tmp_path / "bad.rows.jsonl"
    bad_jsonl.write_text("\n[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSONL object required"):
        mod._read_jsonl(bad_jsonl)
    assert mod.load_fixture_rows(tmp_path / "missing") == []
    assert mod._rows_to_jsonl(mod.load_fixture_rows(REPO)[:1]).endswith("\n")
