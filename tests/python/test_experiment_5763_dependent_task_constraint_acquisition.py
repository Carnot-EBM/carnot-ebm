"""Tests for Exp5763 dependent task constraint acquisition.

Spec refs: REQ-LEARN-5763, REQ-STORE-5763,
SCENARIO-LEARN-5763-DEPENDENT-STREAM,
SCENARIO-LEARN-5763-MATCHED-CONTROLS,
SCENARIO-LEARN-5763-RECOVERY-RESTART,
SCENARIO-STORE-5763.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5763_dependent_task_constraint_acquisition as mod


REPO = Path(__file__).resolve().parents[2]
LEARN_SPEC = REPO / "openspec/capabilities/self-learning/spec.md"
STORE_SPEC = REPO / "openspec/capabilities/constraint-store/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5763_dependent_task_constraint_acquisition.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5763_dependent_task_constraint_acquisition.py "
    "-m pytest tests/python/test_experiment_5763_dependent_task_constraint_acquisition.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5763_dependent_task_constraint_acquisition.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5763_dependent_task_constraint_acquisition.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
TEST_COMMANDS = [
    TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
]
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """REQ-LEARN-5763: build the deterministic dependent-stream artifact once."""

    base = tmp_path_factory.mktemp("exp5763")
    return mod.run(
        result_path=base / mod.RESULT_RELATIVE_PATH.name,
        preconditions_checked=mod.fixture_preconditions(),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )


def test_req_5763_specs_declare_dependent_stream_and_store_contract() -> None:
    """REQ-LEARN-5763/REQ-STORE-5763: OpenSpec anchors fields and gates."""

    learn = LEARN_SPEC.read_text(encoding="utf-8")
    store = STORE_SPEC.read_text(encoding="utf-8")
    learn_section = learn[learn.index("## REQ-LEARN-5763") : learn.index("## REQ-LEARN-5737")]
    store_section = store[store.index("### REQ-STORE-5763") :]

    for marker in (
        "REQ-LEARN-5763",
        "SCENARIO-LEARN-5763-DEPENDENT-STREAM",
        "SCENARIO-LEARN-5763-MATCHED-CONTROLS",
        "SCENARIO-LEARN-5763-RECOVERY-RESTART",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "session_count >= 60",
        "`dependent_task_ca_ready_score`",
        "`rollback_hash_mismatch_count`",
        "online LoRA",
        "GGUF weight writes",
    ):
        assert marker in learn_section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in learn_section
    for marker in (
        "REQ-STORE-5763",
        "SCENARIO-STORE-5763",
        "dependency DAG hash",
        "zero propagation",
    ):
        assert marker in store_section


def test_scenario_5763_artifact_fields_gates_and_deterministic_replay(
    artifact: dict[str, Any],
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-5763-DEPENDENT-STREAM: artifact is sealed and credited."""

    destination = tmp_path / mod.RESULT_RELATIVE_PATH.name
    replay = mod.run(
        result_path=destination,
        preconditions_checked=mod.fixture_preconditions(),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )
    loaded = json.loads(destination.read_text(encoding="utf-8"))

    assert artifact == replay == loaded
    assert mod.validate_artifact(artifact) is True
    assert set(artifact) == set(artifact["field_principles"])
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["preconditions_checked"]["preconditions_ready"] is True
    assert artifact["generator_version"] == mod.GENERATOR_VERSION
    assert artifact["session_count"] >= 60
    assert artifact["heldout_composition_count"] > 0
    assert artifact["continuous_self_learning_target"] is True
    assert artifact["continuous_self_learning_credited"] is True
    assert artifact["model_weight_mutation"] is False
    assert artifact["production_default_enabled"] is False
    assert artifact["verifier_is_oracle"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["dependent_task_ca_ready_score"] == pytest.approx(1.0)
    assert artifact["compositional_exact_accuracy"] == pytest.approx(1.0)
    assert artifact["constraint_recovery_rate"] == pytest.approx(1.0)
    assert artifact["old_task_retention_delta"] >= 0.0
    assert artifact["forward_transfer"] > 0.0
    assert artifact["unsafe_update_count"] == 0
    assert artifact["rejected_update_propagation_count"] == 0
    assert artifact["rollback_hash_mismatch_count"] == 0
    assert artifact["restart_equivalence"]["all_passed"] is True
    assert artifact["test_commands"] == TEST_COMMANDS
    assert artifact["test_exit_codes"] == TEST_EXIT_CODES
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)


def test_req_5763_stream_seals_dependencies_labels_and_recovery_controls(
    artifact: dict[str, Any],
) -> None:
    """REQ-LEARN-5763: exact validators seal dependencies before learner access."""

    ledger = artifact["dependent_session_ledger"]
    operation_receipts = artifact["transition_receipts"]
    relation_types = {edge["relation"] for edge in artifact["dependency_graph"]["edges"]}
    operations = {receipt["operation"] for receipt in operation_receipts}

    assert len(ledger) == artifact["session_count"]
    assert artifact["dependency_graph_hash"] == mod.sha256_json(artifact["dependency_graph"])
    assert artifact["stream_root_hash"] == mod.sha256_json(ledger)
    assert artifact["operation_order_hash"] == mod.sha256_json(
        [row["operation"] for row in operation_receipts]
    )
    assert {"compose", "depend", "supersede", "narrow", "conflict"} <= relation_types
    assert {"add", "refine", "quarantine", "supersede", "forget", "rollback"} <= operations
    assert all(row["exact_validator_receipt"]["label_minted_before_learner"] for row in ledger)
    assert all(row["learner_target_boundary"] == "membership_answer_only" for row in ledger)
    assert all(row["protected_prefix_hash"].startswith("sha256:") for row in ledger)
    assert artifact["heldout_composition_manifest"]["sealed_before_learner_access"] is True
    assert artifact["heldout_composition_count"] == len(
        artifact["heldout_composition_manifest"]["session_ids"]
    )
    assert artifact["conflict_manifest"]["contradictory_update_count"] > 0
    assert artifact["supersession_manifest"]["supersession_count"] > 0
    assert artifact["delayed_counterexample_manifest"]["delayed_count"] > 0
    assert artifact["shift_manifest"]["shift_count"] > 0
    assert artifact["crash_injection_manifest"]["crash_count"] >= len(mod.LIFECYCLE_BOUNDARIES)
    assert artifact["corruption_controls"]["checkpoint_corruption_rejected"] is True
    assert artifact["corruption_controls"]["orphan_ledger_row_rejected"] is True


def test_scenario_5763_matched_controls_and_metric_math(artifact: dict[str, Any]) -> None:
    """SCENARIO-LEARN-5763-MATCHED-CONTROLS: controls are matched and gains are paired."""

    expected_arms = {
        "qualified_query_driven_lifecycle",
        "passive_only_induction",
        "random_query_induction",
        "frozen_model",
        "safe_generic_residual_sidecar",
        "reset_each_session",
    }
    assert set(artifact["control_definitions"]) == expected_arms
    assert set(artifact["per_arm_metrics"]) == expected_arms

    for definition in artifact["control_definitions"].values():
        assert definition["matched_examples"] is True
        assert definition["matched_query_update_opportunities"] is True
        assert definition["matched_state_budget"] == mod.STATE_BUDGET
        assert definition["matched_stopping_rule"] == mod.STOPPING_RULE

    metrics = artifact["per_arm_metrics"]
    best_non_oracle = max(
        metrics[arm]["compositional_exact_accuracy"]
        for arm in mod.NON_ORACLE_NON_RESET_CONTROL_ARMS
    )
    expected_forward = metrics["qualified_query_driven_lifecycle"][
        "compositional_exact_accuracy"
    ] - best_non_oracle
    query_updates = metrics["qualified_query_driven_lifecycle"]["accepted_update_count"]
    query_count = metrics["qualified_query_driven_lifecycle"]["query_count"]

    assert artifact["forward_transfer"] == pytest.approx(round(expected_forward, 6))
    assert artifact["query_efficiency"] == pytest.approx(round(query_updates / query_count, 6))
    assert artifact["dynamic_regret"] == pytest.approx(
        metrics["qualified_query_driven_lifecycle"]["dynamic_regret"]
    )
    assert metrics["qualified_query_driven_lifecycle"]["dynamic_regret"] == pytest.approx(0.0)
    assert metrics["reset_each_session"]["old_task_retention"] < metrics[
        "qualified_query_driven_lifecycle"
    ]["old_task_retention"]
    assert artifact["paired_confidence_intervals"]["forward_transfer_lcb95"] > 0.0


def test_scenario_5763_recovery_certificates_are_exact(artifact: dict[str, Any]) -> None:
    """SCENARIO-LEARN-5763-RECOVERY-RESTART: recovery receipts restore state hashes."""

    certificate = artifact["nonforgetting_certificate"]
    restart = artifact["restart_equivalence"]
    recovery_receipts = artifact["recovery_receipts"]

    assert certificate["protected_prefix_count"] >= 1
    assert certificate["certificate_rate"] == pytest.approx(1.0)
    assert certificate["all_prefixes_exact"] is True
    assert certificate["certificate_hash"] == mod.sha256_json(
        certificate["protected_prefix_receipts"]
    )
    assert restart["restart_hash_mismatch_count"] == 0
    assert restart["rollback_hash_mismatch_count"] == 0
    assert restart["crash_recovery_hash_mismatch_count"] == 0
    assert restart["checkpoint_corruption_hash_mismatch_count"] == 0
    assert all(row["restored_state_hash"] == row["expected_state_hash"] for row in recovery_receipts)
    assert all(row["rejected_update_propagation_count"] == 0 for row in recovery_receipts)


def test_req_5763_validation_and_preconditions_fail_closed(
    artifact: dict[str, Any],
    tmp_path: Path,
) -> None:
    """REQ-LEARN-5763: malformed gates, checksums, and blockers reject."""

    for field, value, expected in (
        ("dependent_task_ca_ready_score", 0.0, "dependent_task_ca_ready_score"),
        ("session_count", 59, "session_count"),
        ("heldout_composition_count", 0, "heldout_composition_count"),
        ("forward_transfer", 0.0, "forward_transfer"),
        ("compositional_exact_accuracy", 0.99, "compositional_exact_accuracy"),
        ("constraint_recovery_rate", 0.99, "constraint_recovery_rate"),
        ("old_task_retention_delta", -0.01, "old_task_retention_delta"),
        ("unsafe_update_count", 1, "unsafe_update_count"),
        ("rejected_update_propagation_count", 1, "rejected_update_propagation_count"),
        ("rollback_hash_mismatch_count", 1, "rollback_hash_mismatch_count"),
        ("model_weight_mutation", True, "model_weight_mutation"),
        ("production_default_enabled", True, "production_default_enabled"),
        ("verifier_is_oracle", False, "verifier_is_oracle"),
        ("inference_substrate", "wrong", "inference_substrate"),
    ):
        bad = deepcopy(artifact)
        bad[field] = value
        bad["continuous_self_learning_credited"] = mod.continuous_self_learning_credited(bad)
        if field != "dependent_task_ca_ready_score":
            bad["dependent_task_ca_ready_score"] = mod.dependent_task_ca_ready_score(bad)
        bad["status"] = "complete" if bad["continuous_self_learning_credited"] else "blocked"
        bad["honest_verdict"] = mod.honest_verdict(bad)
        bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)

    missing = deepcopy(artifact)
    del missing["generator_version"]
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)

    wrapped_gate = deepcopy(artifact)
    wrapped_gate["status"] = {"value": "complete"}
    wrapped_gate["reproducibility_checksum"] = mod.reproducibility_checksum(wrapped_gate)
    assert "bare_gate_fields" in mod.blocked_reasons(wrapped_gate)
    with pytest.raises(ValueError, match="bare_gate_fields"):
        mod.validate_artifact(wrapped_gate)

    missing_principle = deepcopy(artifact)
    missing_principle["field_principles"].pop("stream_root_hash")
    missing_principle["reproducibility_checksum"] = mod.reproducibility_checksum(missing_principle)
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(missing_principle)

    bad_restart = deepcopy(artifact)
    bad_restart["restart_equivalence"]["all_passed"] = False
    assert "restart_equivalence" in mod.blocked_reasons(bad_restart)

    blocked_preconditions = mod.fixture_preconditions()
    blocked_preconditions["preconditions_ready"] = False
    blocked_preconditions["blocked_reasons"] = ["exp5762_positive_gate_replay_failed"]
    blocked = mod.run(
        result_path=tmp_path / "blocked.json",
        preconditions_checked=blocked_preconditions,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )
    assert blocked["status"] == "blocked"
    assert blocked["continuous_self_learning_credited"] is False
    assert blocked["dependent_task_ca_ready_score"] == pytest.approx(0.0)
    assert blocked["honest_verdict"].startswith("blocked:")
    assert mod.validate_artifact(blocked) is True


def test_req_5763_defensive_helpers_fail_closed(tmp_path: Path) -> None:
    """REQ-LEARN-5763: helper parsers and missing inputs fail closed."""

    list_json = tmp_path / "list.json"
    list_json.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        mod._read_json(list_json)

    missing = mod.collect_preconditions(
        exp5762_artifact_path=tmp_path / "missing.json",
        memory_probe=lambda: {"available_mb": 1, "required_mb": mod.RAM_FLOOR_MB, "ok": False},
        disk_probe=lambda: {"available_mb": 1, "required_mb": mod.DISK_FLOOR_MB, "ok": False},
    )
    assert missing["preconditions_ready"] is False
    for reason in (
        "exp5762_positive_gate_replay_failed",
        "memory",
        "disk",
        "fixed_seeds",
        "immutable_base_model_boundary",
    ):
        assert reason in missing["blocked_reasons"]

    assert mod.paired_lcb95([]) == pytest.approx(0.0)
    assert mod.paired_lcb95([0.25]) == pytest.approx(0.25)
    assert mod._percentile([], 0.95) == pytest.approx(0.0)


def test_req_5763_repository_artifact_matches_deterministic_replay_if_present() -> None:
    """REQ-STORE-5763: checked-in result remains deterministic when present."""

    if not RESULT_PATH.exists():
        pytest.skip("repository Exp5763 artifact has not been generated yet")
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = mod.run(
        result_path=RESULT_PATH,
        preconditions_checked=result["preconditions_checked"],
        test_commands=result["test_commands"],
        test_exit_codes=result["test_exit_codes"],
        write=False,
    )
    assert result == replay
    assert mod.validate_artifact(result) is True
