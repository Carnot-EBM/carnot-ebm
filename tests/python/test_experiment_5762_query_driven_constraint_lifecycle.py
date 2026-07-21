"""Tests for Exp5762 query-driven constraint lifecycle.

Spec refs: REQ-LEARN-5762, REQ-STORE-5762,
SCENARIO-LEARN-5762-QUERY-LIFECYCLE,
SCENARIO-LEARN-5762-MATCHED-CONTROLS,
SCENARIO-LEARN-5762-ROLLBACK-RESTART,
SCENARIO-STORE-5762.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5762_query_driven_constraint_lifecycle as mod


REPO = Path(__file__).resolve().parents[2]
LEARN_SPEC = REPO / "openspec/capabilities/self-learning/spec.md"
STORE_SPEC = REPO / "openspec/capabilities/constraint-store/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5762_query_driven_constraint_lifecycle.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5762_query_driven_constraint_lifecycle.py "
    "-m pytest tests/python/test_experiment_5762_query_driven_constraint_lifecycle.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5762_query_driven_constraint_lifecycle.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5762_query_driven_constraint_lifecycle.json"
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
    """REQ-LEARN-5762: build the deterministic lifecycle artifact once."""

    base = tmp_path_factory.mktemp("exp5762")
    return mod.run(
        result_path=base / mod.RESULT_RELATIVE_PATH.name,
        preconditions_checked=mod.fixture_preconditions(),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )


def _walk(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return list(value.keys()) + [item for sub in value.values() for item in _walk(sub)]
    if isinstance(value, list):
        return [item for sub in value for item in _walk(sub)]
    return [value]


def test_req_5762_specs_declare_query_lifecycle_and_store_contract() -> None:
    """REQ-LEARN-5762/REQ-STORE-5762: OpenSpec anchors fields and gates."""

    learn = LEARN_SPEC.read_text(encoding="utf-8")
    store = STORE_SPEC.read_text(encoding="utf-8")
    learn_section = learn[learn.index("## REQ-LEARN-5762") : learn.index("## REQ-LEARN-5737")]
    store_section = store[store.index("### REQ-STORE-5762") :]
    normalized = " ".join(learn_section.split())

    for marker in (
        "REQ-LEARN-5762",
        "SCENARIO-LEARN-5762-QUERY-LIFECYCLE",
        "SCENARIO-LEARN-5762-MATCHED-CONTROLS",
        "SCENARIO-LEARN-5762-ROLLBACK-RESTART",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "`constraint_recovery_gain_lcb`",
        "`prefix_retention_pass_score`",
        "`unsafe_update_count`",
        "`rollback_hash_mismatch_count`",
    ):
        assert marker in learn_section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in learn_section
    for principle in mod.REQUIRED_FIELD_PRINCIPLES.values():
        assert " ".join(principle.split()) in normalized
    for marker in (
        "REQ-STORE-5762",
        "SCENARIO-STORE-5762",
        "membership query hashes",
        "rejected updates have zero propagation",
    ):
        assert marker in store_section


def test_scenario_5762_query_lifecycle_artifact_fields_and_gates(
    artifact: dict[str, Any],
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-5762-QUERY-LIFECYCLE: artifact is sealed and credited."""

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
    assert artifact["continuous_self_learning_target"] is True
    assert artifact["continuous_self_learning_credited"] is True
    assert artifact["model_weight_mutation"] is False
    assert artifact["production_default_enabled"] is False
    assert artifact["verifier_is_oracle"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["behavioral_exact_accuracy"] == pytest.approx(1.0)
    assert artifact["constraint_precision"] == pytest.approx(1.0)
    assert artifact["constraint_recall"] == pytest.approx(1.0)
    assert artifact["constraint_f1"] == pytest.approx(1.0)
    assert artifact["overfit_constraint_removal_rate"] == pytest.approx(1.0)
    assert artifact["missing_constraint_recovery_rate"] == pytest.approx(1.0)
    assert artifact["constraint_recovery_gain"] > 0.0
    assert artifact["constraint_recovery_gain_lcb"] > 0.0
    assert artifact["prefix_retention_pass_score"] == pytest.approx(1.0)
    assert artifact["unsafe_update_count"] == 0
    assert artifact["rejected_update_propagation_count"] == 0
    assert artifact["rollback_hash_mismatch_count"] == 0
    assert artifact["oracle_boundary_violation_count"] == 0
    assert artifact["restart_equivalence"]["all_passed"] is True
    assert artifact["producer_gate_fields"] == list(mod.PRODUCER_GATE_FIELDS)
    for field in mod.PRODUCER_GATE_FIELDS:
        assert field in artifact
        assert not isinstance(artifact[field], dict)
    assert artifact["test_commands"] == TEST_COMMANDS
    assert artifact["test_exit_codes"] == TEST_EXIT_CODES
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)


def test_req_5762_membership_queries_do_not_leak_targets(artifact: dict[str, Any]) -> None:
    """REQ-LEARN-5762: learner receipts expose oracle answers, not target ASTs."""

    leak_tokens = {
        "faithful_accepts",
        "faithful_model_ast",
        "faithful_model_text",
        "expected_repair_operation",
        "expected_repair_receipt",
        "science_repair_receipt",
        "pseudo_label",
        "llm_output",
    }
    query_values = _walk(artifact["membership_query_receipts"])
    ledger_values = _walk(artifact["constraint_lifecycle_ledger"])

    assert artifact["membership_query_receipts"]
    assert not leak_tokens.intersection(str(value) for value in query_values + ledger_values)
    assert all(receipt["oracle_boundary"] == "exact_membership_answer_only" for receipt in artifact["membership_query_receipts"])
    assert all(receipt["query_hash"].startswith("sha256:") for receipt in artifact["membership_query_receipts"])
    assert all(receipt["confidence_after"] >= receipt["confidence_before"] for receipt in artifact["membership_query_receipts"])
    assert len(artifact["membership_query_receipts"]) <= artifact["query_budget"]["total"]
    assert artifact["preconditions_checked"]["oracle_boundary"]["no_target_ast_available_to_learner"] is True


def test_req_5762_lifecycle_receipts_cover_birth_refinement_quarantine_and_restart(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-LEARN-5762-ROLLBACK-RESTART: state receipts replay exactly."""

    ledger = artifact["constraint_lifecycle_ledger"]
    kinds = {row["variant_kind"] for row in ledger}
    birth_ids = {row["episode_id"] for row in artifact["constraint_birth_receipts"]}
    quarantine_ids = {row["episode_id"] for row in artifact["constraint_quarantine_receipts"]}
    refinement_ids = {row["episode_id"] for row in artifact["constraint_refinement_receipts"]}
    supersession_ids = {row["episode_id"] for row in artifact["constraint_supersession_receipts"]}

    assert kinds == {"incomplete", "mixed", "overfit"}
    assert birth_ids
    assert quarantine_ids
    assert refinement_ids == birth_ids
    assert supersession_ids == quarantine_ids
    assert any(row["variant_kind"] == "mixed" and len(row["operations"]) == 2 for row in ledger)
    assert all(row["promotion_gates"]["all_passed"] is True for row in artifact["constraint_birth_receipts"])
    assert all(row["rollback_hash_matches"] is True for row in ledger)
    assert all(row["restart_hash_matches"] is True for row in ledger)
    assert artifact["restart_equivalence"]["restart_hash_mismatch_count"] == 0
    assert artifact["restart_equivalence"]["rollback_hash_mismatch_count"] == 0
    assert artifact["state_growth"]["query_driven_refinement"]["active_constraint_growth"] >= 0


def test_scenario_5762_matched_controls_and_gain_math(artifact: dict[str, Any]) -> None:
    """SCENARIO-LEARN-5762-MATCHED-CONTROLS: gain is paired against controls."""

    expected_arms = {
        "query_driven_refinement",
        "passive_only_induction",
        "random_query_induction",
        "frozen_model",
        "safe_generic_residual_sidecar",
        "exact_query_budget_oracle_upper_bound",
    }
    non_oracle = [
        "passive_only_induction",
        "random_query_induction",
        "frozen_model",
        "safe_generic_residual_sidecar",
    ]
    assert set(artifact["control_definitions"]) == expected_arms
    assert set(artifact["per_arm_metrics"]) == expected_arms
    for definition in artifact["control_definitions"].values():
        assert definition["matched_examples"] is True
        assert definition["matched_query_budget"] == mod.QUERY_BUDGET_PER_EPISODE
        assert definition["matched_candidate_library"] is True
        assert definition["matched_stopping_rule"] == mod.STOPPING_RULE

    metrics = artifact["per_arm_metrics"]
    best_non_oracle = max(metrics[arm]["behavioral_exact_accuracy"] for arm in non_oracle)
    expected_gain = metrics["query_driven_refinement"]["behavioral_exact_accuracy"] - best_non_oracle
    assert artifact["constraint_recovery_gain"] == pytest.approx(round(expected_gain, 6))
    assert artifact["constraint_recovery_gain_lcb"] == pytest.approx(
        mod.paired_lcb95(artifact["paired_recovery_deltas"])
    )
    assert metrics["query_driven_refinement"]["dynamic_regret"] == pytest.approx(0.0)
    assert metrics["exact_query_budget_oracle_upper_bound"]["dynamic_regret"] == pytest.approx(0.0)
    assert metrics["query_driven_refinement"]["query_count"] == len(
        artifact["membership_query_receipts"]
    )
    assert artifact["dynamic_regret"] == metrics["query_driven_refinement"]["dynamic_regret"]


def test_req_5762_validation_and_preconditions_fail_closed(
    artifact: dict[str, Any],
    tmp_path: Path,
) -> None:
    """REQ-LEARN-5762: unsafe claims, bad checksums, and blockers reject."""

    for field, value, expected in (
        ("constraint_recovery_gain_lcb", -0.01, "constraint_recovery_gain_lcb"),
        ("prefix_retention_pass_score", 0.0, "prefix_retention_pass_score"),
        ("unsafe_update_count", 1, "unsafe_update_count"),
        ("rejected_update_propagation_count", 1, "rejected_update_propagation_count"),
        ("rollback_hash_mismatch_count", 1, "rollback_hash_mismatch_count"),
        ("oracle_boundary_violation_count", 1, "oracle_boundary_violation_count"),
        ("model_weight_mutation", True, "model_weight_mutation"),
        ("production_default_enabled", True, "production_default_enabled"),
        ("inference_substrate", "wrong", "inference_substrate"),
    ):
        bad = deepcopy(artifact)
        bad[field] = value
        bad["continuous_self_learning_credited"] = mod.continuous_self_learning_credited(bad)
        bad["status"] = "complete" if bad["continuous_self_learning_credited"] else "blocked"
        bad["honest_verdict"] = mod.honest_verdict(bad)
        bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)

    missing = deepcopy(artifact)
    del missing["query_budget"]
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)

    wrapped_gate = deepcopy(artifact)
    wrapped_gate["unsafe_update_count"] = {"value": 0}
    wrapped_gate["reproducibility_checksum"] = mod.reproducibility_checksum(wrapped_gate)
    with pytest.raises(ValueError, match="producer_gate_fields"):
        mod.validate_artifact(wrapped_gate)

    blocked_preconditions = mod.fixture_preconditions()
    blocked_preconditions["preconditions_ready"] = False
    blocked_preconditions["blocked_reasons"] = ["science_split_or_oracle_boundary_ambiguous"]
    blocked = mod.run(
        result_path=tmp_path / "blocked.json",
        preconditions_checked=blocked_preconditions,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )
    assert blocked["status"] == "blocked"
    assert blocked["continuous_self_learning_credited"] is False
    assert blocked["constraint_recovery_gain_lcb"] == pytest.approx(0.0)
    assert blocked["honest_verdict"].startswith("blocked:")
    assert mod.validate_artifact(blocked) is True


def test_req_5762_defensive_helpers_fail_closed(
    artifact: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-5762: malformed inputs and defensive branches are explicit."""

    list_json = tmp_path / "list.json"
    list_json.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        mod._read_json(list_json)

    missing = mod.collect_preconditions(
        benchmark_artifact_path=tmp_path / "missing.json",
        benchmark_manifest_path=tmp_path / "missing.jsonl",
        lifecycle_artifact_path=tmp_path / "missing_lifecycle.json",
        memory_probe=lambda: {"available_mb": 1, "required_mb": mod.RAM_FLOOR_MB, "ok": False},
        disk_probe=lambda: {"available_mb": 1, "required_mb": mod.DISK_FLOOR_MB, "ok": False},
    )
    assert missing["preconditions_ready"] is False
    for reason in (
        "exp5761_or_lifecycle_replay_failed",
        "memory",
        "disk",
        "benchmark_replay",
        "science_split",
        "oracle_boundary",
        "deterministic_chronological_seeds",
        "lifecycle_checkpoint_compatibility",
    ):
        assert reason in missing["blocked_reasons"]

    rows = mod.exp5761.read_benchmark_manifest(REPO / mod.exp5761.BENCHMARK_MANIFEST_RELATIVE_PATH)
    source_rows = mod._source_rows_by_id()
    row = next(item for item in rows if item["split"] == "train" and item["family"] == "finite_domain_csp")
    source = source_rows[row["source_instance_id"]]
    monkeypatch.setattr(mod, "_generic_candidate_constraints", lambda model_ast: [])
    with pytest.raises(ValueError, match="no train/dev template recovered"):
        mod._choose_missing_constraint_from_train_dev(row, source)

    bad_library = {
        "families": {
            row["family"]: {
                "parameter_rule": "unsupported",
                "template_constraint": {},
                "supporting_train_dev_case_count": 1,
            }
        }
    }
    with pytest.raises(ValueError, match="unsupported template parameter rule"):
        mod._instantiate_template_constraint(row, row["variants"][0]["model_ast"], bad_library)

    assert mod._percentile([], 0.95) == pytest.approx(0.0)
    assert mod.paired_lcb95([]) == pytest.approx(0.0)
    assert mod.paired_lcb95([0.25]) == pytest.approx(0.25)

    bad_restart = deepcopy(artifact)
    bad_restart["restart_equivalence"]["all_passed"] = False
    assert "restart_equivalence" in mod.blocked_reasons(bad_restart)

    bad_target = deepcopy(artifact)
    bad_target["continuous_self_learning_target"] = False
    assert "continuous_self_learning_target" in mod.blocked_reasons(bad_target)

    bad_verifier = deepcopy(artifact)
    bad_verifier["verifier_is_oracle"] = False
    assert "verifier_is_oracle" in mod.blocked_reasons(bad_verifier)

    stale = deepcopy(artifact)
    stale["honest_verdict"] = "blocked: stale"
    stale["reproducibility_checksum"] = mod.reproducibility_checksum(stale)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(stale)


def test_req_5762_repository_artifact_matches_deterministic_replay_if_present() -> None:
    """REQ-STORE-5762: checked-in result remains deterministic when present."""

    if not RESULT_PATH.exists():
        pytest.skip("repository Exp5762 artifact has not been generated yet")
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
