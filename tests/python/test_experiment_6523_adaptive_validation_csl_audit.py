"""Tests for Exp6523 adaptive validation CSL audit.

Spec refs: REQ-STORE-6523, SCENARIO-STORE-6523-REPLAY,
SCENARIO-STORE-6523-ADAPTIVE-PROBABILITIES,
SCENARIO-STORE-6523-SENTINEL-FULL-BACKSTOP,
SCENARIO-STORE-6523-COST-DECISION, SCENARIO-STORE-6523-ATTACKS,
SCENARIO-STORE-6523-SCHEMA.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6523_adaptive_validation_csl_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH
TESTS_RUN = [{"command": command, "exit_code": 0} for command in mod.DEFAULT_TEST_COMMANDS]


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """REQ-STORE-6523: build the audit without writing tracked results."""

    root = tmp_path_factory.mktemp("exp6523")
    return mod.build_artifact(
        repo_root=REPO,
        result_path=root / mod.RESULT_RELATIVE_PATH.name,
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )


def test_req_store_6523_spec_declares_adaptive_validation_contract() -> None:
    """REQ-STORE-6523: OpenSpec owns the adaptive validation audit."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-STORE-6523") :]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-STORE-6523-REPLAY",
        "SCENARIO-STORE-6523-ADAPTIVE-PROBABILITIES",
        "SCENARIO-STORE-6523-SENTINEL-FULL-BACKSTOP",
        "SCENARIO-STORE-6523-COST-DECISION",
        "SCENARIO-STORE-6523-ATTACKS",
        "SCENARIO-STORE-6523-SCHEMA",
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`verifier_is_oracle` to bare `false`",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_store_6523_schema_gate_and_replay_receipts(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-STORE-6523-REPLAY/SCHEMA: gate and rows are independent."""

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete_positive_adaptive_validation_csl_audit"
    assert artifact["honest_verdict"].startswith("complete_positive_")
    assert artifact["verdict_class"] == "positive"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert set(artifact["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
    assert mod.validate_artifact(artifact) == []

    gate = artifact["upstream_gate_receipt"]
    assert gate["path"] == mod.EXP6522_RELATIVE_PATH.as_posix()
    assert gate["field"] == "csl_execution_complete_score"
    assert gate["expected_value"] == 1.0
    assert gate["observed_value"] == 1.0
    assert gate["gate_passed"] is True
    assert gate["artifact_sha256"].startswith("sha256:")
    assert gate["source_method_hash"].startswith("sha256:")
    assert gate["row_counts"]["per_unit_rows"] == 531
    assert gate["resources"]["filesystem_writable"] is True
    assert gate["protected_file_hashes"]

    prior = artifact["prior_failure_receipt"]
    assert {row["path"] for row in prior["rows"]} >= {
        mod.EXP6498_RELATIVE_PATH.as_posix(),
        mod.EXP6522_RELATIVE_PATH.as_posix(),
    }
    assert prior["unsafe_inherited_claim_count"] == 0


def test_scenario_store_6523_recomputes_exp6522_rows(artifact: dict[str, Any]) -> None:
    """SCENARIO-STORE-6523-REPLAY: Exp6522 metrics are recomputed from rows."""

    replay = artifact["independent_csl_row_recomputation"]
    lifecycle = artifact["lifecycle_and_safety_audit"]
    prefix = artifact["prefix_retention_audit"]
    held = artifact["held_future_support_audit"]

    assert replay["forbidden_producer_imports_clean"] is True
    assert replay["exact_answer_mismatch_count"] == 0
    assert replay["row_family_counts"]["per_game_results"] == 91
    assert replay["row_family_counts"]["lifecycle_action_rows"] == 148
    assert replay["row_family_counts"]["held_future_support_rows"] == 7
    assert replay["recomputed_aggregate"]["candidate_score_from_rows"] == 1.0
    assert replay["source_aggregate_matches_recomputed"] is True

    assert lifecycle["unsafe_write_count"] == 0
    assert lifecycle["unsafe_use_count"] == 0
    assert lifecycle["all_exact_answers_equal"] is True
    assert lifecycle["capacity_restart_rollback_passed"] is True
    assert lifecycle["invalid_reuse_vetoed"] is True
    assert lifecycle["interference_safe"] is True

    assert prefix["prefix_retention_within_margin"] is True
    assert prefix["minimum_support_after"] == 1.0

    assert held["claim_eligible_from_full_audit"] is True
    assert held["full_audit_winner"] == "valid_unbounded_reuse"
    assert held["winner_tie_set"] == ["valid_unbounded_reuse", "restart"]
    assert held["benefit_vs_scratch"]["valid_bounded_reuse"] == 80
    assert held["benefit_vs_scratch"]["valid_unbounded_reuse"] == 84


def test_scenario_store_6523_adaptive_probabilities_ipw_and_backstop(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-STORE-6523-ADAPTIVE-PROBABILITIES: IPW stays bounded."""

    contract = artifact["full_fixed_adaptive_arm_contract"]
    assert contract["validation_arms"] == [
        "full_set",
        "fixed_subset",
        "variance_weighted_adaptive",
    ]
    assert contract["release_authority"] == "final_full_held_audit_only"
    assert contract["adaptive_estimator"] == "inverse_probability_weighted"
    assert contract["exact_sentinel_task_ids"] == list(mod.EXACT_SENTINEL_TASK_IDS)
    assert contract["candidate_arms"] == list(mod.CANDIDATE_ARMS)

    held_task_count = len(contract["held_task_ids"])
    iterations = contract["iteration_count"]
    candidates = len(contract["candidate_arms"])

    sentinel_keys = {
        (row["iteration"], row["task_id"]) for row in artifact["exact_sentinel_rows"]
    }
    for iteration in range(1, iterations + 1):
        for task_id in mod.EXACT_SENTINEL_TASK_IDS:
            assert (iteration, task_id) in sentinel_keys

    adaptive_probs = [
        row
        for row in artifact["inclusion_probability_rows"]
        if row["validation_arm"] == "variance_weighted_adaptive"
    ]
    assert len(adaptive_probs) == iterations * held_task_count
    assert all(row["inclusion_probability"] > 0.0 for row in adaptive_probs)
    assert all(row["uses_only_prior_outcomes"] is True for row in adaptive_probs)
    assert all(row["immutable_sentinel"] is True for row in adaptive_probs if row["task_id"] in mod.EXACT_SENTINEL_TASK_IDS)

    adaptive_selected = [
        row
        for row in artifact["validation_selection_rows"]
        if row["validation_arm"] == "variance_weighted_adaptive" and row["selected"]
    ]
    assert len(adaptive_selected) == iterations * (len(mod.EXACT_SENTINEL_TASK_IDS) + 1)
    assert all(row["charged_candidate_evaluations"] == candidates for row in adaptive_selected)

    adaptive_estimates = [
        row
        for row in artifact["ipw_estimate_rows"]
        if row["validation_arm"] == "variance_weighted_adaptive"
    ]
    assert len(adaptive_estimates) == iterations * candidates
    assert all(row["ipw_estimate_valid"] is True for row in adaptive_estimates)
    assert all(row["uncertainty"] >= 0.0 for row in adaptive_estimates)
    assert all(row["full_set_conclusion_agreement"] is True for row in adaptive_estimates)

    final_full = artifact["final_full_audit_rows"]
    assert len(final_full) == held_task_count * candidates
    assert all(row["forced_by_backstop"] is True for row in final_full)
    assert all(row["exact_answer_equal"] is True for row in final_full)


def test_scenario_store_6523_cost_decision_and_attack_matrix(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-STORE-6523-COST-DECISION/ATTACKS: shortcuts fail closed."""

    costs = {row["validation_arm"]: row for row in artifact["cost_and_decision_agreement_rows"]}
    assert costs["variance_weighted_adaptive"]["charged_checks"] < costs["full_set"]["charged_checks"]
    assert costs["variance_weighted_adaptive"]["decision_agreement_with_full"] is True
    assert costs["variance_weighted_adaptive"]["rank_agreement_with_full"] is True
    assert costs["variance_weighted_adaptive"]["cost_saving_vs_full_checks"] > 0
    assert costs["variance_weighted_adaptive"]["final_full_backstop_completed"] is True
    assert costs["full_set"]["winning_candidate"] == "valid_unbounded_reuse"
    assert costs["fixed_subset"]["decision_agreement_with_full"] is True

    expected_attacks = {
        "zero_probability_tasks",
        "future_leakage",
        "self_selection",
        "weight_collapse",
        "sentinel_omission",
        "favorable_stopping",
        "fixed_subset_luck",
        "ipw_instability",
        "hidden_full_audits",
        "cost_saving_changes_winning_decision",
    }
    matrix = artifact["adaptive_attack_matrix"]
    assert expected_attacks == {row["attack_id"] for row in matrix["rows"]}
    assert matrix["all_critical_attacks_fail_closed"] is True
    assert matrix["false_accept_count"] == 0
    assert all(row["fail_closed"] is True for row in matrix["rows"])
    assert artifact["adaptive_validation_ready_score"] == 1.0
    assert artifact["continuous_self_learning_claim_eligible_score"] == 1.0
    assert artifact["gate_check_summary"]["all_gates_passed"] is True


def test_scenario_store_6523_validation_and_cli_roundtrip(
    tmp_path: Path,
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-STORE-6523-SCHEMA: malformed artifacts fail validation."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH.name
    assert mod.main(["--date", "20260823", "--result-path", str(result_path)]) == 0
    assert mod.main(["--validate", "--result-path", str(result_path)]) == 0
    written = json.loads(result_path.read_text(encoding="utf-8"))
    assert written["adaptive_validation_ready_score"] == 1.0

    mutations = [
        ("required field set mismatch", lambda item: item.pop("status")),
        ("status lacks terminal prefix", lambda item: item.__setitem__("status", "running")),
        ("honest_verdict lacks terminal prefix", lambda item: item.__setitem__("honest_verdict", "running")),
        ("verdict_class outside Exp6523 enum", lambda item: item.__setitem__("verdict_class", "circular_positive")),
        ("inference_substrate mismatch", lambda item: item.__setitem__("inference_substrate", "live_llm")),
        ("verifier_is_oracle must be false", lambda item: item.__setitem__("verifier_is_oracle", True)),
        ("field_principles mismatch", lambda item: item.__setitem__("field_principles", {})),
        ("field_provenance must cover required fields", lambda item: item.__setitem__("field_provenance", {})),
        ("upstream gate failed", lambda item: item["upstream_gate_receipt"].__setitem__("gate_passed", False)),
        ("protected files changed", lambda item: item["protected_files_unchanged"].__setitem__("all_protected_files_unchanged", False)),
        ("adaptive_validation_ready_score mismatch", lambda item: item.__setitem__("adaptive_validation_ready_score", 0.0)),
        ("continuous_self_learning_claim_eligible_score mismatch", lambda item: item.__setitem__("continuous_self_learning_claim_eligible_score", 0.0)),
        ("adaptive zero inclusion probability", lambda item: item["inclusion_probability_rows"][0].__setitem__("inclusion_probability", 0.0)),
        ("exact sentinel coverage mismatch", lambda item: item["exact_sentinel_rows"].pop()),
        ("final full audit incomplete", lambda item: item["final_full_audit_rows"].pop()),
        ("adaptive attack matrix failed", lambda item: item["adaptive_attack_matrix"]["rows"][0].__setitem__("fail_closed", False)),
        ("reproducibility_checksum mismatch", lambda item: item.__setitem__("reproducibility_checksum", "sha256:bad")),
    ]
    for expected, mutate in mutations:
        broken = deepcopy(artifact)
        mutate(broken)
        if expected != "reproducibility_checksum mismatch":
            broken["reproducibility_checksum"] = mod.reproducibility_checksum(broken)
        assert expected in mod.validate_artifact(broken)

    invalid_path = tmp_path / "invalid.json"
    invalid = deepcopy(written)
    invalid["status"] = "running"
    invalid["reproducibility_checksum"] = mod.reproducibility_checksum(invalid)
    invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="status lacks terminal prefix"):
        mod.main(["--validate", "--result-path", str(invalid_path)])

    relative = mod.build_artifact(
        repo_root=REPO,
        result_path=Path("relative-exp6523.json"),
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )
    assert relative["preconditions_checked"]["result_path"].endswith("relative-exp6523.json")

    unsat = mod._solve_accounting(
        {
            "variable_count": 1,
            "clauses": [[1], [-1]],
            "schema_version": "fixture",
            "solver_hash": "fixture",
        }
    )
    assert unsat["exact_status"] == "unsat"

    source = json.loads((REPO / mod.EXP6522_RELATIVE_PATH).read_text(encoding="utf-8"))
    invalid_ipw = mod._estimate_for_selection(
        validation_arm="fixed_subset",
        iteration=99,
        selected_task_ids=["a2_held_future"],
        probabilities={"a2_held_future": 0.0},
        held=mod._held_rows(source),
        truth=mod._full_truth(source),
    )
    assert all(row["ipw_estimate_valid"] is False for row in invalid_ipw)

    aggregate = deepcopy(artifact["aggregate_row_recomputation"])
    gates = deepcopy(artifact["gate_check_summary"])
    aggregate["adaptive_ready_from_rows"] = False
    aggregate["adaptive_validation_ready_score_from_rows"] = 0.0
    assert mod.status_and_verdict(aggregate, gates)[2] == "partial"

    incomplete = deepcopy(artifact["aggregate_row_recomputation"])
    incomplete["all_planned_rows_terminal"] = False
    assert mod.status_and_verdict(incomplete, gates)[2] == "partial"

    no_claim = deepcopy(aggregate)
    no_claim["claim_eligible_from_full_audit"] = False
    no_claim["continuous_self_learning_claim_eligible_score_from_rows"] = 0.0
    no_claim["adaptive_ready_from_rows"] = True
    no_claim["adaptive_validation_ready_score_from_rows"] = 1.0
    assert mod.status_and_verdict(no_claim, gates)[2] == "partial"

    null_aggregate = deepcopy(no_claim)
    null_aggregate["adaptive_ready_from_rows"] = False
    null_aggregate["adaptive_validation_ready_score_from_rows"] = 0.0
    assert mod.status_and_verdict(null_aggregate, gates)[2] is None

    blocked_gates = deepcopy(gates)
    blocked_gates["checks"]["upstream_gate_passed"] = False
    blocked_gates["all_gates_passed"] = False
    assert mod.status_and_verdict(aggregate, blocked_gates)[2] == "blocked"

    disqualified = deepcopy(aggregate)
    disqualified["exact_answer_mismatch_count"] = 1
    assert mod.status_and_verdict(disqualified, gates)[2] == "disqualified"
