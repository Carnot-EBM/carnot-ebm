"""Tests for Exp5858 reduced-oracle continuous self-learning.

Spec refs: REQ-LEARN-5858, SCENARIO-LEARN-5858-PRECONDITIONS,
SCENARIO-LEARN-5858-QUERY-SELECTION, SCENARIO-LEARN-5858-PROMOTION-ROLLBACK,
SCENARIO-LEARN-5858-METRICS-CONTROLS, SCENARIO-LEARN-5858-FAIL-CLOSED.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5858_reduced_oracle_continuous_self_learning as mod


REPO = Path(__file__).resolve().parents[2]
SELF_LEARNING_SPEC = REPO / "openspec/capabilities/self-learning/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5858_reduced_oracle_continuous_self_learning.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5858_reduced_oracle_continuous_self_learning.py "
    "-m pytest tests/python/test_experiment_5858_reduced_oracle_continuous_self_learning.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5858_reduced_oracle_continuous_self_learning.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5858_reduced_oracle_continuous_self_learning.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5858_reduced_oracle_continuous_self_learning.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_FILE_COMMAND = (
    '.venv/bin/python -c "from pathlib import Path; '
    "assert Path('scripts/research_conductor.py').exists()\""
)
TEST_COMMANDS = [
    TEST_COMMAND,
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
        row_path=tmp_path / mod.ROW_RELATIVE_PATH.name,
        checkpoint_path=tmp_path / mod.CHECKPOINT_RELATIVE_PATH.name,
        memory_probe=lambda: {
            "available_mb": 8192,
            "required_mb": mod.RAM_FLOOR_MB,
            "ok": True,
        },
        disk_probe=lambda root: {
            "available_mb": 8192,
            "required_mb": mod.DISK_FLOOR_MB,
            "ok": True,
        },
    )


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """REQ-LEARN-5858: build the reduced-oracle A/B artifact once."""

    base = tmp_path_factory.mktemp("exp5858")
    return mod.run(
        result_path=base / mod.RESULT_RELATIVE_PATH.name,
        row_path=base / mod.ROW_RELATIVE_PATH.name,
        checkpoint_path=base / mod.CHECKPOINT_RELATIVE_PATH.name,
        preconditions_checked=_preconditions(base),
        duration_s=2.25,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )


def test_req_learn_5858_spec_declares_sparse_oracle_contract() -> None:
    """REQ-LEARN-5858: OpenSpec preregisters budgets, fields, and scenarios."""

    text = SELF_LEARNING_SPEC.read_text(encoding="utf-8")
    section = text[text.index("## REQ-LEARN-5858") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5858",
        "SCENARIO-LEARN-5858-PRECONDITIONS",
        "SCENARIO-LEARN-5858-QUERY-SELECTION",
        "SCENARIO-LEARN-5858-PROMOTION-ROLLBACK",
        "SCENARIO-LEARN-5858-METRICS-CONTROLS",
        "SCENARIO-LEARN-5858-FAIL-CLOSED",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.ROW_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`continuous_self_learning_ready_score`",
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_learn_5858_terminal_artifact_and_rows_are_hash_bound(
    artifact: dict[str, Any],
    tmp_path: Path,
) -> None:
    """REQ-LEARN-5858: terminal JSON and JSONL rows are deterministic evidence."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH.name
    row_path = tmp_path / mod.ROW_RELATIVE_PATH.name
    checkpoint_path = tmp_path / mod.CHECKPOINT_RELATIVE_PATH.name
    replay = mod.run(
        result_path=result_path,
        row_path=row_path,
        checkpoint_path=checkpoint_path,
        preconditions_checked=_preconditions(tmp_path),
        duration_s=2.25,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )
    loaded = json.loads(result_path.read_text(encoding="utf-8"))
    rows = mod.read_result_rows(row_path)

    assert replay == loaded
    assert mod.validate_artifact(replay) is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(replay)
    assert replay["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    assert replay["reproducibility_checksum"] == mod.reproducibility_checksum(replay)
    assert replay["status"] == "ready"
    assert replay["honest_verdict"].startswith("ready:")
    assert replay["continuous_self_learning_task"] is True
    assert replay["continuous_self_learning_ready_score"] == pytest.approx(1.0)
    assert isinstance(replay["continuous_self_learning_ready_score"], float)
    assert replay["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert replay["verifier_is_oracle"] is True
    assert replay["no_model_weight_mutation"] is True
    assert replay["preconditions_checked"]["preconditions_ready"] is True
    assert replay["upstream_hashes_and_gate_receipts"]["lifecycle_gate"]["ok"] is True
    assert replay["upstream_hashes_and_gate_receipts"]["replay_gate"]["ok"] is True
    assert replay["row_file_receipt"]["row_count"] == 360
    assert replay["row_file_receipt"]["sha256"] == mod.sha256_file(row_path)
    assert replay["row_file_receipt"]["row_receipt_hash_root"].startswith("sha256:")
    assert len(rows) == replay["row_file_receipt"]["row_count"]
    assert rows[0]["row_receipt_hash"].startswith("sha256:")
    assert checkpoint_path.exists()
    assert replay["test_commands"] == TEST_COMMANDS
    assert replay["test_exit_codes"] == TEST_EXIT_CODES


def test_scenario_learn_5858_query_selection_is_prospective_and_matched(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-LEARN-5858-QUERY-SELECTION: reduced feedback is label-blind."""

    protocol = artifact["frozen_protocol_and_query_budgets"]
    parity = artifact["arm_definitions_and_event_parity"]
    visibility = artifact["chronology_and_visibility_receipts"]
    query = artifact["query_selection_and_rejected_buffer_receipts"]

    assert protocol["policy_frozen_before_science"] is True
    assert protocol["reduced_oracle_budget"]["max_fraction_of_full_queries"] == pytest.approx(
        mod.REDUCED_QUERY_FRACTION_MAX
    )
    assert protocol["promotion_objectives"]["full_oracle_lift_fraction_min"] == pytest.approx(
        mod.FULL_ORACLE_LIFT_FRACTION_MIN
    )
    assert protocol["selector_policy"]["uses_future_labels"] is False
    assert protocol["selector_policy"]["uses_direct_label_derived_features"] is False
    assert protocol["selector_policy"]["uses_model_logits"] is False
    assert protocol["selector_policy"]["uses_row_ids"] is False
    assert protocol["selector_policy"]["uses_family_labels"] is False

    assert parity["arms"] == list(mod.ARM_NAMES)
    assert parity["event_parity_passed"] is True
    assert parity["all_arms_event_count"] == 360
    assert parity["same_chronological_event_hash"] is True
    assert len({item["event_hash_root"] for item in parity["arm_event_receipts"].values()}) == 1

    assert visibility["future_labels_visible_before_event_count"] == 0
    assert visibility["direct_label_feature_into_selector_count"] == 0
    assert visibility["future_labels_sealed_until_event"] is True
    assert visibility["sample_visibility_receipts"][0]["future_label_visible_before_decision"] is False

    assert query["reduced_oracle"]["query_event_count"] == 6
    assert query["reduced_oracle"]["exact_queries_used"] == 12
    assert query["full_oracle"]["exact_queries_used"] == 720
    assert query["random_query"]["exact_queries_used"] == query["reduced_oracle"][
        "exact_queries_used"
    ]
    assert query["reduced_oracle"]["exact_query_fraction_of_full"] <= mod.REDUCED_QUERY_FRACTION_MAX
    assert query["reduced_oracle"]["selector_uses_current_or_past_state_only"] is True
    assert query["reduced_oracle"]["selector_feature_hash_root"].startswith("sha256:")
    assert query["reduced_oracle"]["acquisition_precision_recall"]["precision"] == pytest.approx(
        1.0
    )
    assert query["reduced_oracle"]["acquisition_precision_recall"]["recall"] == pytest.approx(
        1.0
    )
    assert query["rejected_buffer"]["rejected_update_count"] == 4
    assert query["rejected_buffer"]["promoted_rejected_update_count"] == 0
    assert query["rejected_buffer"]["buffer_hash_root"].startswith("sha256:")


def test_scenario_learn_5858_promotion_metrics_controls_and_state(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-LEARN-5858-PROMOTION-ROLLBACK/METRICS-CONTROLS: credit is gated."""

    promotion = artifact["versioned_consolidation_and_promotion"]
    metrics = artifact["prospective_and_query_efficiency_metrics"]
    transfer = artifact["forward_transfer_recurrence_and_retention"]
    bounds = artifact["hard_case_and_family_lower_bounds"]
    state = artifact["rollback_restart_and_state_hashes"]
    cap = artifact["memory_cap_accounting"]
    controls = artifact["null_and_ablation_controls"]

    assert promotion["heldout_exact_validation_owns_promotion"] is True
    assert promotion["versioned_consolidation_enabled"] is True
    assert promotion["promoted_memory_count"] == 6
    assert promotion["qualified_replay_source"] == "Exp5857"
    assert promotion["sample_promotion_receipts"][0]["validation_authority"] == "exact_validator"

    assert metrics["arm_metrics"]["reduced_oracle"]["accuracy"] > metrics["arm_metrics"][
        "frozen"
    ]["accuracy"]
    assert metrics["arm_metrics"]["reduced_oracle"]["accuracy"] > metrics["arm_metrics"][
        "random_query"
    ]["accuracy"]
    assert metrics["reduced_minus_frozen"]["ci95"][0] > 0.0
    assert metrics["reduced_minus_random_query"]["ci95"][0] > 0.0
    assert metrics["full_oracle_lift_retained_fraction"] >= mod.FULL_ORACLE_LIFT_FRACTION_MIN
    assert metrics["reduced_query_fraction_of_full"] <= mod.REDUCED_QUERY_FRACTION_MAX
    assert metrics["reduced_lift_per_query"] > metrics["full_oracle_lift_per_query"]

    assert transfer["protected_prefix_retention"]["reduced_oracle"] == pytest.approx(1.0)
    assert transfer["recurrence"]["reduced_minus_random_query"]["ci95"][0] > 0.0
    assert transfer["forward_transfer"]["reduced_minus_random_query"]["ci95"][0] > 0.0
    assert transfer["no_retention_regression"] is True

    assert bounds["no_hard_case_regression"] is True
    assert bounds["no_family_regression"] is True
    assert bounds["all_family_lcbs_non_negative"] is True
    assert bounds["aggregate_family_lcb_positive"] is True
    assert bounds["hardness_summaries"]["hard"]["reduced_minus_random_query"]["ci95"][0] >= 0.0
    assert artifact["unsafe_accept_count"] == 0

    assert state["rollback_hash_mismatch_count"] == 0
    assert state["restart_equivalence"] == pytest.approx(1.0)
    assert state["full_state_hash"] == state["resumed_state_hash"]
    assert state["row_receipt_hash_root"] == artifact["row_file_receipt"]["row_receipt_hash_root"]
    assert cap["cap_compliance"] is True
    assert cap["max_state_size"] <= mod.MEMORY_CAP
    assert cap["max_cap_pressure"] <= 1.0

    assert controls["all_controls_fail_closed"] is True
    for name in (
        "query_order_permutation",
        "random_query",
        "always_query",
        "never_query",
        "shuffled_label_rejection",
        "selector_feature_ablation",
        "memory_reset",
        "duplicate_group",
    ):
        assert controls[name]["ready_score"] == pytest.approx(0.0)
        assert controls[name]["control_passed"] is True
    assert artifact["retirement_decision"]["decision"] == "advance_to_exp5859"


def test_scenario_learn_5858_fail_closed_for_bad_gates_and_tampering(
    artifact: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-LEARN-5858-FAIL-CLOSED: unsafe, null, or leaky evidence cannot promote."""

    blocked = mod.run(
        root=tmp_path,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        row_path=tmp_path / mod.ROW_RELATIVE_PATH.name,
        checkpoint_path=tmp_path / mod.CHECKPOINT_RELATIVE_PATH.name,
        duration_s=2.25,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=False,
    )
    assert blocked["status"] == "blocked"
    assert blocked["continuous_self_learning_ready_score"] == 0.0
    assert blocked["honest_verdict"].startswith("blocked:")
    assert "missing_upstream_file" in blocked["preconditions_checked"]["blocked_reasons"]
    assert mod.validate_artifact(blocked) is True

    failed_tests = mod.run(
        result_path=tmp_path / "failed.json",
        row_path=tmp_path / "failed.rows.jsonl",
        checkpoint_path=tmp_path / "failed.checkpoint.json",
        preconditions_checked=_preconditions(tmp_path / "failed"),
        duration_s=2.25,
        test_commands=TEST_COMMANDS,
        test_exit_codes={**TEST_EXIT_CODES, FULL_TEST_COMMAND: 2},
        write=False,
    )
    assert failed_tests["status"] == "null"
    assert failed_tests["continuous_self_learning_ready_score"] == 0.0
    assert "failed_test_exit_codes" in mod.blocked_reasons(failed_tests)

    unsafe = deepcopy(artifact)
    unsafe["unsafe_accept_count"] = 1
    unsafe["reproducibility_checksum"] = mod.reproducibility_checksum(unsafe)
    assert mod._artifact_status(unsafe) == "unsafe"
    assert mod.honest_verdict(unsafe).startswith("unsafe:")

    tampered = deepcopy(artifact)
    tampered["no_model_weight_mutation"] = False
    assert "no_model_weight_mutation" in mod.blocked_reasons(tampered)

    combined = deepcopy(artifact)
    combined["continuous_self_learning_task"] = False
    combined["inference_substrate"] = "wrong"
    combined["verifier_is_oracle"] = False
    combined["frozen_protocol_and_query_budgets"]["selector_policy"][
        "uses_future_labels"
    ] = True
    combined["unsafe_accept_count"] = 1
    combined["memory_cap_accounting"]["cap_compliance"] = False
    combined["rollback_restart_and_state_hashes"]["restart_equivalence"] = 0.0
    combined["no_model_weight_mutation"] = False
    combined["test_exit_codes"] = {**TEST_EXIT_CODES, FULL_TEST_COMMAND: 2}
    assert set(mod.blocked_reasons(combined)) >= {
        "continuous_self_learning_task",
        "inference_substrate",
        "verifier_is_oracle",
        "selector_policy",
        "unsafe_accept_count",
        "cap_compliance",
        "restart_equivalence",
        "no_model_weight_mutation",
        "failed_test_exit_codes",
    }

    with monkeypatch.context() as scoped:
        scoped.setattr(mod, "continuous_self_learning_ready_score", lambda item: 0.0)
        assert mod.blocked_reasons(artifact) == ["ready_score"]

    for mutate, match in (
        (
            lambda item: item.update({"continuous_self_learning_task": False}),
            "continuous_self_learning_task",
        ),
        (
            lambda item: item["frozen_protocol_and_query_budgets"]["selector_policy"].update(
                {"uses_future_labels": True}
            ),
            "selector_policy",
        ),
        (
            lambda item: item["chronology_and_visibility_receipts"].update(
                {"future_labels_visible_before_event_count": 1}
            ),
            "ready_score",
        ),
        (
            lambda item: item.update({"unsafe_accept_count": 1}),
            "ready_score",
        ),
        (
            lambda item: item["rollback_restart_and_state_hashes"].update(
                {"restart_equivalence": 0.0}
            ),
            "ready_score",
        ),
        (
            lambda item: item["memory_cap_accounting"].update({"cap_compliance": False}),
            "ready_score",
        ),
        (
            lambda item: item.update({"no_model_weight_mutation": False}),
            "ready_score",
        ),
        (
            lambda item: item.update({"inference_substrate": "live_llm"}),
            "inference_substrate",
        ),
        (
            lambda item: item.update({"verifier_is_oracle": False}),
            "verifier_is_oracle",
        ),
        (
            lambda item: item.update({"reproducibility_checksum": mod.sha256_text("wrong")}),
            "reproducibility_checksum",
        ),
    ):
        bad = deepcopy(artifact)
        mutate(bad)
        with pytest.raises(ValueError, match=match):
            mod.validate_artifact(bad)

    missing = deepcopy(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        mod.validate_artifact(missing)

    bad_provenance = deepcopy(artifact)
    bad_provenance["field_provenance"]["status"]["principle"] = "wrong"
    with pytest.raises(ValueError, match="field_provenance:status"):
        mod.validate_artifact(bad_provenance)

    provenance_not_mapping = deepcopy(artifact)
    provenance_not_mapping["field_provenance"] = []
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(provenance_not_mapping)

    invalid_status = deepcopy(artifact)
    invalid_status["status"] = "blocked"
    invalid_status["reproducibility_checksum"] = mod.reproducibility_checksum(invalid_status)
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(invalid_status)

    invalid_verdict = deepcopy(artifact)
    invalid_verdict["honest_verdict"] = "ready: wrong"
    invalid_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(invalid_verdict)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(invalid_verdict)

    with monkeypatch.context() as scoped:
        scoped.setattr(
            mod,
            "_read_json",
            lambda path: (_ for _ in ()).throw(ValueError("corrupt")),
        )
        corrupt = mod.collect_preconditions(
            result_path=tmp_path / "corrupt.json",
            row_path=tmp_path / "corrupt.rows.jsonl",
            checkpoint_path=tmp_path / "corrupt.checkpoint.json",
            memory_probe=lambda: {
                "available_mb": 0,
                "required_mb": mod.RAM_FLOOR_MB,
                "ok": False,
            },
            disk_probe=lambda root: {
                "available_mb": 0,
                "required_mb": mod.DISK_FLOOR_MB,
                "ok": False,
            },
        )
    assert set(corrupt["blocked_reasons"]) >= {
        "corrupt_upstream_json",
        "insufficient_free_ram",
        "insufficient_free_disk",
    }


def test_req_learn_5858_low_level_helpers_are_deterministic(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-5858: helper edges remain deterministic and auditable."""

    scalar_json = tmp_path / "scalar.json"
    scalar_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        mod._read_json(scalar_json)
    bad_jsonl = tmp_path / "bad.rows.jsonl"
    bad_jsonl.write_text("\n[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSONL object required"):
        mod._read_jsonl(bad_jsonl)

    assert mod.load_prospective_events(tmp_path / "missing") == []
    assert mod.read_result_rows(tmp_path / "missing.rows.jsonl") == []
    assert mod.fixture_preconditions(tmp_path)["preconditions_ready"] is True
    assert mod._paired_summary([])["mean_delta"] == 0.0
    assert mod._bootstrap_ci95([]) == [0.0, 0.0]
    assert mod._bootstrap_ci95([0.25]) == [0.25, 0.25]
    assert mod._round(0.123456789) == 0.123457
    assert mod._random_query_indexes(0, 1) == set()

    rows = mod.load_prospective_events(REPO)
    small_eval = mod._evaluate(rows[:3])
    assert mod._subset_deltas(
        small_eval["row_receipts"],
        "reduced_oracle",
        "frozen",
        key="change",
        value="addition",
    )
    signature = mod.event_signature(rows[0])
    assert signature["signature_hash"].startswith("sha256:")
    state = mod.initial_reduced_state()
    decision = mod.reduced_oracle_query_decision(rows[0], state)
    assert decision["query_selected"] is True
    assert decision["selector_feature_hash"].startswith("sha256:")
    assert mod.selector_policy_is_valid(mod._selector_policy()) is True
    bad_policy = mod._selector_policy()
    bad_policy["uses_future_labels"] = True
    assert mod.selector_policy_is_valid(bad_policy) is False
    bad_policy = mod._selector_policy()
    bad_policy["policy_frozen_before_science"] = False
    assert mod.selector_policy_is_valid(bad_policy) is False

    no_write = mod.run(
        result_path=tmp_path / "no-write.json",
        row_path=tmp_path / "no-write.rows.jsonl",
        checkpoint_path=tmp_path / "no-write.checkpoint.json",
        preconditions_checked=_preconditions(tmp_path / "no-write"),
        duration_s=2.25,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=False,
    )
    assert no_write["status"] == "ready"
    assert not (tmp_path / "no-write.json").exists()
    assert not (tmp_path / "no-write.rows.jsonl").exists()
