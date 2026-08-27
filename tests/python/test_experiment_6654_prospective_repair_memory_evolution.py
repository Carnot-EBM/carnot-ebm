"""Tests for prospective exact repair-memory evolution.

Spec refs: REQ-LEARN-6654, REQ-LEARN-6654-PRECONDITIONS,
REQ-LEARN-6654-PREQUENTIAL, REQ-LEARN-6654-MATCHED,
REQ-LEARN-6654-INFLUENCE, REQ-LEARN-6654-PATCHES,
REQ-LEARN-6654-SUPPORT, REQ-LEARN-6654-RECOVERY,
REQ-LEARN-6654-ROWS, REQ-LEARN-6654-ATOMIC,
SCENARIO-LEARN-6654-PREQUENTIAL, SCENARIO-LEARN-6654-MATCHED-ARMS,
SCENARIO-LEARN-6654-INFLUENCE, SCENARIO-LEARN-6654-PATCH-GATES,
SCENARIO-LEARN-6654-FORGETTING-SUPPORT,
SCENARIO-LEARN-6654-RECOVERY, SCENARIO-LEARN-6654-VERDICT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6654_prospective_repair_memory_evolution as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH
PASSING_TESTS = [
    {"command": command, "exit_code": 0, "summary": "passed", "gating": True}
    for command in mod.DEFAULT_TEST_COMMANDS
]


@pytest.fixture(scope="module")
def fixture_artifact() -> dict[str, object]:
    return mod.read_json(REPO / mod.UPSTREAM_RELATIVE_PATH)


@pytest.fixture(scope="module")
def comparison(fixture_artifact: dict[str, object]) -> dict[str, object]:
    preregistration = mod.build_preregistration(fixture_artifact)
    return mod.run_comparison(fixture_artifact, preregistration)


def _artifact(tmp_path: Path, *, write: bool = False) -> dict[str, object]:
    return mod.build_artifact(
        repo_root=REPO,
        output_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        date="20260826",
        duration_s=1.0,
        tests_run=PASSING_TESTS,
        write=write,
    )


def test_req_learn_6654_spec_declares_the_prospective_contract() -> None:
    """REQ-LEARN-6654: OpenSpec owns all prospective memory rules."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("## REQ-LEARN-6654") :]
    for marker in (
        "REQ-LEARN-6654-PRECONDITIONS",
        "REQ-LEARN-6654-PREQUENTIAL",
        "REQ-LEARN-6654-MATCHED",
        "REQ-LEARN-6654-INFLUENCE",
        "REQ-LEARN-6654-PATCHES",
        "REQ-LEARN-6654-SUPPORT",
        "REQ-LEARN-6654-RECOVERY",
        "SCENARIO-LEARN-6654-PREQUENTIAL",
        "SCENARIO-LEARN-6654-MATCHED-ARMS",
        "SCENARIO-LEARN-6654-INFLUENCE",
        "SCENARIO-LEARN-6654-PATCH-GATES",
        "SCENARIO-LEARN-6654-FORGETTING-SUPPORT",
        "SCENARIO-LEARN-6654-RECOVERY",
        "SCENARIO-LEARN-6654-VERDICT",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section


def test_scenario_6654_preconditions_and_preregistration_freeze_before_actions(
    fixture_artifact: dict[str, object],
) -> None:
    """REQ-LEARN-6654-PRECONDITIONS: inputs and policy choices freeze first."""

    gate = mod.upstream_gate_receipt(REPO)
    preregistration = mod.build_preregistration(fixture_artifact)

    assert gate["field"] == "memory_fixture_ready"
    assert gate["value"] is True
    assert gate["passed"] is True
    assert gate["path"] == mod.UPSTREAM_RELATIVE_PATH.as_posix()
    assert str(gate["sha256"]).startswith("sha256:")
    assert len(preregistration["orders"]) >= 3
    assert preregistration["arms"] == list(mod.ARMS)
    assert preregistration["thresholds"] == mod.PATCH_THRESHOLDS
    assert preregistration["support_floor"] == mod.SUPPORT_FLOOR
    assert preregistration["tie_rules"] == mod.TIE_RULES
    assert preregistration["rollback_policy"] == mod.ROLLBACK_POLICY
    assert preregistration["frozen_before_first_action"] is True
    order_sets = [set(row["event_ids"]) for row in preregistration["orders"]]
    assert all(len(row) == mod.EVENTS_PER_ORDER for row in order_sets)
    assert all(row == order_sets[0] for row in order_sets[1:])
    assert len({row["order_sha256"] for row in preregistration["orders"]}) == 3


def test_scenario_6654_candidate_pools_are_exact_matched_and_label_blind(
    fixture_artifact: dict[str, object],
) -> None:
    """REQ-LEARN-6654-MATCHED: selection sees one identical candidate pool."""

    event = mod.evaluation_events(fixture_artifact)[0]
    pool = mod.candidate_pool_for_event(event)
    preregistration = mod.build_preregistration(fixture_artifact)
    frozen = mod.rank_candidates(event, "frozen", preregistration, None)
    context = mod.rank_candidates(event, "context_only", preregistration, None)

    assert len(pool["candidates"]) == len(mod.CANDIDATE_OPERATORS)
    assert pool["pool_sha256"] == mod.sha256_json(pool["candidates"])
    assert frozen["candidate_pool_sha256"] == context["candidate_pool_sha256"]
    assert frozen["memory_retrieved"] is False
    assert context["memory_retrieved"] is False
    assert set(frozen["information_fields"]).isdisjoint(mod.FORBIDDEN_SELECTION_FIELDS)
    assert set(context["information_fields"]).isdisjoint(mod.FORBIDDEN_SELECTION_FIELDS)
    assert "exact_outcome" not in mod.canonical_json(frozen["ranking_basis"])
    assert sorted(row["rank"] for row in frozen["candidate_ranking"]) == list(
        range(1, len(mod.CANDIDATE_OPERATORS) + 1)
    )


def test_scenario_6654_prequential_rows_commit_before_exact_outcomes(
    comparison: dict[str, object],
) -> None:
    """SCENARIO-LEARN-6654-PREQUENTIAL: no same-event evidence selects."""

    rows = comparison["arm_order_event_rows"]
    assert len(rows) == len(mod.ORDER_IDS) * len(mod.ARMS) * mod.EVENTS_PER_ORDER
    assert all(row["action_committed_before_exact_outcome"] for row in rows)
    assert all(row["same_event_pending_write_visible"] is False for row in rows)
    assert all(row["visible_commit_max_index"] < row["event_index"] for row in rows)
    assert all(row["exact_outcome"] in (0, 1) for row in rows)
    assert all(len(row["candidate_exact_outcomes"]) == len(mod.CANDIDATE_OPERATORS) for row in rows)
    assert all(row["row_sha256"] == mod.row_hash(row) for row in rows)


def test_scenario_6654_matched_arms_reset_and_receive_every_opportunity(
    comparison: dict[str, object],
) -> None:
    """SCENARIO-LEARN-6654-MATCHED-ARMS: orders, pools, and resets match."""

    rows = comparison["arm_order_event_rows"]
    for order_id in mod.ORDER_IDS:
        by_arm = {
            arm: [row for row in rows if row["order_id"] == order_id and row["arm"] == arm]
            for arm in mod.ARMS
        }
        event_sequences = [[row["event_id"] for row in by_arm[arm]] for arm in mod.ARMS]
        pool_sequences = [[row["candidate_pool_sha256"] for row in by_arm[arm]] for arm in mod.ARMS]
        assert all(sequence == event_sequences[0] for sequence in event_sequences[1:])
        assert all(sequence == pool_sequences[0] for sequence in pool_sequences[1:])
        assert all(
            by_arm[arm][0]["pre_memory_checksum"] == mod.EMPTY_MEMORY_CHECKSUM for arm in mod.ARMS
        )


def test_scenario_6654_retrieval_credit_requires_live_action_influence(
    fixture_artifact: dict[str, object], comparison: dict[str, object]
) -> None:
    """SCENARIO-LEARN-6654-INFLUENCE: stored but inert items earn no credit."""

    receipts = comparison["retrieval_and_influence_rows"]
    assert any(row["retrieved"] for row in receipts)
    assert any(row["credited"] for row in receipts)
    assert all(row["credited"] is (row["retrieved"] and row["action_changed"]) for row in receipts)

    event = mod.evaluation_events(fixture_artifact)[0]
    preregistration = mod.build_preregistration(fixture_artifact)
    baseline = mod.rank_candidates(event, "context_only", preregistration, None)
    inert_memory = {
        "operator": baseline["selected_operator"],
        "component_type": "test_component",
        "version": 1,
        "support_event_ids": ["prior"],
    }
    influenced = mod.rank_candidates(event, "verified_memory", preregistration, inert_memory)
    assert influenced["memory_retrieved"] is True
    assert influenced["action_changed"] is False
    assert influenced["credited"] is False


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"source_repair": False}, "source_repair_failed"),
        ({"anchor_after": 0, "anchor_before": 1}, "held_anchor_regression"),
        ({"support_after": 0.5}, "recoverable_support_below_floor"),
        ({"targeted_component_count": 2}, "patch_not_localized"),
        ({"operation": "replace_all"}, "patch_operation_not_allowed"),
    ],
)
def test_scenario_6654_patch_gate_rejects_each_unsafe_case(
    overrides: dict[str, object], reason: str
) -> None:
    """SCENARIO-LEARN-6654-PATCH-GATES: each failed gate rejects."""

    inputs: dict[str, object] = {
        "operation": "append",
        "source_repair": True,
        "anchor_before": 1,
        "anchor_after": 1,
        "support_after": 1.0,
        "targeted_component_count": 1,
    }
    inputs.update(overrides)
    decision = mod.patch_gate_decision(**inputs)

    assert decision["admitted"] is False
    assert reason in decision["rejection_reasons"]


def test_scenario_6654_accepted_patches_are_local_and_support_complete(
    comparison: dict[str, object],
) -> None:
    """REQ-LEARN-6654-PATCHES: all accepted patches carry exact gates."""

    patches = comparison["patch_decision_rows"]
    support = comparison["recoverable_support_rows"]
    accepted = [row for row in patches if row["decision"] in {"admit", "retire"}]

    assert accepted
    assert len(support) == len(accepted)
    assert all(row["targeted_component_count"] == 1 for row in patches)
    assert all(row["operation"] in mod.ALLOWED_PATCH_OPERATIONS for row in patches)
    assert all(row["source_repair"]["exact_outcome"] == 1 for row in accepted)
    assert all(row["held_anchor_check"]["regression_count"] == 0 for row in accepted)
    assert all(row["support_check"]["after"] >= mod.SUPPORT_FLOOR for row in accepted)
    assert all(row["before"] >= mod.SUPPORT_FLOOR for row in support)
    assert all(row["after"] >= mod.SUPPORT_FLOOR for row in support)
    assert all(row["fixed_candidate_budget"] == len(mod.CANDIDATE_OPERATORS) for row in support)


def test_scenario_6654_restart_and_rollback_receipts_preserve_state(
    comparison: dict[str, object],
) -> None:
    """SCENARIO-LEARN-6654-RECOVERY: canonical state survives recovery."""

    receipts = comparison["memory_state_receipts"]
    checkpoints = [row for row in receipts if row["receipt_type"] == "restart_checkpoint"]
    assert checkpoints
    assert all(row["restart_equal"] for row in checkpoints)
    assert all(row["checkpoint_checksum"] == row["restart_checksum"] for row in checkpoints)
    assert all("rollback_target_checksum" in row for row in receipts)
    assert all(row["rollback_applied"] is False for row in receipts)

    state = {"version": 2, "items": {"key": {"operator": "repair"}}}
    checkpoint = mod.checkpoint_state(state, lineage="unit-test")
    changed = {"version": 3, "items": {}}
    restored, receipt = mod.rollback_to_checkpoint(changed, checkpoint, reason="forced_gate")
    assert restored == state
    assert receipt["rollback_applied"] is True
    assert receipt["restored_equal"] is True


def test_scenario_6654_metrics_recompute_future_gain_without_forgetting(
    comparison: dict[str, object],
) -> None:
    """SCENARIO-LEARN-6654-FORGETTING-SUPPORT: rows own every metric."""

    rows = comparison["arm_order_event_rows"]
    retrievals = comparison["retrieval_and_influence_rows"]
    patches = comparison["patch_decision_rows"]
    support = comparison["recoverable_support_rows"]
    recomputed = mod.recompute_metrics(rows, retrievals, patches, support)

    assert recomputed == comparison["prospective_metrics"]
    assert recomputed["future_event_delta"]["verified_memory_minus_context_only"] > 0.0
    assert all(
        row["verified_memory_minus_context_only"] > 0.0
        for row in recomputed["order_sensitivity"]["per_order"]
    )
    assert recomputed["forgetting"]["count"] == 0
    assert recomputed["recoverable_support"]["minimum_after"] >= mod.SUPPORT_FLOOR
    assert recomputed["retrieval"]["credited_count"] > 0
    assert recomputed["uncertainty"]["method"] == "wilson_yield_and_order_delta_t_interval"


def test_req_6654_aggregate_rows_and_named_attacks_fail_closed(
    comparison: dict[str, object],
) -> None:
    """REQ-LEARN-6654-ROWS: row reducers and adversarial checks are independent."""

    aggregate = mod.aggregate_row_recomputation(
        comparison["arm_order_event_rows"],
        comparison["retrieval_and_influence_rows"],
        comparison["patch_decision_rows"],
        comparison["recoverable_support_rows"],
        comparison["prospective_metrics"],
    )
    attacks = mod.build_attack_rows(comparison)

    assert aggregate["all_recomputations_match"] is True
    assert len(aggregate["arm_order_rows"]) == len(mod.ORDER_IDS) * len(mod.ARMS)
    assert {row["attack_type"] for row in attacks} == set(mod.ATTACK_TYPES)
    assert all(row["detected"] and row["failed_closed"] for row in attacks)


def test_scenario_6654_verdict_is_positive_null_or_named_block() -> None:
    """SCENARIO-LEARN-6654-VERDICT: completion stays independent of sign."""

    complete_gates = [
        {"check": "comparison_complete", "expected": True, "observed": True, "passed": True},
        {"check": "safety", "expected": True, "observed": True, "passed": True},
    ]
    positive = mod.terminal_fields(complete_gates, future_delta=0.1)
    null = mod.terminal_fields(complete_gates, future_delta=0.0)
    blocked = mod.terminal_fields(
        [
            complete_gates[0],
            {"check": "support", "expected": True, "observed": False, "passed": False},
        ],
        future_delta=0.1,
    )

    assert positive["status"] == "complete_positive"
    assert positive["verdict_class"] == "positive"
    assert positive["prospective_memory_comparison_complete"] is True
    assert null["status"] == "complete_null"
    assert null["verdict_class"] is None
    assert null["prospective_memory_comparison_complete"] is True
    assert blocked["status"] == "blocked_support"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["prospective_memory_comparison_complete"] is False
    assert blocked["gate_check_summary"]["observed_value"] is False


def test_req_6654_artifact_is_atomic_complete_and_row_derived(tmp_path: Path) -> None:
    """REQ-LEARN-6654-ATOMIC: one checksummed artifact carries all evidence."""

    artifact = _artifact(tmp_path, write=True)
    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    written = json.loads(output.read_text(encoding="utf-8"))

    assert written == artifact
    assert mod.validate_artifact(artifact) == []
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete_positive"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["verdict_class"] == "positive"
    assert artifact["prospective_memory_comparison_complete"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["upstream_gate_receipt"]["passed"] is True
    assert artifact["aggregate_row_recomputation"]["all_recomputations_match"] is True
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"])
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
    assert not output.with_suffix(output.suffix + ".tmp").exists()


def test_req_6654_validator_rejects_artifact_tampering(tmp_path: Path) -> None:
    """REQ-LEARN-6654: schema, rows, authority, and checksum drift stay visible."""

    artifact = _artifact(tmp_path)
    mutations = (
        ("missing_required_fields", lambda value: value.pop("status")),
        ("event_row_count_mismatch", lambda value: value["arm_order_event_rows"].pop()),
        ("retrieval_row_count_mismatch", lambda value: value["retrieval_and_influence_rows"].pop()),
        ("per_unit_count_mismatch", lambda value: value["per_unit_rows"].pop()),
        (
            "comparison_complete_mismatch",
            lambda value: value.update(prospective_memory_comparison_complete=False),
        ),
        ("verdict_class_mismatch", lambda value: value.update(verdict_class=None)),
        (
            "upstream_gate_mismatch",
            lambda value: value["upstream_gate_receipt"].update(passed=False),
        ),
        ("inference_substrate_mismatch", lambda value: value.update(inference_substrate="llm")),
        ("oracle_boundary_mismatch", lambda value: value.update(verifier_is_oracle=True)),
        (
            "protected_files_changed",
            lambda value: value["protected_files_unchanged"].update(unchanged=False),
        ),
        ("test_command_failed", lambda value: value["tests_run"][0].update(exit_code=1)),
        (
            "aggregate_recomputation_mismatch",
            lambda value: value["aggregate_row_recomputation"].update(
                all_recomputations_match=False
            ),
        ),
        ("field_provenance_missing", lambda value: value["field_provenance"].pop("status")),
        ("checksum_mismatch", lambda value: value.update(reproducibility_checksum="sha256:bad")),
    )
    for expected, mutate in mutations:
        changed = deepcopy(artifact)
        mutate(changed)
        assert expected in mod.validate_artifact(changed)


def test_req_6654_helpers_and_cli_cover_failure_and_validation_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-6654: input, atomic output, and CLI checks fail closed."""

    assert mod.sha256_file(tmp_path / "missing") is None
    bad = tmp_path / "bad.json"
    bad.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        mod.read_json(bad)

    fixture = mod.read_json(REPO / mod.UPSTREAM_RELATIVE_PATH)
    short_fixture = deepcopy(fixture)
    short_fixture["event_rows"].pop(
        next(
            index
            for index, row in enumerate(short_fixture["event_rows"])
            if row["partition"] in mod.EVALUATION_PARTITIONS
        )
    )
    with pytest.raises(ValueError, match="evaluation_event_count_mismatch"):
        mod.evaluation_events(short_fixture)
    event = mod.evaluation_events(fixture)[0]
    preregistration = mod.build_preregistration(fixture)
    with pytest.raises(ValueError, match="unknown_arm"):
        mod.rank_candidates(event, "unknown", preregistration, None)
    assert mod._wilson(0, 0) is None
    assert mod._order_delta_interval([0.25]) == [0.25, 0.25]
    with pytest.raises(ValueError, match="operator_missing"):
        mod.exact_outcome_from_candidates([], "missing")

    with monkeypatch.context() as scoped:
        scoped.setattr(mod, "_post_commit_safe", lambda _state, _support: False)
        state, patch, support, receipt = mod._propose_patch(
            mod.empty_memory_state(),
            event,
            fixture,
            preregistration,
            order_id="chronological",
            event_index=0,
        )
        assert state == mod.empty_memory_state()
        assert patch["decision"] == "reject"
        assert patch["rollback_applied"] is True
        assert support is None
        assert receipt["rollback_applied"] is True

    with monkeypatch.context() as scoped:
        scoped.setattr(mod, "upstream_gate_receipt", lambda _root: {"passed": False})
        with pytest.raises(ValueError, match="upstream_memory_fixture_not_ready"):
            mod.build_artifact(
                repo_root=REPO,
                output_path=tmp_path / "blocked.json",
                date="20260826",
                duration_s=1.0,
                tests_run=PASSING_TESTS,
                write=False,
            )

    with monkeypatch.context() as scoped:
        scoped.setattr(mod, "validate_artifact", lambda _artifact: ["forced_error"])
        with pytest.raises(ValueError, match="forced_error"):
            mod.build_artifact(
                repo_root=REPO,
                output_path=tmp_path / "forced.json",
                date="20260826",
                duration_s=1.0,
                tests_run=PASSING_TESTS,
                write=False,
            )

    output = tmp_path / "result.json"
    assert mod.main(["--date", "20260826", "--output", str(output), "--duration-s", "1.0"]) == 0
    assert str(output) in capsys.readouterr().out
    assert mod.main(["--validate", "--output", str(output)]) == 0
    assert mod.main(["--check-rows", "--output", str(output)]) == 0

    row_broken = json.loads(output.read_text(encoding="utf-8"))
    row_broken["aggregate_row_recomputation"]["all_recomputations_match"] = False
    output.write_text(json.dumps(row_broken), encoding="utf-8")
    with pytest.raises(ValueError, match="aggregate_recomputation_mismatch"):
        mod.main(["--check-rows", "--output", str(output)])
    mod.atomic_write_json(output, _artifact(tmp_path))

    measured = tmp_path / "measured.json"
    assert mod.main(["--output", str(measured)]) == 0
    payload = json.loads(measured.read_text(encoding="utf-8"))
    assert payload["duration_s"] >= 0.001
    assert payload["reproducibility_checksum"] == mod.reproducibility_checksum(payload)

    broken = json.loads(output.read_text(encoding="utf-8"))
    broken["reproducibility_checksum"] = "sha256:bad"
    output.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum_mismatch"):
        mod.main(["--validate", "--output", str(output)])


def test_e2e_6654_exact_prequential_memory_pipeline(tmp_path: Path) -> None:
    """REQ-LEARN-6654: E2E rebuilds, writes, reloads, and recomputes rows."""

    artifact = _artifact(tmp_path, write=True)
    reloaded = mod.read_json(tmp_path / mod.RESULT_RELATIVE_PATH.name)
    metrics = mod.recompute_metrics(
        reloaded["arm_order_event_rows"],
        reloaded["retrieval_and_influence_rows"],
        reloaded["patch_decision_rows"],
        reloaded["recoverable_support_rows"],
    )

    assert reloaded == artifact
    assert metrics == artifact["prospective_metrics"]
    assert mod.validate_artifact(reloaded) == []
