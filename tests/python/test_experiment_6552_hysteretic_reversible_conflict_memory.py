"""Tests for Exp6552 hysteretic reversible exact-conflict memory.

Spec refs: REQ-STORE-6552,
SCENARIO-STORE-6552-QUERY-FREEZE-ADMISSION,
SCENARIO-STORE-6552-HYSTERESIS-REACTIVATION,
SCENARIO-STORE-6552-CAPACITY-RESTART-ROLLBACK,
SCENARIO-STORE-6552-ATTACKS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6552_hysteretic_reversible_conflict_memory as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH

FOCUSED_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6552_hysteretic_reversible_conflict_memory.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6552_hysteretic_reversible_conflict_memory.py "
    "-m pytest tests/python/test_experiment_6552_hysteretic_reversible_conflict_memory.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6552_hysteretic_reversible_conflict_memory.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6552_hysteretic_reversible_conflict_memory.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6552_hysteretic_reversible_conflict_memory.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6552_hysteretic_reversible_conflict_memory.json"
)
PERSISTENCE_E2E_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6521_transactional_refinement_conflict_memory.py "
    "-q --no-cov -n 0"
)
PIPELINE_E2E_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6549_production_safety_net_adapter.py "
    "tests/python/test_production_safety_net_adapter.py -q --no-cov -n 0"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6552_hysteretic_reversible_conflict_memory "
    "--date 20260823"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6552_hysteretic_reversible_conflict_memory --validate"
)
GIT_STATUS_COMMAND = "git status --short"

TESTS_RUN = [
    {"command": FOCUSED_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
    {"command": SPEC_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": PERSISTENCE_E2E_COMMAND, "exit_code": 0},
    {"command": PIPELINE_E2E_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": GIT_STATUS_COMMAND, "exit_code": 0},
]


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """REQ-STORE-6552: build a temp artifact without touching tracked results."""

    root = tmp_path_factory.mktemp("exp6552")
    return mod.build_artifact(
        repo_root=REPO,
        result_path=root / mod.RESULT_RELATIVE_PATH.name,
        work_root=root / "work",
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )


def test_req_store_6552_spec_declares_reversible_controller_contract() -> None:
    """REQ-STORE-6552: OpenSpec owns the reversible controller contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-STORE-6552") : text.index("REQ-STORE-6522")]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-STORE-6552-QUERY-FREEZE-ADMISSION",
        "SCENARIO-STORE-6552-HYSTERESIS-REACTIVATION",
        "SCENARIO-STORE-6552-CAPACITY-RESTART-ROLLBACK",
        "SCENARIO-STORE-6552-ATTACKS",
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "reversible_memory_controller_ready_score",
        "active",
        "dormant",
        "retired",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_store_6552_query_freeze_and_exact_admission(tmp_path: Path) -> None:
    """SCENARIO-STORE-6552-QUERY-FREEZE-ADMISSION: writes follow exact replay."""

    stream = mod.build_event_stream(seed=655201)
    assert stream[0].to_dict()["event_id"] == "alpha_initial_support"
    thresholds = mod.freeze_thresholds(stream)
    with pytest.raises(ValueError, match="unknown controller arm"):
        mod.ReversibleConflictController(
            arm_id="bad_arm",
            capacity=2,
            thresholds=thresholds,
            persistence_dir=tmp_path / "bad",
            seed=655201,
        )
    controller = mod.ReversibleConflictController(
        arm_id="hysteretic_control",
        capacity=2,
        thresholds=thresholds,
        persistence_dir=tmp_path / "hysteretic",
        seed=655201,
    )
    events = {event.event_id: event for event in mod.build_event_stream(seed=655201)}

    first = controller.process_event(events["alpha_initial_support"])
    assert first["frozen_query_snapshot_hash"] == first["pre_memory_hash"]
    assert first["decision_time_write_count"] == 0
    assert first["action"] == "commit_after_query"
    assert first["exact_receipt"]["exact_replay_valid"] is True
    assert first["post_state"]["alpha"] == "active"
    assert first["unsafe_write"] is False
    assert first["unsafe_use"] is False
    assert first["same_query_write_attempted"] is False

    invalid = controller.process_event(events["invalid_refinement_attempt"])
    assert invalid["action"] == "veto_invalid_refinement"
    assert invalid["exact_receipt"]["exact_replay_valid"] is False
    assert invalid["durable_write_performed"] is False
    assert invalid["unsafe_write"] is False
    assert invalid["post_memory_hash"] == invalid["pre_memory_hash"]

    checkpoint = controller.checkpoint("manual")
    second = controller.process_event(events["beta_initial_support"])
    rollback = controller.rollback("manual")
    assert second["post_memory_hash"] != checkpoint["state_hash"]
    assert rollback["rolled_back"] is True
    assert rollback["state_hash_after"] == checkpoint["state_hash"]


def test_scenario_store_6552_hysteresis_reactivation_and_policy_gate(tmp_path: Path) -> None:
    """SCENARIO-STORE-6552-HYSTERESIS-REACTIVATION: dormant is reversible first."""

    thresholds = mod.freeze_thresholds(mod.build_event_stream(seed=655201))
    controller = mod.ReversibleConflictController(
        arm_id="hysteretic_control",
        capacity=2,
        thresholds=thresholds,
        persistence_dir=tmp_path / "hysteretic",
        seed=655201,
    )
    rows = [controller.process_event(event) for event in mod.build_event_stream(seed=655201)]

    dormant = next(row for row in rows if row["event_id"] == "alpha_stale_support")
    reactivated = next(row for row in rows if row["event_id"] == "alpha_regime_returns")
    blocked = next(row for row in rows if row["event_id"] == "beta_retire_without_policy")
    retired = next(row for row in rows if row["event_id"] == "beta_policy_retirement")

    assert dormant["action"] == "demote_to_dormant"
    assert dormant["post_state"]["alpha"] == "dormant"
    assert reactivated["action"] == "shadow_reactivate"
    assert reactivated["shadow_exact_replay"] is True
    assert reactivated["reactivation"] is True
    assert reactivated["post_state"]["alpha"] == "active"
    assert blocked["action"] == "block_retirement_without_policy"
    assert blocked["post_state"]["beta"] == "dormant"
    assert retired["action"] == "policy_retire"
    assert retired["policy_receipt"]["approved"] is True
    assert retired["post_state"]["beta"] == "retired"

    observe = mod.ReversibleConflictController(
        arm_id="hysteretic_control",
        capacity=2,
        thresholds=thresholds,
        persistence_dir=tmp_path / "observe",
        seed=655201,
    )
    events = {event.event_id: event for event in mod.build_event_stream(seed=655201)}
    observe.process_event(events["alpha_initial_support"])
    assert observe.process_event(events["alpha_initial_support"])["action"] == "hysteretic_observe"

    one_threshold = mod.ReversibleConflictController(
        arm_id="one_threshold",
        capacity=1,
        thresholds=thresholds,
        persistence_dir=tmp_path / "one-threshold",
        seed=655201,
    )
    one_threshold.process_event(events["alpha_initial_support"])
    assert one_threshold.process_event(events["beta_initial_support"])["eviction"] is True


def test_scenario_store_6552_capacity_restart_rollback_and_comparison(tmp_path: Path) -> None:
    """SCENARIO-STORE-6552-CAPACITY-RESTART-ROLLBACK: matched arms replay exactly."""

    comparison = mod.run_controller_comparison(
        persistence_dir=tmp_path / "comparison",
        seeds=mod.DEFAULT_SEEDS,
        capacity=2,
    )
    events = mod.build_event_stream(seed=mod.DEFAULT_SEEDS[0])
    expected_rows = len(events) * len(mod.DEFAULT_SEEDS) * len(mod.CONTROLLER_ARMS)

    assert len(comparison["per_unit_rows"]) == expected_rows
    assert {row["arm_id"] for row in comparison["per_unit_rows"]} == set(mod.CONTROLLER_ARMS)
    assert all(row["capacity"] == 2 for row in comparison["per_unit_rows"])
    assert all(row["record_count_after"] <= 2 for row in comparison["per_unit_rows"])
    assert all(
        receipt["byte_identical_decisions"] and receipt["byte_identical_memory_hashes"]
        for receipt in comparison["restart_and_rollback_receipts"]["restart_receipts"]
    )
    assert all(
        receipt["rolled_back"] and receipt["state_hash_after"] == receipt["target_state_hash"]
        for receipt in comparison["restart_and_rollback_receipts"]["rollback_receipts"]
    )
    assert comparison["aggregate_row_recomputation"]["unsafe_write_count"] == 0
    assert comparison["aggregate_row_recomputation"]["unsafe_use_count"] == 0
    assert comparison["aggregate_row_recomputation"]["ready_score"] == 1.0


def test_scenario_store_6552_artifact_schema_and_validation(
    tmp_path: Path,
    artifact: dict[str, Any],
) -> None:
    """REQ-STORE-6552: artifact fields are row-derived and validate."""

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete_reversible_memory_controller_ready_null"
    assert artifact["honest_verdict"].startswith("complete_null:")
    assert artifact["verdict_class"] == "null"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["reversible_memory_controller_ready_score"] == 1.0
    assert artifact["protected_files_unchanged"]["all_protected_files_unchanged"] is True
    assert set(artifact["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
    assert mod.validate_artifact(artifact) == []

    assert artifact["upstream_gate_receipt"]["gate_passed"] is True
    assert artifact["state_machine_and_threshold_contract"]["threshold_source_splits"] == [
        "train",
        "development",
    ]
    assert artifact["state_machine_and_threshold_contract"]["states"] == [
        "active",
        "dormant",
        "retired",
    ]
    assert artifact["exact_admission_and_refinement_receipts"]["unsafe_admission_count"] == 0
    assert artifact["unsafe_write_and_use_ledger"]["unsafe_write_count"] == 0
    assert artifact["unsafe_write_and_use_ledger"]["unsafe_use_count"] == 0
    assert artifact["attack_matrix"]["all_attacks_fail_closed"] is True
    assert artifact["aggregate_row_recomputation"] == mod.aggregate_row_recomputation(artifact)
    assert all(row["row_hash"] == mod.row_hash(row) for row in artifact["per_unit_rows"])
    assert all(row["row_hash"] == mod.row_hash(row) for row in artifact["transition_rows"])

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH.name
    work_root = tmp_path / "work"
    assert (
        mod.main(
            [
                "--date",
                "20260823",
                "--result-path",
                str(result_path),
                "--work-root",
                str(work_root),
            ]
        )
        == 0
    )
    assert mod.main(["--validate", "--result-path", str(result_path)]) == 0
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["reversible_memory_controller_ready_score"] == 1.0


def test_scenario_store_6552_attack_matrix_and_defensive_validation(
    artifact: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-STORE-6552-ATTACKS: attacks fail closed and validator catches drift."""

    attacks = {row["attack_id"]: row for row in artifact["attack_matrix"]["rows"]}
    assert set(attacks) == set(mod.ATTACK_IDS)
    assert all(row["fail_closed"] for row in attacks.values())
    assert attacks["retirement_without_policy"]["unsafe_write"] is False
    assert attacks["authority_inversion"]["verifier_overridden"] is False
    assert attacks["held_threshold_tuning"]["held_rows_used_for_thresholds"] == 0

    mutations = [
        ("required field set mismatch", lambda item: item.pop("status")),
        (
            "verdict_class must be closed",
            lambda item: item.__setitem__("verdict_class", "circular_positive"),
        ),
        (
            "positive verdict requires comparative benefit",
            lambda item: item.__setitem__("verdict_class", "positive"),
        ),
        (
            "inference_substrate mismatch",
            lambda item: item.__setitem__("inference_substrate", "live_llm"),
        ),
        (
            "verifier_is_oracle must be false",
            lambda item: item.__setitem__("verifier_is_oracle", True),
        ),
        (
            "unsafe write or use detected",
            lambda item: item["unsafe_write_and_use_ledger"].__setitem__("unsafe_write_count", 1),
        ),
        (
            "attacks did not fail closed",
            lambda item: item["attack_matrix"].__setitem__("all_attacks_fail_closed", False),
        ),
        (
            "ready score mismatch",
            lambda item: item.__setitem__("reversible_memory_controller_ready_score", 0.0),
        ),
        (
            "reversible_memory_controller_ready_score must be 0.0 or 1.0",
            lambda item: item.__setitem__("reversible_memory_controller_ready_score", 0.5),
        ),
        (
            "protected files changed",
            lambda item: item["protected_files_unchanged"].__setitem__(
                "all_protected_files_unchanged", False
            ),
        ),
        (
            "field_provenance must cover required fields",
            lambda item: item.__setitem__("field_provenance", {}),
        ),
        (
            "reproducibility_checksum mismatch",
            lambda item: item.__setitem__("reproducibility_checksum", "sha256:bad"),
        ),
        (
            "honest_verdict lacks terminal prefix",
            lambda item: item.__setitem__("honest_verdict", "running"),
        ),
    ]
    for expected, mutate in mutations:
        broken = deepcopy(artifact)
        mutate(broken)
        assert expected in mod.validate_artifact(broken)

    assert mod.sha256_file(tmp_path / "missing.json") == "missing"
    assert mod.upstream_gate_receipt(tmp_path)["gate_passed"] is False

    blocked = deepcopy(artifact)
    blocked["gate_check_summary"]["checks"]["upstream_gate_passed"] = False
    assert mod._status_and_verdict(blocked)[2] == "blocked"
    disqualified = deepcopy(artifact)
    disqualified["aggregate_row_recomputation"]["unsafe_write_count"] = 1
    assert mod._status_and_verdict(disqualified)[2] == "disqualified"
    partial = deepcopy(artifact)
    partial["aggregate_row_recomputation"]["ready_score"] = 0.0
    assert mod._status_and_verdict(partial)[2] == "partial"
    positive = deepcopy(artifact)
    positive["aggregate_row_recomputation"]["comparative_benefit_positive"] = True
    assert mod._status_and_verdict(positive)[2] == "positive"

    dated = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "dated.json",
        work_root=tmp_path / "dated-work",
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260824",
    )
    assert dated["preconditions_checked"]["requested_run_date"] == "20260824"

    monkeypatch.setattr(mod, "validate_artifact", lambda payload: ["forced validation error"])
    with pytest.raises(ValueError, match="forced validation error"):
        mod.build_artifact(
            repo_root=REPO,
            result_path=tmp_path / "bad-artifact.json",
            work_root=tmp_path / "bad-work",
            write=False,
            duration_s=1.0,
            tests_run=TESTS_RUN,
        )
    monkeypatch.undo()

    invalid = deepcopy(artifact)
    invalid["status"] = "running_bootstrap"
    invalid["reproducibility_checksum"] = mod.reproducibility_checksum(invalid)
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="status lacks terminal prefix"):
        mod.main(["--validate", "--result-path", str(invalid_path)])

    missing_path = tmp_path / "missing-result.json"
    with pytest.raises(ValueError, match="artifact not found"):
        mod.main(["--validate", "--result-path", str(missing_path)])
