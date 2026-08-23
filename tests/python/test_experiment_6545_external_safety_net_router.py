"""Tests for Exp6545 external Safety-Net router transfer.

Spec refs: REQ-BENCH-6545, SCENARIO-BENCH-6545-GATE,
SCENARIO-BENCH-6545-TRAIN-CAL, SCENARIO-BENCH-6545-ROUTERS,
SCENARIO-BENCH-6545-RUNTIME, SCENARIO-BENCH-6545-EFFECTS,
SCENARIO-BENCH-6545-ATTACKS, SCENARIO-BENCH-6545-ROLLBACK,
SCENARIO-BENCH-6545-TERMINAL.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6545_external_safety_net_router as mod
from scripts import adversarial_verify


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6545_external_safety_net_router.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6545_external_safety_net_router.py "
    "-m pytest tests/python/test_experiment_6545_external_safety_net_router.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6545_external_safety_net_router.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6545_external_safety_net_router.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6545_external_safety_net_router.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6545_external_safety_net_router.json"
)
EXACT_E2E_COMMAND = ".venv/bin/pytest tests/python/test_z3_live_benchmark.py -q --no-cov -n 0"
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6545_external_safety_net_router "
    "--date 20260823"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6545_external_safety_net_router --validate"
)
GIT_STATUS_COMMAND = "git status --short"

TESTS_RUN = [
    {"command": FOCUSED_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": SPEC_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": EXACT_E2E_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": GIT_STATUS_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
]


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """REQ-BENCH-6545: build a temp artifact from checked-in V566 evidence."""

    root = tmp_path_factory.mktemp("exp6545")
    return mod.build_artifact(
        repo_root=REPO,
        result_path=root / mod.RESULT_RELATIVE_PATH.name,
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )


def test_req_bench_6545_spec_declares_external_safety_net_contract() -> None:
    """REQ-BENCH-6545: OpenSpec owns the external Safety-Net contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-BENCH-6545") :]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-BENCH-6545-GATE",
        "SCENARIO-BENCH-6545-TRAIN-CAL",
        "SCENARIO-BENCH-6545-ROUTERS",
        "SCENARIO-BENCH-6545-RUNTIME",
        "SCENARIO-BENCH-6545-EFFECTS",
        "SCENARIO-BENCH-6545-ATTACKS",
        "SCENARIO-BENCH-6545-ROLLBACK",
        "SCENARIO-BENCH-6545-TERMINAL",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "external_safety_net_ready_score",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_bench_6545_gate_contract_and_models(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6545-GATE/ROUTERS: gates and router family are frozen."""

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete_external_safety_net_router_positive"
    assert artifact["honest_verdict"].startswith("complete_external_safety_net_router_positive")
    assert artifact["verdict_class"] == "positive"
    assert artifact["external_safety_net_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False

    gate = artifact["upstream_gate_receipt"]
    assert gate["path"] == mod.EXP6544_RELATIVE_PATH.as_posix()
    assert gate["field"] == "external_structural_headroom_ready_score"
    assert gate["expected_value"] == 1.0
    assert gate["observed_value"] == 1.0
    assert gate["gate_passed"] is True
    assert gate["sha256"].startswith("sha256:")
    assert gate["input_hashes"]["fixture"] == mod.sha256_file(REPO / mod.FIXTURE_RELATIVE_PATH)
    assert gate["input_hashes"]["exp6520"] == mod.sha256_file(REPO / mod.EXP6520_RELATIVE_PATH)
    assert gate["input_hashes"]["exp6527"] == mod.sha256_file(REPO / mod.EXP6527_RELATIVE_PATH)
    assert gate["cpu_identity"]["cpu_count"] >= 1
    assert "available" in gate["gpu_identity"]
    assert gate["seeds"] == list(mod.EVALUATION_SEEDS)
    assert gate["budgets"]["candidate_budget_rule"] == "all_candidates_preserved"
    assert "scripts/research_conductor.py" in gate["protected_file_hashes_before"]

    contract = artifact["frozen_router_contract"]
    assert contract["arm_ids"] == list(mod.ARM_IDS)
    assert contract["compact_model_families"] == list(mod.MODEL_FAMILIES)
    assert contract["certified_structural_control_source_arm"] == mod.EXP6544_CERTIFIED_ARM
    assert contract["selection_rule_frozen_before_held"] is True
    assert contract["calibration_rule_frozen_before_held"] is True
    assert contract["abstention_rule_frozen_before_held"] is True
    assert contract["exception_table_frozen_before_held"] is True
    assert contract["held_outcomes_used_before_freeze"] is False
    assert contract["forbidden_features"] == list(mod.FORBIDDEN_FEATURES)
    assert contract["candidate_budget_rule"] == "all_candidates_preserved"
    assert contract["native_exact_fallback_required"] is True

    models = artifact["candidate_model_rows"]
    assert [row["model_family"] for row in models] == list(mod.MODEL_FAMILIES)
    assert all(row["eligible"] is True for row in models)
    assert all(row["trained_split"] == "train" for row in models)
    assert all(row["held_rows_used_for_training"] is False for row in models)
    assert all(row["model_hash"].startswith("sha256:") for row in models)
    assert all(row["feature_schema_hash"] == models[0]["feature_schema_hash"] for row in models)


def test_scenario_bench_6545_train_cal_exception_and_runtime(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6545-TRAIN-CAL/RUNTIME: rows keep exact fallback safe."""

    receipts = artifact["training_and_calibration_receipts"]
    assert receipts["split_unit_counts"] == {"development": 20, "held": 18, "train": 21}
    assert receipts["train_rows_used_for_fitting"] == 21
    assert receipts["development_rows_used_for_calibration"] == 20
    assert receipts["held_rows_used_for_fitting"] is False
    assert receipts["held_rows_used_for_calibration"] is False
    assert receipts["held_rows_used_for_model_selection"] is False
    assert receipts["sealed_split_policy_passed"] is True

    table = artifact["exception_table_path_hash_and_freeze_receipt"]
    assert table["exception_table_path"].endswith("#train_only_exception_table")
    assert table["table_hash"].startswith("sha256:")
    assert table["frozen_before_held_evaluation"] is True
    assert table["held_entry_count"] == 0
    assert table["held_write_attempt_count"] == 0
    assert table["immutable_after_freeze"] is True
    assert {entry["split_name"] for entry in table["entries"]} == {"train"}

    calibration_rows = artifact["abstention_calibration_rows"]
    assert calibration_rows
    assert {row["split_name"] for row in calibration_rows} == {"development"}
    assert all(row["calibration_source"] == "development_only" for row in calibration_rows)
    assert all(row["threshold_frozen_before_held"] is True for row in calibration_rows)
    assert any(row["abstain"] is True for row in calibration_rows)
    assert any(row["abstain"] is False for row in calibration_rows)

    rows = artifact["per_unit_rows"]
    expected = 59 * len(mod.EVALUATION_SEEDS) * len(mod.ARM_IDS)
    assert len(rows) == expected
    assert {row["arm_id"] for row in rows} == set(mod.ARM_IDS)
    assert {row["split_name"] for row in rows} == {"development", "held", "train"}
    assert {row["seed"] for row in rows} == set(mod.EVALUATION_SEEDS)

    grouped: dict[tuple[str, int], set[tuple[str, ...]]] = {}
    for row in rows:
        key = (row["local_unit_id"], row["seed"])
        grouped.setdefault(key, set()).add(tuple(row["candidate_hashes"]))
        assert row["candidate_preserved"] is True
        assert row["candidate_deleted_count"] == 0
        assert row["proposal_count"] == len(row["candidate_hashes"])
        assert row["chosen_order"][0] in row["candidate_hashes"]
        assert row["exact_check_count"] == len(row["exact_checks"])
        assert row["fallback_available"] is True
        assert row["exact_equality"] is True
        assert row["timeout"] is False
        assert row["charged_total_cost_units"] == pytest.approx(
            row["proposal_cost_units"]
            + row["control_overhead_units"]
            + row["model_cost_units"]
            + row["lookup_cost_units"]
            + row["exact_check_cost_units"]
            + row["fallback_cost_units"]
        )
        assert row["wall_time_s"] >= 0.0
        assert row["row_hash"].startswith("sha256:")
    assert all(len(candidate_sets) == 1 for candidate_sets in grouped.values())

    held_final = [
        row
        for row in rows
        if row["split_name"] == "held" and row["arm_id"] == mod.SELECTED_SAFETY_NET_ARM
    ]
    assert held_final
    assert all(row["table_hit"] is False for row in held_final)
    assert any(row["abstention"] is True and row["fallback_used"] is True for row in held_final)
    assert any(row["abstention"] is False and row["fallback_used"] is False for row in held_final)
    assert sum(row["charged_total_cost_units"] for row in held_final) < artifact[
        "charged_cost_recomputation"
    ]["held_total_charged_work_by_arm"][mod.CERTIFIED_CONTROL_ARM]


def test_scenario_bench_6545_effects_attacks_rollback_and_checksum(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6545-EFFECTS/ATTACKS/ROLLBACK/TERMINAL: verdict is row-derived."""

    aggregate = artifact["aggregate_row_recomputation"]
    costs = artifact["charged_cost_recomputation"]
    exact = artifact["exact_equality_and_fallback_receipt"]
    attacks = artifact["shortcut_attack_matrix"]
    rollback = artifact["rollback_receipt"]

    assert aggregate["selected_eligible_arm"] == mod.SELECTED_SAFETY_NET_ARM
    assert aggregate["selected_arm_positive_beyond_structural"] is True
    assert aggregate["selected_arm_support_family_count"] > 1
    assert aggregate["selected_arm_support_effort_count"] > 1
    assert aggregate["calibrated_abstention_passed"] is True
    assert aggregate["exception_table_immutable_passed"] is True
    assert aggregate["ready_score_from_rows"] == 1.0
    assert aggregate["verdict_class_from_rows"] == "positive"

    selected_effect = next(
        row for row in artifact["paired_effect_rows"] if row["arm_id"] == mod.SELECTED_SAFETY_NET_ARM
    )
    assert selected_effect["held_effect_vs_certified_control_units"] > 0
    assert selected_effect["paired_unit_count"] == 54
    assert selected_effect["uncertainty"]["paired_std_error_units"] >= 0.0

    effect_rows = artifact["family_and_effort_effect_rows"]
    assert {row["stratum_type"] for row in effect_rows} == {"abstention_bin", "effort", "family"}
    assert any(row["no_headroom_cell"] is True for row in effect_rows)
    assert any(row["headroom_cell"] is True for row in effect_rows)
    assert any(row["held_effect_vs_certified_control_units"] < 0 for row in effect_rows)

    assert costs["all_costs_recomputed_from_rows"] is True
    assert costs["selected_eligible_arm"] == mod.SELECTED_SAFETY_NET_ARM
    assert costs["held_effect_vs_certified_control_units"] > 0
    assert exact["all_exact_equal"] is True
    assert exact["all_candidates_preserved"] is True
    assert exact["native_exact_fallback_reachable"] is True
    assert exact["fallback_used_count"] > 0
    assert exact["held_fallback_used_count"] > 0
    assert exact["verifier_is_oracle"] is False

    assert {row["attack_id"] for row in attacks["rows"]} == set(mod.SHORTCUT_ATTACK_IDS)
    assert attacks["all_shortcuts_fail_closed"] is True
    assert attacks["false_accept_count"] == 0
    assert all(row["fail_closed"] is True for row in attacks["rows"])
    assert rollback["rollback_target_arm"] == mod.CERTIFIED_CONTROL_ARM
    assert rollback["rollback_available"] is True
    assert rollback["unsafe_state_ready_score"] == 0.0
    assert rollback["rollback_checks_passed"] is True

    assert artifact["gate_check_summary"]["all_gates_passed"] is True
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert set(artifact["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["protected_files_unchanged"]["all_protected_files_unchanged"] is True
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
    assert mod.validate_artifact(artifact) == []


def test_scenario_bench_6545_missing_gate_closes_blocked(tmp_path: Path) -> None:
    """SCENARIO-BENCH-6545-GATE/ROLLBACK: missing Exp6544 blocks held claims."""

    blocked = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "blocked.json",
        upstream_path=tmp_path / "missing-exp6544.json",
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )

    assert blocked["status"] == "blocked_external_safety_net_router"
    assert blocked["honest_verdict"].startswith("blocked_")
    assert blocked["verdict_class"] == "blocked"
    assert blocked["external_safety_net_ready_score"] == 0.0
    assert blocked["upstream_gate_receipt"]["exists"] is False
    assert "upstream_gate_passed" in blocked["gate_check_summary"]["failed_checks"]
    assert blocked["rollback_receipt"]["rollback_available"] is True
    assert json.loads((tmp_path / "blocked.json").read_text(encoding="utf-8")) == blocked
    assert mod.validate_artifact(blocked) == []


def test_scenario_bench_6545_validation_main_and_defensive_paths(
    artifact: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-BENCH-6545-TERMINAL: CLI and validator fail closed."""

    validation_mutations = [
        ("required field set mismatch", lambda item: item.pop("status")),
        ("field_principles mismatch", lambda item: item.__setitem__("field_principles", {})),
        (
            "field_provenance must cover required fields",
            lambda item: item.__setitem__("field_provenance", {}),
        ),
        (
            "inference_substrate mismatch",
            lambda item: item.__setitem__("inference_substrate", "local_router"),
        ),
        (
            "verifier_is_oracle must be false",
            lambda item: item.__setitem__("verifier_is_oracle", True),
        ),
        (
            "positive verdict requires ready score 1.0",
            lambda item: item.__setitem__("external_safety_net_ready_score", 0.0),
        ),
        (
            "external_safety_net_ready_score must be 0.0 or 1.0",
            lambda item: item.__setitem__("external_safety_net_ready_score", 0.5),
        ),
        (
            "held table write detected",
            lambda item: item["exception_table_path_hash_and_freeze_receipt"].__setitem__(
                "held_write_attempt_count", 1
            ),
        ),
        (
            "calibration used held rows",
            lambda item: item["training_and_calibration_receipts"].__setitem__(
                "held_rows_used_for_calibration", True
            ),
        ),
        (
            "exact fallback unreachable",
            lambda item: item["exact_equality_and_fallback_receipt"].__setitem__(
                "native_exact_fallback_reachable", False
            ),
        ),
        (
            "shortcut false accept",
            lambda item: item["shortcut_attack_matrix"]["rows"][0].__setitem__(
                "fail_closed", False
            ),
        ),
        (
            "rollback unavailable",
            lambda item: item["rollback_receipt"].__setitem__("rollback_available", False),
        ),
        (
            "protected files changed",
            lambda item: item["protected_files_unchanged"].__setitem__(
                "all_protected_files_unchanged", False
            ),
        ),
        (
            "reproducibility_checksum mismatch",
            lambda item: item.__setitem__("reproducibility_checksum", "sha256:bad"),
        ),
        (
            "honest_verdict terminal prefix mismatch",
            lambda item: item.__setitem__("honest_verdict", "ready"),
        ),
        (
            "verdict_class outside Exp6545 enum",
            lambda item: item.__setitem__("verdict_class", "maybe"),
        ),
    ]
    for expected, mutate in validation_mutations:
        broken = deepcopy(artifact)
        mutate(broken)
        assert expected in mod.validate_artifact(broken)

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH.name
    assert mod.main(["--date", "20260823", "--result-path", str(result_path)]) == 0
    assert mod.main(["--validate", "--result-path", str(result_path)]) == 0
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["preconditions_checked"]["planning_date"] == "20260823"
    assert payload["external_safety_net_ready_score"] == 1.0
    assert adversarial_verify.verify_artifact(result_path)["flag_count"] == 0

    invalid = deepcopy(payload)
    invalid["status"] = "running_bootstrap"
    invalid["reproducibility_checksum"] = mod.reproducibility_checksum(invalid)
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
    assert mod.main(["--validate", "--result-path", str(invalid_path)]) == 1

    relative = mod.build_artifact(
        repo_root=REPO,
        result_path=Path("relative-6545.json"),
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )
    assert relative["preconditions_checked"]["result_path"].endswith("relative-6545.json")

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{", encoding="utf-8")
    assert mod._read_json_with_status(corrupt)[1] == "corrupt_json"
    non_object = tmp_path / "array.json"
    non_object.write_text("[]", encoding="utf-8")
    assert mod._read_json_with_status(non_object)[1] == "non_object"
    assert mod._load_jsonl(tmp_path / "missing.jsonl") == []
    assert mod._model_family_for_arm(mod.CERTIFIED_CONTROL_ARM) == "certified_structural"
    assert mod._target_hash({"candidate_hashes": ["a", "b"]}) == "b"
    assert mod._target_hash({"candidate_hashes": []}) == ""

    source_rows = [{}] * (len(artifact["per_unit_rows"]) // len(mod.ARM_IDS))
    partial_effects = deepcopy(artifact["paired_effect_rows"])
    selected = next(row for row in partial_effects if row["arm_id"] == mod.SELECTED_SAFETY_NET_ARM)
    selected["support_family_count"] = 1
    selected["support_effort_count"] = 1
    partial = mod.aggregate_row_recomputation(
        gate=artifact["upstream_gate_receipt"],
        source_rows=source_rows,
        rows=artifact["per_unit_rows"],
        effects=partial_effects,
        costs=artifact["charged_cost_recomputation"],
        exact=artifact["exact_equality_and_fallback_receipt"],
        attacks=artifact["shortcut_attack_matrix"],
        rollback=artifact["rollback_receipt"],
        training=artifact["training_and_calibration_receipts"],
        exception_table=artifact["exception_table_path_hash_and_freeze_receipt"],
        calibration_rows=artifact["abstention_calibration_rows"],
        protected=artifact["protected_files_unchanged"],
    )
    assert partial["verdict_class_from_rows"] == "partial"
    assert mod._status_and_verdict(partial)[2] == "partial"

    null_effects = deepcopy(artifact["paired_effect_rows"])
    next(row for row in null_effects if row["arm_id"] == mod.SELECTED_SAFETY_NET_ARM)[
        "held_effect_vs_certified_control_units"
    ] = 0.0
    null = mod.aggregate_row_recomputation(
        gate=artifact["upstream_gate_receipt"],
        source_rows=source_rows,
        rows=artifact["per_unit_rows"],
        effects=null_effects,
        costs=artifact["charged_cost_recomputation"],
        exact=artifact["exact_equality_and_fallback_receipt"],
        attacks=artifact["shortcut_attack_matrix"],
        rollback=artifact["rollback_receipt"],
        training=artifact["training_and_calibration_receipts"],
        exception_table=artifact["exception_table_path_hash_and_freeze_receipt"],
        calibration_rows=artifact["abstention_calibration_rows"],
        protected=artifact["protected_files_unchanged"],
    )
    assert null["verdict_class_from_rows"] is None
    assert mod._status_and_verdict(null)[2] is None
    assert mod._status_and_verdict({"verdict_class_from_rows": "disqualified"})[2] == (
        "disqualified"
    )

    monkeypatch.setattr(mod, "validate_artifact", lambda value: ["forced validation error"])
    forced_path = tmp_path / "forced.json"
    assert mod.main(["--date", "20260823", "--result-path", str(forced_path)]) == 1
