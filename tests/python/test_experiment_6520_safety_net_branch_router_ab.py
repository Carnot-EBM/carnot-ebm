"""Tests for Exp6520 safety-net branch router A/B.

Spec refs: REQ-BENCH-6520, SCENARIO-BENCH-6520-GATE,
SCENARIO-BENCH-6520-ARMS, SCENARIO-BENCH-6520-EXCEPTIONS,
SCENARIO-BENCH-6520-RUNTIME, SCENARIO-BENCH-6520-EXHAUSTIVE,
SCENARIO-BENCH-6520-ATTACKS, SCENARIO-BENCH-6520-TERMINAL.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6520_safety_net_branch_router_ab as mod
from scripts import adversarial_verify


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6520_safety_net_branch_router_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6520_safety_net_branch_router_ab.py "
    "-m pytest tests/python/test_experiment_6520_safety_net_branch_router_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6520_safety_net_branch_router_ab.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6520_safety_net_branch_router_ab.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6520_safety_net_branch_router_ab.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6520_safety_net_branch_router_ab.json"
)
TRAINING_E2E_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6518_structural_control_headroom_ab_v2.py "
    "-q --no-cov -n 0"
)
EXACT_E2E_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6477_backend_neutral_exact_constraint_record.py "
    "-q --no-cov -n 0"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6520_safety_net_branch_router_ab --date 20260823"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6520_safety_net_branch_router_ab --validate"
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
    {"command": TRAINING_E2E_COMMAND, "exit_code": 0},
    {"command": EXACT_E2E_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": GIT_STATUS_COMMAND, "exit_code": 0},
]


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """REQ-BENCH-6520: build a temp artifact without touching tracked results."""

    root = tmp_path_factory.mktemp("exp6520")
    return mod.build_artifact(
        repo_root=REPO,
        result_path=root / mod.RESULT_RELATIVE_PATH.name,
        source_path=REPO / mod.EXP6519_RELATIVE_PATH,
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )


def test_req_bench_6520_spec_declares_safety_net_contract() -> None:
    """REQ-BENCH-6520: OpenSpec owns the safety-net router contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-BENCH-6520") : text.index("REQ-BENCH-3389")]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-BENCH-6520-GATE",
        "SCENARIO-BENCH-6520-ARMS",
        "SCENARIO-BENCH-6520-EXCEPTIONS",
        "SCENARIO-BENCH-6520-RUNTIME",
        "SCENARIO-BENCH-6520-EXHAUSTIVE",
        "SCENARIO-BENCH-6520-ATTACKS",
        "SCENARIO-BENCH-6520-TERMINAL",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "exact_solver_is_release_authority=true",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_bench_6520_gate_and_matched_arm_specs(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6520-GATE/ARMS: gate and budgets are frozen."""

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete_safety_net_branch_router_ab_positive"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["verdict_class"] == "positive"
    assert artifact["safety_net_router_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["preconditions_checked"]["exact_solver_is_release_authority"] is True

    gate = artifact["upstream_gate_receipt"]
    assert gate["path"] == mod.EXP6519_RELATIVE_PATH.as_posix()
    assert gate["field"] == "certified_structural_headroom_score"
    assert gate["expected_value"] == 1.0
    assert gate["observed_value"] == 1.0
    assert gate["gate_passed"] is True
    assert gate["sha256"].startswith("sha256:")
    assert gate["resources"]["cpu_count"] >= 1
    assert "python" in gate["framework_versions"]
    assert "scripts/research_conductor.py" in gate["protected_file_hashes_before"]

    specs = artifact["model_and_arm_specs"]
    assert specs["arm_ids"] == list(mod.ARM_IDS)
    assert specs["learned_arm_ids"] == list(mod.LEARNED_ARM_IDS)
    assert specs["feature_schema"]["feature_names"] == list(mod.FEATURE_NAMES)
    assert specs["matched_budget"]["optimization_steps"] == mod.OPTIMIZATION_STEPS
    assert specs["matched_budget"]["confidence_abstain_threshold"] == mod.CONFIDENCE_ABSTAIN_THRESHOLD
    assert specs["matched_budget"]["exact_assignment_budget"] == mod.EXACT_ASSIGNMENT_BUDGET
    assert specs["seed_grid"] == list(mod.MODEL_SEEDS)
    assert all(row["advice_can_remove_candidates"] is False for row in specs["arms"])


def test_scenario_bench_6520_exception_tables_are_train_dev_only(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6520-EXCEPTIONS: held rows cannot enter exception tables."""

    receipts = artifact["train_dev_held_receipts"]
    manifest = artifact["exception_table_manifest"]

    assert receipts["split_unit_counts"] == {"development": 6, "held": 6, "train": 6}
    assert receipts["train_dev_unit_count"] == 12
    assert receipts["held_unit_count"] == 6
    assert receipts["held_rows_used_for_training"] is False
    assert receipts["held_rows_used_for_exception_writes"] is False
    assert receipts["train_only_writes_passed"] is True

    assert manifest["held_rows_in_table_count"] == 0
    assert manifest["all_train_dev_errors_covered"] is True
    assert manifest["key_collision_count"] == 0
    assert manifest["manifest_hash"].startswith("sha256:")
    assert manifest["bounded_table_size_passed"] is True
    for table in manifest["tables"]:
        assert table["arm_id"] in mod.LEARNED_ARM_IDS
        assert table["held_entry_count"] == 0
        assert table["covered_train_dev_error_count"] == table["train_dev_error_count"]
        assert table["table_hash"].startswith("sha256:")
        for entry in table["entries"]:
            assert entry["split"] in {"train", "development"}
            assert entry["key_hash"].startswith("sha256:")
            assert entry["value_hash"].startswith("sha256:")
            assert entry["lineage_hash"].startswith("sha256:")
            assert entry["model_version_hash"] == table["model_version_hash"]
            assert entry["schema_version_hash"] == manifest["schema_version_hash"]


def test_scenario_bench_6520_runtime_preserves_candidates_and_exact_answers(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6520-RUNTIME/EXHAUSTIVE: routing never prunes candidates."""

    rows = artifact["per_game_results"]
    fallback = artifact["exception_abstention_fallback_rows"]
    preservation = artifact["candidate_preservation_rows"]
    equality = artifact["exact_answer_equality_rows"]
    audit = artifact["exhaustive_pilot_audit"]

    assert len(rows) == mod.PILOT_UNIT_COUNT * len(mod.ARM_IDS)
    assert len(fallback) == len(rows)
    assert len(preservation) == len(rows)
    assert len(equality) == len(rows)
    assert {row["split"] for row in rows} == {"development", "held", "train"}
    assert {row["arm_id"] for row in rows} == set(mod.ARM_IDS)

    assert all(row["candidate_preserved"] is True for row in rows)
    assert all(row["candidate_values_available"] == [False, True] for row in rows)
    assert all(row["candidate_pruned_count"] == 0 for row in rows)
    assert all(row["exact_answer_equality"] is True for row in rows)
    assert all(row["exact_solver_is_release_authority"] is True for row in rows)
    assert all(row["terminal_disposition"] in {"sat_model", "unsat_proof"} for row in rows)
    assert all(row["row_hash"].startswith("sha256:") for row in rows)

    assert all(row["candidate_preservation_passed"] is True for row in preservation)
    assert all(row["exact_answer_equality"] is True for row in equality)
    assert any(row["fallback_trigger"] == "abstention" for row in fallback)
    assert any(row["fallback_trigger"] == "exception_hit" for row in fallback)
    assert any(row["runtime_order_source"] == "learned_order" for row in fallback)
    assert all(
        row["exception_hit"] is False
        for row in fallback
        if row["split"] == "held" and row["arm_id"] in mod.LEARNED_ARM_IDS
    )

    assert audit["expected_route_row_count"] == len(rows)
    assert audit["observed_route_row_count"] == len(rows)
    assert audit["bounded_domain_exhaustive"] is True
    assert audit["candidate_preservation_passed"] is True
    assert audit["exact_answer_equality_passed"] is True
    assert audit["changed_decision_count"] > 0
    assert audit["fallback_count"] > 0
    assert audit["abstention_count"] > 0


def test_scenario_bench_6520_costs_attacks_and_row_recomputation(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6520-ATTACKS/TERMINAL: the positive verdict is row-derived."""

    costs = artifact["charged_cost_and_storage_rows"]
    attacks = artifact["attack_matrix"]
    aggregate = artifact["aggregate_row_recomputation"]

    assert len(costs) == len(artifact["per_game_results"])
    assert all(row["total_charged_work_units"] >= row["solver_work_units"] for row in costs)
    assert all(row["storage_charge_units"] >= 0 for row in costs)

    assert aggregate["best_learned_arm"] in mod.LEARNED_ARM_IDS
    assert aggregate["best_learned_model_family"] in {"linear", "mlp", "kan"}
    assert aggregate["best_learned_held_charged_benefit_units"] > 0
    assert aggregate["upstream_best_structural_held_benefit_units"] == 667
    assert aggregate["held_benefit_beyond_best_structural_units"] > 0
    assert aggregate["best_learned_support_problem_family_count"] > 1
    assert aggregate["best_learned_support_problem_seed_count"] > 1
    assert aggregate["positive_supported_model_family_count"] > 1
    assert aggregate["positive_supported_model_seed_count"] > 1
    assert aggregate["exact_answer_equality_passed"] is True
    assert aggregate["candidate_preservation_passed"] is True
    assert aggregate["bounded_table_size_passed"] is True
    assert aggregate["positive_conditions_met"] is True

    assert {row["attack_id"] for row in attacks["rows"]} == set(mod.ATTACK_IDS)
    assert attacks["all_attacks_fail_closed"] is True
    assert attacks["false_accept_count"] == 0
    assert all(row["fail_closed"] is True for row in attacks["rows"])
    assert artifact["gate_check_summary"]["all_gates_passed"] is True
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert set(artifact["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["protected_files_unchanged"]["all_protected_files_unchanged"] is True
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
    assert mod.validate_artifact(artifact) == []


def test_scenario_bench_6520_missing_gate_closes_blocked(tmp_path: Path) -> None:
    """SCENARIO-BENCH-6520-GATE: missing Exp6519 blocks router scoring."""

    blocked = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "blocked.json",
        source_path=tmp_path / "missing-exp6519.json",
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )

    assert blocked["status"] == "blocked_safety_net_branch_router_ab"
    assert blocked["honest_verdict"].startswith("blocked_")
    assert blocked["verdict_class"] == "blocked"
    assert blocked["safety_net_router_ready_score"] == 0.0
    assert blocked["upstream_gate_receipt"]["exists"] is False
    assert "upstream_gate_passed" in blocked["gate_check_summary"]["failed_checks"]
    assert json.loads((tmp_path / "blocked.json").read_text(encoding="utf-8")) == blocked
    assert mod.validate_artifact(blocked) == []


def test_scenario_bench_6520_validation_fails_closed(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6520-TERMINAL: malformed artifacts fail validation."""

    validation_mutations = [
        ("required field set mismatch", lambda item: item.pop("status")),
        ("field_principles mismatch", lambda item: item.__setitem__("field_principles", {})),
        (
            "field_provenance must cover required fields",
            lambda item: item.__setitem__("field_provenance", {}),
        ),
        (
            "inference_substrate mismatch",
            lambda item: item.__setitem__("inference_substrate", "local_compact_router"),
        ),
        (
            "verifier_is_oracle must be false",
            lambda item: item.__setitem__("verifier_is_oracle", True),
        ),
        (
            "exact solver release authority missing",
            lambda item: item["preconditions_checked"].__setitem__(
                "exact_solver_is_release_authority", False
            ),
        ),
        (
            "positive verdict requires ready score 1.0",
            lambda item: item.__setitem__("safety_net_router_ready_score", 0.0),
        ),
        (
            "safety_net_router_ready_score must be 0.0 or 1.0",
            lambda item: item.__setitem__("safety_net_router_ready_score", 0.5),
        ),
        (
            "held contamination detected",
            lambda item: item["exception_table_manifest"].__setitem__(
                "held_rows_in_table_count", 1
            ),
        ),
        (
            "candidate preservation failed",
            lambda item: item["candidate_preservation_rows"][0].__setitem__(
                "candidate_preservation_passed", False
            ),
        ),
        (
            "exact answer equality failed",
            lambda item: item["exact_answer_equality_rows"][0].__setitem__(
                "exact_answer_equality", False
            ),
        ),
        (
            "attack false accept",
            lambda item: item["attack_matrix"]["rows"][0].__setitem__(
                "fail_closed", False
            ),
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
            "honest_verdict lacks terminal prefix",
            lambda item: item.__setitem__("honest_verdict", "ready"),
        ),
        (
            "verdict_class outside Exp6520 enum",
            lambda item: item.__setitem__("verdict_class", "maybe_positive"),
        ),
    ]
    for expected, mutate in validation_mutations:
        broken = deepcopy(artifact)
        mutate(broken)
        assert expected in mod.validate_artifact(broken)


def test_scenario_bench_6520_main_validate_and_defensive_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-BENCH-6520-TERMINAL: CLI and helper paths are explicit."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH.name

    assert (
        mod.main(
            [
                "--date",
                "20260823",
                "--result-path",
                str(result_path),
            ]
        )
        == 0
    )
    assert mod.main(["--validate", "--result-path", str(result_path)]) == 0
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["preconditions_checked"]["planning_date"] == "20260823"
    assert payload["safety_net_router_ready_score"] == 1.0
    assert adversarial_verify.duration_floor_for_artifact(payload)["reason"] == "no_llm_declared"
    assert adversarial_verify.verify_artifact(result_path)["flag_count"] == 0

    invalid = deepcopy(payload)
    invalid["status"] = "running_bootstrap"
    invalid["reproducibility_checksum"] = mod.reproducibility_checksum(invalid)
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="status lacks terminal prefix"):
        mod.main(["--validate", "--result-path", str(invalid_path)])

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{", encoding="utf-8")
    assert mod._read_json_with_status(corrupt)[1] == "corrupt_json"
    non_object = tmp_path / "array.json"
    non_object.write_text("[]", encoding="utf-8")
    assert mod._read_json_with_status(non_object)[1] == "non_object"

    assert mod._terminal_class({"status": "running"}, "parsed") == "nonterminal"
    assert mod._terminal_class({"status": "blocked_x", "verdict_class": "blocked"}, "parsed") == (
        "terminal_blocked"
    )
    assert mod._terminal_class(
        {"status": "disqualified_x", "verdict_class": "disqualified"}, "parsed"
    ) == "terminal_disqualified"
    assert mod._terminal_class({"status": "complete_x", "verdict_class": None}, "parsed") == (
        "terminal_null"
    )
    assert mod._terminal_class({"status": "complete_x", "verdict_class": "other"}, "parsed") == (
        "terminal_other"
    )
    assert mod._terminal_class({}, "missing") == "missing"

    monkeypatch.setattr(mod, "_read_json_with_status", lambda path: ({}, "missing", ""))
    assert mod._load_branch_units(REPO) == []
    monkeypatch.setattr(
        mod,
        "_read_json_with_status",
        lambda path: (
            {
                "branch_counterfactual_rows": [
                    {
                        "base_instance_hash": "b",
                        "checkpoint_id": "c",
                    }
                ]
            },
            "parsed",
            "",
        ),
    )
    assert mod._load_branch_units(REPO) == []
    monkeypatch.undo()

    minimal_unit = {
        "decision_time_features": {
            name: 1 for name in mod.FEATURE_NAMES
        }
    }
    assert mod._model_score(minimal_unit, mod.NATIVE_ARM) == 0.0
    assert mod._predicted_first_value(minimal_unit, mod.NATIVE_ARM) is False
    assert mod._table_entries_by_key({"tables": [None]}) == {}
    assert mod._status_and_verdict(
        {},
        {"failed_checks": ["candidate_preservation_passed"]},
    )[2] == "disqualified"
    assert mod._status_and_verdict(
        {"complete_null_conditions_met": True},
        {"failed_checks": []},
    )[2] is None
    assert mod._status_and_verdict({}, {"failed_checks": []})[2] == "partial"

    relative = mod.build_artifact(
        repo_root=REPO,
        result_path=Path("relative-6520.json"),
        source_path=REPO / mod.EXP6519_RELATIVE_PATH,
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )
    assert relative["preconditions_checked"]["result_path"].endswith("relative-6520.json")

    monkeypatch.setattr(mod, "validate_artifact", lambda value: ["forced validation error"])
    with pytest.raises(ValueError, match="forced validation error"):
        mod.build_artifact(
            repo_root=REPO,
            result_path=tmp_path / "forced-error.json",
            source_path=tmp_path / "missing-exp6519.json",
            write=False,
            duration_s=1.0,
            tests_run=TESTS_RUN,
            run_date="20260823",
        )
