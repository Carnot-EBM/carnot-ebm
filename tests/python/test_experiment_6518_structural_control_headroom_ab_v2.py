"""Tests for Exp6518 structural branch-control headroom.

Spec refs: REQ-BENCH-6518, SCENARIO-BENCH-6518-AUDIT-GATE,
SCENARIO-BENCH-6518-ARM-CONTRACT, SCENARIO-BENCH-6518-LIVE-INFLUENCE,
SCENARIO-BENCH-6518-COST-EQUALITY, SCENARIO-BENCH-6518-HELD-TRANSFER,
SCENARIO-BENCH-6518-ATTACKS, SCENARIO-BENCH-6518-TERMINAL.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6518_structural_control_headroom_ab_v2 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH

FOCUSED_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6518_structural_control_headroom_ab_v2.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6518_structural_control_headroom_ab_v2.py "
    "-m pytest tests/python/test_experiment_6518_structural_control_headroom_ab_v2.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6518_structural_control_headroom_ab_v2.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6518_structural_control_headroom_ab_v2.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6518_structural_control_headroom_ab_v2.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6518_structural_control_headroom_ab_v2.json"
)
EXACT_E2E_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6477_backend_neutral_exact_constraint_record.py "
    "-q --no-cov -n 0"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6518_structural_control_headroom_ab_v2 "
    "--validate"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6518_structural_control_headroom_ab_v2 "
    "--date 20260823"
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
    {"command": EXACT_E2E_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": GIT_STATUS_COMMAND, "exit_code": 0},
]


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """REQ-BENCH-6518: build a temp artifact without changing tracked results."""

    root = tmp_path_factory.mktemp("exp6518")
    return mod.build_artifact(
        repo_root=REPO,
        result_path=root / mod.RESULT_RELATIVE_PATH.name,
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )


def test_req_bench_6518_spec_declares_structural_control_contract() -> None:
    """REQ-BENCH-6518: OpenSpec owns the matched-control contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-BENCH-6518") : text.index("REQ-BENCH-3389")]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-BENCH-6518-AUDIT-GATE",
        "SCENARIO-BENCH-6518-ARM-CONTRACT",
        "SCENARIO-BENCH-6518-LIVE-INFLUENCE",
        "SCENARIO-BENCH-6518-COST-EQUALITY",
        "SCENARIO-BENCH-6518-HELD-TRANSFER",
        "SCENARIO-BENCH-6518-ATTACKS",
        "SCENARIO-BENCH-6518-TERMINAL",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "structural_headroom_candidate_score",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_bench_6518_audit_gate_and_arm_contract(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6518-AUDIT-GATE/ARM-CONTRACT: gates and arms are frozen."""

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"].startswith("complete_")
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["verdict_class"] == "positive"
    assert artifact["structural_control_execution_complete_score"] == 1.0
    assert artifact["structural_headroom_candidate_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["preconditions_checked"]["exact_solver_is_label_authority"] is True

    gate = artifact["upstream_gate_receipt"]
    assert gate["path"] == mod.EXP6517_RELATIVE_PATH.as_posix()
    assert gate["field"] == "branch_pilot_audited_ready_score"
    assert gate["expected_value"] == 1.0
    assert gate["observed_value"] == 1.0
    assert gate["gate_passed"] is True
    assert gate["sha256"].startswith("sha256:")
    assert gate["solver_versions"]["z3_python_available"] is True
    assert gate["resources"]["cpu_count"] >= 1
    assert "scripts/research_conductor.py" in gate["protected_file_hashes_before"]

    contract = artifact["arm_contract"]
    assert contract["arm_ids"] == list(mod.ARM_IDS)
    assert contract["candidate_values"] == [False, True]
    assert contract["advice_can_remove_candidates"] is False
    assert contract["matched_solver_settings"]["assignment_budget"] == mod.EXACT_ASSIGNMENT_BUDGET
    assert contract["matched_solver_settings"]["restart_budget"] == mod.RESTART_BUDGET
    assert contract["matched_solver_settings"]["time_limit_s"] == mod.TIME_LIMIT_S
    assert all(row["candidate_preservation_required"] is True for row in contract["arms"])


def test_scenario_bench_6518_rows_equality_costs_and_live_influence(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6518-LIVE-INFLUENCE/COST-EQUALITY: rows are matched."""

    rows = artifact["per_game_results"]
    influence = artifact["live_influence_rows"]
    equality = artifact["exact_answer_equality_rows"]
    costs = artifact["charged_cost_rows"]
    censoring = artifact["censoring_rows"]

    assert len(rows) == mod.PILOT_BASE_UNIT_COUNT * len(mod.ARM_IDS)
    assert len(influence) == len(rows)
    assert len(equality) == len(rows)
    assert len(costs) == len(rows)
    assert len(censoring) == len(rows)
    assert {row["arm_id"] for row in rows} == set(mod.ARM_IDS)
    assert {row["split"] for row in rows} == {"development", "held", "train"}
    assert {row["family"] for row in rows} >= {"random_3cnf", "pseudo_industrial_3cnf", "tseitin"}
    assert len({row["selection_seed"] for row in rows}) > 1

    assert all(row["candidate_preserved"] is True for row in rows)
    assert all(row["candidate_values_available"] == [False, True] for row in rows)
    assert all(row["exact_answer_equality"] is True for row in rows)
    assert all(row["timeout"] is False for row in rows)
    assert all(row["censored"] is False for row in rows)
    assert all(row["restarts"] == 0 for row in rows)
    assert all(row["terminal_disposition"] in {"sat_model", "unsat_proof"} for row in rows)
    assert all(row["terminal_model_or_proof"]["receipt_valid"] is True for row in rows)
    assert all(row["row_hash"].startswith("sha256:") for row in rows)

    non_native = [row for row in influence if row["arm_id"] != mod.NATIVE_ARM]
    assert any(row["live_influence_detected"] is True for row in non_native)
    assert any(row["first_changed_decision"] is not None for row in non_native)
    assert all(row["native_arm"] == mod.NATIVE_ARM for row in influence)

    assert all(row["exact_status"] == row["z3_status"] for row in equality)
    assert all(row["solver_only_work_units"] > 0 for row in costs)
    assert all(row["total_charged_work_units"] >= row["solver_only_work_units"] for row in costs)
    assert all(row["fallback_cost_units"] >= 1 for row in costs)
    assert any(row["feature_cost_units"] > 0 for row in costs if row["arm_id"] != mod.NATIVE_ARM)
    assert all(row["censoring_passed"] is True for row in censoring)


def test_scenario_bench_6518_held_transfer_attacks_and_checksums(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6518-HELD-TRANSFER/ATTACKS/TERMINAL: scores are row-derived."""

    summary = artifact["family_seed_summary"]
    attacks = artifact["attack_matrix"]
    aggregate = artifact["aggregate_row_recomputation"]

    assert summary["primary_metric"] == mod.PRIMARY_METRIC
    assert summary["best_arm"] in set(mod.ARM_IDS) - {mod.NATIVE_ARM}
    assert summary["best_arm_held_charged_benefit_units"] > 0
    assert summary["best_arm_support_family_count"] > 1
    assert summary["best_arm_support_seed_count"] > 1
    assert summary["best_arm_correctness_equality"] is True
    assert summary["best_arm_live_influence"] is True

    assert {row["attack_id"] for row in attacks["rows"]} == set(mod.ATTACK_IDS)
    assert attacks["all_attacks_fail_closed"] is True
    assert attacks["false_accept_count"] == 0
    assert all(row["fail_closed"] is True for row in attacks["rows"])
    assert aggregate["execution_score_from_rows"] == 1.0
    assert aggregate["candidate_score_from_rows"] == 1.0
    assert artifact["gate_check_summary"]["all_gates_passed"] is True
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert set(artifact["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["protected_files_unchanged"]["all_protected_files_unchanged"] is True
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
    assert mod.validate_artifact(artifact) == []


def test_scenario_bench_6518_missing_audit_gate_closes_blocked(tmp_path: Path) -> None:
    """SCENARIO-BENCH-6518-AUDIT-GATE: missing gate writes a blocked artifact."""

    blocked = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "blocked.json",
        audit_path=tmp_path / "missing-audit.json",
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )

    assert blocked["status"] == "blocked_structural_control_headroom_ab_v2"
    assert blocked["honest_verdict"].startswith("blocked_")
    assert blocked["verdict_class"] == "blocked"
    assert blocked["structural_control_execution_complete_score"] == 0.0
    assert blocked["structural_headroom_candidate_score"] == 0.0
    assert blocked["upstream_gate_receipt"]["exists"] is False
    assert "audit_gate_passed" in blocked["gate_check_summary"]["failed_checks"]
    assert json.loads((tmp_path / "blocked.json").read_text(encoding="utf-8")) == blocked
    assert mod.validate_artifact(blocked) == []


def test_scenario_bench_6518_validation_fails_closed(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6518-ATTACKS/TERMINAL: malformed artifacts fail closed."""

    validation_mutations = [
        ("required field set mismatch", lambda item: item.pop("status")),
        ("field_principles mismatch", lambda item: item.__setitem__("field_principles", {})),
        (
            "field_provenance must cover required fields",
            lambda item: item.__setitem__("field_provenance", {}),
        ),
        (
            "inference_substrate mismatch",
            lambda item: item.__setitem__("inference_substrate", "live_llm_inference"),
        ),
        (
            "verifier_is_oracle must be false",
            lambda item: item.__setitem__("verifier_is_oracle", True),
        ),
        (
            "positive verdict requires candidate score 1.0",
            lambda item: item.__setitem__("structural_headroom_candidate_score", 0.0),
        ),
        (
            "structural_control_execution_complete_score must be 0.0 or 1.0",
            lambda item: item.__setitem__("structural_control_execution_complete_score", 0.5),
        ),
        (
            "structural_headroom_candidate_score must be 0.0 or 1.0",
            lambda item: item.__setitem__("structural_headroom_candidate_score", 0.5),
        ),
        (
            "audit gate failed",
            lambda item: item["upstream_gate_receipt"].__setitem__("observed_value", 0.0),
        ),
        (
            "candidate preservation failed",
            lambda item: item["per_game_results"][0].__setitem__("candidate_preserved", False),
        ),
        (
            "exact answer equality failed",
            lambda item: item["exact_answer_equality_rows"][0].__setitem__(
                "exact_answer_equality", False
            ),
        ),
        (
            "charged cost accounting failed",
            lambda item: item["charged_cost_rows"][0].__setitem__(
                "total_charged_work_units", 0
            ),
        ),
        (
            "censoring failed",
            lambda item: item["censoring_rows"][0].__setitem__("timeout", True),
        ),
        (
            "attack false accept",
            lambda item: item["attack_matrix"]["rows"][0].__setitem__("fail_closed", False),
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
            "verdict_class outside Exp6518 enum",
            lambda item: item.__setitem__("verdict_class", "circular_positive"),
        ),
    ]
    for expected, mutate in validation_mutations:
        broken = deepcopy(artifact)
        mutate(broken)
        assert expected in mod.validate_artifact(broken)


def test_scenario_bench_6518_main_and_validate_roundtrip(tmp_path: Path) -> None:
    """SCENARIO-BENCH-6518-TERMINAL: CLI writes and validates the artifact."""

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
    assert payload["structural_control_execution_complete_score"] == 1.0
    assert payload["structural_headroom_candidate_score"] == 1.0

    invalid = deepcopy(payload)
    invalid["status"] = "running_bootstrap"
    invalid["reproducibility_checksum"] = mod.reproducibility_checksum(invalid)
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="status lacks terminal prefix"):
        mod.main(["--validate", "--result-path", str(invalid_path)])


def test_scenario_bench_6518_defensive_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-BENCH-6518-TERMINAL: defensive helpers stay explicit."""

    disqualified = mod._status_and_verdict(
        {"audit_gate_passed": True, "execution_score_from_rows": 0.0},
        {"failed_checks": ["forced_failure"]},
    )
    assert disqualified[0] == "disqualified_structural_control_headroom_ab_v2"
    assert disqualified[2] == "disqualified"

    null = mod._status_and_verdict(
        {
            "audit_gate_passed": True,
            "execution_score_from_rows": 1.0,
            "candidate_score_from_rows": 0.0,
        },
        {"failed_checks": []},
    )
    assert null[0] == "complete_structural_control_headroom_ab_v2_null"
    assert null[2] is None

    relative = mod.build_artifact(
        repo_root=REPO,
        result_path=Path("relative-6518.json"),
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )
    assert relative["preconditions_checked"]["result_path"].endswith("relative-6518.json")

    monkeypatch.setattr(mod, "validate_artifact", lambda value: ["forced validation error"])
    with pytest.raises(ValueError, match="forced validation error"):
        mod.build_artifact(
            repo_root=REPO,
            result_path=tmp_path / "forced-error.json",
            audit_path=tmp_path / "missing-audit.json",
            write=False,
            duration_s=1.0,
            tests_run=TESTS_RUN,
            run_date="20260823",
        )
