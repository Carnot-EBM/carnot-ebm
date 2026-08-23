"""Tests for Exp6544 external structural headroom.

Spec refs: REQ-BENCH-6544, SCENARIO-BENCH-6544-GATE,
SCENARIO-BENCH-6544-CONTRACT, SCENARIO-BENCH-6544-FAMILY-BLIND,
SCENARIO-BENCH-6544-COST-EQUALITY, SCENARIO-BENCH-6544-EFFECTS,
SCENARIO-BENCH-6544-ATTACKS, SCENARIO-BENCH-6544-TERMINAL.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6544_external_structural_headroom as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6544_external_structural_headroom.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6544_external_structural_headroom.py "
    "-m pytest tests/python/test_experiment_6544_external_structural_headroom.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6544_external_structural_headroom.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6544_external_structural_headroom.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6544_external_structural_headroom.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6544_external_structural_headroom.json"
)
EXACT_E2E_COMMAND = ".venv/bin/pytest tests/python/test_z3_live_benchmark.py -q --no-cov -n 0"
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6544_external_structural_headroom "
    "--date 20260823"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6544_external_structural_headroom --validate"
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
    """REQ-BENCH-6544: build a temp artifact from checked-in external evidence."""

    root = tmp_path_factory.mktemp("exp6544")
    return mod.build_artifact(
        repo_root=REPO,
        result_path=root / mod.RESULT_RELATIVE_PATH.name,
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )


def test_req_bench_6544_spec_declares_external_headroom_contract() -> None:
    """REQ-BENCH-6544: OpenSpec owns the external matched-control contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-BENCH-6544") :]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-BENCH-6544-GATE",
        "SCENARIO-BENCH-6544-CONTRACT",
        "SCENARIO-BENCH-6544-FAMILY-BLIND",
        "SCENARIO-BENCH-6544-COST-EQUALITY",
        "SCENARIO-BENCH-6544-EFFECTS",
        "SCENARIO-BENCH-6544-ATTACKS",
        "SCENARIO-BENCH-6544-TERMINAL",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "external_structural_headroom_ready_score",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_bench_6544_gate_and_frozen_contract(artifact: dict[str, Any]) -> None:
    """SCENARIO-BENCH-6544-GATE/CONTRACT: gate and contracts are frozen."""

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete_external_structural_headroom_positive"
    assert artifact["honest_verdict"].startswith("complete_external_structural_headroom_positive")
    assert artifact["verdict_class"] == "positive"
    assert artifact["external_structural_headroom_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False

    gate = artifact["upstream_gate_receipt"]
    assert gate["path"] == mod.EXP6543_RELATIVE_PATH.as_posix()
    assert gate["field"] == "external_constraint_corpus_audited_ready_score"
    assert gate["expected_value"] == 1.0
    assert gate["observed_value"] == 1.0
    assert gate["gate_passed"] is True
    assert gate["sha256"].startswith("sha256:")
    assert gate["input_hashes"]["fixture"] == mod.sha256_file(REPO / mod.FIXTURE_RELATIVE_PATH)
    assert gate["solver_identity"]["z3_python_available"] is True
    assert gate["resources"]["cpu_count"] >= 1
    assert gate["timeout_s"] == mod.TIMEOUT_S
    assert gate["seeds"] == list(mod.SEED_GRID)
    assert "scripts/research_conductor.py" in gate["protected_file_hashes_before"]

    contract = artifact["frozen_comparison_contract"]
    assert contract["arm_ids"] == list(mod.ARM_IDS)
    assert contract["seed_grid"] == list(mod.SEED_GRID)
    assert contract["candidate_budget_rule"] == "all_candidates_preserved"
    assert contract["exact_check_budget_rule"] == "candidate_count_plus_native_fallback"
    assert contract["timeout_s"] == mod.TIMEOUT_S
    assert contract["stop_rule"] == "stop_after_full_target_state_or_native_fallback"
    assert contract["forbidden_features"] == list(mod.FORBIDDEN_FEATURES)
    assert contract["family_blind_calibration"]["family_labels_used"] is False

    definitions = artifact["control_definitions"]
    assert list(definitions) == list(mod.ARM_IDS)
    assert all(row["learned_model_used"] is False for row in definitions.values())
    assert all(row["may_remove_candidates"] is False for row in definitions.values())
    assert all("family" not in row["ordering_features"] for row in definitions.values())


def test_scenario_bench_6544_rows_charge_equality_and_preservation(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6544-COST-EQUALITY: each row is matched and charged."""

    rows = artifact["per_unit_rows"]
    expected = (
        artifact["family_and_effort_census"]["fixture_row_count"]
        * len(mod.SEED_GRID)
        * len(mod.ARM_IDS)
    )
    assert len(rows) == expected
    assert {row["arm_id"] for row in rows} == set(mod.ARM_IDS)
    assert {row["split_name"] for row in rows} == {"development", "held", "train"}
    assert {row["family"] for row in rows} >= {"logic_grid", "scheduling", "seating"}
    assert {row["seed"] for row in rows} == set(mod.SEED_GRID)

    grouped: dict[tuple[str, int], set[tuple[str, ...]]] = {}
    for row in rows:
        key = (row["local_unit_id"], row["seed"])
        grouped.setdefault(key, set()).add(tuple(row["candidate_hashes"]))
        assert row["candidate_preserved"] is True
        assert row["candidate_deleted_count"] == 0
        assert row["exact_answer_equality"] is True
        assert row["timeout"] is False
        assert row["censored"] is False
        assert row["native_exact_fallback_available"] is True
        assert row["proposal_count"] == len(row["candidate_hashes"])
        assert row["candidate_order"][0] in row["candidate_hashes"]
        assert row["exact_check_count"] == len(row["exact_checks"])
        assert row["solver_effort"]["z3_check_calls"] == row["exact_check_count"]
        assert row["total_charged_work_units"] == (
            row["proposal_cost_units"]
            + row["exact_check_cost_units"]
            + row["control_overhead_units"]
            + row["fallback_cost_units"]
        )
        assert row["wall_time_s"] >= 0.0
        assert row["row_hash"].startswith("sha256:")
    assert all(len(candidate_sets) == 1 for candidate_sets in grouped.values())

    native = [row for row in rows if row["arm_id"] == mod.NATIVE_ARM and row["split_name"] == "held"]
    analytical = [
        row
        for row in rows
        if row["arm_id"] == "analytical" and row["split_name"] == "held"
    ]
    assert sum(row["total_charged_work_units"] for row in analytical) < sum(
        row["total_charged_work_units"] for row in native
    )


def test_scenario_bench_6544_effects_attacks_and_checksum(artifact: dict[str, Any]) -> None:
    """SCENARIO-BENCH-6544-EFFECTS/ATTACKS/TERMINAL: verdict is row-derived."""

    aggregate = artifact["aggregate_row_recomputation"]
    costs = artifact["charged_cost_recomputation"]
    equality = artifact["exact_equality_receipt"]
    preservation = artifact["candidate_preservation_receipt"]
    censoring = artifact["censoring_and_timeout_receipts"]
    attacks = artifact["shortcut_attack_matrix"]

    assert artifact["paired_effect_rows"]
    best = aggregate["best_arm"]
    assert best == "analytical"
    assert aggregate["best_arm_positive_beyond_native"] is True
    assert aggregate["best_arm_positive_beyond_random"] is True
    assert aggregate["best_arm_support_family_count"] > 1
    assert aggregate["ready_score_from_rows"] == 1.0
    assert aggregate["verdict_class_from_rows"] == "positive"

    held_effects = [row for row in artifact["paired_effect_rows"] if row["arm_id"] == best]
    assert held_effects[0]["held_effect_vs_native_units"] > 0
    assert held_effects[0]["held_effect_vs_random_units"] > 0
    assert held_effects[0]["paired_unit_count"] > 0
    assert held_effects[0]["uncertainty"]["paired_std_error_units"] >= 0.0

    family_rows = [row for row in artifact["family_effect_rows"] if row["arm_id"] == best]
    assert {row["family"] for row in family_rows} >= {"logic_grid", "scheduling", "seating"}
    assert any(row["headroom_cell"] is True for row in family_rows)
    assert all("simpson_reversal" in row for row in family_rows)

    assert costs["all_costs_recomputed_from_rows"] is True
    assert costs["best_arm"] == best
    assert equality["all_exact_equal"] is True
    assert equality["verifier_is_oracle"] is False
    assert equality["z3_evaluation_authority"] is True
    assert preservation["all_candidates_preserved"] is True
    assert preservation["candidate_set_identity_passed"] is True
    assert censoring["all_timeout_and_censoring_checks_passed"] is True

    assert {row["attack_id"] for row in attacks["rows"]} == set(mod.SHORTCUT_ATTACK_IDS)
    assert attacks["all_shortcuts_fail_closed"] is True
    assert attacks["false_accept_count"] == 0
    assert all(row["fail_closed"] is True for row in attacks["rows"])
    assert artifact["gate_check_summary"]["all_gates_passed"] is True
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert set(artifact["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["protected_files_unchanged"]["all_protected_files_unchanged"] is True
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
    assert mod.validate_artifact(artifact) == []


def test_scenario_bench_6544_blocked_gate_and_missing_source(tmp_path: Path) -> None:
    """SCENARIO-BENCH-6544-GATE: missing gate or source closes blocked."""

    blocked_gate = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "blocked-gate.json",
        audit_path=tmp_path / "missing-audit.json",
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )
    assert blocked_gate["status"] == "blocked_external_structural_headroom"
    assert blocked_gate["verdict_class"] == "blocked"
    assert blocked_gate["external_structural_headroom_ready_score"] == 0.0
    assert blocked_gate["upstream_gate_receipt"]["exists"] is False
    assert "upstream_gate_passed" in blocked_gate["gate_check_summary"]["failed_checks"]
    assert json.loads((tmp_path / "blocked-gate.json").read_text(encoding="utf-8")) == blocked_gate
    assert mod.validate_artifact(blocked_gate) == []

    blocked_source = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "blocked-source.json",
        source_root=tmp_path / "missing-source",
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )
    assert blocked_source["status"] == "blocked_external_structural_headroom"
    assert blocked_source["verdict_class"] == "blocked"
    assert "source_root_available" in blocked_source["gate_check_summary"]["failed_checks"]
    assert blocked_source["per_unit_rows"] == []
    assert mod.validate_artifact(blocked_source) == []


def test_scenario_bench_6544_validation_and_cli(
    artifact: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-BENCH-6544-TERMINAL: validators and CLI fail closed."""

    malformed = deepcopy(artifact)
    malformed.pop("status")
    malformed["field_principles"] = {}
    malformed["field_provenance"] = {}
    malformed["verdict_class"] = "circular_positive"
    malformed["honest_verdict"] = "bad"
    malformed["inference_substrate"] = "wrong"
    malformed["verifier_is_oracle"] = True
    malformed["external_structural_headroom_ready_score"] = 0.5
    malformed["aggregate_row_recomputation"]["ready_score_from_rows"] = 0.0
    malformed["gate_check_summary"]["all_gates_passed"] = False
    malformed["reproducibility_checksum"] = "sha256:bad"
    errors = mod.validate_artifact(malformed)
    assert "required field set mismatch" in errors
    assert "field_principles mismatch" in errors
    assert "field_provenance must cover required fields" in errors
    assert "verdict_class outside Exp6544 enum" in errors
    assert "honest_verdict terminal prefix mismatch" in errors
    assert "inference_substrate mismatch" in errors
    assert "verifier_is_oracle must be false" in errors
    assert "ready score mismatch" in errors
    assert "positive score requires all gates passed" in errors
    assert "reproducibility_checksum mismatch" in errors

    result_path = tmp_path / "cli-artifact.json"
    assert mod.main(["--result-path", str(result_path), "--date", "20260823"]) == 0
    assert result_path.is_file()
    assert mod.main(["--validate", "--result-path", str(result_path)]) == 0
    assert "validated" in capsys.readouterr().out

    bad_path = tmp_path / "bad.json"
    bad_payload = deepcopy(artifact)
    bad_payload["reproducibility_checksum"] = "sha256:bad"
    bad_path.write_text(json.dumps(bad_payload), encoding="utf-8")
    assert mod.main(["--validate", "--result-path", str(bad_path)]) == 1
    assert "reproducibility_checksum mismatch" in capsys.readouterr().out

    original_validate = mod.validate_artifact
    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: ["forced validation error"])
    forced_path = tmp_path / "forced-main-failure.json"
    assert mod.main(["--result-path", str(forced_path), "--date", "20260823"]) == 1
    assert "forced validation error" in capsys.readouterr().out
    monkeypatch.setattr(mod, "validate_artifact", original_validate)


def test_scenario_bench_6544_defensive_helpers(tmp_path: Path) -> None:
    """SCENARIO-BENCH-6544-GATE/TERMINAL: helper edge paths stay explicit."""

    assert mod._load_jsonl(tmp_path / "missing.jsonl") == []
    blank_jsonl = tmp_path / "blank.jsonl"
    blank_jsonl.write_text("\n[]\n", encoding="utf-8")
    assert mod._load_jsonl(blank_jsonl) == [{"value": []}]

    bad_audit = tmp_path / "bad-audit.json"
    bad_audit.write_text("{bad", encoding="utf-8")
    receipt = mod.upstream_gate_receipt(
        repo_root=REPO,
        audit_path=bad_audit,
        fixture_path=REPO / mod.FIXTURE_RELATIVE_PATH,
        protected_before={},
    )
    assert receipt["parse_status"] == "corrupt_json"
    assert receipt["gate_passed"] is False

    assert mod._solver_assertion_count(None, problem={}, constraints=[{}]) == 1
    no_checker = mod._candidate_receipt(
        checker=None,
        problem={"domain": "logic_grid", "entities": []},
        row={"turn_index": 0, "source_problem_hash": "sha256:test"},
        candidate_index=0,
        turn={"cumulative_constraints": []},
    )
    assert no_checker["exact_label"] == "error"

    class TimeoutChecker:
        def check_satisfiability(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            raise TimeoutError("too slow")

    timeout_receipt = mod._candidate_receipt(
        checker=TimeoutChecker(),
        problem={"domain": "logic_grid", "entities": []},
        row={"turn_index": 0, "source_problem_hash": "sha256:test"},
        candidate_index=0,
        turn={"cumulative_constraints": []},
    )
    assert timeout_receipt["exact_label"] == "timeout"

    source_root = tmp_path / "source"
    source_file = source_root / "data" / "problems" / "test" / "bad.json"
    source_file.parent.mkdir(parents=True)
    source_file.write_text(json.dumps({"turns": {}}), encoding="utf-8")
    row = {"source_file_relpath": "data/problems/test/bad.json", "turn_index": 0}
    assert mod._candidate_set(source_root=source_root, checker=None, row=row, receipt_cache={}) == []

    source_file.write_text(json.dumps({"turns": []}), encoding="utf-8")
    assert mod._candidate_set(
        source_root=source_root,
        checker=None,
        row={**row, "turn_index": 3},
        receipt_cache={},
    ) == []

    source_file.write_text(json.dumps({"turns": [None]}), encoding="utf-8")
    assert mod._candidate_set(source_root=source_root, checker=None, row=row, receipt_cache={}) == []

    bad_preservation = mod.candidate_preservation_receipt(
        [
            {
                "local_unit_id": "u",
                "seed": 1,
                "candidate_hashes": ["a"],
                "candidate_preserved": True,
                "candidate_deleted_count": 0,
            },
            {
                "local_unit_id": "u",
                "seed": 1,
                "candidate_hashes": ["b"],
                "candidate_preserved": True,
                "candidate_deleted_count": 0,
            },
        ]
    )
    assert bad_preservation["all_candidates_preserved"] is False
    assert bad_preservation["bad_groups"]

    synthetic_rows = [
        {
            "local_unit_id": f"u{index}",
            "seed": mod.SEED_GRID[index % len(mod.SEED_GRID)],
            "arm_id": mod.ARM_IDS[index % len(mod.ARM_IDS)],
            "split_name": "held",
            "total_charged_work_units": 1,
        }
        for index in range(len(mod.SEED_GRID) * len(mod.ARM_IDS))
    ]
    true_receipt = {"all_costs_recomputed_from_rows": True}
    true_equality = {"all_exact_equal": True}
    true_preservation = {"all_candidates_preserved": True}
    true_censoring = {"all_timeout_and_censoring_checks_passed": True}
    true_attacks = {"all_shortcuts_fail_closed": True}
    true_protected = {"all_protected_files_unchanged": True}
    partial = mod.aggregate_row_recomputation(
        gate={"gate_passed": True},
        source_root_available=True,
        fixture_rows=[{}],
        rows=synthetic_rows,
        effects=[
            {
                "arm_id": "random",
                "held_effect_vs_native_units": 1,
                "held_effect_vs_random_units": 0,
            }
        ],
        family_effects=[],
        costs=true_receipt,
        equality=true_equality,
        preservation=true_preservation,
        censoring=true_censoring,
        attacks=true_attacks,
        protected=true_protected,
    )
    assert partial["verdict_class_from_rows"] == "partial"
    assert mod._status_and_honest_verdict(partial)[2] == "partial"

    null = mod.aggregate_row_recomputation(
        gate={"gate_passed": True},
        source_root_available=True,
        fixture_rows=[{}],
        rows=synthetic_rows,
        effects=[
            {
                "arm_id": "random",
                "held_effect_vs_native_units": 0,
                "held_effect_vs_random_units": 0,
            }
        ],
        family_effects=[],
        costs=true_receipt,
        equality=true_equality,
        preservation=true_preservation,
        censoring=true_censoring,
        attacks=true_attacks,
        protected=true_protected,
    )
    assert null["verdict_class_from_rows"] is None
    assert mod._status_and_honest_verdict(null)[2] is None

    disqualified = mod.aggregate_row_recomputation(
        gate={"gate_passed": True},
        source_root_available=True,
        fixture_rows=[{}],
        rows=[],
        effects=[],
        family_effects=[],
        costs={},
        equality={},
        preservation={},
        censoring={},
        attacks={},
        protected=true_protected,
    )
    assert disqualified["verdict_class_from_rows"] == "disqualified"
    assert mod._status_and_honest_verdict(disqualified)[2] == "disqualified"

    relative = mod.build_artifact(
        repo_root=REPO,
        result_path=Path("relative-6544.json"),
        audit_path=tmp_path / "missing-audit.json",
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )
    assert relative["preconditions_checked"]["result_path"].endswith("relative-6544.json")
