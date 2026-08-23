"""Tests for Exp6519 structural headroom certification.

Spec refs: REQ-BENCH-6519, SCENARIO-BENCH-6519-MISSING-SOURCE,
SCENARIO-BENCH-6519-INDEPENDENT-ROWS, SCENARIO-BENCH-6519-EXACT-REPLAY,
SCENARIO-BENCH-6519-LIVE-COST-BREADTH, SCENARIO-BENCH-6519-ATTACKS,
SCENARIO-BENCH-6519-TERMINAL.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6519_structural_headroom_certificate as mod
from scripts import adversarial_verify


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6519_structural_headroom_certificate.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6519_structural_headroom_certificate.py "
    "-m pytest tests/python/test_experiment_6519_structural_headroom_certificate.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6519_structural_headroom_certificate.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6519_structural_headroom_certificate.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6519_structural_headroom_certificate.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6519_structural_headroom_certificate.json"
)
EXACT_E2E_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6477_backend_neutral_exact_constraint_record.py "
    "-q --no-cov -n 0"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6519_structural_headroom_certificate --validate"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6519_structural_headroom_certificate "
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
    """REQ-BENCH-6519: build a temp certificate from checked-in row evidence."""

    root = tmp_path_factory.mktemp("exp6519")
    return mod.build_artifact(
        repo_root=REPO,
        result_path=root / mod.RESULT_RELATIVE_PATH.name,
        source_path=REPO / mod.EXP6518_RELATIVE_PATH,
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )


def test_req_bench_6519_spec_declares_certificate_contract() -> None:
    """REQ-BENCH-6519: OpenSpec owns the row-derived certificate."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-BENCH-6519") : text.index("REQ-BENCH-3389")]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-BENCH-6519-MISSING-SOURCE",
        "SCENARIO-BENCH-6519-INDEPENDENT-ROWS",
        "SCENARIO-BENCH-6519-EXACT-REPLAY",
        "SCENARIO-BENCH-6519-LIVE-COST-BREADTH",
        "SCENARIO-BENCH-6519-ATTACKS",
        "SCENARIO-BENCH-6519-TERMINAL",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "certified_structural_headroom_score",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_bench_6519_terminal_source_and_schema(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6519-TERMINAL: valid row evidence certifies positive."""

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete_structural_headroom_certificate_positive"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["verdict_class"] == "positive"
    assert artifact["certified_structural_headroom_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert set(artifact["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["protected_files_unchanged"]["all_protected_files_unchanged"] is True
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
    assert mod.validate_artifact(artifact) == []

    receipt = artifact["upstream_artifact_receipt"]
    assert receipt["path"] == mod.EXP6518_RELATIVE_PATH.as_posix()
    assert receipt["exists"] is True
    assert receipt["parse_status"] == "parsed"
    assert receipt["status"] == "complete_structural_control_headroom_ab_v2_positive"
    assert receipt["verdict_class"] == "positive"
    assert receipt["class"] == "terminal_positive"
    assert receipt["per_unit_row_count"] == 645
    assert receipt["structural_control_game_row_count"] == 126
    assert receipt["resources"]["cpu_count"] >= 1
    assert "scripts/research_conductor.py" in receipt["protected_file_hashes_before"]


def test_scenario_bench_6519_independent_rows_and_exact_replay(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6519-INDEPENDENT-ROWS/EXACT-REPLAY: rows own evidence."""

    recompute = artifact["independent_row_recomputation"]
    replay_rows = artifact["exact_receipt_replay_rows"]
    per_rows = artifact["per_unit_rows"]

    assert recompute["source_row_container"] == "per_unit_rows"
    assert recompute["source_aggregate_fields_used"] is False
    assert recompute["source_terminal_passed"] is True
    assert recompute["sealed_pilot_rejoin_passed"] is True
    assert recompute["missing_unit_count"] == 0
    assert recompute["duplicate_unit_arm_count"] == 0
    assert recompute["post_hoc_modified_unit_count"] == 0
    assert recompute["candidate_value_pair_count"] == mod.EXPECTED_PILOT_UNIT_COUNT
    assert recompute["structural_control_game_row_count"] == mod.EXPECTED_MATCHED_ROW_COUNT
    assert recompute["row_type_counts"]["structural_control_game"] == mod.EXPECTED_MATCHED_ROW_COUNT

    assert len(replay_rows) >= mod.MIN_REPLAY_SAMPLE_ROWS
    assert artifact["aggregate_row_recomputation"]["correctness_discrepancy_count"] == 0
    assert all(row["replay_passed"] is True for row in replay_rows)
    assert all(row["exact_status_matches_row"] is True for row in replay_rows)
    assert all(row["z3_status_matches_row"] is True for row in replay_rows)
    assert all(row["decision_trace_hash_matches_row"] is True for row in replay_rows)
    assert all(row["charged_work_matches_row"] is True for row in replay_rows)
    assert any(row["live_influence_matches_row"] is True for row in replay_rows)

    unit_rows = [row for row in per_rows if row["row_type"] == "structural_headroom_unit_audit"]
    attack_rows = [row for row in per_rows if row["row_type"] == "structural_headroom_attack"]
    assert len(unit_rows) == mod.EXPECTED_MATCHED_ROW_COUNT
    assert len(attack_rows) == len(mod.ATTACK_IDS)
    assert all(row["audit_passed"] is True for row in unit_rows)
    assert all(row["sealed_pilot_join_passed"] is True for row in unit_rows)
    assert all(row["source_row_hash_recomputed"] is True for row in unit_rows)


def test_scenario_bench_6519_live_cost_breadth_and_attacks(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6519-LIVE-COST-BREADTH/ATTACKS: certification is gated."""

    influence = artifact["live_influence_audit"]
    costs = artifact["charged_cost_audit"]
    paired = artifact["paired_effect_rows"]
    breadth = artifact["breadth_and_censoring_audit"]
    attacks = artifact["attack_matrix"]
    aggregate = artifact["aggregate_row_recomputation"]

    assert influence["live_influence_passed"] is True
    assert influence["best_arm_live_influence_passed"] is True
    assert influence["best_arm_changed_decision_rows"] > 0
    assert costs["charged_cost_accounting_passed"] is True
    assert costs["best_arm_total_charged_benefit_units"] > 0
    assert costs["cost_omission_count"] == 0
    assert breadth["breadth_and_censoring_passed"] is True
    assert breadth["best_arm_support_family_count"] > 1
    assert breadth["best_arm_support_seed_count"] > 1
    assert breadth["timeout_count"] == 0
    assert breadth["censored_count"] == 0

    best = next(row for row in paired if row["arm_id"] == aggregate["best_arm"])
    assert best["held_charged_benefit_units"] > 0
    assert best["held_mean_benefit_units"] > 0
    assert best["long_tail_min_benefit_units"] < 0
    assert best["uncertainty_ci95_units"][0] <= best["held_mean_benefit_units"]
    assert best["support_family_count"] > 1
    assert best["support_seed_count"] > 1

    assert {row["attack_id"] for row in attacks["rows"]} == set(mod.ATTACK_IDS)
    assert attacks["all_attacks_fail_closed"] is True
    assert attacks["false_accept_count"] == 0
    assert all(row["fail_closed"] is True for row in attacks["rows"])
    assert aggregate["certification_conditions_met"] is True
    assert aggregate["certified_score_from_rows"] == 1.0
    assert artifact["gate_check_summary"]["all_gates_passed"] is True


def test_scenario_bench_6519_missing_source_closes_zero(tmp_path: Path) -> None:
    """SCENARIO-BENCH-6519-MISSING-SOURCE: absent Exp6518 blocks certification."""

    blocked = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "blocked.json",
        source_path=tmp_path / "missing-exp6518.json",
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )

    assert blocked["status"] == "blocked_structural_headroom_certificate"
    assert blocked["honest_verdict"].startswith("blocked_")
    assert blocked["verdict_class"] == "blocked"
    assert blocked["certified_structural_headroom_score"] == 0.0
    assert blocked["upstream_artifact_receipt"]["exists"] is False
    assert blocked["independent_row_recomputation"]["source_available_and_parsed"] is False
    assert "source_available_and_parsed" in blocked["gate_check_summary"]["failed_checks"]
    assert json.loads((tmp_path / "blocked.json").read_text(encoding="utf-8")) == blocked
    assert mod.validate_artifact(blocked) == []


def test_scenario_bench_6519_validation_and_aggregate_contradiction(
    artifact: dict[str, Any],
    tmp_path: Path,
) -> None:
    """SCENARIO-BENCH-6519-ATTACKS: malformed rows and copied aggregates fail."""

    source = json.loads((REPO / mod.EXP6518_RELATIVE_PATH).read_text(encoding="utf-8"))
    source["aggregate_row_recomputation"]["candidate_score_from_rows"] = 0.0
    source["aggregate_row_recomputation"]["best_arm_held_charged_benefit_units"] = -999
    contradictory_path = tmp_path / "contradictory-exp6518.json"
    contradictory_path.write_text(json.dumps(source), encoding="utf-8")

    contradictory = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "contradictory-result.json",
        source_path=contradictory_path,
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )
    assert contradictory["certified_structural_headroom_score"] == 1.0
    aggregate_attack = next(
        row
        for row in contradictory["attack_matrix"]["rows"]
        if row["attack_id"] == "aggregate_contradiction"
    )
    assert aggregate_attack["fail_closed"] is True
    assert aggregate_attack["observed_value"]["source_aggregate_contradiction_detected"] is True

    validation_mutations = [
        ("required field set mismatch", lambda item: item.pop("status")),
        ("field_principles mismatch", lambda item: item.__setitem__("field_principles", {})),
        (
            "field_provenance must cover required fields",
            lambda item: item.__setitem__("field_provenance", {}),
        ),
        (
            "inference_substrate mismatch",
            lambda item: item.__setitem__("inference_substrate", "procedural_exact_solver"),
        ),
        (
            "verifier_is_oracle must be false",
            lambda item: item.__setitem__("verifier_is_oracle", True),
        ),
        (
            "positive verdict requires certified score 1.0",
            lambda item: item.__setitem__("certified_structural_headroom_score", 0.0),
        ),
        (
            "certified_structural_headroom_score must be 0.0 or 1.0",
            lambda item: item.__setitem__("certified_structural_headroom_score", 0.5),
        ),
        (
            "row recomputation failed",
            lambda item: item["independent_row_recomputation"].__setitem__(
                "duplicate_unit_arm_count", 1
            ),
        ),
        (
            "correctness failed",
            lambda item: item["independent_row_recomputation"].__setitem__(
                "exact_answer_equality_passed", False
            ),
        ),
        (
            "exact replay failed",
            lambda item: item["exact_receipt_replay_rows"][0].__setitem__(
                "replay_passed", False
            ),
        ),
        (
            "live influence failed",
            lambda item: item["live_influence_audit"].__setitem__(
                "best_arm_live_influence_passed", False
            ),
        ),
        (
            "charged cost accounting failed",
            lambda item: item["charged_cost_audit"].__setitem__(
                "charged_cost_accounting_passed", False
            ),
        ),
        (
            "breadth or censoring failed",
            lambda item: item["breadth_and_censoring_audit"].__setitem__(
                "best_arm_support_seed_count", 1
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
            "verdict_class outside Exp6519 enum",
            lambda item: item.__setitem__("verdict_class", "partial"),
        ),
    ]
    for expected, mutate in validation_mutations:
        broken = deepcopy(artifact)
        mutate(broken)
        assert expected in mod.validate_artifact(broken)


def test_scenario_bench_6519_main_validate_and_adversarial_substrate(
    tmp_path: Path,
) -> None:
    """SCENARIO-BENCH-6519-TERMINAL: CLI, validation, and lints accept it."""

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
    assert payload["certified_structural_headroom_score"] == 1.0
    assert (
        adversarial_verify.duration_floor_for_artifact(payload)["reason"]
        == "deterministic_verifier"
    )
    report = adversarial_verify.verify_artifact(result_path)
    assert report["flag_count"] == 0

    invalid = deepcopy(payload)
    invalid["status"] = "running_bootstrap"
    invalid["reproducibility_checksum"] = mod.reproducibility_checksum(invalid)
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="status lacks terminal prefix"):
        mod.main(["--validate", "--result-path", str(invalid_path)])


def test_scenario_bench_6519_defensive_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-BENCH-6519-MISSING-SOURCE/TERMINAL: defensive paths are explicit."""

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

    monkeypatch.setattr(mod, "_read_json_with_status", lambda path: ({}, "missing", ""))
    assert mod._sealed_pilot_index(REPO) == {}
    monkeypatch.undo()

    assert mod._recompute_source_row_hash({}) is False
    assert mod._ci95([5.0]) == [5.0, 5.0]
    assert mod._source_aggregate_contradiction(
        {"aggregate_row_recomputation": []}, {}, 0.0
    ) is True

    tiny = {
        "unit_id": "u",
        "pilot_unit_id": "p",
        "arm_id": mod.NATIVE_ARM,
        "split": "held",
        "family": "f",
        "base_instance_hash": "missing",
        "checkpoint_id": "missing",
        "exact_answer_equality": False,
        "exact_status": "sat",
        "z3_status": "unsat",
        "terminal_model_or_proof": {"receipt_valid": False},
        "decision_trace_hash": "sha256:x",
        "solver_only_work_units": 1,
        "total_charged_work_units": 1,
        "first_changed_decision": None,
        "changed_decision_count": 0,
    }
    sampled = mod._sample_replay_rows([tiny], None)
    assert sampled
    assert all(row == tiny for row in sampled)
    replay = mod.exact_receipt_replay_rows(
        {"per_unit_rows": [{**tiny, "row_type": "structural_control_game"}]},
        repo_root=REPO,
        best_arm=None,
    )
    assert replay[0]["base_row_found"] is False
    assert replay[0]["replay_passed"] is False

    disqualified = mod._status_and_verdict(
        {"certified_score_from_rows": 0.0},
        {"failed_checks": ["row_recomputation_passed"]},
    )
    assert disqualified[0] == "disqualified_structural_headroom_certificate"
    assert disqualified[2] == "disqualified"
    null = mod._status_and_verdict({"certified_score_from_rows": 0.0}, {"failed_checks": []})
    assert null[0] == "complete_structural_headroom_certificate_null"
    assert null[2] is None

    relative = mod.build_artifact(
        repo_root=REPO,
        result_path=Path("relative-6519.json"),
        source_path=REPO / mod.EXP6518_RELATIVE_PATH,
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )
    assert relative["preconditions_checked"]["result_path"].endswith("relative-6519.json")

    monkeypatch.setattr(mod, "validate_artifact", lambda value: ["forced validation error"])
    with pytest.raises(ValueError, match="forced validation error"):
        mod.build_artifact(
            repo_root=REPO,
            result_path=tmp_path / "forced-error.json",
            source_path=tmp_path / "missing-exp6518.json",
            write=False,
            duration_s=1.0,
            tests_run=TESTS_RUN,
            run_date="20260823",
        )
