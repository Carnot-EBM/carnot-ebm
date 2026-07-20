"""Tests for Exp5746 exact proposal utility hard/soft benchmark.

Spec refs: REQ-VERIFY-5746, SCENARIO-VERIFY-5746.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5746_exact_proposal_utility_benchmark as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_5746_exact_proposal_utility_benchmark.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5746_exact_proposal_utility_benchmark.py "
    "-m pytest tests/python/test_experiment_5746_exact_proposal_utility_benchmark.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5746_exact_proposal_utility_benchmark.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5746_exact_proposal_utility_benchmark.json"
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


def _run_fixture(tmp_path: Path) -> dict[str, object]:
    return mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        benchmark_manifest_path=tmp_path / mod.BENCHMARK_MANIFEST_RELATIVE_PATH.name,
        preconditions_checked=mod.fixture_preflight_receipt(),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )


def test_req_verify_5746_spec_declares_dual_receipt_contract() -> None:
    """REQ-VERIFY-5746: OpenSpec anchors fields, receipts, and no-LLM gates."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5746") : spec.index("### REQ-VERIFY-5615")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5746",
        "SCENARIO-VERIFY-5746",
        str(mod.RESULT_RELATIVE_PATH),
        str(mod.BENCHMARK_MANIFEST_RELATIVE_PATH),
        "finite_domain_csp",
        "weighted_maxsat",
        "hard_soft_packing",
        "finite_state_planning",
        "top-1/top-k feasible discovery",
        "`llm_inference_used` SHALL be false",
        "`verifier_is_oracle` SHALL be true",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert " ".join(mod.FIELD_PRINCIPLES[field].split()) in normalized


def test_scenario_verify_5746_generates_balanced_deterministic_manifest(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5746: deterministic rows are balanced, sealed, and ready."""

    artifact = _run_fixture(tmp_path)
    rows = mod.read_benchmark_manifest(tmp_path / mod.BENCHMARK_MANIFEST_RELATIVE_PATH.name)
    rerun = _run_fixture(tmp_path)

    assert artifact == rerun
    assert mod.validate_artifact(artifact) is True
    assert mod.verify_benchmark_manifest(rows, artifact) is True
    assert artifact["instance_count"] == 180
    assert artifact["family_counts"] == {family: 45 for family in mod.REQUIRED_FAMILIES}
    assert artifact["split_manifest"]["split_counts"] == {"dev": 60, "science": 60, "train": 60}
    for split_counts in artifact["split_manifest"]["family_counts"].values():
        assert split_counts == {family: 15 for family in mod.REQUIRED_FAMILIES}
    assert len(artifact["science_row_hashes"]) == 60
    assert artifact["disjoint_from_v512_score"] == pytest.approx(1.0)
    assert artifact["candidate_domain_incomplete_count"] == 0
    assert artifact["structure_receipt_failure_count"] == 0
    assert artifact["solution_receipt_failure_count"] == 0
    assert artifact["validator_disagreement_count"] == 0
    assert artifact["benchmark_ready_score"] == pytest.approx(1.0)
    assert artifact["llm_inference_used"] is False
    assert artifact["verifier_is_oracle"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["benchmark_manifest_hash"] == mod.sha256_file(
        tmp_path / mod.BENCHMARK_MANIFEST_RELATIVE_PATH.name
    )
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text(encoding="utf-8")) == artifact


def test_req_verify_5746_receipts_cover_every_candidate_and_ordering(tmp_path: Path) -> None:
    """REQ-VERIFY-5746: receipts prove structure, solution values, and baselines."""

    artifact = _run_fixture(tmp_path)
    rows = mod.read_benchmark_manifest(tmp_path / mod.BENCHMARK_MANIFEST_RELATIVE_PATH.name)
    by_id = {row["instance_id"]: row for row in rows}

    assert set(artifact).issubset(set(artifact["field_principles"]))
    for family in mod.REQUIRED_FAMILIES:
        row = next(item for item in rows if item["family"] == family)
        row_id = row["instance_id"]
        candidates = row["candidate_pool"]
        candidate_ids = {candidate["candidate_id"] for candidate in candidates}
        pool = artifact["candidate_pool_receipts"][row_id]
        structure = artifact["structure_receipts"][row_id]
        solution = artifact["solution_receipts"][row_id]
        hard = artifact["hard_constraint_receipts"][row_id]
        soft = artifact["soft_objective_receipts"][row_id]
        optimum = artifact["exact_optimum_receipts"][row_id]
        ordering = artifact["baseline_orderings"][row_id]

        assert pool["candidate_count"] == len(candidates)
        assert pool["domain_complete"] is True
        assert structure["structure_complete"] is True
        assert hard["all_candidates_checked"] is True
        assert soft["all_candidates_scored"] is True
        assert solution["candidate_count"] == len(candidates)
        assert set(solution["candidate_evaluations"]) == candidate_ids
        assert set(ordering["exact_solver_native_order"]) == candidate_ids
        assert set(ordering["random_permutation_order"]) == candidate_ids
        assert set(ordering["energy_heuristic_order"]) == candidate_ids
        assert ordering["energy_heuristic_order"][0] in set(optimum["optimal_candidate_ids"])
        assert optimum["feasible_candidate_count"] == len(row["exact_feasible_set"])
        for candidate_id in optimum["optimal_candidate_ids"]:
            evaluation = solution["candidate_evaluations"][candidate_id]
            assert evaluation["feasible"] is True
            assert evaluation["objective_value"] == optimum["optimum_value"]
        assert by_id[row_id]["row_hash"] == artifact["benchmark_row_hashes"][row_id]


def test_req_verify_5746_independent_checks_preflight_and_adversarial_controls() -> None:
    """REQ-VERIFY-5746: independent checks and adversarial controls fail closed."""

    instances = mod.generate_instances()
    preflight = mod.collect_preconditions(
        planned_instance_ids=[row["instance_id"] for row in instances],
        command_runner=lambda name: f"{name} fixture-version",
        memory_probe=lambda: {"available_mb": 8192, "required_mb": 512, "ok": True},
        disk_probe=lambda: {"available_mb": 8192, "required_mb": 512, "ok": True},
        v512_collision_probe=lambda ids: {"collision_count": 0, "colliding_ids": [], "score": 1.0},
    )
    failures = mod.collect_independent_validator_failures(instances)
    controls = mod.build_adversarial_controls(instances)
    missing_command_preflight = mod.collect_preconditions(
        planned_instance_ids=[],
        command_runner=lambda name: (_ for _ in ()).throw(FileNotFoundError(name)),
        memory_probe=lambda: {"available_mb": 1, "required_mb": 512, "ok": False},
        disk_probe=lambda: {"available_mb": 1, "required_mb": 512, "ok": False},
        v512_collision_probe=lambda ids: {"collision_count": len(ids), "colliding_ids": ids, "score": 0.0},
        python_version_ok=False,
        exact_solvers_available=False,
    )

    assert preflight["preflight_ready"] is True
    assert missing_command_preflight["preflight_ready"] is False
    assert "python_version_too_old" in missing_command_preflight["blocked_reasons"]
    assert "required_exact_solver_unavailable" in missing_command_preflight["blocked_reasons"]
    assert len([row for row in failures["sample_receipts"] if row["sampled"]]) >= 20
    assert failures["validator_disagreement_count"] == 0
    assert set(controls) == set(mod.ADVERSARIAL_CONTROL_TYPES)
    assert all(control["detected"] is True for control in controls.values())
    assert controls["omitted_constraint"]["blocked_gate"] == "structure_receipt_failure"
    assert controls["omitted_candidate"]["blocked_gate"] == "candidate_domain_incomplete"
    assert controls["duplicate_candidate"]["blocked_gate"] == "candidate_domain_duplicate"
    assert controls["infeasible_best_score"]["blocked_gate"] == "hard_constraint_receipt"
    assert controls["shortcut"]["blocked_gate"] == "hard_constraint_receipt"
    assert controls["objective_sign"]["blocked_gate"] == "soft_objective_receipt"


def test_req_verify_5746_blocked_preflight_writes_terminal_artifact(tmp_path: Path) -> None:
    """REQ-VERIFY-5746: unavailable exact solvers produce an honest blocked artifact."""

    preflight = mod.fixture_preflight_receipt()
    preflight["preflight_ready"] = False
    preflight["exact_solvers_available"] = False
    preflight["blocked_reasons"] = ["required_exact_solver_unavailable"]
    artifact = mod.run(
        result_path=tmp_path / "blocked.json",
        benchmark_manifest_path=tmp_path / "blocked.jsonl",
        preconditions_checked=preflight,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )

    assert artifact["instance_count"] == 0
    assert artifact["benchmark_ready_score"] == pytest.approx(0.0)
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["preconditions_checked"]["exact_solvers_available"] is False
    assert mod.read_benchmark_manifest(tmp_path / "blocked.jsonl") == []
    assert mod.validate_artifact(artifact) is True


def test_req_verify_5746_validation_and_manifest_replay_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-5746: schema, readiness, checksum, and manifest tampering fail."""

    artifact = _run_fixture(tmp_path)
    rows = mod.read_benchmark_manifest(tmp_path / mod.BENCHMARK_MANIFEST_RELATIVE_PATH.name)

    missing = deepcopy(artifact)
    del missing["verifier_is_oracle"]
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing)

    bad_bool = deepcopy(artifact)
    bad_bool["llm_inference_used"] = True
    bad_bool["benchmark_ready_score"] = mod.benchmark_ready_score(bad_bool)
    bad_bool["honest_verdict"] = mod.honest_verdict(bad_bool)
    bad_bool["reproducibility_checksum"] = mod.reproducibility_checksum(bad_bool)
    with pytest.raises(ValueError, match="llm_inference_used"):
        mod.validate_artifact(bad_bool)

    bad_score = deepcopy(artifact)
    bad_score["benchmark_ready_score"] = 0.0
    bad_score["honest_verdict"] = mod.honest_verdict(bad_score)
    bad_score["reproducibility_checksum"] = mod.reproducibility_checksum(bad_score)
    with pytest.raises(ValueError, match="benchmark_ready_score"):
        mod.validate_artifact(bad_score)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "blocked: wrong"
    bad_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(bad_verdict)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)

    reasoned = deepcopy(artifact)
    reasoned["candidate_domain_incomplete_count"] = 1
    reasoned["disjoint_from_v512_score"] = 0.0
    reasoned["adversarial_controls"]["omitted_candidate"]["detected"] = False
    reasoned["verifier_is_oracle"] = False
    reasons = mod._blocked_reasons(reasoned)
    assert "candidate_domain_incomplete_count" in reasons
    assert "v512_disjointness_failed" in reasons
    assert "adversarial_control_not_detected" in reasons
    assert "verifier_not_oracle" in reasons

    tampered_rows = deepcopy(rows)
    tampered_rows[0]["row_hash"] = "sha256:" + "1" * 64
    with pytest.raises(mod.ManifestReplayError, match="row_hash"):
        mod.verify_benchmark_manifest(tampered_rows, artifact)

    bad_science = deepcopy(artifact)
    bad_science["science_row_hashes"] = []
    with pytest.raises(mod.ManifestReplayError, match="science_row_hashes"):
        mod.verify_benchmark_manifest(rows, bad_science)
