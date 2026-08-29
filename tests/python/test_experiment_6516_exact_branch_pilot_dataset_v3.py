"""Tests for Exp6516 exact branch pilot dataset v3.

Spec refs: REQ-BENCH-6516, SCENARIO-BENCH-6516-DIRECT-IMMUTABLE,
SCENARIO-BENCH-6516-CHECKPOINTS, SCENARIO-BENCH-6516-CANDIDATES,
SCENARIO-BENCH-6516-EXACT-REPLAY, SCENARIO-BENCH-6516-SPLIT-SEALING,
SCENARIO-BENCH-6516-BOUNDED-SHARDS, SCENARIO-BENCH-6516-RESUME-ATOMIC,
SCENARIO-BENCH-6516-ATTACKS, SCENARIO-BENCH-6516-READY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6516_exact_branch_pilot_dataset_v3 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6516_exact_branch_pilot_dataset_v3.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6516_exact_branch_pilot_dataset_v3.py "
    "-m pytest tests/python/test_experiment_6516_exact_branch_pilot_dataset_v3.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6516_exact_branch_pilot_dataset_v3.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6516_exact_branch_pilot_dataset_v3.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6516_exact_branch_pilot_dataset_v3.json"
)
EXCLUSION_LINT_COMMAND = ".venv/bin/python scripts/exclusion_manifest_lint.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6516_exact_branch_pilot_dataset_v3.json"
)
EXACT_E2E_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6477_backend_neutral_exact_constraint_record.py "
    "-q --no-cov -n 0"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6516_exact_branch_pilot_dataset_v3 --validate"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6516_exact_branch_pilot_dataset_v3 --date 20260823"
)
GIT_STATUS_COMMAND = "git status --short"

TESTS_RUN = [
    {"command": FOCUSED_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
    {"command": SPEC_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": EXCLUSION_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": EXACT_E2E_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": GIT_STATUS_COMMAND, "exit_code": 0},
]


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """REQ-BENCH-6516: build a temp artifact without touching tracked results."""

    root = tmp_path_factory.mktemp("exp6516")
    return mod.build_artifact(
        repo_root=REPO,
        result_path=root / mod.RESULT_RELATIVE_PATH.name,
        work_root=root / "transactions",
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )


def test_req_bench_6516_spec_declares_pilot_contract() -> None:
    """REQ-BENCH-6516: OpenSpec owns the pilot dataset contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-BENCH-6516") : text.index("REQ-BENCH-3389")]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-BENCH-6516-DIRECT-IMMUTABLE",
        "SCENARIO-BENCH-6516-CHECKPOINTS",
        "SCENARIO-BENCH-6516-CANDIDATES",
        "SCENARIO-BENCH-6516-EXACT-REPLAY",
        "SCENARIO-BENCH-6516-SPLIT-SEALING",
        "SCENARIO-BENCH-6516-BOUNDED-SHARDS",
        "SCENARIO-BENCH-6516-RESUME-ATOMIC",
        "SCENARIO-BENCH-6516-ATTACKS",
        "SCENARIO-BENCH-6516-READY",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "branch_pilot_dataset_ready_score",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_bench_6516_direct_inputs_and_gates(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6516-DIRECT-IMMUTABLE: gates and inputs are hash-bound."""

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete_exact_branch_pilot_dataset_v3_ready"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["verdict_class"] is None
    assert artifact["branch_pilot_dataset_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True

    gates = {row["gate_id"]: row for row in artifact["upstream_gate_receipts"]}
    assert gates["exp6514_atomic_transaction"]["expected_value"] == 1.0
    assert gates["exp6514_atomic_transaction"]["observed_value"] == 1.0
    assert gates["exp6515_method_contract"]["expected_value"] == 1.0
    assert gates["exp6515_method_contract"]["observed_value"] == 1.0
    assert all(row["sha256"].startswith("sha256:") for row in gates.values())

    receipts = artifact["direct_input_receipts"]
    assert receipts["exp6504"]["path"] == mod.EXP6504_RELATIVE_PATH.as_posix()
    assert receipts["exp6510"]["path"] == mod.EXP6510_RELATIVE_PATH.as_posix()
    assert receipts["exp6504"]["read_mode"] == "direct_immutable_path_and_hash"
    assert receipts["exp6510"]["read_mode"] == "direct_immutable_path_and_hash"
    assert receipts["exp6504"]["sha256"].startswith("sha256:")
    assert receipts["exp6510"]["sha256"].startswith("sha256:")
    assert receipts["exp6510"]["structured_dependency_used"] is False

    prior = artifact["prior_failure_receipts"]
    # "null" since the 2026-08-28 exp6510 correction: a finished replay
    # declares null, not the may-retry partial (REQ-CONDUCTOR-VERDICT-3).
    assert prior["exp6510_verdict_class"] == "null"
    assert prior["exp6510_ready_score"] == 1.0
    assert prior["retired_structured_dependency_used"] is False
    assert prior["exp6511_dataset_missing_or_retired"] is True


def test_scenario_bench_6516_checkpoints_candidates_and_exact_replay(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6516-CHECKPOINTS/CANDIDATES/EXACT-REPLAY: rows are exact."""

    rows = artifact["branch_counterfactual_rows"]
    receipts = artifact["exact_solver_receipts"]
    families = {row["family"] for row in rows}
    scales = {row["scale"] for row in rows}
    splits = {row["split"] for row in rows}
    by_checkpoint: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_checkpoint.setdefault(row["checkpoint_id"], []).append(row)

    assert len(rows) == mod.PILOT_BASE_UNIT_COUNT * 2
    assert len(receipts) == len(rows)
    assert len(families) >= 3
    assert scales == {"medium", "small"}
    assert splits == {"development", "held", "train"}
    assert len({row["selection_seed"] for row in rows}) > 1
    assert {row["candidate_value"] for row in rows} == {False, True}
    assert {row["exact_budget"] for row in rows} == {mod.EXACT_ASSIGNMENT_BUDGET}
    assert all(
        sorted(row["candidate_value"] for row in group) == [False, True]
        for group in by_checkpoint.values()
    )

    assert all(row["terminal_disposition"] in {"sat_model", "unsat_proof"} for row in rows)
    assert all(row["timeout"] is False for row in rows)
    assert all(row["censored"] is False for row in rows)
    assert all(row["exact_receipt"]["valid"] is True for row in rows)
    assert all(row["exact_receipt"]["exact_answer_equality"] is True for row in rows)
    assert all(row["row_hash"].startswith("sha256:") for row in rows)
    assert all(row["conflicts"] >= 0 for row in rows)
    assert all(row["propagations"] >= row["assignments_examined"] for row in rows)
    assert all(row["decisions"] >= row["assignments_examined"] for row in rows)
    assert all(row["restarts"] == 0 for row in rows)

    receipt_by_unit = {receipt["unit_id"]: receipt for receipt in receipts}
    assert set(receipt_by_unit) == {row["unit_id"] for row in rows}
    assert all(receipt["exact_answer_equality"] is True for receipt in receipts)
    assert all(
        receipt["terminal_disposition"] in {"sat_model", "unsat_proof"} for receipt in receipts
    )


def test_scenario_bench_6516_split_shards_schema_and_attacks(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6516-SPLIT-SEALING/BOUNDED-SHARDS/ATTACKS: audits pass."""

    feature_names = {
        feature["name"] for feature in artifact["structural_feature_schema"]["features"]
    }
    forbidden = set(mod.FORBIDDEN_FEATURE_NAMES)
    assert feature_names.isdisjoint(forbidden)
    assert artifact["checkpoint_contract"]["uses_only_decision_time_structural_features"] is True
    assert artifact["checkpoint_contract"]["eligible_values"] == [False, True]

    split = artifact["split_commitment"]
    assert split["sealed_split_passed"] is True
    assert split["base_lineage_overlap_count"] == 0
    assert split["minimum_cell_floor_observed"] >= split["minimum_cell_floor_required"]
    assert split["post_held_repair_count"] == 0

    shard = artifact["shard_manifest"]
    counts = artifact["planned_and_terminal_unit_counts"]
    assert shard["complete"] is True
    assert shard["transaction_schema"] == mod.TRANSACTION_SCHEMA
    assert shard["resume_verified"] is True
    assert shard["corrupt_resume_detected"] is True
    assert shard["final_transaction_verified"] is True
    assert shard["terminal_row_count"] == len(artifact["branch_counterfactual_rows"])
    assert (
        counts["planned_unit_count"]
        == counts["terminal_unit_count"]
        == len(artifact["branch_counterfactual_rows"])
    )
    assert counts["missing_terminal_unit_count"] == 0

    budget_rows = artifact["censoring_and_budget_rows"]
    assert all(row["equal_budget"] is True for row in budget_rows)
    assert all(row["terminal_disposition_present"] is True for row in budget_rows)
    assert all(row["censored"] is False for row in budget_rows)

    attacks = artifact["leakage_attack_matrix"]
    assert {row["attack_id"] for row in attacks["rows"]} == set(mod.ATTACK_IDS)
    assert attacks["all_attacks_fail_closed"] is True
    assert attacks["false_accept_count"] == 0
    assert all(row["fail_closed"] is True for row in attacks["rows"])

    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert set(artifact["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["protected_files_unchanged"]["all_protected_files_unchanged"] is True
    assert artifact["aggregate_row_recomputation"]["ready_score_from_rows"] == 1.0
    assert artifact["gate_check_summary"]["all_gates_passed"] is True
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
    assert mod.validate_artifact(artifact) == []


def test_scenario_bench_6516_validation_fails_closed(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6516-ATTACKS/READY: malformed artifacts fail validation."""

    validation_mutations = [
        ("required field set mismatch", lambda item: item.pop("status")),
        ("field_principles mismatch", lambda item: item.__setitem__("field_principles", {})),
        (
            "field_provenance must cover required fields",
            lambda item: item.__setitem__("field_provenance", {}),
        ),
        (
            "verdict_class cannot be positive",
            lambda item: item.__setitem__("verdict_class", "positive"),
        ),
        (
            "inference_substrate mismatch",
            lambda item: item.__setitem__("inference_substrate", "live_llm_inference"),
        ),
        (
            "verifier_is_oracle must be true",
            lambda item: item.__setitem__("verifier_is_oracle", False),
        ),
        (
            "upstream gate failed",
            lambda item: item["upstream_gate_receipts"][0].__setitem__("observed_value", 0.0),
        ),
        (
            "direct input receipt missing",
            lambda item: item["direct_input_receipts"]["exp6504"].__setitem__("exists", False),
        ),
        (
            "exact receipt failure",
            lambda item: item["branch_counterfactual_rows"][0]["exact_receipt"].__setitem__(
                "valid", False
            ),
        ),
        (
            "duplicate checkpoint detected",
            lambda item: item["branch_counterfactual_rows"][2].__setitem__(
                "checkpoint_id", item["branch_counterfactual_rows"][0]["checkpoint_id"]
            ),
        ),
        (
            "split leakage detected",
            lambda item: item["branch_counterfactual_rows"][2].__setitem__(
                "base_lineage_id", item["branch_counterfactual_rows"][0]["base_lineage_id"]
            ),
        ),
        (
            "asymmetric budget detected",
            lambda item: item["branch_counterfactual_rows"][1].__setitem__(
                "exact_budget", mod.EXACT_ASSIGNMENT_BUDGET + 1
            ),
        ),
        (
            "forbidden feature present",
            lambda item: item["structural_feature_schema"]["features"].append(
                {"name": "row_order", "available_at": "decision_time"}
            ),
        ),
        (
            "transaction resume not verified",
            lambda item: item["shard_manifest"].__setitem__("resume_verified", False),
        ),
        (
            "corrupt resume attack not detected",
            lambda item: item["shard_manifest"].__setitem__("corrupt_resume_detected", False),
        ),
        (
            "omitted hard row detected",
            lambda item: item["planned_and_terminal_unit_counts"].__setitem__(
                "planned_unit_count", len(item["branch_counterfactual_rows"]) + 1
            ),
        ),
        (
            "leakage attack false accept",
            lambda item: item["leakage_attack_matrix"]["rows"][0].__setitem__("fail_closed", False),
        ),
        (
            "branch_pilot_dataset_ready_score mismatch",
            lambda item: item.__setitem__("branch_pilot_dataset_ready_score", 0.0),
        ),
        (
            "branch_pilot_dataset_ready_score must be 0.0 or 1.0",
            lambda item: item.__setitem__("branch_pilot_dataset_ready_score", 0.5),
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
    ]
    for expected, mutate in validation_mutations:
        broken = deepcopy(artifact)
        mutate(broken)
        assert expected in mod.validate_artifact(broken)


def test_scenario_bench_6516_main_and_validate_roundtrip(tmp_path: Path) -> None:
    """SCENARIO-BENCH-6516-RESUME-ATOMIC: CLI writes and validates the artifact."""

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

    assert payload["branch_pilot_dataset_ready_score"] == 1.0
    assert payload["shard_manifest"]["final_transaction_verified"] is True
    assert payload["preconditions_checked"]["run_date"] == "20260823"

    invalid = deepcopy(payload)
    invalid["status"] = "running_bootstrap"
    invalid["reproducibility_checksum"] = mod.reproducibility_checksum(invalid)
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="status lacks terminal prefix"):
        mod.main(["--validate", "--result-path", str(invalid_path)])


def test_scenario_bench_6516_defensive_paths(
    artifact: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-BENCH-6516-READY: blocked and defensive helper paths close."""

    assert mod.sha256_file(tmp_path / "missing.json") == "missing"
    status, honest, verdict_class = mod._status_and_verdict(
        0.0,
        {"blocked_reason": "forced"},
    )
    assert status == "blocked_exact_branch_pilot_dataset_v3"
    assert honest == "blocked_exact_branch_pilot_dataset_v3: forced"
    assert verdict_class == "blocked"

    with pytest.raises(ValueError, match="missing pilot cell"):
        mod._select_pilot_base_rows([])

    exp6504_payload = json.loads((REPO / mod.EXP6504_RELATIVE_PATH).read_text(encoding="utf-8"))
    base = mod._select_pilot_base_rows(exp6504_payload["raw_instance_rows"])[0]
    _contract, checkpoints = mod.freeze_checkpoints([base])
    monkeypatch.setattr(mod, "EXACT_ASSIGNMENT_BUDGET", 0)
    timed = mod._terminal_payload(
        row=base,
        checkpoint=checkpoints[str(base["instance_id"])],
        candidate_value=False,
    )
    assert timed["terminal_disposition"] == "timeout"
    assert timed["timeout"] is True
    monkeypatch.setattr(mod, "EXACT_ASSIGNMENT_BUDGET", 256)

    fake_repo = tmp_path / "fake-repo"
    for relative in (
        mod.EXP6504_RELATIVE_PATH,
        mod.EXP6510_RELATIVE_PATH,
        mod.EXP6514_RELATIVE_PATH,
        mod.EXP6515_RELATIVE_PATH,
    ):
        target = fake_repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO / relative).read_bytes())
    existing_work = fake_repo / "relative-work"
    existing_work.mkdir(parents=True)
    relative_artifact = mod.build_artifact(
        repo_root=fake_repo,
        result_path=Path("relative-result.json"),
        work_root=Path("relative-work"),
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )
    assert relative_artifact["preconditions_checked"]["result_path"].endswith(
        "relative-result.json"
    )

    monkeypatch.setattr(mod, "validate_artifact", lambda value: ["forced validation error"])
    with pytest.raises(ValueError, match="forced validation error"):
        mod.build_artifact(
            repo_root=REPO,
            result_path=tmp_path / "invalid-build.json",
            work_root=tmp_path / "invalid-work",
            write=False,
            duration_s=1.0,
            tests_run=TESTS_RUN,
            run_date="20260823",
        )
