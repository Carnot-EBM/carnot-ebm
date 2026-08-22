"""Tests for Exp6504 exact structural benchmark commitment.

Spec refs: REQ-BENCH-6504, SCENARIO-BENCH-6504-GENERATION,
SCENARIO-BENCH-6504-LABELS, SCENARIO-BENCH-6504-SPLITS,
SCENARIO-BENCH-6504-STRATA, SCENARIO-BENCH-6504-LEAKAGE,
SCENARIO-BENCH-6504-SCHEMA.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6504_exact_structural_benchmark_commitment as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH

FOCUSED_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6504_exact_structural_benchmark_commitment.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6504_exact_structural_benchmark_commitment.py "
    "-m pytest tests/python/test_experiment_6504_exact_structural_benchmark_commitment.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6504_exact_structural_benchmark_commitment.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6504_exact_structural_benchmark_commitment.py"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m "
    "carnot.experiment_6504_exact_structural_benchmark_commitment --date 20260822"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6504_exact_structural_benchmark_commitment.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6504_exact_structural_benchmark_commitment.json"
)
EXACT_E2E_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6477_backend_neutral_exact_constraint_record.py "
    "-q --no-cov -n 0"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6504_exact_structural_benchmark_commitment --validate"
)
RUFF_COMMAND = (
    ".venv/bin/ruff check "
    "python/carnot/experiment_6504_exact_structural_benchmark_commitment.py "
    "tests/python/test_experiment_6504_exact_structural_benchmark_commitment.py "
    "scripts/adversarial_verify.py"
)
GIT_STATUS_COMMAND = "git status --short"
TESTS_RUN = [
    {"command": FOCUSED_COMMAND, "exit_code": 0},
    {"command": COVERAGE_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
    {"command": SPEC_COMMAND, "exit_code": 0},
    {"command": RUFF_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": EXACT_E2E_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": GIT_STATUS_COMMAND, "exit_code": 0},
]


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    result_path = tmp_path_factory.mktemp("exp6504") / mod.RESULT_RELATIVE_PATH.name
    return mod.build_artifact(
        repo_root=REPO,
        result_path=result_path,
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260822",
    )


def test_req_bench_6504_spec_declares_exact_benchmark_contract() -> None:
    """REQ-BENCH-6504: OpenSpec owns the benchmark commitment."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-BENCH-6504") :]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-BENCH-6504-GENERATION",
        "SCENARIO-BENCH-6504-LABELS",
        "SCENARIO-BENCH-6504-SPLITS",
        "SCENARIO-BENCH-6504-STRATA",
        "SCENARIO-BENCH-6504-LEAKAGE",
        "SCENARIO-BENCH-6504-SCHEMA",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`verifier_is_oracle=true`",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_bench_6504_generation_labels_and_replay(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6504-GENERATION/LABELS: exact rows are replayable."""

    raw_rows = artifact["raw_instance_rows"]
    label_rows = artifact["exact_label_rows"]
    replay_rows = artifact["exact_replay_rows"]

    assert len(raw_rows) == mod.INSTANCE_COUNT
    assert len(label_rows) == mod.INSTANCE_COUNT
    assert len(replay_rows) == mod.INSTANCE_COUNT
    assert {row["family"] for row in raw_rows} == set(mod.FAMILIES)
    assert all(row["raw_instance_hash"].startswith("sha256:") for row in raw_rows)
    assert all("exact_label" not in row for row in raw_rows)
    assert all(row["feature_extraction_event_index"] > row["split_commitment_event_index"] for row in raw_rows)

    assert all(row["accepted"] is True for row in label_rows)
    assert all(row["solver_disagreement"] is False for row in label_rows)
    assert all(row["hand_corrected_label"] is False for row in label_rows)
    assert all(row["model_or_proof_valid"] is True for row in label_rows)
    assert {row["exact_label"] for row in label_rows} == {"sat", "unsat"}
    assert all(row["replay_passed"] is True for row in replay_rows)
    assert all(row["deterministic_replay"] is True for row in replay_rows)

    held_labels = Counter(
        (row["family"], row["exact_label"])
        for row in label_rows
        if row["split"] == "held"
    )
    for family in mod.FAMILIES:
        assert held_labels[(family, "sat")] == 30
        assert held_labels[(family, "unsat")] == 30


def test_scenario_bench_6504_splits_strata_cells_and_leakage(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6504-SPLITS/STRATA/LEAKAGE: commitments fail closed."""

    split = artifact["split_commitment"]
    cells = artifact["minimum_held_cell_size"]
    attacks = artifact["leakage_attack_matrix"]
    aggregate = artifact["aggregate_row_recomputation"]

    assert split["label_inspected_before_split"] is False
    assert split["base_lineage_cross_split_count"] == 0
    for family in mod.FAMILIES:
        assert split["family_split_counts"][family] == {
            "train": 10,
            "development": 10,
            "held": 60,
        }

    assert cells["required_minimum_held_units"] == 30
    assert cells["observed_minimum_held_units"] == 30
    assert cells["all_planned_headline_cells_pass"] is True
    assert all(row["held_unit_count"] >= 30 for row in cells["planned_headline_cell_rows"])

    assert {row["attack_id"] for row in attacks["rows"]} == set(mod.LEAKAGE_ATTACK_IDS)
    assert attacks["all_attacks_fail_closed"] is True
    assert attacks["false_accept_count"] == 0
    assert all(row["allowed_as_feature"] is False for row in attacks["rows"])

    assert artifact["stratum_balance_rows"]
    for row in artifact["stratum_balance_rows"]:
        assert row["solver_effort_used_as_model_difficulty_proxy"] is False
        assert row["family"] in mod.FAMILIES
        assert row["label_counts"]["sat"] + row["label_counts"]["unsat"] == row["unit_count"]

    assert aggregate["raw_instance_row_count"] == mod.INSTANCE_COUNT
    assert aggregate["accepted_label_count"] == mod.INSTANCE_COUNT
    assert aggregate["quarantined_label_count"] == 0
    assert aggregate["exact_replay_failure_count"] == 0
    assert aggregate["minimum_held_cell_size_passed"] is True
    assert aggregate["base_structural_benchmark_ready_score_from_rows"] == 1.0


def test_scenario_bench_6504_schema_checksum_and_preconditions(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6504-SCHEMA: artifact validates from raw rows."""

    result_path = Path(artifact["preconditions_checked"]["result_path"])
    written = json.loads(result_path.read_text(encoding="utf-8"))

    assert written == artifact
    assert mod.validate_artifact(artifact) == []
    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete_exact_structural_benchmark_committed"
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["honest_verdict"].startswith(
        "complete_exact_structural_benchmark_commitment:"
    )
    assert artifact["base_structural_benchmark_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)

    receipts = {row["experiment_id"]: row for row in artifact["upstream_gate_receipts"]}
    assert receipts["exp6502"]["observed_value"] == 1.0
    assert receipts["exp6502"]["passed"] is True
    assert receipts["exp6503"]["observed_value"] == 1.0
    assert receipts["exp6503"]["passed"] is True
    assert artifact["preconditions_checked"]["resources"]["cpu"]["logical_cpu_count"] >= 1
    assert artifact["preconditions_checked"]["resources"]["disk"]["free_bytes"] > 0
    assert artifact["preconditions_checked"]["solver_tools"]["z3"]["available"] is True
    assert artifact["preconditions_checked"]["solver_tools"]["exhaustive"]["available"] is True
    assert artifact["protected_files_unchanged"]["all_protected_files_unchanged"] is True


def test_scenario_bench_6504_quarantines_backend_disagreement(
    artifact: dict[str, Any],
    tmp_path: Path,
) -> None:
    """SCENARIO-BENCH-6504-LABELS: backend disagreement blocks readiness."""

    raw = artifact["raw_instance_rows"][0]
    assert mod.sha256_file(tmp_path / "missing.json") == "missing"
    assert mod._status_verdict(0.0, {"blocked_reason": "blocked_test"}) == (
        "blocked_exact_structural_benchmark_commitment",
        "blocked",
        "blocked_exact_structural_benchmark_commitment: blocked_test",
    )

    def disagreeing_z3(_n_vars: int, _clauses: list[list[int]]) -> mod.SolverOutcome:
        return mod.SolverOutcome(
            backend="z3",
            available=True,
            status="unsat" if raw["family"] else "sat",
            model=None,
            assignments_examined=0,
            version="test-disagreement",
            command="z3-python-test",
        )

    label = mod.label_instance(raw, z3_solver=disagreeing_z3)
    assert label["accepted"] is False
    assert label["solver_disagreement"] is True
    assert label["quarantine_reason"] == "backend_disagreement"

    tampered = deepcopy(artifact)
    tampered["base_structural_benchmark_ready_score"] = 0.0
    tampered["reproducibility_checksum"] = mod.reproducibility_checksum(tampered)
    assert "base_structural_benchmark_ready_score mismatch" in mod.validate_artifact(tampered)

    tampered = deepcopy(artifact)
    tampered["exact_replay_rows"][0]["replay_passed"] = False
    tampered["per_unit_rows"] = mod.per_unit_rows(
        tampered["raw_instance_rows"],
        tampered["exact_label_rows"],
        tampered["exact_replay_rows"],
        tampered["split_commitment"]["rows"],
        tampered["stratum_balance_rows"],
        tampered["leakage_attack_matrix"]["rows"],
        tampered["minimum_held_cell_size"]["planned_headline_cell_rows"],
    )
    tampered["aggregate_row_recomputation"] = mod.recompute_aggregates_from_rows(
        tampered["per_unit_rows"]
    )
    tampered["gate_check_summary"] = mod.gate_check_summary(
        tampered["upstream_gate_receipts"],
        tampered["preconditions_checked"]["solver_tools"],
        tampered["aggregate_row_recomputation"],
        tampered["protected_files_unchanged"],
        tampered["tests_run"],
    )
    tampered["reproducibility_checksum"] = mod.reproducibility_checksum(tampered)
    assert tampered["base_structural_benchmark_ready_score"] == 1.0
    assert "base_structural_benchmark_ready_score mismatch" in mod.validate_artifact(tampered)

    validation_mutations = [
        ("required field set mismatch", lambda item: item.pop("status")),
        ("field_principles mismatch", lambda item: item.__setitem__("field_principles", {})),
        (
            "field_provenance must cover required fields",
            lambda item: item.__setitem__("field_provenance", {}),
        ),
        ("verdict_class outside closed enum", lambda item: item.__setitem__("verdict_class", "bad")),
        (
            "inference_substrate mismatch",
            lambda item: item.__setitem__("inference_substrate", "bad"),
        ),
        (
            "verifier_is_oracle must be true for exact labels",
            lambda item: item.__setitem__("verifier_is_oracle", False),
        ),
        (
            "aggregate_row_recomputation mismatch",
            lambda item: item["aggregate_row_recomputation"].__setitem__("row_count", -1),
        ),
        (
            "reproducibility_checksum mismatch",
            lambda item: item.__setitem__("reproducibility_checksum", "sha256:bad"),
        ),
        (
            "minimum_held_cell_size below 30",
            lambda item: item["minimum_held_cell_size"].__setitem__(
                "observed_minimum_held_units",
                29,
            ),
        ),
        (
            "leakage_attack_matrix false accepts",
            lambda item: item["leakage_attack_matrix"].__setitem__("false_accept_count", 1),
        ),
        (
            "honest_verdict lacks terminal prefix",
            lambda item: item.__setitem__("honest_verdict", "not terminal"),
        ),
    ]
    for expected, mutate in validation_mutations:
        broken = deepcopy(artifact)
        mutate(broken)
        assert expected in mod.validate_artifact(broken)


def test_scenario_bench_6504_main_and_validate_roundtrip(tmp_path: Path) -> None:
    """SCENARIO-BENCH-6504-SCHEMA: CLI writes and validates the artifact."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH.name
    assert mod.main(["--date", "20260822", "--result-path", str(result_path)]) == 0
    assert mod.main(["--validate", "--result-path", str(result_path)]) == 0
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["base_structural_benchmark_ready_score"] == 1.0
