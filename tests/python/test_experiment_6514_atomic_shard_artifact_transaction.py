"""Tests for Exp6514 atomic shard artifact transaction contract.

Spec refs: REQ-BENCH-6514, SCENARIO-BENCH-6514-SHARD-IDENTITY,
SCENARIO-BENCH-6514-PLANNED-TERMINAL, SCENARIO-BENCH-6514-RESUME-CRASHES,
SCENARIO-BENCH-6514-CORRUPT-QUARANTINE, SCENARIO-BENCH-6514-ATOMIC-REPLACE,
SCENARIO-BENCH-6514-CONCURRENCY, SCENARIO-BENCH-6514-CLOSED-FAILURE.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6514_atomic_shard_artifact_transaction as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH

FOCUSED_HELPER_COMMAND = (
    ".venv/bin/pytest tests/python/test_atomic_shard_transaction.py -q --no-cov -n 0"
)
FOCUSED_EXPERIMENT_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6514_atomic_shard_artifact_transaction.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/atomic_shard_transaction.py,python/carnot/experiment_6514_atomic_shard_artifact_transaction.py "
    "-m pytest tests/python/test_atomic_shard_transaction.py "
    "tests/python/test_experiment_6514_atomic_shard_artifact_transaction.py -q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/atomic_shard_transaction.py,python/carnot/experiment_6514_atomic_shard_artifact_transaction.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_atomic_shard_transaction.py "
    "tests/python/test_experiment_6514_atomic_shard_artifact_transaction.py"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6514_atomic_shard_artifact_transaction --date 20260822"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6514_atomic_shard_artifact_transaction.json"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6514_atomic_shard_artifact_transaction --validate"
)
GIT_STATUS_COMMAND = "git status --short"

TESTS_RUN = [
    {"command": FOCUSED_HELPER_COMMAND, "exit_code": 0},
    {"command": FOCUSED_EXPERIMENT_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 2},
    {"command": SPEC_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 1},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": GIT_STATUS_COMMAND, "exit_code": 0},
]


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """REQ-BENCH-6514: build a temp artifact without touching tracked results."""

    root = tmp_path_factory.mktemp("exp6514")
    return mod.build_artifact(
        repo_root=REPO,
        result_path=root / mod.RESULT_RELATIVE_PATH.name,
        work_root=root / "transactions",
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260822",
    )


def test_req_bench_6514_spec_declares_transaction_contract() -> None:
    """REQ-BENCH-6514: OpenSpec owns the transaction contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-BENCH-6514") :]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-BENCH-6514-SHARD-IDENTITY",
        "SCENARIO-BENCH-6514-PLANNED-TERMINAL",
        "SCENARIO-BENCH-6514-RESUME-CRASHES",
        "SCENARIO-BENCH-6514-CORRUPT-QUARANTINE",
        "SCENARIO-BENCH-6514-ATOMIC-REPLACE",
        "SCENARIO-BENCH-6514-CONCURRENCY",
        "SCENARIO-BENCH-6514-CLOSED-FAILURE",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_bench_6514_artifact_rows_and_score(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6514-RESUME-CRASHES: every proof row must pass."""

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete_atomic_shard_artifact_transaction_ready"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["verdict_class"] is None
    assert artifact["verdict_class"] != "positive"
    assert artifact["transaction_schema"] == mod.TRANSACTION_SCHEMA
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["atomic_artifact_contract_ready_score"] == 1.0
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert set(artifact["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["protected_files_unchanged"]["all_protected_files_unchanged"] is True
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)

    assert {row["stage"] for row in artifact["crash_injection_rows"]} == set(mod.CRASH_STAGES)
    assert all(row["passed"] is True for row in artifact["crash_injection_rows"])
    assert all(row["passed"] is True for row in artifact["recovery_rows"])
    assert all(row["passed"] is True for row in artifact["shard_integrity_rows"])
    assert all(row["passed"] is True for row in artifact["concurrency_attack_rows"])
    assert all(row["passed"] is True for row in artifact["gate_check_summary"])
    assert all(row["passed"] is True for row in artifact["per_unit_rows"])

    recompute = artifact["aggregate_row_recomputation"]
    assert recompute["all_crash_injection_rows_passed"] is True
    assert recompute["all_recovery_rows_passed"] is True
    assert recompute["all_shard_integrity_rows_passed"] is True
    assert recompute["all_concurrency_attack_rows_passed"] is True
    assert recompute["terminal_write_passed"] is True
    assert recompute["ready_score_from_rows"] == 1.0


def test_scenario_bench_6514_terminal_write_and_filesystem_receipts(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6514-ATOMIC-REPLACE: final JSON is closed and fsynced."""

    result_path = Path(artifact["preconditions_checked"]["result_path"])
    written = json.loads(result_path.read_text(encoding="utf-8"))
    terminal = artifact["terminal_write_receipt"]
    fs_receipt = artifact["filesystem_capability_receipt"]

    assert written == artifact
    assert mod.validate_artifact(artifact) == []
    assert terminal["final_path"] == str(result_path)
    assert terminal["final_path_status"] == "terminal_complete"
    assert terminal["atomic_replace"] is True
    assert terminal["file_fsync"] is True
    assert terminal["directory_fsync_attempted"] is True
    assert terminal["success_path_nonterminal_artifact"] is False
    assert terminal["final_sha256"].startswith("sha256:")
    assert fs_receipt["filesystem_type"]
    assert fs_receipt["output_root_writable"] is True
    assert fs_receipt["available_bytes"] > 0
    assert fs_receipt["process_model"]["pid"] > 0
    assert fs_receipt["protected_file_hashes_before"]["scripts/research_conductor.py"].startswith(
        "sha256:"
    )


def test_scenario_bench_6514_validation_fails_closed(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6514-CLOSED-FAILURE: invalid artifacts fail validation."""

    mutations = [
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
            "verdict_class outside Exp6514 enum",
            lambda item: item.__setitem__("verdict_class", "null"),
        ),
        (
            "transaction_schema mismatch",
            lambda item: item.__setitem__("transaction_schema", "bad.schema"),
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
            "atomic_artifact_contract_ready_score must be 0.0 or 1.0",
            lambda item: item.__setitem__("atomic_artifact_contract_ready_score", 0.5),
        ),
        (
            "ready score mismatch",
            lambda item: item["gate_check_summary"][0].__setitem__("passed", False),
        ),
        (
            "not every transaction proof row passed",
            lambda item: item["crash_injection_rows"][0].__setitem__("passed", False),
        ),
        (
            "ready score mismatch",
            lambda item: item.__setitem__("atomic_artifact_contract_ready_score", 0.0),
        ),
        (
            "success path left nonterminal artifact",
            lambda item: item["terminal_write_receipt"].__setitem__(
                "success_path_nonterminal_artifact", True
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
            lambda item: item.__setitem__("honest_verdict", "running"),
        ),
    ]
    for expected, mutate in mutations:
        broken = deepcopy(artifact)
        mutate(broken)
        assert expected in mod.validate_artifact(broken)


def test_scenario_bench_6514_main_and_validate_roundtrip(tmp_path: Path) -> None:
    """REQ-BENCH-6514: CLI writes and validates the transaction artifact."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH.name
    work_root = tmp_path / "work"

    assert (
        mod.main(
            [
                "--date",
                "20260822",
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

    assert payload["atomic_artifact_contract_ready_score"] == 1.0
    assert payload["status"] == "complete_atomic_shard_artifact_transaction_ready"
    assert payload["preconditions_checked"]["result_path"] == str(result_path)
    assert payload["aggregate_row_recomputation"]["ready_score_from_rows"] == 1.0

    invalid = deepcopy(payload)
    invalid["status"] = "running_bootstrap"
    invalid["reproducibility_checksum"] = mod.reproducibility_checksum(invalid)
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="status lacks terminal prefix"):
        mod.main(["--validate", "--result-path", str(invalid_path)])


def test_scenario_bench_6514_defensive_module_paths(
    artifact: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-BENCH-6514-CLOSED-FAILURE: blocked and defensive paths close."""

    assert mod.sha256_file(tmp_path / "missing.json") == "missing"

    status, honest, verdict_class = mod.status_and_verdict(
        0.0,
        [{"check": "forced", "passed": False}],
    )
    assert status == "blocked_atomic_shard_artifact_transaction"
    assert honest == "blocked_atomic_shard_artifact_transaction: forced"
    assert verdict_class == "blocked"
    assert mod.status_and_verdict(0.0, [])[1].endswith("unknown_gate")

    relative_root = tmp_path / "relative-repo"
    existing_work = relative_root / "relative-work"
    existing_work.mkdir(parents=True)
    relative_artifact = mod.build_artifact(
        repo_root=relative_root,
        result_path=Path("relative-result.json"),
        work_root=Path("relative-work"),
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260822",
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
            run_date="20260822",
        )

    monkeypatch.setattr(mod, "build_artifact", lambda **kwargs: deepcopy(artifact))
    with pytest.raises(ValueError, match="forced validation error"):
        mod.run(
            date="20260822",
            result_path=tmp_path / "invalid-run.json",
            work_root=tmp_path / "invalid-run-work",
        )
