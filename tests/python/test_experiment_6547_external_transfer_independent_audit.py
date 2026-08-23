"""Tests for Exp6547 external transfer independent audit.

Spec refs: REQ-BENCH-6547, SCENARIO-BENCH-6547-ALWAYS-RUN,
SCENARIO-BENCH-6547-ROW-REDUCTION, SCENARIO-BENCH-6547-MODEL-IDENTITY,
SCENARIO-BENCH-6547-SHORTCUTS, SCENARIO-BENCH-6547-EXACT-EQUALITY,
SCENARIO-BENCH-6547-COST-ACCOUNTING, SCENARIO-BENCH-6547-CALIBRATION,
SCENARIO-BENCH-6547-ATOMIC-OUTPUT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6547_external_transfer_independent_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6547_external_transfer_independent_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6547_external_transfer_independent_audit.py "
    "-m pytest tests/python/test_experiment_6547_external_transfer_independent_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6547_external_transfer_independent_audit.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6547_external_transfer_independent_audit.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6547_external_transfer_independent_audit.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6547_external_transfer_independent_audit.json"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6547_external_transfer_independent_audit --validate"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6547_external_transfer_independent_audit "
    "--date 20260823"
)
GIT_STATUS_COMMAND = "git status --short"

TESTS_RUN = [
    {"command": FOCUSED_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": SPEC_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": GIT_STATUS_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
]


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """REQ-BENCH-6547: build a temp audit from checked-in upstream artifacts."""

    root = tmp_path_factory.mktemp("exp6547")
    return mod.build_artifact(
        repo_root=REPO,
        result_path=root / mod.RESULT_RELATIVE_PATH.name,
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )


def test_req_bench_6547_spec_declares_independent_audit_contract() -> None:
    """REQ-BENCH-6547: OpenSpec owns the always-run audit contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-BENCH-6547") :]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-BENCH-6547-ALWAYS-RUN",
        "SCENARIO-BENCH-6547-ROW-REDUCTION",
        "SCENARIO-BENCH-6547-MODEL-IDENTITY",
        "SCENARIO-BENCH-6547-SHORTCUTS",
        "SCENARIO-BENCH-6547-EXACT-EQUALITY",
        "SCENARIO-BENCH-6547-COST-ACCOUNTING",
        "SCENARIO-BENCH-6547-CALIBRATION",
        "SCENARIO-BENCH-6547-ATOMIC-OUTPUT",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "external_transfer_audited_ready_score",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_bench_6547_clean_audit_recomputes_all_lanes(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6547-ROW-REDUCTION: clean verdict comes from rows."""

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete_external_transfer_independent_audit_clean"
    assert artifact["verdict_class"] is None
    assert "router=adopted_passed" in artifact["honest_verdict"]
    assert "cost_guard=adopted_passed" in artifact["honest_verdict"]
    assert artifact["external_transfer_audited_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True

    dispositions = artifact["input_disposition_rows"]
    assert {row["input_id"] for row in dispositions} == set(mod.UPSTREAM_INPUTS)
    assert all(row["exists"] and row["sha256"].startswith("sha256:") for row in dispositions)

    router = artifact["router_row_recomputation"]
    assert router["router_lane_passed"] is True
    assert router["structural"]["matched_row_count"] == 885
    assert router["structural"]["best_arm"] == "analytical"
    assert router["structural"]["best_arm_held_effect_vs_native_units"] == 546
    assert router["learned_router"]["matched_row_count"] == 1770
    assert (
        router["learned_router"]["selected_eligible_arm"]
        == "linear_compact_router_abstention_exception_exact_fallback"
    )
    assert router["learned_router"]["held_effect_vs_certified_control_units"] == 21.0
    assert router["learned_router"]["candidate_preservation_passed"] is True

    cost = artifact["cost_guard_row_recomputation"]
    assert cost["cost_guard_lane_passed"] is True
    assert cost["row_count"] == 36
    assert cost["supporting_model_family_count"] == 3
    assert cost["guarded_token_savings_total"] == 1607
    assert cost["token_and_time_totals_match_rows"] is True

    assert artifact["candidate_and_fallback_audit"]["passed"] is True
    assert artifact["token_time_and_tool_cost_audit"]["passed"] is True
    assert artifact["censoring_and_terminal_coverage"]["passed"] is True
    assert artifact["shortcut_attack_matrix"]["all_attacks_fail_closed"] is True
    assert artifact["lane_dispositions"]["router"]["readiness_score"] == 1.0
    assert artifact["lane_dispositions"]["cost_guard"]["readiness_score"] == 1.0
    assert artifact["gate_check_summary"]["all_gates_passed"] is True
    assert artifact["aggregate_row_recomputation"]["ready_score_from_rows"] == 1.0
    assert mod.validate_artifact(artifact) == []


def test_scenario_bench_6547_exact_replay_source_and_model_identity(
    artifact: dict[str, Any],
) -> None:
    """SCENARIO-BENCH-6547-MODEL-IDENTITY/EXACT-EQUALITY: receipts are checked."""

    replay = artifact["independent_exact_replay_rows"]
    assert len(replay) == 3
    assert {row["split_name"] for row in replay} == {"development", "held", "train"}
    assert all(row["passed"] is True for row in replay)
    assert all(row["fixture_exact_label"] == row["recomputed_exact_label"] for row in replay)

    source_rows = artifact["source_equivalence_rows"]
    assert source_rows
    assert all(row["passed"] for row in source_rows)
    assert {row["audit_type"] for row in source_rows} >= {
        "surface_equivalence",
        "fixture_source_identity",
    }

    models = artifact["model_identity_audit_rows"]
    assert len(models) == len(mod.MANDATED_HF_IDS)
    assert [row["model_hf_id"] for row in models] == list(mod.MANDATED_HF_IDS)
    assert all(row["passed"] for row in models)
    assert all(row["loader"] == "llama_cpp.Llama" for row in models)
    assert all(row["model_path_exists"] for row in models)
    assert all(row["gguf_sha256"].startswith("sha256:") for row in models)

    receipts = artifact["exception_and_fixture_hash_receipts"]
    assert receipts["fixture_sha256"] == mod.sha256_file(REPO / mod.FIXTURE_RELATIVE_PATH)
    assert receipts["exception_table_hash"].startswith("sha256:")
    assert receipts["checkpoint_sha256"].startswith("sha256:")
    assert receipts["protected_file_hashes_before"]["scripts/research_conductor.py"].startswith(
        "sha256:"
    )


def test_scenario_bench_6547_missing_cost_guard_lane_is_partial(
    tmp_path: Path,
) -> None:
    """SCENARIO-BENCH-6547-ALWAYS-RUN: missing lanes still emit diagnostics."""

    paths = mod.default_input_paths(REPO)
    paths["cost_guard"] = tmp_path / "missing-cost-guard.json"
    partial = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "partial.json",
        input_paths=paths,
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )

    assert partial["status"] == "partial_external_transfer_independent_audit"
    assert partial["verdict_class"] == "partial"
    assert partial["external_transfer_audited_ready_score"] == 1.0
    assert partial["lane_dispositions"]["router"]["disposition"] == "adopted_passed"
    assert partial["lane_dispositions"]["cost_guard"]["disposition"] == "blocked_missing_input"
    assert partial["lane_dispositions"]["cost_guard"]["readiness_score"] == 0.0
    assert any(
        row["input_id"] == "cost_guard" and row["disposition"] == "missing"
        for row in partial["input_disposition_rows"]
    )
    assert "cost_guard_input_present" in partial["gate_check_summary"]["failed_checks"]
    assert mod.validate_artifact(partial) == []


def test_scenario_bench_6547_cost_accounting_tamper_disqualifies(
    tmp_path: Path,
) -> None:
    """SCENARIO-BENCH-6547-COST-ACCOUNTING: row tampering beats aggregates."""

    payloads = mod.load_input_payloads(REPO)
    payloads["cost_guard"] = deepcopy(payloads["cost_guard"])
    payloads["cost_guard"]["per_unit_rows"][0]["charged_tokens"] += 1

    audit = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "tampered-cost.json",
        input_payloads=payloads,
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )

    assert audit["status"] == "disqualified_external_transfer_independent_audit"
    assert audit["verdict_class"] == "disqualified"
    assert audit["external_transfer_audited_ready_score"] == 0.0
    assert audit["cost_guard_row_recomputation"]["token_and_time_totals_match_rows"] is False
    assert audit["token_time_and_tool_cost_audit"]["passed"] is False
    assert "cost_guard_accounting_passed" in audit["gate_check_summary"]["failed_checks"]
    assert mod.validate_artifact(audit) == []


def test_scenario_bench_6547_model_identity_tamper_disqualifies(
    tmp_path: Path,
) -> None:
    """SCENARIO-BENCH-6547-MODEL-IDENTITY: model substitution closes the audit."""

    payloads = mod.load_input_payloads(REPO)
    payloads["cost_guard"] = deepcopy(payloads["cost_guard"])
    payloads["cost_guard"]["MODEL_SPECS"][0]["hf_id"] = "not/a-mandated-gguf"

    audit = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "tampered-model.json",
        input_payloads=payloads,
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )

    assert audit["verdict_class"] == "disqualified"
    assert audit["external_transfer_audited_ready_score"] == 0.0
    assert audit["model_identity_audit_rows"][0]["passed"] is False
    assert "model_identity_passed" in audit["gate_check_summary"]["failed_checks"]


def test_scenario_bench_6547_router_fallback_tamper_disqualifies(
    tmp_path: Path,
) -> None:
    """SCENARIO-BENCH-6547-CALIBRATION/EXACT-EQUALITY: router safety is checked."""

    payloads = mod.load_input_payloads(REPO)
    payloads["router"] = deepcopy(payloads["router"])
    payloads["router"]["per_unit_rows"][0]["fallback_available"] = False

    audit = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / "tampered-router.json",
        input_payloads=payloads,
        write=False,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        run_date="20260823",
    )

    assert audit["verdict_class"] == "disqualified"
    assert audit["external_transfer_audited_ready_score"] == 0.0
    assert audit["candidate_and_fallback_audit"]["fallback_reachability_passed"] is False
    assert "router_candidate_fallback_passed" in audit["gate_check_summary"]["failed_checks"]


def test_scenario_bench_6547_atomic_output_and_cli_validation(
    tmp_path: Path,
) -> None:
    """SCENARIO-BENCH-6547-ATOMIC-OUTPUT: CLI writes and validates the artifact."""

    result_path = tmp_path / "cli-exp6547.json"
    exit_code = mod.main(
        [
            "--date",
            "20260823",
            "--result-path",
            str(result_path),
            "--duration-s",
            "1.0",
            "--skip-default-tests-run",
        ]
    )
    assert exit_code == 0
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["reproducibility_checksum"] == mod.reproducibility_checksum(payload)
    assert payload["protected_files_unchanged"]["all_unchanged"] is True

    validate_exit = mod.main(["--result-path", str(result_path), "--validate"])
    assert validate_exit == 0
