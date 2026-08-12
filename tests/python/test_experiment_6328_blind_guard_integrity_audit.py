"""Tests for Exp6328 blind guard integrity audit.

Spec refs: REQ-SAFE-6328, SCENARIO-SAFE-6328-BLIND-ALLOWLIST,
SCENARIO-SAFE-6328-RECONSTRUCTION, SCENARIO-SAFE-6328-ATTACKS.
"""

from __future__ import annotations

from copy import deepcopy
import io
import json
import os
from pathlib import Path

import pytest

from carnot import experiment_6328_blind_guard_integrity_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/verifiable-reasoning/spec.md"
RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6328_blind_guard_integrity_audit "
    "--date 20260812"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6328_blind_guard_integrity_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6328_blind_guard_integrity_audit.py "
    "-m pytest tests/python/test_experiment_6328_blind_guard_integrity_audit.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6328_blind_guard_integrity_audit.py "
    "--fail-under=100 --show-missing"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6328_blind_guard_integrity_audit.py"
)
E2E_COMMAND = "sed -n '1,170p' ops/e2e-test-plan.md"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6328_blind_guard_integrity_audit.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
TEST_COMMANDS = [
    RUN_COMMAND,
    FOCUSED_TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    E2E_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
]
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


def test_req_safe_6328_spec_declares_blind_audit_contract() -> None:
    """REQ-SAFE-6328: OpenSpec anchors the blind audit and artifact fields."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-SAFE-6328") :]

    for marker in (
        "SCENARIO-SAFE-6328-BLIND-ALLOWLIST",
        "SCENARIO-SAFE-6328-RECONSTRUCTION",
        "SCENARIO-SAFE-6328-ATTACKS",
        "results/experiment_6328_blind_guard_integrity_audit.json",
        "solver prose",
        "arm identity",
        "utility_promotion_count",
    ):
        assert marker in section


def test_scenario_safe_6328_checker_allowlist_and_process_boundary() -> None:
    """SCENARIO-SAFE-6328-BLIND-ALLOWLIST: forbidden fields fail closed."""

    payload, _index = mod.build_blind_checker_input()
    assert set(payload) == {"schema", "cases"}
    assert all(set(case) == mod.CHECKER_CASE_FIELDS for case in payload["cases"])

    assert mod.forbidden_checker_input_fields(payload) == []

    in_process = mod.evaluate_blind_checker_input(payload)
    assert in_process["accepted"] is True
    assert in_process["forbidden_input_field_count"] == 0
    assert in_process["case_count"] == 24
    assert in_process["checker_pid"] == os.getpid()

    subprocess_result = mod.run_blind_checker_process(payload)
    assert subprocess_result["accepted"] is True
    assert subprocess_result["process_boundary"]["subprocess"] is True
    assert subprocess_result["process_boundary"]["checker_pid"] != os.getpid()

    leaked = deepcopy(payload)
    leaked["cases"][0]["arm_label"] = "one_raw_candidate"
    rejected = mod.evaluate_blind_checker_input(leaked)
    assert rejected["accepted"] is False
    assert rejected["decision"] == "reject"
    assert rejected["forbidden_input_field_count"] == 1
    assert "cases[0].arm_label" in rejected["forbidden_input_fields"]


def test_scenario_safe_6328_reconstructs_contract_factor_and_fallback_results(
    tmp_path: Path,
) -> None:
    """SCENARIO-SAFE-6328-RECONSTRUCTION: safety receipts are rederived."""

    artifact = mod.run(
        date="20260812",
        result_path=tmp_path / "artifact.json",
        schema_path=tmp_path / "checker_schema.json",
        attack_fixture_path=tmp_path / "attack_fixtures.json",
        duration_s=2.5,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )

    assert mod.validate_artifact(artifact) is True
    assert artifact["status"] == "complete_ready"
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert artifact["guard_integrity_ready_score"] == 1.0
    assert artifact["utility_promotion_count"] == 0
    assert type(artifact["utility_promotion_count"]) is int
    assert artifact["hidden_state_access_count"] == 0
    assert type(artifact["hidden_state_access_count"]) is int
    assert artifact["external_text_scorer_count"] == 0
    assert type(artifact["external_text_scorer_count"]) is int
    assert artifact["verifier_is_oracle"] is True
    assert set(artifact["field_principles"]) >= set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)

    reconstruction = artifact["reconstructed_contract_factor_and_fallback_results"]
    assert reconstruction["all_checker_cases_reconstructed"] is True
    assert reconstruction["checker_case_count"] == 24
    assert reconstruction["energy_histogram"] == {"0": 21, "1": 1, "2": 2}
    assert reconstruction["fallback_verified_family_count"] == 4
    assert reconstruction["upstream_safety_discrepancy_count"] == 0
    assert reconstruction["utility_ready_upstream"] is False
    assert reconstruction["safety_ready_even_when_utility_null"] is True


def test_scenario_safe_6328_mutation_and_leakage_attacks_fail_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-SAFE-6328-ATTACKS: high-severity attacks cannot pass."""

    payload, index = mod.build_blind_checker_input()
    baseline = mod.evaluate_blind_checker_input(payload)
    attack_results = mod.run_attack_suite(payload, index, baseline)

    contract_group = attack_results[
        "vacuous_contract_parser_default_fallback_laundering_spec_mutation_validator_mutation_test_deletion_and_hash_swap_results"
    ]
    leakage_group = attack_results[
        "label_pair_evaluator_duplicate_arm_leakage_rationale_leakage_budget_and_missing_cell_results"
    ]
    expected_contract_attacks = {
        "vacuous_contract",
        "parser_default",
        "fallback_laundering",
        "spec_mutation",
        "validator_mutation",
        "test_deletion",
        "hash_swap",
    }
    expected_leakage_attacks = {
        "label_swap",
        "pair_swap",
        "evaluator_swap",
        "duplicate_rows",
        "hidden_arm_labels",
        "solver_rationale_leakage",
        "budget_mismatch",
        "missing_cells",
    }

    assert set(contract_group["by_attack"]) == expected_contract_attacks
    assert set(leakage_group["by_attack"]) == expected_leakage_attacks
    for group in (contract_group, leakage_group):
        assert group["all_high_severity_attacks_failed_closed"] is True
        for result in group["by_attack"].values():
            assert result["severity"] == "high"
            assert result["final_attack_allowed"] is False
            assert result["decision"] in mod.FAIL_CLOSED_DECISIONS

    assert leakage_group["by_attack"]["label_swap"]["checker_decision_hash_equal_baseline"] is True
    assert leakage_group["by_attack"]["pair_swap"]["checker_decision_hash_equal_baseline"] is True
    assert leakage_group["by_attack"]["label_swap"]["checker_accepted_payload"] is True
    assert leakage_group["by_attack"]["pair_swap"]["checker_accepted_payload"] is True
    fixtures = mod.write_attack_fixtures(
        tmp_path / "attack_fixtures.json",
        attack_results=attack_results,
    )
    assert Path(fixtures["path"]).exists()
    assert fixtures["sha256"] == mod.sha256_file(Path(fixtures["path"]))


def test_req_safe_6328_false_readiness_and_forbidden_counts_reject(
    tmp_path: Path,
) -> None:
    """REQ-SAFE-6328: false readiness and non-bare zero counts reject."""

    artifact = mod.run(
        date="20260812",
        result_path=tmp_path / "artifact.json",
        schema_path=tmp_path / "checker_schema.json",
        attack_fixture_path=tmp_path / "attack_fixtures.json",
        duration_s=1.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=False,
    )

    bad_hidden = deepcopy(artifact)
    bad_hidden["hidden_state_access_count"] = False
    bad_hidden["reproducibility_checksum"] = mod.payload_checksum(bad_hidden)
    with pytest.raises(ValueError, match="hidden_state_access_count"):
        mod.validate_artifact(bad_hidden)

    bad_utility = deepcopy(artifact)
    bad_utility["utility_promotion_count"] = 1
    bad_utility["guard_integrity_ready_score"] = 1.0
    bad_utility["status"] = "complete_ready"
    bad_utility["honest_verdict"] = mod.honest_verdict(bad_utility)
    bad_utility["reproducibility_checksum"] = mod.payload_checksum(bad_utility)
    with pytest.raises(ValueError, match="utility_promotion_count"):
        mod.validate_artifact(bad_utility)

    bad_leak = deepcopy(artifact)
    bad_leak["information_asymmetry_receipts"]["forbidden_input_field_count"] = 1
    bad_leak["guard_integrity_ready_score"] = 1.0
    bad_leak["status"] = "complete_ready"
    bad_leak["honest_verdict"] = mod.honest_verdict(bad_leak)
    bad_leak["reproducibility_checksum"] = mod.payload_checksum(bad_leak)
    with pytest.raises(ValueError, match="guard_integrity_ready_score"):
        mod.validate_artifact(bad_leak)


def test_req_safe_6328_defensive_paths_and_cli_validate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-SAFE-6328: defensive checker paths and CLI validation are covered."""

    payload, index = mod.build_blind_checker_input()
    baseline = mod.evaluate_blind_checker_input(payload)

    bad_types = deepcopy(payload)
    bad_types["cases"][0]["exact_factor_evidence"] = "bad"
    bad_types["cases"][0]["fallback_hash"] = "bad"
    typed_reject = mod.evaluate_blind_checker_input(bad_types)
    assert typed_reject["accepted"] is False
    assert typed_reject["case_results"][0]["errors"] == [
        "exact_factor_evidence_type",
        "fallback_hash_type",
        "exact_factor_evidence_mismatch",
        "fallback_hash_mismatch",
    ]

    assert mod.forbidden_checker_input_fields([]) == ["$"]  # type: ignore[arg-type]
    assert mod.checker_input_field_paths([]) == []  # type: ignore[arg-type]
    assert mod.forbidden_checker_input_fields({"schema": mod.CHECKER_INPUT_SCHEMA, "extra": 1}) == [
        "extra"
    ]
    assert mod.forbidden_checker_input_fields({"schema": mod.CHECKER_INPUT_SCHEMA, "cases": "bad"}) == []
    bad_case = {"schema": mod.CHECKER_INPUT_SCHEMA, "cases": ["bad"]}
    assert mod.forbidden_checker_input_fields(bad_case) == ["cases[0]"]
    bad_outcome = deepcopy(payload)
    bad_outcome["cases"][0]["exact_factor_evidence"]["factor_outcomes"][0] = "bad"
    assert (
        "cases[0].exact_factor_evidence.factor_outcomes[0]"
        in mod.forbidden_checker_input_fields(bad_outcome)
    )

    class FailedProcess:
        returncode = 2
        stderr = "forced checker failure"
        stdout = ""

    with monkeypatch.context() as scoped:
        scoped.setattr(mod.subprocess, "run", lambda *args, **kwargs: FailedProcess())
        process_reject = mod.run_blind_checker_process(payload)
    assert process_reject["accepted"] is False
    assert process_reject["process_boundary"]["exit_code"] == 2

    stdin = io.StringIO(mod.canonical_json(payload))
    stdout = io.StringIO()
    with monkeypatch.context() as scoped:
        scoped.setattr(mod.sys, "stdin", stdin)
        scoped.setattr(mod.sys, "stdout", stdout)
        assert mod.checker_main() == 0
    assert json.loads(stdout.getvalue())["accepted"] is True

    bad_stdin = io.StringIO("[]")
    bad_stdout = io.StringIO()
    with monkeypatch.context() as scoped:
        scoped.setattr(mod.sys, "stdin", bad_stdin)
        scoped.setattr(mod.sys, "stdout", bad_stdout)
        assert mod.checker_main() == 0
    assert json.loads(bad_stdout.getvalue())["accepted"] is False

    cli_stdin = io.StringIO(mod.canonical_json(payload))
    cli_stdout = io.StringIO()
    with monkeypatch.context() as scoped:
        scoped.setattr(mod.sys, "stdin", cli_stdin)
        scoped.setattr(mod.sys, "stdout", cli_stdout)
        assert mod.main(["--checker"]) == 0
    assert json.loads(cli_stdout.getvalue())["accepted"] is True

    checker_rows = deepcopy(baseline["case_results"])
    checker_rows[0]["exact_energy"] = 99
    discrepant_checker = {**baseline, "case_results": checker_rows, "accepted": True}
    exp6326_payload = mod._load_json_object(REPO / mod.EXP6326_RELATIVE_PATH)
    exp6327_payload = mod._load_json_object(REPO / mod.EXP6327_RELATIVE_PATH)
    reconstruction = mod.reconstruct_safety_results(
        checker_result=discrepant_checker,
        checker_index=index,
        exp6326_payload=exp6326_payload,
        exp6327_payload=exp6327_payload,
    )
    assert reconstruction["upstream_safety_discrepancy_count"] > 0

    arm_payload = deepcopy(exp6327_payload)
    first_model = mod.exp6327.MANDATED_MODEL_IDS[0]
    arm_payload[
        "exact_utility_contract_violation_fallback_rate_latency_and_cost_by_model_family_arm_and_seed"
    ][first_model]["access_gate"]["one_raw_candidate"]["accepted"] = False
    arm_rebuild = mod.reconstruct_arm_safety(index, baseline["case_results"], arm_payload)
    assert arm_rebuild["discrepancy_count"] == 1

    bad_payload = deepcopy(payload)
    bad_payload["cases"][0]["arm_label"] = "forbidden"
    assert mod.attack_result("evaluator_swap", bad_payload, index, baseline)["decision"] == "reject"

    zero_payload = {"cases": [{"exact_factor_evidence": {"exact_energy": 0}}]}
    mod._mutate_validator_evidence(zero_payload)
    assert zero_payload["cases"][0]["exact_factor_evidence"]["exact_energy"] == 1

    same_hash_payload = {
        "cases": [{"canonical_contract_hash": "same"}, {"canonical_contract_hash": "same"}]
    }
    mod._swap_contract_hash(same_hash_payload)
    assert same_hash_payload["cases"][0]["canonical_contract_hash"] != "same"

    assert mod.sha256_file(tmp_path / "missing.json") is None
    bad_json = tmp_path / "not_object.json"
    bad_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="json_object"):
        mod._load_json_object(bad_json)

    artifact = mod.run(
        date="20260812",
        result_path=tmp_path / "cli_artifact.json",
        schema_path=tmp_path / "cli_schema.json",
        attack_fixture_path=tmp_path / "cli_attacks.json",
        duration_s=1.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )
    blocked = deepcopy(artifact)
    blocked["preconditions_checked"]["all_upstreams_terminal"] = False
    blocked["guard_integrity_ready_score"] = 0.0
    blocked["status"] = mod.status(blocked)
    assert blocked["status"] == "blocked"
    assert mod.honest_verdict(blocked).startswith("blocked:")
    null_artifact = deepcopy(artifact)
    null_artifact["guard_integrity_ready_score"] = 0.0
    null_artifact["status"] = mod.status(null_artifact)
    assert mod.honest_verdict(null_artifact).startswith("complete_null:")

    assert mod.main(
        [
            "--date",
            "20260812",
            "--result-path",
            str(tmp_path / "main_artifact.json"),
            "--schema-path",
            str(tmp_path / "main_schema.json"),
            "--attack-fixture-path",
            str(tmp_path / "main_attacks.json"),
        ]
    ) == 0
    assert mod.main(["--validate", "--result-path", str(tmp_path / "main_artifact.json")]) == 0
