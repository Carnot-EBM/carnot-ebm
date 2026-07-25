"""Tests for Exp5912 exact-slot CSL requalification.

Spec refs: REQ-LEARN-5912, SCENARIO-LEARN-5912-ATTRIBUTION,
SCENARIO-LEARN-5912-FROZEN-PARITY, SCENARIO-LEARN-5912-RETIREMENT,
SCENARIO-LEARN-5912-READY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5895_shortcut_safe_continuous_self_learning as exp5895
from carnot import experiment_5912_csl_exact_slot_requalification as mod


REPO = Path(__file__).resolve().parents[2]
SELF_LEARNING_SPEC = REPO / "openspec/capabilities/self-learning/spec.md"

COLLECTION_COMMAND = ".venv/bin/pytest tests/python --collect-only -q -n 0"
FOCUSED_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5912_csl_exact_slot_requalification.py -q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5912_csl_exact_slot_requalification.py "
    "-m pytest tests/python/test_experiment_5912_csl_exact_slot_requalification.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5912_csl_exact_slot_requalification.py "
    "--fail-under=100"
)
FULL_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5912_csl_exact_slot_requalification.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py --json "
    "results/experiment_5912_csl_exact_slot_requalification.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_COMMAND = (
    '.venv/bin/python -c "from pathlib import Path; '
    "assert Path('scripts/research_conductor.py').exists()\""
)
E2E_COMMAND = (
    ".venv/bin/python -m pytest "
    "tests/python/test_experiment_5895_shortcut_safe_continuous_self_learning.py "
    "-q --no-cov -n 0"
)
COMMANDS = [
    COLLECTION_COMMAND,
    FOCUSED_COMMAND,
    COVERAGE_COMMAND,
    FULL_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
    PROTECTED_COMMAND,
    E2E_COMMAND,
]


def _memory_probe() -> dict[str, Any]:
    return {"available_mb": 8192, "required_mb": 512, "ok": True}


def _disk_probe(_root: Path) -> dict[str, Any]:
    return {"available_mb": 8192, "required_mb": 512, "ok": True}


def _receipt(
    command: str,
    *,
    exit_code: int = 0,
    phase: str = "passed",
    node_ids: list[str] | None = None,
    ownership_paths: list[str] | None = None,
) -> dict[str, Any]:
    return mod.make_command_receipt(
        command=command,
        phase=phase,
        exit_code=exit_code,
        stdout=f"{command} stdout",
        stderr="" if exit_code == 0 else f"{command} stderr",
        node_ids=node_ids or [],
        ownership_paths=ownership_paths or ["tests/python/test_experiment_5912_csl_exact_slot_requalification.py"],
    )


def _receipts() -> list[dict[str, Any]]:
    return [
        _receipt(COLLECTION_COMMAND, phase="collection", ownership_paths=["tests/python"]),
        *[_receipt(command) for command in COMMANDS[1:]],
    ]


def _run_artifact(tmp_path: Path, receipts: list[dict[str, Any]]) -> dict[str, Any]:
    return mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        replay_result_path=tmp_path / "experiment_5895_replay_tmp.json",
        command_receipts=receipts,
        duration_s=3.5,
        changed_files=[
            "openspec/capabilities/self-learning/spec.md",
            "python/carnot/experiment_5912_csl_exact_slot_requalification.py",
            "tests/python/test_experiment_5912_csl_exact_slot_requalification.py",
            mod.RESULT_RELATIVE_PATH.as_posix(),
        ],
        memory_probe=_memory_probe,
        disk_probe=_disk_probe,
        write=True,
    )


def test_req_learn_5912_spec_declares_exact_requalification_contract() -> None:
    """REQ-LEARN-5912: OpenSpec declares fields, scenarios, and principles."""

    section = SELF_LEARNING_SPEC.read_text(encoding="utf-8")
    section = section[section.index("## REQ-LEARN-5912") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5912",
        "SCENARIO-LEARN-5912-ATTRIBUTION",
        "SCENARIO-LEARN-5912-FROZEN-PARITY",
        "SCENARIO-LEARN-5912-RETIREMENT",
        "SCENARIO-LEARN-5912-READY",
        "python/carnot/experiment_5912_csl_exact_slot_requalification.py",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`csl_exact_slot_ready_score`",
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_learn_5912_ready_artifact_wraps_frozen_exp5895(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-5912-READY: clean current commands requalify the exact slot."""

    artifact = _run_artifact(tmp_path, _receipts())
    loaded = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text(encoding="utf-8"))

    assert loaded == artifact
    assert mod.validate_artifact(artifact) is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete_positive"
    assert artifact["honest_verdict"].startswith("complete_positive:")
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["csl_exact_slot_ready_score"] == pytest.approx(1.0)
    assert isinstance(artifact["csl_exact_slot_ready_score"], float)
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["retired_dependency_chain_used"] is False
    assert artifact["protected_files_unchanged"]["all_unchanged"] is True
    assert artifact["historical_exp5895_hash_and_immutability"]["read_only"] is True
    assert artifact["original_test_failure_receipt"]["recorded_exit_code"] == 2
    assert artifact["current_failure_node_ids_phases_and_ownership"]["failing_node_count"] == 0
    assert artifact["causal_relevance_classification"]["can_alter_exp5895_science"] is False
    assert artifact["repeated_verdict_retirement_decision"]["same_verdict_recurred"] is False
    assert artifact["test_commands"] == COMMANDS
    assert artifact["test_exit_codes"] == {command: 0 for command in COMMANDS}

    frozen = artifact["frozen_rows_arms_seeds_budgets_thresholds_and_ready_logic"]
    assert frozen["row_count"] == 72
    assert frozen["arms"] == list(exp5895.ARM_NAMES)
    assert frozen["seeds"] == dict(exp5895.RANDOM_SEEDS)
    assert frozen["ready_logic_hash"].startswith("sha256:")
    assert frozen["scientific_inputs_changed"] is False

    parity = artifact["deterministic_science_parity"]
    assert parity["matches_historical"] is True
    assert parity["historical_science_hash"] == parity["replay_science_hash"]
    assert parity["temporary_replay_validates"] is True

    receipts = artifact["prospective_lift_retention_safety_rollback_and_state_receipts"]
    assert receipts["prospective_semantic_lift_ci95"][0] > 0.0
    assert receipts["retention"] == pytest.approx(1.0)
    assert receipts["unsafe_accept_count"] == 0
    assert receipts["restart_equivalence"] == pytest.approx(1.0)
    assert receipts["rollback_hash_mismatch_count"] == 0
    assert receipts["state_cap_compliance"] is True
    assert artifact["no_model_weight_mutation"]["all_unchanged"] is True
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert artifact["field_provenance"][field]["principle"] == principle


def test_scenario_learn_5912_repeated_null_retires_scope(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5912-RETIREMENT: repeated global-suite exit 2 retires the retry."""

    receipts = _receipts()
    receipts[COMMANDS.index(FULL_COMMAND)] = _receipt(
        FULL_COMMAND,
        exit_code=2,
        phase="runtime",
        node_ids=["tests/python/test_unrelated_suite_debt.py::test_current_debt"],
        ownership_paths=["tests/python/test_unrelated_suite_debt.py"],
    )
    artifact = _run_artifact(tmp_path, receipts)

    assert artifact["status"] == "retired"
    assert artifact["honest_verdict"].startswith("retired:")
    assert artifact["csl_exact_slot_ready_score"] == 0.0
    assert artifact["deterministic_science_parity"]["matches_historical"] is True
    assert artifact["current_failure_node_ids_phases_and_ownership"]["failing_node_count"] == 1
    assert artifact["current_failure_node_ids_phases_and_ownership"]["failures"][0]["node_ids"] == [
        "tests/python/test_unrelated_suite_debt.py::test_current_debt"
    ]
    assert artifact["causal_relevance_classification"]["can_alter_exp5895_science"] is False
    assert artifact["repeated_verdict_retirement_decision"]["same_verdict_recurred"] is True
    assert "failed_test_exit_codes" in artifact["repeated_verdict_retirement_decision"]["reasons"]
    assert mod.validate_artifact(artifact) is True


def test_scenario_learn_5912_fail_closed_for_science_or_integrity_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-LEARN-5912-FROZEN-PARITY: tampering blocks readiness."""

    artifact = _run_artifact(tmp_path, _receipts())

    mismatched = deepcopy(artifact)
    mismatched["deterministic_science_parity"]["matches_historical"] = False
    mismatched["deterministic_science_parity"]["replay_science_hash"] = mod.sha256_text("drift")
    mismatched["csl_exact_slot_ready_score"] = mod.csl_exact_slot_ready_score(mismatched)
    mismatched["status"] = mod.status(mismatched)
    mismatched["honest_verdict"] = mod.honest_verdict(mismatched)
    mismatched["reproducibility_checksum"] = mod.reproducibility_checksum(mismatched)
    assert mismatched["status"] == "blocked"
    assert "science_parity" in mod.blocked_reasons(mismatched)
    assert mod.validate_artifact(mismatched) is True

    current_owned_failure = mod.classify_causal_relevance(
        [
            _receipt(
                FULL_COMMAND,
                exit_code=1,
                phase="runtime",
                node_ids=[
                    "tests/python/test_experiment_5895_shortcut_safe_continuous_self_learning.py::test_failure"
                ],
                ownership_paths=[
                    "tests/python/test_experiment_5895_shortcut_safe_continuous_self_learning.py"
                ],
            )
        ]
    )
    assert current_owned_failure["can_alter_exp5895_science"] is True
    assert current_owned_failure["classification"] == "exp5895_causally_relevant_failure"

    combined = deepcopy(artifact)
    combined["continuous_self_learning_task"] = False
    combined["inference_substrate"] = "wrong"
    combined["verifier_is_oracle"] = False
    combined["deterministic_science_parity"]["matches_historical"] = False
    combined["causal_relevance_classification"]["can_alter_exp5895_science"] = True
    combined["historical_exp5895_hash_and_immutability"]["read_only"] = False
    combined["protected_files_unchanged"]["all_unchanged"] = False
    combined["no_model_weight_mutation"]["all_unchanged"] = False
    combined["prospective_lift_retention_safety_rollback_and_state_receipts"][
        "unsafe_accept_count"
    ] = 1
    combined["prospective_lift_retention_safety_rollback_and_state_receipts"][
        "restart_equivalence"
    ] = 0.0
    combined["prospective_lift_retention_safety_rollback_and_state_receipts"][
        "rollback_hash_mismatch_count"
    ] = 1
    combined["test_exit_codes"][FULL_COMMAND] = 1
    combined_reasons = set(mod.blocked_reasons(combined))
    assert {
        "continuous_self_learning_task",
        "inference_substrate",
        "verifier_is_oracle",
        "science_parity",
        "causally_relevant_exp5895_failure",
        "historical_exp5895_not_read_only",
        "protected_files_changed",
        "no_model_weight_mutation",
        "unsafe_accept_count",
        "restart_mismatch",
        "rollback_mismatch",
        "failed_test_exit_codes",
    } <= combined_reasons

    null_artifact = deepcopy(artifact)
    null_artifact["prospective_lift_retention_safety_rollback_and_state_receipts"][
        "prospective_semantic_lift_ci95"
    ] = [0.0, 0.0]
    null_artifact["csl_exact_slot_ready_score"] = mod.csl_exact_slot_ready_score(null_artifact)
    null_artifact["status"] = mod.status(null_artifact)
    null_artifact["honest_verdict"] = mod.honest_verdict(null_artifact)
    null_artifact["reproducibility_checksum"] = mod.reproducibility_checksum(null_artifact)
    assert null_artifact["status"] == "complete_null"
    assert null_artifact["honest_verdict"].startswith("complete_null:")
    assert "ready_score" in mod.blocked_reasons(null_artifact)
    assert mod.validate_artifact(null_artifact) is True

    unsafe = deepcopy(artifact)
    unsafe["no_model_weight_mutation"]["all_unchanged"] = False
    unsafe["no_model_weight_mutation"]["gguf_weight_mutation_count"] = 1
    unsafe["csl_exact_slot_ready_score"] = mod.csl_exact_slot_ready_score(unsafe)
    unsafe["status"] = mod.status(unsafe)
    unsafe["honest_verdict"] = mod.honest_verdict(unsafe)
    unsafe["reproducibility_checksum"] = mod.reproducibility_checksum(unsafe)
    assert unsafe["status"] == "unsafe"
    assert "no_model_weight_mutation" in mod.blocked_reasons(unsafe)
    assert mod.validate_artifact(unsafe) is True

    scalar_json = tmp_path / "scalar.json"
    scalar_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        mod.read_json(scalar_json)

    historical = mod.read_json(REPO / mod.HISTORICAL_EXP5895_RELATIVE_PATH)
    with monkeypatch.context() as scoped:
        scoped.setattr(
            mod.exp5895,
            "validate_artifact",
            lambda _item: (_ for _ in ()).throw(ValueError("replay bad")),
        )
        parity = mod._deterministic_science_parity(
            historical,
            historical,
            REPO / mod.HISTORICAL_EXP5895_RELATIVE_PATH,
        )
    assert parity["temporary_replay_validates"] is False

    missing = deepcopy(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        mod.validate_artifact(missing)

    for mutate, match in (
        (lambda item: item.update({"inference_substrate": "wrong"}), "inference_substrate"),
        (lambda item: item.update({"verifier_is_oracle": False}), "verifier_is_oracle"),
        (lambda item: item.update({"csl_exact_slot_ready_score": 0.0}), "ready_score"),
        (lambda item: item.update({"status": "blocked"}), "status"),
        (lambda item: item.update({"honest_verdict": "complete_positive: wrong"}), "honest_verdict"),
        (lambda item: item.update({"reproducibility_checksum": mod.sha256_text("wrong")}), "reproducibility_checksum"),
    ):
        bad = deepcopy(artifact)
        mutate(bad)
        if match != "reproducibility_checksum":
            bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
        with pytest.raises(ValueError, match=match):
            mod.validate_artifact(bad)

    bad_provenance = deepcopy(artifact)
    bad_provenance["field_provenance"]["status"]["principle"] = "wrong"
    with pytest.raises(ValueError, match="field_provenance:status"):
        mod.validate_artifact(bad_provenance)

    bad_type = deepcopy(artifact)
    bad_type["field_provenance"] = []
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(bad_type)
