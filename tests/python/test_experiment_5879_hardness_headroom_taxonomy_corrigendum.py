"""Tests for Exp5879 hardness-headroom taxonomy corrigendum.

Spec refs: REQ-VERIFY-5879, SCENARIO-VERIFY-5879-TAXONOMY,
SCENARIO-VERIFY-5879-HEADROOM, SCENARIO-VERIFY-5879-BLOCKED-DEBT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5879_hardness_headroom_taxonomy_corrigendum as mod


REPO = Path(__file__).resolve().parents[2]
VERIFY_SPEC = REPO / "openspec/capabilities/verification/spec.md"
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5879_hardness_headroom_taxonomy_corrigendum.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5879_hardness_headroom_taxonomy_corrigendum.py "
    "-m pytest tests/python/test_experiment_5879_hardness_headroom_taxonomy_corrigendum.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5879_hardness_headroom_taxonomy_corrigendum.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5879_hardness_headroom_taxonomy_corrigendum.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5879_hardness_headroom_taxonomy_corrigendum.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_FILE_COMMAND = (
    '.venv/bin/python -c "from pathlib import Path; '
    "assert Path('scripts/research_conductor.py').exists()\""
)
TEST_COMMANDS = [
    FOCUSED_TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
    PROTECTED_FILE_COMMAND,
]
TEST_EXIT_CODES = {
    FOCUSED_TEST_COMMAND: 0,
    COVERAGE_COMMAND: 0,
    FULL_TEST_COMMAND: 2,
    SPEC_COMMAND: 0,
    ADVERSARIAL_COMMAND: 0,
    ROOT_CLUTTER_COMMAND: 0,
    PROTECTED_FILE_COMMAND: 0,
}


def _preconditions(tmp_path: Path) -> dict[str, Any]:
    return mod.collect_preconditions(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        memory_probe=lambda: {"available_mb": 32768, "required_mb": 1024, "ok": True},
        disk_probe=lambda root: {"available_mb": 32768, "required_mb": 512, "ok": True},
    )


@pytest.fixture(scope="module")
def exp5879_artifact(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, Any], Path]:
    """REQ-VERIFY-5879: build the corrigendum artifact once."""

    base = tmp_path_factory.mktemp("exp5879")
    conductor = REPO / "scripts/research_conductor.py"
    before_hash = mod.sha256_file(conductor)
    artifact = mod.run(
        result_path=base / mod.RESULT_RELATIVE_PATH.name,
        preconditions_checked=_preconditions(base),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=9.0,
        write=True,
    )
    assert mod.sha256_file(conductor) == before_hash
    return artifact, base


def test_req_verify_5879_spec_declares_taxonomy_corrigendum_contract() -> None:
    """REQ-VERIFY-5879: OpenSpec anchors every required field and principle."""

    text = VERIFY_SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-VERIFY-5879") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5879",
        "SCENARIO-VERIFY-5879-TAXONOMY",
        "SCENARIO-VERIFY-5879-HEADROOM",
        "SCENARIO-VERIFY-5879-BLOCKED-DEBT",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`non_oracle_nuisance_control_metrics`",
        "`oracle_derived_diagnostic_metrics`",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for principle in mod.REQUIRED_FIELD_PRINCIPLES.values():
        assert " ".join(principle.split()) in normalized


def test_req_verify_5879_terminal_artifact_schema_hashes_and_debt_block(
    exp5879_artifact: tuple[dict[str, Any], Path],
) -> None:
    """REQ-VERIFY-5879: terminal artifact is hash-bound and honestly blocked."""

    artifact, base = exp5879_artifact
    rerun = mod.run(
        result_path=base / "rerun.json",
        preconditions_checked=_preconditions(base / "rerun"),
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        duration_s=99.0,
        write=True,
    )

    assert mod.validate_artifact(artifact) is True
    assert json.loads((base / mod.RESULT_RELATIVE_PATH.name).read_text()) == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["hardness_surface_headroom_ready_score"] == 1.0
    assert artifact["duration_s"] == pytest.approx(9.0)
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["test_exit_codes"][FULL_TEST_COMMAND] == 2
    assert artifact["test_debt_classification"]["science_matrix_ready"] is True
    assert artifact["test_debt_classification"]["unrelated_global_suite_debt"] is True
    assert artifact["test_debt_classification"]["owned_checks_passed"] is True
    assert artifact["reproducibility_checksum"] == rerun["reproducibility_checksum"]
    assert artifact["protected_files_unchanged"]["all_unchanged"] is True

    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert artifact["field_provenance"][field]["principle"] == principle


def test_scenario_verify_5879_taxonomy_classes_are_mutually_exclusive(
    exp5879_artifact: tuple[dict[str, Any], Path],
) -> None:
    """SCENARIO-VERIFY-5879-TAXONOMY: each evaluated feature has one class."""

    artifact, _base = exp5879_artifact
    taxonomy = artifact["control_taxonomy"]
    non_oracle = artifact["non_oracle_nuisance_control_metrics"]
    oracle = artifact["oracle_derived_diagnostic_metrics"]

    assert taxonomy["all_features_assigned_once"] is True
    assert taxonomy["class_overlap"] == []
    assert taxonomy["excluded_label_proxy_features"] == [
        "certificate_kind_as_label_proxy",
        "expected_label",
        "solver_result_label",
    ]
    assert set(non_oracle["control_names"]) == set(taxonomy["classes"]["non_oracle_nuisance"])
    assert set(oracle["control_names"]) == set(taxonomy["classes"]["oracle_derived_diagnostic"])
    assert "solver_conflicts" not in non_oracle["control_metrics"]
    assert "solver_conflicts" in oracle["control_metrics"]
    assert "shuffled_label_control" in non_oracle["control_metrics"]
    assert "majority_control" in non_oracle["control_metrics"]
    assert all(
        metric["authority_class"] == "non_oracle_nuisance"
        for metric in non_oracle["control_metrics"].values()
    )
    assert all(
        metric["authority_class"] == "oracle_derived_diagnostic"
        for metric in oracle["control_metrics"].values()
    )


def test_scenario_verify_5879_headroom_uses_only_non_oracle_controls(
    exp5879_artifact: tuple[dict[str, Any], Path],
) -> None:
    """SCENARIO-VERIFY-5879-HEADROOM: oracle telemetry cannot saturate headroom."""

    artifact, _base = exp5879_artifact
    non_oracle = artifact["non_oracle_nuisance_control_metrics"]
    oracle = artifact["oracle_derived_diagnostic_metrics"]
    decision = artifact["saturation_and_skip_decision"]

    assert non_oracle["saturation_ceiling_auroc"] == pytest.approx(0.85)
    assert non_oracle["max_non_oracle_nuisance_auroc"] == pytest.approx(0.583333)
    assert non_oracle["no_non_oracle_nuisance_control_exceeds_ceiling"] is True
    assert non_oracle["saturated_control_names"] == []
    assert oracle["max_oracle_derived_auroc"] == pytest.approx(1.0)
    assert set(oracle["saturated_diagnostic_names"]) >= {
        "solver_conflicts",
        "solver_decisions",
        "solver_time",
    }
    assert oracle["counts_as_learned_energy_win"] is False
    assert oracle["reduces_oracle_distinct_headroom"] is False
    assert decision["skip_model_extraction_for_science"] is False
    assert decision["no_non_oracle_nuisance_control_exceeds_ceiling"] is True
    assert decision["hardness_surface_headroom_ready_score"] == 1.0


def test_scenario_verify_5879_replays_integrity_splits_and_stability(
    exp5879_artifact: tuple[dict[str, Any], Path],
) -> None:
    """REQ-VERIFY-5879: labels, certificates, splits, and relabel stability replay."""

    artifact, _base = exp5879_artifact
    replay = artifact["independent_row_integrity_replay"]
    splits = artifact["leakage_safe_split_receipts"]
    stability = artifact["relabel_and_certificate_stability"]
    design = artifact["oracle_distinct_evaluation_design"]
    held = artifact["held_model_and_constraint_plan"]

    assert replay["row_count"] == 84
    assert replay["all_integrity_checks_passed"] is True
    assert replay["exact_label_disagreement_count"] == 0
    assert replay["certificate_failure_count"] == 0
    assert replay["relabel_equivalence_failure_count"] == 0
    assert replay["label_counts"] == {"satisfiable": 42, "unsatisfiable": 42}
    assert splits["all_splits_leakage_safe"] is True
    assert splits["duplicate_semantic_instances_across_splits"] == []
    assert stability["relabel_stability_rate"] == 1.0
    assert stability["certificate_stability_rate"] == 1.0
    assert stability["all_group_controls_passed"] is True
    assert design["nonempty_held_model_and_constraint_design"] is True
    assert held["nonempty_plan"] is True


def test_scenario_verify_5879_defensive_validation_branches(
    tmp_path: Path,
    exp5879_artifact: tuple[dict[str, Any], Path],
) -> None:
    """REQ-VERIFY-5879: checksum, taxonomy, score, and verdict edits fail closed."""

    artifact, _base = exp5879_artifact

    missing_preconditions = mod.collect_preconditions(
        root=tmp_path / "missing",
        result_path=tmp_path / "missing" / mod.RESULT_RELATIVE_PATH.name,
        memory_probe=lambda: {"available_mb": 32768, "required_mb": 1024, "ok": True},
        disk_probe=lambda root: {"available_mb": 32768, "required_mb": 512, "ok": True},
    )
    assert "missing_exp5879_module_test_or_spec" in missing_preconditions["blocked_reasons"]

    checksum_bad = deepcopy(artifact)
    checksum_bad["honest_verdict"] = "blocked: edited"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(checksum_bad)

    missing = deepcopy(artifact)
    missing.pop("control_taxonomy")
    with pytest.raises(ValueError, match="missing_fields"):
        mod.validate_artifact(missing)

    overlap = deepcopy(artifact)
    overlap["control_taxonomy"]["classes"]["non_oracle_nuisance"].append("solver_conflicts")
    overlap["reproducibility_checksum"] = mod.reproducibility_checksum(overlap)
    with pytest.raises(ValueError, match="control_taxonomy"):
        mod.validate_artifact(overlap)

    saturated = deepcopy(artifact)
    saturated["non_oracle_nuisance_control_metrics"]["control_metrics"]["row_order"][
        "orientation_free_auroc"
    ] = 0.99
    saturated["non_oracle_nuisance_control_metrics"]["max_non_oracle_nuisance_auroc"] = 0.99
    saturated["non_oracle_nuisance_control_metrics"][
        "no_non_oracle_nuisance_control_exceeds_ceiling"
    ] = False
    saturated["saturation_and_skip_decision"] = mod.saturation_and_skip_decision(
        saturated["independent_row_integrity_replay"],
        saturated["leakage_safe_split_receipts"],
        saturated["non_oracle_nuisance_control_metrics"],
        saturated["relabel_and_certificate_stability"],
        saturated["held_model_and_constraint_plan"],
    )
    saturated["hardness_surface_headroom_ready_score"] = mod.hardness_surface_headroom_ready_score(
        saturated
    )
    saturated["status"] = mod.status(saturated)
    saturated["honest_verdict"] = mod.honest_verdict(saturated)
    saturated["reproducibility_checksum"] = mod.reproducibility_checksum(saturated)
    assert saturated["hardness_surface_headroom_ready_score"] == 0.0
    assert saturated["status"] == "complete_null"
    assert mod.validate_artifact(saturated) is True

    ready = deepcopy(artifact)
    ready["test_exit_codes"][FULL_TEST_COMMAND] = 0
    ready["test_debt_classification"] = mod.classify_test_debt(
        TEST_COMMANDS,
        ready["test_exit_codes"],
        science_matrix_ready=True,
    )
    ready["status"] = mod.status(ready)
    ready["honest_verdict"] = mod.honest_verdict(ready)
    ready["reproducibility_checksum"] = mod.reproducibility_checksum(ready)
    assert ready["status"] == "complete_ready"
    assert ready["honest_verdict"].startswith("complete_ready:")
    assert mod.validate_artifact(ready) is True

    owned_failed = deepcopy(artifact)
    owned_failed["test_exit_codes"][FOCUSED_TEST_COMMAND] = 1
    owned_failed["test_debt_classification"] = mod.classify_test_debt(
        TEST_COMMANDS,
        owned_failed["test_exit_codes"],
        science_matrix_ready=True,
    )
    assert mod.status(owned_failed) == "blocked"
    assert mod.honest_verdict(owned_failed) == "blocked: owned_test_exit_codes"

    score_bad = deepcopy(artifact)
    score_bad["hardness_surface_headroom_ready_score"] = 0.0
    with pytest.raises(ValueError, match="hardness_surface_headroom_ready_score"):
        mod.validate_artifact(score_bad)

    status_bad = deepcopy(artifact)
    status_bad["status"] = "complete_ready"
    status_bad["reproducibility_checksum"] = mod.reproducibility_checksum(status_bad)
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(status_bad)

    verdict_bad = deepcopy(artifact)
    verdict_bad["honest_verdict"] = "blocked: wrong"
    verdict_bad["reproducibility_checksum"] = mod.reproducibility_checksum(verdict_bad)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(verdict_bad)
