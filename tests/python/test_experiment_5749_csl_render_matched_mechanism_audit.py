"""Tests for Exp5749 CSL render-matched KAN mechanism audit.

Spec refs: REQ-LEARN-5749,
SCENARIO-LEARN-5749-MATCHED-CONTROLS,
SCENARIO-LEARN-5749-RENDER,
SCENARIO-LEARN-5749-NONFORGETTING,
SCENARIO-LEARN-5749-RELEASE.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5749_csl_render_matched_mechanism_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5749_csl_render_matched_mechanism_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5749_csl_render_matched_mechanism_audit.py "
    "-m pytest tests/python/test_experiment_5749_csl_render_matched_mechanism_audit.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5749_csl_render_matched_mechanism_audit.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5749_csl_render_matched_mechanism_audit.json"
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


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """REQ-LEARN-5749: build the audit artifact once for schema tests."""

    base = tmp_path_factory.mktemp("exp5749")
    return mod.run(
        root=REPO,
        result_path=base / mod.RESULT_RELATIVE_PATH.name,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )


def test_req_learn_5749_spec_declares_render_matched_mechanism_audit() -> None:
    """REQ-LEARN-5749: OpenSpec anchors fields, controls, and residual math."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5749") : spec.index("## REQ-LEARN-5640")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5749",
        "SCENARIO-LEARN-5749-MATCHED-CONTROLS",
        "SCENARIO-LEARN-5749-RENDER",
        "SCENARIO-LEARN-5749-NONFORGETTING",
        "SCENARIO-LEARN-5749-RELEASE",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "best_matched_non_kan_suffix_error - kan_suffix_error_after_all_safety_and_retention_gates",
        "deprecation-enabled and deprecation-disabled ledger views",
        "corrupted-order",
        "rejected-update propagation",
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_learn_5749_matched_controls_and_signed_residual(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-LEARN-5749-MATCHED-CONTROLS: controls share budgets and residual is signed."""

    assert mod.validate_artifact(artifact) is True
    assert artifact["session_count"] >= 30
    assert set(artifact["control_definitions"]) == set(mod.CONTROL_ARMS)
    assert set(artifact["suffix_error_by_arm"]) == set(mod.CONTROL_ARMS)
    assert set(artifact["dynamic_regret_by_arm"]) == set(mod.CONTROL_ARMS)
    assert set(artifact["recovery_time_by_arm"]) == set(mod.CONTROL_ARMS)

    update_receipts = artifact["update_count_match_receipts"]
    parameter_receipts = artifact["parameter_match_receipts"]
    chronology_receipts = artifact["chronology_receipts"]
    assert update_receipts["all_active_arms_matched"] is True
    assert parameter_receipts["parameter_budget_matched"] is True
    assert chronology_receipts["all_headline_rows_replayed_once"] is True
    assert chronology_receipts["corrupted_order_detected"] is True

    suffix_errors = artifact["suffix_error_by_arm"]
    assert suffix_errors[mod.KAN_HEADLINE_ARM] == pytest.approx(0.146067)
    assert suffix_errors[mod.MLP_CONTROL_ARM] == pytest.approx(0.061798)
    assert artifact["kan_mechanism_residual_definition"] == mod.KAN_MECHANISM_RESIDUAL_DEFINITION
    assert artifact["kan_mechanism_residual"] == pytest.approx(-0.084269)
    assert artifact["kan_mechanism_residual"] == pytest.approx(
        suffix_errors[mod.MLP_CONTROL_ARM] - suffix_errors[mod.KAN_HEADLINE_ARM]
    )
    assert artifact["dynamic_regret_by_arm"][mod.KAN_HEADLINE_ARM] == pytest.approx(0.084269)
    assert artifact["arm_metrics"][mod.KAN_HEADLINE_ARM]["mechanism_family"] == "kan"
    assert artifact["arm_metrics"][mod.MLP_CONTROL_ARM]["mechanism_family"] == "non_kan_mlp"


def test_scenario_learn_5749_render_matching_and_controls(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-LEARN-5749-RENDER: deprecation views are presentation matched."""

    render = artifact["render_match_receipts"]
    assert render["all_passed"] is True
    assert render["deprecation_enabled"]["text_length"] == render["deprecation_disabled"]["text_length"]
    assert render["deprecation_enabled"]["field_order_hash"] == render["deprecation_disabled"]["field_order_hash"]
    assert render["deprecation_enabled"]["status_marker_hash"] == render["deprecation_disabled"]["status_marker_hash"]
    assert render["deprecation_enabled"]["candidate_availability_hash"] == render["deprecation_disabled"]["candidate_availability_hash"]
    assert render["matched_receipt_hash"].startswith("sha256:")
    assert artifact["chronology_receipts"]["corrupted_order_detected"] is True
    assert artifact["rejected_update_propagation_count"] == 0
    assert artifact["unsafe_update_count"] == 0


def test_scenario_learn_5749_nonforgetting_and_release_gates(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-LEARN-5749-NONFORGETTING: CerCE certificate gates scale-up."""

    certificate = artifact["nonforgetting_certificate"]
    assert certificate["certificate_style"] == "CerCE_exact_nonforgetting"
    assert certificate["all_passed"] is True
    assert certificate["protected_prefix_count"] >= 30
    assert certificate["lifecycle_state_count"] >= 1
    assert certificate["protected_prefix_mismatch_count"] == 0
    assert certificate["lifecycle_state_mismatch_count"] == 0
    assert certificate["rejected_update_zero_propagation"] is True
    assert certificate["rollback_hash_mismatch_count"] == 0
    assert artifact["rollback_hash_mismatch_count"] == 0
    assert artifact["prefix_retention_delta"] <= 0.0
    assert artifact["prefix_retention_pass_score"] == 1.0
    assert artifact["continuous_self_learning_target"] is True
    assert artifact["continuous_self_learning_credited"] is True
    assert artifact["kan_scaleup_ready_score"] == 0.0
    assert artifact["honest_verdict"] == "complete: kan_mechanism_residual_negative_fr11_safety_retained"
    assert artifact["model_weight_mutation"] is False
    assert artifact["production_default_enabled"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE


def test_req_learn_5749_artifact_fields_hashes_and_replay(
    artifact: dict[str, object],
    tmp_path: Path,
) -> None:
    """REQ-LEARN-5749: run output, hashes, and checksums replay exactly."""

    destination = tmp_path / mod.RESULT_RELATIVE_PATH.name
    written = mod.run(
        root=REPO,
        result_path=destination,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )
    loaded = json.loads(destination.read_text(encoding="utf-8"))

    assert loaded == written
    assert mod.validate_artifact(written) is True
    assert written["reproducibility_checksum"] == mod.reproducibility_checksum(written)
    assert written["operation_ledger_hash"].startswith("sha256:")
    assert written["stream_hashes"] == artifact["stream_hashes"]
    assert written["test_commands"] == TEST_COMMANDS
    assert written["test_exit_codes"] == TEST_EXIT_CODES
    assert all(receipt["verified"] is True for receipt in artifact["upstream_artifact_hashes"].values())
    assert artifact["preconditions_checked"]["all_passed"] is True

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
        assert artifact["field_principles"][field] == mod.REQUIRED_FIELD_PRINCIPLES[field]
    for field in artifact:
        assert field in artifact["field_principles"]


def test_req_learn_5749_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-LEARN-5749: checked-in artifact is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = mod.run(
        root=REPO,
        result_path=RESULT_PATH,
        test_commands=result["test_commands"],
        test_exit_codes=result["test_exit_codes"],
        write=False,
    )

    assert result == replay
    assert result["honest_verdict"].startswith("complete:")
    assert result["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert result["kan_mechanism_residual"] <= 0.0
    mod.validate_artifact(result)


def test_req_learn_5749_validation_fails_closed(artifact: dict[str, object]) -> None:
    """REQ-LEARN-5749: artifact validation rejects unsafe or stale claims."""

    cases: list[tuple[str, dict[str, object]]] = []
    for field, value, expected in (
        ("unsafe_update_count", 1, "unsafe_update_count"),
        ("rejected_update_propagation_count", 1, "rejected_update_propagation_count"),
        ("rollback_hash_mismatch_count", 1, "rollback_hash_mismatch_count"),
        ("prefix_retention_delta", 0.1, "prefix_retention_delta"),
        ("model_weight_mutation", True, "model_weight_mutation"),
        ("production_default_enabled", True, "production_default_enabled"),
        ("continuous_self_learning_target", False, "continuous_self_learning_target"),
        ("continuous_self_learning_credited", False, "continuous_self_learning_credited"),
    ):
        bad = deepcopy(artifact)
        bad[field] = value
        cases.append((expected, bad))

    bad = deepcopy(artifact)
    bad["preconditions_checked"]["all_passed"] = False
    cases.append(("preconditions_checked", bad))

    bad = deepcopy(artifact)
    bad["render_match_receipts"]["all_passed"] = False
    cases.append(("render_match_receipts", bad))

    bad = deepcopy(artifact)
    bad["parameter_match_receipts"]["parameter_budget_matched"] = False
    cases.append(("parameter_match_receipts", bad))

    bad = deepcopy(artifact)
    bad["update_count_match_receipts"]["all_active_arms_matched"] = False
    cases.append(("update_count_match_receipts", bad))

    bad = deepcopy(artifact)
    bad["chronology_receipts"]["all_headline_rows_replayed_once"] = False
    cases.append(("chronology_receipts", bad))

    bad = deepcopy(artifact)
    bad["nonforgetting_certificate"]["all_passed"] = False
    cases.append(("nonforgetting_certificate", bad))

    bad = deepcopy(artifact)
    bad["field_principles"].pop("kan_mechanism_residual")
    cases.append(("field_principles", bad))

    bad = deepcopy(artifact)
    bad.pop("kan_mechanism_residual")
    cases.append(("missing required fields", bad))

    bad = deepcopy(artifact)
    bad["kan_scaleup_ready_score"] = 1.0
    cases.append(("kan_scaleup_ready_score", bad))

    bad = deepcopy(artifact)
    bad["honest_verdict"] = "complete: stale"
    cases.append(("honest_verdict", bad))

    bad = deepcopy(artifact)
    bad["reproducibility_checksum"] = "sha256:bad"
    cases.append(("reproducibility_checksum", bad))

    for expected, bad_artifact in cases:
        if expected not in {"honest_verdict", "reproducibility_checksum", "kan_scaleup_ready_score"}:
            bad_artifact["kan_scaleup_ready_score"] = mod.kan_scaleup_ready_score(bad_artifact)
            bad_artifact["honest_verdict"] = mod.honest_verdict(bad_artifact)
            bad_artifact["reproducibility_checksum"] = mod.reproducibility_checksum(bad_artifact)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad_artifact)


def test_req_learn_5749_helper_edges(artifact: dict[str, object]) -> None:
    """REQ-LEARN-5749: helper edge cases stay deterministic and auditable."""

    assert mod.artifact_errors({}) == ["missing required fields: " + str(list(mod.REQUIRED_ARTIFACT_FIELDS))]
    assert mod.kan_scaleup_ready_score({}) == 0.0
    assert mod.honest_verdict({}).startswith("blocked:")
    assert mod.recovery_time_for_error(0.05, 30, 0.15) == 2
    assert mod.recovery_time_for_error(0.75, 30, 0.15) == 30

    positive = deepcopy(artifact)
    positive["suffix_error_by_arm"][mod.KAN_HEADLINE_ARM] = 0.01
    positive["arm_metrics"][mod.KAN_HEADLINE_ARM]["suffix_exact_error"] = 0.01
    positive["kan_mechanism_residual"] = mod.compute_kan_mechanism_residual(
        positive["suffix_error_by_arm"]
    )
    positive["dynamic_regret_by_arm"] = mod.compute_dynamic_regret_by_arm(
        positive["suffix_error_by_arm"]
    )
    positive["kan_scaleup_ready_score"] = mod.kan_scaleup_ready_score(positive)
    positive["honest_verdict"] = mod.honest_verdict(positive)
    positive["reproducibility_checksum"] = mod.reproducibility_checksum(positive)
    assert positive["kan_mechanism_residual"] > 0.0
    assert positive["kan_scaleup_ready_score"] == 1.0
    assert positive["honest_verdict"].startswith("complete: kan_mechanism_residual_positive")

    non_mapping_principles = deepcopy(artifact)
    non_mapping_principles["field_principles"] = []
    non_mapping_principles["kan_scaleup_ready_score"] = mod.kan_scaleup_ready_score(
        non_mapping_principles
    )
    non_mapping_principles["honest_verdict"] = mod.honest_verdict(non_mapping_principles)
    non_mapping_principles["reproducibility_checksum"] = mod.reproducibility_checksum(
        non_mapping_principles
    )
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(non_mapping_principles)
