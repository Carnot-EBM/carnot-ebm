"""Tests for Exp6347 ARC action-influence preflight.

Spec refs: REQ-ARC-WMTE-6347,
SCENARIO-ARC-WMTE-6347-REGISTRY-PRECHECK,
SCENARIO-ARC-WMTE-6347-WINDOW-RECONSTRUCTION,
SCENARIO-ARC-WMTE-6347-COUNTERFACTUAL-ORDERING,
SCENARIO-ARC-WMTE-6347-ADVERSARIAL-CONTROLS.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_6347_arc_action_influence_preflight as exp6347


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"


def _with_checksum(payload: dict) -> dict:
    payload["reproducibility_checksum"] = exp6347.payload_checksum(payload)
    return payload


def _artifact(tmp_path: Path) -> dict:
    return exp6347.run(
        date="20260812",
        result_path=tmp_path / exp6347.RESULT_RELATIVE_PATH.name,
        live_manifest_path=tmp_path / exp6347.LIVE_WINDOW_MANIFEST_RELATIVE_PATH.name,
        duration_s=1.25,
        test_exit_codes={command: 0 for command in exp6347.DEFAULT_TEST_COMMANDS},
        write=True,
    )


def test_req_arc_wmte_6347_spec_declares_preflight_contract() -> None:
    """REQ-ARC-WMTE-6347: OpenSpec names the action-influence preflight contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-ARC-WMTE-6347") :]
    for marker in (
        "SCENARIO-ARC-WMTE-6347-REGISTRY-PRECHECK",
        "SCENARIO-ARC-WMTE-6347-WINDOW-RECONSTRUCTION",
        "SCENARIO-ARC-WMTE-6347-COUNTERFACTUAL-ORDERING",
        "SCENARIO-ARC-WMTE-6347-ADVERSARIAL-CONTROLS",
        "recorded Exp6321 live-agent windows",
        "default-off A/B eligibility",
        exp6347.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert marker in section
    for field in exp6347.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_arc_wmte_6347_registry_precheck_blocks_duplicates() -> None:
    """SCENARIO-ARC-WMTE-6347-REGISTRY-PRECHECK: duplicate solve proposals fail closed."""

    clean = exp6347.registry_precheck(registry_text="")
    duplicate = exp6347.registry_precheck(registry_text=exp6347.INFLUENCE_TASK_ID)

    assert clean["task_kind"] == "influence_preflight_not_solve"
    assert clean["precheck_order"] == "registry_before_window_reconstruction"
    assert clean["all_selected_targets_nonduplicate"] is True
    assert clean["duplicate_solve_proposal_count"] == 0
    assert clean["registry_update_count"] == 0
    assert duplicate["all_selected_targets_nonduplicate"] is False
    assert duplicate["duplicate_solve_proposal_count"] == 1


def test_req_arc_wmte_6347_helper_receipts_cover_terminal_edges() -> None:
    """REQ-ARC-WMTE-6347: helper receipts cover terminal classes and transition payloads."""

    assert exp6347._terminal_class({"flagged_adversarial": True}) == "flagged"
    assert exp6347._terminal_class({"status": "blocked_precondition"}) == "blocked"
    row = exp6347.reconstruct_live_attempt_windows()[0]
    assert exp6347._transition_payload_for_row(row)


def test_scenario_arc_wmte_6347_reconstructs_live_windows_from_exp6321() -> None:
    """SCENARIO-ARC-WMTE-6347-WINDOW-RECONSTRUCTION: windows come from allowed receipts."""

    windows = exp6347.reconstruct_live_attempt_windows()

    assert len(windows) == exp6347.ELIGIBILITY_MIN_INDEPENDENT_WINDOWS
    assert {row["mechanic"] for row in windows} == {"push_block", "toggle_move"}
    assert all(row["transition_hash_match"] is True for row in windows)
    assert all(row["transition_payload_match"] is True for row in windows)
    assert all(row["reconstructed_from_allowed_fields"] is True for row in windows)
    assert all(row["recorded_action"]["action"] == 4 for row in windows)
    assert all(row["legal_actions"] == [4, 5] for row in windows)
    assert all(row["runtime_reverse_engineering_state"]["sample_size"] == 3 for row in windows)


def test_scenario_arc_wmte_6347_counterfactual_ordering_and_exact_value() -> None:
    """SCENARIO-ARC-WMTE-6347-COUNTERFACTUAL-ORDERING: route changes legal order only."""

    windows = exp6347.reconstruct_live_attempt_windows()
    order = exp6347.action_order_change_results(windows)
    quality = exp6347.one_step_exact_transition_quality(order)

    assert order["route_caused_action_order_change_count"] == len(windows)
    assert order["same_legal_action_set_count"] == len(windows)
    assert all(row["route_off_order"] == [5, 4] for row in order["rows"])
    assert all(row["target_licensed_route_on_order"] == [4, 5] for row in order["rows"])
    assert all(row["changed_top_action_has_exact_one_step_value"] is True for row in order["rows"])
    assert quality["target_licensed_route_on"]["top_action_exact_value_count"] == len(windows)
    assert quality["target_licensed_route_on"]["mean_top_action_changed_cells"] > 0
    assert quality["route_off"]["top_action_exact_value_count"] == 0


def test_scenario_arc_wmte_6347_adversarial_controls_remove_effects() -> None:
    """SCENARIO-ARC-WMTE-6347-ADVERSARIAL-CONTROLS: deletion, mutation, and traps fail closed."""

    windows = exp6347.reconstruct_live_attempt_windows()
    order = exp6347.action_order_change_results(windows)
    controls = exp6347.fixture_mutation_and_route_deletion_results(windows, order)
    leakage = exp6347.leakage_overlap_and_escape_tests(windows, order)

    assert controls["route_deletion_removed_effect_count"] == len(windows)
    assert controls["fixture_mutation_failed_closed_count"] == len(windows)
    assert controls["evidence_permutation_order_unchanged_count"] == len(windows)
    assert controls["all_controls_passed"] is True
    assert leakage["leakage_overlap_count"] == 0
    assert leakage["hidden_source_trap_rejected"] is True
    assert leakage["off_path_adapter_trap_rejected"] is True
    assert leakage["all_escape_tests_passed"] is True


def test_scenario_arc_wmte_6347_artifact_schema_and_no_solve_claim(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6347-ADVERSARIAL-CONTROLS: artifact is complete and zero-credit."""

    artifact = _artifact(tmp_path)
    loaded = json.loads((tmp_path / exp6347.RESULT_RELATIVE_PATH.name).read_text(encoding="utf-8"))

    assert loaded == artifact
    assert set(exp6347.REQUIRED_ARTIFACT_FIELDS) == set(artifact["field_principles"])
    assert set(exp6347.REQUIRED_ARTIFACT_FIELDS) == set(artifact["field_provenance"])
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["arc_action_influence_eligible_score"] == 1.0
    assert artifact["verifier_is_oracle"] == exp6347.EXACT_TRANSITION_CHECKER_NAME
    assert artifact["exact_oracle_claim_boundary"]["not_a_solve_oracle"] is True
    assert artifact["live_attempt_window_manifest_path_and_hash"]["row_count"] == 4
    assert artifact["preconditions_checked"]["eligibility_rule_preregistered"][
        "min_independent_live_windows"
    ] == exp6347.ELIGIBILITY_MIN_INDEPENDENT_WINDOWS
    for field in exp6347.FORBIDDEN_ZERO_FIELDS:
        assert type(artifact[field]) is int
        assert artifact[field] == 0
    assert artifact["reproducibility_checksum"] == exp6347.payload_checksum(artifact)
    exp6347.validate_artifact(artifact)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("hidden_game_source_access_count", 1, "hidden_game_source_access_count"),
        ("offline_ground_truth_bfs_count", 1, "offline_ground_truth_bfs_count"),
        ("hand_game_adapter_count", 1, "hand_game_adapter_count"),
        ("per_game_calibration_count", 1, "per_game_calibration_count"),
        ("solve_claim_count", 1, "solve_claim_count"),
        ("registry_update_count", 1, "registry_update_count"),
        ("llm_call_count", 1, "llm_call_count"),
        ("solve_provenance", "outer_loop_re", "solve_provenance"),
        ("verifier_is_oracle", "wrong_checker", "verifier_is_oracle"),
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
    ],
)
def test_scenario_arc_wmte_6347_validation_rejects_forbidden_drift(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    """SCENARIO-ARC-WMTE-6347-ADVERSARIAL-CONTROLS: forbidden drift is rejected."""

    artifact = _artifact(tmp_path)
    bad = copy.deepcopy(artifact)
    bad[field] = value
    _with_checksum(bad)

    with pytest.raises(ValueError, match=message):
        exp6347.validate_artifact(bad)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda a: a["arc_registry_precheck_path_hash_and_result"].__setitem__(
                "all_selected_targets_nonduplicate", False
            ),
            "arc_registry_precheck_path_hash_and_result",
        ),
        (
            lambda a: a["no_duplicate_solve_receipt"].__setitem__(
                "no_duplicate_solve_proposal", False
            ),
            "no_duplicate_solve_receipt",
        ),
        (
            lambda a: a["route_on_off_counterfactual_contract"].__setitem__(
                "legal_action_sets_identical_by_route_state", False
            ),
            "route_on_off_counterfactual_contract",
        ),
        (
            lambda a: a["action_order_change_results_by_game_window_and_seed"].__setitem__(
                "route_caused_action_order_change_count", 0
            ),
            "action_order_change_results_by_game_window_and_seed",
        ),
        (
            lambda a: a["one_step_exact_transition_quality_by_route_state"][
                "target_licensed_route_on"
            ].__setitem__("top_action_exact_value_count", 0),
            "one_step_exact_transition_quality_by_route_state",
        ),
        (
            lambda a: a["leakage_overlap_and_escape_tests"].__setitem__(
                "all_escape_tests_passed", False
            ),
            "leakage_overlap_and_escape_tests",
        ),
        (
            lambda a: a["fixture_mutation_and_route_deletion_results"].__setitem__(
                "all_controls_passed", False
            ),
            "fixture_mutation_and_route_deletion_results",
        ),
        (
            lambda a: a.__setitem__("field_principles", {}),
            "field_principles",
        ),
        (
            lambda a: a.__setitem__("field_provenance", {}),
            "field_provenance",
        ),
        (
            lambda a: a.__setitem__("honest_verdict", "not_terminal"),
            "honest_verdict",
        ),
        (
            lambda a: a.__setitem__("arc_action_influence_eligible_score", 0.0),
            "arc_action_influence_eligible_score",
        ),
    ],
)
def test_req_arc_wmte_6347_validation_rejects_artifact_guard_drift(
    tmp_path: Path, mutate, message: str
) -> None:
    """REQ-ARC-WMTE-6347: validator catches protected artifact drift."""

    artifact = _artifact(tmp_path)
    bad = copy.deepcopy(artifact)
    mutate(bad)
    _with_checksum(bad)

    with pytest.raises(ValueError, match=message):
        exp6347.validate_artifact(bad)


def test_req_arc_wmte_6347_validation_rejects_missing_and_checksum(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-6347: missing fields and checksum drift fail validation."""

    artifact = _artifact(tmp_path)

    checksum = dict(artifact)
    checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp6347.validate_artifact(checksum)

    missing = dict(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing fields"):
        exp6347.validate_artifact(missing)
