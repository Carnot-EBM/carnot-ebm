"""Tests for Exp6321 target-licensed route live shadow A/B.

Spec refs: REQ-ARC-WMTE-6321,
SCENARIO-ARC-WMTE-6321-DEFAULT-OFF-PARITY,
SCENARIO-ARC-WMTE-6321-SHADOW-COMPUTED-ISOLATION,
SCENARIO-ARC-WMTE-6321-REGISTRY-PRECHECK,
SCENARIO-ARC-WMTE-6321-ARTIFACT-GUARDS.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_6307_arc_target_validated_route_canary as exp6307
from carnot import experiment_6321_arc_target_licensed_route_live_shadow_ab as exp6321


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"


def _artifact(tmp_path: Path) -> dict:
    return exp6321.run(
        date="20260811",
        result_path=tmp_path / exp6321.RESULT_RELATIVE_PATH.name,
        transition_manifest_path=tmp_path / exp6321.TRANSITION_MANIFEST_RELATIVE_PATH.name,
        duration_s=63.0,
        test_exit_codes={command: 0 for command in exp6321.DEFAULT_TEST_COMMANDS},
        model_resolver=exp6307.deterministic_test_model_resolver,
        write=True,
    )


def test_req_arc_wmte_6321_spec_declares_live_shadow_contract() -> None:
    """REQ-ARC-WMTE-6321: OpenSpec names the default-off live shadow contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("### REQ-ARC-WMTE-6321") :]
    for marker in (
        "SCENARIO-ARC-WMTE-6321-DEFAULT-OFF-PARITY",
        "SCENARIO-ARC-WMTE-6321-SHADOW-COMPUTED-ISOLATION",
        "SCENARIO-ARC-WMTE-6321-REGISTRY-PRECHECK",
        "SCENARIO-ARC-WMTE-6321-ARTIFACT-GUARDS",
        "default-off shadow",
        "byte-identical",
        exp6321.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert marker in section
    for field in exp6321.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for model_id in exp6321.MANDATED_MODEL_IDS:
        assert model_id in section


def test_scenario_arc_wmte_6321_registry_precheck_blocks_duplicates() -> None:
    """SCENARIO-ARC-WMTE-6321-REGISTRY-PRECHECK: duplicate selected targets fail closed."""

    clean = exp6321.registry_precheck()
    duplicate = exp6321.registry_precheck(
        registry_text="selected_target: exp6321_shadow_synthetic_push_l0"
    )

    assert clean["registry_read_mode"] == "full_text"
    assert clean["precheck_order"] == "registry_before_attempt_freeze"
    assert clean["all_selected_targets_nonduplicate"] is True
    assert clean["duplicate_selected_target_count"] == 0
    assert duplicate["all_selected_targets_nonduplicate"] is False
    assert duplicate["duplicate_selected_target_count"] == 1


def test_req_arc_wmte_6321_receipt_edges_cover_terminal_and_seed_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-6321: helper receipts cover terminal classes and receipt files."""

    assert exp6321._terminal_class({"flagged_adversarial": True}) == "flagged"
    assert exp6321._terminal_class({"status": "blocked_precondition"}) == "blocked"
    assert exp6321._terminal_class({}) == "unknown"

    models, model_files, tokenizers, cuda = exp6321._model_bundle(None)
    assert {row["hf_id"] for row in models} == set(exp6321.MANDATED_MODEL_IDS)
    assert set(model_files) == set(exp6321.MANDATED_MODEL_IDS)
    assert set(tokenizers) == set(exp6321.MANDATED_MODEL_IDS)
    assert set(cuda) == set(exp6321.MANDATED_MODEL_IDS)

    with pytest.raises(ValueError, match="unknown mechanic"):
        exp6321._transitions_for("bad_mechanic", 1)

    receipt_path = tmp_path / "receipts.json"
    monkeypatch.setattr(exp6321, "EXTERNAL_TEST_RECEIPT_PATH", receipt_path)
    assert exp6321._read_external_test_receipts()[exp6321.RUN_COMMAND] == 0
    receipt_path.write_text("{bad", encoding="utf-8")
    assert exp6321._read_external_test_receipts()[exp6321.RUN_COMMAND] == 0
    receipt_path.write_text(
        json.dumps({exp6321.FOCUSED_TEST_COMMAND: 4, "custom": None}),
        encoding="utf-8",
    )
    receipts = exp6321._read_external_test_receipts()
    assert receipts[exp6321.RUN_COMMAND] == 0
    assert receipts[exp6321.FOCUSED_TEST_COMMAND] == 4
    assert receipts["custom"] is None


def test_scenario_arc_wmte_6321_artifact_schema_and_no_solve_claim(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-6321-ARTIFACT-GUARDS: artifact is complete and zero-credit."""

    artifact = _artifact(tmp_path)
    loaded = json.loads((tmp_path / exp6321.RESULT_RELATIVE_PATH.name).read_text(encoding="utf-8"))

    assert loaded == artifact
    assert set(exp6321.REQUIRED_ARTIFACT_FIELDS) == set(artifact["field_principles"])
    assert set(exp6321.REQUIRED_ARTIFACT_FIELDS) == set(artifact["field_provenance"])
    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["solve_claimed"] is False
    assert artifact["levels_credited"] == 0
    assert artifact["registry_update_count"] == 0
    assert artifact["duration_padding_count"] == 0
    assert artifact["arc_route_live_shadow_ready_score"] == 1.0
    assert artifact["default_off_shadow_wiring_receipt"]["default_enabled"] is False
    assert artifact["default_off_shadow_wiring_receipt"]["computed_arm_enabled"] is True
    assert artifact["action_budget_registry_and_level_state_parity"]["exact_action_parity"] is True
    assert artifact["target_license_evidence_receipts"]["false_license_count"] == 0
    assert artifact["supported_unsupported_and_abstained_proposals_by_arm"]["shadow_computed"][
        "unsupported_proposal_count"
    ] >= 1
    assert artifact["random_seeds"] == list(exp6321.RANDOM_SEEDS)
    assert artifact["reproducibility_checksum"] == exp6321.payload_checksum(artifact)
    exp6321.validate_artifact(artifact)


def test_scenario_arc_wmte_6321_exp6307_random_seed_warning_is_recorded(
    tmp_path: Path,
) -> None:
    """REQ-ARC-WMTE-6321: Exp6307 random_seed methodology gap is corrected in receipt."""

    artifact = _artifact(tmp_path)
    receipt = artifact["exp6307_and_exp6308_paths_hashes_and_terminal_classes"]

    assert receipt["exp6307_checked_in_artifact_has_random_seed"] is False
    assert receipt["exp6307_current_writer_emits_random_seed"] is True
    assert receipt["exp6307_missing_random_seed_methodology_warning_corrected"] is True
    assert artifact["random_seeds"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("hidden_game_source_access_count", 1, "hidden_game_source_access_count"),
        (
            "source_bfs_adapter_prior_game_hidden_state_and_registry_target_access_count",
            1,
            "source_bfs_adapter_prior_game_hidden_state_and_registry_target_access_count",
        ),
        ("registry_update_count", 1, "registry_update_count"),
        ("levels_credited", 1, "levels_credited"),
        ("duration_padding_count", 1, "duration_padding_count"),
        ("solve_claimed", True, "solve_claimed"),
        ("solve_provenance", "offline_proxy", "solve_provenance"),
        ("verifier_is_oracle", True, "verifier_is_oracle"),
    ],
)
def test_scenario_arc_wmte_6321_forbidden_counts_and_solve_guards(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    """SCENARIO-ARC-WMTE-6321-ARTIFACT-GUARDS: forbidden counters stay bare zero."""

    artifact = _artifact(tmp_path)
    bad = copy.deepcopy(artifact)
    bad[field] = value
    bad["reproducibility_checksum"] = exp6321.payload_checksum(bad)

    with pytest.raises(ValueError, match=message):
        exp6321.validate_artifact(bad)


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
            lambda a: a["arc_registry_precheck_path_hash_and_result"].__setitem__(
                "duplicate_selected_target_count", 1
            ),
            "arc_registry_precheck_path_hash_and_result",
        ),
        (
            lambda a: a["default_off_shadow_wiring_receipt"].__setitem__(
                "default_enabled", True
            ),
            "default_off_shadow_wiring_receipt",
        ),
        (
            lambda a: a["action_budget_registry_and_level_state_parity"].__setitem__(
                "exact_action_parity", False
            ),
            "action_budget_registry_and_level_state_parity",
        ),
        (
            lambda a: a["action_budget_registry_and_level_state_parity"].__setitem__(
                "budget_parity", False
            ),
            "action_budget_registry_and_level_state_parity",
        ),
        (
            lambda a: a["action_budget_registry_and_level_state_parity"].__setitem__(
                "level_state_parity", False
            ),
            "action_budget_registry_and_level_state_parity",
        ),
        (
            lambda a: a["action_budget_registry_and_level_state_parity"].__setitem__(
                "registry_hash_parity", False
            ),
            "action_budget_registry_and_level_state_parity",
        ),
        (
            lambda a: a["target_license_evidence_receipts"].__setitem__(
                "false_license_count", 1
            ),
            "target_license_evidence_receipts",
        ),
        (
            lambda a: a["target_license_evidence_receipts"].__setitem__(
                "runtime_transition_only", False
            ),
            "target_license_evidence_receipts",
        ),
        (
            lambda a: a["prospective_action_support_by_arm"]["shadow_computed"].__setitem__(
                "applied_unsupported_proposal_count", 1
            ),
            "prospective_action_support_by_arm",
        ),
        (lambda a: a.__setitem__("MODEL_SPECS", [dict(a["MODEL_SPECS"][0])]), "MODEL_SPECS"),
        (lambda a: a.__setitem__("models_used", [exp6321.MANDATED_MODEL_IDS[0]]), "models_used"),
        (lambda a: a.__setitem__("field_principles", {}), "field_principles"),
        (lambda a: a.__setitem__("field_provenance", {}), "field_provenance"),
        (lambda a: a.__setitem__("inference_substrate", "live_llm_inference"), "inference_substrate"),
        (lambda a: a.__setitem__("honest_verdict", "not_terminal"), "honest_verdict"),
        (
            lambda a: a["default_off_shadow_wiring_receipt"].__setitem__(
                "mutates_shipped_action", True
            ),
            "default_off_shadow_wiring_receipt",
        ),
        (
            lambda a: a["submitted_config_pre_and_post_hashes"].__setitem__(
                "unchanged", False
            ),
            "submitted_config_pre_and_post_hashes",
        ),
        (
            lambda a: a.__setitem__("arc_route_live_shadow_ready_score", 0.0),
            "arc_route_live_shadow_ready_score",
        ),
    ],
)
def test_scenario_arc_wmte_6321_artifact_validation_rejects_drift(
    tmp_path: Path, mutate, message: str
) -> None:
    """SCENARIO-ARC-WMTE-6321-ARTIFACT-GUARDS: validation catches protected drift."""

    artifact = _artifact(tmp_path)
    bad = copy.deepcopy(artifact)
    mutate(bad)
    bad["reproducibility_checksum"] = exp6321.payload_checksum(bad)

    with pytest.raises(ValueError, match=message):
        exp6321.validate_artifact(bad)


def test_req_arc_wmte_6321_validation_rejects_checksum_and_missing_field(
    tmp_path: Path,
) -> None:
    """REQ-ARC-WMTE-6321: checksum drift and missing fields fail validation."""

    artifact = _artifact(tmp_path)

    checksum = dict(artifact)
    checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp6321.validate_artifact(checksum)

    missing = dict(artifact)
    missing.pop("status")
    with pytest.raises(ValueError, match="missing fields"):
        exp6321.validate_artifact(missing)
