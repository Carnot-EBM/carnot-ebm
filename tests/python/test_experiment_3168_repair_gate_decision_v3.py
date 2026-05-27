"""Tests for Exp 3168 conservative repair-gate decision v3.

Spec refs: REQ-VERIFY-3168, SCENARIO-VERIFY-3168.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import repair_gate_decision_v3 as mod


REQUIRED_FIELDS = {
    "repair_gate_decision_v3_ready",
    "repair_gate_state",
    "clean_rerun_artifact_present",
    "false_accept_rate",
    "false_accept_gate_passed",
    "flagged_adversarial",
    "controlled_invariance_passed",
    "exact_authority_ready",
    "selected_repair_rows",
    "repair_blockers",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_common_sources(
    root: Path,
    *,
    include_clean_rerun: bool = True,
    v9_overrides: dict[str, Any] | None = None,
    exact_labels: bool = True,
) -> None:
    (root / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nTests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("exact authority blocks tainted repair\n", encoding="utf-8")
    (root / "openspec/capabilities/verification").mkdir(parents=True, exist_ok=True)
    (root / "openspec/capabilities/verification/spec.md").write_text(
        "REQ-VERIFY-3168\nSCENARIO-VERIFY-3168\n"
        "results/experiment_3168_repair_gate_decision_v3.json\n",
        encoding="utf-8",
    )

    clean_rerun = {
        "artifact": "experiment_3167_clean_live_sota_verifier_rerun_v9",
        "clean_live_verifier_rerun_v9_ready": True,
        "gated_skip": False,
        "gated_skip_reason": "",
        "false_accept_rate": 0.0,
        "false_accept_gate_passed": True,
        "flagged_adversarial": False,
        "regression_rows_included": True,
        "exact_ground_truth_count": 2,
        "controlled_invariance_passed": True,
        "headline_claim_allowed": True,
        "live_call_count": 4,
        "selected_model_ids": ["unit/sota"],
        "planned_rerun_set": {
            "regression_row_ids": ["row-fa"],
            "row_ids": ["row-fa", "row-ok"],
        },
        "inference_substrate": {
            "executes_models": True,
            "live_model_calls": 4,
            "legacy_small_model_used": False,
        },
        "honest_verdict": "complete: clean v9 fixture",
    }
    if v9_overrides:
        clean_rerun.update(v9_overrides)
    if include_clean_rerun:
        _write_json(root, mod.EXP3167_REL_PATH, clean_rerun)

    label_for_false_accept = "INVALID" if exact_labels else ""
    _write_json(
        root,
        mod.EXP3137_REL_PATH,
        {
            "artifact": "experiment_3137_exact_safe_accept_abstain_contract_v1",
            "acceptance_contract_v1_ready": True,
            "known_false_accept_rows_blocked": True,
            "replay_false_accept_rate": 0.0,
            "regression_row_set": ["row-fa"],
            "repair_gate_prerequisites": {"require_exact_label_authority": True},
            "replay_rows": [
                {
                    "row_id": "row-fa",
                    "exact_label": label_for_false_accept,
                    "decision": "abstain",
                    "expected_action": "reject",
                    "matched_rule_id": "ABSTAIN_KNOWN_FALSE_ACCEPT_REGRESSION",
                },
                {
                    "row_id": "row-ok",
                    "exact_label": "VALID" if exact_labels else "",
                    "decision": "accept",
                    "expected_action": "accept",
                    "matched_rule_id": "ACCEPT_EXACT_COVERED_CONSISTENT",
                },
            ],
        },
    )
    _write_json(
        root,
        mod.EXP3138_REL_PATH,
        {
            "artifact": "experiment_3138_canonical_answer_vericot_grounding_pilot_v1",
            "canonical_grounding_pilot_v1_ready": True,
            "false_accept_rows_blocked": 1,
            "residual_false_accept_rows": [],
            "regression_row_replay": [
                {
                    "row_id": "row-fa",
                    "exact_label": label_for_false_accept,
                    "expected_action": "reject",
                    "contract_replay": {"decision": "abstain"},
                    "blocked_by": ["canonicalization", "ledger_replay"],
                    "solver_certificate_summary": {
                        "present": True,
                        "solver_authority": "z3_solver",
                        "solver_exact_label": label_for_false_accept,
                        "minimal_correction_set": {
                            "kind": "remove_conflicting_constraint",
                            "constraints_to_relax": ["constraint_0"],
                        },
                        "unsat_core": ["constraint_0", "constraint_1"],
                    },
                }
            ],
        },
    )
    _write_json(
        root,
        mod.EXP3166_REL_PATH,
        {
            "artifact": "experiment_3166_verifier_invariance_token_suspicion_audit_v1",
            "verifier_invariance_token_suspicion_audit_ready": True,
            "downstream_policy_for_exp3167": {
                "acceptance_requires_exact_authority": True,
                "required_acceptance_authority_fields": [
                    "exact_label",
                    "exact_safe_replay_decision",
                    "canonical_equivalence",
                    "solver_or_test_authority",
                    "monitor_ledger_consistency",
                ],
            },
            "trusted_exact_rows": [
                {
                    "row_id": "row-fa",
                    "exact_label": label_for_false_accept,
                    "expected_action": "reject",
                    "trusted_exact_authority": exact_labels,
                    "acceptance_authority": exact_labels,
                    "source_experiments": ["exp3137", "exp3138"],
                },
                {
                    "row_id": "row-ok",
                    "exact_label": "VALID" if exact_labels else "",
                    "expected_action": "accept",
                    "trusted_exact_authority": exact_labels,
                    "acceptance_authority": exact_labels,
                    "source_experiments": ["exp3137"],
                },
            ],
        },
    )
    _write_json(
        root,
        mod.EXP3153_REL_PATH,
        {
            "experiment": 3153,
            "schema": "blocked_gate_check_v1",
            "honest_verdict": "blocked_gate_check_failed",
        },
    )
    _write_json(
        root,
        mod.EXP3140_REL_PATH,
        {
            "artifact": "experiment_3140_repair_gate_unlock_decision_v1",
            "repair_gate_state": "blocked_other",
            "honest_verdict": "blocked_other: prior taint",
        },
    )


def test_req_verify_3168_spec_anchor_exists() -> None:
    """REQ-VERIFY-3168: OpenSpec declares the v3 repair-gate decision."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3168" in spec
    assert "SCENARIO-VERIFY-3168" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "repair_gate_decision_v3_ready" in spec
    assert "blocked_gated_skip" in spec


def test_scenario_verify_3168_unblocks_only_complete_clean_exact_safe_v9(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-3168: a fully clean v9 artifact selects exact repair rows."""

    _write_common_sources(tmp_path)

    output_path = mod.write_artifact(
        tmp_path,
        started_s=4.0,
        now_s=6.5,
        tests_run=["focused-3168"],
    )
    artifact = json.loads(output_path.read_text(encoding="utf-8"))

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["repair_gate_decision_v3_ready"] is True
    assert artifact["repair_gate_state"] == "unblocked"
    assert artifact["clean_rerun_artifact_present"] is True
    assert artifact["false_accept_rate"] == pytest.approx(0.0)
    assert artifact["false_accept_gate_passed"] is True
    assert artifact["flagged_adversarial"] is False
    assert artifact["controlled_invariance_passed"] is True
    assert artifact["exact_authority_ready"] is True
    assert artifact["repair_blockers"] == []
    assert artifact["selected_repair_rows"] == [
        {
            "row_id": "row-fa",
            "exact_authority_constraints": {
                "exact_label": "INVALID",
                "expected_action": "reject",
                "exact_safe_decision": "abstain",
                "canonical_decision": "abstain",
                "solver_or_test_authority": "z3_solver",
                "minimal_correction_set": {
                    "constraints_to_relax": ["constraint_0"],
                    "kind": "remove_conflicting_constraint",
                },
                "unsat_core": ["constraint_0", "constraint_1"],
            },
        }
    ]
    assert artifact["inference_substrate"]["executes_models"] is False
    assert artifact["inference_substrate"]["live_model_calls"] == 0
    assert artifact["inference_substrate"]["source_live_model_calls_reused"] == 4
    assert artifact["tests_run"] == ["focused-3168"]
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["honest_verdict"].startswith("complete:")


def test_scenario_verify_3168_missing_clean_rerun_writes_complete_block(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-3168: absent v9 evidence blocks without a thin conductor verdict."""

    _write_common_sources(tmp_path, include_clean_rerun=False)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.25)

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["repair_gate_decision_v3_ready"] is True
    assert artifact["repair_gate_state"] == "blocked_missing_clean_rerun"
    assert artifact["clean_rerun_artifact_present"] is False
    assert artifact["false_accept_rate"] == pytest.approx(1.0)
    assert artifact["false_accept_gate_passed"] is False
    assert artifact["selected_repair_rows"] == []
    assert any("clean v9 rerun artifact is missing" in item for item in artifact["repair_blockers"])
    assert artifact["honest_verdict"].startswith("blocked_missing_clean_rerun:")


@pytest.mark.parametrize(
    ("v9_overrides", "expected_state", "blocker_fragment"),
    [
        (
            {"false_accept_rate": 0.25, "false_accept_gate_passed": False},
            "blocked_false_accept",
            "false_accept_rate=0.25",
        ),
        (
            {"flagged_adversarial": True},
            "blocked_flagged_verifier",
            "flagged_adversarial=true",
        ),
        (
            {
                "gated_skip": True,
                "gated_skip_reason": "preflight failed",
                "live_call_count": 0,
                "selected_model_ids": [],
                "inference_substrate": {"executes_models": False, "live_model_calls": 0},
            },
            "blocked_gated_skip",
            "gated_skip=true",
        ),
        (
            {
                "live_call_count": 0,
                "selected_model_ids": [],
                "inference_substrate": {"executes_models": False, "live_model_calls": 0},
            },
            "blocked_missing_live_model",
            "bounded live model rerun is missing",
        ),
        (
            {"controlled_invariance_passed": False},
            "blocked_invariance_failure",
            "controlled_invariance_passed is not true",
        ),
        (
            {"false_accept_gate_passed": False},
            "blocked_false_accept",
            "false_accept_gate_passed is not true",
        ),
        (
            {"clean_live_verifier_rerun_v9_ready": False},
            "blocked_other",
            "clean_live_verifier_rerun_v9_ready is not true",
        ),
        (
            {"regression_rows_included": False},
            "blocked_other",
            "regression_rows_included is not true",
        ),
        (
            {"headline_claim_allowed": False},
            "blocked_other",
            "headline_claim_allowed is not true",
        ),
        (
            {"false_accept_rate": "not-a-rate", "false_accept_gate_passed": False},
            "blocked_false_accept",
            "false_accept_rate=1.0",
        ),
        (
            {"planned_rerun_set": {"regression_row_ids": ["missing-row"]}},
            "blocked_other",
            "selected repair rows with exact authority constraints are missing",
        ),
    ],
)
def test_scenario_verify_3168_classifies_blocked_v9_criteria(
    tmp_path: Path,
    v9_overrides: dict[str, Any],
    expected_state: str,
    blocker_fragment: str,
) -> None:
    """SCENARIO-VERIFY-3168: failed v9 criteria produce actionable blocked states."""

    _write_common_sources(tmp_path, v9_overrides=v9_overrides)

    artifact = mod.build_artifact(tmp_path)

    assert artifact["repair_gate_state"] == expected_state
    assert artifact["selected_repair_rows"] == []
    assert any(blocker_fragment in item for item in artifact["repair_blockers"])
    assert artifact["honest_verdict"].startswith(f"{expected_state}:")


def test_scenario_verify_3168_blocks_missing_exact_authority(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3168: v9 exact counts are not enough without exact labels."""

    _write_common_sources(tmp_path, exact_labels=False)

    artifact = mod.build_artifact(tmp_path)

    assert artifact["repair_gate_state"] == "blocked_missing_exact_labels"
    assert artifact["exact_authority_ready"] is False
    assert artifact["selected_repair_rows"] == []
    assert any("exact authority is not ready" in item for item in artifact["repair_blockers"])


def test_req_verify_3168_exact_authority_helper_edges() -> None:
    """REQ-VERIFY-3168: exact authority fails closed for malformed source edges."""

    clean = {"exact_ground_truth_count": 2}
    exp3137 = {
        "acceptance_contract_v1_ready": True,
        "replay_false_accept_rate": 0.0,
        "replay_rows": [{"row_id": "row-fa", "exact_label": "INVALID"}],
    }
    exp3138 = {
        "canonical_grounding_pilot_v1_ready": True,
        "residual_false_accept_rows": [],
    }
    exp3166 = {
        "verifier_invariance_token_suspicion_audit_ready": True,
        "downstream_policy_for_exp3167": {"acceptance_requires_exact_authority": True},
        "trusted_exact_rows": [
            {
                "row_id": "row-fa",
                "exact_label": "INVALID",
                "trusted_exact_authority": True,
                "acceptance_authority": True,
            }
        ],
    }

    assert mod.exact_authority_ready(clean, exp3166, exp3137, exp3138) is True
    assert (
        mod.exact_authority_ready(
            clean,
            exp3166,
            {**exp3137, "acceptance_contract_v1_ready": False},
            exp3138,
        )
        is False
    )
    assert (
        mod.exact_authority_ready(
            clean,
            exp3166,
            {**exp3137, "replay_false_accept_rate": 0.1},
            exp3138,
        )
        is False
    )
    assert (
        mod.exact_authority_ready(
            clean,
            exp3166,
            exp3137,
            {**exp3138, "canonical_grounding_pilot_v1_ready": False},
        )
        is False
    )
    assert (
        mod.exact_authority_ready(
            clean,
            exp3166,
            exp3137,
            {**exp3138, "residual_false_accept_rows": ["row-fa"]},
        )
        is False
    )
    assert (
        mod.exact_authority_ready(
            clean,
            {**exp3166, "verifier_invariance_token_suspicion_audit_ready": False},
            exp3137,
            exp3138,
        )
        is False
    )
    assert (
        mod.exact_authority_ready(
            clean,
            {**exp3166, "downstream_policy_for_exp3167": {}},
            exp3137,
            exp3138,
        )
        is False
    )
    assert (
        mod.exact_authority_ready(
            clean,
            {
                **exp3166,
                "trusted_exact_rows": [
                    {
                        "row_id": "row-fa",
                        "exact_label": "INVALID",
                        "trusted_exact_authority": False,
                        "acceptance_authority": True,
                    }
                ],
            },
            exp3137,
            exp3138,
        )
        is False
    )


def test_req_verify_3168_validation_rejects_unsafe_terminal_shapes() -> None:
    """REQ-VERIFY-3168: terminal state, blockers, and no-inference metadata are enforced."""

    artifact = {
        "repair_gate_decision_v3_ready": True,
        "repair_gate_state": "unblocked",
        "clean_rerun_artifact_present": True,
        "false_accept_rate": 0.0,
        "false_accept_gate_passed": True,
        "flagged_adversarial": False,
        "controlled_invariance_passed": True,
        "exact_authority_ready": True,
        "selected_repair_rows": [],
        "repair_blockers": [],
        "source_artifacts": [],
        "inference_substrate": {
            "executes_models": False,
            "executes_repairs": False,
            "live_model_calls": 0,
            "repair_calls": 0,
        },
        "honest_verdict": "complete: invalid",
    }
    with pytest.raises(ValueError, match="selected rows"):
        mod.validate_artifact(artifact)

    artifact["repair_gate_state"] = "maybe"
    artifact["selected_repair_rows"] = [{"row_id": "row-fa"}]
    with pytest.raises(ValueError, match="allowed state"):
        mod.validate_artifact(artifact)

    artifact["repair_gate_state"] = "blocked_other"
    artifact["honest_verdict"] = "blocked_other: invalid"
    artifact["repair_blockers"] = ["blocked"]
    artifact["inference_substrate"]["live_model_calls"] = 1
    with pytest.raises(ValueError, match="must not execute"):
        mod.validate_artifact(artifact)
