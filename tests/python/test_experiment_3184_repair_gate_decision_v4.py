"""Tests for Exp 3184 repair-gate decision v4.

Spec refs: REQ-VERIFY-3184, SCENARIO-VERIFY-3184.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import repair_gate_decision_v4 as mod


REQUIRED_FIELDS = {
    "repair_gate_decision_v4_ready",
    "repair_gate_state",
    "unblocking_predicates",
    "blocker_reasons",
    "missing_artifacts",
    "allowed_repair_attempt_budget",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text_sources(root: Path) -> None:
    (root / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nTests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("No live model calls in the gate\n", encoding="utf-8")
    spec = root / mod.SPEC_REL_PATH
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(
        "REQ-VERIFY-3184\nSCENARIO-VERIFY-3184\n"
        "results/experiment_3184_repair_gate_decision_v4.json\n",
        encoding="utf-8",
    )


def _write_standard_sources(
    root: Path,
    *,
    receipt: dict[str, Any] | None = None,
    invariance: dict[str, Any] | None = None,
    clean: dict[str, Any] | None = None,
    certificate: dict[str, Any] | None = None,
    include_certificate: bool = True,
) -> None:
    _write_text_sources(root)
    _write_json(
        root,
        mod.EXP3168_REL_PATH,
        {
            "repair_gate_decision_v3_ready": True,
            "repair_gate_state": "unblocked",
            "false_accept_rate": 0.0,
            "flagged_adversarial": False,
            "honest_verdict": "complete: prior v3 gate fixture",
        },
    )
    receipt_payload = {
        "local_sota_receipt_smoke_v3_ready": True,
        "clean_rerun_allowed": True,
        "substrate_classification": "full_local_sota_receipt",
        "cpu_fallback_used": False,
        "live_call_count": 2,
        "proof_receipts": [{"receipt_id": "a"}, {"receipt_id": "b"}],
        "inference_substrate": {"live_model_calls": 2, "executes_models": True},
    }
    if receipt:
        receipt_payload.update(receipt)
    _write_json(root, mod.EXP3179_REL_PATH, receipt_payload)

    invariance_payload = {
        "controlled_invariance_executor_v2_ready": True,
        "controlled_invariance_passed": True,
        "blocker_reasons": [],
        "exact_row_count": 72,
        "known_false_accept_regression_count": 2,
        "inference_substrate": {"new_live_model_calls": 0, "live_model_calls": 0},
    }
    if invariance:
        invariance_payload.update(invariance)
    _write_json(root, mod.EXP3180_REL_PATH, invariance_payload)

    clean_payload = {
        "clean_live_sota_verifier_rerun_v10_ready": True,
        "gated_skip": False,
        "gate_reasons": [],
        "metrics_computed": True,
        "live_call_count": 6,
        "models_used": ["unsloth/gemma-4-26B-A4B-it-GGUF"],
        "flagged_adversarial": False,
        "headline_claim_allowed": True,
        "false_accept_rate": 0.0,
        "known_false_accepts_accepted": [],
        "inference_substrate": {"live_model_calls": 6, "executes_models": True},
    }
    if clean:
        clean_payload.update(clean)
    _write_json(root, mod.EXP3181_REL_PATH, clean_payload)

    certificate_payload = {
        "counterexample_certificate_expansion_v3_ready": True,
        "repair_call_ready": True,
        "blocker_reasons": [],
        "flagged_adversarial": False,
        "exact_row_count": 72,
        "known_false_accept_rows_covered": 2,
        "inference_substrate": {"new_live_model_calls": 0, "live_model_calls": 0},
    }
    if certificate:
        certificate_payload.update(certificate)
    if include_certificate:
        _write_json(root, mod.EXP3183_REL_PATH, certificate_payload)


def test_req_verify_3184_spec_anchor_exists() -> None:
    """REQ-VERIFY-3184: OpenSpec declares the v4 repair gate artifact."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-3184" in spec
    assert "SCENARIO-VERIFY-3184" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "repair_gate_decision_v4_ready" in spec


def test_scenario_verify_3184_current_blockers_are_all_visible(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3184: failed receipt, verifier, and certificate gates all block."""

    _write_standard_sources(
        tmp_path,
        receipt={
            "clean_rerun_allowed": False,
            "substrate_classification": "cpu_fallback_receipt_only",
            "cpu_fallback_used": True,
        },
        clean={
            "gated_skip": True,
            "gate_reasons": ["exp3179.clean_rerun_allowed=false"],
            "metrics_computed": False,
            "flagged_adversarial": True,
            "headline_claim_allowed": False,
        },
        certificate={
            "repair_call_ready": False,
            "flagged_adversarial": True,
            "blocker_reasons": ["flagged_adversarial_evidence_present"],
        },
    )

    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=3.25, tests_run=["focused"])

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["repair_gate_decision_v4_ready"] is True
    assert artifact["repair_gate_state"] == "blocked_receipt_precondition"
    assert artifact["missing_artifacts"] == []
    assert artifact["duration_s"] == pytest.approx(1.25)
    assert artifact["allowed_repair_attempt_budget"] == {
        "enabled": False,
        "max_total_repair_attempts": 0,
        "max_attempts_per_row": 0,
        "max_distinct_rows": 0,
        "requires_mandated_local_sota": True,
        "requires_exact_authority_acceptance": True,
        "requires_certificate_repair_call_ready": True,
        "stop_on_first_exact_accept_per_row": True,
        "no_headline_claim_from_gate_alone": True,
        "disabled_reason": "blocked_receipt_precondition",
    }
    predicates = artifact["unblocking_predicates"]
    assert predicates["receipt_smoke_clean_rerun_allowed"]["passed"] is False
    assert predicates["controlled_invariance_passed"]["passed"] is True
    assert predicates["clean_verifier_not_flagged_adversarial"]["passed"] is False
    assert predicates["headline_claim_allowed_for_verifier_metrics"]["passed"] is False
    assert predicates["false_accept_gate_acceptable"]["passed"] is False
    assert predicates["certificate_repair_call_ready"]["passed"] is False
    assert "exp3179.clean_rerun_allowed is not true" in artifact["blocker_reasons"]
    assert "exp3181.flagged_adversarial is not false" in artifact["blocker_reasons"]
    assert "exp3183.repair_call_ready is not true" in artifact["blocker_reasons"]
    assert artifact["inference_substrate"]["live_model_calls"] == 0
    assert artifact["inference_substrate"]["repair_calls"] == 0
    assert artifact["honest_verdict"].startswith("blocked_receipt_precondition:")
    assert artifact["tests_run"] == ["focused"]


def test_scenario_verify_3184_unblocks_only_with_all_predicates(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3184: all predicates passing opens only a bounded ladder."""

    _write_standard_sources(tmp_path)

    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/out.json"),
        started_s=1.0,
        now_s=1.5,
        tests_run=["focused"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / "results/out.json"
    assert artifact["repair_gate_state"] == "unblocked_for_bounded_repair_ladder"
    assert all(row["passed"] is True for row in artifact["unblocking_predicates"].values())
    assert artifact["blocker_reasons"] == []
    assert artifact["allowed_repair_attempt_budget"]["enabled"] is True
    assert artifact["allowed_repair_attempt_budget"]["max_total_repair_attempts"] == 4
    assert artifact["allowed_repair_attempt_budget"]["max_attempts_per_row"] == 2
    assert artifact["allowed_repair_attempt_budget"]["max_distinct_rows"] == 2
    assert artifact["allowed_repair_attempt_budget"]["disabled_reason"] == ""
    assert artifact["honest_verdict"].startswith("complete:")


def test_scenario_verify_3184_missing_required_artifact_fails_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-3184: absence is exposed separately from failed predicates."""

    _write_standard_sources(tmp_path, include_certificate=False)

    artifact = mod.build_artifact(tmp_path)

    assert artifact["repair_gate_state"] == "blocked_missing_artifact"
    assert mod.EXP3183_REL_PATH.as_posix() in artifact["missing_artifacts"]
    assert any("required artifact missing" in item for item in artifact["blocker_reasons"])
    assert artifact["allowed_repair_attempt_budget"]["enabled"] is False
    assert artifact["honest_verdict"].startswith("blocked_missing_artifact:")


@pytest.mark.parametrize(
    ("kwargs", "expected_state", "expected_blocker"),
    [
        (
            {"invariance": {"controlled_invariance_passed": False}},
            "blocked_controlled_invariance",
            "exp3180.controlled_invariance_passed is not true",
        ),
        (
            {"clean": {"flagged_adversarial": True}},
            "blocked_clean_verifier_flagged",
            "exp3181.flagged_adversarial is not false",
        ),
        (
            {"clean": {"headline_claim_allowed": False}},
            "blocked_headline_claim_blocked",
            "exp3181.headline_claim_allowed is not true",
        ),
        (
            {"clean": {"false_accept_rate": 0.2}},
            "blocked_false_accept_gate",
            "clean verifier false-accept gate is not acceptable",
        ),
        (
            {"certificate": {"repair_call_ready": False}},
            "blocked_certificate_not_ready",
            "exp3183.repair_call_ready is not true",
        ),
    ],
)
def test_scenario_verify_3184_specific_blocked_states(
    tmp_path: Path,
    kwargs: dict[str, Any],
    expected_state: str,
    expected_blocker: str,
) -> None:
    """REQ-VERIFY-3184: each failed predicate maps to an actionable machine state."""

    _write_standard_sources(tmp_path, **kwargs)

    artifact = mod.build_artifact(tmp_path)

    assert artifact["repair_gate_state"] == expected_state
    assert any(expected_blocker in item for item in artifact["blocker_reasons"])
    assert artifact["honest_verdict"].startswith(f"{expected_state}:")


def test_req_verify_3184_helpers_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-3184: helper edges keep malformed source evidence blocked."""

    malformed = tmp_path / "bad.json"
    malformed.write_text("{bad json", encoding="utf-8")

    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.read_json_object(malformed) == {}
    assert mod.finite_rate("not-a-rate") is None
    assert mod.finite_rate(float("inf")) is None
    assert mod.finite_rate(0.25) == pytest.approx(0.25)
    assert mod.all_predicates_passed({"a": {"passed": True}, "b": {"passed": False}}) is False
    assert mod.all_predicates_passed({"a": {"passed": True}}) is True
    malformed_predicates = {name: {"passed": True} for name in mod.PREDICATE_ORDER}
    malformed_predicates["unexpected_extra_gate"] = {"passed": False}
    assert mod.repair_gate_state([], malformed_predicates) == "blocked_other"


def test_req_verify_3184_validation_rejects_unsafe_shapes() -> None:
    """REQ-VERIFY-3184: validation rejects non-terminal or live-call artifacts."""

    artifact = {
        "repair_gate_decision_v4_ready": True,
        "repair_gate_state": "unblocked_for_bounded_repair_ladder",
        "unblocking_predicates": {
            "receipt_smoke_clean_rerun_allowed": {"passed": True},
            "controlled_invariance_passed": {"passed": True},
            "clean_verifier_not_flagged_adversarial": {"passed": True},
            "headline_claim_allowed_for_verifier_metrics": {"passed": True},
            "false_accept_gate_acceptable": {"passed": True},
            "certificate_repair_call_ready": {"passed": True},
        },
        "blocker_reasons": [],
        "missing_artifacts": [],
        "allowed_repair_attempt_budget": mod.allowed_repair_attempt_budget(
            "unblocked_for_bounded_repair_ladder"
        ),
        "source_artifacts": [],
        "inference_substrate": {"live_model_calls": 0, "repair_calls": 0},
        "honest_verdict": "complete: valid",
    }

    mod.validate_artifact(artifact)
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({k: v for k, v in artifact.items() if k != "honest_verdict"})
    with pytest.raises(ValueError, match="allowed state"):
        mod.validate_artifact({**artifact, "repair_gate_state": "maybe"})
    with pytest.raises(ValueError, match="failed predicates"):
        mod.validate_artifact(
            {
                **artifact,
                "unblocking_predicates": {
                    **artifact["unblocking_predicates"],
                    "certificate_repair_call_ready": {"passed": False},
                },
            }
        )
    with pytest.raises(ValueError, match="must not perform live"):
        mod.validate_artifact({**artifact, "inference_substrate": {"live_model_calls": 1}})
    with pytest.raises(ValueError, match="budget"):
        mod.validate_artifact(
            {**artifact, "allowed_repair_attempt_budget": {"enabled": False}}
        )
    with pytest.raises(ValueError, match="success prefix"):
        mod.validate_artifact({**artifact, "honest_verdict": "blocked_valid"})
