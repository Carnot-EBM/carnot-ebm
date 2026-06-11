"""Tests for Exp 4023 GAP-4 agreement-selector retirement closure.

Spec refs: REQ-VERIFY-4023, SCENARIO-VERIFY-4023.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import agreement_selector_retirement_4023 as mod


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fixture_root(tmp_path: Path) -> Path:
    _write_json(
        tmp_path / "results" / "arc3_gap4_chain_arms_adversarial_verify.json",
        {
            "honest_verdict": (
                "complete: gap4_chain_arms_confirmed_prereg_honestly_failed_"
                "coverage_lift_real_precision_uplift_not_established"
            ),
            "inference_substrate": "aggregation_from_upstream_artifacts",
        },
    )
    _write_json(
        tmp_path / "results" / "experiment_3999_gap4_precision_confirmation_v2.json",
        {
            "honest_verdict": "complete: protocol_preregistered_pending_execution",
            "protocol_preregistered": True,
            "total_codex_calls": 0,
            "n_agreement_events": 0,
            "primary_gate_passed": False,
            "agreement_is_selector_not_label": False,
        },
    )
    _write_json(
        tmp_path / "results" / "experiment_4009_gap4_precision_confirmation_v3.json",
        {
            "honest_verdict": "blocked_execution_floor_unmet",
            "protocol_preregistered": True,
            "execution_floor_met": False,
            "total_codex_calls": 0,
            "n_agreement_events": 0,
            "primary_gate_passed": False,
            "agreement_is_selector_not_label": False,
        },
    )
    _write_text(
        tmp_path / "openspec" / "change-proposals" / "research-roadmap-v370.md",
        "exp3988 precision confirmation was poison-skipped and never executed.\n",
    )
    _write_text(
        tmp_path / "ops" / "known-issues.md",
        "POWERED MULTI-CODEX-CALL EXPERIMENTS MUST BE TASK-SPLIT. "
        "A single agent task that must stay open >80 min will ALWAYS hit the hard cap.\n",
    )
    _write_text(
        tmp_path / "ops" / "verifier_registry.yaml",
        """
verifiers:
  - verifier_id: gap4_program_induction_stack
    agreement_role: confidence_label_only
    agreement_precision_selector: false
    selector_retirement:
      experiment: 4023
      retire_if_same_verdict: true
      retire_if_same_verdict_triggered: true
      safety_gate_kept: true
      no_precision_confirmation_v4_proposed: true
      retirement_rationale: "agreement is a confidence label only; not a precision selector"
""".lstrip(),
    )
    _write_text(
        tmp_path / "ops" / "verifier_gaps.md",
        """
### GAP-4 Agreement Selector Closure (Exp 4023)
- status: retired as selector R&D; agreement is a CONFIDENCE LABEL ONLY, not a precision selector.
- retire_if_same_verdict triggered after the poison-skip, exp3999, and exp4009 non-executions.
- shipped demo-fit execution safety-gate is KEPT.
- no precision-confirmation v4 is proposed.
""".lstrip(),
    )
    return tmp_path


def test_req_verify_4023_spec_anchor_exists() -> None:
    """REQ-VERIFY-4023: OpenSpec declares the closure artifact contract."""

    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")

    assert "REQ-VERIFY-4023" in spec
    assert "SCENARIO-VERIFY-4023" in spec
    assert "results/experiment_4023_retire_agreement_selector.json" in spec
    assert "agreement_role_after_retirement" in spec
    assert "safety_gate_kept" in spec


def test_scenario_verify_4023_builds_terminal_closure_from_evidence(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4023: evidence aggregation retires selector R&D and keeps the gate."""

    root = _fixture_root(tmp_path)

    artifact = mod.build_artifact(root, started_s=10.0, now_s=12.5)

    assert artifact["honest_verdict"] == (
        "complete: agreement_selector_retired_confidence_label_only"
    )
    assert artifact["registry_updated"] is True
    assert artifact["safety_gate_kept"] is True
    assert artifact["agreement_is_precision_selector"] is False
    assert artifact["agreement_role_after_retirement"] == "confidence_label_only"
    assert artifact["retired_r_and_d_line"] == "smart_selector_agreement_precision_confirmation"
    assert artifact["no_precision_confirmation_v4_proposed"] is True
    assert artifact["retire_if_same_verdict_triggered"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] == 2.5
    assert artifact["evidence_chain"] == [
        {
            "source": "results/arc3_gap4_chain_arms_adversarial_verify.json",
            "evidence_type": "adversarial_report",
            "finding": "precision_uplift_not_established_agreement_confidence_label_only",
            "honest_verdict": (
                "complete: gap4_chain_arms_confirmed_prereg_honestly_failed_"
                "coverage_lift_real_precision_uplift_not_established"
            ),
        },
        {
            "source": "openspec/change-proposals/research-roadmap-v370.md",
            "evidence_type": "non_execution",
            "experiment_id": 3988,
            "finding": "poison_skipped_precision_confirmation_never_executed",
        },
        {
            "source": "results/experiment_3999_gap4_precision_confirmation_v2.json",
            "evidence_type": "non_execution",
            "experiment_id": 3999,
            "finding": "protocol_preregistered_pending_execution_zero_calls_zero_agreement_events",
            "total_codex_calls": 0,
            "n_agreement_events": 0,
        },
        {
            "source": "results/experiment_4009_gap4_precision_confirmation_v3.json",
            "evidence_type": "non_execution",
            "experiment_id": 4009,
            "finding": "execution_floor_unmet_zero_calls_zero_agreement_events",
            "total_codex_calls": 0,
            "n_agreement_events": 0,
            "execution_floor_met": False,
        },
        {
            "source": "ops/known-issues.md",
            "evidence_type": "unfeedable_power_finding",
            "finding": "monolithic_powered_multi_call_confirmation_unfeedable_without_task_split",
        },
    ]
    assert artifact["registry_entry"]["agreement_precision_selector"] is False
    assert artifact["gaps_entry"]["safety_gate_kept"] is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    mod.validate_artifact(artifact)

    output_path = mod.write_artifact(root, artifact)
    assert output_path == root / mod.OUTPUT_REL_PATH
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact


def test_req_verify_4023_source_validation_rejects_unexecuted_mismatch(tmp_path: Path) -> None:
    """REQ-VERIFY-4023: non-execution evidence must stay honest and internally consistent."""

    root = _fixture_root(tmp_path)
    _write_json(
        root / "results" / "experiment_4009_gap4_precision_confirmation_v3.json",
        {
            "honest_verdict": "blocked_execution_floor_unmet",
            "protocol_preregistered": True,
            "execution_floor_met": True,
            "total_codex_calls": 0,
            "n_agreement_events": 0,
            "primary_gate_passed": False,
            "agreement_is_selector_not_label": False,
        },
    )

    with pytest.raises(ValueError, match="Exp 4009"):
        mod.build_artifact(root, started_s=0.0, now_s=1.0)


def test_req_verify_4023_current_deliverable_and_ops_docs_are_consistent() -> None:
    """REQ-VERIFY-4023: the checked-in artifact, registry, and gaps entry agree."""

    artifact_path = Path("results/experiment_4023_retire_agreement_selector.json")
    assert artifact_path.exists()
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["registry_updated"] is True
    assert artifact["safety_gate_kept"] is True
    assert artifact["no_precision_confirmation_v4_proposed"] is True
    assert artifact["evidence_chain"][2]["experiment_id"] == 3999
    assert artifact["evidence_chain"][3]["experiment_id"] == 4009

    registry_text = Path("ops/verifier_registry.yaml").read_text(encoding="utf-8")
    gaps_text = Path("ops/verifier_gaps.md").read_text(encoding="utf-8")
    assert mod.registry_has_closure(registry_text)
    assert mod.gaps_have_closure(gaps_text)
    assert "shipped demo-fit execution safety-gate is KEPT" in gaps_text
