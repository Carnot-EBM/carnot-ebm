"""Pin REQ-ARC-WMTE-4743 adversarial_verify ARC null hardening.

Spec refs: REQ-ARC-WMTE-4743, SCENARIO-ARC-WMTE-4743-CARVEOUT-HARDENING.
"""

from __future__ import annotations

import json
from pathlib import Path

import scripts.adversarial_verify as av


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
EXP4726_NONDEGENERATE_NULL = (
    REPO / "results" / "experiment_4726_online_action_learning_driver_valid_test.json"
)
EXP4727_DECLARED_UNRUN_PROBE = (
    REPO / "results" / "experiment_4727_active_probe_disambiguation.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_flags(path: Path) -> list[dict[str, str]]:
    report = av.verify_artifact(path)
    assert report["loaded"] is True
    return report["flags"]


def _payload_lever_flags(payload: dict) -> list[dict[str, str]]:
    flags: list[av.Flag] = []
    av.check_lever_exercise_evidence(payload, flags)
    return [
        flag.to_dict()
        for flag in flags
        if flag.kind == av.LEVER_EXERCISE_EVIDENCE_DEGENERATE_KIND
    ]


def test_req_arc_wmte_4743_spec_declares_carveout_hardening() -> None:
    """REQ-ARC-WMTE-4743: OpenSpec declares the null-delta and probe hardening."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4743" in spec
    assert "SCENARIO-ARC-WMTE-4743-CARVEOUT-HARDENING" in spec
    assert "LEVER_EXERCISE_EVIDENCE_DEGENERATE" in spec
    assert "experiment_4743_adversarial_verify_carveout_hardening.json" in spec


def test_scenario_arc_wmte_4743_exp4726_nondegenerate_null_is_warn_not_critical() -> None:
    """SCENARIO-ARC-WMTE-4743: Exp 4726 equal first-win arms are WARN, not CRITICAL."""

    payload = _load(EXP4726_NONDEGENERATE_NULL)
    assert payload["arms_non_degenerate"] is True
    assert payload["positive_control_passed"] is True
    assert payload["null_delta_methodology_note"].strip()

    flags = _verify_flags(EXP4726_NONDEGENERATE_NULL)
    tautology_flags = [flag for flag in flags if flag["kind"] == "TAUTOLOGY"]

    assert tautology_flags
    assert all(flag["severity"] == "warn" for flag in tautology_flags)
    assert not any(
        flag["kind"] == "TAUTOLOGY" and flag["severity"] == "critical" for flag in flags
    )


def test_scenario_arc_wmte_4743_exp4727_declared_unrun_probe_is_flagged() -> None:
    """SCENARIO-ARC-WMTE-4743: Exp 4727 declared active probe did not run and is flagged."""

    payload = _load(EXP4727_DECLARED_UNRUN_PROBE)
    assert payload["probe_actions_taken"] == 0
    assert payload["hypothesis_posterior_built"] is False
    assert payload["posterior_entropy_reduction"] == 0.0

    flags = [
        flag
        for flag in _verify_flags(EXP4727_DECLARED_UNRUN_PROBE)
        if flag["kind"] == av.LEVER_EXERCISE_EVIDENCE_DEGENERATE_KIND
    ]

    assert flags
    assert any("probe_actions_taken=0" in flag["detail"] for flag in flags)
    assert any("hypothesis_posterior_built=False" in flag["detail"] for flag in flags)


def test_scenario_arc_wmte_4743_positive_exercise_null_not_flagged() -> None:
    """SCENARIO-ARC-WMTE-4743: positive probe exercise evidence survives a flat null."""

    payload = {
        "experiment": "arc_active_probe_positive_exercise_null",
        "schema": "carnot.arc.active_probe.valid_null.v1",
        "honest_verdict": "complete: active_probe_no_first_win_lift_honest_null",
        "inference_substrate": "active-probe replay over cached ARC transitions",
        "duration_s": 1.0,
        "target_game": "bp35",
        "probe_actions_taken": 3,
        "hypothesis_posterior_built": True,
        "posterior_entropy_reduction": 0.25,
        "active_probe_result": {
            "active_probe": True,
            "probe_actions_taken": 3,
            "hypothesis_posterior_built": True,
            "posterior_entropy_reduction": 0.25,
            "budget": 10,
        },
        "baseline_first_win": 0.04,
        "active_probe_first_win": 0.04,
        "active_probe_delta": 0.0,
        "random_seed": 4743,
        "reproducibility_checksum": "sha256:" + "7" * 64,
    }

    assert _payload_lever_flags(payload) == []


def test_scenario_arc_wmte_4743_unvalidated_flat_result_still_critical() -> None:
    """SCENARIO-ARC-WMTE-4743: equal ARC arms without all markers are not excused."""

    base_payload = {
        "experiment": "arc_online_action_learning_unvalidated_flat",
        "schema": "carnot.arc.online_action_learning.v1",
        "honest_verdict": "complete: online_action_learning_no_first_win_lift",
        "inference_substrate": "online_action_learning replay",
        "frozen_first_win": 0.04,
        "online_scratch_first_win": 0.04,
        "online_warm_first_win": 0.04,
    }

    cases = [
        {
            **base_payload,
            "arms_non_degenerate": False,
            "null_delta_methodology_note": "validated flat result",
            "positive_control_passed": True,
        },
        {
            **base_payload,
            "arms_non_degenerate": True,
            "null_delta_methodology_note": "",
            "positive_control_passed": True,
        },
        {
            **base_payload,
            "arms_non_degenerate": True,
            "null_delta_methodology_note": "unvalidated flat result",
            "positive_control_passed": False,
        },
        {
            "experiment": "non_arc_unvalidated_flat",
            "schema": "generic.metric.v1",
            "honest_verdict": "complete: generic_flat",
            "inference_substrate": "generic replay",
            "method_a_first_win": 0.04,
            "method_b_first_win": 0.04,
            "arms_non_degenerate": True,
            "null_delta_methodology_note": "markers are not enough outside ARC",
            "positive_control_passed": True,
        },
    ]

    for payload in cases:
        flags: list[av.Flag] = []
        av.check_tautology(payload, flags)

        assert any(
            flag.kind == "TAUTOLOGY" and flag.severity == "critical" for flag in flags
        )
