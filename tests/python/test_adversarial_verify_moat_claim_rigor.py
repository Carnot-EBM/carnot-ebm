"""Tests for the moat-claim rigor adversarial-verify guard.

Spec refs: REQ-VERIFY-5008, SCENARIO-VERIFY-5008.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import scripts.adversarial_verify as av


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _base_win_fixture(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment": "synthetic_moat_rigor_clean",
        "honest_verdict": "success_verifier_moat_beats_sc_musr_0p120",
        "verifier_is_oracle": False,
        "headroom_present": True,
        "oracle_at_k": 0.82,
        "tuned_sc_accuracy": 0.60,
        "delta_vs_tuned_sc": 0.12,
        "n_flips_possible": 7,
        "paired_ci95": [0.03, 0.20],
        "mcnemar_p": 0.01,
        "random_seed": 20260630,
    }
    payload.update(overrides)
    return payload


def _moat_flags(payload: dict[str, Any]) -> list[dict[str, Any]]:
    flags: list[Any] = []
    av.check_moat_claim_rigor(payload, flags)
    return [flag.to_dict() for flag in flags if flag.kind == "MOAT_CLAIM_RIGOR"]


def _severities(payload: dict[str, Any]) -> set[str]:
    return {flag["severity"] for flag in _moat_flags(payload)}


def test_req_verify_5008_spec_declares_moat_claim_rigor_guard() -> None:
    """REQ-VERIFY-5008: OpenSpec declares the moat-rigor lint contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-5008" in spec
    assert "SCENARIO-VERIFY-5008" in spec
    assert "check_moat_claim_rigor" in spec
    assert "MOAT_CLAIM_RIGOR" in spec


def test_scenario_verify_5008_clean_tuned_sc_win_passes() -> None:
    """SCENARIO-VERIFY-5008: clean oracle-distinct tuned-SC win passes."""

    assert _moat_flags(_base_win_fixture()) == []


def test_scenario_verify_5008_circular_overclaim_is_critical() -> None:
    """SCENARIO-VERIFY-5008: verifier_is_oracle must be exactly false."""

    assert "critical" in _severities(_base_win_fixture(verifier_is_oracle=True))
    assert "critical" in _severities(_base_win_fixture(verifier_is_oracle=None))


def test_scenario_verify_5008_no_headroom_win_is_critical() -> None:
    """SCENARIO-VERIFY-5008: a positive win needs headroom-present evidence."""

    flags = _moat_flags(
        _base_win_fixture(
            headroom_present=False,
            oracle_at_k=0.65,
            tuned_sc_accuracy=0.60,
            n_flips_possible=0,
        )
    )

    assert {flag["severity"] for flag in flags} == {"critical"}
    assert any("headroom_present" in flag["detail"] for flag in flags)


def test_scenario_verify_5008_naive_sc_comparison_warns() -> None:
    """SCENARIO-VERIFY-5008: naive self-consistency is not the tuned baseline."""

    payload = _base_win_fixture(
        honest_verdict="success_verifier_moat_beats_naive_sc_musr_0p120",
        naive_sc_accuracy=0.60,
        delta_vs_naive_sc=0.12,
    )
    payload.pop("tuned_sc_accuracy")
    payload.pop("delta_vs_tuned_sc")

    flags = _moat_flags(payload)

    assert {flag["severity"] for flag in flags} == {"warn"}
    assert any("naive" in flag["detail"].lower() for flag in flags)


def test_scenario_verify_5008_no_paired_significance_is_critical() -> None:
    """SCENARIO-VERIFY-5008: positive beats-SC wins need paired significance."""

    payload = _base_win_fixture()
    payload.pop("paired_ci95")
    payload.pop("mcnemar_p")

    flags = _moat_flags(payload)

    assert {flag["severity"] for flag in flags} == {"critical"}
    assert any("paired_ci95" in flag["detail"] for flag in flags)


def test_scenario_verify_5008_no_headroom_null_warns() -> None:
    """SCENARIO-VERIFY-5008: no-headroom nulls are uninformative bounds."""

    payload = {
        "experiment": "synthetic_moat_rigor_no_headroom_null",
        "honest_verdict": "complete_moat_retired_bounded_does_not_beat_sc",
        "verifier_is_oracle": False,
        "headroom_present": False,
        "oracle_at_k": 0.60,
        "tuned_sc_accuracy": 0.60,
        "delta_vs_tuned_sc": 0.0,
        "n_flips_possible": 0,
        "paired_ci95": [-0.03, 0.02],
        "mcnemar_p": 1.0,
        "random_seed": 20260630,
    }

    flags = _moat_flags(payload)

    assert {flag["severity"] for flag in flags} == {"warn"}
    assert any("uninformative" in flag["detail"].lower() for flag in flags)


def test_scenario_verify_5008_artifact_scan_path_runs_guard(tmp_path: Path) -> None:
    """REQ-VERIFY-5008: verify_artifact wires in check_moat_claim_rigor."""

    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps(_base_win_fixture(verifier_is_oracle=True)),
        encoding="utf-8",
    )

    report = av.verify_artifact(artifact)

    flags = [flag for flag in report["flags"] if flag["kind"] == "MOAT_CLAIM_RIGOR"]
    assert flags
    assert flags[0]["severity"] == "critical"
