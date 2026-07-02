"""Tests for Exp 5152 DiffusionGemma gate reexamination.

Spec refs: REQ-VERIFY-5152, SCENARIO-VERIFY-5152,
SCENARIO-VERIFY-5152-SUCCESS, SCENARIO-VERIFY-5152-MISSING-5151.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import diffusiongemma_gate_reexamination_5152 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_known_issues(root: Path) -> None:
    path = root / mod.KNOWN_ISSUES_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Carnot -- Known Issues\n\n"
        "### 2026-06-30: EXECUTE the off-ARC distributional-energy verifier moat\n\n"
        "> **UPDATE 2026-06-30 (outer-loop): D1 is ANSWERED -- bounded null.** "
        "The trained verifier does NOT beat SC. **Planner: do NOT re-propose D1 "
        "(the trained-embedding-verifier-vs-SC question is decided). "
        "DiffusionGemma stays gated.**\n\n"
        "**Origin:** fixture entry.\n",
        encoding="utf-8",
    )


def _write_inputs(root: Path, *, exp5151: dict[str, Any] | None) -> None:
    _write_json(
        root / mod.D1_REL,
        {
            "experiment": "phase_d_musr_trained_verifier",
            "n_questions": 200,
            "trained_verifier_accuracy": 0.57,
            "sc_accuracy_matched": 0.555,
            "delta_vs_sc": 0.015,
            "delta_ci95": [-0.06, 0.085],
            "moat_realized": False,
            "verifier_is_oracle": False,
            "oracle_distinctness_note": "all-MiniLM embeddings plus LogisticRegression",
            "inference_substrate": "verifier_ensemble_against_cached_candidates",
            "honest_verdict": "complete_phase_d_trained_verifier_does_not_beat_sc",
        },
    )
    _write_json(
        root / mod.EXP4245_REL,
        {
            "experiment": "experiment_4245_arc_set_encoder_beats_vote",
            "held_out_task_n": 52,
            "oracle_distinct_beats_vote": True,
            "set_encoder_minus_vote_delta": 0.4423076923,
            "set_encoder_minus_vote_ci95": [0.3076923077, 0.5961538462],
            "pass_rates": {"vote_at_1": 0.25, "set_encoder_at_1": 0.6923076923},
            "oracle_at_k": 0.8269230769,
            "verifier_is_oracle": False,
            "honest_verdict": "complete: arc_oracle_distinct_set_encoder_beats_vote",
        },
    )
    if exp5151 is not None:
        _write_json(root / mod.EXP5151_REL, exp5151)
    _write_json(
        root / "results" / "diffusiongemma_energy_prior_extracted.json",
        {
            "experiment": "diffusiongemma_energy_prior_extracted",
            "status": "cached",
            "honest_verdict": "complete: energy_prior_extracted",
        },
    )
    module_path = root / "python" / "carnot" / "experiment_4260_diffusiongemma_energy_guided_preflight.py"
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text('"""fixture DiffusionGemma preflight module."""\n', encoding="utf-8")
    _write_known_issues(root)


def _not_hardened_5151() -> dict[str, Any]:
    return {
        "experiment": "experiment_5151_arc_oracle_distinct_hardening_v472",
        "acceptance_gate": True,
        "headline_outcome": "arc_set_encoder_win_not_hardened",
        "honest_verdict": (
            "complete_arc_set_encoder_win_not_hardened: +44pp win does not fully survive "
            "hardening; unresolved_axes=cross_game"
        ),
        "leak_audit_passed": True,
        "multiseed_delta_ci95": [0.4265639953, 0.4888206201],
        "cross_game_blocked_reason": "blocked_arc_game_ids_unrecoverable",
        "cross_game_replication_delta": None,
        "verifier_is_oracle": False,
    }


def _success_5151() -> dict[str, Any]:
    return {
        "experiment": "experiment_5151_arc_oracle_distinct_hardening_v472",
        "acceptance_gate": True,
        "headline_outcome": "arc_set_encoder_win_hardened",
        "honest_verdict": "success_arc_set_encoder_win_survives_hardening",
        "leak_audit_passed": True,
        "multiseed_delta_ci95": [0.42, 0.49],
        "cross_game_blocked_reason": None,
        "cross_game_replication_delta": 0.31,
        "cross_game_replication_ci95": [0.11, 0.50],
        "verifier_is_oracle": False,
    }


def test_req_5152_spec_declares_gate_reexamination_contract() -> None:
    """REQ-VERIFY-5152: OpenSpec declares the reexamination artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-5152",
        "SCENARIO-VERIFY-5152",
        "SCENARIO-VERIFY-5152-SUCCESS",
        "SCENARIO-VERIFY-5152-MISSING-5151",
        "python/carnot/reporting/diffusiongemma_gate_reexamination_5152.py",
        "results/experiment_5152_diffusiongemma_gate_reexamination_v472.json",
        "d1_claim_vs_exp4245_claim_same_hypothesis",
        "recommendation",
        "keep_gated",
        "ungate_now",
        "known_issues_corrigendum",
    ):
        assert marker in spec
    for principle in mod.FIELD_PRINCIPLES.values():
        assert principle in spec


def test_scenario_5152_musr_null_does_not_close_arc_gate(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5152: incomplete ARC hardening keeps gate for corrected reason."""

    _write_inputs(tmp_path, exp5151=_not_hardened_5151())

    artifact = mod.run(tmp_path)
    second = mod.run(tmp_path)

    mod.validate_artifact(artifact)
    assert artifact == second
    assert artifact["recommendation"]["value"] == "keep_gated"
    assert artifact["domain_conflation_found"] is True
    assert artifact["d1_claim_vs_exp4245_claim_same_hypothesis"]["value"] is True
    assert "MuSR reasoning-text" in artifact["d1_claim"]["claim_tested"]
    assert "ARC-1" in artifact["exp4245_claim"]["claim_tested"]
    assert artifact["exp5151_status"]["available"] is True
    assert artifact["exp5151_status"]["supports_ungating"] is False
    assert "not fully hardened" in artifact["recommendation"]["reason"]
    assert artifact["diffusiongemma_artifacts"]["json_result_count"] == 1
    known_issues = (tmp_path / mod.KNOWN_ISSUES_REL).read_text(encoding="utf-8")
    assert known_issues.count(mod.CORRIGENDUM_MARKER) == 1
    assert "DiffusionGemma stays gated" in known_issues
    assert "MuSR D1 null conflated domains" in known_issues
    assert (tmp_path / mod.OUTPUT_REL).exists()


def test_scenario_5152_successful_hardened_arc_win_ungates_now(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5152-SUCCESS: hardened ARC-domain win beats the MuSR null."""

    _write_inputs(tmp_path, exp5151=_success_5151())

    artifact = mod.run(tmp_path)

    mod.validate_artifact(artifact)
    assert artifact["recommendation"]["value"] == "ungate_now"
    assert artifact["exp5151_status"]["supports_ungating"] is True
    assert artifact["honest_verdict"].startswith("success_")
    assert "ARC-domain hardening succeeded" in artifact["recommendation"]["reason"]


def test_scenario_5152_missing_5151_keeps_gated_with_unhardened_caveat(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5152-MISSING-5151: Exp 4245 alone is not decision-grade."""

    _write_inputs(tmp_path, exp5151=None)

    artifact = mod.run(tmp_path)

    mod.validate_artifact(artifact)
    assert artifact["recommendation"]["value"] == "keep_gated"
    assert artifact["exp5151_status"]["available"] is False
    assert artifact["exp5151_status"]["reason"] == "exp5151_absent"
    assert "single-seed" in artifact["recommendation"]["reason"]


def test_req_5152_validation_and_loader_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-5152: schema validation rejects non-actionable artifacts."""

    with pytest.raises(FileNotFoundError):
        mod._read_json_object(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        mod._read_json_object(bad)

    _write_inputs(tmp_path, exp5151=_not_hardened_5151())
    artifact = mod.run(tmp_path)
    invalid_cases = [
        ({key: value for key, value in artifact.items() if key != "recommendation"}, "missing"),
        ({**artifact, "honest_verdict": "pending"}, "terminal-prefixed"),
        (
            {
                **artifact,
                "recommendation": {
                    "value": "maybe",
                    "principle": mod.FIELD_PRINCIPLES["recommendation"],
                },
            },
            "recommendation value",
        ),
        (
            {**artifact, "d1_claim_vs_exp4245_claim_same_hypothesis": {"value": True}},
            "principle",
        ),
        ({**artifact, "domain_conflation_found": {"value": True}}, "bare bool"),
        ({**artifact, "reproducibility_checksum": "nope"}, "checksum"),
        ({**artifact, "field_principles": {}}, "field_principles"),
        ({**artifact, "spec_refs": ["REQ-VERIFY-5152"]}, "spec_refs"),
    ]
    for payload, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(payload)

    assert mod._terminal_prefixed("complete: ok") is True
    assert mod._terminal_prefixed("success_ok") is True
    assert mod._terminal_prefixed("pending") is False
    assert mod._recommendation_from_5151({"available": True, "supports_ungating": False}) == (
        "keep_gated",
        "Exp 5151 is present but not fully hardened, so DiffusionGemma stays gated "
        "for missing decision-grade ARC-domain evidence rather than the MuSR D1 null.",
    )
