"""Tests for Exp 4811 S2-v3 corpus-wide structural-energy trust gate.

Spec refs: REQ-ARC-WMTE-4811,
SCENARIO-ARC-WMTE-4811-CORPUS-WIDE-TRUST-GATE.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_4811_structural_energy_s2v3_corpus_wide_trust_gate as mod
from scripts import adversarial_verify


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"


def _normalise(text: str) -> str:
    return " ".join(text.split())


def _spec_section() -> str:
    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-ARC-WMTE-4811")
    end = spec.index("### REQ-ARC-WMTE-4768", start)
    return spec[start:end]


def _manual_result(
    game: str,
    *,
    energy_recall: float,
    accuracy_recall: float,
    oracle_recall: float,
    effective: bool = True,
) -> mod.GameTrustGateResult:
    if not effective:
        energy_recall = accuracy_recall = oracle_recall
    candidate_rows = [
        {
            "candidate_name": f"{game}/accuracy_seed",
            "candidate_source": "deterministic_transition_induction",
            "genuinely_induced": True,
            "prefix_accuracy": 0.55,
            "heldout_accuracy": 0.31,
            "heldout_cell_recall": accuracy_recall,
            "offpath_structural_energy": 2.0,
            "binary_gate_pass": True,
        },
        {
            "candidate_name": f"{game}/energy_seed",
            "candidate_source": "programmatic_expert_induction",
            "genuinely_induced": True,
            "prefix_accuracy": 0.42,
            "heldout_accuracy": 0.28,
            "heldout_cell_recall": energy_recall,
            "offpath_structural_energy": 1.0,
            "binary_gate_pass": False,
        },
        {
            "candidate_name": f"{game}/headroom_seed",
            "candidate_source": "deterministic_transition_induction",
            "genuinely_induced": True,
            "prefix_accuracy": 0.25,
            "heldout_accuracy": 0.23,
            "heldout_cell_recall": oracle_recall,
            "offpath_structural_energy": 3.0,
            "binary_gate_pass": False,
        },
    ]
    return mod.GameTrustGateResult.from_candidate_rows(
        game=game,
        energy_selected_candidate=f"{game}/energy_seed",
        accuracy_gate_selected_candidate=f"{game}/accuracy_seed",
        candidate_rows=candidate_rows,
    )


def _attempted_25(effective_count: int, *, positive_delta: bool) -> list[mod.GameTrustGateResult]:
    rows: list[mod.GameTrustGateResult] = []
    for i in range(effective_count):
        rows.append(
            _manual_result(
                f"g{i:02d}",
                energy_recall=0.70 if positive_delta else 0.50,
                accuracy_recall=0.40 if positive_delta else 0.50,
                oracle_recall=0.80,
            )
        )
    for i in range(effective_count, 25):
        rows.append(
            _manual_result(
                f"g{i:02d}",
                energy_recall=0.50,
                accuracy_recall=0.50,
                oracle_recall=0.50,
                effective=False,
            )
        )
    return rows


def test_req_arc_wmte_4811_spec_declares_corpus_wide_contract() -> None:
    """REQ-ARC-WMTE-4811: OpenSpec declares the S2-v3 corpus-wide contract first."""

    section = _normalise(_spec_section())

    assert "REQ-ARC-WMTE-4811" in section
    assert "SCENARIO-ARC-WMTE-4811-CORPUS-WIDE-TRUST-GATE" in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert "n_available_games" in section
    assert "n_games_attempted" in section
    assert "max(10, ceil(0.6 * max(n_games_attempted, n_available_games)))" in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert field in section
        assert _normalise(principle["principle"]) in section


def test_req_arc_wmte_4811_required_effective_floor_uses_full_corpus() -> None:
    """REQ-ARC-WMTE-4811: the effective floor is tied to attempted and available games."""

    assert mod.required_effective_games(n_available_games=25, n_games_attempted=25) == 15
    assert mod.required_effective_games(n_available_games=25, n_games_attempted=5) == 15
    assert mod.required_effective_games(n_available_games=8, n_games_attempted=8) == 10
    assert mod.required_effective_games(n_available_games=12, n_games_attempted=20) == 12


def test_scenario_arc_wmte_4811_builds_success_artifact_for_powered_corpus(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4811-CORPUS-WIDE-TRUST-GATE: 15/25 effective can pass."""

    artifact = mod.build_artifact(
        _attempted_25(15, positive_delta=True),
        n_available_games=25,
        preconditions_checked={
            "offline_arcade": True,
            "offline_corpus": True,
            "world_model_verifier_import": True,
        },
        live_path_reachable=True,
        random_seed=4811,
        bootstrap_resamples=100,
        duration_s=1.25,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == mod.SUCCESS_VERDICT
    assert artifact["n_available_games"] == 25
    assert artifact["n_games_attempted"] == 25
    assert artifact["n_effective_games"] == 15
    assert artifact["required_effective_games"] == 15
    assert artifact["min_heldout_games"] == 15
    assert len(artifact["game_results"]) == 25
    assert artifact["s3_authorized"] is True

    flags: list[adversarial_verify.Flag] = []
    adversarial_verify.check_engine_selection_candidate_diversity(artifact, flags)
    assert flags == []

    out = mod.write_artifact(artifact, root=tmp_path)
    assert out == tmp_path / mod.RESULT_RELATIVE_PATH
    assert json.loads(out.read_text(encoding="utf-8")) == artifact


def test_req_arc_wmte_4811_bounded_and_inconclusive_paths() -> None:
    """REQ-ARC-WMTE-4811: bounded needs corpus power; under-diverse full attempts retire S2."""

    bounded = mod.build_artifact(
        _attempted_25(15, positive_delta=False),
        n_available_games=25,
        preconditions_checked={"offline_arcade": True, "offline_corpus": True},
        live_path_reachable=True,
        random_seed=4811,
        bootstrap_resamples=50,
    )
    mod.validate_artifact(bounded)
    assert bounded["honest_verdict"] == mod.BOUNDED_VERDICT
    assert bounded["positive_control_passed"] is True
    assert bounded["s3_authorized"] is False

    inconclusive = mod.build_artifact(
        _attempted_25(14, positive_delta=True),
        n_available_games=25,
        preconditions_checked={"offline_arcade": True, "offline_corpus": True},
        live_path_reachable=True,
        random_seed=4811,
        bootstrap_resamples=50,
    )
    mod.validate_artifact(inconclusive)
    assert inconclusive["honest_verdict"] == mod.INCONCLUSIVE_VERDICT
    assert inconclusive["n_games_attempted"] == 25
    assert inconclusive["n_effective_games"] == 14
    assert inconclusive["energy_minus_accuracy_delta"] is None
    assert inconclusive["retire_if_same_verdict"] is True


def test_req_arc_wmte_4811_validation_rejects_undercovered_or_fabricated_counts() -> None:
    """REQ-ARC-WMTE-4811: artifact counts must match the full corpus and row spreads."""

    artifact = mod.build_artifact(
        _attempted_25(15, positive_delta=True),
        n_available_games=25,
        preconditions_checked={"offline_arcade": True, "offline_corpus": True},
        live_path_reachable=True,
        random_seed=4811,
        bootstrap_resamples=50,
    )

    bad_effective = copy.deepcopy(artifact)
    bad_effective["n_effective_games"] = 1
    bad_effective["reproducibility_checksum"] = mod._checksum_payload(bad_effective)
    with pytest.raises(ValueError, match="n_effective_games"):
        mod.validate_artifact(bad_effective)

    bad_attempted = copy.deepcopy(artifact)
    bad_attempted["n_games_attempted"] = 5
    bad_attempted["reproducibility_checksum"] = mod._checksum_payload(bad_attempted)
    with pytest.raises(ValueError, match="n_games_attempted"):
        mod.validate_artifact(bad_attempted)


def test_req_arc_wmte_4811_preconditions_and_blocked_artifact(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4811: corpus discovery and blocked paths fail closed."""

    env = tmp_path / "environment_files"
    env.mkdir()
    for name in ("b_game", "a_game", "c_game"):
        (env / name).write_text("", encoding="utf-8")

    assert mod.available_offline_games(tmp_path) == ["a_game", "b_game", "c_game"]

    blocked = mod.build_blocked_artifact(
        "blocked_offline_corpus_missing",
        {"offline_arcade": True, "offline_corpus": False},
        random_seed=4811,
    )
    mod.validate_artifact(blocked)
    assert blocked["n_available_games"] == 0
    assert blocked["n_games_attempted"] == 0
    assert blocked["game_results"] == []


def test_req_arc_wmte_4811_committed_artifact_satisfies_schema() -> None:
    """REQ-ARC-WMTE-4811: committed deliverable is schema-valid and corpus-wide."""

    path = REPO / mod.RESULT_RELATIVE_PATH
    artifact = json.loads(path.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["result_path"] == mod.RESULT_RELATIVE_PATH
    assert artifact["n_available_games"] >= 20
    assert artifact["n_games_attempted"] == artifact["n_available_games"]
    assert artifact["verifier_is_oracle"] is False
