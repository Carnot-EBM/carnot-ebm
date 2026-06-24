"""Tests for Exp 4653 energy-fitness QD live generation measurement.

Spec refs: REQ-ARC-WMTE-4653, SCENARIO-ARC-WMTE-4653.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4653_energy_fitness_qd_generation_live as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _attempt(
    signature: str,
    *,
    solved: bool,
    winner_generated: bool = False,
    first_win: bool | None = None,
    depth: int = 0,
    actions: int | None = None,
) -> dict[str, Any]:
    return {
        "variant_signature": signature,
        "game": signature.split("~", 1)[0],
        "attempted": True,
        "solved": bool(solved),
        "winner_generated": bool(winner_generated),
        "first_win": bool(solved if first_win is None else first_win),
        "depth_of_live_solve": int(depth),
        "actions_to_win": actions,
        "reachable_headroom": True,
        "cell_recall_reachable": True,
        "cell_recall": 0.8714,
        "reproduction_gate": {
            "reproduced": bool(solved),
            "claimed_level": int(depth),
            "reached_level": int(depth),
        },
    }


def test_req_arc_wmte_4653_spec_declares_required_artifact_fields() -> None:
    """REQ-ARC-WMTE-4653: OpenSpec declares the QD live artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4653" in spec
    assert "SCENARIO-ARC-WMTE-4653" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4653_success_requires_energy_qd_beating_random(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4653: winner_generated must beat random mutation ablation."""

    baseline = [
        _attempt("tn36~cellrecall", solved=False),
        _attempt("sc25~cellrecall", solved=False),
    ]
    random_qd = [
        _attempt("tn36~cellrecall", solved=False),
        _attempt("sc25~cellrecall", solved=False),
    ]
    qd = [
        _attempt(
            "tn36~cellrecall",
            solved=True,
            winner_generated=True,
            depth=1,
            actions=2,
        ),
        _attempt("sc25~cellrecall", solved=False),
    ]

    artifact = mod.build_artifact(
        root=tmp_path,
        preconditions_checked=mod.ok_preconditions_for_tests(),
        search_measurement=mod.measurement_from_attempts(baseline),
        random_mutation_measurement=mod.measurement_from_attempts(random_qd),
        qd_measurement=mod.measurement_from_attempts(qd),
        live_path_check={"passed": True},
        parity_test={"passed": True},
        duration_s=1.0,
        n_bootstrap=0,
    )

    assert artifact["honest_verdict"] == "success: energy_fitness_qd_winner_generated_1"
    assert artifact["winner_generated"] is True
    assert artifact["winner_generated_count"] == 1
    assert artifact["live_solve_rate_qd"] == pytest.approx(0.5)
    assert artifact["live_solve_rate_search_baseline"] == 0.0
    assert artifact["solve_rate_delta"] == pytest.approx(0.5)
    assert artifact["random_mutation_ablation_passed"] is True
    assert artifact["qd_lift_ci"]["ci95"][0] > 0.0
    assert artifact["chosen_submitted_config"]["qd_generation_enabled"] is True
    assert mod.validate_artifact(artifact) == []


def test_scenario_arc_wmte_4653_honest_null_keeps_config_unchanged(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4653: no generated winner is an explicit P0.1-shadow null."""

    rows = [
        _attempt("tn36~cellrecall", solved=False),
        _attempt("sc25~cellrecall", solved=False),
    ]
    artifact = mod.build_artifact(
        root=tmp_path,
        preconditions_checked=mod.ok_preconditions_for_tests(),
        search_measurement=mod.measurement_from_attempts(rows),
        random_mutation_measurement=mod.measurement_from_attempts(rows),
        qd_measurement=mod.measurement_from_attempts(rows),
        live_path_check={"passed": True},
        parity_test={"passed": True},
        duration_s=1.0,
        n_bootstrap=0,
    )

    assert artifact["honest_verdict"] == (
        "complete: energy_fitness_qd_no_winner_generated_honest_null_gap_sharpened"
    )
    assert artifact["winner_generated"] is False
    assert artifact["winner_generated_count"] == 0
    assert artifact["solve_rate_delta"] == 0.0
    assert "honest no-value null" in artifact["null_delta_methodology_note"]
    assert "P0.1" in artifact["p01_shadow_note"]
    assert artifact["chosen_submitted_config"] == "unchanged"
    assert artifact["residual_bridge_gaps"]
    assert mod.validate_artifact(artifact) == []


def test_req_arc_wmte_4653_validation_rejects_bad_schema(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4653: schema validation catches unsafe field shapes."""

    rows = [_attempt("tn36~cellrecall", solved=False)]
    artifact = mod.build_artifact(
        root=tmp_path,
        preconditions_checked=mod.ok_preconditions_for_tests(),
        search_measurement=mod.measurement_from_attempts(rows),
        random_mutation_measurement=mod.measurement_from_attempts(rows),
        qd_measurement=mod.measurement_from_attempts(rows),
        live_path_check={"passed": True},
        parity_test={"passed": True},
        duration_s=1.0,
        n_bootstrap=0,
    )
    bad = dict(artifact)
    bad["honest_verdict"] = "not terminal"
    bad["verifier_is_oracle"] = True
    bad["field_principles"] = {}
    bad["winner_generated"] = "no"

    errors = mod.validate_artifact(bad)

    assert "honest_verdict must be terminal-prefixed" in errors
    assert "verifier_is_oracle must be false" in errors
    assert "field_principles mismatch" in errors
    assert "winner_generated must be a bare bool" in errors


def test_req_arc_wmte_4653_run_writes_artifact_with_fixture_runner(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4653: run() writes a stable artifact from matched fixture arms."""

    rows = [_attempt("tn36~cellrecall", solved=False)]
    artifact = mod.run(
        root=tmp_path,
        preconditions_checked=mod.ok_preconditions_for_tests(),
        arm_attempts={
            "search": rows,
            "random_mutation": rows,
            "energy_qd": rows,
        },
        live_path_check=lambda _root: {"passed": True},
        parity_test=lambda _root: {"passed": True},
        write=True,
        n_bootstrap=0,
    )

    written = tmp_path / mod.RESULT_RELATIVE_PATH
    assert written.exists()
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    assert mod.validate_artifact(artifact) == []
