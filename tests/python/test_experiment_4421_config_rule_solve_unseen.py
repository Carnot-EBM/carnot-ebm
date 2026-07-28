"""Tests for Exp 4421 unseen config-rule solve.

Spec refs: REQ-REPORT-4421, SCENARIO-REPORT-4421.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_4421_config_rule_solve_unseen as mod


def _preconditions() -> dict[str, object]:
    return {
        "qwen_gguf_cached": True,
        "hip_llama_server_exists": True,
        "offline_env_loads": {"s5i5": True},
        "target_game_prior_best": 0,
    }


def _qwen_proposal() -> dict[str, object]:
    return {
        "model": "unsloth/gemma-4-31B-it-GGUF",
        "n_predict": 2048,
        "no_think": True,
        "raw_sample": "def is_win(controlled_markers, target_markers):",
        "grounded": True,
        "fires_on_win": True,
        "rejects_nonwins": True,
    }


def test_scenario_report_4421_run_writes_success_artifact(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4421: s5i5 L1 is counted only after offline reproduction."""

    result = mod.run(
        tmp_path,
        preconditions=_preconditions(),
        qwen_proposal=_qwen_proposal(),
        reproduce_fn=lambda _solution: {
            "game": "s5i5",
            "reached_level": 1,
            "claimed_level": 1,
            "reproduced": True,
            "mode": "offline_reproduction_gate_no_quota",
        },
        now=lambda: 42.0,
    )
    artifact = json.loads(result.read_text(encoding="utf-8"))

    assert artifact["honest_verdict"] == "success_s5i5_L1_offline_reproduced"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["new_levels_reproduced"] == 1
    assert artifact["verifier_is_oracle"] is True
    assert artifact["missing_verifier_gaps"] == []
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["solver"]["solution"] == ["h_extend"] * 7 + ["v_extend"] * 6
    assert artifact["grounded_win_condition"]["predicate"] == (
        "all target marker coordinates are occupied by controlled marker coordinates"
    )
    assert artifact["qwen_generation"]["grounded"] is True
    assert mod.artifact_schema_errors(artifact) == []


def test_req_report_4421_solver_uses_marker_coverage_predicate() -> None:
    """REQ-REPORT-4421: the path is derived from the grounded marker win rule."""

    controlled = [(9, 33), (30, 9)]
    targets = [(9, 51), (51, 9)]

    assert mod.is_win(controlled, targets) is False
    path = mod.derive_s5i5_l1_path(controlled, targets)
    predicted = mod.predicted_markers_after_path(controlled, path)

    assert path == ["h_extend"] * 7 + ["v_extend"] * 6
    assert predicted == [(9, 51), (51, 9)]
    assert mod.is_win(predicted, targets) is True


def test_req_report_4421_schema_validation_rejects_fabricated_success() -> None:
    """REQ-REPORT-4421: success artifacts must be oracle-backed and reproduced."""

    artifact = {
        "honest_verdict": "success_s5i5_L1_offline_reproduced",
        "offline_reproduced": False,
        "reproduced_levels": "1",
        "verifier_is_oracle": False,
        "missing_verifier_gaps": {},
        "random_seed": 4421,
        "reproducibility_checksum": "x",
    }

    errors = mod.artifact_schema_errors(artifact)

    assert "offline_reproduced must be true for success verdicts" in errors
    assert "reproduced_levels must be bare int" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "missing_verifier_gaps must be list" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors
