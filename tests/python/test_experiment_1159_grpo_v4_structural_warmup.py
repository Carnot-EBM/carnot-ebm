"""Tests for Exp 1159 GRPO v4 structural warm-up.

Spec: REQ-LEARN-1159, SCENARIO-LEARN-1159, SCENARIO-LEARN-1160.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

from carnot.training import grpo_structural_warmup as gsw


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "experiment_1159_grpo_v4_structural_warmup.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("exp1159", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp1159"] = module
    spec.loader.exec_module(module)
    return module


def test_phase_configs_match_structural_warmup_spec():
    """REQ-LEARN-1159-3/4: warm-up is reflection-only; full phase mixes rewards."""
    warmup, full = gsw.build_structural_warmup_phase_configs()

    assert warmup.name == "warmup"
    assert warmup.wall_budget_s == 300.0
    assert warmup.group_size_n == 8
    assert warmup.thinkprm_weight == 0.0
    assert warmup.reflection_weight == 1.0
    assert warmup.diversity_penalty_enabled is True
    assert warmup.proxy_reuse_enabled is False

    assert full.name == "full"
    assert full.wall_budget_s == 900.0
    assert full.group_size_n == 8
    assert full.thinkprm_weight == 1.0
    assert full.reflection_weight == 0.3
    assert full.diversity_penalty_enabled is True
    assert full.proxy_reuse_enabled is True


def test_phase_rewards_follow_warmup_then_full_mixing():
    """SCENARIO-LEARN-1159: Phase 1 uses r_reflect; Phase 2 uses mixed reward."""
    warmup, full = gsw.build_structural_warmup_phase_configs()

    phase1 = gsw.combine_phase_reward_groups(warmup, [0.4, 0.7], [0.5, -0.25])
    phase2 = gsw.combine_phase_reward_groups(full, [0.4, 0.7], [0.5, -0.25])

    assert phase1 == [0.5, -0.25]
    assert phase2 == [0.55, 0.625]


def test_phase_reward_length_mismatch_raises():
    """REQ-LEARN-1159-4: aligned ThinkPRM/reflection groups are required."""
    _, full = gsw.build_structural_warmup_phase_configs()

    with pytest.raises(ValueError, match="length mismatch"):
        gsw.combine_phase_reward_groups(full, [0.1, 0.2], [0.3])


def test_improvement_vs_exp1129_is_derived_metric():
    """REQ-LEARN-1159-6: improvement_vs_exp1129 subtracts the 0.0851 baseline."""
    assert math.isclose(gsw.improvement_vs_exp1129(0.09), 0.0049)
    assert math.isclose(gsw.improvement_vs_exp1129(0.0286), -0.0565)


def test_structural_warmup_honest_verdict_mapping():
    """REQ-LEARN-1159-6: Exp 1159 emits only the canonical verdict labels."""
    assert gsw.derive_structural_warmup_verdict(False, 1.0) == "blocked_no_dualgpu"
    assert gsw.derive_structural_warmup_verdict(True, 0.09) == "structural_warmup_above_0851"
    assert gsw.derive_structural_warmup_verdict(True, 0.02) == "positive_below_exp1129"
    assert gsw.derive_structural_warmup_verdict(True, 0.0) == "neutral"
    assert gsw.derive_structural_warmup_verdict(True, -0.01) == "negative_regression"


def test_required_artifact_fields_for_blocked_dualgpu():
    """SCENARIO-LEARN-1160: blocked artifact keeps every v4 schema field."""
    fields = gsw.build_structural_warmup_artifact_fields(
        cuda_device_count=0,
        dualgpu_used=False,
        training_wall_budget_hit=False,
        advantage_stdev_warmup=0.0,
        advantage_stdev_full=0.0,
        n_eval_questions=0,
        baseline_fraction_correct=0.0,
        trained_fraction_correct=0.0,
        improvement_over_baseline=0.0,
    )

    for key in gsw.REQUIRED_ARTIFACT_FIELDS:
        assert key in fields
    assert fields["dualgpu_used"] is False
    assert fields["cuda_device_count"] == 0
    assert fields["warmup_seconds"] == 300
    assert fields["training_seconds"] == 900
    assert fields["reflection_weight"] == 0.3
    assert fields["structural_warmup_used"] is True
    assert fields["grpo_v4_honest_result"] is True
    assert fields["honest_verdict"] == "blocked_no_dualgpu"


def test_required_artifact_fields_for_success_compute_delta():
    """REQ-LEARN-1159-5: success artifacts report improvement_vs_exp1129."""
    fields = gsw.build_structural_warmup_artifact_fields(
        cuda_device_count=2,
        dualgpu_used=True,
        training_wall_budget_hit=False,
        advantage_stdev_warmup=0.2,
        advantage_stdev_full=0.25,
        n_eval_questions=50,
        baseline_fraction_correct=0.2,
        trained_fraction_correct=0.3,
        improvement_over_baseline=0.1,
    )

    assert fields["dualgpu_used"] is True
    assert fields["advantage_stdev_warmup"] == 0.2
    assert fields["advantage_stdev_full"] == 0.25
    assert fields["n_eval_questions"] == 50
    assert math.isclose(fields["improvement_vs_exp1129"], 0.0149)
    assert fields["honest_verdict"] == "structural_warmup_above_0851"


def test_script_blocked_artifact_uses_required_schema():
    """SCENARIO-LEARN-1160: script-level blocker writes exp1159 schema fields."""
    exp1159 = _load_script_module()

    artifact = exp1159._build_blocked_artifact(
        started_at="2026-05-02T00:00:00Z",
        cuda_device_count=0,
        sota_path=None,
        thinkprm_v2_auroc=0.9946,
        blocked_reason="torch.cuda.device_count() < 2 in active runtime",
    )

    assert artifact["experiment"] == 1159
    assert artifact["status"] == "blocked"
    assert artifact["dualgpu_used"] is False
    assert artifact["cuda_device_count"] == 0
    assert artifact["warmup_seconds"] == 300
    assert artifact["training_seconds"] == 900
    assert artifact["honest_verdict"] == "blocked_no_dualgpu"
    assert artifact["blocked_reason"] == "torch.cuda.device_count() < 2 in active runtime"
    for key in gsw.REQUIRED_ARTIFACT_FIELDS:
        assert key in artifact
