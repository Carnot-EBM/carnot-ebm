"""Tests for Exp 5177 GAP-4 scale-up and decentralization tier.

Spec refs: REQ-REPORT-5177, SCENARIO-REPORT-5177,
SCENARIO-REPORT-5177-LOCAL-GENERATOR.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5177_gap4_scaleup_decentralization_tier_v474 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _row(domain: str, index: int, *, vote: bool, gated: bool, demo: bool = True) -> JsonDict:
    return {
        "pilot_key": f"{domain}:{index}",
        "domain": domain,
        "task": f"{domain}_task_{index:02d}",
        "entry_i": index,
        "cluster_id": f"{domain}:task_{index:02d}",
        "vote_top2": vote,
        "gated_top2": gated,
        "demo_perfect": demo,
        "pred_is_gold": gated and not vote,
        "pred_in_pool": gated,
        "oracle_hit": vote or gated,
        "n_cands": 2,
    }


def _scale_rows() -> list[JsonDict]:
    rows: list[JsonDict] = []
    rows.extend(_row("arc1", i, vote=True, gated=True) for i in range(10))
    rows.extend(_row("arc1", 10 + i, vote=False, gated=True) for i in range(4))
    rows.extend(_row("arc1", 14 + i, vote=False, gated=False, demo=i < 15) for i in range(17))
    rows.extend(_row("arc2", i, vote=i < 2, gated=i < 2, demo=i < 16) for i in range(31))
    return rows


def _local_block() -> JsonDict:
    return {"value": "blocked_local_model_not_cached"}


def _prompt_entry(index: int = 0) -> JsonDict:
    return {
        "local_key": f"arc1:{index}:identity_plus_one",
        "domain": "arc1",
        "task": "identity_plus_one",
        "entry_i": index,
        "demos": [
            {"input": [[0, 1], [1, 0]], "output": [[1, 2], [2, 1]]},
            {"input": [[2, 0], [0, 2]], "output": [[3, 1], [1, 3]]},
        ],
        "test_input": [[1, 1], [0, 2]],
        "candidates": [
            {"grid": [[2, 2], [1, 3]], "correct": True, "votes": 1, "q_mean": 0.0},
            {"grid": [[1, 1], [0, 2]], "correct": False, "votes": 2, "q_mean": 0.0},
        ],
    }


def _plus_one_code(_prompt: str) -> str:
    return """
```python
def transform(grid):
    import numpy as np
    return (np.asarray(grid) + 1).tolist()
```
"""


def test_req_report_5177_spec_declares_scaleup_contract() -> None:
    """REQ-REPORT-5177: OpenSpec declares the scale-up artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-REPORT-5177",
        "SCENARIO-REPORT-5177",
        "SCENARIO-REPORT-5177-LOCAL-GENERATOR",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_report_5177_builds_actual_n_artifact_below_floor() -> None:
    """SCENARIO-REPORT-5177: achieved N and min-6 result are reported honestly."""

    artifact = mod.build_artifact(
        scaleup_rows=_scale_rows(),
        local_generator_arm_result=_local_block()["value"],
        duration_s=65.0,
        partial=False,
        checkpoint_path=mod.CHECKPOINT_RELATIVE_PATH,
        source_artifacts=[{"path": "results/arc3_gap4_rule_exec_verifier.json", "exists": True}],
        preconditions={
            "exp5161_flagged_adversarial": False,
            "qwen36_cached": True,
        },
        remaining_rows=[],
    )

    assert artifact["target_n"]["value"] == 180
    assert artifact["achieved_n"]["value"] == 62
    assert artifact["achieved_n_reason"] == "source_pool_exhausted_before_target"
    assert artifact["checkpoint_resume_used"]["value"] is True
    assert artifact["arc1_slice_result"]["n_entries"] == 31
    assert artifact["arc2_heldout_slice_result"]["n_entries"] == 31
    assert artifact["exact_test_discordant_wins"]["value"] == 4
    assert artifact["exact_test_discordant_losses"] == 0
    assert artifact["exact_test_p_value_two_sided"]["value"] == 0.125
    assert artifact["exact_test_passes_min6_rule"]["value"] is False
    assert artifact["gap4_status_recommendation"]["value"] == "scale_up_recommended"
    assert artifact["local_generator_arm_result"]["value"] == "blocked_local_model_not_cached"
    assert artifact["solve_provenance"]["value"] == "development_proxy"
    assert artifact["inference_substrate"]["value"] == "live_llm_inference"
    assert "n62" in artifact["honest_verdict"]
    assert "floor_not_crossed" in artifact["honest_verdict"]
    assert artifact["reproducibility_checksum"]["value"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_report_5177_filled_requires_min6_zero_loss() -> None:
    """REQ-REPORT-5177: filled is allowed only when the min-6 floor is crossed."""

    rows = [_row("arc1", i, vote=False, gated=True) for i in range(6)]
    artifact = mod.build_artifact(
        scaleup_rows=rows,
        local_generator_arm_result={"status": "completed_real_local_generator_subset"},
        duration_s=70.0,
        partial=False,
        checkpoint_path=mod.CHECKPOINT_RELATIVE_PATH,
        source_artifacts=[],
        preconditions={"exp5161_flagged_adversarial": False, "qwen36_cached": True},
    )

    assert artifact["exact_test_discordant_wins"]["value"] == 6
    assert artifact["exact_test_p_value_two_sided"]["value"] == pytest.approx(0.03125)
    assert artifact["exact_test_passes_min6_rule"]["value"] is True
    assert artifact["gap4_status_recommendation"]["value"] == "filled"
    assert artifact["honest_verdict"].startswith("success_")

    weak = mod.build_artifact(
        scaleup_rows=rows[:5],
        local_generator_arm_result={"status": "completed_real_local_generator_subset"},
        duration_s=70.0,
        partial=False,
        checkpoint_path=mod.CHECKPOINT_RELATIVE_PATH,
        source_artifacts=[],
        preconditions={"exp5161_flagged_adversarial": False, "qwen36_cached": True},
    )
    assert weak["exact_test_passes_min6_rule"]["value"] is False
    assert weak["gap4_status_recommendation"]["value"] == "scale_up_recommended"


def test_scenario_report_5177_local_generator_scores_real_prompted_program(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-5177-LOCAL-GENERATOR: local arm scores generated programs."""

    def extract_code(text: str) -> str | None:
        start = text.find("def transform")
        return None if start < 0 else text[start:].replace("```", "").strip()

    def safe_transform(code: str):
        namespace = {"np": __import__("numpy")}
        exec(code, namespace)  # noqa: S102 - tiny controlled unit-test code block.
        return namespace["transform"]

    def grid_hash(grid: Any) -> str:
        return json.dumps(grid, sort_keys=True)

    monkeypatch.setattr(mod, "_gap4_helpers", lambda: (extract_code, safe_transform, grid_hash))

    result = mod.run_local_generator_arm(
        root=tmp_path,
        prompt_entries=[_prompt_entry(0), _prompt_entry(1)],
        model_path_resolver=lambda _hf_id, _root: "/models/qwen.gguf",
        text_generator=_plus_one_code,
        target_n=2,
        soft_budget_s=1000.0,
        now=lambda: 0.0,
    )

    assert result["status"] == "completed_real_local_generator_subset"
    assert result["target_model"] == mod.LOCAL_GENERATOR_MODEL_ID
    assert result["target_n"] == 2
    assert result["achieved_n"] == 2
    assert result["real_generation"] is True
    assert result["identity_cache_smoke"] is False
    assert result["prompt_kind"] == "arc_transform_induction_from_demos"
    assert result["demo_perfect_count"] == 2
    assert result["pred_in_pool_count"] == 2
    assert result["pool_correct_count"] == 2
    assert result["precision"] == 1.0
    assert result["precision_counts"] == {"numerator": 2, "denominator": 2}
    assert len(result["scored_rows"]) == 2
    assert result["scored_rows"][0]["demo_fit"] == 1.0
    assert result["scored_rows"][0]["pred_is_pool_correct"] is True


def test_checkpoint_resume_and_blocked_precondition_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-5177: checkpoint/resume and upstream quarantine checks are honored."""

    calls = iter([0.0, 1000.0, 2000.0])
    attempted, partial, remaining = mod.run_rows_checkpointed(
        root=tmp_path,
        candidate_rows=_scale_rows()[:3],
        now=lambda: next(calls),
        soft_budget_s=1500.0,
    )
    assert partial is True
    assert [row["pilot_key"] for row in attempted] == ["arc1:0"]
    assert [row["pilot_key"] for row in remaining] == ["arc1:1", "arc1:2"]

    resumed, partial, remaining = mod.run_rows_checkpointed(
        root=tmp_path,
        candidate_rows=_scale_rows()[:3],
        now=lambda: 0.0,
        soft_budget_s=10_000.0,
    )
    assert partial is False
    assert remaining == []
    assert [row["pilot_key"] for row in resumed] == ["arc1:0", "arc1:1", "arc1:2"]

    monkeypatch.setattr(
        mod,
        "load_exp5161_precondition",
        lambda _root: {"passed": False, "flagged_adversarial": True},
    )
    artifact = mod.run(
        root=tmp_path,
        scaleup_row_loader=lambda _root: _scale_rows(),
        local_generator_runner=lambda _root, _rows: {"value": "blocked_local_model_not_cached"},
        source_artifact_loader=lambda _root: [],
        now=lambda: 0.0,
        sleep_fn=lambda _seconds: None,
    )
    assert artifact["honest_verdict"] == "blocked_upstream_still_flagged"
    assert artifact["achieved_n"]["value"] == 0


def test_artifact_schema_rejects_checksum_and_shape_errors() -> None:
    """REQ-REPORT-5177: schema validation protects required scale-up fields."""

    artifact = mod.build_artifact(
        scaleup_rows=_scale_rows(),
        local_generator_arm_result=_local_block()["value"],
        duration_s=65.0,
        partial=False,
        checkpoint_path=mod.CHECKPOINT_RELATIVE_PATH,
        source_artifacts=[],
        preconditions={"exp5161_flagged_adversarial": False, "qwen36_cached": True},
    )
    bad = dict(artifact)
    bad["honest_verdict"] = "not_terminal"
    bad["target_n"] = {"value": 0}
    bad["achieved_n"] = {"value": 181}
    bad["checkpoint_resume_used"] = {"value": False}
    bad["exact_test_discordant_wins"] = {"value": 6}
    bad["exact_test_discordant_losses"] = 0
    bad["exact_test_passes_min6_rule"] = {"value": False}
    bad["exact_test_p_value_two_sided"] = {"value": True}
    bad["gap4_status_recommendation"] = {"value": "maybe"}
    bad["solve_provenance"] = {"value": "fabricated"}
    bad["inference_substrate"] = {"value": "cached_only"}
    bad["random_seed"] = {"value": 0}
    bad["field_principles"] = {}
    bad["reproducibility_checksum"] = {"value": "sha256:bad"}

    errors = mod.artifact_schema_errors(bad)

    assert "honest_verdict_terminal_prefix" in errors
    assert "field_principles" in errors
    assert "target_n" in errors
    assert "achieved_n_bounds" in errors
    assert "checkpoint_resume_used_true" in errors
    assert "exact_test_passes_min6_rule" in errors
    assert "exact_test_p_value_two_sided" in errors
    assert "gap4_status_recommendation" in errors
    assert "solve_provenance_development_proxy" in errors
    assert "inference_substrate_live_llm_inference" in errors
    assert "random_seed" in errors
    assert "reproducibility_checksum" in errors

    missing = dict(artifact)
    missing.pop("duration_s")
    missing["reproducibility_checksum"] = {"value": mod.payload_checksum(missing)}
    assert "missing required field duration_s" in mod.artifact_schema_errors(missing)

    with pytest.raises(ValueError):
        mod.write_artifact(Path("/tmp"), bad)
