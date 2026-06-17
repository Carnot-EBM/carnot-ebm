"""Tests for Exp 4316 budget-aware efficiency cascade router deployment.

Spec refs: REQ-VERIFY-4316, SCENARIO-VERIFY-4316.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import efficiency_cascade_router_deploy_4316 as mod
from carnot.reporting import verifier_efficiency_vs_llm_judge_4284 as exp4284


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_gzip_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)


class FakeJudge:
    def __init__(self, choices: list[int], *, latency_s: float = 2.0) -> None:
        self.choices = list(choices)
        self.latency_s = latency_s
        self.records: list[dict[str, Any]] = []

    def judge(self, problem: str, candidates: list[str]) -> int:
        assert "ARC held-out-family selection" in problem
        assert any("Candidate 0" in candidate for candidate in candidates)
        choice = self.choices.pop(0)
        self.records.append(
            {
                "chosen_index": choice,
                "latency_s": self.latency_s,
                "prompt_tokens": 120,
                "completion_tokens": 24,
                "total_tokens": 144,
                "raw_output": f"Grid reasoning fixture. Final answer: {choice}",
                "parse_status": "parsed_final_answer",
            }
        )
        return choice


def _features(vote: float, confidence: float) -> dict[str, float]:
    return {
        name: (
            float(vote)
            if name in {"vote_weight", "vote_weight_rank_fraction", "set_vote_max"}
            else float(confidence)
            if name in {"cell_confidence_mean", "cell_confidence_rank_fraction"}
            else float(confidence - 0.5)
            if name in {"self_consistency_margin", "cell_confidence_margin"}
            else 4.0
            if name in {"grid_cells", "set_candidate_count"}
            else 2.0
            if name in {"grid_height", "grid_width"}
            else 0.0
        )
        for name in mod.exp4244.FEATURE_NAMES
    }


def _candidate(task_id: str, index: int, *, correct: bool, vote: float, confidence: float) -> dict[str, Any]:
    return {
        "candidate_id": f"{task_id}::candidate{index}",
        "candidate_index": index,
        "grid": [[index, 0], [0, index]],
        "is_correct": correct,
        "features": _features(vote, confidence),
    }


def _make_repo(root: Path, *, task_n: int = 8) -> Path:
    tasks: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for idx in range(task_n):
        task_id = f"gap3_stage2:T{idx}"
        high_margin = idx % 4 in {0, 1}
        correct_index = 1 if high_margin else 2
        energy_index = 1 if high_margin else 0
        margin = 0.65 if high_margin else 0.02
        tasks.append(
            {
                "task_id": task_id,
                "raw_task_id": f"T{idx}",
                "source_id": "gap3_stage2",
                "candidate_count": 3,
                "candidates": [
                    _candidate(task_id, 0, correct=correct_index == 0, vote=0.7, confidence=0.25),
                    _candidate(task_id, 1, correct=correct_index == 1, vote=0.2, confidence=0.95),
                    _candidate(task_id, 2, correct=correct_index == 2, vote=0.1, confidence=0.9),
                ],
            }
        )
        manifest_rows.append(
            {
                "task_id": task_id,
                "raw_task_id": f"T{idx}",
                "source_id": "gap3_stage2",
                "source_kind": "induced",
                "family_id": f"family-{idx}",
                "fold": idx,
                "target_hash": "hash",
                "recovered_by": "fixture",
                "target_hash_recovered": True,
                "source_join_found": True,
            }
        )
        task_rows.append(
            {
                "task_id": task_id,
                "family_id": f"family-{idx}",
                "fold": idx,
                "oracle_hit": True,
                "vote_candidate_id": f"{task_id}::candidate0",
                "vote_correct": correct_index == 0,
                "set_encoder_candidate_id": f"{task_id}::candidate{energy_index}",
                "set_encoder_correct": correct_index == energy_index,
                "set_encoder_score_margin_vs_vote": margin,
                "matched_control_candidate_id": f"{task_id}::candidate0",
                "matched_control_correct": correct_index == 0,
                "online_adapt_candidate_id": f"{task_id}::candidate1",
                "online_adapt_correct": correct_index == 1,
            }
        )

    _write_gzip_json(
        root / exp4284.POOL_REL,
        {
            "schema": "fixture",
            "task_n": len(tasks),
            "candidate_n": sum(len(task["candidates"]) for task in tasks),
            "tasks": tasks,
        },
    )
    _write_json(root / exp4284.MANIFEST_REL, {"schema": "fixture", "rows": manifest_rows})
    _write_json(
        root / exp4284.CROSS_FAMILY_REL,
        {
            "honest_verdict": "complete: fixture",
            "held_out_task_n": len(tasks),
            "pass_rates": {"set_encoder_at_1": 0.5},
            "task_rows": task_rows,
            "verifier_is_oracle": False,
        },
    )
    model_path = root / "results" / "fixture_set_encoder.json"
    _write_json(
        model_path,
        {
            "model": {
                "model_type": "constant_set_score",
                "constant_score": 0.5,
                "feature_names": [],
                "feature_means": [],
                "feature_scales": [],
                "hidden_dim": 1,
                "temperature": 1.0,
            },
            "model_type": "constant_set_score",
            "feature_names": [],
            "verifier_is_oracle": False,
        },
    )
    _write_json(
        root / exp4284.SET_ENCODER_BUILD_REL,
        {
            "aggregator_trained": True,
            "learned_verifier_path": str(model_path),
            "verifier_is_oracle": False,
            "model_specs": {"architecture": "fixture"},
        },
    )
    (root / "models").mkdir(parents=True, exist_ok=True)
    (root / "models" / "judge.gguf").write_bytes(b"judge gguf")
    return root


def _spec(hf_id: str, path: Path) -> dict[str, Any]:
    return {
        "name": hf_id.rsplit("/", 1)[-1].removesuffix("-GGUF"),
        "hf_id": hf_id,
        "model_path": str(path),
        "active_params_b": 3.0,
    }


def _adversarial_clean(_path: Path) -> dict[str, Any]:
    return {"returncode": 0, "reports": [{"flag_count": 0, "flags": [], "max_severity": 0}]}


def test_req_4316_spec_declares_cascade_contract() -> None:
    """REQ-VERIFY-4316: OpenSpec declares the cascade deployment contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4316",
        "SCENARIO-VERIFY-4316",
        "python/carnot/reporting/efficiency_cascade_router_deploy_4316.py",
        "results/experiment_4316_efficiency_cascade_router_deploy.py",
        "results/experiment_4316_efficiency_cascade_router_deploy.json",
        "blocked_judge_models_not_cached",
        "blocked_window_exceeded",
        "cascade_dominates_controls",
        "accuracy_cascade",
        "accuracy_always_energy",
        "accuracy_always_judge",
        "cost_ratio_cascade",
        "escalation_rate",
        "pareto_curve",
        "verifier_is_oracle=false",
        mod.QWEN_JUDGE_ID,
        mod.GEMMA_JUDGE_ID,
    ):
        assert marker in spec
    for principle in mod.FIELD_PRINCIPLES.values():
        assert principle in spec


def test_scenario_4316_tunes_threshold_and_reports_policy_costs(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4316: cascade escalates low-margin rows and reports Pareto fields."""

    root = _make_repo(tmp_path)
    judge = FakeJudge([0, 0, 2, 2, 0, 0, 2, 2])
    artifact = mod.run(
        root,
        judge_specs_provider=lambda: [_spec(mod.QWEN_JUDGE_ID, root / "models" / "judge.gguf")],
        llama_import_checker=lambda: True,
        judge_factory=lambda spec: judge,
        trm_stand_down_checker=lambda _root: (True, "fixture stood down"),
        adversarial_runner=_adversarial_clean,
        random_seed=mod.RANDOM_SEED,
        bootstrap_resamples=200,
        min_tasks=8,
        min_eval_tasks=4,
        max_tasks=8,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["selection_task_n"] == 4
    assert artifact["threshold_tuning"]["tuning_task_n"] == 4
    assert artifact["cascade_threshold"] == pytest.approx(0.02)
    assert artifact["accuracy_always_energy"] == pytest.approx(0.5)
    assert artifact["accuracy_always_judge"] == pytest.approx(0.5)
    assert artifact["accuracy_cascade"] == pytest.approx(1.0)
    assert artifact["escalation_rate"] == pytest.approx(0.5)
    assert artifact["cost_always_energy"]["estimated_dollars_per_1k_selections"] < artifact["cost_cascade"][
        "estimated_dollars_per_1k_selections"
    ]
    assert artifact["cost_cascade"]["estimated_dollars_per_1k_selections"] < artifact["cost_always_judge"][
        "estimated_dollars_per_1k_selections"
    ]
    assert artifact["cascade_dominates_controls"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["model_specs"]["cascade_threshold"] == artifact["cascade_threshold"]
    assert artifact["model_specs"]["corpora"][0]["corpus_id"] == "arc_cross_family_existing_pool"
    assert artifact["adversarial_verify"]["returncode"] == 0
    assert {point["policy"] for point in artifact["pareto_curve"]["points"]} >= {
        "always_energy",
        "always_judge",
        "cascade",
    }
    checkpoint = json.loads((root / mod.CHECKPOINT_REL).read_text(encoding="utf-8"))
    assert len(checkpoint["judges"][mod.QWEN_JUDGE_ID]["selections"]) == 8
    written = json.loads((root / mod.OUTPUT_REL).read_text(encoding="utf-8"))
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]

    resumed = mod.run(
        root,
        judge_specs_provider=lambda: [_spec(mod.QWEN_JUDGE_ID, root / "models" / "judge.gguf")],
        llama_import_checker=lambda: True,
        judge_factory=lambda spec: FakeJudge([]),
        trm_stand_down_checker=lambda _root: (True, "fixture stood down"),
        adversarial_runner=_adversarial_clean,
        random_seed=mod.RANDOM_SEED,
        bootstrap_resamples=200,
        min_tasks=8,
        min_eval_tasks=4,
        max_tasks=8,
    )
    assert resumed["judge_metrics"][0]["checkpoint_resumed"] is True


def test_scenario_4316_blocks_when_judges_are_missing(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4316: missing judge caches produce a blocked terminal artifact."""

    _make_repo(tmp_path)
    artifact = mod.run(
        tmp_path,
        judge_specs_provider=lambda: [],
        llama_import_checker=lambda: True,
        trm_stand_down_checker=lambda _root: (True, "fixture stood down"),
        adversarial_runner=_adversarial_clean,
        min_tasks=8,
        min_eval_tasks=4,
        max_tasks=8,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_judge_models_not_cached"
    assert artifact["cascade_dominates_controls"] is False
    assert artifact["accuracy_cascade"] == 0.0
    assert artifact["pareto_curve"]["points"] == []
    assert artifact["acceptance_gate"] is True
