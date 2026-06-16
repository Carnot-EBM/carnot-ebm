"""Tests for Exp 4294 hardened ARC verifier efficiency versus strong judges.

Spec refs: REQ-VERIFY-4294, SCENARIO-VERIFY-4294.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import verifier_efficiency_harden_strong_judge_4294 as mod


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_gzip_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)


class FakeJudge:
    def __init__(self, choices: list[int], *, latency_s: float = 3.0) -> None:
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
                "prompt_tokens": 150,
                "completion_tokens": 24,
                "total_tokens": 174,
                "raw_output": f"Reasoning: compare grids. Final answer: {choice}",
                "parse_status": "parsed_final_answer",
            }
        )
        return choice


def _candidate(task_id: str, index: int, *, correct: bool, vote: float) -> dict[str, Any]:
    return {
        "candidate_id": f"{task_id}::candidate{index}",
        "candidate_index": index,
        "grid": [[index, 0], [0, index]],
        "is_correct": correct,
        "features": {
            "vote_weight": vote,
            "self_consistency_margin": vote - 0.5,
            "cell_confidence_mean": vote,
            "cell_confidence_margin": 0.1,
            "program_demo_fit": 1.0,
            "program_length": 10.0 + index,
        },
    }


def _make_repo(root: Path) -> Path:
    tasks = []
    manifest_rows = []
    task_rows = []
    for idx in range(3):
        task_id = f"gap3_stage2:T{idx}"
        correct_index = 1 if idx != 1 else 2
        energy_index = 1 if idx != 1 else 0
        tasks.append(
            {
                "task_id": task_id,
                "raw_task_id": f"T{idx}",
                "source_id": "gap3_stage2",
                "candidate_count": 3,
                "candidates": [
                    _candidate(task_id, 0, correct=correct_index == 0, vote=0.7),
                    _candidate(task_id, 1, correct=correct_index == 1, vote=0.2),
                    _candidate(task_id, 2, correct=correct_index == 2, vote=0.1),
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
                "set_encoder_score_margin_vs_vote": 0.2,
                "matched_control_candidate_id": f"{task_id}::candidate0",
                "matched_control_correct": correct_index == 0,
                "online_adapt_candidate_id": f"{task_id}::candidate1",
                "online_adapt_correct": correct_index == 1,
            }
        )

    _write_gzip_json(
        root / mod.exp4284.POOL_REL,
        {
            "schema": "fixture",
            "task_n": len(tasks),
            "candidate_n": sum(len(task["candidates"]) for task in tasks),
            "tasks": tasks,
        },
    )
    _write_json(root / mod.exp4284.MANIFEST_REL, {"schema": "fixture", "rows": manifest_rows})
    _write_json(
        root / mod.exp4284.CROSS_FAMILY_REL,
        {
            "honest_verdict": "complete: fixture",
            "held_out_task_n": len(tasks),
            "pass_rates": {"set_encoder_at_1": 2 / 3},
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
        root / mod.exp4284.SET_ENCODER_BUILD_REL,
        {
            "aggregator_trained": True,
            "learned_verifier_path": str(model_path),
            "verifier_is_oracle": False,
            "model_specs": {"architecture": "fixture"},
        },
    )
    (root / "models").mkdir(parents=True, exist_ok=True)
    (root / "models" / "qwen.gguf").write_bytes(b"qwen gguf")
    (root / "models" / "gemma.gguf").write_bytes(b"gemma gguf")
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "adversarial_verify.py").write_text(
        "import json, sys\nprint(json.dumps({'flag_count': 0, 'artifact': sys.argv[-1]}))\n",
        encoding="utf-8",
    )
    return root


def _spec(hf_id: str, path: Path, active_params_b: float) -> dict[str, Any]:
    return {
        "name": hf_id.rsplit("/", 1)[-1].removesuffix("-GGUF"),
        "hf_id": hf_id,
        "model_path": str(path),
        "active_params_b": active_params_b,
    }


def test_req_4294_spec_declares_hardened_efficiency_contract() -> None:
    """REQ-VERIFY-4294: OpenSpec declares the hardened multi-judge contract."""

    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4294",
        "SCENARIO-VERIFY-4294",
        "python/carnot/reporting/verifier_efficiency_harden_strong_judge_4294.py",
        "results/experiment_4294_verifier_efficiency_harden_strong_judge.py",
        "results/experiment_4294_verifier_efficiency_harden_strong_judge.json",
        "blocked_judge_models_not_cached",
        "efficiency_pareto_holds",
        "accuracy_energy_verifier",
        "accuracy_best_judge",
        "accuracy_delta_ci95",
        "cost_ratio",
        "verifier_is_oracle=false",
        mod.QWEN_JUDGE_ID,
        mod.GEMMA_JUDGE_ID,
    ):
        assert marker in spec
    for principle in mod.FIELD_PRINCIPLES.values():
        assert principle in spec


def test_scenario_4294_scores_two_strong_judges_and_picks_best(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4294: energy is compared with the best well-prompted judge."""

    root = _make_repo(tmp_path)
    specs = [
        _spec(mod.QWEN_JUDGE_ID, root / "models" / "qwen.gguf", 3.0),
        _spec(mod.GEMMA_JUDGE_ID, root / "models" / "gemma.gguf", 31.0),
    ]
    judges = {
        mod.QWEN_JUDGE_ID: FakeJudge([0, 0, 0], latency_s=2.0),
        mod.GEMMA_JUDGE_ID: FakeJudge([1, 2, 1], latency_s=4.0),
    }

    artifact = mod.run(
        root,
        judge_specs_provider=lambda: specs,
        llama_import_checker=lambda: True,
        judge_factory=lambda spec: judges[spec["hf_id"]],
        random_seed=mod.RANDOM_SEED,
        bootstrap_resamples=200,
        min_tasks=3,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["accuracy_energy_verifier"] == pytest.approx(2 / 3)
    assert artifact["accuracy_best_judge"] == pytest.approx(1.0)
    assert artifact["best_judge_id"] == mod.GEMMA_JUDGE_ID
    assert artifact["efficiency_pareto_holds"] is True
    assert artifact["cost_ratio"] <= 0.1
    assert artifact["selection_task_n"] == 3
    assert {row["judge_id"] for row in artifact["judge_metrics"]} == {
        mod.QWEN_JUDGE_ID,
        mod.GEMMA_JUDGE_ID,
    }
    assert artifact["judge_metrics"][1]["accuracy"] == pytest.approx(1.0)
    assert "few-shot" in artifact["model_specs"]["strong_prompt"]["summary"]
    assert artifact["model_specs"]["requested_judge_ggufs"] == [mod.QWEN_JUDGE_ID, mod.GEMMA_JUDGE_ID]
    assert artifact["adversarial_verify"]["returncode"] == 0
    assert all(row["judge_outputs"][mod.GEMMA_JUDGE_ID]["judge_correct"] for row in artifact["per_task"])

    written = json.loads((root / mod.OUTPUT_REL).read_text(encoding="utf-8"))
    assert written == artifact


def test_req_4294_blocks_only_when_all_judges_missing(tmp_path: Path) -> None:
    """REQ-VERIFY-4294: both missing judges write a blocked artifact before inference."""

    root = _make_repo(tmp_path)
    artifact = mod.run(
        root,
        judge_specs_provider=lambda: [],
        llama_import_checker=lambda: pytest.fail("llama import check must not run"),
        judge_factory=lambda _spec: pytest.fail("judge must not load"),
        random_seed=mod.RANDOM_SEED,
        min_tasks=3,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_judge_models_not_cached"
    assert artifact["efficiency_pareto_holds"] is False
    assert artifact["accuracy_energy_verifier"] == 0.0
    assert artifact["accuracy_best_judge"] == 0.0
    assert artifact["cost_ratio"] == 0.0
    assert artifact["acceptance_gate"] is True
    assert artifact["adversarial_verify"]["status"] == "not_run"
    assert [check["available"] for check in artifact["preconditions_checked"][:2]] == [False, False]


def test_req_4294_runs_available_judge_when_other_is_missing(tmp_path: Path) -> None:
    """REQ-VERIFY-4294: one missing judge is recorded as skipped, not a blocker."""

    root = _make_repo(tmp_path)
    qwen = _spec(mod.QWEN_JUDGE_ID, root / "models" / "qwen.gguf", 3.0)
    artifact = mod.run(
        root,
        judge_specs_provider=lambda: [qwen],
        llama_import_checker=lambda: True,
        judge_factory=lambda _spec: FakeJudge([1, 2, 1]),
        random_seed=mod.RANDOM_SEED,
        bootstrap_resamples=50,
        min_tasks=3,
        adversarial_runner=lambda _path: {"returncode": 0, "flag_count": 0},
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["best_judge_id"] == mod.QWEN_JUDGE_ID
    assert artifact["skipped_judge_ids"] == [mod.GEMMA_JUDGE_ID]
    assert len(artifact["judge_metrics"]) == 1
    assert any(
        check["resource"] == f"cached_judge_gguf:{mod.GEMMA_JUDGE_ID}"
        and check["available"] is False
        for check in artifact["preconditions_checked"]
    )


def test_req_4294_prompt_parser_validation_and_runtime_branches(tmp_path: Path) -> None:
    """REQ-VERIFY-4294: strong prompt parsing, validation, and runtime guards are explicit."""

    root = _make_repo(tmp_path)
    prompt = mod._build_strong_prompt("problem", ["Candidate 0: x", "Candidate 1: y"])
    assert "few-shot" in prompt.lower()
    assert "Final answer: <index>" in prompt
    assert mod._parse_strong_choice("Reasoning...\nFinal answer: 1", 2) == (1, "parsed_final_answer")
    assert mod._parse_strong_choice("I choose Candidate 0", 2) == (0, "parsed_candidate_reference")
    assert mod._parse_strong_choice("grid score 7 then 1", 2) == (1, "parsed_first_valid_integer")
    assert mod._parse_strong_choice("No answer", 2) == (0, "defaulted_no_valid_index")
    assert mod._parse_strong_choice("   ", 2) == (0, "defaulted_empty_output")

    class TokenizingLlama:
        def tokenize(self, text: bytes) -> list[int]:
            return list(range(len(text.split())))

        def __call__(self, prompt_text: str, **kwargs: Any) -> dict[str, Any]:
            assert "few-shot" in prompt_text.lower()
            assert kwargs["top_p"] == 1.0
            return {
                "choices": [{"text": "Reasoning...\nFinal answer: 1"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
            }

    class SplitOnlyLlama:
        def __call__(self, prompt_text: str, **_kwargs: Any) -> dict[str, Any]:
            assert "Candidate 0" in prompt_text
            return {"choices": [{"text": "I choose Candidate 0"}]}

    judge = mod.StrongPromptCostMeteredLlmJudge(
        {"model_path": str(root / "models" / "qwen.gguf")},
        llama_factory=lambda **_kwargs: TokenizingLlama(),
        clock=iter([1.0, 1.5]).__next__,
        max_tokens=8,
    )
    assert judge.judge("problem", ["Candidate 0", "Candidate 1"]) == 1
    assert judge.records[0]["latency_s"] == 0.5
    assert judge.records[0]["prompt_tokens"] == 11

    split_judge = mod.StrongPromptCostMeteredLlmJudge(
        {"model_path": str(root / "models" / "qwen.gguf")},
        llama_factory=lambda **_kwargs: SplitOnlyLlama(),
        clock=iter([2.0, 2.25]).__next__,
    )
    assert split_judge.judge("problem", ["Candidate 0"]) == 0
    assert split_judge.records[0]["parse_status"] == "parsed_candidate_reference"

    invalid = mod._blocked_artifact(
        "blocked_judge_models_not_cached",
        [],
        random_seed=mod.RANDOM_SEED,
        duration_s=0.1,
    )
    invalid_cases = [
        ({**invalid, "efficiency_pareto_holds": 1}, "bare bool"),
        ({**invalid, "accuracy_energy_verifier": {"value": 0.0}}, "bare float"),
        ({**invalid, "accuracy_best_judge": True}, "bare float"),
        ({**invalid, "cost_ratio": []}, "bare float"),
        ({**invalid, "verifier_is_oracle": True}, "verifier_is_oracle"),
        ({**invalid, "random_seed": False}, "bare int"),
        ({k: v for k, v in invalid.items() if k != "cost_ratio"}, "missing required fields"),
        ({**invalid, "accuracy_delta_ci95": [0.0]}, "accuracy_delta_ci95"),
        ({**invalid, "preconditions_checked": {}}, "preconditions_checked"),
        ({**invalid, "model_specs": []}, "model_specs"),
        ({**invalid, "judge_metrics": {}}, "judge_metrics"),
        ({**invalid, "reproducibility_checksum": "short"}, "sha256"),
        ({**invalid, "field_principles": {}}, "field_principles"),
        ({**invalid, "spec_refs": []}, "spec_refs"),
        ({**invalid, "honest_verdict": "missing prefix"}, "terminal prefix"),
    ]
    for payload, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(payload)

    missing_path_artifact = mod.run(
        root,
        judge_specs_provider=lambda: [_spec(mod.QWEN_JUDGE_ID, root / "missing.gguf", 3.0)],
        random_seed=mod.RANDOM_SEED,
        min_tasks=3,
    )
    assert missing_path_artifact["honest_verdict"] == "blocked_judge_models_not_cached"

    llama_block = mod.run(
        root,
        judge_specs_provider=lambda: [_spec(mod.QWEN_JUDGE_ID, root / "models" / "qwen.gguf", 3.0)],
        llama_import_checker=lambda: False,
        random_seed=mod.RANDOM_SEED,
        min_tasks=3,
    )
    assert llama_block["honest_verdict"] == "blocked_llama_cpp_unavailable"

    runtime_block = mod.run(
        root,
        judge_specs_provider=lambda: [_spec(mod.QWEN_JUDGE_ID, root / "models" / "qwen.gguf", 3.0)],
        llama_import_checker=lambda: True,
        judge_factory=lambda _spec: (_ for _ in ()).throw(RuntimeError("boom")),
        random_seed=mod.RANDOM_SEED,
        min_tasks=3,
    )
    assert runtime_block["honest_verdict"] == "blocked_llm_judge_runtime"

    missing_candidates = tmp_path / "missing_candidates"
    (missing_candidates / "models").mkdir(parents=True)
    (missing_candidates / "models" / "qwen.gguf").write_bytes(b"qwen")
    upstream_block = mod.run(
        missing_candidates,
        judge_specs_provider=lambda: [
            _spec(mod.QWEN_JUDGE_ID, missing_candidates / "models" / "qwen.gguf", 3.0)
        ],
        llama_import_checker=lambda: True,
        random_seed=mod.RANDOM_SEED,
        min_tasks=3,
    )
    assert upstream_block["honest_verdict"] == "blocked_cross_family_candidates"

    cases, _, _, _, _ = mod.exp4284.load_selection_cases(root, min_tasks=1, max_tasks=1)

    class BadIndexNoRecordJudge:
        def judge(self, _problem: str, _candidates: list[str]) -> int:
            return 999

    selections, judge_cost = mod.run_strong_llm_judge(
        cases,
        BadIndexNoRecordJudge(),
        judge_id=mod.QWEN_JUDGE_ID,
    )
    assert selections[0]["judge_chosen_index"] == 0
    assert selections[0]["judge_cost"]["parse_status"] == "record_missing"
    assert judge_cost["total_tokens"] == 0

    (root / "scripts" / "adversarial_verify.py").write_text("print('not json')\n", encoding="utf-8")
    report = mod._run_adversarial_verify(root, root / mod.OUTPUT_REL)
    assert report["stdout"].strip() == "not json"


def test_req_4294_complete_artifact_verdict_branches(tmp_path: Path) -> None:
    """REQ-VERIFY-4294: complete artifacts expose all honest verdict branches."""

    root = _make_repo(tmp_path)
    energy_cost = {
        "total_wall_clock_s": 0.01,
        "candidate_forward_passes": 3,
        "flops_proxy": 100.0,
        "score_checksum_component": 1.0,
        "estimated_dollars_per_1k_selections": 1e-12,
    }
    judge_cost = {
        "total_wall_clock_s": 10.0,
        "total_tokens": 1000,
        "prompt_tokens": 900,
        "completion_tokens": 100,
    }
    common = {
        "checksums": {"fixture": "sha"},
        "model_path": root / "results" / "fixture_set_encoder.json",
        "build": {"model_specs": {"architecture": "fixture"}},
        "preconditions": [{"resource": "fixture", "available": True}],
        "skipped_judge_ids": [],
        "random_seed": mod.RANDOM_SEED,
        "bootstrap_resamples": 20,
        "duration_s": 1.0,
    }

    def result(*, energy_correct: bool, judge_correct: bool, cost: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "judge_id": mod.QWEN_JUDGE_ID,
            "judge_spec": {
                "hf_id": mod.QWEN_JUDGE_ID,
                "model_path": str(root / "models" / "qwen.gguf"),
                "active_params_b": 3.0,
            },
            "judge_cost": cost or judge_cost,
            "selections": [
                {
                    "task_id": "t",
                    "family_id": "f",
                    "fold": 0,
                    "candidate_count": 2,
                    "all_candidate_count": 2,
                    "energy_candidate_id": "e",
                    "energy_finalist_index": 0,
                    "energy_correct": energy_correct,
                    "judge_id": mod.QWEN_JUDGE_ID,
                    "judge_chosen_index": 1,
                    "judge_candidate_id": "j",
                    "judge_correct": judge_correct,
                    "judge_cost": {"raw_output": "Final answer: 1"},
                }
            ],
        }

    hardened = mod._complete_artifact(
        judge_results=[result(energy_correct=True, judge_correct=False)],
        energy_cost=energy_cost,
        **common,
    )
    assert "hardened_pareto_win" in hardened["honest_verdict"]

    stronger = mod._complete_artifact(
        judge_results=[result(energy_correct=False, judge_correct=True)],
        energy_cost=energy_cost,
        **common,
    )
    assert "stronger_judge_closes_accuracy_gap" in stronger["honest_verdict"]

    no_cost = mod._complete_artifact(
        judge_results=[
            result(
                energy_correct=True,
                judge_correct=True,
                cost={
                    "total_wall_clock_s": 0.01,
                    "total_tokens": 1,
                    "prompt_tokens": 1,
                    "completion_tokens": 0,
                },
            )
        ],
        energy_cost={**energy_cost, "estimated_dollars_per_1k_selections": 1.0},
        **common,
    )
    assert "no_cost_advantage" in no_cost["honest_verdict"]
