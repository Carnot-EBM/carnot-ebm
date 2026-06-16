"""Tests for Exp 4284 ARC verifier efficiency versus LLM-as-judge.

Spec refs: REQ-VERIFY-4284, SCENARIO-VERIFY-4284.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import verifier_efficiency_vs_llm_judge_4284 as mod


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_gzip_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)


class FakeJudge:
    def __init__(self, choices: list[int]) -> None:
        self.choices = list(choices)
        self.records: list[dict[str, Any]] = []

    def judge(self, problem: str, candidates: list[str]) -> int:
        assert problem
        assert candidates
        choice = self.choices.pop(0)
        self.records.append(
            {
                "chosen_index": choice,
                "latency_s": 2.0,
                "prompt_tokens": 90,
                "completion_tokens": 2,
                "total_tokens": 92,
                "raw_output": str(choice),
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
        "votes": int(vote * 100),
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
                "game_id": None,
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

    pool_path = root / mod.POOL_REL
    _write_gzip_json(
        pool_path,
        {
            "schema": "fixture",
            "task_n": len(tasks),
            "candidate_n": sum(len(task["candidates"]) for task in tasks),
            "reproducibility_checksum": "sha256:pool",
            "tasks": tasks,
        },
    )
    _write_json(root / mod.MANIFEST_REL, {"schema": "fixture", "rows": manifest_rows})
    _write_json(
        root / mod.CROSS_FAMILY_REL,
        {
            "honest_verdict": "complete: fixture",
            "held_out_task_n": len(tasks),
            "pass_rates": {"set_encoder_at_1": 2 / 3},
            "task_rows": task_rows,
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:cross-family",
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
            "reproducibility_checksum": "sha256:model",
        },
    )
    _write_json(
        root / mod.SET_ENCODER_BUILD_REL,
        {
            "aggregator_trained": True,
            "learned_verifier_path": str(model_path),
            "verifier_is_oracle": False,
            "model_specs": {"architecture": "fixture"},
        },
    )
    judge_path = root / "models" / "qwen.gguf"
    judge_path.parent.mkdir(parents=True, exist_ok=True)
    judge_path.write_bytes(b"gguf fixture")
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "adversarial_verify.py").write_text(
        "import json, sys\nprint(json.dumps({'flagged': [], 'path': sys.argv[-1]}))\n",
        encoding="utf-8",
    )
    return root


def test_req_4284_spec_declares_arc_efficiency_contract() -> None:
    """REQ-VERIFY-4284: OpenSpec declares the runner, fields, and principles."""

    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4284",
        "SCENARIO-VERIFY-4284",
        "python/carnot/reporting/verifier_efficiency_vs_llm_judge_4284.py",
        "results/experiment_4284_verifier_efficiency_vs_llm_judge.py",
        "results/experiment_4284_verifier_efficiency_vs_llm_judge.json",
        "blocked_judge_model_not_cached",
        "efficiency_parity_at_lower_cost",
        "accuracy_energy_verifier",
        "accuracy_llm_judge",
        "accuracy_delta_ci95",
        "cost_ratio",
        "verifier_is_oracle=false",
    ):
        assert marker in spec
    for principle in mod.FIELD_PRINCIPLES.values():
        assert principle in spec


def test_scenario_4284_measures_accuracy_ci_and_cost(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-4284: energy and LLM judge are compared on the same ARC finalists."""

    root = _make_repo(tmp_path)
    fake_judge = FakeJudge([0, 0, 0])
    artifact = mod.run(
        root,
        judge_spec_provider=lambda: {
            "name": "Qwen3.6-35B-A3B",
            "hf_id": mod.QWEN_JUDGE_ID,
            "model_path": str(root / "models" / "qwen.gguf"),
            "active_params_b": 3.0,
        },
        llama_import_checker=lambda: True,
        judge_factory=lambda _spec: fake_judge,
        random_seed=mod.RANDOM_SEED,
        bootstrap_resamples=200,
        min_tasks=3,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["accuracy_energy_verifier"] == pytest.approx(2 / 3)
    assert artifact["accuracy_llm_judge"] == pytest.approx(0.0)
    assert artifact["accuracy_delta_ci95"][1] >= 0.0
    assert artifact["cost_ratio"] <= 0.1
    assert artifact["efficiency_parity_at_lower_cost"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["selection_task_n"] == 3
    assert artifact["cost_accounting"]["llm_judge"]["total_tokens"] == 276
    assert artifact["model_specs"]["judge_gguf"]["hf_id"] == mod.QWEN_JUDGE_ID
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert artifact["spec_refs"] == ["REQ-VERIFY-4284", "SCENARIO-VERIFY-4284"]
    assert artifact["adversarial_verify"]["returncode"] == 0
    assert all(row["candidate_count"] >= 2 for row in artifact["per_task"])

    written = json.loads((root / mod.OUTPUT_REL).read_text(encoding="utf-8"))
    assert written == artifact


def test_req_4284_blocks_without_cached_judge_before_inference(tmp_path: Path) -> None:
    """REQ-VERIFY-4284: missing GGUF writes blocked artifact and skips judge calls."""

    root = _make_repo(tmp_path)
    artifact = mod.run(
        root,
        judge_spec_provider=lambda: None,
        llama_import_checker=lambda: pytest.fail("llama import check must not run without GGUF"),
        judge_factory=lambda _spec: pytest.fail("judge must not load without GGUF"),
        random_seed=mod.RANDOM_SEED,
        min_tasks=3,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_judge_model_not_cached"
    assert artifact["efficiency_parity_at_lower_cost"] is False
    assert artifact["accuracy_energy_verifier"] == 0.0
    assert artifact["accuracy_llm_judge"] == 0.0
    assert artifact["cost_ratio"] == 0.0
    assert artifact["acceptance_gate"] is True
    assert artifact["preconditions_checked"][0]["resource"] == "cached_judge_gguf"
    assert artifact["preconditions_checked"][0]["available"] is False
    assert artifact["adversarial_verify"]["status"] == "not_run"


def test_req_4284_validation_rejects_non_bare_gate_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-4284: required capstone gate fields stay bare scalars."""

    root = _make_repo(tmp_path)
    artifact = mod.run(
        root,
        judge_spec_provider=lambda: None,
        random_seed=mod.RANDOM_SEED,
        min_tasks=3,
    )
    invalid_cases = [
        ({**artifact, "efficiency_parity_at_lower_cost": 1}, "bare bool"),
        ({**artifact, "accuracy_energy_verifier": {"value": 0.0}}, "bare float"),
        ({**artifact, "accuracy_llm_judge": True}, "bare float"),
        ({**artifact, "cost_ratio": []}, "bare float"),
        ({**artifact, "verifier_is_oracle": True}, "verifier_is_oracle"),
        ({**artifact, "random_seed": False}, "bare int"),
        ({k: v for k, v in artifact.items() if k != "cost_ratio"}, "missing required fields"),
        ({**artifact, "accuracy_delta_ci95": [0.0]}, "accuracy_delta_ci95"),
        ({**artifact, "preconditions_checked": {}}, "preconditions_checked"),
        ({**artifact, "model_specs": []}, "model_specs"),
        ({**artifact, "reproducibility_checksum": "short"}, "sha256"),
        ({**artifact, "field_principles": {}}, "field_principles"),
        ({**artifact, "spec_refs": []}, "spec_refs"),
        ({**artifact, "honest_verdict": "missing prefix"}, "terminal prefix"),
    ]
    for payload, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(payload)


def test_req_4284_live_judge_wrapper_and_helper_branches(tmp_path: Path) -> None:
    """REQ-VERIFY-4284: helper branches are deterministic and cost-metered."""

    class TokenizingLlama:
        def tokenize(self, text: bytes) -> list[int]:
            return list(range(len(text.split())))

        def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
            assert "Candidate 0" in prompt
            assert kwargs["temperature"] == 0.0
            return {"choices": [{"text": "choose 1"}]}

    class SplitOnlyLlama:
        def __call__(self, prompt: str, **_kwargs: Any) -> dict[str, Any]:
            assert prompt
            return {"choices": [{"text": "no usable index"}]}

    class EmptyLlama:
        def __call__(self, _prompt: str, **_kwargs: Any) -> dict[str, Any]:
            return {"choices": [{"text": "   "}]}

    judge_path = tmp_path / "judge.gguf"
    judge_path.write_bytes(b"gguf")
    clock = iter([5.0, 5.25]).__next__
    judge = mod.CostMeteredLlmJudge(
        {"model_path": str(judge_path)},
        llama_factory=lambda **_kwargs: TokenizingLlama(),
        clock=clock,
    )
    assert judge.judge("problem", ["Candidate 0", "Candidate 1"]) == 1
    assert judge.records[0]["latency_s"] == 0.25
    assert judge.records[0]["chosen_index"] == 1

    fallback = mod.CostMeteredLlmJudge(
        {"model_path": str(judge_path)},
        llama_factory=lambda **_kwargs: SplitOnlyLlama(),
        clock=iter([1.0, 1.1]).__next__,
    )
    assert fallback.judge("problem", ["Candidate 0"]) == 0
    assert fallback.records[0]["completion_tokens"] == 3

    empty = mod.CostMeteredLlmJudge(
        {"model_path": str(judge_path)},
        llama_factory=lambda **_kwargs: EmptyLlama(),
        clock=iter([1.0, 1.1]).__next__,
    )
    with pytest.raises(RuntimeError, match="empty LLM judge output"):
        empty.judge("problem", ["Candidate 0"])

    assert mod._parse_choice("nothing useful", 3) == 0
    assert "Chosen index:" in mod._build_prompt("p", ["Candidate 0"])
    assert mod._grid_text([[0] * 1000]).endswith("...[truncated]")
    assert mod._bootstrap_ci95([], random_seed=1, resamples=10) == [0.0, 0.0]
    assert mod._bootstrap_ci95([1.0], random_seed=1, resamples=10) == [1.0, 1.0]
    assert mod._model_active_params_b({"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"}) == 3.0
    assert mod._model_active_params_b({"hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF"}) == 4.0
    assert mod._model_active_params_b({"hf_id": "unsloth/gemma-4-12B-it-GGUF"}) == 12.0
    assert mod._model_active_params_b({"hf_id": "unsloth/gemma-4-31B-it-GGUF"}) == 31.0
    assert mod._model_active_params_b({"hf_id": "unknown"}) == 3.0


def test_req_4284_blocked_and_malformed_input_branches(tmp_path: Path) -> None:
    """REQ-VERIFY-4284: malformed inputs and runtime preconditions fail terminally."""

    root = _make_repo(tmp_path)
    missing_model = mod.run(
        root,
        judge_spec_provider=lambda: {
            "name": "Qwen",
            "hf_id": mod.QWEN_JUDGE_ID,
            "model_path": str(root / "missing.gguf"),
        },
        random_seed=mod.RANDOM_SEED,
        min_tasks=3,
    )
    assert missing_model["honest_verdict"] == "blocked_judge_model_not_cached"

    llama_block = mod.run(
        root,
        judge_spec_provider=lambda: {
            "name": "Qwen",
            "hf_id": mod.QWEN_JUDGE_ID,
            "model_path": str(root / "models" / "qwen.gguf"),
        },
        llama_import_checker=lambda: False,
        random_seed=mod.RANDOM_SEED,
        min_tasks=3,
    )
    assert llama_block["honest_verdict"] == "blocked_llama_cpp_unavailable"
    assert llama_block["acceptance_gate"] is False

    runtime_block = mod.run(
        root,
        judge_spec_provider=lambda: {
            "name": "Qwen",
            "hf_id": mod.QWEN_JUDGE_ID,
            "model_path": str(root / "models" / "qwen.gguf"),
        },
        llama_import_checker=lambda: True,
        judge_factory=lambda _spec: (_ for _ in ()).throw(RuntimeError("boom")),
        random_seed=mod.RANDOM_SEED,
        min_tasks=3,
    )
    assert runtime_block["honest_verdict"] == "blocked_llm_judge_runtime"

    bad_json = root / "bad.json"
    bad_json.write_text("[]", encoding="utf-8")
    with pytest.raises(mod.BlockedRun, match="blocked_malformed_json_artifact"):
        mod._read_json_object(bad_json)

    bad_gzip = root / "bad.json.gz"
    _write_gzip_json(bad_gzip, [])
    with pytest.raises(mod.BlockedRun, match="blocked_malformed_candidate_pool"):
        mod._read_gzip_json_object(bad_gzip)

    with pytest.raises(mod.BlockedRun, match="blocked_energy_verifier_load"):
        mod._resolve_existing(root, "")
    with pytest.raises(mod.BlockedRun, match="blocked_energy_verifier_load"):
        mod._resolve_existing(root, "missing")

    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    with pytest.raises(mod.BlockedRun, match="blocked_cross_family_candidates"):
        mod._load_pool_candidates(empty_root)
    with pytest.raises(mod.BlockedRun, match="blocked_cross_family_manifest"):
        mod._load_manifest(empty_root)
    with pytest.raises(mod.BlockedRun, match="blocked_cross_family_candidates"):
        mod._load_cross_family_rows(empty_root)
    with pytest.raises(mod.BlockedRun, match="blocked_energy_verifier_load"):
        mod._load_energy_verifier(empty_root)

    with pytest.raises(mod.BlockedRun, match="blocked_insufficient_cross_family_tasks"):
        mod.load_selection_cases(root, min_tasks=99)


def test_req_4284_verdict_and_adversarial_branches(tmp_path: Path) -> None:
    """REQ-VERIFY-4284: verdict branches and adversarial parser are explicit."""

    root = _make_repo(tmp_path)
    cases, checksums, energy_model, model_path, build = mod.load_selection_cases(root, min_tasks=3)
    energy_cost = mod.measure_energy_forward_cost(cases, energy_model)
    judge_spec = {
        "name": "Qwen",
        "hf_id": mod.QWEN_JUDGE_ID,
        "model_path": str(root / "models" / "qwen.gguf"),
        "active_params_b": 3.0,
    }
    base_selection = [
        {
            "task_id": "t",
            "family_id": "f",
            "fold": 0,
            "candidate_count": 2,
            "all_candidate_count": 2,
            "energy_candidate_id": "e",
            "energy_finalist_index": 0,
            "energy_correct": False,
            "llm_chosen_index": 1,
            "llm_candidate_id": "j",
            "llm_correct": True,
            "judge_cost": {},
        }
    ]
    common = {
        "checksums": checksums,
        "model_path": model_path,
        "build": build,
        "preconditions": [{"resource": "fixture", "available": True}],
        "random_seed": mod.RANDOM_SEED,
        "bootstrap_resamples": 20,
        "duration_s": 1.0,
    }
    judge_cost = {"total_wall_clock_s": 1.0, "total_tokens": 100, "prompt_tokens": 98, "completion_tokens": 2}
    judge_win = mod._complete_artifact(
        selections=base_selection,
        energy_cost=energy_cost,
        judge_cost=judge_cost,
        judge_spec=judge_spec,
        **common,
    )
    assert "judge_is_more_accurate" in judge_win["honest_verdict"]

    no_cost = mod._complete_artifact(
        selections=[
            {
                **base_selection[0],
                "energy_correct": True,
                "llm_correct": True,
            }
        ],
        energy_cost={
            **energy_cost,
            "estimated_dollars_per_1k_selections": 1.0,
            "total_wall_clock_s": 2.0,
            "flops_proxy": 1e12,
        },
        judge_cost=judge_cost,
        judge_spec=judge_spec,
        **common,
    )
    assert "no_cost_advantage" in no_cost["honest_verdict"]

    (root / "scripts" / "adversarial_verify.py").write_text("print('not json')\n", encoding="utf-8")
    report = mod._run_adversarial_verify(root, root / mod.OUTPUT_REL)
    assert report["returncode"] == 0
    assert report["stdout"].strip() == "not json"


def test_req_4284_defensive_loader_and_selection_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-4284: loader edge cases stay blocked or deterministic."""

    root = _make_repo(tmp_path)

    _write_gzip_json(root / mod.POOL_REL, {"tasks": {}})
    with pytest.raises(mod.BlockedRun, match="blocked_cross_family_candidates"):
        mod._load_pool_candidates(root)
    _write_gzip_json(root / mod.POOL_REL, {"tasks": ["bad", {"task_id": "", "candidates": []}]})
    with pytest.raises(mod.BlockedRun, match="blocked_cross_family_candidates"):
        mod._load_pool_candidates(root)

    root = _make_repo(tmp_path / "manifest")
    _write_json(root / mod.MANIFEST_REL, {"rows": {}})
    with pytest.raises(mod.BlockedRun, match="blocked_cross_family_manifest"):
        mod._load_manifest(root)
    _write_json(root / mod.MANIFEST_REL, {"rows": []})
    with pytest.raises(mod.BlockedRun, match="blocked_cross_family_manifest"):
        mod._load_manifest(root)

    root = _make_repo(tmp_path / "cross")
    cross = json.loads((root / mod.CROSS_FAMILY_REL).read_text(encoding="utf-8"))
    cross["verifier_is_oracle"] = True
    _write_json(root / mod.CROSS_FAMILY_REL, cross)
    with pytest.raises(mod.BlockedRun, match="blocked_cross_family_candidates"):
        mod._load_cross_family_rows(root)
    cross["verifier_is_oracle"] = False
    cross["task_rows"] = []
    _write_json(root / mod.CROSS_FAMILY_REL, cross)
    with pytest.raises(mod.BlockedRun, match="blocked_cross_family_candidates"):
        mod._load_cross_family_rows(root)

    root = _make_repo(tmp_path / "energy")
    build = json.loads((root / mod.SET_ENCODER_BUILD_REL).read_text(encoding="utf-8"))
    build["aggregator_trained"] = False
    _write_json(root / mod.SET_ENCODER_BUILD_REL, build)
    with pytest.raises(mod.BlockedRun, match="blocked_energy_verifier_load"):
        mod._load_energy_verifier(root)
    build["aggregator_trained"] = True
    _write_json(root / mod.SET_ENCODER_BUILD_REL, build)
    model_path = Path(build["learned_verifier_path"])
    model = json.loads(model_path.read_text(encoding="utf-8"))
    model["verifier_is_oracle"] = True
    _write_json(model_path, model)
    with pytest.raises(mod.BlockedRun, match="blocked_energy_verifier_load"):
        mod._load_energy_verifier(root)

    root = _make_repo(tmp_path / "selection")
    cases, _, _, _, _ = mod.load_selection_cases(root, min_tasks=1, max_tasks=1)
    assert len(cases) == 1
    many = [
        mod.ArcCandidate("t", f"c{i}", i, False, [[i]], {"vote_weight": float(20 - i)}, float(20 - i))
        for i in range(mod.MAX_FINALISTS_PER_TASK + 3)
    ]
    task_row = {
        "set_encoder_candidate_id": "c0",
        "vote_candidate_id": "c1",
        "matched_control_candidate_id": "c2",
        "online_adapt_candidate_id": "c3",
    }
    assert len(mod._build_finalists(task_row, many)) == mod.MAX_FINALISTS_PER_TASK

    cross = json.loads((root / mod.CROSS_FAMILY_REL).read_text(encoding="utf-8"))
    cross["task_rows"].insert(0, {"task_id": ""})
    cross["task_rows"].insert(1, {"task_id": "gap3_stage2:T0", "set_encoder_candidate_id": "missing"})
    _write_json(root / mod.CROSS_FAMILY_REL, cross)
    assert len(mod.load_selection_cases(root, min_tasks=3)[0]) == 3

    class BadIndexJudge:
        def __init__(self) -> None:
            self.records: list[dict[str, Any]] = []

        def judge(self, _problem: str, _candidates: list[str]) -> int:
            self.records.append(
                {
                    "chosen_index": 999,
                    "latency_s": 0.1,
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                    "raw_output": "999",
                }
            )
            return 999

    selections, _cost = mod.run_llm_judge(cases, BadIndexJudge())
    assert selections[0]["llm_chosen_index"] == 0
