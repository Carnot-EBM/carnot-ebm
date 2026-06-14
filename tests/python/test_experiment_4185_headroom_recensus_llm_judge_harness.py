"""Tests for Exp 4185 headroom re-census and LLM-as-judge harness.

Spec refs: REQ-VERIFY-4185, SCENARIO-VERIFY-4185.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import headroom_recensus_llm_judge_harness_4185 as mod


QWEN_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _make_code_pool(root: Path) -> Path:
    path = root / "results" / "experiment_1999_code_verification_humaneval.json"
    _write_json(
        path,
        {
            "honest_verdict": "complete: fixture",
            "results": [
                {"task_id": "HumanEval/0", "baseline_passed": False, "repair_passed": True},
                {"task_id": "HumanEval/1", "baseline_passed": True, "repair_passed": True},
                {"task_id": "HumanEval/2", "baseline_passed": False, "repair_passed": False},
                {"task_id": "HumanEval/3", "baseline_passed": False, "repair_passed": True},
            ],
        },
    )
    return path


def _write_second_pool(root: Path) -> None:
    _write_json(
        root / "results" / "arc3_trm_verifier_rerank.json",
        {
            "n_tasks": 3,
            "oracle_ceiling": {"pass@2": 2 / 3},
            "rankers": {"TRM_VOTE": {"pass@1": 1 / 3}},
            "trm_vote_pass2": 1 / 3,
            "per_task": [
                {"task": "s0", "n_candidates": 2, "base_top1_correct": False},
                {"task": "s1", "n_candidates": 2, "base_top1_correct": True},
                {"task": "s2", "n_candidates": 2, "base_top1_correct": False},
            ],
        },
    )


class FakeJudge:
    def __init__(self, choices: list[int]):
        self.choices = list(choices)
        self.records: list[dict[str, Any]] = []

    def judge(self, problem: str, candidates: list[str]) -> int:
        assert problem
        assert len(candidates) == 2
        choice = self.choices.pop(0)
        self.records.append(
            {
                "chosen_index": choice,
                "latency_s": 0.25,
                "prompt_tokens": 20,
                "completion_tokens": 2,
                "total_tokens": 22,
                "raw_output": str(choice),
            }
        )
        return choice


class FakeLlama:
    def __init__(self, output: str):
        self.output = output

    def tokenize(self, text: bytes) -> list[int]:
        return list(range(len(text.split())))

    def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        assert "Candidate 0" in prompt
        assert kwargs["temperature"] == 0.0
        return {"choices": [{"text": self.output}]}


def test_req_4185_spec_declared_and_script_template_exports_cached_pair() -> None:
    """REQ-VERIFY-4185: OpenSpec declares the harness and SOTA cache precondition."""

    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4185",
        "SCENARIO-VERIFY-4185",
        "python/carnot/reporting/headroom_recensus_llm_judge_harness_4185.py",
        "results/experiment_4185_headroom_recensus_llm_judge_harness.py",
        "results/experiment_4185_headroom_recensus_llm_judge_harness.json",
        "blocked_model_not_cached_sota_gguf",
        "llm_judge_ready",
        "judge_cost_meter",
        "gated-fields-must-be-bare",
    ):
        assert marker in spec
    for principle in mod.FIELD_PRINCIPLES.values():
        assert principle in spec

    from scripts.experiment_template import cached_sota_pair

    assert callable(cached_sota_pair)


def test_scenario_4185_ready_artifact_has_bare_headroom_and_cost_meter(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-4185: cached GGUF judge smoke writes the comparator artifact."""

    code_pool = _make_code_pool(tmp_path)
    _write_second_pool(tmp_path)
    model_path = tmp_path / "models" / "judge.gguf"
    model_path.parent.mkdir()
    model_path.write_bytes(b"gguf fixture")
    spec = {"name": "Qwen3.6-35B-A3B", "hf_id": QWEN_ID, "gpu": 0, "model_path": str(model_path)}
    fake_judge = FakeJudge([1, 0, 1])

    artifact = mod.run(
        tmp_path,
        cached_pair_func=lambda: [spec],
        judge_factory=lambda _spec: fake_judge,
        random_seed=mod.RANDOM_SEED,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["max_selectable_headroom"] == pytest.approx(0.5)
    assert isinstance(artifact["max_selectable_headroom"], float)
    assert artifact["headroom_present_domain"] == "code"
    assert artifact["llm_judge_ready"] is True
    assert artifact["judge_pass1_smoke"]["n_candidate_sets"] == 3
    assert artifact["judge_pass1_smoke"]["pass1_accuracy"] == pytest.approx(2 / 3)
    assert artifact["judge_cost_meter"] == {
        "mean_judge_latency_s": 0.25,
        "mean_judge_tokens": 22.0,
        "mean_prompt_tokens": 20.0,
        "mean_completion_tokens": 2.0,
        "n_calls": 3,
    }
    assert artifact["model_specs"]["selected_judge"]["model_path"] == str(model_path)
    assert artifact["model_specs"]["selected_judge"]["hf_id"] == QWEN_ID
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(
        code_pool,
        {
            "hf_id": QWEN_ID,
            "model_path": str(model_path),
            "prompt_version": mod.PROMPT_VERSION,
            "random_seed": mod.RANDOM_SEED,
        },
    )
    assert artifact["per_domain_headroom"]["sudoku"]["selectable_headroom"] == pytest.approx(1 / 3)
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert artifact["spec_refs"] == ["REQ-VERIFY-4185", "SCENARIO-VERIFY-4185"]
    assert artifact["acceptance_gate"] is True

    written = json.loads((tmp_path / mod.OUTPUT_REL).read_text(encoding="utf-8"))
    assert written == artifact


def test_req_4185_blocked_when_sota_gguf_model_path_is_not_cached(tmp_path: Path) -> None:
    """REQ-VERIFY-4185: missing SOTA GGUF writes the accepted blocked artifact."""

    _make_code_pool(tmp_path)
    artifact = mod.run(
        tmp_path,
        cached_pair_func=lambda: [{"name": "Qwen", "hf_id": QWEN_ID, "model_path": None}],
        judge_factory=lambda _spec: pytest.fail("judge must not load without a model path"),
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_model_not_cached_sota_gguf"
    assert artifact["max_selectable_headroom"] == 0.0
    assert isinstance(artifact["max_selectable_headroom"], float)
    assert artifact["headroom_present_domain"] == ""
    assert artifact["llm_judge_ready"] is False
    assert artifact["judge_cost_meter"]["n_calls"] == 0
    assert artifact["acceptance_gate"] is True


def test_cost_metered_llm_judge_parses_choice_and_counts_tokens(tmp_path: Path) -> None:
    """REQ-VERIFY-4185: every judge call records latency and token counts."""

    model_path = tmp_path / "judge.gguf"
    model_path.write_bytes(b"gguf")
    judge = mod.CostMeteredLlmJudge(
        {"name": "Qwen", "hf_id": QWEN_ID, "model_path": str(model_path)},
        llama_factory=lambda **_kwargs: FakeLlama("I choose 1."),
        clock=iter([10.0, 10.5]).__next__,
    )

    assert judge.judge("Pick the better candidate.", ["baseline", "repair"]) == 1
    assert judge.records == [
        {
            "chosen_index": 1,
            "latency_s": 0.5,
            "prompt_tokens": 41,
            "completion_tokens": 3,
            "total_tokens": 44,
            "raw_output": "I choose 1.",
        }
    ]


def test_req_4185_validation_rejects_non_bare_gate_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-4185: A2-gated fields must stay bare scalars."""

    _make_code_pool(tmp_path)
    artifact = mod._blocked_artifact(
        tmp_path,
        "blocked_model_not_cached_sota_gguf",
        "fixture",
        0.1,
        random_seed=mod.RANDOM_SEED,
    )
    mod.validate_artifact(artifact)

    invalid_cases = [
        ({**artifact, "max_selectable_headroom": {"value": 0.0}}, "bare float"),
        ({**artifact, "llm_judge_ready": 0}, "bare bool"),
        ({**artifact, "random_seed": True}, "bare int"),
        ({**artifact, "field_principles": {}}, "field_principles"),
        ({**artifact, "spec_refs": []}, "spec_refs"),
        ({**artifact, "judge_cost_meter": []}, "judge_cost_meter"),
        ({**artifact, "model_specs": []}, "model_specs"),
        ({**artifact, "headroom_present_domain": {}}, "headroom_present_domain"),
        ({**artifact, "honest_verdict": "missing prefix"}, "terminal prefix"),
    ]
    for payload, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(payload)


def test_req_4185_defensive_branches_and_repeat_smoke(tmp_path: Path) -> None:
    """REQ-VERIFY-4185: blocked and defensive paths stay terminal and parseable."""

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSON object"):
        mod._read_json_object(bad_json)

    assert mod._select_judge_spec(None) is None
    assert mod._parse_choice("no usable choice", 2) == 0

    def keyword_only_provider(*, gpu_indices: tuple[int, int]) -> list[dict[str, Any]]:
        assert gpu_indices == (0, 1)
        return []

    assert mod._call_cached_pair(keyword_only_provider) == []

    rows_not_list = tmp_path / "rows_not_list.json"
    _write_json(rows_not_list, {"results": {}})
    assert mod._candidate_sets_from_code_pool(rows_not_list) == []

    filtered_rows = tmp_path / "filtered_rows.json"
    _write_json(
        filtered_rows,
        {
            "results": [
                "not a row",
                {"task_id": "bad", "baseline_passed": "yes", "repair_passed": False},
                {"baseline_passed": True, "repair_passed": False},
                {"task_id": "later", "baseline_passed": False, "repair_passed": True},
            ]
        },
    )
    assert mod._candidate_sets_from_code_pool(filtered_rows, limit=1)[0]["task_id"] == "task-0"

    missing = mod.run(
        tmp_path / "missing-code",
        cached_pair_func=lambda: pytest.fail("model cache should not be checked without code pool"),
    )
    assert missing["honest_verdict"] == "blocked_missing_code_pool"
    assert missing["acceptance_gate"] is False

    model_path = tmp_path / "models" / "judge.gguf"
    model_path.parent.mkdir()
    model_path.write_bytes(b"gguf fixture")
    spec = {"name": "Qwen3.6-35B-A3B", "hf_id": QWEN_ID, "gpu": 0, "model_path": str(model_path)}

    empty_root = tmp_path / "empty-candidates"
    _write_json(empty_root / "results" / "experiment_1999_code_verification_humaneval.json", {"results": []})
    insufficient = mod.run(
        empty_root,
        cached_pair_func=lambda: [spec],
        judge_factory=lambda _spec: pytest.fail("judge should not run without candidate sets"),
    )
    assert insufficient["honest_verdict"] == "blocked_insufficient_real_candidate_sets"
    assert insufficient["per_domain_headroom"]

    repeat_root = tmp_path / "repeat"
    _make_code_pool(repeat_root)
    repeat_judge = FakeJudge([1, 0, 1, 0, 1])
    repeated = mod.run(
        repeat_root,
        cached_pair_func=lambda: [spec],
        judge_factory=lambda _spec: repeat_judge,
        smoke_n=5,
    )
    assert repeated["llm_judge_ready"] is True
    assert repeated["judge_pass1_smoke"]["n_candidate_sets"] == 5
    assert repeated["judge_pass1_smoke"]["unique_task_count"] == 4
    assert any("#repeat" in row["task_id"] for row in repeated["judge_pass1_smoke"]["selections"])

    valid = mod._blocked_artifact(
        tmp_path,
        "blocked_model_not_cached_sota_gguf",
        "fixture",
        0.1,
        random_seed=mod.RANDOM_SEED,
    )
    for payload, message in (
        ({k: v for k, v in valid.items() if k != "honest_verdict"}, "missing required"),
        ({**valid, "reproducibility_checksum": "short"}, "checksum"),
        ({**valid, "llm_judge_ready": True}, "requires smoke judge calls"),
    ):
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(payload)
