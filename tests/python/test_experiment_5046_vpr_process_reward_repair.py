"""Tests for Exp 5046 dense process-reward MuSR repair.

Spec refs: REQ-VERIFY-5046, SCENARIO-VERIFY-5046.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5046_vpr_process_reward_repair as mod
from carnot.moat_benchmark_harness import GuardedCandidate, OracleDistinctnessError


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


class Clock:
    """Deterministic clock for elapsed-time fields."""

    def __init__(self, values: list[float]) -> None:
        self.values = values
        self.index = 0

    def __call__(self) -> float:
        value = self.values[min(self.index, len(self.values) - 1)]
        self.index += 1
        return value


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _cache_row(
    question_index: int,
    candidate_index: int,
    answer: str,
    *,
    gold: str,
    plus_probability: float,
    acceptance_probability: float,
    consequence_probability: float = 0.01,
) -> dict[str, Any]:
    return {
        "schema": mod.CACHE_ROW_SCHEMA,
        "corpus": mod.CORPUS,
        "question_id": f"MuSR/murder_mysteries:{question_index}",
        "question_index": question_index,
        "question": f"Who is responsible in case {question_index}?",
        "context": (
            "The trace must weigh motive, opportunity, alibi consistency, "
            "and the consequence of accusing the wrong suspect."
        ),
        "choices": ["A", "B"],
        "gold": gold,
        "candidate_id": f"MuSR/murder_mysteries:{question_index}/cached-{candidate_index}",
        "candidate_index": candidate_index,
        "answer": answer,
        "completion_text": " +",
        "source": "distributional_energy_verifier_musr_checkpoints",
        "rescored_not_regenerated": True,
        "scoring_model": "gemma-4-12B-it-GGUF",
        "model_id": "forbidden-but-present",
        "mean_logprob": math.log(max(acceptance_probability, 1e-6)),
        "token_logprobs": [math.log(max(acceptance_probability, 1e-6))],
        "top_logprobs": [
            {
                " consistent": math.log(max(acceptance_probability, 1e-6)),
                " supported": math.log(max(acceptance_probability - 0.1, 1e-6)),
                " contradiction": math.log(max(consequence_probability, 1e-6)),
                " unsupported": math.log(max(consequence_probability, 1e-6)),
            }
        ],
        "uprm_marker_logprobs": [
            {
                " +": math.log(max(plus_probability, 1e-6)),
                " -": math.log(max(1.0 - plus_probability, 1e-6)),
            }
        ],
    }


def _write_cache(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_preflight(root: Path) -> None:
    _write_json(
        root / mod.PREFLIGHT_RELATIVE_PATH,
        {
            "honest_verdict": "blocked_judge_server",
            "model_specs": {
                "flagship_moe": {
                    "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                    "preferred_quant": "Q4_K_M",
                    "resolved_path": "/models/qwen.gguf",
                }
            },
            "usable_sota_models": [
                {
                    "role": "flagship_moe",
                    "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                    "model_path": "/models/qwen.gguf",
                }
            ],
            "sota_models_ready": True,
            "sota_judge_ready": False,
        },
    )


def _write_good_cache(root: Path, *, n_questions: int = 2) -> None:
    rows: list[dict[str, Any]] = []
    for question_index in range(n_questions):
        rows.extend(
            [
                _cache_row(
                    question_index,
                    0,
                    "B",
                    gold="A",
                    plus_probability=0.99,
                    acceptance_probability=0.05,
                    consequence_probability=0.70,
                ),
                _cache_row(
                    question_index,
                    1,
                    "A",
                    gold="A",
                    plus_probability=0.55,
                    acceptance_probability=0.92,
                    consequence_probability=0.01,
                ),
            ]
        )
    _write_cache(root / mod.FIXED_B2_CACHE_RELATIVE_PATH, rows)


def test_req_verify_5046_spec_declares_dense_process_contract() -> None:
    """REQ-VERIFY-5046: OpenSpec anchors the process-reward repair contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-5046",
        "SCENARIO-VERIFY-5046",
        "experiment_5046_vpr_process_reward_repair.py",
        "results/experiment_5046_vpr_process_reward_repair.json",
        "step-consistency",
        "verifier-acceptance",
        "consequence-penalty",
        "uncertainty",
        "scalar_marker_only=false",
        "success_process_reward_beats_sc_musr_",
        "complete_process_reward_no_win_musr_",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_verify_5046_dense_features_beat_scalar_marker_leakage() -> None:
    """REQ-VERIFY-5046: dense features can overturn a better final marker."""

    rows = [
        {
            "row_id": "q0",
            "question": "Who is responsible?",
            "context": "Evidence favors A; B has a contradiction.",
            "choices": ["A", "B"],
            "gold": "A",
            "candidates": [
                _cache_row(
                    0,
                    0,
                    "B",
                    gold="A",
                    plus_probability=0.99,
                    acceptance_probability=0.05,
                    consequence_probability=0.70,
                ),
                _cache_row(
                    0,
                    1,
                    "A",
                    gold="A",
                    plus_probability=0.55,
                    acceptance_probability=0.92,
                    consequence_probability=0.01,
                ),
            ],
        }
    ]

    prepared = mod.prepare_rows_with_process_rewards(rows)
    bad, good = prepared[0]["candidates"]

    assert bad["scalar_marker_probability"] > good["scalar_marker_probability"]
    assert good["process_reward_features"]["verifier_acceptance"] > bad["process_reward_features"]["verifier_acceptance"]
    assert good["process_reward_features"]["consequence_penalty"] < bad["process_reward_features"]["consequence_penalty"]
    assert mod.dense_process_reward_energy(GuardedCandidate(good)) < mod.dense_process_reward_energy(GuardedCandidate(bad))
    assert prepared[0]["trace_source"] == "cache_derived_process_trace"
    assert prepared[0]["candidates"][0]["process_trace"]["scalar_marker_only"] is False


def test_scenario_verify_5046_oracle_distinctness_guard_blocks_leaks() -> None:
    """SCENARIO-VERIFY-5046: guarded scorers cannot read gold or model identity."""

    guarded = GuardedCandidate({"gold": "A", "model_id": "leak", "answer": "A"})

    with pytest.raises(OracleDistinctnessError, match="gold"):
        _ = guarded["gold"]
    with pytest.raises(OracleDistinctnessError, match="model_id"):
        _ = guarded.get("model_id")

    assert mod.oracle_distinctness_self_check() is True


def test_req_verify_5046_helper_edge_cases_and_malformed_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-5046: malformed cache/model inputs fail closed."""

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    assert mod._read_json(bad_json) is None
    assert mod._read_json(tmp_path / "missing.json") is None
    assert mod._read_jsonl(tmp_path / "missing.jsonl") == []
    jsonl = tmp_path / "mixed.jsonl"
    jsonl.write_text("\n[]\n" + json.dumps({"ok": True}) + "\n", encoding="utf-8")
    assert mod._read_jsonl(jsonl) == [{"ok": True}]

    assert mod._number(True) is None
    assert mod._number("nope") is None
    assert mod._probability_from_logprob(True) is None
    assert mod._probability_from_logprob(0.25) == pytest.approx(0.25)
    assert mod._probability_from_logprob(2.0) is None
    assert mod._distribution_uncertainty([{" only": 0.0}]) == 1.0
    assert mod._marker_probability({"uprm_marker_logprobs": ["bad", {" +": "bad"}]}) == 0.5
    assert mod._mean_token_probability({"token_logprobs": [math.log(0.5), "bad"]}) == 0.5
    assert mod._answer_context_hit("", "question", "context") == 0.0
    assert math.isinf(mod.dense_process_reward_energy({}))
    assert mod._verdict(None, None, None, False) == "blocked_process_reward_unavailable"
    assert mod._verdict(-0.1, [-0.2, 0.0], 1.0, True).startswith(
        "complete_process_reward_no_win_musr_minus_"
    )

    malformed = tmp_path / "malformed.jsonl"
    malformed_rows = [
        {"schema": "wrong", "question_id": "skip"},
        {"schema": mod.CACHE_ROW_SCHEMA},
        _cache_row(0, 0, "A", gold="A", plus_probability=0.5, acceptance_probability=0.5),
        _cache_row(1, 0, "A", gold="", plus_probability=0.5, acceptance_probability=0.5),
        _cache_row(1, 1, "B", gold="", plus_probability=0.5, acceptance_probability=0.5),
    ]
    _write_cache(malformed, malformed_rows)
    with pytest.raises(RuntimeError, match="only 0 MuSR"):
        mod.load_fixed_b2_cache_rows(malformed, min_questions=1, k_candidates=2)

    no_preflight = tmp_path / "no_preflight"
    selected, preflight = mod._select_process_model_from_preflight(no_preflight)
    assert selected is None and preflight == {}
    monkeypatch.setattr(mod, "resolve_cached_gguf", lambda *_args: "/models/fallback.gguf")
    selected, specs = mod._resolve_mandated_model(no_preflight)
    assert selected is not None
    assert selected["model_path"] == "/models/fallback.gguf"
    assert specs["process_trace_model"] == selected

    unusable = tmp_path / "unusable"
    _write_json(unusable / mod.PREFLIGHT_RELATIVE_PATH, {"usable_sota_models": []})
    assert mod._select_process_model_from_preflight(unusable)[0] is None


def test_req_verify_5046_schema_validator_flags_bad_artifacts() -> None:
    """REQ-VERIFY-5046: schema validation rejects skeleton and bad-typed fields."""

    missing_errors = mod.artifact_schema_errors({})
    assert "honest_verdict" in missing_errors
    assert "schema" in missing_errors

    bad = {field: None for field in mod.REQUIRED_ARTIFACT_FIELDS}
    bad.update(
        {
            "schema": "bad",
            "experiment": "bad",
            "spec_refs": [],
            "model_specs": None,
            "process_reward_available": True,
            "verifier_is_oracle": True,
            "headroom_present": "yes",
            "scalar_marker_only": True,
            "oracle_distinctness_enforced": False,
            "trace_count": 0,
            "honest_verdict": "maybe",
            "process_reward_accuracy": None,
            "genuine_tuned_sc_accuracy": 2.0,
            "mcnemar_p": -1.0,
            "paired_ci95": ["bad"],
            "delta_vs_tuned_sc": "bad",
        }
    )
    errors = mod.artifact_schema_errors(bad)
    for field in (
        "schema",
        "experiment",
        "spec_refs",
        "model_specs",
        "headroom_present",
        "verifier_is_oracle",
        "scalar_marker_only",
        "honest_verdict",
        "genuine_tuned_sc_accuracy",
        "mcnemar_p",
        "paired_ci95",
        "delta_vs_tuned_sc",
        "process_reward_accuracy",
        "trace_count",
        "oracle_distinctness_enforced",
    ):
        assert field in errors

    bad["trace_count"] = -1
    assert "trace_count" in mod.artifact_schema_errors(bad)


def test_scenario_verify_5046_blocked_paths_are_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-5046: blocked resources and oracle violations write artifacts."""

    original_prepare = mod.prepare_rows_with_process_rewards
    original_dense = mod.dense_process_reward_energy
    no_model = tmp_path / "no_model"
    monkeypatch.setattr(mod, "resolve_cached_gguf", lambda *_args: None)
    blocked_model = mod.run(
        root=no_model,
        artifact_path=no_model / "out.json",
        min_questions=1,
        now=Clock([0.0, 1.0]),
        write=True,
    )
    assert blocked_model["honest_verdict"] == "blocked_mandated_process_model_unavailable"
    assert "blocked_error" in blocked_model

    cache_missing = tmp_path / "cache_missing"
    _write_preflight(cache_missing)
    blocked_cache = mod.run(
        root=cache_missing,
        artifact_path=cache_missing / "out.json",
        min_questions=1,
        now=Clock([0.0, 1.0]),
        write=True,
    )
    assert blocked_cache["honest_verdict"] == "blocked_cached_musr_candidates"

    unavailable = tmp_path / "unavailable"
    _write_preflight(unavailable)
    _write_good_cache(unavailable, n_questions=1)
    monkeypatch.setattr(mod, "prepare_rows_with_process_rewards", lambda rows, **_kw: list(rows))
    blocked_unavailable = mod.run(
        root=unavailable,
        artifact_path=unavailable / "out.json",
        min_questions=1,
        k_candidates=2,
        now=Clock([0.0, 1.0]),
        write=True,
    )
    assert blocked_unavailable["honest_verdict"] == "blocked_process_reward_unavailable"

    oracle = tmp_path / "oracle"
    _write_preflight(oracle)
    _write_good_cache(oracle, n_questions=1)
    monkeypatch.setattr(mod, "prepare_rows_with_process_rewards", original_prepare)

    def _leaky(_candidate: Any) -> float:
        raise OracleDistinctnessError("forced leak")

    monkeypatch.setattr(mod, "dense_process_reward_energy", _leaky)
    blocked_oracle = mod.run(
        root=oracle,
        artifact_path=oracle / "out.json",
        min_questions=1,
        k_candidates=2,
        now=Clock([0.0, 1.0]),
        write=True,
    )
    assert blocked_oracle["honest_verdict"] == "blocked_oracle_distinctness_violation"

    eval_oracle = tmp_path / "eval_oracle"
    _write_preflight(eval_oracle)
    _write_good_cache(eval_oracle, n_questions=1)
    monkeypatch.setattr(mod, "dense_process_reward_energy", original_dense)

    def _raising_evaluate(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise OracleDistinctnessError("forced eval leak")

    monkeypatch.setattr(mod, "evaluate_verifier", _raising_evaluate)
    blocked_eval_oracle = mod.run(
        root=eval_oracle,
        artifact_path=eval_oracle / "out.json",
        min_questions=1,
        k_candidates=2,
        now=Clock([0.0, 1.0]),
        write=True,
    )
    assert blocked_eval_oracle["honest_verdict"] == "blocked_oracle_distinctness_violation"


def test_scenario_verify_5046_run_writes_required_schema(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5046: process reward evaluates against genuine tuned-SC."""

    root = tmp_path / "root"
    _write_preflight(root)
    cache_rows: list[dict[str, Any]] = []
    for question_index in range(6):
        cache_rows.extend(
            [
                _cache_row(
                    question_index,
                    0,
                    "B",
                    gold="A",
                    plus_probability=0.99,
                    acceptance_probability=0.05,
                    consequence_probability=0.70,
                ),
                _cache_row(
                    question_index,
                    1,
                    "B",
                    gold="A",
                    plus_probability=0.95,
                    acceptance_probability=0.10,
                    consequence_probability=0.60,
                ),
                _cache_row(
                    question_index,
                    2,
                    "A",
                    gold="A",
                    plus_probability=0.55,
                    acceptance_probability=0.92,
                    consequence_probability=0.01,
                ),
            ]
        )
    _write_cache(root / mod.FIXED_B2_CACHE_RELATIVE_PATH, cache_rows)

    artifact = mod.run(
        root=root,
        artifact_path=root / mod.RESULT_RELATIVE_PATH,
        min_questions=6,
        k_candidates=3,
        limit=6,
        bootstrap_samples=64,
        now=Clock([10.0, 12.0]),
        write=True,
    )

    assert artifact["honest_verdict"].startswith("success_process_reward_beats_sc_musr_")
    assert artifact["process_reward_available"] is True
    assert artifact["process_reward_accuracy"] == 1.0
    assert artifact["genuine_tuned_sc_accuracy"] == 0.0
    assert artifact["delta_vs_tuned_sc"] == 1.0
    assert artifact["trace_count"] == 18
    assert artifact["n_questions"] == 6
    assert artifact["verifier_is_oracle"] is False
    assert artifact["headroom_present"] is True
    assert artifact["scalar_marker_only"] is False
    assert artifact["model_specs"]["process_trace_model"]["hf_id"] == "unsloth/Qwen3.6-35B-A3B-GGUF"
    assert artifact["dense_feature_weights"]["uncertainty"] < 0.0
    assert json.loads((root / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact
    assert mod.artifact_schema_errors(artifact) == []
