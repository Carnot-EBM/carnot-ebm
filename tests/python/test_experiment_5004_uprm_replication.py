"""Tests for Exp 5004 uPRM replication.

Spec refs: REQ-VERIFY-5004, SCENARIO-VERIFY-5004.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5004_uprm_replication as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _audit_clean(_path: Path) -> dict[str, Any]:
    return {"flag_count": 0, "flags": []}


def _row(row_id: str, gold: str, answers: list[str], good_index: int) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for index, answer in enumerate(answers):
        plus = 0.9 if index == good_index else 0.2
        minus = 1.0 - plus
        candidates.append(
            {
                "candidate_id": f"{row_id}/cached-{index}",
                "answer": answer,
                "reasoning": f"Step 1: choose {answer}\nANSWER: {answer}",
                "steps": [f"Step 1: choose {answer}", f"ANSWER: {answer}"],
                "token_logprobs": [math.log(0.8), math.log(0.7)],
                "top_logprobs": [{"x": math.log(0.8)}],
                "uprm_marker_logprobs": [
                    {"+": math.log(plus), "-": math.log(minus)},
                    {" +": math.log(plus), " -": math.log(minus)},
                ],
            }
        )
    return {
        "row_id": row_id,
        "corpus": "MuSR/murder_mysteries",
        "question": f"question {row_id}",
        "context": "context",
        "choices": ["A", "B"],
        "gold": gold,
        "candidates": candidates,
    }


def _complete_rows() -> list[dict[str, Any]]:
    return [
        _row("r0", "A", ["B", "A"], good_index=1),
        _row("r1", "B", ["A", "B"], good_index=1),
    ]


def test_req_verify_5004_spec_declares_uprm_contract() -> None:
    """REQ-VERIFY-5004: OpenSpec anchors uPRM fields and blockers."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-5004",
        "SCENARIO-VERIFY-5004",
        "experiment_5004_uprm_replication.py",
        "results/experiment_5004_uprm_replication.json",
        "arXiv:2605.10158",
        "blocked_<resource>",
        "success_uprm_beats_sc_<corpus>_",
        "complete_uprm_no_win_<corpus>_",
        "S(j)=1[j<=T] log p^-_j + sum_{t<j} log p^+_t",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_verify_5004_uprm_first_error_score_uses_marker_probabilities() -> None:
    """REQ-VERIFY-5004: uPRM score follows the paper's first-error formula."""

    marker_rows = [
        {"+": math.log(0.8), "-": math.log(0.2)},
        {" +": math.log(0.7), " -": math.log(0.3)},
    ]

    normalized = mod.renormalized_marker_logprobs(marker_rows[0])
    first_step_error = mod.uprm_first_error_log_score(marker_rows, first_error_position=1)
    second_step_error = mod.uprm_first_error_log_score(marker_rows, first_error_position=2)
    no_error = mod.uprm_first_error_log_score(marker_rows, first_error_position=3)
    process_score = mod.uprm_candidate_process_score(marker_rows)

    assert math.exp(normalized.log_p_plus) == pytest.approx(0.8)
    assert math.exp(normalized.log_p_minus) == pytest.approx(0.2)
    assert first_step_error == pytest.approx(math.log(0.2))
    assert second_step_error == pytest.approx(math.log(0.8) + math.log(0.3))
    assert no_error == pytest.approx(math.log(0.8) + math.log(0.7))
    assert process_score == pytest.approx((math.log(0.8) + math.log(0.7)) / 2)


def test_req_verify_5004_parses_llama_server_logprob_payload() -> None:
    """REQ-VERIFY-5004: llama-server completion_probabilities become telemetry."""

    parsed = mod.parse_llama_completion_payload(
        {
            "content": "Reasoning\nANSWER: A",
            "completion_probabilities": [
                {
                    "token": "Reasoning",
                    "logprob": -0.1,
                    "top_logprobs": [{"token": "Reasoning", "logprob": -0.1}],
                },
                {
                    "token": " A",
                    "logprob": "-0.2",
                    "top_logprobs": [
                        {"token": " A", "logprob": -0.2},
                        {"token": " B", "logprob": -1.7},
                    ],
                },
            ],
        }
    )

    assert parsed["text"] == "Reasoning\nANSWER: A"
    assert parsed["token_logprobs"] == [-0.1, -0.2]
    assert parsed["top_logprobs"] == [{"Reasoning": -0.1}, {" A": -0.2, " B": -1.7}]
    assert mod.parse_llama_completion_payload({"content": "text"})["token_logprobs"] == []
    choices_payload = {
        "choices": [
            {
                "text": "choice text",
                "logprobs": {
                    "token_logprobs": [True, "bad", "-0.4", float("nan")],
                    "top_logprobs": [{"+": "-0.3", "-": -1.3, "bad": False}],
                },
            }
        ]
    }
    choices = mod.parse_llama_completion_payload(choices_payload)
    skipped = mod.parse_llama_completion_payload(
        {"content": "x", "completion_probabilities": ["bad", {"logprob": "bad"}]}
    )

    assert choices["text"] == "choice text"
    assert choices["token_logprobs"] == [-0.4]
    assert choices["top_logprobs"] == [{"+": -0.3, "-": -1.3}]
    assert skipped["token_logprobs"] == []


def test_req_verify_5004_helper_edge_cases_are_deterministic(tmp_path: Path) -> None:
    """REQ-VERIFY-5004: cache, parsing, and prompt helpers have bounded edges."""

    jsonl = tmp_path / "rows.jsonl"
    assert mod._read_jsonl(jsonl) == []
    mod._write_jsonl(jsonl, [{"a": 1}, ["not", "a", "dict"]])
    jsonl.write_text(jsonl.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert mod._read_jsonl(jsonl) == [{"a": 1}]
    assert mod._number(True) is None
    assert mod._number("bad") is None
    assert mod._number(float("nan")) is None
    assert mod._top_logprob_row({"+": "-0.1", "-": -2.0, "bad": False}) == {
        "+": -0.1,
        "-": -2.0,
    }
    assert mod._match_choice("ANSWER: B", ["A", "B"]) == "B"
    assert mod.split_reasoning_steps("") == []
    assert mod.split_reasoning_steps("First. Second.") == ["First.", "Second."]
    prompt = mod._marker_prompt(
        {"context": "ctx", "question": "q"},
        {"steps": ["s1", "s2"]},
        "s2",
    )
    assert "CURRENT STEP:\ns2" in prompt
    with pytest.raises(mod.UprmScoringError, match="marker tokens"):
        mod.renormalized_marker_logprobs({"+": math.log(1.0)})
    with pytest.raises(mod.UprmScoringError, match="1..T"):
        mod.uprm_first_error_log_score(
            [{"+": math.log(0.5), "-": math.log(0.5)}], first_error_position=0
        )


def test_scenario_verify_5004_blocked_artifact_names_missing_resource(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5004: missing preconditions produce a blocked artifact."""

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.run(
        root=tmp_path,
        artifact_path=artifact_path,
        gguf_resolver=lambda *_args, **_kwargs: None,
        server_probe=lambda *_args, **_kwargs: mod.PreconditionCheck(
            "llama_server_logprobs", False, "server unavailable"
        ),
        corpus_loader=lambda _limit: [],
        audit_runner=_audit_clean,
        summary_runner=lambda _path: 0,
        write=True,
    )

    assert artifact["honest_verdict"] == "blocked_gemma_gguf_cache"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["headroom_present"] is False
    assert artifact["uprm_selection_accuracy"] is None
    assert artifact["preconditions_checked"][0]["available"] is False
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_verify_5004_complete_run_scores_candidates_oracle_distinct(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5004: complete run evaluates through guarded candidates."""

    gguf = tmp_path / "gemma.gguf"
    gguf.write_text("fixture", encoding="utf-8")
    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH

    artifact = mod.run(
        root=tmp_path,
        artifact_path=artifact_path,
        gguf_resolver=lambda *_args, **_kwargs: str(gguf),
        server_probe=lambda *_args, **_kwargs: mod.PreconditionCheck(
            "llama_server_logprobs", True, "server returns completion_probabilities"
        ),
        corpus_loader=lambda limit: _complete_rows()[:limit],
        candidate_rows_builder=lambda **_kwargs: _complete_rows(),
        audit_runner=_audit_clean,
        summary_runner=lambda _path: 0,
        min_questions=2,
        limit=2,
        bootstrap_samples=32,
        write=True,
    )

    assert artifact["honest_verdict"].startswith("complete_uprm_no_win_musr_")
    assert artifact["uprm_selection_accuracy"] == 1.0
    assert artifact["tuned_sc_accuracy"] == 0.0
    assert artifact["delta_vs_tuned_sc"] == 1.0
    assert artifact["oracle_at_k"] == 1.0
    assert artifact["headroom_present"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["oracle_distinctness_enforced"] is True
    assert artifact["adversarial_verify_clean"] is True
    assert artifact["summarize_artifact_exit_code"] == 0
    assert "arXiv:2605.10158" in artifact["uprm_score_methodology_note"]
    assert mod.artifact_schema_errors(artifact) == []


def test_req_verify_5004_oracle_leakage_blocks_complete_claim(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5004: scorers remain inside the guarded candidate view."""

    rows = _complete_rows()
    rows[0]["candidates"][0]["uprm_marker_logprobs"] = []

    with pytest.raises(mod.UprmScoringError, match="uprm_marker_logprobs"):
        mod.prepare_rows_with_uprm_scores(rows)
    missing_rows = _complete_rows()
    missing_rows[0]["candidates"][0].pop("uprm_marker_logprobs")
    with pytest.raises(mod.UprmScoringError, match="uprm_marker_logprobs"):
        mod.prepare_rows_with_uprm_scores(missing_rows)

    leaky_rows = _complete_rows()
    leaky_rows[0]["candidates"][0]["gold"] = "A"
    prepared = mod.prepare_rows_with_uprm_scores(leaky_rows)
    evaluation = mod.evaluate_uprm_rows(prepared, bootstrap_samples=16)

    assert evaluation["verifier"]["accuracy"] == 1.0
    assert evaluation["verifier"]["predictions"] == ["A", "B"]


def test_req_verify_5004_complete_verdict_and_schema_branches(tmp_path: Path) -> None:
    """REQ-VERIFY-5004: artifact verdict gates and schema checks are explicit."""

    base_eval = {
        "verifier": {"accuracy": 0.8, "predictions": []},
        "tuned_self_consistency": {"accuracy": 0.7, "config": {"k": 1}, "predictions": []},
        "verifier_minus_tuned_sc_delta": 0.1,
        "verifier_minus_tuned_sc_ci95": [0.01, 0.19],
        "mcnemar_p": 0.01,
        "headroom_present": True,
        "n_rows": 200,
        "oracle_at_k": 0.9,
    }

    success = mod.build_complete_artifact(
        evaluation=base_eval,
        preconditions_checked=[],
        candidate_cache_path=tmp_path / "cache.jsonl",
        gguf_path=tmp_path / "gemma.gguf",
        duration_s=61.0,
    )
    null = mod.build_complete_artifact(
        evaluation={**base_eval, "verifier_minus_tuned_sc_ci95": [-0.01, 0.2]},
        preconditions_checked=[],
        candidate_cache_path=tmp_path / "cache.jsonl",
        gguf_path=tmp_path / "gemma.gguf",
        duration_s=61.0,
    )
    blocked = mod.build_blocked_artifact(
        missing_resource="target_corpus",
        preconditions_checked=[],
        duration_s=0.1,
        error="missing corpus",
    )
    skeleton = mod.build_skeleton_artifact(
        preconditions_checked=[],
        gguf_path=tmp_path / "gemma.gguf",
        duration_s=0.2,
    )

    assert success["honest_verdict"].startswith("success_uprm_beats_sc_musr_")
    assert null["honest_verdict"].endswith("ci_incl_0")
    assert blocked["blocked_error"] == "missing corpus"
    assert skeleton["deliverable_stage"] == "pregeneration_skeleton"
    assert mod._slug_corpus("GPQA Diamond") == "gpqa_diamond"
    assert mod._compact_adversarial_flags({"reports": [{"flags": [{"kind": "WARN"}, "bad"]}]}) == [
        {"kind": "WARN"}
    ]
    assert mod._audit_is_clean({"flagged_count": 0}) is True
    assert mod._audit_is_clean({"flags": [{"kind": "WARN"}]}) is False
    assert mod.artifact_schema_errors(blocked) == []
    assert "verifier_is_oracle" in mod.artifact_schema_errors(
        {**blocked, "verifier_is_oracle": True}
    )
    assert "paired_ci95" in mod.artifact_schema_errors({**blocked, "paired_ci95": [0.0]})
    assert "spec_refs" in mod.artifact_schema_errors({**blocked, "spec_refs": ["REQ-VERIFY-5004"]})
    assert "uprm_selection_accuracy" in mod.artifact_schema_errors(
        {**blocked, "uprm_selection_accuracy": 2.0}
    )
    assert "honest_verdict" in mod.artifact_schema_errors({**blocked, "honest_verdict": "maybe"})
    missing = dict(blocked)
    missing.pop("duration_s")
    assert "duration_s" in mod.artifact_schema_errors(missing)
    assert "headroom_present" in mod.artifact_schema_errors({**blocked, "headroom_present": "no"})
    assert "delta_vs_tuned_sc" in mod.artifact_schema_errors({**blocked, "delta_vs_tuned_sc": "0"})
    assert "mcnemar_p" in mod.artifact_schema_errors({**blocked, "mcnemar_p": 2.0})
    assert "preconditions_checked" in mod.artifact_schema_errors(
        {**blocked, "preconditions_checked": {}}
    )
    assert "model_specs" in mod.artifact_schema_errors({**blocked, "model_specs": []})
    assert "field_principles" in mod.artifact_schema_errors({**blocked, "field_principles": {}})
    assert "uprm_score_methodology_note" in mod.artifact_schema_errors(
        {**blocked, "uprm_score_methodology_note": ""}
    )


def test_req_verify_5004_precondition_and_run_error_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-5004: corpus and scoring errors block without metric fabrication."""

    gguf = tmp_path / "gemma.gguf"
    gguf.write_text("fixture", encoding="utf-8")
    checks, resolved, rows = mod.check_preconditions(
        root=tmp_path,
        gguf_resolver=lambda *_args, **_kwargs: str(gguf),
        server_probe=lambda *_args, **_kwargs: mod.PreconditionCheck(
            "llama_server_logprobs", True, "ok"
        ),
        corpus_loader=lambda _limit: (_ for _ in ()).throw(RuntimeError("no corpus")),
        candidate_cache_path=tmp_path / "missing.jsonl",
        require_candidate_cache_or_fresh_generation=False,
        min_questions=2,
        server_port=8919,
    )
    assert resolved == gguf
    assert rows == []
    assert checks[-1].resource == "target_corpus"
    assert checks[-1].available is False
    assert mod.first_missing_resource(checks) == "target_corpus"
    cache_checks, _resolved_again, _rows_again = mod.check_preconditions(
        root=tmp_path,
        gguf_resolver=lambda *_args, **_kwargs: str(gguf),
        server_probe=lambda *_args, **_kwargs: mod.PreconditionCheck(
            "llama_server_logprobs", True, "ok"
        ),
        corpus_loader=lambda _limit: _complete_rows(),
        candidate_cache_path=tmp_path / "missing.jsonl",
        require_candidate_cache_or_fresh_generation=True,
        min_questions=2,
        server_port=8919,
    )
    assert cache_checks[-1].resource == "uprm_logprob_candidate_cache"
    assert cache_checks[-1].available is False

    common_kwargs = {
        "root": tmp_path,
        "artifact_path": tmp_path / mod.RESULT_RELATIVE_PATH,
        "gguf_resolver": lambda *_args, **_kwargs: str(gguf),
        "server_probe": lambda *_args, **_kwargs: mod.PreconditionCheck(
            "llama_server_logprobs", True, "ok"
        ),
        "corpus_loader": lambda limit: _complete_rows()[:limit],
        "audit_runner": _audit_clean,
        "summary_runner": lambda _path: 0,
        "min_questions": 2,
        "limit": 2,
        "write": True,
    }

    monkeypatch.setattr(mod, "_oracle_distinctness_enforced", lambda _rows: False)
    oracle_blocked = mod.run(
        **common_kwargs,
        candidate_rows_builder=lambda **_kwargs: _complete_rows(),
    )
    assert oracle_blocked["honest_verdict"] == "blocked_oracle_distinctness_violation"
    assert "shared harness" in oracle_blocked["blocked_error"]

    monkeypatch.setattr(mod, "_oracle_distinctness_enforced", lambda _rows: True)
    generic_blocked = mod.run(
        **common_kwargs,
        candidate_rows_builder=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("bad gen")),
    )
    assert generic_blocked["honest_verdict"] == "blocked_generation_or_scoring_error"
    assert "bad gen" in generic_blocked["blocked_error"]
