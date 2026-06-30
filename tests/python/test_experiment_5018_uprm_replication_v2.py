"""Tests for Exp 5018 uPRM replication v2.

Spec refs: REQ-VERIFY-5018, SCENARIO-VERIFY-5018.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5018_uprm_replication_v2 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _audit_clean(_path: Path) -> dict[str, Any]:
    return {"flag_count": 0, "flags": []}


def _candidate(row_id: str, index: int, answer: str, plus_probability: float) -> dict[str, Any]:
    minus_probability = 1.0 - plus_probability
    return {
        "candidate_id": f"{row_id}/cached-{index}",
        "answer": answer,
        "reasoning": f"Step 1: choose {answer}\nANSWER: {answer}",
        "cache_index": index,
        "temperature": 0.7,
        "generation_model": "gemma-4-12B-it-GGUF",
        "model_id": "forbidden-but-present",
        "token_logprobs": [math.log(0.8), math.log(0.7)],
        "uprm_marker_logprobs": [
            {"+": math.log(plus_probability), "-": math.log(minus_probability)},
            {" +": math.log(plus_probability), " -": math.log(minus_probability)},
        ],
    }


def _cache_row(row_id: str, gold: str, answers: list[str], good_index: int) -> dict[str, Any]:
    candidates = [
        _candidate(row_id, index, answer, 0.9 if index == good_index else 0.2)
        for index, answer in enumerate(answers)
    ]
    return {
        "schema": mod.CACHE_ROW_SCHEMA,
        "row_id": row_id,
        "row_index": int(row_id.rsplit(":", 1)[-1]),
        "corpus": mod.CORPUS,
        "question": f"question {row_id}",
        "context": "fixture",
        "choices": ["A", "B"],
        "gold": gold,
        "candidates_per_question": len(candidates),
        "has_per_token_logprobs": True,
        "candidates": candidates,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_req_verify_5018_spec_declares_uprm_v2_contract() -> None:
    """REQ-VERIFY-5018: OpenSpec anchors the v2 uPRM cache-selector contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-5018",
        "SCENARIO-VERIFY-5018",
        "experiment_5018_uprm_replication_v2.py",
        "results/experiment_5018_uprm_replication_v2.json",
        "arXiv:2605.10158",
        "arXiv:2605.24005",
        "S(j)=1[j<=T] log p^-_j + sum_{t<j} log p^+_t",
        "success_uprm_beats_sc_musr_",
        "complete_uprm_no_win_musr_",
        "blocked_<resource>",
        "genuine_tuned_sc_accuracy",
        "scoring_path",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_verify_5018_cache_loader_scores_complete_rows(tmp_path: Path) -> None:
    """REQ-VERIFY-5018: complete B2 cache rows become uPRM-scored rows."""

    path = tmp_path / mod.SHARED_CACHE_RELATIVE_PATH
    complete = _cache_row("musr:0", "A", ["B", "B", "B", "A", "A"], good_index=3)
    incomplete = _cache_row("musr:1", "B", ["A", "A", "A", "B", "B"], good_index=3)
    incomplete["candidates"][0]["uprm_marker_logprobs"] = [{"+": -0.1}]
    wrong_schema = {**complete, "schema": "wrong", "row_id": "musr:2"}
    _write_jsonl(path, [wrong_schema, complete, ["bad"], incomplete])  # type: ignore[list-item]
    path.write_text("\n" + path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    rows = mod.load_uprm_cache_rows(path, min_questions=1, k_candidates=5, limit=1)
    prepared = mod.prepare_rows_with_uprm_scores(rows)
    first_scores = [candidate["uprm_process_score"] for candidate in prepared[0]["candidates"]]
    check = mod.cache_precondition(path, min_questions=1, k_candidates=5)

    assert len(rows) == 1
    assert rows[0]["row_id"] == "musr:0"
    assert rows[0]["candidates"][0]["source"] == "exp5016_shared_logprob_candidate_cache"
    assert first_scores[3] > first_scores[0]
    assert check.available is True
    assert check.resource == "b2_logprob_cache"
    assert mod._finite_number(True) is False
    assert mod._finite_number("not-a-number") is False
    assert mod._has_marker_pair("not-a-marker-row") is False
    with pytest.raises(RuntimeError, match="only 1 uPRM-ready"):
        mod.load_uprm_cache_rows(path, min_questions=2, k_candidates=5)
    assert mod.cache_precondition(path, min_questions=2, k_candidates=5).available is False


def test_scenario_verify_5018_complete_run_uses_genuine_sc_and_guarded_scoring(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5018: uPRM cache scoring evaluates through the harness."""

    cache_path = tmp_path / mod.SHARED_CACHE_RELATIVE_PATH
    _write_jsonl(
        cache_path,
        [
            _cache_row("musr:0", "A", ["B", "B", "B", "A", "A"], good_index=3),
            _cache_row("musr:1", "B", ["A", "A", "A", "B", "B"], good_index=3),
        ],
    )
    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH

    artifact = mod.run(
        root=tmp_path,
        artifact_path=artifact_path,
        min_questions=2,
        k_candidates=5,
        bootstrap_samples=32,
        audit_runner=_audit_clean,
        summary_runner=lambda _path: 0,
        now=lambda: 10.0,
        write=True,
    )

    assert artifact["honest_verdict"].startswith("complete_uprm_no_win_musr_")
    assert artifact["verifier_is_oracle"] is False
    assert artifact["oracle_distinctness_enforced"] is True
    assert artifact["headroom_present"] is True
    assert artifact["uprm_selection_accuracy"] == 1.0
    assert artifact["genuine_tuned_sc_accuracy"] == 0.0
    assert artifact["delta_vs_tuned_sc"] == 1.0
    assert artifact["oracle_at_k"] == 1.0
    assert artifact["scoring_path"] == "uprm_logprob"
    assert artifact["inference_substrate"] == "verifier_ensemble_against_cached_candidates"
    assert artifact["adversarial_verify_clean"] is True
    assert artifact["summarize_artifact_exit_code"] == 0
    assert "arXiv:2605.10158" in artifact["uprm_score_methodology_note"]
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_verify_5018_missing_cache_writes_blocked_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5018: missing B2 cache and fallback blocks honestly."""

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.run(
        root=tmp_path,
        artifact_path=artifact_path,
        min_questions=2,
        k_candidates=5,
        audit_runner=_audit_clean,
        summary_runner=lambda _path: 0,
        write=True,
    )

    assert artifact["honest_verdict"] == "blocked_b2_logprob_cache"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["headroom_present"] is False
    assert artifact["uprm_selection_accuracy"] is None
    assert artifact["genuine_tuned_sc_accuracy"] is None
    assert artifact["scoring_path"] == "blocked"
    assert artifact["preconditions_checked"][0]["available"] is False
    assert artifact["preconditions_checked"][1]["resource"] == "lc_erd_fallback_inputs"
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert mod.artifact_schema_errors(artifact) == []


def test_req_verify_5018_schema_and_verdict_branches(tmp_path: Path) -> None:
    """REQ-VERIFY-5018: artifact schema and verdict gates are explicit."""

    skeleton = mod.build_skeleton_artifact(
        preconditions_checked=[],
        cache_path=tmp_path / "cache.jsonl",
        duration_s=0.1,
    )
    base_eval = {
        "verifier": {"accuracy": 0.8, "predictions": []},
        "tuned_self_consistency": {"accuracy": 0.7, "config": {"k": 5}, "predictions": []},
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
        cache_path=tmp_path / "cache.jsonl",
        duration_s=2.0,
    )
    null = mod.build_complete_artifact(
        evaluation={**base_eval, "verifier_minus_tuned_sc_ci95": [-0.01, 0.2]},
        preconditions_checked=[],
        cache_path=tmp_path / "cache.jsonl",
        duration_s=2.0,
    )
    gated = mod.build_complete_artifact(
        evaluation={**base_eval, "mcnemar_p": 0.5},
        preconditions_checked=[],
        cache_path=tmp_path / "cache.jsonl",
        duration_s=2.0,
    )
    blocked = mod.build_blocked_artifact(
        missing_resource="b2_logprob_cache",
        preconditions_checked=[],
        cache_path=tmp_path / "missing.jsonl",
        duration_s=0.1,
        error="missing cache",
    )

    assert skeleton["deliverable_stage"] == "schema_skeleton"
    assert success["honest_verdict"].startswith("success_uprm_beats_sc_musr_")
    assert null["honest_verdict"].endswith("ci_incl_0")
    assert gated["honest_verdict"].endswith("mcnemar_or_headroom_gate")
    assert blocked["blocked_error"] == "missing cache"
    assert mod._ci_includes_zero([-0.1, 0.0]) is True
    assert mod._ci_includes_zero([0.1]) is False
    assert mod._slug_corpus("MuSR/murder_mysteries") == "musr"
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
    assert "spec_refs" in mod.artifact_schema_errors({**blocked, "spec_refs": []})
    assert "field_principles" in mod.artifact_schema_errors({**blocked, "field_principles": {}})
    assert "uprm_selection_accuracy" in mod.artifact_schema_errors(
        {**blocked, "uprm_selection_accuracy": 2.0}
    )
    assert "genuine_tuned_sc_accuracy" in mod.artifact_schema_errors(
        {**blocked, "genuine_tuned_sc_accuracy": 2.0}
    )
    assert "delta_vs_tuned_sc" in mod.artifact_schema_errors(
        {**blocked, "delta_vs_tuned_sc": "bad"}
    )
    assert "mcnemar_p" in mod.artifact_schema_errors({**blocked, "mcnemar_p": 2.0})
    assert "preconditions_checked" in mod.artifact_schema_errors(
        {**blocked, "preconditions_checked": {}}
    )
    assert "model_specs" in mod.artifact_schema_errors({**blocked, "model_specs": []})
    assert "honest_verdict" in mod.artifact_schema_errors({**blocked, "honest_verdict": "bad"})
    assert "scoring_path" in mod.artifact_schema_errors({**blocked, "scoring_path": "bad"})
    assert "headroom_present" in mod.artifact_schema_errors(
        {**blocked, "headroom_present": "no"}
    )
    assert "uprm_score_methodology_note" in mod.artifact_schema_errors(
        {**blocked, "uprm_score_methodology_note": ""}
    )
    missing = dict(blocked)
    missing.pop("duration_s")
    assert "duration_s" in mod.artifact_schema_errors(missing)


def test_scenario_verify_5018_run_error_paths_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-5018: oracle leakage or scoring errors block, not null."""

    cache_path = tmp_path / mod.SHARED_CACHE_RELATIVE_PATH
    _write_jsonl(
        cache_path,
        [_cache_row("musr:0", "A", ["B", "B", "B", "A", "A"], good_index=3)],
    )

    monkeypatch.setattr(mod, "_oracle_distinctness_enforced", lambda _rows: False)
    oracle_blocked = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        min_questions=1,
        k_candidates=5,
        audit_runner=_audit_clean,
        summary_runner=lambda _path: 0,
        write=True,
    )

    assert oracle_blocked["honest_verdict"] == "blocked_oracle_distinctness_violation"
    assert "shared harness" in oracle_blocked["blocked_error"]
    assert mod.artifact_schema_errors(oracle_blocked) == []

    monkeypatch.setattr(mod, "_oracle_distinctness_enforced", lambda _rows: True)

    def bad_evaluate(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("bad score")

    monkeypatch.setattr(mod, "evaluate_uprm_rows", bad_evaluate)
    scoring_blocked = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        min_questions=1,
        k_candidates=5,
        audit_runner=_audit_clean,
        summary_runner=lambda _path: 0,
        write=True,
    )

    assert scoring_blocked["honest_verdict"] == "blocked_uprm_scoring_error"
    assert "bad score" in scoring_blocked["blocked_error"]
    assert mod.artifact_schema_errors(scoring_blocked) == []
