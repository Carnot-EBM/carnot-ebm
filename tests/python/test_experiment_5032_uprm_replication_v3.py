"""Tests for Exp 5032 uPRM replication v3.

Spec refs: REQ-VERIFY-5032, SCENARIO-VERIFY-5032.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5032_uprm_replication_v3 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _audit_clean(_path: Path) -> dict[str, Any]:
    return {"flag_count": 0, "flags": []}


def _candidate_row(
    question_index: int,
    candidate_index: int,
    answer: str,
    gold: str,
    plus_probability: float,
) -> dict[str, Any]:
    minus_probability = 1.0 - plus_probability
    return {
        "schema": mod.CACHE_ROW_SCHEMA,
        "corpus": mod.CORPUS,
        "question_id": f"MuSR/murder_mysteries:{question_index}",
        "question_index": question_index,
        "question": f"question {question_index}",
        "context": "fixture context",
        "choices": ["A", "B"],
        "gold": gold,
        "candidate_id": f"MuSR/murder_mysteries:{question_index}/cached-{candidate_index}",
        "candidate_index": candidate_index,
        "answer": answer,
        "source": "distributional_energy_verifier_musr_checkpoints",
        "rescored_not_regenerated": True,
        "scoring_model": "gemma-4-12B-it-GGUF",
        "model_id": "forbidden-but-present",
        "token_logprobs": [math.log(0.8)],
        "uprm_marker_logprobs": [
            {"+": math.log(plus_probability), "-": math.log(minus_probability)},
            {" +": math.log(plus_probability), " -": math.log(minus_probability)},
        ],
    }


def _cache_rows(question_index: int, gold: str, answers: list[str], good_index: int) -> list[dict[str, Any]]:
    return [
        _candidate_row(
            question_index,
            candidate_index,
            answer,
            gold,
            0.9 if candidate_index == good_index else 0.2,
        )
        for candidate_index, answer in enumerate(answers)
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_checkpoint(root: Path, index: int, gold: str, answers: list[str]) -> None:
    path = root / mod.FROZEN_CANDIDATE_RELATIVE_DIR / f"q{index:04d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"q": index, "gold": gold, "answers": answers}, sort_keys=True),
        encoding="utf-8",
    )


def test_req_verify_5032_spec_declares_v3_contract() -> None:
    """REQ-VERIFY-5032: OpenSpec anchors fixed-cache and frozen-fallback scoring."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-5032",
        "SCENARIO-VERIFY-5032",
        "experiment_5032_uprm_replication_v3.py",
        "results/experiment_5032_uprm_replication_v3.json",
        "experiment_5029_shared_logprob_candidate_cache_v2_musr.jsonl",
        "arXiv:2605.10158",
        "S(j)=1[j<=T] log p^-_j + sum_{t<j} log p^+_t",
        "self_supervised_frozen",
        "success_uprm_beats_sc_musr_",
        "complete_uprm_no_win_musr_",
        "genuine_tuned_sc_accuracy",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_verify_5032_fixed_cache_loader_groups_candidate_rows(tmp_path: Path) -> None:
    """REQ-VERIFY-5032: fixed B2 row-per-candidate cache becomes scored rows."""

    path = tmp_path / mod.FIXED_B2_CACHE_RELATIVE_PATH
    complete = _cache_rows(0, "A", ["B", "B", "B", "A", "A"], good_index=3)
    incomplete = _cache_rows(1, "B", ["A", "A", "A", "B", "B"], good_index=3)
    incomplete[0]["uprm_marker_logprobs"] = [{"+": -0.1}]
    wrong_schema = {**complete[0], "schema": "wrong", "question_id": "bad"}
    no_gold = _cache_rows(2, "", ["A", "A", "A", "B", "B"], good_index=0)
    _write_jsonl(path, [wrong_schema, *complete, *incomplete, *no_gold])
    path.write_text("\n" + path.read_text(encoding="utf-8"), encoding="utf-8")

    rows = mod.load_fixed_b2_cache_rows(path, min_questions=1, k_candidates=5, limit=1)
    prepared = mod.prepare_rows_with_process_scores(rows, scoring_path="uprm_logprob")
    scores = [candidate["process_score"] for candidate in prepared[0]["candidates"]]
    check = mod.fixed_b2_cache_precondition(path, min_questions=1, k_candidates=5)

    assert len(rows) == 1
    assert rows[0]["row_id"] == "MuSR/murder_mysteries:0"
    assert rows[0]["candidates"][0]["source"] == "exp5029_fixed_b2_logprob_cache"
    assert scores[3] > scores[0]
    assert check.available is True
    assert check.resource == "fixed_b2_logprob_cache"
    assert mod._finite_number(True) is False
    assert mod._finite_number("not-a-number") is False
    assert mod._has_marker_pair({" +": -0.1, " -": -2.0}) is True
    assert mod._has_marker_pair("not-a-marker-row") is False
    with pytest.raises(RuntimeError, match="only 1 uPRM-ready"):
        mod.load_fixed_b2_cache_rows(path, min_questions=2, k_candidates=5)
    assert mod.fixed_b2_cache_precondition(path, min_questions=2, k_candidates=5).available is False


def test_scenario_verify_5032_primary_run_uses_uprm_logprob_and_guarded_scoring(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5032: fixed-cache uPRM scoring evaluates through the harness."""

    cache_path = tmp_path / mod.FIXED_B2_CACHE_RELATIVE_PATH
    _write_jsonl(
        cache_path,
        [
            *_cache_rows(0, "A", ["B", "B", "B", "A", "A"], good_index=3),
            *_cache_rows(1, "B", ["A", "A", "A", "B", "B"], good_index=3),
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
    assert artifact["no_model_id_shortcut_audit"] is True
    assert artifact["headroom_present"] is True
    assert artifact["uprm_selection_accuracy"] == 1.0
    assert artifact["genuine_tuned_sc_accuracy"] == 0.0
    assert artifact["delta_vs_tuned_sc"] == 1.0
    assert artifact["oracle_at_k"] == 1.0
    assert artifact["scoring_path"] == "uprm_logprob"
    assert artifact["degeneracy_guard"]["degeneracy_flag"] is False
    assert artifact["inference_substrate"] == "verifier_ensemble_against_cached_candidates"
    assert artifact["adversarial_verify_clean"] is True
    assert artifact["summarize_artifact_exit_code"] == 0
    assert "arXiv:2605.10158" in artifact["uprm_score_methodology_note"]
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_verify_5032_fallback_scores_frozen_candidates(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5032: missing fixed cache falls back to frozen candidate text."""

    _write_checkpoint(tmp_path, 98, "A", ["A"])
    (tmp_path / mod.FROZEN_CANDIDATE_RELATIVE_DIR / "q0097.json").write_text(
        json.dumps({"q": 97, "gold": "A", "answers": "A"}),
        encoding="utf-8",
    )
    (tmp_path / mod.FROZEN_CANDIDATE_RELATIVE_DIR / "q0096.json").write_text(
        json.dumps({"q": 96, "answers": ["A", "A", "A", "B", "B"]}),
        encoding="utf-8",
    )
    _write_checkpoint(tmp_path, 0, "A", ["B", "B", "B", "A", "A"])
    _write_checkpoint(tmp_path, 1, "B", ["A", "A", "A", "B", "B"])
    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        min_questions=2,
        k_candidates=5,
        bootstrap_samples=32,
        audit_runner=_audit_clean,
        summary_runner=lambda _path: 0,
        now=lambda: 20.0,
        write=True,
    )

    assert artifact["scoring_path"] == "self_supervised_frozen"
    assert artifact["honest_verdict"].startswith("complete_uprm_no_win_musr_")
    assert artifact["verifier_is_oracle"] is False
    assert artifact["no_model_id_shortcut_audit"] is True
    assert artifact["preconditions_checked"][0]["available"] is False
    assert artifact["preconditions_checked"][1]["available"] is True
    assert artifact["model_specs"]["fallback_score"] == "endogenous_answer_consensus_plus_step_overlap"
    assert mod.artifact_schema_errors(artifact) == []
    assert mod._read_checkpoint(tmp_path / mod.FROZEN_CANDIDATE_RELATIVE_DIR / "q0000.json")["gold"] == "A"
    bad_checkpoint = tmp_path / mod.FROZEN_CANDIDATE_RELATIVE_DIR / "q9999.json"
    bad_checkpoint.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint is not an object"):
        mod._read_checkpoint(bad_checkpoint)
    assert mod._jaccard(set(), set()) == 0.0
    assert mod._process_energy({"process_score": "bad"}) == math.inf
    with pytest.raises(ValueError, match="unknown scoring_path"):
        mod.prepare_rows_with_process_scores([], scoring_path="unknown")


def test_scenario_verify_5032_missing_inputs_writes_blocked_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5032: missing fixed cache and frozen inputs blocks honestly."""

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        min_questions=2,
        k_candidates=5,
        audit_runner=_audit_clean,
        summary_runner=lambda _path: 0,
        write=True,
    )

    assert artifact["honest_verdict"] == "blocked_fixed_b2_logprob_cache"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["headroom_present"] is False
    assert artifact["uprm_selection_accuracy"] is None
    assert artifact["genuine_tuned_sc_accuracy"] is None
    assert artifact["scoring_path"] == "blocked"
    assert artifact["preconditions_checked"][0]["available"] is False
    assert artifact["preconditions_checked"][1]["resource"] == "self_supervised_frozen_candidates"
    assert mod.artifact_schema_errors(artifact) == []


def test_req_verify_5032_schema_and_verdict_branches(tmp_path: Path) -> None:
    """REQ-VERIFY-5032: artifact schema and terminal verdict gates are explicit."""

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
        scoring_path="uprm_logprob",
    )
    null = mod.build_complete_artifact(
        evaluation={**base_eval, "verifier_minus_tuned_sc_ci95": [-0.01, 0.2]},
        preconditions_checked=[],
        cache_path=tmp_path / "cache.jsonl",
        duration_s=2.0,
        scoring_path="self_supervised_frozen",
    )
    gated = mod.build_complete_artifact(
        evaluation={**base_eval, "mcnemar_p": 0.5},
        preconditions_checked=[],
        cache_path=tmp_path / "cache.jsonl",
        duration_s=2.0,
        scoring_path="uprm_logprob",
    )
    blocked = mod.build_blocked_artifact(
        missing_resource="fixed_b2_logprob_cache",
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
    assert "degeneracy_guard" in mod.artifact_schema_errors({**blocked, "degeneracy_guard": []})
    missing = dict(blocked)
    missing.pop("duration_s")
    assert "duration_s" in mod.artifact_schema_errors(missing)


def test_scenario_verify_5032_run_error_paths_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-5032: oracle leakage or scoring errors block, not null."""

    cache_path = tmp_path / mod.FIXED_B2_CACHE_RELATIVE_PATH
    _write_jsonl(
        cache_path,
        _cache_rows(0, "A", ["B", "B", "B", "A", "A"], good_index=3),
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
    monkeypatch.setattr(mod, "_no_model_id_shortcut_enforced", lambda _rows: False)
    model_id_blocked = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        min_questions=1,
        k_candidates=5,
        audit_runner=_audit_clean,
        summary_runner=lambda _path: 0,
        write=True,
    )

    assert model_id_blocked["honest_verdict"] == "blocked_oracle_distinctness_violation"
    assert "model_id" in model_id_blocked["blocked_error"]
    assert mod.artifact_schema_errors(model_id_blocked) == []

    monkeypatch.setattr(mod, "_no_model_id_shortcut_enforced", lambda _rows: True)

    def bad_evaluate(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("bad score")

    monkeypatch.setattr(mod, "evaluate_process_rows", bad_evaluate)
    scoring_blocked = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        min_questions=1,
        k_candidates=5,
        audit_runner=_audit_clean,
        summary_runner=lambda _path: 0,
        write=True,
    )

    assert scoring_blocked["honest_verdict"] == "blocked_process_scoring_error"
    assert "bad score" in scoring_blocked["blocked_error"]
    assert mod.artifact_schema_errors(scoring_blocked) == []
