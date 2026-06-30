"""Tests for Exp 5034 uncertainty-routed MuSR cascade v2.

Spec refs: REQ-VERIFY-5034, SCENARIO-VERIFY-5034.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_5034_uncertainty_routed_cascade_v2 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _audit_clean(_path: Path) -> dict[str, Any]:
    return {"flag_count": 0, "flags": []}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _row(row_id: str, gold: str, answers: list[str], energies: list[float]) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "corpus": "MuSR/murder_mysteries",
        "question": f"question {row_id}",
        "context": "fixture",
        "choices": sorted(set(answers)),
        "gold": gold,
        "candidates": [
            {
                "candidate_id": f"{row_id}/cached-{index}",
                "answer": answer,
                "reasoning": f"Reasoning for {answer}",
                "cache_index": index,
                "temperature": "cached",
                "trivial_energy": energy,
                "cached_energy_selected": index == 0,
            }
            for index, (answer, energy) in enumerate(zip(answers, energies, strict=True))
        ],
    }


def _write_checkpoints(root: Path, rows: list[dict[str, Any]]) -> None:
    checkpoint_dir = root / mod.MUSR_CHECKPOINT_RELATIVE_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        _write_json(
            checkpoint_dir / f"q{index:04d}.json",
            {
                "q": index,
                "gold": row["gold"],
                "answers": [candidate["answer"] for candidate in row["candidates"]],
                "candidate_energies": [
                    candidate["trivial_energy"] for candidate in row["candidates"]
                ],
                "energy_answer": row["candidates"][0]["answer"],
                "question": row["question"],
                "context": row["context"],
            },
        )


def _write_d1(root: Path, predictions: list[str], delta: float = 0.08) -> None:
    _write_json(
        root / mod.D1_ARTIFACT_RELATIVE_PATH,
        {
            "scorer_trained": True,
            "verifier_is_oracle": False,
            "delta_vs_tuned_sc": delta,
            "trained_scorer_accuracy": 0.665,
            "model_specs": {"base_model": "Qwen/Qwen3.5-2B"},
            "evaluation": {"verifier": {"predictions": predictions}},
        },
    )


def _write_d2(root: Path, predictions: list[str], delta: float = -0.11) -> None:
    _write_json(
        root / mod.D2_ARTIFACT_RELATIVE_PATH,
        {
            "scoring_path": "uprm_logprob",
            "verifier_is_oracle": False,
            "delta_vs_tuned_sc": delta,
            "uprm_selection_accuracy": 0.475,
            "model_specs": {"score_formula": "uPRM Eq.6"},
            "evaluation": {"verifier": {"predictions": predictions}},
        },
    )


def test_req_verify_5034_spec_declares_cascade_contract() -> None:
    """REQ-VERIFY-5034: OpenSpec anchors the D6 efficiency cascade fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-5034",
        "SCENARIO-VERIFY-5034",
        "experiment_5034_uncertainty_routed_cascade_v2.py",
        "results/experiment_5034_uncertainty_routed_cascade_v2.json",
        "arXiv:2510.20369",
        "success_cascade_parity_at_<pct>_judge_calls",
        "complete_cascade_no_efficiency_win_musr",
        "cost_quality_frontier",
        "judge_call_fraction",
        "gemma-4-12B-it-GGUF",
        "verifier_is_oracle",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_verify_5034_selects_cheap_verifier_priority(tmp_path: Path) -> None:
    """REQ-VERIFY-5034: D1 trained scorer wins, then D2 uPRM, else registry."""

    _write_d1(tmp_path, ["A"], delta=0.08)
    _write_d2(tmp_path, ["B"], delta=0.30)

    selected = mod.select_cheap_verifier(tmp_path)
    (tmp_path / mod.D1_ARTIFACT_RELATIVE_PATH).unlink()
    d2_selected = mod.select_cheap_verifier(tmp_path)
    (tmp_path / mod.D2_ARTIFACT_RELATIVE_PATH).unlink()
    fallback = mod.select_cheap_verifier(tmp_path)

    assert selected.name == "D1 trained LoRA-EBM"
    assert selected.predictions == ["A"]
    assert d2_selected.name == "D2 uPRM"
    assert d2_selected.predictions == ["B"]
    assert fallback.name == "registry quality ensemble"
    assert fallback.check.available is True


def test_req_verify_5034_frontier_uses_cached_judge_predictions_once() -> None:
    """REQ-VERIFY-5034: swept frontier charges per-threshold cascade calls."""

    rows = [
        _row("musr:0", "A", ["A", "A", "A", "B"], [0.0, 0.1, 0.2, 0.3]),
        _row("musr:1", "B", ["A", "A", "B"], [0.0, 0.1, 0.2]),
    ]
    cheap = mod.CheapVerifier(
        name="fixture cheap",
        predictions=["A", "A"],
        model_specs={"cheap_verifier": "fixture"},
        source_artifact=None,
        check=mod.PreconditionCheck("cheap_verifier", True, "fixture"),
        delta_vs_tuned_sc=0.0,
    )
    judge = mod.CountingJudge(lambda row, _candidates: str(row["gold"]))

    result = mod.evaluate_cascade(
        rows,
        cheap_verifier=cheap,
        judge_answer=judge,
        thresholds=[0.0, 0.6, 1.0],
        bootstrap_samples=32,
    )

    assert result.cheap_verifier_only_accuracy == 0.5
    assert result.judge_only_accuracy == 1.0
    assert result.judge_only_calls == 2
    assert result.cascade_accuracy == 1.0
    assert result.cascade_judge_calls == 1
    assert result.judge_call_fraction == 0.5
    assert result.best_threshold == 0.6
    assert judge.calls == 2
    assert result.cost_quality_frontier == [
        {"routing_threshold": 0.0, "accuracy": 0.5, "judge_calls": 0, "judge_call_fraction": 0.0},
        {"routing_threshold": 0.6, "accuracy": 1.0, "judge_calls": 1, "judge_call_fraction": 0.5},
        {"routing_threshold": 1.0, "accuracy": 1.0, "judge_calls": 2, "judge_call_fraction": 1.0},
    ]


def test_scenario_verify_5034_missing_judge_server_blocks_honestly(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5034: missing judge endpoint writes blocked artifact."""

    rows = [
        _row("musr:0", "A", ["A", "B"], [0.0, 0.1]),
        _row("musr:1", "B", ["A", "B"], [0.0, 0.1]),
    ]
    _write_checkpoints(tmp_path, rows)
    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH

    artifact = mod.run(
        root=tmp_path,
        artifact_path=artifact_path,
        min_questions=2,
        judge_server_probe=lambda _url: mod.PreconditionCheck(
            "judge_server", False, "connection refused"
        ),
        audit_runner=_audit_clean,
        summary_runner=lambda _path: 0,
        now=lambda: 10.0,
        write=True,
    )

    assert artifact["honest_verdict"] == "blocked_judge_server"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["cheap_verifier_source"] == "registry quality ensemble"
    assert artifact["cheap_verifier_only_accuracy"] is None
    assert artifact["judge_only_calls"] == 0
    assert artifact["cascade_judge_calls"] == 0
    assert artifact["preconditions_checked"][1]["available"] is False
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_verify_5034_injected_judge_writes_complete_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5034: injected judge emits all cascade artifact fields."""

    rows = [
        _row("musr:0", "A", ["A", "A", "A", "B"], [0.0, 0.1, 0.2, 0.3]),
        _row("musr:1", "B", ["A", "A", "B"], [0.0, 0.1, 0.2]),
    ]
    _write_checkpoints(tmp_path, rows)
    _write_d1(tmp_path, ["A", "A"])
    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH

    artifact = mod.run(
        root=tmp_path,
        artifact_path=artifact_path,
        min_questions=2,
        thresholds=[0.0, 0.6, 1.0],
        judge_server_probe=lambda _url: mod.PreconditionCheck(
            "judge_server", True, "fixture reachable"
        ),
        judge_answer=mod.CountingJudge(lambda row, _candidates: str(row["gold"])),
        audit_runner=_audit_clean,
        summary_runner=lambda _path: 0,
        now=lambda: 10.0,
        bootstrap_samples=32,
        write=True,
    )

    assert artifact["honest_verdict"] == "complete_cascade_no_efficiency_win_musr"
    assert artifact["cheap_verifier_source"] == "D1 trained LoRA-EBM"
    assert artifact["cheap_verifier_only_accuracy"] == 0.5
    assert artifact["judge_only_accuracy"] == 1.0
    assert artifact["judge_only_calls"] == 2
    assert artifact["cascade_accuracy"] == 1.0
    assert artifact["cascade_judge_calls"] == 1
    assert artifact["judge_call_fraction"] == 0.5
    assert artifact["genuine_tuned_sc_accuracy"] == 0.5
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["adversarial_verify_clean"] is True
    assert artifact["summarize_artifact_exit_code"] == 0
    assert artifact["model_specs"]["strong_judge"]["gpu"] == 0
    assert artifact["judge_call_accounting"]["judge_only_baseline_calls"] == 2
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert mod.artifact_schema_errors(artifact) == []


def test_req_verify_5034_helper_edge_cases_and_schema_errors(tmp_path: Path) -> None:
    """REQ-VERIFY-5034: helpers keep blocked, parsing, and schema branches explicit."""

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    checkpoint_dir = tmp_path / mod.MUSR_CHECKPOINT_RELATIVE_DIR
    checkpoint_dir.mkdir(parents=True)
    bad_json.rename(checkpoint_dir / "q0000.json")
    _write_json(checkpoint_dir / "q0001.json", {"answers": []})
    _write_json(
        checkpoint_dir / "q0002.json",
        {"gold": "A", "answers": ["", "A", "B"], "energy_answer": "A"},
    )
    _write_json(checkpoint_dir / "q0003.json", {"gold": "B", "answers": ["A", "B"]})

    rows = mod._checkpoint_rows(checkpoint_dir, min_questions=1, limit=3)
    missing_check, missing_rows = mod.candidate_cache_precondition(
        tmp_path / "missing", min_questions=1
    )

    assert rows[0]["candidates"][0]["answer"] == "A"
    assert rows[0]["candidates"][0]["trivial_energy"] == 0.0
    assert rows[0]["candidates"][1]["trivial_energy"] == 1.002
    assert missing_check.available is False
    assert missing_rows == []
    assert mod._read_json(checkpoint_dir / "q0000.json") is None
    assert mod._finite_number(True) is False
    assert mod._number("1.5") == 1.5
    assert mod._number(True) is None
    assert mod._candidate_score({"uprm_process_score": 2.5}) == -2.5
    assert mod._match_choice('{"answer": "B"}', ["A", "B"]) == "B"
    assert mod._match_choice("{bad}\nANSWER: A", ["A", "B"]) == "A"
    assert mod._match_choice("The best option is B.", ["A", "B"]) == "B"
    assert mod._match_choice("No listed choice.", ["A", "B"]) is None
    assert mod._as_prediction_list({"evaluation": {"verifier": {"predictions": ["A", None]}}}) == [
        "A",
        None,
    ]
    assert mod._as_prediction_list({"evaluation": {"verifier": {"predictions": "bad"}}}) == []
    assert mod._confidence_for_answer(rows[0], None) == 0.0
    assert mod._confidence_for_answer(rows[0], "missing") == 0.0
    assert mod._first_candidate_id(rows[0], "missing") == ""

    try:
        mod._checkpoint_rows(checkpoint_dir, min_questions=3, limit=1)
    except RuntimeError as exc:
        assert "only 0 cached MuSR rows" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected min_questions failure")
    try:
        mod.load_cached_musr_rows(tmp_path / "missing", min_questions=1)
    except RuntimeError as exc:
        assert "candidate checkpoint directory missing" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected missing directory failure")

    cheap = mod.CheapVerifier(
        name="registry quality ensemble",
        predictions=[],
        model_specs={"cheap_verifier": "fixture"},
        source_artifact=None,
        check=mod.PreconditionCheck("cheap_verifier", True, "fixture"),
        delta_vs_tuned_sc=None,
    )
    no_candidates = mod._cheap_decision({"gold": "A", "candidates": []}, cheap, 0)
    nonfinite = mod._cheap_decision(
        _row("musr:9", "B", ["A", "B"], [float("nan"), 0.1]), cheap, 0
    )
    single_nonfinite = mod._registry_decision(
        {"candidates": [{"candidate_id": "x", "answer": "A", "cache_index": float("nan")}]}
    )
    no_parity = mod.evaluate_cascade(
        [_row(f"musr:{i}", "B", ["A", "A", "B"], [0.0, 0.1, 0.2]) for i in range(4)],
        cheap_verifier=cheap,
        judge_answer=mod.CountingJudge(lambda row, _candidates: str(row["gold"])),
        thresholds=[0.0],
        bootstrap_samples=32,
    )

    assert no_candidates.answer is None
    assert nonfinite.answer == "B"
    assert single_nonfinite == ("A", 0.0, "x")
    assert no_parity.parity_with_judge is False
    assert mod._compact_adversarial_flags(
        {"flags": [{"kind": "A"}], "reports": ["bad", {"flags": [{"kind": "B"}, "bad"]}]}
    ) == [{"kind": "A"}, {"kind": "B"}]
    assert mod._audit_is_clean({"flag_count": 1}) is False
    assert mod._audit_is_clean({"flagged_count": 0, "reports": []}) is True

    success_eval = mod.CascadeEvaluation(
        cheap_verifier_only_accuracy=0.5,
        judge_only_accuracy=1.0,
        judge_only_calls=10,
        cascade_accuracy=1.0,
        cascade_judge_calls=4,
        judge_call_fraction=0.4,
        cost_quality_frontier=[],
        best_threshold=0.2,
        genuine_tuned_sc_accuracy=0.5,
        n_questions=10,
        paired_ci95_cascade_vs_judge=[0.0, 0.0],
        parity_with_judge=True,
        judge_predictions_cached_for_sweep=True,
    )
    success_artifact = mod.build_complete_artifact(
        evaluation=success_eval,
        cheap_verifier=cheap,
        preconditions_checked=[],
        duration_s=61.0,
        root=tmp_path,
    )
    assert success_artifact["honest_verdict"] == "success_cascade_parity_at_40pct_judge_calls"
    assert "honest_verdict" in mod.artifact_schema_errors(
        {key: value for key, value in success_artifact.items() if key != "honest_verdict"}
    )
    assert "spec_refs" in mod.artifact_schema_errors({**success_artifact, "spec_refs": []})
    assert "verifier_is_oracle" in mod.artifact_schema_errors(
        {**success_artifact, "verifier_is_oracle": True}
    )
    assert "field_principles" in mod.artifact_schema_errors(
        {**success_artifact, "field_principles": {}}
    )
    assert "cost_quality_frontier" in mod.artifact_schema_errors(
        {**success_artifact, "cost_quality_frontier": {}}
    )
    assert "paired_ci95_cascade_vs_judge" in mod.artifact_schema_errors(
        {**success_artifact, "paired_ci95_cascade_vs_judge": [0.0]}
    )


def test_scenario_verify_5034_judge_inference_failure_blocks(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5034: live judge failures become blocked resources."""

    _write_checkpoints(
        tmp_path,
        [_row("musr:0", "A", ["A", "B"], [0.0, 0.1])],
    )

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        min_questions=1,
        thresholds=[0.0],
        judge_server_probe=lambda _url: mod.PreconditionCheck(
            "judge_server", True, "fixture reachable"
        ),
        judge_answer=lambda _row, _candidates: (_ for _ in ()).throw(OSError("judge down")),
        audit_runner=_audit_clean,
        summary_runner=lambda _path: 0,
        now=lambda: 10.0,
        bootstrap_samples=32,
        write=False,
    )

    assert artifact["honest_verdict"] == "blocked_judge_inference_failed"
    assert artifact["cheap_verifier_source"] == "registry quality ensemble"
    assert artifact["preconditions_checked"][-1]["resource"] == "judge_inference_failed"
    assert artifact["adversarial_verify_clean"] is True
    assert mod.artifact_schema_errors(artifact) == []
