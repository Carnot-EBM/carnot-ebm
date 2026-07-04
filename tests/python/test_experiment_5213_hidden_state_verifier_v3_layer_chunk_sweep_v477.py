"""Tests for Exp 5213 hidden-state verifier v3 layer/chunk sweep.

Spec refs: REQ-REPORT-5213, SCENARIO-REPORT-5213,
SCENARIO-REPORT-5213-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot import experiment_5213_hidden_state_verifier_v3_layer_chunk_sweep_v477 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _write_headroom(root: Path) -> None:
    path = root / mod.HEADROOM_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "oracle_at_k": 0.35,
                "sc_vote": 0.075,
                "headroom": 0.275,
                "headroom_ci95": [0.15, 0.425],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_pool(root: Path, *, n_questions: int = 8, n_candidates: int = 4) -> None:
    path = root / mod.CANDIDATE_POOL_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for qi in range(n_questions):
        gold = "D"
        for ci in range(n_candidates):
            correct = ci == n_candidates - 1
            rows.append(
                {
                    "question_index": qi,
                    "question_id": f"mmlu-{qi:03d}",
                    "category": "fixture",
                    "k": ci,
                    "gold": gold,
                    "parsed_letter": gold if correct else "A",
                    "correct": correct,
                    "full_text": (
                        f"Step {ci}. {'correct-signal' if correct else 'wrong-cluster'} "
                        f"reasoning for question {qi}.\n\nFinal answer: {gold if correct else 'A'}."
                    ),
                    "token_logprobs": [-4.0] if correct else [-0.01],
                    "top_logprobs": [
                        {"A": -1.3863, "B": -1.3863, "C": -1.3863, "D": -1.3863}
                        if correct
                        else {"A": -0.01, "B": -6.0, "C": -6.0, "D": -6.0}
                    ],
                }
            )
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _repo_with_pool(tmp_path: Path) -> Path:
    _write_headroom(tmp_path)
    _write_pool(tmp_path)
    (tmp_path / "ops").mkdir(parents=True, exist_ok=True)
    (tmp_path / mod.VERIFIER_GAPS_RELATIVE_PATH).write_text("# Verifier gaps\n", encoding="utf-8")
    return tmp_path


def _inventory() -> mod.ModelInventory:
    return mod.ModelInventory(
        cached_sota_pair_attempted=True,
        cached_sota_pair_available=True,
        models_used=list(mod.MANDATED_MODEL_IDS),
        model_specs=[
            {"name": "Qwen3.6-35B-A3B", "hf_id": mod.MANDATED_MODEL_IDS[0], "gpu": 0, "model_path": "/m/qwen.gguf"},
            {"name": "Gemma4-31B-it", "hf_id": mod.MANDATED_MODEL_IDS[1], "gpu": 1, "model_path": "/m/gemma31.gguf"},
            {"name": "Gemma4-26B-A4B-it", "hf_id": mod.MANDATED_MODEL_IDS[2], "gpu": 0, "model_path": "/m/gemma26.gguf"},
        ],
    )


def _signal_status(*, intermediate: bool, chunk: bool = True, halting: bool = True) -> mod.SignalAvailability:
    return mod.SignalAvailability(
        usable=True,
        reason="fixture richer signal path",
        intermediate_layer_available=intermediate,
        chunk_features_available=chunk,
        halting_or_convergence_signal_available=halting,
        extraction_path="fixture_hidden_state_provider",
        transformer_attempt={
            "attempted": True,
            "output_hidden_states_requested": True,
            "status": "fixture_available" if intermediate else "blocked_insufficient_gpu_memory",
            "hf_id": "unsloth/Qwen3.6-35B-A3B",
        },
        tensor_provenance=[
            {
                "model_hf_id": mod.MANDATED_MODEL_IDS[0],
                "feature": "intermediate_layer_state" if intermediate else "gguf_final_token_embedding",
                "candidate_rows": 32,
                "timing_ref": {"embed_s": 0.05, "calls": 32},
            }
        ],
    )


def _rich_features(questions: list[mod.v2.MmluQuestion]) -> mod.FeatureBatch:
    rows: list[list[float]] = []
    keys: list[tuple[int, int]] = []
    for question in questions:
        for candidate in question.candidates:
            if candidate.correct:
                rows.append([4.0, 0.0, 1.0, 0.25, 0.2, 3.0])
            else:
                rows.append([0.0, 3.0, 0.0, 0.95, 0.8, 0.1])
            keys.append((question.question_pos, candidate.candidate_pos))
    return mod.FeatureBatch(np.asarray(rows, dtype=float), keys)


def _flat_features(questions: list[mod.v2.MmluQuestion]) -> mod.FeatureBatch:
    rows: list[list[float]] = []
    keys: list[tuple[int, int]] = []
    for question in questions:
        for candidate in question.candidates:
            rows.append([0.0, 0.0, 0.0, 0.0])
            keys.append((question.question_pos, candidate.candidate_pos))
    return mod.FeatureBatch(np.asarray(rows, dtype=float), keys)


def test_req_report_5213_spec_declares_v3_contract() -> None:
    """REQ-REPORT-5213: OpenSpec declares the layer/chunk sweep contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-5213") :]

    for marker in (
        "REQ-REPORT-5213",
        "SCENARIO-REPORT-5213",
        "SCENARIO-REPORT-5213-BLOCKED-PRECONDITION",
        mod.RESULT_RELATIVE_PATH,
        "output_hidden_states=True",
        "retire_mmlu_hidden_state_path",
    ):
        assert marker in section
    for field in mod.REQUIRED_PRINCIPLED_FIELDS:
        assert f"`{field}`" in section


def test_req_report_5213_model_inventory_calls_cached_pair_first() -> None:
    """REQ-REPORT-5213: cached_sota_pair drives GGUF model resolution first."""

    calls: list[str] = []

    def cached_pair() -> list[dict[str, Any]]:
        calls.append("cached_pair")
        return [
            {"name": "Qwen3.6-35B-A3B", "hf_id": mod.MANDATED_MODEL_IDS[0], "gpu": 0, "model_path": "/m/qwen.gguf"},
            {"name": "Gemma4-26B-A4B-it", "hf_id": mod.MANDATED_MODEL_IDS[2], "gpu": 1, "model_path": "/m/gemma26.gguf"},
        ]

    def resolve(hf_id: str, preferred_quant: str = "Q4_K_M") -> str | None:
        calls.append(f"resolve:{hf_id}:{preferred_quant}")
        return "/m/gemma31.gguf" if hf_id == mod.MANDATED_MODEL_IDS[1] else None

    inventory = mod.resolve_model_inventory(cached_pair_fn=cached_pair, resolve_gguf_fn=resolve)

    assert calls[0] == "cached_pair"
    assert inventory.cached_sota_pair_attempted is True
    assert inventory.cached_sota_pair_available is True
    assert inventory.models_used == list(mod.MANDATED_MODEL_IDS)
    assert [row["hf_id"] for row in inventory.model_specs] == list(mod.MANDATED_MODEL_IDS)
    assert inventory.model_specs[1]["model_path"] == "/m/gemma31.gguf"


def test_scenario_report_5213_builds_richer_signal_success_artifact(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5213: richer signals are compared against all controls."""

    root = _repo_with_pool(tmp_path)
    artifact = mod.run(
        root=root,
        result_path=root / mod.RESULT_RELATIVE_PATH,
        feature_provider=_rich_features,
        signal_status=_signal_status(intermediate=True),
        model_inventory=_inventory(),
        expected_pool_rows=32,
        n_folds=4,
        n_bootstrap=200,
        duration_s=3.25,
        tests_run=["unit fixture"],
    )

    assert artifact["models_used"]["value"] == list(mod.MANDATED_MODEL_IDS)
    assert artifact["intermediate_layer_available"]["value"] is True
    assert artifact["chunk_features_available"]["value"] is True
    assert artifact["halting_or_convergence_signal_available"]["value"] is True
    assert artifact["best_probe_accuracy"]["value"] == pytest.approx(1.0)
    assert artifact["best_probe_accuracy"]["value"] > artifact["tuned_sc_accuracy"]["value"]
    assert artifact["best_probe_accuracy"]["value"] > artifact["self_certainty_accuracy"]["value"]
    assert artifact["best_probe_accuracy"]["value"] > artifact["clue_accuracy"]["value"]
    assert artifact["best_probe_accuracy"]["value"] > artifact["radial_consensus_score_accuracy"]["value"]
    assert artifact["beats_all_controls"]["value"] is True
    assert artifact["retire_mmlu_hidden_state_path"]["value"] is False
    assert artifact["verifier_is_oracle"]["value"] is False
    assert artifact["inference_substrate"]["value"] == "live_llm_hidden_state_extraction"
    assert artifact["honest_verdict"]["value"].startswith("success_")
    for comparison in artifact["control_comparisons"].values():
        assert comparison["delta_ci95"][0] > 0.0
        assert 0.0 <= comparison["mcnemar_p"] <= 1.0
    assert artifact["split_summary"]["leakage_guard"] == "question_id_grouped_train_eval_split"
    assert artifact["reproducibility_checksum"]["value"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []
    assert json.loads((root / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact


def test_scenario_report_5213_retires_when_controls_not_beaten(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5213: a null richer-signal result retires this path."""

    root = _repo_with_pool(tmp_path)
    artifact = mod.run(
        root=root,
        result_path=root / mod.RESULT_RELATIVE_PATH,
        feature_provider=_flat_features,
        signal_status=_signal_status(intermediate=False, chunk=True, halting=False),
        model_inventory=_inventory(),
        expected_pool_rows=32,
        n_folds=4,
        n_bootstrap=120,
        duration_s=3.25,
    )

    assert artifact["intermediate_layer_available"]["value"] is False
    assert artifact["best_probe_accuracy"]["value"] == pytest.approx(0.0)
    assert artifact["beats_all_controls"]["value"] is False
    assert artifact["retire_mmlu_hidden_state_path"]["value"] is True
    assert artifact["honest_verdict"]["value"].startswith("complete_")
    assert "retires_mmlu_hidden_state_path" in artifact["honest_verdict"]["value"]
    gaps = (root / mod.VERIFIER_GAPS_RELATIVE_PATH).read_text(encoding="utf-8")
    assert "experiment_5213_hidden_state_verifier_v3_layer_chunk_sweep_v477" in gaps
    assert "recommendation: retire MMLU-Pro hidden-state verifier path" in gaps


def test_req_report_5213_question_folds_do_not_leak_candidates(tmp_path: Path) -> None:
    """REQ-REPORT-5213: train/eval splits are grouped by question_id."""

    root = _repo_with_pool(tmp_path)
    questions = mod.v2.load_mmlu_questions(root, expected_rows=32)
    folds = mod.v2.question_folds([q.question_id for q in questions], n_folds=4, seed=13)

    seen: set[str] = set()
    for fold in folds:
        assert seen.isdisjoint(fold)
        seen.update(fold)
        train_rows, eval_rows = mod.v2.rows_for_split(questions, fold)
        train_q = {questions[row.question_pos].question_id for row in train_rows}
        eval_q = {questions[row.question_pos].question_id for row in eval_rows}
        assert train_q.isdisjoint(eval_q)
    assert seen == {q.question_id for q in questions}


def test_scenario_report_5213_blocked_candidate_pool_is_terminal(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5213-BLOCKED-PRECONDITION: missing pool blocks honestly."""

    _write_headroom(tmp_path)
    artifact = mod.run(
        root=tmp_path,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        model_inventory=_inventory(),
        signal_status=_signal_status(intermediate=False),
        expected_pool_rows=32,
        duration_s=1.0,
        tests_run=["blocked fixture"],
    )

    assert artifact["honest_verdict"]["value"].startswith("blocked_candidate_pool")
    assert artifact["best_probe_accuracy"]["value"] == 0.0
    assert artifact["beats_all_controls"]["value"] is False
    assert artifact["retire_mmlu_hidden_state_path"]["value"] is True
    assert artifact["verifier_is_oracle"]["value"] is False
    assert artifact["inference_substrate"]["value"] == "live_llm_hidden_state_extraction"
    assert mod.artifact_schema_errors(artifact) == []


def test_req_report_5213_preflight_and_feature_validation_edges(tmp_path: Path) -> None:
    """REQ-REPORT-5213: preflight and feature validation fail closed."""

    assert mod._round_float(None) is None
    assert mod._round_float(float("nan")) is None
    assert mod._round_float(1.23456, 2) == 1.23
    assert mod.payload_checksum({"reproducibility_checksum": "raw"}) .startswith("sha256:")

    low = mod.attempt_transformers_hidden_state_path(
        _inventory(),
        gpu_rows_fn=lambda: [{"memory_free_gb": 1.0}],
    )
    high = mod.attempt_transformers_hidden_state_path(
        _inventory(),
        gpu_rows_fn=lambda: [{"memory_free_gb": 96.0}],
    )

    assert low["status"] == "blocked_insufficient_gpu_memory_for_non_gguf_transformers_load"
    assert high["status"] == "blocked_transformers_load_not_executed_without_explicit_quantized_fit_path"

    questions = mod.v2.load_mmlu_questions(_repo_with_pool(tmp_path), expected_rows=32)
    bad_batch = mod.FeatureBatch(np.asarray([1.0, 2.0]), [(0, 0)])
    with pytest.raises(ValueError, match="expected 2-D"):
        mod._candidate_feature_map(questions, bad_batch)
    with pytest.raises(ValueError, match="expected 2-D"):
        mod.evaluate_selectors(questions, bad_batch, [])
    with pytest.raises(ValueError, match="feature key count"):
        mod.evaluate_selectors(questions, mod.FeatureBatch(np.zeros((2, 2)), [(0, 0)]), [])

    assert not mod._beats_all_controls(
        0.5,
        {"tuned_sc": 0.4},
        {"probe_vs_tuned_sc": {"delta_ci95": [0.0, 0.2]}},
        headline_eligible=True,
    )
    assert mod._verdict(0.0, {"tuned_sc": 0.0, "self_certainty": 0.0, "clue": 0.0, "radial_consensus_score": 0.0}, False, False).startswith(
        "complete_hidden_state_v3_signal_inconclusive"
    )


def test_req_report_5213_auto_provider_path_is_wired(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-5213: default run attempts Transformers then wires live provider."""

    root = _repo_with_pool(tmp_path)

    def fake_attempt(inventory: mod.ModelInventory) -> dict[str, Any]:
        assert inventory.models_used
        return {"attempted": True, "output_hidden_states_requested": True, "status": "fixture"}

    def fake_live(
        inventory: mod.ModelInventory,
        transformer_attempt: dict[str, Any],
    ) -> tuple[mod.SignalAvailability, mod.FeatureProvider]:
        assert transformer_attempt["output_hidden_states_requested"] is True
        return _signal_status(intermediate=True), _rich_features

    monkeypatch.setattr(mod, "attempt_transformers_hidden_state_path", fake_attempt)
    monkeypatch.setattr(mod, "make_live_feature_provider", fake_live)

    artifact = mod.run(
        root=root,
        result_path=root / mod.RESULT_RELATIVE_PATH,
        model_inventory=_inventory(),
        expected_pool_rows=32,
        n_folds=4,
        n_bootstrap=80,
        duration_s=3.25,
    )

    assert artifact["signal_availability"]["transformer_attempt"]["status"] == "fixture_available"
    assert artifact["beats_all_controls"]["value"] is True


def test_scenario_report_5213_blocked_signal_path_is_terminal(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5213-BLOCKED-PRECONDITION: unusable signals block honestly."""

    root = _repo_with_pool(tmp_path)
    status = mod.SignalAvailability(
        usable=False,
        reason="blocked_hidden_state_access_infeasible: fixture",
        intermediate_layer_available=False,
        chunk_features_available=False,
        halting_or_convergence_signal_available=False,
        extraction_path="fixture_blocked",
        transformer_attempt={"attempted": True, "output_hidden_states_requested": True},
        tensor_provenance=[],
    )

    artifact = mod.run(
        root=root,
        result_path=root / mod.RESULT_RELATIVE_PATH,
        feature_provider=_flat_features,
        signal_status=status,
        model_inventory=_inventory(),
        expected_pool_rows=32,
        duration_s=2.0,
    )

    assert artifact["honest_verdict"]["value"].startswith("blocked_hidden_state_access_infeasible")
    assert artifact["retire_mmlu_hidden_state_path"]["value"] is True


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda artifact: {key: value for key, value in artifact.items() if key != "models_used"},
            "missing required fields",
        ),
        (
            lambda artifact: artifact
            | {"verifier_is_oracle": {"value": True, "principle": mod.FIELD_PRINCIPLES["verifier_is_oracle"]}},
            "verifier_is_oracle",
        ),
        (
            lambda artifact: artifact | {"best_probe_accuracy": 1.0},
            "principle-wrapped",
        ),
        (
            lambda artifact: artifact
            | {"honest_verdict": {"value": "done", "principle": mod.FIELD_PRINCIPLES["honest_verdict"]}},
            "honest_verdict",
        ),
        (
            lambda artifact: artifact
            | {"models_used": {"value": [], "principle": "wrong"}},
            "wrong principle",
        ),
        (
            lambda artifact: artifact
            | {"inference_substrate": {"value": "cached", "principle": mod.FIELD_PRINCIPLES["inference_substrate"]}},
            "inference_substrate",
        ),
        (
            lambda artifact: artifact
            | {
                "chunk_features_available": {
                    "value": "yes",
                    "principle": mod.FIELD_PRINCIPLES["chunk_features_available"],
                }
            },
            "chunk_features_available must be bool",
        ),
        (
            lambda artifact: artifact
            | {
                "beats_all_controls": {"value": True, "principle": mod.FIELD_PRINCIPLES["beats_all_controls"]},
                "retire_mmlu_hidden_state_path": {
                    "value": True,
                    "principle": mod.FIELD_PRINCIPLES["retire_mmlu_hidden_state_path"],
                },
            },
            "cannot both be true",
        ),
        (
            lambda artifact: artifact
            | {
                "beats_all_controls": {"value": True, "principle": mod.FIELD_PRINCIPLES["beats_all_controls"]},
                "control_comparisons": {"probe_vs_tuned_sc": {"delta_ci95": [0.0, 1.0], "mcnemar_p": 1.0}},
            },
            "positive CI",
        ),
        (
            lambda artifact: artifact
            | {"model_specs": {"value": "bad", "principle": mod.FIELD_PRINCIPLES["model_specs"]}},
            "model_specs must be a list",
        ),
        (
            lambda artifact: artifact
            | {"model_specs": {"value": [{}], "principle": mod.FIELD_PRINCIPLES["model_specs"]}},
            "model_specs rows",
        ),
        (
            lambda artifact: artifact
            | {
                "model_specs": {
                    "value": [{"name": "x", "hf_id": "y", "gpu": 0}],
                    "principle": mod.FIELD_PRINCIPLES["model_specs"],
                }
            },
            "model_path/load_path",
        ),
        (
            lambda artifact: artifact
            | {
                "reproducibility_checksum": {
                    "value": "not-sha",
                    "principle": mod.FIELD_PRINCIPLES["reproducibility_checksum"],
                }
            },
            "reproducibility_checksum must be sha256",
        ),
        (
            lambda artifact: artifact
            | {
                "reproducibility_checksum": {
                    "value": "sha256:bad",
                    "principle": mod.FIELD_PRINCIPLES["reproducibility_checksum"],
                }
            },
            "checksum mismatch",
        ),
    ],
)
def test_req_report_5213_schema_rejects_bad_artifacts(
    tmp_path: Path,
    mutate: Any,
    message: str,
) -> None:
    """REQ-REPORT-5213: malformed required fields fail closed."""

    artifact = mod.run(
        root=_repo_with_pool(tmp_path),
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        feature_provider=_rich_features,
        signal_status=_signal_status(intermediate=True),
        model_inventory=_inventory(),
        expected_pool_rows=32,
        n_folds=4,
        n_bootstrap=80,
        duration_s=3.25,
    )

    errors = mod.artifact_schema_errors(mutate(artifact))

    assert any(message in error for error in errors)
