"""Tests for Exp 1420 DPO-style verified-pair probe.

Spec: REQ-LEARN-1420, SCENARIO-LEARN-1420.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from carnot.reporting import dpo_verified_pairs_probe as mod


def _exp1395(promoted: list[str], demoted: list[str]) -> dict[str, Any]:
    return {
        "status": "complete",
        "fresh_verified_sample_count": len(promoted),
        "memory_updates": {
            "promoted": [f"dvi_v2:fover:{case_id}" for case_id in promoted],
            "demoted": [f"dvi_v2:fover:{case_id}" for case_id in demoted],
        },
        "honest_verdict": "fr11_self_learning_v5_dvi_v2_secl_headline_allowed_fresh_1508_delta_1449_grpo_0",
    }


def _row(case_id: str, label: str, text: str, confidence: float = 1.0) -> dict[str, Any]:
    return {
        "question_id": case_id,
        "step_text": text,
        "label": label,
        "confidence": confidence,
        "model": "base_model",
        "source": "fover_v4",
        "verifier": "heuristic",
    }


def test_req_learn_1420_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-LEARN-1420-1: bootstrap output contains the required terminal fields."""

    output_path = tmp_path / mod.OUTPUT_FILE

    artifact = mod.write_in_progress_artifact(
        output_path,
        project_root=tmp_path,
        run_date="20260506",
    )

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(written)
    assert written["status"] == "in_progress"
    assert written["verified_pairs_available"] is None
    assert written["dpo_full_finetune_performed"] is None
    assert written["dpo_reranker_fallback_used"] is None
    assert written["headline_result_allowed"] is False
    assert written["model_specs"] == list(mod.MODEL_SPECS)


def test_req_learn_1420_builds_nearest_demoted_preference_pairs() -> None:
    """REQ-LEARN-1420-2: promoted cases pair with nearest demoted FoVer candidates."""

    rows = [
        _row("r0", "correct", "short rejected candidate"),
        _row("p1", "incorrect", "preferred candidate with \\boxed{42} and detailed math"),
        _row("noise", "correct", "unlisted row"),
        _row("p4", "incorrect", "another preferred candidate with many 1 2 3 digits"),
        _row("r5", "correct", "near rejected candidate"),
    ]

    pairs = mod.build_preference_pairs(
        exp1395_artifact=_exp1395(promoted=["p1", "p4"], demoted=["r0", "r5"]),
        fover_rows=rows,
    )

    assert [pair.preferred_id for pair in pairs] == ["p1", "p4"]
    assert [pair.rejected_id for pair in pairs] == ["r0", "r5"]
    assert pairs[0].preferred_verified is True
    assert pairs[0].rejected_verified is False
    assert pairs[0].preferred_text.startswith("preferred candidate")
    assert pairs[1].rejected_text == "near rejected candidate"


def test_req_learn_1420_duplicate_fover_ids_match_exp1395_suffixing() -> None:
    """REQ-LEARN-1420-2: duplicate question IDs receive stable _1 suffixes."""

    rows = [
        _row("same", "correct", "first duplicate demoted"),
        _row("same", "incorrect", "second duplicate promoted"),
    ]

    pairs = mod.build_preference_pairs(
        exp1395_artifact=_exp1395(promoted=["same_1"], demoted=["same"]),
        fover_rows=rows,
    )

    assert len(pairs) == 1
    assert pairs[0].preferred_id == "same_1"
    assert pairs[0].rejected_id == "same"
    assert pairs[0].preferred_text == "second duplicate promoted"


def test_req_learn_1420_missing_rejected_candidate_yields_no_pair() -> None:
    """REQ-LEARN-1420-2: pair construction is honest when no rejection exists."""

    pairs = mod.build_preference_pairs(
        exp1395_artifact=_exp1395(promoted=["p1"], demoted=[]),
        fover_rows=[_row("p1", "incorrect", "preferred without rejected neighbor")],
    )

    assert pairs == []


def test_req_learn_1420_reranker_measures_auc_and_pair_accuracy_delta() -> None:
    """REQ-LEARN-1420-5: fallback metrics are measured from held-out pairs."""

    pairs = [
        mod.PreferencePair(
            pair_id=f"pair_{idx}",
            preferred_id=f"p{idx}",
            rejected_id=f"r{idx}",
            prompt="",
            preferred_text=f"preferred proof with boxed answer {idx} " * 4,
            rejected_text="short rejected",
            preferred_label="incorrect",
            rejected_label="correct",
            preferred_confidence=0.9,
            rejected_confidence=0.9,
            preferred_source="math_z3_v3",
            rejected_source="fover_v4",
            preferred_verified=True,
            rejected_verified=False,
            preferred_corpus_index=idx * 2,
            rejected_corpus_index=idx * 2 + 1,
        )
        for idx in range(10)
    ]

    result = mod.train_reranker_fallback(pairs, train_fraction=0.6, learning_rate=0.2, steps=80)

    assert result.fallback_used is True
    assert result.n_train_pairs == 6
    assert result.n_eval_pairs == 4
    assert result.auroc == pytest.approx(1.0)
    assert result.reranker_pair_accuracy == pytest.approx(1.0)
    assert result.baseline_pair_accuracy == pytest.approx(0.5)
    assert result.improvement_pp == pytest.approx(50.0)


def test_req_learn_1420_reranker_metric_edge_cases_are_explicit() -> None:
    """REQ-LEARN-1420-5: degenerate fallback metrics return null-style values."""

    empty = mod.train_reranker_fallback([])
    assert empty.improvement_pp is None
    assert empty.auroc is None
    assert empty.weights == []

    one_pair = [
        mod.PreferencePair(
            pair_id="one",
            preferred_id="p",
            rejected_id="r",
            prompt="",
            preferred_text="same",
            rejected_text="same",
            preferred_label="correct",
            rejected_label="correct",
            preferred_confidence=0.5,
            rejected_confidence=0.5,
            preferred_source="fover_v4",
            rejected_source="fover_v4",
            preferred_verified=True,
            rejected_verified=False,
            preferred_corpus_index=0,
            rejected_corpus_index=1,
        )
    ]
    single = mod.train_reranker_fallback(one_pair, steps=1)
    assert single.n_train_pairs == 1
    assert single.n_eval_pairs == 1
    assert mod._pair_accuracy([]) is None
    assert mod._auroc([1, 1], [0.2, 0.1]) is None
    assert mod._auroc([1, 0], [0.5, 0.5]) == pytest.approx(0.5)


def test_req_learn_1420_direct_gguf_dpo_is_not_simulated() -> None:
    """REQ-LEARN-1420-4: GGUF DPO support remains false without a trainable path."""

    checks = [
        {
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "cached": True,
            "model_path": "/models/qwen.gguf",
        }
    ]

    feasibility = mod.assess_direct_dpo_support(
        checks,
        import_module_fn=lambda _name: SimpleNamespace(DPOTrainer=object),
    )

    assert feasibility.supported is False
    assert feasibility.reason == "gguf_direct_weight_update_not_supported_by_trl_llama_cpp"
    assert feasibility.packages_checked == {"trl": True}


def test_req_learn_1420_missing_trl_keeps_direct_dpo_blocked() -> None:
    """REQ-LEARN-1420-4: cached GGUF alone is not enough for direct DPO."""

    feasibility = mod.assess_direct_dpo_support(
        [{"cached": True, "model_path": "/models/qwen.gguf"}],
        import_module_fn=lambda _name: (_ for _ in ()).throw(ImportError("missing")),
    )

    assert feasibility.supported is False
    assert feasibility.reason == "trl_not_available_for_dpo"
    assert feasibility.packages_checked == {"trl": False}


def test_scenario_learn_1420_unsupported_gguf_uses_reranker_fallback(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-1420: unsupported direct DPO writes honest fallback artifact."""

    rows = [
        _row("r0", "correct", "short rejected"),
        _row("p1", "incorrect", "preferred proof with boxed answer " * 4),
        _row("r2", "correct", "short rejected"),
        _row("p3", "incorrect", "preferred proof with boxed answer " * 4),
        _row("r4", "correct", "short rejected"),
        _row("p5", "incorrect", "preferred proof with boxed answer " * 4),
        _row("r6", "correct", "short rejected"),
        _row("p7", "incorrect", "preferred proof with boxed answer " * 4),
        _row("r8", "correct", "short rejected"),
        _row("p9", "incorrect", "preferred proof with boxed answer " * 4),
    ]
    output_path = tmp_path / mod.OUTPUT_FILE

    artifact = mod.run(
        exp1395_artifact=_exp1395(
            promoted=["p1", "p3", "p5", "p7", "p9"],
            demoted=["r0", "r2", "r4", "r6", "r8"],
        ),
        fover_rows=rows,
        out_path=output_path,
        project_root=tmp_path,
        run_date="20260506",
        resolver_fn=lambda _hf_id, _preferred_quant: None,
        import_module_fn=lambda _name: (_ for _ in ()).throw(ImportError("missing")),
    )

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["verified_pairs_available"] == 5
    assert artifact["dpo_full_finetune_performed"] is False
    assert artifact["dpo_reranker_fallback_used"] is True
    assert artifact["dpo_improvement_pp"] is not None
    assert artifact["dpo_vs_baseline_auroc"] is not None
    assert artifact["local_sota_model_used"] is False
    assert artifact["headline_result_allowed"] is False
    assert artifact["honest_verdict"] == "gguf_dpo_unsupported_reranker_fallback_measured"


def test_req_learn_1420_run_loads_json_sources_from_paths(tmp_path: Path) -> None:
    """REQ-LEARN-1420-1/6: runner can load source artifact and FoVer JSONL paths."""

    exp1395_path = tmp_path / "exp1395.json"
    fover_path = tmp_path / "fover.jsonl"
    output_path = tmp_path / mod.OUTPUT_FILE
    exp1395_path.write_text(
        json.dumps(_exp1395(promoted=["p1"], demoted=["r0"])),
        encoding="utf-8",
    )
    fover_path.write_text(
        "\n"
        + json.dumps(_row("r0", "correct", "rejected"))
        + "\n"
        + json.dumps(_row("p1", "incorrect", "preferred proof with boxed answer"))
        + "\n",
        encoding="utf-8",
    )

    artifact = mod.run(
        exp1395_path=exp1395_path,
        fover_path=fover_path,
        out_path=output_path,
        project_root=tmp_path,
        resolver_fn=lambda _hf_id, _preferred_quant: "/models/qwen.gguf",
        import_module_fn=lambda _name: SimpleNamespace(DPOTrainer=object),
    )

    assert artifact["verified_pairs_available"] == 1
    assert artifact["gguf_model_checks"][0]["cached"] is True
    assert artifact["direct_dpo_feasibility"]["reason"] == (
        "gguf_direct_weight_update_not_supported_by_trl_llama_cpp"
    )


def test_req_learn_1420_blocked_artifact_and_invalid_source_handling(tmp_path: Path) -> None:
    """REQ-LEARN-1420-6: blocked verdicts and invalid source files are explicit."""

    artifact = mod.build_artifact(
        exp1395_artifact=_exp1395(promoted=["p1"], demoted=[]),
        fover_rows=[_row("p1", "incorrect", "preferred")],
        resolver_fn=lambda _hf_id, _preferred_quant: None,
    )

    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_no_preference_pairs_built"
    assert mod._honest_verdict(
        1,
        mod.DirectDPOFeasibility(True, "supported", {"trl": True}),
    ) == "direct_dpo_supported_not_executed_by_fallback_probe"

    bad_path = tmp_path / "bad_source.json"
    bad_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSON object"):
        mod.load_json(bad_path)
