"""Tests for Exp 5003 LoRA-EBM MuSR scorer.

Spec refs: REQ-VERIFY-5003, SCENARIO-VERIFY-5003.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5003_lora_ebm_scorer_musr as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_checkpoint(path: Path, *, gold: str, answers: list[str | None]) -> None:
    _write_json(
        path,
        {
            "q": int(path.stem.removeprefix("q")),
            "gold": gold,
            "answers": answers,
            "temperature": "cached",
            "energy_answer": answers[0],
        },
    )


def _write_base_cache(root: Path) -> None:
    base_dir = root / "hf" / "models--Qwen--Qwen3.5-4B" / "snapshots" / "abc"
    base_dir.mkdir(parents=True)
    (root / "hf" / "models--Qwen--Qwen3.5-4B" / "refs").mkdir()
    (root / "hf" / "models--Qwen--Qwen3.5-4B" / "refs" / "main").write_text(
        "abc",
        encoding="utf-8",
    )
    (base_dir / "config.json").write_text("{}", encoding="utf-8")
    (base_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (base_dir / "model.safetensors.index.json").write_text("{}", encoding="utf-8")


def _audit_clean(_path: Path) -> dict[str, Any]:
    return {"flag_count": 0, "flags": []}


def test_req_verify_5003_spec_declares_lora_ebm_musr_contract() -> None:
    """REQ-VERIFY-5003: OpenSpec declares fields, blockers, and result path."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-5003",
        "SCENARIO-VERIFY-5003",
        "experiment_5003_lora_ebm_scorer_musr.py",
        "blocked_<resource>",
        "oracle_distinctness_enforced",
        "success_lora_ebm_beats_sc_musr_",
        "complete_lora_ebm_no_win_musr_",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_verify_5003_builds_fover_and_cached_musr_contrastive_pairs(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-5003: FoVer reasoning and gold-labeled MuSR pairs feed training."""

    fover_path = tmp_path / "data" / "fover_train.json"
    ckpt_dir = tmp_path / "results" / "distributional_energy_verifier_musr_checkpoints"
    _write_json(
        fover_path,
        [
            {"question_id": "same", "step_text": "valid derivation", "label": "correct"},
            {"question_id": "same", "step_text": "bad derivation", "label": "incorrect"},
            {"question_id": "pos", "step_text": "another valid derivation", "label": "correct"},
            {"question_id": "neg", "step_text": "another bad derivation", "label": "incorrect"},
        ],
    )
    _write_checkpoint(ckpt_dir / "q0000.json", gold="A", answers=["A", "B", None])

    fover_pairs = mod.load_fover_pairs([fover_path], max_pairs=4)
    musr_pairs = mod.load_musr_training_pairs(ckpt_dir, limit=1, max_pairs=4)
    pairs = mod.build_contrastive_corpus([fover_path], ckpt_dir, limit=1, max_pairs=8)

    assert (fover_pairs[0].good_text, fover_pairs[0].bad_text) == (
        "valid derivation",
        "bad derivation",
    )
    assert musr_pairs[0].good_text.endswith("Final answer: A")
    assert musr_pairs[0].bad_text.endswith("Final answer: B")
    assert {pair.source for pair in pairs} == {"fover", "musr_cached_gold_labeled"}
    assert all(pair.good_text and pair.bad_text for pair in pairs)


def test_req_verify_5003_helper_edge_cases_are_deterministic(tmp_path: Path) -> None:
    """REQ-VERIFY-5003: loaders skip malformed rows and expose bounded failures."""

    fover_path = tmp_path / "data" / "fover_train.json"
    ckpt_dir = tmp_path / "results" / "distributional_energy_verifier_musr_checkpoints"
    snapshot = tmp_path / "hf" / "models--Qwen--Qwen3.5-4B" / "snapshots" / "fallback"
    snapshot.mkdir(parents=True)
    _write_json(
        fover_path,
        {
            "rows": [
                {"question_id": "blank", "step_text": "", "label": "correct"},
                {"question_id": "unknown", "step_text": "ignored", "label": "maybe"},
                {"question_id": "same", "step_text": "good", "label": "valid"},
                {"question_id": "same", "step_text": "bad", "label": "error"},
                {"question_id": "other_good", "step_text": "fallback good", "is_correct": True},
                {"question_id": "other_bad", "step_text": "fallback bad", "is_correct": False},
            ]
        },
    )
    _write_json(ckpt_dir / "q0000.json", ["not", "a", "dict"])
    _write_checkpoint(ckpt_dir / "q0001.json", gold="A", answers=[None])
    _write_checkpoint(ckpt_dir / "q0002.json", gold="A", answers=["A", "B", None])

    limited_fover = mod.load_fover_pairs([tmp_path / "missing.json", fover_path], max_pairs=1)
    unlimited_fover = mod.load_fover_pairs([fover_path], max_pairs=None)
    limited_musr = mod.load_musr_training_pairs(ckpt_dir, limit=None, max_pairs=1)
    rows = mod.load_cached_musr_rows(ckpt_dir, limit=None, min_questions=1)

    assert mod._resolve_snapshot(snapshot.parents[1]) == snapshot
    assert len(limited_fover) == 1
    assert len(unlimited_fover) >= 1
    assert len(limited_musr) == 1
    assert rows[0]["row_id"] == "q0002"
    assert mod._candidate_text({"text": "from text"}) == "from text"
    assert mod._candidate_text({"answer": "from answer"}) == "from answer"
    assert mod._candidate_text({}) == ""
    assert mod._rows_from_json_payload("bad payload") == []
    with pytest.raises(RuntimeError, match="only 1 cached MuSR rows"):
        mod.load_cached_musr_rows(ckpt_dir, limit=None, min_questions=2)


def test_scenario_verify_5003_blocked_artifact_names_missing_resource(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5003: missing preconditions produce an honest blocked artifact."""

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH

    artifact = mod.run(
        root=tmp_path,
        artifact_path=artifact_path,
        hf_cache_root=tmp_path / "empty_hf",
        cuda_available=lambda: False,
        audit_runner=_audit_clean,
        summary_runner=lambda _path: 0,
        write=True,
    )
    loaded = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert loaded == artifact
    assert artifact["honest_verdict"] == "blocked_trainable_qwen_base"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["headroom_present"] is False
    assert artifact["trained_scorer_accuracy"] is None
    assert artifact["preconditions_checked"][0]["available"] is False
    assert mod.artifact_schema_errors(artifact) == []
    assert mod.run(
        root=tmp_path,
        hf_cache_root=tmp_path / "empty_hf",
        cuda_available=lambda: False,
        write=False,
    )["honest_verdict"] == "blocked_trainable_qwen_base"


def test_scenario_verify_5003_complete_run_scores_cached_candidates_oracle_distinct(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5003: complete run evaluates via guarded cached MuSR rows."""

    fover_path = tmp_path / "data" / "fover_train.json"
    ckpt_dir = tmp_path / "results" / "distributional_energy_verifier_musr_checkpoints"
    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    _write_base_cache(tmp_path)
    _write_json(fover_path, [{"question_id": "x", "step_text": "good", "label": "correct"}])
    _write_checkpoint(ckpt_dir / "q0000.json", gold="A", answers=["A", "B"])
    _write_checkpoint(ckpt_dir / "q0001.json", gold="B", answers=["A", "B"])

    def fake_train(
        pairs: list[mod.TrainingPair],
        *,
        config: mod.TrainingConfig,
        checkpoint_dir: Path,
        skeleton_path: Path,
    ) -> mod.TrainedScorer:
        skeleton = json.loads(skeleton_path.read_text(encoding="utf-8"))
        assert skeleton["deliverable_stage"] == "pretrain_skeleton"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

        def scorer(candidate: Any) -> float:
            cid = str(candidate.get("candidate_id"))
            return 0.0 if cid.endswith(("q0000/cached-0", "q0001/cached-1")) else 1.0

        return mod.TrainedScorer(
            scorer=scorer,
            train_loss=0.125,
            n_pairs=len(pairs),
            checkpoint_path=checkpoint_dir,
            model_specs={"base_model": config.base_model_id, "adapter": "fake-lora"},
        )

    artifact = mod.run(
        root=tmp_path,
        artifact_path=artifact_path,
        hf_cache_root=tmp_path / "hf",
        cuda_available=lambda: True,
        trainer=fake_train,
        audit_runner=_audit_clean,
        summary_runner=lambda _path: 0,
        min_questions=2,
        bootstrap_samples=32,
        write=True,
    )

    assert artifact["honest_verdict"].startswith("complete_lora_ebm_no_win_musr_")
    assert artifact["trained_scorer_accuracy"] == 1.0
    assert artifact["tuned_sc_accuracy"] == 0.5
    assert artifact["delta_vs_tuned_sc"] == 0.5
    assert artifact["oracle_at_k"] == 1.0
    assert artifact["headroom_present"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["oracle_distinctness_enforced"] is True
    assert artifact["adversarial_verify_clean"] is True
    assert artifact["summarize_artifact_exit_code"] == 0
    assert artifact["checkpoint_path"].endswith("experiment_5003_lora_ebm_scorer_musr_adapter")
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_verify_5003_oracle_leakage_blocks_complete_claim(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5003: scorers that read gold fail closed."""

    fover_path = tmp_path / "data" / "fover_train.json"
    ckpt_dir = tmp_path / "results" / "distributional_energy_verifier_musr_checkpoints"
    _write_base_cache(tmp_path)
    _write_json(fover_path, [{"question_id": "x", "step_text": "good", "label": "correct"}])
    _write_checkpoint(ckpt_dir / "q0000.json", gold="A", answers=["A", "B"])

    def leaky_train(
        pairs: list[mod.TrainingPair],
        *,
        config: mod.TrainingConfig,
        checkpoint_dir: Path,
        skeleton_path: Path,
    ) -> mod.TrainedScorer:
        def scorer(candidate: Any) -> float:
            return 0.0 if candidate.get("gold") else 1.0

        return mod.TrainedScorer(
            scorer=scorer,
            train_loss=0.0,
            n_pairs=len(pairs),
            checkpoint_path=checkpoint_dir,
            model_specs={"base_model": config.base_model_id},
        )

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        hf_cache_root=tmp_path / "hf",
        cuda_available=lambda: True,
        trainer=leaky_train,
        audit_runner=_audit_clean,
        summary_runner=lambda _path: 0,
        min_questions=1,
        write=True,
    )

    assert artifact["honest_verdict"] == "blocked_oracle_distinctness_violation"
    assert artifact["oracle_distinctness_enforced"] is False
    assert artifact["trained_scorer_accuracy"] is None
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_verify_5003_training_or_eval_error_blocks_and_writes(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5003: malformed training inputs block without metric fabrication."""

    fover_path = tmp_path / "data" / "fover_train.json"
    ckpt_dir = tmp_path / "results" / "distributional_energy_verifier_musr_checkpoints"
    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    _write_base_cache(tmp_path)
    _write_json(fover_path, [{"question_id": "x", "step_text": "only good", "label": "correct"}])
    _write_checkpoint(ckpt_dir / "q0000.json", gold="A", answers=["A"])

    artifact = mod.run(
        root=tmp_path,
        artifact_path=artifact_path,
        hf_cache_root=tmp_path / "hf",
        cuda_available=lambda: True,
        min_questions=1,
        write=True,
    )

    assert artifact["honest_verdict"] == "blocked_training_or_eval_error"
    assert "no_contrastive_pairs" in artifact["blocked_error"]
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert mod.artifact_schema_errors(artifact) == []


def test_req_verify_5003_complete_verdict_and_audit_helper_branches(tmp_path: Path) -> None:
    """REQ-VERIFY-5003: verdict gates and audit report variants stay explicit."""

    config = mod.TrainingConfig(
        base_model_id="Qwen/Qwen3.5-4B",
        base_cache_path="/tmp/base",
    )
    trained = mod.TrainedScorer(
        scorer=lambda _candidate: 0.0,
        train_loss=0.1,
        n_pairs=1,
        checkpoint_path=tmp_path / "adapter",
        model_specs={"base_model": "Qwen/Qwen3.5-4B"},
    )
    pair = mod.TrainingPair("p", "good", "bad", "fover")
    base_eval = {
        "verifier": {"accuracy": 0.8, "predictions": []},
        "tuned_self_consistency": {"accuracy": 0.6, "config": {"k": 1}, "predictions": []},
        "verifier_minus_tuned_sc_delta": 0.2,
        "verifier_minus_tuned_sc_ci95": [0.1, 0.3],
        "mcnemar_p": 0.01,
        "headroom_present": True,
        "n_rows": 200,
        "oracle_at_k": 0.9,
    }

    success = mod.build_complete_artifact(
        evaluation=base_eval,
        trained=trained,
        config=config,
        pairs=[pair],
        preconditions_checked=[],
        duration_s=61.0,
    )
    gated_null = mod.build_complete_artifact(
        evaluation={**base_eval, "mcnemar_p": 0.5},
        trained=trained,
        config=config,
        pairs=[pair],
        preconditions_checked=[],
        duration_s=61.0,
    )

    assert success["honest_verdict"].startswith("success_lora_ebm_beats_sc_musr_")
    assert gated_null["honest_verdict"].endswith("mcnemar_or_headroom_gate")
    assert mod._compact_adversarial_flags({"reports": [{"flags": [{"kind": "WARN"}, "bad"]}]}) == [
        {"kind": "WARN"}
    ]
    assert mod._audit_is_clean({"flagged_count": 0}) is True
    assert mod._audit_is_clean({"flags": [{"kind": "WARN"}]}) is False


def test_req_verify_5003_schema_rejects_oracle_and_bad_metrics(tmp_path: Path) -> None:
    """REQ-VERIFY-5003: schema guard rejects circular or inconsistent artifacts."""

    artifact = mod.build_blocked_artifact(
        missing_resource="cuda",
        preconditions_checked=[
            mod.PreconditionCheck("cuda", False, "torch.cuda.is_available=false").as_dict()
        ],
        duration_s=0.1,
    )

    assert mod.artifact_schema_errors(artifact) == []
    assert "verifier_is_oracle" in mod.artifact_schema_errors(
        {**artifact, "verifier_is_oracle": True}
    )
    assert "paired_ci95" in mod.artifact_schema_errors({**artifact, "paired_ci95": [0.0]})
    assert "spec_refs" in mod.artifact_schema_errors({**artifact, "spec_refs": ["REQ-VERIFY-5003"]})
    assert "trained_scorer_accuracy" in mod.artifact_schema_errors(
        {**artifact, "trained_scorer_accuracy": 2.0}
    )
    assert "headroom_present" in mod.artifact_schema_errors({**artifact, "headroom_present": "no"})
    assert "delta_vs_tuned_sc" in mod.artifact_schema_errors(
        {**artifact, "delta_vs_tuned_sc": "0.1"}
    )
    assert "mcnemar_p" in mod.artifact_schema_errors({**artifact, "mcnemar_p": 2.0})
    assert "preconditions_checked" in mod.artifact_schema_errors(
        {**artifact, "preconditions_checked": {}}
    )
    assert "model_specs" in mod.artifact_schema_errors({**artifact, "model_specs": []})
    assert "field_principles" in mod.artifact_schema_errors({**artifact, "field_principles": {}})
    assert "honest_verdict" in mod.artifact_schema_errors({**artifact, "honest_verdict": "maybe"})
    missing = dict(artifact)
    missing.pop("duration_s")
    assert "duration_s" in mod.artifact_schema_errors(missing)
