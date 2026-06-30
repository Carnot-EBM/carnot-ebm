"""Tests for Exp 5017 trained LoRA-EBM MuSR scorer v2.

Spec refs: REQ-VERIFY-5017, SCENARIO-VERIFY-5017.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5017_lora_ebm_scorer_musr_v2 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


class Clock:
    """Deterministic clock so tests can exercise the >60s training gate."""

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


def _write_base_cache(root: Path) -> None:
    cache_dir = root / "hf" / "models--Qwen--Qwen3.5-1.7B"
    snapshot = cache_dir / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (cache_dir / "refs").mkdir()
    (cache_dir / "refs" / "main").write_text("abc", encoding="utf-8")
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors.index.json").write_text("{}", encoding="utf-8")


def _write_checkpoint(path: Path, *, gold: str, answers: list[str]) -> None:
    _write_json(path, {"q": int(path.stem[1:]), "gold": gold, "answers": answers})


def _shared_cache_row(row_id: str, *, gold: str, answers: list[str]) -> dict[str, Any]:
    return {
        "schema": "carnot.shared_logprob_candidate_cache.row.v1",
        "row_id": row_id,
        "corpus": "MuSR/murder_mysteries",
        "gold": gold,
        "candidates": [
            {
                "candidate_id": f"{row_id}/shared-{index}",
                "answer": answer,
                "reasoning": f"Reasoning for {answer}",
                "cache_index": index,
                "temperature": 0.7,
                "token_logprobs": [-0.1],
                "uprm_marker_logprobs": [{"+": -0.2, "-": -1.3}],
            }
            for index, answer in enumerate(answers)
        ],
    }


def _write_shared_cache(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_fover(path: Path) -> None:
    _write_json(
        path,
        [
            {"question_id": "same", "step_text": "valid reasoning", "label": "correct"},
            {"question_id": "same", "step_text": "bad reasoning", "label": "incorrect"},
        ],
    )


def _audit_clean(_path: Path) -> dict[str, Any]:
    return {"flag_count": 0, "flags": []}


def test_req_verify_5017_spec_declares_trained_lora_ebm_contract() -> None:
    """REQ-VERIFY-5017: OpenSpec anchors the v2 trained-scorer gate."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-5017",
        "SCENARIO-VERIFY-5017",
        "experiment_5017_lora_ebm_scorer_musr_v2.py",
        "results/experiment_5017_lora_ebm_scorer_musr_v2.json",
        "Qwen/Qwen3.5-1.7B",
        "blocked_lora_ebm_train_did_not_run",
        "genuine_tuned_sc_accuracy",
        "success_lora_ebm_beats_sc_musr_",
        "complete_lora_ebm_no_win_musr_",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_verify_5017_builds_fover_and_cached_musr_pairs(tmp_path: Path) -> None:
    """REQ-VERIFY-5017: FoVer and gold-labeled cached MuSR rows form pairs."""

    fover_path = tmp_path / "data" / "fover_train.json"
    shared_path = tmp_path / "results" / "experiment_5016_shared_logprob_candidate_cache_musr.jsonl"
    _write_fover(fover_path)
    bad_short = _shared_cache_row("bad", gold="A", answers=["A"])
    good = _shared_cache_row("musr:0", gold="A", answers=["A", "B", "C"])
    shared_path.parent.mkdir(parents=True, exist_ok=True)
    shared_path.write_text(
        "\n"
        + json.dumps(["not", "a", "dict"])
        + "\n"
        + json.dumps(bad_short)
        + "\n"
        + json.dumps(good)
        + "\n",
        encoding="utf-8",
    )

    rows = mod.load_shared_candidate_cache_rows(
        shared_path, min_questions=1, k_candidates=2, limit=1
    )
    pairs = mod.build_contrastive_corpus([fover_path], rows, max_pairs=8)
    limited_pairs = mod.load_musr_training_pairs_from_rows(rows, max_pairs=1)
    source = mod.select_candidate_source(tmp_path, min_questions=1, k_candidates=2)

    assert len(rows) == 1
    assert source.name == "exp5016_shared_logprob_candidate_cache"
    assert {pair.source for pair in pairs} == {"fover", "musr_cached_gold_labeled"}
    assert any(pair.good_text == "valid reasoning" for pair in pairs)
    assert any(pair.good_text.endswith("Final answer: A") for pair in pairs)
    assert len(limited_pairs) == 1
    assert all(pair.good_text and pair.bad_text for pair in pairs)
    assert mod._candidate_text("x", {"answer": "A"}) == "MuSR cached candidate x\nFinal answer: A"
    with pytest.raises(RuntimeError, match="only 1 shared MuSR rows"):
        mod.load_shared_candidate_cache_rows(shared_path, min_questions=2, k_candidates=2)


def test_req_verify_5017_helper_edge_cases_are_explicit(tmp_path: Path) -> None:
    """REQ-VERIFY-5017: helper branches stay deterministic and schema-visible."""

    failed_path, failed_check = mod.resolve_or_download_base_model(
        tmp_path / "hf",
        allow_download=True,
        downloader=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    checkpoint_dir = tmp_path / "adapter"
    epoch_dir = checkpoint_dir / "epoch_1"
    epoch_dir.mkdir(parents=True)
    (epoch_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    _write_json(epoch_dir / "train_metrics.json", {"train_loss": 0.1, "n_pairs": 1})
    bad_b1 = tmp_path / mod.B1_BASELINE_RELATIVE_PATH
    _write_json(bad_b1, ["bad"])
    missing_b1_root = tmp_path / "missing_b1_root"
    config = mod.TrainingConfig(base_model_id=mod.BASE_MODEL_ID, base_cache_path="/tmp/base")
    trained = mod.TrainedScorer(
        scorer=lambda _candidate: 0.0,
        train_loss=0.1,
        n_pairs=1,
        checkpoint_path=epoch_dir,
        model_specs={"base_model": mod.BASE_MODEL_ID},
        epoch_checkpoints=[epoch_dir],
    )
    source = mod.CandidateSource(
        name="fixture",
        path=tmp_path / "cache.jsonl",
        rows=[],
        check=mod.PreconditionCheck("cached_musr_candidates", True, "fixture"),
    )
    pair = mod.TrainingPair("p", "good", "bad", "fover")
    base_eval = {
        "verifier": {"accuracy": 0.9, "predictions": []},
        "tuned_self_consistency": {"accuracy": 0.6, "config": {"k": 3}, "predictions": []},
        "verifier_minus_tuned_sc_delta": 0.3,
        "verifier_minus_tuned_sc_ci95": [0.1, 0.5],
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
        candidate_source=source,
        root=tmp_path,
        duration_s=61.0,
    )
    non_ci_null = mod.build_complete_artifact(
        evaluation={**base_eval, "mcnemar_p": 0.5},
        trained=trained,
        config=config,
        pairs=[pair],
        preconditions_checked=[],
        candidate_source=source,
        root=tmp_path,
        duration_s=61.0,
    )
    failed_gate = mod.build_complete_artifact(
        evaluation=base_eval,
        trained=trained,
        config=config,
        pairs=[pair],
        preconditions_checked=[],
        candidate_source=source,
        root=tmp_path,
        duration_s=60.0,
    )

    assert failed_path is None
    assert failed_check.available is False
    assert mod._latest_epoch_checkpoint(checkpoint_dir) == epoch_dir
    assert mod._read_b1_baseline(tmp_path)["available"] is False
    assert mod._read_b1_baseline(missing_b1_root)["available"] is False
    assert success["honest_verdict"].startswith("success_lora_ebm_beats_sc_musr_")
    assert non_ci_null["honest_verdict"].endswith("mcnemar_or_headroom_gate")
    assert failed_gate["honest_verdict"] == "blocked_lora_ebm_train_did_not_run"
    assert mod._compact_adversarial_flags({"reports": [{"flags": [{"kind": "WARN"}]}]}) == [
        {"kind": "WARN"}
    ]
    assert mod._audit_is_clean({"flagged_count": 0}) is True
    assert mod._audit_is_clean({"flags": [{"kind": "WARN"}]}) is False


def test_scenario_verify_5017_missing_base_writes_blocked_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5017: missing base cache blocks without fake training metrics."""

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH

    artifact = mod.run(
        root=tmp_path,
        artifact_path=artifact_path,
        hf_cache_root=tmp_path / "empty_hf",
        cuda_available=lambda: True,
        audit_runner=_audit_clean,
        summary_runner=lambda _path: 0,
        min_questions=1,
        allow_download=False,
        write=True,
    )

    assert artifact["honest_verdict"] == "blocked_trainable_qwen_base"
    assert artifact["scorer_trained"] is False
    assert artifact["train_loss"] is None
    assert artifact["n_pairs"] == 0
    assert artifact["trained_scorer_accuracy"] is None
    assert artifact["genuine_tuned_sc_accuracy"] is None
    assert artifact["verifier_is_oracle"] is False
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_verify_5017_complete_run_uses_genuine_sc_and_guarded_scoring(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5017: trained run evaluates through the shared harness."""

    _write_base_cache(tmp_path)
    _write_fover(tmp_path / "data" / "fover_train.json")
    _write_json(
        tmp_path / mod.B1_BASELINE_RELATIVE_PATH,
        {
            "honest_verdict": "success_genuine_sc_baseline_fixed_degeneracy_guard_shipped",
            "genuine_tuned_sc_accuracy": 0.5,
            "oracle_at_k": 1.0,
            "n_questions": 2,
        },
    )
    ckpt = tmp_path / "results" / "distributional_energy_verifier_musr_checkpoints"
    _write_checkpoint(ckpt / "q0000.json", gold="A", answers=["A", "B", "B"])
    _write_checkpoint(ckpt / "q0001.json", gold="B", answers=["A", "B", "B"])

    def fake_train(
        pairs: list[mod.TrainingPair],
        *,
        config: mod.TrainingConfig,
        checkpoint_dir: Path,
        skeleton_path: Path,
    ) -> mod.TrainedScorer:
        skeleton = json.loads(skeleton_path.read_text(encoding="utf-8"))
        assert skeleton["deliverable_stage"] == "pretrain_skeleton"
        epoch_dir = checkpoint_dir / "epoch_1"
        epoch_dir.mkdir(parents=True)
        (epoch_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

        def scorer(candidate: Any) -> float:
            cid = str(candidate.get("candidate_id"))
            return 0.0 if cid.endswith(("q0000/cached-0", "q0001/cached-1")) else 1.0

        return mod.TrainedScorer(
            scorer=scorer,
            train_loss=0.25,
            n_pairs=len(pairs),
            checkpoint_path=epoch_dir,
            model_specs={"base_model": config.base_model_id, "adapter": "fake-lora"},
            epoch_checkpoints=[epoch_dir],
        )

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        hf_cache_root=tmp_path / "hf",
        cuda_available=lambda: True,
        trainer=fake_train,
        audit_runner=_audit_clean,
        summary_runner=lambda _path: 0,
        min_questions=2,
        bootstrap_samples=32,
        now=Clock([0.0, 0.0, 61.5, 61.5, 61.5]),
        allow_download=False,
        write=True,
    )

    assert artifact["honest_verdict"].startswith("complete_lora_ebm_no_win_musr_")
    assert artifact["scorer_trained"] is True
    assert artifact["trained_scorer_accuracy"] == 1.0
    assert artifact["genuine_tuned_sc_accuracy"] == 0.5
    assert artifact["delta_vs_tuned_sc"] == 0.5
    assert artifact["oracle_at_k"] == 1.0
    assert artifact["headroom_present"] is True
    assert artifact["oracle_distinctness_enforced"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["adversarial_verify_clean"] is True
    assert artifact["summarize_artifact_exit_code"] == 0
    assert artifact["epoch_checkpoints"][0].endswith("epoch_1")
    assert artifact["model_specs"]["b1_genuine_sc_baseline_reference"]["available"] is True
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_verify_5017_oracle_leakage_and_short_training_fail_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5017: leakage or skeleton-fast runs cannot become nulls."""

    _write_base_cache(tmp_path)
    _write_fover(tmp_path / "data" / "fover_train.json")
    ckpt = tmp_path / "results" / "distributional_energy_verifier_musr_checkpoints"
    _write_checkpoint(ckpt / "q0000.json", gold="A", answers=["A", "B"])

    def leaky_train(
        pairs: list[mod.TrainingPair],
        *,
        config: mod.TrainingConfig,
        checkpoint_dir: Path,
        skeleton_path: Path,
    ) -> mod.TrainedScorer:
        del pairs, config, checkpoint_dir, skeleton_path

        def scorer(candidate: Any) -> float:
            return 0.0 if candidate.get("gold") else 1.0

        return mod.TrainedScorer(
            scorer=scorer,
            train_loss=0.1,
            n_pairs=1,
            checkpoint_path=tmp_path / "adapter",
            model_specs={"base_model": mod.BASE_MODEL_ID},
            epoch_checkpoints=[],
        )

    leaky = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        hf_cache_root=tmp_path / "hf",
        cuda_available=lambda: True,
        trainer=leaky_train,
        audit_runner=_audit_clean,
        summary_runner=lambda _path: 0,
        min_questions=1,
        now=Clock([0.0, 0.0, 61.5, 61.5]),
        allow_download=False,
        write=True,
    )

    assert leaky["honest_verdict"] == "blocked_oracle_distinctness_violation"
    assert leaky["scorer_trained"] is False
    assert leaky["oracle_distinctness_enforced"] is False
    assert mod.artifact_schema_errors(leaky) == []

    def short_train(
        pairs: list[mod.TrainingPair],
        *,
        config: mod.TrainingConfig,
        checkpoint_dir: Path,
        skeleton_path: Path,
    ) -> mod.TrainedScorer:
        del config, skeleton_path
        checkpoint_dir.mkdir(parents=True)
        return mod.TrainedScorer(
            scorer=lambda _candidate: 0.0,
            train_loss=0.1,
            n_pairs=len(pairs),
            checkpoint_path=checkpoint_dir,
            model_specs={"base_model": mod.BASE_MODEL_ID},
            epoch_checkpoints=[checkpoint_dir],
        )

    short = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        hf_cache_root=tmp_path / "hf",
        cuda_available=lambda: True,
        trainer=short_train,
        audit_runner=_audit_clean,
        summary_runner=lambda _path: 0,
        min_questions=1,
        now=Clock([0.0, 0.0, 12.0, 12.0]),
        allow_download=False,
        write=True,
    )

    assert short["honest_verdict"] == "blocked_lora_ebm_train_did_not_run"
    assert short["scorer_trained"] is False
    assert short["trained_scorer_accuracy"] is None
    assert short["train_loss"] is None
    assert mod.artifact_schema_errors(short) == []


def test_scenario_verify_5017_no_pairs_is_failed_training(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5017: zero pairs is failed training, not a null result."""

    _write_base_cache(tmp_path)
    _write_json(
        tmp_path / "data" / "fover_train.json",
        [{"question_id": "x", "step_text": "only valid", "label": "correct"}],
    )
    ckpt = tmp_path / "results" / "distributional_energy_verifier_musr_checkpoints"
    _write_checkpoint(ckpt / "q0000.json", gold="A", answers=["A"])

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        hf_cache_root=tmp_path / "hf",
        cuda_available=lambda: True,
        audit_runner=_audit_clean,
        summary_runner=lambda _path: 0,
        min_questions=1,
        now=Clock([0.0, 0.0, 1.0, 1.0]),
        allow_download=False,
        write=True,
    )

    assert artifact["honest_verdict"] == "blocked_lora_ebm_train_did_not_run"
    assert "no_contrastive_pairs" in artifact["blocked_error"]
    assert artifact["scorer_trained"] is False
    assert mod.artifact_schema_errors(artifact) == []


def test_req_verify_5017_schema_rejects_bad_terminal_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-5017: artifact schema enforces the anti-skeleton fields."""

    artifact = mod.build_blocked_artifact(
        missing_resource="cuda",
        preconditions_checked=[
            mod.PreconditionCheck("cuda", False, "torch.cuda.is_available=false").as_dict()
        ],
        duration_s=0.1,
    )

    assert mod.artifact_schema_errors(artifact) == []
    assert "scorer_trained" in mod.artifact_schema_errors({**artifact, "scorer_trained": "no"})
    assert "verifier_is_oracle" in mod.artifact_schema_errors(
        {**artifact, "verifier_is_oracle": True}
    )
    assert "paired_ci95" in mod.artifact_schema_errors({**artifact, "paired_ci95": [0.0]})
    assert "spec_refs" in mod.artifact_schema_errors({**artifact, "spec_refs": []})
    assert "field_principles" in mod.artifact_schema_errors({**artifact, "field_principles": {}})
    assert "trained_scorer_accuracy" in mod.artifact_schema_errors(
        {**artifact, "trained_scorer_accuracy": 2.0}
    )
    assert "delta_vs_tuned_sc" in mod.artifact_schema_errors(
        {**artifact, "delta_vs_tuned_sc": "bad"}
    )
    assert "mcnemar_p" in mod.artifact_schema_errors({**artifact, "mcnemar_p": 2.0})
    assert "preconditions_checked" in mod.artifact_schema_errors(
        {**artifact, "preconditions_checked": {}}
    )
    assert "model_specs" in mod.artifact_schema_errors({**artifact, "model_specs": []})
    assert "honest_verdict" in mod.artifact_schema_errors({**artifact, "honest_verdict": "bad"})
    assert "train_loss" in mod.artifact_schema_errors(
        {**artifact, "scorer_trained": True, "duration_s": 61.0, "n_pairs": 1}
    )
    assert "n_pairs" in mod.artifact_schema_errors(
        {**artifact, "scorer_trained": True, "duration_s": 61.0, "train_loss": 0.1}
    )
    assert "duration_s" in mod.artifact_schema_errors(
        {**artifact, "scorer_trained": True, "train_loss": 0.1, "n_pairs": 1}
    )
    assert "scorer_trained" in mod.artifact_schema_errors(
        {**artifact, "honest_verdict": "complete_lora_ebm_no_win_musr_plus_0p000_ci_incl_0"}
    )
    missing = dict(artifact)
    missing.pop("scorer_trained")
    assert "scorer_trained" in mod.artifact_schema_errors(missing)
