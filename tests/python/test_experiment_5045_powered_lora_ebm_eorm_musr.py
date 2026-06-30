"""Tests for Exp 5045 powered LoRA-EBM/EORM MuSR rerun.

Spec refs: REQ-VERIFY-5045, SCENARIO-VERIFY-5045.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5045_powered_lora_ebm_eorm_musr as mod


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


def _write_checkpoint(path: Path, *, gold: str = "GOLD") -> None:
    _write_json(path, {"q": int(path.stem[1:]), "gold": gold, "answers": ["BAD", "BAD", gold]})


def _setup_root(tmp_path: Path, *, n_questions: int = 8) -> Path:
    root = tmp_path / "root"
    ckpt_dir = root / mod.MUSR_CHECKPOINT_RELATIVE_DIR
    for i in range(n_questions):
        _write_checkpoint(ckpt_dir / f"q{i:04d}.json")
    _write_json(
        root / mod.PRIOR_D1_ARTIFACT_RELATIVE_PATH,
        {
            "scorer_trained": True,
            "duration_s": 807.36,
            "checkpoint_path": "/prior/epoch_1",
            "train_loss": 0.237861,
            "n_pairs": 880,
        },
    )
    return root


def _blocked_preflight() -> dict[str, Any]:
    specs = {
        "flagship_moe": {
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "preferred_quant": "Q4_K_M",
            "resolved_path": "/models/qwen.gguf",
        },
        "flagship_dense": {
            "hf_id": "unsloth/gemma-4-31B-it-GGUF",
            "preferred_quant": "Q4_K_M",
            "resolved_path": "missing",
        },
        "middle_moe": {
            "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
            "preferred_quant": "Q4_K_M",
            "resolved_path": "missing",
        },
    }
    return {
        "honest_verdict": "blocked_judge_server",
        "model_specs": specs,
        "usable_sota_models": [
            {
                "role": "flagship_moe",
                "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                "model_path": "/models/qwen.gguf",
            }
        ],
        "sota_models_ready": True,
        "sota_judge_ready": False,
        "top_logprob_or_confidence_ready": False,
        "endpoint_summary": {
            "completion_ready": False,
            "top_logprob_ready": False,
            "confidence_ready": False,
            "probes": [{"endpoint": "http://127.0.0.1:8080"}],
        },
    }


def _ready_preflight() -> dict[str, Any]:
    payload = _blocked_preflight()
    payload["honest_verdict"] = "complete_sota_gguf_judge_preflight_ready"
    payload["sota_judge_ready"] = True
    payload["top_logprob_or_confidence_ready"] = True
    payload["endpoint_summary"] = {
        "completion_ready": True,
        "top_logprob_ready": True,
        "confidence_ready": False,
        "probes": [{"endpoint": "http://127.0.0.1:8080"}],
    }
    return payload


def _trainer(pairs: Any, *, base: Any, out_dir: Any, config: Any) -> dict[str, Any]:
    return {
        "train_loss": 0.123,
        "n_pairs": len(list(pairs)),
        "base_used": base[0],
        "checkpoint_dir": str(Path(out_dir) / "epoch_1"),
        "resumed": True,
        "model_specs": {"base_model": base[0], "adapter": "LoRA"},
    }


def _gold_low_score_fn(_checkpoint: Any, texts: list[str]) -> list[float]:
    return [0.0 if "Candidate answer: GOLD" in text else 1.0 for text in texts]


def test_req_verify_5045_spec_declares_contract() -> None:
    """REQ-VERIFY-5045 and SCENARIO-VERIFY-5045 are anchored in OpenSpec."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-5045",
        "SCENARIO-VERIFY-5045",
        "experiment_5045_powered_lora_ebm_eorm_musr.py",
        "results/experiment_5045_powered_lora_ebm_eorm_musr.json",
        "energy_margin_auc",
        "blocked_lora_ebm_train_did_not_run",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    ):
        assert marker in spec


@pytest.mark.parametrize(
    ("labels", "scores", "expected"),
    [
        ([1, 0], [2.0, 1.0], 1.0),
        ([1, 0], [1.0, 2.0], 0.0),
        ([1, 1], [1.0, 2.0], 0.5),
        ([], [], 0.5),
    ],
)
def test_energy_margin_auc_rank_statistic(
    labels: list[int], scores: list[float], expected: float
) -> None:
    """REQ-VERIFY-5045: margin AUROC is finite and handles degenerate labels."""

    assert mod.energy_margin_auc(labels, scores) == pytest.approx(expected)


def test_helper_edge_cases(tmp_path: Path) -> None:
    """REQ-VERIFY-5045: malformed inputs are bounded, not promoted."""

    assert mod._read_json(tmp_path / "missing.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{bad", encoding="utf-8")
    assert mod._read_json(bad) is None
    assert mod.energy_margin_auc([1, 0], [1.0, 1.0]) == 0.5
    assert mod._median([]) == 0.0
    assert mod._median([3.0]) == 3.0
    assert mod._candidate_refresh({})["blocked_reason"] == "no_mandated_sota_model"
    ok, detail = mod._training_gate(
        train_result={"train_loss": "not-a-number", "n_pairs": 4},
        checkpoint_path="/x",
        duration_evidence_s=120.0,
    )
    assert ok is False and "trained_gate_failed" in detail


def test_margin_telemetry_selects_min_energy_and_threshold_fallback() -> None:
    """SCENARIO-VERIFY-5045: min-energy and margin-aware selectors are both reported."""

    rows = [
        {
            "gold": "GOLD",
            "candidates": [
                {"candidate_id": "a", "answer": "GOLD"},
                {"candidate_id": "b", "answer": "BAD"},
            ],
        },
        {
            "gold": "GOLD",
            "candidates": [
                {"candidate_id": "c", "answer": "BAD"},
                {"candidate_id": "d", "answer": "GOLD"},
            ],
        },
    ]
    telemetry = mod.compute_margin_telemetry(
        rows,
        {"a": 0.0, "b": 2.0, "c": 0.0, "d": 0.1},
        tuned_sc_predictions=["BAD", "GOLD"],
    )
    assert telemetry["min_energy_predictions"] == ["GOLD", "BAD"]
    assert telemetry["min_energy_correct"] == [1, 0]
    assert telemetry["margin_aware_selection"]["accuracy"] == 1.0
    assert telemetry["uncertainty_telemetry"]["mean_margin"] == pytest.approx(1.05)
    assert telemetry["energy_margin_auc"] == 1.0


def test_run_writes_blocked_candidate_refresh_with_cached_powered_panel(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5045: closed SOTA refresh gate blocks headline promotion."""

    root = _setup_root(tmp_path, n_questions=8)
    artifact = mod.run(
        root=root,
        artifact_path=tmp_path / "out.json",
        trainer=_trainer,
        score_fn=_gold_low_score_fn,
        base_resolver=lambda: ("Qwen/Qwen3.5-2B", "/fake/base"),
        preflight_loader=lambda _root: _blocked_preflight(),
        narratives_loader=lambda limit: [{"question": "Q?", "context": "C"} for _ in range(limit)],
        now=Clock([0.0, 5.0]),
        min_questions=4,
        desired_questions=400,
        bootstrap_samples=200,
    )

    assert artifact["honest_verdict"] == "blocked_sota_candidate_refresh_unavailable"
    assert artifact["powered_scorer_available"] is True
    assert artifact["scorer_trained"] is True
    assert artifact["n_questions"] == 8
    assert artifact["n_candidate_rows"] == 24
    assert artifact["candidate_expansion"]["requested_questions"] == 400
    assert artifact["candidate_expansion"]["cap_explained"] is True
    assert artifact["candidate_refresh"]["blocked_reason"] == "sota_judge_ready_false"
    assert artifact["genuine_tuned_sc_accuracy"] == 0.0
    assert artifact["powered_lora_ebm_accuracy"] == 1.0
    assert artifact["delta_vs_tuned_sc"] == 1.0
    assert artifact["verifier_is_oracle"] is False
    assert artifact["checkpoint_path"].endswith("epoch_1")
    assert mod.artifact_schema_errors(artifact) == []
    assert json.loads((tmp_path / "out.json").read_text(encoding="utf-8")) == artifact


def test_run_ready_preflight_without_refresher_blocks_backend(tmp_path: Path) -> None:
    """REQ-VERIFY-5045: a ready preflight still needs an explicit refresh backend."""

    root = _setup_root(tmp_path, n_questions=4)
    artifact = mod.run(
        root=root,
        artifact_path=tmp_path / "out.json",
        trainer=_trainer,
        score_fn=_gold_low_score_fn,
        base_resolver=lambda: ("Qwen/Qwen3.5-2B", "/fake/base"),
        preflight_loader=lambda _root: _ready_preflight(),
        narratives_loader=lambda limit: [{"question": "Q?", "context": "C"} for _ in range(limit)],
        now=Clock([0.0, 5.0]),
        min_questions=4,
        desired_questions=4,
        bootstrap_samples=100,
        write=False,
    )

    assert artifact["honest_verdict"] == "blocked_sota_candidate_refresh_backend_missing"
    assert artifact["candidate_refresh"]["blocked_reason"] == "refresh_backend_missing"
    assert mod.artifact_schema_errors(artifact) == []


def test_skeleton_train_result_is_blocked_not_null(tmp_path: Path) -> None:
    """REQ-VERIFY-5045: null loss/zero pairs rejects skeletons as blocked."""

    root = _setup_root(tmp_path, n_questions=4)

    def skeleton_trainer(pairs: Any, *, base: Any, out_dir: Any, config: Any) -> dict[str, Any]:
        return {"train_loss": None, "n_pairs": 0, "checkpoint_dir": ""}

    artifact = mod.run(
        root=root,
        artifact_path=tmp_path / "out.json",
        trainer=skeleton_trainer,
        score_fn=_gold_low_score_fn,
        base_resolver=lambda: ("Qwen/Qwen3.5-2B", "/fake/base"),
        preflight_loader=lambda _root: _blocked_preflight(),
        narratives_loader=lambda limit: [{"question": "Q?", "context": "C"} for _ in range(limit)],
        now=Clock([0.0, 5.0]),
        min_questions=4,
        desired_questions=4,
        write=False,
    )

    assert artifact["honest_verdict"] == "blocked_lora_ebm_train_did_not_run"
    assert artifact["powered_scorer_available"] is False
    assert artifact["scorer_trained"] is False
    assert artifact["powered_lora_ebm_accuracy"] is None
    assert "trained_gate_failed" in artifact["blocked_error"]
    assert mod.artifact_schema_errors(artifact) == []


def test_skeleton_train_result_write_branch(tmp_path: Path) -> None:
    """REQ-VERIFY-5045: blocked skeleton artifacts are still persisted."""

    root = _setup_root(tmp_path, n_questions=4)

    def skeleton_trainer(pairs: Any, *, base: Any, out_dir: Any, config: Any) -> dict[str, Any]:
        return {"train_loss": None, "n_pairs": 0, "checkpoint_dir": ""}

    artifact_path = tmp_path / "out.json"
    artifact = mod.run(
        root=root,
        artifact_path=artifact_path,
        trainer=skeleton_trainer,
        score_fn=_gold_low_score_fn,
        base_resolver=lambda: ("Qwen/Qwen3.5-2B", "/fake/base"),
        preflight_loader=lambda _root: _blocked_preflight(),
        narratives_loader=lambda limit: [{"question": "Q?", "context": "C"} for _ in range(limit)],
        now=Clock([0.0, 5.0]),
        min_questions=4,
        desired_questions=4,
    )
    assert artifact["honest_verdict"] == "blocked_lora_ebm_train_did_not_run"
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact


def test_insufficient_cached_rows_blocks_and_writes(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5045: fewer than the minimum rows blocks before training."""

    root = _setup_root(tmp_path, n_questions=1)
    artifact_path = tmp_path / "out.json"
    artifact = mod.run(
        root=root,
        artifact_path=artifact_path,
        trainer=_trainer,
        score_fn=_gold_low_score_fn,
        base_resolver=lambda: ("Qwen/Qwen3.5-2B", "/fake/base"),
        preflight_loader=lambda _root: _blocked_preflight(),
        narratives_loader=lambda limit: [{"question": "Q?", "context": "C"} for _ in range(limit)],
        now=Clock([0.0, 5.0]),
        min_questions=4,
        desired_questions=4,
    )
    assert artifact["honest_verdict"] == "blocked_cached_musr_candidates"
    assert artifact["n_questions"] == 1
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact


def test_schema_rejects_complete_skeleton(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5045: schema validation catches fake complete skeletons."""

    root = _setup_root(tmp_path, n_questions=4)
    artifact = mod.run(
        root=root,
        artifact_path=tmp_path / "out.json",
        trainer=_trainer,
        score_fn=_gold_low_score_fn,
        base_resolver=lambda: ("Qwen/Qwen3.5-2B", "/fake/base"),
        preflight_loader=lambda _root: _blocked_preflight(),
        narratives_loader=lambda limit: [{"question": "Q?", "context": "C"} for _ in range(limit)],
        now=Clock([0.0, 5.0]),
        min_questions=4,
        desired_questions=4,
        write=False,
    )

    broken = dict(
        artifact,
        honest_verdict="complete_lora_ebm_no_win_musr_plus_0p000_ci_incl_0",
        scorer_trained=False,
        powered_scorer_available=True,
        train_loss=None,
        n_pairs=0,
        checkpoint_path=None,
        duration_evidence_s=0.0,
    )
    errors = mod.artifact_schema_errors(broken)
    assert "scorer_trained" in errors
    assert "train_loss" in errors
    assert "n_pairs" in errors
    assert "checkpoint_path" in errors
    assert "duration_evidence_s" in errors


def test_schema_errors_cover_malformed_artifacts(tmp_path: Path) -> None:
    """REQ-VERIFY-5045: schema validation reports malformed required fields."""

    root = _setup_root(tmp_path, n_questions=4)
    artifact = mod.run(
        root=root,
        artifact_path=tmp_path / "out.json",
        trainer=_trainer,
        score_fn=_gold_low_score_fn,
        base_resolver=lambda: ("Qwen/Qwen3.5-2B", "/fake/base"),
        preflight_loader=lambda _root: _blocked_preflight(),
        narratives_loader=lambda limit: [{"question": "Q?", "context": "C"} for _ in range(limit)],
        now=Clock([0.0, 5.0]),
        min_questions=4,
        desired_questions=4,
        write=False,
    )

    cases = [
        (lambda a: (a.pop("honest_verdict"), a)[1], "honest_verdict"),
        (lambda a: dict(a, spec_refs=["WRONG"]), "spec_refs"),
        (lambda a: dict(a, model_specs=[]), "model_specs"),
        (lambda a: dict(a, scorer_trained="yes"), "scorer_trained"),
        (lambda a: dict(a, verifier_is_oracle=True), "verifier_is_oracle"),
        (lambda a: dict(a, honest_verdict="running"), "honest_verdict"),
        (lambda a: dict(a, powered_lora_ebm_accuracy=2.0), "powered_lora_ebm_accuracy"),
        (lambda a: dict(a, paired_ci95=[0.0]), "paired_ci95"),
        (lambda a: dict(a, delta_vs_tuned_sc="large"), "delta_vs_tuned_sc"),
    ]
    for mutate, expected in cases:
        assert expected in mod.artifact_schema_errors(mutate(dict(artifact)))
