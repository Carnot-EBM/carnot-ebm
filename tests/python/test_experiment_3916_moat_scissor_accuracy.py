"""Tests for Exp 3916 robust-GGUF moat scissor accuracy.

Spec refs: REQ-VERIFY-3916, SCENARIO-VERIFY-3916,
SCENARIO-VERIFY-3916-BLOCKED.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import moat_scissor_accuracy_3916 as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"


def _row(idx: int, label: str) -> dict[str, Any]:
    answer = 5 if label == "incorrect" else 4
    return {
        "corpus_item_id": f"{label}-{idx}",
        "question_id": f"{label}-{idx}",
        "step_text": f"Fixture step {idx}: 2 + 2 = {answer}.",
        "label": label,
        "source": "fover_fixture",
        "synthetic": False,
    }


def _score_for_row(index: int, row: dict[str, Any]) -> dict[str, Any]:
    is_error = row["label"] == "incorrect"
    score = 0.9 if is_error else 0.1
    return {
        "index": index,
        "corpus_item_id": row["corpus_item_id"],
        "question_id": row["question_id"],
        "label": row["label"],
        "synthetic": bool(row.get("synthetic")),
        "step_text_sha256": hashlib.sha256(row["step_text"].encode("utf-8")).hexdigest(),
        "carnot_ensemble_score": score,
        "carnot_rejects": is_error,
        "ensemble_threshold": 0.5,
        "per_verifier_scores": {"tier0r_curry_howard": score},
    }


def _write_exp3884_fixture(
    root: Path,
    *,
    n_per_class: int = 100,
    recorded_auroc: float = 0.9,
    flagged_adversarial: bool = False,
) -> list[dict[str, Any]]:
    data_dir = root / "data"
    results_dir = root / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = [_row(idx, "incorrect") for idx in range(n_per_class)]
    rows.extend(_row(idx, "correct") for idx in range(n_per_class))
    (data_dir / "in_distribution_error_corpus_v1.json").write_text(
        json.dumps({"schema": "fixture", "items": rows}),
        encoding="utf-8",
    )
    (results_dir / "experiment_3884_in_distribution_error_rich_corpus_scores.json").write_text(
        json.dumps({"schema": "fixture", "items": [_score_for_row(i, row) for i, row in enumerate(rows)]}),
        encoding="utf-8",
    )
    (results_dir / "experiment_3884_in_distribution_error_rich_corpus.json").write_text(
        json.dumps(
            {
                "honest_verdict": "complete: in_distribution_corpus_READY_fixture",
                "corpus_path": "data/in_distribution_error_corpus_v1.json",
                "per_item_ensemble_scores_path": (
                    "results/experiment_3884_in_distribution_error_rich_corpus_scores.json"
                ),
                "n_incorrect_steps": n_per_class,
                "n_total_items": len(rows),
                "carnot_ensemble_auroc_on_corpus": recorded_auroc,
                "corpus_sha256": "c" * 64,
                "per_item_ensemble_scores_sha256": "s" * 64,
                "flagged_adversarial": flagged_adversarial,
            }
        ),
        encoding="utf-8",
    )
    return rows


def _write_exp3894_fixture(
    root: Path,
    *,
    unit_test_passed: bool = True,
    fixture_auroc: float = 0.91,
    harness_module_path: str = "python/carnot/verify/reasoner_self_verification.py",
) -> None:
    results_dir = root / "results"
    module_path = root / harness_module_path
    results_dir.mkdir(parents=True, exist_ok=True)
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text("# fixture harness\n", encoding="utf-8")
    (results_dir / "experiment_3894_reasoner_self_verify_harness.json").write_text(
        json.dumps(
            {
                "honest_verdict": "complete: reasoner_self_verify_harness_READY_fixture",
                "harness_module_path": harness_module_path,
                "unit_test_passed": unit_test_passed,
                "fixture_auroc": fixture_auroc,
                "fixture_n_caught": 6,
            }
        ),
        encoding="utf-8",
    )


def _write_exp3915_fixture(
    root: Path,
    *,
    unit_test_passed: bool = True,
    smoke_tokens: int = 1,
    model_used: str = "gemma-4-26B-A4B-it",
    harness_module_path: str = "python/carnot/verify/gguf_inference.py",
) -> None:
    results_dir = root / "results"
    module_path = root / harness_module_path
    results_dir.mkdir(parents=True, exist_ok=True)
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text("# fixture gguf harness\n", encoding="utf-8")
    (results_dir / "experiment_3915_robust_gguf_inference_harness.json").write_text(
        json.dumps(
            {
                "honest_verdict": "complete: gguf_inference_harness_READY_fixture",
                "harness_module_path": harness_module_path,
                "model_used": model_used,
                "n_gpu_layers_used": -1,
                "smoke_tokens": smoke_tokens,
                "unit_test_passed": unit_test_passed,
                "model_specs": {"selected": {"model_used": model_used, "n_gpu_layers_used": -1}},
            }
        ),
        encoding="utf-8",
    )


def _metrics(**overrides: Any) -> exp.ScissorMetrics:
    base = {
        "residual_catch_rate": 0.62,
        "residual_catch_ci95": {
            "mean": 0.62,
            "low": 0.51,
            "high": 0.72,
            "n_resamples": 1000,
            "bootstrap_seed": 3916,
        },
        "error_overlap_jaccard": 0.5,
        "reasoner_self_verify_auroc": 0.50,
        "carnot_ensemble_auroc": 0.8,
        "n_items": 200,
        "n_residual_errors": 50,
        "n_gold_incorrect": 100,
        "reasoner_caught_error_indices": tuple(range(50)),
        "carnot_caught_error_indices": tuple(range(100)),
    }
    base.update(overrides)
    return exp.ScissorMetrics(**base)


def _scoring(selected_panel: exp.Exp3884Panel, *, n_caught: int, score: float = 0.1) -> exp.SelfVerifyArmScoring:
    preds = [1 if label == 1 and idx < n_caught else 0 for idx, label in enumerate(selected_panel.labels)]
    return exp.SelfVerifyArmScoring(
        raw_responses=["incorrect" if pred else "correct" for pred in preds],
        error_scores=[0.9 if pred else score for pred in preds],
        error_preds=preds,
        parsed_count=len(preds),
        unparsed_count=0,
        parser_constant_prediction=False,
    )


def test_req_verify_3916_spec_anchor_exists() -> None:
    """REQ-VERIFY-3916: the robust-GGUF scissor is OpenSpec anchored."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-VERIFY-3916" in spec
    assert "SCENARIO-VERIFY-3916" in spec
    assert exp.OUTPUT_REL_PATH.as_posix() in spec
    assert "load_gguf_generator(...)" in spec
    assert "gguf_harness_model_used" in spec


def test_req_verify_3916_loads_gguf_fixture_control_judge_and_corpus(tmp_path: Path) -> None:
    """REQ-VERIFY-3916: exp3915, exp3894, and exp3884 are fixed disk inputs."""

    rows = _write_exp3884_fixture(tmp_path, n_per_class=3)
    _write_exp3894_fixture(tmp_path, fixture_auroc=0.61)
    _write_exp3915_fixture(tmp_path, model_used="gemma-4-26B-A4B-it")

    gguf = exp.load_exp3915_gguf_harness_source(tmp_path)
    judge = exp.load_exp3894_harness_source(tmp_path)
    panel = exp.load_exp3884_panel(tmp_path, min_incorrect=3)

    assert gguf["artifact_path"] == exp.EXP3915_ARTIFACT_REL_PATH.as_posix()
    assert gguf["harness_module_path"] == exp.GGUF_HARNESS_MODULE_PATH
    assert gguf["model_used"] == "gemma-4-26B-A4B-it"
    assert gguf["smoke_tokens"] == 1
    assert judge["fixture_auroc"] == 0.61
    assert [row["corpus_item_id"] for row in panel.rows] == [row["corpus_item_id"] for row in rows]


def test_req_verify_3916_gguf_harness_fails_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-3916: an unready exp3915 artifact blocks the robust live path."""

    with pytest.raises(FileNotFoundError):
        exp.load_exp3915_gguf_harness_source(tmp_path)

    _write_exp3915_fixture(tmp_path, unit_test_passed=False)
    with pytest.raises(ValueError, match="unit_test_passed"):
        exp.load_exp3915_gguf_harness_source(tmp_path)

    _write_exp3915_fixture(tmp_path, smoke_tokens=0)
    with pytest.raises(ValueError, match="smoke_tokens"):
        exp.load_exp3915_gguf_harness_source(tmp_path)

    artifact_path = tmp_path / exp.EXP3915_ARTIFACT_REL_PATH
    artifact_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="not a JSON object"):
        exp.load_exp3915_gguf_harness_source(tmp_path)

    _write_exp3915_fixture(tmp_path, harness_module_path="python/carnot/verify/missing.py")
    with pytest.raises(FileNotFoundError, match="harness module path mismatch"):
        exp.load_exp3915_gguf_harness_source(tmp_path)

    _write_exp3915_fixture(tmp_path)
    (tmp_path / exp.GGUF_HARNESS_MODULE_PATH).unlink()
    with pytest.raises(FileNotFoundError, match="harness module is missing"):
        exp.load_exp3915_gguf_harness_source(tmp_path)


def test_req_verify_3916_score_arm_uses_robust_generator_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3916: self-verify is backed by gguf_inference.generate."""

    _write_exp3884_fixture(tmp_path, n_per_class=1)
    panel = exp.load_exp3884_panel(tmp_path, min_incorrect=1)
    generate_calls: list[dict[str, object]] = []
    reasoner_calls: list[dict[str, object]] = []

    class FakeGenerator:
        pass

    generator = FakeGenerator()

    def fake_generate(selected_generator: object, prompt: str, max_tokens: int) -> str:
        generate_calls.append({"generator": selected_generator, "prompt": prompt, "max_tokens": max_tokens})
        return '{"verdict":"incorrect","error_confidence":0.9}'

    def fake_reasoner_self_verify(steps: list[str], model_path: str, **kwargs: object) -> dict[str, object]:
        reasoner_calls.append({"steps": steps, "model_path": model_path, "kwargs": kwargs})
        adapter = kwargs["llama_factory"](model_path="ignored")
        assert adapter("prompt text", max_tokens=7, temperature=0.0).strip()
        return {
            "per_step_pred": [1, 0],
            "per_step_score": [0.9, 0.1],
            "raw_responses": ["incorrect", "correct"],
            "parsed_count": 2,
            "unparsed_count": 0,
            "parser_constant_prediction": False,
            "auroc": 1.0,
            "n_caught": 1,
        }

    monkeypatch.setattr(exp, "gguf_generate", fake_generate)
    monkeypatch.setattr(exp, "reasoner_self_verify", fake_reasoner_self_verify)

    weak = exp.score_reasoner_arm_with_robust_generator(
        panel,
        generator,
        {"gguf_path": "fixture.gguf", "n_gpu_layers_used": -1},
        arm="weak",
        max_tokens=12,
        random_seed=123,
    )
    strong = exp.score_reasoner_arm_with_robust_generator(
        panel,
        generator,
        {"gguf_path": "fixture.gguf", "n_gpu_layers_used": -1},
        arm="strong",
        max_tokens=18,
        random_seed=456,
    )

    assert generate_calls[0] == {"generator": generator, "prompt": "prompt text", "max_tokens": 7}
    assert reasoner_calls[0]["kwargs"]["llama_factory"]
    assert "prompt_builder" not in reasoner_calls[0]["kwargs"]
    assert reasoner_calls[1]["kwargs"]["prompt_builder"] is exp.build_boosted_judge_prompt
    assert reasoner_calls[1]["kwargs"]["random_seed"] == 456
    assert weak.error_preds == (1, 0)
    assert strong.raw_responses == ("incorrect", "correct")

    with pytest.raises(ValueError, match="arm"):
        exp.score_reasoner_arm_with_robust_generator(panel, generator, {"gguf_path": "fixture.gguf"}, arm="other")


def test_req_verify_3916_load_robust_generator_uses_exp3915_prefer_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3916: live loading delegates to the robust GGUF harness."""

    calls: list[dict[str, object]] = []
    generator = object()

    def fake_load_gguf_generator(**kwargs: object) -> tuple[object, dict[str, object]]:
        calls.append(kwargs)
        return generator, {
            "model_used": "gemma-4-26B-A4B-it",
            "gguf_path": "fixture.gguf",
            "n_gpu_layers_used": -1,
            "smoke_tokens": 1,
        }

    monkeypatch.setattr(exp, "load_gguf_generator", fake_load_gguf_generator)
    loaded, meta = exp.load_robust_generator(
        {"model_used": "gemma-4-26B-A4B-it", "n_gpu_layers_used": -1},
        exp.ExperimentConfig(repo_root=tmp_path, n_ctx=3072, max_tokens_weak=11, max_tokens_strong=13),
    )

    assert loaded is generator
    assert calls == [
        {
            "prefer_order": ["gemma-4-26B-A4B-it", "Qwen3.6-35B-A3B", "gemma-4-31B-it"],
            "n_ctx": 3072,
            "max_n_gpu_layers": -1,
        }
    ]
    assert meta["loader"] == "carnot.verify.gguf_inference.load_gguf_generator"
    assert meta["source_exp3915_model_used"] == "gemma-4-26B-A4B-it"
    assert meta["max_tokens_weak"] == 11


def test_scenario_verify_3916_artifact_builder_uses_required_bare_fields(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3916: artifacts expose robust GGUF and strong-arm gate fields."""

    artifact = exp.build_artifact_from_metrics(
        weak_metrics=_metrics(residual_catch_rate=0.7, reasoner_self_verify_auroc=0.546),
        strong_metrics=_metrics(reasoner_self_verify_auroc=0.50),
        config=exp.ExperimentConfig(repo_root=tmp_path, started_at=10.0, clock=lambda: 75.0),
        preconditions_checked=[exp.PreconditionCheck("cuda_available", True, "ok")],
        model_specs={
            "model_used": "gemma-4-26B-A4B-it",
            "gguf_path": "fixture.gguf",
            "n_gpu_layers_used": -1,
        },
        gguf_harness_source={
            "harness_module_path": exp.GGUF_HARNESS_MODULE_PATH,
            "model_used": "gemma-4-26B-A4B-it",
            "smoke_tokens": 1,
        },
        harness_source={"harness_module_path": exp.HARNESS_MODULE_PATH, "fixture_auroc": 0.91},
        corpus_source={"corpus_path": "data/in_distribution_error_corpus_v1.json"},
        panel_sha256="p" * 64,
        weak_reasoner_error_scores=[0.9, 0.1, 0.1],
        strong_reasoner_error_scores=[0.8, 0.1, 0.1],
        carnot_error_scores=[0.9, 0.8, 0.1],
        per_step_results=[{"question_id": "incorrect-0"}],
    )

    exp.validate_artifact(artifact)
    assert artifact["experiment"] == 3916
    assert artifact["honest_verdict"].startswith("complete: moat_scissor_MOAT_SURVIVES")
    assert artifact["gguf_harness_model_used"] == "gemma-4-26B-A4B-it"
    assert artifact["duration_s"] == 65.0
    assert len(artifact["reproducibility_checksum"]) == 64
    assert set(exp.REQUIRED_PRINCIPLE_FIELDS) <= set(artifact["field_principles"])
    assert all(isinstance(artifact["field_principles"][field], str) for field in exp.REQUIRED_PRINCIPLE_FIELDS)


def test_req_verify_3916_fake_dual_arm_scoring_builds_artifact_without_live_llm(tmp_path: Path) -> None:
    """REQ-VERIFY-3916: injected scorers keep unit tests off the live GGUF path."""

    _write_exp3884_fixture(tmp_path, n_per_class=100)
    _write_exp3894_fixture(tmp_path)
    _write_exp3915_fixture(tmp_path)
    panel = exp.load_exp3884_panel(tmp_path)
    judge_source = exp.load_exp3894_harness_source(tmp_path)
    gguf_source = exp.load_exp3915_gguf_harness_source(tmp_path)

    artifact = exp.build_artifact_for_panel(
        panel,
        config=exp.ExperimentConfig(
            repo_root=tmp_path,
            output_path=tmp_path / exp.OUTPUT_REL_PATH,
            started_at=0.0,
            clock=lambda: 70.0,
        ),
        preconditions_checked=[exp.PreconditionCheck("test", True, "injected")],
        model_specs={"model_used": "gemma-4-26B-A4B-it", "gguf_path": "fixture.gguf", "n_gpu_layers_used": -1},
        gguf_harness_source=gguf_source,
        harness_source=judge_source,
        generator=object(),
        weak_reasoner_scorer=lambda selected_panel, _generator, _model_specs: _scoring(selected_panel, n_caught=40),
        strong_reasoner_scorer=lambda selected_panel, _generator, _model_specs: _scoring(selected_panel, n_caught=50),
        write=True,
    )

    assert (tmp_path / exp.OUTPUT_REL_PATH).exists()
    assert artifact["n_items"] == 200
    assert artifact["n_residual_errors_weak"] == 60
    assert artifact["n_residual_errors_strong"] == 50
    assert artifact["honest_verdict"].startswith("complete: moat_scissor_MOAT_SURVIVES")
    assert artifact["gguf_harness_model_used"] == "gemma-4-26B-A4B-it"
    assert artifact["per_step_results"][49]["reasoner_strong_rejects"] is True
    assert artifact["per_step_results"][-1]["carnot_rejects"] is False


def test_req_verify_3916_validate_and_blocked_artifact_failures(tmp_path: Path) -> None:
    """REQ-VERIFY-3916: blocked artifacts stay terminal and schema validation is strict."""

    valid = exp.build_blocked_artifact(
        reason="blocked_no_cuda",
        preconditions_checked=[],
        duration_s=0.1,
    )
    exp.validate_artifact(valid)
    assert valid["gguf_harness_model_used"] is None
    assert valid["residual_catch_rate_weak"] is None
    assert valid["residual_catch_rate_strong"] is None
    assert valid["n_items"] == 0

    with pytest.raises(ValueError, match="missing required"):
        exp.validate_artifact({k: v for k, v in valid.items() if k != "random_seed"})
    with pytest.raises(ValueError, match="terminal prefix"):
        exp.validate_artifact(dict(valid, honest_verdict="not_terminal"))
    with pytest.raises(ValueError, match="field_principles"):
        exp.validate_artifact(dict(valid, field_principles=[]))
    bad_principles = dict(valid["field_principles"])
    bad_principles["random_seed"] = {"principle": "wrapped"}
    with pytest.raises(ValueError, match="random_seed"):
        exp.validate_artifact(dict(valid, field_principles=bad_principles))
    wrapped_metric = dict(valid)
    wrapped_metric["residual_catch_rate_strong"] = {"value": 0.0, "principle": "bad"}
    with pytest.raises(ValueError, match="residual_catch_rate_strong"):
        exp.validate_artifact(wrapped_metric)
    with pytest.raises(ValueError, match="llama_cpp"):
        exp.validate_artifact(dict(valid, inference_substrate="live_llama_cpp_direct"))
    with pytest.raises(ValueError, match="generation headroom"):
        exp.validate_artifact(dict(valid, generation_headroom=True))

    output = tmp_path / exp.OUTPUT_REL_PATH
    persisted = exp.write_blocked_artifact(
        output,
        reason="blocked_no_cuda",
        preconditions_checked=[exp.PreconditionCheck("cuda", False, "no")],
        duration_s=0.5,
    )
    assert json.loads(output.read_text(encoding="utf-8")) == persisted


@pytest.mark.parametrize(
    ("false_resource", "expected_reason"),
    [
        ("cuda_available", "blocked_no_cuda"),
        ("exp3915_gguf_harness_ready", "blocked_upstream_gguf_harness_not_ready"),
        ("carnot_verify_import", "blocked_carnot_verify_import"),
        ("gguf_inference_import", "blocked_carnot_verify_import"),
        ("reasoner_self_verification_import", "blocked_carnot_verify_import"),
        ("exp3894_judge_ready", "blocked_upstream_judge_not_ready"),
        ("exp3884_corpus_in_band", "blocked_upstream_corpus_not_in_band"),
    ],
)
def test_req_verify_3916_probe_preconditions_block_reasons(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    false_resource: str,
    expected_reason: str,
) -> None:
    """REQ-VERIFY-3916: resource failures map to terminal blocked verdicts."""

    def fake_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        available = false_resource != "cuda_available"
        return subprocess.CompletedProcess(command, 0 if available else 1, "ok", "" if available else "no cuda")

    def fake_import(name: str) -> object:
        if false_resource == "carnot_verify_import" and name == "carnot.verify":
            raise RuntimeError("no verify")
        if false_resource == "gguf_inference_import" and name == "carnot.verify.gguf_inference":
            raise RuntimeError("no gguf")
        if false_resource == "reasoner_self_verification_import" and name == "carnot.verify.reasoner_self_verification":
            raise RuntimeError("no judge")
        if name == "llama_cpp":
            raise AssertionError("Exp 3916 preflight must not import llama_cpp directly")
        return object()

    monkeypatch.setattr(exp.importlib, "import_module", fake_import)
    if false_resource != "exp3884_corpus_in_band":
        _write_exp3884_fixture(tmp_path, n_per_class=100)
    if false_resource != "exp3894_judge_ready":
        _write_exp3894_fixture(tmp_path)
    if false_resource != "exp3915_gguf_harness_ready":
        _write_exp3915_fixture(tmp_path)

    preflight = exp.probe_preconditions(
        exp.ExperimentConfig(repo_root=tmp_path),
        command_runner=fake_runner,
    )

    assert preflight.blocked_reason == expected_reason


def test_req_verify_3916_run_experiment_orchestrates_blocked_success_and_load_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3916: run_experiment writes blocked, success, or GGUF-load failure artifacts."""

    blocked_preflight = exp.PreflightResult(
        checks=(exp.PreconditionCheck("cuda", False, "no"),),
        blocked_reason="blocked_no_cuda",
        model_specs={},
        panel=None,
        harness_source=None,
        gguf_harness_source=None,
    )
    monkeypatch.setattr(exp, "probe_preconditions", lambda _config: blocked_preflight)
    blocked = exp.run_experiment(
        exp.ExperimentConfig(repo_root=tmp_path, output_path=tmp_path / "blocked.json", started_at=0.0, clock=lambda: 1.0),
        write=True,
    )
    assert blocked["honest_verdict"] == "blocked_no_cuda"
    assert (tmp_path / "blocked.json").exists()

    _write_exp3884_fixture(tmp_path, n_per_class=100)
    _write_exp3894_fixture(tmp_path)
    _write_exp3915_fixture(tmp_path)
    panel = exp.load_exp3884_panel(tmp_path)
    judge_source = exp.load_exp3894_harness_source(tmp_path)
    gguf_source = exp.load_exp3915_gguf_harness_source(tmp_path)
    success_preflight = exp.PreflightResult(
        checks=(exp.PreconditionCheck("ok", True, "yes"),),
        blocked_reason=None,
        model_specs={"prefer_order": ["gemma-4-26B-A4B-it"]},
        panel=panel,
        harness_source=judge_source,
        gguf_harness_source=gguf_source,
    )
    monkeypatch.setattr(exp, "probe_preconditions", lambda _config: success_preflight)
    monkeypatch.setattr(
        exp,
        "load_robust_generator",
        lambda _source, _config: (object(), {"model_used": "gemma-4-26B-A4B-it", "gguf_path": "fixture.gguf"}),
    )
    monkeypatch.setattr(
        exp,
        "build_artifact_for_panel",
        lambda panel, **kwargs: {"honest_verdict": "complete: fixture", "n_items": len(panel.rows), "kwargs": sorted(kwargs)},
    )
    success = exp.run_experiment(exp.ExperimentConfig(repo_root=tmp_path), write=False)
    assert success["honest_verdict"] == "complete: fixture"
    assert success["n_items"] == 200

    monkeypatch.setattr(
        exp,
        "load_robust_generator",
        lambda _source, _config: (_ for _ in ()).throw(RuntimeError("blocked_all_gguf_inference_failed")),
    )
    failed = exp.run_experiment(
        exp.ExperimentConfig(repo_root=tmp_path, output_path=tmp_path / "failed.json", started_at=0.0, clock=lambda: 2.0),
        write=True,
    )
    assert failed["honest_verdict"] == "blocked_all_gguf_inference_failed"
    assert failed["preconditions_checked"][-1]["resource"] == "robust_gguf_generator_load"

    monkeypatch.setattr(
        exp,
        "load_robust_generator",
        lambda _source, _config: (object(), {"model_used": "gemma-4-26B-A4B-it", "gguf_path": "fixture.gguf"}),
    )
    monkeypatch.setattr(
        exp,
        "build_artifact_for_panel",
        lambda panel, **kwargs: (_ for _ in ()).throw(RuntimeError("judge failed")),
    )
    reasoner_failed = exp.run_experiment(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            output_path=tmp_path / "reasoner_failed.json",
            started_at=0.0,
            clock=lambda: 3.0,
        ),
        write=True,
    )
    assert reasoner_failed["honest_verdict"] == "blocked_reasoner_self_verify_inference_failed"
    assert reasoner_failed["preconditions_checked"][-1]["resource"] == "reasoner_self_verify_inference"

    no_gguf_preflight = exp.PreflightResult(
        checks=(exp.PreconditionCheck("gguf", True, "missing source"),),
        blocked_reason=None,
        model_specs={},
        panel=panel,
        harness_source=judge_source,
        gguf_harness_source=None,
    )
    monkeypatch.setattr(exp, "probe_preconditions", lambda _config: no_gguf_preflight)
    no_gguf = exp.run_experiment(
        exp.ExperimentConfig(repo_root=tmp_path, output_path=tmp_path / "no_gguf.json"),
        write=True,
    )
    assert no_gguf["honest_verdict"] == "blocked_upstream_gguf_harness_not_ready"

    no_judge_preflight = exp.PreflightResult(
        checks=(exp.PreconditionCheck("judge", True, "missing source"),),
        blocked_reason=None,
        model_specs={},
        panel=panel,
        harness_source=None,
        gguf_harness_source=gguf_source,
    )
    monkeypatch.setattr(exp, "probe_preconditions", lambda _config: no_judge_preflight)
    no_judge = exp.run_experiment(
        exp.ExperimentConfig(repo_root=tmp_path, output_path=tmp_path / "no_judge.json"),
        write=True,
    )
    assert no_judge["honest_verdict"] == "blocked_upstream_judge_not_ready"

    no_panel_preflight = exp.PreflightResult(
        checks=(exp.PreconditionCheck("corpus", True, "missing panel"),),
        blocked_reason=None,
        model_specs={},
        panel=None,
        harness_source=judge_source,
        gguf_harness_source=gguf_source,
    )
    monkeypatch.setattr(exp, "probe_preconditions", lambda _config: no_panel_preflight)
    no_panel = exp.run_experiment(
        exp.ExperimentConfig(repo_root=tmp_path, output_path=tmp_path / "no_panel.json"),
        write=True,
    )
    assert no_panel["honest_verdict"] == "blocked_upstream_corpus_not_in_band"


def test_req_verify_3916_cli_main_reports_terminal_status(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3916: CLI adapter reports the written path and blocked status."""

    monkeypatch.setattr(exp, "run_experiment", lambda _config, write: {"honest_verdict": "complete: fixture"})
    assert exp.cli_main(["--repo-root", str(tmp_path)]) == 0
    assert exp.OUTPUT_REL_PATH.name in capsys.readouterr().out

    monkeypatch.setattr(exp, "run_experiment", lambda _config, write: {"honest_verdict": "blocked_no_cuda"})
    assert exp.cli_main(["--repo-root", str(tmp_path), "--output-path", str(tmp_path / "out.json")]) == 1
    assert "blocked_no_cuda" in capsys.readouterr().out
