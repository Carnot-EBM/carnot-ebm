"""Tests for Exp 3895 tested-harness in-distribution moat scissor.

Spec refs: REQ-VERIFY-3895, SCENARIO-VERIFY-3895,
SCENARIO-VERIFY-3895-BLOCKED.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import moat_scissor_tested_harness as exp


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
    corpus_path = data_dir / "in_distribution_error_corpus_v1.json"
    scores_path = results_dir / "experiment_3884_in_distribution_error_rich_corpus_scores.json"
    artifact_path = results_dir / "experiment_3884_in_distribution_error_rich_corpus.json"
    corpus_path.write_text(json.dumps({"schema": "fixture", "items": rows}), encoding="utf-8")
    scores_path.write_text(
        json.dumps({"schema": "fixture", "items": [_score_for_row(i, row) for i, row in enumerate(rows)]}),
        encoding="utf-8",
    )
    artifact_path.write_text(
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
                "fixture_auroc": 0.91,
                "fixture_n_caught": 6,
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
            "bootstrap_seed": 3895,
        },
        "error_overlap_jaccard": 0.5,
        "reasoner_self_verify_auroc": 0.72,
        "carnot_ensemble_auroc": 0.8,
        "n_items": 200,
        "n_residual_errors": 50,
        "n_gold_incorrect": 100,
        "reasoner_caught_error_indices": tuple(range(50)),
        "carnot_caught_error_indices": tuple(range(100)),
    }
    base.update(overrides)
    return exp.ScissorMetrics(**base)


def test_req_verify_3895_spec_anchor_exists() -> None:
    """REQ-VERIFY-3895: the tested-harness scissor is OpenSpec anchored."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-VERIFY-3895" in spec
    assert "SCENARIO-VERIFY-3895" in spec
    assert exp.OUTPUT_REL_PATH.as_posix() in spec
    assert "reasoner_self_verify(steps, model_path, gold_labels=...)" in spec


def test_req_verify_3895_loads_upstream_harness_and_corpus(tmp_path: Path) -> None:
    """REQ-VERIFY-3895: exp3894 and exp3884 disk artifacts are the fixed inputs."""

    rows = _write_exp3884_fixture(tmp_path, n_per_class=3)
    _write_exp3894_fixture(tmp_path)

    harness = exp.load_exp3894_harness_source(tmp_path)
    panel = exp.load_exp3884_panel(tmp_path, min_incorrect=3)

    assert harness["artifact_path"] == exp.EXP3894_ARTIFACT_REL_PATH.as_posix()
    assert harness["harness_module_path"] == exp.HARNESS_MODULE_PATH
    assert harness["unit_test_passed"] is True
    assert len(harness["artifact_sha256"]) == 64
    assert len(harness["harness_module_sha256"]) == 64
    assert [row["corpus_item_id"] for row in panel.rows] == [row["corpus_item_id"] for row in rows]
    assert panel.labels == (1, 1, 1, 0, 0, 0)
    assert panel.carnot_error_preds == (1, 1, 1, 0, 0, 0)


def test_req_verify_3895_upstream_harness_fails_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-3895: an unready exp3894 artifact blocks the scissor."""

    with pytest.raises(FileNotFoundError):
        exp.load_exp3894_harness_source(tmp_path)

    _write_exp3894_fixture(tmp_path, unit_test_passed=False)
    with pytest.raises(ValueError, match="unit_test_passed"):
        exp.load_exp3894_harness_source(tmp_path)

    artifact_path = tmp_path / exp.EXP3894_ARTIFACT_REL_PATH
    artifact_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="not a JSON object"):
        exp.load_exp3894_harness_source(tmp_path)

    _write_exp3894_fixture(tmp_path, harness_module_path="python/carnot/verify/missing.py")
    with pytest.raises(FileNotFoundError, match="harness module"):
        exp.load_exp3894_harness_source(tmp_path)

    _write_exp3894_fixture(tmp_path)
    (tmp_path / exp.HARNESS_MODULE_PATH).unlink()
    with pytest.raises(FileNotFoundError, match="harness module"):
        exp.load_exp3894_harness_source(tmp_path)


def test_scenario_verify_3895_residual_math_uses_reasoner_predictions() -> None:
    """SCENARIO-VERIFY-3895: residual sets use tested-harness preds, not score>0."""

    metrics = exp.compute_tested_harness_scissor_metrics(
        labels=[1, 1, 1, 1, 0, 0],
        reasoner_error_scores=[1.0, 0.2, 0.8, 0.2, 0.2, 0.8],
        reasoner_error_preds=[1, 0, -1, 0, 0, 1],
        carnot_error_scores=[0.9, 0.8, 0.1, 0.2, 0.3, 0.4],
        carnot_error_preds=[1, 1, 0, 0, 0, 0],
        bootstrap_seed=7,
        bootstrap_resamples=1000,
    )

    assert metrics.n_residual_errors == 3
    assert metrics.residual_catch_rate == pytest.approx(1 / 3)
    assert metrics.residual_catch_ci95["n_resamples"] == 1000
    assert metrics.error_overlap_jaccard == pytest.approx(1 / 2)
    assert metrics.reasoner_caught_error_indices == (0,)
    assert metrics.carnot_caught_error_indices == (0, 1)

    with pytest.raises(ValueError, match="lengths"):
        exp.compute_tested_harness_scissor_metrics(
            labels=[1],
            reasoner_error_scores=[0.1, 0.2],
            reasoner_error_preds=[0],
            carnot_error_scores=[0.1],
            carnot_error_preds=[0],
            bootstrap_seed=7,
            bootstrap_resamples=1000,
        )


def test_scenario_verify_3895_terminal_gates() -> None:
    """SCENARIO-VERIFY-3895: verdicts follow the moat falsification gate."""

    assert "MOAT_SURVIVES" in exp.classify_verdict(_metrics())
    assert "MOAT_SUBSUMED" in exp.classify_verdict(
        _metrics(
            residual_catch_rate=0.21,
            residual_catch_ci95={**_metrics().residual_catch_ci95, "mean": 0.21, "low": 0.15, "high": 0.29},
        )
    )
    assert "MOAT_SUBSUMED" in exp.classify_verdict(_metrics(error_overlap_jaccard=0.71))
    assert exp.classify_verdict(
        _metrics(reasoner_self_verify_auroc=0.50, carnot_ensemble_auroc=0.64)
    ).endswith("reasoner_self_verify_auroc_and_carnot_ensemble_auroc")
    assert exp.classify_verdict(_metrics(reasoner_self_verify_auroc=0.98)).endswith(
        "reasoner_self_verify_auroc"
    )
    assert exp.classify_verdict(_metrics(n_residual_errors=29)).endswith("n_residual_errors_lt30")
    assert exp.classify_verdict(
        _metrics(
            residual_catch_rate=0.4,
            residual_catch_ci95={**_metrics().residual_catch_ci95, "low": 0.31, "high": 0.49},
            error_overlap_jaccard=0.65,
        )
    ).endswith("boundary_gate")


def test_req_verify_3895_artifact_builder_uses_bare_fields_and_string_principles(tmp_path: Path) -> None:
    """REQ-VERIFY-3895: artifacts carry bare values, principle strings, and checksum."""

    artifact = exp.build_artifact_from_metrics(
        metrics=_metrics(),
        config=exp.ExperimentConfig(repo_root=tmp_path, started_at=10.0, clock=lambda: 75.0),
        preconditions_checked=[exp.PreconditionCheck("cuda_available", True, "ok")],
        model_specs={"hf_id": "fixture", "model_path": "fixture.gguf"},
        harness_source={"harness_module_path": exp.HARNESS_MODULE_PATH},
        corpus_source={"corpus_path": "data/in_distribution_error_corpus_v1.json"},
        panel_sha256="p" * 64,
        reasoner_error_scores=[1, 0, 1],
        carnot_error_scores=[0.9, 0.1, 0.8],
        per_step_results=[{"question_id": "incorrect-0"}],
    )

    exp.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["duration_s"] == 65.0
    assert len(artifact["reproducibility_checksum"]) == 64
    assert set(exp.REQUIRED_PRINCIPLE_FIELDS) <= set(artifact["field_principles"])
    assert all(isinstance(artifact["field_principles"][field], str) for field in exp.REQUIRED_PRINCIPLE_FIELDS)
    assert artifact["per_step_results"] == [{"question_id": "incorrect-0"}]


def test_req_verify_3895_validate_artifact_failures(tmp_path: Path) -> None:
    """REQ-VERIFY-3895: schema validation rejects non-terminal or wrapped fields."""

    valid = exp.build_blocked_artifact(
        reason="blocked_no_cuda",
        preconditions_checked=[],
        duration_s=0.1,
    )
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
    wrapped_metric["residual_catch_rate"] = {"value": 0.0, "principle": "bad"}
    with pytest.raises(ValueError, match="residual_catch_rate"):
        exp.validate_artifact(wrapped_metric)
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


def test_req_verify_3895_calls_tested_reasoner_harness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3895: live reasoner scoring delegates to exp3894 harness."""

    _write_exp3884_fixture(tmp_path, n_per_class=1)
    panel = exp.load_exp3884_panel(tmp_path, min_incorrect=1)
    calls: list[dict[str, object]] = []

    def fake_reasoner_self_verify(steps: list[str], model_path: str, **kwargs: object) -> dict[str, object]:
        calls.append({"steps": steps, "model_path": model_path, "kwargs": kwargs})
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

    monkeypatch.setattr(exp, "reasoner_self_verify", fake_reasoner_self_verify)

    scoring = exp.score_reasoner_with_tested_harness(
        panel,
        {
            "model_path": "fixture.gguf",
            "n_gpu_layers": -1,
            "n_ctx": 1024,
            "n_batch": 64,
            "offload_kqv": True,
        },
        max_tokens=12,
        random_seed=123,
    )

    assert calls[0]["steps"] == list(panel.texts)
    assert calls[0]["model_path"] == "fixture.gguf"
    assert calls[0]["kwargs"]["gold_labels"] == panel.labels
    assert calls[0]["kwargs"]["max_tokens"] == 12
    assert calls[0]["kwargs"]["random_seed"] == 123
    assert scoring.error_scores == (0.9, 0.1)
    assert scoring.error_preds == (1, 0)
    assert scoring.raw_responses == ("incorrect", "correct")


def test_req_verify_3895_checked_harness_result_validation() -> None:
    """REQ-VERIFY-3895: malformed tested-harness outputs fail closed."""

    with pytest.raises(ValueError, match="not a sequence"):
        exp._checked_sequence("bad", field="raw_responses", n_expected=1)
    with pytest.raises(ValueError, match="length"):
        exp._checked_sequence([1, 2], field="per_step_pred", n_expected=1)


def test_req_verify_3895_fake_scoring_builds_artifact_without_live_llm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3895: injected harness scoring lets unit tests avoid live GGUF inference."""

    _write_exp3884_fixture(tmp_path, n_per_class=100)
    _write_exp3894_fixture(tmp_path)
    panel = exp.load_exp3884_panel(tmp_path)
    harness_source = exp.load_exp3894_harness_source(tmp_path)

    def fake_reasoner(
        selected_panel: exp.Exp3884Panel,
        _model_specs: dict[str, object],
    ) -> exp.TestedHarnessScoring:
        preds = [1 if label == 1 and idx < 50 else 0 for idx, label in enumerate(selected_panel.labels)]
        return exp.TestedHarnessScoring(
            raw_responses=["incorrect" if pred else "correct" for pred in preds],
            error_scores=[0.9 if pred else 0.1 for pred in preds],
            error_preds=preds,
            parsed_count=len(preds),
            unparsed_count=0,
            parser_constant_prediction=False,
        )

    artifact = exp.build_artifact_for_panel(
        panel,
        config=exp.ExperimentConfig(
            repo_root=tmp_path,
            output_path=tmp_path / exp.OUTPUT_REL_PATH,
            started_at=0.0,
            clock=lambda: 70.0,
        ),
        preconditions_checked=[exp.PreconditionCheck("test", True, "injected")],
        model_specs={"hf_id": "fixture", "model_path": "fixture.gguf"},
        harness_source=harness_source,
        reasoner_scorer=fake_reasoner,
        write=True,
    )

    assert (tmp_path / exp.OUTPUT_REL_PATH).exists()
    assert artifact["n_items"] == 200
    assert artifact["n_residual_errors"] == 50
    assert artifact["honest_verdict"].startswith("complete: moat_scissor_MOAT_SURVIVES")
    assert artifact["harness_source"]["unit_test_passed"] is True
    assert artifact["per_step_results"][0]["reasoner_rejects"] is True
    assert artifact["per_step_results"][-1]["carnot_rejects"] is False


@pytest.mark.parametrize(
    ("false_resource", "expected_reason"),
    [
        ("cuda_available", "blocked_no_cuda"),
        ("model_path", "blocked_model_not_cached"),
        ("carnot_verify_import", "blocked_carnot_verify_import"),
        ("llama_cpp_import", "blocked_llama_cpp_not_installed"),
        ("exp3894_harness_ready", "blocked_upstream_harness_not_ready"),
        ("exp3884_corpus_in_band", "blocked_upstream_corpus_not_in_band"),
    ],
)
def test_req_verify_3895_probe_preconditions_block_reasons(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    false_resource: str,
    expected_reason: str,
) -> None:
    """REQ-VERIFY-3895: resource failures map to terminal blocked verdicts."""

    def fake_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        available = false_resource != "cuda_available"
        return subprocess.CompletedProcess(command, 0 if available else 1, "ok", "" if available else "no cuda")

    def fake_import(name: str) -> object:
        if false_resource == "carnot_verify_import" and name == "carnot.verify":
            raise RuntimeError("no verify")
        if false_resource == "llama_cpp_import" and name == "llama_cpp":
            raise RuntimeError("no llama")
        return object()

    monkeypatch.setattr(exp.importlib, "import_module", fake_import)
    monkeypatch.setattr(
        exp,
        "_resolve_reasoner_model",
        lambda: (
            {"hf_id": "fixture", "model_path": None if false_resource == "model_path" else "fixture.gguf"},
            [exp.PreconditionCheck("qwen3.6_35b_gguf_cached", false_resource != "model_path", "fixture")],
        ),
    )
    if false_resource != "exp3884_corpus_in_band":
        _write_exp3884_fixture(tmp_path, n_per_class=100)
    if false_resource != "exp3894_harness_ready":
        _write_exp3894_fixture(tmp_path)

    preflight = exp.probe_preconditions(
        exp.ExperimentConfig(repo_root=tmp_path),
        command_runner=fake_runner,
    )

    assert preflight.blocked_reason == expected_reason


def test_req_verify_3895_model_resolution_prefers_qwen_then_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3895: Qwen is preferred and Gemma fallback is auditable."""

    qwen = tmp_path / "qwen.gguf"
    qwen.write_bytes(b"qwen")
    monkeypatch.setattr(exp, "resolve_cached_gguf", lambda _hf_id: str(qwen))
    model_specs, checks = exp._resolve_reasoner_model()
    assert model_specs["hf_id"] == exp.PRIMARY_REASONER_HF_ID
    assert model_specs["fallback_used"] is False
    assert checks[0].available is True

    gemma = tmp_path / "gemma.gguf"
    gemma.write_bytes(b"gemma")

    def fallback_only(hf_id: str) -> str | None:
        return str(gemma) if hf_id == exp.FALLBACK_REASONER_HF_ID else None

    monkeypatch.setattr(exp, "resolve_cached_gguf", fallback_only)
    fallback_specs, fallback_checks = exp._resolve_reasoner_model()
    assert fallback_specs["hf_id"] == exp.FALLBACK_REASONER_HF_ID
    assert fallback_specs["fallback_used"] is True
    assert [check.available for check in fallback_checks] == [False, True]

    monkeypatch.setattr(exp, "resolve_cached_gguf", lambda _hf_id: None)
    none_specs, none_checks = exp._resolve_reasoner_model()
    assert none_specs["model_path"] is None
    assert [check.available for check in none_checks] == [False, False]


def test_req_verify_3895_probe_cuda_exception_is_recorded() -> None:
    """REQ-VERIFY-3895: CUDA probe exceptions become precondition evidence."""

    def raising_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise RuntimeError("runner boom")

    check = exp._probe_cuda_with_venv(
        exp.ExperimentConfig(repo_root=Path("/tmp")),
        command_runner=raising_runner,
    )
    assert check.available is False
    assert "runner boom" in check.detail


def test_req_verify_3895_run_experiment_orchestrates_blocked_success_and_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3895: run_experiment writes blocked, success, or inference-failed artifacts."""

    blocked_preflight = exp.PreflightResult(
        checks=(exp.PreconditionCheck("cuda", False, "no"),),
        blocked_reason="blocked_no_cuda",
        model_specs={"hf_id": "fixture"},
        panel=None,
        harness_source=None,
    )
    monkeypatch.setattr(exp, "probe_preconditions", lambda _config: blocked_preflight)
    blocked = exp.run_experiment(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            output_path=tmp_path / "blocked.json",
            started_at=0.0,
            clock=lambda: 1.0,
        ),
        write=True,
    )
    assert blocked["honest_verdict"] == "blocked_no_cuda"
    assert (tmp_path / "blocked.json").exists()

    _write_exp3884_fixture(tmp_path, n_per_class=100)
    _write_exp3894_fixture(tmp_path)
    panel = exp.load_exp3884_panel(tmp_path)
    harness_source = exp.load_exp3894_harness_source(tmp_path)
    success_preflight = exp.PreflightResult(
        checks=(exp.PreconditionCheck("ok", True, "yes"),),
        blocked_reason=None,
        model_specs={"hf_id": "fixture", "model_path": "fixture.gguf"},
        panel=panel,
        harness_source=harness_source,
    )
    monkeypatch.setattr(exp, "probe_preconditions", lambda _config: success_preflight)
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
        "build_artifact_for_panel",
        lambda panel, **kwargs: (_ for _ in ()).throw(RuntimeError("inference failed")),
    )
    failed = exp.run_experiment(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            output_path=tmp_path / "failed.json",
            started_at=0.0,
            clock=lambda: 2.0,
        ),
        write=True,
    )
    assert failed["honest_verdict"] == "blocked_llama_cpp_inference_failed"
    assert failed["preconditions_checked"][-1]["resource"] == "tested_harness_inference"

    no_panel_preflight = exp.PreflightResult(
        checks=(exp.PreconditionCheck("corpus", True, "missing panel"),),
        blocked_reason=None,
        model_specs={"hf_id": "fixture", "model_path": "fixture.gguf"},
        panel=None,
        harness_source=harness_source,
    )
    monkeypatch.setattr(exp, "probe_preconditions", lambda _config: no_panel_preflight)
    no_panel = exp.run_experiment(
        exp.ExperimentConfig(repo_root=tmp_path, output_path=tmp_path / "no_panel.json"),
        write=True,
    )
    assert no_panel["honest_verdict"] == "blocked_upstream_corpus_not_in_band"

    no_harness_preflight = exp.PreflightResult(
        checks=(exp.PreconditionCheck("harness", True, "missing harness"),),
        blocked_reason=None,
        model_specs={"hf_id": "fixture", "model_path": "fixture.gguf"},
        panel=panel,
        harness_source=None,
    )
    monkeypatch.setattr(exp, "probe_preconditions", lambda _config: no_harness_preflight)
    no_harness = exp.run_experiment(
        exp.ExperimentConfig(repo_root=tmp_path, output_path=tmp_path / "no_harness.json"),
        write=True,
    )
    assert no_harness["honest_verdict"] == "blocked_upstream_harness_not_ready"


def test_req_verify_3895_cli_main_reports_terminal_status(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3895: CLI adapter reports the written path and blocked status."""

    monkeypatch.setattr(
        exp,
        "run_experiment",
        lambda _config, write: {"honest_verdict": "complete: fixture"},
    )
    assert exp.cli_main(["--repo-root", str(tmp_path)]) == 0
    assert exp.OUTPUT_REL_PATH.name in capsys.readouterr().out

    monkeypatch.setattr(
        exp,
        "run_experiment",
        lambda _config, write: {"honest_verdict": "blocked_no_cuda"},
    )
    assert exp.cli_main(["--repo-root", str(tmp_path), "--output-path", str(tmp_path / "out.json")]) == 1
    assert "blocked_no_cuda" in capsys.readouterr().out
