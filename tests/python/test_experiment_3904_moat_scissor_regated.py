"""Tests for Exp 3904 regated tested-harness moat scissor.

Spec refs: REQ-VERIFY-3904, SCENARIO-VERIFY-3904,
SCENARIO-VERIFY-3904-BLOCKED.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import moat_scissor_regated as exp


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


def _metrics(**overrides: Any) -> exp.ScissorMetrics:
    base = {
        "residual_catch_rate": 0.62,
        "residual_catch_ci95": {
            "mean": 0.62,
            "low": 0.51,
            "high": 0.72,
            "n_resamples": 1000,
            "bootstrap_seed": 3904,
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


def test_req_verify_3904_spec_anchor_exists() -> None:
    """REQ-VERIFY-3904: the regated scissor is OpenSpec anchored."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-VERIFY-3904" in spec
    assert "SCENARIO-VERIFY-3904" in spec
    assert exp.OUTPUT_REL_PATH.as_posix() in spec
    assert "fixture_auroc > 0.6" in spec
    assert "reasoner AUROC as a finding" in spec


def test_req_verify_3904_loads_harness_fixture_control_and_corpus(tmp_path: Path) -> None:
    """REQ-VERIFY-3904: exp3894 fixture AUROC is the harness-validity control."""

    rows = _write_exp3884_fixture(tmp_path, n_per_class=3)
    _write_exp3894_fixture(tmp_path, fixture_auroc=0.61)

    harness = exp.load_exp3894_harness_source(tmp_path)
    panel = exp.load_exp3884_panel(tmp_path, min_incorrect=3)

    assert harness["artifact_path"] == exp.EXP3894_ARTIFACT_REL_PATH.as_posix()
    assert harness["harness_module_path"] == exp.HARNESS_MODULE_PATH
    assert harness["unit_test_passed"] is True
    assert harness["fixture_auroc"] == 0.61
    assert [row["corpus_item_id"] for row in panel.rows] == [row["corpus_item_id"] for row in rows]
    assert panel.labels == (1, 1, 1, 0, 0, 0)

    _write_exp3894_fixture(tmp_path, fixture_auroc=0.6)
    with pytest.raises(ValueError, match="fixture_auroc"):
        exp.load_exp3894_harness_source(tmp_path)

    _write_exp3894_fixture(tmp_path, unit_test_passed=False, fixture_auroc=0.91)
    with pytest.raises(ValueError, match="unit_test_passed"):
        exp.load_exp3894_harness_source(tmp_path)

    artifact_path = tmp_path / exp.EXP3894_ARTIFACT_REL_PATH
    artifact_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="not a JSON object"):
        exp.load_exp3894_harness_source(tmp_path)

    _write_exp3894_fixture(tmp_path, harness_module_path="python/carnot/verify/other.py")
    with pytest.raises(FileNotFoundError, match="harness module path mismatch"):
        exp.load_exp3894_harness_source(tmp_path)

    _write_exp3894_fixture(tmp_path)
    (tmp_path / exp.HARNESS_MODULE_PATH).unlink()
    with pytest.raises(FileNotFoundError, match="harness module is missing"):
        exp.load_exp3894_harness_source(tmp_path)


def test_req_verify_3904_boosted_prompt_and_arm_scoring_delegate_to_tested_harness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3904: strong arm changes only the prompt, not the parser/judge path."""

    prompt = exp.build_boosted_judge_prompt("2 + 2 = 5.")
    assert "few-shot" in prompt.lower()
    assert "is THIS step correct? why?" in prompt
    assert "2 + 2 = 5." in prompt

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

    weak = exp.score_reasoner_arm(
        panel,
        {"model_path": "fixture.gguf", "n_gpu_layers": -1, "n_ctx": 1024, "n_batch": 64, "offload_kqv": True},
        arm="weak",
        max_tokens=12,
        random_seed=123,
    )
    strong = exp.score_reasoner_arm(
        panel,
        {"model_path": "fixture.gguf", "n_gpu_layers": -1, "n_ctx": 1024, "n_batch": 64, "offload_kqv": True},
        arm="strong",
        max_tokens=18,
        random_seed=456,
    )

    assert calls[0]["kwargs"]["gold_labels"] == panel.labels
    assert "prompt_builder" not in calls[0]["kwargs"]
    assert calls[0]["kwargs"]["max_tokens"] == 12
    assert calls[1]["kwargs"]["prompt_builder"] is exp.build_boosted_judge_prompt
    assert calls[1]["kwargs"]["max_tokens"] == 18
    assert calls[1]["kwargs"]["random_seed"] == 456
    assert weak.error_preds == (1, 0)
    assert strong.raw_responses == ("incorrect", "correct")

    with pytest.raises(ValueError, match="arm"):
        exp.score_reasoner_arm(panel, {"model_path": "fixture.gguf"}, arm="other")


def test_scenario_verify_3904_residual_math_and_regated_terminal_gates() -> None:
    """SCENARIO-VERIFY-3904: strong-arm gates use fixture AUROC, not reasoner AUROC."""

    with pytest.raises(ValueError, match="lengths"):
        exp.compute_arm_scissor_metrics(
            labels=[1],
            reasoner_error_scores=[0.1, 0.2],
            reasoner_error_preds=[0],
            carnot_error_scores=[0.1],
            carnot_error_preds=[0],
            bootstrap_seed=7,
            bootstrap_resamples=1000,
        )

    assert "MOAT_SURVIVES" in exp.classify_verdict(
        harness_fixture_auroc=0.91,
        carnot_ensemble_auroc=0.8,
        strong_metrics=_metrics(reasoner_self_verify_auroc=0.50),
    )
    assert exp.classify_verdict(
        harness_fixture_auroc=0.6,
        carnot_ensemble_auroc=0.8,
        strong_metrics=_metrics(),
    ).endswith("harness_fixture_auroc")
    assert exp.classify_verdict(
        harness_fixture_auroc=0.91,
        carnot_ensemble_auroc=0.64,
        strong_metrics=_metrics(carnot_ensemble_auroc=0.64),
    ).endswith("carnot_ensemble_auroc")
    assert exp.classify_verdict(
        harness_fixture_auroc=0.91,
        carnot_ensemble_auroc=0.8,
        strong_metrics=_metrics(n_residual_errors=29),
    ).endswith("n_residual_errors_lt30")
    assert "MOAT_SUBSUMED" in exp.classify_verdict(
        harness_fixture_auroc=0.91,
        carnot_ensemble_auroc=0.8,
        strong_metrics=_metrics(
            residual_catch_rate=0.21,
            residual_catch_ci95={**_metrics().residual_catch_ci95, "mean": 0.21, "low": 0.15, "high": 0.29},
        ),
    )
    assert "MOAT_SUBSUMED" in exp.classify_verdict(
        harness_fixture_auroc=0.91,
        carnot_ensemble_auroc=0.8,
        strong_metrics=_metrics(error_overlap_jaccard=0.71),
    )
    assert exp.classify_verdict(
        harness_fixture_auroc=0.91,
        carnot_ensemble_auroc=0.8,
        strong_metrics=_metrics(
            residual_catch_rate=0.4,
            residual_catch_ci95={**_metrics().residual_catch_ci95, "low": 0.31, "high": 0.49},
            error_overlap_jaccard=0.65,
        ),
    ).endswith("boundary_gate")


def test_req_verify_3904_artifact_builder_uses_required_bare_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-3904: artifact fields are bare values with principle strings."""

    artifact = exp.build_artifact_from_metrics(
        weak_metrics=_metrics(residual_catch_rate=0.7, reasoner_self_verify_auroc=0.546),
        strong_metrics=_metrics(reasoner_self_verify_auroc=0.50),
        config=exp.ExperimentConfig(repo_root=tmp_path, started_at=10.0, clock=lambda: 75.0),
        preconditions_checked=[exp.PreconditionCheck("cuda_available", True, "ok")],
        model_specs={"hf_id": "fixture", "model_path": "fixture.gguf"},
        harness_source={"harness_module_path": exp.HARNESS_MODULE_PATH, "fixture_auroc": 0.91},
        corpus_source={"corpus_path": "data/in_distribution_error_corpus_v1.json"},
        panel_sha256="p" * 64,
        weak_reasoner_error_scores=[0.9, 0.1, 0.1],
        strong_reasoner_error_scores=[0.8, 0.1, 0.1],
        carnot_error_scores=[0.9, 0.8, 0.1],
        per_step_results=[{"question_id": "incorrect-0"}],
    )

    exp.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete: moat_scissor_MOAT_SURVIVES")
    assert artifact["harness_fixture_auroc"] == 0.91
    assert artifact["reasoner_auroc_weak"] == pytest.approx(0.546)
    assert artifact["reasoner_auroc_strong"] == pytest.approx(0.50)
    assert artifact["duration_s"] == 65.0
    assert len(artifact["reproducibility_checksum"]) == 64
    assert set(exp.REQUIRED_PRINCIPLE_FIELDS) <= set(artifact["field_principles"])
    assert all(isinstance(artifact["field_principles"][field], str) for field in exp.REQUIRED_PRINCIPLE_FIELDS)


def test_req_verify_3904_fake_dual_arm_scoring_builds_artifact_without_live_llm(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3904: injected weak/strong scorers let unit tests avoid live GGUF inference."""

    _write_exp3884_fixture(tmp_path, n_per_class=100)
    _write_exp3894_fixture(tmp_path)
    panel = exp.load_exp3884_panel(tmp_path)
    harness_source = exp.load_exp3894_harness_source(tmp_path)

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
        weak_reasoner_scorer=lambda selected_panel, _model_specs: _scoring(selected_panel, n_caught=40),
        strong_reasoner_scorer=lambda selected_panel, _model_specs: _scoring(selected_panel, n_caught=50),
        write=True,
    )

    assert (tmp_path / exp.OUTPUT_REL_PATH).exists()
    assert artifact["n_items"] == 200
    assert artifact["n_residual_errors_weak"] == 60
    assert artifact["n_residual_errors_strong"] == 50
    assert artifact["honest_verdict"].startswith("complete: moat_scissor_MOAT_SURVIVES")
    assert artifact["per_step_results"][0]["reasoner_weak_rejects"] is True
    assert artifact["per_step_results"][41]["reasoner_weak_rejects"] is False
    assert artifact["per_step_results"][49]["reasoner_strong_rejects"] is True
    assert artifact["per_step_results"][-1]["carnot_rejects"] is False


def test_req_verify_3904_validate_and_blocked_artifact_failures(tmp_path: Path) -> None:
    """REQ-VERIFY-3904: blocked artifacts stay terminal and schema validation is strict."""

    valid = exp.build_blocked_artifact(
        reason="blocked_no_cuda",
        preconditions_checked=[],
        duration_s=0.1,
    )
    exp.validate_artifact(valid)
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
        ("model_path", "blocked_model_not_cached"),
        ("carnot_verify_import", "blocked_carnot_verify_import"),
        ("llama_cpp_import", "blocked_llama_cpp_not_installed"),
        ("exp3894_harness_ready", "blocked_upstream_harness_not_ready"),
        ("exp3884_corpus_in_band", "blocked_upstream_corpus_not_in_band"),
    ],
)
def test_req_verify_3904_probe_preconditions_block_reasons(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    false_resource: str,
    expected_reason: str,
) -> None:
    """REQ-VERIFY-3904: resource failures map to terminal blocked verdicts."""

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


def test_req_verify_3904_run_experiment_orchestrates_blocked_success_and_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3904: run_experiment writes blocked, success, or inference-failed artifacts."""

    blocked_preflight = exp.PreflightResult(
        checks=(exp.PreconditionCheck("cuda", False, "no"),),
        blocked_reason="blocked_no_cuda",
        model_specs={"hf_id": "fixture"},
        panel=None,
        harness_source=None,
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
        exp.ExperimentConfig(repo_root=tmp_path, output_path=tmp_path / "failed.json", started_at=0.0, clock=lambda: 2.0),
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


def test_req_verify_3904_cli_main_reports_terminal_status(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3904: CLI adapter reports the written path and blocked status."""

    monkeypatch.setattr(exp, "run_experiment", lambda _config, write: {"honest_verdict": "complete: fixture"})
    assert exp.cli_main(["--repo-root", str(tmp_path)]) == 0
    assert exp.OUTPUT_REL_PATH.name in capsys.readouterr().out

    monkeypatch.setattr(exp, "run_experiment", lambda _config, write: {"honest_verdict": "blocked_no_cuda"})
    assert exp.cli_main(["--repo-root", str(tmp_path), "--output-path", str(tmp_path / "out.json")]) == 1
    assert "blocked_no_cuda" in capsys.readouterr().out
