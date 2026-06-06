"""Tests for Exp 3885 in-distribution moat scissor.

Spec refs: REQ-VERIFY-3885, SCENARIO-VERIFY-3885,
SCENARIO-VERIFY-3885-BLOCKED.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import moat_scissor_in_distribution as exp


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


def _metrics(**overrides: Any) -> exp.ScissorMetrics:
    base = {
        "residual_catch_rate": 0.62,
        "residual_catch_ci95": {
            "mean": 0.62,
            "low": 0.51,
            "high": 0.72,
            "n_resamples": 1000,
            "bootstrap_seed": 3885,
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


def test_req_verify_3885_spec_anchor_exists() -> None:
    """REQ-VERIFY-3885: the in-distribution scissor is OpenSpec anchored."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-VERIFY-3885" in spec
    assert "SCENARIO-VERIFY-3885" in spec
    assert exp.OUTPUT_REL_PATH.as_posix() in spec
    assert "blocked_upstream_corpus_not_in_band" in spec


def test_req_verify_3885_loads_exp3884_corpus_and_scores(tmp_path: Path) -> None:
    """REQ-VERIFY-3885: exp3884 rows and per-item scores are loaded from disk."""

    rows = _write_exp3884_fixture(tmp_path, n_per_class=3)

    panel = exp.load_exp3884_panel(tmp_path, min_incorrect=3)

    assert [row["corpus_item_id"] for row in panel.rows] == [row["corpus_item_id"] for row in rows]
    assert panel.labels == (1, 1, 1, 0, 0, 0)
    assert panel.carnot_error_scores == (0.9, 0.9, 0.9, 0.1, 0.1, 0.1)
    assert panel.carnot_error_preds == (1, 1, 1, 0, 0, 0)
    assert panel.corpus_source["corpus_path"] == "data/in_distribution_error_corpus_v1.json"
    assert panel.corpus_source["n_incorrect_steps"] == 3
    assert len(panel.panel_sha256) == 64


def test_req_verify_3885_upstream_loader_fails_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-3885: malformed or out-of-band exp3884 inputs are not scored."""

    assert exp._json_rows([{"a": 1}, []]) == [{"a": 1}]
    assert exp._json_rows({"items": [{"b": 2}]}) == [{"b": 2}]
    assert exp._json_rows("bad") == []
    assert exp._resolve_repo_path(tmp_path, "/outside/file.json") == Path("/outside/file.json")
    assert exp._relative_to_repo(tmp_path, Path("/outside/file.json")) == "/outside/file.json"
    valid_row = _row(0, "incorrect")
    valid_score = _score_for_row(0, valid_row)
    with pytest.raises(ValueError, match="missing carnot_rejects"):
        exp._validate_alignment(valid_row, {k: v for k, v in valid_score.items() if k != "carnot_rejects"}, 0)
    with pytest.raises(ValueError, match="missing label or step_text"):
        exp._validate_alignment({"label": "incorrect"}, valid_score, 0)
    with pytest.raises(ValueError, match="empty step_text"):
        exp._validate_alignment({**valid_row, "step_text": ""}, valid_score, 0)
    with pytest.raises(ValueError, match="label mismatch"):
        exp._validate_alignment(valid_row, {**valid_score, "label": "correct"}, 0)
    with pytest.raises(ValueError, match="corpus_item_id"):
        exp._validate_alignment(valid_row, {**valid_score, "corpus_item_id": "other"}, 0)
    with pytest.raises(FileNotFoundError):
        exp.load_exp3884_panel(tmp_path, min_incorrect=1)

    artifact_path = tmp_path / "results" / "experiment_3884_in_distribution_error_rich_corpus.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="not a JSON object"):
        exp.load_exp3884_panel(tmp_path, min_incorrect=1)

    _write_exp3884_fixture(tmp_path, n_per_class=3, recorded_auroc=0.64)
    with pytest.raises(ValueError, match="carnot_ensemble_auroc_on_corpus"):
        exp.load_exp3884_panel(tmp_path, min_incorrect=3)

    _write_exp3884_fixture(tmp_path, n_per_class=2, recorded_auroc=0.9)
    with pytest.raises(ValueError, match="incorrect"):
        exp.load_exp3884_panel(tmp_path, min_incorrect=3)

    _write_exp3884_fixture(tmp_path, n_per_class=3, recorded_auroc=0.9, flagged_adversarial=True)
    with pytest.raises(ValueError, match="flagged_adversarial"):
        exp.load_exp3884_panel(tmp_path, min_incorrect=3)

    _write_exp3884_fixture(tmp_path, n_per_class=3, recorded_auroc=0.9)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["corpus_path"] = "data/missing.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="score path"):
        exp.load_exp3884_panel(tmp_path, min_incorrect=3)

    _write_exp3884_fixture(tmp_path, n_per_class=3, recorded_auroc=0.9)
    scores_path = tmp_path / "results" / "experiment_3884_in_distribution_error_rich_corpus_scores.json"
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    scores["items"].pop()
    scores_path.write_text(json.dumps(scores), encoding="utf-8")
    with pytest.raises(ValueError, match="length mismatch"):
        exp.load_exp3884_panel(tmp_path, min_incorrect=3)

    rows = _write_exp3884_fixture(tmp_path, n_per_class=3, recorded_auroc=0.9)
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    scores["items"][0]["step_text_sha256"] = "bad"
    scores_path.write_text(json.dumps(scores), encoding="utf-8")
    with pytest.raises(ValueError, match="step_text_sha256"):
        exp.load_exp3884_panel(tmp_path, min_incorrect=3)
    assert rows


def test_scenario_verify_3885_terminal_gates() -> None:
    """SCENARIO-VERIFY-3885: verdicts follow the in-distribution falsification gate."""

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
    assert exp.classify_verdict(_metrics(n_residual_errors=29)).endswith("n_residual_errors_lt30")
    assert exp.classify_verdict(
        _metrics(
            residual_catch_rate=0.4,
            residual_catch_ci95={**_metrics().residual_catch_ci95, "low": 0.31, "high": 0.49},
            error_overlap_jaccard=0.65,
        )
    ).endswith("boundary_gate")


def test_req_verify_3885_artifact_builder_uses_bare_fields_and_string_principles(tmp_path: Path) -> None:
    """REQ-VERIFY-3885: artifacts carry bare values, principle strings, and checksum."""

    artifact = exp.build_artifact_from_metrics(
        metrics=_metrics(),
        config=exp.ExperimentConfig(repo_root=tmp_path, started_at=10.0, clock=lambda: 75.0),
        preconditions_checked=[exp.PreconditionCheck("cuda_available", True, "ok")],
        model_specs={"hf_id": "fixture", "model_path": "fixture.gguf"},
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


def test_req_verify_3885_validate_artifact_failures(tmp_path: Path) -> None:
    """REQ-VERIFY-3885: schema validation rejects non-terminal or wrapped principles."""

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


def test_req_verify_3885_fake_scoring_builds_artifact_without_live_llm(tmp_path: Path) -> None:
    """REQ-VERIFY-3885: injected reasoner scoring lets unit tests avoid live GGUF inference."""

    _write_exp3884_fixture(tmp_path, n_per_class=100)
    panel = exp.load_exp3884_panel(tmp_path)

    def fake_reasoner(
        selected_panel: exp.Exp3884Panel,
        _model_specs: dict[str, object],
    ) -> exp.ReasonerScoring:
        scores = [
            1 if label == 1 and idx < 50 else 0
            for idx, label in enumerate(selected_panel.labels)
        ]
        return exp.ReasonerScoring(
            raw_responses=["NO" if score else "YES" for score in scores],
            error_scores=scores,
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
        reasoner_scorer=fake_reasoner,
        write=True,
    )

    assert (tmp_path / exp.OUTPUT_REL_PATH).exists()
    assert artifact["n_items"] == 200
    assert artifact["n_residual_errors"] == 50
    assert artifact["honest_verdict"].startswith("complete: moat_scissor_indist_MOAT_SURVIVES")
    assert artifact["per_step_results"][0]["reasoner_rejects"] is True
    assert artifact["per_step_results"][-1]["carnot_rejects"] is False


def test_req_verify_3885_prompt_parser_and_fake_llama_adapter(tmp_path: Path) -> None:
    """REQ-VERIFY-3885: the live adapter preserves Exp3827 YES/NO semantics."""

    _write_exp3884_fixture(tmp_path, n_per_class=1)
    panel = exp.load_exp3884_panel(tmp_path, min_incorrect=1)

    class FakeLlama:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def __call__(self, prompt: str, **kwargs: object) -> dict[str, object]:
            assert "Answer strictly YES if it is correct, and NO if it contains an error." in prompt
            assert kwargs["max_tokens"] == 3
            return {"choices": [{"text": " NO"}]}

    scoring = exp.score_reasoner_with_llama_cpp(
        panel,
        {"model_path": "fixture.gguf"},
        max_tokens=3,
        llama_factory=FakeLlama,
    )

    assert scoring.raw_responses == ("NO", "NO")
    assert scoring.error_scores == (1, 1)
    assert exp.parse_reasoner_error_score("yes") == 0
    assert "Step: bad step" in exp.reasoner_self_verify_prompt("bad step")


@pytest.mark.parametrize(
    ("false_resource", "expected_reason"),
    [
        ("cuda_available", "blocked_no_cuda"),
        ("carnot_verify_import", "blocked_carnot_verify_import"),
        ("model_path", "blocked_model_not_cached"),
        ("llama_cpp_import", "blocked_llama_cpp_not_installed"),
        ("exp3884_corpus_in_band", "blocked_upstream_corpus_not_in_band"),
    ],
)
def test_req_verify_3885_probe_preconditions_block_reasons(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    false_resource: str,
    expected_reason: str,
) -> None:
    """REQ-VERIFY-3885: resource failures map to terminal blocked verdicts."""

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

    preflight = exp.probe_preconditions(
        exp.ExperimentConfig(repo_root=tmp_path),
        command_runner=fake_runner,
    )

    assert preflight.blocked_reason == expected_reason


def test_req_verify_3885_model_resolution_prefers_qwen_then_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3885: Qwen is preferred and Gemma fallback is auditable."""

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


def test_req_verify_3885_probe_cuda_exception_is_recorded() -> None:
    """REQ-VERIFY-3885: CUDA probe exceptions become precondition evidence."""

    def raising_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise RuntimeError("runner boom")

    check = exp._probe_cuda_with_venv(
        exp.ExperimentConfig(repo_root=Path("/tmp")),
        command_runner=raising_runner,
    )
    assert check.available is False
    assert "runner boom" in check.detail


def test_req_verify_3885_run_experiment_orchestrates_blocked_success_and_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3885: run_experiment writes blocked, success, or inference-failed artifacts."""

    blocked_preflight = exp.PreflightResult(
        checks=(exp.PreconditionCheck("cuda", False, "no"),),
        blocked_reason="blocked_no_cuda",
        model_specs={"hf_id": "fixture"},
        panel=None,
    )
    monkeypatch.setattr(exp, "probe_preconditions", lambda _config: blocked_preflight)
    blocked = exp.run_experiment(
        exp.ExperimentConfig(repo_root=tmp_path, output_path=tmp_path / "blocked.json", started_at=0.0, clock=lambda: 1.0),
        write=True,
    )
    assert blocked["honest_verdict"] == "blocked_no_cuda"
    assert (tmp_path / "blocked.json").exists()

    _write_exp3884_fixture(tmp_path, n_per_class=100)
    panel = exp.load_exp3884_panel(tmp_path)
    success_preflight = exp.PreflightResult(
        checks=(exp.PreconditionCheck("ok", True, "yes"),),
        blocked_reason=None,
        model_specs={"hf_id": "fixture", "model_path": "fixture.gguf"},
        panel=panel,
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
    assert failed["preconditions_checked"][-1]["resource"] == "llama_cpp_inference"

    no_panel_preflight = exp.PreflightResult(
        checks=(exp.PreconditionCheck("corpus", True, "missing panel"),),
        blocked_reason=None,
        model_specs={"hf_id": "fixture", "model_path": "fixture.gguf"},
        panel=None,
    )
    monkeypatch.setattr(exp, "probe_preconditions", lambda _config: no_panel_preflight)
    no_panel = exp.run_experiment(
        exp.ExperimentConfig(repo_root=tmp_path, output_path=tmp_path / "no_panel.json"),
        write=True,
    )
    assert no_panel["honest_verdict"] == "blocked_upstream_corpus_not_in_band"


def test_req_verify_3885_cli_main_reports_terminal_status(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3885: CLI adapter reports the written path and blocked status."""

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
