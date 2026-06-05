"""Tests for Exp 3869 existing-corpus moat scissor v4.

Spec refs: REQ-VERIFY-3869, SCENARIO-VERIFY-3869,
SCENARIO-VERIFY-3869-BLOCKED.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import moat_scissor_v4_existing_corpus as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"


def _rows(n_per_class: int = 500) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in range(n_per_class):
        rows.append(
            {
                "question_id": f"bad-{idx}",
                "question": f"Question bad {idx}",
                "step_text": f"Incorrect step {idx}: 2 + 2 = 5.",
                "label": "incorrect",
                "error_axis": "arithmetic",
                "source": "prmbench",
            }
        )
    for idx in range(n_per_class):
        rows.append(
            {
                "question_id": f"ok-{idx}",
                "question": f"Question ok {idx}",
                "step_text": f"Correct step {idx}: 2 + 2 = 4.",
                "label": "correct",
                "error_axis": "none",
                "source": "prmbench",
            }
        )
    return rows


def test_req_verify_3869_spec_anchor_exists() -> None:
    """REQ-VERIFY-3869: the existing-corpus scissor has OpenSpec coverage."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-VERIFY-3869" in spec
    assert "SCENARIO-VERIFY-3869" in spec
    assert exp.OUTPUT_REL_PATH.as_posix() in spec
    assert "generation-headroom gate" in spec


def test_req_verify_3869_loads_existing_corpus_without_sampling(tmp_path: Path) -> None:
    """REQ-VERIFY-3869: the PRMBench corpus is loaded as-is, not rebuilt."""

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    fixture_rows = _rows(n_per_class=3)
    path = data_dir / "step_error_balanced_v2.json"
    path.write_text(json.dumps({"items": fixture_rows}), encoding="utf-8")

    source_path, loaded = exp.load_existing_corpus_rows(tmp_path, min_incorrect=3)
    panel = exp.build_existing_corpus_panel(loaded, source_path)

    assert source_path == path
    assert [row["question_id"] for row in panel.rows] == [row["question_id"] for row in fixture_rows]
    assert panel.labels == (1, 1, 1, 0, 0, 0)
    assert panel.corpus_source["path"] == "data/step_error_balanced_v2.json"
    assert panel.corpus_source["n_incorrect_steps"] == 3
    assert panel.corpus_source["source_values"] == ["prmbench"]


def test_scenario_verify_3869_metrics_and_terminal_gates() -> None:
    """SCENARIO-VERIFY-3869: residual catch, overlap, and gates match the task."""

    survives = exp.ScissorMetrics(
        residual_catch_rate=0.62,
        residual_catch_ci95={
            "mean": 0.62,
            "low": 0.51,
            "high": 0.72,
            "n_resamples": 1000,
            "bootstrap_seed": 3869,
        },
        error_overlap_jaccard=0.5,
        reasoner_self_verify_auroc=0.72,
        carnot_ensemble_auroc=0.66,
        n_items=1000,
        n_residual_errors=120,
        n_gold_incorrect=500,
        reasoner_caught_error_indices=(1, 2),
        carnot_caught_error_indices=(2, 3),
    )
    subsumed = exp.ScissorMetrics(
        residual_catch_rate=0.21,
        residual_catch_ci95={
            "mean": 0.21,
            "low": 0.15,
            "high": 0.29,
            "n_resamples": 1000,
            "bootstrap_seed": 3869,
        },
        error_overlap_jaccard=0.55,
        reasoner_self_verify_auroc=0.72,
        carnot_ensemble_auroc=0.66,
        n_items=1000,
        n_residual_errors=120,
        n_gold_incorrect=500,
        reasoner_caught_error_indices=(1, 2),
        carnot_caught_error_indices=(2, 3),
    )
    reasoner_failed = exp.ScissorMetrics(
        residual_catch_rate=0.62,
        residual_catch_ci95=survives.residual_catch_ci95,
        error_overlap_jaccard=0.5,
        reasoner_self_verify_auroc=0.5,
        carnot_ensemble_auroc=0.66,
        n_items=1000,
        n_residual_errors=120,
        n_gold_incorrect=500,
        reasoner_caught_error_indices=(),
        carnot_caught_error_indices=(),
    )
    carnot_failed = exp.ScissorMetrics(
        residual_catch_rate=0.62,
        residual_catch_ci95=survives.residual_catch_ci95,
        error_overlap_jaccard=0.5,
        reasoner_self_verify_auroc=0.72,
        carnot_ensemble_auroc=0.64,
        n_items=1000,
        n_residual_errors=120,
        n_gold_incorrect=500,
        reasoner_caught_error_indices=(),
        carnot_caught_error_indices=(),
    )

    assert "MOAT_SURVIVES" in exp.classify_verdict(survives)
    assert "MOAT_SUBSUMED" in exp.classify_verdict(subsumed)
    assert exp.classify_verdict(reasoner_failed).endswith("reasoner_self_verify_auroc")
    assert exp.classify_verdict(carnot_failed).endswith("carnot_ensemble_auroc")
    assert exp.classify_verdict(
        exp.ScissorMetrics(
            **{
                **survives.__dict__,
                "reasoner_self_verify_auroc": 0.5,
                "carnot_ensemble_auroc": 0.64,
            }
        )
    ).endswith("reasoner_self_verify_auroc_and_carnot_ensemble_auroc")
    assert exp.classify_verdict(exp.ScissorMetrics(**{**survives.__dict__, "n_residual_errors": 29})).endswith(
        "n_residual_errors_lt30"
    )
    boundary = exp.ScissorMetrics(
        residual_catch_rate=0.4,
        residual_catch_ci95={
            "mean": 0.4,
            "low": 0.31,
            "high": 0.49,
            "n_resamples": 1000,
            "bootstrap_seed": 3869,
        },
        error_overlap_jaccard=0.65,
        reasoner_self_verify_auroc=0.72,
        carnot_ensemble_auroc=0.66,
        n_items=1000,
        n_residual_errors=120,
        n_gold_incorrect=500,
        reasoner_caught_error_indices=(),
        carnot_caught_error_indices=(),
    )
    assert exp.classify_verdict(boundary).endswith("boundary_gate")


def test_scenario_verify_3869_reuses_residual_and_bootstrap_definitions() -> None:
    """SCENARIO-VERIFY-3869: residual and Jaccard reuse the Exp3827 semantics."""

    metrics = exp.compute_scissor_metrics(
        labels=[1, 1, 1, 1, 0, 0],
        reasoner_error_scores=[1, 0, 0, 1, 0, 0],
        carnot_error_scores=[0.9, 0.8, 0.1, 0.2, 0.3, 0.4],
        carnot_error_preds=[1, 1, 0, 0, 0, 0],
        bootstrap_seed=7,
        bootstrap_resamples=1000,
    )

    assert metrics.n_residual_errors == 2
    assert metrics.residual_catch_rate == 0.5
    assert metrics.residual_catch_ci95["n_resamples"] == 1000
    assert metrics.error_overlap_jaccard == 1 / 3


def test_req_verify_3869_artifact_builder_includes_principles_and_checksum(tmp_path: Path) -> None:
    """REQ-VERIFY-3869: terminal artifacts carry required principle notes."""

    metrics = exp.ScissorMetrics(
        residual_catch_rate=1.0,
        residual_catch_ci95={
            "mean": 1.0,
            "low": 1.0,
            "high": 1.0,
            "n_resamples": 1000,
            "bootstrap_seed": 3869,
        },
        error_overlap_jaccard=0.5,
        reasoner_self_verify_auroc=0.75,
        carnot_ensemble_auroc=1.0,
        n_items=1000,
        n_residual_errors=250,
        n_gold_incorrect=500,
        reasoner_caught_error_indices=tuple(range(250)),
        carnot_caught_error_indices=tuple(range(500)),
    )
    artifact = exp.build_artifact_from_metrics(
        metrics=metrics,
        config=exp.ExperimentConfig(
            repo_root=tmp_path,
            started_at=10.0,
            clock=lambda: 75.0,
        ),
        preconditions_checked=[exp.PreconditionCheck("cuda", True, "device_count=2")],
        model_specs={"hf_id": "fixture", "model_path": "fixture.gguf"},
        corpus_source={"path": "data/step_error_balanced_v2.json", "primary_source": "prmbench"},
        panel_sha256="p" * 64,
        reasoner_error_scores=[1, 0, 1],
        carnot_error_scores=[0.9, 0.8, 0.1],
        per_step_results=[{"question_id": "bad-1", "reasoner_rejects": True}],
    )

    exp.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["duration_s"] == 65.0
    assert len(artifact["reproducibility_checksum"]) == 64
    assert "generation_headroom" not in artifact
    assert set(exp.REQUIRED_PRINCIPLE_FIELDS) <= set(artifact["field_principles"])
    assert all(
        "principle" in artifact["field_principles"][field]
        for field in exp.REQUIRED_PRINCIPLE_FIELDS
    )


def test_scenario_verify_3869_blocked_artifact_is_terminal_and_non_fabricated(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3869-BLOCKED: blocked output leaves moat metrics null."""

    output = tmp_path / exp.OUTPUT_REL_PATH
    artifact = exp.write_blocked_artifact(
        output,
        reason="blocked_no_cuda",
        preconditions_checked=[exp.PreconditionCheck("cuda", False, "no CUDA")],
        duration_s=0.5,
    )

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert artifact == persisted
    assert artifact["honest_verdict"] == "blocked_no_cuda"
    assert artifact["residual_catch_rate"] is None
    assert artifact["reasoner_self_verify_auroc"] is None
    assert artifact["n_items"] == 0
    exp.validate_artifact(artifact)


def test_req_verify_3869_fake_scoring_builds_artifact_without_live_llm(tmp_path: Path) -> None:
    """REQ-VERIFY-3869: injected scorers let tests avoid live GGUF inference."""

    source = tmp_path / "data" / "step_error_balanced_v2.json"
    panel = exp.build_existing_corpus_panel(_rows(), source)

    def fake_reasoner(
        selected_panel: exp.ExistingCorpusPanel,
        _model_specs: dict[str, object],
    ) -> exp.ReasonerScoring:
        scores = [1 if label == 1 and idx < 250 else 0 for idx, label in enumerate(selected_panel.labels)]
        return exp.ReasonerScoring(
            raw_responses=["NO" if score else "YES" for score in scores],
            error_scores=scores,
        )

    def fake_carnot(selected_panel: exp.ExistingCorpusPanel) -> exp.CarnotScoring:
        scores = [0.92 if label == 1 else 0.08 for label in selected_panel.labels]
        preds = [1 if score > 0.5 else 0 for score in scores]
        return exp.CarnotScoring(scores=scores, error_preds=preds, threshold=0.5)

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
        carnot_scorer=fake_carnot,
        write=True,
    )

    assert (tmp_path / exp.OUTPUT_REL_PATH).exists()
    assert artifact["n_items"] == 1000
    assert artifact["n_residual_errors"] == 250
    assert artifact["honest_verdict"].startswith("complete: moat_scissor_v4_MOAT_SURVIVES")
    assert artifact["per_step_results"][0]["reasoner_rejects"] is True


def test_req_verify_3869_prompt_parser_and_fake_llama_adapter() -> None:
    """REQ-VERIFY-3869: the live adapter preserves Exp3827 YES/NO semantics."""

    panel = exp.ExistingCorpusPanel(
        rows=({"question_id": "a", "step_text": "bad", "label": "incorrect"},),
        labels=(1,),
        texts=("bad step",),
        panel_sha256="p" * 64,
        corpus_source={"path": "fixture"},
    )

    class FakeLlama:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def __call__(self, prompt: str, **kwargs: object) -> dict[str, object]:
            assert "Answer strictly YES" in prompt
            assert kwargs["max_tokens"] == 3
            return {"choices": [{"text": " NO"}]}

    scoring = exp.score_reasoner_with_llama_cpp(
        panel,
        {"model_path": "fixture.gguf"},
        max_tokens=3,
        llama_factory=FakeLlama,
    )

    assert scoring.raw_responses == ("NO",)
    assert scoring.error_scores == (1,)
    assert scoring.error_preds == (1,)
    assert exp.parse_reasoner_error_score("yes") == 0
    assert "Step: bad step" in exp.reasoner_self_verify_prompt("bad step")


def test_req_verify_3869_carnot_ensemble_uses_exp2837_aggregation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3869: Carnot scoring is 0.9*tier0r + 0.1*tier0u + memory."""

    panel = exp.ExistingCorpusPanel(
        rows=(
            {"question_id": "a", "step_text": "x", "label": "incorrect"},
            {"question_id": "b", "step_text": "y", "label": "correct"},
            {"question_id": "c", "step_text": "z", "label": "incorrect"},
        ),
        labels=(1, 0, 1),
        texts=("x", "y", "z"),
        panel_sha256="p" * 64,
        corpus_source={"path": "fixture"},
    )
    monkeypatch.setattr(
        exp,
        "_score_text_verifiers",
        lambda _texts: {
            "tier0r_curry_howard": [1.0, 0.0, 0.2],
            "tier0u_logical_consistency": [0.0, 1.0, 0.2],
        },
    )
    monkeypatch.setattr(exp, "_load_fr11_memory_index", lambda _root: {"fixture": True})
    monkeypatch.setattr(exp, "_fr11_memory_score", lambda row, _index: 0.5 if row["question_id"] == "c" else 0.0)

    scoring = exp.score_carnot_ensemble(panel, tmp_path)

    assert scoring.scores == (0.9, 0.1, 0.7)
    assert scoring.threshold == 0.7
    assert scoring.error_preds == (1, 0, 0)


@pytest.mark.parametrize(
    ("false_resource", "expected_reason"),
    [
        ("cuda_available", "blocked_no_cuda"),
        ("carnot_verify_import", "blocked_carnot_verify_import"),
        ("model_path", "blocked_model_not_cached_qwen3.6_35b"),
        ("llama_cpp_import", "blocked_llama_cpp_not_installed"),
        ("step_error_balanced_v2_corpus", "blocked_corpus_missing"),
    ],
)
def test_req_verify_3869_probe_preconditions_block_reasons(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    false_resource: str,
    expected_reason: str,
) -> None:
    """REQ-VERIFY-3869: precondition failures map to terminal blocked verdicts."""

    def fake_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        available = false_resource != "cuda_available"
        return subprocess.CompletedProcess(command, 0 if available else 1, "ok", "" if available else "no cuda")

    monkeypatch.setattr(exp.importlib, "import_module", lambda _name: object())
    monkeypatch.setattr(
        exp,
        "_resolve_reasoner_model",
        lambda: (
            {"hf_id": "fixture", "model_path": None if false_resource == "model_path" else "fixture.gguf"},
            [exp.PreconditionCheck("qwen3.6_35b_gguf_cached", false_resource != "model_path", "fixture")],
        ),
    )
    if false_resource == "carnot_verify_import":
        monkeypatch.setattr(
            exp.importlib,
            "import_module",
            lambda name: (_ for _ in ()).throw(RuntimeError("boom")) if name == "carnot.verify" else object(),
        )
    if false_resource == "llama_cpp_import":
        monkeypatch.setattr(
            exp.importlib,
            "import_module",
            lambda name: (_ for _ in ()).throw(RuntimeError("boom")) if name == "llama_cpp" else object(),
        )

    if false_resource != "step_error_balanced_v2_corpus":
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "step_error_balanced_v2.json").write_text(json.dumps({"items": _rows(100)}), encoding="utf-8")

    preflight = exp.probe_preconditions(
        exp.ExperimentConfig(repo_root=tmp_path),
        command_runner=fake_runner,
    )

    assert preflight.blocked_reason == expected_reason


def test_req_verify_3869_model_resolution_prefers_qwen_then_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3869: Qwen is preferred and Gemma is recorded as fallback."""

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


def test_req_verify_3869_run_experiment_orchestrates_blocked_and_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3869: run_experiment writes blocked or selected-panel output."""

    blocked_preflight = exp.PreflightResult(
        checks=(exp.PreconditionCheck("cuda", False, "no"),),
        blocked_reason="blocked_no_cuda",
        model_specs={"hf_id": "fixture"},
        corpus_path=None,
        rows=(),
    )
    monkeypatch.setattr(exp, "probe_preconditions", lambda _config: blocked_preflight)
    blocked = exp.run_experiment(
        exp.ExperimentConfig(repo_root=tmp_path, output_path=tmp_path / "blocked.json", started_at=0.0, clock=lambda: 1.0),
        write=True,
    )
    assert blocked["honest_verdict"] == "blocked_no_cuda"
    assert (tmp_path / "blocked.json").exists()

    success_preflight = exp.PreflightResult(
        checks=(exp.PreconditionCheck("ok", True, "yes"),),
        blocked_reason=None,
        model_specs={"hf_id": "fixture", "model_path": "fixture.gguf"},
        corpus_path=tmp_path / "data" / "step_error_balanced_v2.json",
        rows=tuple(_rows()),
    )
    monkeypatch.setattr(exp, "probe_preconditions", lambda _config: success_preflight)
    monkeypatch.setattr(
        exp,
        "build_artifact_for_panel",
        lambda panel, **kwargs: {"honest_verdict": "complete: fixture", "n_items": len(panel.rows), "kwargs": sorted(kwargs)},
    )
    success = exp.run_experiment(exp.ExperimentConfig(repo_root=tmp_path), write=False)
    assert success["honest_verdict"] == "complete: fixture"
    assert success["n_items"] == 1000

    no_path_preflight = exp.PreflightResult(
        checks=(exp.PreconditionCheck("corpus", True, "missing path"),),
        blocked_reason=None,
        model_specs={"hf_id": "fixture", "model_path": "fixture.gguf"},
        corpus_path=None,
        rows=tuple(_rows()),
    )
    monkeypatch.setattr(exp, "probe_preconditions", lambda _config: no_path_preflight)
    no_path = exp.run_experiment(
        exp.ExperimentConfig(repo_root=tmp_path, output_path=tmp_path / "no_path.json"),
        write=True,
    )
    assert no_path["honest_verdict"] == "blocked_corpus_missing"
    assert (tmp_path / "no_path.json").exists()


def test_req_verify_3869_helper_validation_failures_are_deterministic(tmp_path: Path) -> None:
    """REQ-VERIFY-3869: validation rejects non-terminal or headroom artifacts."""

    assert exp._json_rows([{"a": 1}, []]) == [{"a": 1}]
    assert exp._json_rows("bad") == []
    assert exp._relative_data_path(Path("/tmp/outside.json")) == "/tmp/outside.json"
    with pytest.raises(ValueError, match="missing required keys"):
        exp._validate_existing_row({"label": "correct"}, 0)
    with pytest.raises(ValueError, match="empty step_text"):
        exp._validate_existing_row(
            {
                "error_axis": "axis",
                "label": "correct",
                "question": "q",
                "question_id": "qid",
                "source": "prmbench",
                "step_text": "",
            },
            0,
        )

    def raising_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise RuntimeError("runner boom")

    cuda_check = exp._probe_cuda_with_venv(
        exp.ExperimentConfig(repo_root=Path("/tmp")),
        command_runner=raising_runner,
    )
    assert cuda_check.available is False
    assert "runner boom" in cuda_check.detail

    artifact = exp.build_blocked_artifact(
        reason="blocked_fixture",
        preconditions_checked=[],
        duration_s=0.0,
    )
    broken = dict(artifact)
    broken.pop("honest_verdict")
    with pytest.raises(ValueError, match="missing required"):
        exp.validate_artifact(broken)
    broken = dict(artifact)
    broken["honest_verdict"] = "not_terminal"
    with pytest.raises(ValueError, match="terminal prefix"):
        exp.validate_artifact(broken)
    broken = dict(artifact)
    broken["field_principles"] = []
    with pytest.raises(ValueError, match="field_principles"):
        exp.validate_artifact(broken)
    broken = dict(artifact)
    broken["field_principles"] = dict(exp.FIELD_PRINCIPLES)
    broken["field_principles"]["n_items"] = {}
    with pytest.raises(ValueError, match="principle"):
        exp.validate_artifact(broken)
    broken = dict(artifact)
    broken["generation_headroom"] = True
    with pytest.raises(ValueError, match="generation headroom"):
        exp.validate_artifact(broken)
    with pytest.raises(FileNotFoundError):
        exp.load_existing_corpus_rows(Path("/definitely/missing"), min_incorrect=100)

    tmp_root = tmp_path / "min-incorrect"
    data_dir = tmp_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "step_error_balanced_v2.json").write_text(json.dumps({"items": _rows(1)}), encoding="utf-8")
    with pytest.raises(ValueError, match="required>=2"):
        exp.load_existing_corpus_rows(tmp_root, min_incorrect=2)
