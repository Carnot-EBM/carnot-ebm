"""Tests for Exp 3928 independent-corpus moat scissor replication.

Spec refs: REQ-VERIFY-3928, SCENARIO-VERIFY-3928,
SCENARIO-VERIFY-3928-BLOCKED.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import moat_scissor_replication_3928 as exp3928


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_exp3915_fixture(
    root: Path,
    *,
    unit_test_passed: bool = True,
    smoke_tokens: int = 1,
) -> None:
    module_path = root / exp3928.GGUF_HARNESS_MODULE_PATH
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text("# fixture robust gguf harness\n", encoding="utf-8")
    _write_json(
        root / exp3928.EXP3915_ARTIFACT_REL_PATH,
        {
            "honest_verdict": "complete: gguf_inference_harness_READY_fixture",
            "harness_module_path": exp3928.GGUF_HARNESS_MODULE_PATH,
            "model_used": "gemma-4-26B-A4B-it",
            "n_gpu_layers_used": -1,
            "smoke_tokens": smoke_tokens,
            "unit_test_passed": unit_test_passed,
            "model_specs": {"model_used": "gemma-4-26B-A4B-it"},
        },
    )


def _write_fover_fixture(root: Path) -> None:
    rows = [
        {"question_id": "q0", "step_text": "2 + 2 = 4", "label": "correct"},
        {"question_id": "q1", "step_text": "2 + 2 = 5", "label": "incorrect"},
        {"question_id": "q2", "step_text": "3 * 3 = 9", "label": "correct"},
        {"question_id": "q3", "step_text": "3 * 3 = 8", "label": "incorrect"},
    ]
    _write_json(root / "data" / "fover_test_v4.json", rows)


def _metrics(**overrides: Any) -> exp3928.ScissorMetrics:
    base = {
        "residual_catch_rate": 0.62,
        "residual_catch_ci95": {
            "mean": 0.62,
            "low": 0.51,
            "high": 0.72,
            "n_resamples": 1000,
            "bootstrap_seed": 3928,
        },
        "error_overlap_jaccard": 0.5,
        "reasoner_self_verify_auroc": 0.7,
        "carnot_ensemble_auroc": 0.8,
        "n_items": 120,
        "n_residual_errors": 50,
        "n_gold_incorrect": 60,
        "reasoner_caught_error_indices": tuple(range(10)),
        "carnot_caught_error_indices": tuple(range(60)),
    }
    base.update(overrides)
    return exp3928.ScissorMetrics(**base)


def _panel(n_items: int = 4) -> exp3928.IndependentPanel:
    rows = tuple(
        {
            "corpus_item_id": f"row-{idx}",
            "question_id": f"q-{idx}",
            "step_text": f"step {idx}",
            "label": "incorrect" if idx % 2 else "correct",
        }
        for idx in range(n_items)
    )
    return exp3928.IndependentPanel(
        rows=rows,
        labels=tuple(1 if idx % 2 else 0 for idx in range(n_items)),
        texts=tuple(str(row["step_text"]) for row in rows),
        corpus_used="processbench_slice",
        panel_sha256="p" * 64,
        corpus_source={"dataset": "fixture"},
    )


def _energy(n_items: int = 4) -> exp3928.EnergyScoring:
    return exp3928.EnergyScoring(
        scores=tuple(0.9 if idx % 2 else 0.1 for idx in range(n_items)),
        error_preds=tuple(1 if idx % 2 else 0 for idx in range(n_items)),
        threshold=0.5,
    )


def _reasoner(n_items: int = 4) -> exp3928.SelfVerifyArmScoring:
    return exp3928.SelfVerifyArmScoring(
        raw_responses=tuple("incorrect" if idx == 1 else "correct" for idx in range(n_items)),
        error_scores=tuple(0.9 if idx == 1 else 0.1 for idx in range(n_items)),
        error_preds=tuple(1 if idx == 1 else 0 for idx in range(n_items)),
        parsed_count=n_items,
        unparsed_count=0,
        parser_constant_prediction=False,
    )


def test_req_verify_3928_spec_anchor_exists() -> None:
    """REQ-VERIFY-3928: the independent-corpus replication is OpenSpec anchored."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-3928" in spec
    assert "SCENARIO-VERIFY-3928" in spec
    assert "moat_scissor_replication_3928.py" in spec
    assert exp3928.OUTPUT_REL_PATH.as_posix() in spec


def test_req_verify_3928_processbench_first_error_rows_normalize_to_step_panel() -> None:
    """REQ-VERIFY-3928: ProcessBench first-error labels become per-step gold labels."""

    rows = [
        {
            "id": "gsm8k-0",
            "generator": "fixture",
            "problem": "How many?",
            "steps": ["start correct", "first wrong", "after wrong"],
            "label": 1,
        },
        {
            "id": "gsm8k-1",
            "generator": "fixture",
            "problem": "How many now?",
            "steps": ["wrong immediately", "later"],
            "label": 0,
        },
    ]

    panel = exp3928.panel_from_processbench_rows(rows, source_detail="fixture", min_problem_items=2)

    assert panel.corpus_used == "processbench_slice"
    assert panel.labels == (0, 1, 0, 1, 0)
    assert [row["corpus_item_id"] for row in panel.rows] == [
        "gsm8k-0:step0",
        "gsm8k-0:step1",
        "gsm8k-0:step2",
        "gsm8k-1:step0",
        "gsm8k-1:step1",
    ]
    assert panel.corpus_source["processbench_problem_items"] == 2


def test_req_verify_3928_processbench_loader_and_invalid_row_branches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3928: ProcessBench loader tries math configs and rejects malformed labels."""

    assert exp3928.ExperimentConfig(repo_root=tmp_path).venv_python() == tmp_path / ".venv" / "bin" / "python"
    assert exp3928._json_rows({"rows": [{"x": 1}]}) == [{"x": 1}]
    assert exp3928._json_rows("bad") == []
    assert exp3928._relative_to_repo(tmp_path, tmp_path / "data" / "x.json") == "data/x.json"
    assert exp3928._relative_to_repo(tmp_path, Path("/outside/x.json")) == "/outside/x.json"
    with pytest.raises(ValueError, match="not bool"):
        exp3928._first_error_index({"label": True})

    bad_rows = [
        {"id": "no-steps", "steps": [], "label": 0},
        {"id": "empty-step", "steps": [" "], "label": 0},
        {"id": "bool-label", "steps": ["x"], "label": True},
        {"id": "too-large", "steps": ["x"], "label": 1},
        {"id": "too-small", "steps": ["x"], "label": -2},
    ]
    with pytest.raises(ValueError, match="yielded 0"):
        exp3928.panel_from_processbench_rows(bad_rows, source_detail="bad", min_problem_items=1)
    with pytest.raises(ValueError, match="both correct"):
        exp3928.panel_from_processbench_rows(
            [{"id": "all-correct", "steps": ["a", "b"], "label": -1}],
            source_detail="bad",
            min_problem_items=1,
        )

    assert exp3928._take_rows(({"i": idx} for idx in range(3)), 2) == [{"i": 0}, {"i": 1}]

    calls: list[tuple[str, str]] = []

    def fake_load_dataset(_name: str, config_name: str, *, split: str, streaming: bool) -> list[dict[str, object]]:
        assert streaming is True
        calls.append((config_name, split))
        if len(calls) < 3:
            raise ValueError("missing builder")
        return [
            {"id": "p0", "steps": ["ok", "bad"], "label": 1},
            {"id": "p1", "steps": ["bad", "later"], "label": 0},
        ]

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))
    panel = exp3928.load_processbench_panel(
        exp3928.ExperimentConfig(repo_root=tmp_path, min_processbench_items=2)
    )

    assert calls == [("gsm8k", "test"), ("math", "test"), ("default", "gsm8k")]
    assert panel.corpus_used == "processbench_slice"

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(
            load_dataset=lambda _name, _config_name, *, split, streaming: (_ for _ in ()).throw(
                RuntimeError(f"failed {split} {streaming}")
            )
        ),
    )
    with pytest.raises(RuntimeError, match="gsm8k/test"):
        exp3928.load_processbench_panel(
            exp3928.ExperimentConfig(repo_root=tmp_path, min_processbench_items=2)
        )


def test_req_verify_3928_falls_back_to_local_fover_when_processbench_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3928: ProcessBench failures use data/fover_test_v4.json without fabrication."""

    _write_fover_fixture(tmp_path)

    def boom(_config: exp3928.ExperimentConfig) -> exp3928.IndependentPanel:
        raise RuntimeError("network or schema unavailable")

    monkeypatch.setattr(exp3928, "load_processbench_panel", boom)

    panel = exp3928.load_independent_corpus(
        exp3928.ExperimentConfig(repo_root=tmp_path, min_processbench_items=2)
    )

    assert panel.corpus_used == "fover_test_v4_fallback"
    assert panel.labels == (0, 1, 0, 1)
    assert panel.corpus_source["processbench_failure"] == "RuntimeError('network or schema unavailable')"
    assert panel.corpus_source["corpus_path"] == "data/fover_test_v4.json"


def test_req_verify_3928_fover_fallback_validation_branches(tmp_path: Path) -> None:
    """REQ-VERIFY-3928: malformed fallback files fail before scoring claims."""

    cfg = exp3928.ExperimentConfig(repo_root=tmp_path)
    with pytest.raises(FileNotFoundError, match="fover_test_v4"):
        exp3928.load_fover_fallback_panel(cfg, processbench_failure=RuntimeError("pb"))

    _write_json(
        tmp_path / "data" / "fover_test_v4.json",
        [
            {"question_id": "empty", "step_text": " ", "label": "correct"},
            {"question_id": "bad", "step_text": "x", "label": "maybe"},
            {"question_id": "only", "step_text": "x", "label": "correct"},
        ],
    )
    with pytest.raises(ValueError, match="both labels"):
        exp3928.load_fover_fallback_panel(cfg, processbench_failure=RuntimeError("pb"))


def test_req_verify_3928_energy_ensemble_scores_use_exp2837_aggregation(tmp_path: Path) -> None:
    """REQ-VERIFY-3928: energy scoring reuses the tier0r/tier0u plus FR-11 path."""

    panel = exp3928.IndependentPanel(
        rows=(
            {"question_id": "a", "step_text": "correct", "label": "correct"},
            {"question_id": "b", "step_text": "incorrect", "label": "incorrect"},
        ),
        labels=(0, 1),
        texts=("correct", "incorrect"),
        corpus_used="fover_test_v4_fallback",
        panel_sha256="p" * 64,
        corpus_source={"corpus_path": "data/fover_test_v4.json"},
    )

    def fake_verifiers(texts: tuple[str, ...]) -> dict[str, list[float]]:
        assert texts == ("correct", "incorrect")
        return {
            "tier0r_curry_howard": [0.2, 0.8],
            "tier0u_logical_consistency": [0.0, 1.0],
        }

    def fake_memory(_repo_root: Path) -> dict[str, object]:
        return {"question_ids": {"b"}, "prompt_token_sets": []}

    scores = exp3928.score_energy_ensemble(
        panel,
        tmp_path,
        verifier_scorer=fake_verifiers,
        memory_loader=fake_memory,
    )

    assert scores.scores == pytest.approx((0.18, 1.82))
    assert scores.error_preds == (0, 1)
    assert scores.threshold == pytest.approx(1.0)

    with pytest.raises(ValueError, match="score lengths"):
        exp3928.score_energy_ensemble(
            panel,
            tmp_path,
            verifier_scorer=lambda _texts: {
                "tier0r_curry_howard": [0.1],
                "tier0u_logical_consistency": [0.1],
            },
            memory_loader=fake_memory,
        )


def test_req_verify_3928_strong_reasoner_wrapper_uses_boosted_arm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3928: strong reasoner scoring delegates to the robust 3916 path."""

    calls: list[dict[str, object]] = []

    def fake_score(panel: exp3928.IndependentPanel, generator: object, model_specs: dict[str, object], **kwargs: object) -> exp3928.SelfVerifyArmScoring:
        calls.append({"panel": panel, "generator": generator, "model_specs": model_specs, **kwargs})
        return _reasoner(len(panel.rows))

    monkeypatch.setattr(exp3928, "score_reasoner_arm_with_robust_generator", fake_score)
    generator = object()
    result = exp3928.score_strong_reasoner(
        _panel(),
        generator,
        {"model_used": "fixture"},
        exp3928.ExperimentConfig(repo_root=tmp_path, random_seed=7, max_tokens_strong=11),
    )

    assert result.parsed_count == 4
    assert calls[0]["arm"] == "strong"
    assert calls[0]["max_tokens"] == 11
    assert calls[0]["random_seed"] == 7


def test_scenario_verify_3928_artifact_fields_and_replication_gate(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3928: artifacts expose bare fields and the moat gate."""

    artifact = exp3928.build_artifact_from_metrics(
        metrics=_metrics(),
        config=exp3928.ExperimentConfig(repo_root=tmp_path, started_at=0.0, clock=lambda: 65.0),
        preconditions_checked=[exp3928.PreconditionCheck("cuda_available", True, "ok")],
        model_specs={"model_used": "gemma-4-26B-A4B-it"},
        corpus_used="processbench_slice",
        corpus_source={"dataset": "Qwen/ProcessBench"},
        panel_sha256="p" * 64,
        reasoner_error_scores=[0.9, 0.1],
        carnot_error_scores=[0.8, 0.2],
        per_step_results=[{"index": 0, "label": "correct"}],
    )

    exp3928.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete: moat_scissor_REPLICATES_on_processbench_slice")
    assert artifact["moat_replicates"] is True
    assert artifact["duration_s"] == 65.0
    assert not isinstance(artifact["residual_catch_rate"], dict)
    assert set(exp3928.REQUIRED_PRINCIPLE_FIELDS) <= set(artifact["field_principles"])


def test_req_verify_3928_verdict_branches_and_validation(tmp_path: Path) -> None:
    """REQ-VERIFY-3928: terminal gates and schema validation fail closed."""

    inconclusive = exp3928.classify_verdict(
        _metrics(n_residual_errors=12),
        corpus_used="fover_test_v4_fallback",
    )
    assert inconclusive == "complete: moat_scissor_INCONCLUSIVE_nres12_on_fover_test_v4_fallback"

    not_replicated = exp3928.classify_verdict(
        _metrics(
            residual_catch_rate=0.1,
            residual_catch_ci95={"mean": 0.1, "low": 0.0, "high": 0.2, "n_resamples": 1000, "bootstrap_seed": 1},
            error_overlap_jaccard=0.2,
        ),
        corpus_used="processbench_slice",
    )
    assert not_replicated.startswith("complete: moat_scissor_NOT_REPLICATED_on_processbench_slice")
    boundary = exp3928.classify_verdict(
        _metrics(
            residual_catch_ci95={"mean": 0.45, "low": 0.4, "high": 0.6, "n_resamples": 1000, "bootstrap_seed": 1},
            error_overlap_jaccard=0.4,
        ),
        corpus_used="processbench_slice",
    )
    assert boundary == "complete: moat_scissor_INCONCLUSIVE_boundary_on_processbench_slice"

    blocked = exp3928.build_blocked_artifact(
        reason="blocked_no_cuda",
        preconditions_checked=[],
        duration_s=0.1,
    )
    exp3928.validate_artifact(blocked)
    assert blocked["moat_replicates"] is False
    assert blocked["residual_catch_rate"] is None
    floor = exp3928.build_artifact_from_metrics(
        metrics=_metrics(),
        config=exp3928.ExperimentConfig(repo_root=tmp_path, started_at=0.0, clock=lambda: 1.0),
        preconditions_checked=[],
        model_specs={},
        corpus_used="processbench_slice",
        corpus_source={},
        panel_sha256="p" * 64,
        reasoner_error_scores=[0.1, 0.9],
        carnot_error_scores=[0.1, 0.9],
    )
    assert floor["honest_verdict"] == "blocked_live_duration_floor"
    assert floor["moat_replicates"] is False

    for mutation, pattern in (
        ({}, "missing required"),
        ({"honest_verdict": "not-terminal"}, "terminal prefix"),
        ({"field_principles": []}, "field_principles"),
        ({"field_principles": {**blocked["field_principles"], "n_items": ""}}, "n_items"),
        ({"moat_replicates": "false"}, "bare bool"),
        ({"residual_catch_rate": {"value": 0.1, "principle": "bad"}}, "bare value"),
        ({"n_residual_errors": 1.2}, "bare int"),
        ({"duration_s": "1.0"}, "bare number"),
        ({"reproducibility_checksum": "short"}, "sha256"),
    ):
        candidate = dict(blocked)
        if mutation:
            candidate.update(mutation)
        else:
            candidate.pop("random_seed")
        with pytest.raises(ValueError, match=pattern):
            exp3928.validate_artifact(candidate)


def test_req_verify_3928_preflight_blocks_unready_exp3915(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-3928-BLOCKED: Exp 3915 disk-read readiness is enforced."""

    _write_exp3915_fixture(tmp_path, unit_test_passed=False)

    checks, blocked_reason, source = exp3928.probe_preconditions(
        exp3928.ExperimentConfig(repo_root=tmp_path),
        cuda_probe=lambda _cfg: exp3928.PreconditionCheck("cuda_available", True, "fixture cuda"),
    )

    assert blocked_reason == "blocked_upstream_gguf_harness_not_ready"
    assert source is None
    assert [check.resource for check in checks] == ["cuda_available", "exp3915_gguf_harness_ready"]

    _write_exp3915_fixture(tmp_path, unit_test_passed=True)
    checks, blocked_reason, source = exp3928.probe_preconditions(
        exp3928.ExperimentConfig(repo_root=tmp_path),
        cuda_probe=lambda _cfg: exp3928.PreconditionCheck("cuda_available", True, "fixture cuda"),
    )
    assert blocked_reason is None
    assert source is not None
    assert checks[-1].available is True

    checks, blocked_reason, _source = exp3928.probe_preconditions(
        exp3928.ExperimentConfig(repo_root=tmp_path),
        cuda_probe=lambda _cfg: exp3928.PreconditionCheck("cuda_available", False, "no cuda"),
    )
    assert blocked_reason == "blocked_no_cuda"
    assert checks[0].available is False


def test_req_verify_3928_injected_run_writes_success_without_live_llm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3928: injected scorers keep unit tests off live GGUF inference."""

    panel = exp3928.IndependentPanel(
        rows=tuple(
            {"corpus_item_id": f"row-{idx}", "question_id": f"q-{idx}", "step_text": f"step {idx}", "label": "incorrect" if idx < 60 else "correct"}
            for idx in range(120)
        ),
        labels=tuple(1 if idx < 60 else 0 for idx in range(120)),
        texts=tuple(f"step {idx}" for idx in range(120)),
        corpus_used="processbench_slice",
        panel_sha256="p" * 64,
        corpus_source={"dataset": "fixture"},
    )
    reasoner = exp3928.SelfVerifyArmScoring(
        raw_responses=tuple("incorrect" if idx < 10 else "correct" for idx in range(120)),
        error_scores=tuple(0.9 if idx < 10 else 0.1 for idx in range(120)),
        error_preds=tuple(1 if idx < 10 else 0 for idx in range(120)),
        parsed_count=120,
        unparsed_count=0,
        parser_constant_prediction=False,
    )
    energy = exp3928.EnergyScoring(
        scores=tuple(0.9 if idx < 60 else 0.1 for idx in range(120)),
        error_preds=tuple(1 if idx < 60 else 0 for idx in range(120)),
        threshold=0.5,
    )
    output_path = tmp_path / "results" / "exp3928.json"

    monkeypatch.setattr(
        exp3928,
        "probe_preconditions",
        lambda _config, cuda_probe=exp3928._probe_cuda_with_venv: (
            (exp3928.PreconditionCheck("cuda_available", True, "ok"),),
            None,
            {"model_used": "gemma-4-26B-A4B-it", "n_gpu_layers_used": -1},
        ),
    )

    artifact = exp3928.run_experiment(
        exp3928.ExperimentConfig(
            repo_root=tmp_path,
            output_path=output_path,
            started_at=0.0,
            clock=lambda: 65.0,
        ),
        corpus_loader=lambda _config: panel,
        generator_loader=lambda _source, _config: (
            object(),
            {"model_used": "gemma-4-26B-A4B-it", "gguf_path": "fixture.gguf"},
        ),
        reasoner_scorer=lambda _panel, _generator, _model_specs, _config: reasoner,
        energy_scorer=lambda _panel, _root: energy,
        write=True,
    )

    assert output_path.is_file()
    assert artifact == json.loads(output_path.read_text(encoding="utf-8"))
    assert artifact["n_items"] == 120
    assert artifact["n_residual_errors"] == 50
    assert artifact["moat_replicates"] is True
    assert artifact["per_step_results"][0]["reasoner_strong_rejects"] is True
    assert artifact["per_step_results"][-1]["carnot_rejects"] is False


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("preflight_blocked", "blocked_no_cuda"),
        ("missing_gguf_source", "blocked_upstream_gguf_harness_not_ready"),
        ("corpus", "blocked_independent_corpus_unavailable"),
        ("energy", "blocked_energy_ensemble_scoring_failed"),
        ("generator", "blocked_all_gguf_inference_failed"),
        ("reasoner", "blocked_reasoner_self_verify_inference_failed"),
    ],
)
def test_req_verify_3928_run_experiment_blocked_branches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage: str,
    expected: str,
) -> None:
    """SCENARIO-VERIFY-3928-BLOCKED: every runtime failure writes a terminal artifact."""

    gguf_source = {"model_used": "fixture", "n_gpu_layers_used": -1}
    checks = (exp3928.PreconditionCheck("cuda_available", stage != "preflight_blocked", "fixture"),)
    if stage == "preflight_blocked":
        preflight = (checks, "blocked_no_cuda", gguf_source)
    elif stage == "missing_gguf_source":
        preflight = (checks, None, None)
    else:
        preflight = (checks, None, gguf_source)
    monkeypatch.setattr(exp3928, "probe_preconditions", lambda _config, cuda_probe=exp3928._probe_cuda_with_venv: preflight)

    panel = _panel()
    artifact = exp3928.run_experiment(
        exp3928.ExperimentConfig(
            repo_root=tmp_path,
            output_path=tmp_path / f"{stage}.json",
            started_at=0.0,
            clock=lambda: 2.0,
        ),
        corpus_loader=(
            (lambda _config: (_ for _ in ()).throw(RuntimeError("corpus failed")))
            if stage == "corpus"
            else (lambda _config: panel)
        ),
        energy_scorer=(
            (lambda _panel, _root: (_ for _ in ()).throw(RuntimeError("energy failed")))
            if stage == "energy"
            else (lambda _panel, _root: _energy())
        ),
        generator_loader=(
            (lambda _source, _config: (_ for _ in ()).throw(RuntimeError("load failed")))
            if stage == "generator"
            else (lambda _source, _config: (object(), {"model_used": "fixture"}))
        ),
        reasoner_scorer=(
            (lambda _panel, _generator, _model_specs, _config: (_ for _ in ()).throw(RuntimeError("judge failed")))
            if stage == "reasoner"
            else (lambda _panel, _generator, _model_specs, _config: _reasoner())
        ),
        write=True,
    )

    assert artifact["honest_verdict"] == expected
    assert (tmp_path / f"{stage}.json").is_file()


def test_req_verify_3928_cli_reports_terminal_status(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3928: CLI adapter returns nonzero only for blocked artifacts."""

    monkeypatch.setattr(exp3928, "run_experiment", lambda _config, write: {"honest_verdict": "complete: fixture"})
    assert exp3928.cli_main(["--repo-root", str(tmp_path)]) == 0
    assert exp3928.OUTPUT_REL_PATH.name in capsys.readouterr().out

    monkeypatch.setattr(exp3928, "run_experiment", lambda _config, write: {"honest_verdict": "blocked_no_cuda"})
    assert exp3928.cli_main(["--repo-root", str(tmp_path), "--output-path", str(tmp_path / "out.json")]) == 1
    assert "blocked_no_cuda" in capsys.readouterr().out
