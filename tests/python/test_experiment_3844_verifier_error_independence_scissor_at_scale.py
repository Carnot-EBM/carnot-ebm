"""Tests for Exp 3844 verifier error-independence at scale.

Spec refs: REQ-VERIFY-3844, SCENARIO-VERIFY-3844.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from carnot.eval import verifier_error_independence_scissor_at_scale as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"


def test_req_verify_3844_spec_anchor_exists() -> None:
    """REQ-VERIFY-3844: the at-scale scissor measurement is OpenSpec anchored."""
    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-VERIFY-3844" in spec
    assert "SCENARIO-VERIFY-3844" in spec
    assert exp.OUTPUT_REL_PATH.as_posix() in spec
    assert "residual_catch_rate" in spec


def test_scenario_verify_3844_residual_and_overlap_reuse_exp3827_definitions() -> None:
    """SCENARIO-VERIFY-3844: residual catch and Jaccard match Exp3827 semantics."""
    metrics = exp.compute_scissor_metrics(
        labels=[1, 1, 1, 1, 0, 0],
        reasoner_error_scores=[1, 0, 0, 1, 0, 0],
        carnot_error_scores=[0.9, 0.8, 0.1, 0.2, 0.3, 0.4],
        carnot_error_preds=[1, 1, 0, 0, 0, 0],
        bootstrap_seed=7,
        bootstrap_resamples=1000,
    )

    assert metrics.n_residual_errors == 2
    assert metrics.reasoner_caught_error_indices == (0, 3)
    assert metrics.carnot_caught_error_indices == (0, 1)
    assert metrics.residual_catch_rate == 0.5
    assert metrics.residual_catch_ci95["n_resamples"] == 1000
    assert metrics.error_overlap_jaccard == 1 / 3


def test_scenario_verify_3844_bootstrap_ci_is_seeded_and_defensive() -> None:
    """SCENARIO-VERIFY-3844: bootstrap CI is reproducible and handles empty residuals."""
    ci_a = exp.bootstrap_binary_ci([1, 0, 1, 1], seed=3844, n_resamples=1000)
    ci_b = exp.bootstrap_binary_ci([1, 0, 1, 1], seed=3844, n_resamples=1000)
    empty = exp.bootstrap_binary_ci([], seed=3844, n_resamples=1000)

    assert ci_a == ci_b
    assert ci_a["mean"] == 0.75
    assert 0.0 <= ci_a["low"] <= ci_a["high"] <= 1.0
    assert empty == {
        "mean": 0.0,
        "low": 0.0,
        "high": 0.0,
        "n_resamples": 1000,
        "bootstrap_seed": 3844,
    }


def test_scenario_verify_3844_terminal_gates_cover_survival_subsumption_and_controls() -> None:
    """SCENARIO-VERIFY-3844: positive controls gate the moat verdict."""
    survives = exp.ScissorMetrics(
        residual_catch_rate=0.62,
        residual_catch_ci95={
            "mean": 0.62,
            "low": 0.51,
            "high": 0.72,
            "n_resamples": 1000,
            "bootstrap_seed": 3844,
        },
        error_overlap_jaccard=0.5,
        reasoner_self_verify_auroc=0.72,
        carnot_ensemble_auroc=0.9131,
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
            "bootstrap_seed": 3844,
        },
        error_overlap_jaccard=0.55,
        reasoner_self_verify_auroc=0.72,
        carnot_ensemble_auroc=0.9131,
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
        carnot_ensemble_auroc=0.9131,
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
        carnot_ensemble_auroc=0.86,
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


def test_req_verify_3844_artifact_builder_includes_principles_and_checksum(tmp_path: Path) -> None:
    """REQ-VERIFY-3844: artifacts carry required fields, principles, and checksum."""
    metrics = exp.ScissorMetrics(
        residual_catch_rate=0.75,
        residual_catch_ci95={
            "mean": 0.75,
            "low": 0.61,
            "high": 0.88,
            "n_resamples": 1000,
            "bootstrap_seed": 3844,
        },
        error_overlap_jaccard=0.2,
        reasoner_self_verify_auroc=0.7,
        carnot_ensemble_auroc=0.9131,
        n_items=1000,
        n_residual_errors=160,
        n_gold_incorrect=500,
        reasoner_caught_error_indices=(0, 2),
        carnot_caught_error_indices=(1, 2, 3),
    )
    artifact = exp.build_artifact_from_metrics(
        metrics=metrics,
        config=exp.ExperimentConfig(
            repo_root=tmp_path,
            random_seed=42,
            bootstrap_seed=3844,
            started_at=10.0,
            clock=lambda: 75.5,
        ),
        preconditions_checked=[
            exp.PreconditionCheck("cuda", True, "2 devices"),
            exp.PreconditionCheck("corpus", True, "1000 rows"),
        ],
        model_specs={"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF", "model_path": "model.gguf"},
        cited_upstream_artifacts={"exp3827": {"sha256": "a" * 64}},
        panel_sha256="b" * 64,
        reasoner_error_scores=[1, 0, 1],
        carnot_error_scores=[0.9, 0.1, 0.8],
    )

    exp.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["duration_s"] == 65.5
    assert len(artifact["reproducibility_checksum"]) == 64
    assert set(exp.REQUIRED_PRINCIPLE_FIELDS) <= set(artifact["field_principles"])
    assert all(
        "principle" in artifact["field_principles"][field]
        for field in exp.REQUIRED_PRINCIPLE_FIELDS
    )


def test_req_verify_3844_blocked_artifact_is_terminal_and_non_fabricated(tmp_path: Path) -> None:
    """REQ-VERIFY-3844: failed preconditions emit blocked artifacts without metrics."""
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


def test_req_verify_3844_loads_balanced_fover_json_panel(tmp_path: Path) -> None:
    """REQ-VERIFY-3844: FoVer JSON rows are schema-checked and balanced."""
    rows = [
        {"question_id": f"p{idx}", "step_text": f"bad {idx}", "label": "incorrect"}
        for idx in range(6)
    ] + [
        {"question": f"q{idx}", "step_text": f"good {idx}", "label": "correct"}
        for idx in range(6)
    ]
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "fover_test_v4.json").write_text(json.dumps(rows), encoding="utf-8")

    source_path, loaded = exp.load_fover_test_rows(tmp_path)
    panel = exp.select_balanced_panel(loaded, seed=42, n_items=10)

    assert source_path == data_dir / "fover_test_v4.json"
    assert len(panel.rows) == 10
    assert sum(panel.labels) == 5
    assert len(panel.panel_sha256) == 64


def test_req_verify_3844_fake_scoring_builds_artifact_without_live_llm(tmp_path: Path) -> None:
    """REQ-VERIFY-3844: injected scorers let unit tests avoid live GGUF inference."""
    rows = [
        {"question_id": f"p{idx}", "step_text": f"bad {idx}", "label": "incorrect"}
        for idx in range(500)
    ] + [
        {"question_id": f"n{idx}", "step_text": f"good {idx}", "label": "correct"}
        for idx in range(500)
    ]
    panel = exp.select_balanced_panel(rows, seed=1, n_items=1000)

    def fake_reasoner(selected_panel: exp.FoVerPanel, _model_specs: dict[str, object]) -> exp.ReasonerScoring:
        scores = [1 if label == 1 and idx % 4 != 0 else 0 for idx, label in enumerate(selected_panel.labels)]
        return exp.ReasonerScoring(
            raw_responses=["NO" if score else "YES" for score in scores],
            error_scores=scores,
        )

    def fake_carnot(selected_panel: exp.FoVerPanel) -> exp.CarnotScoring:
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
        cited_upstream_artifacts={},
        reasoner_scorer=fake_reasoner,
        carnot_scorer=fake_carnot,
        write=True,
    )

    assert (tmp_path / exp.OUTPUT_REL_PATH).exists()
    assert artifact["n_items"] == 1000
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_verify_3844_prompt_parser_and_fake_llama_adapter() -> None:
    """REQ-VERIFY-3844: the live adapter parser preserves Exp3827 YES/NO semantics."""
    panel = exp.FoVerPanel(
        rows=({"question_id": "a", "step_text": "bad", "label": "incorrect"},),
        labels=(1,),
        texts=("bad step",),
        panel_sha256="p" * 64,
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


def test_req_verify_3844_carnot_ensemble_uses_exp2837_aggregation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3844: Carnot scoring is 0.9*tier0r + 0.1*tier0u + memory."""
    panel = exp.FoVerPanel(
        rows=(
            {"question_id": "a", "step_text": "x", "label": "incorrect"},
            {"question_id": "b", "step_text": "y", "label": "correct"},
            {"question_id": "c", "step_text": "z", "label": "incorrect"},
        ),
        labels=(1, 0, 1),
        texts=("x", "y", "z"),
        panel_sha256="p" * 64,
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


def test_req_verify_3844_helper_edge_cases_and_validation(tmp_path: Path) -> None:
    """REQ-VERIFY-3844: helper validation failures are deterministic."""
    missing = tmp_path / "missing.txt"
    assert exp._sha256_file(missing) is None
    present = tmp_path / "present.txt"
    present.write_text("abc", encoding="utf-8")
    assert exp._sha256_file(present) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert exp._json_rows({"items": [{"a": 1}, []]}) == [{"a": 1}]
    assert exp._json_rows("bad") == []

    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "fover_test_v3.json").write_text(
        json.dumps(
            [
                {"question_id": "skip", "step_text": "x"},
                {"question_id": "bad-label", "step_text": "x", "label": "maybe"},
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError):
        exp.load_fover_test_rows(tmp_path)

    with pytest.raises(ValueError, match="n_resamples"):
        exp.bootstrap_binary_ci([1], seed=1, n_resamples=0)
    with pytest.raises(ValueError, match="align"):
        exp.compute_scissor_metrics(
            labels=[1],
            reasoner_error_scores=[1, 0],
            carnot_error_scores=[1],
            carnot_error_preds=[1],
            bootstrap_seed=1,
            bootstrap_resamples=1,
        )

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
    broken = dict(artifact)
    broken["honest_verdict"] = "complete: scissor_at_scale_MOAT_SURVIVES_residcatch0.9_ci0.8-1.0_overlap0.1_n1"
    broken["n_items"] = 1
    with pytest.raises(ValueError, match="n_items>=1000"):
        exp.validate_artifact(broken)


def test_scenario_verify_3844_boundary_verdicts() -> None:
    """SCENARIO-VERIFY-3844: boundary cases stay terminal and inconclusive."""
    base = dict(
        residual_catch_rate=0.4,
        residual_catch_ci95={
            "mean": 0.4,
            "low": 0.31,
            "high": 0.49,
            "n_resamples": 1000,
            "bootstrap_seed": 3844,
        },
        error_overlap_jaccard=0.65,
        reasoner_self_verify_auroc=0.72,
        carnot_ensemble_auroc=0.9131,
        n_items=1000,
        n_residual_errors=120,
        n_gold_incorrect=500,
        reasoner_caught_error_indices=(),
        carnot_caught_error_indices=(),
    )
    assert exp.classify_verdict(exp.ScissorMetrics(**base)).endswith("boundary_gate")
    assert exp.classify_verdict(exp.ScissorMetrics(**{**base, "n_items": 999})).endswith("n_items_lt1000")
    assert exp.classify_verdict(exp.ScissorMetrics(**{**base, "n_residual_errors": 29})).endswith(
        "n_residual_errors_lt30"
    )


def test_req_verify_3844_upstream_and_model_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3844: upstream SHA and GGUF resolution are auditable."""
    (tmp_path / "results").mkdir()
    (tmp_path / "scripts" / "experiments").mkdir(parents=True)
    (tmp_path / "scripts").mkdir(exist_ok=True)
    (tmp_path / "results" / "experiment_3827_verifier_error_independence_scissor.json").write_text(
        "{\"flagged_adversarial\": true}",
        encoding="utf-8",
    )
    (tmp_path / "results" / "experiment_2837_fover_memory_leakage_v3.json").write_text(
        "{bad-json",
        encoding="utf-8",
    )
    (tmp_path / "scripts" / "experiments" / "experiment_3827_verifier_error_independence_scissor.py").write_text(
        "print('x')",
        encoding="utf-8",
    )
    (tmp_path / "scripts" / "experiment_2837_fover_memory_leakage_v3.py").write_text(
        "print('y')",
        encoding="utf-8",
    )
    cited = exp.collect_upstream_artifacts(tmp_path)
    assert exp._upstream_flagged(cited) is True
    assert cited["exp3827_result"]["sha256"]
    assert cited["exp2837_result"]["flagged_adversarial"] is False

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


@pytest.mark.parametrize(
    ("false_resource", "expected_reason"),
    [
        ("cuda_available", "blocked_no_cuda"),
        ("carnot_verify_import", "blocked_carnot_verify_import"),
        ("model_path", "blocked_model_not_cached_qwen3.6_35b"),
        ("llama_cpp_import", "blocked_llama_cpp_not_installed"),
        ("fover_test_corpus", "blocked_fover_corpus_not_available"),
        ("fover_balanced_panel_capacity", "blocked_fover_balanced_corpus_not_available"),
        ("upstream_artifacts_available", "blocked_upstream_artifacts_unavailable"),
        ("upstream_adversarial_flags_absent", "blocked_upstream_adversarial_flag"),
    ],
)
def test_req_verify_3844_probe_preconditions_block_reasons(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    false_resource: str,
    expected_reason: str,
) -> None:
    """REQ-VERIFY-3844: precondition failures map to honest blocked verdicts."""

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return false_resource != "cuda_available"

        @staticmethod
        def device_count() -> int:
            return 1 if false_resource != "cuda_available" else 0

    class FakeTorch:
        cuda = FakeCuda

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setattr(exp.importlib, "import_module", lambda _name: object())
    monkeypatch.setattr(
        exp,
        "_resolve_reasoner_model",
        lambda: (
            {"hf_id": "fixture", "model_path": None if false_resource == "model_path" else "fixture.gguf"},
            [exp.PreconditionCheck("qwen3.6_35b_gguf_cached", false_resource != "model_path", "fixture")],
        ),
    )
    monkeypatch.setattr(
        exp,
        "load_fover_test_rows",
        lambda root: (
            root / "data" / "fover_test_v4.json",
            (
                [{"question_id": f"p{idx}", "step_text": "x", "label": "incorrect"} for idx in range(500)]
                + [{"question_id": f"n{idx}", "step_text": "x", "label": "correct"} for idx in range(500)]
            )
            if false_resource != "fover_balanced_panel_capacity"
            else [{"question_id": f"n{idx}", "step_text": "x", "label": "correct"} for idx in range(1000)],
        ),
    )

    def fake_upstream(_root: Path) -> dict[str, object]:
        if false_resource == "upstream_artifacts_available":
            return {"exp3827_result": {"exists": False, "sha256": None}}
        flagged = false_resource == "upstream_adversarial_flags_absent"
        return {"exp3827_result": {"exists": True, "sha256": "a" * 64, "flagged_adversarial": flagged}}

    monkeypatch.setattr(exp, "collect_upstream_artifacts", fake_upstream)
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
    if false_resource == "fover_test_corpus":
        monkeypatch.setattr(exp, "load_fover_test_rows", lambda _root: (_ for _ in ()).throw(FileNotFoundError("no corpus")))

    preflight = exp.probe_preconditions(exp.ExperimentConfig(repo_root=tmp_path))

    assert preflight.blocked_reason == expected_reason


def test_req_verify_3844_run_experiment_orchestrates_blocked_and_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3844: run_experiment writes blocked artifacts or selected-panel output."""
    blocked_preflight = exp.PreflightResult(
        checks=(exp.PreconditionCheck("cuda", False, "no"),),
        blocked_reason="blocked_no_cuda",
        model_specs={"hf_id": "fixture"},
        corpus_path=None,
        rows=(),
        cited_upstream_artifacts={},
    )
    monkeypatch.setattr(exp, "probe_preconditions", lambda _config: blocked_preflight)
    blocked = exp.run_experiment(
        exp.ExperimentConfig(repo_root=tmp_path, output_path=tmp_path / "blocked.json", started_at=0.0, clock=lambda: 1.0),
        write=True,
    )
    assert blocked["honest_verdict"] == "blocked_no_cuda"
    assert (tmp_path / "blocked.json").exists()

    rows = [
        {"question_id": f"p{idx}", "step_text": f"bad {idx}", "label": "incorrect"}
        for idx in range(500)
    ] + [
        {"question_id": f"n{idx}", "step_text": f"good {idx}", "label": "correct"}
        for idx in range(500)
    ]
    success_preflight = exp.PreflightResult(
        checks=(exp.PreconditionCheck("ok", True, "yes"),),
        blocked_reason=None,
        model_specs={"hf_id": "fixture", "model_path": "fixture.gguf"},
        corpus_path=tmp_path / "data" / "fover_test_v4.json",
        rows=tuple(rows),
        cited_upstream_artifacts={},
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
