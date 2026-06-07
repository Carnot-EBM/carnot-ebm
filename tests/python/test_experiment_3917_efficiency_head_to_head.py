"""Tests for the Exp 3917 efficiency head-to-head runner.

Spec refs: REQ-VERIFY-3917, SCENARIO-VERIFY-3917,
SCENARIO-VERIFY-3917-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import efficiency_head_to_head_3917 as exp3917


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"


class ScriptedGenerator:
    """Small robust-generator stand-in for deterministic LLM-judge tests."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def tokenize(self, payload: bytes, add_bos: bool = True, **_kwargs: object) -> list[int]:
        tokens = payload.decode("utf-8", errors="ignore").split()
        return [1, *range(2, len(tokens) + 2)] if add_bos else list(range(len(tokens)))

    def __call__(self, prompt: str, **kwargs: object) -> dict[str, object]:
        self.prompts.append(prompt)
        step_text = prompt.split("Step:\n", 1)[-1]
        incorrect = "= 5" in step_text or "invalid" in step_text
        response = {
            "verdict": "incorrect" if incorrect else "correct",
            "error_confidence": 0.92 if incorrect else 0.08,
        }
        assert kwargs["temperature"] == 0.0
        return {"choices": [{"text": json.dumps(response)}]}


def test_req_verify_3917_spec_anchor_exists() -> None:
    """REQ-VERIFY-3917: the efficiency runner is OpenSpec anchored."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-3917" in spec
    assert "SCENARIO-VERIFY-3917" in spec
    assert "python/carnot/eval/efficiency_head_to_head_3917.py" in spec
    assert "results/experiment_3917_efficiency_head_to_head.json" in spec


def test_req_verify_3917_loads_two_labeled_corpora() -> None:
    """REQ-VERIFY-3917: Exp 3884 and a >=200 item FoVer slice share labels."""

    bundle = exp3917.load_labeled_corpora(REPO_ROOT, random_seed=3917)

    assert len(bundle.items) >= 500
    assert bundle.n_items == len(bundle.items)
    assert {source["name"] for source in bundle.corpus_sources} == {
        "exp3884_in_distribution",
        "fover_corpus_v4_slice",
    }
    assert all(int(source["n_items"]) >= 200 for source in bundle.corpus_sources)
    assert set(bundle.labels) == {0, 1}
    assert all("step_text" in item and "gold_error" in item for item in bundle.items)


def test_req_verify_3917_measures_both_verifiers_with_cost_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3917: scoring uses Exp 3905 cost fields for both verifiers."""

    items = (
        {"step_text": "2 + 2 = 4.", "gold_error": 0},
        {"step_text": "2 + 2 = 5.", "gold_error": 1},
        {"step_text": "All valid implications are valid.", "gold_error": 0},
        {"step_text": "This step is invalid.", "gold_error": 1},
    )
    ticks = iter([10.0, 10.1, 20.0, 21.0])

    def fake_energy(rows: tuple[dict[str, object], ...]) -> dict[str, object]:
        assert rows == items
        return {"scores": [0.1, 0.9, 0.2, 0.8], "est_tokens": 12, "est_flops": 120}

    monkeypatch.setattr(exp3917, "run_energy_verifier", fake_energy)

    measured = exp3917.measure_head_to_head_costs(
        items,
        generator=ScriptedGenerator(),
        model_specs={"gguf_path": "/models/gemma-4-26B-A4B-it.gguf"},
        clock=lambda: next(ticks),
        max_tokens=24,
    )

    assert measured.energy_cost["auroc"] == 1.0
    assert measured.llm_cost["auroc"] == 1.0
    assert measured.energy_cost["per_item_wall_ms"] == pytest.approx(25.0)
    assert measured.llm_cost["per_item_wall_ms"] == pytest.approx(250.0)
    assert measured.energy_scores == (0.1, 0.9, 0.2, 0.8)
    assert measured.llm_scores == (0.08, 0.92, 0.08, 0.92)
    assert measured.llm_cost["est_tokens"] > 0
    assert measured.llm_cost["est_flops"] == (
        2 * 26_000_000_000 * measured.llm_cost["est_tokens"]
    )


def test_req_verify_3917_artifact_uses_bare_metrics_and_parity_gate() -> None:
    """REQ-VERIFY-3917: artifact metrics stay bare and verdict follows gates."""

    bundle = exp3917.CorpusBundle(
        items=(
            {"step_text": "a", "gold_error": 0},
            {"step_text": "b", "gold_error": 0},
            {"step_text": "c", "gold_error": 1},
            {"step_text": "d", "gold_error": 1},
        ),
        labels=(0, 0, 1, 1),
        corpus_sources=({"name": "fixture", "n_items": 4},),
        checksum="fixture-sha",
    )
    measured = exp3917.CostMeasurements(
        energy_cost={
            "auroc": 1.0,
            "total_wall_s": 0.4,
            "per_item_wall_ms": 100.0,
            "est_tokens": 8,
            "est_flops": 1000,
            "n_items": 4,
        },
        llm_cost={
            "auroc": 1.0,
            "total_wall_s": 8.0,
            "per_item_wall_ms": 2000.0,
            "est_tokens": 80,
            "est_flops": 80_000,
            "n_items": 4,
        },
        energy_scores=(0.0, 0.1, 0.9, 1.0),
        llm_scores=(0.0, 0.1, 0.9, 1.0),
    )
    config = exp3917.ExperimentConfig(
        repo_root=REPO_ROOT,
        started_at=0.0,
        clock=lambda: 61.0,
        bootstrap_resamples=20,
    )

    artifact = exp3917.build_artifact(
        config=config,
        bundle=bundle,
        measured=measured,
        preconditions_checked=[exp3917.PreconditionCheck("cuda_available", True, "ok")],
        model_specs={"model_used": "gemma-4-26B-A4B-it", "gguf_path": "/models/gemma.gguf"},
        gguf_harness_source={"artifact_path": "results/experiment_3915_robust_gguf_inference_harness.json"},
        cost_harness_source={"artifact_path": "results/experiment_3905_cost_instrumented_verify_harness.json"},
    )

    exp3917.validate_artifact(artifact)
    assert artifact["energy_auroc"] == 1.0
    assert artifact["llm_judge_auroc"] == 1.0
    assert artifact["accuracy_parity"] is True
    assert artifact["cost_ratio_walltime"] == 20.0
    assert artifact["cost_ratio_flops"] == 80.0
    assert artifact["llm_judge_model_used"] == "gemma-4-26B-A4B-it"
    assert artifact["honest_verdict"].startswith("complete: efficiency_PARITY_AND_20.00x_CHEAPER")
    assert not isinstance(artifact["energy_auroc"], dict)
    assert not isinstance(artifact["accuracy_parity"], dict)


def test_req_verify_3917_artifact_reports_cheaper_not_parity() -> None:
    """REQ-VERIFY-3917: cheaper non-parity is an honest partial result."""

    bundle = exp3917.CorpusBundle(
        items=(
            {"step_text": "a", "gold_error": 0},
            {"step_text": "b", "gold_error": 0},
            {"step_text": "c", "gold_error": 1},
            {"step_text": "d", "gold_error": 1},
        ),
        labels=(0, 0, 1, 1),
        corpus_sources=({"name": "fixture", "n_items": 4},),
        checksum="fixture-sha",
    )
    measured = exp3917.CostMeasurements(
        energy_cost={
            "auroc": 0.0,
            "total_wall_s": 0.4,
            "per_item_wall_ms": 100.0,
            "est_tokens": 8,
            "est_flops": 1000,
            "n_items": 4,
        },
        llm_cost={
            "auroc": 1.0,
            "total_wall_s": 8.0,
            "per_item_wall_ms": 2000.0,
            "est_tokens": 80,
            "est_flops": 80_000,
            "n_items": 4,
        },
        energy_scores=(1.0, 0.9, 0.1, 0.0),
        llm_scores=(0.0, 0.1, 0.9, 1.0),
    )

    artifact = exp3917.build_artifact(
        config=exp3917.ExperimentConfig(
            repo_root=REPO_ROOT,
            started_at=0.0,
            clock=lambda: 61.0,
            bootstrap_resamples=20,
        ),
        bundle=bundle,
        measured=measured,
        preconditions_checked=[],
        model_specs={"model_used": "gemma-4-26B-A4B-it"},
        gguf_harness_source={},
        cost_harness_source={},
    )

    assert artifact["accuracy_parity"] is False
    assert artifact["honest_verdict"].startswith("complete: efficiency_CHEAPER_20.00x_but_NOT_PARITY")


def test_scenario_verify_3917_duration_floor_blocks_short_full_run() -> None:
    """SCENARIO-VERIFY-3917: full-corpus results shorter than 60s are blocked."""

    artifact = exp3917.build_blocked_artifact(
        reason="blocked_llm_judge_not_invoked",
        preconditions_checked=[exp3917.PreconditionCheck("llm_judge_duration_floor", False, "duration_s=3")],
        duration_s=3.0,
        model_specs={"model_used": "fixture"},
    )

    exp3917.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_llm_judge_not_invoked"
    assert artifact["energy_auroc"] is None
    assert artifact["llm_judge_auroc"] is None
    assert artifact["accuracy_parity"] is None
    assert artifact["cost_ratio_walltime"] is None
    assert artifact["n_items"] == 0


def test_scenario_verify_3917_preconditions_block_missing_gguf_harness(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3917-BLOCKED: missing upstream GGUF readiness blocks."""

    (tmp_path / "results").mkdir()
    (tmp_path / "python" / "carnot" / "verify").mkdir(parents=True)
    (tmp_path / "results" / "experiment_3905_cost_instrumented_verify_harness.json").write_text(
        "{}\n",
        encoding="utf-8",
    )

    checks, blocked_reason, _model_specs, _gguf_source, _cost_source = exp3917.probe_preconditions(
        exp3917.ExperimentConfig(repo_root=tmp_path),
        cuda_probe=lambda _config: exp3917.PreconditionCheck("cuda_available", True, "ok"),
    )

    assert blocked_reason == "blocked_upstream_gguf_harness_not_ready"
    assert any(check.resource == "exp3915_gguf_harness_ready" and not check.available for check in checks)


def test_req_verify_3917_helper_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3917: helper branches stay deterministic and explicit."""

    cfg = exp3917.ExperimentConfig(repo_root=tmp_path, output_path=tmp_path / "out.json")
    assert cfg.resolved_output_path() == tmp_path / "out.json"
    assert cfg.venv_python() == tmp_path / ".venv" / "bin" / "python"
    assert exp3917._json_rows("not-json-rows") == []
    assert exp3917._label_to_error_int(True) == 1
    assert exp3917._label_to_error_int(0) == 0
    with pytest.raises(ValueError):
        exp3917._label_to_error_int("unsupported")
    with pytest.raises(ValueError):
        exp3917._step_from_row({}, 3)
    assert exp3917._auroc([0, 1], [0.5, 0.5]) == 0.5
    with pytest.raises(ValueError):
        exp3917._auroc([1, 1], [0.5, 0.6])
    with pytest.raises(ValueError):
        exp3917.bootstrap_ci95([0], [0.1, 0.2], seed=1, resamples=1)
    assert exp3917.bootstrap_ci95([0, 1], [0.1, 0.9], seed=1, resamples=0) == {
        "low": 1.0,
        "high": 1.0,
    }

    class NoTokenizer:
        pass

    assert exp3917._llama_token_count(NoTokenizer(), "one two", add_bos=True) == 3
    with pytest.raises(ValueError):
        exp3917._scores_from_result({"scores": [0.1]}, 2)
    assert exp3917._prefer_order_from_source({"model_used": "fixture-model"})[0] == "fixture-model"
    assert exp3917._cost_ratio(1.0, 0.0) is None
    assert exp3917._classify_verdict(
        accuracy_parity=False,
        cost_ratio_walltime=5.0,
        energy_auroc=0.4,
        llm_auroc=0.9,
    ).startswith("complete: efficiency_NOT_DECISIVELY_CHEAPER")


def test_req_verify_3917_loads_upstream_sources() -> None:
    """REQ-VERIFY-3917: upstream readiness sources are disk-read, not inferred."""

    gguf_source = exp3917.load_exp3915_gguf_harness_source(REPO_ROOT)
    cost_source = exp3917.load_exp3905_cost_harness_source(REPO_ROOT)

    assert gguf_source["unit_test_passed"] is True
    assert int(gguf_source["smoke_tokens"]) > 0
    assert gguf_source["harness_module_path"] == "python/carnot/verify/gguf_inference.py"
    assert cost_source["harness_module_path"] == "python/carnot/verify/cost_instrumented_verification.py"


def test_req_verify_3917_validate_artifact_rejects_bad_shapes() -> None:
    """REQ-VERIFY-3917: validation catches wrappers and malformed gates."""

    artifact = exp3917.build_blocked_artifact(
        reason="blocked_fixture",
        preconditions_checked=[],
        duration_s=1.0,
    )

    missing = dict(artifact)
    missing.pop("energy_auroc")
    with pytest.raises(ValueError, match="missing required fields"):
        exp3917.validate_artifact(missing)

    bad_verdict = dict(artifact, honest_verdict="pending")
    with pytest.raises(ValueError, match="terminal prefix"):
        exp3917.validate_artifact(bad_verdict)

    wrapped = dict(artifact, energy_auroc={"value": None, "principle": "bad"})
    with pytest.raises(ValueError, match="wrapper"):
        exp3917.validate_artifact(wrapped)

    bad_duration = dict(artifact, duration_s="1")
    with pytest.raises(ValueError, match="duration_s"):
        exp3917.validate_artifact(bad_duration)

    bad_checksum = dict(artifact, reproducibility_checksum="short")
    with pytest.raises(ValueError, match="sha256"):
        exp3917.validate_artifact(bad_checksum)

    bad_blocked_count = dict(artifact, n_items=1)
    with pytest.raises(ValueError, match="blocked artifacts"):
        exp3917.validate_artifact(bad_blocked_count)

    complete = dict(
        artifact,
        honest_verdict="complete: fixture",
        energy_auroc=1.0,
        llm_judge_auroc=1.0,
        cost_ratio_walltime=11.0,
        cost_ratio_flops=12.0,
        energy_per_item_ms=1.0,
        llm_per_item_ms=11.0,
        accuracy_parity=True,
        n_items=1,
    )
    exp3917.validate_artifact(complete)
    with pytest.raises(ValueError, match="energy_auroc"):
        exp3917.validate_artifact(dict(complete, energy_auroc="1.0"))
    with pytest.raises(ValueError, match="accuracy_parity"):
        exp3917.validate_artifact(dict(complete, accuracy_parity="true"))
    with pytest.raises(ValueError, match="n_items"):
        exp3917.validate_artifact(dict(complete, n_items=0))


def _fixture_bundle() -> exp3917.CorpusBundle:
    return exp3917.CorpusBundle(
        items=(
            {"step_text": "a", "gold_error": 0},
            {"step_text": "b", "gold_error": 0},
            {"step_text": "c", "gold_error": 1},
            {"step_text": "d", "gold_error": 1},
        ),
        labels=(0, 0, 1, 1),
        corpus_sources=({"name": "fixture", "n_items": 4},),
        checksum="fixture-sha",
    )


def _fixture_measurements(*, energy_flops: int = 1000) -> exp3917.CostMeasurements:
    return exp3917.CostMeasurements(
        energy_cost={
            "auroc": 1.0,
            "total_wall_s": 0.4,
            "per_item_wall_ms": 100.0,
            "est_tokens": 8,
            "est_flops": energy_flops,
            "n_items": 4,
        },
        llm_cost={
            "auroc": 1.0,
            "total_wall_s": 8.0,
            "per_item_wall_ms": 2000.0,
            "est_tokens": 80,
            "est_flops": 80_000,
            "n_items": 4,
        },
        energy_scores=(0.0, 0.1, 0.9, 1.0),
        llm_scores=(0.0, 0.1, 0.9, 1.0),
    )


def test_req_verify_3917_artifact_rejects_zero_flop_denominator() -> None:
    """REQ-VERIFY-3917: FLOP ratio claims require a positive denominator."""

    with pytest.raises(ValueError, match="cost ratios"):
        exp3917.build_artifact(
            config=exp3917.ExperimentConfig(
                repo_root=REPO_ROOT,
                started_at=0.0,
                clock=lambda: 61.0,
                bootstrap_resamples=20,
            ),
            bundle=_fixture_bundle(),
            measured=_fixture_measurements(energy_flops=0),
            preconditions_checked=[],
            model_specs={"model_used": "fixture"},
            gguf_harness_source={},
            cost_harness_source={},
        )


def _write_upstream_artifact_set(tmp_path: Path, *, include_cost: bool) -> None:
    (tmp_path / "results").mkdir(parents=True, exist_ok=True)
    (tmp_path / "python" / "carnot" / "verify").mkdir(parents=True, exist_ok=True)
    (tmp_path / "python" / "carnot" / "verify" / "gguf_inference.py").write_text(
        "# fixture\n",
        encoding="utf-8",
    )
    (tmp_path / "results" / "experiment_3915_robust_gguf_inference_harness.json").write_text(
        json.dumps(
            {
                "unit_test_passed": True,
                "smoke_tokens": 1,
                "harness_module_path": "python/carnot/verify/gguf_inference.py",
                "model_used": "fixture-model",
            }
        ),
        encoding="utf-8",
    )
    if include_cost:
        (tmp_path / "python" / "carnot" / "verify" / "cost_instrumented_verification.py").write_text(
            "# fixture\n",
            encoding="utf-8",
        )
        (tmp_path / "results" / "experiment_3905_cost_instrumented_verify_harness.json").write_text(
            json.dumps(
                {
                    "harness_module_path": "python/carnot/verify/cost_instrumented_verification.py",
                    "unit_test_passed": False,
                }
            ),
            encoding="utf-8",
        )


def test_scenario_verify_3917_precondition_ordering(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3917-BLOCKED: hard-resource blocked reasons are ordered."""

    checks, blocked_reason, _model_specs, _gguf_source, _cost_source = exp3917.probe_preconditions(
        exp3917.ExperimentConfig(repo_root=REPO_ROOT),
        cuda_probe=lambda _config: exp3917.PreconditionCheck("cuda_available", False, "no cuda"),
    )
    assert blocked_reason == "blocked_no_cuda"
    assert checks[0].resource == "cuda_available"

    _write_upstream_artifact_set(tmp_path, include_cost=False)
    _checks, blocked_reason, _model_specs, _gguf_source, _cost_source = exp3917.probe_preconditions(
        exp3917.ExperimentConfig(repo_root=tmp_path),
        cuda_probe=lambda _config: exp3917.PreconditionCheck("cuda_available", True, "ok"),
    )
    assert blocked_reason == "blocked_upstream_cost_harness_not_ready"

    _write_upstream_artifact_set(tmp_path, include_cost=True)
    _checks, blocked_reason, _model_specs, _gguf_source, _cost_source = exp3917.probe_preconditions(
        exp3917.ExperimentConfig(repo_root=tmp_path),
        cuda_probe=lambda _config: exp3917.PreconditionCheck("cuda_available", True, "ok"),
    )
    assert blocked_reason == "blocked_labeled_corpora_not_ready"


def test_scenario_verify_3917_malformed_inputs_are_blocking(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3917-BLOCKED: malformed upstream JSON cannot produce claims."""

    (tmp_path / "results").mkdir(parents=True)
    (tmp_path / "data").mkdir(parents=True)
    (tmp_path / "results" / "experiment_3884_in_distribution_error_rich_corpus.json").write_text(
        "[]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exp3884 artifact"):
        exp3917._load_exp3884_items(tmp_path)

    (tmp_path / "data" / "fover_corpus_v4.json").write_text(
        json.dumps(
            [
                {"step_text": "bad", "label": "incorrect"},
                {"step_text": "good", "label": "correct"},
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="FoVer slice"):
        exp3917._load_fover_slice(tmp_path, random_seed=1, min_items=4)

    (tmp_path / "data" / "fover_corpus_v4.json").write_text(
        json.dumps(
            [
                {"step_text": "bad1", "label": "incorrect"},
                {"step_text": "bad2", "label": "incorrect"},
                {"step_text": "good1", "label": "correct"},
                {"step_text": "good2", "label": "correct"},
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "data" / "tiny_exp3884.json").write_text(
        json.dumps(
            [
                {"step_text": "tiny bad", "label": "incorrect"},
                {"step_text": "tiny good", "label": "correct"},
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "results" / "experiment_3884_in_distribution_error_rich_corpus.json").write_text(
        json.dumps({"corpus_path": "data/tiny_exp3884.json"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fewer than 3"):
        exp3917.load_labeled_corpora(tmp_path, random_seed=1, fover_min_items=3)


def test_req_verify_3917_upstream_source_validation_errors(tmp_path: Path) -> None:
    """REQ-VERIFY-3917: upstream source loaders reject malformed readiness data."""

    (tmp_path / "results").mkdir(parents=True)
    (tmp_path / "python" / "carnot" / "verify").mkdir(parents=True)
    (tmp_path / "python" / "carnot" / "verify" / "gguf_inference.py").write_text(
        "# fixture\n",
        encoding="utf-8",
    )
    exp3915_path = tmp_path / "results" / "experiment_3915_robust_gguf_inference_harness.json"

    exp3915_path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exp3915 artifact"):
        exp3917.load_exp3915_gguf_harness_source(tmp_path)

    exp3915_path.write_text(
        json.dumps(
            {
                "unit_test_passed": False,
                "smoke_tokens": 1,
                "harness_module_path": "python/carnot/verify/gguf_inference.py",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unit_test_passed"):
        exp3917.load_exp3915_gguf_harness_source(tmp_path)

    exp3915_path.write_text(
        json.dumps(
            {
                "unit_test_passed": True,
                "smoke_tokens": 0,
                "harness_module_path": "python/carnot/verify/gguf_inference.py",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="smoke_tokens"):
        exp3917.load_exp3915_gguf_harness_source(tmp_path)

    exp3915_path.write_text(
        json.dumps(
            {
                "unit_test_passed": True,
                "smoke_tokens": 1,
                "harness_module_path": "python/carnot/verify/missing.py",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="harness module"):
        exp3917.load_exp3915_gguf_harness_source(tmp_path)

    (tmp_path / "results" / "experiment_3905_cost_instrumented_verify_harness.json").write_text(
        "[]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exp3905 artifact"):
        exp3917.load_exp3905_cost_harness_source(tmp_path)


def test_req_verify_3917_cuda_probe_uses_venv_python() -> None:
    """REQ-VERIFY-3917: CUDA preflight uses the repository virtualenv command."""

    check = exp3917._probe_cuda_with_venv(exp3917.ExperimentConfig(repo_root=REPO_ROOT))

    assert check.resource == "cuda_available"
    assert check.available is True


def test_req_verify_3917_run_experiment_preflight_blocked_branch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3917: preflight failures write blocked artifacts immediately."""

    monkeypatch.setattr(
        exp3917,
        "probe_preconditions",
        lambda _config: (
            (exp3917.PreconditionCheck("fixture", False, "blocked"),),
            "blocked_fixture",
            {"preflight": True},
            None,
            None,
        ),
    )
    output_path = tmp_path / "blocked.json"

    artifact = exp3917.run_experiment(
        exp3917.ExperimentConfig(
            repo_root=tmp_path,
            output_path=output_path,
            started_at=0.0,
            clock=lambda: 2.0,
        ),
        write=True,
    )

    assert artifact["honest_verdict"] == "blocked_fixture"
    assert output_path.is_file()


def test_req_verify_3917_run_experiment_duration_and_complete_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3917: run_experiment blocks short runs and writes complete runs."""

    checks = (exp3917.PreconditionCheck("cuda_available", True, "ok"),)
    bundle = _fixture_bundle()
    measured = _fixture_measurements()

    monkeypatch.setattr(
        exp3917,
        "probe_preconditions",
        lambda _config: (
            checks,
            None,
            {"preflight": True},
            {"model_used": "fixture"},
            {"harness_module_path": "fixture"},
        ),
    )
    monkeypatch.setattr(exp3917, "load_labeled_corpora", lambda *_args, **_kwargs: bundle)
    monkeypatch.setattr(
        exp3917,
        "load_robust_generator",
        lambda *_args, **_kwargs: (
            ScriptedGenerator(),
            {"model_used": "fixture", "gguf_path": "/models/gemma-4-26B-A4B-it.gguf"},
        ),
    )
    monkeypatch.setattr(exp3917, "measure_head_to_head_costs", lambda *_args, **_kwargs: measured)

    short_ticks = iter([0.0, 3.0])
    short_artifact = exp3917.run_experiment(
        exp3917.ExperimentConfig(repo_root=tmp_path, started_at=None, clock=lambda: next(short_ticks)),
        write=False,
    )
    assert short_artifact["honest_verdict"] == "blocked_llm_judge_not_invoked"

    complete_ticks = iter([0.0, 61.0, 61.0])
    output_path = tmp_path / "complete.json"
    complete_artifact = exp3917.run_experiment(
        exp3917.ExperimentConfig(
            repo_root=tmp_path,
            output_path=output_path,
            started_at=None,
            clock=lambda: next(complete_ticks),
            bootstrap_resamples=20,
        ),
        write=True,
    )
    assert complete_artifact["honest_verdict"].startswith("complete: efficiency_PARITY_AND_")
    assert output_path.is_file()


def test_req_verify_3917_cli_main_reports_terminal_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-VERIFY-3917: CLI returns success only for complete artifacts."""

    monkeypatch.setattr(
        exp3917,
        "run_experiment",
        lambda *_args, **_kwargs: {"honest_verdict": "blocked_fixture"},
    )
    assert exp3917.cli_main(["--repo-root", str(tmp_path)]) == 1
    assert "blocked_fixture" in capsys.readouterr().out

    monkeypatch.setattr(
        exp3917,
        "run_experiment",
        lambda *_args, **_kwargs: {"honest_verdict": "complete: fixture"},
    )
    assert exp3917.cli_main(["--repo-root", str(tmp_path), "--output-path", str(tmp_path / "x.json")]) == 0
    assert "complete: fixture" in capsys.readouterr().out


def test_req_verify_3917_script_wrapper_calls_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-3917: requested script delegates to the module CLI."""

    from scripts.experiments import experiment_3917_efficiency_head_to_head as script

    called: dict[str, Any] = {}

    def fake_cli(argv: list[str]) -> int:
        called["argv"] = argv
        return 7

    monkeypatch.setattr(script, "cli_main", fake_cli)

    assert script.main() == 7
    assert called["argv"] == ["--repo-root", str(script.REPO_ROOT)]
