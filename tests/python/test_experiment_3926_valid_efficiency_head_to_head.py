"""Tests for Exp 3926 valid efficiency head-to-head.

Spec refs: REQ-VERIFY-3926, SCENARIO-VERIFY-3926,
SCENARIO-VERIFY-3926-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import valid_efficiency_head_to_head_3926 as exp3926


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"


class ScriptedGenerator:
    """Small robust-generator stand-in for deterministic judge tests."""

    def tokenize(self, payload: bytes, add_bos: bool = True, **_kwargs: object) -> list[int]:
        tokens = payload.decode("utf-8", errors="ignore").replace("\n", " ").split()
        return [1, *range(len(tokens))] if add_bos else list(range(len(tokens)))

    def __call__(self, prompt: str, **kwargs: object) -> dict[str, object]:
        assert kwargs["temperature"] == 0.0
        step = prompt.split("Step under review:", 1)[-1]
        verdict = "INCORRECT" if "wrong" in step.lower() else "CORRECT"
        return {"choices": [{"text": f"REASON: checked.\nVERDICT: {verdict}"}]}


def _ok_cuda(_config: exp3926.ExperimentConfig) -> exp3926.PreconditionCheck:
    return exp3926.PreconditionCheck("cuda_available", True, "scripted cuda")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_fixture_repo(root: Path, *, n_items: int = 6) -> None:
    rows: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    for index in range(n_items):
        incorrect = index % 2 == 1
        label = "incorrect" if incorrect else "correct"
        corpus_item_id = f"row-{index}"
        step = "This step is wrong." if incorrect else "This step is correct."
        rows.append(
            {
                "corpus_item_id": corpus_item_id,
                "question_id": f"q-{index}",
                "step_text": step,
                "label": label,
                "synthetic": False,
            }
        )
        scores.append(
            {
                "index": index,
                "corpus_item_id": corpus_item_id,
                "label": label,
                "carnot_ensemble_score": 0.9 if incorrect else 0.1,
            }
        )

    _write_json(
        root / "results" / "experiment_3925_competent_judge_build.json",
        {
            "judge_module_path": exp3926.JUDGE_MODULE_PATH,
            "judge_model_used": "gemma-4-26B-A4B-it",
            "unit_test_passed": True,
            "fixture_auroc": 0.9,
            "model_specs": {
                "prefer_order": ["gemma-4-26B-A4B-it"],
                "n_ctx": 1024,
                "max_n_gpu_layers": -1,
            },
            "honest_verdict": "complete: competent_judge_READY_fixture_auroc0.9000",
        },
    )
    _write_json(root / "data" / "corpus.json", {"items": rows})
    _write_json(root / "results" / "scores.json", {"items": scores})
    _write_json(
        root / "results" / "experiment_3884_in_distribution_error_rich_corpus.json",
        {
            "corpus_path": "data/corpus.json",
            "per_item_ensemble_scores_path": "results/scores.json",
        },
    )


def test_req_verify_3926_spec_anchor_exists() -> None:
    """REQ-VERIFY-3926: the valid efficiency run is OpenSpec anchored."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-3926" in spec
    assert "SCENARIO-VERIFY-3926" in spec
    assert "valid_efficiency_head_to_head_3926.py" in spec
    assert "results/experiment_3926_valid_efficiency_head_to_head.json" in spec


def test_req_verify_3926_loads_same_exp3884_order_and_scores(tmp_path: Path) -> None:
    """REQ-VERIFY-3926: Exp 3884 labels and energy scores keep one item order."""

    _write_fixture_repo(tmp_path, n_items=6)

    items, corpus_source = exp3926.load_exp3884_corpus(
        exp3926.ExperimentConfig(repo_root=tmp_path, min_corpus_items=6)
    )

    assert len(items) == 6
    assert [item["corpus_item_id"] for item in items] == [f"row-{index}" for index in range(6)]
    assert [item["gold_error"] for item in items] == [0, 1, 0, 1, 0, 1]
    assert [item["energy_score"] for item in items] == [0.1, 0.9, 0.1, 0.9, 0.1, 0.9]
    assert corpus_source["energy_auroc_from_scores"] == 1.0


def test_scenario_verify_3926_blocked_missing_competent_judge_writes_no_claims(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-3926-BLOCKED: missing Exp 3925 does not fabricate metrics."""

    _write_fixture_repo(tmp_path, n_items=6)
    (tmp_path / "results" / "experiment_3925_competent_judge_build.json").unlink()
    output_path = tmp_path / "results" / "exp3926.json"

    artifact = exp3926.run_experiment(
        exp3926.ExperimentConfig(
            repo_root=tmp_path,
            output_path=output_path,
            min_corpus_items=6,
        ),
        cuda_probe=_ok_cuda,
        write=True,
    )

    assert output_path.is_file()
    assert artifact["honest_verdict"] == "blocked_upstream_competent_judge_not_ready"
    assert artifact["energy_auroc"] is None
    assert artifact["llm_judge_auroc"] is None
    assert artifact["judge_positive_control_passed"] is False
    assert artifact["accuracy_parity"] is False
    assert artifact["pareto_dominates"] is False
    assert artifact == json.loads(output_path.read_text(encoding="utf-8"))


def test_req_verify_3926_artifact_uses_bare_fields_and_valid_gate(tmp_path: Path) -> None:
    """REQ-VERIFY-3926: artifact fields stay bare and verdict follows the gate."""

    labels = [0, 1, 0, 1]
    energy_scores = [0.1, 0.9, 0.2, 0.8]
    llm_scores = [0.2, 0.8, 0.3, 0.7]
    artifact = exp3926.build_artifact(
        config=exp3926.ExperimentConfig(
            repo_root=tmp_path,
            started_monotonic_s=0.0,
            clock=lambda: 65.0,
            bootstrap_reps=20,
            min_corpus_items=4,
        ),
        preconditions_checked=[exp3926.PreconditionCheck("cuda_available", True, "ok")],
        exp3925_source={"judge_model_used": "fixture", "fixture_auroc": 0.9},
        corpus_source={"artifact_path": "results/exp3884.json"},
        model_specs={"model_used": "fixture", "gguf_path": "/models/fixture.gguf"},
        energy_cost={
            "auroc": 1.0,
            "total_wall_s": 0.01,
            "per_item_wall_ms": 1.0,
            "est_tokens": 10,
            "est_flops": 100,
            "n_items": 4,
        },
        llm_cost={
            "auroc": 1.0,
            "total_wall_s": 1.0,
            "per_item_wall_ms": 100.0,
            "est_tokens": 1000,
            "est_flops": 100_000,
            "n_items": 4,
        },
        labels=labels,
        energy_scores=energy_scores,
        llm_scores=llm_scores,
    )

    exp3926.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete: efficiency_VALID_EARNS_PLACE")
    assert artifact["judge_positive_control_passed"] is True
    assert artifact["accuracy_parity"] is True
    assert artifact["pareto_dominates"] is True
    assert artifact["cost_ratio_walltime"] == pytest.approx(100.0)
    assert not isinstance(artifact["energy_auroc"], dict)


def test_scenario_verify_3926_positive_control_blocks_parity_verdict(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3926-BLOCKED: below-threshold judge AUROC blocks parity claims."""

    artifact = exp3926.build_artifact(
        config=exp3926.ExperimentConfig(
            repo_root=tmp_path,
            started_monotonic_s=0.0,
            clock=lambda: 70.0,
            bootstrap_reps=20,
            min_corpus_items=4,
        ),
        preconditions_checked=[],
        exp3925_source={"judge_model_used": "fixture", "fixture_auroc": 0.9},
        corpus_source={"artifact_path": "results/exp3884.json"},
        model_specs={"model_used": "fixture"},
        energy_cost={
            "auroc": 1.0,
            "total_wall_s": 0.01,
            "per_item_wall_ms": 1.0,
            "est_tokens": 10,
            "est_flops": 100,
            "n_items": 4,
        },
        llm_cost={
            "auroc": 0.5,
            "total_wall_s": 1.0,
            "per_item_wall_ms": 100.0,
            "est_tokens": 1000,
            "est_flops": 100_000,
            "n_items": 4,
        },
        labels=[0, 1, 0, 1],
        energy_scores=[0.1, 0.9, 0.2, 0.8],
        llm_scores=[0.5, 0.5, 0.5, 0.5],
    )

    assert artifact["honest_verdict"] == "blocked_competent_judge_failed_positive_control_on_corpus"
    assert artifact["judge_positive_control_passed"] is False


def test_req_verify_3926_scripted_end_to_end_uses_competent_judge(tmp_path: Path) -> None:
    """REQ-VERIFY-3926: both verifier paths run through the cost wrapper."""

    _write_fixture_repo(tmp_path, n_items=6)

    def loader(
        source: dict[str, object],
        _config: exp3926.ExperimentConfig,
    ) -> tuple[object, dict[str, object]]:
        assert source["judge_model_used"] == "gemma-4-26B-A4B-it"
        return ScriptedGenerator(), {
            "model_used": "gemma-4-26B-A4B-it",
            "gguf_path": "/tmp/gemma.gguf",
            "n_gpu_layers_used": -1,
        }

    artifact = exp3926.run_experiment(
        exp3926.ExperimentConfig(
            repo_root=tmp_path,
            started_monotonic_s=0.0,
            clock=lambda: 65.0,
            min_corpus_items=6,
            bootstrap_reps=20,
        ),
        cuda_probe=_ok_cuda,
        generator_loader=loader,
        write=False,
    )

    exp3926.validate_artifact(artifact)
    assert artifact["n_items"] == 6
    assert artifact["energy_cost"]["n_items"] == 6
    assert artifact["llm_cost"]["n_items"] == 6
    assert artifact["energy_auroc"] == 1.0
    assert artifact["llm_judge_auroc"] == 1.0
    assert artifact["judge_positive_control_passed"] is True


def test_req_verify_3926_small_helpers_and_validation_branches(tmp_path: Path) -> None:
    """REQ-VERIFY-3926: helper branches reject malformed terminal payloads."""

    cfg = exp3926.ExperimentConfig(repo_root=tmp_path, output_path=tmp_path / "out.json")

    assert cfg.resolved_output_path() == tmp_path / "out.json"
    assert cfg.start_time() > 0.0
    assert cfg.venv_python() == tmp_path / ".venv" / "bin" / "python"
    assert exp3926.PreconditionCheck("x", True, "ok").as_dict() == {
        "resource": "x",
        "available": True,
        "detail": "ok",
    }
    assert exp3926._label_to_error(True) == 1
    assert exp3926._label_to_error(0) == 0
    with pytest.raises(ValueError, match="unsupported"):
        exp3926._label_to_error("maybe")
    with pytest.raises(ValueError, match="AUROC"):
        exp3926._auroc([0, 0], [0.1, 0.2])
    assert exp3926._bootstrap_ci95([0, 1], [0.1, 0.9], seed=1, reps=1) == (1.0, 1.0)
    assert exp3926._safe_llama_token_count(object(), "hello world", add_bos=True) == 3
    assert exp3926._ratio("bad", 1) is None
    assert exp3926._ratio(1, 0) is None
    assert exp3926._render_metric(None) == "nan"

    blocked = exp3926.build_blocked_artifact(
        reason="blocked_upstream_competent_judge_not_ready",
        preconditions_checked=[],
        duration_s=0.1,
    )
    for mutation, pattern in (
        ({}, "missing required"),
        ({"honest_verdict": "not-terminal"}, "terminal prefix"),
        ({"judge_positive_control_passed": "false"}, "bare bool"),
        ({"energy_auroc": "1.0"}, "bare float"),
        ({"energy_per_item_ms": "1.0"}, "bare float"),
        ({"n_items": 1.2}, "bare int"),
        ({"duration_s": "1.0"}, "bare number"),
        ({"reproducibility_checksum": "short"}, "sha256"),
        ({"energy_auroc": 0.5}, "must not claim"),
    ):
        candidate = dict(blocked)
        if mutation:
            candidate.update(mutation)
        else:
            candidate.pop("random_seed")
        with pytest.raises(ValueError, match=pattern):
            exp3926.validate_artifact(candidate)

    candidate = dict(blocked, corpus_source={"path": "wrapped"})
    with pytest.raises(ValueError, match="corpus_source"):
        exp3926.validate_artifact(candidate)

    positive_blocked = dict(
        blocked,
        honest_verdict="blocked_competent_judge_failed_positive_control_on_corpus",
        judge_positive_control_passed=True,
    )
    with pytest.raises(ValueError, match="positive-control"):
        exp3926.validate_artifact(positive_blocked)


def test_req_verify_3926_upstream_and_corpus_error_branches(tmp_path: Path) -> None:
    """REQ-VERIFY-3926: malformed upstream inputs fail before scoring claims."""

    with pytest.raises(FileNotFoundError, match="3925"):
        exp3926.load_exp3925_source(tmp_path)

    path3925 = tmp_path / "results" / "experiment_3925_competent_judge_build.json"
    _write_json(path3925, [])
    with pytest.raises(ValueError, match="not a JSON object"):
        exp3926.load_exp3925_source(tmp_path)
    _write_json(path3925, {"unit_test_passed": False, "fixture_auroc": 0.9})
    with pytest.raises(ValueError, match="unit_test"):
        exp3926.load_exp3925_source(tmp_path)
    _write_json(path3925, {"unit_test_passed": True, "fixture_auroc": 0.65})
    with pytest.raises(ValueError, match="fixture_auroc"):
        exp3926.load_exp3925_source(tmp_path)

    with pytest.raises(FileNotFoundError, match="3884"):
        exp3926.load_exp3884_corpus(exp3926.ExperimentConfig(repo_root=tmp_path, min_corpus_items=1))

    artifact_path = tmp_path / "results" / "experiment_3884_in_distribution_error_rich_corpus.json"
    _write_json(artifact_path, [])
    with pytest.raises(ValueError, match="not a JSON object"):
        exp3926.load_exp3884_corpus(exp3926.ExperimentConfig(repo_root=tmp_path, min_corpus_items=1))
    _write_json(artifact_path, {"corpus_path": "data/missing.json", "per_item_ensemble_scores_path": "results/scores.json"})
    with pytest.raises(FileNotFoundError, match="corpus missing"):
        exp3926.load_exp3884_corpus(exp3926.ExperimentConfig(repo_root=tmp_path, min_corpus_items=1))
    _write_json(tmp_path / "data" / "corpus.json", {"items": []})
    _write_json(
        artifact_path,
        {"corpus_path": "data/corpus.json", "per_item_ensemble_scores_path": "results/missing.json"},
    )
    with pytest.raises(FileNotFoundError, match="score file"):
        exp3926.load_exp3884_corpus(exp3926.ExperimentConfig(repo_root=tmp_path, min_corpus_items=1))
    _write_json(artifact_path, {"corpus_path": "data/corpus.json", "per_item_ensemble_scores_path": "results/scores.json"})
    _write_json(tmp_path / "results" / "scores.json", {"items": [{"corpus_item_id": "x"}]})
    with pytest.raises(ValueError, match="different lengths"):
        exp3926.load_exp3884_corpus(exp3926.ExperimentConfig(repo_root=tmp_path, min_corpus_items=1))

    _write_json(tmp_path / "data" / "corpus.json", {"items": [{"corpus_item_id": "x", "label": "correct", "step_text": "ok"}]})
    _write_json(tmp_path / "results" / "scores.json", {"items": [{"corpus_item_id": "y", "label": "correct", "carnot_ensemble_score": 0.1}]})
    with pytest.raises(ValueError, match="order mismatch"):
        exp3926.load_exp3884_corpus(exp3926.ExperimentConfig(repo_root=tmp_path, min_corpus_items=1))
    _write_json(tmp_path / "results" / "scores.json", {"items": [{"corpus_item_id": "x", "label": "incorrect", "carnot_ensemble_score": 0.1}]})
    with pytest.raises(ValueError, match="label mismatch"):
        exp3926.load_exp3884_corpus(exp3926.ExperimentConfig(repo_root=tmp_path, min_corpus_items=1))
    _write_json(tmp_path / "data" / "corpus.json", {"items": [{"corpus_item_id": "x", "label": "correct", "step_text": " "}]})
    _write_json(tmp_path / "results" / "scores.json", {"items": [{"corpus_item_id": "x", "label": "correct", "carnot_ensemble_score": 0.1}]})
    with pytest.raises(ValueError, match="lacks step text"):
        exp3926.load_exp3884_corpus(exp3926.ExperimentConfig(repo_root=tmp_path, min_corpus_items=1))
    _write_json(tmp_path / "data" / "corpus.json", {"items": [{"corpus_item_id": "x", "label": "correct", "step_text": "ok"}]})
    with pytest.raises(ValueError, match="required>=2"):
        exp3926.load_exp3884_corpus(exp3926.ExperimentConfig(repo_root=tmp_path, min_corpus_items=2))
    with pytest.raises(ValueError, match="both labels"):
        exp3926.load_exp3884_corpus(exp3926.ExperimentConfig(repo_root=tmp_path, min_corpus_items=1))


def test_scenario_verify_3926_precondition_block_order_and_write_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-3926-BLOCKED: preflight reports the first failed gate."""

    _write_fixture_repo(tmp_path, n_items=6)

    no_cuda = exp3926.probe_preconditions(
        exp3926.ExperimentConfig(repo_root=tmp_path, min_corpus_items=6),
        cuda_probe=lambda _cfg: exp3926.PreconditionCheck("cuda_available", False, "no cuda"),
    )
    assert no_cuda[1] == "blocked_no_cuda"

    real_import = exp3926.importlib.import_module

    def blocked_import(name: str) -> object:
        if name == "carnot.verify.cost_instrumented_verification":
            raise ImportError("blocked")
        return real_import(name)

    monkeypatch.setattr(exp3926.importlib, "import_module", blocked_import)
    blocked_cost = exp3926.probe_preconditions(
        exp3926.ExperimentConfig(repo_root=tmp_path, min_corpus_items=6),
        cuda_probe=_ok_cuda,
    )
    assert blocked_cost[1] == "blocked_upstream_cost_harness"
    monkeypatch.setattr(exp3926.importlib, "import_module", real_import)

    _write_json(
        tmp_path / "results" / "experiment_3884_in_distribution_error_rich_corpus.json",
        {"corpus_path": "data/missing.json", "per_item_ensemble_scores_path": "results/scores.json"},
    )
    blocked_corpus = exp3926.probe_preconditions(
        exp3926.ExperimentConfig(repo_root=tmp_path, min_corpus_items=6),
        cuda_probe=_ok_cuda,
    )
    assert blocked_corpus[1] == "blocked_exp3884_corpus_not_ready"

    _write_fixture_repo(tmp_path, n_items=6)
    output_path = tmp_path / "results" / "written-success.json"
    artifact = exp3926.run_experiment(
        exp3926.ExperimentConfig(
            repo_root=tmp_path,
            output_path=output_path,
            started_monotonic_s=0.0,
            clock=lambda: 65.0,
            min_corpus_items=6,
            bootstrap_reps=20,
        ),
        cuda_probe=_ok_cuda,
        generator_loader=lambda _source, _config: (
            ScriptedGenerator(),
            {"model_used": "gemma-4-26B-A4B-it", "gguf_path": "/tmp/gemma.gguf"},
        ),
        write=True,
    )
    assert output_path.is_file()
    assert artifact == json.loads(output_path.read_text(encoding="utf-8"))


def test_req_verify_3926_verdict_and_llm_default_branches(tmp_path: Path) -> None:
    """REQ-VERIFY-3926: verdict gates and default LLM FLOP parameter path are covered."""

    assert exp3926._classify_verdict(
        judge_positive_control_passed=True,
        accuracy_parity=True,
        pareto_dominates=False,
        cost_ratio_walltime=20.0,
        energy_auroc=0.7,
        llm_judge_auroc=0.7,
        duration_s=1.0,
    ) == "blocked_llm_judge_not_invoked"
    assert exp3926._classify_verdict(
        judge_positive_control_passed=True,
        accuracy_parity=False,
        pareto_dominates=False,
        cost_ratio_walltime=10.0,
        energy_auroc=0.5,
        llm_judge_auroc=0.7,
        duration_s=65.0,
    ) == "complete: efficiency_INCONCLUSIVE_cost_ratio_walltime<=10"
    assert "JUDGE_MORE_ACCURATE" in exp3926._classify_verdict(
        judge_positive_control_passed=True,
        accuracy_parity=False,
        pareto_dominates=False,
        cost_ratio_walltime=20.0,
        energy_auroc=0.5,
        llm_judge_auroc=0.7,
        duration_s=65.0,
    )

    result = exp3926.run_competent_llm_judge_verifier(
        [{"step": "This step is correct.", "gold_error": 0}],
        generator=ScriptedGenerator(),
        model_path=None,
        model_params=None,
    )
    assert result["scores"] == [0.1]
    assert result["est_flops"] > 0
