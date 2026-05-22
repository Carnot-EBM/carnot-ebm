"""Tests for Exp 2841 HaluEval + FEVER factuality readiness pilot.

Spec: REQ-VERIFY-2841,
      SCENARIO-VERIFY-2841-BLOCKED,
      SCENARIO-VERIFY-2841-PILOT.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import halueval_fever_pilot as mod
from carnot.eval.halueval_fever_pilot import (
    ExperimentConfig,
    PilotExample,
    PreconditionCheck,
    model_specs_from_exp2836,
    run_experiment,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_exp2836(path: Path, *, model_path: Path, ready: bool = True) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"gguf")
    _write_json(
        path,
        {
            "sota_runtime_ready": ready,
            "selected_python": "/venv/bin/python",
            "cached_sota_pair_result": {"called": True, "error": None, "result": None},
            "smoke_load_results": [
                {
                    "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                    "model_path": str(model_path),
                    "load_success": True,
                    "headline_usable": True,
                }
            ],
            "sota_models_cached": [
                {
                    "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                    "path": str(model_path),
                    "sha256": "a" * 64,
                    "size_bytes": 123,
                }
            ],
            "model_specs": {
                "primary": list(mod.PRIMARY_SOTA_MODEL_IDS),
                "legacy_cpu_smoke_only": list(mod.LEGACY_CPU_SMOKE_ONLY),
            },
        },
    )


def _examples(dataset: str, count: int = 6) -> list[PilotExample]:
    rows: list[PilotExample] = []
    for idx in range(count):
        label = idx % 2
        rows.append(
            PilotExample(
                dataset=dataset,
                example_id=f"{dataset}-{idx}",
                prompt=f"Context {idx}",
                candidate=(
                    f"grounded answer {idx}"
                    if label == 0
                    else f"hallucinated answer {idx} actually contradicts 1 but 2"
                ),
                label=label,
                source="fixture",
                reference="fixture reference",
            )
        )
    return rows


def test_req_verify_2841_model_specs_record_selected_sota_path(tmp_path: Path) -> None:
    """REQ-VERIFY-2841: Exp2836 selected Python and mandated GGUF path are recorded."""

    model_path = tmp_path / "models" / "gemma-4-26b.gguf"
    _write_exp2836(tmp_path / "results" / mod.EXP2836_FILENAME, model_path=model_path)

    specs = model_specs_from_exp2836(
        mod.load_exp2836_preflight(tmp_path / "results" / mod.EXP2836_FILENAME)
    )

    assert specs["sota_runtime_ready"] is True
    assert specs["selected_python"] == "/venv/bin/python"
    assert specs["selected_model_path"] == str(model_path)
    assert specs["selected_model_hf_id"] == "unsloth/gemma-4-26B-A4B-it-GGUF"
    assert specs["headline_required_any_of"] == list(mod.PRIMARY_SOTA_MODEL_IDS)
    assert specs["legacy_cpu_smoke_only"] == list(mod.LEGACY_CPU_SMOKE_ONLY)
    assert mod._extract_model_paths({"outer": [{"resolved_path": "/cache/model.gguf"}]}) == [
        "/cache/model.gguf"
    ]


def test_scenario_verify_2841_blocked_without_labeled_dataset(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-2841-BLOCKED: no HaluEval/FEVER rows block without AUROC."""

    _write_exp2836(
        tmp_path / "results" / mod.EXP2836_FILENAME,
        model_path=tmp_path / "models" / "gemma.gguf",
    )

    artifact = run_experiment(
        ExperimentConfig(
            repo_root=tmp_path,
            results_dir=tmp_path / "results",
            started_at=2.0,
            clock=lambda: 5.5,
        ),
        dataset_loader=lambda _config: {},
        scorer=lambda _example: {"ensemble_energy": 0.5, "per_verifier_energy": {}},
        write=True,
    )

    assert artifact["honest_verdict"] == "blocked_labeled_pilot_datasets"
    assert artifact["pilot_only"] is True
    assert artifact["datasets_loaded"] == []
    assert artifact["n_examples"] == 0
    assert artifact["pilot_auroc_by_dataset"] == {}
    assert artifact["duration_s"] == pytest.approx(3.5)
    assert artifact["candidate_generation"]["loaded_candidate_count"] == 0
    saved = json.loads((tmp_path / "results" / mod.OUTPUT_FILENAME).read_text(encoding="utf-8"))
    assert saved == artifact


def test_scenario_verify_2841_pilot_success_reports_dataset_aurocs(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-2841-PILOT: loaded corpora emit pilot AUROC and CI fields."""

    _write_exp2836(
        tmp_path / "results" / mod.EXP2836_FILENAME,
        model_path=tmp_path / "models" / "gemma.gguf",
    )

    def score(example: PilotExample) -> dict[str, object]:
        base = 0.2 if example.label == 0 else 0.8
        return {
            "ensemble_energy": base,
            "per_verifier_energy": {
                "tier0r_curry_howard": base,
                "tier0u_logical_consistency": base / 2.0,
            },
        }

    artifact = run_experiment(
        ExperimentConfig(
            repo_root=tmp_path,
            results_dir=tmp_path / "results",
            started_at=10.0,
            clock=lambda: 18.0,
            bootstrap_reps=50,
        ),
        dataset_loader=lambda _config: {
            "HaluEval": _examples("HaluEval"),
            "FEVER": _examples("FEVER"),
        },
        scorer=score,
        write=False,
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["pilot_only"] is True
    assert artifact["datasets_loaded"] == ["FEVER", "HaluEval"]
    assert artifact["n_examples"] == 12
    assert set(artifact["pilot_auroc_by_dataset"]) == {"FEVER", "HaluEval"}
    for metrics in artifact["pilot_auroc_by_dataset"].values():
        assert metrics["auroc"] == pytest.approx(1.0)
        assert metrics["ci_method"] == "bootstrap_percentile"
        assert metrics["label_counts"] == {"0": 3, "1": 3}
        assert metrics["energy_stability"]["all_finite"] is True
        assert metrics["ready_for_full_benchmark"] is True
    assert "N>=500" in artifact["recommendation"]
    assert artifact["model_specs"]["scorer_or_generator_model_paths_used"] == [
        str(tmp_path / "models" / "gemma.gguf")
    ]


def test_req_verify_2841_normalizes_halueval_and_fever_rows() -> None:
    """REQ-VERIFY-2841: HaluEval/FEVER labels map to hallucination-positive rows."""

    halueval = mod.examples_from_halueval_rows(
        [
            {
                "knowledge": "Paris is in France.",
                "question": "Where is Paris?",
                "right_answer": "France.",
                "hallucinated_answer": "Australia.",
            }
        ],
        limit=2,
    )
    fever = mod.examples_from_fever_rows(
        [
            {
                "id": 1,
                "claim": "Paris is in France.",
                "evidence": "Paris is the capital of France.",
                "label": "SUPPORTS",
            },
            {
                "id": 2,
                "claim": "Paris is in Australia.",
                "evidence": "Paris is the capital of France.",
                "label": "REFUTES",
            },
        ],
        limit=2,
    )

    assert [row.label for row in halueval] == [0, 1]
    assert [row.label for row in fever] == [0, 1]
    assert halueval[0].score_text.startswith("Context:")
    assert fever[1].score_text.startswith("Evidence:")


def test_req_verify_2841_scoring_and_probe_failure_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-2841: probes, scoring guards, and CLI paths stay explicit."""

    assert mod.compute_auroc([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2]) == pytest.approx(1.0)
    assert mod.compute_auroc([0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="both positive and negative"):
        mod.compute_auroc([1, 1], [0.1, 0.2])

    assert mod.bootstrap_auroc_ci([0, 1], [0.1, 0.9], reps=0) == (1.0, 1.0)
    assert mod._blocked_verdict([PreconditionCheck("mystery", False, "x")]) == "blocked_mystery"
    assert mod._run_json_probe(
        selected_python="",
        repo_root=tmp_path,
        script="print('x')",
        resource="probe",
    ) == PreconditionCheck("probe", False, "selected_python missing")

    def failing_runner(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(["python"], 2, "not-json", "boom")

    failed = mod._run_json_probe(
        selected_python="/venv/python",
        repo_root=tmp_path,
        script="print('x')",
        resource="probe",
        command_runner=failing_runner,
    )
    assert failed == PreconditionCheck("probe", False, "boom")

    def invalid_json_runner(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del kwargs
        return subprocess.CompletedProcess(args[0], 0, "not-json", "")

    invalid = mod._run_json_probe(
        selected_python="/venv/python",
        repo_root=tmp_path,
        script="print('x')",
        resource="probe",
        command_runner=invalid_json_runner,
    )
    assert invalid.available is False
    assert "invalid JSON" in invalid.detail

    def ok_runner(
        command: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: int,
        check: bool,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        assert capture_output is True and text is True and check is False
        assert timeout > 0 and "PYTHONPATH" in env
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps({"available": True, "detail": "dataset ok"}),
            "",
        )

    check = mod._dataset_probe_check(
        "/venv/python",
        tmp_path,
        resource="halueval_dataset",
        script="print('ok')",
        command_runner=ok_runner,
    )
    assert check == PreconditionCheck("halueval_dataset", True, "dataset ok")

    calls: list[ExperimentConfig] = []

    def fake_run_experiment(config: ExperimentConfig) -> dict[str, object]:
        calls.append(config)
        return {"honest_verdict": "blocked_unit_test"}

    monkeypatch.setattr(mod, "run_experiment", fake_run_experiment)
    assert (
        mod.main(
            [
                "--repo-root",
                str(tmp_path),
                "--results-dir",
                str(tmp_path / "custom-results"),
                "--sample-per-dataset",
                "7",
            ]
        )
        == 0
    )
    assert calls[0].repo_root == tmp_path
    assert calls[0].results_dir == tmp_path / "custom-results"
    assert calls[0].sample_per_dataset == 7


def test_req_verify_2841_edge_paths_for_new_code_coverage(tmp_path: Path) -> None:
    """REQ-VERIFY-2841: edge cases stay explicit instead of implicit fallthrough."""

    config = ExperimentConfig(repo_root=tmp_path, exp2836_path=tmp_path / "custom.json")
    assert config.preflight_path() == tmp_path / "custom.json"
    assert mod.load_exp2836_preflight(tmp_path / "missing.json") == {}
    assert PilotExample("x", "id", "Prompt", "Candidate", 0, "fixture").score_text == (
        "Prompt\nCandidate: Candidate"
    )

    fever = mod.examples_from_fever_rows(
        [
            {"id": "skip-label", "claim": "ignored", "label": "UNKNOWN"},
            {"id": "skip-claim", "claim": "", "label": "SUPPORTS"},
            {
                "id": "numeric",
                "claim": "A numeric label row.",
                "label": "1",
                "evidence": [["E", 1], "plain"],
            },
        ],
        limit=3,
    )
    assert len(fever) == 1
    assert fever[0].label == 1
    assert fever[0].reference == "E 1 plain"
    assert mod._stringify_evidence({"not": "a list"}) == "{'not': 'a list'}"

    score_payload = mod.default_score_example(
        PilotExample(
            dataset="fixture",
            example_id="default-score",
            prompt="Context: 1 + 1 = 2.",
            candidate="The answer is 2.",
            label=0,
            source="fixture",
        )
    )
    assert 0.0 <= float(score_payload["ensemble_energy"]) <= 1.0

    with pytest.raises(ValueError, match="same length"):
        mod.compute_auroc([0, 1], [0.1])
    assert mod.bootstrap_auroc_ci([0, 1], [0.1, 0.9], reps=1, seed=0) == (1.0, 1.0)

    bad_metrics = mod.evaluate_dataset(
        [
            PilotExample("fixture", "bad-payload", "p", "c", 0, "fixture"),
            PilotExample("fixture", "nan", "p", "c", 1, "fixture"),
            PilotExample("fixture", "one-class", "p", "c", 0, "fixture"),
        ],
        scorer=lambda example: (
            {"ensemble_energy": "not-float"}
            if example.example_id == "bad-payload"
            else (
                {"ensemble_energy": float("nan"), "per_verifier_energy": {}}
                if example.example_id == "nan"
                else {"ensemble_energy": 0.3, "per_verifier_energy": {"bad": object()}}
            )
        ),
        bootstrap_reps=1,
        seed=1,
    )
    assert bad_metrics["score_failures"] == 2
    assert bad_metrics["auroc"] is None
    assert bad_metrics["ready_for_full_benchmark"] is False

    assert mod._recommendation(
        {
            "Ready": {"ready_for_full_benchmark": True},
            "Blocked": {"ready_for_full_benchmark": False},
        }
    ).startswith("Scale Ready")
    assert mod._recommendation({"Blocked": {"ready_for_full_benchmark": False}}).startswith(
        "Do not scale"
    )

    _write_exp2836(
        tmp_path / "results" / mod.EXP2836_FILENAME,
        model_path=tmp_path / "models" / "gemma.gguf",
    )

    def raising_loader(_config: ExperimentConfig) -> dict[str, list[PilotExample]]:
        raise RuntimeError("dataset boom")

    blocked = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        dataset_loader=raising_loader,
        write=False,
    )
    assert blocked["honest_verdict"] == "blocked_labeled_pilot_datasets"
    assert "dataset boom" in json.dumps(blocked["preconditions_checked"])
