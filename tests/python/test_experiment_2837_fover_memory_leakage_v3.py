"""Tests for Exp 2837 FoVer memory-leakage isolation v3.

Spec: REQ-VERIFY-2837,
      SCENARIO-VERIFY-2837,
      SCENARIO-VERIFY-2837-BLOCKED.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import fover_memory_leakage_v3 as mod
from carnot.eval.fover_memory_leakage_v3 import (
    ConditionMeasurement,
    ExperimentConfig,
    PreconditionCheck,
    compute_auroc,
    discover_fr11_state_files,
    model_specs_from_exp2836,
    run_experiment,
    score_fover_subset,
    state_files_restored_sha_match,
    temporarily_move_state_files,
)


SEEDS = (42, 137)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_fover_rows(path: Path, n_per_class: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for idx in range(n_per_class):
        rows.append(
            {
                "question_id": f"ok_{idx}",
                "step_text": f"First compute {idx} + {idx} = {2 * idx}. Therefore {2 * idx}.",
                "label": "correct",
            }
        )
        rows.append(
            {
                "question_id": f"bad_{idx}",
                "step_text": f"{idx} apples. Initial state contradicts command {idx + 9}.",
                "label": "incorrect",
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_exp2836(path: Path, *, ready: bool = True, selected_python: str = "/venv/python") -> None:
    _write_json(
        path,
        {
            "sota_runtime_ready": ready,
            "selected_python": selected_python,
            "cached_sota_pair_result": {"called": True, "error": None, "result": None},
            "sota_models_cached": [
                {
                    "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                    "path": "/cache/gemma-4-26B-A4B-it-Q4_K_M.gguf",
                    "sha256": "a" * 64,
                    "size_bytes": 123,
                }
            ],
            "smoke_load_results": [
                {
                    "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                    "model_path": "/cache/gemma-4-26B-A4B-it-Q4_K_M.gguf",
                    "load_success": True,
                    "headline_usable": True,
                }
            ],
            "model_specs": {
                "primary": list(mod.PRIMARY_SOTA_MODEL_IDS),
                "legacy_cpu_smoke_only": list(mod.LEGACY_CPU_SMOKE_ONLY),
            },
        },
    )


def _write_state(root: Path) -> None:
    session_state = {
        "schema": "carnot.session_memory.v1",
        "case_memory": {
            "entries": [
                {
                    "key": {
                        "benchmark_slice": "fover:bad_0",
                        "violation_families": ["fr11_v7_dvi_verified_incorrect"],
                        "prompt_sketch": "initial state contradicts command",
                    },
                    "prompt_tokens": ["initial", "state", "contradicts", "command"],
                    "violation_types": ["fr11_v7_dvi_verified_incorrect"],
                }
            ]
        },
    }
    _write_json(
        root / "results" / "session_memory_1447" / "fr11_v7" / "session_state.json",
        session_state,
    )
    _write_json(root / "results" / "nexus_constraint_memory_v2.json", {"violations": {}})
    (root / "data").mkdir(exist_ok=True)
    (root / "data" / "fr11_zenil_distill_v2.jsonl").write_text(
        json.dumps({"question_id": "bad_1", "is_correct": False}) + "\n",
        encoding="utf-8",
    )


def _minimal_repo(root: Path, *, ready: bool = True, selected_python: str = "/venv/python") -> None:
    _write_fover_rows(root / "data" / "fover_corpus.jsonl", n_per_class=8)
    _write_state(root)
    _write_exp2836(
        root / "results" / "experiment_2836_sota_runtime_preflight.json",
        ready=ready,
        selected_python=selected_python,
    )


def test_req_verify_2837_model_specs_record_exp2836_smoke_path(tmp_path: Path) -> None:
    """REQ-VERIFY-2837: Exp 2836 usable SOTA path is recorded even when pair is null."""

    preflight_path = tmp_path / "results" / "experiment_2836_sota_runtime_preflight.json"
    _write_exp2836(preflight_path, selected_python="/tmp/venv/bin/python")
    preflight = mod.load_exp2836_preflight(preflight_path)
    specs = model_specs_from_exp2836(preflight)

    assert specs["sota_runtime_ready"] is True
    assert specs["selected_python"] == "/tmp/venv/bin/python"
    assert specs["cached_sota_pair_model_paths"] == []
    assert specs["selected_model_path"] == "/cache/gemma-4-26B-A4B-it-Q4_K_M.gguf"
    assert specs["selected_model_hf_id"] == "unsloth/gemma-4-26B-A4B-it-GGUF"
    assert specs["headline_required_any_of"] == list(mod.PRIMARY_SOTA_MODEL_IDS)
    assert specs["legacy_cpu_smoke_only"] == list(mod.LEGACY_CPU_SMOKE_ONLY)


def test_scenario_verify_2837_blocked_runtime_gate_writes_null_metrics(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-2837-BLOCKED: exp2836 not ready blocks without AUROC."""

    _minimal_repo(tmp_path, ready=False)
    artifact = run_experiment(
        ExperimentConfig(
            repo_root=tmp_path,
            results_dir=tmp_path / "results",
            n_examples=4,
            random_seeds=SEEDS,
            started_at=10.0,
            clock=lambda: 12.0,
        ),
        write=True,
    )

    assert artifact["honest_verdict"] == "blocked_sota_runtime_not_ready"
    assert artifact["condition_a_production_auroc_mean"] is None
    assert artifact["condition_b_architecture_only_auroc_mean"] is None
    assert artifact["learning_contribution"] is None
    assert artifact["per_seed_results"] == []
    assert artifact["state_files_restored_sha_match"] is True
    checks = {entry["resource"]: entry for entry in artifact["preconditions_checked"]}
    assert checks["exp2836_sota_runtime_ready"]["available"] is False
    saved = json.loads((tmp_path / "results" / mod.OUTPUT_FILENAME).read_text(encoding="utf-8"))
    assert saved == artifact


def test_req_verify_2837_state_manifest_move_and_restore(tmp_path: Path) -> None:
    """REQ-VERIFY-2837: FR-11 files are moved aside and restored with matching SHA."""

    _minimal_repo(tmp_path)
    state_files = discover_fr11_state_files(tmp_path)
    assert state_files

    with temporarily_move_state_files(tmp_path, state_files, tmp_path / "backup"):
        assert discover_fr11_state_files(tmp_path) == []
        for item in state_files:
            assert not (tmp_path / str(item["path"])).exists()

    assert state_files_restored_sha_match(tmp_path, state_files) is True
    assert discover_fr11_state_files(tmp_path) == state_files

    (tmp_path / state_files[0]["path"]).write_text("changed", encoding="utf-8")
    assert state_files_restored_sha_match(tmp_path, state_files) is False


def test_req_verify_2837_scoring_uses_memory_only_for_production(tmp_path: Path) -> None:
    """REQ-VERIFY-2837: architecture-only scoring omits FR-11 memory signal."""

    _minimal_repo(tmp_path)
    production = score_fover_subset(
        repo_root=tmp_path,
        seed=42,
        n_examples=4,
        condition=mod.CONDITION_PRODUCTION,
        require_no_state=False,
    )
    architecture = score_fover_subset(
        repo_root=tmp_path,
        seed=42,
        n_examples=4,
        condition=mod.CONDITION_ARCHITECTURE_ONLY,
        require_no_state=False,
    )

    assert production.condition == mod.CONDITION_PRODUCTION
    assert production.n_examples == 4
    assert production.state_visible_count > 0
    assert production.fr11_state_loaded is True
    assert "fr11_session_memory" in production.per_verifier_auroc
    assert "fr11_session_memory" not in architecture.per_verifier_auroc
    assert production.auroc >= architecture.auroc
    assert compute_auroc([0, 0, 1, 1], [0.1, 0.1, 0.9, 0.9]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="both positive and negative"):
        compute_auroc([1, 1], [0.1, 0.2])


def test_scenario_verify_2837_success_dual_condition_summary(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-2837: each seed has A/B AUROC and B sees no state."""

    _minimal_repo(tmp_path, selected_python=sys.executable)
    calls: list[tuple[int, str, int]] = []

    def fake_condition_runner(
        config: ExperimentConfig,
        selected_python: str,
        seed: int,
        condition: str,
        require_no_state: bool,
    ) -> ConditionMeasurement:
        visible = len(discover_fr11_state_files(config.repo_root))
        calls.append((seed, condition, visible))
        if require_no_state:
            assert visible == 0
        offset = 0.01 if seed == 137 else 0.0
        is_prod = condition == mod.CONDITION_PRODUCTION
        return ConditionMeasurement(
            seed=seed,
            condition=condition,
            auroc=(0.91 + offset) if is_prod else (0.84 + offset),
            per_verifier_auroc={"tier0r_curry_howard": 0.88 + offset},
            n_examples=4,
            state_visible_count=visible,
            fr11_state_loaded=is_prod and visible > 0,
            subset_sha256=f"subset-{seed}",
            python_executable=selected_python,
        )

    artifact = run_experiment(
        ExperimentConfig(
            repo_root=tmp_path,
            results_dir=tmp_path / "results",
            n_examples=4,
            random_seeds=SEEDS,
            started_at=1.0,
            clock=lambda: 9.0,
        ),
        condition_runner=fake_condition_runner,
        write=True,
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["condition_a_production_auroc_mean"] == pytest.approx(0.915)
    assert artifact["condition_b_architecture_only_auroc_mean"] == pytest.approx(0.845)
    assert artifact["learning_contribution"] == pytest.approx(0.07)
    assert artifact["condition_a_production_auroc_ci95"]["low"] < 0.915
    assert artifact["condition_b_architecture_only_auroc_ci95"]["high"] > 0.845
    assert len(artifact["per_seed_results"]) == 2
    assert all(row["condition_b_state_visible_count"] == 0 for row in artifact["per_seed_results"])
    assert all(row["python_restarted_between_conditions"] is True for row in artifact["per_seed_results"])
    assert artifact["state_files_restored_sha_match"] is True
    assert calls == [
        (42, mod.CONDITION_PRODUCTION, 3),
        (42, mod.CONDITION_ARCHITECTURE_ONLY, 0),
        (137, mod.CONDITION_PRODUCTION, 3),
        (137, mod.CONDITION_ARCHITECTURE_ONLY, 0),
    ]


def test_req_verify_2837_subprocess_runner_uses_selected_python(tmp_path: Path) -> None:
    """REQ-VERIFY-2837: condition subprocess command uses Exp 2836 selected_python."""

    _minimal_repo(tmp_path)
    config = ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results", n_examples=4)
    expected = ConditionMeasurement(
        seed=42,
        condition=mod.CONDITION_ARCHITECTURE_ONLY,
        auroc=0.75,
        per_verifier_auroc={"tier0r_curry_howard": 0.75},
        n_examples=4,
        state_visible_count=0,
        fr11_state_loaded=False,
        subset_sha256="abc",
        python_executable="/venv/python",
    )
    captured: dict[str, Any] = {}

    def fake_runner(
        command: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: int,
        check: bool,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        captured.update(
            command=command,
            capture_output=capture_output,
            text=text,
            timeout=timeout,
            check=check,
            env=env,
        )
        return subprocess.CompletedProcess(command, 0, json.dumps(expected.as_dict()), "")

    got = mod.score_condition_via_subprocess(
        config,
        "/venv/python",
        42,
        mod.CONDITION_ARCHITECTURE_ONLY,
        True,
        command_runner=fake_runner,
    )

    assert got == expected
    assert captured["command"][:3] == ["/venv/python", "-m", "carnot.eval.fover_memory_leakage_v3"]
    assert "--score-condition" in captured["command"]
    assert "--require-no-state" in captured["command"]
    assert captured["env"]["PYTHONPATH"].startswith(str(tmp_path / "python"))

    def failing_runner(*_: Any, **__: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(["bad"], 7, "", "boom")

    with pytest.raises(mod.ConditionScoringError, match="boom"):
        mod.score_condition_via_subprocess(
            config,
            "/venv/python",
            42,
            mod.CONDITION_ARCHITECTURE_ONLY,
            True,
            command_runner=failing_runner,
        )


def test_req_verify_2837_probe_preconditions_happy_path(tmp_path: Path) -> None:
    """REQ-VERIFY-2837: precondition probe names every live-resource gate."""

    _minimal_repo(tmp_path, selected_python=sys.executable)
    state_files = discover_fr11_state_files(tmp_path)
    preflight = mod.load_exp2836_preflight(
        tmp_path / "results" / "experiment_2836_sota_runtime_preflight.json"
    )
    specs = model_specs_from_exp2836(preflight)
    checks = mod.probe_preconditions(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results", n_examples=4),
        state_files,
        specs,
    )

    assert all(isinstance(check, PreconditionCheck) for check in checks)
    by_resource = {check.resource: check for check in checks}
    assert by_resource["exp2836_artifact"].available is True
    assert by_resource["exp2836_sota_runtime_ready"].available is True
    assert by_resource["exp2836_selected_python"].available is True
    assert by_resource["mandated_sota_model_path"].available is True
    assert by_resource["fover_corpus"].available is True
    assert by_resource["fr11_state_files"].available is True


def test_req_verify_2837_defensive_preflight_and_model_branches(tmp_path: Path) -> None:
    """REQ-VERIFY-2837: missing files and nested model paths are handled honestly."""

    custom_preflight = tmp_path / "custom" / "exp2836.json"
    config = ExperimentConfig(repo_root=tmp_path, exp2836_path=custom_preflight)
    assert config.preflight_path() == custom_preflight
    assert mod.load_exp2836_preflight(custom_preflight) == {}

    nested = model_specs_from_exp2836(
        {
            "sota_runtime_ready": True,
            "selected_python": "/venv/python",
            "cached_sota_pair_result": {
                "result": [{"models": [{"model_path": "/pair/model.gguf"}]}]
            },
            "smoke_load_results": [
                {
                    "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                    "model_path": "/smoke/model.gguf",
                    "load_success": True,
                    "headline_usable": True,
                }
            ],
        }
    )
    assert nested["cached_sota_pair_model_paths"] == ["/pair/model.gguf"]
    assert nested["selected_model_path"] == "/pair/model.gguf"

    checks = mod.probe_preconditions(
        config,
        [],
        nested,
    )
    by_resource = {check.resource: check for check in checks}
    assert by_resource["exp2836_artifact"].available is False
    assert by_resource["fover_corpus"].available is False
    assert by_resource["fr11_state_files"].available is False


def test_req_verify_2837_defensive_state_and_corpus_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-2837: corrupt memory and malformed FoVer rows do not fabricate data."""

    _minimal_repo(tmp_path)
    state_files = discover_fr11_state_files(tmp_path)
    missing = dict(state_files[0])
    (tmp_path / str(missing["path"])).unlink()
    with pytest.raises(mod.ConditionScoringError, match="disappeared"):
        with temporarily_move_state_files(tmp_path, [missing], tmp_path / "backup"):
            pass

    _minimal_repo(tmp_path)
    # Exercise malformed/blank FoVer rows and unsupported-label rejection.
    fover = tmp_path / "data" / "fover_corpus.jsonl"
    fover.write_text(
        "\n"
        + json.dumps({"question_id": "skip", "label": "maybe", "step_text": "bad"})
        + "\n"
        + json.dumps({"question_id": "ok", "label": "correct", "step_text": "1 + 1 = 2"})
        + "\n",
        encoding="utf-8",
    )
    rows = mod._read_fover_rows(fover)
    assert [row["question_id"] for row in rows] == ["ok"]
    with pytest.raises(ValueError, match="unsupported"):
        mod._label_to_int("maybe")
    with pytest.raises(mod.ConditionScoringError, match="class balance"):
        mod._select_balanced_subset(rows, seed=42, n_examples=2)

    # Exercise corrupt session JSON, math_v3 id normalization, blank/bad JSONL,
    # unreadable JSONL, and empty prompt-token candidate handling.
    corrupt = tmp_path / "results" / "session_memory_999" / "bad" / "session_state.json"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_text("{not json", encoding="utf-8")
    math_state = {
        "case_memory": {
            "entries": [
                {
                    "key": {
                        "benchmark_slice": "fover:math_v3_bad_0",
                        "violation_families": ["fr11_v7_dvi_verified_incorrect"],
                        "prompt_sketch": "",
                    },
                    "prompt_tokens": [],
                    "violation_types": ["fr11_v7_dvi_verified_incorrect"],
                }
            ]
        }
    }
    _write_json(tmp_path / "results" / "session_memory_1000" / "x" / "session_state.json", math_state)
    noisy_jsonl = tmp_path / "data" / "fr11_noisy.jsonl"
    noisy_jsonl.write_text("\nnot json\n{\"question_id\": \"jsonl_bad\", \"is_correct\": false}\n", encoding="utf-8")
    unreadable = tmp_path / "data" / "fr11_unreadable.jsonl"
    unreadable.write_text("{}", encoding="utf-8")
    original_read_text = Path.read_text

    def patched_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        if self == unreadable:
            raise OSError("unreadable")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", patched_read_text)
    index = mod._load_fr11_memory_index(tmp_path)
    assert "bad_0" in index["question_ids"]
    assert "jsonl_bad" in index["question_ids"]
    assert mod._fr11_memory_score({"question_id": "bad_0", "step_text": ""}, index) == 1.0
    assert mod._fr11_memory_score(
        {"question_id": "none", "step_text": ""},
        {"question_ids": set(), "prompt_token_sets": [set()]},
    ) == 0.0


def test_req_verify_2837_defensive_scoring_and_subprocess_branches(tmp_path: Path) -> None:
    """REQ-VERIFY-2837: scoring rejects invalid states and subprocess JSON."""

    _minimal_repo(tmp_path)
    with pytest.raises(mod.ConditionScoringError, match="saw"):
        score_fover_subset(
            repo_root=tmp_path,
            seed=42,
            n_examples=4,
            condition=mod.CONDITION_ARCHITECTURE_ONLY,
            require_no_state=True,
        )
    with pytest.raises(mod.ConditionScoringError, match="unknown condition"):
        score_fover_subset(
            repo_root=tmp_path,
            seed=42,
            n_examples=4,
            condition="surprise",
            require_no_state=False,
        )
    with pytest.raises(ValueError, match="same length"):
        compute_auroc([0], [0.1, 0.2])

    def invalid_json_runner(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del kwargs
        return subprocess.CompletedProcess(args[0], 0, "not-json", "")

    with pytest.raises(mod.ConditionScoringError, match="invalid JSON"):
        mod.score_condition_via_subprocess(
            ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results", n_examples=4),
            sys.executable,
            42,
            mod.CONDITION_ARCHITECTURE_ONLY,
            False,
            command_runner=invalid_json_runner,
        )
    assert mod._ci95([0.5]) == {"mean": 0.5, "low": 0.5, "high": 0.5}


def test_req_verify_2837_cli_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-2837: CLI score and full-run paths delegate correctly."""

    _minimal_repo(tmp_path)
    assert (
        mod.main(
            [
                "--score-condition",
                "--repo-root",
                str(tmp_path),
                "--seed",
                "42",
                "--n-examples",
                "4",
                "--condition",
                mod.CONDITION_ARCHITECTURE_ONLY,
            ]
        )
        == 0
    )
    scored = json.loads(capsys.readouterr().out)
    assert scored["condition"] == mod.CONDITION_ARCHITECTURE_ONLY
    assert scored["n_examples"] == 4

    calls: list[ExperimentConfig] = []

    def fake_run_experiment(config: ExperimentConfig) -> dict[str, object]:
        calls.append(config)
        return {"honest_verdict": "complete: cli delegated"}

    monkeypatch.setattr(mod, "run_experiment", fake_run_experiment)
    assert (
        mod.main(
            [
                "--repo-root",
                str(tmp_path),
                "--results-dir",
                str(tmp_path / "custom-results"),
                "--n-examples",
                "4",
            ]
        )
        == 0
    )
    assert calls[0].repo_root == tmp_path
    assert calls[0].results_dir == tmp_path / "custom-results"
    assert calls[0].n_examples == 4
