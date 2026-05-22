"""Tests for Exp 2836 FoVer FR-11 memory isolation.

Spec: REQ-VERIFY-2836,
      SCENARIO-VERIFY-2836,
      SCENARIO-VERIFY-2836-LIVE.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import fover_memory_leakage_isolation_2836 as mod
from carnot.eval.fover_memory_leakage_isolation_2836 import (
    ConditionMeasurement,
    ExperimentConfig,
    discover_fr11_state_files,
    run_experiment,
    state_files_restored_sha_match,
    temporarily_move_state_files,
)


SEEDS = (42, 137, 271, 314, 1729)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_fover_rows(path: Path, n_rows: int = 1000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(n_rows):
        label = "incorrect" if idx % 2 else "correct"
        rows.append(
            json.dumps(
                {
                    "question_id": f"q{idx}",
                    "label": label,
                    "step_text": f"FoVer row {idx} is {label}.",
                }
            )
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_state(root: Path) -> None:
    _write_json(root / "results" / "nexus_constraint_memory_v2.json", {"rules": ["nexus"]})
    _write_json(root / "data" / "constraint_templates_713.json", {"templates": ["carry"]})
    _write_json(root / "results" / "constraint_patterns_v4.json", {"patterns": ["borrow"]})
    _write_json(
        root / "results" / "session_memory_1471" / "fover" / "session_state.json",
        {"case_memory": {"entries": [{"question_id": "q1"}]}},
    )
    (root / "data" / "fr11_zenil_distill_v2.jsonl").write_text(
        json.dumps({"question_id": "q3", "is_correct": False}) + "\n",
        encoding="utf-8",
    )


def _write_qwen_model(root: Path) -> Path:
    model = root / "models" / "qwen" / "Qwen3.6-35B-A3B-Q4_K_M.gguf"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"test qwen gguf sentinel")
    return model


def _minimal_repo(root: Path, *, qwen: bool = True, n_rows: int = 1000) -> None:
    _write_fover_rows(root / "data" / "fover_corpus.jsonl", n_rows=n_rows)
    _write_state(root)
    if qwen:
        _write_qwen_model(root)


def _cuda_success_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    assert cmd[0].endswith(".venv/bin/python3")
    assert cmd[2] == mod.CUDA_ASSERT_CODE
    return subprocess.CompletedProcess(cmd, 0, "", "")


def test_scenario_verify_2836_missing_qwen_cache_blocks_without_metrics(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-2836: a .no_exist marker is not a cached Qwen GGUF."""

    _minimal_repo(tmp_path, qwen=False)
    marker = (
        tmp_path
        / "models"
        / ".no_exist"
        / "a483e9e6cbd595906af30beda3187c2663a1118c"
        / "Qwen3.6-35B-A3B-Q4_K_M.gguf"
    )
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_bytes(b"not a cache hit")

    artifact = run_experiment(
        ExperimentConfig(
            repo_root=tmp_path,
            results_dir=tmp_path / "results",
            model_search_roots=(tmp_path / "models",),
            started_at=10.0,
            clock=lambda: 12.5,
        ),
        command_runner=_cuda_success_runner,
        write=True,
    )

    assert artifact["honest_verdict"].startswith("blocked_model_cache:")
    assert artifact["condition_a_production_auroc_mean"] is None
    assert artifact["condition_b_architecture_only_auroc_mean"] is None
    assert artifact["learning_contribution"] is None
    assert artifact["per_verifier_learning_contribution"] == {}
    assert artifact["n_examples"] == 1000
    assert artifact["n_seeds"] == 5
    assert artifact["random_seeds_used"] == list(SEEDS)
    assert artifact["model_specs"]["name"] == "Qwen3.6-35B-A3B-GGUF"
    assert artifact["model_specs"]["cached"] is False
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["state_files_restored_sha_match"] is True
    assert artifact["state_reset_attempted"] is False

    checks = {entry["resource"]: entry for entry in artifact["preconditions_checked"]}
    assert checks["venv_torch_cuda"]["available"] is True
    assert ".venv/bin/python3" in " ".join(checks["venv_torch_cuda"]["command"])
    assert checks["qwen36_35b_a3b_gguf_cache"]["available"] is False
    on_disk = json.loads((tmp_path / "results" / mod.OUTPUT_FILENAME).read_text())
    assert on_disk == artifact


def test_req_verify_2836_cuda_regression_uses_venv_and_records_torch_versions(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-2836: CUDA failure uses blocked_cuda_post_fix_regression."""

    _minimal_repo(tmp_path)
    commands: list[list[str]] = []

    def failing_cuda_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(cmd)
        assert cmd[0].endswith(".venv/bin/python3")
        if cmd[2] == mod.CUDA_ASSERT_CODE:
            return subprocess.CompletedProcess(cmd, 1, "", "AssertionError: cuda false")
        assert cmd[2] == mod.CUDA_DIAGNOSTIC_CODE
        return subprocess.CompletedProcess(
            cmd,
            0,
            json.dumps(
                {
                    "torch_version": "2.11.0+cu128",
                    "torch_cuda": "12.8",
                    "cuda_available": False,
                    "device_count": 0,
                },
                sort_keys=True,
            )
            + "\n",
            "",
        )

    artifact = run_experiment(
        ExperimentConfig(repo_root=tmp_path, results_dir=tmp_path / "results"),
        command_runner=failing_cuda_runner,
        write=False,
    )

    assert artifact["honest_verdict"].startswith("blocked_cuda_post_fix_regression:")
    assert artifact["torch_version_output"]["torch_version"] == "2.11.0+cu128"
    assert artifact["condition_a_production_auroc_std"] is None
    assert len(commands) == 2
    assert all(command[0].endswith(".venv/bin/python3") for command in commands)


def test_req_verify_2836_state_manifest_move_restore_and_checksum(tmp_path: Path) -> None:
    """REQ-VERIFY-2836: all FR-11/NEXUS/session state files are restored by SHA."""

    _minimal_repo(tmp_path)
    state_files = discover_fr11_state_files(tmp_path)
    paths = [entry["path"] for entry in state_files]

    assert paths == sorted(paths)
    assert "data/constraint_templates_713.json" in paths
    assert "results/nexus_constraint_memory_v2.json" in paths
    assert "results/session_memory_1471/fover/session_state.json" in paths
    assert all(len(entry["sha256"]) == 64 for entry in state_files)
    assert all(entry["n_bytes"] > 0 for entry in state_files)

    with temporarily_move_state_files(tmp_path, state_files, tmp_path / "backup"):
        assert discover_fr11_state_files(tmp_path) == []
        for entry in state_files:
            assert not (tmp_path / entry["path"]).exists()

    assert state_files_restored_sha_match(tmp_path, state_files) is True
    assert discover_fr11_state_files(tmp_path) == state_files

    (tmp_path / state_files[0]["path"]).write_text("changed\n", encoding="utf-8")
    assert state_files_restored_sha_match(tmp_path, state_files) is False


def test_scenario_verify_2836_live_scorer_protocol_summarizes_dual_condition(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-2836-LIVE: fake scorer confirms the state-reset protocol."""

    _minimal_repo(tmp_path)
    calls: list[tuple[int, str, int, bool]] = []

    def scorer(
        config: ExperimentConfig,
        seed: int,
        condition: str,
        require_no_state: bool,
    ) -> ConditionMeasurement:
        visible = len(discover_fr11_state_files(config.repo_root))
        calls.append((seed, condition, visible, require_no_state))
        if require_no_state:
            assert visible == 0
        offset = 0.02 if seed == 137 else 0.0
        is_production = condition == mod.CONDITION_PRODUCTION
        return ConditionMeasurement(
            seed=seed,
            condition=condition,
            auroc=(0.92 + offset) if is_production else (0.81 + offset),
            per_verifier_auroc={
                "nexus_constraint_memory": (0.90 + offset) if is_production else (0.72 + offset),
                "tier0r_curry_howard": (0.86 + offset) if is_production else (0.84 + offset),
            },
            n_examples=config.n_examples,
            state_visible_count=visible,
            fr11_state_loaded=is_production and visible > 0,
            subset_sha256=f"subset-{seed}",
            python_executable="live-gpu-python",
        )

    artifact = run_experiment(
        ExperimentConfig(
            repo_root=tmp_path,
            results_dir=tmp_path / "results",
            n_examples=4,
            random_seeds=(42, 137),
            backup_parent=tmp_path / "tmp",
            started_at=1.0,
            clock=lambda: 11.0,
        ),
        command_runner=_cuda_success_runner,
        condition_scorer=scorer,
        write=False,
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["condition_a_production_auroc_mean"] == pytest.approx(0.93)
    assert artifact["condition_a_production_auroc_std"] == pytest.approx(0.01)
    assert artifact["condition_b_architecture_only_auroc_mean"] == pytest.approx(0.82)
    assert artifact["learning_contribution"] == pytest.approx(0.11)
    assert artifact["per_verifier_learning_contribution"] == pytest.approx(
        {"nexus_constraint_memory": 0.18, "tier0r_curry_howard": 0.02}
    )
    assert artifact["state_reset_attempted"] is True
    assert artifact["state_files_restored_sha_match"] is True
    assert all(row["condition_b_state_visible_count"] == 0 for row in artifact["per_seed_results"])
    assert calls == [
        (42, mod.CONDITION_PRODUCTION, 5, False),
        (42, mod.CONDITION_ARCHITECTURE_ONLY, 0, True),
        (137, mod.CONDITION_PRODUCTION, 5, False),
        (137, mod.CONDITION_ARCHITECTURE_ONLY, 0, True),
    ]


def test_req_verify_2836_live_runner_absent_blocks_after_preconditions(tmp_path: Path) -> None:
    """REQ-VERIFY-2836: no production ensemble backend means no AUROC fields."""

    _minimal_repo(tmp_path)
    artifact = run_experiment(
        ExperimentConfig(
            repo_root=tmp_path,
            results_dir=tmp_path / "results",
            started_at=3.0,
            clock=lambda: 4.0,
        ),
        command_runner=_cuda_success_runner,
        write=False,
    )

    checks = {entry["resource"]: entry for entry in artifact["preconditions_checked"]}
    assert artifact["honest_verdict"].startswith("blocked_live_verifier_runner:")
    assert checks["production_verifier_ensemble_v7b_runner"]["available"] is False
    assert artifact["condition_results_by_seed"] == []
    assert artifact["per_seed_results"] == []
    assert "No AUROC values were inferred" in artifact["methodology_note"]


def test_req_verify_2836_defensive_branches_and_model_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-2836: defensive branches preserve honest metadata."""

    measurement = ConditionMeasurement(
        seed=42,
        condition=mod.CONDITION_PRODUCTION,
        auroc=0.5,
        per_verifier_auroc={"v": 0.5},
        n_examples=1,
        state_visible_count=0,
        fr11_state_loaded=False,
        subset_sha256="subset",
        python_executable="python",
    )
    assert measurement.as_dict()["per_verifier_auroc"] == {"v": 0.5}

    proc = mod._default_command_runner(
        [str(Path.cwd() / ".venv" / "bin" / "python3"), "-c", "print('ok')"]
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == "ok"

    hf_root = tmp_path / "hf"
    snapshot = (
        hf_root
        / "hub"
        / "models--unsloth--Qwen3.6-35B-A3B-GGUF"
        / "snapshots"
        / "1234567890abcdef"
        / "Qwen3.6-35B-A3B-Q4_K_M.gguf"
    )
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(b"real qwen sentinel")
    monkeypatch.setenv("HF_HOME", str(hf_root))

    specs = mod.find_qwen36_gguf(ExperimentConfig(repo_root=tmp_path))
    assert specs["selected_path"] == str(snapshot)
    assert specs["quant"] == "Q4_K_M"
    assert specs["revision_sha"] == "1234567890abcdef"
    assert mod.find_qwen36_gguf(
        ExperimentConfig(repo_root=tmp_path, model_search_roots=(tmp_path / "missing",))
    )["cached"] is False

    assert mod._parse_json_or_raw(subprocess.CompletedProcess(["x"], 7, "", "")) == {
        "returncode": 7
    }
    assert mod._parse_json_or_raw(subprocess.CompletedProcess(["x"], 0, "not-json", "")) == {
        "output": "not-json",
        "returncode": 0,
    }

    _minimal_repo(tmp_path)
    state_files = discover_fr11_state_files(tmp_path)
    missing = dict(state_files[0])
    (tmp_path / missing["path"]).unlink()
    with pytest.raises(RuntimeError, match="disappeared"):
        with temporarily_move_state_files(tmp_path, [missing], tmp_path / "backup-missing"):
            pass

    _minimal_repo(tmp_path)
    state_files = discover_fr11_state_files(tmp_path)
    backup = tmp_path / "backup-existing"
    target = backup / state_files[0]["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("already here\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="backup path already exists"):
        with temporarily_move_state_files(tmp_path, [state_files[0]], backup):
            pass

    no_fover = tmp_path / "no-fover"
    _write_state(no_fover)
    _write_qwen_model(no_fover)
    artifact = run_experiment(
        ExperimentConfig(repo_root=no_fover, results_dir=no_fover / "results"),
        command_runner=_cuda_success_runner,
        write=False,
    )
    assert artifact["honest_verdict"].startswith("blocked_fover_corpus:")
