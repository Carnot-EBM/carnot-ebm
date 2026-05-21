from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import experiment_2828_fover_memory_leakage_isolation as exp2828


SEEDS = [42, 137, 271, 314, 1729]


def _write_rows(path: Path, n_rows: int = 1000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        json.dumps({"id": idx, "question": f"q{idx}", "label": idx % 2})
        for idx in range(n_rows)
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _minimal_repo(root: Path) -> None:
    _write_rows(root / "data" / "fover_corpus.jsonl")
    nexus = root / "results" / "nexus_constraint_memory_v2.json"
    nexus.parent.mkdir(parents=True, exist_ok=True)
    nexus.write_text('{"rules": [{"pattern": "fover"}]}\n', encoding="utf-8")
    session = root / "results" / "session_memory_1471" / "fover" / "session_state.json"
    session.parent.mkdir(parents=True, exist_ok=True)
    session.write_text('{"template_keys": ["arithmetic"]}\n', encoding="utf-8")
    patterns = root / "results" / "constraint_patterns_v4.json"
    patterns.write_text('{"patterns": ["carry"]}\n', encoding="utf-8")


def _cuda_failure_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    assert cmd == ["python3", "-c", "import torch; assert torch.cuda.is_available()"]
    return subprocess.CompletedProcess(
        cmd,
        1,
        "",
        "ModuleNotFoundError: No module named 'torch'",
    )


def _cuda_success_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    assert cmd == ["python3", "-c", "import torch; assert torch.cuda.is_available()"]
    return subprocess.CompletedProcess(cmd, 0, "", "")


def test_req_verify_2828_blocks_without_cuda_and_keeps_required_schema(tmp_path: Path) -> None:
    """REQ-VERIFY-2828 / SCENARIO-VERIFY-2828: failed CUDA precondition blocks."""

    _minimal_repo(tmp_path)

    artifact = exp2828.run_experiment(
        repo_root=tmp_path,
        write=False,
        command_runner=_cuda_failure_runner,
    )

    required_fields = {
        "honest_verdict",
        "condition_a_production_auroc_mean",
        "condition_a_production_auroc_std",
        "condition_b_architecture_only_auroc_mean",
        "condition_b_architecture_only_auroc_std",
        "learning_contribution",
        "per_verifier_learning_contribution",
        "fr11_state_files",
        "state_files_restored_sha_match",
        "n_examples",
        "n_seeds",
        "random_seeds_used",
        "reproducibility_checksum",
        "model_specs",
        "duration_s",
        "preconditions_checked",
        "methodology_note",
    }
    assert required_fields <= set(artifact)
    assert artifact["schema"].startswith("blocked")
    assert artifact["honest_verdict"].startswith("blocked_cuda:")
    assert artifact["condition_a_production_auroc_mean"] is None
    assert artifact["condition_b_architecture_only_auroc_mean"] is None
    assert artifact["learning_contribution"] is None
    assert artifact["per_verifier_learning_contribution"] == {}
    assert artifact["state_files_restored_sha_match"] is True
    assert artifact["n_examples"] == 1000
    assert artifact["n_seeds"] == 5
    assert artifact["random_seeds_used"] == SEEDS
    assert artifact["state_reset_attempted"] is False

    checks = {entry["resource"]: entry for entry in artifact["preconditions_checked"]}
    assert checks["python3_torch_cuda"]["available"] is False
    assert checks["fover_corpus"]["available"] is True
    assert checks["nexus_constraint_memory_v2"]["available"] is True
    assert checks["qwen36_35b_a3b_gguf_cache"]["available"] is False


def test_req_verify_2828_state_hashes_and_checksum_are_stable(tmp_path: Path) -> None:
    """REQ-VERIFY-2828: FR-11 state files are named with SHA256 and byte size."""

    _minimal_repo(tmp_path)
    model_path = tmp_path / "models" / "gguf" / "Qwen3.6-35B-A3B-Q4_K_M.gguf"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"tiny test model sentinel")

    first = exp2828.run_experiment(
        repo_root=tmp_path,
        write=False,
        command_runner=_cuda_success_runner,
    )
    second = exp2828.run_experiment(
        repo_root=tmp_path,
        write=False,
        command_runner=_cuda_success_runner,
    )

    assert first["honest_verdict"].startswith("blocked_live_runner:")
    assert first["reproducibility_checksum"] == second["reproducibility_checksum"]
    assert first["model_specs"] == {
        "name": "Qwen3.6-35B-A3B-GGUF",
        "quant": "Q4_K_M",
        "revision_sha": None,
        "cached": True,
        "cache_paths": ["models/gguf/Qwen3.6-35B-A3B-Q4_K_M.gguf"],
    }

    state_files = first["fr11_state_files"]
    assert [entry["path"] for entry in state_files] == sorted(entry["path"] for entry in state_files)
    assert "results/nexus_constraint_memory_v2.json" in {entry["path"] for entry in state_files}
    assert "results/session_memory_1471/fover/session_state.json" in {
        entry["path"] for entry in state_files
    }
    assert all(len(entry["sha256"]) == 64 for entry in state_files)
    assert all(entry["n_bytes"] > 0 for entry in state_files)


def test_scenario_verify_2828_writes_blocked_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-2828: blocked run still writes a stable JSON artifact."""

    _minimal_repo(tmp_path)
    artifact = exp2828.run_experiment(
        repo_root=tmp_path,
        write=True,
        command_runner=_cuda_failure_runner,
    )

    artifact_path = tmp_path / "results" / exp2828.OUTPUT_FILENAME
    assert artifact_path.exists()
    on_disk = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert on_disk == artifact
    assert on_disk["condition_results_by_seed"] == []
    assert on_disk["per_verifier_condition_a_auroc"] == {}
    assert on_disk["per_verifier_condition_b_auroc"] == {}


def test_req_verify_2828_rejects_too_small_fover_corpus(tmp_path: Path) -> None:
    """REQ-VERIFY-2828: n=1000 FoVer sampling cannot run on undersized data."""

    _minimal_repo(tmp_path)
    _write_rows(tmp_path / "data" / "fover_corpus.jsonl", n_rows=999)

    artifact = exp2828.run_experiment(
        repo_root=tmp_path,
        write=False,
        command_runner=_cuda_success_runner,
    )

    checks = {entry["resource"]: entry for entry in artifact["preconditions_checked"]}
    assert artifact["honest_verdict"].startswith("blocked_fover_corpus:")
    assert checks["fover_corpus"]["available"] is False
    assert "1000" in checks["fover_corpus"]["detail"]
    assert artifact["condition_a_production_auroc_std"] is None


def test_req_verify_2828_measurement_callback_computes_learning_delta(tmp_path: Path) -> None:
    """REQ-VERIFY-2828: successful live callback output is reconciled consistently."""

    _minimal_repo(tmp_path)
    model_path = tmp_path / "models" / "gguf" / "Qwen3.6-35B-A3B-Q4_K_M.gguf"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"tiny test model sentinel")

    def measurement_runner(context: dict) -> dict:
        assert context["n_examples"] == 1000
        assert context["random_seeds_used"] == SEEDS
        return {
            "condition_a_production_auroc_mean": 0.91,
            "condition_a_production_auroc_std": 0.02,
            "condition_b_architecture_only_auroc_mean": 0.84,
            "condition_b_architecture_only_auroc_std": 0.03,
            "per_verifier_condition_a_auroc": {"tier0r": 0.9, "nexus": 0.88},
            "per_verifier_condition_b_auroc": {"tier0r": 0.86, "nexus": 0.78},
            "condition_results_by_seed": [{"seed": 42, "condition_a_auroc": 0.91}],
        }

    artifact = exp2828.run_experiment(
        repo_root=tmp_path,
        write=False,
        command_runner=_cuda_success_runner,
        measurement_runner=measurement_runner,
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["schema"] == "carnot.fover_memory_leakage_isolation.v1"
    assert artifact["learning_contribution"] == pytest.approx(0.07)
    assert artifact["per_verifier_learning_contribution"] == pytest.approx(
        {"nexus": 0.1, "tier0r": 0.04}
    )
    assert "Condition A keeps FR-11" in artifact["methodology_note"]


def test_req_verify_2828_missing_fover_file_and_default_runner(tmp_path: Path) -> None:
    """REQ-VERIFY-2828: missing FoVer data blocks before model scoring."""

    process = exp2828._default_command_runner(["python3", "--version"])
    assert process.returncode == 0

    nexus = tmp_path / "results" / "nexus_constraint_memory_v2.json"
    nexus.parent.mkdir(parents=True, exist_ok=True)
    nexus.write_text("{}\n", encoding="utf-8")

    model_path = tmp_path / "models" / "gguf" / "Qwen3.6-35B-A3B-Q4_K_M.gguf"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"tiny test model sentinel")

    artifact = exp2828.run_experiment(
        repo_root=tmp_path,
        write=False,
        command_runner=_cuda_success_runner,
    )

    checks = {entry["resource"]: entry for entry in artifact["preconditions_checked"]}
    assert artifact["honest_verdict"].startswith("blocked_fover_corpus:")
    assert checks["fover_corpus"]["detail"] == "missing"
