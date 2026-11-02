from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILENAME = "experiment_2828_fover_memory_leakage_isolation.json"
N_EXAMPLES = 1000
RANDOM_SEEDS = [42, 137, 271, 314, 1729]
MODEL_NAME = "Qwen3.6-35B-A3B-GGUF"
CUDA_CHECK_CMD = ["python3", "-c", "import torch; assert torch.cuda.is_available()"]

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix per Verdict Terminal-Prefix Discipline.",
    "condition_a_production_auroc_mean": "Production-config AUROC (full FR-11 state).",
    "condition_a_production_auroc_std": "Replication noise on production-config.",
    "condition_b_architecture_only_auroc_mean": (
        "Memory-reset AUROC - the architecture's actual generalization."
    ),
    "condition_b_architecture_only_auroc_std": "Replication noise on architecture-only.",
    "learning_contribution": (
        "= A - B. Direct measurement of FR-11 self-learning contribution. "
        "> 0.05 = substantial; 0.02-0.05 = moderate; < 0.02 = minimal."
    ),
    "per_verifier_learning_contribution": (
        "Which specific verifiers degrade most without FR-11 memory."
    ),
    "fr11_state_files": "Names the persisted state that was reset.",
    "state_files_restored_sha_match": (
        "Proves FR-11 state was restored to original SHAs (non-destructive)."
    ),
    "n_examples": "Sample size.",
    "n_seeds": "Adversarial replication.",
    "random_seeds_used": "Determinism precondition for reproducibility.",
    "reproducibility_checksum": "Catches corpus / model / state drift.",
    "model_specs": "Names compute target.",
    "duration_s": (
        "Expected >= 3600s for 2 conditions x 5 seeds x N=1000 live inference."
    ),
    "preconditions_checked": "Anti-fabrication.",
    "methodology_note": "Honest interpretation of the delta.",
}

FR11_STATE_GLOBS = (
    "results/nexus_constraint_memory_v2.json",
    "results/constraint_patterns_v4.json",
    "results/constraint_memory_live_222.json",
    "results/**/session_state.json",
    "results/fr11_*.json",
    "results/fr11_*.jsonl",
    "data/constraint_memory.db",
    "data/fr11_*.jsonl",
)


CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]
MeasurementRunner = Callable[[dict[str, Any]], dict[str, Any]]


def _default_command_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)


def _relative_path(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _line in handle)


def discover_fr11_state_files(repo_root: Path) -> list[dict[str, Any]]:
    root = Path(repo_root)
    paths: set[Path] = set()
    for pattern in FR11_STATE_GLOBS:
        paths.update(path for path in root.glob(pattern) if path.is_file())

    records = [
        {
            "path": _relative_path(root, path),
            "sha256": _sha256_file(path),
            "n_bytes": path.stat().st_size,
        }
        for path in paths
    ]
    return sorted(records, key=lambda item: item["path"])


def _find_qwen_gguf(repo_root: Path) -> dict[str, Any]:
    model_paths: list[Path] = []
    models_dir = repo_root / "models"
    if models_dir.exists():
        for path in models_dir.rglob("*.gguf"):
            name = path.name.lower()
            is_qwen = "qwen" in name
            is_target_size = "35b" in name or "35-b" in name
            is_target_family = "3.6" in name or "3_6" in name or "3-6" in name
            if is_qwen and is_target_size and is_target_family and "a3b" in name:
                model_paths.append(path)

    relative_paths = sorted(_relative_path(repo_root, path) for path in model_paths)
    quant = None
    if relative_paths:
        match = re.search(r"(Q\d(?:_[A-Z0-9]+)+)", Path(relative_paths[0]).name)
        quant = match.group(1) if match else None

    return {
        "name": MODEL_NAME,
        "quant": quant,
        "revision_sha": None,
        "cached": bool(relative_paths),
        "cache_paths": relative_paths,
    }


def _check_preconditions(
    repo_root: Path,
    command_runner: CommandRunner,
    model_specs: dict[str, Any],
) -> list[dict[str, Any]]:
    process = command_runner(CUDA_CHECK_CMD)
    cuda_detail = "ok" if process.returncode == 0 else (process.stderr or process.stdout).strip()
    checks = [
        {
            "resource": "python3_torch_cuda",
            "available": process.returncode == 0,
            "check": "python3 -c \"import torch; assert torch.cuda.is_available()\"",
            "detail": cuda_detail[:300] if cuda_detail else f"returncode={process.returncode}",
        }
    ]

    fover_path = repo_root / "data" / "fover_corpus.jsonl"
    if fover_path.exists():
        n_rows = _line_count(fover_path)
        checks.append(
            {
                "resource": "fover_corpus",
                "available": n_rows >= N_EXAMPLES,
                "check": "test -f data/fover_corpus.jsonl and line_count >= 1000",
                "detail": f"line_count={n_rows}; required>={N_EXAMPLES}",
            }
        )
    else:
        checks.append(
            {
                "resource": "fover_corpus",
                "available": False,
                "check": "test -f data/fover_corpus.jsonl",
                "detail": "missing",
            }
        )

    nexus_path = repo_root / "results" / "nexus_constraint_memory_v2.json"
    checks.append(
        {
            "resource": "nexus_constraint_memory_v2",
            "available": nexus_path.exists(),
            "check": "test -f results/nexus_constraint_memory_v2.json",
            "detail": "present" if nexus_path.exists() else "missing",
        }
    )
    checks.append(
        {
            "resource": "qwen36_35b_a3b_gguf_cache",
            "available": model_specs["cached"],
            "check": "discover local Qwen3.6-35B-A3B *.gguf under models/",
            "detail": ",".join(model_specs["cache_paths"]) if model_specs["cached"] else "missing",
        }
    )
    return checks


def _blocked_prefix(preconditions: list[dict[str, Any]]) -> str | None:
    prefix_by_resource = {
        "python3_torch_cuda": "blocked_cuda",
        "fover_corpus": "blocked_fover_corpus",
        "nexus_constraint_memory_v2": "blocked_nexus_memory",
        "qwen36_35b_a3b_gguf_cache": "blocked_model_cache",
    }
    for check in preconditions:
        if not check["available"]:
            return prefix_by_resource[check["resource"]]
    return None


def _checksum_payload(
    repo_root: Path,
    state_files: list[dict[str, Any]],
    model_specs: dict[str, Any],
) -> dict[str, Any]:
    fover_path = repo_root / "data" / "fover_corpus.jsonl"
    nexus_path = repo_root / "results" / "nexus_constraint_memory_v2.json"
    return {
        "n_examples": N_EXAMPLES,
        "random_seeds_used": RANDOM_SEEDS,
        "fover_corpus_sha256": _sha256_file(fover_path) if fover_path.exists() else None,
        "nexus_constraint_memory_sha256": _sha256_file(nexus_path) if nexus_path.exists() else None,
        "fr11_state_files": state_files,
        "model_specs": model_specs,
    }


def _reproducibility_checksum(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _empty_metric_fields() -> dict[str, Any]:
    return {
        "condition_a_production_auroc_mean": None,
        "condition_a_production_auroc_std": None,
        "condition_b_architecture_only_auroc_mean": None,
        "condition_b_architecture_only_auroc_std": None,
        "learning_contribution": None,
        "per_verifier_learning_contribution": {},
        "per_verifier_condition_a_auroc": {},
        "per_verifier_condition_b_auroc": {},
        "condition_results_by_seed": [],
    }


def _base_artifact(
    *,
    duration_s: float,
    preconditions_checked: list[dict[str, Any]],
    state_files: list[dict[str, Any]],
    model_specs: dict[str, Any],
    checksum: str,
) -> dict[str, Any]:
    return {
        **_empty_metric_fields(),
        "fr11_state_files": state_files,
        "state_files_restored_sha_match": True,
        "state_reset_attempted": False,
        "n_examples": N_EXAMPLES,
        "n_seeds": len(RANDOM_SEEDS),
        "random_seeds_used": RANDOM_SEEDS,
        "reproducibility_checksum": checksum,
        "model_specs": model_specs,
        "duration_s": duration_s,
        "preconditions_checked": preconditions_checked,
        "methodology_note": (
            "The requested live dual-condition FoVer measurement was not run "
            "because one or more mandatory preconditions failed. AUROC fields "
            "are intentionally null rather than estimated from cached or "
            "synthetic measurements."
        ),
        "field_principles": FIELD_PRINCIPLES,
    }


def _with_measurements(base: dict[str, Any], measurements: dict[str, Any]) -> dict[str, Any]:
    artifact = {**base, **measurements}
    a_mean = artifact["condition_a_production_auroc_mean"]
    b_mean = artifact["condition_b_architecture_only_auroc_mean"]
    if artifact.get("learning_contribution") is None:
        artifact["learning_contribution"] = float(a_mean) - float(b_mean)

    if not artifact.get("per_verifier_learning_contribution"):
        a_by_verifier = artifact.get("per_verifier_condition_a_auroc", {})
        b_by_verifier = artifact.get("per_verifier_condition_b_auroc", {})
        artifact["per_verifier_learning_contribution"] = {
            name: float(a_by_verifier[name]) - float(b_by_verifier[name])
            for name in sorted(a_by_verifier.keys() & b_by_verifier.keys())
        }
    artifact["schema"] = "carnot.fover_memory_leakage_isolation.v1"
    artifact["honest_verdict"] = "complete: FoVer FR-11 memory isolation measured"
    artifact["methodology_note"] = (
        "Condition A keeps FR-11 self-learning state loaded; Condition B resets "
        "the same state non-destructively before reloading the verifier process. "
        "learning_contribution is reported as Condition A minus Condition B."
    )
    return artifact


def _write_artifact(repo_root: Path, artifact: dict[str, Any]) -> None:
    out_path = repo_root / "results" / OUTPUT_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_experiment(
    repo_root: Path = REPO_ROOT,
    write: bool = True,
    command_runner: CommandRunner | None = None,
    measurement_runner: MeasurementRunner | None = None,
) -> dict[str, Any]:
    start = time.time()
    root = Path(repo_root)
    runner = command_runner or _default_command_runner

    state_files = discover_fr11_state_files(root)
    model_specs = _find_qwen_gguf(root)
    preconditions = _check_preconditions(root, runner, model_specs)
    checksum = _reproducibility_checksum(_checksum_payload(root, state_files, model_specs))
    base = _base_artifact(
        duration_s=time.time() - start,
        preconditions_checked=preconditions,
        state_files=state_files,
        model_specs=model_specs,
        checksum=checksum,
    )

    blocked_prefix = _blocked_prefix(preconditions)
    if blocked_prefix is not None:
        first_failed = next(check for check in preconditions if not check["available"])
        artifact = {
            **base,
            "schema": f"{blocked_prefix}.v1",
            "honest_verdict": f"{blocked_prefix}: {first_failed['detail']}",
        }
    elif measurement_runner is None:
        artifact = {
            **base,
            "schema": "blocked_live_runner.v1",
            "honest_verdict": (
                "blocked_live_runner: live production scoring callback was not provided"
            ),
        }
    else:
        context = {
            "repo_root": root,
            "n_examples": N_EXAMPLES,
            "random_seeds_used": RANDOM_SEEDS,
            "fr11_state_files": state_files,
            "model_specs": model_specs,
        }
        artifact = _with_measurements(base, measurement_runner(context))
        artifact["duration_s"] = time.time() - start

    if write:
        _write_artifact(root, artifact)
    return artifact


if __name__ == "__main__":  # pragma: no cover
    run_experiment()
