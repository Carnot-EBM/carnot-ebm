"""Exp 2836 FoVer FR-11 memory-leakage isolation runner.

This module is deliberately strict about preconditions because the requested
measurement is expensive and easy to misreport. It records the FR-11 state
manifest and can run the dual-condition reset protocol when a real production
ensemble-v7b scorer is supplied. Without that scorer, or without the mandated
Qwen3.6 GGUF cache, it writes a blocked artifact with null AUROC fields.

Spec: REQ-VERIFY-2836,
      SCENARIO-VERIFY-2836,
      SCENARIO-VERIFY-2836-LIVE.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_FILENAME = "experiment_2836_fover_memory_leakage_isolation.json"
MODEL_NAME = "Qwen3.6-35B-A3B-GGUF"
N_EXAMPLES = 1000
RANDOM_SEEDS = (42, 137, 271, 314, 1729)
CONDITION_PRODUCTION = "production"
CONDITION_ARCHITECTURE_ONLY = "architecture_only"
CUDA_ASSERT_CODE = (
    "import torch; assert torch.cuda.is_available() and torch.cuda.device_count() >= 1"
)
CUDA_DIAGNOSTIC_CODE = (
    "import json, torch; "
    "print(json.dumps({"
    "'torch_version': torch.__version__, "
    "'torch_cuda': torch.version.cuda, "
    "'cuda_available': torch.cuda.is_available(), "
    "'device_count': torch.cuda.device_count()"
    "}, sort_keys=True))"
)

FR11_STATE_GLOBS = (
    "data/constraint_memory.db",
    "data/constraint_templates*.json",
    "data/fr11_*.json",
    "data/fr11_*.jsonl",
    "results/constraint_memory*.json",
    "results/constraint_patterns*.json",
    "results/constraint_templates*.json",
    "results/fr11_*.json",
    "results/fr11_*.jsonl",
    "results/nexus_constraint_memory*.json",
    "results/session_memory_*/**/session_state.json",
    "results/exp_448_session_memory/**/session_state.json",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix per Verdict Terminal-Prefix Discipline.",
    "condition_a_production_auroc_mean": "Production-config AUROC (full FR-11 state).",
    "condition_a_production_auroc_std": "Replication noise on production-config.",
    "condition_b_architecture_only_auroc_mean": (
        "Memory-reset AUROC - the architecture's actual generalization."
    ),
    "condition_b_architecture_only_auroc_std": "Replication noise on architecture-only.",
    "learning_contribution": "= A - B. Direct measurement of FR-11 self-learning contribution.",
    "per_verifier_learning_contribution": (
        "Which specific verifiers degrade most without FR-11 memory."
    ),
    "fr11_state_files": "Names the persisted state that was reset.",
    "state_files_restored_sha_match": (
        "Proves FR-11 state was restored to original SHAs (non-destructive)."
    ),
    "n_examples": "Sample size.",
    "n_seeds": "Adversarial replication.",
    "random_seeds_used": "Determinism.",
    "reproducibility_checksum": "Catches drift.",
    "model_specs": "Names compute target.",
    "duration_s": "Real wall time; no padding.",
    "preconditions_checked": ("Anti-fabrication. The CUDA check uses .venv/bin/python3."),
    "methodology_note": "Honest interpretation of the delta.",
}


@dataclass(frozen=True)
class PreconditionCheck:
    """One resource gate checked before live measurement."""

    resource: str
    available: bool
    detail: str
    command: list[str] | None = None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "resource": self.resource,
            "available": self.available,
            "detail": self.detail,
        }
        if self.command is not None:
            payload["command"] = list(self.command)
        return payload


@dataclass(frozen=True)
class ConditionMeasurement:
    """A real scorer's output for one seed and one memory condition."""

    seed: int
    condition: str
    auroc: float
    per_verifier_auroc: dict[str, float]
    n_examples: int
    state_visible_count: int
    fr11_state_loaded: bool
    subset_sha256: str
    python_executable: str

    def as_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "condition": self.condition,
            "auroc": self.auroc,
            "per_verifier_auroc": dict(self.per_verifier_auroc),
            "n_examples": self.n_examples,
            "state_visible_count": self.state_visible_count,
            "fr11_state_loaded": self.fr11_state_loaded,
            "subset_sha256": self.subset_sha256,
            "python_executable": self.python_executable,
        }


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for the Exp 2836 artifact writer."""

    repo_root: Path = REPO_ROOT
    results_dir: Path | None = None
    n_examples: int = N_EXAMPLES
    random_seeds: tuple[int, ...] = RANDOM_SEEDS
    model_search_roots: tuple[Path, ...] = ()
    backup_parent: Path = Path("/tmp")
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def output_dir(self) -> Path:
        return self.results_dir if self.results_dir is not None else self.repo_root / "results"

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def qwen_search_roots(self) -> tuple[Path, ...]:
        if self.model_search_roots:
            return tuple(Path(path) for path in self.model_search_roots)
        roots = [self.repo_root / "models"]
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            roots.append(Path(hf_home) / "hub")
        roots.append(Path.home() / ".cache" / "huggingface" / "hub")
        return tuple(dict.fromkeys(roots))


CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]
ConditionScorer = Callable[[ExperimentConfig, int, str, bool], ConditionMeasurement]


def _default_command_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)


def _display_venv_command(code: str) -> list[str]:
    return [".venv/bin/python3", "-c", code]


def _venv_command(repo_root: Path, code: str) -> list[str]:
    return [str(Path(repo_root) / ".venv" / "bin" / "python3"), "-c", code]


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


def discover_fr11_state_files(repo_root: Path) -> list[dict[str, object]]:
    """Return the FR-11/NEXUS/template/session state manifest."""

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
    return sorted(records, key=lambda item: str(item["path"]))


def state_files_restored_sha_match(
    repo_root: Path,
    state_files: Sequence[dict[str, object]],
) -> bool:
    """Verify that every state file still exists with the recorded SHA."""

    root = Path(repo_root)
    for item in state_files:
        path = root / str(item["path"])
        if not path.is_file() or _sha256_file(path) != str(item["sha256"]):
            return False
    return True


@contextmanager
def temporarily_move_state_files(
    repo_root: Path,
    state_files: Sequence[dict[str, object]],
    backup_root: Path,
) -> Iterator[None]:
    """Move state files aside and restore them even if scoring fails."""

    root = Path(repo_root)
    backup = Path(backup_root)
    moved: list[tuple[Path, Path]] = []
    try:
        for item in state_files:
            rel = Path(str(item["path"]))
            source = root / rel
            target = backup / rel
            if not source.is_file():
                raise RuntimeError(f"state file disappeared before reset: {rel.as_posix()}")
            if target.exists():
                raise RuntimeError(f"backup path already exists: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            moved.append((source, target))
        yield
    finally:
        for source, target in reversed(moved):
            if target.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(target), str(source))
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)


_QUANT_RE = re.compile(r"(UD-)?Q\d(?:_[A-Z0-9]+)+|IQ\d(?:_[A-Z0-9]+)+|BF16|F16")


def _is_qwen36_a3b_gguf(path: Path) -> bool:
    name = path.name.lower()
    return (
        path.suffix.lower() == ".gguf"
        and "qwen" in name
        and ("3.6" in name or "3_6" in name or "3-6" in name)
        and ("35b" in name or "35-b" in name)
        and "a3b" in name
    )


def _revision_from_path(path: Path) -> str | None:
    parts = path.parts
    if "snapshots" not in parts:
        return None
    idx = parts.index("snapshots")
    return parts[idx + 1] if idx + 1 < len(parts) else None


def _quant_from_path(path: Path) -> str | None:
    match = _QUANT_RE.search(path.name)
    return match.group(0) if match else None


def find_qwen36_gguf(config: ExperimentConfig) -> dict[str, object]:
    """Find a real cached Qwen3.6-35B-A3B GGUF, excluding cache miss sentinels."""

    candidates: list[Path] = []
    no_exist_markers: list[str] = []
    selected: Path | None = None
    for root in config.qwen_search_roots():
        if not root.exists():
            continue
        root_candidates: list[Path] = []
        for path in root.rglob("*.gguf"):
            if not _is_qwen36_a3b_gguf(path):
                continue
            if ".no_exist" in path.parts:
                no_exist_markers.append(str(path))
                continue
            if path.is_file() and path.stat().st_size > 0:
                root_candidates.append(path)
        root_candidates = sorted(root_candidates, key=lambda item: str(item))
        if selected is None and root_candidates:
            selected = root_candidates[0]
        candidates.extend(root_candidates)

    return {
        "name": MODEL_NAME,
        "quant": _quant_from_path(selected) if selected is not None else None,
        "revision_sha": _revision_from_path(selected) if selected is not None else None,
        "cached": selected is not None,
        "cache_paths": [str(path) for path in sorted(candidates, key=lambda item: str(item))],
        "selected_path": str(selected) if selected is not None else None,
        "ignored_no_exist_markers": sorted(no_exist_markers),
    }


def _parse_json_or_raw(process: subprocess.CompletedProcess[str]) -> object:
    text = (process.stdout or process.stderr or "").strip()
    if not text:
        return {"returncode": process.returncode}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"output": text, "returncode": process.returncode}


def _probe_cuda(
    repo_root: Path,
    command_runner: CommandRunner,
) -> tuple[PreconditionCheck, object | None]:
    command = _venv_command(repo_root, CUDA_ASSERT_CODE)
    process = command_runner(command)
    if process.returncode == 0:
        return (
            PreconditionCheck(
                "venv_torch_cuda",
                True,
                "ok",
                _display_venv_command(CUDA_ASSERT_CODE),
            ),
            None,
        )

    diagnostic = command_runner(_venv_command(repo_root, CUDA_DIAGNOSTIC_CODE))
    torch_output = _parse_json_or_raw(diagnostic)
    detail = (process.stderr or process.stdout or f"returncode={process.returncode}").strip()
    return (
        PreconditionCheck(
            "venv_torch_cuda",
            False,
            detail,
            _display_venv_command(CUDA_ASSERT_CODE),
        ),
        torch_output,
    )


def _probe_preconditions(
    config: ExperimentConfig,
    command_runner: CommandRunner,
    model_specs: dict[str, object],
    state_files: Sequence[dict[str, object]],
    condition_scorer: ConditionScorer | None,
) -> tuple[list[PreconditionCheck], object | None]:
    cuda_check, torch_output = _probe_cuda(config.repo_root, command_runner)
    checks = [cuda_check]

    fover_path = config.repo_root / "data" / "fover_corpus.jsonl"
    if fover_path.exists():
        n_rows = _line_count(fover_path)
        checks.append(
            PreconditionCheck(
                "fover_corpus",
                n_rows >= config.n_examples,
                f"line_count={n_rows}; required>={config.n_examples}",
                ["test", "-f", "data/fover_corpus.jsonl"],
            )
        )
    else:
        checks.append(
            PreconditionCheck(
                "fover_corpus",
                False,
                "missing",
                ["test", "-f", "data/fover_corpus.jsonl"],
            )
        )

    nexus_path = config.repo_root / "results" / "nexus_constraint_memory_v2.json"
    checks.append(
        PreconditionCheck(
            "nexus_constraint_memory_v2",
            nexus_path.is_file(),
            "present" if nexus_path.is_file() else "missing",
            ["test", "-f", "results/nexus_constraint_memory_v2.json"],
        )
    )
    checks.append(
        PreconditionCheck(
            "qwen36_35b_a3b_gguf_cache",
            bool(model_specs.get("cached")),
            str(model_specs.get("selected_path") or "missing"),
            ["find", "models ~/.cache/huggingface", "-name", "*Qwen3.6-35B-A3B*.gguf"],
        )
    )
    checks.append(
        PreconditionCheck("fr11_state_files", bool(state_files), f"count={len(state_files)}")
    )
    checks.append(
        PreconditionCheck(
            "production_verifier_ensemble_v7b_runner",
            condition_scorer is not None,
            "provided" if condition_scorer is not None else "missing",
        )
    )
    return checks, torch_output


def _blocked_prefix(checks: Sequence[PreconditionCheck]) -> str | None:
    prefix_by_resource = {
        "venv_torch_cuda": "blocked_cuda_post_fix_regression",
        "fover_corpus": "blocked_fover_corpus",
        "nexus_constraint_memory_v2": "blocked_nexus_memory",
        "qwen36_35b_a3b_gguf_cache": "blocked_model_cache",
        "fr11_state_files": "blocked_fr11_state_files",
        "production_verifier_ensemble_v7b_runner": "blocked_live_verifier_runner",
    }
    for check in checks:
        if not check.available:
            return prefix_by_resource[check.resource]
    return None


def _file_sha_or_none(path: Path) -> str | None:
    return _sha256_file(path) if path.is_file() else None


def _reproducibility_checksum(
    config: ExperimentConfig,
    state_files: Sequence[dict[str, object]],
    model_specs: dict[str, object],
) -> str:
    payload = {
        "n_examples": config.n_examples,
        "random_seeds_used": list(config.random_seeds),
        "fover_corpus_sha256": _file_sha_or_none(config.repo_root / "data" / "fover_corpus.jsonl"),
        "nexus_constraint_memory_sha256": _file_sha_or_none(
            config.repo_root / "results" / "nexus_constraint_memory_v2.json"
        ),
        "fr11_state_files": list(state_files),
        "model_specs": model_specs,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _empty_metric_fields() -> dict[str, object]:
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
        "per_seed_results": [],
    }


def _base_artifact(
    *,
    config: ExperimentConfig,
    duration_s: float,
    state_files: Sequence[dict[str, object]],
    checks: Sequence[PreconditionCheck],
    model_specs: dict[str, object],
    torch_output: object | None,
) -> dict[str, object]:
    return {
        **_empty_metric_fields(),
        "artifact": "experiment_2836_fover_memory_leakage_isolation",
        "fr11_state_files": list(state_files),
        "state_files_restored_sha_match": state_files_restored_sha_match(
            config.repo_root, state_files
        ),
        "state_reset_attempted": False,
        "n_examples": config.n_examples,
        "n_seeds": len(config.random_seeds),
        "random_seeds_used": list(config.random_seeds),
        "reproducibility_checksum": _reproducibility_checksum(config, state_files, model_specs),
        "model_specs": model_specs,
        "duration_s": duration_s,
        "preconditions_checked": [check.as_dict() for check in checks],
        "torch_version_output": torch_output,
        "field_principles": FIELD_PRINCIPLES,
    }


def _blocked_artifact(
    *,
    config: ExperimentConfig,
    duration_s: float,
    state_files: Sequence[dict[str, object]],
    checks: Sequence[PreconditionCheck],
    model_specs: dict[str, object],
    torch_output: object | None,
) -> dict[str, object]:
    prefix = _blocked_prefix(checks) or "blocked_unknown_resource"
    failed = [check for check in checks if not check.available]
    first = failed[0] if failed else PreconditionCheck("unknown", False, "unknown")
    artifact = _base_artifact(
        config=config,
        duration_s=duration_s,
        state_files=state_files,
        checks=checks,
        model_specs=model_specs,
        torch_output=torch_output,
    )
    artifact.update(
        {
            "schema": f"{prefix}.v1",
            "honest_verdict": f"{prefix}: {first.detail}",
            "blocked_resources": [check.resource for check in failed],
            "methodology_note": (
                "Blocked before FoVer dual-condition scoring because required "
                "resources were unavailable: "
                + ", ".join(check.resource for check in failed)
                + ". No AUROC values were inferred."
            ),
        }
    )
    return artifact


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _population_std(values: Sequence[float]) -> float:
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _per_seed_row(
    seed: int,
    condition_a: ConditionMeasurement,
    condition_b: ConditionMeasurement,
    restored: bool,
) -> dict[str, object]:
    return {
        "seed": seed,
        "n_examples": condition_a.n_examples,
        "condition_a_production_auroc": condition_a.auroc,
        "condition_b_architecture_only_auroc": condition_b.auroc,
        "learning_contribution": condition_a.auroc - condition_b.auroc,
        "condition_a_per_verifier_auroc": dict(condition_a.per_verifier_auroc),
        "condition_b_per_verifier_auroc": dict(condition_b.per_verifier_auroc),
        "condition_a_state_visible_count": condition_a.state_visible_count,
        "condition_b_state_visible_count": condition_b.state_visible_count,
        "condition_a_fr11_state_loaded": condition_a.fr11_state_loaded,
        "condition_b_fr11_state_loaded": condition_b.fr11_state_loaded,
        "condition_a_python": condition_a.python_executable,
        "condition_b_python": condition_b.python_executable,
        "python_restarted_between_conditions": True,
        "subset_sha256": condition_a.subset_sha256,
        "condition_b_subset_sha256": condition_b.subset_sha256,
        "state_restored_sha_match_after_seed": restored,
    }


def _run_dual_condition_protocol(
    config: ExperimentConfig,
    state_files: Sequence[dict[str, object]],
    condition_scorer: ConditionScorer,
) -> list[dict[str, object]]:
    per_seed: list[dict[str, object]] = []
    for seed in config.random_seeds:
        condition_a = condition_scorer(config, seed, CONDITION_PRODUCTION, False)
        backup_root = config.backup_parent / f"fr11_state_backup_{seed}"
        with temporarily_move_state_files(config.repo_root, state_files, backup_root):
            condition_b = condition_scorer(config, seed, CONDITION_ARCHITECTURE_ONLY, True)
        restored = state_files_restored_sha_match(config.repo_root, state_files)
        per_seed.append(_per_seed_row(seed, condition_a, condition_b, restored))
    return per_seed


def _success_artifact(
    *,
    config: ExperimentConfig,
    duration_s: float,
    state_files: Sequence[dict[str, object]],
    checks: Sequence[PreconditionCheck],
    model_specs: dict[str, object],
    torch_output: object | None,
    per_seed_results: Sequence[dict[str, object]],
) -> dict[str, object]:
    a_values = [float(row["condition_a_production_auroc"]) for row in per_seed_results]
    b_values = [float(row["condition_b_architecture_only_auroc"]) for row in per_seed_results]
    a_mean = _mean(a_values)
    b_mean = _mean(b_values)
    per_a: dict[str, list[float]] = {}
    per_b: dict[str, list[float]] = {}
    for row in per_seed_results:
        for name, value in dict(row["condition_a_per_verifier_auroc"]).items():
            per_a.setdefault(str(name), []).append(float(value))
        for name, value in dict(row["condition_b_per_verifier_auroc"]).items():
            per_b.setdefault(str(name), []).append(float(value))

    artifact = _base_artifact(
        config=config,
        duration_s=duration_s,
        state_files=state_files,
        checks=checks,
        model_specs=model_specs,
        torch_output=torch_output,
    )
    artifact.update(
        {
            "schema": "carnot.fover_memory_leakage_isolation_2836.v1",
            "honest_verdict": (
                "complete: FoVer dual-condition FR-11 isolation measured with "
                "production and architecture-only memory conditions"
            ),
            "condition_a_production_auroc_mean": a_mean,
            "condition_a_production_auroc_std": _population_std(a_values),
            "condition_b_architecture_only_auroc_mean": b_mean,
            "condition_b_architecture_only_auroc_std": _population_std(b_values),
            "learning_contribution": a_mean - b_mean,
            "per_verifier_condition_a_auroc": per_a,
            "per_verifier_condition_b_auroc": per_b,
            "per_verifier_learning_contribution": {
                name: _mean(per_a[name]) - _mean(per_b[name])
                for name in sorted(per_a.keys() & per_b.keys())
            },
            "condition_results_by_seed": list(per_seed_results),
            "per_seed_results": list(per_seed_results),
            "state_reset_attempted": True,
            "state_files_restored_sha_match": state_files_restored_sha_match(
                config.repo_root, state_files
            ),
            "methodology_note": (
                "Condition A scored the same FoVer subset with FR-11 state visible. "
                "Condition B moved FR-11/NEXUS/constraint-template/session-memory "
                "files to the configured backup path, restarted scoring through the "
                "provided live ensemble-v7b scorer, and restored files before "
                "computing learning_contribution as A minus B."
            ),
        }
    )
    return artifact


def write_artifact(results_dir: Path, artifact: dict[str, object]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / OUTPUT_FILENAME).write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    command_runner: CommandRunner = _default_command_runner,
    condition_scorer: ConditionScorer | None = None,
    write: bool = True,
) -> dict[str, object]:
    """Run the Exp 2836 protocol or emit an honest blocked artifact."""

    config = config or ExperimentConfig()
    start = config.start_time()
    state_files = discover_fr11_state_files(config.repo_root)
    model_specs = find_qwen36_gguf(config)
    checks, torch_output = _probe_preconditions(
        config, command_runner, model_specs, state_files, condition_scorer
    )
    blocked = _blocked_prefix(checks)
    if blocked is not None:
        artifact = _blocked_artifact(
            config=config,
            duration_s=config.clock() - start,
            state_files=state_files,
            checks=checks,
            model_specs=model_specs,
            torch_output=torch_output,
        )
    else:
        assert condition_scorer is not None
        per_seed_results = _run_dual_condition_protocol(config, state_files, condition_scorer)
        artifact = _success_artifact(
            config=config,
            duration_s=config.clock() - start,
            state_files=state_files,
            checks=checks,
            model_specs=model_specs,
            torch_output=torch_output,
            per_seed_results=per_seed_results,
        )

    if write:
        write_artifact(config.output_dir(), artifact)
    return artifact
