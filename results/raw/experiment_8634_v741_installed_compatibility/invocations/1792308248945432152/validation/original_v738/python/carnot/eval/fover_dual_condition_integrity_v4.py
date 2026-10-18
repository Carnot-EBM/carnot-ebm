"""Exp 2850 FoVer dual-condition integrity rerun.

This runner repeats the useful part of Exp 2837: score the same FoVer
dataset-only verifier ensemble with production FR-11 state visible and with
that state removed. The important correction is provenance discipline. This
module does not call a local LLM, so the artifact explicitly sets
``live_model_invoked=false`` and ``compute_bound_claim=false`` and never emits
GGUF, CUDA, or ``model_specs`` claims.

Spec: REQ-VERIFY-2850,
      SCENARIO-VERIFY-2850,
      SCENARIO-VERIFY-2850-BLOCKED.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.eval.fover_memory_leakage_v3 import (
    CONDITION_ARCHITECTURE_ONLY,
    CONDITION_PRODUCTION,
    DEFAULT_RANDOM_SEEDS,
    ConditionMeasurement,
    ConditionScoringError,
    PreconditionCheck,
    discover_fr11_state_files,
    score_fover_subset,
    sha256_file,
    state_files_restored_sha_match,
    temporarily_move_state_files,
)


OUTPUT_FILENAME = "experiment_2850_fover_dual_condition_integrity_v4.json"
RUN_DATE = "20260522"
DEFAULT_N_EXAMPLES = 1000
REPO_ROOT = Path(__file__).resolve().parents[3]

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal complete:/blocked_ verdict; no inferred success.",
    "live_model_invoked": "False because this rerun scores dataset rows only.",
    "compute_bound_claim": "False because no local LLM inference is used.",
    "condition_a_production_auroc_mean": "Measured with FR-11 state files visible.",
    "condition_b_architecture_only_auroc_mean": "Measured with FR-11 state files absent.",
    "reproducibility_checksum": "Hashes inputs, state manifest, seeds, and score rows.",
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime configuration for the corrected FoVer rerun.

    The defaults encode the requested N=1000, five-seed protocol. Tests can use
    smaller values, but the production script should rely on the defaults so the
    artifact schema remains comparable to Exp 2837.
    """

    repo_root: Path = REPO_ROOT
    results_dir: Path | None = None
    random_seeds: tuple[int, ...] = DEFAULT_RANDOM_SEEDS
    n_examples: int = DEFAULT_N_EXAMPLES
    random_seed: int = DEFAULT_RANDOM_SEEDS[0]
    run_date: str = RUN_DATE
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def output_dir(self) -> Path:
        return self.results_dir if self.results_dir is not None else self.repo_root / "results"

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at


ConditionRunner = Callable[[ExperimentConfig, int, str, bool], ConditionMeasurement]
AdversarialVerifyRunner = Callable[[Path], dict[str, Any]]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _population_std(values: Sequence[float]) -> float:
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _count_labeled_fover_rows(path: Path) -> int:
    count = 0
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("label") in {"correct", "incorrect", 0, 1, "0", "1"}:
                    count += 1
    except OSError:
        return 0
    return count


def _dependency_detail(module_name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - exercised only when env breaks.
        return False, f"import failed: {exc}"
    version = getattr(module, "__version__", "unknown")
    return True, f"import ok; version={version}"


def input_file_hashes(repo_root: Path) -> dict[str, dict[str, object]]:
    """Return SHA256 evidence for non-state input files.

    The checksum separates corpus and optional NEXUS inputs from the FR-11 state
    manifest so a later reader can tell whether score drift came from the data,
    the state, or the score rows.
    """

    root = Path(repo_root)
    files = [
        root / "data" / "fover_corpus.jsonl",
        root / "results" / "nexus_constraint_memory_v2.json",
    ]
    hashes: dict[str, dict[str, object]] = {}
    for path in files:
        rel = path.relative_to(root).as_posix()
        if path.is_file():
            hashes[rel] = {
                "present": True,
                "n_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        else:
            hashes[rel] = {"present": False, "n_bytes": 0, "sha256": None}
    return hashes


def probe_preconditions(
    config: ExperimentConfig,
    state_files: Sequence[dict[str, object]],
) -> list[PreconditionCheck]:
    """Check dataset-only prerequisites without probing live-model resources."""

    root = Path(config.repo_root)
    fover_path = root / "data" / "fover_corpus.jsonl"
    labeled_count = _count_labeled_fover_rows(fover_path)
    sklearn_ok, sklearn_detail = _dependency_detail("sklearn")
    numpy_ok, numpy_detail = _dependency_detail("numpy")
    nexus_path = root / "results" / "nexus_constraint_memory_v2.json"
    return [
        PreconditionCheck(
            "fover_corpus",
            fover_path.is_file() and labeled_count >= config.n_examples,
            f"labeled_rows={labeled_count}; required>={config.n_examples}",
        ),
        PreconditionCheck(
            "nexus_constraint_memory_v2",
            nexus_path.is_file(),
            "present" if nexus_path.is_file() else "missing_optional",
        ),
        PreconditionCheck("sklearn", sklearn_ok, sklearn_detail),
        PreconditionCheck("numpy", numpy_ok, numpy_detail),
        PreconditionCheck(
            "fr11_state_files",
            bool(state_files),
            f"count={len(state_files)}",
        ),
    ]


def _blocked_verdict(checks: Sequence[PreconditionCheck]) -> str | None:
    for check in checks:
        if check.resource == "nexus_constraint_memory_v2":
            continue
        if check.available:
            continue
        if check.resource == "fover_corpus":
            return "blocked_fover_dataset"
        if check.resource in {"sklearn", "numpy"}:
            return "blocked_metrics_dependency"
        return f"blocked_{check.resource}"
    return None


def score_condition(
    config: ExperimentConfig,
    seed: int,
    condition: str,
    require_no_state: bool,
) -> ConditionMeasurement:
    """Score one seed in the current Python process using dataset-only verifiers."""

    return score_fover_subset(
        repo_root=config.repo_root,
        seed=seed,
        n_examples=config.n_examples,
        condition=condition,
        require_no_state=require_no_state,
    )


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
        "condition_a_per_verifier_auroc": condition_a.per_verifier_auroc,
        "condition_b_per_verifier_auroc": condition_b.per_verifier_auroc,
        "condition_a_state_visible_count": condition_a.state_visible_count,
        "condition_b_state_visible_count": condition_b.state_visible_count,
        "condition_a_fr11_state_loaded": condition_a.fr11_state_loaded,
        "condition_b_fr11_state_loaded": condition_b.fr11_state_loaded,
        "condition_a_python": condition_a.python_executable,
        "condition_b_python": condition_b.python_executable,
        "subset_sha256": condition_a.subset_sha256,
        "condition_b_subset_sha256": condition_b.subset_sha256,
        "state_restored_sha_match_after_seed": restored,
    }


def build_reproducibility_checksum(
    *,
    input_hashes: dict[str, dict[str, object]],
    state_files: Sequence[dict[str, object]],
    seeds: Sequence[int],
    n_examples: int,
    score_rows: Sequence[dict[str, object]],
) -> str:
    payload = {
        "input_hashes": input_hashes,
        "state_files": list(state_files),
        "random_seeds_used": list(seeds),
        "n_examples": n_examples,
        "score_rows": list(score_rows),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _base_artifact(
    *,
    config: ExperimentConfig,
    duration_s: float,
    state_files: Sequence[dict[str, object]],
    checks: Sequence[PreconditionCheck],
    input_hashes: dict[str, dict[str, object]],
    score_rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    return {
        "artifact": "experiment_2850_fover_dual_condition_integrity_v4",
        "schema": "carnot.fover_dual_condition_integrity_v4",
        "run_date": config.run_date,
        "n_examples": config.n_examples,
        "n_seeds": len(config.random_seeds),
        "random_seed": config.random_seed,
        "random_seeds_used": list(config.random_seeds),
        "fr11_state_files": list(state_files),
        "state_files_restored_sha_match": state_files_restored_sha_match(
            config.repo_root, state_files
        ),
        "live_model_invoked": False,
        "compute_bound_claim": False,
        "preconditions_checked": [check.as_dict() for check in checks],
        "duration_s": duration_s,
        "input_file_hashes": input_hashes,
        "reproducibility_checksum": build_reproducibility_checksum(
            input_hashes=input_hashes,
            state_files=state_files,
            seeds=config.random_seeds,
            n_examples=config.n_examples,
            score_rows=score_rows,
        ),
        "adversarial_verify_passed": False,
        "adversarial_verify_flags": [],
        "adversarial_verify_summary": {"status": "not_run"},
        "field_principles": FIELD_PRINCIPLES,
    }


def _blocked_artifact(
    *,
    config: ExperimentConfig,
    duration_s: float,
    state_files: Sequence[dict[str, object]],
    checks: Sequence[PreconditionCheck],
    input_hashes: dict[str, dict[str, object]],
) -> dict[str, object]:
    failed = [
        check
        for check in checks
        if not check.available and check.resource != "nexus_constraint_memory_v2"
    ]
    artifact = _base_artifact(
        config=config,
        duration_s=duration_s,
        state_files=state_files,
        checks=checks,
        input_hashes=input_hashes,
        score_rows=[],
    )
    artifact.update(
        {
            "honest_verdict": _blocked_verdict(checks) or "blocked_unknown_resource",
            "blocked_resources": [check.resource for check in failed],
            "condition_a_production_auroc_mean": None,
            "condition_a_production_auroc_std": None,
            "condition_b_architecture_only_auroc_mean": None,
            "condition_b_architecture_only_auroc_std": None,
            "learning_contribution": None,
            "per_seed_results": [],
            "per_verifier_condition_a_auroc": {},
            "per_verifier_condition_b_auroc": {},
            "per_verifier_learning_contribution": {},
            "methodology_note": (
                "Blocked before FoVer dual-condition scoring. No AUROC values "
                "were inferred and no live model was invoked."
            ),
        }
    )
    return artifact


def _summarize_success(
    *,
    config: ExperimentConfig,
    duration_s: float,
    state_files: Sequence[dict[str, object]],
    checks: Sequence[PreconditionCheck],
    input_hashes: dict[str, dict[str, object]],
    per_seed_results: Sequence[dict[str, object]],
) -> dict[str, object]:
    a_values = [float(row["condition_a_production_auroc"]) for row in per_seed_results]
    b_values = [float(row["condition_b_architecture_only_auroc"]) for row in per_seed_results]
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
        input_hashes=input_hashes,
        score_rows=per_seed_results,
    )
    a_mean = _mean(a_values)
    b_mean = _mean(b_values)
    artifact.update(
        {
            "honest_verdict": (
                "complete: FoVer dual-condition integrity rerun measured "
                "dataset-only production-vs-architecture contribution"
            ),
            "condition_a_production_auroc_mean": a_mean,
            "condition_a_production_auroc_std": _population_std(a_values),
            "condition_b_architecture_only_auroc_mean": b_mean,
            "condition_b_architecture_only_auroc_std": _population_std(b_values),
            "learning_contribution": a_mean - b_mean,
            "per_seed_results": list(per_seed_results),
            "per_verifier_condition_a_auroc": per_a,
            "per_verifier_condition_b_auroc": per_b,
            "per_verifier_learning_contribution": {
                name: _mean(per_a[name]) - _mean(per_b[name])
                for name in sorted(per_a.keys() & per_b.keys())
            },
            "methodology_note": (
                "Condition A scored FoVer rows with production FR-11 state "
                "files visible. Condition B moved FR-11/NEXUS/session state "
                "files aside, scored the same seed protocol with state absent, "
                "then restored the files and checked SHA256 equality. This is "
                "a dataset-only verifier score; no local LLM was invoked."
            ),
        }
    )
    return artifact


def write_artifact(results_dir: Path, artifact: dict[str, object]) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    output = results_dir / OUTPUT_FILENAME
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def run_adversarial_verify(path: Path) -> dict[str, Any]:
    script = REPO_ROOT / "scripts" / "adversarial_verify.py"
    if not script.is_file():
        return {
            "loaded": False,
            "flag_count": 0,
            "flags": [{"kind": "NOT_RUN", "severity": "info", "detail": "script missing"}],
        }
    command = [sys.executable, str(script), "--json", str(path)]
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {
            "loaded": False,
            "flag_count": 1,
            "flags": [
                {
                    "kind": "ADVERSARIAL_VERIFY_ERROR",
                    "severity": "warn",
                    "detail": (proc.stderr or proc.stdout or "invalid verifier output").strip(),
                }
            ],
        }
    reports = payload.get("reports") or []
    if reports:
        report = dict(reports[0])
        report["returncode"] = proc.returncode
        return report
    return {"loaded": False, "flag_count": 0, "flags": [], "returncode": proc.returncode}


def _attach_adversarial_report(
    artifact: dict[str, object],
    report: dict[str, Any],
) -> dict[str, object]:
    flags = list(report.get("flags") or [])
    artifact["adversarial_verify_passed"] = int(report.get("flag_count") or 0) == 0
    artifact["adversarial_verify_flags"] = flags
    artifact["adversarial_verify_summary"] = {
        "loaded": report.get("loaded"),
        "flag_count": report.get("flag_count", len(flags)),
        "max_severity": report.get("max_severity"),
        "returncode": report.get("returncode"),
    }
    return artifact


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    condition_runner: ConditionRunner = score_condition,
    adversarial_verify_runner: AdversarialVerifyRunner | None = run_adversarial_verify,
    write: bool = True,
) -> dict[str, object]:
    """Run the Exp 2850 FoVer integrity rerun or write an honest blocker."""

    config = config or ExperimentConfig()
    start = config.start_time()
    state_files = discover_fr11_state_files(config.repo_root)
    input_hashes = input_file_hashes(config.repo_root)
    checks = probe_preconditions(config, state_files)
    blocked = _blocked_verdict(checks)

    if blocked is not None:
        artifact = _blocked_artifact(
            config=config,
            duration_s=config.clock() - start,
            state_files=state_files,
            checks=checks,
            input_hashes=input_hashes,
        )
    else:
        per_seed_results: list[dict[str, object]] = []
        for seed in config.random_seeds:
            condition_a = condition_runner(config, seed, CONDITION_PRODUCTION, False)
            backup_root = Path("/tmp") / f"carnot_exp2850_fr11_state_backup_{os.getpid()}_{seed}"
            with temporarily_move_state_files(config.repo_root, state_files, backup_root):
                condition_b = condition_runner(config, seed, CONDITION_ARCHITECTURE_ONLY, True)
            restored = state_files_restored_sha_match(config.repo_root, state_files)
            if not restored:
                raise ConditionScoringError("FR-11 state restore SHA256 mismatch")
            per_seed_results.append(_per_seed_row(seed, condition_a, condition_b, restored))

        artifact = _summarize_success(
            config=config,
            duration_s=config.clock() - start,
            state_files=state_files,
            checks=checks,
            input_hashes=input_hashes,
            per_seed_results=per_seed_results,
        )

    if write:
        output = write_artifact(config.output_dir(), artifact)
        if adversarial_verify_runner is not None:
            artifact = _attach_adversarial_report(artifact, adversarial_verify_runner(output))
            write_artifact(config.output_dir(), artifact)
    return artifact
