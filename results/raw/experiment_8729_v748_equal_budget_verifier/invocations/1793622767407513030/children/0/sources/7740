#!/usr/bin/env python3
"""Exp 3767: local mechanical FoVer headline reproducer.

Spec: REQ-PUBLISH-3767,
      SCENARIO-PUBLISH-3767,
      SCENARIO-PUBLISH-3767B.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
for _path in (REPO_ROOT, PYTHON_ROOT):
    if str(_path) not in sys.path:  # pragma: no cover - direct script execution guard.
        sys.path.insert(0, str(_path))

from scripts import reproduce_fover_headline as reproducer  # noqa: E402


EXPERIMENT_ID = 3767
ARTIFACT_NAME = "experiment_3767_g2_mechanical_reproducer"
OUTPUT_REL_PATH = Path("results/experiment_3767_g2_mechanical_reproducer.json")
FROZEN_SOURCE_REL_PATH = Path("results/experiment_2837_fover_memory_leakage_v3.json")
FROZEN_AUROC_CI95 = (0.9027, 0.9235)
RANDOM_SEEDS = (42, 137, 271, 314, 1729)
N_EXAMPLES = 1000
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
DEPENDENCY_IMPORTS = ("yaml", "numpy", "sklearn")
VERIFIER_IMPORTS = {
    "fr11_session_memory": "carnot.eval.fover_memory_leakage_v3",
    "tier0r_curry_howard": "carnot.verify.tier0r_curry_howard",
    "tier0s_arithmetic_gap": "carnot.verify.tier0s_halluguard",
    "tier0u_logical_consistency": "carnot.verify.tier0u_logical_consistency",
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "reproduced_auroc_mean",
    "reproduced_auroc_ci95",
    "auroc_in_ci95",
    "per_seed_aurocs",
    "frozen_headline_unchanged",
    "preconditions_checked",
    "model_specs",
    "random_seeds_used",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix; blocked_<resource> if a precondition failed.",
    "inference_substrate": (
        "Bare verifier-scoring substrate; scores cached triples on CPU, no live LLM; "
        "1s floor."
    ),
    "reproduced_auroc_mean": (
        "The locally-reproduced mean AUROC; confirms the frozen headline still "
        "computes in the current environment."
    ),
    "reproduced_auroc_ci95": (
        "The reproduced 95% CI across seeds; the spread that must overlap the "
        "frozen [0.9027, 0.9235]."
    ),
    "auroc_in_ci95": (
        "BARE bool; true iff reproduced mean is within the frozen CI95. "
        "Downstream G2-local gate."
    ),
    "per_seed_aurocs": (
        "The five per-seed AUROCs; sample-size and reproducibility evidence, not "
        "a single point estimate."
    ),
    "frozen_headline_unchanged": (
        "Confirms this reproduces rather than re-versioning the frozen 0.9131 "
        "headline."
    ),
    "preconditions_checked": (
        "Interpreter, dependencies, corpus, and verifier imports checked before "
        "scoring; anti-fabrication gate."
    ),
    "model_specs": (
        "Names the FoVer corpus and four scoring verifiers; explicitly not a live "
        "LLM."
    ),
    "random_seeds_used": "The five seeds; determinism and reproducibility precondition.",
    "reproducibility_checksum": "Content hash catches silent corpus/protocol drift.",
    "duration_s": "Wall-clock plausibility floor; seconds for verifier scoring.",
}


@dataclass(frozen=True)
class PreconditionCheck:
    """One local resource checked before any scoring is allowed."""

    resource: str
    available: bool
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {
            "resource": self.resource,
            "available": self.available,
            "detail": self.detail,
        }


Importer = Callable[[str], object]
Clock = Callable[[], float]
ReproductionRunner = Callable[[Path, tuple[int, ...], int], Mapping[str, Any]]


def _round_metric(value: float | int, digits: int = 6) -> float:
    return round(float(value), digits)


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _line in handle)


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_checksum(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _seed_ci95(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    numeric = [float(value) for value in values]
    mean = sum(numeric) / len(numeric)
    if len(numeric) < 2:
        return {
            "mean": _round_metric(mean),
            "low": _round_metric(mean),
            "high": _round_metric(mean),
        }
    t_crit_by_n = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
    t_crit = t_crit_by_n.get(len(numeric), 1.96)
    sample_std = math.sqrt(sum((value - mean) ** 2 for value in numeric) / (len(numeric) - 1))
    half_width = t_crit * sample_std / math.sqrt(len(numeric))
    return {
        "mean": _round_metric(mean),
        "low": _round_metric(mean - half_width),
        "high": _round_metric(mean + half_width),
    }


def _venv_python(repo_root: Path) -> Path:
    return Path(repo_root) / ".venv" / "bin" / "python"


def _uses_repo_venv(python_executable: str, repo_root: Path) -> bool:
    return Path(python_executable) == _venv_python(repo_root)


def _missing_imports(names: Sequence[str], importer: Importer) -> list[str]:
    missing: list[str] = []
    for name in names:
        try:
            importer(name)
        except ImportError:
            missing.append(name)
    return missing


def check_preconditions(
    repo_root: Path = REPO_ROOT,
    *,
    python_executable: str = sys.executable,
    importer: Importer = importlib.import_module,
    n_examples: int = N_EXAMPLES,
) -> list[PreconditionCheck]:
    """Check every anti-fabrication precondition before scoring."""

    root = Path(repo_root)
    expected_python = _venv_python(root)
    interpreter_ok = _uses_repo_venv(python_executable, root)
    checks = [
        PreconditionCheck(
            "interpreter",
            interpreter_ok,
            f"actual={python_executable}; expected={expected_python}",
        )
    ]

    missing_dependencies = _missing_imports(DEPENDENCY_IMPORTS, importer)
    checks.append(
        PreconditionCheck(
            "python_dependencies",
            not missing_dependencies,
            "ok" if not missing_dependencies else "missing=" + ",".join(missing_dependencies),
        )
    )

    corpus_path = root / "data" / "fover_corpus.jsonl"
    if corpus_path.is_file():
        n_rows = _line_count(corpus_path)
        corpus_ok = n_rows >= n_examples
        detail = f"path={corpus_path}; line_count={n_rows}; required>={n_examples}"
    else:
        corpus_ok = False
        detail = f"path={corpus_path}; missing"
    checks.append(PreconditionCheck("fover_corpus", corpus_ok, detail))

    missing_verifiers = [
        verifier
        for verifier, module_name in VERIFIER_IMPORTS.items()
        if module_name in _missing_imports((module_name,), importer)
    ]
    checks.append(
        PreconditionCheck(
            "verifier_modules",
            not missing_verifiers,
            "ok" if not missing_verifiers else "missing=" + ",".join(missing_verifiers),
        )
    )
    return checks


def first_blocked_verdict(preconditions: Sequence[PreconditionCheck]) -> str | None:
    verdict_by_resource = {
        "interpreter": "blocked_interpreter_not_venv",
        "python_dependencies": "blocked_python_dependencies_missing",
        "fover_corpus": "blocked_fover_corpus_missing",
        "verifier_modules": "blocked_verifier_modules_unavailable",
        "reproducer_scoring": "blocked_reproducer_scoring_failed",
        "reproducer_execution": "blocked_reproducer_execution_failed",
    }
    for check in preconditions:
        if not check.available:
            return verdict_by_resource.get(check.resource, f"blocked_{check.resource}")
    return None


def _per_seed_aurocs(reproduction_result: Mapping[str, Any]) -> list[float]:
    rows = list(reproduction_result.get("per_seed_results") or [])
    return [
        _round_metric(float(dict(row)["condition_a_production_auroc"]))
        for row in rows
        if "condition_a_production_auroc" in dict(row)
    ]


def _reproduced_mean(
    reproduction_result: Mapping[str, Any],
    per_seed_aurocs: Sequence[float],
) -> float | None:
    raw = reproduction_result.get("condition_a_production_auroc_mean")
    if raw is not None:
        return _round_metric(float(raw))
    if per_seed_aurocs:
        return _round_metric(sum(per_seed_aurocs) / len(per_seed_aurocs))
    return None


def _reproduced_ci95(
    reproduction_result: Mapping[str, Any],
    per_seed_aurocs: Sequence[float],
) -> dict[str, float] | None:
    raw = reproduction_result.get("condition_a_production_auroc_ci95")
    if isinstance(raw, Mapping) and {"mean", "low", "high"} <= set(raw):
        return {
            "mean": _round_metric(float(raw["mean"])),
            "low": _round_metric(float(raw["low"])),
            "high": _round_metric(float(raw["high"])),
        }
    return _seed_ci95(per_seed_aurocs)


def _frozen_headline_source(repo_root: Path) -> dict[str, Any]:
    source_path = Path(repo_root) / FROZEN_SOURCE_REL_PATH
    if not source_path.is_file():
        return {
            "path": FROZEN_SOURCE_REL_PATH.as_posix(),
            "present": False,
            "headline_matches_frozen_0_9131": False,
        }
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    source_mean = payload.get("condition_a_production_auroc_mean")
    rounded = None if source_mean is None else round(float(source_mean), 4)
    return {
        "path": FROZEN_SOURCE_REL_PATH.as_posix(),
        "present": True,
        "condition_a_production_auroc_mean": source_mean,
        "headline_matches_frozen_0_9131": rounded == 0.9131,
    }


def _model_specs(repo_root: Path, n_examples: int) -> dict[str, Any]:
    return {
        "corpus": "FoVer step-error corpus",
        "corpus_path": str(Path(repo_root) / "data" / "fover_corpus.jsonl"),
        "candidate_source": "cached_fover_candidate_triples",
        "n_examples": n_examples,
        "dual_condition_protocol": True,
        "scoring_verifiers": list(VERIFIER_IMPORTS),
        "verifier_imports": dict(VERIFIER_IMPORTS),
        "live_model": None,
        "live_model_invoked": False,
        "source_headline_artifact": FROZEN_SOURCE_REL_PATH.as_posix(),
    }


def _reproducibility_checksum(
    *,
    repo_root: Path,
    seeds: Sequence[int],
    n_examples: int,
    upstream_checksum: str | None,
) -> str:
    corpus_path = Path(repo_root) / "data" / "fover_corpus.jsonl"
    payload = {
        "corpus_sha256": _sha256_file(corpus_path) or "missing",
        "random_seeds_used": list(seeds),
        "n_examples": n_examples,
        "scoring_verifiers": list(VERIFIER_IMPORTS),
        "frozen_source_artifact": FROZEN_SOURCE_REL_PATH.as_posix(),
        "upstream_reproducer_checksum": upstream_checksum,
    }
    return _json_checksum(payload)


def _base_artifact(
    *,
    repo_root: Path,
    preconditions: Sequence[PreconditionCheck],
    duration_s: float,
    seeds: Sequence[int],
    n_examples: int,
    upstream_checksum: str | None,
) -> dict[str, Any]:
    corpus_path = Path(repo_root) / "data" / "fover_corpus.jsonl"
    return {
        "experiment": EXPERIMENT_ID,
        "artifact": ARTIFACT_NAME,
        "schema": "carnot.g2_mechanical_reproducer.v1",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "live_model_invoked": False,
        "n_examples": n_examples,
        "n_seeds": len(seeds),
        "random_seeds_used": list(seeds),
        "preconditions_checked": [check.as_dict() for check in preconditions],
        "model_specs": _model_specs(repo_root, n_examples),
        "corpus_sha256": _sha256_file(corpus_path),
        "source_headline": _frozen_headline_source(repo_root),
        "field_principles": FIELD_PRINCIPLES,
        "duration_s": duration_s,
        "reproducibility_checksum": _reproducibility_checksum(
            repo_root=repo_root,
            seeds=seeds,
            n_examples=n_examples,
            upstream_checksum=upstream_checksum,
        ),
        "methodology_note": (
            "Verifier ensemble against cached FoVer candidates on CPU; no live "
            "LLM loaded or invoked. This confirms the frozen 0.9131 headline "
            "rather than moving it."
        ),
    }


def blocked_artifact(
    *,
    repo_root: Path,
    preconditions: Sequence[PreconditionCheck],
    duration_s: float,
    seeds: Sequence[int],
    n_examples: int,
    verdict_override: str | None = None,
) -> dict[str, Any]:
    verdict = verdict_override or first_blocked_verdict(preconditions) or "blocked_unknown_resource"
    failed = [check.resource for check in preconditions if not check.available]
    artifact = _base_artifact(
        repo_root=repo_root,
        preconditions=preconditions,
        duration_s=duration_s,
        seeds=seeds,
        n_examples=n_examples,
        upstream_checksum=None,
    )
    artifact.update(
        {
            "honest_verdict": verdict,
            "blocked_resources": failed,
            "reproduced_auroc_mean": None,
            "reproduced_auroc_ci95": None,
            "auroc_in_ci95": False,
            "per_seed_aurocs": [],
            "frozen_headline_unchanged": False,
            "discrepancy": None,
        }
    )
    return artifact


def artifact_from_reproduction(
    *,
    repo_root: Path,
    reproduction_result: Mapping[str, Any],
    preconditions: Sequence[PreconditionCheck],
    duration_s: float,
    seeds: Sequence[int],
    n_examples: int,
) -> dict[str, Any]:
    per_seed_aurocs = _per_seed_aurocs(reproduction_result)
    reproduced_mean = _reproduced_mean(reproduction_result, per_seed_aurocs)
    reproduced_ci95 = _reproduced_ci95(reproduction_result, per_seed_aurocs)
    auroc_in_ci95 = bool(
        reproduced_mean is not None
        and FROZEN_AUROC_CI95[0] <= reproduced_mean <= FROZEN_AUROC_CI95[1]
    )
    source = _frozen_headline_source(repo_root)
    frozen_headline_unchanged = bool(source.get("headline_matches_frozen_0_9131"))
    mean_token = "none" if reproduced_mean is None else f"{reproduced_mean:.6f}"
    bool_token = str(auroc_in_ci95).lower()
    verdict_suffix = (
        "frozen_headline_confirmed_unchanged"
        if auroc_in_ci95
        else "discrepancy_recorded_frozen_headline_unchanged"
    )
    artifact = _base_artifact(
        repo_root=repo_root,
        preconditions=preconditions,
        duration_s=duration_s,
        seeds=seeds,
        n_examples=n_examples,
        upstream_checksum=str(reproduction_result.get("reproducibility_checksum") or ""),
    )
    artifact.update(
        {
            "honest_verdict": (
                "complete: g2_local_reproducer_committed_auroc_"
                f"{mean_token}_in_ci95_{bool_token}_{verdict_suffix}"
            ),
            "reproduced_auroc_mean": reproduced_mean,
            "reproduced_auroc_ci95": reproduced_ci95,
            "auroc_in_ci95": auroc_in_ci95,
            "per_seed_aurocs": per_seed_aurocs,
            "frozen_headline_unchanged": frozen_headline_unchanged,
            "discrepancy": None
            if auroc_in_ci95
            else {
                "frozen_ci95": list(FROZEN_AUROC_CI95),
                "reproduced_auroc_mean": reproduced_mean,
            },
            "upstream_reproducer_checksum": reproduction_result.get(
                "reproducibility_checksum"
            ),
            "condition_b_architecture_only_auroc_mean": reproduction_result.get(
                "condition_b_architecture_only_auroc_mean"
            ),
            "learning_contribution_ci95": reproduction_result.get(
                "learning_contribution_ci95"
            ),
        }
    )
    return artifact


def _default_reproduction_runner(
    repo_root: Path,
    seeds: tuple[int, ...],
    n_examples: int,
) -> Mapping[str, Any]:
    return reproducer.run_reproduction(repo_root, seeds, n_examples)


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_experiment(
    *,
    repo_root: Path = REPO_ROOT,
    output_path: Path | None = None,
    python_executable: str = sys.executable,
    importer: Importer = importlib.import_module,
    clock: Clock = time.time,
    reproduction_runner: ReproductionRunner = _default_reproduction_runner,
) -> Path:
    started_s = clock()
    root = Path(repo_root)
    target = Path(output_path) if output_path is not None else root / OUTPUT_REL_PATH
    checks = check_preconditions(root, python_executable=python_executable, importer=importer)
    blocked = first_blocked_verdict(checks)
    if blocked is not None:
        artifact = blocked_artifact(
            repo_root=root,
            preconditions=checks,
            duration_s=clock() - started_s,
            seeds=RANDOM_SEEDS,
            n_examples=N_EXAMPLES,
            verdict_override=blocked,
        )
        return write_artifact(target, artifact)

    try:
        result = reproduction_runner(root, RANDOM_SEEDS, N_EXAMPLES)
    except Exception as exc:
        failed_checks = [
            *checks,
            PreconditionCheck("reproducer_execution", False, f"{type(exc).__name__}: {exc}"),
        ]
        artifact = blocked_artifact(
            repo_root=root,
            preconditions=failed_checks,
            duration_s=clock() - started_s,
            seeds=RANDOM_SEEDS,
            n_examples=N_EXAMPLES,
        )
        return write_artifact(target, artifact)

    verdict = str(result.get("honest_verdict") or "")
    if verdict.startswith("blocked"):
        failed_checks = [*checks, PreconditionCheck("reproducer_scoring", False, verdict)]
        artifact = blocked_artifact(
            repo_root=root,
            preconditions=failed_checks,
            duration_s=clock() - started_s,
            seeds=RANDOM_SEEDS,
            n_examples=N_EXAMPLES,
            verdict_override=verdict,
        )
        return write_artifact(target, artifact)

    artifact = artifact_from_reproduction(
        repo_root=root,
        reproduction_result=result,
        preconditions=checks,
        duration_s=clock() - started_s,
        seeds=RANDOM_SEEDS,
        n_examples=N_EXAMPLES,
    )
    return write_artifact(target, artifact)


def main() -> int:
    path = run_experiment()
    artifact = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact.get("auroc_in_ci95") is True else 1


if __name__ == "__main__":  # pragma: no cover - script entry point.
    raise SystemExit(main())
