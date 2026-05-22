"""Exp 2837 FoVer FR-11 memory-leakage isolation.

This evaluator answers a narrow question: does persistent FR-11 state add
measurable FoVer discrimination, or does the verifier architecture still work
when that state is absent?  The runner is intentionally conservative. It gates
on Exp 2836's SOTA runtime preflight, records every FR-11/NEXUS/session state
file by SHA256, scores each condition in a fresh Python subprocess, moves state
files aside only for the architecture-only condition, and restores them before
reporting success.

Spec: REQ-VERIFY-2837,
      SCENARIO-VERIFY-2837,
      SCENARIO-VERIFY-2837-BLOCKED.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OUTPUT_FILENAME = "experiment_2837_fover_memory_leakage_v3.json"
EXP2836_FILENAME = "experiment_2836_sota_runtime_preflight.json"
CONDITION_PRODUCTION = "production"
CONDITION_ARCHITECTURE_ONLY = "architecture_only"
DEFAULT_RANDOM_SEEDS = (42, 137, 271, 314, 1729)
DEFAULT_N_EXAMPLES = 1000
PRIMARY_SOTA_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
LEGACY_CPU_SMOKE_ONLY = ("Qwen3.5-0.8B", "gemma-4-E4B-it")
REPO_ROOT = Path(__file__).resolve().parents[3]
FR11_MEMORY_BOOST = 1.0

FR11_STATE_GLOBS = (
    "data/constraint_memory.db",
    "data/fr11_*.json",
    "data/fr11_*.jsonl",
    "results/constraint_memory*.json",
    "results/constraint_patterns*.json",
    "results/fr11_*.json",
    "results/fr11_*.jsonl",
    "results/nexus_constraint_memory*.json",
    "results/session_memory_*/**/session_state.json",
    "results/exp_448_session_memory/**/session_state.json",
)

FIELD_PRINCIPLES = {
    "honest_verdict": 'MUST start with "complete:" / "success:" or "blocked_".',
    "condition_a_production_auroc_mean": "Production headline condition; never inferred.",
    "condition_b_architecture_only_auroc_mean": (
        "Architecture-only control condition; never inferred."
    ),
    "learning_contribution": "A - B quantifies FR-11 contribution.",
    "per_seed_results": "5-seed replication required.",
    "fr11_state_files": "Memory-leakage audit trail.",
    "state_files_restored_sha_match": "Non-destructive reset proof.",
    "model_specs": "Mandated SOTA GGUF recorded.",
    "preconditions_checked": "Explains blocks without fabrication.",
    "duration_s": "Real compute wall-time; no sleep padding.",
}


@dataclass(frozen=True)
class PreconditionCheck:
    """One prerequisite checked before any FoVer scoring."""

    resource: str
    available: bool
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {
            "resource": self.resource,
            "available": self.available,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ConditionMeasurement:
    """One condition score for one seed."""

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

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ConditionMeasurement":
        return cls(
            seed=int(payload["seed"]),
            condition=str(payload["condition"]),
            auroc=float(payload["auroc"]),
            per_verifier_auroc={
                str(key): float(value)
                for key, value in dict(payload["per_verifier_auroc"]).items()
            },
            n_examples=int(payload["n_examples"]),
            state_visible_count=int(payload["state_visible_count"]),
            fr11_state_loaded=bool(payload["fr11_state_loaded"]),
            subset_sha256=str(payload["subset_sha256"]),
            python_executable=str(payload["python_executable"]),
        )


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime configuration for Exp 2837."""

    repo_root: Path = REPO_ROOT
    results_dir: Path | None = None
    exp2836_path: Path | None = None
    random_seeds: tuple[int, ...] = DEFAULT_RANDOM_SEEDS
    n_examples: int = DEFAULT_N_EXAMPLES
    started_at: float | None = None
    clock: Callable[[], float] = time.time
    subprocess_timeout_s: int = 120

    def output_dir(self) -> Path:
        return self.results_dir if self.results_dir is not None else self.repo_root / "results"

    def preflight_path(self) -> Path:
        if self.exp2836_path is not None:
            return self.exp2836_path
        return self.output_dir() / EXP2836_FILENAME

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at


class ConditionScoringError(RuntimeError):
    """Raised when a condition subprocess cannot produce a valid score."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def discover_fr11_state_files(repo_root: Path) -> list[dict[str, object]]:
    """Enumerate FR-11/NEXUS/session state files without mutating them."""

    root = Path(repo_root)
    paths: set[Path] = set()
    for pattern in FR11_STATE_GLOBS:
        paths.update(path for path in root.glob(pattern) if path.is_file())

    records = [
        {
            "path": _relative_path(root, path),
            "n_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    return sorted(records, key=lambda item: str(item["path"]))


def state_files_restored_sha_match(
    repo_root: Path,
    state_files: Sequence[dict[str, object]],
) -> bool:
    for item in state_files:
        path = Path(repo_root) / str(item["path"])
        if not path.is_file() or sha256_file(path) != str(item["sha256"]):
            return False
    return True


@contextmanager
def temporarily_move_state_files(
    repo_root: Path,
    state_files: Sequence[dict[str, object]],
    backup_root: Path,
) -> Iterator[None]:
    """Move state files aside and always restore them before returning."""

    root = Path(repo_root)
    backup = Path(backup_root)
    moved: list[tuple[Path, Path]] = []
    try:
        for item in state_files:
            rel = Path(str(item["path"]))
            source = root / rel
            if not source.is_file():
                raise ConditionScoringError(f"state file disappeared before reset: {rel}")
            target = backup / rel
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


def load_exp2836_preflight(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_model_paths(value: Any) -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"model_path", "path", "resolved_gguf", "resolved_path"} and item:
                paths.append(str(item))
            else:
                paths.extend(_extract_model_paths(item))
    elif isinstance(value, list | tuple):
        for item in value:
            paths.extend(_extract_model_paths(item))
    return paths


def model_specs_from_exp2836(preflight: dict[str, Any]) -> dict[str, object]:
    """Normalize Exp 2836 model/runtime evidence for the Exp 2837 artifact."""

    cached_pair = dict(preflight.get("cached_sota_pair_result") or {})
    cached_pair_paths = _extract_model_paths(cached_pair.get("result"))
    smoke_results = [
        dict(row)
        for row in preflight.get("smoke_load_results", [])
        if row.get("load_success") and row.get("headline_usable") and row.get("model_path")
    ]
    smoke_paths = [str(row["model_path"]) for row in smoke_results]
    selected_path = (cached_pair_paths or smoke_paths or [None])[0]
    selected_hf_id = str(smoke_results[0].get("hf_id")) if smoke_results else None

    raw_specs = dict(preflight.get("model_specs") or {})
    return {
        "headline_required_any_of": list(raw_specs.get("primary") or PRIMARY_SOTA_MODEL_IDS),
        "legacy_cpu_smoke_only": list(
            raw_specs.get("legacy_cpu_smoke_only") or LEGACY_CPU_SMOKE_ONLY
        ),
        "sota_runtime_ready": bool(preflight.get("sota_runtime_ready")),
        "selected_python": preflight.get("selected_python"),
        "cached_sota_pair_result": cached_pair,
        "cached_sota_pair_model_paths": cached_pair_paths,
        "smoke_model_paths": smoke_paths,
        "selected_model_path": selected_path,
        "selected_model_hf_id": selected_hf_id,
        "sota_models_cached": list(preflight.get("sota_models_cached") or []),
    }


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _line in handle)


def probe_preconditions(
    config: ExperimentConfig,
    state_files: Sequence[dict[str, object]],
    model_specs: dict[str, object],
) -> list[PreconditionCheck]:
    """Check Exp 2837 gates before condition scoring begins."""

    preflight_exists = config.preflight_path().is_file()
    selected_python = model_specs.get("selected_python")
    selected_model_path = model_specs.get("selected_model_path")
    checks = [
        PreconditionCheck(
            "exp2836_artifact",
            preflight_exists,
            str(config.preflight_path()) if preflight_exists else "missing",
        ),
        PreconditionCheck(
            "exp2836_sota_runtime_ready",
            bool(model_specs.get("sota_runtime_ready")),
            f"sota_runtime_ready={model_specs.get('sota_runtime_ready')}",
        ),
        PreconditionCheck(
            "exp2836_selected_python",
            bool(selected_python),
            str(selected_python) if selected_python else "missing",
        ),
        PreconditionCheck(
            "mandated_sota_model_path",
            bool(selected_model_path),
            str(selected_model_path) if selected_model_path else "missing",
        ),
    ]

    fover_path = config.repo_root / "data" / "fover_corpus.jsonl"
    if fover_path.exists():
        n_rows = _line_count(fover_path)
        checks.append(
            PreconditionCheck(
                "fover_corpus",
                n_rows >= config.n_examples,
                f"line_count={n_rows}; required>={config.n_examples}",
            )
        )
    else:
        checks.append(PreconditionCheck("fover_corpus", False, "missing"))
    checks.append(
        PreconditionCheck(
            "fr11_state_files",
            bool(state_files),
            f"count={len(state_files)}",
        )
    )
    return checks


def _blocked_verdict(checks: Sequence[PreconditionCheck]) -> str | None:
    verdict_by_resource = {
        "exp2836_artifact": "blocked_exp2836_missing",
        "exp2836_sota_runtime_ready": "blocked_sota_runtime_not_ready",
        "exp2836_selected_python": "blocked_selected_python_missing",
        "mandated_sota_model_path": "blocked_model_path",
        "fover_corpus": "blocked_fover_corpus",
        "fr11_state_files": "blocked_fr11_state_files",
    }
    for check in checks:
        if not check.available:
            return verdict_by_resource.get(check.resource, f"blocked_{check.resource}")
    return None


def _read_fover_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        label = row.get("label")
        if label not in {"correct", "incorrect", 0, 1, "0", "1"}:
            continue
        rows.append(row)
    return rows


def _label_to_int(label: Any) -> int:
    if label in {"incorrect", 1, "1"}:
        return 1
    if label in {"correct", 0, "0"}:
        return 0
    raise ValueError(f"unsupported FoVer label: {label!r}")


def _select_balanced_subset(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int,
    n_examples: int,
) -> list[dict[str, Any]]:
    positives = [row for row in rows if _label_to_int(row["label"]) == 1]
    negatives = [row for row in rows if _label_to_int(row["label"]) == 0]
    n_pos = n_examples // 2
    n_neg = n_examples - n_pos
    if len(positives) < n_pos or len(negatives) < n_neg:
        raise ConditionScoringError(
            f"FoVer corpus lacks class balance for n={n_examples}: "
            f"positives={len(positives)}, negatives={len(negatives)}"
        )
    rng = random.Random(seed)
    subset = [*rng.sample(positives, n_pos), *rng.sample(negatives, n_neg)]
    rng.shuffle(subset)
    return subset


def compute_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute AUROC where label 1 is the incorrect/error class."""

    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    n_pos = sum(1 for label in labels if int(label) == 1)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC requires both positive and negative labels")

    ranked = sorted(enumerate(scores), key=lambda item: float(item[1]))
    ranks = [0.0] * len(scores)
    cursor = 0
    while cursor < len(ranked):
        end = cursor + 1
        while end < len(ranked) and float(ranked[end][1]) == float(ranked[cursor][1]):
            end += 1
        avg_rank = (cursor + 1 + end) / 2.0
        for offset in range(cursor, end):
            ranks[ranked[offset][0]] = avg_rank
        cursor = end

    pos_rank_sum = sum(rank for rank, label in zip(ranks, labels, strict=True) if int(label) == 1)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text) if len(token) > 2}


def _load_fr11_memory_index(repo_root: Path) -> dict[str, object]:
    question_ids: set[str] = set()
    prompt_token_sets: list[set[str]] = []
    for item in discover_fr11_state_files(repo_root):
        path = repo_root / str(item["path"])
        if path.name == "session_state.json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            entries = payload.get("case_memory", {}).get("entries", [])
            for entry in entries:
                key = dict(entry.get("key") or {})
                violation_text = " ".join(
                    str(part)
                    for part in [
                        *key.get("violation_families", []),
                        *entry.get("violation_types", []),
                    ]
                )
                benchmark_slice = str(key.get("benchmark_slice", ""))
                if benchmark_slice.startswith("fover:") and "incorrect" in violation_text:
                    suffix = benchmark_slice.split(":", 1)[1]
                    question_ids.add(suffix)
                    if suffix.startswith("math_v3_"):
                        question_ids.add(suffix.removeprefix("math_v3_"))
                    prompt_tokens = set(entry.get("prompt_tokens") or [])
                    prompt_tokens.update(_tokens(str(key.get("prompt_sketch", ""))))
                    if prompt_tokens:
                        prompt_token_sets.append({str(token).lower() for token in prompt_tokens})
        elif path.suffix == ".jsonl" and "fr11_" in path.name:
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("is_correct") is False or row.get("verifier_verdict") == "incorrect":
                    qid = row.get("question_id")
                    if qid is not None:
                        question_ids.add(str(qid))
    return {"question_ids": question_ids, "prompt_token_sets": prompt_token_sets}


def _fr11_memory_score(row: dict[str, Any], memory_index: dict[str, object]) -> float:
    question_ids = memory_index.get("question_ids", set())
    prompt_token_sets = memory_index.get("prompt_token_sets", [])
    qid = str(row.get("question_id", ""))
    if qid in question_ids or f"math_v3_{qid}" in question_ids:
        return 1.0
    row_tokens = _tokens(str(row.get("step_text", "")))
    best_overlap = 0.0
    for token_set in prompt_token_sets:  # type: ignore[assignment]
        candidate = set(token_set)
        if not candidate:
            continue
        best_overlap = max(best_overlap, len(row_tokens & candidate) / len(candidate))
    return min(1.0, best_overlap)


def _subset_sha(rows: Sequence[dict[str, Any]]) -> str:
    payload = [
        {
            "question_id": row.get("question_id"),
            "label": row.get("label"),
            "step_text_sha256": hashlib.sha256(
                str(row.get("step_text", "")).encode("utf-8")
            ).hexdigest(),
        }
        for row in rows
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _score_text_verifiers(texts: Sequence[str]) -> dict[str, list[float]]:
    from carnot.verify.tier0r_curry_howard import Tier0rVerifier
    from carnot.verify.tier0s_halluguard import Tier0sVerifier
    from carnot.verify.tier0u_logical_consistency import Tier0uVerifier

    tier0r = Tier0rVerifier()
    tier0s = Tier0sVerifier()
    tier0u = Tier0uVerifier()
    return {
        "tier0r_curry_howard": [float(tier0r.score(text)) for text in texts],
        "tier0u_logical_consistency": [float(tier0u.score(text)) for text in texts],
        "tier0s_arithmetic_gap": [
            min(1.0, float(tier0s.halluguard_ntk_score(text)) / 100.0) for text in texts
        ],
    }


def score_fover_subset(
    *,
    repo_root: Path,
    seed: int,
    n_examples: int,
    condition: str,
    require_no_state: bool,
) -> ConditionMeasurement:
    """Score one FoVer seed in the current Python process."""

    state_visible_count = len(discover_fr11_state_files(repo_root))
    if require_no_state and state_visible_count != 0:
        raise ConditionScoringError(
            f"architecture-only condition saw {state_visible_count} FR-11 state files"
        )
    rows = _select_balanced_subset(
        _read_fover_rows(repo_root / "data" / "fover_corpus.jsonl"),
        seed=seed,
        n_examples=n_examples,
    )
    labels = [_label_to_int(row["label"]) for row in rows]
    texts = [str(row.get("step_text", "")) for row in rows]
    verifier_scores = _score_text_verifiers(texts)

    architecture_scores = [
        0.9 * r_score + 0.1 * u_score
        for r_score, u_score in zip(
            verifier_scores["tier0r_curry_howard"],
            verifier_scores["tier0u_logical_consistency"],
            strict=True,
        )
    ]
    fr11_state_loaded = False
    if condition == CONDITION_PRODUCTION and state_visible_count > 0:
        memory_index = _load_fr11_memory_index(repo_root)
        memory_scores = [_fr11_memory_score(row, memory_index) for row in rows]
        verifier_scores["fr11_session_memory"] = memory_scores
        architecture_scores = [
            score + FR11_MEMORY_BOOST * memory_score
            for score, memory_score in zip(architecture_scores, memory_scores, strict=True)
        ]
        fr11_state_loaded = bool(memory_index["question_ids"] or memory_index["prompt_token_sets"])
    elif condition != CONDITION_ARCHITECTURE_ONLY:
        raise ConditionScoringError(f"unknown condition: {condition}")

    per_verifier_auroc = {
        name: compute_auroc(labels, scores) for name, scores in verifier_scores.items()
    }
    return ConditionMeasurement(
        seed=seed,
        condition=condition,
        auroc=compute_auroc(labels, architecture_scores),
        per_verifier_auroc=per_verifier_auroc,
        n_examples=len(rows),
        state_visible_count=state_visible_count,
        fr11_state_loaded=fr11_state_loaded,
        subset_sha256=_subset_sha(rows),
        python_executable=sys.executable,
    )


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def score_condition_via_subprocess(
    config: ExperimentConfig,
    selected_python: str,
    seed: int,
    condition: str,
    require_no_state: bool,
    *,
    command_runner: CommandRunner = subprocess.run,
) -> ConditionMeasurement:
    """Run one condition in a fresh selected-Python process."""

    command = [
        selected_python,
        "-m",
        "carnot.eval.fover_memory_leakage_v3",
        "--score-condition",
        "--repo-root",
        str(config.repo_root),
        "--seed",
        str(seed),
        "--n-examples",
        str(config.n_examples),
        "--condition",
        condition,
    ]
    if require_no_state:
        command.append("--require-no-state")

    env = os.environ.copy()
    python_dir = str(config.repo_root / "python")
    env["PYTHONPATH"] = python_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = command_runner(
        command,
        capture_output=True,
        text=True,
        timeout=config.subprocess_timeout_s,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or f"returncode={proc.returncode}").strip()
        raise ConditionScoringError(detail)
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ConditionScoringError(f"condition subprocess returned invalid JSON: {exc}") from exc
    return ConditionMeasurement.from_dict(payload)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _population_std(values: Sequence[float]) -> float:
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _ci95(values: Sequence[float]) -> dict[str, float]:
    mean = _mean(values)
    if len(values) < 2:
        return {"mean": mean, "low": mean, "high": mean}
    t_crit_by_n = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
    t_crit = t_crit_by_n.get(len(values), 1.96)
    sample_std = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
    half_width = t_crit * sample_std / math.sqrt(len(values))
    return {"mean": mean, "low": mean - half_width, "high": mean + half_width}


def _reproducibility_checksum(
    *,
    state_files: Sequence[dict[str, object]],
    model_specs: dict[str, object],
    seeds: Sequence[int],
    n_examples: int,
) -> str:
    payload = {
        "state_files": list(state_files),
        "model_specs": model_specs,
        "random_seeds_used": list(seeds),
        "n_examples": n_examples,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _base_artifact(
    *,
    config: ExperimentConfig,
    duration_s: float,
    state_files: Sequence[dict[str, object]],
    checks: Sequence[PreconditionCheck],
    model_specs: dict[str, object],
) -> dict[str, object]:
    return {
        "artifact": "experiment_2837_fover_memory_leakage_v3",
        "schema": "carnot.fover_memory_leakage_v3",
        "n_examples": config.n_examples,
        "n_seeds": len(config.random_seeds),
        "random_seeds_used": list(config.random_seeds),
        "fr11_state_files": list(state_files),
        "state_files_restored_sha_match": state_files_restored_sha_match(
            config.repo_root, state_files
        ),
        "model_specs": model_specs,
        "preconditions_checked": [check.as_dict() for check in checks],
        "duration_s": duration_s,
        "reproducibility_checksum": _reproducibility_checksum(
            state_files=state_files,
            model_specs=model_specs,
            seeds=config.random_seeds,
            n_examples=config.n_examples,
        ),
        "field_principles": FIELD_PRINCIPLES,
    }


def _blocked_artifact(
    *,
    config: ExperimentConfig,
    duration_s: float,
    state_files: Sequence[dict[str, object]],
    checks: Sequence[PreconditionCheck],
    model_specs: dict[str, object],
) -> dict[str, object]:
    verdict = _blocked_verdict(checks) or "blocked_unknown_resource"
    failed = [check for check in checks if not check.available]
    artifact = _base_artifact(
        config=config,
        duration_s=duration_s,
        state_files=state_files,
        checks=checks,
        model_specs=model_specs,
    )
    artifact.update(
        {
            "honest_verdict": verdict,
            "blocked_resources": [check.resource for check in failed],
            "condition_a_production_auroc_mean": None,
            "condition_a_production_auroc_std": None,
            "condition_a_production_auroc_ci95": None,
            "condition_b_architecture_only_auroc_mean": None,
            "condition_b_architecture_only_auroc_std": None,
            "condition_b_architecture_only_auroc_ci95": None,
            "learning_contribution": None,
            "learning_contribution_ci95": None,
            "per_seed_results": [],
            "per_verifier_condition_a_auroc": {},
            "per_verifier_condition_b_auroc": {},
            "per_verifier_learning_contribution": {},
            "methodology_note": (
                "Blocked before FoVer dual-condition scoring because required "
                "resources were missing: "
                + ", ".join(check.resource for check in failed)
                + ". No AUROC values were inferred."
            ),
        }
    )
    return artifact


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
        "python_restarted_between_conditions": True,
        "condition_a_python": condition_a.python_executable,
        "condition_b_python": condition_b.python_executable,
        "subset_sha256": condition_a.subset_sha256,
        "condition_b_subset_sha256": condition_b.subset_sha256,
        "state_restored_sha_match_after_seed": restored,
    }


def _summarize_success(
    *,
    config: ExperimentConfig,
    duration_s: float,
    state_files: Sequence[dict[str, object]],
    checks: Sequence[PreconditionCheck],
    model_specs: dict[str, object],
    per_seed_results: Sequence[dict[str, object]],
) -> dict[str, object]:
    a_values = [float(row["condition_a_production_auroc"]) for row in per_seed_results]
    b_values = [float(row["condition_b_architecture_only_auroc"]) for row in per_seed_results]
    learning_values = [a - b for a, b in zip(a_values, b_values, strict=True)]
    a_mean = _mean(a_values)
    b_mean = _mean(b_values)

    per_a: dict[str, list[float]] = {}
    per_b: dict[str, list[float]] = {}
    for row in per_seed_results:
        for name, value in dict(row["condition_a_per_verifier_auroc"]).items():
            per_a.setdefault(name, []).append(float(value))
        for name, value in dict(row["condition_b_per_verifier_auroc"]).items():
            per_b.setdefault(name, []).append(float(value))

    artifact = _base_artifact(
        config=config,
        duration_s=duration_s,
        state_files=state_files,
        checks=checks,
        model_specs=model_specs,
    )
    artifact.update(
        {
            "honest_verdict": (
                "complete: FoVer memory-leakage v3 measured with production "
                "state and architecture-only reset conditions"
            ),
            "condition_a_production_auroc_mean": a_mean,
            "condition_a_production_auroc_std": _population_std(a_values),
            "condition_a_production_auroc_ci95": _ci95(a_values),
            "condition_b_architecture_only_auroc_mean": b_mean,
            "condition_b_architecture_only_auroc_std": _population_std(b_values),
            "condition_b_architecture_only_auroc_ci95": _ci95(b_values),
            "learning_contribution": a_mean - b_mean,
            "learning_contribution_ci95": _ci95(learning_values),
            "per_seed_results": list(per_seed_results),
            "per_verifier_condition_a_auroc": per_a,
            "per_verifier_condition_b_auroc": per_b,
            "per_verifier_learning_contribution": {
                name: _mean(per_a[name]) - _mean(per_b[name])
                for name in sorted(per_a.keys() & per_b.keys())
            },
            "methodology_note": (
                "Condition A scored the FoVer subset in a fresh selected-Python "
                "process with FR-11 state visible. Condition B moved every "
                "FR-11/NEXUS/session state file aside, restarted Python for "
                "architecture-only scoring, verified zero state files were "
                "visible, then restored all files and checked SHA256 equality."
            ),
        }
    )
    return artifact


ConditionRunner = Callable[[ExperimentConfig, str, int, str, bool], ConditionMeasurement]


def write_artifact(results_dir: Path, artifact: dict[str, object]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / OUTPUT_FILENAME).write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    precondition_probe: Callable[
        [ExperimentConfig, Sequence[dict[str, object]], dict[str, object]],
        list[PreconditionCheck],
    ] = probe_preconditions,
    condition_runner: ConditionRunner = score_condition_via_subprocess,
    write: bool = True,
) -> dict[str, object]:
    """Run Exp 2837 or write a blocked artifact before scoring."""

    config = config or ExperimentConfig()
    start = config.start_time()
    state_files = discover_fr11_state_files(config.repo_root)
    preflight = load_exp2836_preflight(config.preflight_path())
    model_specs = model_specs_from_exp2836(preflight)
    checks = precondition_probe(config, state_files, model_specs)
    selected_python = str(model_specs.get("selected_python") or "")

    blocked = _blocked_verdict(checks)
    if blocked is not None:
        artifact = _blocked_artifact(
            config=config,
            duration_s=config.clock() - start,
            state_files=state_files,
            checks=checks,
            model_specs=model_specs,
        )
    else:
        per_seed_results: list[dict[str, object]] = []
        for seed in config.random_seeds:
            condition_a = condition_runner(
                config,
                selected_python,
                seed,
                CONDITION_PRODUCTION,
                False,
            )
            backup_root = Path("/tmp") / f"carnot_exp2837_fr11_state_backup_{os.getpid()}_{seed}"
            with temporarily_move_state_files(config.repo_root, state_files, backup_root):
                condition_b = condition_runner(
                    config,
                    selected_python,
                    seed,
                    CONDITION_ARCHITECTURE_ONLY,
                    True,
                )
            restored = state_files_restored_sha_match(config.repo_root, state_files)
            per_seed_results.append(_per_seed_row(seed, condition_a, condition_b, restored))

        artifact = _summarize_success(
            config=config,
            duration_s=config.clock() - start,
            state_files=state_files,
            checks=checks,
            model_specs=model_specs,
            per_seed_results=per_seed_results,
        )

    if write:
        write_artifact(config.output_dir(), artifact)
    return artifact


def _score_condition_main(args: argparse.Namespace) -> int:
    measurement = score_fover_subset(
        repo_root=Path(args.repo_root),
        seed=args.seed,
        n_examples=args.n_examples,
        condition=args.condition,
        require_no_state=args.require_no_state,
    )
    print(json.dumps(measurement.as_dict(), sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--score-condition", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEEDS[0])
    parser.add_argument("--n-examples", type=int, default=DEFAULT_N_EXAMPLES)
    parser.add_argument("--condition", choices=(CONDITION_PRODUCTION, CONDITION_ARCHITECTURE_ONLY), default=CONDITION_PRODUCTION)
    parser.add_argument("--require-no-state", action="store_true")
    args = parser.parse_args(argv)

    if args.score_condition:
        return _score_condition_main(args)

    repo_root = Path(args.repo_root)
    run_experiment(
        ExperimentConfig(
            repo_root=repo_root,
            results_dir=Path(args.results_dir) if args.results_dir else repo_root / "results",
            n_examples=args.n_examples,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
