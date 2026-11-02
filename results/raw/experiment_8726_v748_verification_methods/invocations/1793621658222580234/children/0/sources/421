"""Exp 2864 HaluEval/FEVER full calibration from resolved local manifests.

This runner is intentionally narrower than a live benchmark. It calibrates the
existing verifier-energy signal against dataset-provided HaluEval and FEVER
labels, using the Exp 2863 manifest contract for filenames and checksums. No
fresh model generation happens here, so the artifact must not inherit or imply
live-model provenance.

Spec: REQ-BENCH-2864, SCENARIO-BENCH-2864.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.eval.halueval_fever_pilot import (
    PilotExample,
    bootstrap_auroc_ci,
    compute_auroc,
    default_score_example as pilot_score_example,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260522"
OUTPUT_FILENAME = "experiment_2864_halueval_fever_full_calibration_v3.json"
EXP2863_REL_PATH = Path("results/experiment_2863_eval_manifest_contract_v2.json")
DATASET_KEYS = ("halueval", "fever")
DEFAULT_RANDOM_SEED = 2864
DEFAULT_BOOTSTRAP_REPS = 500

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefix verdict based on explicit preconditions and metrics.",
    "halueval_fever_ready": "True only when Exp 2863 readiness and manifest checksums pass.",
    "full_benchmark_ready": "True only when both datasets produce AUROC and verification passes.",
    "live_model_invoked": "Always false for this dataset/verifier calibration.",
    "manifest_paths_used": "Copied from Exp 2863; plain filename aliases are never guessed.",
    "manifest_sha256_used": "Copied from Exp 2863 and verified against bytes on disk.",
    "auroc_fields": "Null when a dataset lacks both binary labels or finite scores.",
    "random_seed": "Controls deterministic bootstrap confidence intervals.",
    "reproducibility_checksum": "Hashes inputs and computed metric payload, excluding wall time.",
    "duration_s": "Measured wall-clock runtime; no sleep padding.",
}


@dataclass(frozen=True)
class CalibrationExample:
    """One local manifest row prepared for verifier-energy calibration."""

    dataset_key: str
    stable_id: str
    prompt: str
    candidate: str
    label: int
    source_name: str
    reference: str = ""

    @property
    def score_text(self) -> str:
        return f"{self.prompt}\nReference: {self.reference}\nCandidate: {self.candidate}"


@dataclass(frozen=True)
class CalibrationConfig:
    """Runtime options that make the Exp 2864 artifact reproducible."""

    repo_root: Path = REPO_ROOT
    results_dir: Path | None = None
    exp2863_path: Path | None = None
    random_seed: int = DEFAULT_RANDOM_SEED
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def output_dir(self) -> Path:
        return self.results_dir if self.results_dir is not None else self.repo_root / "results"

    def contract_path(self) -> Path:
        if self.exp2863_path is not None:
            return self.exp2863_path
        return self.repo_root / EXP2863_REL_PATH

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at


ScoreFn = Callable[[CalibrationExample], float]
AdversarialVerifier = Callable[[Path], dict[str, Any]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_exp2863(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _check(check: str, passed: bool, observed: Any) -> dict[str, Any]:
    return {"check": check, "passed": bool(passed), "observed": observed}


def _coerce_label(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value in {0, 1}:
        return value
    text = str(value).strip()
    if text in {"0", "1"}:
        return int(text)
    return None


def _candidate_from_row(row: dict[str, Any], dataset_key: str) -> str:
    if dataset_key == "fever":
        return str(row.get("claim") or row.get("candidate") or "").strip()
    return str(row.get("candidate") or row.get("claim") or "").strip()


def load_manifest_examples(path: Path, dataset_key: str) -> list[CalibrationExample]:
    """Load valid binary-label rows from an Exp 2863-resolved JSONL manifest."""

    examples: list[CalibrationExample] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            label = _coerce_label(row.get("label"))
            candidate = _candidate_from_row(row, dataset_key)
            if label is None or not candidate:
                continue
            examples.append(
                CalibrationExample(
                    dataset_key=dataset_key,
                    stable_id=str(row.get("stable_id") or f"{dataset_key}-{len(examples)}"),
                    prompt=str(row.get("prompt") or ""),
                    candidate=candidate,
                    label=label,
                    source_name=str(row.get("source_name") or ""),
                    reference=str(row.get("reference") or ""),
                )
            )
    return examples


def default_score_example(example: CalibrationExample) -> float:
    """Score a row with the existing deterministic text-verifier ensemble."""

    payload = pilot_score_example(
        PilotExample(
            dataset=example.dataset_key,
            example_id=example.stable_id,
            prompt=example.prompt,
            candidate=example.candidate,
            label=example.label,
            source=example.source_name,
            reference=example.reference,
        )
    )
    return float(payload["ensemble_energy"])


def evaluate_examples(
    examples: Sequence[CalibrationExample],
    *,
    scorer: ScoreFn,
    bootstrap_reps: int,
    seed: int,
) -> dict[str, Any]:
    """Compute AUROC only when scored examples contain both binary classes."""

    scored_labels: list[int] = []
    scores: list[float] = []
    label_counts = {str(label): count for label, count in sorted(Counter(e.label for e in examples).items())}
    score_failures = 0
    for example in examples:
        try:
            score = float(scorer(example))
        except (TypeError, ValueError):
            score_failures += 1
            continue
        if not math.isfinite(score):
            score_failures += 1
            continue
        scored_labels.append(example.label)
        scores.append(score)

    try:
        auroc = compute_auroc(scored_labels, scores)
        ci_low, ci_high = bootstrap_auroc_ci(
            scored_labels,
            scores,
            reps=bootstrap_reps,
            seed=seed,
        )
        ci95: list[float] | None = [ci_low, ci_high]
    except ValueError:
        auroc = None
        ci95 = None

    return {
        "n_examples": len(examples),
        "n_scored": len(scores),
        "score_failures": score_failures,
        "label_counts": label_counts,
        "auroc": auroc,
        "auroc_ci95": ci95,
    }


def _metric_value(metrics: dict[str, Any]) -> float | None:
    value = metrics.get("auroc")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def _ci_value(metrics: dict[str, Any]) -> list[float] | None:
    value = metrics.get("auroc_ci95")
    if not isinstance(value, list) or len(value) != 2:
        return None
    if not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
        return None
    return [float(value[0]), float(value[1])]


def _resolve_from_exp2863(
    config: CalibrationConfig,
) -> tuple[dict[str, str], dict[str, str], list[dict[str, Any]], bool]:
    payload = _load_exp2863(config.contract_path())
    paths_raw = dict(payload.get("resolved_manifest_paths") or {})
    sha_raw = dict(payload.get("resolved_manifest_sha256") or {})
    checks = [
        _check("exp2863_artifact", bool(payload), str(config.contract_path()) if payload else "missing"),
    ]
    manifest_paths: dict[str, str] = {}
    manifest_sha: dict[str, str] = {}
    for key in DATASET_KEYS:
        source_ready = bool(payload.get(f"{key}_ready"))
        checks.append(_check(f"exp2863_{key}_ready", source_ready, source_ready))
        raw_path = str(paths_raw.get(key) or "")
        resolved_path = Path(raw_path)
        if raw_path and not resolved_path.is_absolute():
            resolved_path = config.repo_root / resolved_path
        declared_sha = str(sha_raw.get(key) or "")
        actual_sha = _sha256(resolved_path) if resolved_path.is_file() else ""
        manifest_paths[key] = str(resolved_path) if raw_path else ""
        manifest_sha[key] = declared_sha
        checks.append(_check(f"manifest_file_{key}", resolved_path.is_file(), str(resolved_path)))
        checks.append(
            _check(
                f"manifest_checksum_{key}",
                bool(declared_sha and actual_sha == declared_sha),
                {"declared": declared_sha, "actual": actual_sha},
            )
        )
    ready = all(check["passed"] for check in checks)
    return manifest_paths, manifest_sha, checks, ready


def _reproducibility_checksum(payload: dict[str, Any]) -> str:
    checksum_payload = {
        "manifest_paths_used": payload["manifest_paths_used"],
        "manifest_sha256_used": payload["manifest_sha256_used"],
        "halueval_auroc": payload["halueval_auroc"],
        "fever_auroc": payload["fever_auroc"],
        "halueval_n_examples": payload["halueval_n_examples"],
        "fever_n_examples": payload["fever_n_examples"],
        "auroc_ci95_by_dataset": payload["auroc_ci95_by_dataset"],
        "label_counts_by_dataset": payload["label_counts_by_dataset"],
        "random_seed": payload["random_seed"],
        "run_date": payload["run_date"],
    }
    encoded = json.dumps(checksum_payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _base_artifact(
    *,
    config: CalibrationConfig,
    duration_s: float,
    manifest_paths: dict[str, str],
    manifest_sha: dict[str, str],
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "honest_verdict": "blocked_eval_manifest_contract",
        "halueval_fever_ready": False,
        "full_benchmark_ready": False,
        "live_model_invoked": False,
        "manifest_paths_used": manifest_paths,
        "manifest_sha256_used": manifest_sha,
        "halueval_auroc": None,
        "fever_auroc": None,
        "halueval_n_examples": 0,
        "fever_n_examples": 0,
        "auroc_ci95_by_dataset": {},
        "label_counts_by_dataset": {},
        "random_seed": config.random_seed,
        "reproducibility_checksum": "",
        "preconditions_checked": checks,
        "adversarial_verify_passed": False,
        "adversarial_verify_flags": [],
        "field_principles": dict(FIELD_PRINCIPLES),
        "run_date": RUN_DATE,
        "duration_s": max(0.0, duration_s),
    }


def _apply_verdict(artifact: dict[str, Any], metrics_ready: bool) -> None:
    if not artifact["halueval_fever_ready"]:
        artifact["honest_verdict"] = "blocked_eval_manifest_contract"
    elif not metrics_ready:
        artifact["honest_verdict"] = "blocked_unavailable_auroc"
    elif not artifact["adversarial_verify_passed"]:
        artifact["honest_verdict"] = "blocked_adversarial_verify"
    else:
        artifact["honest_verdict"] = "complete: HaluEval/FEVER local calibration ready"
    artifact["full_benchmark_ready"] = bool(
        artifact["halueval_fever_ready"]
        and metrics_ready
        and artifact["adversarial_verify_passed"]
    )


def write_artifact(results_dir: Path, artifact: dict[str, Any]) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / OUTPUT_FILENAME
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def _real_adversarial_verify(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "passed": False,
            "flags": [
                {
                    "kind": "ARTIFACT_MISSING",
                    "severity": "critical",
                    "detail": f"artifact missing: {path}",
                }
            ],
        }
    script = REPO_ROOT / "scripts" / "adversarial_verify.py"
    if not script.is_file():
        return {
            "passed": False,
            "flags": [
                {
                    "kind": "ADVERSARIAL_VERIFY_UNAVAILABLE",
                    "severity": "warn",
                    "detail": str(script),
                }
            ],
        }
    proc = subprocess.run(
        [sys.executable, str(script), "--json", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {
            "passed": False,
            "flags": [
                {
                    "kind": "ADVERSARIAL_VERIFY_ERROR",
                    "severity": "critical",
                    "detail": (proc.stderr or proc.stdout or f"returncode={proc.returncode}").strip(),
                }
            ],
        }
    reports = list(payload.get("reports") or [])
    report = dict(reports[0]) if reports else {}
    flags = list(report.get("flags") or [])
    return {"passed": proc.returncode == 0 and not flags, "flags": flags}


def run_calibration(
    config: CalibrationConfig | None = None,
    *,
    scorer: ScoreFn = default_score_example,
    adversarial_verifier: AdversarialVerifier | None = _real_adversarial_verify,
    write: bool = True,
) -> dict[str, Any]:
    """Run Exp 2864 and return the exact artifact payload that was written."""

    config = config or CalibrationConfig()
    started = config.start_time()
    manifest_paths, manifest_sha, checks, ready = _resolve_from_exp2863(config)
    artifact = _base_artifact(
        config=config,
        duration_s=config.clock() - started,
        manifest_paths=manifest_paths,
        manifest_sha=manifest_sha,
        checks=checks,
    )
    metrics_ready = False
    if ready:
        metrics: dict[str, dict[str, Any]] = {}
        for key in DATASET_KEYS:
            examples = load_manifest_examples(Path(manifest_paths[key]), key)
            metrics[key] = evaluate_examples(
                examples,
                scorer=scorer,
                bootstrap_reps=config.bootstrap_reps,
                seed=config.random_seed,
            )
        artifact.update(
            {
                "halueval_fever_ready": True,
                "halueval_auroc": _metric_value(metrics["halueval"]),
                "fever_auroc": _metric_value(metrics["fever"]),
                "halueval_n_examples": int(metrics["halueval"]["n_examples"]),
                "fever_n_examples": int(metrics["fever"]["n_examples"]),
                "auroc_ci95_by_dataset": {
                    key: _ci_value(metrics[key]) for key in DATASET_KEYS
                },
                "label_counts_by_dataset": {
                    key: metrics[key]["label_counts"] for key in DATASET_KEYS
                },
            }
        )
        metrics_ready = all(artifact[f"{key}_auroc"] is not None for key in DATASET_KEYS)

    artifact["duration_s"] = max(0.0, config.clock() - started)
    artifact["reproducibility_checksum"] = _reproducibility_checksum(artifact)
    _apply_verdict(artifact, metrics_ready)

    output_path: Path | None = None
    if write:
        output_path = write_artifact(config.output_dir(), artifact)
    if write and adversarial_verifier is not None and output_path is not None:
        verify_result = adversarial_verifier(output_path)
        artifact["adversarial_verify_passed"] = bool(verify_result.get("passed"))
        artifact["adversarial_verify_flags"] = list(verify_result.get("flags") or [])
        _apply_verdict(artifact, metrics_ready)
        artifact["duration_s"] = max(0.0, config.clock() - started)
        write_artifact(config.output_dir(), artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--no-adversarial-verify", action="store_true")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root)
    run_calibration(
        CalibrationConfig(
            repo_root=repo_root,
            results_dir=Path(args.results_dir) if args.results_dir else repo_root / "results",
            random_seed=args.random_seed,
            bootstrap_reps=args.bootstrap_reps,
        ),
        adversarial_verifier=None if args.no_adversarial_verify else _real_adversarial_verify,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
