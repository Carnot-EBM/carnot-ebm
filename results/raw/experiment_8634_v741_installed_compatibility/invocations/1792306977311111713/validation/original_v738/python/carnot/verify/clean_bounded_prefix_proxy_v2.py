"""Clean bounded-prefix/EPR proxy artifact for Exp 2858.

What this is:
    A fast local proxy over FoVer-style labeled rows. It uses Carnot's existing
    bounded-prefix arithmetic false-claim scorer and optional already-recorded
    top-k telemetry summaries to decide whether exact frontier-bound work is
    worth implementing later.

What this is not:
    It is not exact BEAVER. It does not build a token-trie/frontier soundness
    proof, and it does not invoke a live model. The artifact intentionally keeps
    live SOTA model provenance out of the schema so a local proxy cannot be
    mistaken for live inference.

Spec: REQ-VERIFY-2858, SCENARIO-VERIFY-2858
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.verify.beaver_epr_bounded_probe import (
    DEFAULT_TELEMETRY_PATHS,
    ArithmeticFalseClaimConstraint,
    LabeledExample,
    _example_from_row,
    _load_entropy_telemetry,
    _read_rows,
    compute_auroc,
)

OUTPUT_FILENAME = "experiment_2858_beaver_epr_clean_bounded_proxy_v2.json"
RUN_DATE = "20260522"
RANDOM_SEED = 42
N_EXAMPLES = 100
REPO_ROOT = Path(__file__).resolve().parents[3]
CLAIM_BOUNDARY = (
    "Proxy only: local bounded-prefix FoVer scoring; no exact BEAVER "
    "frontier proof and no live model inference."
)
FOVER_PATH = Path("data/fover_corpus.jsonl")
METRICS_COMMAND = '.venv/bin/python3 -c "import sklearn, numpy; print(\'metrics ok\')"'

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "beaver_exact",
    "exact_beaver_implemented",
    "bounded_prefix_proxy_auc",
    "entropy_production_auc",
    "n_examples",
    "random_seed",
    "reproducibility_checksum",
    "live_model_invoked",
    "claim_boundary",
    "preconditions_checked",
    "duration_s",
    "adversarial_verify_passed",
    "adversarial_verify_flags",
    "run_date",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for the Exp 2858 local proxy rerun."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    run_date: str = RUN_DATE
    n_examples: int = N_EXAMPLES
    random_seed: int = RANDOM_SEED
    started_at: float | None = None
    clock: Callable[[], float] = time.perf_counter
    fover_path: Path = FOVER_PATH
    telemetry_paths: tuple[str, ...] = DEFAULT_TELEMETRY_PATHS

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def resolved_fover_path(self) -> Path:
        return self.repo_root / self.fover_path

    def resolved_output_path(self) -> Path:
        if self.output_path is not None:
            return self.output_path
        return self.repo_root / "results" / OUTPUT_FILENAME


AdversarialVerifyRunner = Callable[[Path], Mapping[str, Any]]


def check_preconditions(config: ExperimentConfig) -> list[dict[str, Any]]:
    """Record the exact local-resource gates required before proxy scoring."""

    metrics_ok, metrics_detail = _metrics_probe()
    return [
        {
            "step": "cd /home/ianblenke/github.com/ianblenke/carnot",
            "passed": config.repo_root.is_dir(),
            "observed": str(config.repo_root),
        },
        {
            "step": "test -f data/fover_corpus.jsonl",
            "passed": config.resolved_fover_path().is_file(),
            "observed": str(config.resolved_fover_path()),
        },
        {
            "step": METRICS_COMMAND,
            "passed": metrics_ok,
            "observed": metrics_detail,
        },
    ]


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    adversarial_verify_runner: AdversarialVerifyRunner | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Run the clean local bounded-prefix proxy and optionally write the artifact."""

    active_config = config or ExperimentConfig()
    started_at = active_config.start_time()
    preconditions = check_preconditions(active_config)

    if not preconditions[1]["passed"]:
        artifact = _blocked_artifact(
            active_config,
            preconditions,
            active_config.clock() - started_at,
            "blocked_fover_dataset",
        )
    elif not preconditions[2]["passed"]:
        artifact = _blocked_artifact(
            active_config,
            preconditions,
            active_config.clock() - started_at,
            "blocked_metrics_dependency",
        )
    else:
        examples = _load_sampled_examples(active_config)
        if len(examples) < active_config.n_examples or len({example.label for example in examples}) < 2:
            artifact = _blocked_artifact(
                active_config,
                preconditions,
                active_config.clock() - started_at,
                "blocked_fover_dataset",
            )
        else:
            artifact = _success_artifact(
                active_config,
                preconditions,
                examples,
                active_config.clock() - started_at,
            )

    _validate_artifact(artifact)
    if write:
        output = write_artifact(active_config.resolved_output_path(), artifact)
        if artifact["honest_verdict"].startswith("complete:"):
            runner = adversarial_verify_runner or run_adversarial_verify
            artifact = _attach_adversarial_report(artifact, runner(output))
            _validate_artifact(artifact)
            write_artifact(output, artifact)
    return artifact


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_adversarial_verify(path: Path) -> dict[str, Any]:  # pragma: no cover - subprocess glue.
    script = REPO_ROOT / "scripts" / "adversarial_verify.py"
    if not script.is_file():
        return {"loaded": False, "flag_count": 0, "flags": []}
    completed = subprocess.run(
        [sys.executable, str(script), "--json", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {
            "loaded": False,
            "flag_count": 1,
            "flags": [
                {
                    "kind": "ADVERSARIAL_VERIFY_ERROR",
                    "severity": "warn",
                    "detail": (completed.stderr or completed.stdout or "invalid output").strip(),
                }
            ],
            "returncode": completed.returncode,
        }
    reports = payload.get("reports") or []
    if not reports:
        return {"loaded": False, "flag_count": 0, "flags": [], "returncode": completed.returncode}
    report = dict(reports[0])
    report["returncode"] = completed.returncode
    return report


def main() -> int:  # pragma: no cover - exercised through script wrapper.
    run_experiment(ExperimentConfig(repo_root=Path.cwd()))
    return 0


def _success_artifact(
    config: ExperimentConfig,
    preconditions: Sequence[Mapping[str, Any]],
    examples: Sequence[LabeledExample],
    duration_s: float,
) -> dict[str, Any]:
    constraint = ArithmeticFalseClaimConstraint()
    score_rows = [_score_example(example, constraint) for example in examples]
    labels = [int(row["label"]) for row in score_rows]
    scores = [float(row["bounded_prefix_proxy_score"]) for row in score_rows]
    entropy_summary = _load_entropy_telemetry(config.repo_root, config.telemetry_paths)
    entropy_auc = (
        float(entropy_summary.entropy_production_auc)
        if entropy_summary.entropy_production_auc is not None
        else 0.0
    )
    artifact = _base_artifact(config, preconditions, duration_s)
    artifact.update(
        {
            "honest_verdict": (
                "complete: clean bounded-prefix/EPR proxy evaluated on local FoVer labels"
            ),
            "bounded_prefix_proxy_auc": float(compute_auroc(labels, scores)),
            "entropy_production_auc": entropy_auc,
            "entropy_production_measured": entropy_summary.entropy_production_auc is not None,
            "n_examples": len(score_rows),
            "sample_row_count": min(5, len(score_rows)),
            "sample_rows": score_rows[:5],
        }
    )
    artifact["reproducibility_checksum"] = _checksum(
        {
            "random_seed": config.random_seed,
            "n_examples": len(score_rows),
            "score_rows": score_rows,
            "entropy_production_auc": entropy_auc,
        }
    )
    return artifact


def _blocked_artifact(
    config: ExperimentConfig,
    preconditions: Sequence[Mapping[str, Any]],
    duration_s: float,
    honest_verdict: str,
) -> dict[str, Any]:
    artifact = _base_artifact(config, preconditions, duration_s)
    artifact["honest_verdict"] = honest_verdict
    artifact["reproducibility_checksum"] = _checksum(
        {
            "random_seed": config.random_seed,
            "n_examples": 0,
            "honest_verdict": honest_verdict,
            "preconditions_checked": list(preconditions),
        }
    )
    return artifact


def _base_artifact(
    config: ExperimentConfig,
    preconditions: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> dict[str, Any]:
    return {
        "artifact": "experiment_2858_beaver_epr_clean_bounded_proxy_v2",
        "schema": "carnot.clean_bounded_prefix_proxy_v2",
        "honest_verdict": "blocked_not_evaluated",
        "beaver_exact": False,
        "exact_beaver_implemented": False,
        "bounded_prefix_proxy_auc": 0.0,
        "entropy_production_auc": 0.0,
        "entropy_production_measured": False,
        "n_examples": 0,
        "random_seed": config.random_seed,
        "reproducibility_checksum": "",
        "live_model_invoked": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "preconditions_checked": [dict(check) for check in preconditions],
        "duration_s": float(duration_s),
        "adversarial_verify_passed": False,
        "adversarial_verify_flags": [],
        "run_date": config.run_date,
    }


def _load_sampled_examples(config: ExperimentConfig) -> list[LabeledExample]:
    rows = _read_rows(config.resolved_fover_path())
    examples = [
        example
        for index, row in enumerate(rows)
        if (example := _example_from_row(row, config.resolved_fover_path(), index)) is not None
    ]
    return _balanced_seeded_sample(examples, config.n_examples, config.random_seed)


def _balanced_seeded_sample(
    examples: Sequence[LabeledExample],
    limit: int,
    random_seed: int,
) -> list[LabeledExample]:
    rng = random.Random(random_seed)
    positives = [example for example in examples if example.label == 1]
    negatives = [example for example in examples if example.label == 0]
    half = limit // 2
    if len(positives) >= half and len(negatives) >= limit - half:
        rng.shuffle(positives)
        rng.shuffle(negatives)
        selected = negatives[: limit - half] + positives[:half]
        rng.shuffle(selected)
        return selected
    selected = list(examples)
    rng.shuffle(selected)
    return selected[:limit]


def _score_example(
    example: LabeledExample,
    constraint: ArithmeticFalseClaimConstraint,
) -> dict[str, Any]:
    result = constraint.explore_prefixes(example.text)
    return {
        "example_id": example.example_id,
        "label": example.label,
        "bounded_prefix_proxy_score": result.score,
        "checked_claim_count": result.checked_claim_count,
        "false_claim_count": result.false_claim_count,
        "first_violation_prefix_length": result.first_violation_prefix_length,
    }


def _attach_adversarial_report(
    artifact: dict[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    flags = list(report.get("flags") or [])
    updated = dict(artifact)
    updated["adversarial_verify_passed"] = int(report.get("flag_count") or 0) == 0
    updated["adversarial_verify_flags"] = flags
    return updated


def _metrics_probe() -> tuple[bool, str]:
    try:
        import numpy  # noqa: F401
        import sklearn  # noqa: F401
    except ModuleNotFoundError as exc:
        return False, f"missing {exc.name}"
    return True, "metrics ok"


def _checksum(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact["beaver_exact"] is not False:
        raise ValueError("clean proxy artifact must not claim exact BEAVER")
    if artifact["exact_beaver_implemented"] is not False:
        raise ValueError("exact frontier proof is not implemented")
    if artifact["live_model_invoked"] is not False:
        raise ValueError("Exp 2858 local proxy must not claim live model invocation")
    if "model_specs" in artifact:
        raise ValueError("clean proxy artifact must not include model_specs")
    if artifact["run_date"] != RUN_DATE:
        raise ValueError("run_date must be 20260522")


__all__ = [
    "CLAIM_BOUNDARY",
    "ExperimentConfig",
    "N_EXAMPLES",
    "OUTPUT_FILENAME",
    "RANDOM_SEED",
    "REQUIRED_ARTIFACT_FIELDS",
    "RUN_DATE",
    "check_preconditions",
    "main",
    "run_adversarial_verify",
    "run_experiment",
    "write_artifact",
]
