"""Exp 3918 classifier-first cascade router prototype.

The cascade uses the cheap energy verifier score for every item and escalates
only energy-margin close calls to the cached Exp 3917 LLM-judge score. This is
an aggregation-only experiment: it reads the already measured Exp 3917 per-item
scores and costs, tunes the escalation band on calibration rows, and reports
heldout AUROC and cost savings without live inference.

Spec refs: REQ-VERIFY-3918, SCENARIO-VERIFY-3918,
SCENARIO-VERIFY-3918-BLOCKED.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from math import inf
from pathlib import Path
from typing import Any

from carnot.verify.cost_instrumented_verification import _auroc


OUTPUT_REL_PATH = Path("results/experiment_3918_cascade_router_prototype.json")
EXP3917_ARTIFACT_REL_PATH = Path("results/experiment_3917_efficiency_head_to_head.json")
TITLE = "cascade_router_prototype"
EXPERIMENT_ID = 3918
DEFAULT_RANDOM_SEED = 3918
DEFAULT_ENERGY_THRESHOLD = 0.5
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts:no_live_model_reuses_exp3917_scores"

REQUIRED_FIELDS = {
    "cascade_auroc",
    "pure_llm_auroc",
    "escalation_fraction",
    "cascade_cost_ratio",
    "auroc_gap",
    "band_tuned_on_calibration",
    "n_calibration",
    "n_heldout",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "inference_substrate",
    "honest_verdict",
}

FIELD_PRINCIPLES: dict[str, str] = {
    "cascade_auroc": (
        "Accuracy of the classifier-first cascade - should approach pure-LLM AUROC."
    ),
    "pure_llm_auroc": "The pure-LLM-judge baseline the cascade must (near-)match.",
    "escalation_fraction": (
        "Fraction of items escalated to the LLM - the cost driver; low => big savings."
    ),
    "cascade_cost_ratio": (
        "BARE FLOAT - pure_llm_cost / cascade_cost; the deployable 'Nx cheaper at matched accuracy' number."
    ),
    "auroc_gap": "pure_llm_auroc - cascade_auroc; small (<0.02) => matched accuracy.",
    "band_tuned_on_calibration": (
        "Methodology - band tuned on calibration, evaluated held-out; no leakage."
    ),
    "n_calibration": (
        "Methodology - calibration split size used only for band tuning."
    ),
    "n_heldout": "Methodology - heldout split size used for reported metrics.",
    "preconditions_checked": (
        "Methodology - cached Exp 3917 score and cost evidence checked before aggregation."
    ),
    "random_seed": "Methodology - deterministic split and reproducibility seed.",
    "reproducibility_checksum": (
        "Methodology - hash over upstream artifact, split, tuned band, costs, and heldout scores."
    ),
    "duration_s": "Methodology - wall-clock aggregation duration.",
    "inference_substrate": (
        "Methodology - aggregation substrate; reuses exp3917 per-item LLM scores so no double-inference."
    ),
}

WRAPPED_VALUE_FORBIDDEN_FIELDS = (
    "cascade_auroc",
    "pure_llm_auroc",
    "escalation_fraction",
    "cascade_cost_ratio",
    "auroc_gap",
    "band_tuned_on_calibration",
    "n_calibration",
    "n_heldout",
    "duration_s",
)


@dataclass(frozen=True)
class PreconditionCheck:
    """One cached-evidence check run before Exp 3918 aggregation."""

    resource: str
    available: bool
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {"resource": self.resource, "available": self.available, "detail": self.detail}


@dataclass(frozen=True)
class ScoreRow:
    """One aligned Exp 3917 row with both verifier scores."""

    index: int
    gold_error: int
    energy_score: float
    llm_score: float
    corpus_source: str
    source_index: int

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "gold_error": self.gold_error,
            "energy_score": self.energy_score,
            "llm_score": self.llm_score,
            "corpus_source": self.corpus_source,
            "source_index": self.source_index,
        }


@dataclass(frozen=True)
class CascadeEvidence:
    """Cached Exp 3917 rows and costs used by the cascade."""

    rows: tuple[ScoreRow, ...]
    energy_per_item_ms: float
    llm_per_item_ms: float
    cost_ratio_walltime: float
    artifact_path: Path
    artifact_sha256: str
    upstream_artifact: dict[str, object]


@dataclass(frozen=True)
class CascadeConfig:
    """Runtime configuration for Exp 3918."""

    repo_root: Path
    output_path: Path | None = None
    random_seed: int = DEFAULT_RANDOM_SEED
    calibration_fraction: float = 0.5
    energy_threshold: float = DEFAULT_ENERGY_THRESHOLD
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def resolved_output_path(self) -> Path:
        return self.output_path if self.output_path is not None else self.repo_root / OUTPUT_REL_PATH

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at


def _checksum(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_to_repo(repo_root: Path, path: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:  # pragma: no cover - provenance for unusual absolute output paths.
        return path.as_posix()


def _require_float(payload: dict[str, Any], key: str) -> float:
    if key not in payload:
        raise ValueError(f"missing {key}")
    value = float(payload[key])
    if value <= 0.0:
        raise ValueError(f"{key} must be positive")
    return value


def _score_row(raw: dict[str, Any], fallback_index: int) -> ScoreRow:
    missing = [
        key
        for key in ("gold_error", "energy_score", "llm_judge_score")
        if key not in raw
    ]
    if missing:
        raise ValueError(f"cached row {fallback_index} missing {', '.join(missing)}")
    label = int(raw["gold_error"])
    if label not in (0, 1):
        raise ValueError(f"cached row {fallback_index} has non-binary gold_error")
    return ScoreRow(
        index=int(raw.get("index", fallback_index)),
        gold_error=label,
        energy_score=float(raw["energy_score"]),
        llm_score=float(raw["llm_judge_score"]),
        corpus_source=str(raw.get("corpus_source", "exp3917")),
        source_index=int(raw.get("source_index", fallback_index)),
    )


def load_exp3917_evidence(repo_root: Path) -> CascadeEvidence:
    """Load and validate cached Exp 3917 scores and measured cost fields."""

    artifact_path = repo_root / EXP3917_ARTIFACT_REL_PATH
    if not artifact_path.is_file():
        raise FileNotFoundError(EXP3917_ARTIFACT_REL_PATH.as_posix())
    artifact = _read_json(artifact_path)
    if not isinstance(artifact, dict):
        raise ValueError("exp3917 artifact is not a JSON object")
    raw_rows = artifact.get("per_item_results")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("exp3917 artifact has no per_item_results")
    rows = tuple(_score_row(dict(raw), index) for index, raw in enumerate(raw_rows))
    labels = [row.gold_error for row in rows]
    if set(labels) != {0, 1}:
        raise ValueError("exp3917 rows must contain both gold classes")
    return CascadeEvidence(
        rows=rows,
        energy_per_item_ms=_require_float(artifact, "energy_per_item_ms"),
        llm_per_item_ms=_require_float(artifact, "llm_per_item_ms"),
        cost_ratio_walltime=_require_float(artifact, "cost_ratio_walltime"),
        artifact_path=artifact_path,
        artifact_sha256=_sha256_file(artifact_path),
        upstream_artifact=artifact,
    )


def probe_preconditions(
    repo_root: Path,
) -> tuple[tuple[PreconditionCheck, ...], str | None, CascadeEvidence | None]:
    """Check the only hard resource for Exp 3918: cached Exp 3917 evidence."""

    try:
        evidence = load_exp3917_evidence(repo_root)
    except Exception as exc:
        return (
            (
                PreconditionCheck(
                    "exp3917_cached_scores_ready",
                    False,
                    repr(exc),
                ),
            ),
            "blocked_upstream_efficiency_missing",
            None,
        )
    return (
        (
            PreconditionCheck(
                "exp3917_cached_scores_ready",
                True,
                f"rows={len(evidence.rows)} ratio={evidence.cost_ratio_walltime:.6f}",
            ),
        ),
        None,
        evidence,
    )


def split_rows(
    rows: Sequence[ScoreRow],
    *,
    random_seed: int,
    calibration_fraction: float,
) -> tuple[tuple[ScoreRow, ...], tuple[ScoreRow, ...]]:
    """Create a deterministic stratified calibration/heldout split."""

    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1")
    by_label = {0: [row for row in rows if row.gold_error == 0], 1: [row for row in rows if row.gold_error == 1]}
    if not by_label[0] or not by_label[1]:
        raise ValueError("split requires both labels")
    rng = random.Random(random_seed)
    calibration: list[ScoreRow] = []
    heldout: list[ScoreRow] = []
    for label_rows in by_label.values():
        shuffled = list(label_rows)
        rng.shuffle(shuffled)
        n_calibration = max(1, min(len(shuffled) - 1, round(len(shuffled) * calibration_fraction)))
        calibration.extend(shuffled[:n_calibration])
        heldout.extend(shuffled[n_calibration:])
    calibration.sort(key=lambda row: row.index)
    heldout.sort(key=lambda row: row.index)
    return tuple(calibration), tuple(heldout)


def _labels(rows: Sequence[ScoreRow]) -> tuple[int, ...]:
    return tuple(row.gold_error for row in rows)


def _llm_scores(rows: Sequence[ScoreRow]) -> tuple[float, ...]:
    return tuple(row.llm_score for row in rows)


def _candidate_bands(rows: Sequence[ScoreRow], threshold: float) -> tuple[float, ...]:
    margins = sorted({abs(row.energy_score - threshold) for row in rows})
    bands = [0.0]
    bands.extend(float(margin) + 1e-12 for margin in margins)
    return tuple(dict.fromkeys(bands))


def apply_cascade_scores(
    rows: Sequence[ScoreRow],
    *,
    threshold: float,
    band: float,
) -> tuple[tuple[float, ...], tuple[bool, ...]]:
    """Return final cascade scores and escalation flags for a fixed band."""

    scores: list[float] = []
    escalated: list[bool] = []
    for row in rows:
        should_escalate = abs(row.energy_score - threshold) < band
        escalated.append(should_escalate)
        scores.append(row.llm_score if should_escalate else row.energy_score)
    return tuple(scores), tuple(escalated)


def _cost_ratio(
    *,
    n_items: int,
    n_escalated: int,
    energy_per_item_ms: float,
    llm_per_item_ms: float,
) -> float:
    if n_items <= 0:
        raise ValueError("cost ratio requires at least one item")
    pure_llm_cost = n_items * llm_per_item_ms
    cascade_cost = (n_items * energy_per_item_ms) + (n_escalated * llm_per_item_ms)
    if cascade_cost <= 0.0:
        raise ValueError("cascade cost must be positive")
    return pure_llm_cost / cascade_cost


def evaluate_cascade(
    rows: Sequence[ScoreRow],
    *,
    threshold: float,
    band: float,
    energy_per_item_ms: float,
    llm_per_item_ms: float,
) -> dict[str, float]:
    """Evaluate AUROC, escalation, and cost for a fixed cascade band."""

    scores, escalated = apply_cascade_scores(rows, threshold=threshold, band=band)
    labels = _labels(rows)
    pure_llm_auroc = _auroc(labels, _llm_scores(rows))
    cascade_auroc = _auroc(labels, scores)
    escalation_fraction = sum(escalated) / len(rows)
    cascade_cost_ratio = _cost_ratio(
        n_items=len(rows),
        n_escalated=sum(escalated),
        energy_per_item_ms=energy_per_item_ms,
        llm_per_item_ms=llm_per_item_ms,
    )
    return {
        "cascade_auroc": float(cascade_auroc),
        "pure_llm_auroc": float(pure_llm_auroc),
        "auroc_gap": float(pure_llm_auroc - cascade_auroc),
        "escalation_fraction": float(escalation_fraction),
        "cascade_cost_ratio": float(cascade_cost_ratio),
        "n_escalated": float(sum(escalated)),
    }


def tune_band(
    rows: Sequence[ScoreRow],
    *,
    threshold: float,
    energy_per_item_ms: float,
    llm_per_item_ms: float,
) -> dict[str, float]:
    """Tune the escalation band on calibration rows only."""

    best_band = 0.0
    best_metrics: dict[str, float] | None = None
    best_key: tuple[float, float, float, float] = (inf, inf, inf, inf)
    for band in _candidate_bands(rows, threshold):
        metrics = evaluate_cascade(
            rows,
            threshold=threshold,
            band=band,
            energy_per_item_ms=energy_per_item_ms,
            llm_per_item_ms=llm_per_item_ms,
        )
        feasible = metrics["auroc_gap"] < 0.02 and metrics["cascade_cost_ratio"] > 3.0
        if feasible:
            key = (0.0, -metrics["cascade_cost_ratio"], -metrics["cascade_auroc"], band)
        else:
            key = (1.0, max(0.0, metrics["auroc_gap"]), -metrics["cascade_cost_ratio"], band)
        if key < best_key:
            best_key = key
            best_band = band
            best_metrics = metrics
    if best_metrics is None:  # pragma: no cover - candidate generation always yields band 0.0.
        raise ValueError("no calibration bands were evaluated")
    return {"band": float(best_band), **best_metrics}


def _classify_verdict(*, auroc_gap: float, cascade_cost_ratio: float, escalation_fraction: float) -> str:
    gap = f"{auroc_gap:.4f}"
    ratio = f"{cascade_cost_ratio:.2f}"
    esc = f"{escalation_fraction:.4f}"
    if auroc_gap < 0.02 and cascade_cost_ratio > 3.0:
        return (
            "complete: "
            f"cascade_router_WINS_gap{gap}_{ratio}x_cheaper_at_matched_accuracy_escfrac{esc}"
        )
    return f"complete: cascade_router_MARGINAL_gap{gap}_ratio{ratio}_escfrac{esc}"


def _source_artifacts(config: CascadeConfig, evidence: CascadeEvidence) -> dict[str, object]:
    return {
        "exp3917": {
            "path": _relative_to_repo(config.repo_root, evidence.artifact_path),
            "sha256": evidence.artifact_sha256,
            "cost_ratio_walltime": evidence.cost_ratio_walltime,
            "n_items": len(evidence.rows),
        }
    }


def build_complete_artifact(
    *,
    config: CascadeConfig,
    evidence: CascadeEvidence,
    preconditions_checked: Sequence[PreconditionCheck],
    started_at: float,
) -> dict[str, object]:
    """Build the terminal heldout Exp 3918 cascade artifact."""

    calibration, heldout = split_rows(
        evidence.rows,
        random_seed=config.random_seed,
        calibration_fraction=config.calibration_fraction,
    )
    tuned = tune_band(
        calibration,
        threshold=config.energy_threshold,
        energy_per_item_ms=evidence.energy_per_item_ms,
        llm_per_item_ms=evidence.llm_per_item_ms,
    )
    heldout_metrics = evaluate_cascade(
        heldout,
        threshold=config.energy_threshold,
        band=tuned["band"],
        energy_per_item_ms=evidence.energy_per_item_ms,
        llm_per_item_ms=evidence.llm_per_item_ms,
    )
    finished_at = config.clock()
    verdict = _classify_verdict(
        auroc_gap=heldout_metrics["auroc_gap"],
        cascade_cost_ratio=heldout_metrics["cascade_cost_ratio"],
        escalation_fraction=heldout_metrics["escalation_fraction"],
    )
    checksum_payload = {
        "experiment": EXPERIMENT_ID,
        "exp3917_sha256": evidence.artifact_sha256,
        "random_seed": config.random_seed,
        "energy_threshold": config.energy_threshold,
        "band": tuned["band"],
        "calibration_indices": [row.index for row in calibration],
        "heldout_indices": [row.index for row in heldout],
        "heldout_metrics": heldout_metrics,
        "calibration_metrics": tuned,
        "costs": {
            "energy_per_item_ms": evidence.energy_per_item_ms,
            "llm_per_item_ms": evidence.llm_per_item_ms,
        },
    }
    artifact: dict[str, object] = {
        "experiment": EXPERIMENT_ID,
        "title": TITLE,
        "run_date": datetime.fromtimestamp(finished_at, tz=UTC).strftime("%Y%m%d"),
        "started_at": datetime.fromtimestamp(started_at, tz=UTC).isoformat().replace("+00:00", "Z"),
        "finished_at": datetime.fromtimestamp(finished_at, tz=UTC).isoformat().replace("+00:00", "Z"),
        "honest_verdict": verdict,
        "status": verdict,
        "cascade_auroc": heldout_metrics["cascade_auroc"],
        "pure_llm_auroc": heldout_metrics["pure_llm_auroc"],
        "escalation_fraction": heldout_metrics["escalation_fraction"],
        "cascade_cost_ratio": heldout_metrics["cascade_cost_ratio"],
        "auroc_gap": heldout_metrics["auroc_gap"],
        "band_tuned_on_calibration": tuned["band"],
        "energy_threshold": config.energy_threshold,
        "n_calibration": len(calibration),
        "n_heldout": len(heldout),
        "n_escalated_heldout": int(heldout_metrics["n_escalated"]),
        "preconditions_checked": [check.as_dict() for check in preconditions_checked],
        "random_seed": config.random_seed,
        "random_seeds_used": {"calibration_split": config.random_seed},
        "reproducibility_checksum": _checksum(checksum_payload),
        "duration_s": finished_at - started_at,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "no_new_inference": True,
        "frozen_fover_auroc_unchanged": 0.9131,
        "source_artifacts": _source_artifacts(config, evidence),
        "calibration_metrics": tuned,
        "heldout_metrics": heldout_metrics,
        "cost_accounting": {
            "energy_per_item_ms": evidence.energy_per_item_ms,
            "llm_per_item_ms": evidence.llm_per_item_ms,
            "pure_llm_cost_ms": len(heldout) * evidence.llm_per_item_ms,
            "cascade_cost_ms": (len(heldout) * evidence.energy_per_item_ms)
            + (int(heldout_metrics["n_escalated"]) * evidence.llm_per_item_ms),
        },
        "field_principles": FIELD_PRINCIPLES,
    }
    validate_artifact(artifact)
    return artifact


def build_blocked_artifact(
    *,
    config: CascadeConfig,
    reason: str,
    preconditions_checked: Sequence[PreconditionCheck],
    started_at: float,
) -> dict[str, object]:
    """Build a blocked artifact without fabricated cascade metrics."""

    finished_at = config.clock()
    artifact: dict[str, object] = {
        "experiment": EXPERIMENT_ID,
        "title": TITLE,
        "run_date": datetime.fromtimestamp(finished_at, tz=UTC).strftime("%Y%m%d"),
        "started_at": datetime.fromtimestamp(started_at, tz=UTC).isoformat().replace("+00:00", "Z"),
        "finished_at": datetime.fromtimestamp(finished_at, tz=UTC).isoformat().replace("+00:00", "Z"),
        "honest_verdict": reason,
        "status": reason,
        "cascade_auroc": None,
        "pure_llm_auroc": None,
        "escalation_fraction": None,
        "cascade_cost_ratio": None,
        "auroc_gap": None,
        "band_tuned_on_calibration": None,
        "energy_threshold": config.energy_threshold,
        "n_calibration": 0,
        "n_heldout": 0,
        "n_escalated_heldout": 0,
        "preconditions_checked": [check.as_dict() for check in preconditions_checked],
        "random_seed": config.random_seed,
        "random_seeds_used": {},
        "reproducibility_checksum": _checksum(
            {
                "experiment": EXPERIMENT_ID,
                "reason": reason,
                "preconditions_checked": [check.as_dict() for check in preconditions_checked],
                "random_seed": config.random_seed,
            }
        ),
        "duration_s": finished_at - started_at,
        "inference_substrate": "none_blocked_preflight",
        "no_new_inference": True,
        "frozen_fover_auroc_unchanged": 0.9131,
        "source_artifacts": {},
        "calibration_metrics": {},
        "heldout_metrics": {},
        "cost_accounting": {},
        "field_principles": FIELD_PRINCIPLES,
    }
    validate_artifact(artifact)
    return artifact


def _terminal_verdict(verdict: str) -> bool:
    return verdict.startswith(("complete:", "blocked_"))


def validate_artifact(artifact: dict[str, object]) -> None:
    """Validate required Exp 3918 fields and bare-scalar discipline."""

    missing = sorted(REQUIRED_FIELDS - artifact.keys())
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = str(artifact["honest_verdict"])
    if not _terminal_verdict(verdict):
        raise ValueError(f"honest_verdict lacks terminal prefix: {verdict}")
    for key in WRAPPED_VALUE_FORBIDDEN_FIELDS:
        if isinstance(artifact.get(key), dict):
            raise ValueError(f"{key} must not be a value/principle wrapper")
    if not isinstance(artifact["duration_s"], (int, float)):
        raise ValueError("duration_s must be a bare number")
    if len(str(artifact["reproducibility_checksum"])) != 64:
        raise ValueError("reproducibility_checksum must be a sha256 hex string")
    if verdict.startswith("blocked_"):
        if artifact["n_calibration"] != 0 or artifact["n_heldout"] != 0:
            raise ValueError("blocked artifacts must have zero split counts")
        return
    for key in (
        "cascade_auroc",
        "pure_llm_auroc",
        "escalation_fraction",
        "cascade_cost_ratio",
        "auroc_gap",
        "band_tuned_on_calibration",
    ):
        if not isinstance(artifact[key], float):
            raise ValueError(f"{key} must be a bare float")
    if not isinstance(artifact["n_calibration"], int) or not isinstance(artifact["n_heldout"], int):
        raise ValueError("split counts must be bare ints")
    if int(artifact["n_calibration"]) <= 0 or int(artifact["n_heldout"]) <= 0:
        raise ValueError("split counts must be positive for complete artifacts")


def write_artifact(output_path: Path, artifact: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_experiment(config: CascadeConfig | None = None, *, write: bool = True) -> dict[str, object]:
    """Run Exp 3918 from cached Exp 3917 scores, or write a blocked artifact."""

    config = config or CascadeConfig(repo_root=Path(__file__).resolve().parents[3])
    started_at = config.start_time()
    active_config = replace(config, started_at=started_at)
    checks, blocked_reason, evidence = probe_preconditions(active_config.repo_root)
    if blocked_reason is not None or evidence is None:
        artifact = build_blocked_artifact(
            config=active_config,
            reason=blocked_reason or "blocked_upstream_efficiency_missing",
            preconditions_checked=checks,
            started_at=started_at,
        )
    else:
        artifact = build_complete_artifact(
            config=active_config,
            evidence=evidence,
            preconditions_checked=checks,
            started_at=started_at,
        )
    if write:
        write_artifact(active_config.resolved_output_path(), artifact)
    return artifact


def cli_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-path", type=Path, default=None)
    args = parser.parse_args(argv)
    artifact = run_experiment(
        CascadeConfig(repo_root=args.repo_root, output_path=args.output_path),
        write=True,
    )
    output_path = args.output_path if args.output_path is not None else args.repo_root / OUTPUT_REL_PATH
    print(f"{output_path.name} wrote {artifact['honest_verdict']}")
    return 0 if str(artifact["honest_verdict"]).startswith("complete:") else 1
