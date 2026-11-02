"""Exp 4375: verifier-as-detector measurement on zero-selection-headroom FoVer.

Spec refs: REQ-VERIFY-4375, SCENARIO-VERIFY-4375.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.eval.fover_memory_leakage_v3 import (
    FR11_MEMORY_BOOST,
    _fr11_memory_score,
    _label_to_int,
    _load_fr11_memory_index,
    _read_fover_rows,
    _score_text_verifiers,
    compute_auroc,
    discover_fr11_state_files,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = ROOT / "results" / "experiment_4375_verifier_as_detector_measurement.json"
CORPUS_PATH = ROOT / "data" / "fover_corpus.jsonl"
REGISTRY_PATH = ROOT / "ops" / "verifier_registry.yaml"
RANDOM_SEED = 4375
RANDOM_SEEDS_USED = (4375,)
BOOTSTRAP_RESAMPLES = 2500
MIN_CANDIDATES = 1000
SPEC_REFS = ["REQ-VERIFY-4375", "SCENARIO-VERIFY-4375"]
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"


FIELD_PRINCIPLES = {
    "honest_verdict": (
        'Terminal-prefixed. A detection-AUROC-beats-chance result (the verifier '
        "carries oracle-distinct discriminating signal beyond selection) and an "
        "honest null (the verifier cannot detect on this ~0-headroom corpus) are "
        "BOTH decision-grade."
    ),
    "detector_auroc": (
        "BARE float: the verifier-ensemble step-error detection AUROC on the "
        "~0-selection-headroom corpus -- the oracle-distinct discrimination "
        "signal (the capstone reads this)."
    ),
    "detector_beats_chance": (
        "BARE bool: true iff the detection AUROC CI95 lower bound > 0.5 -- the "
        "verifier carries discriminating signal where SELECTION cannot express it."
    ),
    "selection_headroom": (
        "dict {oracle_at_k, vote_at_1, headroom} -- confirms the ~0 selection "
        "headroom so a high detection AUROC is the COMPLEMENT of selection (not "
        "double-counting), and a null is 'no detection', not 'no headroom' "
        "(FALSE_NEGATIVE_RISK guard)."
    ),
    "detector_auroc_ci95": (
        "CI95 of the detection AUROC (5-seed or bootstrap >=2000 resamples) -- "
        "the statistical bar for the beats-chance claim."
    ),
    "n_candidates": (
        "BARE int: scored candidate count -- MUST be >= 1000 for a sub-percentage "
        "detection claim (sample-size rigor)."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- a learned/energy detection signal scored against "
        "cached candidates, oracle-distinct."
    ),
    "preconditions_checked": (
        "Records the cached-candidate + ensemble availability + TRM-stand-down; "
        "pre-empts the silent-missing-resource fabrication mode."
    ),
    "random_seed": (
        "Determinism precondition for the (stochastic) ensemble scoring + the bootstrap."
    ),
    "reproducibility_checksum": (
        "Hash of the corpus + the ensemble config + the AUROC computation; lets "
        "a third party re-run."
    ),
    "model_specs": (
        "The verifier ensemble + the corpus + the selection-headroom-~0 condition "
        "+ n + seeds; required methodology."
    ),
}


@dataclass(frozen=True)
class PreconditionCheck:
    """One prerequisite checked before the cached-corpus measurement runs."""

    resource: str
    available: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "resource": self.resource,
            "available": bool(self.available),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ScoreBundle:
    """Production FoVer ensemble scores plus per-verifier diagnostic scores."""

    scores: list[float]
    per_verifier_scores: dict[str, list[float]]


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime configuration for Exp 4375."""

    repo_root: Path = ROOT
    corpus_path: Path = CORPUS_PATH
    registry_path: Path = REGISTRY_PATH
    artifact_path: Path = ARTIFACT_PATH
    min_candidates: int = MIN_CANDIDATES
    random_seed: int = RANDOM_SEED
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at


Scorer = Callable[[list[dict[str, Any]], Path], ScoreBundle]
AdversarialVerifyRunner = Callable[[Path], dict[str, Any]]


def label_to_error(label: Any) -> int:
    """Map FoVer labels to the detector target, where 1 means step-error."""

    return int(_label_to_int(label))


def round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return round(float(value), digits)


def read_labeled_fover_rows(path: Path) -> list[dict[str, Any]]:
    """Read cached FoVer rows with labels understood by the existing scorer."""

    return _read_fover_rows(path)


def compute_detector_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute step-error AUROC using the existing FoVer rank convention."""

    return float(compute_auroc(labels, scores))


def bootstrap_auroc_ci95(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    seed: int,
    resamples: int,
) -> list[float | None]:
    """Bootstrap a deterministic verifier AUROC over cached candidate rows."""

    if len(labels) != len(scores) or len(set(int(label) for label in labels)) < 2:
        return [None, None]
    rng = random.Random(seed)
    n = len(labels)
    values: list[float] = []
    for _ in range(resamples):
        sample_labels: list[int] = []
        sample_scores: list[float] = []
        for _row in range(n):
            idx = rng.randrange(n)
            sample_labels.append(int(labels[idx]))
            sample_scores.append(float(scores[idx]))
        if len(set(sample_labels)) < 2:
            continue
        values.append(compute_detector_auroc(sample_labels, sample_scores))
    if not values:
        return [None, None]
    values.sort()
    lo = int(0.025 * (len(values) - 1))
    hi = int(0.975 * (len(values) - 1))
    return [round_float(values[lo]), round_float(values[hi])]


def detector_beats_chance(ci95: Sequence[float | None]) -> bool:
    """Return the exact gate: CI95 lower bound strictly exceeds chance."""

    return bool(ci95 and ci95[0] is not None and float(ci95[0]) > 0.5)


def selection_headroom_for_single_candidate_rows(labels: Sequence[int]) -> dict[str, float]:
    """Selection headroom for K=1 rows; oracle and vote see the same candidate."""

    n = len(labels)
    correct_rate = sum(1 for label in labels if int(label) == 0) / max(1, n)
    rounded = round_float(correct_rate)
    return {
        "oracle_at_k": rounded if rounded is not None else 0.0,
        "vote_at_1": rounded if rounded is not None else 0.0,
        "headroom": 0.0,
    }


def score_fover_production_ensemble(
    rows: list[dict[str, Any]],
    repo_root: Path,
) -> ScoreBundle:
    """Score rows with the same production FoVer ensemble used by Exp 2850.

    This is deliberately a cached-candidate scoring path. It loads the text
    verifiers and optional FR-11 memory state, but it does not run an LLM,
    train TRM, or generate new candidates.
    """

    texts = [str(row.get("step_text", "")) for row in rows]
    per_verifier = _score_text_verifiers(texts)
    architecture_scores = [
        0.9 * r_score + 0.1 * u_score
        for r_score, u_score in zip(
            per_verifier["tier0r_curry_howard"],
            per_verifier["tier0u_logical_consistency"],
            strict=True,
        )
    ]
    memory_index = _load_fr11_memory_index(repo_root)
    memory_scores = [_fr11_memory_score(row, memory_index) for row in rows]
    per_verifier["fr11_session_memory"] = memory_scores
    scores = [
        score + FR11_MEMORY_BOOST * memory_score
        for score, memory_score in zip(architecture_scores, memory_scores, strict=True)
    ]
    return ScoreBundle(scores=[float(score) for score in scores], per_verifier_scores=per_verifier)


def _scoring_path_loads() -> bool:
    score_fover_production_ensemble(
        [{"step_text": "Compute 1+1=2.", "label": "correct", "question_id": "probe"}],
        ROOT,
    )
    return True


def _registry_has_fover_ensemble(path: Path) -> bool:
    if not path.is_file():
        return False
    return "fover_production_ensemble" in path.read_text(encoding="utf-8")


def _corpus_check(path: Path, min_candidates: int) -> PreconditionCheck:
    if not path.is_file():
        return PreconditionCheck("cached_fover_rows", False, "missing")
    try:
        rows = read_labeled_fover_rows(path)
    except Exception as exc:
        return PreconditionCheck("cached_fover_rows", False, f"unreadable: {exc}")
    labels = [label_to_error(row["label"]) for row in rows]
    if len(rows) < min_candidates:
        return PreconditionCheck(
            "cached_fover_rows",
            False,
            f"labeled_rows={len(rows)}; required>={min_candidates}",
        )
    if len(set(labels)) < 2:
        return PreconditionCheck(
            "cached_fover_rows",
            False,
            f"labeled_rows={len(rows)}; needs both correct and incorrect labels",
        )
    return PreconditionCheck(
        "cached_fover_rows",
        True,
        (
            f"labeled_rows={len(rows)}; correct={sum(1 for label in labels if label == 0)}; "
            f"incorrect={sum(1 for label in labels if label == 1)}"
        ),
    )


def check_preconditions(
    *,
    repo_root: Path,
    corpus_path: Path,
    registry_path: Path,
    min_candidates: int,
    scoring_path_checker: Callable[[], bool] = _scoring_path_loads,
) -> list[PreconditionCheck]:
    """Check cached-corpus, ensemble, and no-training preconditions."""

    checks = [_corpus_check(corpus_path, min_candidates)]
    registry_ok = _registry_has_fover_ensemble(registry_path)
    checks.append(
        PreconditionCheck(
            "verifier_registry",
            registry_ok,
            "fover_production_ensemble present" if registry_ok else "missing fover_production_ensemble",
        )
    )
    try:
        scoring_ok = bool(scoring_path_checker())
        scoring_detail = "production FoVer scoring path imports and scores a probe row"
    except Exception as exc:
        scoring_ok = False
        scoring_detail = f"scoring path failed: {exc}"
    checks.append(PreconditionCheck("fover_scoring_path", scoring_ok, scoring_detail))
    checks.append(
        PreconditionCheck(
            "trm_training_stand_down",
            True,
            "not invoked; this experiment scores cached FoVer candidates only",
        )
    )
    checks.append(
        PreconditionCheck(
            "fr11_state_files",
            True,
            f"count={len(discover_fr11_state_files(repo_root))}",
        )
    )
    return checks


def hash_sources(
    source_paths: Sequence[Path],
    *,
    payload: dict[str, Any],
) -> str:
    digest = hashlib.sha256()
    for path in sorted({Path(path) for path in source_paths}, key=lambda item: str(item)):
        digest.update(str(path).encode("utf-8"))
        if not path.exists():
            digest.update(b"\0MISSING\0")
            continue
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return "sha256:" + digest.hexdigest()


def _class_values(
    rows: Sequence[dict[str, Any]],
    labels: Sequence[int],
    scores: Sequence[float],
) -> dict[str, dict[str, list[float] | list[int]]]:
    grouped: dict[str, dict[str, list[float] | list[int]]] = defaultdict(
        lambda: {"labels": [], "scores": []}
    )
    for row, label, score in zip(rows, labels, scores, strict=True):
        key = str(row.get("problem_type") or row.get("source") or "untyped")
        grouped[key]["labels"].append(int(label))  # type: ignore[union-attr]
        grouped[key]["scores"].append(float(score))  # type: ignore[union-attr]
    return dict(grouped)


def missing_verifier_gaps(
    rows: Sequence[dict[str, Any]],
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    min_class_n: int = 100,
) -> list[dict[str, Any]]:
    """Summarize classes where the current ensemble lacks detection signal."""

    gaps: list[dict[str, Any]] = []
    for class_name, values in _class_values(rows, labels, scores).items():
        cls_labels = [int(value) for value in values["labels"]]
        cls_scores = [float(value) for value in values["scores"]]
        if len(cls_labels) < min_class_n or len(set(cls_labels)) < 2:
            continue
        cls_auroc = compute_detector_auroc(cls_labels, cls_scores)
        if cls_auroc <= 0.55:
            gaps.append(
                {
                    "gap_id": f"GAP-FOVER-DETECTOR-CLASS-{class_name}",
                    "status": "open",
                    "error_class": class_name,
                    "n": len(cls_labels),
                    "detector_auroc": round_float(cls_auroc),
                    "missing_discriminator": (
                        "A FoVer step-error feature that separates this class beyond "
                        "the production text/memory ensemble."
                    ),
                }
            )
    return gaps


def _per_verifier_aurocs(
    labels: Sequence[int],
    per_verifier_scores: dict[str, Sequence[float]],
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for name, values in sorted(per_verifier_scores.items()):
        if len(values) != len(labels) or len(set(labels)) < 2:
            result[name] = None
        else:
            result[name] = round_float(compute_detector_auroc(labels, values))
    return result


def _model_specs(
    *,
    n_candidates: int,
    corpus_path: Path,
    registry_path: Path,
    bootstrap_resamples: int,
    random_seed: int,
) -> dict[str, Any]:
    return {
        "verifier_ensemble_id": "fover_production_ensemble",
        "verifier_registry_path": str(registry_path),
        "scoring_code_path": "python/carnot/eval/fover_memory_leakage_v3.py",
        "corpus": str(corpus_path),
        "corpus_condition": "full_cached_fover_rows_not_exp2850_balanced_seed_subset",
        "selection_condition": "single_candidate_k1",
        "label_target": "step_error",
        "n": int(n_candidates),
        "random_seed": int(random_seed),
        "random_seeds_used": list(RANDOM_SEEDS_USED),
        "bootstrap_resamples": int(bootstrap_resamples),
        "ensemble_components": [
            "tier0r_curry_howard",
            "tier0u_logical_consistency",
            "fr11_session_memory",
        ],
        "trm_training": "stood_down_not_invoked",
        "live_generation": False,
    }


def build_complete_artifact(
    *,
    rows: list[dict[str, Any]],
    labels: list[int],
    scores: list[float],
    per_verifier_scores: dict[str, Sequence[float]],
    preconditions_checked: list[dict[str, Any]],
    source_paths: Sequence[Path],
    duration_s: float,
    random_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    detector_auroc = round_float(compute_detector_auroc(labels, scores))
    ci95 = bootstrap_auroc_ci95(
        labels,
        scores,
        seed=random_seed,
        resamples=bootstrap_resamples,
    )
    beats = detector_beats_chance(ci95)
    selection_headroom = selection_headroom_for_single_candidate_rows(labels)
    if beats and selection_headroom["headroom"] == 0.0:
        verdict = "complete: detector_beats_chance_zero_selection_headroom_fover"
    else:
        verdict = "complete: detector_null_zero_selection_headroom_fover"
    checksum_payload = {
        "detector_auroc": detector_auroc,
        "detector_auroc_ci95": ci95,
        "selection_headroom": selection_headroom,
        "n_candidates": len(rows),
        "random_seed": random_seed,
        "bootstrap_resamples": bootstrap_resamples,
        "scoring_code_path": "python/carnot/eval/fover_memory_leakage_v3.py",
    }
    return {
        "experiment": "experiment_4375_verifier_as_detector_measurement",
        "schema": "carnot.verifier_as_detector_measurement.v1",
        "honest_verdict": verdict,
        "detector_auroc": detector_auroc,
        "detector_beats_chance": beats,
        "selection_headroom": selection_headroom,
        "detector_auroc_ci95": ci95,
        "n_candidates": int(len(rows)),
        "verifier_is_oracle": False,
        "preconditions_checked": preconditions_checked,
        "random_seed": int(random_seed),
        "random_seeds_used": list(RANDOM_SEEDS_USED),
        "bootstrap_resamples": int(bootstrap_resamples),
        "reproducibility_checksum": hash_sources(source_paths, payload=checksum_payload),
        "model_specs": _model_specs(
            n_candidates=len(rows),
            corpus_path=Path(source_paths[0]) if source_paths else CORPUS_PATH,
            registry_path=Path(source_paths[1]) if len(source_paths) > 1 else REGISTRY_PATH,
            bootstrap_resamples=bootstrap_resamples,
            random_seed=random_seed,
        ),
        "per_verifier_auroc": _per_verifier_aurocs(labels, per_verifier_scores),
        "label_balance": {
            "correct": sum(1 for label in labels if label == 0),
            "incorrect": sum(1 for label in labels if label == 1),
            "error_rate": round_float(sum(labels) / max(1, len(labels))),
        },
        "missing_verifier_gaps": missing_verifier_gaps(rows, labels, scores),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "duration_s": round_float(duration_s, 3),
        "inference_substrate": INFERENCE_SUBSTRATE,
    }


def build_blocked_artifact(
    *,
    preconditions_checked: list[dict[str, Any]],
    source_paths: Sequence[Path],
    duration_s: float,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4375_verifier_as_detector_measurement",
        "schema": "carnot.verifier_as_detector_measurement.v1",
        "honest_verdict": "blocked_cached_candidates_unavailable",
        "detector_auroc": None,
        "detector_beats_chance": False,
        "selection_headroom": {"oracle_at_k": None, "vote_at_1": None, "headroom": None},
        "detector_auroc_ci95": [None, None],
        "n_candidates": 0,
        "verifier_is_oracle": False,
        "preconditions_checked": preconditions_checked,
        "random_seed": RANDOM_SEED,
        "random_seeds_used": list(RANDOM_SEEDS_USED),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "reproducibility_checksum": hash_sources(
            source_paths,
            payload={"blocked": "blocked_cached_candidates_unavailable"},
        ),
        "model_specs": {
            "verifier_ensemble_id": "fover_production_ensemble",
            "selection_condition": "single_candidate_k1",
            "trm_training": "stood_down_not_invoked",
            "live_generation": False,
        },
        "missing_verifier_gaps": [],
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "duration_s": round_float(duration_s, 3),
        "inference_substrate": INFERENCE_SUBSTRATE,
    }


def run_adversarial_verify(path: Path, repo_root: Path = ROOT) -> dict[str, Any]:
    """Run the artifact verifier and keep a compact embedded report."""

    script = repo_root / "scripts" / "adversarial_verify.py"
    if not script.is_file():
        return {"returncode": None, "flags": [], "stderr": "scripts/adversarial_verify.py missing"}
    proc = subprocess.run(
        [sys.executable, str(script), str(path)],
        cwd=str(repo_root),
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    return {
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def _write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    scorer: Scorer = score_fover_production_ensemble,
    adversarial_verify_runner: AdversarialVerifyRunner = run_adversarial_verify,
    write: bool = True,
) -> dict[str, Any]:
    """Run Exp 4375 and optionally write the JSON artifact."""

    cfg = config or ExperimentConfig()
    started = cfg.start_time()
    source_paths = [cfg.corpus_path, cfg.registry_path]
    checks = check_preconditions(
        repo_root=cfg.repo_root,
        corpus_path=cfg.corpus_path,
        registry_path=cfg.registry_path,
        min_candidates=cfg.min_candidates,
    )
    preconditions = [check.as_dict() for check in checks]
    if not all(check.available for check in checks):
        artifact = build_blocked_artifact(
            preconditions_checked=preconditions,
            source_paths=source_paths,
            duration_s=cfg.clock() - started,
        )
        if write:
            _write_artifact(cfg.artifact_path, artifact)
        return artifact

    rows = read_labeled_fover_rows(cfg.corpus_path)
    labels = [label_to_error(row["label"]) for row in rows]
    score_bundle = scorer(rows, cfg.repo_root)
    artifact = build_complete_artifact(
        rows=rows,
        labels=labels,
        scores=score_bundle.scores,
        per_verifier_scores=score_bundle.per_verifier_scores,
        preconditions_checked=preconditions,
        source_paths=source_paths,
        duration_s=cfg.clock() - started,
        random_seed=cfg.random_seed,
        bootstrap_resamples=cfg.bootstrap_resamples,
    )
    if write:
        _write_artifact(cfg.artifact_path, artifact)
        adversarial = adversarial_verify_runner(cfg.artifact_path)
        artifact["adversarial_verify"] = adversarial
        _write_artifact(cfg.artifact_path, artifact)
    else:
        artifact["adversarial_verify"] = {"returncode": None, "skipped": True}
    return artifact


def main() -> int:
    artifact = run_experiment(write=True)
    print(
        "[exp4375] "
        f"{artifact['honest_verdict']} "
        f"detector_auroc={artifact['detector_auroc']} "
        f"detector_beats_chance={artifact['detector_beats_chance']} "
        f"n={artifact['n_candidates']} -> {ARTIFACT_PATH}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
