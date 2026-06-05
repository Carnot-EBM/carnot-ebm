"""Exp 3836 formal-core certified abstention operating point.

Spec: REQ-SPOE-3836, SCENARIO-SPOE-3836.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import importlib
import json
import math
from pathlib import Path
import time
from typing import Any

from carnot.eval.fover_memory_leakage_v3 import _label_to_int, _score_text_verifiers
from carnot.pipeline import certified_abstention_operating_point_3771 as exp3771
from carnot.pipeline.risk_coverage_abstention_3718 import AbstentionExample


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path(
    "results/experiment_3836_formal_core_certified_abstention_operating_point.json"
)
EXP3835_REL_PATH = Path("results/experiment_3835_formal_core_5seed_ci.json")
EXP3771_REL_PATH = Path("results/experiment_3771_certified_abstention_operating_point.json")
FOVER_TEST_REL_PATH = Path("data/fover_test_v4.json")
FOVER_CERTIFICATION_REL_PATH = Path("data/fover_corpus_v4.json")
FORMAL_SCORE_DEFINITION = "0.9*tier0r_curry_howard + 0.1*tier0u_logical_consistency"
RANDOM_SEED = exp3771.RANDOM_SEED
TARGET_RISK = exp3771.TARGET_RISK
CONFORMAL_DELTA = exp3771.DELTA
FORMAL_ONLY_AUROC_GATE = 0.85
INFERENCE_SUBSTRATE = (
    "verifier-scoring-only formal core "
    "(tier0r_curry_howard + tier0u_logical_consistency; no fr11_session_memory; "
    "no trained weights)."
)

VERDICT_WEAK_PREFIX = "complete: formal_core_certified_abstention_WEAK_coverage"
VERDICT_SHIPPED_PREFIX = "complete: formal_core_certified_abstention_threshold"

REQUIRED_ARTIFACT_FIELDS = (
    "formal_core_certified_threshold",
    "formal_core_certified_coverage_at_risk_0_05",
    "coverage_delta_vs_full_ensemble",
    "conformal_delta",
    "n_calibration",
    "n_test",
    "cited_upstream_artifacts",
    "preconditions_checked",
    "honest_verdict",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "inference_substrate",
    "field_provenance",
)

FIELD_PROVENANCE = {
    "formal_core_certified_threshold": {
        "principle": "the deployable tau on the contamination-free scorer",
    },
    "formal_core_certified_coverage_at_risk_0_05": {
        "principle": ("fraction of candidates served confidently under the certified risk bound"),
    },
    "coverage_delta_vs_full_ensemble": {
        "principle": (
            "the honest cost of the no-learned-weights guarantee vs exp3771's "
            "full-ensemble point; formal coverage minus full-ensemble coverage"
        ),
    },
    "conformal_delta": {
        "principle": "the split-conformal/PAC confidence budget used by Exp 3771",
    },
    "n_calibration": {"principle": "calibration sample size for threshold selection"},
    "n_test": {"principle": "held-out test sample size for certified risk evaluation"},
    "cited_upstream_artifacts": {
        "principle": "exp3835 gate and exp3771 comparison evidence with sha256 drift checks",
    },
    "preconditions_checked": {
        "principle": "resources checked before scoring; blocks prevent fabricated metrics",
    },
    "honest_verdict": {
        "principle": "terminal verdict prefix, or blocked_<resource> on precondition failure",
    },
    "random_seed": {"principle": "deterministic Exp 3771 calibration/test split seed"},
    "reproducibility_checksum": {
        "principle": "content hash over deterministic certification fields",
    },
    "duration_s": {"principle": "real wall-clock duration, not part of the checksum"},
    "inference_substrate": {
        "principle": "records the no-live-LLM, no-learned-weight scoring substrate",
    },
    "field_provenance": {
        "principle": "machine-readable principles for required artifact fields",
    },
}


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
    min_examples: int = 200,
) -> JsonDict:
    """Build the Exp 3836 artifact from cached FoVer candidate rows."""

    start = time.perf_counter() if started_s is None else float(started_s)
    root_path = Path(root)
    preconditions, blocked_verdict = check_preconditions(root_path)
    if blocked_verdict is not None:
        return build_blocked_artifact(
            blocked_verdict,
            preconditions_checked=preconditions,
            started_s=start,
            now_s=now_s,
            tests_run=tests_run,
        )

    examples, corpus_status = load_formal_core_examples(root_path)
    upstream = load_upstream_artifacts(root_path)
    return build_artifact_from_examples(
        examples,
        started_s=start,
        now_s=now_s,
        tests_run=tests_run,
        min_examples=min_examples,
        preconditions_checked=preconditions,
        cited_upstream_artifacts=upstream,
        extra={
            "corpus_status": corpus_status,
            "output_path": str(_repo_path(root_path, OUTPUT_REL_PATH)),
        },
    )


def build_artifact_from_examples(
    examples: Sequence[AbstentionExample],
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
    min_examples: int = 200,
    preconditions_checked: Sequence[Mapping[str, Any]] | None = None,
    cited_upstream_artifacts: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Assemble the formal-core abstention artifact from scored examples."""

    start = time.perf_counter() if started_s is None else float(started_s)
    clean = _clean_examples(examples)
    if len(clean) < int(min_examples) or len({example.label for example in clean}) < 2:
        return build_blocked_artifact(
            "blocked_formal_core_candidate_rows_unavailable",
            preconditions_checked=list(preconditions_checked or []),
            started_s=start,
            now_s=now_s,
            tests_run=tests_run,
            cited_upstream_artifacts=cited_upstream_artifacts,
            extra=extra,
        )

    labels = [example.label for example in clean]
    scores = [example.energy_score for example in clean]
    result = exp3771.split_conformal_certification(labels, scores)
    certified = bool(
        result["usable_operating_point_exists"] and result["certified_risk_bound"] <= TARGET_RISK
    )
    certified_threshold = result["selected_threshold"] if certified else None
    certified_coverage = result["coverage_at_operating_point"] if certified else 0.0
    full_coverage = _full_ensemble_coverage(cited_upstream_artifacts)
    coverage_delta = (
        _round(certified_coverage - full_coverage) if full_coverage is not None else None
    )
    verdict = _verdict(certified_threshold, certified_coverage)
    finished = time.perf_counter() if now_s is None else float(now_s)
    artifact = _base_artifact(
        verdict=verdict,
        duration_s=_round(max(0.0, finished - start)),
        tests_run=tests_run,
    )
    artifact.update(
        {
            "formal_core_certified_threshold": certified_threshold,
            "formal_core_certified_coverage_at_risk_0_05": _round(certified_coverage),
            "formal_core_certified_risk_bound": result["certified_risk_bound"],
            "formal_core_candidate_threshold": result["selected_threshold"],
            "formal_core_candidate_coverage_at_selected_threshold": result[
                "coverage_at_operating_point"
            ],
            "formal_core_candidate_risk_bound": result["certified_risk_bound"],
            "formal_core_score_definition": FORMAL_SCORE_DEFINITION,
            "coverage_delta_vs_full_ensemble": coverage_delta,
            "full_ensemble_certified_coverage_reference": full_coverage,
            "conformal_delta": CONFORMAL_DELTA,
            "risk_target": TARGET_RISK,
            "aurc": result["aurc"],
            "risk_coverage_curve": risk_coverage_curve(labels, scores),
            "certification_method": result["certification_method"],
            "n_calibration": result["n_calibration"],
            "n_test": result["n_test"],
            "n_candidates_scored": len(clean),
            "formal_core_certified_operating_point_exists": bool(
                certified and certified_coverage > 0.90
            ),
            "contamination_free_components": [
                "tier0r_curry_howard",
                "tier0u_logical_consistency",
            ],
            "excluded_components": ["fr11_session_memory"],
            "cited_upstream_artifacts": dict(cited_upstream_artifacts or {}),
            "preconditions_checked": [dict(item) for item in preconditions_checked or []],
            "honest_comparison": _honest_comparison(certified_coverage, full_coverage),
            "doc_update_proposal": doc_update_proposal(certified_threshold, certified_coverage),
        }
    )
    artifact.update(dict(extra or {}))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def build_blocked_artifact(
    verdict: str,
    *,
    preconditions_checked: Sequence[Mapping[str, Any]],
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
    cited_upstream_artifacts: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Return a blocked artifact without fabricated formal-core metrics."""

    start = time.perf_counter() if started_s is None else float(started_s)
    finished = time.perf_counter() if now_s is None else float(now_s)
    artifact = _base_artifact(
        verdict=verdict,
        duration_s=_round(max(0.0, finished - start)),
        tests_run=tests_run,
    )
    artifact.update(_empty_measurements())
    artifact.update(
        {
            "cited_upstream_artifacts": dict(cited_upstream_artifacts or {}),
            "preconditions_checked": [dict(item) for item in preconditions_checked],
            "doc_update_proposal": (
                "No doc change proposed: Exp 3836 blocked before certified "
                "formal-core metrics were available."
            ),
        }
    )
    artifact.update(dict(extra or {}))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def check_preconditions(root: Path | str) -> tuple[list[JsonDict], str | None]:
    """Check required gates and return the first blocked verdict, if any."""

    root_path = Path(root)
    checks: list[JsonDict] = []
    try:
        importlib.import_module("carnot.verify")
        checks.append(
            {
                "resource": "carnot_verify_import",
                "available": True,
                "detail": "import carnot.verify succeeded",
            }
        )
    except Exception as exc:
        checks.append(
            {
                "resource": "carnot_verify_import",
                "available": False,
                "detail": repr(exc),
            }
        )

    fover_test_path = root_path / FOVER_TEST_REL_PATH
    checks.append(
        {
            "resource": "fover_test_v4",
            "available": fover_test_path.is_file(),
            "detail": str(fover_test_path),
        }
    )
    certification_path = root_path / FOVER_CERTIFICATION_REL_PATH
    checks.append(
        {
            "resource": "fover_corpus_v4",
            "available": certification_path.is_file(),
            "detail": str(certification_path),
        }
    )
    exp3835_path = root_path / EXP3835_REL_PATH
    exp3835 = _read_json_if_exists(exp3835_path)
    formal_auroc = _numeric_or_none(exp3835.get("formal_only_auroc_mean"))
    checks.append(
        {
            "resource": "exp3835_formal_only_auroc_gate",
            "available": bool(formal_auroc is not None and formal_auroc >= FORMAL_ONLY_AUROC_GATE),
            "detail": (
                f"{exp3835_path}; formal_only_auroc_mean={formal_auroc}; "
                f"required>={FORMAL_ONLY_AUROC_GATE}"
            ),
        }
    )
    exp3771_path = root_path / EXP3771_REL_PATH
    exp3771_artifact = _read_json_if_exists(exp3771_path)
    exp3771_coverage = _numeric_or_none(exp3771_artifact.get("coverage_at_operating_point"))
    checks.append(
        {
            "resource": "exp3771_full_ensemble_reference",
            "available": exp3771_coverage is not None,
            "detail": f"{exp3771_path}; coverage_at_operating_point={exp3771_coverage}",
        }
    )
    verdict_by_resource = {
        "carnot_verify_import": "blocked_carnot_verify_import",
        "fover_test_v4": "blocked_fover_test_v4_missing",
        "fover_corpus_v4": "blocked_fover_corpus_v4_missing",
        "exp3835_formal_only_auroc_gate": "blocked_exp3835_missing_or_weak",
        "exp3771_full_ensemble_reference": "blocked_exp3771_reference_missing",
    }
    for check in checks:
        if not check["available"]:
            return checks, verdict_by_resource[str(check["resource"])]
    return checks, None


def load_formal_core_examples(root: Path | str) -> tuple[list[AbstentionExample], JsonDict]:
    """Score the Exp 3771-aligned FoVer rows with the formal core only."""

    root_path = Path(root)
    path = root_path / FOVER_CERTIFICATION_REL_PATH
    if not path.is_file():
        return [], {"status": "missing", "path": str(path), "n_examples": 0}
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        return [], {"status": "blocked", "path": str(path), "n_examples": 0}
    texts = [str(row.get("step_text", "")) for row in rows if isinstance(row, Mapping)]
    verifier_scores = _score_text_verifiers(texts)
    formal_scores = formal_core_scores_from_verifier_scores(verifier_scores)
    examples: list[AbstentionExample] = []
    for idx, (row, score) in enumerate(zip(rows, formal_scores, strict=False)):
        if not isinstance(row, Mapping):
            continue
        examples.append(
            AbstentionExample(
                label=_label_to_int(row.get("label")),
                energy_score=score,
                baseline_score=0.0,
                example_id=f"formal-core-{row.get('question_id', idx)}-{idx}",
            )
        )
    return examples, {
        "status": "loaded",
        "path": str(path),
        "n_examples": len(examples),
        "formal_score_definition": FORMAL_SCORE_DEFINITION,
    }


def formal_core_scores_from_verifier_scores(
    verifier_scores: Mapping[str, Sequence[float]],
) -> list[float]:
    """Return 0.9*tier0r + 0.1*tier0u and reject FR-11 memory input."""

    if "fr11_session_memory" in verifier_scores:
        raise ValueError("formal-core certification must not use fr11_session_memory")
    tier0r = verifier_scores.get("tier0r_curry_howard")
    tier0u = verifier_scores.get("tier0u_logical_consistency")
    if tier0r is None or tier0u is None:
        raise ValueError(
            "formal-core scoring requires tier0r_curry_howard and tier0u_logical_consistency"
        )
    if len(tier0r) != len(tier0u):
        raise ValueError("tier0r and tier0u score lengths must match")
    return [_round(0.9 * float(r) + 0.1 * float(u)) for r, u in zip(tier0r, tier0u, strict=True)]


def risk_coverage_curve(labels: Sequence[int], scores: Sequence[float]) -> list[JsonDict]:
    """Build a tau sweep where scores <= tau are served confidently."""

    clean = [
        (float(score), 1 if int(label) else 0)
        for label, score in zip(labels, scores, strict=False)
        if math.isfinite(float(score))
    ]
    if not clean:
        return []
    clean.sort(key=lambda item: item[0])
    rows: list[JsonDict] = []
    errors = 0
    n = len(clean)
    for idx, (score, label) in enumerate(clean):
        errors += label
        next_score = clean[idx + 1][0] if idx + 1 < n else None
        if next_score is not None and next_score == score:
            continue
        kept = idx + 1
        rows.append(
            {
                "tau": _round(score),
                "coverage": _round(kept / n),
                "selective_risk": _round(errors / kept),
                "kept": kept,
                "errors_kept": errors,
            }
        )
    return rows


def load_upstream_artifacts(root: Path | str) -> JsonDict:
    """Load Exp 3835 and Exp 3771 references with SHA256 checksums."""

    root_path = Path(root)
    exp3835_path = root_path / EXP3835_REL_PATH
    exp3771_path = root_path / EXP3771_REL_PATH
    exp3835_artifact = _read_json_if_exists(exp3835_path)
    exp3771_artifact = _read_json_if_exists(exp3771_path)
    return {
        "exp3835": {
            "path": str(EXP3835_REL_PATH),
            "sha256": _sha256_file(exp3835_path),
            "formal_only_auroc_mean": exp3835_artifact.get("formal_only_auroc_mean"),
        },
        "exp3771": {
            "path": str(EXP3771_REL_PATH),
            "sha256": _sha256_file(exp3771_path),
            "selected_threshold": exp3771_artifact.get("selected_threshold"),
            "coverage_at_operating_point": exp3771_artifact.get("coverage_at_operating_point"),
        },
    }


def doc_update_proposal(threshold: float | None, coverage: float) -> str:
    """Return the operator-doc proposal text without editing curated docs."""

    if threshold is None:
        return (
            "Proposal only: document Exp 3836 as an honest weak formal-core "
            "certification result; keep Exp 3771 as the product abstention point."
        )
    return (
        "Proposal only: add Exp 3836 as the contamination-free formal-core "
        f"certified abstention point with tau={_round(threshold)} and "
        f"coverage={_round(coverage)} at selective-risk<=0.05; note that the "
        "score uses tier0r/tier0u only and excludes fr11_session_memory."
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3836 artifact contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    verdict = str(artifact.get("honest_verdict", ""))
    if not (
        verdict.startswith(VERDICT_SHIPPED_PREFIX)
        or verdict.startswith(VERDICT_WEAK_PREFIX)
        or verdict.startswith("blocked_")
    ):
        raise ValueError("honest_verdict is not an accepted Exp 3836 terminal verdict")
    duration = artifact.get("duration_s")
    if not isinstance(duration, int | float) or float(duration) < 0.0:
        raise ValueError("duration_s must be a non-negative number")
    provenance = artifact.get("field_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("field_provenance must be a mapping")
    missing_provenance = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in provenance]
    if missing_provenance:
        raise ValueError(f"field_provenance is missing principles: {missing_provenance}")
    if artifact.get("conformal_delta") != CONFORMAL_DELTA:
        raise ValueError("conformal_delta must match the Exp 3771 delta")
    coverage = artifact.get("formal_core_certified_coverage_at_risk_0_05")
    if not isinstance(coverage, int | float) or float(coverage) < 0.0:
        raise ValueError("formal_core_certified_coverage_at_risk_0_05 must be non-negative")
    if not verdict.startswith("blocked_"):
        if not isinstance(artifact.get("n_calibration"), int) or artifact["n_calibration"] < 100:
            raise ValueError("n_calibration must be an integer >= 100")
        if not isinstance(artifact.get("n_test"), int) or artifact["n_test"] < 100:
            raise ValueError("n_test must be an integer >= 100")
    if verdict.startswith(VERDICT_SHIPPED_PREFIX):
        if artifact.get("formal_core_certified_threshold") is None:
            raise ValueError("shipped verdict requires a certified threshold")
        if float(coverage) <= 0.90:
            raise ValueError("shipped verdict requires certified coverage > 0.90")
    if verdict.startswith(VERDICT_WEAK_PREFIX) and float(coverage) > 0.90:
        raise ValueError("weak verdict cannot report certified coverage > 0.90")


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash deterministic Exp 3836 certification fields."""

    payload = {
        "formal_core_certified_threshold": artifact.get("formal_core_certified_threshold"),
        "formal_core_certified_coverage_at_risk_0_05": artifact.get(
            "formal_core_certified_coverage_at_risk_0_05"
        ),
        "formal_core_certified_risk_bound": artifact.get("formal_core_certified_risk_bound"),
        "coverage_delta_vs_full_ensemble": artifact.get("coverage_delta_vs_full_ensemble"),
        "aurc": artifact.get("aurc"),
        "n_calibration": artifact.get("n_calibration"),
        "n_test": artifact.get("n_test"),
        "n_candidates_scored": artifact.get("n_candidates_scored"),
        "cited_upstream_artifacts": artifact.get("cited_upstream_artifacts"),
        "random_seed": RANDOM_SEED,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and persist the Exp 3836 artifact."""

    root_path = Path(root)
    output = _repo_path(root_path, Path(output_path))
    artifact = build_artifact(root_path, tests_run=tests_run)
    validate_artifact(artifact)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _base_artifact(
    *,
    verdict: str,
    duration_s: float,
    tests_run: Sequence[str] | None,
) -> JsonDict:
    return {
        "artifact": "experiment_3836_formal_core_certified_abstention_operating_point",
        "schema": "carnot.formal_core_certified_abstention_3836.v1",
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "duration_s": duration_s,
        "field_provenance": dict(FIELD_PROVENANCE),
        "tests_run": list(tests_run or []),
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
    }


def _empty_measurements() -> JsonDict:
    return {
        "formal_core_certified_threshold": None,
        "formal_core_certified_coverage_at_risk_0_05": 0.0,
        "formal_core_certified_risk_bound": None,
        "formal_core_candidate_threshold": None,
        "formal_core_candidate_coverage_at_selected_threshold": 0.0,
        "formal_core_candidate_risk_bound": None,
        "formal_core_score_definition": FORMAL_SCORE_DEFINITION,
        "coverage_delta_vs_full_ensemble": None,
        "full_ensemble_certified_coverage_reference": None,
        "conformal_delta": CONFORMAL_DELTA,
        "risk_target": TARGET_RISK,
        "aurc": None,
        "risk_coverage_curve": [],
        "certification_method": "none",
        "n_calibration": 0,
        "n_test": 0,
        "n_candidates_scored": 0,
        "formal_core_certified_operating_point_exists": False,
        "contamination_free_components": [
            "tier0r_curry_howard",
            "tier0u_logical_consistency",
        ],
        "excluded_components": ["fr11_session_memory"],
        "honest_comparison": "blocked before comparison",
    }


def _clean_examples(examples: Sequence[AbstentionExample]) -> list[AbstentionExample]:
    clean = []
    for example in examples:
        score = float(example.energy_score)
        if math.isfinite(score):
            clean.append(
                AbstentionExample(
                    label=1 if int(example.label) else 0,
                    energy_score=score,
                    baseline_score=0.0,
                    example_id=str(example.example_id),
                )
            )
    return clean


def _full_ensemble_coverage(upstream: Mapping[str, Any] | None) -> float | None:
    if not upstream:
        return None
    exp3771_ref = upstream.get("exp3771")
    if not isinstance(exp3771_ref, Mapping):
        return None
    return _numeric_or_none(exp3771_ref.get("coverage_at_operating_point"))


def _verdict(threshold: float | None, coverage: float) -> str:
    coverage_text = str(_round(coverage))
    if threshold is not None and coverage > 0.90:
        return (
            f"{VERDICT_SHIPPED_PREFIX}{_round(threshold)}_coverage{coverage_text}"
            "_at_risk_0.05_contamination_free"
        )
    return (
        f"{VERDICT_WEAK_PREFIX}{coverage_text}"
        "_clean_core_low_coverage_full_ensemble_remains_product"
    )


def _honest_comparison(coverage: float, full_coverage: float | None) -> str:
    if full_coverage is None:
        return "Exp 3771 full-ensemble coverage reference unavailable."
    delta = _round(coverage - full_coverage)
    if delta >= 0.0:
        return (
            "No certified coverage sacrifice on the Exp 3771-aligned corpus: "
            f"formal core coverage is {delta} above the full-ensemble reference."
        )
    return f"Formal core sacrifices {_round(-delta)} coverage versus the full ensemble."


def _read_json_if_exists(path: Path) -> JsonDict:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric_or_none(value: Any) -> float | None:
    if isinstance(value, int | float) and math.isfinite(float(value)):
        return float(value)
    return None


def _repo_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _round(value: float) -> float:
    return round(float(value), 6)


__all__ = [
    "CONFORMAL_DELTA",
    "OUTPUT_REL_PATH",
    "REQUIRED_ARTIFACT_FIELDS",
    "TARGET_RISK",
    "build_artifact",
    "build_artifact_from_examples",
    "build_blocked_artifact",
    "check_preconditions",
    "doc_update_proposal",
    "formal_core_scores_from_verifier_scores",
    "load_formal_core_examples",
    "load_upstream_artifacts",
    "reproducibility_checksum",
    "risk_coverage_curve",
    "validate_artifact",
    "write_artifact",
]
