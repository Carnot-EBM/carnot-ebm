"""Exp 3837 FoVer learned-contribution category characterization.

Spec: REQ-VERIFY-3837, SCENARIO-VERIFY-3837.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any

from carnot.eval.fover_memory_leakage_v3 import (
    DEFAULT_N_EXAMPLES,
    DEFAULT_RANDOM_SEEDS,
    _fr11_memory_score,
    _label_to_int,
    _load_fr11_memory_index,
    _read_fover_rows,
    _score_text_verifiers,
    _select_balanced_subset,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_3837_fover_error_category_learned_contribution.json")
FOVER_CORPUS_REL_PATH = Path("data/fover_corpus.jsonl")
FOVER_TEST_REL_PATH = Path("data/fover_test_v4.json")
EXP3826_REL_PATH = Path("results/experiment_3826_fover_ablation_faithful.json")
INFERENCE_SUBSTRATE = (
    "verifier-scoring-only over cached Exp 3826 FoVer candidate panels; "
    "formal_only=0.9*tier0r_curry_howard+0.1*tier0u_logical_consistency; "
    "learned_only=fr11_session_memory; no live LLM load."
)
VERDICT_TOPGAP_PREFIX = "complete: learned_contribution_characterized_topgap_"
VERDICT_UNIFORM = "complete: learned_contribution_characterized_NO_category_signal_delta_uniform"
EXPLICIT_CATEGORY_FIELDS = (
    "error_type",
    "error_category",
    "failure_type",
    "failure_category",
    "category",
    "error_subtype",
)
COARSE_CATEGORY_ORDER = ("arithmetic", "logical", "formal-tool-checkable", "other")

REQUIRED_ARTIFACT_FIELDS = (
    "learned_contribution_by_category",
    "formal_core_gap_categories",
    "both_wrong_categories",
    "category_derivation_method",
    "n_candidates_scored",
    "preconditions_checked",
    "cited_upstream_artifacts",
    "honest_verdict",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "inference_substrate",
    "field_provenance",
)

FIELD_PROVENANCE: dict[str, JsonDict] = {
    "learned_contribution_by_category": {
        "principle": (
            "where the +0.0184 lives -- the categories the formal core misses; "
            "the paper's limitations data"
        ),
    },
    "formal_core_gap_categories": {
        "principle": (
            "named error subtypes a contamination-free-only verifier would not "
            "catch -- honest scope of the clean-core claim"
        ),
    },
    "both_wrong_categories": {
        "principle": (
            "ensemble blind spots -- neither formal nor learned catches these "
            "(residual for Tier-4 / future verifiers)"
        ),
    },
    "category_derivation_method": {
        "principle": (
            "states whether categories came from a corpus field or were "
            "text-derived -- no silent methodology gap"
        ),
    },
    "n_candidates_scored": {"principle": "number of scored Exp 3826 candidate instances"},
    "preconditions_checked": {
        "principle": "resources checked before scoring; blockers prevent fabricated counts"
    },
    "cited_upstream_artifacts": {
        "principle": "Exp 3826 partition/aggregation source with sha256 drift evidence"
    },
    "honest_verdict": {"principle": "terminal prefix naming top gap or uniform category signal"},
    "random_seed": {"principle": "Exp 3826 deterministic five-seed candidate panels"},
    "reproducibility_checksum": {"principle": "deterministic hash over scores, counts, and provenance"},
    "duration_s": {"principle": "real wall-clock duration, excluded from checksum"},
    "inference_substrate": {"principle": "declares cached verifier scoring only; no live LLM"},
    "field_provenance": {"principle": "machine-readable principles for required fields"},
}


@dataclass(frozen=True)
class ScoredCandidate:
    """One candidate row scored by the formal-only and learned-only paths."""

    candidate_id: str
    question_id: str
    label: int
    formal_score: float
    learned_score: float
    category: str
    step_text: str


def check_preconditions(root: Path | str) -> tuple[list[JsonDict], str | None]:
    """Check Exp 3837 resources and return the first blocked verdict."""

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
    except Exception as exc:  # noqa: BLE001 - the artifact needs exact blocker text.
        checks.append(
            {
                "resource": "carnot_verify_import",
                "available": False,
                "detail": repr(exc),
            }
        )
    checks.extend(
        [
            _file_check(root_path, FOVER_CORPUS_REL_PATH, "fover_corpus"),
            _file_check(root_path, FOVER_TEST_REL_PATH, "fover_test_v4"),
            _file_check(root_path, EXP3826_REL_PATH, "exp3826_artifact"),
        ]
    )
    verdict_by_resource = {
        "carnot_verify_import": "blocked_carnot_verify_import",
        "fover_corpus": "blocked_fover_corpus",
        "fover_test_v4": "blocked_fover_test_v4",
        "exp3826_artifact": "blocked_exp3826_artifact",
    }
    for check in checks:
        if not check["available"]:
            return checks, verdict_by_resource[str(check["resource"])]
    return checks, None


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
    random_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
    n_examples: int = DEFAULT_N_EXAMPLES,
) -> JsonDict:
    """Build the Exp 3837 artifact from cached FoVer candidate rows."""

    start = time.perf_counter() if started_s is None else float(started_s)
    root_path = Path(root)
    preconditions, blocked = check_preconditions(root_path)
    if blocked is not None:
        return build_blocked_artifact(
            blocked,
            preconditions_checked=preconditions,
            started_s=start,
            now_s=now_s,
            tests_run=tests_run,
            random_seed=list(random_seeds),
        )
    candidates, panel_meta = score_exp3826_candidate_panels(
        root_path,
        random_seeds=random_seeds,
        n_examples=n_examples,
    )
    category_method = dict(panel_meta.get("category_derivation_method") or {})
    if not category_method:
        category_method = category_derivation_method(_read_json_rows(root_path / FOVER_TEST_REL_PATH))
    return build_artifact_from_scored_candidates(
        candidates,
        category_derivation_method=category_method,
        preconditions_checked=preconditions,
        cited_upstream_artifacts=load_upstream_artifacts(root_path),
        started_s=start,
        now_s=now_s,
        tests_run=tests_run,
        random_seed=list(random_seeds),
        extra={"candidate_panel": dict(panel_meta)},
    )


def score_exp3826_candidate_panels(
    root: Path | str,
    *,
    random_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS,
    n_examples: int = DEFAULT_N_EXAMPLES,
) -> tuple[list[ScoredCandidate], JsonDict]:
    """Score the five Exp 3826 candidate panels with formal and learned paths."""

    root_path = Path(root)
    all_rows = _read_fover_rows(root_path / FOVER_CORPUS_REL_PATH)
    method = category_derivation_method(all_rows)
    memory_index = _load_fr11_memory_index(root_path)
    candidates: list[ScoredCandidate] = []
    unique_ids: set[str] = set()
    for seed in random_seeds:
        subset = _select_balanced_subset(all_rows, seed=int(seed), n_examples=int(n_examples))
        texts = [str(row.get("step_text", "")) for row in subset]
        verifier_scores = _score_text_verifiers(texts)
        memory_scores = [_fr11_memory_score(row, memory_index) for row in subset]
        for idx, row in enumerate(subset):
            question_id = str(row.get("question_id", f"row-{idx}"))
            step_text = str(row.get("step_text", ""))
            text_sha = hashlib.sha256(step_text.encode("utf-8")).hexdigest()[:16]
            unique_ids.add(f"{question_id}:{text_sha}:{row.get('label')}")
            formal_score = (
                0.9 * float(verifier_scores["tier0r_curry_howard"][idx])
                + 0.1 * float(verifier_scores["tier0u_logical_consistency"][idx])
            )
            candidates.append(
                ScoredCandidate(
                    candidate_id=f"seed{int(seed)}:{idx}:{question_id}:{text_sha}",
                    question_id=question_id,
                    label=_label_to_int(row.get("label")),
                    formal_score=formal_score,
                    learned_score=float(memory_scores[idx]),
                    category=derive_category(row, method),
                    step_text=step_text,
                )
            )
    return candidates, {
        "n_seed_candidate_instances": len(candidates),
        "n_unique_candidates": len(unique_ids),
        "n_candidates_per_seed": int(n_examples),
        "random_seeds": [int(seed) for seed in random_seeds],
        "category_derivation_method": method,
        "formal_score_definition": "0.9*tier0r_curry_howard + 0.1*tier0u_logical_consistency",
        "learned_score_definition": "fr11_session_memory",
    }


def category_derivation_method(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return explicit category-field provenance or the text-derived fallback."""

    for field in EXPLICIT_CATEGORY_FIELDS:
        if any(isinstance(row, Mapping) and row.get(field) not in (None, "") for row in rows):
            return {
                "method": "corpus_field",
                "field": field,
                "detail": f"Using explicit corpus field '{field}' for error categories.",
            }
    return {
        "method": "text_derived",
        "categories": list(COARSE_CATEGORY_ORDER),
        "detail": (
            "No explicit error-category field was found in the FoVer corpus/test "
            "schema; categories were derived from step_text using coarse "
            "arithmetic/logical/formal-tool-checkable/other rules."
        ),
    }


def derive_category(row: Mapping[str, Any], method: Mapping[str, Any]) -> str:
    """Derive the category for one row according to the recorded method."""

    if method.get("method") == "corpus_field":
        field = str(method.get("field", ""))
        value = row.get(field)
        return _slug(str(value)) if value not in (None, "") else "uncategorized"
    return coarse_text_category(str(row.get("step_text", "")))


def coarse_text_category(text: str) -> str:
    """Map a step text into the required coarse fallback categories."""

    lowered = text.lower()
    if re.search(r"\d+\s*(?:[+*/%=-]|-)\s*\d+", lowered) or "<<" in lowered:
        return "arithmetic"
    if re.search(
        r"\b(contradict\w*|inconsistent|implies|therefore not|cannot|must not|if and only if)\b",
        lowered,
    ):
        return "logical"
    if re.search(r"\\\(|\\\[|\b(z3|sat|unsat|constraint|theorem|proof|solve for|let [a-z]\s*=)\b", lowered):
        return "formal-tool-checkable"
    return "other"


def select_operating_threshold(labels: Sequence[int], scores: Sequence[float]) -> JsonDict:
    """Choose the scorer's own threshold by maximizing balanced accuracy."""

    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    clean = [(1 if int(label) else 0, float(score)) for label, score in zip(labels, scores, strict=True)]
    if not clean or not all(math.isfinite(score) for _label, score in clean):
        raise ValueError("scores must be finite")
    if len({label for label, _score in clean}) < 2:
        raise ValueError("operating threshold selection requires both classes")

    best: tuple[float, float, float, float, float, int, int, int, int] | None = None
    for threshold in sorted({score for _label, score in clean}):
        tp, tn, fp, fn = _confusion(labels, scores, threshold)
        tpr = tp / (tp + fn) if tp + fn else 0.0
        tnr = tn / (tn + fp) if tn + fp else 0.0
        accuracy = (tp + tn) / len(clean)
        balanced = (tpr + tnr) / 2.0
        candidate = (balanced, accuracy, tnr, tpr, threshold, tp, tn, fp, fn)
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    balanced, accuracy, tnr, tpr, threshold, tp, tn, fp, fn = best
    return {
        "threshold": _round(threshold),
        "selection_rule": "max_balanced_accuracy_on_scored_characterization_panel",
        "balanced_accuracy": _round(balanced),
        "accuracy": _round(accuracy),
        "tpr": _round(tpr),
        "tnr": _round(tnr),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def build_artifact_from_scored_candidates(
    candidates: Sequence[ScoredCandidate],
    *,
    operating_thresholds: Mapping[str, float] | None = None,
    category_derivation_method: Mapping[str, Any] | None = None,
    preconditions_checked: Sequence[Mapping[str, Any]] | None = None,
    cited_upstream_artifacts: Mapping[str, Any] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
    random_seed: Sequence[int] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Assemble the Exp 3837 artifact from pre-scored candidate rows."""

    start = time.perf_counter() if started_s is None else float(started_s)
    rows = list(candidates)
    if len(rows) == 0 or len({row.label for row in rows}) < 2:
        return build_blocked_artifact(
            "blocked_candidate_scores_unavailable",
            preconditions_checked=list(preconditions_checked or []),
            cited_upstream_artifacts=cited_upstream_artifacts,
            started_s=start,
            now_s=now_s,
            tests_run=tests_run,
            random_seed=list(random_seed or DEFAULT_RANDOM_SEEDS),
        )

    labels = [row.label for row in rows]
    formal_scores = [row.formal_score for row in rows]
    learned_scores = [row.learned_score for row in rows]
    threshold_payload = _threshold_payload(labels, formal_scores, learned_scores, operating_thresholds)
    category_rows = paired_correctness_by_category(rows, threshold_payload)
    gap_categories = _gap_categories(category_rows)
    blind_spots = _both_wrong_categories(category_rows)
    verdict = _verdict_for_categories(gap_categories)
    finished = time.perf_counter() if now_s is None else float(now_s)
    artifact = _base_artifact(
        verdict=verdict,
        duration_s=_round(max(0.0, finished - start)),
        tests_run=tests_run,
        random_seed=list(random_seed or DEFAULT_RANDOM_SEEDS),
    )
    artifact.update(
        {
            "learned_contribution_by_category": category_rows,
            "formal_core_gap_categories": gap_categories,
            "both_wrong_categories": blind_spots,
            "category_derivation_method": dict(category_derivation_method or {}),
            "n_candidates_scored": len(rows),
            "n_unique_candidates_scored": len({row.question_id + ":" + row.step_text for row in rows}),
            "operating_thresholds": threshold_payload,
            "preconditions_checked": [dict(item) for item in preconditions_checked or []],
            "cited_upstream_artifacts": dict(cited_upstream_artifacts or {}),
            "category_signal_summary": _category_signal_summary(category_rows),
        }
    )
    artifact.update(dict(extra or {}))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def paired_correctness_by_category(
    candidates: Sequence[ScoredCandidate],
    thresholds: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    """Compute the requested four paired correctness cells per category."""

    formal_threshold = float(thresholds["formal_only"]["threshold"])
    learned_threshold = float(thresholds["learned_only"]["threshold"])
    buckets: dict[str, JsonDict] = {}
    for candidate in candidates:
        bucket = buckets.setdefault(candidate.category, _empty_category_row(candidate.category))
        label_is_error = int(candidate.label) == 1
        formal_pred_error = float(candidate.formal_score) >= formal_threshold
        learned_pred_error = float(candidate.learned_score) >= learned_threshold
        formal_correct = formal_pred_error == label_is_error
        learned_correct = learned_pred_error == label_is_error
        bucket["total"] += 1
        if formal_correct and learned_correct:
            bucket["formal_correct_learned_correct"] += 1
        elif (not formal_correct) and learned_correct:
            bucket["formal_wrong_learned_correct"] += 1
            if label_is_error:
                bucket["learned_error_catches"] += 1
            else:
                bucket["learned_clean_rescues"] += 1
        elif formal_correct and not learned_correct:
            bucket["formal_correct_learned_wrong"] += 1
        else:
            bucket["both_wrong"] += 1
            if label_is_error:
                bucket["both_wrong_error_misses"] += 1
            else:
                bucket["both_wrong_clean_false_alarms"] += 1

    total_contrib = sum(int(row["formal_wrong_learned_correct"]) for row in buckets.values())
    ordered = sorted(
        buckets.values(),
        key=lambda row: (
            -int(row["formal_wrong_learned_correct"]),
            COARSE_CATEGORY_ORDER.index(row["category"])
            if row["category"] in COARSE_CATEGORY_ORDER
            else len(COARSE_CATEGORY_ORDER),
            str(row["category"]),
        ),
    )
    for row in ordered:
        row["learned_contribution_share"] = (
            _round(int(row["formal_wrong_learned_correct"]) / total_contrib)
            if total_contrib
            else 0.0
        )
    return ordered


def build_blocked_artifact(
    verdict: str,
    *,
    preconditions_checked: Sequence[Mapping[str, Any]],
    cited_upstream_artifacts: Mapping[str, Any] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
    random_seed: Sequence[int] | None = None,
) -> JsonDict:
    """Return a blocked artifact without fabricated category counts."""

    start = time.perf_counter() if started_s is None else float(started_s)
    finished = time.perf_counter() if now_s is None else float(now_s)
    artifact = _base_artifact(
        verdict=verdict,
        duration_s=_round(max(0.0, finished - start)),
        tests_run=tests_run,
        random_seed=list(random_seed or DEFAULT_RANDOM_SEEDS),
    )
    artifact.update(
        {
            "learned_contribution_by_category": [],
            "formal_core_gap_categories": [],
            "both_wrong_categories": [],
            "category_derivation_method": {},
            "n_candidates_scored": 0,
            "n_unique_candidates_scored": 0,
            "operating_thresholds": {},
            "preconditions_checked": [dict(item) for item in preconditions_checked],
            "cited_upstream_artifacts": dict(cited_upstream_artifacts or {}),
            "category_signal_summary": {"status": "blocked"},
        }
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def load_upstream_artifacts(root: Path | str) -> JsonDict:
    """Load Exp 3826 with SHA256 provenance."""

    root_path = Path(root)
    path = root_path / EXP3826_REL_PATH
    payload = _read_json_object(path)
    return {
        "exp3826": {
            "path": str(EXP3826_REL_PATH),
            "sha256": _sha256_file(path),
            "full_ensemble_auroc": payload.get("full_ensemble_auroc"),
            "formal_only_auroc": payload.get("formal_only_auroc"),
            "learned_only_auroc": payload.get("learned_only_auroc"),
            "n_candidates_scored": payload.get("n_candidates_scored"),
            "verifier_partition": payload.get("verifier_partition"),
        }
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3837 artifact contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    verdict = str(artifact.get("honest_verdict", ""))
    if not (
        verdict.startswith(VERDICT_TOPGAP_PREFIX)
        or verdict == VERDICT_UNIFORM
        or verdict.startswith("blocked_")
    ):
        raise ValueError("honest_verdict is not an accepted Exp 3837 terminal verdict")
    duration = artifact.get("duration_s")
    if not isinstance(duration, int | float) or float(duration) < 0.0:
        raise ValueError("duration_s must be a non-negative number")
    provenance = artifact.get("field_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("field_provenance must be a mapping")
    missing_provenance = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in provenance]
    if missing_provenance:
        raise ValueError(f"field_provenance is missing principles: {missing_provenance}")
    n_scored = artifact.get("n_candidates_scored")
    if not isinstance(n_scored, int) or n_scored < 0:
        raise ValueError("n_candidates_scored must be a non-negative integer")
    if not verdict.startswith("blocked_") and n_scored <= 0:
        raise ValueError("complete artifacts must score at least one candidate")


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash deterministic Exp 3837 characterization fields."""

    payload = {
        "learned_contribution_by_category": artifact.get("learned_contribution_by_category"),
        "formal_core_gap_categories": artifact.get("formal_core_gap_categories"),
        "both_wrong_categories": artifact.get("both_wrong_categories"),
        "category_derivation_method": artifact.get("category_derivation_method"),
        "operating_thresholds": artifact.get("operating_thresholds"),
        "n_candidates_scored": artifact.get("n_candidates_scored"),
        "n_unique_candidates_scored": artifact.get("n_unique_candidates_scored"),
        "cited_upstream_artifacts": artifact.get("cited_upstream_artifacts"),
        "random_seed": artifact.get("random_seed"),
        "inference_substrate": artifact.get("inference_substrate"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def format_breakdown_table(artifact: Mapping[str, Any]) -> str:
    """Return the terminal table requested by the experiment prompt."""

    rows = list(artifact.get("learned_contribution_by_category") or [])
    if not rows:
        return "Category | Total | F&L correct | F wrong/L correct | F correct/L wrong | Both wrong\n"
    lines = [
        "Category | Total | F&L correct | F wrong/L correct | F correct/L wrong | Both wrong",
        "--- | ---: | ---: | ---: | ---: | ---:",
    ]
    for row in rows:
        lines.append(
            f"{row['category']} | {row['total']} | "
            f"{row['formal_correct_learned_correct']} | "
            f"{row['formal_wrong_learned_correct']} | "
            f"{row['formal_correct_learned_wrong']} | {row['both_wrong']}"
        )
    return "\n".join(lines)


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and persist the Exp 3837 result JSON."""

    root_path = Path(root)
    output = output_path if Path(output_path).is_absolute() else root_path / Path(output_path)
    artifact = build_artifact(root_path, tests_run=tests_run)
    validate_artifact(artifact)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _threshold_payload(
    labels: Sequence[int],
    formal_scores: Sequence[float],
    learned_scores: Sequence[float],
    operating_thresholds: Mapping[str, float] | None,
) -> JsonDict:
    if operating_thresholds is None:
        return {
            "formal_only": select_operating_threshold(labels, formal_scores),
            "learned_only": select_operating_threshold(labels, learned_scores),
        }
    return {
        "formal_only": {"threshold": _round(float(operating_thresholds["formal_only"]))},
        "learned_only": {"threshold": _round(float(operating_thresholds["learned_only"]))},
    }


def _confusion(
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float,
) -> tuple[int, int, int, int]:
    tp = tn = fp = fn = 0
    for label, score in zip(labels, scores, strict=True):
        is_error = int(label) == 1
        pred_error = float(score) >= float(threshold)
        if is_error and pred_error:
            tp += 1
        elif is_error:
            fn += 1
        elif pred_error:
            fp += 1
        else:
            tn += 1
    return tp, tn, fp, fn


def _gap_categories(category_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "category": str(row["category"]),
            "formal_wrong_learned_correct": int(row["formal_wrong_learned_correct"]),
            "learned_error_catches": int(row["learned_error_catches"]),
            "learned_contribution_share": float(row["learned_contribution_share"]),
        }
        for row in sorted(
            (row for row in category_rows if int(row["formal_wrong_learned_correct"]) > 0),
            key=lambda item: (
                -int(item["formal_wrong_learned_correct"]),
                -int(item["learned_error_catches"]),
                str(item["category"]),
            ),
        )
    ]


def _both_wrong_categories(category_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "category": str(row["category"]),
            "both_wrong": int(row["both_wrong"]),
            "both_wrong_error_misses": int(row["both_wrong_error_misses"]),
            "both_wrong_clean_false_alarms": int(row["both_wrong_clean_false_alarms"]),
        }
        for row in sorted(
            (row for row in category_rows if int(row["both_wrong"]) > 0),
            key=lambda item: (-int(item["both_wrong"]), str(item["category"])),
        )
    ]


def _category_signal_summary(category_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    counts = [
        int(row["formal_wrong_learned_correct"])
        for row in category_rows
        if int(row["formal_wrong_learned_correct"]) > 0
    ]
    if not counts:
        return {"status": "no_learned_contribution_cells", "uniform": True}
    uniform = len(counts) > 1 and max(counts) == min(counts)
    return {
        "status": "uniform" if uniform else "topgap",
        "uniform": uniform,
        "top_count": max(counts),
        "nonzero_category_count": len(counts),
    }


def _verdict_for_categories(gap_categories: Sequence[Mapping[str, Any]]) -> str:
    counts = [int(row["formal_wrong_learned_correct"]) for row in gap_categories]
    if not counts or (len(counts) > 1 and max(counts) == min(counts)):
        return VERDICT_UNIFORM
    top = _slug(str(gap_categories[0]["category"])).replace("-", "_")
    return f"{VERDICT_TOPGAP_PREFIX}{top}_formal_core_blindspots_documented"


def _empty_category_row(category: str) -> JsonDict:
    return {
        "category": category,
        "total": 0,
        "formal_correct_learned_correct": 0,
        "formal_wrong_learned_correct": 0,
        "formal_correct_learned_wrong": 0,
        "both_wrong": 0,
        "learned_error_catches": 0,
        "learned_clean_rescues": 0,
        "both_wrong_error_misses": 0,
        "both_wrong_clean_false_alarms": 0,
        "learned_contribution_share": 0.0,
    }


def _base_artifact(
    *,
    verdict: str,
    duration_s: float,
    tests_run: Sequence[str] | None,
    random_seed: Sequence[int],
) -> JsonDict:
    return {
        "artifact": "experiment_3837_fover_error_category_learned_contribution",
        "schema": "carnot.fover_error_category_learned_contribution.v1",
        "honest_verdict": verdict,
        "random_seed": list(random_seed),
        "reproducibility_checksum": "",
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_provenance": dict(FIELD_PROVENANCE),
        "tests_run": list(tests_run or []),
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
    }


def _file_check(root: Path, rel_path: Path, resource: str) -> JsonDict:
    path = root / rel_path
    return {"resource": resource, "available": path.is_file(), "detail": str(path)}


def _read_json_rows(path: Path) -> list[JsonDict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [dict(row) for row in payload] if isinstance(payload, list) else []


def _read_json_object(path: Path) -> JsonDict:
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


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "uncategorized"


def _round(value: float) -> float:
    return round(float(value), 6)


__all__ = [
    "DEFAULT_N_EXAMPLES",
    "DEFAULT_RANDOM_SEEDS",
    "FIELD_PROVENANCE",
    "OUTPUT_REL_PATH",
    "REQUIRED_ARTIFACT_FIELDS",
    "ScoredCandidate",
    "build_artifact",
    "build_artifact_from_scored_candidates",
    "build_blocked_artifact",
    "category_derivation_method",
    "check_preconditions",
    "coarse_text_category",
    "derive_category",
    "format_breakdown_table",
    "load_upstream_artifacts",
    "paired_correctness_by_category",
    "reproducibility_checksum",
    "score_exp3826_candidate_panels",
    "select_operating_threshold",
    "validate_artifact",
    "write_artifact",
]
