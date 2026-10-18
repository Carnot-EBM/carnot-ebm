"""Exp 5018: replicate uPRM from the shared MuSR logprob cache.

Spec refs: REQ-VERIFY-5018, SCENARIO-VERIFY-5018.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot import experiment_5004_uprm_replication as uprm  # noqa: E402
from carnot import moat_benchmark_harness as harness  # noqa: E402
from carnot.moat_benchmark_harness import (  # noqa: E402
    DEFAULT_RANDOM_SEED,
    OracleDistinctnessError,
    evaluate_verifier,
)


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
AuditRunner = Callable[[Path], JsonDict]
SummaryRunner = Callable[[Path], int]
Clock = Callable[[], float]

EXPERIMENT_ID = 5018
EXPERIMENT_NAME = "experiment_5018_uprm_replication_v2"
RESULT_RELATIVE_PATH = "results/experiment_5018_uprm_replication_v2.json"
SHARED_CACHE_RELATIVE_PATH = "results/experiment_5016_shared_logprob_candidate_cache_musr.jsonl"
LC_ERD_FALLBACK_RELATIVE_PATHS = (
    "results/experiment_5018_lc_erd_fallback_inputs.jsonl",
    "results/lc_erd_consistency_musr_candidates.jsonl",
)
CACHE_ROW_SCHEMA = "carnot.shared_logprob_candidate_cache.row.v1"
MODEL_HF_ID = "unsloth/gemma-4-12B-it-GGUF"
MODEL_NAME = "gemma-4-12B-it-GGUF"
CORPUS = harness.MUSR_CORPUS_NAME
SPEC_REFS = ["REQ-VERIFY-5018", "SCENARIO-VERIFY-5018"]
RANDOM_SEED = DEFAULT_RANDOM_SEED
DEFAULT_K = 5
DEFAULT_LIMIT = 200

METHODOLOGY_NOTE = (
    "arXiv:2605.10158 uPRM scores a candidate first-error position j for "
    "trajectory steps y_1..y_T as S(j)=1[j<=T] log p^-_j + sum_{t<j} "
    "log p^+_t, where p^+_t and p^-_t are the generator LLM next-token "
    "probabilities of '+' and '-' marker tokens after step t, renormalized "
    "over {+,-}. This v2 runner consumes the Exp 5016 cached generator "
    "logprobs, never gold, to score each candidate by the mean no-error "
    "log-likelihood S(T+1)/T; gold is read only after selection for evaluation, "
    "so the selector is unsupervised and oracle-distinct."
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; a win is success_uprm_beats_sc_musr_<delta>, a clean "
            "null is complete_uprm_no_win_musr_<delta>_ci_incl_0."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "false -- uPRM is UNSUPERVISED (scored from the generator's own "
            "logprobs); it never reads gold (must pass check_circular_moat_overclaim)."
        )
    },
    "headroom_present": {
        "principle": (
            "true required for an informative result vs the GENUINE tuned-SC "
            "(FALSE_NEGATIVE_RISK guard)."
        )
    },
    "uprm_selection_accuracy": {
        "principle": (
            "the oracle-distinct selection accuracy of the uPRM process score "
            "(the headline)."
        )
    },
    "genuine_tuned_sc_accuracy": {
        "principle": (
            "the B1 GENUINE K-way tuned-SC (NOT k=1) -- the honest baseline to beat."
        )
    },
    "delta_vs_tuned_sc": {
        "principle": (
            "uprm_selection_accuracy - genuine_tuned_sc_accuracy; the paper reports up to +0.069."
        )
    },
    "paired_ci95": {
        "principle": "paired bootstrap CI95 of the delta; a win requires CI95 excluding 0."
    },
    "mcnemar_p": {"principle": "McNemar paired p; a win requires p<0.05."},
    "scoring_path": {
        "principle": (
            "uprm_logprob (primary) or lc_erd_consistency (fallback) -- which path "
            "scored; the LC-ERD path keeps the no-model-id audit."
        )
    },
    "uprm_score_methodology_note": {
        "principle": (
            "the exact arXiv:2605.10158 first-error formula (so a third party can "
            "replicate) + the unsupervised-not-circular justification."
        )
    },
    "n_questions": {"principle": ">=200 for the headline delta (sample-size rigor)."},
    "model_specs": {
        "principle": (
            "the cached-candidate generator (gemma-4-12B-it-GGUF) -- the methodology stamp."
        )
    },
    "inference_substrate": {
        "principle": (
            "verifier_ensemble_against_cached_candidates (scores the B2 cached "
            "candidates; 1s floor) -- declare live_llm_inference ONLY if it "
            "regenerates fresh."
        )
    },
    "random_seed": {"principle": "determinism for the bootstrap."},
    "preconditions_checked": {
        "principle": (
            "records the B2-cache / LC-ERD-fallback checks; a missing cache+fallback "
            "emits blocked_."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "oracle_at_k",
    "candidate_cache_path",
    "oracle_distinctness_enforced",
    "adversarial_verify_clean",
    "adversarial_verify_flags",
    "summarize_artifact_exit_code",
    "duration_s",
    "field_principles",
    "reproducibility_checksum",
)


@dataclass(frozen=True)
class PreconditionCheck:
    """One checked resource recorded before any uPRM result claim."""

    resource: str
    available: bool
    detail: str
    path: str | None = None

    def as_dict(self) -> JsonDict:
        payload: JsonDict = {
            "resource": self.resource,
            "available": bool(self.available),
            "detail": self.detail,
        }
        if self.path is not None:
            payload["path"] = self.path
        return payload


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[JsonDict]:
    if not path.exists():
        return []
    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        loaded = json.loads(line)
        if isinstance(loaded, dict):
            rows.append(loaded)
    return rows


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _has_marker_pair(marker_row: Any) -> bool:
    if not isinstance(marker_row, Mapping):
        return False
    markers = {str(token).strip() for token in marker_row if _finite_number(marker_row[token])}
    return {"-", "+"}.issubset(markers)


def _candidate_has_uprm_logprobs(candidate: JsonMap) -> bool:
    token_logprobs = candidate.get("token_logprobs")
    marker_rows = candidate.get("uprm_marker_logprobs")
    return (
        isinstance(token_logprobs, list)
        and bool(token_logprobs)
        and all(_finite_number(value) for value in token_logprobs)
        and isinstance(marker_rows, list)
        and bool(marker_rows)
        and all(_has_marker_pair(row) for row in marker_rows)
    )


def _candidate_answer(candidate: JsonMap) -> str:
    return str(candidate.get("answer") or candidate.get("final_answer") or "").strip()


def _normalise_candidate(row_id: str, candidate: JsonMap) -> JsonDict:
    cache_index = candidate.get("cache_index", candidate.get("index", 0))
    return {
        **dict(candidate),
        "candidate_id": str(candidate.get("candidate_id") or f"{row_id}/cached-{cache_index}"),
        "answer": _candidate_answer(candidate),
        "cache_index": cache_index,
        "source": "exp5016_shared_logprob_candidate_cache",
    }


def load_uprm_cache_rows(
    path: Path,
    *,
    min_questions: int = DEFAULT_LIMIT,
    k_candidates: int = DEFAULT_K,
    limit: int | None = None,
) -> list[JsonDict]:
    """Load complete B2 cache rows with candidate token and uPRM marker logprobs."""

    rows: list[JsonDict] = []
    for payload in _read_jsonl(path):
        if payload.get("schema") != CACHE_ROW_SCHEMA or payload.get("corpus") != CORPUS:
            continue
        candidates = [
            _normalise_candidate(str(payload.get("row_id") or len(rows)), candidate)
            for candidate in payload.get("candidates", [])
            if isinstance(candidate, Mapping)
            and _candidate_answer(candidate)
            and _candidate_has_uprm_logprobs(candidate)
        ]
        if len(candidates) < k_candidates or not str(payload.get("gold") or ""):
            continue
        rows.append(
            {
                "row_id": str(payload.get("row_id") or f"musr:{len(rows)}"),
                "corpus": CORPUS,
                "question": str(payload.get("question") or ""),
                "context": str(payload.get("context") or ""),
                "choices": list(payload.get("choices") or []),
                "gold": str(payload.get("gold") or ""),
                "candidate_cache_path": path.as_posix(),
                "candidates": candidates[:k_candidates],
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    if len(rows) < min_questions:
        raise RuntimeError(
            f"only {len(rows)} uPRM-ready B2 cache rows available; need {min_questions}"
        )
    return rows


def cache_precondition(
    path: Path,
    *,
    min_questions: int = DEFAULT_LIMIT,
    k_candidates: int = DEFAULT_K,
) -> PreconditionCheck:
    try:
        rows = load_uprm_cache_rows(path, min_questions=min_questions, k_candidates=k_candidates)
    except Exception as exc:
        return PreconditionCheck(
            "b2_logprob_cache",
            False,
            f"{type(exc).__name__}: {exc}",
            path.as_posix(),
        )
    return PreconditionCheck(
        "b2_logprob_cache",
        True,
        f"{len(rows)} MuSR rows x K>={k_candidates} with token and marker logprobs",
        path.as_posix(),
    )


def lc_erd_fallback_precondition(root: Path) -> PreconditionCheck:
    paths = [root / relative for relative in LC_ERD_FALLBACK_RELATIVE_PATHS]
    existing = [path for path in paths if path.exists()]
    detail = (
        f"{len(existing)} LC-ERD fallback input file(s) available"
        if existing
        else "no LC-ERD fallback input files found"
    )
    return PreconditionCheck(
        "lc_erd_fallback_inputs",
        bool(existing),
        detail,
        ",".join(path.as_posix() for path in existing) or paths[0].as_posix(),
    )


def check_preconditions(
    *,
    root: Path,
    cache_path: Path,
    min_questions: int = DEFAULT_LIMIT,
    k_candidates: int = DEFAULT_K,
) -> tuple[list[PreconditionCheck], list[JsonDict]]:
    cache_check = cache_precondition(
        cache_path,
        min_questions=min_questions,
        k_candidates=k_candidates,
    )
    fallback_check = lc_erd_fallback_precondition(root)
    rows = (
        load_uprm_cache_rows(
            cache_path,
            min_questions=min_questions,
            k_candidates=k_candidates,
            limit=min_questions,
        )
        if cache_check.available
        else []
    )
    return [cache_check, fallback_check], rows


def first_missing_resource(checks: Sequence[PreconditionCheck]) -> str | None:
    cache_missing = checks and not checks[0].available
    fallback_missing = len(checks) > 1 and not checks[1].available
    if cache_missing and fallback_missing:
        return checks[0].resource
    return None


def prepare_rows_with_uprm_scores(rows: Sequence[JsonMap]) -> list[JsonDict]:
    """Attach uPRM process scores to cached candidates without reading gold."""

    return uprm.prepare_rows_with_uprm_scores(rows)


def _uprm_energy(candidate: Mapping[str, Any]) -> float:
    return -float(candidate.get("uprm_process_score"))


def evaluate_uprm_rows(
    rows: Sequence[JsonMap],
    *,
    seed: int = RANDOM_SEED,
    bootstrap_samples: int = 2000,
) -> JsonDict:
    """Evaluate uPRM-scored rows against genuine tuned self-consistency."""

    return evaluate_verifier(
        rows,
        scorer=_uprm_energy,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
        headroom_threshold=harness.HEADROOM_THRESHOLD,
    )


def _oracle_distinctness_enforced(rows: Sequence[JsonMap]) -> bool:
    try:
        evaluate_verifier(rows, scorer=lambda candidate: candidate["gold"], bootstrap_samples=8)
    except OracleDistinctnessError:
        return True
    return False  # pragma: no cover - indicates the shared harness regressed


def _slug_corpus(corpus: str) -> str:
    return (
        "musr"
        if corpus.lower().startswith("musr")
        else re.sub(r"[^a-z0-9]+", "_", corpus.lower()).strip("_")
    )


def _format_delta(delta: float) -> str:
    return f"{delta:+.3f}".replace("+", "plus_").replace("-", "minus_").replace(".", "p")


def _ci_includes_zero(ci95: Sequence[float]) -> bool:
    return len(ci95) == 2 and float(ci95[0]) <= 0.0 <= float(ci95[1])


def reproducibility_checksum(payload: JsonMap) -> str:
    return "sha256:" + hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def _base_artifact(
    *,
    honest_verdict: str,
    preconditions_checked: Sequence[JsonDict],
    cache_path: Path,
    duration_s: float,
) -> JsonDict:
    blocked = honest_verdict.startswith("blocked_")
    return {
        "schema": "carnot.experiment_5018_uprm_replication_v2.v1",
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": (REPO_ROOT / RESULT_RELATIVE_PATH).as_posix(),
        "honest_verdict": honest_verdict,
        "verifier_is_oracle": False,
        "headroom_present": False,
        "uprm_selection_accuracy": None,
        "genuine_tuned_sc_accuracy": None,
        "delta_vs_tuned_sc": None,
        "paired_ci95": None,
        "mcnemar_p": None,
        "scoring_path": "blocked" if blocked else "uprm_logprob",
        "uprm_score_methodology_note": METHODOLOGY_NOTE,
        "n_questions": 0,
        "oracle_at_k": None,
        "model_specs": {
            "cached_candidate_generator": MODEL_NAME,
            "generator_hf_id": MODEL_HF_ID,
            "candidate_cache_schema": CACHE_ROW_SCHEMA,
            "requires_token_logprobs": True,
            "requires_uprm_marker_logprobs": True,
        },
        "inference_substrate": "precondition_check_only",
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum(
            {"honest_verdict": honest_verdict, "cache_path": cache_path.as_posix()}
        ),
        "preconditions_checked": list(preconditions_checked),
        "candidate_cache_path": cache_path.as_posix(),
        "oracle_distinctness_enforced": False,
        "adversarial_verify_clean": False,
        "adversarial_verify_flags": [],
        "summarize_artifact_exit_code": None,
        "duration_s": round(float(duration_s), 6),
        "field_principles": FIELD_PRINCIPLES,
    }


def build_blocked_artifact(
    *,
    missing_resource: str,
    preconditions_checked: Sequence[JsonDict],
    cache_path: Path,
    duration_s: float,
    error: str | None = None,
) -> JsonDict:
    artifact = _base_artifact(
        honest_verdict=f"blocked_{missing_resource}",
        preconditions_checked=preconditions_checked,
        cache_path=cache_path,
        duration_s=duration_s,
    )
    if error:
        artifact["blocked_error"] = error[:500]
    return artifact


def build_skeleton_artifact(
    *,
    preconditions_checked: Sequence[JsonDict],
    cache_path: Path,
    duration_s: float,
) -> JsonDict:
    artifact = _base_artifact(
        honest_verdict="running_uprm_v2_schema_skeleton",
        preconditions_checked=preconditions_checked,
        cache_path=cache_path,
        duration_s=duration_s,
    )
    artifact["deliverable_stage"] = "schema_skeleton"
    return artifact


def build_complete_artifact(
    *,
    evaluation: JsonDict,
    preconditions_checked: Sequence[JsonDict],
    cache_path: Path,
    duration_s: float,
) -> JsonDict:
    uprm_accuracy = float(evaluation["verifier"]["accuracy"])
    tuned_accuracy = float(evaluation["tuned_self_consistency"]["accuracy"])
    delta = float(evaluation["verifier_minus_tuned_sc_delta"])
    ci95 = [float(value) for value in evaluation["verifier_minus_tuned_sc_ci95"]]
    mcnemar_p = float(evaluation["mcnemar_p"])
    headroom_present = bool(evaluation["headroom_present"])
    corpus_slug = _slug_corpus(CORPUS)
    verdict_delta = _format_delta(delta)
    win = delta > 0.0 and ci95[0] > 0.0 and mcnemar_p < 0.05 and headroom_present
    if win:
        honest_verdict = f"success_uprm_beats_sc_{corpus_slug}_{verdict_delta}"
    elif _ci_includes_zero(ci95):
        honest_verdict = f"complete_uprm_no_win_{corpus_slug}_{verdict_delta}_ci_incl_0"
    else:
        honest_verdict = (
            f"complete_uprm_no_win_{corpus_slug}_{verdict_delta}_mcnemar_or_headroom_gate"
        )
    artifact = _base_artifact(
        honest_verdict=honest_verdict,
        preconditions_checked=preconditions_checked,
        cache_path=cache_path,
        duration_s=duration_s,
    )
    artifact.update(
        {
            "headroom_present": headroom_present,
            "uprm_selection_accuracy": round(uprm_accuracy, 6),
            "genuine_tuned_sc_accuracy": round(tuned_accuracy, 6),
            "delta_vs_tuned_sc": round(delta, 6),
            "paired_ci95": ci95,
            "mcnemar_p": mcnemar_p,
            "scoring_path": "uprm_logprob",
            "n_questions": int(evaluation["n_rows"]),
            "oracle_at_k": float(evaluation["oracle_at_k"]),
            "model_specs": {
                **artifact["model_specs"],
                "candidate_cache_path": cache_path.as_posix(),
                "score_formula": "uPRM Eq.6 first-error marker score",
                "candidate_aggregation": "mean no-error log-likelihood S(T+1)/T",
                "tuned_self_consistency_config": evaluation["tuned_self_consistency"]["config"],
            },
            "inference_substrate": "verifier_ensemble_against_cached_candidates",
            "reproducibility_checksum": reproducibility_checksum(
                {
                    "model": MODEL_HF_ID,
                    "cache_path": cache_path.as_posix(),
                    "evaluation": evaluation,
                    "seed": RANDOM_SEED,
                }
            ),
            "oracle_distinctness_enforced": True,
            "evaluation": evaluation,
        }
    )
    return artifact


def _compact_adversarial_flags(report: JsonDict) -> list[JsonDict]:
    if "reports" in report and isinstance(report["reports"], list) and report["reports"]:
        report = report["reports"][0]
    flags = report.get("flags", []) if isinstance(report, dict) else []
    return [flag for flag in flags if isinstance(flag, dict)]


def _audit_is_clean(report: JsonDict) -> bool:
    if "flagged_count" in report:
        return int(report.get("flagged_count") or 0) == 0
    if "flag_count" in report:
        return int(report.get("flag_count") or 0) == 0
    return not _compact_adversarial_flags(report)


def run_adversarial_verify(path: Path) -> JsonDict:  # pragma: no cover - subprocess-adjacent glue
    script_path = REPO_ROOT / "scripts" / "adversarial_verify.py"
    spec = importlib.util.spec_from_file_location("carnot_adversarial_verify_5018", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load scripts/adversarial_verify.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_artifact(path)


def run_summarize_artifact(path: Path) -> int:  # pragma: no cover - reviewer CLI glue
    script_path = REPO_ROOT / "scripts" / "summarize_artifact.py"
    spec = importlib.util.spec_from_file_location("carnot_summarize_artifact_5018", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load scripts/summarize_artifact.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return int(module.summarize(path))


def attach_audit(
    artifact: JsonDict,
    *,
    artifact_path: Path,
    audit_runner: AuditRunner,
    summary_runner: SummaryRunner,
) -> JsonDict:
    write_json(artifact_path, artifact)
    audit_report = audit_runner(artifact_path)
    updated = dict(artifact)
    updated["adversarial_verify_clean"] = _audit_is_clean(audit_report)
    updated["adversarial_verify_flags"] = _compact_adversarial_flags(audit_report)
    updated["adversarial_verify_report"] = audit_report
    updated["summarize_artifact_exit_code"] = int(summary_runner(artifact_path))
    write_json(artifact_path, updated)
    return updated


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(field)
    if artifact.get("spec_refs") != SPEC_REFS:
        errors.append("spec_refs")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    if artifact.get("scoring_path") not in {"blocked", "uprm_logprob", "lc_erd_consistency"}:
        errors.append("scoring_path")
    ci95 = artifact.get("paired_ci95")
    if ci95 is not None and (
        not isinstance(ci95, list)
        or len(ci95) != 2
        or not all(isinstance(value, (int, float)) for value in ci95)
    ):
        errors.append("paired_ci95")
    for field in ("headroom_present", "oracle_distinctness_enforced", "adversarial_verify_clean"):
        if not isinstance(artifact.get(field), bool):
            errors.append(field)
    for field in ("uprm_selection_accuracy", "genuine_tuned_sc_accuracy", "oracle_at_k"):
        value = artifact.get(field)
        if value is not None and not (
            isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0
        ):
            errors.append(field)
    if artifact.get("delta_vs_tuned_sc") is not None and not isinstance(
        artifact.get("delta_vs_tuned_sc"), (int, float)
    ):
        errors.append("delta_vs_tuned_sc")
    if artifact.get("mcnemar_p") is not None and not (
        isinstance(artifact.get("mcnemar_p"), (int, float))
        and 0.0 <= float(artifact.get("mcnemar_p")) <= 1.0
    ):
        errors.append("mcnemar_p")
    if not isinstance(artifact.get("preconditions_checked"), list):
        errors.append("preconditions_checked")
    if not isinstance(artifact.get("model_specs"), dict):
        errors.append("model_specs")
    if not str(artifact.get("honest_verdict", "")).startswith(
        ("blocked_", "running_", "complete_", "success_")
    ):
        errors.append("honest_verdict")
    if not isinstance(artifact.get("uprm_score_methodology_note"), str) or not artifact.get(
        "uprm_score_methodology_note"
    ):
        errors.append("uprm_score_methodology_note")
    return sorted(set(errors))


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    audit_runner: AuditRunner = run_adversarial_verify,
    summary_runner: SummaryRunner = run_summarize_artifact,
    min_questions: int = DEFAULT_LIMIT,
    k_candidates: int = DEFAULT_K,
    bootstrap_samples: int = 2000,
    random_seed: int = RANDOM_SEED,
    now: Clock = time.time,
    write: bool = True,
) -> JsonDict:
    root = Path(root)
    artifact_path = Path(artifact_path) if artifact_path else root / RESULT_RELATIVE_PATH
    cache_path = root / SHARED_CACHE_RELATIVE_PATH
    start = float(now())

    if write:
        write_json(
            artifact_path,
            build_skeleton_artifact(
                preconditions_checked=[],
                cache_path=cache_path,
                duration_s=float(now()) - start,
            ),
        )

    checks, rows = check_preconditions(
        root=root,
        cache_path=cache_path,
        min_questions=min_questions,
        k_candidates=k_candidates,
    )
    preconditions = [check.as_dict() for check in checks]
    missing = first_missing_resource(checks)
    if missing is not None:
        artifact = build_blocked_artifact(
            missing_resource=missing,
            preconditions_checked=preconditions,
            cache_path=cache_path,
            duration_s=float(now()) - start,
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact

    try:
        prepared_rows = prepare_rows_with_uprm_scores(rows)
        if not _oracle_distinctness_enforced(prepared_rows):
            raise OracleDistinctnessError("shared harness did not block gold access")
        evaluation = evaluate_uprm_rows(
            prepared_rows,
            seed=random_seed,
            bootstrap_samples=bootstrap_samples,
        )
    except OracleDistinctnessError as exc:
        artifact = build_blocked_artifact(
            missing_resource="oracle_distinctness_violation",
            preconditions_checked=preconditions,
            cache_path=cache_path,
            duration_s=float(now()) - start,
            error=str(exc),
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact
    except Exception as exc:
        artifact = build_blocked_artifact(
            missing_resource="uprm_scoring_error",
            preconditions_checked=preconditions,
            cache_path=cache_path,
            duration_s=float(now()) - start,
            error=f"{type(exc).__name__}: {exc}",
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact

    artifact = build_complete_artifact(
        evaluation=evaluation,
        preconditions_checked=preconditions,
        cache_path=cache_path,
        duration_s=float(now()) - start,
    )
    if write:
        artifact = attach_audit(
            artifact,
            artifact_path=artifact_path,
            audit_runner=audit_runner,
            summary_runner=summary_runner,
        )
    return artifact


def main() -> int:  # pragma: no cover - exercised by requested entrypoint
    artifact = run()
    errors = artifact_schema_errors(artifact)
    path = REPO_ROOT / RESULT_RELATIVE_PATH
    print(f"{path}: {artifact.get('honest_verdict')}")
    if errors:
        print(f"schema_errors={errors}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
