"""Exp 5004: replicate uPRM as an oracle-distinct selector.

Spec refs: REQ-VERIFY-5004, SCENARIO-VERIFY-5004.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot import moat_benchmark_harness as harness  # noqa: E402
from carnot.moat_benchmark_harness import (  # noqa: E402
    DEFAULT_RANDOM_SEED,
    GenerationConfig,
    OracleDistinctnessError,
    evaluate_verifier,
)


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
AuditRunner = Callable[[Path], JsonDict]
SummaryRunner = Callable[[Path], int]
Clock = Callable[[], float]

EXPERIMENT_ID = 5004
RESULT_RELATIVE_PATH = "results/experiment_5004_uprm_replication.json"
CANDIDATE_CACHE_RELATIVE_PATH = "results/experiment_5004_uprm_candidates_musr.jsonl"
MODEL_HF_ID = "unsloth/gemma-4-12B-it-GGUF"
MODEL_NAME = "gemma-4-12B-it-GGUF"
CORPUS = harness.MUSR_CORPUS_NAME
SPEC_REFS = ["REQ-VERIFY-5004", "SCENARIO-VERIFY-5004"]
RANDOM_SEED = DEFAULT_RANDOM_SEED
DEFAULT_K = 8
DEFAULT_LIMIT = 200
DEFAULT_SERVER_PORT = 8919
MARKER_LOGPROBS_TOP_K = 20
FRESH_GENERATION_OPT_IN_ENV = "CARNOT_UPRM_ENABLE_FRESH_GENERATION"

METHODOLOGY_NOTE = (
    "arXiv:2605.10158 uPRM scores a candidate first-error position j for "
    "trajectory steps y_1..y_T as S(j)=1[j<=T] log p^-_j + sum_{t<j} "
    "log p^+_t, where p^+_t and p^-_t are the generator LLM next-token "
    "probabilities of '+' and '-' marker tokens after step t, renormalized "
    "over {+,-}. This runner uses the generator's own logprob/top-logprob "
    "telemetry, never gold, to score each candidate by the mean no-error "
    "log-likelihood S(T+1)/T as a direct oracle-distinct selector; gold is "
    "read only after selection for evaluation."
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; a win is success_uprm_beats_sc_<corpus>_<delta>, "
            "a null is complete_uprm_no_win_<corpus>_ci_incl_0."
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
            "true required for an informative result ((oracle@K - tuned_sc) >= 0.10, "
            "flips>0); FALSE_NEGATIVE_RISK guard."
        )
    },
    "uprm_selection_accuracy": {
        "principle": (
            "the oracle-distinct selection accuracy of the uPRM process score (the headline)."
        )
    },
    "tuned_sc_accuracy": {"principle": "the TUNED-SC baseline (headroom-control)."},
    "delta_vs_tuned_sc": {
        "principle": (
            "uprm_selection_accuracy - tuned_sc_accuracy; the paper reports up to +0.069."
        )
    },
    "paired_ci95": {
        "principle": "paired bootstrap CI95 of the delta; a win requires CI95 excluding 0."
    },
    "mcnemar_p": {"principle": "McNemar paired p; a win requires p<0.05."},
    "uprm_score_methodology_note": {
        "principle": (
            "the exact paper formulation of the next-token-prob first-error score "
            "(so a third party can replicate); the unsupervised-not-circular justification."
        )
    },
    "corpus": {
        "principle": "the headroom-present oracle-distinct corpus (MuSR / harder-math / GPQA slice)."
    },
    "n_questions": {"principle": ">=200 for the headline delta (sample-size rigor)."},
    "model_specs": {
        "principle": (
            "gemma-4-12B-it-GGUF (the SOTA generator providing logprobs) -- the methodology stamp."
        )
    },
    "inference_substrate": {
        "principle": "live_llm_inference (live generation with logprobs; >=60s floor)."
    },
    "random_seed": {"principle": "determinism for generation + bootstrap."},
    "preconditions_checked": {
        "principle": "records GGUF-cached/logprob/corpus checks; a missing resource emits blocked_."
    },
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "verifier_is_oracle",
    "headroom_present",
    "uprm_selection_accuracy",
    "tuned_sc_accuracy",
    "delta_vs_tuned_sc",
    "paired_ci95",
    "mcnemar_p",
    "uprm_score_methodology_note",
    "corpus",
    "n_questions",
    "oracle_at_k",
    "model_specs",
    "inference_substrate",
    "random_seed",
    "reproducibility_checksum",
    "preconditions_checked",
    "oracle_distinctness_enforced",
    "adversarial_verify_clean",
    "adversarial_verify_flags",
    "duration_s",
    "field_principles",
    "spec_refs",
)


class UprmScoringError(RuntimeError):
    """Raised when uPRM marker probabilities are absent or malformed."""


@dataclass(frozen=True)
class PreconditionCheck:
    """One resource check recorded before any uPRM result claim."""

    resource: str
    available: bool
    detail: str
    path: str | None = None

    def as_dict(self) -> JsonDict:
        out: JsonDict = {
            "resource": self.resource,
            "available": bool(self.available),
            "detail": self.detail,
        }
        if self.path is not None:
            out["path"] = self.path
        return out


@dataclass(frozen=True)
class MarkerLogprobs:
    """Renormalized log-probabilities over the paper's '+'/'-' markers."""

    log_p_plus: float
    log_p_minus: float


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[JsonDict]:
    if not path.exists():
        return []
    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: Sequence[JsonMap]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, sort_keys=True) for row in rows)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _top_logprob_row(raw: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(raw, Mapping):
        for token, logprob in raw.items():
            value = _number(logprob)
            if value is not None:
                out[str(token)] = value
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for item in raw:
            if isinstance(item, Mapping) and "token" in item:
                value = _number(item.get("logprob"))
                if value is not None:
                    out[str(item["token"])] = value
    return out


def parse_llama_completion_payload(payload: JsonMap) -> JsonDict:
    """Extract text, chosen-token logprobs, and top-logprob rows from llama-server output."""

    text = str(payload.get("content") or "")
    probabilities = payload.get("completion_probabilities")
    if isinstance(probabilities, Sequence) and not isinstance(probabilities, (str, bytes)):
        token_logprobs: list[float] = []
        top_logprobs: list[dict[str, float]] = []
        for item in probabilities:
            if not isinstance(item, Mapping):
                continue
            value = _number(item.get("logprob"))
            if value is not None:
                token_logprobs.append(value)
            row = _top_logprob_row(item.get("top_logprobs"))
            if row:
                top_logprobs.append(row)
        return {"text": text, "token_logprobs": token_logprobs, "top_logprobs": top_logprobs}

    choices = payload.get("choices")
    if isinstance(choices, Sequence) and choices and isinstance(choices[0], Mapping):
        choice = choices[0]
        text = str(choice.get("text") or text)
        logprobs = choice.get("logprobs")
        if isinstance(logprobs, Mapping):
            token_logprobs = [
                value
                for raw in logprobs.get("token_logprobs") or []
                if (value := _number(raw)) is not None
            ]
            top_logprobs = [_top_logprob_row(row) for row in logprobs.get("top_logprobs") or []]
            return {
                "text": text,
                "token_logprobs": token_logprobs,
                "top_logprobs": [row for row in top_logprobs if row],
            }
    return {"text": text, "token_logprobs": [], "top_logprobs": []}


def _marker_logprob(top_logprobs: Mapping[str, float], marker: str) -> float | None:
    hits = [
        float(logprob)
        for token, logprob in top_logprobs.items()
        if str(token).strip() == marker and math.isfinite(float(logprob))
    ]
    return max(hits) if hits else None


def renormalized_marker_logprobs(top_logprobs: Mapping[str, float]) -> MarkerLogprobs:
    """Return uPRM marker probabilities renormalized over `{+,-}`."""

    plus = _marker_logprob(top_logprobs, "+")
    minus = _marker_logprob(top_logprobs, "-")
    if plus is None or minus is None:
        raise UprmScoringError("top_logprobs must include both '+' and '-' marker tokens")
    normalizer = max(plus, minus) + math.log(
        math.exp(plus - max(plus, minus)) + math.exp(minus - max(plus, minus))
    )
    return MarkerLogprobs(log_p_plus=plus - normalizer, log_p_minus=minus - normalizer)


def uprm_first_error_log_score(
    marker_top_logprobs: Sequence[Mapping[str, float]],
    *,
    first_error_position: int,
) -> float:
    """Compute Eq. 6 first-error score for a 1-indexed position j."""

    if first_error_position < 1 or first_error_position > len(marker_top_logprobs) + 1:
        raise UprmScoringError("first_error_position must be in 1..T+1")
    markers = [renormalized_marker_logprobs(row) for row in marker_top_logprobs]
    total = sum(marker.log_p_plus for marker in markers[: first_error_position - 1])
    if first_error_position <= len(markers):
        total += markers[first_error_position - 1].log_p_minus
    return float(total)


def uprm_candidate_process_score(marker_top_logprobs: Sequence[Mapping[str, float]]) -> float:
    """Aggregate the first-error score into a candidate-level no-error process score."""

    if not marker_top_logprobs:
        raise UprmScoringError("uprm_marker_logprobs must contain at least one step")
    no_error = uprm_first_error_log_score(
        marker_top_logprobs,
        first_error_position=len(marker_top_logprobs) + 1,
    )
    return float(no_error / len(marker_top_logprobs))


def prepare_rows_with_uprm_scores(rows: Sequence[JsonMap]) -> list[JsonDict]:
    """Attach uPRM process scores to every candidate without reading gold."""

    prepared: list[JsonDict] = []
    for row in rows:
        copied = dict(row)
        candidates: list[JsonDict] = []
        for candidate in row.get("candidates", []):
            marker_rows = candidate.get("uprm_marker_logprobs")
            if not isinstance(marker_rows, Sequence) or isinstance(marker_rows, (str, bytes)):
                raise UprmScoringError("uprm_marker_logprobs missing from candidate")
            score = uprm_candidate_process_score(marker_rows)  # type: ignore[arg-type]
            scored = dict(candidate)
            scored["uprm_process_score"] = round(score, 12)
            scored["uprm_process_energy"] = round(-score, 12)
            scored["uprm_first_error_scores"] = [
                round(
                    uprm_first_error_log_score(marker_rows, first_error_position=position),
                    12,
                )
                for position in range(1, len(marker_rows) + 2)
            ]
            candidates.append(scored)
        copied["candidates"] = candidates
        prepared.append(copied)
    return prepared


def _uprm_energy(candidate: Mapping[str, Any]) -> float:
    score = _number(candidate.get("uprm_process_score"))
    return -score if score is not None else math.inf


def evaluate_uprm_rows(
    rows: Sequence[JsonMap],
    *,
    seed: int = RANDOM_SEED,
    bootstrap_samples: int = 2000,
) -> JsonDict:
    """Evaluate uPRM-scored rows against tuned self-consistency."""

    return evaluate_verifier(
        rows,
        scorer=_uprm_energy,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
        headroom_threshold=harness.HEADROOM_THRESHOLD,
    )


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
    duration_s: float,
) -> JsonDict:
    blocked = honest_verdict.startswith("blocked_")
    return {
        "experiment": "experiment_5004_uprm_replication",
        "schema": "carnot.experiment_5004_uprm_replication.v1",
        "honest_verdict": honest_verdict,
        "verifier_is_oracle": False,
        "headroom_present": False,
        "uprm_selection_accuracy": None,
        "tuned_sc_accuracy": None,
        "delta_vs_tuned_sc": None,
        "paired_ci95": None,
        "mcnemar_p": None,
        "uprm_score_methodology_note": METHODOLOGY_NOTE,
        "corpus": CORPUS,
        "n_questions": 0,
        "oracle_at_k": None,
        "model_specs": {
            "generator_model": MODEL_NAME,
            "generator_hf_id": MODEL_HF_ID,
            "requires_token_logprobs": True,
            "requires_top_logprobs_for_markers": True,
        },
        "inference_substrate": "precondition_check_only" if blocked else "live_llm_inference",
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum(
            {"honest_verdict": honest_verdict, "preconditions": list(preconditions_checked)}
        ),
        "preconditions_checked": list(preconditions_checked),
        "oracle_distinctness_enforced": False,
        "adversarial_verify_clean": False,
        "adversarial_verify_flags": [],
        "summarize_artifact_exit_code": None,
        "duration_s": round(float(duration_s), 6),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "candidate_cache_path": None,
        "n_candidates_per_question": DEFAULT_K,
        "evaluation": {},
    }


def build_blocked_artifact(
    *,
    missing_resource: str,
    preconditions_checked: Sequence[JsonDict],
    duration_s: float,
    error: str | None = None,
) -> JsonDict:
    artifact = _base_artifact(
        honest_verdict=f"blocked_{missing_resource}",
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
    )
    if error:
        artifact["blocked_error"] = error[:500]
    return artifact


def build_skeleton_artifact(
    *,
    preconditions_checked: Sequence[JsonDict],
    gguf_path: Path,
    duration_s: float,
) -> JsonDict:
    artifact = _base_artifact(
        honest_verdict="running_uprm_replication_pregeneration_skeleton",
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
    )
    artifact["deliverable_stage"] = "pregeneration_skeleton"
    artifact["model_specs"] = {
        **artifact["model_specs"],
        "gguf_path": gguf_path.as_posix(),
        "cuda_gpu": 0,
    }
    return artifact


def build_complete_artifact(
    *,
    evaluation: JsonDict,
    preconditions_checked: Sequence[JsonDict],
    candidate_cache_path: Path,
    gguf_path: Path,
    duration_s: float,
) -> JsonDict:
    uprm_accuracy = float(evaluation["verifier"]["accuracy"])
    tuned_accuracy = float(evaluation["tuned_self_consistency"]["accuracy"])
    delta = float(evaluation["verifier_minus_tuned_sc_delta"])
    ci95 = [float(value) for value in evaluation["verifier_minus_tuned_sc_ci95"]]
    mcnemar_p = float(evaluation["mcnemar_p"])
    headroom_present = bool(evaluation["headroom_present"])
    corpus_slug = _slug_corpus(CORPUS)
    win = delta > 0.0 and ci95[0] > 0.0 and mcnemar_p < 0.05 and headroom_present
    verdict_delta = _format_delta(delta)
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
        duration_s=duration_s,
    )
    artifact.update(
        {
            "headroom_present": headroom_present,
            "uprm_selection_accuracy": round(uprm_accuracy, 6),
            "tuned_sc_accuracy": round(tuned_accuracy, 6),
            "delta_vs_tuned_sc": round(delta, 6),
            "paired_ci95": ci95,
            "mcnemar_p": mcnemar_p,
            "n_questions": int(evaluation["n_rows"]),
            "oracle_at_k": float(evaluation["oracle_at_k"]),
            "model_specs": {
                **artifact["model_specs"],
                "gguf_path": gguf_path.as_posix(),
                "cuda_gpu": 0,
                "score_formula": "uPRM Eq.6 first-error marker score",
                "candidate_aggregation": "mean no-error log-likelihood S(T+1)/T",
                "tuned_self_consistency_config": evaluation["tuned_self_consistency"]["config"],
            },
            "inference_substrate": "live_llm_inference",
            "reproducibility_checksum": reproducibility_checksum(
                {
                    "model": MODEL_HF_ID,
                    "gguf_path": gguf_path.as_posix(),
                    "candidate_cache_path": candidate_cache_path.as_posix(),
                    "evaluation": evaluation,
                    "seed": RANDOM_SEED,
                }
            ),
            "oracle_distinctness_enforced": True,
            "candidate_cache_path": candidate_cache_path.as_posix(),
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
    spec = importlib.util.spec_from_file_location("carnot_adversarial_verify_5004", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load scripts/adversarial_verify.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_artifact(path)


def run_summarize_artifact(path: Path) -> int:  # pragma: no cover - reviewer CLI glue
    script_path = REPO_ROOT / "scripts" / "summarize_artifact.py"
    spec = importlib.util.spec_from_file_location("carnot_summarize_artifact_5004", script_path)
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
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
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
    for field in ("uprm_selection_accuracy", "tuned_sc_accuracy", "oracle_at_k"):
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
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if not isinstance(artifact.get("uprm_score_methodology_note"), str) or not artifact.get(
        "uprm_score_methodology_note"
    ):
        errors.append("uprm_score_methodology_note")
    if not str(artifact.get("honest_verdict", "")).startswith(
        ("blocked_", "running_", "complete_", "success_")
    ):
        errors.append("honest_verdict")
    return sorted(set(errors))


def _resolve_gemma_gguf(
    *_args: Any, **_kwargs: Any
) -> str | None:  # pragma: no cover - host cache probe
    from carnot.inference.sota_models import resolve_cached_gguf

    return resolve_cached_gguf(MODEL_HF_ID, preferred_quant="Q4_K_M")


def default_corpus_loader(
    limit: int,
) -> list[JsonDict]:  # pragma: no cover - dataset cache boundary
    return harness.load_musr_murder_mysteries(limit=limit)


def default_server_probe(  # pragma: no cover - live HTTP boundary
    *,
    port: int = DEFAULT_SERVER_PORT,
    timeout_s: int = 30,
) -> PreconditionCheck:
    try:
        payload = llama_server_completion(
            "Respond with exactly one token: +",
            port=port,
            seed=RANDOM_SEED,
            max_tokens=1,
            temperature=0.0,
            logprobs=MARKER_LOGPROBS_TOP_K,
            timeout_s=timeout_s,
        )
        parsed = parse_llama_completion_payload(payload)
        ok = bool(parsed["token_logprobs"]) and bool(parsed["top_logprobs"])
    except Exception as exc:
        return PreconditionCheck("llama_server_logprobs", False, f"{type(exc).__name__}: {exc}")
    return PreconditionCheck(
        "llama_server_logprobs",
        ok,
        "server returns completion_probabilities with top_logprobs"
        if ok
        else "server response lacked token_logprobs/top_logprobs",
        f"http://127.0.0.1:{port}/completion",
    )


def llama_server_completion(  # pragma: no cover - live HTTP boundary
    prompt: str,
    *,
    port: int,
    seed: int,
    max_tokens: int,
    temperature: float,
    logprobs: int,
    timeout_s: int,
    stop: Sequence[str] | None = None,
) -> JsonDict:
    import urllib.request

    payload: JsonDict = {
        "prompt": prompt,
        "n_predict": int(max_tokens),
        "temperature": float(temperature),
        "cache_prompt": True,
        "seed": int(seed),
        "logprobs": int(logprobs),
    }
    if stop:
        payload["stop"] = list(stop)
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        loaded = json.load(response)
    if not isinstance(loaded, dict):
        raise RuntimeError("llama-server returned non-object payload")
    return loaded


def _match_choice(text: str, choices: Sequence[str]) -> str | None:
    return harness._match_choice(text, choices)  # noqa: SLF001


def split_reasoning_steps(text: str, *, max_steps: int = 8) -> list[str]:
    if not text.strip():
        return []
    steps = [line.strip() for line in text.splitlines() if line.strip()]
    if len(steps) <= 1:
        steps = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
    return steps[:max_steps]


def _marker_prompt(row: JsonMap, candidate: JsonMap, step: str) -> str:
    context = str(row.get("context") or "")[:3000]
    question = str(row.get("question") or "")
    prior = "\n".join(str(item) for item in candidate.get("steps", [])[:8])
    return (
        "You are a strict reasoning judge. Reply with exactly '+' if the current "
        "step is logically correct so far, or '-' if it is incorrect.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\n"
        f"CANDIDATE TRACE SO FAR:\n{prior}\n\nCURRENT STEP:\n{step}\n\nMARKER:"
    )


def _annotate_candidate_markers(  # pragma: no cover - live HTTP boundary
    row: JsonMap,
    candidate: JsonDict,
    *,
    port: int,
    seed: int,
) -> JsonDict:
    steps = split_reasoning_steps(str(candidate.get("reasoning") or ""))
    if not steps:
        steps = [str(candidate.get("reasoning") or candidate.get("answer") or "")]
    annotated = dict(candidate)
    annotated["steps"] = steps
    marker_rows: list[dict[str, float]] = []
    for index, step in enumerate(steps):
        payload = llama_server_completion(
            _marker_prompt(row, {**annotated, "steps": steps[: index + 1]}, step),
            port=port,
            seed=seed + index,
            max_tokens=1,
            temperature=0.0,
            logprobs=MARKER_LOGPROBS_TOP_K,
            timeout_s=120,
            stop=["\n"],
        )
        parsed = parse_llama_completion_payload(payload)
        if not parsed["top_logprobs"]:
            raise UprmScoringError("marker completion lacked top_logprobs")
        marker_rows.append(parsed["top_logprobs"][0])
    annotated["uprm_marker_logprobs"] = marker_rows
    return annotated


def default_candidate_rows_builder(  # pragma: no cover - live generation boundary
    *,
    corpus_rows: Sequence[JsonMap],
    candidate_cache_path: Path,
    k_candidates: int,
    random_seed: int,
    server_port: int,
) -> list[JsonDict]:
    cached_rows = _read_jsonl(candidate_cache_path)
    if len(cached_rows) >= len(corpus_rows):
        return cached_rows[: len(corpus_rows)]

    rows: list[JsonDict] = []

    def generator(prompt: str, *, seed: int, config: GenerationConfig) -> JsonDict:
        payload = llama_server_completion(
            prompt,
            port=server_port,
            seed=seed,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            logprobs=MARKER_LOGPROBS_TOP_K,
            timeout_s=300,
            stop=["<|im_end|>", "<end_of_turn>", "<|endoftext|>"],
        )
        parsed = parse_llama_completion_payload(payload)
        return {
            "text": parsed["text"],
            "token_logprobs": parsed["token_logprobs"],
            "top_logprobs": parsed["top_logprobs"],
        }

    config = GenerationConfig(k=k_candidates, model=MODEL_NAME, gpu=0, max_tokens=512)
    for row_index, row in enumerate(corpus_rows):
        candidates = harness.generate_candidates_with_logprobs(
            row,
            generator=generator,
            config=config,
            seed=random_seed + row_index * 1000,
        )
        annotated = [
            _annotate_candidate_markers(
                row,
                {
                    **candidate,
                    "top_logprobs": candidate.get("top_logprobs", []),
                    "answer": candidate.get("answer")
                    or _match_choice(
                        str(candidate.get("reasoning") or ""), list(row.get("choices") or [])
                    ),
                },
                port=server_port,
                seed=random_seed + row_index * 1000 + index * 100,
            )
            for index, candidate in enumerate(candidates)
        ]
        merged = dict(row)
        merged["candidates"] = annotated
        rows.append(merged)
        _write_jsonl(candidate_cache_path, rows)
    return rows


def check_preconditions(
    *,
    root: Path,
    gguf_resolver: Callable[..., str | None],
    server_probe: Callable[..., PreconditionCheck],
    corpus_loader: Callable[[int], list[JsonDict]],
    candidate_cache_path: Path,
    require_candidate_cache_or_fresh_generation: bool,
    min_questions: int,
    server_port: int,
) -> tuple[list[PreconditionCheck], Path | None, list[JsonDict]]:
    gguf_raw = gguf_resolver(MODEL_HF_ID)
    gguf_path = Path(gguf_raw) if gguf_raw else None
    gguf_ok = bool(gguf_path and gguf_path.exists() and gguf_path.is_file())
    checks = [
        PreconditionCheck(
            "gemma_gguf_cache",
            gguf_ok,
            f"{MODEL_HF_ID} resolved" if gguf_ok else f"{MODEL_HF_ID} not resolved as a GGUF",
            gguf_path.as_posix() if gguf_path else None,
        )
    ]
    checks.append(server_probe(port=server_port))
    try:
        rows = corpus_loader(min_questions)
    except Exception as exc:
        rows = []
        detail = f"{type(exc).__name__}: {exc}"
    else:
        detail = f"{len(rows)} cached target-corpus row(s), required >= {min_questions}"
    checks.append(
        PreconditionCheck(
            "target_corpus",
            len(rows) >= min_questions,
            detail,
            root.as_posix(),
        )
    )
    if require_candidate_cache_or_fresh_generation:
        cached_rows = _read_jsonl(candidate_cache_path)
        fresh_enabled = os.environ.get(FRESH_GENERATION_OPT_IN_ENV) == "1"
        checks.append(
            PreconditionCheck(
                "uprm_logprob_candidate_cache",
                len(cached_rows) >= min_questions or fresh_enabled,
                (
                    f"{len(cached_rows)} cached uPRM logprob row(s), required >= {min_questions}; "
                    f"{FRESH_GENERATION_OPT_IN_ENV}={os.environ.get(FRESH_GENERATION_OPT_IN_ENV, '')!r}"
                ),
                candidate_cache_path.as_posix(),
            )
        )
    return checks, gguf_path if gguf_ok else None, rows


def first_missing_resource(checks: Sequence[PreconditionCheck]) -> str | None:
    for check in checks:
        if not check.available:
            return check.resource
    return None


def _precondition_dicts(checks: Sequence[PreconditionCheck]) -> list[JsonDict]:
    return [check.as_dict() for check in checks]


def _oracle_distinctness_enforced(rows: Sequence[JsonMap]) -> bool:
    try:
        evaluate_verifier(rows, scorer=lambda candidate: candidate["gold"], bootstrap_samples=8)
    except OracleDistinctnessError:
        return True
    return False  # pragma: no cover - indicates the shared harness regressed


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    gguf_resolver: Callable[..., str | None] = _resolve_gemma_gguf,
    server_probe: Callable[..., PreconditionCheck] = default_server_probe,
    corpus_loader: Callable[[int], list[JsonDict]] = default_corpus_loader,
    candidate_rows_builder: Callable[..., list[JsonDict]] = default_candidate_rows_builder,
    audit_runner: AuditRunner = run_adversarial_verify,
    summary_runner: SummaryRunner = run_summarize_artifact,
    min_questions: int = DEFAULT_LIMIT,
    limit: int = DEFAULT_LIMIT,
    k_candidates: int = DEFAULT_K,
    bootstrap_samples: int = 2000,
    random_seed: int = RANDOM_SEED,
    server_port: int = DEFAULT_SERVER_PORT,
    now: Clock = time.time,
    write: bool = True,
) -> JsonDict:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    root = Path(root)
    artifact_path = Path(artifact_path) if artifact_path else root / RESULT_RELATIVE_PATH
    candidate_cache_path = root / CANDIDATE_CACHE_RELATIVE_PATH
    start = float(now())

    checks, gguf_path, corpus_rows = check_preconditions(
        root=root,
        gguf_resolver=gguf_resolver,
        server_probe=server_probe,
        corpus_loader=corpus_loader,
        candidate_cache_path=candidate_cache_path,
        require_candidate_cache_or_fresh_generation=candidate_rows_builder
        is default_candidate_rows_builder,
        min_questions=min_questions,
        server_port=server_port,
    )
    preconditions = _precondition_dicts(checks)
    missing = first_missing_resource(checks)
    if missing is not None:
        artifact = build_blocked_artifact(
            missing_resource=missing,
            preconditions_checked=preconditions,
            duration_s=float(now()) - start,
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact

    assert gguf_path is not None
    skeleton = build_skeleton_artifact(
        preconditions_checked=preconditions,
        gguf_path=gguf_path,
        duration_s=float(now()) - start,
    )
    if write:
        write_json(artifact_path, skeleton)

    try:
        selected_rows = list(corpus_rows)[:limit]
        candidate_rows = candidate_rows_builder(
            corpus_rows=selected_rows,
            candidate_cache_path=candidate_cache_path,
            k_candidates=k_candidates,
            random_seed=random_seed,
            server_port=server_port,
        )
        prepared_rows = prepare_rows_with_uprm_scores(candidate_rows)
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
            duration_s=float(now()) - start,
            error=str(exc),
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact
    except Exception as exc:
        artifact = build_blocked_artifact(
            missing_resource="generation_or_scoring_error",
            preconditions_checked=preconditions,
            duration_s=float(now()) - start,
            error=f"{type(exc).__name__}: {exc}",
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact

    artifact = build_complete_artifact(
        evaluation=evaluation,
        preconditions_checked=preconditions,
        candidate_cache_path=candidate_cache_path,
        gguf_path=gguf_path,
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
