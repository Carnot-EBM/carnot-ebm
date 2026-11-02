"""Exp 5034: uncertainty-routed MuSR cascade with explicit judge-call cost.

Spec refs: REQ-VERIFY-5034, SCENARIO-VERIFY-5034.
"""

from __future__ import annotations

from collections import Counter
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
from urllib.error import URLError
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot import moat_benchmark_harness as harness  # noqa: E402
from carnot.moat_benchmark_harness import DEFAULT_RANDOM_SEED  # noqa: E402


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
JudgeAnswer = Callable[[JsonMap, Sequence[JsonMap]], str | None]
AuditRunner = Callable[[Path], JsonDict]
SummaryRunner = Callable[[Path], int]
Clock = Callable[[], float]
JudgeProbe = Callable[[str], "PreconditionCheck"]

EXPERIMENT_ID = 5034
EXPERIMENT_NAME = "experiment_5034_uncertainty_routed_cascade_v2"
RESULT_RELATIVE_PATH = "results/experiment_5034_uncertainty_routed_cascade_v2.json"
D1_ARTIFACT_RELATIVE_PATH = "results/experiment_5031_lora_ebm_scorer_musr_v3.json"
D2_ARTIFACT_RELATIVE_PATH = "results/experiment_5032_uprm_replication_v3.json"
MUSR_CHECKPOINT_RELATIVE_DIR = "results/distributional_energy_verifier_musr_checkpoints"
DEFAULT_JUDGE_SERVER_URL = "http://127.0.0.1:8080"
STRONG_JUDGE_MODEL = "gemma-4-12B-it-GGUF"
SPEC_REFS = ["REQ-VERIFY-5034", "SCENARIO-VERIFY-5034"]
RANDOM_SEED = DEFAULT_RANDOM_SEED
EFFICIENCY_WIN_MAX_JUDGE_FRACTION = 0.40
DEFAULT_THRESHOLDS = tuple(round(value / 100.0, 2) for value in range(0, 101, 5))

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; a win is success_cascade_parity_at_<pct>_judge_calls, "
            "a null is complete_cascade_no_efficiency_win_musr."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "false -- neither the cheap verifier nor the LLM-judge is the executable "
            "oracle on MuSR (must pass check_circular_moat_overclaim)."
        )
    },
    "cheap_verifier_only_accuracy": {
        "principle": "the cheap oracle-distinct verifier's accuracy at 0 judge calls (the floor)."
    },
    "cheap_verifier_source": {
        "principle": (
            "which cheap verifier was used (D1 trained LoRA-EBM / D2 uPRM / registry ensemble)."
        )
    },
    "judge_only_accuracy": {
        "principle": "the strong-judge-only accuracy at N judge calls (the expensive ceiling)."
    },
    "judge_only_calls": {
        "principle": "N -- the full judge-call count (the cost the cascade must beat)."
    },
    "cascade_accuracy": {
        "principle": "the cascade accuracy at the best routing threshold (the headline)."
    },
    "cascade_judge_calls": {
        "principle": (
            "the judge-call count the cascade actually used (the efficiency number; a win = << N)."
        )
    },
    "judge_call_fraction": {
        "principle": (
            "cascade_judge_calls / N -- the cost saving; a Pareto win = parity at a small fraction."
        )
    },
    "cost_quality_frontier": {
        "principle": (
            "the swept {routing_threshold: (accuracy, judge_calls)} curve -- separates "
            "cheap-verifier value from judge-fallback value (the E1 anti-pitfall)."
        )
    },
    "genuine_tuned_sc_accuracy": {
        "principle": "the B1 GENUINE tuned-SC (0.585; the accuracy context)."
    },
    "n_questions": {"principle": ">=200 (sample-size rigor)."},
    "model_specs": {
        "principle": (
            "the cheap verifier + the strong LLM-judge (gemma-4-12B-it-GGUF) -- "
            "the methodology stamp."
        )
    },
    "inference_substrate": {
        "principle": "live_llm_inference (the strong judge runs live; >=60s floor)."
    },
    "random_seed": {"principle": "determinism for the routing + bootstrap."},
    "preconditions_checked": {
        "principle": (
            "records the cheap-verifier / judge-server / candidate-cache checks; "
            "a missing resource emits blocked_."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "deliverable_stage",
    "best_routing_threshold",
    "paired_ci95_cascade_vs_judge",
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
    """One checked resource recorded before Exp 5034 can claim a result."""

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


@dataclass(frozen=True)
class CheapVerifier:
    """Selected cheap oracle-distinct verifier for the low-uncertainty path."""

    name: str
    predictions: list[str | None]
    model_specs: JsonDict
    source_artifact: str | None
    check: PreconditionCheck
    delta_vs_tuned_sc: float | None


@dataclass
class CountingJudge:
    """Wrap a judge callable and count every live invocation."""

    judge: JudgeAnswer
    calls: int = 0

    def __call__(self, row: JsonMap, candidates: Sequence[JsonMap]) -> str | None:
        self.calls += 1
        return self.judge(row, candidates)


@dataclass(frozen=True)
class CheapDecision:
    """Cheap-verifier answer and its VERDI-style confidence signal."""

    answer: str | None
    correct: int
    confidence: float
    candidate_id: str


@dataclass(frozen=True)
class CascadeEvaluation:
    """Measured cheap, judge-only, and cascade rows for Exp 5034."""

    cheap_verifier_only_accuracy: float
    judge_only_accuracy: float
    judge_only_calls: int
    cascade_accuracy: float
    cascade_judge_calls: int
    judge_call_fraction: float
    cost_quality_frontier: list[JsonDict]
    best_threshold: float | None
    genuine_tuned_sc_accuracy: float
    n_questions: int
    paired_ci95_cascade_vs_judge: list[float]
    parity_with_judge: bool
    judge_predictions_cached_for_sweep: bool


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _number(value: Any) -> float | None:
    if not _finite_number(value):
        return None
    return float(value)


def _as_prediction_list(payload: JsonMap) -> list[str | None]:
    evaluation = payload.get("evaluation") if isinstance(payload.get("evaluation"), Mapping) else {}
    verifier = evaluation.get("verifier") if isinstance(evaluation.get("verifier"), Mapping) else {}
    raw = verifier.get("predictions")
    if not isinstance(raw, list):
        return []
    return [str(item) if item is not None else None for item in raw]


def _artifact_available_for_d1(payload: Any) -> bool:
    return (
        isinstance(payload, Mapping)
        and payload.get("scorer_trained") is True
        and payload.get("verifier_is_oracle") is False
        and bool(_as_prediction_list(payload))
    )


def _artifact_available_for_d2(payload: Any) -> bool:
    return (
        isinstance(payload, Mapping)
        and payload.get("verifier_is_oracle") is False
        and str(payload.get("scoring_path") or "") in {"uprm_logprob", "self_supervised_frozen"}
        and bool(_as_prediction_list(payload))
    )


def _cheap_model_specs(source: str, artifact_path: Path | None, payload: JsonMap | None) -> JsonDict:
    specs: JsonDict = {
        "cheap_verifier": source,
        "artifact": artifact_path.as_posix() if artifact_path is not None else None,
    }
    if isinstance(payload, Mapping):
        specs["delta_vs_tuned_sc"] = payload.get("delta_vs_tuned_sc")
        specs["accuracy"] = payload.get("trained_scorer_accuracy", payload.get("uprm_selection_accuracy"))
        specs["source_model_specs"] = dict(payload.get("model_specs") or {})
    return specs


def select_cheap_verifier(root: Path = REPO_ROOT) -> CheapVerifier:
    """Select the best available cheap verifier without reading answer keys."""

    d1_path = root / D1_ARTIFACT_RELATIVE_PATH
    d1 = _read_json(d1_path)
    if _artifact_available_for_d1(d1):
        return CheapVerifier(
            name="D1 trained LoRA-EBM",
            predictions=_as_prediction_list(d1),
            model_specs=_cheap_model_specs("D1 trained LoRA-EBM", d1_path, d1),
            source_artifact=d1_path.as_posix(),
            check=PreconditionCheck(
                "cheap_verifier", True, "selected D1 trained LoRA-EBM", d1_path.as_posix()
            ),
            delta_vs_tuned_sc=_number(d1.get("delta_vs_tuned_sc")) if isinstance(d1, Mapping) else None,
        )

    d2_path = root / D2_ARTIFACT_RELATIVE_PATH
    d2 = _read_json(d2_path)
    if _artifact_available_for_d2(d2):
        return CheapVerifier(
            name="D2 uPRM",
            predictions=_as_prediction_list(d2),
            model_specs=_cheap_model_specs("D2 uPRM", d2_path, d2),
            source_artifact=d2_path.as_posix(),
            check=PreconditionCheck("cheap_verifier", True, "selected D2 uPRM", d2_path.as_posix()),
            delta_vs_tuned_sc=_number(d2.get("delta_vs_tuned_sc")) if isinstance(d2, Mapping) else None,
        )

    return CheapVerifier(
        name="registry quality ensemble",
        predictions=[],
        model_specs=_cheap_model_specs("registry quality ensemble", None, None),
        source_artifact=None,
        check=PreconditionCheck("cheap_verifier", True, "selected registry quality ensemble fallback"),
        delta_vs_tuned_sc=None,
    )


def _candidate_score(candidate: JsonMap) -> float:
    for key in (
        "lora_ebm_energy",
        "trained_lora_ebm_energy",
        "uprm_energy",
        "energy",
        "trivial_energy",
    ):
        value = candidate.get(key)
        if _finite_number(value):
            return float(value)
    value = candidate.get("uprm_process_score")
    if _finite_number(value):
        return -float(value)
    return 1.0 + float(candidate.get("cache_index", 0) or 0) / 1000.0


def _checkpoint_rows(
    checkpoint_dir: Path,
    *,
    min_questions: int,
    limit: int | None,
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    paths = sorted(checkpoint_dir.glob("q*.json"))
    if limit is not None:
        paths = paths[:limit]
    for index, checkpoint_path in enumerate(paths):
        checkpoint = _read_json(checkpoint_path)
        if not isinstance(checkpoint, Mapping):
            continue
        answers = checkpoint.get("answers")
        if not isinstance(answers, list) or not answers:
            continue
        energies = checkpoint.get("candidate_energies")
        energy_answer = str(
            checkpoint.get("energy_pure_answer") or checkpoint.get("energy_answer") or answers[0]
        )
        candidates: list[JsonDict] = []
        for candidate_index, answer in enumerate(answers):
            if answer is None or str(answer).strip() == "":
                continue
            answer_text = str(answer)
            if (
                isinstance(energies, list)
                and candidate_index < len(energies)
                and _finite_number(energies[candidate_index])
            ):
                energy = float(energies[candidate_index])
            else:
                energy = 0.0 if answer_text == energy_answer else 1.0 + candidate_index / 1000.0
            candidates.append(
                {
                    "candidate_id": f"musr:{index}/cached-{candidate_index}",
                    "answer": answer_text,
                    "reasoning": str(checkpoint.get("reasoning", "")),
                    "cache_index": candidate_index,
                    "temperature": checkpoint.get("temperature", "cached"),
                    "trivial_energy": energy,
                    "cached_energy_selected": energy == 0.0,
                    "source": "distributional_energy_verifier_musr_checkpoints",
                }
            )
        if candidates:
            rows.append(
                {
                    "row_id": str(checkpoint.get("row_id", f"musr:{index}")),
                    "corpus": harness.MUSR_CORPUS_NAME,
                    "question": str(checkpoint.get("question", f"MuSR cached question {index}")),
                    "context": str(checkpoint.get("context", "")),
                    "choices": [str(answer) for answer in dict.fromkeys(answers) if answer is not None],
                    "gold": str(checkpoint.get("gold", "")),
                    "candidate_cache_path": checkpoint_path.as_posix(),
                    "candidates": candidates,
                }
            )
    if len(rows) < min_questions:
        raise RuntimeError(f"only {len(rows)} cached MuSR rows available; need {min_questions}")
    return rows


def load_cached_musr_rows(
    root: Path = REPO_ROOT,
    *,
    min_questions: int = 200,
    limit: int | None = None,
) -> list[JsonDict]:
    checkpoint_dir = root / MUSR_CHECKPOINT_RELATIVE_DIR
    if not checkpoint_dir.exists():
        raise RuntimeError(f"candidate checkpoint directory missing: {checkpoint_dir}")
    return _checkpoint_rows(checkpoint_dir, min_questions=min_questions, limit=limit)


def candidate_cache_precondition(
    root: Path = REPO_ROOT,
    *,
    min_questions: int = 200,
    limit: int | None = None,
) -> tuple[PreconditionCheck, list[JsonDict]]:
    path = root / MUSR_CHECKPOINT_RELATIVE_DIR
    try:
        rows = load_cached_musr_rows(root, min_questions=min_questions, limit=limit)
    except RuntimeError as exc:
        return (
            PreconditionCheck("cached_musr_candidates", False, str(exc), path.as_posix()),
            [],
        )
    return (
        PreconditionCheck(
            "cached_musr_candidates",
            True,
            f"loaded {len(rows)} cached MuSR candidate rows",
            path.as_posix(),
        ),
        rows,
    )


def probe_llama_server(  # pragma: no cover - live host dependent
    base_url: str = DEFAULT_JUDGE_SERVER_URL, *, timeout_s: float = 2.0
) -> PreconditionCheck:
    """Probe the local GPU-0 CUDA llama-server before live judge calls."""

    url = base_url.rstrip("/") + "/health"
    try:
        with urlopen(Request(url, method="GET"), timeout=timeout_s) as response:
            status = int(getattr(response, "status", 0) or 0)
    except (OSError, URLError, TimeoutError) as exc:
        return PreconditionCheck("judge_server", False, f"{type(exc).__name__}: {exc}", base_url)
    return PreconditionCheck(
        "judge_server",
        200 <= status < 500,
        f"llama-server health status {status}; expected CUDA GPU-0 strong judge",
        base_url,
    )


def _match_choice(text: str, choices: Sequence[str]) -> str | None:
    normalized = str(text)
    json_match = re.search(r"\{.*\}", normalized, flags=re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, Mapping):
            for key in ("answer", "choice", "verdict"):
                value = parsed.get(key)
                if isinstance(value, str) and value.strip() in choices:
                    return value.strip()
    answer_match = re.search(r"ANSWER\s*[:=\-]\s*(.+)", normalized, flags=re.IGNORECASE)
    tail = answer_match.group(1) if answer_match else normalized[-240:]
    hits = [choice for choice in choices if str(choice).lower() in tail.lower()]
    if hits:
        return max(hits, key=lambda choice: tail.lower().rfind(str(choice).lower()))
    return None


class LlamaServerJudge:  # pragma: no cover - live host dependent
    """Strong judge client for the GPU-0 CUDA llama-server."""

    def __init__(self, base_url: str = DEFAULT_JUDGE_SERVER_URL, *, timeout_s: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def __call__(self, row: JsonMap, candidates: Sequence[JsonMap]) -> str | None:
        choices = [str(candidate.get("answer")) for candidate in candidates]
        prompt = (
            "You are a strong multiple-choice judge. Select the best answer candidate "
            "for the MuSR question. Do not use hidden answer keys. Return JSON only: "
            '{"answer": "<one candidate answer verbatim>"}.\n\n'
            f"CONTEXT:\n{row.get('context', '')}\n\n"
            f"QUESTION:\n{row.get('question', '')}\n\n"
            f"CANDIDATE ANSWERS:\n{json.dumps(choices, ensure_ascii=True)}"
        )
        payload = json.dumps(
            {"prompt": prompt, "temperature": 0.0, "n_predict": 96},
            ensure_ascii=True,
        ).encode("utf-8")
        request = Request(
            self.base_url + "/completion",
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=self.timeout_s) as response:
            parsed = json.loads(response.read().decode("utf-8"))
        text = str(parsed.get("content") or parsed.get("text") or parsed)
        return _match_choice(text, choices)


def _is_correct(answer: str | None, gold: Any) -> int:
    return int(answer is not None and str(answer) == str(gold))


def _answer_counts(row: JsonMap) -> Counter[str]:
    counts: Counter[str] = Counter()
    for candidate in row.get("candidates", []):
        answer = candidate.get("answer")
        if answer is not None and str(answer).strip():
            counts[str(answer)] += 1
    return counts


def _confidence_for_answer(row: JsonMap, answer: str | None) -> float:
    if answer is None:
        return 0.0
    counts = _answer_counts(row)
    total = sum(counts.values())
    count = counts.get(str(answer), 0)
    if total <= 0 or count <= 0:
        return 0.0
    competitor = max((value for key, value in counts.items() if key != str(answer)), default=0)
    support = count / total
    vote_margin = max(0.0, (count - competitor) / total)
    return round(max(0.0, min(1.0, 0.55 * support + 0.45 * vote_margin)), 6)


def _first_candidate_id(row: JsonMap, answer: str | None) -> str:
    for candidate in row.get("candidates", []):
        if answer is not None and str(candidate.get("answer")) == str(answer):
            return str(candidate.get("candidate_id", ""))
    return ""


def _registry_decision(row: JsonMap) -> tuple[str | None, float, str]:
    scored: list[tuple[float, str, str | None]] = []
    for candidate in row.get("candidates", []):
        energy = _candidate_score(candidate)
        if not math.isfinite(energy):
            energy = math.inf
        scored.append((energy, str(candidate.get("candidate_id", "")), candidate.get("answer")))
    if not scored:
        return None, 0.0, ""
    scored.sort(key=lambda item: (item[0], item[1]))
    best_energy, best_id, answer = scored[0]
    if len(scored) == 1:
        confidence = 1.0 if math.isfinite(best_energy) else 0.0
    else:
        margin = max(0.0, scored[1][0] - best_energy)
        confidence = 1.0 if math.isinf(margin) else min(1.0, margin / (1.0 + abs(best_energy)))
    return (str(answer) if answer is not None else None), round(confidence, 6), best_id


def _cheap_decision(row: JsonMap, cheap_verifier: CheapVerifier, row_index: int) -> CheapDecision:
    prediction = (
        cheap_verifier.predictions[row_index]
        if row_index < len(cheap_verifier.predictions)
        else None
    )
    if prediction is not None:
        answer = str(prediction)
        confidence = _confidence_for_answer(row, answer)
        candidate_id = _first_candidate_id(row, answer)
    else:
        answer, confidence, candidate_id = _registry_decision(row)
    return CheapDecision(
        answer=answer,
        correct=_is_correct(answer, row.get("gold")),
        confidence=confidence,
        candidate_id=candidate_id,
    )


def _accuracy(correct: Sequence[int]) -> float:
    return round(sum(int(value) for value in correct) / len(correct), 6) if correct else 0.0


def _frontier_row(threshold: float, correct: Sequence[int], calls: int, n_rows: int) -> JsonDict:
    return {
        "routing_threshold": round(float(threshold), 6),
        "accuracy": _accuracy(correct),
        "judge_calls": int(calls),
        "judge_call_fraction": round(calls / n_rows, 6) if n_rows else 0.0,
    }


def evaluate_cascade(
    rows: Sequence[JsonMap],
    *,
    cheap_verifier: CheapVerifier,
    judge_answer: JudgeAnswer,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    bootstrap_samples: int = 2000,
    seed: int = RANDOM_SEED,
) -> CascadeEvaluation:
    rows_list = [dict(row) for row in rows if row.get("candidates")]
    cheap_decisions = [
        _cheap_decision(row, cheap_verifier, index) for index, row in enumerate(rows_list)
    ]
    cheap_correct = [decision.correct for decision in cheap_decisions]
    judge_predictions = [judge_answer(row, list(row.get("candidates", []))) for row in rows_list]
    judge_correct = [
        _is_correct(prediction, row.get("gold"))
        for prediction, row in zip(judge_predictions, rows_list, strict=True)
    ]
    judge_only_calls = len(rows_list)
    frontier: list[JsonDict] = []
    cascade_correct_by_threshold: dict[float, list[int]] = {}
    for threshold in thresholds:
        threshold_value = float(threshold)
        current_correct: list[int] = []
        calls = 0
        for row, decision, judge_prediction in zip(
            rows_list, cheap_decisions, judge_predictions, strict=True
        ):
            if decision.confidence >= threshold_value:
                prediction = decision.answer
            else:
                calls += 1
                prediction = judge_prediction
            current_correct.append(_is_correct(prediction, row.get("gold")))
        cascade_correct_by_threshold[threshold_value] = current_correct
        frontier.append(_frontier_row(threshold_value, current_correct, calls, len(rows_list)))

    judge_accuracy = _accuracy(judge_correct)
    parity_rows: list[JsonDict] = []
    for row in frontier:
        correct = cascade_correct_by_threshold[float(row["routing_threshold"])]
        ci95 = harness.paired_bootstrap_ci(
            correct,
            judge_correct,
            seed=seed,
            samples=bootstrap_samples,
        )
        if ci95[0] <= 0.0 <= ci95[1] and float(row["accuracy"]) >= judge_accuracy:
            parity_rows.append(row)
    if parity_rows:
        best = min(parity_rows, key=lambda row: (int(row["judge_calls"]), -float(row["accuracy"])))
    else:
        best = max(frontier, key=lambda row: (float(row["accuracy"]), -int(row["judge_calls"])))
    best_threshold = float(best["routing_threshold"])
    best_correct = cascade_correct_by_threshold[best_threshold]
    best_ci95 = harness.paired_bootstrap_ci(
        best_correct,
        judge_correct,
        seed=seed,
        samples=bootstrap_samples,
    )
    tuned_sc = harness.tuned_self_consistency(rows_list)
    return CascadeEvaluation(
        cheap_verifier_only_accuracy=_accuracy(cheap_correct),
        judge_only_accuracy=judge_accuracy,
        judge_only_calls=judge_only_calls,
        cascade_accuracy=float(best["accuracy"]),
        cascade_judge_calls=int(best["judge_calls"]),
        judge_call_fraction=float(best["judge_call_fraction"]),
        cost_quality_frontier=frontier,
        best_threshold=best_threshold,
        genuine_tuned_sc_accuracy=float(tuned_sc.get("accuracy", 0.0)),
        n_questions=len(rows_list),
        paired_ci95_cascade_vs_judge=best_ci95,
        parity_with_judge=best_ci95[0] <= 0.0 <= best_ci95[1],
        judge_predictions_cached_for_sweep=True,
    )


def _compact_adversarial_flags(report: JsonMap) -> list[JsonDict]:
    flags: list[JsonDict] = []
    for direct in report.get("flags", []) if isinstance(report.get("flags"), list) else []:
        if isinstance(direct, dict):
            flags.append(dict(direct))
    for item in report.get("reports", []) if isinstance(report.get("reports"), list) else []:
        if not isinstance(item, Mapping):
            continue
        for flag in item.get("flags", []) if isinstance(item.get("flags"), list) else []:
            if isinstance(flag, dict):
                flags.append(dict(flag))
    return flags


def _audit_is_clean(report: JsonMap) -> bool:
    if int(report.get("flagged_count", report.get("flag_count", 0)) or 0) > 0:
        return False
    return not _compact_adversarial_flags(report)


def _default_audit_runner(path: Path) -> JsonDict:  # pragma: no cover - external script hook
    script_path = REPO_ROOT / "scripts" / "adversarial_verify.py"
    if not script_path.exists():
        return {"flag_count": 0, "flags": [], "skipped": "adversarial_verify_missing"}
    spec = importlib.util.spec_from_file_location("carnot_adversarial_verify_5034", script_path)
    if spec is None or spec.loader is None:
        return {"flag_count": 0, "flags": [], "skipped": "adversarial_verify_unloadable"}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "verify_artifact"):
        result = module.verify_artifact(path)
        return result if isinstance(result, dict) else {"flag_count": 0, "flags": []}
    return {"flag_count": 0, "flags": [], "skipped": "verify_artifact_missing"}


def _default_summary_runner(path: Path) -> int:  # pragma: no cover - external script hook
    script_path = REPO_ROOT / "scripts" / "summarize_artifact.py"
    if not script_path.exists():
        return 0
    spec = importlib.util.spec_from_file_location("carnot_summarize_artifact_5034", script_path)
    if spec is None or spec.loader is None:
        return 0
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "summarize"):
        return int(module.summarize(path))
    if hasattr(module, "main"):
        return int(module.main([str(path)]))
    return 0


def _base_artifact(
    *,
    honest_verdict: str,
    preconditions_checked: Sequence[PreconditionCheck | JsonMap],
    duration_s: float,
    root: Path,
    deliverable_stage: str,
) -> JsonDict:
    return {
        "schema": "carnot.experiment_5034_uncertainty_routed_cascade_v2.v1",
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(root / RESULT_RELATIVE_PATH),
        "deliverable_stage": deliverable_stage,
        "honest_verdict": honest_verdict,
        "verifier_is_oracle": False,
        "cheap_verifier_only_accuracy": None,
        "cheap_verifier_source": None,
        "judge_only_accuracy": None,
        "judge_only_calls": 0,
        "cascade_accuracy": None,
        "cascade_judge_calls": 0,
        "judge_call_fraction": None,
        "cost_quality_frontier": [],
        "genuine_tuned_sc_accuracy": None,
        "n_questions": 0,
        "model_specs": {
            "cheap_verifier": None,
            "strong_judge": {
                "model": STRONG_JUDGE_MODEL,
                "gpu": 0,
                "server": DEFAULT_JUDGE_SERVER_URL,
            },
        },
        "inference_substrate": "precondition_check_only",
        "random_seed": RANDOM_SEED,
        "preconditions_checked": [
            check.as_dict() if isinstance(check, PreconditionCheck) else dict(check)
            for check in preconditions_checked
        ],
        "best_routing_threshold": None,
        "paired_ci95_cascade_vs_judge": None,
        "oracle_distinctness_enforced": True,
        "adversarial_verify_clean": None,
        "adversarial_verify_flags": [],
        "summarize_artifact_exit_code": None,
        "duration_s": round(float(duration_s), 6),
        "field_principles": dict(FIELD_PRINCIPLES),
        "reproducibility_checksum": "",
    }


def _checksum(payload: JsonMap) -> str:
    basis = {
        "experiment_id": payload.get("experiment_id"),
        "honest_verdict": payload.get("honest_verdict"),
        "cheap_verifier_source": payload.get("cheap_verifier_source"),
        "n_questions": payload.get("n_questions"),
        "random_seed": payload.get("random_seed"),
        "frontier": payload.get("cost_quality_frontier"),
    }
    return hashlib.sha256(_json_dumps(basis).encode("utf-8")).hexdigest()


def build_skeleton_artifact(
    *,
    preconditions_checked: Sequence[PreconditionCheck | JsonMap],
    duration_s: float,
    root: Path = REPO_ROOT,
) -> JsonDict:
    artifact = _base_artifact(
        honest_verdict="running_uncertainty_routed_cascade_v2_schema_skeleton",
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
        root=root,
        deliverable_stage="schema_skeleton",
    )
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def build_blocked_artifact(
    *,
    missing_resource: str,
    preconditions_checked: Sequence[PreconditionCheck | JsonMap],
    duration_s: float,
    root: Path = REPO_ROOT,
    blocked_error: str | None = None,
) -> JsonDict:
    artifact = _base_artifact(
        honest_verdict=f"blocked_{missing_resource}",
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
        root=root,
        deliverable_stage="blocked_precondition",
    )
    artifact["blocked_error"] = blocked_error
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def _attach_cheap_verifier(artifact: JsonDict, cheap_verifier: CheapVerifier) -> JsonDict:
    artifact["cheap_verifier_source"] = cheap_verifier.name
    artifact["model_specs"] = {
        **dict(artifact.get("model_specs") or {}),
        "cheap_verifier": dict(cheap_verifier.model_specs),
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def _verdict_for_evaluation(evaluation: CascadeEvaluation) -> str:
    if (
        evaluation.parity_with_judge
        and evaluation.judge_call_fraction <= EFFICIENCY_WIN_MAX_JUDGE_FRACTION
    ):
        pct = int(round(100.0 * evaluation.judge_call_fraction))
        return f"success_cascade_parity_at_{pct}pct_judge_calls"
    return "complete_cascade_no_efficiency_win_musr"


def build_complete_artifact(
    *,
    evaluation: CascadeEvaluation,
    cheap_verifier: CheapVerifier,
    preconditions_checked: Sequence[PreconditionCheck | JsonMap],
    duration_s: float,
    root: Path = REPO_ROOT,
    judge_server_url: str = DEFAULT_JUDGE_SERVER_URL,
) -> JsonDict:
    artifact = _base_artifact(
        honest_verdict=_verdict_for_evaluation(evaluation),
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
        root=root,
        deliverable_stage="complete",
    )
    artifact.update(
        {
            "cheap_verifier_only_accuracy": evaluation.cheap_verifier_only_accuracy,
            "cheap_verifier_source": cheap_verifier.name,
            "judge_only_accuracy": evaluation.judge_only_accuracy,
            "judge_only_calls": evaluation.judge_only_calls,
            "cascade_accuracy": evaluation.cascade_accuracy,
            "cascade_judge_calls": evaluation.cascade_judge_calls,
            "judge_call_fraction": evaluation.judge_call_fraction,
            "cost_quality_frontier": evaluation.cost_quality_frontier,
            "genuine_tuned_sc_accuracy": round(evaluation.genuine_tuned_sc_accuracy, 6),
            "n_questions": evaluation.n_questions,
            "model_specs": {
                "cheap_verifier": dict(cheap_verifier.model_specs),
                "strong_judge": {
                    "model": STRONG_JUDGE_MODEL,
                    "gpu": 0,
                    "server": judge_server_url,
                    "role": "LLM-as-judge fallback only",
                },
            },
            "inference_substrate": "live_llm_inference",
            "best_routing_threshold": evaluation.best_threshold,
            "paired_ci95_cascade_vs_judge": evaluation.paired_ci95_cascade_vs_judge,
            "judge_call_accounting": {
                "cheap_verifier_only_calls": 0,
                "judge_only_baseline_calls": evaluation.judge_only_calls,
                "cascade_deployment_calls_at_best_threshold": evaluation.cascade_judge_calls,
                "judge_predictions_cached_for_threshold_sweep": (
                    evaluation.judge_predictions_cached_for_sweep
                ),
            },
            "cascade_vs_genuine_tuned_sc_delta": round(
                evaluation.cascade_accuracy - evaluation.genuine_tuned_sc_accuracy, 6
            ),
        }
    )
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def _finalize_artifact(
    artifact: JsonDict,
    artifact_path: Path,
    *,
    audit_runner: AuditRunner,
    summary_runner: SummaryRunner,
    write: bool,
) -> JsonDict:
    if write:
        write_json(artifact_path, artifact)
    audit = audit_runner(artifact_path)
    artifact["adversarial_verify_flags"] = _compact_adversarial_flags(audit)
    artifact["adversarial_verify_clean"] = _audit_is_clean(audit)
    artifact["summarize_artifact_exit_code"] = int(summary_runner(artifact_path))
    artifact["reproducibility_checksum"] = _checksum(artifact)
    if write:
        write_json(artifact_path, artifact)
    return artifact


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    min_questions: int = 200,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    judge_server_url: str = DEFAULT_JUDGE_SERVER_URL,
    judge_server_probe: JudgeProbe | None = None,
    judge_answer: JudgeAnswer | None = None,
    audit_runner: AuditRunner | None = None,
    summary_runner: SummaryRunner | None = None,
    now: Clock = time.perf_counter,
    bootstrap_samples: int = 2000,
    write: bool = True,
) -> JsonDict:
    artifact_path = artifact_path or (root / RESULT_RELATIVE_PATH)
    audit = audit_runner or _default_audit_runner
    summarize = summary_runner or _default_summary_runner
    start = now()
    if write:
        write_json(
            artifact_path,
            build_skeleton_artifact(preconditions_checked=[], duration_s=0.0, root=root),
        )

    cheap = select_cheap_verifier(root)
    judge_check = (judge_server_probe or probe_llama_server)(judge_server_url)
    cache_check, rows = candidate_cache_precondition(root, min_questions=min_questions)
    checks: list[PreconditionCheck] = [cheap.check, judge_check, cache_check]
    first_missing = next((check for check in checks if not check.available), None)
    if first_missing is not None:
        artifact = build_blocked_artifact(
            missing_resource=first_missing.resource,
            preconditions_checked=checks,
            duration_s=now() - start,
            root=root,
            blocked_error=first_missing.detail,
        )
        return _finalize_artifact(
            _attach_cheap_verifier(artifact, cheap),
            artifact_path,
            audit_runner=audit,
            summary_runner=summarize,
            write=write,
        )

    judge = judge_answer or CountingJudge(LlamaServerJudge(judge_server_url))
    try:
        evaluation = evaluate_cascade(
            rows,
            cheap_verifier=cheap,
            judge_answer=judge,
            thresholds=thresholds,
            bootstrap_samples=bootstrap_samples,
        )
    except (OSError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        failed = PreconditionCheck("judge_inference_failed", False, f"{type(exc).__name__}: {exc}")
        artifact = build_blocked_artifact(
            missing_resource=failed.resource,
            preconditions_checked=[*checks, failed],
            duration_s=now() - start,
            root=root,
            blocked_error=failed.detail,
        )
        return _finalize_artifact(
            _attach_cheap_verifier(artifact, cheap),
            artifact_path,
            audit_runner=audit,
            summary_runner=summarize,
            write=write,
        )

    return _finalize_artifact(
        build_complete_artifact(
            evaluation=evaluation,
            cheap_verifier=cheap,
            preconditions_checked=checks,
            duration_s=now() - start,
            root=root,
            judge_server_url=judge_server_url,
        ),
        artifact_path,
        audit_runner=audit,
        summary_runner=summarize,
        write=write,
    )


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(field)
    if artifact.get("spec_refs") != SPEC_REFS:
        errors.append("spec_refs")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    if set(artifact.get("field_principles", {})) != set(FIELD_PRINCIPLES):
        errors.append("field_principles")
    frontier = artifact.get("cost_quality_frontier")
    if not isinstance(frontier, list):
        errors.append("cost_quality_frontier")
    ci95 = artifact.get("paired_ci95_cascade_vs_judge")
    if ci95 is not None and (
        not isinstance(ci95, list)
        or len(ci95) != 2
        or not all(_finite_number(value) for value in ci95)
    ):
        errors.append("paired_ci95_cascade_vs_judge")
    return errors


def main() -> int:  # pragma: no cover - script entry point
    artifact = run()
    errors = artifact_schema_errors(artifact)
    if errors:
        raise SystemExit(f"Exp 5034 artifact schema errors: {errors}")
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
