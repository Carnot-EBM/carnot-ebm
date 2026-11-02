"""Exp 1487 V_1-style pairwise self-verification vs energy.

Spec: REQ-VERIFY-1487, SCENARIO-VERIFY-1487.

This evaluator is intentionally bounded to the Exp 1486 executable-constraint
manifest.  It pairs each model transcript with a deterministic gold transcript,
asks a mandated local GGUF verifier to choose the better answer, and compares
that pairwise choice against Carnot's executable constraint energy plus simple
surface baselines.  The point is to test whether pairwise self-verification
adds signal beyond the constraints that already know the answer is executable.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from carnot.eval import cctu_executable_constraint_microbenchmark as cctu

JsonDict = dict[str, Any]

RUN_DATE = "20260507"
SCHEMA_VERSION = 1
DEFAULT_ARTIFACT_PATH = Path("results/experiment_1487_v1_pairwise_self_verification_vs_energy.json")
DEFAULT_DIAGNOSTIC_PATH = Path("results/v1_pairwise_verification_1487.json")
DEFAULT_EXP1486_ARTIFACT_PATH = (
    Path("results") / "experiment_1486_cctu_executable_constraint_microbenchmark.json"
)
DEFAULT_EXP1486_MANIFEST_PATH = Path("results/cctu_microbenchmark_manifest_1486.jsonl")

MANDATED_MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "name": "Qwen3.6-35B-A3B",
        "role": "flagship_moe_primary_pairwise_verifier",
        "gpu": 0,
    },
    {
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "name": "Gemma4-31B-it",
        "role": "flagship_dense_secondary_pairwise_verifier",
        "gpu": 1,
    },
    {
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "name": "Gemma4-26B-A4B-it",
        "role": "middle_moe_secondary_pairwise_verifier",
        "gpu": 1,
    },
)

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "model_specs",
    "live_sota_model_inference_used",
    "pairwise_verification_complete",
    "benchmark_cases_loaded",
    "candidate_pairs_evaluated",
    "pairwise_accuracy",
    "energy_ranking_accuracy",
    "random_baseline_accuracy",
    "superficial_baseline_accuracy",
    "pairwise_delta_over_energy",
    "improvement_allowed",
    "diagnostic_path",
    "tests_run",
    "honest_verdict",
}

PAIRWISE_PROMPT_TEMPLATE = """You are a pairwise verifier for executable tool-use answers.
Choose the better answer for the same case. Use the local tool constraints,
tool-result consistency, final-answer correctness, and verifier self-check.
Return exactly one JSON object: {{"choice": "A"}} or {{"choice": "B"}}.

Case prompt:
{case_prompt}

Answer A:
{answer_a}

Answer B:
{answer_b}
"""


@dataclass(frozen=True)
class CandidatePair:
    """One valid/invalid answer pair for a single Exp 1486 case.

    The pair stores the case object rather than only text so every score can be
    recomputed by the executable validator.  That keeps the evaluation honest:
    a low energy score means the transcript satisfies the actual local tool
    constraints, not just a lexical heuristic.
    """

    pair_id: str
    case_id: str
    case: cctu.BenchmarkCase
    answer_a: str
    answer_b: str
    correct_label: str
    invalid_source: str
    source_model_hf_id: str | None

    @property
    def answer_by_label(self) -> dict[str, str]:
        return {"A": self.answer_a, "B": self.answer_b}


CollectPairwiseChoicesFn = Callable[[JsonDict, list[CandidatePair]], JsonDict]
ResolverFn = Callable[[str], str | None]
LlamaImporterFn = Callable[[], tuple[bool, type[Any] | None, str | None]]


def write_in_progress_artifact(
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    *,
    diagnostic_path: Path | str = DEFAULT_DIAGNOSTIC_PATH,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """REQ-VERIFY-1487: write the durable startup artifact before row loading."""

    artifact: JsonDict = {
        "status": "in_progress",
        "run_date": run_date,
        "schema_version": SCHEMA_VERSION,
        "model_specs": [spec["hf_id"] for spec in MANDATED_MODEL_SPECS],
        "live_sota_model_inference_used": False,
        "pairwise_verification_complete": False,
        "benchmark_cases_loaded": 0,
        "candidate_pairs_evaluated": 0,
        "pairwise_accuracy": None,
        "energy_ranking_accuracy": None,
        "random_baseline_accuracy": None,
        "superficial_baseline_accuracy": None,
        "pairwise_delta_over_energy": None,
        "improvement_allowed": False,
        "diagnostic_path": _display_path(diagnostic_path),
        "tests_run": [],
        "honest_verdict": "in_progress",
    }
    _write_json(Path(output_path), artifact)
    return artifact


def load_exp1486_rows(
    artifact_path: Path | str = DEFAULT_EXP1486_ARTIFACT_PATH,
    manifest_path: Path | str = DEFAULT_EXP1486_MANIFEST_PATH,
) -> tuple[JsonDict, list[JsonDict]]:
    """Load the completed Exp 1486 artifact and JSONL manifest."""

    artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in Path(manifest_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if artifact.get("status") != "complete":
        raise ValueError("Exp 1486 artifact must be complete")
    if artifact.get("executable_constraint_benchmark_ready") is not True:
        raise ValueError("Exp 1486 benchmark must be ready")
    if artifact.get("live_sota_model_inference_used") is not True:
        raise ValueError("Exp 1486 must use live SOTA inference")
    return artifact, rows


def construct_candidate_pairs(
    manifest_rows: Iterable[Mapping[str, Any]],
    cases: Iterable[cctu.BenchmarkCase] | None = None,
) -> list[CandidatePair]:
    """Build valid/invalid candidate pairs from Exp 1486 rows.

    Every pair contains a deterministic compliant transcript and an invalid
    transcript.  If a model row is already invalid, that row becomes the
    negative candidate.  If a row is valid, the negative candidate is a bounded
    mutation of the compliant transcript so the benchmark still contributes a
    pair without importing any non-1486 data.
    """

    case_by_id = {case.case_id: case for case in (cases or cctu.build_benchmark_cases())}
    pairs: list[CandidatePair] = []
    seen_case_ids: set[str] = set()
    for row in manifest_rows:
        case_id = str(row.get("case_id") or "")
        if case_id in seen_case_ids:
            continue
        case = case_by_id.get(case_id)
        if case is None:
            continue
        model_output = str(row.get("model_output") or "")
        if _row_is_base_valid(row):
            invalid_answer = _invalid_transcript_for_case(case)
            invalid_source = "synthetic_invalid_from_valid_output"
        else:
            invalid_answer = model_output
            invalid_source = "exp1486_model_output"
        valid_answer = cctu.compliant_transcript_for_case(case)
        pair_index = len(pairs)
        valid_is_a = pair_index % 2 == 0
        answer_a = valid_answer if valid_is_a else invalid_answer
        answer_b = invalid_answer if valid_is_a else valid_answer
        pairs.append(
            CandidatePair(
                pair_id=f"exp1487_pair_{pair_index + 1:03d}_{case.case_id}",
                case_id=case.case_id,
                case=case,
                answer_a=answer_a,
                answer_b=answer_b,
                correct_label="A" if valid_is_a else "B",
                invalid_source=invalid_source,
                source_model_hf_id=(
                    str(row.get("model_hf_id")) if row.get("model_hf_id") else None
                ),
            )
        )
        seen_case_ids.add(case_id)
    return pairs


def carnot_energy(answer_text: str, case: cctu.BenchmarkCase) -> int:
    """Return executable-constraint energy where zero means fully valid."""

    validation = cctu.validate_transcript(case, answer_text)
    validator = validation["validator_result"]
    failed_checks = [
        not bool(validator["tool_call_structure_valid"]),
        not bool(validator["tool_result_consistent"]),
        not bool(validator["final_answer_valid"]),
        not bool(validator["verifier_outcome_valid"]),
    ]
    if validator["parse_error"] is not None:
        failed_checks.append(True)
    if validator["tool_result_error"] is not None:
        failed_checks.append(True)
    return sum(failed_checks)


def energy_decision(pair: CandidatePair) -> str | None:
    """Choose the lower Carnot energy answer, or return None on a tie."""

    energy_a = carnot_energy(pair.answer_a, pair.case)
    energy_b = carnot_energy(pair.answer_b, pair.case)
    if energy_a == energy_b:
        return None
    return "A" if energy_a < energy_b else "B"


def beaver_style_decision(pair: CandidatePair) -> str | None:
    """Choose the lower deterministic unsafe-bound answer, or None on a tie.

    This is a BEAVER-style bound proxy rather than a broad BEAVER reproduction:
    candidates that violate any executable terminal constraint get unsafe mass
    1.0, while fully valid transcripts get 0.0.  The bounded form is enough for
    this experiment because both candidates are scored on the same fixed local
    executable constraint.
    """

    bound_a = _beaver_style_unsafe_bound(pair.answer_a, pair.case)
    bound_b = _beaver_style_unsafe_bound(pair.answer_b, pair.case)
    if bound_a == bound_b:
        return None
    return "A" if bound_a < bound_b else "B"


def parse_pairwise_choice(output_text: str) -> str | None:
    """Parse a bounded local-verifier A/B choice."""

    obj = cctu.extract_json_object(output_text)
    if obj is not None:
        for key in ("choice", "winner", "answer"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip().upper() in {"A", "B"}:
                return value.strip().upper()
    stripped = output_text.strip().upper()
    if stripped in {"A", "B"}:
        return stripped
    match = re.search(r"\b(?:ANSWER|CHOICE|WINNER)\s*[:=]\s*([AB])\b", stripped)
    if match:
        return match.group(1)
    return None


def collect_live_pairwise_choices(
    spec: JsonDict,
    pairs: list[CandidatePair],
    *,
    resolver: ResolverFn | None = None,
    llama_importer: LlamaImporterFn | None = None,
    env_preparer: Callable[[], JsonDict] | None = None,
) -> JsonDict:
    """Ask one mandated local GGUF model for pairwise choices."""

    hf_id = str(spec.get("hf_id") or "")
    resolver_fn = resolver or cctu._default_resolver  # type: ignore[attr-defined]
    model_path = spec.get("model_path") or resolver_fn(hf_id)
    if not model_path:
        return {
            "summary": {
                "hf_id": hf_id,
                "model_name": spec.get("name"),
                "model_used": False,
                "blocker": "model_not_cached",
            },
            "decisions": [],
        }

    prepare_env = env_preparer or cctu.prepare_llama_environment
    env_details = prepare_env()
    importer = llama_importer or cctu._default_llama_importer  # type: ignore[attr-defined]
    ok, llama_class, import_error = importer()
    if not ok or llama_class is None:
        return {
            "summary": {
                "hf_id": hf_id,
                "model_name": spec.get("name"),
                "model_path": str(model_path),
                "model_used": False,
                "blocker": import_error or "llama_cpp_import_failed",
                "env_details": env_details,
            },
            "decisions": [],
        }

    llm = None
    decisions: list[JsonDict] = []
    load_start = time.monotonic()
    try:
        llm = llama_class(
            model_path=str(model_path),
            n_gpu_layers=-1,
            main_gpu=int(spec.get("gpu") or 0),
            n_ctx=4096,
            seed=1487,
            verbose=False,
        )
    except Exception as exc:
        return {
            "summary": {
                "hf_id": hf_id,
                "model_name": spec.get("name"),
                "model_path": str(model_path),
                "model_used": False,
                "blocker": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.monotonic() - load_start, 6),
                "env_details": env_details,
            },
            "decisions": [],
        }

    try:
        for pair in pairs:
            started = time.monotonic()
            prompt = pairwise_prompt(pair)
            try:
                result = llm(
                    prompt,
                    max_tokens=24,
                    temperature=0.0,
                    top_p=1.0,
                    stop=["</s>", "<eos>"],
                    echo=False,
                )
                raw_output = cctu._completion_text(result)  # type: ignore[attr-defined]
                blocker = None if raw_output.strip() else "empty_generation"
            except Exception as exc:
                raw_output = ""
                blocker = f"{type(exc).__name__}: {exc}"
            decisions.append(
                {
                    "pair_id": pair.pair_id,
                    "choice": parse_pairwise_choice(raw_output),
                    "raw_output": raw_output,
                    "blocker": blocker,
                    "elapsed_seconds": round(time.monotonic() - started, 6),
                }
            )
    finally:
        cctu._close_llama(llm)  # type: ignore[attr-defined]

    model_used = any(decision.get("blocker") is None for decision in decisions)
    return {
        "summary": {
            "hf_id": hf_id,
            "model_name": spec.get("name"),
            "model_path": str(model_path),
            "model_used": model_used,
            "blocker": None if model_used else "no_usable_pairwise_generations",
            "env_details": env_details,
        },
        "decisions": decisions,
    }


def pairwise_prompt(pair: CandidatePair) -> str:
    """Build the local verifier prompt for one candidate pair."""

    return PAIRWISE_PROMPT_TEMPLATE.format(
        case_prompt=pair.case.prompt,
        answer_a=pair.answer_a,
        answer_b=pair.answer_b,
    )


def score_pairs(
    pairs: list[CandidatePair],
    pairwise_decisions: Mapping[str, str | None],
) -> tuple[list[JsonDict], JsonDict]:
    """Score pairwise, energy, BEAVER-style, and superficial baselines."""

    scored_pairs: list[JsonDict] = []
    for pair in pairs:
        energy_a = carnot_energy(pair.answer_a, pair.case)
        energy_b = carnot_energy(pair.answer_b, pair.case)
        beaver_a = _beaver_style_unsafe_bound(pair.answer_a, pair.case)
        beaver_b = _beaver_style_unsafe_bound(pair.answer_b, pair.case)
        pairwise_choice = pairwise_decisions.get(pair.pair_id)
        length_choice = _shorter_length_decision(pair)
        format_choice = _format_validity_decision(pair)
        scored_pairs.append(
            {
                "pair_id": pair.pair_id,
                "case_id": pair.case_id,
                "correct_label": pair.correct_label,
                "source_model_hf_id": pair.source_model_hf_id,
                "invalid_source": pair.invalid_source,
                "pairwise_choice": pairwise_choice,
                "pairwise_correct": pairwise_choice == pair.correct_label,
                "energy_scores": {"A": energy_a, "B": energy_b},
                "energy_choice": _lower_score_decision(energy_a, energy_b),
                "energy_correct": _lower_score_decision(energy_a, energy_b) == pair.correct_label,
                "beaver_style_unsafe_bounds": {"A": beaver_a, "B": beaver_b},
                "beaver_style_choice": _lower_score_decision(beaver_a, beaver_b),
                "beaver_style_correct": _lower_score_decision(beaver_a, beaver_b)
                == pair.correct_label,
                "lengths": {"A": len(pair.answer_a), "B": len(pair.answer_b)},
                "length_choice": length_choice,
                "length_correct": length_choice == pair.correct_label,
                "format_valid": {
                    "A": cctu.extract_json_object(pair.answer_a) is not None,
                    "B": cctu.extract_json_object(pair.answer_b) is not None,
                },
                "format_choice": format_choice,
                "format_correct": format_choice == pair.correct_label,
            }
        )

    pairwise_accuracy = _choice_accuracy(
        [(row["pairwise_choice"], row["correct_label"]) for row in scored_pairs],
        tie_credit=0.0,
    )
    energy_accuracy = _choice_accuracy(
        [(row["energy_choice"], row["correct_label"]) for row in scored_pairs]
    )
    beaver_accuracy = _choice_accuracy(
        [(row["beaver_style_choice"], row["correct_label"]) for row in scored_pairs]
    )
    length_accuracy = _choice_accuracy(
        [(row["length_choice"], row["correct_label"]) for row in scored_pairs]
    )
    format_accuracy = _choice_accuracy(
        [(row["format_choice"], row["correct_label"]) for row in scored_pairs]
    )
    superficial_accuracy = max(length_accuracy, format_accuracy)
    metrics: JsonDict = {
        "pairwise_accuracy": pairwise_accuracy,
        "energy_ranking_accuracy": energy_accuracy,
        "beaver_style_ranking_accuracy": beaver_accuracy,
        "random_baseline_accuracy": 0.5 if scored_pairs else 0.0,
        "response_length_accuracy": length_accuracy,
        "format_validity_accuracy": format_accuracy,
        "superficial_baseline_accuracy": superficial_accuracy,
        "pairwise_delta_over_energy": round(pairwise_accuracy - energy_accuracy, 6),
        "candidate_pairs_evaluated": len(scored_pairs),
    }
    metrics["improvement_allowed"] = improvement_allowed(
        pairwise_accuracy=pairwise_accuracy,
        energy_ranking_accuracy=energy_accuracy,
        superficial_baseline_accuracy=superficial_accuracy,
    )
    return scored_pairs, metrics


def improvement_allowed(
    *,
    pairwise_accuracy: float,
    energy_ranking_accuracy: float,
    superficial_baseline_accuracy: float,
) -> bool:
    """Return whether the V_1 pairwise improvement gate is satisfied."""

    return (
        pairwise_accuracy > energy_ranking_accuracy
        and pairwise_accuracy > superficial_baseline_accuracy
    )


def build_terminal_artifact(
    *,
    benchmark_cases_loaded: int,
    metrics: Mapping[str, Any],
    model_attempts: list[JsonDict],
    diagnostic_path: Path | str,
    tests_run: list[str] | None = None,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Build the completed REQ-VERIFY-1487 artifact."""

    live_used = any(attempt.get("model_used") is True for attempt in model_attempts)
    pairwise_complete = live_used and int(metrics["candidate_pairs_evaluated"]) > 0
    delta = float(metrics["pairwise_delta_over_energy"])
    artifact: JsonDict = {
        "status": "complete",
        "run_date": run_date,
        "schema_version": SCHEMA_VERSION,
        "model_specs": [spec["hf_id"] for spec in MANDATED_MODEL_SPECS],
        "live_sota_model_inference_used": live_used,
        "pairwise_verification_complete": pairwise_complete,
        "benchmark_cases_loaded": int(benchmark_cases_loaded),
        "candidate_pairs_evaluated": int(metrics["candidate_pairs_evaluated"]),
        "pairwise_accuracy": float(metrics["pairwise_accuracy"]),
        "energy_ranking_accuracy": float(metrics["energy_ranking_accuracy"]),
        "random_baseline_accuracy": float(metrics["random_baseline_accuracy"]),
        "superficial_baseline_accuracy": float(metrics["superficial_baseline_accuracy"]),
        "pairwise_delta_over_energy": delta,
        "improvement_allowed": bool(metrics["improvement_allowed"]),
        "diagnostic_path": _display_path(diagnostic_path),
        "tests_run": list(tests_run or []),
        "honest_verdict": _honest_verdict(pairwise_complete, metrics),
        "models_used": [
            str(attempt["hf_id"])
            for attempt in model_attempts
            if attempt.get("model_used") is True and attempt.get("hf_id")
        ],
        "model_attempts": model_attempts,
        "beaver_style_ranking_accuracy": float(metrics["beaver_style_ranking_accuracy"]),
        "response_length_accuracy": float(metrics["response_length_accuracy"]),
        "format_validity_accuracy": float(metrics["format_validity_accuracy"]),
    }
    validate_terminal_artifact(artifact)
    return artifact


def validate_terminal_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the required artifact fields and improvement gate."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if artifact["status"] != "complete":
        raise ValueError("terminal artifact status must be complete")
    if artifact["candidate_pairs_evaluated"] <= 0:
        raise ValueError("candidate_pairs_evaluated must be positive")
    expected_gate = improvement_allowed(
        pairwise_accuracy=float(artifact["pairwise_accuracy"]),
        energy_ranking_accuracy=float(artifact["energy_ranking_accuracy"]),
        superficial_baseline_accuracy=float(artifact["superficial_baseline_accuracy"]),
    )
    if bool(artifact["improvement_allowed"]) is not expected_gate:
        raise ValueError("improvement_allowed must follow the strict gate")


def run_evaluation(
    *,
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    diagnostic_path: Path | str = DEFAULT_DIAGNOSTIC_PATH,
    exp1486_artifact_path: Path | str = DEFAULT_EXP1486_ARTIFACT_PATH,
    exp1486_manifest_path: Path | str = DEFAULT_EXP1486_MANIFEST_PATH,
    run_date: str = RUN_DATE,
    model_specs: Iterable[JsonDict] = MANDATED_MODEL_SPECS,
    collect_pairwise_choices_fn: CollectPairwiseChoicesFn | None = None,
    max_models: int = 1,
    tests_run: list[str] | None = None,
) -> JsonDict:
    """Run Exp 1487 and write diagnostic plus terminal artifacts."""

    output = Path(output_path)
    diagnostic = Path(diagnostic_path)
    write_in_progress_artifact(output, diagnostic_path=diagnostic, run_date=run_date)

    _artifact, rows = load_exp1486_rows(exp1486_artifact_path, exp1486_manifest_path)
    cases = cctu.build_benchmark_cases()
    pairs = construct_candidate_pairs(rows, cases)
    collector = collect_pairwise_choices_fn or collect_live_pairwise_choices
    specs = [dict(spec) for spec in model_specs]
    model_attempts: list[JsonDict] = []
    pairwise_decisions: dict[str, str | None] = {}
    raw_pairwise_rows: list[JsonDict] = []

    for index, spec in enumerate(specs):
        if index >= max_models:
            model_attempts.append(
                {
                    "hf_id": spec.get("hf_id"),
                    "model_name": spec.get("name"),
                    "model_used": False,
                    "blocker": "not_attempted_runtime_budget",
                }
            )
            continue
        collection = collector(spec, pairs)
        summary = dict(collection.get("summary") or {})
        model_attempts.append(summary)
        decisions = list(collection.get("decisions") or [])
        raw_pairwise_rows.extend(decisions)
        for decision in decisions:
            pair_id = str(decision.get("pair_id") or "")
            if pair_id and pair_id not in pairwise_decisions:
                choice = decision.get("choice")
                pairwise_decisions[pair_id] = choice if choice in {"A", "B"} else None
        if any(decision.get("choice") in {"A", "B"} for decision in decisions):
            for remaining_spec in specs[index + 1 :]:
                model_attempts.append(
                    {
                        "hf_id": remaining_spec.get("hf_id"),
                        "model_name": remaining_spec.get("name"),
                        "model_used": False,
                        "blocker": "not_attempted_runtime_budget",
                    }
                )
            break

    scored_pairs, metrics = score_pairs(pairs, pairwise_decisions)
    diagnostic_payload: JsonDict = {
        "schema_version": SCHEMA_VERSION,
        "run_date": run_date,
        "spec": ["REQ-VERIFY-1487", "SCENARIO-VERIFY-1487"],
        "source_artifacts": [
            _display_path(exp1486_artifact_path),
            _display_path(exp1486_manifest_path),
        ],
        "model_attempts": model_attempts,
        "raw_pairwise_decisions": raw_pairwise_rows,
        "baseline_accuracies": {
            "energy_ranking": metrics["energy_ranking_accuracy"],
            "beaver_style_ranking": metrics["beaver_style_ranking_accuracy"],
            "random": metrics["random_baseline_accuracy"],
            "response_length": metrics["response_length_accuracy"],
            "format_validity": metrics["format_validity_accuracy"],
            "best_superficial": metrics["superficial_baseline_accuracy"],
        },
        "pairs": scored_pairs,
    }
    _write_json(diagnostic, diagnostic_payload)

    artifact = build_terminal_artifact(
        benchmark_cases_loaded=len(rows),
        metrics=metrics,
        model_attempts=model_attempts,
        diagnostic_path=diagnostic,
        tests_run=tests_run,
        run_date=run_date,
    )
    _write_json(output, artifact)
    return artifact


def _row_is_base_valid(row: Mapping[str, Any]) -> bool:
    verifier_result = row.get("verifier_result")
    if isinstance(verifier_result, Mapping):
        return bool(verifier_result.get("base_valid"))
    validator = row.get("validator_result")
    if not isinstance(validator, Mapping):
        return False
    return all(
        bool(validator.get(key))
        for key in (
            "tool_call_structure_valid",
            "tool_result_consistent",
            "final_answer_valid",
            "verifier_outcome_valid",
        )
    )


def _invalid_transcript_for_case(case: cctu.BenchmarkCase) -> str:
    payload = json.loads(cctu.compliant_transcript_for_case(case))
    payload["final_answer"] = "__invalid__"
    payload["verifier"] = {"accept": True}
    return json.dumps(payload, sort_keys=True)


def _beaver_style_unsafe_bound(answer_text: str, case: cctu.BenchmarkCase) -> float:
    return 0.0 if carnot_energy(answer_text, case) == 0 else 1.0


def _lower_score_decision(score_a: float, score_b: float) -> str | None:
    if score_a == score_b:
        return None
    return "A" if score_a < score_b else "B"


def _shorter_length_decision(pair: CandidatePair) -> str | None:
    if len(pair.answer_a) == len(pair.answer_b):
        return None
    return "A" if len(pair.answer_a) < len(pair.answer_b) else "B"


def _format_validity_decision(pair: CandidatePair) -> str | None:
    valid_a = cctu.extract_json_object(pair.answer_a) is not None
    valid_b = cctu.extract_json_object(pair.answer_b) is not None
    if valid_a == valid_b:
        return None
    return "A" if valid_a else "B"


def _choice_accuracy(
    choices: list[tuple[str | None, str]],
    *,
    tie_credit: float = 0.5,
) -> float:
    if not choices:
        return 0.0
    total = 0.0
    for choice, correct in choices:
        if choice is None:
            total += tie_credit
        elif choice == correct:
            total += 1.0
    return round(total / len(choices), 6)


def _honest_verdict(pairwise_complete: bool, metrics: Mapping[str, Any]) -> str:
    if not pairwise_complete:
        return "blocked: no mandated local SOTA pairwise verifier completed"
    if metrics["improvement_allowed"]:
        return "complete: pairwise verifier improved over energy and superficial baselines"
    if metrics["pairwise_delta_over_energy"] <= 0:
        return "complete: no V_1 pairwise improvement over executable Carnot energy"
    return "complete: pairwise delta explained by superficial baseline"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _display_path(path: Path | str) -> str:
    as_path = Path(path)
    try:
        return str(as_path.resolve().relative_to(_repo_root()))
    except ValueError:
        return str(as_path)


def _write_json(path: Path, payload: JsonDict) -> JsonDict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for conductor and manual runs."""

    args = list(sys.argv[1:] if argv is None else argv)
    max_models = int(os.getenv("CARNOT_V1_PAIRWISE_1487_MAX_MODELS", "1"))
    if "--all-models" in args:
        max_models = len(MANDATED_MODEL_SPECS)
    artifact = run_evaluation(max_models=max_models)
    print(
        "[exp1487] "
        f"pairwise_complete={artifact['pairwise_verification_complete']} "
        f"pairs={artifact['candidate_pairs_evaluated']} "
        f"pairwise={artifact['pairwise_accuracy']} "
        f"energy={artifact['energy_ranking_accuracy']} "
        f"allowed={artifact['improvement_allowed']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by conductor.
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ARTIFACT_PATH",
    "DEFAULT_DIAGNOSTIC_PATH",
    "MANDATED_MODEL_SPECS",
    "REQUIRED_ARTIFACT_FIELDS",
    "CandidatePair",
    "beaver_style_decision",
    "build_terminal_artifact",
    "carnot_energy",
    "collect_live_pairwise_choices",
    "construct_candidate_pairs",
    "energy_decision",
    "improvement_allowed",
    "load_exp1486_rows",
    "pairwise_prompt",
    "parse_pairwise_choice",
    "run_evaluation",
    "score_pairs",
    "validate_terminal_artifact",
    "write_in_progress_artifact",
]
