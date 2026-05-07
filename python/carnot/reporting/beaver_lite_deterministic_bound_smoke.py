"""Exp 1470 BEAVER-lite deterministic-bound smoke.

This module is intentionally narrow: it reuses the existing BEAVER-lite
prefix-bound code, evaluates exactly three tiny arithmetic constraints, and
writes the terminal fit artifact selected by Exp 1465.  It does not implement a
VNN-COMP runner, VNNLIB parser, or broad BEAVER benchmark reproduction.

Spec: REQ-VERIFY-1470, SCENARIO-VERIFY-1470.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from carnot.verify.beaver_lite import BEAVERLiteBounder, CompletionCandidate, MockLogprobProvider

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260507"
BENCHMARK_FAMILY = "BEAVER-style deterministic bounds"
SCHEMA = "beaver_lite_deterministic_bound_smoke_v1"
EXPERIMENT = "1470_beaver_lite_deterministic_bound_smoke"

DEFAULT_OUT_PATH = (
    REPO_ROOT / "results" / "experiment_1470_beaver_lite_deterministic_bound_smoke.json"
)
DEFAULT_EXP1468_ARTIFACT_PATH = (
    REPO_ROOT / "results" / "experiment_1468_live_sota_logprob_telemetry_preflight.json"
)
DEFAULT_EXP1468_MANIFEST_PATH = REPO_ROOT / "results" / "live_sota_telemetry_manifest_1468.jsonl"

MANDATED_SOTA_GGUF_MODELS = {
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
}
REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "benchmark_family",
    "questions_evaluated",
    "prefix_closed_constraint",
    "unsafe_mass_bounds",
    "empirical_violation_rates",
    "bound_is_sound",
    "mock_or_live_logprobs",
    "external_fit_verdict",
    "broad_benchmark_deferred",
    "honest_verdict",
}
DEFAULT_TOY_QUESTIONS = (
    "Janet has 10 marbles and gives away 3. How many remain?",
    "A box has 4 red balls and 5 blue balls. How many balls are in the box?",
    "Luis read 12 pages on Monday and 8 on Tuesday. How many pages did he read?",
)
DEFAULT_TOY_EXPECTED_ANSWERS = ("7", "9", "20")
FINAL_INTEGER_CONSTRAINT_DESCRIPTION = (
    "terminal response text must end with an integer in the inclusive range [0, 9999]"
)


class Exp1468TelemetryLogprobProvider:
    """Expose one existing Exp 1468 live telemetry row as a BEAVER-lite provider."""

    mock_logprobs_used = False

    def __init__(self, row: Mapping[str, Any]) -> None:
        self._row = dict(row)

    def enumerate_completions(
        self,
        prompt: str,
        top_k: int,
        max_tokens: int,
    ) -> list[CompletionCandidate]:
        """Return the logged completion without making a fresh LLM call."""

        del prompt, top_k, max_tokens
        token_logprobs = [float(value) for value in self._row["token_logprobs"]]
        token_texts = tuple(str(token) for token in self._row.get("token_texts", ()))
        text = str(self._row["response_text"])
        return [
            CompletionCandidate(
                text=text,
                tokens=token_texts + ("<eos>",),
                logprob=sum(token_logprobs),
                terminal=True,
            )
        ]


def write_in_progress_artifact(path: str | Path = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-VERIFY-1470-1: write the durable startup artifact first."""

    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "run_date": RUN_DATE,
        "spec": ["REQ-VERIFY-1470", "SCENARIO-VERIFY-1470"],
        "status": "in_progress",
        "benchmark_family": BENCHMARK_FAMILY,
        "questions_evaluated": [],
        "prefix_closed_constraint": [],
        "unsafe_mass_bounds": [],
        "empirical_violation_rates": [],
        "bound_is_sound": False,
        "mock_or_live_logprobs": "pending",
        "external_fit_verdict": "pending",
        "broad_benchmark_deferred": True,
        "honest_verdict": "in_progress",
    }
    return _write_json(Path(path), artifact)


def compatible_exp1468_rows(
    artifact_path: str | Path = DEFAULT_EXP1468_ARTIFACT_PATH,
    manifest_path: str | Path = DEFAULT_EXP1468_MANIFEST_PATH,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Load compatible live logprob rows from Exp 1468 without generating text."""

    artifact_file = Path(artifact_path)
    manifest_file = Path(manifest_path)
    if not artifact_file.exists() or not manifest_file.exists():
        return []
    summary = json.loads(artifact_file.read_text(encoding="utf-8"))
    if not _summary_has_compatible_exp1468_logprobs(summary):
        return []

    rows: list[dict[str, Any]] = []
    for line in manifest_file.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if _row_has_compatible_exp1468_logprobs(row):
            rows.append(row)
        if len(rows) == limit:
            break
    return rows


def build_artifact(
    *,
    questions: Sequence[str],
    constraints: Sequence[Mapping[str, Any]],
    unsafe_mass_bounds: Sequence[float],
    empirical_violation_rates: Sequence[float],
    mock_or_live_logprobs: str,
    model_used: str | None,
    n_completions: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Build and validate the terminal Exp 1470 artifact."""

    bounds = [float(value) for value in unsafe_mass_bounds]
    rates = [float(value) for value in empirical_violation_rates]
    sound = all(0.0 <= bound <= 1.0 and bound + 1e-12 >= rate for bound, rate in zip(bounds, rates))
    if not sound:
        honest_verdict = "bound_violated_bug"
    elif mock_or_live_logprobs == "live_exp1468":
        honest_verdict = "sound_bound_live_exp1468"
    else:
        honest_verdict = "sound_bound_mock_logprobs"

    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "run_date": RUN_DATE,
        "spec": ["REQ-VERIFY-1470", "SCENARIO-VERIFY-1470"],
        "status": "complete",
        "benchmark_family": BENCHMARK_FAMILY,
        "questions_evaluated": list(questions),
        "prefix_closed_constraint": [dict(item) for item in constraints],
        "unsafe_mass_bounds": bounds,
        "empirical_violation_rates": rates,
        "bound_is_sound": sound,
        "mock_or_live_logprobs": mock_or_live_logprobs,
        "model_used": model_used,
        "n_completions": list(n_completions) if n_completions is not None else [],
        "external_fit_verdict": (
            "adopted_minimal_beaver_smoke_fit" if sound else "beaver_smoke_bound_unsound_bug"
        ),
        "broad_benchmark_deferred": True,
        "honest_verdict": honest_verdict,
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the REQ-VERIFY-1470 schema and soundness gate."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if artifact["mock_or_live_logprobs"] not in {"live_exp1468", "mock_logprobs"}:
        raise ValueError("mock_or_live_logprobs must be live_exp1468 or mock_logprobs")
    questions = list(artifact["questions_evaluated"])
    constraints = list(artifact["prefix_closed_constraint"])
    bounds = [float(value) for value in artifact["unsafe_mass_bounds"]]
    rates = [float(value) for value in artifact["empirical_violation_rates"]]
    if not (len(questions) == len(constraints) == len(bounds) == len(rates) == 3):
        raise ValueError("artifact must evaluate exactly three questions")
    for bound, rate in zip(bounds, rates):
        if not 0.0 <= bound <= 1.0:
            raise ValueError("unsafe mass bounds must be in [0, 1]")
        if not 0.0 <= rate <= 1.0:
            raise ValueError("empirical violation rates must be in [0, 1]")
        if bound + 1e-12 < rate:
            raise ValueError("unsafe mass bound is below empirical violation rate")


def run(
    output_path: str | Path = DEFAULT_OUT_PATH,
    exp1468_artifact_path: str | Path = DEFAULT_EXP1468_ARTIFACT_PATH,
    exp1468_manifest_path: str | Path = DEFAULT_EXP1468_MANIFEST_PATH,
    top_k: int = 10,
    max_tokens: int = 8,
) -> dict[str, Any]:
    """Run the three-question smoke and write the terminal artifact."""

    write_in_progress_artifact(output_path)
    live_rows = compatible_exp1468_rows(exp1468_artifact_path, exp1468_manifest_path, limit=3)
    if len(live_rows) == 3:
        questions = [str(row["prompt"]) for row in live_rows]
        constraints = [_constraint_record(index, row) for index, row in enumerate(live_rows)]
        providers = [Exp1468TelemetryLogprobProvider(row) for row in live_rows]
        mode = "live_exp1468"
        model_used = str(live_rows[0]["hf_id"])
        k = 1
    else:
        questions = list(DEFAULT_TOY_QUESTIONS)
        constraints = [
            _constraint_record(index, None) for index in range(len(DEFAULT_TOY_QUESTIONS))
        ]
        providers = [MockLogprobProvider() for _ in DEFAULT_TOY_QUESTIONS]
        mode = "mock_logprobs"
        model_used = None
        k = top_k

    results = [
        BEAVERLiteBounder(provider=provider, top_k=k, max_tokens=max_tokens).bound_prefix_violation(
            question
        )
        for question, provider in zip(questions, providers)
    ]
    artifact = build_artifact(
        questions=questions,
        constraints=constraints,
        unsafe_mass_bounds=[result.upper_bound for result in results],
        empirical_violation_rates=[result.empirical_rate for result in results],
        mock_or_live_logprobs=mode,
        model_used=model_used,
        n_completions=[result.n_completions for result in results],
    )
    return _write_json(Path(output_path), artifact)


def _summary_has_compatible_exp1468_logprobs(summary: Mapping[str, Any]) -> bool:
    return (
        summary.get("status") == "complete"
        and summary.get("live_sota_model_inference_used") is True
        and summary.get("topk_logprobs_available") is True
        and any(model in MANDATED_SOTA_GGUF_MODELS for model in summary.get("models_used", []))
    )


def _row_has_compatible_exp1468_logprobs(row: Mapping[str, Any]) -> bool:
    token_logprobs = row.get("token_logprobs")
    return (
        row.get("generation_source") == "live_sota_llamacpp"
        and row.get("hf_id") in MANDATED_SOTA_GGUF_MODELS
        and row.get("response_text_available") is True
        and row.get("token_logprobs_available") is True
        and isinstance(token_logprobs, list)
        and len(token_logprobs) > 0
        and str(row.get("family")) in {"gsm8k_style", "fover_style"}
        and _response_matches_expected_answer(row)
    )


def _response_matches_expected_answer(row: Mapping[str, Any]) -> bool:
    expected = str(row.get("expected_answer", "")).strip()
    response = str(row.get("response_text", "")).strip()
    return bool(expected) and response.endswith(expected)


def _constraint_record(index: int, row: Mapping[str, Any] | None) -> dict[str, Any]:
    expected_answer = (
        str(row["expected_answer"]) if row is not None else DEFAULT_TOY_EXPECTED_ANSWERS[index]
    )
    source_case_id = str(row["case_id"]) if row is not None else f"mock_toy_{index + 1}"
    return {
        "constraint_id": f"toy_{index + 1}_terminal_final_integer_0_to_9999",
        "description": FINAL_INTEGER_CONSTRAINT_DESCRIPTION,
        "prefix_closed": True,
        "terminal_only": True,
        "expected_answer": expected_answer,
        "source_case_id": source_case_id,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


__all__ = [
    "BENCHMARK_FAMILY",
    "DEFAULT_OUT_PATH",
    "Exp1468TelemetryLogprobProvider",
    "MANDATED_SOTA_GGUF_MODELS",
    "REQUIRED_ARTIFACT_FIELDS",
    "build_artifact",
    "compatible_exp1468_rows",
    "run",
    "validate_artifact",
    "write_in_progress_artifact",
]
