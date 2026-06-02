"""Matched-compute FLOP accounting harness for Exp 3727.

Spec: REQ-AR-053, SCENARIO-AR-053-01, SCENARIO-AR-053-02,
SCENARIO-AR-053-03.

This module is deliberately an instrument, not a model runner.  Callers provide
held-out examples plus deterministic EBT and AR generation callbacks; the
harness accounts for inference FLOPs, tunes AR best-of-M to the EBT budget, and
scores both sides at equal compute.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


EXPERIMENT_ID = 3727
REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_PATH = Path("results/experiment_3727_matched_compute_eval_harness.json")
TEST_FILE = "tests/python/test_matched_compute_eval_harness.py"
RANDOM_SEED = 20260602
DEFAULT_TOLERANCE = 0.001

INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts (principle: builds an instrument, "
    "runs no live model)."
)
FLOP_MODEL_DESCRIPTION = (
    "FLOP model: total inference FLOPs = parameter_count * sequence_tokens * "
    "forward_passes. sequence_tokens = prompt_tokens + generated_tokens. EBT "
    "forward_passes = 1 initial sequence pass + K energy-descent passes; AR "
    "best-of-M forward_passes = M independent generation passes. The model "
    "omits attention-cache and hardware constants so matched-compute "
    "comparisons are symmetric and auditable."
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "flop_model_description",
    "unit_tests_added",
    "unit_tests_passed",
    "budget_matcher_tolerance",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix; this is a build+test task, terminal on tests passing.",
    "inference_substrate": (
        "aggregation_from_upstream_artifacts because this builds an instrument "
        "and runs no live model."
    ),
    "flop_model_description": (
        "A transparent documented FLOP model makes matched compute auditable "
        "rather than hand-wavy."
    ),
    "unit_tests_added": "Names the test file; tests must run and assert.",
    "unit_tests_passed": "Honest pass count; a failing harness cannot judge P0.1.",
    "budget_matcher_tolerance": (
        "Defines how tightly AR FLOPs are equalized to EBT FLOPs."
    ),
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Content hash catches drift.",
    "duration_s": "Wall-clock plausibility floor.",
}


@dataclass(frozen=True)
class ReasoningExample:
    """One held-out example with token lengths needed for FLOP accounting."""

    example_id: str
    prompt: str
    gold_answer: str
    prompt_tokens: int
    generated_tokens: int


@dataclass(frozen=True)
class Prediction:
    """A generated answer plus an optional selector score."""

    answer: str
    score: float = 0.0


@dataclass(frozen=True)
class BudgetMatch:
    """Integer AR best-of-M choice and its FLOP error against the EBT target."""

    ar_best_of_m: int
    target_total_flops: int
    ar_total_flops: int
    ar_single_sample_total_flops: int
    relative_error: float
    within_tolerance: bool


@dataclass(frozen=True)
class MatchedComputeResult:
    """Accuracy comparison after AR best-of-M has been matched to EBT FLOPs."""

    n_examples: int
    ebt_correct: int
    ar_correct: int
    ebt_accuracy: float
    ar_accuracy: float
    ebt_total_flops: int
    ar_total_flops: int
    budget_match: BudgetMatch
    verdict: str
    rows: list[dict[str, Any]]


@dataclass(frozen=True)
class UnitTestSummary:
    """Focused unit-test result stored in the artifact."""

    test_file: str
    passed: int
    total: int
    command: str

    def count_string(self) -> str:
        """Return the compact P_of_T_pass value used by the terminal verdict."""
        return f"{self.passed}_of_{self.total}_pass"


EBTPredictor = Callable[[ReasoningExample], Prediction | str]
ARSampler = Callable[[ReasoningExample, int, int], Sequence[Prediction | str]]
ARSelector = Callable[[Sequence[Prediction]], Prediction]


def _require_positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _require_nonnegative_int(name: str, value: int) -> int:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _sequence_tokens(prompt_tokens: int, generated_tokens: int) -> int:
    prompt = _require_nonnegative_int("prompt_tokens", prompt_tokens)
    generated = _require_nonnegative_int("generated_tokens", generated_tokens)
    return _require_positive_int("sequence_tokens", prompt + generated)


def sequence_forward_flops(
    *,
    parameter_count: int,
    sequence_tokens: int,
    forward_passes: int,
) -> int:
    """Return `parameter_count * sequence_tokens * forward_passes` FLOPs."""
    params = _require_positive_int("parameter_count", parameter_count)
    tokens = _require_positive_int("sequence_tokens", sequence_tokens)
    passes = _require_positive_int("forward_passes", forward_passes)
    return params * tokens * passes


def ebt_generation_flops(
    *,
    parameter_count: int,
    prompt_tokens: int,
    generated_tokens: int,
    energy_descent_steps: int,
) -> int:
    """Return EBT FLOPs for one prediction with K energy-descent steps."""
    steps = _require_nonnegative_int("energy_descent_steps", energy_descent_steps)
    return sequence_forward_flops(
        parameter_count=parameter_count,
        sequence_tokens=_sequence_tokens(prompt_tokens, generated_tokens),
        forward_passes=1 + steps,
    )


def ar_generation_flops(
    *,
    parameter_count: int,
    prompt_tokens: int,
    generated_tokens: int,
    best_of_m: int,
) -> int:
    """Return AR FLOPs for one best-of-M prediction."""
    return sequence_forward_flops(
        parameter_count=parameter_count,
        sequence_tokens=_sequence_tokens(prompt_tokens, generated_tokens),
        forward_passes=_require_positive_int("best_of_m", best_of_m),
    )


def total_ebt_flops(
    examples: Sequence[ReasoningExample],
    *,
    parameter_count: int,
    energy_descent_steps: int,
) -> int:
    """Return summed EBT FLOPs over a held-out set."""
    return sum(
        ebt_generation_flops(
            parameter_count=parameter_count,
            prompt_tokens=example.prompt_tokens,
            generated_tokens=example.generated_tokens,
            energy_descent_steps=energy_descent_steps,
        )
        for example in examples
    )


def total_ar_flops(
    examples: Sequence[ReasoningExample],
    *,
    parameter_count: int,
    best_of_m: int,
) -> int:
    """Return summed AR FLOPs over a held-out set."""
    return sum(
        ar_generation_flops(
            parameter_count=parameter_count,
            prompt_tokens=example.prompt_tokens,
            generated_tokens=example.generated_tokens,
            best_of_m=best_of_m,
        )
        for example in examples
    )


def match_ar_best_of_m(
    *,
    target_total_flops: int,
    ar_single_sample_total_flops: int,
    tolerance: float,
) -> BudgetMatch:
    """Choose integer AR best-of-M nearest to the target FLOP budget."""
    target = _require_positive_int("target_total_flops", target_total_flops)
    single = _require_positive_int(
        "ar_single_sample_total_flops", ar_single_sample_total_flops
    )
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")

    ratio = target / single
    candidates = {max(1, math.floor(ratio)), max(1, math.ceil(ratio))}
    best_m = min(candidates, key=lambda m: (abs(single * m - target), m))
    ar_total = single * best_m
    relative_error = abs(ar_total - target) / target
    return BudgetMatch(
        ar_best_of_m=int(best_m),
        target_total_flops=target,
        ar_total_flops=int(ar_total),
        ar_single_sample_total_flops=single,
        relative_error=relative_error,
        within_tolerance=relative_error <= tolerance,
    )


def _coerce_prediction(value: Prediction | str) -> Prediction:
    return value if isinstance(value, Prediction) else Prediction(str(value), score=0.0)


def select_highest_score(predictions: Sequence[Prediction]) -> Prediction:
    """Select the highest-scored AR candidate for best-of-M scoring."""
    if not predictions:
        raise ValueError("predictions must be non-empty")
    return max(enumerate(predictions), key=lambda item: (item[1].score, -item[0]))[1]


def exact_match(answer: str, gold_answer: str) -> bool:
    """Case-insensitive exact-match scorer for synthetic held-out fixtures."""
    return str(answer).strip().lower() == str(gold_answer).strip().lower()


def compare_matched_compute(
    examples: Sequence[ReasoningExample],
    *,
    ebt_predictor: EBTPredictor,
    ar_sampler: ARSampler,
    ebt_parameter_count: int,
    ar_parameter_count: int,
    energy_descent_steps: int,
    tolerance: float = DEFAULT_TOLERANCE,
    random_seed: int = RANDOM_SEED,
    ar_selector: ARSelector = select_highest_score,
) -> MatchedComputeResult:
    """Score EBT and AR on the same held-out set at matched total FLOPs."""
    heldout = list(examples)
    if not heldout:
        raise ValueError("examples must be non-empty")

    ebt_budget = total_ebt_flops(
        heldout,
        parameter_count=ebt_parameter_count,
        energy_descent_steps=energy_descent_steps,
    )
    ar_single = total_ar_flops(heldout, parameter_count=ar_parameter_count, best_of_m=1)
    budget_match = match_ar_best_of_m(
        target_total_flops=ebt_budget,
        ar_single_sample_total_flops=ar_single,
        tolerance=tolerance,
    )

    rows: list[dict[str, Any]] = []
    ebt_correct = 0
    ar_correct = 0
    for index, example in enumerate(heldout):
        ebt_prediction = _coerce_prediction(ebt_predictor(example))
        ar_predictions = [
            _coerce_prediction(prediction)
            for prediction in ar_sampler(
                example, budget_match.ar_best_of_m, random_seed + index
            )
        ]
        if len(ar_predictions) != budget_match.ar_best_of_m:
            raise ValueError("ar_sampler must return exactly best_of_m predictions")
        ar_prediction = ar_selector(ar_predictions)
        ebt_is_correct = exact_match(ebt_prediction.answer, example.gold_answer)
        ar_is_correct = exact_match(ar_prediction.answer, example.gold_answer)
        ebt_correct += int(ebt_is_correct)
        ar_correct += int(ar_is_correct)
        rows.append(
            {
                "example_id": example.example_id,
                "gold_answer": example.gold_answer,
                "ebt_answer": ebt_prediction.answer,
                "ar_answer": ar_prediction.answer,
                "ebt_correct": ebt_is_correct,
                "ar_correct": ar_is_correct,
            }
        )

    n_examples = len(heldout)
    ebt_accuracy = ebt_correct / n_examples
    ar_accuracy = ar_correct / n_examples
    verdict = (
        "ebt_higher_at_equal_flops"
        if ebt_accuracy > ar_accuracy
        else "ar_higher_at_equal_flops"
        if ar_accuracy > ebt_accuracy
        else "tie_at_equal_flops"
    )
    return MatchedComputeResult(
        n_examples=n_examples,
        ebt_correct=ebt_correct,
        ar_correct=ar_correct,
        ebt_accuracy=ebt_accuracy,
        ar_accuracy=ar_accuracy,
        ebt_total_flops=ebt_budget,
        ar_total_flops=budget_match.ar_total_flops,
        budget_match=budget_match,
        verdict=verdict,
        rows=rows,
    )


def synthetic_matched_compute_fixture() -> MatchedComputeResult:
    """Run a deterministic no-model fixture that exercises the full driver."""
    examples = [
        ReasoningExample("p1", "question 1", "A", prompt_tokens=2, generated_tokens=3),
        ReasoningExample("p2", "question 2", "B", prompt_tokens=2, generated_tokens=3),
        ReasoningExample("p3", "question 3", "C", prompt_tokens=2, generated_tokens=3),
        ReasoningExample("p4", "question 4", "D", prompt_tokens=2, generated_tokens=3),
    ]

    def ebt_predict(example: ReasoningExample) -> Prediction:
        answers = {"p1": "A", "p2": "B", "p3": "C", "p4": "wrong"}
        return Prediction(answers[example.example_id], score=0.0)

    def ar_sample(
        example: ReasoningExample,
        best_of_m: int,
        seed: int,
    ) -> list[Prediction]:
        selected_answers = {"p1": "A", "p2": "wrong", "p3": "wrong", "p4": "D"}
        return [
            Prediction(selected_answers[example.example_id], score=1.0 - index * 0.01)
            for index in range(best_of_m)
        ]

    return compare_matched_compute(
        examples,
        ebt_predictor=ebt_predict,
        ar_sampler=ar_sample,
        ebt_parameter_count=100,
        ar_parameter_count=100,
        energy_descent_steps=4,
        tolerance=DEFAULT_TOLERANCE,
        random_seed=RANDOM_SEED,
    )


def comparison_to_dict(result: MatchedComputeResult) -> dict[str, Any]:
    """Convert a comparison result into stable JSON primitives."""
    return {
        "n_examples": result.n_examples,
        "ebt_correct": result.ebt_correct,
        "ar_correct": result.ar_correct,
        "ebt_accuracy": result.ebt_accuracy,
        "ar_accuracy": result.ar_accuracy,
        "ebt_total_flops": result.ebt_total_flops,
        "ar_total_flops": result.ar_total_flops,
        "budget_match": asdict(result.budget_match),
        "matched_compute_verdict": result.verdict,
        "rows": result.rows,
    }


def reproducibility_checksum(
    result: MatchedComputeResult,
    unit_tests: UnitTestSummary,
) -> str:
    """Hash the deterministic fixture, FLOP model, tolerance, seed, and tests."""
    payload = {
        "experiment": EXPERIMENT_ID,
        "flop_model_description": FLOP_MODEL_DESCRIPTION,
        "random_seed": RANDOM_SEED,
        "comparison": comparison_to_dict(result),
        "unit_tests": asdict(unit_tests),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def build_artifact(
    result: MatchedComputeResult,
    unit_tests: UnitTestSummary,
    *,
    duration_s: float,
) -> dict[str, Any]:
    """Build the required Exp 3727 artifact with bare top-level values."""
    test_count = unit_tests.count_string()
    return {
        "schema": "carnot.experiment_3727_matched_compute_eval_harness.v1",
        "experiment": EXPERIMENT_ID,
        "honest_verdict": (
            "complete: matched_compute_eval_harness_built_flop_accounting_"
            f"documented_unit_tests_{test_count}"
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "flop_model_description": FLOP_MODEL_DESCRIPTION,
        "unit_tests_added": TEST_FILE,
        "unit_tests_passed": test_count,
        "unit_test_command": unit_tests.command,
        "budget_matcher_tolerance": DEFAULT_TOLERANCE,
        "matched_compute_report": comparison_to_dict(result),
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum(result, unit_tests),
        "duration_s": round(float(duration_s), 6),
    }


def validate_artifact(artifact: dict[str, Any]) -> list[str]:
    """Return schema problems that would make the Exp 3727 artifact invalid."""
    errors: list[str] = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")
    if not str(artifact.get("honest_verdict", "")).startswith("complete:"):
        errors.append("honest_verdict must start with complete:")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must record the aggregation instrument substrate")
    if artifact.get("flop_model_description") != FLOP_MODEL_DESCRIPTION:
        errors.append("flop_model_description must match the documented model")
    unit_tests_passed = str(artifact.get("unit_tests_passed", ""))
    if unit_tests_passed not in str(artifact.get("honest_verdict", "")):
        errors.append("unit_tests_passed must match the terminal verdict count")
    if artifact.get("unit_tests_added") != TEST_FILE:
        errors.append("unit_tests_added must name the focused unit test file")
    if artifact.get("budget_matcher_tolerance") != DEFAULT_TOLERANCE:
        errors.append("budget_matcher_tolerance must equal the harness tolerance")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed must equal the harness seed")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or len(checksum) != 64:
        errors.append("reproducibility_checksum must be a sha256 hex string")
    if float(artifact.get("duration_s") or 0.0) <= 0.0:
        errors.append("duration_s must be positive")
    principles = artifact.get("field_principles")
    if not isinstance(principles, dict) or set(REQUIRED_ARTIFACT_FIELDS) - set(principles):
        errors.append("field_principles must cover all required artifact fields")
    return errors


def write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    """Write a stable JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_unit_tests() -> UnitTestSummary:
    """Run the focused harness unit tests and return an honest pass count."""
    python_executable = REPO_ROOT / ".venv" / "bin" / "python"
    command = [
        str(python_executable if python_executable.exists() else Path(sys.executable)),
        "-m",
        "pytest",
        TEST_FILE,
        "-q",
        "--basetemp=/tmp/carnot-exp3727-pytest",
        "--override-ini",
        "addopts=",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    output = f"{completed.stdout}\n{completed.stderr}"
    if completed.returncode != 0:
        raise RuntimeError(output.strip())
    passed = 0
    total = 0
    words = output.replace("\n", " ").split()
    for index, token in enumerate(words[:-1]):
        if token.isdigit() and words[index + 1] == "passed":
            passed = int(token)
            total = int(token)
            break
    if passed <= 0:
        raise RuntimeError(f"could not parse pytest pass count from: {output.strip()}")
    return UnitTestSummary(
        test_file=TEST_FILE,
        passed=passed,
        total=total,
        command=" ".join(command),
    )


def run_experiment(result_path: Path = RESULT_PATH) -> dict[str, Any]:
    """Run the synthetic instrument check, focused tests, and write the artifact."""
    start = time.monotonic()
    result = synthetic_matched_compute_fixture()
    unit_tests = run_unit_tests()
    artifact = build_artifact(result, unit_tests, duration_s=time.monotonic() - start)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    write_artifact(result_path, artifact)
    return artifact
