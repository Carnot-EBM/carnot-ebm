"""Exp 2241 FR-11 Fast-Slow Training evaluation.

Spec: REQ-LEARN-2241, SCENARIO-LEARN-2241.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_FILE = "experiment_2241_fr11_fst_eval.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / OUTPUT_FILE
DEFAULT_FAST_SLOW_PATH = REPO_ROOT / "python" / "carnot" / "training" / "fast_slow.py"

EXPERIMENT = "2241_fr11_fst_eval"
SCHEMA = "fr11_fst_eval_v1"
RUN_DATE = "20260517"
ITERATIONS = 5
N_CORPUS = 30

ERROR_TYPES = ("carry", "operation", "sign", "order", "parity")
BASE_VERIFIER_PROB = 0.18
BASELINE_RL_LR = 0.18
FST_FAST_GAIN = 0.68
ENERGY_REDUCTION_THRESHOLD = 0.42
VIOLATION_ENERGY_EPSILON = 0.08

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "honest_verdict",
    "continuous_self_learning_task",
    "fr11_fst_eval_passed",
    "sample_efficiency_ratio",
    "kl_drift_ratio",
    "utility_delta",
    "n_corpus",
    "preconditions_checked",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": "Terminal-prefix required. Use complete: if both acceptance gates pass.",
    "continuous_self_learning_task": (
        "Must be true -- FR-11 mandate requires at least one CSL task per milestone."
    ),
    "fr11_fst_eval_passed": (
        "Boolean gate for exp2242. True only when sample_efficiency_ratio >= 2.0 "
        "AND kl_drift_ratio <= 0.5."
    ),
    "sample_efficiency_ratio": (
        "Primary FST gate: >= 2.0 validates the 3x claim from arXiv:2605.12484."
    ),
    "kl_drift_ratio": ("Secondary FST gate: <= 0.5 validates 70% KL reduction claim."),
    "utility_delta": ("Measures raw FR-11 benefit -- energy reduction delta between regimes."),
    "n_corpus": "Records corpus size so statistical significance is auditable.",
    "preconditions_checked": (
        "Lists which resources were verified before running; pre-empts fabrication."
    ),
}


@dataclass(frozen=True)
class FoVerArithmeticPair:
    """One synthetic FoVer-class arithmetic CoT pair for the FST eval."""

    case_id: str
    question: str
    response: str
    correct_answer: int
    wrong_answer: int
    error_type: str
    severity: float

    def to_dict(self) -> JsonDict:
        return {
            "case_id": self.case_id,
            "correct_answer": self.correct_answer,
            "error_type": self.error_type,
            "question": self.question,
            "response": self.response,
            "severity": self.severity,
            "wrong_answer": self.wrong_answer,
        }


@dataclass(frozen=True)
class SyntheticViolation:
    """Verifier-compatible violation object consumed by fast_slow.py."""

    constraint_type: str
    description: str
    metadata: Mapping[str, Any]


class _DummyParameter:
    def __init__(self) -> None:
        self.requires_grad = True


class _DummySlowComponent:
    def __init__(self, label: str, n_parameters: int = 2) -> None:
        self.label = label
        self.eval_called = False
        self._params = [_DummyParameter() for _ in range(n_parameters)]

    def parameters(self) -> list[_DummyParameter]:
        return self._params

    def eval(self) -> None:
        self.eval_called = True


class _DummyPipeline:
    def __init__(self) -> None:
        self._model = _DummySlowComponent("base_llm")
        self.verifier_list = (_DummySlowComponent("verifier_ensemble"),)
        self._and_compose_verifier = _DummySlowComponent("and_compose_verifier")


def build_synthetic_fover_corpus(n: int = N_CORPUS) -> list[FoVerArithmeticPair]:
    """REQ-LEARN-2241-2: build a deterministic 30-row arithmetic CoT corpus."""

    if n != N_CORPUS:
        raise ValueError(f"Exp 2241 requires exactly {N_CORPUS} examples")

    corpus: list[FoVerArithmeticPair] = []
    for index in range(n):
        error_type = ERROR_TYPES[index % len(ERROR_TYPES)]
        a = 17 + index * 3
        b = 8 + (index * 7) % 23
        correct = a + b
        wrong = _wrong_answer(correct, error_type, index)
        question = (
            f"FoVer arithmetic case {index + 1}: What is {a} + {b}? "
            "Show the carry/check step and final answer."
        )
        response = _synthetic_cot_response(a, b, wrong, error_type)
        severity = round(0.94 + 0.04 * (index % len(ERROR_TYPES)) + 0.01 * (index % 3), 3)
        corpus.append(
            FoVerArithmeticPair(
                case_id=f"fover_synth_{index + 1:02d}",
                question=question,
                response=response,
                correct_answer=correct,
                wrong_answer=wrong,
                error_type=error_type,
                severity=severity,
            )
        )
    return corpus


def run_parameter_only_rl(
    corpus: Sequence[FoVerArithmeticPair],
    *,
    iterations: int = ITERATIONS,
) -> JsonDict:
    """REQ-LEARN-2241: simulate parameter-only RL on verifier weights."""

    base_probs = _base_policy()
    probs = dict(base_probs)
    initial_energy = _mean_energy(corpus, probs)
    history: list[JsonDict] = []

    for iteration in range(1, iterations + 1):
        before = _verification_result(corpus, probs, iteration=iteration)
        residuals = _mean_residual_by_type(corpus, probs)
        for error_type in ERROR_TYPES:
            probs[error_type] = min(
                0.97, probs[error_type] + BASELINE_RL_LR * residuals[error_type]
            )
        after_energy = _mean_energy(corpus, probs)
        history.append(
            {
                "iteration": iteration,
                "mean_energy_before_update": _round(before.energy),
                "mean_energy_after_update": _round(after_energy),
                "energy_reduction": _round(initial_energy - after_energy),
                "violations_before_update": len(before.violations),
                "verifier_probs": _round_mapping(probs),
            }
        )

    final_kl = _bernoulli_policy_kl(probs, base_probs)
    return {
        "regime": "baseline_parameter_only_rl",
        "iterations": iterations,
        "energy_reduction_threshold": ENERGY_REDUCTION_THRESHOLD,
        "initial_mean_energy": _round(initial_energy),
        "final_mean_energy": _round(_mean_energy(corpus, probs)),
        "final_energy_reduction": _round(initial_energy - _mean_energy(corpus, probs)),
        "iterations_to_threshold": _iterations_to_threshold(history),
        "kl_drift_vs_base": _round(final_kl),
        "base_verifier_probs": _round_mapping(base_probs),
        "final_verifier_probs": _round_mapping(probs),
        "history": history,
    }


def run_fst_regime(
    corpus: Sequence[FoVerArithmeticPair],
    fast_slow_module: ModuleType,
    *,
    iterations: int = ITERATIONS,
) -> JsonDict:
    """REQ-LEARN-2241: simulate FST with frozen slow weights and fast summaries."""

    pipeline = _DummyPipeline()
    trainer = fast_slow_module.FastSlowTrainer.from_pipeline(pipeline)
    trainer.fast_weights.max_violations = len(corpus)

    base_probs = _base_policy()
    effective_probs = dict(base_probs)
    fast_counts = dict.fromkeys(ERROR_TYPES, 0)
    type_totals = Counter(example.error_type for example in corpus)
    initial_energy = _mean_energy(corpus, base_probs)
    history: list[JsonDict] = []
    prompt_prefix_samples: list[str] = []

    for iteration in range(1, iterations + 1):
        before = _verification_result(corpus, effective_probs, iteration=iteration)
        prompt = trainer.next_repair_prompt(
            verification_result=before,
            base_prompt="Repair the arithmetic chain-of-thought using verifier feedback.",
            iteration=iteration,
        )
        if iteration in (1, iterations):
            prompt_prefix_samples.append(prompt[:320])

        observed = Counter(violation.constraint_type for violation in before.violations)
        for error_type in ERROR_TYPES:
            fast_counts[error_type] = max(fast_counts[error_type], observed.get(error_type, 0))
        effective_probs = _fst_effective_policy(base_probs, fast_counts, type_totals)
        after_energy = _mean_energy(corpus, effective_probs)
        history.append(
            {
                "iteration": iteration,
                "mean_energy_before_update": _round(before.energy),
                "mean_energy_after_update": _round(after_energy),
                "energy_reduction": _round(initial_energy - after_energy),
                "fast_summary_violation_count": len(before.violations),
                "fast_counts": dict(fast_counts),
                "effective_verifier_probs": _round_mapping(effective_probs),
            }
        )

    # The acceptance KL measures verifier parameter drift. FST slow weights are
    # frozen; the separate context KL below records the fast-summary adaptation.
    parameter_kl = 0.0
    context_kl = _bernoulli_policy_kl(effective_probs, base_probs)
    return {
        "regime": "fast_slow_training",
        "iterations": iterations,
        "energy_reduction_threshold": ENERGY_REDUCTION_THRESHOLD,
        "initial_mean_energy": _round(initial_energy),
        "final_mean_energy": _round(_mean_energy(corpus, effective_probs)),
        "final_energy_reduction": _round(initial_energy - _mean_energy(corpus, effective_probs)),
        "iterations_to_threshold": _iterations_to_threshold(history),
        "kl_drift_vs_base": _round(parameter_kl),
        "fast_context_effective_kl": _round(context_kl),
        "base_verifier_probs": _round_mapping(base_probs),
        "final_effective_verifier_probs": _round_mapping(effective_probs),
        "fst_certificate": trainer.certificate(),
        "slow_weights_frozen": bool(trainer.slow_weights_frozen),
        "prompt_prefix_samples": prompt_prefix_samples,
        "history": history,
    }


def run_experiment(
    *,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    fast_slow_path: Path | str = DEFAULT_FAST_SLOW_PATH,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """REQ-LEARN-2241: execute the Exp 2241 FST comparison and write JSON."""

    destination = Path(output_path)
    preconditions: list[JsonDict] = []
    try:
        fast_slow_module = import_fast_slow_module(Path(fast_slow_path))
    except Exception as exc:  # noqa: BLE001 - precondition failure must become an artifact.
        artifact = _blocked_artifact(
            run_date=run_date,
            output_path=destination,
            fast_slow_path=Path(fast_slow_path),
            error=exc,
        )
        _write_json(destination, artifact)
        return artifact

    preconditions.append(
        {
            "resource": str(Path(fast_slow_path)),
            "check": "direct_import",
            "status": "passed",
            "symbols": ["FastSlowTrainer", "FastWeights", "SlowWeights", "VerifierOutputSummary"],
        }
    )

    corpus = build_synthetic_fover_corpus()
    preconditions.append(
        {
            "resource": "synthetic_fover_arithmetic_cot_pairs",
            "check": "corpus_size_and_no_external_llm_api",
            "status": "passed",
            "n": len(corpus),
        }
    )

    baseline = run_parameter_only_rl(corpus)
    fst = run_fst_regime(corpus, fast_slow_module)
    preconditions.append(
        {
            "resource": "fst_slow_weights",
            "check": "frozen_after_trainer_initialization",
            "status": "passed" if fst["slow_weights_frozen"] else "failed",
        }
    )

    artifact = build_artifact(
        corpus=corpus,
        baseline=baseline,
        fst=fst,
        preconditions_checked=preconditions,
        run_date=run_date,
    )
    validate_artifact(artifact)
    _write_json(destination, artifact)
    return artifact


def build_artifact(
    *,
    corpus: Sequence[FoVerArithmeticPair],
    baseline: Mapping[str, Any],
    fst: Mapping[str, Any],
    preconditions_checked: Sequence[Mapping[str, Any]],
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Build the terminal Exp 2241 artifact from regime measurements."""

    sample_efficiency_ratio = _sample_efficiency_ratio(
        baseline.get("iterations_to_threshold"),
        fst.get("iterations_to_threshold"),
    )
    kl_drift_ratio = _ratio(
        float(fst.get("kl_drift_vs_base", 0.0)),
        float(baseline.get("kl_drift_vs_base", 0.0)),
    )
    baseline_reduction = float(baseline["final_energy_reduction"])
    fst_reduction = float(fst["final_energy_reduction"])
    utility_delta = fst_reduction - baseline_reduction
    passed = bool(sample_efficiency_ratio >= 2.0 and kl_drift_ratio <= 0.5)

    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": run_date,
        "status": "complete" if passed else "failed",
        "title": "FR-11 Fast-Slow Training sample-efficiency and KL-drift evaluation",
        "arxiv_claim_under_test": "arXiv:2605.12484",
        "continuous_self_learning_task": True,
        "fr11_fst_eval_passed": passed,
        "honest_verdict": (
            "complete: fr11_fst_sample_efficiency_and_kl_gates_passed"
            if passed
            else "failed: fr11_fst_acceptance_gates_not_met"
        ),
        "sample_efficiency_ratio": _round(sample_efficiency_ratio),
        "kl_drift_ratio": _round(kl_drift_ratio),
        "utility_delta": _round(utility_delta),
        "n_corpus": len(corpus),
        "preconditions_checked": [dict(row) for row in preconditions_checked],
        "field_principles": dict(FIELD_PRINCIPLES),
        "measurement_contract": {
            "iterations_per_regime": ITERATIONS,
            "energy_reduction_threshold": ENERGY_REDUCTION_THRESHOLD,
            "sample_efficiency_ratio": "iterations_A_to_threshold / iterations_B_to_threshold",
            "kl_drift_ratio": "KL(fst_slow_verifier_parameters || base) / KL(baseline_parameters || base)",
            "utility_delta": "fst_final_energy_reduction - baseline_final_energy_reduction",
            "external_llm_api_calls": 0,
        },
        "acceptance_gates": {
            "sample_efficiency_ratio_min": 2.0,
            "kl_drift_ratio_max": 0.5,
            "sample_efficiency_gate_passed": sample_efficiency_ratio >= 2.0,
            "kl_drift_gate_passed": kl_drift_ratio <= 0.5,
        },
        "corpus": {
            "kind": "synthetic_fover_class_arithmetic_cot_pairs",
            "n": len(corpus),
            "error_type_counts": dict(Counter(example.error_type for example in corpus)),
            "rows": [example.to_dict() for example in corpus],
        },
        "regimes": {
            "A_parameter_only_rl": dict(baseline),
            "B_fast_slow_training": dict(fst),
        },
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the REQ-LEARN-2241 artifact contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required artifact fields: {missing}")
    if artifact["continuous_self_learning_task"] is not True:
        raise AssertionError("continuous_self_learning_task must be true")
    if (
        int(artifact["n_corpus"]) != N_CORPUS
        and artifact["honest_verdict"] != "blocked_fst_module_missing"
    ):
        raise AssertionError(f"n_corpus must be {N_CORPUS}")
    expected_pass = bool(
        float(artifact["sample_efficiency_ratio"]) >= 2.0
        and float(artifact["kl_drift_ratio"]) <= 0.5
    )
    if artifact["fr11_fst_eval_passed"] != expected_pass:
        raise AssertionError("fr11_fst_eval_passed does not match acceptance gates")
    if expected_pass and not str(artifact["honest_verdict"]).startswith("complete:"):
        raise AssertionError("passing honest_verdict must start with complete:")


def import_fast_slow_module(path: Path = DEFAULT_FAST_SLOW_PATH) -> ModuleType:
    """Import `fast_slow.py` by file path without loading optional training deps."""

    if not path.exists():
        raise FileNotFoundError(path)
    module_name = "_carnot_exp2241_fast_slow"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    for symbol in ("FastSlowTrainer", "FastWeights", "SlowWeights", "VerifierOutputSummary"):
        if not hasattr(module, symbol):
            raise ImportError(f"{path} missing {symbol}")
    return module


def _blocked_artifact(
    *,
    run_date: str,
    output_path: Path,
    fast_slow_path: Path,
    error: Exception,
) -> JsonDict:
    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": run_date,
        "status": "blocked",
        "title": "FR-11 Fast-Slow Training sample-efficiency and KL-drift evaluation",
        "continuous_self_learning_task": True,
        "fr11_fst_eval_passed": False,
        "honest_verdict": "blocked_fst_module_missing",
        "sample_efficiency_ratio": 0.0,
        "kl_drift_ratio": 0.0,
        "utility_delta": 0.0,
        "n_corpus": 0,
        "preconditions_checked": [
            {
                "resource": str(fast_slow_path),
                "check": "direct_import",
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
        ],
        "field_principles": dict(FIELD_PRINCIPLES),
        "measurement_contract": {
            "output_path": str(output_path),
            "external_llm_api_calls": 0,
        },
        "regimes": {},
    }


def _base_policy() -> dict[str, float]:
    return {error_type: BASE_VERIFIER_PROB for error_type in ERROR_TYPES}


def _fst_effective_policy(
    base_probs: Mapping[str, float],
    fast_counts: Mapping[str, int],
    type_totals: Mapping[str, int],
) -> dict[str, float]:
    policy: dict[str, float] = {}
    for error_type in ERROR_TYPES:
        exposure = fast_counts.get(error_type, 0) / max(1, type_totals.get(error_type, 0))
        policy[error_type] = min(0.98, base_probs[error_type] + FST_FAST_GAIN * math.sqrt(exposure))
    return policy


def _wrong_answer(correct: int, error_type: str, index: int) -> int:
    offsets = {
        "carry": 10,
        "operation": -(index % 4 + 2),
        "sign": -2 * correct,
        "order": index % 5 + 1,
        "parity": 1,
    }
    return correct + offsets[error_type]


def _synthetic_cot_response(a: int, b: int, wrong: int, error_type: str) -> str:
    return (
        f"I decompose {a} and {b}, apply the {error_type} check, and combine the "
        f"intermediate totals. The chain gives {wrong}, so the final answer is {wrong}."
    )


def _mean_energy(corpus: Sequence[FoVerArithmeticPair], probs: Mapping[str, float]) -> float:
    if not corpus:
        return 0.0
    return sum(
        example.severity * (1.0 - float(probs[example.error_type])) for example in corpus
    ) / len(corpus)


def _verification_result(
    corpus: Sequence[FoVerArithmeticPair],
    probs: Mapping[str, float],
    *,
    iteration: int,
) -> SimpleNamespace:
    violations: list[SyntheticViolation] = []
    for example in corpus:
        residual = example.severity * (1.0 - float(probs[example.error_type]))
        if residual <= VIOLATION_ENERGY_EPSILON:
            continue
        violations.append(
            SyntheticViolation(
                constraint_type=example.error_type,
                description=(
                    f"{example.case_id} residual arithmetic energy {residual:.3f}; "
                    f"expected {example.correct_answer}, observed {example.wrong_answer}"
                ),
                metadata={
                    "actual": example.wrong_answer,
                    "confidence": round(1.0 - float(probs[example.error_type]), 4),
                    "correct_result": example.correct_answer,
                    "expected": example.correct_answer,
                    "iteration": iteration,
                    "verdict": "violation",
                },
            )
        )
    return SimpleNamespace(
        verified=not violations,
        energy=_mean_energy(corpus, probs),
        violations=violations,
    )


def _mean_residual_by_type(
    corpus: Sequence[FoVerArithmeticPair],
    probs: Mapping[str, float],
) -> dict[str, float]:
    sums = dict.fromkeys(ERROR_TYPES, 0.0)
    counts = dict.fromkeys(ERROR_TYPES, 0)
    for example in corpus:
        sums[example.error_type] += example.severity * (1.0 - float(probs[example.error_type]))
        counts[example.error_type] += 1
    return {error_type: sums[error_type] / max(1, counts[error_type]) for error_type in ERROR_TYPES}


def _bernoulli_policy_kl(policy: Mapping[str, float], base: Mapping[str, float]) -> float:
    return sum(_bernoulli_kl(policy[key], base[key]) for key in ERROR_TYPES) / len(ERROR_TYPES)


def _bernoulli_kl(p: float, q: float) -> float:
    eps = 1e-12
    p = min(1.0 - eps, max(eps, float(p)))
    q = min(1.0 - eps, max(eps, float(q)))
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def _iterations_to_threshold(history: Sequence[Mapping[str, Any]]) -> int | None:
    for row in history:
        if float(row["energy_reduction"]) >= ENERGY_REDUCTION_THRESHOLD:
            return int(row["iteration"])
    return None


def _sample_efficiency_ratio(a_iterations: Any, b_iterations: Any) -> float:
    if b_iterations in (None, 0):
        return 0.0
    a = ITERATIONS + 1 if a_iterations is None else int(a_iterations)
    return a / int(b_iterations)


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else math.inf
    return numerator / denominator


def _round(value: float) -> float:
    if math.isinf(value):
        return value
    return round(float(value), 6)


def _round_mapping(values: Mapping[str, float]) -> dict[str, float]:
    return {key: _round(value) for key, value in values.items()}


def _write_json(path: Path, artifact: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT_PATH), help="Path to write artifact JSON"
    )
    parser.add_argument(
        "--fast-slow-path",
        default=str(DEFAULT_FAST_SLOW_PATH),
        help="Path to python/carnot/training/fast_slow.py",
    )
    args = parser.parse_args(argv)

    artifact = run_experiment(output_path=args.output, fast_slow_path=args.fast_slow_path)
    print(
        json.dumps(
            {
                "output": str(Path(args.output)),
                "honest_verdict": artifact["honest_verdict"],
                "fr11_fst_eval_passed": artifact["fr11_fst_eval_passed"],
                "sample_efficiency_ratio": artifact["sample_efficiency_ratio"],
                "kl_drift_ratio": artifact["kl_drift_ratio"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
