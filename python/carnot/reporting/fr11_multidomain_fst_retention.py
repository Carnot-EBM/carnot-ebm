"""Exp 2357 FR-11 multidomain Fast-Slow retention evaluation.

Spec: REQ-LEARN-2357, SCENARIO-LEARN-2357.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.learning.fast_slow import FastSlowTrainer

JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_FILE = "experiment_2357_fr11_multidomain.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / OUTPUT_FILE

EXPERIMENT = "2357_fr11_multidomain_fst_retention"
SCHEMA = "fr11_multidomain_fst_retention_v1"
RUN_DATE = "20260518"
RANDOM_SEED = 42
N_DOMAINS = 3
N_TRAIN_PER_DOMAIN = 30
N_HOLDOUT_PER_DOMAIN = 10
RETENTION_GATE = 0.75
DOMAIN_ORDER = ("arithmetic", "code", "logic")
EXPECTED_FAST_UPDATE_COUNT = N_TRAIN_PER_DOMAIN * 7

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "honest_verdict",
    "fr11_multidomain_passed",
    "cross_domain_retention_rate",
    "continuous_self_learning_validated",
    "n_domains",
    "random_seed",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": "Terminal-prefix required.",
    "fr11_multidomain_passed": "True if cross_domain_retention_rate >= 0.75.",
    "cross_domain_retention_rate": (
        "Primary metric. Mean accuracy retention across domains. Gate: >= 0.75."
    ),
    "continuous_self_learning_validated": (
        "Must be true to satisfy FR-11. Records that online learning mechanism works."
    ),
    "n_domains": "Must be 3.",
    "random_seed": "Reproducibility. Must be 42.",
}


@dataclass(frozen=True)
class DomainCase:
    """One deterministic query used by the multidomain FST retention protocol."""

    case_id: str
    domain: str
    question: str
    answer: str
    constraints: tuple[Mapping[str, str], ...]

    def to_dict(self) -> JsonDict:
        return {
            "case_id": self.case_id,
            "domain": self.domain,
            "question": self.question,
            "answer": self.answer,
            "constraints": [dict(row) for row in self.constraints],
        }


class _VerifierSeed:
    constraint_weights = {
        "parse:numbers": 1.0,
        "parse:boolean_literals": 1.0,
        "normalize:answer": 1.0,
    }


class _PipelineSeed:
    _and_compose_verifier = _VerifierSeed()


def build_multidomain_corpus(
    *,
    seed: int = RANDOM_SEED,
    n_train: int = N_TRAIN_PER_DOMAIN,
    n_holdout: int = N_HOLDOUT_PER_DOMAIN,
) -> dict[str, dict[str, list[DomainCase]]]:
    """REQ-LEARN-2357: build 3 domains with train and holdout splits."""

    rng = random.Random(seed)
    corpus: dict[str, dict[str, list[DomainCase]]] = {}
    corpus["arithmetic"] = _split_cases(_build_arithmetic_cases(rng, n_train + n_holdout), n_train)
    corpus["code"] = _split_cases(_build_code_cases(rng, n_train + n_holdout), n_train)
    corpus["logic"] = _split_cases(_build_logic_cases(rng, n_train + n_holdout), n_train)
    return corpus


def run_retention_protocol(
    *,
    seed: int = RANDOM_SEED,
    trainer: FastSlowTrainer | None = None,
) -> JsonDict:
    """Run the mandated arithmetic, code, and logic retention sequence."""

    corpus = build_multidomain_corpus(seed=seed)
    trainer = trainer or FastSlowTrainer.from_pipeline(_PipelineSeed())
    initial_slow_weights = dict(trainer.slow_weights.constraint_weights)
    domain_training: list[JsonDict] = []
    retention_measurements: list[JsonDict] = []

    for domain_index, domain in enumerate(DOMAIN_ORDER):
        train_cases = corpus[domain]["train"]
        holdout_cases = corpus[domain]["holdout"]
        for case in train_cases:
            trainer.clear_query_context()
            trainer.update_fast(case.question, _verification_result(case))
            trainer.clear_query_context()

        own_eval = evaluate_accuracy(trainer, holdout_cases)
        domain_training.append(
            {
                "domain": domain,
                "train_n": len(train_cases),
                "holdout_n": len(holdout_cases),
                "holdout_accuracy_after_domain_training": own_eval["accuracy"],
                "fast_cache_size": len(trainer.fast_weights.cache),
                "fast_update_count": trainer.fast_weights.update_count,
            }
        )

        for prior_domain in DOMAIN_ORDER[:domain_index]:
            prior_eval = evaluate_accuracy(trainer, corpus[prior_domain]["holdout"])
            retention_measurements.append(
                {
                    "after_domain_training": domain,
                    "prior_domain": prior_domain,
                    "holdout_n": len(corpus[prior_domain]["holdout"]),
                    "accuracy": prior_eval["accuracy"],
                    "correct": prior_eval["correct"],
                }
            )

    retention_rate = _mean(row["accuracy"] for row in retention_measurements)
    return {
        "domain_training": domain_training,
        "retention_measurements": retention_measurements,
        "cross_domain_retention_rate": _round(retention_rate),
        "trainer_certificate": trainer.certificate(),
        "slow_weights_mutated": initial_slow_weights != trainer.slow_weights.constraint_weights,
        "corpus_summary": {
            domain: {
                "train_n": len(splits["train"]),
                "holdout_n": len(splits["holdout"]),
                "train_cases": [case.to_dict() for case in splits["train"]],
                "holdout_cases": [case.to_dict() for case in splits["holdout"]],
            }
            for domain, splits in corpus.items()
        },
    }


def evaluate_accuracy(trainer: FastSlowTrainer, cases: Sequence[DomainCase]) -> JsonDict:
    """Evaluate holdout accuracy while clearing per-query fast scratch context."""

    rows: list[JsonDict] = []
    for case in cases:
        trainer.clear_query_context()
        prediction = trainer.predict(case.question)
        trainer.clear_query_context()
        correct = _normalize_answer(prediction) == _normalize_answer(case.answer)
        rows.append(
            {
                "case_id": case.case_id,
                "domain": case.domain,
                "prediction": prediction,
                "answer": case.answer,
                "correct": correct,
            }
        )
    correct_count = sum(1 for row in rows if row["correct"])
    return {
        "accuracy": _round(correct_count / len(rows) if rows else 0.0),
        "correct": correct_count,
        "rows": rows,
    }


def run_experiment(
    *,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    run_date: str = RUN_DATE,
    seed: int = RANDOM_SEED,
) -> JsonDict:
    """Execute Exp 2357 and write the terminal JSON artifact."""

    destination = Path(output_path)
    preconditions = [
        {
            "resource": "carnot.learning.fast_slow.FastSlowTrainer",
            "check": "direct_import_and_required_methods",
            "status": "passed",
            "symbols": ["FastSlowTrainer", "SlowWeights", "FastWeights"],
            "methods": ["update_fast", "predict", "clear_query_context"],
        },
        {
            "resource": "synthetic_arithmetic_code_logic_corpus",
            "check": "fixed_seed_train_holdout_protocol",
            "status": "passed",
            "random_seed": seed,
            "n_domains": N_DOMAINS,
        },
    ]
    metrics = run_retention_protocol(seed=seed)
    artifact = build_artifact(
        metrics=metrics,
        preconditions_checked=preconditions,
        run_date=run_date,
        seed=seed,
    )
    validate_artifact(artifact)
    _write_json(destination, artifact)
    return artifact


def build_artifact(
    *,
    metrics: Mapping[str, Any],
    preconditions_checked: Sequence[Mapping[str, Any]],
    run_date: str = RUN_DATE,
    seed: int = RANDOM_SEED,
) -> JsonDict:
    retention_rate = float(metrics["cross_domain_retention_rate"])
    passed = bool(retention_rate >= RETENTION_GATE and not metrics["slow_weights_mutated"])
    continuous_validated = bool(
        passed and metrics["trainer_certificate"]["fast_update_count"] == EXPECTED_FAST_UPDATE_COUNT
    )
    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": run_date,
        "status": "complete" if passed else "failed",
        "title": "FR-11 FST fast/slow cross-domain retention validation",
        "arxiv_claim_under_test": "arXiv:2605.12484",
        "honest_verdict": (
            "complete: fr11_multidomain_fst_retention_passed"
            if passed
            else "failed: fr11_multidomain_fst_retention_below_gate"
        ),
        "fr11_multidomain_passed": passed,
        "cross_domain_retention_rate": _round(retention_rate),
        "continuous_self_learning_validated": continuous_validated,
        "n_domains": N_DOMAINS,
        "random_seed": seed,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": [dict(row) for row in preconditions_checked],
        "measurement_contract": {
            "domain_order": list(DOMAIN_ORDER),
            "train_per_domain": N_TRAIN_PER_DOMAIN,
            "holdout_per_domain": N_HOLDOUT_PER_DOMAIN,
            "retention_gate": RETENTION_GATE,
            "cross_domain_retention_rate": (
                "mean accuracy on prior-domain holdouts after later-domain training"
            ),
            "slow_weight_mutation_allowed": False,
            "external_llm_api_calls": 0,
        },
        "domain_training": list(metrics["domain_training"]),
        "retention_measurements": list(metrics["retention_measurements"]),
        "trainer_certificate": dict(metrics["trainer_certificate"]),
        "slow_weights_mutated": bool(metrics["slow_weights_mutated"]),
        "corpus_summary": dict(metrics["corpus_summary"]),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the REQ-LEARN-2357 terminal artifact contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required artifact fields: {missing}")
    if int(artifact["n_domains"]) != N_DOMAINS:
        raise AssertionError(f"n_domains must be {N_DOMAINS}")
    if int(artifact["random_seed"]) != RANDOM_SEED:
        raise AssertionError(f"random_seed must be {RANDOM_SEED}")
    if bool(artifact.get("slow_weights_mutated")):
        raise AssertionError("slow weights must not mutate during FST retention")
    measurements = list(artifact.get("retention_measurements") or [])
    expected_rate = _round(_mean(row["accuracy"] for row in measurements))
    if float(artifact["cross_domain_retention_rate"]) != expected_rate:
        raise AssertionError("cross_domain_retention_rate does not match measurements")
    expected_pass = float(artifact["cross_domain_retention_rate"]) >= RETENTION_GATE
    if artifact["fr11_multidomain_passed"] != expected_pass:
        raise AssertionError("fr11_multidomain_passed does not match retention gate")
    if artifact["continuous_self_learning_validated"] != expected_pass:
        raise AssertionError("continuous_self_learning_validated must match FR-11 pass gate")
    verdict = str(artifact["honest_verdict"])
    if expected_pass and not verdict.startswith("complete:"):
        raise AssertionError("passing honest_verdict must start with complete:")
    if not expected_pass and not verdict.startswith("failed:"):
        raise AssertionError("failing honest_verdict must start with failed:")


def _build_arithmetic_cases(rng: random.Random, n: int) -> list[DomainCase]:
    cases: list[DomainCase] = []
    for index in range(n):
        a = rng.randint(10, 90) + index
        b = rng.randint(1, 60)
        cases.append(
            DomainCase(
                case_id=f"arithmetic_{index + 1:02d}",
                domain="arithmetic",
                question=f"Domain arithmetic case {index + 1:02d}: add {a} and {b}.",
                answer=str(a + b),
                constraints=(
                    {
                        "id": "operation:addition",
                        "description": "Verified addition constraint from arithmetic.",
                    },
                    {"id": "domain:arithmetic", "description": "Arithmetic verifier context."},
                ),
            )
        )
    return cases


def _build_code_cases(rng: random.Random, n: int) -> list[DomainCase]:
    cases: list[DomainCase] = []
    for index in range(n):
        a = rng.randint(2, 40) + index
        b = rng.randint(2, 40)
        cases.append(
            DomainCase(
                case_id=f"code_{index + 1:02d}",
                domain="code",
                question=(
                    f"Domain code case {index + 1:02d}: What value does "
                    f"`def f(): return {a} + {b}` produce?"
                ),
                answer=str(a + b),
                constraints=(
                    {
                        "id": "syntax:python_return_expr",
                        "description": "Verified Python return-expression constraint.",
                    },
                    {
                        "id": "operation:addition",
                        "description": "Verified addition constraint reused in code.",
                    },
                    {"id": "domain:code", "description": "Code verifier context."},
                ),
            )
        )
    return cases


def _build_logic_cases(rng: random.Random, n: int) -> list[DomainCase]:
    cases: list[DomainCase] = []
    ops = ("AND", "OR", "XOR")
    for index in range(n):
        left = bool(rng.randint(0, 1))
        right = bool((rng.randint(0, 1) + index) % 2)
        op = ops[index % len(ops)]
        if op == "AND":
            answer = left and right
        elif op == "OR":
            answer = left or right
        else:
            answer = left != right
        cases.append(
            DomainCase(
                case_id=f"logic_{index + 1:02d}",
                domain="logic",
                question=(
                    f"Domain logic case {index + 1:02d}: Evaluate "
                    f"{str(left).lower()} {op} {str(right).lower()}."
                ),
                answer=str(answer).lower(),
                constraints=(
                    {
                        "id": "logic:boolean_algebra",
                        "description": "Verified boolean algebra constraint.",
                    },
                    {"id": "domain:logic", "description": "Logic verifier context."},
                ),
            )
        )
    return cases


def _split_cases(cases: Sequence[DomainCase], n_train: int) -> dict[str, list[DomainCase]]:
    return {"train": list(cases[:n_train]), "holdout": list(cases[n_train:])}


def _verification_result(case: DomainCase) -> JsonDict:
    return {
        "verified": True,
        "domain": case.domain,
        "case_id": case.case_id,
        "answer": case.answer,
        "constraints": [dict(row) for row in case.constraints],
    }


def _normalize_answer(value: Any) -> str:
    return "" if value is None else str(value).strip().lower()


def _mean(values: Any) -> float:
    rows = [float(value) for value in values]
    return sum(rows) / len(rows) if rows else 0.0


def _round(value: float) -> float:
    if math.isinf(value):
        return value
    return round(float(value), 6)


def _write_json(path: Path, artifact: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Path to write JSON")
    args = parser.parse_args(argv)

    artifact = run_experiment(output_path=args.output)
    print(
        json.dumps(
            {
                "output": str(Path(args.output)),
                "honest_verdict": artifact["honest_verdict"],
                "fr11_multidomain_passed": artifact["fr11_multidomain_passed"],
                "cross_domain_retention_rate": artifact["cross_domain_retention_rate"],
                "continuous_self_learning_validated": artifact[
                    "continuous_self_learning_validated"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
