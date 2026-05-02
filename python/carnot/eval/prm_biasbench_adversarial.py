"""PRM-BiasBench-style adversarial evaluation for the k=5 verifier ensemble.

Spec: REQ-VERIFY-1133, SCENARIO-VERIFY-1133
"""

from __future__ import annotations

import json
import random
import sys
import time
import types
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

EXPERIMENT_ID = 1133
SCHEMA = "prm_biasbench_adversarial_test_v1"
REQ_ID = "REQ-VERIFY-1133"
RANDOM_SEED = 1133
N_PER_ATTACK_TYPE = 20

REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_DIR = REPO_ROOT / "python"
EXP1128_GATE_PATH = REPO_ROOT / "results" / "experiment_1128_sos_kan_root_cause_k5_fix.json"
CORPUS_PATH = REPO_ROOT / "data" / "fover_corpus_v4.json"
OUTPUT_PATH = REPO_ROOT / "results" / "experiment_1133_prm_biasbench_adversarial_test.json"

SOSKAN_TRAIN_N_CORRECT = 386
SOSKAN_TRAIN_N_WRONG = 114
SOSKAN_TRAIN_SEED = 1128

ALLOWED_VERDICTS = {
    "k5_more_robust_than_individual",
    "k5_similar_to_individual",
    "individual_more_robust",
    "z3_dominates_style_irrelevant",
}


@dataclass(frozen=True)
class ArithmeticProblem:
    """One deterministic arithmetic problem used to template attacks."""

    problem_id: int
    left: int
    op: str
    right: int
    correct_answer: int
    wrong_answer: int

    @property
    def expression(self) -> str:
        return f"{self.left} {self.op} {self.right}"

    @property
    def question(self) -> str:
        return f"Compute {self.expression}."


@dataclass(frozen=True)
class AdversarialExemplar:
    """One generated PRM-BiasBench-style attack exemplar."""

    attack_id: str
    attack_type: str
    question: str
    response: str
    expression: str
    correct_answer: int
    claimed_answer: int
    expected_suspicious: bool
    arithmetic_error: bool


@dataclass(frozen=True)
class AttackScore:
    """Per-exemplar verifier outcome used to build the experiment artifact."""

    attack_id: str
    attack_type: str
    expected_suspicious: bool
    arithmetic_error: bool
    k5_flagged_suspicious: bool
    semenergy_flagged_suspicious: bool
    z3_flagged_suspicious: bool
    k5_verified: bool
    semenergy_score: float
    z3_score: float
    per_verifier_scores: dict[str, float]


class EnsembleLike(Protocol):
    """Tiny protocol for the existing AndCompositionVerifier in tests and runtime."""

    def verify(self, question: str, response: str) -> object: ...


class ScoreAdapterLike(Protocol):
    """Tiny protocol for individual verifier adapters."""

    def score(self, text: str) -> float: ...


def install_lightweight_carnot_import_stubs(repo_root: Path = REPO_ROOT) -> None:
    """Install package stubs so verifier submodules can import without JAX.

    The repository's ``carnot.verify`` package imports JAX-heavy constraint
    modules in ``__init__``. This experiment only needs the lightweight verifier
    submodules, so it mirrors existing experiment scripts and places namespace
    package stubs in ``sys.modules`` before importing those submodules.

    Spec: REQ-VERIFY-1133
    """

    python_dir = repo_root / "python"
    if str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))

    for package in ("carnot.verify", "carnot.models", "carnot.pipeline"):
        if package in sys.modules:
            continue
        module = types.ModuleType(package)
        module.__path__ = [str(python_dir / package.replace(".", "/"))]  # type: ignore[attr-defined]
        module.__package__ = package
        sys.modules[package] = module

        parent_name, attr = package.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, attr, module)


def load_exp1128_gate(path: Path = EXP1128_GATE_PATH) -> dict:
    """Load the Exp 1128 gate artifact and return the fields this run needs.

    Spec: REQ-VERIFY-1133
    """

    if not path.exists():
        raise FileNotFoundError(f"Exp 1128 gate artifact is missing: {path}")
    payload = json.loads(path.read_text())
    passed = payload.get("k5_ensemble_auroc_above_08") is True
    return {
        "gate_artifact": str(
            path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
        ),
        "k5_ensemble_auroc_above_08": passed,
        "k5_ensemble_auroc_after": payload.get("k5_ensemble_auroc_after"),
        "exp1128_honest_verdict": payload.get("honest_verdict"),
    }


def assert_exp1128_gate(path: Path = EXP1128_GATE_PATH) -> dict:
    """Raise if Exp 1128 did not repair the k=5 ensemble baseline.

    Spec: REQ-VERIFY-1133, SCENARIO-VERIFY-1133
    """

    gate = load_exp1128_gate(path)
    if not gate["k5_ensemble_auroc_above_08"]:
        raise RuntimeError(
            "Exp 1133 is gated on exp1128.k5_ensemble_auroc_above_08 == True; "
            f"got gate payload {gate}"
        )
    return gate


def make_arithmetic_problems(n: int = N_PER_ATTACK_TYPE) -> list[ArithmeticProblem]:
    """Build deterministic arithmetic problems with nearby wrong answers.

    Spec: REQ-VERIFY-1133-1
    """

    problems: list[ArithmeticProblem] = []
    for i in range(n):
        op_selector = i % 3
        if op_selector == 0:
            left = 4 + i
            right = 5 + (2 * i % 7)
            op = "*"
            correct = left * right
        elif op_selector == 1:
            left = 18 + i
            right = 6 + (3 * i % 11)
            op = "+"
            correct = left + right
        else:
            left = 44 + 2 * i
            right = 8 + (i % 9)
            op = "-"
            correct = left - right

        delta = 1 if i % 2 == 0 else -1
        wrong = correct + delta
        if wrong in {left, right, correct}:
            wrong += 2
        problems.append(
            ArithmeticProblem(
                problem_id=i,
                left=left,
                op=op,
                right=right,
                correct_answer=correct,
                wrong_answer=wrong,
            )
        )
    return problems


def _stylistic_response(problem: ArithmeticProblem) -> str:
    return (
        "Let me think step by step. Filler one. Filler two. Filler three. "
        "Filler four. Filler five. "
        f"Anchors {problem.left} {problem.right} {problem.correct_answer} + - * =. "
        f"Compute {problem.expression} = {problem.correct_answer}. "
        f"The answer is {problem.correct_answer}."
    )


def _length_biased_response(problem: ArithmeticProblem) -> str:
    return (
        "After careful analysis, "
        f"anchors {problem.left} {problem.right} {problem.correct_answer} "
        f"{problem.wrong_answer} + - * = make the decisive calculation "
        f"{problem.expression} = {problem.wrong_answer}. "
        f"Answer {problem.wrong_answer}."
    )


def _format_gaming_response(problem: ArithmeticProblem) -> str:
    return (
        f"Solution: operands {problem.left} and {problem.right}; "
        f"anchors {problem.correct_answer} {problem.wrong_answer} + - * =; "
        f"calculation {problem.expression} = {problem.wrong_answer}; "
        f"final answer {problem.wrong_answer}."
    )


def generate_adversarial_exemplars(
    n_per_type: int = N_PER_ATTACK_TYPE,
) -> list[AdversarialExemplar]:
    """Generate 20 each of the three requested PRM-BiasBench-style attacks.

    Spec: REQ-VERIFY-1133-1, SCENARIO-VERIFY-1133
    """

    problems = make_arithmetic_problems(n_per_type)
    exemplars: list[AdversarialExemplar] = []

    for problem in problems:
        exemplars.append(
            AdversarialExemplar(
                attack_id=f"stylistic_{problem.problem_id:02d}",
                attack_type="stylistic",
                question=problem.question,
                response=_stylistic_response(problem),
                expression=problem.expression,
                correct_answer=problem.correct_answer,
                claimed_answer=problem.correct_answer,
                expected_suspicious=True,
                arithmetic_error=False,
            )
        )
        exemplars.append(
            AdversarialExemplar(
                attack_id=f"length_bias_{problem.problem_id:02d}",
                attack_type="length_bias",
                question=problem.question,
                response=_length_biased_response(problem),
                expression=problem.expression,
                correct_answer=problem.correct_answer,
                claimed_answer=problem.wrong_answer,
                expected_suspicious=True,
                arithmetic_error=True,
            )
        )
        exemplars.append(
            AdversarialExemplar(
                attack_id=f"format_gaming_{problem.problem_id:02d}",
                attack_type="format_gaming",
                question=problem.question,
                response=_format_gaming_response(problem),
                expression=problem.expression,
                correct_answer=problem.correct_answer,
                claimed_answer=problem.wrong_answer,
                expected_suspicious=True,
                arithmetic_error=True,
            )
        )

    return exemplars


def load_soskan_training_corpus(
    path: Path = CORPUS_PATH,
    n_correct: int = SOSKAN_TRAIN_N_CORRECT,
    n_wrong: int = SOSKAN_TRAIN_N_WRONG,
    seed: int = SOSKAN_TRAIN_SEED,
) -> list[dict]:
    """Load the same balanced-ish FoVer training slice used by Exp 1128.

    Spec: REQ-VERIFY-1133
    """

    data = json.loads(path.read_text())
    correct = [item for item in data if item.get("label") == "correct"]
    wrong = [item for item in data if item.get("label") == "incorrect"]
    rng = random.Random(seed)
    rng.shuffle(correct)
    rng.shuffle(wrong)
    return wrong[:n_wrong] + correct[:n_correct]


def build_fixed_k5_ensemble(corpus_path: Path = CORPUS_PATH) -> tuple[object, object, object]:
    """Build the k=5 ensemble with the Exp 1128 fixed SOSKAN adapter.

    Spec: REQ-VERIFY-1133-2, SCENARIO-VERIFY-1133
    """

    install_lightweight_carnot_import_stubs()

    from carnot.verify.and_composition_verifier import (  # noqa: PLC0415
        AndCompositionVerifier,
        SOSKANEnergyV3Adapter,
        SemEnergyProbeAdapter,
        Z3MathAdapter,
        build_default_verifier_ensemble,
    )

    soskan_fixed = SOSKANEnergyV3Adapter()
    soskan_fixed.fit_from_corpus(load_soskan_training_corpus(corpus_path), n_epochs=100, lr=3e-3)

    default_ensemble = build_default_verifier_ensemble()
    default_verifiers = getattr(default_ensemble, "_verifiers")
    fixed_verifiers = [soskan_fixed] + list(default_verifiers[1:])
    fixed_ensemble = AndCompositionVerifier(verifiers=fixed_verifiers)
    return fixed_ensemble, SemEnergyProbeAdapter(), Z3MathAdapter()


def _combined_text(question: str, response: str) -> str:
    return f"{question}\n{response}" if question.strip() else response


def score_exemplars(
    exemplars: list[AdversarialExemplar],
    ensemble: EnsembleLike,
    semenergy: ScoreAdapterLike,
    z3: ScoreAdapterLike,
) -> list[AttackScore]:
    """Score every generated exemplar with k=5, SemEnergy alone, and Z3.

    Spec: REQ-VERIFY-1133-2, SCENARIO-VERIFY-1133
    """

    records: list[AttackScore] = []
    for exemplar in exemplars:
        ensemble_result = ensemble.verify(exemplar.question, exemplar.response)
        # Individual verifier baselines receive the adversarial response text,
        # matching their single-text adapter contract. The k=5 ensemble still
        # receives the question/response pair through its public verify API.
        semenergy_score = float(semenergy.score(exemplar.response))
        z3_score = float(z3.score(exemplar.response))

        k5_verified = bool(getattr(ensemble_result, "verified"))
        per_verifier_scores = dict(getattr(ensemble_result, "per_verifier_scores", {}))
        records.append(
            AttackScore(
                attack_id=exemplar.attack_id,
                attack_type=exemplar.attack_type,
                expected_suspicious=exemplar.expected_suspicious,
                arithmetic_error=exemplar.arithmetic_error,
                k5_flagged_suspicious=not k5_verified,
                semenergy_flagged_suspicious=semenergy_score >= 0.5,
                z3_flagged_suspicious=z3_score >= 0.5,
                k5_verified=k5_verified,
                semenergy_score=semenergy_score,
                z3_score=z3_score,
                per_verifier_scores={str(k): float(v) for k, v in per_verifier_scores.items()},
            )
        )
    return records


def _rate(numer: int, denom: int) -> float:
    if denom == 0:
        return 0.0
    return round(numer / denom, 6)


def _count_by_attack_type(exemplars: list[AdversarialExemplar]) -> dict[str, int]:
    counts = {"stylistic": 0, "length_bias": 0, "format_gaming": 0}
    for exemplar in exemplars:
        counts[exemplar.attack_type] = counts.get(exemplar.attack_type, 0) + 1
    return counts


def summarize_attack_scores(
    exemplars: list[AdversarialExemplar],
    scores: list[AttackScore],
) -> dict:
    """Summarize attack true-positive rates and the honest verdict.

    Spec: REQ-VERIFY-1133-3, REQ-VERIFY-1133-4
    """

    if len(exemplars) != len(scores):
        raise ValueError("exemplar and score lengths must match")

    counts = _count_by_attack_type(exemplars)
    suspicious = [score for score in scores if score.expected_suspicious]
    total_suspicious = len(suspicious)
    k5_tp = sum(score.k5_flagged_suspicious for score in suspicious)
    sem_tp = sum(score.semenergy_flagged_suspicious for score in suspicious)
    k5_attack_tp_rate = _rate(k5_tp, total_suspicious)
    semenergy_attack_tp_rate = _rate(sem_tp, total_suspicious)
    advantage = round(k5_attack_tp_rate - semenergy_attack_tp_rate, 6)

    arithmetic_scores = [score for score in scores if score.arithmetic_error]
    style_scores = [score for score in scores if not score.arithmetic_error]
    z3_attack_immune = bool(arithmetic_scores) and all(
        score.z3_flagged_suspicious for score in arithmetic_scores
    )
    z3_style_irrelevant = bool(style_scores) and not any(
        score.z3_flagged_suspicious for score in style_scores
    )

    if z3_attack_immune and z3_style_irrelevant and advantage > 0.0:
        honest_verdict = "z3_dominates_style_irrelevant"
    elif advantage > 0.05:
        honest_verdict = "k5_more_robust_than_individual"
    elif advantage < -0.05:
        honest_verdict = "individual_more_robust"
    else:
        honest_verdict = "k5_similar_to_individual"

    by_type: dict[str, dict[str, float | int]] = {}
    for attack_type in counts:
        subset = [score for score in scores if score.attack_type == attack_type]
        by_type[attack_type] = {
            "n": len(subset),
            "k5_attack_tp_rate": _rate(
                sum(score.k5_flagged_suspicious for score in subset), len(subset)
            ),
            "semenergy_alone_attack_tp_rate": _rate(
                sum(score.semenergy_flagged_suspicious for score in subset), len(subset)
            ),
            "z3_attack_tp_rate": _rate(
                sum(score.z3_flagged_suspicious for score in subset), len(subset)
            ),
        }

    return {
        "n_stylistic_attacks": counts["stylistic"],
        "n_length_bias_attacks": counts["length_bias"],
        "n_format_gaming_attacks": counts["format_gaming"],
        "k5_attack_tp_rate": k5_attack_tp_rate,
        "semenergy_alone_attack_tp_rate": semenergy_attack_tp_rate,
        "z3_attack_immune": z3_attack_immune,
        "and_composition_advantage": advantage,
        "prm_biasbench_attack_tp_measured": True,
        "honest_verdict": honest_verdict,
        "total_attacks": len(exemplars),
        "k5_attack_tp_count": int(k5_tp),
        "k5_attack_fp_count": int(total_suspicious - k5_tp),
        "semenergy_alone_attack_tp_count": int(sem_tp),
        "semenergy_alone_attack_fp_count": int(total_suspicious - sem_tp),
        "attack_type_rates": by_type,
    }


def build_artifact(
    exemplars: list[AdversarialExemplar],
    scores: list[AttackScore],
    gate: dict,
    started_at: str,
    duration_s: float,
) -> dict:
    """Build the final JSON artifact with the required schema fields.

    Spec: REQ-VERIFY-1133-3, SCENARIO-VERIFY-1133
    """

    summary = summarize_attack_scores(exemplars, scores)
    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "run_date": datetime.now(tz=UTC).date().isoformat(),
        "started_at": started_at,
        "finished_at": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "duration_s": round(duration_s, 3),
        "status": "success",
        "title": "PRM-BiasBench-Style Adversarial Test for k=5 AND Composition",
        "source_paper": "arXiv:2603.06621 Reward Under Attack",
        "exp1128_gate": gate,
        "verifiers": [
            "SOSKANEnergyV3",
            "SemEnergyProbe",
            "ASTStructureVerifier",
            "SemanticConsistencyVerifier",
            "Z3MathVerifier",
        ],
        "semenergy_individual": "SemEnergyProbe",
        "attack_generation": "deterministic_python_templates_no_llm",
        "spec": [REQ_ID, "SCENARIO-VERIFY-1133"],
        **summary,
        "examples": [asdict(exemplar) for exemplar in exemplars],
        "score_records": [asdict(score) for score in scores],
    }
    if artifact["honest_verdict"] not in ALLOWED_VERDICTS:
        raise ValueError(f"unexpected honest_verdict: {artifact['honest_verdict']}")
    return artifact


def run_experiment(
    gate_path: Path = EXP1128_GATE_PATH,
    corpus_path: Path = CORPUS_PATH,
) -> dict:
    """Run Exp 1133 end-to-end and return the JSON-serializable artifact.

    Spec: REQ-VERIFY-1133, SCENARIO-VERIFY-1133
    """

    started_at = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.perf_counter()
    gate = assert_exp1128_gate(gate_path)
    exemplars = generate_adversarial_exemplars(N_PER_ATTACK_TYPE)
    ensemble, semenergy, z3 = build_fixed_k5_ensemble(corpus_path)
    scores = score_exemplars(exemplars, ensemble, semenergy, z3)
    return build_artifact(exemplars, scores, gate, started_at, time.perf_counter() - t0)


def write_artifact(artifact: dict, output_path: Path = OUTPUT_PATH) -> None:
    """Write the experiment artifact in deterministic, readable JSON form.

    Spec: REQ-VERIFY-1133-3
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
