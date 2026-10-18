"""Exp 3646 trained EBM judge OOD counterpoint.

Spec: REQ-VERIFY-3646, SCENARIO-VERIFY-3646.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any

import numpy as np

from carnot.verify.retrieval_nli_grounding_verifier import RetrievalNLIGroundingVerifier


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_3646_trained_ebm_judge_ood_counterpoint_v2.json")
FOVER_CORPUS_REL_PATH = Path("data/fover_corpus.jsonl")
EXP3640_REL_PATH = Path("results/experiment_3640_build_factual_corpus_v3.json")
EXP3641_REL_PATH = Path("results/experiment_3641_code_corpus_verifiers_fire_transfer_v3.json")
DEFAULT_SEEDS = (3646, 3647, 3648)
RANDOM_SEED = DEFAULT_SEEDS[0]
FIXED_ENSEMBLE_OOD_REFERENCE_AUROC = 0.331
FEATURE_NAMES = (
    "validity_signal",
    "confidence_error_signal",
    "log1p_char_count",
    "digit_fraction",
    "format_noise_signal",
)
N_FEATURES = len(FEATURE_NAMES)
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates "
    "(principle: trains a small head on cached labels + scores cached corpora; GPU use: none, CPU-only numpy)."
)
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "judge_recipe",
    "in_domain_judge_auroc",
    "ood_judge_auroc",
    "ood_domains_tested",
    "shuffled_label_control_auroc",
    "trained_judge_vs_fixed_ensemble_ood_delta",
    "trained_judge_transfers_ood",
    "n_examples_per_domain",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)
FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": (
        "The train/eval substrate and GPU disclosure; this prototype trains only a small CPU head over cached scores."
    ),
    "judge_recipe": "The trained judge's features/loss/epochs -- reproducibility of the prototype.",
    "in_domain_judge_auroc": (
        "Sanity: the trained judge must work in-domain before any OOD claim is meaningful."
    ),
    "ood_judge_auroc": (
        "The core number: does the trained judge detect errors in a domain it was NOT trained on?"
    ),
    "ood_domains_tested": (
        "Which OOD domains (code/facts) were evaluated -- scopes the transfer claim."
    ),
    "shuffled_label_control_auroc": (
        "Adversarial control: a near-0.5 here confirms the judge is not learning a degenerate prior."
    ),
    "trained_judge_vs_fixed_ensemble_ood_delta": (
        "Does TRAINING the judge buy the OOD transfer the fixed ensemble (.329-.333) lacked -- the foundation-model-path signal."
    ),
    "trained_judge_transfers_ood": (
        "True iff the trained judge beats both confidence-only and shuffled-control on OOD -- the falsifiable transfer claim."
    ),
    "n_examples_per_domain": "Sample-size rigor.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor (a real train+eval takes minutes, not seconds).",
}


@dataclass(frozen=True)
class JudgeExample:
    """One labeled candidate row for the small energy judge."""

    domain: str
    text: str
    error_label: int
    validity_signal: float
    confidence_error_signal: float


class SmallEnergyJudge:
    """Tiny logistic energy head trained to assign high scores to errors."""

    def __init__(self, *, epochs: int = 200, lr: float = 0.35, l2: float = 1e-3) -> None:
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.l2 = float(l2)
        self.weights = np.zeros(N_FEATURES, dtype=np.float64)
        self.bias = 0.0
        self._mu = np.zeros(N_FEATURES, dtype=np.float64)
        self._sigma = np.ones(N_FEATURES, dtype=np.float64)
        self._fitted = False

    @property
    def n_params(self) -> int:
        return N_FEATURES + 1

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int], *, seed: int) -> "SmallEnergyJudge":
        Xa = np.asarray(X, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        if Xa.ndim != 2 or Xa.shape[1] != N_FEATURES:
            raise ValueError(f"expected (_, {N_FEATURES}) feature matrix, got {Xa.shape}")
        rng = np.random.default_rng(int(seed))
        self.weights = rng.normal(0.0, 0.01, size=N_FEATURES)
        self.bias = 0.0
        self._mu = Xa.mean(axis=0)
        sigma = Xa.std(axis=0)
        sigma[sigma < 1e-8] = 1.0
        self._sigma = sigma
        Z = self._standardise(Xa)
        n = max(1, Z.shape[0])
        for _ in range(self.epochs):
            logits = np.clip(Z @ self.weights + self.bias, -50.0, 50.0)
            probs = 1.0 / (1.0 + np.exp(-logits))
            err = probs - ya
            grad_w = Z.T @ err / n + self.l2 * self.weights
            grad_b = float(err.mean())
            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b
        self._fitted = True
        return self

    def predict_scores(self, X: Sequence[Sequence[float]]) -> list[float]:
        if not self._fitted:
            raise RuntimeError("small energy judge is not fitted")
        Xa = np.asarray(X, dtype=np.float64)
        Z = self._standardise(Xa)
        logits = np.clip(Z @ self.weights + self.bias, -50.0, 50.0)
        return [float(1.0 / (1.0 + math.exp(-float(logit)))) for logit in logits]

    def _standardise(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mu) / self._sigma


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    math_examples: Sequence[JudgeExample] | None = None,
    ood_examples_by_domain: Mapping[str, Sequence[JudgeExample]] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    epochs: int = 200,
    lr: float = 0.35,
    l2: float = 1e-3,
    max_math_examples: int = 2000,
    force_no_trainable_substrate: bool = False,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the terminal Exp 3646 artifact."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    if math_examples is None or ood_examples_by_domain is None:
        loaded_math, loaded_ood = load_corpora(root_path, max_math_examples=max_math_examples)
        if math_examples is None:
            math_examples = loaded_math
        if ood_examples_by_domain is None:
            ood_examples_by_domain = loaded_ood

    math_rows = list(math_examples)
    ood_rows_by_domain = {
        str(domain): list(rows)
        for domain, rows in dict(ood_examples_by_domain or {}).items()
        if rows
    }
    ood_domains = sorted(ood_rows_by_domain)
    no_substrate = bool(force_no_trainable_substrate or not trainable_math_corpus(math_rows))
    evaluation = (
        empty_evaluation()
        if no_substrate
        else evaluate_judge(
            math_rows,
            ood_rows_by_domain,
            seeds=seeds,
            epochs=epochs,
            lr=lr,
            l2=l2,
        )
    )
    trained_transfer = bool(
        evaluation["ood_judge_auroc"] is not None
        and evaluation["shuffled_label_control_auroc"] is not None
        and evaluation["confidence_only_baseline_auroc"] is not None
        and float(evaluation["ood_judge_auroc"])
        > float(evaluation["shuffled_label_control_auroc"])
        and float(evaluation["ood_judge_auroc"])
        > float(evaluation["confidence_only_baseline_auroc"])
    )
    finished = time.perf_counter() if now_s is None else float(now_s)
    artifact: JsonDict = {
        "artifact": "experiment_3646_trained_ebm_judge_ood_counterpoint_v2",
        "schema": "carnot.trained_ebm_judge_ood_counterpoint.v2",
        "honest_verdict": terminal_verdict(
            no_trainable_substrate=no_substrate,
            ood_domains_tested=ood_domains,
            trained_judge_transfers_ood=trained_transfer,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "judge_recipe": judge_recipe(epochs=epochs, lr=lr, l2=l2, seeds=seeds),
        "in_domain_judge_auroc": evaluation["in_domain_judge_auroc"],
        "in_domain_judge_auroc_ci95": evaluation["in_domain_judge_auroc_ci95"],
        "ood_judge_auroc": evaluation["ood_judge_auroc"],
        "ood_judge_auroc_ci95": evaluation["ood_judge_auroc_ci95"],
        "ood_domains_tested": ood_domains,
        "ood_judge_auroc_by_domain": evaluation["ood_judge_auroc_by_domain"],
        "shuffled_label_control_auroc": evaluation["shuffled_label_control_auroc"],
        "shuffled_label_control_auroc_ci95": evaluation["shuffled_label_control_auroc_ci95"],
        "confidence_only_baseline_auroc": evaluation["confidence_only_baseline_auroc"],
        "confidence_only_baseline_auroc_ci95": evaluation["confidence_only_baseline_auroc_ci95"],
        "trained_judge_vs_fixed_ensemble_ood_delta": delta_vs_fixed(
            evaluation["ood_judge_auroc"]
        ),
        "fixed_ensemble_ood_reference_auroc": FIXED_ENSEMBLE_OOD_REFERENCE_AUROC,
        "fixed_ensemble_ood_reference_range": [0.329, 0.333],
        "trained_judge_transfers_ood": trained_transfer,
        "n_examples_per_domain": {
            "math": len(math_rows),
            **{domain: len(rows) for domain, rows in ood_rows_by_domain.items()},
        },
        "random_seed": int(seeds[0]) if seeds else RANDOM_SEED,
        "random_seeds_used": [int(seed) for seed in seeds],
        "reproducibility_checksum": reproducibility_checksum(
            math_rows,
            ood_rows_by_domain,
            evaluation,
            seeds,
        ),
        "duration_s": round(max(0.0, finished - start), 6),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate": {
            "condition": (
                "in_domain_judge_auroc present AND shuffled_label_control_auroc present "
                "AND ood_judge_auroc present"
            ),
            "passed": bool(
                evaluation["in_domain_judge_auroc"] is not None
                and evaluation["shuffled_label_control_auroc"] is not None
                and evaluation["ood_judge_auroc"] is not None
            ),
            "principle": (
                "An OOD-transfer claim requires an in-domain sanity pass AND a shuffled-label "
                "adversarial control -- without both the OOD number could be a degenerate-prior artifact."
            ),
        },
        "per_seed_results": evaluation["per_seed_results"],
        "source_artifacts": [str(EXP3640_REL_PATH), str(EXP3641_REL_PATH), str(FOVER_CORPUS_REL_PATH)],
        "tests_run": list(tests_run or []),
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
    }
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build and persist the Exp 3646 artifact."""

    root_path = Path(root)
    output = root_path / Path(output_path)
    artifact = build_artifact(root_path, tests_run=tests_run)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def evaluate_judge(
    math_examples: Sequence[JudgeExample],
    ood_examples_by_domain: Mapping[str, Sequence[JudgeExample]],
    *,
    seeds: Sequence[int],
    epochs: int,
    lr: float,
    l2: float,
) -> JsonDict:
    """Train on math labels and score held-out math plus available OOD rows."""

    ood_flat = [
        example
        for domain in sorted(ood_examples_by_domain)
        for example in ood_examples_by_domain[domain]
    ]
    per_seed: list[JsonDict] = []
    in_domain_values: list[float] = []
    ood_values: list[float] = []
    shuffled_values: list[float] = []
    confidence_values: list[float] = []
    ood_by_domain_values: dict[str, list[float]] = {domain: [] for domain in ood_examples_by_domain}
    for seed in seeds:
        train, heldout = stratified_train_eval_split(math_examples, seed=int(seed))
        X_train = [feature_vector(example) for example in train]
        y_train = [example.error_label for example in train]
        judge = SmallEnergyJudge(epochs=epochs, lr=lr, l2=l2).fit(X_train, y_train, seed=int(seed))
        heldout_labels = [example.error_label for example in heldout]
        heldout_scores = judge.predict_scores([feature_vector(example) for example in heldout])
        in_auc = tie_aware_auroc(heldout_labels, heldout_scores)
        in_domain_values.append(in_auc)
        seed_row: JsonDict = {"seed": int(seed), "in_domain_judge_auroc": round(in_auc, 6)}
        if ood_flat:
            ood_labels = [example.error_label for example in ood_flat]
            ood_scores = judge.predict_scores([feature_vector(example) for example in ood_flat])
            ood_auc = tie_aware_auroc(ood_labels, ood_scores)
            shuffled_y = shuffled_labels(y_train, seed=int(seed))
            shuffled_judge = SmallEnergyJudge(epochs=epochs, lr=lr, l2=l2).fit(
                X_train,
                shuffled_y,
                seed=int(seed) + 7919,
            )
            shuffled_scores = shuffled_judge.predict_scores(
                [feature_vector(example) for example in ood_flat]
            )
            shuffled_auc = tie_aware_auroc(ood_labels, shuffled_scores)
            confidence_auc = tie_aware_auroc(
                ood_labels,
                [example.confidence_error_signal for example in ood_flat],
            )
            ood_values.append(ood_auc)
            shuffled_values.append(shuffled_auc)
            confidence_values.append(confidence_auc)
            seed_row.update(
                {
                    "ood_judge_auroc": round(ood_auc, 6),
                    "shuffled_label_control_auroc": round(shuffled_auc, 6),
                    "confidence_only_baseline_auroc": round(confidence_auc, 6),
                }
            )
            for domain, rows in ood_examples_by_domain.items():
                labels = [example.error_label for example in rows]
                scores = judge.predict_scores([feature_vector(example) for example in rows])
                domain_auc = tie_aware_auroc(labels, scores)
                ood_by_domain_values.setdefault(domain, []).append(domain_auc)
        per_seed.append(seed_row)

    in_point, in_ci = summarize_values(in_domain_values)
    ood_point, ood_ci = summarize_values(ood_values)
    shuffled_point, shuffled_ci = summarize_values(shuffled_values)
    confidence_point, confidence_ci = summarize_values(confidence_values)
    return {
        "in_domain_judge_auroc": in_point,
        "in_domain_judge_auroc_ci95": in_ci,
        "ood_judge_auroc": ood_point,
        "ood_judge_auroc_ci95": ood_ci,
        "ood_judge_auroc_by_domain": {
            domain: metric_object(values) for domain, values in sorted(ood_by_domain_values.items())
        },
        "shuffled_label_control_auroc": shuffled_point,
        "shuffled_label_control_auroc_ci95": shuffled_ci,
        "confidence_only_baseline_auroc": confidence_point,
        "confidence_only_baseline_auroc_ci95": confidence_ci,
        "per_seed_results": per_seed,
    }


def empty_evaluation() -> JsonDict:
    return {
        "in_domain_judge_auroc": None,
        "in_domain_judge_auroc_ci95": None,
        "ood_judge_auroc": None,
        "ood_judge_auroc_ci95": None,
        "ood_judge_auroc_by_domain": {},
        "shuffled_label_control_auroc": None,
        "shuffled_label_control_auroc_ci95": None,
        "confidence_only_baseline_auroc": None,
        "confidence_only_baseline_auroc_ci95": None,
        "per_seed_results": [],
    }


def load_corpora(
    root: Path | str = REPO_ROOT,
    *,
    max_math_examples: int = 2000,
) -> tuple[list[JudgeExample], dict[str, list[JudgeExample]]]:
    """Load FoVer math plus every runnable OOD corpus declared by upstream artifacts."""

    root_path = Path(root)
    math_examples = load_math_examples(root_path / FOVER_CORPUS_REL_PATH, max_examples=max_math_examples)
    ood: dict[str, list[JudgeExample]] = {}
    code_examples = load_code_examples(root_path)
    if code_examples:
        ood["code"] = code_examples
    fact_examples = load_fact_examples(root_path)
    if fact_examples:
        ood["facts"] = fact_examples
    return math_examples, ood


def load_math_examples(path: Path, *, max_examples: int) -> list[JudgeExample]:
    rows = _read_jsonl(path)
    examples: list[JudgeExample] = []
    for row in rows[:max_examples]:
        label = str(row.get("label") or "").strip().lower()
        if label not in {"correct", "incorrect"}:
            continue
        text = str(row.get("step_text") or row.get("text") or "")
        examples.append(
            JudgeExample(
                domain="math",
                text=text,
                error_label=1 if label == "incorrect" else 0,
                validity_signal=math_reasoning_validity_signal(text),
                confidence_error_signal=1.0 - _clamp(_coerce_float(row.get("confidence"), 0.5)),
            )
        )
    return examples


def load_code_examples(root: Path) -> list[JudgeExample]:
    artifact = _read_json_object(root / EXP3641_REL_PATH)
    if artifact.get("code_verifiers_fire") is not True:
        return []
    corpus_path = artifact.get("code_corpus_path")
    if not isinstance(corpus_path, str) or not corpus_path:
        return []
    rows = _read_jsonl(root / corpus_path)
    examples: list[JudgeExample] = []
    for row in rows:
        if "label" not in row:
            continue
        code = str(row.get("candidate_code") or "")
        examples.append(
            JudgeExample(
                domain="code",
                text=code,
                error_label=0 if bool(row.get("label")) else 1,
                validity_signal=code_validity_signal(code),
                confidence_error_signal=code_confidence_error_signal(row),
            )
        )
    return examples


def load_fact_examples(root: Path) -> list[JudgeExample]:
    artifact = _read_json_object(root / EXP3640_REL_PATH)
    if artifact.get("facts_corpus_validated") is not True:
        return []
    corpus_path = artifact.get("corpus_path_used")
    if not isinstance(corpus_path, str) or not corpus_path:
        return []
    rows = _read_jsonl(root / corpus_path)
    verifier = RetrievalNLIGroundingVerifier()
    examples: list[JudgeExample] = []
    for row in rows:
        if "is_hallucination" not in row:
            continue
        answer = str(row.get("answer") or "")
        evidence = str(row.get("evidence_passage") or "")
        text = f"{row.get('question', '')}\n{answer}\n{evidence}"
        examples.append(
            JudgeExample(
                domain="facts",
                text=text,
                error_label=1 if bool(row.get("is_hallucination")) else 0,
                validity_signal=float(verifier.verify(answer, evidence)),
                confidence_error_signal=1.0
                - _clamp(_coerce_float(row.get("model_confidence"), 0.5)),
            )
        )
    return examples


def feature_vector(example: JudgeExample) -> list[float]:
    text = str(example.text)
    n_chars = len(text)
    n_digits = sum(1 for char in text if char.isdigit())
    return [
        float(example.validity_signal),
        float(example.confidence_error_signal),
        math.log1p(n_chars),
        n_digits / max(1, n_chars),
        format_noise_signal(text),
    ]


def math_reasoning_validity_signal(text: str) -> float:
    arithmetic = arithmetic_error_rate(text)
    marker_noise = format_noise_signal(text)
    correction_markers = len(re.findall(r"\b(wait|wrong|mistake|incorrect|contradict)\b", text.lower()))
    correction_signal = min(1.0, correction_markers / 3.0)
    return _clamp(0.65 * arithmetic + 0.25 * marker_noise + 0.10 * correction_signal)


def code_validity_signal(code: str) -> float:
    try:
        ast.parse(code)
        syntax_error = 0.0
    except SyntaxError:
        syntax_error = 1.0
    return _clamp(0.75 * syntax_error + 0.25 * format_noise_signal(code))


def code_confidence_error_signal(row: Mapping[str, Any]) -> float:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
    syntax_success = metadata.get("syntax_success")
    runtime_success = metadata.get("runtime_success")
    if syntax_success is False or runtime_success is False:
        return 0.75
    if syntax_success is True and runtime_success is True:
        return 0.25
    candidate_index = _coerce_float(metadata.get("candidate_index"), 1.0)
    return _clamp(0.25 + min(candidate_index, 3.0) / 6.0)


def arithmetic_error_rate(text: str) -> float:
    matches = list(
        re.finditer(
            r"(-?\d+(?:\.\d+)?)\s*([+\-*/xX×÷])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)",
            text,
        )
    )
    if not matches:
        return 0.0
    wrong = 0
    for match in matches:
        left = float(match.group(1))
        op = match.group(2)
        right = float(match.group(3))
        claimed = float(match.group(4))
        expected = _apply_arithmetic(left, op, right)
        wrong += int(expected is not None and abs(expected - claimed) > 1e-6)
    return wrong / len(matches)


def format_noise_signal(text: str) -> float:
    lowered = text.lower()
    markers = lowered.count("```") + lowered.count("<channel|>") + lowered.count("***")
    long_line_penalty = 1 if any(len(line) > 240 for line in text.splitlines() or [text]) else 0
    repetition_penalty = 1 if repeated_token_fraction(text) > 0.35 else 0
    return _clamp((markers + long_line_penalty + repetition_penalty) / 5.0)


def repeated_token_fraction(text: str) -> float:
    tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
    if not tokens:
        return 0.0
    return 1.0 - len(set(tokens)) / len(tokens)


def trainable_math_corpus(examples: Sequence[JudgeExample]) -> bool:
    labels = [example.error_label for example in examples]
    return len(labels) >= 4 and 0 in labels and 1 in labels


def stratified_train_eval_split(
    examples: Sequence[JudgeExample],
    *,
    seed: int,
    train_fraction: float = 0.8,
) -> tuple[list[JudgeExample], list[JudgeExample]]:
    rng = np.random.default_rng(int(seed))
    train_indices: list[int] = []
    eval_indices: list[int] = []
    for label in (0, 1):
        indices = [idx for idx, example in enumerate(examples) if example.error_label == label]
        shuffled = list(rng.permutation(indices))
        n_train = min(len(shuffled) - 1, max(1, int(round(len(shuffled) * train_fraction))))
        train_indices.extend(int(idx) for idx in shuffled[:n_train])
        eval_indices.extend(int(idx) for idx in shuffled[n_train:])
    return [examples[idx] for idx in train_indices], [examples[idx] for idx in eval_indices]


def shuffled_labels(labels: Sequence[int], *, seed: int) -> list[int]:
    rng = np.random.default_rng(int(seed) + 17)
    return [int(value) for value in rng.permutation(np.asarray(labels, dtype=np.int64))]


def tie_aware_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = [float(score) for label, score in zip(labels, scores, strict=False) if int(label) == 1]
    negatives = [float(score) for label, score in zip(labels, scores, strict=False) if int(label) == 0]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def summarize_values(values: Sequence[float]) -> tuple[float | None, list[float] | None]:
    if not values:
        return None, None
    arr = np.asarray(values, dtype=np.float64)
    ci_low, ci_high = np.percentile(arr, [2.5, 97.5])
    return round(float(arr.mean()), 6), [round(float(ci_low), 6), round(float(ci_high), 6)]


def metric_object(values: Sequence[float]) -> JsonDict:
    point, ci = summarize_values(values)
    return {"point": point, "ci95": ci, "seed_values": [round(float(v), 6) for v in values]}


def terminal_verdict(
    *,
    no_trainable_substrate: bool,
    ood_domains_tested: Sequence[str],
    trained_judge_transfers_ood: bool,
) -> str:
    if no_trainable_substrate:
        return "complete: blocked_no_trainable_substrate"
    if not ood_domains_tested:
        return "complete: blocked_no_ood_eval_corpus"
    if trained_judge_transfers_ood:
        return "complete: trained_ebm_judge_transfers_ood_fixed_ensemble_was_the_bottleneck"
    return "complete: trained_ebm_judge_also_math_only_transfer_not_a_training_artifact"


def judge_recipe(*, epochs: int, lr: float, l2: float, seeds: Sequence[int]) -> JsonDict:
    return {
        "model": "tiny_logistic_energy_head",
        "features": list(FEATURE_NAMES),
        "loss": "L2-regularized binary cross-entropy on math error_label; score is P(error).",
        "epochs": int(epochs),
        "learning_rate": float(lr),
        "l2": float(l2),
        "train_eval_split": "80/20 stratified FoVer math split per seed; OOD labels are never used for training.",
        "random_seeds": [int(seed) for seed in seeds],
    }


def delta_vs_fixed(ood_auroc: float | None) -> float | None:
    if ood_auroc is None:
        return None
    return round(float(ood_auroc) - FIXED_ENSEMBLE_OOD_REFERENCE_AUROC, 6)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be present")
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles]
    if missing_principles:
        raise ValueError(f"missing field principles: {missing_principles}")
    if artifact.get("honest_verdict") not in {
        "complete: trained_ebm_judge_transfers_ood_fixed_ensemble_was_the_bottleneck",
        "complete: trained_ebm_judge_also_math_only_transfer_not_a_training_artifact",
        "complete: blocked_no_ood_eval_corpus",
        "complete: blocked_no_trainable_substrate",
    }:
        raise ValueError("honest_verdict is not an allowed Exp 3646 terminal verdict")
    if type(artifact.get("trained_judge_transfers_ood")) is not bool:
        raise ValueError("trained_judge_transfers_ood must be a bare bool")
    if not isinstance(artifact.get("acceptance_gate"), Mapping):
        raise ValueError("acceptance_gate must be present")
    duration = artifact.get("duration_s")
    if not isinstance(duration, int | float) or float(duration) < 0.0:
        raise ValueError("duration_s must be a non-negative number")


def reproducibility_checksum(
    math_examples: Sequence[JudgeExample],
    ood_examples_by_domain: Mapping[str, Sequence[JudgeExample]],
    evaluation: Mapping[str, Any],
    seeds: Sequence[int],
) -> str:
    payload = {
        "math": _example_checksum_payload(math_examples),
        "ood": {
            domain: _example_checksum_payload(rows)
            for domain, rows in sorted(ood_examples_by_domain.items())
        },
        "evaluation": evaluation,
        "seeds": [int(seed) for seed in seeds],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _example_checksum_payload(examples: Sequence[JudgeExample]) -> JsonDict:
    return {
        "n": len(examples),
        "labels": [int(example.error_label) for example in examples[:20]],
        "signals": [
            [round(float(example.validity_signal), 6), round(float(example.confidence_error_signal), 6)]
            for example in examples[:20]
        ],
    }


def _read_json_object(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path) -> list[JsonDict]:
    if not path.exists():
        return []
    rows: list[JsonDict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _apply_arithmetic(left: float, op: str, right: float) -> float | None:
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op in {"*", "x", "X", "×"}:
        return left * right
    if op in {"/", "÷"} and right != 0:
        return left / right
    return None


def _coerce_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))
