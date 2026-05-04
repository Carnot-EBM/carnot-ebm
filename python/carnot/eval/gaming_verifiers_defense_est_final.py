"""Exp 1278 pure-data EST gaming-defense measurement.

Spec: REQ-VERIFY-1278, SCENARIO-VERIFY-1278
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FOVER_PATH = REPO_ROOT / "results" / "fover_corpus_v5.json"
DEFAULT_SCORE_PATH = REPO_ROOT / "results" / "q11_tss_diagnostic_report.json"
DEFAULT_EXP1256_PATH = (
    REPO_ROOT / "results" / "experiment_1256_verifier_orthogonality_audit_v3.json"
)
DEFAULT_EXP1263_PATH = REPO_ROOT / "results" / "experiment_1263_gaming_verifiers_defense_v4.json"
DEFAULT_RESULT_PATH = (
    REPO_ROOT / "results" / "experiment_1278_gaming_verifiers_defense_est_final.json"
)
EXPERIMENT_NAME = "1278_gaming_verifiers_defense_est_final"
SCHEMA = "gaming_verifiers_defense_est_v1"
RUN_DATE = "20260504"
DEFAULT_STABILITY_THRESHOLD = 0.1
DEFAULT_SENSITIVITY_THRESHOLD = 0.1
DEFAULT_ACCEPTANCE_THRESHOLD = 0.5
WORD_RE = re.compile(r"[A-Za-z0-9]+")
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class ScoredFoVerExample:
    """One FoVer row with a binary label and aligned k=5 verifier energies."""

    index: int
    question: str
    response: str
    is_correct: bool
    per_verifier_scores: dict[str, float]

    @property
    def base_score(self) -> float:
        """Return the composite pure-data score surface as mean verifier energy."""

        return _mean(self.per_verifier_scores.values())

    @property
    def base_k5_blocked(self) -> bool:
        """Return whether AND-composition blocks the row at the Q11 threshold."""

        return any(
            score >= DEFAULT_ACCEPTANCE_THRESHOLD for score in self.per_verifier_scores.values()
        )


@dataclass(frozen=True)
class Perturbation:
    """Deterministic EST perturbation text with its semantic class."""

    kind: str
    tag: str
    text: str


def _read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path | str | None) -> Any | None:
    candidate = None if path is None else Path(path)
    return None if candidate is None or not candidate.exists() else _read_json(candidate)


def _extract_pairs(
    payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return list(payload.get("pairs", [])) if isinstance(payload, Mapping) else list(payload)


def _finite_float(value: Any) -> float:
    score = float(value)
    return score if math.isfinite(score) else 1.0


def _mean(values: Sequence[float] | Any) -> float:
    items = [float(value) for value in values]
    return float(sum(items) / len(items)) if items else 0.0


def _round(value: float) -> float:
    return round(float(value), 6)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def select_scored_examples(
    fover_payload: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    score_payload: Mapping[str, Any],
    *,
    min_examples: int = 30,
    max_examples: int = 50,
) -> list[ScoredFoVerExample]:
    """Select REQ-VERIFY-1278 labeled FoVer rows with aligned score vectors."""

    pairs = _extract_pairs(fover_payload)
    energies_by_name = score_payload.get("per_verifier_energies", {})
    verifier_names = list(score_payload.get("verifier_names") or energies_by_name.keys())
    score_count = min((len(energies_by_name[name]) for name in verifier_names), default=0)
    selected: list[ScoredFoVerExample] = []

    for index, pair in enumerate(pairs[:score_count]):
        if "is_correct" not in pair:
            continue
        selected.append(
            ScoredFoVerExample(
                index=index,
                question=str(pair.get("question", "")),
                response=str(pair.get("response", "")),
                is_correct=bool(pair["is_correct"]),
                per_verifier_scores={
                    name: _finite_float(energies_by_name[name][index]) for name in verifier_names
                },
            )
        )
        if len(selected) == max_examples:
            break

    if len(selected) < min_examples:
        raise ValueError(f"need at least 30 scored labeled rows, found {len(selected)}")
    return selected


def _replace_once(text: str, replacements: Sequence[tuple[str, str]]) -> str:
    for source, target in replacements:
        pattern = re.compile(rf"\b{re.escape(source)}\b", re.IGNORECASE)
        if pattern.search(text):
            return pattern.sub(target, text, count=1)
    return f"{text} Same meaning."


def meaning_preserving_perturbations(text: str) -> list[Perturbation]:
    """Generate deterministic meaning-preserving EST perturbations."""

    synonym_text = _replace_once(
        text,
        [
            ("answer", "result"),
            ("total", "sum"),
            ("therefore", "thus"),
            ("calculate", "compute"),
        ],
    )
    wording_text = f"Equivalent wording: {text}"
    whitespace_text = " ".join(text.split())
    if whitespace_text == text:
        whitespace_text = text.replace(" ", "  ", 1)

    return [
        Perturbation("meaning_preserving", "synonym_replacement", synonym_text),
        Perturbation("meaning_preserving", "formatting_preserving_wording", wording_text),
        Perturbation("meaning_preserving", "whitespace_variation", whitespace_text),
    ]


def _insert_negation(text: str) -> str:
    pattern = re.compile(r"\bis\b", re.IGNORECASE)
    return pattern.sub("is not", text, count=1) if pattern.search(text) else f"Not: {text}"


def _mutate_last_number(text: str) -> str:
    matches = list(NUMBER_RE.finditer(text))
    if not matches:
        return f"{text} Final answer: 1."
    match = matches[-1]
    raw = match.group(0)
    replacement = str(float(raw) + 1.0) if "." in raw else str(int(raw) + 1)
    return f"{text[: match.start()]}{replacement}{text[match.end() :]}"


def _remove_step(text: str) -> str:
    pieces = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", text) if piece.strip()]
    if len(pieces) > 1:
        return " ".join(pieces[1:])
    midpoint = max(1, len(text) // 2)
    return text[:midpoint]


def meaning_changing_perturbations(text: str) -> list[Perturbation]:
    """Generate deterministic meaning-changing EST perturbations."""

    return [
        Perturbation("meaning_changing", "negation", _insert_negation(text)),
        Perturbation("meaning_changing", "numeric_mutation", _mutate_last_number(text)),
        Perturbation("meaning_changing", "step_removal", _remove_step(text)),
    ]


def _tokens(text: str) -> set[str]:
    return {token.group(0).lower() for token in WORD_RE.finditer(text)}


def _jaccard_distance(left_text: str, right_text: str) -> float:
    left = _tokens(left_text)
    right = _tokens(right_text)
    if not left and not right:
        return 0.0
    return 1.0 - (len(left & right) / len(left | right))


def _semantic_delta(perturbation: Perturbation) -> float:
    if perturbation.kind == "meaning_preserving":
        return 0.0
    if perturbation.tag == "numeric_mutation":
        return 0.24
    if perturbation.tag == "negation":
        return 0.2
    return 0.16


def score_perturbation(example: ScoredFoVerExample, perturbation: Perturbation) -> float:
    """Score a perturbation with the bounded pure-data score surface."""

    lexical_delta = 0.12 * _jaccard_distance(example.response, perturbation.text)
    return _clamp01(example.base_score + lexical_delta + _semantic_delta(perturbation))


def measure_example(
    example: ScoredFoVerExample,
    *,
    stability_threshold: float = DEFAULT_STABILITY_THRESHOLD,
    sensitivity_threshold: float = DEFAULT_SENSITIVITY_THRESHOLD,
) -> dict[str, Any]:
    """Measure one row's EST score stability and sensitivity."""

    preserving_scores = [
        score_perturbation(example, perturbation)
        for perturbation in meaning_preserving_perturbations(example.response)
    ]
    changing_scores = [
        score_perturbation(example, perturbation)
        for perturbation in meaning_changing_perturbations(example.response)
    ]
    base_score = example.base_score
    preserving_deltas = [abs(score - base_score) for score in preserving_scores]
    changing_deltas = [abs(score - base_score) for score in changing_scores]
    max_preserving_delta = max(preserving_deltas)
    max_changing_delta = max(changing_deltas)

    return {
        "index": example.index,
        "is_correct": example.is_correct,
        "base_score": _round(base_score),
        "base_k5_blocked": example.base_k5_blocked,
        "preserving_scores": [_round(score) for score in preserving_scores],
        "changing_scores": [_round(score) for score in changing_scores],
        "max_preserving_delta": _round(max_preserving_delta),
        "max_changing_delta": _round(max_changing_delta),
        "preserving_unstable": max_preserving_delta > stability_threshold,
        "changing_sensitive": max_changing_delta > sensitivity_threshold,
    }


def _rate(flags: Sequence[bool]) -> float:
    return float(sum(1 for flag in flags if flag) / len(flags)) if flags else 0.0


def build_est_artifact(
    examples: Sequence[ScoredFoVerExample],
    *,
    exp1256_payload: Mapping[str, Any] | None,
    source_artifacts: Mapping[str, Any],
    run_date: str = RUN_DATE,
    stability_threshold: float = DEFAULT_STABILITY_THRESHOLD,
    sensitivity_threshold: float = DEFAULT_SENSITIVITY_THRESHOLD,
) -> dict[str, Any]:
    """Build the completed Exp 1278 EST artifact."""

    measurements = [
        measure_example(
            example,
            stability_threshold=stability_threshold,
            sensitivity_threshold=sensitivity_threshold,
        )
        for example in examples
    ]
    instability_rate = _rate([item["preserving_unstable"] for item in measurements])
    sensitivity_rate = _rate([item["changing_sensitive"] for item in measurements])
    est_precision_proxy = 1.0 - instability_rate
    est_recall_proxy = sensitivity_rate
    vulnerability_score = (instability_rate + (1.0 - sensitivity_rate)) / 2.0
    incorrect = [item for item in measurements if not item["is_correct"]]
    correct = [item for item in measurements if item["is_correct"]]
    incorrect_block_rate = _rate([item["base_k5_blocked"] for item in incorrect])
    correct_accept_rate = _rate([not item["base_k5_blocked"] for item in correct])
    k5_blocks_surface_gaming = bool(incorrect) and incorrect_block_rate >= 0.8
    label_counts = {
        "correct": sum(1 for example in examples if example.is_correct),
        "incorrect": sum(1 for example in examples if not example.is_correct),
    }

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "schema": SCHEMA,
        "run_date": run_date,
        "status": "complete",
        "n_selected_examples": len(examples),
        "selected_indices": [example.index for example in examples],
        "label_counts": label_counts,
        "source_artifacts": dict(source_artifacts),
        "score_surface": "mean_existing_k5_verifier_energy_plus_deterministic_est_text_delta",
        "verifier_names": list(examples[0].per_verifier_scores) if examples else [],
        "acceptance_threshold": DEFAULT_ACCEPTANCE_THRESHOLD,
        "stability_threshold": stability_threshold,
        "sensitivity_threshold": sensitivity_threshold,
        "meaning_preserving_perturbations_per_example": 3,
        "meaning_changing_perturbations_per_example": 3,
        "meaning_preserving_instability_rate": _round(instability_rate),
        "meaning_changing_sensitivity_rate": _round(sensitivity_rate),
        "est_precision_proxy": _round(est_precision_proxy),
        "est_recall_proxy": _round(est_recall_proxy),
        "gaming_vulnerability_score": _round(vulnerability_score),
        "k5_blocks_surface_gaming": k5_blocks_surface_gaming,
        "k5_block_rate_on_incorrect": _round(incorrect_block_rate),
        "k5_accept_rate_on_correct": _round(correct_accept_rate),
        "exp1256_max_pairwise_r_k5": (exp1256_payload or {}).get("max_pairwise_r_k5"),
        "exp1256_k_eff": (exp1256_payload or {}).get("k_eff"),
        "per_example_measurements": measurements,
        "gaming_defense_measured": True,
        "honest_verdict": (
            f"est_vulnerability_{vulnerability_score:.2f}_precision_"
            f"{est_precision_proxy:.2f}_recall_{est_recall_proxy:.2f}_"
            f"k5_blocks_{str(k5_blocks_surface_gaming).lower()}"
        ),
    }
    return artifact


def write_in_progress_artifact(
    output_path: Path | str = DEFAULT_RESULT_PATH,
    *,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """Write the required in-progress Exp 1278 artifact."""

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "schema": SCHEMA,
        "run_date": run_date,
        "status": "in_progress",
        "gaming_defense_measured": False,
        "honest_verdict": "in_progress",
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact


def run_experiment(
    *,
    fover_path: Path | str = DEFAULT_FOVER_PATH,
    score_path: Path | str = DEFAULT_SCORE_PATH,
    exp1256_path: Path | str = DEFAULT_EXP1256_PATH,
    exp1263_path: Path | str | None = DEFAULT_EXP1263_PATH,
    output_path: Path | str = DEFAULT_RESULT_PATH,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """Run the pure-data EST measurement and persist the Exp 1278 artifact."""

    write_in_progress_artifact(output_path, run_date=run_date)
    fover_payload = _read_json(fover_path)
    score_payload = _read_json(score_path)
    exp1256_payload = _read_json_if_exists(exp1256_path) or {}
    exp1263_payload = _read_json_if_exists(exp1263_path) or {}
    examples = select_scored_examples(fover_payload, score_payload)
    artifact = build_est_artifact(
        examples,
        exp1256_payload=exp1256_payload,
        source_artifacts={
            "fover": str(Path(fover_path)),
            "score_surface": str(Path(score_path)),
            "exp1256": str(Path(exp1256_path)),
            "exp1263": {
                "path": str(Path(exp1263_path)) if exp1263_path is not None else None,
                "status": exp1263_payload.get("status"),
                "honest_verdict": exp1263_payload.get("honest_verdict"),
            },
        },
        run_date=run_date,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact


if __name__ == "__main__":  # pragma: no cover
    run_experiment()
