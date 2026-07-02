"""Exp 5163: MMLU-Pro few-shot verifier-vs-cheap-baseline rescale.

Spec refs: REQ-VERIFY-5163, SCENARIO-VERIFY-5163,
SCENARIO-VERIFY-5163-BLOCKED-POOL.

This module reuses the cached 5-shot MMLU-Pro candidate pool from the July 1
headroom run. It does not generate candidates. The complete path scores the
same MiniLM+LogisticRegression verifier and the same cheap text-feature
LogisticRegression baseline used by the prior zero-shot verifier experiment,
then writes a small terminal artifact for the conductor.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from scripts.experiments.exp_mmlu_pro_verifier_vs_cheap_baseline import (
    bootstrap_ci95_delta,
    cheap_features,
    embed_texts as _legacy_embed_texts,
    leave_one_question_out_scores,
    selection_accuracy,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5163_mmlu_pro_verifier_rescale_v473"
RESULT_RELATIVE_PATH = "results/experiment_5163_mmlu_pro_verifier_rescale_v473.json"
POOL_RELATIVE_PATH = "results/experiment_mmlu_pro_fewshot_candidate_pool.jsonl"
ZEROSHOT_RESULT_RELATIVE_PATH = "results/experiment_mmlu_pro_verifier_vs_cheap_baseline.json"
SPEC_REFS = [
    "REQ-VERIFY-5163",
    "SCENARIO-VERIFY-5163",
    "SCENARIO-VERIFY-5163-BLOCKED-POOL",
]

EXPECTED_N_QUESTIONS = 40
K_SAMPLES = 6
RANDOM_SEED = 20260701
DEFAULT_N_BOOT = 2000
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_", "blocked_")
ZERO_SHOT_HEADROOM_CONTEXT = 0.275

FIELD_PRINCIPLES = {
    "fewshot_oracle_at_k": (
        "The selectable ceiling in the cached 5-shot pool; it is not verifier performance."
    ),
    "fewshot_sc_vote_accuracy": (
        "The genuine self-consistency baseline on the same cached K=6 rows."
    ),
    "fewshot_verifier_selection_accuracy": (
        "Top-candidate selection accuracy from MiniLM reasoning embeddings plus logistic regression, leave-one-question-out."
    ),
    "fewshot_cheap_baseline_selection_accuracy": (
        "Matched logistic-regression selector over non-learned text-statistical features, same folds and rows."
    ),
    "verifier_vs_cheap_delta": (
        "The primary claim variable: learned verifier selection accuracy minus cheap-baseline selection accuracy."
    ),
    "verifier_vs_cheap_delta_ci95": (
        "Bootstrap CI over questions for the primary delta; the verdict must state whether it excludes zero."
    ),
    "vs_zeroshot_pool_comparison": (
        "Direct comparison to the saved zero-shot-pool verifier result, including point estimate and CI status."
    ),
    "still_underpowered": (
        "True when n=40 or the CI crosses zero/class imbalance remains too severe for a decisive claim."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- the verifier sees candidate reasoning only; gold is used after selection for evaluation."
    ),
    "random_seed": (
        "The deterministic seed shared by the pool sample, logistic regression, and bootstrap."
    ),
    "reproducibility_checksum": (
        "Hash of the cached pool, zero-shot comparison artifact, scores, metrics, and verdict."
    ),
    "honest_verdict": (
        "Must start with complete:/complete_/success:/success_ or blocked_, and must plainly state whether the CI excludes 0."
    ),
}
REQUIRED_PRINCIPLED_FIELDS = tuple(FIELD_PRINCIPLES)

JsonDict = dict[str, Any]
ScoreCandidatesFn = Callable[[Sequence[Mapping[str, Any]], Mapping[int, int]], tuple[np.ndarray, np.ndarray]]
OptionCountsLoader = Callable[[Sequence[Mapping[str, Any]], Path], Mapping[int, int]]
EmbedTextsFn = Callable[[Sequence[str]], np.ndarray]


class PoolIncomplete(ValueError):
    """Raised when the cached 5-shot pool cannot support the verifier test."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _wrap(value: Any, principle: str) -> JsonDict:
    return {"value": value, "principle": principle}


def _principled(field: str, value: Any) -> JsonDict:
    return _wrap(value, FIELD_PRINCIPLES[field])


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = json.loads(json.dumps(dict(artifact), sort_keys=True, default=str))
    checksum = payload.get("reproducibility_checksum")
    if isinstance(checksum, Mapping):
        checksum = dict(checksum)
        checksum["value"] = ""
        payload["reproducibility_checksum"] = checksum
    else:
        payload["reproducibility_checksum"] = {"value": ""}
    return "sha256:" + hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> JsonDict:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return dict(loaded) if isinstance(loaded, Mapping) else {}


def pool_precondition_errors(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_questions: int = EXPECTED_N_QUESTIONS,
    k_samples: int = K_SAMPLES,
) -> list[str]:
    errors: list[str] = []
    if len(rows) != expected_questions * k_samples:
        errors.append(
            f"expected {expected_questions * k_samples} rows, found {len(rows)}"
        )
    by_q: dict[int, list[Mapping[str, Any]]] = {}
    for i, row in enumerate(rows, 1):
        qi = row.get("question_index")
        if not isinstance(qi, int):
            errors.append(f"row {i} missing integer question_index")
            continue
        by_q.setdefault(qi, []).append(row)
        text = row.get("full_text")
        if not isinstance(text, str) or len(text.strip()) < 20:
            errors.append(f"row {i} missing full reasoning text")
        if "correct" not in row:
            errors.append(f"row {i} missing correct label")
        if "gold" not in row:
            errors.append(f"row {i} missing gold label")
    if len(by_q) != expected_questions:
        errors.append(f"expected {expected_questions} questions, found {len(by_q)}")
    bad_counts = {qi: len(q_rows) for qi, q_rows in by_q.items() if len(q_rows) != k_samples}
    if bad_counts:
        errors.append(f"questions without K={k_samples} coverage: {bad_counts}")
    return errors


def load_candidate_pool(
    path: Path,
    *,
    expected_questions: int = EXPECTED_N_QUESTIONS,
    k_samples: int = K_SAMPLES,
) -> list[JsonDict]:
    if not path.exists():
        raise PoolIncomplete([f"missing pool: {path}"])
    rows: list[JsonDict] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            loaded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PoolIncomplete([f"row {i} is not valid JSON: {exc}"]) from exc
        rows.append(dict(loaded) if isinstance(loaded, Mapping) else {"_bad_row": loaded})
    errors = pool_precondition_errors(
        rows, expected_questions=expected_questions, k_samples=k_samples
    )
    if errors:
        raise PoolIncomplete(errors)
    return rows


def compute_pool_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_q: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_q.setdefault(int(row["question_index"]), []).append(row)

    oracle_hits = sum(any(bool(row.get("correct")) for row in q_rows) for q_rows in by_q.values())
    sc_hits = 0
    for q_rows in by_q.values():
        letters = [row.get("parsed_letter") for row in q_rows if row.get("parsed_letter")]
        if not letters:
            continue
        vote = Counter(letters).most_common(1)[0][0]
        if vote == q_rows[0].get("gold"):
            sc_hits += 1

    n_questions = len(by_q)
    return {
        "n_questions": n_questions,
        "n_candidates": len(rows),
        "n_correct_candidates": sum(1 for row in rows if bool(row.get("correct"))),
        "n_unparseable_candidates": sum(1 for row in rows if row.get("parsed_letter") is None),
        "oracle_at_k": oracle_hits / n_questions if n_questions else 0.0,
        "sc_vote_accuracy": sc_hits / n_questions if n_questions else 0.0,
    }


def _default_embed_texts(texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - external model path
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return _legacy_embed_texts(list(texts), device)


def score_candidates(
    rows: Sequence[Mapping[str, Any]],
    option_counts_by_question: Mapping[int, int],
    *,
    embed_texts_fn: EmbedTextsFn = _default_embed_texts,
) -> tuple[np.ndarray, np.ndarray]:
    question_idx = np.array([int(row["question_index"]) for row in rows])
    y = np.array([1 if row.get("correct") else 0 for row in rows])
    n_options_by_row = [
        int(option_counts_by_question.get(int(row["question_index"]), 10)) for row in rows
    ]

    cheap_X = np.array(
        [
            cheap_features(str(row.get("full_text", "")), n_options)
            for row, n_options in zip(rows, n_options_by_row)
        ]
    )
    cheap_scores = leave_one_question_out_scores(cheap_X, y, question_idx)

    emb_X = np.asarray(
        embed_texts_fn([str(row.get("full_text", "")) for row in rows]), dtype=float
    )
    verifier_scores = leave_one_question_out_scores(emb_X, y, question_idx)
    return verifier_scores, cheap_scores


_OPTION_RE = re.compile(r"[\(\s]([A-J])[\)\.]")


def _infer_option_count_from_text(text: str, gold: Any, parsed: Any) -> int:  # pragma: no cover - fallback only
    letters = {match.group(1) for match in _OPTION_RE.finditer(text)}
    for value in (gold, parsed):
        if isinstance(value, str) and len(value) == 1 and "A" <= value <= "J":
            letters.add(value)
    if not letters:
        return 10
    return max(ord(letter) - ord("A") + 1 for letter in letters)


def load_option_counts_from_mmlu_pro(
    rows: Sequence[Mapping[str, Any]], root: Path
) -> Mapping[int, int]:  # pragma: no cover - exercised by the real experiment run
    try:
        from datasets import load_dataset

        max_qi = max(int(row["question_index"]) for row in rows)
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        ds = ds.shuffle(seed=RANDOM_SEED).select(range(max_qi + 1))
        return {qi: len(row["options"]) for qi, row in enumerate(ds)}
    except Exception:
        by_q: dict[int, list[Mapping[str, Any]]] = {}
        for row in rows:
            by_q.setdefault(int(row["question_index"]), []).append(row)
        return {
            qi: max(
                _infer_option_count_from_text(
                    str(row.get("full_text", "")), row.get("gold"), row.get("parsed_letter")
                )
                for row in q_rows
            )
            for qi, q_rows in by_q.items()
        }


def _ci_excludes_zero(ci95: Sequence[float]) -> bool:
    return bool(ci95[0] > 0.0 or ci95[1] < 0.0)


def _format_ci(ci95: Sequence[float]) -> str:
    return f"[{ci95[0]:.3f},{ci95[1]:.3f}]"


def _zero_value(zero_artifact: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = zero_artifact.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _zero_ci(zero_artifact: Mapping[str, Any]) -> list[float]:
    value = zero_artifact.get("delta_verifier_vs_cheap_baseline_ci95", [0.0, 0.0])
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
    ):
        return [float(value[0]), float(value[1])]
    return [0.0, 0.0]


def _zeroshot_comparison(
    *,
    delta: float,
    ci95: Sequence[float],
    zero_artifact: Mapping[str, Any],
    fewshot_headroom: float,
) -> str:
    zero_delta = _zero_value(zero_artifact, "delta_verifier_vs_cheap_baseline")
    zero_ci = _zero_ci(zero_artifact)
    tolerance = 5e-4
    direction_phrase = (
        "more favorable than zero-shot"
        if delta > zero_delta + tolerance
        else "less favorable than zero-shot"
        if delta < zero_delta - tolerance
        else "unchanged relative to zero-shot"
    )
    ci_status = "excludes 0" if _ci_excludes_zero(ci95) else "includes 0"
    return (
        f"The saved zero-shot-pool verifier result reports delta {zero_delta:+.3f} "
        f"with CI95={_format_ci(zero_ci)}; the 5-shot pool raises headroom to "
        f"{fewshot_headroom:.3f} versus {ZERO_SHOT_HEADROOM_CONTEXT:.3f} in the zero-shot "
        "headroom check. The 5-shot verifier-vs-cheap delta is "
        f"{delta:+.3f} with CI95={_format_ci(ci95)}, so the point estimate is "
        f"{direction_phrase} and the verifier-vs-cheap CI {ci_status}."
    )


def build_blocked_artifact(
    *,
    errors: Sequence[str],
    pool_path: Path,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "pool_path": str(pool_path),
        "pool_precondition": _wrap(
            {"complete": False, "errors": list(errors)},
            "Records why the required cached few-shot pool could not support scoring.",
        ),
        "pool_reused": _wrap(False, "True only when the complete cached pool is scored."),
        "candidate_generation_performed": _wrap(
            False, "This task must never regenerate the candidate pool."
        ),
        "fewshot_oracle_at_k": _principled("fewshot_oracle_at_k", None),
        "fewshot_sc_vote_accuracy": _principled("fewshot_sc_vote_accuracy", None),
        "fewshot_verifier_selection_accuracy": _principled(
            "fewshot_verifier_selection_accuracy", None
        ),
        "fewshot_cheap_baseline_selection_accuracy": _principled(
            "fewshot_cheap_baseline_selection_accuracy", None
        ),
        "verifier_vs_cheap_delta": _principled("verifier_vs_cheap_delta", None),
        "verifier_vs_cheap_delta_ci95": _principled("verifier_vs_cheap_delta_ci95", None),
        "vs_zeroshot_pool_comparison": _principled(
            "vs_zeroshot_pool_comparison",
            "Blocked before zero-shot comparison because the required 5-shot pool was incomplete.",
        ),
        "still_underpowered": _principled("still_underpowered", True),
        "verifier_is_oracle": _principled("verifier_is_oracle", False),
        "random_seed": _principled("random_seed", int(random_seed)),
        "reproducibility_checksum": _principled("reproducibility_checksum", ""),
        "honest_verdict": _principled(
            "honest_verdict", "blocked_fewshot_pool_incomplete"
        ),
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    artifact["reproducibility_checksum"]["value"] = payload_checksum(artifact)
    return artifact


def build_complete_artifact(
    *,
    rows: Sequence[Mapping[str, Any]],
    verifier_scores: np.ndarray,
    cheap_scores: np.ndarray,
    zero_artifact: Mapping[str, Any],
    pool_sha256: str,
    zero_sha256: str,
    random_seed: int = RANDOM_SEED,
    n_boot: int = DEFAULT_N_BOOT,
) -> JsonDict:
    metrics = compute_pool_metrics(rows)
    verifier_selection_acc = selection_accuracy(list(rows), np.asarray(verifier_scores, dtype=float))
    cheap_selection_acc = selection_accuracy(list(rows), np.asarray(cheap_scores, dtype=float))
    delta = verifier_selection_acc - cheap_selection_acc
    ci95 = bootstrap_ci95_delta(
        list(rows),
        np.asarray(verifier_scores, dtype=float),
        np.asarray(cheap_scores, dtype=float),
        seed=random_seed,
        n_boot=n_boot,
    )
    ci_excludes_zero = _ci_excludes_zero(ci95)
    still_underpowered = bool(
        metrics["n_questions"] <= EXPECTED_N_QUESTIONS
        or metrics["n_correct_candidates"] < 50
        or not ci_excludes_zero
    )
    verdict_status = "CI_excludes_0" if ci_excludes_zero else "CI_includes_0"
    verdict_prefix = "success" if ci_excludes_zero and delta > 0 else "complete"
    honest_verdict = (
        f"{verdict_prefix}_mmlu_pro_fewshot_verifier_vs_cheap_delta_{delta:+.3f}_"
        f"CI95_{_format_ci(ci95)}_{verdict_status}"
    )

    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "pool_path": POOL_RELATIVE_PATH,
        "zeroshot_result_path": ZEROSHOT_RESULT_RELATIVE_PATH,
        "pool_sha256": pool_sha256,
        "zeroshot_result_sha256": zero_sha256,
        "pool_precondition": _wrap(
            {"complete": True, "errors": []},
            "Records that the cached few-shot pool satisfied the no-regeneration precondition.",
        ),
        "pool_reused": _wrap(True, "True only when the complete cached pool is scored."),
        "candidate_generation_performed": _wrap(
            False, "This task must never regenerate the candidate pool."
        ),
        "n_questions": metrics["n_questions"],
        "n_candidates": metrics["n_candidates"],
        "n_correct_candidates": metrics["n_correct_candidates"],
        "n_unparseable_candidates": metrics["n_unparseable_candidates"],
        "oracle_at_k_ceiling": round(float(metrics["oracle_at_k"]), 4),
        "sc_vote_accuracy": round(float(metrics["sc_vote_accuracy"]), 4),
        "verifier_selection_accuracy": round(float(verifier_selection_acc), 4),
        "cheap_baseline_selection_accuracy": round(float(cheap_selection_acc), 4),
        "fewshot_oracle_at_k": _principled(
            "fewshot_oracle_at_k", round(float(metrics["oracle_at_k"]), 4)
        ),
        "fewshot_sc_vote_accuracy": _principled(
            "fewshot_sc_vote_accuracy", round(float(metrics["sc_vote_accuracy"]), 4)
        ),
        "fewshot_verifier_selection_accuracy": _principled(
            "fewshot_verifier_selection_accuracy", round(float(verifier_selection_acc), 4)
        ),
        "fewshot_cheap_baseline_selection_accuracy": _principled(
            "fewshot_cheap_baseline_selection_accuracy", round(float(cheap_selection_acc), 4)
        ),
        "verifier_vs_cheap_delta": _principled(
            "verifier_vs_cheap_delta", round(float(delta), 4)
        ),
        "verifier_vs_cheap_delta_ci95": _principled("verifier_vs_cheap_delta_ci95", ci95),
        "vs_zeroshot_pool_comparison": _principled(
            "vs_zeroshot_pool_comparison",
            _zeroshot_comparison(
                delta=delta,
                ci95=ci95,
                zero_artifact=zero_artifact,
                fewshot_headroom=float(metrics["oracle_at_k"] - metrics["sc_vote_accuracy"]),
            ),
        ),
        "still_underpowered": _principled("still_underpowered", still_underpowered),
        "verifier_is_oracle": _principled("verifier_is_oracle", False),
        "random_seed": _principled("random_seed", int(random_seed)),
        "reproducibility_checksum": _principled("reproducibility_checksum", ""),
        "honest_verdict": _principled("honest_verdict", honest_verdict),
        "zero_shot_pool_metrics": {
            "oracle_at_k_ceiling": zero_artifact.get("oracle_at_k_ceiling"),
            "sc_vote_accuracy": zero_artifact.get("sc_vote_accuracy"),
            "verifier_selection_accuracy": zero_artifact.get("verifier_selection_accuracy"),
            "cheap_baseline_selection_accuracy": zero_artifact.get(
                "cheap_baseline_selection_accuracy"
            ),
            "delta_verifier_vs_cheap_baseline": zero_artifact.get(
                "delta_verifier_vs_cheap_baseline"
            ),
            "delta_verifier_vs_cheap_baseline_ci95": zero_artifact.get(
                "delta_verifier_vs_cheap_baseline_ci95"
            ),
        },
        "model_specs": {
            "generator_pool": "unsloth/gemma-4-12B-it-GGUF 5-shot CoT cached candidates",
            "learned_verifier": "sentence-transformers/all-MiniLM-L6-v2 + LogisticRegression",
            "cheap_baseline": "8 non-learned text-statistical features + LogisticRegression",
        },
        "selection_protocol": (
            "Leave-one-question-out CV; for each held-out question, select the highest-scored "
            "candidate among K=6 and evaluate that candidate against the MMLU-Pro gold letter."
        ),
        "bootstrap": {"n_boot": int(n_boot), "seed": int(random_seed), "unit": "question"},
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    artifact["reproducibility_checksum"]["value"] = payload_checksum(artifact)
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError(f"invalid Exp 5163 artifact: {errors}")
    return artifact


def _is_principle_wrapped(artifact: Mapping[str, Any], field: str) -> bool:
    value = artifact.get(field)
    return (
        isinstance(value, Mapping)
        and "value" in value
        and value.get("principle") == FIELD_PRINCIPLES[field]
    )


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = [field for field in REQUIRED_PRINCIPLED_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    for field in REQUIRED_PRINCIPLED_FIELDS:
        if field in artifact and not _is_principle_wrapped(artifact, field):
            errors.append(f"{field} must be principle-wrapped")
    verdict = artifact.get("honest_verdict", {})
    verdict_value = verdict.get("value") if isinstance(verdict, Mapping) else None
    if not isinstance(verdict_value, str) or not verdict_value.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must use a terminal prefix")
    if isinstance(verdict_value, str) and verdict_value != "blocked_fewshot_pool_incomplete":
        if "CI_includes_0" not in verdict_value and "CI_excludes_0" not in verdict_value:
            errors.append("honest_verdict must state whether the CI excludes 0")
    verifier = artifact.get("verifier_is_oracle", {})
    if not isinstance(verifier, Mapping) or verifier.get("value") is not False:
        errors.append("verifier_is_oracle must be false")
    ci = artifact.get("verifier_vs_cheap_delta_ci95", {})
    ci_value = ci.get("value") if isinstance(ci, Mapping) else None
    if ci_value is not None:
        if (
            not isinstance(ci_value, Sequence)
            or isinstance(ci_value, (str, bytes))
            or len(ci_value) != 2
        ):
            errors.append("verifier_vs_cheap_delta_ci95 must be a two-value CI95")
    seed = artifact.get("random_seed", {})
    if not isinstance(seed, Mapping) or not isinstance(seed.get("value"), int):
        errors.append("random_seed must be an int")
    checksum = artifact.get("reproducibility_checksum", {})
    if not isinstance(checksum, Mapping) or not str(checksum.get("value", "")).startswith(
        "sha256:"
    ):
        errors.append("reproducibility_checksum must be a sha256 string")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must match REQ-VERIFY-5163")
    return errors


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError(f"invalid Exp 5163 artifact: {errors}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(
    *,
    root: Path | str = REPO_ROOT,
    pool_path: Path | None = None,
    zero_result_path: Path | None = None,
    result_path: Path | None = None,
    expected_questions: int = EXPECTED_N_QUESTIONS,
    k_samples: int = K_SAMPLES,
    score_candidates_fn: ScoreCandidatesFn | None = None,
    option_counts_loader: OptionCountsLoader = load_option_counts_from_mmlu_pro,
    n_boot: int = DEFAULT_N_BOOT,
) -> JsonDict:
    root_path = Path(root)
    pool = pool_path or (root_path / POOL_RELATIVE_PATH)
    zero_path = zero_result_path or (root_path / ZEROSHOT_RESULT_RELATIVE_PATH)
    result = result_path or (root_path / RESULT_RELATIVE_PATH)
    try:
        rows = load_candidate_pool(
            pool, expected_questions=expected_questions, k_samples=k_samples
        )
    except PoolIncomplete as exc:
        artifact = build_blocked_artifact(errors=exc.errors, pool_path=pool)
        write_artifact(result, artifact)
        return artifact

    zero_artifact = _read_json(zero_path)
    option_counts = option_counts_loader(rows, root_path)
    scorer = score_candidates_fn or score_candidates
    verifier_scores, cheap_scores = scorer(rows, option_counts)
    artifact = build_complete_artifact(
        rows=rows,
        verifier_scores=verifier_scores,
        cheap_scores=cheap_scores,
        zero_artifact=zero_artifact,
        pool_sha256=_sha256_file(pool),
        zero_sha256=_sha256_file(zero_path),
        random_seed=RANDOM_SEED,
        n_boot=n_boot,
    )
    write_artifact(result, artifact)
    return artifact


def main() -> int:  # pragma: no cover - CLI path
    artifact = run()
    print(json.dumps({"honest_verdict": artifact["honest_verdict"]["value"]}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
