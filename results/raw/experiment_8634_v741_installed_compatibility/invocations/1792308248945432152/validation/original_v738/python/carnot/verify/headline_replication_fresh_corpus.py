"""Exp 3719 fresh-corpus replication of the frozen FoVer discriminator.

The module scores cached, non-FoVer GSM8K process-integrity rows with the same
production score formula used by the frozen FoVer headline path. It does not
run live LLM inference and does not mutate the frozen FoVer artifact.

Spec: REQ-VERIFY-3719, SCENARIO-VERIFY-3719.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]
ClockFn = Callable[[], float]
TextScorer = Callable[[list[str]], Mapping[str, Sequence[float]]]
MemoryScorer = Callable[["FreshCorpusRow"], float]

OUTPUT_REL_PATH = Path("results/experiment_3719_headline_replication_fresh_corpus.json")
EXP235_REL_PATH = Path("results/experiment_235_results.json")
EXP248_REL_PATH = Path("results/experiment_248_results.json")
EXP2850_REL_PATH = Path("results/experiment_2850_fover_dual_condition_integrity_v4.json")
PROCESS_CORPUS_REL_PATH = Path("data/research/process_integrity_corpus_248.jsonl")
FOVER_DERIVED_PRM_REL_PATH = Path("data/step_level_prm_training.jsonl")

FROZEN_FOVER_AUROC = 0.9131
DEFAULT_RANDOM_SEED = 3719
DEFAULT_RANDOM_SEEDS = (42, 137, 271, 314, 1729)
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
FRESH_CORPUS_SOURCE = (
    "Exp 248 GSM8K process-integrity reasoning rows joined to Exp 235 "
    "checked-in candidate traces; distinct from FoVer"
)
GENERALIZES_VERDICT = "complete: headline_discrimination_generalizes_to_fresh_corpus_g1_strengthened"
FOVER_SPECIFIC_VERDICT = (
    "complete: headline_discrimination_is_fover_specific_generalization_narrowed_honest"
)
BLOCKED_VERDICT = "complete: blocked_no_fresh_step_error_corpus"
VERIFIER_NAMES = (
    "fr11_session_memory",
    "tier0r_curry_howard",
    "tier0s_arithmetic_gap",
    "tier0u_logical_consistency",
)
ERROR_PROCESS_LABELS = {
    "repair_fixed_outcome_only",
    "right_answer_wrong_process",
    "unsupported_step",
    "wrong_answer_partially_sound_process",
}
CLEAN_PROCESS_LABELS = {"clean", "repair_fixed_process_and_outcome"}
FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": (
        "verifier_ensemble_against_cached_candidates because this scores cached candidates only."
    ),
    "fresh_corpus_source": "The distinct non-FoVer checked-in process corpus provenance.",
    "fresh_corpus_auroc": "AUROC for the FoVer production score formula on the fresh corpus.",
    "fresh_corpus_auroc_ci": "Small-n CI95 over the five seeded balanced subsets.",
    "frozen_fover_auroc": "0.9131 frozen FoVer headline comparison; not replaced.",
    "generalizes_beyond_fover": "Bare bool: true iff the fresh-corpus CI brackets 0.9131.",
    "n_seeds": "Replication count.",
    "n_examples": "Per-seed balanced subset size used for fresh-corpus AUROC.",
    "frozen_headline_unchanged_assert": "The publication headline still rounds to 0.9131.",
    "adversarial_verify_clean": "True iff the terminal artifact has no adversarial flags.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection across source hashes, seeds, and score rows.",
    "duration_s": "Measured runtime.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


@dataclass(frozen=True)
class FreshCorpusRow:
    """One cached candidate row with an error label."""

    corpus_id: str
    text: str
    label: int
    process_label: str
    case_id: str = ""
    iteration: int = 0

    def checksum_payload(self) -> JsonDict:
        return {
            "corpus_id": self.corpus_id,
            "label": int(self.label),
            "process_label": self.process_label,
            "text_sha256": _sha256_text(self.text),
        }


@dataclass(frozen=True)
class FreshCorpus:
    """Assembled non-FoVer corpus or a blocked reason."""

    rows: Sequence[FreshCorpusRow]
    source: str | None
    source_paths: Sequence[str]
    source_sha256: str | None
    disqualified_sources: Sequence[JsonDict]
    blocked_reason: str | None = None

    @property
    def balance(self) -> dict[str, int]:
        correct = sum(1 for row in self.rows if int(row.label) == 0)
        incorrect = sum(1 for row in self.rows if int(row.label) == 1)
        return {"correct": correct, "incorrect": incorrect}


@dataclass(frozen=True)
class SeedScoreResult:
    """AUROC statistics for one seeded balanced subset."""

    seed: int
    auroc: float
    n_examples: int
    subset_sha256: str
    per_verifier_auroc: Mapping[str, float]

    def as_dict(self) -> JsonDict:
        return {
            "seed": int(self.seed),
            "fresh_corpus_auroc": _round(self.auroc),
            "n_examples": int(self.n_examples),
            "subset_sha256": self.subset_sha256,
            "per_verifier_auroc": {
                name: _round(self.per_verifier_auroc[name]) for name in sorted(self.per_verifier_auroc)
            },
        }


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime config for Exp 3719."""

    repo_root: Path = Path(__file__).resolve().parents[3]
    output_path: Path | None = None
    random_seeds: Sequence[int] = DEFAULT_RANDOM_SEEDS
    random_seed: int = DEFAULT_RANDOM_SEED
    n_examples: int | None = None
    started_at: float | None = None
    clock: ClockFn = time.perf_counter

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def resolved_output_path(self) -> Path:
        if self.output_path is not None:
            return self.output_path
        return self.repo_root / OUTPUT_REL_PATH


def assemble_fresh_corpus(repo_root: Path) -> FreshCorpus:
    """Assemble a distinct GSM8K process-error corpus from checked-in traces."""
    root = Path(repo_root)
    disqualified = _disqualified_sources(root)
    exp235_path = root / EXP235_REL_PATH
    process_path = root / PROCESS_CORPUS_REL_PATH
    if not exp235_path.is_file() or not process_path.is_file():
        return FreshCorpus(
            rows=[],
            source=None,
            source_paths=[],
            source_sha256=None,
            disqualified_sources=disqualified,
            blocked_reason="no_distinct_process_corpus",
        )

    text_index = _exp235_verify_repair_text_index(_read_json(exp235_path))
    rows: list[FreshCorpusRow] = []
    for raw in _read_jsonl(process_path):
        if raw.get("domain") != "reasoning":
            continue
        label = _process_label_to_int(raw.get("process_label"))
        if label is None:
            continue
        key = (
            str(raw.get("benchmark", "")),
            str(raw.get("model", "")),
            str(raw.get("case_id", "")),
            int(raw.get("iteration", 0)),
        )
        text = text_index.get(key, "")
        if not text.strip():
            continue
        rows.append(
            FreshCorpusRow(
                corpus_id=str(raw.get("corpus_id", "")),
                text=text,
                label=label,
                process_label=str(raw.get("process_label", "")),
                case_id=str(raw.get("case_id", "")),
                iteration=int(raw.get("iteration", 0)),
            )
        )

    balance = FreshCorpus(rows, FRESH_CORPUS_SOURCE, (), "", disqualified).balance
    if balance["correct"] == 0 or balance["incorrect"] == 0:
        return FreshCorpus(
            rows=[],
            source=None,
            source_paths=[],
            source_sha256=None,
            disqualified_sources=disqualified,
            blocked_reason="no_distinct_process_corpus",
        )
    return FreshCorpus(
        rows=rows,
        source=FRESH_CORPUS_SOURCE,
        source_paths=(EXP235_REL_PATH.as_posix(), EXP248_REL_PATH.as_posix(), PROCESS_CORPUS_REL_PATH.as_posix()),
        source_sha256=_combined_file_sha256(root, (EXP235_REL_PATH, EXP248_REL_PATH, PROCESS_CORPUS_REL_PATH)),
        disqualified_sources=disqualified,
    )


def build_artifact(
    config: ExperimentConfig | None = None,
    *,
    text_scorer: TextScorer | None = None,
    memory_scorer: MemoryScorer | None = None,
    adversarial_verify_clean: bool = False,
) -> JsonDict:
    """Build the Exp 3719 artifact without running adversarial verification."""
    active = config or ExperimentConfig()
    started = active.start_time()
    corpus = assemble_fresh_corpus(active.repo_root)
    if corpus.blocked_reason is not None:
        return _blocked_artifact(corpus, started, active.clock(), active.random_seed)

    n_examples = active.n_examples or _balanced_n_examples(corpus.rows)
    try:
        scorer = text_scorer or _score_text_verifiers
        memory = memory_scorer or _fresh_memory_scorer(active.repo_root)
        seed_results = [
            score_fresh_seed(
                corpus.rows,
                seed=int(seed),
                n_examples=n_examples,
                text_scorer=scorer,
                memory_scorer=memory,
            )
            for seed in active.random_seeds
        ]
        frozen_ok = frozen_headline_still_unchanged(active.repo_root)
    except Exception as exc:  # noqa: BLE001
        return _blocked_artifact(
            corpus,
            started,
            active.clock(),
            active.random_seed,
            detail=f"{type(exc).__name__}: {exc}",
        )

    return build_artifact_from_seed_results(
        corpus=corpus,
        seed_results=seed_results,
        started_at=started,
        now=active.clock(),
        adversarial_verify_clean=adversarial_verify_clean,
        frozen_headline_unchanged_assert=frozen_ok,
        random_seed=active.random_seed,
    )


def score_fresh_seed(
    rows: Sequence[FreshCorpusRow],
    *,
    seed: int,
    n_examples: int,
    text_scorer: TextScorer,
    memory_scorer: MemoryScorer,
) -> SeedScoreResult:
    """Score one seeded balanced subset with the frozen production formula."""
    subset = _select_balanced_rows(rows, seed=seed, n_examples=n_examples)
    labels = [int(row.label) for row in subset]
    text_scores = text_scorer([row.text for row in subset])
    _assert_verifier_columns(text_scores, len(subset))
    memory_scores = [float(memory_scorer(row)) for row in subset]
    scores_by_verifier = {
        "fr11_session_memory": memory_scores,
        "tier0r_curry_howard": [float(v) for v in text_scores["tier0r_curry_howard"]],
        "tier0s_arithmetic_gap": [float(v) for v in text_scores["tier0s_arithmetic_gap"]],
        "tier0u_logical_consistency": [
            float(v) for v in text_scores["tier0u_logical_consistency"]
        ],
    }
    ensemble_scores = [
        memory + 0.9 * tier0r + 0.1 * tier0u
        for memory, tier0r, tier0u in zip(
            scores_by_verifier["fr11_session_memory"],
            scores_by_verifier["tier0r_curry_howard"],
            scores_by_verifier["tier0u_logical_consistency"],
            strict=True,
        )
    ]
    per_verifier = {
        name: _compute_auroc(labels, scores) for name, scores in scores_by_verifier.items()
    }
    return SeedScoreResult(
        seed=int(seed),
        auroc=_compute_auroc(labels, ensemble_scores),
        n_examples=len(subset),
        subset_sha256=_subset_sha256(subset),
        per_verifier_auroc=per_verifier,
    )


def build_artifact_from_seed_results(
    *,
    corpus: FreshCorpus,
    seed_results: Sequence[SeedScoreResult],
    started_at: float,
    now: float,
    adversarial_verify_clean: bool,
    frozen_headline_unchanged_assert: bool,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> JsonDict:
    """Build a terminal artifact from scored seed rows."""
    if not seed_results:
        raise ValueError("at least one seed result is required")
    aurocs = [float(result.auroc) for result in seed_results]
    ci = _seed_t_ci95(aurocs)
    generalizes = bool(ci["low"] <= FROZEN_FOVER_AUROC <= ci["high"])
    verdict = GENERALIZES_VERDICT if generalizes else FOVER_SPECIFIC_VERDICT
    artifact = _base_artifact(
        corpus=corpus,
        started_at=started_at,
        now=now,
        random_seed=random_seed,
        adversarial_verify_clean=adversarial_verify_clean,
        frozen_headline_unchanged_assert=frozen_headline_unchanged_assert,
    )
    artifact.update(
        {
            "honest_verdict": verdict,
            "fresh_corpus_auroc": _round(ci["mean"]),
            "fresh_corpus_auroc_ci": {
                "mean": _round(ci["mean"]),
                "low": _round(ci["low"]),
                "high": _round(ci["high"]),
            },
            "generalizes_beyond_fover": generalizes,
            "n_seeds": len(seed_results),
            "n_examples": int(seed_results[0].n_examples),
            "per_seed_results": [result.as_dict() for result in seed_results],
            "per_verifier_auroc_mean": _per_verifier_mean(seed_results),
            "reproducibility_checksum": reproducibility_checksum(corpus, seed_results, random_seed),
            "acceptance_gate_passed": bool(
                adversarial_verify_clean and frozen_headline_unchanged_assert
            ),
            "methodology_note": (
                "Seeded balanced subsets mirror the frozen FoVer production score formula: "
                "fr11_session_memory + 0.9*tier0r_curry_howard + "
                "0.1*tier0u_logical_consistency. tier0s_arithmetic_gap is scored and "
                "reported as one of the four FoVer-scoring verifier columns but has zero "
                "weight in the frozen production AUROC formula."
            ),
        }
    )
    _validate_artifact(artifact)
    return artifact


def write_artifact(
    config: ExperimentConfig | None = None,
    *,
    adversarial_verify_runner: Callable[[Path], JsonDict] | None = None,
) -> JsonDict:
    """Build, write, adversarial-check, and rewrite the terminal artifact."""
    active = config or ExperimentConfig()
    path = active.resolved_output_path()
    payload = build_artifact(active, adversarial_verify_clean=False)
    _write_json(path, payload)
    runner = run_adversarial_verify if adversarial_verify_runner is None else adversarial_verify_runner
    report = runner(path)
    payload["adversarial_verify_clean"] = _report_clean(report)
    payload["adversarial_verify_report"] = _compact_adversarial_report(report)
    payload["acceptance_gate_passed"] = bool(
        payload.get("fresh_corpus_auroc") is not None
        and payload.get("fresh_corpus_auroc_ci") is not None
        and payload.get("fresh_corpus_source") is not None
        and payload.get("adversarial_verify_clean") is True
        and payload.get("frozen_headline_unchanged_assert") is True
    )
    _write_json(path, payload)
    return payload


def run_adversarial_verify(path: Path) -> JsonDict:
    """Run the repo adversarial verifier in JSON mode."""
    script = Path(__file__).resolve().parents[3] / "scripts" / "adversarial_verify.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--json", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {
            "flag_count": 1,
            "flags": [
                {
                    "kind": "ADVERSARIAL_VERIFY_ERROR",
                    "severity": "warn",
                    "detail": (proc.stderr or proc.stdout or "invalid JSON").strip(),
                }
            ],
            "returncode": proc.returncode,
        }
    reports = payload.get("reports") or []
    report = dict(reports[0]) if reports else {"flag_count": 0, "flags": []}
    report["returncode"] = proc.returncode
    return report


def frozen_headline_still_unchanged(repo_root: Path) -> bool:
    """Return true when Exp 2850 still rounds to the frozen 0.9131 headline."""
    path = Path(repo_root) / EXP2850_REL_PATH
    if not path.is_file():
        return False
    payload = _read_json(path)
    value = payload.get("condition_a_production_auroc_mean")
    return value is not None and round(float(value), 4) == FROZEN_FOVER_AUROC


def _base_artifact(
    *,
    corpus: FreshCorpus,
    started_at: float,
    now: float,
    random_seed: int,
    adversarial_verify_clean: bool,
    frozen_headline_unchanged_assert: bool,
) -> JsonDict:
    return {
        "artifact": "experiment_3719_headline_replication_fresh_corpus",
        "schema": "carnot.headline_replication_fresh_corpus.v1",
        "honest_verdict": BLOCKED_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "fresh_corpus_source": corpus.source,
        "fresh_corpus_source_paths": list(corpus.source_paths),
        "fresh_corpus_source_sha256": corpus.source_sha256,
        "fresh_corpus_total_rows": len(corpus.rows),
        "fresh_corpus_balance": corpus.balance,
        "fresh_corpus_auroc": None,
        "fresh_corpus_auroc_ci": None,
        "frozen_fover_auroc": FROZEN_FOVER_AUROC,
        "generalizes_beyond_fover": False,
        "n_seeds": 0,
        "n_examples": 0,
        "frozen_headline_unchanged_assert": bool(frozen_headline_unchanged_assert),
        "adversarial_verify_clean": bool(adversarial_verify_clean),
        "random_seed": int(random_seed),
        "random_seeds_used": list(DEFAULT_RANDOM_SEEDS),
        "reproducibility_checksum": "",
        "duration_s": _round(max(0.0, float(now) - float(started_at))),
        "verifier_names": list(VERIFIER_NAMES),
        "verifier_ensemble_formula": {
            "fr11_session_memory": 1.0,
            "tier0r_curry_howard": 0.9,
            "tier0s_arithmetic_gap": 0.0,
            "tier0u_logical_consistency": 0.1,
        },
        "disqualified_sources": list(corpus.disqualified_sources),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate_passed": False,
    }


def _blocked_artifact(
    corpus: FreshCorpus,
    started_at: float,
    now: float,
    random_seed: int,
    detail: str | None = None,
) -> JsonDict:
    artifact = _base_artifact(
        corpus=corpus,
        started_at=started_at,
        now=now,
        random_seed=random_seed,
        adversarial_verify_clean=False,
        frozen_headline_unchanged_assert=False,
    )
    artifact["blocked_reason"] = corpus.blocked_reason or "scoring_blocked"
    if detail is not None:
        artifact["blocked_detail"] = detail
    artifact["reproducibility_checksum"] = reproducibility_checksum(corpus, [], random_seed)
    return artifact


def _exp235_verify_repair_text_index(payload: Mapping[str, Any]) -> dict[tuple[str, str, str, int], str]:
    index: dict[tuple[str, str, str, int], str] = {}
    for run in payload.get("paired_runs") or []:
        if not isinstance(run, Mapping) or run.get("mode") != "verify_repair":
            continue
        benchmark = str(run.get("benchmark", ""))
        model = str(run.get("model_name", ""))
        for case in run.get("cases") or []:
            if not isinstance(case, Mapping):
                continue
            case_id = str(case.get("case_id", ""))
            for hist in case.get("history") or []:
                if not isinstance(hist, Mapping):
                    continue
                response = str(hist.get("response") or "")
                index[(benchmark, model, case_id, int(hist.get("iteration", 0)))] = response
    return index


def _process_label_to_int(label: Any) -> int | None:
    text = str(label)
    if text in ERROR_PROCESS_LABELS:
        return 1
    if text in CLEAN_PROCESS_LABELS:
        return 0
    return None


def _disqualified_sources(root: Path) -> list[JsonDict]:
    path = root / FOVER_DERIVED_PRM_REL_PATH
    if not path.is_file():
        return []
    return [
        {
            "path": FOVER_DERIVED_PRM_REL_PATH.as_posix(),
            "reason": "fover_derived",
            "detail": "Exp 1084 step-level PRM training rows were generated from FoVer pairs.",
            "sha256": _sha256_file(path),
        }
    ]


def _fresh_memory_scorer(repo_root: Path) -> MemoryScorer:
    from carnot.eval.fover_memory_leakage_v3 import _fr11_memory_score, _load_fr11_memory_index

    memory_index = _load_fr11_memory_index(Path(repo_root))

    def score(row: FreshCorpusRow) -> float:
        return float(
            _fr11_memory_score(
                {"question_id": row.case_id or row.corpus_id, "step_text": row.text},
                memory_index,
            )
        )

    return score


def _score_text_verifiers(texts: list[str]) -> Mapping[str, Sequence[float]]:
    from carnot.eval.fover_memory_leakage_v3 import _score_text_verifiers as score_text

    return score_text(texts)


def _assert_verifier_columns(scores: Mapping[str, Sequence[float]], n_rows: int) -> None:
    missing = set(VERIFIER_NAMES[1:]) - set(scores)
    if missing:
        raise ValueError(f"missing verifier score columns: {sorted(missing)}")
    for name in VERIFIER_NAMES[1:]:
        if len(scores[name]) != n_rows:
            raise ValueError(f"verifier {name} returned {len(scores[name])} scores for {n_rows} rows")


def _balanced_n_examples(rows: Sequence[FreshCorpusRow]) -> int:
    balance = FreshCorpus(rows, None, (), None, ()).balance
    return 2 * min(balance["correct"], balance["incorrect"])


def _select_balanced_rows(
    rows: Sequence[FreshCorpusRow],
    *,
    seed: int,
    n_examples: int,
) -> list[FreshCorpusRow]:
    positives = [row for row in rows if int(row.label) == 1]
    negatives = [row for row in rows if int(row.label) == 0]
    n_pos = n_examples // 2
    n_neg = n_examples - n_pos
    if len(positives) < n_pos or len(negatives) < n_neg:
        raise ValueError(
            f"fresh corpus lacks class balance for n={n_examples}: "
            f"positives={len(positives)}, negatives={len(negatives)}"
        )
    rng = random.Random(seed)
    subset = [*rng.sample(positives, n_pos), *rng.sample(negatives, n_neg)]
    rng.shuffle(subset)
    return subset


def _compute_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    n_pos = sum(1 for label in labels if int(label) == 1)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC requires both positive and negative labels")
    ranked = sorted(enumerate(scores), key=lambda item: float(item[1]))
    ranks = [0.0] * len(scores)
    cursor = 0
    while cursor < len(ranked):
        end = cursor + 1
        while end < len(ranked) and float(ranked[end][1]) == float(ranked[cursor][1]):
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for offset in range(cursor, end):
            ranks[ranked[offset][0]] = rank
        cursor = end
    pos_rank_sum = sum(rank for rank, label in zip(ranks, labels, strict=True) if int(label) == 1)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _seed_t_ci95(values: Sequence[float]) -> dict[str, float]:
    numeric = [float(value) for value in values]
    if not numeric:
        raise ValueError("at least one value is required")
    mean = sum(numeric) / len(numeric)
    if len(numeric) < 2:
        return {"mean": mean, "low": mean, "high": mean}
    t_crit_by_n = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
    t_crit = t_crit_by_n.get(len(numeric), 1.96)
    std = math.sqrt(sum((value - mean) ** 2 for value in numeric) / (len(numeric) - 1))
    half = t_crit * std / math.sqrt(len(numeric))
    return {"mean": mean, "low": mean - half, "high": mean + half}


def _per_verifier_mean(seed_results: Sequence[SeedScoreResult]) -> JsonDict:
    values: dict[str, list[float]] = {name: [] for name in VERIFIER_NAMES}
    for result in seed_results:
        for name, value in result.per_verifier_auroc.items():
            values.setdefault(name, []).append(float(value))
    return {name: _round(sum(vals) / len(vals)) for name, vals in values.items() if vals}


def reproducibility_checksum(
    corpus: FreshCorpus,
    seed_results: Sequence[SeedScoreResult],
    random_seed: int,
) -> str:
    payload = {
        "source": corpus.source,
        "source_paths": list(corpus.source_paths),
        "source_sha256": corpus.source_sha256,
        "rows": [row.checksum_payload() for row in corpus.rows],
        "random_seed": int(random_seed),
        "seed_results": [result.as_dict() for result in seed_results],
    }
    return _sha256_json(payload)


def _subset_sha256(rows: Sequence[FreshCorpusRow]) -> str:
    return _sha256_json([row.checksum_payload() for row in rows])


def _validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    if not isinstance(artifact.get("generalizes_beyond_fover"), bool):
        raise ValueError("generalizes_beyond_fover must be a bare bool")
    if artifact.get("fresh_corpus_auroc") == artifact.get("frozen_fover_auroc"):
        raise ValueError("fresh_corpus_auroc must not be copied from frozen_fover_auroc")


def _report_clean(report: Mapping[str, Any]) -> bool:
    flags = [dict(flag) for flag in report.get("flags") or []]
    return not [flag for flag in flags if str(flag.get("severity", "")).lower() in {"warn", "critical"}]


def _compact_adversarial_report(report: Mapping[str, Any]) -> JsonDict:
    flags = [dict(flag) for flag in report.get("flags") or []]
    return {
        "flag_count": int(report.get("flag_count") or len(flags)),
        "returncode": report.get("returncode"),
        "flags": flags,
    }


def _read_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _read_jsonl(path: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _combined_file_sha256(root: Path, paths: Sequence[Path]) -> str:
    payload = {
        path.as_posix(): _sha256_file(root / path) if (root / path).is_file() else None
        for path in paths
    }
    return _sha256_json(payload)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _round(value: float) -> float:
    return round(float(value), 6)
