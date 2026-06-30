"""Shared Phase D moat benchmark harness.

Spec refs: REQ-KONA-5002, SCENARIO-KONA-5002-SMOKE,
SCENARIO-KONA-5002-ORACLE-DISTINCT, SCENARIO-KONA-5002-BLOCKED.

The module is intentionally small and data-oriented. D arms provide a scorer
`f(candidate) -> energy` where lower energy is better; this harness owns the
candidate-pool metrics, tuned self-consistency baseline, paired uncertainty, and
oracle-distinct scorer guard.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
import ast
import json
import math
from pathlib import Path
import random
import re
from typing import Any


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
Scorer = Callable[[Mapping[str, Any]], float]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MUSR_CHECKPOINT_DIR = (
    REPO_ROOT / "results" / "distributional_energy_verifier_musr_checkpoints"
)
MUSR_CORPUS_NAME = "MuSR/murder_mysteries"
HEADROOM_THRESHOLD = 0.10
DEFAULT_RANDOM_SEED = 20260630
FORBIDDEN_SCORER_KEYS = frozenset({"gold", "answer_index", "answer_choice", "model_id"})
ABSTENTION_DEGENERACY_THRESHOLD = 0.50


class CorpusUnavailableError(RuntimeError):
    """Raised when a requested corpus is not available from the local cache."""


class CandidateCacheError(RuntimeError):
    """Raised when a required cached candidate pool is missing or malformed."""


class GenerationUnavailableError(RuntimeError):
    """Raised when fresh generation is requested without a generator backend."""


class OracleDistinctnessError(ValueError):
    """Raised when a scorer tries to read answer-key or model-identity fields."""


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for D-arm fresh generation that needs token logprobs."""

    k: int = 8
    model: str = "gemma-4-12B-it-GGUF"
    gpu: int = 0
    temperature: float = 0.7
    max_tokens: int = 512
    require_logprobs: bool = True


class GuardedCandidate(Mapping[str, Any]):
    """Mapping view that denies oracle and model-identity keys to scorers."""

    def __init__(self, candidate: JsonMap):
        self._candidate = candidate

    def _check_key(self, key: object) -> None:
        if str(key) in FORBIDDEN_SCORER_KEYS:
            raise OracleDistinctnessError(
                f"oracle-distinct scorer attempted to read forbidden key {key!r}"
            )

    def __getitem__(self, key: str) -> Any:
        self._check_key(key)
        return self._candidate[key]

    def __iter__(self) -> Iterator[str]:
        for key in self._candidate:
            if key not in FORBIDDEN_SCORER_KEYS:
                yield str(key)

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __contains__(self, key: object) -> bool:
        self._check_key(key)
        return key in self._candidate

    def get(self, key: str, default: Any = None) -> Any:
        self._check_key(key)
        return self._candidate.get(key, default)


def _json_load(path: Path) -> JsonDict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _choices(row: JsonMap) -> list[str]:
    choices = row.get("choices") or row.get("options")
    if isinstance(choices, str):
        try:
            choices = ast.literal_eval(choices)
        except (SyntaxError, ValueError):
            choices = [choice.strip() for choice in choices.strip("[]").split(",")]
    if isinstance(choices, Sequence) and not isinstance(choices, (bytes, str)):
        return [str(choice).strip().strip("'\"") for choice in choices]
    return []


def _normalize_gold_from_letter(gold: Any, choices: Sequence[str]) -> str:
    text = str(gold or "").strip()
    if len(text) == 1 and text.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        index = ord(text.upper()) - ord("A")
        if 0 <= index < len(choices):
            return str(choices[index])
    return text


def normalize_corpus_row(row: JsonMap, *, corpus: str, index: int) -> JsonDict:
    choices = _choices(row)
    gold = row.get("gold", row.get("answer_choice", row.get("answer", row.get("target", ""))))
    context = row.get("context", row.get("narrative", row.get("story", row.get("problem", ""))))
    return {
        "row_id": str(row.get("row_id", row.get("id", f"{corpus}:{index}"))),
        "corpus": corpus,
        "question": str(row.get("question", row.get("prompt", ""))).strip(),
        "context": str(context or "").strip(),
        "choices": choices,
        "gold": _normalize_gold_from_letter(gold, choices),
    }


def _load_dataset_local(dataset_name: str, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
    try:
        from datasets import DownloadConfig, load_dataset
    except ImportError as exc:  # pragma: no cover - depends on optional package install
        raise CorpusUnavailableError("datasets package is not importable") from exc
    try:
        download_config = DownloadConfig(local_files_only=True)
        return load_dataset(dataset_name, *args, download_config=download_config, **kwargs)
    except Exception as exc:
        raise CorpusUnavailableError(f"{dataset_name} is not available in the local cache") from exc


def load_musr_murder_mysteries(*, limit: int | None = None) -> list[JsonDict]:
    """Load locally cached MuSR/murder_mysteries rows for evaluation only."""

    dataset = _load_dataset_local("TAUR-Lab/MuSR")
    try:
        split = dataset["murder_mysteries"]
    except (KeyError, TypeError) as exc:
        raise CorpusUnavailableError("TAUR-Lab/MuSR lacks murder_mysteries split") from exc
    rows: list[JsonDict] = []
    for index, raw in enumerate(split):
        rows.append(normalize_corpus_row(raw, corpus=MUSR_CORPUS_NAME, index=index))
        if limit is not None and len(rows) >= limit:
            break
    if not rows:
        raise CorpusUnavailableError("MuSR/murder_mysteries returned no rows")
    return rows


def _take_optional_rows(dataset: Any, *, corpus: str, limit: int | None) -> list[JsonDict]:
    if isinstance(dataset, Mapping):
        split = dataset.get("test") or dataset.get("validation") or dataset.get("train")
    else:
        split = dataset
    if split is None:
        return []
    rows: list[JsonDict] = []
    for index, raw in enumerate(split):
        rows.append(normalize_corpus_row(raw, corpus=corpus, index=index))
        if limit is not None and len(rows) >= limit:
            break
    return rows


def load_gpqa_cached(*, limit: int | None = None) -> list[JsonDict]:
    for subset in ("gpqa_diamond", "gpqa_main", None):
        try:
            args = (subset,) if subset else ()
            dataset = _load_dataset_local("Idavidrein/gpqa", *args)
            rows = _take_optional_rows(dataset, corpus="GPQA", limit=limit)
            if rows:
                return rows
        except CorpusUnavailableError:
            continue
    raise CorpusUnavailableError("GPQA is not available in the local cache")


def load_mmlu_pro_hard_cached(*, limit: int | None = None) -> list[JsonDict]:
    dataset = _load_dataset_local("TIGER-Lab/MMLU-Pro")
    rows = _take_optional_rows(dataset, corpus="MMLU-Pro-hard", limit=None)
    hard_rows = [
        row
        for row in rows
        if "hard" in str(row.get("difficulty", row.get("level", "hard"))).lower()
    ]
    selected = hard_rows or rows
    return selected[:limit] if limit is not None else selected


def load_math_500_hard_cached(*, limit: int | None = None) -> list[JsonDict]:
    dataset = _load_dataset_local("HuggingFaceH4/MATH-500")
    rows = _take_optional_rows(dataset, corpus="MATH-500-hard", limit=None)
    hard_rows = [
        row
        for row in rows
        if str(row.get("level", row.get("difficulty", "hard"))).lower() in {"4", "5", "hard"}
    ]
    selected = hard_rows or rows
    return selected[:limit] if limit is not None else selected


def discover_available_corpora(*, limit: int = 1) -> list[str]:
    """Return corpus names that load from local cache without requiring downloads."""

    available: list[str] = []
    loaders: list[tuple[str, Callable[..., list[JsonDict]]]] = [
        (MUSR_CORPUS_NAME, load_musr_murder_mysteries),
        ("GPQA", load_gpqa_cached),
        ("MMLU-Pro-hard", load_mmlu_pro_hard_cached),
        ("MATH-500-hard", load_math_500_hard_cached),
    ]
    for name, loader in loaders:
        try:
            if loader(limit=limit):
                available.append(name)
        except CorpusUnavailableError:
            continue
    return available


def attach_musr_cached_candidates(
    corpus_rows: Sequence[JsonMap],
    *,
    checkpoint_dir: Path = DEFAULT_MUSR_CHECKPOINT_DIR,
    limit: int | None = None,
) -> list[JsonDict]:
    """Attach cached MuSR checkpoint answer pools without regenerating candidates."""

    if not checkpoint_dir.exists():
        raise CandidateCacheError(f"candidate cache missing: {checkpoint_dir}")
    selected_rows = list(corpus_rows)[:limit] if limit is not None else list(corpus_rows)
    rows: list[JsonDict] = []
    for row_index, row in enumerate(selected_rows):
        checkpoint_path = checkpoint_dir / f"q{row_index:04d}.json"
        if not checkpoint_path.exists():
            raise CandidateCacheError(f"candidate checkpoint missing: {checkpoint_path}")
        checkpoint = _json_load(checkpoint_path)
        answers = checkpoint.get("answers")
        if not isinstance(answers, list):
            raise CandidateCacheError(f"candidate checkpoint lacks answers: {checkpoint_path}")
        energy_answer = str(
            checkpoint.get("energy_pure_answer") or checkpoint.get("energy_answer") or ""
        )
        candidates: list[JsonDict] = []
        for candidate_index, answer in enumerate(answers):
            if answer is None or str(answer).strip() == "":
                continue
            answer_text = str(answer)
            energy_selected = answer_text == energy_answer
            candidates.append(
                {
                    "candidate_id": f"{row.get('row_id', row_index)}/cached-{candidate_index}",
                    "answer": answer_text,
                    "cache_index": candidate_index,
                    "temperature": checkpoint.get("temperature", "cached"),
                    "cached_energy_selected": energy_selected,
                    "cached_energy_answer": energy_answer,
                    "trivial_energy": 0.0 if energy_selected else 1.0 + candidate_index / 1000.0,
                    "source": "distributional_energy_verifier_musr_checkpoints",
                }
            )
        if not candidates:
            raise CandidateCacheError(
                f"candidate checkpoint has no valid answers: {checkpoint_path}"
            )
        merged = dict(row)
        merged["gold"] = str(row.get("gold") or checkpoint.get("gold") or "")
        merged["candidate_cache_path"] = checkpoint_path.as_posix()
        merged["candidates"] = candidates
        rows.append(merged)
    return rows


def _candidate_pool(
    row: JsonMap, *, k: int | None = None, temperature: Any = None
) -> list[JsonMap]:
    candidates = list(row.get("candidates") or [])
    if temperature is not None:
        filtered = [
            candidate
            for candidate in candidates
            if candidate.get("temperature", "cached") == temperature
        ]
        candidates = filtered or candidates
    if k is not None:
        candidates = candidates[:k]
    return candidates


def _candidate_pool_counts(rows: Sequence[JsonMap], *, temperature: Any = None) -> list[int]:
    return [len(_candidate_pool(row, temperature=temperature)) for row in rows]


def _available_candidates_per_question(rows: Sequence[JsonMap], *, temperature: Any = None) -> int:
    counts = _candidate_pool_counts(rows, temperature=temperature)
    return min(counts) if counts else 0


def _default_sc_k_values(candidates_per_question: int) -> list[int]:
    if candidates_per_question <= 0:
        return []
    return list(range(1, candidates_per_question + 1, 2))


def _sanitize_sc_k_values(k_values: Sequence[int], *, candidates_per_question: int) -> list[int]:
    seen: set[int] = set()
    values: list[int] = []
    for raw in k_values:
        k = int(raw)
        if k <= 0 or k > candidates_per_question or k % 2 == 0 or k in seen:
            continue
        seen.add(k)
        values.append(k)
    return values


def _majority_answer(candidates: Sequence[JsonMap]) -> str | None:
    counts: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    for index, candidate in enumerate(candidates):
        answer = candidate.get("answer")
        if answer is None or str(answer).strip() == "":
            continue
        answer_text = str(answer)
        counts[answer_text] += 1
        first_seen.setdefault(answer_text, index)
    if not counts:
        return None
    return max(counts, key=lambda answer: (counts[answer], -first_seen[answer]))


def _is_correct(answer: str | None, gold: Any) -> int:
    return int(answer is not None and str(answer) == str(gold))


def _available_temperatures(rows: Sequence[JsonMap]) -> list[Any]:
    temperatures = {
        candidate.get("temperature", "cached")
        for row in rows
        for candidate in row.get("candidates", [])
    }
    return sorted(temperatures, key=str) or ["cached"]


def tuned_self_consistency(
    rows: Sequence[JsonMap],
    *,
    k_values: Sequence[int] | None = None,
    temperatures: Sequence[Any] | None = None,
) -> JsonDict:
    if not rows:
        return {
            "accuracy": 0.0,
            "config": {"k": 0, "temperature": None},
            "predictions": [],
            "correct": [],
            "k_sweep": {},
            "temperature_sweeps": {},
            "tuned_k": 0,
            "candidates_per_question": 0,
            "candidate_pool_counts": [],
            "degenerate_candidate_pool": False,
            "oracle_degenerate": False,
        }
    candidate_temperatures = (
        list(temperatures) if temperatures is not None else _available_temperatures(rows)
    )
    best: JsonDict | None = None
    temperature_sweeps: dict[str, dict[str, float]] = {}
    for temperature in candidate_temperatures:
        candidates_per_question = _available_candidates_per_question(rows, temperature=temperature)
        candidate_k_values = (
            _sanitize_sc_k_values(k_values, candidates_per_question=candidates_per_question)
            if k_values is not None
            else _default_sc_k_values(candidates_per_question)
        )
        k_sweep: dict[str, float] = {}
        temperature_records: list[JsonDict] = []
        for k in candidate_k_values:
            predictions = [
                _majority_answer(_candidate_pool(row, k=k, temperature=temperature)) for row in rows
            ]
            correct = [
                _is_correct(prediction, row.get("gold"))
                for prediction, row in zip(predictions, rows)
            ]
            accuracy = sum(correct) / len(rows)
            config = {"k": int(k), "temperature": temperature}
            rounded_accuracy = round(accuracy, 6)
            k_sweep[str(k)] = rounded_accuracy
            current = {
                "accuracy": rounded_accuracy,
                "config": config,
                "predictions": predictions,
                "correct": correct,
                "tuned_k": int(k),
                "candidates_per_question": candidates_per_question,
                "candidate_pool_counts": _candidate_pool_counts(rows, temperature=temperature),
                "degenerate_candidate_pool": candidates_per_question == 1,
                "oracle_degenerate": candidates_per_question == 1,
            }
            temperature_records.append(current)
        temperature_sweeps[str(temperature)] = k_sweep
        if not temperature_records:
            continue
        temperature_best = max(
            temperature_records,
            key=lambda item: (float(item["accuracy"]), -int(item["config"]["k"])),
        )
        temperature_best["k_sweep"] = dict(k_sweep)
        temperature_best["temperature_sweeps"] = dict(temperature_sweeps)
        if best is None or (
            float(temperature_best["accuracy"]),
            -int(temperature_best["config"]["k"]),
            str(temperature),
        ) > (
            float(best["accuracy"]),
            -int(best["config"]["k"]),
            str(best["config"]["temperature"]),
        ):
            best = temperature_best
    if best is None:
        return {
            "accuracy": 0.0,
            "config": {"k": 0, "temperature": None},
            "predictions": [None for _row in rows],
            "correct": [0 for _row in rows],
            "k_sweep": {},
            "temperature_sweeps": dict(temperature_sweeps),
            "tuned_k": 0,
            "candidates_per_question": 0,
            "candidate_pool_counts": _candidate_pool_counts(rows),
            "degenerate_candidate_pool": False,
            "oracle_degenerate": False,
        }
    best["temperature_sweeps"] = dict(temperature_sweeps)
    return best


def oracle_at_k(
    rows: Sequence[JsonMap],
    *,
    k: int | None = None,
    temperature: Any = None,
) -> tuple[float, list[int]]:
    correct = []
    oracle_k = (
        k if k is not None else _available_candidates_per_question(rows, temperature=temperature)
    )
    for row in rows:
        gold = str(row.get("gold"))
        row_correct = any(
            str(candidate.get("answer")) == gold
            for candidate in _candidate_pool(row, k=oracle_k, temperature=temperature)
        )
        correct.append(int(row_correct))
    return (round(sum(correct) / len(rows), 6) if rows else 0.0), correct


def abstention_degeneracy_guard(
    abstain_rate: float,
    *,
    threshold: float = ABSTENTION_DEGENERACY_THRESHOLD,
) -> JsonDict:
    rate = float(abstain_rate)
    if not math.isfinite(rate):
        raise ValueError("abstain_rate must be finite")
    limit = float(threshold)
    if not math.isfinite(limit):
        raise ValueError("threshold must be finite")
    degeneracy_flag = rate > limit
    if degeneracy_flag:
        rate_label = f"{rate:.3f}".replace(".", "p")
        threshold_label = f"{limit:.2f}".replace(".", "p")
        verdict = f"degenerate_abstaining_selector_abstain_rate_{rate_label}_gt_{threshold_label}"
    else:
        verdict = "nondegenerate_abstaining_selector"
    return {
        "verdict": verdict,
        "degeneracy_flag": degeneracy_flag,
        "abstain_rate": round(rate, 6),
        "threshold": round(limit, 6),
    }


def _select_verifier_answer(row: JsonMap, scorer: Scorer) -> str | None:
    scored: list[tuple[float, str, str | None]] = []
    for candidate in row.get("candidates", []):
        guarded = GuardedCandidate(candidate)
        energy = float(scorer(guarded))
        if not math.isfinite(energy):
            energy = math.inf
        scored.append((energy, str(candidate.get("candidate_id", "")), candidate.get("answer")))
    if not scored:
        return None
    _energy, _candidate_id, answer = min(scored, key=lambda item: (item[0], item[1]))
    return str(answer) if answer is not None else None


def paired_bootstrap_ci(
    verifier_correct: Sequence[int],
    baseline_correct: Sequence[int],
    *,
    seed: int = DEFAULT_RANDOM_SEED,
    samples: int = 2000,
) -> list[float]:
    pairs = list(zip(verifier_correct, baseline_correct))
    if not pairs:
        return [0.0, 0.0]
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(samples):
        total = 0
        for _item in pairs:
            verifier, baseline = pairs[rng.randrange(len(pairs))]
            total += verifier - baseline
        deltas.append(total / len(pairs))
    deltas.sort()
    lo_index = int(0.025 * (samples - 1))
    hi_index = int(0.975 * (samples - 1))
    return [round(deltas[lo_index], 6), round(deltas[hi_index], 6)]


def mcnemar_exact_p(verifier_correct: Sequence[int], baseline_correct: Sequence[int]) -> float:
    baseline_only = sum(
        1
        for verifier, baseline in zip(verifier_correct, baseline_correct)
        if baseline and not verifier
    )
    verifier_only = sum(
        1
        for verifier, baseline in zip(verifier_correct, baseline_correct)
        if verifier and not baseline
    )
    discordant = baseline_only + verifier_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, count) * (0.5**discordant)
        for count in range(0, min(baseline_only, verifier_only) + 1)
    )
    return round(min(1.0, 2.0 * tail), 6)


def evaluate_verifier(
    rows: Sequence[JsonMap],
    *,
    scorer: Scorer,
    seed: int = DEFAULT_RANDOM_SEED,
    bootstrap_samples: int = 2000,
    headroom_threshold: float = HEADROOM_THRESHOLD,
) -> JsonDict:
    """Evaluate one oracle-distinct verifier scorer against tuned SC."""

    rows_list = [dict(row) for row in rows if row.get("candidates")]
    tuned_sc = tuned_self_consistency(rows_list)
    sc_correct = [int(value) for value in tuned_sc.get("correct", [])]
    sc_predictions = list(tuned_sc.get("predictions", []))
    oracle_k = int(tuned_sc.get("candidates_per_question") or 0)
    oracle_temperature = tuned_sc.get("config", {}).get("temperature")
    oracle_accuracy, oracle_correct = oracle_at_k(
        rows_list,
        k=oracle_k,
        temperature=oracle_temperature,
    )
    verifier_predictions = [_select_verifier_answer(row, scorer) for row in rows_list]
    verifier_correct = [
        _is_correct(prediction, row.get("gold"))
        for prediction, row in zip(verifier_predictions, rows_list)
    ]
    n_flips_possible = sum(
        1 for sc_ok, oracle_ok in zip(sc_correct, oracle_correct) if not sc_ok and oracle_ok
    )
    verifier_accuracy = sum(verifier_correct) / len(rows_list) if rows_list else 0.0
    delta = verifier_accuracy - float(tuned_sc["accuracy"])
    ci95 = paired_bootstrap_ci(
        verifier_correct,
        sc_correct,
        seed=seed,
        samples=bootstrap_samples,
    )
    return {
        "n_rows": len(rows_list),
        "tuned_self_consistency": {
            "accuracy": tuned_sc["accuracy"],
            "config": tuned_sc["config"],
            "predictions": sc_predictions,
            "k_sweep": dict(tuned_sc.get("k_sweep") or {}),
            "tuned_k": int(tuned_sc.get("tuned_k") or tuned_sc["config"]["k"]),
            "candidates_per_question": int(tuned_sc.get("candidates_per_question") or 0),
            "candidate_pool_counts": list(tuned_sc.get("candidate_pool_counts") or []),
            "degenerate_candidate_pool": bool(tuned_sc.get("degenerate_candidate_pool")),
            "oracle_degenerate": bool(tuned_sc.get("oracle_degenerate")),
        },
        "oracle_at_k": oracle_accuracy,
        "oracle_k": oracle_k,
        "n_flips_possible": n_flips_possible,
        "headroom_present": bool(
            (oracle_accuracy - float(tuned_sc["accuracy"])) >= headroom_threshold
            and n_flips_possible > 0
        ),
        "verifier": {
            "accuracy": round(verifier_accuracy, 6),
            "predictions": verifier_predictions,
        },
        "verifier_minus_tuned_sc_delta": round(delta, 6),
        "verifier_minus_tuned_sc_ci95": ci95,
        "mcnemar_p": mcnemar_exact_p(verifier_correct, sc_correct),
        "paired_correct": {
            "verifier": verifier_correct,
            "tuned_self_consistency": sc_correct,
            "oracle_at_k": oracle_correct,
        },
    }


def _match_choice(text: str, choices: Sequence[str]) -> str | None:
    if not text:
        return None
    match = re.search(r"ANSWER\s*[:\-]?\s*(.+)", text, re.IGNORECASE)
    tail = (match.group(1) if match else text[-160:]).lower()
    hits = [choice for choice in choices if str(choice).lower() in tail]
    if len(hits) == 1:
        return str(hits[0])
    if hits:
        return str(max(hits, key=lambda choice: tail.rfind(str(choice).lower())))
    full = text.lower()
    hits = [choice for choice in choices if str(choice).lower() in full]
    return str(hits[-1]) if hits else None


def build_generation_prompt(row: JsonMap) -> str:
    choices = list(row.get("choices") or [])
    context = str(row.get("context") or row.get("narrative") or "")[:6000]
    return (
        "Read the context and answer the multiple-choice question with careful reasoning.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {row.get('question', '')}\n"
        f"CHOICES: {choices}\n\n"
        "End with a final line exactly: ANSWER: <one choice verbatim>."
    )


def generate_candidates_with_logprobs(
    row: JsonMap,
    *,
    generator: Callable[[str], JsonMap] | None = None,
    config: GenerationConfig | None = None,
    seed: int = DEFAULT_RANDOM_SEED,
) -> list[JsonDict]:
    """Generate fresh candidates through an injected logprob-capable backend."""

    cfg = config or GenerationConfig()
    if generator is None:
        raise GenerationUnavailableError(
            "fresh generation requires a logprob-capable backend for "
            f"{cfg.model} on CUDA GPU-{cfg.gpu}"
        )
    prompt = build_generation_prompt(row)
    candidates: list[JsonDict] = []
    for offset in range(cfg.k):
        payload = generator(prompt, seed=seed + offset, config=cfg)  # type: ignore[misc]
        text = str(payload.get("text", ""))
        answer = payload.get("answer") or _match_choice(text, list(row.get("choices") or []))
        candidates.append(
            {
                "candidate_id": f"{row.get('row_id', 'row')}/fresh-{offset}",
                "answer": answer,
                "reasoning": text,
                "token_logprobs": list(payload.get("token_logprobs") or []),
                "top_logprobs": list(payload.get("top_logprobs") or []),
                "mean_logprob": payload.get("mean_logprob"),
                "cache_index": offset,
                "temperature": cfg.temperature,
                "generation_model": cfg.model,
                "gpu": cfg.gpu,
                "source": "fresh_generation_with_logprobs",
            }
        )
    return candidates


def cached_trivial_energy(candidate: Mapping[str, Any]) -> float:
    """Trivial smoke scorer that uses cached non-oracle energy selection metadata."""

    return float(candidate.get("trivial_energy", 1.0))
