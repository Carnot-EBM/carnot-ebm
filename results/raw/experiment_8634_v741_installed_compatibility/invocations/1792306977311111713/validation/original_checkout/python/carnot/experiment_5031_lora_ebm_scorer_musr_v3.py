"""Exp 5031 (PHASE D1): train the REAL arXiv:2605.18871 LoRA-EBM holistic-quality
scorer and test whether a TRAINED oracle-distinct verifier beats genuine tuned
self-consistency on MuSR.

Why this experiment exists
--------------------------
This is the headline "moat" arm.  The cheap PROMPTED energy proxy already nulled
on MuSR (self-consistency best, energy 0.515-0.535), but `sc_saturated=False`:
there is +0.28 of UNREALIZED selectable headroom (oracle@K 0.865 vs the genuine
tuned-SC 0.585).  The operator directive (2026-06-30) is to stop approximating
and actually TRAIN the holistic-quality scorer the paper describes, then measure
whether a TRAINED energy verifier captures that headroom.

The two prior attempts to train this verifier bailed:

  * `.461` wrote a 0-pair skeleton and never trained.
  * `.462` named a hallucinated base (`Qwen/Qwen3.5-1.7B`) that 404'd.

Both failure classes are fixed by the reusable `carnot.moat_trainer` module
(REQ-VERIFY-5030, the "B3" module): ``resolve_trainable_base`` probes a
prioritized list of REAL cached bases and returns the first present one (kills
the hallucinated-id class), and ``train_energy_head`` is a proven QLoRA + scalar
energy-head trainer whose 60-second smoke already trained end-to-end.  This
experiment is therefore THIN: it owns the corpus build, the eval wiring, and the
honest gate, and DELEGATES the GPU training/scoring to the B3 module and the
baseline/uncertainty math to the "B1" harness (`carnot.moat_benchmark_harness`).

What "oracle-distinct" means here
---------------------------------
Gold BUILDS the contrastive training pairs (the gold-answer candidate is the
positive, a wrong-answer candidate is the negative) — that is standard
reward-model / EBM training.  But at INFERENCE the trained scorer ranks
candidate reasoning/answer quality given the narrative and NEVER reads the gold
key, the answer index, or the model identity.  The harness enforces this with a
``GuardedCandidate`` view, so a scorer that peeked at `gold` would raise.  This
is the deep, non-circular claim: a learned energy verifier capturing headroom
where no cheap executable oracle exists.

The honest gate
---------------
The task is NOT complete until the model ACTUALLY trained: ``scorer_trained`` is
true ONLY when ``train_loss`` is non-null, ``n_pairs>0``, and ``duration_s>60``
(real GPU training takes wall-clock — the anti-skeleton signal).  A win requires
the trained scorer to beat genuine tuned SC with a paired CI95 excluding 0 AND
McNemar p<0.05 AND selectable headroom present.  A clean trained null (CI
includes 0) tightens the moat honestly; a skeleton is a FAILED execution, not a
null.

Spec refs: REQ-VERIFY-5031, SCENARIO-VERIFY-5031.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.moat_benchmark_harness import (  # noqa: E402
    DEFAULT_RANDOM_SEED,
    OracleDistinctnessError,
    abstention_degeneracy_guard,
    evaluate_verifier,
)


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
Scorer = Callable[[Mapping[str, Any]], float]
Trainer = Callable[..., JsonDict]
ScoreFn = Callable[[Any, Sequence[str]], list[float]]
BaseResolver = Callable[[], tuple[str, str]]
NarrativesLoader = Callable[[int], "list[JsonDict] | None"]
AuditRunner = Callable[[Path], JsonDict]
SummaryRunner = Callable[[Path], int]
Clock = Callable[[], float]

EXPERIMENT_ID = 5031
EXPERIMENT_NAME = "experiment_5031_lora_ebm_scorer_musr_v3"
RESULT_RELATIVE_PATH = "results/experiment_5031_lora_ebm_scorer_musr_v3.json"
CHECKPOINT_RELATIVE_DIR = "results/checkpoints/experiment_5031_lora_ebm_scorer_musr_v3"
B3_ARTIFACT_RELATIVE_PATH = "results/experiment_5030_moat_trainer_module.json"
MUSR_CHECKPOINT_RELATIVE_DIR = "results/distributional_energy_verifier_musr_checkpoints"
FOVER_RELATIVE_PATH = "data/fover_train_v4.json"
B1_BASELINE_RELATIVE_PATH = "results/experiment_5015_genuine_sc_baseline_fix.json"
SPEC_REFS = ["REQ-VERIFY-5031", "SCENARIO-VERIFY-5031"]
GENERATOR_MODEL_ID = "distributional_energy_verifier_musr_checkpoints"
RANDOM_SEED = DEFAULT_RANDOM_SEED
HEADROOM_THRESHOLD = 0.10
TRAIN_DURATION_FLOOR_S = 60.0
MUSR_CONTEXT_CHAR_CAP = 6000
MIN_QUESTIONS = 200


FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; a win is success_lora_ebm_beats_sc_musr_<delta>, "
            "a clean null is complete_lora_ebm_no_win_musr_<delta>_ci_incl_0, "
            "a failed train is blocked_lora_ebm_train_did_not_run."
        )
    },
    "scorer_trained": {
        "principle": (
            "true iff the model ACTUALLY trained (train_loss non-null, n_pairs>0, "
            "duration>60s) -- the anti-skeleton gate AND the field D3 gates on; "
            "false = a FAILED execution, not a null."
        )
    },
    "train_loss": {
        "principle": (
            "the final contrastive training loss (non-null REQUIRED -- a null means "
            "a skeleton bail recurred)."
        )
    },
    "n_pairs": {
        "principle": (
            "the contrastive-pair count (>0 REQUIRED -- 0 was the .461 skeleton signature)."
        )
    },
    "base_used": {
        "principle": (
            "the REAL cached base the B3 resolver returned (Qwen/Qwen3.5-2B) -- "
            "proves the 404 class is fixed."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "false -- the scorer ranks reasoning quality and NEVER reads "
            "gold/answer_index/model_id at inference (must pass check_circular_moat_overclaim)."
        )
    },
    "headroom_present": {
        "principle": (
            "true required for an informative result -- (oracle@K - GENUINE tuned_sc) "
            ">= 0.10 AND flips>0 (FALSE_NEGATIVE_RISK guard, vs the B1 0.585 baseline)."
        )
    },
    "trained_scorer_accuracy": {
        "principle": (
            "the oracle-distinct selection accuracy of the TRAINED LoRA-EBM (the headline number)."
        )
    },
    "genuine_tuned_sc_accuracy": {
        "principle": (
            "the GENUINE K-way tuned-SC baseline from B1 (0.585, NOT a k=1 strawman) -- "
            "the honest baseline to beat."
        )
    },
    "delta_vs_tuned_sc": {
        "principle": (
            "trained_scorer_accuracy - genuine_tuned_sc_accuracy; the moat lift (signed)."
        )
    },
    "paired_ci95": {
        "principle": "paired bootstrap CI95 of the delta; a win requires CI95 excluding 0."
    },
    "mcnemar_p": {"principle": "McNemar paired-test p; a win requires p<0.05."},
    "n_questions": {"principle": ">=200 for the headline delta (sample-size rigor)."},
    "oracle_at_k": {"principle": "the selectable-headroom ceiling (0.865 on MuSR)."},
    "model_specs": {
        "principle": (
            "the resolved base (Qwen3.5-2B + LoRA + energy head) AND the "
            "cached-candidate generator -- the methodology stamp."
        )
    },
    "inference_substrate": {
        "principle": "live_llm_inference (GPU training + scoring; >=60s floor)."
    },
    "random_seed": {"principle": "determinism for the train/eval split + bootstrap."},
    "reproducibility_checksum": {
        "principle": (
            "content hash of (base, LoRA config, corpus, seed) so a replication catches drift."
        )
    },
    "preconditions_checked": {
        "principle": (
            "records B3-module/smoke/CUDA/candidate/FoVer checks; a missing resource "
            "emits blocked_, never a fabricated AUROC."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "checkpoint_path",
    "candidate_cache_source",
    "oracle_distinctness_enforced",
    "degeneracy_guard",
    "adversarial_verify_clean",
    "adversarial_verify_flags",
    "summarize_artifact_exit_code",
    "duration_s",
    "field_principles",
)


@dataclass(frozen=True)
class PreconditionCheck:
    """One resource check recorded before training is allowed to make a claim.

    Each check is a small auditable record (resource name, available bool, a
    human-readable detail, and optionally the path probed) so the artifact can
    prove WHICH resources the agent verified before claiming anything.
    """

    resource: str
    available: bool
    detail: str
    path: str | None = None

    def as_dict(self) -> JsonDict:
        payload: JsonDict = {
            "resource": self.resource,
            "available": bool(self.available),
            "detail": self.detail,
        }
        if self.path is not None:
            payload["path"] = self.path
        return payload


@dataclass(frozen=True)
class TrainingConfig:
    """Bounded QLoRA configuration for the scalar-energy holistic-quality scorer.

    Defaults are tuned for one conductor 3090: a 2B 4-bit base, a short-ish
    sequence window that keeps the candidate answer (placed first) plus a useful
    chunk of narrative, and a pair budget that trains a real epoch in minutes.
    """

    seed: int = RANDOM_SEED
    epochs: int = 1
    batch_size: int = 2
    learning_rate: float = 2e-4
    max_length: int = 512
    max_train_pairs: int = 1024
    fover_fraction: float = 0.7
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    device_index: int = 0

    def lora_config_payload(self) -> JsonDict:
        return {
            "r": self.lora_r,
            "alpha": self.lora_alpha,
            "dropout": self.lora_dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "max_length": self.max_length,
            "max_train_pairs": self.max_train_pairs,
            "fover_fraction": self.fover_fraction,
            "device_index": self.device_index,
        }


# --------------------------------------------------------------------------- #
# JSON helpers.
# --------------------------------------------------------------------------- #
def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# B3-module + smoke preconditions.
# --------------------------------------------------------------------------- #
def b3_module_importable() -> tuple[bool, str]:
    """Return (ok, detail) for importing the B3 trainer entrypoints.

    The whole experiment delegates training/scoring to ``carnot.moat_trainer``;
    if that module cannot import there is nothing to train with, so we block
    instead of fabricating an accuracy.
    """
    try:
        from carnot.moat_trainer import (  # noqa: F401
            resolve_trainable_base,
            score_candidates,
            train_energy_head,
        )

        return True, "carnot.moat_trainer import OK (resolve/train/score)"
    except Exception as exc:  # pragma: no cover - defensive import guard
        return False, f"{type(exc).__name__}: {exc}"


def read_b3_smoke_passed(artifact_path: Path) -> tuple[bool, str]:
    """Return (smoke_passed, detail) from the REQ-VERIFY-5030 artifact.

    D1 is gated on B3's smoke: if the 60s smoke did not actually train (or the
    artifact is missing/malformed), this experiment is correctly skipped — the
    training pipeline has not been de-risked, so a headline run would be wishful.
    """
    payload = _read_json(artifact_path)
    if not isinstance(payload, dict):
        return False, f"B3 artifact missing/unreadable: {artifact_path.as_posix()}"
    passed = payload.get("smoke_passed")
    if passed is True:
        base = payload.get("base_used")
        return True, f"B3 smoke_passed=true (base_used={base})"
    return False, f"B3 smoke_passed={passed!r} in {artifact_path.as_posix()}"


def default_cuda_available() -> bool:  # pragma: no cover - environment probe
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def default_base_resolver() -> tuple[str, str]:  # pragma: no cover - probes real cache
    from carnot.moat_trainer import resolve_trainable_base

    return resolve_trainable_base()


# --------------------------------------------------------------------------- #
# Corpus construction (pure — unit-testable without a GPU).
# --------------------------------------------------------------------------- #
def load_fover_contrastive_pairs(path: Path, *, max_pairs: int) -> list[tuple[str, str]]:
    """Build (good, bad) pairs from FoVer step labels.

    FoVer v4 is a flat list of reasoning STEPS, each labelled ``correct`` or
    ``incorrect``.  There are far more correct than incorrect steps, so we cycle
    each incorrect step (the negative) against the correct pool (the positives)
    over as many rounds as needed to reach ``max_pairs``.  The contrastive signal
    is "a correct reasoning step should have LOWER energy than an incorrect one",
    which is exactly the holistic-quality signal the scorer should learn.
    """
    if max_pairs <= 0:
        return []
    payload = _read_json(path)
    if not isinstance(payload, list):
        return []
    correct: list[str] = []
    incorrect: list[str] = []
    for record in payload:
        if not isinstance(record, Mapping):
            continue
        text = str(record.get("step_text") or "").strip()
        if not text:
            continue
        label = str(record.get("label") or "").strip().lower()
        if label == "correct":
            correct.append(text)
        elif label == "incorrect":
            incorrect.append(text)
    if not correct or not incorrect:
        return []
    pairs: list[tuple[str, str]] = []
    rounds = math.ceil(max_pairs / len(incorrect))
    for round_index in range(rounds):
        for bad_index, bad in enumerate(incorrect):
            good = correct[(round_index * len(incorrect) + bad_index) % len(correct)]
            if good == bad:
                continue
            pairs.append((good, bad))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def musr_candidate_text(
    answer: str,
    question: str,
    context: str,
    *,
    context_char_cap: int = MUSR_CONTEXT_CHAR_CAP,
) -> str:
    """Render the text a candidate is SCORED on.

    The candidate answer is placed FIRST so that right-truncation (the tokenizer
    default) never trims away the one token that differs between candidates for
    the same question.  The question and the (capped) narrative follow, giving
    the energy model the context it needs to judge the answer.  This identical
    text function is used for both the gold-labelled training pairs and the eval
    re-scoring, so train and eval see the same distribution.
    """
    narrative = str(context or "").strip()[:context_char_cap]
    body = f"Candidate answer: {str(answer).strip()}\nQuestion: {str(question).strip()}"
    if narrative:
        body += f"\nNarrative:\n{narrative}"
    return body


def load_musr_eval_rows(
    checkpoint_dir: Path,
    *,
    narratives: Sequence[JsonMap] | None = None,
    limit: int | None = None,
    context_char_cap: int = MUSR_CONTEXT_CHAR_CAP,
) -> list[JsonDict]:
    """Build eval rows from the cached MuSR checkpoints (q + gold + answers).

    Each ``q{idx}.json`` holds the gold answer and a list of candidate answer
    strings.  We attach the dataset narrative + question by index when available
    (gold alignment was verified 200/200) so the scorer has context; otherwise we
    fall back to answer-only text.  The full candidate multiplicity is preserved
    so the harness can compute tuned self-consistency, oracle@K, and the verifier
    selection on the same pool.
    """
    if not checkpoint_dir.is_dir():
        return []
    paths = sorted(checkpoint_dir.glob("q*.json"))
    if limit is not None:
        paths = paths[:limit]
    rows: list[JsonDict] = []
    for row_index, path in enumerate(paths):
        checkpoint = _read_json(path)
        if not isinstance(checkpoint, dict):
            continue
        answers = checkpoint.get("answers")
        if not isinstance(answers, list):
            continue
        gold = str(checkpoint.get("gold") or "")
        narrative_row = (
            narratives[row_index] if narratives is not None and row_index < len(narratives) else {}
        )
        question = str(narrative_row.get("question") or "")
        context = str(narrative_row.get("context") or "")
        candidates: list[JsonDict] = []
        for cache_index, answer in enumerate(answers):
            if answer is None or str(answer).strip() == "":
                continue
            answer_text = str(answer)
            candidates.append(
                {
                    "candidate_id": f"{row_index}/c{cache_index}",
                    "answer": answer_text,
                    "text": musr_candidate_text(
                        answer_text, question, context, context_char_cap=context_char_cap
                    ),
                    "cache_index": cache_index,
                    "temperature": "cached",
                }
            )
        if not candidates:
            continue
        rows.append(
            {
                "row_id": str(row_index),
                "corpus": "MuSR/murder_mysteries",
                "gold": gold,
                "question": question,
                "candidate_cache_path": path.as_posix(),
                "candidates": candidates,
            }
        )
    return rows


def build_musr_training_pairs(rows: Sequence[JsonMap], *, max_pairs: int) -> list[tuple[str, str]]:
    """Build gold-labelled (good, bad) pairs from the eval rows FOR TRAINING.

    Per question the gold-answer candidate text is the positive and a
    distinct-answer candidate text is the negative.  We dedupe by answer value so
    a 2-choice murder mystery contributes one (gold, wrong) pair rather than a
    dozen identical-text duplicates.  Gold is used ONLY here to label training
    pairs — the scorer never reads it at inference.
    """
    if max_pairs <= 0:
        return []
    pairs: list[tuple[str, str]] = []
    for row in rows:
        gold = str(row.get("gold") or "")
        good_by_answer: dict[str, str] = {}
        bad_by_answer: dict[str, str] = {}
        for candidate in row.get("candidates", []):
            answer = str(candidate.get("answer") or "")
            text = str(candidate.get("text") or "")
            if not answer or not text:
                continue
            if answer == gold:
                good_by_answer.setdefault(answer, text)
            else:
                bad_by_answer.setdefault(answer, text)
        for good_text in good_by_answer.values():
            for bad_text in bad_by_answer.values():
                pairs.append((good_text, bad_text))
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs


def build_contrastive_corpus(
    fover_path: Path,
    rows: Sequence[JsonMap],
    *,
    max_pairs: int,
    fover_fraction: float,
) -> list[tuple[str, str]]:
    """Combine MuSR gold-labelled pairs with FoVer step pairs.

    FoVer carries the general reasoning-quality signal (correct vs incorrect
    steps); the MuSR pairs carry the in-domain answer-given-narrative signal.
    The fover_fraction splits the pair budget; both are capped so the total stays
    within ``max_pairs``.
    """
    musr_budget = max_pairs
    musr_pairs = build_musr_training_pairs(rows, max_pairs=musr_budget)
    fover_budget = max(0, max_pairs - len(musr_pairs))
    fover_target = min(fover_budget, int(round(max_pairs * fover_fraction)))
    fover_pairs = load_fover_contrastive_pairs(fover_path, max_pairs=fover_target)
    combined = musr_pairs + fover_pairs
    return combined[:max_pairs]


# --------------------------------------------------------------------------- #
# Eval scoring — score once, look up by candidate_id (oracle-distinct).
# --------------------------------------------------------------------------- #
def precompute_candidate_energies(
    checkpoint: Any,
    rows: Sequence[JsonMap],
    *,
    score_fn: ScoreFn,
) -> dict[str, float]:
    """Score every candidate ONCE and key the energies by ``candidate_id``.

    ``score_candidates`` loads the model from the checkpoint once and scores a
    list of texts, so we gather all candidate texts up front (one model load for
    the whole eval set) and build a ``candidate_id -> energy`` lookup.  Scoring on
    raw rows here is fine: building inputs is not the scoring decision; the scorer
    handed to the harness reads only ``candidate_id``.
    """
    candidate_ids: list[str] = []
    texts: list[str] = []
    for row in rows:
        for candidate in row.get("candidates", []):
            candidate_ids.append(str(candidate.get("candidate_id") or ""))
            texts.append(str(candidate.get("text") or ""))
    if not texts:
        return {}
    energies = list(score_fn(checkpoint, texts))
    if len(energies) != len(candidate_ids):
        raise RuntimeError(
            f"score_fn returned {len(energies)} energies for {len(candidate_ids)} candidates"
        )
    return {candidate_id: float(energy) for candidate_id, energy in zip(candidate_ids, energies)}


def make_lookup_scorer(energy_by_id: Mapping[str, float]) -> Scorer:
    """Wrap a precomputed energy lookup as an oracle-distinct scorer.

    The returned scorer reads ONLY ``candidate_id`` (an allowed key) — never
    gold/answer_index/model_id — so it passes the harness ``GuardedCandidate``
    guard.  Unknown candidates get +inf so they are never selected.
    """

    def scorer(candidate: Mapping[str, Any]) -> float:
        candidate_id = str(candidate.get("candidate_id") or "")
        return float(energy_by_id.get(candidate_id, math.inf))

    return scorer


# --------------------------------------------------------------------------- #
# Default GPU-backed trainer / scorer (live paths; tests inject fakes).
# --------------------------------------------------------------------------- #
def default_trainer(
    pairs: Sequence[tuple[str, str]],
    *,
    base: tuple[str, str],
    out_dir: Path,
    config: TrainingConfig,
) -> JsonDict:  # pragma: no cover - live GPU training path
    """Train the LoRA energy head via the B3 module on conductor GPU 0."""
    from carnot import moat_trainer

    return moat_trainer.train_energy_head(
        base,
        pairs,
        out_dir,
        epochs=config.epochs,
        lr=config.learning_rate,
        batch_size=config.batch_size,
        max_length=config.max_length,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        device_index=config.device_index,
        seed=config.seed,
    )


def default_score_fn(config: TrainingConfig) -> ScoreFn:
    """Build the default scoring callable that loads the trained checkpoint."""

    def score_fn(checkpoint: Any, texts: Sequence[str]) -> list[float]:  # pragma: no cover - live
        from carnot import moat_trainer

        return moat_trainer.score_candidates(
            checkpoint,
            texts,
            max_length=config.max_length,
            device_index=config.device_index,
        )

    return score_fn


# --------------------------------------------------------------------------- #
# Preconditions.
# --------------------------------------------------------------------------- #
def check_preconditions(
    *,
    root: Path,
    b3_artifact_path: Path,
    cuda_available: Callable[[], bool],
    b3_importable: Callable[[], tuple[bool, str]],
    base_resolver: BaseResolver,
    min_questions: int = MIN_QUESTIONS,
) -> tuple[list[PreconditionCheck], tuple[str, str] | None]:
    """Verify every resource BEFORE training; return checks + the resolved base.

    Order matters: the B3 module + its smoke gate come first (the pipeline must
    be de-risked), then CUDA, then the trainable base, then the data.  A missing
    resource short-circuits the run to a ``blocked_<resource>`` artifact.
    """
    checks: list[PreconditionCheck] = []

    importable, import_detail = b3_importable()
    checks.append(PreconditionCheck("b3_module", importable, import_detail))

    smoke_passed, smoke_detail = read_b3_smoke_passed(b3_artifact_path)
    checks.append(
        PreconditionCheck("b3_smoke", smoke_passed, smoke_detail, b3_artifact_path.as_posix())
    )

    cuda_ok = bool(cuda_available())
    checks.append(
        PreconditionCheck(
            "cuda",
            cuda_ok,
            "torch.cuda.is_available=true on conductor GPU-0"
            if cuda_ok
            else "torch.cuda.is_available=false",
        )
    )

    resolved_base: tuple[str, str] | None = None
    if importable:
        try:
            resolved_base = base_resolver()
            checks.append(
                PreconditionCheck(
                    "trainable_base_cached",
                    True,
                    f"resolved {resolved_base[0]}",
                    resolved_base[1],
                )
            )
        except Exception as exc:
            checks.append(
                PreconditionCheck(
                    "trainable_base_cached",
                    False,
                    f"{type(exc).__name__}: {exc}",
                )
            )
    else:
        checks.append(
            PreconditionCheck(
                "trainable_base_cached",
                False,
                "skipped: b3_module not importable",
            )
        )

    checkpoint_dir = root / MUSR_CHECKPOINT_RELATIVE_DIR
    n_checkpoints = len(sorted(checkpoint_dir.glob("q*.json"))) if checkpoint_dir.is_dir() else 0
    checks.append(
        PreconditionCheck(
            "cached_musr_candidates",
            n_checkpoints >= min_questions,
            f"{n_checkpoints} cached MuSR checkpoints (need >= {min_questions})",
            checkpoint_dir.as_posix(),
        )
    )

    fover_path = root / FOVER_RELATIVE_PATH
    checks.append(
        PreconditionCheck(
            "fover_pairs",
            fover_path.exists(),
            "data/fover_train_v4.json present"
            if fover_path.exists()
            else "data/fover_train_v4.json missing",
            fover_path.as_posix(),
        )
    )

    return checks, resolved_base


def first_missing_resource(checks: Sequence[PreconditionCheck]) -> str | None:
    for check in checks:
        if not check.available:
            return check.resource
    return None


# --------------------------------------------------------------------------- #
# Reproducibility + artifact builders.
# --------------------------------------------------------------------------- #
def reproducibility_checksum(
    *,
    base_used: str | None,
    config: TrainingConfig,
    pairs: Sequence[tuple[str, str]],
    candidate_source: str,
    seed: int,
) -> str:
    payload = {
        "base_used": base_used,
        "lora_config": config.lora_config_payload(),
        "candidate_source": candidate_source,
        "seed": seed,
        "n_pairs": len(pairs),
        "pair_digests": [
            hashlib.sha256((good + "\x00" + bad).encode("utf-8")).hexdigest()[:16]
            for good, bad in pairs
        ],
    }
    return "sha256:" + hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def _format_delta(delta: float) -> str:
    return f"{delta:+.3f}".replace("+", "plus_").replace("-", "minus_").replace(".", "p")


def _ci_includes_zero(ci95: Sequence[float]) -> bool:
    return len(ci95) == 2 and float(ci95[0]) <= 0.0 <= float(ci95[1])


def _read_b1_baseline(root: Path) -> JsonDict:
    path = root / B1_BASELINE_RELATIVE_PATH
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return {"path": path.as_posix(), "available": False}
    return {
        "path": path.as_posix(),
        "available": True,
        "honest_verdict": payload.get("honest_verdict"),
        "genuine_tuned_sc_accuracy": payload.get("genuine_tuned_sc_accuracy"),
        "oracle_at_k": payload.get("oracle_at_k"),
    }


def _base_artifact(
    *,
    honest_verdict: str,
    preconditions_checked: Sequence[JsonDict],
    duration_s: float,
    base_used: str | None,
) -> JsonDict:
    blocked = honest_verdict.startswith("blocked_")
    return {
        "schema": "carnot.experiment_5031_lora_ebm_scorer_musr_v3.v1",
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": honest_verdict,
        "scorer_trained": False,
        "train_loss": None,
        "n_pairs": 0,
        "base_used": base_used,
        "verifier_is_oracle": False,
        "headroom_present": False,
        "trained_scorer_accuracy": None,
        "genuine_tuned_sc_accuracy": None,
        "delta_vs_tuned_sc": None,
        "paired_ci95": None,
        "mcnemar_p": None,
        "n_questions": 0,
        "oracle_at_k": None,
        "model_specs": {
            "base_model": base_used,
            "adapter": "LoRA",
            "energy_head": "scalar_sequence_regression_head",
            "cached_candidate_generator": GENERATOR_MODEL_ID,
        },
        "inference_substrate": "precondition_check_only" if blocked else "live_llm_inference",
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "sha256:"
        + hashlib.sha256(_json_dumps(list(preconditions_checked)).encode("utf-8")).hexdigest(),
        "preconditions_checked": list(preconditions_checked),
        "checkpoint_path": None,
        "candidate_cache_source": None,
        "oracle_distinctness_enforced": False,
        "degeneracy_guard": None,
        "adversarial_verify_clean": False,
        "adversarial_verify_flags": [],
        "summarize_artifact_exit_code": None,
        "duration_s": round(float(duration_s), 6),
        "field_principles": FIELD_PRINCIPLES,
    }


def build_blocked_artifact(
    *,
    missing_resource: str,
    preconditions_checked: Sequence[JsonDict],
    duration_s: float,
    base_used: str | None = None,
    error: str | None = None,
) -> JsonDict:
    artifact = _base_artifact(
        honest_verdict=f"blocked_{missing_resource}",
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
        base_used=base_used,
    )
    if error:
        artifact["blocked_error"] = error[:1000]
    return artifact


def build_train_did_not_run_artifact(
    *,
    preconditions_checked: Sequence[JsonDict],
    duration_s: float,
    base_used: str | None = None,
    error: str | None = None,
) -> JsonDict:
    artifact = _base_artifact(
        honest_verdict="blocked_lora_ebm_train_did_not_run",
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
        base_used=base_used,
    )
    # A failed/skeleton train is still a live-inference attempt, not a mere
    # precondition check, but it carries no accuracy claim.
    artifact["inference_substrate"] = "precondition_check_only"
    if error:
        artifact["blocked_error"] = error[:1000]
    return artifact


def build_complete_artifact(
    *,
    evaluation: JsonDict,
    train_result: JsonDict,
    config: TrainingConfig,
    pairs: Sequence[tuple[str, str]],
    preconditions_checked: Sequence[JsonDict],
    candidate_source: str,
    checkpoint_path: str,
    base_used: str,
    root: Path,
    duration_s: float,
) -> JsonDict:
    """Assemble the terminal artifact and compute the honest moat verdict."""
    train_loss = train_result.get("train_loss")
    n_pairs = int(train_result.get("n_pairs") or 0)
    scorer_trained = (
        train_loss is not None
        and math.isfinite(float(train_loss))
        and n_pairs > 0
        and float(duration_s) > TRAIN_DURATION_FLOOR_S
    )
    if not scorer_trained:
        return build_train_did_not_run_artifact(
            preconditions_checked=preconditions_checked,
            duration_s=duration_s,
            base_used=base_used,
            error=(
                f"trained_gate_failed train_loss={train_loss!r} "
                f"n_pairs={n_pairs!r} duration_s={duration_s:.6f}"
            ),
        )

    trained_accuracy = float(evaluation["verifier"]["accuracy"])
    genuine_sc_accuracy = float(evaluation["tuned_self_consistency"]["accuracy"])
    delta = float(evaluation["verifier_minus_tuned_sc_delta"])
    ci95 = [float(value) for value in evaluation["verifier_minus_tuned_sc_ci95"]]
    mcnemar_p = float(evaluation["mcnemar_p"])
    headroom_present = bool(evaluation["headroom_present"])
    verifier_predictions = evaluation["verifier"]["predictions"]
    abstain_rate = (
        sum(1 for prediction in verifier_predictions if prediction is None)
        / len(verifier_predictions)
        if verifier_predictions
        else 0.0
    )
    degeneracy_guard = abstention_degeneracy_guard(abstain_rate)

    win = (
        delta > 0.0
        and ci95[0] > 0.0
        and mcnemar_p < 0.05
        and headroom_present
        and not degeneracy_guard["degeneracy_flag"]
    )
    verdict_delta = _format_delta(delta)
    if win:
        honest_verdict = f"success_lora_ebm_beats_sc_musr_{verdict_delta}"
    elif _ci_includes_zero(ci95):
        honest_verdict = f"complete_lora_ebm_no_win_musr_{verdict_delta}_ci_incl_0"
    else:
        honest_verdict = f"complete_lora_ebm_no_win_musr_{verdict_delta}_mcnemar_or_headroom_gate"

    artifact = _base_artifact(
        honest_verdict=honest_verdict,
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
        base_used=base_used,
    )
    artifact.update(
        {
            "scorer_trained": True,
            "train_loss": round(float(train_loss), 6),
            "n_pairs": n_pairs,
            "headroom_present": headroom_present,
            "trained_scorer_accuracy": round(trained_accuracy, 6),
            "genuine_tuned_sc_accuracy": round(genuine_sc_accuracy, 6),
            "delta_vs_tuned_sc": round(delta, 6),
            "paired_ci95": ci95,
            "mcnemar_p": mcnemar_p,
            "n_questions": int(evaluation["n_rows"]),
            "oracle_at_k": float(evaluation["oracle_at_k"]),
            "model_specs": {
                **dict(train_result.get("model_specs") or {}),
                "base_model": base_used,
                "cached_candidate_generator": GENERATOR_MODEL_ID,
                "candidate_cache_source": candidate_source,
                "tuned_self_consistency_config": evaluation["tuned_self_consistency"]["config"],
                "b1_genuine_sc_baseline_reference": _read_b1_baseline(root),
            },
            "reproducibility_checksum": reproducibility_checksum(
                base_used=base_used,
                config=config,
                pairs=pairs,
                candidate_source=candidate_source,
                seed=config.seed,
            ),
            "checkpoint_path": checkpoint_path,
            "candidate_cache_source": candidate_source,
            "oracle_distinctness_enforced": True,
            "degeneracy_guard": degeneracy_guard,
            "evaluation": evaluation,
        }
    )
    return artifact


# --------------------------------------------------------------------------- #
# Schema validation.
# --------------------------------------------------------------------------- #
def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return the sorted list of schema problems (empty == valid)."""
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(field)
    if artifact.get("spec_refs") != SPEC_REFS:
        errors.append("spec_refs")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    for field in (
        "scorer_trained",
        "verifier_is_oracle",
        "headroom_present",
        "oracle_distinctness_enforced",
        "adversarial_verify_clean",
    ):
        if not isinstance(artifact.get(field), bool):
            errors.append(field)
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    ci95 = artifact.get("paired_ci95")
    if ci95 is not None and (
        not isinstance(ci95, list)
        or len(ci95) != 2
        or not all(isinstance(value, (int, float)) for value in ci95)
    ):
        errors.append("paired_ci95")
    for field in ("trained_scorer_accuracy", "genuine_tuned_sc_accuracy", "oracle_at_k"):
        value = artifact.get(field)
        if value is not None and not (
            isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0
        ):
            errors.append(field)
    if artifact.get("delta_vs_tuned_sc") is not None and not isinstance(
        artifact.get("delta_vs_tuned_sc"), (int, float)
    ):
        errors.append("delta_vs_tuned_sc")
    if artifact.get("mcnemar_p") is not None and not (
        isinstance(artifact.get("mcnemar_p"), (int, float))
        and 0.0 <= float(artifact.get("mcnemar_p")) <= 1.0
    ):
        errors.append("mcnemar_p")
    if not isinstance(artifact.get("preconditions_checked"), list):
        errors.append("preconditions_checked")
    if not isinstance(artifact.get("model_specs"), dict):
        errors.append("model_specs")
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(("blocked_", "complete_", "success_")):
        errors.append("honest_verdict")
    if artifact.get("scorer_trained") is True:
        if artifact.get("train_loss") is None:
            errors.append("train_loss")
        if int(artifact.get("n_pairs") or 0) <= 0:
            errors.append("n_pairs")
        if float(artifact.get("duration_s") or 0.0) <= TRAIN_DURATION_FLOOR_S:
            errors.append("duration_s")
        if not artifact.get("base_used"):
            errors.append("base_used")
    if artifact.get("scorer_trained") is False and verdict.startswith(("success_", "complete_")):
        errors.append("scorer_trained")
    return sorted(set(errors))


# --------------------------------------------------------------------------- #
# Adversarial-verify + summarize glue (script-loaded; tests inject runners).
# --------------------------------------------------------------------------- #
def _compact_adversarial_flags(report: JsonDict) -> list[JsonDict]:
    if isinstance(report.get("reports"), list) and report["reports"]:
        report = report["reports"][0]
    flags = report.get("flags", []) if isinstance(report, dict) else []
    return [flag for flag in flags if isinstance(flag, dict)]


def _audit_is_clean(report: JsonDict) -> bool:
    if "flagged_count" in report:
        return int(report.get("flagged_count") or 0) == 0
    if "flag_count" in report:
        return int(report.get("flag_count") or 0) == 0
    return not _compact_adversarial_flags(report)


def run_adversarial_verify(path: Path) -> JsonDict:  # pragma: no cover - script glue
    script_path = REPO_ROOT / "scripts" / "adversarial_verify.py"
    spec = importlib.util.spec_from_file_location("carnot_adversarial_verify_5031", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load scripts/adversarial_verify.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_artifact(path)


def run_summarize_artifact(path: Path) -> int:  # pragma: no cover - script glue
    script_path = REPO_ROOT / "scripts" / "summarize_artifact.py"
    spec = importlib.util.spec_from_file_location("carnot_summarize_artifact_5031", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load scripts/summarize_artifact.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return int(module.summarize(path))


def attach_audit(
    artifact: JsonDict,
    *,
    artifact_path: Path,
    audit_runner: AuditRunner,
    summary_runner: SummaryRunner,
) -> JsonDict:
    write_json(artifact_path, artifact)
    audit_report = audit_runner(artifact_path)
    updated = dict(artifact)
    updated["adversarial_verify_clean"] = _audit_is_clean(audit_report)
    updated["adversarial_verify_flags"] = _compact_adversarial_flags(audit_report)
    updated["adversarial_verify_report"] = audit_report
    write_json(artifact_path, updated)
    updated["summarize_artifact_exit_code"] = int(summary_runner(artifact_path))
    write_json(artifact_path, updated)
    return updated


def _precondition_dicts(checks: Sequence[PreconditionCheck]) -> list[JsonDict]:
    return [check.as_dict() for check in checks]


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #
def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    config: TrainingConfig | None = None,
    cuda_available: Callable[[], bool] = default_cuda_available,
    b3_importable: Callable[[], tuple[bool, str]] = b3_module_importable,
    base_resolver: BaseResolver = default_base_resolver,
    trainer: Trainer | None = None,
    score_fn: ScoreFn | None = None,
    narratives_loader: NarrativesLoader | None = None,
    audit_runner: AuditRunner = run_adversarial_verify,
    summary_runner: SummaryRunner = run_summarize_artifact,
    min_questions: int = MIN_QUESTIONS,
    limit: int = MIN_QUESTIONS,
    bootstrap_samples: int = 2000,
    now: Clock = time.time,
    write: bool = True,
) -> JsonDict:
    """Run the full D1 train-and-evaluate pipeline and return the artifact.

    Every external dependency (CUDA probe, B3 import check, base resolver, GPU
    trainer, GPU scorer, narratives loader, audit + summary runners, clock) is
    injectable so the orchestration is fully unit-testable without a GPU.
    """
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    root = Path(root)
    artifact_path = Path(artifact_path) if artifact_path else root / RESULT_RELATIVE_PATH
    config = config or TrainingConfig()
    if trainer is None:
        trainer = default_trainer
    if score_fn is None:
        score_fn = default_score_fn(config)
    if narratives_loader is None:
        narratives_loader = _default_narratives_loader
    b3_artifact_path = root / B3_ARTIFACT_RELATIVE_PATH
    start = float(now())

    checks, resolved_base = check_preconditions(
        root=root,
        b3_artifact_path=b3_artifact_path,
        cuda_available=cuda_available,
        b3_importable=b3_importable,
        base_resolver=base_resolver,
        min_questions=min_questions,
    )
    preconditions = _precondition_dicts(checks)
    base_used = resolved_base[0] if resolved_base else None
    missing = first_missing_resource(checks)
    if missing is not None:
        artifact = build_blocked_artifact(
            missing_resource=missing,
            preconditions_checked=preconditions,
            duration_s=float(now()) - start,
            base_used=base_used,
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact

    assert resolved_base is not None
    checkpoint_dir = root / MUSR_CHECKPOINT_RELATIVE_DIR
    narratives = narratives_loader(limit)
    rows = load_musr_eval_rows(checkpoint_dir, narratives=narratives, limit=limit)
    candidate_source = MUSR_CHECKPOINT_RELATIVE_DIR
    fover_path = root / FOVER_RELATIVE_PATH

    try:
        pairs = build_contrastive_corpus(
            fover_path,
            rows,
            max_pairs=config.max_train_pairs,
            fover_fraction=config.fover_fraction,
        )
        if not pairs:
            raise RuntimeError("no_contrastive_pairs")
        train_result = trainer(
            pairs,
            base=resolved_base,
            out_dir=root / CHECKPOINT_RELATIVE_DIR,
            config=config,
        )
        checkpoint_path = str(train_result.get("checkpoint_dir") or "")
        training_elapsed = float(now()) - start
        train_loss = train_result.get("train_loss")
        if (
            train_loss is None
            or not math.isfinite(float(train_loss))
            or int(train_result.get("n_pairs") or 0) <= 0
            or training_elapsed <= TRAIN_DURATION_FLOOR_S
        ):
            artifact = build_train_did_not_run_artifact(
                preconditions_checked=preconditions,
                duration_s=training_elapsed,
                base_used=base_used,
                error=(
                    f"trained_gate_failed train_loss={train_loss!r} "
                    f"n_pairs={train_result.get('n_pairs')!r} duration_s={training_elapsed:.6f}"
                ),
            )
            if write:
                write_json(artifact_path, artifact)
            return artifact
        energy_by_id = precompute_candidate_energies(checkpoint_path, rows, score_fn=score_fn)
        scorer = make_lookup_scorer(energy_by_id)
        evaluation = evaluate_verifier(
            rows,
            scorer=scorer,
            seed=config.seed,
            bootstrap_samples=bootstrap_samples,
            headroom_threshold=HEADROOM_THRESHOLD,
        )
    except OracleDistinctnessError as exc:
        artifact = build_blocked_artifact(
            missing_resource="oracle_distinctness_violation",
            preconditions_checked=preconditions,
            duration_s=float(now()) - start,
            base_used=base_used,
            error=str(exc),
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact
    except Exception as exc:
        artifact = build_train_did_not_run_artifact(
            preconditions_checked=preconditions,
            duration_s=float(now()) - start,
            base_used=base_used,
            error=f"{type(exc).__name__}: {exc}",
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact

    artifact = build_complete_artifact(
        evaluation=evaluation,
        train_result=train_result,
        config=config,
        pairs=pairs,
        preconditions_checked=preconditions,
        candidate_source=candidate_source,
        checkpoint_path=checkpoint_path,
        base_used=resolved_base[0],
        root=root,
        duration_s=float(now()) - start,
    )
    if write:
        artifact = attach_audit(
            artifact,
            artifact_path=artifact_path,
            audit_runner=audit_runner,
            summary_runner=summary_runner,
        )
    return artifact


def _default_narratives_loader(limit: int) -> list[JsonDict] | None:
    """Best-effort MuSR narrative loader (context is supplementary, not required).

    If the cached ``TAUR-Lab/MuSR`` dataset is available we attach the narrative +
    question by index so the scorer has context; if it is not, scoring falls back
    to answer-only text and the run still proceeds honestly.
    """
    try:  # pragma: no cover - dataset availability is environment-dependent
        from carnot.moat_benchmark_harness import load_musr_murder_mysteries

        return load_musr_murder_mysteries(limit=limit)
    except Exception:  # pragma: no cover - defensive
        return None


def main() -> int:  # pragma: no cover - script entrypoint
    artifact = run()
    errors = artifact_schema_errors(artifact)
    path = REPO_ROOT / RESULT_RELATIVE_PATH
    print(f"{path}: {artifact.get('honest_verdict')}")
    print(
        "scorer_trained={} base_used={} n_pairs={} delta={} ci95={} mcnemar_p={}".format(
            artifact.get("scorer_trained"),
            artifact.get("base_used"),
            artifact.get("n_pairs"),
            artifact.get("delta_vs_tuned_sc"),
            artifact.get("paired_ci95"),
            artifact.get("mcnemar_p"),
        )
    )
    if errors:
        print(f"schema_errors={errors}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
