"""Resumable GSM8K generation-corpus builder helpers (P0.1 / exp3448).

**Why this module exists (plain-language summary):**
    The single most important test in the project — P0.1, "does energy-based
    selection/voting on continuous latents BEAT plain token-sampling
    self-consistency at equal compute?" — has failed to land THREE times. The
    most recent failure (exp3437) was NOT scientific: it was a 1201-second
    idle-timeout. A single in-session job tried to do live 35B generation over
    ``200 x k`` samples AND score energy/self-consistency, ran silently past the
    agent's ~20-minute wall-clock+idle budget, and produced no artifact at all.

    The structural fix is to DECOUPLE the work. Generation is the expensive,
    non-deterministic, GPU-bound half; scoring is cheap and deterministic. This
    experiment (exp3448) does ONLY generation: it samples candidate solutions
    from the SOTA GGUF and writes them to an append-only, resumable corpus at
    ``data/p01_gsm8k_generations.jsonl``. A separate scoring task (exp3449) then
    reads that corpus with NO live model — so it can never idle-timeout — and
    answers the P0.1 question deterministically.

    This module holds the pure, GPU-free, unit-testable pieces of the builder:
      * the resume contract (which problems are already done on disk),
      * the per-problem corpus-row shape that the scoring task consumes,
      * the warm-up self-consistency self-check that guards against the exp3426
        bug where per-sample answer extraction silently produced all-null
        predictions (and therefore a 0.0 self-consistency accuracy), and
      * the terminal-verdict mapping (a partial corpus is a SUCCESS, not a
        failure — the builder simply resumes next milestone).

    Keeping this logic GPU-free means a reviewer (or CI) can exercise every
    scientific decision the builder makes without loading a 26B model, which is
    exactly the reproducibility property the adversarial-verify discipline asks
    for. The slow live-LLM I/O (loading the GGUF, sampling, capturing logprobs)
    stays in the experiment script.

Spec: REQ-KONA-3448 (resumable generation-corpus builder), SCENARIO-KONA-3448,
SCENARIO-KONA-3448-RESUME, SCENARIO-KONA-3448-BLOCKED.

What we are approximating (honest-heuristic disclosure, per CLAUDE.md Verifier
Authenticity Discipline): the downstream self-certainty Best-of-N selector
(arXiv:2502.18581) needs the per-token chosen-token logprobs. llama.cpp exposes
only the chosen-token logprob (not the full per-step distribution), so each
generation stores the list of chosen-token logprobs plus their mean. The mean is
a faithful monotone proxy for sequence confidence; the scoring task converts the
stored logprobs to ``mean_token_confidence`` exactly as exp3426 did.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

# Re-use the already-tested GSM8K answer-extraction + scoring primitives so the
# builder and the scoring task agree on exactly one extraction surface. These
# are unchanged from the v1/v2 premise modules.
from carnot.phase3.energy_descent_premise import (  # noqa: F401
    extract_final_answer,
    is_correct,
    majority_vote,
)

# Target shape of the corpus the builder accumulates toward. These are the
# defaults the experiment script uses; the module functions accept the values
# as arguments so tests can drive smaller sizes.
DEFAULT_N_PROBLEMS = 120
DEFAULT_K_SAMPLES = 6
WARMUP_MIN_PROBLEMS = 20
WARMUP_SC_FLOOR = 0.30


@dataclass(frozen=True)
class GenerationSample:
    """One generated solution (greedy or sampled) with everything scoring needs.

    ``text`` is the raw model output; ``answer`` is the extracted integer (or
    ``None`` when the generation produced no parseable number — that candidate
    scores as a miss downstream rather than crashing the run). ``token_logprobs``
    is the list of chosen-token logprobs llama.cpp returned (empty when the model
    refused to emit them), and ``mean_token_logprob`` is their mean — the
    self-certainty proxy. ``n_tokens`` records how many scorable tokens the
    generation had so a reviewer can spot truncated/degenerate generations.
    """

    text: str
    answer: int | None
    token_logprobs: list[float]
    mean_token_logprob: float | None
    n_tokens: int

    def to_dict(self) -> dict:
        """JSON-serialisable form written into a corpus row."""
        return {
            "text": self.text,
            "answer": self.answer,
            "token_logprobs": self.token_logprobs,
            "mean_token_logprob": self.mean_token_logprob,
            "n_tokens": self.n_tokens,
        }


def mean_logprob(token_logprobs: list[float] | None) -> float | None:
    """Mean of the finite chosen-token logprobs, or ``None`` when there are none.

    We average the raw logprobs (not their exponentials) so the stored scalar is
    a stable, scale-free confidence summary; the downstream self-certainty
    selector can recover ``mean_token_confidence`` by exponentiating the stored
    per-token list. ``None`` (rather than 0.0) signals "no scorable tokens" so a
    generation that emitted no logprobs is never mistaken for a maximally
    confident one.
    """
    if not token_logprobs:
        return None
    finite = [lp for lp in token_logprobs if lp is not None and math.isfinite(lp)]
    if not finite:
        return None
    return sum(finite) / len(finite)


def make_sample(text: str, token_logprobs: list[float] | None) -> GenerationSample:
    """Build a :class:`GenerationSample` from a raw generation + its logprobs.

    Centralises answer extraction and mean-logprob computation so the experiment
    script never re-implements them (and so a test exercises exactly the code the
    live run uses). A missing/empty logprob list yields an empty stored list, a
    ``None`` mean, and ``n_tokens == 0`` — the honest "no confidence signal"
    state.
    """
    finite = [lp for lp in (token_logprobs or []) if lp is not None and math.isfinite(lp)]
    return GenerationSample(
        text=text,
        answer=extract_final_answer(text),
        token_logprobs=finite,
        mean_token_logprob=mean_logprob(finite),
        n_tokens=len(finite),
    )


def build_corpus_row(
    *,
    problem_id: str,
    question: str,
    gold: int,
    greedy: GenerationSample,
    samples: list[GenerationSample],
    temperature: float,
) -> dict:
    """Pack one fully-generated problem into an append-only JSONL row.

    The row is the exact contract the scoring task (exp3449) reads: the problem
    identity + gold label, the single greedy generation, and the ``k`` sampled
    generations. Each generation carries its raw text, extracted answer, and
    per-token logprobs, so the scoring task can compute self-consistency,
    self-certainty Best-of-N, and energy selection without any live model.
    ``k`` is recorded explicitly so a reader can detect a short/interrupted row.
    """
    return {
        "problem_id": problem_id,
        "question": question,
        "gold": gold,
        "greedy": greedy.to_dict(),
        "samples": [s.to_dict() for s in samples],
        "k": len(samples),
        "temperature": temperature,
    }


def row_is_complete(row: dict, *, k_samples: int) -> bool:
    """True iff a parsed corpus row carries a greedy generation and ``k`` samples.

    The resume contract only skips a problem when its row is COMPLETE; a row that
    was somehow truncated (fewer than ``k`` samples, or no greedy generation) is
    treated as not-done so a re-invocation regenerates it cleanly rather than
    feeding a half-built row to the scoring task.
    """
    if "problem_id" not in row or "gold" not in row:
        return False
    if not isinstance(row.get("greedy"), dict):
        return False
    samples = row.get("samples")
    return isinstance(samples, list) and len(samples) >= k_samples


def read_corpus_rows(jsonl_path: str | Path) -> list[dict]:
    """Read every well-formed JSON row from the corpus file (empty if absent).

    Malformed/partial trailing lines (e.g. a row half-written when a previous
    process was killed mid-append) are skipped rather than raising, so a single
    bad line never blocks a resume. Missing file → empty list, which the builder
    treats as "fresh start".
    """
    path = Path(jsonl_path)
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A truncated final line from an interrupted append — ignore it.
            continue
    return rows


def completed_problem_ids(jsonl_path: str | Path, *, k_samples: int) -> set[str]:
    """Return the set of problem ids that already have a COMPLETE row on disk.

    This is the heart of the resume contract: the builder loads this set and
    skips every problem in it, generating only for the remainder. Only complete
    rows count (see :func:`row_is_complete`), so an interrupted partial row does
    not cause a problem to be silently skipped with missing generations.
    """
    ids: set[str] = set()
    for row in read_corpus_rows(jsonl_path):
        if row_is_complete(row, k_samples=k_samples) and "problem_id" in row:
            ids.add(str(row["problem_id"]))
    return ids


@dataclass(frozen=True)
class WarmupSelfCheck:
    """Result of the warm-up self-consistency / per-sample-extraction self-check.

    ``non_degenerate`` is the gate: True iff self-consistency accuracy ``>=``
    greedy accuracy AND ``> WARMUP_SC_FLOOR``. When it is False the per-sample
    answer extraction is suspect (the exp3426 0.0 bug), so ``examples`` carries
    the raw extracted answers of three problems for diagnosis — but the builder
    keeps generating regardless (the corpus is still useful; the scoring task
    re-validates this gate on the full corpus).
    """

    non_degenerate: bool
    self_consistency_accuracy: float
    greedy_accuracy: float
    n_problems: int
    examples: list[dict]


def warmup_self_consistency_check(
    rows: list[dict],
    *,
    min_problems: int = WARMUP_MIN_PROBLEMS,
    sc_floor: float = WARMUP_SC_FLOOR,
) -> WarmupSelfCheck:
    """Compute the warm-up self-consistency vs greedy accuracies over the corpus.

    Over the available completed rows (only meaningful once ``>= min_problems``),
    we recompute self-consistency (majority vote over the per-sample extracted
    answers) and greedy accuracy directly from the stored answers — exactly what
    the scoring task will do — so a broken extraction surfaces HERE, early, with
    diagnosable examples, instead of silently producing a useless corpus.

    With fewer than ``min_problems`` rows we cannot yet judge degeneracy, so we
    report ``non_degenerate=False`` with the accuracies computed so far and no
    examples; the experiment script treats "not enough data yet" as "keep going",
    distinct from a confirmed-broken extraction.
    """
    n = len(rows)
    sc_correct = 0
    greedy_correct = 0
    examples: list[dict] = []
    for row in rows:
        gold = row.get("gold")
        greedy = row.get("greedy") or {}
        samples = row.get("samples") or []
        sample_answers = [s.get("answer") for s in samples]
        sc_pred = majority_vote(sample_answers)
        greedy_pred = greedy.get("answer")
        if isinstance(gold, int):
            if is_correct(sc_pred, gold):
                sc_correct += 1
            if is_correct(greedy_pred, gold):
                greedy_correct += 1
        if len(examples) < 3:
            examples.append(
                {
                    "problem_id": row.get("problem_id"),
                    "gold": gold,
                    "greedy_answer": greedy_pred,
                    "sample_answers": sample_answers,
                    "majority_vote": sc_pred,
                }
            )

    sc_acc = sc_correct / n if n else 0.0
    greedy_acc = greedy_correct / n if n else 0.0
    if n < min_problems:
        # Not enough warm-up data to judge degeneracy yet.
        return WarmupSelfCheck(
            non_degenerate=False,
            self_consistency_accuracy=round(sc_acc, 4),
            greedy_accuracy=round(greedy_acc, 4),
            n_problems=n,
            examples=[],
        )
    non_degenerate = (sc_acc >= greedy_acc) and (sc_acc > sc_floor)
    return WarmupSelfCheck(
        non_degenerate=non_degenerate,
        self_consistency_accuracy=round(sc_acc, 4),
        greedy_accuracy=round(greedy_acc, 4),
        n_problems=n,
        # Examples only matter when the gate failed — keep them for diagnosis.
        examples=[] if non_degenerate else examples,
    )


def derive_corpus_verdict(n_completed: int, n_target: int) -> str:
    """Map (completed, target) to exactly one terminal `complete:`-prefixed verdict.

    A partial corpus is a SUCCESS, not a failure — the whole point of the
    decoupled builder is that it accumulates across milestones. The three bands
    mirror the exp3448 spec:

      * ``n_completed >= n_target``   -> corpus complete.
      * ``30 <= n_completed < target`` -> partial but resumable (CLT-minimum met).
      * ``n_completed < 30``          -> seeded; resume next milestone.

    Every branch is `complete:`-prefixed so the conductor reconciler classifies
    the run as terminal regardless of how far the corpus got.
    """
    if n_completed >= n_target:
        return f"complete: p01_generation_corpus_complete_n={n_completed}"
    if n_completed >= 30:
        return f"complete: p01_generation_corpus_partial_resumable_n={n_completed}"
    return (
        f"complete: p01_generation_corpus_seeded_n={n_completed}_resume_next_milestone"
    )


def corpus_reproducibility_checksum(
    *,
    corpus_path: str | Path,
    model_path: str,
    seed: int,
    n_target: int,
    k_samples: int,
) -> str:
    """Content hash over the GSM8K split file + model + seed + (n, k) budget.

    A third party who re-runs with the same source corpus, the same GGUF, the
    same seed, and the same target budget gets the same checksum — the audit
    trail the adversarial-verify discipline requires for a compute-bound
    artifact. We hash the source corpus *contents* (not just its path) so a
    swapped-out corpus is detectable; the model path string stands in for the
    weights (hashing a 16GB GGUF every run is wasteful and the path encodes the
    quant + snapshot).
    """
    hasher = hashlib.sha256()
    hasher.update(Path(corpus_path).read_bytes())
    hasher.update(
        f"|model={model_path}|seed={seed}|n={n_target}|k={k_samples}".encode()
    )
    return hasher.hexdigest()[:16]
