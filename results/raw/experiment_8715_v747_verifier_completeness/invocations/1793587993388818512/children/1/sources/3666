"""Hard-math HEADROOM generation-corpus helpers (P0.1 / exp3471).

**Why this module exists (plain-language summary):**
    P0.1 — "does energy-based selection/voting on continuous latents BEAT plain
    self-consistency at equal compute?" — is the single most important test in
    the project, and it kept coming back a TIE. The reason was not the selector:
    it was the benchmark. On GSM8K a strong model gets self-consistency (SC)
    accuracy ~0.91 — almost a CEILING — so there is simply no room left for ANY
    re-ranker to add accuracy on top of the majority vote (exp3460 measured the
    trained-energy vote tying SC exactly). You can only show a selector helps
    where SC has HEADROOM: where SC is materially below 1.0 so that some of the
    samples the majority vote gets wrong could be rescued by a better chooser.

    The process-reward literature (e.g. arXiv:2602.11570 "PRIME", +8-9% on AIME
    from process-aware verification) shows the regime where verifier selection
    beats SC is HARD math, scored as a PROCESS reward over the per-STEP reasoning
    trace rather than only over the final answer. This module holds the pure,
    GPU-free, unit-testable pieces of the corpus builder that creates exactly
    that substrate:

      * answer handling for math (extract the ``\\boxed{}`` final answer, a
        ``#### <n>`` coda, or the last number; normalise LaTeX so equivalent
        surface forms compare equal; decide correctness against a string gold);
      * the NEW capability over the GSM8K builders — splitting a chain-of-thought
        into a list of discrete reasoning STEPS, so the FoVer step-error verifier
        (the 0.9131 ensemble) can be scored as a process reward downstream;
      * the per-problem corpus-row shape the scoring task consumes;
      * the warm-up HEADROOM self-check — recomputing SC vs greedy accuracy over
        the corpus and the all-important ``self_consistency_in_headroom_band``
        boolean that decides whether the corpus is even usable for P0.1;
      * the terminal-verdict mapping (a partial corpus is a SUCCESS that resumes
        next milestone; a split whose SC falls outside the band is an honest
        ``blocked_no_headroom`` so the next run can switch splits).

    Keeping this logic GPU-free means a reviewer (or CI) can exercise every
    scientific decision the builder makes without loading a 26B model — exactly
    the reproducibility property the adversarial-verify discipline asks for. The
    slow live-LLM I/O (loading the GGUF, sampling, capturing logprobs) and the
    ``datasets`` benchmark download stay in the experiment script.

Spec: REQ-KONA-3471 (hard-math headroom corpus + per-step traces),
SCENARIO-KONA-3471, SCENARIO-KONA-3471-RESUME, SCENARIO-KONA-3471-NO-HEADROOM.

What we are approximating (honest-heuristic disclosure, per CLAUDE.md Verifier
Authenticity Discipline): ``parse_reasoning_steps`` is a TEXT-statistical step
segmenter, not a learned parser. It splits the chain-of-thought on newline
boundaries (the PRM800K / FoVer step convention) and falls back to sentence
segmentation for a single-line CoT. ``normalize_math_answer`` is the standard
MATH surface-normalisation (strip ``$``/``\\left``/spaces, fold
``\\dfrac``->``\\frac``, drop a ``\\text{}`` wrapper) — it catches common
equivalent forms, NOT full symbolic equivalence (``1/2`` vs ``0.5`` will not
match). Both are honest proxies the downstream scorer re-validates.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

# Re-use the already-tested resume/IO primitives so the hard-math builder and the
# GSM8K builders agree on exactly one corpus-IO surface (read rows, find completed
# ids, average logprobs). Only the math-specific scoring + step parsing is new.
from carnot.phase3.p01_generation_corpus import (  # noqa: F401
    completed_problem_ids,
    mean_logprob,
    read_corpus_rows,
    row_is_complete,
)

# Default corpus shape the builder accumulates toward. The functions take the
# values as arguments so tests can drive smaller sizes.
DEFAULT_N_TARGET = 80
SCORABLE_FLOOR = 40
HEADLINE_FLOOR = 80
# Below this many scored problems we cannot trust the SC estimate enough to call
# a split "no headroom" — too few to judge, so the run just resumes.
BAND_JUDGE_FLOOR = 8
# The HEADROOM band: SC must be materially below ceiling (so a selector has room
# to help) but not so low the corpus is mostly noise. arXiv:2602.11570 regime.
SC_BAND_LO = 0.40
SC_BAND_HI = 0.70
# Cap on parsed reasoning steps so a pathological generation cannot blow up a row.
MAX_REASONING_STEPS = 64

# The default hard-math benchmark: MATH (Hendrycks) Level 5, the PRM-literature
# standard for "hard math with headroom". The experiment script downloads it via
# ``datasets``; the level filter + problem build happens in ``build_math_problems``.
DEFAULT_BENCHMARK_ID = "EleutherAI/hendrycks_math:test:Level 5"


# ---------------------------------------------------------------------------
# Math answer extraction + correctness
# ---------------------------------------------------------------------------

_HASH_CODA = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)
_ANY_NUMBER = re.compile(r"-?\$?[0-9][0-9,]*(?:\.[0-9]+)?")


def extract_boxed(text: str) -> str | None:
    """Return the contents of the LAST ``\\boxed{...}`` in ``text`` (or ``None``).

    MATH ground-truth answers — and the answers we ask the model to emit — live
    inside a ``\\boxed{}``. We scan for the last ``\\boxed`` occurrence (the model
    may show intermediate boxes; the final one is the declared answer) and walk
    the braces with a depth counter so a nested ``\\frac{a}{b}`` inside the box is
    captured whole rather than truncated at the first ``}``. Returns ``None`` when
    there is no ``\\boxed{`` at all, so the caller can fall back to another coda.
    """
    if not text:
        return None
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    i = idx + len("\\boxed")
    # Allow whitespace between \boxed and its opening brace.
    while i < len(text) and text[i] == " ":
        i += 1
    if i >= len(text) or text[i] != "{":
        # A "\boxed 5" form with no brace — not the shape we emit; give up.
        return None
    depth = 0
    start = i
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : j]
    # Unbalanced braces (truncated generation) — return what we have after '{'.
    return text[start + 1 :]


def extract_math_answer(text: str) -> str | None:
    """Extract the model's declared final answer from a math chain-of-thought.

    Preference order, most-to-least explicit:
      1. the last ``\\boxed{...}`` (what the prompt asks the model to emit);
      2. a ``#### <answer>`` coda (the GSM8K-style fallback some models default to);
      3. the last standalone number in the text (last-ditch heuristic).

    Returns ``None`` only when none of those is present, so a no-answer generation
    scores as a miss downstream rather than crashing the run. The returned value
    is the RAW surface string; comparison goes through :func:`normalize_math_answer`.
    """
    if not text:
        return None
    boxed = extract_boxed(text)
    if boxed is not None and boxed.strip():
        return boxed.strip()
    coda = list(_HASH_CODA.finditer(text))
    if coda:
        candidate = coda[-1].group(1).strip()
        if candidate:
            return candidate
    numbers = _ANY_NUMBER.findall(text)
    if numbers:
        return numbers[-1].strip()
    return None


def normalize_math_answer(answer: str | None) -> str | None:
    """Normalise a math answer's LaTeX surface form for equality comparison.

    This is the standard MATH surface normalisation (Hendrycks / lm-eval): strip
    the cosmetic LaTeX that two correct answers can differ by — ``$`` delimiters,
    ``\\left``/``\\right``, spacing macros, a ``\\text{}`` wrapper, ``\\dfrac`` vs
    ``\\frac`` — remove commas inside numbers, and trim wrapping braces and a
    trailing period. It is deliberately NOT a symbolic equivalence checker: it
    catches the common surface variants (``\\dfrac12`` vs ``\\frac{1}{2}`` after
    brace-stripping) but will not equate ``1/2`` with ``0.5``. Returns ``None``
    for a ``None`` input so a missing answer never compares equal to anything.
    """
    if answer is None:
        return None
    s = answer.strip()
    # Drop a single \text{...} wrapper (e.g. "\text{cm}" units) before stripping.
    s = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", s)
    # Fold display/inline fraction macros onto the plain one.
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    # Remove purely-cosmetic LaTeX tokens and delimiters.
    for tok in (
        "\\left",
        "\\right",
        "\\!",
        "\\,",
        "\\;",
        "\\:",
        "\\ ",
        "$",
        "\\$",
        "%",
        " ",
        "\n",
    ):
        s = s.replace(tok, "")
    # Numbers: drop thousands commas so "1,000" == "1000".
    s = s.replace(",", "")
    # Strip wrapping braces and a single trailing period.
    while len(s) >= 2 and s[0] == "{" and s[-1] == "}":
        s = s[1:-1]
    s = s.rstrip(".")
    # Drop a redundant leading + sign on a number ("+3" == "3").
    if s.startswith("+"):
        s = s[1:]
    return s.lower()


def math_is_correct(prediction: str | None, gold: str | None) -> bool:
    """True iff the prediction normalises to the same surface form as the gold.

    Both sides go through :func:`normalize_math_answer`; a ``None`` on either side
    (no answer extracted, or no gold) is never correct. This is the per-candidate
    correctness label the warm-up self-check and the downstream scorer both use.
    """
    if prediction is None or gold is None:
        return False
    norm_pred = normalize_math_answer(prediction)
    norm_gold = normalize_math_answer(gold)
    if not norm_pred or not norm_gold:
        return False
    return norm_pred == norm_gold


def majority_vote_str(answers: list[str | None]) -> str | None:
    """Self-consistency aggregation over NORMALISED string answers.

    The equal-compute AR control: given the same ``k`` samples the energy/process
    selector consumes, the AR-native way to aggregate is the modal answer (Wang et
    al. self-consistency). We bucket by the NORMALISED form so ``\\dfrac12`` and
    ``\\frac{1}{2}`` vote together, but return the FIRST raw surface form seen for
    the winning bucket (so the reported answer is human-readable). Ties break
    toward the bucket whose first member appeared earliest — deterministic given
    input order. Returns ``None`` when every sample failed to produce an answer.
    """
    counts: dict[str, int] = {}
    first_raw: dict[str, str] = {}
    order: list[str] = []
    for answer in answers:
        norm = normalize_math_answer(answer)
        if not norm:
            continue
        if norm not in counts:
            counts[norm] = 0
            first_raw[norm] = answer if answer is not None else norm
            order.append(norm)
        counts[norm] += 1
    if not order:
        return None
    best = max(order, key=lambda a: (counts[a], -order.index(a)))
    return first_raw[best]


# ---------------------------------------------------------------------------
# Per-step reasoning-trace parsing (the NEW .320 capability)
# ---------------------------------------------------------------------------


def parse_reasoning_steps(text: str, *, max_steps: int = MAX_REASONING_STEPS) -> list[str]:
    """Split a chain-of-thought into a list of discrete reasoning STEPS.

    This is the new capability the hard-math corpus adds over the GSM8K builders:
    the FoVer step-error verifier scores a PROCESS reward, so each generation must
    carry its reasoning broken into the units the verifier judges. We use the
    PRM800K / FoVer convention — a step boundary is a newline — because that is
    how the model lays out its reasoning and how step-level corpora are labelled.
    For a degenerate single-line CoT (no newlines) we fall back to sentence
    segmentation on ``.``/``!``/``?`` so the verifier still gets more than one
    unit to score. Blank fragments are dropped and the list is capped at
    ``max_steps`` so a pathological generation cannot blow up a corpus row.
    Returns ``[]`` for empty text.
    """
    if not text or not text.strip():
        return []
    raw = [line.strip() for line in text.split("\n")]
    steps = [line for line in raw if line]
    if len(steps) <= 1:
        # Single-line CoT — segment on sentence terminators instead.
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        steps = [part.strip() for part in parts if part.strip()]
    return steps[:max_steps]


# ---------------------------------------------------------------------------
# Corpus-row shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeadroomSample:
    """One generated solution (greedy or sampled) with everything scoring needs.

    ``text`` is the raw model output; ``answer`` is the extracted ``\\boxed{}``
    surface string (or ``None`` when nothing parseable was produced — that
    candidate scores as a miss downstream rather than crashing the run).
    ``steps`` is the parsed per-step reasoning trace the process verifier scores.
    ``token_logprobs`` is the chosen-token logprobs llama.cpp returned and
    ``mean_token_logprob`` their mean (the self-certainty proxy). ``n_tokens``
    records how many scorable tokens the generation had so a reviewer can spot a
    truncated/degenerate generation.
    """

    text: str
    answer: str | None
    steps: list[str]
    token_logprobs: list[float]
    mean_token_logprob: float | None
    n_tokens: int

    def to_dict(self) -> dict:
        """JSON-serialisable form written into a corpus row."""
        return {
            "text": self.text,
            "answer": self.answer,
            "steps": list(self.steps),
            "n_steps": len(self.steps),
            "token_logprobs": self.token_logprobs,
            "mean_token_logprob": self.mean_token_logprob,
            "n_tokens": self.n_tokens,
        }


def make_headroom_sample(
    text: str, token_logprobs: list[float] | None
) -> HeadroomSample:
    """Build a :class:`HeadroomSample` from a raw generation + its logprobs.

    Centralises answer extraction, step parsing, and mean-logprob computation so
    the experiment script never re-implements them (and a test exercises exactly
    the code the live run uses). A missing/empty logprob list yields an empty
    stored list, a ``None`` mean, and ``n_tokens == 0`` — the honest "no
    confidence signal" state. Non-finite logprobs are dropped so a single ``inf``
    cannot poison the mean.
    """
    import math  # noqa: PLC0415 — tiny, kept local so the module import stays light

    finite = [
        lp for lp in (token_logprobs or []) if lp is not None and math.isfinite(lp)
    ]
    return HeadroomSample(
        text=text,
        answer=extract_math_answer(text),
        steps=parse_reasoning_steps(text),
        token_logprobs=finite,
        mean_token_logprob=mean_logprob(finite),
        n_tokens=len(finite),
    )


def build_headroom_row(
    *,
    problem_id: str,
    question: str,
    gold: str,
    level: str,
    greedy: HeadroomSample,
    samples: list[HeadroomSample],
    temperature: float,
) -> dict:
    """Pack one fully-generated hard-math problem into an append-only JSONL row.

    The row is the exact contract the scoring task (exp3472/3473/3475) reads: the
    problem identity + the (string) gold answer + its difficulty ``level``, the
    single greedy generation, and the ``k`` sampled generations. Each generation
    carries its raw text, extracted answer, per-step trace, and per-token logprobs
    so the scorer can compute self-consistency, self-certainty Best-of-N, and the
    PROCESS-reward energy selection without any live model. ``k`` is recorded so a
    reader can detect a short/interrupted row.
    """
    return {
        "problem_id": problem_id,
        "question": question,
        "gold": gold,
        "level": level,
        "greedy": greedy.to_dict(),
        "samples": [s.to_dict() for s in samples],
        "k": len(samples),
        "temperature": temperature,
    }


# ---------------------------------------------------------------------------
# Warm-up headroom self-check
# ---------------------------------------------------------------------------


def sc_in_headroom_band(
    sc_accuracy: float, *, lo: float = SC_BAND_LO, hi: float = SC_BAND_HI
) -> bool:
    """True iff self-consistency accuracy lands in the closed headroom band.

    The band is the whole point of this corpus: a selector can only beat SC where
    SC is materially below ceiling (so there is accuracy left to recover) and not
    so low the corpus is mostly noise. ``lo``/``hi`` are parameters so a test can
    drive a narrower band.
    """
    return lo <= sc_accuracy <= hi


@dataclass(frozen=True)
class HeadroomWarmup:
    """Result of the warm-up SC-vs-greedy self-check over the corpus.

    ``in_band`` is the headline gate (G1 HEADROOM-CONFIRMED): True iff SC accuracy
    lands in ``[SC_BAND_LO, SC_BAND_HI]`` AND there were enough problems to judge.
    ``examples`` carries three problems' raw extracted answers for diagnosis.
    """

    self_consistency_accuracy: float
    greedy_accuracy: float
    n_problems: int
    in_band: bool
    examples: list[dict]


def headroom_warmup_check(
    rows: list[dict],
    *,
    min_problems: int = BAND_JUDGE_FLOOR,
    band_lo: float = SC_BAND_LO,
    band_hi: float = SC_BAND_HI,
) -> HeadroomWarmup:
    """Compute SC vs greedy accuracy + the headroom-band boolean over the corpus.

    For each completed row we recompute self-consistency (majority vote over the
    per-sample extracted answers) and greedy accuracy directly from the stored
    answers — exactly what the downstream scorer will do — so a broken extraction
    or a no-headroom split surfaces HERE, with diagnosable examples, instead of
    silently producing a useless corpus. ``in_band`` requires BOTH a SC inside the
    band AND ``>= min_problems`` rows (we will not call a split "no headroom" off
    a handful of problems). Returns the accuracies computed so far regardless.
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
        sc_pred = majority_vote_str(sample_answers)
        greedy_pred = greedy.get("answer")
        if math_is_correct(sc_pred, gold):
            sc_correct += 1
        if math_is_correct(greedy_pred, gold):
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
    sc_acc = round(sc_correct / n, 4) if n else 0.0
    greedy_acc = round(greedy_correct / n, 4) if n else 0.0
    in_band = (n >= min_problems) and sc_in_headroom_band(
        sc_acc, lo=band_lo, hi=band_hi
    )
    return HeadroomWarmup(
        self_consistency_accuracy=sc_acc,
        greedy_accuracy=greedy_acc,
        n_problems=n,
        in_band=in_band,
        examples=examples,
    )


# ---------------------------------------------------------------------------
# Verdict + gates
# ---------------------------------------------------------------------------


def derive_headroom_verdict(
    n_completed: int,
    sc_accuracy: float,
    in_band: bool,
    *,
    n_completed_to_judge: int = BAND_JUDGE_FLOOR,
    scorable_floor: int = SCORABLE_FLOOR,
    headline_floor: int = HEADLINE_FLOOR,
) -> str:
    """Map (completed, SC, in-band) to exactly one terminal `complete:` verdict.

    A partial corpus is a SUCCESS — the decoupled builder accumulates across
    milestones — so every branch is `complete:`-prefixed (Verdict Terminal-Prefix
    Discipline). The bands mirror the exp3471 spec:

      * too few problems to judge the band yet (``n < n_completed_to_judge``) ->
        a partial that resumes next milestone (we do NOT blame the split off a
        handful of problems);
      * enough problems but SC OUTSIDE the band -> an honest
        ``blocked_no_headroom`` so the next run switches to a harder/easier split;
      * in band, ``n >= headline_floor`` -> HEADLINE-eligible (exp3472 can report
        a headline process-reward verdict);
      * in band, ``scorable_floor <= n < headline_floor`` -> scorable-partial
        (enough for a preliminary verdict, resume toward headline);
      * in band, below the scorable floor -> partial, resume next milestone.
    """
    sc_str = f"{sc_accuracy:.3f}"
    if n_completed < n_completed_to_judge:
        return (
            f"complete: p01_headroom_corpus_partial_n={n_completed}"
            f"_resume_next_milestone"
        )
    if not in_band:
        return "complete: blocked_no_headroom_benchmark_sc_outside_band"
    if n_completed >= headline_floor:
        return (
            f"complete: p01_headroom_corpus_headline_eligible_n={n_completed}"
            f"_sc={sc_str}"
        )
    if n_completed >= scorable_floor:
        return (
            f"complete: p01_headroom_corpus_scorable_partial_n={n_completed}"
            f"_resume_next_milestone"
        )
    return (
        f"complete: p01_headroom_corpus_partial_n={n_completed}_resume_next_milestone"
    )


def headroom_acceptance_gates(
    in_band: bool,
    n_completed: int,
    per_step_traces_captured: bool,
    *,
    scorable_floor: int = SCORABLE_FLOOR,
) -> dict[str, bool]:
    """Report the two named acceptance gates for the headroom corpus.

    * **G1 HEADROOM-CONFIRMED** — SC landed in ``[0.4, 0.7]``; without it exp3472
      cannot test the premise (the property GSM8K lacked at SC~0.91).
    * **G2 SCORABLE** — enough problems (``>= scorable_floor``) carry per-step
      traces for exp3472 to report at least a preliminary process-energy-vs-SC
      verdict.
    """
    return {
        "g1_headroom_confirmed": bool(in_band),
        "g2_scorable": (n_completed >= scorable_floor) and bool(per_step_traces_captured),
    }


# ---------------------------------------------------------------------------
# Benchmark loading (pure half; the datasets download lives in the script)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MathProblem:
    """One hard-math problem with its (string) gold answer and difficulty level.

    ``problem_id`` is a deterministic content hash so the resume contract is
    stable across runs; ``question`` is fed to the model verbatim; ``answer`` is
    the ``\\boxed{}`` gold extracted from the dataset's solution; ``level`` is the
    MATH difficulty tier (kept on every corpus row for auditability).
    """

    problem_id: str
    question: str
    answer: str
    level: str


def build_math_problems(
    records,
    *,
    levels: set[str],
    n: int,
    seed: int,
) -> list[MathProblem]:
    """Build a deterministic ``n``-problem slice of MATH at the requested levels.

    ``records`` is any iterable of dataset rows (dicts) with ``problem``,
    ``level``, and ``solution`` fields — the experiment script passes the
    ``datasets`` rows; tests pass plain dicts, so this stays GPU/datasets-free.
    We keep only rows whose ``level`` is in ``levels`` AND whose solution carries a
    parseable ``\\boxed{}`` gold (a problem with no extractable answer cannot be
    scored, so it is dropped rather than silently mislabelled). Problems are
    deduplicated on their content hash, shuffled once with a seeded RNG, and the
    first ``n`` are returned — a pure function of ``(records, levels, n, seed)``,
    which is also what the reproducibility checksum hashes over. Returns fewer
    than ``n`` (possibly empty) when the filtered pool is smaller; the caller
    decides whether that is enough.
    """
    import random  # noqa: PLC0415 — kept local so the module import stays light

    pool: list[MathProblem] = []
    seen: set[str] = set()
    for record in records:
        level = str(record.get("level", ""))
        if level not in levels:
            continue
        question = record.get("problem")
        solution = record.get("solution") or ""
        if not question:
            continue
        gold = extract_boxed(solution)
        if gold is None or not gold.strip():
            continue
        pid = "math-" + hashlib.sha1(str(question).encode("utf-8")).hexdigest()[:12]
        if pid in seen:
            continue
        seen.add(pid)
        pool.append(
            MathProblem(
                problem_id=pid,
                question=str(question),
                answer=gold.strip(),
                level=level,
            )
        )
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n]


def headroom_reproducibility_checksum(
    *,
    benchmark_id: str,
    model_path: str,
    seed: int,
    n_target: int,
    k_samples: int,
    levels: set[str],
) -> str:
    """Content hash over the benchmark id + model + seed + (n, k, levels) budget.

    A third party who re-runs against the same benchmark split, the same GGUF, the
    same seed, and the same budget gets the same checksum — the audit trail the
    adversarial-verify discipline requires for a compute-bound artifact. The
    benchmark is identified by its stable ``datasets`` id + split + level rather
    than by hashing a multi-GB on-disk arrow cache; the model path string stands
    in for the weights (it encodes the quant + snapshot).
    """
    hasher = hashlib.sha256()
    payload = (
        f"benchmark={benchmark_id}|model={model_path}|seed={seed}"
        f"|n={n_target}|k={k_samples}|levels={','.join(sorted(levels))}"
    )
    hasher.update(payload.encode("utf-8"))
    return hasher.hexdigest()[:16]


def corpus_problem_ids(jsonl_path: str | Path, *, k_samples: int) -> set[str]:
    """Thin alias around :func:`completed_problem_ids` for the hard-math corpus.

    Exists so the experiment script imports its resume contract from this module
    (one import surface for everything hard-math) while reusing the already-tested
    GSM8K row-completeness logic underneath.
    """
    return completed_problem_ids(jsonl_path, k_samples=k_samples)
