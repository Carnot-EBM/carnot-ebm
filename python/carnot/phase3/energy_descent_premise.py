"""Energy-descent-vs-autoregressive premise test helpers (P0.1 / exp3312).

**Why this module exists (plain-language summary):**
    The whole Phase-3 / Kona endgame bets that *energy-descent reasoning on a
    continuous latent* is a better way to reason than *autoregressive (AR) token
    sampling*.  That bet has never been settled on a real task — only on toy
    5x5 grids.  This module holds the pure, deterministic, unit-testable pieces
    of the head-to-head experiment so that the experiment script (which does the
    heavy live-LLM I/O) stays thin and every scientific decision is covered by a
    test.

    The split of responsibilities is deliberate:
      * The *experiment script* loads the GGUF model, generates text, and trains
        the energy substrate — all the slow, environment-dependent work.
      * *This module* turns raw model outputs into a verdict: it loads the
        corpus, extracts answers, runs the bounded-depth energy descent that
        SELECTS an answer without sampling any tokens, and computes the paired
        significance test that decides whether the premise holds.

    Keeping the judgement logic here means a reviewer can re-run the gate on the
    saved per-problem outcomes without a GPU, which is exactly the
    reproducibility property the adversarial-verify discipline asks for.

Spec: REQ-KONA-3312 (premise test), SCENARIO-KONA-3312, SCENARIO-KONA-3312-BLOCKED.
Reasoning-mode invariant: REQ-KONA-001 (no token sampling inside the latent
refinement loop) and REQ-KONA-002 (bounded-depth refinement).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GsmProblem:
    """One real GSM8K problem with its integer ground-truth answer.

    ``problem_id`` lets us audit exactly which problems were scored; ``question``
    is fed to both conditions verbatim (paired), and ``answer`` is the integer
    the extracted prediction is checked against.
    """

    problem_id: str
    question: str
    answer: int


def load_gsm8k_subset(
    path: str | Path,
    *,
    n: int = 200,
    seed: int = 3312,
) -> list[GsmProblem]:
    """Load a deterministic ``n``-problem held-out slice of real GSM8K.

    The corpus file is the JSONL produced by exp281 — each line carries the
    *original* (unmodified) GSM8K question and its integer answer alongside a
    synthetic adversarial variant.  We deliberately use the ``original_question``
    / ``original_answer`` fields: those are the genuine GSM8K items with
    human-verified integer labels, which is what the premise test needs (the
    adversarial variants are a different study).

    Determinism matters because the premise test is *paired* — both conditions
    must see the exact same problems in the exact same order.  We seed a local
    RNG, shuffle the full pool once, and take the first ``n``.  The returned list
    is therefore a pure function of ``(path, n, seed)``, which is also what the
    reproducibility checksum hashes over.

    Raises ``ValueError`` if fewer than ``n`` problems are available, because a
    silently-truncated corpus would make the ``n>=200`` CLT guarantee a lie.
    """

    rows: list[GsmProblem] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            question = record.get("original_question")
            answer = record.get("original_answer")
            pid = str(record.get("question_id", f"row-{len(rows)}"))
            if question is None or answer is None:
                continue
            try:
                answer_int = int(answer)
            except (TypeError, ValueError):
                continue
            rows.append(GsmProblem(problem_id=pid, question=str(question), answer=answer_int))

    # Deduplicate on problem_id so a corpus with repeated originals can't inflate n.
    seen: set[str] = set()
    unique: list[GsmProblem] = []
    for row in rows:
        if row.problem_id in seen:
            continue
        seen.add(row.problem_id)
        unique.append(row)

    if len(unique) < n:
        raise ValueError(
            f"corpus {path} has only {len(unique)} unique problems; need n={n} "
            f"for a CLT-valid accuracy delta"
        )

    rng = random.Random(seed)
    rng.shuffle(unique)
    return unique[:n]


# ---------------------------------------------------------------------------
# Answer extraction + scoring
# ---------------------------------------------------------------------------

_HASH_ANSWER = re.compile(r"####\s*(-?\$?[0-9][0-9,]*(?:\.[0-9]+)?)")
_ANY_NUMBER = re.compile(r"-?\$?[0-9][0-9,]*(?:\.[0-9]+)?")


def _to_number(token: str) -> float | None:
    """Parse a GSM8K-style numeric token (commas / $ / trailing dot stripped)."""

    cleaned = token.replace(",", "").replace("$", "").rstrip(".")
    if cleaned in ("", "-"):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_final_answer(text: str) -> int | None:
    """Extract the model's final integer answer from a generated CoT.

    GSM8K answers are integers.  We first look for the canonical ``#### <number>``
    coda that the prompt asks the model to emit — that is the model's *declared*
    final answer and is the least ambiguous signal.  If the model never emitted
    the coda (common when a generation is truncated mid-thought), we fall back to
    the last standalone number in the text, which is the conventional GSM8K
    extraction heuristic.

    Returns ``None`` when no number is present at all, so the caller can score a
    no-answer generation as a miss rather than crashing.  We round to the nearest
    integer because GSM8K ground truth is integer-valued; a model that writes
    ``18.0`` should match a gold answer of ``18``.
    """

    if not text:
        return None
    matches = list(_HASH_ANSWER.finditer(text))
    if matches:
        value = _to_number(matches[-1].group(1))
        return int(round(value)) if value is not None else None
    numbers = _ANY_NUMBER.findall(text)
    for token in reversed(numbers):
        value = _to_number(token)
        if value is not None:
            return int(round(value))
    return None


def is_correct(prediction: int | None, gold: int) -> bool:
    """A prediction is correct iff it is a number equal to the gold integer."""

    return prediction is not None and prediction == gold


def majority_vote(answers: list[int | None]) -> int | None:
    """Self-consistency aggregation: the most common non-null extracted answer.

    This is the *equal-compute* AR control: given the same N samples the
    energy-descent condition consumes, the AR-native way to aggregate them is to
    take the modal answer (Wang et al. self-consistency).  Reporting it next to
    the energy-descent number is what keeps the compute comparison honest — if
    energy selection only beats single-greedy but not majority vote, the artifact
    must say so.  Ties break toward the answer that appears earliest, which is
    deterministic given the input order.
    """

    counts: dict[int, int] = {}
    order: list[int] = []
    for answer in answers:
        if answer is None:
            continue
        if answer not in counts:
            counts[answer] = 0
            order.append(answer)
        counts[answer] += 1
    if not order:
        return None
    best = max(order, key=lambda a: (counts[a], -order.index(a)))
    return best


# ---------------------------------------------------------------------------
# Energy descent (the non-AR reasoning condition)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnergyDescentResult:
    """Outcome of selecting one candidate by bounded-depth latent energy descent.

    ``selected_index`` is the chosen candidate; ``initial_energies`` /
    ``final_energies`` are the per-candidate energy before and after descent, so
    a reviewer can confirm the descent actually lowered energy (it should) and
    that selection used the *refined* latent, not the raw embedding.
    """

    selected_index: int
    initial_energies: list[float]
    final_energies: list[float]
    n_steps: int


def energy_descent_select(
    candidate_texts: list[str],
    energy_model: torch.nn.Module,
    *,
    visible_dim: int = 16,
    n_steps: int = 8,
    lr: float = 0.05,
    embed_fn=None,
) -> EnergyDescentResult:
    """Select an answer by descending a trained energy over each candidate latent.

    This is the heart of the non-AR condition and the operationalisation of
    REQ-KONA-001/002: each candidate's reasoning text is projected into a
    *continuous* latent ``z`` (the Boltzmann-GPT visible-feature space), and we
    run a **bounded number** of gradient-descent steps on ``z`` to minimise the
    learned verifier energy ``E(z)``.  Crucially, **no tokens are sampled inside
    this loop** — the refinement happens entirely in the continuous latent, and
    a discrete answer is only read off (decoded) at the coda by picking the
    candidate whose refined latent reached the lowest energy basin.

    The energy model is trained contrastively so that correct-looking reasoning
    has *low* energy; therefore the minimum-energy candidate is the one the
    learned manifold judges most correct.  We descend every candidate the same
    number of steps with the same learning rate (comparable compute per
    candidate) and select the global argmin of the refined energies.

    ``embed_fn`` defaults to the Boltzmann-GPT ``embed_texts`` projection but is
    injectable so tests can drive the selector with a trivial deterministic
    embedding.  Returns the full energy trajectory summary for auditability.
    """

    if not candidate_texts:
        raise ValueError("energy_descent_select requires at least one candidate")
    if embed_fn is None:
        from carnot.phase3.boltzmann_gpt import embed_texts as embed_fn  # noqa: PLC0415

    base = embed_fn(candidate_texts, visible_dim=visible_dim).detach().clone()

    initial: list[float] = []
    final: list[float] = []
    for row in range(base.shape[0]):
        z = base[row : row + 1].clone().requires_grad_(True)
        with torch.no_grad():
            initial.append(float(energy_model(z).item()))
        optimizer = torch.optim.SGD([z], lr=lr)
        for _ in range(n_steps):
            optimizer.zero_grad()
            energy = energy_model(z).sum()
            energy.backward()
            optimizer.step()
        with torch.no_grad():
            final.append(float(energy_model(z).item()))

    selected = min(range(len(final)), key=lambda i: final[i])
    return EnergyDescentResult(
        selected_index=selected,
        initial_energies=initial,
        final_energies=final,
        n_steps=n_steps,
    )


# ---------------------------------------------------------------------------
# Paired significance
# ---------------------------------------------------------------------------


def _binom_two_sided_p(b: int, c: int) -> float:
    """Exact two-sided binomial p-value for McNemar's discordant counts.

    With ``b`` and ``c`` the two discordant cell counts, under the null each
    discordant pair is a fair coin flip.  The exact test sums binomial tail
    probability at ``p=0.5`` for outcomes at least as extreme as observed.  This
    is the right test at our sample sizes (no large-sample chi-square
    approximation needed) and avoids the continuity-correction debate.
    """

    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # Two-sided: twice the lower tail, capped at 1.0.
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
    return min(1.0, 2.0 * tail)


def mcnemar_test(ar_correct: list[bool], ed_correct: list[bool]) -> dict[str, float]:
    """Paired McNemar test on the per-problem correctness of the two conditions.

    McNemar is the correct significance test for *paired* binary outcomes: it
    looks only at the problems where the two conditions disagree.  ``b`` is the
    count where energy-descent is right and AR is wrong (energy-descent wins);
    ``c`` is where AR is right and energy-descent is wrong (AR wins).  A
    significant imbalance between ``b`` and ``c`` is the only evidence that one
    method genuinely beats the other; problems where both agree carry no signal.

    Returns the discordant counts, the exact two-sided p-value, and the signed
    direction (``+1`` energy-descent favoured, ``-1`` AR favoured, ``0`` tie) so
    the caller never has to re-derive which way the effect points.
    """

    if len(ar_correct) != len(ed_correct):
        raise ValueError("paired test requires equal-length correctness vectors")
    b = sum(1 for a, e in zip(ar_correct, ed_correct, strict=True) if e and not a)
    c = sum(1 for a, e in zip(ar_correct, ed_correct, strict=True) if a and not e)
    p_value = _binom_two_sided_p(b, c)
    direction = 0
    if b > c:
        direction = 1
    elif c > b:
        direction = -1
    return {
        "energy_descent_wins": float(b),
        "ar_wins": float(c),
        "p_value": float(p_value),
        "direction": float(direction),
    }


def paired_bootstrap_ci(
    ar_correct: list[bool],
    ed_correct: list[bool],
    *,
    n_boot: int = 2000,
    seed: int = 3312,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI on the paired accuracy delta (energy-descent − AR).

    Bootstrapping the *paired* per-problem deltas preserves the correlation
    between the two conditions (they answer the same problems), giving a tighter
    and more honest interval than treating the two accuracies as independent.
    We resample problem indices with replacement ``n_boot`` times, recompute the
    accuracy delta on each resample, and report the central ``1-alpha``
    percentile interval.  A CI whose lower bound is above 0 is independent
    corroboration of a McNemar-significant win.
    """

    if len(ar_correct) != len(ed_correct):
        raise ValueError("paired bootstrap requires equal-length vectors")
    n = len(ar_correct)
    if n == 0:
        return (0.0, 0.0)
    diffs = [
        (1.0 if e else 0.0) - (1.0 if a else 0.0)
        for a, e in zip(ar_correct, ed_correct, strict=True)
    ]
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(n_boot):
        total = 0.0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        deltas.append(total / n)
    deltas.sort()
    lo_idx = max(0, int((alpha / 2.0) * n_boot))
    hi_idx = min(n_boot - 1, int((1.0 - alpha / 2.0) * n_boot))
    return (deltas[lo_idx], deltas[hi_idx])


# ---------------------------------------------------------------------------
# Verdict mapping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PremiseVerdict:
    """The terminal classification of the premise test plus its two gates.

    ``g1_premise_viable`` is non-inferiority (energy-descent at least matches AR);
    ``g2_premise_validated`` is strict significant superiority.  ``verdict`` is
    the ``complete:``-prefixed string the conductor reconciler reads.
    """

    verdict: str
    g1_premise_viable: bool
    g2_premise_validated: bool


def derive_premise_verdict(
    ar_accuracy: float,
    energy_descent_accuracy: float,
    p_value: float,
    ci: tuple[float, float],
    *,
    direction: float,
    alpha: float = 0.05,
) -> PremiseVerdict:
    """Map measured accuracies + paired significance to a terminal verdict.

    The gates follow the exp3312 spec exactly:

      * **G2 (premise validated):** energy-descent is *strictly* more accurate
        AND the paired test is significant in energy-descent's favour
        (``p < alpha`` with a positive direction, corroborated by a CI whose
        lower bound clears 0).  This is the only outcome that justifies the
        foundation-model endgame.
      * **G1 (premise viable):** energy-descent is *non-inferior* — its accuracy
        is at least AR's, or any shortfall is not statistically significant
        (``p >= alpha``).  Below this, the non-AR mode is not even competitive.

    The verdict string is chosen so the conductor classifies it as terminal
    regardless of outcome — refutation is as publishable as validation here.
    """

    significant = p_value < alpha
    g2 = (
        energy_descent_accuracy > ar_accuracy
        and significant
        and direction > 0
        and ci[0] > 0.0
    )
    # Non-inferior: either it matches/beats AR, or the shortfall isn't significant.
    g1 = (energy_descent_accuracy >= ar_accuracy) or (not significant)

    if g2:
        verdict = "complete: energy_descent_beats_ar_premise_validated"
        return PremiseVerdict(verdict, True, True)
    if g1:
        verdict = "complete: energy_descent_viable_not_superior_at_scale"
        return PremiseVerdict(verdict, True, False)
    verdict = "complete: energy_descent_below_ar_premise_unsupported_at_scale"
    return PremiseVerdict(verdict, False, False)


def reproducibility_checksum(
    *,
    corpus_path: str | Path,
    n_problems: int,
    seed: int,
    substrate_signature: str,
) -> str:
    """Content hash over corpus + substrate + seed for reproducibility.

    A third party who re-runs with the same corpus file, the same ``n``, the
    same seed, and the same substrate signature must get the same checksum — that
    is the audit trail the adversarial-verify discipline requires for any
    compute-bound artifact.  We hash the corpus file *contents* (not just its
    path) so a swapped-out corpus is detectable.
    """

    hasher = hashlib.sha256()
    hasher.update(Path(corpus_path).read_bytes())
    hasher.update(f"|n={n_problems}|seed={seed}|{substrate_signature}".encode())
    return hasher.hexdigest()[:16]
