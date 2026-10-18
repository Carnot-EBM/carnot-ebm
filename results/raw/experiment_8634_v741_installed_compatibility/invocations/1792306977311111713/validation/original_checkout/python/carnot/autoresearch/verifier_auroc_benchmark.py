"""Fitness target #2: verifier AUROC, independently measured (REQ-AUTO-025).

**Why this exists.** Fitness target #1 (`toy_benchmarks.py`, REQ-AUTO-021)
drove `double_well`/`rosenbrock` to machine-precision zero on the very first
production autoresearch fire (2026-09-13). Every fire since has landed
`accepted > 0, committed = 0` -- not a bug, but benchmark saturation: there is
no headroom left to close on those two toy problems (see
`docs/research-notes/rsi-levels-and-autoresearch-fitness-target-2026-09-14.md`).
That note named this exact gap as the next real step. This module closes it.

**The benchmark.** A held-out slice of `data/fover_corpus_v4.json` (6548
labeled reasoning-step rows, `label` in {"correct", "incorrect"}). A
hypothesis tunes the two weights of `carnot.verify.pcib_probe.PCIBProbe`
(`entity_weight`, `falsifiability_weight` -- a text-statistical hallucination
probe, honestly disclosed there as an approximation, no GPU/LLM needed) to
best separate "incorrect" from "correct" steps by AUROC.

**Real, measured headroom (not assumed) -- for the FIRST accepted
hypothesis only.** The probe's own documented defaults (0.5, 0.5) score
AUROC 0.3465 on this corpus's held-out split -- WORSE than chance. A single
sign flip (entity_weight=-1.0, falsifiability_weight=1.0) reaches AUROC
0.7219. That gap is real and was reproduced by an independent 2026-09-16
adversarial review (Fable 5.1), which also measured its statistical size:
the held-out slice has 71 positive / 4488 negative rows, giving a
Hanley-McNeil standard error of about 0.034 at AUROC~0.73 -- roughly 125x
the evaluator's default acceptance tolerance. Because the score is a linear
function of the two weights (`w0*entity_uptake + w1*falsifiability_score`),
its AUROC depends only on the (w0, w1) angle, so ONE honest grid/gradient
search finds essentially the whole landscape. After that first genuine
improvement, further "improvements" this benchmark can report are
plausibly sampling noise on a FIXED held-out set, not real generalization --
the same review found a train-optimal angle and a held-out-optimal angle
differing by 0.00013 in energy, below the acceptance tolerance. Treat any
`verifier_auroc` acceptance AFTER the first as suspect until this is
addressed structurally (candidate fix: don't echo the exact held-out energy
value back into the hypothesis-generator prompt across rounds, or widen the
tolerance for this benchmark to roughly one Hanley-McNeil standard error).

**The trust boundary (REQ-AUTO-021's pattern, unchanged) -- and what it
does NOT cover.** A hypothesis reports `final_state = [entity_weight,
falsifiability_weight]` -- the point its search landed on -- never a
self-reported AUROC number. `recompute_verifier_auroc_energy` is the one
function trusted harness code calls to turn that claim into a real,
independently-measured energy, and (2026-09-16 fix, see
`autoresearch_conductor_round.py`'s `_subprocess_recompute_energy`) this
call now happens in a FRESH interpreter process, not the process that just
ran the hypothesis's own code -- `sandbox.py`'s own docstring already says
its in-process execution is "not a security boundary" against code that
shares the harness's `sys.modules`, and the same 2026-09-16 review
reproduced a hypothesis monkeypatching `_binary_auroc` (or mutating the
`lru_cache`d held-out row dicts in place) to make ANY claimed weights
recompute to a fabricated energy of 0.0, which then got git-committed as a
real "discovery." A fresh subprocess closes that: it re-imports every
dependency from a clean process, so nothing the sandboxed code did to the
CALLER's module state can reach it. What it does NOT close: the sandbox
still runs hypothesis code in-process, so a sufficiently motivated
hypothesis that imports `carnot.autoresearch.verifier_auroc_benchmark`
directly (blocked as of the same 2026-09-16 fix -- see
`autoresearch_conductor_round.py`'s `sandbox_config` construction in
`run_round`, which blocks the `carnot` import root for the sandboxed
execution and hands `PCIBProbe` to the hypothesis via `benchmark_data`
instead) could otherwise have called `_split_corpus()` itself and searched
the held-out rows directly rather than the training rows it is meant to
learn from. Both fixes now hold together: process isolation for the
RECOMPUTE step, import blocking for the EXECUTION step.

**The split is fixed, not re-randomized.** `_bucket_of(question_id)` is a
deterministic hash, computed once, never reseeded per round -- a hypothesis
across rounds always trains against the same ~30% and is always scored
against the same disjoint ~70% it never sees during training. Splitting by
`question_id` (not by row) keeps multiple reasoning-step rows from the same
problem on the same side of the split, so no training/held-out leakage
through a shared question.

**Energy convention.** `evaluator.py`'s gate treats a LOWER `final_energy` as
better (matches `toy_benchmarks.py`'s minimization convention), so this
module reports `1.0 - auroc`, not the AUROC itself.

Spec: REQ-AUTO-025
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Any

import numpy as np

from carnot.paths import repo_path
from carnot.verify.pcib_probe import PCIBProbe

VERIFIER_AUROC_BENCHMARK_NAME = "verifier_auroc"

# A numeric-stability bound on a hypothesis-controlled weight, so an extreme
# magnitude (1e300, say) cannot interact pathologically with PCIBProbe's own
# arithmetic. CORRECTION (2026-09-16 adversarial review): this bound does
# NOT prevent a "degenerate always-same-sign scorer" -- AUROC is invariant
# to uniformly rescaling both weights (the ranking induced by (w0, w1) and
# (10*w0, 10*w1) is identical), so [-10, 10] and [-1, 1] give the SAME
# AUROC for a fixed ratio. The actual degenerate case (both weights exactly
# 0.0, or any pair that makes every row score identically) is caught
# separately below, in `recompute_verifier_auroc_energy`, by rejecting a
# held-out score set with no variation at all.
MAX_ABS_WEIGHT = 10.0

# bucket < _TRAIN_BUCKET_CEILING -> training split (hypothesis may read this);
# bucket >= _TRAIN_BUCKET_CEILING -> held-out split (only the trusted harness
# ever scores against this). 30/70 split, chosen so the held-out AUROC
# estimate (the number that actually gates acceptance) rests on the larger,
# more statistically stable side of a corpus that is 1.7% positive-labeled
# (114 of 6548 rows) -- a smaller held-out slice would make a single accepted
# hypothesis's AUROC noisier than the effect it is trying to measure.
_TRAIN_BUCKET_CEILING = 3
_BUCKET_MODULUS = 10


def _bucket_of(question_id: Any) -> int:
    """Deterministic, stable hash bucket in [0, _BUCKET_MODULUS) for a
    question_id -- fixed forever, so the train/held-out split never drifts
    between two calls, two processes, or two autoresearch rounds."""
    digest = hashlib.sha256(str(question_id).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % _BUCKET_MODULUS


@lru_cache(maxsize=1)
def _load_corpus_rows() -> tuple[dict[str, Any], ...]:
    """The full fover_corpus_v4.json corpus, loaded once per process.

    Returns an empty tuple (never raises) if the corpus file is missing or
    corrupt -- callers treat an empty corpus the same as "cannot score this
    benchmark", not as a crash (same fail-toward-unscoreable posture as
    `toy_benchmarks.recompute_final_energy`).
    """
    path = repo_path("data", "fover_corpus_v4.json")
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    if not isinstance(rows, list):
        return ()
    return tuple(r for r in rows if isinstance(r, dict) and "step_text" in r and "label" in r)


@lru_cache(maxsize=1)
def _split_corpus() -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """(train_rows, held_out_rows) -- computed once, fixed for the process
    lifetime. See module docstring for why the split is by question_id and
    why it must never be re-randomized."""
    rows = _load_corpus_rows()
    train = tuple(r for r in rows if _bucket_of(r.get("question_id")) < _TRAIN_BUCKET_CEILING)
    held_out = tuple(r for r in rows if _bucket_of(r.get("question_id")) >= _TRAIN_BUCKET_CEILING)
    return train, held_out


def train_rows_for_prompt() -> list[dict[str, str]]:
    """The training split, as plain {"step_text", "label"} dicts -- the only
    form of the corpus handed to a hypothesis via
    `benchmark_data["verifier_auroc_train_rows"]`. The held-out split is
    never exposed THIS way. CORRECTION (2026-09-16 adversarial review): an
    earlier version of this docstring claimed the held-out split "exists
    only inside this module" as if that were unconditionally true -- it is
    only true because, as of the same fix, sandboxed hypothesis code cannot
    import this module at all (`run_round` blocks the `carnot` import root
    for the sandbox; see `autoresearch_conductor_round.py`). Before that fix,
    a hypothesis could `import carnot.autoresearch.verifier_auroc_benchmark`
    directly and call `_split_corpus()` itself to read the held-out rows.
    """
    train, _ = _split_corpus()
    return [{"step_text": r["step_text"], "label": r["label"]} for r in train]


def _binary_auroc(labels: list[int], scores: list[float]) -> float | None:
    """Mann-Whitney AUROC: fraction of (positive, negative) score pairs where
    the positive scores higher, with ties counted as half a win. Returns
    None (never raises) when there are no positive or no negative examples
    to compare -- an AUROC is undefined in that case, not zero or one.
    """
    label_array = np.asarray(labels)
    score_array = np.asarray(scores, dtype=np.float64)
    positive = score_array[label_array == 1]
    negative = score_array[label_array == 0]
    if positive.size == 0 or negative.size == 0:
        return None
    wins = 0.0
    for p in positive:
        wins += float(np.sum(p > negative))
        wins += 0.5 * float(np.sum(p == negative))
    return float(wins / (positive.size * negative.size))


def _validate_weights(final_state: Any) -> tuple[float, float] | None:
    """Exactly two finite floats, each bounded by MAX_ABS_WEIGHT -- anything
    else (wrong length, non-numeric, NaN/inf, out of range) is unscoreable
    and returns None rather than raising, matching
    `toy_benchmarks.recompute_final_energy`'s contract.

    CORRECTION (2026-09-16 adversarial review): the original version of this
    function caught only ``(TypeError, ValueError)`` around ``float(...)``,
    but a value like ``10**400`` (a plain Python int too large for a float)
    raises ``OverflowError`` instead, and any object with a pathological
    ``__float__`` could raise anything at all. Either one propagated past
    `recompute_verifier_auroc_energy` uncaught, which killed the whole
    autoresearch round before its receipt was written -- contradicting this
    function's own "never raises" claim. Catching ``Exception`` broadly here
    is deliberate: `final_state` is untrusted, LLM-generated data, and ANY
    conversion failure means "unscoreable," never a crash.
    """
    if not isinstance(final_state, list) or len(final_state) != 2:  # noqa: PLR2004
        return None
    try:
        w0, w1 = float(final_state[0]), float(final_state[1])
    except Exception:  # noqa: BLE001 -- untrusted input; see docstring above
        return None
    for w in (w0, w1):
        if w != w or w in (float("inf"), float("-inf")):  # NaN/inf guard
            return None
        if abs(w) > MAX_ABS_WEIGHT:
            return None
    return w0, w1


def recompute_verifier_auroc_energy(final_state: Any) -> float | None:
    """The one function trusted harness code calls to turn a hypothesis's
    claimed weights into a real, independently-measured energy.

    Scores every row of the FIXED held-out split with `PCIBProbe(final_state
    [0], final_state[1])`, computes AUROC via `_binary_auroc`, and returns
    `1.0 - auroc` (lower is better, matching every other benchmark's
    convention). Never raises -- returns None on any unscoreable input
    (malformed state, empty/missing corpus, a held-out split with no
    positive or no negative examples, or -- 2026-09-16 adversarial review,
    REAL_BUG 4 -- a weight pair that scores every held-out row IDENTICALLY,
    such as (0.0, 0.0). A constant scorer always produces AUROC exactly 0.5
    by the tie-counting rule in `_binary_auroc`, which happened to score
    BETTER than this benchmark's own (worse-than-chance) seed baseline and
    so was accepted and committed as a fabricated "improvement" that
    discriminates nothing -- a degenerate result must never look like a
    real one.
    """
    weights = _validate_weights(final_state)
    if weights is None:
        return None
    entity_weight, falsifiability_weight = weights
    _, held_out = _split_corpus()
    if not held_out:
        return None
    probe = PCIBProbe(entity_weight=entity_weight, falsifiability_weight=falsifiability_weight)
    labels: list[int] = []
    scores: list[float] = []
    try:
        for row in held_out:
            labels.append(1 if row.get("label") == "incorrect" else 0)
            scores.append(probe.score(row["step_text"], ""))
    except Exception:  # noqa: BLE001 -- an unscoreable row must not crash the round
        return None
    if len(set(scores)) <= 1:
        return None  # a constant scorer discriminates nothing -- see docstring
    auroc = _binary_auroc(labels, scores)
    if auroc is None:
        return None
    return 1.0 - auroc


def measure_default_weight_energy() -> float:
    """The REAL, measured energy of PCIBProbe's own documented defaults
    (0.5, 0.5) on this corpus's held-out split -- used to seed the baseline
    registry honestly (see `autoresearch_conductor_round.py:seed_baselines`)
    instead of a hand-typed placeholder that could silently drift from the
    actual corpus and probe logic. Falls back to 0.5 (an uninformative
    "AUROC == chance" energy) only if the corpus cannot be scored at all --
    this should not happen once `data/fover_corpus_v4.json` is present.
    """
    energy = recompute_verifier_auroc_energy([0.5, 0.5])
    return energy if energy is not None else 0.5
