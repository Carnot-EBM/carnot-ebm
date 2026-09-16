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

**Real, measured headroom (not assumed).** The probe's own documented
defaults (0.5, 0.5) score AUROC 0.3465 on this corpus's held-out split --
WORSE than chance. A single sign flip (entity_weight=-1.0,
falsifiability_weight=1.0) reaches AUROC 0.7219. That gap is the fitness
signal: `entity_uptake` (novel numbers = suspicious) does not actually track
this corpus's error class, so a hypothesis that discovers this by searching
rather than assuming the paper's own weighting is a genuine win, not a
degenerate one. Measured directly against the checked-in corpus with the
default weights and the sign-flipped weights; not fabricated.

**The trust boundary (REQ-AUTO-021's pattern, unchanged).** A hypothesis
reports `final_state = [entity_weight, falsifiability_weight]` -- the point
its search landed on -- never a self-reported AUROC number.
`recompute_verifier_auroc_energy` is the one function trusted harness code
calls to turn that claim into a real, independently-measured energy: it
rebuilds `PCIBProbe` from the claimed weights and rescores the HELD-OUT split
only (never the training split the hypothesis was allowed to see), computing
AUROC itself via `_binary_auroc` below. A hypothesis has no channel to
report a number directly, exactly as REQ-AUTO-021 established for the two
toy benchmarks.

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

# A hypothesis-controlled weight must not blow up PCIBProbe's arithmetic or
# produce a degenerate always-same-sign scorer -- same bounding intent as
# toy_benchmarks.py's MAX_DIM, applied to this benchmark's own parameters.
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
    form of the corpus a hypothesis's sandboxed code is handed
    (`benchmark_data["verifier_auroc_train_rows"]`). The held-out split is
    NEVER exposed this way; it exists only inside this module, read only by
    `recompute_verifier_auroc_energy`.
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
    `toy_benchmarks.recompute_final_energy`'s contract."""
    if not isinstance(final_state, list) or len(final_state) != 2:  # noqa: PLR2004
        return None
    try:
        w0, w1 = float(final_state[0]), float(final_state[1])
    except (TypeError, ValueError):
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
    (malformed state, empty/missing corpus, or a held-out split with no
    positive or no negative examples).
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
