"""Fitness target #3: a trained calibrated-decision selector (REQ-AUTO-018).

**Why this exists.** CLAUDE.md's "Energy-Based Calibrated-Decision Training
Floor" (2026-09-18) names an open gap already logged three times in
`ops/verifier_gaps.md` under `GAP-ORACLE-DISTINCT` and
`GAP-DETECTOR-AUROC-4208`: an existing verifier/detector already produces a
discriminating signal, but nothing trains a SELECTION POLICY that converts
that signal into a calibrated decision. `verifier_auroc_benchmark.py`
(fitness target #2) only tunes a LINEAR combination of two fixed features
(the AUROC is invariant to the (w0, w1) angle alone, so one grid search
finds the whole landscape -- see that module's own docstring). This module
is the next step: a hypothesis trains a small nonlinear energy model
(`carnot.models.gibbs.GibbsModel`) over the SAME two PCIB features via NCE
(`carnot.training.nce.nce_loss`), reusing the training-loop PATTERN already
present in `code_improvement.py`'s hypothesis templates -- which are not
wired into this project's standing autoresearch rotation at all.

**The benchmark.** The same held-out split of `data/fover_corpus_v4.json`
used by `verifier_auroc_benchmark.py` (6548 labeled reasoning-step rows),
but with an INDEPENDENT deterministic split (a different hash salt) so
this benchmark does not silently depend on that module's private split
internals. For each row, two raw PCIB signals are precomputed once via
`PCIBProbe.compute_entity_uptake` / `.compute_falsifiability_score` --
weight-INDEPENDENT features, unlike `verifier_auroc_benchmark.py`'s
weighted combination. A hypothesis trains a small `GibbsModel`
(`input_dim=2, hidden_dims=[4]`, architecture FIXED so the harness can
validate and rescore any submitted weights against a known shape) via
`nce_loss(model, correct_features, incorrect_features)` -- reusing NCE's
data/noise framing as a binary classifier (correct rows should get LOW
energy, incorrect rows HIGH energy), the same reuse pattern already
established in `code_improvement.py`'s hypothesis templates, not a new
misuse invented here.

**Two scored metrics, not one.** `final_energy = 1.0 - auroc` (matches
every other benchmark's "lower is better" convention and is the sole
GATING metric -- this module does not change `evaluator.py`'s read of
`final_energy`). `brier` is an ADDITIONAL, non-gating field: the model's
raw energy per row is mapped to a pseudo-probability via
`sigmoid(energy)` (an NCE-trained model pushes incorrect-row energy high,
correct-row energy low, so `sigmoid(energy)` reads naturally as
"probability this row is incorrect"), and Brier score is the mean squared
error between that probability and the true 0/1 label -- bounded in
[0, 1], no binning degenerate cases (Expected Calibration Error's usual
failure mode), and directly answers "does this model's confidence track
its correctness" -- the calibration question this whole floor exists to
measure. Reported, not yet gated on; see CLAUDE.md's floor rule.

**The trust boundary (same shape as REQ-AUTO-021/025).** A hypothesis
reports `final_state` -- the trained weights it converged to, as plain
JSON-serializable nested lists/floats -- never a self-reported energy or
Brier score. `recompute_calibrated_decision_metrics` is the one function
trusted harness code calls (from a FRESH subprocess, same as
`verifier_auroc_benchmark.py`'s `recompute_verifier_auroc_energy` --
see `scripts/autoresearch_conductor_round.py`'s `_subprocess_recompute_*`
functions) to turn that claim into real, independently-measured numbers.
`GibbsModel`/`GibbsConfig`/`nce_loss` are handed to the hypothesis
directly as objects in `benchmark_data` (the same "Variant A" pattern
already used for `PCIBProbe`), never importable, since
`scripts/autoresearch_conductor_round.py`'s `run_round` blocks the
`carnot` import root for sandboxed execution entirely.

**A degenerate weight set is honestly the untrained default, and is
rejected only from the ACCEPT path, not the seed measurement.** A freshly
constructed `GibbsModel` has a ZERO-initialized output layer by
construction (see `GibbsModel.__init__`), so its energy is exactly 0.0 for
every input regardless of the hidden layer -- a perfectly constant scorer.
`_binary_auroc`'s tie-counting rule gives a constant scorer AUROC exactly
0.5 (chance), which is mathematically correct and is the honest default
seed (see `measure_default_calibration_energy`, which calls the recompute
path with `reject_degenerate=False`). But a HYPOTHESIS that submits a
degenerate (all-equal-score) weight set must never be accepted as an
"improvement" over that seed -- exactly the verifier_auroc incident this
project already lived through (REAL_BUG 4: a constant scorer at AUROC 0.5
beat a worse-than-chance baseline and was fabricated-accepted). So
`recompute_calibrated_decision_metrics` rejects (returns None) a
degenerate weight set by default; only the seed measurement bypasses that
check, since 0.5 there is real, not gamed.

Spec: REQ-AUTO-018
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Any

import numpy as np

from carnot.paths import repo_path
from carnot.verify.pcib_probe import PCIBProbe

CALIBRATED_DECISION_BENCHMARK_NAME = "calibrated_decision"

# The architecture is FIXED (not hypothesis-chosen) so a submitted final_state
# always has a known shape the harness can validate and rescore -- the same
# reason verifier_auroc_benchmark.py fixes its weight count at exactly two.
INPUT_DIM = 2
HIDDEN_DIM = 4

# A numeric-stability bound on a hypothesis-controlled weight, mirroring
# verifier_auroc_benchmark.py's MAX_ABS_WEIGHT -- prevents an extreme
# magnitude from interacting pathologically with the forward pass, without
# constraining the AUROC-relevant shape of the decision function.
MAX_ABS_WEIGHT = 50.0

# A different hash salt than verifier_auroc_benchmark.py's `_bucket_of`, so
# this benchmark's train/held-out split is deterministic and independent --
# it does not read that module's private split function or its cache.
_SALT = "calibrated_decision_v1"
_TRAIN_BUCKET_CEILING = 3
_BUCKET_MODULUS = 10


def _bucket_of(question_id: Any) -> int:
    """Deterministic, stable hash bucket in [0, _BUCKET_MODULUS) -- fixed
    forever so the split never drifts between two calls, two processes, or
    two autoresearch rounds. Salted independently of
    `verifier_auroc_benchmark._bucket_of` so the two benchmarks' splits do
    not silently coincide or drift together."""
    digest = hashlib.sha256(f"{_SALT}:{question_id}".encode()).hexdigest()
    return int(digest[:8], 16) % _BUCKET_MODULUS


@lru_cache(maxsize=1)
def _load_corpus_rows() -> tuple[dict[str, Any], ...]:
    """Same corpus file as verifier_auroc_benchmark.py, loaded independently
    (own cache, own filter) so this module has no import-time dependency on
    that one. Returns an empty tuple (never raises) if the file is missing
    or corrupt -- same fail-toward-unscoreable posture as every other
    recompute function in this project."""
    path = repo_path("data", "fover_corpus_v4.json")
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    if not isinstance(rows, list):
        return ()
    return tuple(r for r in rows if isinstance(r, dict) and "step_text" in r and "label" in r)


@lru_cache(maxsize=1)
def _pcib_features() -> tuple[tuple[float, float, int], ...]:
    """(entity_uptake, falsifiability_score, is_incorrect) for every corpus
    row, computed ONCE via PCIBProbe's weight-independent signal methods --
    these features do not depend on entity_weight/falsifiability_weight,
    unlike verifier_auroc_benchmark.py's weighted `.score()` combination."""
    probe = PCIBProbe()
    out: list[tuple[float, float, int]] = []
    for row in _load_corpus_rows():
        eu = probe.compute_entity_uptake(row["step_text"], "")
        fs = probe.compute_falsifiability_score(row["step_text"], "")
        label = 1 if row.get("label") == "incorrect" else 0
        out.append((eu, fs, label))
    return tuple(out)


@lru_cache(maxsize=1)
def _split_features() -> tuple[
    tuple[tuple[float, float, int], ...], tuple[tuple[float, float, int], ...]
]:
    """(train_features, held_out_features) -- computed once, fixed for the
    process lifetime. Split by question_id (not row) so multiple rows from
    the same question stay on the same side, matching
    verifier_auroc_benchmark.py's leakage-avoidance reasoning."""
    rows = _load_corpus_rows()
    features = _pcib_features()
    train: list[tuple[float, float, int]] = []
    held_out: list[tuple[float, float, int]] = []
    for row, feat in zip(rows, features):
        bucket = _bucket_of(row.get("question_id"))
        (train if bucket < _TRAIN_BUCKET_CEILING else held_out).append(feat)
    return tuple(train), tuple(held_out)


def train_features_for_prompt() -> dict[str, list[list[float]]]:
    """The training split's raw PCIB features, split by label, as the ONLY
    form of the corpus handed to a hypothesis via
    `benchmark_data["calibrated_decision_train_correct"]` /
    `["calibrated_decision_train_incorrect"]`. The held-out split is never
    exposed this way -- a hypothesis cannot import this module at all
    (`run_round` blocks the `carnot` import root for the sandbox)."""
    train, _ = _split_features()
    correct = [[eu, fs] for eu, fs, label in train if label == 0]
    incorrect = [[eu, fs] for eu, fs, label in train if label == 1]
    return {"correct": correct, "incorrect": incorrect}


def _validate_final_state(
    final_state: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    """(w1, b1, w_out, b_out) as numpy arrays/float, or None for any
    malformed input. Never raises -- `final_state` is untrusted,
    LLM-generated data (same posture as
    `verifier_auroc_benchmark._validate_weights`, whose docstring explains
    why a broad ``except Exception`` is deliberate here)."""
    if not isinstance(final_state, dict):
        return None
    try:
        w1 = np.asarray(final_state["w1"], dtype=np.float64)
        b1 = np.asarray(final_state["b1"], dtype=np.float64)
        w_out = np.asarray(final_state["w_out"], dtype=np.float64)
        b_out = float(final_state["b_out"])
    except Exception:  # noqa: BLE001 -- untrusted input; see docstring above
        return None
    if w1.shape != (HIDDEN_DIM, INPUT_DIM) or b1.shape != (HIDDEN_DIM,):
        return None
    if w_out.shape != (HIDDEN_DIM,):
        return None
    for arr in (w1, b1, w_out, np.asarray([b_out])):
        if not np.all(np.isfinite(arr)):
            return None
        if np.any(np.abs(arr) > MAX_ABS_WEIGHT):
            return None
    return w1, b1, w_out, b_out


def _forward_energy(
    w1: np.ndarray, b1: np.ndarray, w_out: np.ndarray, b_out: float, x: np.ndarray
) -> float:
    """One forward pass through the fixed-shape network, in plain numpy --
    the recompute worker does not need JAX (no gradients are taken here,
    only the trained weights are evaluated), matching SiLU activation to
    `GibbsModel`'s default so a hypothesis's real GibbsModel-trained
    weights recompute to the SAME energy here as during its own training."""
    h = w1 @ x + b1
    h = h * (1.0 / (1.0 + np.exp(-h)))  # SiLU: h * sigmoid(h)
    return float(w_out @ h + b_out)


def _binary_auroc(labels: list[int], scores: list[float]) -> float | None:
    """Mann-Whitney AUROC with ties counted as half a win -- identical
    convention to `verifier_auroc_benchmark._binary_auroc`, duplicated here
    (not imported) so this module carries no dependency on that one's
    private internals. Returns None when there is no positive or no
    negative example to compare."""
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


def recompute_calibrated_decision_metrics(
    final_state: Any, *, reject_degenerate: bool = True
) -> dict[str, float] | None:
    """The one function trusted harness code calls to turn a hypothesis's
    claimed weights into real, independently-measured `final_energy` and
    `brier`. Never raises -- returns None on any unscoreable input
    (malformed state, empty/missing corpus, a held-out split with no
    positive or no negative examples, or -- when `reject_degenerate` is
    True, the default -- a weight set that scores every held-out row
    identically, the exact fabrication class `verifier_auroc_benchmark.py`
    was hardened against; see module docstring)."""
    weights = _validate_final_state(final_state)
    if weights is None:
        return None
    w1, b1, w_out, b_out = weights
    _, held_out = _split_features()
    if not held_out:
        return None
    labels: list[int] = []
    scores: list[float] = []
    try:
        for eu, fs, label in held_out:
            x = np.asarray([eu, fs], dtype=np.float64)
            scores.append(_forward_energy(w1, b1, w_out, b_out, x))
            labels.append(label)
    except Exception:  # noqa: BLE001 -- an unscoreable row must not crash the round
        return None
    if reject_degenerate and len(set(scores)) <= 1:
        return None  # a constant scorer discriminates nothing -- see module docstring
    auroc = _binary_auroc(labels, scores)
    if auroc is None:
        return None
    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.float64)
    probabilities = 1.0 / (1.0 + np.exp(-score_array))  # sigmoid(energy) -> P(incorrect)
    brier = float(np.mean((probabilities - label_array) ** 2))
    return {"final_energy": 1.0 - auroc, "brier": brier}


def measure_default_calibration_energy() -> dict[str, float]:
    """The REAL, measured (final_energy, brier) of a freshly constructed,
    UNTRAINED GibbsModel (`input_dim=2, hidden_dims=[4]`) on this corpus's
    held-out split -- used to seed the baseline registry honestly, matching
    `verifier_auroc_benchmark.measure_default_weight_energy`'s pattern. A
    fresh model's output layer is zero-initialized (see `GibbsModel.
    __init__`), so this is exactly the degenerate/chance case (AUROC 0.5,
    Brier 0.25) -- real and honest, not gamed, hence `reject_degenerate=
    False` here only. Falls back to {"final_energy": 0.5, "brier": 0.25}
    (the exact chance-level numbers) if the corpus cannot be scored at all."""
    zero_state = {
        "w1": [[0.0] * INPUT_DIM for _ in range(HIDDEN_DIM)],
        "b1": [0.0] * HIDDEN_DIM,
        "w_out": [0.0] * HIDDEN_DIM,
        "b_out": 0.0,
    }
    result = recompute_calibrated_decision_metrics(zero_state, reject_degenerate=False)
    return result if result is not None else {"final_energy": 0.5, "brier": 0.25}
