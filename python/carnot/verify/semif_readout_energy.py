"""Readout-energy product of experts for calibrated decisions (REQ-VERIFY-7750).

**What this is.** `docs/research-notes/semif-ebm-arc-experiment-plan-2026-09-20.md`
section A1 asks one question: does a SemIf-style option readout carry
source-grounded signal that the existing two PCIB features and the existing
Gibbs verifier do not already have? A readout is one forward pass over one
prompt. The prompt declares three options: `accept`, `reject`, `escalate`. We
read the three option logits from that one pass. We never generate free text
to reach a decision -- that is the whole point of a readout, and it is also
why this module can afford one forward pass per corpus row instead of a full
generation per row.

**Why a product of experts, not just a bigger feature vector.** The plan asks
for a specific, falsifiable combination:

    E_joint(row) = alpha * E_readout(row) + beta * E_verifier(row)

`alpha` and `beta` must be non-negative and fit only on training folds. This
keeps the readout and the verifier as separate, auditable experts. A negative
weight would let one expert cancel the other instead of adding evidence, and
a weight fit on the evaluation fold would leak the answer into the score.

**Why this reuses `calibrated_decision_benchmark`'s split, not its own.**
CLAUDE.md's Test-Run Record Integrity Discipline and this project's own
history of near-duplicate hash splits drifting apart (see
`calibrated_decision_benchmark.py`'s own comment on `verifier_auroc_benchmark`)
make "reuse the existing split verbatim" safer than "reimplement the same
split by hand." Every row here carries the split label that
`calibrated_decision_benchmark._split_features` already computed --
imported directly, not re-derived, so the two can never silently drift.

**Why the readout energy is `-log p(accept)`, not something more exotic.**
`calibrated_decision_benchmark.py` already treats a HIGH Gibbs energy as
"probably incorrect" (`sigmoid(energy) -> P(incorrect)`). A readout that is
confident the step is correct puts most of its probability mass on `accept`,
which makes `-log p(accept)` LOW. A readout that thinks the step is wrong
puts less mass on `accept`, which makes `-log p(accept)` HIGH. That is the
same sign convention as the Gibbs verifier, so the two energies can be added
directly in the product of experts without a sign flip.

**The corrigendum this module must not repeat.** `ops/verifier_gaps.md`'s
2026-09-28 correction (search `corrigendum-20260928-oracle-distinct`) found a
prior "oracle-distinct" win that leaked the answer through a hard-coded
confidence feature. The readout energy here comes from a genuine forward
pass over the row's own reasoning text -- it never reads the row's `label`
field, and the degenerate-case guards below exist precisely to catch a
readout, a verifier, or a fit that has stopped carrying real signal.

Spec: REQ-VERIFY-7750
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import scipy.stats

from carnot.autoresearch import calibrated_decision_benchmark as cdb
from carnot.models.gibbs import GibbsConfig, GibbsModel
from carnot.training.nce import nce_loss

# --------------------------------------------------------------------------
# The three declared options and their single-token surrogate labels.
#
# "accept" / "reject" / "escalate" are the option NAMES the plan specifies.
# Most tokenizers split those words into more than one token, and a SemIf-
# style readout needs each option to land on exactly one token so a single
# forward pass can read its logit directly. We therefore ask the model to
# answer with a letter (A/B/C) and map the letter back to the option name.
# `resolve_option_token_ids` below proves each letter is really one token on
# the loaded model before any row is scored -- never assumed.
# --------------------------------------------------------------------------
OPTIONS: tuple[str, ...] = ("accept", "reject", "escalate")
OPTION_LETTERS: dict[str, str] = {"accept": "A", "reject": "B", "escalate": "C"}

# The fixed 2-4-1 architecture reused verbatim from calibrated_decision_benchmark.py
# so the verifier-only energy here is trained the same shape as the one that
# benchmark trains -- no separate architecture decision to drift.
GIBBS_INPUT_DIM = cdb.INPUT_DIM
GIBBS_HIDDEN_DIM = cdb.HIDDEN_DIM

# A different hash salt from both calibrated_decision_benchmark's outer
# train/held-out split and verifier_auroc_benchmark's split, so the nested
# k-fold used for A1's out-of-fold cross-validation cannot silently coincide
# with either of them.
_KFOLD_SALT = "semif_readout_kfold_v1"
DEFAULT_KFOLD_K = 5

# Grid for the non-negative alpha/beta (and single-expert scale) fit. Zero is
# included so a degenerate expert can be fit away entirely instead of forced
# to contribute noise.
DEFAULT_WEIGHT_GRID: tuple[float, ...] = (
    0.0,
    0.03125,
    0.0625,
    0.125,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
    32.0,
    64.0,
)


# ==========================================================================
# Single-pass readout: prompt, token resolution, logit read, decision.
# ==========================================================================


def build_readout_prompt(step_text: str) -> str:
    """One prompt built from the row's own reasoning text.

    The corpus (`data/fover_corpus_v4.json`) has no separate grounding
    document per row -- `step_text` is the only source text a row carries.
    So "the row's available source text" (per the A1 plan) is `step_text`
    itself, and the prompt asks the model to judge that text on its own
    terms, never quoting the row's `label`.
    """
    return (
        "You are grading one step of a written solution.\n"
        "Read the step below. Decide if its final claim is correct.\n\n"
        f"Step:\n{step_text}\n\n"
        "Answer with exactly one letter.\n"
        "A) accept -- the step's claim is correct\n"
        "B) reject -- the step's claim is incorrect\n"
        "C) escalate -- you cannot tell from this step alone\n"
        "Answer:"
    )


def resolve_option_token_ids(llama: Any) -> dict[str, int] | None:
    """Prove each option letter is exactly one token on THIS model.

    Tries a leading-space form first (the common BPE encoding for a letter
    that follows a word boundary, e.g. after "Answer:"), then a bare form.
    Returns None -- never raises -- if any option cannot be resolved to
    exactly one token, so the caller can emit an honest blocked verdict
    instead of reading a logit at the wrong vocabulary index.
    """
    token_ids: dict[str, int] = {}
    for option in OPTIONS:
        letter = OPTION_LETTERS[option]
        resolved: int | None = None
        for surface in (f" {letter}", letter):
            try:
                tokens = llama.tokenize(surface.encode("utf-8"), add_bos=False)
            except Exception:  # noqa: BLE001 -- tokenizer call on an untrusted binding
                tokens = []
            if len(tokens) == 1:
                resolved = int(tokens[0])
                break
        if resolved is None:
            return None
        token_ids[option] = resolved
    # Three options that all resolved to the SAME token id would make every
    # readout uniform by construction -- a tokenizer collision, not a real
    # three-way question. Treat it as unresolved rather than silently degenerate.
    if len(set(token_ids.values())) != len(OPTIONS):
        return None
    return token_ids


def read_option_logits(
    llama: Any, prompt: str, option_token_ids: Mapping[str, int]
) -> dict[str, float | None]:
    """One forward pass over `prompt`; read the final-position logit for each
    declared option token id.

    This is a single `reset` + `eval` + read of the LAST VALID row of
    `llama.scores`. `beaver_lite.py`'s `_next_token_logprobs` reads
    `llama.scores[-1]` instead -- we measured directly (2026-09-28, this
    experiment) that this is wrong on the installed llama-cpp-python: after
    `reset()`, `llama.n_tokens` is 0, and `eval()` only fills rows
    `[0, n_tokens)` of a buffer allocated for the FULL context window
    (`n_ctx`) when `logits_all=True`. `scores[-1]` therefore reads an
    uninitialized row far past what this call wrote (confirmed: exactly
    0.0 for every vocabulary index, regardless of prompt, while
    `create_completion` on the same model correctly predicts real text).
    The valid last row is `llama.scores[llama.n_tokens - 1]`. `n_tokens`
    doubles as the guard against an unevaluated (`n_tokens == 0`) context.
    """
    prompt_tokens = llama.tokenize(prompt.encode("utf-8"), add_bos=True)
    llama.reset()
    llama.eval(list(prompt_tokens))
    n_tokens = int(llama.n_tokens)
    if n_tokens <= 0:
        return dict.fromkeys(option_token_ids, None)
    logits = np.asarray(llama.scores[n_tokens - 1], dtype=np.float64)
    out: dict[str, float | None] = {}
    for option, token_id in option_token_ids.items():
        if 0 <= token_id < logits.shape[0] and math.isfinite(float(logits[token_id])):
            out[option] = float(logits[token_id])
        else:
            out[option] = None
    return out


@dataclass(frozen=True)
class ReadoutResult:
    """The outcome of turning raw option logits into a usable signal.

    `probs` and `energy_accept` are None exactly when a degenerate input
    forced `decision` to `"escalate"` -- callers must exclude such a row from
    any real fit rather than substitute a stand-in number (SCENARIO-VERIFY-
    7750-DEGENERATE, case c).
    """

    probs: dict[str, float] | None
    decision: str
    energy_accept: float | None
    degenerate_reason: str | None


def readout_from_logits(raw_logits: Mapping[str, float | None]) -> ReadoutResult:
    """Pure function: raw per-option logits -> a `ReadoutResult`.

    This is the one place the "missing or non-finite logit forces escalate,
    never accept" guard lives (SCENARIO-VERIFY-7750-DEGENERATE, case c). It
    is called on real model output and on hand-built fixtures alike, so the
    guard is exercised by both live use and the degenerate-case tests --
    never a rule that only tests assert and the implementation never runs.
    """
    values: list[float | None] = [raw_logits.get(option) for option in OPTIONS]
    if any(v is None or not math.isfinite(v) for v in values):
        return ReadoutResult(
            probs=None,
            decision="escalate",
            energy_accept=None,
            degenerate_reason="missing_or_non_finite_option_logit",
        )
    arr = np.asarray(values, dtype=np.float64)
    top = float(arr.max())
    shifted = np.exp(arr - top)
    denom = float(shifted.sum())
    log_denom = top + math.log(denom)
    probs = {option: float(shifted[i] / denom) for i, option in enumerate(OPTIONS)}
    # E_readout(accept) = -log p(accept) = -z_accept + log-sum-exp(z).
    energy_accept = float(-arr[0] + log_denom)
    decision = OPTIONS[int(np.argmax(arr))]
    return ReadoutResult(
        probs=probs, decision=decision, energy_accept=energy_accept, degenerate_reason=None
    )


# ==========================================================================
# Corpus rows: reuse calibrated_decision_benchmark's split, add the readout.
# ==========================================================================


@dataclass(frozen=True)
class CorpusRow:
    """One `fover_corpus_v4.json` row plus the two static PCIB features and
    the split label `calibrated_decision_benchmark` already assigned it."""

    question_id: str
    step_text: str
    label: int  # 1 = incorrect, 0 = correct -- same convention as calibrated_decision_benchmark
    entity_uptake: float
    falsifiability: float
    split: str  # "train" or "held_out"


def load_corpus_rows_with_features() -> tuple[CorpusRow, ...]:
    """Every scoreable corpus row, carrying the SAME train/held-out split
    `calibrated_decision_benchmark._split_features` computes -- imported and
    replayed here, never re-derived, so the two splits cannot drift apart
    (the plan's "preserve the question-ID grouping" instruction, taken
    literally).
    """
    rows = cdb._load_corpus_rows()
    feats = cdb._pcib_features()
    out: list[CorpusRow] = []
    for row, (entity_uptake, falsifiability, label) in zip(rows, feats):
        question_id = str(row.get("question_id"))
        bucket = cdb._bucket_of(row.get("question_id"))
        split = "train" if bucket < cdb._TRAIN_BUCKET_CEILING else "held_out"
        out.append(
            CorpusRow(
                question_id=question_id,
                step_text=str(row["step_text"]),
                label=label,
                entity_uptake=entity_uptake,
                falsifiability=falsifiability,
                split=split,
            )
        )
    return tuple(out)


def kfold_bucket(question_id: Any, k: int) -> int:
    """Deterministic hash bucket in [0, k) for the NESTED out-of-fold split.

    Salted independently of `calibrated_decision_benchmark._bucket_of` (the
    outer train/held-out split) and of `verifier_auroc_benchmark._bucket_of`,
    so this nested split cannot silently coincide with either.
    """
    digest = hashlib.sha256(f"{_KFOLD_SALT}:{question_id}".encode()).hexdigest()
    return int(digest[:8], 16) % k


# ==========================================================================
# Verifier-only energy: a real, trained 2-4-1 Gibbs model via NCE.
# ==========================================================================


def train_gibbs_verifier(
    train_correct: Sequence[Sequence[float]],
    train_incorrect: Sequence[Sequence[float]],
    seed: int,
    n_epochs: int = 400,
    lr: float = 0.05,
) -> GibbsModel:
    """Train the fixed 2-4-1 Gibbs energy on the two PCIB features via NCE.

    This is the same manual-gradient-descent NCE loop already established in
    `code_improvement.py`'s hypothesis templates (the reuse the Energy-Based
    Calibrated-Decision Training Floor names as the starting substrate), not
    a new training pattern invented for this module. `correct` rows are the
    NCE "data" term (should end up low-energy); `incorrect` rows are the
    "noise" term (should end up high-energy) -- matching
    `calibrated_decision_benchmark.py`'s own convention.
    """
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom

    cfg = GibbsConfig(input_dim=GIBBS_INPUT_DIM, hidden_dims=[GIBBS_HIDDEN_DIM], activation="silu")
    model = GibbsModel(cfg, key=jrandom.PRNGKey(seed))
    correct = jnp.asarray(train_correct, dtype=jnp.float32)
    incorrect = jnp.asarray(train_incorrect, dtype=jnp.float32)

    def loss_fn(layers: list[tuple[Any, Any]], out_w: Any, out_b: Any) -> Any:
        def energy_fn(x: Any) -> Any:
            h = x
            for w, b in layers:
                h = w @ h + b
                h = h * jax.nn.sigmoid(h)  # SiLU, matching GibbsModel's default
            return out_w @ h + out_b

        return nce_loss(energy_fn, correct, incorrect)

    for _ in range(n_epochs):
        layers = [(w, b) for w, b in model.layers]
        out_w = model.output_weight
        out_b = jnp.asarray(model.output_bias, dtype=jnp.float32)
        grads = jax.grad(loss_fn, argnums=(0, 1, 2))(layers, out_w, out_b)
        new_layers = []
        for (w, b), (gw, gb) in zip(model.layers, grads[0]):
            new_layers.append((w - lr * gw, b - lr * gb))
        model.layers = new_layers
        model.output_weight = out_w - lr * grads[1]
        model.output_bias = float(out_b - lr * grads[2])
    return model


def gibbs_energy_batch(model: GibbsModel, features: Sequence[Sequence[float]]) -> np.ndarray:
    """Evaluate the trained verifier's energy on a batch of (entity_uptake,
    falsifiability) rows -- no training, a plain forward pass per row."""
    import jax
    import jax.numpy as jnp

    x = jnp.asarray(features, dtype=jnp.float32)
    energies = jax.vmap(model.energy)(x)
    return np.asarray(energies, dtype=np.float64)


def is_degenerate_energy(values: np.ndarray, decimals: int = 9) -> bool:
    """True when an energy array carries at most one distinct value -- a
    scorer that discriminates nothing, the same fabrication class
    `calibrated_decision_benchmark.py`'s `reject_degenerate` guard exists to
    catch. Non-finite entries are ignored for the distinctness count; an
    array with no finite values at all is treated as degenerate."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return True
    return len({round(float(v), decimals) for v in finite}) <= 1


# ==========================================================================
# Product of experts: combination, fitting, and metrics.
# ==========================================================================


def poe_energy(
    alpha: float, beta: float, e_readout: np.ndarray, e_verifier: np.ndarray
) -> np.ndarray:
    """E_joint = alpha * E_readout + beta * E_verifier, per the plan's formula."""
    return alpha * np.asarray(e_readout, dtype=np.float64) + beta * np.asarray(
        e_verifier, dtype=np.float64
    )


def energy_to_probability(energy: np.ndarray) -> np.ndarray:
    """sigmoid(energy) -> P(incorrect), the same mapping
    `calibrated_decision_benchmark.py` already uses for a single Gibbs energy."""
    return 1.0 / (1.0 + np.exp(-np.asarray(energy, dtype=np.float64)))


def fit_nonnegative_weights(
    energy_arrays: Sequence[np.ndarray],
    labels: Sequence[int],
    grid: Sequence[float] = DEFAULT_WEIGHT_GRID,
) -> tuple[tuple[float, ...], float]:
    """Grid search over non-negative weights (one per energy array) that
    minimizes negative log-likelihood on `labels`. Works for one energy
    (a single-expert scale fit) or two (the alpha/beta PoE fit) via the same
    code path -- CPU-only, no gradient descent needed for a 2-parameter grid.
    """
    y = np.asarray(labels, dtype=np.float64)
    arrays = [np.asarray(e, dtype=np.float64) for e in energy_arrays]
    best_weights: tuple[float, ...] = tuple(0.0 for _ in arrays)
    best_nll = math.inf
    for weights in product(grid, repeat=len(arrays)):
        joint = np.zeros_like(y)
        for w, e in zip(weights, arrays):
            joint = joint + w * e
        p = np.clip(1.0 / (1.0 + np.exp(-joint)), 1e-9, 1.0 - 1e-9)
        nll = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
        if nll < best_nll:
            best_nll = nll
            best_weights = weights
    return best_weights, best_nll


def fit_poe_with_guards(
    e_readout: np.ndarray,
    e_verifier: np.ndarray,
    labels: Sequence[int],
    grid: Sequence[float] = DEFAULT_WEIGHT_GRID,
) -> dict[str, float] | None:
    """Fit alpha/beta, but refuse (return None) if the verifier energy is
    degenerate -- SCENARIO-VERIFY-7750-DEGENERATE case (b). This mirrors
    `calibrated_decision_benchmark.recompute_calibrated_decision_metrics`'s
    own `reject_degenerate` guard: a PoE fit on top of a collapsed verifier
    would silently credit the readout for the verifier's whole contribution.
    """
    e_verifier = np.asarray(e_verifier, dtype=np.float64)
    if is_degenerate_energy(e_verifier):
        return None
    weights, nll = fit_nonnegative_weights(
        [np.asarray(e_readout, dtype=np.float64), e_verifier], labels, grid=grid
    )
    return {"alpha": weights[0], "beta": weights[1], "train_nll": nll}


def spearman_rank_agreement(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation between two score arrays, as a plain float."""
    corr, _ = scipy.stats.spearmanr(
        np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    )
    return float(corr)


def check_uniform_readout_preserves_verifier_ranking(
    e_verifier: np.ndarray, alpha: float = 1.0, beta: float = 1.0, constant: float = 3.7
) -> float:
    """SCENARIO-VERIFY-7750-DEGENERATE case (a): if the readout assigns the
    identical energy to every row, the joint ranking must equal the
    verifier-only ranking up to a constant. Returns the Spearman rank
    correlation between the joint order and the verifier-only order; a
    correct implementation gives exactly 1.0 for any alpha, beta, constant.
    """
    e_readout_constant = np.full_like(np.asarray(e_verifier, dtype=np.float64), float(constant))
    joint = poe_energy(alpha, beta, e_readout_constant, e_verifier)
    return spearman_rank_agreement(joint, e_verifier)


# ---- Calibration and selective-risk metrics -------------------------------


def brier_score(probs: Sequence[float], labels: Sequence[int]) -> float:
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def log_loss_score(probs: Sequence[float], labels: Sequence[int]) -> float:
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    y = np.asarray(labels, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def ece_fixed_bins(probs: Sequence[float], labels: Sequence[int], n_bins: int = 10) -> float:
    """Expected Calibration Error over fixed equal-width bins -- the same
    weighted-by-bin-count formula `PlattScaler.compute_ece` uses, reimplemented
    here in plain numpy so this module has no JAX dependency for a metric that
    needs none."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    n = len(p)
    if n == 0:
        return 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        bin_acc = float(y[in_bin].mean())
        bin_conf = float(p[in_bin].mean())
        ece += (count / n) * abs(bin_acc - bin_conf)
    return float(ece)


def auroc_score(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Mann-Whitney AUROC with ties as half a win -- reuses
    `calibrated_decision_benchmark._binary_auroc` directly (not a duplicate),
    so this module's AUROC can never silently drift from that one's."""
    return cdb._binary_auroc(list(labels), list(scores))


def aurc_and_coverage_at_risk(
    labels: Sequence[int], probs: Sequence[float], target_risk: float = 0.05
) -> dict[str, float]:
    """Area under the risk-coverage curve, and the largest coverage at which
    cumulative selective risk stays at or below `target_risk`.

    Confidence is `max(p, 1-p)` (how far the predicted probability is from a
    coin flip); the decision at threshold 0.5 is "incorrect" when p >= 0.5.
    Rows are ranked by confidence, most confident first, and risk is the
    running error rate over the retained prefix -- the standard selective-
    classification construction (Geifman & El-Yaniv, 2017 shape).
    """
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64)
    n = len(y)
    if n == 0:
        return {"aurc": 0.0, "coverage_at_5pct_risk": 0.0}
    confidence = np.maximum(p, 1.0 - p)
    order = np.argsort(-confidence, kind="stable")
    y_sorted = y[order]
    pred_sorted = (p[order] >= 0.5).astype(np.float64)
    errors = (pred_sorted != y_sorted).astype(np.float64)
    counts = np.arange(1, n + 1, dtype=np.float64)
    cum_risk = np.cumsum(errors) / counts
    coverage = counts / n
    aurc = float(np.trapezoid(cum_risk, coverage))
    within_target = np.where(cum_risk <= target_risk)[0]
    coverage_at_risk = float(coverage[within_target[-1]]) if within_target.size else 0.0
    return {"aurc": aurc, "coverage_at_5pct_risk": coverage_at_risk}


# ==========================================================================
# Positive control: a fully synthetic lane, never mixed with the real fit.
# ==========================================================================


def run_positive_control(seed: int = 20260928, n: int = 2000) -> dict[str, Any]:
    """SCENARIO-VERIFY-7750-POSITIVE-CONTROL: a readout-shaped feature that
    is a noisy copy of the independent gold label must be detected by the
    PoE fit -- a strictly positive fitted `alpha`, and a PoE Brier score
    that beats a verifier with no signal at all. This lane never touches
    `data/fover_corpus_v4.json`, the real readout, or the real Gibbs
    verifier -- it is entirely synthetic, per the plan's "never mix this
    lane with the real fit" instruction.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n).astype(np.float64)
    signal = 2.0 * labels - 1.0  # +1 for incorrect, -1 for correct
    e_readout_noisy_gold = 2.0 * signal + rng.normal(scale=1.5, size=n)
    e_verifier_no_signal = rng.normal(scale=1.0, size=n)  # a verifier that knows nothing

    half = n // 2
    train_slice = slice(0, half)
    test_slice = slice(half, n)

    fit = fit_poe_with_guards(
        e_readout_noisy_gold[train_slice], e_verifier_no_signal[train_slice], labels[train_slice]
    )
    assert fit is not None, "synthetic verifier column must not be degenerate by construction"
    alpha, beta = fit["alpha"], fit["beta"]

    joint_test = poe_energy(
        alpha, beta, e_readout_noisy_gold[test_slice], e_verifier_no_signal[test_slice]
    )
    p_poe = energy_to_probability(joint_test)
    brier_poe = brier_score(p_poe, labels[test_slice])

    verifier_scale, _ = fit_nonnegative_weights(
        [e_verifier_no_signal[train_slice]], labels[train_slice]
    )
    p_verifier_only = energy_to_probability(verifier_scale[0] * e_verifier_no_signal[test_slice])
    brier_verifier_only = brier_score(p_verifier_only, labels[test_slice])

    return {
        "alpha": alpha,
        "beta": beta,
        "brier_poe": brier_poe,
        "brier_verifier_only_no_signal": brier_verifier_only,
        "alpha_is_positive": bool(alpha > 0.0),
        "poe_beats_no_signal_verifier": bool(brier_poe < brier_verifier_only),
        "passed": bool(alpha > 0.0 and brier_poe < brier_verifier_only),
        "n": n,
    }


# ==========================================================================
# Readout logit cache -- so a rerun, or a future A2/A3, never repeats a
# model call for a row already scored.
# ==========================================================================


def readout_cache_key(question_id: str, step_text: str) -> str:
    """Content-addressed key so a cache entry is tied to what was actually
    scored, not just a row index that could drift if the corpus is re-cut."""
    return hashlib.sha256(f"{question_id}:{step_text}".encode()).hexdigest()[:24]


def load_readout_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_readout_cache(path: Path, cache: Mapping[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(cache)), encoding="utf-8")


# ==========================================================================
# Grouped out-of-fold cross-validation over the held-out split.
# ==========================================================================


@dataclass(frozen=True)
class ScoredRow:
    """One held-out row with every energy A1 needs, joined for the OOF CV."""

    question_id: str
    label: int
    e_readout: float
    e_verifier: float


def run_oof_cross_validation(
    rows: Sequence[ScoredRow],
    k: int = DEFAULT_KFOLD_K,
    grid: Sequence[float] = DEFAULT_WEIGHT_GRID,
) -> dict[str, Any]:
    """Grouped k-fold out-of-fold cross-validation, fitting alpha/beta (and
    each single-expert scale) on k-1 folds and scoring the held fold, for
    every fold. Folds are grouped by `question_id` via `kfold_bucket`, so no
    question's rows split across the fit and the evaluation side of a fold.
    """
    fold_of: dict[str, int] = {}
    for row in rows:
        if row.question_id not in fold_of:
            fold_of[row.question_id] = kfold_bucket(row.question_id, k)

    per_fold: list[dict[str, Any]] = []
    pooled_question_ids: list[str] = []
    pooled_labels: list[int] = []
    pooled_p_poe: list[float] = []
    pooled_p_readout: list[float] = []
    pooled_p_verifier: list[float] = []

    for fold in range(k):
        train_rows = [r for r in rows if fold_of[r.question_id] != fold]
        eval_rows = [r for r in rows if fold_of[r.question_id] == fold]
        if not train_rows or not eval_rows:
            continue

        e_readout_train = np.asarray([r.e_readout for r in train_rows], dtype=np.float64)
        e_verifier_train = np.asarray([r.e_verifier for r in train_rows], dtype=np.float64)
        labels_train = [r.label for r in train_rows]

        poe_fit = fit_poe_with_guards(e_readout_train, e_verifier_train, labels_train, grid=grid)
        if poe_fit is None:
            per_fold.append({"fold": fold, "status": "degenerate_verifier_rejected"})
            continue
        readout_scale, _ = fit_nonnegative_weights([e_readout_train], labels_train, grid=grid)
        verifier_scale, _ = fit_nonnegative_weights([e_verifier_train], labels_train, grid=grid)

        e_readout_eval = np.asarray([r.e_readout for r in eval_rows], dtype=np.float64)
        e_verifier_eval = np.asarray([r.e_verifier for r in eval_rows], dtype=np.float64)
        labels_eval = [r.label for r in eval_rows]

        p_poe = energy_to_probability(
            poe_energy(poe_fit["alpha"], poe_fit["beta"], e_readout_eval, e_verifier_eval)
        )
        p_readout = energy_to_probability(readout_scale[0] * e_readout_eval)
        p_verifier = energy_to_probability(verifier_scale[0] * e_verifier_eval)

        fold_result = {
            "fold": fold,
            "status": "ok",
            "n_train": len(train_rows),
            "n_eval": len(eval_rows),
            "alpha": poe_fit["alpha"],
            "beta": poe_fit["beta"],
            "readout_only_scale": readout_scale[0],
            "verifier_only_scale": verifier_scale[0],
            "brier_poe": brier_score(p_poe, labels_eval),
            "brier_readout_only": brier_score(p_readout, labels_eval),
            "brier_verifier_only": brier_score(p_verifier, labels_eval),
            "log_loss_poe": log_loss_score(p_poe, labels_eval),
            "log_loss_readout_only": log_loss_score(p_readout, labels_eval),
            "log_loss_verifier_only": log_loss_score(p_verifier, labels_eval),
            "ece_poe": ece_fixed_bins(p_poe, labels_eval),
            "ece_readout_only": ece_fixed_bins(p_readout, labels_eval),
            "ece_verifier_only": ece_fixed_bins(p_verifier, labels_eval),
            "auroc_poe": auroc_score(labels_eval, p_poe),
            "auroc_readout_only": auroc_score(labels_eval, p_readout),
            "auroc_verifier_only": auroc_score(labels_eval, p_verifier),
            **{f"poe_{k2}": v for k2, v in aurc_and_coverage_at_risk(labels_eval, p_poe).items()},
        }
        per_fold.append(fold_result)

        pooled_question_ids.extend(r.question_id for r in eval_rows)
        pooled_labels.extend(labels_eval)
        pooled_p_poe.extend(float(v) for v in p_poe)
        pooled_p_readout.extend(float(v) for v in p_readout)
        pooled_p_verifier.extend(float(v) for v in p_verifier)

    return {
        "per_fold": per_fold,
        "pooled": {
            "question_id": pooled_question_ids,
            "label": pooled_labels,
            "p_poe": pooled_p_poe,
            "p_readout_only": pooled_p_readout,
            "p_verifier_only": pooled_p_verifier,
        },
    }


def paired_group_bootstrap_brier_delta(
    pooled: Mapping[str, Sequence[Any]],
    n_boot: int = 2000,
    seed: int = 7750,
) -> dict[str, Any]:
    """95 percent paired group bootstrap intervals for PoE Brier delta
    against each single expert, resampling QUESTION GROUPS with replacement
    (never individual rows) so a question with several rows is resampled as
    one unit -- the plan's "paired group bootstrap intervals by question ID."
    """
    question_id = np.asarray(pooled["question_id"])
    label = np.asarray(pooled["label"], dtype=np.float64)
    p_poe = np.asarray(pooled["p_poe"], dtype=np.float64)
    p_readout = np.asarray(pooled["p_readout_only"], dtype=np.float64)
    p_verifier = np.asarray(pooled["p_verifier_only"], dtype=np.float64)

    unique_groups = np.unique(question_id)
    group_row_indices = {g: np.where(question_id == g)[0] for g in unique_groups}
    rng = np.random.default_rng(seed)

    deltas_vs_readout = np.empty(n_boot, dtype=np.float64)
    deltas_vs_verifier = np.empty(n_boot, dtype=np.float64)
    ece_deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([group_row_indices[g] for g in sampled_groups])
        brier_poe_b = brier_score(p_poe[idx], label[idx])
        brier_readout_b = brier_score(p_readout[idx], label[idx])
        brier_verifier_b = brier_score(p_verifier[idx], label[idx])
        deltas_vs_readout[b] = brier_poe_b - brier_readout_b
        deltas_vs_verifier[b] = brier_poe_b - brier_verifier_b
        ece_poe_b = ece_fixed_bins(p_poe[idx], label[idx])
        ece_better_single_b = min(
            ece_fixed_bins(p_readout[idx], label[idx]), ece_fixed_bins(p_verifier[idx], label[idx])
        )
        ece_deltas[b] = ece_poe_b - ece_better_single_b

    def _ci(values: np.ndarray) -> list[float]:
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    point_brier_poe = brier_score(p_poe, label)
    point_brier_readout = brier_score(p_readout, label)
    point_brier_verifier = brier_score(p_verifier, label)

    return {
        "n_boot": n_boot,
        "n_groups": int(len(unique_groups)),
        "n_rows": int(len(question_id)),
        "brier_delta_vs_readout_only": {
            "point": point_brier_poe - point_brier_readout,
            "ci95": _ci(deltas_vs_readout),
        },
        "brier_delta_vs_verifier_only": {
            "point": point_brier_poe - point_brier_verifier,
            "ci95": _ci(deltas_vs_verifier),
        },
        "brier_delta_vs_better_single_expert": {
            "point": point_brier_poe - min(point_brier_readout, point_brier_verifier),
        },
        "ece_delta_vs_better_single_expert": {
            "point": ece_fixed_bins(p_poe, label)
            - min(ece_fixed_bins(p_readout, label), ece_fixed_bins(p_verifier, label)),
            "ci95": _ci(ece_deltas),
        },
    }
