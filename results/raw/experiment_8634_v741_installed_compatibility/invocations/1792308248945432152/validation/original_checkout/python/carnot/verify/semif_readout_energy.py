"""Readout-energy product of experts for calibrated decisions (REQ-VERIFY-7750,
REQ-VERIFY-7751).

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

**A2, the target-workload calibrator tournament (REQ-VERIFY-7751).** A1's
post-hoc diagnostic found the RAW per-option probabilities do carry real
signal even though the registered collapsed scalar does not (see the
corrigendum above). A2 asks a separate, narrower question: does a standard
calibrator (scalar temperature scaling, true two-parameter Platt scaling,
or isotonic regression) fix the raw probabilities' calibration on any one
option channel. It reuses A1's cached logits -- zero new model calls.

Spec: REQ-VERIFY-7750, REQ-VERIFY-7751
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
    labels: Sequence[int],
    probs: Sequence[float],
    target_risk: float = 0.05,
    confidence: Sequence[float] | None = None,
    fixed_coverages: Sequence[float] = (),
) -> dict[str, float | dict[str, float]]:
    """Area under the risk-coverage curve, and the largest coverage at which
    cumulative selective risk stays at or below `target_risk`.

    Confidence defaults to `max(p, 1-p)` (how far the predicted probability
    is from a coin flip); the decision at threshold 0.5 is "incorrect" when
    p >= 0.5. Rows are ranked by confidence, most confident first, and risk
    is the running error rate over the retained prefix -- the standard
    selective-classification construction (Geifman & El-Yaniv, 2017 shape).

    `confidence` (added for REQ-VERIFY-7752 / A3) lets a caller rank rows by
    a DIFFERENT signal than the classification probability itself -- for
    example, `three_way_entropy_confidence` ranks by the raw 3-way readout's
    own predictive entropy while still classifying with a 2-way probability.
    Passing `None` (the default) preserves A1/A2's original behaviour
    exactly, so every existing caller is unaffected.

    `fixed_coverages` (also added for A3) reports selective risk at each
    named coverage level (e.g. 0.50, 0.80, 0.90) in an added
    `risk_at_fixed_coverage` key. An empty tuple (the default) omits that
    key entirely, so existing callers see the same two-key dict as before.
    """
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64)
    n = len(y)
    if n == 0:
        out: dict[str, float | dict[str, float]] = {"aurc": 0.0, "coverage_at_5pct_risk": 0.0}
        if fixed_coverages:
            out["risk_at_fixed_coverage"] = {f"{c:.2f}": 0.0 for c in fixed_coverages}
        return out
    conf = (
        np.maximum(p, 1.0 - p) if confidence is None else np.asarray(confidence, dtype=np.float64)
    )
    order = np.argsort(-conf, kind="stable")
    y_sorted = y[order]
    pred_sorted = (p[order] >= 0.5).astype(np.float64)
    errors = (pred_sorted != y_sorted).astype(np.float64)
    counts = np.arange(1, n + 1, dtype=np.float64)
    cum_risk = np.cumsum(errors) / counts
    coverage = counts / n
    aurc = float(np.trapezoid(cum_risk, coverage))
    within_target = np.where(cum_risk <= target_risk)[0]
    coverage_at_risk = float(coverage[within_target[-1]]) if within_target.size else 0.0
    result: dict[str, float | dict[str, float]] = {
        "aurc": aurc,
        "coverage_at_5pct_risk": coverage_at_risk,
    }
    if fixed_coverages:
        risk_at_fixed: dict[str, float] = {}
        for c in fixed_coverages:
            k = max(1, min(n, int(round(c * n))))
            risk_at_fixed[f"{c:.2f}"] = float(cum_risk[k - 1])
        result["risk_at_fixed_coverage"] = risk_at_fixed
    return result


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


# ==========================================================================
# A2: target-workload calibrator tournament (REQ-VERIFY-7751).
#
# A1 fit one collapsed scalar (-log p(accept)) into a product of experts and
# found alpha=0.0 in every fold. A1's own post-hoc diagnostic then found the
# RAW per-option probabilities DO carry real signal -- the collapse, not an
# absence of signal, explains the alpha=0 result (see this module's
# corrigendum note above, and ops/verifier_gaps.md's
# GAP-SEMIF-ACCEPT-ENERGY-COLLAPSES-REJECT-SIGNAL-7750 entry).
#
# A2 asks a narrower, separate question: are the raw per-option
# probabilities themselves miscalibrated, and does a standard calibrator
# fix that. Every function below reuses A1's cached logits. None makes a
# model call.
# ==========================================================================

ISOTONIC_MIN_POSITIVES = 20
ISOTONIC_MIN_NEGATIVES = 20
DEFAULT_ECE_CEILING = 0.05
DEFAULT_TIE_TOLERANCE = 1e-4


@dataclass(frozen=True)
class A2ScoredRow:
    """One held-out row carrying the three raw per-option logits A1 cached,
    plus the same binary label convention used throughout this module
    (1 = incorrect, 0 = correct)."""

    question_id: str
    label: int
    raw_logits: dict[str, float]


def channel_probability(raw_logits: Mapping[str, float | None], channel: str) -> float | None:
    """Softmax probability for one declared option, from the raw 3-way logits.

    Returns None when any of the three raw logits is missing or non-finite.
    This is the same degenerate condition `readout_from_logits` guards,
    exposed here so A2 can read a single channel's calibrated-input
    probability without re-deriving the softmax.
    """
    result = readout_from_logits(raw_logits)
    if result.probs is None:
        return None
    return result.probs[channel]


def fit_two_parameter_platt(
    logits: Sequence[float], labels: Sequence[int], n_steps: int = 300, lr: float = 0.05
) -> dict[str, float]:
    """Fit `calibrated = sigmoid(a * logit + b)` by gradient descent on NLL.

    This is the TRUE two-parameter Platt scaling the plan asks for. It is
    NOT the same as `PlattScaler`
    (`python/carnot/training/platt_scaler.py`), which fits one temperature
    only (`calibrated = sigmoid(logit / T)`, equivalent to `a = 1/T, b =
    0`). The extra free parameter `b` is what lets this calibrator correct
    an affine distortion that has BOTH a scale and a shift -- temperature
    scaling alone cannot correct a shift.
    """
    import jax
    import jax.numpy as jnp

    z = jnp.asarray(logits, dtype=jnp.float32)
    y = jnp.asarray(labels, dtype=jnp.float32)
    a, b = 1.0, 0.0

    def _nll(a_: float, b_: float) -> jnp.ndarray:
        p = jax.nn.sigmoid(a_ * z + b_)
        p = jnp.clip(p, 1e-8, 1.0 - 1e-8)
        return -jnp.mean(y * jnp.log(p) + (1.0 - y) * jnp.log(1.0 - p))

    grad_fn = jax.grad(_nll, argnums=(0, 1))
    for _ in range(n_steps):
        ga, gb = grad_fn(a, b)
        a = float(jnp.clip(a - lr * float(ga), -50.0, 50.0))
        b = float(jnp.clip(b - lr * float(gb), -50.0, 50.0))
    return {"a": a, "b": b}


def calibrate_two_parameter_platt(logits: Sequence[float], a: float, b: float) -> np.ndarray:
    """Apply a fitted two-parameter Platt calibrator to raw logits."""
    z = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-(a * z + b)))


def probability_to_logit(prob: Sequence[float] | np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Log-odds of a probability, clipped away from 0 and 1.

    Temperature and Platt scaling expect a raw score on the same scale as
    `log(p / (1 - p))` -- a well-behaved calibration input, usually a few
    units in magnitude. The RAW per-option vocabulary logit A1 caches is
    NOT that: it is an un-normalized language-model output score, often
    10-20 units in magnitude, with no fixed relationship to the softmax
    probability of one option. Feeding that raw vocabulary logit straight
    into a calibrator saturates the sigmoid and produces garbage (measured
    directly in this experiment: catastrophic Brier scores and NaN Platt
    fits). The fix is to calibrate the log-odds of the ALREADY-SOFTMAXED
    channel probability instead -- at temperature 1 this is an exact
    pass-through (`sigmoid(probability_to_logit(p)) == p`), which is the
    correct behaviour for an unfit calibrator.
    """
    p = np.clip(np.asarray(prob, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def enough_examples_for_isotonic(labels: Sequence[int]) -> bool:
    """The sample-size floor the plan requires: at least 20 positive AND 20
    negative examples in the training data. Below this floor, isotonic
    regression is omitted for that fold rather than silently pooling test
    data to reach it."""
    labels_arr = np.asarray(labels)
    n_pos = int((labels_arr == 1).sum())
    n_neg = int((labels_arr == 0).sum())
    return n_pos >= ISOTONIC_MIN_POSITIVES and n_neg >= ISOTONIC_MIN_NEGATIVES


def fit_isotonic_calibrator(scores: Sequence[float], labels: Sequence[int]) -> Any | None:
    """Fit an isotonic calibrator, or return None if the sample-size floor
    is not met.

    `increasing="auto"` lets scikit-learn pick the monotonic direction from
    the training data's own Spearman sign, rather than assuming every
    channel's raw score increases with the incorrect-label rate. A1's
    diagnostic found `reject` does (higher probability, more likely
    incorrect) but `escalate` does the opposite (higher probability, LESS
    likely incorrect) -- `increasing="auto"` handles both without a
    per-channel special case.
    """
    from sklearn.isotonic import IsotonicRegression

    if not enough_examples_for_isotonic(labels):
        return None
    reg = IsotonicRegression(out_of_bounds="clip", increasing="auto")
    reg.fit(np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.float64))
    return reg


def validate_probabilities(probs: Sequence[float]) -> None:
    """Hard failure (SCENARIO-VERIFY-7751-DEGENERATE case c): an empty
    array, a non-finite value, or a probability outside [0, 1] raises
    rather than silently propagating a bad number into a metric."""
    arr = np.asarray(probs, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("empty probability array")
    if not np.all(np.isfinite(arr)):
        raise ValueError("non-finite probability")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError("probability outside [0, 1]")


def validate_two_class_fold(labels: Sequence[int]) -> None:
    """Hard failure (SCENARIO-VERIFY-7751-DEGENERATE case c): a training
    fold with only one label class present raises rather than fitting a
    calibrator that has never seen the other class."""
    unique = {int(v) for v in labels}
    if len(unique) < 2:
        raise ValueError(f"one-class fold: only label(s) {unique} present")


def max_calibration_error(probs: Sequence[float], labels: Sequence[int], n_bins: int = 10) -> float:
    """The largest (not weighted-average) per-bin |accuracy - confidence|
    gap, over the same fixed-width bins `ece_fixed_bins` uses. An empty bin
    contributes nothing -- there is no gap to report, never a silent zero
    mistaken for a perfect bin."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    n = len(p)
    if n == 0:
        return 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    worst = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        bin_acc = float(y[in_bin].mean())
        bin_conf = float(p[in_bin].mean())
        worst = max(worst, abs(bin_acc - bin_conf))
    return float(worst)


def reliability_table(
    probs: Sequence[float], labels: Sequence[int], n_bins: int = 10
) -> list[dict[str, float | int]]:
    """Per-bin (low, high, count, mean confidence, mean accuracy, gap) rows
    for the calibration report. Empty bins are left out of the table --
    there is nothing to report for them."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float | int]] = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        bin_acc = float(y[in_bin].mean())
        bin_conf = float(p[in_bin].mean())
        rows.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "count": count,
                "mean_confidence": bin_conf,
                "mean_accuracy": bin_acc,
                "gap": abs(bin_acc - bin_conf),
            }
        )
    return rows


def isotonic_preserves_order(
    raw_scores: Sequence[float], calibrated: Sequence[float], increasing: bool = True
) -> bool:
    """SCENARIO-VERIFY-7751-DEGENERATE case (b): isotonic calibration must
    never invert the rank order of two raw scores -- it can only merge
    ties (pool adjacent violators). Sorting by the raw score and checking
    the calibrated values never move backward is a direct, deterministic
    test of that invariant, independent of any particular fit."""
    order = np.argsort(np.asarray(raw_scores, dtype=np.float64), kind="stable")
    cal_sorted = np.asarray(calibrated, dtype=np.float64)[order]
    diffs = np.diff(cal_sorted)
    if increasing:
        return bool(np.all(diffs >= -1e-9))
    return bool(np.all(diffs <= 1e-9))


def run_calibrator_tournament(
    rows: Sequence[A2ScoredRow],
    channels: Sequence[str] = OPTIONS,
    k: int = DEFAULT_KFOLD_K,
) -> dict[str, Any]:
    """Grouped out-of-fold calibrator tournament, per option channel.

    For every channel (`accept`, `reject`, `escalate`) and every fold: fit
    scalar temperature scaling, true two-parameter Platt scaling, and
    (sample-size permitting) isotonic regression on the training rows
    only, then score the held-out fold. Reuses `kfold_bucket` -- the same
    question-ID-grouped split function A1 uses -- so no question's rows
    ever split across a fold's train and evaluation side.
    """
    from carnot.training.platt_scaler import PlattScaler

    fold_of: dict[str, int] = {}
    for row in rows:
        if row.question_id not in fold_of:
            fold_of[row.question_id] = kfold_bucket(row.question_id, k)

    per_channel: dict[str, Any] = {}
    for channel in channels:
        per_fold: list[dict[str, Any]] = []
        pooled_qid: list[str] = []
        pooled_label: list[int] = []
        pooled_raw: list[float] = []
        pooled_temp: list[float] = []
        pooled_platt: list[float] = []
        pooled_iso: list[float | None] = []
        isotonic_omitted: list[dict[str, Any]] = []

        for fold in range(k):
            train_rows = [r for r in rows if fold_of[r.question_id] != fold]
            eval_rows = [r for r in rows if fold_of[r.question_id] == fold]
            if not train_rows or not eval_rows:
                per_fold.append({"fold": fold, "status": "skipped_empty_split"})
                continue

            labels_train = [r.label for r in train_rows]
            labels_eval = [r.label for r in eval_rows]
            try:
                validate_two_class_fold(labels_train)
            except ValueError as exc:
                per_fold.append({"fold": fold, "status": "one_class_train_fold", "error": str(exc)})
                continue

            raw_prob_train = np.asarray(
                [channel_probability(r.raw_logits, channel) for r in train_rows], dtype=np.float64
            )
            raw_prob_eval = np.asarray(
                [channel_probability(r.raw_logits, channel) for r in eval_rows], dtype=np.float64
            )
            validate_probabilities(raw_prob_train)
            validate_probabilities(raw_prob_eval)
            # Calibrate the log-odds of the SOFTMAXED channel probability,
            # never the raw un-normalized vocabulary logit -- see
            # `probability_to_logit`'s docstring for why the raw logit
            # breaks both temperature and Platt scaling.
            z_train = probability_to_logit(raw_prob_train)
            z_eval = probability_to_logit(raw_prob_eval)

            temp_scaler = PlattScaler()
            temp_scaler.fit(z_train, np.asarray(labels_train, dtype=np.float64))
            p_temp_eval = np.asarray(temp_scaler.calibrate(z_eval), dtype=np.float64)

            platt_fit = fit_two_parameter_platt(z_train, labels_train)
            p_platt_eval = calibrate_two_parameter_platt(z_eval, platt_fit["a"], platt_fit["b"])

            iso = fit_isotonic_calibrator(raw_prob_train, labels_train)
            if iso is None:
                isotonic_omitted.append(
                    {
                        "fold": fold,
                        "reason": "sample_size_floor",
                        "n_pos_train": int(sum(1 for lbl in labels_train if lbl == 1)),
                        "n_neg_train": int(sum(1 for lbl in labels_train if lbl == 0)),
                    }
                )
                p_iso_eval = None
            else:
                p_iso_eval = np.asarray(iso.predict(raw_prob_eval), dtype=np.float64)
                validate_probabilities(p_iso_eval)

            per_fold.append(
                {
                    "fold": fold,
                    "status": "ok",
                    "n_train": len(train_rows),
                    "n_eval": len(eval_rows),
                    "temperature": temp_scaler.T,
                    "platt_a": platt_fit["a"],
                    "platt_b": platt_fit["b"],
                    "isotonic_fit": iso is not None,
                    "brier_raw": brier_score(raw_prob_eval, labels_eval),
                    "brier_temperature": brier_score(p_temp_eval, labels_eval),
                    "brier_platt2": brier_score(p_platt_eval, labels_eval),
                    "brier_isotonic": (
                        brier_score(p_iso_eval, labels_eval) if p_iso_eval is not None else None
                    ),
                    "ece_raw": ece_fixed_bins(raw_prob_eval, labels_eval),
                    "ece_temperature": ece_fixed_bins(p_temp_eval, labels_eval),
                    "ece_platt2": ece_fixed_bins(p_platt_eval, labels_eval),
                    "ece_isotonic": (
                        ece_fixed_bins(p_iso_eval, labels_eval) if p_iso_eval is not None else None
                    ),
                    "log_loss_raw": log_loss_score(raw_prob_eval, labels_eval),
                    "log_loss_temperature": log_loss_score(p_temp_eval, labels_eval),
                    "log_loss_platt2": log_loss_score(p_platt_eval, labels_eval),
                    "log_loss_isotonic": (
                        log_loss_score(p_iso_eval, labels_eval) if p_iso_eval is not None else None
                    ),
                    "mce_raw": max_calibration_error(raw_prob_eval, labels_eval),
                    "mce_temperature": max_calibration_error(p_temp_eval, labels_eval),
                    "mce_platt2": max_calibration_error(p_platt_eval, labels_eval),
                    "mce_isotonic": (
                        max_calibration_error(p_iso_eval, labels_eval)
                        if p_iso_eval is not None
                        else None
                    ),
                }
            )

            pooled_qid.extend(r.question_id for r in eval_rows)
            pooled_label.extend(labels_eval)
            pooled_raw.extend(float(v) for v in raw_prob_eval)
            pooled_temp.extend(float(v) for v in p_temp_eval)
            pooled_platt.extend(float(v) for v in p_platt_eval)
            pooled_iso.extend(
                (float(v) for v in p_iso_eval)
                if p_iso_eval is not None
                else ([None] * len(eval_rows))
            )

        per_channel[channel] = {
            "per_fold": per_fold,
            "isotonic_omitted": isotonic_omitted,
            "reliability_table_raw": reliability_table(pooled_raw, pooled_label),
            "pooled": {
                "question_id": pooled_qid,
                "label": pooled_label,
                "raw": pooled_raw,
                "temperature": pooled_temp,
                "platt2": pooled_platt,
                "isotonic": pooled_iso,
            },
        }
    return per_channel


def paired_group_bootstrap_calibrator_deltas(
    pooled: Mapping[str, Sequence[Any]], n_boot: int = 2000, seed: int = 7751
) -> dict[str, Any]:
    """95 percent paired group bootstrap for each calibrator's Brier delta
    against the raw (uncalibrated) probability, resampling question
    groups with replacement -- the same paired-group-by-question-ID
    construction A1's bootstrap uses.

    Rows where isotonic was omitted for that fold (`None` in the pooled
    `isotonic` list) are excluded from the isotonic arm's own bootstrap
    only; they still count toward the temperature and Platt arms. This is
    the sample-size floor's "omit, never silently pool" rule applied at
    the bootstrap stage.
    """
    question_id = np.asarray(pooled["question_id"])
    label = np.asarray(pooled["label"], dtype=np.float64)
    raw = np.asarray(pooled["raw"], dtype=np.float64)
    arms = {
        "temperature": np.asarray(pooled["temperature"], dtype=np.float64),
        "platt2": np.asarray(pooled["platt2"], dtype=np.float64),
    }
    iso_raw = pooled["isotonic"]
    iso_mask = np.asarray([v is not None for v in iso_raw])

    def _ci(values: np.ndarray) -> list[float]:
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    def _bootstrap_arm(
        qid: np.ndarray, y: np.ndarray, arm_probs: np.ndarray, base_probs: np.ndarray, rng_seed: int
    ) -> dict[str, Any]:
        rng = np.random.default_rng(rng_seed)
        unique_groups = np.unique(qid)
        if unique_groups.size == 0:
            return {
                "brier_point": None,
                "brier_delta_vs_raw": {"point": None, "ci95": [None, None]},
                "ece_point": None,
                "ece_ci95": [None, None],
            }
        group_row_indices = {g: np.where(qid == g)[0] for g in unique_groups}
        deltas = np.empty(n_boot, dtype=np.float64)
        ece_vals = np.empty(n_boot, dtype=np.float64)
        for b in range(n_boot):
            sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            idx = np.concatenate([group_row_indices[g] for g in sampled_groups])
            deltas[b] = brier_score(arm_probs[idx], y[idx]) - brier_score(base_probs[idx], y[idx])
            ece_vals[b] = ece_fixed_bins(arm_probs[idx], y[idx])
        return {
            "brier_point": brier_score(arm_probs, y),
            "brier_delta_vs_raw": {
                "point": brier_score(arm_probs, y) - brier_score(base_probs, y),
                "ci95": _ci(deltas),
            },
            "ece_point": ece_fixed_bins(arm_probs, y),
            "ece_ci95": _ci(ece_vals),
        }

    result: dict[str, Any] = {}
    for i, (arm_name, arm_probs) in enumerate(arms.items()):
        result[arm_name] = _bootstrap_arm(question_id, label, arm_probs, raw, seed + i)

    if bool(iso_mask.any()):
        iso_vals = np.asarray([v if v is not None else 0.0 for v in iso_raw], dtype=np.float64)
        result["isotonic"] = _bootstrap_arm(
            question_id[iso_mask],
            label[iso_mask],
            iso_vals[iso_mask],
            raw[iso_mask],
            seed + len(arms),
        )
        result["isotonic"]["n_rows_isotonic_available"] = int(iso_mask.sum())
    else:
        result["isotonic"] = {
            "omitted_entirely": True,
            "reason": "sample_size_floor never met in any fold",
        }

    result["brier_raw_point"] = brier_score(raw, label)
    result["ece_raw_point"] = ece_fixed_bins(raw, label)
    result["n_boot"] = n_boot
    result["n_groups"] = int(len(np.unique(question_id)))
    result["n_rows"] = int(len(question_id))
    return result


def select_calibrator(
    bootstrap_result: Mapping[str, Any],
    ece_ceiling: float = DEFAULT_ECE_CEILING,
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
) -> dict[str, Any]:
    """SCENARIO-VERIFY-7751-GATE: select a calibrator only when its 95
    percent Brier-delta interval is fully below zero AND its 95 percent
    ECE interval's upper bound is below `ece_ceiling`. A tie (within
    `tie_tolerance` of the best Brier point) prefers temperature scaling,
    because it has fewer parameters. No eligible calibrator means the
    channel stays diagnostic-only -- never forced to a pass.
    """
    eligible: list[tuple[str, float]] = []
    for arm in ("temperature", "platt2", "isotonic"):
        entry = bootstrap_result.get(arm)
        if not entry or entry.get("omitted_entirely"):
            continue
        delta = entry.get("brier_delta_vs_raw", {})
        ci = delta.get("ci95")
        ece_ci = entry.get("ece_ci95")
        if not ci or ci[1] is None or not ece_ci or ece_ci[1] is None:
            continue
        if ci[1] < 0.0 and ece_ci[1] < ece_ceiling:
            eligible.append((arm, entry["brier_point"]))
    if not eligible:
        return {
            "selected": None,
            "verdict": "diagnostic_only_no_method_cleared_gate",
            "eligible": [],
        }
    best_point = min(point for _, point in eligible)
    tied = [name for name, point in eligible if point - best_point <= tie_tolerance]
    selected = "temperature" if "temperature" in tied else min(eligible, key=lambda kv: kv[1])[0]
    return {
        "selected": selected,
        "verdict": f"selected_{selected}",
        "eligible": [name for name, _ in eligible],
    }


def calibrator_kill_check(
    bootstrap_result: Mapping[str, Any],
    labels: Sequence[int],
    per_fold: Sequence[Mapping[str, Any]],
    ece_ceiling: float = DEFAULT_ECE_CEILING,
) -> dict[str, Any]:
    """SCENARIO-VERIFY-7751-GATE kill criterion: drop probability-valued
    use on this channel if no calibrator clears the ECE ceiling, the best
    calibrator's Brier is no better than the prevalence baseline, or the
    fitted temperature reverses direction fold to fold (a temperature
    meaningfully below 1 in one fold and meaningfully above 1 in another --
    the model's own over/under-confidence direction flipping, not sampling
    noise around 1.0).
    """
    prevalence = float(np.mean(labels)) if len(labels) else 0.0
    prevalence_probs = np.full(len(labels), prevalence, dtype=np.float64)
    prevalence_brier = brier_score(prevalence_probs, labels)

    live_arms = [
        arm
        for arm in ("temperature", "platt2", "isotonic")
        if bootstrap_result.get(arm) and not bootstrap_result[arm].get("omitted_entirely")
    ]
    no_method_clears_ece = all(
        (bootstrap_result[arm].get("ece_ci95") or [None, None])[1] is None
        or (bootstrap_result[arm].get("ece_ci95") or [None, None])[1] >= ece_ceiling
        for arm in live_arms
    )
    best_brier = min(
        (bootstrap_result[arm]["brier_point"] for arm in live_arms),
        default=bootstrap_result.get("brier_raw_point", 1.0),
    )
    brier_no_better_than_prevalence = best_brier >= prevalence_brier

    ok_temps = [f["temperature"] for f in per_fold if f.get("status") == "ok"]
    fold_to_fold_reverses = bool(
        len(ok_temps) >= 2 and any(t < 0.9 for t in ok_temps) and any(t > 1.1 for t in ok_temps)
    )

    kill = bool(no_method_clears_ece or brier_no_better_than_prevalence or fold_to_fold_reverses)
    return {
        "prevalence_brier": prevalence_brier,
        "best_calibrator_brier": best_brier,
        "no_method_clears_ece": no_method_clears_ece,
        "brier_no_better_than_prevalence": brier_no_better_than_prevalence,
        "fold_to_fold_calibration_reverses": fold_to_fold_reverses,
        "kill": kill,
    }


# ---- Positive controls (fully synthetic; never mixed with the real fit) ---


def run_temperature_recovery_positive_control(
    seed: int = 20260928, n: int = 2000, true_temperature: float = 3.0
) -> dict[str, Any]:
    """SCENARIO-VERIFY-7751-POSITIVE-CONTROL, lane 1: a synthetic
    ground-truth logit is scaled by a known `true_temperature` to simulate
    an overconfident model's raw output. Scalar temperature scaling, fit
    on a training split, MUST recover a lower Brier score on a held-out
    split than the raw, uncalibrated (overconfident) probability."""
    from carnot.training.platt_scaler import PlattScaler

    rng = np.random.default_rng(seed)
    z_true = rng.normal(scale=1.5, size=n)
    labels = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-z_true))).astype(np.float64)
    z_observed = z_true * true_temperature

    half = n // 2
    train_slice, test_slice = slice(0, half), slice(half, n)

    scaler = PlattScaler()
    fitted_t = scaler.fit(z_observed[train_slice], labels[train_slice])
    p_calibrated_test = np.asarray(scaler.calibrate(z_observed[test_slice]))
    p_raw_test = 1.0 / (1.0 + np.exp(-z_observed[test_slice]))

    brier_raw = brier_score(p_raw_test, labels[test_slice])
    brier_calibrated = brier_score(p_calibrated_test, labels[test_slice])
    return {
        "true_temperature": true_temperature,
        "fitted_temperature": fitted_t,
        "brier_raw_overconfident": brier_raw,
        "brier_temperature_calibrated": brier_calibrated,
        "passed": bool(brier_calibrated < brier_raw),
        "n": n,
    }


def run_platt_affine_recovery_positive_control(
    seed: int = 20260929, n: int = 2000, true_scale: float = 2.0, true_shift: float = -1.5
) -> dict[str, Any]:
    """SCENARIO-VERIFY-7751-POSITIVE-CONTROL, lane 2: a synthetic
    ground-truth logit is distorted by a known affine transform (a scale
    AND a shift). Two-parameter Platt scaling MUST recover a lower Brier
    score on a held-out split than scalar temperature scaling alone --
    temperature scaling has no free parameter to correct the shift term,
    so it must leave residual bias that two-parameter Platt corrects."""
    from carnot.training.platt_scaler import PlattScaler

    rng = np.random.default_rng(seed)
    z_true = rng.normal(scale=1.5, size=n)
    labels = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-z_true))).astype(np.float64)
    z_observed = true_scale * z_true + true_shift

    half = n // 2
    train_slice, test_slice = slice(0, half), slice(half, n)

    temp_scaler = PlattScaler()
    temp_scaler.fit(z_observed[train_slice], labels[train_slice])
    p_temp_test = np.asarray(temp_scaler.calibrate(z_observed[test_slice]))

    platt_fit = fit_two_parameter_platt(z_observed[train_slice], labels[train_slice])
    p_platt_test = calibrate_two_parameter_platt(
        z_observed[test_slice], platt_fit["a"], platt_fit["b"]
    )

    brier_temp = brier_score(p_temp_test, labels[test_slice])
    brier_platt2 = brier_score(p_platt_test, labels[test_slice])
    return {
        "true_scale": true_scale,
        "true_shift": true_shift,
        "fitted_a": platt_fit["a"],
        "fitted_b": platt_fit["b"],
        "brier_temperature_only": brier_temp,
        "brier_two_parameter_platt": brier_platt2,
        "passed": bool(brier_platt2 < brier_temp),
        "n": n,
    }


# ---- Degenerate-case checks (real production code paths, not test-only) ---


def check_constant_probability_calibration_is_a_no_op() -> dict[str, Any]:
    """SCENARIO-VERIFY-7751-DEGENERATE case (a): a constant raw
    probability/logit must remain constant after every calibrator -- no
    calibrator may manufacture variation from a single repeated value."""
    from carnot.training.platt_scaler import PlattScaler

    n = 100
    z_const = np.full(n, 0.7)
    labels = np.array([0, 1] * (n // 2))

    temp_scaler = PlattScaler()
    temp_scaler.fit(z_const, labels)
    p_temp = np.asarray(temp_scaler.calibrate(z_const))
    temp_stays_constant = bool(np.ptp(p_temp) < 1e-9)

    platt_fit = fit_two_parameter_platt(z_const, labels)
    p_platt = calibrate_two_parameter_platt(z_const, platt_fit["a"], platt_fit["b"])
    platt_stays_constant = bool(np.ptp(p_platt) < 1e-9)

    prob_const = np.full(n, 0.5)
    iso = fit_isotonic_calibrator(prob_const, labels)
    iso_stays_constant = True
    if iso is not None:
        p_iso = np.asarray(iso.predict(prob_const))
        iso_stays_constant = bool(np.ptp(p_iso) < 1e-9)

    return {
        "temperature_stays_constant": temp_stays_constant,
        "platt2_stays_constant": platt_stays_constant,
        "isotonic_stays_constant": iso_stays_constant,
        "passes": bool(temp_stays_constant and platt_stays_constant and iso_stays_constant),
    }


def check_isotonic_no_false_rank_improvement(seed: int = 20260930, n: int = 200) -> dict[str, Any]:
    """SCENARIO-VERIFY-7751-DEGENERATE case (b): isotonic regression must
    never invert the raw score's rank order. A calibrator fit on a
    genuine-signal training split, then applied to a FRESH held-out score
    set, must preserve that held-out set's own raw order."""
    rng = np.random.default_rng(seed)
    raw_scores = rng.uniform(size=n)
    labels = (rng.uniform(size=n) < raw_scores).astype(np.float64)

    half = n // 2
    iso = fit_isotonic_calibrator(raw_scores[:half], labels[:half])
    if iso is None:
        return {
            "order_preserved": None,
            "passes": False,
            "error": "fixture hit the sample-size floor",
        }
    calibrated_eval = np.asarray(iso.predict(raw_scores[half:]))
    order_preserved = isotonic_preserves_order(raw_scores[half:], calibrated_eval, increasing=True)
    return {"order_preserved": order_preserved, "passes": order_preserved}


def check_hard_failures_are_raised() -> dict[str, Any]:
    """SCENARIO-VERIFY-7751-DEGENERATE case (c): empty input, a one-class
    fold, and an out-of-[0, 1] probability must each raise -- never
    silently return a plausible-looking number."""
    results: dict[str, bool] = {}

    try:
        validate_probabilities([])
        results["empty_probabilities_raises"] = False
    except ValueError:
        results["empty_probabilities_raises"] = True

    try:
        validate_two_class_fold([1, 1, 1, 1])
        results["one_class_fold_raises"] = False
    except ValueError:
        results["one_class_fold_raises"] = True

    try:
        validate_probabilities([0.2, 1.5, 0.3])
        results["out_of_range_probability_raises"] = False
    except ValueError:
        results["out_of_range_probability_raises"] = True

    return {**results, "passes": all(results.values())}


# ==========================================================================
# A3: calibrated accept/reject/escalate policy (REQ-VERIFY-7752).
#
# A1 found the registered scalar readout feature collapses to alpha=0 in
# every fold -- there is no separate "A1 signal" to route with. A2 found
# real, usable calibration on the `reject` and `escalate` option channels
# (isotonic regression cleared the gate on both) but correctly SELF-KILLED
# the `accept` channel (its best calibrated Brier was no better than the
# trivial prevalence baseline). A3's honest input set is therefore:
#
#   - the verifier-only probability (the Gibbs energy alone -- unchanged by
#     A1's collapse, since alpha=0 makes the PoE numerically equal to the
#     verifier-only expert)
#   - A2's calibrated `reject` probability (isotonic regression)
#   - A2's calibrated `escalate` probability (isotonic regression)
#
# The `accept` channel's calibration is NEVER used here -- it did not clear
# A2's kill criterion, and using a killed arm would misrepresent an
# unvalidated signal as validated.
# ==========================================================================

A3_ACTIONS: tuple[str, ...] = ("accept", "reject", "escalate")

# Pre-registered BEFORE evaluation (SCENARIO-VERIFY-7752-GATE: "evaluate a
# pre-registered grid of escalation costs instead of choosing one cost
# after seeing results"). `A3_PRIMARY_COST` is the single grid point used
# for the headline confusion matrix and the gate's "all three actions
# occur" check -- also chosen here, before any row is scored, not picked
# after looking at which point looks best.
A3_COST_GRID: tuple[float, ...] = (
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
)
A3_PRIMARY_COST = 0.20

# Matches the existing abstention measurement floor
# (`python/carnot/pipeline/risk_coverage_abstention_3718.py:29-40`).
A3_FIXED_COVERAGES: tuple[float, ...] = (0.50, 0.80, 0.90)
A3_TARGET_RISK = 0.05
A3_MIN_ROWS = 1000
A3_MIN_GROUPS = 30


def combine_calibrated_risk(
    p_verifier: Sequence[float], p_reject_cal: Sequence[float], p_escalate_cal: Sequence[float]
) -> np.ndarray:
    """The combined P(incorrect) estimate the A3 policy routes on: an
    unweighted mean of three independently-calibrated signals -- the
    verifier-only baseline, A2's calibrated `reject` channel, and A2's
    calibrated `escalate` channel.

    An unweighted mean is the simplest defensible combiner given the task's
    explicit input set. It needs no fitting on the evaluation data (so it
    cannot leak or overfit), unlike a learned weighted combiner. The
    `accept` channel is deliberately excluded -- it did not clear A2's kill
    criterion (see this section's module-level docstring).
    """
    a = np.asarray(p_verifier, dtype=np.float64)
    b = np.asarray(p_reject_cal, dtype=np.float64)
    c = np.asarray(p_escalate_cal, dtype=np.float64)
    return (a + b + c) / 3.0


def three_way_entropy_confidence(probs_by_option: Mapping[str, Sequence[float]]) -> np.ndarray:
    """1 minus the normalized Shannon entropy of the raw (uncalibrated)
    3-way softmax over accept/reject/escalate -- the "entropy control"
    ranking signal for the A3 gate.

    This is deliberately NOT the same as `max(p, 1-p)` confidence on a
    2-way collapsed probability: for a binary probability those two
    rankings are mathematically equivalent (entropy is a monotonic
    function of distance from 0.5), so an "entropy baseline" would be
    decorative if built from a 2-way score. Computing entropy over the
    genuine 3-way distribution makes it a real, distinct alternative
    signal -- correlated with our combined-risk score, but not identical.
    """
    eps = 1e-12
    arrays = [np.clip(np.asarray(probs_by_option[o], dtype=np.float64), eps, 1.0) for o in OPTIONS]
    entropy = -sum(a * np.log(a) for a in arrays)
    return 1.0 - entropy / math.log(len(OPTIONS))


def chow_reject_option_decisions(p_incorrect: Sequence[float], cost_escalate: float) -> np.ndarray:
    """The standard Chow (1970) reject-option rule, applied per row.

    `accept` when confidently correct (p < cost_escalate), `reject` when
    confidently incorrect (p > 1 - cost_escalate), `escalate` in the
    uncertain zone between the two thresholds. `cost_escalate` is clipped
    to [0, 0.5] (the rule is only meaningful in that range -- a cost above
    0.5 would make escalating always at least as cheap as guessing).

    The decision threshold is DERIVED ANALYTICALLY from the pre-registered
    cost -- it is never fit or searched for on the evaluation data, which
    is what "no new model forward, a threshold sweep... over cached
    out-of-fold rows" (the plan's cost line) means in practice.
    """
    p = np.asarray(p_incorrect, dtype=np.float64)
    cost = float(np.clip(cost_escalate, 0.0, 0.5))
    decisions = np.full(p.shape, "escalate", dtype=object)
    decisions[p < cost] = "accept"
    decisions[p > 1.0 - cost] = "reject"
    return decisions


def policy_confusion_matrix(labels: Sequence[int], decisions: Sequence[str]) -> dict[str, Any]:
    """Counts of every (action, true label) pair, plus the per-action
    totals SCENARIO-VERIFY-7752-DEGENERATE reads to decide `always_accept`
    / `always_reject` / `always_escalate`."""
    y = np.asarray(labels)
    d = np.asarray(decisions, dtype=object)
    counts = {action: int((d == action).sum()) for action in A3_ACTIONS}
    by_action_and_label: dict[str, dict[str, int]] = {}
    for action in A3_ACTIONS:
        mask = d == action
        by_action_and_label[action] = {
            "n": int(mask.sum()),
            "n_label_correct_0": int(((y == 0) & mask).sum()),
            "n_label_incorrect_1": int(((y == 1) & mask).sum()),
        }
    return {"counts": counts, "by_action_and_label": by_action_and_label, "n_total": int(len(y))}


def check_action_balance(
    counts: Mapping[str, int], cost_escalate: float, epsilon: float = 1e-9
) -> dict[str, Any]:
    """SCENARIO-VERIFY-7752-DEGENERATE: a policy using at most two of the
    three actions is not a three-way policy, UNLESS the registered cost
    matrix mathematically forces the collapse.

    Chow's rule's uncertain zone has width `1 - 2 * cost_escalate`. At
    `cost_escalate >= 0.5` the zone has zero width, so `escalate` can never
    fire -- a provable, not accidental, collapse. At `cost_escalate <= 0`
    the zone covers the whole `[0, 1]` range, so `accept` and `reject` can
    never fire. Both are provable from the cost alone. A collapse at any
    OTHER grid point is a real balance failure, the exact shape the prior
    decision artifact hit (6,613 accepts, 2 escalations, 0 rejects --
    `results/experiment_7385_v648_decision_training.json:128-142`).
    """
    zone_width = max(0.0, 1.0 - 2.0 * float(cost_escalate))
    zero_actions = [action for action in A3_ACTIONS if counts.get(action, 0) == 0]
    if not zero_actions:
        return {"degenerate": False, "zero_actions": [], "justified_by_cost_matrix": None}
    justified = False
    reason = "not mathematically forced by the cost grid -- a real balance failure"
    if zone_width <= epsilon and "escalate" in zero_actions:
        justified = True
        reason = f"cost_escalate={cost_escalate} >= 0.5 forces an empty escalate zone (Chow's rule)"
    elif cost_escalate <= epsilon and {"accept", "reject"} <= set(zero_actions):
        justified = True
        reason = (
            f"cost_escalate={cost_escalate} <= 0 forces the escalate zone "
            "to cover the whole [0, 1] range (Chow's rule)"
        )
    return {
        "degenerate": True,
        "zero_actions": zero_actions,
        "justified_by_cost_matrix": justified,
        "reason": reason,
    }


def escalation_value(
    labels: Sequence[int],
    p_incorrect: Sequence[float],
    decisions: Sequence[str],
    cost_escalate: float,
) -> dict[str, Any]:
    """Compares the escalation-aware policy's realized total cost against a
    FORCED-DECISION baseline that must always accept or reject at the same
    0.5 threshold, with no escalate option at all.

    This is the direct measurement for the kill criterion's "escalation
    saves no errors after its measured cost is charged": escalation is
    only worth using at a given `cost_escalate` if the policy's total
    realized cost (decided-row errors, plus `cost_escalate` charged per
    escalated row) is LOWER than the forced-decision baseline's error
    count.
    """
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(p_incorrect, dtype=np.float64)
    d = np.asarray(decisions, dtype=object)

    forced_pred = (p >= 0.5).astype(np.float64)
    forced_errors = int((forced_pred != y).sum())

    decided_mask = d != "escalate"
    decided_pred = (p[decided_mask] >= 0.5).astype(np.float64)
    decided_errors = int((decided_pred != y[decided_mask]).sum())
    n_escalated = int((d == "escalate").sum())
    policy_cost = float(decided_errors) + float(cost_escalate) * n_escalated

    return {
        "cost_escalate": float(cost_escalate),
        "forced_decision_errors": forced_errors,
        "forced_decision_total_cost": float(forced_errors),
        "policy_decided_errors": decided_errors,
        "policy_n_escalated": n_escalated,
        "policy_total_cost": policy_cost,
        "errors_avoided_by_escalation": forced_errors - decided_errors,
        "escalation_saves_cost": bool(policy_cost < forced_errors),
    }


def a3_kill_check(
    grid_results: Sequence[Mapping[str, Any]], coverage_at_5pct_risk: float
) -> dict[str, Any]:
    """SCENARIO-VERIFY-7752-GATE kill criterion: drop the policy if every
    calibrated arm collapses to one action across the WHOLE registered
    grid, if useful coverage stays below 25 percent, or if escalation never
    saves cost at any registered grid point.
    """
    all_single_action = all(
        len([a for a in A3_ACTIONS if r["confusion_matrix"]["counts"].get(a, 0) > 0]) <= 1
        for r in grid_results
    )
    coverage_below_floor = bool(coverage_at_5pct_risk < 0.25)
    no_escalation_ever_saves_cost = all(
        not r["escalation_value"]["escalation_saves_cost"] for r in grid_results
    )
    kill = bool(all_single_action or coverage_below_floor or no_escalation_ever_saves_cost)
    return {
        "all_grid_points_single_action": all_single_action,
        "coverage_below_25pct_floor": coverage_below_floor,
        "no_escalation_ever_saves_cost_at_any_grid_point": no_escalation_ever_saves_cost,
        "kill": kill,
    }


def run_a3_positive_control(seed: int = 20260928, n: int = 2000) -> dict[str, Any]:
    """SCENARIO-VERIFY-7752-POSITIVE-CONTROL: an independent noisy-gold
    score must produce a better (lower) AURC than an uninformative entropy
    control and a trivial prevalence control.

    Fully synthetic, kept apart from the real fit. The entropy-control lane
    mirrors `three_way_entropy_confidence`'s construction on the real
    corpus (a genuine 3-way distribution, scored by its own predictive
    entropy) but the logits are pure noise with NO relationship to the
    label -- so a working AURC/entropy-confidence pipeline should rank it
    worse than the informative noisy-gold lane.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n).astype(np.float64)

    signal = 2.0 * labels - 1.0
    z_noisy_gold = 2.0 * signal + rng.normal(scale=1.0, size=n)
    p_noisy_gold = 1.0 / (1.0 + np.exp(-z_noisy_gold))
    aurc_noisy_gold = aurc_and_coverage_at_risk(labels, p_noisy_gold)["aurc"]

    random_logits = rng.normal(size=(n, 3))
    exp_logits = np.exp(random_logits - random_logits.max(axis=1, keepdims=True))
    random_probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    entropy_confidence = three_way_entropy_confidence(
        {
            "accept": random_probs[:, 0],
            "reject": random_probs[:, 1],
            "escalate": random_probs[:, 2],
        }
    )
    aurc_entropy_control = aurc_and_coverage_at_risk(
        labels, random_probs[:, 1], confidence=entropy_confidence
    )["aurc"]

    prevalence = float(np.mean(labels))
    p_prevalence_control = np.full(n, prevalence)
    aurc_prevalence_control = aurc_and_coverage_at_risk(labels, p_prevalence_control)["aurc"]

    passed = bool(
        aurc_noisy_gold < aurc_entropy_control and aurc_noisy_gold < aurc_prevalence_control
    )
    return {
        "aurc_noisy_gold": aurc_noisy_gold,
        "aurc_entropy_control": aurc_entropy_control,
        "aurc_prevalence_control": aurc_prevalence_control,
        "noisy_gold_beats_entropy_control": bool(aurc_noisy_gold < aurc_entropy_control),
        "noisy_gold_beats_prevalence_control": bool(aurc_noisy_gold < aurc_prevalence_control),
        "passed": passed,
        "n": n,
    }


def paired_group_bootstrap_aurc_delta(
    question_ids: Sequence[str],
    labels: Sequence[int],
    main_probs: Sequence[float],
    control_probs: Sequence[float],
    n_boot: int,
    seed: int,
    main_confidence: Sequence[float] | None = None,
    control_confidence: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Paired group bootstrap for `AURC(main) - AURC(control)`, resampling
    QUESTION GROUPS with replacement -- same construction as A1's and A2's
    own bootstraps. A negative delta means `main` has a LOWER (better)
    AURC than `control`; the A3 gate requires the 95 percent interval's
    upper bound to stay below zero against both the entropy control and
    the verifier-only control.
    """
    qid = np.asarray(question_ids)
    y = np.asarray(labels, dtype=np.float64)
    p_main = np.asarray(main_probs, dtype=np.float64)
    p_control = np.asarray(control_probs, dtype=np.float64)
    c_main = None if main_confidence is None else np.asarray(main_confidence, dtype=np.float64)
    c_control = (
        None if control_confidence is None else np.asarray(control_confidence, dtype=np.float64)
    )

    unique_groups = np.unique(qid)
    group_row_indices = {g: np.where(qid == g)[0] for g in unique_groups}
    rng = np.random.default_rng(seed)

    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([group_row_indices[g] for g in sampled_groups])
        aurc_main = aurc_and_coverage_at_risk(
            y[idx], p_main[idx], confidence=None if c_main is None else c_main[idx]
        )["aurc"]
        aurc_control = aurc_and_coverage_at_risk(
            y[idx], p_control[idx], confidence=None if c_control is None else c_control[idx]
        )["aurc"]
        deltas[b] = aurc_main - aurc_control

    point_main = aurc_and_coverage_at_risk(y, p_main, confidence=c_main)["aurc"]
    point_control = aurc_and_coverage_at_risk(y, p_control, confidence=c_control)["aurc"]
    return {
        "point": point_main - point_control,
        "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        "n_boot": n_boot,
        "n_groups": int(len(unique_groups)),
    }


def run_a3_policy_evaluation(
    question_ids: Sequence[str],
    labels: Sequence[int],
    p_verifier: Sequence[float],
    p_reject_cal: Sequence[float],
    p_escalate_cal: Sequence[float],
    p_accept_raw: Sequence[float],
    p_reject_raw: Sequence[float],
    p_escalate_raw: Sequence[float],
    seed: int,
    n_boot: int = 500,
    cost_grid: Sequence[float] = A3_COST_GRID,
    primary_cost: float = A3_PRIMARY_COST,
) -> dict[str, Any]:
    """One full A3 evaluation pass for ONE bootstrap seed: the combined-risk
    score, the entropy and verifier-only controls, the AURC/coverage
    metrics for all three, the paired group bootstraps against both
    controls, the full pre-registered cost-grid sweep (confusion matrix,
    balance check, escalation value per grid point), and the kill check.
    Called once per fixed seed by the driver so seed-to-seed stability is
    directly comparable.
    """
    combined_risk = combine_calibrated_risk(p_verifier, p_reject_cal, p_escalate_cal)
    entropy_confidence = three_way_entropy_confidence(
        {"accept": p_accept_raw, "reject": p_reject_raw, "escalate": p_escalate_raw}
    )

    main_metrics = aurc_and_coverage_at_risk(
        labels,
        combined_risk,
        target_risk=A3_TARGET_RISK,
        fixed_coverages=A3_FIXED_COVERAGES,
    )
    verifier_only_metrics = aurc_and_coverage_at_risk(
        labels,
        p_verifier,
        target_risk=A3_TARGET_RISK,
        fixed_coverages=A3_FIXED_COVERAGES,
    )
    entropy_control_metrics = aurc_and_coverage_at_risk(
        labels,
        p_reject_raw,
        target_risk=A3_TARGET_RISK,
        fixed_coverages=A3_FIXED_COVERAGES,
        confidence=entropy_confidence,
    )

    delta_vs_entropy = paired_group_bootstrap_aurc_delta(
        question_ids,
        labels,
        combined_risk,
        p_reject_raw,
        n_boot=n_boot,
        seed=seed,
        control_confidence=entropy_confidence,
    )
    delta_vs_verifier_only = paired_group_bootstrap_aurc_delta(
        question_ids,
        labels,
        combined_risk,
        p_verifier,
        n_boot=n_boot,
        seed=seed + 1,
    )

    grid_results: list[dict[str, Any]] = []
    for cost in cost_grid:
        decisions = chow_reject_option_decisions(combined_risk, cost)
        confusion = policy_confusion_matrix(labels, decisions)
        balance = check_action_balance(confusion["counts"], cost)
        esc_value = escalation_value(labels, combined_risk, decisions, cost)
        grid_results.append(
            {
                "cost_escalate": cost,
                "confusion_matrix": confusion,
                "balance_check": balance,
                "escalation_value": esc_value,
            }
        )

    primary_result = next(
        (r for r in grid_results if abs(r["cost_escalate"] - primary_cost) < 1e-12), None
    )
    kill_check = a3_kill_check(
        grid_results, coverage_at_5pct_risk=main_metrics["coverage_at_5pct_risk"]
    )

    brier = brier_score(combined_risk, labels)
    ece = ece_fixed_bins(combined_risk, labels)

    return {
        "seed": seed,
        "n_rows": int(len(labels)),
        "combined_risk_brier": brier,
        "combined_risk_ece": ece,
        "main_metrics": main_metrics,
        "verifier_only_control_metrics": verifier_only_metrics,
        "entropy_control_metrics": entropy_control_metrics,
        "aurc_delta_vs_entropy_control": delta_vs_entropy,
        "aurc_delta_vs_verifier_only_control": delta_vs_verifier_only,
        "cost_grid_results": grid_results,
        "primary_cost_result": primary_result,
        "kill_check": kill_check,
    }
