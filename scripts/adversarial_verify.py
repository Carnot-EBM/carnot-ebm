#!/usr/bin/env python3
"""Adversarial-verify pass for experiment artifacts.

Detects patterns that indicate fabrication, tautology, methodology
gaps, or statistically-unsupported claims. Designed to run after every
experiment lands an artifact, AND as a sweep-pass over historical
results to flag earlier artifacts that slipped through.

Detection rules (each emits a discrete flag with severity):

  TAUTOLOGY
    Two distinct numerical metrics with floating-point values that
    agree to more than 5 significant figures. Example: exp1938 reported
    nrgpt_grad_norm == ce_grad_norm to 10+ digits — two different
    loss functions cannot share gradient norms by coincidence.

  IMPLAUSIBLE_PERFECT
    A metric that should be in [0, 1] reaching exactly 0.0 or 1.0,
    OR an error metric reaching exactly 0.0 with non-trivial sample
    size. Examples: exp1851 NLA probe reported TPR=1.0 with 10
    adversarial examples; this is the "fabricated" signature.

  SIGN_ANOMALY
    Optimization that claims "minimization" but final_value > initial,
    or vice versa for maximization. Example: exp1941 EBT bridge
    reported initial_energy -0.40 -> final_energy -0.10 (energy went
    up) while claiming success.

  DURATION_TOO_SHORT
    Artifact claims to invoke a live SOTA model (any GGUF in
    target_model / model_specs / responses) but duration_s < 60s.
    Loading and running a 30B+ GGUF takes minutes; if the artifact
    completed faster, the model was not actually invoked.

  SAMPLE_SIZE_BELOW_CLAIM
    Artifact reports distributional metrics (KL divergence, KS test,
    mean/variance estimates, etc.) but n_samples is below the
    statistical threshold needed for the claim type. Example:
    exp1850 reported KL=0.278 at n_spins=128 with only 100 samples;
    100 samples on a 2^128 state space cannot estimate KL meaningfully.

  GATE_PASSED_WITHOUT_DATA
    acceptance_gate_passed=true but key numerical fields referenced
    in the gate are null, missing, or zero. The agent set the gate
    flag without populating the supporting data.

  METHODOLOGY_MISSING
    Compute-bound artifact (claims model invocation, GPU, hardware)
    but lacks model_specs, random_seed, OR reproducibility_checksum.
    Methodology is unverifiable from the artifact alone.

  IMPLAUSIBLE_TIGHT_CI
    A bootstrap or empirical confidence interval whose width is far
    smaller than 1/sqrt(N) on the metric's natural scale. Real
    stochastic measurements at finite N have a minimum CI width set
    by sample variance; CIs much tighter than that signal either
    (a) the metric is deterministic-by-construction (an invariant
    rather than a measurement — should be disclosed in
    methodology_note), or (b) the bootstrap used 1 seed reshuffled
    instead of N independent draws. Example: exp1693 reported
    delta_alpha=0.15054 with bootstrap_ci_95=[0.15040, 0.15070]
    (width 3e-4) at n_seeds=30. Sample-variance-floor for a
    metric near 0.15 at N=30 is ~0.18/sqrt(30) ~ 0.03 — the
    observed CI is 100x tighter than the floor.

Output: a JSON report listing flagged artifacts with per-flag details.
Exit code: 0 if no flags, 1 if any flags present.

Usage:

    # Single artifact
    python scripts/adversarial_verify.py results/experiment_1938_*.json

    # Sweep all artifacts from a milestone range
    python scripts/adversarial_verify.py --milestone-range 1900 2000

    # All .149-era artifacts (after the structural fixes landed)
    python scripts/adversarial_verify.py --milestone-range 1900 1950
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

# Floating-point agreement threshold for tautology detection. Two
# distinct metrics agreeing to 5 significant figures is suspicious;
# legitimate cases (two implementations of the same function with the
# same seed) should be deliberate and documented.
TAUTOLOGY_DIGITS = 5

# Compute-bound markers — if the artifact mentions any of these, the
# experiment was supposed to invoke real hardware/model work.
COMPUTE_BOUND_MARKERS = (
    "unsloth/",
    "Qwen3.6-",
    "Qwen3.5-",
    "Qwen1.5-",  # 0.5B fallback — also compute-bound
    "gemma-4-",
    "GGUF",
    "DualGPURunner",
    "DualGPUHarness",
    "llama.cpp",
    "torch.cuda",
    ".cuda(",
)

# Minimum duration (seconds) for a compute-bound artifact. Loading +
# inference on even a 0.5B GGUF takes at least this long.
COMPUTE_BOUND_MIN_DURATION_S = 60.0

# Verifier-scoring artifacts: experiments that score an ensemble of
# verifiers against already-cached (input, candidate, label) triples
# WITHOUT invoking a fresh LLM forward pass. The GGUF markers in their
# model_specs are vestigial template declarations -- the task spec
# mandated naming the SOTA model, but the actual run does not load it.
# These experiments finish in seconds (verifier scoring of a few
# hundred candidates), not minutes, so the standard
# COMPUTE_BOUND_MIN_DURATION_S threshold produces false positives.
#
# Recognition is by either:
#   (a) explicit `inference_substrate` field set to the canonical
#       sentinel below (preferred -- forward-only), OR
#   (b) one of the known verifier-scoring schema prefixes (legacy
#       -- covers artifacts that pre-date the explicit field).
VERIFIER_SCORING_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
VERIFIER_SCORING_SCHEMA_PREFIXES = (
    "carnot.fover_memory_leakage_",
    "carnot.cross_corpus_verifier_matrix",
    "carnot.mbpp_dual_condition_v",
    "carnot.halueval_fever_pilot",
    "carnot.mbpp_humaneval_generated_code_clean_row",
)
VERIFIER_SCORING_MIN_DURATION_S = 1.0

# Aggregation-only artifacts: milestone capstones, archive/activate
# transitions, paper-table synthesizers, cross-corpus-matrix builders.
# These don't invoke the model OR score verifiers -- they read existing
# artifacts off disk, compute deltas / format tables / build manifests,
# and write the result. Their wall-clock measures JSON-read + arithmetic,
# legitimately a few milliseconds. The GGUF markers in their model_specs
# (when present) are forwarded from the upstream artifacts they cite.
AGGREGATION_SUBSTRATE = "aggregation_from_upstream_artifacts"
AGGREGATION_SCHEMA_PREFIXES = (
    "capstone_v",
    "carnot.milestone_capstone",
    "carnot.paper_v6_capstone",
    "carnot.archive_activation",
    "carnot.paper_v6_multicorpus_table",
)
AGGREGATION_MIN_DURATION_S = 0.0001  # 100us floor catches truly-zero/missing

# Deterministic verifier artifacts replay or aggregate checked-in candidate
# evidence. They can honestly cite upstream GGUF model names while making no
# new LLM call of their own, so they need a JSON-work duration floor rather
# than the live-model floor.
DETERMINISTIC_VERIFIER_SUBSTRATES = (
    "deterministic_verifier",
    "deterministic_verifier_plus_replay",
)
DETERMINISTIC_VERIFIER_MIN_DURATION_S = 0.0001


class Flag:
    """A single detected concern on an artifact."""

    SEVERITY_RANK = {"info": 0, "warn": 1, "critical": 2}

    def __init__(self, kind: str, severity: str, detail: str) -> None:
        self.kind = kind
        self.severity = severity
        self.detail = detail

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "severity": self.severity, "detail": self.detail}


def _is_finite_number(v: Any) -> bool:
    """True if v is a real, finite numeric value."""
    if isinstance(v, bool):
        # bool is a subclass of int; explicitly reject so True/False
        # don't trigger tautology comparison.
        return False
    if isinstance(v, (int, float)):
        return math.isfinite(float(v))
    return False


def _significant_digits_match(a: float, b: float, digits: int) -> bool:
    """Return True if a and b agree to at least `digits` significant figures.

    Handles both small and large values via relative tolerance.
    """
    if a == b:
        return True
    if a == 0.0 or b == 0.0:
        # One zero and the other not -> they disagree by definition
        return False
    rel = abs(a - b) / max(abs(a), abs(b))
    # 5 sig figs => rel < 1e-5
    return rel < 10 ** (-digits)


def _numeric_pairs(d: dict[str, Any]) -> list[tuple[str, str, float, float]]:
    """Return all distinct pairs of finite-numeric top-level keys."""
    items = [(k, float(v)) for k, v in d.items() if _is_finite_number(v)]
    pairs = []
    for i, (k1, v1) in enumerate(items):
        for k2, v2 in items[i + 1:]:
            pairs.append((k1, k2, v1, v2))
    return pairs


def _is_count_field(name: str) -> bool:
    """Field whose name implies a combinatorial count — small-integer
    coincidence is plausible, not suspicious."""
    nl = name.lower()
    count_markers = (
        "count", "n_", "_n", "num_", "_num", "total", "size",
        "_index", "step", "iteration", "epoch",
    )
    return any(m in nl for m in count_markers)


def _is_integer_value(v: float) -> bool:
    """True if the float value is exactly an integer."""
    return float(v).is_integer()


_IDENTIFIER_FIELDS = frozenset(
    {
        "experiment", "experiment_id", "exp_id", "id", "run_id", "task_id",
        "random_seed", "seed", "rng_seed", "torch_seed", "np_seed", "jax_seed",
        "milestone", "milestone_id", "schema_version", "version",
        "pid", "port", "gpu_id", "device_id", "rank", "world_size",
    }
)


def _is_identifier_field(k: str) -> bool:
    """True if the field is an identifier / seed / metadata field, not a
    measured metric. Identifiers legitimately coincide (e.g. random_seed ==
    experiment_id) and must be excluded from tautology comparison."""
    kl = k.lower()
    if kl in _IDENTIFIER_FIELDS:
        return True
    return kl.endswith("_seed") or kl.endswith("_id") or kl.endswith("_seed_used")


def check_false_negative_risk(d: dict[str, Any], flags: list[Flag]) -> None:
    """Detect NULL/negative claims that lack a valid positive control or that
    rest on a degenerate (un-exercised) method — the false-negative trap.

    Origin: exp3507 reported the process-energy reranker "does not beat
    self-consistency" (a null claim) while flip_count==0 — the reranker never
    actually changed a single selection, so the corpus had no headroom for the
    method to act on. A null claim from a method that was never exercised is
    not evidence the method fails; it's evidence the TEST was degenerate.

    Three independent risk signals (any one fires a warn; we never auto-pass a
    null claim that lacks a demonstrated positive control):

      1. NON-DEGENERACY: a `*flip*count*` / `*n_changed*` field == 0 means the
         method under test produced identical output to the baseline — the
         experiment cannot distinguish "method fails" from "no headroom."
      2. POSITIVE CONTROL / ORACLE: if an oracle/optimal upper bound is present
         and it does NOT exceed the baseline, the corpus has no selectable
         headroom; no method could win, so the null is uninformative.
      3. EXPLICIT G2-STYLE GATE: a `*non_degenerate*` / `*g2*` acceptance gate
         recorded as False is the experiment self-reporting its own degeneracy.
    """
    verdict_raw = d.get("honest_verdict") or ""
    verdict = (verdict_raw if isinstance(verdict_raw, str) else "").lower()
    null_markers = (
        "no_improvement", "does_not", "doesnt", "no_delta", "no_gain",
        "not_beat", "no_beat", "no_effect", "refuted", "null_result",
        "no_lift", "no_advantage", "no_benefit", "fails_to_beat",
        "not_better", "no_headroom",
    )
    is_null_claim = any(m in verdict for m in null_markers)
    if not is_null_claim:
        return

    # Signal 1: degenerate non-exercise (flip/change count == 0)
    for k, v in d.items():
        kl = k.lower()
        if not _is_finite_number(v):
            continue
        if ("flip" in kl and "count" in kl) or kl.startswith("n_changed") or (
            "n_flips" in kl
        ):
            if float(v) == 0.0:
                flags.append(
                    Flag(
                        kind="FALSE_NEGATIVE_RISK",
                        severity="warn",
                        detail=(
                            f"Null claim ({verdict[:48]!r}) but {k}=0: the "
                            f"method never changed any selection. Cannot "
                            f"distinguish 'method fails' from 'no headroom'. "
                            f"Re-run on a corpus where {k}>0 (positive control) "
                            f"before treating this as evidence the method fails."
                        ),
                    )
                )

    # Signal 2: oracle/optimal upper bound does not exceed baseline
    oracle = None
    baseline = None
    for k, v in d.items():
        if not _is_finite_number(v):
            continue
        kl = k.lower()
        if oracle is None and any(
            s in kl for s in ("oracle", "optimal", "upper_bound", "best_possible")
        ) and any(s in kl for s in ("acc", "rate", "solve", "score")):
            oracle = float(v)
        if baseline is None and any(
            s in kl for s in ("self_consistency", "baseline", "majority")
        ) and any(s in kl for s in ("acc", "rate", "solve", "score")):
            baseline = float(v)
    if oracle is not None and baseline is not None and oracle <= baseline:
        flags.append(
            Flag(
                kind="FALSE_NEGATIVE_RISK",
                severity="warn",
                detail=(
                    f"Null claim but oracle/optimal upper bound ({oracle}) does "
                    f"not exceed the baseline ({baseline}): the corpus has no "
                    f"selectable headroom, so NO method could win here. The null "
                    f"is uninformative about the method. Build a difficulty-"
                    f"matched corpus with oracle>baseline before re-testing."
                ),
            )
        )

    # Signal 3: an explicit non-degeneracy / G2 gate self-reported False
    for k, v in d.items():
        kl = k.lower()
        if isinstance(v, bool) and v is False and (
            "non_degenerate" in kl or "g2" in kl or "headroom" in kl
        ):
            flags.append(
                Flag(
                    kind="FALSE_NEGATIVE_RISK",
                    severity="warn",
                    detail=(
                        f"Null claim and the experiment's own gate {k}=False: "
                        f"it self-reports the test was degenerate. Do not "
                        f"propagate this null to a forward-facing claim."
                    ),
                )
            )


def check_tautology(d: dict[str, Any], flags: list[Flag]) -> None:
    """Detect distinct metrics agreeing to TAUTOLOGY_DIGITS sig figs.

    Tightened to avoid false positives on small-integer count
    coincidences (e.g. `completed_count == failed_count` is just
    arithmetic chance, not bug evidence). Genuine tautology signals
    are floating-point values matching to many decimal places —
    that's where two distinct loss-function gradient norms shouldn't
    coincide.
    """
    for k1, k2, v1, v2 in _numeric_pairs(d):
        if _legitimate_pair(k1, k2):
            continue
        # Skip identifier / seed / metadata fields. These are NOT metrics:
        # `experiment_id`, `experiment`, and `random_seed` legitimately all
        # equal the experiment number (seeding the RNG off the experiment ID
        # is good reproducibility practice — see exp3505/3506/3496/3481 which
        # this rule was false-flagging as TAUTOLOGY). Two identifiers agreeing
        # is structural, not a coincidence between two distinct measurements.
        if _is_identifier_field(k1) or _is_identifier_field(k2):
            continue
        # Skip count-coincidence pairs: both names imply counts AND
        # both values are small integers.
        if (
            _is_count_field(k1)
            and _is_count_field(k2)
            and _is_integer_value(v1)
            and _is_integer_value(v2)
            and abs(v1) < 1000
        ):
            continue
        # Skip integer-value pairs entirely if either value is small
        # — small integer coincidence is uninformative.
        if (
            _is_integer_value(v1)
            and _is_integer_value(v2)
            and abs(v1) < 100
            and abs(v2) < 100
        ):
            continue
        if _significant_digits_match(v1, v2, TAUTOLOGY_DIGITS):
            flags.append(
                Flag(
                    kind="TAUTOLOGY",
                    severity="critical",
                    detail=(
                        f"{k1}={v1!r} and {k2}={v2!r} agree to "
                        f">{TAUTOLOGY_DIGITS} sig figs. Two distinct "
                        f"metrics matching this precisely is more likely "
                        f"a bug than a finding."
                    ),
                )
            )


def _legitimate_pair(k1: str, k2: str) -> bool:
    """Pairs where bit-identity is structurally legitimate."""
    legit_suffixes = ("_abs", "_pct", "_ratio", "_mean", "_var", "_std", "_min", "_max")
    if k1.startswith(k2) or k2.startswith(k1):
        for s in legit_suffixes:
            if k1.endswith(s) or k2.endswith(s):
                return True
    # Same metric measured at two timepoints (initial vs final) often
    # legitimately matches when convergence is reached. Allow.
    paired_prefixes = (("initial_", "final_"), ("baseline_", "best_"))
    for a, b in paired_prefixes:
        if (k1.startswith(a) and k2.startswith(b)) or (
            k1.startswith(b) and k2.startswith(a)
        ):
            if k1[len(a):] == k2[len(b):] or k1[len(b):] == k2[len(a):]:
                return True
    # Two same-type metrics of the same family coinciding is a legitimate
    # research outcome, not a bug. The canonical case is an ablation /
    # selection-strategy sweep where two different aggregation conditions
    # produce the SAME exact-match accuracy on the SAME corpus (e.g.
    # energy-weighted-vote vs self-consistency tying when the energy never
    # flips the majority — exp3449). Accuracy is a bounded rational
    # (correct / n), so exact coincidence across strategies is expected and
    # meaningful, exactly like the initial/final convergence carve-out above.
    # Narrow guard: BOTH names must end in the same metric-family suffix.
    same_family_suffixes = ("_accuracy", "_solve_rate", "_pass_rate")
    for s in same_family_suffixes:
        if k1.endswith(s) and k2.endswith(s):
            return True
    return False


def check_implausible_perfect(d: dict[str, Any], flags: list[Flag]) -> None:
    """Detect implausibly perfect metrics (TPR/acc=1.0, error=0.0)."""
    perfect_score_fields = (
        "tpr", "accuracy", "auroc", "f1", "precision", "recall",
        "pass_rate", "success_rate", "agreement_rate",
    )
    perfect_error_fields = (
        "error", "loss", "delta", "divergence", "violations",
    )

    for k, v in d.items():
        if not _is_finite_number(v):
            continue
        kl = k.lower()
        vf = float(v)
        # Implausible 1.0 on a [0,1] score field
        if any(s in kl for s in perfect_score_fields) and vf == 1.0:
            n = d.get("n_samples") or d.get("n_adversarial_examples") or 0
            try:
                n = int(n)
            except (TypeError, ValueError):
                n = 0
            if n > 0 and n < 1000:
                flags.append(
                    Flag(
                        kind="IMPLAUSIBLE_PERFECT",
                        severity="warn",
                        detail=(
                            f"{k}={vf} on {n} samples is implausibly clean. "
                            f"Real classifiers exhibit non-zero error at small N."
                        ),
                    )
                )
        # Implausible 0.0 on an error/loss field
        if any(s in kl for s in perfect_error_fields) and vf == 0.0:
            # Allow 0.0 only if the field name is a clear baseline marker
            if "baseline" not in kl and "zero" not in kl:
                flags.append(
                    Flag(
                        kind="IMPLAUSIBLE_PERFECT",
                        severity="info",
                        detail=(
                            f"{k}={vf} (exactly zero). Confirm this is "
                            f"not a stub default."
                        ),
                    )
                )


def check_sign_anomaly(d: dict[str, Any], flags: list[Flag]) -> None:
    """Detect optimization that went the wrong direction."""
    init_keys = [k for k in d if k.startswith("initial_") and _is_finite_number(d[k])]
    for ik in init_keys:
        fk = "final_" + ik[len("initial_"):]
        if fk not in d or not _is_finite_number(d[fk]):
            continue
        iv = float(d[ik])
        fv = float(d[fk])
        metric_name = ik[len("initial_"):]
        # Energy / loss / error should DECREASE during optimization.
        decrease_expected = any(
            m in metric_name.lower()
            for m in ("energy", "loss", "error", "violation", "regret")
        )
        # Accuracy / reward / score should INCREASE.
        increase_expected = any(
            m in metric_name.lower()
            for m in ("accuracy", "reward", "score", "lift")
        )

        if decrease_expected and fv > iv:
            flags.append(
                Flag(
                    kind="SIGN_ANOMALY",
                    severity="warn",
                    detail=(
                        f"initial_{metric_name}={iv} -> final_{metric_name}={fv} "
                        f"({metric_name} INCREASED). Optimization was supposed "
                        f"to minimize this. Either methodology bug, sign "
                        f"reversal, or a real finding that needs explicit "
                        f"acknowledgment."
                    ),
                )
            )
        elif increase_expected and fv < iv:
            flags.append(
                Flag(
                    kind="SIGN_ANOMALY",
                    severity="warn",
                    detail=(
                        f"initial_{metric_name}={iv} -> final_{metric_name}={fv} "
                        f"({metric_name} DECREASED). Optimization was supposed "
                        f"to maximize this."
                    ),
                )
            )


def _has_compute_bound_marker(d: dict[str, Any]) -> bool:
    """Walk dict for any compute-bound marker string."""
    text = json.dumps(d)
    return any(m in text for m in COMPUTE_BOUND_MARKERS)


def _is_verifier_scoring_only(d: dict[str, Any]) -> bool:
    """True if the artifact declares it scored verifiers against
    cached candidate triples without invoking LLM inference.

    The GGUF / CUDA markers in such artifacts are vestigial template
    declarations -- the experiment's wall-clock should be measured
    against the verifier-scoring loop, not against model loading +
    inference. Two recognition modes (either is sufficient):

    1. Explicit declaration -- `inference_substrate` field equals the
       canonical sentinel `verifier_ensemble_against_cached_candidates`.
       This is the forward-only path; planner-emitted task prompts
       should require this declaration for verifier-scoring tasks.

    2. Known verifier-only schema prefix. Covers historical artifacts
       (exp2837 fover_memory_leakage_v3, cross_corpus_verifier_matrix
       family, etc.) that were authored before the explicit-field
       discipline shipped.
    """
    if d.get("inference_substrate") == VERIFIER_SCORING_SUBSTRATE:
        return True
    schema = str(d.get("schema") or d.get("schema_version") or "")
    return any(schema.startswith(p) for p in VERIFIER_SCORING_SCHEMA_PREFIXES)


def _is_aggregation_only(d: dict[str, Any]) -> bool:
    """True if the artifact is a synthesis / aggregation over upstream
    artifacts and not itself a compute-bound experiment.

    Capstones, archive/activate transitions, and paper-table builders
    legitimately finish in milliseconds. Their model_specs / GGUF
    markers (when present) are inherited from upstream sources cited
    in the artifact body, not invoked by the artifact itself.
    """
    if d.get("inference_substrate") == AGGREGATION_SUBSTRATE:
        return True
    schema = str(d.get("schema") or d.get("schema_version") or "")
    return any(schema.startswith(p) for p in AGGREGATION_SCHEMA_PREFIXES)


def _is_deterministic_verifier(d: dict[str, Any]) -> bool:
    """True when the artifact declares replay over checked-in verifier evidence."""
    return d.get("inference_substrate") in DETERMINISTIC_VERIFIER_SUBSTRATES


def check_duration_vs_claim(d: dict[str, Any], flags: list[Flag]) -> None:
    """Compute-bound artifact with implausibly short duration."""
    duration = d.get("duration_s")
    if not _is_finite_number(duration):
        return
    if not _has_compute_bound_marker(d):
        return
    # Verifier-scoring artifacts run in seconds because they score
    # cached candidates -- their GGUF markers are vestigial. Apply
    # the tighter verifier-scoring threshold instead of the full
    # model-inference threshold.
    if _is_verifier_scoring_only(d):
        if float(duration) < VERIFIER_SCORING_MIN_DURATION_S:
            flags.append(
                Flag(
                    kind="DURATION_TOO_SHORT",
                    severity="critical",
                    detail=(
                        f"duration_s={duration} but artifact declares "
                        f"verifier-scoring substrate. Even verifier "
                        f"scoring of a few hundred candidates takes "
                        f">={VERIFIER_SCORING_MIN_DURATION_S}s; this "
                        f"completed too fast to have scored anything."
                    ),
                )
            )
        return
    # Aggregation-only artifacts (capstones, archive/activate,
    # paper-table builders) just read upstream JSON and arithmetic.
    # Milliseconds are honest. The GGUF markers are inherited from
    # the upstream artifacts they cite, not invoked here.
    if _is_aggregation_only(d):
        if float(duration) < AGGREGATION_MIN_DURATION_S:
            flags.append(
                Flag(
                    kind="DURATION_TOO_SHORT",
                    severity="critical",
                    detail=(
                        f"duration_s={duration} but artifact declares "
                        f"aggregation substrate. Even loading upstream "
                        f"JSON takes microseconds; a value below "
                        f"{AGGREGATION_MIN_DURATION_S}s suggests the "
                        f"duration was not measured at all."
                    ),
                )
            )
        return
    if _is_deterministic_verifier(d):
        if float(duration) < DETERMINISTIC_VERIFIER_MIN_DURATION_S:
            flags.append(
                Flag(
                    kind="DURATION_TOO_SHORT",
                    severity="critical",
                    detail=(
                        f"duration_s={duration} but artifact declares "
                        f"deterministic-verifier substrate. Even loading "
                        f"checked-in JSON takes microseconds; a value below "
                        f"{DETERMINISTIC_VERIFIER_MIN_DURATION_S}s suggests "
                        f"the duration was not measured at all."
                    ),
                )
            )
        return
    if float(duration) < COMPUTE_BOUND_MIN_DURATION_S:
        flags.append(
            Flag(
                kind="DURATION_TOO_SHORT",
                severity="critical",
                detail=(
                    f"duration_s={duration} but artifact references "
                    f"compute-bound markers (GGUF / CUDA / live model). "
                    f"Loading and running a real model takes "
                    f">={COMPUTE_BOUND_MIN_DURATION_S}s minimum; this "
                    f"completed too fast to have invoked the model."
                ),
            )
        )


def check_sample_size(d: dict[str, Any], flags: list[Flag]) -> None:
    """Distributional claims with sample size below statistical threshold."""
    n = d.get("n_samples") or d.get("n_examples")
    if n is None:
        return
    try:
        n = int(n)
    except (TypeError, ValueError):
        return

    has_kl = "kl_divergence" in d or "kl" in d
    has_ks = "ks_p_value" in d or "ks_statistic" in d
    has_dist_mean = any(k.startswith("mean_") and "delta" in k for k in d)

    n_spins = d.get("n_spins")
    if has_kl or has_ks or has_dist_mean:
        # Heuristic: for an n_spins=N substrate, you need many more
        # samples than 2^(N/k) for some k. At minimum, 10k samples
        # for any non-toy claim.
        min_required = 1000
        if isinstance(n_spins, (int, float)) and n_spins >= 32:
            min_required = max(min_required, int(10 * n_spins))
        if isinstance(n_spins, (int, float)) and n_spins >= 64:
            min_required = max(min_required, 10000)
        if n < min_required:
            flags.append(
                Flag(
                    kind="SAMPLE_SIZE_BELOW_CLAIM",
                    severity="warn",
                    detail=(
                        f"n_samples={n} below statistical threshold "
                        f"~{min_required} for distributional claims "
                        f"(KL/KS/mean delta) at n_spins={n_spins!r}. "
                        f"KL is dominated by sample-size noise; ks_p "
                        f"can pass spuriously on small N."
                    ),
                )
            )


def check_gate_passed_without_data(d: dict[str, Any], flags: list[Flag]) -> None:
    """acceptance_gate_passed=true but key metric fields are null/missing/0."""
    if d.get("acceptance_gate_passed") is not True:
        return
    # Look for keys mentioned in the gate definitions: anything with
    # "gate", "threshold", "delta", "rate", "score". If any is null/0,
    # flag.
    suspect = []
    for k, v in d.items():
        kl = k.lower()
        if any(s in kl for s in ("threshold", "delta", "rate", "score", "lift", "ratio")):
            if v is None:
                suspect.append(f"{k}=null")
            elif _is_finite_number(v) and float(v) == 0.0:
                suspect.append(f"{k}=0")
    if suspect:
        flags.append(
            Flag(
                kind="GATE_PASSED_WITHOUT_DATA",
                severity="critical",
                detail=(
                    f"acceptance_gate_passed=true but these fields are "
                    f"null/zero: {', '.join(suspect[:6])}"
                ),
            )
        )


def check_methodology_present(d: dict[str, Any], flags: list[Flag]) -> None:
    """Compute-bound artifact missing methodology evidence."""
    if not _has_compute_bound_marker(d):
        return
    # Aggregation-only artifacts inherit methodology from the upstream
    # sources they cite; they aren't themselves a measurement, so this
    # check would be a category error.
    if _is_aggregation_only(d):
        return
    has_model_spec = (
        d.get("model_specs") or d.get("target_model") or d.get("models_tested")
    )
    # Recognize singular (`random_seed`, `seed`) and plural
    # (`random_seeds_used`, `seeds`) forms. Multi-seed experiments
    # legitimately use the plural list-of-seeds form.
    has_seed = (
        d.get("random_seed") is not None
        or d.get("seed") is not None
        or d.get("random_seeds_used")
        or d.get("seeds")
    )
    has_repro = d.get("reproducibility_checksum")
    missing = []
    if not has_model_spec:
        missing.append("model_specs/target_model")
    if not has_seed:
        missing.append("random_seed")
    if not has_repro:
        missing.append("reproducibility_checksum")
    if missing:
        flags.append(
            Flag(
                kind="METHODOLOGY_MISSING",
                severity="warn",
                detail=(
                    f"Compute-bound artifact missing: "
                    f"{', '.join(missing)}. Methodology unverifiable."
                ),
            )
        )


def check_implausible_tight_ci(d: dict[str, Any], flags: list[Flag]) -> None:
    """Detect bootstrap/empirical CIs much tighter than sample-variance floor.

    The minimum CI half-width on a stochastic estimator at N seeds is
    approximately sigma_est / sqrt(N), where sigma_est is the metric's
    natural sample stddev (heuristically ~ 0.1-0.5 of the metric value
    for a [0,1]-scale measurement, or ~ 0.2 on dimensionless deltas).

    A reported 95% CI whose total width is below 0.1 / sqrt(N) on a
    metric near 0.1+ is at least an order of magnitude tighter than
    the variance floor — that signals either a deterministic-by-
    construction invariant (which must be disclosed) or a bootstrap
    bug (e.g., one seed reshuffled N times).

    This rule was added 2026-05-15 after exp1693 (`.171 Phase 4
    delta_alpha at n=64) reported [0.15040, 0.15070] CI at N=30
    seeds — 100x tighter than the floor. The literature finding
    (arXiv:2512.15605 AR-LM↔EBM bijection) provides a plausible
    explanation (alpha_t is bijection-invariant by construction),
    but that finding came AFTER the measurement; this linter would
    have flagged it at task-completion-time.
    """
    # Look for fields named *_bootstrap_ci_95, *_ci, *_confidence_interval
    # whose value is a 2-element list [lower, upper].
    ci_field_suffixes = (
        "_bootstrap_ci_95", "_bootstrap_ci_90", "_bootstrap_ci",
        "_ci_95", "_ci_90", "_ci",
        "_confidence_interval", "_credible_interval",
    )

    n_seeds_raw = d.get("n_seeds") or d.get("seeds") or 0
    try:
        n_seeds = int(n_seeds_raw)
    except (TypeError, ValueError):
        n_seeds = 0

    # If we don't know N, we can't bound the floor — fall back to a
    # conservative absolute-width gate at 1e-3 on any reported CI.
    # That catches the exp1693 pattern even if n_seeds is absent.
    if n_seeds <= 0:
        # Still flag absurdly-tight CIs even without N — width < 1e-3
        # on a metric in the [0, 1] range is implausible.
        n_seeds_for_floor = 1000  # treat as "even at N=1000 this is tight"
    else:
        n_seeds_for_floor = n_seeds

    # Heuristic variance floor: assume sigma_est >= 0.05 on the metric
    # scale (very forgiving — most real measurements have larger
    # variance than this).
    assumed_sigma = 0.05
    floor_half_width = assumed_sigma / math.sqrt(n_seeds_for_floor)
    # 95% CI half-width is ~1.96 * sigma / sqrt(N), so full width
    # floor is ~3.92 * sigma / sqrt(N).
    floor_full_width = 3.92 * floor_half_width
    # Discount the floor by 10x to avoid false positives on
    # genuinely-well-converged measurements.
    flagging_threshold = floor_full_width / 10.0

    for k, v in d.items():
        if not isinstance(v, list) or len(v) != 2:
            continue
        if not all(_is_finite_number(x) for x in v):
            continue
        if not any(k.lower().endswith(s) for s in ci_field_suffixes):
            continue
        lo, hi = float(v[0]), float(v[1])
        if hi <= lo:
            continue  # invalid CI, ignore
        ci_width = hi - lo
        # Skip CIs whose metric scale is itself tiny (avoid false
        # positives on metrics that genuinely live near zero, like
        # variance or KL on near-identical distributions).
        midpoint = abs((lo + hi) / 2.0)
        if midpoint < 0.01:
            continue
        if ci_width < flagging_threshold:
            n_seeds_msg = (
                f"n_seeds={n_seeds}" if n_seeds > 0 else "n_seeds unknown"
            )
            flags.append(
                Flag(
                    kind="IMPLAUSIBLE_TIGHT_CI",
                    severity="warn",
                    detail=(
                        f"{k}={v} has CI width {ci_width:.2g} on midpoint "
                        f"{midpoint:.3g} ({n_seeds_msg}). Sample-variance "
                        f"floor at sigma>=0.05 and N>={n_seeds_for_floor} "
                        f"is ~{floor_full_width:.2g}; observed CI is "
                        f"{floor_full_width/max(ci_width,1e-12):.0f}x tighter. "
                        f"Likely deterministic-by-construction (should be "
                        f"disclosed in methodology_note) OR bootstrap bug "
                        f"(e.g., one seed reshuffled instead of N independent "
                        f"draws)."
                    ),
                )
            )


def _flatten_metrics(d: dict[str, Any]) -> dict[str, Any]:
    """Flatten common nested-metric dicts (`metrics`, `report`,
    `summary`, etc.) up to the top level so the checks see them.
    Preserves top-level keys; nested keys merged with their parent
    name stripped (e.g. `metrics.initial_energy` -> `initial_energy`).
    """
    out: dict[str, Any] = dict(d)
    nested_containers = ("metrics", "report", "summary", "case_scores", "results")
    for container in nested_containers:
        v = d.get(container)
        if isinstance(v, dict):
            for k, sub in v.items():
                if k not in out:
                    out[k] = sub
    return out


def verify_artifact(path: Path) -> dict[str, Any]:
    """Run all checks on a single artifact. Return a report dict."""
    try:
        with open(path) as f:
            d_raw = json.load(f)
    except Exception as e:
        return {
            "artifact": str(path),
            "loaded": False,
            "error": str(e),
            "flags": [],
        }

    flags: list[Flag] = []

    # Skip non-dict top-level artifacts (some experiments emit a list
    # at top level — those aren't standard results files).
    if not isinstance(d_raw, dict):
        return {
            "artifact": str(path),
            "loaded": True,
            "non_dict_top_level": True,
            "flags": [],
        }

    schema_raw = d_raw.get("schema", "") or ""
    schema = schema_raw if isinstance(schema_raw, str) else ""

    # Skip blocked-gate artifacts — those are conductor diagnostic
    # outputs, not experimental results.
    if schema.startswith("blocked"):
        return {
            "artifact": str(path),
            "loaded": True,
            "blocked_gate_artifact": True,
            "flags": [],
        }

    # Skip retrospective artifacts — they summarize other experiments
    # and reference compute-bound markers without invoking models
    # themselves. They have a different verification profile (their
    # rigor is "did they read the artifacts they claim to summarize"
    # which is a different check entirely).
    title_raw = d_raw.get("title") or d_raw.get("name") or ""
    title = title_raw if isinstance(title_raw, str) else ""
    is_retro = (
        "retro" in path.name.lower()
        or "retro" in schema.lower()
        or "retrospective" in title.lower()
    )
    if is_retro:
        return {
            "artifact": str(path),
            "loaded": True,
            "is_retro": True,
            "exp_id": d_raw.get("experiment") or d_raw.get("experiment_id"),
            "title": title[:80],
            "flag_count": 0,
            "max_severity": -1,
            "flags": [],
        }

    # Flatten nested-metric containers (e.g. `metrics`,`report`) so
    # checks see the inner fields even when the agent emitted them
    # one level deep. Several agents emit
    # `{"metrics": {"initial_energy": ..., "final_energy": ...}}`
    # instead of top-level fields.
    d = _flatten_metrics(d_raw)

    check_tautology(d, flags)
    check_implausible_perfect(d, flags)
    check_sign_anomaly(d, flags)
    check_duration_vs_claim(d, flags)
    check_sample_size(d, flags)
    check_gate_passed_without_data(d, flags)
    check_methodology_present(d, flags)
    check_implausible_tight_ci(d, flags)
    check_false_negative_risk(d, flags)

    verdict_raw = d_raw.get("honest_verdict") or ""
    verdict = verdict_raw if isinstance(verdict_raw, str) else ""
    return {
        "artifact": str(path),
        "loaded": True,
        "exp_id": d_raw.get("experiment") or d_raw.get("experiment_id"),
        "title": title[:80],
        "honest_verdict": verdict[:80],
        "flag_count": len(flags),
        "max_severity": max(
            (Flag.SEVERITY_RANK[f.severity] for f in flags), default=-1
        ),
        "flags": [f.to_dict() for f in flags],
    }


def sweep_milestone_range(
    results_dir: Path, low: int, high: int
) -> list[dict[str, Any]]:
    """Verify all experiment_NNNN_*.json files where low <= NNNN <= high."""
    reports = []
    pattern = re.compile(r"experiment_(\d+)_.*\.json$")
    for path in sorted(results_dir.glob("experiment_*.json")):
        m = pattern.match(path.name)
        if not m:
            continue
        exp = int(m.group(1))
        if low <= exp <= high:
            reports.append(verify_artifact(path))
    return reports


# High-precision fabrication-signal kinds. These are safe to backfill-stamp
# retroactively across all of history because a false positive is very unlikely:
# a duration_s far below the model-load floor while claiming a live GGUF/CUDA
# model is almost certainly fabricated, and a gate marked passed with the gated
# metric null/missing is structurally untrustworthy. TAUTOLOGY is deliberately
# NOT here — retroactively it over-flags legitimate coincidental old findings
# (e.g. abstention having no measured effect, or a tiny-corpus recall tie), so
# historical TAUTOLOGY is left to the completion-time gate + operator review.
HIGH_PRECISION_KINDS = ("DURATION_TOO_SHORT", "GATE_PASSED_WITHOUT_DATA")


def _claims_live_model(d: dict[str, Any]) -> bool:
    """True if the artifact AFFIRMATIVELY claims it ran a named live model
    (declares model_specs / target_model / models, or
    inference_substrate=live_llm_inference). DURATION_TOO_SHORT is only a real
    fabrication signal when paired with such a claim — an aggregation/audit
    artifact that merely mentions 'GGUF' in prose but declares no model didn't
    claim a live run, so a sub-floor duration is expected, not suspicious. This
    guard keeps the retroactive backfill high-precision (see exp1877/1498/1459
    aggregation false positives vs exp1851/1782 real live-claim fabrications)."""
    if str(d.get("inference_substrate", "")).lower() == "live_llm_inference":
        return True
    for key in ("model_specs", "target_model", "models", "model"):
        v = d.get(key)
        if v:
            return True
    return False


def backfill_stamps(
    paths: list[Path],
    apply: bool = False,
    kinds_filter: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Backstop for the conductor's completion-time fabrication gate.

    The gate in `research_conductor._log_experiment_completion` only fires for
    artifacts that complete inside `research_step`. Artifacts written out-of-band
    (manual reruns, batch scripts, historical pre-gate experiments) escape it.
    This sweep re-verifies a set of artifacts and stamps `flagged_adversarial`
    on any UNSTAMPED one whose critical flags intersect `kinds_filter`
    (None = any critical kind).

    NON-DESTRUCTIVE: only ADDS `flagged_adversarial: true` +
    `corrigendum_pending` (the offending flags) + a `corrigendum_note`. Never
    deletes or alters existing fields, per the never-prune discipline. Idempotent
    (skips already-stamped artifacts).

    Returns one record per artifact that HAS a qualifying critical flag, whether
    or not it was written (so dry-run reports the full scope).
    """
    out: list[dict[str, Any]] = []
    for p in paths:
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if not isinstance(d, dict) or d.get("flagged_adversarial"):
            continue
        try:
            rep = verify_artifact(p)
        except Exception:
            continue
        crit = [
            f for f in rep.get("flags", [])
            if str(f.get("severity", "")).lower() == "critical"
        ]
        if kinds_filter is not None:
            crit = [f for f in crit if f.get("kind") in kinds_filter]
        # Precision guard: DURATION_TOO_SHORT only counts when the artifact
        # affirmatively claims a live model run. Otherwise an aggregation/audit
        # artifact that merely references compute markers in prose would be
        # mislabeled (the operator's explicit false-positive concern).
        crit = [
            f
            for f in crit
            if f.get("kind") != "DURATION_TOO_SHORT" or _claims_live_model(d)
        ]
        if not crit:
            continue
        kinds = sorted({str(f.get("kind")) for f in crit})
        written = False
        if apply:
            d["flagged_adversarial"] = True
            d.setdefault("corrigendum_pending", []).extend(crit)
            d["corrigendum_note"] = (
                "backfill_stamps: flagged_adversarial added by "
                "adversarial_verify.py --backfill (completion-gate backstop). "
                "Excluded from headline / capstone aggregation."
            )
            p.write_text(json.dumps(d, indent=2))
            written = True
        out.append({"path": str(p), "kinds": kinds, "written": written})
    return out


def main(argv: list[str]) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="*", help="Specific artifact files to verify")
    parser.add_argument(
        "--milestone-range",
        nargs=2,
        type=int,
        metavar=("LOW", "HIGH"),
        help="Sweep experiment_LOW..HIGH inclusive",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results",
    )
    parser.add_argument("--json", action="store_true", help="Output full JSON report")
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Completion-gate backstop: scan results/ and stamp unstamped "
        "real-critical artifacts (dry-run unless --apply).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="With --backfill: actually write the stamps (default is dry-run).",
    )
    parser.add_argument(
        "--high-precision-only",
        action="store_true",
        help="With --backfill: restrict to %s (safe for retroactive history)."
        % ", ".join(HIGH_PRECISION_KINDS),
    )
    parser.add_argument(
        "--since-hours",
        type=float,
        default=None,
        help="With --backfill: only artifacts modified within the last N hours.",
    )
    args = parser.parse_args(argv[1:])

    if args.backfill:
        import time

        paths = sorted(args.results_dir.glob("experiment_*.json"))
        if args.since_hours is not None:
            cutoff = time.time() - args.since_hours * 3600.0
            paths = [p for p in paths if p.stat().st_mtime >= cutoff]
        kinds_filter = HIGH_PRECISION_KINDS if args.high_precision_only else None
        recs = backfill_stamps(paths, apply=args.apply, kinds_filter=kinds_filter)
        mode = "APPLIED" if args.apply else "DRY-RUN"
        scope = (
            f"high-precision({','.join(HIGH_PRECISION_KINDS)})"
            if args.high_precision_only
            else "any-critical"
        )
        win = f", last {args.since_hours}h" if args.since_hours else ""
        print(
            f"[backfill {mode}] scanned {len(paths)} artifact(s){win}; scope={scope}; "
            f"{len(recs)} qualifying unstamped critical artifact(s):"
        )
        for r in recs:
            tag = "stamped" if r["written"] else "would-stamp"
            print(f"  [{tag}] {Path(r['path']).name}: {r['kinds']}")
        return 1 if recs else 0

    reports: list[dict[str, Any]] = []
    if args.milestone_range:
        reports = sweep_milestone_range(
            args.results_dir, args.milestone_range[0], args.milestone_range[1]
        )
    for a in args.artifacts:
        reports.append(verify_artifact(Path(a)))

    if not reports:
        parser.print_help()
        return 2

    flagged = [r for r in reports if r.get("flag_count", 0) > 0]

    if args.json:
        print(json.dumps({"reports": reports, "flagged_count": len(flagged)}, indent=2))
    else:
        print(f"Scanned {len(reports)} artifact(s); {len(flagged)} flagged.\n")
        for r in flagged:
            sev_label = {2: "CRITICAL", 1: "WARN", 0: "INFO"}.get(
                r.get("max_severity", -1), "?"
            )
            print(
                f"[{sev_label}] exp{r.get('exp_id')} ({r.get('title')}) -- "
                f"{r.get('flag_count')} flag(s)"
            )
            for f in r.get("flags", []):
                print(f"  - {f['severity'].upper():8s} {f['kind']:30s} {f['detail']}")
            print()

    return 1 if flagged else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv))
