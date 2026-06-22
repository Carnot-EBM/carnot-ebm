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

  DEGENERATE_SEPARATION
    A selector-vs-vote transfer artifact with a near-perfect delta
    while vote or matched control is near zero, OR a perfect
    set-encoder selector with oracle@K saturated at 1.0. Example:
    exp4282 reported cross_family_delta=1.0, vote_at_1=0.0,
    set_encoder_at_1=1.0, and oracle_at_k=1.0 on a wrong-majority
    ARC-GEN pool; that is a pool-construction artifact, not transfer.

  DEGENERATE_CONTROLS
    A condition-arm artifact whose distinct control arms report
    bit-identical accuracy/pass-rate values. Example: exp4293 reported
    rfg == unguided == entrgi in condition_accuracy, which is the
    signature of no-op or aliased controls rather than differentiated
    baselines.

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

DEGENERATE_DELTA_THRESHOLD = 0.95
DEGENERATE_BASELINE_THRESHOLD = 0.05
DEGENERATE_DELTA_KEYS = ("cross_generator_delta", "cross_family_delta")
DEGENERATE_BASELINE_KEYS = ("vote_at_1", "matched_control_at_1")
CONDITION_ARM_MAP_KEYS = ("condition_accuracy", "per_condition_accuracy", "arms")
CONDITION_ARM_METRIC_KEYS = (
    "condition_accuracy",
    "per_condition_accuracy",
    "accuracy",
    "pass_rate",
    "success_rate",
    "solve_rate",
    "score",
)
CONTROL_ARM_MARKERS = (
    "baseline",
    "control",
    "entrgi",
    "no_guidance",
    "no-guidance",
    "random",
    "rfg",
    "self_guidance",
    "self-guidance",
    "unguided",
    "unconditioned",
)
PLACEBO_OR_REPLICATE_MARKERS = ("placebo", "replicate", "replica", "duplicate")

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
    "torch",
    "torch.cuda",
    ".cuda(",
)

# Minimum duration (seconds) for a compute-bound artifact. Loading +
# inference on even a 0.5B GGUF takes at least this long.
COMPUTE_BOUND_MIN_DURATION_S = 60.0
LIVE_LLM_SUBSTRATE = "live_llm_inference"

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

# Offline ARC solve / learned-verifier artifacts do not have a model to name:
# their methodology is the solver entrypoint, reproduce gate/checksum, and
# learned-verifier checkpoint. Treat those fields as the methodology descriptor
# only when the substrate already says this is cached-candidate verifier work or
# upstream aggregation, never for live LLM inference.
OFFLINE_ARC_METHOD_DESCRIPTOR_KEYS = (
    "offline_reproduced",
    "primitive_persisted",
    "solver_module",
    "solver_modules",
    "reproduction_gate",
    "reproduce_gate",
    "reproduce_gate_checksum",
    "verifier_checkpoint",
    "verifier_checkpoints",
    "learned_verifier_checkpoint",
)
OFFLINE_ARC_DESCRIPTOR_METADATA_KEYS = frozenset(
    {
        "field_principles",
        "required_artifact_fields",
        "tests_added_pass",
    }
)


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


_POSITIVE_CONTROL_NULL_VERDICT_MARKERS = (
    "honest_null",
    "null",
    "no_deeper",
    "no_value",
    "no_value_added",
    "no_lever",
    "no_lift",
    "no_gain",
    "no_delta",
    "unmoved",
    "unchanged",
    "positive_control_failed",
)
_POSITIVE_CONTROL_NULL_DOMAIN_MARKERS = (
    "efficiency",
    "transfer",
    "proposer",
    "reinduction",
    "llm_proposer",
    "core_efficiency",
    "generic_transfer",
)
_POSITIVE_CONTROL_NULL_DELTA_KEYS = (
    "efficiency_delta",
    "core_efficiency_delta",
    "generic_transfer_delta",
    "transfer_delta",
)


def _finite_float(d: dict[str, Any], key: str) -> float | None:
    value = d.get(key)
    return float(value) if _is_finite_number(value) else None


def _metric_pair_equal(d: dict[str, Any], left: str, right: str) -> bool:
    left_value = _finite_float(d, left)
    right_value = _finite_float(d, right)
    if left_value is None or right_value is None:
        return False
    return math.isclose(left_value, right_value, rel_tol=0.0, abs_tol=1e-12)


def _has_positive_control_null_metric(d: dict[str, Any]) -> bool:
    for key in _POSITIVE_CONTROL_NULL_DELTA_KEYS:
        value = _finite_float(d, key)
        if value is not None and math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1e-12):
            return True
    return any(
        _metric_pair_equal(d, left, right)
        for left, right in (
            ("core_efficiency_baseline", "core_efficiency_best"),
            ("core_efficiency_baseline", "core_efficiency_integrated"),
            ("generic_transfer_rate_baseline", "generic_transfer_rate_with_verifier"),
            ("generic_transfer_rate_baseline", "generic_transfer_rate_integrated"),
        )
    )


def _is_positive_control_null_claim(d: dict[str, Any], verdict: str) -> bool:
    """True for efficiency/proposer/transfer nulls that require a positive control."""
    if not (
        "positive_control_passed" in d
        or "false_negative_risk_checked" in d
        or "positive_control" in verdict
    ):
        return False
    text = " ".join([verdict, " ".join(str(key).lower() for key in d)])
    if not any(marker in text for marker in _POSITIVE_CONTROL_NULL_DOMAIN_MARKERS):
        return False
    verdict_declares_null = any(
        marker in verdict for marker in _POSITIVE_CONTROL_NULL_VERDICT_MARKERS
    )
    return verdict_declares_null or _has_positive_control_null_metric(d)


def _positive_control_failed_or_unchecked(d: dict[str, Any]) -> bool:
    """A null is informative only when the positive control and FNR check passed."""
    return d.get("positive_control_passed") is not True or d.get(
        "false_negative_risk_checked"
    ) is not True


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
    if _is_positive_control_null_claim(d, verdict) and _positive_control_failed_or_unchecked(d):
        flags.append(
            Flag(
                kind="FALSE_NEGATIVE_RISK",
                severity="warn",
                detail=(
                    "false_negative_risk_open: null efficiency/proposer/transfer "
                    f"claim ({verdict[:64]!r}) lacks a passed positive control "
                    "and completed false-negative-risk check "
                    f"(positive_control_passed={d.get('positive_control_passed')!r}, "
                    "false_negative_risk_checked="
                    f"{d.get('false_negative_risk_checked')!r}). Treat this as a "
                    "broken-test signal, not as evidence for a clean null."
                ),
            )
        )
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


_COMPARISON_VERDICT_MARKERS = (
    "beats", "_vs_", "generaliz", "outperform", "superior", "wins",
    "better_than", "_beat_", "dominates",
)
_TRIVIAL_BASELINE_MARKERS = (
    "vanilla", "greedy", "random", "single", "descent", "naive", "trivial",
    "default", "sequential", "baseline",
)


def _is_comparative_claim(d: dict[str, Any]) -> bool:
    """True if the artifact asserts one method beats another — the precondition
    for a CEILING_SATURATION false positive to matter."""
    v = str(d.get("honest_verdict", "")).lower()
    if any(m in v for m in _COMPARISON_VERDICT_MARKERS):
        return True
    # a weak baseline field strictly below the headline rate (exclude the
    # exact/oracle upper bound, which is supposed to be at the ceiling)
    headline = d.get("solve_rate") or d.get("accuracy") or d.get("pass_rate")
    if _is_finite_number(headline):
        for k, vv in d.items():
            kl = k.lower()
            if "baseline" in kl and "exact" not in kl and "oracle" not in kl:
                if _is_finite_number(vv) and float(vv) < float(headline):
                    return True
    return False


def check_ceiling_saturation(d: dict[str, Any], flags: list[Flag]) -> None:
    """Positive-claim partner to FALSE_NEGATIVE_RISK. A method-superiority claim
    is uninformative if the corpus is ceiling-saturated: every method variant
    (including a TRIVIAL baseline) and/or every difficulty tier hits the same
    maximum. Then the corpus has no headroom on the METHOD side — it cannot
    distinguish a powerful method from a trivial one, so "method X generalizes /
    beats Y" over-claims.

    Origin: exp3518 reported "energy global inference generalizes to graph
    coloring, solve_rate 1.00 vs AR 0.50", but vanilla_descent (no annealing, no
    tempering) ALSO solved 100% on the 'extreme' difficulty tier — all 5
    optimizers and all 4 difficulties tied at 1.0. The win only proves greedy-AR
    has a known ordering pathology, not that energy inference is uniquely
    capable. Only fires on comparative claims (gated) to avoid flagging
    legitimately-easy sanity checks that make no superiority claim."""
    if not _is_comparative_claim(d):
        return
    CEIL = 0.99

    # Signal 1 — a trivial baseline variant saturates alongside the method.
    for k, v in d.items():
        if not isinstance(v, dict):
            continue
        kl = k.lower()
        if not any(
            s in kl
            for s in ("by_optimizer", "by_variant", "by_method", "by_model",
                      "by_approach", "by_sampler", "by_solver")
        ):
            continue
        nums = {kk: float(vv) for kk, vv in v.items() if _is_finite_number(vv)}
        if len(nums) < 2 or max(nums.values()) < CEIL:
            continue
        at_ceiling = [kk for kk, vv in nums.items() if vv >= CEIL]
        trivial = [
            kk for kk in at_ceiling
            if any(m in kk.lower() for m in _TRIVIAL_BASELINE_MARKERS)
        ]
        if len(at_ceiling) >= 2 and trivial:
            flags.append(
                Flag(
                    kind="CEILING_SATURATION",
                    severity="warn",
                    detail=(
                        f"{k}: a trivial baseline ({trivial[0]}) also saturates "
                        f"at the ceiling (>= {CEIL}); {len(at_ceiling)} variants "
                        f"tie at {max(nums.values())}. The corpus cannot "
                        f"discriminate a powerful method from a trivial one, so "
                        f"the superiority claim is uninformative. Harden the "
                        f"corpus until the trivial baseline drops below ceiling."
                    ),
                )
            )

    # Signal 2 — every difficulty tier saturates: the difficulty axis is inert.
    for k, v in d.items():
        if not isinstance(v, dict):
            continue
        kl = k.lower()
        if not any(
            s in kl for s in ("by_difficulty", "by_tier", "by_hardness", "by_level")
        ):
            continue
        nums = {kk: float(vv) for kk, vv in v.items() if _is_finite_number(vv)}
        if len(nums) >= 2 and min(nums.values()) >= CEIL:
            flags.append(
                Flag(
                    kind="CEILING_SATURATION",
                    severity="warn",
                    detail=(
                        f"{k}: every difficulty tier saturates at the ceiling "
                        f"({sorted(nums.items())}). The hardest tier is as easy "
                        f"as the easiest, so the difficulty axis is inert and a "
                        f"'solves hard instances' claim is unsupported. Add "
                        f"genuinely harder instances until the top tier drops."
                    ),
                )
            )


def _metric_from_top_or_pass_rates(d: dict[str, Any], key: str) -> float | None:
    value = d.get(key)
    if _is_finite_number(value):
        return float(value)
    pass_rates = d.get("pass_rates")
    if isinstance(pass_rates, dict) and _is_finite_number(pass_rates.get(key)):
        return float(pass_rates[key])
    return None


def _metric_items_from_top_or_pass_rates(
    d: dict[str, Any], keys: tuple[str, ...]
) -> list[tuple[str, float]]:
    items: list[tuple[str, float]] = []
    pass_rates = d.get("pass_rates")
    for key in keys:
        value = d.get(key)
        if _is_finite_number(value):
            items.append((key, float(value)))
            continue
        if isinstance(pass_rates, dict) and _is_finite_number(pass_rates.get(key)):
            items.append((key, float(pass_rates[key])))
    return items


def check_degenerate_separation(d: dict[str, Any], flags: list[Flag]) -> None:
    """Detect synthetic selection wins where vote cannot win and oracle saturates.

    Exp 4282 exposed a false-positive pattern that the older artifact checks
    missed: a candidate pool filtered to wrong-majority rows with only a few
    candidates per task can produce `delta=1.0`, `vote@1=0.0`, and
    `oracle@K=1.0`. That proves the pool construction is separable, not that a
    learned selector generalized. The guard also catches the saturated-oracle
    variant where a perfect set-encoder selector reaches oracle@K exactly.
    """
    verdict = " ".join(
        str(d.get(key, "")).lower() for key in ("honest_verdict", "experiment", "schema")
    )
    if not any(
        marker in verdict
        for marker in ("arcgen", "cross_generator", "cross_family", "generaliz")
    ):
        return

    deltas = _metric_items_from_top_or_pass_rates(d, DEGENERATE_DELTA_KEYS)
    baselines = _metric_items_from_top_or_pass_rates(d, DEGENERATE_BASELINE_KEYS)
    oracle_at_k = _metric_from_top_or_pass_rates(d, "oracle_at_k")
    set_encoder_at_1 = _metric_from_top_or_pass_rates(d, "set_encoder_at_1")

    near_perfect_delta = [
        (key, value) for key, value in deltas if value >= DEGENERATE_DELTA_THRESHOLD
    ]
    trivial_baseline = [
        (key, value) for key, value in baselines if value <= DEGENERATE_BASELINE_THRESHOLD
    ]
    saturated_oracle_selector = (
        oracle_at_k is not None
        and set_encoder_at_1 is not None
        and math.isclose(oracle_at_k, 1.0, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(set_encoder_at_1, 1.0, rel_tol=0.0, abs_tol=1e-12)
    )

    if not ((near_perfect_delta and trivial_baseline) or saturated_oracle_selector):
        return

    delta_detail = (
        ", ".join(f"{key}={value}" for key, value in near_perfect_delta)
        if near_perfect_delta
        else "no near-perfect delta field"
    )
    baseline_detail = (
        ", ".join(f"{key}={value}" for key, value in trivial_baseline)
        if trivial_baseline
        else "no near-zero baseline field"
    )
    oracle_detail = "missing" if oracle_at_k is None else str(oracle_at_k)
    selector_detail = "missing" if set_encoder_at_1 is None else str(set_encoder_at_1)
    flags.append(
        Flag(
            kind="DEGENERATE_SEPARATION",
            severity="critical",
            detail=(
                f"Selector-vs-vote degenerate signal: {delta_detail}; "
                f"baselines {baseline_detail}; oracle_at_k={oracle_detail}; "
                f"set_encoder_at_1={selector_detail}. This matches a "
                f"wrong-majority/trivial-separation pool signature. Rebuild "
                f"the candidate pool with vote-winning tasks, realistic "
                f"candidate counts, and oracle_at_k<1 before claiming transfer."
            ),
        )
    )


def _is_control_arm_key(key: str) -> bool:
    """True when an arm name describes a baseline/control condition."""
    kl = key.lower()
    return any(marker in kl for marker in CONTROL_ARM_MARKERS)


def _is_placebo_or_replicate_key(key: str) -> bool:
    """True for intentionally duplicated placebo/replicate controls."""
    kl = key.lower()
    return any(marker in kl for marker in PLACEBO_OR_REPLICATE_MARKERS)


def _documented_identical_controls(d: dict[str, Any]) -> bool:
    """True when the artifact explicitly says identical controls are expected."""
    for key, value in d.items():
        kl = key.lower()
        if (
            isinstance(value, bool)
            and value is True
            and "identical" in kl
            and ("control" in kl or "arm" in kl or "placebo" in kl)
        ):
            return True
        if isinstance(value, str):
            vl = value.lower()
            if (
                ("deliberately identical" in vl or "intentionally identical" in vl)
                and ("control" in vl or "arm" in vl or "placebo" in vl)
            ):
                return True
    return False


def _arm_numeric_value(value: Any) -> float | None:
    """Extract the accuracy-like value from a flat or nested arm entry."""
    if _is_finite_number(value):
        return float(value)
    if not isinstance(value, dict):
        return None
    for metric_key in CONDITION_ARM_METRIC_KEYS:
        metric = value.get(metric_key)
        if _is_finite_number(metric):
            return float(metric)
    return None


def check_degenerate_controls(d: dict[str, Any], flags: list[Flag]) -> None:
    """Detect condition-arm maps where distinct controls are bit-identical.

    Exp 4293 exposed a no-op-controls failure mode: the headline arm differed,
    but every control arm reported the exact same aggregate accuracy. That is a
    harness bug signature for in-generation comparisons because model-self,
    unguided, and alternate-control arms should be independently exercised.
    """
    documented_identical = _documented_identical_controls(d)
    for map_key in CONDITION_ARM_MAP_KEYS:
        arm_map = d.get(map_key)
        if not isinstance(arm_map, dict):
            continue
        controls: list[tuple[str, float]] = []
        for arm_key, value in arm_map.items():
            if not isinstance(arm_key, str) or not _is_control_arm_key(arm_key):
                continue
            numeric_value = _arm_numeric_value(value)
            if numeric_value is None:
                continue
            controls.append((arm_key, numeric_value))
        if len(controls) < 2:
            continue

        by_bits: dict[str, list[tuple[str, float]]] = {}
        for arm_key, numeric_value in controls:
            by_bits.setdefault(float(numeric_value).hex(), []).append((arm_key, numeric_value))

        for duplicate_controls in by_bits.values():
            if len(duplicate_controls) < 2:
                continue
            if documented_identical or all(
                _is_placebo_or_replicate_key(arm_key) for arm_key, _ in duplicate_controls
            ):
                continue
            detail = ", ".join(
                f"{arm_key}={numeric_value:g}"
                for arm_key, numeric_value in sorted(duplicate_controls)
            )
            flags.append(
                Flag(
                    kind="DEGENERATE_CONTROLS",
                    severity="critical",
                    detail=(
                        f"{map_key}: distinct control arms have bit-identical "
                        f"values ({detail}). This matches a no-op/aliased-controls "
                        f"signature; rerun with independently exercised controls "
                        f"before claiming the condition-arm comparison."
                    ),
                )
            )


# Tokens an artifact's honest_verdict uses to DECLARE a no-value null result. A genuine fabrication
# would never self-label as a null, so requiring one of these is the load-bearing safety condition
# for the control-vs-treatment carve-out below.
_HONEST_NULL_VERDICT_TOKENS = (
    "honest_null", "no_value_added", "no_lever_raises", "no_value", "no_improvement",
    "no_metric_moved", "no_delta",
)
# Qualifier tokens that mark a metric key as one ARM of a control/treatment ablation (X_baseline vs
# X_with_verifier vs X_integrated). A pair where at least one side carries one of these, in an artifact
# that declares an honest null, is an EXPECTED ablation equality (the treatment changed nothing) — not
# a coincidence between two distinct measurements.
_CONTROL_TREATMENT_QUALIFIERS = (
    "_baseline_reference", "_baseline", "_with_verifier", "_integrated", "_random_router",
    "_control", "_treatment", "_reference", "_ablation",
)


def _is_declared_honest_null(d: dict[str, Any]) -> bool:
    """True if the artifact's honest_verdict declares a no-value/no-lever null result."""
    v = str(d.get("honest_verdict", "")).lower()
    return any(tok in v for tok in _HONEST_NULL_VERDICT_TOKENS)


def _has_control_treatment_qualifier(k: str) -> bool:
    """True if the metric key carries a control/treatment ablation-arm qualifier."""
    kl = k.lower()
    return any(q in kl for q in _CONTROL_TREATMENT_QUALIFIERS)


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
        # Skip DECLARED control-vs-treatment HONEST NULLS. When an ablation artifact's own
        # honest_verdict declares a no-value null (verifier_router_no_value_added,
        # no_lever_raises_a_metric, ...), a control==treatment equality where one side is an
        # ablation arm (X_baseline == X_with_verifier == 0.04) is the EXPECTED, MEANINGFUL outcome —
        # the treatment changed nothing — not a coincidence between two distinct measurements. Gate
        # on BOTH the honest-null verdict (a fabrication never self-labels a null) AND a
        # control/treatment qualifier key, so a generic two-metric coincidence is still flagged.
        # Origin: exp4556 (HEADLINE generic_transfer null) + exp4560 (integration gate null) were
        # spuriously quarantined ~8x across .420/.421, excluding the project's two most important ARC
        # measurements from capstone aggregation. Mirrors the _is_identifier_field carve-out above.
        if _is_declared_honest_null(d) and (
            _has_control_treatment_qualifier(k1) or _has_control_treatment_qualifier(k2)
        ):
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
    same_family_suffixes = ("_accuracy", "_solve_rate", "_pass_rate", "_cross_family_delta")
    for s in same_family_suffixes:
        if k1.endswith(s) and k2.endswith(s):
            return True
    # An INDEPENDENT REPRODUCTION / re-score / clean-room replication of a metric is
    # DESIGNED to reproduce the original value (the G2-reproducer pattern; CLAUDE.md
    # publication gate G2). When such a field matches the metric it reproduces, that is
    # a SUCCESSFUL reproduction — the evidence of reproducibility we WANT — NOT two
    # distinct metrics coinciding by bug. Canonical case: exp4257's
    # `independent_rescore_delta == single_seed_4245_delta` (0.4423) confirming the +44pp
    # ARC oracle-distinct win reproduces off the persisted artifact via a second code
    # path. A reproduction field carries an explicit re-score/replication marker, so this
    # cannot mask a genuine two-distinct-metric coincidence (those have no such marker).
    repro_markers = ("rescore", "re_score", "reproduc", "replicat", "cleanroom", "clean_room")
    kl1, kl2 = k1.lower(), k2.lower()
    if any(m in kl1 for m in repro_markers) or any(m in kl2 for m in repro_markers):
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


def _inference_substrate_text(d: dict[str, Any]) -> str:
    """Return the declared substrate as a stripped string."""
    return str(d.get("inference_substrate") or "").strip()


def _inference_substrate_matches(d: dict[str, Any], canonical: str) -> bool:
    """True when `inference_substrate` declares a canonical substrate value.

    Newer ARC artifacts often store the canonical value followed by a human
    principle explanation (`value -- why this floor applies`). Matching the
    leading value keeps that reader annotation from changing verifier behavior.
    """
    raw = _inference_substrate_text(d)
    if raw == canonical:
        return True
    prefix = raw.split("--", 1)[0].strip()
    return prefix == canonical or raw.startswith(f"{canonical} ")


def _is_live_llm_inference(d: dict[str, Any]) -> bool:
    """True when the artifact declares a live LLM inference substrate."""
    return _inference_substrate_matches(d, LIVE_LLM_SUBSTRATE)


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
    if _inference_substrate_matches(d, VERIFIER_SCORING_SUBSTRATE):
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
    if _inference_substrate_matches(d, AGGREGATION_SUBSTRATE):
        return True
    schema = str(d.get("schema") or d.get("schema_version") or "")
    return any(schema.startswith(p) for p in AGGREGATION_SCHEMA_PREFIXES)


def _is_deterministic_verifier(d: dict[str, Any]) -> bool:
    """True when the artifact declares replay / reconciliation over checked-in evidence.

    Besides the canonical sentinels, recognise substrate names that self-declare
    deterministic replay or ledger reconciliation over checked-in artifacts and make
    NO live LLM call of their own (e.g. registry/gaps-hygiene tasks declaring
    `cached_gap4_replay_and_ledger_reconciliation`). These finish in milliseconds and
    must use the JSON-work floor, not the 60s live-model floor. Their GGUF/CUDA markers
    are inherited from the upstream artifacts they replay, not invoked here. This was a
    recurring DURATION_TOO_SHORT false-positive on .390 infra tasks (B2/D2). Live
    inference artifacts declare `live_llm_inference`, so this never masks a real
    fast-fabrication of a live-model claim. (Inference-Substrate Declaration Discipline.)
    """
    sub = _inference_substrate_text(d)
    if sub in DETERMINISTIC_VERIFIER_SUBSTRATES:
        return True
    return any(tok in sub for tok in ("replay", "reconciliation"))


def _descriptor_key_present(value: Any, wanted: str) -> bool:
    """True if a real artifact field named `wanted` appears outside metadata.

    Several artifacts explain required fields inside `field_principles`; those
    prose-only mentions must not count as methodology evidence. Nested result
    rows are different: exp4572 stores many real `reproduction_gate` objects in
    per-game rows, and those should count.
    """
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in OFFLINE_ARC_DESCRIPTOR_METADATA_KEYS:
                continue
            if key == wanted and nested is not None:
                return True
            if _descriptor_key_present(nested, wanted):
                return True
    elif isinstance(value, list):
        return any(_descriptor_key_present(item, wanted) for item in value)
    return False


def offline_arc_methodology_descriptor(d: dict[str, Any]) -> dict[str, Any] | None:
    """Return the recognized offline ARC methodology descriptor, if present.

    Offline ARC artifacts are compute-bound in the sense that they may mention
    torch/CUDA/CNN artifacts, but their run did not invoke a live LLM. A model
    spec would be misleading there. The auditable methodology is instead:
    an offline solver/reproduce gate or verifier checkpoint plus a stable
    reproducibility checksum. This helper is deliberately gated by substrate so
    a live_llm_inference artifact cannot use these fields to avoid naming the
    model it claims to have run.
    """
    if not (_is_verifier_scoring_only(d) or _is_aggregation_only(d)):
        return None

    evidence_fields = [
        key for key in OFFLINE_ARC_METHOD_DESCRIPTOR_KEYS
        if _descriptor_key_present(d, key)
    ]
    if not evidence_fields or not d.get("reproducibility_checksum"):
        return None

    evidence_with_checksum = sorted(set(evidence_fields + ["reproducibility_checksum"]))
    return {
        "kind": "offline_arc_methodology_descriptor",
        "substrate": (
            VERIFIER_SCORING_SUBSTRATE
            if _is_verifier_scoring_only(d)
            else AGGREGATION_SUBSTRATE
        ),
        "evidence_fields": evidence_with_checksum,
        "reason": (
            "offline ARC solver/reproduce/checkpoint methodology; no live model_specs "
            "required"
        ),
    }


def duration_floor_for_artifact(d: dict[str, Any]) -> dict[str, Any] | None:
    """Return the duration floor selected from the artifact substrate.

    The return value is intentionally small and JSON-like so
    `summarize_artifact.py` can print it directly for reviewer-facing
    diagnostics. `None` means no compute-bound marker or floor-bearing
    substrate was declared.
    """
    if _is_verifier_scoring_only(d):
        return {
            "substrate": VERIFIER_SCORING_SUBSTRATE,
            "min_duration_s": VERIFIER_SCORING_MIN_DURATION_S,
            "reason": "verifier_scoring",
        }
    if _is_aggregation_only(d):
        return {
            "substrate": AGGREGATION_SUBSTRATE,
            "min_duration_s": AGGREGATION_MIN_DURATION_S,
            "reason": "aggregation",
        }
    if _is_deterministic_verifier(d):
        return {
            "substrate": _inference_substrate_text(d) or "deterministic_verifier",
            "min_duration_s": DETERMINISTIC_VERIFIER_MIN_DURATION_S,
            "reason": "deterministic_verifier",
        }
    if _is_live_llm_inference(d):
        return {
            "substrate": LIVE_LLM_SUBSTRATE,
            "min_duration_s": COMPUTE_BOUND_MIN_DURATION_S,
            "reason": "live_model",
        }
    if _has_compute_bound_marker(d):
        return {
            "substrate": _inference_substrate_text(d) or "compute_bound_marker",
            "min_duration_s": COMPUTE_BOUND_MIN_DURATION_S,
            "reason": "live_model",
        }
    return None


def check_duration_vs_claim(d: dict[str, Any], flags: list[Flag]) -> None:
    """Compute-bound artifact with implausibly short duration."""
    duration = d.get("duration_s")
    if not _is_finite_number(duration):
        return
    if not _has_compute_bound_marker(d) and not _is_live_llm_inference(d):
        return
    floor = duration_floor_for_artifact(d)
    if floor is None:
        return
    # Verifier-scoring artifacts run in seconds because they score
    # cached candidates -- their GGUF markers are vestigial. Apply
    # the tighter verifier-scoring threshold instead of the full
    # model-inference threshold.
    if floor["reason"] == "verifier_scoring":
        min_duration = float(floor["min_duration_s"])
        if float(duration) < min_duration:
            flags.append(
                Flag(
                    kind="DURATION_TOO_SHORT",
                    severity="critical",
                    detail=(
                        f"duration_s={duration} but artifact declares "
                        f"verifier-scoring substrate. Even verifier "
                        f"scoring of a few hundred candidates takes "
                        f">={min_duration}s; this "
                        f"completed too fast to have scored anything."
                    ),
                )
            )
        return
    # Aggregation-only artifacts (capstones, archive/activate,
    # paper-table builders) just read upstream JSON and arithmetic.
    # Milliseconds are honest. The GGUF markers are inherited from
    # the upstream artifacts they cite, not invoked here.
    if floor["reason"] == "aggregation":
        min_duration = float(floor["min_duration_s"])
        if float(duration) < min_duration:
            flags.append(
                Flag(
                    kind="DURATION_TOO_SHORT",
                    severity="critical",
                    detail=(
                        f"duration_s={duration} but artifact declares "
                        f"aggregation substrate. Even loading upstream "
                        f"JSON takes microseconds; a value below "
                        f"{min_duration}s suggests the "
                        f"duration was not measured at all."
                    ),
                )
            )
        return
    if floor["reason"] == "deterministic_verifier":
        min_duration = float(floor["min_duration_s"])
        if float(duration) < min_duration:
            flags.append(
                Flag(
                    kind="DURATION_TOO_SHORT",
                    severity="critical",
                    detail=(
                        f"duration_s={duration} but artifact declares "
                        f"deterministic-verifier substrate. Even loading "
                        f"checked-in JSON takes microseconds; a value below "
                        f"{min_duration}s suggests "
                        f"the duration was not measured at all."
                    ),
                )
            )
        return
    min_duration = float(floor["min_duration_s"])
    if float(duration) < min_duration:
        flags.append(
            Flag(
                kind="DURATION_TOO_SHORT",
                severity="critical",
                detail=(
                    f"duration_s={duration} but artifact references "
                    f"compute-bound markers (GGUF / CUDA / live model). "
                    f"Loading and running a real model takes "
                    f">={min_duration}s minimum; this "
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
    if not _has_compute_bound_marker(d) and not _is_live_llm_inference(d):
        return
    if offline_arc_methodology_descriptor(d) is not None:
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


_MOAT_HEADLINE_MARKERS = (
    "moat_won",
    "moat_proven",
    "moat proven",
    "moat-proven",
    "efficiency_moat_won",
    "verifier_value_added_true",
    "verifier_efficiency_win",
)


def _claims_moat(d: dict[str, Any]) -> bool:
    """True if the artifact headlines a verifier moat / superiority win."""
    if d.get("verifier_value_added") is True or d.get("verifier_efficiency_win") is True:
        return True
    for key in ("honest_verdict", "headline_outcome", "headline"):
        v = d.get(key)
        if isinstance(v, str) and any(m in v.lower() for m in _MOAT_HEADLINE_MARKERS):
            return True
    return False


def _flips_gate(d: dict[str, Any]) -> bool:
    """True if the artifact asserts a (DiffusionGemma) gate is MET / flipped."""
    v = d.get("diffusiongemma_gate_status")
    if isinstance(v, str) and v.strip().upper() == "MET":
        return True
    g = d.get("diffusiongemma_gate")
    if isinstance(g, dict) and (
        g.get("met") is True or str(g.get("status", "")).strip().upper() == "MET"
    ):
        return True
    hv = d.get("honest_verdict")
    if isinstance(hv, str) and ("gate_met" in hv.lower() or "diffusiongemma_met" in hv.lower()):
        return True
    return False


def check_circular_moat_overclaim(d: dict[str, Any], flags: list[Flag]) -> None:
    """Circularity / oracle-distinctness guard (2026-06-14, operator-directed).

    A verifier "moat" / efficiency win is a NON-CIRCULAR claim only when the
    verifier is INDEPENDENT of the executable oracle that defines correctness.
    Where the verifier IS the oracle (run the unit tests on HumanEval,
    check_sudoku_validity, etc.), beating self-consistency or an LLM-judge is
    true-but-circular: it does NOT show a learned/energy verifier adds value, and
    it must NOT headline a moat or flip a gate. Every moat/gate claim must declare
    ``verifier_is_oracle: bool``.

    Origin: the .387/.388 capstones over-claimed the code/efficiency win as
    "moat proven" and flipped the DiffusionGemma gate to MET on a circular
    (verifier==oracle) result; only the operator caught it, twice. This turns that
    manual catch into a mechanical guard. See CLAUDE.md "Circularity /
    Oracle-Distinctness Discipline".
    """
    if not (_claims_moat(d) or _flips_gate(d)):
        return
    vio = d.get("verifier_is_oracle")
    if _flips_gate(d) and vio is not False:
        # The precise over-claim: headlining a gate-flip while the verifier is
        # the oracle (or undeclared). CRITICAL -> the artifact may not flip the
        # gate; a circular result is not gate-eligible.
        flags.append(
            Flag(
                kind="CIRCULAR_MOAT_OVERCLAIM",
                severity="critical",
                detail=(
                    f"Artifact flips a gate/headlines a moat but verifier_is_oracle={vio!r} "
                    "(not False). A circular or oracle-undeclared verifier cannot flip a gate "
                    "or headline 'moat proven' -- only an oracle-DISTINCT (learned/energy) win "
                    "is headline/gate-eligible. Declare verifier_is_oracle=False and use an "
                    "oracle-distinct verifier, or do not flip the gate."
                ),
            )
        )
    elif vio is None:
        flags.append(
            Flag(
                kind="CIRCULAR_MOAT_OVERCLAIM",
                severity="warn",
                detail=(
                    "Verifier-value/moat claim does not declare verifier_is_oracle. Is the "
                    "verifier the SAME executable oracle that defines correctness (circular, "
                    "NOT headline-eligible) or oracle-distinct (learned/energy, headline-"
                    "eligible)? Declare verifier_is_oracle: bool."
                ),
            )
        )
    elif vio is True:
        flags.append(
            Flag(
                kind="CIRCULAR_MOAT_OVERCLAIM",
                severity="warn",
                detail=(
                    "verifier_is_oracle=True: EXECUTION-GROUNDED CIRCULAR win (the verifier IS "
                    "the oracle). Valid as a result, but NOT headline-eligible as 'moat proven' "
                    "and may not flip a gate -- the non-circular claim (oracle-distinct "
                    "learned/energy verifier) remains the open frontier."
                ),
            )
        )


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
    check_ceiling_saturation(d, flags)
    check_degenerate_separation(d, flags)
    check_degenerate_controls(d, flags)
    check_circular_moat_overclaim(d, flags)

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
    if _is_live_llm_inference(d):
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
