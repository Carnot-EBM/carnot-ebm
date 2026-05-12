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


def check_duration_vs_claim(d: dict[str, Any], flags: list[Flag]) -> None:
    """Compute-bound artifact with implausibly short duration."""
    duration = d.get("duration_s")
    if not _is_finite_number(duration):
        return
    if not _has_compute_bound_marker(d):
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
    has_model_spec = (
        d.get("model_specs") or d.get("target_model") or d.get("models_tested")
    )
    has_seed = d.get("random_seed") is not None or d.get("seed") is not None
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
    args = parser.parse_args(argv[1:])

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
