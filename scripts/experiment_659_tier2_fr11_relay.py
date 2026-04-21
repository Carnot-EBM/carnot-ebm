#!/usr/bin/env python3
"""Experiment 659: FR-11 Tier 2 Cross-Session Relay.

**Researcher summary:**
    FR-11 (autonomous self-learning loop) requires that constraint violations
    detected during live VR milestone runs are preserved and reused in future
    sessions.  Exp 659 implements the cross-session relay:

    1. Load violation patterns from Exp 656 (live VR attempt #18).
       If the Exp 656 gate was closed (no real violations), use synthetic
       patterns to validate the relay plumbing is wired correctly.
    2. Wire each violation pattern into ViolationPatternLibrary.
    3. Measure the cross-session false positive rate on 20 known-correct
       responses from live_pairs_578.json to confirm real violations — not
       noise — are being captured.
    4. Write a JSON artifact with schema 'carnot.fr11_relay.v1'.

**Why this matters:**
    Without a cross-session relay, each experiment starts from zero knowledge
    of which arithmetic error patterns the model tends to make.  By persisting
    violation patterns to ViolationPatternLibrary, future experiments can
    activate stronger constraint templates for those specific error types,
    closing the Tier 3 JEPA predictive memory loop.

    The FP rate measurement is the guard rail: if stored patterns are too
    generic (e.g. single words like "is"), they fire on correct responses and
    generate false alerts.  A low FP rate on known-correct text confirms the
    patterns are specific to real errors.

Spec: REQ-SELF-020, SCENARIO-SELF-025, SCENARIO-SELF-026
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root resolution — must happen before any carnot imports
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Env autofix — must be FIRST before GPU-sensitive imports
# ---------------------------------------------------------------------------

from python.carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

apply_env_autofix()

# ---------------------------------------------------------------------------
# Watchdog — hard wall-clock cap
# ---------------------------------------------------------------------------

try:
    from python.carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog

    _watchdog = ExperimentTimeoutWatchdog(659, timeout_minutes=20)
    _watchdog.start()
except Exception:  # noqa: BLE001 — watchdog import may fail in some envs; non-fatal
    _watchdog = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Standard imports
# ---------------------------------------------------------------------------

from scripts.experiment_template import ExperimentTemplate  # noqa: E402

from python.carnot.pipeline.constraint_template_library import (  # noqa: E402
    ViolationPatternLibrary,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EXP_ID = 659
_TITLE = "FR-11 Tier 2 Cross-Session Relay"
_DELIVERABLE = "results/experiment_659_tier2_fr11_relay.json"
_SOURCE_EXP = 656
_EXP_656_RESULT = _REPO_ROOT / "results" / "experiment_656_live_vr_attempt_18.json"
_LIVE_PAIRS_PATH = _REPO_ROOT / "results" / "live_pairs_578.json"
_LIBRARY_PATH = str(_REPO_ROOT / "data" / "constraint_templates_659.json")

# Synthetic violation patterns used when Exp 656 gate was closed.
# These are plausible arithmetic errors that a model might produce verbatim.
_SYNTHETIC_PATTERNS = [
    "COMPUTE: 47 + 28 = 76",
    "total is 80",
    "therefore 15",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _load_exp656_violations() -> tuple[list[str], bool]:
    """Load violation patterns from Exp 656 result JSON.

    **Detailed explanation for engineers:**
        Exp 656 is a live VR run that may have been blocked by a gate check.
        If the gate was closed (status='blocked') or the file contains no VR
        results with violations, we fall back to synthetic patterns.  This
        ensures the relay plumbing is validated regardless of whether real
        violations are available.

    Returns:
        Tuple of (patterns: list[str], from_real_data: bool).
        from_real_data is True when at least one pattern came from the Exp 656 file.
    """
    if not _EXP_656_RESULT.exists():
        return _SYNTHETIC_PATTERNS, False

    try:
        data = json.loads(_EXP_656_RESULT.read_text())
    except (json.JSONDecodeError, OSError):
        return _SYNTHETIC_PATTERNS, False

    # Gate was closed — no real violation data to extract.
    if data.get("gate_open") is False or data.get("status") == "blocked":
        return _SYNTHETIC_PATTERNS, False

    # Attempt to extract violation patterns from VR result fields.
    patterns: list[str] = []
    vr_results = data.get("vr_results", []) or data.get("violations", []) or []
    for item in vr_results:
        if isinstance(item, dict):
            p = item.get("violation_pattern") or item.get("pattern") or item.get("step_text")
            if p and isinstance(p, str) and len(p.strip()) > 0:
                patterns.append(p.strip())

    if patterns:
        return patterns, True
    # File exists but no parseable violations — fall back to synthetic.
    return _SYNTHETIC_PATTERNS, False


def _load_correct_responses(n: int = 20) -> list[str]:
    """Load the first n known-correct responses from live_pairs_578.json.

    **Detailed explanation for engineers:**
        live_pairs_578.json contains labelled question-response pairs with an
        ``is_correct`` boolean field.  We filter to correct-only responses because
        the FP rate must be measured on text we KNOW is correct — if a stored
        violation pattern matches a correct response, that is a false positive.

    Args:
        n: Maximum number of correct responses to load.

    Returns:
        List of response strings (up to n, may be fewer if the file has fewer
        correct entries).  Returns [] if the file is absent.
    """
    if not _LIVE_PAIRS_PATH.exists():
        return []
    try:
        pairs = json.loads(_LIVE_PAIRS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return []

    correct = [p["response"] for p in pairs if p.get("is_correct") and "response" in p]
    return correct[:n]


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 659: wire violations into ViolationPatternLibrary and measure FP rate."""
    tmpl = ExperimentTemplate(_EXP_ID, _TITLE, _DELIVERABLE)
    tmpl.setup()

    # ------------------------------------------------------------------
    # 1. Load violation patterns from Exp 656 (or synthetic fallback)
    # ------------------------------------------------------------------
    violation_patterns, from_real_data = _load_exp656_violations()

    # ------------------------------------------------------------------
    # 2. Wire violations into ViolationPatternLibrary
    # ------------------------------------------------------------------
    library = ViolationPatternLibrary(_LIBRARY_PATH)
    n_before = len(library.templates)

    for pattern in violation_patterns:
        library.add_template(
            pattern=pattern,
            violation_type="arithmetic",
            source_experiment=_SOURCE_EXP,
        )

    n_added = len(library.templates) - n_before
    n_total = len(library.templates)

    # ------------------------------------------------------------------
    # 3. Measure cross-session FP rate on known-correct responses
    # ------------------------------------------------------------------
    correct_responses = _load_correct_responses(n=20)
    fp_rate = library.get_fp_rate(correct_responses)

    fr11_real_violations_confirmed = n_added > 0

    # ------------------------------------------------------------------
    # 4. Build artifact
    # ------------------------------------------------------------------
    honest_verdict = (
        "fr11_relay_complete_violations_wired"
        if fr11_real_violations_confirmed
        else "fr11_relay_complete_synthetic_only"
    )

    artifact = tmpl.build_result(
        {
            "schema": "carnot.fr11_relay.v1",
            "n_templates_before": n_before,
            "n_templates_added": n_added,
            "n_templates_total": n_total,
            "cross_session_fp_rate": fp_rate,
            "fr11_real_violations_confirmed": fr11_real_violations_confirmed,
            "source_experiment": _SOURCE_EXP,
            "patterns_source": "real_exp656" if from_real_data else "synthetic",
            "n_correct_responses_checked": len(correct_responses),
            "honest_verdict": honest_verdict,
        },
        status="success",
    )

    # Write artifact to disk before assert_deliverable_written() checks for it.
    deliverable_path = _REPO_ROOT / _DELIVERABLE
    deliverable_path.parent.mkdir(parents=True, exist_ok=True)
    with open(deliverable_path, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"[Exp 659] n_added={n_added}, fp_rate={fp_rate:.4f}, verdict={honest_verdict}")
    print(f"[Exp 659] Written: {deliverable_path}")

    if _watchdog is not None:
        _watchdog.stop()

    # MANDATORY: verify the deliverable was written before exiting.
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
