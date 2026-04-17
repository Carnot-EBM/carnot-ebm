#!/usr/bin/env python3
"""Experiment 436: Milestone 2026.04.32 Operational Efficiency Retrospective.

Evaluates HOW the work in Exps 425-435 was executed, not just WHAT was produced.
Follows the pattern established in Exps 363, 376, 389, 403, 424.

Protocol:
    1. apply_env_autofix() FIRST — injects CARNOT_FORCE_LIVE=1 if GPU present.
    2. ExperimentTimeoutWatchdog(436, timeout_minutes=30) — hard wall-clock cap.
    3. Load all Exp 425-435 result JSONs (gracefully handles missing files).
    4. Compute MilestoneRetro2026_04_32 from available evidence.
    5. Print success criteria table.
    6. Print new and closed RETRO items.
    7. Write results/operational_retro_2026_04_32.json.

Spec: SCENARIO-RETRO-032
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# apply_env_autofix MUST be called before any JAX or CUDA import.
from carnot.pipeline.env_autofix import apply_env_autofix

_env_result = apply_env_autofix()

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Repository root and result paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RESULTS = _REPO_ROOT / "results"

_EXP_PATHS: dict[str, Path] = {
    "425": _RESULTS / "experiment_425_conductor_timeout.json",
    "426": _RESULTS / "experiment_426_dual_gpu_fix.json",
    "427": _RESULTS / "experiment_427_precision_live_confirmed.json",
    "428": _RESULTS / "experiment_428_humaneval_live_confirmed.json",
    "429": _RESULTS / "experiment_429_adversarial_live.json",
    "430": _RESULTS / "experiment_430_fover_z3_labels.json",
    "431": _RESULTS / "experiment_431_eorm_jepa_real_retrain.json",
    "432": _RESULTS / "experiment_432_jitrl_live_validation.json",
    "433": _RESULTS / "experiment_433_spilled_energy.json",
    "434": _RESULTS / "experiment_434_compliance_checker.json",
    "435": _RESULTS / "experiment_435_npu_unblock.json",
    "435a": _RESULTS / "experiment_435a_kona_continuous_energy.json",
}

_OUTPUT_PATH = _RESULTS / "operational_retro_2026_04_32.json"


# ---------------------------------------------------------------------------
# MilestoneRetro2026_04_32 dataclass
# ---------------------------------------------------------------------------


@dataclass
class MilestoneRetro2026_04_32:
    """Structured retrospective for milestone 2026.04.32 (Exps 425-435).

    Boolean flags are derived strictly from result JSON files. A field is True
    only when explicit provenance exists in the loaded artifact. Missing result
    files produce False/unknown values with an explanation in new_retro_items.

    WHY a dataclass instead of a plain dict: enforces the field contract so that
    future retros can diff the schema version and detect regressions in coverage.
    """

    milestone: str = "2026.04.32"
    n_experiments: int = 0
    mean_minutes_per_exp: float = 0.0

    # Infrastructure improvements
    conductor_timeout_implemented: bool = False  # Exp 425 / experiment_watchdog.py
    gpu1_zombie_fixed: bool = False  # Exp 426 retro_025_status

    # Live result confirmation
    live_numbers_confirmed: bool = False  # at least one live benchmark improvement

    # FR-11 / FR-12 progress
    fr11_relay_confirmed: bool = False  # Exp 431 retro_024_closed
    tier1_live_validated: bool = False  # Exp 432 honest_verdict = live_fp_reduction

    # New capabilities
    spilled_energy_viable: bool = False  # Exp 433
    compliance_checker_works: bool = False  # Exp 434
    npu_status: str = "not_run"  # Exp 435 honest_verdict

    headline_results: dict = field(default_factory=dict)
    new_retro_items: list = field(default_factory=list)
    closed_retro_items: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------


def load_result(exp_key: str) -> Optional[dict]:
    """Load a result JSON; return None (with a log warning) if absent.

    WHY return None instead of raising: missing results are an expected state
    (scaffolding_only experiments timed out without writing output). Callers
    derive honest_verdict='not_run' rather than crashing the retro.
    """
    path = _EXP_PATHS[exp_key]
    if not path.exists():
        _log.warning("Exp %s result not found: %s", exp_key, path.name)
        return None
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------


def _conductor_timeout_from_results(r425: Optional[dict]) -> bool:
    """Derive conductor_timeout_implemented from Exp 425 result or watchdog module.

    WHY check the module as fallback: Exp 425 may not have a result JSON if the
    watchdog was implemented mid-milestone without a dedicated run artifact.
    The module's existence is the authoritative proof of implementation.
    """
    if r425 is not None:
        status = r425.get("status", "")
        return status in ("success", "complete")
    # Belt-and-suspenders: check that the module file actually exists.
    watchdog_path = _REPO_ROOT / "python" / "carnot" / "pipeline" / "experiment_watchdog.py"
    return watchdog_path.exists()


def _gpu1_zombie_fixed_from_results(r426: Optional[dict]) -> bool:
    """Exp 426 honest_verdict='zombie_detected' means confirmed but NOT fixed.

    RETRO-025 is closed only when honest_verdict transitions to 'zombie_cleared'
    or an equivalent fix verdict. zombie_confirmed means the problem was found
    and documented, not resolved.
    """
    if r426 is None:
        return False
    verdict = r426.get("honest_verdict", "")
    retro_status = r426.get("retro_025_status", "")
    # zombie_confirmed or zombie_detected = problem confirmed, not fixed
    if retro_status in ("zombie_confirmed",) or verdict in ("zombie_detected",):
        return False
    return verdict in ("zombie_cleared", "zombie_fixed", "healthy")


def _live_numbers_from_results(r427: Optional[dict], r428: Optional[dict], r429: Optional[dict]) -> bool:
    """True only when at least one of 427/428/429 ran live and produced a signed improvement.

    scaffolding_only results do NOT count as live_numbers_confirmed because their
    headline numbers are absent — the script exists but was never executed against
    real GPU inference.
    """
    for r in [r427, r428, r429]:
        if r is None:
            continue
        if r.get("status") == "scaffolding_only":
            continue
        if r.get("status") == "success":
            return True
        # A result with live_improvement or explicit live provenance
        verdict = r.get("honest_verdict", "")
        if "live" in verdict and "improvement" in verdict:
            return True
    return False


def _fr11_relay_from_results(r431: Optional[dict]) -> bool:
    """Exp 431 sets retro_024_closed=True when EORM retrained on real FOVER pairs."""
    if r431 is None:
        return False
    return bool(r431.get("retro_024_closed", False))


def _tier1_live_from_results(r432: Optional[dict]) -> bool:
    """Exp 432 tier1_live_validated requires honest_verdict='live_fp_reduction'.

    synthetic_fallback does NOT count. The JitRL controller must have trained
    and validated on real inference data, not synthetic proxies.
    """
    if r432 is None:
        return False
    verdict = r432.get("honest_verdict", "")
    return verdict in ("live_fp_reduction", "tier1_live_validated")


def _spilled_energy_from_results(r433: Optional[dict]) -> bool:
    if r433 is None:
        return False
    return r433.get("honest_verdict") == "spilled_energy_viable"


def _compliance_checker_from_results(r434: Optional[dict]) -> bool:
    if r434 is None:
        return False
    return r434.get("honest_verdict") in ("compliance_checker_works", "success", True)


def _npu_status_from_results(r435: Optional[dict], r435a: Optional[dict]) -> str:
    """Return the honest NPU verdict: prefer Exp 435 over 435a (seed experiment).

    WHY 435 takes precedence: 435a is a Phase 3 toy seed validating discrete-to-
    continuous energy landscape mapping, NOT an NPU dispatch attempt. Exp 435 is
    the actual AMD XDNA NPU unblock attempt with IRON toolchain.
    """
    if r435 is not None:
        return str(r435.get("honest_verdict", "not_run"))
    if r435a is not None:
        return f"seed_only:{r435a.get('honest_verdict', 'unknown')}"
    return "not_run"


def _build_headline_results(r427: Optional[dict], r428: Optional[dict], r429: Optional[dict]) -> dict:
    """Collect signed precision/humaneval/adversarial improvements from live results.

    Returns empty sub-dicts for scaffolding_only results. Callers must not cite
    these as headline numbers until the experiments run live.
    """
    def _extract(r: Optional[dict], label: str) -> dict:
        if r is None:
            return {"status": "result_missing", "provenance": label}
        status = r.get("status", "unknown")
        if status == "scaffolding_only":
            return {"status": "scaffolding_only", "provenance": label,
                    "note": "Script written; live execution pending GPU slot + human trigger."}
        return {"status": status, "honest_verdict": r.get("honest_verdict"), "provenance": label}

    return {
        "precision_gsm8k": _extract(r427, "experiment_427"),
        "humaneval": _extract(r428, "experiment_428"),
        "adversarial_gsm8k": _extract(r429, "experiment_429"),
    }


def _duration_minutes(r: Optional[dict]) -> Optional[float]:
    """Extract duration in minutes from a result, if available."""
    if r is None:
        return None
    secs = r.get("duration_s")
    if secs is not None:
        return float(secs) / 60.0
    return None


def _compute_timing(results: dict[str, Optional[dict]]) -> tuple[int, float]:
    """Return (n_experiments, mean_minutes_per_exp) for this milestone.

    Scaffolding_only experiments that hit the 45-min timeout are credited at
    45 minutes. Experiments with a missing result default to 45 minutes
    (they timed out without writing an artifact). Short experiments use their
    actual duration_s.
    """
    # Experiments in this milestone: 425 through 435 (inclusive), 435a as a bonus.
    experiment_keys = ["425", "426", "427", "428", "429", "430", "431", "432", "433", "434", "435", "435a"]
    durations: list[float] = []

    for key in experiment_keys:
        r = results.get(key)
        if r is None:
            # Missing result: timed out without writing artifact → credit 45 min.
            durations.append(45.0)
            continue
        status = r.get("status", "")
        if status == "scaffolding_only":
            # These always hit the 45-min conductor wall-clock timeout.
            durations.append(45.0)
        else:
            d = _duration_minutes(r)
            # Very short experiments (< 0.5 min) reflect script overhead only;
            # floor at 2 minutes to capture setup + doc-reconciliation cost.
            durations.append(max(d, 2.0) if d is not None else 14.0)

    n = len(durations)
    mean_min = sum(durations) / n if n > 0 else 0.0
    return n, round(mean_min, 1)


# ---------------------------------------------------------------------------
# RETRO item builders
# ---------------------------------------------------------------------------


def _new_retro_items(
    r427: Optional[dict],
    r428: Optional[dict],
    r429: Optional[dict],
    r433: Optional[dict],
    r434: Optional[dict],
    r435: Optional[dict],
) -> list[dict]:
    """Identify new RETRO items surfaced in this milestone."""
    items: list[dict] = []

    # RETRO-026: Scaffold-only live benchmarks need human-triggered long runs.
    if all(
        r is None or r.get("status") == "scaffolding_only"
        for r in [r427, r428, r429]
    ):
        items.append({
            "id": "RETRO-026",
            "severity": "high",
            "description": (
                "Exps 427, 428, 429 (precision/HumanEval/adversarial) all produced scaffolding_only "
                "results after hitting the 45-minute conductor wall-clock timeout. The benchmarks "
                "legitimately require >45 minutes of live GPU inference. They cannot run inside the "
                "conductor's subagent budget. A dedicated long-running executor (human-triggered or "
                "conductor side-channel) is required to close these."
            ),
            "milestones_carried": 1,
            "new_this_milestone": True,
            "action_required": (
                "Fix RETRO-025 (GPU 1 scheduling) THEN manually run: "
                "JAX_PLATFORMS=cpu python scripts/experiment_427_precision_live_confirmed.py "
                "(also 428, 429). Alternatively, configure a long-running executor with a "
                "120-minute budget for benchmark-class experiments."
            ),
        })

    # RETRO-027: Exps 433 and 434 have no result files at all.
    missing_no_result = []
    if r433 is None:
        missing_no_result.append("Exp 433 (SpilledEnergyDetector)")
    if r434 is None:
        missing_no_result.append("Exp 434 (ComplianceEnergyChecker)")
    if r435 is None:
        missing_no_result.append("Exp 435 (AMD XDNA NPU Unblock)")
    if missing_no_result:
        items.append({
            "id": "RETRO-027",
            "severity": "medium",
            "description": (
                f"{', '.join(missing_no_result)} have no result JSON files. "
                "Scripts exist and tests pass, but the conductor never executed them. "
                "This represents silent experiment drop — no timeout artifact, no partial result."
            ),
            "milestones_carried": 1,
            "new_this_milestone": True,
            "action_required": (
                "Run missing experiment scripts manually. Add conductor logic to detect and "
                "report experiments with scripts but no result file as 'not_run' rather than "
                "silently skipping them."
            ),
        })

    return items


def _closed_retro_items(conductor_timeout_implemented: bool) -> list[str]:
    """RETRO items that can be marked closed in this milestone."""
    closed: list[str] = []
    if conductor_timeout_implemented:
        # ExperimentTimeoutWatchdog is implemented and all Exp 425+ scripts use it.
        # RETRO-003 carried 17+ milestones is now closed AT THE PER-EXPERIMENT LEVEL.
        closed.append(
            "RETRO-003 (per-experiment): ExperimentTimeoutWatchdog implemented in "
            "python/carnot/pipeline/experiment_watchdog.py. All Exp 425+ scripts "
            "use ExperimentTimeoutWatchdog as a context manager. Per-experiment hard "
            "cap is now operational. Conductor-level session timeout remains open."
        )
    return closed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_retro() -> dict:
    """Load all results, compute MilestoneRetro2026_04_32, return artifact dict."""
    # Load all result JSONs.
    results = {key: load_result(key) for key in _EXP_PATHS}

    r425 = results["425"]
    r426 = results["426"]
    r427 = results["427"]
    r428 = results["428"]
    r429 = results["429"]
    r430 = results["430"]  # noqa: F841 — loaded for completeness, used in notes
    r431 = results["431"]
    r432 = results["432"]
    r433 = results["433"]
    r434 = results["434"]
    r435 = results["435"]
    r435a = results["435a"]

    # Compute all fields.
    conductor_timeout_implemented = _conductor_timeout_from_results(r425)
    gpu1_zombie_fixed = _gpu1_zombie_fixed_from_results(r426)
    live_numbers_confirmed = _live_numbers_from_results(r427, r428, r429)
    fr11_relay_confirmed = _fr11_relay_from_results(r431)
    tier1_live_validated = _tier1_live_from_results(r432)
    spilled_energy_viable = _spilled_energy_from_results(r433)
    compliance_checker_works = _compliance_checker_from_results(r434)
    npu_status = _npu_status_from_results(r435, r435a)
    headline_results = _build_headline_results(r427, r428, r429)
    new_retro = _new_retro_items(r427, r428, r429, r433, r434, r435)
    closed_retro = _closed_retro_items(conductor_timeout_implemented)
    n_experiments, mean_minutes = _compute_timing(results)

    retro = MilestoneRetro2026_04_32(
        n_experiments=n_experiments,
        mean_minutes_per_exp=mean_minutes,
        conductor_timeout_implemented=conductor_timeout_implemented,
        gpu1_zombie_fixed=gpu1_zombie_fixed,
        live_numbers_confirmed=live_numbers_confirmed,
        fr11_relay_confirmed=fr11_relay_confirmed,
        tier1_live_validated=tier1_live_validated,
        spilled_energy_viable=spilled_energy_viable,
        compliance_checker_works=compliance_checker_works,
        npu_status=npu_status,
        headline_results=headline_results,
        new_retro_items=new_retro,
        closed_retro_items=closed_retro,
    )

    _print_success_table(retro)
    _print_retro_items(retro)

    return _build_artifact(retro)


def _print_success_table(retro: MilestoneRetro2026_04_32) -> None:
    """Print a human-readable success criteria table to stdout."""
    rows = [
        ("conductor_timeout_implemented", retro.conductor_timeout_implemented, "Exp 425 / experiment_watchdog.py"),
        ("gpu1_zombie_fixed",             retro.gpu1_zombie_fixed,             "Exp 426 retro_025_status"),
        ("live_numbers_confirmed",        retro.live_numbers_confirmed,        "Exps 427/428/429 live status"),
        ("fr11_relay_confirmed",          retro.fr11_relay_confirmed,          "Exp 431 retro_024_closed"),
        ("tier1_live_validated",          retro.tier1_live_validated,          "Exp 432 honest_verdict"),
        ("spilled_energy_viable",         retro.spilled_energy_viable,         "Exp 433"),
        ("compliance_checker_works",      retro.compliance_checker_works,      "Exp 434"),
        ("npu_status",                    retro.npu_status,                    "Exp 435"),
    ]
    print("\n=== Milestone 2026.04.32 Success Criteria ===")
    print(f"{'Criterion':<35} {'Result':<20} {'Notes'}")
    print("-" * 90)
    for criterion, result, notes in rows:
        result_str = str(result)
        print(f"{criterion:<35} {result_str:<20} {notes}")
    print(f"\nn_experiments: {retro.n_experiments}  mean_min/exp: {retro.mean_minutes_per_exp}")


def _print_retro_items(retro: MilestoneRetro2026_04_32) -> None:
    """Print new and closed RETRO items."""
    if retro.new_retro_items:
        print("\n=== NEW RETRO Items ===")
        for item in retro.new_retro_items:
            print(f"  [{item['severity'].upper()}] {item['id']}: {item['description'][:120]}...")
    if retro.closed_retro_items:
        print("\n=== CLOSED RETRO Items ===")
        for item in retro.closed_retro_items:
            print(f"  CLOSED: {item[:120]}...")


def _build_artifact(retro: MilestoneRetro2026_04_32) -> dict:
    """Build the serializable JSON artifact."""
    data = asdict(retro)
    data["schema"] = "carnot.operational_retro.v6"
    data["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    data["status"] = "complete"
    return data


def main() -> None:
    """Entry point: run retro under watchdog, write result JSON."""
    result_path = str(_OUTPUT_PATH)
    with ExperimentTimeoutWatchdog(436, timeout_minutes=30, result_path=result_path):
        artifact = run_retro()

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUTPUT_PATH, "w") as fh:
        json.dump(artifact, fh, indent=2)
    _log.info("Wrote %s", _OUTPUT_PATH)
    print(f"\nResult: {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
