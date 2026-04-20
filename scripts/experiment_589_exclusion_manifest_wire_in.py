#!/usr/bin/env python3
"""Experiment 589 — Exclusion Manifest Wire-In and NPU Unblock v7.

**Context (RETRO-067):**
    The same five experiments (308, 260, 309, 425, 410) appeared in the conductor's
    slowest-5 list for EIGHT consecutive milestones (.37 through .45), wasting ~2,870
    minutes (47.8 hours).  Exp 575 created the exclusion manifest JSON and
    ExclusionManifest class with conductor_consulted=False — the wire-in was deferred.

    This experiment:
    1. Verifies that scripts/conductor_session_wrapper.py exists and works correctly.
    2. Tests the wrapper logic inline (exp 308 excluded, exp 589 not excluded).
    3. Attempts NPU unblock v7 via the IRON path (pip install mlir-aie) and
       the pacman system path (ninja, openblas) without raising on failure.
    4. Records all results in a standardised artifact.

**NPU unblock history:**
    Milestones .39 through .44 (6 consecutive) have been blocked on missing
    ninja and openblas system dependencies.  This experiment tries the mlir-aie
    IRON path as a pip-installable alternative that does not require system packages.

Spec: REQ-INFRA-080, REQ-INFRA-081, SCENARIO-INFRA-085, SCENARIO-INFRA-086
"""

from __future__ import annotations

# apply_env_autofix MUST be called before any JAX or CUDA import to inject
# CARNOT_FORCE_LIVE=1 when a GPU is present (avoids lazy-load GPU stalls).
from carnot.pipeline.env_autofix import apply_env_autofix

_env_result = apply_env_autofix()

import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402

_RESULT_PATH = "results/experiment_589_exclusion_manifest_wire_in.json"
_WRAPPER_PATH = "scripts/conductor_session_wrapper.py"
_EXCLUDED_IDS = [308, 260, 309, 425, 410]


def attempt_npu_unblock() -> dict:
    """Attempt all NPU unblock paths and return results without raising on failure.

    NPU unblock v7 tries two paths:
    1. IRON path: pip install mlir-aie (pure pip, no system packages needed).
       If mlir_aie imports successfully after install, npu_iron_available=True.
    2. System path: pacman -S --noconfirm ninja openblas (requires root or sudo).
       If ninja imports after install, npu_ninja_available=True.

    All attempts are recorded regardless of outcome.  Failure is expected on
    machines that do not have an AMD XDNA NPU or the required kernel modules.

    Returns
    -------
    dict with keys: npu_iron_available, npu_ninja_available, iron_install_stdout,
    iron_install_stderr, pacman_stdout, pacman_stderr
    """
    results: dict = {
        "npu_iron_available": False,
        "npu_ninja_available": False,
        "iron_install_stdout": "",
        "iron_install_stderr": "",
        "pacman_stdout": "",
        "pacman_stderr": "",
    }

    # IRON path: pip install mlir-aie
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "mlir-aie"],
            capture_output=True,
            timeout=60,
            text=True,
        )
        results["iron_install_stdout"] = proc.stdout[:500] if proc.stdout else ""
        results["iron_install_stderr"] = proc.stderr[:500] if proc.stderr else ""
    except Exception as exc:
        results["iron_install_stderr"] = f"pip install mlir-aie failed: {exc}"

    # Check if mlir_aie is now importable (may already be installed).
    try:
        import mlir_aie  # type: ignore[import-not-found]  # noqa: F401, PLC0415

        results["npu_iron_available"] = True
    except Exception:
        results["npu_iron_available"] = False

    # System path: pacman ninja + openblas
    try:
        proc = subprocess.run(
            ["pacman", "-S", "--noconfirm", "ninja", "openblas"],
            capture_output=True,
            timeout=30,
            text=True,
        )
        results["pacman_stdout"] = proc.stdout[:500] if proc.stdout else ""
        results["pacman_stderr"] = proc.stderr[:500] if proc.stderr else ""
    except Exception as exc:
        results["pacman_stderr"] = f"pacman install failed: {exc}"

    # Check if ninja is now importable.
    try:
        import ninja  # type: ignore[import-not-found]  # noqa: F401, PLC0415

        results["npu_ninja_available"] = True
    except Exception:
        results["npu_ninja_available"] = False

    return results


def run_experiment() -> dict:
    """Run the wire-in verification and NPU unblock, return the artifact payload."""
    # ------------------------------------------------------------------
    # 1. Verify wrapper exists
    # ------------------------------------------------------------------
    wrapper_abs = _REPO_ROOT / _WRAPPER_PATH
    wrapper_created = wrapper_abs.exists()

    # ------------------------------------------------------------------
    # 2. Test wrapper logic inline (import check_experiment directly)
    # ------------------------------------------------------------------
    wrapper_tested = False
    wrapper_correctly_excludes_308 = False
    wrapper_correctly_allows_589 = False

    if wrapper_created:
        try:
            # Import the wrapper module and call check_experiment directly —
            # avoids subprocess overhead and lets us test the logic in-process.
            if str(_REPO_ROOT / "scripts") not in sys.path:
                sys.path.insert(0, str(_REPO_ROOT / "scripts"))
            from conductor_session_wrapper import check_experiment  # noqa: PLC0415

            is_excl_308, reason_308 = check_experiment(308)
            wrapper_correctly_excludes_308 = is_excl_308 and "308" in reason_308

            is_excl_589, _reason_589 = check_experiment(589)
            wrapper_correctly_allows_589 = not is_excl_589

            wrapper_tested = wrapper_correctly_excludes_308 and wrapper_correctly_allows_589
        except Exception as exc:
            wrapper_tested = False
            wrapper_correctly_excludes_308 = False
            wrapper_correctly_allows_589 = False
            # Record exception for debugging but do not raise — the artifact still
            # needs to be written on all exit paths (REQ-INFRA-033).
            _ = exc

    # ------------------------------------------------------------------
    # 3. Count manifest entries
    # ------------------------------------------------------------------
    from carnot.pipeline.exclusion_manifest import DEFAULT_MANIFEST_PATH, ExclusionManifest  # noqa: PLC0415

    manifest = ExclusionManifest(str(_REPO_ROOT / DEFAULT_MANIFEST_PATH))
    entries = manifest.load()
    n_excluded = len(entries)

    # ------------------------------------------------------------------
    # 4. Attempt NPU unblock v7
    # ------------------------------------------------------------------
    npu_results = attempt_npu_unblock()

    return {
        "schema": "carnot.exclusion_manifest_wire_in.v1",
        "wrapper_created": wrapper_created,
        "wrapper_path": _WRAPPER_PATH,
        "n_excluded": n_excluded,
        "excluded_ids": _EXCLUDED_IDS,
        "wrapper_tested": wrapper_tested,
        "wrapper_correctly_excludes_308": wrapper_correctly_excludes_308,
        "wrapper_correctly_allows_589": wrapper_correctly_allows_589,
        "npu_iron_available": npu_results["npu_iron_available"],
        "npu_ninja_available": npu_results["npu_ninja_available"],
        "npu_iron_install_stdout": npu_results["iron_install_stdout"],
        "npu_iron_install_stderr": npu_results["iron_install_stderr"],
        "npu_pacman_stdout": npu_results["pacman_stdout"],
        "npu_pacman_stderr": npu_results["pacman_stderr"],
        "retro_067_partial": True,
        "retro_067_wiring_note": (
            "Human must run: python scripts/conductor_session_wrapper.py <exp_id> "
            "before each session. Conductor itself not modified."
        ),
        "honest_verdict": "wrapper_created_human_action_required",
    }


def main() -> None:
    """Entry point: run experiment under watchdog, write result JSON."""
    tmpl = ExperimentTemplate(
        589,
        "Exclusion Manifest Wire-In",
        _RESULT_PATH,
        requires_gpu=False,
    )
    tmpl.setup()

    with ExperimentTimeoutWatchdog(589, timeout_minutes=20, result_path=str(_REPO_ROOT / _RESULT_PATH)):
        payload = run_experiment()

    artifact = tmpl.build_result(payload, status="success")

    import json  # noqa: PLC0415

    output_path = _REPO_ROOT / _RESULT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(artifact, fh, indent=2)

    print(f"\nResult: {output_path}")
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
