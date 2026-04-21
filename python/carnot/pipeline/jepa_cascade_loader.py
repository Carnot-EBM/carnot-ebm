"""JEPA v14 Platt-calibrated cascade loader — dynamic dependency resolution for Exp 670.

**Why this module exists (researcher summary):**
    Exp 657 (JEPA v14 cascade deploy) was blocked because it hardcoded the path to the
    Exp 646 result JSON.  Exp 646 succeeded (honest_verdict='platt_calibrated') and wrote
    its result, but to a different filename than Exp 657 expected.  A glob-based search
    removes the hardcoded dependency and makes the loader resilient to filename variations
    across conductor runs.

**What Platt scaling does (layman explanation):**
    A raw energy score from the EORM/JEPA model can be poorly calibrated — the numbers
    may be in the right order (correct < incorrect) but the scale may be off, making
    confidence estimates unreliable.  Platt scaling fits a single temperature parameter T
    such that dividing the raw energy by T produces a well-calibrated probability via the
    sigmoid function: p = sigmoid(E / T).  Exp 646 found T=0.38 reduces ECE from 19.1%
    to 2.3% (87.9% reduction), so using this T in Tier 2 makes the pipeline's skip
    decisions much better calibrated without retraining the underlying model.

**How the loader is used:**
    loader = load_platt_jepa(project_root)
    if loader.loaded:
        pipeline = ThreeTierPipeline(..., platt_temperature=loader.platt_temperature)

Spec: REQ-VERIFY-150, SCENARIO-VERIFY-198, SCENARIO-VERIFY-199
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass


@dataclass
class PlattCalibratedJEPA:
    """Result of attempting to load JEPA v14 Platt calibration from Exp 646.

    **Detailed explanation for engineers:**
        This dataclass is the return value of load_platt_jepa().  Check `loaded`
        before using `platt_temperature` — if loaded=False, the temperature is 1.0
        (identity, no scaling).  model_path is preserved for future use when a saved
        JEPA v14 checkpoint is available, but is not required for Platt scaling alone.

    Fields
    ------
    platt_temperature : float
        The optimal Platt temperature T from Exp 646 (T_optimal field).
        When loaded=False, this is 1.0 (no scaling applied).
    model_path : str | None
        Path to a saved JEPA v14 model checkpoint, if one was recorded in the
        Exp 646 result.  None when no checkpoint path was found.
    loaded : bool
        True when the Exp 646 result was found and T_optimal was extracted
        successfully.  False means the pipeline should fall back to the uncalibrated
        EORM baseline.

    Spec: REQ-VERIFY-150
    """

    platt_temperature: float
    model_path: str | None
    loaded: bool


def find_exp646_result(project_root: str) -> dict | None:
    """Search results/ for the Exp 646 Platt scaling result JSON.

    **Why glob instead of a hardcoded path?**
        Exp 657 failed because the exact filename ("experiment_646_jepa_v14_platt.json")
        was hardcoded.  The conductor may generate different filenames across runs.
        Globbing for "experiment_646_*.json" matches any Exp 646 result regardless of
        the suffix chosen at run time.

    Parameters
    ----------
    project_root : str
        Absolute path to the repository root (e.g. "/home/user/carnot").

    Returns
    -------
    dict | None
        Parsed JSON content of the first matching file, or None if no file is found
        or if the file cannot be parsed as JSON.

    Spec: REQ-VERIFY-150-1, SCENARIO-VERIFY-198
    """
    pattern = os.path.join(project_root, "results", "experiment_646_*.json")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    try:
        with open(matches[0], "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def extract_platt_temperature(exp646_result: dict) -> float | None:
    """Extract the Platt temperature T_optimal from an Exp 646 result dict.

    **What T_optimal is:**
        The temperature found by Platt scaling calibration in Exp 646.  It is stored
        under the key "T_optimal" in the result JSON.  When this key is missing (e.g.
        the experiment was blocked before calibration ran), None is returned so the
        caller can fall back gracefully instead of crashing.

    Parameters
    ----------
    exp646_result : dict
        Parsed Exp 646 result JSON.  Expected to contain "T_optimal" as a float.

    Returns
    -------
    float | None
        The temperature value as a float, or None if "T_optimal" is absent or not
        convertible to float.

    Spec: REQ-VERIFY-150-2, SCENARIO-VERIFY-199
    """
    value = exp646_result.get("T_optimal")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_platt_jepa(project_root: str) -> PlattCalibratedJEPA:
    """Load JEPA v14 Platt calibration from the Exp 646 result, if available.

    **Full pipeline (layman explanation):**
        1. Search results/ for any file matching "experiment_646_*.json".
        2. Parse the JSON and extract T_optimal (the Platt temperature).
        3. Return a PlattCalibratedJEPA with loaded=True and the temperature.
        If any step fails (file missing, key absent, parse error), return
        loaded=False with platt_temperature=1.0 so callers can use this as a
        no-op identity scaling without special-casing.

    Parameters
    ----------
    project_root : str
        Absolute path to the repository root.

    Returns
    -------
    PlattCalibratedJEPA
        Dataclass with loaded=True and the calibrated temperature when Exp 646
        result is available; loaded=False with platt_temperature=1.0 otherwise.

    Spec: REQ-VERIFY-150, REQ-VERIFY-150-3
    """
    result = find_exp646_result(project_root)
    if result is None:
        return PlattCalibratedJEPA(platt_temperature=1.0, model_path=None, loaded=False)

    temperature = extract_platt_temperature(result)
    if temperature is None:
        return PlattCalibratedJEPA(platt_temperature=1.0, model_path=None, loaded=False)

    # Extract optional model checkpoint path if recorded.
    # Exp 646 records this under "scaler_saved" (the Platt scaler file path).
    # We store it in model_path for future use but do not require it now.
    model_path: str | None = result.get("scaler_saved")

    return PlattCalibratedJEPA(
        platt_temperature=temperature,
        model_path=model_path,
        loaded=True,
    )
