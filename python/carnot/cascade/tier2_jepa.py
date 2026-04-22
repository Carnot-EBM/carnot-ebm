"""Tier 2 JEPA cascade wrapper — loads JEPALambdaRankV18 as the active Tier 2 predictor.

WHY THIS MODULE EXISTS:
    The cascade previously used JEPA v17 as Tier 2, but v17 is blocked in the exclusion
    manifest (OOD AUC=0.4819, below random chance).  This module is the single
    authoritative place that loads JEPA v18, enforces the version-pin, and exposes
    the scorer to the rest of the pipeline.

VERSION SAFETY (the main invariant):
    Callers must NOT hardcode a version string.  The correct call is always:

        model = load_v18_from_manifest()

    If the manifest changes (e.g. v19 supersedes v18), this module is updated — not
    every caller.  Passing an explicit version string that is blocked raises ValueError
    immediately so broken code is caught before it reaches production inference.

CHECKPOINT LOADING:
    JEPALambdaRankV18 weights are stored as a NumPy .npz file.  The caller passes the
    path to a saved checkpoint (e.g. produced by Exp 717 or a warm-up training run).
    When no checkpoint is provided, the model starts with random weights (useful for
    testing the load path without a real checkpoint).

Spec: REQ-INFRA-043, SCENARIO-INFRA-052, SCENARIO-INFRA-053
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

from carnot.samplers.jepa_v18_lambdarank import JEPALambdaRankV18

# ---------------------------------------------------------------------------
# Blocked versions — mirrors conductor_exclusion_manifest.json
# ---------------------------------------------------------------------------

_BLOCKED_JEPA_VERSIONS: frozenset[str] = frozenset({"v15", "v16", "v17"})
"""JEPA versions that are blocked by the exclusion manifest.

WHY these three: v15 OOD AUC=0.4751, v16=0.4759, v17=0.4819 — all below random
chance (0.5).  Deploying any of these as Tier 2 would actively degrade verification
quality compared to a random scorer.  The exclusion manifest blocks them
permanently (REQ-INFRA-038, REQ-INFRA-040) until a superseding version is validated.
"""

_ACTIVE_VERSION = "v18"
"""The currently active JEPA tier version, controlled by the exclusion manifest."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_v18_from_manifest(
    version: str = _ACTIVE_VERSION,
    checkpoint_path: Optional[str] = None,
) -> JEPALambdaRankV18:
    """Load a JEPALambdaRankV18 model, enforcing the exclusion manifest version block.

    This is the only correct way to obtain a Tier 2 JEPA model.  Passing a blocked
    version raises ValueError immediately — the error message includes the version
    string so callers can identify the stale reference.

    WHY the manifest check is here and not in the caller: the pipeline has many
    entry points (cascade scripts, the conductor, integration tests).  Centralising
    the block check here means it is impossible to accidentally route around it.

    Parameters
    ----------
    version : str
        JEPA version string to load.  Default is the current active version ("v18").
        Passing "v15", "v16", or "v17" raises ValueError.
    checkpoint_path : str | None
        Path to a .npz file produced by ``save_checkpoint()``.  When None, the model
        is initialised with random weights (useful for unit tests and smoke tests that
        verify the load path without a real trained checkpoint).

    Returns
    -------
    JEPALambdaRankV18
        A loaded (or freshly initialised) v18 ranking model.

    Raises
    ------
    ValueError
        When ``version`` is in the blocked set (v15 / v16 / v17).

    Spec: REQ-INFRA-043, SCENARIO-INFRA-052, SCENARIO-INFRA-053
    """
    if version in _BLOCKED_JEPA_VERSIONS:
        raise ValueError(
            f"JEPA version '{version}' is blocked by the exclusion manifest "
            f"(OOD AUC below random chance).  Use '{_ACTIVE_VERSION}' instead.  "
            f"Blocked versions: {sorted(_BLOCKED_JEPA_VERSIONS)}"
        )

    model = JEPALambdaRankV18()

    if checkpoint_path is not None:
        _load_weights_from_npz(model, checkpoint_path)

    return model


def save_checkpoint(model: JEPALambdaRankV18, path: str) -> None:
    """Save all model weight arrays to a NumPy .npz archive.

    WHY .npz: it is a single file, supports arbitrary named arrays, and requires
    only NumPy (already a hard dependency).  safetensors would be preferred for
    production cross-language use (Rust ↔ Python), but for the NumPy-only v18 model
    .npz gives zero additional dependencies.

    Parameters
    ----------
    model : JEPALambdaRankV18
        Trained model whose weights to save.
    path : str
        Destination file path (e.g. "results/jepa_v18_weights.npz").
        Parent directories are created automatically.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(
        path,
        W1=model.W1,
        b1=model.b1,
        W2=model.W2,
        b2=model.b2,
        W3=model.W3,
        b3=model.b3,
    )


def _load_weights_from_npz(model: JEPALambdaRankV18, path: str) -> None:
    """Load weight arrays from a .npz file into an existing model instance (in-place).

    Parameters
    ----------
    model : JEPALambdaRankV18
        Model instance to populate.  Its existing weights are overwritten.
    path : str
        Path to a .npz file produced by ``save_checkpoint()``.

    Raises
    ------
    FileNotFoundError
        When ``path`` does not exist.
    KeyError
        When a required weight key is missing from the archive.
    """
    archive = np.load(path)
    model.W1 = archive["W1"].astype(np.float32)
    model.b1 = archive["b1"].astype(np.float32)
    model.W2 = archive["W2"].astype(np.float32)
    model.b2 = archive["b2"].astype(np.float32)
    model.W3 = archive["W3"].astype(np.float32)
    model.b3 = archive["b3"].astype(np.float32)
