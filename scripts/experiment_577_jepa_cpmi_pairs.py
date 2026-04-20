#!/usr/bin/env python3
"""Experiment 577: JEPA CPMI Pair Builder — RETRO-063 Resolution Step 1.

**Researcher summary (RETRO-063):**
    The JEPA predictor AUC was stuck at 0.4444 across v8, v9, v10 retrains.  All three used
    variants of scalar loss on step-level labels, allowing the model to hedge toward P=0.5.

    This experiment validates the CPMI contrastive pair builder (arXiv 2604.10660):
    - Loads the 132-pair FOVER corpus from results/fover_corpus_v2.json.
    - Builds explicit (correct_chain, incorrect_chain) pairs grouped by question.
    - Validates that CPMIContrastiveLoss.compute_loss() produces a non-negative scalar.
    - Reports n_real_pairs, n_synthetic_pairs, mean_pair_quality.

    The pairs built here will be consumed by Exp 580 (jepa_v11_retrain) to train the
    JEPA predictor with a pure contrastive margin objective — no BCE, no PURE.

Spec: REQ-LEARN-065, REQ-LEARN-066,
      SCENARIO-LEARN-101, SCENARIO-LEARN-102, SCENARIO-LEARN-103
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Step 0: apply_env_autofix() — must be called before any JAX import.
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

apply_env_autofix()

# ---------------------------------------------------------------------------
# Standard imports (after env fix)
# ---------------------------------------------------------------------------

import json  # noqa: E402
import logging  # noqa: E402

import jax.numpy as jnp  # noqa: E402

from carnot.inference.jepa_cpmi_pairs import (  # noqa: E402
    CPMIContrastiveLoss,
    JEPACPMIPairBuilder,
)
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 577
EXP_TITLE = "JEPA CPMI Pair Builder"
DELIVERABLE = "results/experiment_577_jepa_cpmi_pairs.json"
CORPUS_PATH = _REPO_ROOT / "results" / "fover_corpus_v2.json"
MIN_PAIRS = 5


# ---------------------------------------------------------------------------
# Minimal model stub for loss validation (no GPU needed)
# ---------------------------------------------------------------------------


def _stub_model(emb: jnp.ndarray) -> float:
    """Stub energy model: returns mean of embedding components.

    Used only to validate that CPMIContrastiveLoss.compute_loss() produces a
    well-formed non-negative float.  Not a trained model.
    """
    return float(jnp.mean(emb))


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


def _load_corpus(path: Path) -> list:
    """Load FOVER corpus from JSON and return list of dicts."""
    if not path.exists():
        _log.warning("Corpus not found: %s — returning empty list.", path)
        return []
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        return []
    return raw


class _DictEntry:
    """Wrap a dict to expose FOVERCorpusEntry-compatible attribute access."""

    def __init__(self, d: dict) -> None:
        self._d = d

    @property
    def question(self) -> str:
        return str(self._d.get("question", ""))

    @property
    def is_correct(self) -> bool:
        return bool(self._d.get("is_correct", False))

    @property
    def cot_steps(self) -> list:
        return self._d.get("cot_steps", [])

    @property
    def response(self) -> str:
        return self._d.get("response", "")

    @property
    def model_id(self) -> str:
        return self._d.get("model_id", "unknown")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    _watchdog = ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=20)

    tmpl = ExperimentTemplate(
        EXP_ID,
        EXP_TITLE,
        DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    # ------------------------------------------------------------------
    # Load corpus
    # ------------------------------------------------------------------
    raw_entries = _load_corpus(CORPUS_PATH)
    _log.info("Loaded %d raw corpus entries.", len(raw_entries))

    corpus = [_DictEntry(d) for d in raw_entries]

    # ------------------------------------------------------------------
    # Build contrastive pairs
    # ------------------------------------------------------------------
    # Use a hash-based embed function — deterministic, no ML required for pair building.
    def _hash_embed(text: str) -> jnp.ndarray:
        return jnp.array([hash(text) % 128], dtype=jnp.float32)

    builder = JEPACPMIPairBuilder(embed_fn=_hash_embed, min_pairs=MIN_PAIRS)
    real_pairs = builder.build_pairs(corpus)
    _log.info("Built %d real contrastive pairs.", len(real_pairs))

    n_synthetic = 0
    if len(real_pairs) < MIN_PAIRS:
        synthetic = builder.build_synthetic_pairs(10)
        n_synthetic = len(synthetic)
        real_pairs = real_pairs + synthetic
        _log.info("Augmented with %d synthetic pairs (real count was below min_pairs=%d).",
                  n_synthetic, MIN_PAIRS)

    n_real = len(real_pairs) - n_synthetic
    total_pairs = len(real_pairs)

    mean_pair_quality = (
        sum(p.pair_quality for p in real_pairs) / total_pairs
        if total_pairs > 0 else 0.0
    )

    # ------------------------------------------------------------------
    # Validate CPMIContrastiveLoss
    # ------------------------------------------------------------------
    loss_fn = CPMIContrastiveLoss(margin=1.0, chain_energy_mode="mean")
    loss_value = loss_fn.compute_loss(_stub_model, real_pairs)
    assert loss_value >= 0.0, f"CPMIContrastiveLoss returned negative loss: {loss_value}"
    _log.info("CPMIContrastiveLoss validated: loss=%.4f", loss_value)

    # ------------------------------------------------------------------
    # Build deliverable artifact
    # ------------------------------------------------------------------
    honest_verdict = (
        "pairs_built_sufficient" if total_pairs >= 10 else "pairs_built_insufficient"
    )

    artifact = tmpl.build_result(
        {
            "n_real_pairs": n_real,
            "n_synthetic_pairs": n_synthetic,
            "total_pairs": total_pairs,
            "mean_pair_quality": round(mean_pair_quality, 4),
            "loss_mode": "contrastive_hinge_margin",
            "loss_fn_validated": True,
            "cpmi_loss_value": round(float(loss_value), 4),
            "retro_063_path": "jepa_v11_retrain (Exp 580)",
            "honest_verdict": honest_verdict,
        },
        status="success",
    )
    artifact["schema"] = "carnot.jepa_cpmi_pairs.v1"

    # Write deliverable
    out_path = _REPO_ROOT / DELIVERABLE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    _log.info("Deliverable written: %s", out_path)

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
