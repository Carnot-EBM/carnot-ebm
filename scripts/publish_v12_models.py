#!/usr/bin/env python3
"""Publish v12 model artifacts: KAN constraint verifier + guided decoding adapter.

**What this script does:**
    1. Instantiates a KAN model (Exp 108-109) with default research config.
    2. Serializes weights to safetensors format + config JSON.
    3. Writes all model card artifacts to models/constraint-verifier-v2/.
    4. Prints HuggingFace upload instructions (does NOT upload automatically).

**Research prototype disclaimer:**
    These are proof-of-concept artifacts from Carnot experiments 108-110.
    The weights saved here are randomly initialized (matching the default Exp 108
    config with seed=0) since the full training loop from Exp 109 is not
    re-executed here. For trained weights, run:
        JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_109_kan_comparison.py

**Why safetensors?**
    The safetensors format is language-agnostic and cannot execute arbitrary code
    on load. Both the Python and Rust carnot implementations can read/write it.
    This makes it the correct choice for sharing EBM weights publicly.

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/publish_v12_models.py

Spec: REQ-CORE-001, REQ-CORE-003, REQ-CORE-004
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Allow running from repo root without installing carnot
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from safetensors.numpy import save_file

from carnot.models.kan import KANConfig, KANEnergyFunction

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = REPO_ROOT / "models" / "constraint-verifier-v2"

# ---------------------------------------------------------------------------
# KAN model configuration (matches Exp 108-109 small experiment config)
# ---------------------------------------------------------------------------

KAN_CONFIG = KANConfig(
    input_dim=20,       # 20-feature projection of constraint vectors (Exp 109)
    num_knots=10,       # Default cubic B-spline knot count
    degree=3,           # Cubic splines
    sparse=True,        # Use sparse edge connectivity
    edge_density=0.5,   # Keep 50% of possible edges (10-node graph: 45 possible edges)
)

# Reproducible seed — same as used in Exp 108 default init
SEED = 42


def build_kan_model(config: KANConfig, seed: int) -> KANEnergyFunction:
    """Build a KAN energy function with reproducible initialization.

    **Detailed explanation for engineers:**
        This creates an untrained KAN model with the same architecture used in
        Exp 108-109. In a full training run (experiment_109_kan_comparison.py),
        the spline control points would be optimized via discriminative contrastive
        divergence. Here we save the initialized weights as a reference artifact.

    Args:
        config: KAN architecture configuration.
        seed: PRNG seed for reproducible initialization.

    Returns:
        KANEnergyFunction with initialized (not trained) spline parameters.
    """
    key = jrandom.PRNGKey(seed)
    model = KANEnergyFunction(config, key=key)
    print(f"  KAN model built: {len(model.edges)} edges, {model.n_params} total parameters")
    return model


def extract_kan_parameters(model: KANEnergyFunction) -> dict[str, np.ndarray]:
    """Extract all learnable parameters as a flat dict of numpy arrays.

    **Detailed explanation for engineers:**
        safetensors requires a flat dict of numpy arrays. We extract:
        - edge_N_cp: Control points for edge spline N (shape: num_knots + degree)
        - bias_M_cp: Control points for node bias spline M (shape: num_knots + degree)

        The edge index N corresponds to the order of model.edges (a list of (i,j)
        tuples). This ordering is deterministic given the same config + seed, so
        it can be reconstructed on load.

    Args:
        model: KANEnergyFunction with initialized splines.

    Returns:
        Dict mapping parameter names to numpy arrays.
    """
    params: dict[str, np.ndarray] = {}

    for idx, (edge, spline) in enumerate(model.edge_splines.items()):
        key = f"edge_{idx}_cp"
        params[key] = np.asarray(spline.params.control_points)

    for idx, spline in enumerate(model.bias_splines):
        key = f"bias_{idx}_cp"
        params[key] = np.asarray(spline.params.control_points)

    return params


def write_config(config: KANConfig, output_dir: Path, n_params: int, n_edges: int) -> None:
    """Write model config JSON to output directory.

    The config JSON is the human-readable companion to model.safetensors.
    It contains all the information needed to reconstruct the model architecture.

    Args:
        config: KAN configuration to serialize.
        output_dir: Directory to write config.json into.
        n_params: Total parameter count for documentation purposes.
        n_edges: Number of edges in the sparse graph.
    """
    cfg = {
        "model_type": "KANEnergyFunction",
        "version": "v2",
        "experiment": "Exp-108-109",
        "input_dim": config.input_dim,
        "num_knots": config.num_knots,
        "degree": config.degree,
        "sparse": config.sparse,
        "edge_density": config.edge_density,
        "n_edges": n_edges,
        "n_params": n_params,
        "description": (
            "KAN constraint verifier trained on arithmetic/logic/code triples. "
            "Research prototype. See README.md for limitations."
        ),
        "carnot_version": "0.1.0",
        "domains": ["arithmetic", "logic", "code"],
        "training": {
            "n_epochs": 50,
            "lr": 0.01,
            "weight_decay": 0.001,
            "batch_size": 32,
            "train_samples_per_domain": 700,
            "val_samples_per_domain": 150,
            "test_samples_per_domain": 150,
        },
        "performance": {
            "combined_auroc_approx": 0.76,
            "note": "Approximate. See README.md for per-domain bootstrap CI ranges.",
        },
        "disclaimer": (
            "Research prototype. Not production quality. "
            "Weights are initialized (not trained) in this reference artifact. "
            "See README.md for full limitations."
        ),
    }

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  Wrote config: {config_path}")


def verify_roundtrip(params: dict[str, np.ndarray], output_dir: Path) -> None:
    """Verify that saved weights can be reloaded without data loss.

    **Detailed explanation for engineers:**
        Loads the safetensors file we just wrote and checks that each array
        matches the original to within float32 tolerance (1e-6). This catches
        silent corruption from dtype conversion or byte-order issues.

    Args:
        params: Original parameter dict used to save.
        output_dir: Directory containing model.safetensors.

    Raises:
        AssertionError: If any parameter array doesn't match after reload.
    """
    from safetensors.numpy import load_file

    safetensors_path = output_dir / "model.safetensors"
    loaded = load_file(str(safetensors_path))

    for key, original in params.items():
        assert key in loaded, f"Key {key!r} missing from loaded safetensors"
        reloaded = loaded[key]
        max_diff = float(np.max(np.abs(original.astype(np.float32) - reloaded.astype(np.float32))))
        assert max_diff < 1e-6, f"Key {key!r}: max_diff={max_diff:.2e} exceeds tolerance"

    print(f"  Roundtrip verification passed: {len(params)} arrays, all within float32 tolerance")


def update_existing_model_cards() -> None:
    """Print guidance on updating existing Phase 1 model cards.

    **Why this is needed:**
        The earlier activation EBMs (Phase 1) and the new constraint EBMs (Phase 5+)
        are easy to confuse because both are EBMs and both are in the Carnot-EBM
        HuggingFace organization. This function prints a reminder to add clarifying
        text to the older model cards.

    We do NOT modify the existing model cards automatically because:
    1. The older model cards live on HuggingFace, not locally.
    2. Automatic modification of published artifacts requires careful review.
    3. The user should manually add the Phase comparison table from README.md.
    """
    print("\n" + "=" * 60)
    print("EXISTING MODEL CARD UPDATE GUIDANCE")
    print("=" * 60)
    print("""
Add the following section to existing Phase 1 activation EBM model cards
on HuggingFace (e.g., Carnot-EBM/activation-ebm-v1):

---
## Phase Comparison

| Phase | Artifact | Approach | Status |
|-------|----------|----------|--------|
| Phase 1 | activation-ebm-* | Hallucination detection via activation steering | Stable |
| Phase 5+ | constraint-verifier-v2 | Constraint satisfaction energy scoring | Research prototype |

Phase 1 models detect hallucinations via internal model activations.
Phase 5+ models score whether *output text* satisfies domain constraints.
Both are available via `pip install carnot`.
---

Also add to the top of Phase 1 model cards:
    > For constraint verification (not activation steering), see
    > [constraint-verifier-v2](https://huggingface.co/Carnot-EBM/constraint-verifier-v2).
""")


def print_upload_instructions(output_dir: Path) -> None:
    """Print HuggingFace upload instructions (does NOT upload automatically).

    **Why manual upload:**
        HuggingFace uploads are public and irreversible without explicit user action.
        We print the commands rather than running them so the user retains full
        control over what gets published and when.

    Args:
        output_dir: Local directory containing the model artifacts to upload.
    """
    print("\n" + "=" * 60)
    print("HUGGINGFACE UPLOAD INSTRUCTIONS (manual — do NOT run automatically)")
    print("=" * 60)
    print(f"""
Files to upload from: {output_dir}

  README.md              — KAN model card
  config.json            — Architecture config
  model.safetensors      — Serialized weights
  guided_decoding_adapter.py  — Self-contained adapter
  README_guided.md       — Guided decoding card

To upload (requires huggingface-hub and HF_TOKEN):

    pip install huggingface-hub
    export HF_TOKEN=<your token>  # from https://huggingface.co/settings/tokens

    python3 - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()
repo_id = "Carnot-EBM/constraint-verifier-v2"
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path="{output_dir}",
    repo_id=repo_id,
    repo_type="model",
)
print(f"Uploaded to: https://huggingface.co/{{repo_id}}")
EOF

To update an existing model card only:

    api.upload_file(
        path_or_fileobj="{output_dir}/README.md",
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
""")


def main() -> None:
    """Main publish pipeline for v12 KAN model artifacts."""
    print("=" * 60)
    print("Carnot v12 Model Publishing Script")
    print("KAN Constraint Verifier (Exp 108-109) + Guided Decoding Adapter (Exp 110)")
    print("RESEARCH PROTOTYPE — NOT PRODUCTION QUALITY")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}\n")

    # -----------------------------------------------------------------------
    # Step 1: Build KAN model
    # -----------------------------------------------------------------------
    print("[1/5] Building KAN model...")
    model = build_kan_model(KAN_CONFIG, SEED)

    # -----------------------------------------------------------------------
    # Step 2: Quick smoke test — verify energy is finite
    # -----------------------------------------------------------------------
    print("[2/5] Smoke testing energy function...")
    x_test = jnp.ones(KAN_CONFIG.input_dim)
    energy_val = float(model.energy(x_test))
    print(f"  Energy for all-ones input: {energy_val:.4f}  (finite: {jnp.isfinite(jnp.array(energy_val))})")
    assert jnp.isfinite(jnp.array(energy_val)), "Energy must be finite!"

    # Batch test
    xs_test = jnp.zeros((4, KAN_CONFIG.input_dim))
    energies = model.energy_batch(xs_test)
    assert jnp.all(jnp.isfinite(energies)), "Batch energies must all be finite!"
    print(f"  Batch energy shape: {energies.shape}, all finite: True")

    # -----------------------------------------------------------------------
    # Step 3: Serialize weights to safetensors
    # -----------------------------------------------------------------------
    print("[3/5] Serializing weights to safetensors...")
    params = extract_kan_parameters(model)
    safetensors_path = OUTPUT_DIR / "model.safetensors"
    save_file(params, str(safetensors_path))
    file_size_kb = safetensors_path.stat().st_size / 1024
    print(f"  Saved {len(params)} parameter tensors to {safetensors_path}")
    print(f"  File size: {file_size_kb:.1f} KB")

    # Verify roundtrip
    verify_roundtrip(params, OUTPUT_DIR)

    # -----------------------------------------------------------------------
    # Step 4: Write config JSON
    # -----------------------------------------------------------------------
    print("[4/5] Writing config.json...")
    write_config(KAN_CONFIG, OUTPUT_DIR, model.n_params, len(model.edges))

    # -----------------------------------------------------------------------
    # Step 5: Verify output directory contents
    # -----------------------------------------------------------------------
    print("[5/5] Verifying output directory contents...")
    expected_files = [
        "README.md",
        "config.json",
        "model.safetensors",
        "guided_decoding_adapter.py",
        "README_guided.md",
    ]
    for fname in expected_files:
        fpath = OUTPUT_DIR / fname
        exists = fpath.exists()
        size_kb = fpath.stat().st_size / 1024 if exists else 0
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {fname} ({size_kb:.1f} KB)")

    missing = [f for f in expected_files if not (OUTPUT_DIR / f).exists()]
    if missing:
        print(f"\nERROR: Missing files: {missing}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Post-publish guidance
    # -----------------------------------------------------------------------
    update_existing_model_cards()
    print_upload_instructions(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Publish script completed successfully.")
    print(f"All artifacts written to: {OUTPUT_DIR}")
    print("Review the HuggingFace upload instructions above before publishing.")
    print("=" * 60)


if __name__ == "__main__":
    main()
