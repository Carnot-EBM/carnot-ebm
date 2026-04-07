#!/usr/bin/env python3
"""Experiment 32: Weight Structure Profiling — Pure Weight Analysis.

Analyzes the frozen weight matrices of transformer and MoE models to extract
structural features that may predict hallucination-prone regions. This script
requires ZERO inference and ZERO labeled data — it reads only the model's
parameter tensors.

For each layer, we compute:
  - Effective rank (via singular value decay)
  - Per-neuron L2 norm distribution (mean, std, min, max)
  - Weight matrix condition number
  - Spectral gap (ratio of top-2 singular values)

For MoE models (like Mixtral-8x7B), we additionally compute:
  - Per-expert specialization (L2 distance from mean expert)
  - Expert overlap (cosine similarity matrix between all expert pairs)
  - Router weight clustering (what input regions map to which experts)
  - Channel magnitude profiles (FC1/FC2 norm alignment per expert)

Output: A JSON "weight health map" for the model, plus visualization.

REQ: Roadmap v5, Phase 1, Experiment 32
SCENARIO: SCENARIO-WEIGHT-PROFILE-001 — extract weight structure without inference

Usage:
    python scripts/experiment_32_weight_profiling.py --model mistralai/Mixtral-8x7B-v0.1
    python scripts/experiment_32_weight_profiling.py --model Qwen/Qwen3-0.6B
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def compute_effective_rank(singular_values: np.ndarray) -> float:
    """Compute effective rank via Shannon entropy of normalized singular values.

    The effective rank measures how many singular values meaningfully contribute
    to the matrix. A matrix with one dominant singular value has effective rank ~1.
    A matrix with all equal singular values has effective rank = min(m, n).

    This tells us how "spread out" the information is in a weight matrix.
    High effective rank = the layer uses its full capacity.
    Low effective rank = the layer could be compressed without loss.

    Reference: Roy & Vetterli, "The effective rank: A measure of effective
    dimensionality," 2007.
    """
    # Normalize singular values to form a probability distribution.
    sv = singular_values / singular_values.sum()
    # Remove zeros to avoid log(0).
    sv = sv[sv > 1e-10]
    # Shannon entropy of the normalized singular value distribution.
    entropy = -np.sum(sv * np.log(sv))
    # Effective rank = exp(entropy), bounded by the matrix dimensions.
    return float(np.exp(entropy))


def compute_condition_number(singular_values: np.ndarray) -> float:
    """Ratio of largest to smallest nonzero singular value.

    A high condition number means the matrix is near-singular — small input
    changes cause large output changes. For a weight matrix, this means the
    layer is "fragile" and may behave unpredictably on out-of-distribution inputs.

    We clamp the minimum singular value to avoid division by zero.
    """
    sv_min = singular_values[singular_values > 1e-10]
    if len(sv_min) == 0:
        return float("inf")
    return float(singular_values[0] / sv_min[-1])


def compute_spectral_gap(singular_values: np.ndarray) -> float:
    """Ratio of the top singular value to the second singular value.

    A large spectral gap means one direction dominates the matrix — the layer
    compresses most information into a single direction. This can indicate
    either strong learned structure (good) or rank deficiency (fragile).
    """
    if len(singular_values) < 2 or singular_values[1] < 1e-10:
        return float("inf")
    return float(singular_values[0] / singular_values[1])


def profile_weight_matrix(
    name: str, weight: np.ndarray, max_sv_dim: int = 4096
) -> dict:
    """Compute all structural features for a single weight matrix.

    Args:
        name: Human-readable name for this weight matrix (e.g. "layer.0.mlp.up_proj").
        weight: The 2D weight tensor as a numpy array.
        max_sv_dim: Maximum dimension for SVD computation. Matrices larger than this
                    are sampled randomly to keep computation tractable.

    Returns:
        Dictionary of structural features for this weight matrix.
    """
    if weight.ndim != 2:
        return {"name": name, "shape": list(weight.shape), "skipped": "not 2D"}

    m, n = weight.shape
    profile = {
        "name": name,
        "shape": [m, n],
        "num_parameters": m * n,
    }

    # Per-neuron L2 norms (output neurons = rows).
    # Each row is one output neuron. Its norm tells us how "active" that neuron is.
    # Neurons with very low norms are effectively dead — they contribute almost
    # nothing to the output regardless of input.
    row_norms = np.linalg.norm(weight, axis=1)
    profile["neuron_norms"] = {
        "mean": float(row_norms.mean()),
        "std": float(row_norms.std()),
        "min": float(row_norms.min()),
        "max": float(row_norms.max()),
        "dead_fraction": float((row_norms < 1e-6).mean()),
    }

    # Per-column norms (input features).
    # Each column is one input feature. Its norm tells us how much the layer
    # "listens to" that input dimension.
    col_norms = np.linalg.norm(weight, axis=0)
    profile["feature_norms"] = {
        "mean": float(col_norms.mean()),
        "std": float(col_norms.std()),
        "min": float(col_norms.min()),
        "max": float(col_norms.max()),
        "dead_fraction": float((col_norms < 1e-6).mean()),
    }

    # Frobenius norm (overall matrix "energy").
    profile["frobenius_norm"] = float(np.linalg.norm(weight, "fro"))

    # SVD-based features. For very large matrices, we subsample to keep
    # computation under a minute per matrix.
    svd_weight = weight
    if min(m, n) > max_sv_dim:
        # Randomly sample rows and columns to reduce SVD cost.
        row_idx = np.random.choice(m, max_sv_dim, replace=False)
        col_idx = np.random.choice(n, max_sv_dim, replace=False)
        svd_weight = weight[np.ix_(row_idx, col_idx)]
        profile["svd_sampled"] = True

    try:
        sv = np.linalg.svd(svd_weight, compute_uv=False)
        profile["effective_rank"] = compute_effective_rank(sv)
        profile["condition_number"] = compute_condition_number(sv)
        profile["spectral_gap"] = compute_spectral_gap(sv)
        # Top-10 singular value ratios (how energy is distributed).
        top_k = min(10, len(sv))
        profile["top_singular_values"] = [float(v) for v in sv[:top_k]]
        # Fraction of total "energy" in top-1, top-5, top-10 singular values.
        sv_squared = sv**2
        total_energy = sv_squared.sum()
        if total_energy > 0:
            profile["energy_concentration"] = {
                "top1": float(sv_squared[0] / total_energy),
                "top5": float(sv_squared[:5].sum() / total_energy),
                "top10": float(sv_squared[:10].sum() / total_energy),
            }
    except np.linalg.LinAlgError:
        profile["svd_failed"] = True

    return profile


def profile_moe_experts(
    expert_weights: dict[str, dict[str, np.ndarray]],
) -> dict:
    """Compute MoE-specific structural features across all experts.

    For each expert, we already have per-matrix profiles. Here we compute
    CROSS-EXPERT features that reveal the model's knowledge organization:

    - Specialization: how different each expert is from the "average expert"
    - Overlap: cosine similarity between expert weight matrices
    - Channel magnitude alignment: Nemotron's discovery that FC1 output channel
      norms align with FC2 input channel norms in structured patterns

    Args:
        expert_weights: Dict mapping expert_id -> {matrix_name -> numpy array}.
                        Expected keys per expert: "gate_proj" (FC1), "up_proj", "down_proj" (FC2).
    """
    if not expert_weights:
        return {"skipped": "no experts found"}

    expert_ids = sorted(expert_weights.keys())
    n_experts = len(expert_ids)

    result = {"num_experts": n_experts}

    # Compute per-expert FC1 (gate_proj) output channel norms and
    # FC2 (down_proj) input channel norms — the Nemotron channel magnitude pattern.
    fc1_norms_per_expert = {}
    fc2_norms_per_expert = {}

    for eid in expert_ids:
        matrices = expert_weights[eid]
        if "gate_proj" in matrices:
            # gate_proj shape: [intermediate_size, hidden_size]
            # Output channel norms = row norms.
            fc1_norms_per_expert[eid] = np.linalg.norm(
                matrices["gate_proj"], axis=1
            )
        if "down_proj" in matrices:
            # down_proj shape: [hidden_size, intermediate_size]
            # Input channel norms = column norms.
            fc2_norms_per_expert[eid] = np.linalg.norm(
                matrices["down_proj"], axis=0
            )

    # Channel magnitude alignment (Nemotron finding): do FC1 output norms
    # correlate with FC2 input norms within each expert?
    # High correlation = structured knowledge; the expert has learned which
    # channels to use and which to ignore.
    if fc1_norms_per_expert and fc2_norms_per_expert:
        alignments = {}
        for eid in expert_ids:
            if eid in fc1_norms_per_expert and eid in fc2_norms_per_expert:
                fc1_n = fc1_norms_per_expert[eid]
                fc2_n = fc2_norms_per_expert[eid]
                if len(fc1_n) == len(fc2_n):
                    # Pearson correlation between FC1 output and FC2 input channel norms.
                    corr = np.corrcoef(fc1_n, fc2_n)[0, 1]
                    alignments[eid] = float(corr) if not np.isnan(corr) else 0.0

                    # Count "dead channels" — channels where BOTH FC1 output and
                    # FC2 input have near-zero norm. These are unused capacity.
                    dead_mask = (fc1_n < 1e-6) & (fc2_n < 1e-6)
                    alignments[f"{eid}_dead_channels"] = float(dead_mask.mean())

        result["channel_alignment"] = alignments
        if alignments:
            corr_values = [
                v for k, v in alignments.items() if not k.endswith("_dead_channels")
            ]
            result["mean_channel_alignment"] = float(np.mean(corr_values))

    # Expert specialization: L2 distance of each expert's gate_proj from the mean.
    # Highly specialized experts are far from the mean — they handle specific domains.
    # Generic experts are close to the mean — they handle everything a little.
    if fc1_norms_per_expert:
        all_norms = np.stack(
            [fc1_norms_per_expert[eid] for eid in expert_ids if eid in fc1_norms_per_expert]
        )
        mean_norms = all_norms.mean(axis=0)
        specializations = {}
        for i, eid in enumerate(expert_ids):
            if eid in fc1_norms_per_expert:
                dist = float(np.linalg.norm(fc1_norms_per_expert[eid] - mean_norms))
                specializations[eid] = dist
        result["expert_specialization"] = specializations

    # Expert overlap: cosine similarity between expert gate_proj weight matrices.
    # Flattened to vectors, then pairwise cosine similarity.
    # High overlap = redundant experts. Low overlap = complementary knowledge.
    if len(expert_ids) > 1 and all(
        "gate_proj" in expert_weights[eid] for eid in expert_ids
    ):
        flat_weights = []
        for eid in expert_ids:
            w = expert_weights[eid]["gate_proj"].flatten()
            flat_weights.append(w / (np.linalg.norm(w) + 1e-10))
        flat_weights = np.stack(flat_weights)

        # Cosine similarity matrix.
        overlap_matrix = flat_weights @ flat_weights.T
        # Extract upper triangle (excluding diagonal).
        triu_idx = np.triu_indices(n_experts, k=1)
        overlaps = overlap_matrix[triu_idx]
        result["expert_overlap"] = {
            "mean": float(overlaps.mean()),
            "std": float(overlaps.std()),
            "min": float(overlaps.min()),
            "max": float(overlaps.max()),
        }

    return result


def profile_router(router_weight: np.ndarray) -> dict:
    """Analyze the MoE router's weight matrix.

    The router weight matrix W_gate ∈ R^{num_experts × hidden_dim} defines
    how the input space is partitioned into expert domains. Its structure
    tells us how the model organizes knowledge.

    We analyze:
    - How many distinct "regions" the router defines (effective rank)
    - How evenly the input space is partitioned (router weight norm balance)
    - Whether some experts are preferred (router bias analysis)
    """
    if router_weight.ndim != 2:
        return {"skipped": f"unexpected shape {router_weight.shape}"}

    n_experts, hidden_dim = router_weight.shape
    result = {
        "num_experts": n_experts,
        "hidden_dim": hidden_dim,
    }

    # Per-expert router weight norms. Experts with higher-norm router weights
    # are selected more aggressively — they're the "default" experts.
    expert_norms = np.linalg.norm(router_weight, axis=1)
    result["expert_norms"] = {
        "values": [float(v) for v in expert_norms],
        "mean": float(expert_norms.mean()),
        "std": float(expert_norms.std()),
        "ratio_max_min": float(expert_norms.max() / (expert_norms.min() + 1e-10)),
    }

    # Router effective rank — how many independent "routing directions" exist.
    sv = np.linalg.svd(router_weight, compute_uv=False)
    result["effective_rank"] = compute_effective_rank(sv)
    result["condition_number"] = compute_condition_number(sv)

    # Router weight cosine similarity between experts.
    # Experts with similar router weights activate on similar inputs.
    normed = router_weight / (np.linalg.norm(router_weight, axis=1, keepdims=True) + 1e-10)
    sim_matrix = normed @ normed.T
    triu_idx = np.triu_indices(n_experts, k=1)
    sims = sim_matrix[triu_idx]
    result["router_similarity"] = {
        "mean": float(sims.mean()),
        "std": float(sims.std()),
        "min": float(sims.min()),
        "max": float(sims.max()),
    }

    return result


def load_and_profile(model_name: str, output_dir: str) -> dict:
    """Load a model's weights and compute the full structural profile.

    Uses safetensors for efficient partial loading — we never need to load
    the full model into memory at once.
    """
    try:
        from safetensors import safe_open
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: requires safetensors and huggingface_hub packages")
        sys.exit(1)

    print(f"Loading model: {model_name}")
    model_path = snapshot_download(model_name)
    model_dir = Path(model_path)

    # Find all safetensors files.
    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        print(f"ERROR: No .safetensors files found in {model_dir}")
        sys.exit(1)

    print(f"Found {len(st_files)} safetensors files")

    # First pass: enumerate all parameter names and shapes without loading weights.
    all_params = {}
    for sf in st_files:
        with safe_open(str(sf), framework="numpy") as f:
            for key in f.keys():
                all_params[key] = {"file": sf.name, "shape": list(f.get_tensor(key).shape)}

    print(f"Model has {len(all_params)} parameter tensors")

    # Identify model type.
    is_moe = any("experts" in k for k in all_params)
    print(f"Model type: {'MoE' if is_moe else 'Dense'}")

    result = {
        "model_name": model_name,
        "is_moe": is_moe,
        "num_parameters_total": sum(
            np.prod(v["shape"]) for v in all_params.values()
        ),
        "num_tensors": len(all_params),
        "layer_profiles": [],
    }

    # Second pass: profile each weight matrix.
    # We process one safetensors file at a time to control memory.
    moe_expert_weights = {}  # layer_idx -> expert_idx -> {matrix_name -> np.array}
    router_profiles = {}

    for sf in st_files:
        print(f"  Processing {sf.name}...")
        with safe_open(str(sf), framework="numpy") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)

                # Skip 1D tensors (biases, norms) — they're not informative for
                # weight structure analysis.
                if tensor.ndim < 2:
                    continue

                # Profile this weight matrix.
                profile = profile_weight_matrix(key, tensor)
                result["layer_profiles"].append(profile)

                # Collect MoE expert weights for cross-expert analysis.
                if is_moe and "experts" in key:
                    # Parse expert index from key like
                    # "model.layers.0.block_sparse_moe.experts.3.w1.weight"
                    parts = key.split(".")
                    try:
                        expert_idx_pos = parts.index("experts") + 1
                        layer_idx_pos = parts.index("layers") + 1
                        expert_idx = parts[expert_idx_pos]
                        layer_idx = parts[layer_idx_pos]
                        layer_key = f"layer_{layer_idx}"

                        if layer_key not in moe_expert_weights:
                            moe_expert_weights[layer_key] = {}
                        if expert_idx not in moe_expert_weights[layer_key]:
                            moe_expert_weights[layer_key][expert_idx] = {}

                        # Identify matrix type from the key.
                        # Mixtral uses w1 (gate_proj), w2 (down_proj), w3 (up_proj).
                        if "w1" in key or "gate_proj" in key:
                            moe_expert_weights[layer_key][expert_idx]["gate_proj"] = tensor
                        elif "w2" in key or "down_proj" in key:
                            moe_expert_weights[layer_key][expert_idx]["down_proj"] = tensor
                        elif "w3" in key or "up_proj" in key:
                            moe_expert_weights[layer_key][expert_idx]["up_proj"] = tensor
                    except (ValueError, IndexError):
                        pass

                # Collect router weights.
                if "gate" in key and "experts" not in key and "proj" not in key:
                    router_profiles[key] = profile_router(tensor)

    # MoE cross-expert analysis.
    if moe_expert_weights:
        result["moe_analysis"] = {}
        for layer_key, experts in moe_expert_weights.items():
            print(f"  Analyzing MoE experts for {layer_key} ({len(experts)} experts)...")
            result["moe_analysis"][layer_key] = profile_moe_experts(experts)

    if router_profiles:
        result["router_analysis"] = router_profiles

    # Summary statistics across all layers.
    profiles = [p for p in result["layer_profiles"] if "skipped" not in p]
    if profiles:
        ranks = [p["effective_rank"] for p in profiles if "effective_rank" in p]
        conditions = [
            p["condition_number"]
            for p in profiles
            if "condition_number" in p and p["condition_number"] < 1e10
        ]
        dead_fracs = [
            p["neuron_norms"]["dead_fraction"] for p in profiles if "neuron_norms" in p
        ]

        result["summary"] = {
            "num_profiled_matrices": len(profiles),
            "effective_rank": {
                "mean": float(np.mean(ranks)) if ranks else None,
                "std": float(np.std(ranks)) if ranks else None,
                "min": float(np.min(ranks)) if ranks else None,
                "max": float(np.max(ranks)) if ranks else None,
            },
            "condition_number": {
                "mean": float(np.mean(conditions)) if conditions else None,
                "median": float(np.median(conditions)) if conditions else None,
                "max": float(np.max(conditions)) if conditions else None,
            },
            "dead_neuron_fraction": {
                "mean": float(np.mean(dead_fracs)) if dead_fracs else None,
                "max": float(np.max(dead_fracs)) if dead_fracs else None,
            },
        }

    # Save results.
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    safe_model_name = model_name.replace("/", "_")
    json_path = output_path / f"weight_profile_{safe_model_name}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nProfile saved to {json_path}")

    # Print summary.
    print(f"\n{'='*60}")
    print(f"WEIGHT PROFILE SUMMARY: {model_name}")
    print(f"{'='*60}")
    print(f"  Type: {'MoE' if is_moe else 'Dense'}")
    print(f"  Parameters: {result['num_parameters_total']:,}")
    print(f"  Tensors: {result['num_tensors']}")
    print(f"  Profiled 2D matrices: {len(profiles)}")
    if "summary" in result:
        s = result["summary"]
        if s["effective_rank"]["mean"]:
            print(f"  Avg effective rank: {s['effective_rank']['mean']:.1f}")
        if s["condition_number"]["median"]:
            print(f"  Median condition number: {s['condition_number']['median']:.1f}")
        if s["dead_neuron_fraction"]["mean"]:
            print(f"  Mean dead neuron fraction: {s['dead_neuron_fraction']['mean']:.4f}")
    if "moe_analysis" in result:
        for layer_key, moe in result["moe_analysis"].items():
            if "mean_channel_alignment" in moe:
                print(f"  {layer_key} channel alignment: {moe['mean_channel_alignment']:.3f}")
            if "expert_overlap" in moe:
                print(f"  {layer_key} expert overlap: {moe['expert_overlap']['mean']:.3f}")
    print(f"{'='*60}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 32: Weight Structure Profiling"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mixtral-8x7B-v0.1",
        help="HuggingFace model ID to profile",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/weight_profiles",
        help="Output directory for profile JSON",
    )
    parser.add_argument(
        "--max-sv-dim",
        type=int,
        default=4096,
        help="Max dimension for SVD (larger = slower but more accurate)",
    )
    args = parser.parse_args()

    np.random.seed(42)
    load_and_profile(args.model, args.output_dir)


if __name__ == "__main__":
    main()
