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
import torch

# Use GPU for all linear algebra if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_effective_rank(sv: torch.Tensor) -> float:
    """Compute effective rank via Shannon entropy of normalized singular values."""
    sv = sv / sv.sum()
    sv = sv[sv > 1e-10]
    entropy = -(sv * sv.log()).sum()
    return float(entropy.exp())


def compute_condition_number(sv: torch.Tensor) -> float:
    """Ratio of largest to smallest nonzero singular value."""
    sv_min = sv[sv > 1e-10]
    if len(sv_min) == 0:
        return float("inf")
    return float(sv[0] / sv_min[-1])


def compute_spectral_gap(sv: torch.Tensor) -> float:
    """Ratio of top to second singular value."""
    if len(sv) < 2 or sv[1] < 1e-10:
        return float("inf")
    return float(sv[0] / sv[1])


def profile_weight_matrix(
    name: str, weight: torch.Tensor, max_sv_dim: int = 4096
) -> dict:
    """Compute structural features for a weight matrix using GPU-accelerated SVD.

    Args:
        name: Name of this weight matrix.
        weight: 2D tensor on GPU (or CPU).
        max_sv_dim: Max dimension for SVD. Larger matrices are sampled.

    Returns:
        Dictionary of structural features.
    """
    if weight.ndim != 2:
        return {"name": name, "shape": list(weight.shape), "skipped": "not 2D"}

    m, n = weight.shape
    profile = {
        "name": name,
        "shape": [m, n],
        "num_parameters": m * n,
    }

    # Per-neuron L2 norms (rows)
    row_norms = weight.norm(dim=1)
    profile["neuron_norms"] = {
        "mean": float(row_norms.mean()),
        "std": float(row_norms.std()),
        "min": float(row_norms.min()),
        "max": float(row_norms.max()),
        "dead_fraction": float((row_norms < 1e-6).float().mean()),
    }

    # Per-column norms (input features)
    col_norms = weight.norm(dim=0)
    profile["feature_norms"] = {
        "mean": float(col_norms.mean()),
        "std": float(col_norms.std()),
        "min": float(col_norms.min()),
        "max": float(col_norms.max()),
        "dead_fraction": float((col_norms < 1e-6).float().mean()),
    }

    # Frobenius norm
    profile["frobenius_norm"] = float(weight.norm())

    # GPU-accelerated SVD
    svd_weight = weight
    if min(m, n) > max_sv_dim:
        row_idx = torch.randperm(m, device=weight.device)[:max_sv_dim]
        col_idx = torch.randperm(n, device=weight.device)[:max_sv_dim]
        svd_weight = weight[row_idx][:, col_idx]
        profile["svd_sampled"] = True

    try:
        sv = torch.linalg.svdvals(svd_weight)
        profile["effective_rank"] = compute_effective_rank(sv)
        profile["condition_number"] = compute_condition_number(sv)
        profile["spectral_gap"] = compute_spectral_gap(sv)
        top_k = min(10, len(sv))
        profile["top_singular_values"] = [float(v) for v in sv[:top_k]]
        sv_sq = sv**2
        total_energy = sv_sq.sum()
        if total_energy > 0:
            profile["energy_concentration"] = {
                "top1": float(sv_sq[0] / total_energy),
                "top5": float(sv_sq[:5].sum() / total_energy),
                "top10": float(sv_sq[:10].sum() / total_energy),
            }
    except RuntimeError:
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

    def _to_gpu(w):
        if isinstance(w, np.ndarray):
            return torch.from_numpy(w).float().to(DEVICE)
        return w.float().to(DEVICE)

    for eid in expert_ids:
        matrices = expert_weights[eid]
        if "gate_proj" in matrices:
            fc1_norms_per_expert[eid] = _to_gpu(matrices["gate_proj"]).norm(dim=1).cpu().numpy()
        if "down_proj" in matrices:
            fc2_norms_per_expert[eid] = _to_gpu(matrices["down_proj"]).norm(dim=0).cpu().numpy()

    # Channel magnitude alignment (Nemotron finding)
    if fc1_norms_per_expert and fc2_norms_per_expert:
        alignments = {}
        for eid in expert_ids:
            if eid in fc1_norms_per_expert and eid in fc2_norms_per_expert:
                fc1_n = fc1_norms_per_expert[eid]
                fc2_n = fc2_norms_per_expert[eid]
                if len(fc1_n) == len(fc2_n):
                    corr = np.corrcoef(fc1_n, fc2_n)[0, 1]
                    alignments[eid] = float(corr) if not np.isnan(corr) else 0.0
                    dead_mask = (fc1_n < 1e-6) & (fc2_n < 1e-6)
                    alignments[f"{eid}_dead_channels"] = float(dead_mask.mean())

        result["channel_alignment"] = alignments
        if alignments:
            corr_values = [v for k, v in alignments.items() if not k.endswith("_dead_channels")]
            result["mean_channel_alignment"] = float(np.mean(corr_values))

    # Expert specialization
    if fc1_norms_per_expert:
        all_norms = np.stack(
            [fc1_norms_per_expert[eid] for eid in expert_ids if eid in fc1_norms_per_expert]
        )
        mean_norms = all_norms.mean(axis=0)
        specializations = {}
        for eid in expert_ids:
            if eid in fc1_norms_per_expert:
                specializations[eid] = float(np.linalg.norm(fc1_norms_per_expert[eid] - mean_norms))
        result["expert_specialization"] = specializations

    # Expert overlap via GPU cosine similarity
    if len(expert_ids) > 1 and all("gate_proj" in expert_weights[eid] for eid in expert_ids):
        flat_weights = []
        for eid in expert_ids:
            w = _to_gpu(expert_weights[eid]["gate_proj"]).flatten()
            flat_weights.append(w / (w.norm() + 1e-10))
        stacked = torch.stack(flat_weights)
        overlap_matrix = (stacked @ stacked.T).cpu().numpy()
        triu_idx = np.triu_indices(n_experts, k=1)
        overlaps = overlap_matrix[triu_idx]
        result["expert_overlap"] = {
            "mean": float(overlaps.mean()),
            "std": float(overlaps.std()),
            "min": float(overlaps.min()),
            "max": float(overlaps.max()),
        }
        del stacked

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

    # Convert to GPU tensor
    if isinstance(router_weight, np.ndarray):
        rw = torch.from_numpy(router_weight).float().to(DEVICE)
    else:
        rw = router_weight.float().to(DEVICE)

    expert_norms = rw.norm(dim=1)
    result["expert_norms"] = {
        "values": [float(v) for v in expert_norms],
        "mean": float(expert_norms.mean()),
        "std": float(expert_norms.std()),
        "ratio_max_min": float(expert_norms.max() / (expert_norms.min() + 1e-10)),
    }

    sv = torch.linalg.svdvals(rw)
    result["effective_rank"] = compute_effective_rank(sv)
    result["condition_number"] = compute_condition_number(sv)

    normed = rw / (rw.norm(dim=1, keepdim=True) + 1e-10)
    sim_matrix = (normed @ normed.T).cpu().numpy()
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
        with safe_open(str(sf), framework="pt") as f:
            for key in f.keys():
                all_params[key] = {"file": sf.name, "shape": list(f.get_tensor(key).shape)}

    print(f"Model has {len(all_params)} parameter tensors")

    # Identify model type.
    is_moe = any("experts" in k for k in all_params)
    print(f"Model type: {'MoE' if is_moe else 'Dense'}")

    result = {
        "model_name": model_name,
        "is_moe": is_moe,
        "num_parameters_total": int(sum(
            np.prod(v["shape"]) for v in all_params.values()
        )),
        "num_tensors": len(all_params),
        "layer_profiles": [],
    }

    # Second pass: profile each weight matrix.
    # We process one safetensors file at a time to control memory.
    moe_expert_weights = {}  # layer_idx -> expert_idx -> {matrix_name -> np.array}
    router_profiles = {}

    print(f"Using device: {DEVICE}")

    for sf in st_files:
        print(f"  Processing {sf.name}...")
        with safe_open(str(sf), framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key).float().to(DEVICE)

                # Skip 1D tensors (biases, norms)
                if tensor.ndim < 2:
                    del tensor
                    continue

                # Profile this weight matrix on GPU.
                profile = profile_weight_matrix(key, tensor)
                result["layer_profiles"].append(profile)
                n_profiled = len(result["layer_profiles"])
                if n_profiled % 20 == 0:
                    print(f"    {n_profiled} matrices profiled...")

                # Collect MoE expert weights for cross-expert analysis.
                keep_tensor = False
                if is_moe and "experts" in key:
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

                        # Store on CPU to free GPU memory
                        tensor_cpu = tensor.cpu()
                        if "w1" in key or "gate_proj" in key:
                            moe_expert_weights[layer_key][expert_idx]["gate_proj"] = tensor_cpu
                        elif "w2" in key or "down_proj" in key:
                            moe_expert_weights[layer_key][expert_idx]["down_proj"] = tensor_cpu
                        elif "w3" in key or "up_proj" in key:
                            moe_expert_weights[layer_key][expert_idx]["up_proj"] = tensor_cpu
                    except (ValueError, IndexError):
                        pass

                # Collect router weights.
                if "gate" in key and "experts" not in key and "proj" not in key:
                    router_profiles[key] = profile_router(tensor)

                del tensor
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

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

    # Custom encoder to handle numpy/torch types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    safe_model_name = model_name.replace("/", "_")
    json_path = output_path / f"weight_profile_{safe_model_name}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
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
    torch.manual_seed(42)
    load_and_profile(args.model, args.output_dir)


if __name__ == "__main__":
    main()
