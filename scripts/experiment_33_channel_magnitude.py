#!/usr/bin/env python3
"""Experiment 33: Channel Magnitude Introspection (Nemotron-Inspired).

Nemotron 3 Super's training revealed that MoE expert weights develop
structured channel magnitude patterns (Figure 6 in the paper). FC1 output
channels and FC2 input channels converge toward zero in correlated patterns.
These patterns encode what each expert has learned — its "knowledge shape."

This experiment tests whether these patterns exist in other MoE models
(Mixtral-8x7B) and whether they predict hallucination at inference time.

Phase 1 (this script): Static weight analysis — zero inference needed.
  - Extract channel magnitude profiles for every expert
  - Measure FC1↔FC2 alignment per expert
  - Identify "dead" channels and "specialist" channels
  - Compare channel structure across experts

Phase 2 (future): Inference-time analysis — forward pass on unlabeled text.
  - Measure per-token activation alignment with expert channel profiles
  - Test whether low-alignment tokens correlate with hallucination

REQ: Roadmap v5, Phase 1, Experiment 33
SCENARIO: SCENARIO-CHANNEL-MAG-001 — static channel magnitude analysis

Usage:
    python scripts/experiment_33_channel_magnitude.py --model mistralai/Mixtral-8x7B-v0.1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def extract_expert_channel_profiles(model_name: str) -> dict:
    """Extract the channel magnitude profile for every expert in every MoE layer.

    For each expert, we compute:
      - FC1 (gate_proj/w1) output channel norms: shape [intermediate_size]
        These tell us which output dimensions this expert "produces."
      - FC2 (down_proj/w2) input channel norms: shape [intermediate_size]
        These tell us which input dimensions this expert "consumes."
      - FC1↔FC2 alignment: Pearson correlation between these two norm vectors.
        High correlation means the expert has developed the Nemotron pattern —
        channels that are active in FC1 output are also active in FC2 input.
      - Dead channels: channels where BOTH FC1 and FC2 have near-zero norms.
        These represent unused expert capacity.
      - Specialist channels: channels with high FC1 norm but low variance across
        inputs — the expert always uses these channels, suggesting specialized
        knowledge encoded in specific dimensions.

    Returns:
        Dictionary with per-layer, per-expert channel analysis.
    """
    try:
        from safetensors import safe_open
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: requires safetensors and huggingface_hub")
        sys.exit(1)

    print(f"Loading model: {model_name}")
    model_path = snapshot_download(model_name)
    model_dir = Path(model_path)

    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        print(f"ERROR: No .safetensors files in {model_dir}")
        sys.exit(1)

    # Collect all expert weight matrices, organized by layer and expert.
    # We need gate_proj (FC1) and down_proj (FC2) for channel magnitude analysis.
    experts = {}  # layer_idx -> expert_idx -> {"gate_proj": array, "down_proj": array}

    for sf in st_files:
        print(f"  Scanning {sf.name}...")
        with safe_open(str(sf), framework="numpy") as f:
            for key in f.keys():
                if "experts" not in key:
                    continue

                parts = key.split(".")
                try:
                    layer_idx = parts[parts.index("layers") + 1]
                    expert_idx = parts[parts.index("experts") + 1]
                except (ValueError, IndexError):
                    continue

                layer_key = f"layer_{layer_idx}"
                if layer_key not in experts:
                    experts[layer_key] = {}
                if expert_idx not in experts[layer_key]:
                    experts[layer_key][expert_idx] = {}

                tensor = f.get_tensor(key)

                # Mixtral naming: w1 = gate_proj (FC1), w2 = down_proj (FC2), w3 = up_proj
                if "w1" in key or "gate_proj" in key:
                    experts[layer_key][expert_idx]["gate_proj"] = tensor
                elif "w2" in key or "down_proj" in key:
                    experts[layer_key][expert_idx]["down_proj"] = tensor

    if not experts:
        print("ERROR: No MoE expert weights found. Is this a MoE model?")
        sys.exit(1)

    print(f"\nFound {len(experts)} MoE layers")

    # Analyze channel magnitudes per expert.
    result = {
        "model_name": model_name,
        "num_moe_layers": len(experts),
        "layers": {},
    }

    for layer_key in sorted(experts.keys(), key=lambda x: int(x.split("_")[1])):
        layer_experts = experts[layer_key]
        n_experts = len(layer_experts)
        print(f"\n  {layer_key}: {n_experts} experts")

        layer_result = {
            "num_experts": n_experts,
            "experts": {},
        }

        # Per-expert analysis.
        all_fc1_norms = []
        all_fc2_norms = []
        all_alignments = []

        for eid in sorted(layer_experts.keys(), key=int):
            expert = layer_experts[eid]
            expert_result = {}

            if "gate_proj" not in expert or "down_proj" not in expert:
                expert_result["incomplete"] = True
                layer_result["experts"][eid] = expert_result
                continue

            # gate_proj (FC1): [intermediate_size, hidden_size]
            # Row norms = output channel magnitudes.
            fc1 = expert["gate_proj"]
            fc1_norms = np.linalg.norm(fc1, axis=1)

            # down_proj (FC2): [hidden_size, intermediate_size]
            # Column norms = input channel magnitudes.
            fc2 = expert["down_proj"]
            fc2_norms = np.linalg.norm(fc2, axis=0)

            expert_result["fc1_shape"] = list(fc1.shape)
            expert_result["fc2_shape"] = list(fc2.shape)
            expert_result["intermediate_size"] = int(fc1.shape[0])

            # Channel magnitude statistics.
            expert_result["fc1_norms"] = {
                "mean": float(fc1_norms.mean()),
                "std": float(fc1_norms.std()),
                "min": float(fc1_norms.min()),
                "max": float(fc1_norms.max()),
                # Coefficient of variation: how "uneven" the channel usage is.
                # High CV = some channels dominate, others are near-dead.
                "cv": float(fc1_norms.std() / (fc1_norms.mean() + 1e-10)),
            }
            expert_result["fc2_norms"] = {
                "mean": float(fc2_norms.mean()),
                "std": float(fc2_norms.std()),
                "min": float(fc2_norms.min()),
                "max": float(fc2_norms.max()),
                "cv": float(fc2_norms.std() / (fc2_norms.mean() + 1e-10)),
            }

            # FC1↔FC2 alignment — the Nemotron pattern.
            if len(fc1_norms) == len(fc2_norms):
                corr = np.corrcoef(fc1_norms, fc2_norms)[0, 1]
                alignment = float(corr) if not np.isnan(corr) else 0.0
                expert_result["fc1_fc2_alignment"] = alignment
                all_alignments.append(alignment)

                # Dead channels: both FC1 output and FC2 input have tiny norms.
                dead_threshold = 0.01 * fc1_norms.mean()  # 1% of mean norm
                dead_mask = (fc1_norms < dead_threshold) & (fc2_norms < dead_threshold)
                expert_result["dead_channel_fraction"] = float(dead_mask.mean())
                expert_result["dead_channel_count"] = int(dead_mask.sum())

                # "Hot" channels: top 10% by combined FC1+FC2 norm.
                combined_norms = fc1_norms + fc2_norms
                top_10_threshold = np.percentile(combined_norms, 90)
                hot_mask = combined_norms >= top_10_threshold
                expert_result["hot_channel_fraction"] = float(hot_mask.mean())

                # Norm distribution shape: is it bimodal (clear dead vs active)
                # or uniform (all channels roughly equal)?
                # We measure this via the ratio of median to mean.
                # Bimodal: median << mean (many near-zero values pull median down).
                # Uniform: median ≈ mean.
                expert_result["fc1_median_mean_ratio"] = float(
                    np.median(fc1_norms) / (fc1_norms.mean() + 1e-10)
                )

            all_fc1_norms.append(fc1_norms)
            all_fc2_norms.append(fc2_norms)
            layer_result["experts"][eid] = expert_result

        # Cross-expert analysis for this layer.
        if all_alignments:
            layer_result["mean_fc1_fc2_alignment"] = float(np.mean(all_alignments))
            layer_result["std_fc1_fc2_alignment"] = float(np.std(all_alignments))

        # Expert diversity: how different are the channel profiles across experts?
        # Stack all FC1 norm vectors and compute pairwise cosine similarity.
        if len(all_fc1_norms) > 1:
            norms_matrix = np.stack(all_fc1_norms)
            normed = norms_matrix / (
                np.linalg.norm(norms_matrix, axis=1, keepdims=True) + 1e-10
            )
            similarity = normed @ normed.T
            triu_idx = np.triu_indices(len(all_fc1_norms), k=1)
            sims = similarity[triu_idx]
            layer_result["expert_channel_similarity"] = {
                "mean": float(sims.mean()),
                "std": float(sims.std()),
                "min": float(sims.min()),
                "max": float(sims.max()),
            }

            # Are experts using the same channels or different channels?
            # If all experts have high norms on the same channels, they're redundant.
            # If they have high norms on different channels, they're specialized.
            channel_usage_entropy = []
            for ch_idx in range(norms_matrix.shape[1]):
                col = norms_matrix[:, ch_idx]
                col_norm = col / (col.sum() + 1e-10)
                col_norm = col_norm[col_norm > 1e-10]
                if len(col_norm) > 0:
                    entropy = -np.sum(col_norm * np.log(col_norm))
                    channel_usage_entropy.append(entropy)
            if channel_usage_entropy:
                layer_result["channel_usage_entropy"] = {
                    "mean": float(np.mean(channel_usage_entropy)),
                    "std": float(np.std(channel_usage_entropy)),
                }

        result["layers"][layer_key] = layer_result
        print(f"    Mean FC1↔FC2 alignment: {layer_result.get('mean_fc1_fc2_alignment', 'N/A')}")
        if "expert_channel_similarity" in layer_result:
            print(
                f"    Expert channel similarity: {layer_result['expert_channel_similarity']['mean']:.3f}"
            )

    # Global summary.
    all_layer_alignments = [
        v["mean_fc1_fc2_alignment"]
        for v in result["layers"].values()
        if "mean_fc1_fc2_alignment" in v
    ]
    if all_layer_alignments:
        result["global_mean_alignment"] = float(np.mean(all_layer_alignments))
        print(f"\n  Global mean FC1↔FC2 alignment: {result['global_mean_alignment']:.3f}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 33: Channel Magnitude Introspection"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mixtral-8x7B-v0.1",
        help="HuggingFace MoE model ID to analyze",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/channel_profiles",
        help="Output directory for profile JSON",
    )
    args = parser.parse_args()

    result = extract_expert_channel_profiles(args.model)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    safe_name = args.model.replace("/", "_")
    json_path = output_path / f"channel_profile_{safe_name}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nProfile saved to {json_path}")


if __name__ == "__main__":
    main()
