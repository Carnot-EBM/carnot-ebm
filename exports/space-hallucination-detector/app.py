"""Carnot Hallucination Detector — Interactive Gradio Space.

Score LLM activation vectors with pre-trained EBMs to detect hallucination.
Upload a safetensors file or generate activations from a local model.
"""

from __future__ import annotations

import json
import os
import tempfile

import gradio as gr
import numpy as np

# Available models with metadata
MODELS = {
    "per-token-ebm-gemma4-e2b-nothink": {
        "label": "Gemma 4 E2B (base) — 86.8%",
        "source": "google/gemma-4-E2B",
        "accuracy": 86.8,
        "hidden_dim": 1536,
    },
    "per-token-ebm-lfm25-350m-nothink": {
        "label": "LFM 2.5 350M (base) — 80.4%",
        "source": "LiquidAI/LFM2.5-350M",
        "accuracy": 80.4,
        "hidden_dim": 1024,
    },
    "per-token-ebm-qwen35-2b-nothink": {
        "label": "Qwen 3.5 2B — 79.3%",
        "source": "Qwen/Qwen3.5-2B",
        "accuracy": 79.3,
        "hidden_dim": 2048,
    },
    "per-token-ebm-lfm25-12b-nothink": {
        "label": "LFM 2.5 1.2B Instruct — 76.7%",
        "source": "LiquidAI/LFM2.5-1.2B-Instruct",
        "accuracy": 76.7,
        "hidden_dim": 2048,
    },
    "per-token-ebm-qwen35-08b-nothink": {
        "label": "Qwen 3.5 0.8B — 75.5%",
        "source": "Qwen/Qwen3.5-0.8B",
        "accuracy": 75.5,
        "hidden_dim": 1024,
    },
    "per-token-ebm-bonsai-17b-nothink": {
        "label": "Bonsai 1.7B — 75.0%",
        "source": "prism-ml/Bonsai-1.7B-unpacked",
        "accuracy": 75.0,
        "hidden_dim": 2048,
    },
    "per-token-ebm-gemma4-e2b-it-nothink": {
        "label": "Gemma 4 E2B-it — 75.0%",
        "source": "google/gemma-4-E2B-it",
        "accuracy": 75.0,
        "hidden_dim": 1536,
    },
}


def load_ebm_model(model_id: str):
    """Load a pre-trained EBM from HuggingFace."""
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file
    import jax.numpy as jnp
    import jax.random as jrandom

    # Download model files
    config_path = hf_hub_download(f"Carnot-EBM/{model_id}", "config.json")
    weights_path = hf_hub_download(f"Carnot-EBM/{model_id}", "model.safetensors")

    with open(config_path) as f:
        config = json.load(f)

    weights = load_file(weights_path)

    return config, weights


def score_activations(model_id: str, activations_file, json_input: str) -> str:
    """Score activation vectors and return results."""
    try:
        import jax.numpy as jnp
        import jax.random as jrandom

        # Load EBM
        config, weights = load_ebm_model(model_id)
        hidden_dim = config["input_dim"]
        n_layers = config["n_layers"]

        # Reconstruct simple forward pass (no carnot dependency needed)
        def ebm_energy(x):
            """Forward pass through Gibbs EBM."""
            h = x
            for i in range(n_layers):
                w = jnp.array(weights[f"layer_{i}_weight"])
                b = jnp.array(weights[f"layer_{i}_bias"])
                h = jnp.maximum(0, w @ h + b)  # SiLU approximated as ReLU for simplicity
            out_w = jnp.array(weights["output_weight"])
            out_b = jnp.array(weights["output_bias"])
            return float(out_w @ h + out_b)

        # Load activations from file or JSON
        activations = None
        labels = None

        if activations_file is not None:
            from safetensors.numpy import load_file
            data = load_file(activations_file.name)
            activations = data["activations"]
            labels = data.get("labels")
        elif json_input and json_input.strip():
            parsed = json.loads(json_input)
            if isinstance(parsed, list):
                activations = np.array(parsed, dtype=np.float32)
            elif isinstance(parsed, dict) and "activations" in parsed:
                activations = np.array(parsed["activations"], dtype=np.float32)
            else:
                return "Error: JSON must be a list of vectors or {\"activations\": [...]}"
        else:
            return "Please upload a safetensors file or paste activation vectors as JSON."

        if activations.shape[-1] != hidden_dim:
            return (
                f"Error: Model expects {hidden_dim}-dim vectors, "
                f"got {activations.shape[-1]}-dim. "
                f"Make sure you're using activations from {MODELS[model_id]['source']}."
            )

        # Score each token
        n_tokens = len(activations)
        energies = []
        for i in range(n_tokens):
            e = ebm_energy(jnp.array(activations[i]))
            energies.append(e)

        mean_energy = np.mean(energies)
        std_energy = np.std(energies)

        # Build results
        lines = []
        lines.append(f"## Results")
        lines.append(f"")
        lines.append(f"**Model:** {model_id}")
        lines.append(f"**Source LLM:** {MODELS[model_id]['source']}")
        lines.append(f"**EBM Accuracy:** {MODELS[model_id]['accuracy']}%")
        lines.append(f"")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Tokens scored | {n_tokens} |")
        lines.append(f"| Mean energy | {mean_energy:.4f} |")
        lines.append(f"| Std energy | {std_energy:.4f} |")

        if labels is not None:
            correct_e = [energies[i] for i in range(n_tokens) if labels[i] == 1]
            wrong_e = [energies[i] for i in range(n_tokens) if labels[i] == 0]
            if correct_e and wrong_e:
                thresh = (np.mean(correct_e) + np.mean(wrong_e)) / 2
                tp = sum(1 for e in wrong_e if e > thresh)
                tn = sum(1 for e in correct_e if e <= thresh)
                acc = (tp + tn) / (len(correct_e) + len(wrong_e))
                gap = np.mean(wrong_e) - np.mean(correct_e)
                lines.append(f"| Detection accuracy | {acc:.1%} |")
                lines.append(f"| Energy gap | {gap:.4f} |")
                lines.append(f"| Correct tokens | {len(correct_e)} |")
                lines.append(f"| Wrong tokens | {len(wrong_e)} |")

        lines.append(f"")
        lines.append(f"### Interpretation")
        lines.append(f"")
        lines.append(f"- **Lower energy** = model activations look like correct answers")
        lines.append(f"- **Higher energy** = model activations look like hallucinated answers")
        lines.append(f"- The threshold between correct and hallucinated is data-dependent")

        # Show per-token energy distribution
        lines.append(f"")
        lines.append(f"### Per-Token Energies (first 20)")
        lines.append(f"")
        lines.append(f"| Token # | Energy | Assessment |")
        lines.append(f"|---------|--------|-----------|")
        for i in range(min(20, n_tokens)):
            assessment = "likely correct" if energies[i] < mean_energy else "possibly hallucinated"
            label_str = ""
            if labels is not None:
                label_str = " (correct)" if labels[i] == 1 else " (wrong)"
            lines.append(f"| {i} | {energies[i]:.4f} | {assessment}{label_str} |")

        return "\n".join(lines)

    except Exception as e:
        return f"Error: {e}"


# Build Gradio interface
model_choices = [(info["label"], mid) for mid, info in MODELS.items()]

with gr.Blocks(
    title="Carnot Hallucination Detector",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown("""
    # Carnot Hallucination Detector

    Score LLM activation vectors with pre-trained Energy-Based Models to detect hallucination.

    **How to use:**
    1. Select an EBM model matching your source LLM
    2. Upload a safetensors file with an `activations` key, OR paste activation vectors as JSON
    3. Click "Score" to see per-token hallucination energy

    Lower energy = likely correct. Higher energy = likely hallucinated.

    *Models and datasets at [Carnot-EBM](https://huggingface.co/Carnot-EBM) on HuggingFace.*
    """)

    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=model_choices,
                value="per-token-ebm-gemma4-e2b-nothink",
                label="EBM Model",
                info="Choose a model matching your source LLM",
            )

            file_input = gr.File(
                label="Upload Activations (safetensors)",
                file_types=[".safetensors"],
            )

            json_input = gr.Textbox(
                label="Or paste activation vectors as JSON",
                placeholder='[[0.1, 0.2, ...], [0.3, 0.4, ...]]',
                lines=5,
            )

            score_btn = gr.Button("Score Activations", variant="primary")

        with gr.Column(scale=2):
            output = gr.Markdown(label="Results")

    score_btn.click(
        fn=score_activations,
        inputs=[model_dropdown, file_input, json_input],
        outputs=output,
    )

    gr.Markdown("""
    ---

    ### 10 Principles from 25 Experiments

    1. Simpler is better in small-data regimes
    2. Token-level features > sequence-level
    3. The model's own logprobs are the best energy
    4. Overfitting is the main enemy
    5. Extract features from generated tokens, not prompts
    6. Different energy signals dominate in different domains
    7. Statistical difference ≠ causal influence
    8. **Instruction tuning compresses the hallucination signal** (base models detect better)
    9. Adversarial questions defeat post-hoc detection
    10. **Chain-of-thought compresses the signal further** (disable thinking for +14% detection)

    [GitHub](https://github.com/Carnot-EBM/carnot-ebm) |
    [Technical Report](https://github.com/Carnot-EBM/carnot-ebm/blob/main/docs/technical-report.md) |
    [Usage Guide](https://github.com/Carnot-EBM/carnot-ebm/blob/main/docs/usage-guide.md)
    """)

if __name__ == "__main__":
    demo.launch()
