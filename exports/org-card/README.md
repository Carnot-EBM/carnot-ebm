# Carnot-EBM

**Energy-Based Models for LLM Hallucination Detection**

Carnot is an open-source framework that trains lightweight EBM classifiers on frozen LLM activations to detect hallucinations. No fine-tuning required — the target LLM's weights are never modified.

## How It Works

1. Run a frozen LLM on a question and extract per-token hidden states from the last layer
2. Score each token's activation vector through a trained Gibbs EBM
3. Low energy = likely correct, high energy = likely hallucination

The EBM is a small MLP ([hidden_dim → 256 → 64 → 1]) trained via Noise Contrastive Estimation on correct vs incorrect answer activations. Training takes ~30 seconds on a single GPU.

## Pre-Trained Models

Each model is trained on 200 TruthfulQA questions and works only with activations from the matching source LLM.

| Model | Source LLM | EBM Accuracy | Hidden Dim |
|-------|-----------|-------------|-----------|
| [per-token-ebm-gemma4-e2b-nothink](Carnot-EBM/per-token-ebm-gemma4-e2b-nothink) | Gemma 4 E2B (base) | **86.8%** | 1536 |
| [per-token-ebm-lfm25-350m-nothink](Carnot-EBM/per-token-ebm-lfm25-350m-nothink) | LFM 2.5 350M (base) | **80.4%** | 1024 |
| [per-token-ebm-qwen35-2b-nothink](Carnot-EBM/per-token-ebm-qwen35-2b-nothink) | Qwen 3.5 2B | **79.3%** | 2048 |
| [per-token-ebm-lfm25-12b-nothink](Carnot-EBM/per-token-ebm-lfm25-12b-nothink) | LFM 2.5 1.2B Instruct | **76.7%** | 2048 |
| [per-token-ebm-bonsai-17b-nothink](Carnot-EBM/per-token-ebm-bonsai-17b-nothink) | Bonsai 1.7B | **75.0%** | 2048 |
| [per-token-ebm-gemma4-e2b-it-nothink](Carnot-EBM/per-token-ebm-gemma4-e2b-it-nothink) | Gemma 4 E2B-it | **75.0%** | 1536 |
| [per-token-ebm-qwen35-08b-nothink](Carnot-EBM/per-token-ebm-qwen35-08b-nothink) | Qwen 3.5 0.8B | **75.5%** | 1024 |

*More models (4B, 9B, 20B, 27B, 35B) are being trained and will appear here as they complete.*

## Datasets

| Dataset | Description |
|---------|-------------|
| [token-activations](Carnot-EBM/token-activations) | Per-token activation vectors with correct/wrong labels from multiple LLMs |

## Quick Start

```python
from carnot.inference.ebm_loader import load_ebm

# Downloads from HuggingFace automatically
ebm = load_ebm("per-token-ebm-gemma4-e2b-nothink")

# Score an activation vector (from Gemma 4 E2B hidden states)
energy = float(ebm.energy(activation_vector))
# Low energy = likely correct, high energy = likely hallucination
```

Or via CLI:
```bash
pip install carnot
carnot score --list-models
carnot score --model per-token-ebm-gemma4-e2b-nothink --activations-file my_data.safetensors
```

## 10 Principles Learned

From 25 experiments across multiple model families:

1. **Simpler is better** in small-data regimes
2. **Token-level features > sequence-level** — mean-pooling kills signal
3. **The model's own logprobs are the best energy** — no external EBM needed for rejection sampling
4. **Overfitting is the main enemy** when examples < dimensions
5. **Extract features from generated tokens**, not prompts
6. **Different energy signals dominate in different domains** — logprobs for QA, structural tests for code
7. **Statistical difference ≠ causal influence** — detection ≠ steering
8. **Instruction tuning compresses the hallucination signal** — base models detect better (86.8% vs 75.0%)
9. **Adversarial questions defeat post-hoc detection** — must move upstream
10. **Chain-of-thought compresses the signal further** — disable thinking for +14% detection

## Links

- [GitHub Repository](https://github.com/ianblenke/carnot)
- [Technical Report](https://github.com/ianblenke/carnot/blob/main/docs/technical-report.md)
- [Usage Guide](https://github.com/ianblenke/carnot/blob/main/docs/usage-guide.md)
- [Interactive Demo](https://huggingface.co/spaces/Carnot-EBM/hallucination-detector) *(coming soon)*

## License

Apache 2.0
