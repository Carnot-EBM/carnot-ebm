"""Minimal usage example for the guided-decoding-adapter.

Run from the carnot repo root:
    JAX_PLATFORMS=cpu python exports/guided-decoding-adapter/example.py
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from unittest.mock import MagicMock
import torch

from carnot.inference.guided_decoding import GuidedDecoder

# Load adapter from this directory (local usage)
# To load from HuggingFace Hub swap the path for the repo ID:
#   decoder = GuidedDecoder.from_pretrained("Carnot-EBM/guided-decoding-adapter")
decoder = GuidedDecoder.from_pretrained("exports/guided-decoding-adapter")

# --- Minimal mock model and tokenizer (no GPU / model download needed) ---
# Replace these two blocks with real HF model/tokenizer for production use:
#   model     = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B")
#   tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
step = [0]

def _forward(input_ids):
    logits = torch.zeros(1, input_ids.shape[1], 10)
    logits[0, -1, 1 if step[0] >= 3 else 0] = 10.0  # EOS after 3 tokens
    step[0] += 1
    out = MagicMock(); out.logits = logits; return out

model = MagicMock()
model.side_effect = _forward
model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))

tokenizer = MagicMock()
tokenizer.eos_token_id = 1
tokenizer.encode = MagicMock(return_value=torch.tensor([[2, 3, 4]]))
tokenizer.decode = MagicMock(side_effect=lambda ids, **kw: "" if ids.item() == 1 else "A")

result = decoder.generate(model, tokenizer, "What is 47 + 28?")
print("Generated:", result.text)
print(f"Tokens: {result.tokens_generated}  Checks: {result.energy_checks}  "
      f"Mean penalty: {result.mean_penalty:.3f}  Latency: {result.latency_seconds*1000:.1f}ms")
