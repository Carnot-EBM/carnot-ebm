
# MODEL SELECTION - MANDATORY for any live-data or verify-repair experiment:
# Always try `cached_sota_pair()` first.
from carnot.inference.sota_models import cached_sota_pair
MODEL_SPECS = [
    {"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"},
    {"hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF"},
    {"hf_id": "unsloth/gemma-4-31B-it-GGUF"},
]
# Record `models_used` in every artifact with the exact hub IDs.
