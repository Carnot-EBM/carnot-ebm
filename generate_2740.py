import json
import os

data = {
  "honest_verdict": "blocked_gguf_qwen36_not_cached",
  "verifier_discriminative": False,
  "energy_values": [0.0]*30,
  "n_non_zero": 0,
  "energy_mean": 0.0,
  "energy_std": 0.0,
  "fover_energies": [0.0]*5,
  "model_loaded": False,
  "model_load_time_s": 0.0,
  "model_specs": {
    "name": "Qwen3.6-35B-A3B-GGUF",
    "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
    "role": "live_verification_diagnostic",
    "quantization": "Q4_K_M"
  },
  "random_seed": 42,
  "reproducibility_checksum": "",
  "cuda_available": True,
  "duration_s": 0.0,
  "preconditions_checked": [
    {
      "resource": "cuda",
      "available": True,
      "check": "NVIDIA GeForce RTX 3090"
    },
    {
      "resource": "gguf_qwen36",
      "available": False,
      "check": "ls ~/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF/"
    },
    {
      "resource": "carnot_importable",
      "available": True,
      "check": ".venv/bin/python -c 'import carnot.pipeline'"
    },
    {
      "resource": "fover_corpus",
      "available": True,
      "check": "8829 lines"
    }
  ]
}

os.makedirs('results', exist_ok=True)
with open('results/experiment_2740_verifier_energy_debug_v2_live_gpu.json', 'w') as f:
    json.dump(data, f, indent=2)

print("Created results/experiment_2740_verifier_energy_debug_v2_live_gpu.json")
