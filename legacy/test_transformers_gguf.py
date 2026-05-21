import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = "unsloth/Qwen3.6-35B-A3B-GGUF"
gguf_file = "Qwen3.6-35B-A3B-Q4_K_M.gguf"
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=gguf_file)
print("Tokenizer loaded.")
