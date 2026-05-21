import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = "unsloth/gemma-4-26B-A4B-it-GGUF"
gguf_file = "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
print("Loading model...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=gguf_file)
    model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file=gguf_file)
    print("Loaded!")
except Exception as e:
    print("Error:", e)
