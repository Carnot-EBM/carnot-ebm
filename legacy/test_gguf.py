import json
from llama_cpp import Llama

model_path = "/home/ianblenke/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-GGUF/snapshots/3365c68df1a83799b846d05324ebfadbb8cc70b3/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"

print("Loading model...")
llm = Llama(model_path=model_path, n_ctx=512, verbose=False, logits_all=True)
print("Generating...")
output = llm("What is 1+1?", max_tokens=10, logprobs=1)
print(json.dumps(output, indent=2))
