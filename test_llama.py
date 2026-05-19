from llama_cpp import Llama
import numpy as np

model_path = "/home/ianblenke/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-GGUF/snapshots/3365c68df1a83799b846d05324ebfadbb8cc70b3/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
llm = Llama(model_path=model_path, n_ctx=512, verbose=False)
output = llm("What is the capital of France?", max_tokens=10, logprobs=1)
logprobs = output["choices"][0]["logprobs"]["token_logprobs"]
print(f"Logprobs: {logprobs}")
energy = -np.mean(logprobs)
print(f"Energy: {energy}")
