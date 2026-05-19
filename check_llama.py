from llama_cpp import Llama

llm = Llama(
    model_path="/home/ianblenke/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-GGUF/snapshots/3365c68df1a83799b846d05324ebfadbb8cc70b3/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
    n_ctx=128,
    verbose=False
)
res = llm("Hello", max_tokens=10, logprobs=1)
print(res["choices"][0]["logprobs"].keys())
