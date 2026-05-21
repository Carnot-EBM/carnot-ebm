from llama_cpp import Llama
import time
model_path = "/home/ianblenke/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-GGUF/snapshots/3365c68df1a83799b846d05324ebfadbb8cc70b3/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
for threads in [4, 12, 24, 32]:
    print(f"Testing with n_threads={threads}")
    llm = Llama(model_path=model_path, n_ctx=512, n_threads=threads, n_gpu_layers=-1, verbose=False, logits_all=True)
    prompt = "What happened in Tiananmen Square in June 1989?"
    t0 = time.time()
    out = llm(prompt, max_tokens=10, logprobs=1)
    t1 = time.time()
    print(f"10 tokens took {t1-t0} seconds\n")
