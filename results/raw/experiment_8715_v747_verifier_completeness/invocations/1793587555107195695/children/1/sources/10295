import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

from llama_cpp import Llama
from carnot.inference.sota_models import SOTA_GGUF_MODELS, resolve_cached_gguf

def test():
    model_path = None
    for spec in SOTA_GGUF_MODELS:
        path = resolve_cached_gguf(spec["hf_id"])
        if path:
            model_path = path
            break

    print(f"Model path: {model_path}")
    llm = Llama(model_path=model_path, n_ctx=128, verbose=False, n_gpu_layers=-1, logits_all=True)
    result = llm("What is 1+1?", max_tokens=10, logprobs=5)
    print(result['choices'][0]['logprobs'].keys())
    print(result['choices'][0]['logprobs']['top_logprobs'][0])

if __name__ == "__main__":
    test()
