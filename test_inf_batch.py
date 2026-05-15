import llama_cpp
import time
start = time.time()
path = '/home/ianblenke/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-GGUF/snapshots/04028bd1aa552ebf46a986375418cb92ffeae774/gemma-4-26B-A4B-it-UD-IQ2_XXS.gguf'
llm = llama_cpp.Llama(model_path=path, embedding=True, verbose=False, n_threads=16, n_batch=64)
print('Load time:', time.time() - start)
start = time.time()
emb = llm.create_embedding(["Hello", "How are you?"])
print('Inf time batch 2:', time.time() - start)
