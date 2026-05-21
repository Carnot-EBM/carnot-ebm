import llama_cpp
import time
start = time.time()
try:
    path = '/home/ianblenke/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-GGUF/snapshots/04028bd1aa552ebf46a986375418cb92ffeae774/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf'
    llm = llama_cpp.Llama(model_path=path, embedding=True, verbose=False)
    emb = llm.create_embedding("Hello, how are you?")
    print('Embedding shape:', len(emb['data'][0]['embedding']))
except Exception as e:
    print('Failed:', e)
print('Elapsed:', time.time() - start)