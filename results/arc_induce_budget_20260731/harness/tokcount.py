"""Token-count the ft09 evidence through the GENERATOR'S OWN tokenizer.

Why this and not an estimate: the whole Phase-1 question is whether a 4096-token
completion budget was the binding constraint on ft09's round-1 induce. A character
or line count cannot answer that; only the model's own vocabulary can. Loaded via
`vocab_only=True` off the .gguf PATH (never AutoTokenizer on a GGUF repo id --
CLAUDE.md GGUF tokenizer rule), so this costs no VRAM and no GPU.
"""
import sys
from llama_cpp import Llama

GGUF = "/home/ianblenke/.cache/huggingface/hub/models--unsloth--gemma-4-31B-it-GGUF/snapshots/f130ba51393346288f5862e30e9586b9b021513f/gemma-4-31B-it-Q4_K_M.gguf"
llm = Llama(model_path=GGUF, vocab_only=True, verbose=False)

for path in sys.argv[1:]:
    txt = open(path).read()
    toks = llm.tokenize(txt.encode(), add_bos=False, special=False)
    print(f"{len(toks):>7} tokens  {len(txt):>8} chars  {txt.count(chr(10)):>6} lines  {path}")
