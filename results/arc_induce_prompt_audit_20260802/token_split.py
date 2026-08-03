#!/usr/bin/env python3
"""Where do the prompt's tokens actually GO? Instructions / static layout grid / evidence.

Reading sc25 by hand showed the transitions block is dominated by ONE 64-row run-length grid,
with the eleven observed deltas costing a few characters each. This splits every dumped
`as_sent_all.txt` into three spans and tokenizes each with the LIVE generator's own vocabulary:

  header    -- everything before "OBSERVED TRANSITIONS:" (the code-only directive, the encoding
               explanation, the interface contract). Fixed cost, identical on every game bar
               the grid dimensions and colour list.
  layout    -- the "INITIAL GRID ..." block: ONE full static grid. Carries geometry, not dynamics.
  evidence  -- the "--- ACTIONn ... changed cells" lines. This is the ONLY part of the prompt
               that says what an action DOES, i.e. the only part that can teach dynamics.

vocab_only tokenizer: no weights, no GPU, no server.
"""

import json
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
DUMP = HERE / "out" / "prompts"
GGUF = (
    "/home/ianblenke/.cache/huggingface/hub/models--unsloth--gemma-4-31B-it-qat-GGUF/"
    "snapshots/43cc1aeb31adf47ec06a854507ce552cd9862e6f/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf"
)

from llama_cpp import Llama  # noqa: E402

llm = Llama(model_path=GGUF, vocab_only=True, verbose=False)
ntok = lambda s: len(llm.tokenize(s.encode(), add_bos=False, special=False)) if s else 0  # noqa: E731

rows = []
for d in sorted(DUMP.iterdir()):
    f = d / "as_sent_all.txt"
    if not f.exists():
        continue
    txt = f.read_text()
    marker = "OBSERVED TRANSITIONS:\n"
    i = txt.find(marker)
    if i < 0:
        continue
    header, body = txt[: i + len(marker)], txt[i + len(marker) :]
    lines = body.splitlines(keepends=True)
    layout, evidence, tail = [], [], []
    mode = "layout"
    for ln in lines:
        if ln.startswith("--- ACTION"):
            mode = "evidence"
        elif mode == "evidence" and not ln.startswith("--- ACTION"):
            # trailing instructions after the last delta line
            mode = "tail"
        (layout if mode == "layout" else evidence if mode == "evidence" else tail).append(ln)
    th, tl, te, tt = (
        ntok(header),
        ntok("".join(layout)),
        ntok("".join(evidence)),
        ntok("".join(tail)),
    )
    tot = th + tl + te + tt
    rows.append(
        {
            "game": d.name,
            "tok_header": th,
            "tok_layout_grid": tl,
            "tok_evidence_deltas": te,
            "tok_tail": tt,
            "tok_total": tot,
            "pct_layout": round(100 * tl / tot, 1),
            "pct_evidence": round(100 * te / tot, 1),
            "n_delta_lines": sum(1 for ln in evidence if ln.startswith("--- ACTION")),
        }
    )


def q(vals):
    v = sorted(vals)
    n = len(v)
    pct = lambda p: v[min(n - 1, max(0, int(round(p * (n - 1)))))]  # noqa: E731
    return {
        "min": v[0],
        "q1": pct(0.25),
        "median": median(v),
        "q3": pct(0.75),
        "max": v[-1],
        "n": n,
    }


out = {
    "n_games": len(rows),
    "pct_of_prompt_that_is_the_static_layout_grid": q([r["pct_layout"] for r in rows]),
    "pct_of_prompt_that_is_transition_evidence": q([r["pct_evidence"] for r in rows]),
    "tok_layout_grid": q([r["tok_layout_grid"] for r in rows]),
    "tok_evidence_deltas": q([r["tok_evidence_deltas"] for r in rows]),
    "tok_header": q([r["tok_header"] for r in rows]),
    "n_games_layout_exceeds_evidence": sum(
        1 for r in rows if r["tok_layout_grid"] > r["tok_evidence_deltas"]
    ),
    "per_game": rows,
}
(HERE / "out" / "token_split.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "per_game"}, indent=1))
