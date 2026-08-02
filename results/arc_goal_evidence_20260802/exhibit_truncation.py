#!/usr/bin/env python3
"""Capture the RAW completion the treatment prompt produces, so the failure is readable.

WHY THIS EXISTS AND WHAT IT IS NOT. `generate()` returns only a message on failure -- "local
model code unusable after 3 tries (syntax error line N: ...)" -- and throws the text away. A
one-line error is enough to SCORE a cell and not enough to UNDERSTAND it, and "the model
rambles past its budget" is a mechanism claim that should be shown rather than asserted.

THIS IS NOT PART OF THE MEASUREMENT. It runs AFTER the scored run, contributes no cell, enters
no rate and no test, and its output is labelled as an exhibit. It calls the server directly
rather than through `generate()` precisely so the raw text survives; that means it is NOT a
faithful replay of a scored cell either (no retry loop, no code extraction), and it must not be
quoted as one. Its only claim is "here is what the model writes when it is shown the
transitions", against the control prompt for contrast.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRATCH = Path(
    "/tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/"
    "87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad/goalev"
)
os.environ["CARNOT_ARC_E3_DIR"] = str(SCRATCH / "exhibit_e3")
sys.path.insert(0, str(ROOT / "python"))

PORT = int(os.environ.get("GEV_PORT", "41871"))
GAME = os.environ.get("GEV_EXHIBIT_GAME", "ar25")


def complete(prompt: str, seed: int, n_predict: int = 4096) -> dict:
    body = json.dumps(
        {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": 0.3,
            "seed": seed,
            "cache_prompt": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/completion",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read().decode())


def main() -> int:
    from carnot.agentic import arc_executable_world_model as e3

    windows = pickle.loads((SCRATCH / "windows.pkl").read_bytes())
    shown, _held, _cell = windows[GAME]
    prop = e3.LocalGGUFProposer(port=PORT)

    out: dict = {
        "what_this_is": "an EXHIBIT, not a measurement. No cell, no rate, no test uses it.",
        "game": GAME,
        "server_port": PORT,
        "arms": {},
    }
    for tag, flag in (("control_no_transitions", False), ("treatment_with_transitions", True)):
        if flag:
            os.environ["CARNOT_ARC_GOAL_PROMPT_TRANSITIONS"] = "1"
        else:
            os.environ.pop("CARNOT_ARC_GOAL_PROMPT_TRANSITIONS", None)
        prompt = prop._goal_only_prompt(GAME, None, shown)  # noqa: SLF001
        res = complete(prompt, seed=8300)
        text = res.get("content", "")
        (HERE / "out" / f"exhibit_{tag}.txt").write_text(text)
        out["arms"][tag] = {
            "prompt_chars": len(prompt),
            "tokens_predicted": res.get("tokens_predicted"),
            "tokens_evaluated": res.get("tokens_evaluated"),
            "stopped_eos": res.get("stopped_eos"),
            # THE WHOLE POINT. `stopped_limit` true means the model hit n_predict and was cut
            # off mid-sentence -- which is how a code block ends up unclosed and unparseable.
            "stopped_limit": res.get("stopped_limit"),
            "closed_code_fence": text.count("```") >= 2,
            "chars": len(text),
            "saved_to": f"results/arc_goal_evidence_20260802/out/exhibit_{tag}.txt",
        }
    os.environ.pop("CARNOT_ARC_GOAL_PROMPT_TRANSITIONS", None)
    (HERE / "out" / "exhibit_truncation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
