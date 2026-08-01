#!/usr/bin/env python3
"""Re-score ONE frozen A/B cell under every mask arm, in a KILLABLE process.

WHY A SUBPROCESS, restating the reason the A/B's own worker records so it is not lost when this
harness is read alone: this executes LLM-written code. At least one non-terminating induced
engine exists in this corpus (it wedged a generation loop for 13 minutes on 2026-07-31), and an
in-process alarm would be SWALLOWED -- `WorldModelVerifier.score` wraps `engine(...)` in
`except Exception`, so a SIGALRM-raised exception is caught and recorded as an ordinary
per-transition failure. A hang would silently become a CLEAN ZERO, which is worse than the hang
because it is invisible. Only an external kill is sound.

A timed-out cell is UNDETERMINED, never 0.0: it leaves both numerator and denominator.
"""

from __future__ import annotations

import ast
import json
import os
import pathlib
import pickle
import sys

REPO = pathlib.Path(os.environ["CARNOT_REPO"])
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CARNOT_ARC_E3_DIR", "/tmp/arc_hudms/e3")
os.makedirs(os.environ["CARNOT_ARC_E3_DIR"], exist_ok=True)
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def reads_data_param(src: str) -> bool:
    """Does `engine`'s body ever READ its third parameter? AST, never a substring scan.

    Copied deliberately from the A/B's `rescore_worker.py` rather than re-invented: the word
    `data` appears in the prose docstring of nearly every induced engine, so `"data" in src`
    would call a coordinate-BLIND engine aware. Kept here so the blindness column in this
    artifact means the same thing it means in the A/B's.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name == "engine":
            if len(fn.args.args) < 3:
                return False
            nm = fn.args.args[2].arg
            return any(
                isinstance(n, ast.Name) and n.id == nm and isinstance(n.ctx, ast.Load)
                for n in ast.walk(fn)
            )
    return False


def main() -> int:
    job = json.loads(pathlib.Path(sys.argv[1]).read_text())
    out: dict = {"status": "ok", "cell": job["cell"], "game": job["game"]}

    import numpy as np  # noqa: F401  (engines are exec'd with `np` in scope)
    from hud_masks import masks_for
    from score_arms import score_all_arms

    with open(job["window_pkl"], "rb") as fh:
        win = pickle.load(fh)
    shown, held = list(win["shown"]), list(win["held"])
    # The guard's corpus is the WHOLE window. `shown` is what the model was given and `held` is
    # what it is graded on; whether a mask covers the GAME is a property of the game, so both
    # halves are evidence about it. See score_arms.py for why the tail alone is not enough.
    full_corpus = shown + held

    code_path = pathlib.Path(job["code_path"])
    if not code_path.exists():
        out["status"] = "no_engine_file"
        print(json.dumps(out))
        return 0
    src = code_path.read_text()
    out["reads_data_param"] = reads_data_param(src)

    ns: dict = {"np": np, "numpy": np}
    try:
        exec(compile(src, str(code_path), "exec"), ns)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        out["status"] = f"unrunnable:{type(exc).__name__}"
        out["error"] = str(exc)[:200]
        print(json.dumps(out))
        return 0
    engine = ns.get("engine")
    if not callable(engine):
        out["status"] = "no_engine_symbol"
        print(json.dumps(out))
        return 0

    m = masks_for(job["game"])
    out["mask_meta"] = m["meta"]
    out.update(score_all_arms(engine, held, full_corpus, m))
    out["n_graded_transitions"] = len(held)
    out["n_full_corpus_transitions"] = len(full_corpus)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
