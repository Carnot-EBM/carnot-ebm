"""QUALITY grading for the f16-vs-q8_0 KV-cache arm.

"q8_0 KV is near-lossless" is a claim this project INHERITED from its retired 9B stack. It has
never been checked for a 31B, and a KV-quantization error compounds over a long generation, so
the claim is exactly the kind of thing that can be true at 9B and false at 31B. This grades the
ACTUAL artifact the induce path exists to produce -- a world_model.py -- rather than eyeballing
text or computing a perplexity that nobody consumes.

Metrics mirror the ones the codebase's own generator head-to-head used
(arc_executable_world_model.py's GENERATOR SWITCH block):

  induce_ok     -- the emitted code parses, imports, and defines BOTH engine() and
                   is_level_complete(). This was the DOMINANT driver in that head-to-head
                   (the 27B failed to emit an importable file on 18 of 39 attempts), so it
                   is reported first and separately.
  heldout       -- per-transition cell accuracy of engine() against transitions the prompt
                   NEVER SHOWED (indices k.. of the collected 25; the prompt shows k=8).
  fail_as_zero  -- mean heldout over ALL games, scoring a non-importable engine as 0.0. The
                   HONEST column: a survivor-only mean is survivorship-biased upward exactly
                   when a config degrades loadability, which is the failure mode we are
                   testing for.
  survivor_mean -- mean over importable engines only. Reported alongside, never alone.
  nonzero_cells -- games whose engine actually CHANGED at least one cell. An engine that
                   returns the grid untouched scores well on mostly-static ARC frames while
                   having induced nothing; without this column a degenerate identity engine
                   is indistinguishable from a good one.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import traceback

import numpy as np

SCRATCH = os.path.dirname(os.path.abspath(__file__))
K_SHOWN = 8  # induce_prompt(k=8): transitions 0..7 are shown, 8.. are HELD OUT


def extract_code(text: str) -> str:
    """Pull the python block out of the completion. Models fence it or emit it bare.

    The chat-endpoint arms wrap the model's separate reasoning channel in <think>...</think>
    ahead of the real answer (the production folding). Everything before </think> is the model
    thinking out loud and routinely contains fenced pseudo-code, so grading it would score the
    scratchpad rather than the deliverable -- drop it first.
    """
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    if "```" in text:
        parts, best = text.split("```"), ""
        for i in range(1, len(parts), 2):
            blk = parts[i]
            if blk.startswith("python"):
                blk = blk[len("python"):]
            if len(blk) > len(best):
                best = blk
        if best.strip():
            return best
    return text


def load_engine(code: str):
    """Import the emitted module in a temp file. Any failure is induce_ok=False, with the
    reason RECORDED -- a silent except would hide the very thing being measured."""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "wm.py")
        open(path, "w").write(code)
        spec = importlib.util.spec_from_file_location("wm_under_test", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        eng = getattr(mod, "engine", None)
        ilc = getattr(mod, "is_level_complete", None)
        if not callable(eng) or not callable(ilc):
            raise AttributeError(
                f"missing engine={callable(eng)} is_level_complete={callable(ilc)}"
            )
        return eng, ilc


def grade_game(code: str, npz_path: str) -> dict:
    out = {"induce_ok": False, "heldout_cell_acc": 0.0, "heldout_exact": 0.0,
           "n_heldout": 0, "changed_any_cell": False}
    try:
        eng, _ = load_engine(code)
    except Exception as e:
        out["import_error"] = f"{type(e).__name__}: {e}"
        out["traceback_tail"] = traceback.format_exc()[-300:]
        return out
    out["induce_ok"] = True

    d = np.load(npz_path)
    grids, next_grids, actions = d["grids"], d["next_grids"], d["actions"]
    accs, exact, changed, errs = [], 0, False, 0
    for i in range(K_SHOWN, len(grids)):
        g, nx, a = grids[i], next_grids[i], int(actions[i])
        try:
            pred = np.asarray(eng(g.copy(), a, None))
        except Exception:
            errs += 1
            accs.append(0.0)   # an engine that RAISES predicts nothing -- score it 0, not skip
            continue
        if pred.shape != nx.shape:
            accs.append(0.0)
            continue
        accs.append(float((pred == nx).mean()))
        exact += int(np.array_equal(pred, nx))
        if not np.array_equal(pred, g):
            changed = True
    out["n_heldout"] = len(accs)
    out["engine_raised_on"] = errs
    out["heldout_cell_acc"] = round(float(np.mean(accs)), 4) if accs else 0.0
    out["heldout_exact"] = round(exact / len(accs), 4) if accs else 0.0
    out["changed_any_cell"] = changed
    return out


def grade_config(tag: str) -> dict:
    gen_dir = os.path.join(SCRATCH, "gen", tag)
    if not os.path.isdir(gen_dir):
        return {"tag": tag, "error": "no generations"}
    per = {}
    for fn in sorted(os.listdir(gen_dir)):
        game = fn[:-4]
        raw = open(os.path.join(gen_dir, fn)).read()
        npz = os.path.join(SCRATCH, f"trans_{game}.npz")
        if not os.path.exists(npz):
            continue
        r = grade_game(extract_code(raw), npz)
        r["gen_chars"] = len(raw)
        r["gen_sha256"] = hashlib.sha256(raw.encode()).hexdigest()[:16]
        per[game] = r
    n = len(per)
    ok = [g for g, r in per.items() if r["induce_ok"]]
    accs_all = [r["heldout_cell_acc"] for r in per.values()]
    accs_ok = [per[g]["heldout_cell_acc"] for g in ok]
    return {
        "tag": tag,
        "n_games": n,
        "induce_ok": f"{len(ok)}/{n}",
        "fail_as_zero": round(float(np.mean(accs_all)), 4) if accs_all else 0.0,
        "survivor_mean": round(float(np.mean(accs_ok)), 4) if accs_ok else 0.0,
        "nonzero_cells_games": sum(1 for r in per.values() if r["changed_any_cell"]),
        "mean_heldout_exact": round(
            float(np.mean([r["heldout_exact"] for r in per.values()])), 4) if per else 0.0,
        "per_game": per,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--out", default="quality_results.json")
    a = ap.parse_args()
    outp = os.path.join(SCRATCH, a.out)
    res = json.load(open(outp)) if os.path.exists(outp) else {}
    for t in a.tags:
        res[t] = grade_config(t)
        s = res[t]
        print(f"{t}: induce_ok={s.get('induce_ok')} fail_as_zero={s.get('fail_as_zero')} "
              f"survivor={s.get('survivor_mean')} nonzero={s.get('nonzero_cells_games')}")
    json.dump(res, open(outp, "w"), indent=1)
    print(json.dumps({"out": outp}))
