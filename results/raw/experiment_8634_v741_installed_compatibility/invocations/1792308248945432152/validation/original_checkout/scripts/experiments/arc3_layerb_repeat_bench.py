"""Repeated-seed Layer-B grounding-RATE benchmark -- beats down the 2-game small-sample noise.

Operator 2026-06-19: run base Qwen3.5-9B-MTP vs DeepSeek-V4-Flash-MTP 3-5x each on ka59+tn36 (different
seeds) for a grounding RATE, and raise the token cap to rule out truncation (DeepSeek-Flash hit the 1100
cap on both games -- was it rambling, or just truncated before finishing a correct rule?).

Reuses the AR harness wholesale (server lifecycle, prompt build, /no_think + fairness CLARIFY, _best_code
extraction, verify). Loads the model server ONCE, then runs each game N times at seeds 0..N-1 with a higher
n_predict, recording grounded/not per run -> per-game grounding rate. iGPU, offline, --mtp optional.

Usage: python scripts/experiments/arc3_layerb_repeat_bench.py <model_key> [--repeat N] [--n-predict M] [--mtp] [games...]"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

_spec = importlib.util.spec_from_file_location(
    "ar", str(REPO / "scripts" / "experiments" / "arc3_layerb_ar_model_test.py"))
ar = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ar)


def _argval(flag, default):
    if flag in sys.argv:
        return sys.argv[sys.argv.index(flag) + 1]
    return default


def main():
    if not sys.argv[1:] or sys.argv[1] not in ar.MODELS:
        print(f"usage: <model_key in {list(ar.MODELS)}> [--repeat N] [--n-predict M] [--mtp] [games...]", flush=True); return 1
    key = sys.argv[1]
    repeat = int(_argval("--repeat", 4))
    n_predict = int(_argval("--n-predict", 2560))
    mtp = "--mtp" in sys.argv
    skip = {"--repeat", "--n-predict", "--mtp", _argval("--repeat", None), _argval("--n-predict", None), key}
    games = [a for a in sys.argv[2:] if a not in skip] or ["ka59", "tn36"]
    glob_pat, no_think = ar.MODELS[key]
    hits = list(ar.HUB.glob(glob_pat))
    if not hits:
        print(f"  GGUF not found for {key}", flush=True); return 1
    gguf = str(hits[0])
    print(f"== REPEAT bench: {key} mtp={mtp} repeat={repeat} n_predict={n_predict} games={games} ==", flush=True)
    sf = ar._scaffold()
    proc = ar.start_server(gguf, mtp=mtp)
    if proc is None:
        print("  server failed", flush=True); return 1
    per_game = {}
    try:
        for g in games:
            sf.GAME = g
            scene, eb, bg, ref_box, win, ch, nonwins = sf.collect()
            if eb is None or win is None or len(nonwins) < 2:
                per_game[g] = {"blocked": True}; print(f"  {g}: BLOCKED", flush=True); continue
            win_sub = sf._edit_sub(win, eb)
            base_prompt = (no_think or "") + ar.CLARIFY + sf.build_prompt(scene, eb, bg, ref_box, win, nonwins)
            runs = []
            for s in range(repeat):
                try:
                    raw, dt, ntok = ar.gen(base_prompt, n_predict=n_predict, seed=s)
                except Exception as ex:
                    runs.append({"seed": s, "error": str(ex)[:80]}); continue
                code = ar._best_code(raw)
                grounded = False; capped = (ntok >= n_predict)
                if "def is_win" in code:
                    ns = {}
                    try:
                        exec(code, ns)  # noqa: S102
                        v = sf.verify(ns.get("is_win", lambda x: False), win, nonwins)
                        fpr = v.get("false_positive_rate")
                        grounded = bool(v.get("fires_on_win")) and fpr is not None and fpr < 0.2
                    except Exception:
                        pass
                runs.append({"seed": s, "grounded": grounded, "tokens": ntok, "capped": capped, "tok_per_s": round(ntok / max(dt, 0.01), 2)})
            n_gnd = sum(1 for r in runs if r.get("grounded"))
            n_cap = sum(1 for r in runs if r.get("capped"))
            per_game[g] = {"grounded_rate": f"{n_gnd}/{repeat}", "n_grounded": n_gnd, "n_capped": n_cap, "runs": runs}
            mt = sum(r.get("tokens", 0) for r in runs) / max(1, len(runs))
            print(f"  {g}: grounded {n_gnd}/{repeat} | capped {n_cap}/{repeat} | mean_tokens {mt:.0f} | "
                  f"per-seed: {[ (r.get('grounded'), r.get('tokens')) for r in runs ]}", flush=True)
    finally:
        proc.terminate()
    out = {"experiment": "arc3_layerb_repeat_bench", "model_key": key, "mtp": mtp, "repeat": repeat,
           "n_predict": n_predict, "per_game": per_game,
           "honest_verdict": f"complete_repeatbench_{key.replace('.','_').replace('-','_')}_mtp_{mtp}",
           "inference_substrate": f"offline_arc_agi3_{key}{'_mtp' if mtp else ''}_iGPU_repeat{repeat}"}
    tag = key.replace('.', '_') + ("_mtp" if mtp else "")
    (REPO / "results" / f"arc3_layerb_repeat_bench_{tag}.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"  -> {out['honest_verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
