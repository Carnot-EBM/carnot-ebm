"""DiffusionGemma-26B-A4B (block-diffusion, ~4B active MoE) on Config Layer B -- speed + accuracy, vs
gemma-4-12B and Qwopus-27B. Operator request (2026-06-18): compare the new
unsloth/diffusiongemma-26B-A4B-it-GGUF.

DiffusionGemma is NON-autoregressive: it denoises a fixed-length masked canvas over N steps (the whole
canvas in PARALLEL per step), via the dedicated `llama-diffusion-cli` runner (standard llama-server
cannot generate from it). We built llama-diffusion-cli for the HIP/iGPU backend (port-free CLI, NOT the
3090s, per the 2026-06-17 directive). Smoke: 18.4 tok/s effective (256-tok canvas, 11 entropy-bound
steps) -- ~4x gemma-12B's 4.2 tok/s because the canvas denoises in parallel. This is ALSO the project's
pending DiffusionGemma gate model (docs/research-notes/diffusiongemma-energy-guided-diffusion-spec.md).

Runs the SAME scaffolded Layer-B prompt as the gemma + Qwopus sweeps (prompt via -f file; is_win code
extracted from stdout; grounded by the same verifier), so results are directly comparable. Reports the
diffusion throughput (from the CLI's own timing line) + the grounding tier.

Usage: python scripts/experiments/arc3_layerb_diffusiongemma_test.py [game ...]   (default: ka59 tn36)"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

DIFF_CLI = Path.home() / ".cache" / "llama.cpp-master" / "build-hip" / "bin" / "llama-diffusion-cli"
N_PREDICT = 1024  # this is a REASONING model (<|channel>thought); the fixed canvas must hold
# the chain-of-thought AND the code. 384 filled with reasoning before the code; 1024 lets it finish.


def _find_gguf():
    hits = list((Path.home() / ".cache" / "huggingface" / "hub").glob(
        "models--unsloth--diffusiongemma-26B-A4B-it-GGUF/snapshots/*/diffusiongemma-26B-A4B-it-Q4_K_M.gguf"))
    return str(hits[0]) if hits else None


def _scaffold():
    spec = importlib.util.spec_from_file_location(
        "sf", str(REPO / "scripts" / "experiments" / "arc3_config_layerb_scaffolded.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def diffusion_generate(gguf, prompt):
    """Run llama-diffusion-cli on the iGPU; return (stdout_text, wall_s, tok_per_s)."""
    pf = REPO / "results" / "_diff_prompt.txt"
    pf.write_text(prompt)
    t0 = time.time()
    proc = subprocess.run([str(DIFF_CLI), "-m", gguf, "-ngl", "999", "-n", str(N_PREDICT), "-f", str(pf)],
                          capture_output=True, text=True, timeout=600)
    dt = time.time() - t0
    tps = None
    ms = re.findall(r"throughput:\s*([\d.]+)\s*tok/s", proc.stderr)
    if ms:
        tps = float(ms[-1])
    return proc.stdout, dt, tps


def main():
    games = sys.argv[1:] or ["ka59", "tn36"]
    print(f"== DiffusionGemma-26B-A4B Layer-B test games={games} (iGPU, block-diffusion) ==", flush=True)
    gguf = _find_gguf()
    if not gguf or not DIFF_CLI.exists():
        print(f"  blocked: gguf={bool(gguf)} cli={DIFF_CLI.exists()}", flush=True); return 1
    sf = _scaffold()
    from carnot.agentic.arc_executable_world_model import _extract_python
    rows = []
    for g in games:
        sf.GAME = g
        scene, eb, bg, ref_box, win, ch, nonwins = sf.collect()
        if eb is None or win is None or len(nonwins) < 2:
            rows.append({"game": g, "tier": "BLOCKED"}); print(f"  {g}: BLOCKED", flush=True); continue
        win_sub = sf._edit_sub(win, eb)
        prompt = sf.build_prompt(scene, eb, bg, ref_box, win, nonwins)
        if len(prompt) // 4 > 4500:
            rows.append({"game": g, "tier": "SKIPPED_PROMPT_TOO_LARGE", "approx_tok": len(prompt) // 4})
            print(f"  {g}: SKIPPED (~{len(prompt)//4} tok)", flush=True); continue
        try:
            out, dt, tps = diffusion_generate(gguf, prompt)
        except Exception as ex:
            rows.append({"game": g, "tier": "GEN_ERROR", "err": str(ex)[:120]}); print(f"  {g}: GEN_ERROR {ex}", flush=True); continue
        code = _extract_python(out) or out
        row = {"game": g, "wall_s": round(dt, 1), "tok_per_s": tps, "coherent": "def is_win" in code, "raw_sample": out[:600]}
        if "def is_win" in code:
            ns = {}
            try:
                exec(code, ns)  # noqa: S102
                v = sf.verify(ns.get("is_win", lambda x: False), win, nonwins)
                fpr = v.get("false_positive_rate")
                grounded = bool(v.get("fires_on_win")) and fpr is not None and fpr < 0.2
                literal = sf._looks_literal_hardcode(code, win_sub)
                row["verification"] = v; row["grounded"] = grounded; row["literal"] = bool(literal)
                row["tier"] = ("TIER2_GROUNDED_RELATIONAL" if grounded and not literal else
                               "TIER1_GROUNDED_LITERAL" if grounded else "TIER0_COHERENT_NOT_GROUNDED")
            except Exception as ex:
                row["tier"] = "TIER0_UNCOMPILABLE"; row["exec_error"] = str(ex)[:100]
        else:
            row["tier"] = "TIER0_FAIL"
        rows.append(row)
        print(f"  {g}: {row['tier']:28} {tps} tok/s ({row['wall_s']}s) coherent={row.get('coherent')}", flush=True)
    out = {"experiment": "arc3_layerb_diffusiongemma_test", "model": "diffusiongemma-26B-A4B-it-Q4_K_M",
           "runner": "llama-diffusion-cli (HIP/iGPU)", "n_predict": N_PREDICT, "per_game": rows,
           "grounded_games": [r["game"] for r in rows if r.get("grounded")],
           "honest_verdict": f"complete_diffusiongemma_layerb_grounded_{len([r for r in rows if r.get('grounded')])}",
           "inference_substrate": "offline_arc_agi3_diffusiongemma26b_block_diffusion_iGPU"}
    (REPO / "results" / "arc3_layerb_diffusiongemma_test.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  grounded={out['grounded_games']} -> {out['honest_verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
