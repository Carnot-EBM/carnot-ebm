"""Generalized AR-model Layer-B benchmark: run any GGUF model through the SAME scaffolded grounding
harness used for gemma-12B / Qwopus / DiffusionGemma, so 16GB-fit candidates are directly comparable.

Operator 2026-06-18 ("1 + 2"): benchmark the top-two 16GB-fit shortlist picks
(docs/research-notes/arc-16gb-model-alternatives-2026-06-18.md) against the gemma-4-12B baseline:
  1. Qwen3.5-9B (Apache, 5.7GB Q4 -> ~10GB KV headroom) -- hybrid thinking; the KEY test is whether
     thinking-OFF (/no_think) yields clean grounded predicates (fixing the forced-CoT-floods-output
     failure we hit on DiffusionGemma).
  2. Qwen2.5-Coder-14B-Instruct (Apache, 9.0GB Q4) -- dedicated small coder for the predicate-gen #1 need.

All on the offline-legal iGPU (port 8921, NOT the 3090s, per the 2026-06-17 directive); zero quota.
Reports tokens/sec + the grounding tier per game, identical metrics to the prior comparisons.

Usage: python scripts/experiments/arc3_layerb_ar_model_test.py <model_key> [game ...]
  model_key in: qwen3.5-9b | qwen2.5-coder-14b | gemma-12b   (gemma re-run = same-harness baseline)"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

LLAMA_SERVER = Path.home() / ".cache" / "llama.cpp-master" / "build-hip" / "bin" / "llama-server"
HUB = Path.home() / ".cache" / "huggingface" / "hub"
PORT = 8921
SERVER_LOG = Path("/tmp/ar_model_test_server.log")

# model_key -> (gguf glob, no_think_prefix). no_think_prefix is prepended to the prompt to suppress a
# hybrid-thinking model's chain-of-thought (Qwen3 honors /no_think); None = model has no thinking mode.
MODELS = {
    "qwen3.5-9b": (
        "models--unsloth--Qwen3.5-9B-GGUF/snapshots/*/Qwen3.5-9B-Q4_K_M.gguf",
        "/no_think\n",
    ),
    "qwen3.5-9b-mtp": (
        "models--unsloth--Qwen3.5-9B-MTP-GGUF/snapshots/*/Qwen3.5-9B-Q4_K_M.gguf",
        "/no_think\n",
    ),
    # DeepSeek-V4-Flash reasoning-distill of Qwen3.5-9B + MTP heads -- tests whether the distill improves
    # grounding (accuracy) over base Qwen3.5-9B while keeping the speed.
    "deepseek-flash-mtp": (
        "models--Jackrong--Qwen3.5-9B-DeepSeek-V4-Flash-MTP-GGUF/snapshots/*/Qwen3.5-9B-DeepSeek-V4-Flash-MTP-Q4_K_M.gguf",
        "/no_think\n",
    ),
    "qwen3-14b": (
        "models--unsloth--Qwen3-14B-GGUF/snapshots/*/Qwen3-14B-Q4_K_M.gguf",
        "/no_think\n",
    ),
    "phi-4": (
        "models--unsloth--phi-4-GGUF/snapshots/*/phi-4-Q4_K_M.gguf",
        None,
    ),  # base Phi-4: no thinking mode
    "qwen2.5-coder-14b": (
        "models--unsloth--Qwen2.5-Coder-14B-Instruct-GGUF/snapshots/*/Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf",
        None,
    ),
    # Qwopus-27B-Coder at a SMALLER quant (Q3_K_M 13.5GB) to fit 16GB -- tests whether a quantized larger
    # coder beats the small Q4 models on accuracy. Qwen3.6-based, so /no_think applies.
    "qwopus-q3km": (
        "models--Jackrong--Qwopus3.6-27B-Coder-MTP-GGUF/snapshots/*/Qwopus3.6-27B-Coder-MTP-Q3_K_M.gguf",
        "/no_think\n",
    ),
    "gemma-12b": (
        "models--unsloth--gemma-4-12B-it-GGUF/snapshots/*/gemma-4-12b-it-Q4_K_M.gguf",
        None,
    ),
}


def _best_code(raw):
    """Pick the COMPLETE is_win block. Some models (Qwen3.5) emit a stub block first (`e = grid[...]` +
    `# ...`) then the real predicate later; the project's first-block extractor grabs the stub. Prefer the
    LAST ```python block that has both `def is_win` and a `return`; fall back progressively."""
    from carnot.agentic.arc_executable_world_model import _extract_python

    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", raw, re.DOTALL)
    complete = [b for b in blocks if "def is_win" in b and "return" in b]
    if complete:
        return complete[-1]
    any_win = [b for b in blocks if "def is_win" in b]
    if any_win:
        return any_win[-1]
    return _extract_python(raw) or raw


def _scaffold():
    spec = importlib.util.spec_from_file_location(
        "sf", str(REPO / "scripts" / "experiments" / "arc3_config_layerb_scaffolded.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _healthy():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def start_server(gguf, mtp=False):
    args = [
        str(LLAMA_SERVER),
        "-m",
        gguf,
        "-ngl",
        "999",
        "-c",
        "6144",
        "--port",
        str(PORT),
        "--host",
        "127.0.0.1",
    ]
    if mtp:
        # SELF-DRAFT ONLY IF THE MODEL REALLY IS A SELF-DRAFTING MTP BUILD (guarded 2026-07-28).
        # `--model-draft <the main gguf>` is correct for a `-MTP-` build (Qwen3.5-9B-MTP carries its
        # nextn heads inside the main GGUF) and WRONG for every other model. It is wrong silently:
        # llama.cpp warns and then serves with speculation DISABLED, so an arm labelled "mtp=True"
        # measures the mtp-off regime and nothing in the output says so. The live pin
        # (gemma-4-31B-it) keeps its MTP head in a SEPARATE `mtp-*.gguf` file, so if this script is
        # ever pointed at it the guard drops the flags and says so rather than mislabelling the arm.
        if "-mtp" in Path(gguf).name.lower() or "_mtp" in Path(gguf).name.lower():
            args += ["--spec-type", "draft-mtp", "--model-draft", gguf]
        else:
            print(
                f"[layerb] MTP NOT ENGAGED: {Path(gguf).name!r} is not a self-drafting MTP build. "
                "Serving WITHOUT speculative decoding; this arm is MTP-OFF.",
                flush=True,
            )
    log = open(SERVER_LOG, "w")
    proc = subprocess.Popen(args, stdout=log, stderr=log)
    for _ in range(180):
        if _healthy():
            return proc
        time.sleep(2)
    proc.terminate()
    return None


# the fairness clarification (reused by the repeated-benchmark driver): ka59's reference region is not
# static, so tell models the digest counts are fixed constants -- measures whether they find the right
# RELATION, not whether they hardcode vs recompute a non-static reference.
CLARIFY = (
    "IMPORTANT: the per-colour counts in the reference digest are FIXED values measured "
    "once; use those numbers as numeric CONSTANTS in your predicate. Do NOT recompute "
    "reference-region counts from the live `grid` (the reference region is not static).\n\n"
)


def gen(prompt, n_predict=1100, seed=None):
    body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0.2, "cache_prompt": True}
    if seed is not None:
        body["seed"] = int(seed)
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/completion",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        resp = json.load(r)
    dt = time.time() - t0
    return resp.get("content", ""), dt, (resp.get("tokens_predicted") or 0)


def main():
    if not sys.argv[1:] or sys.argv[1] not in MODELS:
        print(f"usage: <model_key in {list(MODELS)}> [games...]", flush=True)
        return 1
    key = sys.argv[1]
    mtp = "--mtp" in sys.argv
    games = [a for a in sys.argv[2:] if a != "--mtp"] or ["ka59", "tn36"]
    glob_pat, no_think = MODELS[key]
    hits = list(HUB.glob(glob_pat))
    if not hits:
        print(f"  GGUF not found for {key}", flush=True)
        return 1
    gguf = str(hits[0])
    print(
        f"== Layer-B benchmark: {key} (no_think={'yes' if no_think else 'n/a'}, mtp={mtp}) games={games} ==",
        flush=True,
    )
    sf = _scaffold()
    proc = start_server(gguf, mtp=mtp)
    if proc is None:
        print(f"  server failed to start (see {SERVER_LOG})", flush=True)
        return 1
    rows = []
    try:
        for g in games:
            try:
                sf.GAME = g
                scene, eb, bg, ref_box, win, ch, nonwins = sf.collect()
            except Exception as ex:
                rows.append({"game": g, "tier": "COLLECT_ERROR", "err": str(ex)[:100]})
                continue
            if eb is None or win is None or len(nonwins) < 2:
                rows.append({"game": g, "tier": "BLOCKED"})
                print(f"  {g}: BLOCKED", flush=True)
                continue
            win_sub = sf._edit_sub(win, eb)
            prompt = sf.build_prompt(scene, eb, bg, ref_box, win, nonwins)
            # FAIRNESS FIX: ka59's reference region is NOT static (its colour counts drift between the
            # digest snapshot and the win grid), so a rule that RECOMPUTES reference counts from the live
            # `grid` fails to ground while a hardcoded digest constant succeeds. Stronger models wrote the
            # (better) dynamic-recompute rule and were penalised. Tell every model the digest counts are
            # fixed constants so the comparison measures whether they find the right RELATION, not whether
            # they happened to hardcode vs recompute a non-static reference.
            prompt = (no_think or "") + CLARIFY + prompt
            if len(prompt) // 4 > 4500:
                rows.append({"game": g, "tier": "SKIPPED_PROMPT_TOO_LARGE"})
                print(f"  {g}: SKIPPED big prompt", flush=True)
                continue
            try:
                raw, dt, ntok = gen(prompt)
            except Exception as ex:
                rows.append(
                    {
                        "game": g,
                        "tier": "GEN_ERROR",
                        "err": str(ex)[:120],
                        "server_tail": SERVER_LOG.read_text()[-200:],
                    }
                )
                print(f"  {g}: GEN_ERROR {ex}", flush=True)
                continue
            tps = round(ntok / max(dt, 0.01), 2)
            code = _best_code(raw)
            row = {
                "game": g,
                "tok_per_s": tps,
                "tokens": ntok,
                "dur_s": round(dt, 1),
                "coherent": "def is_win" in code,
                "raw_sample": raw[:400],
            }
            if "def is_win" in code:
                ns = {}
                try:
                    exec(code, ns)  # noqa: S102
                    v = sf.verify(ns.get("is_win", lambda x: False), win, nonwins)
                    fpr = v.get("false_positive_rate")
                    grounded = bool(v.get("fires_on_win")) and fpr is not None and fpr < 0.2
                    literal = sf._looks_literal_hardcode(code, win_sub)
                    row["verification"] = v
                    row["grounded"] = grounded
                    row["literal"] = bool(literal)
                    row["tier"] = (
                        "TIER2_GROUNDED_RELATIONAL"
                        if grounded and not literal
                        else "TIER1_GROUNDED_LITERAL"
                        if grounded
                        else "TIER0_COHERENT_NOT_GROUNDED"
                    )
                except Exception as ex:
                    row["tier"] = "TIER0_UNCOMPILABLE"
                    row["exec_error"] = str(ex)[:100]
            else:
                row["tier"] = "TIER0_FAIL"
            rows.append(row)
            print(
                f"  {g}: {row['tier']:28} {tps} tok/s ({row['dur_s']}s, {ntok} tok) coherent={row.get('coherent')}",
                flush=True,
            )
    finally:
        proc.terminate()
    grounded = [r["game"] for r in rows if r.get("grounded")]
    out = {
        "experiment": "arc3_layerb_ar_model_test",
        "model_key": key,
        "gguf": Path(gguf).name,
        "no_think": bool(no_think),
        "mtp": mtp,
        "per_game": rows,
        "grounded_games": grounded,
        "honest_verdict": f"complete_layerb_{key.replace('.', '_').replace('-', '_')}_mtp_{mtp}_grounded_{len(grounded)}",
        "inference_substrate": f"offline_arc_agi3_{key}{'_mtp' if mtp else ''}_iGPU_port8921",
    }
    tag = key.replace(".", "_") + ("_mtp" if mtp else "")
    (REPO / "results" / f"arc3_layerb_ar_model_test_{tag}.json").write_text(
        json.dumps(out, indent=2, default=str)
    )
    print(f"\n  grounded={grounded} -> {out['honest_verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
