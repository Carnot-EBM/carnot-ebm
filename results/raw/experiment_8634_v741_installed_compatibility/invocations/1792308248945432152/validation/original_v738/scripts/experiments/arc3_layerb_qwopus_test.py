"""Qwopus3.6-27B-Coder-MTP vs gemma-4-12B on the Config Layer B task -- speed + accuracy comparison.

Operator request (2026-06-18): try the OSS SOTA coder Jackrong/Qwopus3.6-27B-Coder-MTP-GGUF with a recent
llama.cpp (MTP support) to see if we get FASTER and MORE ACCURATE Layer-B rule induction. gemma-12B-Q4
(non-coder, ~4.2 tok/s on the iGPU) grounded ka59 + tn36 (Tier 2) but TIER0_FAILed cd82/wa30. Qwopus is a
27B agentic CODER (SWE-bench 67%, Opus-distilled) -- expected MORE accurate; speed depends on the iGPU
(27B is ~2x gemma's params; MTP's ~1.66x speculative speedup roughly offsets that).

Runs the SAME scaffolded prompt (arc3_config_layerb_scaffolded) through a Qwopus llama-server on the iGPU
(port 8921, NOT the 3090s per the 2026-06-17 directive) and reports tokens/sec + the grounding tier, so
it is directly comparable to the gemma sweep. --mtp toggles native llama.cpp MTP speculative decoding
(--spec-type draft-mtp). NOT the conductor's gemma server (port 8920) -- separate port, separate model.

Usage: python scripts/experiments/arc3_layerb_qwopus_test.py [game ...] [--mtp]   (default games: cd82 ka59)"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

LLAMA_SERVER = Path.home() / ".cache" / "llama.cpp-master" / "build-hip" / "bin" / "llama-server"
PORT = 8921


def _find_gguf():
    base = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--Jackrong--Qwopus3.6-27B-Coder-MTP-GGUF"
    )
    hits = list(base.glob("snapshots/*/Qwopus3.6-27B-Coder-MTP-Q4_K_M.gguf"))
    return str(hits[0]) if hits else None


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


SERVER_LOG = Path("/tmp/qwopus_test_server.log")


def start_server(gguf, mtp=False):
    # -c 6144 fits the count-class prompts (~0.7-1k tok) + 1100 predict with headroom, and keeps the 27B
    # KV cache small enough to avoid iGPU OOM (a 10k-token prompt at -c 16384 crashed the server).
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
    for _ in range(180):  # 27B load on the iGPU is slower than 12B
        if _healthy():
            return proc
        time.sleep(2)
    proc.terminate()
    return None


def gen(prompt, n_predict=1100):
    body = json.dumps(
        {"prompt": prompt, "n_predict": n_predict, "temperature": 0.2, "cache_prompt": True}
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/completion",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        resp = json.load(r)
    dt = time.time() - t0
    ntok = resp.get("tokens_predicted") or 0
    return resp.get("content", ""), dt, ntok


def main():
    argv = [a for a in sys.argv[1:] if a != "--mtp"]
    mtp = "--mtp" in sys.argv
    games = argv or ["cd82", "ka59"]
    gguf = _find_gguf()
    print(f"== Qwopus-27B-Coder Layer-B test (mtp={mtp}) games={games} ==", flush=True)
    if not gguf:
        print("  GGUF not found -- download incomplete", flush=True)
        return 1
    sf = _scaffold()
    proc = start_server(gguf, mtp=mtp)
    if proc is None:
        out = {
            "experiment": "arc3_layerb_qwopus_test",
            "honest_verdict": "blocked_qwopus_server_failed_to_start",
            "mtp": mtp,
            "gguf": gguf,
            "inference_substrate": "offline_arc_agi3_qwopus27b_iGPU_port8921",
        }
        (REPO / "results" / "arc3_layerb_qwopus_test.json").write_text(
            json.dumps(out, indent=2, default=str)
        )
        print("  -> server failed to start", flush=True)
        return 1
    rows = []
    try:
        for g in games:
            try:
                sf.GAME = g
                scene, eb, bg, ref_box, win, ch, nonwins = sf.collect()
            except Exception as ex:
                rows.append({"game": g, "tier": "COLLECT_ERROR", "err": str(ex)[:100]})
                print(f"  {g}: COLLECT_ERROR {ex}", flush=True)
                continue
            if eb is None or win is None or len(nonwins) < 2:
                rows.append({"game": g, "tier": "BLOCKED"})
                print(f"  {g}: BLOCKED (no win)", flush=True)
                continue
            win_sub = sf._edit_sub(win, eb)
            prompt = sf.build_prompt(scene, eb, bg, ref_box, win, nonwins)
            if len(prompt) // 4 > 4500:  # would exceed -c 6144 with the 1100-token predict budget
                rows.append(
                    {"game": g, "tier": "SKIPPED_PROMPT_TOO_LARGE", "approx_tok": len(prompt) // 4}
                )
                print(
                    f"  {g}: SKIPPED prompt ~{len(prompt) // 4} tok (large 2-D editable; wrong digest class)",
                    flush=True,
                )
                continue
            try:
                raw, dt, ntok = gen(prompt)
            except Exception as ex:
                tail = ""
                try:
                    tail = SERVER_LOG.read_text()[-300:]
                except Exception:
                    pass
                rows.append(
                    {"game": g, "tier": "GEN_ERROR", "err": str(ex)[:120], "server_tail": tail}
                )
                print(f"  {g}: GEN_ERROR {ex}", flush=True)
                continue
            tps = round(ntok / max(dt, 0.01), 2)
            from carnot.agentic.arc_executable_world_model import _extract_python

            code = _extract_python(raw) or raw
            row = {
                "game": g,
                "tok_per_s": tps,
                "tokens": ntok,
                "dur_s": round(dt, 1),
                "coherent": "def is_win" in code,
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
                f"  {g}: {row['tier']:28} {tps} tok/s ({ntok} tok / {row['dur_s']}s) coherent={row.get('coherent')}",
                flush=True,
            )
    finally:
        proc.terminate()
    out = {
        "experiment": "arc3_layerb_qwopus_test",
        "model": "Qwopus3.6-27B-Coder-MTP-Q4_K_M",
        "mtp": mtp,
        "per_game": rows,
        "mean_tok_per_s": round(
            float(np.mean([r["tok_per_s"] for r in rows if "tok_per_s" in r] or [0])), 2
        ),
        "grounded_games": [r["game"] for r in rows if r.get("grounded")],
        "honest_verdict": f"complete_qwopus27b_layerb_mtp_{mtp}_grounded_{len([r for r in rows if r.get('grounded')])}",
        "inference_substrate": "offline_arc_agi3_qwopus27b_iGPU_port8921",
    }
    tag = "_mtp" if mtp else ""
    (REPO / "results" / f"arc3_layerb_qwopus_test{tag}.json").write_text(
        json.dumps(out, indent=2, default=str)
    )
    print(
        f"\n  mean {out['mean_tok_per_s']} tok/s | grounded={out['grounded_games']} | -> {out['honest_verdict']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
