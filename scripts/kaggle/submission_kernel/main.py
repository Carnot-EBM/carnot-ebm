"""ARC-AGI-3 Kaggle competition SUBMISSION notebook — CarnotAgent.

Modeled EXACTLY on the canonical arcprize control notebook pattern (Ronan McGovern's
`arc3-random-control`, the reference submission shape). The competition is a CODE
competition with an INTERNAL game GATEWAY — NOT the offline-bundled-games shape our
earlier checklist assumed. The real contract:

  * Two modes, gated on env `KAGGLE_IS_COMPETITION_RERUN`:
      - NON-rerun (when you Save & Run to submit): write the placeholder
        `/kaggle/working/submission.parquet` so the notebook is a valid submission.
      - RERUN (Kaggle's actual scored eval): wait for the internal game gateway at
        `http://gateway:8001/api/games`, copy the COMPETITION-PROVIDED ARC-AGI-3-Agents
        framework, drop in our CarnotAgent as `my_agent.py`, register it, and run
        `main.py --agent=carnotagent` against the gateway.
  * Dependencies come from competition-provided wheels (`--no-index`); internet is OFF.
    The game API is the internal gateway, not the public three.arcprize.org.

Our local Qwen3.5-9B-MTP generator runs OFFLINE via the bundled `llama-server` binary +
GGUF (attached datasets) — internet-off is fine because that path is a pure disk read +
local GPU. If the bundled engine is unavailable the agent degrades gracefully to the
CPU graph-explore cascade (try/except in arc_executable_world_model._induce_and_plan), so
the submission still plays games even if the LLM tier is unavailable.

Attach as datasets: carnot-agent-code, carnot-llamacpp-mtp-binary, carnot-qwen35-9b-mtp-gguf.
Add the competition as a data source. GPU on. Internet OFF (the gateway is internal).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

COMP = "/kaggle/input/competitions/arc-prize-2026-arc-agi-3"

# 1) competition-provided wheels, offline (mirrors the canonical control notebook)
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--find-links",
        f"{COMP}/arc_agi_3_wheels",
        "arc-agi",
        "python-dotenv",
        "--quiet",
    ],
    check=False,
)

# 2) author my_agent.py — our CarnotAgent wired to the bundled OFFLINE generator.
#    The carnot import (jax + the agentic stack) is validated to import offline on the
#    Kaggle image (agent dry-run: 3.6s). The bundled llama-server is copied to a writable
#    path (/kaggle/input is read-only) and pointed at via CARNOT_LLAMA_SERVER / GGUF env.
AGENT_SRC = r"""
import os, shutil, sys, time, subprocess, urllib.request
from pathlib import Path

inp = Path("/kaggle/input")
# self-locate the bundled carnot package (mount nests under .../datasets/<owner>/<slug>/);
# arc_competition_agent.py is at .../python/carnot/agentic/... -> sys.path needs .../python
carnot = next(p.parents[2] for p in inp.rglob("carnot/agentic/arc_competition_agent.py"))
sys.path.insert(0, str(carnot))

# HYBRID EXPLORER DIVERSITY (validated 2026-06-21). The depth_first_ride StepwiseExplorer over-commits to
# the top-salient branch and MISSES easy "structure-missed" first-level wins (r11l/sp80/cd82). With this
# flag set, once the search STALLS (no new level for CARNOT_ARC_EXPLORE_STALL=150 moves) the explorer pops a
# RANDOM untested action among the top-K instead of the most-salient pop(0) -- recovering those wins WITHOUT
# costing the efficient ones. Measured end-to-end through the authoritative scorer: 4/11 first-win + eff-sum
# 2.0804 vs the structured baseline's 1/11 + 2.0069. Set UNCONDITIONALLY here (the explorer runs whether or
# not the LLM tier below loads); parity-safe (default OFF in the code). General lever (diversity, not game-
# specific) -> expected to transfer to the hidden eval games' own structure-missed wins.
os.environ["CARNOT_ARC_EXPLORE_DIVERSITY"] = "1"

# --- generator (LLM tier) resolution + LOUD visibility (2026-06-21) --------------------------------
# The v3=0.08 run could NOT be diagnosed because nothing logged whether the Qwen generator loaded or
# silently degraded to the CPU graph-explore cascade (env vars were set inside `if server and gguf:`
# with no else, and the agent launches llama-server with stderr=DEVNULL). Make it self-reporting in the
# eval log so the operator can grep "LLM GENERATOR HEALTHY/FAILED" on the next run. We do NOT change the
# operator-frozen stack; the probe tests the REAL config the agent will run.
# (STALE COMMENT CORRECTED 2026-07-27: this said "MTP stays on ... only RECOMMENDS MTP=0 on OOM",
# but the line ~20 below unconditionally sets CARNOT_ARC_MTP=0, and has since 2026-06-21. The probe
# reads `_mtp` from that env var, so it does test the real config -- the comment describing the
# config was simply describing the wrong one, directly above the line that sets it.)
server = next(iter(inp.rglob("llama-server")), None)
# match the Qwen GGUF by name so an order-undefined rglob can't bind a stale/second .gguf
_ggufs = [g for g in inp.rglob("*.gguf") if ("Qwen3.5-9B" in g.name or "Q4_K_M" in g.name)] or list(inp.rglob("*.gguf"))
gguf = _ggufs[0] if _ggufs else None
if len(_ggufs) > 1:
    print(f"LLM TIER WARNING: {len(_ggufs)} GGUFs under /kaggle/input, using {gguf.name}; all={[g.name for g in _ggufs]}", flush=True)

if server and gguf:
    run_server = Path("/kaggle/working/llama-server")
    shutil.copy2(server, run_server)
    os.chmod(run_server, 0o755)
    os.environ["LD_LIBRARY_PATH"] = f"{server.parent}:" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["CARNOT_LLAMA_SERVER"] = str(run_server)
    os.environ["CARNOT_ARC_GGUF_PATH"] = str(gguf)
    # 16GB P100/T4 (2026-06-21, evidence-backed by the carnot-arc-binary-smoke probe): the MTP
    # self-draft loads a 2nd ~5.9GB copy of the model (probe: 11.8GB used vs 5.9GB MTP-off) for NO
    # throughput gain on this GPU (probe: 27.5 vs 25.3 tok/s, MTP-off slightly FASTER). Disable it to
    # free ~5.8GB of VRAM for KV headroom. Speculative decoding is exact, so output quality is
    # unchanged. (The frozen stack's MTP speedup was validated on the iGPU, not the P100.)
    #
    # NOTE FOR ANYONE READING THE VRAM ENVELOPE: because MTP is OFF here, the published envelope
    # (`MiB = 10547 + 0.02519*n_ctx + 206.83*slots`, exp5866) does NOT describe this launch -- it
    # was fit with `--spec-type draft-mtp` ON and over-predicts the scored footprint by ~6.1 GB.
    # Directly measured mtp-OFF per-PID residency on an RTX 3090 (2026-07-27): 5950 MiB at
    # n_ctx=16384, 7380 MiB at n_ctx=81920, i.e. the n_ctx fix costs ~1430 MiB here, not the
    # ~1668 MiB the mtp-on envelope predicts. Peak residency with 4 concurrent full-budget
    # requests in flight equalled idle residency to the MiB, because llama.cpp preallocates the
    # whole `-c` pool at load.
    os.environ["CARNOT_ARC_MTP"] = "0"
    _mtp = os.environ.get("CARNOT_ARC_MTP", "1") != "0"
    # READ the context-pool size and completion budget from the SHIPPED defaults instead of
    # repeating literals here. The old code printed "ctx=16384" and probed with -c 16384 as
    # hardcoded strings; if the agent's own default had moved, the probe would have validated
    # a configuration the agent never used -- and validated it as HEALTHY. That is the
    # measure-one-thing-ship-another shape of the 0.08 incident, in the diagnostic itself.
    from carnot.agentic.arc_executable_world_model import (
        _INDUCE_WORST_CASE_PROMPT_TOKENS,
        _LLAMA_SERVER_DEFAULT_SLOTS,
        _default_induce_n_ctx,
    )
    _ctx = str(_default_induce_n_ctx())
    _maxtok = int(os.environ.get("CARNOT_ARC_INDUCE_MAX_TOKENS", "4096"))
    # WHAT THE SCORED RUN ACTUALLY GETS. machine_shape in kernel-metadata.json is a free-form
    # string kagglesdk cannot validate locally (kaggle_api_extended.py:4648 says the allowed
    # names live in an enum not shipped with the SDK), and the only nvidia-smi read this project
    # holds is a P100 16GB from a DIFFERENT kernel. So print it: one line here finally settles
    # what "NvidiaL4" delivers, instead of another round of inferring it.
    try:
        _smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        print(f"LLM GPU HARDWARE: {_smi.stdout.strip() or _smi.stderr.strip()!r}", flush=True)
    except Exception as _se:
        print(f"LLM GPU HARDWARE: unavailable ({_se!r})", flush=True)
    print(f"LLM TIER RESOLVED: server={run_server} gguf={gguf.name} mtp={_mtp} ctx={_ctx} "
          f"max_tokens={_maxtok} slots_expected={_LLAMA_SERVER_DEFAULT_SLOTS} "
          f"worst_prompt_tokens={_INDUCE_WORST_CASE_PROMPT_TOKENS} kv=q8_0", flush=True)
    # one-shot health probe: spawn the generator with the SAME args the agent uses and confirm it
    # actually LOADS on this GPU (stderr CAPTURED, not swallowed like the agent's DEVNULL launch),
    # then free the port. Wrapped so a probe failure can NEVER crash the agent / zero the submission.
    try:
        _pp = 8945
        _err = open("/kaggle/working/llm_probe.err", "w")
        # CARNOT_ARC_NGL (default 999=all-GPU): the prefill-to-RAM lever. Lower it to spill weight layers
        # into system RAM, freeing VRAM for KV + the coexisting live CNN fit. Probe the SAME ngl the agent uses.
        _ngl = os.environ.get("CARNOT_ARC_NGL", "999")
        _args = [str(run_server), "-m", str(gguf), "-ngl", _ngl, "-c", _ctx,
                 "--port", str(_pp), "--host", "127.0.0.1", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
        if _mtp:
            _args += ["--spec-type", "draft-mtp", "--model-draft", str(gguf)]
        _proc = subprocess.Popen(_args, stdout=_err, stderr=_err)
        _ok = False
        for _ in range(150):  # up to ~300s for a cold GPU load
            if _proc.poll() is not None:
                break
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{_pp}/health", timeout=2) as r:
                    if r.status == 200:
                        _ok = True; break
            except Exception:
                time.sleep(2)
        # CONCURRENCY PROBE (2026-07-27, HARDENED after adversarial review the same day).
        # The pre-2026-07-27 probe only checked /health, i.e. concurrency 1 -- the exact blind
        # spot that hid the context-pool-exhaustion fault for the whole life of this submission.
        # swarm.py starts ONE THREAD PER GAME with no pool, so induce requests arrive together
        # and llama.cpp serves its own slot count concurrently, queueing the rest.
        #
        # THE FIRST HARDENED VERSION WAS STILL WRONG IN TWO WAYS, both measured:
        #   1. IT PROBED K=2 WHILE DEFENDING K=4. Admission needs n_ctx >= K*(prompt+n_predict).
        #      Its own synthetic prompt measures 17238 tokens through the model's tokenizer, so
        #      K=2 needs 42668 cells (passes at 81920) but K=4 needs 85336 (FAILS at 81920).
        #      Directly measured on an RTX 3090: 4/4 HTTP 500 "Context size has been exceeded",
        #      per-slot n_tokens 20469..20493 at release == exactly 81920/4. The probe would
        #      have printed "LLM CONCURRENCY OK" for a configuration that fails at the K the
        #      eval actually produces. Probe at the SLOT COUNT, read from the module that sizes
        #      the pool, so the two cannot disagree.
        #   2. IT WAS HTTP-STATUS-ONLY. That is exactly the gate shape the fix investigation's
        #      own load-bearing result says would have shipped the bug: `--parallel 1` passes an
        #      HTTP gate 4/4 at LOWER VRAM while generating 648/650/184/648 tokens against a
        #      4096 budget -- mode C, silent truncation, the defect under investigation. A
        #      `stop` on a newline made this strictly worse by halting generation after ~1
        #      token, so the probe could not have observed truncation even if it had looked.
        #      So: no stop sequence, READ the body, and compare tokens_predicted to n_predict.
        # The prompt is built to the SAME measured worst-case token count the pool is sized for
        # (_INDUCE_WORST_CASE_PROMPT_TOKENS), trimmed against the server's OWN /tokenize rather
        # than eyeballed -- a probe prompt bigger than the pool admits tests the wrong thing.
        _conc = "not_probed"
        if _ok:
            try:
                import json as _json
                from concurrent.futures import ThreadPoolExecutor as _TPE

                _K = int(_LLAMA_SERVER_DEFAULT_SLOTS)

                def _ntok(_text):
                    _r = urllib.request.Request(
                        f"http://127.0.0.1:{_pp}/tokenize",
                        data=_json.dumps({"content": _text}).encode(),
                        headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(_r, timeout=120) as _resp:
                        return len(_json.load(_resp).get("tokens") or [])

                # READ total_slots + the pool size from the server itself. The scored run uses a
                # DIFFERENT bundled binary from this repo's local build; if its no---parallel
                # default is not 4, the pool is mis-sized and every number above is wrong.
                _slots = _props_ctx = None
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{_pp}/props", timeout=30) as _r:
                        _props = _json.load(_r)
                    _slots = _props.get("total_slots")
                    _props_ctx = (_props.get("default_generation_settings") or {}).get("n_ctx")
                except Exception as _pe:
                    print(f"LLM PROPS READ FAILED (non-fatal): {_pe!r}", flush=True)
                print(f"LLM SERVER PROPS: total_slots={_slots} n_ctx={_props_ctx} "
                      f"(expected slots={_K} n_ctx={_ctx})", flush=True)
                if _slots is not None and int(_slots) != _K:
                    print(f"LLM SLOT COUNT MISMATCH: server reports total_slots={_slots} but the "
                          f"pool was sized for {_K}. The admission arithmetic "
                          f"n_ctx >= K*(prompt+n_predict) is sized for the WRONG K -- induction "
                          f"will refuse or silently truncate. Operator: set CARNOT_ARC_INDUCE_N_CTX "
                          f">= {_slots} x ({_INDUCE_WORST_CASE_PROMPT_TOKENS} + {_maxtok}).",
                          flush=True)
                    _K = int(_slots)  # probe what this binary will ACTUALLY run concurrently

                # Build a prompt at the measured worst-case size, verified by /tokenize.
                _row = "Row: " + " ".join("1234567890" for _ in range(60)) + "\n"
                _big = _row * 26
                try:
                    _t = _ntok(_big)
                    while _t > _INDUCE_WORST_CASE_PROMPT_TOKENS and len(_big) > len(_row):
                        _big = _big[: -len(_row)]
                        _t = _ntok(_big)
                    _ptok = _t
                except Exception as _te:
                    _ptok = None
                    print(f"LLM TOKENIZE READ FAILED (non-fatal, probing untrimmed prompt): "
                          f"{_te!r}", flush=True)
                _body = _json.dumps({"prompt": _big, "n_predict": _maxtok,
                                     "temperature": 0.3, "cache_prompt": True}).encode()

                def _one(_i):
                    _r = urllib.request.Request(f"http://127.0.0.1:{_pp}/completion", data=_body,
                                                headers={"Content-Type": "application/json"})
                    try:
                        with urllib.request.urlopen(_r, timeout=900) as _resp:
                            _payload = _json.load(_resp)
                        _tim = _payload.get("timings") or {}
                        _gen = _tim.get("predicted_n")
                        if not isinstance(_gen, int):
                            _u = _payload.get("usage") or {}
                            _gen = _u.get("completion_tokens")
                        return {"status": _resp.status, "stop_type": _payload.get("stop_type"),
                                "generated": _gen,
                                "chars": len(_payload.get("content") or "")}
                    except Exception as _ex:
                        # READ the body: the 500 says "Context size has been exceeded." and the
                        # 400 says "...try increasing it" -- literally the fix, thrown away
                        # unread twice before _describe_http_failure started printing it.
                        _detail = ""
                        try:
                            _detail = _ex.read().decode("utf-8", "replace")[:200]
                        except Exception:
                            _detail = str(_ex)[:200]
                        return {"status": f"{type(_ex).__name__}:{getattr(_ex, 'code', '')}",
                                "body": _detail}

                with _TPE(max_workers=_K) as _ex2:
                    _res = list(_ex2.map(_one, range(_K)))
                _codes = [_r.get("status") for _r in _res]
                # MODE C: HTTP 200 that stopped on `limit` far short of the budget we asked for.
                # This is the failure a status-only gate cannot see, and the one that silently
                # degrades induction quality instead of loudly refusing.
                _trunc = [_r for _r in _res
                          if _r.get("status") == 200 and _r.get("stop_type") == "limit"
                          and isinstance(_r.get("generated"), int)
                          and _r["generated"] < _maxtok - 8]
                _alive = False
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{_pp}/health", timeout=5) as r:
                        _alive = r.status == 200
                except Exception:
                    _alive = False
                _conc = (f"K{_K}_prompt_tokens={_ptok} results={_res} "
                         f"pool_exhaustion_truncations={len(_trunc)} server_alive_after={_alive}")
                if all(c == 200 for c in _codes) and _alive and not _trunc:
                    print(f"LLM CONCURRENCY OK -- {_K} simultaneous full-budget requests all "
                          f"succeeded at ctx={_ctx} with no pool truncation ({_conc})", flush=True)
                elif _trunc:
                    print(f"LLM CONCURRENCY SILENTLY TRUNCATED at ctx={_ctx}/max_tokens={_maxtok}: "
                          f"{len(_trunc)} of {_K} requests returned HTTP 200 but stopped on "
                          f"'limit' far short of the budget -- the prompt consumed the shared "
                          f"pool and induction will be quietly degraded, NOT refused. This is the "
                          f"failure an HTTP-status gate cannot see. Operator: raise "
                          f"CARNOT_ARC_INDUCE_N_CTX (needs >= {_K} x (prompt + {_maxtok})); "
                          f"raising max_tokens would make it WORSE. ({_conc})", flush=True)
                else:
                    print(f"LLM CONCURRENCY FAILED at ctx={_ctx}/max_tokens={_maxtok} ({_conc}). "
                          f"The eval runs one thread per game, so induction WILL degrade "
                          f"silently. Operator: raise CARNOT_ARC_INDUCE_N_CTX (needs >= "
                          f"{_K} x (prompt + {_maxtok})).", flush=True)
            except Exception as _ce:
                _conc = f"probe_error:{_ce!r}"
                print(f"LLM CONCURRENCY PROBE ERROR (non-fatal): {_ce!r}", flush=True)
        _proc.terminate()
        try:
            _proc.wait(timeout=15)
        except Exception:
            _proc.kill()
        _err.close()
        if _ok:
            print(f"LLM GENERATOR HEALTHY -- loaded on GPU, /health ok (generator tier ENGAGED); "
                  f"concurrency: {_conc}", flush=True)
        else:
            _tail = Path("/kaggle/working/llm_probe.err").read_text()[-1000:]
            print(f"LLM GENERATOR FAILED TO LOAD (likely OOM at mtp={_mtp}/ctx={_ctx}) -- agent will "
                  f"run CPU graph-explore ONLY. Operator: consider CARNOT_ARC_MTP=0 or a lower "
                  f"CARNOT_ARC_INDUCE_N_CTX. stderr tail:\n{_tail}",
                  flush=True)
    except Exception as _e:
        print(f"LLM PROBE ERROR (non-fatal, agent continues with LLM env set): {_e!r}", flush=True)
else:
    print("LLM TIER DISABLED: llama-server/gguf NOT FOUND under /kaggle/input -- running CPU graph-explore "
          f"ONLY (server={server}, gguf={gguf}). Verify the carnot-llamacpp + qwen GGUF datasets are attached.",
          flush=True)

from agents.agent import Agent
from carnot.agentic.arc_competition_agent import make_carnot_agent

# the verifier-routed cascade (graph-explore -> E3 induction via the bundled Qwen).
# registered under "carnotagent" in the rewritten agents/__init__.py below.
CarnotAgent = make_carnot_agent(Agent)
"""
Path("/kaggle/working/my_agent.py").write_text(AGENT_SRC)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    # 3) wait for the internal game gateway to come up (canonical pattern)
    subprocess.run(
        "curl --fail --retry 999 --retry-all-errors --retry-delay 5 "
        "--retry-max-time 600 http://gateway:8001/api/games",
        shell=True,
        check=False,
    )
    # 4) copy the COMPETITION-PROVIDED framework to a writable location
    fw = "/kaggle/working/ARC-AGI-3-Agents"
    shutil.rmtree(fw, ignore_errors=True)
    shutil.copytree(f"{COMP}/ARC-AGI-3-Agents", fw)
    shutil.copy2("/kaggle/working/my_agent.py", f"{fw}/agents/templates/my_agent.py")
    # 5) minimal agents/__init__.py — the stock one eagerly imports langgraph / smolagents
    #    templates whose deps are NOT installed (would crash). Register only what we use.
    Path(f"{fw}/agents/__init__.py").write_text(
        "from typing import Type, cast\n"
        "from dotenv import load_dotenv\n"
        "from .agent import Agent, Playback\n"
        "from .swarm import Swarm\n"
        "from .templates.random_agent import Random\n"
        "from .templates.my_agent import CarnotAgent\n"
        "load_dotenv()\n"
        "AVAILABLE_AGENTS: dict[str, Type[Agent]] = "
        '{"random": Random, "carnotagent": CarnotAgent}\n'
    )
    # 6) CRITICAL: point main.py at the gateway. main.py builds its game-API URL from
    #    SCHEME/HOST/PORT env (default localhost:8001) and loads .env LAST with override=True.
    #    Without this .env it queries localhost, finds no games, records no scorecard -> ERROR.
    #    (This is the omission that errored submission 53862044; mirrors the canonical control nb.)
    Path(f"{fw}/.env").write_text(
        "SCHEME=http\n"
        "HOST=gateway\n"
        "PORT=8001\n"
        "ARC_API_KEY=test-key-123\n"
        "ARC_BASE_URL=http://gateway:8001/\n"
        "OPERATION_MODE=online\n"
        "ENVIRONMENTS_DIR=\n"
        "RECORDINGS_DIR=/kaggle/working/server_recording\n"
    )
    # 7) play all gateway games (12h Kaggle cap). main.py fetches the game list from the
    #    gateway, runs the swarm, and the gateway records the scorecard that is scored.
    run_env = os.environ.copy()
    run_env["MPLBACKEND"] = "agg"  # headless matplotlib (canonical nb sets this)
    subprocess.run(
        [sys.executable, "main.py", "--agent", "carnotagent"],
        cwd=fw,
        env=run_env,
        timeout=43200,
        check=False,
    )
else:
    # NON-rerun: write the placeholder submission so Save & Run produces a valid entry.
    import pandas as pd

    pd.DataFrame(
        [["1_0", "1", True, 1]],
        columns=["row_id", "game_id", "end_of_game", "score"],
    ).to_parquet("/kaggle/working/submission.parquet", index=False)

print("CarnotAgent submission notebook complete.")
