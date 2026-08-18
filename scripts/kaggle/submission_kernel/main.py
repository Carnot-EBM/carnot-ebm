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

Our local generator runs OFFLINE via the bundled `llama-server` binary + GGUF (attached
datasets) — internet-off is fine because that path is a pure disk read + local GPU. If the
bundled engine is unavailable the agent degrades gracefully to the CPU graph-explore cascade
(try/except in arc_executable_world_model._induce_and_plan), so the submission still plays
games even if the LLM tier is unavailable.

Attach as datasets: carnot-agent-code, carnot-llamacpp-mtp-binary, carnot-gemma4-31b-it-qat-gguf
(this ONE dataset carries BOTH the target gguf and its matching MTP drafter -- the resolution
logic below identifies each by filename pattern, not by which dataset slug it came from).
Add the competition as a data source. GPU on. Internet OFF (the gateway is internal).

=========================== 2026-07-28 GENERATOR SWITCH — READ ===========================
The generator moved from Qwen3.5-9B-MTP (5.9GB Q4) to gemma-4-31B-it (18.3GB Q4) by operator
directive, on the grounds that the assumed 16GB Kaggle VRAM ceiling that forced the 9B pin is
void ("the Kaggle hardware is 96G since May"). Head-to-head evidence, 13 games x 3 replicates:
0.3843 fail-as-zero for the 31B vs 0.0627, matched tally 11-0-2, sign p=0.00098. See the
ARC_LIVE_GENERATOR_* block in python/carnot/agentic/arc_executable_world_model.py.

TWO THINGS ARE NOT DONE AND WILL BREAK THE NEXT SUBMISSION IF IGNORED:

 1. THE DATASET DOES NOT EXIST YET. `kernel-metadata.json` now requests
    `iancblenke/carnot-gemma4-31b-it-gguf`, which the OPERATOR must create and upload (the
    18.3GB gemma-4-31B-it-Q4_K_M.gguf). Until then the push fails at dataset resolution --
    deliberately a LOUD failure rather than silently running the old 9B.

    RESOLVED 2026-08-10, DIFFERENTLY THAN DESCRIBED ABOVE (never-prune: original text kept,
    correction appended). The project switched to the QAT quantization on 2026-08-09
    (`unsloth/gemma-4-31B-it-qat-GGUF`, statistically indistinguishable from Q4_K_M offline,
    p=1.0, but ~1GB smaller and ships a matching QAT MTP drafter). `kernel-metadata.json` now
    requests `iancblenke/carnot-gemma4-31b-it-qat-gguf` -- ONE dataset carrying BOTH the
    17.3GB target and its 491MB drafter -- not the two-dataset non-QAT pair this paragraph
    describes. That dataset exists and is uploaded (verified via `kaggle datasets files`,
    byte-exact against the local QAT snapshot). This correction is also why a 2026-08-09
    submission (ref 55393553) scored 0.02, well below the 0.08/0.12 prior baselines: it used a
    STALE, hand-maintained COPY of this kernel script from a staging directory that had
    silently diverged from this tracked file and was missing the concurrency-probe fix and the
    current n_ctx resolution below. See `ops/known-issues.md`'s 2026-08-09 "Kaggle score
    regression" entry for the full incident. The staging directory should now always be
    re-populated from THIS file immediately before push, never hand-edited in parallel.

 2. `machine_shape` IS "NvidiaRtxPro6000" AND IS UNVERIFIED BY US. It was chosen on two
    independent pieces of evidence — the arcprize.org 2026 starter kit names an `rtx6000`
    accelerator ("Nvidia RTX 6000 (g4-standard-48), Heavy ML; ARC-AGI-3 exclusive"), and a
    real Kaggle-pulled 3rd-place kernel in THIS competition
    (external/arc-m1-3rd-forge/kernel-metadata.json, server-assigned id_no 124697453) requests
    exactly that string alongside a gemma-4-31b model source. But the local kagglesdk CANNOT
    validate it: kernels_api_service.py documents only NvidiaTeslaT4/P100/Tpu1VmV38 and omits
    even NvidiaL4, which we have been using successfully — so the SDK's silence is stale
    documentation, not counter-evidence. Availability is also not allocation: nothing here
    proves what a requested shape actually delivers. The kernel already prints an
    `LLM GPU HARDWARE:` nvidia-smi line below; the operator's next submission log settles it.
    DO NOT submit merely to confirm this — read it off the next real run.

    18.3GB of weights plus an 81920-cell q8 KV pool does NOT fit the 24GB NvidiaL4 this
    previously requested, so reverting machine_shape without also shrinking the model or the
    context is a guaranteed OOM.

    VERIFIED 2026-08-11 (operator directive, submission ref 55425907): the RtxPro6000 shape
    WAS actually allocated for a real run -- no longer "UNVERIFIED BY US" above (that framing
    is kept, never-prune, for the reasoning trail; read it as historical). Operator's own
    words, load-bearing: "We absolutely were able to allocate an GPU RTX Pro 6000 for our
    submission run. This should be noted as it is not an L4x4."

    THAT LAST CLAUSE MATTERS STRUCTURALLY, NOT JUST AS A LABEL. Every VRAM-fit and MTP
    comment in this file above (the 4x24GB=96GB math, "all 4 L4s stay visible and the
    weights spread across 96 GB", the CUDA_VISIBLE_DEVICES avoidance reasoning) was written
    assuming FOUR separate 24GB devices that llama-server layer-splits across. `kernel-
    metadata.json`'s `machine_shape` string is "NvidiaRtxPro6000" with NO "x4"-style
    multiplier (contrast "NvidiaL4x4", which explicitly has one) -- consistent with this
    being ONE card, not four. If so, the multi-device layer-splitting machinery this file
    reasons about doesn't apply at all (nothing to split across); what matters instead is
    simply whether that one card's total VRAM covers the ~26.6GB requirement at n_ctx=106496.
    NOT YET CONFIRMED: the exact RTX PRO 6000 SKU (the Ada-generation "RTX 6000 Ada" is
    48GB; a Blackwell-generation "RTX PRO 6000 Blackwell" can be 96GB) and therefore its
    single-card VRAM ceiling. The code does NOT need to change for this uncertainty --
    `_generator_cuda_min_free_mb` plus the live nvidia-smi read a few hundred lines below
    already MEASURE the real free VRAM at runtime rather than assuming a topology, so the
    fit check is correct regardless of 1-card vs 4-card. This note exists so a future reader
    does not re-derive the stale 4x24GB mental model from the paragraphs above it. Read the
    `LLM GPU HARDWARE:` line from ref 55425907's actual scored-run log once available for the
    definitive card count and VRAM total.
==========================================================================================
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

COMP = "/kaggle/input/competitions/arc-prize-2026-arc-agi-3"

# CARNOT_ARC_SERVER_LOG_DIR (2026-08-08 adversarial review, Gaps finding 7). Unset, the agent's
# llama-server stderr log falls back to tempfile.gettempdir() -- inside the ephemeral Kaggle
# container, that is gone the moment the run ends. That log is the ONLY discriminator between a
# recoverable context-overflow (HTTP 500, server survives) and a server death via
# ggml_abort/SIGSEGV (server gone, every later request RemoteDisconnected) -- without it, a
# mid-eval generator death is undiagnosable after the fact, one hop downstream of where that
# logging chain was built. `setdefault` so an operator override still wins. Set here, at the top
# of the OUTER script, before `run_env = os.environ.copy()` below -- child processes (the swarm,
# and whatever it spawns per game) inherit this through normal environment inheritance.
os.environ.setdefault("CARNOT_ARC_SERVER_LOG_DIR", "/kaggle/working")

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
# LOUD ON MISSING (2026-08-08 adversarial review, Gaps finding 6): a bare next() over this
# rglob raised a message-free StopIteration at import time -- a missing or re-laid-out dataset
# killed the run before any game, with the kernel still printing "complete" downstream. list()
# the hits first so a missing dataset gets one clear, actionable line instead of a bare
# traceback nobody reading the tail of a 12h log would recognize.
_carnot_hits = list(inp.rglob("carnot/agentic/arc_competition_agent.py"))
if not _carnot_hits:
    print(
        "FATAL: carnot/agentic/arc_competition_agent.py not found under /kaggle/input -- "
        "attach the carnot package dataset to this kernel and re-run.",
        flush=True,
    )
    raise SystemExit("carnot package dataset not attached")
carnot = _carnot_hits[0].parents[2]
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

# SEEDED TOOL-LOOP REPAIR (operator-enabled 2026-08-17, REQ-ARC-WMTE-6470). When an induced
# engine is rejected by the trust layer AND its cell_recall is below the 0.6 gate, re-run
# induction ONCE as a tool-calling loop seeded with the failed engine, instead of accepting a
# useless engine. Measured on 4 seeded catastrophic cells: 2 convert outright to recall 1.0 /
# accuracy 1.0 and flip trust reject -> accept (sb26, tail 1.0, memorisation scan clean; sp80,
# scan clean, no scoreable tail exists on that window), 2 return their seed unchanged. Stated as
# a conversion rate on purpose -- the arithmetic mean of +0.44 describes no cell in the set.
# COST OF A MISS, accepted: ~9-11 min and one of two per-game slots, because the counter
# increments when the repair FIRES, before the outcome exists. Bounded by the turn cap below.
# SAFE BY CONSTRUCTION, and confirmed empirically at 7 of 7: the loop is seeded with the failed
# engine and cannot return a candidate with more visible mismatches, so no run regressed below
# its seed. A trust-ACCEPTED engine is never re-rolled (`decide_resample` returns
# `downstream_accepted_engine` first), so this can only act on engines the pipeline was going to
# discard anyway.
# THIS IS AN ENGINE-QUALITY LEVER, NOT A LEVELS LEVER. The induced goal predicate has never
# fired on a real win state (0 of 31), so levels are won by exploration, not by planning. Expect
# better engines, not more levels.
# Set HERE rather than as a code default deliberately: the evidence is scored-path evidence. The
# conductor runs orders of magnitude more inductions, where ~10 min per miss is a different
# trade-off that nobody has measured. Code default stays OFF and its tests still pin that.
os.environ["CARNOT_ARC_INDUCE_TOOL_LOOP"] = "repair"
# Turn cap 12 -> 8. Every win across both arms converged by turn 7 (unseeded 3,3,4,4,5,6; seeded
# 3,5,5,6,7). Seven runs burned the 12-turn cap at 17-24 min each and not one recovered. So 8
# preserves all 11 observed wins and cuts each cap-burner by a third.
os.environ["CARNOT_ARC_INDUCE_TOOL_TURNS"] = "8"

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
# Match the generator GGUF by name so an order-undefined rglob can't bind a stale/second .gguf.
# 2026-07-28: the name arm moved from "Qwen3.5-9B" to "gemma-4-31B" with the generator switch. The
# `or "Q4_K_M"` fallback arm is deliberately kept LAST and is NOT disambiguating on its own -- it
# matches any Q4_K_M file, so if two quantized GGUFs are ever attached it picks by rglob order.
# That is why the len>1 warning below exists and why the resolved name is printed: the attached
# dataset should contain exactly one .gguf.
# TWO GGUFs ARE NOW ATTACHED, AND THEY MUST BE TOLD APART EXPLICITLY, NOT BY LUCK.
# `iancblenke/carnot-gemma4-31b-it-gguf` (18.3 GB main weights) and
# `iancblenke/carnot-gemma4-31b-mtp-head` (491 MB draft head) both mount under /kaggle/input and
# both end in `.gguf`. The previous filter was `"gemma-4-31B" in name or "Q4_K_M" in name` over an
# order-undefined rglob:
#   * the head is named `mtp-gemma-4-31B-it-Q8_0.gguf`, so it MATCHES the first arm, and
#   * the `or "Q4_K_M"` arm matches any quantized file at all.
# So the main model was being selected by rglob order between two files that both matched. Binding
# the 491 MB head as the generator would load, serve, and answer nonsense -- a silent failure.
# Now: the head is identified POSITIVELY and excluded from the main-model candidates, and the main
# model must match its own canonical filename stem.
# 2026-08-16: generator moved gemma-4-31B -> Qwen3.8-27B. The structure above is unchanged and
# still matters -- the head is excluded POSITIVELY and the main model must match its own name, so
# neither is chosen by rglob order. Note no MTP draft head ships for Qwen3.8-27B today, so `_heads`
# is expected to be empty and MTP resolves off. That costs roughly 1.8x decode speed against
# gemma's speculative path, which is a known and accepted part of this swap. The head arm is kept
# rather than deleted so that attaching a Qwen3.8 draft head later needs no code change.
# `mtp-` rather than a model-specific head name. A head must be excluded from the main-model
# candidates NO MATTER WHICH MODEL IT DRAFTS FOR -- a leftover gemma head under /kaggle/input must
# not be bindable as the Qwen main model. Pinning the head string to one model reintroduced exactly
# the silent failure described above, and the order-independence test caught it.
_HEAD_SUBSTR = "mtp-"
_MAIN_SUBSTR = "Qwen3.8-27B-NVFP4"


# A draft head is marked by the mtp- PREFIX, case-insensitively -- never by mtp appearing
# anywhere in the name. Mirrors _is_mtp_head_file in arc_executable_world_model.py; this file
# cannot import it, because the kernel resolves the GGUF before the carnot package is importable.
#
# The distinction is load-bearing here, not pedantic. A SELF-DRAFTING model is named
# Qwen3.8-27B-NVFP4-MTP-HIGHEST.gguf -- it CONTAINS MTP- and it IS the 23 GB main model. The
# previous plain substring test was case-sensitive, so it classified that file correctly only
# because upstream happened to capitalise MTP. A lowercase rename would have made the only main
# candidate vanish, leaving gguf = None and the whole LLM tier silently dead.
def _is_head(name: str) -> bool:
    return name.lower().startswith(_HEAD_SUBSTR)


# VLLM BACKEND SELECTION (REQ-ARC-WMTE-6510, operator-directed 2026-08-18). Default OFF: with
# no safetensors model dir attached this whole block is inert and the GGUF path below runs
# exactly as before. When the NVFP4 safetensors dataset IS attached, select vLLM -- measured on
# this card at 651.8 tok/s aggregate (k=32, native FP4 + fp8 KV) against 228.3 for the best
# llama.cpp config and ~52 for what shipped, which takes the induction budget from does-not-fit
# to ~2.6h of the 11.5h cap.
#
# Resolve by rglob, never by fixed depth: Kaggle's dataset mount depth is NOT stable across runs
# (observed directly -- one run served the version dir, the next served its child).
_vllm_cfg = next((c for c in inp.rglob("config.json") if list(c.parent.glob("*.safetensors"))), None)
if _vllm_cfg is not None:
    os.environ["CARNOT_ARC_LLM_BACKEND"] = "vllm"
    os.environ["CARNOT_ARC_VLLM_MODEL_DIR"] = str(_vllm_cfg.parent)
    # The two environment repairs the benchmark proved necessary, both applied BEFORE the agent
    # starts so flashinfer's startup JIT of the native SM120 FP4 GEMM can compile and link:
    #   1. pip resolves an INCOHERENT CUDA stack (nvcc 13.3 against 13.0 headers) -> the wheels
    #      dataset pins the coherent 13.0.x five, and CUDA_HOME/PATH point at it.
    #   2. `ld` wants unversioned libcudart.so / libcuda.so; the wheels ship only .so.13 and the
    #      driver only .so.1 -> symlink both into a dir on LIBRARY_PATH, locating the driver via
    #      ldconfig rather than guessing a path.
    _cu = next((Path(b) / "nvidia" / "cu13" for b in
                ("/usr/local/lib/python3.12/dist-packages", "/usr/lib/python3/dist-packages")
                if (Path(b) / "nvidia" / "cu13" / "bin" / "nvcc").exists()), None)
    if _cu is not None:
        os.environ["CUDA_HOME"] = os.environ["CUDA_PATH"] = str(_cu)
        os.environ["PATH"] = f"{_cu}/bin:" + os.environ.get("PATH", "")
        _ldl = Path("/kaggle/working/ldlinks"); _ldl.mkdir(exist_ok=True)
        _driver = next((l.split()[-1] for l in subprocess.run(
            ["ldconfig", "-p"], capture_output=True, text=True).stdout.splitlines()
            if "libcuda.so" in l), None)
        for _n, _src in (("libcudart.so", _cu / "lib" / "libcudart.so.13"),
                         ("libcuda.so", Path(_driver) if _driver else None)):
            if _src and Path(_src).exists() and not (_ldl / _n).exists():
                (_ldl / _n).symlink_to(_src)
        os.environ["LIBRARY_PATH"] = f"{_ldl}:{_cu}/lib:" + os.environ.get("LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{_cu}/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
    print(f"LLM BACKEND: vllm (model={_vllm_cfg.parent}, cuda_home={_cu})", flush=True)
else:
    print("LLM BACKEND: llama.cpp (no safetensors model dir attached)", flush=True)

_all_ggufs = list(inp.rglob("*.gguf"))
_heads = [g for g in _all_ggufs if _is_head(g.name)]
_mains = [g for g in _all_ggufs if not _is_head(g.name) and _MAIN_SUBSTR in g.name] or [
    g for g in _all_ggufs if not _is_head(g.name)
]
gguf = _mains[0] if _mains else None
mtp_head = _heads[0] if _heads else None
if len(_mains) > 1:
    print(f"LLM TIER WARNING: {len(_mains)} candidate MAIN GGUFs under /kaggle/input, using {gguf.name}; all={[g.name for g in _mains]}", flush=True)
if len(_heads) > 1:
    print(f"LLM TIER WARNING: {len(_heads)} candidate MTP HEAD GGUFs under /kaggle/input, using {mtp_head.name}; all={[g.name for g in _heads]}", flush=True)
print(f"LLM TIER GGUF RESOLUTION: main={gguf.name if gguf else None} mtp_head={mtp_head.name if mtp_head else None} (all={[g.name for g in _all_ggufs]})", flush=True)

if server and gguf:
    run_server = Path("/kaggle/working/llama-server")
    shutil.copy2(server, run_server)
    os.chmod(run_server, 0o755)

    # PRE-LAUNCH PROBE (2026-08-18). v23's generator died with COMPLETELY EMPTY stderr, and the
    # kernel's own guess -- "likely OOM" -- was refuted by its own fit line (needed 52168 MiB of
    # 97250 free). An empty stderr means the process never got far enough to log, which a real OOM
    # never does. Everything checkable locally was eliminated (all shared objects resolve from the
    # dataset directory with system paths stripped), so the remaining candidates are environmental
    # and NOT observable from a dev box: the CUDA runtime major version against this image's
    # driver, most likely.
    #
    # So stop guessing and make the run say it. `--version` loads the full shared-object graph and
    # touches the CUDA runtime, but allocates no model, so it separates "the binary cannot start
    # here" from "the model did not fit" -- the exact ambiguity that cost a submission cycle.
    # Cheap (well under a second), and it prints whether or not the launch later succeeds.
    try:
        _probe = subprocess.run(
            [str(run_server), "--version"],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "LD_LIBRARY_PATH": f"{server.parent}:" + os.environ.get("LD_LIBRARY_PATH", "")},
        )
        print(
            f"LLM TIER SERVER PROBE: rc={_probe.returncode} "
            f"stdout={_probe.stdout.strip()[:200]!r} stderr={_probe.stderr.strip()[:600]!r}",
            flush=True,
        )
    except Exception as _exc:  # never let a diagnostic break the run it is diagnosing
        print(f"LLM TIER SERVER PROBE: raised {_exc!r}", flush=True)
    try:
        _drv = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60,
        )
        print(f"LLM TIER DRIVER: {_drv.stdout.strip()[:200]!r} rc={_drv.returncode}", flush=True)
    except Exception as _exc:
        print(f"LLM TIER DRIVER: raised {_exc!r}", flush=True)
    os.environ["LD_LIBRARY_PATH"] = f"{server.parent}:" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["CARNOT_LLAMA_SERVER"] = str(run_server)
    os.environ["CARNOT_ARC_GGUF_PATH"] = str(gguf)
    # 16GB P100/T4 (2026-06-21, evidence-backed by the carnot-arc-binary-smoke probe): the MTP
    # self-draft loads a 2nd ~5.9GB copy of the model (probe: 11.8GB used vs 5.9GB MTP-off) for NO
    # throughput gain on this GPU (probe: 27.5 vs 25.3 tok/s, MTP-off slightly FASTER). Disable it to
    # free ~5.8GB of VRAM for KV headroom. Speculative decoding is exact, so output quality is
    # unchanged. (The frozen stack's MTP speedup was validated on the iGPU, not the P100.)
    #
    # SUPERSEDED 2026-07-28 BY THE GENERATOR SWITCH, kept because it is the provenance of every
    # pre-switch VRAM number: the paragraph below describes Qwen3.5-9B-MTP, and the whole reason
    # this line exists (MTP loads a second copy of the weights) no longer applies -- gemma-4-31B-it
    # is not an MTP model, its GGUF declares no nextn_predict_layers, so `--spec-type draft-mtp`
    # is not something the agent would emit for it anyway. The line stays as belt-and-braces in
    # case CARNOT_ARC_GGUF_PATH is ever pointed back at a genuine MTP model here.
    # The CURRENT envelope for the shipped generator (gemma-4-31B-it Q4_K_M, mtp off, q8 KV,
    # 4 slots, measured on an RTX 3090 2026-07-28) is:
    #     MiB = 18940.7 + 0.050293*n_ctx + 206.83*slots
    # i.e. 21416 MiB at n_ctx 32768 and 23888 MiB at n_ctx 81920. The per-cell KV term is ~2x the
    # 9B's, which is why the local free-VRAM guard had to be refit alongside the model swap.
    #
    # NOTE FOR ANYONE READING THE VRAM ENVELOPE: because MTP is OFF here, the published envelope
    # (`MiB = 10547 + 0.02519*n_ctx + 206.83*slots`, exp5866) does NOT describe this launch -- it
    # was fit with `--spec-type draft-mtp` ON and over-predicts the scored footprint by ~6.1 GB.
    # Directly measured mtp-OFF per-PID residency on an RTX 3090 (2026-07-27): 5950 MiB at
    # n_ctx=16384, 7380 MiB at n_ctx=81920, i.e. the n_ctx fix costs ~1430 MiB here, not the
    # ~1668 MiB the mtp-on envelope predicts. Peak residency with 4 concurrent full-budget
    # requests in flight equalled idle residency to the MiB, because llama.cpp preallocates the
    # whole `-c` pool at load.
    # MTP IS ON FOR THE SCORED RUN (2026-07-28, operator-authorised: "when we submit we will want
    # MTP enabled for speed when running on the Kaggle 96G GPU hardware"), and it is enabled from
    # the CANONICAL SCORED CONSTANT rather than a re-typed literal here.
    #
    # WHAT CHANGED THE ANSWER. The paragraph preserved above concluded MTP-off on two premises,
    # both now measured false for this model: (a) that gemma-4-31B has no MTP -- it does, via a
    # SEPARATE 491 MiB head declaring `gemma4-assistant`; and (b) that enabling it would load the
    # 18.3 GB weights twice -- it does not, it loads the head, +862 MiB at n_ctx 32768 and
    # +1290 MiB at 81920. Measured with THE BINARY THIS KERNEL BUNDLES: 35.88 -> 50.16 tok/s
    # (1.398x), 319/576 drafted tokens accepted.
    #
    # WHY A SEPARATE CONSTANT FROM THE LOCAL DEFAULT. On a 24 GB dev 3090 MTP-on forces ~14 FFN
    # blocks to system RAM to fit, and that offload costs more decode than MTP returns -- a NET
    # LOSS locally. On the 96 GB scored card no offload is needed, so it is a pure win. Two
    # different hardware answers, so two named constants; `ARC_LIVE_GENERATOR_MTP_DEFAULT` stays
    # "0" and is still the right local answer.
    #
    # SAFE WHEN THE HEAD IS MISSING. If the mtp-head dataset is not attached, `mtp_head` is None,
    # we do not set CARNOT_ARC_MTP_GGUF_PATH, and `_ensure_server()` drops the MTP flags entirely
    # rather than passing the main weights as the draft -- because llama.cpp ACCEPTS that, warns,
    # and then serves with speculation silently disabled, which is indistinguishable from working
    # MTP except by tok/s.
    from carnot.agentic.arc_executable_world_model import (
        ARC_LIVE_GENERATOR_MTP_DEFAULT,  # noqa: F401  (local default; kept for provenance/audit)
        ARC_LIVE_GENERATOR_MTP_SCORED_DEFAULT,
    )

    # THE SCORED MTP STATE IS `constant AND head-present`, AND BOTH HALVES ARE LOAD-BEARING.
    #
    # This line used to be `os.environ["CARNOT_ARC_MTP"] = "1" if mtp_head else "0"` -- an
    # UNCONDITIONAL assignment, which made the `os.environ.get(..., SCORED_DEFAULT)` below
    # structurally unreachable: the key was always set, so the default was never consulted and
    # `ARC_LIVE_GENERATOR_MTP_SCORED_DEFAULT` had ZERO runtime effect in the kernel. Head presence
    # alone decided. The comment above claimed MTP "is enabled from the canonical scored constant",
    # which was simply not what the code did.
    #
    # Why that mattered rather than being cosmetic: flipping the constant to "0" -- the documented
    # way to turn scored MTP off -- would NOT have turned it off. It would only have desynced
    # `SUBMITTED_AGENT_CONFIG["frozen_generator"]["mtp"]` and the exp4744/exp4754 readiness gates
    # (all three of which DO read the constant) from what the kernel actually launches. The knob
    # would have reported a change it did not make.
    #
    # AND-ing is the honest composition of the two facts. The constant is the OPERATOR'S INTENT
    # ("we want MTP for the scored run"); head presence is a PHYSICAL PRECONDITION (no head, no
    # speculation possible -- and drafting against the main weights is the silent-degradation trap,
    # not a fallback). Either one being false correctly yields MTP off.
    _scored_mtp_intent = ARC_LIVE_GENERATOR_MTP_SCORED_DEFAULT != "0"
    _mtp = bool(_scored_mtp_intent and mtp_head)
    os.environ["CARNOT_ARC_MTP"] = "1" if _mtp else "0"
    if mtp_head:
        os.environ["CARNOT_ARC_MTP_GGUF_PATH"] = str(mtp_head)
    if not _scored_mtp_intent:
        print("LLM TIER: scored MTP is DISABLED by ARC_LIVE_GENERATOR_MTP_SCORED_DEFAULT "
              f"={ARC_LIVE_GENERATOR_MTP_SCORED_DEFAULT!r}. Speculative decoding will NOT be "
              "requested even though a head is present.", flush=True)
    elif not mtp_head:
        print("LLM TIER WARNING: MTP head GGUF not found under /kaggle/input -- attach "
              "iancblenke/carnot-gemma4-31b-mtp-head. Running WITHOUT speculative decoding "
              "(~1.4x slower decode); NOT falling back to drafting against the main weights, "
              "which llama.cpp would accept and then silently ignore.", flush=True)

    # PARITY ASSERTION: what this kernel is about to launch must match what the submission
    # DECLARES it launches. `SUBMITTED_AGENT_CONFIG["frozen_generator"]` is what the readiness
    # gates (exp4744, exp4754) and the parity test assert against, and it is derived from the
    # scored constant -- so if the kernel's resolved state ever diverges from it, every one of
    # those gates is green while describing a different run. Printed rather than raised when the
    # cause is a missing head: that is a degraded-but-valid scored run, and aborting the whole
    # evaluation over a lost 1.4x would be a worse outcome than running slower.
    try:
        from carnot.agentic.arc_competition_agent import SUBMITTED_AGENT_CONFIG as _SAC
        _declared_mtp = bool(_SAC.get("frozen_generator", {}).get("mtp"))
        if _declared_mtp != _mtp:
            print(f"LLM TIER MTP DECLARED-VS-ACTUAL MISMATCH: SUBMITTED_AGENT_CONFIG declares "
                  f"mtp={_declared_mtp} but this kernel resolved mtp={_mtp} "
                  f"(scored_constant={ARC_LIVE_GENERATOR_MTP_SCORED_DEFAULT!r}, "
                  f"head_present={bool(mtp_head)}). The readiness gates describe a configuration "
                  "this run is NOT using.", flush=True)
    except Exception as _e:
        print(f"LLM TIER: could not cross-check declared MTP state ({_e!r})", flush=True)
    # BUDGET EXPANSION FOR THE SCORED RUN ONLY (2026-08-07, operator directive). The local
    # defaults -- max_tokens=4096 (LocalGGUFProposer, pinned by
    # tests/python/test_arc_submitted_agent_parity.py, NOT touched here), induce timeout floor
    # 600s -- were sized under the 16GB/P100-parity assumption
    # (docs/research-notes/arc-agi3-cuda-submission-runbook-2026-06-30.md), which is void: the
    # scored card is now requested as a 96GB-class NvidiaRtxPro6000 (kernel-metadata.json). Raise
    # both env-only, for this kernel's launch alone. `setdefault` so an operator override set some
    # other way is not clobbered. The VRAM-fit check a few lines below prints a WARNING (does not
    # abort) if the actual attached card can't hold the raised n_ctx -- machine_shape is a
    # free-form string the SDK cannot validate locally, so that check is the real safety net here,
    # not a guess at this point in the code. 1200s covers the slowest local induce observed
    # (572s at n_ctx 32768 on a 3090, single-stream) with margin for the live path's 4 shared
    # slots running slower per-request; a timeout firing silently degrades the agent to LLM-off
    # rather than erroring, so under-shooting this is a silent-failure risk, not just slowness.
    # RAISED AGAIN 2026-08-16 for the Qwen3.8-27B generator, which is a reasoning model and
    # writes far longer completions than gemma-4-31B. Seven measured inductions ran 49,244 to
    # 83,544 tokens (median 61,284), so 8192 covers 0 of 7 and 65,536 covers only 4 of 7 -- a
    # truncated completion is an unusable engine, not a slow one. 131,072 covers all seven with
    # margin for the tail we have not sampled, and stays inside the model's own trained context
    # (qwen35.context_length = 262144). max_tokens is a CAP: unused headroom costs pool VRAM,
    # never wall clock. Timeout follows the same logic -- the worst measured Qwen3.8 induction is
    # ~1053s on this card at one stream, and a timeout that fires degrades the agent to LLM-off
    # silently, so it is set well clear of that rather than close to it.
    os.environ.setdefault("CARNOT_ARC_INDUCE_MAX_TOKENS", "131072")
    os.environ.setdefault("CARNOT_ARC_INDUCE_TIMEOUT", "2400")
    # READ the context-pool size and completion budget from the SHIPPED defaults instead of
    # repeating literals here. The old code printed "ctx=16384" and probed with -c 16384 as
    # hardcoded strings; if the agent's own default had moved, the probe would have validated
    # a configuration the agent never used -- and validated it as HEALTHY. That is the
    # measure-one-thing-ship-another shape of the 0.08 incident, in the diagnostic itself.
    from carnot.agentic.arc_executable_world_model import (
        _INDUCE_WORST_CASE_PROMPT_TOKENS,
        _LLAMA_SERVER_DEFAULT_SLOTS,
        _llama_server_slots,
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
    # HOST RAM, next to the GPU line (2026-08-08 adversarial review, Gaps finding 5).
    # MAX_ACTIONS=2000 in arc_competition_agent.py is sized on an ADMITTEDLY UNCONFIRMED 16 GiB
    # host-RAM assumption (the competition framework retains every frame for the whole episode,
    # so RAM grows with action count). Nothing before this print has ever recorded what RAM the
    # scored container actually has -- the same "measure, don't infer" gap the GPU line above
    # closed for VRAM. REPORTS, DOES NOT ABORT, same reasoning as the VRAM fit check below.
    try:
        with open("/proc/meminfo") as _mf:
            _mem_total_line = next((_l for _l in _mf if _l.startswith("MemTotal:")), None)
        print(f"HOST RAM: {(_mem_total_line or 'MemTotal line not found').strip()}", flush=True)
    except Exception as _me:
        print(f"HOST RAM: unavailable ({_me!r})", flush=True)
    # THE FIT CHECK ON THE **SCORED** CARD. `_generator_cuda_min_free_mb()` is the project's VRAM
    # arithmetic, but on this path it is otherwise NEVER EVALUATED: `_generator_server_and_env()`
    # returns at priority 1 on CARNOT_LLAMA_SERVER (which this kernel always sets), so the guard,
    # the auto-fit and the fit invariant all protect the dev box only. Nothing checked whether the
    # configuration about to launch actually FITS the hardware Kaggle handed us.
    #
    # That gap is not hypothetical. `machine_shape` in kernel-metadata.json is a free-form string
    # the SDK cannot validate locally, so which card the scored run gets is not knowable before it
    # runs. At n_ctx 81920 with MTP on, the requirement is ~25.2 GB -- more than a 24 GB-class card
    # HAS. On such a card the server would cudaMalloc-fail, `_ensure_server()` would burn its retry
    # budget, and the agent would proceed LLM-OFF while still reporting itself as the LLM-on scored
    # path. This prints the comparison so the failure is legible in the log instead of being
    # inferred later from a suspiciously low score.
    #
    # CORRECTION 2026-08-08 (REQ-ARC-WMTE-6227): the illustrative "81920 / ~25.2 GB" pair above is
    # historical. `_INDUCE_WORST_CASE_PROMPT_TOKENS` moved 15767 -> 22352 (a stale-constant fix),
    # which raises the shipped n_ctx to 106496 and the no-offload requirement past 26.6 GB. The
    # print below reads `_generator_cuda_min_free_mb()` live, so the ACTUAL number on any given run
    # is always current; only this illustrative prose example is now out of date.
    #
    # REPORTS, DOES NOT ABORT. A wrong-but-running submission is worth more than no submission, and
    # the remedy (a different machine_shape, a lower n_ctx, MTP off) is an operator decision that
    # cannot be taken from inside the scored run.
    try:
        from carnot.agentic.arc_executable_world_model import _generator_cuda_min_free_mb
        _need = _generator_cuda_min_free_mb(0, _mtp)
        _free_smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
        _first = (_free_smi.stdout.strip().splitlines() or [""])[0]
        _free_mb, _total_mb = (int(x.strip()) for x in _first.split(",")[:2])
        _fits = _need <= _free_mb
        print(f"LLM TIER VRAM FIT: needs {_need} MiB (n_ctx={_ctx}, mtp={_mtp}, 0 FFN offload), "
              f"card has {_free_mb} MiB free of {_total_mb} MiB total -> "
              f"{'FITS' if _fits else 'DOES NOT FIT'}", flush=True)
        if not _fits:
            print("LLM TIER VRAM WARNING: the scored card CANNOT hold this configuration. The "
                  "generator will cudaMalloc-fail and the agent will run LLM-OFF while still "
                  "reporting itself as the LLM-on scored path. Remedies (operator-side): a larger "
                  "machine_shape, a lower CARNOT_ARC_INDUCE_N_CTX, or "
                  "ARC_LIVE_GENERATOR_MTP_SCORED_DEFAULT='0'.", flush=True)
    except Exception as _ve:
        print(f"LLM TIER VRAM FIT: could not evaluate ({_ve!r})", flush=True)
    print(f"LLM TIER RESOLVED: server={run_server} gguf={gguf.name} mtp={_mtp} ctx={_ctx} "
          f"max_tokens={_maxtok} slots_expected={_llama_server_slots()} "
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
        # `--model-draft` is the HEAD, never `gguf` (the main weights). This probe used to pass
        # the main file, i.e. it validated as HEALTHY exactly the configuration in which
        # speculation is silently disabled -- the measure-one-thing-ship-another shape this whole
        # probe exists to prevent, inside the probe itself.
        if _mtp and mtp_head:
            _args += ["--spec-type", "draft-mtp", "--model-draft", str(mtp_head)]
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

                # Read the RESOLVER, not the constant. `_default_induce_n_ctx()` sizes
                # the pool from the resolver, so a probe reading the raw constant would
                # validate K=4 while the pool was built for K=8 -- the exact
                # probe-one-thing-ship-another gap the comment above forbids.
                _K = int(_llama_server_slots())

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
        # MTP ENGAGEMENT MUST BE READ FROM THE POSITIVE MARKER, NOT INFERRED FROM HEALTH.
        # When `--spec-type draft-mtp` is given a draft the runtime cannot use, llama.cpp does NOT
        # fail: it warns and serves normally with speculation silently disabled, so /health 200 and
        # correct output are consistent with MTP being completely off. The ONLY in-log evidence
        # that speculation is actually wired is the marker below; the only other evidence is a
        # tok/s delta we cannot measure here without a matched control run. So we assert on the
        # marker and say plainly when it is absent, rather than letting "healthy" imply "fast".
        _mtp_log = ""
        try:
            _mtp_log = Path("/kaggle/working/llm_probe.err").read_text()
        except Exception:
            _mtp_log = ""
        _mtp_engaged = "adding speculative implementation 'draft-mtp'" in _mtp_log
        _mtp_degraded = ("doesn't contain MTP layers" in _mtp_log
                         or "no implementations specified for speculative decoding" in _mtp_log)
        if _mtp and not _mtp_engaged:
            print("LLM MTP NOT ENGAGED: --spec-type draft-mtp was requested but the server never "
                  "logged \"adding speculative implementation 'draft-mtp'\". llama.cpp serves "
                  "normally in this state with speculation SILENTLY DISABLED, so this is NOT "
                  f"visible as an error anywhere else. degradation_warnings_seen={_mtp_degraded} "
                  f"draft={mtp_head}. Expect ~1.4x slower decode than planned.", flush=True)
        elif _mtp:
            print("LLM MTP ENGAGED: server logged the draft-mtp speculative implementation "
                  f"(draft={mtp_head.name if mtp_head else None}).", flush=True)
        if _ok:
            print(f"LLM GENERATOR HEALTHY -- loaded on GPU, /health ok (generator tier ENGAGED); "
                  f"mtp_requested={_mtp} mtp_engaged={_mtp_engaged}; "
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
          f"ONLY (server={server}, gguf={gguf}). Verify the carnot-llamacpp-mtp-binary + "
          "carnot-gemma4-31b-it-qat-gguf datasets are attached (that one dataset carries BOTH "
          "the target weights and the matching MTP drafter).",
          flush=True)
"""

# The framework bind is SPLIT OUT of the diagnostics block above (2026-08-11) so the preview
# branch below can execute the diagnostics WITHOUT the framework import. `from agents.agent
# import Agent` only resolves inside the competition rerun (the framework is copied to
# /kaggle/working and my_agent.py runs from within it); in preview mode that package does not
# exist, which is exactly why the diagnostics have never appeared in any log a human can read:
# they lived downstream of an import that only happens in the branch whose log is hidden.
AGENT_BIND_TAIL = r"""
from agents.agent import Agent
from carnot.agentic.arc_competition_agent import make_carnot_agent

# the verifier-routed cascade (graph-explore -> E3 induction via the bundled Qwen).
# registered under "carnotagent" in the rewritten agents/__init__.py below.
CarnotAgent = make_carnot_agent(Agent)
"""
Path("/kaggle/working/my_agent.py").write_text(AGENT_SRC + AGENT_BIND_TAIL)

# PREVIEW-MODE GENERATOR CONFIG A/B (2026-08-11). This source runs ONLY in the preview
# branch below. It never runs on the scored path, so it cannot change a scored result.
# It measures three server levers on the real scored hardware at zero submission cost.
PREVIEW_AB_SRC = r'''
"""Generator config A/B harness. Preview (Save and Run) branch only.

WHY THIS EXISTS. The scored rerun log is not readable. The preview log is readable
through `kaggle kernels output`. Preview runs on the same machine shape, with the same
datasets and the same GPU. So a config A/B here measures the real scored hardware.

WHAT IT MEASURES. Three levers, one line per arm:
  1. slots      -- the server slot count (a server relaunch, `--parallel N`).
  2. concurrency -- K simultaneous requests to one server (a request-side change).
  3. n_ctx      -- the shared context pool size (a server relaunch, `-c N`).
The concurrency ladder is the important one. The project ASSUMES N samples cost about
one sample's wall clock under continuous batching. Nobody has measured that. K=1 vs
K=4 vs K=8 on an identical prompt measures it directly.

THE PROMPTS ARE SYNTHETIC, AND THAT IS A REAL LIMIT. `kernel-metadata.json` attaches
three datasets: the carnot code package, the llama.cpp binary, and the GGUF weights.
None of them carries an ARC transition corpus, and this harness never reads game
source. So the prompts are fixed synthetic digit rows, trimmed to the measured
worst-case induce prompt length. These numbers describe SERVING BEHAVIOUR at a
realistic prompt size. They do NOT describe induction quality.
"""

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Soft budget. Every step checks it before starting. The outer kernel also hard-kills
# this process a little later, so a wedged server load can never brick the preview run.
AB_BUDGET_S = float(os.environ.get("CARNOT_AB_BUDGET_S", "2400"))
DEADLINE = time.time() + AB_BUDGET_S
# Do not start a session that cannot finish. A cold load of the weights costs minutes,
# so an aborted session spends the budget and returns nothing. Scaled from the budget,
# not a fixed number: a fixed 300s reserve blocks every session at a small budget.
SESSION_MIN_RESERVE_S = min(300.0, AB_BUDGET_S * 0.125)

# Short completions keep the budget for the arms themselves. We compare arms against
# each other, so an identical small budget is enough. Raise it to study long decodes.
N_PREDICT = int(os.environ.get("CARNOT_AB_N_PREDICT", "256"))

inp = Path("/kaggle/input")
_hits = list(inp.rglob("carnot/agentic/arc_competition_agent.py"))
if _hits:
    sys.path.insert(0, str(_hits[0].parents[2]))

# Read the shipped defaults from the code the agent runs. Re-typed literals drift, and
# then the A/B measures a configuration nothing ships. The fallback keeps the harness
# alive if the code dataset is missing, and the START line names which path was taken.
SLOTS = 4
WORST_PROMPT_TOKENS = 22352
MTP_SCORED = "1"
HEAD_SUBSTR = "mtp-gemma-4-31B-it"
CONSTS_SOURCE = "shipped"
try:
    from carnot.agentic.arc_executable_world_model import (
        ARC_LIVE_GENERATOR_MTP_HEAD_SUBSTR as HEAD_SUBSTR,
    )
    from carnot.agentic.arc_executable_world_model import (
        ARC_LIVE_GENERATOR_MTP_SCORED_DEFAULT as MTP_SCORED,
    )
    from carnot.agentic.arc_executable_world_model import (
        _INDUCE_WORST_CASE_PROMPT_TOKENS as WORST_PROMPT_TOKENS,
    )
    from carnot.agentic.arc_executable_world_model import (
        _llama_server_slots as _slots_resolver,
    )
    from carnot.agentic.arc_executable_world_model import _default_induce_n_ctx

    BASE_CTX = int(_default_induce_n_ctx())
    # Slots come from the RESOLVER, not the raw constant, so an operator who sets
    # CARNOT_ARC_LLAMA_SERVER_SLOTS gets a harness that measures the K the pool was
    # actually sized for. Reading the constant here while `_default_induce_n_ctx()`
    # reads the resolver is the probe-one-thing-ship-another gap this file already
    # records having hit once.
    SLOTS = int(_slots_resolver())
except Exception as _ce:
    # The fallback values above are a GUESS at the shipped configuration. They match it
    # as of 2026-08-11, but nothing keeps them in step. The START line says so out loud,
    # because an A/B that measures a configuration nothing ships is worse than no A/B.
    CONSTS_SOURCE = "fallback:" + repr(_ce)
    BASE_CTX = int(-(-(SLOTS * (WORST_PROMPT_TOKENS + 4096)) // 4096) * 4096)

# Resolve the weights the same way the agent does. The MTP draft head and the main
# weights both end in `.gguf`, so the head is identified first and then excluded.
_all_ggufs = list(inp.rglob("*.gguf"))
_heads = [g for g in _all_ggufs if HEAD_SUBSTR in g.name]
_mains = [g for g in _all_ggufs if HEAD_SUBSTR not in g.name and "gemma-4-31B" in g.name] or [
    g for g in _all_ggufs if HEAD_SUBSTR not in g.name
]
GGUF = _mains[0] if _mains else None
MTP_HEAD = _heads[0] if _heads else None
SERVER_BIN = next(iter(inp.rglob("llama-server")), None)

# Mirror the scored MTP decision: operator intent AND a present head. The A/B must run
# the configuration the scored path runs, or it measures the wrong server.
MTP_ON = bool(str(MTP_SCORED) != "0" and MTP_HEAD)

# A short synthetic prompt. It never fills the pool, so the concurrency ladder on this
# prompt isolates batching throughput from context-pool admission.
SMALL_PROMPT = ("Row: " + " ".join("1234567890" for _ in range(20)) + "\n") * 2


def remaining():
    return DEADLINE - time.time()


def post_json(url, payload, timeout):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.load(resp)


def health(port, timeout=5):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def stop(proc):
    try:
        proc.terminate()
        proc.wait(timeout=20)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def vram_note(n_ctx, parallel):
    """Predicted VRAM against measured free VRAM. Reports, never aborts.

    A relaunch that cannot fit still runs. It then fails to become healthy and the
    session records that. This line tells the reader which of the two happened.
    """
    try:
        from carnot.agentic.arc_executable_world_model import (
            _VRAM_PER_SLOT_MIB,
            _predicted_generator_vram_mib,
        )

        need = _predicted_generator_vram_mib(n_ctx, 0, MTP_ON)
        if parallel:
            need += _VRAM_PER_SLOT_MIB * (int(parallel) - int(SLOTS))
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        free_mb = int((smi.stdout.strip().splitlines() or ["0"])[0].strip())
        return f"vram_need_mib={need:.0f} vram_free_mib={free_mb} fits={need <= free_mb}"
    except Exception as exc:
        return f"vram_check_unavailable={exc!r}"


def launch(port, n_ctx, parallel):
    """Start one llama-server and wait for /health. Return (proc, log) or (None, None)."""
    args = [
        str(RUN_SERVER),
        "-m",
        str(GGUF),
        "-ngl",
        os.environ.get("CARNOT_ARC_NGL", "999"),
        "-c",
        str(n_ctx),
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
    ]
    if parallel:
        args += ["--parallel", str(parallel)]
    if MTP_ON:
        args += ["--spec-type", "draft-mtp", "--model-draft", str(MTP_HEAD)]
    log = open(f"/kaggle/working/ab_server_{port}.err", "w")
    proc = subprocess.Popen(args, stdout=log, stderr=log)
    # Cap the wait. One slow cold load must not eat the budget of the later sessions.
    wait_s = min(420.0, max(30.0, remaining() - 120.0))
    end = time.time() + wait_s
    while time.time() < end:
        if proc.poll() is not None:
            break
        if health(port, timeout=3):
            return proc, log
        time.sleep(2)
    stop(proc)
    try:
        log.close()
    except Exception:
        pass
    return None, None


def n_tokens(port, text):
    _s, body = post_json(f"http://127.0.0.1:{port}/tokenize", {"content": text}, 180)
    return len(body.get("tokens") or [])


def build_worst_prompt(port):
    """Synthetic text trimmed to the measured worst-case induce prompt length.

    We trim against the server's own tokenizer. A guessed prompt size tests the wrong
    thing: a prompt larger than the pool admits fails for a reason we did not choose.
    """
    row = "Row: " + " ".join("1234567890" for _ in range(60)) + "\n"
    text = row * 40
    count = None
    try:
        count = n_tokens(port, text)
        while count > WORST_PROMPT_TOKENS and len(text) > len(row):
            text = text[: -len(row)]
            count = n_tokens(port, text)
    except Exception as exc:
        print(f"PREVIEW AB NOTE | tokenize failed, prompt left untrimmed ({exc!r})", flush=True)
    return text, count


def run_arm(name, port, k, prompt):
    """Fire k identical requests at once. Print one fixed-format result line."""
    # cache_prompt is OFF on purpose. With it on, a later arm reuses an earlier arm's
    # prefill and looks faster than it is. Every arm must pay the same prefill cost.
    payload = {
        "prompt": prompt,
        "n_predict": N_PREDICT,
        "temperature": 0.3,
        "cache_prompt": False,
    }

    def one(_i):
        try:
            status, body = post_json(f"http://127.0.0.1:{port}/completion", payload, 900)
            timings = body.get("timings") or {}
            gen = timings.get("predicted_n")
            if not isinstance(gen, int):
                gen = (body.get("usage") or {}).get("completion_tokens")
            return {
                "status": status,
                "stop_type": body.get("stop_type"),
                "generated": gen,
            }
        except Exception as exc:
            # Read the body. The 500 says the context size was exceeded, which is the
            # answer. Earlier probes threw that text away and left the cause unknown.
            try:
                detail = exc.read().decode("utf-8", "replace")[:160]
            except Exception:
                detail = str(exc)[:160]
            return {
                "status": f"{type(exc).__name__}:{getattr(exc, 'code', '')}",
                "generated": 0,
                "error": detail,
            }

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=k) as pool:
        results = list(pool.map(one, range(k)))
    wall = time.time() - t0
    tok = sum((r.get("generated") or 0) for r in results)
    tok_s = (tok / wall) if wall > 0 else 0.0
    # Truncation means HTTP 200 that stopped on the token limit well short of the
    # budget. The shared pool ran out. An HTTP status check cannot see this.
    truncated = any(
        r.get("status") == 200
        and r.get("stop_type") == "limit"
        and isinstance(r.get("generated"), int)
        and r["generated"] < N_PREDICT - 8
        for r in results
    )
    # A 200 with no token count reads as tok=0 in the fixed line, which looks like a
    # failure and is not one. Count those separately so the fixed line is never the
    # only evidence, and truncation cannot be judged for them either.
    unknown = sum(
        1
        for r in results
        if r.get("status") == 200 and not isinstance(r.get("generated"), int)
    )
    alive = health(port)
    print(
        f"PREVIEW AB | arm={name} | wall_s={wall:.2f} | tok={tok} | tok_s={tok_s:.2f} | "
        f"truncated={truncated} | alive={alive}",
        flush=True,
    )
    print(
        f"PREVIEW AB DETAIL | arm={name} | k={k} | n_predict={N_PREDICT} | "
        f"unknown_token_count={unknown} | results={results}",
        flush=True,
    )
    return alive


def session(label, port, n_ctx, parallel, ladder):
    """Launch one server, run its arms in order, then stop it."""
    if remaining() < SESSION_MIN_RESERVE_S:
        print(
            f"PREVIEW AB SKIP | session={label} | reason=budget_exhausted | "
            f"remaining_s={remaining():.0f} | needed_s={SESSION_MIN_RESERVE_S:.0f}",
            flush=True,
        )
        return
    print(
        f"PREVIEW AB SESSION | label={label} | n_ctx={n_ctx} | "
        f"parallel={parallel if parallel else 'default'} | mtp={MTP_ON} | "
        f"{vram_note(n_ctx, parallel)}",
        flush=True,
    )
    t0 = time.time()
    proc, log = launch(port, n_ctx, parallel)
    if proc is None:
        print(
            f"PREVIEW AB SKIP | session={label} | reason=server_never_became_healthy | "
            f"load_s={time.time() - t0:.0f} | log=/kaggle/working/ab_server_{port}.err",
            flush=True,
        )
        return
    print(
        f"PREVIEW AB SESSION READY | label={label} | load_s={time.time() - t0:.0f}",
        flush=True,
    )
    try:
        worst, worst_tok = build_worst_prompt(port)
        try:
            small_tok = n_tokens(port, SMALL_PROMPT)
        except Exception:
            small_tok = None
        print(
            f"PREVIEW AB PROMPT | label={label} | synthetic_worst_tokens={worst_tok} | "
            f"target={WORST_PROMPT_TOKENS} | synthetic_small_tokens={small_tok}",
            flush=True,
        )
        for kind, k in ladder:
            if remaining() < 60:
                print(
                    f"PREVIEW AB TIMEOUT | session={label} | reason=budget_exhausted_mid_session",
                    flush=True,
                )
                break
            alive = run_arm(
                f"{label}_k{k}_{kind}", port, k, worst if kind == "worst" else SMALL_PROMPT
            )
            if not alive:
                print(
                    f"PREVIEW AB SESSION DIED | label={label} | last_arm=k{k}_{kind} | "
                    f"log=/kaggle/working/ab_server_{port}.err",
                    flush=True,
                )
                break
    finally:
        stop(proc)
        try:
            log.close()
        except Exception:
            pass


def main():
    print(
        f"PREVIEW AB START | budget_s={AB_BUDGET_S:.0f} | constants={CONSTS_SOURCE} | "
        f"base_n_ctx={BASE_CTX} | default_slots={SLOTS} | "
        f"worst_prompt_tokens={WORST_PROMPT_TOKENS} | n_predict={N_PREDICT} | "
        f"mtp={MTP_ON} | gguf={GGUF.name}",
        flush=True,
    )
    print(
        "PREVIEW AB PROMPT SOURCE | synthetic. No ARC transition corpus is mounted, and "
        "this harness reads no game source. The numbers below describe serving behaviour "
        "at a realistic prompt size. They do not describe induction quality.",
        flush=True,
    )
    if CONSTS_SOURCE != "shipped":
        print(
            "PREVIEW AB CONFIG WARNING | the carnot code package did not import, so the "
            "slot count, worst-case prompt size, context pool size and MTP state above are "
            "HARDCODED GUESSES, not the shipped values. Treat every arm below as measuring "
            "the hardware, not the shipped configuration.",
            flush=True,
        )
    # Session 1 is the shipped configuration. It carries the full concurrency ladder.
    # The small-prompt arms isolate batching. The worst-prompt arms add pool pressure.
    session(
        "base",
        8946,
        BASE_CTX,
        None,
        [("small", 1), ("small", 4), ("small", 8), ("worst", 1), ("worst", 4), ("worst", 8)],
    )
    # Session 2 raises the slot count. Read this result with care: an explicit
    # `--parallel N` also DIVIDES the shared pool into N slices, so each slot gets far
    # fewer cells. A worst-prompt failure here is the divided pool, not the slot count.
    session(
        "slots8",
        8947,
        BASE_CTX,
        8,
        [("small", 1), ("small", 8), ("worst", 1), ("worst", 8)],
    )
    # Session 3 doubles the pool at the default slot count. It answers whether pool size
    # alone removes the worst-prompt pressure seen in session 1.
    session("ctx2x", 8948, BASE_CTX * 2, None, [("worst", 4), ("worst", 8)])
    print(f"PREVIEW AB DONE | elapsed_s={AB_BUDGET_S - remaining():.0f}", flush=True)


if SERVER_BIN and GGUF:
    RUN_SERVER = Path("/kaggle/working/llama-server-ab")
    try:
        shutil.copy2(SERVER_BIN, RUN_SERVER)
        os.chmod(RUN_SERVER, 0o755)
        os.environ["LD_LIBRARY_PATH"] = f"{SERVER_BIN.parent}:" + os.environ.get(
            "LD_LIBRARY_PATH", ""
        )
        main()
    except Exception as _exc:
        print(f"PREVIEW AB ERROR | {_exc!r}", flush=True)
else:
    print(
        f"PREVIEW AB ABORT | reason=binary_or_weights_missing | server={SERVER_BIN} | "
        f"gguf={GGUF}",
        flush=True,
    )
'''

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
    #
    # TIMEOUT MARGIN + VISIBLE EXIT STATUS (2026-08-08 adversarial review, Gaps finding 4). The
    # timeout used to equal the FULL 12h cap with `TimeoutExpired` uncaught and the return code
    # never inspected -- a run that dies AT the deadline ended as a hard kill or an uncaught
    # traceback, and a swarm that crashed at t=0 was indistinguishable from a clean 12h success by
    # the terminal "complete" line below. 41400s (30 min short of the 43200s cap) gives this
    # process room to observe and print the outcome before Kaggle's own harness kills the kernel;
    # `check=False` is unchanged (a non-zero exit still lets Save & Run produce whatever partial
    # submission the swarm managed), but the exit is now NAMED instead of silent.
    run_env = os.environ.copy()
    run_env["MPLBACKEND"] = "agg"  # headless matplotlib (canonical nb sets this)
    _swarm_t0 = time.time()
    try:
        _swarm_result = subprocess.run(
            [sys.executable, "main.py", "--agent", "carnotagent"],
            cwd=fw,
            env=run_env,
            timeout=41400,
            check=False,
        )
        print(
            f"SWARM EXITED rc={_swarm_result.returncode} after {time.time() - _swarm_t0:.0f}s",
            flush=True,
        )
    except subprocess.TimeoutExpired:
        print(
            f"SWARM TIMED OUT after {time.time() - _swarm_t0:.0f}s "
            "(41400s budget) -- killed before the swarm process exited on its own",
            flush=True,
        )
else:
    # NON-rerun: write the placeholder submission so Save & Run produces a valid entry.
    # WRITTEN FIRST, before the diagnostics below, so nothing the probe does can ever cost us
    # a valid submission entry.
    import pandas as pd

    pd.DataFrame(
        [["1_0", "1", True, 1]],
        columns=["row_id", "game_id", "end_of_game", "score"],
    ).to_parquet("/kaggle/working/submission.parquet", index=False)

    # PREVIEW-MODE GENERATOR DIAGNOSTICS (2026-08-11, operator-driven). The scored rerun's log
    # is NOT retrievable: the Kaggle UI and `kaggle kernels output` both return only THIS
    # branch's log (verified directly on ref 55425907 -- the operator's submission-page log was
    # the 7.6s preview, nothing more). So every LLM TIER RESOLVED / GENERATOR HEALTHY /
    # CONCURRENCY / GPU HARDWARE line the rerun prints goes somewhere no human can read, and
    # whether the LLM tier has EVER engaged in a scored run is unobservable. Suspicious pattern
    # that motivated this: scores barely moved across a total generator swap (Qwen-9B/L4 era
    # 0.08-0.12 vs gemma-31B/RtxPro6000 0.09), which is consistent with the generator never
    # engaging at all.
    #
    # Preview runs on the SAME machine_shape with the SAME datasets and GPU attached, so
    # running the diagnostics HERE answers -- in a log we can actually read -- whether the
    # generator loads, fits, and survives slot-count concurrency on the real scored hardware,
    # and what that hardware actually is (settles the RtxPro6000 single-vs-multi-card / VRAM
    # question). Not a byte-for-byte witness of the rerun itself (different branch, no
    # gateway), but the same binary + weights + env on the same hardware shape is the closest
    # observable proxy that exists.
    #
    # Runs in a SUBPROCESS with a hard timeout: a wedged llama-server load must not hang the
    # kernel, and any crash is contained (the placeholder above is already on disk either way).
    # AGENT_SRC ends before the framework bind (see AGENT_BIND_TAIL), so executing it here
    # needs no `agents` package.
    Path("/kaggle/working/preview_llm_diag.py").write_text(AGENT_SRC)
    _diag_t0 = time.time()
    try:
        _diag = subprocess.run(
            [sys.executable, "/kaggle/working/preview_llm_diag.py"],
            timeout=3600,
            check=False,
        )
        print(
            f"PREVIEW LLM DIAGNOSTICS EXITED rc={_diag.returncode} "
            f"after {time.time() - _diag_t0:.0f}s",
            flush=True,
        )
    except subprocess.TimeoutExpired:
        print(
            f"PREVIEW LLM DIAGNOSTICS TIMED OUT after {time.time() - _diag_t0:.0f}s "
            "(3600s budget) -- the generator load or concurrency probe wedged; treat as a "
            "FAILED health signal for the scored hardware.",
            flush=True,
        )

    # PREVIEW-MODE GENERATOR CONFIG A/B (2026-08-11). The diagnostics above answer
    # "does the generator load?". This answers "which server configuration is best?".
    # It runs AFTER them, so the two never share a port or contend for VRAM.
    #
    # WHY IT IS SAFE. The placeholder submission is already on disk, written at the top
    # of this branch. A/B failure cannot cost a valid submission entry. The harness runs
    # in its own process with a soft 2400s budget. The hard timeout below is 2580s, so
    # the harness gets 180s to print its own summary before the kernel kills it.
    #
    # RESIDUAL, NAMED ON PURPOSE. The hard kill below stops the harness, not any
    # llama-server it started. That is acceptable only because this is the LAST step of
    # the preview run. Add work after this block and you must kill the process group.
    _ab_t0 = time.time()
    try:
        Path("/kaggle/working/preview_ab_harness.py").write_text(PREVIEW_AB_SRC)
        _ab = subprocess.run(
            [sys.executable, "/kaggle/working/preview_ab_harness.py"],
            timeout=2580,
            check=False,
        )
        print(
            f"PREVIEW AB HARNESS EXITED rc={_ab.returncode} after {time.time() - _ab_t0:.0f}s",
            flush=True,
        )
    except subprocess.TimeoutExpired:
        print(
            f"PREVIEW AB HARNESS TIMED OUT after {time.time() - _ab_t0:.0f}s "
            "(2580s hard kill, 2400s soft budget) -- a server load or an arm wedged. Read "
            "the PREVIEW AB lines already printed; the arms that completed are still valid.",
            flush=True,
        )
    except Exception as _abe:
        print(f"PREVIEW AB HARNESS ERROR (non-fatal): {_abe!r}", flush=True)

print("CarnotAgent submission notebook complete.")
