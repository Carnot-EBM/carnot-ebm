"""THE ONE ARC SHOT: does counterexample-guided refinement help ONCE the model can see the
engine it is refining -- or not?

THE PRE-COMMITTED STOPPING RULE (operator, 2026-08-03, before any number here existed). If
refinement-with-the-engine-visible does NOT beat single-shot induction on gradeable acceptance
cells, the ARC induction line CLOSES. A clean null is a SUCCESS for this task. No follow-up
"the instrument was still broken" experiment is authorised. If a further instrument defect is
found it is RECORDED and the result is still reported against this rule.

WHY THIS MEASUREMENT WAS NOT POSSIBLE BEFORE
--------------------------------------------
Two defects, both reproduced by calling shipped code over the 13 real offline windows
(results/outer_loop_arc_refine_instrument_repro_20260803.json):

  D1  THE REFACTOR PROMPT NEVER CONTAINED THE ENGINE. 0 of 454 substantive engine source lines
      reached the rendered prompt, on 13 of 13 games. The only matches were this codebase's own
      REQUIRED OUTPUT STRUCTURE boilerplate. So every shipped "refinement" round was a BLIND
      RE-INDUCTION from <=5 failing deltas, told to "keep the cases it already gets right" about
      code it could not see. FIXED behind `CARNOT_ARC_REFACTOR_SHOW_ENGINE` (default OFF).

  D2  30.8% OF ACCEPTANCE CELLS WERE UNGRADEABLE. Under the shipped two-way split, sp80 / r11l /
      vc33 / ft09 (12 of 39 cells) have ZERO gradeable acceptance rows, because the only changing
      row in the tail is the level-up row that `WorldModelVerifier.score` correctly refuses to
      grade. A PERFECT ORACLE engine scores 0.0 there -- an unfalsifiable gate reported as a
      failure. Turning ON the already-shipped `CARNOT_ARC_CEGIS_ACCEPT_SPLIT` recovers sp80 and
      ft09 (oracle 1.0). r11l and vc33 (both n=3 windows) remain structurally undecidable and
      LEAVE THE DENOMINATOR EXPLICITLY -- named in the artifact, never silently dropped.

DESIGN -- three arms, PAIRED AT THE SAMPLE, not merely at the game
------------------------------------------------------------------
Per (game, trial) cell there is ONE induce call. Both refinement arms then fork from that SAME
round-0 engine source, restored to disk before each arm runs. So the treatment-vs-control
contrast carries ZERO round-0 sampling noise -- the arms differ in exactly one byte-level thing,
whether `CARNOT_ARC_REFACTOR_SHOW_ENGINE` is 1 or 0.

  single_shot       round-0 engine, graded on the acceptance block. THE BASELINE THAT MATTERS:
                    refinement must beat NOT refining.
  refine_control    round-0 engine + R refactor rounds, SHIPPED blind prompt.
  refine_treatment  round-0 engine + R refactor rounds, engine visible.

DIVERGENCE DISCLOSED. This drives the round loop itself rather than calling
`execute_bounded_llm_reinduction`, because that function cannot fork two arms off one induce.
Every STEP is the shipped function -- `proposer.induce`, `WorldModelVerifier.score`,
`_counterexample_result`, `proposer.refactor`, `refactor_prompt`, `split_refinement_acceptance`,
`_proposal_prefix` -- and the order mirrors the shipped loop. What is reimplemented is the
for-loop, not the prompts, not the metrics, not the split.

PURITY, VERIFIED PER CELL RATHER THAN ASSERTED. No acceptance row may shape refinement. The
induce evidence is `_proposal_prefix` minus the reserved rows (the shipped filter) and the
counterexample corpus is `refinable`. That is CHECKED by searching every rendered prompt string
for each acceptance row's own delta encoding -- delivery, not availability -- and the count is
recorded on every cell, including when it is zero.

METRIC. Primary is held-out `change_accuracy` on the acceptance block, the quantity the
stopping rule names. Reported beside it, PRE-REGISTERED as secondary because the primary is
coarse at these window sizes: `change_fidelity` (continuous, symmetric cell-level union
fidelity) and `cell_recall`. Rows are stratified by changed-cell count -- a 1-cell row is a
progress counter, not dynamics.

CONTROLS.
  * IDENTITY control (returns its input). On the acceptance block every gradeable row CHANGES
    and `n_noop` is 0, so identity scores 0.0 BY CONSTRUCTION -- that control is VACUOUS there
    and is reported as vacuous, not as evidence. It is therefore ALSO run on the full window,
    where no-op rows exist and the score can move.
  * ORACLE control (returns the recorded next_grid). It must reach 1.0 on every gradeable
    acceptance block, else the block is unfalsifiable and the game is excluded.

SUBSTRATE: live gemma-4-31B-it Q4_K_M on GPU 1 via its own llama-server on a NON-DEFAULT port.
GPU 0 belongs to the conductor and is not touched. Preconditions are checked BEFORE any
inference and a CPU fallback is refused, never silently accepted.

NOT A SOLVE. No level is claimed, nothing is submitted, no scored/online game is played, no
shipped default is flipped, and `results/arc_e3` is restored byte-for-byte at the end.

Spec: REQ-ARC-WMTE-6091
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "python") not in sys.path:
    sys.path.insert(0, str(REPO / "python"))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

ARTIFACT = REPO / "results" / "experiment_6091_refine_engine_visible_ab.json"
SHARD = REPO / "results" / "exp6091_refine_engine_visible_shard.jsonl"
EVIDENCE_DIRS = ("results/arc_e3", "results/arc_logo_snapshot", "results/arc_e3_origin_fixtures")

# ---- run configuration -----------------------------------------------------------------------
GPU_INDEX = 1  # operator: the conductor owns GPU 0.
PORT = 8977  # NON-DEFAULT (LocalGGUFProposer's default is 8919; prior runs used 8968-8972).
BUDGET = int(os.environ.get("CARNOT_6091_BUDGET") or "4096")  # the SHIPPED LIVE default
NCTX_LADDER = [32768, 24576, 20480]
REFACTOR_ROUNDS = int(os.environ.get("CARNOT_6091_ROUNDS") or "2")
TRIALS = [int(x) for x in (os.environ.get("CARNOT_6091_TRIALS") or "0,1").split(",")]
MAX_WALL_S = float(os.environ.get("CARNOT_6091_MAX_WALL_S") or "28800")  # 8h default
GPU_IDLE_MAX_MIB = 2000
SEED = 6091

# SUBSTRATE PIN -- DERIVED FROM THE LIVE CONSTANTS, NEVER HARDCODED (corrected 2026-08-03,
# operator-caught). This block previously hardcoded `gemma-4-31B-it` + a literal snapshot hash
# pointing at `gemma-4-31B-it-Q4_K_M.gguf` -- the NON-QAT build -- while the live ARC generator is
# pinned to `gemma-4-31B-it-qat-UD-Q4_K_XL.gguf`. Measuring the induction wall on a different
# quantization than the live generator actually runs is a substrate mismatch, and a hardcoded
# snapshot hash is how that drift survived unnoticed. Deriving from the module constants means the
# experiment cannot silently diverge from the live stack again.
#
# MTP IS DELIBERATELY OFF HERE and that is not an omission: ARC_LIVE_GENERATOR_MTP_DEFAULT is "0"
# for dev hardware because on a 24 GB card the offload MTP forces costs more throughput than the
# ~1.4x it returns. MTP-on is the SCORED (Kaggle 96 GB) default, where no offload is needed. The
# drafter must also come from the SAME repo as the target -- a non-QAT drafter paired with a QAT
# target is accepted by llama.cpp and silently degrades, which _resolve_mtp_head() guards.
from carnot.agentic.arc_executable_world_model import (  # noqa: E402
    ARC_LIVE_GENERATOR_MODEL_FILENAME as _LIVE_GGUF_NAME,
    ARC_LIVE_GENERATOR_MODEL_ID as _LIVE_HF_ID,
    ARC_LIVE_GENERATOR_REPO_SUBSTR as _LIVE_REPO_SUBSTR,
)


def _resolve_live_gguf() -> str:
    """Locate the pinned QAT weights in the HF cache without baking a snapshot hash."""
    root = pathlib.Path.home() / ".cache/huggingface/hub"
    hits = sorted(root.glob(f"models--*/snapshots/*/{_LIVE_GGUF_NAME}"))
    if not hits:
        raise SystemExit(
            f"blocked_model_not_cached: {_LIVE_GGUF_NAME} not found under {root}. "
            "PRECONDITIONS: do not fall back to another quantization -- that is the "
            "substrate mismatch this block was rewritten to prevent."
        )
    return str(hits[0])


GEMMA: dict[str, Any] = {
    "repo_substr": _LIVE_REPO_SUBSTR,
    "hf_id": _LIVE_HF_ID,
    "gguf": _resolve_live_gguf(),
    "kv_quant": "q8_0",
    "timeout": 1800,
}
# THE GENERATOR BINARY, ENV-OVERRIDABLE -- and the override is not a convenience.
# MEASURED 2026-08-03: four consecutive attempts at this run had their llama-server killed
# mid-generation, every time with the same server-log signature (`operator(): cleaning up before
# exit...` / `Received second interrupt, terminating immediately.`) while it was healthily
# decoding at ~34 tok/s. Ruled OUT by measurement: host OOM (94 GB available), the 2-hour orphan
# janitor (only targets python >2h old, never llama-server), the CUDA capacity guard, and a
# `--parallel` slot-arithmetic abort. `setsid` on the PARENT did not stop it, and neither did
# `SIG_IGN` on SIGINT/SIGTERM in the child -- llama.cpp installs its OWN console handler at
# startup, which overwrites an inherited SIG_IGN. A signal that survives both of those, on a
# machine concurrently running the conductor and its codex children, is a reaper matching the
# process by NAME. So the binary is COPIED to a privately-named path and run from there, with
# LD_LIBRARY_PATH pointing back at the real build dir for the CUDA shared objects.
# This is isolation from a noisy shared machine, NOT a change to what is measured: it is the same
# bytes (sha256 pinned in the artifact) and the substrate witness still proves CUDA per cell.
LLAMA_SERVER = Path(
    os.environ.get("CARNOT_6091_LLAMA_SERVER")
    or (Path.home() / ".cache" / "llama.cpp-master" / "build" / "bin" / "llama-server")
)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ==============================================================================================
# GPU / server
# ==============================================================================================
def _gpu_mem_used_mib(index: int) -> Optional[int]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return int(out.stdout.strip().splitlines()[index].strip())
    except Exception:
        return None


def _server_args(n_ctx: int) -> list[str]:
    # `--parallel 1` IS LOAD-BEARING, and its absence killed the first attempt at this run.
    # llama-server with no explicit `--parallel` sets n_parallel=4 with a UNIFIED KV cache, so one
    # slot only gets n_ctx/4. At n_ctx=32768 that is 8192 tokens against a 16384-token budget plus
    # a ~6k-token prompt, and the server does not degrade gracefully -- it aborts
    # (`GGML_ASSERT(logits != nullptr)`, the failure `_default_induce_n_ctx`'s docstring
    # describes). OBSERVED 2026-08-03: the CUDA server on port 8977 went <defunct> mid-cell, the
    # proposer's `_ensure_server` relaunched on the SAME port, the CUDA guard refused GPU 1
    # (it sizes against `CARNOT_ARC_INDUCE_N_CTX`, which defaulted to 81920 -> 25388 MiB required
    # on a 24576 MiB card), and it fell back to the AMD iGPU HIP build at ~2 tok/s. The run was
    # DISCARDED, not repaired: the substrate had changed mid-measurement.
    # This run is strictly sequential -- one generation at a time -- so one slot is correct and
    # gives the whole pool to the single request.
    return [
        str(LLAMA_SERVER),
        "-m",
        GEMMA["gguf"],
        "-ngl",
        "999",
        "-c",
        str(n_ctx),
        "--parallel",
        "1",
        "--port",
        str(PORT),
        "--host",
        "127.0.0.1",
        "--cache-type-k",
        GEMMA["kv_quant"],
        "--cache-type-v",
        GEMMA["kv_quant"],
        "-fit",
        "off",
    ]


SERVER_LOG = Path(os.environ.get("CARNOT_6091_SERVER_LOG") or "/tmp/exp6091_llama_server.log")


def serving_pid_on_port(port: int) -> Optional[int]:
    """The PID actually holding the listening socket. This is the CALLEE side: 'my server is
    healthy' is a claim about a port, not about which binary answers on it."""
    try:
        out = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return None
    for line in out.splitlines():
        if f":{port} " in line and "pid=" in line:
            try:
                return int(line.split("pid=")[1].split(",")[0])
            except (IndexError, ValueError):
                return None
    return None


def substrate_witness(port: int) -> dict[str, Any]:
    """PROVE the generator substrate from the serving process, never from our own intent.

    Reads the exe path and the loaded shared objects of whatever PID owns the port. A HIP build
    on the AMD iGPU and a CUDA build on an RTX 3090 both answer /health with 200 and both
    generate correct text; the ONLY difference visible from the client is throughput. So the
    check has to look at the process."""
    pid = serving_pid_on_port(port)
    if pid is None:
        return {"pid": None, "is_cuda": False, "reason": "no process owns the port"}
    try:
        exe = os.readlink(f"/proc/{pid}/exe")
    except OSError:
        exe = ""
    try:
        maps = Path(f"/proc/{pid}/maps").read_text()
    except OSError:
        maps = ""
    has_cuda = "libggml-cuda" in maps or "libcudart" in maps
    has_hip = "libggml-hip" in maps or "libamdhip" in maps or "librocblas" in maps
    return {
        "pid": pid,
        "exe": exe,
        "loaded_cuda": has_cuda,
        "loaded_hip": has_hip,
        "is_cuda": bool(has_cuda and not has_hip),
        "reason": "ok" if (has_cuda and not has_hip) else "NOT the CUDA build",
    }


def _launch_one(n_ctx: int) -> subprocess.Popen:
    args = _server_args(n_ctx)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(GPU_INDEX))
    log(f"  launch n_ctx={n_ctx} CUDA_VISIBLE_DEVICES={GPU_INDEX} port={PORT}")
    # Server stdout/stderr go to a FILE, not DEVNULL. The first attempt at this run sent them to
    # DEVNULL, so when the server died mid-cell there was no record of why and the cause had to
    # be reconstructed from `<defunct>` in the process table.
    logf = SERVER_LOG.open("ab")

    def _detach() -> None:
        """Put the server in its OWN session and make it deaf to SIGINT/SIGTERM.

        WHY, MEASURED. Three consecutive attempts at this run died mid-cell with the server log
        reading `operator(): cleaning up before exit...` / `Received second interrupt,
        terminating immediately.` -- i.e. SIGINT, twice, to a healthy server that was generating
        at ~34 tok/s. It was NOT an OOM (94 GB host RAM available), NOT the 2-hour orphan
        janitor (which only targets python older than 2h), and NOT the CUDA guard. It is a
        signal arriving from the surrounding process group on a machine that is also running the
        conductor and its codex children. `setsid` on the PARENT was not enough, because the
        server inherits the parent's group.
        So: `os.setsid()` gives the server a session of its own, and ignoring SIGINT/SIGTERM
        makes a stray group-directed signal a no-op instead of a lost measurement. This run
        therefore reaps with SIGKILL (see `terminate`), which no handler can ignore -- the
        server is never left behind.
        """
        os.setsid()
        import signal as _sig

        _sig.signal(_sig.SIGINT, _sig.SIG_IGN)
        _sig.signal(_sig.SIGTERM, _sig.SIG_IGN)

    proc = subprocess.Popen(
        args, stdout=logf, stderr=subprocess.STDOUT, env=env, preexec_fn=_detach
    )
    deadline = time.time() + GEMMA["timeout"]
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early (code {proc.returncode})")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if b"ok" in r.read():
                    return proc
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("llama-server did not become healthy before timeout")


def terminate(proc: Optional[subprocess.Popen]) -> None:
    """REAP WHAT YOU START. SIGKILL, because `_detach` makes the server ignore SIGTERM --
    a `terminate()` here would hang for 30s and then fall through to kill anyway."""
    if proc is None:
        return
    try:
        proc.kill()
        proc.wait(timeout=30)
    except Exception:
        pass


def launch_server_ladder() -> tuple[subprocess.Popen, int, int, int]:
    last = ""
    for n_ctx in NCTX_LADDER:
        v0 = _gpu_mem_used_mib(GPU_INDEX)
        try:
            proc = _launch_one(n_ctx)
        except Exception as exc:
            last = f"n_ctx={n_ctx}: {type(exc).__name__}: {exc}"
            log(f"  FAILED {last}")
            time.sleep(4)
            continue
        v1 = _gpu_mem_used_mib(GPU_INDEX)
        jump = (v1 - v0) if (v0 is not None and v1 is not None) else None
        log(f"  healthy n_ctx={n_ctx}; VRAM gpu{GPU_INDEX} {v0}->{v1} MiB (jump {jump})")
        if jump is not None and jump < 1000:
            terminate(proc)
            last = f"n_ctx={n_ctx}: VRAM jump {jump} MiB < 1GB -- NOT on GPU"
            log(f"  {last} -- refusing CPU fallback")
            time.sleep(4)
            continue
        return proc, n_ctx, int(v0 or 0), int(v1 or 0)
    raise RuntimeError(f"no n_ctx launched with real GPU offload. last={last}")


# ==============================================================================================
# evidence integrity
# ==============================================================================================
def evidence_checksum() -> dict[str, str]:
    out = {}
    for d in EVIDENCE_DIRS:
        p = REPO / d
        if not p.exists():
            out[d] = "absent"
            continue
        h = hashlib.sha256()
        for f in sorted(p.rglob("*")):
            if f.is_file():
                h.update(str(f.relative_to(p)).encode())
                h.update(f.read_bytes())
        out[d] = h.hexdigest()
    return out


# ==============================================================================================
# scoring
# ==============================================================================================
def grade(rows: list, engine) -> dict[str, Any]:
    """The shipped verifier over a row block. Every field is read off the VerifyResult."""
    from carnot.agentic.arc_executable_world_model import WorldModelVerifier

    if not rows:
        return {"gradeable_n": 0, "n_changing": 0, "change_accuracy": None}
    vr = WorldModelVerifier(list(rows), hud_mask=None).score(engine)
    return {
        "n_rows": len(rows),
        "gradeable_n": int(vr.n),
        "n_levelup_rows_excluded": int(vr.n_levelup_rows_excluded),
        "n_changing": int(vr.n_changing),
        "n_changes_correct": int(vr.n_changes_correct),
        "change_accuracy": round(float(vr.change_accuracy), 6),
        "change_fidelity": round(float(vr.change_fidelity), 6),
        "cell_recall": round(float(vr.cell_recall), 6),
        "accuracy": round(float(vr.accuracy), 6),
        "n_noop": int(vr.n_noop),
        "noop_channel_measurable": bool(vr.noop_channel_measurable),
        "n_engine_raised": int(vr.n_engine_raised),
        "n_output_equals_input": int(vr.n_output_equals_input),
        "correct_changed_cells": int(vr.correct_changed_cells),
        "invented_changed_cells": int(vr.invented_changed_cells),
    }


def identity_engine(grid, action, data=None):
    return np.asarray(grid).copy()


def make_oracle(rows: list):
    table = {}
    for t in rows:
        table[(np.asarray(t.grid).tobytes(), int(t.action))] = np.asarray(t.next_grid).copy()

    def engine(grid, action, data=None):
        hit = table.get((np.asarray(grid).tobytes(), int(action)))
        return hit.copy() if hit is not None else np.asarray(grid).copy()

    return engine


def gradeable_changed_cell_counts(rows: list) -> list[int]:
    out = []
    for t in rows:
        if int(getattr(t, "level_after", 0)) > int(getattr(t, "level_before", 0)):
            continue
        g0, g1 = np.asarray(t.grid), np.asarray(t.next_grid)
        if not np.array_equal(g0, g1):
            out.append(int((g0 != g1).sum()))
    return out


# ==============================================================================================
# purity: does an acceptance row's ANSWER reach any rendered prompt?
# ==============================================================================================
def acceptance_leak_probe(acceptance_rows: list, prompts: list[str]) -> dict[str, Any]:
    """DELIVERY check on rendered text. Each gradeable acceptance row is identified by its own
    run-length delta encoding (`_rle_delta_compact`, the induce prompt's form) AND by its
    `_delta` tuple list rendered as JSON (the refactor prompt's `true_change` form). A hit on
    either means a grading row's observed answer reached a prompt."""
    from carnot.agentic.arc_executable_world_model import _delta, _rle_delta_compact

    hits: list[dict[str, Any]] = []
    for i, t in enumerate(acceptance_rows):
        if int(getattr(t, "level_after", 0)) > int(getattr(t, "level_before", 0)):
            continue
        g0, g1 = np.asarray(t.grid), np.asarray(t.next_grid)
        if np.array_equal(g0, g1):
            continue
        rle = _rle_delta_compact(g0, g1)
        tuples = [list(x) for x in _delta(g0, g1)]
        # A single tuple could collide by chance; require the FIRST THREE (or all, if fewer)
        # to appear together, which no incidental code literal will satisfy.
        probe = json.dumps(tuples[:3])[1:-1]
        for p_i, text in enumerate(prompts):
            if (rle and rle in text) or (probe and probe in text):
                hits.append({"acceptance_row": i, "prompt_index": p_i})
    return {"n_leaks": len(hits), "leaks": hits[:8]}


# ==============================================================================================
# CALLEE-SIDE DELIVERY WITNESS (added 2026-08-03, before the measurement was taken)
# ==============================================================================================
# WHY THE HARNESS'S OWN `prompt_contains_engine` IS NOT SUFFICIENT EVIDENCE. `run_cell` renders
# `refactor_prompt(...)` itself to record that flag, and then calls `prop.refactor(...)`, which
# renders the prompt AGAIN. Two independent renders: the recorded flag describes the harness's
# copy, not the bytes the generator received. That is the availability-vs-delivery substitution
# this whole experiment exists to correct, so it must not be repeated in the instrument that
# measures it -- and this session separately watched a shipped fix reach its call site 0 of 128
# times while every surrounding indicator stayed green.
#
# So `generate` -- the deepest callee before the HTTP transport -- is wrapped once, and every
# refactor round asserts delivery against THE PROMPT THAT WENT TO THE MODEL. The wrapper only
# observes; it forwards to the real method unchanged.
_LAST_GEN: dict[str, Any] = {}


def install_generate_witness(prop: Any) -> None:
    """Observe `generate` WITHOUT changing what it is called with.

    The passthrough is `*args, **kwargs` DELIBERATELY, and this is not defensive style. A first
    version of this wrapper spelled out `(prompt, required, tries, codeonly_eligible)` -- which
    is the signature `refactor` happens to use -- and would therefore have SILENTLY DROPPED
    `validate=` and `engine_transitions=`, both of which the shipped `induce` passes. That is the
    instrument altering the very round-0 induction the two refinement arms fork from: not a
    measurement bug at the edges, a corrupted shared baseline. Caught before the run generated
    anything, and recorded here so the shape is not reintroduced. An observer must be transparent.
    """
    real = prop.generate

    def witnessed(prompt, *args, **kwargs):
        _LAST_GEN.clear()
        _LAST_GEN["prompt"] = prompt
        _LAST_GEN["chars"] = len(prompt)
        _LAST_GEN["codeonly_eligible"] = bool(kwargs.get("codeonly_eligible", False))
        return real(prompt, *args, **kwargs)

    prop.generate = witnessed


def delivery_witness(engine_source: str) -> dict[str, Any]:
    """Did THIS engine's own source reach the prompt the generator was handed?

    Counts only SUBSTANTIVE lines -- non-blank, longer than 8 characters after stripping -- so
    the template boilerplate the shipped prompt already contains (`import numpy as np`,
    `def engine(grid, action, data):`) cannot masquerade as delivery. That distinction is not
    theoretical: the pre-flight probe measured exactly 2 such boilerplate matches with the flag
    OFF versus 9 real source lines with it ON.
    """
    prompt = _LAST_GEN.get("prompt")
    if prompt is None:
        return {"generator_called": False}
    tmpl = ("import numpy as np", "def engine(grid, action, data):", "def is_level_complete(grid):")
    subst = [
        ln
        for ln in engine_source.splitlines()
        if ln.strip() and len(ln.strip()) > 8 and ln.strip() not in tmpl
    ]
    delivered = [ln for ln in subst if ln in prompt]
    return {
        "generator_called": True,
        "prompt_chars_at_generator": _LAST_GEN.get("chars"),
        "engine_header_at_generator": "THE CURRENT ENGINE YOU ARE FIXING" in prompt,
        "n_substantive_engine_lines": len(subst),
        "n_substantive_engine_lines_delivered": len(delivered),
        "engine_delivered": bool(subst) and len(delivered) >= max(1, len(subst) // 2),
    }


# ==============================================================================================
# the cell
# ==============================================================================================
def run_cell(game: str, trial: int, window: list, cell: int, prop: Any) -> dict[str, Any]:
    from carnot.agentic.arc_executable_world_model import (
        E3_DIR,
        WorldModelVerifier,
        induce_prompt,
        load_engine,
        refactor_prompt,
    )
    from carnot.agentic.arc_llm_reinduction import _counterexample_result, _proposal_prefix
    from carnot.agentic.arc_world_model_trust_energy import split_refinement_acceptance

    t0 = time.time()
    split = split_refinement_acceptance(list(window))
    reserved = {id(r) for r in split.acceptance}
    induction_evidence = [r for r in _proposal_prefix(list(window)) if id(r) not in reserved]
    refinable = list(split.refinable)
    acceptance = list(split.acceptance)

    oracle_grade = grade(acceptance, make_oracle(list(window)))
    row: dict[str, Any] = {
        "game": game,
        "trial": trial,
        "random_seed": SEED,
        "window_n": len(window),
        "n_refinable": len(refinable),
        "n_acceptance": len(acceptance),
        "acceptance_decidable": bool(split.decidable),
        "acceptance_reason": str(split.reason),
        "n_acceptance_gradeable": int(split.n_acceptance_gradeable),
        "acceptance_changed_cells_per_row": gradeable_changed_cell_counts(acceptance),
        "oracle_acceptance": oracle_grade,
        "oracle_reaches_1": oracle_grade.get("change_accuracy") == 1.0,
        "identity_acceptance": grade(acceptance, identity_engine),
        "identity_full_window": grade(list(window), identity_engine),
        "refactor_rounds_configured": REFACTOR_ROUNDS,
    }

    wm_path = E3_DIR / game / "world_model.py"
    prompts_seen: list[str] = []

    # ---- round 0: ONE induce, shared by both refinement arms ---------------------------------
    try:
        wm_path.unlink()
    except FileNotFoundError:
        pass
    prompts_seen.append(induce_prompt(game, list(induction_evidence), int(cell)))
    t_ind = time.time()
    induce_ok, induce_msg = prop.induce(game, list(induction_evidence), int(cell))
    row["induce_ok"] = bool(induce_ok)
    row["induce_wall_s"] = round(time.time() - t_ind, 1)
    if induce_msg:
        row["induce_message"] = str(induce_msg)[:200]

    engine0 = None
    source0 = ""
    try:
        engine0, _goal0 = load_engine(game)
        source0 = wm_path.read_text()
    except Exception as exc:
        row["round0_load_error"] = f"{type(exc).__name__}: {exc}"[:200]

    row["round0_engine_loaded"] = engine0 is not None
    row["round0_source_chars"] = len(source0)
    row["single_shot"] = (
        grade(acceptance, engine0) if engine0 is not None else {"change_accuracy": None}
    )
    row["single_shot_refinable"] = (
        grade(refinable, engine0) if engine0 is not None else {"change_accuracy": None}
    )

    # ---- the two refinement arms, both forked from source0 -----------------------------------
    for arm, flag in (("refine_control", "0"), ("refine_treatment", "1")):
        arm_rows: list[dict[str, Any]] = []
        if engine0 is None or not source0.strip():
            row[arm] = {"skipped": "no_round0_engine", "rounds": []}
            continue
        wm_path.parent.mkdir(parents=True, exist_ok=True)
        wm_path.write_text(source0)  # FORK POINT: identical starting engine for both arms
        engine = engine0
        prev_flag = os.environ.get("CARNOT_ARC_REFACTOR_SHOW_ENGINE")
        os.environ["CARNOT_ARC_REFACTOR_SHOW_ENGINE"] = flag
        try:
            for r in range(1, REFACTOR_ROUNDS + 1):
                rr: dict[str, Any] = {"round": r}
                try:
                    rv = WorldModelVerifier(list(refinable), hud_mask=None).score(engine)
                except Exception as exc:
                    rr["verify_error"] = f"{type(exc).__name__}: {exc}"[:160]
                    arm_rows.append(rr)
                    break
                cx = {
                    "kind": "heldout_transition_verification_failed",
                    "real_n": rv.n,
                    "real_n_correct": rv.n_correct,
                    "real_accuracy": float(rv.accuracy),
                    "real_mismatches": list(rv.mismatches),
                }
                vr_obj = _counterexample_result(cx)
                rendered = refactor_prompt(game, vr_obj)
                prompts_seen.append(rendered)
                rr["prompt_chars"] = len(rendered)
                rr["prompt_contains_engine"] = bool("THE CURRENT ENGINE YOU ARE FIXING" in rendered)
                rr["n_mismatches_available"] = len(rv.mismatches)
                # The engine ON DISK at this instant is what `refactor_prompt` will read, so it
                # is the exact text whose delivery the witness must check.
                try:
                    src_now = wm_path.read_text()
                except OSError:
                    src_now = ""
                _LAST_GEN.clear()
                t_r = time.time()
                ok, msg = prop.refactor(game, vr_obj)
                rr["refactor_ok"] = bool(ok)
                rr["wall_s"] = round(time.time() - t_r, 1)
                # DELIVERY, read at the callee -- not from the harness's own render above.
                rr["delivery_witness"] = delivery_witness(src_now)
                if msg:
                    rr["message"] = str(msg)[:200]
                try:
                    engine, _g = load_engine(game)
                    rr["engine_loaded"] = True
                    rr["source_chars"] = len(wm_path.read_text())
                except Exception as exc:
                    rr["engine_loaded"] = False
                    rr["load_error"] = f"{type(exc).__name__}: {exc}"[:160]
                    arm_rows.append(rr)
                    break
                rr["acceptance"] = grade(acceptance, engine)
                rr["refinable"] = grade(refinable, engine)
                arm_rows.append(rr)
        finally:
            if prev_flag is None:
                os.environ.pop("CARNOT_ARC_REFACTOR_SHOW_ENGINE", None)
            else:
                os.environ["CARNOT_ARC_REFACTOR_SHOW_ENGINE"] = prev_flag

        scored = [x["acceptance"] for x in arm_rows if isinstance(x.get("acceptance"), dict)]
        best_ca = max(
            [s["change_accuracy"] for s in scored if s.get("change_accuracy") is not None],
            default=None,
        )
        best_cf = max(
            [s["change_fidelity"] for s in scored if s.get("change_fidelity") is not None],
            default=None,
        )
        best_cr = max(
            [s["cell_recall"] for s in scored if s.get("cell_recall") is not None], default=None
        )
        row[arm] = {
            "rounds": arm_rows,
            "n_rounds_run": len(arm_rows),
            "n_engine_loaded": sum(1 for x in arm_rows if x.get("engine_loaded")),
            "n_prompts_with_engine": sum(1 for x in arm_rows if x.get("prompt_contains_engine")),
            # The callee-side count. If this disagrees with `n_prompts_with_engine` above, the
            # harness's render and the model's prompt diverged and the cell is not trustworthy.
            "n_generator_prompts_with_engine": sum(
                1 for x in arm_rows if (x.get("delivery_witness") or {}).get("engine_delivered")
            ),
            "n_generator_calls_witnessed": sum(
                1 for x in arm_rows if (x.get("delivery_witness") or {}).get("generator_called")
            ),
            "best_change_accuracy": best_ca,
            "best_change_fidelity": best_cf,
            "best_cell_recall": best_cr,
            "final": scored[-1] if scored else None,
        }

    # ---- restore the fork point so the store is left in a defined state ----------------------
    if source0.strip():
        wm_path.write_text(source0)

    row["acceptance_purity"] = acceptance_leak_probe(acceptance, prompts_seen)
    row["wall_s"] = round(time.time() - t0, 1)
    return row


# ==============================================================================================
# shard IO
# ==============================================================================================
def load_shard() -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    if SHARD.exists():
        for line in SHARD.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[(r["game"], int(r["trial"]))] = r
    return rows


def append_shard(row: dict[str, Any]) -> None:
    SHARD.parent.mkdir(parents=True, exist_ok=True)
    with SHARD.open("a") as f:
        f.write(json.dumps(row) + "\n")


# ==============================================================================================
# preconditions -- BEFORE any inference (Pre-Launch Preconditions Discipline)
# ==============================================================================================
def check_preconditions() -> tuple[bool, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []

    def add(resource: str, ok: bool, detail: str = "") -> None:
        checks.append({"resource": resource, "available": bool(ok), "detail": detail})

    add("gguf_cached::gemma-4-31B-it", Path(GEMMA["gguf"]).exists(), GEMMA["gguf"])
    add("llama_server_binary", LLAMA_SERVER.exists(), str(LLAMA_SERVER))
    try:
        from llama_cpp import llama_cpp as _b

        offload = bool(_b.llama_supports_gpu_offload())
    except Exception as exc:
        offload = False
        add("llama_cpp_import", False, f"{type(exc).__name__}: {exc}")
    add("llama_cpp_gpu_offload", offload, "llama_supports_gpu_offload()")
    used = _gpu_mem_used_mib(GPU_INDEX)
    add(
        f"gpu{GPU_INDEX}_idle",
        used is not None and used < GPU_IDLE_MAX_MIB,
        f"used={used} MiB (< {GPU_IDLE_MAX_MIB})",
    )
    try:
        ldd = subprocess.run(
            ["ldd", str(LLAMA_SERVER)], capture_output=True, text=True, timeout=30
        ).stdout
    except Exception:
        ldd = ""
    add("llama_server_links_cuda", "libcuda" in ldd or "libggml-cuda" in ldd, "ldd")
    return all(c["available"] for c in checks), checks


# ==============================================================================================
# main
# ==============================================================================================
def main() -> int:
    t_start = time.time()
    # THE ENGINE STORE MUST BE REDIRECTED, AND IT MUST BE REDIRECTED BEFORE THIS INTERPRETER
    # STARTED. `E3_DIR` is resolved at IMPORT time from `CARNOT_ARC_E3_DIR`, so setting the var
    # here would be a no-op that LOOKS like a safeguard -- exactly the class of silent
    # non-firing this project keeps finding. `induce`/`refactor` WRITE `<E3_DIR>/<game>/
    # world_model.py`, and `results/arc_e3` is read-only evidence, so refuse to run rather than
    # write it.
    from carnot.agentic.arc_executable_world_model import E3_DIR as _E3

    if _E3.resolve() == (REPO / "results" / "arc_e3").resolve():
        log(
            "REFUSING: CARNOT_ARC_E3_DIR is unset, so induce/refactor would write the tracked "
            "evidence store. Re-run with CARNOT_ARC_E3_DIR pointing at a scratch directory."
        )
        ARTIFACT.write_text(
            json.dumps(
                {
                    "experiment": "experiment_6091_refine_engine_visible_ab",
                    "spec": "REQ-ARC-WMTE-6091",
                    "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "honest_verdict": "blocked_e3_dir_not_redirected",
                    "duration_s": round(time.time() - t_start, 3),
                },
                indent=1,
            )
        )
        return 1
    log(f"engine store redirected to {_E3}")
    ok, checks = check_preconditions()
    if not ok:
        missing = [c["resource"] for c in checks if not c["available"]]
        out = {
            "experiment": "experiment_6091_refine_engine_visible_ab",
            "spec": "REQ-ARC-WMTE-6091",
            "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "preconditions_checked": checks,
            "honest_verdict": f"blocked_{missing[0]}",
            "duration_s": round(time.time() - t_start, 3),
        }
        ARTIFACT.write_text(json.dumps(out, indent=1))
        log(f"BLOCKED: {missing}")
        return 1
    log("preconditions OK: " + ", ".join(c["resource"] for c in checks))

    from carnot.agentic import arc_actions_to_progress as atp
    from carnot.experiment_5760_cegis_refinement_induction_ab import ROSTER

    ev_before = evidence_checksum()

    windows: dict[str, tuple[list, int]] = {}
    for game in ROSTER:
        try:
            built = atp.build_progress_window(game)
        except Exception as exc:
            log(f"{game}: window build raised {type(exc).__name__}: {exc}")
            built = None
        if built:
            windows[game] = (list(built[0]), int(built[2]))
            log(f"{game}: window n={len(built[0])} cell={built[2]}")
        else:
            log(f"{game}: NO WINDOW")

    # PURITY + FALSIFIABILITY: the acceptance split must be ON for this measurement.
    os.environ["CARNOT_ARC_CEGIS_ACCEPT_SPLIT"] = "1"
    from carnot.agentic.arc_world_model_trust_energy import cegis_accept_split_enabled

    assert cegis_accept_split_enabled(), "acceptance split did not turn on"

    done = load_shard()
    pending = [(g, t) for t in TRIALS for g in ROSTER if g in windows and (g, t) not in done]
    log(f"resume: {len(done)} cells in shard; {len(pending)} pending")

    proc = None
    n_server_relaunches = 0
    server_meta: dict[str, Any] = {}
    try:
        if pending:
            proc, n_ctx, v0, v1 = launch_server_ladder()
            server_meta = {
                "n_ctx": n_ctx,
                "port": PORT,
                "gpu_index": GPU_INDEX,
                "vram_before_mib": v0,
                "vram_after_mib": v1,
                "vram_jump_mib": v1 - v0,
            }
            from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

            prop = LocalGGUFProposer(
                repo_substr=GEMMA["repo_substr"],
                port=PORT,
                mtp=False,
                kv_quant=GEMMA["kv_quant"],
                n_ctx=n_ctx,
                max_tokens=BUDGET,
                timeout=GEMMA["timeout"],
                use_chat_template=True,
                model_path=GEMMA["gguf"],
            )
            # GENERATION CONFIG = THE SHIPPED LIVE ONE, not exp5760's diagnostic one, and that is
            # a deliberate change of arm with a stated reason.
            # exp5760/5764/5766 used `max_tokens=16384` + `/think` + `CARNOT_ARC_CODEONLY_INDUCE=0`.
            # At the measured 34 tok/s that is ~8 minutes PER CALL, and this machine kills the
            # generator on a 1.6-13 minute timescale (see the LLAMA_SERVER comment), so a cell
            # built from 8-minute calls cannot complete here at all. The shipped LIVE default --
            # `LocalGGUFProposer.max_tokens = 4096`, code-only induce ON -- is both fast enough to
            # finish inside that window AND the configuration the scored agent actually runs, so
            # it is the more faithful arm, not a weaker one. Refactor is NOT code-only either way
            # (`codeonly_eligible=False` is set by the shipped `refactor`), so the treatment and
            # control prompts are unaffected by this choice; it changes only the shared round-0
            # induce and the shared per-round budget, identically for all three arms.
            # tries=2, not 1. MEASURED at BUDGET=4096: tu93 round-0 induce returned
            # "HIT n_predict=4096 OUTPUT LIMIT before completing" and produced no engine, so
            # the whole cell was unusable. A truncated emission is a generation accident, not
            # a property of any arm, and it costs the cell for ALL THREE arms equally -- so
            # retrying it is not arm-favouring. The shipped default is tries=3.
            prop.tries = int(os.environ.get("CARNOT_6091_TRIES") or "2")
            os.environ["CARNOT_ARC_CODEONLY_INDUCE"] = "1"
            # Observe the deepest callee before transport, so every refactor round can prove the
            # engine reached the MODEL's prompt rather than the harness's copy of it.
            install_generate_witness(prop)

            for i, (game, trial) in enumerate(pending, 1):
                if time.time() - t_start > MAX_WALL_S:
                    log(f"WALL BUDGET reached; stopping with {len(pending) - i + 1} cells unrun")
                    break
                w, cell = windows[game]
                # SUBSTRATE ASSERTION, BEFORE the cell and again AFTER it. A cell is only
                # admissible if the CUDA build owned the port for its whole duration. If the
                # proposer silently relaunched onto the iGPU HIP build (the failure that
                # discarded the first attempt), STOP -- do not write a cell that would be
                # indistinguishable in the shard from a genuine GPU one.
                wit_before = substrate_witness(PORT)
                if not wit_before["is_cuda"]:
                    # A DEAD server is recoverable; a HIP server is NOT.
                    # This machine SIGINTs the generator on a timescale of minutes (see the
                    # LLAMA_SERVER comment). Aborting the whole run on that would mean this
                    # measurement can never be taken here, while silently CONTINUING would be
                    # the iGPU-substrate lie the witness exists to prevent. So: if nothing owns
                    # the port, relaunch the CUDA server and re-witness. If a HIP build owns it,
                    # stop -- that is a wrong-substrate condition, not a transient one.
                    if wit_before.get("loaded_hip"):
                        log(f"ABORT: HIP build owns the port before {game}: {wit_before}")
                        break
                    log(f"  server gone before {game}; relaunching CUDA server")
                    terminate(proc)
                    try:
                        proc, n_ctx, v0, v1 = launch_server_ladder()
                        n_server_relaunches += 1
                    except Exception as exc:
                        log(f"ABORT: relaunch failed before {game}: {exc}")
                        break
                    wit_before = substrate_witness(PORT)
                    if not wit_before["is_cuda"]:
                        log(f"ABORT: relaunched server is not CUDA: {wit_before}")
                        break
                log(f"[{i}/{len(pending)}] {game} trial={trial} (n={len(w)})")
                try:
                    r = run_cell(game, trial, w, cell, prop)
                except Exception as exc:
                    r = {
                        "game": game,
                        "trial": trial,
                        "error": f"{type(exc).__name__}: {exc}"[:300],
                    }
                wit_after = substrate_witness(PORT)
                r["substrate_witness_before"] = wit_before
                r["substrate_witness_after"] = wit_after
                r["substrate_cuda_throughout"] = bool(
                    wit_before["is_cuda"]
                    and wit_after["is_cuda"]
                    and wit_before["pid"] == wit_after["pid"]
                )
                append_shard(r)
                # A cell whose server died MID-CELL is recorded (never dropped) but is marked
                # not-CUDA-throughout, and the analysis excludes it. That is the honest
                # treatment: the cell ran, we cannot vouch for its substrate for its whole
                # duration, and "missing is not zero" cuts both ways -- it is named, not deleted.
                if not r["substrate_cuda_throughout"]:
                    log(
                        f"  NOTE: substrate changed during {game} "
                        f"({wit_before.get('pid')} -> {wit_after.get('pid')}); "
                        "cell recorded and EXCLUDED; continuing"
                    )
                    if wit_after.get("loaded_hip"):
                        log("ABORT: a HIP build took the port; stopping rather than measuring it")
                        break
                log(
                    f"    ss={r.get('single_shot', {}).get('change_accuracy')} "
                    f"ctl={r.get('refine_control', {}).get('best_change_accuracy')} "
                    f"trt={r.get('refine_treatment', {}).get('best_change_accuracy')} "
                    f"({r.get('wall_s')}s)"
                )
    finally:
        terminate(proc)
        log("server terminated (reaped)")

    # ---- the evidence tree must be byte-identical, because nothing here ever wrote it --------
    # The engine store is REDIRECTED (see the CARNOT_ARC_E3_DIR assertion in main's preamble),
    # so `results/arc_e3` is only ever READ. This is a VERIFICATION, not a repair: there is
    # deliberately no `git checkout` here -- blanket-reverting a path is the data-loss move the
    # "Never Stash -- Always Commit-First" rule exists to prevent, and it would also mask a real
    # write rather than surface it.
    ev_after = evidence_checksum()

    out = {
        "experiment": "experiment_6091_refine_engine_visible_ab",
        "spec": "REQ-ARC-WMTE-6091",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "preconditions_checked": checks,
        "server": server_meta,
        "config": {
            "roster": ROSTER,
            "trials": TRIALS,
            "refactor_rounds": REFACTOR_ROUNDS,
            "budget_max_tokens": BUDGET,
            "cegis_accept_split": "1",
            "refactor_show_engine_treatment": "1",
            "refactor_show_engine_control": "0",
        },
        "shard": str(SHARD.relative_to(REPO)),
        "evidence_checksum_before": ev_before,
        "evidence_checksum_after": ev_after,
        "evidence_unchanged": ev_before == ev_after,
        "duration_s": round(time.time() - t_start, 3),
        "honest_verdict": "complete_shard_written_see_analysis",
        "note": "Analysis + the stopping-rule verdict are produced by the sibling analyse script.",
    }
    out["reproducibility_checksum"] = hashlib.sha256(
        json.dumps({k: v for k, v in out.items()}, sort_keys=True, default=str).encode()
    ).hexdigest()
    ARTIFACT.write_text(json.dumps(out, indent=1))
    log(f"wrote {ARTIFACT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
