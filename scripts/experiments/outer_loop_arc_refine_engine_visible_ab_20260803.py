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
BUDGET = 16384  # matched to exp5760/5764/5766 so round 0 is comparable to their single-shot.
NCTX_LADDER = [32768, 24576, 20480]
REFACTOR_ROUNDS = int(os.environ.get("CARNOT_6091_ROUNDS") or "2")
TRIALS = [int(x) for x in (os.environ.get("CARNOT_6091_TRIALS") or "0,1").split(",")]
MAX_WALL_S = float(os.environ.get("CARNOT_6091_MAX_WALL_S") or "28800")  # 8h default
GPU_IDLE_MAX_MIB = 2000
SEED = 6091

GEMMA: dict[str, Any] = {
    "repo_substr": "gemma-4-31B-it",
    "hf_id": "unsloth/gemma-4-31B-it-GGUF",
    "gguf": (
        "/home/ianblenke/.cache/huggingface/hub/models--unsloth--gemma-4-31B-it-GGUF/"
        "snapshots/f130ba51393346288f5862e30e9586b9b021513f/gemma-4-31B-it-Q4_K_M.gguf"
    ),
    "kv_quant": "q8_0",
    "timeout": 1800,
}
LLAMA_SERVER = Path.home() / ".cache" / "llama.cpp-master" / "build" / "bin" / "llama-server"


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
    return [
        str(LLAMA_SERVER),
        "-m",
        GEMMA["gguf"],
        "-ngl",
        "999",
        "-c",
        str(n_ctx),
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


def _launch_one(n_ctx: int) -> subprocess.Popen:
    args = _server_args(n_ctx)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(GPU_INDEX))
    log(f"  launch n_ctx={n_ctx} CUDA_VISIBLE_DEVICES={GPU_INDEX} port={PORT}")
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
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
    """REAP WHAT YOU START."""
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=30)
    except Exception:
        try:
            proc.kill()
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
                rr["prompt_contains_engine"] = bool(
                    "THE CURRENT ENGINE YOU ARE FIXING" in rendered
                )
                rr["n_mismatches_available"] = len(rv.mismatches)
                t_r = time.time()
                ok, msg = prop.refactor(game, vr_obj)
                rr["refactor_ok"] = bool(ok)
                rr["wall_s"] = round(time.time() - t_r, 1)
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

        scored = [
            x["acceptance"] for x in arm_rows if isinstance(x.get("acceptance"), dict)
        ]
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
    pending = [
        (g, t) for t in TRIALS for g in ROSTER if g in windows and (g, t) not in done
    ]
    log(f"resume: {len(done)} cells in shard; {len(pending)} pending")

    proc = None
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
            prop.no_think_prefix = "/think\n"
            prop.tries = 1
            os.environ["CARNOT_ARC_CODEONLY_INDUCE"] = "0"

            for i, (game, trial) in enumerate(pending, 1):
                if time.time() - t_start > MAX_WALL_S:
                    log(f"WALL BUDGET reached; stopping with {len(pending) - i + 1} cells unrun")
                    break
                w, cell = windows[game]
                log(f"[{i}/{len(pending)}] {game} trial={trial} (n={len(w)})")
                try:
                    r = run_cell(game, trial, w, cell, prop)
                except Exception as exc:
                    r = {
                        "game": game,
                        "trial": trial,
                        "error": f"{type(exc).__name__}: {exc}"[:300],
                    }
                append_shard(r)
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
