"""Does TOOL-ROUTED refinement fix the measured-destructive CEGIS refactor round?

CONTEXT. The shipped refactor round improved held-out accuracy on 0 of 83 measured cells
(exp5760 + exp5766, three generators) and collapsed all 8 partially-correct round-0 engines
to 0.0. REQ-ARC-WMTE-6091 found the cause: the refactor prompt never contains the engine
being fixed, so the round is a blind re-induction from at most 5 mismatches. This experiment
measures the tool-loop repair alternative (REQ-ARC-WMTE-6480): the model sees the failed
engine WITH its measured report and can execute candidate repairs before submitting.

DESIGN -- paired at the sample. Per (game, trial): ONE real induce produces round-0 source.
Both arms then run the REAL `execute_bounded_llm_reinduction` (the exact shipped loop, with
the REQ-6480 wiring) from that SAME round-0 source, replayed into the store by a delegating
proposer wrapper whose only override is `induce`. The arms differ in exactly one env flag:

  text_blind  CARNOT_ARC_CEGIS_TOOL_LOOP unset  -> the shipped blind text refactor
  tool_loop   CARNOT_ARC_CEGIS_TOOL_LOOP=1      -> the tool-loop repair round

PRIMARY METRIC. Per-round `heldout_accuracy` from the loop's own record (the exp5760/5766
metric): delta_heldout = round-2 heldout minus round-1 (replayed round-0) heldout. Round-1
heldout must be IDENTICAL across arms (same source, same corpus) -- asserted per cell, so a
round-0 sampling-noise confound is structurally impossible, not just unlikely.

DISCLOSED DIVERGENCES from the shipped defaults, all shared by both arms identically:
  * max_rounds=2 (1 induce + 1 refactor), not 3. exp5760/5766 show the two shipped refactor
    rounds behave identically (refactor_heldouts [0,0] on every fired cell), so one round
    carries the contrast at half the wall.
  * Round-0 is a replay, so its `induce` wall is paid once, not per arm.
  * Generation config is the LIVE one (code-only induce, budget 4096), exp6091's choice for
    the same survivability reasons; exp6091 already reproduced the blind collapse under it.

LEVER-FIRED EVIDENCE, per tool cell: the round row's `tool_loop` stats (turns,
tool_calls_total, decode_tokens_total, mismatch/holdout trajectories) plus the round action
(`refactor_tool_loop` vs a fallback `refactor`). A cell whose tool arm fell back to text is
reported as such, never counted as a tool measurement.

EVIDENCE SAFETY. Requires CARNOT_ARC_E3_DIR pointing at a scratch store (refuses otherwise,
same guard as the exp6091 driver). Writes only results/arc_cegis_tool_loop_20260817/.
GPU/server/witness infrastructure is IMPORTED from the exp6091 driver -- one infra surface.

Spec: REQ-ARC-WMTE-6480
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "python") not in sys.path:
    sys.path.insert(0, str(REPO / "python"))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

OUT_DIR = REPO / "results" / "arc_cegis_tool_loop_20260817"
SHARD = OUT_DIR / "shard.jsonl"
ARTIFACT = OUT_DIR / "experiment_cegis_tool_loop_ab.json"

# ---- run configuration -----------------------------------------------------------------------
GPU_INDEX = int(os.environ.get("CARNOT_CEGIS_TL_GPU") or "1")
PORT = int(os.environ.get("CARNOT_CEGIS_TL_PORT") or "8991")  # never 8919 (default), never 8994
GAMES = [g for g in (os.environ.get("CARNOT_CEGIS_TL_GAMES") or "tu93,sb26,ar25").split(",") if g]
TRIALS = [int(x) for x in (os.environ.get("CARNOT_CEGIS_TL_TRIALS") or "0,1,2").split(",")]
MAX_WALL_S = float(os.environ.get("CARNOT_CEGIS_TL_MAX_WALL_S") or "10800")  # 3h default
MAX_ROUNDS = 2  # 1 replayed induce + 1 refinement round; see DISCLOSED DIVERGENCES.

_BASE_PATH = (
    REPO / "scripts" / "experiments" / "outer_loop_arc_refine_engine_visible_ab_20260803.py"
)


def _load_base():
    """Import the exp6091 driver as a module: server ladder, witness, reaping, GEMMA pin."""
    spec = importlib.util.spec_from_file_location("exp6091_base", _BASE_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class ReplayInduceProposer:
    """Delegates EVERYTHING to the real proposer except `induce`, which replays round-0.

    This is what lets both arms fork from one identical round-0 engine while still running
    the real shipped loop end to end (refactor, tool loop, retention, scoring untouched).
    """

    def __init__(self, real: Any, store: Path, game: str, source0: str) -> None:
        self._real = real
        self._store = Path(store)
        self._game = game
        self._source0 = source0
        self.replayed = 0

    def induce(self, game, trans, cell, **kwargs):
        path = self._store / game / "world_model.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._source0)
        self.replayed += 1
        return True, "replayed shared round-0 source"

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def run_cell(base, game: str, trial: int, window: list, cell: int, prop: Any) -> dict[str, Any]:
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic.arc_llm_reinduction import (
        _proposal_prefix,
        execute_bounded_llm_reinduction,
    )

    t0 = time.time()
    store = e3.E3_DIR
    wm_path = store / game / "world_model.py"
    row: dict[str, Any] = {"game": game, "trial": trial, "window_n": len(window), "cell": cell}

    # ---- one real induce, shared by both arms ------------------------------------------------
    try:
        wm_path.unlink()
    except FileNotFoundError:
        pass
    t_ind = time.time()
    ok, msg = prop.induce(game, list(_proposal_prefix(list(window))), int(cell))
    row["induce_ok"] = bool(ok)
    row["induce_wall_s"] = round(time.time() - t_ind, 1)
    row["induce_message"] = str(msg)[:200]
    source0 = ""
    try:
        source0 = wm_path.read_text()
    except OSError:
        pass
    row["round0_source_chars"] = len(source0)
    if not ok or not source0.strip():
        row["skipped"] = "no_round0_engine"
        row["wall_s"] = round(time.time() - t0, 1)
        return row

    root = window[0].grid

    for arm, flag in (("text_blind", None), ("tool_loop", "1")):
        arm_row: dict[str, Any] = {}
        wm_path.parent.mkdir(parents=True, exist_ok=True)
        wm_path.write_text(source0)  # FORK POINT: identical starting engine for both arms
        replay = ReplayInduceProposer(prop, store, game, source0)
        prev = os.environ.get("CARNOT_ARC_CEGIS_TOOL_LOOP")
        if flag is None:
            os.environ.pop("CARNOT_ARC_CEGIS_TOOL_LOOP", None)
        else:
            os.environ["CARNOT_ARC_CEGIS_TOOL_LOOP"] = flag
        t_arm = time.time()
        try:
            outcome = execute_bounded_llm_reinduction(
                game=game,
                transitions=list(window),
                cell=int(cell),
                root_grid=root,
                proposer=replay,
                candidate_provider=lambda engine, goal: [("loaded_world_model.py", engine, goal)],
                load_engine=e3.load_engine,
                plan_in_model=lambda engine, goal, grid: None,
                max_rounds=MAX_ROUNDS,
                min_heldout_accuracy=1.0,
            )
            arm_row["rounds"] = [
                {
                    k: r.get(k)
                    for k in (
                        "round",
                        "action",
                        "proposer_ok",
                        "heldout_accuracy",
                        "prefix_accuracy",
                        "retained_as_best_engine",
                        "skipped",
                        "tool_loop",
                        "message",
                    )
                    if k in r
                }
                for r in outcome.rounds
            ]
            arm_row["refinement_rounds_used"] = int(outcome.refinement_rounds_used)
            arm_row["loop_skipped"] = str(outcome.skipped or "")
            hs = [r.get("heldout_accuracy") for r in outcome.rounds]
            arm_row["round_heldouts"] = hs
            r0 = hs[0] if hs else None
            refined = [h for h in hs[1:] if h is not None]
            arm_row["round0_heldout"] = r0
            arm_row["best_refined_heldout"] = max(refined) if refined else None
            arm_row["delta_heldout"] = (
                round(max(refined) - r0, 6) if (refined and r0 is not None) else None
            )
        except Exception as exc:  # record, never drop
            arm_row["error"] = f"{type(exc).__name__}: {exc}"[:300]
        finally:
            if prev is None:
                os.environ.pop("CARNOT_ARC_CEGIS_TOOL_LOOP", None)
            else:
                os.environ["CARNOT_ARC_CEGIS_TOOL_LOOP"] = prev
        arm_row["replayed_induces"] = replay.replayed
        arm_row["wall_s"] = round(time.time() - t_arm, 1)
        row[arm] = arm_row

    # Consistency check: the replayed round-0 must score identically in both arms.
    r0a = (row.get("text_blind") or {}).get("round0_heldout")
    r0b = (row.get("tool_loop") or {}).get("round0_heldout")
    row["round0_heldout_identical_across_arms"] = r0a == r0b
    row["wall_s"] = round(time.time() - t0, 1)
    return row


def main() -> int:
    t_start = time.time()
    from carnot.agentic import arc_executable_world_model as e3

    if e3.E3_DIR.resolve() == (REPO / "results" / "arc_e3").resolve():
        log("REFUSING: CARNOT_ARC_E3_DIR is unset; would write the tracked evidence store.")
        return 1
    log(f"engine store redirected to {e3.E3_DIR}")

    base = _load_base()
    base.PORT = PORT
    base.GPU_INDEX = GPU_INDEX
    base.SERVER_LOG = Path(
        os.environ.get("CARNOT_CEGIS_TL_SERVER_LOG") or "/tmp/cegis_tool_loop_llama.log"
    )
    ok, checks = base.check_preconditions()
    if not ok:
        log(f"BLOCKED preconditions: {[c['resource'] for c in checks if not c['available']]}")
        return 1
    log("preconditions OK")

    from carnot.agentic import arc_actions_to_progress as atp

    windows: dict[str, tuple[list, int]] = {}
    for game in GAMES:
        try:
            built = atp.build_progress_window(game)
        except Exception as exc:
            log(f"{game}: window build raised {type(exc).__name__}: {exc}")
            built = None
        if built:
            windows[game] = (list(built[0]), int(built[2]))
            log(f"{game}: window n={len(built[0])} cell={built[2]}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, int]] = set()
    if SHARD.exists():
        for line in SHARD.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["game"], int(r["trial"])))
            except Exception:
                pass
    pending = [(g, t) for g in GAMES for t in TRIALS if g in windows and (g, t) not in done]
    log(f"resume: {len(done)} done; {len(pending)} pending")
    if not pending:
        return 0

    proc = None
    try:
        proc, n_ctx, v0, v1 = base.launch_server_ladder()
        log(f"server up n_ctx={n_ctx} vram {v0}->{v1} MiB")
        from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

        prop = LocalGGUFProposer(
            repo_substr=base.GEMMA["repo_substr"],
            port=PORT,
            mtp=False,
            kv_quant=base.GEMMA["kv_quant"],
            n_ctx=n_ctx,
            max_tokens=int(os.environ.get("CARNOT_CEGIS_TL_BUDGET") or "4096"),
            timeout=1800,
            use_chat_template=True,
            model_path=base.GEMMA["gguf"],
        )
        prop.tries = 2
        os.environ["CARNOT_ARC_CODEONLY_INDUCE"] = "1"
        os.environ.setdefault("CARNOT_ARC_CHAT_EMPTY_CONTENT_FALLBACK", "1")
        # End perception-only tool cells at the fallback instead of burning the turn cap.
        os.environ.setdefault("CARNOT_ARC_INDUCE_TOOL_STALL_TURNS", "4")

        for i, (game, trial) in enumerate(pending, 1):
            if time.time() - t_start > MAX_WALL_S:
                log(f"WALL BUDGET reached; {len(pending) - i + 1} cells unrun")
                break
            wit_before = base.substrate_witness(PORT)
            if not wit_before["is_cuda"]:
                if wit_before.get("loaded_hip"):
                    log(f"ABORT: HIP build owns the port: {wit_before}")
                    break
                log("server gone; relaunching")
                base.terminate(proc)
                proc, n_ctx, v0, v1 = base.launch_server_ladder()
                wit_before = base.substrate_witness(PORT)
                if not wit_before["is_cuda"]:
                    log("ABORT: relaunched server is not CUDA")
                    break
            w, cell = windows[game]
            log(f"[{i}/{len(pending)}] {game} trial={trial} (n={len(w)})")
            try:
                r = run_cell(base, game, trial, w, cell, prop)
            except Exception as exc:
                r = {"game": game, "trial": trial, "error": f"{type(exc).__name__}: {exc}"[:300]}
            wit_after = base.substrate_witness(PORT)
            r["substrate_witness_before"] = wit_before
            r["substrate_witness_after"] = wit_after
            r["substrate_cuda_throughout"] = bool(
                wit_before["is_cuda"]
                and wit_after["is_cuda"]
                and wit_before["pid"] == wit_after["pid"]
            )
            with SHARD.open("a") as f:
                f.write(json.dumps(r, default=str) + "\n")
            tb = (r.get("text_blind") or {}).get("delta_heldout")
            tl = (r.get("tool_loop") or {}).get("delta_heldout")
            log(
                f"    r0={((r.get('text_blind') or {}).get('round0_heldout'))} "
                f"text_delta={tb} tool_delta={tl} ({r.get('wall_s')}s)"
            )
    finally:
        base.terminate(proc)
        log("server terminated (reaped)")

    # ---- summary artifact --------------------------------------------------------------------
    rows = [json.loads(line) for line in SHARD.read_text().splitlines() if line.strip()]
    cells = [r for r in rows if "text_blind" in r and "tool_loop" in r]
    paired = [
        (
            r["game"],
            r["trial"],
            (r["text_blind"] or {}).get("delta_heldout"),
            (r["tool_loop"] or {}).get("delta_heldout"),
        )
        for r in cells
    ]
    out = {
        "experiment": "outer_loop_arc_cegis_tool_loop_ab_20260817",
        "spec": "REQ-ARC-WMTE-6480",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inference_substrate": "live_llm_inference",
        "solve_provenance": "development_proxy",
        "verifier_is_oracle": False,
        "read_game_source": False,
        "random_seed": 0,
        "n_cells": len(cells),
        "paired_deltas": paired,
        "config": {
            "games": GAMES,
            "trials": TRIALS,
            "max_rounds": MAX_ROUNDS,
            "min_heldout_accuracy": 1.0,
            "budget_max_tokens": int(os.environ.get("CARNOT_CEGIS_TL_BUDGET") or "4096"),
            "codeonly_induce": "1",
            "accept_split": os.environ.get("CARNOT_ARC_CEGIS_ACCEPT_SPLIT", "unset"),
            "port": PORT,
            "gpu_index": GPU_INDEX,
        },
        "duration_s": round(time.time() - t_start, 1),
        "honest_verdict": "complete_shard_written_see_report",
    }
    ARTIFACT.write_text(json.dumps(out, indent=1))
    log(f"wrote {ARTIFACT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
