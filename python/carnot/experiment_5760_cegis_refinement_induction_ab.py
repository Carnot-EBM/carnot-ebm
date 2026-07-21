"""Exp 5760: Does verifier-grounded CEGIS refinement lift world-model induction quality?
(REQ-ARC-WMTE-5760) -- the ONE experiment scoped by
docs/research-notes/arc-induction-quality-improvement-design-2026-07-20.md sec 5.

WHY THIS EXISTS
---------------
Tonight's induction-quality diagnosis (REQ-ARC-WMTE-5726 +
docs/research-notes/arc-world-model-induction-quality-diagnosis-2026-07-20.md) found that
37 SUCCESSFUL world-model inductions score heldout_accuracy min/median/mean/max =
0.0/0.0/0.124/1.0, with 29/37 at EXACTLY 0.0 -- the induced engine() never once reproduces
a full observed transition. The synthesis: the binding wall is GENERATING a correct world
model, not selecting/searching among candidates.

BUT (design sec 3b): every one of those numbers is a SINGLE-SHOT induction number. Tonight's
harness (arc_actions_to_progress.run_seeded_progress) calls proposer.induce() DIRECTLY and
stops (n_refinement_rounds=0 hardcoded at arc_actions_to_progress.py:764). The project's
verifier-grounded CEGIS refinement loop -- execute_bounded_llm_reinduction
(arc_llm_reinduction.py:654), which is BUILT, WIRED, and DEFAULT-ON LIVE
(arc_competition_agent.py:3885/4005) -- was simply BYPASSED. Whether that refinement lifts
the heldout numbers is UNMEASURED offline.

The literature is genuinely split on whether verifier-grounded counterexample refinement helps
a SMALL FROZEN model: WorldCoder (arXiv:2402.12275) + the E3 paper (arXiv:2605.05138, GPT-5.5
solves 15/25) say YES -- with a FRONTIER model. "Falsification, Not Exposure"
(arXiv:2606.31511) preregistered that for small FROZEN code models self-repair feedback does
NOT improve pass rate (only the falsification/FILTER component carries signal). Carnot's frozen
9B (and the 27B ThinkingCap) sit exactly on that disputed boundary. No one has measured which
side Carnot falls on. THAT is this experiment.

WHAT IT MEASURES
----------------
Re-run the SAME games/windows/budgets as tonight, but route induction through the EXISTING
execute_bounded_llm_reinduction CEGIS loop (min_heldout_accuracy=1.0 so the dynamics-refactor
rounds actually FIRE -- a lower threshold would let the loop accept-and-stop before refining,
defeating the measurement, design sec 6 point 3) instead of single-shot proposer.induce().
For each (game, trial, model) capture the per-round heldout_accuracy trajectory ALREADY recorded
in outcome.rounds[*]["heldout_accuracy"] (arc_llm_reinduction.py:790):

  * round 0 (round_no==1, action="induce")   = single-shot baseline (comparable to tonight)
  * rounds 1-2 (action="refactor")           = counterexample-guided refactor

PRIMARY metric:   delta_heldout = heldout(best refined round) - heldout(round 0), per game +
                  pooled, bootstrap 95% CI.
SECONDARY:        window-memorization detector -- a structural AST scan of the induced engine()
                  source BEFORE vs AFTER refinement, counting hardcoded literal coordinate
                  constants matching observed-window cells (the ls20 failure mode, design sec 5).
ATTRIBUTION:      refactor-round code-EMISSION rate -- so a null is attributable (budget-overrun
                  mechanical artifact vs. genuine no-improvement, design sec 5 branch 3).

PRE-REGISTERED FALSIFIABLE GATE (design sec 5, three honest branches):
  POSITIVE:          pooled mean delta_heldout > 0.15 across >=12 games, CI excludes 0, positive
                     on >=50% of games, paired sign-test p<0.05, memorization rate drops >=0.2
                     absolute, degradation guard holds (sp80/ft09 don't drop below round 0),
                     refactor-emission rate >0.6.
  HONEST-NEGATIVE:   delta_heldout <=0.05 pooled AND memorization unchanged AND emission healthy.
  EMISSION-CONFOUND: refactor-emission rate <=0.6 (mechanical artifact, NOT evidence against
                     refinement -- "fix code-emission first, then re-judge").

PROVENANCE: development_proxy on PUBLIC games (NOT a hidden-game self-discovery solve). This is
a DIAGNOSTIC over EXISTING live machinery (the refinement loop is what runs live) -- NOT a
live-path modification and NOT an orphan module (Live-Path Reachability Discipline: the mechanism
under test IS the live mechanism). verifier_is_oracle False (win oracle = the level counter).
NEVER flips the frozen live default, NEVER submits.

Prior-failure block (Failed-Experiment Rerun Discipline) -- names REQ-ARC-WMTE-5726 (single-shot
diagnosis) + the 7 peripheral-tweak nulls (REQ-ARC-FCP-5590/5728/5729/5730/5732/5740/5756); root
cause of those nulls was (a) single-shot induction bypassing the refinement loop and (b) a
level-gain delta on a near-zero-headroom corpus. What is different: this exercises the
EXISTING-but-offline-unmeasured refinement mechanism and measures a CONTINUOUS induction-quality
metric with REAL headroom (baseline median heldout is a measured 0.0). retire_if_same_verdict:
if HONEST-NEGATIVE, do NOT re-propose small-model induction-refinement variants; change the model
class (bigger offline model) or the architecture (reactive-with-filter).

RESUMABLE: every (arm, game, trial) cell appends to a JSONL shard as it completes.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Reuse tonight's proven dual-server model-serving setup VERBATIM (the task brief: "Reuse its
# model-serving setup rather than reinventing it"). Importing exp5726 runs only module-level
# config (GENERATORS/paths + os.environ.setdefault); run_all() is under its own __main__.
from carnot.experiment_5726_thinkingcap_16k_dualgpu_reason_ab import (  # noqa: E402
    BUDGET,
    GENERATORS,
    LLAMA_SERVER,
    N_CTX,
    _gpu_mem_used_mib,
    launch_server,
    make_proposer,
    terminate,
)

REPO = Path(__file__).resolve().parents[2]
SHARD = REPO / "results" / "exp5760_cegis_refinement_induction_shard.jsonl"
ARTIFACT = REPO / "results" / "experiment_5760_cegis_refinement_induction_ab.json"

# ---- PRE-REGISTERED design (fixed BEFORE the run; written to the artifact's preregistration
# block). Roster = the 12 fastest-to-induce (by tonight's mean ThinkingCap wall) of the 17 games
# ThinkingCap induced on tonight, PLUS ft09 -- so BOTH degradation-guard games named in design
# sec 5 (sp80 AND ft09, the round-0 heldout=1.0 template-match games) are present. 13 games >= the
# gate's >=12 floor while keeping the ~1-day GPU wall bounded (the 5 excluded games -- s5i5, cn04,
# bp35 induced 0/2, ft09-neighbours su15/m0r0, and lf52 at 1054s/cell -- are the slowest/lowest
# tonight; ft09 is re-added explicitly for the degradation guard). Fixed order = ascending tonight
# TC wall so the fastest/most-reliable games shard first.
ROSTER = [
    "tu93",
    "tr87",
    "sp80",
    "sb26",
    "re86",
    "g50t",
    "r11l",
    "cd82",
    "ar25",
    "lp85",
    "sk48",
    "vc33",
    "ft09",
]
# >=3 trials/seeds per game (design sec 5; tonight used 2). LLM sampling is server-side stochastic
# (no fixed sampling seed -- matches tonight's design exactly); trial index is the replicate id +
# recorded random_seed for provenance.
TRIALS = [0, 1, 2]
GEN_ORDER = ["thinkingcap27", "qwen9b"]  # ThinkingCap first (the higher-emission arm shards early)
ARM = {"thinkingcap27": "thinkingcap27_cegis", "qwen9b": "qwen9b_cegis"}
# round-0-heldout==1.0 template-match games (design sec 5 degradation guard: refinement must NOT
# corrupt an already-correct model).
DEGRADATION_GUARD_GAMES = ("sp80", "ft09")

MIN_HELDOUT_ACCURACY = 1.0  # design sec 5/6.3: force the dynamics-refactor rounds to fire
MEMORIZATION_MIN_MATCHES = 3  # >=3 window-matching large-coordinate literals -> flagged memorizing
MEMORIZATION_MIN_COORD = 10  # ignore literals <10 (action codes 0-9 / small colors), keep 10..63


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------------------
# Window-memorization detector (design sec 5): a structural AST scan of the induced engine()
# source counting hardcoded literal coordinate constants that match observed-window cell indices.
# The ls20 failure mode (diagnosis sec 4 pattern 4) hardcodes literal pixel coordinates -- e.g.
# "Place a block at (61, 13)" -- rather than inferring a rule, which is the mechanism behind
# high cell_recall + zero heldout (memorized coords reproduce the fitted transitions, generalize
# to nothing). The refactor prompt's "replace special cases with shared rules" is the direct
# countermeasure, so the RATE at which engines are flagged memorizing should DROP after refinement.
#
# HONEST heuristic caveat (documented, not hidden): on dense-change windows most coords 10..63
# appear as "changed" so the ABSOLUTE flag rate can be inflated by chance-matching. The GATED
# quantity is the BEFORE/AFTER delta, which controls for that confound symmetrically (both engines
# scored against the same window coords). Raw counts are recorded so the >=3 threshold is
# sensitivity-checkable post-hoc.
# ---------------------------------------------------------------------------
def _engine_source(full_source: str) -> str:
    """Return the source segment of the top-level `engine` function ('' if absent/unparseable)."""
    try:
        tree = ast.parse(full_source)
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "engine":
            seg = ast.get_source_segment(full_source, node)
            return seg or ""
    return ""


def _int_literals(src: str) -> list[int]:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    out: list[int] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, int)
            and not isinstance(node.value, bool)
        ):
            out.append(int(node.value))
    return out


def _window_changed_coords(window: list) -> set[int]:
    """Row/col indices (>= MEMORIZATION_MIN_COORD) of cells that CHANGED anywhere in the window --
    the coordinate constants a memorizing engine would hardcode."""
    coords: set[int] = set()
    for t in window:
        try:
            g = np.asarray(t.grid)
            ng = np.asarray(t.next_grid)
        except Exception:
            continue
        if g.shape != ng.shape:
            continue
        rs, cs = np.where(g != ng)
        for v in rs.tolist():
            coords.add(int(v))
        for v in cs.tolist():
            coords.add(int(v))
    return {v for v in coords if v >= MEMORIZATION_MIN_COORD}


def memorization_scan(full_source: str, coord_set: set[int]) -> dict[str, Any]:
    eng_src = _engine_source(full_source or "")
    lits = [v for v in _int_literals(eng_src) if v >= MEMORIZATION_MIN_COORD]
    n_match = sum(1 for v in lits if v in coord_set)
    return {
        "engine_source_found": bool(eng_src),
        "n_large_int_literals": len(lits),
        "n_window_coord_literals": int(n_match),
        "is_memorizing": bool(n_match >= MEMORIZATION_MIN_MATCHES),
    }


# ---------------------------------------------------------------------------
# The CEGIS refinement cell
# ---------------------------------------------------------------------------
def run_cegis_cell(
    game: str, prop: Any, *, trial: int, window: list, full_traj: list, cell: int
) -> dict[str, Any]:
    """Route induction through execute_bounded_llm_reinduction (min_heldout_accuracy=1.0) instead
    of single-shot proposer.induce(), and capture the per-round heldout_accuracy trajectory +
    the before/after window-memorization scan + the refactor emission rate.

    The proposer is configured to MIRROR tonight's reason-induce (codeonly OFF, /think, tries=1,
    the 16384 budget) so round 0 is the comparable single-shot baseline. Round 0 uses the loop's
    STANDARD proposer.induce (which is exactly what the LIVE path uses) rather than tonight's
    _induce_no_fence variant -- a minor divergence disclosed here; the rigorous quantity is the
    INTERNAL delta (round 0 vs refined, both via the same loop, same heldout metric)."""
    from carnot.agentic.arc_executable_world_model import E3_DIR, load_engine, plan_in_model
    from carnot.agentic.arc_llm_reinduction import (
        MAX_REFINEMENT_ROUNDS,
        execute_bounded_llm_reinduction,
    )

    root_grid = full_traj[0].grid if full_traj else None
    coord_set = _window_changed_coords(window)

    # Snapshot the on-disk engine() source after EACH successful load_engine. sources[0] is the
    # round-0 (induce) engine; sources[-1] is the final refined engine. Using first/last is robust
    # to any mid-loop selection exception (which would break a per-round index alignment) -- the
    # heldout TRAJECTORY comes from outcome.rounds[*] (recorded by the loop itself), NOT from this.
    sources: list[str] = []

    def _wrapped_load_engine(g: str):
        eng, goal = load_engine(g)  # raises if refactor wrote nothing/broken -> no snapshot
        try:
            src = (E3_DIR / g / "world_model.py").read_text()
        except Exception:
            src = ""
        sources.append(src)
        return eng, goal

    # Capture completion diagnostics across every generate call (overran attribution).
    stop_log: list[str] = []
    raw_len_log: list[int] = []
    orig_record = prop._record_completion_diagnostics

    def _record(response: dict, _orig=orig_record) -> None:
        _orig(response)
        stop_log.append(str(response.get("stop_type") or ""))
        raw_len_log.append(len(str(response.get("content") or "")))

    prop._record_completion_diagnostics = _record  # type: ignore[assignment]

    # Mirror tonight's reason-induce config; FORCE exact-match heldout (not cell_recall).
    saved_env_codeonly = os.environ.get("CARNOT_ARC_CODEONLY_INDUCE")
    saved_env_trust = os.environ.get("CARNOT_ARC_TRUST_METRIC")
    saved_prop = (prop.no_think_prefix, prop.max_tokens, prop.tries)
    os.environ["CARNOT_ARC_CODEONLY_INDUCE"] = "0"
    os.environ.pop(
        "CARNOT_ARC_TRUST_METRIC", None
    )  # default -> exact-match heldout (design metric)
    prop.no_think_prefix = "/think\n"
    prop.max_tokens = BUDGET
    prop.tries = 1

    t0 = time.time()
    err: Optional[str] = None
    outcome = None
    # exp5722 stale-engine guard: delete the prior engine so a FAILED induce cannot leave an
    # earlier run's world_model.py on disk for load_engine to silently re-read.
    _wm = E3_DIR / game / "world_model.py"
    try:
        _wm.unlink()
    except FileNotFoundError:
        pass
    try:
        outcome = execute_bounded_llm_reinduction(
            game=game,
            transitions=list(window),
            cell=int(cell),
            root_grid=root_grid,
            proposer=prop,
            # candidate_provider returns JUST the loaded engine (design sec 5): a single candidate
            # so select_trusted_world_model scores exactly that engine's heldout each round.
            candidate_provider=lambda engine, goal: [("loaded_world_model.py", engine, goal)],
            load_engine=_wrapped_load_engine,
            plan_in_model=plan_in_model,
            max_rounds=MAX_REFINEMENT_ROUNDS,
            min_heldout_accuracy=MIN_HELDOUT_ACCURACY,
        )
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"[:300]
    finally:
        prop._record_completion_diagnostics = orig_record  # type: ignore[assignment]
        if saved_env_codeonly is None:
            os.environ.pop("CARNOT_ARC_CODEONLY_INDUCE", None)
        else:
            os.environ["CARNOT_ARC_CODEONLY_INDUCE"] = saved_env_codeonly
        if saved_env_trust is not None:
            os.environ["CARNOT_ARC_TRUST_METRIC"] = saved_env_trust
        prop.no_think_prefix, prop.max_tokens, prop.tries = saved_prop

    return _summarize_cell(
        game=game,
        trial=trial,
        outcome=outcome,
        sources=sources,
        coord_set=coord_set,
        stop_log=stop_log,
        raw_len_log=raw_len_log,
        wall_s=round(time.time() - t0, 1),
        err=err,
    )


def _summarize_cell(
    *,
    game: str,
    trial: int,
    outcome: Any,
    sources: list[str],
    coord_set: set[int],
    stop_log: list[str],
    raw_len_log: list[int],
    wall_s: float,
    err: Optional[str],
) -> dict[str, Any]:
    rounds = list(getattr(outcome, "rounds", []) or []) if outcome is not None else []

    def _heldout(r: dict) -> Optional[float]:
        v = r.get("heldout_accuracy")
        return float(v) if isinstance(v, (int, float)) else None

    round0 = next((r for r in rounds if r.get("round") == 1 and r.get("action") == "induce"), None)
    round0_heldout = _heldout(round0) if round0 else None
    induce_ok = bool(round0.get("proposer_ok")) if round0 else False
    round1_loaded = round0_heldout is not None

    refactor_rows = [r for r in rounds if r.get("action") == "refactor"]
    refactor_heldouts = [h for r in refactor_rows if (h := _heldout(r)) is not None]
    n_refactor_attempted = len(refactor_rows)
    n_refactor_emitted = sum(
        1 for r in refactor_rows if r.get("proposer_ok") and _heldout(r) is not None
    )
    refactor_emission_rate = (
        round(n_refactor_emitted / n_refactor_attempted, 4) if n_refactor_attempted else None
    )

    best_refined_heldout = max(refactor_heldouts) if refactor_heldouts else None
    # delta_heldout = heldout(best refined round) - heldout(round 0). If no refactor round ran
    # (early-returned at round 0, or induce failed), no refinement occurred -> delta 0.0 when round0
    # is measured (best refined == round 0), None when round0 itself is unmeasured (induce failed).
    if round0_heldout is None:
        delta_heldout = None
    elif best_refined_heldout is None:
        delta_heldout = 0.0
    else:
        delta_heldout = round(best_refined_heldout - round0_heldout, 6)

    mem_before = memorization_scan(sources[0] if sources else "", coord_set)
    mem_after = memorization_scan(sources[-1] if sources else "", coord_set)

    slim_rounds = [
        {
            "round": r.get("round"),
            "action": r.get("action"),
            "proposer_ok": bool(r.get("proposer_ok")),
            "heldout_accuracy": _heldout(r),
            "prefix_accuracy": r.get("prefix_accuracy"),
            "plan_reaches_goal": r.get("plan_reaches_goal"),
            "skipped": r.get("skipped"),
        }
        for r in rounds
    ]

    return {
        "game": game,
        "trial": trial,
        # per-round heldout trajectory (the PRIMARY signal, recorded by the loop itself)
        "round0_heldout": round0_heldout,
        "refactor_heldouts": refactor_heldouts,
        "best_refined_heldout": best_refined_heldout,
        "delta_heldout": delta_heldout,
        # emission attribution
        "induce_ok": induce_ok,
        "round1_loaded": round1_loaded,
        "n_rounds": len(rounds),
        "n_refactor_attempted": n_refactor_attempted,
        "n_refactor_emitted": n_refactor_emitted,
        "refactor_emission_rate": refactor_emission_rate,
        # window-memorization (before vs after refinement)
        "mem_before": mem_before,
        "mem_after": mem_after,
        "mem_before_is_memorizing": mem_before["is_memorizing"],
        "mem_after_is_memorizing": mem_after["is_memorizing"],
        "n_engine_snapshots": len(sources),
        # loop outcome + overrun diagnostics
        "planned": bool(getattr(outcome, "planned", False)) if outcome is not None else False,
        "refinement_rounds_used": int(getattr(outcome, "refinement_rounds_used", 0) or 0)
        if outcome is not None
        else 0,
        "final_heldout": getattr(outcome, "heldout_accuracy", None)
        if outcome is not None
        else None,
        "loop_skipped": str(getattr(outcome, "skipped", "")) if outcome is not None else "",
        "n_generate_calls": len(stop_log),
        "n_limit_stops": sum(1 for s in stop_log if s == "limit"),
        "max_raw_completion_len": max(raw_len_log) if raw_len_log else 0,
        "rounds": slim_rounds,
        "wall_s": wall_s,
        "error": err,
    }


# ---------------------------------------------------------------------------
# Shard IO (resumable) -- mirrors exp5726
# ---------------------------------------------------------------------------
def _load_shard() -> dict[tuple[str, str, int], dict[str, Any]]:
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    if SHARD.exists():
        for line in SHARD.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[(r["arm"], r["game"], int(r["trial"]))] = r
    return rows


def _append_shard(row: dict[str, Any]) -> None:
    SHARD.parent.mkdir(parents=True, exist_ok=True)
    with SHARD.open("a") as f:
        f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Preconditions (Pre-Launch Preconditions Discipline) -- BEFORE any inference
# ---------------------------------------------------------------------------
def check_preconditions() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def _add(resource: str, ok: bool, detail: str = "") -> None:
        checks.append({"resource": resource, "available": bool(ok), "detail": detail})

    for gen in GEN_ORDER:
        gguf = GENERATORS[gen]["gguf"]
        ok = Path(gguf).exists()
        _add(f"gguf_cached::{gen}", ok, gguf if ok else f"MISSING {gguf}")

    binary_ok = Path(LLAMA_SERVER).exists()
    _add("llama_server_binary", binary_ok, str(LLAMA_SERVER))

    # CLAUDE.md 2026-07-06 CUDA-build rule: a CPU-only llama-cpp wheel is a red flag for the rig's
    # health even though THIS harness routes inference through the native llama-server; a False
    # here means the venv regressed to a CPU wheel (a genuine problem worth stopping + surfacing).
    try:
        from llama_cpp import llama_cpp as _b

        offload_ok = bool(_b.llama_supports_gpu_offload())
        _add("llama_cpp_gpu_offload", offload_ok, "llama_supports_gpu_offload()")
    except Exception as exc:
        _add("llama_cpp_gpu_offload", False, f"import failed: {type(exc).__name__}: {exc}"[:160])

    all_ok = all(c["available"] for c in checks)
    return {"all_ok": all_ok, "checks": checks}


def _write_blocked_artifact(precond: dict[str, Any], duration_s: float) -> None:
    missing = [c["resource"] for c in precond["checks"] if not c["available"]]
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(
        json.dumps(
            {
                "experiment": "experiment_5760_cegis_refinement_induction_ab",
                "schema": "carnot.exp5760.cegis_refinement_induction_ab.v1",
                "requirements": ["REQ-ARC-WMTE-5760"],
                "honest_verdict": f"blocked_{'_'.join(missing)[:80]}",
                "inference_substrate": "live_llm_inference",
                "preconditions_checked": precond["checks"],
                "solve_provenance": "development_proxy",
                "verifier_is_oracle": False,
                "duration_s": round(duration_s, 2),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


# ---------------------------------------------------------------------------
# Run loop -- per-model server, sequential (no contention, no world_model.py race)
# ---------------------------------------------------------------------------
def run_all() -> list[dict[str, Any]]:
    from carnot.agentic import arc_actions_to_progress as atp

    done = _load_shard()
    total = len(GEN_ORDER) * len(ROSTER) * len(TRIALS)
    log(f"resume: {len(done)}/{total} cells already in shard")
    log(f"building {len(ROSTER)} windows...")
    windows: dict[str, Any] = {}
    for game in ROSTER:
        w = atp.build_progress_window(game)
        windows[game] = w
        if w is None:
            log(f"SKIP {game}: no offline L1 window")

    for gen in GEN_ORDER:
        cfg = GENERATORS[gen]
        arm = ARM[gen]
        pending = [
            (game, t)
            for game in ROSTER
            for t in TRIALS
            if windows.get(game) is not None and (arm, game, t) not in done
        ]
        if not pending:
            log(f"generator {gen}: all cells present, skipping")
            continue
        log(
            f"=== {gen} ({cfg['repo_substr']}) : {len(pending)} cells | CUDA={cfg['cuda_visible']} "
            f"mtp={cfg['mtp']} extra='{cfg['extra']}' budget={BUDGET} ==="
        )
        try:
            server_gpu = int(str(cfg["cuda_visible"]).split(",")[0])
        except Exception:
            server_gpu = 0
        v_before = _gpu_mem_used_mib(server_gpu)
        proc = None
        try:
            proc = launch_server(cfg, cfg["gguf"])
            v_after = _gpu_mem_used_mib(server_gpu)
            log(
                f"  server healthy. VRAM gpu{server_gpu} {v_before}->{v_after} MiB "
                f"(gpu0={_gpu_mem_used_mib(0)} gpu1={_gpu_mem_used_mib(1)})"
            )
            # Runtime GPU-offload assertion (native-server analogue of the llama_cpp precond): the
            # model MUST have loaded onto the GPU. A CPU fallback would silently run ~1 day slow.
            if v_before is not None and v_after is not None and (v_after - v_before) < 1000:
                raise RuntimeError(
                    f"llama-server did not offload to GPU{server_gpu}: VRAM only "
                    f"{v_before}->{v_after} MiB (<1GB jump) -- refusing to run on CPU"
                )
            prop = make_proposer(cfg)
            for game, t in pending:
                window, full_traj, cell = windows[game]
                log(f"RUN {arm} {game} trial={t}")
                c0 = time.time()
                try:
                    row = run_cegis_cell(
                        game, prop, trial=t, window=window, full_traj=full_traj, cell=cell
                    )
                except Exception as exc:
                    row = {
                        "game": game,
                        "trial": t,
                        "error": f"cell_crash: {type(exc).__name__}: {exc}"[:300],
                        "delta_heldout": None,
                        "round0_heldout": None,
                        "wall_s": round(time.time() - c0, 1),
                    }
                row["generator"] = gen
                row["arm"] = arm
                row["game"] = game
                row["trial"] = t
                _append_shard(row)
                done[(arm, game, t)] = row
                log(
                    f"  -> round0={row.get('round0_heldout')} refactor={row.get('refactor_heldouts')} "
                    f"delta={row.get('delta_heldout')} emit={row.get('n_refactor_emitted')}/"
                    f"{row.get('n_refactor_attempted')} mem {row.get('mem_before_is_memorizing')}->"
                    f"{row.get('mem_after_is_memorizing')} planned={row.get('planned')} "
                    f"wall={row.get('wall_s')}s ({time.time() - c0:.0f}s)"
                )
        finally:
            terminate(proc)
    return list(done.values())


# ---------------------------------------------------------------------------
# Artifact + pre-registered gate evaluation
# ---------------------------------------------------------------------------
def _mean(xs: list[float]) -> Optional[float]:
    return float(np.mean(xs)) if xs else None


def _bootstrap_ci(per_game_vals: list[float], n_boot: int = 10000, seed: int = 0) -> dict[str, Any]:
    """Bootstrap 95% CI of the mean over GAMES (the unit of analysis)."""
    if len(per_game_vals) < 2:
        return {"lo": None, "hi": None, "n": len(per_game_vals), "excludes_0": False}
    rng = np.random.default_rng(seed)
    arr = np.asarray(per_game_vals, dtype=float)
    means = arr[rng.integers(0, len(arr), size=(n_boot, len(arr)))].mean(axis=1)
    lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    return {
        "lo": round(lo, 6),
        "hi": round(hi, 6),
        "n": len(arr),
        "excludes_0": bool(lo > 0 or hi < 0),
    }


def _per_game_delta(rows: list[dict[str, Any]], *, gen: Optional[str] = None) -> dict[str, float]:
    """Mean delta_heldout per game over trials (optionally restricted to one generator)."""
    by_game: dict[str, list[float]] = {}
    for r in rows:
        if gen is not None and r.get("generator") != gen:
            continue
        d = r.get("delta_heldout")
        if isinstance(d, (int, float)):
            by_game.setdefault(r["game"], []).append(float(d))
    return {g: float(np.mean(v)) for g, v in by_game.items() if v}


def _mem_rate(
    rows: list[dict[str, Any]], key: str, *, gen: Optional[str] = None
) -> Optional[float]:
    """Fraction of cells (with a round-0 engine emitted) flagged memorizing under `key`."""
    vals = [
        bool(r.get(key))
        for r in rows
        if (gen is None or r.get("generator") == gen) and r.get("round1_loaded")
    ]
    return round(float(np.mean(vals)), 4) if vals else None


def build_artifact(duration_s: float, precond: dict[str, Any]) -> dict[str, Any]:
    from carnot.agentic.arc_actions_to_progress import _sign_test_p

    rows = list(_load_shard().values())

    # ---- primary: per-game delta_heldout (pooled across both models, and per-model)
    pooled_delta = _per_game_delta(rows)
    per_game_list = sorted(pooled_delta.items())
    pooled_vals = [v for _, v in per_game_list]
    pooled_mean = _mean(pooled_vals)
    ci = _bootstrap_ci(pooled_vals)
    n_games = len(pooled_vals)
    positive_frac = round(float(np.mean([v > 0 for v in pooled_vals])), 4) if pooled_vals else None
    wins = sum(1 for v in pooled_vals if v > 1e-9)
    losses = sum(1 for v in pooled_vals if v < -1e-9)
    sign_p = _sign_test_p(wins, losses)

    per_model = {}
    for gen in GEN_ORDER:
        d = _per_game_delta(rows, gen=gen)
        vals = list(d.values())
        per_model[gen] = {
            "n_games": len(vals),
            "mean_delta_heldout": round(_mean(vals), 6) if vals else None,
            "positive_game_frac": round(float(np.mean([v > 0 for v in vals])), 4) if vals else None,
            "per_game": {g: round(v, 6) for g, v in sorted(d.items())},
        }

    # ---- secondary: window-memorization rate before vs after (pooled + per-model)
    mem_before_rate = _mem_rate(rows, "mem_before_is_memorizing")
    mem_after_rate = _mem_rate(rows, "mem_after_is_memorizing")
    mem_drop = (
        round(mem_before_rate - mem_after_rate, 4)
        if (mem_before_rate is not None and mem_after_rate is not None)
        else None
    )
    mem_by_model = {
        gen: {
            "before": _mem_rate(rows, "mem_before_is_memorizing", gen=gen),
            "after": _mem_rate(rows, "mem_after_is_memorizing", gen=gen),
        }
        for gen in GEN_ORDER
    }

    # ---- attribution: refactor code-emission rate (pooled)
    tot_attempted = sum(int(r.get("n_refactor_attempted") or 0) for r in rows)
    tot_emitted = sum(int(r.get("n_refactor_emitted") or 0) for r in rows)
    emission_rate = round(tot_emitted / tot_attempted, 4) if tot_attempted else None
    emission_by_model = {}
    for gen in GEN_ORDER:
        a = sum(int(r.get("n_refactor_attempted") or 0) for r in rows if r.get("generator") == gen)
        e = sum(int(r.get("n_refactor_emitted") or 0) for r in rows if r.get("generator") == gen)
        emission_by_model[gen] = round(e / a, 4) if a else None

    # ---- degradation guard: on round0==1.0 games (sp80/ft09), refined must NOT drop below round0
    degradation_violations = []
    for r in rows:
        if r.get("game") in DEGRADATION_GUARD_GAMES:
            r0 = r.get("round0_heldout")
            br = r.get("best_refined_heldout")
            if (
                isinstance(r0, (int, float))
                and r0 >= 0.999
                and isinstance(br, (int, float))
                and br < r0 - 1e-9
            ):
                degradation_violations.append(
                    {
                        "game": r["game"],
                        "trial": r["trial"],
                        "gen": r.get("generator"),
                        "round0": r0,
                        "best_refined": br,
                    }
                )
    degradation_guard_holds = len(degradation_violations) == 0

    # ---- pre-registered gate (design sec 5), three honest branches + an explicit partial catch-all
    emission_healthy = emission_rate is not None and emission_rate > 0.6
    ci_excludes_0 = bool(ci.get("excludes_0"))
    positive_gate = (
        pooled_mean is not None
        and pooled_mean > 0.15
        and n_games >= 12
        and ci_excludes_0
        and positive_frac is not None
        and positive_frac >= 0.5
        and sign_p is not None
        and sign_p < 0.05
        and mem_drop is not None
        and mem_drop >= 0.2
        and degradation_guard_holds
        and emission_healthy
    )
    mem_unchanged = mem_drop is not None and mem_drop < 0.2
    honest_negative_gate = (
        pooled_mean is not None and pooled_mean <= 0.05 and mem_unchanged and emission_healthy
    )
    emission_confound_gate = emission_rate is not None and emission_rate <= 0.6

    if emission_confound_gate:
        branch = "emission_confound"
        verdict = (
            f"complete_cegis_refinement_untestable_emission_confound_rate_{emission_rate}_le_0.6_"
            f"fix_code_emission_first_pooled_delta_{round(pooled_mean, 4) if pooled_mean is not None else None}_N{n_games}"
        )
    elif positive_gate:
        branch = "positive"
        verdict = (
            f"success_cegis_refinement_lifts_induction_pooled_delta_{round(pooled_mean, 4)}_"
            f"gt0.15_CI_{ci['lo']}_{ci['hi']}_posfrac_{positive_frac}_signp_{sign_p}_"
            f"memdrop_{mem_drop}_N{n_games}"
        )
    elif honest_negative_gate:
        branch = "honest_negative"
        verdict = (
            f"complete_cegis_refinement_null_small_frozen_model_pooled_delta_{round(pooled_mean, 4)}_"
            f"le0.05_memdrop_{mem_drop}_emission_{emission_rate}_healthy_corroborates_arxiv_2606.31511_N{n_games}"
        )
    else:
        branch = "partial_inconclusive"
        verdict = (
            f"complete_cegis_refinement_partial_pooled_delta_"
            f"{round(pooled_mean, 4) if pooled_mean is not None else None}_"
            f"CI_{ci['lo']}_{ci['hi']}_posfrac_{positive_frac}_signp_{sign_p}_memdrop_{mem_drop}_"
            f"emission_{emission_rate}_N{n_games}_does_not_cleanly_meet_a_preregistered_branch"
        )

    tc = GENERATORS["thinkingcap27"]
    qw = GENERATORS["qwen9b"]
    return {
        "experiment": "experiment_5760_cegis_refinement_induction_ab",
        "schema": "carnot.exp5760.cegis_refinement_induction_ab.v1",
        "requirements": ["REQ-ARC-WMTE-5760"],
        "prior_work_extended": ["REQ-ARC-WMTE-5726", "REQ-ARC-WMTE-5720"],
        "question": (
            "Does the EXISTING verifier-grounded CEGIS refinement loop "
            "(execute_bounded_llm_reinduction, min_heldout_accuracy=1.0) LIFT world-model induction "
            "quality (per-round heldout_accuracy) over tonight's SINGLE-SHOT baseline, for a frozen "
            "9B and 27B model? Which side of the WorldCoder/E3-vs-'Falsification-Not-Exposure' "
            "small-frozen-model boundary does Carnot fall on?"
        ),
        "inference_substrate": "live_llm_inference",
        "honest_verdict": verdict,
        "gate_branch": branch,
        "solve_provenance": "development_proxy",
        "verifier_is_oracle": False,
        "read_game_source": False,
        "used_env_source": True,
        "submitted_to_leaderboard": False,
        "random_seed": TRIALS[0],
        "trials_per_arm": len(TRIALS),
        "preregistration": {
            "roster": ROSTER,
            "roster_n": len(ROSTER),
            "roster_selection_rule": (
                "the 12 fastest-to-induce (by tonight REQ-ARC-WMTE-5726 mean ThinkingCap wall) of "
                "the 17 games ThinkingCap induced on tonight, PLUS ft09 (so both degradation-guard "
                "games sp80+ft09 are present). Fixed BEFORE the run; order = ascending TC wall."
            ),
            "trials": TRIALS,
            "models": [tc["repo_substr"], qw["repo_substr"]],
            "min_heldout_accuracy": MIN_HELDOUT_ACCURACY,
            "primary_metric": "delta_heldout = heldout(best refined round) - heldout(round 0), per game + pooled",
            "gate_positive": (
                "pooled mean delta_heldout>0.15 across >=12 games AND bootstrap 95% CI excludes 0 "
                "AND positive on >=50% of games AND paired sign-test p<0.05 AND memorization rate "
                "drops >=0.2 absolute AND degradation guard holds AND refactor-emission rate>0.6"
            ),
            "gate_honest_negative": "delta<=0.05 pooled AND memorization unchanged AND emission healthy",
            "gate_emission_confound": "refactor-emission rate<=0.6 (mechanical artifact, not evidence against refinement)",
            "retire_if_same_verdict": (
                "on HONEST-NEGATIVE, do NOT re-propose small-model induction-refinement variants; "
                "change the model class (bigger offline model) or the architecture (reactive-with-filter)"
            ),
        },
        "model_specs": [
            {
                "name": tc["repo_substr"],
                "hf_id": tc["hf_id"],
                "quant": "Q4_K_M",
                "gguf_path": tc["gguf"],
                "role": "primary refinement arm (higher tonight emission 31/40); /v1/chat/completions",
                "n_ctx": N_CTX,
                "budget": BUDGET,
            },
            {
                "name": qw["repo_substr"],
                "hf_id": qw["hf_id"],
                "quant": "Q4_K_M",
                "gguf_path": qw["gguf"],
                "role": "frozen live-generator arm (raw /completion); locates the model-size threshold",
                "n_ctx": N_CTX,
                "budget": BUDGET,
            },
        ],
        "primary_result": {
            "pooled_mean_delta_heldout": round(pooled_mean, 6) if pooled_mean is not None else None,
            "bootstrap_95ci": ci,
            "n_games": n_games,
            "positive_game_frac": positive_frac,
            "paired_sign_test_p": sign_p,
            "wins": wins,
            "losses": losses,
            "per_game_delta_pooled": {g: round(v, 6) for g, v in per_game_list},
            "per_model": per_model,
        },
        "memorization_detector": {
            "definition": (
                "structural AST scan of the induced engine() source; is_memorizing := >=3 int "
                "literals (>=10, i.e. coordinate range not action/color codes) matching an "
                "observed-window changed-cell row/col index. GATED quantity = before/after DELTA "
                "(cancels the dense-window chance-match confound symmetrically)."
            ),
            "rate_before_refinement": mem_before_rate,
            "rate_after_refinement": mem_after_rate,
            "rate_drop": mem_drop,
            "by_model": mem_by_model,
        },
        "emission_attribution": {
            "refactor_emission_rate_pooled": emission_rate,
            "refactor_rounds_attempted": tot_attempted,
            "refactor_rounds_emitted": tot_emitted,
            "by_model": emission_by_model,
            "healthy_threshold": 0.6,
        },
        "degradation_guard": {
            "games": list(DEGRADATION_GUARD_GAMES),
            "holds": degradation_guard_holds,
            "violations": degradation_violations,
        },
        "field_principles": {
            "honest_verdict": "terminal-prefixed, numbers-first; a continuous heldout delta with real headroom cannot come back 'no headroom'.",
            "inference_substrate": "live_llm_inference -- real ThinkingCap-27B + Qwen-9B GGUF generation across up to 3 CEGIS rounds/cell; 60s floor.",
            "random_seed": "LLM sampling is server-side stochastic; trials are per-game replicates (same seeded window) -- paired by GAME, bootstrap over games.",
            "reproducibility_checksum": "content hash over harness + reinduction + world-model code + generator/roster config + rows.",
            "solve_provenance": "development_proxy -- PUBLIC-game offline measurement of the LIVE refinement mechanism; NOT a hidden-game solve.",
            "verifier_is_oracle": "False -- heldout_accuracy is exact-match against real recorded transitions the engine was NOT fit to (held-out 1/3 split); oracle-distinct.",
            "delta_heldout": "the induction-quality lift the refinement loop produces; PRIMARY -- the exact mechanism the diagnosis named as the binding wall.",
            "refactor_emission_rate": "attribution guard -- a low rate means a null is a budget-overrun mechanical artifact, not evidence against refinement (design sec 5 branch 3).",
        },
        "preconditions_checked": precond["checks"],
        "sample_size": {
            "games": n_games,
            "roster_n": len(ROSTER),
            "trials_per_game": len(TRIALS),
            "paired_unit": "game (delta_heldout averaged over trials + models, paired by game)",
            "note": (
                "17 games ThinkingCap induced on tonight; 13 pre-registered here (>=12 gate floor). "
                "Trials add per-game stability, not additional independent degrees of freedom."
            ),
        },
        "methodology_note": (
            "Route induction through execute_bounded_llm_reinduction (arc_llm_reinduction.py:654, "
            "min_heldout_accuracy=1.0, candidate_provider=the loaded engine, load_engine + "
            "plan_in_model from arc_executable_world_model) instead of single-shot proposer.induce. "
            "Per-round heldout from outcome.rounds[*]['heldout_accuracy'] (exact-match on the held-out "
            "1/3 split via select_trusted_world_model; CARNOT_ARC_TRUST_METRIC forced OFF so the "
            "metric is exact, not cell_recall). round 0 = induce, rounds 1-2 = refactor. Proposer "
            "mirrors tonight's reason-induce (codeonly OFF, /think, tries=1, 16384 budget); round 0 "
            "uses the loop's STANDARD induce (the live path) -- disclosed minor divergence from "
            "tonight's _induce_no_fence, but the rigorous quantity is the INTERNAL round-0-vs-refined "
            "delta. Servers sequential (no contention, no world_model.py race). This is a DIAGNOSTIC "
            "over the EXISTING live refinement mechanism, NOT a live-path modification."
        ),
        "duration_s": round(duration_s, 2),
        "reproducibility_checksum": _repro_checksum(rows),
    }


def _repro_checksum(rows: list[dict[str, Any]]) -> str:
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic import arc_llm_reinduction as reind

    h = hashlib.sha256()
    for mod_file in (__file__, e3.__file__, reind.__file__):
        h.update(Path(mod_file).read_bytes())
    h.update(
        json.dumps(
            {
                "roster": ROSTER,
                "trials": TRIALS,
                "budget": BUDGET,
                "n_ctx": N_CTX,
                "min_heldout": MIN_HELDOUT_ACCURACY,
                "gen_order": GEN_ORDER,
            },
            sort_keys=True,
        ).encode()
    )
    h.update(json.dumps(sorted(json.dumps(r, sort_keys=True, default=str) for r in rows)).encode())
    return "sha256:" + h.hexdigest()


def main() -> int:
    started = time.time()
    precond = check_preconditions()
    for c in precond["checks"]:
        log(f"PRECOND {c['resource']}: available={c['available']} {c['detail']}")
    if not precond["all_ok"]:
        log("PRECONDITIONS FAILED -- writing blocked artifact and STOPPING (no inference).")
        _write_blocked_artifact(precond, time.time() - started)
        return 0
    log("preconditions OK -- starting CEGIS refinement A/B run.")
    run_all()
    artifact = build_artifact(time.time() - started, precond)
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    log(f"DONE. verdict={artifact['honest_verdict']}")
    log(f"artifact -> {ARTIFACT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
