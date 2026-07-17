"""Experiment 5714: RESCOPED `/think` vs `/no_think` induction-quality A/B on the frozen
live ARC generator (Qwen3.5-9B-MTP), re-testing the previously-negative REQ-ARC-WMTE-5594.

WHY RE-RUN A NEGATIVE (Failed-Experiment Rerun Discipline -- named prior + root cause +
what is different + falsifiable gate; see the `prior_failures` block below and
REQ-ARC-WMTE-5714 in openspec/capabilities/arc-human-replay-frame-change/spec.md):

  PRIOR: REQ-ARC-WMTE-5594 / results/experiment_5594_think_mode_induction_quality_ab.json
    verdict `complete: think_mode_ab_equal_success_no_think_higher_accuracy`. On N=2 games
    (m0r0, sk48) both arms induced (4/4) but mean heldout_accuracy favored /no_think
    (0.75) over /think (0.5). Disclosed limits: (1) N=2 is below this project's sample-size
    floor for any percentage-point claim; (2) heldout_accuracy scores world-model INDUCTION
    dynamics only, not a win-recognition / actions-to-first-win metric; (3) NEITHER game's
    10-transition window contained a real level-up, so `score_goal_predicate_consistency`
    (REQ-ARC-WMTE-5593, the goal-hypothesis half of induction quality) was NEVER exercised.

  WHAT IS DIFFERENT NOW (the four deltas):
    1. Live-agent code churn since 2026-07-13 (the REQ-ARC-FCP-5699-* series, incl. -38/-39
       which fixed four real goal-energy-bias bugs that could have confounded planning
       quality asymmetrically between arms). The prior test predates that fix; it is stale.
    2. MTP re-enabled on Kaggle-L4-matched hardware. The live SUBMISSION kernel hardcodes
       CARNOT_ARC_MTP=0 (a June decision under a presumed tight 16GB ceiling); Kaggle's scored
       hardware has been NvidiaL4 (24GB/card) for weeks. This experiment runs against a FRESH
       MTP-ON llama-server pinned to ONE local RTX 3090 (24GB) -- i.e. VRAM-matched to a single
       Kaggle L4 card -- via CARNOT_ARC_GENERATOR_CUDA_GPU. (The prior test's proposer passed
       mtp=True but CONNECTED to a pre-warmed port-8920 server of UNKNOWN launch config, so its
       true MTP state is not actually knowable from the artifact -- this run removes that ambiguity.)
    3. Materially larger, level-up-targeted roster (adaptered games solved to L1 offline; each
       window STRADDLES a real L0->L1 level-up so the goal-predicate metric fires -- the prior's
       biggest gap).
    4. A fuller, more decision-relevant metric than static heldout_accuracy: per arm we ALSO
       measure whether the induced `is_level_complete` RECOGNIZES the real win at the actual
       level-up transition (`levelup_positive_recall`) -- the induction-quality signal that
       most determines whether a discovered hypothesis leads to real progress, which is the
       thing GPT-5.6's reasoning-effort scaling on ARC-AGI-3 actually improved.

MECHANISM (found by REQ-ARC-WMTE-5594, RE-CONFIRMED here, not re-derived): the frozen live
stack induces with CARNOT_ARC_CODEONLY_INDUCE ON, and codeonly mode's `_L2_CODEONLY_DIRECTIVE`
module constant hardcodes `/no_think\n` as its FIRST line AND its BODY says "Do NOT reason ...
Skip all reasoning." So (a) the `no_think_prefix` instance attribute is dead on the induce
path, and (b) merely swapping the toggle `/no_think`->`/think` inside that directive is INERT
-- the "skip all reasoning" body overrides the toggle and the model emits code with NO <think>
trace (verified directly: codeonly-toggle output opens with `import numpy`, no reasoning tag).
Genuinely engaging reasoning requires codeonly OFF + a `/think` prefix, at which point the model
reasons at length (opens with `<think>`), often exhausting its token budget before emitting code.

We therefore measure TWO distinct comparisons, both on the SAME shared per-game level-up window
(only the /think vs /no_think axis differs within a comparison -- no arm-dependent-window
confound, unlike the prior which ran run_game separately per arm):

  Comparison A -- FROZEN-STACK TOGGLE (codeonly ON): A1 no_think  vs  A2 think_toggle.
    This is the operator's LITERAL decision ("flip the live stack's /no_think to /think?").
  Comparison B -- GENUINE REASONING (codeonly OFF): B1 no_think_plain vs B2 think_plain.
    This tests the GPT-5.6 hypothesis (does REAL reasoning improve induction quality?), a
    pipeline change larger than a toggle flip -- reported as a secondary, smaller-N signal
    because B2's full-reasoning calls are ~10-20x slower.

GUARDRAIL: OFFLINE DEV MEASUREMENT ONLY. This NEVER edits the frozen submission stack
(scripts/kaggle/submission_kernel/main.py, arc_competition_agent.py live defaults). It reports
a delta + an explicit recommendation; the operator decides whether to act (CLAUDE.md
frozen-live-stack + Operator-Only disciplines). solve_provenance=development_proxy.

Spec refs: REQ-ARC-WMTE-5714, SCENARIO-ARC-WMTE-5714-CODEONLY-TOGGLE-INERT,
SCENARIO-ARC-WMTE-5714-GENUINE-THINK-ENGAGES, SCENARIO-ARC-WMTE-5714-BLOCKS-CLEANLY.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for _p in (PYTHON_ROOT, REPO_ROOT, SCRIPTS_ROOT):
    if str(_p) not in sys.path:  # pragma: no cover - direct script guard
        sys.path.insert(0, str(_p))

JsonDict = dict[str, Any]

EXPERIMENT_ID = "experiment_5714_think_mode_rescoped_ab"
RESULT_RELATIVE_PATH = "results/experiment_5714_think_mode_rescoped_ab.json"
SCHEMA = "carnot.exp5714.think_mode_rescoped_ab.v1"
INFERENCE_SUBSTRATE = "live_llm_inference"
RANDOM_SEED = 5714

# Adaptered games solvable to L1 offline (cheap, reliable, arm-independent window source).
DEFAULT_ROSTER = (
    "dc22",
    "m0r0",
    "sk48",
    "ls20",
    "cd82",
    "ft09",
    "tr87",
    "vc33",
    "cn04",
    "sp80",
    "ar25",
    "ka59",
)
# Comparison B (genuine-reasoning, no-fence single call) runs on the SAME roster as A -- a clean
# no-fence /think call reasons concisely then emits code (~30-60s), fast enough for full coverage;
# an occasional long-reasoning overrun surfaces as induction_ok=False (the honest finding).
DEFAULT_B_ROSTER = DEFAULT_ROSTER
WINDOW_K = 12
GGUF_REPO_SUBSTR = "Qwen3.5-9B-MTP"
CUDA_GPU_INDEX = "1"  # outer-loop's RTX 3090 (24GB) -- VRAM-matched to one Kaggle L4 card
SERVER_PORT = 8930
REASONING_TAGS = ("<think", "</think", "<thinking", "<reasoning")

MODEL_SPECS = [
    {
        "name": "Qwen3.5-9B-MTP",
        "hf_id": "unsloth/Qwen3.5-9B-MTP-GGUF",
        "role": "E3AgentPolicy default tier-3 world-model induction proposer",
        "server": "fresh MTP-on CUDA llama-server, -ngl 999, q8_0 KV, pinned to one RTX 3090",
        "mtp": True,
        "kaggle_parity": "single RTX 3090 (24GB) == single Kaggle NvidiaL4 card (24GB)",
    }
]

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "model_specs",
    "mtp_enabled",
    "roster",
    "b_roster",
    "per_game_results",
    "comparison_a_summary",
    "comparison_b_summary",
    "comparison_genuine_vs_frozen_summary",
    "codeonly_toggle_inert",
    "solve_provenance",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal-prefixed; /think helping, hurting, tying, or being INERT under "
        "codeonly are all distinct, real, citable outcomes -- the last is the decision-critical one"
    },
    "inference_substrate": {
        "principle": "live_llm_inference -- every arm invokes the real MTP-on Qwen3.5-9B-MTP server "
        "on a 3090, not mocked; 60s duration floor applies to the whole run"
    },
    "mtp_enabled": {
        "principle": "delta #2: run against a FRESH MTP-ON server (not the submission kernel's "
        "hardcoded CARNOT_ARC_MTP=0) on L4-VRAM-matched hardware, removing the prior's unknown-MTP ambiguity"
    },
    "codeonly_toggle_inert": {
        "principle": "the load-bearing mechanistic finding: under the frozen codeonly path, flipping "
        "/no_think->/think produces NO reasoning trace, so the operator's literal toggle flip is a no-op"
    },
    "levelup_positive_recall": {
        "principle": "delta #4: does the induced is_level_complete RECOGNIZE the real win at the actual "
        "level-up transition? This is the win-recognition signal reasoning-effort is supposed to improve, "
        "and the prior never measured it (no level-up in any prior window)"
    },
    "solve_provenance": {
        "principle": "development_proxy -- adaptered offline dev-twin windows, NOT a live-agent hidden-game "
        "self-discovery solve; declared so adversarial_verify can tell them apart"
    },
    "random_seed": {"principle": "determinism precondition for reproducibility"},
    "reproducibility_checksum": {"principle": "content hash catches silent drift on replay"},
}


# --------------------------------------------------------------------------------------
# preconditions
# --------------------------------------------------------------------------------------
def preconditions(root: Path = REPO_ROOT) -> JsonDict:
    checks: dict[str, bool] = {}
    try:
        from carnot.agentic import arc_solver_kit as kit  # noqa: F401
        from carnot.agentic import arc_game_adapters as adapters

        checks["offline_arcade_importable"] = True
        checks["adapters_available"] = len(adapters.adaptered_games()) >= 5
    except Exception:
        checks["offline_arcade_importable"] = False
        checks["adapters_available"] = False
    try:
        from carnot.agentic.arc_executable_world_model import (  # noqa: F401
            LocalGGUFProposer,
            WorldModelVerifier,
            load_engine,
            score_goal_predicate_consistency,
            _resolve_gguf,
        )

        checks["e3_import"] = True
        checks["gguf_cached"] = _resolve_gguf(GGUF_REPO_SUBSTR) is not None
    except Exception:
        checks["e3_import"] = False
        checks["gguf_cached"] = False
    try:
        base = Path.home() / ".cache" / "llama.cpp-master"
        checks["cuda_llama_server_present"] = (base / "build" / "bin" / "llama-server").exists()
    except Exception:
        checks["cuda_llama_server_present"] = False
    # The MTP-on server needs ~11.5GB. Either it is ALREADY up on our port (reuse it, no launch
    # -> no fresh headroom needed) OR GPU 1 must have >=13000 MB free to launch it fresh.
    server_already_up = False
    try:
        import urllib.request

        with urllib.request.urlopen(f"http://127.0.0.1:{SERVER_PORT}/health", timeout=2) as resp:
            server_already_up = b"ok" in resp.read()
    except Exception:
        server_already_up = False
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        idx = int(CUDA_GPU_INDEX)
        has_headroom = 0 <= idx < len(lines) and int(lines[idx]) >= 13000
        checks["cuda_server_up_or_gpu1_headroom"] = bool(server_already_up or has_headroom)
    except Exception:
        checks["cuda_server_up_or_gpu1_headroom"] = bool(server_already_up)
    checks["ok"] = all(v for k, v in checks.items() if k != "ok")
    return checks


def _first_precondition_miss(preconds: JsonDict) -> Optional[str]:
    for key, value in preconds.items():
        if key != "ok" and not value:
            return key
    return None


def _checksum(payload: JsonDict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


# --------------------------------------------------------------------------------------
# shared level-up window construction (arm-independent, deterministic, no LLM)
# --------------------------------------------------------------------------------------
def _select_levelup_window(trans: list, k: int = WINDOW_K) -> Optional[list]:
    """Pure window-selection: from a replayed transition list, return up to `k` transitions
    ENDING at (and including) the LAST real level-up transition (level_after > level_before),
    a single-level-boundary window per score_goal_predicate_consistency's caller contract.
    Returns None if no real level-up is present."""
    levelups = [i for i, t in enumerate(trans) if t.level_after > t.level_before]
    if not levelups:
        return None
    j = levelups[-1]
    return trans[max(0, j - (k - 1)) : j + 1]


def build_levelup_window(game: str, k: int = WINDOW_K) -> Optional[tuple[list, int]]:
    """Solve `game` to L1 with its offline adapter, replay the winning labels against the
    offline env, and return (window, cell): up to `k` Transition objects ENDING at (and
    including) the L0->L1 level-up transition -- a single-level-boundary window that
    satisfies score_goal_predicate_consistency's caller contract and guarantees the
    goal-predicate metric fires. Returns None if no real level-up is captured."""
    import arc_loop_solve as loop
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic import arc_game_adapters as adapters
    from carnot.agentic.arc_executable_world_model import Transition, to_logical, detect_cell
    from carnot.agentic.arc_agi3_world_model import grid_of
    from carnot.agentic.arc_solver_kit import frame_level

    res = loop.solve_adaptered(game, 1)
    labels = res.get("solution_labels") or []
    if not labels or int(res.get("reached_level", 0)) < 1:
        return None
    ad = adapters.get_adapter(game)
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    f = env.reset()
    cell = detect_cell(grid_of(f))
    if ad.warmup_label is not None:
        f = ad.apply(env, ad.warmup_label, f)
    prev_g = to_logical(grid_of(f), cell)
    prev_lvl = frame_level(f)
    trans: list = []
    for lbl in labels:
        f = ad.apply(env, lbl, f)
        g1 = to_logical(grid_of(f), cell)
        lvl = frame_level(f)
        act = (
            json.loads(lbl)
            if isinstance(lbl, str) and lbl.strip().startswith("{")
            else {"action": lbl}
        )
        trans.append(
            Transition(prev_g, int(act.get("action", 0)), act.get("data"), g1, prev_lvl, lvl)
        )
        prev_g, prev_lvl = g1, lvl
    window = _select_levelup_window(trans, k)
    if window is None:
        return None
    return window, cell


# --------------------------------------------------------------------------------------
# one arm on a shared window
# --------------------------------------------------------------------------------------
def _configure_arm(prop, arm: str) -> Any:
    """Set env + proposer config for an arm; return the original _L2_CODEONLY_DIRECTIVE so the
    caller restores it in a finally. Arms:
      A1 no_think        : codeonly ON, directive unchanged (/no_think). Frozen-stack baseline.
      A2 think_toggle    : codeonly ON, directive /no_think->/think. The operator's literal toggle.
      B1 no_think_plain  : codeonly OFF, NO fence, no_think_prefix=/no_think. Non-codeonly baseline.
      B2 think_plain     : codeonly OFF, NO fence, no_think_prefix=/think. Genuinely engages reasoning.
    """
    from carnot.agentic import arc_executable_world_model as e3

    orig_directive = e3._L2_CODEONLY_DIRECTIVE
    if arm in ("A1", "A2"):
        os.environ["CARNOT_ARC_CODEONLY_INDUCE"] = "1"
        prop.no_think_prefix = "/no_think\n"
        prop.max_tokens = 6144 if arm == "A2" else 2560
        prop.tries = 3
        if arm == "A2":
            e3._L2_CODEONLY_DIRECTIVE = orig_directive.replace("/no_think\n", "/think\n", 1)
    else:
        os.environ["CARNOT_ARC_CODEONLY_INDUCE"] = "0"
        prop.no_think_prefix = "/think\n" if arm == "B2" else "/no_think\n"
        prop.max_tokens = 8192 if arm == "B2" else 4096
        prop.tries = 1  # a genuine-reasoning /think call is ~2-3 min; cap cost + let an overrun
        # (reasons past the token budget without ever emitting code) surface as induction_ok=False,
        # which is itself the finding, not a bug to retry away.
    return orig_directive


def _induce_no_fence(prop, game: str, window: list, cell: int) -> tuple[bool, str]:
    """GENUINE-REASONING induce path for the B arms. The real induce() force-appends a
    `\\n```python\\n` fence-opener to its combined prompt, which makes the model continue INSIDE
    a code block and SUPPRESSES a /think reasoning trace (the model never emits <think>). To
    actually test reasoning we call generate() directly with codeonly OFF and NO pre-opened
    fence, so a /think prefix reasons first and THEN emits the ```python block that
    _extract_python recovers. Single combined call, no split-induce fallback -- a /think overrun
    (reasoning past the budget, no code) is recorded as induction_ok=False, the honest finding."""
    from carnot.agentic.arc_executable_world_model import induce_prompt, _induce_transitions_k

    prompt = (
        induce_prompt(game, window, cell, k=_induce_transitions_k())
        + "\n\nReturn ONLY one ```python code block with engine + is_level_complete.\n"
    )
    ok, code = prop.generate(
        prompt, ("engine", "is_level_complete"), tries=prop.tries, codeonly_eligible=False
    )
    if not ok:
        return False, code
    return prop._write_world_model(game, code)


def run_arm(prop, game: str, arm: str, window: list, cell: int) -> JsonDict:
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic.arc_executable_world_model import (
        WorldModelVerifier,
        load_engine,
        score_goal_predicate_consistency,
    )

    # Capture the raw completion of EVERY generate() call this induce makes (induce may fall
    # back to focused split calls), so reason_engaged reflects whether reasoning engaged at ANY
    # point -- not just the last call (the bug that made probe3 report a false reason_engaged=False).
    raw_log: list[str] = []
    orig_record = prop._record_completion_diagnostics

    def _record(response, _orig=orig_record):
        _orig(response)
        raw_log.append(str(response.get("content") or ""))

    prop._record_completion_diagnostics = _record  # type: ignore[assignment]
    orig_directive = _configure_arm(prop, arm)
    t0 = time.time()
    try:
        # A arms route through the real frozen induce() (faithful to the live codeonly pipeline,
        # fence and all); B arms use the no-fence path so /think can genuinely reason.
        if arm in ("A1", "A2"):
            ok, detail = prop.induce(game, window, cell)
        else:
            ok, detail = _induce_no_fence(prop, game, window, cell)
    except Exception as exc:  # never let one arm crash the whole run
        e3._L2_CODEONLY_DIRECTIVE = orig_directive
        prop._record_completion_diagnostics = orig_record  # type: ignore[assignment]
        return {"game": game, "arm": arm, "induction_ok": False, "error": repr(exc)[:200]}
    finally:
        e3._L2_CODEONLY_DIRECTIVE = orig_directive
        prop._record_completion_diagnostics = orig_record  # type: ignore[assignment]

    reason_engaged = any(any(tag in c for tag in REASONING_TAGS) for c in raw_log)
    max_raw_len = max((len(c) for c in raw_log), default=0)
    row: JsonDict = {
        "game": game,
        "arm": arm,
        "induce_s": round(time.time() - t0, 1),
        "induction_ok": bool(ok),
        "reason_engaged": bool(reason_engaged),
        "max_raw_completion_len": max_raw_len,
        "n_generate_calls": len(raw_log),
    }
    if not ok:
        row["induction_failure_detail"] = str(detail)[:200]
        return row
    engine, is_lc = load_engine(game)
    vr = WorldModelVerifier(window).score(engine)
    row["heldout_accuracy"] = round(vr.accuracy, 4)
    row["cell_recall"] = round(vr.cell_recall, 4)
    if is_lc is not None:
        gp = score_goal_predicate_consistency(is_lc, window)
        row["goal_predicate_accuracy"] = round(gp.accuracy, 4)
        row["n_real_levelups"] = gp.n_real_levelups
        # win-recognition: does is_level_complete correctly fire on the REAL level-up next_grids?
        pos_hits, pos_total = 0, 0
        for t in window:
            if t.level_after > t.level_before:
                pos_total += 1
                try:
                    if bool(is_lc(t.next_grid)):
                        pos_hits += 1
                except Exception:
                    pass
        row["levelup_positive_recall"] = round(pos_hits / max(1, pos_total), 4)
    else:
        row["goal_predicate_accuracy"] = None
        row["levelup_positive_recall"] = None
    return row


# --------------------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------------------
def _summarize_pair(rows: list[JsonDict], no_think_arm: str, think_arm: str) -> JsonDict:
    def by_arm(a: str) -> list[JsonDict]:
        return [r for r in rows if r.get("arm") == a and r.get("induction_ok")]

    nt, th = by_arm(no_think_arm), by_arm(think_arm)

    def mean(rs: list[JsonDict], key: str) -> Optional[float]:
        vals = [r[key] for r in rs if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else None

    # per-game head-to-head on the two decision-relevant metrics
    nt_by_game = {r["game"]: r for r in nt}
    th_by_game = {r["game"]: r for r in th}
    common = sorted(set(nt_by_game) & set(th_by_game))
    win = {
        "goal_predicate_accuracy": [0, 0, 0],
        "levelup_positive_recall": [0, 0, 0],
    }  # think_win, tie, no_think_win
    for g in common:
        for metric in win:
            a = th_by_game[g].get(metric)
            b = nt_by_game[g].get(metric)
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                continue
            if a > b:
                win[metric][0] += 1
            elif a == b:
                win[metric][1] += 1
            else:
                win[metric][2] += 1
    return {
        "no_think_arm": no_think_arm,
        "think_arm": think_arm,
        "n_games_both_induced": len(common),
        "no_think_induction_success": len(nt),
        "think_induction_success": len(th),
        "no_think_reason_engaged_frac": round(
            sum(1 for r in nt if r.get("reason_engaged")) / max(1, len(nt)), 3
        ),
        "think_reason_engaged_frac": round(
            sum(1 for r in th if r.get("reason_engaged")) / max(1, len(th)), 3
        ),
        "mean_heldout_no_think": mean(nt, "heldout_accuracy"),
        "mean_heldout_think": mean(th, "heldout_accuracy"),
        "mean_goal_predicate_no_think": mean(nt, "goal_predicate_accuracy"),
        "mean_goal_predicate_think": mean(th, "goal_predicate_accuracy"),
        "mean_levelup_recall_no_think": mean(nt, "levelup_positive_recall"),
        "mean_levelup_recall_think": mean(th, "levelup_positive_recall"),
        "per_game_headtohead_think_tie_nothink": win,
    }


def build_artifact(
    *,
    roster: tuple[str, ...] = DEFAULT_ROSTER,
    b_roster: tuple[str, ...] = DEFAULT_B_ROSTER,
    root: Path = REPO_ROOT,
) -> JsonDict:
    os.environ["CARNOT_ARC_GENERATOR_CUDA_GPU"] = CUDA_GPU_INDEX
    preconds = preconditions(root)
    started_at = time.time()
    miss = _first_precondition_miss(preconds)
    if miss:
        return _finalize(
            {
                "honest_verdict": f"complete: blocked_{miss}",
                "mtp_enabled": True,
                "roster": list(roster),
                "b_roster": list(b_roster),
                "per_game_results": [],
                "comparison_a_summary": {},
                "comparison_b_summary": {},
                "comparison_genuine_vs_frozen_summary": {},
                "codeonly_toggle_inert": None,
            },
            preconds,
            started_at,
        )

    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    prop = LocalGGUFProposer(
        repo_substr=GGUF_REPO_SUBSTR,
        port=SERVER_PORT,
        mtp=True,
        kv_quant="q8_0",
        max_tokens=2560,
        no_think_prefix="/no_think\n",
    )
    if not prop._ensure_server():
        return _finalize(
            {
                "honest_verdict": "complete: blocked_mtp_cuda_server_failed_to_start",
                "mtp_enabled": True,
                "roster": list(roster),
                "b_roster": list(b_roster),
                "per_game_results": [],
                "comparison_a_summary": {},
                "comparison_b_summary": {},
                "comparison_genuine_vs_frozen_summary": {},
                "codeonly_toggle_inert": None,
            },
            preconds,
            started_at,
        )

    verbose = os.environ.get("CARNOT_EXP5714_VERBOSE") == "1"

    def _log(msg: str) -> None:
        if verbose:
            print(f"[exp5714] {msg}", flush=True)

    rows: list[JsonDict] = []
    windows: dict[str, tuple[list, int]] = {}
    for game in roster:
        try:
            built = build_levelup_window(game)
        except Exception as exc:
            rows.append({"game": game, "arm": "window", "window_error": repr(exc)[:200]})
            _log(f"{game} window ERROR {repr(exc)[:80]}")
            continue
        if built is None:
            rows.append({"game": game, "arm": "window", "no_levelup_window": True})
            _log(f"{game} no_levelup_window (skipped)")
            continue
        windows[game] = built
        window, cell = built
        _log(f"{game} window n={len(window)} cell={cell}")
        # Comparison A (frozen-stack toggle) on ALL games -- sequential per game so the two
        # arms' world_model.py writes never interleave (induce -> load -> score before next arm).
        for arm in ("A1", "A2"):
            r = run_arm(prop, game, arm, window, cell)
            rows.append(r)
            _log(
                f"{game} {arm} ok={r.get('induction_ok')} reason={r.get('reason_engaged')} "
                f"heldout={r.get('heldout_accuracy')} lu_recall={r.get('levelup_positive_recall')} "
                f"s={r.get('induce_s')}"
            )
        # Comparison B (genuine reasoning) on the B-roster subset only (B2 is ~10-20x slower).
        if game in b_roster:
            for arm in ("B1", "B2"):
                r = run_arm(prop, game, arm, window, cell)
                rows.append(r)
                _log(
                    f"{game} {arm} ok={r.get('induction_ok')} reason={r.get('reason_engaged')} "
                    f"heldout={r.get('heldout_accuracy')} lu_recall={r.get('levelup_positive_recall')} "
                    f"s={r.get('induce_s')}"
                )

    comp_a = _summarize_pair(rows, "A1", "A2")  # frozen-stack toggle (operator's literal decision)
    comp_b = _summarize_pair(
        rows, "B1", "B2"
    )  # no-fence control: is a B2 gain from reasoning or the no-fence path?
    comp_gen = _summarize_pair(
        rows, "A1", "B2"
    )  # frozen no_think baseline vs GENUINE reasoning (the GPT-5.6 transfer question)
    # codeonly_toggle_inert: is the frozen-stack toggle a no-op? True iff NO A2 (think_toggle)
    # arm ever produced a reasoning trace (the mechanistic finding, measured not assumed).
    a2_rows = [r for r in rows if r.get("arm") == "A2" and r.get("induction_ok")]
    codeonly_toggle_inert = bool(a2_rows) and not any(r.get("reason_engaged") for r in a2_rows)

    verdict = _verdict(comp_a, comp_gen, codeonly_toggle_inert)
    return _finalize(
        {
            "honest_verdict": verdict,
            "mtp_enabled": True,
            "roster": list(roster),
            "b_roster": list(b_roster),
            "per_game_results": rows,
            "comparison_a_summary": comp_a,
            "comparison_b_summary": comp_b,
            "comparison_genuine_vs_frozen_summary": comp_gen,
            "codeonly_toggle_inert": codeonly_toggle_inert,
        },
        preconds,
        started_at,
    )


def _verdict(comp_a: JsonDict, comp_gen: JsonDict, inert: bool) -> str:
    """Compound verdict. The PRIMARY axis is the operator's literal decision (Comparison A:
    the frozen codeonly toggle). The SECONDARY axis is whether GENUINE reasoning (B2) beats the
    frozen no_think baseline (A1) on win-recognition -- the GPT-5.6 transfer question. Both are
    encoded so the headline reflects the full, honest picture."""
    gen = comp_gen.get("per_game_headtohead_think_tie_nothink", {}).get(
        "levelup_positive_recall", [0, 0, 0]
    )
    gen_think_win, _gen_tie, gen_frozen_win = gen[0], gen[1], gen[2]
    if gen_think_win > gen_frozen_win:
        genuine = f"genuine_reasoning_improves_winrecognition_{gen_frozen_win}_to_{gen_think_win}"
    elif gen_think_win < gen_frozen_win:
        genuine = f"genuine_reasoning_worse_winrecognition_{gen_think_win}_to_{gen_frozen_win}"
    else:
        genuine = "genuine_reasoning_no_winrecognition_delta"
    if inert:
        # The operator's LITERAL toggle flip is a no-op (no reasoning trace under codeonly), but
        # the genuine-reasoning path (a larger pipeline change) may still help -- report both.
        return f"complete: think_toggle_inert_under_codeonly_but_{genuine}"
    a_win = comp_a.get("per_game_headtohead_think_tie_nothink", {}).get(
        "levelup_positive_recall", [0, 0, 0]
    )
    if a_win[0] > a_win[2]:
        return (
            f"complete: think_toggle_higher_winrecognition_{a_win[2]}_to_{a_win[0]}_and_{genuine}"
        )
    if a_win[0] < a_win[2]:
        return f"complete: no_think_higher_winrecognition_{a_win[0]}_to_{a_win[2]}_and_{genuine}"
    return f"complete: think_toggle_null_but_{genuine}"


def _finalize(core: JsonDict, preconds: JsonDict, started_at: float) -> JsonDict:
    artifact: JsonDict = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "model_specs": MODEL_SPECS,
        "field_principles": FIELD_PRINCIPLES,
        "solve_provenance": "development_proxy",
        "random_seed": RANDOM_SEED,
        "prior_failures": [
            {
                "experiment_id": "exp5594-think-mode-induction-quality-ab",
                "req": "REQ-ARC-WMTE-5594",
                "verdict": "complete: think_mode_ab_equal_success_no_think_higher_accuracy",
                "root_cause": "N=2 below sample-size floor; heldout-only metric; NO level-up in any "
                "window so goal-predicate never fired; arm-dependent windows (run_game per arm)",
                "addressed_by": "larger adaptered roster; SHARED level-up window per game (goal-predicate "
                "now fires + no arm-window confound); win-recognition metric; MTP-on L4-matched hardware; "
                "post-REQ-ARC-FCP-5699-38/39 live-agent code",
                "retire_if_same_verdict": False,
            }
        ],
    }
    artifact.update(core)
    artifact["duration_s"] = round(time.time() - started_at, 3)
    artifact["preconditions_checked"] = preconds
    artifact["reproducibility_checksum"] = ""
    artifact["reproducibility_checksum"] = _checksum(
        {k: v for k, v in artifact.items() if k != "reproducibility_checksum"}
    )
    return artifact


def main() -> None:  # pragma: no cover - thin CLI wrapper, exercised manually
    artifact = build_artifact()
    out_path = REPO_ROOT / RESULT_RELATIVE_PATH
    out_path.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out_path} -- honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
