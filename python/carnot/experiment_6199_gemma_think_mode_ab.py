"""Experiment 6199: gemma-4-31B `/think` vs no-think induction-quality A/B (REQ-ARC-WMTE-6198).

WHY THIS IS NOT A RERUN OF THE PRIOR THINK-MODE NULLS (Failed-Experiment Rerun Discipline --
named priors + root cause + what is different + falsifiable gate):

  PRIOR 1: REQ-ARC-WMTE-5594 (results/experiment_5594_think_mode_induction_quality_ab.json).
    N=2 games, no level-up window, /no_think beat /think on heldout_accuracy alone.
  PRIOR 2: REQ-ARC-WMTE-5714 (results/experiment_5714_think_mode_rescoped_ab.json, the direct
    template this script is adapted from). Found the frozen-stack `/no_think`->`/think` STRING
    TOGGLE is INERT under codeonly (the directive's "skip all reasoning" body overrides it) --
    but that finding, and the whole experiment, ran against Qwen3.5-9B-MTP, which HAD an
    in-band `/think`-`/no_think` control token to begin with.

  WHAT IS DIFFERENT NOW: the live generator is gemma-4-31B-it-qat (moved 2026-07-28), which has
  NO such token -- `/no_think`/`/think` are literal inert text on gemma (this project's own
  arc_executable_world_model.py constants-block comment says so explicitly, and exp5764/5766
  independently confirmed it). So neither prior result is evidence FOR OR AGAINST gemma
  reasoning helping -- they tested a mechanism gemma does not have. REQ-ARC-WMTE-6198 (this
  session, commit 2a08fe0b13) built the mechanism gemma DOES have: `induce_think_on()` routes
  through gemma's native thought channel via the /v1/chat/completions endpoint (the ONLY
  endpoint that splits `reasoning_content`, per exp5764: n_reason_engaged=39/39 there vs the
  raw-completion codeonly path's 0 by design) -- a real toggle, not a dead string swap. This is
  the FIRST measurement of that mechanism. Falsifiable gate: if this run also comes back null
  (no reasoning engagement, or reasoning engages but induction quality does not improve), the
  mechanism -- not just the string -- is the thing to retire, and this experiment_id is that
  retirement record.

MECHANISM UNDER TEST (built this session, default OFF -- see induce_think_on()'s own docstring
in arc_executable_world_model.py for the full account): `CARNOT_ARC_INDUCE_THINK=1` makes
`LocalGGUFProposer.generate()` skip the codeonly directive + pre-opened fence and route through
the chat-completions endpoint regardless of `use_chat_template`; `induce()`'s combined call
correspondingly omits its own caller-side pre-opened fence. Both arms otherwise use the SAME
proposer, SAME shared per-game level-up window (no arm-dependent-window confound), SAME
max_tokens (16384, the value the exp6091 truncation-fix rerun proved non-truncating for gemma at
this budget class -- a shared, generous budget is required because a `/think` call spends tokens
on reasoning before code, and an asymmetric budget would confound "does thinking help" with "did
the control arm merely have less room").

Simpler than the Qwen-era template: gemma's think toggle does not need the old script's
four-arm (A1/A2/B1/B2) codeonly-on/off split, because `induce_think_on()` already internally
overrides codeonly when it fires -- there is no "does the toggle even reach a reasoning-capable
code path" question here the way there was for the inert Qwen string swap. Two arms are the
whole comparison: `no_think` (flag off, the shipped default) vs `think` (flag on).

GUARDRAIL: OFFLINE DEV MEASUREMENT ONLY. Never edits the frozen submission stack
(scripts/kaggle/submission_kernel/main.py, arc_competition_agent.py live defaults, or
ARC_LIVE_GENERATOR_THINK_SCORED_DEFAULT). Reports a delta + a recommendation; flipping either
live/scored think default is a separate operator decision, per this project's standing
convention that a live-path behavior change ships only after a matched-budget offline A/B.
solve_provenance=development_proxy (adaptered offline dev-twin windows, not a live hidden-game
self-discovery solve).

Spec refs: REQ-ARC-WMTE-6198 (the mechanism), this experiment closes its stated forward-work
item ("the actual gemma think-mode A/B... needs GPU time unavailable this pass").
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

EXPERIMENT_ID = "experiment_6199_gemma_think_mode_ab"
RESULT_RELATIVE_PATH = "results/experiment_6199_gemma_think_mode_ab.json"
SCHEMA = "carnot.exp6199.gemma_think_mode_ab.v1"
INFERENCE_SUBSTRATE = "live_llm_inference"
RANDOM_SEED = 6199

# Same adaptered-to-L1 roster the Qwen-era template used (dc22, m0r0, ... proven reliable,
# arm-independent window sources); reused rather than re-derived.
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
WINDOW_K = 12
GGUF_REPO_SUBSTR = "gemma-4-31B-it-qat"  # ARC_LIVE_GENERATOR_REPO_SUBSTR, the live pin
CUDA_GPU_INDEX = "1"  # outer-loop's RTX 3090 -- GPU 0 is the conductor's
SERVER_PORT = 8940  # non-default, avoids colliding with any conductor-owned warm server
# Shared budget for BOTH arms (methodological requirement -- see module docstring). 16384 is the
# value results/experiment_6091_refine_engine_visible_ab.json's completed rerun (this session)
# measured as non-truncating for gemma at this budget class.
SHARED_MAX_TOKENS = 16384
# HTTP timeout per induce attempt, in seconds. `LocalGGUFProposer.timeout` is a plain dataclass
# field (default 300s) -- it does NOT read CARNOT_ARC_INDUCE_TIMEOUT, that is a separate,
# unrelated helper used by other call sites. An env var alone does nothing here; this value
# must be passed explicitly to the constructor. 300s x 2 retry attempts is why every think-arm
# row in the first two runs of this experiment timed out at ~600s even after CARNOT_ARC_INDUCE_
# TIMEOUT=1500 was set on the process -- the override never reached this field. 1500 gives a
# full 16384-token think-mode completion (~800s at ~20 tok/s single-card decode) real headroom.
INDUCE_TIMEOUT_S = 1500
REASONING_TAGS = ("<think", "</think", "<thinking", "<reasoning")

MODEL_SPECS = [
    {
        "name": "gemma-4-31B-it-qat",
        "hf_id": "unsloth/gemma-4-31B-it-qat-GGUF",
        "role": "E3AgentPolicy tier-3 world-model induction proposer (the live pin since 2026-07-28)",
        "server": "fresh CUDA llama-server, -ngl 999, q8_0 KV, MTP OFF (local-dev default; MTP "
        "interaction with think mode is a separate, untested axis this run deliberately does not "
        "conflate -- see REQ-ARC-WMTE-6198's own risk note on this)",
        "mtp": False,
    }
]

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "model_specs",
    "mtp_enabled",
    "roster",
    "per_game_results",
    "comparison_summary",
    "reason_engaged_at_all",
    "solve_provenance",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal-prefixed; think helping, hurting, tying, or never engaging "
        "reasoning at all are all distinct, real, citable outcomes"
    },
    "inference_substrate": {
        "principle": "live_llm_inference -- both arms invoke the real gemma-4-31B-it-qat server "
        "on a 3090, not mocked; 60s duration floor applies to the whole run"
    },
    "reason_engaged_at_all": {
        "principle": "the load-bearing mechanistic finding: did CARNOT_ARC_INDUCE_THINK=1 ever "
        "actually produce a reasoning trace on gemma? If false, the mechanism (not just a prior "
        "string toggle) is inert and should be retired, not the induction-quality comparison itself"
    },
    "solve_provenance": {
        "principle": "development_proxy -- adaptered offline dev-twin windows, not a live-agent "
        "hidden-game self-discovery solve; declared so adversarial_verify can tell them apart"
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
            induce_think_on,
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
        # gemma-4-31B-it-qat Q4_K_XL is ~21.4GB resident at n_ctx 32768 (measured this session,
        # exp6091's rerun log); require real headroom on a 24GB card if a fresh launch is needed.
        has_headroom = 0 <= idx < len(lines) and int(lines[idx]) >= 22000
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
    levelups = [i for i, t in enumerate(trans) if t.level_after > t.level_before]
    if not levelups:
        return None
    j = levelups[-1]
    return trans[max(0, j - (k - 1)) : j + 1]


def build_levelup_window(game: str, k: int = WINDOW_K) -> Optional[tuple[list, int]]:
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
def _configure_arm(prop, arm: str) -> None:
    """Set env + proposer config for an arm. Two arms, not four -- see module docstring for why
    gemma's real think toggle does not need the Qwen-era template's codeonly-on/off split."""
    prop.no_think_prefix = ""  # gemma live pin; /no_think is inert text on gemma either way
    prop.max_tokens = SHARED_MAX_TOKENS
    if arm == "think":
        os.environ["CARNOT_ARC_INDUCE_THINK"] = "1"
        prop.tries = 1  # a think overrun is the honest finding, not a retry bug (exp5714 precedent)
    else:
        os.environ.pop("CARNOT_ARC_INDUCE_THINK", None)
        prop.tries = 3


def run_arm(prop, game: str, arm: str, window: list, cell: int) -> JsonDict:
    from carnot.agentic.arc_executable_world_model import (
        WorldModelVerifier,
        load_engine,
        score_goal_predicate_consistency,
    )

    raw_log: list[str] = []
    orig_record = prop._record_completion_diagnostics

    def _record(response, _orig=orig_record):
        _orig(response)
        raw_log.append(str(response.get("content") or ""))

    prop._record_completion_diagnostics = _record  # type: ignore[assignment]
    _configure_arm(prop, arm)
    t0 = time.time()
    try:
        ok, detail = prop.induce(game, window, cell)
    except Exception as exc:  # never let one arm crash the whole run
        return {"game": game, "arm": arm, "induction_ok": False, "error": repr(exc)[:200]}
    finally:
        prop._record_completion_diagnostics = orig_record  # type: ignore[assignment]
        os.environ.pop("CARNOT_ARC_INDUCE_THINK", None)  # never leak into the next arm

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
def _summarize_pair(rows: list[JsonDict]) -> JsonDict:
    def by_arm(a: str) -> list[JsonDict]:
        return [r for r in rows if r.get("arm") == a and r.get("induction_ok")]

    nt, th = by_arm("no_think"), by_arm("think")

    def mean(rs: list[JsonDict], key: str) -> Optional[float]:
        vals = [r[key] for r in rs if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else None

    nt_by_game = {r["game"]: r for r in nt}
    th_by_game = {r["game"]: r for r in th}
    common = sorted(set(nt_by_game) & set(th_by_game))
    win = {"goal_predicate_accuracy": [0, 0, 0], "levelup_positive_recall": [0, 0, 0]}
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


def build_artifact(*, roster: tuple[str, ...] = DEFAULT_ROSTER, root: Path = REPO_ROOT) -> JsonDict:
    os.environ["CARNOT_ARC_GENERATOR_CUDA_GPU"] = CUDA_GPU_INDEX
    # Operator directive (2026-08-07): this A/B measures CUDA-substrate decode/reasoning behavior
    # specifically -- a silent fallback to the ~2 tok/s iGPU HIP build does not degrade the
    # measurement, it CORRUPTS it (induce calls time out and look like ordinary induction
    # failures). REQUIRE_CUDA makes _generator_server_and_env() raise instead of silently
    # substituting HIP; see GeneratorCudaRequiredError's docstring for the exp6199 incident this
    # closes. Opt-in and scoped to this process's env, so it does not touch the conductor's own
    # generator resolution.
    os.environ["CARNOT_ARC_GENERATOR_REQUIRE_CUDA"] = "1"
    preconds = preconditions(root)
    started_at = time.time()
    miss = _first_precondition_miss(preconds)
    if miss:
        return _finalize(
            {
                "honest_verdict": f"complete: blocked_{miss}",
                "mtp_enabled": False,
                "roster": list(roster),
                "per_game_results": [],
                "comparison_summary": {},
                "reason_engaged_at_all": None,
            },
            preconds,
            started_at,
        )

    from carnot.agentic.arc_executable_world_model import (
        GeneratorCudaRequiredError,
        LocalGGUFProposer,
    )

    prop = LocalGGUFProposer(
        repo_substr=GGUF_REPO_SUBSTR,
        port=SERVER_PORT,
        mtp=False,
        kv_quant="q8_0",
        max_tokens=SHARED_MAX_TOKENS,
        no_think_prefix="",
        timeout=INDUCE_TIMEOUT_S,
    )
    try:
        server_up = prop._ensure_server()
    except GeneratorCudaRequiredError as exc:
        # The guard refused the requested CUDA card and (per REQUIRE_CUDA above) declined to
        # silently substitute the HIP build. Distinct verdict from the generic
        # blocked_cuda_server_failed_to_start below -- this is "CUDA was busy/unavailable", not
        # "the server binary itself would not launch".
        return _finalize(
            {
                "honest_verdict": "complete: blocked_cuda_unavailable",
                "cuda_required_error": str(exc)[:400],
                "mtp_enabled": False,
                "roster": list(roster),
                "per_game_results": [],
                "comparison_summary": {},
                "reason_engaged_at_all": None,
            },
            preconds,
            started_at,
        )
    if not server_up:
        return _finalize(
            {
                "honest_verdict": "complete: blocked_cuda_server_failed_to_start",
                "mtp_enabled": False,
                "roster": list(roster),
                "per_game_results": [],
                "comparison_summary": {},
                "reason_engaged_at_all": None,
            },
            preconds,
            started_at,
        )

    verbose = os.environ.get("CARNOT_EXP6199_VERBOSE") == "1"

    def _log(msg: str) -> None:
        if verbose:
            print(f"[exp6199] {msg}", flush=True)

    rows: list[JsonDict] = []
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
        window, cell = built
        _log(f"{game} window n={len(window)} cell={cell}")
        for arm in ("no_think", "think"):
            r = run_arm(prop, game, arm, window, cell)
            rows.append(r)
            _log(
                f"{game} {arm} ok={r.get('induction_ok')} reason={r.get('reason_engaged')} "
                f"heldout={r.get('heldout_accuracy')} lu_recall={r.get('levelup_positive_recall')} "
                f"s={r.get('induce_s')}"
            )

    comp = _summarize_pair(rows)
    think_rows = [r for r in rows if r.get("arm") == "think" and r.get("induction_ok")]
    reason_engaged_at_all = any(r.get("reason_engaged") for r in think_rows)

    verdict = _verdict(comp, reason_engaged_at_all)
    return _finalize(
        {
            "honest_verdict": verdict,
            "mtp_enabled": False,
            "roster": list(roster),
            "per_game_results": rows,
            "comparison_summary": comp,
            "reason_engaged_at_all": reason_engaged_at_all,
        },
        preconds,
        started_at,
    )


def _verdict(comp: JsonDict, reason_engaged_at_all: bool) -> str:
    if not reason_engaged_at_all:
        return "complete: think_mode_mechanism_inert_no_reasoning_trace_ever_engaged"
    win = comp.get("per_game_headtohead_think_tie_nothink", {}).get(
        "levelup_positive_recall", [0, 0, 0]
    )
    think_win, tie, nothink_win = win[0], win[1], win[2]
    if think_win > nothink_win:
        return f"complete: think_higher_winrecognition_{nothink_win}_to_{think_win}"
    if think_win < nothink_win:
        return f"complete: no_think_higher_winrecognition_{think_win}_to_{nothink_win}"
    return f"complete: think_toggle_null_winrecognition_tie_{tie}"


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
                "root_cause": "N=2, Qwen-only, no level-up window in either run",
                "addressed_by": "not addressed directly -- superseded by the generator switch; see "
                "exp5714 below for the mechanism that actually explains the Qwen-era result",
                "retire_if_same_verdict": False,
            },
            {
                "experiment_id": "exp5714-think-mode-rescoped-ab",
                "req": "REQ-ARC-WMTE-5714",
                "verdict": "complete: think_toggle_inert_under_codeonly_but_genuine_reasoning_...",
                "root_cause": "tested Qwen3.5-9B-MTP's /think-/no_think STRING TOGGLE, which the "
                "live stack's codeonly directive structurally overrides -- an inert mechanism on "
                "the model this project used at the time",
                "addressed_by": "gemma-4-31B-it-qat (the live pin since 2026-07-28) has no such "
                "string token at all; REQ-ARC-WMTE-6198 built a genuinely different mechanism "
                "(chat-endpoint routing to gemma's native thought channel) this experiment tests "
                "for the first time -- not a rerun of the same toggle on the same model",
                "retire_if_same_verdict": True,
            },
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
