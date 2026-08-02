"""What does the SHIPPED engine-defect re-ask gate actually COST?

Three arms on the live induce path.

THE DEFECT BEING MEASURED. `generate()`'s own comment claimed a defect re-ask "NEVER FAILS WHERE
THE OLD PATH SUCCEEDED". That claim is false, and the 2026-08-01 goal-variant A/B falsified it:
induction HARD-FAILED on 17 of 21 treatment cells against 1 of 22 control and 0 of 21 A/A. The
mechanism is that `attempt < tries - 1` stops only the LAST attempt from `continue`-ing out of
the loop; it does not stop an EARLIER re-ask from spending the attempt that would have BEEN the
accept.

THE SAME STRUCTURE IS IN THE SHIPPED ENGINE GATE AND IT IS LIVE. `_induce_defect_reasks()`
returns 1 by DEFAULT and `_defect_check_on` arms on every engine induce call. The gate's headline
(13/36 -> 22/36) counted USABLE ENGINES and never counted hard failures, so the shipped agent may
be running a net-negative gate right now and nobody has measured it.

THE PRIMARY IS THEREFORE `usable MINUS hard failures`, NOT `usable`. Scoring on usable alone is
precisely the metric that hid this for a week. Both components are also reported separately so a
reader can see which one moved.

ARMS (one server process, one corpus, one seed schedule):
  A `a_off`      CARNOT_ARC_INDUCE_DEFECT_REASKS=0            -- gate OFF
  B `b_shipped`  nothing set                                  -- the SHIPPED state, untouched
  C `c_owns`     CARNOT_ARC_INDUCE_DEFECT_OWNS_ATTEMPTS=1     -- the 2026-08-01 fix, default-off
  AA `aa`        same env as A at a DIFFERENT seed base       -- the noise floor

A, B and C share a seed base, so within a (game, replicate) all three send the IDENTICAL prompt
at the IDENTICAL seed on attempt 0. Any divergence is the gate firing rather than the sampler.

DEFAULTS ARE NOT TOUCHED. This measures whether one SHOULD change; flipping is a separate,
operator-visible decision. NOT SUBMITTED: no scored or online ARC game is played.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Derived, never hardcoded (CLAUDE.md Test-Run Record Integrity rule 4): an absolute path baked
# into source makes a fresh clone write into the operator's checkout. <repo>/results/<exp>/.
ROOT = HERE.parents[1]

# ---- isolation: E3_DIR is read at IMPORT time, so it must be set BEFORE the import -----------
# `_guard_engine_write` is scoped to PYTEST ONLY, so a measurement driver is exactly the caller
# nothing protects. One rewrote results/arc_e3/<game>/world_model.py -- tracked, read-only
# EVIDENCE -- within 90 seconds. A per-cell store is ALSO what stops arm A's engine being read
# by arm B.
_SCRATCH_ROOT = Path(
    "/tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/"
    "87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad/reaskcost"
)
# Per-RUN store, not just per-cell. `run_cell` rmtree's its cell dir, and the cell dirs are keyed
# `{game}__r{rep}__{tag}` -- which COLLIDE across runs. Without this, the supplementary run would
# delete the main run's engines, i.e. the evidence its own comparison is against.
SCRATCH = _SCRATCH_ROOT / os.environ.get("RNC_STORE_SUBDIR", "e3_store")
SCRATCH.mkdir(parents=True, exist_ok=True)
os.environ["CARNOT_ARC_E3_DIR"] = str(SCRATCH)

if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))

PY = os.environ.get("RNC_PY", "/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/.venv/bin/python")
# NOT the default port: the default often holds a stale server, and one is running right now
# (a 20 GB gemma-4-31B on the other card that this session does not own and must not touch).
PORT = int(os.environ.get("RNC_PORT", "41931"))
GPU = os.environ.get("RNC_GPU", "0")
N_REPLICATES = int(os.environ.get("RNC_REPLICATES", "2"))
SEED_BASE = int(os.environ.get("RNC_SEED_BASE", "8100"))
AA_SEED_BASE = int(os.environ.get("RNC_AA_SEED_BASE", "8200"))
WALL_BUDGET_S = float(os.environ.get("RNC_WALL_BUDGET_S", "30000"))

# The windows the 2026-08-01 sibling built, reused rather than rebuilt so that this experiment's
# corpus IS that experiment's corpus. Rebuilding steps a real environment and has no internal
# bound (tr87 took a whole pass down twice at 100% CPU in a sibling run).
WINDOW_DIR = SCRATCH.parent.parent / "goalab" / "windows"

# SUPPLEMENTARY COUNTERFACTUAL CONDITION (RNC_REPEAT_PENALTY). The main run found the gate
# fires ZERO times on the shipped stack, so its cost WHEN IT FIRES could not be observed. The
# suspected reason is `_INDUCE_REPEAT_PENALTY = 1.1`, which shipped on 2026-07-31 alongside the
# gate and is ON in every arm above; its own source note says the penalty carries 11 of the 13
# paired wins and the re-ask only 2. Setting this to "1.0" restores the pre-2026-07-31 payload
# BYTE-FOR-BYTE (llama.cpp treats 1.0 as identity), which is the regime the gate's 13/36 -> 22/36
# headline was measured in. That makes the gate's cost observable again.
# THIS IS NOT THE SHIPPED STACK and every artifact field derived from it says so.
REPEAT_PENALTY = os.environ.get("RNC_REPEAT_PENALTY")
OUT = HERE / os.environ.get("RNC_OUT_SUBDIR", "out")
OUT.mkdir(exist_ok=True)
CELLS = OUT / "cells"
CELLS.mkdir(exist_ok=True)

FLAG_REASKS = "CARNOT_ARC_INDUCE_DEFECT_REASKS"
FLAG_OWNS = "CARNOT_ARC_INDUCE_DEFECT_OWNS_ATTEMPTS"
# Must stay unset in EVERY arm. `_goal_defects` returns [] immediately when this is unset, so the
# goal gate is inert and cannot contaminate a measurement of the ENGINE gate.
FLAG_GOAL = "CARNOT_ARC_INDUCE_GOAL_DEFECT_CHECK"
FLAG_GOAL_PROMPT = "CARNOT_ARC_GOAL_PROMPT_TRANSITIONS"

ROSTER = [
    "ar25", "bp35", "cd82", "cn04", "g50t", "ka59", "lf52", "lp85", "ls20", "m0r0",
    "re86", "s5i5", "sb26", "sc25", "sk48", "su15", "tn36", "tr87", "tu93", "wa30",
]  # fmt: skip

# (arm, tag, seed_base). `aa` is the SAME arm as `a_off` at a different seed: the noise floor,
# run through the identical analysis. The generator sends no seed to the sampler by default and
# `sampling_seed`'s docstring records a MEASURED 40% run-to-run divergence without one, "at least
# as large as any treatment effect yet measured on this path" -- hence the floor is mandatory.
ARMS = [
    ("a_off", "a_off", SEED_BASE),
    ("b_shipped", "b_shipped", SEED_BASE),
    ("c_owns", "c_owns", SEED_BASE),
    ("a_off", "aa", AA_SEED_BASE),
]
# RNC_ARMS restricts which tags run, so the supplementary condition can drop the A/A arm (its
# noise floor is already measured at n=40 on the same corpus and the same seeds).
if os.environ.get("RNC_ARMS"):
    _want = {t.strip() for t in os.environ["RNC_ARMS"].split(",") if t.strip()}
    ARMS = [a for a in ARMS if a[1] in _want]


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def server_props(port: int) -> dict:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=10) as r:
            return json.loads(r.read().decode())
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def listening_pid(port: int) -> int | None:
    out = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True, check=False).stdout
    for line in out.splitlines():
        if f"127.0.0.1:{port}" in line and "pid=" in line:
            try:
                return int(line.split("pid=")[1].split(",")[0])
            except (IndexError, ValueError):
                return None
    return None


def apply_arm_env(arm: str) -> None:
    """Set EXACTLY the env that defines an arm, clearing the other arm's knob every time.

    Written as one function rather than inline so the three arms cannot drift: a knob left set
    from the previous cell is how an A/B silently measures two treatments under one label.
    """
    # The goal gate is inert in EVERY arm of this experiment -- it is a different intervention
    # and would confound the engine contrast if it fired.
    os.environ.pop(FLAG_GOAL, None)
    os.environ.pop(FLAG_GOAL_PROMPT, None)
    # Set IDENTICALLY in every arm of a given run, so it can never confound an arm contrast --
    # it defines the REGIME the whole run is measured in, not a treatment within it.
    if REPEAT_PENALTY is None:
        os.environ.pop("CARNOT_ARC_INDUCE_REPEAT_PENALTY", None)
    else:
        os.environ["CARNOT_ARC_INDUCE_REPEAT_PENALTY"] = REPEAT_PENALTY
    if arm == "a_off":
        os.environ[FLAG_REASKS] = "0"
        os.environ.pop(FLAG_OWNS, None)
    elif arm == "b_shipped":
        # NOTHING set. The control-for-the-fix arm must be the SHIPPED path itself, not a
        # re-specification of it that happens to agree today.
        os.environ.pop(FLAG_REASKS, None)
        os.environ.pop(FLAG_OWNS, None)
    elif arm == "c_owns":
        os.environ.pop(FLAG_REASKS, None)
        os.environ[FLAG_OWNS] = "1"
    else:  # pragma: no cover - guarded by the ARMS table
        raise SystemExit(f"unknown arm {arm}")


def main() -> int:  # noqa: C901, PLR0915
    t0 = time.time()
    from carnot.agentic import arc_executable_world_model as e3

    assert Path(e3.E3_DIR) == SCRATCH, f"E3_DIR isolation failed: {e3.E3_DIR}"

    # ---------------- PRECONDITIONS -------------------------------------------------
    gguf = e3._resolve_gguf(e3.ARC_LIVE_GENERATOR_REPO_SUBSTR)  # noqa: SLF001
    conductor = subprocess.run(
        ["systemctl", "--user", "is-active", "carnot-conductor.service"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    free_mb = e3._cuda_gpu_free_mb(int(GPU))  # noqa: SLF001
    dry = os.environ.get("RNC_DRY") == "1"
    n_windows = len(list(WINDOW_DIR.glob("*.pkl"))) if WINDOW_DIR.exists() else 0
    pre = [
        {
            "resource": "gemma-4-31B-it-qat gguf cached",
            "available": bool(gguf),
            "detail": str(gguf),
            "principle": "the live generator must be on disk; without it the honest verdict is "
            "blocked_model_not_cached, never a fabricated run",
        },
        {
            "resource": "conductor_inactive",
            "available": conductor != "active",
            "detail": f"systemctl is-active -> {conductor!r}",
            "principle": "a live conductor would contend for the same GPU and interleave its "
            "own induction into this store",
        },
        {
            # SCOPED TO REAL RUNS. A dry run issues no LLM call and binds no card, so demanding
            # VRAM from it would block the witness pass -- the part that is SUPPOSED to run
            # before a GPU is free -- on a resource it does not use. Not relaxed for the measured
            # run: RNC_DRY also stops execution before the first LLM call, so there is no path
            # on which this passes vacuously and a model then loads.
            "resource": f"cuda_gpu_{GPU}_has_headroom",
            "available": free_mb >= 20000 or dry,
            "detail": f"free={free_mb} MiB"
            + (" (not required: dry run makes no LLM call)" if dry else ""),
            "principle": "a concurrent workflow holds the other card; launching without "
            "headroom would evict work this session does not own",
        },
        {
            "resource": "shipped_defect_reasks_is_1",
            "available": e3._induce_defect_reasks() == 1,  # noqa: SLF001
            "detail": f"_induce_defect_reasks() -> {e3._induce_defect_reasks()}",  # noqa: SLF001
            "principle": "arm B is defined as THE SHIPPED STATE; if the shipped default were "
            "already 0 this would measure OFF vs OFF and report a null as a finding",
        },
        {
            "resource": "owns_attempts_default_off",
            "available": not e3._defect_gate_owns_attempts(),  # noqa: SLF001
            "principle": "the fix must be default-off or arm B is silently arm C",
        },
        {
            "resource": "goal_defect_flag_default_off",
            "available": not e3._goal_defect_check_on(),  # noqa: SLF001
            "principle": "a live goal gate would fire inside every arm and confound a "
            "measurement of the ENGINE gate with a second, different intervention",
        },
        {
            "resource": "port_free",
            "available": listening_pid(PORT) is None,
            "detail": f"port {PORT}",
            "principle": "reusing a stale server on the default port is how an arm silently "
            "gets a different model or context size",
        },
        {
            "resource": "cached_windows_present",
            "available": n_windows >= len(ROSTER),
            "detail": f"{n_windows} pkl in {WINDOW_DIR}",
            "principle": "the corpus must be the sibling run's corpus, not a re-derivation that "
            "could silently differ",
        },
    ]
    if not all(p["available"] for p in pre):
        missing = [p["resource"] for p in pre if not p["available"]]
        (OUT / "blocked.json").write_text(
            json.dumps(
                {
                    "honest_verdict": "blocked_precondition_" + "_".join(missing)[:120],
                    "preconditions_checked": pre,
                },
                indent=2,
            )
        )
        print("BLOCKED:", missing)
        return 1
    print("preconditions OK")

    # ---------------- WINDOWS (no LLM yet) ------------------------------------------
    windows: dict[str, tuple] = {}
    split_meta: dict[str, dict] = {}
    for game in ROSTER:
        with open(WINDOW_DIR / f"{game}.pkl", "rb") as fh:
            w = pickle.load(fh)
        shown, held, cell = w["shown"], w["held"], int(w["cell"])
        windows[game] = (shown, held, cell)
        split_meta[game] = {
            "n_shown": len(shown),
            "n_heldout": len(held),
            "cell": cell,
            "window_pkl": str(WINDOW_DIR / f"{game}.pkl"),
            "window_pkl_sha256": hashlib.sha256(
                (WINDOW_DIR / f"{game}.pkl").read_bytes()
            ).hexdigest(),
        }
        # The gate dry-runs against `shown`; a game with none would be structurally unmeasurable
        # in every arm.
        assert len(shown) >= 1, f"{game} has an empty shown prefix"
    print(f"windows loaded for {len(windows)} games")

    # ---------------- TREATMENT WITNESS (still no LLM) -------------------------------
    # Half of this treatment is a POST-GENERATION accept check, which leaves the induce prompt
    # BYTE-IDENTICAL between arms. A prompt diff would therefore show "no difference" and prove
    # nothing. Four checks instead, so this cannot ship as a silent no-op measured as a null:
    #   1. the induce prompt IS byte-identical across arms (stated, not hidden)
    #   2. the re-ask BUDGET resolver differs across arms (the knob is armed)
    #   3. the DETECTOR bites on a known-defective engine (the check half is real)
    #   4. at analysis time, engine_defect_reasks_delta > 0 in B/C and == 0 in A/AA (it FIRED)
    defective_engine = "import numpy as np\ndef engine(grid, action, data):\n    x = 1\n"
    probe = e3.LocalGGUFProposer(port=PORT)
    treatment = []
    for game, (shown, _held, cell) in windows.items():
        prompts, budgets, owns = {}, {}, {}
        for arm in ("a_off", "b_shipped", "c_owns"):
            apply_arm_env(arm)
            prompts[arm] = sha(e3.induce_prompt(game, shown, cell))
            budgets[arm] = e3._induce_defect_reasks()  # noqa: SLF001
            owns[arm] = e3._defect_gate_owns_attempts()  # noqa: SLF001
        apply_arm_env("b_shipped")
        detector = probe._engine_defects(defective_engine, shown)  # noqa: SLF001
        treatment.append(
            {
                "game": game,
                "prompt_identical_across_arms": len(set(prompts.values())) == 1,
                "reask_budget_by_arm": budgets,
                "owns_attempts_by_arm": owns,
                "detector_bites_on_known_defect": bool(detector),
                "detector_kinds": detector,
                "goal_gate_inert": not e3._goal_defect_check_on(),  # noqa: SLF001
                # The REGIME witness. Read back through the resolver the live payload uses, so
                # a typo'd override (which falls back to the default rather than raising) cannot
                # be mistaken for the counterfactual having been applied.
                "repeat_penalty_effective": e3._induce_repeat_penalty(),  # noqa: SLF001
                "repeat_penalty_requested": REPEAT_PENALTY,
            }
        )
    armed = (
        all(t["prompt_identical_across_arms"] for t in treatment)
        and all(t["reask_budget_by_arm"]["a_off"] == 0 for t in treatment)
        and all(t["reask_budget_by_arm"]["b_shipped"] == 1 for t in treatment)
        and all(t["reask_budget_by_arm"]["c_owns"] == 1 for t in treatment)
        and all(t["owns_attempts_by_arm"]["c_owns"] for t in treatment)
        and not any(t["owns_attempts_by_arm"]["b_shipped"] for t in treatment)
        and all(t["detector_bites_on_known_defect"] for t in treatment)
        and all(t["goal_gate_inert"] for t in treatment)
        # The regime must be what was ASKED for. A silently-ignored override would make the
        # supplementary run a duplicate of the main one under a different name.
        and all(
            t["repeat_penalty_effective"]
            == (1.1 if REPEAT_PENALTY is None else float(REPEAT_PENALTY))
            for t in treatment
        )
    )
    print(f"treatment witness armed={armed}")
    if not armed:
        (OUT / "blocked.json").write_text(
            json.dumps(
                {"honest_verdict": "blocked_treatment_not_armed", "treatment_witness": treatment},
                indent=2,
            )
        )
        return 1

    # ---------------- PRE-REGISTRATION (written BEFORE any result exists) ------------
    n_games = len(ROSTER)
    prereg = {
        "question": "Is the SHIPPED engine-defect re-ask gate net-positive once the hard "
        "failures it causes are counted against the usable engines it buys?",
        "arms": {
            "a_off": f"{FLAG_REASKS}=0 -- gate off",
            "b_shipped": "nothing set -- the shipped state, unmodified",
            "c_owns": f"{FLAG_OWNS}=1 -- a defect re-ask GRANTS an attempt, not consumes one",
            "aa": "same env as a_off at a different seed base -- the noise floor",
        },
        "PRIMARY": {
            "metric": "net = usable - hard_failures",
            "why_not_usable_alone": "scoring on usable alone is the metric that hid this defect "
            "for a week: the gate's headline 13/36 -> 22/36 counted usable engines and never "
            "counted the cells where induction hard-failed instead.",
            "cell_score": "+1 if induce succeeded AND the emitted engine has zero mechanical "
            "defects; 0 if induce succeeded but the engine is defective; -1 if induce HARD-FAILED "
            "(content failure). A server failure or a driver exception is a MISSING OBSERVATION, "
            "excluded, never a zero.",
            "unit_of_analysis": "GAME. Replicates within a game are pseudo-replicates, not "
            "independent trials -- treating them as independent inflated a p from 0.125 to 0.049 "
            "in a sibling run on 2026-08-01 and had to be corrected.",
            "test": "two-sided exact sign test on the per-game paired difference in net",
        },
        "SECONDARY": [
            "usable count per arm (the gate's own historical metric, reported so a reader can "
            "see which component moved)",
            "hard_failure count per arm (the component the historical metric omitted)",
            "engine_defect_reasks_delta (armedness: the gate must FIRE in B and C, never in A/AA)",
        ],
        "contrasts": ["b_vs_a", "c_vs_a", "c_vs_b", "aa_vs_a (noise floor)"],
        "MIN_REACHABLE_P": {
            "n_games": n_games,
            "test": "two-sided exact sign test",
            "if_all_games_discordant_and_unanimous": round(2.0 * 0.5**n_games, 12),
            "min_discordant_games_for_p_le_0.05": 6,
            "stated_before_results": "With 20 games the design can reach p = 1.9e-6. It reaches "
            "p <= 0.05 only if AT LEAST 6 games are discordant and unanimous (2 * 0.5^6 = "
            "0.03125). Fewer than 6 discordant games makes p <= 0.05 UNREACHABLE regardless of "
            "how large the effect looks.",
        },
        "roster": ROSTER,
        "n_replicates": N_REPLICATES,
        "seed_bases": {"paired_arms": SEED_BASE, "aa": AA_SEED_BASE},
        "pairing": "a_off, b_shipped and c_owns share a seed base, so within a (game, replicate) "
        "all three send the identical prompt at the identical seed. Attempt 0 is the same draw "
        "in all three; any divergence is the gate firing rather than the sampler.",
        "missing_is_never_zero": "a cell whose server failed or whose driver raised is EXCLUDED "
        "and counted in the missingness table. It is never scored 0 and never scored -1.",
        "REGIME": {
            "repeat_penalty_requested": REPEAT_PENALTY,
            "is_the_shipped_stack": REPEAT_PENALTY is None,
            "note": "the shipped stack has `_INDUCE_REPEAT_PENALTY = 1.1`. A run with "
            "repeat_penalty_requested='1.0' restores the pre-2026-07-31 payload byte-for-byte "
            "and is a COUNTERFACTUAL condition, NOT the shipped agent -- it exists because the "
            "shipped-stack run found the gate fires zero times, leaving its cost-when-it-fires "
            "unobservable. Identical in every arm of a given run, so it defines the regime "
            "rather than confounding an arm contrast.",
        },
        "flags_remain_default_off": True,
        "solve_provenance": "development_proxy",
        "not_submitted": "no scored or online ARC game is played; submission is operator-only",
    }
    prereg_text = json.dumps(prereg, indent=2, sort_keys=True)
    (OUT / "preregistration.json").write_text(prereg_text)
    prereg_sha = "sha256:" + sha(prereg_text)
    print(f"pre-registration written {prereg_sha}")

    if dry:
        (OUT / "meta_dry.json").write_text(
            json.dumps(
                {
                    "split_meta": split_meta,
                    "treatment_witness": treatment,
                    "prereg_sha256": prereg_sha,
                    "preconditions_checked": pre,
                },
                indent=2,
            )
        )
        print("DRY RUN: stopping before the first LLM call")
        return 0

    # ---------------- SERVER (one, all arms) ----------------------------------------
    # n_ctx 32768, NOT the shipped 81920, DECLARED because an undeclared config deviation is how
    # a measurement silently stops describing the thing it names. The shipped pool exists for
    # CONCURRENCY; this harness is strictly sequential. Identical in every arm, so it cannot
    # confound the contrast.
    os.environ["CARNOT_ARC_INDUCE_N_CTX"] = os.environ.get("RNC_N_CTX", "32768")
    # NOTE: CUDA_VISIBLE_DEVICES is deliberately NOT set. Setting it together with
    # CARNOT_ARC_GENERATOR_CUDA_GPU renumbers the cards, the headroom probe finds nothing, and
    # the generator silently falls back to the AMD iGPU HIP build while the artifact still says
    # "3090". The /proc witness below is what refuses that outcome rather than describing it.
    os.environ["CARNOT_ARC_GENERATOR_CUDA_GPU"] = GPU
    prop = e3.LocalGGUFProposer(port=PORT)
    print(f"launching {prop.repo_substr} on port {PORT} gpu {GPU} n_ctx {prop.n_ctx} ...")
    if not prop._ensure_server():  # noqa: SLF001
        (OUT / "blocked.json").write_text(
            json.dumps({"honest_verdict": "blocked_generator_server_failed_to_start"}, indent=2)
        )
        return 1
    actual_port = prop.port
    pid = listening_pid(actual_port)
    exe = os.readlink(f"/proc/{pid}/exe") if pid else None
    props = server_props(actual_port)
    server_witness = {
        "pid": pid,
        "exe_from_proc": exe,
        "port_requested": PORT,
        "port_actual": actual_port,
        "cuda_gpu": GPU,
        "reuse_refusals": list(getattr(prop, "reuse_refusals", [])),
        "n_ctx_declared": prop.n_ctx,
        "n_ctx_from_props": props.get("default_generation_settings", {}).get("n_ctx")
        or props.get("n_ctx"),
        "model_from_props": props.get("model_path") or props.get("model"),
        # The HIP build lives in build-hip/ and is the AMD iGPU path; CUDA is build/. A run that
        # silently landed on the iGPU would be ~6x slower and a DIFFERENT SUBSTRATE than the one
        # the artifact declares. Taken from /proc BEFORE the first measured call, and refused
        # outright rather than recorded as a caveat.
        "is_cuda_build": bool(exe and "build-hip" not in exe and exe.endswith("llama-server")),
        "mtp": prop.mtp,
        "kv_quant": prop.kv_quant,
        "n_gpu_layers": prop.n_gpu_layers,
        "max_tokens": prop.max_tokens,
        "tries": prop.tries,
        "timeout": prop.timeout,
    }
    (OUT / "server_witness.json").write_text(json.dumps(server_witness, indent=2))
    if not server_witness["is_cuda_build"]:
        (OUT / "blocked.json").write_text(
            json.dumps(
                {
                    "honest_verdict": "blocked_generator_not_on_the_cuda_build",
                    "server_witness": server_witness,
                },
                indent=2,
            )
        )
        print("BLOCKED: not the CUDA build ->", exe)
        return 1
    print("server:", json.dumps(server_witness))

    # ---------------- CELLS ----------------------------------------------------------
    def run_cell(game: str, rep: int, arm: str, tag: str, seed_base: int) -> dict:
        shown, held, cell = windows[game]
        cell_dir = SCRATCH / f"{game}__r{rep}__{tag}"
        # A per-cell store is not tidiness: a SHARED store is a cross-arm confound, because
        # arm A's engine gets read by arm B.
        assert SCRATCH in cell_dir.parents or cell_dir.parent == SCRATCH, "store escaped scratch"
        if cell_dir.exists():
            shutil.rmtree(cell_dir)
        cell_dir.mkdir(parents=True)
        e3.E3_DIR = cell_dir  # module global, read at call time by write + load

        apply_arm_env(arm)
        os.environ["CARNOT_ARC_GENERATOR_SEED"] = str(seed_base + rep)

        prompt = e3.induce_prompt(game, shown, cell)
        # Read off the instance with NO getattr default: a rename must fail loudly rather than
        # silently returning 0 forever and turning every missing observation into a real zero.
        sf0, cf0 = prop.n_server_failures, prop.n_content_failures
        gr0, er0 = prop.n_goal_defect_reasks, prop.n_induce_defect_reasks
        t = time.time()
        try:
            ok, msg = prop.induce(game, shown, cell)
            exc = None
        except Exception as e:  # noqa: BLE001
            ok, msg, exc = False, "", f"{type(e).__name__}: {e}"[:300]
        elapsed = time.time() - t

        row = {
            "game": game,
            "replicate": rep,
            "arm": arm,
            "tag": tag,
            "seed": seed_base + rep,
            "elapsed_s": round(elapsed, 2),
            "induce_ok": bool(ok),
            "induce_msg": str(msg)[:300],
            "exception": exc,
            "prompt_chars": len(prompt),
            "prompt_sha256": sha(prompt),
            "server_failures_delta": prop.n_server_failures - sf0,
            "content_failures_delta": prop.n_content_failures - cf0,
            # THE ARMEDNESS EVIDENCE. Must be >0 somewhere in b_shipped/c_owns and exactly 0
            # everywhere in a_off/aa, or the treatment never fired and this is a non-test.
            "engine_defect_reasks_delta": prop.n_induce_defect_reasks - er0,
            "goal_defect_reasks_delta": prop.n_goal_defect_reasks - gr0,
            "n_shown": len(shown),
            "n_heldout": len(held),
            "e3_dir": str(cell_dir),
            "reask_budget_seen": __import__(
                "carnot.agentic.arc_executable_world_model", fromlist=["x"]
            )._induce_defect_reasks(),  # noqa: SLF001
        }
        row["arm_flags_consistent"] = (
            (os.environ.get(FLAG_REASKS) == "0") == (arm == "a_off")
            and (os.environ.get(FLAG_OWNS) == "1") == (arm == "c_owns")
            and FLAG_GOAL not in os.environ
        )
        wm = cell_dir / game / "world_model.py"
        row["engine_file_exists"] = wm.exists()
        if wm.exists():
            code = wm.read_text()
            row["engine_sha256"] = sha(code)
            row["engine_bytes"] = len(code)
            row["split_induce_signature"] = "split induce" in str(msg)
        return row

    jobs = [
        (g, r, arm, tag, sb) for r in range(N_REPLICATES) for g in ROSTER for (arm, tag, sb) in ARMS
    ]
    rows = []
    for i, (g, r, arm, tag, sb) in enumerate(jobs):
        cache = CELLS / f"{g}__r{r}__{tag}.json"
        if cache.exists():
            rows.append(json.loads(cache.read_text()))
            continue
        if time.time() - t0 > WALL_BUDGET_S:
            print(f"wall budget reached after {i} of {len(jobs)} cells; stopping cleanly")
            break
        row = run_cell(g, r, arm, tag, sb)
        cache.write_text(json.dumps(row, indent=1))
        rows.append(row)
        print(
            f"[{i + 1}/{len(jobs)}] {g} r{r} {tag}: ok={row['induce_ok']} "
            f"{row['elapsed_s']}s reasks={row['engine_defect_reasks_delta']} "
            f"cf={row['content_failures_delta']}",
            flush=True,
        )

    (OUT / "rows.json").write_text(json.dumps(rows, indent=1))
    (OUT / "meta.json").write_text(
        json.dumps(
            {
                "prereg_sha256": prereg_sha,
                "preconditions_checked": pre,
                "server_witness": server_witness,
                "split_meta": split_meta,
                "treatment_witness": treatment,
                "n_cells": len(rows),
                "n_jobs": len(jobs),
                "duration_s": round(time.time() - t0, 1),
            },
            indent=2,
        )
    )
    print(f"done: {len(rows)} cells in {round(time.time() - t0, 1)}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
