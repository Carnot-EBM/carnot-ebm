"""A/B: does rejecting a CLEAN-BUT-INERT induced engine cost usable-engine yield?

WHAT IS BEING TESTED. `CARNOT_ARC_INDUCE_REJECT_INERT=1` (shipped 2026-08-01, DEFAULT OFF) makes
`LocalGGUFProposer._engine_defects` report `engine_inert` when a mechanically-clean engine
predicts that no action changes anything, which spends one of the existing
`_INDUCE_DEFECT_REASKS` budget on a re-ask. The 2026-08-01 taxonomy
(`results/outer_loop_arc_generation_taxonomy_20260801.json`) found inertness is the largest
single failure class -- 26 of 172 candidates, 15.1%, more than every code-validity class combined
-- and the only class the live path took no action on.

THE PRIMARY MEASURES THE COST, NOT THE BENEFIT, AND THAT IS DELIBERATE. Usable-engine yield is
the SHIPPED `arc_engine_static_validation.validate_engine_code` definition, which calls an inert
engine CLEAN. So under this metric the intervention CANNOT gain: rejecting inert engines can only
leave yield flat (the re-ask returns something also usable) or LOWER it (the re-ask returns
something defective, and the budget is spent so it is accepted anyway). That is exactly the
question worth asking of a live-path change whose benefit is already argued from a frozen corpus:
what does it cost? Using the shipped definition also keeps the measurement non-circular -- if
inertness were folded into `validate_engine_code`, the treatment and the outcome would be the
same object and arm B would "win" by definition.

The BENEFIT the taxonomy predicted (+9.8pp live engines) is a NAMED SECONDARY below, reported
with its own caveat: arm B targets it, so part of any gain is definitional.

WHAT MAKES THE ARMS COMPARABLE, and why this A/B is unusually clean:

  * ONE llama-server serves BOTH arms, on a non-default port, CUDA build proven from
    /proc/<pid>/exe, n_ctx read back from /props.
  * THE PROMPT IS IDENTICAL IN BOTH ARMS. This flag changes no prompt text; it changes an
    accept/reject decision AFTER a completion arrives. The witness below asserts byte-equality of
    the induce prompt across arms per game, which is a stronger treatment witness than a
    block-diff -- there is nothing to diff.
  * THE SAMPLER IS SEEDED (`CARNOT_ARC_GENERATOR_SEED`), the same seed in both arms of a
    (game, replicate) pair. `LocalGGUFProposer.sampling_seed` records a MEASURED 40% run-to-run
    divergence under identical code when the seed is absent.
  * AN A/A ARM RUNS ANYWAY. Seeding is a claim about the sampler, not about the server, and the
    2026-08-01 object-perception run's A/A came back byte-identical on only 1 of 4 cells. Without
    the floor an effect is uninterpretable, so it is measured rather than assumed.

CLUSTERING IS AT THE GAME. Replicates within a game are averaged into ONE per-game mean before
pairing. 20 games x 4 replicates is 20 independent units, not 80. Treating replicates as trials
inflated a p from 0.125 to 0.049 on 2026-07-31 and had to be corrected.

NO GENERATED CODE IS EXECUTED IN THIS INTERPRETER. Everything this file runs against a generated
engine goes through `arc_engine_static_validation`, whose execution paths are killable
subprocesses with a wall-clock bound. The verifier score and the state-graph probe -- which are
NOT bounded -- are deferred to `score.py`'s per-cell worker processes.

NOT SUBMITTED: no scored or online ARC game is played. Submission is operator-only.
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
ROOT = Path("/home/ianblenke/github.com/ianblenke/carnot")

# E3_DIR is read at IMPORT time, so it must be set before the import. results/arc_e3 is EVIDENCE.
SCRATCH = HERE / "e3_store"
SCRATCH.mkdir(parents=True, exist_ok=True)
os.environ["CARNOT_ARC_E3_DIR"] = str(SCRATCH)

if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))

PORT = int(os.environ.get("INERT_PORT", "41791"))
GPU = os.environ.get("INERT_GPU", "0")
N_REPLICATES = int(os.environ.get("INERT_REPLICATES", "4"))
SEED_BASE = int(os.environ.get("INERT_SEED_BASE", "8100"))
WALL_BUDGET_S = float(os.environ.get("INERT_WALL_BUDGET_S", "30000"))
FLAG = "CARNOT_ARC_INDUCE_REJECT_INERT"

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
CELLS = OUT / "cells"
CELLS.mkdir(exist_ok=True)
ENGINES = OUT / "engines"
ENGINES.mkdir(exist_ok=True)
WINDOWS = OUT / "windows"
WINDOWS.mkdir(exist_ok=True)

# The 2026-08-01 object-perception roster, reused UNCHANGED so this measurement lands on the same
# 20 games as the corpus the taxonomy and the metric-validity analysis were computed over. Its
# rule: builds a progress window AND its held-out tail has >=1 VERIFIER-GRADABLE changing
# transition. Reusing it rather than re-deriving one also means the roster cannot have been
# chosen after seeing which games favour the treatment.
ROSTER = [
    "ls20",
    "s5i5",
    "tu93",
    "cn04",
    "m0r0",
    "sk48",
    "ar25",
    "tr87",
    "g50t",
    "re86",
    "bp35",
    "sb26",
    "lf52",
    "su15",
    "lp85",
    "cd82",
    "wa30",
    "sc25",
    "tn36",
    "ka59",
]


def sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


def min_reachable_two_sided_p(n_disc: int) -> float:
    """The smallest two-sided sign-test p attainable at `n_disc` discordant pairs -- the
    all-one-direction outcome. Stated BEFORE results so a design that cannot reach 0.05 is
    reported as such rather than as a null."""
    if n_disc <= 0:
        return 1.0
    return min(1.0, 2.0 * (0.5**n_disc))


def server_props(port: int) -> dict:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=20) as r:
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


def main() -> int:  # noqa: C901
    t0 = time.time()
    from carnot.agentic import arc_actions_to_progress as atp
    from carnot.agentic import arc_engine_static_validation as sv
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic import arc_world_model_trust_energy as wmte

    assert Path(e3.E3_DIR) == SCRATCH, f"E3_DIR isolation failed: {e3.E3_DIR}"

    # ---------------- PRECONDITIONS -------------------------------------------------
    import numpy as np

    gguf = e3._resolve_gguf(e3.ARC_LIVE_GENERATOR_REPO_SUBSTR)
    conductor = subprocess.run(
        ["systemctl", "--user", "is-active", "carnot-conductor.service"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    free_mb = e3._cuda_gpu_free_mb(int(GPU))
    # Does the treatment actually DO anything? A flag that cannot fire makes the ON arm the
    # control relabelled -- the failure mode a sibling experiment shipped with its HUD mask.
    probe_g = np.zeros((6, 6), dtype=int)

    class _T:
        def __init__(self, g, a):
            self.grid, self.action, self.data = g, a, None

    probe_trans = [_T(probe_g.copy(), a) for a in (1, 2, 3)]
    identity = "def engine(grid, action, data=None):\n    return grid.copy()\n"
    treatment_can_fire = sv.engine_inertness_defect(identity, probe_trans) is not None

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
            "principle": "a live conductor would contend for the same GPU and interleave its own "
            "induction into this store",
        },
        {
            "resource": f"cuda_gpu_{GPU}_has_headroom",
            "available": free_mb >= 20000,
            "detail": f"free={free_mb} MiB",
            "principle": "a concurrent workflow may hold the other card; launching without "
            "headroom would evict work this session does not own",
        },
        {
            "resource": "inertness_detector_fires",
            "available": treatment_can_fire,
            "principle": "if the treatment cannot fire, the ON arm is the control relabelled",
        },
        {
            "resource": "reject_inert_flag_default_off",
            "available": not e3._reject_inert_engines(),
            "principle": "the shipped default is the control arm; if it were already on, this "
            "measures ON vs ON",
        },
        {
            "resource": "port_free",
            "available": listening_pid(PORT) is None,
            "detail": f"port {PORT}",
            "principle": "reusing a stale server on the default port is how an arm silently gets a "
            "different model or context size",
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

    # ---------------- WINDOWS + EXPLICIT SPLIT (no LLM yet) --------------------------
    windows: dict[str, tuple] = {}
    split_meta: dict[str, dict] = {}
    for game in ROSTER:
        w = atp.build_progress_window(game)
        if w is None:
            raise SystemExit(f"roster game {game} lost its window since the probe")
        win, _full, cell = w
        win = list(win)
        shown, held = wmte._split_prefix_heldout(win)
        n_grad = sum(
            1 for t in held if t.level_after <= t.level_before and not (t.grid == t.next_grid).all()
        )
        windows[game] = (shown, held, int(cell))
        split_meta[game] = {
            "n_transitions": len(win),
            "n_shown": len(shown),
            "n_heldout": len(held),
            "heldout_gradable_changing": n_grad,
            "cell": int(cell),
            "shown_n_changing": int(sum(1 for t in shown if not (t.grid == t.next_grid).all())),
        }
        assert n_grad >= 1, f"{game} has no verifier-gradable changing held-out row"
        # Persist the split so `score.py`'s workers grade against the EXACT transitions this run
        # withheld, rather than rebuilding a window later and hoping it comes out the same.
        with open(WINDOWS / f"{game}.pkl", "wb") as fh:
            pickle.dump({"shown": shown, "held": held, "cell": int(cell)}, fh)
    print(f"windows built for {len(windows)} games")

    # ---------------- TREATMENT WITNESS: THE PROMPT DOES NOT MOVE --------------------
    # This flag changes an accept/reject decision, not any prompt text. So the witness is
    # byte-EQUALITY across arms, not a block diff. If these ever differ, something other than the
    # intervention is varying and the contrast is confounded.
    treatment = []
    for game, (shown, held, cell) in windows.items():
        os.environ.pop(FLAG, None)
        p_off = e3.induce_prompt(game, shown, cell)
        os.environ[FLAG] = "1"
        p_on = e3.induce_prompt(game, shown, cell)
        os.environ.pop(FLAG, None)
        treatment.append(
            {
                "game": game,
                "off_sha256": sha(p_off),
                "on_sha256": sha(p_on),
                "prompt_identical_across_arms": p_off == p_on,
                "prompt_chars": len(p_off),
            }
        )
    bad = [t["game"] for t in treatment if not t["prompt_identical_across_arms"]]
    if bad:
        (OUT / "blocked.json").write_text(
            json.dumps(
                {
                    "honest_verdict": "blocked_treatment_witness_prompt_moved",
                    "failing_games": bad,
                    "treatment": treatment,
                },
                indent=2,
            )
        )
        print("TREATMENT WITNESS FAILED (prompt differs across arms):", bad)
        return 1
    print(f"treatment witness OK: prompt byte-identical across arms on {len(treatment)} games")

    # ---------------- PRE-REGISTRATION (before the first LLM call) --------------------
    prereg = {
        "experiment": "arc_inert_engine_rejection_ab_usable_yield",
        "written_before_any_llm_call": True,
        "question": "Does rejecting a clean-but-INERT induced engine and re-asking cost "
        "usable-engine yield, as measured by the SHIPPED validate_engine_code?",
        "arms": {
            "control": f"{FLAG} unset (the shipped default)",
            "treatment": f"{FLAG}=1",
            "aa": f"{FLAG} unset, re-run at the identical seed -- the nondeterminism floor",
        },
        "PRIMARY": {
            "metric": "usable-engine yield: the fraction of a game's replicates whose FINAL "
            "accepted engine has validate_engine_code(...) == [] under the SHIPPED "
            "definition (required=('engine',), the cell's own transitions, the "
            "proposer's stop_type and budget) -- i.e. exactly the arguments "
            "_engine_defects passes on the live path",
            "direction_of_interest": "NON-INFERIORITY. Under the shipped definition an inert "
            "engine is CLEAN, so the treatment cannot gain here; it can "
            "only stay flat or lose. The question is what it costs.",
            "why_the_shipped_definition": "folding inertness into validate_engine_code would make "
            "the treatment and the outcome the same object, and arm "
            "B would win by construction",
            "test": "exact two-sided paired sign test over GAMES (ties dropped)",
            "clustering": "GAME. Replicates within a game are averaged into ONE per-game mean "
            "before pairing: 20 games x 4 replicates is 20 units, not 80.",
            "alpha": 0.05,
        },
        "SECONDARY_the_taxonomy_claim": {
            "metric": "live-engine yield: usable AND engine_changes_anything_bounded is True",
            "predicted": "+9.8pp live engines per candidate (taxonomy leave-one-out within game)",
            "CAVEAT_stated_before_results": "arm B TARGETS this metric, so part of any gain is "
            "definitional. It is not fully definitional: the "
            "re-ask budget is 1, so a still-inert second answer "
            "is accepted anyway and counts against the treatment.",
        },
        "SECONDARIES_downstream_quality": [
            "held-out change_fidelity (WorldModelVerifier) -- a yield win that produces only "
            "inert or wrong engines must be visible as such",
            "probe_depth_reached -- the bounded 600-call state-graph probe that "
            "results/outer_loop_arc_metric_validity_20260801.json found predicts plannability "
            "(AUC 0.787) where change_fidelity does not (AUC 0.609, CI contains chance). "
            "Imported as the currently-best-known proxy, NOT as a validated selector: that "
            "artifact states it was SELECTED as a family maximum and needs a prospective test.",
            "accuracy, cell_recall, change_accuracy",
        ],
        "COST_secondaries": ["completion calls per cell", "wall seconds per cell"],
        "multiplicity": "One primary. The secondaries are exploratory; a p<0.05 on any of them "
        "reads against a Bonferroni threshold of 0.05/6 = 0.00833.",
        "roster": ROSTER,
        "roster_provenance": "reused UNCHANGED from "
        "results/arc_object_perception_ab_change_fidelity_20260801, so it "
        "cannot have been chosen after seeing which games favour the "
        "treatment",
        "n_games": len(ROSTER),
        "n_replicates_per_cell": N_REPLICATES,
        "POWER_STATED_UP_FRONT": {
            "min_reachable_two_sided_p_if_all_20_games_discordant": round(
                min_reachable_two_sided_p(len(ROSTER)), 12
            ),
            "n_discordant_needed_for_p_below_0.05": 6,
            "HONEST_EXPECTATION": "The treatment can only fire on a cell whose attempt-0 engine "
            "is clean AND inert. The taxonomy's base rate is 12-15%, so "
            "over 20 games x 4 replicates roughly 10-12 cells are expected "
            "to fire, spread across perhaps 8-10 distinct games. On the "
            "PRIMARY a fired cell only creates a discordance if the re-ask "
            "returns DEFECTIVE code, which the taxonomy puts at roughly 1 "
            "in 5. So the expected number of primary-discordant games is "
            "about 2-3, and 6 are needed for p<0.05.",
            "CAN_THE_PRIMARY_REACH_0.05": "PROBABLY NOT. This design is well powered to detect a "
            "LARGE cost (a re-ask that usually returns junk) and "
            "underpowered for a small one. A null on the primary "
            "must therefore be reported as 'no large cost "
            "detected', NOT as 'no cost'. Said here, before any "
            "result, rather than as a post-hoc excuse.",
            "the_secondary_is_better_powered": "every fired cell that converts inert -> live is a "
            "discordance, so roughly 6-8 discordant games are "
            "plausible there",
        },
        "STOPPING_RULE": "every roster game x both arms x N replicates, or the wall budget. "
        "Analysis runs ONCE, after collection stops. No peeking-and-extending. "
        "Only (game, replicate) pairs where BOTH arms ran enter the analysis.",
        "MISSING_VS_ZERO": {
            "missing": "server failure, HTTP error, harness exception, or a completion truncated "
            "by the token cap. The cell is EXCLUDED and counted -- never scored 0.",
            "zero": "a complete response whose code is mechanically defective. That is a real "
            "failure and scores 0 on usable-engine yield.",
        },
        "generator": {
            "repo_substr": e3.ARC_LIVE_GENERATOR_REPO_SUBSTR,
            "gguf": str(gguf),
            "one_server_both_arms": True,
            "port": PORT,
            "cuda_gpu": GPU,
            "sampler_seeded": True,
            "seed_scheme": "CARNOT_ARC_GENERATOR_SEED = SEED_BASE + replicate, the SAME seed in "
            "both arms of a (game, replicate) pair",
        },
        "AA_CONTROL": "replicate 0's control arm is re-run at the identical seed on 6 games. If "
        "A/A is not byte-identical the residual nondeterminism is reported as a "
        "FLOOR on any effect claim. The 2026-08-01 object-perception A/A came back "
        "byte-identical on only 1 of 4 cells, so a failure here is expected.",
        "MECHANISTIC_WITNESS": "for every (game, replicate) pair, whether arm A's final engine "
        "was inert is recorded. The arms may only diverge where it was. A "
        "divergence with no inert trigger is the nondeterminism floor, not "
        "the treatment, and is counted separately.",
        "flag_remains_default_off": True,
        "not_submitted": "no scored or online ARC game is played; submission is operator-only",
    }
    prereg_text = json.dumps(prereg, indent=2, sort_keys=True)
    (OUT / "preregistration.json").write_text(prereg_text)
    prereg_sha = "sha256:" + sha(prereg_text)
    print(f"pre-registration {prereg_sha}")
    print(f"  PRIMARY = usable-engine yield (shipped validate_engine_code); {len(ROSTER)} games")
    print(
        f"  min reachable two-sided p = {min_reachable_two_sided_p(len(ROSTER)):.3e}; "
        f"p<0.05 needs >=6 discordant pairs"
    )
    print("  STATED UP FRONT: the primary is probably underpowered -- see POWER_STATED_UP_FRONT")

    if os.environ.get("INERT_DRY") == "1":
        (OUT / "meta_dry.json").write_text(
            json.dumps(
                {
                    "split_meta": split_meta,
                    "treatment_witness": treatment,
                    "prereg_sha256": prereg_sha,
                },
                indent=2,
            )
        )
        print("DRY RUN: stopping before the first LLM call")
        return 0

    # ---------------- SERVER (one, both arms) ---------------------------------------
    # n_ctx 32768 rather than the shipped 81920: this harness issues requests strictly
    # sequentially, so the 4-slot concurrency pool the larger value exists for cannot be used, and
    # at 81920 the guard offloads FFN layers to system RAM. Identical in BOTH arms, so it cannot
    # confound the contrast; it only buys replicates. Same choice, same reason, as the sibling
    # object-perception run.
    os.environ["CARNOT_ARC_INDUCE_N_CTX"] = os.environ.get("INERT_N_CTX", "32768")
    os.environ["CARNOT_ARC_GENERATOR_CUDA_GPU"] = GPU
    prop = e3.LocalGGUFProposer(port=PORT)
    print(f"launching {prop.repo_substr} on port {PORT} gpu {GPU} n_ctx {prop.n_ctx} ...")
    if not prop._ensure_server():
        (OUT / "blocked.json").write_text(
            json.dumps({"honest_verdict": "blocked_generator_server_failed_to_start"}, indent=2)
        )
        return 1
    # `_ensure_server` MOVES to a fresh port if ours is unusable, so read it back off the instance
    # rather than trusting the one we asked for.
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
        # The HIP build lives in build-hip/ and is the AMD iGPU path; the CUDA build is build/.
        # A run that silently landed on the iGPU would be ~6x slower and is a different substrate.
        "is_cuda_build": bool(exe and "build-hip" not in exe and exe.endswith("llama-server")),
        "mtp": prop.mtp,
        "kv_quant": prop.kv_quant,
        "n_gpu_layers": prop.n_gpu_layers,
        "max_tokens": prop.max_tokens,
    }
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
    (OUT / "server_witness.json").write_text(json.dumps(server_witness, indent=2))

    # ---------------- CELLS ----------------------------------------------------------
    def run_cell(game: str, rep: int, arm: str, tag: str) -> dict:
        shown, held, cell = windows[game]
        cell_dir = SCRATCH / f"{game}__r{rep}__{tag}"
        if cell_dir.exists():
            shutil.rmtree(cell_dir)
        cell_dir.mkdir(parents=True)
        e3.E3_DIR = cell_dir  # module global, read at call time by write + load

        if arm == "on":
            os.environ[FLAG] = "1"
        else:
            os.environ.pop(FLAG, None)
        os.environ["CARNOT_ARC_GENERATOR_SEED"] = str(SEED_BASE + rep)

        prompt = e3.induce_prompt(game, shown, cell)
        # Read these off the instance with NO default. An earlier sibling harness used a
        # misspelled attribute, getattr silently returned 0 forever, and the MISSING-vs-ZERO rule
        # could never fire -- a missing observation scored as a real zero.
        sf0, cf0 = prop.n_server_failures, prop.n_content_failures
        calls0, reask0 = prop.n_completion_calls, prop.n_induce_defect_reasks
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
            "seed": SEED_BASE + rep,
            "elapsed_s": round(elapsed, 2),
            "induce_ok": bool(ok),
            "induce_msg": str(msg)[:300],
            "exception": exc,
            "prompt_chars": len(prompt),
            "prompt_sha256": sha(prompt),
            "server_failures_delta": prop.n_server_failures - sf0,
            "content_failures_delta": prop.n_content_failures - cf0,
            "completion_calls_delta": prop.n_completion_calls - calls0,
            "defect_reasks_delta": prop.n_induce_defect_reasks - reask0,
            "flag_seen_by_proposer": e3._reject_inert_engines(),
            "last_stop_type": prop.last_stop_type,
            "max_tokens": prop.max_tokens,
            "n_shown": len(shown),
            "n_heldout": len(held),
            "cell": cell,
        }
        # arm integrity: the ON arm must see the flag on, the OFF arm must not
        row["arm_flag_consistent"] = row["flag_seen_by_proposer"] == (arm == "on")

        wm = cell_dir / game / "world_model.py"
        row["engine_file_exists"] = wm.exists()
        code = wm.read_text() if wm.exists() else None
        if code is not None:
            row["engine_sha256"] = sha(code)
            row["engine_bytes"] = len(code)
            # Keep the emitted engine. `score.py` reads it from here, so scoring never depends on
            # the scratch store surviving.
            (ENGINES / f"{game}__r{rep}__{tag}.py").write_text(code)

        # THE PRIMARY, computed with the SHIPPED definition and the SHIPPED arguments. Both
        # execution paths reachable from here (`dry_run_defects`) are killable subprocesses with a
        # wall-clock bound, so no generated code runs unbounded in this interpreter.
        row["usable"] = False
        row["defect_kinds"] = None
        row["engine_inert"] = None
        if code is not None:
            defects = sv.validate_engine_code(
                code,
                transitions=list(shown),
                stop_type=prop.last_stop_type,
                required=("engine",),
                budget=prop.max_tokens,
            )
            row["defect_kinds"] = sorted({d.kind for d in defects})
            row["usable"] = not defects
            # Recorded for EVERY cell in BOTH arms -- this is the mechanistic witness. In arm A it
            # says whether the treatment would have had anything to bite on; in arm B it says
            # whether the re-ask actually fixed it.
            row["engine_inert"] = sv.engine_changes_anything_bounded(code, list(shown)) is False
        row["live"] = bool(row["usable"] and row["engine_inert"] is False)

        # MISSING vs ZERO. A server-side failure or a harness exception is a missing observation.
        # A complete response that produced defective code is a real zero.
        row["missing"] = bool(row["server_failures_delta"] > 0 or exc is not None)
        row["missing_reason"] = (
            "server_failure"
            if row["server_failures_delta"] > 0
            else ("harness_exception" if exc else None)
        )

        (CELLS / f"{game}__r{rep}__{tag}.json").write_text(json.dumps(row, indent=2))
        return row

    order: list[tuple[str, int, str, str]] = []
    for rep in range(N_REPLICATES):
        for game in ROSTER:
            # alternate which arm goes first so any server drift is not confounded with arm
            arms = ["off", "on"] if (rep + ROSTER.index(game)) % 2 == 0 else ["on", "off"]
            for a in arms:
                order.append((game, rep, a, a))
    # A/A control: replicate 0's control arm, repeated at the identical seed.
    for game in ROSTER[:6]:
        order.append((game, 0, "off", "offAA"))

    rows: list[dict] = []
    print(f"\n{len(order)} cells queued\n")
    for i, (game, rep, arm, tag) in enumerate(order, 1):
        if time.time() - t0 > WALL_BUDGET_S:
            print(f"WALL BUDGET reached after {i - 1} cells; stopping collection")
            break
        r = run_cell(game, rep, arm, tag)
        rows.append(r)
        print(
            f"[{i}/{len(order)}] {game} r{rep} {tag:6} ok={r['induce_ok']} "
            f"usable={r['usable']} inert={r['engine_inert']} live={r['live']} "
            f"reask={r['defect_reasks_delta']} calls={r['completion_calls_delta']} "
            f"{r['elapsed_s']}s"
        )
        (OUT / "rows.json").write_text(json.dumps(rows, indent=2))

    meta = {
        "prereg_sha256": prereg_sha,
        "server_witness": server_witness,
        "split_meta": split_meta,
        "treatment_witness": treatment,
        "n_cells_run": len(rows),
        "n_cells_queued": len(order),
        "duration_s": round(time.time() - t0, 2),
        "seed_base": SEED_BASE,
        "n_replicates": N_REPLICATES,
        "roster": ROSTER,
        "gguf": str(gguf),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\ncollection done: {len(rows)} cells in {meta['duration_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
