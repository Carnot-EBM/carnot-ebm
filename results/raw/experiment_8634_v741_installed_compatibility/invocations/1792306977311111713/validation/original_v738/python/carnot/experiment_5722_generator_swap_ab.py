"""Exp 5722/5723: generator-swap induction-floor test (REQ-ARC-WMTE-5722, REQ-ARC-WMTE-5723).

WHY THIS EXISTS
---------------
The REQ-ARC-WMTE-5714 (`/think` vs `/no_think`) and REQ-ARC-WMTE-5716/5717/5718
(playbook retrieval) A/Bs, both re-run on the REQ-ARC-WMTE-5720/5721 actions-to-progress
harness, landed on the SAME wall: the frozen live generator (Qwen3.5-9B-MTP) sat at a hard
`heldout_accuracy=0.0` floor on EVERY game/trial with ZERO real level-ups, and neither
reasoning-mode nor retrieval-augmentation moved it. Both of those levers change the
induction PROMPT. This experiment tests the one hypothesis they could not: that the binding
constraint is model CAPACITY, not prompting -- by swapping a much stronger open-weight SOTA
generator into the induce call ONLY and holding everything else (harness, roster, seeded
level-up windows, planner, verifier, execution) identical.

Generators (arm is fixed = "frozen" for all; ONLY the LLM differs):
  * qwen9b  -- unsloth/Qwen3.5-9B-MTP-GGUF   (the frozen live baseline, re-measured FRESH)
  * gemma31 -- unsloth/gemma-4-31B-it-GGUF   (~18.3GB Q4_K_M -- PRIMARY candidate, REQ-5722)
  * gemma12 -- unsloth/gemma-4-12B-it-GGUF   (~7.1GB  Q4_K_M -- secondary size datum, REQ-5723)

CLEAN SWAP. The `frozen` arm sets CARNOT_ARC_CODEONLY_INDUCE=1, so `LocalGGUFProposer.generate`
takes the codeonly branch, which prepends the model-agnostic `_L2_CODEONLY_DIRECTIVE` and skips
the proposer's `no_think_prefix` FIELD. So the induce prompt is byte-identical in STRUCTURE
across generators -- the ONLY content difference is the model. (Note: `_L2_CODEONLY_DIRECTIVE`
itself opens with a literal `/no_think` token, so `/no_think` IS fed to every generator
including Gemma -- but IDENTICALLY, so it is not a DIFFERENTIAL confound; it does not bias the
generator ranking. The `no_think_prefix` FIELD-level `/think` vs `/no_think` toggle -- the
separate, already-ruled-out reasoning lever -- is what is not applied here.) We deliberately do
NOT re-run the reason/retrieval/static arms per-generator: the field-level `/think`/`/no_think`
are Qwen tokens meaningless to Gemma, and reasoning/retrieval were already ruled out, so mixing
them would conflate the generator question with prompt questions already answered.

ATTRIBUTION FIX (2026-07-18). An adversarial review of this experiment found the reused
run_seeded_progress harness discarded proposer.induce's success flag and reported induction_ok
if ANY per-GAME world_model.py loaded -- so a FAILED induce silently re-read (and was scored on)
an EARLIER generator's engine. Fixed in arc_actions_to_progress.py (delete the prior engine
before each induce + gate induction_ok on the real induce_ok + record induce_ok per row). The
pre-fix gemma12 half was retracted and re-run clean; gemma31/qwen9b remain valid (proven
self-induced by the trial0!=trial1 differential + mtimes -- see the artifact's
attribution_integrity block).

Substrate: live_llm_inference. Each generator induces on its own CUDA llama-server pinned to
GPU 1 (the outer-loop's card) via CARNOT_ARC_GENERATOR_CUDA_GPU=1 -- servers run sequentially
(one generator's block completes and its server is terminated before the next launches) so the
18GB Gemma-31B and the 11.5GB Qwen never contend for the 24GB card. MTP is OFF for the dense
Gemma GGUFs (no self-draft heads; MTP is speed-only/content-neutral). The Qwen baseline is
re-measured FRESH in this same invocation (not reused from the exp5720 shard) for a clean
matched comparison; it is expected to reproduce the 0.0 floor as a sanity check.

Provenance: development_proxy on PUBLIC games -- NOT a hidden-game self-discovery solve. The
win oracle is the level counter (verifier_is_oracle=False); the dense progress proxy reads the
live runtime game object via the adapter's public hand_verifier (used_env_source=True), never a
game's .py source (read_game_source=False). This NEVER flips the frozen live default (a bigger,
operator-only graduation decision) and NEVER submits. Wall-clock here is NOT Kaggle-representative
(24GB 3090 dev card, not the ~16GB eval GPU); this tests the CONTENT question only.

RESUMABLE: every (generator, game, trial) cell is appended to a JSONL shard as it completes, so
an interrupted run resumes without redoing finished cells (single-synchronous-resume-accumulate).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

# Pin the generator llama-server to GPU 1 (the outer-loop's card) BEFORE anything imports the
# proposer module -- _generator_server_and_env reads this env at server-launch time.
os.environ.setdefault("CARNOT_ARC_GENERATOR_CUDA_GPU", "1")
# One-shot induce (no bounded-refinement loop): keeps each cell bounded + FAIR across generators
# and directly single-shot-comparable to the prior REQ-ARC-WMTE-5714/5719/5720 experiments.
os.environ.setdefault("CARNOT_ARC_STALL_REFACTOR_LOOP", "0")

from carnot.agentic import arc_actions_to_progress as atp  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SHARD = REPO / "results" / "exp5722_genswap_shard.jsonl"
GEMMA31_ARTIFACT = REPO / "results" / "experiment_5722_generator_swap_gemma31_ab.json"
GEMMA12_ARTIFACT = REPO / "results" / "experiment_5723_generator_swap_gemma12_ab.json"

# SAME roster / trials as REQ-ARC-WMTE-5720/5721 (the experiments this extends).
ROSTER = ["ls20", "tr87", "lp85", "g50t", "m0r0", "ft09"]
TRIALS = [0, 1]

# Each generator's proposer config. arm is ALWAYS "frozen" (codeonly default); the ONLY thing
# that varies across generators is the underlying GGUF. no_think_prefix is set by apply_arm(
# "frozen") anyway but is NEVER used because the codeonly branch is taken -- declared here only to
# match the frozen live Qwen config exactly. Distinct ports so a stale server is never reused.
GENERATORS: dict[str, dict[str, Any]] = {
    "qwen9b": {
        "repo_substr": "Qwen3.5-9B-MTP",
        "port": 8946,
        "mtp": True,
        "kv_quant": "q8_0",
        "no_think_prefix": "/no_think\n",
        "hf_id": "unsloth/Qwen3.5-9B-MTP-GGUF",
        "role": "the FROZEN live generator (re-measured fresh as the matched baseline)",
    },
    "gemma31": {
        "repo_substr": "gemma-4-31B-it",
        "port": 8944,
        "mtp": False,
        "kv_quant": "q8_0",
        "no_think_prefix": "",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "role": "PRIMARY candidate: dense 31B SOTA generator (REQ-ARC-WMTE-5722)",
    },
    "gemma12": {
        "repo_substr": "gemma-4-12B-it",
        "port": 8945,
        "mtp": False,
        "kv_quant": "q8_0",
        "no_think_prefix": "",
        "hf_id": "unsloth/gemma-4-12B-it-GGUF",
        "role": "secondary size datum: dense 12B SOTA generator (REQ-ARC-WMTE-5723)",
    },
}
# Run order: baseline first (fast, sanity-check the floor reproduces), then the candidates.
GEN_ORDER = ["qwen9b", "gemma31", "gemma12"]

PROGRESS_METRICS = ["reached_levelup", "hv_progress", "plan_found"]
INDUCTION_METRICS = [
    "heldout_accuracy",
    "cell_recall",
    "goal_predicate_accuracy",
    "levelup_positive_recall",
]
ALL_METRICS = PROGRESS_METRICS + INDUCTION_METRICS


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


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
            rows[(r["generator"], r["game"], int(r["trial"]))] = r
    return rows


def _append_shard(row: dict[str, Any]) -> None:
    SHARD.parent.mkdir(parents=True, exist_ok=True)
    with SHARD.open("a") as f:
        f.write(json.dumps(row) + "\n")


def check_preconditions() -> dict[str, Any]:
    from carnot.agentic.arc_executable_world_model import _resolve_gguf

    checks: dict[str, Any] = {}
    try:
        from carnot.agentic import arc_solver_kit as kit

        kit.offline_arcade()
        checks["offline_arcade_import"] = True
    except Exception as exc:
        checks["offline_arcade_import"] = False
        checks["offline_arcade_error"] = repr(exc)[:200]
    # GPU offload must be real (not a silent CPU wheel) -- the CLAUDE.md llama-cpp-python note.
    try:
        from llama_cpp import llama_cpp as _b

        checks["llama_supports_gpu_offload"] = bool(_b.llama_supports_gpu_offload())
    except Exception as exc:
        checks["llama_supports_gpu_offload"] = False
        checks["llama_offload_error"] = repr(exc)[:200]
    checks["generator_cuda_gpu"] = os.environ.get("CARNOT_ARC_GENERATOR_CUDA_GPU")
    resolved: dict[str, Optional[str]] = {}
    for gen, cfg in GENERATORS.items():
        p = _resolve_gguf(cfg["repo_substr"])
        resolved[gen] = p
        checks[f"{gen}_gguf_cached"] = bool(p)
    checks["resolved_gguf_paths"] = resolved
    return checks


def _make_proposer(gen: str):
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    cfg = GENERATORS[gen]
    return LocalGGUFProposer(
        repo_substr=cfg["repo_substr"],
        port=cfg["port"],
        mtp=cfg["mtp"],
        kv_quant=cfg["kv_quant"],
        no_think_prefix=cfg["no_think_prefix"],
        max_tokens=4096,
        timeout=600,
    )


def run_all() -> list[dict[str, Any]]:
    done = _load_shard()
    total = len(GEN_ORDER) * len(ROSTER) * len(TRIALS)
    log(f"resume: {len(done)}/{total} cells already in shard")

    # Pre-build the per-game induction windows ONCE (build_window solves the game offline; slow).
    windows: dict[str, Any] = {}
    for game in ROSTER:
        w = atp.build_progress_window(game)
        windows[game] = w
        if w is None:
            log(f"SKIP {game}: no offline L1 window (build_progress_window returned None)")

    for gen in GEN_ORDER:
        # Skip the whole generator block if every cell is already sharded.
        pending = [
            (g, t)
            for g in ROSTER
            for t in TRIALS
            if windows.get(g) is not None and (gen, g, t) not in done
        ]
        if not pending:
            log(f"generator {gen}: all cells present, skipping")
            continue
        log(f"=== generator {gen} ({GENERATORS[gen]['repo_substr']}) : {len(pending)} cells ===")
        prop = _make_proposer(gen)
        try:
            if not prop._ensure_server():
                log(f"  !! server failed to start for {gen}; recording blocked cells")
                for game, trial in pending:
                    row = {
                        "generator": gen,
                        "game": game,
                        "arm": gen,
                        "trial": trial,
                        "induction_ok": False,
                        "plan_found": False,
                        "reached_levelup": False,
                        "error": "server_failed_to_start",
                        "wall_s": 0.0,
                    }
                    _append_shard(row)
                    done[(gen, game, trial)] = row
                continue
            for game, trial in pending:
                window, full_traj, cell = windows[game]
                log(f"RUN gen={gen} {game} trial={trial}")
                t0 = time.time()
                res = atp.run_seeded_progress(
                    game,
                    "frozen",
                    proposer=prop,
                    trial=trial,
                    window=window,
                    full_traj=full_traj,
                    cell=cell,
                )
                row = res.to_row()
                # Relabel: the harness records arm="frozen"; encode the GENERATOR in `arm` so
                # paired_by_game pairs generators (and keep an explicit `generator` field).
                row["generator"] = gen
                row["arm"] = gen
                _append_shard(row)
                done[(gen, game, trial)] = row
                log(
                    f"  -> ind_ok={row['induction_ok']} plan={row['plan_found']} "
                    f"levelup={row['reached_levelup']} hv={row['hv_progress']} "
                    f"heldout={row['heldout_accuracy']} cellrec={row['cell_recall']} "
                    f"goalpred={row['goal_predicate_accuracy']} lurec={row['levelup_positive_recall']} "
                    f"wall={row['wall_s']}s ({time.time() - t0:.0f}s)"
                )
        finally:
            # Free the card for the next generator's server (sequential, no VRAM contention).
            proc = getattr(prop, "_proc", None)
            if proc is not None:
                try:
                    proc.terminate()
                    proc.wait(timeout=15)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            time.sleep(3)
    return list(done.values())


def _repro_checksum(rows: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    h.update(Path(atp.__file__).read_bytes())
    h.update(Path(__file__).read_bytes())
    h.update(
        json.dumps(
            {"roster": ROSTER, "trials": TRIALS, "generators": GENERATORS, "gen_order": GEN_ORDER},
            sort_keys=True,
        ).encode()
    )
    h.update(json.dumps(sorted(json.dumps(r, sort_keys=True) for r in rows)).encode())
    return "sha256:" + h.hexdigest()


FIELD_PRINCIPLES = {
    "honest_verdict": "terminal-prefixed self-declared state (Verdict Terminal-Prefix Discipline).",
    "inference_substrate": "live_llm_inference -- the candidate GGUF really induces via the live "
    "LocalGGUFProposer.induce path on a CUDA llama-server; 60s floor applies "
    "per cell aggregate is not asserted (per-cell durations disclosed).",
    "random_seed": "the LLM sampling is stochastic (disclosed); trials are per-game replicates -- "
    "why we pair by GAME (average over trials) and report win/tie/loss + outlier-fragility.",
    "reproducibility_checksum": "content hash over harness+driver code + generator config + all rows.",
    "solve_provenance": "development_proxy -- PUBLIC-game offline dev measurement of the LIVE induce->"
    "plan->execute mechanism with the generator swapped, NOT a hidden-game solve.",
    "verifier_is_oracle": "False -- the win oracle is the level counter (frame.levels_completed); "
    "hand_verifier is only a dense progress MEASUREMENT, oracle-distinct.",
    "preconditions_checked": "GGUF cache + offline arcade + REAL GPU offload verified before inference.",
    "generator_is_only_variable": "arm is fixed 'frozen' (codeonly, model-agnostic prompt) for every "
    "generator; the induce LLM is the ONLY thing that changes, so any "
    "delta is attributable to model capacity, not prompting.",
}


def _verdict(prefix: str, comps: list[dict[str, Any]]) -> str:
    """Terminal-prefixed verdict. Prefer the primary progress metric (reached_levelup); if it is
    fully floored (no discriminating pairs), fall back to the densest induction discriminator
    (heldout_accuracy), so the verdict never over-claims a progress signal that isn't there."""

    def pick(metric):
        return next((c for c in comps if c.get("metric") == metric and c.get("n_game_pairs")), None)

    primary = pick("reached_levelup")
    floored = primary is None or (primary.get("mean_treat") == 0 and primary.get("mean_base") == 0)
    driver = pick("heldout_accuracy") if floored else primary
    if driver is None:
        return f"complete_{prefix}_no_comparable_pairs_metric_unavailable"
    tag = "reached_levelup_floored_heldout_fallback" if floored else "reached_levelup"
    p, md, frag = (
        driver.get("sign_test_p"),
        driver.get("mean_delta", 0.0),
        driver.get("outlier_fragile"),
    )
    if p is not None and p < 0.05 and not frag and abs(md) > 1e-9:
        direction = "helps" if md > 0 else "hurts"
        return f"complete_{prefix}_{tag}_stronger_generator_{direction}_signif_delta_{md}"
    if floored and abs(md) < 1e-9:
        return f"complete_{prefix}_floor_persists_stronger_generator_no_movement_delta_0.0"
    return f"complete_{prefix}_{tag}_no_reliable_signal_delta_{md}_signp_{p}_fragile_{bool(frag)}"


def build_artifact(
    *,
    exp_id: str,
    req: str,
    treat: str,
    base: str,
    question: str,
    rows: list[dict[str, Any]],
    extra_comparisons: list[dict[str, Any]],
    preconditions: dict[str, Any],
    duration_s: float,
) -> dict[str, Any]:
    comparisons = [
        {**atp.paired_by_game(rows, treat, base, metric=m), "contrast": f"{treat}_vs_{base}"}
        for m in ALL_METRICS
    ]
    comparisons += extra_comparisons
    used = [
        r
        for r in rows
        if r.get("arm") in (treat, base)
        or r.get("arm") in {c.get("treat") for c in extra_comparisons}
        or r.get("arm") in {c.get("base") for c in extra_comparisons}
    ]
    n_games = len({r["game"] for r in rows if r.get("arm") == treat})
    n_pairs_max = max(
        (
            c.get("n_game_pairs", 0)
            for c in comparisons
            if c.get("contrast") == f"{treat}_vs_{base}"
        ),
        default=0,
    )
    tcfg = GENERATORS[treat]
    bcfg = GENERATORS[base]
    any_levelup_treat = any(r.get("reached_levelup") for r in rows if r.get("arm") == treat)
    any_plan_treat = any(r.get("plan_found") for r in rows if r.get("arm") == treat)
    floor_moved = any(
        (r.get("heldout_accuracy") or 0) > 0 or r.get("reached_levelup")
        for r in rows
        if r.get("arm") == treat
    )
    return {
        "experiment": exp_id,
        "schema": f"carnot.{exp_id}.generator_swap_ab.v1",
        "requirements": [req],
        "prior_work_extended": [
            "REQ-ARC-WMTE-5720",
            "REQ-ARC-WMTE-5721",
            "REQ-ARC-WMTE-5714",
            "REQ-ARC-WMTE-5716",
            "REQ-ARC-WMTE-5717",
            "REQ-ARC-WMTE-5718",
        ],
        "question": question,
        "inference_substrate": "live_llm_inference",
        "model_specs": [
            {
                "name": tcfg["repo_substr"],
                "hf_id": tcfg["hf_id"],
                "quant": "Q4_K_M",
                "role": f"TREATMENT generator ({tcfg['role']})",
                "gguf_path": preconditions.get("resolved_gguf_paths", {}).get(treat),
                "mtp": tcfg["mtp"],
                "kv_quant": tcfg["kv_quant"],
                "server": f"CUDA llama-server GPU1 (CARNOT_ARC_GENERATOR_CUDA_GPU=1) port {tcfg['port']}, "
                f"-ngl 999, q8_0 KV, MTP={'on' if tcfg['mtp'] else 'off (dense GGUF, no self-draft heads)'}",
            },
            {
                "name": bcfg["repo_substr"],
                "hf_id": bcfg["hf_id"],
                "quant": "Q4_K_M",
                "role": f"BASELINE generator ({bcfg['role']})",
                "gguf_path": preconditions.get("resolved_gguf_paths", {}).get(base),
                "mtp": bcfg["mtp"],
                "kv_quant": bcfg["kv_quant"],
                "server": f"CUDA llama-server GPU1 port {bcfg['port']}, -ngl 999, q8_0 KV, "
                f"MTP={'on' if bcfg['mtp'] else 'off'}",
            },
        ],
        "honest_verdict": _verdict(
            exp_id.split("_")[1] + f"_{treat}_vs_{base}",
            [c for c in comparisons if c.get("contrast") == f"{treat}_vs_{base}"],
        ),
        "solve_provenance": "development_proxy",
        "verifier_is_oracle": False,
        "read_game_source": False,
        "used_env_source": True,
        "random_seed": TRIALS[0],
        "trials_per_arm": len(TRIALS),
        "field_principles": FIELD_PRINCIPLES,
        "preconditions_checked": preconditions,
        "sample_size": {
            "games": n_games,
            "trials_per_generator": len(TRIALS),
            "generators_compared": sorted(
                {treat, base}
                | {c.get("treat") for c in extra_comparisons}
                | {c.get("base") for c in extra_comparisons}
            ),
            "paired_unit": "game (metrics averaged over trials, paired by game)",
            "max_game_pairs": n_pairs_max,
            "disclosure": "SMALL N (<=6 game pairs); stochastic proposer. Report paired game "
            "win/tie/loss + exact sign-test p + outlier-fragility, NOT a bare mean. "
            "With <=6 pairs the sign test cannot reach p<0.05 unless every game agrees. "
            "A positive finding here is a DIRECTION to investigate, not a significance claim.",
        },
        "attribution_integrity": {
            "stale_engine_bug_found": "2026-07-18 adversarial review of this experiment found the "
            "REQ-ARC-WMTE-5720 harness discarded proposer.induce's success "
            "flag and marked a cell induction_ok if ANY per-GAME "
            "world_model.py loaded -- so a FAILED induce silently re-read an "
            "earlier generator's engine (run order qwen9b->gemma31->gemma12 "
            "over a shared file).",
            "fix": "arc_actions_to_progress.run_seeded_progress now DELETES the prior engine before "
            "each induce AND gates induction_ok on the real induce_ok, so a failed induce -> "
            "engine=None -> honest induction_ok=False (no mis-attributed stale score). Each "
            "row now carries induce_ok for audit.",
            "clean_rerun": "ALL THREE generators were RE-RUN from scratch under the fix (every cell "
            "carries induce_ok, so every scored engine is attributable to that cell's own "
            "generator). The pre-fix run is preserved at "
            "results/exp5722_genswap_shard.buggy_prefix_20260718.jsonl + "
            "results/experiment_5723_generator_swap_gemma12_ab.buggy_prefix_20260718.json.",
            "what_the_bug_had_hidden": "Two pre-fix artifacts were WRONG and are RETRACTED: (1) "
            "gemma12's apparent 'floor moved on 4 cells' were 100% stale re-reads of gemma31's "
            "engines (proven by file mtimes + byte-identical scores); the clean gemma12 FAILS to "
            "induce on 11/12 cells (induce_ok=False). (2) gemma31's apparent 'heldout 0.25 on "
            "tr87/m0r0 t1' did NOT replicate under clean attribution -- clean gemma31 is heldout "
            "0.0 on ALL 12 cells; those pre-fix nonzeros were small-N stochastic-sampling noise on "
            "a genuinely self-induced but non-reproducible engine, NOT a real capacity effect.",
        },
        "measurement_integrity": {
            "arm_held_constant": "frozen (codeonly, model-agnostic prompt) for BOTH generators; "
            "the induce LLM is the only variable. NOTE: the codeonly directive "
            "(_L2_CODEONLY_DIRECTIVE) itself begins with a '/no_think' token, so "
            "'/no_think' IS fed to every generator including Gemma -- but "
            "IDENTICALLY across generators, so it is not a differential confound "
            "(it does not bias the generator ranking).",
            "induce_ok_recorded_per_cell": True,
            "baseline_freshly_remeasured": True,
            "baseline_note": "Qwen3.5-9B-MTP re-measured in THIS invocation (not reused from the "
            "exp5720 shard) for a clean matched comparison; expected to reproduce "
            "the 0.0 heldout_accuracy floor as a sanity check.",
            "treatment_any_real_levelup": any_levelup_treat,
            "treatment_any_plan_found": any_plan_treat,
            "treatment_floor_moved_off_zero": floor_moved,
            "floor_movement_definition": "treatment produced heldout_accuracy>0 OR a real level-up on "
            "at least one cell. If False, the floor persists with the "
            "stronger generator -- a terminal result pointing AWAY from "
            "model size toward prompt/architecture/gate.",
        },
        "comparisons": comparisons,
        "per_run_rows": used,
        "methodology_note": (
            "SEEDED induce->plan->execute (the REQ-ARC-WMTE-5720 harness, arm='frozen'), with the "
            "induce LLM swapped per generator and EVERYTHING else identical: same build_progress_window "
            "seed input, same live plan_in_model planner, same WorldModelVerifier scoring, same offline "
            "execution + real level-up check. Generators run sequentially on GPU 1 (server terminated "
            "between generators so the 18GB Gemma-31B and 11.5GB Qwen never contend for the 24GB card). "
            "MTP off for the dense Gemma GGUFs (speed-only, content-neutral). Paired by GAME (metrics "
            "averaged over trials). ONLY the generator changed; the arm/prompt/planner/verifier are held."
        ),
        "recommendation_scope": (
            "This is a CONTENT test on a dev 24GB 3090, NOT a deployment decision. A positive result "
            "would recommend a stronger generator as a candidate PENDING a separate real-VRAM/latency "
            "feasibility check on the ~16GB Kaggle eval GPU. It does NOT flip the frozen live default "
            "(operator-only, and a bigger change than the reasoning/retrieval toggles)."
        ),
        "duration_s": round(duration_s, 2),
        "reproducibility_checksum": _repro_checksum(rows),
    }


def main() -> None:
    t0 = time.time()
    pre = check_preconditions()
    log(
        f"preconditions: {json.dumps({k: v for k, v in pre.items() if k != 'resolved_gguf_paths'})}"
    )
    blocking = (
        not pre.get("offline_arcade_import")
        or not pre.get("gemma31_gguf_cached")
        or not pre.get("qwen9b_gguf_cached")
        or not pre.get("llama_supports_gpu_offload")
    )
    if blocking:
        log(f"PRECONDITION FAIL: {pre}")
        for path, exp, req in (
            (GEMMA31_ARTIFACT, "experiment_5722_generator_swap_gemma31_ab", "REQ-ARC-WMTE-5722"),
            (GEMMA12_ARTIFACT, "experiment_5723_generator_swap_gemma12_ab", "REQ-ARC-WMTE-5723"),
        ):
            path.write_text(
                json.dumps(
                    {
                        "experiment": exp,
                        "requirements": [req],
                        "inference_substrate": "live_llm_inference",
                        "honest_verdict": "complete_blocked_preconditions_unmet",
                        "preconditions_checked": pre,
                        "random_seed": TRIALS[0],
                        "duration_s": round(time.time() - t0, 2),
                        "reproducibility_checksum": "sha256:"
                        + hashlib.sha256(json.dumps(pre, sort_keys=True).encode()).hexdigest(),
                    },
                    indent=2,
                )
            )
        return

    rows = run_all()

    # PRIMARY (REQ-5722): gemma31 vs the fresh qwen9b baseline.
    g31_art = build_artifact(
        exp_id="experiment_5722_generator_swap_gemma31_ab",
        req="REQ-ARC-WMTE-5722",
        treat="gemma31",
        base="qwen9b",
        question="Does swapping the stronger Gemma-4-31B-it generator into the induce call (arm held "
        "frozen/codeonly) move the induction floor vs the frozen Qwen3.5-9B-MTP baseline? "
        "(extends REQ-ARC-WMTE-5720/5721 -- prompting levers already ruled out)",
        rows=rows,
        extra_comparisons=[],
        preconditions=pre,
        duration_s=time.time() - t0,
    )
    GEMMA31_ARTIFACT.write_text(json.dumps(g31_art, indent=2))
    log(f"WROTE {GEMMA31_ARTIFACT.name}: {g31_art['honest_verdict']}")

    # SECONDARY (REQ-5723): gemma12 vs qwen9b baseline, plus the gemma31_vs_gemma12 size contrast.
    if any(r.get("arm") == "gemma12" for r in rows):
        size_contrast = [
            {
                **atp.paired_by_game(rows, "gemma31", "gemma12", metric=m),
                "contrast": "gemma31_vs_gemma12",
            }
            for m in ALL_METRICS
        ]
        g12_art = build_artifact(
            exp_id="experiment_5723_generator_swap_gemma12_ab",
            req="REQ-ARC-WMTE-5723",
            treat="gemma12",
            base="qwen9b",
            question="Does the lighter Gemma-4-12B-it generator move the induction floor vs the frozen "
            "Qwen3.5-9B-MTP baseline, and how does it compare to Gemma-4-31B-it "
            "(size-vs-capability datum)? (secondary to REQ-ARC-WMTE-5722)",
            rows=rows,
            extra_comparisons=size_contrast,
            preconditions=pre,
            duration_s=time.time() - t0,
        )
        GEMMA12_ARTIFACT.write_text(json.dumps(g12_art, indent=2))
        log(f"WROTE {GEMMA12_ARTIFACT.name}: {g12_art['honest_verdict']}")
    else:
        log("gemma12 not run (no rows); skipping REQ-5723 artifact")
    log(f"DONE total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
