"""Experiment 5719: 3-arm matched-budget A/B of playbook injection on the STALL /
first-contact world-model induction (REQ-ARC-WMTE-5718).

ARMS (identical window, proposer, budget; only the injected prefix differs):
  - none      : include_playbook_exemplars = False   (byte-identical to production)
  - static    : include_playbook_exemplars = True    (the fixed generic block, REQ-5717)
  - retrieval : include_playbook_exemplars = <top-K patterns retrieved for THIS game's stuck
                situation> (REQ-5718 RAG: embed the query with the live GGUF, retrieve from the
                offline models/arc_playbook_index/)

The question: does RETRIEVAL beat STATIC (does the retrieval step earn its complexity over
just dumping a fixed set), and does either beat NONE? Metric is the graded per-changed-cell
reproduction recall of the induced engine on the full winning trajectory (exact-match accuracy
floors at 0 for single-shot first-contact induction, per exp5717). Reported floor+variance-
aware: with a temperature>0 proposer and small N, a single lucky induction must not decide it.

Substrate: live_llm_inference for the inductions (real GGUF generation) PLUS a bounded
live_llm_embedding_extraction step to build the per-game retrieval blocks. Honest: the proposer
is stochastic and N is small; a floored/no-signal result is a valid finding, not hidden.
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
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

JsonDict = dict[str, Any]

RANDOM_SEED = 5719
DEFAULT_ROSTER = ("lp85", "g50t", "cn04", "ft09")
TRIALS_PER_ARM = 2
RETRIEVAL_TOPK = 4
CUDA_GPU_INDEX = "1"
INFERENCE_SUBSTRATE = "live_llm_inference"
FLOOR_CELL_RECALL = 0.05

MODEL_SPECS = [
    {
        "name": "Qwen3.5-9B-MTP",
        "hf_id": "unsloth/Qwen3.5-9B-MTP-GGUF",
        "role": "induction proposer + playbook-query embedder (same GGUF, matched space)",
        "quant": "Q4_K_M",
    }
]

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal-prefixed self-declared state (Verdict Terminal-Prefix)."
    },
    "inference_substrate": {
        "principle": "live_llm_inference for inductions + a bounded live_llm_embedding_extraction "
        "step for the retrieval query; real GGUF, no fabrication."
    },
    "preconditions_checked": {
        "principle": "GGUF + built index verified before any inference (Pre-Launch Preconditions)."
    },
    "random_seed": {
        "principle": "harness determinism; the LLM sampling stays stochastic (disclosed)."
    },
    "reproducibility_checksum": {"principle": "content hash over config + rows catches drift."},
    "retrieval_vs_none_delta": {
        "principle": "treatment-minus-control mean graded cell-recall for retrieval vs no injection; "
        "directional under small N, reported with floor + outlier-fragility guards."
    },
    "retrieval_vs_static_delta": {
        "principle": "does the retrieval step earn its complexity over a fixed exemplar block; the "
        "core question of this redesign, reported honestly even when null."
    },
    "metric_floored": {
        "principle": "true when graded cell-recall is at its floor for all arms -> the offline "
        "metric cannot detect an effect (NOT evidence for or against the feature)."
    },
}


# --------------------------------------------------------------------------------------
# preconditions
# --------------------------------------------------------------------------------------
def preconditions() -> JsonDict:
    from carnot.agentic.arc_executable_world_model import _resolve_gguf, _resolve_llama_server
    from carnot.agentic.arc_playbook_retrieval import INDEX_DIR

    gguf = _resolve_gguf("Qwen3.5-9B-MTP") or _resolve_gguf("Qwen3.5-9B")
    server = _resolve_llama_server()
    index_ok = (INDEX_DIR / "index.json").exists() and (INDEX_DIR / "embeddings.npy").exists()
    return {
        "gguf_path": gguf,
        "server_path": str(server),
        "preconditions_checked": [
            {"resource": "qwen3.5_9b_gguf_cached", "available": bool(gguf)},
            {
                "resource": "llama_server_binary",
                "available": bool(server and Path(server).exists()),
            },
            {"resource": "playbook_index_built", "available": bool(index_ok)},
        ],
    }


def _first_precondition_miss(preconds: JsonDict) -> Optional[str]:
    for check in preconds["preconditions_checked"]:
        if not check["available"]:
            return str(check["resource"])
    return None


# --------------------------------------------------------------------------------------
# retrieval blocks (one bounded embedding pass per game, then the embedder is freed)
# --------------------------------------------------------------------------------------
def _query_text(game: str, full: list) -> str:
    actions = sorted({int(getattr(t, "action", 0)) for t in full})
    shape = "x".join(str(d) for d in getattr(full[0], "grid").shape) if full else "grid"
    has_click = 6 in actions
    return (
        f"ARC-AGI-3 game {game}: the agent is stuck making no level progress on a {shape} board; "
        f"observed action types {actions}"
        f"{'; uses click/coordinate actions' if has_click else '; keyboard/directional actions'}; "
        f"needs an exploration strategy to induce the world model and find the win condition."
    )


def build_retrieval_blocks(gguf_path: str, windows: dict[str, tuple]) -> dict[str, dict[str, Any]]:
    """Embed each game's stuck-situation query with the GGUF (embedding mode) and retrieve the
    top-K patterns. Returns per-game {block, pattern_ids, query_text}. The embedder is created
    once and freed here so it does not hold VRAM during the generation inductions."""
    import numpy as np
    from llama_cpp import Llama
    from llama_cpp.llama_cpp import LLAMA_POOLING_TYPE_LAST

    from carnot.agentic import arc_playbook_retrieval as rag

    index = rag.load_index()
    embedder = Llama(
        model_path=gguf_path,
        embedding=True,
        pooling_type=LLAMA_POOLING_TYPE_LAST,
        n_ctx=2048,
        n_gpu_layers=-1,
        verbose=False,
    )
    out: dict[str, dict[str, Any]] = {}
    try:
        for game, (_window, full, _cell) in windows.items():
            qtext = _query_text(game, full)
            raw = embedder.embed(qtext, normalize=False, truncate=True)
            vec = np.asarray(raw, dtype=np.float32)
            vec = vec if vec.ndim == 1 else vec.reshape(-1)
            tags = rag.infer_query_mechanic_tags(game=game)
            top = rag.retrieve(index, vec, top_k=RETRIEVAL_TOPK, query_tags=tags)
            out[game] = {
                "block": rag.format_injection(top),
                "pattern_ids": [r["pattern_id"] for r in top],
                "query_tags": list(tags),
                "query_text": qtext,
            }
    finally:
        del embedder
    return out


# --------------------------------------------------------------------------------------
# one induction arm
# --------------------------------------------------------------------------------------
def run_arm(prop, game: str, injection, full: list, window: list, cell: int) -> JsonDict:
    from carnot.agentic.arc_executable_world_model import WorldModelVerifier, load_engine

    prop.include_playbook_exemplars = injection
    t0 = time.time()
    try:
        ok, detail = prop.induce(game, window, cell)
    except Exception as exc:
        return {
            "game": game,
            "induction_ok": False,
            "error": repr(exc)[:200],
            "induce_s": round(time.time() - t0, 1),
        }
    row: JsonDict = {
        "game": game,
        "induction_ok": bool(ok),
        "induce_s": round(time.time() - t0, 1),
        "cell_recall": None,
        "reproduction_accuracy": None,
    }
    if not ok:
        row["induction_failure_detail"] = str(detail)[:200]
        return row
    try:
        engine, _is_done = load_engine(game)
        vr = WorldModelVerifier(full).score(engine)
        row["reproduction_accuracy"] = round(float(vr.accuracy), 4)
        row["cell_recall"] = round(float(getattr(vr, "cell_recall", 0.0) or 0.0), 4)
    except Exception as exc:
        row["verify_error"] = repr(exc)[:200]
    return row


# --------------------------------------------------------------------------------------
# aggregation + verdict (reuses exp5717's floor + leave-one-out fragility logic per arm-pair)
# --------------------------------------------------------------------------------------
def _arm_summary(rows: list[JsonDict], arm: str) -> JsonDict:
    a = [r for r in rows if r.get("arm") == arm]
    ok = [r for r in a if r.get("induction_ok")]
    recalls = [r["cell_recall"] for r in ok if r.get("cell_recall") is not None]
    return {
        "runs": len(a),
        "induction_ok": len(ok),
        "induction_ok_rate": round(len(ok) / len(a), 4) if a else 0.0,
        "mean_cell_recall": round(sum(recalls) / len(recalls), 4) if recalls else None,
        "max_cell_recall": round(max(recalls), 4) if recalls else None,
        "mean_induce_s": round(sum(r["induce_s"] for r in a) / len(a), 1) if a else None,
    }


def _pair_delta(rows: list[JsonDict], treat: str, base: str) -> JsonDict:
    from carnot.experiment_5717_playbook_exemplars_stall_induction_ab import _leave_one_out_fragile

    t = [
        r["cell_recall"]
        for r in rows
        if r.get("arm") == treat and r.get("induction_ok") and r.get("cell_recall") is not None
    ]
    b = [
        r["cell_recall"]
        for r in rows
        if r.get("arm") == base and r.get("induction_ok") and r.get("cell_recall") is not None
    ]
    if not t or not b:
        return {"delta": None, "outlier_fragile": True, "n_treat": len(t), "n_base": len(b)}
    delta = round(sum(t) / len(t) - sum(b) / len(b), 4)
    fragile = _leave_one_out_fragile(b, t, delta, 0.02)
    return {"delta": delta, "outlier_fragile": bool(fragile), "n_treat": len(t), "n_base": len(b)}


def _direction(pair: JsonDict) -> str:
    if pair["delta"] is None:
        return "no_scored_runs"
    if pair["outlier_fragile"]:
        return "no_reliable_signal_high_variance"
    if pair["delta"] > 0.02:
        return "improved"
    if pair["delta"] < -0.02:
        return "hurt"
    return "inconclusive"


def _verdict(summaries: dict[str, JsonDict], pairs: dict[str, JsonDict]) -> tuple[str, bool]:
    recalls = [
        s["mean_cell_recall"] for s in summaries.values() if s["mean_cell_recall"] is not None
    ]
    if not recalls:
        return "complete_retrieval_ab_no_scored_runs_inconclusive", True
    floored = max(recalls) < FLOOR_CELL_RECALL
    if floored:
        return "complete_retrieval_ab_metric_floored_inconclusive_all_arms_near_zero", True
    rvn = _direction(pairs["retrieval_vs_none"])
    rvs = _direction(pairs["retrieval_vs_static"])
    return f"complete_retrieval_vs_none_{rvn}__retrieval_vs_static_{rvs}", False


def _checksum(payload: JsonDict) -> str:
    return (
        "sha256:"
        + hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    )


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def run(roster: tuple[str, ...] = DEFAULT_ROSTER, trials: int = TRIALS_PER_ARM) -> JsonDict:
    from carnot.experiment_5717_playbook_exemplars_stall_induction_ab import build_window

    started = time.time()
    preconds = preconditions()
    base: JsonDict = {
        "experiment": "exp5719-playbook-retrieval-ab",
        "req": "REQ-ARC-WMTE-5718",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "model_specs": MODEL_SPECS,
        "random_seed": RANDOM_SEED,
        "roster": list(roster),
        "trials_per_arm": trials,
        "retrieval_topk": RETRIEVAL_TOPK,
        "field_provenance": FIELD_PRINCIPLES,
        "preconditions_checked": preconds["preconditions_checked"],
        "gguf_path": preconds["gguf_path"],
    }
    miss = _first_precondition_miss(preconds)
    if miss:
        base["honest_verdict"] = f"complete: blocked_{miss}"
        base["duration_s"] = round(time.time() - started, 3)
        base["reproducibility_checksum"] = _checksum({"blocked": miss})
        return base

    os.environ.setdefault("CARNOT_ARC_GENERATOR_CUDA_GPU", CUDA_GPU_INDEX)

    windows: dict[str, tuple] = {}
    skipped: list[str] = []
    for game in roster:
        try:
            got = build_window(game)
        except Exception as exc:
            got = None
            skipped.append(f"{game}:build_error:{repr(exc)[:80]}")
        if got is None:
            skipped.append(f"{game}:no_l1_window")
            continue
        windows[game] = got
    if not windows:
        base["honest_verdict"] = "complete_retrieval_ab_no_solvable_windows_blocked"
        base["skipped"] = skipped
        base["duration_s"] = round(time.time() - started, 3)
        base["reproducibility_checksum"] = _checksum({"skipped": skipped})
        return base

    # Build retrieval blocks first (bounded embedding pass), then free the embedder before generation.
    retrieval = build_retrieval_blocks(preconds["gguf_path"], windows)

    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    prop = LocalGGUFProposer(
        repo_substr="Qwen3.5-9B-MTP",
        model_path=preconds["gguf_path"],
        mtp=(os.environ.get("CARNOT_ARC_AB_MTP", "1") != "0"),
        kv_quant="q8_0",
        no_think_prefix="/no_think\n",
        port=int(os.environ.get("CARNOT_ARC_AB_PORT", "8932")),
        max_tokens=int(os.environ.get("CARNOT_ARC_INDUCE_MAX_TOKENS", "4096")),
        timeout=int(os.environ.get("CARNOT_ARC_INDUCE_TIMEOUT", "240")),
    )

    arms = {"none": False, "static": True}  # "retrieval" injection is per-game
    rows: list[JsonDict] = []
    for game, (window, full, cell) in windows.items():
        block = retrieval[game]["block"]
        for trial in range(trials):
            for arm_name, injection in (("none", False), ("static", True), ("retrieval", block)):
                row = run_arm(prop, game, injection, full, window, cell)
                row["arm"] = arm_name
                row["trial"] = trial
                rows.append(row)

    summaries = {arm: _arm_summary(rows, arm) for arm in ("none", "static", "retrieval")}
    pairs = {
        "retrieval_vs_none": _pair_delta(rows, "retrieval", "none"),
        "retrieval_vs_static": _pair_delta(rows, "retrieval", "static"),
        "static_vs_none": _pair_delta(rows, "static", "none"),
    }
    verdict, floored = _verdict(summaries, pairs)

    base.update(
        {
            "honest_verdict": verdict,
            "skipped": skipped,
            "n_runs": len(rows),
            "arm_summaries": summaries,
            "pair_deltas": pairs,
            "retrieval_vs_none_delta": pairs["retrieval_vs_none"]["delta"],
            "retrieval_vs_static_delta": pairs["retrieval_vs_static"]["delta"],
            "metric_floored": bool(floored),
            "retrieval_blocks": {
                g: {"pattern_ids": r["pattern_ids"], "query_tags": r["query_tags"]}
                for g, r in retrieval.items()
            },
            "rows": rows,
            "methodology_note": (
                "3 arms share the identical window, proposer config, and budget; only the injected "
                "prefix differs (none / static block / retrieved top-K). Exact-match accuracy floors "
                "at 0, so cell-recall is the discriminator; metric_floored=true means even that is at "
                "the floor for all arms (unmeasurable offline, mirrors AUTO_HUD_MASK). Pair deltas are "
                "leave-one-out-guarded: a direction the stochastic proposer's single lucky induction "
                "could flip is reported no_reliable_signal_high_variance, not improved/hurt."
            ),
            "verifier_is_oracle": False,
            "duration_s": round(time.time() - started, 3),
        }
    )
    base["reproducibility_checksum"] = _checksum(
        {"rows": rows, "roster": list(roster), "trials": trials}
    )
    return base


def main() -> None:
    roster = DEFAULT_ROSTER
    trials = TRIALS_PER_ARM
    if len(sys.argv) > 1:
        roster = tuple(sys.argv[1].split(","))
    if len(sys.argv) > 2:
        trials = int(sys.argv[2])
    result = run(roster, trials)
    out = REPO_ROOT / "results" / "experiment_5719_playbook_retrieval_ab.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"verdict: {result.get('honest_verdict')}")
    for arm, s in (result.get("arm_summaries") or {}).items():
        print(f"  {arm:10} {s}")
    print(f"pairs: {result.get('pair_deltas')}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
