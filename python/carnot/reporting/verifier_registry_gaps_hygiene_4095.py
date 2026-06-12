"""Exp 4095 GAP-4 offline regression guard and verifier-ledger hygiene.

Spec refs: REQ-VERIFY-4095, SCENARIO-VERIFY-4095.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from carnot.reporting import verifier_registry_and_gaps_hygiene_4051 as base


REPO_ROOT = Path(__file__).resolve().parents[3]
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"

EXP4095_ARTIFACT_PATH = "results/experiment_4095_verifier_registry_gaps_hygiene.json"
REGISTRY_PATH = base.REGISTRY_PATH
GAPS_PATH = base.GAPS_PATH

ARC1_POOL_PATH = "results/arc3_gap3_stage2_eval_pool.json.gz"
ARC1_PROGRAMS_PATH = "results/arc3_gap4_induced_programs.json"
EXP4087_PATH = "results/experiment_4087_certification_precision_rescue.json"
EXP4090_GLOB = "results/experiment_4090*.json"

GAP4_VERIFIER_ID = base.GAP4_VERIFIER_ID
PRECISION_BLOCK_ID = "GAP-5-CERTIFICATION-PRECISION-RESCUE-4087"
RFT_BLOCK_ID = "GAP-RFT-A-VS-B-4090"
RFT_ROLE_ID = "verifier_as_reward_rft_4090"
PRECISION_FLOOR = 0.85

EXPECTED_ARC1 = {
    "n": 31,
    "vote_pass2": 0.4516,
    "gated_pass2": 0.5806,
    "headroom_recovered": 4,
    "vote_wins_lost": 0,
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "gap4_arc1_reproduced",
    "precision_rescue_recorded",
    "rft_outcome_recorded",
    "registry_updated",
    "gaps_updated",
    "inference_substrate",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefix summary of the cached ARC-1 replay and ledger updates.",
    "gap4_arc1_reproduced": (
        "Regression guard: the shipped GAP-4 ARC-1 gate must rederive vote 0.4516 "
        "to gated 0.5806 from cached candidates and saved program outputs."
    ),
    "precision_rescue_recorded": (
        "Bare bool; Exp 4087's certification-precision operating point is represented in ledgers."
    ),
    "rft_outcome_recorded": (
        "Bare bool; Exp 4090 A-vs-B state is represented, including absent/pending state."
    ),
    "registry_updated": "Bare bool; registry carries the Exp 4095 replay and precision point.",
    "gaps_updated": "Bare bool; gaps ledger carries the Exp 4095 precision and RFT blocks.",
    "inference_substrate": "Cached verifier candidates only; no Codex, GGUF, or live inference.",
}


def grid_hash(grid: Any) -> str:
    """Return the same stable ARC grid hash used by the GAP-4 rerank artifact."""
    arr = np.asarray(grid, dtype=np.int8)
    return hashlib.sha1(arr.shape.__repr__().encode() + arr.tobytes()).hexdigest()


def _round4(value: float) -> float:
    return round(float(value), 4)


def _load_pool(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def replay_gap4_arc1(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Replay ARC-1 vote and gated pass@2 from cached pool rows and saved outputs."""
    pool = _load_pool(repo_root / ARC1_POOL_PATH)
    programs = base._load_json(repo_root / ARC1_PROGRAMS_PATH)
    replay = replay_gap4_arc1_fixture(pool, programs)
    replay["cached_pool_path"] = ARC1_POOL_PATH
    replay["cached_programs_path"] = ARC1_PROGRAMS_PATH
    return replay


def replay_gap4_arc1_fixture(
    pool_artifact: dict[str, Any],
    induced_programs_artifact: dict[str, Any],
) -> dict[str, Any]:
    """Recompute the GAP-4 ARC-1 gate from already-loaded cached artifacts."""
    entries = list(pool_artifact.get("entries", []))
    programs_by_entry = _programs_by_entry(induced_programs_artifact)

    vote_hits: set[int] = set()
    gated_hits: set[int] = set()
    oracle_hits: set[int] = set()
    for index, entry in enumerate(entries):
        cands = list(entry.get("candidates", []))
        program = programs_by_entry.get(index)
        pred_hash = _trusted_pred_hash(program)
        if any(bool(cand.get("correct")) for cand in cands):
            oracle_hits.add(index)
        vote_ranked = sorted(cands, key=lambda cand: (-int(cand.get("votes", 0)),))
        gated_ranked = sorted(
            cands,
            key=lambda cand: (
                -_exec_match(cand, pred_hash),
                -int(cand.get("votes", 0)),
            ),
        )
        if _top2_hit(vote_ranked):
            vote_hits.add(index)
        if _top2_hit(gated_ranked):
            gated_hits.add(index)

    observed = {
        "n": len(entries),
        "vote_pass2": _round4(len(vote_hits) / max(1, len(entries))),
        "gated_pass2": _round4(len(gated_hits) / max(1, len(entries))),
        "headroom_recovered": len((gated_hits - vote_hits) & oracle_hits),
        "vote_wins_lost": len(vote_hits - gated_hits),
    }
    reproduced = observed == EXPECTED_ARC1
    return {
        "gap4_arc1_reproduced": reproduced,
        "arc1_rule_exec": observed,
        "expected": {"arc1_rule_exec": deepcopy(EXPECTED_ARC1)},
        "no_codex_calls": True,
        "no_gguf_inference": True,
    }


def _programs_by_entry(induced_programs_artifact: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    fallback_index = 0
    for program in induced_programs_artifact.get("programs", []):
        if not isinstance(program, dict):
            continue
        raw_index = program.get("entry_i", fallback_index)
        try:
            entry_index = int(raw_index)
        except (TypeError, ValueError):
            entry_index = fallback_index
        out[entry_index] = program
        fallback_index += 1
    return out


def _trusted_pred_hash(program: dict[str, Any] | None) -> str | None:
    if not program or not bool(program.get("demo_perfect")):
        return None
    pred_hash = program.get("pred_hash")
    if pred_hash:
        return str(pred_hash)
    pred_grid = program.get("pred_grid")
    return grid_hash(pred_grid) if pred_grid is not None else None


def _exec_match(candidate: dict[str, Any], pred_hash: str | None) -> int:
    if pred_hash is None:
        return 0
    return int(grid_hash(candidate.get("grid")) == pred_hash)


def _top2_hit(ranked_candidates: list[dict[str, Any]]) -> bool:
    return any(bool(cand.get("correct")) for cand in ranked_candidates[:2])


def classify_precision_rescue(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Summarize Exp 4087's best precision/recall operating point."""
    path = repo_root / EXP4087_PATH
    if not path.exists():
        return {
            "precision_rescue_recorded": "precision_rescue_pending",
            "status": "pending",
            "artifact_path": EXP4087_PATH,
            "reason": "missing_exp4087_artifact",
            "best_certified_precision": 0.0,
            "best_op_point_recall": 0.0,
            "best_operating_point": {},
            "precision_floor": PRECISION_FLOOR,
            "precision_floor_reached": False,
            "any_stack_reached_0_85": False,
            "honest_verdict": "",
        }

    artifact = base._load_json(path)
    best_precision = float(artifact.get("best_certified_precision", 0.0))
    best_recall = float(artifact.get("best_op_point_recall", 0.0))
    frontier = artifact.get("frontier", [])
    any_reached = any(
        float(point.get("precision", 0.0)) >= PRECISION_FLOOR
        for point in frontier
        if isinstance(point, dict)
    )
    floor_reached = best_precision >= PRECISION_FLOOR or any_reached
    return {
        "precision_rescue_recorded": (
            "precision_rescue_succeeded" if floor_reached else "precision_rescue_floor_not_reached"
        ),
        "status": "complete",
        "artifact_path": EXP4087_PATH,
        "honest_verdict": str(artifact.get("honest_verdict", "")),
        "best_certified_precision": _round4(best_precision),
        "best_op_point_recall": _round4(best_recall),
        "best_operating_point": dict(artifact.get("best_operating_point", {})),
        "precision_floor": PRECISION_FLOOR,
        "precision_floor_reached": floor_reached,
        "any_stack_reached_0_85": floor_reached,
        "n_tasks_scored": int(artifact.get("n_tasks_scored", 0)),
        "n_codex_calls": int(artifact.get("n_codex_calls", 0)),
        "inference_substrate": str(artifact.get("inference_substrate", "")),
    }


def classify_rft_outcome(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Summarize the Exp 4090 RFT A-vs-B artifact when it exists."""
    path = _find_exp4090_artifact(repo_root)
    if path is None:
        return {
            "rft_outcome_recorded": "rft_a_vs_b_pending_absent",
            "present": False,
            "status": "pending",
            "artifact_path": EXP4090_GLOB,
            "reason": "missing_exp4090_artifact",
            "honest_verdict": "",
            "arm_a_vs_b_delta": None,
            "arm_a_score": None,
            "arm_b_score": None,
        }

    artifact = base._load_json(path)
    verdict = str(artifact.get("honest_verdict", ""))
    status = str(artifact.get("status", "")).lower()
    if status == "blocked" or verdict.startswith("blocked_"):
        label = "rft_a_vs_b_blocked"
        normalized = "blocked"
    elif status == "complete" or verdict.startswith("complete:"):
        label = "rft_a_vs_b_complete"
        normalized = "complete"
    else:
        normalized = status or "accumulating"
        label = f"rft_a_vs_b_{normalized}"

    return {
        "rft_outcome_recorded": label,
        "present": True,
        "status": normalized,
        "artifact_path": str(path.relative_to(repo_root)),
        "honest_verdict": verdict,
        "arm_a_vs_b_delta": _first_present(
            artifact,
            "arm_a_vs_b_delta",
            "rft_a_vs_b_delta",
            "armA_vs_armB_delta",
            "arm_a_minus_b",
            "delta",
        ),
        "arm_a_score": _first_present(
            artifact,
            "arm_a_score",
            "armA_score",
            "rft_correct_score",
            "arm_a_passrate",
        ),
        "arm_b_score": _first_present(
            artifact,
            "arm_b_score",
            "armB_score",
            "rft_ablation_score",
            "arm_b_passrate",
        ),
    }


def _find_exp4090_artifact(repo_root: Path) -> Path | None:
    matches = sorted((repo_root / "results").glob("experiment_4090*.json"))
    return matches[0] if matches else None


def _first_present(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def ensure_ledgers_record_outcomes(
    registry: dict[str, Any],
    gaps_text: str,
    offline_replay: dict[str, Any],
    precision_rescue: dict[str, Any],
    rft_outcome: dict[str, Any],
) -> tuple[dict[str, Any], str, dict[str, bool]]:
    """Return registry and gaps text with Exp 4095 outcomes represented idempotently."""
    updated_registry = deepcopy(registry)
    _ensure_gap4_eval(updated_registry, offline_replay)
    _ensure_precision_operating_point(updated_registry, precision_rescue)
    _ensure_rft_role_if_present(updated_registry, rft_outcome)

    updated_gaps = base._replace_marked_block(
        gaps_text,
        "exp4095-precision-rescue",
        _precision_rescue_block(precision_rescue),
    )
    updated_gaps = base._replace_marked_block(
        updated_gaps,
        "exp4095-rft-a-vs-b",
        _rft_outcome_block(rft_outcome),
    )
    precision_recorded = _gaps_contain_precision(updated_gaps, precision_rescue)
    rft_recorded = _gaps_contain_rft(updated_gaps, rft_outcome)
    return (
        updated_registry,
        updated_gaps,
        {
            "registry_updated": _registry_contains_outcomes(updated_registry),
            "gaps_updated": precision_recorded and rft_recorded,
            "precision_rescue_recorded": precision_recorded,
            "rft_outcome_recorded": rft_recorded,
        },
    )


def _ensure_gap4_eval(registry: dict[str, Any], offline_replay: dict[str, Any]) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:
        entry = {"verifier_id": GAP4_VERIFIER_ID, "domain": "arc_agi2_grid", "eval": {}}
        registry.setdefault("verifiers", []).append(entry)
    eval_block = entry.setdefault("eval", {})
    arc1 = offline_replay.get("arc1_rule_exec", {})
    eval_block.update(
        {
            "eval_exp_4095": EXP4095_ARTIFACT_PATH,
            "exp4095_gap4_arc1_reproduced": bool(offline_replay.get("gap4_arc1_reproduced")),
            "exp4095_arc1_rule_exec_vote_pass2": arc1.get("vote_pass2"),
            "exp4095_arc1_rule_exec_gated_pass2": arc1.get("gated_pass2"),
            "exp4095_arc1_headroom_recovered": arc1.get("headroom_recovered"),
            "exp4095_arc1_vote_wins_lost": arc1.get("vote_wins_lost"),
        }
    )


def _ensure_precision_operating_point(
    registry: dict[str, Any],
    precision_rescue: dict[str, Any],
) -> None:
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:  # pragma: no cover - _ensure_gap4_eval creates this first.
        return
    point = precision_rescue.get("best_operating_point", {})
    entry["certification_precision_operating_point"] = {
        "experiment": precision_rescue.get("artifact_path", EXP4087_PATH),
        "status": precision_rescue.get("precision_rescue_recorded"),
        "best_certified_precision": precision_rescue.get("best_certified_precision"),
        "best_op_point_recall": precision_rescue.get("best_op_point_recall"),
        "filter_stack": point.get("filter_stack"),
        "threshold": point.get("threshold"),
        "n_certified": point.get("n_certified"),
        "precision_floor": precision_rescue.get("precision_floor", PRECISION_FLOOR),
        "precision_floor_reached": bool(precision_rescue.get("precision_floor_reached")),
        "eval_exp_4095": EXP4095_ARTIFACT_PATH,
    }


def _ensure_rft_role_if_present(registry: dict[str, Any], rft_outcome: dict[str, Any]) -> None:
    if not bool(rft_outcome.get("present")):
        return
    entry = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if entry is None:  # pragma: no cover - _ensure_gap4_eval creates this first.
        return
    old_roles = list(entry.get("training_time_roles", []))
    role = {
        "role_id": RFT_ROLE_ID,
        "experiment": rft_outcome.get("artifact_path"),
        "role": "rft_a_vs_b_eval",
        "status": rft_outcome.get("status"),
        "outcome": rft_outcome.get("rft_outcome_recorded"),
        "honest_verdict": rft_outcome.get("honest_verdict", ""),
    }
    entry["training_time_roles"] = [r for r in old_roles if r.get("role_id") != RFT_ROLE_ID] + [
        role
    ]


def _precision_rescue_block(outcome: dict[str, Any]) -> str:
    point = outcome.get("best_operating_point", {})
    return (
        f"### {PRECISION_BLOCK_ID}: Exp 4095 precision-rescue registry update\n"
        f"- status: {outcome['precision_rescue_recorded']}\n"
        f"- evidence: `{outcome.get('artifact_path', EXP4087_PATH)}`; "
        f"best_certified_precision={outcome.get('best_certified_precision')}; "
        f"best_op_point_recall={outcome.get('best_op_point_recall')}; "
        f"filter_stack={point.get('filter_stack')}; threshold={point.get('threshold')}; "
        f"n_certified={point.get('n_certified')}; "
        f"any_stack_reached_0_85={str(bool(outcome.get('any_stack_reached_0_85'))).lower()}.\n"
        "- failure mode: raw demo-perfect certification can carry false positives when visible demos "
        "underdetermine the hidden test transformation.\n"
        "- missing discriminator: certification-precision calibration strong enough for reward-data use.\n"
        "- candidate design: retain the Exp 4087 operating point until a stronger calibrated filter "
        "beats its precision/recall tradeoff on held-out tasks.\n"
        "- priority: high\n"
    )


def _rft_outcome_block(outcome: dict[str, Any]) -> str:
    return (
        f"### {RFT_BLOCK_ID}: Exp 4095 RFT A-vs-B outcome update\n"
        f"- status: {outcome['rft_outcome_recorded']}\n"
        f"- evidence: `{outcome.get('artifact_path', EXP4090_GLOB)}`; "
        f"present={str(bool(outcome.get('present'))).lower()}; status={outcome.get('status')}; "
        f"honest_verdict={outcome.get('honest_verdict')}; "
        f"arm_a_vs_b_delta={outcome.get('arm_a_vs_b_delta')}.\n"
        "- failure mode: verifier-as-reward RFT has not produced a decision-grade A-vs-B win unless "
        "the held-out Exp 4090 artifact exists and reports it.\n"
        "- missing discriminator: measured evidence that the verifier-certified arm beats the ablation "
        "arm under the same training/eval pipeline.\n"
        "- candidate design: run or consume Exp 4090 only after the precision-calibrated corpus and "
        "training artifacts exist; keep absent/pending state out of headline claims.\n"
        "- priority: high\n"
    )


def _registry_contains_outcomes(registry: dict[str, Any]) -> bool:
    gap4 = base._find_verifier(registry, GAP4_VERIFIER_ID)
    if gap4 is None:
        return False
    eval_ok = gap4.get("eval", {}).get("eval_exp_4095") == EXP4095_ARTIFACT_PATH
    point = gap4.get("certification_precision_operating_point", {})
    point_ok = (
        point.get("experiment") == EXP4087_PATH
        and "best_certified_precision" in point
        and "best_op_point_recall" in point
    )
    return eval_ok and point_ok


def _gaps_contain_precision(gaps_text: str, precision_rescue: dict[str, Any]) -> bool:
    return (
        PRECISION_BLOCK_ID in gaps_text
        and precision_rescue["precision_rescue_recorded"] in gaps_text
        and "any_stack_reached_0_85=" in gaps_text
    )


def _gaps_contain_rft(gaps_text: str, rft_outcome: dict[str, Any]) -> bool:
    return RFT_BLOCK_ID in gaps_text and rft_outcome["rft_outcome_recorded"] in gaps_text


def build_artifact(
    *,
    offline_replay: dict[str, Any],
    precision_rescue: dict[str, Any],
    rft_outcome: dict[str, Any],
    registry_updated: bool,
    gaps_updated: bool,
    precision_rescue_recorded: bool,
    rft_outcome_recorded: bool,
    duration_s: float,
) -> dict[str, Any]:
    """Build the Exp 4095 terminal JSON payload."""
    gap4_ok = bool(offline_replay.get("gap4_arc1_reproduced"))
    precision_ok = bool(precision_rescue_recorded)
    rft_ok = bool(rft_outcome_recorded)
    prefix = "complete:" if gap4_ok and precision_ok and rft_ok else "blocked_"
    separator = " " if prefix.endswith(":") else ""
    artifact = {
        "experiment": "experiment_4095_verifier_registry_gaps_hygiene",
        "schema": "carnot.experiment_4095_verifier_registry_gaps_hygiene.v1",
        "honest_verdict": (
            f"{prefix}{separator}gap4_arc1_reproduced_{gap4_ok}_"
            f"precision_{precision_rescue['precision_rescue_recorded']}_"
            f"rft_{rft_outcome['rft_outcome_recorded']}"
        ),
        "gap4_arc1_reproduced": gap4_ok,
        "precision_rescue_recorded": precision_ok,
        "rft_outcome_recorded": rft_ok,
        "registry_updated": bool(registry_updated),
        "gaps_updated": bool(gaps_updated),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(duration_s, 3),
        "offline_replay": offline_replay,
        "precision_rescue": precision_rescue,
        "rft_outcome": rft_outcome,
        "registry_path": REGISTRY_PATH,
        "gaps_path": GAPS_PATH,
        "field_principles": FIELD_PRINCIPLES,
        "cited_upstream_artifacts": [
            ARC1_POOL_PATH,
            ARC1_PROGRAMS_PATH,
            EXP4087_PATH,
            rft_outcome.get("artifact_path", EXP4090_GLOB),
        ],
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Validate required fields before writing the result file."""
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required artifact field: {field}")  # pragma: no cover
    if not str(artifact["honest_verdict"]).startswith(("complete:", "blocked_", "success:")):
        raise ValueError("honest_verdict must use a terminal prefix")  # pragma: no cover
    for field in (
        "gap4_arc1_reproduced",
        "precision_rescue_recorded",
        "rft_outcome_recorded",
        "registry_updated",
        "gaps_updated",
    ):
        if not isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be a bare bool")  # pragma: no cover
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError(f"inference_substrate must be {INFERENCE_SUBSTRATE}")  # pragma: no cover


def run_hygiene(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Run Exp 4095 and write the terminal artifact plus registry/gaps ledgers."""
    started = time.time()
    registry_path = repo_root / REGISTRY_PATH
    gaps_path = repo_root / GAPS_PATH

    registry = base._load_registry(registry_path)
    gaps_text = gaps_path.read_text(encoding="utf-8")
    offline_replay = replay_gap4_arc1(repo_root)
    precision_rescue = classify_precision_rescue(repo_root)
    rft_outcome = classify_rft_outcome(repo_root)

    registry, gaps_text, ledger_summary = ensure_ledgers_record_outcomes(
        registry,
        gaps_text,
        offline_replay,
        precision_rescue,
        rft_outcome,
    )
    base._write_registry(registry_path, registry)
    gaps_path.write_text(gaps_text, encoding="utf-8")

    artifact = build_artifact(
        offline_replay=offline_replay,
        precision_rescue=precision_rescue,
        rft_outcome=rft_outcome,
        registry_updated=ledger_summary["registry_updated"],
        gaps_updated=ledger_summary["gaps_updated"],
        precision_rescue_recorded=ledger_summary["precision_rescue_recorded"],
        rft_outcome_recorded=ledger_summary["rft_outcome_recorded"],
        duration_s=time.time() - started,
    )
    base._write_json(repo_root / EXP4095_ARTIFACT_PATH, artifact)
    return artifact


def main() -> None:  # pragma: no cover - exercised through the experiment command.
    artifact = run_hygiene(REPO_ROOT)
    print(f"Wrote {REPO_ROOT / EXP4095_ARTIFACT_PATH}")
    print(f"honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
