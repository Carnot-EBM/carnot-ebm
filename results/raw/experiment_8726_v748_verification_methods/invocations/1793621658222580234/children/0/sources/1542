"""Exp 5197: GAP-4 scale-up continuation with a real checkpoint.

Spec refs: REQ-REPORT-5197, SCENARIO-REPORT-5197,
SCENARIO-REPORT-5197-LOCAL-GENERATOR.

This module is deliberately small and extends Exp 5177. The prior run reached
62 rows but left no resumable checkpoint. This runner restores that operational
contract first: it writes a JSON-list checkpoint before doing any broad testing,
then reports the actual reached N without rounding up to the n=180 target.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from scipy.stats import binomtest

from carnot import experiment_5177_gap4_scaleup_decentralization_tier_v474 as exp5177


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5197_gap4_scaleup_real_checkpoint_v476"
EXPERIMENT_ID = 5197
SCHEMA = "carnot.gap4_scaleup_real_checkpoint_5197.v1"
RESULT_RELATIVE_PATH = "results/experiment_5197_gap4_scaleup_real_checkpoint_v476.json"
CHECKPOINT_RELATIVE_PATH = (
    "results/experiment_5197_gap4_scaleup_real_checkpoint_v476.checkpoint.json"
)
PRIOR_RESULT_RELATIVE_PATH = exp5177.RESULT_RELATIVE_PATH
LOCAL_QWEN_CHECKPOINT_RELATIVE_PATH = (
    "results/decentralization_moe_qwen35a3b_arc1_k8.checkpoint.json"
)

TARGET_N = exp5177.TARGET_N
RANDOM_SEED = exp5177.RANDOM_SEED
LOCAL_TARGET_CALLS = 30
LOCAL_MODEL_USED = "unsloth/Qwen3.6-35B-A3B-GGUF"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
SOFT_BUDGET_ENV = "EXP5197_SOFT_BUDGET_S"
DEFAULT_SOFT_BUDGET_S = 3900.0
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_", "blocked_")
SPEC_REFS = [
    "REQ-REPORT-5197",
    "SCENARIO-REPORT-5197",
    "SCENARIO-REPORT-5197-LOCAL-GENERATOR",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "n_reached": {
        "principle": (
            "The actual number of scored rows at whatever point this task stops -- "
            "never round up to the n=180 target."
        )
    },
    "checkpoint_file_written": {
        "principle": (
            "Proves genuine resumability exists for a FUTURE continuation, closing "
            "the gap exp5177 left (a declared-but-nonexistent checkpoint path, "
            "confirmed absent this planning pass)."
        )
    },
    "exact_test_discordant_wins": {
        "principle": "Actual discordant wins from reached rows under the Exp 5177 row shape."
    },
    "exact_test_discordant_losses": {
        "principle": "Actual discordant losses from reached rows under the Exp 5177 row shape."
    },
    "exact_test_p_value_two_sided": {
        "principle": "Two-sided scipy.stats.binomtest p-value over discordant pairs."
    },
    "exact_test_passes_min6_rule": {
        "principle": (
            "The GAP-4 significance floor: at least six discordant wins, zero losses, "
            "and two-sided exact p < 0.05."
        )
    },
    "decentralization_tier_local_generator_result": {
        "principle": (
            "CLAUDE.md rule 1 requires every capability work end-to-end with "
            "locally-hosted open-weight models; a cache-existence check alone does "
            "not satisfy this."
        )
    },
    "random_seed": {
        "principle": "The Exp 5177 seed is reused so bootstrap and row ordering stay comparable."
    },
    "reproducibility_checksum": {
        "principle": "Content-addressed hash catches silent artifact or row drift."
    },
    "inference_substrate": {
        "principle": "Required substrate declaration for this replay/checkpoint scorer."
    },
    "honest_verdict": {
        "principle": (
            "Must start with complete:/complete_/success:/success_, and must not "
            "claim the significance floor was crossed unless "
            "exact_test_passes_min6_rule is actually true."
        )
    },
}

REQUIRED_FIELDS = (
    "experiment",
    "experiment_id",
    "schema",
    "spec_refs",
    "result_path",
    "target_n",
    "n_reached",
    "already_scored_prior_n",
    "new_rows_scored",
    "source_pool_rows_available",
    "source_pool_exhausted_before_new_rows",
    "checkpoint_file_written",
    "checkpoint_path",
    "exact_test_discordant_wins",
    "exact_test_discordant_losses",
    "exact_test_p_value_two_sided",
    "exact_test_passes_min6_rule",
    "cluster_bootstrap_delta_ci95",
    "decentralization_tier_local_generator_result",
    "random_seed",
    "reproducibility_checksum",
    "inference_substrate",
    "honest_verdict",
    "gap4_status_recommendation",
    "arc1_slice_result",
    "arc2_heldout_slice_result",
    "scaleup_rows",
    "remaining_rows",
    "partial",
    "source_artifacts",
    "duration_s",
    "field_principles",
)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _principled(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]["principle"]}


def _wrapped_value(value: Any) -> Any:
    if isinstance(value, Mapping) and "value" in value:
        return value.get("value")
    return value


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = json.loads(json.dumps(dict(artifact), sort_keys=True, default=str))
    checksum = payload.get("reproducibility_checksum")
    if isinstance(checksum, dict):
        checksum["value"] = ""
    else:
        payload["reproducibility_checksum"] = {"value": ""}
    return "sha256:" + hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> JsonDict:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(loaded) if isinstance(loaded, Mapping) else {}


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_soft_budget_s(env: Mapping[str, str] | None = None) -> float:
    source = os.environ if env is None else env
    raw = str(source.get(SOFT_BUDGET_ENV, "")).strip()
    if not raw:
        return DEFAULT_SOFT_BUDGET_S
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_SOFT_BUDGET_S
    return parsed if parsed > 0.0 else DEFAULT_SOFT_BUDGET_S


def _row_key(row: Mapping[str, Any]) -> str:
    return str(
        row.get("pilot_key") or f"{row.get('domain')}:{row.get('entry_i')}:{row.get('task')}"
    )


def load_prior_rows(root: Path | str = REPO_ROOT) -> list[JsonDict]:
    payload = _read_json(Path(root) / PRIOR_RESULT_RELATIVE_PATH)
    rows = payload.get("scaleup_rows")
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def load_candidate_rows(root: Path | str = REPO_ROOT) -> list[JsonDict]:
    return exp5177.load_scaleup_rows(root)


def load_checkpoint(root: Path | str) -> list[JsonDict]:
    path = Path(root) / CHECKPOINT_RELATIVE_PATH
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, Mapping)]
    rows = payload.get("rows") if isinstance(payload, Mapping) else None
    return [dict(row) for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []


def _write_checkpoint(root: Path | str, rows: Sequence[Mapping[str, Any]]) -> bool:
    path = Path(root) / CHECKPOINT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps([dict(row) for row in rows], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)
    return True


def score_new_rows_checkpointed(
    *,
    root: Path | str,
    prior_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    now: Callable[[], float] = time.time,
    soft_budget_s: float | None = None,
) -> tuple[list[JsonDict], bool, list[JsonDict], int, bool]:
    budget = resolve_soft_budget_s() if soft_budget_s is None else float(soft_budget_s)
    started = float(now())
    checkpoint_rows = load_checkpoint(root)
    done: list[JsonDict] = [dict(row) for row in prior_rows]
    done_keys = {_row_key(row) for row in done}
    for row in checkpoint_rows:
        key = _row_key(row)
        if key not in done_keys:
            done.append(dict(row))
            done_keys.add(key)

    rows = [dict(row) for row in candidate_rows][:TARGET_N]
    checkpoint_written = _write_checkpoint(root, done)
    new_rows_scored = 0
    for row in rows:
        key = _row_key(row)
        if key in done_keys:
            continue
        if float(now()) - started >= budget:
            remaining = [dict(item) for item in rows if _row_key(item) not in done_keys]
            return done, True, remaining, new_rows_scored, checkpoint_written
        done.append(row)
        done_keys.add(key)
        new_rows_scored += 1
        checkpoint_written = _write_checkpoint(root, done)

    return done, False, [], new_rows_scored, checkpoint_written


def exact_test(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    wins = sum(
        1 for row in rows if row.get("gated_top2") is True and row.get("vote_top2") is not True
    )
    losses = sum(
        1 for row in rows if row.get("vote_top2") is True and row.get("gated_top2") is not True
    )
    discordant = wins + losses
    p_value = 1.0 if discordant == 0 else float(binomtest(wins, discordant, 0.5).pvalue)
    p_value = round(p_value, 10)
    return {
        "wins": wins,
        "losses": losses,
        "ties": len(rows) - discordant,
        "p_value_two_sided": p_value,
        "passes_min6_rule": bool(wins >= 6 and losses == 0 and p_value < 0.05),
    }


def _vote_order(candidates: Sequence[Mapping[str, Any]]) -> list[int]:
    return sorted(
        range(len(candidates)),
        key=lambda idx: (-int(candidates[idx].get("votes") or 0), idx),
    )


def _pass2(candidates: Sequence[Mapping[str, Any]], order: Sequence[int]) -> bool:
    return any(candidates[idx].get("correct") is True for idx in list(order)[:2])


def _selected_candidate_index(
    candidates: Sequence[Mapping[str, Any]], pred_hash: str | None, hash_fn: Callable[[Any], str]
) -> int | None:
    if pred_hash is None:
        return None
    for idx, candidate in enumerate(candidates):
        if hash_fn(candidate.get("grid")) == pred_hash:
            return idx
    return None


def _selected_local_samples(
    checkpoint_payload: Mapping[str, Any], target_n: int = LOCAL_TARGET_CALLS
) -> list[JsonDict]:
    tasks = checkpoint_payload.get("tasks")
    if not isinstance(tasks, Mapping):
        return []
    selected: list[JsonDict] = []
    for task in sorted(tasks):
        samples = tasks.get(task)
        if not isinstance(samples, list) or not samples:
            continue
        ordered = sorted(
            [dict(sample) for sample in samples if isinstance(sample, Mapping)],
            key=lambda sample: int(sample.get("draw_index") or 0),
        )
        if ordered:
            row = dict(ordered[0])
            row["task"] = str(row.get("task") or task)
            selected.append(row)
        if len(selected) >= int(target_n):
            return selected[: int(target_n)]
    if len(selected) >= int(target_n):
        return selected[: int(target_n)]
    for task in sorted(tasks):
        samples = tasks.get(task)
        if not isinstance(samples, list):
            continue
        for sample in sorted(
            [dict(item) for item in samples if isinstance(item, Mapping)],
            key=lambda item: int(item.get("draw_index") or 0),
        )[1:]:
            row = dict(sample)
            row["task"] = str(row.get("task") or task)
            selected.append(row)
            if len(selected) >= int(target_n):
                return selected[: int(target_n)]
    return selected


def _raw_text_from_sample(sample: Mapping[str, Any]) -> str:
    code = sample.get("code")
    if not isinstance(code, str) or not code.strip():
        return ""
    return "```python\n" + code.strip() + "\n```"


def score_local_call_rows(
    *,
    root: Path | str = REPO_ROOT,
    checkpoint_rel_path: str = LOCAL_QWEN_CHECKPOINT_RELATIVE_PATH,
    target_n: int = LOCAL_TARGET_CALLS,
    prompt_loader: Callable[[Path | str], list[JsonDict]] = exp5177.load_local_prompt_entries,
    scorer: Callable[[Mapping[str, Any], str], JsonDict] = exp5177.score_generated_program,
    hash_fn: Callable[[Any], str] | None = None,
) -> JsonDict:
    checkpoint_path = Path(root) / checkpoint_rel_path
    payload = _read_json(checkpoint_path)
    samples = _selected_local_samples(payload, target_n)
    entries_by_task = {str(entry.get("task")): dict(entry) for entry in prompt_loader(root)}
    if hash_fn is None:
        _, _, hash_fn = exp5177._gap4_helpers()

    scored_rows: list[JsonDict] = []
    wins = 0
    losses = 0
    for sample in samples:
        task = str(sample.get("task"))
        entry = entries_by_task.get(task)
        if entry is None:
            scored_rows.append(
                {
                    "task": task,
                    "status": "missing_prompt_entry",
                    "vote_top2": False,
                    "local_gated_top2": False,
                }
            )
            continue
        candidates = [dict(row) for row in entry.get("candidates", []) if isinstance(row, Mapping)]
        vote_order = _vote_order(candidates)
        vote_top2 = _pass2(candidates, vote_order)
        scored = scorer(entry, _raw_text_from_sample(sample))
        match_idx = _selected_candidate_index(candidates, scored.get("pred_hash"), hash_fn)
        if scored.get("demo_perfect") is True and match_idx is not None:
            gated_order = [match_idx] + [idx for idx in vote_order if idx != match_idx]
            local_top2 = _pass2(candidates, gated_order)
        else:
            local_top2 = vote_top2
        if local_top2 and not vote_top2:
            wins += 1
        elif vote_top2 and not local_top2:
            losses += 1
        scored_rows.append(
            {
                "task": task,
                "draw_index": int(sample.get("draw_index") or 0),
                "status": scored.get("status"),
                "demo_perfect": scored.get("demo_perfect") is True,
                "pred_in_pool": scored.get("pred_in_pool") is True,
                "pred_is_pool_correct": scored.get("pred_is_pool_correct") is True,
                "vote_top2": vote_top2,
                "local_gated_top2": local_top2,
            }
        )

    return {
        "n_calls": len(samples),
        "model_used": LOCAL_MODEL_USED,
        "discordant_wins": wins,
        "discordant_losses": losses,
        "source_checkpoint_path": checkpoint_rel_path,
        "source_checkpoint_model": payload.get("local_model_used"),
        "call_selection": "first_draw_per_task_then_round_robin_to_target",
        "scored_rows": scored_rows,
    }


def describe_source_artifacts(root: Path | str = REPO_ROOT) -> list[JsonDict]:
    root_path = Path(root)
    rel_paths = [
        PRIOR_RESULT_RELATIVE_PATH,
        exp5177.ARC1_ARTIFACT_RELATIVE_PATH,
        exp5177.ARC2_ARTIFACT_RELATIVE_PATH,
        exp5177.ARC1_POOL_RELATIVE_PATH,
        exp5177.ARC2_POOL_RELATIVE_PATH,
        exp5177.ARC1_PROGRAMS_RELATIVE_PATH,
        exp5177.ARC2_PROGRAMS_RELATIVE_PATH,
        LOCAL_QWEN_CHECKPOINT_RELATIVE_PATH,
    ]
    out: list[JsonDict] = []
    for rel in rel_paths:
        path = root_path / rel
        row: JsonDict = {"path": rel, "exists": path.exists()}
        if path.exists():
            row["sha256"] = _sha256(path)
        out.append(row)
    return out


def _status_recommendation(stats: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    if stats.get("passes_min6_rule") is True:
        return "filled"
    if not rows:
        return "checkpoint_progress_only"
    return "scale_up_recommended"


def _verdict(*, n: int, crossed: bool, recommendation: str, pool_exhausted: bool) -> str:
    floor = "floor_crossed" if crossed else "floor_not_crossed"
    source = "source_pool_exhausted" if pool_exhausted else "checkpoint_progress"
    prefix = "success" if crossed else "complete"
    return f"{prefix}_gap4_scaleup_v476_n{n}_{source}_{floor}_{recommendation}"


def build_artifact(
    *,
    scaleup_rows: Sequence[Mapping[str, Any]],
    prior_n: int,
    new_rows_scored: int,
    source_pool_rows_available: int,
    checkpoint_file_written: bool,
    local_generator_result: Mapping[str, Any],
    duration_s: float,
    partial: bool,
    remaining_rows: Sequence[Mapping[str, Any]],
    source_artifacts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    rows = [dict(row) for row in scaleup_rows]
    stats = exact_test(rows)
    crossed = bool(stats["passes_min6_rule"])
    pool_exhausted_before_new = bool(
        new_rows_scored == 0 and len(rows) >= source_pool_rows_available
    )
    recommendation = _status_recommendation(stats, rows)
    local_payload = dict(local_generator_result)
    local_payload.setdefault(
        "closed_weight_cloud_generator_comparison",
        {
            "codex_first_arm_n_rows": len(rows),
            "codex_first_discordant_wins": stats["wins"],
            "codex_first_discordant_losses": stats["losses"],
            "comparison_basis": "call_level_local_rows_vs_task_level_cached_codex_first_rows",
        },
    )
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "target_n": TARGET_N,
        "n_reached": _principled("n_reached", len(rows)),
        "already_scored_prior_n": int(prior_n),
        "new_rows_scored": int(new_rows_scored),
        "source_pool_rows_available": int(source_pool_rows_available),
        "source_pool_exhausted_before_new_rows": pool_exhausted_before_new,
        "checkpoint_file_written": _principled("checkpoint_file_written", checkpoint_file_written),
        "checkpoint_path": CHECKPOINT_RELATIVE_PATH,
        "exact_test_discordant_wins": _principled("exact_test_discordant_wins", stats["wins"]),
        "exact_test_discordant_losses": _principled(
            "exact_test_discordant_losses", stats["losses"]
        ),
        "exact_test_p_value_two_sided": _principled(
            "exact_test_p_value_two_sided", stats["p_value_two_sided"]
        ),
        "exact_test_passes_min6_rule": _principled(
            "exact_test_passes_min6_rule", stats["passes_min6_rule"]
        ),
        "cluster_bootstrap_delta_ci95": exp5177.cluster_bootstrap_delta_ci(rows, seed=RANDOM_SEED),
        "decentralization_tier_local_generator_result": _principled(
            "decentralization_tier_local_generator_result", local_payload
        ),
        "random_seed": _principled("random_seed", RANDOM_SEED),
        "reproducibility_checksum": _principled("reproducibility_checksum", ""),
        "inference_substrate": _principled("inference_substrate", INFERENCE_SUBSTRATE),
        "honest_verdict": _verdict(
            n=len(rows),
            crossed=crossed,
            recommendation=recommendation,
            pool_exhausted=pool_exhausted_before_new,
        ),
        "gap4_status_recommendation": recommendation,
        "arc1_slice_result": exp5177._slice_result(
            [row for row in rows if row.get("domain") == "arc1"]
        ),
        "arc2_heldout_slice_result": exp5177._slice_result(
            [row for row in rows if row.get("domain") == "arc2"]
        ),
        "scaleup_rows": rows,
        "remaining_rows": [dict(row) for row in remaining_rows],
        "partial": bool(partial),
        "source_artifacts": [dict(row) for row in source_artifacts],
        "duration_s": max(0.0, round(float(duration_s), 6)),
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    artifact["reproducibility_checksum"]["value"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_terminal_prefix")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    n_reached = _wrapped_value(artifact.get("n_reached"))
    if not isinstance(n_reached, int) or n_reached < 0 or n_reached > TARGET_N:
        errors.append("n_reached_bounds")
    if _wrapped_value(artifact.get("checkpoint_file_written")) is not True:
        errors.append("checkpoint_file_written_true")
    wins = _wrapped_value(artifact.get("exact_test_discordant_wins"))
    losses = _wrapped_value(artifact.get("exact_test_discordant_losses"))
    p_value = _wrapped_value(artifact.get("exact_test_p_value_two_sided"))
    scaleup_rows = artifact.get("scaleup_rows")
    recomputed_stats = None
    if isinstance(scaleup_rows, list):
        recomputed_stats = exact_test(
            [dict(row) for row in scaleup_rows if isinstance(row, Mapping)]
        )
        if wins != recomputed_stats["wins"]:
            errors.append("exact_test_discordant_wins")
        if losses != recomputed_stats["losses"]:
            errors.append("exact_test_discordant_losses")
    p_value_is_numeric = not isinstance(p_value, bool) and isinstance(p_value, int | float)
    if recomputed_stats is None:
        expected_min6 = bool(
            isinstance(wins, int)
            and not isinstance(wins, bool)
            and isinstance(losses, int)
            and not isinstance(losses, bool)
            and p_value_is_numeric
            and wins >= 6
            and losses == 0
            and float(p_value) < 0.05
        )
    else:
        expected_min6 = bool(recomputed_stats["passes_min6_rule"])
    if _wrapped_value(artifact.get("exact_test_passes_min6_rule")) is not expected_min6:
        errors.append("exact_test_passes_min6_rule")
    if not p_value_is_numeric or (
        recomputed_stats is not None
        and float(p_value) != float(recomputed_stats["p_value_two_sided"])
    ):
        errors.append("exact_test_p_value_two_sided")
    local = _wrapped_value(artifact.get("decentralization_tier_local_generator_result"))
    if not isinstance(local, Mapping):
        errors.append("decentralization_tier_local_generator_result")
    else:
        for key in ("n_calls", "model_used", "discordant_wins", "discordant_losses"):
            if key not in local:
                errors.append(f"decentralization_tier_local_generator_result.{key}")
    if _wrapped_value(artifact.get("random_seed")) != RANDOM_SEED:
        errors.append("random_seed")
    if _wrapped_value(artifact.get("inference_substrate")) != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, Mapping) or checksum.get("value") != payload_checksum(artifact):
        errors.append("reproducibility_checksum")
    return sorted(dict.fromkeys(errors))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_artifact(root: Path | str, artifact: Mapping[str, Any]) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = Path(root) / RESULT_RELATIVE_PATH
    _write_json(path, artifact)
    return path


def run(
    *,
    root: Path | str = REPO_ROOT,
    prior_row_loader: Callable[[Path | str], list[JsonDict]] = load_prior_rows,
    candidate_row_loader: Callable[[Path | str], list[JsonDict]] = load_candidate_rows,
    local_result_loader: Callable[[Path | str], JsonDict] = lambda root_path: score_local_call_rows(
        root=root_path
    ),
    source_artifact_loader: Callable[[Path | str], list[JsonDict]] = describe_source_artifacts,
    now: Callable[[], float] = time.time,
) -> JsonDict:
    root_path = Path(root)
    started = float(now())
    prior_rows = prior_row_loader(root_path)
    candidate_rows = candidate_row_loader(root_path)
    rows, partial, remaining, new_rows_scored, checkpoint_written = score_new_rows_checkpointed(
        root=root_path,
        prior_rows=prior_rows,
        candidate_rows=candidate_rows,
        now=now,
        soft_budget_s=resolve_soft_budget_s(),
    )
    artifact = build_artifact(
        scaleup_rows=rows,
        prior_n=len(prior_rows),
        new_rows_scored=new_rows_scored,
        source_pool_rows_available=len(candidate_rows),
        checkpoint_file_written=checkpoint_written,
        local_generator_result=local_result_loader(root_path),
        duration_s=float(now()) - started,
        partial=partial,
        remaining_rows=remaining,
        source_artifacts=source_artifact_loader(root_path),
    )
    write_artifact(root_path, artifact)
    return artifact


def main() -> int:  # pragma: no cover
    artifact = run()
    print(f"wrote {RESULT_RELATIVE_PATH}")
    print(artifact["honest_verdict"])
    print(f"n_reached={artifact['n_reached']['value']}")
    print(f"checkpoint_file_written={artifact['checkpoint_file_written']['value']}")
    print(f"exact_test_passes_min6_rule={artifact['exact_test_passes_min6_rule']['value']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
