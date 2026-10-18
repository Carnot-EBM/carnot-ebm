"""Offline GAP-5 cross-example consistency selector for saved ARC programs."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .arc_world_model_synth import grade_predictions


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "experiments"
if str(SCRIPTS_DIR) not in sys.path:  # pragma: no cover - pytest already injects it
    sys.path.insert(0, str(SCRIPTS_DIR))

from arc3_gap3_stage2_transition_ebm import SEED  # noqa: E402
from arc3_gap4_rule_exec_verifier import safe_transform_from_code  # noqa: E402


ARC2_POOL = REPO_ROOT / "results" / "arc3_gap4_arc2_eval_pool.json.gz"
ARC2_PROGRAMS = REPO_ROOT / "results" / "arc3_gap4_arc2_induced_programs.json"
ARC1_PROGRAMS = REPO_ROOT / "results" / "arc3_gap4_induced_programs.json"
CHAIN_ARTIFACT = REPO_ROOT / "results" / "arc3_gap4_arc2_chain_ensemble.json"
OUTPUT = REPO_ROOT / "results" / "experiment_4010_gap5_cross_example_consistency_selector.json"
INFERENCE_SUBSTRATE = "offline_saved_gap4_program_replay_cross_example_consistency"

REQUIRED_FIELDS = [
    "cross_example_precision",
    "output_agreement_precision_ref",
    "cross_example_coverage",
    "selector_beats_output_agreement",
    "sibling_abstention_gold_rate",
    "n_tasks_scored",
    "n_codex_calls",
    "missing_verifier_gaps",
    "random_seed",
    "honest_verdict",
    "duration_s",
    "inference_substrate",
]

FIELD_PRINCIPLES = {
    "cross_example_precision": (
        "BARE FLOAT -- P(gold | cross-example-consistency selected) on the ARC-2 pool "
        "(the precision side)."
    ),
    "output_agreement_precision_ref": (
        "BARE FLOAT -- the plain output-agreement precision baseline (the must-beat)."
    ),
    "cross_example_coverage": (
        "BARE FLOAT -- fraction of tasks with a confident cross-example selection "
        "(the coverage side)."
    ),
    "selector_beats_output_agreement": (
        "BARE BOOL -- does the cross-example selector beat output-agreement on precision "
        "and/or coverage with a CI excluding 0 (the verifier-improvement claim)."
    ),
    "sibling_abstention_gold_rate": (
        "BARE FLOAT -- gold rate on tasks where the selector abstains due to sibling-input "
        "disagreement (the GAP-5 tripwire datum)."
    ),
    "n_tasks_scored": "Coverage provenance: ARC-2 chain-feasible tasks with >=2 demo-perfect programs.",
    "n_codex_calls": "Coverage provenance: 0 means pure offline replay over saved programs.",
    "missing_verifier_gaps": (
        "What the cross-example selector still cannot disambiguate (the next gap per the "
        "Missing-Verifier Gap Logging mandate)."
    ),
    "random_seed": "Reproducibility seed for the paired bootstrap.",
    "honest_verdict": "Terminal-prefix verdict.",
    "duration_s": "Wall-clock seconds for the offline replay.",
    "inference_substrate": "Aggregation/offline substrate.",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_eval_pool(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def group_entries_by_task(entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        grouped.setdefault(str(entry["task"]), []).append(entry)
    return grouped


def selected_tasks_from_chain_artifact(chain_artifact: dict[str, Any]) -> list[str]:
    prereg_tasks = chain_artifact.get("preregistration", {}).get("tasks")
    if prereg_tasks:
        return [str(task) for task in prereg_tasks]
    return [str(row["task"]) for row in chain_artifact.get("per_task", []) if "task" in row]


def _arms_by_task(chain_artifact: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        str(row["task"]): list(row.get("arms", []))
        for row in chain_artifact.get("per_task", [])
        if "task" in row
    }


def _grid_hash(grid: Any) -> str | None:
    if grid is None:
        return None
    arr = np.asarray(grid, dtype=np.int64)
    payload = repr(tuple(arr.shape)).encode("ascii") + arr.tobytes()
    return hashlib.sha1(payload).hexdigest()


def _grid_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    left_arr = np.asarray(left, dtype=np.int64)
    right_arr = np.asarray(right, dtype=np.int64)
    return left_arr.shape == right_arr.shape and bool(np.array_equal(left_arr, right_arr))


def _gold_for_entry(entry: dict[str, Any]) -> np.ndarray | None:
    for candidate in entry.get("candidates", []):
        if candidate.get("correct") is True and "grid" in candidate:
            return np.asarray(candidate["grid"], dtype=np.int64)
    return None


def _call_transform(fn: Any, grid: Any) -> np.ndarray | None:
    try:
        out = fn(grid)
    except Exception:
        return None
    if out is None:
        return None
    return np.asarray(out, dtype=np.int64)


def _demo_reproduction_rate(fn: Any, demos: list[dict[str, Any]]) -> float:
    if fn is None or not demos:
        return 0.0

    def predict(grid: Any, _akey: tuple[Any, ...]) -> Any:
        return _call_transform(fn, grid)

    held_out = [(demo["input"], (), demo["output"]) for demo in demos]
    try:
        grade = grade_predictions(predict, held_out)
        return round(float(grade.get("transition_exact_rate") or 0.0), 4)
    except ValueError:
        exact = 0
        for demo in demos:
            exact += int(_grid_equal(_call_transform(fn, demo["input"]), demo["output"]))
        return round(exact / len(demos), 4)


def _compile_candidates(
    arms: list[dict[str, Any]],
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    demos = list(entries[0].get("demos", [])) if entries else []
    compiled = []
    for idx, arm in enumerate(arms):
        source = str(arm.get("source") or f"candidate{idx}")
        code = str(arm.get("code") or "")
        fn = safe_transform_from_code(code) if code else None
        predictions = [_call_transform(fn, entry["test_input"]) if fn else None for entry in entries]
        demo_rate = _demo_reproduction_rate(fn, demos)
        compiled.append(
            {
                "source": source,
                "fn": fn,
                "executable": fn is not None,
                "demo_reproduction_rate": demo_rate,
                "selectable": bool(fn is not None and demo_rate >= 1.0),
                "predictions": predictions,
            }
        )
    return compiled


def _sibling_agreement(
    candidate: dict[str, Any],
    all_candidates: list[dict[str, Any]],
    sibling_indices: list[int],
) -> float:
    others = [other for other in all_candidates if other is not candidate and other["executable"]]
    if not sibling_indices or not others:
        return 0.0
    matching = 0
    total = 0
    for input_idx in sibling_indices:
        own_pred = candidate["predictions"][input_idx]
        for other in others:
            total += 1
            matching += int(_grid_equal(own_pred, other["predictions"][input_idx]))
    return round(matching / total, 4) if total else 0.0


def _output_agreement_baseline(
    candidates: list[dict[str, Any]],
    primary_idx: int,
    target: np.ndarray | None,
) -> dict[str, Any]:
    selectable = [candidate for candidate in candidates if candidate["selectable"]]
    hashes = [
        _grid_hash(candidate["predictions"][primary_idx])
        for candidate in selectable
        if candidate["predictions"][primary_idx] is not None
    ]
    counts = Counter(hash_value for hash_value in hashes if hash_value is not None)
    if not counts:
        return {"selected": False, "selected_gold": False, "selected_hash": None, "n_matching": 0}
    top_hash, top_count = counts.most_common(1)[0]
    tied = [hash_value for hash_value, count in counts.items() if count == top_count]
    if top_count < 2 or len(tied) > 1:
        return {
            "selected": False,
            "selected_gold": False,
            "selected_hash": top_hash,
            "n_matching": int(top_count),
        }
    selected_pred = None
    for candidate in selectable:
        pred = candidate["predictions"][primary_idx]
        if _grid_hash(pred) == top_hash:
            selected_pred = pred
            break
    return {
        "selected": True,
        "selected_gold": _grid_equal(selected_pred, target),
        "selected_hash": top_hash,
        "n_matching": int(top_count),
    }


def _has_prediction_disagreement(candidates: list[dict[str, Any]], input_indices: list[int]) -> bool:
    selectable = [candidate for candidate in candidates if candidate["selectable"]]
    for input_idx in input_indices:
        hashes = {
            _grid_hash(candidate["predictions"][input_idx])
            for candidate in selectable
            if candidate["predictions"][input_idx] is not None
        }
        if len(hashes) > 1:
            return True
    return False


def score_task(
    task: str,
    entries: list[dict[str, Any]],
    arms: list[dict[str, Any]],
    primary_idx: int = 0,
) -> dict[str, Any]:
    candidates = _compile_candidates(arms, entries)
    selectable = [candidate for candidate in candidates if candidate["selectable"]]
    sibling_indices = [idx for idx in range(len(entries)) if idx != primary_idx]
    target = _gold_for_entry(entries[primary_idx]) if entries else None

    for candidate in candidates:
        sibling = _sibling_agreement(candidate, candidates, sibling_indices)
        candidate["sibling_agreement"] = sibling
        candidate["combined_score"] = round(
            (float(candidate["demo_reproduction_rate"]) + sibling) / 2.0,
            4,
        )

    baseline = _output_agreement_baseline(candidates, primary_idx, target)
    selected = None
    abstain_reason = None
    if len(selectable) < 2:
        abstain_reason = "fewer_than_two_demo_perfect_candidates"
    else:
        max_score = max(float(candidate["combined_score"]) for candidate in selectable)
        top = [
            candidate
            for candidate in selectable
            if abs(float(candidate["combined_score"]) - max_score) <= 1e-12
        ]
        if len(top) == 1:
            selected = top[0]
        elif not sibling_indices:
            abstain_reason = "no_sibling_inputs"
        elif _has_prediction_disagreement(candidates, sibling_indices):
            abstain_reason = "sibling_input_disagreement"
        else:
            abstain_reason = "score_tie"

    selected_pred = selected["predictions"][primary_idx] if selected is not None else None
    row_candidates = []
    for candidate in candidates:
        primary_pred = candidate["predictions"][primary_idx] if entries else None
        row_candidates.append(
            {
                "source": candidate["source"],
                "executable": bool(candidate["executable"]),
                "selectable": bool(candidate["selectable"]),
                "demo_reproduction_rate": float(candidate["demo_reproduction_rate"]),
                "sibling_agreement": float(candidate["sibling_agreement"]),
                "combined_score": float(candidate["combined_score"]),
                "primary_pred_hash": _grid_hash(primary_pred),
            }
        )

    return {
        "task": task,
        "n_entries": len(entries),
        "n_candidates": len(candidates),
        "n_demo_perfect_candidates": len(selectable),
        "n_sibling_inputs": len(sibling_indices),
        "candidate_scores": row_candidates,
        "cross_example_selected": selected is not None,
        "cross_example_selected_source": selected["source"] if selected is not None else None,
        "cross_example_selected_gold": _grid_equal(selected_pred, target),
        "cross_example_abstain_reason": abstain_reason,
        "output_agreement_selected": bool(baseline["selected"]),
        "output_agreement_selected_gold": bool(baseline["selected_gold"]),
        "output_agreement_n_matching": int(baseline["n_matching"]),
    }


def _rate(successes: int, total: int) -> float:
    return round(successes / total, 4) if total else 0.0


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    n = len(rows)
    cross_selected = [row for row in rows if row["cross_example_selected"]]
    output_selected = [row for row in rows if row["output_agreement_selected"]]
    return {
        "cross_precision": _rate(
            sum(int(row["cross_example_selected_gold"]) for row in cross_selected),
            len(cross_selected),
        ),
        "output_precision": _rate(
            sum(int(row["output_agreement_selected_gold"]) for row in output_selected),
            len(output_selected),
        ),
        "cross_coverage": _rate(len(cross_selected), n),
        "output_coverage": _rate(len(output_selected), n),
    }


def paired_bootstrap_ci(
    rows: list[dict[str, Any]],
    seed: int = SEED,
    bootstrap_iters: int = 5000,
) -> dict[str, Any]:
    if not rows or bootstrap_iters <= 0:
        return {
            "precision_lift": 0.0,
            "precision_lift_ci95": [0.0, 0.0],
            "coverage_lift": 0.0,
            "coverage_lift_ci95": [0.0, 0.0],
        }
    base = _metrics(rows)
    precision_diffs = []
    coverage_diffs = []
    rng = np.random.default_rng(seed)
    n = len(rows)
    for _ in range(bootstrap_iters):
        sample = [rows[int(idx)] for idx in rng.integers(0, n, size=n)]
        sampled = _metrics(sample)
        precision_diffs.append(sampled["cross_precision"] - sampled["output_precision"])
        coverage_diffs.append(sampled["cross_coverage"] - sampled["output_coverage"])
    return {
        "precision_lift": round(base["cross_precision"] - base["output_precision"], 4),
        "precision_lift_ci95": [
            round(float(np.percentile(precision_diffs, 2.5)), 4),
            round(float(np.percentile(precision_diffs, 97.5)), 4),
        ],
        "coverage_lift": round(base["cross_coverage"] - base["output_coverage"], 4),
        "coverage_lift_ci95": [
            round(float(np.percentile(coverage_diffs, 2.5)), 4),
            round(float(np.percentile(coverage_diffs, 97.5)), 4),
        ],
    }


def check_preconditions(
    pool_path: Path,
    arc2_programs_path: Path,
    arc1_programs_path: Path,
    chain_artifact_path: Path,
) -> tuple[list[dict[str, Any]], str | None]:
    preconditions: list[dict[str, Any]] = []
    saved_ok = True
    for resource, path, loader in (
        ("arc2_saved_programs", arc2_programs_path, load_json),
        ("arc1_saved_programs", arc1_programs_path, load_json),
        ("chain_candidate_sets", chain_artifact_path, load_json),
    ):
        try:
            loader(path)
            available = True
        except Exception as exc:
            available = False
            saved_ok = False
            preconditions.append({"resource": resource, "available": False, "error": str(exc)})
        if available:
            preconditions.append({"resource": resource, "available": True})
    try:
        load_eval_pool(pool_path)
        pool_ok = True
        preconditions.append({"resource": "eval_pool", "available": True})
    except Exception as exc:
        pool_ok = False
        preconditions.append({"resource": "eval_pool", "available": False, "error": str(exc)})
    if not saved_ok:
        return preconditions, "blocked_saved_programs_missing"
    if not pool_ok:
        return preconditions, "blocked_eval_pool_unreadable"
    return preconditions, None


def _missing_gaps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gaps = []
    for row in rows:
        if row["cross_example_selected"] and not row["cross_example_selected_gold"]:
            gaps.append(
                {
                    "task": row["task"],
                    "failure_mode": "cross_example_selector_selected_wrong",
                    "missing_discriminator": (
                        "rule correctness beyond demo reproduction and sibling agreement"
                    ),
                }
            )
        elif not row["cross_example_selected"]:
            reason = str(row["cross_example_abstain_reason"])
            gaps.append(
                {
                    "task": row["task"],
                    "failure_mode": f"cross_example_selector_abstained_{reason}",
                    "missing_discriminator": (
                        "higher-order rule consistency beyond demo reproduction and sibling agreement"
                    ),
                }
            )
    return gaps


def _sibling_abstention_gold_rate(rows: list[dict[str, Any]]) -> float:
    abstained = [
        row
        for row in rows
        if row["cross_example_abstain_reason"] == "sibling_input_disagreement"
    ]
    return _rate(
        sum(int(row["output_agreement_selected_gold"]) for row in abstained),
        len(abstained),
    )


def _base_artifact(honest_verdict: str, duration_s: float) -> dict[str, Any]:
    return {
        "experiment": "experiment_4010_gap5_cross_example_consistency_selector",
        "schema": "carnot.experiment_4010_gap5_cross_example_consistency_selector.v1",
        "title": "GAP-5 cross-example consistency selector over saved GAP-4 programs",
        "cross_example_precision": 0.0,
        "output_agreement_precision_ref": 0.0,
        "cross_example_coverage": 0.0,
        "output_agreement_coverage_ref": 0.0,
        "selector_beats_output_agreement": False,
        "sibling_abstention_gold_rate": 0.0,
        "n_tasks_scored": 0,
        "n_codex_calls": 0,
        "missing_verifier_gaps": [],
        "random_seed": int(SEED),
        "honest_verdict": honest_verdict,
        "duration_s": round(float(duration_s), 1),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
    }


def _verdict(beats: bool, metrics: dict[str, float], comparison: dict[str, Any]) -> str:
    if beats:
        return (
            "success: gap5_cross_example_selector_beats_agreement_"
            f"prec{metrics['cross_precision']:.4f}_cov{metrics['cross_coverage']:.4f}"
        )
    if metrics["cross_precision"] < metrics["output_precision"]:
        reason = "precision_lower"
    elif metrics["cross_coverage"] < metrics["output_coverage"]:
        reason = "coverage_lower"
    elif comparison["precision_lift_ci95"][0] <= 0 and comparison["coverage_lift_ci95"][0] <= 0:
        reason = "ci_includes_zero"
    else:
        reason = "no_scored_tasks"
    return f"complete: gap5_cross_example_no_better_than_agreement_{reason}"


def validate_artifact(artifact: dict[str, Any]) -> None:
    for field in REQUIRED_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required field: {field}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not (
        verdict.startswith("complete:")
        or verdict.startswith("success:")
        or verdict.startswith("blocked_")
    ):
        raise ValueError("honest_verdict must use a terminal prefix")
    for field in (
        "cross_example_precision",
        "output_agreement_precision_ref",
        "cross_example_coverage",
        "sibling_abstention_gold_rate",
        "duration_s",
    ):
        if not isinstance(artifact[field], float) or isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be a bare float")
    for field in ("selector_beats_output_agreement",):
        if not isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be a bare bool")
    for field in ("n_tasks_scored", "n_codex_calls", "random_seed"):
        if not isinstance(artifact[field], int) or isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be a bare int")
    if not isinstance(artifact["missing_verifier_gaps"], list):
        raise ValueError("missing_verifier_gaps must be a list")
    if not isinstance(artifact["inference_substrate"], str):
        raise ValueError("inference_substrate must be a string")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(
    pool_path: Path = ARC2_POOL,
    arc2_programs_path: Path = ARC2_PROGRAMS,
    arc1_programs_path: Path = ARC1_PROGRAMS,
    chain_artifact_path: Path = CHAIN_ARTIFACT,
    output_path: Path = OUTPUT,
    bootstrap_iters: int = 5000,
    write: bool = True,
) -> dict[str, Any]:
    started = time.time()
    preconditions, blocker = check_preconditions(
        pool_path,
        arc2_programs_path,
        arc1_programs_path,
        chain_artifact_path,
    )
    if blocker:
        artifact = _base_artifact(blocker, time.time() - started)
        artifact["preconditions_checked"] = preconditions
        validate_artifact(artifact)
        if write:
            _write_json(output_path, artifact)
        return artifact

    pool = load_eval_pool(pool_path)
    arc2_programs = load_json(arc2_programs_path)
    arc1_programs = load_json(arc1_programs_path)
    chain_artifact = load_json(chain_artifact_path)
    entries_by_task = group_entries_by_task(pool.get("entries", []))
    arms_by_task = _arms_by_task(chain_artifact)
    rows = []
    skipped = []
    for task in selected_tasks_from_chain_artifact(chain_artifact):
        entries = entries_by_task.get(task, [])
        arms = arms_by_task.get(task, [])
        if not entries or not arms:
            skipped.append({"task": task, "reason": "missing_entries_or_arms"})
            continue
        row = score_task(task, entries, arms)
        if row["n_demo_perfect_candidates"] < 2:
            skipped.append({"task": task, "reason": "fewer_than_two_demo_perfect_candidates"})
            continue
        rows.append(row)

    metrics = _metrics(rows)
    comparison = paired_bootstrap_ci(rows, seed=SEED, bootstrap_iters=bootstrap_iters)
    beats = bool(
        comparison["precision_lift_ci95"][0] > 0.0
        or comparison["coverage_lift_ci95"][0] > 0.0
    )
    verdict = _verdict(beats, metrics, comparison)
    artifact = _base_artifact(verdict, time.time() - started)
    artifact.update(
        {
            "preconditions_checked": preconditions,
            "saved_program_counts": {
                "arc2_programs": len(arc2_programs.get("programs", [])),
                "arc1_programs": len(arc1_programs.get("programs", [])),
                "chain_tasks": len(arms_by_task),
            },
            "cross_example_precision": float(metrics["cross_precision"]),
            "output_agreement_precision_ref": float(metrics["output_precision"]),
            "cross_example_coverage": float(metrics["cross_coverage"]),
            "output_agreement_coverage_ref": float(metrics["output_coverage"]),
            "selector_beats_output_agreement": beats,
            "sibling_abstention_gold_rate": _sibling_abstention_gold_rate(rows),
            "n_tasks_scored": len(rows),
            "paired_comparison": comparison,
            "missing_verifier_gaps": _missing_gaps(rows),
            "per_task": rows,
            "skipped_tasks": skipped,
        }
    )
    validate_artifact(artifact)
    if write:
        _write_json(output_path, artifact)
    return artifact
