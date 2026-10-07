"""REQ-VERIFY-8222: fit finite corrections without refitting authenticated base heads.

Training reads only original head_fit labels. Separate calibration rows choose
the saved depth; no reserved target can enter either operation.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import utility_kernel_8221 as kernel
from carnot.verify import utility_patch_methods_8219 as frozen

Json = dict[str, Any]
PROTOCOL = frozen.PROTOCOL_VALUE
SPECS = [
    (a, s)
    for a in PROTOCOL["static_arms"]
    for s in (PROTOCOL["random_control"]["seeds"] if a.endswith("random") else [None])
]


def role_hashes(data: Json) -> Json:
    """Hash the complete original manifests so missing sources cannot move roles."""
    return {role: canonical_hash(rows) for role, rows in data["roles"].items()}


def validate(data: Json) -> None:
    """Reject future labels before fitting, scoring or accepting a checkpoint."""
    if any(r["role"] not in ["fit", "tune"] for r in data["rows"]) or any(
        r["y"] is not None for r in data["reserved_mask"]
    ):
        raise ValueError("future_target")


def base_rows(data: Json, arm: str) -> list[Json]:
    """Original baseline probabilities define membership and accept permissions."""
    head = next(h for h in data["heads"] if h["arm"] == arm)
    lookup = {
        r["unit_id"]: role
        for role in ["head_fit", "temperature_fit", "calibration"]
        for r in data["roles"][role]
    }
    return [
        dict(
            r,
            **kernel.rule.predict(head, frozen.qualified.query(r), data["baseline"]),
            baseline_p=frozen.baseline_probability(r, data["baseline"]),
            condition=lookup[r["unit_id"]],
        )
        for r in data["rows"]
    ]


def fit(data: Json, checkpoints: Path, *, deadline_s: float = 600) -> tuple[Json, list[Json]]:
    """Durable arm completion permits a restart without extending a fitting batch."""
    validate(data)
    checkpoints.mkdir(parents=True, exist_ok=True)
    bound = canonical_hash(dict(data=data, protocol=PROTOCOL))
    deadline = time.monotonic() + min(600, deadline_s)
    models, costs = {}, []
    cached = {a: base_rows(data, a) for a in kernel.rule.ARMS}
    for index, (arm, seed) in enumerate(SPECS):
        frozen.progress("before_benchmark_fit_" + arm, index, len(SPECS) - index)
        if time.monotonic() >= deadline:
            raise TimeoutError("fit_deadline")
        key = arm + ":" + (str(seed) if seed is not None else "none")
        path = checkpoints / (key + ".json")
        if path.is_file():
            saved = json.loads(path.read_bytes())
            if saved["input_sha256"] != bound:
                raise ValueError("checkpoint_input")
        else:
            start, wall = time.monotonic_ns(), time.time_ns()
            head, mode = arm.split("_")
            training = [r for r in cached[head] if r["condition"] == "head_fit"]
            model = kernel.fit_patches(training, mode=mode, **({"seed": seed} if seed else {}))
            end = time.monotonic_ns()
            saved = dict(
                input_sha256=bound,
                model=model,
                cost=dict(
                    arm=arm,
                    seed=seed,
                    duration_s=(end - start) / 1e9,
                    started_monotonic_ns=start,
                    ended_monotonic_ns=end,
                    started_wall_ns=wall,
                    training_count=len(training),
                    available_count=sum(r["p"] is not None for r in training),
                    fitted_operations=len(model["patches"]),
                ),
            )
            atomic_json(path, saved)
        models[key] = saved["model"]
        costs.append(saved["cost"])
        frozen.progress("after_benchmark_fit_" + arm, index + 1, len(SPECS) - index - 1)
    return models, costs


def reduce(data: Json, models: Json) -> Json:
    """Depth and comparator selection use calibration costs before future evaluation."""
    validate(data)
    cached = {a: base_rows(data, a) for a in kernel.rule.ARMS}
    result: Json = dict(
        rows=[], selected_depths={}, selected_models={}, depth_grid={}, utility_residuals={}
    )
    for arm, seed in SPECS:
        key = arm + ":" + (str(seed) if seed is not None else "none")
        head = arm.split("_")[0]
        records = cached[head]
        tune = [r for r in records if r["condition"] == "calibration"]
        candidates = []
        for depth in PROTOCOL["comparator"]["depths"]:
            model = dict(models[key], patches=deepcopy(models[key]["patches"][:depth]))
            cost, brier, false_accepts = kernel.losses(
                tune, [kernel.predict(model, r) for r in tune]
            )
            candidates.append(
                dict(
                    depth=depth,
                    model=model,
                    cost=cost,
                    brier=brier,
                    false_accepts=false_accepts,
                    all_slot_denominator=len(tune),
                    brier_denominator=sum(r["p"] is not None for r in tune),
                )
            )
        chosen = min(candidates, key=lambda c: (c["cost"], c["brier"], c["depth"]))
        result["depth_grid"][key] = candidates
        result["selected_depths"][key] = chosen["depth"]
        result["selected_models"][key] = chosen["model"]
        scored = []
        for row in records:
            p = kernel.predict(chosen["model"], row)
            action = kernel.rule.action(p, row["baseline_action"])
            good, bad = kernel.energies(p) if p is not None else (None, None)
            scored.append(
                dict(
                    row,
                    arm=arm,
                    seed=seed,
                    p=p,
                    energy_good=good,
                    energy_bad=bad,
                    action=action,
                    metric="fit_tune_typed_cost",
                    denominator=1,
                    numerator=kernel.rule.base.loss(action, row["y"])
                    if row["y"] in (0, 1)
                    else None,
                    status="completed" if p is not None and row["y"] in (0, 1) else "excluded",
                    exclusion_reason=row["exclusion_reason"]
                    or ("missing_target" if row["y"] is None else "missing_features")
                    if p is None or row["y"] is None
                    else None,
                    original_status=row["status"],
                    original_exclusion_reason=row["exclusion_reason"],
                    brier=(p - row["y"]) ** 2 if p is not None and row["y"] in (0, 1) else None,
                )
            )
        result["rows"].extend(scored)
        result["utility_residuals"][key] = {
            role: frozen.residuals(
                [r for r in scored if r["condition"] == role], PROTOCOL["witness_dictionary"]
            )
            for role in ["head_fit", "temperature_fit", "calibration"]
        }
    eligible = {
        a: result["depth_grid"][a + ":none"][result["selected_depths"][a + ":none"]]
        for a in PROTOCOL["comparator"]["eligible"]
    }
    arm = min(eligible, key=lambda a: (eligible[a]["cost"], eligible[a]["brier"], a))
    result["frozen_comparator"] = dict(arm=arm, **eligible[arm], eligible=list(eligible))
    result["mandatory_group_controls"] = ["additive_group", "logistic_group"]
    return result
