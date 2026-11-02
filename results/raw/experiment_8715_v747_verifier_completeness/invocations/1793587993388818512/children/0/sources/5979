"""REQ-VERIFY-8225: reuse frozen patch and admission transitions on delayed feedback.

The small fitted heads operate on cached generator probabilities. Probability
mixtures keep their exact arithmetic, so a restart cannot silently alter a
clipping operation or use a label that had not arrived at the issuing clock.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
import os
import random
from pathlib import Path
import time
from typing import Any
from unittest.mock import patch

import numpy as np
from scipy.special import expit

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import calibrated_memory_methods_8180 as calibration
from carnot.verify import calibrated_memory_trajectory_8211 as durable
from carnot.verify import utility_kernel_8221 as kernel
from carnot.verify import utility_patch_methods_8219 as frozen

Json = dict[str, Any]
ARMS = ["frozen", "global_only", "global_plus_group", "local_only", "global_plus_random"]
BUDGET_S = 1200
BASE_PREDICT = kernel.predict
BASE_FIT = kernel.fit_patches
append, journal = durable.append, durable.journal


def predict(model: Json, row: Json) -> float | None:
    """Global leaves retain absolute Qwen calibration; patch trees retain clipping order."""
    if row["p"] is None:
        return None
    if model["kind"] == "global":
        p = kernel.clip(row["p"])
        return kernel.clip(
            float(expit(model["scale"] * math.log(p / (1 - p)) + model["intercept"]))
        )
    with patch.object(kernel, "predict", predict):
        return BASE_PREDICT(model, row)


def parameters(model: Json) -> Json:
    """A probability mixture has two heads; weighted parameters only initialize the next fit."""
    if model["kind"] == "input":
        return dict(scale=1.0, intercept=0.0)
    if model["kind"] == "global":
        return {key: model[key] for key in ["scale", "intercept"]}
    left = parameters(model["base"])
    if model["kind"] == "patch":
        return left
    right = parameters(model["candidate"])
    return {key: (1 - model["step"]) * left[key] + model["step"] * right[key] for key in left}


def rebase(model: Json, head: Json) -> Json:
    """Retain previously admitted deltas and mixtures while replacing absolute calibration leaves."""
    if model["kind"] in ["input", "global"]:
        return dict(
            kind="global", scale=head["scale"], intercept=head["intercept"], fit=head["fit"]
        )
    result = deepcopy(model)
    result["base"] = rebase(model["base"], head)
    if model["kind"] == "mixture":
        result["candidate"] = rebase(model["candidate"], head)
    return result


def operands(head: Json, geometry: Json, pool: list[Json]) -> tuple[Any, Any]:
    """The qualified optimizer needs just its Qwen-logit and intercept columns here."""
    offsets = [math.log(kernel.clip(r["p"]) / (1 - kernel.clip(r["p"]))) for r in pool]
    return np.column_stack((offsets, np.ones(len(pool)))), np.asarray(
        [r["y"] for r in pool], dtype=float
    )


def propose(pool: list[Json], model: Json, arm: str, seed: int) -> Json:
    """Missing rows remain patch denominators; unfinished global solves never reach admission."""
    base = model
    if arm in ["global_only", "global_plus_group", "global_plus_random"]:
        available = [r for r in pool if r["p"] is not None and r["y"] in [0, 1]]
        if not available:
            return dict(kind="patch", base=deepcopy(model), patches=[])
        initial = dict(parameters(model), centers=[], weights=[], optimizer_step=0)
        with patch.object(calibration, "operands", operands):
            head = calibration.train(initial, {}, available)
        if not head["fit"]["converged"]:
            return dict(kind="patch", base=deepcopy(model), patches=[])
        base = rebase(model, head)
    mode = dict(
        frozen="original",
        global_only="global",
        global_plus_group="group",
        local_only="local",
        global_plus_random="random",
    )[arm]
    rng = random.Random(seed)
    with (
        patch.object(kernel, "predict", predict),
        patch.object(kernel.random, "Random", return_value=rng),
    ):
        result = BASE_FIT(pool, model=base, mode=mode, seed=seed)
    result["global_fit_changed"] = base != model
    result["fit_rng_state"] = json.loads(json.dumps(rng.getstate()))
    return result


def prepare(public: list[Json], baseline: Json | None = None) -> list[Json]:
    """Stream-frozen V707 permissions use public features, independently of fitting outcomes."""
    result = []
    for row in public:
        p = None if row["values"] is None else kernel.clip(float(expit(row["values"][0])))
        fp = (
            None
            if row["values"] is None
            else p
            if baseline is None
            else frozen.baseline_probability(dict(historical_x=row["values"]), baseline)
        )
        result.append(dict(row, p=p, baseline_p=fp, baseline_action=kernel.rule.base.action(fp)))
    return result


def fixture(raw: Path) -> tuple[list[Json], Path, list[Json]]:
    """Known private targets exercise missing evidence without replacing any natural input."""
    raw.mkdir(parents=True, exist_ok=True)
    rows, labels = kernel.fixture("missing")
    path = raw / "fixture-labels.json"
    atomic_json(
        path,
        dict(
            rows=[
                dict(
                    slot=r["slot"],
                    unit_id=r["unit_id"],
                    source_cluster_id=r["source_cluster_id"],
                    y=y,
                )
                for r, y in zip(rows, labels, strict=True)
            ]
        ),
    )
    return rows, path, deepcopy(rows[:64])


class Labels(durable.ReleasedLabels):
    """Only a durable issue boundary authorizes decoding one historical evaluator fragment."""

    def __init__(self, path: Path, rows: list[Json], output: Path):
        super().__init__(path, rows, output)
        self.deliveries = journal(output)

    def __getitem__(self, index: Any) -> Any:
        value = super().__getitem__(index)
        self.deliveries = journal(self.output)
        return value


def enrich(state: Json, rows: list[Json]) -> None:
    """Save the missing mask and pending release clocks without decoding pending targets."""
    state.update(
        missing_mask=[r["p"] is None for r in rows],
        global_parameters=parameters(state["model"]),
        predicates=kernel.GROUPS,
        pending_labels=[dict(slot=i, release_slot=i + 20) for i in state["pending"]],
    )


def run(
    rows: list[Json],
    path: Path,
    seed: int,
    raw: Path,
    *,
    state: Json | None = None,
    crash_slot: int = 0,
) -> Json:
    """Run each registered arm with the qualified kernel and durable rollback checks."""
    raw.mkdir(parents=True, exist_ok=True)
    states = deepcopy(state or {})
    started = time.monotonic()
    for arm in ARMS:
        directory = raw / arm
        directory.mkdir(parents=True, exist_ok=True)
        current = states.get(arm)
        issued = journal(directory / "issued.jsonl")
        released = journal(directory / "released.jsonl")
        if issued and (
            current is None
            or [r["issued"] for r in issued] != current["issued"]
            or [r["label_slot"] for r in released] != current["consumed"]
        ):
            raise ValueError("rollback")
        labels = Labels(path, rows, directory / "released.jsonl")
        labels.state = current or {}

        def fit(pool: list[Json], **kwargs: Any) -> Json:
            # Rebuild the original-role window, including missing rows that the
            # generic kernel cannot itself include in its complete-target pool.
            window = [
                dict(rows[r["label_slot"] - 1], y=r["outcome"]["y"])
                for r in labels.deliveries
                if kernel.roles.bucket(rows[r["label_slot"] - 1])
            ][-64:]
            pool[:] = window
            before = time.process_time_ns()
            result = propose(window, kwargs["model"], arm, seed)
            labels.state["rng_state"] = result.get("fit_rng_state", labels.state["rng_state"])
            append(
                directory / "timing.jsonl",
                dict(
                    kind="update",
                    slot=labels.state["cursor"],
                    cpu_ns=time.process_time_ns() - before,
                    fit_ids=[r["slot"] for r in window],
                    missing_mask=[r["p"] is None or r["y"] is None for r in window],
                ),
            )
            return result

        def seal(kind: str, active: Json) -> None:
            if time.monotonic() - started > BUDGET_S:
                raise TimeoutError("cpu_science_budget")
            labels.state = active
            enrich(active, rows)
            append(
                directory / "issued.jsonl",
                dict(seed=seed, arm=arm, issued=active["issued"][-1], pending=active["pending"]),
            )
            for event in active["events"]:
                if event["kind"] == "commit_candidate":
                    target = directory / f"candidate-{event['slot']}.json"
                    if not target.exists():
                        atomic_json(target, event)
            if arm == "global_plus_group" and active["cursor"] == crash_slot:
                states[arm] = active
                atomic_json(raw / "crash.json", states)
                print(
                    f"[exp8225] phase=intentional_hard_exit completed={crash_slot} pending={len(active['pending'])}",
                    flush=True,
                )
                os._exit(73)

        def lookup(model: Json, row: Json) -> float | None:
            """Time only issue-time probability lookup, separate from durable file writes."""
            began = time.process_time_ns()
            p = predict(model, row)
            cpu = time.process_time_ns() - began
            if labels.state.get("phase", "issue") == "issue":
                append(
                    directory / "timing.jsonl",
                    dict(kind="issue_lookup", slot=row["slot"], cpu_ns=cpu),
                )
            return p

        before = time.process_time_ns()
        with patch.object(kernel, "predict", lookup), patch.object(kernel, "fit_patches", fit):
            final = kernel.run(rows, labels, seed, state=current, seal=seal)
        enrich(final, rows)
        states[arm] = final
        append(
            directory / "timing.jsonl",
            dict(
                kind="trajectory_lookup_and_journal",
                cpu_ns=time.process_time_ns() - before,
                issued_count=256,
            ),
        )
        atomic_json(directory / "final.json", final)
        print(
            f"[exp8225] phase=arm_complete arm={arm} completed={ARMS.index(arm) + 1} pending={len(ARMS) - ARMS.index(arm) - 1}",
            flush=True,
        )
    atomic_json(raw / "final.json", states)
    return states
