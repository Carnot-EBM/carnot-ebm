"""REQ-VERIFY-8144: independently check causal heads and later source losses.

The reader reuses public geometry and typed decision costs. Its transition,
scalar gradient and admission calculations do not call producer transitions.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.verify import delayed_energy_memory_8143 as upstream

Json = dict[str, Any]
ROOT = upstream.ROOT
NAME = "experiment_8144_v704_learning_audit"
CLI = f"scripts/experiments/{NAME}.py"
MODULE = "python/carnot/verify/learning_audit_8144.py"
RUNNER = "python/carnot/reporting/learning_audit_execution_8144.py"
TEST = "tests/python/test_learning_audit_8144.py"
UPSTREAM = f"results/{upstream.NAME}.json"
UPSTREAM_HASH = "sha256:424f737b8f52fa1bf96fbd4051519a4bc24dea254bdfa6fab1af63e787b4f2a4"
ARMS = upstream.engine.ARMS
MODEL_SPECS: list[Json] = []
CONFIG: Json = dict(seed=7048144, draws=10000, valid_minimum=9500, blocks=[16, 8, 32], alpha=0.025)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed real counts make a bounded CPU reader visible to its supervisor."""
    print(f"[exp8144] phase={phase} completed={completed} pending={pending}", flush=True)


def equal(expected: Any, actual: Any) -> None:
    """Independent arithmetic can differ by rounding, but never by structure."""
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise ValueError("structure_drift")
        for key in expected:
            equal(expected[key], actual[key])
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            raise ValueError("length_drift")
        for x, y in zip(expected, actual, strict=True):
            equal(x, y)
    elif isinstance(expected, float):
        if not np.isclose(expected, actual, atol=1e-12, rtol=0):
            raise ValueError("equation_drift")
    elif expected != actual:
        raise ValueError("identity_drift")


def train(head: Json, geometry: Json, pool: list[Json]) -> Json:
    """Scalar sums rebuild the four SGD steps without the producer trainer."""
    result = deepcopy(head)
    phi = upstream.engine.historical.radial.design(
        dict(centers=head["centers"], geometry=geometry), [r["values"] for r in pool]
    )
    theta = np.array([head["intercept"], *head["weights"]])
    for _ in range(4):
        errors = [
            upstream.engine.scalar_probability(
                dict(head, intercept=float(theta[0]), weights=theta[1:].tolist()),
                geometry,
                r["values"],
            )
            - r["y"]
            for r in pool
        ]
        gradient = np.array(
            [
                sum(float(phi[i, j]) * errors[i] for i in range(len(pool))) / len(pool)
                for j in range(len(theta))
            ]
        )
        gradient[1:] += 0.01 * theta[1:]
        gradient /= max(1.0, float(np.linalg.norm(gradient)))
        theta -= 0.05 * gradient
        theta[1:] = np.clip(theta[1:], -4, 4)
    result.update(
        intercept=float(theta[0]),
        weights=theta[1:].tolist(),
        optimizer_step=head["optimizer_step"] + 4,
    )
    return result


def interpolate(candidate: Json, step: float) -> Json:
    """The sealed step grid shrinks one fixed proposal without fitting again."""
    result = deepcopy(candidate["head"])
    base = candidate["base"]
    result["weights"] = [
        b + step * (w - b) for b, w in zip(base["weights"], result["weights"], strict=True)
    ]
    result["intercept"] = base["intercept"] + step * (result["intercept"] - base["intercept"])
    return result


def losses(predictions: list[float], targets: list[int]) -> tuple[float, float, int]:
    """False accepts retain their count rather than disappearing inside mean cost."""
    radial = upstream.engine.historical.radial
    actions = [radial.action(p) for p in predictions]
    return (
        sum(radial.loss(d, y) for d, y in zip(actions, targets, strict=True)) / len(targets),
        sum((p - y) ** 2 for p, y in zip(predictions, targets, strict=True)) / len(targets),
        sum(d == "accept" and y == 1 for d, y in zip(actions, targets, strict=True)),
    )


def admission(candidates: Json) -> Json:
    """Recompute the largest passing sealed step using twelve later observations."""
    records = next(iter(candidates.values()))["labels"]
    targets = [r["y"] for r in records]
    result = {}
    for arm in candidates:
        incumbent = losses([r["predictions"][arm] for r in records], targets)
        frozen = losses([r["predictions"][ARMS[0]] for r in records], targets)
        chosen = 0.0
        if min(targets.count(0), targets.count(1)) >= 2:
            for step in [1.0, 0.5, 0.25, 0.125]:
                candidate = losses([r["grid"][arm][str(step)] for r in records], targets)
                if (
                    candidate[0] <= incumbent[0]
                    and candidate[1] <= incumbent[1]
                    and candidate[0] <= frozen[0] + 0.02
                    and candidate[1] <= frozen[1] + 0.01
                    and candidate[2] <= min(incumbent[2], frozen[2])
                ):
                    chosen = step
                    break
        result[arm] = chosen
    return result


def proposal(
    heads: Json, geometry: Json, reserved: list[Json], pool: list[Json], seed: int, slot: int
) -> Json:
    """Rebuild label-blind random and frozen-error selections from released updates."""
    pool = pool[-64:]
    if len(pool) < 16 or sum(abs(r["y"] - r["frozen_prediction"]) >= 0.5 for r in pool) < 4:
        return {}
    selected = dict(
        fixed_public_center=reserved[(slot // 64 - 1) * 4 : slot // 64 * 4],
        random_past_center=random.Random(seed * 1000 + slot).sample(pool, 4),
        error_center=sorted(
            pool, key=lambda r: (-abs(r["y"] - r["frozen_prediction"]), r["source_cluster_id"])
        )[:4],
    )
    result = {}
    for arm in ARMS[1:]:
        base = deepcopy(heads[arm])
        for row in selected[arm]:
            center = (
                row
                if arm == "fixed_public_center"
                else dict(
                    source_id=row["source_cluster_id"],
                    x=((np.array(row["values"]) - geometry["mean"]) / geometry["std"]).tolist(),
                )
            )
            base["centers"].append(deepcopy(center))
            base["weights"].append(0.0)
        result[arm] = dict(base=base, head=train(base, geometry, pool), commit=slot, labels=[])
    return result


def reconstruct(state: Json, rows: list[Json], label_path: Path) -> Json:
    """Primitive events drive a second clock; no producer run function is called."""
    upstream.engine.historical.public_rows(rows, 256)
    warm = [r for r in rows[:64] if r["values"] is not None]
    initial = upstream.engine.historical.radial.initialize(
        [r["values"] for r in warm], [r["source_cluster_id"] for r in warm]
    )
    geometry = initial["geometry"]
    head = dict(centers=initial["centers"], weights=[0.0] * 16, intercept=0.0, optimizer_step=0)
    heads = {arm: deepcopy(head) for arm in ARMS}
    equal(initial["reserved"], state["reserved"])
    equal(256, len(state["issued"]))
    equal(geometry, state["geometry"])
    equal(canonical_hash(rows), state["baseline_hash"])
    equal(32, state["capacity"])
    equal(257, state["cursor"])
    equal("issue", state["phase"])
    vault = upstream.engine.historical.LabelVault(label_path, rows)
    pending, released, training, used, pool, lost = [], [], [], [], [], []
    candidates: Json = {}
    events = iter(state["events"])
    checkpoints, decisions, opened = [], [], {}
    rng_state = None

    def consume(kind: str, slot: int) -> Json:
        item = next(events, {})
        equal(kind, item.get("kind"))
        equal(slot, item["slot"])
        equal(pending, item["pending"])
        equal(canonical_hash(heads), item["state_hash"])
        return item

    for slot, row in enumerate(rows, 1):
        if slot in [64, 128, 192, 256] and candidates:
            consume("defer_candidate", slot)
            decisions.append(dict(slot=slot, disposition="deferred", steps={}))
            candidates = {}
        issued = state["issued"][slot - 1]
        equal(slot, issued["slot"])
        expected = {
            arm: None
            if row["values"] is None
            else upstream.engine.scalar_probability(h, geometry, row["values"])
            for arm, h in heads.items()
        }
        equal(expected, issued["predictions"])
        grid = {
            arm: {
                str(step): None
                if row["values"] is None
                else upstream.engine.scalar_probability(
                    interpolate(c, step), geometry, row["values"]
                )
                for step in [1.0, 0.5, 0.25, 0.125]
            }
            for arm, c in candidates.items()
        }
        equal(grid, issued["grid"])
        equal(canonical_hash(issued["predictions"]), issued["prediction_hash"])
        pending.append(slot)
        equal(True, len(pending) <= 32)
        item = consume("issue_prediction", slot)
        equal(issued["prediction_hash"], item["prediction_hash"])
        consume("durable_commit", slot)
        checkpoints.append(
            dict(
                slot=slot,
                heads_hash=canonical_hash(heads),
                pending=list(pending),
                released=list(released),
                used_admission=list(used),
            )
        )
        if slot in [64, 128, 192]:
            rebuilt = proposal(heads, geometry, initial["reserved"], pool, state["seed"], slot)
            if rebuilt:
                rng = random.Random(state["seed"] * 1000 + slot)
                rng.sample(pool[-64:], 4)
                rng_state = json.loads(json.dumps(rng.getstate()))
            item = next(events)
            equal(rebuilt, item["candidates"])
            candidates = deepcopy(item["candidates"])
            for arm, c in candidates.items():
                heads[arm] = deepcopy(c["base"])
            equal(
                dict(
                    item,
                    kind="commit_candidate",
                    slot=slot,
                    pending=pending,
                    state_hash=canonical_hash(heads),
                    disposition="committed" if candidates else "ineligible",
                ),
                item,
            )
            if candidates:
                item = consume("select_future_admission", slot)
                equal(slot + 1, item["minimum_original_slot"])
        old_slot = slot - 20
        if old_slot in pending:
            pending.remove(old_slot)
            item = consume("release_feedback", slot)
            equal(old_slot, item["label_slot"])
            released.append(old_slot)
            label = vault.release(old_slot, slot, sealed=True)
            opened[str(old_slot)] = label
            old = rows[old_slot - 1]
            if old["values"] is not None and label["y"] is not None:
                if upstream.engine.bucket(old):
                    training.append(old_slot)
                    pool.append(
                        dict(
                            old,
                            y=label["y"],
                            frozen_prediction=state["issued"][old_slot - 1]["predictions"][ARMS[0]],
                        )
                    )
                elif candidates and old_slot > next(iter(candidates.values()))["commit"]:
                    used.append(old_slot)
                    record = dict(state["issued"][old_slot - 1], y=label["y"], label_slot=old_slot)
                    for c in candidates.values():
                        c["labels"].append(record)
                    if len(next(iter(candidates.values()))["labels"]) == 12:
                        steps = admission(candidates)
                        for arm, step in steps.items():
                            if step:
                                heads[arm] = interpolate(candidates[arm], step)
                        item = next(events)
                        equal(steps, item["steps"])
                        equal(
                            [r["label_slot"] for r in next(iter(candidates.values()))["labels"]],
                            item["labels"],
                        )
                        # Verify equations first, then retain exact producer rounding for byte hashes.
                        update = next(events)
                        equal(heads, update["heads"])
                        heads = deepcopy(update["heads"])
                        for event, kind in [(item, "admit_once"), (update, "durable_update")]:
                            equal(
                                dict(
                                    event,
                                    kind=kind,
                                    slot=slot,
                                    pending=pending,
                                    state_hash=canonical_hash(heads),
                                ),
                                event,
                            )
                        decisions.append(dict(slot=slot, disposition="admitted", steps=steps))
                        candidates = {}
    equal(rng_state, state["rng_state"])
    equal(None, next(events, None))
    for key, expected in dict(
        arms=heads,
        pending=pending,
        released=released,
        consumed=released,
        training=training,
        used_admission=used,
        pool=pool,
        lost=lost,
        candidates=candidates,
    ).items():
        equal(expected, state[key])
    return dict(
        passed=True,
        seed=state["seed"],
        heads=heads,
        geometry=geometry,
        opened=opened,
        pending=pending,
        released_count=len(released),
        overflow_count=len(lost),
        admission_reuse_count=len(used) - len(set(used)),
        opportunities=decisions,
        checkpoints=checkpoints,
        final_state_hash=canonical_hash(state),
    )


def bootstrap(gains: list[float | None], length: int) -> Json:
    """Resample original-slot windows so exclusions never close temporal gaps."""
    if len(gains) < length:
        return dict(
            block_length=length,
            draws=10000,
            valid_draws=0,
            confidence=0.975,
            lower_bound=None,
            mean_gain=None,
        )
    data = np.array([np.nan if x is None else x for x in gains])
    rng = np.random.default_rng(CONFIG["seed"])
    starts = rng.integers(
        0, len(data) - length + 1, size=(CONFIG["draws"], int(np.ceil(len(data) / length)))
    )
    indices = (starts[:, :, None] + np.arange(length)).reshape(CONFIG["draws"], -1)[:, : len(data)]
    selected = data[indices]
    counts = np.isfinite(selected).sum(1)
    means = np.nansum(selected, axis=1) / np.maximum(counts, 1)
    valid = means[counts > 0]
    return dict(
        block_length=length,
        draws=CONFIG["draws"],
        valid_draws=len(valid),
        confidence=0.975,
        lower_bound=float(np.quantile(valid, 0.025)) if len(valid) else None,
        mean_gain=float(np.nanmean(data)) if np.isfinite(data).any() else None,
        method="moving original-slot blocks; seeds averaged inside source; masks retained",
    )


def statistics(rows: list[Json]) -> Json:
    """Source averages, support floors and safety gates precede any benefit claim."""
    grouped: dict[tuple[str, int], list[Json]] = {}
    for row in rows:
        grouped.setdefault((row["condition"], row["slot"]), []).append(row)
    panels: dict[str, list[Json]] = {"later_stream": [], "retention": []}
    for (condition, slot), group in sorted(grouped.items()):
        first = group[0]
        means: Json = {}
        complete = all(r["status"] == "completed" for r in group)
        for arm in ARMS:
            values = [r for r in group if r["arm"] == arm]
            means[arm] = (
                {
                    metric: float(
                        np.mean(
                            [
                                r["numerator"] / r["denominator"]
                                for r in values
                                if r["metric"] == metric
                            ]
                        )
                    )
                    for metric in ["typed_cost", "brier"]
                }
                if complete
                else {}
            )
            means[arm]["false_accepts"] = (
                float(
                    np.mean(
                        [
                            int(
                                upstream.engine.historical.radial.action(r["prediction"])
                                == "accept"
                                and r["y"] == 1
                            )
                            for r in values
                            if r["metric"] == "brier"
                        ]
                    )
                )
                if complete
                else None
            )
            means[arm]["changed_predictions"] = sum(
                abs(
                    r["prediction"]
                    - next(
                        t["prediction"]
                        for t in group
                        if t["arm"] == ARMS[0] and t["metric"] == "brier" and t["seed"] == r["seed"]
                    )
                )
                > 1e-12
                for r in values
                if r["metric"] == "brier" and complete
            )
        gain = (
            means["fixed_public_center"]["typed_cost"] - means["error_center"]["typed_cost"]
            if complete
            else None
        )
        panels[condition].append(
            dict(
                slot=slot,
                unit_id=first["unit_id"],
                source_cluster_id=first["source_cluster_id"],
                status=first["status"],
                exclusion_reason=first["exclusion_reason"],
                y=first["y"],
                arms=means,
                gain=gain,
            )
        )
    later, retention = panels["later_stream"], panels["retention"]
    completed = [r for r in later if r["gain"] is not None]
    retained = [r for r in retention if r["gain"] is not None]
    intervals = [bootstrap([r["gain"] for r in later], length) for length in CONFIG["blocks"]]
    blocks = sum(
        any(r["gain"] is not None for r in later[start : start + 16])
        for start in range(0, len(later), 16)
    )
    classes = {str(y): sum(r["y"] == y for r in completed) for y in [0, 1]}
    retention_classes = {str(y): sum(r["y"] == y for r in retained) for y in [0, 1]}

    def delta(panel: list[Json], arm: str, control: str, metric: str) -> float | None:
        return (
            float(np.mean([r["arms"][arm][metric] - r["arms"][control][metric] for r in panel]))
            if panel
            else None
        )

    comparison = dict(
        brier_increase=delta(completed, "error_center", "fixed_public_center", "brier"),
        extra_false_accepts=delta(
            completed, "error_center", "fixed_public_center", "false_accepts"
        ),
        other_control_cost_advantage=delta(
            completed, "error_center", "random_past_center", "typed_cost"
        ),
    )
    retention_checks = {
        arm: dict(
            cost_increase=delta(retained, arm, ARMS[0], "typed_cost"),
            brier_increase=delta(retained, arm, ARMS[0], "brier"),
            extra_false_accepts=delta(retained, arm, ARMS[0], "false_accepts"),
        )
        for arm in ARMS[1:]
    }
    support = len(completed) >= 128 and min(classes.values()) >= 8 and blocks >= 8
    retained_support = len(retained) >= 48 and min(retention_classes.values()) >= 8
    improved = sum(r["gain"] > 0 for r in completed)
    benefit = bool(
        support
        and intervals[0]["valid_draws"] >= 9500
        and intervals[0]["lower_bound"] > 0.02
        and improved >= 5
        and comparison["brier_increase"] <= 0.01
        and comparison["extra_false_accepts"] <= 0
        and comparison["other_control_cost_advantage"] <= 0.02
    )
    preserved = bool(
        retained_support
        and all(
            r["cost_increase"] <= 0.02
            and r["brier_increase"] <= 0.01
            and r["extra_false_accepts"] <= 0
            for r in retention_checks.values()
        )
    )
    return dict(
        per_source_results=later,
        retention_rows=retention,
        paired_gain_interval=intervals[0],
        sensitivity_intervals=intervals[1:],
        completed_count=len(completed),
        class_counts=classes,
        nonoverlapping_blocks=blocks,
        support_sufficient=support,
        retention_support_sufficient=retained_support,
        retention_class_counts=retention_classes,
        improved_sources=improved,
        matched_comparison=comparison,
        retention_checks=retention_checks,
        h2_passed=benefit,
        retention_passed=preserved,
        available_typed_cost_headroom=float(
            np.mean([r["arms"]["fixed_public_center"]["typed_cost"] for r in completed])
        )
        if completed
        else None,
        changed_predictions={
            arm: sum(r["arms"][arm]["changed_predictions"] for r in later + retention)
            for arm in ARMS
        },
    )


def control_rows(*, improved: bool = True) -> list[Json]:
    """A known prediction change must be visible to the exact natural reducer."""
    rows = []
    for condition, slots in [("later_stream", range(65, 257)), ("retention", range(1, 65))]:
        for slot in slots:
            y = slot % 2
            public = dict(
                slot=slot,
                unit_id=f"{condition}-{slot}",
                source_cluster_id=f"{condition}-{slot}",
                exclusion_reason=None,
            )
            predictions = {
                arm: (0.99 if y else 0.01)
                if condition == "retention" or arm == "error_center" and improved
                else 0.4
                for arm in ARMS
            }
            prediction = dict(predictions=predictions, prediction_hash=canonical_hash(predictions))
            for seed in [101, 102]:
                rows.extend(
                    upstream.engine.historical.scored(
                        public, prediction, dict(y=y), condition, seed
                    )
                )
    return rows


def positive_control() -> Json:
    """The private oracle contrast qualifies detection, without natural benefit credit."""
    positive, negative = statistics(control_rows()), statistics(control_rows(improved=False))
    return dict(
        passed=positive["h2_passed"] and positive["retention_passed"] and not negative["h2_passed"],
        verdict_class="circular_positive",
        verifier_is_oracle=True,
        gain_lower_bound=positive["paired_gain_interval"]["lower_bound"],
        negative_h2_passed=negative["h2_passed"],
    )


def reference(path: Path) -> Json:
    """Every saved primitive is bound to its exact byte identity."""
    return dict(path=str(path), sha256=sha256_file(path))


def measure(root: Path, raw: Path, *, fixture: bool = False) -> Json:
    """Authenticate inputs and seal independent predictions before retained targets."""
    started = time.monotonic()
    raw.mkdir(parents=True, exist_ok=True)
    binder = upstream.engine.methods.Custody(raw / "inputs")
    binder.upstream = "exp8143-delayed-energy-memory"
    work: Json = dict(
        input_ready=0,
        fixture_mode=fixture,
        rows=[],
        causal_order_checks=[],
        restart_parity_rows=[],
        source_artifact_hashes=[],
        raw_shard_hashes=[],
        cited_upstream_artifacts=[],
        phase_spans=[],
    )
    progress("before_input_custody")
    try:
        path = root / UPSTREAM
        value = binder.read(path, None if fixture else UPSTREAM_HASH)
        for key, expected in [
            ("experiment_id", 8143),
            ("learning_trajectory_ready_score", 1),
            ("required_checks_passed", True),
            ("flagged_adversarial", False),
        ]:
            binder.require(path, key, expected, value.get(key))
        if not fixture:
            terminal = upstream.engine.methods.historical.terminal(path, value, binder)
            binder.require(path, "terminal.report.passed", True, terminal["report"].get("passed"))
            binder.require(
                path,
                "normal_process_exit",
                True,
                binder.read(Path(value["terminal_validation_sidecar_path"])).get(
                    "normal_process_exit"
                ),
            )
            binder.require(
                path,
                "numerical_protocol",
                dict(
                    delayed_memory=upstream.engine.protocol(),
                    statistical_plan=upstream.engine.methods.protocol()["statistical_plan"],
                ),
                value["numerical_protocol"],
            )
        manifests = value["input_manifests"]
        public = {
            role: binder.read(
                Path(manifests[role + "_feature_manifest"]["path"]),
                manifests[role + "_feature_manifest"]["sha256"],
            )["rows"]
            for role in ["stream", "retention"]
        }
        states = [
            binder.read(Path(entry["state"]["path"]), entry["state"]["sha256"])
            for entry in value["state_manifest"]
        ]
        binder.require(
            path,
            "seeds",
            [101, 102] if fixture else list(range(101, 121)),
            [s["seed"] for s in states],
        )
        for ref in manifests["evaluator_label_manifests"].values():
            binder.bind(Path(ref["path"]), ref["sha256"])
        work.update(
            input_ready=1,
            input_manifests=manifests,
            state_manifest=value["state_manifest"],
            upstream_primary=reference(path),
        )
        work["cited_upstream_artifacts"] = [
            dict(
                reference(path),
                fields_imported=["state_manifest", "input_manifests"],
                scope="historical_model_receipts",
                historical_model_provenance=value.get("cited_upstream_artifacts", []),
            )
        ]
    except (OSError, ValueError, KeyError, TypeError) as error:
        if not binder.failures:
            binder.failures.append(
                dict(
                    check="authenticated_primitives",
                    upstream="exp8143",
                    path=str(root / UPSTREAM),
                    hash=None,
                    artifact_field="authenticated_primitives",
                    op="==",
                    expected=True,
                    observed=str(error),
                    passed=False,
                )
            )
    work["gate_check_summary"] = binder.checks + [
        r for r in binder.failures if r not in binder.checks
    ]
    work["source_artifact_hashes"] = binder.refs
    work["phase_spans"].append(dict(phase="custody", duration_s=time.monotonic() - started))
    progress("after_input_custody", work["input_ready"])
    work["positive_control"] = positive_control()
    try:
        if work["input_ready"]:
            progress("before_independent_head_benchmark", 0, len(states))
            retained_predictions, head_rows = [], []
            stream_labels = Path(manifests["evaluator_label_manifests"]["stream"]["path"])
            for index, state in enumerate(states):
                reconstructed = reconstruct(state, public["stream"], stream_labels)
                head_rows.append(
                    dict(
                        seed=state["seed"],
                        heads=reconstructed["heads"],
                        geometry=reconstructed["geometry"],
                    )
                )
                predictions = []
                for row in public["retention"]:
                    pred = {
                        arm: None
                        if row["values"] is None
                        else upstream.engine.scalar_probability(
                            head, reconstructed["geometry"], row["values"]
                        )
                        for arm, head in reconstructed["heads"].items()
                    }
                    predictions.append(
                        dict(
                            slot=row["slot"], predictions=pred, prediction_hash=canonical_hash(pred)
                        )
                    )
                retained_predictions.append(dict(seed=state["seed"], rows=predictions))
                work["rows"].extend(
                    upstream.score_stream(
                        public["stream"], state, reconstructed["opened"], state["seed"]
                    )
                )
                work["causal_order_checks"].append(
                    {
                        k: reconstructed[k]
                        for k in [
                            "passed",
                            "seed",
                            "pending",
                            "released_count",
                            "overflow_count",
                            "admission_reuse_count",
                            "opportunities",
                            "final_state_hash",
                        ]
                    }
                )
                work.setdefault("head_diagnostics", []).append(
                    dict(seed=state["seed"], arms=upstream.head_diagnostics(state))
                )
                work["restart_parity_rows"].append(
                    dict(
                        seed=state["seed"],
                        condition="independent_full_reconstruction",
                        passed=True,
                        head_hash=canonical_hash(reconstructed["heads"]),
                    )
                )
                if index == 0 and not fixture:
                    invocation = Path(value["state_manifest"][0]["state"]["path"]).parents[1]
                    crash = binder.read(invocation / "crash/crash_checkpoint.json")
                    resumed = binder.read(invocation / "resume/final_state.json")
                    equal(state, resumed)
                    equal(state["events"][: len(crash["events"])], crash["events"])
                    boundary = reconstructed["checkpoints"][63]
                    equal(boundary["heads_hash"], canonical_hash(crash["arms"]))
                    for key in ["pending", "released", "used_admission"]:
                        equal(boundary[key], crash[key])
                    work["restart_parity_rows"].append(
                        dict(
                            seed=state["seed"],
                            condition="historical_abrupt_restart_independent_prefix",
                            passed=True,
                            crash=reference(invocation / "crash/crash_checkpoint.json"),
                            resumed=reference(invocation / "resume/final_state.json"),
                        )
                    )
                progress("independent_seed_complete", index + 1, len(states) - index - 1)
            atomic_json(
                raw / "independent_retention_predictions.json", dict(rows=retained_predictions)
            )
            atomic_json(raw / "independent_final_heads.json", dict(rows=head_rows))
            work["retention_prediction_manifest"] = reference(
                raw / "independent_retention_predictions.json"
            )
            work["final_head_manifest"] = reference(raw / "independent_final_heads.json")
            progress("independent_heads_and_retention_sealed", len(states))
            if not fixture:
                sealed = binder.read(
                    Path(value["retention_prediction_manifest"]["path"]),
                    value["retention_prediction_manifest"]["sha256"],
                )
                for expected, actual in zip(retained_predictions, sealed["rows"], strict=True):
                    equal(expected["seed"], actual["seed"])
                    for x, y in zip(expected["rows"], actual["rows"], strict=True):
                        equal(x["predictions"], y["predictions"])
            retention_vault = upstream.engine.historical.LabelVault(
                Path(manifests["evaluator_label_manifests"]["retention"]["path"]),
                public["retention"],
            )
            progress("before_reserved_label_scoring", 0, len(public["retention"]))
            for group in retained_predictions:
                for row, pred in zip(public["retention"], group["rows"], strict=True):
                    label = retention_vault.release(row["slot"], 256, sealed=True, retention=True)
                    work["rows"].extend(
                        upstream.engine.historical.scored(
                            row, pred, label, "retention", group["seed"]
                        )
                    )
            work["original_slot_mask"] = {
                role: [r["values"] is not None for r in panel] for role, panel in public.items()
            }
            progress("after_reserved_label_scoring", len(public["retention"]))
            progress("after_independent_head_benchmark", len(states))
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        work["owned_reconstruction_error"] = type(error).__name__ + ": " + str(error)
        progress("owned_reconstruction_failed", len(work["causal_order_checks"]))
    work["source_artifact_hashes"] = binder.refs
    work["opportunity_summary"] = {
        arm: dict(
            accepted=sum(
                d["steps"].get(arm, 0) > 0
                for r in work["causal_order_checks"]
                for d in r["opportunities"]
            ),
            rejected=sum(
                d["disposition"] == "admitted" and d["steps"].get(arm, 0) == 0
                for r in work["causal_order_checks"]
                for d in r["opportunities"]
            ),
            deferred=sum(
                d["disposition"] == "deferred"
                for r in work["causal_order_checks"]
                for d in r["opportunities"]
            ),
        )
        for arm in ARMS[1:]
    }
    work["audit_statistics"] = statistics(work["rows"])
    work["changed_source_predictions"] = {
        condition: {
            arm: sum(r["arms"][arm]["changed_predictions"] > 0 for r in panel) for arm in ARMS
        }
        for condition, panel in [
            ("later_stream", work["audit_statistics"]["per_source_results"]),
            ("retention", work["audit_statistics"]["retention_rows"]),
        ]
    }
    work["reductions"] = upstream.engine.historical.reductions(work["rows"])
    atomic_json(raw / "primitive_rows.json", dict(rows=work["rows"]))
    atomic_json(
        raw / "causal_transcript.json",
        dict(rows=work["causal_order_checks"], restart=work["restart_parity_rows"]),
    )
    work["code_config_hashes"] = {
        p: sha256_file(ROOT / p)
        for p in [
            MODULE,
            RUNNER,
            CLI,
            TEST,
            upstream.MODULE,
            upstream.engine.MODULE,
            "python/carnot/verify/radial_memory_8085.py",
            "python/carnot/verify/independent_online_memory_8116.py",
            "openspec/change-proposals/v702-methods-and-stream-protocol.md",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
        ]
    }
    work["config"] = CONFIG
    work["raw_shard_hashes"] = [reference(p) for p in sorted(raw.glob("*.json"))]
    work["duration_s"] = time.monotonic() - started
    work["phase_spans"].append(
        dict(
            phase="independent_reconstruction_and_scoring",
            duration_s=work["duration_s"] - work["phase_spans"][0]["duration_s"],
        )
    )
    atomic_json(raw / "measurement.json", work)
    progress("measurement_normal_exit", len(work["causal_order_checks"]))
    return work


def build(work: Json, raw: Path, receipts: list[Json]) -> Json:
    """Reconstruction readiness and supported development benefit are separate."""
    stats = work["audit_statistics"]
    owned = (
        bool(receipts)
        and all(r["passed"] for r in receipts)
        and work["positive_control"]["passed"]
        and "owned_reconstruction_error" not in work
    )
    ready = int(
        owned
        and work["input_ready"]
        and all(r["passed"] for r in work["causal_order_checks"] + work["restart_parity_rows"])
    )
    signal = int(
        ready and stats["h2_passed"] and stats["retention_passed"] and not work["fixture_mode"]
    )
    verdict = (
        "disqualified"
        if not owned
        else "blocked"
        if not work["input_ready"]
        else "circular_positive"
        if work["fixture_mode"]
        else "positive"
        if signal
        else "null"
    )
    reason = (
        "owned_validation"
        if not owned
        else next(
            (r["check"] for r in work["gate_check_summary"] if not r["passed"]),
            "supported_development_benefit"
            if signal
            else "insufficient_support"
            if not stats["support_sufficient"]
            else "no_later_typed_cost_benefit",
        )
    )
    counts = {
        status: sum(r["status"] == status for r in stats["per_source_results"])
        for status in ["completed", "excluded", "censored"]
    }
    value = dict(
        work,
        **stats,
        experiment_id=8144,
        task_id="exp8144-learning-audit",
        milestone="2026.10.704",
        honest_verdict=f"complete_{verdict}_{reason}",
        verdict_class=verdict,
        learning_audit_ready_score=ready,
        h2_development_signal_score=signal,
        verifier_is_oracle=work["fixture_mode"],
        required_checks_passed=owned,
        flagged_adversarial=False,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=MODEL_SPECS,
        model_invocation_counts=dict(upstream.engine.historical.ZERO_INVOCATION_COUNTS),
        call_ledger=[],
        trained_head_specs=[
            dict(
                kind="reconstructed_small_Gaussian_residual",
                arms=ARMS[1:],
                optimizer="four_clipped_SGD_steps",
                scope="imported_trajectory_no_new_deployed_training",
            )
        ]
        if ready
        else [],
        claim_scope="Independent within-run exposed-development later loss and retention audit; no lifelong safety or convex regret claim",
        exposure_scope="private_circular_fixture"
        if work["fixture_mode"]
        else "exposed_development_within_run_disjoint",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        intended_count=192,
        eligible_count=counts["completed"],
        independent_count=counts["completed"] if not work["fixture_mode"] else 0,
        excluded_count=counts["excluded"],
        censored_count=counts["censored"],
        failed_count=int("owned_reconstruction_error" in work),
        sample_size_budget=dict(
            later_original_slots=192,
            retention_original_slots=64,
            seeds_nested_inside_sources=20,
            repeats_add_independent_sources=0,
        ),
        run_date="20261005",
        random_seed=CONFIG["seed"],
        acceptance_gates=upstream.engine.methods.protocol()["statistical_plan"],
        methodology_note="Independent scalar heads and admission reconstruction precede reserved label decoding. Source averages preserve original-slot masks. Private oracle controls are circular. A changing head alone establishes no benefit.",
    )
    value["field_principles"] = {
        k: "Exact byte-bound evidence; exposed development supplies zero external generalization credit."
        for k in value
    }
    value["field_principles"].update(
        learning_audit_ready_score="Independent reconstruction and owned normal checks; a supported null may be ready.",
        h2_development_signal_score="Source-level later cost benefit and retention; distinct from state change and reconstruction.",
        verdict_class="Unchanged external blocks terminate once; owned failures disqualify; partial is unfinished owned work only.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Cold readers rehash custody and recompute equations rather than trust summaries."""
    try:
        value = json.loads(path.read_text())
        checksum = value.pop("reproducibility_checksum")
        equal(checksum, canonical_hash(value))
        for relative, digest in value["code_config_hashes"].items():
            equal(digest, sha256_file(ROOT / relative))
        for ref in value["raw_shard_hashes"] + value["source_artifact_hashes"]:
            equal(ref["sha256"], sha256_file(Path(ref["path"])))
            if "snapshot_path" in ref:
                equal(ref["sha256"], sha256_file(Path(ref["snapshot_path"])))
        equal(value["audit_statistics"], statistics(value["rows"]))
        equal(value["reductions"], upstream.engine.historical.reductions(value["rows"]))
        if "owned_reconstruction_error" in value:
            return (
                value["verdict_class"] == "disqualified"
                and value["learning_audit_ready_score"] == 0
                and value["required_checks_passed"] is False
            )
        recomputed = []
        if value["input_ready"]:
            manifests = value["input_manifests"]
            public = {
                role: json.loads(Path(manifests[role + "_feature_manifest"]["path"]).read_text())[
                    "rows"
                ]
                for role in ["stream", "retention"]
            }
            sealed = json.loads(Path(value["retention_prediction_manifest"]["path"]).read_text())[
                "rows"
            ]
            head_rows = json.loads(Path(value["final_head_manifest"]["path"]).read_text())["rows"]
            for index, entry in enumerate(value["state_manifest"]):
                state = json.loads(Path(entry["state"]["path"]).read_text())
                reconstructed = reconstruct(
                    state,
                    public["stream"],
                    Path(manifests["evaluator_label_manifests"]["stream"]["path"]),
                )
                equal(
                    head_rows[index],
                    dict(
                        seed=state["seed"],
                        heads=reconstructed["heads"],
                        geometry=reconstructed["geometry"],
                    ),
                )
                recomputed.extend(
                    upstream.score_stream(
                        public["stream"], state, reconstructed["opened"], state["seed"]
                    )
                )
                for row, pred in zip(public["retention"], sealed[index]["rows"], strict=True):
                    expected = {
                        arm: None
                        if row["values"] is None
                        else upstream.engine.scalar_probability(
                            h, reconstructed["geometry"], row["values"]
                        )
                        for arm, h in reconstructed["heads"].items()
                    }
                    equal(expected, pred["predictions"])
                    equal(canonical_hash(pred["predictions"]), pred["prediction_hash"])
                progress("cold_seed_reconstructed", index + 1, len(head_rows) - index - 1)
            vault = upstream.engine.historical.LabelVault(
                Path(manifests["evaluator_label_manifests"]["retention"]["path"]),
                public["retention"],
            )
            for group in sealed:
                for row, pred in zip(public["retention"], group["rows"], strict=True):
                    recomputed.extend(
                        upstream.engine.historical.scored(
                            row,
                            pred,
                            vault.release(row["slot"], 256, sealed=True, retention=True),
                            "retention",
                            group["seed"],
                        )
                    )
        equal(recomputed, value["rows"])
        rebuilt = build(
            value,
            Path(value["terminal_validation_sidecar_path"]).parent,
            value["validation_receipts"],
        )
        for key in [
            *value["audit_statistics"],
            "learning_audit_ready_score",
            "h2_development_signal_score",
            "verdict_class",
            "honest_verdict",
        ]:
            equal(rebuilt[key], value[key])
        return True
    except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration):
        return False
