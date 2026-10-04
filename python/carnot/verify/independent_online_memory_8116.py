"""REQ-VERIFY-8116: learn a bounded residual without fitting or loading Qwen.

Original slots define time. The evaluator releases one authenticated target only
once its prediction is sealed, and retention targets open after all heads seal.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random
import time
from typing import Any, Callable

import numpy as np
from scipy.special import expit

from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting import v702_contract_custody as custody
from carnot.verify import methods_stream_custody_8111 as methods
from carnot.verify import radial_memory_8085 as radial

Json = dict[str, Any]
ROOT = methods.ROOT
NAME = "experiment_8116_v702_independent_online_memory"
TASK = "exp8116-independent-online-memory"
MODULE = "python/carnot/verify/independent_online_memory_8116.py"
RUNNER = "python/carnot/reporting/independent_memory_execution_8116.py"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_independent_online_memory_8116.py"
UPSTREAM = "results/experiment_8111_v702_methods_and_stream_custody.json"
UPSTREAM_HASH = "sha256:9f0c89a0d205488ceb545ca93497e6341bbc3fd8d01a94d69c773ed6535e9ece"
ARMS = ["frozen", "coefficient", "random", "error"]
TRACE_FIELDS = dict(
    issued="issued_state_rows",
    updates="update_rows",
    admissions="admission_rows",
    pending="pending_rows",
    proposals="proposals",
    decisions="decisions",
)
CONFIG: Json = dict(
    seeds=list(range(101, 121)),
    delay=8,
    growth=[96, 128, 160],
    initial_centers=16,
    maximum_centers=28,
    additions=4,
    pool=32,
    admission_labels=8,
    admission_margin=0.01,
    lr=0.01,
    l2=0.01,
    steps=4,
    coefficient_bound=4,
    gradient_norm=1,
    normalization_slots=[1, 64],
    admission="original_slot_modulo4_zero",
    tail="no_flush",
    fitted_head_required=False,
    candidate_weights="zero additions, shared training only after admission",
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed real counts make slow validation and replay visible to supervisors."""
    print(f"[exp8116] phase={phase} completed={completed} pending={pending}", flush=True)


def public_rows(rows: list[Json], total: int) -> None:
    """Reject missing original slots and evaluator metadata before geometry exists."""
    if (
        len(rows) != total
        or [r["slot"] for r in rows] != list(range(1, total + 1))
        or len({r["source_cluster_id"] for r in rows}) != total
    ):
        raise ValueError("original_slots")
    if any(set(r) & {"y", "label", "human_target", "labels", "quality"} for r in rows):
        raise ValueError("public_label")
    radial.matrix([r["values"] for r in rows if r["values"] is not None])


class LabelVault:
    """Keep evaluator rows opaque until a particular original slot can release.

    A lexical array split avoids decoding later targets when an earlier one is
    requested. No evaluator object is handed to the adaptive head.
    """

    def __init__(self, path: Path, rows: list[Json]):
        text = path.read_text()
        start = text.index('"rows"')
        start = text.index("[", start) + 1
        depth, quoted, escaped, begin = 0, False, False, start
        self.fragments: list[str] = []
        for i in range(start, len(text)):
            char = text[i]
            if not quoted and depth == 0 and char == "]":
                break
            if not quoted and depth == 0 and char == ",":
                self.fragments.append(text[begin:i])
                begin = i + 1
            elif quoted:
                if char == '"' and not escaped:
                    quoted = False
                escaped = char == "\\" and not escaped
            elif char == '"':
                quoted = True
            elif char in "[{":
                depth += 1
            elif char in "]}":
                depth -= 1
        self.fragments.append(text[begin:i])
        if len(self.fragments) != len(rows):
            raise ValueError("evaluator_slots")
        self.rows = rows

    def release(self, slot: int, clock: int, *, sealed: bool, retention: bool = False) -> Json:
        """An explicit clock and seal prohibit future or unissued target access."""
        if not retention and slot + CONFIG["delay"] > clock:
            raise ValueError("future_label")
        if not sealed:
            raise ValueError("prediction_seal")
        row: Json = json.loads(self.fragments[slot - 1])
        public = self.rows[slot - 1]
        if (row["slot"], row["unit_id"], row["source_cluster_id"]) != (
            slot,
            public["unit_id"],
            public["source_cluster_id"],
        ) or row["y"] not in [None, 0, 1]:
            raise ValueError("evaluator_identity")
        return row


def genesis(rows: list[Json], seed: int) -> Json:
    """Public first64 geometry gives all arms the same zero-residual starting point."""
    public_rows(rows, 256)
    warmup = [r for r in rows[:64] if r["values"] is not None]
    if len(warmup) < 28:
        raise ValueError("warmup")
    base = radial.initialize(
        [r["values"] for r in warmup], [r["source_cluster_id"] for r in warmup]
    )
    head = dict(centers=base["centers"], weights=[0.0] * 16, intercept=0.0, optimizer_step=0)
    return dict(
        seed=seed,
        baseline_hash=canonical_hash(rows),
        geometry=base["geometry"],
        arms={a: deepcopy(head) for a in ARMS},
        cursor=1,
        phase="issue",
        issued=[],
        updates=[],
        proposals=[],
        admissions=[],
        decisions=[],
        rejected_labels=[],
        released=[],
        candidates={},
        pending=[],
        used_admission=[],
        rng_state=None,
    )


def probability(head: Json, geometry: Json, values: list[float]) -> float:
    """Preserve the original Qwen logit as a fixed offset outside the residual."""
    phi = radial.design(dict(centers=head["centers"], geometry=geometry), [values])[0, 1:]
    return float(expit(values[0] + head["intercept"] + phi @ np.asarray(head["weights"])))


def train(head: Json, geometry: Json, pool: list[Json]) -> Json:
    """Four clipped gradient steps change only the residual, with bounded weights."""
    before = deepcopy(head)
    phi = radial.design(
        dict(centers=head["centers"], geometry=geometry), [r["values"] for r in pool]
    )
    offsets = np.asarray([r["values"][0] for r in pool])
    targets = np.asarray([r["y"] for r in pool])
    theta = np.asarray([head["intercept"], *head["weights"]])
    norms = []
    for _ in range(CONFIG["steps"]):
        error = expit(offsets + phi @ theta) - targets
        gradient = phi.T @ error / len(pool) + np.r_[0.0, CONFIG["l2"] * theta[1:]]
        gradient /= max(1.0, float(np.linalg.norm(gradient)))
        norms.append(float(np.linalg.norm(gradient)))
        theta -= CONFIG["lr"] * gradient
        theta[1:] = np.clip(theta[1:], -4.0, 4.0)
    head.update(
        intercept=float(theta[0]),
        weights=theta[1:].tolist(),
        optimizer_step=head["optimizer_step"] + 4,
    )
    return dict(
        before_hash=canonical_hash(before),
        after_hash=canonical_hash(head),
        before_weights=before["weights"],
        after_weights=head["weights"],
        before_intercept=before["intercept"],
        after_intercept=head["intercept"],
        touched_coefficients=np.flatnonzero(
            theta != np.asarray([before["intercept"], *before["weights"]])
        ).tolist(),
        gradient_norms=norms,
        steps=4,
    )


def propose(state: Json, slot: int) -> bool:
    """Commit equally sized growth candidates from already released training rows."""
    if state["candidates"]:
        state["proposals"].append(
            dict(kind="missed_growth", clock_slot=slot, reason="candidate_pending")
        )
        return False
    pool = state["released"][-32:]
    capacity = min(4, 28 - len(state["arms"]["error"]["centers"]), len(pool))
    if not capacity:
        state["proposals"].append(
            dict(kind="missed_growth", clock_slot=slot, reason="empty_pool_or_capacity")
        )
        return False
    rng = random.Random(state["seed"] * 1000 + slot)
    selections = dict(
        random=rng.sample(range(len(pool)), capacity),
        error=sorted(
            range(len(pool)),
            key=lambda i: (
                -abs(pool[i]["y"] - pool[i]["frozen_prediction"]),
                pool[i]["source_cluster_id"],
            ),
        )[:capacity],
    )
    state["rng_state"] = json.loads(json.dumps(rng.getstate()))
    record = dict(
        kind="candidate_commit",
        clock_slot=slot,
        pool=deepcopy(pool),
        selections=selections,
        capacity=capacity,
        states_before={},
        states_after={},
        proposal_training={},
    )
    g = state["geometry"]
    for arm in ["random", "error"]:
        head = deepcopy(state["arms"][arm])
        record["states_before"][arm] = deepcopy(head)
        for index in selections[arm]:
            row = pool[index]
            x = (
                (np.asarray(row["values"]) - np.asarray(g["mean"])) / np.asarray(g["std"])
            ).tolist()
            head["centers"].append(
                dict(
                    source_id=row["source_cluster_id"],
                    x=x,
                    feedback_origin="released_training",
                    label_slot=row["slot"],
                )
            )
            head["weights"].append(0.0)
        record["proposal_training"][arm] = dict(
            steps=0,
            touched_coefficients=[],
            reason="capacity proposal; training starts only after admission",
        )
        record["states_after"][arm] = deepcopy(head)
        state["candidates"][arm] = dict(
            head=head,
            commit_slot=slot,
            commit_hash=canonical_hash(head),
            scores=[],
            current_scores=[],
            frozen_scores=[],
            label_slots=[],
        )
    state["proposals"].append(record)
    return True


def admit(state: Json, clock: int) -> bool:
    """A joint guard keeps installed capacity matched and consumes labels once."""
    candidates = state["candidates"]
    if not candidates or min(len(c["scores"]) for c in candidates.values()) < 8:
        return False
    checks = {
        a: dict(
            candidate_brier=float(np.mean(c["scores"])),
            current_brier=float(np.mean(c["current_scores"])),
            frozen_brier=float(np.mean(c["frozen_scores"])),
        )
        for a, c in candidates.items()
    }
    passed = {
        a: v["candidate_brier"] <= v["current_brier"] + 0.01
        and v["candidate_brier"] <= v["frozen_brier"] + 0.01
        for a, v in checks.items()
    }
    accepted = all(passed.values())
    record = dict(
        clock_slot=clock,
        accepted=accepted,
        checks=checks,
        individual_passed=passed,
        labels=candidates["error"]["label_slots"],
        before={a: deepcopy(state["arms"][a]) for a in candidates},
    )
    if accepted:
        for arm, candidate in candidates.items():
            state["arms"][arm] = candidate["head"]
    record["after"] = {a: deepcopy(state["arms"][a]) for a in candidates}
    state["decisions"].append(record)
    state["candidates"] = {}
    return True


def restore(state: Json, rows: list[Json]) -> Json:
    """A resumed head must refer to exactly the same public baseline and clock."""
    public_rows(rows, 256)
    if state["baseline_hash"] != canonical_hash(rows):
        raise ValueError("baseline_hash")
    return state


def execute(
    state: Json,
    rows: list[Json],
    vault: LabelVault,
    *,
    seal: Callable[[str, Json], None] | None = None,
    stop_event: str = "",
) -> None:
    """Persist prediction and candidate boundaries before opening delayed targets."""
    restore(state, rows)
    while state["cursor"] <= 256:
        slot = state["cursor"]
        row = rows[slot - 1]
        event = ""
        if state["phase"] == "issue":
            predictions = {
                a: None
                if row["values"] is None
                else probability(h, state["geometry"], row["values"])
                for a, h in state["arms"].items()
            }
            issued = dict(
                slot=slot,
                source_cluster_id=row["source_cluster_id"],
                predictions=predictions,
                states={a: canonical_hash(h) for a, h in state["arms"].items()},
                state_before=deepcopy(state["arms"]),
                admission_only=slot % 4 == 0,
                candidate_predictions={
                    a: None
                    if row["values"] is None
                    else probability(c["head"], state["geometry"], row["values"])
                    for a, c in state["candidates"].items()
                },
                release_slot=slot + 8,
                prediction_hash=canonical_hash(predictions),
            )
            state["issued"].append(issued)
            state["pending"].append(dict(slot=slot, release_slot=slot + 8))
            state["phase"] = "candidate"
            event = "issued"
        elif state["phase"] == "candidate":
            if slot in CONFIG["growth"] and propose(state, slot):
                event = "candidate_commit"
            state["phase"] = "release"
        else:
            released = slot - 8
            if released >= 1:
                label = vault.release(released, slot, sealed=len(state["issued"]) >= released)
                old = rows[released - 1]
                issued = state["issued"][released - 1]
                state["pending"] = [r for r in state["pending"] if r["slot"] != released]
                if label["y"] is None or old["values"] is None:
                    state["rejected_labels"].append(
                        dict(
                            label_slot=released,
                            clock_slot=slot,
                            reason=label["exclusion_reason"]
                            or old["exclusion_reason"]
                            or "excluded",
                        )
                    )
                elif released % 4:
                    observation = dict(
                        old, y=label["y"], frozen_prediction=issued["predictions"]["frozen"]
                    )
                    state["released"].append(observation)
                    for arm in ARMS[1:]:
                        update = train(state["arms"][arm], state["geometry"], [observation])
                        state["updates"].append(
                            dict(update, arm=arm, label_slot=released, clock_slot=slot)
                        )
                elif state["candidates"] and released > state["candidates"]["error"]["commit_slot"]:
                    record = dict(label_slot=released, clock_slot=slot, y=label["y"], scores={})
                    for arm, candidate in state["candidates"].items():
                        p = issued["candidate_predictions"][arm]
                        scores = [
                            float((p - label["y"]) ** 2),
                            float((issued["predictions"][arm] - label["y"]) ** 2),
                            float((issued["predictions"]["frozen"] - label["y"]) ** 2),
                        ]
                        for key, score in zip(
                            ["scores", "current_scores", "frozen_scores"], scores, strict=True
                        ):
                            candidate[key].append(score)
                        candidate["label_slots"].append(released)
                        record["scores"][arm] = scores
                    state["admissions"].append(record)
                    state["used_admission"].append(released)
                    if admit(state, slot):
                        event = "admission"
            state["cursor"] += 1
            state["phase"] = "issue"
        if event and seal:
            seal(event, state)
        if stop_event and event == stop_event:
            return
        if state["phase"] == "issue" and slot % 64 == 0:
            progress("seed_slots", slot, 256 - slot)


def authenticate(
    root: Path, raw: Path, fixture: bool, binder: methods.Custody
) -> tuple[Json, methods.Custody]:
    """Only Exp8111's independently qualified stream authorizes this replay."""
    binder.upstream = "exp8111-methods-and-stream-custody"
    path = root / UPSTREAM
    value = binder.read(path, None if fixture else UPSTREAM_HASH)
    terminal = custody.terminal(path, value, binder)
    side = binder.read(Path(value["terminal_validation_sidecar_path"]))
    binder.require(
        Path(value["terminal_validation_sidecar_path"]),
        "normal_process_exit",
        True,
        side.get("normal_process_exit"),
    )
    for key, expected in [
        ("experiment_id", 8111),
        ("methods_ready_score", 1),
        ("stream_input_ready_score", 1),
        ("required_checks_passed", True),
        ("flagged_adversarial", False),
    ]:
        binder.require(path, key, expected, value.get(key))
    binder.require(path, "terminal.report.passed", True, terminal["report"].get("passed"))
    for ref in value["raw_shard_hashes"]:
        binder.bind(Path(ref["path"]), ref["sha256"])
    for label, digest in value["code_config_hashes"].items():
        binder.bind(ROOT / label, digest)
    return value, binder


def scored(public: Json, prediction: Json, label: Json, role: str, seed: int) -> list[Json]:
    """Brier and typed decision cost retain excluded original units in every arm."""
    result = []
    for arm, p in prediction["predictions"].items():
        eligible = p is not None and label["y"] is not None
        reason = None if eligible else public["exclusion_reason"] or label["exclusion_reason"]
        for metric in ["brier", "typed_cost"]:
            numerator = (
                0.0
                if not eligible
                else (p - label["y"]) ** 2
                if metric == "brier"
                else radial.loss(radial.action(p), label["y"])
            )
            result.append(
                dict(
                    unit_id=public["unit_id"],
                    source_cluster_id=public["source_cluster_id"],
                    arm=arm,
                    condition=role,
                    metric=metric,
                    numerator=numerator,
                    denominator=1,
                    status="completed"
                    if eligible
                    else "censored"
                    if reason == "feedback_unresolved_tail"
                    else "excluded",
                    exclusion_reason=reason,
                    seed=seed,
                    slot=public["slot"],
                    y=label["y"],
                    prediction=p,
                    prediction_hash=prediction["prediction_hash"],
                )
            )
    return result


def reductions(rows: list[Json]) -> Json:
    """Average seeds within original sources before computing any comparative mean."""
    grouped: dict[tuple[str, str, str, str], list[float]] = {}
    for row in rows:
        if row["status"] == "completed":
            key = (row["condition"], row["arm"], row["metric"], row["source_cluster_id"])
            grouped.setdefault(key, []).append(row["numerator"] / row["denominator"])
    means: dict[str, list[float]] = {}
    for (condition, arm, metric, _), values in grouped.items():
        means.setdefault("/".join([condition, arm, metric]), []).append(float(np.mean(values)))
    return {
        key: dict(mean=float(np.mean(values)), independent_sources=len(values))
        for key, values in means.items()
    }


def measure(
    root: Path,
    raw: Path,
    *,
    fixture: bool = False,
    seeds: list[int] | None = None,
    mutation: str = "",
) -> Json:
    """Run finite CPU heads, preserving all exact inputs and every causal boundary."""
    started = time.monotonic()
    raw.mkdir(parents=True, exist_ok=True)
    binder = methods.Custody(raw / "inputs")
    work: Json = dict(
        execution_ready=0,
        owned_failure=bool(mutation),
        rows=[],
        issued_state_rows=[],
        update_rows=[],
        admission_rows=[],
        pending_rows=[],
        proposals=[],
        decisions=[],
        final_state_hashes={},
        retention_predictions=[],
        resume_checks=[],
        effective_center_counts={},
        source_artifact_hashes=[],
        raw_shard_hashes=[],
        gate_check_summary=[],
        phase_spans=[],
        config=deepcopy(CONFIG),
        seeds=seeds or CONFIG["seeds"],
    )
    progress("before_authentication")
    try:
        upstream, binder = authenticate(root, raw, fixture, binder)
        work["historical_model_provenance"] = upstream.get("historical_model_provenance", {})
        features = {
            role: binder.read(
                Path(upstream[role + "_feature_manifest"]["path"]),
                upstream[role + "_feature_manifest"]["sha256"],
            )["rows"]
            for role in ["stream", "retention"]
        }
        for role, total in [("stream", 256), ("retention", 64)]:
            public_rows(features[role], total)
            ref = upstream["evaluator_label_manifests"][role]
            binder.bind(Path(ref["path"]), ref["sha256"])
        work["input_manifests"] = {
            role: {
                "features": binder.bind(Path(upstream[role + "_feature_manifest"]["path"])),
                "labels": binder.bind(Path(upstream["evaluator_label_manifests"][role]["path"])),
            }
            for role in features
        }
        access = {
            role: LabelVault(
                Path(work["input_manifests"][role]["labels"]["snapshot_path"]), features[role]
            )
            for role in features
        }
        work["phase_spans"].append(
            dict(phase="authentication", start_s=0.0, duration_s=time.monotonic() - started)
        )
        progress("after_authentication", 320)
        heads = []
        for index, seed in enumerate(work["seeds"]):
            phase = time.monotonic()
            progress("before_CPU_head", index, len(work["seeds"]) - index)
            state = genesis(features["stream"], seed)
            if mutation == "future-label":
                access["stream"].release(16, 1, sealed=True)

            def seal(event: str, current: Json) -> None:
                atomic_json(
                    raw / f"seed-{seed}" / f"{current['cursor']:03d}-{event}.json",
                    current["issued"][-1] if event == "issued" else current,
                )

            execute(state, features["stream"], access["stream"], seal=seal)
            heads.append(state)
            atomic_json(raw / f"seed-{seed}" / "final_state.json", state)
            for key, target in TRACE_FIELDS.items():
                work[target].extend(dict(r, seed=seed) for r in state[key])
            work["final_state_hashes"][str(seed)] = canonical_hash(state)
            work["effective_center_counts"][str(seed)] = {
                arm: dict(
                    installed=len(h["centers"]),
                    unique=len({canonical_hash(c["x"]) for c in h["centers"]}),
                )
                for arm, h in state["arms"].items()
            }
            for boundary in ["issued", "candidate_commit", "admission"]:
                recovered = genesis(features["stream"], seed)
                execute(recovered, features["stream"], access["stream"], stop_event=boundary)
                checkpoint = raw / f"seed-{seed}" / f"resume-{boundary}.json"
                atomic_json(checkpoint, recovered)
                snapshot = json.loads(checkpoint.read_text())
                cold_access = LabelVault(
                    Path(work["input_manifests"]["stream"]["labels"]["snapshot_path"]),
                    features["stream"],
                )
                execute(restore(snapshot, features["stream"]), features["stream"], cold_access)
                work["resume_checks"].append(
                    dict(
                        seed=seed,
                        boundary=boundary,
                        checkpoint_path=str(checkpoint),
                        checkpoint_sha256=sha256_file(checkpoint),
                        mode="durable_JSON_reload_with_fresh_label_vault",
                        passed=canonical_hash(snapshot) == canonical_hash(state),
                    )
                )
            for row in features["retention"]:
                predictions = {
                    a: None
                    if row["values"] is None
                    else probability(h, state["geometry"], row["values"])
                    for a, h in state["arms"].items()
                }
                work["retention_predictions"].append(
                    dict(
                        seed=seed,
                        slot=row["slot"],
                        predictions=predictions,
                        prediction_hash=canonical_hash(predictions),
                        final_state_hash=canonical_hash(state),
                    )
                )
            work["phase_spans"].append(
                dict(
                    phase=f"CPU_seed_{seed}",
                    start_s=phase - started,
                    duration_s=time.monotonic() - phase,
                )
            )
            progress("after_CPU_head", index + 1, len(work["seeds"]) - index - 1)
        work["retention_prediction_manifest"] = methods.cohort.immutable(
            raw / "retention_predictions.json", dict(rows=work["retention_predictions"])
        )
        progress(
            "retention_predictions_sealed_before_evaluator", len(work["retention_predictions"])
        )
        for seed, state in zip(work["seeds"], heads, strict=True):
            for row, issued in zip(features["stream"], state["issued"], strict=True):
                label = (
                    access["stream"].release(row["slot"], 256, sealed=True)
                    if row["slot"] <= 248
                    else dict(y=None, exclusion_reason="feedback_unresolved_tail")
                )
                work["rows"].extend(
                    scored(
                        row,
                        issued,
                        label,
                        "stream_warmup" if row["slot"] <= 64 else "stream_later",
                        seed,
                    )
                )
            for row in features["retention"]:
                prediction = next(
                    r
                    for r in work["retention_predictions"]
                    if r["seed"] == seed and r["slot"] == row["slot"]
                )
                label = access["retention"].release(row["slot"], 256, sealed=True, retention=True)
                work["rows"].extend(scored(row, prediction, label, "retention", seed))
        work["execution_ready"] = int(all(r["passed"] for r in work["resume_checks"]))
        if not work["execution_ready"]:
            work["owned_failure"] = True
            binder.checks.append(
                dict(
                    check="resume_checks",
                    upstream=TASK,
                    path=str(raw),
                    hash=None,
                    artifact_field="resume_checks",
                    op="==",
                    expected=True,
                    observed=False,
                    passed=False,
                )
            )
    except (OSError, ValueError, KeyError, TypeError) as error:
        if not binder.failures:
            work["owned_failure"] = True
            binder.checks.append(
                dict(
                    check="owned_execution",
                    upstream=TASK,
                    path=str(raw),
                    hash=None,
                    artifact_field="owned_execution",
                    op="==",
                    expected="valid",
                    observed=str(error),
                    passed=False,
                )
            )
    return finish(work, binder, raw, started)


def finish(work: Json, binder: methods.Custody, raw: Path, started: float) -> Json:
    """Seal primitives and code bytes so replay can reject drift independently."""
    for operand in [
        MODULE,
        RUNNER,
        CLI,
        TEST,
        "AGENTS.md",
        "CODEX.md",
        "CLAUDE.md",
        "ops/e2e-test-plan.md",
        "ops/exclusion_manifest.yaml",
        "openspec/capabilities/verification/spec.md",
        "openspec/capabilities/research-reporting/spec.md",
        "openspec/change-proposals/research-roadmap-vNEXT.md",
        "scripts/experiment_template.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/primary_publication.py",
        "python/carnot/verify/radial_memory_8085.py",
        "python/carnot/verify/development_methods_8098.py",
        "python/carnot/verify/delayed_confidence_8000.py",
        "python/carnot/verify/qwen_learning_stream_capture_8102.py",
        "scripts/experiments/experiment_8076_v699_projected_online_learning.py",
        "tests/python/test_primary_publication_7928.py",
        *[".venv/bin/" + n for n in ["python", "pytest", "coverage", "ruff", "mypy"]],
    ]:
        binder.bind(ROOT / operand)
    if (raw / "validation_commands.json").exists():
        binder.bind(raw / "validation_commands.json")
    work["source_artifact_hashes"] = binder.refs
    work["gate_check_summary"] = binder.checks + [
        r for r in binder.failures if r not in binder.checks
    ]
    work["code_config_hashes"] = {p: sha256_file(ROOT / p) for p in [MODULE, RUNNER, CLI, TEST]}
    work["duration_s"] = time.monotonic() - started
    if not work["rows"]:
        work["rows"] = [
            dict(
                unit_id=f"missing-{i}",
                source_cluster_id=f"missing-{i}",
                arm="frozen",
                condition="missing_original_slot",
                metric="brier",
                numerator=0,
                denominator=1,
                status="excluded",
                exclusion_reason="upstream_unavailable",
                seed=101,
                slot=i,
            )
            for i in range(1, 321)
        ]
    work["reductions"] = reductions(work["rows"])
    work["retention_prediction_manifest"] = work.get("retention_prediction_manifest")
    for name, keys in [
        ("primitive_rows", ["rows"]),
        (
            "causal_evidence",
            [
                "issued_state_rows",
                "update_rows",
                "admission_rows",
                "pending_rows",
                "proposals",
                "decisions",
            ],
        ),
        (
            "final_evidence",
            [
                "final_state_hashes",
                "retention_predictions",
                "resume_checks",
                "effective_center_counts",
            ],
        ),
    ]:
        work["raw_shard_hashes"].append(
            methods.cohort.immutable(raw / (name + ".json"), {k: work[k] for k in keys})
        )
    atomic_json(raw / "measurement.json", work)
    progress("measurement_normal_exit", len(work["seeds"]) if work["execution_ready"] else 0)
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Valid execution can finish null; external blocks and owned errors differ."""
    owned = bool(receipts) and all(r["passed"] for r in receipts) and not work["owned_failure"]
    verdict = (
        "disqualified"
        if not owned
        else "blocked"
        if not work["execution_ready"]
        else "circular_positive"
        if fixture
        else "null"
    )
    failed = next(
        (r["check"] for r in work["gate_check_summary"] if not r["passed"]),
        "offline_prequential_exposed_development",
    )
    unique = {
        r["source_cluster_id"]: r
        for r in work["rows"]
        if r["arm"] == "frozen" and r["metric"] == "brier"
    }
    counts = {
        status: sum(r["status"] == status for r in unique.values())
        for status in ["completed", "excluded", "censored", "failed"]
    }
    value = dict(
        work,
        experiment_id=8116,
        task_id=TASK,
        milestone="2026.10.702",
        honest_verdict="complete_"
        + verdict
        + "_"
        + ("owned_validation" if verdict == "disqualified" else failed),
        verdict_class=verdict,
        verifier_is_oracle=fixture,
        required_checks_passed=owned,
        flagged_adversarial=False,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        inference_substrate="verifier_ensemble_against_cached_candidates"
        if work["execution_ready"]
        else "aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        call_ledger=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        trained_head_specs=[]
        if not work["execution_ready"]
        else [
            dict(
                kind="Gaussian_residual",
                seeds=work["seeds"],
                arms=ARMS[1:],
                frozen_qwen_offset=True,
                config=CONFIG,
            )
        ],
        learning_mode="offline_prequential_exposed_development",
        learning_trajectory_ready_score=int(owned and work["execution_ready"]),
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        claim_scope="Finite delayed offline residual updates and descriptive source-level Brier/cost; no independently qualified learning-benefit test",
        exposure_scope="exposed_development_within_run_disjoint",
        intended_count=320,
        eligible_count=counts["completed"],
        independent_count=sum(
            k.startswith("sha256:") for k, r in unique.items() if r["status"] == "completed"
        ),
        completed_count=counts["completed"],
        excluded_count=counts["excluded"],
        censored_count=counts["censored"],
        failed_count=counts["failed"],
        sample_size_budget=dict(
            stream=256,
            stream_capture_exclusions=17,
            retention=64,
            retention_capture_exclusions=3,
            seeds=work["seeds"],
            independent_unit="original_source_cluster",
            human_exclusions="additional; retain original clock",
        ),
        run_date="20261004",
        random_seed=101,
        acceptance_gates=dict(
            custody="Exact Exp8111 bytes, terminal and referenced public/evaluator hashes",
            causality="prediction before delay8 release, no admission training or tail flush",
            admission="joint candidate mean Brier <= incumbent+.01 and <= frozen+.01; matched capacity",
            execution="all recovery checks and owned validation pass",
            science="descriptive exposed development; no independent benefit or lifelong generalization claim",
        ),
        methodology_note="Sealed original Qwen logits plus zero-start bounded Gaussian residual; four clipped SGD steps per released training label, no batch solver or model invocation. Operator delay8/eight-admission overrides preserve the historical Exp8111 design.",
    )
    value["field_principles"] = {
        k: f"Record {k} to bind finite offline evidence without implying independent generalization."
        for k in value
    }
    value["field_principles"].update(
        verdict_class="External blocks are terminal; owned validation failures disqualify; completed null is final.",
        rows="Original source units, not seeds, determine independent support.",
        issued_state_rows="Sealed predictions precede evaluator access and later updates.",
        pending_rows="Unreleased tail feedback cannot train or contribute reported future benefit.",
        model_invocation_counts="Historical Qwen provenance is not current model work.",
        learning_trajectory_ready_score="Valid causal execution need not show benefit.",
        admission_rows="Fresh admission labels are synchronized, single-use and never training data.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Cold-recompute rows, seals and reductions rather than trust headline totals."""
    try:
        value = json.loads(path.read_text())
        checksum = value.pop("reproducibility_checksum")
        if checksum != canonical_hash(value):
            return False
        raw = Path(value["terminal_validation_sidecar_path"]).parent
        work = json.loads((raw / "measurement.json").read_text())
        for ref in [*work["source_artifact_hashes"], *work["raw_shard_hashes"]]:
            if sha256_file(Path(ref.get("snapshot_path", ref["path"]))) != ref["sha256"]:
                return False
        primitive = json.loads((raw / "primitive_rows.json").read_text())["rows"]
        if primitive != work["rows"] or reductions(primitive) != work["reductions"]:
            return False
        for row in primitive:
            if (
                row.get("prediction") is not None
                and row.get("y") is not None
                and row["status"] == "completed"
            ):
                expected = (
                    (row["prediction"] - row["y"]) ** 2
                    if row["metric"] == "brier"
                    else radial.loss(radial.action(row["prediction"]), row["y"])
                )
                if row["numerator"] != expected:
                    return False
        for row in work["issued_state_rows"]:
            if row["prediction_hash"] != canonical_hash(row["predictions"]):
                return False
        if not verify_trajectory(work):
            return False
        receipts = json.loads((raw / "validation_receipts.json").read_text())["rows"]
        for receipt in receipts:
            if (
                "log_path" in receipt
                and sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]
            ):
                return False
        rebuilt = build(work, raw, receipts, fixture=value["verifier_is_oracle"])
        return canonical_hash(rebuilt) == canonical_hash(
            dict(value, reproducibility_checksum=checksum)
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def verify_trajectory(work: Json) -> bool:
    """Reexecute from sealed public inputs to detect even consistently rehashed fraud."""
    if not work["execution_ready"]:
        return True
    manifests = work["input_manifests"]
    features = {
        role: json.loads(Path(refs["features"]["snapshot_path"]).read_text())["rows"]
        for role, refs in manifests.items()
    }
    access = {
        role: LabelVault(Path(refs["labels"]["snapshot_path"]), features[role])
        for role, refs in manifests.items()
    }
    rows, retention = [], []
    for index, seed in enumerate(work["seeds"]):
        progress("independent_cold_seed", index, len(work["seeds"]) - index)
        state = genesis(features["stream"], seed)
        execute(state, features["stream"], access["stream"])
        if canonical_hash(state) != work["final_state_hashes"][str(seed)]:
            return False
        for key, target in TRACE_FIELDS.items():
            if [dict(r, seed=seed) for r in state[key]] != [
                r for r in work[target] if r["seed"] == seed
            ]:
                return False
        for public, issued in zip(features["stream"], state["issued"], strict=True):
            label = (
                access["stream"].release(public["slot"], 256, sealed=True)
                if public["slot"] <= 248
                else dict(y=None, exclusion_reason="feedback_unresolved_tail")
            )
            rows.extend(
                scored(
                    public,
                    issued,
                    label,
                    "stream_warmup" if public["slot"] <= 64 else "stream_later",
                    seed,
                )
            )
        for public in features["retention"]:
            predictions = {
                a: None
                if public["values"] is None
                else probability(h, state["geometry"], public["values"])
                for a, h in state["arms"].items()
            }
            prediction = dict(
                seed=seed,
                slot=public["slot"],
                predictions=predictions,
                prediction_hash=canonical_hash(predictions),
                final_state_hash=canonical_hash(state),
            )
            retention.append(prediction)
            label = access["retention"].release(public["slot"], 256, sealed=True, retention=True)
            rows.extend(scored(public, prediction, label, "retention", seed))
    return rows == work["rows"] and retention == work["retention_predictions"]
