"""REQ-VERIFY-8138: check finite delayed heads without loading a language model.

Original slots keep time even when their public features are missing. Candidates
use only released update labels; future admission labels never train a head.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
import random
from typing import Any, Callable

import numpy as np
from scipy.special import expit

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.verify import independent_online_memory_8116 as historical
from carnot.verify import methods_stream_custody_8111 as methods

Json = dict[str, Any]
ROOT = methods.ROOT
NAME = "experiment_8138_v704_learning_protocol"
CLI = f"scripts/experiments/{NAME}.py"
MODULE = "python/carnot/verify/learning_protocol_8138.py"
RUNNER = "python/carnot/reporting/learning_protocol_execution_8138.py"
TEST = "tests/python/test_learning_protocol_8138.py"
ARMS = ["frozen_qwen_offset", "fixed_public_center", "random_past_center", "error_center"]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush measured counts so waiting cannot imply model activity."""
    print(f"[exp8138] phase={phase} completed={completed} pending={pending}", flush=True)


def protocol() -> Json:
    """The sealed design, rather than later override prose, supplies all numbers."""
    return dict(methods.protocol()["delayed_memory"])


def bucket(row: Json) -> int:
    """Public source identity fixes label use before the target is available."""
    return (
        int(hashlib.sha256(row.get("source_id", row["source_cluster_id"]).encode()).hexdigest(), 16)
        % 4
    )


def fixture(total: int = 256, role: str = "stream") -> tuple[list[Json], list[int | None]]:
    """Private known targets check mechanics and carry no natural-data credit."""
    rows = [
        dict(
            slot=i,
            unit_id=f"{role}-{i}",
            source_cluster_id=canonical_hash([role, i]),
            values=[(-1.0) ** i, *[float(i * j % 19) for j in range(1, 9)]],
            exclusion_reason=None,
        )
        for i in range(1, total + 1)
    ]
    return rows, [i % 2 for i in range(1, total + 1)]


def genesis(rows: list[Json], seed: int) -> Json:
    """Warmup geometry and zero coefficients depend on public values only."""
    historical.public_rows(rows, 256)
    warm = [r for r in rows[:64] if r["values"] is not None]
    base = historical.radial.initialize(
        [r["values"] for r in warm], [r["source_cluster_id"] for r in warm]
    )
    head = dict(centers=base["centers"], weights=[0.0] * 16, intercept=0.0, optimizer_step=0)
    return dict(
        seed=seed,
        phase="issue",
        cursor=1,
        geometry=base["geometry"],
        reserved=base["reserved"],
        arms={a: deepcopy(head) for a in ARMS},
        pending=[],
        lost=[],
        issued=[],
        released=[],
        consumed=[],
        training=[],
        used_admission=[],
        pool=[],
        candidates={},
        events=[],
        baseline_hash=canonical_hash(rows),
        rng_state=None,
    )


def probability(head: Json, geometry: Json, values: list[float]) -> float:
    """The historical Qwen logit remains a fixed offset, outside learned weights."""
    return historical.probability(head, geometry, values)


def scalar_probability(head: Json, geometry: Json, values: list[float]) -> float:
    """Separate scalar arithmetic detects mistakes in vectorized predictions."""
    z = [(x - m) / s for x, m, s in zip(values, geometry["mean"], geometry["std"], strict=True)]
    phi = [
        math.exp(
            -sum((x - y) ** 2 for x, y in zip(z, c["x"], strict=True))
            / (2 * geometry["sigma"] ** 2)
        )
        for c in head["centers"]
    ]
    logit = (
        values[0]
        + head["intercept"]
        + sum(w * p for w, p in zip(head["weights"], phi, strict=True))
    )
    return 1 / (1 + math.exp(-logit))


def train(head: Json, geometry: Json, pool: list[Json], *, reference: bool = False) -> Json:
    """Four clipped SGD steps produce a proposal; admission controls installation."""
    h = deepcopy(head)
    phi = historical.radial.design(
        dict(centers=h["centers"], geometry=geometry), [r["values"] for r in pool]
    )
    theta = np.array([h["intercept"], *h["weights"]])
    for _ in range(4):
        errors = expit(np.array([r["values"][0] for r in pool]) + phi @ theta) - [
            r["y"] for r in pool
        ]
        if reference:
            gradient = np.array(
                [
                    sum(float(phi[i, j]) * float(errors[i]) for i in range(len(pool))) / len(pool)
                    for j in range(len(theta))
                ]
            )
        else:
            gradient = phi.T @ errors / len(pool)
        gradient[1:] += 0.01 * theta[1:]
        gradient /= max(1.0, float(np.linalg.norm(gradient)))
        theta -= 0.05 * gradient
        theta[1:] = np.clip(theta[1:], -4, 4)
    h.update(
        intercept=float(theta[0]),
        weights=theta[1:].tolist(),
        optimizer_step=h["optimizer_step"] + 4,
    )
    return h


def shard(raw: Path, value: Json) -> Json:
    """Repeated state bytes share one exact content address instead of many copies."""
    path = raw / "shards" / (canonical_hash(value).split(":")[1] + ".json")
    if not path.exists():
        atomic_json(path, value)
    return dict(path=str(path), sha256=sha256_file(path))


def event(state: Json, kind: str, slot: int, **detail: Any) -> None:
    """Every causal boundary includes the installed-head identity and pending clock."""
    state["events"].append(
        dict(
            kind=kind,
            slot=slot,
            pending=list(state["pending"]),
            state_hash=canonical_hash(state["arms"]),
            **detail,
        )
    )


def propose(state: Json, rows: list[Json], slot: int) -> None:
    """Install zero-weight capacity, then seal trained candidates before future labels."""
    pool = state["pool"][-64:]
    errors = [r for r in pool if abs(r["y"] - r["frozen_prediction"]) >= 0.5]
    if len(pool) < 16 or len(errors) < 4:
        event(state, "commit_candidate", slot, disposition="ineligible", candidates={})
        return
    rng = random.Random(state["seed"] * 1000 + slot)
    selected = {
        "fixed_public_center": state["reserved"][(slot // 64 - 1) * 4 : slot // 64 * 4],
        "random_past_center": rng.sample(pool, 4),
        "error_center": sorted(
            pool, key=lambda r: (-abs(r["y"] - r["frozen_prediction"]), r["source_cluster_id"])
        )[:4],
    }
    import json

    state["rng_state"] = json.loads(json.dumps(rng.getstate()))
    state["candidates"] = {}
    for arm in ARMS[1:]:
        h = state["arms"][arm]
        for row in selected[arm]:
            center = (
                row
                if arm == "fixed_public_center"
                else dict(
                    source_id=row["source_cluster_id"],
                    x=(
                        (np.array(row["values"]) - state["geometry"]["mean"])
                        / state["geometry"]["std"]
                    ).tolist(),
                )
            )
            h["centers"].append(deepcopy(center))
            h["weights"].append(0.0)
        proposed = train(h, state["geometry"], pool)
        checked = train(h, state["geometry"], pool, reference=True)
        if not np.allclose(proposed["weights"], checked["weights"], atol=1e-12, rtol=0):
            raise ValueError("gradient_reference")
        state["candidates"][arm] = dict(head=proposed, base=deepcopy(h), labels=[], commit=slot)
    event(
        state,
        "commit_candidate",
        slot,
        disposition="committed",
        candidates=deepcopy(state["candidates"]),
    )
    event(state, "select_future_admission", slot, minimum_original_slot=slot + 1)


def interpolate(candidate: Json, step: float) -> Json:
    """A sealed grid permits shrinking the proposed change without another fit."""
    h = deepcopy(candidate["head"])
    base = candidate["base"]
    h["weights"] = (
        np.array(base["weights"]) + step * (np.array(h["weights"]) - base["weights"])
    ).tolist()
    h["intercept"] = base["intercept"] + step * (h["intercept"] - base["intercept"])
    return h


def losses(probabilities: list[float], targets: list[int]) -> tuple[float, float, int]:
    """Decision cost and false accepts check safety in addition to squared error."""
    actions = [historical.radial.action(p) for p in probabilities]
    return (
        sum(historical.radial.loss(a, y) for a, y in zip(actions, targets, strict=True))
        / len(targets),
        sum((p - y) ** 2 for p, y in zip(probabilities, targets, strict=True)) / len(targets),
        sum(a == "accept" and y == 1 for a, y in zip(actions, targets, strict=True)),
    )


def admit(state: Json, slot: int) -> None:
    """Use exactly twelve future labels once; choose the largest safe grid step."""
    records = next(iter(state["candidates"].values()))["labels"]
    if len(records) != 12:
        return
    targets = [r["y"] for r in records]
    decisions = {}
    for arm, candidate in state["candidates"].items():
        incumbent = losses([r["predictions"][arm] for r in records], targets)
        frozen = losses([r["predictions"][ARMS[0]] for r in records], targets)
        chosen = 0.0
        if min(targets.count(0), targets.count(1)) >= 2:
            for step in [1.0, 0.5, 0.25, 0.125]:
                result = losses([r["grid"][arm][str(step)] for r in records], targets)
                if (
                    result[0] <= incumbent[0]
                    and result[1] <= incumbent[1]
                    and result[0] <= frozen[0] + 0.02
                    and result[1] <= frozen[1] + 0.01
                    and result[2] <= incumbent[2]
                    and result[2] <= frozen[2]
                ):
                    chosen = step
                    break
        decisions[arm] = chosen
        if chosen:
            state["arms"][arm] = interpolate(candidate, chosen)
    event(state, "admit_once", slot, labels=[r["label_slot"] for r in records], steps=decisions)
    state["candidates"] = {}
    event(state, "durable_update", slot, heads=deepcopy(state["arms"]))


def run(
    rows: list[Json],
    labels: list[int | None],
    seed: int,
    *,
    capacity: int = 32,
    state: Json | None = None,
    stop: int = 256,
    seal: Callable[[str, Json], None] | None = None,
) -> Json:
    """Predict, commit, release, admit and persist in original-slot order."""
    historical.public_rows(rows, 256)
    if len(labels) != 256:
        raise ValueError("evaluator_slots")
    s = genesis(rows, seed) if state is None else state
    if s["baseline_hash"] != canonical_hash(rows):
        raise ValueError("baseline_hash")
    s["capacity"] = capacity
    while s["cursor"] <= stop:
        slot = s["cursor"]
        if s["phase"] == "issue":
            row = rows[slot - 1]
            if slot in [64, 128, 192, 256] and s["candidates"]:
                event(s, "defer_candidate", slot, reason="admission_deadline")
                s["candidates"] = {}
            prediction = {
                a: None if row["values"] is None else probability(h, s["geometry"], row["values"])
                for a, h in s["arms"].items()
            }
            grid = {
                a: {
                    str(t): None
                    if row["values"] is None
                    else probability(interpolate(c, t), s["geometry"], row["values"])
                    for t in [1.0, 0.5, 0.25, 0.125]
                }
                for a, c in s["candidates"].items()
            }
            issued = dict(
                slot=slot,
                predictions=prediction,
                grid=grid,
                prediction_hash=canonical_hash(prediction),
            )
            s["issued"].append(issued)
            s["pending"].append(slot)
            if len(s["pending"]) > capacity:
                s["lost"].append(s["pending"].pop(0))
            event(s, "issue_prediction", slot, prediction_hash=issued["prediction_hash"])
            event(s, "durable_commit", slot)
            s["phase"] = "candidate"
            if seal:
                seal("durable_commit", s)
        if s["phase"] == "candidate":
            if slot in [64, 128, 192]:
                propose(s, rows, slot)
            s["phase"] = "release"
            if seal:
                seal("commit_candidate", s)
        released = slot - 20
        if released in s["pending"]:
            if released in s["consumed"] or released in s["used_admission"]:
                raise ValueError("reused_label")
            s["pending"].remove(released)
            s["released"].append(released)
            old, y = rows[released - 1], labels[released - 1]
            if y not in [0, 1, None]:
                raise ValueError("evaluator_label")
            event(s, "release_feedback", slot, label_slot=released)
            s["consumed"].append(released)
            if old["values"] is not None and y is not None:
                if bucket(old):
                    s["training"].append(released)
                    s["pool"].append(
                        dict(
                            old,
                            y=y,
                            frozen_prediction=s["issued"][released - 1]["predictions"][ARMS[0]],
                        )
                    )
                elif s["candidates"] and released > next(iter(s["candidates"].values()))["commit"]:
                    observation = dict(s["issued"][released - 1], y=y, label_slot=released)
                    s["used_admission"].append(released)
                    for c in s["candidates"].values():
                        c["labels"].append(observation)
                    admit(s, slot)
        s["phase"] = "issue"
        s["cursor"] += 1
        if seal:
            seal("durable_update", s)
    verify_events(s, rows)
    return s


def verify_events(state: Json, rows: list[Json]) -> bool:
    """An independent clock and scalar head check every saved causal boundary."""
    pending: set[int] = set()
    issued, released, last = set(), set(), (0, -1)
    reference = genesis(rows, state["seed"])
    heads = reference["arms"]
    candidates: Json = {}
    grouped: dict[int, list[str]] = {}
    ranks = {
        "defer_candidate": -1,
        "issue_prediction": 0,
        "durable_commit": 1,
        "commit_candidate": 2,
        "select_future_admission": 3,
        "release_feedback": 4,
        "admit_once": 5,
        "durable_update": 6,
    }
    for item in state["events"]:
        slot, kind = item["slot"], item["kind"]
        if kind not in ranks or (slot, ranks[kind]) <= last:
            raise ValueError("event_reference")
        last = slot, ranks[kind]
        grouped.setdefault(slot, []).append(kind)
        if kind == "defer_candidate":
            if not candidates or slot not in [64, 128, 192, 256]:
                raise ValueError("event_reference")
            candidates = {}
        elif kind == "issue_prediction":
            issued.add(slot)
            pending.add(slot)
            if len(pending) > state["capacity"]:
                pending.remove(min(pending))
            prediction = state["issued"][slot - 1]
            values = rows[slot - 1]["values"]
            for arm, head in heads.items():
                expected = (
                    None
                    if values is None
                    else scalar_probability(head, reference["geometry"], values)
                )
                observed = prediction["predictions"][arm]
                if (expected is None) != (observed is None) or (
                    expected is not None and abs(expected - observed) > 1e-12
                ):
                    raise ValueError("event_reference")
            if item["prediction_hash"] != canonical_hash(prediction["predictions"]):
                raise ValueError("event_reference")
        elif kind == "commit_candidate":
            candidates = deepcopy(item["candidates"])
            for arm, candidate in candidates.items():
                before, base = heads[arm], candidate["base"]
                if (
                    base["centers"][: len(before["centers"])] != before["centers"]
                    or base["weights"] != before["weights"] + [0.0] * 4
                    or base["intercept"] != before["intercept"]
                ):
                    raise ValueError("event_reference")
                heads[arm] = deepcopy(base)
        elif kind == "release_feedback":
            old = item.get("label_slot")
            if old != slot - 20 or old not in issued or old not in pending or old in released:
                raise ValueError("event_reference")
            pending.remove(old)
            released.add(old)
        elif kind == "admit_once":
            labels = item["labels"]
            if (
                not candidates
                or len(set(labels)) != 12
                or not set(labels) <= released
                or any(
                    bucket(rows[i - 1]) != 0 or i <= candidates["error_center"]["commit"]
                    for i in labels
                )
            ):
                raise ValueError("event_reference")
            for arm, step in item["steps"].items():
                if step:
                    heads[arm] = interpolate(candidates[arm], step)
            candidates = {}
        elif kind == "durable_update":
            if item["heads"] != heads:
                raise ValueError("event_reference")
        if set(item["pending"]) != pending or item["state_hash"] != canonical_hash(heads):
            raise ValueError("event_reference")
    for slot in range(1, state["cursor"]):
        kinds = grouped[slot]
        required = ["issue_prediction", "durable_commit"]
        if slot in [64, 128, 192]:
            required.append("commit_candidate")
        if slot > 20 and slot - 20 not in state["lost"]:
            required.append("release_feedback")
        if any(kind not in kinds for kind in required):
            raise ValueError("event_reference")
        if ("admit_once" in kinds) != ("durable_update" in kinds):
            raise ValueError("event_reference")
    if issued != set(range(1, state["cursor"])) or released != set(state["released"]):
        raise ValueError("event_reference")
    return True


def authenticate(root: Path, raw: Path, b: methods.Custody) -> tuple[Json, methods.Custody]:
    """Direct historical primitive checks need neither fit capture nor Exp8137."""
    upstream, b = historical.authenticate(root, raw, False, b)
    views = {r: methods.read_ref(b, upstream["role_manifests"][r]) for r in ["stream", "retention"]}
    direct = methods.authenticate_stream(root / methods.STREAM, b, views, False)
    for role, total, floor in [("stream", 256, 224), ("retention", 64, 48)]:
        feature = methods.read_ref(b, upstream[role + "_feature_manifest"])["rows"]
        historical.public_rows(feature, total)
        b.require(
            root / historical.UPSTREAM,
            role + "_usable_sources_floor",
            True,
            sum(r["values"] is not None for r in feature) >= floor,
        )
        b.require(
            root / historical.UPSTREAM,
            role + "_manifest_agreement",
            upstream[role + "_feature_manifest"],
            direct[role + "_feature_manifest"],
        )
        b.bind(
            Path(upstream["evaluator_label_manifests"][role]["path"]),
            upstream["evaluator_label_manifests"][role]["sha256"],
        )
    return dict(upstream, direct_stream_authentication=direct), b


def durable_record(kind: str, state: Json) -> Json:
    """Phase and source-bound hashes make interrupted prediction commits replayable."""
    return dict(
        kind=kind,
        cursor=state["cursor"],
        phase=state["phase"],
        pending=state["pending"],
        consumed=state["consumed"],
        used_admission=state["used_admission"],
        baseline_hash=state["baseline_hash"],
        heads_hash=canonical_hash(state["arms"]),
        pool_hash=canonical_hash(state["pool"]),
        issued_hash=canonical_hash(state["issued"][-1]),
        rng_hash=canonical_hash(state["rng_state"]),
    )


def measure(root: Path, raw: Path, *, fixture_mode: bool = False) -> Json:
    """Authenticate historical data independently, then check the full private schedule."""
    import time

    started = time.monotonic()
    raw.mkdir(parents=True, exist_ok=True)
    b = methods.Custody(raw / "inputs")
    work: Json = dict(
        input_ready=0,
        fixture_mode=fixture_mode,
        rows=[],
        source_artifact_hashes=[],
        raw_shard_hashes=[],
        gate_check_summary=[],
        phase_spans=[],
        cited_upstream_artifacts=[],
        stream_feature_manifest=None,
        retention_feature_manifest=None,
    )
    progress("before_input_custody")
    if fixture_mode:
        work["input_ready"] = 1
    else:
        try:
            upstream, b = authenticate(root, raw, b)
            work.update(
                input_ready=1,
                stream_feature_manifest=upstream["stream_feature_manifest"],
                retention_feature_manifest=upstream["retention_feature_manifest"],
                original_slot_mask=upstream["direct_stream_authentication"]["original_slot_mask"],
            )
            work["cited_upstream_artifacts"].append(
                dict(
                    path=str(root / historical.UPSTREAM),
                    sha256=sha256_file(root / historical.UPSTREAM),
                    scope="historical_model_receipts",
                    historical_model_provenance=upstream.get("historical_model_provenance"),
                )
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            if not b.failures:
                b.checks.append(
                    dict(
                        check="historical_stream_custody",
                        upstream="exp8111/exp8102",
                        path=str(root / historical.UPSTREAM),
                        hash=None,
                        artifact_field="authenticated_primitives",
                        op="==",
                        expected=True,
                        observed=str(error),
                        passed=False,
                    )
                )
        old = root / "results/experiment_8116_v702_independent_online_memory.json"
        if old.exists():
            import json

            previous = json.loads(old.read_text())
            work["cited_upstream_artifacts"].append(
                dict(
                    path=str(old),
                    sha256=sha256_file(old),
                    scope="historical_disqualification",
                    honest_verdict=previous.get("honest_verdict"),
                    configuration=previous.get("config"),
                    operator_override_accepted=False,
                    reason="No real dated directive supplied; unchanged V702 seal is numerical authority",
                )
            )
    work["gate_check_summary"] = b.checks + [r for r in b.failures if r not in b.checks]
    work["source_artifact_hashes"] = b.refs
    work["phase_spans"].append(dict(phase="input_custody", duration_s=time.monotonic() - started))
    progress("after_input_custody", work["input_ready"])
    rows, labels = fixture()
    inputs = shard(raw, dict(rows=rows, labels=labels))
    work["raw_shard_hashes"].append(inputs)
    work["fixture_input_manifest"] = inputs
    work["event_reference_rows"] = []
    progress("before_full_size_benchmark", 0, 20)
    for index, seed in enumerate(range(101, 121)):
        import json
        import os

        journal = raw / f"seed-{seed}-durable.jsonl"
        with journal.open("w") as stream:

            def seal(kind: str, current: Json) -> None:
                record = durable_record(kind, current)
                for key in ["arms", "pool"]:
                    record[key + "_ref"] = shard(raw, {key: current[key]})
                record["issued_ref"] = shard(raw, {"issued": current["issued"][-1]})
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

            state = run(rows, labels, seed, seal=seal)
        work["raw_shard_hashes"].append(dict(path=str(journal), sha256=sha256_file(journal)))
        ref = shard(raw, state)
        work["raw_shard_hashes"].append(ref)
        work["event_reference_rows"].append(
            dict(
                seed=seed,
                transcript=ref,
                durable_journal=dict(path=str(journal), sha256=sha256_file(journal)),
                event_count=len(state["events"]),
                passed=True,
                final_state_hash=canonical_hash(state["arms"]),
            )
        )
        for row, issued in zip(rows, state["issued"], strict=True):
            label = dict(
                y=labels[row["slot"] - 1] if row["slot"] <= 236 else None,
                exclusion_reason=None if row["slot"] <= 236 else "feedback_unresolved_tail",
            )
            work["rows"].extend(
                historical.scored(row, issued, label, "private_stream_fixture", seed)
            )
        retention, targets = fixture(64, "retention")
        sealed = [
            dict(
                slot=r["slot"],
                predictions={
                    a: probability(h, state["geometry"], r["values"])
                    for a, h in state["arms"].items()
                },
            )
            for r in retention
        ]
        # All retention predictions are sealed before evaluator-only targets open.
        for row, pred, y in zip(retention, sealed, targets, strict=True):
            pred["prediction_hash"] = canonical_hash(pred["predictions"])
            work["rows"].extend(
                historical.scored(
                    row, pred, dict(y=y, exclusion_reason=None), "private_retention_fixture", seed
                )
            )
        progress("full_size_seed_complete", index + 1, 19 - index)
    progress("after_full_size_benchmark", 20)
    work["full_size_validation_receipts"] = dict(
        passed=True,
        seeds=20,
        arms=4,
        original_slots=256,
        prediction_count=256 * 4 * 20,
        independent_fixture_sources=320,
        truth="circular_positive",
        natural_learning_benefit=0,
        reference_boundaries=sum(r["event_count"] for r in work["event_reference_rows"]),
    )
    work["method_source_map"] = [
        dict(
            url="https://arxiv.org/html/2606.11711v1",
            use="Finite pending capacity and permanent loss of untracked feedback",
            limit="Oldest eviction and empirical guards do not inherit randomized-scheduler regret bounds",
        ),
        dict(
            url="https://arxiv.org/html/2602.02634v1",
            use="Separate issuance, delayed release and learner update events",
            limit="No continuous-time delay-reduction theorem is claimed for fitted radial heads",
        ),
    ]
    work["protocol_interpretation"] = dict(
        updates="Each eligible growth block proposes four SGD steps on newest64 released update rows",
        admission="Seal coefficient proposals; evaluate next12 unused admission source labels on issued grid probabilities",
        rejected_update="Retain incumbent coefficients; keep installed zero-weight growth centers",
        deadline="Defer incomplete candidates before next opportunity or before slot256",
        historical_override="Exp8116 operator-override wording rejected; no dated directive supplied",
    )
    work["numerical_protocol"] = dict(
        delayed_memory=protocol(), statistical_plan=methods.protocol()["statistical_plan"]
    )
    work["reductions"] = historical.reductions(work["rows"])
    work["code_config_hashes"] = {
        p: sha256_file(ROOT / p)
        for p in [
            MODULE,
            RUNNER,
            CLI,
            TEST,
            historical.MODULE,
            methods.MODULE,
            "python/carnot/verify/radial_memory_8085.py",
            "openspec/change-proposals/v702-methods-and-stream-protocol.md",
        ]
    }
    referenced = {r["path"] for r in work["raw_shard_hashes"]}
    work["raw_shard_hashes"].extend(
        dict(path=str(p), sha256=sha256_file(p))
        for p in sorted((raw / "shards").glob("*.json"))
        if str(p) not in referenced
    )
    work["duration_s"] = time.monotonic() - started
    work["phase_spans"].append(
        dict(
            phase="full_size_private_fixture",
            duration_s=work["duration_s"] - work["phase_spans"][0]["duration_s"],
        )
    )
    atomic_json(raw / "measurement.json", work)
    progress("measurement_normal_exit", 20)
    return work


def build(work: Json, raw: Path, receipts: list[Json]) -> Json:
    """A qualified fixture proves protocol readiness, never natural learning benefit."""
    owned = bool(receipts) and all(r["passed"] for r in receipts)
    ready = int(owned and work["input_ready"] and work["full_size_validation_receipts"]["passed"])
    verdict = (
        "disqualified"
        if not owned
        else "blocked"
        if not work["input_ready"]
        else "circular_positive"
    )
    failed = next(
        (r["check"] for r in work["gate_check_summary"] if not r["passed"]),
        "protocol_fixture_ready",
    )
    value = dict(
        work,
        experiment_id=8138,
        task_id="exp8138-learning-protocol",
        milestone="2026.10.704",
        honest_verdict="complete_" + verdict + "_" + ("owned_validation" if not owned else failed),
        verdict_class=verdict,
        verifier_is_oracle=True,
        learning_protocol_ready_score=ready,
        required_checks_passed=owned,
        flagged_adversarial=False,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        call_ledger=[],
        model_invocation_counts=dict(historical.ZERO_INVOCATION_COUNTS),
        trained_head_specs=[
            dict(
                kind="small_Gaussian_residual",
                scope="private_fixture_only",
                arms=ARMS[1:],
                seeds=list(range(101, 121)),
                optimizer="four_clipped_SGD_steps",
            )
        ],
        claim_scope="Historical input custody and full-size private protocol conformance; no natural learning outcome",
        exposure_scope="private_circular_fixture_and_exposed_historical_development",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        intended_count=320,
        eligible_count=300,
        independent_count=0,
        completed_count=300,
        excluded_count=0,
        censored_count=20,
        failed_count=0,
        sample_size_budget=dict(
            stream=256,
            retention=64,
            arms=4,
            seeds=20,
            independent_natural_outcome_sources=0,
            fixture_sources=320,
        ),
        run_date="20261004",
        random_seed=101,
        acceptance_gates=dict(
            custody="Direct Exp8111/Exp8102 primitive custody; stream>=224 retention>=48",
            conformance="256 original slots,4 arms,20 seeds; independent event and numerical reference",
            validation="All owned checks normal exit; full-size strict validators <=300 seconds",
        ),
        methodology_note="No language model is loaded. Four-step residual proposals use released update-only labels. "
        "Twelve future source-bucket admission labels select the largest safe step. End-of-stream does not flush. "
        "All measured learner outcomes are private circular fixtures; historical Qwen evidence is cited only.",
    )
    value["full_size_validation_receipts"] = dict(
        work["full_size_validation_receipts"],
        terminal_checks=[
            r for r in receipts if r.get("name") in ["adversarial_verify", "strict_row_lint"]
        ],
    )
    value["field_principles"] = {
        k: "Exact finite receipt; no independent generalization credit." for k in value
    }
    value["field_principles"].update(
        learning_protocol_ready_score="Independent historical custody plus full fixture conformance and owned coverage.",
        verdict_class="External operand failures block; owned failures disqualify.",
        rows="Seeds repeat original fixture sources and add no independent natural evidence.",
        numerical_protocol="Unchanged V702 seal; no unsubstantiated operator override.",
        model_invocation_counts="Current zero calls are separate from historical Qwen provenance.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Reexecute sealed transcripts and independently reduce rows to reject byte drift."""
    import json

    try:
        value = json.loads(path.read_text())
        checksum = value.pop("reproducibility_checksum")
        if (
            checksum != canonical_hash(value)
            or value["numerical_protocol"]["delayed_memory"] != protocol()
        ):
            return False
        for p, digest in value["code_config_hashes"].items():
            if sha256_file(ROOT / p) != digest:
                return False
        for ref in value["raw_shard_hashes"] + value["source_artifact_hashes"]:
            target = Path(ref.get("snapshot_path", ref["path"]))
            if sha256_file(target) != ref["sha256"]:
                return False
        if historical.reductions(value["rows"]) != value["reductions"]:
            return False
        inputs = json.loads(Path(value["fixture_input_manifest"]["path"]).read_text())
        recomputed_rows: list[Json] = []
        for replay_index, row in enumerate(value["event_reference_rows"]):
            state = json.loads(Path(row["transcript"]["path"]).read_text())
            verify_events(state, inputs["rows"])
            for public, prediction in zip(inputs["rows"], state["issued"], strict=True):
                slot = public["slot"]
                label = dict(
                    y=inputs["labels"][slot - 1] if slot <= 236 else None,
                    exclusion_reason=None if slot <= 236 else "feedback_unresolved_tail",
                )
                recomputed_rows.extend(
                    historical.scored(
                        public, prediction, label, "private_stream_fixture", row["seed"]
                    )
                )
            retention, targets = fixture(64, "retention")
            for public, y in zip(retention, targets, strict=True):
                predictions = {
                    a: scalar_probability(h, state["geometry"], public["values"])
                    for a, h in state["arms"].items()
                }
                # Preserve the issued vectorized bytes; compare independent scalar values above.
                saved = {
                    a: probability(h, state["geometry"], public["values"])
                    for a, h in state["arms"].items()
                }
                if any(abs(saved[a] - predictions[a]) > 1e-12 for a in ARMS):
                    return False
                prediction = dict(predictions=saved, prediction_hash=canonical_hash(saved))
                recomputed_rows.extend(
                    historical.scored(
                        public,
                        prediction,
                        dict(y=y, exclusion_reason=None),
                        "private_retention_fixture",
                        row["seed"],
                    )
                )
            journal_rows = [
                json.loads(line)
                for line in Path(row["durable_journal"]["path"]).read_text().splitlines()
            ]
            offset = 0

            def check_durable(kind: str, current: Json) -> None:
                nonlocal offset
                expected = durable_record(kind, current)
                observed = journal_rows[offset]
                if any(observed[k] != v for k, v in expected.items()):
                    raise ValueError("durable_reference")
                offset += 1

            replayed = run(inputs["rows"], inputs["labels"], row["seed"], seal=check_durable)
            if offset != len(journal_rows):
                return False
            if canonical_hash(replayed) != canonical_hash(state):
                return False
            progress(
                "cold_seed_complete",
                replay_index + 1,
                len(value["event_reference_rows"]) - replay_index - 1,
            )

        def unit_key(r: Json) -> tuple[Any, ...]:
            return (r["seed"], r["condition"], r["slot"], r["arm"], r["metric"])

        if sorted(value["rows"], key=unit_key) != sorted(recomputed_rows, key=unit_key):
            return False
        expected = build(
            {k: v for k, v in value.items() if k != "field_principles"},
            Path(value["terminal_validation_sidecar_path"]).parent,
            value["validation_receipts"],
        )
        if expected["learning_protocol_ready_score"] != value["learning_protocol_ready_score"]:
            return False
        return True
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False
