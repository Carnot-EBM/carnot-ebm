"""REQ-VERIFY-8221: finite corrections preserve frozen public membership.

The probability tree retains each clipping operation and admitted mixture.
This makes static lookup and delayed restart use exactly the same arithmetic.
"""

from __future__ import annotations

from copy import deepcopy
import math
import random
from typing import Any, Callable

from carnot.reporting.current_work_receipt import canonical_hash
from carnot.verify import learning_protocol_8138 as roles
from carnot.verify import restricted_action_rule_8207 as rule
from carnot.verify import utility_patch_methods_8219 as frozen

Json = dict[str, Any]
GROUPS: list[Json] = frozen.PROTOCOL_VALUE["witness_dictionary"]
SCHEMA: Json = dict(
    version=1,
    probability_tree=["input", "patch", "mixture"],
    state_fields=[
        "seed",
        "cursor",
        "phase",
        "baseline_hash",
        "model",
        "pending",
        "issued",
        "released",
        "consumed",
        "training",
        "used_admission",
        "pool",
        "candidate",
        "events",
        "rng_state",
    ],
    event_order=["issue", "durable", "release", "admit", "expire", "propose"],
    global_parameters="Consumer supplies authenticated base probabilities; no generator mutation.",
)


def clip(p: float) -> float:
    """Each patch clips immediately because reversing two deltas need not cancel."""
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("probability")
    return min(1 - 1e-6, max(1e-6, p))


def predict(model: Json, row: Json) -> float | None:
    """Evaluate saved final-probability mixtures without scaling their deltas."""
    if row["p"] is None:
        return None
    if model["kind"] == "input":
        return clip(row["p"])
    p = predict(model["base"], row)
    assert p is not None
    if model["kind"] == "mixture":
        q = predict(model["candidate"], row)
        assert q is not None
        return (1 - model["step"]) * p + model["step"] * q
    if model["kind"] != "patch":
        raise ValueError("model_kind")
    for operation in model["patches"]:
        # Membership reads the original baseline even after earlier corrections.
        p = min(
            1 - 1e-6, max(1e-6, p + operation["delta"] * frozen.member(row, operation["group"]))
        )
    return p


def energies(p: float) -> tuple[float, float]:
    """Energy normalization is another encoding of the same probability."""
    p = clip(p)
    return -math.log1p(-p), -math.log(p)


def fit_patches(
    rows: list[Json],
    *,
    model: Json | None = None,
    groups: list[Json] | None = None,
    mode: str = "group",
    seed: int = 101,
    maximum: int = 4,
) -> Json:
    """Only supported frozen witnesses compete; missing rows retain the denominator."""
    selected = GROUPS if groups is None else groups
    if any(g not in GROUPS for g in selected) or len({g["name"] for g in selected}) != len(
        selected
    ):
        raise ValueError("group")
    if mode not in ["original", "global", "group", "local", "random"] or not 0 <= maximum <= 4:
        raise ValueError("mode")
    selected = [
        g
        for g in GROUPS
        if g in selected
        and (mode != "global" or g["name"] == "global")
        and (mode != "local" or g["name"] != "global")
    ]
    result = dict(kind="patch", base=deepcopy(model or dict(kind="input")), patches=[])
    rng = random.Random(seed)
    for _ in range(0 if mode == "original" else maximum):
        scored = [dict(r, p=predict(result, r)) for r in rows]
        witnesses = [r for r in frozen.residuals(scored, selected) if r["eligible"]]
        if not witnesses:
            break
        best = (
            rng.choice(witnesses)
            if mode == "random"
            else max(witnesses, key=lambda r: abs(r["probability_residual"]))
        )
        if abs(best["probability_residual"]) <= 0.01:
            break
        group = next(g for g in selected if g["name"] == best["group"])
        delta = min(0.05, max(-0.05, 0.5 * best["numerator"] / best["available_members"]))
        result["patches"].append(dict(group=deepcopy(group), delta=delta, witness=best))
    return result


def losses(records: list[Json], probabilities: list[float | None]) -> tuple[float, float, int]:
    """The original accept permission and missing escalation remain safety gates."""
    actions = [
        rule.action(p, r["baseline_action"]) for p, r in zip(probabilities, records, strict=True)
    ]
    available = [(p, r["y"]) for p, r in zip(probabilities, records, strict=True) if p is not None]
    return (
        math.fsum(rule.base.loss(a, r["y"]) for a, r in zip(actions, records, strict=True))
        / len(records),
        math.fsum((p - y) ** 2 for p, y in available) / len(available) if available else math.inf,
        sum(a == "accept" and r["y"] == 1 for a, r in zip(actions, records, strict=True)),
    )


def admission(candidate: Json, installed: Json, records: list[Json]) -> tuple[Json, Json]:
    """Twelve future labels choose a grid point but never change fitted operations."""
    if (
        len(records) != 12
        or any(r["y"] not in [0, 1] for r in records)
        or min(sum(r["y"] == y for r in records) for y in [0, 1]) < 2
    ):
        return installed, dict(step=0, gates=[])
    incumbent = losses(records, [predict(installed, r) for r in records])
    baseline = losses(records, [predict(dict(kind="input"), r) for r in records])
    gates = []
    for step in [1.0, 0.5, 0.25, 0.125]:
        mixture = dict(
            kind="mixture", base=deepcopy(installed), candidate=deepcopy(candidate), step=step
        )
        result = losses(records, [predict(mixture, r) for r in records])
        passed = (
            result[0] <= incumbent[0]
            and result[1] <= incumbent[1]
            and result[0] <= baseline[0] + 0.02
            and result[1] <= baseline[1] + 0.01
            and result[2] <= incumbent[2]
            and result[2] <= baseline[2]
        )
        gates.append(
            dict(
                step=step,
                result=list(result),
                incumbent=list(incumbent),
                frozen=list(baseline),
                passed=passed,
            )
        )
        if passed:
            return mixture, dict(step=step, gates=gates)
    return installed, dict(step=0, gates=gates)


def fixture(case: str) -> tuple[list[Json], list[int | None]]:
    """Known private probabilities move decisions without natural-data benefit credit."""
    rows, labels = [], []
    admission_slots = set(range(65, 77)) | set(range(145, 157))
    if case == "late_label":
        admission_slots = set(range(120, 132))
    for slot in range(1, 257):
        y = slot % 2
        nonce = 0
        row = dict(
            slot=slot,
            unit_id=f"private-{slot}",
            source_cluster_id=f"private-{slot}",
            p=0.09 if y == 0 else 0.49,
            baseline_p=0.09 if y == 0 else 0.49,
            baseline_action="accept" if y == 0 else "escalate",
        )
        while True:
            row["source_id"] = f"fixture-{slot}-{nonce}"
            if (roles.bucket(row) == 0) == (slot in admission_slots):
                break
            nonce += 1
        if case == "no_signal":
            row["p"] = float(y)
        if case == "missing" and slot % 5 == 0:
            row["p"] = None
        rows.append(row)
        labels.append(
            0
            if case == "single_class"
            else 1 - y
            if case == "rejected" and slot in admission_slots
            else y
        )
    return rows, labels


def run(
    rows: list[Json],
    labels: list[int | None],
    seed: int,
    *,
    state: Json | None = None,
    seal: Callable[[str, Json], None] | None = None,
) -> Json:
    """Issue before release so genuine hard exits preserve predictions still awaiting labels."""
    if len(rows) != 256 or len(labels) != 256 or [r["slot"] for r in rows] != list(range(1, 257)):
        raise ValueError("stream_schema")
    if any(y not in [0, 1, None] for y in labels) or len({r["unit_id"] for r in rows}) != 256:
        raise ValueError("stream_schema")
    s = (
        deepcopy(state)
        if state is not None
        else dict(
            seed=seed,
            cursor=1,
            phase="issue",
            baseline_hash=canonical_hash(rows),
            model=dict(kind="input"),
            pending=[],
            issued=[],
            released=[],
            consumed=[],
            training=[],
            used_admission=[],
            pool=[],
            candidate=None,
            events=[],
            rng_state=json_rng(seed),
            schema_version=1,
            public_warmup_hash=canonical_hash(rows[:64]),
        )
    )
    if s["baseline_hash"] != canonical_hash(rows) or s["seed"] != seed:
        raise ValueError("baseline_hash")
    while s["cursor"] <= 256:
        slot = s["cursor"]
        if s["phase"] == "issue":
            row = rows[slot - 1]
            p = predict(s["model"], row)
            s["issued"].append(
                dict(
                    slot=slot,
                    p=p,
                    action=rule.action(p, row["baseline_action"]),
                    model_sha256=canonical_hash(s["model"]),
                )
            )
            s["pending"].append(slot)
            s["events"].append(
                dict(
                    kind="durable_commit",
                    slot=slot,
                    pending=list(s["pending"]),
                    model_sha256=canonical_hash(s["model"]),
                )
            )
            s["phase"] = "release"
            if seal:
                seal("durable_commit", s)
        old = slot - 20
        if old in s["pending"]:
            s["pending"].remove(old)
            s["released"].append(old)
            s["consumed"].append(old)
            row, y = rows[old - 1], labels[old - 1]
            s["events"].append(dict(kind="release_feedback", slot=slot, label_slot=old, y=y))
            if y is not None and row["p"] is not None or not roles.bucket(row):
                if roles.bucket(row):
                    s["training"].append(old)
                    s["pool"].append(dict(row, y=y))
                elif (
                    s["candidate"]
                    and old > s["candidate"]["commit"]
                    and old not in s["used_admission"]
                ):
                    s["used_admission"].append(old)
                    candidate = s["candidate"]
                    candidate["labels"].append(dict(row, y=y))
                    if len(candidate["labels"]) == 12:
                        before = canonical_hash(s["model"])
                        s["model"], detail = admission(
                            candidate["model"], s["model"], candidate["labels"]
                        )
                        s["events"].append(
                            dict(
                                kind="admit_once",
                                slot=slot,
                                **detail,
                                labels=[r["slot"] for r in candidate["labels"]],
                                before_hash=before,
                                after_hash=canonical_hash(s["model"]),
                                mixture=deepcopy(s["model"]),
                            )
                        )
                        s["candidate"] = None
        if slot in [144, 224] and s["candidate"]:
            s["events"].append(dict(kind="defer_candidate", slot=slot, reason="admission_deadline"))
            s["candidate"] = None
        if slot in [64, 144]:
            pool = s["pool"][-64:]
            eligible = min(sum(r["y"] == y for r in pool) for y in [0, 1]) >= 2
            proposed = fit_patches(pool, model=s["model"], seed=seed) if eligible else None
            s["candidate"] = (
                dict(model=proposed, commit=slot, labels=[])
                if proposed and (proposed["patches"] or proposed.get("global_fit_changed"))
                else None
            )
            s["events"].append(
                dict(
                    kind="commit_candidate",
                    slot=slot,
                    fit_ids=[r["slot"] for r in pool],
                    candidate=deepcopy(s["candidate"]),
                    eligible=eligible,
                )
            )
        s["cursor"] += 1
        s["phase"] = "issue"
        if slot % 32 == 0:
            print(
                f"[exp8221] phase=fixture_slots completed={slot} pending={256 - slot}", flush=True
            )
    return s


def json_rng(seed: int) -> list[Any]:
    """Save the exact RNG state even when deterministic witness selection uses no draws."""
    import json

    return list(json.loads(json.dumps(random.Random(seed).getstate())))


def summary(state: Json) -> Json:
    """Only changed actions strictly after an admitted state qualify future use."""
    installs = [r["slot"] for r in state["events"] if r["kind"] == "admit_once" and r["step"]]
    first = min(installs, default=257)
    rows, _ = fixture("learnable")
    changed = sum(
        r["slot"] > first
        and r["action"]
        != rule.action(rows[r["slot"] - 1]["p"], rows[r["slot"] - 1]["baseline_action"])
        for r in state["issued"]
    )
    return dict(
        install_slot=first,
        changed_later_decisions=changed,
        passed=first <= 208 and changed >= 32,
        admission_fit_overlap=sorted(set(state["training"]) & set(state["used_admission"])),
    )
