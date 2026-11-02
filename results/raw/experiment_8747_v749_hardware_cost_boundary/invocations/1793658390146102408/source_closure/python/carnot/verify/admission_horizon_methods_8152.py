"""REQ-VERIFY-8152: qualify useful future exposure without new natural outcomes.

Only scheduling changes. The qualified trainer and admission guards remain the
numerical authority; private learnable targets can qualify mechanics only.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random
import time
from typing import Any, Callable

import numpy as np

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.verify import learning_protocol_8138 as engine
from carnot.verify import learning_audit_8144 as scalar

Json = dict[str, Any]
ROOT = engine.ROOT
NAME = "experiment_8152_v705_admission_horizon_methods"
CLI = f"scripts/experiments/{NAME}.py"
MODULE = "python/carnot/verify/admission_horizon_methods_8152.py"
RUNNER = "python/carnot/reporting/admission_horizon_execution_8152.py"
TEST = "tests/python/test_admission_horizon_methods_8152.py"
PROTOCOL = "openspec/change-proposals/v705-admission-horizon-protocol.json"
PROTOCOL_HASH = "sha256:ff18d138210428fe9b13c256d17ab49762882e1e3fc80ee70aa6e53f42eb19fb"
MODEL_SPECS: list[Json] = []


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual counts so CPU work cannot look like unobserved inference."""
    print(f"[exp8152] phase={phase} completed={completed} pending={pending}", flush=True)


def protocol() -> Json:
    """Immutable numerical bytes prevent changing analysis after outcomes open."""
    if sha256_file(ROOT / PROTOCOL) != PROTOCOL_HASH:
        raise ValueError("protocol_hash")
    return json.loads((ROOT / PROTOCOL).read_text())  # type: ignore[no-any-return]


def fixture(case: str) -> tuple[list[Json], list[int | None]]:
    """Known private targets put delayed admissions on both sides of old expiry."""
    rows, _ = engine.fixture()
    labels: list[int | None] = []
    admissions = set(range(76, 121, 4)) | set(range(148, 193, 4))
    if case == "late_label":
        admissions = set(range(120, 165, 4))
    for row in rows:
        slot = row["slot"]
        y = (slot // 4) % 2
        row["values"] = [1.0 if y == 0 else -1.0, *[float(slot * j % 19) for j in range(1, 9)]]
        nonce = 0
        while True:
            row["source_id"] = f"private-{slot}-{nonce}"
            if (engine.bucket(row) == 0) == (slot in admissions):
                break
            nonce += 1
        labels.append(1 - y if case == "rejected" and slot in admissions else y)
    return rows, labels


def propose(state: Json, slot: int, opportunity: int) -> None:
    """Reserve centers by opportunity number so the new144 clock cannot skip any."""
    pool = state["pool"][-64:]
    if len(pool) < 16 or sum(abs(r["y"] - r["frozen_prediction"]) >= 0.5 for r in pool) < 4:
        engine.event(state, "commit_candidate", slot, disposition="ineligible", candidates={})
        return
    rng = random.Random(state["seed"] * 1000 + slot)
    chosen = {
        "fixed_public_center": state["reserved"][opportunity * 4 : (opportunity + 1) * 4],
        "random_past_center": rng.sample(pool, 4),
        "error_center": sorted(
            pool, key=lambda r: (-abs(r["y"] - r["frozen_prediction"]), r["source_cluster_id"])
        )[:4],
    }
    state["rng_state"] = json.loads(json.dumps(rng.getstate()))
    for arm in engine.ARMS[1:]:
        head = state["arms"][arm]
        for row in chosen[arm]:
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
            head["centers"].append(deepcopy(center))
            head["weights"].append(0.0)
        state["candidates"][arm] = dict(
            head=engine.train(head, state["geometry"], pool),
            base=deepcopy(head),
            labels=[],
            commit=slot,
        )
    engine.event(
        state,
        "commit_candidate",
        slot,
        disposition="committed",
        candidates=deepcopy(state["candidates"]),
    )
    engine.event(state, "select_future_admission", slot, minimum_original_slot=slot + 1)


def run(
    rows: list[Json],
    labels: list[int | None],
    seed: int,
    *,
    capacity: int = 32,
    state: Json | None = None,
    seal: Callable[[str, Json], None] | None = None,
) -> Json:
    """Durable issue precedes release; admissions get their expiry-slot chance."""
    engine.historical.public_rows(rows, 256)
    if len(labels) != 256:
        raise ValueError("evaluator_slots")
    s = engine.genesis(rows, seed) if state is None else state
    if s["baseline_hash"] != canonical_hash(rows):
        raise ValueError("baseline_hash")
    s["capacity"] = capacity
    while s["cursor"] <= 256:
        slot = s["cursor"]
        if s["phase"] == "issue":
            values = rows[slot - 1]["values"]
            predictions = {
                a: None if values is None else engine.probability(h, s["geometry"], values)
                for a, h in s["arms"].items()
            }
            grid = {
                a: {
                    str(t): None
                    if values is None
                    else engine.probability(engine.interpolate(c, t), s["geometry"], values)
                    for t in [1.0, 0.5, 0.25, 0.125]
                }
                for a, c in s["candidates"].items()
            }
            s["issued"].append(
                dict(
                    slot=slot,
                    predictions=predictions,
                    grid=grid,
                    prediction_hash=canonical_hash(predictions),
                )
            )
            s["pending"].append(slot)
            if len(s["pending"]) > capacity:
                s["lost"].append(s["pending"].pop(0))
            engine.event(s, "issue_prediction", slot, prediction_hash=canonical_hash(predictions))
            engine.event(s, "durable_commit", slot)
            s["phase"] = "release"
            if seal:
                seal("durable_commit", s)
        old_slot = slot - 20
        if old_slot in s["pending"]:
            if old_slot in s["consumed"] or old_slot in s["used_admission"]:
                raise ValueError("reused_label")
            s["pending"].remove(old_slot)
            s["released"].append(old_slot)
            s["consumed"].append(old_slot)
            engine.event(s, "release_feedback", slot, label_slot=old_slot)
            old, y = rows[old_slot - 1], labels[old_slot - 1]
            if y not in [0, 1, None]:
                raise ValueError("evaluator_label")
            if old["values"] is not None and y is not None:
                if engine.bucket(old):
                    s["training"].append(old_slot)
                    s["pool"].append(
                        dict(
                            old,
                            y=y,
                            frozen_prediction=s["issued"][old_slot - 1]["predictions"][
                                engine.ARMS[0]
                            ],
                        )
                    )
                elif s["candidates"] and old_slot > next(iter(s["candidates"].values()))["commit"]:
                    s["used_admission"].append(old_slot)
                    for c in s["candidates"].values():
                        c["labels"].append(
                            dict(s["issued"][old_slot - 1], y=y, label_slot=old_slot)
                        )
                    engine.admit(s, slot)
        if slot in [144, 224] and s["candidates"]:
            engine.event(s, "defer_candidate", slot, reason="admission_deadline")
            s["candidates"] = {}
        if slot in [64, 144]:
            propose(s, slot, [64, 144].index(slot))
        s["phase"] = "issue"
        s["cursor"] += 1
        if slot % 32 == 0:
            progress("fixture_slots", slot, 256 - slot)
    return s


def reference(state: Json, rows: list[Json], labels: list[int | None]) -> Json:
    """A scalar interpreter checks saved events without producer transitions."""
    initial = engine.genesis(rows, state["seed"])
    heads, geometry = initial["arms"], initial["geometry"]
    pending: list[int] = []
    pool: list[Json] = []
    released: list[int] = []
    used: list[int] = []
    lost: list[int] = []
    training: list[int] = []
    candidates: Json = {}
    events = iter(state["events"])
    rng_state = None

    def take(kind: str, slot: int) -> Json:
        item: Json = next(events, {})
        scalar.equal(
            [kind, slot, pending, canonical_hash(heads)],
            [item.get("kind"), item.get("slot"), item.get("pending"), item.get("state_hash")],
        )
        return item

    for slot, row in enumerate(rows, 1):
        issued = state["issued"][slot - 1]
        values = row["values"]
        predictions = {
            a: None if values is None else engine.scalar_probability(h, geometry, values)
            for a, h in heads.items()
        }
        grid = {
            a: {
                str(t): None
                if values is None
                else engine.scalar_probability(scalar.interpolate(c, t), geometry, values)
                for t in [1.0, 0.5, 0.25, 0.125]
            }
            for a, c in candidates.items()
        }
        scalar.equal(
            [slot, predictions, grid], [issued["slot"], issued["predictions"], issued["grid"]]
        )
        scalar.equal(canonical_hash(issued["predictions"]), issued["prediction_hash"])
        pending.append(slot)
        if len(pending) > state["capacity"]:
            lost.append(pending.pop(0))
        scalar.equal(issued["prediction_hash"], take("issue_prediction", slot)["prediction_hash"])
        take("durable_commit", slot)
        old_slot = slot - 20
        if old_slot in pending:
            pending.remove(old_slot)
            released.append(old_slot)
            scalar.equal(old_slot, take("release_feedback", slot)["label_slot"])
            old, y = rows[old_slot - 1], labels[old_slot - 1]
            if old["values"] is not None and y is not None:
                if engine.bucket(old):
                    training.append(old_slot)
                    pool.append(
                        dict(
                            old,
                            y=y,
                            frozen_prediction=state["issued"][old_slot - 1]["predictions"][
                                engine.ARMS[0]
                            ],
                        )
                    )
                elif candidates and old_slot > next(iter(candidates.values()))["commit"]:
                    used.append(old_slot)
                    for c in candidates.values():
                        c["labels"].append(
                            dict(state["issued"][old_slot - 1], y=y, label_slot=old_slot)
                        )
                    records = next(iter(candidates.values()))["labels"]
                    if len(records) == 12:
                        steps = scalar.admission(candidates)
                        for a, step in steps.items():
                            if step:
                                heads[a] = scalar.interpolate(candidates[a], step)
                        admit, update = next(events), next(events)
                        scalar.equal(
                            [steps, [r["label_slot"] for r in records]],
                            [admit["steps"], admit["labels"]],
                        )
                        scalar.equal(heads, update["heads"])
                        heads = deepcopy(update["heads"])
                        for item, kind in [(admit, "admit_once"), (update, "durable_update")]:
                            scalar.equal(
                                [kind, slot, pending, canonical_hash(heads)],
                                [item["kind"], item["slot"], item["pending"], item["state_hash"]],
                            )
                        candidates = {}
        if slot in [144, 224] and candidates:
            scalar.equal("admission_deadline", take("defer_candidate", slot)["reason"])
            candidates = {}
        if slot in [64, 144]:
            # The scalar historical proposal gets the same opportunity slices at64/144.
            rebuilt = scalar.proposal(
                heads, geometry, initial["reserved"], pool, state["seed"], slot
            )
            item = next(events)
            scalar.equal(rebuilt, item["candidates"])
            candidates = deepcopy(item["candidates"])
            for a, c in candidates.items():
                heads[a] = deepcopy(c["base"])
            scalar.equal(
                [
                    "commit_candidate",
                    slot,
                    pending,
                    canonical_hash(heads),
                    "committed" if candidates else "ineligible",
                ],
                [
                    item["kind"],
                    item["slot"],
                    item["pending"],
                    item["state_hash"],
                    item["disposition"],
                ],
            )
            if candidates:
                rng = random.Random(state["seed"] * 1000 + slot)
                rng.sample(pool[-64:], 4)
                rng_state = json.loads(json.dumps(rng.getstate()))
                scalar.equal(
                    slot + 1, take("select_future_admission", slot)["minimum_original_slot"]
                )
    scalar.equal(None, next(events, None))
    for key, expected in dict(
        arms=heads,
        pending=pending,
        pool=pool,
        released=released,
        consumed=released,
        used_admission=used,
        lost=lost,
        training=training,
        candidates=candidates,
        rng_state=rng_state,
        cursor=257,
        phase="issue",
        geometry=geometry,
        reserved=initial["reserved"],
        baseline_hash=canonical_hash(rows),
    ).items():
        scalar.equal(expected, state[key])
    return dict(
        passed=True,
        event_count=len(state["events"]),
        release_count=len(released),
        overflow_count=len(lost),
        admission_reuse_count=len(used) - len(set(used)),
    )


def exposure(state: Json, rows: list[Json]) -> Json:
    """Count actual future issued differences, rather than assuming installation helps."""
    installations = [
        e["slot"] for e in state["events"] if e["kind"] == "admit_once" and any(e["steps"].values())
    ]
    first = min(installations, default=257)
    slots = [
        r["slot"]
        for r, p in zip(rows, state["issued"], strict=True)
        if r["slot"] > first
        and r["values"] is not None
        and abs(p["predictions"]["error_center"] - p["predictions"][engine.ARMS[0]]) > 1e-12
    ]
    return dict(
        installation_slot=first if installations else None,
        usable_changed_later_count=len(slots),
        later_issue_slots=slots,
        passed=first <= 208 and len(slots) >= 32,
    )


def authenticate(root: Path, raw: Path, binder: engine.methods.Custody) -> Json:
    """Historical code belongs to its sealed snapshots, not today's corrected files."""
    methods = engine.methods
    binder.upstream = "exp8111-methods-and-stream-custody"
    path = root / engine.historical.UPSTREAM
    value = binder.read(path, engine.historical.UPSTREAM_HASH)
    terminal = methods.historical.terminal(path, value, binder)
    side = binder.read(Path(value["terminal_validation_sidecar_path"]))
    for field, expected in [
        ("methods_ready_score", 1),
        ("stream_input_ready_score", 1),
        ("required_checks_passed", True),
        ("flagged_adversarial", False),
    ]:
        binder.require(path, field, expected, value.get(field))
    binder.require(path, "normal_process_exit", True, side.get("normal_process_exit"))
    binder.require(path, "terminal.report.passed", True, terminal["report"].get("passed"))
    for ref in value["raw_shard_hashes"]:
        binder.bind(Path(ref["path"]), ref["sha256"])
    for name, digest in value["code_config_hashes"].items():
        ref = next(
            r
            for r in value["source_artifact_hashes"]
            if r["path"] == str(ROOT / name) and r["sha256"] == digest
        )
        binder.bind(Path(ref["snapshot_path"]), digest)
    binder.bind(methods.DESIGN)
    views = {
        role: methods.read_ref(binder, value["role_manifests"][role])
        for role in ["stream", "retention"]
    }
    direct = methods.authenticate_stream(root / methods.STREAM, binder, views, False)
    for role, total, floor in [("stream", 256, 224), ("retention", 64, 48)]:
        rows = methods.read_ref(binder, direct[role + "_feature_manifest"])["rows"]
        engine.historical.public_rows(rows, total)
        observed = sum(r["values"] is not None for r in rows)
        check = dict(
            check=role + "_usable_sources_floor",
            upstream="exp8102",
            path=str(root / methods.STREAM),
            hash=methods.UPSTREAM_HASHES[8102],
            artifact_field=role + "_usable_sources",
            op=">=",
            expected=floor,
            observed=observed,
            passed=observed >= floor,
        )
        binder.checks.append(check)
        if not check["passed"]:
            binder.failures.append(check)
            raise ValueError(check["check"])
        binder.bind(
            Path(value["evaluator_label_manifests"][role]["path"]),
            value["evaluator_label_manifests"][role]["sha256"],
        )
        binder.require(
            path,
            role + "_manifest_agreement",
            value[role + "_feature_manifest"],
            direct[role + "_feature_manifest"],
        )
    return dict(value, direct_stream_authentication=direct)


def history(root: Path, binder: engine.methods.Custody) -> tuple[list[Json], Json]:
    """Read V704 saved event clocks and issuing heads; never refit historical heads."""
    path = root / scalar.UPSTREAM
    binder.upstream = "exp8143-delayed-energy-memory"
    value = binder.read(path, scalar.UPSTREAM_HASH)
    terminal = engine.methods.historical.terminal(path, value, binder)
    for field, expected in [
        ("learning_trajectory_ready_score", 1),
        ("required_checks_passed", True),
        ("flagged_adversarial", False),
    ]:
        binder.require(path, field, expected, value.get(field))
    binder.require(path, "terminal.report.passed", True, terminal["report"].get("passed"))
    result = []
    for index, entry in enumerate(value["state_manifest"]):
        state = engine.methods.read_ref(binder, entry["state"])
        commits = [r for r in state["events"] if r["kind"] == "commit_candidate"]
        arrivals = [r for r in state["events"] if r["kind"] == "release_feedback"]
        for candidate in commits:
            commit = candidate["slot"]
            expiry = next(
                (
                    r["slot"]
                    for r in state["events"]
                    if r["kind"] == "defer_candidate" and r["slot"] > commit
                ),
                None,
            )
            admission = next(
                (
                    r
                    for r in state["events"]
                    if r["kind"] == "admit_once"
                    and r["slot"] > commit
                    and (expiry is None or r["slot"] < expiry)
                ),
                None,
            )
            later = [
                p["slot"]
                for p in state["issued"]
                if admission
                and p["slot"] > admission["slot"]
                and p["predictions"]["error_center"] is not None
            ]
            result.append(
                dict(
                    seed=entry["seed"],
                    commitment_slot=commit,
                    disposition=candidate["disposition"],
                    expiry_slot=expiry if admission is None else None,
                    installation_slot=admission["slot"] if admission else None,
                    admission_label_slots=admission["labels"] if admission else [],
                    label_arrivals=[
                        dict(issue_slot=r["label_slot"], release_slot=r["slot"])
                        for r in arrivals
                        if commit < r["label_slot"]
                    ],
                    later_issue_slots=later,
                    usable_resolved_later_slots=[s for s in later if s <= 236],
                    primitive_state=entry["state"],
                )
            )
        progress("historical_seed_read", index + 1, len(value["state_manifest"]) - index - 1)
    return result, value


def measure(root: Path, raw: Path, *, fixture_mode: bool = False) -> Json:
    """Seal the protocol before authenticating inputs, then measure private mechanics."""
    import sys
    import yaml

    started = time.monotonic()
    progress("before_preconditions")
    numerical = protocol()
    raw.mkdir(parents=True, exist_ok=True)
    atomic_json(raw / "protocol.json", numerical)
    binder = engine.methods.Custody(raw / "inputs")
    work: Json = dict(
        input_ready=0,
        fixture_mode=fixture_mode,
        rows=[],
        event_reference_rows=[],
        historical_installation_rows=[],
        source_artifact_hashes=[],
        raw_shard_hashes=[],
        phase_spans=[],
        numerical_protocol=numerical,
        protocol_path=str(raw / "protocol.json"),
        protocol_sha256=sha256_file(raw / "protocol.json"),
        method_map=numerical["method_map"],
        preconditions_checked=dict(
            runtime_executable=sys.executable,
            python_version=sys.version,
            model_loads=0,
            protocol_hash=PROTOCOL_HASH,
        ),
    )
    try:
        if fixture_mode:
            work["input_ready"] = 1
        else:
            exclusion = root / "ops/exclusion_manifest.yaml"
            binder.bind(exclusion)
            retired = yaml.safe_load(exclusion.read_text()).get("retired_experiments", [])
            binder.require(
                exclusion,
                "experiment_not_retired",
                True,
                not any(r.get("experiment_id") == 8152 for r in retired),
            )
            for tool in ["python", "pytest", "coverage", "ruff", "mypy"]:
                binder.require(
                    root / ".venv/bin" / tool,
                    "runtime_" + tool,
                    True,
                    (root / ".venv/bin" / tool).is_file(),
                )
            upstream = authenticate(root, raw, binder)
            work["input_manifests"] = {
                k: upstream[k]
                for k in [
                    "stream_feature_manifest",
                    "retention_feature_manifest",
                    "evaluator_label_manifests",
                ]
            }
            work["original_slot_mask"] = upstream["direct_stream_authentication"][
                "original_slot_mask"
            ]
            work["historical_model_provenance"] = upstream["direct_stream_authentication"][
                "historical_model_provenance"
            ]
            work["historical_installation_rows"], old = history(root, binder)
            for n, expected in [
                (8138, "learning_protocol_ready_score"),
                (8144, "learning_audit_ready_score"),
            ]:
                path = next((root / "results").glob(f"experiment_{n}_*.json"))
                value = binder.read(path)
                binder.require(path, expected, 1, value.get(expected))
            binder.require(
                root / scalar.UPSTREAM,
                "historical_protocol",
                engine.protocol(),
                old["numerical_protocol"]["delayed_memory"],
            )
            for method in work["method_map"]:
                path = (
                    root / "results/raw" / NAME / "method_sources" / (method["arxiv"] + "v1.html")
                )
                binder.bind(path)
            work["input_ready"] = 1
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        if not binder.failures:
            binder.checks.append(
                dict(
                    check="historical_custody",
                    upstream="exp8102/8111/8143",
                    path=str(root),
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
    work["phase_spans"].append(
        dict(phase="preconditions_and_historical_custody", duration_s=time.monotonic() - started)
    )
    progress("after_preconditions", work["input_ready"])
    cases = [("positive", seed) for seed in range(101, 121)] + [
        (case, 101) for case in ["rejected", "late_label", "overflow"]
    ]
    progress("before_private_head_benchmark", 0, len(cases))
    for index, (case, seed) in enumerate(cases):
        rows, labels = fixture(case)
        inputs = engine.shard(raw, dict(rows=rows, labels=labels))
        state = run(rows, labels, seed, capacity=8 if case == "overflow" else 32)
        checked = reference(state, rows, labels)
        transcript = engine.shard(raw, state)
        work["event_reference_rows"].append(
            dict(
                case=case,
                seed=seed,
                inputs=inputs,
                transcript=transcript,
                **checked,
                exposure=exposure(state, rows),
            )
        )
        if case == "positive":
            for row, issued in zip(rows, state["issued"], strict=True):
                label = dict(
                    y=labels[row["slot"] - 1] if row["slot"] <= 236 else None,
                    exclusion_reason=None if row["slot"] <= 236 else "feedback_unresolved_tail",
                )
                work["rows"].extend(
                    engine.historical.scored(row, issued, label, "private_learnable_fixture", seed)
                )
            retention, targets = engine.fixture(64, "retention")
            sealed = [
                dict(
                    predictions={
                        a: engine.probability(h, state["geometry"], row["values"])
                        for a, h in state["arms"].items()
                    }
                )
                for row in retention
            ]
            retention_ref = engine.shard(raw, dict(rows=sealed))
            work["event_reference_rows"][-1]["retention_predictions"] = retention_ref
            for row, pred, y in zip(retention, sealed, targets, strict=True):
                pred["prediction_hash"] = canonical_hash(pred["predictions"])
                work["rows"].extend(
                    engine.historical.scored(
                        row,
                        pred,
                        dict(y=y, exclusion_reason=None),
                        "private_retention_fixture",
                        seed,
                    )
                )
        progress("private_case_complete", index + 1, len(cases) - index - 1)
    rows, labels = fixture("positive")
    old = engine.run(rows, labels, 101)
    work["old_expiry_fixture"] = dict(
        transcript=engine.shard(raw, old),
        passed=not any(
            any(r["steps"].values()) for r in old["events"] if r["kind"] == "admit_once"
        ),
        claim="old finite-horizon failure, no natural benefit conclusion",
    )
    checkpoint: Json = {}

    def crash(kind: str, current: Json) -> None:
        if current["cursor"] == 144:
            checkpoint.update(deepcopy(current))
            raise RuntimeError("private_crash")

    try:
        run(rows, labels, 101, seal=crash)
    except RuntimeError:
        pass
    checkpoint_ref = engine.shard(raw, checkpoint)
    resumed = run(rows, labels, 101, state=checkpoint)
    baseline = json.loads(Path(work["event_reference_rows"][0]["transcript"]["path"]).read_text())
    work["restart_fixture"] = dict(
        passed=resumed == baseline,
        checkpoint=checkpoint_ref,
        resumed_transcript=engine.shard(raw, resumed),
        scalar_reference=reference(resumed, rows, labels),
    )
    progress("after_private_head_benchmark", len(cases))
    work["reductions"] = engine.historical.reductions(work["rows"])
    work["code_config_hashes"] = {
        p: sha256_file(ROOT / p)
        for p in [
            MODULE,
            RUNNER,
            CLI,
            TEST,
            PROTOCOL,
            engine.MODULE,
            scalar.MODULE,
            engine.historical.MODULE,
            engine.methods.MODULE,
            "python/carnot/verify/radial_memory_8085.py",
            "openspec/change-proposals/v702-methods-and-stream-protocol.md",
        ]
    }
    work["raw_shard_hashes"] = [
        dict(path=str(p), sha256=sha256_file(p))
        for p in sorted(raw.rglob("*.json"))
        if "inputs" not in p.relative_to(raw).parts
    ]
    work["duration_s"] = time.monotonic() - started
    work["phase_spans"].append(
        dict(
            phase="private_reference_qualification",
            duration_s=work["duration_s"] - work["phase_spans"][0]["duration_s"],
        )
    )
    atomic_json(raw / "measurement.json", work)
    progress("measurement_normal_exit", len(cases))
    return work


def build(work: Json, raw: Path, receipts: list[Json]) -> Json:
    """Fixture success qualifies methods only; external gates and owned failures differ."""
    owned = bool(receipts) and all(r["passed"] and r.get("normal_exit", True) for r in receipts)
    input_ready = work["input_ready"] and all(r["passed"] for r in work["gate_check_summary"])
    conformance = (
        all(r["passed"] for r in work["event_reference_rows"])
        and work["old_expiry_fixture"]["passed"]
        and work["restart_fixture"]["passed"]
    )
    future = int(
        owned
        and conformance
        and all(
            r["exposure"]["passed"] for r in work["event_reference_rows"] if r["case"] == "positive"
        )
    )
    verdict = (
        "disqualified"
        if not owned or not conformance or not future
        else "blocked"
        if not input_ready
        else "circular_positive"
    )
    failed = next(
        (r["check"] for r in work["gate_check_summary"] if not r["passed"]),
        "release_aware_methods_ready",
    )
    counts = {
        status: sum(
            r["status"] == status
            for r in work["rows"]
            if r["seed"] == 101 and r["arm"] == engine.ARMS[0] and r["metric"] == "brier"
        )
        for status in ["completed", "excluded", "censored"]
    }
    value = dict(
        work,
        experiment_id=8152,
        task_id="exp8152-admission-horizon-methods",
        milestone="2026.10.705",
        honest_verdict="complete_"
        + verdict
        + "_"
        + ("owned_validation" if verdict == "disqualified" else failed),
        verdict_class=verdict,
        learning_protocol_ready_score=int(owned and future and input_ready),
        future_exposure_fixture_score=future,
        verifier_is_oracle=True,
        claim_scope="Release-aware methods and learnable private fixture causality; no new natural outcome or benefit",
        exposure_scope="private_circular_fixture_and_exposed_historical_development",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned and conformance,
        flagged_adversarial=False,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=MODEL_SPECS,
        trained_head_specs=[
            dict(work["numerical_protocol"]["trained_head_specs"], scope="private_fixture_only")
        ],
        model_invocation_counts=dict(engine.historical.ZERO_INVOCATION_COUNTS),
        call_ledger=[],
        intended_count=320,
        eligible_count=counts["completed"],
        independent_count=0,
        completed_count=counts["completed"],
        excluded_count=counts["excluded"],
        censored_count=counts["censored"],
        failed_count=0,
        sample_size_budget=dict(
            stream=256,
            retention=64,
            arms=4,
            seeds=20,
            independent_natural_outcome_sources=0,
            repeats_add_independent_sources=0,
        ),
        run_date="20261005",
        random_seed=101,
        acceptance_gates=dict(
            historical_custody="authenticated256/64 original slots; usable224/48",
            fixture_installation_latest=208,
            fixture_changed_later_minimum=32,
            validation="normal owned commands and100 percent new statement coverage",
        ),
        methodology_note="No LLM loads or calls. Historical model provenance is imported separately. Qualified four-step heads train only on released update labels;12 future admission labels select the unchanged safety grid. Issue precedes release, admission precedes expiry, and new commitment comes last. Private fixtures qualify causality only; nonlinear heads inherit no convex regret bound.",
    )
    value["field_principles"] = {
        k: "Bound finite evidence; fixtures and repeated seeds add no independent natural benefit."
        for k in value
    }
    value["field_principles"].update(
        verdict_class="Unchanged external operands block terminally; owned failures disqualify.",
        learning_protocol_ready_score="Original custody, independent scalar fixtures and normal owned validation required.",
        model_invocation_counts="Current zero calls exclude imported historical provenance.",
        protocol_sha256="Frozen H2, small-head parameters and release-aware schedule precede new outcomes.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Rebuild fixture rows and state evidence so even rehashed tampering fails."""
    try:
        value = json.loads(path.read_text())
        checksum = value.pop("reproducibility_checksum")
        if checksum != canonical_hash(value) or value["numerical_protocol"] != protocol():
            return False
        if sha256_file(Path(value["protocol_path"])) != value["protocol_sha256"]:
            return False
        expected = build(
            value,
            Path(value["terminal_validation_sidecar_path"]).parent,
            value["validation_receipts"],
        )
        fields = [
            "learning_protocol_ready_score",
            "future_exposure_fixture_score",
            "verdict_class",
            "completed_count",
            "censored_count",
            "excluded_count",
        ]
        if (
            any(expected[k] != value[k] for k in fields)
            or engine.historical.reductions(value["rows"]) != value["reductions"]
        ):
            return False
        for p, digest in value["code_config_hashes"].items():
            if sha256_file(ROOT / p) != digest:
                return False
        for ref in value["raw_shard_hashes"] + value["source_artifact_hashes"]:
            if sha256_file(Path(ref.get("snapshot_path", ref["path"]))) != ref["sha256"]:
                return False
        recomputed = []
        for index, entry in enumerate(value["event_reference_rows"]):
            inputs = json.loads(Path(entry["inputs"]["path"]).read_text())
            fixture_rows, fixture_labels = fixture(entry["case"])
            scalar.equal(dict(rows=fixture_rows, labels=fixture_labels), inputs)
            state = json.loads(Path(entry["transcript"]["path"]).read_text())
            scalar.equal(
                reference(state, inputs["rows"], inputs["labels"]),
                {
                    k: entry[k]
                    for k in [
                        "passed",
                        "event_count",
                        "release_count",
                        "overflow_count",
                        "admission_reuse_count",
                    ]
                },
            )
            scalar.equal(exposure(state, inputs["rows"]), entry["exposure"])
            if entry["case"] == "positive":
                for row, pred in zip(inputs["rows"], state["issued"], strict=True):
                    label = dict(
                        y=inputs["labels"][row["slot"] - 1] if row["slot"] <= 236 else None,
                        exclusion_reason=None if row["slot"] <= 236 else "feedback_unresolved_tail",
                    )
                    recomputed.extend(
                        engine.historical.scored(
                            row, pred, label, "private_learnable_fixture", entry["seed"]
                        )
                    )
                retention, targets = engine.fixture(64, "retention")
                sealed = json.loads(Path(entry["retention_predictions"]["path"]).read_text())[
                    "rows"
                ]
                for row, pred, y in zip(retention, sealed, targets, strict=True):
                    scalar.equal(
                        {
                            a: engine.scalar_probability(h, state["geometry"], row["values"])
                            for a, h in state["arms"].items()
                        },
                        pred["predictions"],
                    )
                    pred["prediction_hash"] = canonical_hash(pred["predictions"])
                    recomputed.extend(
                        engine.historical.scored(
                            row,
                            pred,
                            dict(y=y, exclusion_reason=None),
                            "private_retention_fixture",
                            entry["seed"],
                        )
                    )
            progress(
                "cold_case_complete", index + 1, len(value["event_reference_rows"]) - index - 1
            )
        key = lambda row: (row["seed"], row["condition"], row["slot"], row["arm"], row["metric"])
        if (
            sorted(recomputed, key=key) != sorted(value["rows"], key=key)
            or engine.historical.reductions(recomputed) != value["reductions"]
        ):
            return False
        rows, labels = fixture("positive")
        old = json.loads(Path(value["old_expiry_fixture"]["transcript"]["path"]).read_text())
        if old != engine.run(rows, labels, 101) or not value["old_expiry_fixture"]["passed"]:
            return False
        checkpoint = json.loads(Path(value["restart_fixture"]["checkpoint"]["path"]).read_text())
        resumed = run(rows, labels, 101, state=checkpoint)
        baseline = json.loads(
            Path(value["event_reference_rows"][0]["transcript"]["path"]).read_text()
        )
        if resumed != baseline or not value["restart_fixture"]["passed"]:
            return False
        expected = build(
            value,
            Path(value["terminal_validation_sidecar_path"]).parent,
            value["validation_receipts"],
        )
        return all(
            expected[k] == value[k]
            for k in [
                "learning_protocol_ready_score",
                "future_exposure_fixture_score",
                "verdict_class",
                "completed_count",
                "censored_count",
                "excluded_count",
            ]
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration):
        return False
