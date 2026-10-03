"""REQ-REPORT-8002: measure complete CPU requests on immutable producer inputs.

Repeated timings describe engineering cost. Only original source groups supply
independent observations; cached acquisition keeps its producer date.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.stats import t  # type: ignore[import-untyped]

from carnot.reporting import service_cost_7989 as historical
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.evidence_features_custody_7980 import checked, operand, reference
from carnot.verify import evidence_features_7980 as features
from carnot.verify import qwen_energy_calibration_7972 as scalar
from carnot.verify import selective_feedback_7998 as learning
from carnot.verify import sparse_energy_7996 as sparse
from carnot.verify.typed_development_7997 import action

Json = dict[str, Any]
ROOT = historical.ROOT
PINS = {
    7995: (
        "experiment_7995_v693_qwen_development_capture.json",
        "capture_ready_score",
        "sha256:df6eb8b559181455348b1d806f23c36d13c7f49202c2d49b2233aa80edebe02d",
    ),
    7996: (
        "experiment_7996_v693_sparse_energy_training.json",
        "sparse_fit_ready_score",
        "sha256:ab4193d06c43b8abae95d2dc866a544246eff17179ec6fb90f79d7bab8b20bda",
    ),
    7998: (
        "experiment_7998_v693_selective_feedback_learning.json",
        "learning_measurement_ready_score",
        "sha256:3c4edde6a059e945e22d02334a8b6fb8d5983562376c181ae0a76926ab964ba6",
    ),
}
CASES = ("no_write", "empty_durable", "sparse_update_durable")
PHASES = (
    "byte_parsing",
    "public_features",
    "head_prediction",
    "typed_decision",
    "acquisition_selection",
    "feedback_processing",
    "checkpoint_serialization",
    "storage_fsync",
)
CONFIG = dict(
    random_seed=69302,
    maximum_sources=64,
    warmups=1,
    repetitions=10,
    cases=list(CASES),
    current_pretrained_calls=0,
)
SCALAR_REF = dict(
    path=str(ROOT / "results/raw/experiment_7972_v691_qwen_energy_calibration/heads.json"),
    sha256="sha256:40a7876cd2ada1755c02d740501924e686eb2eb722388776f42a01cbd5acad8b",
)


def snapshot(ref: Json, directory: Path) -> Json:
    """Copy checked bytes so mutable producer paths cannot replace measured inputs."""
    original = checked(ref)
    destination = directory / (ref["sha256"].split(":")[1] + original.suffix)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(original.read_bytes())
    return dict(reference(destination), original_reference=ref)


def fixture() -> Json:
    """Explicit feedback controls exercise locality without natural-data claims."""
    heads = dict(
        spline=dict(
            arm="spline",
            seed=17,
            parameters=[0.0] * 109,
            decay_scale=1.0,
            temperature=1.0,
            scaler=dict(minimum=[0.0] * 9, maximum=[1.0] * 9),
        ),
        scalar=historical.fixture()[1]["heads"]["scalar"][0],
    )
    requests = []
    for i, q in enumerate((0.4, None)):
        requests.append(
            dict(
                family_id=f"f{i}",
                source_cluster_id=f"c{i}",
                q=q,
                status="fixture",
                public=dict(
                    family_id=f"f{i}",
                    source_bytes=b"The cat is black.".hex(),
                    answer_bytes=b"The cat is black.".hex(),
                ),
                acquisition=None,
                feedback_scope="fixture_cost_only",
                feedback=dict(
                    receipt_id=f"fixture:{i}", origin_slot=0, due_slot=20, y=1, weight=2.0
                ),
                update_source=dict(q=0.4, features=[0.3] * 8),
                state=dict(head=copy.deepcopy(heads["spline"]), next_slot=20, seen_label_ids=[]),
            )
        )
    return dict(
        requests=requests,
        heads=heads,
        source_artifact_hashes=[],
        branch_eligibility={},
        gate_check_summary=[],
        load_amortization={},
    )


def authenticate(root: Path, directory: Path) -> Json:
    """Qualify alternatives independently; a failed learner only limits updates."""
    plan: Json = dict(
        requests=[],
        heads={},
        source_artifact_hashes=[],
        branch_eligibility={},
        gate_check_summary=[],
        load_amortization={},
    )
    upstream: Json = {}
    for eid, (name, field, pin) in PINS.items():
        path = root / "results" / name
        checks = [operand(eid, path, "sha256", pin, sha256_file(path) if path.is_file() else None)]
        if checks[0]["passed"]:
            ref = snapshot(reference(path), directory)
            value = json.loads(checked(ref).read_bytes())
            for key, want in {field: 1, "flagged_adversarial": False, "experiment_id": eid}.items():
                if key not in value:
                    raise ValueError("upstream_contract:" + key)
                checks.append(operand(eid, path, key, want, value[key]))
            ref.update(
                producer_id=eid,
                producer_invocation_date=value["execution_date"],
                imported_fields=[field, "checkpoints", "rows"],
            )
            plan["source_artifact_hashes"].append(ref)
            if all(c["passed"] for c in checks):
                upstream[str(eid)] = value
        plan["branch_eligibility"][str(eid)] = dict(
            eligible=all(c["passed"] for c in checks), operands=checks
        )
        plan["gate_check_summary"] += [c for c in checks if not c["passed"]]

    for producer_key in list(upstream):
        producer = upstream[producer_key]
        assets = (
            [producer["checkpoints"]["heads"]]
            if producer_key == "7996"
            else [
                SCALAR_REF,
                producer["public_role_manifests"]["stream"],
                *producer["original_capture_code_snapshot"],
                *producer["raw_response_shards"],
            ]
            if producer_key == "7995"
            else [producer["checkpoints"]["targeted_ipw-101"]]
        )
        for ref in assets:
            path = Path(ref["path"])
            check = operand(
                producer_key,
                path,
                "sha256",
                ref["sha256"],
                sha256_file(path) if path.is_file() else None,
            )
            plan["branch_eligibility"][producer_key]["operands"].append(check)
            if not check["passed"]:
                plan["gate_check_summary"].append(check)
                plan["branch_eligibility"][producer_key]["eligible"] = False
        if not plan["branch_eligibility"][producer_key]["eligible"]:
            del upstream[producer_key]

    def load(ref: Json) -> Json:
        sealed = snapshot(ref, directory)
        plan["source_artifact_hashes"].append(sealed)
        return json.loads(checked(sealed).read_bytes())  # type: ignore[no-any-return]

    if "7996" in upstream:
        plan["heads"]["spline"] = load(upstream["7996"]["checkpoints"]["heads"])["heads"]["spline"][
            0
        ]
    if "7995" in upstream:
        capture = upstream["7995"]
        if capture["model_identity_receipt"]["authenticated"] is not True:
            raise ValueError("capture_identity")
        plan["heads"]["scalar"] = load(SCALAR_REF)["heads"]["gibbs"][0]
        view = load(capture["public_role_manifests"]["stream"])
        for ref in capture["original_capture_code_snapshot"]:
            load(ref) if Path(ref["path"]).suffix == ".json" else plan[
                "source_artifact_hashes"
            ].append(snapshot(ref, directory))
        raw_rows = {
            r["family_id"]: r for r in [load(ref) for ref in capture["raw_response_shards"]]
        }
        by_id = {r["family_id"]: r for r in capture["rows"] if r["role"] == "stream"}
        for public in view["request_rows"][:64]:
            row = by_id[public["family_id"]]
            if row["public_hash"] != canonical_hash(public) or row != raw_rows[public["family_id"]]:
                raise ValueError("acquisition_source_hash")
            plan["requests"].append(
                dict(
                    family_id=public["family_id"],
                    source_cluster_id=features.extract(public)["source_normalized_hash"],
                    public=public,
                    q=row["parsed"]["probability"],
                    status=row["status"],
                    acquisition=dict(
                        duration_s=row["duration_s"],
                        producer_id=7995,
                        producer_date=capture["execution_date"],
                        public_hash=row["public_hash"],
                        source_hash=features.extract(public)["source_normalized_hash"],
                        response_hash=canonical_hash(row["raw_response"]),
                        model_generation_s=row["raw_response"]
                        .get("timings", {})
                        .get("predicted_ms", 0)
                        / 1000,
                        prompt_processing_s=row["raw_response"]
                        .get("timings", {})
                        .get("prompt_ms", 0)
                        / 1000,
                        input_tokens=row["input_tokens"],
                        output_tokens=row["parsed"]["usage"]["completion_tokens"],
                    ),
                )
            )
        completed = sum(r["status"] == "generated" for r in capture["rows"])
        plan["load_amortization"] = dict(
            producer_id=7995,
            producer_date=capture["execution_date"],
            historical=True,
            load_s=capture["model_identity_receipt"]["duration_s"],
            completed_requests=completed,
            per_request_s=capture["model_identity_receipt"]["duration_s"] / completed,
            current_model_loads=0,
        )
    elif "7996" in upstream:
        old = historical.authenticate(ROOT)
        plan["requests"] = [dict(r, acquisition=None) for r in old["requests"]]
        for ref in old["source_artifact_hashes"]:
            plan["source_artifact_hashes"].append(snapshot(ref, directory))
    if "7998" in upstream and "spline" in plan["heads"] and "7995" in upstream:
        learner = upstream["7998"]
        trajectory = load(learner["checkpoints"]["targeted_ipw-101"])
        view = load(upstream["7995"]["public_role_manifests"]["stream"])
        public = {r["family_id"]: r for r in view["request_rows"]}
        for receipt in [
            r for r in learner["update_rows"] if r["arm"] == "targeted_ipw" and r["seed"] == 101
        ]:
            item = next(
                (r for r in plan["requests"] if r["family_id"] == receipt["family_id"]), None
            )
            if item is not None:
                state_path = (
                    Path(trajectory["state_directory"])
                    / f"committed-{receipt['due_slot'] - 1:04d}.json"
                )
                commit = load(reference(state_path))
                sealed_commit = next(
                    r
                    for r in trajectory["trajectory"]["checkpoint_rows"]
                    if r["slot"] == receipt["due_slot"] - 1
                )
                if (
                    canonical_hash(commit["state"]) != commit["checksum"]
                    or commit["checksum"] != sealed_commit["checksum"]
                ):
                    raise ValueError("historical_commit_hash")
                source = public[receipt["family_id"]]
                item.update(
                    feedback=receipt,
                    state=commit["state"],
                    update_source=dict(
                        q=by_id[receipt["family_id"]]["parsed"]["probability"],
                        features=features.extract(source)["values"],
                    ),
                    feedback_scope="exp7998_actual_receipt",
                )
                item["state"]["next_slot"] = receipt["due_slot"]
    plan["branch_eligibility"].update(
        sparse_fit=plan["branch_eligibility"]["7996"],
        capture_scalar=plan["branch_eligibility"]["7995"],
    )
    return plan


def request(path: Path, head: Json, arm: str, case: str, output: Path) -> Json:
    """Time adjacent boundaries, including checkpoint bytes and actual fsync calls."""
    ticks = [time.perf_counter_ns()]
    raw = path.read_bytes()
    item = json.loads(raw)
    head = item.get("prediction_heads", {}).get(arm, head)
    state = copy.deepcopy(item.get("state", dict(head=head, seen_label_ids=[], next_slot=20)))
    ticks.append(time.perf_counter_ns())
    extracted = features.extract(item["public"])
    source = dict(q=item["q"], features=extracted["values"])
    ticks.append(time.perf_counter_ns())
    p = None
    if item["q"] is not None and extracted["values"] is not None:
        p = (
            float(scalar.predict(head, np.asarray([item["q"]]))[0])
            if arm == "scalar"
            else float(sparse.predict(head, sparse.inputs([source]))[0])
        )
    ticks.append(time.perf_counter_ns())
    response = dict(family_id=item["family_id"], probability=p, action=action(p), verified=False)
    ticks.append(time.perf_counter_ns())
    pi = 0.0 if p is None else (0.5 if 0.05 <= p <= 0.75 else 0.125)
    draw = (
        int(
            canonical_hash(dict(source=item["source_cluster_id"], seed=69302)).split(":")[1][:13],
            16,
        )
        / 16**13
    )
    response.update(acquisition_probability=pi, selected=draw < pi)
    ticks.append(time.perf_counter_ns())
    before = list(state["head"]["parameters"])
    update = None
    if case == "sparse_update_durable" and arm == "spline" and item.get("feedback"):
        update = learning.apply_feedback(state, item["feedback"], item["update_source"])
    changed = sum(a != b for a, b in zip(before, state["head"]["parameters"], strict=True))
    ticks.append(time.perf_counter_ns())
    encoded = json.dumps(dict(response=response, state=state), sort_keys=True).encode()
    ticks.append(time.perf_counter_ns())
    if case != "no_write":
        with output.open("wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        os.fsync(directory)
        os.close(directory)
    ticks.append(time.perf_counter_ns())
    acquisition = item.get("acquisition")
    return dict(
        family_id=item["family_id"],
        source_cluster_id=item["source_cluster_id"],
        arm=arm,
        case=case,
        response=response,
        ticks_ns=ticks,
        exclusive_phase_spans=dict(zip(PHASES, np.diff(ticks).tolist(), strict=True)),
        wall_ns=ticks[-1] - ticks[0],
        bytes_read=len(raw),
        bytes_written=len(encoded) if case != "no_write" else 0,
        serialized_bytes=len(encoded),
        changed_coefficients=changed,
        coefficient_touches=update["coefficient_touches"] if update else 0,
        logical_decay_coefficients=109 if update else 0,
        feedback_scope=item.get("feedback_scope", "empty_feedback"),
        request_sha256=canonical_hash(item),
        state_checksum=canonical_hash(state),
        cached_total_s=(ticks[-1] - ticks[0]) / 1e9,
        complete_total_s=acquisition["duration_s"] + (ticks[-1] - ticks[0]) / 1e9
        if acquisition
        else None,
        acquisition=acquisition,
        numerator=ticks[-1] - ticks[0],
        denominator=1,
        eligibility=p is not None,
        failure_status=False,
        censor_status=item["q"] is None,
    )


def reduce(rows: list[Json]) -> Json:
    """Average repetitions within sources before estimating paired uncertainty."""
    cached, complete, intervals = [], [], []
    for arm in sorted({r["arm"] for r in rows}):
        for case in CASES:
            selected = [r for r in rows if r["arm"] == arm and r["case"] == case]
            times = [r["cached_total_s"] for r in selected]
            known = [r["complete_total_s"] for r in selected if r["complete_total_s"] is not None]
            cached.append(
                dict(
                    arm=arm,
                    case=case,
                    p50_s=float(np.quantile(times, 0.5)),
                    p95_s=float(np.quantile(times, 0.95)),
                    observations=len(times),
                )
            )
            complete.append(
                dict(
                    arm=arm,
                    case=case,
                    p50_s=float(np.quantile(known, 0.5)) if known else None,
                    p95_s=float(np.quantile(known, 0.95)) if known else None,
                    matched_rows=len(known),
                    missing_rows=len(times) - len(known),
                    boundary="matching producer acquisition plus current complete CPU service; load amortization separate",
                )
            )
            by_source = {}
            for row in selected:
                baseline = next(
                    r
                    for r in rows
                    if (r["family_id"], r["arm"], r["repetition"], r["case"])
                    == (row["family_id"], arm, row["repetition"], "no_write")
                )
                by_source.setdefault(row["source_cluster_id"], []).append(
                    (row["wall_ns"] - baseline["wall_ns"]) / 1e9
                )
            delta = np.asarray([np.mean(v) for v in by_source.values()])
            mean = float(delta.mean())
            margin = (
                float(t.ppf(0.975, len(delta) - 1) * np.std(delta, ddof=1) / np.sqrt(len(delta)))
                if len(delta) > 1
                else 0.0
            )
            accelerated = sum(
                r["exclusive_phase_spans"]["head_prediction"] for r in selected
            ) / sum(r["wall_ns"] for r in selected)
            intervals.append(
                dict(
                    arm=arm,
                    case=case,
                    independent=len(delta),
                    mean_difference_s=mean,
                    paired_95_interval_s=[mean - margin, mean + margin] if len(delta) > 1 else None,
                    method="Student t on paired source means",
                    measured_cached_accelerated_fraction=accelerated,
                    maximum_ideal_cached_acceleration=1 / (1 - accelerated),
                    maximum_ideal_complete_acceleration=1
                    / (
                        1
                        - sum(r["exclusive_phase_spans"]["head_prediction"] / 1e9 for r in selected)
                        / sum(known)
                    )
                    if len(known) == len(times)
                    else None,
                    component_100x_status="unmeasured_target",
                    sparse_sub_microsecond_status="unmeasured_target",
                )
            )
    return dict(
        cached_service_cost=cached,
        complete_service_cost=complete,
        paired_latency_intervals=intervals,
    )


def measure(plan: Json, directory: Path) -> Json:
    """One warmup and ten paired repeats include actual durable state writes."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, item in enumerate(plan["requests"]):
        path = directory / f"request-{index}.json"
        atomic_json(path, dict(item, prediction_heads=plan["heads"]))
        paths.append(path)
    rows = []
    combinations = [(arm, case) for arm in plan["heads"] for case in CASES]
    rng = np.random.default_rng(CONFIG["random_seed"])
    for repetition in range(-1, 10):
        print(f"[exp8002] before_benchmark repetition={repetition}", flush=True)
        for index, path in enumerate(paths):
            for position in rng.permutation(len(combinations)):
                arm, case = combinations[position]
                row = request(path, plan["heads"][arm], arm, case, directory / "state.json")
                if repetition >= 0:
                    rows.append(dict(row, repetition=repetition, seed=69302))
            if index % 8 == 0:
                print(f"[exp8002] source={index + 1}/{len(paths)} rows={len(rows)}", flush=True)
        print(f"[exp8002] after_benchmark repetition={repetition}", flush=True)
        atomic_json(directory / "measurement_checkpoint.json", dict(rows=rows))
    budget = dict(
        intended=len(paths),
        eligible=sum(r["q"] is not None for r in plan["requests"]),
        started=len(paths),
        completed=len(paths),
        excluded=0,
        failed=0,
        censored=sum(r["q"] is None for r in plan["requests"]),
        independent=len({r["source_cluster_id"] for r in plan["requests"]}),
        timing_repetitions=10,
        named_warmups=1,
        seeds_are_independent=False,
    )
    return dict(
        rows=rows,
        **reduce(rows),
        replay_inputs=plan,
        sample_size_budget=budget,
        update_touch_rows=[
            dict(
                family_id=r["family_id"],
                arm=r["arm"],
                case=r["case"],
                repetition=r["repetition"],
                changed_coefficients=r["changed_coefficients"],
                coefficient_touches=r["coefficient_touches"],
                logical_decay_coefficients=r["logical_decay_coefficients"],
                feedback_scope=r["feedback_scope"],
                feedback_processing_s=r["exclusive_phase_spans"]["feedback_processing"] / 1e9,
            )
            for r in rows
        ],
        durable_write_rows=[
            dict(
                family_id=r["family_id"],
                arm=r["arm"],
                case=r["case"],
                repetition=r["repetition"],
                bytes_written=r["bytes_written"],
                serialized_bytes=r["serialized_bytes"],
                state_checksum=r["state_checksum"],
            )
            for r in rows
        ],
    )


def replay(value: Json) -> None:
    """Recompute deterministic service work and reject altered measured accounting."""
    for ref in (
        value.get("source_artifact_hashes", [])
        + value.get("measurement_code_snapshot", [])
        + value.get("raw_shard_hashes", [])
    ):
        checked(ref)
    for ref in value.get("code_config_hashes", []):
        checked(ref)
    plan = value["replay_inputs"]
    expected = {r["family_id"]: r for r in plan["requests"]}
    import tempfile

    with tempfile.TemporaryDirectory(prefix="carnot-8002-replay-", dir="/tmp") as folder:
        directory = Path(folder)
        memo = {}
        for row in value["rows"]:
            key = (row["family_id"], row["arm"], row["case"])
            if key not in memo:
                atomic_json(
                    directory / "input.json",
                    dict(expected[row["family_id"]], prediction_heads=plan["heads"]),
                )
                memo[key] = request(
                    directory / "input.json",
                    plan["heads"][row["arm"]],
                    row["arm"],
                    row["case"],
                    directory / "state",
                )
            rebuilt = memo[key]
            for field in (
                "response",
                "bytes_written",
                "serialized_bytes",
                "changed_coefficients",
                "coefficient_touches",
                "logical_decay_coefficients",
                "state_checksum",
                "request_sha256",
                "acquisition",
            ):
                if rebuilt[field] != row[field]:
                    raise ValueError("row_drift:" + field)
            spans = dict(zip(PHASES, np.diff(row["ticks_ns"]).tolist(), strict=True))
            if (
                spans != row["exclusive_phase_spans"]
                or min(spans.values()) < 0
                or sum(spans.values()) != row["wall_ns"]
                or row["cached_total_s"] != row["wall_ns"] / 1e9
                or row["numerator"] != row["wall_ns"]
            ):
                raise ValueError("span_drift")
            acquisition = row["acquisition"]
            if row["complete_total_s"] != (
                acquisition["duration_s"] + row["cached_total_s"] if acquisition else None
            ):
                raise ValueError("join_drift")
    expected_pairs = (
        {
            (r["family_id"], arm, case, repetition)
            for r in plan["requests"]
            for arm in plan["heads"]
            for case in CASES
            for repetition in range(10)
        }
        if value["rows"]
        else set()
    )
    observed_pairs = {(r["family_id"], r["arm"], r["case"], r["repetition"]) for r in value["rows"]}
    if len(observed_pairs) != len(value["rows"]) or observed_pairs != expected_pairs:
        raise ValueError("paired_rows_drift")
    if any(value[k] != v for k, v in reduce(value["rows"]).items()):
        raise ValueError("reduction_drift")
