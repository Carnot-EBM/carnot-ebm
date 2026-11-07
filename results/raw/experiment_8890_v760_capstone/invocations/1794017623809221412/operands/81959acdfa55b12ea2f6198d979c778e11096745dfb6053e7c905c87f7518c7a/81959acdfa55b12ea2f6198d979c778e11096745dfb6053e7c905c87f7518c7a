"""REQ-REPORT-8860: reconstruct sealed linear evidence without fitting or selection.

Exact symbolic labels check constructed records only. Paired stream inference
cannot establish natural-language correctness or generator-training benefit.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np

from carnot.reporting import linear_ablation_methods_8854 as a
from carnot.reporting import linear_causal_runner_8856 as producer
from carnot.reporting.causal_learning_audit_8810 import load, reference, require
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
ROOT = producer.ROOT
NAME = "experiment_8860_v758_linear_benefit_audit"
CLI = "scripts/experiments/" + NAME + ".py"
SOURCE = ROOT / "results/experiment_8856_v758_linear_causal_streams.json"
PIN = "sha256:b537ac5ee2d13fe6db13292b70ea02682d608498bd8cf280811697bd7fdec0ce"
OWNED = [
    "python/carnot/reporting/linear_benefit_audit_8860.py",
    "python/carnot/reporting/linear_benefit_runner_8860.py",
    CLI,
]
TESTS = ["tests/python/test_linear_benefit_audit_8860.py"]
CONTRASTS = [
    ("linear_combined", arm)
    for arm in (
        "linear_frozen",
        "linear_calibration_only",
        "linear_shuffled_combined",
        "linear_additions_only",
    )
] + [("linear_calibration_only", "linear_frozen"), ("linear_additions_only", "linear_frozen")]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Expose real boundaries and counts while preserving the fixed deadline."""
    print(f"[exp8860] phase={phase} completed={completed} pending={pending}", flush=True)


def preflight(raw: Path, source_path: Path) -> tuple[Json, Json, list[Json]]:
    """Bind original authorities and raw bytes before reading comparative outcomes."""
    raw.mkdir(parents=True, exist_ok=True)
    raw.chmod(0o700)
    probe = raw / ".probe"
    probe.write_bytes(b"private")
    probe.unlink()
    free = shutil.disk_usage(raw).free
    if free < 2**30 or Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        raise RuntimeError("internal_resources")
    source = load(dict(path=str(source_path), sha256=PIN))
    if (
        source.get("stream_rows_ready_score") != 1
        or source.get("required_checks_passed") is not True
    ):
        raise PermissionError("stream_rows_ready_score:observed_zero")
    methods = load(dict(path=str(producer.SOURCE), sha256=producer.SOURCE_PIN))
    history = load(dict(path=str(producer.authority.AUDIT), sha256=producer.authority.AUDIT_PIN))
    require(
        methods["linear_methods_ready_score"] == 1
        and methods["required_checks_passed"]
        and history["learning_audit_ready_score"] == 1,
        "original_readiness",
    )
    refs = [reference(source_path), reference(producer.SOURCE), reference(producer.authority.AUDIT)]
    for value in (source, methods, history):
        refs.extend(
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        )
    refs = list({(r["path"], r["sha256"]): r for r in refs}.values())
    a.prior.upstream.bindings(refs, "original_operand_hash")
    work = load(source["raw_shard_hashes"][0])
    require(work["protocol"] == a.protocol() == source["protocol"], "frozen_protocol")
    require(source["rows"] == producer.m.result_rows(work, ""), "producer_row_join")
    require(len(work["stream_refs"]) == len(work["stream_results"]) == 24, "stream_census")
    for ref, stream in zip(work["stream_refs"], work["stream_results"], strict=True):
        require(load(ref) == stream, "stream_shard_join")
        a.prior.upstream.bindings(stream["final_state_refs"], "final_state_operand")
    return (
        work,
        dict(
            private_output=str(raw),
            free_disk_bytes=free,
            minimum_free_disk_bytes=2**30,
            virtualenv=sys.prefix,
            time_budget_s=4200,
            dependencies=dict(numpy=np.__version__, jax=a.jax.__version__),
            numerical_training_calls=0,
            model_loads=0,
            authenticated_reference_count=len(refs),
            configuration=work["protocol"]["configuration"],
        ),
        refs,
    )


def reconstruct_stream(stream: Json, initial: list[Json]) -> Json:
    """Rebuild causal state from recorded coefficients, never invoking an optimizer."""
    producer.m.integrity(stream)
    plan = stream["stream"]
    expected = a.census(dict(cohort_manifest=[plan]), "")
    identity_keys = ("unit_id", "cluster_id", "seed", "arm", "split", "checkpoint", "ordinal")
    require(
        {tuple(r[k] for k in identity_keys) for r in expected}
        == {tuple(r[k] for k in identity_keys) for r in stream["rows"]},
        "row_identity",
    )
    require(
        stream["retention_rows"] == [r for r in stream["rows"] if r["split"] == "retention"],
        "retention_join",
    )
    worlds = {split: [a.payload(w) for w in plan[split]] for split in ("online", "retention")}
    integrity, reuse, movement = [], [], []
    probabilities: dict[str, list[float]] = {}
    for arm, settings in a.ARMS.items():
        identity = dict(cluster_id=plan["stream_id"], seed=plan["seed"], arm=arm)
        local = [r for r in stream["rows"] if r["arm"] == arm]
        online = {r["ordinal"]: r for r in local if r["split"] == "online"}
        feedback = {r["event_id"]: r for r in stream["feedback_rows"] if r["arm"] == arm}
        head = next(r["parameters"] for r in initial if r["seed"] == plan["seed"])
        state = dict(templates=[], parameters=deepcopy(head), pending={}, acks=[], released=[])
        snapshots = {0: deepcopy(state)}
        admissions: Json = {}
        actual_reuse, moves = [], []
        delays = [0] * 48 if settings["diagnostic"] else plan["delays"]
        for ordinal in range(1, 61):
            if ordinal <= 48:
                row, value = online[ordinal], worlds["online"][ordinal - 1]
                observation = a.c.m.observe(value, state["templates"])
                cache = canonical_hash([state["parameters"], observation["counts"]])
                if cache not in probabilities:
                    probabilities[cache] = a.c.probabilities(
                        state["parameters"], observation["counts"]
                    )
                p = probabilities[cache]
                event, template = f"q{ordinal}", a.c.m.normalize(value)
                active = [template] if template in state["templates"] else []
                sealed = dict(
                    observation,
                    event_id=event,
                    ordinal=ordinal,
                    prediction=p,
                    action=a.c.m.action(p, 0.5),
                    state_hash=canonical_hash(state),
                    evidence_hash=canonical_hash(value),
                    template_uses=active,
                )
                require(
                    row["sealed"] == sealed
                    and row["prediction"] == p
                    and row["delay"] == delays[ordinal - 1]
                    and row["future_window"] == (ordinal >= 25)
                    and row["truth"] == a.c.m.decide(value)["decision"],
                    "sealed_prediction",
                )
                for template in active:
                    admission = admissions[template]
                    require(admission["feedback_ordinal"] < ordinal, "early_template_use")
                    actual_reuse.append(
                        dict(
                            identity,
                            event_id=event,
                            ordinal=ordinal,
                            template=template,
                            provenance=state["template_provenance"][template],
                            prediction_seal=canonical_hash(sealed),
                            future_window=ordinal >= 25,
                        )
                    )
                    reuse.append(
                        dict(
                            identity,
                            admission_event=admission["event_id"],
                            feedback_ordinal=admission["feedback_ordinal"],
                            use_event=event,
                            use_ordinal=ordinal,
                            future_window=ordinal >= 25,
                            changed_predicates=observation["counts"]
                            != a.c.m.observe(value, [])["counts"],
                            template_hash=canonical_hash(template),
                        )
                    )
                state["pending"][event] = sealed
            for at, delay in enumerate(delays, 1):
                if at + delay != ordinal:
                    continue
                event, value = f"q{at}", worlds["online"][at - 1]
                row, pending = feedback[event], state["pending"][event]
                label = str(a.c.m.decide(value)["decision"])
                prefix = state["released"] + [label]
                training = prefix[(ordinal * 7) % len(prefix)] if "shuffled" in arm else label
                template = a.c.m.normalize(value)
                added = (
                    settings["additions"]
                    and training == label
                    and template not in state["templates"]
                    and len(state["templates"]) < 32
                )
                require(
                    row["state_before"] == canonical_hash(state)
                    and row["parameters_before"] == state["parameters"]
                    and row["prediction_hash"] == canonical_hash(pending)
                    and row["prediction_ordinal"] == at
                    and row["feedback_ordinal"] == ordinal
                    and row["feedback_label"] == training
                    and row["verified_label"] == label
                    and row["released_prefix"] == prefix
                    and row["template"] == template
                    and row["template_added"] == added
                    and row["optimizer_steps"] == int(settings["calibration"]),
                    "feedback_boundary",
                )
                before = state["parameters"]
                after = row["parameters_after"]
                delta = float(
                    sum(
                        np.sum(
                            (np.asarray(x, dtype=np.float32) - np.asarray(y, dtype=np.float32)) ** 2
                        )
                        for x, y in zip(before["weights"], after["weights"], strict=True)
                    )
                    ** 0.5
                )
                require(
                    delta == row["parameter_delta"]
                    and row["weights_changed"] == (before != after)
                    and row["templates_changed"] == added
                    and (settings["calibration"] or before == after),
                    "parameter_movement",
                )
                state["parameters"] = deepcopy(after)
                if added:
                    state["templates"].append(template)
                    state.setdefault("template_provenance", {})[template] = dict(
                        event_id=event,
                        evidence_hash=canonical_hash(value),
                        query=value["query"],
                        verifier=a.c.m.decide(value),
                        binding_witnesses=a.c.binding_provenance(value),
                    )
                    admissions[template] = row
                require(
                    row["addition_witness"] == state.get("template_provenance", {}).get(template),
                    "template_witness",
                )
                state["pending"].pop(event)
                state["acks"].append(event)
                state["released"].append(label)
                require(row["checkpoint_hash"] == canonical_hash(state), "commit_hash")
                moves.append(dict(row, parameter_delta=delta))
            if ordinal == 24:
                restart = [r for r in stream["restart_rows"] if r["arm"] == arm]
                require(
                    len(restart) == 1
                    and restart[0]["before"] == restart[0]["after"] == canonical_hash(state),
                    "restart_parity",
                )
                snapshots[24] = deepcopy(state)
        snapshots[48] = state
        final = [r for r in stream["final_state_refs"] if r["arm"] == arm]
        require(
            len(final) == 1
            and load(final[0]) == state
            and not state["pending"]
            and len(state["acks"]) == 48,
            "tail_drain",
        )
        for row in local:
            if row["split"] != "retention":
                continue
            snapshot = snapshots[row["checkpoint"]]
            value = worlds["retention"][row["ordinal"] - 1]
            counts = a.c.m.observe(value, snapshot["templates"])["counts"]
            p = a.c.probabilities(snapshot["parameters"], counts)
            require(
                row["state_hash"] == canonical_hash(snapshot)
                and row["counts"] == counts
                and row["prediction"] == p
                and row["truth"] == a.c.m.decide(value)["decision"],
                "untouched_retention",
            )
        expected_reuse = [
            {k: v for k, v in r.items() if k != "wall_clock_ns"}
            for r in stream["template_reuse_rows"]
            if r["arm"] == arm
        ]
        require(actual_reuse == expected_reuse, "actual_template_reuse")
        for low, high in ((1, 24), (25, 48)):
            selected = [r for r in moves if low <= r["prediction_ordinal"] <= high]
            movement.append(
                dict(
                    identity,
                    prediction_window=[low, high],
                    optimizer_steps=sum(r["optimizer_steps"] for r in selected),
                    gradient_norm_sum=sum(r["gradient_norm"] for r in selected),
                    parameter_path_length=sum(r["parameter_delta"] for r in selected),
                    template_admissions=sum(r["template_added"] for r in selected),
                )
            )
        integrity.append(
            dict(
                identity,
                passed=True,
                restart_parity=True,
                tail_drained=True,
                prediction_before_feedback=True,
                prefix_only_shuffle=True,
                no_retention_feedback=True,
                shuffled_labels=sum(r["feedback_label"] != r["verified_label"] for r in moves),
            )
        )
    require(movement == stream["movement_rows"], "movement_reduction")
    return dict(rows=stream["rows"], integrity=integrity, reuse=reuse, movement=movement)


def intervals(values: np.ndarray, names: list[str], tail: float) -> list[Json]:
    """Resample complete paired streams; seeds never increase the sample count."""
    rng = np.random.default_rng(8860)
    draws = np.empty((10000, len(names)))
    for offset in range(0, 10000, 1000):
        indices = rng.integers(0, len(values), size=(1000, len(values)))
        draws[offset : offset + 1000] = values[indices].mean(axis=1)
        progress("bootstrap", offset + 1000, 9000 - offset)
    bounds = np.quantile(draws, [tail, 1 - tail], axis=0)
    return [
        dict(
            comparison=name,
            estimate=float(values[:, i].mean()),
            lower=float(bounds[0, i]),
            upper=float(bounds[1, i]),
            draws=10000,
            independent_count=len(values),
            familywise_confidence=0.99,
            tail_probability=tail,
            random_seed=8860,
            direction="left_minus_right; lower loss is better",
        )
        for i, name in enumerate(names)
    ]


def gates(paired: list[Json], retained: list[Json]) -> Json:
    """Joint benefit needs future gain, label specificity, addition and retention."""
    future = paired[0]["upper"] <= -0.01
    specific = paired[2]["upper"] < 0
    addition = paired[1]["upper"] <= -0.005
    retention = len(retained) == 4 and all(r["upper"] <= 0.01 for r in retained)
    return dict(
        causal_benefit_score=int(future and specific and addition and retention),
        addition_benefit_score=int(addition),
        retention_pass_score=int(retention),
        future_gain_score=int(future),
        true_label_specificity_score=int(specific),
    )


def reduce(work: Json) -> Json:
    """Reduce fixed endpoints only after independently reconstructing every stream."""
    require(work["protocol"] == a.protocol() and len(work["stream_results"]) == 24, "frozen_roster")
    rows, integrity, reuse, movement = [], [], [], []
    with a.prior.upstream.primitives():
        for i, stream in enumerate(work["stream_results"]):
            require(stream["stream"] == work["protocol"]["cohort_manifest"][i], "stream_identity")
            reconstructed = reconstruct_stream(stream, work["initialization_rows"])
            rows.extend(reconstructed["rows"])
            integrity.extend(reconstructed["integrity"])
            reuse.extend(reconstructed["reuse"])
            movement.extend(reconstructed["movement"])
            progress("reconstructed_streams", i + 1, 23 - i)
    groups: dict[tuple[Any, ...], list[Json]] = defaultdict(list)
    for row in rows:
        if row["split"] == "retention" or row["future_window"]:
            groups[(row["cluster_id"], row["arm"], row["split"], row["checkpoint"])].append(row)
    metrics = ("metric", "nll", "decisive_error", "covered", "false_accept", "false_reject")
    means = {
        key: {m: float(np.mean([r[m] for r in local])) for m in metrics}
        for key, local in groups.items()
    }
    streams = [s["stream_id"] for s in work["protocol"]["cohort_manifest"]]
    deltas = np.asarray(
        [
            [
                means[(s, left, "online", None)]["metric"]
                - means[(s, right, "online", None)]["metric"]
                for left, right in CONTRASTS
            ]
            for s in streams
        ]
    )
    names = [left + "_minus_" + right for left, right in CONTRASTS]
    paired = intervals(deltas, names, 0.01 / (2 * len(CONTRASTS)))
    specs = [(step, metric) for step in (24, 48) for metric in ("metric", "decisive_error")]
    retained = np.asarray(
        [
            [
                means[(s, "linear_combined", "retention", step)][metric]
                - means[(s, "linear_combined", "retention", 0)][metric]
                for step, metric in specs
            ]
            for s in streams
        ]
    )
    retention = intervals(
        retained,
        [f"linear_combined_step{step}_minus_0_{metric}" for step, metric in specs],
        0.01 / 8,
    )
    influence = [
        dict(
            omitted_stream=s,
            comparison=names[i],
            full_estimate=float(deltas[:, i].mean()),
            estimate_without_stream=float(np.delete(deltas[:, i], j).mean()),
            sign_changes=bool((np.delete(deltas[:, i], j).mean() < 0) != (deltas[:, i].mean() < 0)),
            share_of_absolute_effect=float(
                abs(deltas[j, i]) / max(np.abs(deltas[:, i]).sum(), 1e-30)
            ),
        )
        for j, s in enumerate(streams)
        for i in range(len(CONTRASTS))
    ]
    return dict(
        rows=rows,
        causal_integrity_rows=integrity,
        template_reuse_rows=reuse,
        movement_intervals=movement,
        paired_interval_rows=paired,
        retention_interval_rows=retention,
        per_stream_results=[
            dict(
                cluster_id=k[0],
                arm=k[1],
                seed=groups[k][0]["seed"],
                denominator=len(groups[k]),
                **m,
            )
            for k, m in means.items()
            if k[2] == "online"
        ],
        untouched_retention_rows=[
            dict(cluster_id=k[0], arm=k[1], checkpoint=k[3], denominator=len(groups[k]), **m)
            for k, m in means.items()
            if k[2] == "retention"
        ],
        delay_subgroups=producer.m.descriptives(rows),
        influence_rows=influence,
        effect_concentration=dict(
            any_leave_one_stream_sign_change=any(r["sign_changes"] for r in influence),
            maximum_absolute_effect_share=max(r["share_of_absolute_effect"] for r in influence),
        ),
        immediate_feedback_diagnostic=dict(
            diagnostic_only=True,
            mean_future_brier_difference=float(
                np.mean(
                    [
                        means[(s, "linear_immediate_combined", "online", None)]["metric"]
                        - means[(s, "linear_combined", "online", None)]["metric"]
                        for s in streams
                    ]
                )
            ),
        ),
        independent_count=24,
        intended_count=27648,
        completed_count=len(rows),
        failed_count=0,
        censored_count=0,
        excluded_count=0,
        learning_audit_ready_score=1,
        **gates(paired, retention),
    )


def empty(reason: str) -> Json:
    """Keep the complete scheduled roster when external evidence cannot be opened."""
    return dict(
        rows=a.census(a.protocol(), reason),
        intended_count=27648,
        completed_count=0,
        failed_count=0,
        censored_count=27648,
        excluded_count=0,
        independent_count=0,
        learning_audit_ready_score=0,
        causal_benefit_score=0,
        addition_benefit_score=0,
        retention_pass_score=0,
        future_gain_score=0,
        true_label_specificity_score=0,
        paired_interval_rows=[],
        retention_interval_rows=[],
        template_reuse_rows=[],
        movement_intervals=[],
        delay_subgroups=[],
        causal_integrity_rows=[],
        per_stream_results=[],
        untouched_retention_rows=[],
        influence_rows=[],
        effect_concentration={},
        immediate_feedback_diagnostic={},
    )


def replay(path: Path) -> Json:
    """Reconstruct fixed endpoints again so repaired summary hashes cannot grant trust."""
    value = json.loads(path.read_bytes())
    a.prior.upstream.bindings(
        value["source_artifact_hashes"] + value["code_config_hashes"] + value["raw_shard_hashes"],
        "replay_operand",
    )
    require(value["acceptance_gates"] == a.GATES, "acceptance_drift")
    work = load(value["raw_shard_hashes"][0])
    if work:
        trusted = load(dict(path=str(SOURCE), sha256=PIN))
        require(
            value["raw_shard_hashes"][0]["sha256"] == trusted["raw_shard_hashes"][0]["sha256"],
            "original_evidence_hash",
        )
    expected = reduce(work) if work else empty(value["rows"][0]["missing_reason"])
    if value["verdict_class"] in {"blocked", "disqualified"}:
        expected.update({k: 0 for k in expected if k.endswith("_score")})
    for key, observed in expected.items():
        require(value[key] == observed, "reduction_drift:" + key)
    for receipt in value["validation_receipts"]:
        if "log_path" in receipt:
            log = Path(receipt["log_path"])
            a.prior.upstream.bindings(
                [
                    dict(
                        path=str(log if log.is_absolute() else ROOT / log),
                        sha256=receipt["log_sha256"],
                    )
                ],
                "validation_log",
            )
    return dict(passed=True, replay_passed=True)
