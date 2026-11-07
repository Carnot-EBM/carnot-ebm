"""REQ-REPORT-8846: audit frozen future rows without fitting or choosing thresholds.

Exact symbolic truth checks constructed streams only. Reconstructing serving
states and resampling whole streams does not establish external generalization.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np

from carnot.reporting import causal_streams_8843 as producer
from carnot.reporting.causal_learning_audit_8810 import load, reference, require, score
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
ROOT = producer.ROOT
NAME = "experiment_8846_v757_future_benefit_audit"
CLI = "scripts/experiments/" + NAME + ".py"
SOURCE = ROOT / "results/experiment_8843_v757_causal_streams.json"
PIN = "sha256:9b4b5c910bb3d9f0ba5261c0c86a0e1701792ad7324ee941fd8349b1156911a6"
OWNED = [
    "python/carnot/reporting/future_benefit_audit_8846.py",
    "python/carnot/reporting/future_benefit_runner_8846.py",
    CLI,
]
TESTS = ["tests/python/test_future_benefit_audit_8846.py"]
CONTRASTS = [
    ("categorical_combined", arm)
    for arm in (
        "categorical_frozen",
        "categorical_calibration_only",
        "categorical_shuffled_combined",
    )
] + [("linear_combined", "linear_frozen"), ("categorical_combined", "linear_combined")]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase boundaries so a bounded audit remains visible to its operator."""
    print(f"[exp8846] phase={phase} completed={completed} pending={pending}", flush=True)


def preflight(raw: Path, source_path: Path) -> tuple[Json, Json, list[Json]]:
    """Bind original authorities and sealed operands before opening audit outcomes."""
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
    refs = [reference(source_path)]
    refs.extend(
        source["source_artifact_hashes"] + source["code_config_hashes"] + source["raw_shard_hashes"]
    )
    work = load(source["raw_shard_hashes"][0])
    refs.extend(producer.upstream.references(work))
    refs.extend(source["preconditions_checked"]["original_method_inputs"])
    for authority in source["source_artifact_hashes"]:
        value = load(authority)
        refs.extend(value["code_config_hashes"] + value["raw_shard_hashes"])
        for shard in value["raw_shard_hashes"]:
            refs.extend(producer.upstream.references(load(shard)))
    historical = dict(
        path=str(ROOT / producer.validation.AUDIT), sha256=producer.validation.AUDIT_PIN
    )
    load(historical)
    refs.append(historical)
    refs = list({(r["path"], r["sha256"]): r for r in refs}.values())
    producer.upstream.bindings(refs, "original_operand_hash")
    require(work["stream_manifest"] == producer.manifest(), "frozen_manifest")
    for key in producer.COLLECTIONS + ("stream_manifest", "initialization_rows"):
        require(source[key] == work[key], "producer_reduction:" + key)
    for stream, ref in zip(work["stream_manifest"], work["stream_checkpoint_refs"], strict=True):
        checkpoint = load(ref)
        for key in producer.COLLECTIONS:
            require(
                checkpoint[key] == [r for r in work[key] if r["cluster_id"] == stream["stream_id"]],
                "shard_join:" + key,
            )
    checked = dict(
        virtualenv=sys.prefix,
        private_output=str(raw),
        free_disk_bytes=free,
        minimum_free_disk_bytes=2**30,
        dependencies=dict(numpy=np.__version__, jax=producer.jax.__version__),
        time_budget_s=4200,
        model_loads=0,
        numerical_training_calls=0,
        configuration=source["preconditions_checked"]["configuration"],
        authenticated_reference_count=len(refs),
    )
    return work, checked, refs


def group_key(row: Json) -> tuple[str, str]:
    """Arms share a stream seed and cannot become extra independent samples."""
    return str(row["cluster_id"]), str(row["arm"])


def reconstruct(work: Json) -> tuple[list[Json], list[Json], list[Json]]:
    """Rebuild state boundaries from admissions, sealed predictions and coefficient snapshots.

    Coefficients are authenticated recorded operands, not refitted parameters.
    Every serving probability is recalculated from its pre-feedback snapshot.
    """
    producer.integrity(work, producer.manifest())
    groups: dict[str, Any] = {key: defaultdict(list) for key in producer.COLLECTIONS}
    for key in producer.COLLECTIONS:
        for row in work[key]:
            groups[key][group_key(row)].append(row)
    heads = {(r["seed"], r["objective"]): r["parameters"] for r in work["initialization_rows"]}
    output, integrity, reuse = [], [], []
    probabilities: dict[str, list[float]] = {}
    with producer.upstream.primitives():
        for stream in producer.manifest():
            worlds = {
                split: [producer.payload(w) for w in stream[split]]
                for split in ("online", "retention")
            }
            for arm in producer.ARMS:
                key = (stream["stream_id"], arm)
                selected = {name: groups[name][key] for name in producer.COLLECTIONS}
                online = {r["ordinal"]: r for r in selected["per_event_results"]}
                releases = {r["feedback_ordinal"]: r for r in selected["label_release_rows"]}
                gradients = {r["event_id"]: r for r in selected["gradient_rows"]}
                uses = {r["event_id"]: r for r in selected["template_use_rows"]}
                transitions = iter(selected["state_transition_rows"])
                state = dict(
                    templates=[],
                    parameters=deepcopy(heads[(stream["seed"], arm.split("_")[0])]),
                    pending={},
                    acks=[],
                    released=[],
                )
                snapshots = {0: deepcopy(state)}
                admissions: Json = {}
                shuffled = 0
                for ordinal in range(1, 57):
                    if ordinal <= 48:
                        row, payload = online[ordinal], worlds["online"][ordinal - 1]
                        require(
                            row["seed"] == stream["seed"]
                            and row["future_window"] == (ordinal >= 25),
                            "seed_window",
                        )
                        observation = producer.c.m.observe(payload, state["templates"])
                        cache_key = canonical_hash([state["parameters"], observation["counts"]])
                        if cache_key not in probabilities:
                            probabilities[cache_key] = producer.c.probabilities(
                                state["parameters"], observation["counts"]
                            )
                        p = probabilities[cache_key]
                        event = f"q{ordinal}"
                        template = producer.c.m.normalize(payload)
                        active = [template] if template in state["templates"] else []
                        sealed = dict(
                            observation,
                            event_id=event,
                            ordinal=ordinal,
                            prediction=p,
                            action=producer.c.m.action(p, 0.5),
                            state_hash=canonical_hash(state),
                            evidence_hash=canonical_hash(payload),
                            template_uses=active,
                        )
                        require(
                            sealed == row["sealed"]
                            and uses[event]["template_uses"] == active
                            and uses[event]["prediction_seal"] == canonical_hash(sealed),
                            "sealed_prediction",
                        )
                        for template in active:
                            admitted = admissions[template]
                            require(admitted["feedback_ordinal"] < ordinal, "early_template_use")
                            reuse.append(
                                dict(
                                    cluster_id=key[0],
                                    arm=arm,
                                    admission_event=admitted["event_id"],
                                    feedback_ordinal=admitted["feedback_ordinal"],
                                    use_event=event,
                                    use_ordinal=ordinal,
                                    future_window=ordinal >= 25,
                                    changed_predicates=observation["counts"]
                                    != producer.c.m.observe(payload, [])["counts"],
                                    template_hash=canonical_hash(template),
                                )
                            )
                        before = canonical_hash(state)
                        state["pending"][event] = sealed
                        require(
                            next(transitions)
                            == dict(
                                cluster_id=key[0],
                                seed=stream["seed"],
                                arm=arm,
                                operation="predict",
                                ordinal=ordinal,
                                before=before,
                                after=canonical_hash(state),
                                event_id=event,
                            ),
                            "prediction_transition",
                        )
                        truth = str(producer.c.m.decide(payload)["decision"])
                        require(truth == row["truth"], "independent_truth")
                        output.append(
                            dict(
                                row,
                                **score(p, truth, 0.5),
                                log_loss=-math.log(max(p[producer.c.m.LABELS.index(truth)], 1e-15)),
                                decisive_error=row["false_accept"] + row["false_reject"],
                            )
                        )
                    if ordinal >= 9:
                        released = releases[ordinal]
                        event = f"q{ordinal - 8}"
                        pending = state["pending"][event]
                        payload = worlds["online"][ordinal - 9]
                        label = str(producer.c.m.decide(payload)["decision"])
                        prefix = state["released"] + [label]
                        training = (
                            prefix[(ordinal * 7) % len(prefix)] if "shuffled" in arm else label
                        )
                        template = producer.c.m.normalize(payload)
                        added = (
                            "combined" in arm
                            and training == label
                            and template not in state["templates"]
                            and len(state["templates"]) < 32
                        )
                        require(
                            released["prediction_ordinal"] == ordinal - 8
                            and released["feedback_label"] == training
                            and released["verified_label"] == label
                            and released["released_prefix"] == prefix
                            and released["prediction_hash"] == canonical_hash(pending)
                            and released["template_added"] == added
                            and released["template"] == template
                            and released["acknowledgement"] == event,
                            "release_boundary",
                        )
                        before = canonical_hash(state)
                        if not arm.endswith("frozen"):
                            gradient = gradients[event]
                            require(
                                gradient["parameters_before"] == state["parameters"],
                                "coefficient_chain",
                            )
                            state["parameters"] = deepcopy(gradient["parameters_after"])
                        if added:
                            state["templates"].append(template)
                            state.setdefault("template_provenance", {})[template] = dict(
                                event_id=event,
                                evidence_hash=canonical_hash(payload),
                                verified_label=label,
                                binding_witnesses=producer.c.binding_provenance(payload),
                            )
                            admissions[template] = released
                        state["pending"].pop(event)
                        state["acks"].append(event)
                        state["released"].append(label)
                        require(canonical_hash(state) == released["checkpoint_hash"], "commit_hash")
                        require(
                            next(transitions)
                            == dict(
                                cluster_id=key[0],
                                seed=stream["seed"],
                                arm=arm,
                                operation="release",
                                ordinal=ordinal,
                                before=before,
                                after=canonical_hash(state),
                                event_id=event,
                            ),
                            "release_transition",
                        )
                        shuffled += int(training != label)
                    if ordinal == 24:
                        restart = selected["restart_rows"]
                        require(
                            len(restart) == 1
                            and restart[0]["before"]
                            == restart[0]["after"]
                            == canonical_hash(state),
                            "restart_parity",
                        )
                        snapshots[24] = deepcopy(state)
                snapshots[48] = state
                require(
                    not state["pending"]
                    and len(state["acks"]) == 48
                    and load(selected["final_state_refs"][0]) == state,
                    "tail_drain",
                )
                require(next(transitions, None) is None, "extra_transition")
                for row in selected["retention_rows"]:
                    snapshot = snapshots[row["checkpoint"]]
                    value = worlds["retention"][row["ordinal"] - 1]
                    counts = producer.c.m.observe(value, snapshot["templates"])["counts"]
                    p = producer.c.probabilities(snapshot["parameters"], counts)
                    truth = str(producer.c.m.decide(value)["decision"])
                    require(
                        row["state_hash"] == canonical_hash(snapshot)
                        and row["counts"] == counts
                        and row["prediction"] == p
                        and row["truth"] == truth
                        and row["seed"] == stream["seed"],
                        "untouched_retention",
                    )
                    output.append(
                        dict(
                            row,
                            **score(p, truth, 0.5),
                            log_loss=-math.log(max(p[producer.c.m.LABELS.index(truth)], 1e-15)),
                            decisive_error=row["false_accept"] + row["false_reject"],
                        )
                    )
                integrity.append(
                    dict(
                        cluster_id=key[0],
                        arm=arm,
                        delay=8,
                        prefix_only_shuffle=True,
                        shuffled_labels=shuffled,
                        restart_parity=True,
                        tail_drained=True,
                        no_retention_feedback=True,
                        admissions=len(admissions),
                        passed=True,
                    )
                )
            progress("reconstructed_streams", len(integrity) // 6, 24 - len(integrity) // 6)
    return output, integrity, reuse


def intervals(values: np.ndarray, names: list[str], tail: float) -> list[Json]:
    """Resample complete paired streams; repeated seeds stay inside their stream."""
    rng = np.random.default_rng(8846)
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
            random_seed=8846,
            direction="left_minus_right; lower loss is better",
        )
        for i, name in enumerate(names)
    ]


def gates(paired: list[Json], retained: list[Json]) -> Json:
    """Separate causal benefit, addition contribution, retention and head complexity."""
    causal = paired[0]["upper"] <= -0.01 and -paired[2]["upper"] > 0
    return dict(
        causal_benefit_score=int(causal),
        addition_benefit_score=int(causal and paired[1]["upper"] <= -0.005),
        retention_pass_score=int(all(r["upper"] <= 0.01 for r in retained)),
        categorical_noninferiority_score=int(paired[4]["upper"] <= 0.005),
    )


def reduce(work: Json) -> Json:
    """Compute registered endpoints and descriptive influence from authenticated rows."""
    rows, integrity, reuse = reconstruct(work)
    require(
        any(r["future_window"] and r["changed_predicates"] for r in reuse),
        "missing_future_template_reuse",
    )
    groups: dict[tuple[Any, ...], list[Json]] = defaultdict(list)
    for row in rows:
        if row["split"] == "retention" or row["ordinal"] >= 25:
            groups[(row["cluster_id"], row["arm"], row["split"], row["checkpoint"])].append(row)
    per_stream, retained, subgroups = [], [], []
    means: dict[tuple[Any, ...], Json] = {}
    for key, local in groups.items():
        metrics = {
            m: float(np.mean([r[m] for r in local]))
            for m in (
                "metric",
                "log_loss",
                "covered",
                "false_accept",
                "false_reject",
                "decisive_error",
            )
        }
        metrics["brier"] = metrics.pop("metric")
        means[key] = metrics
        base = dict(
            cluster_id=key[0], arm=key[1], seed=local[0]["seed"], denominator=len(local), **metrics
        )
        if key[2] == "online":
            per_stream.append(
                dict(
                    base,
                    decisive_count=sum(r["covered"] for r in local),
                    false_accepts=sum(r["false_accept"] for r in local),
                    false_rejects=sum(r["false_reject"] for r in local),
                    support_denominator=sum(r["truth"] == "support" for r in local),
                    nonsupport_denominator=sum(r["truth"] != "support" for r in local),
                )
            )
    streams = [s["stream_id"] for s in producer.manifest()]
    deltas = np.asarray(
        [
            [
                means[(s, a, "online", None)]["brier"] - means[(s, b, "online", None)]["brier"]
                for a, b in CONTRASTS
            ]
            for s in streams
        ]
    )
    names = [a + "_minus_" + b for a, b in CONTRASTS]
    paired = intervals(deltas, names, 0.01 / (2 * 5))
    retention_specs = [
        (a, step, metric)
        for a in producer.ARMS
        if not a.endswith("frozen")
        for step in (24, 48)
        for metric in ("brier", "decisive_error")
    ]
    retention_values = np.asarray(
        [
            [
                means[(s, a, "retention", step)][metric] - means[(s, a, "retention", 0)][metric]
                for a, step, metric in retention_specs
            ]
            for s in streams
        ]
    )
    retention_names = [f"{a}_step{step}_minus_0_{metric}" for a, step, metric in retention_specs]
    retention_intervals = intervals(
        retention_values, retention_names, 0.01 / (2 * len(retention_specs))
    )
    for s in streams:
        for a in producer.ARMS:
            for step in (24, 48):
                baseline, current = means[(s, a, "retention", 0)], means[(s, a, "retention", step)]
                retained.append(
                    dict(
                        cluster_id=s,
                        arm=a,
                        checkpoint=step,
                        denominator=12,
                        brier_change=current["brier"] - baseline["brier"],
                        log_loss_change=current["log_loss"] - baseline["log_loss"],
                        decisive_error_increase=current["decisive_error"]
                        - baseline["decisive_error"],
                    )
                )
    influence = [
        dict(
            omitted_stream=s,
            comparison=names[i],
            estimate_without_stream=float(np.delete(deltas[:, i], j).mean()),
            full_estimate=float(deltas[:, i].mean()),
            sign_changes=bool((np.delete(deltas[:, i], j).mean() < 0) != (deltas[:, i].mean() < 0)),
        )
        for j, s in enumerate(streams)
        for i in range(5)
    ]
    future = [r for r in rows if r["split"] == "online" and r["ordinal"] >= 25]
    for arm in producer.ARMS:
        for error in producer.ERRORS:
            local = [r for r in future if r["arm"] == arm and r["error_type"] == error]
            subgroups.append(
                dict(
                    arm=arm,
                    error_type=error,
                    denominator=len(local),
                    brier=float(np.mean([r["metric"] for r in local])),
                    log_loss=float(np.mean([r["log_loss"] for r in local])),
                    decisive_coverage=float(np.mean([r["covered"] for r in local])),
                    false_accepts=sum(r["false_accept"] for r in local),
                    false_rejects=sum(r["false_reject"] for r in local),
                    descriptive_only=True,
                )
            )
    passed = gates(paired, retention_intervals)
    return dict(
        rows=rows,
        per_stream_results=per_stream,
        paired_interval_rows=paired,
        retention_interval_rows=retention_intervals,
        causal_integrity_rows=integrity,
        untouched_retention_rows=retained,
        template_reuse_rows=reuse,
        influence_rows=influence,
        subgroup_descriptives=subgroups,
        independent_count=24,
        intended_count=12096,
        completed_count=len(rows),
        failed_count=0,
        censored_count=0,
        excluded_count=0,
        learning_audit_ready_score=1,
        **passed,
        next_head_recommendation=dict(
            read_only=True,
            next_head="categorical" if passed["categorical_noninferiority_score"] else "linear",
            reason="categorical within frozen0.005 Brier margin"
            if passed["categorical_noninferiority_score"]
            else "categorical fails frozen0.005 Brier margin; prefer simpler linear head",
            calibration_only_improvement=-(paired[0]["estimate"] - paired[1]["estimate"]),
            addition_specific_improvement=-paired[1]["estimate"],
            benefit_concentrated_in_one_stream=any(r["sign_changes"] for r in influence),
            no_posthoc_selection=True,
        ),
    )


def replay(path: Path) -> Json:
    """Recompute sealed reductions so rehashed aggregate edits cannot acquire trust."""
    value = json.loads(path.read_bytes())
    producer.upstream.bindings(
        value["code_config_hashes"] + value["source_artifact_hashes"], "operand_hash"
    )
    work = load(value["raw_shard_hashes"][0])
    if work:
        trusted = load(dict(path=str(SOURCE), sha256=PIN))
        producer.upstream.bindings(trusted["code_config_hashes"], "original_code_hash")
        require(
            value["raw_shard_hashes"][0]["sha256"] == trusted["raw_shard_hashes"][0]["sha256"],
            "original_evidence_hash",
        )
    expected = reduce(work) if work or value["learning_audit_ready_score"] else {}
    if work and value["verdict_class"] == "disqualified":
        expected.update({k: 0 for k in expected if k.endswith("_score")})
    for key, observed in expected.items():
        require(value[key] == observed, "reduction_drift:" + key)
    for receipt in value["validation_receipts"]:
        if "log_path" in receipt:
            log = Path(receipt["log_path"])
            producer.upstream.bindings(
                [
                    dict(
                        path=str(log if log.is_absolute() else ROOT / log),
                        sha256=receipt["log_sha256"],
                    )
                ],
                "validation_log",
            )
    return dict(passed=True, replay_passed=True)
