"""REQ-VERIFY-8212: separate shared calibration from structural decision value.

Original slots and held-out retention labels determine the audit. Repeated seeds
increase computation, while independent source counts stay unchanged.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import time
from typing import Any
from unittest.mock import patch

import numpy as np
import yaml

from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.verify import calibrated_memory_trajectory_8211 as upstream
from carnot.verify import learning_benefit_audit_8172 as qualified

Json = dict[str, Any]
m = upstream.m
legacy = qualified.previous
ROOT = upstream.ROOT
NAME = "experiment_8212_v709_memory_benefit_audit"
TASK = "exp8212-memory-benefit-audit"
MODULE = "python/carnot/verify/memory_benefit_audit_8212.py"
RUNNER = "python/carnot/reporting/memory_benefit_execution_8212.py"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_memory_benefit_audit_8212.py"
RUN_DATE = "20261006"
UPSTREAM = "results/experiment_8211_v709_calibrated_memory_trajectory.json"
UPSTREAM_HASH = "sha256:51fb7180404aff65936949496e4a9a16eddf19f2423dc6613d4cc1b4deba249f"
MODEL_SPECS: list[Json] = []
run_check = upstream.run_check


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Attribute current audit work to this invocation and flush actual counts."""
    print(f"[exp8212] phase={phase} completed={completed} pending={pending}", flush=True)


def statistics(rows: list[Json]) -> Json:
    """Reuse qualified original-slot bootstrap and add the calibration control."""
    slots: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["condition"], row["source_cluster_id"])
        legacy.equal(slots.setdefault(key, row["slot"]), row["slot"])
    with patch.object(legacy, "ARMS", m.ARMS):
        return qualified.statistics(rows)


def effects(rows: list[Json], arm: str, control: str) -> Json:
    """Probability movement cannot substitute for actual action movement."""
    grouped = {
        (r["condition"], r["slot"], r["seed"], r["arm"]): r for r in rows if r["metric"] == "brier"
    }
    probability = action = 0
    differences = []
    for key, row in grouped.items():
        if key[0] != "later_stream" or key[-1] != arm or row["status"] != "completed":
            continue
        other = grouped[(*key[:3], control)]
        probability += abs(row["prediction"] - other["prediction"]) > 1e-12
        action += m.engine.historical.radial.action(
            row["prediction"]
        ) != m.engine.historical.radial.action(other["prediction"])
        differences.append(abs(row["prediction"] - other["prediction"]))
    reduced = m.engine.historical.reductions(rows)
    gain = {
        metric + "_gain": reduced["/".join(["later_stream", control, metric])]["mean"]
        - reduced["/".join(["later_stream", arm, metric])]["mean"]
        for metric in ["typed_cost", "brier"]
        if "/".join(["later_stream", arm, metric]) in reduced
    }
    return dict(
        gain,
        comparison=f"{arm} versus {control}",
        probability_movement=int(probability),
        action_movement=int(action),
        mean_absolute_probability_movement=float(np.mean(differences)) if differences else None,
        decision_benefit_claim=False,
        repeat_credit="movement counts are seed executions, not independent sources",
    )


def reconstruct(value: Json, raw: Path) -> Json:
    """Scalar predictions and original labels rebuild losses after the input seal."""
    sealed = json.loads(Path(value["trajectory_path"]).read_bytes())
    manifests = value["input_manifests"]
    public = json.loads(Path(manifests["stream_feature_manifest"]["path"]).read_bytes())["rows"]
    retained = json.loads(Path(manifests["retention_feature_manifest"]["path"]).read_bytes())[
        "rows"
    ]
    rows: list[Json] = []
    centers: list[Json] = []
    states = sealed["states"]
    for index, (state, releases) in enumerate(zip(states, sealed["releases"], strict=True)):
        with patch.object(m.engine, "ARMS", m.ARMS):
            heads = m.genesis(public, state["seed"])["arms"]
        deliveries = {r["label_slot"]: r["outcome"] for r in releases}
        updates = {e["slot"]: e["heads"] for e in state["events"] if e["kind"] == "durable_update"}
        for row, issued in zip(public, state["issued"], strict=True):
            issued = dict(issued, predictions={a: issued["predictions"][a] for a in m.ARMS})
            prediction = {
                a: None
                if row["values"] is None
                else m.scalar_probability(h, state["geometry"], row["values"])
                for a, h in heads.items()
            }
            legacy.equal(prediction, issued["predictions"])
            legacy.equal(canonical_hash(issued["predictions"]), issued["prediction_hash"])
            if row["slot"] >= 65:
                target = deliveries.get(
                    row["slot"], dict(y=None, exclusion_reason="feedback_unresolved_tail")
                )
                rows.extend(
                    m.engine.historical.scored(row, issued, target, "later_stream", state["seed"])
                )
            heads = updates.get(row["slot"], heads)
        progress("independent_preupdate_seed", index + 1, len(states) - index - 1)
    atomic_json(
        raw / "preupdate_seal.json",
        dict(rows=rows, retention_predictions=sealed["evidence"]["retention_predictions"]),
    )
    progress("preupdate_and_retention_predictions_sealed", len(rows), 0)
    vault = m.engine.historical.LabelVault(
        Path(manifests["evaluator_label_manifests"]["retention"]["path"]), retained
    )
    for prediction in sealed["evidence"]["retention_predictions"]:
        prediction = dict(prediction, predictions={a: prediction["predictions"][a] for a in m.ARMS})
        row = retained[prediction["slot"] - 1]
        state = states[prediction["seed"] - 101]
        scalar = {
            a: None
            if row["values"] is None
            else m.scalar_probability(h, state["geometry"], row["values"])
            for a, h in state["arms"].items()
        }
        legacy.equal(scalar, prediction["predictions"])
        legacy.equal(canonical_hash(prediction["predictions"]), prediction["prediction_hash"])
        label = vault.release(row["slot"], 256, sealed=True, retention=True)
        rows.extend(
            m.engine.historical.scored(row, prediction, label, "retention", prediction["seed"])
        )
        if row["values"] is not None and label["y"] is not None:
            head = deepcopy(state["arms"]["error_center"])
            head["weights"] = [0.0] * len(head["weights"])
            calibration = m.scalar_probability(head, state["geometry"], row["values"])
            without_new = deepcopy(state["arms"]["error_center"])
            without_new["weights"][16:] = [0.0] * len(without_new["weights"][16:])
            existing_centers = m.scalar_probability(without_new, state["geometry"], row["values"])
            z = [
                (x - v) / s
                for x, v, s in zip(
                    row["values"], state["geometry"]["mean"], state["geometry"]["std"], strict=True
                )
            ]
            activation = sum(
                np.exp(
                    -sum((x - v) ** 2 for x, v in zip(z, c["x"], strict=True))
                    / (2 * state["geometry"]["sigma"] ** 2)
                )
                for c in head["centers"][16:]
            )
            centers.append(
                dict(
                    seed=state["seed"],
                    slot=row["slot"],
                    source_cluster_id=row["source_cluster_id"],
                    shared_calibration_probability=calibration,
                    new_center_probability_contribution=scalar["error_center"] - existing_centers,
                    center_probability_contribution=scalar["error_center"] - calibration,
                    activation_overlap=float(activation),
                    retention_error=(scalar["error_center"] - label["y"]) ** 2,
                )
            )
    converted = [dict(r, arm=m.NAMES[r["arm"]]) for r in rows]
    legacy.equal(converted, value["rows"])
    directory = Path(value["trajectory_path"]).parent
    baseline = json.loads((directory / "uninterrupted/final.json").read_bytes())
    resumed = json.loads((directory / "restart/final.json").read_bytes())
    restart = m.stable(baseline) == m.stable(resumed) == m.stable(states[0]) and upstream.journal(
        directory / "uninterrupted/issued.jsonl"
    ) == upstream.journal(directory / "restart/issued.jsonl")
    labels: list[int | None] = [None] * 256
    for release in sealed["releases"][0]:
        labels[release["label_slot"] - 1] = release["outcome"]["y"]
    labels[159] = None if labels[159] is None else 1 - labels[159]
    progress("before_future_mutation_benchmark", 0, 1)
    mutated = m.run(public, labels, 101)
    progress("after_future_mutation_benchmark", 1, 0)
    invariant = mutated["issued"][:180] == states[0]["issued"][:180]
    atomic_json(
        raw / "future_mutation.json",
        dict(mutated_issued=mutated["issued"], changed_origin_slot=160, unchanged_prefix=180),
    )
    legacy.equal(True, restart and invariant)
    panel = [
        [r for r in centers if r["source_cluster_id"] == source]
        for source in sorted({r["source_cluster_id"] for r in centers})
    ]
    xs = [float(np.mean([r["activation_overlap"] for r in group])) for group in panel]
    ys = [float(np.mean([r["retention_error"] for r in group])) for group in panel]
    correlation = (
        float(np.corrcoef(xs, ys)[0, 1]) if xs and np.std(xs) > 0 and np.std(ys) > 0 else None
    )
    return dict(
        rows=converted,
        legacy_rows=rows,
        causal_checks=dict(
            restart_equal=restart,
            future_mutation_invariant=invariant,
            child_exit_codes=[r["actual_exit"] for r in value["child_exit_rows"]],
        ),
        center_contribution_rows=centers,
        activation_retention_correlation=dict(
            correlation=correlation,
            diagnostic_only=True,
            winner_selection=False,
            independent_count=len(panel),
            unit="seed means per original retention source; descriptive association only",
        ),
    )


def measure(root: Path, raw: Path, *, fixture: bool = False, **kwargs: Any) -> Json:
    """Missing upstream evidence blocks; owned arithmetic failures disqualify."""
    started, wall = time.monotonic_ns(), time.time_ns()
    raw.mkdir(parents=True, exist_ok=True)
    binder = m.engine.methods.Custody(raw)
    binder.upstream = "exp8211-calibrated-memory-trajectory"
    work: Json = dict(
        input_ready=0,
        owned_failure=None,
        rows=[],
        legacy_rows=[],
        causal_checks={},
        center_contribution_rows=[],
        activation_retention_correlation={},
    )
    progress("preconditions_start")
    try:
        for tool in ["python", "pytest", "coverage", "ruff", "mypy"]:
            path = ROOT / ".venv/bin" / tool
            binder.require(path, "runtime_" + tool, True, path.is_file())
        probe = raw / "write_probe.json"
        atomic_json(probe, dict(private_writable=True))
        binder.bind(probe)
        path = root / UPSTREAM
        value = binder.read(path, None if fixture else UPSTREAM_HASH)
        for field, expected in [
            ("experiment_id", 8211),
            ("learning_trajectory_ready_score", 1),
            ("required_checks_passed", True),
            ("flagged_adversarial", False),
        ]:
            binder.require(path, field, expected, value.get(field))
        terminal = m.engine.methods.historical.terminal(path, value, binder)
        binder.require(path, "terminal.report.passed", True, terminal["report"].get("passed"))
        binder.bind(ROOT / m.PROTOCOL, m.PROTOCOL_HASH)
        exclusions = ROOT / "ops/exclusion_manifest.yaml"
        binder.bind(exclusions)
        binder.require(
            exclusions,
            "experiment8212_not_retired",
            True,
            not any(
                r.get("experiment_id") == 8212
                for r in yaml.safe_load(exclusions.read_text()).get("retired_experiments", [])
            ),
        )
        binder.bind(Path(value["trajectory_path"]), value["trajectory_sha256"])
        # The primary contains large duplicated row arrays. Release this copy
        # before the upstream verifier decodes its own copy of the same bytes.
        del value
        progress("before_upstream_cold_benchmark")
        binder.require(path, "primitive_trajectory_replay", True, upstream.replay(path))
        progress("after_upstream_cold_benchmark", 1, 0)
        value = binder.read(path, None if fixture else UPSTREAM_HASH)
        for operand in Path(
            os.environ.get("CARNOT_8212_PREFLIGHT", str(raw / "absent_preflight"))
        ).glob("*"):
            binder.bind(operand)
        for role in ["stream", "retention"]:
            for ref in [
                value["input_manifests"][role + "_feature_manifest"],
                value["input_manifests"]["evaluator_label_manifests"][role],
            ]:
                binder.bind(Path(ref["path"]), ref["sha256"])
        progress("preconditions_complete", len(binder.checks), 0)
        work.update(reconstruct(value, raw))
        work.update(input_ready=1, upstream_primary=str(path), upstream_sha256=sha256_file(path))
    except m.engine.methods.historical.InputFailure:
        progress("external_operand_blocked", len(binder.checks), 0)
    except (ValueError, KeyError, TypeError, OSError) as error:
        work["owned_failure"] = str(error)
        progress("owned_reduction_disqualified")
    work.update(
        gate_check_summary=binder.checks,
        source_artifact_hashes=binder.refs,
        cited_upstream_artifacts=[
            dict(
                experiment_id=8211,
                path=str(root / UPSTREAM),
                sha256=next(
                    (r["sha256"] for r in binder.refs if r["path"] == str(root / UPSTREAM)), None
                ),
                fields_imported=[
                    "trajectory_path",
                    "trajectory_sha256",
                    "input_manifests",
                    "learning_trajectory_ready_score",
                ],
            )
        ],
        preconditions_checked=dict(
            runtime_executable=str(ROOT / ".venv/bin/python"),
            private_writable_storage=True,
            no_model_load=True,
        ),
        code_config_hashes={
            p: sha256_file(ROOT / p)
            for p in [
                MODULE,
                RUNNER,
                CLI,
                TEST,
                m.PROTOCOL,
                m.MODULE,
                legacy.MODULE,
                qualified.MODULE,
            ]
        },
        raw_shard_hashes=[
            dict(path=str(p), sha256=sha256_file(p)) for p in sorted(raw.glob("*.json"))
        ],
        started_monotonic_ns=started,
        started_wall_ns=wall,
        ended_monotonic_ns=time.monotonic_ns(),
        duration_s=(time.monotonic_ns() - started) / 1e9,
        fixture_mode=fixture,
    )
    atomic_json(raw / "measurement.json", work)
    progress("measurement_complete", len(work["rows"]), 0)
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """A qualified null is complete; exposed oracle success stays circular."""
    checked = (
        bool(receipts)
        and all(r["passed"] and r.get("normal_exit", True) for r in receipts)
        and work["owned_failure"] is None
    )
    stats = statistics(work["legacy_rows"])
    calibration = effects(work["legacy_rows"], "calibration_only", "frozen_qwen_offset")
    structural = effects(work["legacy_rows"], "error_center", "fixed_public_center")
    success = stats["h2_passed"] and stats["retention_passed"] and structural["action_movement"] > 0
    verdict = (
        "disqualified"
        if not checked
        else "blocked"
        if not work["input_ready"]
        else "circular_positive"
        if success
        else "null"
    )
    structural["decision_benefit_claim"] = bool(success and checked and work["input_ready"])
    units = [
        r
        for r in work["rows"]
        if r["condition"] == "later_stream"
        and r["seed"] == 101
        and r["arm"] == "frozen_qwen_offset"
        and r["metric"] == "brier"
    ]
    values = dict(
        work,
        experiment_id=8212,
        task_id=TASK,
        milestone="2026.10.709",
        run_date=RUN_DATE,
        honest_verdict="complete_" + verdict + "_independent_memory_benefit_audit",
        verdict_class=verdict,
        required_checks_passed=checked,
        learning_audit_ready_score=int(checked and bool(work["input_ready"])),
        H2=stats,
        calibration_only_effect=calibration,
        structural_memory_effect=structural,
        retention_rows=stats["retention_rows"],
        h2_development_signal_score=int(success and checked and bool(work["input_ready"])),
        verifier_is_oracle=True,
        exposure_scope="historically_exposed_development_replay",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=MODEL_SPECS,
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        intended_count=192,
        completed_count=stats["completed_count"],
        independent_count=stats["completed_count"],
        excluded_count=sum(r["status"] == "excluded" for r in units) if units else 192,
        censored_count=sum(r["status"] == "censored" for r in units),
        failed_count=0,
        random_seed=legacy.CONFIG["seed"],
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        acceptance_gates=json.loads((ROOT / m.PROTOCOL).read_text())["statistical_plan"],
        flagged_adversarial=False,
        reductions=m.engine.historical.reductions(work["rows"]),
        methodology_note="Independent scalar pre-update predictions and original evaluator labels; unchanged V707 H2 moving original-slot bootstrap, seeds averaged inside source. Calibration, action movement and structural centers are separate. No independent lifelong-learning claim.",
    )
    from scripts.experiment_template import normalize_artifact_for_template_write

    values = normalize_artifact_for_template_write(values)
    values["field_principles"] = {
        k: "Exact byte-bound source evidence; seeds are repeated executions; blocked operands differ from measured zero; exposed labels grant no independent generalization."
        for k in values
    }
    values["reproducibility_checksum"] = canonical_hash(values)
    return values


def replay(path: Path) -> bool:
    """Fresh reduction rejects changed primitives even after an aggregate rehash."""
    from tempfile import TemporaryDirectory

    try:
        value = json.loads(path.read_bytes())
        checksum = value.pop("reproducibility_checksum")
        if checksum != canonical_hash(value) or value["experiment_id"] != 8212:
            return False
        headline = build(
            value,
            Path(value["terminal_validation_sidecar_path"]).parent,
            value["validation_receipts"],
        )
        fields = [
            "H2",
            "retention_rows",
            "calibration_only_effect",
            "structural_memory_effect",
            "learning_audit_ready_score",
            "h2_development_signal_score",
            "completed_count",
            "independent_count",
            "excluded_count",
            "censored_count",
            "verdict_class",
            "honest_verdict",
            "required_checks_passed",
            "reductions",
        ]
        if any(value[k] != headline[k] for k in fields):
            return False
        for name, digest in value["code_config_hashes"].items():
            legacy.equal(digest, sha256_file(ROOT / name))
        for ref in value["raw_shard_hashes"] + value["source_artifact_hashes"]:
            legacy.equal(ref["sha256"], sha256_file(Path(ref.get("snapshot_path", ref["path"]))))
        for receipt in value["validation_receipts"]:
            for stream in ["stdout", "stderr"]:
                if stream + "_path" in receipt:
                    legacy.equal(
                        receipt[stream + "_sha256"], sha256_file(Path(receipt[stream + "_path"]))
                    )
        if value["input_ready"]:
            original = Path(value["upstream_primary"])
            legacy.equal(value["upstream_sha256"], sha256_file(original))
            if not upstream.replay(original):
                return False
            with TemporaryDirectory(prefix="carnot-8212-cold-") as directory:
                reconstructed = reconstruct(json.loads(original.read_bytes()), Path(directory))
            for key, observed in reconstructed.items():
                legacy.equal(value[key], observed)
        return True
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False
