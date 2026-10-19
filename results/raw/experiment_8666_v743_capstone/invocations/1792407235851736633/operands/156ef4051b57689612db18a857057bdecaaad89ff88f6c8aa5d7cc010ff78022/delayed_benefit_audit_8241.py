"""REQ-VERIFY-8241: reconstruct delayed decisions before measuring their benefit.

Committed heads and opaque evaluator fragments supply the evidence. Repeated
seeds are averaged within original sources and cannot create generalization.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
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
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.verify import qualified_delayed_learning_8240 as upstream
from carnot.verify import learning_audit_8144 as bootstrapper

Json = dict[str, Any]
k = upstream.legacy.k
ROOT = upstream.ROOT
NAME = "experiment_8241_v712_delayed_benefit_audit"
TASK = "exp8241-delayed-benefit-audit"
MODULE = "python/carnot/verify/delayed_benefit_audit_8241.py"
RUNNER = MODULE
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_delayed_benefit_audit_8241.py"
OWNED = [MODULE, CLI]
RUN_DATE = "20261007"
UPSTREAM = "results/" + upstream.NAME + ".json"
UPSTREAM_HASH = "sha256:5dcfe543abe6e6c28dd108d188e7771269adab2478f0a10a9d11f1770e40ac67"
MODEL_SPECS: list[Json] = []
H2 = upstream.legacy.qualified.PROTOCOL_VALUE["H2"]
run_check = upstream.run_check
equal = bootstrapper.equal


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual phase counts so a bounded parent can observe current work."""
    print(f"[exp8241] phase={phase} completed={completed} pending={pending}", flush=True)


def scored(row: Json, issued: Json, y: Any, condition: str, seed: int, arm: str) -> Json:
    """Keep missing slots and escalation cost separate from proper-loss support."""
    p, action = issued["p"], issued["action"]
    status = (
        "censored"
        if condition == "later_stream" and row["slot"] > 236
        else "excluded"
        if p is None or y is None
        else "completed"
    )
    return dict(
        slot=row["slot"],
        unit_id=row["unit_id"],
        source_cluster_id=row["source_cluster_id"],
        seed=seed,
        arm=arm,
        condition=condition,
        status=status,
        missing=p is None,
        exclusion_reason=None
        if status == "completed"
        else "feedback_unresolved_tail"
        if status == "censored"
        else "missing_operand",
        p=p,
        y=y,
        action=action,
        metric="typed_decision_cost",
        denominator=1,
        numerator=0.5
        if action == "escalate"
        else k.kernel.rule.base.loss(action, y)
        if y is not None
        else None,
        brier=(p - y) ** 2 if p is not None and y is not None else None,
        calibration_residual=y - p if p is not None and y is not None else None,
        false_accept=int(action == "accept" and y == 1) if y is not None else None,
    )


def reconstruct(data: Json, raw: Path) -> Json:
    """Rebuild issue-time heads before opening independent retention fragments."""
    public, retained = data["public"], data["retained"]
    equal(list(range(1, 257)), [r["slot"] for r in public])
    equal(list(range(1, 65)), [r["slot"] for r in retained])
    if (
        len({r["source_cluster_id"] for r in retained}) != 64
        or len({r["source_cluster_id"] for r in public}) != 256
        or {r["source_cluster_id"] for r in public} & {r["source_cluster_id"] for r in retained}
    ):
        raise ValueError("source_roster")
    stream_vault = k.calibration.engine.historical.LabelVault(
        Path(data["stream_label_path"]), public
    )
    rows, installs, checked_retention = [], [], []
    for index, states in enumerate(data["states"]):
        equal(set(k.ARMS), set(states))
        access = []
        for arm, state in states.items():
            seed, model = state["seed"], dict(kind="input")
            equal(canonical_hash(public), state["baseline_hash"])
            releases = [r for r in data["release_log"] if r["seed"] == seed and r["arm"] == arm]
            equal(list(range(1, 237)), [r["label_slot"] for r in releases])
            for release in releases:
                equal(release["label_slot"] + 20, release["release_slot"])
            access.append(
                [(r["label_slot"], r["release_slot"], r["outcome"]["y"]) for r in releases]
            )
            targets = {r["label_slot"]: r["outcome"]["y"] for r in releases}
            events: dict[int, list[Json]] = {}
            for event in state["events"]:
                events.setdefault(event["slot"], []).append(event)
            equal(256, len(state["issued"]))
            for row, issued in zip(public, state["issued"], strict=True):
                p = k.predict(model, row)
                equal(
                    dict(
                        slot=row["slot"],
                        p=p,
                        action=k.kernel.rule.action(p, row["baseline_action"]),
                        model_sha256=canonical_hash(model),
                    ),
                    issued,
                )
                if row["slot"] >= 65:
                    rows.append(
                        scored(row, issued, targets.get(row["slot"]), "later_stream", seed, arm)
                    )
                for event in events.get(row["slot"], []):
                    if event["kind"] == "release_feedback":
                        original = stream_vault.release(
                            event["label_slot"], event["slot"], sealed=True
                        )
                        equal(original["y"], event["y"])
                        equal(targets[event["label_slot"]], event["y"])
                        equal(event["label_slot"] + 20, event["slot"])
                    if event["kind"] == "commit_candidate":
                        if any(
                            i + 20 > event["slot"] or not k.kernel.roles.bucket(public[i - 1])
                            for i in event["fit_ids"]
                        ):
                            raise ValueError("future_fit")
                    if event["kind"] == "admit_once":
                        equal(canonical_hash(model), event["before_hash"])
                        if any(i + 20 > event["slot"] for i in event["labels"]):
                            raise ValueError("future_admission")
                        model = event["mixture"]
                        equal(canonical_hash(model), event["after_hash"])
                        installs.append(
                            dict(
                                seed=seed,
                                arm=arm,
                                install_slot=event["slot"],
                                step=event["step"],
                                later_predictions_reached=sum(
                                    r["slot"] > event["slot"] and r["p"] is not None for r in public
                                ),
                                disposition="installed" if event["step"] else "rejected",
                            )
                        )
            equal(model, state["model"])
            predictions = [
                r for r in data["retention_predictions"] if r["seed"] == seed and r["arm"] == arm
            ]
            equal(64, len(predictions))
            for row, issued in zip(retained, predictions, strict=True):
                p = k.predict(model, row)
                equal(
                    dict(
                        slot=row["slot"],
                        unit_id=row["unit_id"],
                        source_cluster_id=row["source_cluster_id"],
                        seed=seed,
                        arm=arm,
                        p=p,
                        action=k.kernel.rule.action(p, row["baseline_action"]),
                        missing=p is None,
                        state_sha256=canonical_hash(state),
                        labels_opened=False,
                    ),
                    issued,
                )
                checked_retention.append(issued)
        equal([access[0]] * len(access), access)
        progress("cold_issue_seed", index + 1, len(data["states"]) - index - 1)
    sealed_ns = time.monotonic_ns()
    atomic_json(
        raw / "prediction_seal.json",
        dict(rows=rows, retention_predictions=checked_retention, predictions_sealed_ns=sealed_ns),
    )
    progress("retention_predictions_authenticated_and_sealed", len(checked_retention))
    opened_ns = time.monotonic_ns()
    vault = k.calibration.engine.historical.LabelVault(Path(data["retention_label_path"]), retained)
    labels = {
        r["slot"]: vault.release(r["slot"], 256, sealed=True, retention=True)["y"] for r in retained
    }
    for issued in checked_retention:
        rows.append(
            scored(
                retained[issued["slot"] - 1],
                issued,
                labels[issued["slot"]],
                "retention",
                issued["seed"],
                issued["arm"],
            )
        )
    progress("retention_labels_opened", 64)
    return dict(
        rows=rows,
        install_exposure_rows=installs,
        retention_access=dict(predictions_sealed_ns=sealed_ns, labels_opened_ns=opened_ns),
    )


def statistics(rows: list[Json]) -> Json:
    """Apply the unchanged source bootstrap and every per-seed harm safeguard."""
    grouped: dict[tuple[str, int], dict[tuple[int, str], Json]] = {}
    seeds = sorted({r["seed"] for r in rows})
    identities: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["condition"], row["source_cluster_id"])
        equal(identities.setdefault(key, row["slot"]), row["slot"])
        group = grouped.setdefault((row["condition"], row["slot"]), {})
        pair = (row["seed"], row["arm"])
        if pair in group:
            raise ValueError("duplicate_source_arm")
        group[pair] = row
    panels: Json = dict(later_stream=[], retention=[])
    for (condition, slot), group in sorted(grouped.items()):
        equal({(s, a) for s in seeds for a in k.ARMS}, set(group))
        complete = all(r["status"] == "completed" for r in group.values())
        first = next(iter(group.values()))
        equal(
            [first["source_cluster_id"]] * len(group),
            [r["source_cluster_id"] for r in group.values()],
        )
        equal([first["y"]] * len(group), [r["y"] for r in group.values()])
        means = {
            a: {
                m: float(np.mean([group[s, a][m] for s in seeds])) if complete else None
                for m in ["numerator", "brier", "calibration_residual", "false_accept"]
            }
            for a in k.ARMS
        }
        gain = (
            means["global_only"]["numerator"] - means["global_plus_group"]["numerator"]
            if complete
            else None
        )
        panels[condition].append(
            dict(
                slot=slot,
                source_cluster_id=first["source_cluster_id"],
                unit_id=first["unit_id"],
                status=first["status"],
                y=first["y"],
                gain=gain,
                numerator=gain,
                denominator=1,
                arms=means,
            )
        )
    later = panels["later_stream"]
    completed = [r for r in later if r["gain"] is not None]
    classes = Counter(r["y"] for r in completed)
    gains = {r["slot"]: r["gain"] for r in later}
    progress("before_moving_slot_bootstrap", 0, 3)
    with patch.dict(bootstrapper.CONFIG, seed=H2["seed"]):
        intervals = [
            bootstrapper.bootstrap([gains.get(i) for i in range(65, 257)], length)
            for length in [16, 8, 32]
        ]
    progress("after_moving_slot_bootstrap", 3, 0)
    safety = []
    for condition, panel in panels.items():
        for seed in seeds:
            for arm in ["global_plus_group"] if condition == "later_stream" else k.ARMS[1:]:
                for control in (
                    [a for a in k.ARMS if a != arm]
                    if condition == "later_stream"
                    else [a for a in ["frozen", "global_only"] if a != arm]
                ):
                    paired = [
                        (
                            grouped[condition, r["slot"]][seed, arm],
                            grouped[condition, r["slot"]][seed, control],
                        )
                        for r in panel
                        if all(
                            grouped[condition, r["slot"]][seed, a]["status"] == "completed"
                            for a in [arm, control]
                        )
                    ]
                    counts = Counter(a["y"] for a, b in paired)
                    cost = (
                        float(np.mean([a["numerator"] - b["numerator"] for a, b in paired]))
                        if paired
                        else None
                    )
                    brier = (
                        float(np.mean([a["brier"] - b["brier"] for a, b in paired]))
                        if paired
                        else None
                    )
                    extra = sum(a["false_accept"] and not b["false_accept"] for a, b in paired)
                    support = (
                        len(paired) >= (48 if condition == "retention" else 128)
                        and min(counts[0], counts[1]) >= 8
                    )
                    passed = support and cost <= 0.02 and brier <= 0.01 and extra == 0
                    safety.append(
                        dict(
                            condition=condition,
                            seed=seed,
                            arm=arm,
                            comparator=control,
                            complete_count=len(paired),
                            class_counts={str(y): counts[y] for y in [0, 1]},
                            cost_worsening=cost,
                            brier_worsening=brier,
                            extra_false_accepts=extra,
                            passed=bool(passed),
                        )
                    )
    blocks = sum(
        any(gains.get(i) is not None for i in range(start, start + 16))
        for start in range(65, 257, 16)
    )
    operands = dict(
        complete_sources=len(completed) >= 128,
        per_class=min(classes[0], classes[1]) >= 8,
        nonoverlapping_blocks=blocks >= 8,
        valid_draws=intervals[0]["valid_draws"] >= 9500,
        lower_gain=intervals[0]["lower_bound"] is not None and intervals[0]["lower_bound"] > 0.02,
        improved_sources=sum(r["gain"] > 0 for r in completed) >= 5,
        per_seed_safety=bool(seeds)
        and all(r["passed"] for r in safety if r["condition"] == "later_stream"),
    )
    reductions = []
    for condition in panels:
        for arm in k.ARMS:
            rs = [r for r in rows if r["condition"] == condition and r["arm"] == arm]
            for metric in ["numerator", "brier", "calibration_residual", "false_accept"]:
                values = [r[metric] for r in rs if r[metric] is not None]
                reductions.append(
                    dict(
                        condition=condition,
                        arm=arm,
                        metric="typed_decision_cost" if metric == "numerator" else metric,
                        numerator=sum(values),
                        denominator=len(values),
                        intended_denominator=len(rs),
                        mean=float(np.mean(values)) if values else None,
                    )
                )
    return dict(
        protocol=H2,
        measured_here=True,
        passed=all(operands.values()),
        operands=operands,
        failed_conditions=[key for key, passed in operands.items() if not passed],
        completed_count=len(completed),
        class_counts={str(y): classes[y] for y in [0, 1]},
        nonoverlapping_blocks=blocks,
        improved_sources=sum(r["gain"] > 0 for r in completed),
        per_source_deltas=later,
        block_bootstrap_diagnostics=intervals,
        retention_rows=panels["retention"],
        retention_passed=bool(seeds)
        and all(r["passed"] for r in safety if r["condition"] == "retention"),
        excess_brier_rows=safety,
        reductions=reductions,
    )


def authenticate(root: Path, raw: Path, work: Json) -> Json:
    """Exact terminal and primitive hashes authorize the current qualified input."""
    gate, bind = k.frozen.gate, k.frozen.bind
    path = root / UPSTREAM
    gate(work, path, "exists", True, True if path.is_file() else None)
    value = json.loads(path.read_bytes())
    gate(work, path, "input_schema", True, isinstance(value, dict))
    for field, expected in [
        ("experiment_id", 8240),
        ("task_id", upstream.TASK),
        ("utility_trajectory_ready_score", 1),
        ("required_checks_passed", True),
        ("flagged_adversarial", False),
        ("retention_labels_opened", False),
    ]:
        gate(work, path, field, expected, value.get(field))
    work["input_snapshot"] = str(bind(work, path, UPSTREAM_HASH, raw))
    terminal = Path(value["terminal_validation_sidecar_path"])
    bind(work, terminal, sha256_file(terminal), raw)
    receipt = Path(json.loads(terminal.read_bytes())["publication"]["sidecar_path"])
    gate(
        work,
        path,
        "terminal.report.passed",
        True,
        read_bound_sidecar(path, receipt)["report"]["passed"],
    )
    bind(work, receipt, sha256_file(receipt), raw)
    for ref in value["code_config_hashes"] + value["raw_shard_hashes"]:
        bind(work, Path(ref["path"]), ref["sha256"], raw)
    for role in ["stream", "retention"]:
        for ref in [
            value["input_manifests"][role + "_feature_manifest"],
            value["input_manifests"]["evaluator_label_manifests"][role],
        ]:
            bind(work, Path(ref["path"]), ref["sha256"], raw)
    work["upstream_dispositions"] = [
        dict(
            experiment_id=8240,
            path=str(path),
            sha256=UPSTREAM_HASH,
            honest_verdict=value["honest_verdict"],
            verdict_class=value["verdict_class"],
        )
    ]
    historical = ROOT / "results/experiment_8225_v711_delayed_utility_learning.json"
    bind(work, historical, upstream.validation.ORIGINAL_SHA256, raw)
    old = json.loads(historical.read_bytes())
    work["upstream_dispositions"].append(
        dict(
            experiment_id=8225,
            path=str(historical),
            sha256=sha256_file(historical),
            honest_verdict=old["honest_verdict"],
            verdict_class=old["verdict_class"],
        )
    )
    previous_audit = ROOT / "results/experiment_8226_learning_audit.json"
    bind(
        work,
        previous_audit,
        "sha256:0a379f79fac44727419151008566f64a1546fa6ceff60a723888523f0f3d9c1e",
        raw,
    )
    prior = json.loads(previous_audit.read_bytes())
    work["upstream_dispositions"].append(
        dict(
            experiment_id=8226,
            path=str(previous_audit),
            sha256=sha256_file(previous_audit),
            honest_verdict=prior["honest_verdict"],
            verdict_class=prior.get("verdict_class"),
            gate_check_summary=prior["gate_check_summary"],
            original_schema_preserved=True,
        )
    )
    return data_from_value(value)


def data_from_value(value: Json) -> Json:
    """Import primitive operands from the authenticated primary and sealed files."""
    public = json.loads(Path(value["trajectory_path"]).read_bytes())["rows"]
    retained = k.prepare(
        json.loads(
            Path(value["input_manifests"]["retention_feature_manifest"]["path"]).read_bytes()
        )["rows"]
    )
    return dict(
        public=public,
        retained=retained,
        states=value["states"],
        release_log=value["release_log"],
        retention_predictions=value["retention_predictions"],
        retention_label_path=value["input_manifests"]["evaluator_label_manifests"]["retention"][
            "path"
        ],
        stream_label_path=value["input_manifests"]["evaluator_label_manifests"]["stream"]["path"],
    )


def measure(
    root: Path, raw: Path, *, fixture: bool = False, stream_path: Path | None = None, **kwargs: Any
) -> Json:
    """External operand failures block; failures after authentication disqualify."""
    started, wall = time.monotonic_ns(), time.time_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    work: Json = dict(
        checks=[],
        refs=[],
        input_ready=0,
        owned_failure="",
        rows=[],
        install_exposure_rows=[],
        retention_access={},
        upstream_dispositions=[],
        fixture_mode=fixture,
    )
    progress("preconditions_start")
    try:
        with TemporaryDirectory(prefix="carnot-8241-probe-") as directory:
            probe = Path(directory) / "write_probe"
            probe.write_bytes(b"private writable scratch")
            k.frozen.gate(
                work,
                probe,
                "private_scratch_writable",
                True,
                probe.read_bytes() == b"private writable scratch",
            )
        for tool in ["python", "pytest", "coverage", "ruff", "mypy"]:
            path = ROOT / ".venv/bin" / tool
            k.frozen.gate(work, path, "runtime_" + tool, True, path.is_file())
        k.frozen.bind(work, ROOT / k.frozen.PROTOCOL, k.frozen.PIN, raw)
        k.frozen.gate(
            work, raw, "storage_available", True, shutil.disk_usage(raw).free > 32 * 1024 * 1024
        )
        exclusion = ROOT / "ops/exclusion_manifest.yaml"
        k.frozen.bind(work, exclusion, sha256_file(exclusion), raw)
        inventory = yaml.safe_load(exclusion.read_bytes())
        k.frozen.gate(
            work,
            exclusion,
            "experiment8241_not_retired",
            True,
            not any(
                r.get("experiment_id") == 8241
                for key in ["retired", "retired_experiments"]
                for r in inventory.get(key, [])
            ),
        )
        if fixture and stream_path:
            k.frozen.gate(
                work,
                stream_path,
                "private_fixture",
                True,
                not stream_path.resolve().is_relative_to(ROOT / "results"),
            )
            work["input_snapshot"] = str(
                k.frozen.bind(work, stream_path, sha256_file(stream_path), raw)
            )
            data = json.loads(stream_path.read_bytes())
        else:
            data = authenticate(root, raw, work)
        for role in ["stream", "retention"]:
            labels = Path(data[role + "_label_path"])
            k.frozen.bind(work, labels, sha256_file(labels), raw)
        atomic_json(raw / "audit_inputs.json", data)
        work["input_ready"] = 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        if all(r["passed"] for r in work["checks"]):
            work["checks"].append(
                dict(
                    path=str(root),
                    upstream=str(root),
                    hash=None,
                    artifact_field="input_schema",
                    op="==",
                    expected="valid",
                    observed=str(error),
                    passed=False,
                )
            )
    progress("preconditions_complete", len(work["checks"]))
    if work["input_ready"]:
        try:
            work.update(reconstruct(data, raw))
        except (OSError, ValueError, KeyError, TypeError) as error:
            work["owned_failure"] = str(error)
            progress("owned_reconstruction_failed")
    work.update(
        duration_s=(time.monotonic_ns() - started) / 1e9,
        clock=dict(
            started_monotonic_ns=started,
            started_wall_ns=wall,
            ended_monotonic_ns=time.monotonic_ns(),
        ),
        code_config_hashes=[
            k.frozen.reference(ROOT / p)
            for p in [
                *OWNED,
                TEST,
                k.frozen.PROTOCOL,
                bootstrapper.MODULE,
                upstream.MODULE,
                upstream.legacy.MODULE,
                "python/carnot/verify/delayed_utility_learning_8225.py",
                "python/carnot/reporting/current_work_receipt.py",
                "python/carnot/reporting/primary_publication.py",
                "python/carnot/reporting/methods_stream_execution_8111.py",
                "python/carnot/reporting/v709_execution.py",
            ]
        ],
        raw_shard_hashes=[
            k.frozen.reference(p)
            for p in sorted(raw.rglob("*"))
            if p.is_file()
            and p.name not in ["measurement.json", "measurement.stdout", "measurement.stderr"]
        ],
    )
    atomic_json(raw / "measurement.json", work)
    progress("measurement_complete", len(work["rows"]))
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Readiness reflects owned audit completion; only cost and safety can pass H2."""
    stats = statistics(work["rows"])
    checked = (
        bool(receipts)
        and all(r["passed"] and r.get("normal_exit", True) for r in receipts)
        and not work["owned_failure"]
    )
    ready = bool(checked and work["input_ready"])
    success = ready and stats["passed"] and stats["retention_passed"]
    verdict = (
        "disqualified"
        if not checked
        else "blocked"
        if not ready
        else "circular_positive"
        if success
        else "null"
    )
    failures = [r for r in work["checks"] if not r["passed"]]
    operand = (
        Path(failures[0]["path"]).stem
        if failures
        else "owned_validation"
        if not checked
        else "delayed_decision_benefit"
    )
    units = [
        r
        for r in work["rows"]
        if r["seed"] == 101 and r["arm"] == "frozen" and r["condition"] == "later_stream"
    ]
    value = dict(
        work,
        experiment_id=8241,
        task_id=TASK,
        milestone="2026.10.712",
        run_date=RUN_DATE,
        honest_verdict="complete_" + verdict + "_" + operand,
        verdict_class=verdict,
        gate_check_summary=work["checks"],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=MODEL_SPECS,
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        trained_head_specs=dict(
            kind="imported_global_scale_intercept_and_ordered_probability_patches",
            fitted_here=False,
            historical_qwen_provenance="cached only; zero current calls",
        ),
        intended_count=192,
        completed_count=stats["completed_count"],
        failed_count=0,
        censored_count=sum(r["status"] == "censored" for r in units),
        excluded_count=sum(r["status"] == "excluded" for r in units) if units else 192,
        independent_count=0 if fixture else stats["completed_count"],
        verifier_is_oracle=True,
        exposure_scope="retention_sources_disjoint_from_stream; inherited_historical_development_exposure",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=bool(checked),
        flagged_adversarial=False,
        acceptance_gates=dict(
            input_authentication=bool(work["input_ready"]), owned_validation=bool(checked), H2=H2
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=work["checks"],
        random_seed=H2["seed"],
        source_artifact_hashes=work["refs"],
        phase_spans=[
            dict(
                phase="authenticated_cold_reconstruction",
                duration_s=work["duration_s"],
                **work["clock"],
            )
        ],
        cited_upstream_artifacts=[
            dict(
                path=r.get("upstream_path", r["path"]),
                sha256=r["sha256"],
                fields_imported=[
                    "qualified readiness, committed heads, issue/release clocks, original masks and evaluator fragments"
                ],
            )
            for r in work["refs"]
        ],
        learning_audit_ready_score=int(ready),
        h2_development_signal_score=int(success),
        H2=stats,
        per_source_deltas=stats["per_source_deltas"],
        retention_rows=stats["retention_rows"],
        block_bootstrap_diagnostics=stats["block_bootstrap_diagnostics"],
        excess_brier_rows=stats["excess_brier_rows"],
        reductions=stats["reductions"],
        methodology_note="Cold issue reconstruction from committed heads and strictly earlier feedback; retention predictions and final heads authenticated before labels. Original slots retain missing masks. Seeds averaged within source. Signed calibration residual, Brier, false accepts and actual decision cost are distinct. Recovery and probability movement cannot satisfy H2. Qualified null is terminal; no independent generalization.",
    )
    from scripts.experiment_template import normalize_artifact_for_template_write

    value = normalize_artifact_for_template_write(value)
    value["field_principles"] = {
        key: "Bind actual invocation and primitive bytes; missing evidence is distinct from zero; audit readiness is separate from exposed development benefit."
        for key in value
    }
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Fresh primitive reconstruction detects drift even when aggregates are rehashed."""
    try:
        value = json.loads(path.read_bytes())
        checksum = value.pop("reproducibility_checksum")
        if canonical_hash(value) != checksum:
            return False
        raw = Path(value["terminal_validation_sidecar_path"]).parent
        for ref in (
            value["code_config_hashes"]
            + value["raw_shard_hashes"]
            + value["source_artifact_hashes"]
        ):
            for key in ["path", "upstream_path"]:
                if key in ref and sha256_file(Path(ref[key])) != ref["sha256"]:
                    return False
        for receipt in value["validation_receipts"]:
            for stream in ["stdout", "stderr"]:
                if (
                    stream + "_path" in receipt
                    and sha256_file(Path(receipt[stream + "_path"])) != receipt[stream + "_sha256"]
                ):
                    return False
        work = json.loads((raw / "measurement.json").read_bytes())
        if work["input_ready"] and not work["owned_failure"]:
            data = json.loads((raw / "audit_inputs.json").read_bytes())
            snapshot = json.loads(Path(work["input_snapshot"]).read_bytes())
            expected_data = snapshot if "public" in snapshot else data_from_value(snapshot)
            equal(expected_data, data)
            with TemporaryDirectory(prefix="carnot-8241-replay-") as directory:
                rebuilt = reconstruct(data, Path(directory))
            for key in ["rows", "install_exposure_rows"]:
                equal(work[key], rebuilt[key])
            if (
                not 0
                < work["retention_access"]["predictions_sealed_ns"]
                < work["retention_access"]["labels_opened_ns"]
            ):
                return False
        return build(work, raw, value["validation_receipts"], fixture=work["fixture_mode"]) == dict(
            value, reproducibility_checksum=checksum
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def manifest(private: Path, candidate: Path) -> Json:
    """Reuse qualified supervision while freezing only this audit's added statements."""
    with (
        patch.object(upstream, "OWNED", OWNED),
        patch.object(upstream, "CLI", CLI),
        patch.object(upstream, "TEST", TEST),
    ):
        specs = upstream.manifest(private, candidate)
    for spec in specs["commands"]:
        if spec["name"] == "owned_unit_and_private_CLI":
            spec["argv"].remove("tests/python/test_utility_kernel_8221.py")
    return specs


def main(argv: list[str] | None = None) -> int:
    """Share bounded measurement, private CLI and atomic terminal publication."""
    execution = upstream.legacy.execution
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
        patch.object(execution, "manifest", manifest),
        patch.object(execution, "run_check", run_check),
    ):
        return int(execution.main(argv))
