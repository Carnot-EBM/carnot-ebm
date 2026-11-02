"""REQ-REPORT-7982: bounded cached-input fitting with no decision-benefit claim.

Only fit and tune evaluator views are opened. Original policy sources receive
sealed predictions, so Exp7983 can design and evaluate a policy afterwards.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any

import numpy as np
import yaml

from carnot.reporting import evidence_features_custody_7980 as custody
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.verify import evidence_features_7980 as features
from carnot.verify import multivariate_energy_7982 as m
from carnot.verify import qwen_energy_calibration_7972 as scalar
from carnot.verify import qwen_response_risk_7958 as risk

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_7982_v692_multivariate_energy"
TASK = "exp7982-multivariate-energy"
MODEL_SPECS: list[str] = []
OWNED = [
    f"python/carnot/{NAME}.py",
    "python/carnot/verify/multivariate_energy_7982.py",
    "python/carnot/reporting/multivariate_validation_7982.py",
    f"scripts/experiments/{NAME}.py",
]
TESTS = ["tests/python/test_multivariate_energy_7982.py", f"tests/python/test_{NAME}.py"]
INPUTS = {
    7980: (
        "results/experiment_7980_v692_evidence_features.json",
        "feature_views_ready_score",
        "2026.10.692",
    ),
    7969: (
        "results/experiment_7969_v691_qwen_calibration_capture.json",
        "qwen_capture_ready_score",
        "2026.10.691",
    ),
    7972: (
        "results/experiment_7972_v691_qwen_energy_calibration.json",
        "qwen_calibration_ready_score",
        "2026.10.691",
    ),
}
PINS = {
    7980: "sha256:dc06fadccb5a0bfce0b545256a9e8133e61f002df642b2d71d829753322438ed",
    7969: custody.PINS[7969],
    7972: custody.PINS[7972],
}
reference, checked, operand = custody.reference, custody.checked, custody.operand


def progress(phase: str) -> None:
    """Flushed boundaries expose real work without inflating elapsed duration."""
    print(f"[exp7982] phase={phase}", flush=True)


def authenticate(root: Path) -> tuple[list[Json], Json]:
    """Bind exact qualified producers directly; retired training gates are unused."""
    checks, upstream, refs = [], {}, []
    for eid, (relative, ready, milestone) in INPUTS.items():
        path = root / relative
        checks.append(
            operand(eid, path, "sha256", PINS[eid], sha256_file(path) if path.is_file() else None)
        )
        value = json.loads(path.read_text()) if checks[-1]["passed"] else {}
        expected = dict(
            experiment_id=eid,
            run_date="20261001",
            execution_date="20261001",
            milestone=milestone,
            flagged_adversarial=False,
            **{ready: 1},
        )
        expected["task_id"] = {
            7980: "exp7980-evidence-features",
            7969: "exp7969-qwen-calibration-capture",
            7972: "exp7972-qwen-energy-calibration",
        }[eid]
        checks.extend(operand(eid, path, k, v, value.get(k)) for k, v in expected.items())
        checks.append(
            operand(
                eid,
                path,
                "eligible_verdict",
                True,
                value.get("verdict_class") in {"null", "positive"},
            )
        )
        upstream[eid] = value
        if value:
            refs.append(
                dict(
                    reference(path),
                    producer_id=eid,
                    producer_invocation_date=value["run_date"],
                    producer_invocation_timestamp=value.get(
                        "invocation_timestamp", value.get("started_at")
                    ),
                    imported_fields=[ready, "rows", "public_role_manifests", "heads_seal"],
                )
            )
    if all(r["passed"] for r in checks):
        f, q, s = (upstream[eid] for eid in (7980, 7969, 7972))
        items = [f["public_features"], f["q_manifest"], s["heads_seal"]]
        items += list(f["public_role_manifests"].values())
        items += [f["evaluator_role_manifests"][r] for r in ("fit", "tune")]
        items += q["raw_response_shards"]
        for item in items:
            p = Path(item["path"])
            checks.append(
                operand(
                    "sidecar", p, "sha256", item["sha256"], sha256_file(p) if p.is_file() else None
                )
            )
        retired = yaml.safe_load((root / "ops/exclusion_manifest.yaml").read_text())
        for eid in INPUTS:
            checks.append(
                operand(
                    eid,
                    root / "ops/exclusion_manifest.yaml",
                    "retired",
                    False,
                    any(r.get("experiment_id") == eid for r in retired.get("retired", [])),
                )
            )
        refs += items
    return [r for r in checks if not r["passed"]], dict(upstream=upstream, checks=checks, refs=refs)


def load_data(plan: Json) -> tuple[Json, list[Json]]:
    """Read original rosters first and join authenticated q without null substitution."""
    f, q = plan["upstream"][7980], plan["upstream"][7969]
    predictions = {r["family_id"]: r for r in q["rows"]}
    feature_rows = {
        r["family_id"]: r for r in json.loads(checked(f["public_features"]).read_text())["rows"]
    }
    data, events, normalized = {}, [], {}
    for role, item in f["public_role_manifests"].items():
        view = json.loads(checked(item).read_text())
        if view["role"] != role:
            raise ValueError("role_roster")
        for r in view["request_rows"]:
            if set(r) != {"family_id", "source_bytes", "answer_bytes"}:
                raise ValueError("public_fields")
            key = features.normalized(bytes.fromhex(r["source_bytes"]))
            if normalized.get(key, role) != role:
                raise ValueError("cross_role")
            normalized[key] = role
        if role not in ("fit", "tune", "policy_design"):
            continue
        labels = {}
        if role in ("fit", "tune"):
            label_ref = f["evaluator_role_manifests"][role]
            labels = {r["family_id"]: r for r in json.loads(checked(label_ref).read_text())["rows"]}
            events.append(
                dict(
                    role=role,
                    purpose="optimization" if role == "fit" else "temperature_selection_only",
                    path=label_ref["path"],
                    sha256=label_ref["sha256"],
                    heads_sealed=False,
                )
            )
        data[role] = []
        for r in view["request_rows"]:
            fid, feature = r["family_id"], feature_rows[r["family_id"]]
            row = predictions[fid]
            parsed = risk.transport.parse_response(row["raw_response"], row["visible_ids"])
            if parsed != row["parsed"] or row["role"] != role:
                raise ValueError("parse_drift")
            label = labels.get(fid, {})
            data[role].append(
                dict(
                    family_id=fid,
                    source_cluster_id=feature["source_normalized_hash"],
                    q=parsed["probability"],
                    features=feature["values"],
                    y=label.get("y"),
                    status=row["status"],
                )
            )
    m.validate_data(data)
    return data, events


def base(failures: list[Json]) -> Json:
    """Unavailable evidence stays empty, and readiness never implies advantage."""
    return dict(
        experiment_id=7982,
        task_id=TASK,
        milestone="2026.10.692",
        run_date="20261001",
        execution_date="20261001",
        invocation_timestamp=datetime.now(UTC).isoformat(),
        honest_verdict="complete_blocked_multivariate_inputs"
        if failures
        else "complete_null_multivariate_fit",
        verdict_class="blocked" if failures else "null",
        gate_check_summary=failures,
        inference_substrate="aggregation_from_upstream_artifacts"
        if failures
        else "verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        duration_s=0.0,
        phase_spans=[],
        random_seed=69201,
        reproducibility_checksum=None,
        cited_upstream_artifacts=[],
        code_config_hashes=[],
        raw_shard_hashes=[],
        preconditions_checked=[],
        validation_command_manifest_path=None,
        validation_receipts=[],
        coverage_statement_counts={},
        rows=[],
        sample_size_budget=dict(
            intended=352,
            eligible=0,
            started=0,
            completed=0,
            failed=0,
            censored=0,
            excluded=352,
            independent=0,
            unit="original_source",
            seeds_are_independent=False,
        ),
        verifier_is_oracle=False,
        claim_scope="Reproducible small conditional heads only. Decision benefit belongs to Exp7983. No foundation model or normalization advantage claim.",
        acceptance_gate_results=dict(validity=False, readiness=False, decision_benefit=False),
        flagged_adversarial=False,
        terminal_validation_sidecar_path=None,
        energy_fit_ready_score=0,
        trained_head_specs=[],
        checkpoints={},
        fit_support={},
        tune_support={},
        gradient_checks=[],
        feature_normalization={},
        optimizer_work={},
        heads_seal=None,
        prediction_seal=None,
        label_access_events=[],
        historical_required_failures=[],
        repository_health={},
        generator_weights_changed=False,
        production_defaults_changed=False,
        config=m.CONFIG,
        methodology="Fit-only normalization and Adam gradients; tune Brier temperature selection; exact two-label probabilities. Frozen scalar heads are descriptive. Seeds are averaged, not independent sources.",
        methodology_note="Synthetic separable and shuffled targets are circular wiring controls. No policy or evaluation targets are opened.",
    )


def measure(data: Json, raw: Path, *, fixture: bool) -> Json:
    """Export coefficients and predictions before any decision-target access."""
    value = base([])
    m.validate_data(data)
    fs, ts = (
        scalar.support(m.valid(data["fit"]), 128, 16),
        scalar.support(m.valid(data["tune"]), 32, 4),
    )
    atomic_json(raw / "fit_input.json", dict(data=data))
    if not fs["passed"] or not ts["passed"]:
        checks = []
        for role, s in [("fit", fs), ("tune", ts)]:
            bounds = [("independent", s["minimum"], s["independent"])] + [
                ("class_counts." + y, s["per_class"], s["class_counts"][y]) for y in ("0", "1")
            ]
            for field, expected, observed in bounds:
                if observed < expected:
                    check = operand(
                        "source_support",
                        raw / "fit_input.json",
                        role + "." + field,
                        expected,
                        observed,
                    )
                    check.update(op=">=", passed=False)
                    checks.append(check)
        value = base(checks)
        value.update(fit_support=fs, tune_support=ts)
        return value
    progress("fit_benchmark_before")
    measured = m.fit(data)
    progress("fit_benchmark_after")
    atomic_json(raw / "heads.json", measured)
    value["heads_seal"] = reference(raw / "heads.json")
    rows = m.predictions(measured, data)
    atomic_json(raw / "predictions.json", dict(rows=rows))
    value["prediction_seal"] = reference(raw / "predictions.json")
    for key in (
        "fit_support",
        "tune_support",
        "gradient_checks",
        "feature_normalization",
        "optimizer_work",
        "summary",
        "parity_max_error",
    ):
        value[key] = measured[key]
    value.update(
        checkpoints=dict(
            heads=value["heads_seal"],
            predictions=value["prediction_seal"],
            inputs=reference(raw / "fit_input.json"),
        ),
        rows=rows,
        energy_fit_ready_score=1,
        heads_frozen_before_decision_labels=True,
        no_fit_conversion=dict(
            formula="sigmoid((E0-E1)/temperature)",
            fitted_parameters=0,
            parity_max_error=measured["parity_max_error"],
            tolerance=1e-12,
        ),
        trained_head_specs=[
            dict(
                arm=a,
                seeds=list(m.SEEDS),
                parameter_count_per_seed=m.COUNTS[a],
                pretrained=False,
                input_dim=9,
                primary=a == "gibbs",
            )
            for a in m.ARMS
        ],
    )
    n = sum(len(r) for r in data.values())
    eligible = sum(
        r["q"] is not None and r["features"] is not None for rr in data.values() for r in rr
    )
    value["sample_size_budget"].update(
        intended=n,
        eligible=eligible,
        started=eligible,
        completed=eligible,
        excluded=n - eligible,
        independent=eligible,
    )
    value["acceptance_gate_results"].update(validity=True, readiness=True)
    if fixture:
        value.update(
            verdict_class="circular_positive",
            honest_verdict="complete_circular_positive_multivariate_wiring",
            verifier_is_oracle=True,
        )
    if (
        not all(r["passed"] for r in measured["gradient_checks"])
        or measured["parity_max_error"] > 1e-12
    ):
        apply_validation(
            value, [dict(name="numerical_integrity", passed=False, required=True, exit_code=1)]
        )
    return value


def replay(value: Json) -> Json:
    """Cold reconstruction checks sealed public predictions and actual work counts."""
    if value["verdict_class"] in {"blocked", "disqualified"} and value["energy_fit_ready_score"]:
        raise ValueError("unsafe_readiness")
    if not value["heads_seal"]:
        return dict(passed=True, blocked=True)
    measured = json.loads(checked(value["heads_seal"]).read_text())
    data = json.loads(checked(value["checkpoints"]["inputs"]).read_text())["data"]
    sealed = json.loads(checked(value["prediction_seal"]).read_text())["rows"]
    m.validate_data(data)
    rows = m.predictions(measured, data)
    if rows != sealed or rows != value["rows"]:
        raise ValueError("prediction_drift")
    for k in (
        "fit_support",
        "tune_support",
        "gradient_checks",
        "feature_normalization",
        "optimizer_work",
        "summary",
        "parity_max_error",
    ):
        if measured[k] != value[k]:
            raise ValueError(k + "_drift")
    for item in value["code_config_hashes"] + value["raw_shard_hashes"]:
        checked(item)
    return dict(passed=True, rows=len(rows), heads_sha256=value["heads_seal"]["sha256"])


def apply_validation(value: Json, receipts: list[Json]) -> None:
    """Owned failures remain terminal while unrelated health retains its own scope."""
    value["validation_receipts"].extend(receipts)
    value["repository_health"] = dict(
        current=[r for r in value["validation_receipts"] if not r.get("required", True)]
    )
    if any(not r["passed"] for r in receipts if r.get("required", True)):
        value.update(
            verdict_class="disqualified",
            honest_verdict="complete_disqualified_owned_validation",
            energy_fit_ready_score=0,
        )
        value["acceptance_gate_results"].update(validity=False, readiness=False)


def terminal_check(candidate: Path) -> Json:
    """External validators and cold replay inspect the same prospective final bytes."""
    replay(json.loads(candidate.read_text()))
    commands = [
        CommandSpec(
            name,
            (str(ROOT / ".venv/bin/python"), "-u", str(ROOT / script), flag, str(candidate)),
            "terminal",
            60,
        )
        for name, script, flag in [
            ("adversarial", "scripts/adversarial_verify.py", "--json"),
            ("strict_rows", "scripts/verdict_row_consistency_lint.py", "--strict"),
        ]
    ]
    receipts = run_commands(
        ROOT, commands, log_dir=candidate.parent / "terminal_logs", heartbeat_s=15
    )
    with TemporaryDirectory(prefix="carnot-7982-cold-") as directory:
        receipts += run_commands(
            Path(directory),
            [
                CommandSpec(
                    "primary_cold_replay",
                    (
                        "/usr/bin/env",
                        "-u",
                        "PYTHONPATH",
                        str(ROOT / ".venv/bin/python"),
                        "-u",
                        str(ROOT / OWNED[-1]),
                        "--cold-replay",
                        str(candidate.absolute()),
                    ),
                    "terminal",
                    60,
                )
            ],
            log_dir=candidate.parent / "terminal_logs" / "cold",
            heartbeat_s=15,
        )
    return dict(
        passed=all(r["passed"] for r in receipts),
        receipts=receipts,
        flagged_adversarial=not receipts[0]["passed"],
    )


def publish(output: Path, value: Json) -> None:
    """The atomic primary carries terminal checks bound to its final byte hash."""
    raw = output.parent / "raw" / output.stem
    value["terminal_validation_sidecar_path"] = str(raw / "terminal_validation.json")
    value["field_principles"] = {
        k: "Bind current identity, exact evidence and measured work; readiness does not establish benefit."
        for k in value
    }
    atomic_json(raw / "terminal_candidate.json", value)
    report = terminal_check(raw / "terminal_candidate.json")
    if not report["passed"]:
        apply_validation(
            value,
            [
                dict(
                    name="terminal_before_correction",
                    required=True,
                    passed=False,
                    exit_code=1,
                    report=report,
                )
            ],
        )
        value["flagged_adversarial"] = report.get("flagged_adversarial", False)
        value["historical_required_failures"].append(
            dict(scope="owned_terminal_before_correction", report=report)
        )
    published = publish_primary(output, value, terminal_check)
    atomic_json(raw / "terminal_validation.json", published)
    receipt = reader_receipt(
        TASK,
        output.parent,
        field="energy_fit_ready_score",
        expected=value["energy_fit_ready_score"],
    )
    atomic_json(raw / "primary_resolution.json", receipt)
    if not receipt["passed"]:
        raise ValueError("primary_resolution")


def run_validation(raw: Path, scratch: Path) -> tuple[Json, list[Json], Json]:
    """Freeze the owned command list before private science or production fitting."""
    from carnot.reporting import multivariate_validation_7982 as validation

    manifest = validation.freeze(raw, scratch)
    receipts = validation.execute(manifest, raw)
    return manifest, receipts, validation.coverage_counts(scratch)


def main(argv: list[str] | None = None) -> int:
    """One CPU entrypoint supports private fixtures, blocked gates and cold replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261001", choices=["20261001"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / f"{NAME}.json")
    parser.add_argument("--fixture-input", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--validation-worker", action="store_true")
    args = parser.parse_args(argv)
    started = time.monotonic()
    progress("begin")
    try:
        if args.cold_replay:
            replay(json.loads(args.cold_replay.read_text()))
            progress("replay_passed")
            return 0
        output = args.output.absolute()
        raw = output.parent / "raw" / output.stem
        manifest, receipts, counts = {}, [], {}
        with TemporaryDirectory(prefix="carnot-7982-") as directory:
            if not args.validation_worker:
                progress("freeze_and_validate")
                manifest, receipts, counts = run_validation(raw, Path(directory))
            progress("authenticate")
            if args.fixture_input:
                data = json.loads(args.fixture_input.read_text())
                failures, plan = [], dict(checks=[], refs=[], upstream={})
                accesses = [
                    dict(
                        role=r,
                        purpose="optimization" if r == "fit" else "temperature_selection_only",
                        path=str(args.fixture_input),
                        sha256=sha256_file(args.fixture_input),
                    )
                    for r in ("fit", "tune")
                ]
            else:
                failures, plan = authenticate(args.root)
                data, accesses = ({}, []) if failures else load_data(plan)
            value = (
                base(failures) if failures else measure(data, raw, fixture=bool(args.fixture_input))
            )
            value["label_access_events"] = accesses
            value["preconditions_checked"] = plan["checks"]
            value["cited_upstream_artifacts"] = plan["refs"]
            value["historical_required_failures"] = [
                r
                for v in plan["upstream"].values()
                for r in v.get("historical_required_failures", [])
            ]
            if plan["upstream"] and not failures and value["heads_seal"]:
                controls = json.loads(checked(plan["upstream"][7972]["heads_seal"]).read_text())[
                    "heads"
                ]
                # Scalar controls keep their old coefficients and remain descriptive.
                control_rows = []
                for role, rr in data.items():
                    for r in rr:
                        for arm, heads in controls.items():
                            p = (
                                float(
                                    np.mean(
                                        [scalar.predict(h, np.array([r["q"]]))[0] for h in heads]
                                    )
                                )
                                if r["q"] is not None
                                else None
                            )
                            control_rows.append(
                                dict(family_id=r["family_id"], role=role, arm="exp7972_" + arm, p=p)
                            )
                atomic_json(
                    raw / "scalar_controls.json",
                    dict(rows=control_rows, heads=plan["upstream"][7972]["heads_seal"]),
                )
                value["frozen_scalar_controls"] = reference(raw / "scalar_controls.json")
                value["upstream_q_sidecar_audit"] = dict(
                    path=plan["upstream"][7980]["q_manifest"]["path"],
                    imported=False,
                    reason="Exp7980 q sidecar contains nulls; direct authenticated Exp7969 probability join.",
                )
            value["code_config_hashes"] = [reference(ROOT / p) for p in OWNED]
            value["raw_shard_hashes"] = list(value["checkpoints"].values())
            value["reproducibility_checksum"] = canonical_hash(
                dict(
                    config=m.CONFIG,
                    code=value["code_config_hashes"],
                    inputs=value["raw_shard_hashes"],
                    upstream=plan["refs"],
                )
            )
            if manifest:
                value["validation_command_manifest_path"] = str(raw / "validation_manifest.json")
                apply_validation(value, receipts)
            value["coverage_statement_counts"] = counts
            value["duration_s"] = time.monotonic() - started
            value["phase_spans"] = [
                dict(phase="current_cached_fit_and_validation", duration_s=value["duration_s"])
            ]
            progress("publish")
            publish(output, value)
            progress("complete")
            return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"[exp7982] failed={error}", flush=True)
        return 1
