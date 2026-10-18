"""REQ-REPORT-7996: qualify historical sparse fitting without new target access.

Private inputs, checkpoints and command receipts bind this CPU invocation.
Training readiness does not establish natural stream benefit or retention.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np

from carnot import experiment_7982_v692_multivariate_energy as prior
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.verify import sparse_energy_7996 as m
from carnot.verify import qwen_energy_calibration_7972 as scalar

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_7996_v693_sparse_energy_training"
TASK = "exp7996-sparse-energy-training"
MODEL_SPECS: list[str] = []
OWNED = [
    f"python/carnot/{NAME}.py",
    "python/carnot/verify/sparse_energy_7996.py",
    "python/carnot/reporting/sparse_validation_7996.py",
    f"scripts/experiments/{NAME}.py",
]
TESTS = ["tests/python/test_sparse_energy_7996.py", f"tests/python/test_{NAME}.py"]
reference, checked = prior.reference, prior.checked


def progress(phase: str) -> None:
    """Immediate boundaries let the supervisor distinguish work from a stall."""
    print(f"[exp7996] phase={phase}", flush=True)


def authenticate(root: Path) -> tuple[list[Json], Json]:
    """Reuse exact historical producer pins without comparing their code to today."""
    return prior.authenticate(root)


def load_data(plan: Json) -> tuple[Json, list[Json]]:
    """Join only fit/tune rosters; no evaluation or reserved target file is opened."""
    feature = plan["upstream"][7980]
    responses = {r["family_id"]: r for r in plan["upstream"][7969]["rows"]}
    values = {
        r["family_id"]: r
        for r in json.loads(checked(feature["public_features"]).read_text())["rows"]
    }
    data, events = {}, []
    for role in ("fit", "tune"):
        public = json.loads(checked(feature["public_role_manifests"][role]).read_text())
        ref = feature["evaluator_role_manifests"][role]
        labels = {r["family_id"]: r for r in json.loads(checked(ref).read_text())["rows"]}
        events.append(
            dict(role=role, purpose="fit" if role == "fit" else "temperature_only", **ref)
        )
        data[role] = []
        for row in public["request_rows"]:
            fid = row["family_id"]
            response = responses[fid]
            f = values[fid]
            parsed = prior.risk.transport.parse_response(
                response["raw_response"], response["visible_ids"]
            )
            if parsed != response["parsed"] or response["role"] != role:
                raise ValueError("response_custody")
            data[role].append(
                dict(
                    family_id=fid,
                    source_cluster_id=f["source_normalized_hash"],
                    q=parsed["probability"],
                    features=f["values"],
                    y=labels[fid]["y"],
                    status=response["status"],
                )
            )
    m.validate_data(data)
    return data, events


def base(failures: list[Json]) -> Json:
    """A missing prerequisite produces a terminal blocked artifact with no fit."""
    return dict(
        experiment_id=7996,
        task_id=TASK,
        milestone="2026.10.693",
        run_date="20261001",
        execution_date="20261001",
        invocation_timestamp=datetime.now(UTC).isoformat(),
        schema="carnot.sparse_energy_training.v1",
        honest_verdict="complete_blocked_sparse_training"
        if failures
        else "complete_null_sparse_training",
        verdict_class="blocked" if failures else "null",
        gate_check_summary=failures,
        inference_substrate="aggregation_from_upstream_artifacts"
        if failures
        else "verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        trained_head_specs=[],
        duration_s=0.0,
        phase_spans=[],
        random_seed=17,
        reproducibility_checksum=None,
        config=m.CONFIG,
        cited_upstream_artifacts=[],
        code_config_hashes=[],
        raw_shard_hashes=[],
        rows=[],
        sample_size_budget=dict(
            intended=320,
            eligible=0,
            started=0,
            completed=0,
            excluded=320,
            failed=0,
            censored=0,
            independent=0,
            seeds_are_independent=False,
        ),
        verifier_is_oracle=False,
        claim_scope="Historical training mechanism only; natural adaptation and retention unmeasured. CPU path supplies no FPGA speed evidence.",
        acceptance_gate_results=dict(readiness=False, natural_benefit=False),
        positive_control_results={},
        preconditions_checked=[],
        validation_command_manifest_path=None,
        validation_receipts=[],
        coverage_statement_counts={},
        flagged_adversarial=False,
        terminal_validation_sidecar_path=None,
        sparse_fit_ready_score=0,
        checkpoints={},
        optimizer_work={},
        gradient_rows=[],
        dense_sparse_parity={},
        feature_scaler={},
        spline_basis_spec=dict(
            degree=3,
            knots=m.KNOTS,
            boundary_multiplicity=4,
            internal_knots=8,
            coefficients_per_input=12,
            parameter_count=109,
        ),
        active_coefficient_counts={},
        state_bytes={},
        equivalent_classifier_parity={},
        objective_direction="minimize binary log loss plus L2 .001; E0=0, E1=-z",
        optimization_note="All seeds saved; primary 17. Sigmoid identity is not a competitor or a normalization win. Lazy L2 changes all logical coefficients via one multiplier.",
        label_access_events=[],
        historical_required_failures=[],
        repository_health={},
    )


def prediction_rows(measured: Json, data: Json) -> list[Json]:
    """Each arm/seed/source row retains unknowns and never multiplies source support."""
    rows = []
    for role, sources in data.items():
        for source in sources:
            eligible = source in m.usable([source])
            for arm, heads in sorted(measured["heads"].items()):
                for head in heads:
                    p = float(m.predict(head, m.inputs([source]))[0]) if eligible else None
                    rows.append(
                        dict(
                            family_id=source["family_id"],
                            source_cluster_id=source["source_cluster_id"],
                            role=role,
                            arm=arm,
                            seed=head["seed"],
                            probability=p,
                            numerator=p,
                            denominator=int(eligible),
                            eligibility=eligible,
                            failure=not eligible,
                            censored=False,
                            status="completed" if eligible else "excluded",
                        )
                    )
    return rows


def measure(data: Json, raw: Path, plan: Json) -> Json:
    """Seal a runnable head before any future cohort target can be accessed."""
    progress("numerical_training_begin")
    measured = m.train(data, raw / "seeds")
    measured["heads"]["sigmoid_equivalent"] = [
        dict(h, arm="sigmoid_equivalent") for h in measured["heads"]["spline"]
    ]
    f = m.inputs(m.usable(data["fit"]))
    checks = [dict(m.audit(h, f[0]), seed=h["seed"]) for h in measured["heads"]["spline"]]
    primary = measured["heads"]["spline"][0]
    low, high = (np.asarray(primary["scaler"][k]) for k in ("minimum", "maximum"))
    probes = [low + t * (high - low) for t in sorted(set(m.KNOTS))]
    probes += [np.array([q] + [v] * 8) for q, v in [(0.0, -1e6), (1.0, 1e6)]]
    for i, x in enumerate(probes):
        checks.append(dict(m.audit(primary, x), seed=17, probe=i))
        progress(f"numerical_probe_completed_{i + 1}/{len(probes)}")
    fd = {
        arm: max(m.finite_difference(h, f[0]) for h in heads)
        for arm, heads in measured["heads"].items()
    }
    parity = max(
        float(np.max(np.abs(m.predict(h, f) - m.sigmoid_predict(h, f))))
        for h in measured["heads"]["spline"]
    )
    measured.update(
        gradient_checks=checks,
        finite_difference_controls=fd,
        equivalent_classifier_parity=dict(
            max_absolute_error=parity, tolerance=1e-10, extra_fit_steps=0, passed=parity <= 1e-10
        ),
    )
    measured["state_bytes"] = {
        arm: dict(
            coefficient_bytes=heads[0]["parameter_count"] * 8,
            scaler_bytes=144,
            knot_bytes=128 if arm in ("spline", "sigmoid_equivalent") else 0,
            temperature_bytes=8,
            decay_multiplier_bytes=8,
            total=heads[0]["parameter_count"] * 8
            + 160
            + (128 if arm in ("spline", "sigmoid_equivalent") else 0),
            dtype="float64",
            json_serialized_bytes=len(json.dumps(heads[0], sort_keys=True).encode()),
        )
        for arm, heads in measured["heads"].items()
    }
    rows = prediction_rows(measured, data)
    atomic_json(raw / "fit_input.json", dict(data=data))
    atomic_json(raw / "heads.json", measured)
    atomic_json(raw / "rows.json", dict(rows=rows))
    value = base([])
    for field in (
        "feature_scaler",
        "optimizer_work",
        "positive_control_results",
        "state_bytes",
        "equivalent_classifier_parity",
    ):
        value[field] = measured[field]
    value.update(
        rows=rows,
        checkpoints={
            k: reference(raw / p)
            for k, p in [
                ("inputs", "fit_input.json"),
                ("heads", "heads.json"),
                ("rows", "rows.json"),
            ]
        },
        gradient_rows=[r for c in checks for r in c["rows"]],
        dense_sparse_parity=dict(
            passed=all(c["passed"] for c in checks) and all(v < 1e-7 for v in fd.values()),
            checks=checks,
            control_finite_differences=fd,
        ),
        trained_head_specs=[
            dict(
                arm=a,
                parameter_count_per_seed=m.COUNTS[a],
                seeds=list(m.SEEDS),
                primary_seed=17,
                pretrained=False,
            )
            for a in m.COUNTS
        ],
        active_coefficient_counts=dict(
            maximum_data_coefficients=37,
            maximum_per_feature=4,
            observed_maximum=max(r["coefficient_touches"] for c in checks for r in c["rows"]),
            global_decay_writes=1,
            logical_decay_coefficients=109,
        ),
    )
    eligible = sum(len(m.usable(r)) for r in data.values())
    intended = sum(len(rows) for rows in data.values())
    value["sample_size_budget"].update(
        intended=intended,
        eligible=eligible,
        started=eligible,
        completed=eligible,
        excluded=intended - eligible,
        independent=eligible,
    )
    if plan["upstream"]:
        ref = plan["upstream"][7972]["heads_seal"]
        scalar_heads = json.loads(checked(ref).read_text())["heads"]
        comparisons = [
            dict(
                family_id=r["family_id"],
                role=role,
                probability=float(
                    np.mean(
                        [scalar.predict(h, np.array([r["q"]]))[0] for h in scalar_heads["gibbs"]]
                    )
                ),
            )
            for role, sources in data.items()
            for r in m.usable(sources)
        ]
        atomic_json(raw / "frozen_scalar.json", dict(rows=comparisons, checkpoint=ref))
        value["frozen_scalar_comparator"] = dict(
            reference(raw / "frozen_scalar.json"), source_checkpoint=ref, fitted_current_steps=0
        )
    value["numerical_passed"] = (
        value["dense_sparse_parity"]["passed"]
        and measured["positive_control_results"]["passed"]
        and parity <= 1e-10
    )
    progress("numerical_training_end_checkpoints_sealed")
    return value


def replay(value: Json) -> Json:
    """Recompute every prediction from byte-checked inputs and serialized heads."""
    if value["verdict_class"] in ("blocked", "disqualified") and value["sparse_fit_ready_score"]:
        raise ValueError("unsafe_readiness")
    if not value["checkpoints"]:
        return dict(passed=True, blocked=True)
    data = json.loads(checked(value["checkpoints"]["inputs"]).read_text())["data"]
    heads = json.loads(checked(value["checkpoints"]["heads"]).read_text())
    sealed = json.loads(checked(value["checkpoints"]["rows"]).read_text())["rows"]
    m.validate_data(data)
    if prediction_rows(heads, data) != sealed or sealed != value["rows"]:
        raise ValueError("prediction_drift")
    for field in (
        "feature_scaler",
        "optimizer_work",
        "state_bytes",
        "equivalent_classifier_parity",
        "positive_control_results",
    ):
        if heads[field] != value[field]:
            raise ValueError(field + "_drift")
    for ref in value["code_config_hashes"] + value["raw_shard_hashes"]:
        checked(ref)
    return dict(passed=True, rows=len(sealed))


def apply_validation(value: Json, receipts: list[Json], counts: Json) -> None:
    """Owned failures clear readiness while full-suite health remains diagnostic."""
    value["validation_receipts"] = receipts
    value["coverage_statement_counts"] = counts
    value["repository_health"] = dict(current=[r for r in receipts if not r["required"]])
    good_coverage = set(counts) == set(OWNED) and all(
        c["num_statements"] > 0 and c["missing_lines"] == 0 for c in counts.values()
    )
    failed = any(not r["passed"] for r in receipts if r["required"])
    if (
        failed
        or (receipts and value["checkpoints"] and not good_coverage)
        or (value["checkpoints"] and not value["numerical_passed"])
    ):
        value.update(
            verdict_class="disqualified",
            honest_verdict="complete_disqualified_sparse_training",
            sparse_fit_ready_score=0,
        )
    elif receipts and good_coverage and value["checkpoints"]:
        value["sparse_fit_ready_score"] = 1
    value["acceptance_gate_results"]["readiness"] = bool(value["sparse_fit_ready_score"])


def terminal_check(candidate: Path) -> Json:
    """Independent validators inspect the exact candidate bytes with short heartbeats."""
    replay(json.loads(candidate.read_text()))
    commands = [
        prior.CommandSpec(
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
    receipts = prior.run_commands(
        ROOT, commands, log_dir=candidate.parent / "terminal_logs", heartbeat_s=15
    )
    return dict(
        passed=all(r["passed"] for r in receipts),
        receipts=receipts,
        flagged_adversarial=not receipts[0]["passed"],
    )


def publish(output: Path, value: Json, scratch: Path) -> None:
    """Publish checked primary bytes and bind both conductor readers to that file."""
    raw = output.parent / "raw" / output.stem
    value["terminal_validation_sidecar_path"] = str(raw / "terminal_validation.json")
    value["field_principles"] = {
        k: "Bind actual bytes and owned work; readiness is not benefit." for k in value
    }
    candidate = scratch / "candidate.json"
    atomic_json(candidate, value)
    first = terminal_check(candidate)
    if not first["passed"]:
        value["flagged_adversarial"] = first["flagged_adversarial"]
        value["historical_required_failures"].append(first)
        apply_validation(
            value,
            value["validation_receipts"]
            + [
                dict(
                    name="terminal_failure", required=True, passed=False, exit_code=1, report=first
                )
            ],
            value["coverage_statement_counts"],
        )
    receipt = publish_primary(output, value, terminal_check)
    atomic_json(raw / "terminal_validation.json", receipt)
    reader = reader_receipt(
        TASK,
        output.parent,
        field="sparse_fit_ready_score",
        expected=value["sparse_fit_ready_score"],
    )
    atomic_json(raw / "primary_resolution.json", reader)
    if (
        not reader["passed"]
        or reader["gate_path"] != str(output)
        or reader["gate_sha256"] != sha256_file(output)
    ):
        raise ValueError("primary_resolution")


def main(argv: list[str] | None = None) -> int:
    """Direct CLI runs guarded private checks and exposes a no-fit cold replay route."""
    from carnot.reporting import sparse_validation_7996 as validation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261001", choices=["20261001"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--fixture-input", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--validation-worker", action="store_true")
    args = parser.parse_args(argv)
    began = time.monotonic()
    progress("begin")
    try:
        if args.cold_replay:
            replay(json.loads(args.cold_replay.read_text()))
            progress("replay_passed")
            return 0
        scratch = Path(tempfile.mkdtemp(prefix="carnot-7996-"))
        raw = scratch / "evidence"
        output = args.output.absolute()
        manifest = validation.freeze(raw, scratch) if not args.validation_worker else {}
        progress("configuration_and_commands_frozen")
        if args.fixture_input:
            data = json.loads(args.fixture_input.read_text())
            failures, plan = [], dict(checks=[], refs=[], upstream={})
            events = [
                dict(role=r, path=str(args.fixture_input), sha256=sha256_file(args.fixture_input))
                for r in ("fit", "tune")
            ]
        else:
            failures, plan = authenticate(args.root)
            data, events = ({}, []) if failures else load_data(plan)
        value = base(failures) if failures else measure(data, raw, plan)
        if args.fixture_input:
            value.update(
                verdict_class="circular_positive",
                honest_verdict="complete_circular_positive_sparse_wiring",
                verifier_is_oracle=True,
            )
        value.update(
            label_access_events=events,
            preconditions_checked=plan["checks"],
            cited_upstream_artifacts=plan["refs"],
            historical_required_failures=[
                r
                for p in plan["upstream"].values()
                for r in p.get("historical_required_failures", [])
            ],
        )
        if manifest:
            progress("validation_begin")
            receipts = validation.execute(manifest, raw)
            counts = validation.coverage_counts(scratch)
            value["validation_command_manifest_path"] = str(raw / "validation_manifest.json")
            apply_validation(value, receipts, counts)
            progress("validation_end")
        value["code_config_hashes"] = [reference(ROOT / p) for p in OWNED + TESTS]
        value["raw_shard_hashes"] = list(value["checkpoints"].values())
        value["reproducibility_checksum"] = canonical_hash(
            dict(
                config=m.CONFIG,
                code=value["code_config_hashes"],
                raw=value["raw_shard_hashes"],
                upstream=plan["refs"],
            )
        )
        value["duration_s"] = time.monotonic() - began
        value["phase_spans"] = [
            dict(phase="historical_fit_and_validation", duration_s=value["duration_s"])
        ]
        value["scratch_root_receipt"] = dict(
            path=str(scratch), outside_checkout=True, retained=True
        )
        progress("publish_begin")
        publish(output, value, scratch)
        progress("publish_end")
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"[exp7996] failed={error}", flush=True)
        return 1
