"""REQ-REPORT-7984: bounded cached ablation with original-source truth only.

Qualified historical capture supplies scalar confidence. This invocation fits
small heads on CPU and never loads a pretrained model or generates text.
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

from carnot import experiment_7982_v692_multivariate_energy as upstream
from carnot.reporting import evidence_features_custody_7980 as custody
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.verify import evidence_ablation_7984 as a
from carnot.verify import evidence_features_7980 as features
from carnot.verify import multivariate_energy_7982 as m
from carnot.verify import qwen_energy_calibration_7972 as scalar
from carnot.verify import qwen_response_risk_7958 as risk

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_7984_v692_evidence_ablation"
TASK = "exp7984-evidence-ablation"
MODEL_SPECS: list[str] = []
OWNED = [
    f"python/carnot/{NAME}.py",
    "python/carnot/verify/evidence_ablation_7984.py",
    "python/carnot/reporting/evidence_ablation_validation_7984.py",
    f"scripts/experiments/{NAME}.py",
]
TESTS = ["tests/python/test_evidence_ablation_7984.py", f"tests/python/test_{NAME}.py"]
DEPENDENCIES = [
    f"python/carnot/{upstream.NAME}.py",
    "python/carnot/verify/multivariate_energy_7982.py",
    "python/carnot/verify/qwen_energy_calibration_7972.py",
    "python/carnot/verify/qwen_response_risk_7958.py",
    "python/carnot/verify/evidence_features_7980.py",
    "python/carnot/verify/source_alignment.py",
    "python/carnot/reporting/evidence_features_custody_7980.py",
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/primary_publication.py",
    "python/carnot/reporting/experiment_7303_validation_scope.py",
    "ops/exclusion_manifest.yaml",
    "research-roadmap.yaml",
]
ROLES = ("fit", "tune", "policy_design", "evaluation")
PINS = {
    7982: "sha256:9daef912592c6c17e9b2db46c4059e7be99adcb3d76be897169e1e0afc238d54",
    7958: custody.PINS[7958],
    7968: custody.PINS[7968],
}
reference, checked, operand = custody.reference, custody.checked, custody.operand


def progress(phase: str) -> None:
    """Flushed boundaries expose measured progress without padding runtime."""
    print(f"[exp7984] phase={phase}", flush=True)


def authenticate(root: Path) -> tuple[list[Json], Json]:
    """Authenticate exact qualified training and historical pairs without fresh gates."""
    _, plan = upstream.authenticate(root)
    for eid, path_name, ready, task, milestone in [
        (7982, upstream.NAME, "energy_fit_ready_score", upstream.TASK, "2026.10.692"),
        (
            7958,
            "experiment_7958_v690_qwen_response_risk",
            "qwen_response_measurement_ready_score",
            "exp7958-qwen-response-risk",
            "2026.09.690",
        ),
        (
            7968,
            "experiment_7968_v691_response_role_targets",
            "response_roles_ready_score",
            "exp7968-response-role-targets",
            "2026.10.691",
        ),
    ]:
        path = root / "results" / (path_name + ".json")
        plan["checks"].append(
            operand(eid, path, "sha256", PINS[eid], sha256_file(path) if path.is_file() else None)
        )
        value = json.loads(path.read_text()) if plan["checks"][-1]["passed"] else {}
        expected = dict(
            experiment_id=eid,
            task_id=task,
            run_date="20261001",
            milestone=milestone,
            flagged_adversarial=False,
            **{ready: 1},
        )
        for k, v in expected.items():
            plan["checks"].append(operand(eid, path, k, v, value.get(k)))
        plan["checks"].append(
            operand(
                eid,
                path,
                "eligible_verdict",
                True,
                value.get("verdict_class") in {"positive", "null"},
            )
        )
        plan["upstream"][eid] = value
        if value:
            plan["refs"].append(
                dict(
                    reference(path),
                    producer_id=eid,
                    producer_invocation_date=value["run_date"],
                    producer_invocation_timestamp=value.get(
                        "invocation_timestamp", value.get("started_at")
                    ),
                    imported_fields=[ready, "rows", "checkpoints", "evaluator_role_manifests"],
                )
            )
            items = list(value.get("checkpoints", {}).values()) + value.get(
                "raw_response_shards", []
            )
            if eid == 7968:
                items += [value["evaluator_role_manifests"][r] for r in ROLES]
            for item in items:
                p = Path(item["path"])
                plan["checks"].append(
                    operand(
                        eid, p, "sha256", item["sha256"], sha256_file(p) if p.is_file() else None
                    )
                )
                plan["refs"].append(item)
    return [r for r in plan["checks"] if not r["passed"]], plan


def load_public(plan: Json) -> Json:
    """Open only four original public roles; reserved-panel targets stay unavailable."""
    public = {}
    seen: dict[str, str] = {}
    for role in ROLES:
        view = json.loads(
            checked(plan["upstream"][7980]["public_role_manifests"][role]).read_text()
        )
        if view["role"] != role:
            raise ValueError("role_roster")
        public[role] = view["request_rows"]
        for row in public[role]:
            if set(row) != features.PUBLIC_KEYS:
                raise ValueError("public_fields")
            key = features.normalized(bytes.fromhex(row["source_bytes"]))
            if seen.get(key, role) != role:
                raise ValueError("cross_role")
            seen[key] = role
    return public


def labels(plan: Json, role: str, events: list[Json], sealed: bool) -> Json:
    """Record evaluator access after public mapping, heads and policies are frozen."""
    item = plan["upstream"][7968]["evaluator_role_manifests"][role]
    value = json.loads(checked(item).read_text())
    events.append(
        dict(
            role=role,
            path=item["path"],
            sha256=item["sha256"],
            heads_sealed=sealed,
            purpose={
                "fit": "optimization",
                "tune": "temperature_selection_only",
                "policy_design": "policy_only",
                "evaluation": "original_old64_truth_only",
            }[role],
        )
    )
    return {r["family_id"]: r["y"] for r in value["rows"]}


def base(failures: list[Json]) -> Json:
    """Blocked work claims neither measurement nor added information."""
    return dict(
        experiment_id=7984,
        task_id=TASK,
        milestone="2026.10.692",
        run_date="20261001",
        execution_date="20261001",
        invocation_timestamp=datetime.now(UTC).isoformat(),
        honest_verdict="complete_blocked_evidence_ablation_inputs"
        if failures
        else "complete_null_evidence_ablation",
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
        random_seed=69284,
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
            intended=64,
            eligible=0,
            started=0,
            completed=0,
            failed=0,
            censored=0,
            excluded=64,
            independent=0,
            unit="original_evaluation_source",
            seeds_are_independent=False,
        ),
        verifier_is_oracle=False,
        claim_scope="Feature information and source reliance on exposed old64 development sources only. No energy architecture advantage, heldout generalization or intervention truth claim.",
        acceptance_gate_results=dict(validity=False, readiness=False, added_information=False),
        flagged_adversarial=False,
        terminal_validation_sidecar_path=None,
        ablation_ready_score=0,
        added_information_score=0,
        intervention_manifest={},
        paired_comparisons={},
        duplicate_parity={},
        checkpoints={},
        label_access_events=[],
        historical_required_failures=[],
        repository_health={},
        trained_head_specs=[],
        generator_weights_changed=False,
        production_defaults_changed=False,
        config=a.CONFIG,
        methodology="Matched 97-parameter nine-input Gibbs heads, Exp7982 fit/tune budgets. Original policy_design thresholds; original old64 paired Brier and cost with 10000 source bootstrap/sign-flip draws and Holm across four tests.",
    )


def measure(public: Json, plan: Json, raw: Path, *, fixture_data: Json | None) -> Json:
    """Seal public interventions before labels, heads before policy, and policy before truth."""
    value, events = base([]), []
    progress("freeze_interventions")
    views = a.interventions(public)
    atomic_json(raw / "public.json", public)
    atomic_json(raw / "interventions.json", views)
    value["intervention_manifest"] = dict(
        reference(raw / "interventions.json"),
        frozen_before_labels=True,
        seed=69284,
        length_bins=a.CONFIG["length_bins"],
        same_source_excluded=True,
    )
    qrows = {r["family_id"]: r for r in plan.get("upstream", {}).get(7969, {}).get("rows", [])}
    qrows.update(
        {
            r["family_id"]: dict(r, role="evaluation")
            for r in plan.get("upstream", {}).get(7958, {}).get("rows", [])
            if r["arm"] == "full_source"
        }
    )
    data = {}
    feature_rows = (
        {
            r["family_id"]: r["values"]
            for r in json.loads(checked(plan["upstream"][7980]["public_features"]).read_text())[
                "rows"
            ]
        }
        if fixture_data is None
        else {}
    )
    for role in ROLES:
        targets = (
            labels(plan, role, events, False)
            if fixture_data is None and role in ("fit", "tune")
            else {}
        )
        data[role] = []
        for row in public[role]:
            fid = row["family_id"]
            if fixture_data is not None:
                r = next(r for r in fixture_data[role] if r["family_id"] == fid)
                q, y, status = r["q"], r["y"] if role in ("fit", "tune") else None, r["status"]
            else:
                r = qrows[fid]
                parsed = risk.transport.parse_response(r["raw_response"], r["visible_ids"])
                if parsed != r["parsed"] or r["role"] != role:
                    raise ValueError("parse_drift")
                q, y, status = parsed["probability"], targets.get(fid), r["status"]
            data[role].append(
                dict(
                    family_id=fid,
                    source_cluster_id=features.normalized(bytes.fromhex(row["source_bytes"])),
                    q=q,
                    y=y,
                    status=status,
                    features=r["features"] if fixture_data is not None else feature_rows[fid],
                )
            )
    failures = []
    matched = a.matched_data(data, views)
    for arm in a.ARMS:
        for role, minimum, per_class in [("fit", 128, 16), ("tune", 32, 4)]:
            support = scalar.support(m.valid(matched[arm][role]), minimum, per_class)
            for field, expected, actual in [("independent", minimum, support["independent"])] + [
                ("class_counts." + y, per_class, support["class_counts"][y]) for y in ("0", "1")
            ]:
                if actual < expected:
                    check = operand(
                        "source_support",
                        raw / "interventions.json",
                        arm + "." + role + "." + field,
                        expected,
                        actual,
                    )
                    check.update(op=">=", passed=False)
                    failures.append(check)
    if failures:
        blocked = base(failures)
        blocked.update(
            intervention_manifest=value["intervention_manifest"], label_access_events=events
        )
        return blocked
    progress("fitting_benchmark_before")
    fitted = a.fit(data, views)
    progress("fitting_benchmark_after")
    atomic_json(raw / "heads.json", fitted)
    for role in ("policy_design", "evaluation"):
        targets = (
            labels(plan, role, events, True)
            if fixture_data is None
            else {r["family_id"]: r["y"] for r in fixture_data[role]}
        )
        for row in data[role]:
            row["y"] = targets.get(row["family_id"])
        if role == "policy_design":
            policies = a.design(fitted, data[role], views[role])
            atomic_json(raw / "policies.json", policies)
    progress("evaluation_benchmark_before")
    result = a.evaluate(fitted, policies, data["evaluation"], views["evaluation"])
    progress("evaluation_benchmark_after")
    atomic_json(raw / "data.json", data)
    value.update(
        result,
        label_access_events=events,
        optimizer_work=fitted["optimizer_work"],
        checkpoints={
            k: reference(raw / (k + ".json"))
            for k in ("public", "interventions", "heads", "policies", "data")
        },
        trained_head_specs=[
            dict(
                arm=arm,
                parameter_count_per_seed=97,
                input_dim=9,
                pretrained=False,
                seeds=list(m.SEEDS),
                config=a.CONFIG,
            )
            for arm in a.ARMS
        ],
    )
    n, eligible = len(data["evaluation"]), len(m.valid(data["evaluation"]))
    value["sample_size_budget"].update(
        intended=n,
        eligible=eligible,
        started=n,
        completed=eligible,
        failed=sum(r["q"] is None and r["y"] is not None for r in data["evaluation"]),
        censored=sum(r["status"] == "censored" for r in data["evaluation"]),
        excluded=sum(r["y"] is None or r["features"] is None for r in data["evaluation"]),
        independent=result["evaluation_support"]["independent"],
    )
    value["ablation_ready_score"] = 1
    value["acceptance_gate_results"].update(
        validity=True, readiness=True, added_information=bool(value["added_information_score"])
    )
    if value["added_information_score"]:
        value.update(
            verdict_class="positive", honest_verdict="complete_positive_added_feature_information"
        )
    if not result["evaluation_support"]["passed"]:
        value["honest_verdict"] = "complete_null_insufficient_evaluation_support"
    if fixture_data is not None:
        value.update(
            verdict_class="circular_positive",
            honest_verdict="complete_circular_positive_ablation_wiring",
            verifier_is_oracle=True,
        )
    numerical = all(c["passed"] for arm in fitted["arms"].values() for c in arm["gradient_checks"])
    if not numerical or not result["duplicate_parity"]["passed"]:
        apply_validation(
            value, [dict(name="numerical_integrity", required=True, passed=False, exit_code=1)]
        )
    contrast = {}
    for row in plan.get("upstream", {}).get(7958, {}).get("rows", []):
        contrast.setdefault(row["family_id"], {})[row["arm"]] = row["parsed"]["probability"]
    value["protocol_contrast"] = dict(
        producer_id=7958,
        truth_claim=False,
        current_model_calls=0,
        scope="Separate historical LLM full/erased protocol; probability movement is not improved truth detection.",
        rows=[
            dict(
                family_id=fid,
                full_q=r.get("full_source"),
                erased_q=r.get("source_erased"),
                delta=r["source_erased"] - r["full_source"]
                if r.get("source_erased") is not None and r.get("full_source") is not None
                else None,
            )
            for fid, r in contrast.items()
        ],
    )
    return value


def replay(value: Json) -> Json:
    """Cold reconstruction rejects changed bytes, interventions and scientific reductions."""
    if value["verdict_class"] in {"blocked", "disqualified"} and value["ablation_ready_score"]:
        raise ValueError("unsafe_readiness")
    for item in value["code_config_hashes"] + value["raw_shard_hashes"]:
        checked(item)
    if not value["checkpoints"]:
        return dict(passed=True, blocked=True)
    parts = {k: json.loads(checked(v).read_text()) for k, v in value["checkpoints"].items()}
    if a.interventions(parts["public"]) != parts["interventions"]:
        raise ValueError("intervention_drift")
    reduced = a.evaluate(
        parts["heads"],
        parts["policies"],
        parts["data"]["evaluation"],
        parts["interventions"]["evaluation"],
    )
    for k, v in reduced.items():
        if value[k] != v and not (
            k == "added_information_score" and value["verdict_class"] == "disqualified"
        ):
            raise ValueError("reduction_drift:" + k)
    return dict(passed=True, rows=len(reduced["rows"]))


def apply_validation(value: Json, receipts: list[Json]) -> None:
    """Preserve all failed owned checks and keep repository health separate."""
    value["validation_receipts"].extend(receipts)
    value["repository_health"] = dict(
        current=[r for r in value["validation_receipts"] if not r.get("required", True)]
    )
    if any(not r["passed"] for r in receipts if r.get("required", True)):
        value.update(
            verdict_class="disqualified",
            honest_verdict="complete_disqualified_owned_validation",
            ablation_ready_score=0,
            added_information_score=0,
        )
        value["acceptance_gate_results"].update(
            validity=False, readiness=False, added_information=False
        )


def terminal_check(candidate: Path) -> Json:
    """External validators and cold replay inspect exactly the proposed final bytes."""
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
    with TemporaryDirectory(prefix="carnot-7984-cold-") as directory:
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
    """Publish one atomic primary with byte-bound terminal and consumer receipts."""
    raw = output.parent / "raw" / output.stem
    value["terminal_validation_sidecar_path"] = str(raw / "terminal_validation.json")
    value["field_principles"] = {
        k: "Bind current identity and actual work to exact evidence; readiness does not establish added information."
        for k in value
    }
    atomic_json(raw / "terminal_candidate.json", value)
    report = terminal_check(raw / "terminal_candidate.json")
    if not report["passed"]:
        apply_validation(
            value,
            [
                dict(
                    name="terminal_before_disqualification",
                    required=True,
                    passed=False,
                    exit_code=1,
                    report=report,
                )
            ],
        )
        value["flagged_adversarial"] = report.get("flagged_adversarial", False)
        value["historical_required_failures"].append(dict(scope="owned_terminal", report=report))
    receipt = publish_primary(output, value, terminal_check)
    atomic_json(raw / "terminal_validation.json", receipt)
    readers = reader_receipt(
        TASK, output.parent, field="ablation_ready_score", expected=value["ablation_ready_score"]
    )
    atomic_json(raw / "primary_resolution.json", readers)
    if not readers["passed"]:
        raise ValueError("primary_resolution")


def main(argv: list[str] | None = None) -> int:
    """Bounded CPU CLI supports actual private routes and independent cold replay."""
    from carnot.reporting import evidence_ablation_validation_7984 as validation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261001", choices=["20261001"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--fixture-input", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--validation-worker", action="store_true")
    args = parser.parse_args(argv)
    started, phases = time.monotonic(), []
    progress("begin")
    try:
        if args.cold_replay:
            replay(json.loads(args.cold_replay.read_text()))
            progress("replay_passed")
            return 0
        output = args.output.absolute()
        raw = output.parent / "raw" / output.stem
        with TemporaryDirectory(prefix="carnot-7984-") as directory:
            manifest, receipts, counts = {}, [], {}
            if not args.validation_worker:
                progress("freeze_validation_manifest")
                manifest = validation.freeze(raw, Path(directory))
            progress("authenticate_inputs")
            mark = time.monotonic()
            if args.fixture_input:
                fixture = json.loads(args.fixture_input.read_text())
                failures, plan = (
                    [],
                    dict(checks=[], refs=[reference(args.fixture_input)], upstream={}),
                )
                public, fixture_data = fixture["public"], fixture["data"]
            else:
                failures, plan = authenticate(args.root)
                public, fixture_data = {} if failures else load_public(plan), None
            phases.append(dict(phase="authentication", duration_s=time.monotonic() - mark))
            mark = time.monotonic()
            value = (
                base(failures)
                if failures
                else measure(public, plan, raw, fixture_data=fixture_data)
            )
            phases.append(dict(phase="cached_ablation", duration_s=time.monotonic() - mark))
            value.update(
                preconditions_checked=plan["checks"],
                cited_upstream_artifacts=plan["refs"],
                code_config_hashes=[reference(ROOT / p) for p in OWNED + DEPENDENCIES],
                raw_shard_hashes=list(value["checkpoints"].values()),
            )
            value["historical_required_failures"].extend(
                dict(producer_id=eid, scope="historical_upstream", failure=r)
                for eid, v in plan["upstream"].items()
                for r in v.get("historical_required_failures", [])
            )
            if manifest:
                progress("owned_validation")
                mark = time.monotonic()
                receipts = validation.execute(manifest, raw)
                counts = validation.coverage_counts(Path(directory))
                value["validation_command_manifest_path"] = str(raw / "validation_manifest.json")
                apply_validation(value, receipts)
                phases.append(dict(phase="validation", duration_s=time.monotonic() - mark))
            value["coverage_statement_counts"] = counts
            value["reproducibility_checksum"] = canonical_hash(
                dict(
                    config=a.CONFIG,
                    code=value["code_config_hashes"],
                    inputs=value["raw_shard_hashes"],
                    upstream=plan["refs"],
                )
            )
            value["duration_s"], value["phase_spans"] = time.monotonic() - started, phases
            progress("publish")
            publish(output, value)
            progress("complete")
            return 0
    except (ValueError, OSError, KeyError, TypeError, StopIteration) as error:
        print(f"[exp7984] failed={error}", flush=True)
        return 1
