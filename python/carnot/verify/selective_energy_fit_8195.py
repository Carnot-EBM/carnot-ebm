"""REQ-VERIFY-8195: seal separate weight, temperature and set calibration.

Reused exposed Qwen evidence can qualify execution but cannot prove independent
benefit. The reserved feature and label shards are never read by this module.
"""

from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import time
from typing import Any
from unittest.mock import patch

import numpy as np
import yaml

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar, reader_receipt
from carnot.verify import selective_methods_8194 as methods
from carnot.verify import selective_rule_8194 as n
from carnot.verify import sentence_energy_fit_8183 as fit

Json = dict[str, Any]
ROOT = fit.ROOT
NAME = "experiment_8195_v708_selective_energy_fit"
TASK = "exp8195-selective-energy-fit"
MODULE = "python/carnot/verify/selective_energy_fit_8195.py"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_selective_energy_fit_8195.py"
OWNED = [MODULE, CLI]
RUN_DATE = "20261006"
MODEL_SPECS: list[Json] = []
UPSTREAM = "results/experiment_8194_v708_selective_methods.json"
UPSTREAM_PIN = "sha256:bbf947488354d6d777d429a6f74bcefdb4019761049f454dca00dc12f9bbc707"
CONFIG = dict(seed=n.SEED, alpha=0.05, costs=n.base.CONFIG["costs"], point_thresholds=[0.1, 0.5])
execution = fit.execution
reference = fit.reference
BASE_MAIN = fit.main


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Actual completed counts distinguish a slow child from an idle process."""
    print(f"[exp8195] phase={phase} completed={completed} pending={pending}", flush=True)


def reduce(data: Json) -> Json:
    """Use scalar dot products and original targets to recompute every source."""
    rows, roles, heads = data["rows"], data["roles"], data["heads"]
    if any(
        set(r)
        - {
            "unit_id",
            "source_cluster_id",
            "source_id",
            "role",
            "x",
            "y",
            "status",
            "exclusion_reason",
            "historical_paired_control",
            "slot",
        }
        for r in rows
    ):
        raise ValueError("evaluator_label")
    manifest = n.freeze_roles(rows, roles["reserved"])
    if manifest != roles:
        raise ValueError("source_identity")
    role_lookup = {
        r["unit_id"]: role
        for role in ("head_fit", "temperature_fit", "calibration")
        for r in roles[role]
    }
    features = [
        dict(
            unit_id=r["unit_id"],
            source_cluster_id=r["source_cluster_id"],
            x=r["x"],
            historical_x=r.get("historical_paired_control", {}).get("x"),
            status=r["status"],
            exclusion_reason=r["exclusion_reason"],
        )
        for r in rows
    ]
    vector = n.predict(features, heads, data["control"])
    scalar = n.predict(features, heads, data["control"], scalar=True)
    targets = {r["unit_id"]: r["y"] for r in rows}
    scored, parity = [], []
    # A calibration source can sit exactly on its order statistic. Canonical
    # scalar decisions retain that tie; vector arithmetic checks probabilities
    # separately so a rounding error cannot choose a different action.
    for expected, actual in zip(vector, scalar, strict=True):
        p, q = actual["p"], expected["p"]
        err = abs(p - q) if p is not None else 0.0
        passed = err <= 1e-10
        if not passed:
            raise ValueError("probability_decision_parity")
        target = targets[actual["unit_id"]]
        row = methods.audit.score(actual, target)
        row.update(
            condition=role_lookup[row["unit_id"]],
            role=role_lookup[row["unit_id"]],
            brier=(p - target) ** 2 if p is not None and target in (0, 1) else None,
            log_loss=-math.log(max(1e-15, p if target else 1 - p))
            if p is not None and target in (0, 1)
            else None,
            set_size=len(actual["prediction_set"])
            if actual["prediction_set"] is not None
            else None,
            singleton_coverage=int(actual["prediction_set"] == [target]),
        )
        scored.append(row)
        if actual["arm"] == "equivalent_logistic_set":
            local = next(
                r for r in scalar if r["unit_id"] == actual["unit_id"] and r["arm"] == "local_set"
            )
            parity.append(
                dict(
                    unit_id=actual["unit_id"],
                    maximum_absolute_error=abs(p - local["p"]) if p is not None else 0.0,
                    passed=passed
                    and actual["action"] == local["action"]
                    and actual["prediction_set"] == local["prediction_set"],
                )
            )
    summaries = []
    for role in ("head_fit", "temperature_fit", "calibration"):
        for arm in n.ARMS:
            selected = [r for r in scored if r["role"] == role and r["arm"] == arm]
            available = [r for r in selected if r["brier"] is not None]
            summaries.append(
                dict(
                    role=role,
                    arm=arm,
                    intended=len(selected),
                    completed=len(available),
                    typed_cost=sum(r["numerator"] for r in selected if r["numerator"] is not None)
                    / sum(r["denominator"] for r in selected),
                    brier=sum(r["brier"] for r in available) / len(available)
                    if available
                    else None,
                    log_loss=sum(r["log_loss"] for r in available) / len(available)
                    if available
                    else None,
                    set_size=sum(r["set_size"] for r in selected if r["set_size"] is not None)
                    / len(selected),
                    singleton_coverage=sum(r["singleton_coverage"] for r in selected)
                    / len(selected),
                )
            )
    shifts = [
        dict(
            shift=s,
            passed=abs(n.logit_probability([s, s + 2], 0.5) - n.logit_probability([0, 2], 0.5))
            <= 1e-10,
        )
        for s in (-100.0, 0.0, 100.0)
    ]
    return dict(
        rows=scored,
        summaries=summaries,
        eligible_count=sum(r["x"] is not None and r["y"] in (0, 1) for r in rows),
        equivalent_logistic_parity=dict(passed=all(r["passed"] for r in parity), rows=parity),
        common_logit_shift_invariance=dict(passed=all(r["passed"] for r in shifts), rows=shifts),
    )


def diagnostics(rows: list[Json], roles: Json, raw: Path) -> list[Json]:
    """Diagnostic heads test the plumbing and never select a treatment or alpha."""
    output = []
    for condition in ("shuffled_fit_labels", "no_signal"):
        progress("before_benchmark_" + condition)
        changed = deepcopy(rows)
        available = {r["unit_id"] for r in rows if r["x"] is not None and r["y"] in (0, 1)}
        selected = {r["unit_id"] for r in roles["head_fit"]} & available
        if condition == "shuffled_fit_labels":
            labels = [r["y"] for r in changed if r["unit_id"] in selected]
            np.random.default_rng(n.SEED).shuffle(labels)
            for row, label in zip(
                [r for r in changed if r["unit_id"] in selected], labels, strict=True
            ):
                row["y"] = label
        else:
            for row in changed:
                if row["x"] is not None:
                    row["x"] = [0.0] * 16
        trained = n.train(changed, roles, raw / condition)
        output.append(dict(condition=condition, diagnostic_only=True, **trained))
        progress("after_benchmark_" + condition, len(output), 2 - len(output))
    return output


def measure(root: Path, raw: Path, *, fixture_mode: bool = False, mutation: str = "") -> Json:
    """Check external gates before materializing only exposed fit/tune roles."""
    began = time.monotonic()
    raw.mkdir(parents=True, exist_ok=True)
    work: Json = dict(
        checks=[],
        refs=[],
        evidence={},
        raw_shard_hashes=[],
        trained_head_specs=[],
        historical_model_provenance={},
        cited_upstream_artifacts=[],
        owned_failure="",
        role_manifest={},
        class_support_by_role={},
        precondition_receipts=[],
        materialized_rows=[],
    )
    progress("before_preconditions")
    try:
        fit.gate(
            work,
            Path(sys.executable),
            "python_runtime_supported",
            True,
            sys.version_info >= (3, 11),
        )
        probe = raw / ".storage_probe"
        probe.write_bytes(b"private writable custody")
        fit.gate(
            work,
            raw,
            "private_writable_storage",
            True,
            probe.read_bytes() == b"private writable custody",
        )
        probe.unlink()
        protocol = fit.bind(
            work, dict(path=str(ROOT / methods.PROTOCOL), sha256=methods.PROTOCOL_PIN), raw
        )
        retired_path = ROOT / "ops/exclusion_manifest.yaml"
        retired = yaml.safe_load(retired_path.read_text())
        retired_ids = {
            r.get("experiment_id")
            for k in ("retired", "retired_experiments")
            for r in retired.get(k, [])
        }
        fit.gate(
            work,
            retired_path,
            "upstream_not_retired",
            True,
            not bool({8154, 8179, 8182, 8194} & retired_ids),
        )
        if fixture_mode:
            rows, reserved, control = methods.fixture()
            plan: Json = dict(refs=[], checks=[], upstream=[], historical_model_provenance={})
        else:
            for name in (UPSTREAM, *fit.PINS):
                progress("before_subprocess_summarize_" + Path(name).stem)
                spec = dict(
                    name="summarize_" + Path(name).stem,
                    argv=[
                        str(ROOT / ".venv/bin/python"),
                        "-u",
                        "scripts/summarize_artifact.py",
                        str(root / name),
                    ],
                    deadline_s=60,
                    expected_exit=0,
                )
                with (
                    (raw / (spec["name"] + "_supervision.log")).open("x") as stream,
                    redirect_stdout(stream),
                ):
                    receipt = execution.run_check(
                        ROOT, spec, raw, raw / "precondition_logs", heartbeat_s=20
                    )
                work["precondition_receipts"].append(receipt)
                progress("after_subprocess_summarize_" + Path(name).stem, 1, 0)
                fit.gate(
                    work, root / name, "upstream_summary_normal_exit", 0, receipt["actual_exit"]
                )
            upstream = fit.bind(work, dict(path=str(root / UPSTREAM), sha256=UPSTREAM_PIN), raw)
            for field, expected in (
                ("selective_protocol_ready_score", 1),
                ("required_checks_passed", True),
                ("flagged_adversarial", False),
            ):
                fit.gate(work, root / UPSTREAM, field, expected, upstream.get(field))
            report = read_bound_sidecar(
                root / UPSTREAM, fit.historical.publication_sidecar(upstream)
            )
            fit.gate(
                work, root / UPSTREAM, "upstream_terminal_passed", True, report["report"]["passed"]
            )
            for name, identity in (
                (UPSTREAM, 8194),
                (fit.UPSTREAM, 8182),
                (fit.METHODS, 8179),
                (fit.CONTROL, 8154),
            ):
                value = json.loads((root / name).read_text())
                receipt = reader_receipt(
                    value["task_id"],
                    root / "results",
                    field="required_checks_passed",
                    expected=True,
                )
                fit.gate(
                    work, root / name, "terminal_reader_" + str(identity), True, receipt["passed"]
                )
            plan = fit.inputs(root, raw / "fit_inputs")
            work["checks"].extend(plan["checks"])
            work["refs"].extend(plan["refs"])
            if any(not c["passed"] for c in plan["checks"]):
                raise ValueError("fit_inputs")
            rows, reserved = plan["rows"], protocol["role_manifest"]["reserved"]
            for row in rows:
                row["source_id"] = protocol["fit_source_id_map"][row["unit_id"]]
            control = next(h for h in plan["controls"] if h["arm"] == "radial16")
            work["cited_upstream_artifacts"] = [
                *plan["upstream"],
                dict(
                    experiment_id=8194,
                    sha256=UPSTREAM_PIN,
                    fields_imported=[
                        "selective_protocol_ready_score",
                        "role_manifest",
                        "protocol_sha256",
                    ],
                ),
            ]
            work["historical_model_provenance"] = plan["historical_model_provenance"]
        if mutation:
            fit.gate(work, root / UPSTREAM, "selective_protocol_ready_score", 1, 0)
        roles = n.freeze_roles(rows, reserved)
        if not fixture_mode:
            normalize = lambda rs: sorted(rs, key=lambda r: r["source_cluster_id"])
            fit.gate(
                work,
                ROOT / methods.PROTOCOL,
                "frozen_role_ids",
                {k: normalize(v) for k, v in protocol["role_manifest"].items()},
                {k: normalize(v) for k, v in roles.items()},
            )
            for ref in protocol["source_artifact_hashes"]:
                if "8184" not in ref["path"] and "8185" not in ref["path"]:
                    fit.gate(
                        work,
                        Path(ref["path"]),
                        "versioned_operand_sha256",
                        ref["sha256"],
                        sha256_file(Path(ref["path"])),
                    )
        work["role_manifest"], work["class_support_by_role"] = roles, n.class_support(rows, roles)
        work["materialized_rows"] = rows
        for role, minimum, per_class in (
            ("head_fit", 72, 0),
            ("temperature_fit", 24, 8),
            ("calibration", 48, 20),
        ):
            support = work["class_support_by_role"][role]
            methods.record_minimum(
                work, ROOT / methods.PROTOCOL, role + "_completed", minimum, support["completed"]
            )
            for label in (0, 1):
                methods.record_minimum(
                    work,
                    ROOT / methods.PROTOCOL,
                    role + "_class_" + str(label),
                    per_class,
                    support["classes"][str(label)],
                )
        progress("after_preconditions", 192, 0)
        try:
            progress("before_benchmark_natural_fit")
            trained = n.train(rows, roles, raw / "natural_fit")
            calibration_ids = {r["unit_id"] for r in roles["calibration"]}
            for head in trained["heads"]:
                head["quantiles"] = [
                    n.quantile(
                        [
                            n.probability(head, r["x"], scalar=True)
                            if label == 0
                            else 1 - n.probability(head, r["x"], scalar=True)
                            for r in rows
                            if r["unit_id"] in calibration_ids
                            and r["x"] is not None
                            and r["y"] == label
                        ]
                    )
                    for label in (0, 1)
                ]
            fixtures = diagnostics(rows, roles, raw / "diagnostics")
            progress("after_benchmark_natural_fit", 2, 0)
        except (ValueError, TimeoutError) as error:
            work["owned_failure"] = str(error)
            raise
        data = dict(rows=rows, roles=roles, heads=trained["heads"], control=control)
        reduced = reduce(data)
        frozen = dict(
            **trained,
            controls=control,
            diagnostic_heads=fixtures,
            feature_order=protocol["features"],
            costs=protocol["costs"],
            point_thresholds=protocol["point_thresholds"],
            alpha=protocol["alpha"],
            arms=n.ARMS,
            role_manifest=roles,
            source_hashes=work["refs"],
            source_masks=[
                dict(
                    unit_id=r["unit_id"],
                    source_cluster_id=r["source_cluster_id"],
                    status=r["status"],
                    exclusion_reason=r["exclusion_reason"],
                )
                for r in rows
            ],
            protocol_sha256=methods.PROTOCOL_PIN,
            decision_arithmetic="scalar_fsum_inclusive_ties",
        )
        work["evidence"] = data
        for condition, hs in [
            ("natural", trained["heads"]),
            *[(d["condition"], d["heads"]) for d in fixtures],
        ]:
            work["trained_head_specs"].extend(
                dict(
                    arm=h["arm"],
                    condition=condition,
                    dimensions=h["dimensions"],
                    parameter_count=len(h["weights"]) + 1,
                    training_sources=len(h["head_fit_source_ids"]),
                    seed=n.SEED,
                    converged=h["converged"],
                )
                for h in hs
            )
        for name, content in (
            ("frozen_heads", frozen),
            ("primitive_evidence", data),
            ("independent_reduction", reduced),
            ("role_manifest", roles),
        ):
            atomic_json(raw / (name + ".json"), content)
            work["raw_shard_hashes"].append(reference(raw / (name + ".json")))
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        if not work["owned_failure"] and all(c["passed"] for c in work["checks"]):
            work["checks"].append(
                dict(
                    check="input_custody",
                    upstream=str(root / UPSTREAM),
                    path=str(root / UPSTREAM),
                    hash=None,
                    artifact_field="input_custody",
                    op="==",
                    expected="authenticated_fit_calibration",
                    observed=str(error),
                    passed=False,
                )
            )
    work.update(
        duration_s=time.monotonic() - began,
        phase_spans=[
            dict(
                phase="authenticate_fit_temperature_calibration_seal",
                start_s=0,
                duration_s=time.monotonic() - began,
            )
        ],
        code_config_hashes=[
            reference(ROOT / p)
            for p in [
                *OWNED,
                TEST,
                methods.NUMERIC,
                fit.NUMERIC,
                fit.historical.NUMERIC,
                methods.PROTOCOL,
            ]
        ],
    )
    atomic_json(raw / "measurement.json", work)
    progress("measurement_complete", int(bool(work["evidence"])), 0)
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Execution and support gate readiness; favorable predictions are unnecessary."""
    reduced = (
        reduce(work["evidence"])
        if work["evidence"]
        else dict(
            rows=[],
            summaries=[],
            eligible_count=0,
            equivalent_logistic_parity=dict(passed=False, rows=[]),
            common_logit_shift_invariance=dict(passed=False, rows=[]),
        )
    )
    failures = [c for c in work["checks"] if not c["passed"]]
    checked = (
        bool(receipts)
        and all(r.get("passed") is True for r in receipts)
        and not work["owned_failure"]
    )
    support = bool(work["evidence"]) and not failures
    ready = int(checked and support and reduced["equivalent_logistic_parity"]["passed"])
    verdict = (
        "disqualified"
        if not checked
        else "blocked"
        if failures
        else "circular_positive"
        if fixture
        else "null"
    )
    suffix = (
        "owned_validation"
        if not checked
        else failures[0]["check"]
        if failures
        else "fixture"
        if fixture
        else "selective_fit_sealed"
    )
    rows = work["materialized_rows"]
    lookup = {r["unit_id"]: r for r in rows}
    role_rows = {
        role: [lookup[r["unit_id"]] for r in work["role_manifest"].get(role, [])]
        for role in ("head_fit", "temperature_fit", "calibration")
    }
    frozen = raw / "frozen_heads.json"
    value: Json = dict(
        experiment_id=8195,
        task_id=TASK,
        milestone="2026.10.708",
        run_date=RUN_DATE,
        honest_verdict=f"complete_{verdict}_{suffix}",
        verdict_class=verdict,
        verifier_is_oracle=fixture,
        fixture_protocol_only=fixture,
        claim_scope="sealed selective heads on exposed fit/temperature/calibration; reserved audit unmeasured; no superiority over equivalent logistic scoring",
        exposure_scope="exposed_development_within_run_disjoint",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=checked,
        flagged_adversarial=False,
        validation_receipts=receipts,
        precondition_receipts=work["precondition_receipts"],
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=True,
        gate_check_summary=work["checks"],
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        trained_head_specs=work["trained_head_specs"],
        model_invocation_counts=dict(model_loads=0, generations=0, live_model_calls=0),
        call_ledger=[],
        historical_model_provenance=work["historical_model_provenance"],
        cited_upstream_artifacts=work["cited_upstream_artifacts"],
        intended_count=192,
        independent_count=reduced["eligible_count"],
        completed_count=reduced["eligible_count"],
        excluded_count=192 - reduced["eligible_count"],
        failed_count=0,
        censored_count=0,
        sample_size_budget=dict(
            head_fit=96,
            temperature_fit=32,
            calibration=64,
            support=dict(
                head_fit=72,
                temperature_fit=24,
                temperature_class=8,
                calibration=48,
                calibration_class=20,
            ),
        ),
        duration_s=work["duration_s"],
        random_seed=n.SEED,
        reproducibility_checksum=canonical_hash(
            dict(config=CONFIG, refs=work["refs"], reduced=reduced)
        ),
        source_artifact_hashes=work["refs"],
        raw_shard_hashes=work["raw_shard_hashes"],
        code_config_hashes=work["code_config_hashes"],
        phase_spans=work["phase_spans"],
        acceptance_gates=CONFIG,
        field_principles=dict(
            readiness="Normal owned validation and calibration support; null benefit does not block downstream audit.",
            inference="Imported Qwen calls are historical; current calls are zero.",
            scope="Exposed data and oracle fixtures cannot establish independent generalization.",
            calibration="Weights, temperatures and finite-sample set thresholds use separate frozen source roles.",
            masks="Failed source slots remain and missing features escalate.",
        ),
        selective_fit_ready_score=ready,
        support_sufficient=support,
        frozen_heads_path=str(frozen) if frozen.is_file() else None,
        frozen_heads_sha256=sha256_file(frozen) if frozen.is_file() else None,
        fit_rows=role_rows["head_fit"],
        temperature_rows=role_rows["temperature_fit"],
        calibration_rows=role_rows["calibration"],
        quantile_rows=[
            dict(arm=h["arm"], label=y, **q)
            for h in work["evidence"].get("heads", [])
            for y, q in enumerate(h["quantiles"])
        ],
        class_support_by_role=work["class_support_by_role"],
        role_manifest=work["role_manifest"],
        reserved_outcomes_opened=False,
        measurement_reference=reference(raw / "measurement.json"),
        repository_health=work.get("repository_health", {}),
        methodology_note="Qualified fit-only Gaussian geometry and ridge objective; separate bounded temperature and class-conditional inclusive quantiles, exact infinity convention; frozen costs and seven arms; diagnostic shuffled labels and no signal; zero LLM calls. Calibration coverage is descriptive on exposed evidence.",
        **reduced,
    )
    return value


def replay(path: Path) -> bool:
    """Rehash sealed custody and independently reject producer aggregate changes."""
    try:
        value = json.loads(path.read_text())
        for ref in [
            value["measurement_reference"],
            *value["source_artifact_hashes"],
            *value["raw_shard_hashes"],
            *value["code_config_hashes"],
        ]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        for receipt in [*value["validation_receipts"], *value["precondition_receipts"]]:
            if (
                receipt.get("log_path")
                and sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]
            ):
                return False
        work = json.loads(Path(value["measurement_reference"]["path"]).read_text())
        if work["evidence"]:
            for name, expected in (
                ("primitive_evidence", work["evidence"]),
                ("independent_reduction", reduce(work["evidence"])),
            ):
                ref = next(r for r in work["raw_shard_hashes"] if Path(r["path"]).stem == name)
                if json.loads(Path(ref["path"]).read_text()) != expected:
                    return False
        return bool(
            value
            == build(
                work,
                Path(value["terminal_validation_sidecar_path"]).parent,
                value["validation_receipts"],
                fixture=value["fixture_protocol_only"],
            )
        )
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze explicit paths and include only statements added by this task."""
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
    ):
        specs = execution.manifest(private, candidate)
    specs["commands"][0]["argv"].remove("-s")
    specs["commands"][1]["argv"] = [
        str(ROOT / ".venv/bin/pytest"),
        "-n0",
        "-o",
        "addopts=",
        "--no-cov",
        "-q",
        "tests/python/test_source_boundary_7852.py",
        "tests/python/test_experiment_7942_v689_sentence_labels.py",
        "tests/python/test_primary_publication_7928.py",
    ]
    specs["repository_health"]["deadline_s"] = 300
    return specs


def main(argv: list[str] | None = None) -> int:
    """Reuse the qualified supervisor, log sealing and atomic primary publisher."""
    with ExitStack() as stack:
        for name in (
            "ROOT",
            "NAME",
            "TASK",
            "MODULE",
            "CLI",
            "TEST",
            "OWNED",
            "RUN_DATE",
            "MODEL_SPECS",
            "progress",
            "manifest",
            "measure",
            "build",
            "replay",
        ):
            stack.enter_context(patch.object(fit, name, globals()[name]))
        return int(BASE_MAIN(argv))
