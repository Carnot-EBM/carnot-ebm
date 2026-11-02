"""REQ-REPORT-8046: seal exposed development protocols without new model claims.

Public identities and methods are frozen before any current evaluator access.
Small numerical controls test the acceptance interface; scientific trajectories
and likelihood capture belong to later experiments.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from carnot import experiment_8032_v696_sealed_methods as prior
from carnot import experiment_8045_v697_scorer_workspace as venue
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary
from carnot.verify.evidence_features_7980 import normalized
from carnot.verify.learning_retention_audit_8026 import action, cost

Json = dict[str, Any]
Array = NDArray[np.float64]
ROOT = prior.ROOT
NAME = "experiment_8046_v697_branch_protocols"
CLI = f"scripts/experiments/{NAME}.py"
MODULE = f"python/carnot/{NAME}.py"
TEST = "tests/python/test_branch_protocols_8046.py"
ROLE_COUNTS = dict(fit=64, tune=32, evaluation=96, stream=256, retention=64)
UPSTREAM = {
    8032: prior.NAME,
    8039: "experiment_8039_v696_learning_benefit_audit",
    8020: prior.UPSTREAM[8020],
}
INPUTS = [
    "AGENTS.md",
    "CODEX.md",
    "CLAUDE.md",
    "ops/e2e-test-plan.md",
    "openspec/capabilities/research-reporting/spec.md",
    "scripts/experiment_template.py",
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/primary_publication.py",
    f"python/carnot/{prior.NAME}.py",
    "python/carnot/experiment_8019_v695_eligible_targets.py",
    "python/carnot/verify/windowed_online_8038.py",
    "python/carnot/verify/learning_benefit_8039.py",
    "research-references.md",
    "tests/fixtures/v697/design.md.gz",
    venue.MODULE,
    venue.old.OWNED[0],
    "python/carnot/reporting/experiment_7303_validation_scope.py",
    "python/carnot/verify/evidence_features_7980.py",
    "python/carnot/verify/learning_retention_audit_8026.py",
]
METHODS: Json = dict(
    source=dict(
        roles=dict(fit=64, tune=32, evaluation=96),
        conditions=["full_A", "full_B", "no_source_A", "no_source_B"],
        native_context="independent fresh_full per call",
        context_limit=4096,
        answer_limit=384,
        truncation=False,
        replacements=False,
        denominator="all original slots including failures",
        duplicate_absolute_mean_nll_tolerance=1e-6,
        duplicate_conditions=["full", "no_source"],
        normalization="finite aligned target-token log probabilities with normalized vocabulary",
        features=["mean full-source NLL", "no-source minus full-source mean NLL"],
        floors=dict(fit=[48, 8], tune=[24, 4], evaluation=[72, 8]),
    ),
    learning=dict(
        roles=dict(stream=256, retention=64),
        delay=20,
        durable_issue_first=True,
        partition="SHA256 canonical original source-cluster ID integer modulo4; bucket0 guard",
        arms=["unconstrained", "feedback_constrained", "frozen_no_write"],
        guard_gradients=False,
        guard_minimum=8,
        guard_per_class=2,
        updates_per_block=4,
        block_update_releases=16,
        step=0.01,
        ridge=0.001,
        cap_steps=64,
        seeds=list(range(101, 121)),
        numerical_budget_s=900,
        selection="cumulative update-only pool; canonical hash order then shared seeded uniform without replacement within block",
        candidate="each adaptive arm uses its own current state; identical released information and selected IDs",
        unknown_targets="excluded without compressing original slots",
        terminal_flush=False,
        retention_access="evaluator only after final heads sealed",
    ),
    guard=dict(
        alphas=[1, 0.5, 0.25, 0.125, 0],
        tolerance=1e-12,
        baseline="frozen initial calibrated head",
        rows="ALL released guard rows",
        predicates=[
            "mean Brier nonincrease",
            "mean typed cost nonincrease",
            "no new row false accept",
        ],
        constrained="largest admissible alpha; none restores frozen head",
        unconstrained="compute identical diagnostics; commit alpha1",
        guarantee="empirical guard only",
    ),
    costs=dict(
        unsupported_accept=5,
        supported_reject=1,
        escalate=0.5,
        correct=0,
        probability="unsupportedness",
        accept="p<.1",
        reject="p>.5",
        ties="escalate",
    ),
    hypotheses=[
        dict(
            id="H1",
            metric="Brier",
            comparison="source-feature logistic versus full-NLL-only",
            margin=0.01,
        ),
        dict(
            id="H2",
            metric="typed cost",
            comparison="spline versus same-feature logistic",
            margin=0.02,
        ),
        dict(
            id="H3",
            metric="later update-role typed cost",
            comparison="constrained versus unconstrained",
            margin=0.02,
        ),
    ],
    safety=dict(
        H1_cost_drift=0.01,
        H2_brier_drift=0.01,
        no_added_false_accepts=True,
        H3_per_seed_false_accept_baselines=["unconstrained", "frozen_no_write"],
        beneficial_changes=5,
        later_support=[120, 15],
        later_slots="common update-role future slots after common first update, before own release",
        retention_support=[48, 8],
        retention_brier_drift=0.01,
        retention_cost_drift=0.02,
        retention_arms=["unconstrained", "feedback_constrained"],
    ),
    statistics=dict(
        draws=10000,
        H1_H2="paired original source-group bootstrap",
        H3="paired chronological moving-block bootstrap on original256 slots; seed mean within source",
        primary_block=32,
        sensitivity_blocks=[16, 64],
        alpha=0.05,
        holm_family=["H1", "H2", "H3"],
        test="invert one-sided nonzero-margin tests",
        invalid_or_unavailable_p=1,
        uncertainty="conditional development only",
        independent_n="source groups; seeds, draws and duplicate calls add zero",
    ),
    access=dict(
        scoring="public source and complete answer only",
        fitter="fit labels only",
        tuner="tune labels only",
        evaluation="after sealed methods and durable prediction",
        feedback="original slot+20 after durable prediction; no future labels",
    ),
)


def partition(identity: str) -> int:
    """Hash a public cluster before labels so errors cannot change its role."""
    return int(canonical_hash(identity).split(":")[1], 16) % 4


def copy_evidence(ref: Json, raw: Path) -> list[Json]:
    """Split large historical bytes without losing their original file identity."""
    source = checked(ref)
    if source.stat().st_size < 90_000_000:
        return [prior.copy_bound(ref, raw, "history")]
    chunks = []
    with source.open("rb") as stream:
        while block := stream.read(32_000_000):
            path = raw / "history" / (ref["sha256"].split(":")[1] + f"-part{len(chunks)}.bin")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as output:
                output.write(block)
                output.flush()
                os.fsync(output.fileno())
            chunks.append(reference(path))
            prior.progress(
                "8046_historical_chunk",
                len(chunks),
                max(0, (source.stat().st_size - stream.tell() + 31_999_999) // 32_000_000),
            )
    manifest = prior.immutable(
        raw / "history" / (ref["sha256"].split(":")[1] + "-chunks.json"),
        dict(original=ref, chunks=chunks, byte_count=source.stat().st_size),
    )
    return [manifest, *chunks]


def accept_candidate(
    theta: Array, delta: Array, initial: Array, x: Array, y: Array, arm: str
) -> Json:
    """Check every candidate against the initial head, keeping rejection and reset separate."""
    if arm not in METHODS["learning"]["arms"]:
        raise ValueError("arm")
    if arm == "frozen_no_write":
        return dict(
            status="frozen",
            parameters=initial.tolist(),
            diagnostics=[],
            reset=False,
            rejected=False,
        )
    if len(y) < 8 or min(int(sum(y == c)) for c in (0, 1)) < 2:
        return dict(
            status="waiting_guard",
            parameters=theta.tolist(),
            diagnostics=[],
            reset=False,
            rejected=False,
        )

    def metrics(parameters: Array) -> tuple[float, float, Array]:
        p = 1 / (1 + np.exp(-np.clip(x @ parameters, -700, 700)))
        decisions = [action(float(v)) for v in p]
        return (
            float(np.mean((p - y) ** 2)),
            float(np.mean([cost(a, int(v)) for a, v in zip(decisions, y, strict=True)])),
            (p < 0.1) & (y == 1),
        )

    brier, typed, false_accept = metrics(initial)
    diagnostics = []
    for alpha in METHODS["guard"]["alphas"]:
        b, c, f = metrics(theta + alpha * delta)
        added = int(sum(f & ~false_accept))
        diagnostics.append(
            dict(
                alpha=alpha,
                brier=b,
                typed_cost=c,
                new_false_accepts=added,
                baseline_brier=brier,
                baseline_cost=typed,
                numerator=len(y),
                denominator=len(y),
                admissible=b <= brier + 1e-12 and c <= typed + 1e-12 and added == 0,
            )
        )
    admissible = [r["alpha"] for r in diagnostics if r["admissible"]]
    alpha = 1 if arm == "unconstrained" else admissible[0] if admissible else None
    return dict(
        status="reset" if alpha is None else "commit",
        alpha=alpha,
        parameters=(initial if alpha is None else theta + alpha * delta).tolist(),
        diagnostics=diagnostics,
        reset=alpha is None,
        rejected=alpha != 1,
    )


def causal_step(rows: list[Json], now: int, theta: Array, raw: Path) -> Json:
    """A private prefix control stores its prediction before reading due labels.

    This single-step control tests the protocol interface. It is not a measured
    learning trajectory. Released guard rows cannot enter its gradient pool.
    """
    raw.mkdir(parents=True, exist_ok=True)
    prediction = dict(
        slot=now, p=float(1 / (1 + np.exp(-np.array(rows[now]["x"]) @ theta))), durable=True
    )
    atomic_json(raw / "prediction.json", prediction)
    released = [r for r in rows if r["slot"] <= now - 20 and r["y"] in (0, 1)]
    guard = [r for r in released if partition(r["source_cluster_id"]) == 0]
    updates = [r for r in released if partition(r["source_cluster_id"]) != 0]
    ordered = sorted(updates, key=lambda r: canonical_hash(dict(seed=101, identity=r["family_id"])))
    selected = (
        prior.select(ordered, "cumulative", 101, len(updates) // 16)
        if len(updates) >= 16
        and len(guard) >= 8
        and min(sum(r["y"] == c for r in guard) for c in (0, 1)) >= 2
        else []
    )
    delta = np.zeros_like(theta)
    for row in selected:
        x = np.array(row["x"], dtype=float)
        delta -= 0.01 * (
            (float(1 / (1 + np.exp(-x @ (theta + delta)))) - row["y"]) * x + 0.001 * (theta + delta)
        )
    decisions = {
        arm: accept_candidate(
            theta,
            delta,
            theta,
            np.array([r["x"] for r in guard]),
            np.array([r["y"] for r in guard]),
            arm,
        )
        for arm in METHODS["learning"]["arms"]
    }
    return dict(
        prediction=prediction,
        selected_ids=[r["family_id"] for r in selected],
        guard_ids=[r["family_id"] for r in guard],
        gradient_displacement=delta.tolist(),
        acceptance=decisions,
    )


def seal(root: Path, raw: Path) -> Json:
    """Bind historical public slots and all overlaps without opening evaluator vaults."""
    prior.progress("8046_preconditions")
    refs: list[Json] = []
    checks = []
    values = {}
    for relative in [*INPUTS, *(f"results/{n}.json" for n in UPSTREAM.values())]:
        path = root / relative
        checks.append(
            venue.old.operand(
                path,
                "resource_exists",
                True,
                True if path.is_file() else "missing_resource",
                "exp8046_input",
            )
        )
        if path.is_file():
            refs.append(prior.copy_bound(reference(path), raw, "inputs"))
    for tool in ("python", "pytest", "coverage", "ruff", "mypy"):
        p = ROOT / ".venv/bin" / tool
        checks.append(
            venue.old.operand(p, "resource_exists", True, p.is_file(), "python_environment")
        )
    for identity, name in UPSTREAM.items():
        path = root / "results" / (name + ".json")
        if not path.is_file():
            continue
        value = json.loads(path.read_text())
        expected = dict(experiment_id=identity, verdict_class="null", flagged_adversarial=False)
        expected.update(
            {
                8032: dict(
                    protocol_ready_score=1,
                    likelihood_panel_ready_score=1,
                    learning_inputs_ready_score=1,
                ),
                8020: dict(energy_fit_ready_score=1),
                8039: {},
            }[identity]
        )
        checks += [
            venue.old.operand(
                path, k, v, value.get(k, "missing_field_contract_error"), f"exp{identity}"
            )
            for k, v in expected.items()
        ]
        try:
            sidecar = Path(value.get("terminal_validation_sidecar_path", "missing_sidecar"))
            prior.require(sidecar, "terminal_exists", True, sidecar.is_file())
            binding = json.loads(sidecar.read_text())
            binding = binding.get("publication", binding)
            report_path = Path(binding["sidecar_path"])
            prior.require(report_path, "sidecar_exists", True, report_path.is_file())
            report = json.loads(report_path.read_text())
            for p, obj in [(sidecar, binding), (report_path, report)]:
                for k, v in dict(
                    primary_sha256=reference(path)["sha256"], primary_path=str(path)
                ).items():
                    prior.require(p, k, v, obj.get(k))
                refs.append(prior.copy_bound(reference(p), raw, "inputs"))
            prior.require(
                report_path, "report.passed", True, report.get("report", {}).get("passed")
            )
            for ref in value["raw_shard_hashes"]:
                refs.extend(copy_evidence(ref, raw))
            values[identity] = value
        except (prior.Contract, KeyError, OSError, ValueError) as error:
            checks.append(
                error.gate
                if isinstance(error, prior.Contract)
                else venue.old.operand(
                    path, "contract", "valid terminal bytes", str(error), f"exp{identity}"
                )
            )
    roles: Json = {r: [] for r in ROLE_COUNTS}
    head: Json = {}
    protocol: Json = {}
    if 8032 in values and all(r["passed"] for r in checks):
        protocol = json.loads(checked(values[8032]["methods_reference"]).read_text())
        head = protocol["head"]
        for role in ("fit", "tune", "evaluation"):
            originals = [r for r in protocol["likelihood"] if r["role"] == role]
            roles[role] = [
                dict(
                    family_id=r["family_id"],
                    slot=i,
                    source_cluster_id=r["normalized_source"],
                    source_bytes=r["source_bytes"],
                    answer_bytes=r["answer_bytes"],
                    target_tokens=r["target_tokens"],
                    public_eligible=True,
                    exclusion_reason=None,
                )
                for i, r in enumerate(originals)
            ]
        for role in ("stream", "retention"):
            originals = json.loads(checked(protocol["public"][role]).read_text())["rows"]
            roles[role] = [
                dict(r, source_cluster_id=normalized(bytes.fromhex(r["source_bytes"])))
                for r in originals
            ]
        for role, rows in roles.items():
            checks.append(
                venue.old.operand(
                    root / "results" / (prior.NAME + ".json"),
                    "original_slots." + role,
                    list(range(ROLE_COUNTS[role])),
                    [r["slot"] for r in rows],
                    "exp8032",
                )
            )
    code = [prior.copy_bound(reference(ROOT / p), raw, "code") for p in [MODULE, CLI, TEST]]
    rows = [
        dict(
            family_id=r["family_id"],
            source_cluster_id=r["source_cluster_id"],
            role=role,
            original_slot=r["slot"],
            eligible=r["public_eligible"],
            exclusion_reason=r["exclusion_reason"],
            historically_exposed=True,
            condition="public_custody",
            arm="protocol",
            seed=None,
            numerator=int(r["public_eligible"]),
            denominator=1,
            measured=False,
        )
        for role, items in roles.items()
        for r in items
    ]
    groups: Json = {}
    for row in rows:
        groups.setdefault(row["source_cluster_id"], []).append(
            dict(role=row["role"], slot=row["original_slot"], family_id=row["family_id"])
        )
    partitions = [
        dict(
            family_id=r["family_id"],
            original_slot=r["slot"],
            source_cluster_id=r["source_cluster_id"],
            bucket=partition(r["source_cluster_id"]),
            feedback_role="guard" if partition(r["source_cluster_id"]) == 0 else "update",
        )
        for r in roles["stream"]
    ]
    manifests = {
        role: prior.immutable(raw / "roles" / (role + ".json"), dict(role=role, rows=items))
        for role, items in roles.items()
    }
    plan = dict(
        methods=METHODS,
        checks=checks,
        inputs=refs,
        code=code,
        roles=roles,
        rows=rows,
        role_manifests=manifests,
        guard_partition_rows=partitions,
        head=head,
        overlaps={k: v for k, v in groups.items() if len(v) > 1},
        evaluator_access_log=[],
        frozen_at_ns=time.time_ns(),
        historical_exposure=dict(
            all_roles_development_exposed=True,
            newly_unseen=False,
            imported_verdicts={str(k): v["honest_verdict"] for k, v in values.items()},
            prior_failures_preserved=True,
            no_current_evaluator_reads=True,
        ),
        inherited_eligibility="complete human targets only; unknown never supported; no replacements",
        inherited_support=protocol.get("frozen_support_by_role", {}),
    )
    prior.immutable(raw / "methods.json", plan)
    prior.progress("8046_methods_sealed", len(rows), 0)
    return plan


def commands(scratch: Path) -> list[CommandSpec]:
    """Reuse the qualified child venue while keeping the acceptance scope explicit."""
    specs = venue.commands(scratch)
    (scratch / "coverage.ini").write_text(
        "[run]\nparallel = True\ndata_file = "
        + str(scratch / ".coverage")
        + "\ninclude =\n"
        + "".join("    " + str(ROOT / p) + "\n" for p in [MODULE, CLI])
    )
    adapted = []
    for spec in specs:
        if spec.name == "focused_pytest":
            continue
        argv = tuple(
            a.replace(venue.MODULE, MODULE).replace(venue.CLI, CLI).replace(venue.TEST, TEST)
            for a in spec.argv
            if a not in {venue.old.TEST, venue.old.OWNED[0]}
        )
        adapted.append(replace(spec, argv=argv))
    return adapted


def validate(scratch: Path, raw: Path) -> Json:
    """Preserve real process exits and require all added statements to execute."""
    specs = commands(scratch)
    receipts = run_commands(
        ROOT,
        specs,
        log_dir=raw / "validation_logs",
        heartbeat_s=20,
        extra_env=dict(
            CARNOT_8046_COVERAGE_CONFIG=str(scratch / "coverage.ini"), JAX_PLATFORMS="cpu"
        ),
    )
    coverage = (
        json.loads((scratch / "coverage.json").read_text())
        if (scratch / "coverage.json").is_file()
        else dict(files={})
    )
    counts = {k: v["summary"] for k, v in coverage["files"].items() if k in {MODULE, CLI}}
    return dict(
        receipts=[dict(r, expected_exit_code=0, actual_exit_code=r["exit_code"]) for r in receipts],
        coverage=counts,
        coverage_report=coverage,
    )


def build(plan: Json, work: Json, raw: Path, validation: Json) -> Json:
    """Protocol readiness requires qualified custody and owned checks, not a win."""
    rows = plan["rows"]
    owned = [r for r in validation["receipts"] if r["scope"] == "owned"]
    counts = validation["coverage"]
    coverage_ok = set(counts) == {MODULE, CLI} and all(
        v["num_statements"] > 0 and v["missing_lines"] == 0 for v in counts.values()
    )
    failed = [r for r in plan["checks"] if not r["passed"]]
    gates = dict(
        inputs=not failed,
        owned_checks=bool(work["acceptance_manifest"])
        and [r["name"] for r in owned] == work["acceptance_manifest"]
        and all(r["passed"] for r in owned),
        coverage=coverage_ok,
        source_roles=all(
            len(plan["roles"][r]) == n
            for r, n in ROLE_COUNTS.items()
            if r not in {"stream", "retention"}
        ),
        learning_roles=len(plan["roles"]["stream"]) == 256
        and len(plan["roles"]["retention"]) == 64
        and bool(plan["head"]),
        evaluator_closed=plan["evaluator_access_log"] == [],
    )
    good = gates["inputs"] and gates["owned_checks"] and coverage_ok and gates["evaluator_closed"]
    kind = (
        "blocked"
        if failed
        else "null"
        if not work["acceptance_manifest"] or good
        else "disqualified"
    )
    eligible = sum(r["eligible"] for r in rows)
    sizes = dict(
        intended_count=512,
        eligible_count=eligible,
        completed_count=len(rows),
        excluded_count=len(rows) - eligible,
        failed_count=0,
        censored_count=512 - len(rows),
        independent_count=len({r["source_cluster_id"] for r in rows if r["eligible"]}),
    )
    value = dict(
        experiment_id=8046,
        task_id="exp8046-branch-protocols",
        milestone="2026.10.697",
        schema="carnot.v697.branch_protocols.v1",
        run_date="20261003",
        claim_scope="This invocation seals original historically exposed public source roles and feedback acceptance. It measures custody and owned validation only; no live scoring, new training trajectory, unseen evaluation or deployment benefit.",
        honest_verdict="complete_blocked_" + Path(failed[0]["path"]).stem.replace(".", "_")
        if failed
        else "complete_" + kind + "_branch_protocols",
        verdict_class=kind,
        gate_check_summary=plan["checks"],
        rows=rows,
        **sizes,
        sample_size_budget=dict(
            **sizes,
            unit="original public source-cluster custody slots",
            source_slots=192,
            learning_slots=320,
            model_calls=0,
            seeds=20,
            candidate_step_cap=64,
            numerical_budget_s=900,
        ),
        random_seed=69746,
        reproducibility_checksum=canonical_hash(dict(plan=plan, work=work, validation=validation)),
        cited_upstream_artifacts=plan["inputs"],
        code_config_hashes=plan["code"],
        raw_shard_hashes=[
            reference(raw / p)
            for p in ["methods.json", "work.json", "validation.json", "validation_commands.json"]
        ]
        + [plan["role_manifests"][r] for r in sorted(plan["role_manifests"])],
        checkpoint_references=[reference(raw / "methods.json")],
        acceptance_gate_results=gates,
        verifier_is_oracle=False,
        genuine_headroom=dict(measured=False, scope="protocol only"),
        positive_control_results=dict(
            scope="synthetic private guard and causal mutation unit controls; no scientific observations",
            empirical_safety_only=True,
        ),
        generalized_learning_benefit_score=0,
        validation_receipts=owned,
        repository_health=[r for r in validation["receipts"] if r["scope"] == "repository_health"],
        coverage_statement_counts=counts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        flagged_adversarial=False,
        preconditions_checked=plan["checks"],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        trained_head_specs=[],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        source_protocol_ready_score=int(good and gates["source_roles"]),
        learning_protocol_ready_score=int(good and gates["learning_roles"]),
        role_manifests=plan["role_manifests"],
        historical_exposure=plan["historical_exposure"],
        sealed_methods_hash=reference(raw / "methods.json")["sha256"],
        evaluator_access_log=plan["evaluator_access_log"],
        guard_partition_rows=plan["guard_partition_rows"],
        primary_hypotheses=[
            dict(
                h, p_value=1, measured=False, disposition="unavailable in protocol-only invocation"
            )
            for h in METHODS["hypotheses"]
        ],
        guard_acceptance_rules=METHODS["guard"],
        methods=METHODS,
        source_cluster_overlaps=plan["overlaps"],
        qualified_head=plan["head"],
        inherited_support=plan["inherited_support"],
        substrate_declaration=dict(
            reduction="aggregation_from_upstream_artifacts",
            numerical_work="verifier_scoring",
            mode="no_model_load",
            MODEL_SPECS=[],
            pretrained_model_calls=0,
        ),
        methodology_note="Methods and public identities precede current evaluator access. Historical development exposure prevents deployment or generalized benefit credit. Guard controls establish only finite empirical behavior.",
    )
    value["field_principles"] = {
        k: "Bind "
        + k
        + " to this invocation and its exact durable bytes; historical or synthetic evidence supplies no current scientific benefit."
        for k in value
    }
    for field in ("rows", "sample_size_budget", *sizes):
        value["field_principles"][field] = (
            "Count original custody slots and source clusters; preserve exclusions and denominators. Seeds and duplicate calls add no independent observations."
        )
    value["field_principles"]["gate_check_summary"] = (
        "Name the exact upstream operand, path and bytes with expected and observed values; missing fields are contract errors, never measured zeros."
    )
    for field in ("source_protocol_ready_score", "learning_protocol_ready_score"):
        value["field_principles"][field] = (
            "Require qualified protocol custody and all owned checks; readiness needs no scientific win and certifies no deployment correctness."
        )
    for field in (
        "historical_exposure",
        "guard_acceptance_rules",
        "primary_hypotheses",
        "evaluator_access_log",
        "sealed_methods_hash",
    ):
        value["field_principles"][field] = (
            "Seal method choice before current evaluator access; expose historical development reuse and finite empirical guard limits."
        )
    return value


def replay(value: Json) -> None:
    """Cold reduction checks primitive identities and every cited byte before counts."""
    for ref in (
        value["raw_shard_hashes"] + value["code_config_hashes"] + value["cited_upstream_artifacts"]
    ):
        checked(ref)
        if ref["path"].endswith("-chunks.json"):
            manifest = json.loads(Path(ref["path"]).read_text())
            digest = hashlib.sha256()
            size = 0
            for chunk in manifest["chunks"]:
                blob = checked(chunk).read_bytes()
                digest.update(blob)
                size += len(blob)
            if (
                "sha256:" + digest.hexdigest() != manifest["original"]["sha256"]
                or size != manifest["byte_count"]
            ):
                raise ValueError("historical_chunk_drift")
    raw = Path(value["terminal_validation_sidecar_path"]).parent
    plan, work, validation = [
        json.loads((raw / p).read_text()) for p in ["methods.json", "work.json", "validation.json"]
    ]
    if plan["methods"] != METHODS:
        raise ValueError("methods_drift")
    for role, ref in plan["role_manifests"].items():
        if json.loads(checked(ref).read_text())["rows"] != plan["roles"][role]:
            raise ValueError("role_drift")
    for row in plan["rows"]:
        original = plan["roles"][row["role"]][row["original_slot"]]
        if (
            row["source_cluster_id"] != normalized(bytes.fromhex(original["source_bytes"]))
            or row["eligible"] != original["public_eligible"]
        ):
            raise ValueError("primitive_row_drift")
    for receipt in validation["receipts"]:
        checked(dict(path=str(ROOT / receipt["log_path"]), sha256=receipt["log_sha256"]))
    if build(plan, work, raw, validation) != value:
        raise ValueError("cold_reduction_drift")


def terminal(path: Path) -> Json:
    """A fresh process and shared terminal readers inspect exact publication bytes."""
    raw = Path(json.loads(path.read_text())["terminal_validation_sidecar_path"]).parent
    cold = run_commands(
        ROOT,
        [
            CommandSpec(
                "cold_reduction",
                (str(ROOT / ".venv/bin/python"), "-u", str(ROOT / CLI), "--cold-replay", str(path)),
                "terminal",
                60,
            )
        ],
        log_dir=raw / "terminal_logs" / path.name / "cold",
        heartbeat_s=20,
    )
    result = venue.terminal_readers(path)
    return dict(
        passed=all(r["passed"] for r in cold) and result["passed"],
        receipts=cold + result["receipts"],
    )


def main(argv: list[str] | None = None) -> int:
    """Seal once and publish only checked bytes; private runs cannot earn readiness."""
    started = time.monotonic()
    prior.progress("8046_start_preconditions")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261003"], default="20261003")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument(
        "--seal-only", action="store_true", help="private custody route; readiness remains zero"
    )
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            replay(json.loads(args.cold_replay.read_text()))
            prior.progress("8046_cold_reduction_passed")
            return 0
        output = args.output.absolute()
        raw = output.parent / "raw" / output.stem
        if (raw / "methods.json").exists():
            raise ValueError("original_work_preserved_use_cold_replay")
        raw.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="carnot-8046-") as temporary:
            scratch = Path(temporary)
            specs = commands(scratch)
            atomic_json(
                raw / "validation_commands.json",
                dict(
                    commands=[asdict(s) for s in specs],
                    private_seal_only=args.seal_only,
                    coverage_configuration=(scratch / "coverage.ini").read_text()
                    if (scratch / "coverage.ini").is_file()
                    else "private unit command fixture",
                ),
            )
            plan = seal(args.root, raw)
            prior.progress("8046_before_owned_validation", len(plan["rows"]), len(specs))
            validation = (
                dict(receipts=[], coverage={}) if args.seal_only else validate(scratch, raw)
            )
            prior.progress("8046_after_owned_validation", len(validation["receipts"]), 0)
            duration = time.monotonic() - started
            work = dict(
                duration_s=duration,
                phase_spans=[
                    dict(name="preconditions_seal_and_owned_validation", duration_s=duration)
                ],
                acceptance_manifest=[]
                if args.seal_only
                else [s.name for s in specs if s.scope == "owned"],
            )
            atomic_json(raw / "work.json", work)
            atomic_json(raw / "validation.json", validation)
            value = build(plan, work, raw, validation)
            prior.progress("8046_before_publication", len(value["rows"]), 1)
            publication = publish_primary(output, value, terminal)
            post = terminal(output)
            if not post["passed"]:
                raise ValueError("published_validation")
            atomic_json(
                raw / "terminal_validation.json",
                dict(publication=publication, published=post, owned_invocation_exit=0),
            )
            prior.progress("8046_complete", len(value["rows"]), 0)
        return 0
    except (ValueError, KeyError, OSError) as error:
        prior.progress("8046_rejected_" + str(error))
        return 1
