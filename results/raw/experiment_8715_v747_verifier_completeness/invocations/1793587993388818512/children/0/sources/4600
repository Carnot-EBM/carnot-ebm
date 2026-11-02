"""REQ-REPORT-8234: authenticate current methods without claiming measured benefit.

Frozen source bytes preserve the native probabilities and original role geometry.
Authority agreement and executable methods remain separate readiness operands.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Callable
import json
import re
from pathlib import Path
from typing import Any

import yaml
import numpy as np

from carnot.reporting import v685_authority_lifecycle as authority
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, canonical_hash
from carnot.reporting.v710_contract_replay import snapshot, failure as qualified_failure
from carnot.verify import decision_margin_8234 as n
from carnot.verify import evidence_energy_8154 as base
from carnot.verify import utility_patch_methods_8219 as historical

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8234_v712_decision_margin_methods"
TASK = "exp8234-decision-margin-methods"
MILESTONE = "2026.10.712"
CLI = "scripts/experiments/" + NAME + ".py"
TEST = "tests/python/test_decision_margin_methods_8234.py"
OWNED = [
    "python/carnot/verify/decision_margin_8234.py",
    "python/carnot/reporting/decision_margin_methods_8234.py",
    "python/carnot/reporting/decision_margin_runner_8234.py",
    CLI,
]
PROTOCOL = "openspec/change-proposals/v712-decision-margin-protocol.json"
PIN = "sha256:aeb0fb6b93214772e4bb5f3acc44b9617d9dd5ef8de188bf6384296a0f0689ab"
PROTOCOL_VALUE: Json = json.loads((ROOT / PROTOCOL).read_bytes())
MODEL_SPECS: list[Json] = []
DESIGN = "openspec/change-proposals/research-roadmap-vNEXT.md"
ACTIVE = "research-roadmap.yaml"
STAGED = "research-roadmap-next.yaml"
HISTORY = "openspec/change-proposals/research-roadmap-v711-preserved-20261007.md"
INPUTS = [
    "AGENTS.md",
    "CODEX.md",
    "CLAUDE.md",
    "ops/e2e-test-plan.md",
    "openspec/capabilities/research-reporting/spec.md",
    "openspec/capabilities/verification/spec.md",
    "scripts/experiment_template.py",
    "python/carnot/reporting/primary_publication.py",
    "ops/exclusion_manifest.yaml",
    "python/carnot/reporting/roadmap_contract.py",
    "python/carnot/reporting/v711_current_contract.py",
    "python/carnot/verify/utility_patch_methods_8219.py",
    "research-references.md",
    DESIGN,
    ACTIVE,
    HISTORY,
    PROTOCOL,
]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase counts so a parent can distinguish work from a silent stall."""
    print(f"[exp8234] phase={phase} completed={completed} pending={pending}", flush=True)


def failure(
    path: Path, field: str, expected: Any, observed: Any, digest: str | None = None
) -> Json:
    """Name the unavailable source so an external block can be checked directly."""
    return dict(qualified_failure(path, field, expected, observed, digest), upstream_id=path.name)


def primitives(read: Callable[[Json], Json], protocol: Json) -> list[Json]:
    """Join only original source identities; reserved targets stay closed."""
    evidence = read(protocol["fit_measurement"])["evidence"]
    native = {r["unit_id"]: r for r in read(protocol["native_fit_rows"])["rows"]}
    reserved = {r["unit_id"]: r for r in read(protocol["reserved_primary"])["feature_rows"]}
    rp = {
        r["unit_id"]: r for r in read(protocol["native_reserved_measurement"])["plan"]["baseline"]
    }
    sealed = read(protocol["sealed_predictions"])
    if sealed["labels_opened"] or evidence["roles"] != protocol["role_manifest"]:
        raise ValueError("source_schema")
    n.validate_roles(protocol["role_manifest"])
    accepted = {r["unit_id"]: r["baseline_action"] for r in sealed["rows"] if r["arm"] == "energy"}
    fit = {r["unit_id"]: r for r in evidence["rows"]}
    if (
        len(fit) != 192
        or len(reserved) != 128
        or len(accepted) != 128
        or set(fit)
        != {
            r["unit_id"]
            for k in ["head_fit", "temperature_fit", "calibration"]
            for r in protocol["role_manifest"][k]
        }
    ):
        raise ValueError("source_schema")
    rows = []
    for role, slots in protocol["role_manifest"].items():
        for slot in slots:
            uid = slot["unit_id"]
            row = reserved[uid] if role == "reserved" else fit[uid]
            paired = rp[uid] if role == "reserved" else native[uid]
            if row["source_cluster_id"] != slot["source_cluster_id"]:
                raise ValueError("source_schema")
            if paired["source_cluster_id"] != slot["source_cluster_id"]:
                raise ValueError("source_schema")
            x, p0 = row["x"], paired["holistic_probability"]
            if x is not None and (len(x) != 16 or not np.isfinite(x).all()):
                raise ValueError("source_schema")
            permission = (
                accepted[uid]
                if role == "reserved"
                else base.action(historical.baseline_probability(row, evidence["baseline"]))
            )
            weight = n.margin_weight(p0, permission) if x is not None else None
            rows.append(
                dict(
                    unit_id=uid,
                    source_cluster_id=slot["source_cluster_id"],
                    role=role,
                    x=x,
                    p0=p0,
                    baseline_action=permission,
                    weight=weight,
                    y=None if role == "reserved" else row["y"],
                    status="completed" if weight is not None else "missing",
                    exposure_scope="exposed_development",
                )
            )
    return rows


def measure(root: Path, raw: Path) -> Json:
    """Authenticate each operand before reducing it; missing inputs remain blocked."""
    progress("preconditions_before", 0, 14)
    work: Json = dict(
        protocol=deepcopy(PROTOCOL_VALUE),
        refs=[],
        code=[],
        failures=[],
        public_rows=[],
        contract={},
        checks=[],
        history=[],
    )
    raw.mkdir(parents=True, exist_ok=True)
    copies = {}
    specs = [dict(path=str(ROOT / PROTOCOL), sha256=PIN), *PROTOCOL_VALUE["source_artifact_hashes"]]
    for i, spec in enumerate(specs):
        path = root / Path(spec["path"]).relative_to(ROOT)
        ref = snapshot(path, raw / "inputs", "input")
        work["refs"].append(ref)
        if ref["sha256"] != spec["sha256"]:
            work["failures"].append(
                failure(path, "sha256", spec["sha256"], ref["sha256"], ref["sha256"])
            )
        else:
            copies[spec["path"]] = Path(ref["snapshot_path"])
        progress("authenticated_inputs", i + 1, len(specs) - i - 1)
    for name in INPUTS:
        ref = snapshot(root / name, raw / "context", "context")
        work["refs"].append(ref)
        if not ref["exists"] and name != ACTIVE:
            work["failures"].append(failure(root / name, "exists", True, None))
    work["code"] = [snapshot(ROOT / p, raw / "code", "code") for p in OWNED]
    try:
        work["contract"] = authority.assess_authorities(
            root / DESIGN,
            root / STAGED,
            root / ACTIVE,
            raw / "authority",
            milestone=MILESTONE,
            first_id=8234,
            count=14,
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError, yaml.YAMLError) as error:
        work["contract"] = dict(
            activated=False,
            planning_matched=False,
            contract_rows=[],
            canonical_tasks_sha256=None,
            authority_snapshots={},
            gate_check_summary=[failure(root / DESIGN, "authority_readable", True, str(error))],
        )
    if not work["failures"]:
        try:
            read: Callable[[Json], Json] = lambda ref: json.loads(copies[ref["path"]].read_bytes())
            work["public_rows"] = primitives(read, work["protocol"])
            global_model = read(work["protocol"]["v711_energy_global_measurement"])["models"][
                work["protocol"]["v711_energy_global_model_key"]
            ]
            if global_model["kind"] != "patch" or global_model["base"]["kind"] != "input":
                raise ValueError("energy_global_schema")
            work["frozen_v711_energy_global_sha256"] = canonical_hash(global_model)
            for history_path in [
                "results/experiment_8224_v711_utility_audit.json",
                "results/experiment_8233_v711_capstone.json",
            ]:
                v = read(dict(path=str(ROOT / history_path)))
                work["history"].append(
                    dict(
                        path=history_path,
                        honest_verdict=v["honest_verdict"],
                        verdict_class=v["verdict_class"],
                        imported=True,
                    )
                )
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            work["failures"].append(
                failure(root / PROTOCOL, "source_schema", True, str(error), PIN)
            )
    progress("preconditions_after", 14, 0)
    return work


def reduce(work: Json, receipts: list[Json]) -> Json:
    """Owned failures zero readiness; completed method controls grant no benefit."""
    owned_ok = bool(receipts) and all(r["passed"] for r in receipts)
    method_ok = not work["failures"] and len(work["public_rows"]) == 320
    contract_ok = work["contract"]["activated"]
    failures = [*work["failures"], *work["contract"]["gate_check_summary"]]
    verdict = "disqualified" if not owned_ok else "blocked" if failures else "null"
    suffix = "required_checks" if not owned_ok else "decision_margin_methods"
    if verdict == "blocked":
        operand = failures[0]
        suffix = re.sub(
            "[^a-z0-9_]",
            "_",
            (
                Path(operand.get("path", operand.get("artifact_path"))).stem
                + "_"
                + str(operand["artifact_field"])
            ).lower(),
        )
    rows = []
    source = work["contract"]["contract_rows"]
    for i in range(14):
        r = source[i] if i < len(source) else {}
        checks = r.get("checks", {})
        missing = not r or r["status"] == "unstarted"
        rows.append(
            dict(
                unit_id=f"exp{8234 + i}",
                source_cluster_id="V712_authority",
                arm="current_contract",
                condition="complete_task_agreement",
                seed=None,
                status="censored" if missing else "completed",
                missing_status=missing,
                completed=not missing,
                failed=not missing and not all(checks.values()),
                censored=missing,
                excluded=False,
                checks=checks,
                metric="contract_agreement",
                numerator=None if missing else sum(checks.values()),
                denominator=len(checks),
                effective_independent_groups=0,
            )
        )
    p = work["protocol"]
    return dict(
        experiment_id=8234,
        task_id=TASK,
        milestone=MILESTONE,
        run_date="20261007",
        schema="carnot.v712.decision-margin-methods.v1",
        honest_verdict="complete_" + verdict + "_" + suffix,
        verdict_class=verdict,
        gate_check_summary=failures,
        rows=rows,
        intended_count=14,
        completed_count=sum(r["completed"] for r in rows),
        failed_count=sum(r["failed"] for r in rows),
        censored_count=sum(r["censored"] for r in rows),
        excluded_count=0,
        independent_count=0,
        verifier_is_oracle=False,
        exposure_scope=p["exposure_scope"],
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        call_ledger=[],
        trained_head_specs=[
            dict(arm=a, coefficients=17, generator=False, fitted_here=False) for a in p["arms"]
        ],
        required_checks_passed=owned_ok,
        flagged_adversarial=False,
        acceptance_gates=dict(
            owned_checks=dict(passed=owned_ok),
            contract=dict(passed=contract_ok),
            margin_protocol=dict(passed=method_ok),
        ),
        validation_receipts=receipts,
        preconditions_checked=[
            dict(path=r["path"], sha256=r["sha256"], exists=r["exists"]) for r in work["refs"]
        ],
        source_artifact_hashes=work["refs"],
        cited_upstream_artifacts=work["refs"],
        code_config_hashes=work["code"],
        current_code_snapshots=work["code"],
        current_contract_ready_score=int(owned_ok and contract_ok),
        margin_protocol_ready_score=int(owned_ok and method_ok),
        protocol_path=str(ROOT / PROTOCOL),
        protocol_sha256=PIN,
        objective_equations=p["objective_equations"],
        frozen_roles=p["role_manifest"],
        canonical_tasks_sha256=work["contract"]["canonical_tasks_sha256"],
        staged_readiness=work["contract"]["planning_matched"],
        activated_readiness=contract_ok,
        authority_snapshots=work["contract"]["authority_snapshots"],
        original_reserved_mask=[r for r in work["public_rows"] if r["role"] == "reserved"],
        native_margin_weight_rows=[
            {k: r[k] for k in ["unit_id", "role", "p0", "baseline_action", "weight", "status"]}
            for r in work["public_rows"]
        ],
        H1=dict(p["H1"], measured_here=False),
        H2=dict(p["H2"], measured_here=False),
        historical_dispositions=work["history"],
        frozen_v711_energy_global_sha256=work.get("frozen_v711_energy_global_sha256"),
        historical_generator_provenance="cached Qwen calls only",
        scientific_benefit_measured=False,
        external_publication_authorized=False,
        methodology_note="Original native Qwen probabilities set public action-margin weights. "
        "Six matched small-head objectives and calibration-only comparator rules are frozen before fitting. "
        "Only cached reductions run here. Missing original slots remain missing; exposed development "
        "sources cannot establish independent generalization. Methods readiness is not observed benefit.",
    )
