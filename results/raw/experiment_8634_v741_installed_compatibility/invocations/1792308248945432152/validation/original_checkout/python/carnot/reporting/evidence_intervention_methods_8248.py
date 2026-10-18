"""REQ-REPORT-8248: source custody and protocol readiness grant no benefit.

This task prepares views and freezes methods. Later tasks acquire predictions,
fit the heads and test the hypotheses against independent human targets.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import yaml

from carnot.reporting import decision_margin_methods_8234 as inherited
from carnot.reporting import v685_authority_lifecycle as authority
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v710_contract_replay import snapshot, failure
from carnot.verify import evidence_intervention_8248 as n
from carnot.verify.sentence_transport_methods_8179 import tokenizer

Json = dict[str, Any]
ROOT = inherited.ROOT
NAME = "experiment_8248_v713_evidence_intervention_methods"
TASK = "exp8248-evidence-intervention-methods"
MILESTONE = "2026.10.713"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_evidence_intervention_methods_8248.py"
OWNED = [
    "python/carnot/verify/evidence_intervention_8248.py",
    "python/carnot/reporting/evidence_intervention_methods_8248.py",
    "python/carnot/reporting/evidence_intervention_runner_8248.py",
    CLI,
]
PROTOCOL = "openspec/change-proposals/v713-evidence-intervention-protocol.json"
PIN = "sha256:f4c06e17a9f8eb72f1be80b265ab7e3507cbc5f6f3b918df6421fe49dae02018"
PROTOCOL_VALUE: Json = json.loads((ROOT / PROTOCOL).read_bytes())
MODEL_SPECS: list[Json] = []
DESIGN, ACTIVE, STAGED = inherited.DESIGN, inherited.ACTIVE, inherited.STAGED
INPUTS = [p for p in inherited.INPUTS if p not in [inherited.PROTOCOL, inherited.HISTORY]] + [
    PROTOCOL,
    "python/carnot/reporting/decision_margin_methods_8234.py",
    "python/carnot/verify/sentence_transport_8179.py",
    "python/carnot/verify/sentence_energy_8183.py",
]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush source counts so supervision can distinguish progress from a stall."""
    print(f"[exp8248] phase={phase} completed={completed} pending={pending}", flush=True)


def primitives(
    read: Callable[[Json], Json], protocol: Json, count: Callable[[str], int]
) -> list[Json]:
    """Read public source fields only; human targets cannot select an intervention."""
    n.validate_roles(protocol)
    get: Callable[[str], Json] = lambda key: read(dict(path=protocol[key]))
    slots = get("fit_slots_reference")["slots"] + get("evaluation_slots_reference")["plan"]["slots"]
    primaries = [get("fit_primary"), get("evaluation_primary")]
    cached: Json = {}
    features: Json = {}
    baseline: Json = {}
    for primary in primaries:
        for r in primary["sentence_rows"]:
            cached.setdefault(r["unit_id"], []).append(r)
        features.update({r["unit_id"]: r["x"] for r in primary["feature_rows"]})
        baseline.update(
            {
                r["unit_id"]: dict(p=r["p"], action=r["action"])
                for r in primary.get("prediction_rows", [])
                if r["arm"] == "local_evidence_radial16"
            }
        )
    baseline.update(
        {
            r["unit_id"]: dict(p=r["p"], action=r["action"])
            for r in get("fit_predictions_primary")["rows"]
            if r["arm"] == "local_evidence_radial16"
        }
    )
    lookup = {r["unit_id"]: r for r in slots}
    if len(lookup) != 320:
        raise ValueError("source_schema")
    rows = []
    for role, members in protocol["role_manifest"].items():
        for member in members:
            slot = lookup[member["unit_id"]]
            digest = "sha256:" + hashlib.sha256(bytes.fromhex(slot["source_bytes"])).hexdigest()
            if (
                digest != member["original_source_sha256"]
                or slot["source_cluster_id"] != member["source_cluster_id"]
            ):
                raise ValueError("source_identity")
            intervention = n.view(slot, cached.get(member["unit_id"], []), count)
            rows.append(
                dict(
                    member,
                    role=role,
                    historical_x=features.get(member["unit_id"]),
                    frozen_v707=baseline.get(member["unit_id"], dict(p=None, action="escalate")),
                    intervention=intervention,
                    exposure_scope="exposed_development",
                )
            )
            progress("views_frozen", len(rows), 320 - len(rows))
    return rows


def sealed_labels(read: Callable[[Json], Json], protocol: Json) -> list[Json]:
    """Keep fit and tune labels apart from public inputs; never open retention targets."""
    original = read(dict(path=protocol["historical_fit_reference"]))["evidence"]["rows"]
    lookup = {r["unit_id"]: r for r in original}
    return [
        dict(member, role=role, y=lookup[member["unit_id"]]["y"])
        for role in ["fit", "calibration", "selection"]
        for member in protocol["role_manifest"][role]
    ]


def measure(root: Path, raw: Path) -> Json:
    """Authenticate originals and terminal reports before preparing any source view."""
    progress("preconditions_before", 0, 14)
    work: Json = dict(
        protocol=deepcopy(PROTOCOL_VALUE),
        refs=[],
        code=[],
        failures=[],
        public_rows=[],
        contract={},
        history=[],
        tokenizer_receipt={},
        label_reference=None,
    )
    copies: Json = {}
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
    for name in ["python", "pytest", "coverage", "ruff", "mypy"]:
        tool = ROOT / ".venv/bin" / name
        if not tool.is_file():
            work["failures"].append(failure(tool, "available", True, None))
    reused = [
        *inherited.OWNED,
        "python/carnot/reporting/v709_execution.py",
        "python/carnot/reporting/v711_current_runner.py",
        "python/carnot/reporting/v685_authority_lifecycle.py",
        "python/carnot/reporting/roadmap_contract.py",
        "python/carnot/verify/sentence_transport_methods_8179.py",
        "scripts/experiment_template.py",
    ]
    work["code"] = [snapshot(ROOT / name, raw / "code", "code") for name in [*OWNED, TEST, *reused]]
    try:
        work["contract"] = authority.assess_authorities(
            root / DESIGN,
            root / STAGED,
            root / ACTIVE,
            raw / "authority",
            milestone=MILESTONE,
            first_id=8248,
            count=14,
        )
        _, work["contract"]["tasks"] = parse_design(
            (root / DESIGN).read_text(), milestone=MILESTONE
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError, yaml.YAMLError) as error:
        work["contract"] = dict(
            activated=False,
            planning_matched=False,
            contract_rows=[],
            tasks=[],
            canonical_tasks_sha256=None,
            authority_snapshots={},
            gate_check_summary=[failure(root / DESIGN, "authority_readable", True, str(error))],
        )
    if not work["failures"]:
        try:
            read: Callable[[Json], Json] = lambda ref: json.loads(copies[ref["path"]].read_bytes())
            for binding in PROTOCOL_VALUE["terminal_bindings"]:
                primary = root / Path(binding["primary"]).relative_to(ROOT)
                sidecar = root / Path(binding["sidecar"]).relative_to(ROOT)
                terminal = read(dict(path=binding["terminal"]))
                report = read_bound_sidecar(primary, sidecar)
                if report["report"]["passed"] is not True or terminal["publication"][
                    "primary_sha256"
                ] != sha256_file(primary):
                    raise ValueError("qualified_terminal")
            plan: Json = dict(checks=[])
            count, work["tokenizer_receipt"] = tokenizer(plan, fixture=root != ROOT)
            work["failures"].extend(r for r in plan["checks"] if not r["passed"])
            if not work["failures"]:
                progress("source_views_before", 0, 320)
                work["public_rows"] = primitives(read, work["protocol"], count)
                labels = raw / "sealed_fit_tune_labels.json"
                atomic_json(labels, sealed_labels(read, work["protocol"]))
                work["label_reference"] = dict(
                    path=str(labels),
                    sha256=sha256_file(labels),
                    target_roles=["fit", "calibration", "selection"],
                    retention_targets_opened=False,
                )
                progress("source_views_after", 320, 0)
            for name in [
                "results/experiment_8239_v712_margin_decision_audit.json",
                "results/experiment_8241_v712_delayed_benefit_audit.json",
                "results/experiment_8247_v712_capstone.json",
            ]:
                value = read(dict(path=str(ROOT / name)))
                work["history"].append(
                    {
                        k: value[k]
                        for k in [
                            "experiment_id",
                            "honest_verdict",
                            "verdict_class",
                            "gate_check_summary",
                            "H1",
                            "H2",
                            "task_dispositions",
                        ]
                        if k in value
                    }
                )
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            work["failures"].append(
                failure(root / PROTOCOL, "source_schema", True, str(error), PIN)
            )
    progress("preconditions_after", 14, 0)
    return work


def reduce(work: Json, receipts: list[Json]) -> Json:
    """Owned failure and external absence remain distinct terminal dispositions."""
    owned_ok = bool(receipts) and all(r["passed"] for r in receipts)
    method_ok = not work["failures"] and len(work["public_rows"]) == 320
    contract = work["contract"]
    contract_ok = contract["activated"]
    failures = [*work["failures"], *contract["gate_check_summary"]]
    verdict = "disqualified" if not owned_ok else "blocked" if failures else "null"
    suffix = "required_checks" if not owned_ok else "evidence_intervention_methods"
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
    for i in range(14):
        primitive = contract["contract_rows"][i] if i < len(contract["contract_rows"]) else {}
        checks = primitive.get("checks", {})
        missing = not primitive or primitive["status"] == "unstarted"
        rows.append(
            dict(
                unit_id=f"exp{8248 + i}",
                source_cluster_id="V713_authority",
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
        experiment_id=8248,
        task_id=TASK,
        milestone=MILESTONE,
        run_date="20261007",
        schema="carnot.v713.evidence-intervention-methods.v1",
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
        exposure_scope="exposed_development",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        call_ledger=[],
        trained_head_specs=[
            dict(h, generator=False, fitted_here=False) for h in p["trained_head_specs"]
        ],
        required_checks_passed=owned_ok,
        flagged_adversarial=False,
        acceptance_gates=dict(
            owned_checks=dict(passed=owned_ok),
            contract=dict(passed=contract_ok),
            intervention_protocol=dict(passed=method_ok),
        ),
        validation_receipts=receipts,
        preconditions_checked=[
            dict(path=r["path"], sha256=r["sha256"], exists=r["exists"]) for r in work["refs"]
        ],
        source_artifact_hashes=work["refs"],
        cited_upstream_artifacts=[
            dict(
                r,
                fields_imported=next(
                    (
                        s.get("fields_imported", [])
                        for s in p["source_artifact_hashes"]
                        if Path(s["path"]).name == Path(r["path"]).name
                    ),
                    ["context bytes"],
                ),
            )
            for r in work["refs"]
        ],
        code_config_hashes=work["code"],
        current_code_snapshots=work["code"],
        current_contract_ready_score=int(owned_ok and method_ok and contract_ok),
        intervention_protocol_ready_score=int(owned_ok and method_ok),
        protocol_path=str(ROOT / PROTOCOL),
        protocol_sha256=PIN,
        objective_equations=p["objective_equations"],
        canonical_tasks_sha256=contract["canonical_tasks_sha256"],
        task_contract=contract.get("tasks", []),
        staged_readiness=contract["planning_matched"],
        activated_readiness=contract_ok,
        authority_snapshots=contract["authority_snapshots"],
        frozen_roles=p["role_manifest"],
        role_manifest_hashes={
            role: canonical_hash(members) for role, members in p["role_manifest"].items()
        },
        original_role_counts={k: len(v) for k, v in p["original_roles"].items()},
        source_view_rows=work["public_rows"],
        current_intervention_predictions_count=0,
        tokenizer_receipt=work["tokenizer_receipt"],
        fixture_protocol_only=work["tokenizer_receipt"].get("status") == "fixture_byte_counter",
        H1=dict(p["H1"], measured_here=False),
        H2=dict(p["H2"], measured_here=False),
        feature_definitions=p["feature_definitions"],
        control_matching_rule=p["control_matching_rule"],
        intervention_scope=p["intervention_scope"],
        continuous_mechanism=p["continuous_mechanism"],
        label_seal=p["label_seal"],
        label_reference=work["label_reference"],
        historical_dispositions=work["history"],
        historical_generator_provenance="cached V707 Qwen sentence predictions; zero current LLM calls",
        scientific_benefit_measured=False,
        external_publication_authorized=False,
        methodology_note="Authenticate original source bytes and qualified terminal reports before freezing complete sentence deletions. "
        "Lexical selection and token-length matching do not use human targets or citations. "
        "Five added model-prediction features and equal-information controls precede capture. "
        "All original slots and missing views remain visible. Both hypotheses concern exposed development only. "
        "No heads fit and no current model predictions are acquired here. Readiness does not imply scientific benefit.",
    )
