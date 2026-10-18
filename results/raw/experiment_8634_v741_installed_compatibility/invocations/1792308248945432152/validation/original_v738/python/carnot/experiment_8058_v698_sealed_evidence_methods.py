"""REQ-REPORT-8058: freeze development methods before current outcome access.

Readiness qualifies a protocol, not a result. Public identities and historical
eligibility remain separate from the evaluator labels consumed by later tasks.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

from carnot.experiment_8046_v697_branch_protocols import METHODS, ROLE_COUNTS
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.evidence_features_custody_7980 import reference
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar
from carnot.reporting.v686_contract_validation import CONSUMERS, run_check
from carnot.verify.evidence_features_7980 import normalized

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_8058_v698_sealed_evidence_methods"
TASK = "exp8058-sealed-evidence-methods"
CLI = f"scripts/experiments/{NAME}.py"
MODULE = f"python/carnot/{NAME}.py"
TEST = "tests/python/test_sealed_evidence_methods_8058.py"
OWNED = [MODULE, CLI]
INPUTS = [
    "AGENTS.md",
    "CODEX.md",
    "CLAUDE.md",
    "ops/e2e-test-plan.md",
    "openspec/capabilities/research-reporting/spec.md",
    "scripts/experiment_template.py",
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/primary_publication.py",
    "ops/exclusion_manifest.yaml",
    "python/carnot/experiment_8046_v697_branch_protocols.py",
    "python/carnot/experiment_8019_v695_eligible_targets.py",
    "python/carnot/verify/evidence_features_7980.py",
    "python/carnot/verify/learning_benefit_8052.py",
    "research-references.md",
    "openspec/change-proposals/research-roadmap-vNEXT.md",
]
PARENTS = {
    "experiment_8046_v697_branch_protocols": "sha256:948372dc5860c4a41754bf3c7b0b9b13463a916e4aa2317a88fd19f197a28a42",
    "experiment_8019_v695_eligible_targets": "sha256:310fbb2c5d0b254ceb2e00cd742c1d83c856ff4082075b7cd61e592c940ea518",
}
START = time.monotonic()


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real counts so an idle child cannot look like completed work."""
    print(
        f"[exp8058] phase={phase} elapsed_s={time.monotonic() - START:.3f} completed={completed} pending={pending}",
        flush=True,
    )


def partition(source_id: str) -> int:
    """Hash literal UTF-8 identity bytes before labels can influence roles."""
    return int(hashlib.sha256(source_id.encode()).hexdigest(), 16) % 4


def methods() -> Json:
    """Copy historical source controls and explicitly replace the learning contract."""
    v = deepcopy(METHODS)
    v["learning"] = dict(
        roles=dict(stream=256, retention=64),
        delay=20,
        partition="sha256(source_id UTF-8 bytes) integer modulo4; bucket0 admission-only; others update-only",
        source_id="normalized original source_cluster_id",
        arms=["frozen", "unconditional", "reused_guard", "fresh_admission"],
        attempt_slots=[64, 128, 192],
        updates_per_block=4,
        step=0.01,
        ridge=0.001,
        newest_update_rows=32,
        minimum_update_rows=16,
        update_per_class=2,
        admission_rows=12,
        guard_per_class=2,
        guard_gradients=False,
        cap_steps=12,
        seeds=list(range(101, 121)),
        numerical_budget_s=1200,
        candidate="four full-batch gradients from each arm own incumbent on its newest released update rows; commit candidate and incumbent hashes first",
        admission="after commitment select next12 eligible admission-only rows with release_slot>commit_slot, stable release order; never reuse fresh rows",
        matched_clocks="all four arms wait to same block last release; block must finish strictly before next attempt or stream end256",
        insufficient="defer without relaxation for missing rows or <2/class; no alpha search without adequate guard evidence",
        reused="all eligible admission-role rows released at commitment, immutable snapshot; may reuse across attempts",
        unconditional="commit alpha1 at matched decision time; frozen never writes",
        state="separate incumbent, candidate, optimizer and checkpoints for each arm/seed",
        durable_issue_first=True,
        terminal_flush=False,
        unknown_targets="exclude, retain original timeline slots and denominators",
        retention_access="evaluator only after all final head hashes sealed",
        event_order=[
            "prediction_issue",
            "candidate_commit",
            "block_select",
            "label_release",
            "admission_consume",
            "durable_state_commit",
        ],
        outstanding_feedback="issued minus released labels at every original slot; persist pending/censored candidate blocks",
    )
    v["guard"] = dict(
        alphas=[1, 0.5, 0.25, 0.125, 0],
        tolerance=0,
        versus_initial=dict(brier=0.01, typed_cost=0.02),
        versus_incumbent=dict(brier=0, typed_cost=0),
        false_accepts="no new row false accepts versus either initial or incumbent",
        selection="largest passing alpha; if none preserve incumbent, never reset initial",
        guarantee="empirical only; paired-binomial and bounded-loss bounds diagnostic; dependent exposed batches have no iid safety certificate",
    )
    v["hypotheses"][2]["comparison"] = "fresh_admission versus reused_guard"
    v["safety"].update(
        later_support=[80, 10],
        H3_per_seed_false_accept_baselines=["reused_guard", "unconditional", "frozen"],
        H3_noninferiority_baselines=["unconditional", "frozen"],
        H3_noninferiority_cost_margin=0.02,
        retention_arms=["reused_guard", "fresh_admission"],
        retention_baseline="frozen",
        later_slots="common eligible update-role source slots after first shared completed admission opportunity, prediction before own-label release",
    )
    v["timeline_masks"] = dict(
        domain="all original256 slots, no compression",
        masks=[
            "public_eligible",
            "complete_target_eligible",
            "update_role",
            "admission_role",
            "released",
            "pending",
            "censored",
            "after_shared_first_decision",
            "prediction_before_own_release",
        ],
        seeds="101-120 averaged within each source before resampling original timeline",
        retention="original64 identities after final head seals",
    )
    v["method_source_map"] = [
        dict(
            id="2607.04223",
            url="https://arxiv.org/abs/2607.04223",
            adopted="fixed-answer source removal, separated duplicates, complete independent human targets",
            deferred="new detector or answer generation",
            inapplicable="fixture repeatability does not prove source utility",
        ),
        dict(
            id="2609.10873",
            url="https://arxiv.org/html/2609.10873v1",
            adopted="committed candidates, one-use fresh admission, pool-level missed opportunities",
            deferred="formal certification until independent support exists",
            inapplicable="iid paired-binomial theorem for dependent historically exposed12-row blocks; paper closed-loop diagnostic favored unconditional replay",
        ),
        dict(
            id="2602.02634",
            url="https://arxiv.org/abs/2602.02634",
            adopted="issue, commitment, release, consumption and outstanding-feedback accounting",
            deferred="full delayed convex reduction",
            inapplicable="convex regret guarantee for typed discontinuous decision safety",
        ),
        dict(
            id="2511.12828",
            url="https://arxiv.org/abs/2511.12828",
            adopted="explicit retention and changed-decision tests",
            deferred="generator fine-tuning",
            inapplicable="local spline support as a forgetting guarantee",
        ),
    ]
    return v


def seal(root: Path, raw: Path, *, mutate: bool = False) -> Json:
    """Bind public roles first and use only historical eligibility metadata afterward."""
    progress("preconditions_before")
    raw.mkdir(parents=True, exist_ok=True)
    plan: Json = dict(
        rows=[],
        role_manifests={},
        source_artifact_hashes=[],
        failures=[],
        source_valid=False,
        learning_valid=False,
        outcome_access_ledger=[],
        methods=methods(),
        qualified_head={},
        repository_health=[],
        validation_manifest=[],
    )

    def require(path: Path, field: str, expected: Any, observed: Any, branch: str = "both") -> bool:
        if expected == observed:
            return True
        plan["failures"].append(
            dict(
                check=field,
                upstream=path.stem,
                path=str(path.absolute()),
                hash=sha256_file(path) if path.is_file() else None,
                field=field,
                op="==",
                expected=expected,
                observed=observed,
                passed=False,
                branch=branch,
            )
        )
        return False

    def bind(path: Path, expected: str | None = None) -> Path | None:
        if not require(path, "resource_exists", True, path.is_file()):
            return None
        ref = reference(path)
        if expected and not require(path, "sha256", expected, ref["sha256"]):
            return None
        snapshot = raw / "inputs" / (ref["sha256"][7:] + path.suffix)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
        plan["source_artifact_hashes"].append(dict(ref, snapshot_path=str(snapshot)))
        return snapshot

    for relative in INPUTS:
        bind(root / relative)
    for tool in ["python", "pytest", "coverage", "ruff", "mypy"]:
        require(
            ROOT / ".venv/bin" / tool,
            "resource_exists",
            True,
            (ROOT / ".venv/bin" / tool).is_file(),
        )
    require(Path(sys.executable), "python_version>=3.11", True, sys.version_info >= (3, 11))
    parents = {}
    for name, digest in PARENTS.items():
        path = root / "results" / (name + ".json")
        snapshot = bind(path, digest)
        if snapshot is None:
            continue
        value = json.loads(snapshot.read_text())
        require(
            path, "flagged_adversarial", False, value.get("flagged_adversarial", "MISSING_FIELD")
        )
        sidecar = Path(value.get("terminal_validation_sidecar_path", "missing_sidecar"))
        side_snapshot = bind(sidecar)
        if side_snapshot is None:
            continue
        publication = json.loads(side_snapshot.read_text())
        publication = publication.get("publication", publication)
        report_path = Path(publication["sidecar_path"])
        bound = bind(report_path)
        if bound is None:
            continue
        try:
            report = read_bound_sidecar(path, report_path)
        except ValueError as error:
            require(
                report_path,
                "primary_binding",
                "authenticated primary bytes and location",
                str(error),
            )
            continue
        require(report_path, "primary_path", str(path.absolute()), report.get("primary_path"))
        require(report_path, "report.passed", True, report["report"]["passed"])
        require(sidecar, "primary_sha256", digest, publication.get("primary_sha256"))
        parents[name] = value
    progress("preconditions_after", len(plan["source_artifact_hashes"]), 0)
    if len(parents) != 2:
        return plan
    upstream = parents["experiment_8046_v697_branch_protocols"]
    require(
        root / "results" / (next(iter(PARENTS)) + ".json"),
        "source_protocol_ready_score",
        1,
        upstream.get("source_protocol_ready_score"),
        "source",
    )
    require(
        root / "results" / (next(iter(PARENTS)) + ".json"),
        "learning_protocol_ready_score",
        1,
        upstream.get("learning_protocol_ready_score"),
        "learning",
    )
    roles = {}
    for role, ref in upstream["role_manifests"].items():
        snapshot = bind(Path(ref["path"]), ref["sha256"])
        if snapshot is not None:
            roles[role] = json.loads(snapshot.read_text())["rows"]
    if set(roles) != set(ROLE_COUNTS):
        return plan
    if mutate:
        roles["tune"][0].update(
            {k: roles["fit"][0][k] for k in ("source_bytes", "source_cluster_id")}
        )
    progress("role_identity_seal_before", sum(map(len, roles.values())), 0)
    for role, rows in roles.items():
        atomic_json(raw / "roles" / (role + ".json"), dict(role=role, rows=rows))
        plan["role_manifests"][role] = reference(raw / "roles" / (role + ".json"))
    plan["outcome_access_ledger"].append(
        dict(
            event="public_source_ids_and_role_hashes_sealed",
            order=0,
            role_hashes=deepcopy(plan["role_manifests"]),
            current_outcomes_opened=0,
        )
    )
    # This primary exposes annotation completeness, not private target values.
    eligibility = {
        r["family_id"]: r
        for r in parents["experiment_8019_v695_eligible_targets"]["eligibility_rows"]
    }
    source_seen: dict[str, str] = {}
    for role, originals in roles.items():
        branch = "learning" if role in ("stream", "retention") else "source"
        require(
            raw / "roles" / (role + ".json"),
            "original_slots",
            list(range(ROLE_COUNTS[role])),
            [r["slot"] for r in originals],
            branch,
        )
        for r in originals:
            identity = normalized(bytes.fromhex(r["source_bytes"]))
            require(
                raw / "roles" / (role + ".json"),
                "source_cluster_id",
                identity,
                r["source_cluster_id"],
                branch,
            )
            if branch == "source":
                require(
                    raw / "roles" / (role + ".json"),
                    "source_cluster_overlap",
                    role,
                    source_seen.get(identity, role),
                    branch,
                )
                source_seen[identity] = role
            old = eligibility.get(r["family_id"], {})
            known = old.get("target_eligible") is True and old.get("complete_annotation") is True
            eligible = r["public_eligible"] and known
            reason = r["exclusion_reason"] or (
                None if known else old.get("exclusion_reason") or "unknown_original_target"
            )
            plan["rows"].append(
                dict(
                    unit=f"{role}/{r['slot']}",
                    source=identity,
                    source_id=identity,
                    family_id=r["family_id"],
                    role=role,
                    slot=r["slot"],
                    arm="protocol",
                    seed=None,
                    numerator=int(eligible),
                    denominator=1,
                    eligible=eligible,
                    status="completed" if eligible else "excluded",
                    exclusion_reason=reason,
                    historical_development_exposure=True,
                    complete_annotation=old.get("complete_annotation"),
                    target_status="eligible_complete" if known else "unknown_or_incomplete",
                    release_slot=r["slot"] + 20 if role == "stream" else None,
                    feedback_role=("admission" if partition(identity) == 0 else "update")
                    if role == "stream"
                    else None,
                )
            )
    plan["outcome_access_ledger"].append(
        dict(
            event="historical_eligibility_metadata_imported",
            order=1,
            current_outcomes_opened=0,
            private_evaluator_files_opened=0,
        )
    )
    plan["source_valid"] = not any(f["branch"] in ("source", "both") for f in plan["failures"])
    plan["learning_valid"] = not any(f["branch"] in ("learning", "both") for f in plan["failures"])
    plan["qualified_head"] = upstream["qualified_head"]
    plan["repository_health"] = upstream["repository_health"]
    for receipt in plan["repository_health"]:
        bind(ROOT / receipt["log_path"], receipt["log_sha256"])
    progress("methods_sealed", len(plan["rows"]), 0)
    return plan


def build(
    plan: Json, raw: Path, receipts: list[Json], coverage: Json, fixture: bool, duration: float
) -> Json:
    """Reduce protocol rows; branch custody and owned checks must both qualify readiness."""
    rows = plan["rows"]
    coverage_ok = all(
        p in coverage
        and coverage[p]["summary"]["num_statements"] > 0
        and coverage[p]["summary"]["missing_lines"] == 0
        for p in OWNED
    )
    checks_ok = (
        bool(plan["validation_manifest"])
        and [r["name"] for r in receipts] == plan["validation_manifest"]
        and all(r["passed"] for r in receipts)
        and coverage_ok
        and not fixture
    )
    kind = "blocked" if plan["failures"] else "null" if checks_ok or fixture else "disqualified"
    gates = list(plan["failures"])
    if not fixture and not checks_ok:
        gates.append(
            dict(
                check="required_validation",
                upstream=TASK,
                path=str(raw / "validation.json"),
                hash=sha256_file(raw / "validation.json")
                if (raw / "validation.json").exists()
                else None,
                field="required_checks_passed",
                op="==",
                expected=True,
                observed=False,
                passed=False,
            )
        )
        for receipt in receipts:
            if not receipt["passed"]:
                gates.append(
                    dict(
                        check=receipt["name"],
                        upstream=TASK,
                        path=receipt.get("log_path", str(raw / "validation.json")),
                        hash=receipt.get("log_sha256"),
                        field="exit_code",
                        op="==",
                        expected=receipt.get("expected_exit", 0),
                        observed=receipt.get("exit_code", "FAILED_CHECK"),
                        argv=receipt.get("argv", []),
                        passed=False,
                    )
                )
    sizes = dict(
        intended_count=512,
        eligible_count=sum(r["eligible"] for r in rows),
        independent_count=len({r["source"] for r in rows if r["eligible"]}),
        completed_count=len(rows),
        censored_count=512 - len(rows),
        excluded_count=sum(not r["eligible"] for r in rows),
        failed_count=0,
    )
    spans = [dict(name="preconditions_method_seal_owned_validation", duration_s=duration)]
    current = build_current_work_receipt(
        run_id=str(raw.absolute()),
        owner_pid=plan.get("owner_pid", 0),
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_details=dict(no_model_load=True),
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=plan.get("started_monotonic_ns", 0),
        ended_monotonic_ns=plan.get("started_monotonic_ns", 0) + int(duration * 1e9),
        phase_spans=spans,
    )
    value = dict(
        experiment_id=8058,
        task_id=TASK,
        milestone="2026.10.698",
        run_date="20261003",
        schema="carnot.v698.sealed_evidence_methods.v1",
        honest_verdict="complete_blocked_" + Path(gates[0]["path"]).stem.replace(".", "_")
        if kind == "blocked"
        else "complete_" + kind + "_sealed_evidence_methods",
        verdict_class=kind,
        verifier_is_oracle=False,
        claim_scope="Method readiness for historically exposed development roles only. No new source outcomes, learned benefit or complete-service benefit measured. Private runner controls earn no scientific credit.",
        flagged_adversarial=False,
        required_checks_passed=checks_ok,
        gate_check_summary=gates,
        source_protocol_ready_score=int(checks_ok and plan["source_valid"]),
        learning_protocol_ready_score=int(checks_ok and plan["learning_valid"]),
        rows=rows,
        **sizes,
        sample_size_budget=dict(
            **sizes,
            unit="original source-cluster custody slots",
            independent_environments=0,
            source_role_counts=dict(fit=64, tune=32, evaluation=96),
            learning_role_counts=dict(stream=256, retention=64),
            seeds=20,
            model_calls=0,
            numerical_budget_s=1200,
            gradients_per_adaptive_arm_seed=12,
        ),
        random_seed=6988058,
        generalized_learning_benefit_score=0,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=current["invocation_counts"],
        current_work_receipt=current,
        substrate_declaration=dict(
            reduction="aggregation_from_upstream_artifacts", mode="no_model_load", MODEL_SPECS=[]
        ),
        validation_receipts=receipts,
        coverage_statement_counts=coverage,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        source_artifact_hashes=plan["source_artifact_hashes"],
        raw_shard_hashes=[
            reference(raw / p)
            for p in ["seal.json", "work.json", "validation.json", "validation_commands.json"]
            if (raw / p).is_file()
        ]
        + list(plan["role_manifests"].values()),
        code_config_hashes={p: sha256_file(ROOT / p) for p in [*OWNED, TEST]},
        reproducibility_checksum=canonical_hash(
            dict(plan=plan, receipts=receipts, coverage=coverage)
        ),
        duration_s=duration,
        phase_spans=spans,
        role_manifests=plan["role_manifests"],
        eligibility_rules="Public completeness AND bound historical complete-response annotation AND known target; missing/unknown stays unknown; no replacement or timeline compression.",
        outcome_access_ledger=plan["outcome_access_ledger"],
        method_source_map=plan["methods"]["method_source_map"],
        method_freeze=dict(
            sha256=canonical_hash(plan["methods"]),
            methods=plan["methods"],
            current_outcomes_opened=0,
            thresholds_tuned_after_outcomes=False,
        ),
        hypothesis_family=[
            dict(h, measured=False, p_value=1, disposition="sealed_not_measured")
            for h in plan["methods"]["hypotheses"]
        ],
        historical_exposure=dict(
            all_roles_development_exposed=True,
            newly_unseen=False,
            independent_environments=0,
            cross_branch_overlap="source evaluation is historical stream material; source-only roles have no overlap",
        ),
        sample_size_feasibility=dict(
            eligible_by_role={
                role: sum(r["eligible"] for r in rows if r["role"] == role) for role in ROLE_COUNTS
            },
            required_support=plan["methods"]["source"]["floors"],
            later_support=[80, 10],
            retention_support=[48, 8],
            class_support="not accessed; independent evaluator checks class counts after prediction seals",
            admission_certificate="12 dependent development labels cannot earn an iid certificate",
            admission_upper_bound=sum(
                r["eligible"] and r["feedback_role"] == "admission" for r in rows
            ),
            hypothesis_support_failure="block affected hypothesis only; no role expansion or threshold relaxation",
        ),
        qualified_head=plan["qualified_head"],
        qualified_head_sha256=canonical_hash(plan["qualified_head"]),
        repository_health=plan["repository_health"],
        methodology_note="Dated V698 primary-method scan adopted before outcomes. Original identities, complete eligibility, four-arm candidate/admission contract and H1-H3 fixed. Methods-only readiness does not close any PRD scientific gap.",
    )
    value["field_principles"] = {
        k: "Bind "
        + k
        + " to exact current protocol evidence; imported history and private controls add no current scientific observations."
        for k in value
    }
    for k in ("rows", *sizes, "sample_size_budget"):
        value["field_principles"][k] = (
            "Count original slots and distinct sources; preserve exclusions and denominators; seeds and bootstrap draws never inflate independent support."
        )
    for k in ("source_protocol_ready_score", "learning_protocol_ready_score"):
        value["field_principles"][k] = (
            "Qualify this branch independently with owned checks, never with prospect of a favorable outcome."
        )
    value["field_principles"]["gate_check_summary"] = (
        "Name exact failed operand and bytes; distinguish missing evidence from measured scientific failure."
    )
    value["field_principles"]["generalized_learning_benefit_score"] = (
        "Historically exposed finite development cohorts cannot close generalized lifelong learning."
    )
    return value


def replay(path: Path) -> bool:
    """Rebuild from sealed rows and exact input/log bytes without any evaluator read."""
    try:
        value = json.loads(path.read_text())
        raw = Path(value["terminal_validation_sidecar_path"]).parent
        for ref in value["source_artifact_hashes"] + value["raw_shard_hashes"]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
            if (
                ref.get("snapshot_path")
                and sha256_file(Path(ref["snapshot_path"])) != ref["sha256"]
            ):
                return False
        if any(
            sha256_file(ROOT / p) != digest for p, digest in value["code_config_hashes"].items()
        ):
            return False
        for receipt in value["validation_receipts"]:
            if sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]:
                return False
        plan = json.loads((raw / "seal.json").read_text())
        work = json.loads((raw / "work.json").read_text())
        validation = json.loads((raw / "validation.json").read_text())
        if plan["methods"] != methods():
            return False
        historical = {}
        if plan["role_manifests"]:
            ref = next(
                r
                for r in value["source_artifact_hashes"]
                if Path(r["path"]).name == "experiment_8019_v695_eligible_targets.json"
            )
            historical = {
                r["family_id"]: r
                for r in json.loads(Path(ref["snapshot_path"]).read_text())["eligibility_rows"]
            }
        primitives = []
        for role, ref in plan["role_manifests"].items():
            originals = json.loads(Path(ref["path"]).read_text())["rows"]
            actual = [r for r in plan["rows"] if r["role"] == role]
            if len(originals) != ROLE_COUNTS[role] or len(actual) != len(originals):
                return False
            for r, p in zip(actual, originals, strict=True):
                metadata = historical.get(p["family_id"], {})
                complete = (
                    metadata.get("target_eligible") is True
                    and metadata.get("complete_annotation") is True
                )
                expected_eligibility = p["public_eligible"] and complete
                if (
                    r["source"] != normalized(bytes.fromhex(p["source_bytes"]))
                    or r["slot"] != p["slot"]
                    or r["family_id"] != p["family_id"]
                    or r["numerator"] != int(r["eligible"])
                    or r["denominator"] != 1
                    or r["eligible"] != expected_eligibility
                    or r["release_slot"] != (p["slot"] + 20 if role == "stream" else None)
                    or r["feedback_role"]
                    != (
                        ("admission" if partition(r["source"]) == 0 else "update")
                        if role == "stream"
                        else None
                    )
                ):
                    return False
                primitives.append(r["unit"])
        return (
            len(set(primitives)) == len(plan["rows"])
            and build(
                plan,
                raw,
                validation["receipts"],
                validation["coverage"],
                work["fixture"],
                work["duration_s"],
            )
            == value
        )
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False


def manifest(private: Path) -> list[Json]:
    """Freeze explicit bounded checks and include only newly added statement coverage."""
    py, cov, pytest, ruff, mypy = [
        str(ROOT / ".venv/bin" / n) for n in ["python", "coverage", "pytest", "ruff", "mypy"]
    ]
    config = private / "coverage.ini"
    config.write_text(
        "[run]\nparallel = true\ndata_file = "
        + str(private / ".coverage")
        + "\ninclude =\n"
        + "".join("    " + str(ROOT / p) + "\n" for p in OWNED)
    )
    common = ["-n", "0", "-o", "addopts=", "--no-cov", "-q", "-s"]
    include = "--include=" + ",".join(str(ROOT / p) for p in OWNED)
    e2e = str(ROOT / "scripts/experiments/experiment_7868_v683_intervention_protocol.py")
    commands = [
        (
            "python_environment",
            [
                py,
                "-u",
                "-c",
                "import sys, pytest, coverage, ruff, mypy; print(sys.version, flush=True); assert sys.version_info >= (3,11)",
            ],
            30,
        ),
        (
            "focused_unit_and_cli",
            [
                cov,
                "run",
                "--rcfile=" + str(config),
                "-m",
                "pytest",
                *common,
                "--basetemp=" + str(private / "pytest"),
                TEST,
            ],
            180,
        ),
        (
            "consumer_tests",
            [
                pytest,
                *common,
                "--basetemp=" + str(private / "consumers"),
                *CONSUMERS,
                "tests/python/test_primary_publication_7928.py",
            ],
            180,
        ),
        (
            "E2E-015",
            [
                pytest,
                *common,
                "--basetemp=" + str(private / "e2e015"),
                "tests/python/test_source_boundary_7852.py",
            ],
            120,
        ),
        (
            "E2E-016_fixture",
            [py, e2e, "--date", "20260929", "--fixture-e2e", str(private / "e2e016.json")],
            120,
        ),
        (
            "E2E-016_cold_replay",
            [py, e2e, "--date", "20260929", "--cold-replay", str(private / "e2e016.json")],
            60,
        ),
        ("coverage_combine", [cov, "combine", "--rcfile=" + str(config)], 30),
        (
            "coverage_report",
            [
                cov,
                "report",
                "--rcfile=" + str(config),
                include,
                "--show-missing",
                "--fail-under=100",
            ],
            30,
        ),
        (
            "coverage_json",
            [cov, "json", "--rcfile=" + str(config), include, "-o", str(private / "coverage.json")],
            30,
        ),
        ("ruff_check", [ruff, "check", *OWNED, TEST], 30),
        ("ruff_format", [ruff, "format", "--check", *OWNED, TEST], 30),
        ("strict_mypy", [mypy, "--strict", "--follow-imports=silent", *OWNED], 60),
        ("scoped_spec_coverage", [py, "scripts/check_spec_coverage.py", TEST], 30),
    ]
    return [
        dict(name=n, argv=a, deadline_s=t, expected_exit=0, classification="required")
        for n, a, t in commands
    ]


def terminal(path: Path) -> Json:
    """Fresh subprocesses check cold reductions and shared readers before publication."""
    raw = Path(json.loads(path.read_text())["terminal_validation_sidecar_path"]).parent
    py = str(ROOT / ".venv/bin/python")
    commands = [
        ("cold_replay", [py, "-u", str(ROOT / CLI), "--cold-replay", str(path)]),
        ("adversarial", [py, str(ROOT / "scripts/adversarial_verify.py"), "--json", str(path)]),
        (
            "strict_rows",
            [py, str(ROOT / "scripts/verdict_row_consistency_lint.py"), "--strict", str(path)],
        ),
    ]
    with tempfile.TemporaryDirectory(prefix="carnot-8058-terminal-") as temp:
        receipts = [
            run_check(
                ROOT,
                dict(name=n, argv=a, deadline_s=60, expected_exit=0),
                Path(temp),
                raw / "terminal_logs" / path.name,
            )
            for n, a in commands
        ]
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def main(argv: list[str] | None = None) -> int:
    """Seal once and publish only checked bytes; private CLI routes cannot earn readiness."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    started = time.monotonic_ns()
    progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261003"], default="20261003")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--mutate", action="store_true")
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = replay(args.cold_replay)
        progress("cold_replay_passed" if passed else "cold_replay_rejected")
        return 0 if passed else 1
    output = (args.fixture_output or args.output).absolute()
    raw = output.parent / "raw" / output.stem
    if (raw / "seal.json").exists():
        progress("existing_seal_preserved")
        return 1
    try:
        with tempfile.TemporaryDirectory(prefix="carnot-8058-") as temp:
            private = Path(temp)
            specs = manifest(private)
            atomic_json(
                raw / "validation_commands.json",
                dict(
                    commands=specs,
                    code_config_hashes={p: sha256_file(ROOT / p) for p in [*OWNED, TEST]},
                    terminal_commands=[
                        dict(
                            name="cold_replay",
                            argv=[
                                str(ROOT / ".venv/bin/python"),
                                "-u",
                                str(ROOT / CLI),
                                "--cold-replay",
                                str(raw / "terminal_candidate.json"),
                            ],
                        ),
                        dict(
                            name="adversarial",
                            argv=[
                                str(ROOT / ".venv/bin/python"),
                                str(ROOT / "scripts/adversarial_verify.py"),
                                "--json",
                                str(raw / "terminal_candidate.json"),
                            ],
                        ),
                        dict(
                            name="strict_rows",
                            argv=[
                                str(ROOT / ".venv/bin/python"),
                                str(ROOT / "scripts/verdict_row_consistency_lint.py"),
                                "--strict",
                                str(raw / "terminal_candidate.json"),
                            ],
                        ),
                    ],
                    terminal_deadline_s=60,
                ),
            )
            plan = seal(args.root, raw, mutate=args.mutate)
            plan.update(
                owner_pid=os.getpid(),
                started_monotonic_ns=started,
                validation_manifest=[s["name"] for s in specs],
                sealed_at_utc=datetime.now(UTC).isoformat(),
            )
            atomic_json(raw / "seal.json", plan)
            progress("owned_validation_before", len(plan["rows"]), len(specs))
            os.environ["CARNOT_8058_COVERAGE_CONFIG"] = str(private / "coverage.ini")
            receipts = (
                []
                if args.fixture_output
                else [run_check(ROOT, s, private, raw / "validation_logs") for s in specs]
            )
            coverage = (
                json.loads((private / "coverage.json").read_text())["files"]
                if (private / "coverage.json").is_file()
                else {}
            )
            duration = (time.monotonic_ns() - started) / 1e9
            atomic_json(
                raw / "work.json", dict(fixture=bool(args.fixture_output), duration_s=duration)
            )
            atomic_json(raw / "validation.json", dict(receipts=receipts, coverage=coverage))
            progress("owned_validation_after", len(receipts), 0)
            value = build(plan, raw, receipts, coverage, bool(args.fixture_output), duration)
            progress("publication_before", len(value["rows"]), 1)
            publication = publish_primary(output, value, terminal)
            atomic_json(
                raw / "terminal_validation.json",
                dict(publication=publication, owned_invocation_exit=0),
            )
            progress("complete", len(value["rows"]), 0)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        progress("rejected_" + str(error))
        return 1
