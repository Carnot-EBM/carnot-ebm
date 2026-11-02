"""REQ-REPORT-8166: freeze sentence methods without acquiring model outputs.

Historical text, baseline outputs and models retain their own provenance. This
run qualifies a protocol; it cannot establish new semantic accuracy or benefit.
"""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import methods_stream_execution_8111 as execution
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.verify import sentence_evidence_8166 as sentence

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8166_v706_sentence_evidence_methods"
TASK = "exp8166-sentence-evidence-methods"
MODULE = "python/carnot/verify/sentence_methods_8166.py"
NUMERIC = "python/carnot/verify/sentence_evidence_8166.py"
CLI = f"scripts/experiments/{NAME}.py"
RUNNER = MODULE
TEST = "tests/python/test_sentence_methods_8166.py"
PARSER_TEST = "tests/python/test_sentence_evidence_8166.py"
OWNED = [MODULE, NUMERIC, CLI]
RUN_DATE = "20261005"
MODEL_SPECS: list[Json] = []
ROLES = dict(fit=128, tune=64, evaluation=128)
PROTOCOL = "openspec/change-proposals/v706-sentence-evidence-protocol.json"
PROTOCOL_HASH = "sha256:190baab92def47aa6d0671bee2f64b1c3b814cbadb9da00bbca28e872af5aa95"
METHOD = "openspec/change-proposals/research-roadmap-v705-preserved-20261005.md"
METHOD_HASH = "sha256:9a7607e6240a2b857733cd3d9d5b0da6d581827eb818c7c7aa082983ef3537cc"
PINS = {
    "results/experiment_8151_v705_source_method_custody.json": "sha256:29d912b64552af30b1742e8c80cdf93585cdf9e81520986215bbee3317af351f",
    "results/experiment_8153_v705_fit_evidence_capture.json": "sha256:c4d1be0937d6fe6d5e538e009d8ed82242973d8616569c8369cc0ea7172103ec",
    "results/experiment_8154_v705_evidence_energy_fit.json": "sha256:5ab0405b7eb9d06fb461bbde2038f88e2c44c32a29ab02b2f650983521200350",
    "results/experiment_8156_v705_decision_audit.json": "sha256:65a13925f00541b524172bb9c3ea2be765e078cee8bafda3d55619b9a8a4de13",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush measured counters so child waits remain visible to the operator."""
    print(f"[exp8166] phase={phase} completed={completed} pending={pending}", flush=True)


def reference(path: Path) -> Json:
    """Bind exact bytes so replay detects both changed inputs and changed code."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path))


def select_control(rows: list[Json]) -> Json:
    """Choose on tune only, charging escalation to every missing original slot."""
    costs = {}
    for arm in ("scalar_span", "linear12", "radial16"):
        selected = [r for r in rows if r.get("role") == "tune" and r["arm"] == arm]
        if len({r["unit_id"] for r in selected}) != len(selected) or len(selected) > 64:
            raise ValueError("baseline_duplicate")
        total = sum(float(r["numerator"]) if r["status"] == "completed" else 0.5 for r in selected)
        costs[arm] = (total + (64 - len(selected)) * 0.5) / 64
    return dict(
        selected_control=min(costs, key=lambda a: costs[a]),
        control_costs=costs,
        selection_role="tune",
        source_denominator=64,
        retained_controls=sorted({r["arm"] for r in rows if r.get("role") == "tune"})
        or list(costs),
        descriptive_tune_rows=[r for r in rows if r.get("role") == "tune"],
        missing_action="escalate",
    )


def inputs(root: Path, raw: Path, *, fixture: bool = False) -> Json:
    """Authenticate original roles and baselines before preparing any request.

    Private fixtures relax historical file pins only. Exact fixture references,
    source separation and public field checks still run through the same code.
    """
    plan: Json = dict(
        checks=[],
        refs=[],
        sources=[],
        baselines=[],
        baseline_ref=None,
        upstream=[],
        provenance=[],
        role_refs={},
    )

    def require(path: Path, field: str, expected: Any, observed: Any) -> None:
        plan["checks"].append(
            dict(
                check=field,
                upstream=str(path),
                path=str(path.absolute()),
                hash=sha256_file(path) if path.is_file() else None,
                artifact_field=field,
                op="==",
                expected=expected,
                observed=observed,
                passed=expected == observed,
            )
        )
        if expected != observed:
            raise ValueError(field)

    def bind(ref: Json) -> Json:
        path = Path(ref["path"])
        require(path, "input_sha256", ref["sha256"], sha256_file(path) if path.is_file() else None)
        target = raw / "inputs" / (ref["sha256"].split(":")[-1] + "-" + path.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        plan["refs"].append(reference(target))
        return dict(json.loads(target.read_text()))

    progress("before_input_authentication")
    try:
        values = {}
        for name, pin in PINS.items():
            path = root / name
            require(path, "upstream_exists", True, path.is_file())
            value = bind(reference(path) if fixture else dict(path=str(path), sha256=pin))
            for field, expected in [
                ("required_checks_passed", True),
                ("flagged_adversarial", False),
            ]:
                require(path, field, expected, value.get(field))
            if not fixture:
                terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_text())
                sidecar = Path(terminal["publication"]["sidecar_path"])
                require(
                    path,
                    "terminal_passed",
                    True,
                    read_bound_sidecar(path, sidecar)["report"]["passed"],
                )
                plan["refs"].append(reference(sidecar))
            values[name] = value
            plan["upstream"].append(
                dict(
                    experiment_id=int(Path(name).name.split("_")[1]),
                    fields_imported=["source_role_manifests", "raw_shard_hashes", "rows"],
                    sha256=sha256_file(path),
                )
            )
            plan["provenance"].append(
                dict(
                    path=str(path),
                    MODEL_SPECS=value.get("MODEL_SPECS", []),
                    model_invocation_counts=value.get("model_invocation_counts", {}),
                )
            )
        require(Path(sys.executable), "python_runtime_supported", True, sys.version_info >= (3, 11))
        for tool in ("python", "pytest", "coverage", "ruff", "mypy"):
            require(
                ROOT / ".venv/bin" / tool,
                "runtime_tool_executable",
                True,
                os.access(ROOT / ".venv/bin" / tool, os.X_OK),
            )
        require(ROOT / PROTOCOL, "protocol_sha256", PROTOCOL_HASH, sha256_file(ROOT / PROTOCOL))
        require(ROOT / METHOD, "immutable_v705_methods", METHOD_HASH, sha256_file(ROOT / METHOD))
        config = json.loads((ROOT / PROTOCOL).read_text())
        h2 = ROOT / config["H2_protocol"]["path"]
        require(h2, "exact_H2_sha256", config["H2_protocol"]["sha256"], sha256_file(h2))
        plan["refs"] += [reference(ROOT / PROTOCOL), reference(ROOT / METHOD), reference(h2)]
        custody = values[next(iter(PINS))]
        seen: set[str] = set()
        for role, count in ROLES.items():
            manifest = bind(custody["source_role_manifests"][role])
            plan["role_refs"][role] = plan["refs"][-1]
            public = {r["family_id"]: r for r in manifest["request_rows"]}
            require(
                Path(custody["source_role_manifests"][role]["path"]),
                "role_count",
                count,
                len(public),
            )
            require(root, "roster_count", count, len(manifest["roster"]))
            for i, row in enumerate(manifest["roster"]):
                item = public[row["unit_id"]]
                require(
                    root,
                    "public_fields",
                    ["answer_bytes", "family_id", "source_bytes"],
                    sorted(item),
                )
                require(root, "role_separation", False, row["source_cluster_id"] in seen)
                require(root, "original_role", role, row["role"])
                require(root, "original_slot", i + 1, row["slot"])
                seen.add(row["source_cluster_id"])
                plan["sources"].append(dict(row, **item))
        for name, value in values.items():
            for ref in value["raw_shard_hashes"]:
                primitive = bind(ref)
                if "8154" in name and (fixture or ref["path"].endswith("primitive_decisions.json")):
                    plan["baselines"] = primitive["rows"]
                    plan["baseline_ref"] = plan["refs"][-1]
    except (OSError, ValueError, KeyError, TypeError) as error:
        if not any(not r["passed"] for r in plan["checks"]):
            plan["checks"].append(
                dict(
                    check="input_structure",
                    upstream=str(root),
                    path=str(root),
                    hash=None,
                    artifact_field="input_structure",
                    op="==",
                    expected="authentic complete inputs",
                    observed=str(error),
                    passed=False,
                )
            )
    progress("after_input_authentication", len(plan["sources"]), 320 - len(plan["sources"]))
    return plan


def freeze_sources(sources: list[Json]) -> list[Json]:
    """Save every original source slot, including escalation and intact bytes."""
    rows = []
    for source in sources:
        request = sentence.requests(source)
        completed = request["status"] == "completed"
        row = dict(source, **request)
        row.update(
            arm="sentence_protocol",
            condition="original",
            metric="lossless_request_ready",
            numerator=int(completed),
            denominator=1,
            status="completed" if completed else "excluded",
        )
        rows.append(row)
    return rows


def reduce_rows(rows: list[Json]) -> Json:
    """Count source clusters independently of sentences and repeated requests."""
    statuses = Counter(r["status"] for r in rows)
    return dict(
        intended_count=320,
        eligible_count=statuses["completed"],
        independent_count=len({r["source_cluster_id"] for r in rows}),
        completed_count=statuses["completed"],
        excluded_count=statuses["excluded"],
        censored_count=0,
        failed_count=320 - len(rows),
        sentence_count=sum(len(r["sentences"]) for r in rows),
        request_count=sum(len(r["requests"]) for r in rows),
    )


def measure(
    root: Path,
    raw: Path,
    *,
    fixture: bool = False,
    stream_path: Path | None = None,
    mutation: str = "",
) -> Json:
    """Freeze qualified source requests; no language model or fitted head runs."""
    start = time.monotonic()
    raw.mkdir(parents=True, exist_ok=True)
    plan = inputs(root, raw, fixture=fixture)
    if mutation:
        changed = dict(plan["sources"][0])
        field, altered = {
            "labels": ("human_target", 1),
            "roles": ("role", "tune"),
            "slots": ("slot", -1),
            "source": ("source_bytes", b"altered source.".hex()),
        }[mutation]
        changed[field] = altered
        plan["checks"].append(
            dict(
                check="private_" + mutation + "_tamper",
                upstream=str(root),
                path=str(root),
                hash=None,
                artifact_field=field,
                op="==",
                expected=plan["sources"][0].get(field),
                observed=changed[field],
                passed=plan["sources"][0].get(field) == changed[field],
            )
        )
        atomic_json(
            raw / "private_mutation.json",
            dict(original=plan["sources"][0], observed=changed, diagnostic_human_target=None),
        )
    progress("before_sentence_freeze", 0, len(plan["sources"]))
    rows = freeze_sources(plan["sources"])
    baseline = select_control(plan["baselines"])
    config = json.loads((ROOT / PROTOCOL).read_text())
    atomic_json(raw / "sentence_requests.json", dict(rows=rows))
    atomic_json(raw / "baseline_manifest.json", baseline)
    atomic_json(raw / "independent_reduction.json", reduce_rows(rows))
    work = dict(
        plan=plan,
        rows=rows,
        baseline=baseline,
        config=config,
        fixture=fixture,
        raw_shard_hashes=[
            reference(raw / p)
            for p in [
                "sentence_requests.json",
                "baseline_manifest.json",
                "independent_reduction.json",
            ]
        ],
        code_config_hashes=[
            reference(ROOT / p)
            for p in [
                *OWNED,
                PROTOCOL,
                "python/carnot/reporting/methods_stream_execution_8111.py",
                "python/carnot/reporting/current_work_receipt.py",
                "python/carnot/reporting/primary_publication.py",
                "python/carnot/reporting/v686_contract_validation.py",
            ]
        ],
        duration_s=time.monotonic() - start,
        phase_spans=[dict(phase="authenticate_and_freeze", duration_s=time.monotonic() - start)],
    )
    atomic_json(raw / "measurement.json", work)
    progress("after_sentence_freeze", len(rows), 320 - len(rows))
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Readiness certifies methods and custody; it grants no semantic benefit."""
    plan, config = work["plan"], work["config"]
    owned = bool(receipts) and all(r["passed"] for r in receipts)
    failures = [r for r in plan["checks"] if not r["passed"]]
    verdict = (
        "disqualified"
        if not owned
        else "blocked"
        if failures
        else "circular_positive"
        if fixture
        else "null"
    )
    suffix = (
        "owned_validation"
        if not owned
        else failures[0]["check"]
        if failures
        else "sentence_protocol_frozen"
    )
    value = dict(
        experiment_id=8166,
        task_id=TASK,
        run_date=RUN_DATE,
        honest_verdict="complete_" + verdict + "_" + suffix,
        verdict_class=verdict,
        sentence_protocol_ready_score=int(owned and not failures),
        artifact_resolution_ready_score=0,
        required_checks_passed=owned,
        flagged_adversarial=False,
        verifier_is_oracle=fixture,
        claim_scope="protocol qualification only; no local semantic outputs or decision benefit measured",
        exposure_scope="exposed_development_within_run_disjoint",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        MODEL_SPECS=[],
        trained_head_specs=[],
        planned_head_specs=[config["head"]],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        call_ledger=[],
        historical_model_provenance=plan["provenance"],
        cited_upstream_artifacts=plan["upstream"],
        rows=work["rows"],
        **reduce_rows(work["rows"]),
        source_manifest=plan["role_refs"],
        baseline_manifest=work["baseline"],
        feature_schema=config["features"],
        statistical_plan=config["statistical_plan"],
        H1=config["H1"],
        H2=config["H2"],
        method_map=config["method_map"],
        protocol_path=str(ROOT / PROTOCOL),
        protocol_sha256=PROTOCOL_HASH,
        source_artifact_hashes=plan["refs"],
        raw_shard_hashes=work["raw_shard_hashes"],
        code_config_hashes=work["code_config_hashes"],
        random_seed=70666,
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        sample_size_budget=dict(
            ROLES,
            intended=320,
            independent_unit="source_cluster",
            sentences_and_repeats_add_no_sources=True,
        ),
        preconditions_checked=True,
        gate_check_summary=plan["checks"],
        validation_receipts=receipts,
        measurement_reference=reference(raw / "measurement.json"),
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        repository_health=work.get("global_health"),
        fixture_protocol_only=fixture,
        acceptance_gates=dict(
            H1=config["H1"],
            owned="normal exits and 100% added statements",
            H2="exact unchanged V705 admission protocol",
        ),
        field_principles=dict(
            verdict="External blocks terminate; owned failures disqualify.",
            inference="Zero current calls are separate from historical model provenance.",
            independence="Only source clusters count. Exposed development earns zero generalization.",
            local_evidence="Quote byte membership is not entailment; no inherited diagnostic targets.",
            readiness="Protocol custody and normal validation only; no measured decision gain.",
        ),
        methodology_note="Frozen full text and conservative sentence coverage; tune-only strongest V705 control. No LLM, fitted head, semantic capture or independent-generalization claim.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Reconstruct primitives and headlines independently, rejecting altered bytes."""
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
        for receipt in value["validation_receipts"]:
            if (
                receipt.get("log_path")
                and sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]
            ):
                return False
        work = json.loads(Path(value["measurement_reference"]["path"]).read_text())
        if work["config"] != json.loads((ROOT / PROTOCOL).read_text()):
            return False
        if work["plan"]["baseline_ref"]:
            baseline_rows = json.loads(Path(work["plan"]["baseline_ref"]["path"]).read_text())[
                "rows"
            ]
            if baseline_rows != work["plan"]["baselines"]:
                return False
        source_rows = []
        for role in ROLES:
            if role not in work["plan"]["role_refs"]:
                continue
            ref = work["plan"]["role_refs"][role]
            manifest = json.loads(Path(ref["path"]).read_text())
            public = {r["family_id"]: r for r in manifest["request_rows"]}
            source_rows += [dict(r, **public[r["unit_id"]]) for r in manifest["roster"]]
        if (
            freeze_sources(source_rows) != work["rows"]
            or select_control(work["plan"]["baselines"]) != work["baseline"]
        ):
            return False
        return bool(
            build(
                work,
                Path(value["terminal_validation_sidecar_path"]).parent,
                value["validation_receipts"],
                fixture=value["fixture_protocol_only"],
            )
            == value
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


BASE_MANIFEST = execution.manifest


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze real tool paths before measurement; only pytest receives selectors."""
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
    ):
        specs = BASE_MANIFEST(private, candidate)
    specs["commands"][0]["argv"].append(PARSER_TEST)
    specs["commands"][0]["deadline_s"] = 180
    coverage = candidate.parent / "coverage" / (private.name + ".json")
    coverage.parent.mkdir(parents=True, exist_ok=True)
    specs["commands"][4]["argv"][-1] = str(coverage)
    specs["commands"][1]["name"] = "qualified_E2E015_016_and_protocol"
    specs["commands"][1]["argv"] = [
        str(ROOT / ".venv/bin/pytest"),
        "-n0",
        "-o",
        "addopts=",
        "--no-cov",
        "-q",
        "tests/python/test_source_boundary_7852.py",
        "tests/python/test_experiment_7868_v683_intervention_protocol.py",
        "tests/python/test_evidence_protocol_8124.py",
        "tests/python/test_primary_publication_7928.py",
    ]
    for index in (5, 6, 8):
        specs["commands"][index]["argv"].append(PARSER_TEST)
    specs["repository_health"]["deadline_s"] = 300
    return specs


def main(argv: list[str] | None = None) -> int:
    """Use the qualified heartbeat and atomic publication from any directory."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "manifest", manifest),
    ):
        return execution.main(argv)
