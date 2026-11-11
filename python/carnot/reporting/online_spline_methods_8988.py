"""REQ-REPORT-8988-CAUSAL: authenticate fitted states independently of collection."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from carnot.autoresearch import spline_basis_8986 as n
from carnot.memory import online_spline_8988 as a
from carnot.reporting import spline_basis_methods_8986 as base
from carnot.reporting import matched_corpus_a_8949 as qualification
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, canonical_hash

Json = dict[str, Any]
ROOT = base.ROOT
NAME, TASK, SEED = (
    "experiment_8988_v768_online_spline_methods",
    "exp8988-online-spline-methods",
    768088,
)
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/memory/online_spline_8988.py",
    "python/carnot/reporting/online_spline_methods_8988.py",
    "python/carnot/reporting/online_spline_fixtures_8988.py",
    "python/carnot/reporting/online_spline_runner_8988.py",
    CLI,
]
TESTS = ["tests/python/test_online_spline_methods_8988.py"]
INPUT = "results/experiment_8987_v768_spline_development_admission.json"
INPUT_PIN = "sha256:32fbf7d2e347f56cf3baf5d92a73b7d62ab8bd68373518d0e5346a2ac485e920"
PROTOCOL = "openspec/change-proposals/v768-online-spline-protocol.json"
reference, retain, checksum, seal = base.reference, base.retain, base.checksum, base.seal


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual phase counts without padding elapsed time."""
    print(f"[exp8988] phase={phase} completed={completed} pending={pending}", flush=True)


def protocol() -> Json:
    """Read the immutable owned online contract; offline basis bytes remain unchanged."""
    return dict(json.loads((ROOT / PROTOCOL).read_bytes()))


def authenticate(root: Path, raw: Path) -> Json:
    """Collection readiness and prior signal failure are context, never prerequisites."""
    evidence: Json = dict(checks=[], failures=[], references=[], producer={}, method_source_rows=[])
    contract = protocol()
    operands = [
        (INPUT, INPUT_PIN, [("fitted_states_ready_score", 1), ("required_checks_passed", True)])
    ]
    operands += [(r["path"], r["sha256"], []) for r in contract["historical_inputs"]]
    for name, pin, gates in operands:
        path = root / name
        check = base.gate(
            path, "sha256", pin, reference(path)["sha256"] or "missing_file", "input_bytes", name
        )
        evidence["checks"].append(check)
        if not check["passed"]:
            continue
        saved = retain(reference(path), raw)
        evidence["references"].append(saved)
        value = json.loads(Path(saved["path"]).read_bytes())
        for field, expected in gates:
            evidence["checks"].append(
                base.gate(
                    path,
                    field,
                    expected,
                    value.get(field, "missing_field"),
                    "fitted_states_gate",
                    name,
                )
            )
        if name == INPUT:
            evidence["producer"] = value
    producer = evidence["producer"]
    if producer and all(r["passed"] for r in evidence["checks"]):
        refs = [producer["trained_state_reference"], *producer["feature_references"]]
        for ref in refs:
            path = Path(ref["path"])
            row = base.gate(
                path,
                "sha256",
                ref["sha256"],
                reference(path)["sha256"] or "missing_file",
                "retained_operand",
                "exp8987",
            )
            evidence["checks"].append(row)
            if row["passed"]:
                evidence["references"].append(retain(ref, raw))
    evidence["references"].append(retain(reference(ROOT / PROTOCOL), raw))
    evidence["failures"] = [r for r in evidence["checks"] if not r["passed"]]
    return evidence


def load(evidence: Json, ref: Json) -> Json:
    """Resolve only authenticated retained copies, not mutable upstream paths."""
    saved = next(r for r in evidence["references"] if r.get("source_path") == ref["path"])
    if reference(Path(saved["path"]))["sha256"] != ref["sha256"]:
        raise ValueError("retained_operand_drift")
    return dict(json.loads(Path(saved["path"]).read_bytes()))


def request(evidence: Json, label_path: Path, *, no_signal: bool = False) -> Json:
    """Private constructed streams reuse actual offline states and all original guard graphs."""
    source = evidence["producer"]
    states = load(evidence, source["trained_state_reference"])
    initial: Json = {}
    guard_design: Json = {}
    online_design: Json = {}
    for kind, source_arm in [("spline", "spline"), ("linear", "converged_linear")]:
        heads = sorted(
            [h for h in states["heads"] if h["arm"] == source_arm], key=lambda h: h["seed"]
        )
        coefficients = [h["coefficients"] for h in heads]
        for arm in a.ARMS:
            if (arm == "adaptive_linear") == (kind == "linear"):
                initial[arm] = json.loads(json.dumps(coefficients))
        guard_design[kind], online_design[kind] = [], []
        for head in heads:
            ref = next(
                r
                for r in source["feature_references"]
                if r["seed"] == head["seed"] and r["role"] == "guard"
            )
            features = np.asarray(load(evidence, ref)["scaled"])
            guard_design[kind].append(n.design(features, kind == "spline").tolist())
            private = np.repeat(features[:1], 32, axis=0) if no_signal else features[:32]
            online_design[kind].append(n.design(private, kind == "spline").tolist())
    guard_rows = [
        r
        for r in source["rows"]
        if r["condition"] == "exposed_development_guard" and r["arm"] == "spline"
    ]
    return dict(
        initial=initial,
        online_design=online_design,
        guard_design=guard_design,
        guard_labels=[r["error"] for r in guard_rows],
        label_path=str(label_path),
        temperature=1.0,
    )


def evaluate(evidence: Json, raw: Path, replay: bool = False) -> Json:
    """Run causal, arithmetic and real child recovery checks without loading an LLM."""
    from carnot.reporting.online_spline_fixtures_8988 import fixtures

    return fixtures(evidence, raw, replay)


def build(work: Json, validation: Json, fixture: bool) -> Json:
    """Only normal-exit checks and measured owned coverage can qualify readiness."""
    evidence, result = work["evidence"], work["result"]
    owned = qualification.qualified(validation, OWNED)
    passed = bool(result) and all(
        r["passed"] for r in result["causal_fixture_rows"] + result["crash_restart_rows"]
    )
    disqualified = (
        bool(work.get("owned_failure"))
        or (not fixture and not owned)
        or (bool(result) and not passed)
    )
    blocked = bool(evidence["failures"])
    cls = "blocked" if blocked else "disqualified" if disqualified else "null"
    ready = int(owned and passed and not blocked and not disqualified and not fixture)
    reason = (
        evidence["failures"][0]["check"]
        if blocked
        else "owned_validation"
        if disqualified
        else "online_spline_methods_qualified"
    )
    contract = protocol()
    rows = result.get("rows", [])
    value: Json = dict(
        result,
        experiment_id=8988,
        task_id=TASK,
        milestone="2026.11.768",
        run_date="20261111",
        honest_verdict=f"complete_{cls}_{reason}",
        verdict_class=cls,
        status="blocked" if blocked else "error" if disqualified else "success",
        online_methods_ready_score=ready,
        spline_methods_ready_score=ready,
        required_checks_passed=owned,
        flagged_adversarial=False,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        preconditions_checked=work["preconditions"],
        gate_check_summary=evidence["checks"],
        verifier_is_oracle=False,
        independent_generalization_score=0,
        claim_scope="Delayed fixed-basis coefficient mechanics only; no scientific benefit or oracle-distinct verifier win.",
        exposure_scope="Constructed private streams and exposed original64 historical guard graphs; cheap BFS graph labels exist.",
        fixture_only=fixture,
        scientific_evidence_fixture_only=True,
        inference_substrate="verifier_ensemble_against_cached_candidates"
        if result
        else "aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        model_invoked=False,
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        trained_head_specs=[
            dict(
                arm=arm,
                parameter_count=28 if arm == "adaptive_linear" else 217,
                encoder_seeds=[71, 72, 73],
                temperature=1.0,
                initial_source_sha256=INPUT_PIN,
                update="frozen" if arm == "frozen_spline" else "anchored clipped AdaGrad",
            )
            for arm in a.ARMS
        ],
        rows=rows,
        intended_count=320,
        independent_count=0,
        completed_count=len(rows),
        excluded_count=0,
        censored_count=320 - len(rows),
        failed_count=0,
        random_seed=SEED,
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        source_artifact_hashes=evidence["references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=work["raw_references"],
        global_test_debt=work["global_test_debt"],
        online_protocol_path=str(ROOT / PROTOCOL),
        online_protocol_sha256=reference(ROOT / PROTOCOL)["sha256"],
        arm_definitions=contract["arm_definitions"],
        update_equations=contract["update_equations"],
        ten_bound_contract=contract["ten_bound_contract"],
        state_schema=contract["state_schema"],
        causal_fixture_rows=result.get("causal_fixture_rows", []),
        crash_restart_rows=result.get("crash_restart_rows", []),
        promotion_guard_rows=result.get("promotion_guard_rows", []),
        operation_counts=result.get("operation_counts", {}),
        cited_upstream_artifacts=[
            dict(
                experiment_id=8987,
                sha256=INPUT_PIN,
                fields_imported=[
                    "fitted_states_ready_score",
                    "required_checks_passed",
                    "trained_state_reference",
                    "feature_references",
                    "rows",
                ],
            ),
            dict(
                experiment_id=8967,
                sha256=contract["historical_inputs"][1]["sha256"],
                fields_imported=["honest_verdict", "continuation_decision"],
            ),
        ],
        acceptance_gates=dict(
            authenticated_inputs=not blocked,
            causal_mechanics=passed,
            owned_checks=owned,
            later_scientific_bounds="unmeasured",
        ),
        methodology_note="Three frozen encoder probabilities form one ensemble. Per-label anchored clipped AdaGrad updates only fixed coefficients. Guard rejection rolls back coefficients and accumulator while acknowledging labels. Real SIGKILL/restart at query16 must reproduce exact state and predictions. "
        + contract["cost_boundary"],
        execution_venue="host",
        fresh_collection_permitted=False,
    )
    value["field_principles"] = {
        k: "Retained primitive bytes and actual validation qualify mechanics only; fixtures do not establish independent scientific benefit."
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = checksum(value)
    return value
