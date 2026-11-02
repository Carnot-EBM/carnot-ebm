"""REQ-REPORT-8702: original deployment authority, independent of failed wrappers.

Only exact original inputs authorize execution. Readiness proves that bounded
installed batches preserve numerical work; it never implies a speed benefit.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any
import zipfile

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.v746_contract_authority import authority

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8702_v746_installed_batch_prototype"
TASK = "exp8702-installed-batch-prototype"
CLI = f"scripts/experiments/{NAME}.py"
TESTS = [
    "tests/python/test_installed_batch_8702.py",
    "tests/python/test_installed_batch_report_8702.py",
]
OWNED = (
    ["python/carnot/inference/resident_batch.py"]
    + [
        f"python/carnot/reporting/installed_batch{suffix}_8702.py"
        for suffix in ["", "_measure", "_runner"]
    ]
    + [CLI]
)
PINS = [
    (
        "results/experiment_8660_v743_installed_compatibility.json",
        "sha256:c4247e2fe93227fe3235972396b3811592cde37c2721a2074bea05c8ef7199af",
        {
            "experiment_id": 8660,
            "task_id": "exp8660-installed-compatibility",
            "installed_measurement_ready_score": 1,
        },
    ),
    (
        "results/experiment_8661_v743_installed_complete_cost.json",
        "sha256:6670350979ad4fdfcaac26996a1a8705d061d2f4d3cf9dc494621c69cafae4d8",
        {
            "experiment_id": 8661,
            "task_id": "exp8661-installed-complete-cost",
            "installed_cost_ready_score": 1,
        },
    ),
    (
        "openspec/change-proposals/v744-methods-protocol.json",
        "sha256:842e28d27098b7dadf41290c407e72cfa4019a2cbb09429c492355aa0f55cb67",
        {},
    ),
]


def reference(path: Path) -> Json:
    """Use byte identity so a replacement at the same location cannot qualify."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path))


def load(ref: Json) -> Any:
    """Check immutable primitives before trusting fields or aggregate results."""
    if reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
        raise ValueError("changed_bytes")
    return json.loads(Path(ref["path"]).read_bytes())


def authenticate(ref: Json, fields: Json, *, terminal: bool = False) -> Json:
    """Report every failed assertion without conflating absence and observed zero."""
    path, failures, refs, value = Path(ref["path"]), [], [ref], {}

    def fail(field: str, expected: Any, observed: Any, state: str) -> None:
        """An exact gate row tells a later owner which operand must change."""
        failures.append(
            dict(
                upstream=path.stem,
                check=field,
                path=str(path),
                sha256=ref["sha256"],
                field=field,
                operator="==",
                expected=expected,
                observed=observed,
                observed_state=state,
            )
        )

    if not path.is_file():
        fail("sha256", ref["sha256"], None, "missing_file")
    elif reference(path)["sha256"] != ref["sha256"]:
        fail("sha256", ref["sha256"], reference(path)["sha256"], "changed_bytes")
    else:
        try:
            value = load(ref)
            for key, expected in fields.items():
                observed = value.get(key)
                if key not in value or type(observed) is not type(expected) or observed != expected:
                    state = (
                        "missing_field"
                        if key not in value
                        else "observed_zero"
                        if type(observed) in {int, float} and observed == 0
                        else "observed"
                    )
                    fail(key, expected, observed, state)
            if terminal:
                side = reference(Path(value["terminal_validation_sidecar_path"]))
                refs.append(side)
                report = load(side)
                if (
                    report.get("primary_sha256") != ref["sha256"]
                    or report.get("report", {}).get("passed") is not True
                ):
                    fail("terminal_hash_and_acceptance", ref["sha256"], report, "observed")
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            fail(
                "readable_bound_operand",
                True,
                str(error),
                "missing_field" if isinstance(error, KeyError) else "unreadable",
            )
    return dict(passed=not failures, value=value, references=refs, failures=failures)


def inputs(root: Path) -> Json:
    """Read original compatibility, cost and method bytes without executing their controllers."""
    refs, failures, values = [], [], []
    for name, pin, fields in PINS:
        terminal = bool(fields)
        expected = (
            dict(fields, required_checks_passed=True, flagged_adversarial=False)
            if terminal
            else fields
        )
        checked = authenticate(dict(path=str(root / name), sha256=pin), expected, terminal=terminal)
        refs += checked["references"]
        failures += checked["failures"]
        values.append(checked["value"])
    contract = authority(root)
    failures += contract["gate_check_summary"]
    refs += contract["references"]
    result: Json = dict(
        passed=False, gate_check_summary=failures, references=refs, contract=contract
    )
    if failures:
        return result
    compatibility, cost, protocol = values
    method = protocol["installed_batching_protocol"]
    operands = [
        method["checkpoint_reference"],
        protocol["installed_entry_2x2"]["original_request_reference"],
        compatibility["target_model"]["native_extension_build"],
        compatibility["wheel_reference"],
    ]
    operands += compatibility["original_node_union"]["authority"]["test_sources"]
    operands.append(
        dict(
            path=str(
                root / "openspec/change-proposals/research-roadmap-v745-preserved-20261101.md"
            ),
            sha256="sha256:01fb99526e816d6ff017196bcd17aad215562ef217bcdf113746fcbfc8e38a63",
        )
    )
    for operand in operands:
        ref = {k: operand[k] for k in ["path", "sha256"]}
        checked = (
            authenticate(ref, {})
            if not ref["path"].endswith((".so", ".whl", ".py", ".md"))
            else None
        )
        if checked is not None:
            failures += checked["failures"]
        elif not Path(ref["path"]).is_file() or reference(Path(ref["path"])) != ref:
            failures.append(
                dict(
                    upstream="original_deployment",
                    check="operand_hash",
                    **ref,
                    field="sha256",
                    operator="==",
                    expected=ref["sha256"],
                    observed=reference(Path(ref["path"]))["sha256"]
                    if Path(ref["path"]).is_file()
                    else None,
                    observed_state="changed_bytes"
                    if Path(ref["path"]).is_file()
                    else "missing_file",
                )
            )
        refs.append(ref)
    if failures:
        return result
    head = load(operands[0])["heads"]["spline9"][0]
    requests = load(operands[1])
    if (
        [r["unit_id"] for r in requests] != method["request_identities"]
        or cost["target_model"] != compatibility["target_model"]
        or len(head["parameters"]) != method["parameter_count"]
    ):
        raise ValueError("original_assertions")
    result.update(
        passed=True,
        head=head,
        requests=requests,
        method=method,
        target_model=compatibility["target_model"],
        extension=operands[2],
        original_nodes=compatibility["original_node_union"]["original_nodes"],
        preserved_quarantines=protocol["historical_quarantines"],
        historical_model_provenance=cost["historical_model_provenance"],
    )
    return result


def roster(method: Json) -> list[Json]:
    """Freeze Exp8703's original complete-cost cells without calling future outputs evidence."""
    return [
        dict(
            unit_id=f"B{size}-R{repeat}-{arm}",
            batch=size,
            repeat=repeat,
            arm=arm,
            panels=method["panels_per_session"],
            item_evaluations=method["panels_per_session"] * method["requests_per_panel"],
            status="pending_Exp8703",
        )
        for size in method["batch_sizes"]
        for repeat in range(method["paired_repeats"])
        for arm in (
            method["arithmetic_arms"] if repeat % 2 == 0 else method["arithmetic_arms"][::-1]
        )
    ]


def protocol_added_lines() -> set[int]:
    """Measure only newly owned protocol statements, preserving historical coverage scope."""
    compatibility = load(dict(path=str(ROOT / PINS[0][0]), sha256=PINS[0][1]))
    with zipfile.ZipFile(compatibility["wheel_reference"]["path"]) as archive:
        before = archive.read("carnot/inference/resident_protocol.py").decode().splitlines()
    after = (ROOT / "python/carnot/inference/resident_protocol.py").read_text().splitlines()
    return {
        line + 1
        for tag, _, _, first, last in difflib.SequenceMatcher(
            a=before, b=after, autojunk=False
        ).get_opcodes()
        if tag in {"insert", "replace"}
        for line in range(first, last)
    }


def build(work: Json, output: Path) -> Json:
    """Derive terminal readiness from actual owned checks, never a historical score alone."""
    measured, validation = work["measurement"], work["validation"]
    checks = validation["checks"]
    owned = all(row["passed"] for row in checks) and validation["coverage"]["complete"]
    blocked = bool(work["inputs"]["gate_check_summary"])
    ready = owned and not blocked and not work.get("owned_error") and measured.get("passed", False)
    verdict = (
        "disqualified"
        if not owned or work.get("owned_error")
        else "blocked"
        if blocked
        else "circular_positive"
        if ready
        else "disqualified"
    )
    rows = measured.get("rows", [])
    value = dict(
        experiment_id=8702,
        task_id=TASK,
        milestone="2026.11.746",
        run_date="20261101",
        invocation_id=work["invocation_id"],
        honest_verdict=f"complete_{verdict}_v746_installed_batch_prototype",
        verdict_class=verdict,
        gate_check_summary=work["inputs"]["gate_check_summary"],
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(language_model_calls=0, model_loads=0),
        current_language_model_calls=0,
        target_model=work["inputs"].get("target_model"),
        historical_model_provenance=work["inputs"].get("historical_model_provenance", {}),
        rows=rows,
        intended_count=2048,
        completed_count=len(rows),
        failed_count=sum(not r["passed"] for r in rows),
        censored_count=2048 - len(rows),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(item_evaluations=2048, original_items=64, independent_questions=0),
        verifier_is_oracle=True,
        exposure_scope="original exposed fixed deployment corpus",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned,
        flagged_adversarial=not owned or bool(work.get("owned_error")),
        acceptance_gates=checks,
        validation_receipts=validation["receipts"],
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        adversarial_findings=work.get("adversarial_findings", []),
        preconditions_checked=work["preconditions"],
        duration_s=work["duration_s"],
        started_at_utc=work["started_at_utc"],
        ended_at_utc=work["ended_at_utc"],
        phase_spans=work["phase_spans"],
        random_seed=7448675,
        source_artifact_hashes=work["inputs"]["references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=validation["children"],
        cited_upstream_artifacts=work["inputs"]["references"],
        branch_gate_values=dict(batch_service_ready_score=int(ready)),
        batch_service_ready_score=int(ready),
        installed_wheel_reference=measured.get("installed_wheel_reference"),
        loaded_extension_reference=measured.get("loaded_extension_reference"),
        batch_parity_rows=rows,
        envelope_rejection_rows=measured.get("envelope_rejection_rows", []),
        deployment_operand_references=work["inputs"]["references"],
        complete_cost_roster=roster(work["inputs"]["method"]) if work["inputs"]["passed"] else [],
        original_node_union=validation.get("original_node_union", {}),
        owned_coverage=validation["coverage"],
        preserved_historical_nulls=[
            "V733 source-information",
            "V735 cold-cost",
            "V737 supervisor",
            "V743 batch-one NFR-01",
        ],
        preserved_quarantines=work["inputs"].get("preserved_quarantines", []),
        preserved_v745_authority="openspec/change-proposals/research-roadmap-v745-preserved-20261101.md",
        preserved_v745_flags={
            "Exp8681": "disqualified: empty final coverage and missing original rejection line113",
            "Exp8691": "disqualified: owned validation failed; later source repair does not clear history",
            "Exp8692": "constructed CPU32/24 arithmetic only",
            "Exp8693": "blocked: scientific/device evidence absent",
            "Exp8694": "blocked: scientific/device evidence absent",
        },
        throughput_benefit_score=0,
        physical_device_execution_score=0,
        methodology="Independently authenticate original deployment producers and V746 contract. Build and install current wheel offline outside checkout. Recompute all64 items for every batch size1-16 in Python and native arms, retain actual frames, verify complete responses and envelope rejection. Run original122 compatibility assertions in private namespaces. Readiness establishes deployability/parity only.",
        primitive_reference=reference(Path(work["raw"]) / "work.json"),
        owned_error=work.get("owned_error"),
    )
    value["field_principles"] = {
        key: "Bind exact invocation operands and actual validation; administrative deployability supplies no benefit or independent questions."
        for key in value
    }
    value["reproducibility_checksum"] = canonical_hash(value)
    return value
