"""REQ-REPORT-8743: bind feature reuse to original installed scientific operands.

Historical attribution selects this prototype. Current parity and ownership
checks qualify execution only; they cannot establish a throughput improvement.
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any
import zipfile

from carnot.reporting import installed_batch_8702 as original
from carnot.reporting.primary_publication import read_bound_sidecar

Json = dict[str, Any]
ROOT = original.ROOT
NAME, TASK = (
    "experiment_8743_v749_single_extraction_prototype",
    "exp8743-single-extraction-prototype",
)
CLI = f"scripts/experiments/{NAME}.py"
TESTS = ["tests/python/test_single_extraction_8743.py"]
OWNED = [
    "python/carnot/inference/resident_features_8743.py",
    "python/carnot/inference/resident_reuse_probe_8743.py",
    "python/carnot/reporting/single_extraction_8743.py",
    "python/carnot/reporting/single_extraction_measure_8743.py",
    "python/carnot/reporting/single_extraction_runner_8743.py",
    CLI,
]
MODIFIED = [f"python/carnot/inference/resident_{n}.py" for n in ["batch", "energy", "protocol"]]
PINS = [
    (
        "results/experiment_8733_v748_installed_cost_attribution.json",
        "sha256:955b3ccac700d28094d896f458ca9c8aea44b73f25917f76ad83b2529fe99b44",
        dict(experiment_id=8733, installed_attribution_ready_score=1),
    ),
    (
        "results/experiment_8716_v747_installed_import_isolation.json",
        "sha256:7152eb60177c873d646ec02305b63819269e7c4410471f7ea5a00c28e7c77c13",
        dict(experiment_id=8716, batch_service_ready_score=1),
    ),
]
reference, load, atomic_json, canonical_hash = (
    original.reference,
    original.load,
    original.atomic_json,
    original.canonical_hash,
)


def inputs(root: Path) -> Json:
    """Read pinned producer seals without rerunning their retired queue controllers."""
    result: Json = dict(passed=False, gate_check_summary=[], references=[])
    values = []
    for name, pin, fields in PINS:
        ref = dict(path=str(root / name), sha256=pin)
        checked = original.authenticate(
            ref, dict(fields, required_checks_passed=True, flagged_adversarial=False)
        )
        result["references"] += checked["references"]
        result["gate_check_summary"] += checked["failures"]
        if not checked["passed"]:
            continue
        try:
            value = checked["value"]
            wrapper = reference(Path(value["terminal_validation_sidecar_path"]))
            side = Path(load(wrapper)["publication"]["sidecar_path"])
            if read_bound_sidecar(root / name, side)["report"]["passed"] is not True:
                raise ValueError("terminal_receipt")
            work = load(value["primitive_reference"])
            if load(work["seal_reference"]) != {
                k: v for k, v in work.items() if k != "seal_reference"
            }:
                raise ValueError("producer_seal")
            result["references"] += [
                wrapper,
                reference(side),
                value["primitive_reference"],
                work["seal_reference"],
            ]
            values.append((value, work))
        except (OSError, ValueError, KeyError, TypeError) as error:
            result["gate_check_summary"].append(
                dict(
                    upstream=name,
                    path=str(root / name),
                    sha256=pin,
                    check="bound_terminal_and_seal",
                    field="terminal_validation_sidecar_path",
                    operator="==",
                    expected=True,
                    observed=str(error),
                    observed_state="unreadable",
                )
            )
    if result["gate_check_summary"]:
        return result
    attribution, deployment = values
    fraction = next(
        r
        for r in attribution[0]["component_fraction_intervals"]
        if r["arm"] == "native"
        and r["clock_domain"] == "child_serve"
        and r["operation"] == "duplicate_feature_digest_extraction"
    )
    if fraction["estimate"] != 0.45628422029939814:
        raise ValueError("historical_fraction")
    frozen = deployment[0]["complete_cost_protocol"]["frozen_installed_workload"]
    for key in ["wheel", "extension", "config"]:
        ref = frozen[key]
        operand = Path(ref["path"])
        observed = reference(operand)["sha256"] if operand.is_file() else None
        if observed != ref["sha256"]:
            result["gate_check_summary"].append(
                dict(
                    upstream="exp8716-installed-import-isolation",
                    **ref,
                    check="original_operand_hash",
                    field="sha256",
                    operator="==",
                    expected=ref["sha256"],
                    observed=observed,
                    observed_state="missing_file" if observed is None else "changed_bytes",
                )
            )
        result["references"].append(ref)
    if result["gate_check_summary"]:
        return result
    result.update(
        passed=True,
        head=load(frozen["config"])["head"],
        requests=frozen["requests"],
        method=frozen["method"],
        extension=frozen["extension"],
        duplicate_fraction=fraction,
        original_nodes=deployment[1]["inputs"]["original_nodes"],
        original_wheel=frozen["wheel"],
    )
    return result


def roster() -> list[Json]:
    """The original panel repeats across twelve cells without new independent cases."""
    return [
        dict(
            unit_id=f"{arm}-B{size}-{mode}",
            arm=arm,
            batch=size,
            mode=mode,
            item_evaluations=64,
            status="censored",
            passed=False,
        )
        for arm in ["python", "native"]
        for size in [1, 4, 16]
        for mode in ["baseline", "reuse"]
    ]


def added_lines(name: str) -> set[int]:
    """Measure the changed statements against the authenticated pre-prototype wheel."""
    prior = original.load(dict(path=str(ROOT / PINS[1][0]), sha256=PINS[1][1]))
    with zipfile.ZipFile(prior["installed_wheel_reference"]["path"]) as archive:
        before = archive.read(name.removeprefix("python/")).decode().splitlines()
    after = (ROOT / name).read_text().splitlines()
    return {
        i + 1
        for tag, _, _, first, last in difflib.SequenceMatcher(
            a=before, b=after, autojunk=False
        ).get_opcodes()
        if tag in {"insert", "replace"}
        for i in range(first, last)
    }


def build(work: Json, output: Path) -> Json:
    """Only current complete validation can grant prototype readiness."""
    measured, validation = work["measurement"], work["validation"]
    private = any(r.get("name") == "private_control" for r in validation["checks"])
    owned = private or (
        validation["coverage"]["complete"] and all(r["passed"] for r in validation["checks"])
    )
    blocked = bool(work["inputs"]["gate_check_summary"])
    ready = bool(
        owned
        and not private
        and not blocked
        and not work.get("owned_error")
        and measured.get("passed", False)
    )
    verdict = (
        "disqualified"
        if not owned or work.get("owned_error")
        else "blocked"
        if blocked
        else "circular_positive"
        if ready
        else "null"
        if private
        else "disqualified"
    )
    rows = measured.get("rows", roster())
    value: Json = dict(
        experiment_id=8743,
        task_id=TASK,
        milestone="2026.11.749",
        run_date="20261102",
        invocation_id=work["invocation_id"],
        honest_verdict=f"complete_{verdict}_single_extraction_prototype",
        verdict_class=verdict,
        gate_check_summary=work["inputs"]["gate_check_summary"],
        inference_substrate="verifier_ensemble_against_cached_candidates"
        if measured
        else "aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(language_model_calls=0, model_loads=0),
        rows=rows,
        intended_count=12,
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        excluded_count=0,
        independent_count=0,
        verifier_is_oracle=True,
        exposure_scope="original exposed64 panel; engineering parity only",
        independent_generalization_score=0,
        acceptance_gates=validation["checks"],
        required_checks_passed=bool(owned),
        flagged_adversarial=bool(not owned or work.get("owned_error")),
        validation_receipts=validation["receipts"] + measured.get("receipts", []),
        owned_coverage=validation["coverage"],
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        preconditions_checked=work["preconditions"],
        duration_s=work["duration_s"],
        started_at_utc=work["started_at_utc"],
        ended_at_utc=work["ended_at_utc"],
        phase_spans=work["phase_spans"],
        random_seed=7498743,
        source_artifact_hashes=work["inputs"]["references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=validation["children"] + measured.get("traces", []),
        cited_upstream_artifacts=work["inputs"]["references"],
        feature_reuse_ready_score=int(ready),
        branch_gate_values=dict(feature_reuse_ready_score=int(ready)),
        installed_wheel_reference=measured.get("installed_wheel_reference"),
        loaded_extension_reference=measured.get("loaded_extension_reference"),
        feature_extraction_count_rows=measured.get("feature_extraction_count_rows", []),
        wire_parity_rows=measured.get("wire_parity_rows", []),
        invalid_input_rows=measured.get("invalid_input_rows", []),
        ownership_mutation_rows=measured.get("ownership_mutation_rows", []),
        original_import_contract_rows=measured.get("original_import_contract_rows", []),
        original_node_union=validation.get("original_node_union", {}),
        frozen_8744_workload=measured.get("frozen_8744_workload"),
        historical_duplicate_fraction=work["inputs"].get("duplicate_fraction"),
        throughput_benefit_score=0,
        physical_device_execution_score=0,
        generalized_learning_benefit_score=0,
        adversarial_findings=work.get("adversarial_findings", []),
        owned_error=work.get("owned_error"),
        primitive_reference=reference(Path(work["raw"]) / "work.json"),
        methodology="Freeze two providers, baseline/reuse and B1/4/16 on the original64 panel. Build real wheel and extension, run outside checkout, count successful extractions and compare exact same-arm serialized decisions/digests. Keep original import assertions, parser errors, mutation controls, real JUnit/SQLite and separately sealed replay. No timing, learning, hidden-generalization or physical acceleration claim.",
    )
    value["field_principles"] = {
        k: "Bind actual producer bytes and current validation; execution parity gives no speed or generalization evidence."
        for k in value
    }
    value["reproducibility_checksum"] = canonical_hash(value)
    return value
