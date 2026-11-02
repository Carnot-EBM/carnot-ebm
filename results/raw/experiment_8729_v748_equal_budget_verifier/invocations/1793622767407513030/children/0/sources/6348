"""REQ-VERIFY-8219 / REQ-REPORT-8219: freeze useful correction choices.

Only cached fit/tune probabilities are reduced. Registering a method cannot
establish its benefit; future experiments own correction fitting and evaluation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time
from typing import Any
from unittest.mock import patch

import numpy as np

from carnot.reporting import methods_stream_execution_8111 as execution
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.verify import restricted_action_methods_8207 as qualified
from carnot.verify import restricted_action_rule_8207 as rule
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8219_v710_utility_patch_methods"
TASK = "exp8219-utility-patch-methods"
MODULE = "python/carnot/verify/utility_patch_methods_8219.py"
RUNNER = MODULE
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_utility_patch_methods_8219.py"
OWNED = [MODULE, CLI]
RUN_DATE = "20261007"
MODEL_SPECS: list[Json] = []
PROTOCOL = "openspec/change-proposals/v710-utility-patch-protocol.json"
PIN = "sha256:8ea372ec911cf3e4feeed75a02f8dad76ad160ed5aaa8def0e52ecfe0f5680d1"
PROTOCOL_VALUE: Json = json.loads((ROOT / PROTOCOL).read_bytes())
BASE_MANIFEST = execution.manifest


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed counts let the parent distinguish bounded work from a stalled child."""
    print(f"[exp8219] phase={phase} completed={completed} pending={pending}", flush=True)


def member(row: Json, group: Json) -> bool:
    """Frozen baseline membership prevents a patch from moving its own boundaries."""
    p, bounds = row["baseline_p"], group["interval"]
    return bool(
        (
            bounds is None
            or p is not None
            and bounds[0] <= p
            and (p < bounds[1] or bounds[1] == p == 1)
        )
        and (not group["reject_only"] or row["baseline_action"] == "reject")
    )


def freeze_dictionary(public: list[Json]) -> list[Json]:
    """Deduplicate public fit vectors without inspecting targets or source names."""
    if any(
        set(r) != {"unit_id", "source_cluster_id", "baseline_p", "baseline_action"}
        or r["baseline_p"] is not None
        and (not math.isfinite(r["baseline_p"]) or not 0 <= r["baseline_p"] <= 1)
        for r in public
    ):
        raise ValueError("public_schema")
    groups, seen = [], set()
    intervals = [[0, 0.1], [0.1, 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1]]
    predicates = [dict(name="global", interval=None, reject_only=False)] + [
        dict(name=f"bin_{i}" + ("_reject" if reject else ""), interval=b, reject_only=reject)
        for reject in [False, True]
        for i, b in enumerate(intervals)
    ]
    for g in predicates:
        vector = tuple(member(r, g) for r in public)
        if vector not in seen:
            seen.add(vector)
            groups.append(
                dict(
                    g,
                    fit_membership_sha256=canonical_hash(list(vector)),
                    distinct_fit_members=sum(vector),
                )
            )
    return groups


def residuals(rows: list[Json], groups: list[Json]) -> list[Json]:
    """Affine action costs share probability residuals; missing rows keep n fixed."""
    result = []
    for g in groups:
        selected = [r for r in rows if member(r, g) and r["p"] is not None and r["y"] in (0, 1)]
        residual = math.fsum(r["y"] - r["p"] for r in selected) / len(rows) if rows else None
        result.append(
            dict(
                group=g["name"],
                numerator=math.fsum(r["y"] - r["p"] for r in selected),
                denominator=len(rows),
                available_members=len(selected),
                distinct_members=len({r["source_cluster_id"] for r in selected}),
                eligible=len({r["source_cluster_id"] for r in selected}) >= 8,
                probability_residual=residual,
                accept_cost_residual=5 * residual if residual is not None else None,
                reject_cost_residual=-residual if residual is not None else None,
                escalate_cost_residual=0,
            )
        )
    return result


def baseline_probability(row: Json, baseline: Json) -> float | None:
    """The original qualified head defines groups, independent of corrected heads."""
    if row["historical_x"] is None:
        return None
    phi = rule.base.design("radial16", np.asarray([row["historical_x"]]), baseline["geometry"])[0]
    offset, slope = baseline["calibration"]
    z = offset + slope * math.fsum(
        float(a) * float(b) for a, b in zip(phi, baseline["weights"], strict=True)
    )
    return rule.probability(0, -z, 1)


def reduce_primitives(data: Json, protocol: Json) -> Json:
    """Recompute fit/tune witnesses and energy normalization without future targets."""
    if any(r["role"] not in ("fit", "tune") for r in data["rows"]) or any(
        r["y"] is not None for r in data["reserved_mask"]
    ):
        raise ValueError("future_target")
    records, errors = {}, []
    for index, head in enumerate(data["heads"]):
        progress("before_benchmark_cached_" + head["arm"], index, len(data["heads"]) - index)
        scored = []
        for row in data["rows"]:
            query = qualified.query(row)
            prediction = rule.predict(head, query, data["baseline"])
            p = prediction["p"]
            fp = baseline_probability(row, data["baseline"])
            scored.append(
                dict(row, p=p, baseline_p=fp, baseline_action=prediction["baseline_action"])
            )
            if p is not None:
                clipped = min(1 - 1e-6, max(1e-6, p))
                errors.append(
                    abs(rule.probability(-math.log1p(-clipped), -math.log(clipped), 1) - clipped)
                )
        records[head["arm"]] = {
            role: residuals(
                [r for r in scored if r["unit_id"] in {s["unit_id"] for s in data["roles"][role]}],
                protocol["witness_dictionary"],
            )
            for role in ["head_fit", "temperature_fit", "calibration"]
        }
        progress("after_benchmark_cached_" + head["arm"], index + 1, len(data["heads"]) - index - 1)
    return dict(
        residuals=records,
        energy_logistic_maximum_error=max(errors, default=0),
        targets_inspected=["head_fit", "temperature_fit", "calibration"],
        fitted_patch_count=0,
        scientific_benefit_measured=False,
    )


def reference(path: Path) -> Json:
    """Hash exact bytes, so evidence remains checkable in another interpreter."""
    return dict(path=str(path), sha256=sha256_file(path))


def gate(work: Json, path: Path, field: str, expected: Any, observed: Any) -> None:
    """A missing operand stays null instead of being mistaken for measured zero."""
    work["checks"].append(
        dict(
            check=field,
            path=str(path),
            upstream=str(path),
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


def bind(work: Json, path: Path, expected: str, raw: Path) -> Path:
    """Keep immutable snapshots without altering any historical source bytes."""
    gate(work, path, "sha256", expected, sha256_file(path) if path.is_file() else None)
    target = raw / "custody" / (expected.split(":")[1] + ".bin")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(path.read_bytes())
    work["refs"].append(dict(reference(target), upstream_path=str(path)))
    return target


def measure(
    root: Path,
    raw: Path,
    *,
    fixture: bool = False,
    stream_path: Path | None = None,
    mutation: str = "",
) -> Json:
    """Authenticate resources before reducing only the permitted cached rows."""
    began, wall = time.monotonic_ns(), time.time_ns()
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    work: Json = dict(
        checks=[],
        refs=[],
        evidence={},
        diagnostics={},
        owned_failure="",
        protocol=PROTOCOL_VALUE,
        precondition_receipts=[],
    )
    progress("before_preconditions")
    try:
        gate(work, Path(sys.executable), "python_supported", True, sys.version_info >= (3, 11))
        probe = raw / ".probe"
        probe.write_bytes(b"private writable scratch")
        gate(
            work,
            raw,
            "private_scratch_writable",
            True,
            probe.read_bytes() == b"private writable scratch",
        )
        probe.unlink()
        bind(work, ROOT / PROTOCOL, PIN, raw)
        protocol = work["protocol"]
        copies = {}
        for index, ref in enumerate(protocol["source_artifact_hashes"]):
            path = root / Path(ref["path"]).relative_to(ROOT)
            if mutation:
                gate(work, path, "source_custody", "authenticated", None)
            copies[ref["path"]] = bind(work, path, ref["sha256"], raw)
            if path.name.endswith(".receipt.json"):
                receipt = json.loads(copies[ref["path"]].read_bytes())
                gate(work, path, "literature_access_passed", True, receipt["passed"])
                work["precondition_receipts"].append(receipt)
            progress(
                "authenticated_inputs",
                index + 1,
                len(protocol["source_artifact_hashes"]) - index - 1,
            )
        for name in [
            "experiment_8208_v709_restricted_energy_fit",
            "experiment_8212_v709_memory_benefit_audit",
        ]:
            path = root / "results" / (name + ".json")
            primary = json.loads(copies[str(ROOT / "results" / (name + ".json"))].read_bytes())
            gate(work, path, "required_checks_passed", True, primary.get("required_checks_passed"))
            gate(work, path, "flagged_adversarial", False, primary.get("flagged_adversarial"))
            terminal = json.loads(copies[primary["terminal_validation_sidecar_path"]].read_bytes())
            bound = root / Path(terminal["publication"]["sidecar_path"]).relative_to(ROOT)
            gate(
                work,
                path,
                "terminal_publication_passed",
                True,
                read_bound_sidecar(path, bound)["report"]["passed"],
            )
        evidence = json.loads(copies[protocol["fit_measurement"]["path"]].read_bytes())["evidence"]
        sealed = json.loads(copies[protocol["sealed_predictions"]["path"]].read_bytes())
        gate(work, raw, "reserved_targets_closed", False, sealed["labels_opened"])
        mask = [
            dict(
                unit_id=r["unit_id"],
                source_cluster_id=r["source_cluster_id"],
                slot=r["slot"],
                original_status=r["status"],
                exclusion_reason=r["exclusion_reason"],
                y=None,
            )
            for r in sealed["rows"]
            if r["arm"] == "energy"
        ]
        gate(work, raw, "reserved_slot_count", 128, len(mask))
        gate(
            work,
            raw,
            "reserved_identity_set",
            sorted(r["unit_id"] for r in protocol["role_manifest"]["reserved"]),
            sorted(r["unit_id"] for r in mask),
        )
        gate(work, raw, "role_manifest", protocol["role_manifest"], evidence["roles"])
        public = [
            dict(
                unit_id=r["unit_id"],
                source_cluster_id=r["source_cluster_id"],
                baseline_p=baseline_probability(r, evidence["baseline"]),
                baseline_action=rule.base.action(baseline_probability(r, evidence["baseline"])),
            )
            for r in evidence["rows"]
            if r["unit_id"] in {s["unit_id"] for s in evidence["roles"]["head_fit"]}
        ]
        gate(
            work,
            raw,
            "frozen_witness_dictionary",
            protocol["witness_dictionary"],
            freeze_dictionary(public),
        )
        work["evidence"] = {k: evidence[k] for k in ["rows", "roles", "heads", "baseline"]}
        work["evidence"]["reserved_mask"] = mask
    except (OSError, ValueError, KeyError, TypeError) as error:
        if all(c["passed"] for c in work["checks"]):
            work["checks"].append(
                dict(
                    check="authenticated_input_schema",
                    path=str(root),
                    hash=None,
                    artifact_field="authenticated_input_schema",
                    op="==",
                    expected="valid",
                    observed=str(error),
                    passed=False,
                )
            )
    progress("after_preconditions", len(work["checks"]), 0)
    if all(c["passed"] for c in work["checks"]):
        try:
            work["diagnostics"] = reduce_primitives(work["evidence"], work["protocol"])
            if work["diagnostics"]["energy_logistic_maximum_error"] > 1e-10:
                raise ValueError("energy_logistic_identity")
        except ValueError as error:
            work["owned_failure"] = str(error)
    work["code_config_hashes"] = []
    for path in [ROOT / p for p in [*OWNED, TEST, PROTOCOL]]:
        snapshot = bind(dict(checks=[], refs=[]), path, sha256_file(path), raw)
        work["code_config_hashes"].append(dict(reference(path), snapshot_path=str(snapshot)))
    atomic_json(raw / "fixture_primitives.json", work["evidence"])
    work["raw_shard_hashes"] = [
        reference(p)
        for p in [raw / "fixture_primitives.json", raw / "validation_commands.json"]
        if p.is_file()
    ]
    work["duration_s"] = (time.monotonic_ns() - began) / 1e9
    work["clock"] = dict(
        started_monotonic_ns=began, ended_monotonic_ns=time.monotonic_ns(), started_wall_ns=wall
    )
    atomic_json(raw / "measurement.json", work)
    progress("measurement_complete", len(work["diagnostics"]), 0)
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Execution readiness is separate from unmeasured H1 and H2 benefit."""
    protocol = work["protocol"]
    failures = [c for c in work["checks"] if not c["passed"]]
    checked = bool(receipts) and all(r["passed"] for r in receipts) and not work["owned_failure"]
    ready = int(checked and not failures and bool(work["diagnostics"]))
    verdict = (
        "disqualified"
        if not checked
        else "blocked"
        if failures
        else "circular_positive"
        if fixture
        else "null"
    )
    operand = Path(failures[0]["path"]).stem.lower() if failures else "utility_patch_methods"
    mask = {r["unit_id"]: r for r in work["evidence"].get("reserved_mask", [])}
    rows = [
        dict(
            **r,
            arm="utility_protocol_registration",
            seed=7108219,
            condition="original_reserved_slot",
            metric="protocol_registered",
            numerator=ready,
            denominator=1,
            status="completed" if ready else "excluded",
            exclusion_reason=None if ready else "protocol_unavailable",
            semantic_metric=None,
            original_missing=None
            if r["unit_id"] not in mask
            else mask[r["unit_id"]]["original_status"] == "excluded",
            original_exclusion_reason=mask.get(r["unit_id"], {}).get("exclusion_reason"),
            evaluation_status="unmeasured",
        )
        for r in protocol["role_manifest"]["reserved"]
    ]
    value: Json = dict(
        experiment_id=8219,
        task_id=TASK,
        milestone="2026.10.710",
        run_date=RUN_DATE,
        honest_verdict="complete_" + verdict + "_" + operand,
        verdict_class=verdict,
        gate_check_summary=work["checks"],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(
            model_loads=0, generate_calls=0, forward_calls=0, model_count=0
        ),
        call_ledger=[],
        trained_head_specs=[
            dict(
                arm=h["arm"],
                current_fit=False,
                source=protocol["fit_measurement"],
                coefficients=len(h["weights"]),
            )
            for h in work["evidence"].get("heads", [])
        ],
        rows=rows,
        intended_count=128,
        completed_count=128 if ready else 0,
        failed_count=0,
        censored_count=0,
        excluded_count=0 if ready else 128,
        independent_count=128,
        verifier_is_oracle=fixture,
        exposure_scope="exposed_development_within_run_disjoint",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        utility_protocol_ready_score=ready,
        protocol_path=str(ROOT / PROTOCOL),
        protocol_sha256=PIN,
        witness_dictionary=protocol["witness_dictionary"],
        fit_tune_roles=protocol["fit_tune_roles"],
        H1=protocol["H1"],
        H2=protocol["H2"],
        method_mapping=protocol["method_mapping"],
        role_manifest=protocol["role_manifest"],
        static_arms=protocol["static_arms"],
        primary_comparator_selection=protocol["comparator"],
        primary_comparator_selected=None,
        original_reserved_mask=work["evidence"].get("reserved_mask", []),
        descriptive_utility_residuals=work["diagnostics"],
        owned_failure=work["owned_failure"],
        flagged_adversarial=False,
        required_checks_passed=checked,
        acceptance_gates=dict(
            owned_validation=checked,
            input_authentication=not failures,
            frozen_method_usable=bool(work["diagnostics"]),
            H1="registered_unmeasured",
            H2="registered_unmeasured",
        ),
        validation_receipts=receipts,
        changed_code_coverage_reference=reference(raw / "logs" / "changed_code_coverage.json")
        if (raw / "logs" / "changed_code_coverage.json").is_file()
        else None,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=[
            dict(resource=c["check"], available=c["passed"], path=c["path"]) for c in work["checks"]
        ],
        precondition_receipts=work["precondition_receipts"],
        duration_s=work["duration_s"],
        random_seed=7108219,
        source_artifact_hashes=work["refs"],
        raw_shard_hashes=work["raw_shard_hashes"],
        code_config_hashes=work["code_config_hashes"],
        measurement_clocks=work["clock"],
        phase_spans=[
            dict(phase="authenticate_and_reduce", **work["clock"], duration_s=work["duration_s"])
        ],
        measurement_reference=reference(raw / "measurement.json"),
        fixture_protocol_only=fixture,
        repository_health=work.get("global_health", {}),
        cited_upstream_artifacts=[
            dict(
                path=r["path"],
                sha256=r["sha256"],
                fields_imported=fields,
            )
            for operand, fields in [
                (
                    "experiment_8208_v709_restricted_energy_fit.json",
                    [
                        "required_checks_passed",
                        "flagged_adversarial",
                        "terminal_validation_sidecar_path",
                        "measurement_reference",
                        "role_manifest",
                    ],
                ),
                (
                    "experiment_8212_v709_memory_benefit_audit.json",
                    [
                        "required_checks_passed",
                        "flagged_adversarial",
                        "terminal_validation_sidecar_path",
                    ],
                ),
                (
                    Path(protocol["fit_measurement"]["path"]).name,
                    ["evidence.rows", "evidence.roles", "evidence.heads", "evidence.baseline"],
                ),
                (
                    Path(protocol["sealed_predictions"]["path"]).name,
                    [
                        "labels_opened",
                        "rows.unit_id",
                        "rows.source_cluster_id",
                        "rows.slot",
                        "rows.status",
                        "rows.exclusion_reason",
                    ],
                ),
            ]
            for r in protocol["source_artifact_hashes"]
            if Path(r["path"]).name == operand
        ],
        claim_scope="Frozen finite utility method and fit/tune descriptive residuals only; no benefit measurement.",
        methodology_note="Cached source annotations supply fit/tune residuals. No patch was fitted here. Reserved identities and original missing masks are target free. H1/H2 remain registered and unmeasured; exposed development supplies no independent generalization or distribution-free safety claim.",
    )
    value = normalize_artifact_for_template_write(value)
    value["field_principles"] = {
        k: "Bind frozen method readiness to authenticated current work; registration and historical exposure supply no scientific benefit."
        for k in value
    }
    value["field_principles"].update(
        utility_protocol_ready_score="One only for usable authenticated inputs, a frozen method and passing owned checks.",
        descriptive_utility_residuals="Fixed finite fit/tune witnesses describe empirical cost calibration; no adaptive evaluation selection.",
        trained_head_specs="Imported small trained heads are cached evidence, not current LLM loads or new head fits.",
        original_reserved_mask="Preserve every original source and missing slot without opening evaluation targets.",
        repository_health="Separate bounded global diagnostic; existing failures cannot become a claim of global green.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """A fresh process rehashes primitives and independently rebuilds the result."""
    try:
        value = json.loads(path.read_bytes())
        checksum = value.pop("reproducibility_checksum")
        if canonical_hash(value) != checksum:
            return False
        for ref in [
            value["measurement_reference"],
            *value["source_artifact_hashes"],
            *value["raw_shard_hashes"],
            *value["code_config_hashes"],
        ]:
            if sha256_file(Path(ref["path"])) != ref["sha256"] or (
                "snapshot_path" in ref and sha256_file(Path(ref["snapshot_path"])) != ref["sha256"]
            ):
                return False
        for receipt in [*value["validation_receipts"], *value["precondition_receipts"]]:
            for label in ["stdout", "stderr"]:
                if (
                    label + "_path" in receipt
                    and sha256_file(Path(receipt[label + "_path"])) != receipt[label + "_sha256"]
                ):
                    return False
        work = json.loads(Path(value["measurement_reference"]["path"]).read_bytes())
        raw = Path(value["terminal_validation_sidecar_path"]).parent
        if work["evidence"] != json.loads((raw / "fixture_primitives.json").read_bytes()):
            return False
        if work["diagnostics"]:
            if (
                work["protocol"] != PROTOCOL_VALUE
                or reduce_primitives(work["evidence"], work["protocol"]) != work["diagnostics"]
            ):
                return False
            upstream = next(
                r
                for r in work["refs"]
                if r.get("upstream_path") == work["protocol"]["fit_measurement"]["path"]
            )
            data = json.loads(Path(upstream["path"]).read_bytes())["evidence"]
            if any(work["evidence"][k] != data[k] for k in ["rows", "roles", "heads", "baseline"]):
                return False
        rebuilt = build(
            work, raw, value["validation_receipts"], fixture=value["fixture_protocol_only"]
        )
        return rebuilt == dict(value, reproducibility_checksum=checksum)
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False


def run_check(root: Path, spec: Json, private: Path, raw: Path, *, heartbeat_s: float = 20) -> Json:
    """Keep the measured coverage report after private validation scratch expires."""
    receipt = qualified.run_check(root, spec, private, raw, heartbeat_s=heartbeat_s)
    if spec["name"] == "coverage_json" and receipt["passed"]:
        report = Path(spec["argv"][spec["argv"].index("-o") + 1])
        target = raw / "changed_code_coverage.json"
        target.write_bytes(report.read_bytes())
        receipt["coverage_reference"] = reference(target)
    return receipt


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze exact owned commands and private scratch before measurement starts."""
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
    ):
        specs = BASE_MANIFEST(private, candidate)
    config = private / "coverage.ini"
    config.write_text(config.read_text() + "[report]\nexclude_lines =\n")
    for spec in specs["commands"]:
        if spec["name"] == "owned_unit_and_private_CLI":
            spec["argv"].append("--basetemp=" + str(private / "owned_pytest"))
            spec["deadline_s"] = 240
        if spec["name"] == "consumer_and_E2E015_019":
            start = spec["argv"].index("tests/python/test_development_methods_8098.py")
            spec["argv"][start:] = [
                "tests/python/test_primary_publication_7928.py",
                "tests/python/test_source_boundary_7852.py",
                "tests/python/test_experiment_7942_v689_sentence_labels.py",
                "tests/python/test_restricted_action_methods_8207.py",
                "--basetemp=" + str(private / "consumer_pytest"),
            ]
            spec["deadline_s"] = 240
        if spec["name"] == "strict_mypy":
            spec["argv"] = [
                a.replace("--follow-imports=silent", "--follow-imports=skip") for a in spec["argv"]
            ]
        if spec["name"] == "spec_coverage":
            spec["argv"].insert(-1, "--files")
    specs["repository_health"]["deadline_s"] = 180
    specs["repository_health"]["argv"][:0] = [
        "/usr/bin/env",
        "COVERAGE_FILE=" + str(private / "repository_health.coverage"),
    ]
    return specs


def main(argv: list[str] | None = None) -> int:
    """The qualified parent supervises normal exit and unchanged terminal checks."""
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
        patch.object(execution, "manifest", manifest),
        patch.object(execution, "run_check", run_check),
    ):
        return int(execution.main(argv))
