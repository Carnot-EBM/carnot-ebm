"""REQ-REPORT-8362: original bytes supply evidence, while proof limits remain visible."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import yaml

from carnot.reporting import spline_table_fidelity_8352 as old
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    ZERO_INVOCATION_COUNTS,
)
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.roadmap_contract import parse_design
from carnot.verify import threshold_guard_8362 as n
from carnot.verify import spline_table_fidelity_8352 as numeric

Json = dict[str, Any]
ROOT = old.ROOT
NAME = "experiment_8362_v721_threshold_guard"
TASK = "exp8362-threshold-guard"
CLI = "scripts/experiments/" + NAME + ".py"
TEST = "tests/python/test_threshold_guard_8362.py"
OWNED = [
    "python/carnot/verify/threshold_guard_8362.py",
    "python/carnot/reporting/threshold_guard_8362.py",
    "python/carnot/reporting/threshold_guard_execution_8362.py",
    CLI,
]
TASK_PIN = "sha256:00dab126853cdaf458f81aa2af1359b06833ffc1aae71fa4ef98f6fc90ed70de"
UPSTREAM = "results/experiment_8352_v720_spline_table_fidelity.json"
PINS = {
    **old.PINS,
    UPSTREAM: "sha256:d6c5cc72e8560ec3cd87c785f85eb66df789c1ee8622c8de828dd99a25a1e431",
}
OperandError, require, reference = old.OperandError, old.require, old.reference
MODEL_SPECS: list[Json] = []
PROTOCOL = old.PROTOCOL


def authenticate(root: Path, raw: Path) -> Json:
    """Use pinned successful head/table seals without replaying mutable historical authority."""
    refs: list[Json] = []

    def bind(path: Path, pin: str | None = None, parse: bool = True) -> Any:
        observed = sha256_file(path) if path.is_file() else None
        require(
            path, "sha256" if pin else "exists", pin or True, observed if pin else path.is_file()
        )
        destination = raw / "inputs" / (str(observed)[7:] + "-" + path.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        refs.append(dict(reference(destination), source_path=str(path), source_sha256=observed))
        return json.loads(destination.read_bytes()) if parse else destination

    require(
        root / "research-roadmap.yaml", "exists", True, (root / "research-roadmap.yaml").is_file()
    )
    active = yaml.safe_load((root / "research-roadmap.yaml").read_bytes())
    design = root / "openspec/change-proposals/research-roadmap-vNEXT.md"
    require(design, "exists", True, design.is_file())
    task = next(
        t for t in parse_design(design.read_text(), milestone="2026.10.721")[1] if t["id"] == TASK
    )
    require(design, "task_sha256", TASK_PIN, canonical_hash(task))
    require(
        root / "research-roadmap.yaml",
        "task",
        task,
        next((t for t in active["tasks"] if t["id"] == TASK), None),
    )
    require(root / "research-roadmap.yaml", "milestone", "2026.10.721", active.get("milestone"))
    bind(design, parse=False)
    bind(root / "research-roadmap.yaml", parse=False)
    bind(
        root / "openspec/change-proposals/research-roadmap-v720-preserved-20261009.md", parse=False
    )
    primary = bind(root / old.HEAD, PINS[old.HEAD])
    protocol = bind(root / old.PROTOCOL, PINS[old.PROTOCOL])
    checkpoint = bind(Path(primary["checkpoint_hashes"][0]["path"]), old.CHECKPOINT)
    table_primary = bind(root / UPSTREAM, PINS[UPSTREAM])
    for path, value in [(root / old.HEAD, primary), (root / UPSTREAM, table_primary)]:
        terminal = bind(Path(value["terminal_validation_sidecar_path"]))
        sidepath = Path(terminal["publication"]["sidecar_path"])
        try:
            side = read_bound_sidecar(path, sidepath)
        except (OSError, ValueError):
            require(sidepath, "bound_terminal_sidecar", True, False)
        bind(sidepath)
        require(sidepath, "report.passed", True, side["report"].get("passed"))
        require(
            path,
            "qualified_terminal",
            [True, False, "circular_positive" if path.name == Path(UPSTREAM).name else "null"],
            [
                value.get("required_checks_passed"),
                value.get("flagged_adversarial"),
                value.get("verdict_class"),
            ],
        )
    selected = table_primary["selected_candidate"]
    require(root / UPSTREAM, "configuration_id", "65-int16-linear", selected["configuration_id"])
    values = np.load(
        bind(
            Path(selected["table_reference"]["path"]), selected["table_reference"]["sha256"], False
        ),
        allow_pickle=False,
    )
    head = next(h for h in checkpoint["heads"] if h["arm"] == "spline34")
    require(root / UPSTREAM, "table_bytes", 520, values.nbytes)
    require(
        root / UPSTREAM,
        "table_reconstruction",
        True,
        np.array_equal(values, numeric.table(head, 65, "int16")[0]),
    )
    require(root / old.PROTOCOL, "knots", numeric.kernel.KNOTS, protocol["head"]["knots"])
    historical_work = bind(
        Path(table_primary["work_reference"]["path"]), table_primary["work_reference"]["sha256"]
    )
    panel_ref = next(
        r for r in historical_work["primitive_refs"] if Path(r["path"]).name == "panel.npy"
    )
    frozen_panel = np.load(
        bind(Path(panel_ref["path"]), panel_ref["sha256"], False), allow_pickle=False
    )
    require(
        root / UPSTREAM, "frozen_panel", True, np.array_equal(frozen_panel, numeric.panel(head)[0])
    )
    for path in [
        "python/carnot/verify/spline_table_fidelity_8352.py",
        "python/carnot/verify/local_update_isolation_8306.py",
        "python/carnot/experiment_7425_v651_spline_prototype.py",
    ]:
        bind(root / path, parse=False)
    return dict(
        head=head,
        table=values.tolist(),
        refs=refs,
        task=task,
        protocol=protocol,
        historical_model_provenance=primary["historical_model_provenance"],
    )


def reconstruct(inputs: Json) -> Json:
    """Write primitives before checks; constructed updates reuse the existing sparse refresh."""
    head = inputs["head"]
    table: n.Array = np.asarray(inputs["table"], dtype="<i2")
    x, kinds = n.panel(head)
    n.progress("before_benchmark_guard")
    rows = n.evaluate(head, table, x, kinds)
    n.progress("after_benchmark_guard", len(rows), 0)
    cert = n.certificate(head, table)
    controls = []
    under = deepcopy(cert)
    under["cells"][0][0]["logit_error_bound"] = 0
    for name, data, supplied in [
        ("underbound", table, under),
        ("swapped", table[::-1], cert),
        ("saturated", np.full((4, 65), 32767, dtype="<i2"), cert),
    ]:
        row = n.guard(head, x[-1], data, supplied)
        controls.append(dict(name=name, **row))
    n.progress("before_sparse_refresh")
    refresh = []
    current, values = head, table
    for index, vector in enumerate(numeric.events()):
        updated = numeric.update(current, vector, index % 2)
        sparse, entries = numeric.refresh(current, updated, values, "int16")
        full, _ = numeric.table(updated, 65, "int16")
        refresh.append(
            dict(
                index=index,
                x=vector.tolist(),
                target=index % 2,
                coefficients=updated["coefficients"],
                affected_entries=entries,
                identical_bytes=sparse.tobytes() == full.tobytes(),
            )
        )
        current, values = updated, sparse
    n.progress("after_sparse_refresh", len(refresh), 0)
    return dict(rows=rows, certificate=cert, controls=controls, refresh=refresh)


def measure(inputs: Json, raw: Path) -> Json:
    """Persist measured primitives before qualification opens any validation results."""
    began = time.monotonic_ns()
    primitives = reconstruct(inputs)
    atomic_json(raw / "primitives.json", primitives)
    return dict(
        inputs=inputs,
        configurations=[],
        primitive_refs=[reference(raw / "primitives.json")],
        numeric_summary=reduce(primitives),
        phase_spans=[
            dict(
                phase="numeric_measurement",
                started_monotonic_ns=began,
                ended_monotonic_ns=time.monotonic_ns(),
            )
        ],
    )


def reduce(primitives: Json) -> Json:
    """Only actual certified fast paths count toward useful deployment."""
    rows = primitives["rows"]
    random = [r for r in rows if r["kind"] == "random"]
    escapes = sum(
        not r["probability_interval"][0] <= r["direct_probability"] <= r["probability_interval"][1]
        for r in rows
    )
    mismatch = sum(r["guarded_action"] != r["direct_action"] for r in rows)
    flips = sum(r["direct_action"] != r["table_action"] for r in rows[:4174])
    return dict(
        interval_escape_count=escapes,
        guarded_action_mismatch_count=mismatch,
        original_table_flip_count=flips,
        direct_reference_probability_error_max=max(
            abs(r["reference_probability"] - r["direct_probability"]) for r in rows
        ),
        vector_count=len(rows),
        probability_error_max=max(
            abs(r["table_probability"] - r["direct_probability"]) for r in rows
        ),
        fast_path_fraction=sum(r["fast_path"] for r in random) / len(random),
        empirical_candidate_fast_path_fraction=sum(r["empirical_fast_path"] for r in random)
        / len(random),
        fallback_counts=dict(Counter(r["fallback_reason"] for r in rows)),
        spline_proof_passed=primitives["certificate"]["spline_proof_passed"],
        independent_extrema_passed=primitives["certificate"]["independent_extrema_passed"],
        numerical_policy_proof_passed=primitives["certificate"]["numerical_policy_proof_passed"],
        refresh_passed=all(r["identical_bytes"] for r in primitives["refresh"]),
        deliberate_controls_passed=all(
            not r["fast_path"] and r["fallback_reason"] != "unproven_numerical_policy"
            for r in primitives["controls"]
        ),
    )


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Qualified empirical evidence cannot promote an incomplete numerical proof."""
    inputs, summary = work.get("inputs", {}), work.get("numeric_summary", {})
    plan = json.loads(Path(work["manifest_reference"]["path"]).read_bytes())["commands"]
    executed = {r["name"]: r for r in receipts}
    covered = work.get("coverage", {})
    owned = bool(receipts) and all(
        r["passed"] and r["exit_code"] == r["expected_exit"] and not r["timed_out"]
        for r in receipts
    )
    owned = owned and all(
        s["name"] in executed and executed[s["name"]]["argv"] == s["argv"] for s in plan
    )
    owned = (
        owned
        and covered.get("totals", {}).get("percent_covered") == 100
        and all(
            p in covered.get("files", {}) and covered["files"][p]["summary"]["missing_lines"] == 0
            for p in OWNED
        )
    )
    numeric_ok = (
        bool(summary)
        and summary["interval_escape_count"] == summary["guarded_action_mismatch_count"] == 0
        and summary["original_table_flip_count"] == 3
        and summary["refresh_passed"]
        and summary["deliberate_controls_passed"]
        and summary["independent_extrema_passed"]
        and summary["direct_reference_probability_error_max"] <= 1e-10
    )
    failures = work.get("failures", [])
    kind = (
        ("disqualified" if any(not r["passed"] for r in receipts) else "blocked")
        if failures
        else "disqualified"
        if not owned or not numeric_ok
        else "null"
    )
    ready = (
        owned
        and numeric_ok
        and not failures
        and summary.get("numerical_policy_proof_passed", False)
    )
    primitive = (
        json.loads(Path(work["primitive_refs"][0]["path"]).read_bytes())
        if work.get("primitive_refs")
        else {}
    )
    value: Json = dict(
        experiment_id=8362,
        task_id=TASK,
        milestone="2026.10.721",
        run_date="20261010",
        honest_verdict="complete_" + kind + "_empirical_threshold_guard",
        verdict_class=kind,
        gate_check_summary=[
            *failures,
            *[
                dict(
                    upstream="owned_validation",
                    path=r["stdout_path"],
                    hash=r["stdout_sha256"],
                    field=r["name"],
                    operator="==",
                    expected=r["expected_exit"],
                    observed=r["exit_code"],
                )
                for r in receipts
                if not r["passed"]
            ],
        ],
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        no_model_load=True,
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        historical_model_provenance=inputs.get("historical_model_provenance", []),
        rows=[dict(configuration_id="65-int16-linear-direct-fallback", **summary)]
        if summary
        else [],
        intended_count=4218,
        completed_count=summary.get("vector_count", 0),
        failed_count=summary.get("guarded_action_mismatch_count", 0),
        censored_count=4218 - summary.get("vector_count", 0),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            frozen_random_vectors=4096,
            original_vectors=4174,
            nextafter_vectors=44,
            local_updates=64,
            independent_natural_sources=0,
        ),
        verifier_is_oracle=True,
        exposure_scope="constructed_label_free_vectors_from_exposed_development_head",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=bool(owned),
        flagged_adversarial=any(not r["passed"] for r in receipts if r["name"] == "adversarial"),
        acceptance_gates=dict(
            authenticated_inputs=not failures,
            owned_checks=owned,
            empirical_validation=bool(numeric_ok),
            complete_arithmetic_proof=bool(ready),
            useful=bool(ready and summary["fast_path_fraction"] >= 0.5),
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / output.stem / "terminal_validation.json"
        ),
        adversarial_findings=work.get("adversarial_findings", []),
        preconditions_checked=work.get("preconditions", {}),
        duration_s=work.get("duration_s", 0.0),
        phase_spans=work.get("phase_spans", [])
        + [
            dict(
                phase=r["name"],
                started_monotonic_ns=r["started_monotonic_ns"],
                ended_monotonic_ns=r["ended_monotonic_ns"],
            )
            for r in receipts
        ],
        random_seed=numeric.SEED,
        source_artifact_hashes={r["source_path"]: r["sha256"] for r in inputs.get("refs", [])},
        code_config_hashes={r["path"]: r["sha256"] for r in work.get("code_refs", [])},
        raw_shard_hashes={r["path"]: r["sha256"] for r in work.get("primitive_refs", [])},
        cited_upstream_artifacts=inputs.get("refs", []),
        guard_ready_score=int(ready),
        guard_useful_score=int(ready and summary["fast_path_fraction"] >= 0.5),
        table_candidate_score=int(ready),
        bound_derivation=primitive.get("certificate", {}),
        assumptions=primitive.get("certificate", {}).get("unresolved_assumptions", []),
        per_vector_rows=work.get("primitive_refs", []),
        work_reference=reference(raw / "measurement.json"),
        methodology_note="Analytic spline remainder uses exact rational extrema, not sampled error. Float64 expit has no authenticated universal rounding contract. Intervals are empirical only; all actual actions use unchanged direct fallback. Three old flips are a positive control. Zero disagreement is numerical policy preservation, never language truth, utility or generalization.",
        **summary,
    )
    defaults: Json = dict(
        interval_escape_count=None,
        guarded_action_mismatch_count=None,
        fallback_counts={},
        fast_path_fraction=None,
        probability_error_max=None,
    )
    for field, default in defaults.items():
        value.setdefault(field, default)
    value["field_principles"] = {
        k: "Bind "
        + k
        + " to authenticated original bytes, replayed numeric evidence and explicit empirical proof limits."
        for k in value
    }
    value["field_principles"].update(
        fast_path_fraction="Actual certified fast paths on 4096 frozen random vectors; unresolved proof forces zero.",
        empirical_candidate_fast_path_fraction="Potential coverage of empirical intervals, without certification or deployment permission.",
        bound_derivation="Exact binary rational spline proof and independent piecewise-polynomial extrema; numerical sigmoid proof remains unresolved.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Recompute primitive meaning so repaired hashes cannot authorize changed actions."""
    try:
        value = json.loads(path.read_bytes())
        ref = value["work_reference"]
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            return False
        work = json.loads(Path(ref["path"]).read_bytes())
        raw = Path(ref["path"]).parent
        refs = [
            *work.get("inputs", {}).get("refs", []),
            *work.get("code_refs", []),
            *work.get("primitive_refs", []),
            work["manifest_reference"],
        ]
        if any(sha256_file(Path(r["path"])) != r["sha256"] for r in refs):
            return False
        for receipt in value["validation_receipts"]:
            for label in ["stdout", "stderr"]:
                if sha256_file(Path(receipt[label + "_path"])) != receipt[label + "_sha256"]:
                    return False
        if work.get("numeric_summary"):
            inputs = work["inputs"]
            copies = {r["source_sha256"]: Path(r["path"]) for r in inputs["refs"]}
            if any(sha256_file(copies[pin]) != pin for pin in [*PINS.values(), old.CHECKPOINT]):
                return False
            checkpoint = json.loads(copies[old.CHECKPOINT].read_bytes())
            head = next(h for h in checkpoint["heads"] if h["arm"] == "spline34")
            if (
                inputs["head"] != head
                or canonical_hash(inputs["task"]) != TASK_PIN
                or inputs["protocol"] != json.loads(copies[PINS[old.PROTOCOL]].read_bytes())
            ):
                return False
            activated = yaml.safe_load(
                next(
                    Path(r["path"])
                    for r in inputs["refs"]
                    if Path(r["source_path"]).name == "research-roadmap.yaml"
                ).read_bytes()
            )
            if (
                activated.get("milestone") != "2026.10.721"
                or next(t for t in activated["tasks"] if t["id"] == TASK) != inputs["task"]
            ):
                return False
            tasks = parse_design(
                next(
                    Path(r["path"])
                    for r in inputs["refs"]
                    if Path(r["source_path"]).name == "research-roadmap-vNEXT.md"
                ).read_text(),
                milestone="2026.10.721",
            )[1]
            if inputs["task"] != next(t for t in tasks if t["id"] == TASK) or not np.array_equal(
                np.asarray(inputs["table"], dtype="<i2"), numeric.table(head, 65, "int16")[0]
            ):
                return False
            primitives = json.loads(Path(work["primitive_refs"][0]["path"]).read_bytes())
            if primitives != reconstruct(inputs) or work["numeric_summary"] != reduce(primitives):
                return False
        output = Path(value["terminal_validation_sidecar_path"]).parents[2] / (NAME + ".json")
        return bool(value == build(work, value["validation_receipts"], raw, output))
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False


def write_note(value: Json, path: Path) -> None:
    """Keep proof limits beside measured outcomes before readers can cite them."""
    path = path.with_name("v721-threshold-error-bound.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Threshold error bound, 2026-10-10\n\n"
        "The unchanged 65-point signed-int16 table uses 520 bytes and eleven fractional bits.\n"
        "Cubic second derivatives are piecewise linear. Exact binary-rational recursion gives their extrema at split-piece endpoints.\n"
        "Each whole-cell bound is h² sup|f″|/8 plus 1/4096 rounding and an outward basic-arithmetic allowance.\n"
        "The basis error recurrence is e[k] <= 12 e[k-1] + 64u, e[0]=0, u=2^-53. Widths exceed .19.\n"
        "Three stages give 10048u. Both endpoint and direct dot arithmetic are included. SciPy PPoly checks extrema independently.\n"
        "Four cell intervals add before unchanged slope, intercept and temperature arithmetic. Monotone sigmoid maps the empirical logit endpoints.\n"
        "No all-input error contract was authenticated for unchanged SciPy expit and platform exp. The 1e-12 sigmoid allowance is empirical.\n"
        "This prevents complete numerical certification. The deployed guard always returns the original direct float64 action.\n"
        "Strict regions are accept below .25, reject above .75 and escalate on [.25,.75]. Threshold intersections fall back.\n"
        f"Verdict: {value['honest_verdict']}. Original flips: {value.get('original_table_flip_count')}.\n"
        f"Observed interval escapes: {value['interval_escape_count']}; guarded mismatches: {value['guarded_action_mismatch_count']}.\n"
        f"Guard readiness: {value['guard_ready_score']}; usefulness: {value['guard_useful_score']}; actual fast-path fraction: {value['fast_path_fraction']}.\n"
        f"Empirical candidate fraction: {value.get('empirical_candidate_fast_path_fraction')}. This fraction grants no deployment permission.\n"
        "Every vector, interval, action, fault and sparse refresh remains in byte-bound raw primitives.\n"
        "Cached heads are exposed development. Current model calls and both generalization scores are zero.\n"
        "Next evidence: authenticate a universal error contract for the unchanged numeric sigmoid before enabling any fast result.\n"
    )
