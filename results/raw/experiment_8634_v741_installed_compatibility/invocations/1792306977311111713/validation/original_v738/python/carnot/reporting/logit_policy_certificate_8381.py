"""REQ-REPORT-8381: bind an isolated mathematical policy to sealed operands."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import yaml

from carnot.reporting import threshold_guard_8362 as old
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    ZERO_INVOCATION_COUNTS,
)
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.verify import dyadic_logit_v1 as n
from carnot.verify import threshold_guard_8362 as numeric_previous

Json = dict[str, Any]
ROOT = old.ROOT
NAME = "experiment_8381_v722_logit_policy_certificate"
TASK = "exp8381-logit-policy-certificate"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_logit_policy_certificate_8381.py"
OWNED = [
    "python/carnot/verify/dyadic_logit_v1.py",
    "python/carnot/reporting/logit_policy_certificate_8381.py",
    "python/carnot/reporting/logit_policy_execution_8381.py",
    CLI,
]
TASK_PIN = "sha256:4608d3d60d35bd0db2aa7f61278fad3c4a63ab94332a6e059844e8649145b056"
UPSTREAM = "results/experiment_8362_v721_threshold_guard.json"
DELAYED = "results/experiment_8348_v720_continuous_local_learning.json"
METHODS = "results/experiment_8374_v722_contract_methods.json"
PROTOCOL = old.PROTOCOL
PINS = {
    UPSTREAM: "sha256:4d8d4fc193edc5856dbd3ddbb7d4009407261b3001a59238374d7c5126de7f56",
    DELAYED: "sha256:c98cbd678aef6aa5055cd7033a2c36404dc8d20337a5e601bd90b2d7dba00f07",
    METHODS: "sha256:04f800406dfea9de086787cb64fb18722b3ce110367446df75ba982429dd715e",
    PROTOCOL: old.PINS[PROTOCOL],
    "openspec/change-proposals/v721-deployment-protocol.json": "sha256:d4441f7619a1d349a4958038af93c36f2c4df97b31fd32278c4ea6def7c7b4eb",
}
OperandError, require, reference = old.OperandError, old.require, old.reference
MODEL_SPECS: list[Json] = []


def authenticate(root: Path, raw: Path) -> Json:
    """Pinned qualified producers supply operands, not their historical policy readiness."""
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

    active_path = bind(root / "research-roadmap.yaml", parse=False)
    active = yaml.safe_load(active_path.read_bytes())
    task: Json = next((t for t in active["tasks"] if t["id"] == TASK), {})
    require(root / "research-roadmap.yaml", "task_sha256", TASK_PIN, canonical_hash(task))
    require(root / "research-roadmap.yaml", "milestone", "2026.10.722", active.get("milestone"))
    bind(root / "openspec/change-proposals/research-roadmap-vNEXT.md", parse=False)
    bind(
        root / "openspec/change-proposals/research-roadmap-v721-preserved-20261010.md", parse=False
    )
    bind(root / "ops/exclusion_manifest.yaml", parse=False)
    imported = {p: bind(root / p, pin) for p, pin in PINS.items()}
    for p in [UPSTREAM, DELAYED, METHODS]:
        value = imported[p]
        terminal = bind(Path(value["terminal_validation_sidecar_path"]))
        try:
            side = read_bound_sidecar(root / p, Path(terminal["publication"]["sidecar_path"]))
        except (OSError, ValueError):
            require(root / p, "bound_terminal_sidecar", True, False)
        bind(Path(terminal["publication"]["sidecar_path"]))
        require(
            root / p,
            "qualified_terminal",
            [True, False, True],
            [
                value.get("required_checks_passed"),
                value.get("flagged_adversarial"),
                side["report"]["passed"],
            ],
        )
    require(
        root / METHODS,
        "direct_inputs_ready_score",
        1,
        imported[METHODS].get("direct_inputs_ready_score"),
    )
    ref = imported[UPSTREAM]["work_reference"]
    prior = bind(Path(ref["path"]), ref["sha256"])
    inputs = prior["inputs"]
    for r in inputs["refs"]:
        bind(Path(r["path"]), r["sha256"], parse=False)
    checkpoint = next(r for r in inputs["refs"] if r["sha256"] == old.old.CHECKPOINT)
    head = next(
        h
        for h in json.loads(Path(checkpoint["path"]).read_bytes())["heads"]
        if h["arm"] == "spline34"
    )
    require(root / UPSTREAM, "original_head", head, inputs["head"])
    table: numeric_previous.Array = np.asarray(inputs["table"], dtype="<i2")
    require(
        root / UPSTREAM,
        "table_reconstruction",
        True,
        bool(np.array_equal(table, numeric_previous.old.table(head, 65, "int16")[0])),
    )
    ref = imported[DELAYED]["measurement_reference"]
    trajectory = bind(Path(ref["path"]), ref["sha256"])
    updates = [
        r["coefficients"] for r in trajectory["state"]["updates"] if r["arm"] == "online_sparse"
    ]
    return dict(
        head={k: head[k] for k in ["coefficients", "temperature"]},
        table=inputs["table"],
        refreshed_coefficients=updates,
        task=task,
        refs=refs,
        historical_model_provenance=[
            dict(
                producer_path=UPSTREAM,
                producer_sha256=PINS[UPSTREAM],
                scope="cached_exposed_development_zero_current_calls",
            )
        ],
    )


def reconstruct(inputs: Json) -> Json:
    """Refresh actual delayed coefficients without rerunning their closed utility procedure."""
    head = inputs["head"]
    table: numeric_previous.Array = np.asarray(inputs["table"], dtype="<i2")
    n.progress("before_benchmark_dyadic")
    rows = n.evaluate(head, table)
    n.progress("after_benchmark_dyadic", len(rows), 0)
    cert = n.certificate(head, table)
    under = deepcopy(cert)
    under["cells"][0][0] = "0"
    controls = []
    x = [F(0), *[F(1, 2)] * 4]
    for name, data, supplied, h in [
        ("incorrect_bound", table, under, head),
        ("changed_rounding", table, dict(cert, rounding="nearest"), head),
        ("stale_version", table, dict(cert, version="old"), head),
        ("swapped_table", table[::-1], cert, head),
        ("saturated", np.full((4, 65), 32767, dtype="<i2"), cert, head),
        ("changed_head", table, cert, dict(head, temperature=3)),
    ]:
        controls.append(dict(name=name, **n.guard(h, x, data, supplied)))
    refresh = []
    n.progress("before_benchmark_delayed_refresh")
    for i, coefficients in enumerate(inputs["refreshed_coefficients"]):
        h = dict(head, coefficients=coefficients)
        data, saturation = numeric_previous.old.table(h, 65, "int16")
        supplied = n.certificate(h, data)
        row = n.guard(h, x, data, supplied)
        z = n.exact(h, x, polynomial=True)
        refresh.append(
            dict(
                index=i,
                head=h,
                table=data.tolist(),
                certificate=supplied,
                saturation=saturation,
                reference_logit=str(z),
                reference_action=n.region(z, z),
                interval_escape=bool(
                    row["interval"] and not F(row["interval"][0]) <= z <= F(row["interval"][1])
                ),
                **row,
            )
        )
        if (i + 1) % 8 == 0:
            n.progress("delayed_refresh", i + 1, len(inputs["refreshed_coefficients"]) - i - 1)
    n.progress("after_benchmark_delayed_refresh", len(refresh), 0)
    return dict(rows=rows, certificate=cert, controls=controls, refresh=refresh)


def reduce(primitives: Json) -> Json:
    """Usefulness cannot repair any escaped interval or mismatched action."""
    rows = primitives["rows"]
    random = [r for r in rows if r["kind"] == "random"]
    return dict(
        vector_count=len(rows),
        interval_escape_count=sum(r["interval_escape"] for r in rows + primitives["refresh"]),
        action_mismatch_count=sum(
            r["action"] != r["reference_action"] for r in rows + primitives["refresh"]
        ),
        fast_path_fraction=sum(r["fast_path"] for r in random) / len(random),
        original_policy_difference_rows=[r for r in rows if r["action"] != r["original_action"]],
        deliberate_controls_passed=all(not r["fast_path"] for r in primitives["controls"]),
        refresh_passed=bool(primitives["refresh"])
        and all(
            r["action"] == r["reference_action"]
            and not r["saturation"]
            and not r["interval_escape"]
            and r["certificate"]["independent_remainder_audit"]
            for r in primitives["refresh"]
        ),
        independent_remainder_audit=primitives["certificate"]["independent_remainder_audit"],
    )


def measure(inputs: Json, raw: Path) -> Json:
    """Write primitive evidence before validation outcomes can influence qualification."""
    began = time.monotonic_ns()
    primitives = reconstruct(inputs)
    atomic_json(raw / "primitives.json", primitives)
    health = ROOT / "results/raw" / NAME / "repository_health/full_python_suite_once.receipt.json"
    return dict(
        repository_health_reference=reference(health) if health.is_file() else None,
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


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Qualification requires actual owned coverage, numeric safety and every frozen command."""
    summary, inputs = work.get("numeric_summary", {}), work.get("inputs", {})
    plan = json.loads(Path(work["manifest_reference"]["path"]).read_bytes())["commands"]
    executed = {r["name"]: r for r in receipts}
    coverage = work.get("coverage", {})
    owned = bool(receipts) and all(
        r["passed"] and r["exit_code"] == r["expected_exit"] and not r["timed_out"]
        for r in receipts
    )
    owned = owned and all(
        p["name"] in executed and executed[p["name"]]["argv"] == p["argv"] for p in plan
    )
    owned = (
        owned
        and coverage.get("totals", {}).get("percent_covered") == 100
        and all(
            coverage.get("files", {}).get(p, {}).get("summary", {}).get("missing_lines") == 0
            for p in OWNED
        )
    )
    safe = (
        bool(summary)
        and summary["interval_escape_count"] == summary["action_mismatch_count"] == 0
        and all(
            summary[k]
            for k in ["deliberate_controls_passed", "refresh_passed", "independent_remainder_audit"]
        )
    )
    failures = work.get("failures", [])
    kind = (
        "blocked"
        if failures and not any(not r["passed"] for r in receipts)
        else "disqualified"
        if not owned or not safe
        else "circular_positive"
    )
    ready = bool(owned and safe and not failures)
    primitive = (
        json.loads(Path(work["primitive_refs"][0]["path"]).read_bytes())
        if work.get("primitive_refs")
        else {}
    )
    rows = [
        dict(
            r,
            arm=n.VERSION,
            status="completed",
            absolute_metric=int(not r["interval_escape"] and r["action"] == r["reference_action"]),
            missing_reason=None,
        )
        for r in primitive.get("rows", [])
    ]
    if not rows:
        rows = [
            dict(
                vector_id=i,
                arm=n.VERSION,
                status="unstarted",
                absolute_metric=None,
                missing_reason="external_operand_absent",
                censored=True,
            )
            for i in range(4166)
        ]
    value: Json = dict(
        invocation_id=raw.name,
        task_authority_sha256=canonical_hash(inputs["task"]) if inputs else None,
        repository_health_reference=work.get("repository_health_reference"),
        experiment_id=8381,
        task_id=TASK,
        milestone="2026.10.722",
        run_date="20261010",
        honest_verdict="complete_" + kind + "_isolated_dyadic_logit_certificate",
        verdict_class=kind,
        gate_check_summary=[
            *[dict(f, check=f.get("check", f.get("field", "external_input"))) for f in failures],
            *[
                dict(
                    check=r["name"],
                    upstream="owned_validation",
                    path=r["stdout_path"],
                    hash=r["stdout_sha256"],
                    field="exit_code",
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
        rows=rows,
        intended_count=4166,
        completed_count=summary.get("vector_count", 0),
        failed_count=summary.get("action_mismatch_count", 0),
        censored_count=4166 - summary.get("vector_count", 0),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            seeded_label_free_vectors=4096,
            knot_and_neighbor_vectors=64,
            exact_threshold_and_neighbor_vectors=6,
            delayed_refresh_count=len(inputs.get("refreshed_coefficients", [])),
            independent_natural_sources=0,
        ),
        verifier_is_oracle=True,
        exposure_scope="constructed_label_free_panel_from_exposed_development_head",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=bool(owned),
        flagged_adversarial=any(not r["passed"] for r in receipts if r["name"] == "adversarial"),
        acceptance_gates=dict(
            authenticated_inputs=not failures,
            owned_validation=bool(owned),
            numeric_safety=bool(safe),
            certificate_ready=ready,
            useful=bool(ready and summary.get("fast_path_fraction", 0) >= 0.5),
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / output.stem / "terminal_validation.json"
        ),
        adversarial_findings=work.get("adversarial_findings", []),
        preconditions_checked=dict(work.get("preconditions", {}), **inputs.get("resources", {})),
        duration_s=work.get("duration_s", 0),
        phase_spans=work.get("phase_spans", [])
        + [
            dict(
                phase=r["name"],
                started_monotonic_ns=r["started_monotonic_ns"],
                ended_monotonic_ns=r["ended_monotonic_ns"],
            )
            for r in receipts
        ],
        random_seed=numeric_previous.old.SEED,
        source_artifact_hashes={r["source_path"]: r["sha256"] for r in inputs.get("refs", [])},
        code_config_hashes={r["path"]: r["sha256"] for r in work.get("code_refs", [])},
        raw_shard_hashes={r["path"]: r["sha256"] for r in work.get("primitive_refs", [])},
        cited_upstream_artifacts=inputs.get("refs", []),
        action_certificate_ready_score=int(ready),
        useful_score=int(ready and summary.get("fast_path_fraction", 0) >= 0.5),
        table_candidate_score=int(ready),
        new_policy_id=n.VERSION,
        exact_threshold=dict(hex=n.L_HEX, rational=str(n.L)),
        proof_assumptions=[
            "Original binary64 coefficients, knots and positive temperature interpreted as exact rationals.",
            "Cubic C2 spline remainder h^2 sup|f''|/8; independent rational polynomial curvature audit.",
            "Exact measured endpoint residuals audit int16 construction; no assumed floating error allowance.",
            n.ROUNDING,
            "All global multiplication, interpolation, addition and division use Fraction; one nextafter widens each finite binary64 endpoint.",
            "Exact rational holistic threshold witnesses extend the binary64 random panel; old-policy comparisons use their binary64 projection.",
        ],
        production_migration_authorized=False,
        calibrated_confidence_certified=False,
        default_enabled=False,
        direct_service_gate=False,
        arc_gate=False,
        confidence_scope="SciPy probabilities and Brier retain empirical scope; no semantic truth certification.",
        work_reference=reference(raw / "measurement.json"),
        interval_escape_count=summary.get("interval_escape_count"),
        action_mismatch_count=summary.get("action_mismatch_count"),
        original_policy_difference_rows=summary.get("original_policy_difference_rows", []),
        fast_path_fraction=summary.get("fast_path_fraction"),
        bound_derivation=primitive.get("certificate", {}),
        methodology_note="Independent rational polynomial checks an exact new action policy. Oracle-checked numerical correctness grants no semantic benefit or production migration. Missing inputs are absent, not measured zero.",
    )
    value["field_principles"] = {
        k: "Bind "
        + k
        + " to sealed operands and independently replayed evidence; action correctness grants no semantic or deployment claim."
        for k in value
    }
    value["field_principles"].update(
        fast_path_fraction="Actual certified random-panel actions divided by 4096; exact fallback is not a fast action.",
        original_policy_difference_rows="Every observed migration difference is retained; SciPy history remains unchanged.",
        action_certificate_ready_score="Owned qualification of only dyadic_logit_v1, with zero certified interval escapes and action disagreements.",
        useful_score="Safety-qualified random fast coverage >=0.5; never changes the safety threshold.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Reconstruct numerical meaning so even repaired primitive hashes cannot grant readiness."""
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
            *(
                [work["repository_health_reference"]]
                if work.get("repository_health_reference")
                else []
            ),
        ]
        if any(
            sha256_file(Path(r["path"])) != r["sha256"]
            or r.get("source_sha256", r["sha256"]) != r["sha256"]
            for r in refs
        ):
            return False
        for r in value["validation_receipts"]:
            if any(
                sha256_file(Path(r[k + "_path"])) != r[k + "_sha256"] for k in ["stdout", "stderr"]
            ):
                return False
        if work.get("numeric_summary"):
            inputs = work["inputs"]
            copies = {r["source_sha256"]: Path(r["path"]) for r in inputs["refs"]}
            if any(pin not in copies or sha256_file(copies[pin]) != pin for pin in PINS.values()):
                return False
            prior = json.loads(copies[PINS[UPSTREAM]].read_bytes())
            prior_work = json.loads(copies[prior["work_reference"]["sha256"]].read_bytes())
            head = prior_work["inputs"]["head"]
            delayed = json.loads(copies[PINS[DELAYED]].read_bytes())
            trajectory = json.loads(copies[delayed["measurement_reference"]["sha256"]].read_bytes())
            active = yaml.safe_load(
                next(
                    Path(r["path"])
                    for r in inputs["refs"]
                    if r["source_path"] == str(ROOT / "research-roadmap.yaml")
                ).read_bytes()
            )
            task = next(t for t in active["tasks"] if t["id"] == TASK)
            if (
                canonical_hash(task) != TASK_PIN
                or task != inputs["task"]
                or inputs["head"] != {k: head[k] for k in ["coefficients", "temperature"]}
                or inputs["table"] != prior_work["inputs"]["table"]
                or inputs["refreshed_coefficients"]
                != [
                    r["coefficients"]
                    for r in trajectory["state"]["updates"]
                    if r["arm"] == "online_sparse"
                ]
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
    """Keep mathematical assumptions and migration limits beside measured evidence."""
    path = path.with_name("v722-logit-policy-certificate.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Isolated dyadic logit certificate, 2026-10-10\n\n"
        + "Policy dyadic_logit_v1 is default-off. It compares exact rational logits with +/-"
        + n.L_HEX
        + ". Equality escalates. This changes the mathematical policy.\n\n"
        + "\n".join(value["proof_assumptions"])
        + "\n\n"
        + f"Verdict: {value['honest_verdict']}. Certified escapes: {value['interval_escape_count']}; action disagreements: {value['action_mismatch_count']}; random fast fraction: {value['fast_path_fraction']}. Old-policy differences: {len(value['original_policy_difference_rows'])}.\n\n"
        + "Exact interpolation and global arithmetic avoid unproved floating operation allowances. Int16 endpoint residuals include actual table construction error. Piecewise rational polynomials independently check curvature and reference actions. Overflow and malformed certificates use exact fallback. The certificate makes no speed claim.\n\nConfidence probabilities and Brier remain empirical. No calibrated confidence, semantic truth, generalization, direct-service gate, ARC gate or production migration is authorized. V717/V721 protocol bytes and historical results remain unchanged.\n"
    )
