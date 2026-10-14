"""REQ-REPORT-8513: stronger controls require original custody, not a new panel split.

The shipped evaluator retains its loss definitions and prediction seal order.
These adapters add descriptive raw-score differences and future comparator slots.
They do not train a head or turn exposed labels into independent evidence.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import signal
import time
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import patch

from carnot.reporting import v726_criterion_decision_audit as shipped
from carnot.reporting import v730_source_information_audit as history
from carnot.reporting import v732_contract_methods as methods
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import read_bound_sidecar, reader_receipt
from carnot.verify import v730_information_metrics as n
from carnot.verify import v730_source_control_fit as fit

Json = dict[str, Any]
ROOT = methods.ROOT
NAME, TASK = "experiment_8513_v732_strongest_baseline_audit", "exp8513-strongest-baseline-audit"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v732_strongest_baseline_8513.py"
OWNED = [
    "python/carnot/reporting/v732_strongest_baseline_audit.py",
    "python/carnot/reporting/v732_strongest_baseline_runner.py",
    CLI,
]
UPSTREAM, UPSTREAM_PIN = history.UPSTREAM, history.UPSTREAM_PIN
AUDIT, AUDIT_PIN = (
    "results/experiment_8488_v730_source_information_audit.json",
    "sha256:64a811a3b880da8446f4bc37bacdef23d00f4558c0d348bb0994e69e583d07cd",
)
METHODS_PIN = "sha256:d8aad6fcd8662fe8abc98acaa95add49df44b3b5359700ab09bbaafee3114eec"
COMPARATORS = (
    "raw_p2",
    "legacy_full",
    "identity_brier_scalar",
    "identity_brier_full",
    "identity_brier_permuted",
)
MODEL_SPECS: list[Json] = []
read, reference, checksum, sha256_file = (
    methods.read,
    methods.reference,
    methods.checksum,
    methods.sha256_file,
)
_METHODS = SimpleNamespace(**dict(vars(methods), legacy=shipped.methods.legacy))
_BUILD, _REPLAY = shipped.build, shipped.replay
LITERATURE = "results/raw/experiment_8513_v732_strongest_baseline_audit/literature/mapping.json"
LITERATURE_PIN = "sha256:463937fb240b466126921d7b5d11a601ba1974a386e4101698868b74d94511db"


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Actual counts let the supervisor detect stalled readers without model activity."""
    print(f"[exp8513] phase={phase} completed={completed} pending={pending}", flush=True)


def authority(root: Path, raw: Path, *, gate_root: Path | None = None) -> Json:
    """Exact observed gate fields distinguish missing methods from a measured zero."""
    result = dict(methods.authority(root, raw))
    gate_root = root if gate_root is None else gate_root
    path = gate_root / "results" / (methods.NAME + ".json")
    value: Json = {}
    try:
        value = read(dict(path=str(path), sha256=METHODS_PIN))
        terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
        bound = read_bound_sidecar(path, Path(terminal["publication"]["sidecar_path"]))
        if not bound["report"]["passed"] or value["flagged_adversarial"]:
            raise ValueError("methods_publication")
    except (OSError, ValueError, KeyError) as error:
        result["gate_check_summary"].append(
            shipped.gate(
                path,
                "methods_primary_authenticated",
                METHODS_PIN,
                None if isinstance(error, FileNotFoundError) else str(error),
                upstream=methods.TASK,
                digest=METHODS_PIN,
            )
        )
    for field in ["methods_ready_score", "source_inputs_ready_score", "required_checks_passed"]:
        consumer = reader_receipt(methods.TASK, gate_root / "results", field=field)
        result["gate_check_summary"].append(
            dict(
                shipped.gate(
                    path, field, 1, value.get(field), upstream=methods.TASK, digest=METHODS_PIN
                ),
                consumer_passed=consumer["passed"],
            )
        )
    result["required_gate_observations"] = deepcopy(result["gate_check_summary"])
    result["gate_check_summary"] = [
        r
        for r in result["gate_check_summary"]
        if not r["passed"] or r.get("consumer_passed") is False
    ]
    result["activated"] = not result["gate_check_summary"]
    return result


def comparator_manifest(root: Path) -> Json:
    """Reconstruct original roles, vectors and donors before registering future fitting slots."""
    _, panel, _, features = operands(root)
    controls = read(dict(path=str(root / fit.UPSTREAM), sha256=fit.UPSTREAM_PIN))
    _, view = fit.operands(root)
    frozen = fit.controls.freeze(panel, view["original_heads"], fit.methods.protocol(), [])
    for key, original in [
        ("question_roles", "question_roles"),
        ("donor_mappings", "donor_mappings"),
        ("class_support", "class_support"),
        ("operator_manifest", "operators"),
    ]:
        if controls[key] != frozen[original]:
            raise ValueError("original_" + key + "_binding")
    if features != dict(rows=frozen["rows"], operators=frozen["operators"]):
        raise ValueError("raw_feature_donor_binding")
    protocol = methods.protocol()
    basis = protocol["identity_brier"]["basis_reference"]
    if sha256_file(Path(basis["path"])) != basis["sha256"]:
        raise ValueError("original_basis_binding")
    rows = {a: [r for r in features["rows"] if r["arm"] == a] for a in n.ARMS}
    inputs = dict(zip(COMPARATORS, ["raw_p2", n.FULL, n.SCALAR, n.FULL, n.PERMUTED], strict=True))
    slots = []
    for arm, source in inputs.items():
        for row in rows[source]:
            for seed in n.SEEDS:
                slots.append(
                    dict(
                        answer_id=row["answer_id"],
                        question_hash=row["question_hash"],
                        role=row["role"],
                        arm=arm,
                        seed=seed,
                        input_vector_hash=row["predictor_vector_hash"],
                        donor_answer_id=row["donor_answer_id"],
                        raw_p2=row["q"],
                        status="future_dependency"
                        if arm.startswith("identity_")
                        else "authenticated_cached_input",
                        missing_reason="Exp8514_not_run" if arm.startswith("identity_") else None,
                    )
                )
    return dict(
        schema="carnot.v732.strongest_comparators.v1",
        arms=list(COMPARATORS),
        answers=900,
        questions=300,
        question_roles=deepcopy(controls["question_roles"]),
        class_support=deepcopy(controls["class_support"]),
        donor_mappings=deepcopy(controls["donor_mappings"]),
        operators=features["operators"],
        basis_reference=basis,
        scalar_basis_reference=protocol["scalar_basis_reference"],
        identity_offset_already_present=True,
        changed_mechanism="Brier fitting at T=1 plus tune-only shrinkage including alpha=0; raw-logit offset inherited",
        identity_brier=protocol["identity_brier"],
        optimization_diagnostics=protocol["optimization_diagnostics"],
        future_producer="exp8514-identity-brier-fit",
        arm_protocol_names=dict(zip(COMPARATORS, protocol["identity_brier"]["arms"], strict=True)),
        slots=slots,
        original_feature_reference=controls["feature_manifest_reference"],
        original_label_reference=panel["evaluator_custody"],
        original_trainer_reference=controls["trainer_view"],
        original_authorities=fit.methods.protocol()["historical_authorities"],
        test_label_access="already-exposed descriptive analysis; no fitting",
    )


@contextmanager
def binding() -> Iterator[None]:
    """Narrow scoped adapters preserve shipped thresholds, resource checks and seal order."""
    numeric = SimpleNamespace(**dict(vars(n), evaluate=evaluate))
    with (
        patch.object(_METHODS, "authority", authority),
        patch.multiple(
            shipped,
            NAME=NAME,
            TASK=TASK,
            CLI=CLI,
            TEST=TEST,
            OWNED=OWNED,
            UPSTREAM=UPSTREAM,
            UPSTREAM_PIN=UPSTREAM_PIN,
            methods=_METHODS,
            operands=operands,
            predict_rows=history.predict_rows,
            n=numeric,
            progress=progress,
        ),
    ):
        yield


@contextmanager
def measurement_deadline() -> Iterator[None]:
    """Reserve validation time without extending the invocation's enclosing hard deadline."""
    previous = signal.setitimer(signal.ITIMER_REAL, 3600)
    start = time.monotonic()
    try:
        yield
    finally:
        remaining = max(0.001, previous[0] - (time.monotonic() - start)) if previous[0] else 0
        signal.setitimer(signal.ITIMER_REAL, remaining, previous[1])


@measurement_deadline()
def measure(root: Path, raw: Path) -> Json:
    """CPU prerequisites and exact task authority precede all descriptive measurements."""
    with binding():
        work = dict(shipped.measure(root, raw))
    for failure in work["failures"]:
        if failure.get("field") == "head_ready_score_and_authenticated_operands":
            failure.update(
                upstream=fit.TASK,
                check="head_controls_ready_score_and_authenticated_operands",
                field="head_controls_ready_score_and_authenticated_operands",
            )
    work["preconditions"].update(measurement_cap_s=3600, optional_cuda_and_boards_gate_cpu=False)
    work["literature_mapping"] = []
    work["comparator_manifest_reference"] = None
    if work["result"]:
        progress("freeze_comparators_begin", 0, 22500)
        try:
            manifest = comparator_manifest(root)
            atomic_json(raw / "comparator_manifest.json", manifest)
            work["comparator_manifest_reference"] = reference(raw / "comparator_manifest.json")
            work["primitive_files"].append(work["comparator_manifest_reference"])
            literature = read(dict(path=str(ROOT / LITERATURE), sha256=LITERATURE_PIN))
            for entry in literature["mapping"]:
                for ref in entry["references"]:
                    if sha256_file(Path(ref["path"])) != ref["sha256"]:
                        raise ValueError("literature_byte_drift")
                    work["source_refs"].append(ref)
            work["literature_mapping"] = literature["mapping"]
            work["source_refs"] += [
                reference(ROOT / LITERATURE),
                dict(path=str(ROOT / AUDIT), sha256=AUDIT_PIN),
                dict(path=str(ROOT / "results" / (methods.NAME + ".json")), sha256=METHODS_PIN),
                dict(path=str(ROOT / methods.PROTOCOL), sha256=methods.PROTOCOL_PIN),
                *manifest["original_authorities"]["V726"],
                manifest["original_feature_reference"],
                manifest["original_label_reference"],
                manifest["original_trainer_reference"],
            ]
        except (OSError, ValueError, KeyError) as error:
            work["owned_failure"] = True
            work["failures"].append(
                shipped.gate(raw, "comparator_and_literature_custody", True, str(error))
            )
        progress("freeze_comparators_end", 22500 if not work["owned_failure"] else 0, 0)
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Input qualification never claims scientific improvement on exposed labels."""
    with binding():
        value = dict(_BUILD(work, receipts, raw, output))
    result = work["result"] or {}
    value["gate_check_summary"] = [
        r
        for r in work["authority"].get("required_gate_observations", [])
        if r["passed"] and r.get("consumer_passed", True)
    ] + value["gate_check_summary"]
    value.update(
        experiment_id=8513,
        task_id=TASK,
        milestone="2026.10.732",
        run_date="20261014",
        honest_verdict=f"complete_{value['verdict_class']}_v732_strongest_baseline_audit",
        comparator_inputs_ready_score=value["decision_audit_ready_score"],
        comparator_manifest_reference=work["comparator_manifest_reference"],
        raw_vs_full_question_rows=result.get("raw_vs_full_question_rows", []),
        historical_arm_metrics=dict(A1=result.get("A1_metrics"), A2=result.get("A2_sensitivity")),
        identity_offset_already_present=True,
        literature_mapping=work["literature_mapping"],
        external_new_label_condition="Independently adjudicated disjoint questions, frozen generators/heads before access, and resolved exposure; public releases alone cannot certify who accessed questions or labels. Never re-split the exposed panel or import RAIM judges as truth.",
        future_dependencies=[
            dict(experiment_id=n, status="future_dependency_not_observation")
            for n in [8514, 8515, 8521]
        ],
        methodology_note="Reproduce all five old A1/A2 arm losses and fixed typed decisions from original sealed V730 rows; report full-minus-raw question differences descriptively without new thresholds. Freeze strongest controls and fit/tune optimization prefixes for future Brier fitting. No current optimizer steps, LLM calls, paper replication or sample-efficiency claim.",
    )
    value["acceptance_gates"].update(
        original_custody_and_basis=bool(work["comparator_manifest_reference"])
        and not work["owned_failure"],
        historical_loss_reproduction=bool(result),
        literature_bytes_inspected=bool(work["literature_mapping"]),
    )
    value["historical_model_provenance"][0]["original_producer_hashes"] = work["source_refs"]
    value["field_principles"].update(
        {
            k: "Bind original exposed custody and fixed methods; qualification is separate from fresh scientific benefit."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["field_principles"].update(
        comparator_inputs_ready_score="One only after exact raw/features/labels, original basis, current authority and owned validation qualify.",
        raw_vs_full_question_rows="Full-minus-raw losses average siblings and seeds within question; descriptive only, without threshold selection.",
        comparator_manifest_reference="Every original answer/arm/seed retains a slot; future trained predictions remain explicitly unmeasured.",
        literature_mapping="Hashed inspected primary sources with exact adoption and defer decisions; no theorem transfer.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Independent original operands defeat self-consistently rehashed future manifests."""
    try:
        value = json.loads(path.read_bytes())
        work = read(value["primitive_reference"])
        root = Path(work["upstream_reference"]["path"]).parents[1]
        if work["result"]:
            if read(work["comparator_manifest_reference"]) != comparator_manifest(root):
                return False
            literature = read(dict(path=str(ROOT / LITERATURE), sha256=LITERATURE_PIN))
            if work["literature_mapping"] != literature["mapping"]:
                return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    with (
        binding(),
        patch.object(
            _METHODS,
            "authority",
            lambda location, scratch: authority(location, scratch, gate_root=root),
        ),
        patch.object(shipped, "build", build),
    ):
        return bool(_REPLAY(path))


def operands(root: Path) -> tuple[Json, Json, Json, Json]:
    """Original producer pins authenticate predictions, labels and inherited basis bindings."""
    upstream, panel, frozen, features = history.operands(root)
    audit = read(dict(path=str(root / AUDIT), sha256=AUDIT_PIN))
    original = read(audit["prediction_reference"])["rows"]
    predicted = history.predict_rows(frozen, features["rows"], panel["stratum_roster"])
    if predicted != original or audit["run_date"] != "20261013":
        raise ValueError("original_prediction_custody")
    controls = read(dict(path=str(root / fit.UPSTREAM), sha256=fit.UPSTREAM_PIN))
    sealed_work = read(audit["primitive_reference"])
    original_authority = [
        r for r in sealed_work["authority"]["authority_snapshots"].values() if r.get("exists")
    ]
    for ref in [
        *controls["source_artifact_hashes"],
        *panel["source_artifact_hashes"],
        *original_authority,
        *audit["code_config_hashes"],
    ]:
        if (
            "path" in ref
            and sha256_file(Path(ref.get("snapshot_path", ref["path"]))) != ref["sha256"]
        ):
            raise ValueError("original_source_or_method_custody")
    return upstream, panel, frozen, features


def question_differences(rows: list[Json]) -> list[Json]:
    """Average siblings and deterministic repeats before describing each question difference."""
    groups: dict[str, list[Json]] = defaultdict(list)
    for prediction in rows:
        groups[prediction["question_hash"]].append(prediction)
    result = []
    for question, group in sorted(groups.items()):
        row: Json = dict(question_hash=question, descriptive_only=True)
        for target, label in [("A1", "y"), ("A2", "y_sensitivity")]:
            arms = {
                a: [r for r in group if r["arm"] == a and r.get(label) in (0, 1)]
                for a in ["raw_p2", n.FULL]
            }
            keys = [{(r["unit_id"], r["seed"]) for r in rs} for rs in arms.values()]
            complete = (
                bool(keys[0])
                and keys[0] == keys[1]
                and all(r["probability"] is not None for rs in arms.values() for r in rs)
            )
            metrics = (
                {a: n.shipped.summary(rs, label) for a, rs in arms.items()} if complete else {}
            )
            row[target] = dict(
                full_minus_raw_brier=metrics[n.FULL]["brier"] - metrics["raw_p2"]["brier"]
                if complete
                else None,
                full_minus_raw_typed_loss=metrics[n.FULL]["decision_loss"]
                - metrics["raw_p2"]["decision_loss"]
                if complete
                else None,
                sibling_seed_count=len(keys[0]),
                missing_reason=None
                if complete
                else "missing_paired_probability_or_observed_target",
            )
        result.append(row)
    return result


def evaluate(predictions: list[Json], targets: list[Json]) -> Json:
    """All five historical losses must reproduce before descriptive headroom is reported."""
    result = dict(n.evaluate(predictions, targets))
    original = read(dict(path=str(ROOT / AUDIT), sha256=AUDIT_PIN))
    if (
        result["A1_metrics"] != original["A1_metrics"]
        or result["A2_sensitivity"] != original["A2_sensitivity"]
    ):
        raise ValueError("historical_loss_reproduction")
    result.update(
        raw_vs_full_question_rows=question_differences(result["rows"]), exploratory_benefit_score=0
    )
    return result
