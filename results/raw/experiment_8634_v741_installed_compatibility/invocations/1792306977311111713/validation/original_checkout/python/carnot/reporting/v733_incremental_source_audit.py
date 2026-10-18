"""REQ-REPORT-8529: reduce exact sealed predictions without fitting or model calls.

Current authority authenticates this task. Original producer hashes retain the
historical roles, answers and exposed targets without changing their dates.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import patch

from carnot.reporting import v726_criterion_decision_audit as shipped
from carnot.reporting import v733_contract_methods as methods
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar, reader_receipt
from carnot.reporting.v733_source_diagnostics import diagnostic_rows
from carnot.verify import v733_identity_brier_fit as fit
from carnot.verify import v733_incremental_source_metrics as n

Json = dict[str, Any]
ROOT = methods.ROOT
NAME, TASK = "experiment_8529_v733_incremental_source_audit", "exp8529-incremental-source-audit"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v733_incremental_source_8529.py"
OWNED = [
    "python/carnot/verify/v733_incremental_source_metrics.py",
    "python/carnot/reporting/v733_incremental_source_audit.py",
    "python/carnot/reporting/v733_incremental_source_runner.py",
    CLI,
]
UPSTREAM = f"results/{fit.NAME}.json"
UPSTREAM_PIN = "sha256:a93b5a605a87f5b125898ffe19a13a59538abf405ec3f6abc67a3e44ae4e28ef"
REGISTRATION = dict(
    path=str(ROOT / "openspec/change-proposals/v732-methods-protocol.json"),
    sha256="sha256:66b761adbfee7c7671d66a9b93ec78afda3508097d95a206ac7d83f1faba2c94",
)
MODEL_SPECS: list[Json] = []
read, reference, checksum = methods.read, methods.reference, methods.checksum
_METHODS = SimpleNamespace(**dict(vars(methods), legacy=shipped.methods.legacy))
_BUILD = shipped.build


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase counts so the CPU audit remains observable without model activity."""
    print(f"[exp8529] phase={phase} completed={completed} pending={pending}", flush=True)


def operands(root: Path) -> tuple[Json, Json, Json, Json]:
    """Independent producer pins bind current reductions to the exact sealed predictions."""
    upstream = read(dict(path=str(root / UPSTREAM), sha256=UPSTREAM_PIN))
    terminal = json.loads(Path(upstream["terminal_validation_sidecar_path"]).read_bytes())
    bound = read_bound_sidecar(root / UPSTREAM, Path(terminal["publication"]["sidecar_path"]))
    consumer = reader_receipt(fit.TASK, root / "results", field="identity_heads_ready_score")
    if not (
        upstream["required_checks_passed"]
        and not upstream["flagged_adversarial"]
        and upstream["identity_heads_ready_score"] == 1
        and bound["report"]["passed"]
        and consumer["passed"]
    ):
        raise ValueError("identity_heads_unqualified")
    for ref in upstream["code_config_hashes"]:
        if "path" in ref and sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("original_predictor_method_drift")
    original = read(REGISTRATION)["primary_contrasts"]
    if methods.protocol()["primary_contrasts"] != original:
        raise ValueError("primary_contrasts_registration_drift")
    controls, view = fit.operands(root)
    manifest = view["manifest"]
    panel = read(dict(path=str(ROOT / shipped.fit.UPSTREAM), sha256=shipped.fit.UPSTREAM_PIN))
    panel["evaluator_custody"] = manifest["original_label_reference"]
    cached = read(upstream["predictions_reference"])
    frozen = read(upstream["frozen_heads_reference"])
    if (
        cached["rows"] != upstream["rows"]
        or cached["test_label_reads"] != 0
        or upstream["seal_clock"]["evaluator_opened"]
        or upstream["seal_clock"]["heads_sealed_ns"]
        >= upstream["seal_clock"]["predictions_sealed_ns"]
    ):
        raise ValueError("prediction_seal_drift")
    frozen["cached_predictions"] = cached["rows"]
    return (
        dict(
            upstream,
            additional_refs=[
                upstream["predictions_reference"],
                upstream["frozen_heads_reference"],
                controls["comparator_manifest_reference"],
                manifest["original_label_reference"],
                REGISTRATION,
            ],
        ),
        panel,
        frozen,
        view["panel"],
    )


def predict_rows(frozen: Json, features: list[Json], strata: list[Json]) -> list[Json]:
    """Normalize exact upstream rows; missing heads retain the original intended roster."""
    metadata = {r["unit_id"]: r for r in strata}
    cached = frozen.get("cached_predictions")
    if cached is None:
        return [
            dict(
                unit_id=r["unit_id"],
                question_hash=r["question_hash"],
                dataset=metadata[r["unit_id"]]["dataset"],
                generator=metadata[r["unit_id"]]["generator"],
                arm=a,
                seed=s,
                probability=None,
                missing_reason="missing_required_identity_predictions",
            )
            for r in features
            for a in n.ARMS
            for s in n.SEEDS
        ]
    return [
        dict(
            unit_id=r["answer_id"],
            question_hash=r["question_hash"],
            dataset=metadata[r["answer_id"]]["dataset"],
            generator=metadata[r["answer_id"]]["generator"],
            arm=r["arm"],
            seed=r["seed"],
            probability=r["absolute_metric"],
            missing_reason=r["missing_reason"],
        )
        for r in cached
    ]


@contextmanager
def binding() -> Iterator[None]:
    """Scoped bindings preserve the shipped measurement, custody and rejection paths."""
    with patch.multiple(
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
        predict_rows=predict_rows,
        n=n,
        progress=progress,
    ):
        yield


def measure(root: Path, raw: Path) -> Json:
    """Authenticate CPU resources and current authority before opening the sealed targets."""
    with binding():
        work = dict(shipped.measure(root, raw))
    work["preconditions"].update(measurement_cap_s=3600, optional_cuda_and_boards_gate_cpu=False)
    work["primary_contrasts"] = read(REGISTRATION)["primary_contrasts"]
    work["code_refs"].extend(
        reference(ROOT / p)
        for p in [
            "python/carnot/verify/v730_information_metrics.py",
            "python/carnot/verify/v726_decision_metrics.py",
        ]
    )
    work["upstream_diagnostics"] = {}
    if work["result"]:
        upstream, _, _, _ = operands(root)
        work["source_refs"].extend(upstream["additional_refs"])
        work["upstream_diagnostics"] = {
            k: upstream[k] for k in ["alpha_choices", "clip_counts", "seal_clock"]
        }
    work["failures"] = diagnostic_rows(work["failures"])
    for failure in work["failures"]:
        if failure["field"] == "head_ready_score_and_authenticated_operands":
            failure.update(
                check="qualified_identity_predictions",
                upstream=fit.TASK,
                field="identity_heads_ready_score_and_original_prediction_seals",
            )
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Qualify a measured null independently from the registered incremental benefit."""
    with binding():
        value = dict(_BUILD(work, receipts, raw, output))
    result = work["result"] or {}
    ready = value["decision_audit_ready_score"]
    benefit = value["exploratory_benefit_score"]
    diagnostics = work["upstream_diagnostics"]
    value.update(
        experiment_id=8529,
        task_id=TASK,
        milestone="2026.10.733",
        run_date="20261015",
        honest_verdict=f"complete_{value['verdict_class']}_v733_incremental_source_audit",
        source_audit_ready_score=ready,
        incremental_source_benefit_score=benefit,
        primary_contrasts=work["primary_contrasts"],
        primary_contrast_rows=[
            dict(
                comparator=a,
                treatment=n.FULL,
                brier_gain=result.get("contrasts", {}).get(a),
                interval=result.get("familywise_intervals", {}).get(a),
                typed_cost_delta=result.get("decision_loss_deltas", {}).get(a),
            )
            for a in n.CONTROLS
        ],
        typed_cost_deltas=result.get("decision_loss_deltas"),
        gate_check_summary=diagnostic_rows(value["gate_check_summary"]),
        alpha0_degeneracy={
            a: dict(
                alpha_by_seed=choices,
                alpha_zero_count=sum(v == 0 for v in choices.values()),
                exact_identity_when_alpha_zero=True,
            )
            for a, choices in diagnostics.get("alpha_choices", {}).items()
        },
        clipping=dict(
            probability_clipping_applied=False,
            internal_logit_epsilon=0.0001,
            upstream_clip_counts=diagnostics.get("clip_counts"),
        ),
        unchanged_source_tuning_closed=bool(ready and not benefit),
        next_evidence_condition="A registered changed information source or independently adjudicated disjoint questions; never re-split this exposed release.",
        familywise_confidence_level=0.95,
        per_contrast_confidence_level=n.CONFIDENCE,
        methodology_note="Exact Exp8528 predictions; all five answer losses reconstructed with fixed typed cost1/0/0.2. Average siblings and seeds within100 questions before10000 paired draws at seed7318502. Three registered contrasts use98.3333333333 percent two-sided Bonferroni intervals and95 percent familywise confidence. Legacy full-source and A2 are descriptive. No model loads, threshold search or independent generalization.",
    )
    for key in [
        "paired_question_rows",
        "contrasts",
        "familywise_intervals",
        "repeat_fit_audit",
        "matched_coverage_risk",
        "class_support",
    ]:
        value[key] = result.get(key)
    value["acceptance_gates"].update(
        all_registered_brier_lower_bounds_positive=bool(result)
        and all(v and v[0] > 0 for v in result["familywise_intervals"].values()),
        raw_and_scalar_typed_cost_nonincrease=bool(result)
        and all(
            result["decision_loss_deltas"][a] is not None and result["decision_loss_deltas"][a] <= 0
            for a in [n.RAW, n.SCALAR]
        ),
        no_identical_strong_control=bool(result)
        and not any(
            d["identical_predictions"] for d in result["negative_control_degeneracy"].values()
        ),
    )
    value["historical_model_provenance"][0]["original_producer_hashes"] = work["source_refs"]
    value["field_principles"].update(
        {
            k: "Exact sealed exposed evidence and original registration; qualification and benefit are separate."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["field_principles"].update(
        source_audit_ready_score="Complete qualified audit, including measured nulls.",
        incremental_source_benefit_score="Every registered adjusted Brier lower bound exceeds zero, typed cost does not increase versus raw/scalar, and no strong control is identical.",
        primary_contrasts="Exact original V732 numerical operands, unchanged in V733.",
        brier_gain_vs_logistic="Compatibility field for the current brier_scalar minus brier_full contrast; no current logistic comparison.",
        brier_gain_ci95="Compatibility field for the registered98.3333333333 percent brier_scalar contrast interval; the field name does not set its confidence level.",
        typed_cost_deltas="Full minus comparator cost after sibling/seed averaging within question; nonpositive is noninferior.",
        next_evidence_condition="Failed registered contrasts close unchanged source tuning; exposed release cannot be re-split.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Rebuild losses from independent upstream seals, not repaired local envelope hashes."""
    try:
        value = json.loads(path.read_bytes())
        work = read(value["primitive_reference"])
        if work["primary_contrasts"] != read(REGISTRATION)["primary_contrasts"]:
            return False
        if work["result"]:
            upstream, _, _, _ = operands(Path(work["upstream_reference"]["path"]).parents[1])
            if work["upstream_diagnostics"] != {
                k: upstream[k] for k in ["alpha_choices", "clip_counts", "seal_clock"]
            }:
                return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    with binding(), patch.object(shipped, "build", build):
        return bool(shipped.replay(path))
